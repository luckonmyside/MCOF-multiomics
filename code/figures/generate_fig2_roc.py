from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


DATASET_ORDER = ["BRCA", "STAD"]

CLASS_NAMES = {
    "BRCA": [
        "Basal-like",
        "HER2-enriched",
        "Luminal A",
        "Luminal B",
    ],
    "STAD": [
        "CIN",
        "GS",
        "MSI",
    ],
}

EXPECTED_COUNTS = {
    "BRCA": [180, 80, 547, 195],
    "STAD": [122, 48, 47],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate repeated-holdout mean ROC curves for formal MCOF "
            "BRCA and STAD checkpoints."
        )
    )
    parser.add_argument("--mcof_code_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grid_points", type=int, default=1001)
    return parser.parse_args()


def predict_probabilities(
    model: torch.nn.Module,
    x_test: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_test), batch_size):
            batch = x_test[start : start + batch_size]
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)
            outputs.append(probabilities.detach().cpu().numpy())

    return np.concatenate(outputs, axis=0)


def interpolate_roc(
    y_binary: np.ndarray,
    probabilities: np.ndarray,
    mean_fpr: np.ndarray,
) -> Tuple[np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_binary, probabilities)
    interpolated = np.interp(mean_fpr, fpr, tpr)
    interpolated[0] = 0.0
    interpolated[-1] = 1.0
    auc_value = roc_auc_score(y_binary, probabilities)
    return interpolated, float(auc_value)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run on a GPU node or change the script "
            "explicitly if CPU inference is intended."
        )

    device = torch.device("cuda")
    code_root = Path(args.mcof_code_root).resolve()
    sys.path.insert(0, str(code_root))

    from train_utils_v2 import (  # type: ignore
        OmicsPreprocessor,
        load_checkpoint,
        read_multiomics_dataset,
        restore_model_from_checkpoint,
    )

    data_root = Path(args.data_root)
    split_root = Path(args.split_root)
    checkpoint_root = Path(args.checkpoint_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mean_fpr = np.linspace(0.0, 1.0, args.grid_points)

    prediction_rows: List[Dict[str, object]] = []
    auc_rows: List[Dict[str, object]] = []
    source_curve_rows: List[Dict[str, object]] = []
    dataset_results: Dict[str, Dict[str, object]] = {}

    for dataset in DATASET_ORDER:
        omics_list, labels, _ = read_multiomics_dataset(
            data_root / dataset
        )
        labels = np.asarray(labels, dtype=np.int64)

        num_classes = len(CLASS_NAMES[dataset])
        observed_counts = np.bincount(
            labels,
            minlength=num_classes,
        ).tolist()

        if observed_counts != EXPECTED_COUNTS[dataset]:
            raise ValueError(
                f"{dataset}: observed label counts {observed_counts} do not "
                f"match expected class order/counts {EXPECTED_COUNTS[dataset]}. "
                "Do not use class names until label mapping is verified."
            )

        split_table = pd.read_csv(
            split_root / f"{dataset}_outer_splits.csv"
        )

        class_tprs: Dict[int, List[np.ndarray]] = {
            class_index: []
            for class_index in range(num_classes)
        }
        class_aucs: Dict[int, List[float]] = {
            class_index: []
            for class_index in range(num_classes)
        }
        macro_tprs: List[np.ndarray] = []
        weighted_tprs: List[np.ndarray] = []
        macro_aucs: List[float] = []
        weighted_aucs: List[float] = []

        for repeat in range(1, 6):
            checkpoint_path = (
                checkpoint_root
                / dataset
                / "mcof_se"
                / f"repeat_{repeat}"
                / "best_model.pt"
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)

            checkpoint = load_checkpoint(
                checkpoint_path,
                map_location=device,
            )
            model = restore_model_from_checkpoint(
                checkpoint,
                device=device,
            )
            preprocessor = OmicsPreprocessor.from_state_dict(
                checkpoint["preprocessor_state"]
            )

            transformed = preprocessor.transform(omics_list)
            x_all = np.concatenate(
                transformed,
                axis=1,
            ).astype(np.float32, copy=False)

            current = split_table[
                split_table["repeat"].eq(repeat)
            ]
            test_idx = (
                current.loc[
                    current["subset"].eq("test"),
                    "sample_index",
                ]
                .astype(int)
                .to_numpy()
            )

            y_test = labels[test_idx]
            x_test = torch.from_numpy(
                x_all[test_idx]
            ).to(device)

            probabilities = predict_probabilities(
                model,
                x_test,
                args.batch_size,
            )

            if probabilities.shape != (
                len(test_idx),
                num_classes,
            ):
                raise ValueError(
                    f"{dataset} repeat {repeat}: probability shape "
                    f"{probabilities.shape} is unexpected."
                )

            y_binary = label_binarize(
                y_test,
                classes=np.arange(num_classes),
            )

            repeat_class_tprs: List[np.ndarray] = []
            supports = np.bincount(
                y_test,
                minlength=num_classes,
            ).astype(float)
            weights = supports / supports.sum()

            for class_index in range(num_classes):
                interpolated, auc_value = interpolate_roc(
                    y_binary[:, class_index],
                    probabilities[:, class_index],
                    mean_fpr,
                )
                class_tprs[class_index].append(interpolated)
                class_aucs[class_index].append(auc_value)
                repeat_class_tprs.append(interpolated)

                auc_rows.append(
                    {
                        "dataset": dataset,
                        "repeat": repeat,
                        "curve": CLASS_NAMES[dataset][class_index],
                        "auc": auc_value,
                    }
                )

            repeat_class_matrix = np.vstack(repeat_class_tprs)
            macro_tpr = repeat_class_matrix.mean(axis=0)
            weighted_tpr = np.average(
                repeat_class_matrix,
                axis=0,
                weights=weights,
            )

            macro_auc = float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )
            )
            weighted_auc = float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                )
            )

            macro_tprs.append(macro_tpr)
            weighted_tprs.append(weighted_tpr)
            macro_aucs.append(macro_auc)
            weighted_aucs.append(weighted_auc)

            auc_rows.extend(
                [
                    {
                        "dataset": dataset,
                        "repeat": repeat,
                        "curve": "Macro-average",
                        "auc": macro_auc,
                    },
                    {
                        "dataset": dataset,
                        "repeat": repeat,
                        "curve": "Weighted-average",
                        "auc": weighted_auc,
                    },
                ]
            )

            for row_index, sample_index in enumerate(test_idx):
                row: Dict[str, object] = {
                    "dataset": dataset,
                    "repeat": repeat,
                    "sample_index": int(sample_index),
                    "y_true": int(y_test[row_index]),
                }
                for class_index in range(num_classes):
                    row[f"prob_class_{class_index}"] = float(
                        probabilities[row_index, class_index]
                    )
                prediction_rows.append(row)

            del model
            del checkpoint
            del preprocessor
            del x_test
            torch.cuda.empty_cache()

        curves: List[Dict[str, object]] = []

        for class_index in range(num_classes):
            tpr_matrix = np.vstack(class_tprs[class_index])
            mean_tpr = tpr_matrix.mean(axis=0)
            sd_tpr = tpr_matrix.std(axis=0, ddof=1)
            auc_values = np.asarray(
                class_aucs[class_index],
                dtype=float,
            )
            curves.append(
                {
                    "label": CLASS_NAMES[dataset][class_index],
                    "mean_tpr": mean_tpr,
                    "sd_tpr": sd_tpr,
                    "auc_mean": float(auc_values.mean()),
                    "auc_sd": float(auc_values.std(ddof=1)),
                    "linestyle": "-",
                    "linewidth": 1.8,
                }
            )

        for label, tprs, aucs, linestyle, linewidth in [
            (
                "Macro-average",
                macro_tprs,
                macro_aucs,
                "--",
                2.6,
            ),
            (
                "Weighted-average",
                weighted_tprs,
                weighted_aucs,
                ":",
                2.6,
            ),
        ]:
            tpr_matrix = np.vstack(tprs)
            auc_values = np.asarray(aucs, dtype=float)
            curves.append(
                {
                    "label": label,
                    "mean_tpr": tpr_matrix.mean(axis=0),
                    "sd_tpr": tpr_matrix.std(axis=0, ddof=1),
                    "auc_mean": float(auc_values.mean()),
                    "auc_sd": float(auc_values.std(ddof=1)),
                    "linestyle": linestyle,
                    "linewidth": linewidth,
                }
            )

        dataset_results[dataset] = {
            "curves": curves,
        }

        for curve in curves:
            for index, fpr_value in enumerate(mean_fpr):
                source_curve_rows.append(
                    {
                        "dataset": dataset,
                        "curve": curve["label"],
                        "fpr": float(fpr_value),
                        "mean_tpr": float(
                            curve["mean_tpr"][index]
                        ),
                        "sd_tpr": float(
                            curve["sd_tpr"][index]
                        ),
                        "auc_mean": float(curve["auc_mean"]),
                        "auc_sd": float(curve["auc_sd"]),
                    }
                )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.4),
        sharex=True,
        sharey=True,
    )

    for panel_index, dataset in enumerate(DATASET_ORDER):
        ax = axes[panel_index]
        curves = dataset_results[dataset]["curves"]

        for curve in curves:
            ax.plot(
                mean_fpr,
                curve["mean_tpr"],
                linestyle=curve["linestyle"],
                linewidth=curve["linewidth"],
                label=(
                    f"{curve['label']} "
                    f"(AUC={curve['auc_mean']:.3f}±"
                    f"{curve['auc_sd']:.3f})"
                ),
            )

        ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            linewidth=1.0,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_title(dataset)
        ax.grid(
            linestyle=":",
            linewidth=0.6,
            alpha=0.6,
        )
        ax.legend(
            loc="lower right",
            frameon=False,
            fontsize=8.3,
        )
        ax.text(
            -0.12,
            1.05,
            f"({chr(97 + panel_index)})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    axes[0].set_ylabel("True positive rate")

    fig.tight_layout()

    pdf_path = output_root / "Fig2_MCOF_ROC_BRCA_STAD.pdf"
    png_path = output_root / "Fig2_MCOF_ROC_BRCA_STAD.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    pd.DataFrame(prediction_rows).to_csv(
        output_root / "Fig2_MCOF_predictions_all_repeats.csv",
        index=False,
    )
    pd.DataFrame(auc_rows).to_csv(
        output_root / "Fig2_MCOF_AUC_by_repeat.csv",
        index=False,
    )
    pd.DataFrame(source_curve_rows).to_csv(
        output_root / "Fig2_MCOF_ROC_source_data.csv",
        index=False,
    )

    with (
        output_root / "Fig2_class_mapping.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "class_names": CLASS_NAMES,
                "expected_counts": EXPECTED_COUNTS,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("Generated:")
    print(pdf_path)
    print(png_path)
    print(output_root / "Fig2_MCOF_AUC_by_repeat.csv")
    print(output_root / "Fig2_MCOF_ROC_source_data.csv")


if __name__ == "__main__":
    main()
