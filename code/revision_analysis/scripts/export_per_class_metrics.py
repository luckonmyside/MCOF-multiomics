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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from common import (
    DATASET_CONFIG,
    MODEL_ORDER,
    ensure_dir,
    infer_probability_columns,
    mean_sd,
    parse_numeric_labels,
    read_fixed_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct MCOF held-out predictions from checkpoints and summarize "
            "per-class metrics/confusion matrices for all seven models."
        )
    )
    parser.add_argument("--toolkit_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--mcof_main_root", required=True)
    parser.add_argument("--ml_root", required=True)
    parser.add_argument("--diablo_root", required=True)
    parser.add_argument("--mogonet_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["BRCA", "STAD"])
    return parser.parse_args()


def add_src(toolkit_root: Path) -> None:
    src = str((toolkit_root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def predict_mcof(
    dataset: str,
    repeat: int,
    data_root: Path,
    split_root: Path,
    mcof_main_root: Path,
    output_root: Path,
    batch_size: int,
    device: torch.device,
) -> Path:
    from train_utils_v2 import (  # type: ignore
        OmicsPreprocessor,
        load_checkpoint,
        read_multiomics_dataset,
        restore_model_from_checkpoint,
    )

    data_dir = data_root / dataset
    omics_list, labels, _ = read_multiomics_dataset(data_dir)
    split_table = read_fixed_splits(
        split_root / f"{dataset}_outer_splits.csv", n_samples=len(labels), repeats=5
    )
    test_indices = (
        split_table.loc[
            split_table["repeat"].eq(repeat) & split_table["subset"].eq("test"),
            "sample_index",
        ]
        .astype(int)
        .to_numpy()
    )

    checkpoint_path = (
        mcof_main_root / dataset / "mcof_se" / f"repeat_{repeat}" / "best_model.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = restore_model_from_checkpoint(checkpoint, device=device)
    preprocessor = OmicsPreprocessor.from_state_dict(checkpoint["preprocessor_state"])
    transformed = preprocessor.transform(omics_list)
    x_all = np.concatenate(transformed, axis=1).astype(np.float32, copy=False)
    x_test = torch.from_numpy(x_all[test_indices]).to(device)

    probabilities: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_test), batch_size):
            logits = model(x_test[start : start + batch_size])
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    prob = np.concatenate(probabilities, axis=0)
    pred = np.argmax(prob, axis=1)

    frame = pd.DataFrame(
        {
            "sample_index": test_indices,
            "y_true": labels[test_indices].astype(int),
            "y_pred": pred.astype(int),
        }
    )
    for class_index in range(prob.shape[1]):
        frame[f"prob_class_{class_index}"] = prob[:, class_index]

    path = ensure_dir(output_root / "predictions" / dataset / "MCOF" / f"repeat_{repeat}") / "predictions_test.csv"
    frame.to_csv(path, index=False)

    # Confirm that the reconstructed metrics agree with the archived checkpoint.
    archived_metrics = checkpoint.get("metrics", {})
    calculated_acc = float((frame["y_true"] == frame["y_pred"]).mean())
    if "acc" in archived_metrics and abs(float(archived_metrics["acc"]) - calculated_acc) > 1e-6:
        raise RuntimeError(
            f"{dataset} repeat {repeat}: reconstructed ACC {calculated_acc} does not match checkpoint {archived_metrics['acc']}"
        )
    return path


def prediction_path(
    model: str,
    dataset: str,
    repeat: int,
    ml_root: Path,
    diablo_root: Path,
    mogonet_root: Path,
    reconstructed_root: Path,
) -> Path:
    if model == "MCOF":
        return reconstructed_root / "predictions" / dataset / "MCOF" / f"repeat_{repeat}" / "predictions_test.csv"
    if model in {"KNN", "SVM", "NB", "RF"}:
        return ml_root / dataset / model / f"repeat_{repeat}" / "predictions_test.csv"
    if model == "DIABLO":
        return diablo_root / dataset / "DIABLO" / f"repeat_{repeat}" / "predictions_test.csv"
    if model == "MOGONET":
        return mogonet_root / dataset / "MOGONET" / f"repeat_{repeat}" / "predictions_test.csv"
    raise ValueError(model)


def read_predictions(path: Path, dataset: str, expected_indices: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray | None]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"sample_index", "y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")

    frame = frame.copy()
    frame["sample_index"] = pd.to_numeric(frame["sample_index"], errors="raise").astype(int)
    if frame["sample_index"].duplicated().any():
        raise ValueError(f"{path} contains duplicated sample indices")
    if set(frame["sample_index"]) != set(expected_indices.tolist()):
        missing_indices = sorted(set(expected_indices.tolist()).difference(frame["sample_index"]))[:10]
        extra_indices = sorted(set(frame["sample_index"]).difference(expected_indices.tolist()))[:10]
        raise ValueError(
            f"{path}: test indices differ from fixed split; missing={missing_indices}, extra={extra_indices}"
        )
    frame = frame.set_index("sample_index").loc[expected_indices].reset_index()
    frame["y_true_encoded"] = parse_numeric_labels(frame["y_true"], dataset)
    frame["y_pred_encoded"] = parse_numeric_labels(frame["y_pred"], dataset)

    score_columns = infer_probability_columns(frame)
    scores = None
    if score_columns:
        scores = frame[score_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        expected_classes = len(DATASET_CONFIG[dataset]["class_names"])
        if scores.shape[1] != expected_classes:
            raise ValueError(
                f"{path}: found {scores.shape[1]} score columns; expected {expected_classes}"
            )
    return frame, scores


def per_class_rows(
    dataset: str,
    model: str,
    repeat: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None,
) -> Tuple[List[Dict[str, object]], np.ndarray]:
    class_names = DATASET_CONFIG[dataset]["class_names"]
    n_classes = len(class_names)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(n_classes),
        zero_division=0,
    )
    rows: List[Dict[str, object]] = []
    total = matrix.sum()
    for class_index, class_name in enumerate(class_names):
        tp = int(matrix[class_index, class_index])
        fn = int(matrix[class_index, :].sum() - tp)
        fp = int(matrix[:, class_index].sum() - tp)
        tn = int(total - tp - fn - fp)
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        auc = float("nan")
        if scores is not None:
            binary = (y_true == class_index).astype(int)
            if len(np.unique(binary)) == 2:
                auc = float(roc_auc_score(binary, scores[:, class_index]))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "repeat": repeat,
                "class_index": class_index,
                "class_name": class_name,
                "support": int(support[class_index]),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "sensitivity_recall": float(recall[class_index]),
                "specificity": float(specificity),
                "precision": float(precision[class_index]),
                "f1": float(f1[class_index]),
                "one_vs_rest_auc": auc,
            }
        )
    return rows, matrix


def plot_aggregate_confusions(
    dataset: str,
    matrices: Dict[str, np.ndarray],
    output_root: Path,
) -> None:
    class_names = DATASET_CONFIG[dataset]["class_names"]
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.5), constrained_layout=True)
    axes_flat = axes.ravel()
    image = None
    for panel, model in enumerate(MODEL_ORDER):
        ax = axes_flat[panel]
        matrix = matrices[model].astype(float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        image = ax.imshow(normalized, vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(model)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{normalized[i, j]:.2f}\n(n={int(matrix[i,j])})", ha="center", va="center", fontsize=7)
    axes_flat[-1].axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="Row-normalized proportion")
    fig.suptitle(
        f"{dataset}: confusion matrices summed over five repeated held-out test sets",
        fontsize=12,
    )
    fig.savefig(output_root / f"{dataset}_all_models_confusion_matrices.pdf", bbox_inches="tight")
    fig.savefig(output_root / f"{dataset}_all_models_confusion_matrices.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    toolkit_root = Path(args.toolkit_root)
    add_src(toolkit_root)
    data_root = Path(args.data_root)
    split_root = Path(args.split_root)
    mcof_main_root = Path(args.mcof_main_root)
    ml_root = Path(args.ml_root)
    diablo_root = Path(args.diablo_root)
    mogonet_root = Path(args.mogonet_root)
    output_root = ensure_dir(args.output_root)
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")

    rows: List[Dict[str, object]] = []
    confusion_rows: List[Dict[str, object]] = []
    integrity_rows: List[Dict[str, object]] = []

    for dataset in args.datasets:
        if dataset not in {"BRCA", "STAD"}:
            raise ValueError("Per-class revision analysis is currently defined for BRCA and STAD")
        labels = pd.read_csv(data_root / dataset / "labels_all.csv", header=None).iloc[:, 0]
        labels_encoded, _ = pd.factorize(labels, sort=True)
        split_table = read_fixed_splits(
            split_root / f"{dataset}_outer_splits.csv", n_samples=len(labels), repeats=5
        )

        for repeat in range(1, 6):
            predict_mcof(
                dataset,
                repeat,
                data_root,
                split_root,
                mcof_main_root,
                output_root,
                args.batch_size,
                device,
            )
            expected_indices = (
                split_table.loc[
                    split_table["repeat"].eq(repeat) & split_table["subset"].eq("test"),
                    "sample_index",
                ]
                .astype(int)
                .to_numpy()
            )
            expected_y = labels_encoded[expected_indices]
            for model in MODEL_ORDER:
                path = prediction_path(
                    model,
                    dataset,
                    repeat,
                    ml_root,
                    diablo_root,
                    mogonet_root,
                    output_root,
                )
                frame, scores = read_predictions(path, dataset, expected_indices)
                y_true = frame["y_true_encoded"].to_numpy(dtype=int)
                y_pred = frame["y_pred_encoded"].to_numpy(dtype=int)
                labels_match = bool(np.array_equal(y_true, expected_y))
                if not labels_match:
                    raise ValueError(f"{path}: y_true does not match labels_all.csv for the fixed split")
                current_rows, matrix = per_class_rows(dataset, model, repeat, y_true, y_pred, scores)
                rows.extend(current_rows)
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        confusion_rows.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "repeat": repeat,
                                "observed_class_index": i,
                                "observed_class_name": DATASET_CONFIG[dataset]["class_names"][i],
                                "predicted_class_index": j,
                                "predicted_class_name": DATASET_CONFIG[dataset]["class_names"][j],
                                "count": int(matrix[i, j]),
                            }
                        )
                integrity_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "repeat": repeat,
                        "prediction_file": str(path),
                        "n_test": len(frame),
                        "fixed_indices_match": True,
                        "labels_match": labels_match,
                        "accuracy_from_predictions": float((y_true == y_pred).mean()),
                    }
                )

    detailed = pd.DataFrame(rows)
    detailed.to_csv(output_root / "per_class_metrics_by_repeat.csv", index=False)
    confusions = pd.DataFrame(confusion_rows)
    confusions.to_csv(output_root / "confusion_matrices_by_repeat_long.csv", index=False)
    pd.DataFrame(integrity_rows).to_csv(output_root / "prediction_integrity_checks.csv", index=False)

    summary_rows = []
    for keys, group in detailed.groupby(["dataset", "model", "class_index", "class_name"], sort=False):
        dataset, model, class_index, class_name = keys
        row: Dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "class_index": class_index,
            "class_name": class_name,
            "n_repeats": len(group),
            "support_total_over_repeats": int(group["support"].sum()),
            "support_per_repeat": ";".join(map(str, group.sort_values("repeat")["support"].astype(int))),
        }
        for metric in ["sensitivity_recall", "specificity", "precision", "f1", "one_vs_rest_auc"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            if len(values):
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                row[f"{metric}_mean_sd"] = mean_sd(values)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_root / "per_class_metrics_summary.csv", index=False)

    for dataset in args.datasets:
        aggregate: Dict[str, np.ndarray] = {}
        n_classes = len(DATASET_CONFIG[dataset]["class_names"])
        for model in MODEL_ORDER:
            subset = confusions[
                confusions["dataset"].eq(dataset) & confusions["model"].eq(model)
            ]
            matrix = np.zeros((n_classes, n_classes), dtype=int)
            for _, row in subset.iterrows():
                matrix[int(row["observed_class_index"]), int(row["predicted_class_index"])] += int(row["count"])
            aggregate[model] = matrix
        plot_aggregate_confusions(dataset, aggregate, output_root)

    protocol = {
        "interpretation": (
            "Confusion matrices are summed over five repeated held-out test sets. "
            "Because test sets can overlap across repetitions, the summed counts are descriptive "
            "and are not treated as independent observations."
        ),
        "models": MODEL_ORDER,
        "datasets": args.datasets,
        "mcof_predictions": "reconstructed from the archived repeat-level checkpoints and fixed split files",
    }
    (output_root / "per_class_analysis_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(f"Saved per-class metrics and confusion matrices to {output_root}")


if __name__ == "__main__":
    main()
