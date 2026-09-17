from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from train_utils_v2 import (
    OmicsPreprocessor,
    concat_omics,
    read_multiomics_dataset,
)


DATASETS = ["BRCA", "STAD", "ROSMAP", "SCZ"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KNN, SVM, NB and RF on the exact formal MCOF-se outer splits."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--random_seed", type=int, default=1)
    return parser.parse_args()


def model_factories(seed: int) -> dict[str, Callable[[], object]]:
    # Preserve Wangqi's original baseline definitions wherever possible:
    # KNN k=10; default RBF SVC with probability output; Gaussian NB;
    # RF with 100 trees.
    return {
        "KNN": lambda: KNeighborsClassifier(n_neighbors=10),
        "SVM": lambda: SVC(
            kernel="rbf",
            probability=True,
            random_state=seed,
        ),
        "NB": lambda: GaussianNB(),
        "RF": lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> dict:
    # Convert every model output to a valid probability matrix.
    # This prevents high-dimensional GaussianNB numerical errors
    # from invalidating multiclass AUC calculation.
    y_prob = np.asarray(y_prob, dtype=np.float64)

    non_finite = ~np.isfinite(y_prob)
    if non_finite.any():
        print(
            f"WARNING: replacing {int(non_finite.sum())} "
            "non-finite probability values."
        )

    y_prob = np.nan_to_num(
        y_prob,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y_prob = np.clip(y_prob, 0.0, None)

    row_sums = y_prob.sum(axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= 0.0

    if zero_rows.any():
        print(
            f"WARNING: assigning uniform probabilities to "
            f"{int(zero_rows.sum())} zero-sum rows."
        )
        y_prob[zero_rows, :] = 1.0 / float(num_classes)

    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    y_pred = np.argmax(y_prob, axis=1)
    output = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": json.dumps(
            confusion_matrix(
                y_true,
                y_pred,
                labels=np.arange(num_classes),
            ).tolist()
        ),
    }

    if num_classes == 2:
        output.update(
            {
                "f1": float(
                    f1_score(y_true, y_pred, zero_division=0)
                ),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "precision": float(
                    precision_score(
                        y_true, y_pred, zero_division=0
                    )
                ),
                "recall": float(
                    recall_score(
                        y_true, y_pred, zero_division=0
                    )
                ),
                "auc": float(
                    roc_auc_score(y_true, y_prob[:, 1])
                ),
            }
        )
    else:
        output.update(
            {
                "f1_macro": float(
                    f1_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "f1_weighted": float(
                    f1_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "auc_macro_ovr": float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                    )
                ),
                "auc_weighted_ovr": float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        average="weighted",
                    )
                ),
            }
        )

    return output


def align_probabilities(
    model,
    probabilities: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    aligned = np.zeros((len(probabilities), num_classes), dtype=float)
    for source_col, class_label in enumerate(model.classes_):
        aligned[:, int(class_label)] = probabilities[:, source_col]
    return aligned


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    split_root = Path(args.split_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metric_rows = []

    for dataset in DATASETS:
        print(f"\n{'=' * 80}\nDataset: {dataset}\n{'=' * 80}")
        omics_list, labels, _ = read_multiomics_dataset(
            data_root / dataset
        )
        split_df = pd.read_csv(
            split_root / f"{dataset}_outer_splits.csv"
        )
        num_classes = int(len(np.unique(labels)))

        for repeat in range(1, 6):
            current = split_df[split_df["repeat"] == repeat]
            train_idx = (
                current.loc[current["subset"] == "train", "sample_index"]
                .astype(int)
                .to_numpy()
            )
            test_idx = (
                current.loc[current["subset"] == "test", "sample_index"]
                .astype(int)
                .to_numpy()
            )

            y_train = labels[train_idx]
            y_test = labels[test_idx]
            raw_train = [x[train_idx] for x in omics_list]
            raw_test = [x[test_idx] for x in omics_list]

            # Exactly match the formal MCOF-se preprocessing boundary:
            # median imputation and standard scaling fitted on outer train only.
            preprocessor = OmicsPreprocessor(
                max_missing_rate=1.0,
                max_zero_rate=1.0,
                impute_strategy="median",
                scaler="standard",
                select_k=None,
                selector="none",
            ).fit(raw_train, y_train)

            x_train = concat_omics(
                preprocessor.transform(raw_train)
            )
            x_test = concat_omics(
                preprocessor.transform(raw_test)
            )

            for model_name, factory in model_factories(
                args.random_seed + repeat
            ).items():
                print(
                    f"{dataset} repeat {repeat}: fitting {model_name}"
                )
                model = factory()
                model.fit(x_train, y_train)
                probabilities = model.predict_proba(x_test)
                probabilities = align_probabilities(
                    model,
                    probabilities,
                    num_classes,
                )
                predictions = np.argmax(probabilities, axis=1)
                metrics = compute_metrics(
                    y_test,
                    probabilities,
                    num_classes,
                )

                run_dir = (
                    output_root
                    / dataset
                    / model_name
                    / f"repeat_{repeat}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)

                prediction_df = pd.DataFrame(
                    {
                        "sample_index": test_idx,
                        "y_true": y_test,
                        "y_pred": predictions,
                    }
                )
                for class_idx in range(num_classes):
                    prediction_df[
                        f"prob_class_{class_idx}"
                    ] = probabilities[:, class_idx]
                prediction_df.to_csv(
                    run_dir / "predictions_test.csv",
                    index=False,
                )

                row = {
                    "dataset": dataset,
                    "model": model_name,
                    "repeat": repeat,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
                row.update(metrics)
                metric_rows.append(row)

                with (run_dir / "metrics_test.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(
                        row,
                        handle,
                        indent=2,
                        ensure_ascii=False,
                    )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_file = output_root / "ML_repeat_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)

    numeric_columns = [
        col
        for col in metrics_df.columns
        if col
        not in {
            "dataset",
            "model",
            "repeat",
            "confusion_matrix",
        }
        and pd.api.types.is_numeric_dtype(metrics_df[col])
    ]

    summary_rows = []
    for (dataset, model), group in metrics_df.groupby(
        ["dataset", "model"]
    ):
        for metric in numeric_columns:
            values = group[metric].dropna()
            if len(values) == 0:
                continue
            summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "n_repeats": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "mean_sd_3dp": (
                        f"{values.mean():.3f} ± "
                        f"{values.std(ddof=1):.3f}"
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_root / "ML_summary_long.csv"
    summary_df.to_csv(summary_file, index=False)

    wide = (
        summary_df.pivot_table(
            index=["dataset", "model"],
            columns="metric",
            values="mean_sd_3dp",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    wide_file = output_root / "ML_summary_wide.csv"
    wide.to_csv(wide_file, index=False)

    print("\nAll ML baselines completed successfully.")
    print(f"Repeat metrics: {metrics_file}")
    print(f"Long summary: {summary_file}")
    print(f"Wide summary: {wide_file}")


if __name__ == "__main__":
    main()
