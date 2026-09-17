
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _clean_repeat_label(value: object) -> str:
    text = str(value).strip()
    digits = "".join(character for character in text if character.isdigit())
    return f"Repeat {int(digits)}" if digits else text


def read_pairwise_matrix(path: Path) -> pd.DataFrame:
    """
    Read a five-repeat pairwise matrix from either:
    1) a square CSV with row labels in the first column;
    2) a plain square numeric CSV;
    3) a long pairwise CSV containing two repeat columns and one value column.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)

    # Long-format detection.
    lower = {str(column).lower(): column for column in raw.columns}
    repeat_candidates = [
        column
        for column in raw.columns
        if "repeat" in str(column).lower()
        or str(column).lower() in {"run1", "run2", "i", "j"}
    ]

    numeric_candidates = [
        column
        for column in raw.columns
        if pd.api.types.is_numeric_dtype(raw[column])
        and column not in repeat_candidates
    ]

    if len(repeat_candidates) >= 2 and numeric_candidates:
        value_priority = [
            column
            for column in numeric_candidates
            if any(
                token in str(column).lower()
                for token in ["spearman", "jaccard", "cor", "similarity", "value"]
            )
        ]
        value_column = (
            value_priority[0]
            if value_priority
            else numeric_candidates[-1]
        )
        first_repeat = repeat_candidates[0]
        second_repeat = repeat_candidates[1]

        labels = sorted(
            {
                _clean_repeat_label(value)
                for value in pd.concat(
                    [raw[first_repeat], raw[second_repeat]],
                    ignore_index=True,
                )
            },
            key=lambda text: int("".join(filter(str.isdigit, text)) or 0),
        )

        matrix = pd.DataFrame(
            np.eye(len(labels), dtype=float),
            index=labels,
            columns=labels,
        )

        for _, row in raw.iterrows():
            first = _clean_repeat_label(row[first_repeat])
            second = _clean_repeat_label(row[second_repeat])
            value = float(row[value_column])
            matrix.loc[first, second] = value
            matrix.loc[second, first] = value

        return matrix

    # Try reading with the first column as a row index.
    indexed = pd.read_csv(path, index_col=0)
    indexed_numeric = indexed.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if (
        indexed_numeric.shape[0] == indexed_numeric.shape[1]
        and indexed_numeric.notna().all().all()
    ):
        indexed_numeric.index = [
            _clean_repeat_label(value)
            for value in indexed_numeric.index
        ]
        indexed_numeric.columns = [
            _clean_repeat_label(value)
            for value in indexed_numeric.columns
        ]
        return indexed_numeric

    # Plain square numeric CSV.
    plain = pd.read_csv(path, header=None)
    plain_numeric = plain.apply(pd.to_numeric, errors="coerce")

    if (
        plain_numeric.shape[0] == plain_numeric.shape[1]
        and plain_numeric.notna().all().all()
    ):
        labels = [
            f"Repeat {index}"
            for index in range(1, len(plain_numeric) + 1)
        ]
        plain_numeric.index = labels
        plain_numeric.columns = labels
        return plain_numeric

    raise ValueError(
        f"Could not recognize pairwise matrix structure in {path}. "
        f"Columns: {list(raw.columns)}"
    )


def upper_triangle_values(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    indices = np.triu_indices_from(values, k=1)
    result = values[indices]
    return result[np.isfinite(result)]


def locate_file(
    dataset_dir: Path,
    preferred_patterns: List[str],
) -> Path:
    for pattern in preferred_patterns:
        matches = sorted(dataset_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No matching file in {dataset_dir}. "
        f"Patterns: {preferred_patterns}"
    )

import argparse
import csv
import json
import shutil
from typing import Dict


DATASETS = ["BRCA", "STAD", "ROSMAP", "SCZ"]
MODEL_ORDER = [
    "DIABLO",
    "KNN",
    "SVM",
    "NB",
    "RF",
    "MOGONET",
    "MCOF",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_performance_csv", required=True)
    parser.add_argument("--ablation_repeat_csv", required=True)
    parser.add_argument("--efficiency_repeat_csv", required=True)
    parser.add_argument("--stability_root", required=True)
    parser.add_argument("--diablo_markers_csv", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--enrichment_file",
        default="",
        help=(
            "Optional current enrichment CSV/TSV/XLSX. "
            "Do not pass an old single-run enrichment result."
        ),
    )
    return parser.parse_args()


def write_table_s1(output_root: Path) -> Path:
    rows = [
        {
            "model": "MCOF",
            "model_family": "Deep learning",
            "input_strategy": (
                "Omics-specific projectors followed by SE-style "
                "channel recalibration and a convolutional classifier"
            ),
            "outer_evaluation": (
                "Five stratified 80:20 external holdout repeats"
            ),
            "inner_validation_or_tuning": (
                "Five-fold CV within each external training set"
            ),
            "main_hyperparameters": (
                "Adam; learning rate=0.001; batch size=64; "
                "maximum epochs=300; patience=30; hidden dimension=128; "
                "convolution channels=64; kernel size=5; dropout=0.3"
            ),
            "test_metric_protocol": (
                "Probability-based AUC; external test set evaluated once "
                "per repeat"
            ),
        },
        {
            "model": "DIABLO",
            "model_family": "Statistical integration",
            "input_strategy": "block.splsda using separate omics blocks",
            "outer_evaluation": (
                "Same five stratified 80:20 external holdout repeats"
            ),
            "inner_validation_or_tuning": (
                "Five-fold M-fold tuning within each external training set"
            ),
            "main_hyperparameters": (
                "Design off-diagonal=0.1; centroids distance; "
                "ncomp selected by weighted-vote overall BER; "
                "keepX candidates=5,6,7,8,9,10,12,14,16,18,20,25,30"
            ),
            "test_metric_protocol": (
                "Weighted prediction scores used for probability-based AUC"
            ),
        },
        {
            "model": "KNN",
            "model_family": "Machine learning",
            "input_strategy": "Concatenated preprocessed omics features",
            "outer_evaluation": (
                "Same five stratified 80:20 external holdout repeats"
            ),
            "inner_validation_or_tuning": "No additional tuning",
            "main_hyperparameters": "n_neighbors=10",
            "test_metric_protocol": "predict_proba used for AUC",
        },
        {
            "model": "SVM",
            "model_family": "Machine learning",
            "input_strategy": "Concatenated preprocessed omics features",
            "outer_evaluation": (
                "Same five stratified 80:20 external holdout repeats"
            ),
            "inner_validation_or_tuning": "No additional tuning",
            "main_hyperparameters": (
                "RBF kernel; C=1; gamma='scale'; probability=True"
            ),
            "test_metric_protocol": "predict_proba used for AUC",
        },
        {
            "model": "NB",
            "model_family": "Machine learning",
            "input_strategy": "Concatenated preprocessed omics features",
            "outer_evaluation": (
                "Same five stratified 80:20 external holdout repeats"
            ),
            "inner_validation_or_tuning": "No additional tuning",
            "main_hyperparameters": "GaussianNB defaults",
            "test_metric_protocol": (
                "Normalized predict_proba output used for AUC"
            ),
        },
        {
            "model": "RF",
            "model_family": "Machine learning",
            "input_strategy": "Concatenated preprocessed omics features",
            "outer_evaluation": (
                "Same five stratified 80:20 external holdout repeats"
            ),
            "inner_validation_or_tuning": "No additional tuning",
            "main_hyperparameters": (
                "100 trees; repeat-specific random seed; n_jobs=-1"
            ),
            "test_metric_protocol": "predict_proba used for AUC",
        },
        {
            "model": "MOGONET",
            "model_family": "Graph deep learning",
            "input_strategy": (
                "Omics-specific sample-similarity GCNs integrated by VCDN"
            ),
            "outer_evaluation": (
                "Same five stratified 80:20 external holdout repeats"
            ),
            "inner_validation_or_tuning": (
                "Fixed training schedule following the supplied implementation"
            ),
            "main_hyperparameters": (
                "GCN pretraining=500 epochs; joint training=2500 epochs; "
                "lr_pretrain=0.001; lr_encoder=0.0005; lr_VCDN=0.001; "
                "adjacency parameter=10 for BRCA/STAD and 2 for ROSMAP/SCZ"
            ),
            "test_metric_protocol": (
                "Softmax probabilities used for AUC after fixed training"
            ),
        },
    ]

    path = output_root / "Table_S1_model_configurations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_table_s2(
    performance_path: Path,
    output_root: Path,
) -> Path:
    df = pd.read_csv(performance_path)

    metric_columns = [
        "acc",
        "f1",
        "auc",
        "mcc",
        "f1_macro",
        "f1_weighted",
        "auc_macro_ovr",
        "auc_weighted_ovr",
    ]
    keep = [
        column
        for column in [
            "dataset",
            "model",
            "repeat",
            *metric_columns,
        ]
        if column in df.columns
    ]

    result = df[keep].copy()
    result["dataset"] = pd.Categorical(
        result["dataset"],
        categories=DATASETS,
        ordered=True,
    )
    result["model"] = pd.Categorical(
        result["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    result = result.sort_values(
        ["dataset", "model", "repeat"]
    ).reset_index(drop=True)

    expected_rows = 4 * 7 * 5
    if len(result) != expected_rows:
        raise ValueError(
            f"Table S2 expected {expected_rows} rows, found {len(result)}."
        )

    path = output_root / "Table_S2_repeat_level_performance.csv"
    result.to_csv(path, index=False)
    return path


def build_table_s3(
    ablation_path: Path,
    output_root: Path,
) -> Path:
    df = pd.read_csv(ablation_path)
    rename_map = {
        "manuscript_model": "model",
    }
    df = df.rename(columns=rename_map)

    metric_columns = [
        "acc",
        "f1",
        "auc",
        "mcc",
        "f1_macro",
        "f1_weighted",
        "auc_macro_ovr",
        "auc_weighted_ovr",
        "balanced_acc",
        "loss",
    ]
    keep = [
        column
        for column in [
            "dataset",
            "model",
            "code_model",
            "repeat",
            *metric_columns,
        ]
        if column in df.columns
    ]

    result = df[keep].copy()
    result = result.sort_values(
        ["dataset", "model", "repeat"]
    ).reset_index(drop=True)

    expected_rows = 4 * 3 * 5
    if len(result) != expected_rows:
        raise ValueError(
            f"Table S3 expected {expected_rows} rows, found {len(result)}."
        )

    path = output_root / "Table_S3_strict_ablation_repeat_results.csv"
    result.to_csv(path, index=False)
    return path


def build_table_s4(
    efficiency_path: Path,
    output_root: Path,
) -> Path:
    df = pd.read_csv(efficiency_path)

    metrics = [
        "trainable_parameters",
        "forward_testset_ms_mean",
        "latency_ms_per_sample",
        "throughput_samples_per_second",
        "peak_allocated_mb",
    ]

    rows = []
    for dataset in DATASETS:
        for model in ["MCOF", "MOGONET"]:
            group = df[
                df["dataset"].eq(dataset)
                & df["model"].eq(model)
            ]

            if len(group) != 5:
                raise ValueError(
                    f"{dataset}/{model}: expected 5 efficiency repeats, "
                    f"found {len(group)}."
                )

            row: Dict[str, object] = {
                "dataset": dataset,
                "model": model,
                "n_repeats": 5,
            }

            for metric in metrics:
                values = pd.to_numeric(
                    group[metric],
                    errors="raise",
                )
                mean_value = float(values.mean())
                sd_value = float(values.std(ddof=1))

                if metric == "trainable_parameters":
                    row[metric] = int(round(mean_value))
                else:
                    row[f"{metric}_mean"] = mean_value
                    row[f"{metric}_sd"] = sd_value
                    row[f"{metric}_mean_sd"] = (
                        f"{mean_value:.6f} ± {sd_value:.6f}"
                    )

            rows.append(row)

    path = output_root / "Table_S4_computational_characteristics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def choose_stable_feature_file(
    dataset_dir: Path,
    dataset: str,
) -> Path:
    patterns = [
        f"{dataset}_stable_top100_annotated.csv",
        f"{dataset}_stable_top100.csv",
        "*stable_top100*annotated*.csv",
        "*stable_top100*.csv",
        f"{dataset}_IG_stability_all_features_annotated.csv",
        f"{dataset}_IG_stability_all_features.csv",
        "*IG_stability_all_features*annotated*.csv",
        "*IG_stability_all_features*.csv",
    ]
    return locate_file(dataset_dir, patterns)


def build_table_s5(
    stability_root: Path,
    output_root: Path,
) -> Path:
    frames = []

    for dataset in DATASETS:
        dataset_dir = stability_root / dataset
        path = choose_stable_feature_file(
            dataset_dir,
            dataset,
        )
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", dataset)
        frame.insert(1, "source_file", str(path))

        # If this is an all-feature file, keep the top 100 using the most
        # likely ranking column.
        if len(frame) > 100:
            ranking_candidates = [
                column
                for column in frame.columns
                if any(
                    token in str(column).lower()
                    for token in [
                        "mean_rank",
                        "average_rank",
                        "avg_rank",
                        "median_rank",
                        "rank_mean",
                    ]
                )
            ]
            if ranking_candidates:
                frame = frame.sort_values(
                    ranking_candidates[0],
                    ascending=True,
                ).head(100)
            else:
                importance_candidates = [
                    column
                    for column in frame.columns
                    if any(
                        token in str(column).lower()
                        for token in [
                            "mean_abs",
                            "mean_importance",
                            "average_importance",
                            "importance_mean",
                        ]
                    )
                ]
                if importance_candidates:
                    frame = frame.sort_values(
                        importance_candidates[0],
                        ascending=False,
                    ).head(100)
                else:
                    frame = frame.head(100)

        frames.append(frame)

    result = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    path = output_root / "Table_S5_stable_IG_biomarkers.csv"
    result.to_csv(path, index=False)
    return path


def find_frequency_column(
    frame: pd.DataFrame,
    top_k: int,
) -> Optional[str]:
    for column in frame.columns:
        lower = str(column).lower()
        if (
            str(top_k) in lower
            and any(
                token in lower
                for token in ["freq", "count", "times", "selection"]
            )
        ):
            return str(column)
    return None


def build_table_s6(
    stability_root: Path,
    output_root: Path,
) -> Path:
    rows = []

    for dataset in DATASETS:
        dataset_dir = stability_root / dataset

        spearman_path = locate_file(
            dataset_dir,
            [
                f"{dataset}_repeat_rank_spearman.csv",
                "*repeat_rank_spearman*.csv",
            ],
        )
        spearman_matrix = read_pairwise_matrix(
            spearman_path
        )
        spearman_values = upper_triangle_values(
            spearman_matrix
        )

        row: Dict[str, object] = {
            "dataset": dataset,
            "n_repeat_pairs": len(spearman_values),
            "spearman_mean": float(
                spearman_values.mean()
            ),
            "spearman_sd": float(
                spearman_values.std(ddof=1)
            ),
            "spearman_median": float(
                np.median(spearman_values)
            ),
            "spearman_min": float(
                spearman_values.min()
            ),
            "spearman_max": float(
                spearman_values.max()
            ),
        }

        for top_k in [50, 100, 200]:
            jaccard_path = locate_file(
                dataset_dir,
                [
                    f"{dataset}_top{top_k}_jaccard.csv",
                    f"*top{top_k}*jaccard*.csv",
                ],
            )
            values = upper_triangle_values(
                read_pairwise_matrix(jaccard_path)
            )
            row[f"jaccard_top{top_k}_mean"] = float(
                values.mean()
            )
            row[f"jaccard_top{top_k}_sd"] = float(
                values.std(ddof=1)
            )
            row[f"jaccard_top{top_k}_median"] = float(
                np.median(values)
            )

        stability_path = locate_file(
            dataset_dir,
            [
                f"{dataset}_IG_stability_all_features_annotated.csv",
                f"{dataset}_IG_stability_all_features.csv",
                "*IG_stability_all_features*annotated*.csv",
                "*IG_stability_all_features*.csv",
            ],
        )
        stability = pd.read_csv(stability_path)
        row["n_features_evaluated"] = len(stability)

        top50_column = find_frequency_column(
            stability,
            50,
        )
        top100_column = find_frequency_column(
            stability,
            100,
        )

        if top50_column:
            values = pd.to_numeric(
                stability[top50_column],
                errors="coerce",
            )
            row["n_features_top50_in_at_least_4_repeats"] = int(
                (values >= 4).sum()
            )
        else:
            row[
                "n_features_top50_in_at_least_4_repeats"
            ] = np.nan

        if top100_column:
            values = pd.to_numeric(
                stability[top100_column],
                errors="coerce",
            )
            row["n_features_top100_in_all_5_repeats"] = int(
                (values >= 5).sum()
            )
        else:
            row[
                "n_features_top100_in_all_5_repeats"
            ] = np.nan

        rows.append(row)

    path = output_root / "Table_S6_IG_ranking_stability.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_table_s7(
    diablo_path: Path,
    output_root: Path,
) -> Path:
    df = pd.read_csv(diablo_path)
    path = output_root / "Table_S7_DIABLO_selected_features.csv"
    df.to_csv(path, index=False)
    return path


def handle_table_s8(
    enrichment_file: str,
    output_root: Path,
) -> Path:
    if enrichment_file:
        source = Path(enrichment_file)
        if not source.is_file():
            raise FileNotFoundError(source)

        suffix = source.suffix.lower()
        destination = (
            output_root
            / f"Table_S8_current_enrichment_results{suffix}"
        )
        shutil.copy2(source, destination)
        return destination

    path = output_root / "Table_S8_NOT_GENERATED.txt"
    path.write_text(
        (
            "Table S8 was not generated because no enrichment result based "
            "on the current five-repeat stable IG biomarkers was supplied.\n"
            "Do not reuse the old single-run enrichment table as a formal "
            "supplementary result.\n"
        ),
        encoding="utf-8",
    )
    return path


def write_manifest(
    paths: List[Path],
    output_root: Path,
) -> Path:
    captions = {
        "Table_S1_model_configurations.csv": (
            "Model configurations and training hyperparameters."
        ),
        "Table_S2_repeat_level_performance.csv": (
            "Five-repeat external-test performance of MCOF and all "
            "baseline models."
        ),
        "Table_S3_strict_ablation_repeat_results.csv": (
            "Five-repeat performance of the complete and strictly "
            "ablated MCOF variants."
        ),
        "Table_S4_computational_characteristics.csv": (
            "Computational characteristics of MCOF and MOGONET."
        ),
        "Table_S5_stable_IG_biomarkers.csv": (
            "Stable biomarkers identified by MCOF using integrated gradients."
        ),
        "Table_S6_IG_ranking_stability.csv": (
            "Cross-repeat stability statistics for integrated-gradients "
            "feature rankings."
        ),
        "Table_S7_DIABLO_selected_features.csv": (
            "Features selected by DIABLO across datasets and repeats."
        ),
    }

    rows = []
    for path in paths:
        rows.append(
            {
                "file": path.name,
                "caption": captions.get(
                    path.name,
                    (
                        "Current enrichment results."
                        if path.name.startswith("Table_S8_current")
                        else "Status note: current enrichment not generated."
                    ),
                ),
                "size_bytes": path.stat().st_size,
            }
        )

    manifest = output_root / "Supplementary_tables_manifest.csv"
    pd.DataFrame(rows).to_csv(
        manifest,
        index=False,
    )
    return manifest


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = [
        write_table_s1(output_root),
        build_table_s2(
            Path(args.final_performance_csv),
            output_root,
        ),
        build_table_s3(
            Path(args.ablation_repeat_csv),
            output_root,
        ),
        build_table_s4(
            Path(args.efficiency_repeat_csv),
            output_root,
        ),
        build_table_s5(
            Path(args.stability_root),
            output_root,
        ),
        build_table_s6(
            Path(args.stability_root),
            output_root,
        ),
        build_table_s7(
            Path(args.diablo_markers_csv),
            output_root,
        ),
        handle_table_s8(
            args.enrichment_file,
            output_root,
        ),
    ]

    manifest = write_manifest(
        paths,
        output_root,
    )

    print("Generated supplementary tables:")
    for path in paths:
        print(path)
    print(manifest)


if __name__ == "__main__":
    main()
