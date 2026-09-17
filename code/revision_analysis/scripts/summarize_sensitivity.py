from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common import DATASET_CONFIG, DATASET_ORDER, exact_sign_flip_pvalue, holm_adjust, mean_sd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize modality, PAM50, and feature-budget sensitivity analyses.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--formal_main_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--allow_incomplete", action="store_true")
    return parser.parse_args()


def read_repeat_metrics(path: Path, dataset: str, variant: str, analysis_type: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if len(frame) != 5:
        raise ValueError(f"{path}: expected 5 repeats, found {len(frame)}")
    frame = frame.copy()
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
    if set(frame["repeat"]) != set(range(1, 6)):
        raise ValueError(f"{path}: repeat identifiers are not 1..5")
    frame.insert(0, "analysis_type", analysis_type)
    frame.insert(1, "dataset", dataset)
    frame.insert(2, "variant", variant)
    frame.insert(3, "model", "MCOF")
    return frame


def full_metrics(formal_main_root: Path, dataset: str) -> pd.DataFrame:
    path = formal_main_root / dataset / "mcof_se" / "repeat_test_metrics.csv"
    frame = read_repeat_metrics(path, dataset, "full_multi_omics", "reference")
    return frame


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    formal_main_root = Path(args.formal_main_root)

    task_frame = pd.read_csv(workspace / "manifests" / "all_tasks.csv")
    all_frames: List[pd.DataFrame] = []
    missing_rows: List[Dict[str, str]] = []

    for dataset in DATASET_ORDER:
        all_frames.append(full_metrics(formal_main_root, dataset))

    for _, task in task_frame.iterrows():
        path = Path(str(task["output_dir"])) / "repeat_test_metrics.csv"
        if not path.is_file():
            missing_rows.append(
                {
                    "dataset": str(task["dataset"]),
                    "variant": str(task["variant"]),
                    "expected_file": str(path),
                }
            )
            continue
        all_frames.append(
            read_repeat_metrics(
                path,
                str(task["dataset"]),
                str(task["variant"]),
                str(task["analysis_type"]),
            )
        )

    if missing_rows and not args.allow_incomplete:
        pd.DataFrame(missing_rows).to_csv(output_root / "missing_tasks.csv", index=False)
        raise FileNotFoundError(
            f"{len(missing_rows)} sensitivity tasks are incomplete. See {output_root / 'missing_tasks.csv'}"
        )

    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    combined.to_csv(output_root / "sensitivity_repeat_metrics.csv", index=False)

    summary_rows = []
    for (analysis_type, dataset, variant), group in combined.groupby(
        ["analysis_type", "dataset", "variant"], sort=False
    ):
        for metric in DATASET_CONFIG[dataset]["metrics"]:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            if values.isna().all():
                continue
            summary_rows.append(
                {
                    "analysis_type": analysis_type,
                    "dataset": dataset,
                    "variant": variant,
                    "metric": metric,
                    "n_repeats": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "mean_sd": mean_sd(values.dropna().to_numpy()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "sensitivity_summary_long.csv", index=False)

    delta_rows = []
    for dataset in DATASET_ORDER:
        reference = combined[
            combined["dataset"].eq(dataset) & combined["variant"].eq("full_multi_omics")
        ]
        for variant in sorted(
            combined.loc[
                combined["dataset"].eq(dataset) & ~combined["variant"].eq("full_multi_omics"),
                "variant",
            ].unique()
        ):
            comparator = combined[
                combined["dataset"].eq(dataset) & combined["variant"].eq(variant)
            ]
            analysis_type = str(comparator["analysis_type"].iloc[0])
            for metric in DATASET_CONFIG[dataset]["metrics"]:
                if metric not in reference.columns or metric not in comparator.columns:
                    continue
                paired = reference[["repeat", metric]].merge(
                    comparator[["repeat", metric]],
                    on="repeat",
                    suffixes=("_full", "_variant"),
                    validate="one_to_one",
                )
                if len(paired) != 5:
                    raise ValueError(f"{dataset}/{variant}/{metric}: expected 5 pairs")
                differences = (
                    pd.to_numeric(paired[f"{metric}_full"], errors="raise")
                    - pd.to_numeric(paired[f"{metric}_variant"], errors="raise")
                ).to_numpy()
                delta_rows.append(
                    {
                        "analysis_type": analysis_type,
                        "dataset": dataset,
                        "variant": variant,
                        "metric": metric,
                        "mean_delta_full_minus_variant": float(differences.mean()),
                        "sd_delta": float(differences.std(ddof=1)),
                        "median_delta": float(np.median(differences)),
                        "full_wins": int((differences > 0).sum()),
                        "ties": int(np.isclose(differences, 0.0, atol=1e-12).sum()),
                        "variant_wins": int((differences < 0).sum()),
                        "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                    }
                )

    deltas = pd.DataFrame(delta_rows)
    if not deltas.empty:
        deltas["holm_p_within_analysis_dataset"] = np.nan
        for (analysis_type, dataset), indices in deltas.groupby(
            ["analysis_type", "dataset"]
        ).groups.items():
            index_list = list(indices)
            deltas.loc[index_list, "holm_p_within_analysis_dataset"] = holm_adjust(
                deltas.loc[index_list, "exact_sign_flip_p"].to_numpy()
            )
    deltas.to_csv(output_root / "paired_deltas_full_vs_sensitivity_variants.csv", index=False)

    # Reader-friendly wide table for the primary metric of each dataset.
    primary_rows = []
    for dataset in DATASET_ORDER:
        metric = DATASET_CONFIG[dataset]["primary_metric"]
        subset = summary[
            summary["dataset"].eq(dataset) & summary["metric"].eq(metric)
        ]
        for _, row in subset.iterrows():
            primary_rows.append(
                {
                    "dataset": dataset,
                    "primary_metric": metric,
                    "analysis_type": row["analysis_type"],
                    "variant": row["variant"],
                    "performance_mean_sd": row["mean_sd"],
                }
            )
    pd.DataFrame(primary_rows).to_csv(
        output_root / "primary_metric_sensitivity_table.csv", index=False
    )

    report_lines = [
        "MCOF reviewer-requested sensitivity analysis summary",
        "=" * 72,
        "",
        "Important scope: all sensitivity analyses remain conditional on the existing fixed candidate feature panels.",
        "Feature-budget variants apply f_classif inside each training fold and outer training set, but do not remove the upstream fixed-panel limitation.",
        "",
    ]
    for dataset in DATASET_ORDER:
        metric = DATASET_CONFIG[dataset]["primary_metric"]
        report_lines.append(f"{dataset} — primary metric: {metric}")
        subset = summary[
            summary["dataset"].eq(dataset) & summary["metric"].eq(metric)
        ]
        for _, row in subset.sort_values("mean", ascending=False).iterrows():
            report_lines.append(
                f"  {row['variant']}: {row['mean_sd']} ({row['analysis_type']})"
            )
        report_lines.append("")

    if missing_rows:
        report_lines.append(f"INCOMPLETE TASKS: {len(missing_rows)}")
        for row in missing_rows:
            report_lines.append(f"  {row['dataset']} / {row['variant']}")
    (output_root / "sensitivity_summary_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"Saved sensitivity summaries to {output_root}")
    if missing_rows:
        print(f"Warning: {len(missing_rows)} task outputs were missing")


if __name__ == "__main__":
    main()
