from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_ORDER = ["BRCA", "STAD", "ROSMAP", "SCZ"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcof_file", required=True)
    parser.add_argument("--mogonet_file", required=True)
    parser.add_argument("--output_root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mcof = pd.read_csv(args.mcof_file)
    mogonet = pd.read_csv(args.mogonet_file)
    combined = pd.concat([mcof, mogonet], ignore_index=True, sort=False)

    assert len(combined) == 40
    counts = combined.groupby(["dataset", "model"]).size()
    assert counts.eq(5).all()

    combined.to_csv(
        output_root / "efficiency_all_models_by_repeat.csv",
        index=False,
    )

    metrics = [
        "total_parameters",
        "trainable_parameters",
        "parameter_memory_fp32_mb",
        "checkpoint_size_mb",
        "forward_testset_ms_mean",
        "latency_ms_per_sample",
        "throughput_samples_per_second",
        "peak_allocated_mb",
        "incremental_peak_allocated_mb",
    ]

    summary_rows = []
    for dataset in DATASET_ORDER:
        for model in ["MCOF", "MOGONET"]:
            group = combined[
                combined["dataset"].eq(dataset)
                & combined["model"].eq(model)
            ]
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="raise")
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                        "mean_sd": (
                            f"{values.mean():.6f} ± "
                            f"{values.std(ddof=1):.6f}"
                        ),
                    }
                )

    summary_long = pd.DataFrame(summary_rows)
    summary_long.to_csv(
        output_root / "efficiency_summary_long.csv",
        index=False,
    )

    comparison_rows = []
    for dataset in DATASET_ORDER:
        row = {"dataset": dataset}
        model_values = {}

        for model in ["MCOF", "MOGONET"]:
            group = combined[
                combined["dataset"].eq(dataset)
                & combined["model"].eq(model)
            ]
            model_values[model] = {
                metric: float(
                    pd.to_numeric(group[metric]).mean()
                )
                for metric in metrics
            }
            for metric, value in model_values[model].items():
                row[f"{model}_{metric}"] = value

        mcof_values = model_values["MCOF"]
        mogonet_values = model_values["MOGONET"]

        row["MCOF_parameter_reduction_pct_vs_MOGONET"] = (
            100.0
            * (
                mogonet_values["total_parameters"]
                - mcof_values["total_parameters"]
            )
            / mogonet_values["total_parameters"]
        )
        row["MCOF_forward_speedup_vs_MOGONET"] = (
            mogonet_values["forward_testset_ms_mean"]
            / mcof_values["forward_testset_ms_mean"]
        )
        row["MCOF_peak_memory_reduction_pct_vs_MOGONET"] = (
            100.0
            * (
                mogonet_values["peak_allocated_mb"]
                - mcof_values["peak_allocated_mb"]
            )
            / mogonet_values["peak_allocated_mb"]
        )

        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        output_root / "MCOF_vs_MOGONET_efficiency_comparison.csv",
        index=False,
    )

    print("\nEfficiency comparison:")
    display_columns = [
        "dataset",
        "MCOF_total_parameters",
        "MOGONET_total_parameters",
        "MCOF_parameter_reduction_pct_vs_MOGONET",
        "MCOF_forward_testset_ms_mean",
        "MOGONET_forward_testset_ms_mean",
        "MCOF_forward_speedup_vs_MOGONET",
        "MCOF_peak_allocated_mb",
        "MOGONET_peak_allocated_mb",
        "MCOF_peak_memory_reduction_pct_vs_MOGONET",
    ]
    print(comparison[display_columns].to_string(index=False))

    print("\nSaved:")
    print(output_root / "efficiency_all_models_by_repeat.csv")
    print(output_root / "efficiency_summary_long.csv")
    print(output_root / "MCOF_vs_MOGONET_efficiency_comparison.csv")


if __name__ == "__main__":
    main()
