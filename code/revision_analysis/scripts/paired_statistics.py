from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common import DATASET_CONFIG, DATASET_ORDER, exact_sign_flip_pvalue, holm_adjust


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exploratory paired exact sign-flip tests for MCOF versus every "
            "baseline using the same five split identifiers."
        )
    )
    parser.add_argument("--repeat_metrics_csv", required=True)
    parser.add_argument("--output_root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.repeat_metrics_csv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    required = {"dataset", "model", "repeat"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["dataset"] = frame["dataset"].astype(str)
    frame["model"] = frame["model"].astype(str)
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)

    rows: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        dataset_frame = frame[frame["dataset"].eq(dataset)]
        mcof = dataset_frame[dataset_frame["model"].eq("MCOF")]
        if len(mcof) != 5:
            raise ValueError(f"{dataset}: expected five MCOF repeats, found {len(mcof)}")
        baselines = sorted(set(dataset_frame["model"]).difference({"MCOF"}))
        for metric in DATASET_CONFIG[dataset]["metrics"]:
            if metric not in frame.columns:
                raise ValueError(f"Missing metric column {metric}")
            mcof_metric = mcof[["repeat", metric]].rename(columns={metric: "mcof"})
            for baseline in baselines:
                comparison = dataset_frame[dataset_frame["model"].eq(baseline)]
                if len(comparison) != 5:
                    raise ValueError(
                        f"{dataset}/{baseline}: expected five repeats, found {len(comparison)}"
                    )
                paired = mcof_metric.merge(
                    comparison[["repeat", metric]].rename(columns={metric: "baseline"}),
                    on="repeat",
                    how="inner",
                    validate="one_to_one",
                )
                if len(paired) != 5:
                    raise ValueError(f"{dataset}/{baseline}/{metric}: expected five paired repeats")
                differences = (
                    pd.to_numeric(paired["mcof"], errors="raise")
                    - pd.to_numeric(paired["baseline"], errors="raise")
                ).to_numpy(dtype=float)
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "baseline": baseline,
                        "n_pairs": len(differences),
                        "mcof_mean": float(paired["mcof"].mean()),
                        "baseline_mean": float(paired["baseline"].mean()),
                        "mean_delta_MCOF_minus_baseline": float(differences.mean()),
                        "sd_delta": float(differences.std(ddof=1)),
                        "median_delta": float(np.median(differences)),
                        "wins": int((differences > 0).sum()),
                        "ties": int(np.isclose(differences, 0.0, atol=1e-12).sum()),
                        "losses": int((differences < 0).sum()),
                        "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(differences),
                    }
                )

    result = pd.DataFrame(rows)
    result["holm_p_within_dataset"] = np.nan
    for dataset, indices in result.groupby("dataset").groups.items():
        index_list = list(indices)
        result.loc[index_list, "holm_p_within_dataset"] = holm_adjust(
            result.loc[index_list, "exact_two_sided_sign_flip_p"].to_numpy()
        )

    result["holm_p_within_dataset_metric"] = np.nan
    for (_, _), indices in result.groupby(["dataset", "metric"]).groups.items():
        index_list = list(indices)
        result.loc[index_list, "holm_p_within_dataset_metric"] = holm_adjust(
            result.loc[index_list, "exact_two_sided_sign_flip_p"].to_numpy()
        )

    result.to_csv(output_root / "paired_exact_sign_flip_tests.csv", index=False)

    primary = []
    for dataset in DATASET_ORDER:
        metric = DATASET_CONFIG[dataset]["primary_metric"]
        primary.append(result[result["dataset"].eq(dataset) & result["metric"].eq(metric)])
    pd.concat(primary, ignore_index=True).to_csv(
        output_root / "paired_tests_primary_metrics.csv", index=False
    )

    note = (
        "Statistical interpretation note:\n"
        "The tests use five matched split identifiers and enumerate all 2^5 sign assignments. "
        "Because held-out samples can overlap across repeated holdouts, the resulting p-values "
        "are exploratory sensitivity analyses rather than evidence from five independent cohorts. "
        "With n=5, two-sided exact p-values are intrinsically coarse; effect sizes and win/tie/loss "
        "patterns should be emphasized.\n"
    )
    (output_root / "statistical_interpretation_note.txt").write_text(note, encoding="utf-8")
    print(f"Saved paired statistical comparisons to {output_root}")


if __name__ == "__main__":
    main()
