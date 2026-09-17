from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DATASET_ORDER = ["BRCA", "STAD", "ROSMAP", "SCZ"]
MODEL_ORDER = [
    "DIABLO",
    "KNN",
    "SVM",
    "NB",
    "RF",
    "MOGONET",
    "MCOF",
]

EXPECTED_METRICS: Dict[str, List[str]] = {
    "BRCA": [
        "acc",
        "f1_macro",
        "f1_weighted",
        "auc_macro_ovr",
        "auc_weighted_ovr",
    ],
    "STAD": [
        "acc",
        "f1_macro",
        "f1_weighted",
        "auc_macro_ovr",
        "auc_weighted_ovr",
    ],
    "ROSMAP": [
        "acc",
        "f1",
        "auc",
        "mcc",
    ],
    "SCZ": [
        "acc",
        "f1",
        "auc",
        "mcc",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current_repeat_csv", required=True)
    parser.add_argument("--mogonet_repeat_csv", required=True)
    parser.add_argument("--output_root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    current = pd.read_csv(args.current_repeat_csv)
    mogonet = pd.read_csv(args.mogonet_repeat_csv)

    required = {"dataset", "model", "repeat"}
    for name, frame in [
        ("current", current),
        ("MOGONET", mogonet),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{name} file is missing columns: {sorted(missing)}"
            )

    mogonet = mogonet.copy()
    mogonet["model"] = "MOGONET"

    combined = pd.concat(
        [current, mogonet],
        ignore_index=True,
        sort=False,
    )

    combined["dataset"] = combined["dataset"].astype(str)
    combined["model"] = combined["model"].astype(str)
    combined["repeat"] = pd.to_numeric(
        combined["repeat"],
        errors="raise",
    ).astype(int)

    combined = combined[
        combined["dataset"].isin(DATASET_ORDER)
        & combined["model"].isin(MODEL_ORDER)
    ].copy()

    duplicates = combined.duplicated(
        ["dataset", "model", "repeat"],
        keep=False,
    )
    if duplicates.any():
        raise ValueError(
            "Duplicated dataset/model/repeat rows:\n"
            + combined.loc[
                duplicates,
                ["dataset", "model", "repeat"],
            ].to_string(index=False)
        )

    counts = (
        combined.groupby(["dataset", "model"])
        .size()
        .rename("n_repeats")
        .reset_index()
    )

    expected_rows = 4 * 7 * 5
    if len(combined) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} repeat rows, found {len(combined)}."
        )
    if not counts["n_repeats"].eq(5).all():
        raise ValueError(
            "At least one dataset/model combination lacks five repeats."
        )

    for dataset in DATASET_ORDER:
        group = combined[
            combined["dataset"].eq(dataset)
        ]
        for metric in EXPECTED_METRICS[dataset]:
            if metric not in group.columns:
                raise ValueError(
                    f"{dataset}: missing metric column {metric}"
                )
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )
            if values.isna().any():
                bad = group.loc[
                    values.isna(),
                    ["dataset", "model", "repeat", metric],
                ]
                raise ValueError(
                    f"{dataset}: missing {metric} values:\n"
                    + bad.to_string(index=False)
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{dataset}: non-finite values in {metric}"
                )

    combined["dataset"] = pd.Categorical(
        combined["dataset"],
        categories=DATASET_ORDER,
        ordered=True,
    )
    combined["model"] = pd.Categorical(
        combined["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    combined = combined.sort_values(
        ["dataset", "model", "repeat"]
    ).reset_index(drop=True)

    repeat_path = (
        output_root / "all_7_models_repeat_metrics.csv"
    )
    combined.to_csv(repeat_path, index=False)

    summary_rows = []
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            group = combined[
                combined["dataset"].eq(dataset)
                & combined["model"].eq(model)
            ]
            for metric in EXPECTED_METRICS[dataset]:
                values = pd.to_numeric(
                    group[metric],
                    errors="raise",
                )
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

    summary = pd.DataFrame(summary_rows)
    summary_path = (
        output_root / "all_7_models_summary_long.csv"
    )
    summary.to_csv(summary_path, index=False)

    table_rows = []
    for dataset in DATASET_ORDER:
        for metric in EXPECTED_METRICS[dataset]:
            metric_subset = summary[
                summary["dataset"].eq(dataset)
                & summary["metric"].eq(metric)
            ]
            best_mean = metric_subset["mean"].max()

            row = {
                "dataset": dataset,
                "metric": metric,
            }
            for model in MODEL_ORDER:
                selected = metric_subset[
                    metric_subset["model"].eq(model)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"Missing summary for {dataset}/{metric}/{model}"
                    )
                row[model] = selected.iloc[0]["mean_sd_3dp"]
                row[f"{model}_is_best"] = bool(
                    np.isclose(
                        float(selected.iloc[0]["mean"]),
                        best_mean,
                    )
                )
            table_rows.append(row)

    table = pd.DataFrame(table_rows)
    table_path = (
        output_root / "Table2_all_7_models.csv"
    )
    table.to_csv(table_path, index=False)

    delta_rows = []
    for dataset in DATASET_ORDER:
        for metric in EXPECTED_METRICS[dataset]:
            mcof = (
                combined[
                    combined["dataset"].eq(dataset)
                    & combined["model"].eq("MCOF")
                ][["repeat", metric]]
                .rename(columns={metric: "mcof"})
            )

            for baseline in MODEL_ORDER:
                if baseline == "MCOF":
                    continue

                baseline_frame = (
                    combined[
                        combined["dataset"].eq(dataset)
                        & combined["model"].eq(baseline)
                    ][["repeat", metric]]
                    .rename(columns={metric: "baseline"})
                )

                paired = mcof.merge(
                    baseline_frame,
                    on="repeat",
                    how="inner",
                    validate="one_to_one",
                )
                if len(paired) != 5:
                    raise ValueError(
                        f"{dataset}/{metric}/{baseline}: "
                        f"expected five paired repeats."
                    )

                delta = paired["mcof"] - paired["baseline"]
                tolerance = 1e-12

                delta_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "baseline": baseline,
                        "n_pairs": len(delta),
                        "mean_delta_MCOF_minus_baseline": float(
                            delta.mean()
                        ),
                        "sd_delta": float(delta.std(ddof=1)),
                        "wins": int((delta > tolerance).sum()),
                        "ties": int(
                            (delta.abs() <= tolerance).sum()
                        ),
                        "losses": int(
                            (delta < -tolerance).sum()
                        ),
                    }
                )

    delta = pd.DataFrame(delta_rows)
    delta_path = (
        output_root / "paired_delta_MCOF_vs_all_baselines.csv"
    )
    delta.to_csv(delta_path, index=False)

    print("\nRun counts:")
    print(counts.to_string(index=False))

    print("\nManuscript-style performance table:")
    print(
        table[
            ["dataset", "metric"] + MODEL_ORDER
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(repeat_path)
    print(summary_path)
    print(table_path)
    print(delta_path)


if __name__ == "__main__":
    main()
