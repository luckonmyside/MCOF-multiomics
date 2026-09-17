from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


BASE = Path("/path/to/mcof-workspace")

MAIN_ROOT = (
    BASE
    / "MCOFv2_runs"
    / "formal_main_20260429_191748"
)

ABLATION_ROOT = (
    BASE
    / "MCOFv2_runs"
    / "formal_ablation_clean_20260719_230137"
)

OUTPUT_PREFIX = BASE / "MCOF_clean_ablation_20260721"

DATASETS = ["BRCA", "STAD", "ROSMAP", "SCZ"]

MODEL_SOURCES = {
    "MCOF": {
        "code_model": "mcof_se",
        "root": MAIN_ROOT,
    },
    "MCOF w/o channel attention": {
        "code_model": "mcof_no_se",
        "root": ABLATION_ROOT,
    },
    "MCOF w/o convolution": {
        "code_model": "mcof_no_conv",
        "root": ABLATION_ROOT,
    },
}

PRIMARY_METRICS = {
    "BRCA": [
        "acc",
        "f1_macro",
        "f1_weighted",
        "auc_macro_ovr",
    ],
    "STAD": [
        "acc",
        "f1_macro",
        "f1_weighted",
        "auc_macro_ovr",
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

NON_METRIC_COLUMNS = {
    "dataset",
    "manuscript_model",
    "code_model",
    "repeat",
    "checkpoint",
    "seed",
    "random_seed",
}


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 20260721,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    sampled = rng.choice(
        values,
        size=(n_bootstrap, len(values)),
        replace=True,
    )

    means = sampled.mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def paired_wilcoxon_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return np.nan

    if np.allclose(values, 0):
        return 1.0

    if wilcoxon is None:
        return np.nan

    try:
        result = wilcoxon(
            values,
            alternative="two-sided",
            zero_method="wilcox",
        )
        return float(result.pvalue)
    except ValueError:
        return np.nan


repeat_frames: list[pd.DataFrame] = []
summary_rows: list[dict] = []
missing_files: list[str] = []


# ---------------------------------------------------------
# Read and validate the three model variants
# ---------------------------------------------------------
for dataset in DATASETS:
    for manuscript_model, config in MODEL_SOURCES.items():
        code_model = config["code_model"]
        root = config["root"]

        model_dir = root / dataset / code_model
        repeat_file = model_dir / "repeat_test_metrics.csv"
        summary_file = model_dir / "summary.json"

        if not repeat_file.is_file():
            missing_files.append(str(repeat_file))
            continue

        if not summary_file.is_file():
            missing_files.append(str(summary_file))
            continue

        df = pd.read_csv(repeat_file)

        if len(df) != 5:
            raise ValueError(
                f"{repeat_file}: expected 5 repeats, "
                f"found {len(df)}"
            )

        if "repeat" not in df.columns:
            df.insert(0, "repeat", np.arange(1, len(df) + 1))

        df["repeat"] = pd.to_numeric(
            df["repeat"],
            errors="raise",
        ).astype(int)

        expected_repeats = {1, 2, 3, 4, 5}
        observed_repeats = set(df["repeat"])

        if observed_repeats != expected_repeats:
            raise ValueError(
                f"{repeat_file}: repeat IDs are "
                f"{sorted(observed_repeats)}, expected 1–5"
            )

        df.insert(0, "code_model", code_model)
        df.insert(0, "manuscript_model", manuscript_model)
        df.insert(0, "dataset", dataset)

        repeat_frames.append(df)

        numeric_metrics = []

        for column in df.columns:
            if column in NON_METRIC_COLUMNS:
                continue

            converted = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if converted.notna().any():
                df[column] = converted
                numeric_metrics.append(column)

        for metric in numeric_metrics:
            values = df[metric].dropna()

            if len(values) == 0:
                continue

            mean = float(values.mean())
            std = float(values.std(ddof=1))

            summary_rows.append(
                {
                    "dataset": dataset,
                    "manuscript_model": manuscript_model,
                    "code_model": code_model,
                    "metric": metric,
                    "n_repeats": len(values),
                    "mean": mean,
                    "std": std,
                    "mean_sd_3dp": (
                        f"{mean:.3f} ± {std:.3f}"
                    ),
                }
            )


if missing_files:
    print("Missing required files:")

    for path in missing_files:
        print("  ", path)

    raise SystemExit(1)


repeat_df = pd.concat(
    repeat_frames,
    ignore_index=True,
    sort=False,
)

summary_df = pd.DataFrame(summary_rows)


# ---------------------------------------------------------
# Save repeat-level and summary-level results
# ---------------------------------------------------------
repeat_output = Path(
    f"{OUTPUT_PREFIX}_repeat_metrics.csv"
)
summary_long_output = Path(
    f"{OUTPUT_PREFIX}_summary_long.csv"
)
summary_wide_output = Path(
    f"{OUTPUT_PREFIX}_summary_wide.csv"
)
primary_output = Path(
    f"{OUTPUT_PREFIX}_primary_table.csv"
)

repeat_df.to_csv(repeat_output, index=False)
summary_df.to_csv(summary_long_output, index=False)

summary_wide = (
    summary_df
    .pivot_table(
        index=[
            "dataset",
            "manuscript_model",
            "code_model",
        ],
        columns="metric",
        values="mean_sd_3dp",
        aggfunc="first",
    )
    .reset_index()
)

summary_wide.columns.name = None
summary_wide.to_csv(summary_wide_output, index=False)


primary_rows = []

for dataset in DATASETS:
    for manuscript_model in MODEL_SOURCES:
        model_rows = summary_df[
            (summary_df["dataset"] == dataset)
            & (
                summary_df["manuscript_model"]
                == manuscript_model
            )
        ]

        row = {
            "dataset": dataset,
            "model": manuscript_model,
        }

        for metric in PRIMARY_METRICS[dataset]:
            matched = model_rows[
                model_rows["metric"] == metric
            ]

            row[metric] = (
                matched.iloc[0]["mean_sd_3dp"]
                if len(matched)
                else ""
            )

        primary_rows.append(row)

primary_df = pd.DataFrame(primary_rows)
primary_df.to_csv(primary_output, index=False)


# ---------------------------------------------------------
# Paired comparisons:
# positive improvement_delta always favors MCOF
# ---------------------------------------------------------
paired_rows = []

metric_columns = [
    column
    for column in repeat_df.columns
    if column not in NON_METRIC_COLUMNS
    and pd.api.types.is_numeric_dtype(repeat_df[column])
]

for dataset in DATASETS:
    mcof = repeat_df[
        (repeat_df["dataset"] == dataset)
        & (repeat_df["manuscript_model"] == "MCOF")
    ].copy()

    for ablation_model in [
        "MCOF w/o channel attention",
        "MCOF w/o convolution",
    ]:
        ablation = repeat_df[
            (repeat_df["dataset"] == dataset)
            & (
                repeat_df["manuscript_model"]
                == ablation_model
            )
        ].copy()

        merged = mcof.merge(
            ablation,
            on=["dataset", "repeat"],
            suffixes=("_mcof", "_ablation"),
            validate="one_to_one",
        )

        for metric in metric_columns:
            mcof_column = f"{metric}_mcof"
            ablation_column = f"{metric}_ablation"

            if (
                mcof_column not in merged.columns
                or ablation_column not in merged.columns
            ):
                continue

            paired = merged[
                [
                    "repeat",
                    mcof_column,
                    ablation_column,
                ]
            ].dropna()

            if len(paired) == 0:
                continue

            raw_delta = (
                paired[mcof_column]
                - paired[ablation_column]
            ).to_numpy(dtype=float)

            # Higher values are better except for loss.
            if metric == "loss":
                improvement_delta = -raw_delta
                interpretation = (
                    "positive = lower MCOF loss"
                )
            else:
                improvement_delta = raw_delta
                interpretation = (
                    "positive = higher MCOF metric"
                )

            ci_low, ci_high = bootstrap_mean_ci(
                improvement_delta
            )

            tolerance = 1e-12

            wins = int(
                (improvement_delta > tolerance).sum()
            )
            ties = int(
                (
                    np.abs(improvement_delta)
                    <= tolerance
                ).sum()
            )
            losses = int(
                (improvement_delta < -tolerance).sum()
            )

            paired_rows.append(
                {
                    "dataset": dataset,
                    "comparison": (
                        f"MCOF vs {ablation_model}"
                    ),
                    "ablation_model": ablation_model,
                    "metric": metric,
                    "n_pairs": len(improvement_delta),
                    "mcof_mean": float(
                        paired[mcof_column].mean()
                    ),
                    "ablation_mean": float(
                        paired[ablation_column].mean()
                    ),
                    "mean_improvement_delta": float(
                        improvement_delta.mean()
                    ),
                    "sd_improvement_delta": float(
                        improvement_delta.std(ddof=1)
                    ),
                    "bootstrap_ci_2.5": ci_low,
                    "bootstrap_ci_97.5": ci_high,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "wilcoxon_p": paired_wilcoxon_p(
                        improvement_delta
                    ),
                    "interpretation": interpretation,
                }
            )


paired_df = pd.DataFrame(paired_rows)

paired_output = Path(
    f"{OUTPUT_PREFIX}_paired_comparisons.csv"
)
paired_df.to_csv(paired_output, index=False)


# ---------------------------------------------------------
# Concise text report
# ---------------------------------------------------------
report_output = Path(
    f"{OUTPUT_PREFIX}_report.txt"
)

with report_output.open(
    "w",
    encoding="utf-8",
) as report:
    report.write(
        "Clean MCOF ablation summary\n"
    )
    report.write("=" * 80 + "\n\n")

    report.write(
        "Models:\n"
        "  MCOF = mcof_se\n"
        "  MCOF w/o channel attention = mcof_no_se\n"
        "  MCOF w/o convolution = mcof_no_conv\n\n"
    )

    report.write(
        "Primary performance table\n"
    )
    report.write("-" * 80 + "\n")
    report.write(
        primary_df.to_string(index=False)
    )
    report.write("\n\n")

    report.write(
        "Paired primary-metric comparisons\n"
    )
    report.write("-" * 80 + "\n")

    selected = paired_df[
        paired_df.apply(
            lambda row: (
                row["metric"]
                in PRIMARY_METRICS[row["dataset"]]
            ),
            axis=1,
        )
    ]

    report.write(
        selected[
            [
                "dataset",
                "ablation_model",
                "metric",
                "mean_improvement_delta",
                "bootstrap_ci_2.5",
                "bootstrap_ci_97.5",
                "wins",
                "ties",
                "losses",
                "wilcoxon_p",
            ]
        ].to_string(index=False)
    )
    report.write("\n")


print("Saved:")
print(repeat_output)
print(summary_long_output)
print(summary_wide_output)
print(primary_output)
print(paired_output)
print(report_output)

print()
print(
    "Dataset-model combinations:",
    summary_df[
        [
            "dataset",
            "manuscript_model",
        ]
    ].drop_duplicates().shape[0],
)
print("Expected: 12 = 4 datasets × 3 models")

print()
print("Primary table:")
print(primary_df.to_string(index=False))
