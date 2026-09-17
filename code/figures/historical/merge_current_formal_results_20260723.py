from pathlib import Path

import numpy as np
import pandas as pd


MCOF_ROOT = Path(
    "/path/to/mcof-workspace/"
    "MCOFv2_runs/formal_main_20260429_191748"
)

ML_FILE = Path(
    "/path/to/mcof-workspace/"
    "MCOF_baselines_20260722/ML/"
    "ML_repeat_metrics.csv"
)

DIABLO_FILE = Path(
    "/path/to/mcof-workspace/"
    "MCOF_baselines_20260722/"
    "DIABLO_final_completed_20260723/"
    "DIABLO_repeat_metrics.csv"
)

OUTPUT_ROOT = Path(
    "/path/to/mcof-workspace/"
    "MCOF_baselines_20260722/"
    "combined_current_20260723"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


DATASET_ORDER = ["BRCA", "STAD", "ROSMAP", "SCZ"]
MODEL_ORDER = ["DIABLO", "KNN", "SVM", "NB", "RF", "MCOF"]

EXPECTED_METRICS = {
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

ALIASES = {
    "repeat": [
        "repeat",
        "repeat_id",
        "external_repeat",
        "run",
        "seed",
    ],
    "acc": [
        "acc",
        "accuracy",
        "test_acc",
        "test_accuracy",
    ],
    "f1": [
        "f1",
        "test_f1",
        "f1_score",
    ],
    "auc": [
        "auc",
        "test_auc",
        "roc_auc",
        "auc_roc",
    ],
    "mcc": [
        "mcc",
        "test_mcc",
    ],
    "f1_macro": [
        "f1_macro",
        "macro_f1",
        "test_f1_macro",
        "test_macro_f1",
    ],
    "f1_weighted": [
        "f1_weighted",
        "weighted_f1",
        "test_f1_weighted",
        "test_weighted_f1",
    ],
    "auc_macro_ovr": [
        "auc_macro_ovr",
        "macro_auc",
        "auc_macro",
        "test_auc_macro_ovr",
        "test_macro_auc",
    ],
    "auc_weighted_ovr": [
        "auc_weighted_ovr",
        "weighted_auc",
        "auc_weighted",
        "test_auc_weighted_ovr",
        "test_weighted_auc",
    ],
}


def rename_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    lower_to_original = {
        str(column).lower(): column
        for column in df.columns
    }

    rename_map = {}

    for target, candidates in ALIASES.items():
        if target in df.columns:
            continue

        for candidate in candidates:
            if candidate.lower() in lower_to_original:
                rename_map[
                    lower_to_original[candidate.lower()]
                ] = target
                break

    return df.rename(columns=rename_map)


def normalize_repeat_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "repeat" not in df.columns:
        raise ValueError(
            f"No repeat column found. Columns: {list(df.columns)}"
        )

    repeat_text = df["repeat"].astype(str)

    extracted = repeat_text.str.extract(
        r"(\d+)",
        expand=False,
    )

    df["repeat"] = pd.to_numeric(
        extracted,
        errors="raise",
    ).astype(int)

    return df


def find_mcof_file(dataset: str) -> Path:
    exact = (
        MCOF_ROOT
        / dataset
        / "mcof_se"
        / "repeat_test_metrics.csv"
    )

    if exact.is_file():
        return exact

    candidates = []

    for path in MCOF_ROOT.rglob("repeat_test_metrics.csv"):
        parts_lower = [
            part.lower()
            for part in path.parts
        ]

        if (
            dataset.lower() in parts_lower
            and "mcof_se" in parts_lower
        ):
            candidates.append(path)

    if len(candidates) != 1:
        raise FileNotFoundError(
            f"{dataset}: expected one MCOF result file, "
            f"found {len(candidates)}: {candidates}"
        )

    return candidates[0]


def read_mcof_results() -> pd.DataFrame:
    frames = []

    for dataset in DATASET_ORDER:
        path = find_mcof_file(dataset)
        print(f"MCOF {dataset}: {path}")

        df = pd.read_csv(path)
        df = rename_metric_columns(df)
        df = normalize_repeat_column(df)

        df["dataset"] = dataset
        df["model"] = "MCOF"

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def normalize_external_results(
    path: Path,
    permitted_models: set[str],
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = rename_metric_columns(df)
    df = normalize_repeat_column(df)

    required = {"dataset", "model", "repeat"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {sorted(missing)}"
        )

    df["dataset"] = df["dataset"].astype(str)
    df["model"] = df["model"].astype(str)

    df = df[df["model"].isin(permitted_models)].copy()

    return df


mcof = read_mcof_results()

ml = normalize_external_results(
    ML_FILE,
    {"KNN", "SVM", "NB", "RF"},
)

diablo = normalize_external_results(
    DIABLO_FILE,
    {"DIABLO"},
)

all_results = pd.concat(
    [diablo, ml, mcof],
    ignore_index=True,
    sort=False,
)

keep_columns = [
    "dataset",
    "model",
    "repeat",
    "acc",
    "f1",
    "auc",
    "mcc",
    "f1_macro",
    "f1_weighted",
    "auc_macro_ovr",
    "auc_weighted_ovr",
]

for column in keep_columns:
    if column not in all_results.columns:
        all_results[column] = np.nan

all_results = all_results[keep_columns].copy()

all_results["dataset"] = pd.Categorical(
    all_results["dataset"],
    categories=DATASET_ORDER,
    ordered=True,
)

all_results["model"] = pd.Categorical(
    all_results["model"],
    categories=MODEL_ORDER,
    ordered=True,
)

all_results = all_results.sort_values(
    ["dataset", "model", "repeat"]
).reset_index(drop=True)


# Integrity checks
duplicates = all_results.duplicated(
    ["dataset", "model", "repeat"],
    keep=False,
)

if duplicates.any():
    print(
        all_results.loc[
            duplicates,
            ["dataset", "model", "repeat"],
        ].to_string(index=False)
    )
    raise ValueError("Duplicated dataset-model-repeat rows found.")

counts = (
    all_results.groupby(
        ["dataset", "model"],
        observed=True,
    )
    .size()
    .rename("n_repeats")
    .reset_index()
)

print("\nRun counts:")
print(counts.to_string(index=False))

expected_rows = (
    len(DATASET_ORDER)
    * len(MODEL_ORDER)
    * 5
)

assert len(all_results) == expected_rows, (
    f"Expected {expected_rows} rows, "
    f"found {len(all_results)}"
)

assert counts["n_repeats"].eq(5).all(), (
    "At least one dataset-model combination "
    "does not contain five repeats."
)

for dataset in DATASET_ORDER:
    group = all_results[
        all_results["dataset"].eq(dataset)
    ]

    for metric in EXPECTED_METRICS[dataset]:
        values = pd.to_numeric(
            group[metric],
            errors="coerce",
        )

        if values.isna().any():
            bad = group.loc[
                values.isna(),
                ["dataset", "model", "repeat", metric],
            ]

            print(bad.to_string(index=False))

            raise ValueError(
                f"{dataset}: missing values in {metric}"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{dataset}: non-finite values in {metric}"
            )


# Save repeat-level combined data
repeat_file = (
    OUTPUT_ROOT
    / "all_models_repeat_metrics_current.csv"
)

all_results.to_csv(
    repeat_file,
    index=False,
)


# Long summary
summary_rows = []

for dataset in DATASET_ORDER:
    for model in MODEL_ORDER:
        group = all_results[
            all_results["dataset"].eq(dataset)
            & all_results["model"].eq(model)
        ]

        for metric in EXPECTED_METRICS[dataset]:
            values = pd.to_numeric(
                group[metric],
                errors="raise",
            )

            mean_value = values.mean()
            sd_value = values.std(ddof=1)

            summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "n_repeats": len(values),
                    "mean": mean_value,
                    "std": sd_value,
                    "mean_sd_3dp": (
                        f"{mean_value:.3f} ± "
                        f"{sd_value:.3f}"
                    ),
                }
            )

summary_long = pd.DataFrame(summary_rows)

summary_long_file = (
    OUTPUT_ROOT
    / "all_models_summary_long_current.csv"
)

summary_long.to_csv(
    summary_long_file,
    index=False,
)


# Manuscript-style table: dataset/metric rows × models
table_rows = []

for dataset in DATASET_ORDER:
    for metric in EXPECTED_METRICS[dataset]:
        row = {
            "dataset": dataset,
            "metric": metric,
        }

        metric_subset = summary_long[
            summary_long["dataset"].eq(dataset)
            & summary_long["metric"].eq(metric)
        ]

        best_mean = metric_subset["mean"].max()

        for model in MODEL_ORDER:
            selected = metric_subset[
                metric_subset["model"].eq(model)
            ]

            assert len(selected) == 1

            row[model] = selected.iloc[0][
                "mean_sd_3dp"
            ]

            row[f"{model}_is_best"] = bool(
                np.isclose(
                    selected.iloc[0]["mean"],
                    best_mean,
                )
            )

        table_rows.append(row)

publication_table = pd.DataFrame(table_rows)

table_file = (
    OUTPUT_ROOT
    / "Table2_current_without_MOGONET.csv"
)

publication_table.to_csv(
    table_file,
    index=False,
)


# Paired deltas: positive means MCOF is higher
delta_rows = []

for dataset in DATASET_ORDER:
    for metric in EXPECTED_METRICS[dataset]:
        mcof_values = (
            all_results[
                all_results["dataset"].eq(dataset)
                & all_results["model"].eq("MCOF")
            ][["repeat", metric]]
            .rename(columns={metric: "mcof"})
        )

        for baseline in MODEL_ORDER:
            if baseline == "MCOF":
                continue

            baseline_values = (
                all_results[
                    all_results["dataset"].eq(dataset)
                    & all_results["model"].eq(baseline)
                ][["repeat", metric]]
                .rename(columns={metric: "baseline"})
            )

            paired = mcof_values.merge(
                baseline_values,
                on="repeat",
                how="inner",
                validate="one_to_one",
            )

            assert len(paired) == 5

            delta = (
                paired["mcof"]
                - paired["baseline"]
            )

            tolerance = 1e-12

            delta_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "baseline": baseline,
                    "n_pairs": len(delta),
                    "mean_delta_MCOF_minus_baseline": (
                        delta.mean()
                    ),
                    "sd_delta": delta.std(ddof=1),
                    "wins": int(
                        (delta > tolerance).sum()
                    ),
                    "ties": int(
                        (delta.abs() <= tolerance).sum()
                    ),
                    "losses": int(
                        (delta < -tolerance).sum()
                    ),
                }
            )

delta_table = pd.DataFrame(delta_rows)

delta_file = (
    OUTPUT_ROOT
    / "paired_delta_vs_MCOF_current.csv"
)

delta_table.to_csv(
    delta_file,
    index=False,
)


print("\nCombined formal results completed.")
print("Repeat-level:", repeat_file)
print("Summary:", summary_long_file)
print("Current Table 2:", table_file)
print("Paired deltas:", delta_file)

print("\nCurrent manuscript-style table:")
display_columns = (
    ["dataset", "metric"]
    + MODEL_ORDER
)
print(
    publication_table[
        display_columns
    ].to_string(index=False)
)
