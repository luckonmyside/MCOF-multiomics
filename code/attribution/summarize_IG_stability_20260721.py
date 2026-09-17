from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize IG feature-ranking stability across repeated experiments."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--n_repeats", type=int, default=5)
    return parser.parse_args()


def infer_omics(feature: str) -> str:
    name = str(feature).lower()
    return "miRNA" if name.startswith("hsa-") else "mRNA"


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else np.nan


def main() -> None:
    args = parse_args()

    dataset = args.dataset.upper()
    input_dir = Path(args.input_root) / dataset
    output_dir = Path(args.output_root) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    reference_features: set[str] | None = None
    expected_n: int | None = None

    for repeat_index in range(1, args.n_repeats + 1):
        repeat_name = f"repeat_{repeat_index}"
        input_file = (
            input_dir
            / repeat_name
            / "biomarker_importance_overall.csv"
        )

        if not input_file.is_file():
            raise FileNotFoundError(f"Missing file: {input_file}")

        df = pd.read_csv(input_file)

        required_columns = {"feature", "importance"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"{input_file} must contain columns: "
                f"{sorted(required_columns)}"
            )

        if df["feature"].isna().any():
            raise ValueError(f"Missing feature names in {input_file}")

        if df["feature"].duplicated().any():
            duplicated = df.loc[
                df["feature"].duplicated(), "feature"
            ].head().tolist()
            raise ValueError(
                f"Duplicated features in {input_file}: {duplicated}"
            )

        internal_names = df["feature"].astype(str).str.match(
            r"^omics\d+_f\d+$"
        )
        if internal_names.any():
            examples = df.loc[
                internal_names, "feature"
            ].head().tolist()
            raise ValueError(
                f"Internal placeholder names remain in {input_file}: "
                f"{examples}"
            )

        df = (
            df[["feature", "importance"]]
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        df["rank"] = np.arange(1, len(df) + 1)
        df["repeat"] = repeat_name

        feature_set = set(df["feature"].astype(str))

        if reference_features is None:
            reference_features = feature_set
            expected_n = len(df)
        else:
            if len(df) != expected_n:
                raise ValueError(
                    f"Feature count mismatch in {input_file}: "
                    f"expected {expected_n}, found {len(df)}"
                )

            if feature_set != reference_features:
                missing = sorted(reference_features - feature_set)[:10]
                extra = sorted(feature_set - reference_features)[:10]
                raise ValueError(
                    f"Feature set mismatch in {input_file}. "
                    f"Missing examples: {missing}; "
                    f"extra examples: {extra}"
                )

        frames.append(df)
        print(
            f"Loaded {repeat_name}: "
            f"{len(df)} features"
        )

    combined = pd.concat(frames, ignore_index=True)

    combined_out = (
        output_dir
        / f"{dataset}_IG_all_repeats_long.csv"
    )
    combined.to_csv(combined_out, index=False)

    summary = (
        combined
        .groupby("feature", as_index=False)
        .agg(
            mean_abs_IG=("importance", "mean"),
            sd_abs_IG=("importance", "std"),
            median_abs_IG=("importance", "median"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            sd_rank=("rank", "std"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
            n_repeats=("repeat", "nunique"),
        )
    )

    for k in (50, 100, 200):
        top = combined.loc[
            combined["rank"] <= k,
            ["feature", "repeat"],
        ]

        counts = (
            top.groupby("feature")
            .size()
            .rename(f"top{k}_count")
            .reset_index()
        )

        summary = summary.merge(
            counts,
            on="feature",
            how="left",
        )

        summary[f"top{k}_count"] = (
            summary[f"top{k}_count"]
            .fillna(0)
            .astype(int)
        )
        summary[f"top{k}_frequency"] = (
            summary[f"top{k}_count"]
            / args.n_repeats
        )

    summary["omics"] = summary["feature"].map(infer_omics)

    # Stability first, attribution magnitude second.
    summary = summary.sort_values(
        [
            "top50_count",
            "top100_count",
            "mean_rank",
            "mean_abs_IG",
        ],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    summary.insert(0, "stability_rank", np.arange(1, len(summary) + 1))

    summary_out = (
        output_dir
        / f"{dataset}_IG_stability_all_features.csv"
    )
    summary.to_csv(summary_out, index=False)

    summary.head(50).to_csv(
        output_dir / f"{dataset}_stable_top50.csv",
        index=False,
    )
    summary.head(100).to_csv(
        output_dir / f"{dataset}_stable_top100.csv",
        index=False,
    )

    # Spearman correlation of full feature rankings.
    rank_wide = combined.pivot(
        index="feature",
        columns="repeat",
        values="rank",
    )
    spearman = rank_wide.corr(method="spearman")
    spearman.to_csv(
        output_dir
        / f"{dataset}_repeat_rank_spearman.csv"
    )

    # Jaccard overlap matrices for top 50, 100, and 200.
    repeat_names = [
        f"repeat_{i}"
        for i in range(1, args.n_repeats + 1)
    ]

    for k in (50, 100, 200):
        feature_sets = {
            repeat: set(
                combined.loc[
                    (combined["repeat"] == repeat)
                    & (combined["rank"] <= k),
                    "feature",
                ].astype(str)
            )
            for repeat in repeat_names
        }

        matrix = pd.DataFrame(
            index=repeat_names,
            columns=repeat_names,
            dtype=float,
        )

        for r1 in repeat_names:
            for r2 in repeat_names:
                matrix.loc[r1, r2] = jaccard(
                    feature_sets[r1],
                    feature_sets[r2],
                )

        matrix.to_csv(
            output_dir
            / f"{dataset}_top{k}_jaccard.csv"
        )

    pairwise_rho = [
        spearman.loc[a, b]
        for a, b in combinations(spearman.columns, 2)
    ]

    print()
    print("Finished successfully.")
    print(f"Dataset: {dataset}")
    print(f"Features per repeat: {expected_n}")
    print(f"Combined rows: {len(combined)}")
    print(
        "Median pairwise Spearman rank correlation: "
        f"{np.median(pairwise_rho):.4f}"
    )
    print(f"Saved to: {output_dir}")
    print()
    print("Top 20 stable features:")
    columns = [
        "stability_rank",
        "feature",
        "omics",
        "mean_abs_IG",
        "mean_rank",
        "top50_count",
        "top50_frequency",
        "top100_frequency",
        "top200_frequency",
    ]
    print(summary.loc[:, columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
