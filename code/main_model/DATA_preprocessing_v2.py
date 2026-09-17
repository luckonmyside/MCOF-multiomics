from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe preprocessing entry point. "
            "This script only harmonizes raw omics tables and labels into 1_all.csv / 2_all.csv / labels_all.csv. "
            "Imputation, scaling and feature selection should be done inside training folds, not here."
        )
    )
    parser.add_argument("--omics", nargs="+", required=True, help="Paths to omics CSV files")
    parser.add_argument("--labels", required=True, help="Path to label CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save 1_all.csv, 2_all.csv, ..., labels_all.csv")
    parser.add_argument("--sample_col", default=None, help="Column containing sample IDs. If omitted, the first column is used.")
    parser.add_argument("--label_col", required=True, help="Column in labels file containing the class label")
    parser.add_argument("--transpose_omics", action="store_true", help="Transpose each omics table after loading")
    parser.add_argument("--drop_duplicate_features", action="store_true", help="Merge duplicated feature names by mean")
    return parser.parse_args()



def _load_table(path: str, sample_col: str | None, transpose: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if transpose:
        df = df.transpose().reset_index()
    if sample_col is None:
        sample_col = df.columns[0]
    df = df.rename(columns={sample_col: "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str)
    return df



def _collapse_duplicate_features(df: pd.DataFrame) -> pd.DataFrame:
    sample_ids = df[["sample_id"]].copy()
    features = df.drop(columns=["sample_id"])
    features = features.groupby(features.columns, axis=1).mean()
    return pd.concat([sample_ids, features], axis=1)



def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    omics_tables: List[pd.DataFrame] = []
    shared_samples = None
    for path in args.omics:
        table = _load_table(path, sample_col=args.sample_col, transpose=args.transpose_omics)
        if args.drop_duplicate_features:
            table = _collapse_duplicate_features(table)
        omics_tables.append(table)
        sample_set = set(table["sample_id"])
        shared_samples = sample_set if shared_samples is None else shared_samples & sample_set

    labels = pd.read_csv(args.labels)
    if args.sample_col is None:
        sample_col = labels.columns[0]
    else:
        sample_col = args.sample_col
    labels = labels.rename(columns={sample_col: "sample_id", args.label_col: "label"})[["sample_id", "label"]]
    labels["sample_id"] = labels["sample_id"].astype(str)
    shared_samples = shared_samples & set(labels["sample_id"])
    shared_samples = sorted(shared_samples)

    labels = labels.set_index("sample_id").loc[shared_samples].reset_index()
    label_codes, _ = pd.factorize(labels["label"], sort=True)
    pd.DataFrame(label_codes).to_csv(output_dir / "labels_all.csv", index=False, header=False)

    for idx, table in enumerate(omics_tables, start=1):
        table = table.set_index("sample_id").loc[shared_samples]
        feat_names = table.columns.astype(str).tolist()
        table.to_csv(output_dir / f"{idx}_all.csv", index=False, header=False)
        pd.DataFrame(feat_names).to_csv(output_dir / f"{idx}_featname.csv", index=False, header=False)

    print(f"Saved harmonized all-sample matrices to: {output_dir.resolve()}")
    print("Important: run feature selection / imputation / scaling inside the training pipeline, not before splitting.")


if __name__ == "__main__":
    main()
