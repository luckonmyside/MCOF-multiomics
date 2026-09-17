from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

from common import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a transparent feature-exclusion sensitivity dataset from a "
            "prespecified identifier list. This utility does not decide which features "
            "are scientifically appropriate to exclude."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--block_index", required=True, type=int)
    parser.add_argument("--exclude_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--case_insensitive", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize(value: str, case_insensitive: bool) -> str:
    text = str(value).strip().strip('"').strip("'")
    text = text.split("|")[0].strip()
    text = re.sub(r"\.\d+$", "", text)
    return text.upper() if case_insensitive else text


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} exists; use --force")
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    excluded = {
        normalize(line, args.case_insensitive)
        for line in Path(args.exclude_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not excluded:
        raise ValueError("The exclusion file contains no identifiers")

    block = 1
    report_rows = []
    while (data_dir / f"{block}_all.csv").is_file():
        matrix_path = data_dir / f"{block}_all.csv"
        names_path = data_dir / f"{block}_featname.csv"
        if not names_path.is_file():
            raise FileNotFoundError(names_path)
        if block == args.block_index:
            matrix = pd.read_csv(matrix_path, header=None)
            names = pd.read_csv(names_path, header=None).iloc[:, 0].astype(str)
            normalized = names.map(lambda value: normalize(value, args.case_insensitive))
            remove = normalized.isin(excluded)
            for index in names.index[remove]:
                report_rows.append(
                    {
                        "block_index": block,
                        "column_index": int(index),
                        "feature": names.loc[index],
                        "normalized_identifier": normalized.loc[index],
                    }
                )
            matrix.loc[:, ~remove.to_numpy()].to_csv(
                output_dir / f"{block}_all.csv", header=False, index=False
            )
            names.loc[~remove].to_csv(
                output_dir / f"{block}_featname.csv", header=False, index=False
            )
        else:
            (output_dir / f"{block}_all.csv").symlink_to(matrix_path.resolve())
            (output_dir / f"{block}_featname.csv").symlink_to(names_path.resolve())
        block += 1

    if block == 1:
        raise FileNotFoundError(f"No omics blocks found in {data_dir}")
    if args.block_index >= block:
        raise IndexError(f"Requested block {args.block_index}, but only {block-1} blocks exist")
    (output_dir / "labels_all.csv").symlink_to((data_dir / "labels_all.csv").resolve())

    report = pd.DataFrame(report_rows)
    report.to_csv(output_dir / "excluded_features.csv", index=False)
    write_json(
        {
            "source_data_dir": str(data_dir.resolve()),
            "block_index": args.block_index,
            "exclude_file": str(Path(args.exclude_file).resolve()),
            "case_insensitive": args.case_insensitive,
            "n_requested_identifiers": len(excluded),
            "n_matched_features": len(report),
        },
        output_dir / "feature_exclusion_manifest.json",
    )
    print(f"Created {output_dir}; removed {len(report)} matched features")


if __name__ == "__main__":
    main()
