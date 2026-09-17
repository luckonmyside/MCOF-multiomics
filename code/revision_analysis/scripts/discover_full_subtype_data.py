from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from common import ensure_dir

TARGET_TERMS = [
    "normal-like",
    "normal like",
    "epstein-barr",
    "epstein barr",
    "ebv",
    "basal-like",
    "luminal a",
    "luminal b",
    "her2-enriched",
    "chromosomal instability",
    "genomically stable",
    "microsatellite instability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search project trees for full-subtype BRCA/STAD label files and raw/high-dimensional "
            "data candidates that were not found by filename-only searches."
        )
    )
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--max_file_mb", type=float, default=50.0)
    parser.add_argument("--max_candidates", type=int, default=5000)
    return parser.parse_args()


def candidate_files(roots: Iterable[Path], max_bytes: int) -> Iterable[Path]:
    extensions = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"}
    keywords = re.compile(r"label|phenotype|subtype|clinical|class|group|pam50|brca|stad|ebv", re.I)
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for directory, _, filenames in os.walk(root):
            for filename in filenames:
                path = Path(directory) / filename
                if path.suffix.lower() not in extensions:
                    continue
                if not keywords.search(str(path)):
                    continue
                try:
                    if path.stat().st_size > max_bytes:
                        continue
                except OSError:
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield path


def read_candidate(path: Path) -> pd.DataFrame | None:
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, header=None, nrows=5000, dtype=str)
        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(path, header=None, nrows=5000, dtype=str, sep=None, engine="python")
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, header=None, nrows=5000, dtype=str)
        if suffix == ".json":
            return pd.read_json(path, dtype=False)
    except Exception:
        return None
    return None


def main() -> None:
    args = parse_args()
    roots = [Path(value) for value in args.roots]
    output_root = ensure_dir(args.output_root)
    rows: List[Dict[str, object]] = []
    inspected = 0

    for path in candidate_files(roots, int(args.max_file_mb * 1024 * 1024)):
        inspected += 1
        if inspected > args.max_candidates:
            break
        frame = read_candidate(path)
        if frame is None or frame.empty:
            continue
        values = frame.fillna("").astype(str)
        flattened = values.to_numpy().ravel()
        joined = "\n".join(flattened[: min(len(flattened), 20000)]).lower()
        matched_terms = sorted({term for term in TARGET_TERMS if term in joined})

        likely_label_columns = []
        unique_counts = []
        for column in values.columns:
            series = values[column].str.strip()
            nonempty = series[series.ne("")]
            unique = int(nonempty.nunique())
            if 2 <= unique <= 20:
                likely_label_columns.append(str(column))
                unique_counts.append(unique)

        n_rows_observed = len(values)
        likely_full_brca = bool(
            any(term in matched_terms for term in ["normal-like", "normal like"])
            or (1000 <= n_rows_observed <= 1100 and any(value == 5 for value in unique_counts))
        )
        likely_full_stad = bool(
            any(term in matched_terms for term in ["epstein-barr", "epstein barr", "ebv"])
            or (220 <= n_rows_observed <= 270 and any(value == 4 for value in unique_counts))
        )

        if matched_terms or likely_full_brca or likely_full_stad:
            rows.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "rows_read": n_rows_observed,
                    "columns_read": values.shape[1],
                    "matched_terms": ";".join(matched_terms),
                    "likely_label_columns": ";".join(likely_label_columns),
                    "unique_counts_for_likely_label_columns": ";".join(map(str, unique_counts)),
                    "likely_full_BRCA": likely_full_brca,
                    "likely_full_STAD": likely_full_stad,
                }
            )

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(
            ["likely_full_BRCA", "likely_full_STAD", "matched_terms"],
            ascending=[False, False, False],
        )
    report.to_csv(output_root / "full_subtype_candidate_files.csv", index=False)

    # Filename-only inventory of large/raw-looking objects, without opening them.
    filename_rows = []
    filename_pattern = re.compile(
        r"pam50|normal.?like|epstein|\bebv\b|brca|stad|raw|clinical|phenotype|subtype|label",
        re.I,
    )
    for root in roots:
        if not root.exists():
            continue
        for directory, _, filenames in os.walk(root):
            for filename in filenames:
                if not filename_pattern.search(filename):
                    continue
                path = Path(directory) / filename
                try:
                    filename_rows.append(
                        {
                            "path": str(path.resolve()),
                            "size_bytes": path.stat().st_size,
                            "suffix": path.suffix.lower(),
                        }
                    )
                except OSError:
                    continue
    pd.DataFrame(filename_rows).drop_duplicates("path").to_csv(
        output_root / "keyword_filename_inventory.csv", index=False
    )

    summary = [
        f"Roots searched: {', '.join(map(str, roots))}",
        f"Tabular candidates inspected: {inspected}",
        f"Files with subtype evidence: {len(report)}",
        "",
        "A zero-result report means that full-class source files were not found under the supplied roots;",
        "it does not prove that the data never existed in an external repository or an unsearched path.",
    ]
    (output_root / "full_subtype_discovery_summary.txt").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    print("\n".join(summary))
    print(f"Report: {output_root / 'full_subtype_candidate_files.csv'}")


if __name__ == "__main__":
    main()
