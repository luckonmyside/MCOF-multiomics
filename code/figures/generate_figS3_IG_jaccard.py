
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _clean_repeat_label(value: object) -> str:
    text = str(value).strip()
    digits = "".join(character for character in text if character.isdigit())
    return f"Repeat {int(digits)}" if digits else text


def read_pairwise_matrix(path: Path) -> pd.DataFrame:
    """
    Read a five-repeat pairwise matrix from either:
    1) a square CSV with row labels in the first column;
    2) a plain square numeric CSV;
    3) a long pairwise CSV containing two repeat columns and one value column.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)

    # Long-format detection.
    lower = {str(column).lower(): column for column in raw.columns}
    repeat_candidates = [
        column
        for column in raw.columns
        if "repeat" in str(column).lower()
        or str(column).lower() in {"run1", "run2", "i", "j"}
    ]

    numeric_candidates = [
        column
        for column in raw.columns
        if pd.api.types.is_numeric_dtype(raw[column])
        and column not in repeat_candidates
    ]

    if len(repeat_candidates) >= 2 and numeric_candidates:
        value_priority = [
            column
            for column in numeric_candidates
            if any(
                token in str(column).lower()
                for token in ["spearman", "jaccard", "cor", "similarity", "value"]
            )
        ]
        value_column = (
            value_priority[0]
            if value_priority
            else numeric_candidates[-1]
        )
        first_repeat = repeat_candidates[0]
        second_repeat = repeat_candidates[1]

        labels = sorted(
            {
                _clean_repeat_label(value)
                for value in pd.concat(
                    [raw[first_repeat], raw[second_repeat]],
                    ignore_index=True,
                )
            },
            key=lambda text: int("".join(filter(str.isdigit, text)) or 0),
        )

        matrix = pd.DataFrame(
            np.eye(len(labels), dtype=float),
            index=labels,
            columns=labels,
        )

        for _, row in raw.iterrows():
            first = _clean_repeat_label(row[first_repeat])
            second = _clean_repeat_label(row[second_repeat])
            value = float(row[value_column])
            matrix.loc[first, second] = value
            matrix.loc[second, first] = value

        return matrix

    # Try reading with the first column as a row index.
    indexed = pd.read_csv(path, index_col=0)
    indexed_numeric = indexed.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if (
        indexed_numeric.shape[0] == indexed_numeric.shape[1]
        and indexed_numeric.notna().all().all()
    ):
        indexed_numeric.index = [
            _clean_repeat_label(value)
            for value in indexed_numeric.index
        ]
        indexed_numeric.columns = [
            _clean_repeat_label(value)
            for value in indexed_numeric.columns
        ]
        return indexed_numeric

    # Plain square numeric CSV.
    plain = pd.read_csv(path, header=None)
    plain_numeric = plain.apply(pd.to_numeric, errors="coerce")

    if (
        plain_numeric.shape[0] == plain_numeric.shape[1]
        and plain_numeric.notna().all().all()
    ):
        labels = [
            f"Repeat {index}"
            for index in range(1, len(plain_numeric) + 1)
        ]
        plain_numeric.index = labels
        plain_numeric.columns = labels
        return plain_numeric

    raise ValueError(
        f"Could not recognize pairwise matrix structure in {path}. "
        f"Columns: {list(raw.columns)}"
    )


def upper_triangle_values(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    indices = np.triu_indices_from(values, k=1)
    result = values[indices]
    return result[np.isfinite(result)]


def locate_file(
    dataset_dir: Path,
    preferred_patterns: List[str],
) -> Path:
    for pattern in preferred_patterns:
        matches = sorted(dataset_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No matching file in {dataset_dir}. "
        f"Patterns: {preferred_patterns}"
    )

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASETS = ["BRCA", "STAD", "ROSMAP", "SCZ"]
TOP_K = [50, 100, 200]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability_root", required=True)
    parser.add_argument("--output_root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stability_root = Path(args.stability_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pairwise_rows = []
    summary_rows = []

    for dataset in DATASETS:
        dataset_dir = stability_root / dataset

        for top_k in TOP_K:
            path = locate_file(
                dataset_dir,
                [
                    f"{dataset}_top{top_k}_jaccard.csv",
                    f"*top{top_k}*jaccard*.csv",
                ],
            )
            matrix = read_pairwise_matrix(path)
            values = upper_triangle_values(matrix)

            if len(values) == 0:
                raise ValueError(
                    f"{dataset} top-{top_k}: no pairwise values found."
                )

            for pair_index, value in enumerate(values, start=1):
                pairwise_rows.append(
                    {
                        "dataset": dataset,
                        "top_k": top_k,
                        "pair_index": pair_index,
                        "jaccard": float(value),
                        "source_file": str(path),
                    }
                )

            summary_rows.append(
                {
                    "dataset": dataset,
                    "top_k": top_k,
                    "n_pairs": len(values),
                    "mean_jaccard": float(values.mean()),
                    "sd_jaccard": float(
                        values.std(ddof=1)
                    ),
                    "median_jaccard": float(
                        np.median(values)
                    ),
                    "min_jaccard": float(values.min()),
                    "max_jaccard": float(values.max()),
                }
            )

    pairwise = pd.DataFrame(pairwise_rows)
    summary = pd.DataFrame(summary_rows)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 8.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    deterministic_offsets = np.linspace(-0.12, 0.12, 10)

    for panel_index, dataset in enumerate(DATASETS):
        ax = axes.flat[panel_index]
        current_pairs = pairwise[
            pairwise["dataset"].eq(dataset)
        ]
        current_summary = summary[
            summary["dataset"].eq(dataset)
        ]

        for x_index, top_k in enumerate(TOP_K):
            values = (
                current_pairs[
                    current_pairs["top_k"].eq(top_k)
                ]["jaccard"]
                .to_numpy(dtype=float)
            )

            offsets = deterministic_offsets[: len(values)]
            ax.scatter(
                np.full(len(values), x_index) + offsets,
                values,
                s=18,
                alpha=0.65,
            )

            row = current_summary[
                current_summary["top_k"].eq(top_k)
            ].iloc[0]

            ax.errorbar(
                x_index,
                float(row["mean_jaccard"]),
                yerr=float(row["sd_jaccard"]),
                fmt="o",
                capsize=4,
                markersize=7,
                linewidth=1.4,
            )

        ax.set_xticks(range(len(TOP_K)))
        ax.set_xticklabels(
            [f"Top {value}" for value in TOP_K]
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_title(dataset)
        ax.grid(
            axis="y",
            linestyle=":",
            alpha=0.55,
        )
        ax.text(
            -0.16,
            1.08,
            f"({chr(97 + panel_index)})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    axes[0, 0].set_ylabel("Pairwise Jaccard similarity")
    axes[1, 0].set_ylabel("Pairwise Jaccard similarity")

    pdf_path = output_root / "FigS3_IG_topk_jaccard.pdf"
    png_path = output_root / "FigS3_IG_topk_jaccard.png"
    pairwise_path = output_root / "FigS3_IG_topk_jaccard_pairwise.csv"
    summary_path = output_root / "FigS3_IG_topk_jaccard_summary.csv"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    pairwise.to_csv(pairwise_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Generated:")
    print(pdf_path)
    print(png_path)
    print(pairwise_path)
    print(summary_path)


if __name__ == "__main__":
    main()
