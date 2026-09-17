
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

    matrices = {}
    source_rows = []
    all_values = []

    for dataset in DATASETS:
        dataset_dir = stability_root / dataset
        path = locate_file(
            dataset_dir,
            [
                f"{dataset}_repeat_rank_spearman.csv",
                "*repeat_rank_spearman*.csv",
            ],
        )
        matrix = read_pairwise_matrix(path)
        matrices[dataset] = matrix

        off_diagonal = upper_triangle_values(matrix)
        all_values.extend(off_diagonal.tolist())

        for row_label in matrix.index:
            for column_label in matrix.columns:
                source_rows.append(
                    {
                        "dataset": dataset,
                        "repeat_row": row_label,
                        "repeat_column": column_label,
                        "spearman": float(
                            matrix.loc[row_label, column_label]
                        ),
                        "source_file": str(path),
                    }
                )

    vmin = -1.0 if min(all_values) < 0 else 0.0
    vmax = 1.0

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 8.6),
        constrained_layout=True,
    )

    image = None

    for panel_index, dataset in enumerate(DATASETS):
        ax = axes.flat[panel_index]
        matrix = matrices[dataset]

        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_xticklabels(
            matrix.columns,
            rotation=45,
            ha="right",
            fontsize=8.5,
        )
        ax.set_yticklabels(
            matrix.index,
            fontsize=8.5,
        )
        ax.set_title(dataset)
        ax.text(
            -0.16,
            1.08,
            f"({chr(97 + panel_index)})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = float(
                    matrix.iloc[row_index, column_index]
                )
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                )

    if image is not None:
        colorbar = fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.84,
            pad=0.03,
        )
        colorbar.set_label("Spearman rank correlation")

    pdf_path = output_root / "FigS2_IG_rank_spearman_heatmaps.pdf"
    png_path = output_root / "FigS2_IG_rank_spearman_heatmaps.png"
    source_path = output_root / "FigS2_IG_rank_spearman_source_data.csv"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(source_rows).to_csv(
        source_path,
        index=False,
    )

    print("Generated:")
    print(pdf_path)
    print(png_path)
    print(source_path)


if __name__ == "__main__":
    main()
