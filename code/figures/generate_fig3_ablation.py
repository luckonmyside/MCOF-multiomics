from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "MCOF",
    "MCOF w/o channel attention",
    "MCOF w/o convolution",
]

MODEL_LABELS = {
    "MCOF": "MCOF",
    "MCOF w/o channel attention": "Without channel attention",
    "MCOF w/o convolution": "Without convolution",
}

DATASET_METRICS: Dict[str, List[Tuple[str, str]]] = {
    "BRCA": [
        ("acc", "ACC"),
        ("f1_macro", "Macro-F1"),
        ("f1_weighted", "Weighted-F1"),
        ("auc_macro_ovr", "Macro-AUC"),
    ],
    "STAD": [
        ("acc", "ACC"),
        ("f1_macro", "Macro-F1"),
        ("f1_weighted", "Weighted-F1"),
        ("auc_macro_ovr", "Macro-AUC"),
    ],
    "ROSMAP": [
        ("acc", "ACC"),
        ("f1", "F1"),
        ("auc", "AUC"),
        ("mcc", "MCC"),
    ],
    "SCZ": [
        ("acc", "ACC"),
        ("f1", "F1"),
        ("auc", "AUC"),
        ("mcc", "MCC"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_root", required=True)
    return parser.parse_args()


def parse_mean_sd(value: object) -> Tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan

    match = re.match(
        r"\s*([-+]?[0-9]*\.?[0-9]+)\s*±\s*"
        r"([-+]?[0-9]*\.?[0-9]+)\s*",
        str(value),
    )
    if not match:
        raise ValueError(f"Cannot parse mean ± SD value: {value}")

    return float(match.group(1)), float(match.group(2))


def prepare_data(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    records = []

    for _, row in df.iterrows():
        dataset = str(row["dataset"])
        model = str(row["model"])

        if dataset not in DATASET_METRICS:
            continue
        if model not in MODEL_ORDER:
            continue

        for metric, metric_label in DATASET_METRICS[dataset]:
            mean_value, sd_value = parse_mean_sd(row[metric])
            records.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "metric_label": metric_label,
                    "mean": mean_value,
                    "sd": sd_value,
                }
            )

    result = pd.DataFrame(records)

    expected = 4 * 3 * 4
    if len(result) != expected:
        raise ValueError(
            f"Expected {expected} dataset-model-metric rows, "
            f"found {len(result)}."
        )

    return result


def draw(
    plot_df: pd.DataFrame,
    datasets: List[str],
    filename: str,
    output_root: Path,
    y_limits: Tuple[float, float],
    figure_size: Tuple[float, float],
) -> None:
    categories = []
    metadata = []

    for dataset in datasets:
        for metric, metric_label in DATASET_METRICS[dataset]:
            categories.append(f"{dataset}\n{metric_label}")
            metadata.append((dataset, metric))

    x_positions = np.arange(len(categories))
    offsets = [-0.22, 0.0, 0.22]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=figure_size)

    for model, offset, marker in zip(
        MODEL_ORDER,
        offsets,
        markers,
    ):
        means = []
        sds = []

        for dataset, metric in metadata:
            selected = plot_df[
                plot_df["dataset"].eq(dataset)
                & plot_df["model"].eq(model)
                & plot_df["metric"].eq(metric)
            ]

            if len(selected) != 1:
                raise ValueError(
                    f"Expected one result for "
                    f"{dataset}/{model}/{metric}, found {len(selected)}."
                )

            means.append(float(selected.iloc[0]["mean"]))
            sds.append(float(selected.iloc[0]["sd"]))

        ax.errorbar(
            x_positions + offset,
            means,
            yerr=sds,
            fmt=marker,
            linestyle="none",
            capsize=3,
            markersize=6,
            linewidth=1.2,
            label=MODEL_LABELS[model],
        )

    for boundary in range(4, len(categories), 4):
        ax.axvline(
            boundary - 0.5,
            linestyle="--",
            linewidth=0.8,
            alpha=0.55,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Performance (mean ± SD)")
    ax.set_ylim(*y_limits)
    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.55,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()

    pdf_path = output_root / f"{filename}.pdf"
    png_path = output_root / f"{filename}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(pdf_path)
    print(png_path)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    plot_df = prepare_data(input_csv)
    plot_df.to_csv(
        output_root / "Fig3_ablation_source_data.csv",
        index=False,
    )

    draw(
        plot_df,
        datasets=["BRCA", "STAD"],
        filename="Fig3_MCOF_strict_ablation_BRCA_STAD",
        output_root=output_root,
        y_limits=(0.70, 1.01),
        figure_size=(10.5, 5.4),
    )

    draw(
        plot_df,
        datasets=["BRCA", "STAD", "ROSMAP", "SCZ"],
        filename="FigS1_MCOF_strict_ablation_all_datasets",
        output_root=output_root,
        y_limits=(0.35, 1.02),
        figure_size=(16.5, 5.8),
    )


if __name__ == "__main__":
    main()
