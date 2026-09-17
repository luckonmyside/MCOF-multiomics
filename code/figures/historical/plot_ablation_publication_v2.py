from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

INPUT_FILE = Path(
    "/path/to/mcof-workspace/"
    "MCOF_clean_ablation_20260721_primary_table.csv"
)

OUTPUT_DIR = Path(
    "/path/to/mcof-workspace/"
    "figures_20260721_v2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Plot configuration
# =========================================================

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

MARKERS = {
    "MCOF": "o",
    "MCOF w/o channel attention": "s",
    "MCOF w/o convolution": "^",
}

DATASET_METRICS = {
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


# Journal-scale typography.
# Liberation Sans is usually available on Linux;
# Arial and DejaVu Sans are fallbacks.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Liberation Sans",
            "Arial",
            "DejaVu Sans",
        ],
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def parse_mean_sd(value: object) -> tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan

    match = re.match(
        r"\s*([0-9.]+)\s*±\s*([0-9.]+)\s*",
        str(value),
    )

    if match is None:
        raise ValueError(
            f"Cannot parse mean ± SD value: {value}"
        )

    return (
        float(match.group(1)),
        float(match.group(2)),
    )


# =========================================================
# Read plotting data
# =========================================================

source_df = pd.read_csv(INPUT_FILE)

records: list[dict] = []

for _, row in source_df.iterrows():
    dataset = str(row["dataset"]).strip()
    model = str(row["model"]).strip()

    for metric, metric_label in DATASET_METRICS[dataset]:
        mean, sd = parse_mean_sd(row[metric])

        records.append(
            {
                "dataset": dataset,
                "model": model,
                "metric": metric,
                "metric_label": metric_label,
                "mean": mean,
                "sd": sd,
            }
        )

plot_df = pd.DataFrame(records)

plot_df.to_csv(
    OUTPUT_DIR / "ablation_plot_data.csv",
    index=False,
)


def get_result(
    dataset: str,
    model: str,
    metric: str,
) -> tuple[float, float]:
    selected = plot_df[
        (plot_df["dataset"] == dataset)
        & (plot_df["model"] == model)
        & (plot_df["metric"] == metric)
    ]

    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one result for "
            f"{dataset} / {model} / {metric}; "
            f"found {len(selected)}"
        )

    return (
        float(selected.iloc[0]["mean"]),
        float(selected.iloc[0]["sd"]),
    )


def save_figure(
    fig: plt.Figure,
    filename: str,
) -> None:
    for extension in ["pdf", "svg"]:
        path = OUTPUT_DIR / f"{filename}.{extension}"
        fig.savefig(
            path,
            bbox_inches="tight",
            facecolor="white",
        )
        print("Saved:", path)

    png_path = OUTPUT_DIR / f"{filename}.png"
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    print("Saved:", png_path)


# =========================================================
# Main Figure 3: BRCA and STAD
# =========================================================

def draw_cancer_ablation() -> None:
    datasets = ["BRCA", "STAD"]

    metadata: list[tuple[str, str]] = []
    metric_labels: list[str] = []

    for dataset in datasets:
        for metric, metric_label in DATASET_METRICS[dataset]:
            metadata.append((dataset, metric))
            metric_labels.append(metric_label)

    x = np.arange(len(metadata), dtype=float)

    model_offsets = {
        "MCOF": -0.20,
        "MCOF w/o channel attention": 0.00,
        "MCOF w/o convolution": 0.20,
    }

    fig, ax = plt.subplots(
        figsize=(7.2, 3.9)
    )

    for plot_index, model in enumerate(MODEL_ORDER):
        means = []
        sds = []

        for dataset, metric in metadata:
            mean, sd = get_result(
                dataset,
                model,
                metric,
            )
            means.append(mean)
            sds.append(sd)

        marker_options = {}

        # Complete MCOF is filled; ablations are hollow.
        if model != "MCOF":
            marker_options["markerfacecolor"] = "white"
            marker_options["markeredgewidth"] = 1.1

        ax.errorbar(
            x + model_offsets[model],
            means,
            yerr=sds,
            fmt=MARKERS[model],
            linestyle="none",
            capsize=2.6,
            capthick=1.0,
            elinewidth=1.0,
            markersize=5.2,
            label=MODEL_LABELS[model],
            zorder=3 - plot_index,
            **marker_options,
        )

    # Neutral separator between BRCA and STAD.
    ax.axvline(
        3.5,
        linewidth=0.8,
        linestyle="--",
        color="0.78",
        zorder=0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.tick_params(
        axis="x",
        length=0,
    )

    ax.set_ylabel("Performance (mean ± SD)")

    # Zoomed range is acceptable for dot/error-bar figures
    # and remains explicit on the axis.
    ax.set_ylim(0.70, 1.005)
    ax.set_yticks(
        np.arange(0.70, 1.001, 0.05)
    )
    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.2f")
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        color="0.82",
        zorder=0,
    )

    # Dataset labels centered under their metric blocks.
    ax.text(
        1.5,
        -0.16,
        "BRCA",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontweight="bold",
    )

    ax.text(
        5.5,
        -0.16,
        "STAD",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontweight="bold",
    )

    # No in-panel title; the manuscript caption provides it.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=0.24,
        top=0.87,
    )

    save_figure(
        fig,
        "Fig3_modulewise_ablation_BRCA_STAD_v2",
    )

    plt.close(fig)


# =========================================================
# Supplementary Figure S2:
# vertical forest plot across four datasets
# =========================================================

def draw_all_dataset_ablation() -> None:
    datasets = [
        "BRCA",
        "STAD",
        "ROSMAP",
        "SCZ",
    ]

    metadata: list[tuple[str, str]] = []
    row_labels: list[str] = []

    for dataset in datasets:
        for metric, metric_label in DATASET_METRICS[dataset]:
            metadata.append((dataset, metric))
            row_labels.append(
                f"{dataset}   {metric_label}"
            )

    y = np.arange(len(metadata), dtype=float)

    model_offsets = {
        "MCOF": -0.20,
        "MCOF w/o channel attention": 0.00,
        "MCOF w/o convolution": 0.20,
    }

    fig, ax = plt.subplots(
        figsize=(7.2, 7.0)
    )

    for plot_index, model in enumerate(MODEL_ORDER):
        means = []
        sds = []

        for dataset, metric in metadata:
            mean, sd = get_result(
                dataset,
                model,
                metric,
            )
            means.append(mean)
            sds.append(sd)

        marker_options = {}

        if model != "MCOF":
            marker_options["markerfacecolor"] = "white"
            marker_options["markeredgewidth"] = 1.1

        ax.errorbar(
            means,
            y + model_offsets[model],
            xerr=sds,
            fmt=MARKERS[model],
            linestyle="none",
            capsize=2.4,
            capthick=1.0,
            elinewidth=1.0,
            markersize=5.0,
            label=MODEL_LABELS[model],
            zorder=3 - plot_index,
            **marker_options,
        )

    # Dataset-group separators.
    for boundary in [3.5, 7.5, 11.5]:
        ax.axhline(
            boundary,
            linewidth=0.8,
            linestyle="--",
            color="0.78",
            zorder=0,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(row_labels)
    ax.invert_yaxis()

    ax.tick_params(
        axis="y",
        length=0,
    )

    ax.set_xlim(0.35, 1.01)
    ax.set_xticks(
        np.arange(0.40, 1.01, 0.10)
    )
    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    ax.set_xlabel("Performance (mean ± SD)")

    ax.grid(
        axis="x",
        linestyle=":",
        linewidth=0.6,
        color="0.82",
        zorder=0,
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.24,
        right=0.99,
        bottom=0.08,
        top=0.92,
    )

    save_figure(
        fig,
        "FigS2_modulewise_ablation_all_datasets_forest_v2",
    )

    plt.close(fig)


if __name__ == "__main__":
    draw_cancer_ablation()
    draw_all_dataset_ablation()

    print()
    print("Finished successfully.")
    print("Output directory:", OUTPUT_DIR)
