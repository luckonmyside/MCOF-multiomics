from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path(
    "/path/to/mcof-workspace/"
    "MCOF_clean_ablation_20260721_primary_table.csv"
)

OUTPUT_DIR = Path(
    "/path/to/mcof-workspace/figures_20260721"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_ORDER = [
    "MCOF",
    "MCOF w/o channel attention",
    "MCOF w/o convolution",
]

MODEL_LABELS = {
    "MCOF": "MCOF",
    "MCOF w/o channel attention": "w/o channel attention",
    "MCOF w/o convolution": "w/o convolution",
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


def parse_mean_sd(value):
    if pd.isna(value):
        return np.nan, np.nan

    match = re.match(
        r"\s*([0-9.]+)\s*±\s*([0-9.]+)\s*",
        str(value),
    )

    if not match:
        raise ValueError(f"Cannot parse value: {value}")

    return float(match.group(1)), float(match.group(2))


df = pd.read_csv(INPUT)

records = []

for _, row in df.iterrows():
    dataset = row["dataset"]
    model = row["model"]

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


def draw_figure(
    datasets,
    filename,
    title,
    y_limits,
    figure_size,
):
    categories = []
    metadata = []

    for dataset in datasets:
        for metric, metric_label in DATASET_METRICS[dataset]:
            categories.append(f"{dataset}\n{metric_label}")
            metadata.append((dataset, metric))

    x = np.arange(len(categories))
    offsets = [-0.22, 0.00, 0.22]
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
                (plot_df["dataset"] == dataset)
                & (plot_df["model"] == model)
                & (plot_df["metric"] == metric)
            ]

            if len(selected) != 1:
                raise ValueError(
                    f"Missing or duplicated result: "
                    f"{dataset}, {model}, {metric}"
                )

            means.append(selected.iloc[0]["mean"])
            sds.append(selected.iloc[0]["sd"])

        ax.errorbar(
            x + offset,
            means,
            yerr=sds,
            fmt=marker,
            linestyle="none",
            capsize=3,
            markersize=6,
            linewidth=1.2,
            label=MODEL_LABELS[model],
        )

    # Separate datasets visually
    for boundary in range(4, len(categories), 4):
        ax.axvline(
            boundary - 0.5,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Performance (mean ± SD)")
    ax.set_title(title)
    ax.set_ylim(*y_limits)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()

    png = OUTPUT_DIR / f"{filename}.png"
    pdf = OUTPUT_DIR / f"{filename}.pdf"

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", png)
    print("Saved:", pdf)


draw_figure(
    datasets=["BRCA", "STAD"],
    filename="Fig3_strict_ablation_BRCA_STAD",
    title="Strict ablation analysis in cancer subtype classification",
    y_limits=(0.70, 1.01),
    figure_size=(10.5, 5.4),
)

draw_figure(
    datasets=["BRCA", "STAD", "ROSMAP", "SCZ"],
    filename="FigS2_strict_ablation_all_datasets",
    title="Strict ablation analysis across four multi-omics datasets",
    y_limits=(0.35, 1.02),
    figure_size=(16.5, 5.8),
)

print("\nFinished.")
print("Output directory:", OUTPUT_DIR)
