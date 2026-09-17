import pandas as pd
import numpy as np
from pathlib import Path


base = Path(
    "/path/to/mcof-workspace/"
    "MCOFv2_runs/biomarker_IG_20260720_fixed2/BRCA"
)

out_dir = Path(
    "/path/to/mcof-workspace/"
    "MCOFv2_runs/biomarker_IG_stability_20260720/BRCA"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True
)

repeats = [
    "repeat_1",
    "repeat_2",
    "repeat_3",
    "repeat_4",
    "repeat_5"
]

all_rank = []

for r in repeats:

    file = (
        base /
        r /
        "biomarker_importance_overall.csv"
    )

    print("Loading:", file)

    df = pd.read_csv(file)

    df["rank"] = np.arange(1, len(df)+1)

    df["repeat"] = r

    all_rank.append(df)


combined = pd.concat(
    all_rank,
    axis=0,
    ignore_index=True
)


summary = (
    combined
    .groupby("feature")
    .agg(
        mean_IG=("importance","mean"),
        sd_IG=("importance","std"),
        mean_rank=("rank","mean"),
        sd_rank=("rank","std")
    )
    .reset_index()
)


for k in [50,100,200]:

    freq = (
        combined[
            combined["rank"]<=k
        ]
        .groupby("feature")
        .size()
        .reset_index(
            name=f"top{k}_frequency"
        )
    )

    freq[f"top{k}_frequency"] = (
        freq[f"top{k}_frequency"]/5
    )

    summary = summary.merge(
        freq,
        on="feature",
        how="left"
    )


summary = summary.fillna(0)


def get_omics(x):

    if x.startswith("hsa-"):
        return "miRNA"
    else:
        return "mRNA"


summary["omics"] = summary["feature"].apply(get_omics)


summary = summary.sort_values(
    [
        "top50_frequency",
        "mean_IG"
    ],
    ascending=False
)


summary.to_csv(
    out_dir /
    "BRCA_IG_stability_all_features.csv",
    index=False
)


summary.head(50).to_csv(
    out_dir /
    "BRCA_stable_top50.csv",
    index=False
)


summary.head(100).to_csv(
    out_dir /
    "BRCA_stable_top100.csv",
    index=False
)


print("\nFinished.")
print("Saved:", out_dir)

print("\nTop 20 stable biomarkers:")
print(
    summary.head(20).to_string(
        index=False
    )
)
