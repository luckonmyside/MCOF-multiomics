from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--main_root", required=True, help="formal_main_... directory containing dataset/mcof_se")
    p.add_argument("--ablation_root", required=True, help="formal_ablation_clean_... directory")
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("summary", obj)


def main() -> None:
    args = parse_args()
    main_root = Path(args.main_root)
    abl_root = Path(args.ablation_root)
    prefix = Path(args.output_prefix)
    datasets = ["BRCA", "STAD", "ROSMAP", "SCZ"]
    specs = [
        ("MCOF", "mcof_se", main_root),
        ("MCOF-minus-SE", "mcof_no_se", abl_root),
        ("MCOF-minus-Conv", "mcof_no_conv", abl_root),
    ]

    summary_rows = []
    repeat_frames = []
    for dataset in datasets:
        for paper_name, code_name, root in specs:
            d = root / dataset / code_name
            sf = d / "summary.json"
            rf = d / "repeat_test_metrics.csv"
            if not sf.is_file() or not rf.is_file():
                raise FileNotFoundError(f"Missing summary/repeat file under {d}")
            summary = load_summary(sf)
            for metric, values in summary.items():
                if isinstance(values, dict) and {"mean", "std"}.issubset(values):
                    summary_rows.append({
                        "dataset": dataset,
                        "model": paper_name,
                        "code_model": code_name,
                        "metric": metric,
                        "mean": float(values["mean"]),
                        "std": float(values["std"]),
                        "mean_sd": f'{float(values["mean"]):.3f} ± {float(values["std"]):.3f}',
                    })
            df = pd.read_csv(rf)
            df.insert(0, "code_model", code_name)
            df.insert(0, "model", paper_name)
            df.insert(0, "dataset", dataset)
            repeat_frames.append(df)

    long_df = pd.DataFrame(summary_rows)
    repeat_df = pd.concat(repeat_frames, ignore_index=True, sort=False)
    long_df.to_csv(prefix.with_name(prefix.name + "_summary_long.csv"), index=False)
    repeat_df.to_csv(prefix.with_name(prefix.name + "_repeat_metrics.csv"), index=False)
    wide_df = long_df.pivot_table(
        index=["dataset", "model", "code_model"],
        columns="metric", values="mean_sd", aggfunc="first"
    ).reset_index()
    wide_df.to_csv(prefix.with_name(prefix.name + "_summary_wide.csv"), index=False)

    print(wide_df.to_string(index=False))
    print("\nSaved files with prefix:", prefix)


if __name__ == "__main__":
    main()
