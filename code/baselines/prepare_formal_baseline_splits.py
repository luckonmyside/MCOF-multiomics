from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


DATASETS = ["BRCA", "STAD", "ROSMAP", "SCZ"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate the exact MCOF-se outer splits for baseline benchmarking."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1,
        help="Formal MCOF-se manifest used random_seed=1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "test_size": args.test_size,
        "repeats": args.repeats,
        "random_seed": args.random_seed,
        "splitter": "sklearn.model_selection.StratifiedShuffleSplit",
        "index_base": 0,
        "datasets": {},
    }

    for dataset in DATASETS:
        dataset_dir = data_root / dataset
        labels_file = dataset_dir / "labels_all.csv"
        if not labels_file.is_file():
            raise FileNotFoundError(labels_file)

        raw_labels = pd.read_csv(labels_file, header=None).iloc[:, 0]
        labels, unique_labels = pd.factorize(raw_labels, sort=True)
        labels = labels.astype(np.int64)

        splitter = StratifiedShuffleSplit(
            n_splits=args.repeats,
            test_size=args.test_size,
            random_state=args.random_seed,
        )

        rows = []
        for repeat, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros_like(labels), labels),
            start=1,
        ):
            for idx in train_idx:
                rows.append(
                    {
                        "repeat": repeat,
                        "sample_index": int(idx),
                        "subset": "train",
                        "encoded_label": int(labels[idx]),
                    }
                )
            for idx in test_idx:
                rows.append(
                    {
                        "repeat": repeat,
                        "sample_index": int(idx),
                        "subset": "test",
                        "encoded_label": int(labels[idx]),
                    }
                )

        split_df = pd.DataFrame(rows).sort_values(
            ["repeat", "subset", "sample_index"]
        )
        split_file = output_root / f"{dataset}_outer_splits.csv"
        split_df.to_csv(split_file, index=False)

        counts = (
            split_df.groupby(["repeat", "subset"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        counts.to_csv(
            output_root / f"{dataset}_split_counts.csv",
            index=False,
        )

        manifest["datasets"][dataset] = {
            "n_samples": int(len(labels)),
            "label_levels_sorted": [str(x) for x in unique_labels],
            "split_file": str(split_file),
        }

        print(f"{dataset}: {len(labels)} samples")
        print(counts.to_string(index=False))
        print(f"Saved: {split_file}\n")

    with (output_root / "split_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("All formal outer splits generated successfully.")
    print(f"Output: {output_root.resolve()}")


if __name__ == "__main__":
    main()
