from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from common import DATASET_CONFIG, DATASET_ORDER, MODEL_ORDER, read_fixed_splits, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict preflight check for the MCOF revision analysis toolkit.")
    parser.add_argument("--toolkit_root", required=True)
    parser.add_argument("--root", default="/path/to/mcof-workspace")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    toolkit_root = Path(args.toolkit_root)
    root = Path(args.root)
    work = root / "MCOF_publication_work_20260716"
    data_root = root / "MCOF_fixed_v1_run" / "data"
    split_root = root / "MCOF_baselines_20260722" / "splits"
    main_root = root / "MCOFv2_runs" / "formal_main_20260429_191748"
    ml_root = root / "MCOF_baselines_20260722" / "ML"
    diablo_root = root / "MCOF_baselines_20260722" / "DIABLO_final_completed_20260723"
    mogonet_root = work / "MOGONET_formal_20260723" / "results" / "formal_main_20260723"
    ig_root = root / "MCOFv2_runs" / "biomarker_IG_20260720_fixed2"

    errors: List[str] = []
    checks: List[Dict[str, object]] = []

    src = str((toolkit_root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import models_v2  # type: ignore
        import train_utils_v2  # type: ignore
        checks.append({"check": "toolkit_imports", "status": "PASS", "details": src})
    except Exception as error:
        errors.append(f"Toolkit import failure: {error!r}")
        checks.append({"check": "toolkit_imports", "status": "FAIL", "details": repr(error)})

    for dataset in DATASET_ORDER:
        data_dir = data_root / dataset
        labels_file = data_dir / "labels_all.csv"
        if not labels_file.is_file():
            errors.append(f"Missing {labels_file}")
            continue
        labels = pd.read_csv(labels_file, header=None).iloc[:, 0]
        omics_count = 0
        feature_dims = []
        while (data_dir / f"{omics_count+1}_all.csv").is_file():
            omics_count += 1
            matrix_path = data_dir / f"{omics_count}_all.csv"
            names_path = data_dir / f"{omics_count}_featname.csv"
            if not names_path.is_file():
                errors.append(f"Missing {names_path}")
                continue
            matrix_header = pd.read_csv(matrix_path, header=None, nrows=1)
            names = pd.read_csv(names_path, header=None)
            feature_dims.append(matrix_header.shape[1])
            if len(names) != matrix_header.shape[1]:
                errors.append(f"{dataset} block {omics_count}: feature-name mismatch")
        if omics_count != len(DATASET_CONFIG[dataset]["omics_names"]):
            errors.append(f"{dataset}: found {omics_count} omics blocks")
        try:
            read_fixed_splits(
                split_root / f"{dataset}_outer_splits.csv", len(labels), repeats=5
            )
            split_status = "PASS"
        except Exception as error:
            split_status = "FAIL"
            errors.append(f"{dataset} split failure: {error!r}")
        checkpoint_count = len(
            list((main_root / dataset / "mcof_se").glob("repeat_*/best_model.pt"))
        )
        if checkpoint_count != 5:
            errors.append(f"{dataset}: expected 5 checkpoints, found {checkpoint_count}")
        ig_count = len(
            list((ig_root / dataset).glob("repeat_*/biomarker_importance_by_class.csv"))
        )
        if ig_count != 5:
            errors.append(f"{dataset}: expected 5 class-specific IG files, found {ig_count}")
        checks.append(
            {
                "check": f"dataset_{dataset}",
                "status": "PASS" if split_status == "PASS" and checkpoint_count == 5 and ig_count == 5 else "FAIL",
                "n_samples": len(labels),
                "class_counts": json.dumps(labels.value_counts().sort_index().to_dict()),
                "omics_count": omics_count,
                "feature_dims": ";".join(map(str, feature_dims)),
                "checkpoint_count": checkpoint_count,
                "ig_class_file_count": ig_count,
                "split_status": split_status,
            }
        )

    # Baseline prediction coverage for BRCA/STAD.
    prediction_roots = {
        "KNN": ml_root,
        "SVM": ml_root,
        "NB": ml_root,
        "RF": ml_root,
        "DIABLO": diablo_root,
        "MOGONET": mogonet_root,
    }
    for dataset in ["BRCA", "STAD"]:
        for model, model_root in prediction_roots.items():
            if model in {"KNN", "SVM", "NB", "RF"}:
                pattern_root = model_root / dataset / model
            else:
                pattern_root = model_root / dataset / model
            count = len(list(pattern_root.glob("repeat_*/predictions_test.csv")))
            status = "PASS" if count == 5 else "FAIL"
            if count != 5:
                errors.append(f"{dataset}/{model}: expected 5 prediction files, found {count}")
            checks.append(
                {
                    "check": f"predictions_{dataset}_{model}",
                    "status": status,
                    "count": count,
                    "root": str(pattern_root),
                }
            )

    checks.append(
        {
            "check": "cuda_visible_now",
            "status": "PASS" if torch.cuda.is_available() else "INFO",
            "details": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not visible on login node; expected before Slurm allocation"
            ),
        }
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(output, index=False)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "passed": not errors,
                "errors": errors,
                "n_checks": len(checks),
                "toolkit_root": str(toolkit_root.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Preflight report: {output}")
    if errors:
        print("PRECHECK FAILED")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)
    print("PRECHECK PASSED")


if __name__ == "__main__":
    main()
