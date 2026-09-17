from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from common import DATASET_CONFIG, DATASET_ORDER, ensure_dir, read_fixed_splits, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare reviewer-requested single-omics, leave-one-omics-out, "
            "PAM50-exclusion, and feature-budget sensitivity tasks."
        )
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--pam50_file",
        default=str(Path(__file__).resolve().parents[1] / "config" / "pam50_genes.txt"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_dataset_inventory(dataset_dir: Path, dataset: str) -> Dict[str, Any]:
    omics_files: List[Path] = []
    feature_files: List[Path] = []
    block = 1
    n_samples = None
    while True:
        data_file = dataset_dir / f"{block}_all.csv"
        if not data_file.is_file():
            break
        frame = pd.read_csv(data_file, header=None)
        feature_file = dataset_dir / f"{block}_featname.csv"
        if not feature_file.is_file():
            raise FileNotFoundError(feature_file)
        names = pd.read_csv(feature_file, header=None)
        if len(names) != frame.shape[1]:
            raise ValueError(
                f"{dataset} block {block}: {len(names)} feature names for {frame.shape[1]} columns"
            )
        if n_samples is None:
            n_samples = int(frame.shape[0])
        elif frame.shape[0] != n_samples:
            raise ValueError(f"{dataset}: omics blocks have inconsistent sample counts")
        omics_files.append(data_file)
        feature_files.append(feature_file)
        block += 1

    if not omics_files:
        raise FileNotFoundError(f"No numbered omics files found in {dataset_dir}")
    labels_file = dataset_dir / "labels_all.csv"
    if not labels_file.is_file():
        raise FileNotFoundError(labels_file)
    labels = pd.read_csv(labels_file, header=None).iloc[:, 0]
    if len(labels) != n_samples:
        raise ValueError(f"{dataset}: labels contain {len(labels)} rows; expected {n_samples}")

    expected_omics = DATASET_CONFIG[dataset]["omics_names"]
    if len(omics_files) != len(expected_omics):
        raise ValueError(
            f"{dataset}: found {len(omics_files)} omics blocks, expected {len(expected_omics)}"
        )

    return {
        "dataset": dataset,
        "dataset_dir": dataset_dir,
        "n_samples": int(n_samples),
        "class_counts": labels.value_counts().sort_index().to_dict(),
        "omics_files": omics_files,
        "feature_files": feature_files,
        "labels_file": labels_file,
        "feature_dims": [
            int(pd.read_csv(path, header=None, nrows=1).shape[1])
            for path in omics_files
        ],
    }


def safe_symlink(source: Path, destination: Path, force: bool) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source:
            return
        if not force:
            raise FileExistsError(
                f"{destination} already exists and does not point to {source}; use --force"
            )
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.symlink_to(source)


def create_block_subset(
    inventory: Dict[str, Any],
    variant_dir: Path,
    included_indices: Sequence[int],
    force: bool,
) -> None:
    ensure_dir(variant_dir)
    for target_index, source_index in enumerate(included_indices, start=1):
        safe_symlink(
            inventory["omics_files"][source_index],
            variant_dir / f"{target_index}_all.csv",
            force,
        )
        safe_symlink(
            inventory["feature_files"][source_index],
            variant_dir / f"{target_index}_featname.csv",
            force,
        )
    safe_symlink(inventory["labels_file"], variant_dir / "labels_all.csv", force)


def normalize_gene_symbol(value: str) -> str:
    text = str(value).strip().strip('"').strip("'")
    # Common annotation layouts: SYMBOL|ENSG..., SYMBOL (description), SYMBOL.1
    text = text.split("|")[0].strip()
    text = re.sub(r"\s*\(.*\)$", "", text)
    text = re.sub(r"\.\d+$", "", text)
    text = text.upper()
    aliases = {"ORC6": "ORC6L"}
    return aliases.get(text, text)


def create_pam50_exclusion(
    inventory: Dict[str, Any],
    variant_dir: Path,
    pam50_file: Path,
    force: bool,
    report_path: Path,
) -> Dict[str, Any]:
    if inventory["dataset"] != "BRCA":
        raise ValueError("PAM50 exclusion is defined only for BRCA")
    pam50 = {
        normalize_gene_symbol(line)
        for line in pam50_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(pam50) != 50:
        raise ValueError(f"Expected 50 unique PAM50 genes, found {len(pam50)}")

    names_path = inventory["feature_files"][0]
    data_path = inventory["omics_files"][0]
    names = pd.read_csv(names_path, header=None).iloc[:, 0].astype(str)
    normalized = names.map(normalize_gene_symbol)
    remove_mask = normalized.isin(pam50).to_numpy()
    removed_indices = np.flatnonzero(remove_mask)

    report = pd.DataFrame(
        {
            "original_column_index": removed_indices,
            "feature_name": names.iloc[removed_indices].to_numpy(),
            "normalized_symbol": normalized.iloc[removed_indices].to_numpy(),
            "canonical_PAM50_member": True,
        }
    )
    report.to_csv(report_path, index=False)

    if variant_dir.exists() and force:
        shutil.rmtree(variant_dir)
    ensure_dir(variant_dir)
    filtered_names_path = variant_dir / "1_featname.csv"
    filtered_data_path = variant_dir / "1_all.csv"

    if filtered_data_path.exists() and not force:
        raise FileExistsError(f"{filtered_data_path} exists; use --force")

    data = pd.read_csv(data_path, header=None)
    if data.shape[1] != len(names):
        raise ValueError("BRCA RNA matrix and feature-name file are inconsistent")
    keep_mask = ~remove_mask
    data.loc[:, keep_mask].to_csv(filtered_data_path, header=False, index=False)
    names.loc[keep_mask].to_csv(filtered_names_path, header=False, index=False)

    # Preserve miRNA as the second omics block.
    safe_symlink(inventory["omics_files"][1], variant_dir / "2_all.csv", force)
    safe_symlink(inventory["feature_files"][1], variant_dir / "2_featname.csv", force)
    safe_symlink(inventory["labels_file"], variant_dir / "labels_all.csv", force)

    metadata = {
        "pam50_reference_file": str(pam50_file.resolve()),
        "pam50_reference_sha256": sha256_file(pam50_file),
        "n_pam50_genes": len(pam50),
        "n_rna_features_before": int(len(names)),
        "n_pam50_features_removed": int(remove_mask.sum()),
        "n_rna_features_after": int(keep_mask.sum()),
        "removed_features": report.to_dict(orient="records"),
    }
    write_json(metadata, variant_dir / "pam50_exclusion_manifest.json")
    return metadata


def task_record(
    task_id: int,
    phase: str,
    analysis_type: str,
    dataset: str,
    variant: str,
    data_dir: Path,
    split_file: Path,
    output_dir: Path,
    included_omics: Sequence[str],
    excluded_omics: Sequence[str],
    select_k: int | None = None,
    selector: str = "none",
    description: str = "",
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "phase": phase,
        "analysis_type": analysis_type,
        "dataset": dataset,
        "variant": variant,
        "description": description,
        "data_dir": str(data_dir.resolve()),
        "split_file": str(split_file.resolve()),
        "output_dir": str(output_dir.resolve()),
        "included_omics": ";".join(included_omics),
        "excluded_omics": ";".join(excluded_omics),
        "select_k": "" if select_k is None else int(select_k),
        "selector": selector,
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    split_root = Path(args.split_root)
    workspace = Path(args.workspace)
    pam50_file = Path(args.pam50_file)

    variant_root = ensure_dir(workspace / "variant_data")
    manifest_root = ensure_dir(workspace / "manifests")
    result_root = ensure_dir(workspace / "results")
    report_root = ensure_dir(workspace / "reports")
    ensure_dir(workspace / "logs")
    ensure_dir(workspace / "status")

    inventories: Dict[str, Dict[str, Any]] = {}
    inventory_rows = []
    for dataset in DATASET_ORDER:
        inventory = read_dataset_inventory(data_root / dataset, dataset)
        split_path = split_root / f"{dataset}_outer_splits.csv"
        read_fixed_splits(split_path, inventory["n_samples"], repeats=5)
        inventories[dataset] = inventory
        inventory_rows.append(
            {
                "dataset": dataset,
                "n_samples": inventory["n_samples"],
                "class_counts": json.dumps(inventory["class_counts"], sort_keys=True),
                "omics_names": ";".join(DATASET_CONFIG[dataset]["omics_names"]),
                "feature_dims": ";".join(map(str, inventory["feature_dims"])),
                "split_file": str(split_path.resolve()),
            }
        )
    pd.DataFrame(inventory_rows).to_csv(report_root / "input_inventory.csv", index=False)

    tasks: List[Dict[str, Any]] = []
    task_id = 0

    # Single-omics models. For two-modality datasets, these are also the
    # leave-one-omics-out experiments requested by the reviewers.
    for dataset in DATASET_ORDER:
        inventory = inventories[dataset]
        omics_names = DATASET_CONFIG[dataset]["omics_names"]
        split_file = split_root / f"{dataset}_outer_splits.csv"
        for source_index, omics_name in enumerate(omics_names):
            variant = f"single_{omics_name}"
            data_dir = variant_root / dataset / variant
            create_block_subset(inventory, data_dir, [source_index], args.force)
            tasks.append(
                task_record(
                    task_id,
                    "core",
                    "single_omics",
                    dataset,
                    variant,
                    data_dir,
                    split_file,
                    result_root / "core" / dataset / variant / "mcof_se",
                    [omics_name],
                    [value for value in omics_names if value != omics_name],
                    description=f"Unimodal MCOF using only {omics_name}",
                )
            )
            task_id += 1

        if len(omics_names) >= 3:
            for omitted_index, omitted_name in enumerate(omics_names):
                included_indices = [
                    index for index in range(len(omics_names)) if index != omitted_index
                ]
                included_names = [omics_names[index] for index in included_indices]
                variant = f"without_{omitted_name}"
                data_dir = variant_root / dataset / variant
                create_block_subset(inventory, data_dir, included_indices, args.force)
                tasks.append(
                    task_record(
                        task_id,
                        "core",
                        "leave_one_omics_out",
                        dataset,
                        variant,
                        data_dir,
                        split_file,
                        result_root / "core" / dataset / variant / "mcof_se",
                        included_names,
                        [omitted_name],
                        description=f"MCOF with {omitted_name} omitted",
                    )
                )
                task_id += 1

    # BRCA PAM50-gene exclusion sensitivity analysis.
    pam50_variant = variant_root / "BRCA" / "without_PAM50_genes"
    pam50_metadata = create_pam50_exclusion(
        inventories["BRCA"],
        pam50_variant,
        pam50_file,
        args.force,
        report_root / "BRCA_PAM50_overlap_report.csv",
    )
    tasks.append(
        task_record(
            task_id,
            "core",
            "label_definition_sensitivity",
            "BRCA",
            "without_PAM50_genes",
            pam50_variant,
            split_root / "BRCA_outer_splits.csv",
            result_root / "core" / "BRCA" / "without_PAM50_genes" / "mcof_se",
            DATASET_CONFIG["BRCA"]["omics_names"],
            [f"{pam50_metadata['n_pam50_features_removed']} PAM50-panel RNA features"],
            description="BRCA sensitivity analysis after removing retained PAM50 genes from RNA",
        )
    )
    task_id += 1

    # Training-set-only feature-budget sensitivity within the fixed candidate
    # panels. This does not erase the upstream fixed-panel limitation; it answers
    # whether conclusions are sensitive to the downstream feature budget.
    budget_grid = {
        "BRCA": [100, 250, 500],
        "STAD": [100, 250, 500],
        "ROSMAP": [50, 100],
        "SCZ": [50, 100],
    }
    for dataset, k_values in budget_grid.items():
        for k in k_values:
            variant = f"train_only_f_classif_k{k}_per_omics"
            tasks.append(
                task_record(
                    task_id,
                    "budget",
                    "feature_budget_sensitivity",
                    dataset,
                    variant,
                    data_root / dataset,
                    split_root / f"{dataset}_outer_splits.csv",
                    result_root / "budget" / dataset / variant / "mcof_se",
                    DATASET_CONFIG[dataset]["omics_names"],
                    [],
                    select_k=k,
                    selector="f_classif",
                    description=(
                        f"Training-set-only ANOVA F selection of {k} features per omics block "
                        "within the fixed candidate panel"
                    ),
                )
            )
            task_id += 1

    task_frame = pd.DataFrame(tasks)
    task_frame.to_csv(manifest_root / "all_tasks.csv", index=False)
    task_frame[task_frame["phase"].eq("core")].reset_index(drop=True).to_csv(
        manifest_root / "core_tasks.csv", index=False
    )
    task_frame[task_frame["phase"].eq("budget")].reset_index(drop=True).to_csv(
        manifest_root / "budget_tasks.csv", index=False
    )

    write_json(
        {
            "data_root": str(data_root.resolve()),
            "split_root": str(split_root.resolve()),
            "workspace": str(workspace.resolve()),
            "n_core_tasks": int(task_frame["phase"].eq("core").sum()),
            "n_budget_tasks": int(task_frame["phase"].eq("budget").sum()),
            "pam50_sensitivity": pam50_metadata,
            "important_scope_note": (
                "All analyses use the existing fixed candidate panels. The feature-budget experiments "
                "perform train-only selection within those panels and do not retroactively remove the "
                "upstream fixed-panel limitation."
            ),
        },
        manifest_root / "preparation_manifest.json",
    )

    print(f"Prepared {len(task_frame)} tasks in {workspace}")
    print(f"Core tasks: {(task_frame['phase'] == 'core').sum()}")
    print(f"Feature-budget tasks: {(task_frame['phase'] == 'budget').sum()}")
    print(
        f"PAM50 overlap: removed {pam50_metadata['n_pam50_features_removed']} "
        f"of {pam50_metadata['n_rna_features_before']} retained BRCA RNA features"
    )
    print(f"Core manifest: {manifest_root / 'core_tasks.csv'}")
    print(f"Budget manifest: {manifest_root / 'budget_tasks.csv'}")


if __name__ == "__main__":
    main()
