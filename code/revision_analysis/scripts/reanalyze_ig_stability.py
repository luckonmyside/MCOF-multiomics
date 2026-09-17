from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from common import DATASET_CONFIG, DATASET_ORDER, ensure_dir, mean_sd, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reanalyze repeat-level IG tables with explicit feature universes, "
            "prevalence- and equally-weighted class aggregation, chance-adjusted "
            "Jaccard similarity, and machine-readable complete rankings."
        )
    )
    parser.add_argument("--ig_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--datasets", nargs="+", default=DATASET_ORDER)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--absolute_k", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    return parser.parse_args()


def locate_repeat_file(dataset_dir: Path, repeat: int, filename: str) -> Path:
    candidates = [
        dataset_dir / f"repeat_{repeat}" / filename,
        dataset_dir / f"repeat_{repeat}" / "biomarker" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(dataset_dir.glob(f"**/repeat_{repeat}/**/{filename}"))
    # Ignore explicitly archived pre-fix directories when an exact current path exists.
    matches = [
        path
        for path in matches
        if "before_annotation_fix" not in str(path)
        and "/test" not in str(path).replace("\\", "/")
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No {filename} found for repeat {repeat} under {dataset_dir}"
        )
    raise RuntimeError(
        f"Multiple candidate {filename} files for {dataset_dir.name} repeat {repeat}: "
        + ", ".join(map(str, matches))
    )


def feature_metadata(data_dir: Path, dataset: str) -> pd.DataFrame:
    omics_names = DATASET_CONFIG[dataset]["omics_names"]
    rows: List[Dict[str, Any]] = []
    global_index = 0
    for omics_index, omics_name in enumerate(omics_names, start=1):
        matrix_path = data_dir / f"{omics_index}_all.csv"
        names_path = data_dir / f"{omics_index}_featname.csv"
        if not matrix_path.is_file() or not names_path.is_file():
            raise FileNotFoundError(f"Missing {matrix_path} or {names_path}")
        n_features = pd.read_csv(matrix_path, header=None, nrows=1).shape[1]
        names = pd.read_csv(names_path, header=None).iloc[:, 0].astype(str)
        if len(names) != n_features:
            raise ValueError(
                f"{dataset} {omics_name}: {len(names)} names for {n_features} columns"
            )
        for within_index, name in enumerate(names):
            rows.append(
                {
                    "feature_index": global_index,
                    "omics_index": omics_index,
                    "omics": omics_name,
                    "feature_within_omics": within_index,
                    "feature": str(name),
                    "feature_key": f"{omics_name}::{name}",
                }
            )
            global_index += 1
    metadata = pd.DataFrame(rows)
    if metadata["feature_key"].duplicated().any():
        duplicated = metadata.loc[
            metadata["feature_key"].duplicated(keep=False), "feature_key"
        ].head(20)
        raise ValueError(f"{dataset}: duplicated feature identifiers: {duplicated.tolist()}")
    return metadata


def parse_generic_feature(value: str) -> Tuple[int, int] | None:
    match = re.fullmatch(r"omics[_-]?(\d+)_f(\d+)", str(value).strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def harmonize_class_file(
    path: Path,
    metadata: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"class", "feature", "importance"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["class"] = pd.to_numeric(frame["class"], errors="raise").astype(int)
    frame["importance"] = pd.to_numeric(frame["importance"], errors="raise").abs()
    frame["feature"] = frame["feature"].astype(str)

    metadata_by_index = metadata.set_index("feature_index")
    metadata_by_key = metadata.set_index("feature_key")

    if "feature_index" in frame.columns:
        frame["feature_index"] = pd.to_numeric(
            frame["feature_index"], errors="raise"
        ).astype(int)
        invalid = set(frame["feature_index"]).difference(metadata_by_index.index)
        if invalid:
            raise ValueError(f"{path}: invalid feature_index values: {sorted(invalid)[:10]}")
        resolved = metadata_by_index.loc[frame["feature_index"].to_numpy()].reset_index()
    elif "omics" in frame.columns:
        frame["omics"] = frame["omics"].astype(str)
        keys = frame["omics"] + "::" + frame["feature"]
        if set(keys).issubset(metadata_by_key.index):
            resolved = metadata_by_key.loc[keys.to_numpy()].reset_index()
        else:
            # Historical files sometimes used omics_1/omics_2 rather than the
            # biological names. Resolve generic identifiers by block and index.
            resolved_rows = []
            for feature in frame["feature"]:
                parsed = parse_generic_feature(feature)
                if parsed is None:
                    raise ValueError(
                        f"{path}: cannot match feature {feature!r} to the fixed universe"
                    )
                omics_index, within_index = parsed
                selected = metadata[
                    metadata["omics_index"].eq(omics_index)
                    & metadata["feature_within_omics"].eq(within_index)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"{path}: generic feature {feature} resolved to {len(selected)} rows"
                    )
                resolved_rows.append(selected.iloc[0])
            resolved = pd.DataFrame(resolved_rows).reset_index(drop=True)
    else:
        # Try direct unique feature-name mapping, then generic identifiers.
        name_counts = metadata["feature"].value_counts()
        unique_name_map = metadata[name_counts.loc[metadata["feature"]].to_numpy() == 1].set_index("feature", drop=False)
        resolved_rows = []
        for feature in frame["feature"]:
            if feature in unique_name_map.index:
                resolved_rows.append(unique_name_map.loc[feature])
                continue
            parsed = parse_generic_feature(feature)
            if parsed is None:
                raise ValueError(
                    f"{path}: feature {feature!r} is not uniquely resolvable; add omics or feature_index columns"
                )
            omics_index, within_index = parsed
            selected = metadata[
                metadata["omics_index"].eq(omics_index)
                & metadata["feature_within_omics"].eq(within_index)
            ]
            if len(selected) != 1:
                raise ValueError(f"{path}: cannot resolve generic feature {feature}")
            resolved_rows.append(selected.iloc[0])
        resolved = pd.DataFrame(resolved_rows).reset_index(drop=True)

    result = resolved[
        [
            "feature_index",
            "omics_index",
            "omics",
            "feature_within_omics",
            "feature",
            "feature_key",
        ]
    ].copy()
    result.insert(0, "class", frame["class"].to_numpy())
    result["importance"] = frame["importance"].to_numpy(dtype=float)

    # Every class must contain every feature exactly once.
    expected_features = set(metadata["feature_index"])
    for class_index, group in result.groupby("class"):
        if group["feature_index"].duplicated().any():
            raise ValueError(f"{path}: class {class_index} contains duplicate features")
        if set(group["feature_index"]) != expected_features:
            missing_features = sorted(expected_features.difference(group["feature_index"]))[:10]
            extra_features = sorted(set(group["feature_index"]).difference(expected_features))[:10]
            raise ValueError(
                f"{path}: class {class_index} feature universe mismatch; "
                f"missing={missing_features}, extra={extra_features}"
            )
    return result


def pairwise_jaccard(keys_by_repeat: Dict[int, set[str]], p: int, k: int) -> pd.DataFrame:
    rows = []
    expected = k / (2 * p - k) if (2 * p - k) > 0 else 1.0
    for repeat_a, repeat_b in combinations(sorted(keys_by_repeat), 2):
        set_a = keys_by_repeat[repeat_a]
        set_b = keys_by_repeat[repeat_b]
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        raw = intersection / union if union else 1.0
        adjusted = (raw - expected) / (1.0 - expected) if expected < 1.0 else 1.0
        kuncheva = (intersection * p - k * k) / (k * (p - k)) if 0 < k < p else 1.0
        rows.append(
            {
                "repeat_a": repeat_a,
                "repeat_b": repeat_b,
                "feature_universe_size": p,
                "k": k,
                "k_fraction": k / p,
                "intersection": intersection,
                "union": union,
                "raw_jaccard": raw,
                "chance_expected_jaccard": expected,
                "chance_adjusted_jaccard": adjusted,
                "kuncheva_index": kuncheva,
            }
        )
    return pd.DataFrame(rows)


def aggregate_repeat(
    class_frame: pd.DataFrame,
    class_weights: np.ndarray,
    aggregation: str,
    repeat: int,
) -> pd.DataFrame:
    pivot = class_frame.pivot(
        index="feature_index", columns="class", values="importance"
    ).sort_index()
    n_classes = pivot.shape[1]
    if aggregation == "prevalence_weighted":
        if len(class_weights) != n_classes:
            raise ValueError("Class-weight vector and class-specific IG table are inconsistent")
        importance = pivot.to_numpy() @ class_weights
    elif aggregation == "equal_class_weighted":
        importance = pivot.mean(axis=1).to_numpy()
    else:
        raise ValueError(aggregation)

    metadata = (
        class_frame[
            [
                "feature_index",
                "omics_index",
                "omics",
                "feature_within_omics",
                "feature",
                "feature_key",
            ]
        ]
        .drop_duplicates("feature_index")
        .set_index("feature_index")
        .sort_index()
    )
    result = metadata.reset_index()
    result["importance"] = importance
    result = result.sort_values(
        ["importance", "feature_key"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result.insert(0, "repeat", repeat)
    result.insert(0, "aggregation", aggregation)
    return result


def summarize_aggregation(
    dataset: str,
    aggregation: str,
    long_frame: pd.DataFrame,
    output_dir: Path,
    absolute_k: Sequence[int],
    fractions: Sequence[float],
) -> Dict[str, Any]:
    p = int(long_frame["feature_index"].nunique())
    repeats = sorted(long_frame["repeat"].unique())
    rank_wide = long_frame.pivot(index="feature_key", columns="repeat", values="rank")
    if rank_wide.isna().any().any():
        raise ValueError(f"{dataset}/{aggregation}: incomplete rank matrix")
    spearman = rank_wide.corr(method="spearman")
    spearman.to_csv(output_dir / f"{dataset}_{aggregation}_spearman_matrix.csv")

    spearman_pairs = []
    for a, b in combinations(repeats, 2):
        spearman_pairs.append(
            {
                "dataset": dataset,
                "aggregation": aggregation,
                "repeat_a": a,
                "repeat_b": b,
                "spearman": float(spearman.loc[a, b]),
            }
        )
    spearman_pair_frame = pd.DataFrame(spearman_pairs)
    spearman_pair_frame.to_csv(
        output_dir / f"{dataset}_{aggregation}_spearman_pairs.csv", index=False
    )

    k_specs: List[Tuple[str, int]] = []
    for k in absolute_k:
        k_specs.append((f"absolute_{k}", min(int(k), p)))
    for fraction in fractions:
        if not (0 < fraction <= 1):
            raise ValueError(f"Invalid feature fraction: {fraction}")
        k = max(1, int(round(p * fraction)))
        k_specs.append((f"fraction_{fraction:.4f}", k))

    overlap_frames = []
    for label, k in k_specs:
        sets = {
            int(repeat): set(
                long_frame[
                    long_frame["repeat"].eq(repeat)
                ].nsmallest(k, "rank")["feature_key"]
            )
            for repeat in repeats
        }
        pair_frame = pairwise_jaccard(sets, p=p, k=k)
        pair_frame.insert(0, "k_definition", label)
        pair_frame.insert(0, "aggregation", aggregation)
        pair_frame.insert(0, "dataset", dataset)
        overlap_frames.append(pair_frame)
    overlap = pd.concat(overlap_frames, ignore_index=True)
    overlap.to_csv(
        output_dir / f"{dataset}_{aggregation}_jaccard_pairs.csv", index=False
    )

    importance_wide = long_frame.pivot(
        index="feature_key", columns="repeat", values="importance"
    )
    metadata = (
        long_frame[
            [
                "feature_key",
                "feature_index",
                "omics_index",
                "omics",
                "feature_within_omics",
                "feature",
            ]
        ]
        .drop_duplicates("feature_key")
        .set_index("feature_key")
    )
    stability = metadata.copy()
    stability["mean_abs_IG"] = importance_wide.mean(axis=1)
    stability["sd_abs_IG"] = importance_wide.std(axis=1, ddof=1)
    stability["mean_rank"] = rank_wide.mean(axis=1)
    stability["median_rank"] = rank_wide.median(axis=1)
    stability["min_rank"] = rank_wide.min(axis=1).astype(int)
    stability["max_rank"] = rank_wide.max(axis=1).astype(int)

    for k in sorted(set([min(int(value), p) for value in absolute_k])):
        count = (rank_wide <= k).sum(axis=1).astype(int)
        stability[f"top{k}_count"] = count
        stability[f"top{k}_frequency"] = count / len(repeats)

    # Preserve the historical lexicographic stability ranking, but make every
    # criterion explicit. Also supply a simpler median-rank-only ordering.
    sort_columns = []
    ascending = []
    for k in sorted(set([min(int(value), p) for value in absolute_k])):
        sort_columns.append(f"top{k}_count")
        ascending.append(False)
    sort_columns.extend(["mean_rank", "median_rank", "mean_abs_IG"])
    ascending.extend([True, True, False])
    stability = stability.sort_values(
        sort_columns, ascending=ascending, kind="mergesort"
    ).reset_index()
    stability.insert(0, "stability_rank", np.arange(1, len(stability) + 1))
    stability["median_rank_only_rank"] = (
        stability["median_rank"].rank(method="min", ascending=True).astype(int)
    )
    stability.to_csv(
        output_dir / f"{dataset}_{aggregation}_stability_all_features.csv",
        index=False,
    )

    summary = {
        "dataset": dataset,
        "aggregation": aggregation,
        "feature_universe_size": p,
        "n_repeats": len(repeats),
        "n_repeat_pairs": len(spearman_pair_frame),
        "spearman_mean": float(spearman_pair_frame["spearman"].mean()),
        "spearman_sd": float(spearman_pair_frame["spearman"].std(ddof=1)),
        "spearman_median": float(spearman_pair_frame["spearman"].median()),
    }
    for label, group in overlap.groupby("k_definition"):
        summary[f"{label}_k"] = int(group["k"].iloc[0])
        summary[f"{label}_k_fraction"] = float(group["k_fraction"].iloc[0])
        summary[f"{label}_raw_jaccard_mean"] = float(group["raw_jaccard"].mean())
        summary[f"{label}_adjusted_jaccard_mean"] = float(
            group["chance_adjusted_jaccard"].mean()
        )
        summary[f"{label}_kuncheva_mean"] = float(group["kuncheva_index"].mean())
    return summary


def main() -> None:
    args = parse_args()
    ig_root = Path(args.ig_root)
    data_root = Path(args.data_root)
    output_root = ensure_dir(args.output_root)

    all_summaries = []
    protocol_rows = []

    for dataset in args.datasets:
        dataset_output = ensure_dir(output_root / dataset)
        metadata = feature_metadata(data_root / dataset, dataset)
        p = len(metadata)
        labels = pd.read_csv(
            data_root / dataset / "labels_all.csv", header=None
        ).iloc[:, 0]
        encoded, levels = pd.factorize(labels, sort=True)
        class_counts = np.bincount(encoded, minlength=len(levels)).astype(float)
        prevalence_weights = class_counts / class_counts.sum()

        repeat_aggregates = []
        class_long_frames = []
        source_files = []
        metadata_files = []

        for repeat in range(1, args.repeats + 1):
            class_path = locate_repeat_file(
                ig_root / dataset, repeat, "biomarker_importance_by_class.csv"
            )
            source_files.append(str(class_path))
            class_frame = harmonize_class_file(class_path, metadata, dataset)
            observed_classes = sorted(class_frame["class"].unique().tolist())
            expected_classes = list(range(len(levels)))
            if observed_classes != expected_classes:
                raise ValueError(
                    f"{class_path}: classes {observed_classes}; expected {expected_classes}"
                )
            class_frame.insert(0, "repeat", repeat)
            class_long_frames.append(class_frame)

            for aggregation in ["prevalence_weighted", "equal_class_weighted"]:
                repeat_aggregates.append(
                    aggregate_repeat(
                        class_frame.drop(columns="repeat"),
                        prevalence_weights,
                        aggregation,
                        repeat,
                    )
                )

            metadata_path = class_path.parent / "biomarker_run_metadata.json"
            if metadata_path.is_file():
                metadata_files.append(str(metadata_path))
                metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                protocol_rows.append(
                    {
                        "dataset": dataset,
                        "repeat": repeat,
                        "ig_steps": metadata_payload.get("ig_steps"),
                        "implementation": metadata_payload.get(
                            "integrated_gradients_implementation"
                        ),
                        "baseline": "all-zero vector in checkpoint-specific standardized input space",
                        "samples_attributed": "all samples",
                        "target_classes": "every output class",
                        "source_metadata": str(metadata_path),
                    }
                )
            else:
                protocol_rows.append(
                    {
                        "dataset": dataset,
                        "repeat": repeat,
                        "ig_steps": np.nan,
                        "implementation": "metadata file unavailable",
                        "baseline": "all-zero vector according to submitted attribution code",
                        "samples_attributed": "all samples according to submitted attribution code",
                        "target_classes": "every output class according to submitted attribution code",
                        "source_metadata": "",
                    }
                )

        class_long = pd.concat(class_long_frames, ignore_index=True)
        class_long.to_csv(
            dataset_output / f"{dataset}_class_specific_IG_all_repeats.csv",
            index=False,
        )
        aggregate_long = pd.concat(repeat_aggregates, ignore_index=True)
        if aggregate_long.groupby(["aggregation", "repeat"]).size().ne(p).any():
            raise ValueError(f"{dataset}: incomplete aggregate ranking")
        aggregate_long.to_csv(
            dataset_output / f"{dataset}_IG_complete_rankings_both_aggregations.csv",
            index=False,
        )

        for aggregation in ["prevalence_weighted", "equal_class_weighted"]:
            current = aggregate_long[aggregate_long["aggregation"].eq(aggregation)]
            summary = summarize_aggregation(
                dataset,
                aggregation,
                current,
                dataset_output,
                absolute_k=args.absolute_k,
                fractions=args.fractions,
            )
            summary["class_counts"] = json.dumps(
                {str(index): int(value) for index, value in enumerate(class_counts)},
                sort_keys=True,
            )
            summary["class_weights"] = json.dumps(prevalence_weights.tolist())
            all_summaries.append(summary)

        write_json(
            {
                "dataset": dataset,
                "feature_universe_size": p,
                "feature_universe_definition": (
                    "The same fixed preselected feature panel was present in every repeat. "
                    "Every feature received an IG score for every target class in every repeat."
                ),
                "class_counts": {
                    str(index): int(value) for index, value in enumerate(class_counts)
                },
                "prevalence_weights": prevalence_weights.tolist(),
                "source_class_specific_files": source_files,
                "source_run_metadata_files": metadata_files,
                "stability_rank_definition": (
                    "Lexicographic ordering by Top-k selection counts (smallest k to largest k), "
                    "then lower mean rank, lower median rank, and higher mean absolute IG. "
                    "A separate median_rank_only_rank is also provided."
                ),
                "chance_adjusted_jaccard_definition": (
                    "J_adj=(J-J0)/(1-J0), where J0=k/(2P-k) is the plug-in chance expectation "
                    "based on the expected random intersection."
                ),
                "kuncheva_index_definition": (
                    "K=(rP-k^2)/(k(P-k)), where r is the observed intersection. K has expected "
                    "value zero for independently sampled sets of size k."
                ),
            },
            dataset_output / f"{dataset}_IG_reanalysis_manifest.json",
        )

    summary_frame = pd.DataFrame(all_summaries)
    summary_frame.to_csv(output_root / "IG_reanalysis_summary_all_datasets.csv", index=False)
    pd.DataFrame(protocol_rows).to_csv(output_root / "IG_protocol_by_repeat.csv", index=False)

    print(f"Saved IG stability reanalysis to {output_root}")
    for _, row in summary_frame.iterrows():
        print(
            f"{row['dataset']} / {row['aggregation']}: "
            f"Spearman={row['spearman_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
