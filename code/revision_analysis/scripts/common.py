from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

DATASET_ORDER = ["BRCA", "STAD", "ROSMAP", "SCZ"]
MODEL_ORDER = ["DIABLO", "KNN", "SVM", "NB", "RF", "MOGONET", "MCOF"]

DATASET_CONFIG: Dict[str, Dict[str, Any]] = {
    "BRCA": {
        "omics_names": ["RNA", "miRNA"],
        "class_names": ["Basal-like", "HER2-enriched", "Luminal A", "Luminal B"],
        "primary_metric": "f1_macro",
        "metrics": ["acc", "f1_macro", "f1_weighted", "auc_macro_ovr", "auc_weighted_ovr"],
    },
    "STAD": {
        "omics_names": ["RNA", "miRNA"],
        "class_names": ["CIN", "GS", "MSI"],
        "primary_metric": "f1_macro",
        "metrics": ["acc", "f1_macro", "f1_weighted", "auc_macro_ovr", "auc_weighted_ovr"],
    },
    "ROSMAP": {
        "omics_names": ["mRNA", "miRNA", "DNA_methylation"],
        "class_names": ["Control", "Alzheimer disease"],
        "primary_metric": "mcc",
        "metrics": ["acc", "f1", "auc", "mcc"],
    },
    "SCZ": {
        "omics_names": ["Protein", "Metabolite"],
        "class_names": ["Control", "Schizophrenia"],
        "primary_metric": "mcc",
        "metrics": ["acc", "f1", "auc", "mcc"],
    },
}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_fixed_splits(path: str | Path, n_samples: int, repeats: int = 5) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"repeat", "sample_index", "subset"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing split columns: {sorted(missing)}")

    frame = frame.copy()
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
    frame["sample_index"] = pd.to_numeric(frame["sample_index"], errors="raise").astype(int)
    frame["subset"] = frame["subset"].astype(str).str.lower().str.strip()

    unknown = set(frame["subset"]).difference({"train", "test"})
    if unknown:
        raise ValueError(f"{path} contains unsupported subset labels: {sorted(unknown)}")

    expected_repeats = set(range(1, repeats + 1))
    observed_repeats = set(frame["repeat"].unique())
    if observed_repeats != expected_repeats:
        raise ValueError(
            f"{path}: expected repeats {sorted(expected_repeats)}, found {sorted(observed_repeats)}"
        )

    for repeat in range(1, repeats + 1):
        current = frame[frame["repeat"].eq(repeat)]
        if current["sample_index"].duplicated().any():
            raise ValueError(f"{path}: repeat {repeat} contains duplicated sample indices")
        indices = current["sample_index"].to_numpy()
        if len(indices) != n_samples:
            raise ValueError(
                f"{path}: repeat {repeat} has {len(indices)} rows; expected {n_samples}"
            )
        if indices.min() < 0 or indices.max() >= n_samples:
            raise IndexError(f"{path}: repeat {repeat} contains out-of-range indices")
        if set(indices.tolist()) != set(range(n_samples)):
            raise ValueError(f"{path}: repeat {repeat} does not partition every sample exactly once")
        if not (current["subset"].eq("train").any() and current["subset"].eq("test").any()):
            raise ValueError(f"{path}: repeat {repeat} lacks train or test samples")

    return frame.sort_values(["repeat", "subset", "sample_index"]).reset_index(drop=True)


def infer_probability_columns(frame: pd.DataFrame) -> List[str]:
    candidates: List[Tuple[int, str]] = []
    for column in frame.columns:
        match = re.fullmatch(r"(?:prob|score)_class_(\d+)", str(column))
        if match:
            candidates.append((int(match.group(1)), str(column)))
    return [column for _, column in sorted(candidates)]


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """Two-sided exact sign-flip randomization p-value for paired differences.

    The statistic is the absolute mean paired difference. Zeros are retained;
    all 2^n sign assignments are enumerated, which is trivial for n=5.
    """

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("differences must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("differences contain non-finite values")

    observed = abs(float(values.mean()))
    permuted = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(values)):
        permuted.append(abs(float(np.mean(values * np.asarray(signs)))))
    permuted_array = np.asarray(permuted)
    return float(np.mean(permuted_array >= observed - 1e-15))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if len(values) == 0:
        return values
    if np.any((values < 0) | (values > 1) | ~np.isfinite(values)):
        raise ValueError("p_values must be finite and within [0, 1]")

    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running_max = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        current = min(1.0, (m - rank) * values[index])
        running_max = max(running_max, current)
        adjusted_sorted[rank] = running_max

    adjusted = np.empty(len(values), dtype=float)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted


def mean_sd(values: Sequence[float], digits: int = 3) -> str:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        sd = 0.0
    else:
        sd = float(array.std(ddof=1))
    return f"{array.mean():.{digits}f} ± {sd:.{digits}f}"


def parse_numeric_labels(series: pd.Series, dataset: str) -> np.ndarray:
    """Convert prediction labels to encoded class integers.

    Numeric labels are used directly. A small set of human-readable aliases is
    accepted for resilience to R factor output. Unknown labels fail loudly.
    """

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int).to_numpy()

    aliases: Mapping[str, Mapping[str, int]] = {
        "BRCA": {
            "basal-like": 0,
            "basal": 0,
            "her2-enriched": 1,
            "her2": 1,
            "luminal a": 2,
            "luma": 2,
            "luminal b": 3,
            "lumb": 3,
        },
        "STAD": {"cin": 0, "gs": 1, "msi": 2},
        "ROSMAP": {
            "control": 0,
            "normal control": 0,
            "0": 0,
            "ad": 1,
            "alzheimer disease": 1,
            "alzheimer's disease": 1,
            "1": 1,
        },
        "SCZ": {
            "control": 0,
            "normal control": 0,
            "0": 0,
            "scz": 1,
            "schizophrenia": 1,
            "1": 1,
        },
    }
    mapping = aliases[dataset]
    normalized = series.astype(str).str.strip().str.lower()
    converted = normalized.map(mapping)
    if converted.isna().any():
        unknown = sorted(normalized[converted.isna()].unique().tolist())
        raise ValueError(f"{dataset}: unable to encode labels {unknown}")
    return converted.astype(int).to_numpy()


def write_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
