from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


DATASET_ORDER = ["BRCA", "STAD", "ROSMAP", "SCZ"]


def read_dataset(dataset_dir: Path):
    """Read 1_all.csv, 2_all.csv, optional later views, and labels."""
    omics_list = []
    view_index = 1

    while True:
        data_file = dataset_dir / f"{view_index}_all.csv"

        if not data_file.is_file():
            break

        matrix = pd.read_csv(
            data_file,
            header=None,
        ).to_numpy(dtype=np.float64)

        if matrix.ndim != 2:
            raise ValueError(
                f"{data_file}: expected a two-dimensional matrix."
            )

        omics_list.append(matrix)
        view_index += 1

    if not omics_list:
        raise FileNotFoundError(
            f"No omics data files found in {dataset_dir}"
        )

    labels_file = dataset_dir / "labels_all.csv"

    raw_labels = pd.read_csv(
        labels_file,
        header=None,
    ).iloc[:, 0]

    labels, levels = pd.factorize(
        raw_labels,
        sort=True,
    )

    labels = labels.astype(np.int64)

    for current_view, matrix in enumerate(
        omics_list,
        start=1,
    ):
        if matrix.shape[0] != len(labels):
            raise ValueError(
                f"View {current_view}: "
                f"{matrix.shape[0]} samples, "
                f"but labels contain {len(labels)} samples."
            )

    label_map = {
        str(level): int(index)
        for index, level in enumerate(levels)
    }

    return omics_list, labels, label_map


def train_only_preprocess(
    matrix: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
):
    """Median imputation and standardization fitted on train only."""
    train = np.asarray(
        matrix[train_idx],
        dtype=np.float64,
    ).copy()

    test = np.asarray(
        matrix[test_idx],
        dtype=np.float64,
    ).copy()

    medians = np.nanmedian(train, axis=0)
    invalid_medians = ~np.isfinite(medians)
    medians[invalid_medians] = 0.0

    train_missing = ~np.isfinite(train)
    test_missing = ~np.isfinite(test)

    if train_missing.any():
        train[train_missing] = np.take(
            medians,
            np.where(train_missing)[1],
        )

    if test_missing.any():
        test[test_missing] = np.take(
            medians,
            np.where(test_missing)[1],
        )

    means = train.mean(axis=0)
    scales = train.std(axis=0, ddof=0)

    invalid_scales = (
        ~np.isfinite(scales)
        | (scales < 1e-12)
    )
    scales[invalid_scales] = 1.0

    train = (train - means) / scales
    test = (test - means) / scales

    diagnostics = {
        "n_features": int(train.shape[1]),
        "train_imputed_values": int(train_missing.sum()),
        "test_imputed_values": int(test_missing.sum()),
        "all_missing_train_features": int(
            invalid_medians.sum()
        ),
        "zero_variance_train_features": int(
            invalid_scales.sum()
        ),
    }

    return (
        train.astype(np.float32, copy=False),
        test.astype(np.float32, copy=False),
        diagnostics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark existing MOGONET checkpoints without retraining."
    )
    parser.add_argument("--mogonet_code_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timing_repeats", type=int, default=100)
    return parser.parse_args()


def synchronize() -> None:
    torch.cuda.synchronize()


def benchmark_cuda(
    forward_once: Callable[[], None],
    warmup: int,
    timing_repeats: int,
) -> Dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            forward_once()
        synchronize()

        base_allocated = torch.cuda.memory_allocated()
        base_reserved = torch.cuda.memory_reserved()

        torch.cuda.reset_peak_memory_stats()
        forward_once()
        synchronize()

        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

        elapsed_ms: List[float] = []
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(timing_repeats):
            starter.record()
            forward_once()
            ender.record()
            synchronize()
            elapsed_ms.append(float(starter.elapsed_time(ender)))

    values = np.asarray(elapsed_ms, dtype=np.float64)

    return {
        "forward_testset_ms_mean": float(values.mean()),
        "forward_testset_ms_sd": float(values.std(ddof=1)),
        "forward_testset_ms_median": float(np.median(values)),
        "forward_testset_ms_min": float(values.min()),
        "forward_testset_ms_max": float(values.max()),
        "baseline_allocated_mb": float(base_allocated / 1024**2),
        "baseline_reserved_mb": float(base_reserved / 1024**2),
        "peak_allocated_mb": float(peak_allocated / 1024**2),
        "peak_reserved_mb": float(peak_reserved / 1024**2),
        "incremental_peak_allocated_mb": float(
            (peak_allocated - base_allocated) / 1024**2
        ),
    }


def summarize(df: pd.DataFrame, output_root: Path) -> None:
    numeric_columns = [
        "total_parameters",
        "trainable_parameters",
        "parameter_memory_fp32_mb",
        "checkpoint_size_mb",
        "n_test",
        "forward_testset_ms_mean",
        "forward_testset_ms_sd",
        "latency_ms_per_sample",
        "throughput_samples_per_second",
        "baseline_allocated_mb",
        "baseline_reserved_mb",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "incremental_peak_allocated_mb",
    ]

    rows = []
    for dataset in DATASET_ORDER:
        group = df[df["dataset"].eq(dataset)]
        if len(group) != 5:
            raise ValueError(
                f"{dataset}: expected 5 MOGONET repeats, found {len(group)}"
            )

        for metric in numeric_columns:
            values = pd.to_numeric(group[metric], errors="raise")
            rows.append(
                {
                    "dataset": dataset,
                    "model": "MOGONET",
                    "metric": metric,
                    "n_repeats": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "mean_sd": f"{values.mean():.6f} ± {values.std(ddof=1):.6f}",
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        output_root / "MOGONET_efficiency_summary_long.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this script inside a GPU allocation."
        )

    device = torch.device("cuda")
    mogonet_code_root = Path(args.mogonet_code_root).resolve()
    sys.path.insert(0, str(mogonet_code_root))

    from models import init_model_dict  # type: ignore
    from utils import gen_test_adj_mat_tensor  # type: ignore

    data_root = Path(args.data_root)
    split_root = Path(args.split_root)
    result_root = Path(args.result_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))
    print("MOGONET code root:", mogonet_code_root)

    rows = []

    for dataset in DATASET_ORDER:
        omics_all, labels, _ = read_dataset(
            data_root / dataset
        )
        split_table = pd.read_csv(
            split_root / f"{dataset}_outer_splits.csv"
        )

        for repeat in range(1, 6):
            run_dir = (
                result_root
                / dataset
                / "MOGONET"
                / f"repeat_{repeat}"
            )
            metrics_path = run_dir / "metrics_test.json"
            model_dir = run_dir / "models"

            if not metrics_path.is_file():
                raise FileNotFoundError(metrics_path)

            with metrics_path.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)

            current_split = split_table[
                split_table["repeat"].eq(repeat)
            ]
            train_idx = (
                current_split.loc[
                    current_split["subset"].eq("train"),
                    "sample_index",
                ]
                .astype(int)
                .to_numpy()
            )
            test_idx = (
                current_split.loc[
                    current_split["subset"].eq("test"),
                    "sample_index",
                ]
                .astype(int)
                .to_numpy()
            )

            train_matrices = []
            test_matrices = []
            for matrix in omics_all:
                train_matrix, test_matrix, _ = (
                    train_only_preprocess(
                        matrix,
                        train_idx,
                        test_idx,
                    )
                )
                train_matrices.append(train_matrix)
                test_matrices.append(test_matrix)

            all_tensors = [
                torch.from_numpy(
                    np.concatenate(
                        [train_matrix, test_matrix],
                        axis=0,
                    )
                ).to(device)
                for train_matrix, test_matrix in zip(
                    train_matrices,
                    test_matrices,
                )
            ]

            adaptive_thresholds = [
                float(value)
                for value in saved["adaptive_thresholds"]
            ]
            trte_idx = {
                "tr": list(range(len(train_idx))),
                "te": list(
                    range(
                        len(train_idx),
                        len(train_idx) + len(test_idx),
                    )
                ),
            }

            test_adjacencies = [
                gen_test_adj_mat_tensor(
                    all_tensor,
                    trte_idx,
                    threshold,
                )
                for all_tensor, threshold in zip(
                    all_tensors,
                    adaptive_thresholds,
                )
            ]

            num_views = int(saved["num_views"])
            num_classes = int(saved["num_classes"])
            input_dims = [int(value) for value in saved["input_dims"]]
            hidden_dims = [int(value) for value in saved["hidden_dims"]]
            vcdn_dimension = int(num_classes ** num_views)

            model_dict = init_model_dict(
                num_views,
                num_classes,
                input_dims,
                hidden_dims,
                vcdn_dimension,
            )

            for model_name, model in model_dict.items():
                checkpoint_path = model_dir / f"{model_name}.pt"
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(checkpoint_path)

                state_dict = torch.load(
                    checkpoint_path,
                    map_location=device,
                    weights_only=True,
                )
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()

            total_parameters = int(
                sum(
                    parameter.numel()
                    for model in model_dict.values()
                    for parameter in model.parameters()
                )
            )
            trainable_parameters = int(
                sum(
                    parameter.numel()
                    for model in model_dict.values()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
            )
            parameter_memory_fp32_mb = (
                total_parameters * 4 / 1024**2
            )
            checkpoint_size_mb = float(
                sum(path.stat().st_size for path in model_dir.glob("*.pt"))
                / 1024**2
            )

            test_positions = list(
                range(
                    len(train_idx),
                    len(train_idx) + len(test_idx),
                )
            )

            def forward_once() -> None:
                view_logits = []
                for view_index in range(num_views):
                    view_logits.append(
                        model_dict[f"C{view_index + 1}"](
                            model_dict[f"E{view_index + 1}"](
                                all_tensors[view_index],
                                test_adjacencies[view_index],
                            )
                        )
                    )

                if num_views >= 2:
                    logits = model_dict["C"](view_logits)
                else:
                    logits = view_logits[0]

                _ = F.softmax(
                    logits[test_positions, :],
                    dim=1,
                )

            timing = benchmark_cuda(
                forward_once,
                warmup=args.warmup,
                timing_repeats=args.timing_repeats,
            )

            n_test = int(len(test_idx))
            latency = (
                timing["forward_testset_ms_mean"] / n_test
            )
            throughput = (
                n_test
                / (timing["forward_testset_ms_mean"] / 1000.0)
            )

            row = {
                "dataset": dataset,
                "model": "MOGONET",
                "repeat": repeat,
                "model_dir": str(model_dir),
                "n_test": n_test,
                "warmup": args.warmup,
                "timing_repeats": args.timing_repeats,
                "num_views": num_views,
                "num_classes": num_classes,
                "input_dims": json.dumps(input_dims),
                "hidden_dims": json.dumps(hidden_dims),
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "parameter_memory_fp32_mb": parameter_memory_fp32_mb,
                "checkpoint_size_mb": checkpoint_size_mb,
                **timing,
                "latency_ms_per_sample": latency,
                "throughput_samples_per_second": throughput,
            }
            rows.append(row)

            print(
                f"MOGONET {dataset} repeat {repeat}: "
                f"params={total_parameters:,}; "
                f"forward={timing['forward_testset_ms_mean']:.4f} ms/test set; "
                f"peak={timing['peak_allocated_mb']:.2f} MB"
            )

            del model_dict
            del all_tensors
            del test_adjacencies
            gc.collect()
            torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    assert len(result) == 20
    assert (
        result.groupby(["dataset", "model"])
        .size()
        .eq(5)
        .all()
    )

    repeat_file = output_root / "MOGONET_efficiency_by_repeat.csv"
    result.to_csv(repeat_file, index=False)
    summarize(result, output_root)

    print("\nSaved:")
    print(repeat_file)
    print(output_root / "MOGONET_efficiency_summary_long.csv")


if __name__ == "__main__":
    main()
