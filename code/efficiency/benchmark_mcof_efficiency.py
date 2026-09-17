from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


DATASET_ORDER = ["BRCA", "STAD", "ROSMAP", "SCZ"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark existing MCOF-se checkpoints without retraining."
    )
    parser.add_argument("--mcof_code_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
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
                f"{dataset}: expected 5 MCOF repeats, found {len(group)}"
            )

        for metric in numeric_columns:
            values = pd.to_numeric(group[metric], errors="raise")
            rows.append(
                {
                    "dataset": dataset,
                    "model": "MCOF",
                    "metric": metric,
                    "n_repeats": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "mean_sd": f"{values.mean():.6f} ± {values.std(ddof=1):.6f}",
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(
        output_root / "MCOF_efficiency_summary_long.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this script inside a GPU allocation."
        )

    device = torch.device("cuda")
    mcof_code_root = Path(args.mcof_code_root).resolve()
    sys.path.insert(0, str(mcof_code_root))

    from train_utils_v2 import (  # type: ignore
        OmicsPreprocessor,
        load_checkpoint,
        read_multiomics_dataset,
        restore_model_from_checkpoint,
    )

    data_root = Path(args.data_root)
    split_root = Path(args.split_root)
    checkpoint_root = Path(args.checkpoint_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))
    print("MCOF code root:", mcof_code_root)

    rows = []

    for dataset in DATASET_ORDER:
        omics_list, labels, _ = read_multiomics_dataset(
            data_root / dataset
        )
        split_table = pd.read_csv(
            split_root / f"{dataset}_outer_splits.csv"
        )

        for repeat in range(1, 6):
            checkpoint_path = (
                checkpoint_root
                / dataset
                / "mcof_se"
                / f"repeat_{repeat}"
                / "best_model.pt"
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)

            checkpoint = load_checkpoint(
                checkpoint_path,
                map_location=device,
            )
            model = restore_model_from_checkpoint(
                checkpoint,
                device=device,
            )
            model.eval()

            preprocessor = OmicsPreprocessor.from_state_dict(
                checkpoint["preprocessor_state"]
            )
            transformed = preprocessor.transform(omics_list)

            current_split = split_table[
                split_table["repeat"].eq(repeat)
            ]
            test_idx = (
                current_split.loc[
                    current_split["subset"].eq("test"),
                    "sample_index",
                ]
                .astype(int)
                .to_numpy()
            )

            x_all = np.concatenate(
                transformed,
                axis=1,
            ).astype(np.float32, copy=False)
            x_test = torch.from_numpy(
                x_all[test_idx]
            ).to(device)

            total_parameters = int(
                sum(parameter.numel() for parameter in model.parameters())
            )
            trainable_parameters = int(
                sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
            )
            parameter_memory_fp32_mb = (
                total_parameters * 4 / 1024**2
            )
            checkpoint_size_mb = (
                checkpoint_path.stat().st_size / 1024**2
            )

            def forward_once() -> None:
                for start in range(0, len(x_test), args.batch_size):
                    batch = x_test[
                        start : start + args.batch_size
                    ]
                    logits = model(batch)
                    _ = torch.softmax(logits, dim=1)

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

            model_config = checkpoint.get("model_config", {})
            row = {
                "dataset": dataset,
                "model": "MCOF",
                "repeat": repeat,
                "checkpoint": str(checkpoint_path),
                "n_test": n_test,
                "batch_size": args.batch_size,
                "warmup": args.warmup,
                "timing_repeats": args.timing_repeats,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "parameter_memory_fp32_mb": parameter_memory_fp32_mb,
                "checkpoint_size_mb": checkpoint_size_mb,
                "model_config": json.dumps(
                    model_config,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                **timing,
                "latency_ms_per_sample": latency,
                "throughput_samples_per_second": throughput,
            }
            rows.append(row)

            print(
                f"MCOF {dataset} repeat {repeat}: "
                f"params={total_parameters:,}; "
                f"forward={timing['forward_testset_ms_mean']:.4f} ms/test set; "
                f"peak={timing['peak_allocated_mb']:.2f} MB"
            )

            del model
            del checkpoint
            del preprocessor
            del x_test
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

    repeat_file = output_root / "MCOF_efficiency_by_repeat.csv"
    result.to_csv(repeat_file, index=False)
    summarize(result, output_root)

    print("\nSaved:")
    print(repeat_file)
    print(output_root / "MCOF_efficiency_summary_long.csv")


if __name__ == "__main__":
    main()
