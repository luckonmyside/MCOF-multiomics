from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_utils_v2 import DEFAULT_DEVICE, TrainingConfig, run_repeated_holdout_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized MCOF training pipeline with repeated holdout + inner CV.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing 1_all.csv, 2_all.csv, ..., labels_all.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and metrics")
    parser.add_argument(
        "--model_type",
        type=str,
        default="mcof_attn",
        choices=["mcof_attn", "mcof", "mcof_se", "mca", "cnn"],
        help="mcof_attn: true cross-omics attention; mcof/mcof_se: SE-style channel attention; mca: no convolution; cnn: no attention",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--conv_channels", type=int, default=64)
    parser.add_argument("--conv_kernel_size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--gradient_clip_norm", type=float, default=5.0)
    parser.add_argument("--monitor_metric", type=str, default="auto")
    parser.add_argument("--impute_strategy", type=str, default="median", choices=["mean", "median"])
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "robust", "none"])
    parser.add_argument("--max_missing_rate", type=float, default=1.0)
    parser.add_argument("--max_zero_rate", type=float, default=1.0)
    parser.add_argument("--select_k", type=int, default=None)
    parser.add_argument("--selector", type=str, default="none", choices=["none", "f_classif", "variance"])
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu") if args.cpu else DEFAULT_DEVICE
    cfg = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        test_size=args.test_size,
        repeats=args.repeats,
        inner_folds=args.inner_folds,
        random_seed=args.random_seed,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        conv_channels=args.conv_channels,
        conv_kernel_size=args.conv_kernel_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gradient_clip_norm=args.gradient_clip_norm,
        monitor_metric=args.monitor_metric,
        impute_strategy=args.impute_strategy,
        scaler=args.scaler,
        max_missing_rate=args.max_missing_rate,
        max_zero_rate=args.max_zero_rate,
        select_k=args.select_k,
        selector=args.selector,
    )
    result = run_repeated_holdout_experiment(cfg, device=device)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"Saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
