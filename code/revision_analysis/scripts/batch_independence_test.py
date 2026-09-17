from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from common import DATASET_ORDER, ensure_dir, read_fixed_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that each sample's MCOF output and gradient are independent "
            "of other samples placed in the same inference batch."
        )
    )
    parser.add_argument("--toolkit_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--mcof_main_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--samples_per_test", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    toolkit_root = Path(args.toolkit_root)
    src = str((toolkit_root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)

    from train_utils_v2 import (  # type: ignore
        OmicsPreprocessor,
        load_checkpoint,
        read_multiomics_dataset,
        restore_model_from_checkpoint,
    )

    data_root = Path(args.data_root)
    split_root = Path(args.split_root)
    mcof_main_root = Path(args.mcof_main_root)
    output_root = ensure_dir(args.output_root)
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")

    rows: List[Dict[str, object]] = []

    for dataset in DATASET_ORDER:
        omics_list, labels, _ = read_multiomics_dataset(data_root / dataset)
        split_table = read_fixed_splits(
            split_root / f"{dataset}_outer_splits.csv", n_samples=len(labels), repeats=5
        )
        for repeat in range(1, 6):
            checkpoint_path = (
                mcof_main_root / dataset / "mcof_se" / f"repeat_{repeat}" / "best_model.pt"
            )
            checkpoint = load_checkpoint(checkpoint_path, map_location=device)
            model = restore_model_from_checkpoint(checkpoint, device=device)
            preprocessor = OmicsPreprocessor.from_state_dict(checkpoint["preprocessor_state"])
            transformed = preprocessor.transform(omics_list)
            x_all = np.concatenate(transformed, axis=1).astype(np.float32, copy=False)
            test_indices = (
                split_table.loc[
                    split_table["repeat"].eq(repeat) & split_table["subset"].eq("test"),
                    "sample_index",
                ]
                .astype(int)
                .to_numpy()
            )
            selected = test_indices[: min(args.samples_per_test, len(test_indices))]
            batch = torch.from_numpy(x_all[selected]).to(device)
            model.eval()

            with torch.inference_mode():
                batch_logits = model(batch).detach()
                reversed_logits = model(batch.flip(0)).flip(0).detach()

            alone_logits = []
            with torch.inference_mode():
                for row_index in range(len(batch)):
                    alone_logits.append(model(batch[row_index : row_index + 1]).detach())
            alone = torch.cat(alone_logits, dim=0)

            max_alone_difference = float((batch_logits - alone).abs().max().item())
            max_order_difference = float((batch_logits - reversed_logits).abs().max().item())

            gradient_input = batch.detach().clone().requires_grad_(True)
            logits = model(gradient_input)
            target_class = int(torch.argmax(logits[0]).item())
            target_score = logits[0, target_class]
            gradient = torch.autograd.grad(target_score, gradient_input, retain_graph=False)[0]
            if len(batch) > 1:
                max_other_sample_gradient = float(gradient[1:].abs().max().item())
            else:
                max_other_sample_gradient = 0.0
            own_sample_gradient_l1 = float(gradient[0].abs().sum().item())

            passed = (
                max_alone_difference <= args.tolerance
                and max_order_difference <= args.tolerance
                and max_other_sample_gradient <= args.tolerance
            )
            rows.append(
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "checkpoint": str(checkpoint_path),
                    "n_samples_in_test_batch": len(batch),
                    "target_class_for_gradient_test": target_class,
                    "max_abs_logit_difference_alone_vs_batch": max_alone_difference,
                    "max_abs_logit_difference_original_vs_reversed_batch": max_order_difference,
                    "max_abs_gradient_on_other_samples": max_other_sample_gradient,
                    "own_sample_gradient_l1": own_sample_gradient_l1,
                    "tolerance": args.tolerance,
                    "passed": bool(passed),
                }
            )
            if not passed:
                raise AssertionError(
                    f"Batch-independence test failed for {dataset} repeat {repeat}: "
                    f"alone={max_alone_difference}, order={max_order_difference}, "
                    f"other_grad={max_other_sample_gradient}"
                )

            del model, checkpoint, batch, gradient_input, logits, gradient
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    result.to_csv(output_root / "batch_independence_all_checkpoints.csv", index=False)
    summary = {
        "n_checkpoints_tested": int(len(result)),
        "all_passed": bool(result["passed"].all()),
        "maximum_alone_vs_batch_difference": float(
            result["max_abs_logit_difference_alone_vs_batch"].max()
        ),
        "maximum_batch_order_difference": float(
            result["max_abs_logit_difference_original_vs_reversed_batch"].max()
        ),
        "maximum_cross_sample_gradient": float(
            result["max_abs_gradient_on_other_samples"].max()
        ),
        "interpretation": (
            "Each sample was evaluated alone, in its original batch, and after reversing "
            "the batch order. The gradient of sample 0's selected logit was also calculated "
            "with respect to every batch row. A zero gradient on other rows confirms absence "
            "of cross-sample information flow in the implemented model."
        ),
    }
    (output_root / "batch_independence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
