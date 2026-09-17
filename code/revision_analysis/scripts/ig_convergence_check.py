from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from common import DATASET_ORDER, ensure_dir, read_fixed_splits

try:
    from captum.attr import IntegratedGradients as CaptumIntegratedGradients  # type: ignore

    IntegratedGradients = CaptumIntegratedGradients
    IG_BACKEND = (
        "captum.attr.IntegratedGradients with method='gausslegendre'"
    )

except Exception:

    class IntegratedGradients:
        """Captum-compatible deterministic Gauss-Legendre IG fallback."""

        def __init__(self, model: torch.nn.Module) -> None:
            self.model = model

        @staticmethod
        def _select_target(outputs: torch.Tensor, target) -> torch.Tensor:
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            if outputs.ndim != 2:
                raise ValueError(
                    "Expected model outputs with shape "
                    f"(batch, classes), received {tuple(outputs.shape)}"
                )

            if isinstance(target, (int, np.integer)):
                return outputs[:, int(target)]

            target_tensor = torch.as_tensor(
                target,
                device=outputs.device,
                dtype=torch.long,
            )

            if target_tensor.ndim == 0:
                return outputs[:, int(target_tensor.item())]

            target_tensor = target_tensor.reshape(-1)

            if len(target_tensor) != len(outputs):
                raise ValueError(
                    "Per-sample target length does not match batch size: "
                    f"{len(target_tensor)} versus {len(outputs)}"
                )

            return outputs.gather(
                1,
                target_tensor.view(-1, 1),
            ).squeeze(1)

        def attribute(
            self,
            inputs: torch.Tensor,
            baselines: torch.Tensor | None = None,
            target=None,
            n_steps: int = 50,
            method: str = "gausslegendre",
            return_convergence_delta: bool = False,
            **unused_kwargs,
        ):
            if target is None:
                raise ValueError("A target class is required.")

            if int(n_steps) < 1:
                raise ValueError(
                    f"n_steps must be positive; received {n_steps}"
                )

            if method != "gausslegendre":
                raise ValueError(
                    "The manual fallback supports only "
                    "method='gausslegendre'."
                )

            self.model.eval()

            inputs_detached = inputs.detach()

            if baselines is None:
                baseline_detached = torch.zeros_like(inputs_detached)
            else:
                baseline_detached = baselines.detach().to(
                    device=inputs.device,
                    dtype=inputs.dtype,
                )

                if baseline_detached.shape != inputs_detached.shape:
                    baseline_detached = baseline_detached.expand_as(
                        inputs_detached
                    )

            difference = inputs_detached - baseline_detached

            # n-point Gauss-Legendre quadrature on [0, 1].
            nodes, weights = np.polynomial.legendre.leggauss(
                int(n_steps)
            )

            alphas = torch.as_tensor(
                (nodes + 1.0) / 2.0,
                device=inputs.device,
                dtype=inputs.dtype,
            )

            quadrature_weights = torch.as_tensor(
                weights / 2.0,
                device=inputs.device,
                dtype=inputs.dtype,
            )

            integrated_gradient = torch.zeros_like(inputs_detached)

            for alpha, weight in zip(
                alphas,
                quadrature_weights,
            ):
                interpolated = (
                    baseline_detached + alpha * difference
                ).detach()

                interpolated.requires_grad_(True)

                outputs = self.model(interpolated)

                selected_outputs = self._select_target(
                    outputs,
                    target,
                )

                gradient = torch.autograd.grad(
                    selected_outputs.sum(),
                    interpolated,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                integrated_gradient = (
                    integrated_gradient
                    + weight * gradient.detach()
                )

            attributions = difference * integrated_gradient

            if not return_convergence_delta:
                return attributions

            with torch.no_grad():
                output_at_inputs = self._select_target(
                    self.model(inputs_detached),
                    target,
                )

                output_at_baseline = self._select_target(
                    self.model(baseline_detached),
                    target,
                )

                attribution_sum = attributions.reshape(
                    attributions.shape[0],
                    -1,
                ).sum(dim=1)

                # Same completeness convention used by Captum:
                # sum(attributions) - [F(input) - F(baseline)].
                convergence_delta = (
                    attribution_sum
                    - (
                        output_at_inputs
                        - output_at_baseline
                    )
                )

            return attributions, convergence_delta


    IG_BACKEND = (
        "manual Integrated Gradients using deterministic "
        "n-point Gauss-Legendre quadrature"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check IG numerical convergence by comparing 25, 50, and 100-step "
            "rankings and completeness deltas on deterministic stratified "
            "held-out sample subsets."
        )
    )
    parser.add_argument("--toolkit_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--mcof_main_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[25, 50, 100])
    parser.add_argument("--samples_per_class", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--datasets", nargs="+", default=DATASET_ORDER)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def rank_vector(values: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(values)), -values))
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def spearman_from_ranks(first: np.ndarray, second: np.ndarray) -> float:
    return float(pd.Series(first).corr(pd.Series(second), method="spearman"))


def stratified_subset(test_indices: np.ndarray, labels: np.ndarray, per_class: int) -> np.ndarray:
    selected = []
    for class_index in sorted(np.unique(labels[test_indices])):
        current = test_indices[labels[test_indices] == class_index]
        selected.extend(current[: min(per_class, len(current))].tolist())
    return np.asarray(selected, dtype=int)


def main() -> None:
    args = parse_args()
    if IntegratedGradients is None:
        raise RuntimeError(
            "Captum is required for the convergence check. Activate the mcof environment "
            "used for the formal IG analysis or install captum in that environment."
        )
    if sorted(args.steps) != args.steps or len(set(args.steps)) != len(args.steps):
        raise ValueError("--steps must be unique and in increasing order")

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
    main_root = Path(args.mcof_main_root)
    output_root = ensure_dir(args.output_root)
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")

    comparison_rows: List[Dict[str, object]] = []
    delta_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []

    for dataset in args.datasets:
        omics_list, labels, _ = read_multiomics_dataset(data_root / dataset)
        split_table = read_fixed_splits(
            split_root / f"{dataset}_outer_splits.csv", n_samples=len(labels), repeats=5
        )
        n_classes = len(np.unique(labels))
        prevalence = np.bincount(labels, minlength=n_classes).astype(float)
        prevalence /= prevalence.sum()

        for repeat in range(1, 6):
            checkpoint_path = main_root / dataset / "mcof_se" / f"repeat_{repeat}" / "best_model.pt"
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
            selected_indices = stratified_subset(test_indices, labels, args.samples_per_class)
            for index in selected_indices:
                sample_rows.append(
                    {
                        "dataset": dataset,
                        "repeat": repeat,
                        "sample_index": int(index),
                        "class": int(labels[index]),
                    }
                )

            x_selected = torch.from_numpy(x_all[selected_indices]).to(device)
            baseline = torch.zeros_like(x_selected)
            ig = IntegratedGradients(model)
            aggregated_by_step: Dict[int, np.ndarray] = {}

            for step in args.steps:
                class_importances = []
                for target_class in range(n_classes):
                    attributions = []
                    deltas = []
                    for start in range(0, len(x_selected), args.batch_size):
                        current = x_selected[start : start + args.batch_size]
                        current_baseline = baseline[start : start + args.batch_size]
                        model.zero_grad(set_to_none=True)
                        attr, convergence_delta = ig.attribute(
                            current,
                            baselines=current_baseline,
                            target=target_class,
                            n_steps=step,
                            method="gausslegendre",
                            return_convergence_delta=True,
                        )
                        attributions.append(attr.detach().cpu().numpy())
                        deltas.append(convergence_delta.detach().cpu().numpy())
                    attribution_array = np.concatenate(attributions, axis=0)
                    delta_array = np.concatenate(deltas, axis=0)
                    class_importances.append(np.mean(np.abs(attribution_array), axis=0))
                    delta_rows.append(
                        {
                            "dataset": dataset,
                            "repeat": repeat,
                            "steps": step,
                            "target_class": target_class,
                            "n_samples": len(x_selected),
                            "mean_abs_completeness_delta": float(np.mean(np.abs(delta_array))),
                            "max_abs_completeness_delta": float(np.max(np.abs(delta_array))),
                        }
                    )
                class_matrix = np.vstack(class_importances)
                # The original overall ranking used prevalence weights. Equal
                # weighting is also stored for sensitivity analysis.
                aggregated_by_step[step] = prevalence @ class_matrix
                equal_values = class_matrix.mean(axis=0)
                np.save(
                    ensure_dir(output_root / "rank_vectors" / dataset / f"repeat_{repeat}")
                    / f"equal_weighted_importance_steps_{step}.npy",
                    equal_values,
                )
                np.save(
                    output_root
                    / "rank_vectors"
                    / dataset
                    / f"repeat_{repeat}"
                    / f"prevalence_weighted_importance_steps_{step}.npy",
                    aggregated_by_step[step],
                )

            for first, second in zip(args.steps[:-1], args.steps[1:]):
                first_values = aggregated_by_step[first]
                second_values = aggregated_by_step[second]
                first_ranks = rank_vector(first_values)
                second_ranks = rank_vector(second_values)
                k = min(args.top_k, len(first_values))
                first_top = set(np.argsort(first_values)[::-1][:k].tolist())
                second_top = set(np.argsort(second_values)[::-1][:k].tolist())
                jaccard = len(first_top & second_top) / len(first_top | second_top)
                relative_l1 = float(
                    np.sum(np.abs(second_values - first_values))
                    / max(np.sum(np.abs(second_values)), 1e-12)
                )
                comparison_rows.append(
                    {
                        "dataset": dataset,
                        "repeat": repeat,
                        "steps_first": first,
                        "steps_second": second,
                        "n_features": len(first_values),
                        "spearman_rank_correlation": spearman_from_ranks(first_ranks, second_ranks),
                        "top_k": k,
                        "top_k_jaccard": jaccard,
                        "relative_l1_change": relative_l1,
                    }
                )

            del model, checkpoint, x_selected, baseline, ig
            if device.type == "cuda":
                torch.cuda.empty_cache()

    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(output_root / "IG_step_convergence_comparisons.csv", index=False)
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(output_root / "IG_completeness_deltas.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(output_root / "IG_convergence_sample_manifest.csv", index=False)

    summary = (
        comparisons.groupby(["dataset", "steps_first", "steps_second"], as_index=False)
        .agg(
            spearman_mean=("spearman_rank_correlation", "mean"),
            spearman_sd=("spearman_rank_correlation", "std"),
            top_k_jaccard_mean=("top_k_jaccard", "mean"),
            top_k_jaccard_sd=("top_k_jaccard", "std"),
            relative_l1_change_mean=("relative_l1_change", "mean"),
            relative_l1_change_sd=("relative_l1_change", "std"),
        )
    )
    summary.to_csv(output_root / "IG_step_convergence_summary.csv", index=False)
    delta_summary = (
        deltas.groupby(["dataset", "steps"], as_index=False)
        .agg(
            mean_abs_completeness_delta=("mean_abs_completeness_delta", "mean"),
            max_abs_completeness_delta=("max_abs_completeness_delta", "max"),
        )
    )
    delta_summary.to_csv(output_root / "IG_completeness_summary.csv", index=False)

    protocol = {
        "integration_method": IG_BACKEND,
        "steps": args.steps,
        "baseline": "all-zero vector in the checkpoint-specific training-standardized feature space",
        "sample_selection": (
            f"Deterministic first {args.samples_per_class} held-out samples per observed class "
            "within each repeat, ordered by fixed sample_index"
        ),
        "targets": "every output class",
        "aggregation_for_step_comparison": "original class-prevalence-weighted overall importance",
        "random_seed": "not applicable; the path and selected sample indices are deterministic",
    }
    (output_root / "IG_convergence_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(f"Saved IG convergence checks to {output_root}")


if __name__ == "__main__":
    main()
