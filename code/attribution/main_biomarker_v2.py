from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

from train_utils_v2 import OmicsPreprocessor, load_checkpoint, read_multiomics_dataset, restore_model_from_checkpoint


try:
    from captum.attr import IntegratedGradients  # type: ignore
except Exception:
    IntegratedGradients = None


def _iter_slices(n: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


class ManualIntegratedGradients:
    """Lightweight fallback when captum is unavailable."""

    def __init__(self, model: torch.nn.Module, steps: int = 50) -> None:
        self.model = model
        self.steps = steps

    def attribute(self, inputs: torch.Tensor, baselines: torch.Tensor, target: int) -> torch.Tensor:
        self.model.eval()
        alphas = torch.linspace(0.0, 1.0, steps=self.steps + 1, device=inputs.device)[1:]
        total_grad = torch.zeros_like(inputs)
        diff = inputs - baselines

        for alpha in alphas:
            x = baselines + alpha * diff
            x.requires_grad_(True)
            logits = self.model(x)
            score = logits[:, target].sum()
            grad = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]
            total_grad += grad.detach()

        avg_grad = total_grad / float(self.steps)
        return diff * avg_grad



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated Gradients biomarker ranking for optimized MCOF checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt from the optimized pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing 1_all.csv, 2_all.csv, ..., labels_all.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save attribution tables")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--ig_steps", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model = restore_model_from_checkpoint(checkpoint, device=device)
    preprocessor = OmicsPreprocessor.from_state_dict(checkpoint["preprocessor_state"])

    omics_list, labels, _raw_feature_names = read_multiomics_dataset(args.data_dir)

    transformed = preprocessor.transform(omics_list)
    x_all = np.concatenate(transformed, axis=1).astype(np.float32)

    # Use biological feature annotations from input data
    if _raw_feature_names is not None:
        feature_names = [
            name
            for sub in _raw_feature_names
            for name in sub
        ]
    else:
        feature_names_nested = checkpoint.get("feature_names")
        if feature_names_nested is None:
            raise ValueError("No feature names available.")

        feature_names = [
            name
            for sub in feature_names_nested
            for name in sub
        ]

    print("First 10 biological features:")
    print(feature_names[:10])
    print("Total features:", len(feature_names))

    inputs = torch.from_numpy(x_all).to(device)
    
    baseline = torch.zeros_like(inputs)
    y = labels.astype(int)
    num_classes = int(checkpoint["model_config"]["num_classes"])

    ig = IntegratedGradients(model) if IntegratedGradients is not None else ManualIntegratedGradients(model, steps=args.ig_steps)
    class_scores: Dict[int, np.ndarray] = {}

    for class_idx in range(num_classes):
        batch_attr = []
        for sl in _iter_slices(len(inputs), args.batch_size):
            inp = inputs[sl]
            base = baseline[sl]
            if IntegratedGradients is not None:
                attr = ig.attribute(inp, baselines=base, target=class_idx)
            else:
                attr = ig.attribute(inp, baselines=base, target=class_idx)
            batch_attr.append(attr.detach().cpu().numpy())
        attr_all = np.concatenate(batch_attr, axis=0)
        class_scores[class_idx] = np.mean(np.abs(attr_all), axis=0)

    class_weights = np.bincount(y, minlength=num_classes).astype(np.float64)
    class_weights = class_weights / class_weights.sum()
    overall = np.zeros_like(next(iter(class_scores.values())))
    for class_idx in range(num_classes):
        overall += class_weights[class_idx] * class_scores[class_idx]

    overall_df = pd.DataFrame({"feature": feature_names, "importance": overall}).sort_values("importance", ascending=False)
    overall_df.to_csv(output_dir / "biomarker_importance_overall.csv", index=False)
    overall_df.head(args.top_k).to_csv(output_dir / f"top_{args.top_k}_biomarkers.csv", index=False)

    class_frames = []
    for class_idx in range(num_classes):
        df = pd.DataFrame({"class": class_idx, "feature": feature_names, "importance": class_scores[class_idx]}).sort_values(
            ["class", "importance"], ascending=[True, False]
        )
        class_frames.append(df)
    class_df = pd.concat(class_frames, axis=0, ignore_index=True)
    class_df.to_csv(output_dir / "biomarker_importance_by_class.csv", index=False)

    print(f"Saved overall ranking to: {(output_dir / 'biomarker_importance_overall.csv').resolve()}")
    print(f"Saved top-{args.top_k} ranking to: {(output_dir / f'top_{args.top_k}_biomarkers.csv').resolve()}")
    print(f"Saved class-specific ranking to: {(output_dir / 'biomarker_importance_by_class.csv').resolve()}")


if __name__ == "__main__":
    main()
