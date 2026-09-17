from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch

from train_utils_v2 import (
    OmicsPreprocessor,
    load_checkpoint,
    read_multiomics_dataset,
    restore_model_from_checkpoint,
)

try:
    from captum.attr import IntegratedGradients  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    IntegratedGradients = None


def _iter_slices(n_samples: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n_samples, batch_size):
        yield slice(start, min(start + batch_size, n_samples))


class ManualIntegratedGradients:
    """Fallback integrated-gradients implementation when Captum is absent."""

    def __init__(self, model: torch.nn.Module, steps: int = 50) -> None:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        self.model = model
        self.steps = int(steps)

    def attribute(
        self,
        inputs: torch.Tensor,
        baselines: torch.Tensor,
        target: int,
    ) -> torch.Tensor:
        self.model.eval()
        alphas = torch.linspace(
            0.0,
            1.0,
            steps=self.steps + 1,
            device=inputs.device,
        )[1:]
        total_grad = torch.zeros_like(inputs)
        difference = inputs - baselines

        for alpha in alphas:
            interpolated = baselines + alpha * difference
            interpolated.requires_grad_(True)
            logits = self.model(interpolated)
            score = logits[:, target].sum()
            gradient = torch.autograd.grad(
                score,
                interpolated,
                retain_graph=False,
                create_graph=False,
            )[0]
            total_grad += gradient.detach()

        return difference * (total_grad / float(self.steps))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrated-gradients feature ranking for a trained publication "
            "MCOF checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a repeat-level best_model.pt file.",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing the all-sample omics matrices.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory in which attribution tables are saved.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Number of highest-ranked features written to the Top-k file.",
    )
    parser.add_argument(
        "--ig_steps",
        type=int,
        default=50,
        help="Number of integration steps used by Captum or the fallback.",
    )
    parser.add_argument(
        "--omics_names",
        nargs="+",
        default=None,
        help=(
            "Optional human-readable omics names in input order, for example "
            "--omics_names mRNA miRNA."
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _valid_feature_name_layout(
    names: object,
    transformed: Sequence[np.ndarray],
) -> bool:
    if not isinstance(names, list) or len(names) != len(transformed):
        return False
    return all(
        isinstance(block, list) and len(block) == matrix.shape[1]
        for block, matrix in zip(names, transformed)
    )


def _feature_metadata(
    names_nested: Sequence[Sequence[str]],
    omics_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    global_index = 0
    for omics_index, (omics_name, names) in enumerate(
        zip(omics_names, names_nested),
        start=1,
    ):
        for within_index, feature_name in enumerate(names):
            rows.append(
                {
                    "feature_index": global_index,
                    "omics_index": omics_index,
                    "omics": omics_name,
                    "feature_within_omics": within_index,
                    "feature": str(feature_name),
                }
            )
            global_index += 1
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top_k must be at least 1")
    if args.ig_steps < 1:
        raise ValueError("--ig_steps must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device("cpu")
        if args.cpu or not torch.cuda.is_available()
        else torch.device("cuda")
    )

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model = restore_model_from_checkpoint(checkpoint, device=device)
    preprocessor = OmicsPreprocessor.from_state_dict(
        checkpoint["preprocessor_state"]
    )

    omics_list, labels, raw_feature_names = read_multiomics_dataset(
        args.data_dir
    )
    transformed = preprocessor.transform(omics_list)
    x_all = np.concatenate(transformed, axis=1).astype(
        np.float32,
        copy=False,
    )

    data_file_names = preprocessor.transform_feature_names(raw_feature_names)
    checkpoint_names = checkpoint.get("feature_names")

    def _all_names_are_generic(names_nested: Sequence[Sequence[str]]) -> bool:
        flattened = [str(value) for block in names_nested for value in block]
        return bool(flattened) and all(
            re.fullmatch(r"omics\d+_f\d+", value) is not None
            for value in flattened
        )

    # Prefer current *_featname.csv identifiers when available. This avoids
    # historical checkpoints that contain correctly sized but generic names.
    if (
        _valid_feature_name_layout(data_file_names, transformed)
        and not _all_names_are_generic(data_file_names)
    ):
        feature_names_nested = [list(map(str, block)) for block in data_file_names]
        feature_name_source = "data_files_transformed_by_checkpoint_preprocessor"
    elif _valid_feature_name_layout(checkpoint_names, transformed):
        feature_names_nested = [
            [str(value) for value in block]
            for block in checkpoint_names
        ]
        feature_name_source = "checkpoint"
    else:
        feature_names_nested = [list(map(str, block)) for block in data_file_names]
        feature_name_source = "generated_or_data_file_names"

    if args.omics_names is None:
        omics_names = [
            f"omics_{index}"
            for index in range(1, len(transformed) + 1)
        ]
    else:
        if len(args.omics_names) != len(transformed):
            raise ValueError(
                "The number of --omics_names values must equal the number "
                f"of omics blocks ({len(transformed)})."
            )
        omics_names = [str(value) for value in args.omics_names]

    metadata = _feature_metadata(feature_names_nested, omics_names)
    if len(metadata) != x_all.shape[1]:
        raise ValueError(
            "Feature-name dimension mismatch after preprocessing: "
            f"{len(metadata)} names for {x_all.shape[1]} model inputs."
        )

    inputs = torch.from_numpy(x_all).to(device)
    baseline = torch.zeros_like(inputs)
    labels = labels.astype(int)
    num_classes = int(checkpoint["model_config"]["num_classes"])

    if IntegratedGradients is not None:
        ig_method: object = IntegratedGradients(model)
        implementation = "captum.attr.IntegratedGradients"
    else:
        ig_method = ManualIntegratedGradients(model, steps=args.ig_steps)
        implementation = "manual_fallback"

    class_scores: Dict[int, np.ndarray] = {}

    for class_index in range(num_classes):
        batches = []
        for current_slice in _iter_slices(len(inputs), args.batch_size):
            current_inputs = inputs[current_slice]
            current_baseline = baseline[current_slice]
            model.zero_grad(set_to_none=True)

            if IntegratedGradients is not None:
                attribution = ig_method.attribute(  # type: ignore[attr-defined]
                    current_inputs,
                    baselines=current_baseline,
                    target=class_index,
                    n_steps=args.ig_steps,
                )
            else:
                attribution = ig_method.attribute(  # type: ignore[attr-defined]
                    current_inputs,
                    baselines=current_baseline,
                    target=class_index,
                )
            batches.append(attribution.detach().cpu().numpy())

        attribution_all = np.concatenate(batches, axis=0)
        class_scores[class_index] = np.mean(
            np.abs(attribution_all),
            axis=0,
        )

    class_weights = np.bincount(
        labels,
        minlength=num_classes,
    ).astype(np.float64)
    class_weights /= class_weights.sum()

    overall = np.zeros_like(next(iter(class_scores.values())))
    for class_index in range(num_classes):
        overall += class_weights[class_index] * class_scores[class_index]

    overall_df = metadata.copy()
    overall_df["importance"] = overall
    overall_df = overall_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)
    overall_df.insert(0, "rank", np.arange(1, len(overall_df) + 1))

    overall_path = output_dir / "biomarker_importance_overall.csv"
    top_path = output_dir / f"top_{args.top_k}_biomarkers.csv"
    overall_df.to_csv(overall_path, index=False)
    overall_df.head(args.top_k).to_csv(top_path, index=False)

    class_frames = []
    for class_index in range(num_classes):
        frame = metadata.copy()
        frame.insert(0, "class", class_index)
        frame["importance"] = class_scores[class_index]
        frame = frame.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)
        frame.insert(1, "rank", np.arange(1, len(frame) + 1))
        class_frames.append(frame)

    class_path = output_dir / "biomarker_importance_by_class.csv"
    pd.concat(class_frames, ignore_index=True).to_csv(
        class_path,
        index=False,
    )

    run_metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "device": str(device),
        "integrated_gradients_implementation": implementation,
        "ig_steps": int(args.ig_steps),
        "batch_size": int(args.batch_size),
        "top_k": int(args.top_k),
        "num_classes": num_classes,
        "class_weights": class_weights.tolist(),
        "omics_names": omics_names,
        "feature_name_source": feature_name_source,
        "n_features": int(x_all.shape[1]),
        "model_config": checkpoint["model_config"],
    }
    with (output_dir / "biomarker_run_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(run_metadata, handle, indent=2, ensure_ascii=False)

    print(f"Saved overall ranking to: {overall_path.resolve()}")
    print(f"Saved top-{args.top_k} ranking to: {top_path.resolve()}")
    print(f"Saved class-specific ranking to: {class_path.resolve()}")


if __name__ == "__main__":
    main()
