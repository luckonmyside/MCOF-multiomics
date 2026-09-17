from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common import exact_sign_flip_pvalue, holm_adjust  # noqa: E402
from models_v2 import ModelConfig, build_model  # noqa: E402


def check_model(input_dims: list[int], num_classes: int) -> None:
    config = ModelConfig(
        model_type="mcof_se",
        input_dims=input_dims,
        num_classes=num_classes,
        hidden_dim=32,
        conv_channels=8,
        conv_kernel_size=5,
        dropout=0.1,
    )
    model = build_model(config).eval()
    x = torch.randn(6, sum(input_dims))
    with torch.inference_mode():
        batch_logits = model(x)
        alone_logits = torch.cat([model(x[i : i + 1]) for i in range(len(x))], dim=0)
        reversed_logits = model(x.flip(0)).flip(0)
    assert batch_logits.shape == (6, num_classes)
    assert torch.allclose(batch_logits, alone_logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(batch_logits, reversed_logits, atol=1e-6, rtol=1e-6)

    x_grad = x.clone().requires_grad_(True)
    score = model(x_grad)[0, 0]
    gradient = torch.autograd.grad(score, x_grad)[0]
    assert torch.max(torch.abs(gradient[1:])).item() <= 1e-8


def main() -> None:
    check_model([1000], 4)
    check_model([1000, 1000], 4)
    check_model([200, 200, 200], 2)
    assert exact_sign_flip_pvalue([1, 1, 1, 1, 1]) == 0.0625
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.all((adjusted >= 0) & (adjusted <= 1))
    print("All toolkit smoke tests passed.")


if __name__ == "__main__":
    main()
