from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration required to construct MCOF and its strict ablations.

    ``num_heads`` is retained for compatibility with archived checkpoints from
    the development codebase; it is not used by the SE-style publication model.
    """

    model_type: str
    input_dims: List[int]
    num_classes: int
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.30
    conv_channels: int = 64
    conv_kernel_size: int = 5


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x + residual


class OmicsProjector(nn.Module):
    """Map one omics block to the shared latent representation."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLPBlock(hidden_dim, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenSEGate(nn.Module):
    """Squeeze-and-excitation-style channel gate over omics tokens."""

    def __init__(self, num_omics: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(1, num_omics // reduction)
        self.gate = nn.Sequential(
            nn.Linear(num_omics, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_omics, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return recalibrated tokens and sample-specific channel weights.

        Parameters
        ----------
        tokens:
            Tensor with shape ``[batch, number_of_omics, hidden_dimension]``.
        """

        pooled = tokens.mean(dim=-1)
        weights = self.gate(pooled)
        return tokens * weights.unsqueeze(-1), weights


class ConvHead1D(nn.Module):
    """One-dimensional convolutional classification head used by MCOF."""

    def __init__(
        self,
        input_len: int,
        num_classes: int,
        conv_channels: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # input_len is retained in the constructor for checkpoint/API
        # compatibility. Adaptive pooling makes the head dimension robust.
        _ = input_len
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(
                num_groups=8 if conv_channels >= 8 else 1,
                num_channels=conv_channels,
            ),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                conv_channels,
                conv_channels * 2,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GroupNorm(
                num_groups=8 if conv_channels * 2 >= 8 else 1,
                num_channels=conv_channels * 2,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels * 2 * 16, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x.unsqueeze(1)))


class MLPHead(nn.Module):
    """Non-convolutional head used only for the strict no-convolution ablation."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseOmicsModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dims = list(config.input_dims)
        self.num_omics = len(self.input_dims)
        self.num_classes = config.num_classes
        self.hidden_dim = config.hidden_dim
        self.projectors = nn.ModuleList(
            [
                OmicsProjector(
                    in_dim=dimension,
                    hidden_dim=config.hidden_dim,
                    dropout=config.dropout,
                )
                for dimension in self.input_dims
            ]
        )
        self.reset_parameters()

    def split_inputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(
                f"Expected a 2D tensor [batch, features], got {tuple(x.shape)}"
            )
        expected = sum(self.input_dims)
        if x.shape[1] != expected:
            raise ValueError(
                "Input feature dimension mismatch. "
                f"Expected {expected}, got {x.shape[1]}; "
                f"input_dims={self.input_dims}"
            )
        return list(torch.split(x, self.input_dims, dim=1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        parts = self.split_inputs(x)
        tokens = [
            projector(part)
            for projector, part in zip(self.projectors, parts)
        ]
        return torch.stack(tokens, dim=1)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class MCOFSEV2(BaseOmicsModel):
    """Publication MCOF model: omics projectors + channel gate + Conv1D head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.se_gate = TokenSEGate(num_omics=self.num_omics)
        self.conv_head = ConvHead1D(
            input_len=config.hidden_dim,
            num_classes=config.num_classes,
            conv_channels=config.conv_channels,
            kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(x)
        weighted_tokens, _ = self.se_gate(tokens)
        fused = weighted_tokens.sum(dim=1)
        return self.conv_head(fused)


class MCOFNoSEV2(BaseOmicsModel):
    """Strict ablation with channel attention removed.

    Omics-specific projectors and the convolutional head are unchanged. Omics
    tokens are fused by an unweighted sum.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.conv_head = ConvHead1D(
            input_len=config.hidden_dim,
            num_classes=config.num_classes,
            conv_channels=config.conv_channels,
            kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_head(self.encode(x).sum(dim=1))


class MCOFNoConvV2(BaseOmicsModel):
    """Strict ablation with the convolutional head removed.

    Omics-specific projectors and the channel gate are unchanged. The fused
    latent representation is passed to an MLP classifier.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.se_gate = TokenSEGate(num_omics=self.num_omics)
        self.classifier = MLPHead(
            input_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(x)
        weighted_tokens, _ = self.se_gate(tokens)
        return self.classifier(weighted_tokens.sum(dim=1))


def build_model(config: ModelConfig) -> nn.Module:
    """Construct the publication model or one of its strict ablations."""

    model_type = config.model_type.lower()
    if model_type in {"mcof", "mcof_se", "mcof_se_v2", "legacy_fixed"}:
        return MCOFSEV2(config)
    if model_type in {"mcof_no_se", "no_se", "minus_se"}:
        return MCOFNoSEV2(config)
    if model_type in {"mcof_no_conv", "no_conv", "minus_conv"}:
        return MCOFNoConvV2(config)
    raise ValueError(
        f"Unsupported model_type: {config.model_type}. "
        "Choose from mcof_se, mcof_no_se, or mcof_no_conv."
    )
