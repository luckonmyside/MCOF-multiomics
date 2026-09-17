from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
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
    """SE-style channel gate over omics tokens."""

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
        """
        tokens: [B, M, D]
        returns weighted tokens and weights [B, M]
        """
        pooled = tokens.mean(dim=-1)  # [B, M]
        weights = self.gate(pooled)
        return tokens * weights.unsqueeze(-1), weights


class TokenAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.ffn(tokens)
        gate_logits = self.gate(tokens).squeeze(-1)  # [B, M]
        gate_weights = torch.softmax(gate_logits, dim=1)
        fused = torch.sum(tokens * gate_weights.unsqueeze(-1), dim=1)
        return fused, gate_weights


class ConvHead1D(nn.Module):
    def __init__(
        self,
        input_len: int,
        num_classes: int,
        conv_channels: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(num_groups=8 if conv_channels >= 8 else 1, num_channels=conv_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(num_groups=8 if conv_channels * 2 >= 8 else 1, num_channels=conv_channels * 2),
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
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


class MLPHead(nn.Module):
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
            [OmicsProjector(in_dim=d, hidden_dim=config.hidden_dim, dropout=config.dropout) for d in self.input_dims]
        )
        self.reset_parameters()

    def split_inputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor [batch, features], got {tuple(x.shape)}")
        expected = sum(self.input_dims)
        if x.shape[1] != expected:
            raise ValueError(
                f"Input feature dimension mismatch. Expected {expected}, got {x.shape[1]}. input_dims={self.input_dims}"
            )
        return list(torch.split(x, self.input_dims, dim=1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        parts = self.split_inputs(x)
        tokens = [proj(part) for proj, part in zip(self.projectors, parts)]
        return torch.stack(tokens, dim=1)  # [B, M, D]

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
    """Closer to the current implementation: channel attention + convolution.

    Difference from the legacy code:
    - variable omics dimensions are supported;
    - no hard-coded 200/1000 branches;
    - no BatchNorm dependence on tiny batches;
    - projector makes convolution operate on learned latent dimensions instead of raw feature ordering.
    """

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


class MCOFAttnV2(BaseOmicsModel):
    """True cross-omics attention version, closer to the manuscript description."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for MultiheadAttention.")
        self.attn_fusion = TokenAttentionFusion(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.conv_head = ConvHead1D(
            input_len=config.hidden_dim,
            num_classes=config.num_classes,
            conv_channels=config.conv_channels,
            kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(x)
        fused, _ = self.attn_fusion(tokens)
        return self.conv_head(fused)


class MCAOnlyV2(BaseOmicsModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.attn_fusion = TokenAttentionFusion(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.classifier = MLPHead(config.hidden_dim, config.num_classes, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(x)
        fused, _ = self.attn_fusion(tokens)
        return self.classifier(fused)


class CNNOnlyV2(BaseOmicsModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.concat_mixer = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * self.num_omics),
            nn.Linear(config.hidden_dim * self.num_omics, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.conv_head = ConvHead1D(
            input_len=config.hidden_dim,
            num_classes=config.num_classes,
            conv_channels=config.conv_channels,
            kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(x)
        fused = self.concat_mixer(tokens.flatten(start_dim=1))
        return self.conv_head(fused)


class MCOFNoSEV2(BaseOmicsModel):
    """Clean ablation of MCOF-SE with the channel gate removed.

    The omics-specific projectors and convolutional classification head are
    identical to MCOFSEV2. Fusion is an unweighted sum, which is equivalent
    to fixing every channel weight to one.
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
        tokens = self.encode(x)
        fused = tokens.sum(dim=1)
        return self.conv_head(fused)


class MCOFNoConvV2(BaseOmicsModel):
    """Clean ablation of MCOF-SE with the convolutional head removed.

    The omics-specific projectors and SE-style channel gate are identical to
    MCOFSEV2. The fused latent vector is classified by the existing MLPHead.
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
        fused = weighted_tokens.sum(dim=1)
        return self.classifier(fused)


def build_model(config: ModelConfig) -> nn.Module:
    model_type = config.model_type.lower()
    if model_type in {"mcof", "mcof_se", "mcof_se_v2", "legacy_fixed"}:
        return MCOFSEV2(config)
    if model_type in {"mcof_attn", "mcof_v2", "attn"}:
        return MCOFAttnV2(config)
    if model_type in {"mcof_no_se", "no_se", "minus_se"}:
        return MCOFNoSEV2(config)
    if model_type in {"mcof_no_conv", "no_conv", "minus_conv"}:
        return MCOFNoConvV2(config)
    if model_type in {"mca", "mca_v2", "mca_only"}:
        return MCAOnlyV2(config)
    if model_type in {"cnn", "cnn_v2", "cnn_only"}:
        return CNNOnlyV2(config)
    raise ValueError(f"Unsupported model_type: {config.model_type}")
