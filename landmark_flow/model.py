"""PyTorch model for conditional rectified-flow landmark correction."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class LandmarkFlowModel(nn.Module):
    def __init__(
        self,
        num_landmarks: int = 13,
        cond_dim: int = 4,
        hidden_size: int = 256,
        time_dim: int = 32,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 4, 2),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.cond_dim = cond_dim
        self.config = {
            "num_landmarks": num_landmarks,
            "cond_dim": cond_dim,
            "hidden_size": hidden_size,
            "time_dim": time_dim,
            "kernel_size": kernel_size,
            "dilations": list(dilations),
            "dropout": dropout,
        }
        in_channels = num_landmarks * (2 + cond_dim) + time_dim
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.in_proj = nn.Conv1d(in_channels, hidden_size, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_size, kernel_size, d, dropout) for d in dilations])
        self.out_proj = nn.Conv1d(hidden_size, num_landmarks * 2, 1)

    def forward(self, y_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, frames, landmarks, _ = y_t.shape
        features = torch.cat([y_t, condition], dim=-1).reshape(batch, frames, landmarks * (2 + self.cond_dim))
        time_features = self.time_embed(t).unsqueeze(1).expand(-1, frames, -1)
        x = torch.cat([features, time_features], dim=-1).transpose(1, 2)
        x = self.in_proj(x)
        x = self.blocks(x)
        velocity = self.out_proj(x).transpose(1, 2).reshape(batch, frames, landmarks, 2)
        return velocity

