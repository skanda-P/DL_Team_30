"""Autoencoder architectures used by Assignment 4 Task 2."""

from __future__ import annotations

import torch
from torch import nn


class OneHiddenAutoencoder(nn.Module):
    """784 -> bottleneck -> 784 autoencoder."""

    def __init__(self, bottleneck_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 784),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


class ThreeHiddenAutoencoder(nn.Module):
    """784 -> 400 -> bottleneck -> 400 -> 784 autoencoder."""

    def __init__(self, bottleneck_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.Sigmoid(),
            nn.Linear(400, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 400),
            nn.Sigmoid(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


def build_autoencoder(architecture: str, bottleneck_dim: int) -> nn.Module:
    """Build a Task 2 autoencoder with a linear bottleneck."""
    if architecture == "ae1":
        return OneHiddenAutoencoder(bottleneck_dim)
    if architecture == "ae3":
        return ThreeHiddenAutoencoder(bottleneck_dim)
    raise ValueError("architecture must be 'ae1' or 'ae3'")
