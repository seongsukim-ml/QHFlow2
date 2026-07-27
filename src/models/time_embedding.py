# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

"""Sinusoidal embeddings for continuous or discrete time coordinates."""

from __future__ import annotations

import math

import torch


def sinusoidal_time_embedding(
    values: torch.Tensor,
    embedding_dim: int,
    *,
    max_period: float = 2_000.0,
    scale: float = 1.0,
    sin_first: bool = True,
    include_endpoint: bool = True,
) -> torch.Tensor:
    """Map a one-dimensional time tensor to deterministic Fourier features.

    Frequencies are geometrically spaced from 1 down to ``1 / max_period``.
    ``include_endpoint=False`` uses ``half_dim`` logarithmic intervals instead
    of ``half_dim - 1`` and is provided for older model configurations.
    """
    if values.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got {tuple(values.shape)}")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if max_period <= 1:
        raise ValueError("max_period must be greater than one")

    half_dim = embedding_dim // 2
    if half_dim == 0:
        return torch.zeros(
            (values.shape[0], embedding_dim),
            dtype=torch.float32,
            device=values.device,
        )

    interval_count = half_dim - 1 if include_endpoint and half_dim > 1 else half_dim
    exponents = torch.arange(half_dim, dtype=torch.float32, device=values.device)
    frequencies = torch.exp(-math.log(float(max_period)) * exponents / interval_count)
    angles = values.to(dtype=torch.float32).unsqueeze(1) * float(scale) * frequencies

    sine = torch.sin(angles)
    cosine = torch.cos(angles)
    embedding = torch.cat((sine, cosine), dim=1) if sin_first else torch.cat((cosine, sine), dim=1)
    if embedding_dim % 2:
        embedding = torch.cat((embedding, embedding.new_zeros((embedding.shape[0], 1))), dim=1)
    return embedding


def get_time_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_positions: float = 2_000.0,
) -> torch.Tensor:
    """Return QHFlow's backward-compatible time encoding."""
    return sinusoidal_time_embedding(
        timesteps,
        embedding_dim,
        max_period=max_positions,
        scale=max_positions,
        sin_first=True,
        include_endpoint=True,
    )
