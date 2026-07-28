# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

import importlib.util
import math
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "models" / "time_embedding.py"
_SPEC = importlib.util.spec_from_file_location("qhflow2_time_embedding", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
get_time_embedding = _MODULE.get_time_embedding
sinusoidal_time_embedding = _MODULE.sinusoidal_time_embedding


def test_qhflow_embedding_preserves_the_previous_numeric_convention():
    values = torch.tensor([0.0, 0.25, 1.0])
    embedding_dim = 8
    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(2_000.0)
        * torch.arange(half_dim, dtype=torch.float32)
        / (half_dim - 1)
    )
    angles = values[:, None] * 2_000.0 * frequencies[None, :]
    expected = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)

    actual = get_time_embedding(values, embedding_dim)

    torch.testing.assert_close(actual, expected)


def test_embedding_supports_small_odd_dimensions_without_nan():
    values = torch.tensor([0.0, 1.0])
    actual = sinusoidal_time_embedding(values, 3)

    assert actual.shape == (2, 3)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[:, -1], torch.zeros(2))


def test_embedding_rejects_non_vector_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        sinusoidal_time_embedding(torch.zeros(2, 1), 8)


def test_cosine_first_convention_uses_half_dim_intervals():
    values = torch.tensor([0.5, 1.0])
    embedding_dim = 6
    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half_dim, dtype=torch.float32)
        / half_dim
    )
    angles = values[:, None] * frequencies[None, :]
    expected = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)

    actual = sinusoidal_time_embedding(
        values,
        embedding_dim,
        max_period=10_000.0,
        sin_first=False,
        include_endpoint=False,
    )

    torch.testing.assert_close(actual, expected)
