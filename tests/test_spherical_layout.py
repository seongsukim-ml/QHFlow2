# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

"""Regression for the multi-channel eSCN/e3nn boundary (proper rotations)."""

import importlib.util
from pathlib import Path

import pytest
import torch
from e3nn import o3

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spherical_layout", ROOT / "src/models/spherical_layout.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
escn_to_e3nn = MODULE.escn_to_e3nn


def rotate_native(x, rotation, lmax):
    return torch.cat([
        torch.einsum(
            "ij,njc->nic", o3.Irrep(l, 1).D_from_matrix(rotation),
            x[:, l * l:(l + 1) ** 2],
        )
        for l in range(lmax + 1)
    ], dim=1)


def relative_error(actual, expected):
    return (actual - expected).norm() / expected.norm().clamp_min(1e-30)


@pytest.mark.parametrize("channels", [1, 2, 64, 128])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_layout_commutes_with_rotation(channels, seed):
    torch.manual_seed(seed)
    x = torch.randn(3, 25, channels, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=x.dtype)
    rotated = rotate_native(x, rotation, 4)
    # Independent e3nn multiplicity representation, not the conversion itself.
    irreps = o3.Irreps([(channels, (l, 1)) for l in range(5)])
    d = irreps.D_from_matrix(rotation)
    torch.testing.assert_close(
        escn_to_e3nn(rotated, 4), escn_to_e3nn(x, 4) @ d.T,
        rtol=1e-10, atol=1e-10,
    )
    legacy_error = relative_error(rotated.flatten(1), x.flatten(1) @ d.T)
    if channels > 1:
        assert legacy_error > 0.1  # Reproduce the old reshape bug.
    else:
        assert legacy_error < 1e-10


def test_layout_preserves_gradients_and_accepts_noncontiguous_input():
    x = torch.randn(2, 4, 25, dtype=torch.float64, requires_grad=True)
    packed = escn_to_e3nn(x.transpose(1, 2), 4)
    packed.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x)


@pytest.mark.parametrize("shape,lmax", [((2, 24, 4), 4), ((2, 25), 4), ((2, 1, 4), -1)])
def test_layout_rejects_incompatible_shapes(shape, lmax):
    with pytest.raises(ValueError, match="Expected"):
        escn_to_e3nn(torch.zeros(shape), lmax)


@pytest.mark.parametrize("channels", [64, 128])
def test_actual_diagonal_head_rotation(monkeypatch, channels):
    """Exercise the legacy head layers and the all-even Expansion convention."""
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from models.layers import Expansion, SelfNetLayer
    from models.layers_v2 import construct_o3irrps, construct_o3irrps_base

    torch.manual_seed(123)
    base = o3.Irreps(construct_o3irrps_base(channels, 4))
    hidden = o3.Irreps(construct_o3irrps(channels, 4))
    bottle = o3.Irreps(construct_o3irrps(32, 4))
    output = o3.Irreps("3x0e+2x1e+1x2e")
    layers = [SelfNetLayer(
        base, base, base, o3.Irreps.spherical_harmonics(4), 16, channels,
    ).double().eval() for _ in range(2)]
    linear = o3.Linear(hidden, bottle).double().eval()
    expansion = Expansion(
        o3.Irreps(construct_o3irrps_base(32, 4)), output, output,
    ).double().eval()
    weights = torch.randn(3, expansion.num_path_weight, dtype=torch.float64)
    bias = torch.randn(3, expansion.num_bias, dtype=torch.float64)

    def head(x):
        previous = None
        for layer in layers:
            current = layer(None, x, previous)
            previous = current if previous is None else previous + current
        h = expansion(linear(previous), weights, bias)
        return (h + h.transpose(-1, -2)) / 2

    with torch.no_grad():
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            x = torch.randn(3, 25, channels, dtype=torch.float64)
            rotation = o3.rand_matrix(dtype=x.dtype)
            rotated = rotate_native(x, rotation, 4)
            d = output.D_from_matrix(rotation)
            expected = d @ head(escn_to_e3nn(x, 4)) @ d.T
            actual = head(escn_to_e3nn(rotated, 4))
            # Layer constants may have been constructed in float32 before .double().
            assert relative_error(actual, expected) < 1e-6
            assert relative_error(head(rotated.flatten(1)), d @ head(x.flatten(1)) @ d.T) > 0.1
