# SPDX-FileCopyrightText: 2026 Seongsu Kim
# SPDX-License-Identifier: MIT
"""End-to-end SO(3)/O(3) checks, including the input AO convention."""
import sys
from pathlib import Path

import pytest
import torch
from e3nn import o3
from torch_geometric.data import Data

pytest.importorskip("fairchem.core")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.QHFlow2_equivariant import QHFlow2_order_fixed, QHFlow2_O3


def graph(dtype):
    n = 4
    def symmetric():
        a = torch.randn(n, 14, 14, dtype=dtype)
        return (a + a.mT) / 2
    return Data(
        pos=torch.tensor([[0., 0., 0.], [1., .2, .1], [.1, 1.1, .3], [.2, .1, 1.3]], dtype=dtype),
        atoms=torch.tensor([[6], [7], [8], [6]]), batch=torch.zeros(n, dtype=torch.long),
        ptr=torch.tensor([0, n]), natoms=torch.tensor([n]), cell=torch.zeros(1, 3, 3, dtype=dtype),
        t=torch.tensor([.1, .3, .7, .9], dtype=dtype), diagonal_init_ham=symmetric(), diagonal_overlap=symmetric(),
    ), symmetric()


def rotation_in_ao_basis(rotation):
    p = torch.eye(3, dtype=rotation.dtype)[[1, 2, 0]]
    # Physical orbital parity: p is odd even though the legacy SO(3) head
    # used all-even labels (proper rotations cannot distinguish parity).
    return o3.Irreps("3x0e+2x1o+1x2e").D_from_matrix(p @ rotation @ p.T)


@pytest.mark.parametrize("cls", [QHFlow2_order_fixed, QHFlow2_O3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complete_model_rotation_and_projection(cls, dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        torch.manual_seed(42)
        model = cls(hidden_size=4, bottle_hidden_size=2, num_gnn_layers=1,
                    num_ham_gnn_layers=1, use_block_S=True, use_block_H=True).eval()
        data, h = graph(dtype)
        saved = data.clone()
        with torch.no_grad():
            baseline = model(data, h, keep_blocks=True)
            for seed in (0, 1, 2):
                torch.manual_seed(seed)
                r = o3.rand_matrix(dtype=dtype)
                if cls is QHFlow2_O3 and seed != 0:
                    r = -r if seed == 1 else -torch.eye(3, dtype=dtype)
                d = rotation_in_ao_basis(r)
                transformed = data.clone()
                transformed.pos = data.pos @ r.T + torch.tensor([.3, -.4, .2], dtype=dtype)
                for key in ("diagonal_init_ham", "diagonal_overlap"):
                    transformed[key] = d @ data[key] @ d.T
                actual = model(transformed, d @ h @ d.T, keep_blocks=True)
                for key in ("hamiltonian_diagonal_blocks", "hamiltonian_non_diagonal_blocks"):
                    expected = d @ baseline[key] @ d.T
                    tolerance = 5e-5 if dtype == torch.float32 else 1e-9
                    torch.testing.assert_close(actual[key], expected, atol=tolerance, rtol=tolerance)
        assert set(data.keys()) == set(saved.keys())
        for key in data.keys():
            torch.testing.assert_close(data[key], saved[key], rtol=0, atol=0)
        # Atom permutations must carry the corresponding H/H0/S blocks.
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = data.clone()
        for key in ("pos", "atoms", "t", "diagonal_init_ham", "diagonal_overlap"):
            permuted[key] = data[key][permutation]
        with torch.no_grad():
            actual = model(permuted, h[permutation], keep_blocks=True)
        tol = 5e-5 if dtype == torch.float32 else 1e-9
        torch.testing.assert_close(actual["hamiltonian_diagonal_blocks"],
                                   baseline["hamiltonian_diagonal_blocks"][permutation], atol=tol, rtol=tol)
        old_pairs = {tuple(edge): i for i, edge in enumerate(baseline["full_edge_index"].T.tolist())}
        mapping = [old_pairs[tuple(edge)] for edge in permutation[actual["full_edge_index"]].T.tolist()]
        torch.testing.assert_close(actual["hamiltonian_non_diagonal_blocks"],
                                   baseline["hamiltonian_non_diagonal_blocks"][mapping], atol=tol, rtol=tol)
        h.requires_grad_()
        output = model(data, h, keep_blocks=True)
        sum(output[k].square().sum() for k in output if k.startswith("hamiltonian")).backward()
        assert torch.isfinite(h.grad).all()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    finally:
        torch.set_default_dtype(old_dtype)


def test_ao_rotation_matches_pyscf_orbitals():
    import numpy as np
    from pyscf import gto
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    mol = gto.M(atom="C 0 0 0", basis="def2-svp", verbose=0)
    torch.manual_seed(7)
    points = torch.randn(20, 3, dtype=torch.float64)
    permutation = [0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13]
    values = mol.eval_gto("GTOval_sph", points.numpy())[:, permutation]
    try:
        for r in (o3.rand_matrix(dtype=points.dtype), -torch.eye(3, dtype=points.dtype)):
            actual = mol.eval_gto("GTOval_sph", (points @ r.T).numpy())[:, permutation]
            expected = values @ rotation_in_ao_basis(r).numpy().T
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    finally:
        torch.set_default_dtype(previous_dtype)
