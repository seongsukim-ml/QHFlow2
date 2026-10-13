# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

"""Regression checks for the PySCF on-site Q tensor backend."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from pyscf import dft, gto
from pyscf.gto.basis import parse_gaussian

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import Onsite_3idx_Overlap_Integral


ATOMS = [("H", 1, 5), ("C", 6, 14), ("N", 7, 14), ("O", 8, 14), ("F", 9, 14)]
AUXILIARY_GBS = (
    Path(__file__).resolve().parents[1] / "src/experiment/auxiliary.gbs"
)


def test_q_table_without_psi4_or_working_directory_basis(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "psi4", None)
    monkeypatch.chdir(tmp_path)
    table = Onsite_3idx_Overlap_Integral(
        atom_list=[atom for atom, _, _ in ATOMS]
    ).Q_table()

    assert set(table) == {z for _, z, _ in ATOMS}
    for _, z, nao in ATOMS:
        q = table[z]
        assert q.shape == (60, nao, nao)
        assert q.dtype == torch.float64
        assert q.device.type == "cpu"
        assert torch.isfinite(q).all()
        torch.testing.assert_close(q, q.transpose(1, 2), rtol=0, atol=2e-14)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("atom,z,nao", ATOMS)
def test_q_matches_independent_numerical_overlap(atom, z, nao):
    """Check normalization, shell order, and all three AO axes by quadrature.

    Load the original GBS definition independently of the production basis
    and integrate AO products on a dense grid, without int3c1e.
    """
    mol = gto.M(atom=f"{atom} 0 0 0", basis="def2-svp", spin=z % 2, verbose=0)
    auxmol = gto.M(
        atom=mol.atom,
        basis={atom: parse_gaussian.load(str(AUXILIARY_GBS), atom, optimize=False)},
        spin=mol.spin,
        verbose=0,
    )
    grid = dft.gen_grid.Grids(mol)
    grid.atom_grid = (200, 302)
    grid.prune = None
    grid.build()
    ao = mol.eval_gto("GTOval_sph", grid.coords)
    aux = auxmol.eval_gto("GTOval_sph", grid.coords)
    expected = np.einsum("gi,gj,gk,g->kij", ao, ao, aux, grid.weights, optimize=True)

    calculator = Onsite_3idx_Overlap_Integral(atom_list=[atom])
    actual = calculator.transform_Q(calculator.calc_Q(atom), atom).numpy()
    assert actual.shape == (60, nao, nao)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


def test_calc_q_retains_legacy_spherical_order():
    """The raw p components remain z,x,y; Q_table exposes x,y,z."""
    calculator = Onsite_3idx_Overlap_Integral(atom_list=["H"])
    raw = calculator.calc_Q("H")
    # H has two s functions followed by a p shell. An s/p/p overlap is
    # nonzero only when the two p components have the same direction.
    assert raw[16, 0, 2] > 0  # auxiliary p_z, orbital p_z
    assert raw[17, 0, 3] > 0  # auxiliary p_x, orbital p_x
    assert raw[18, 0, 4] > 0  # auxiliary p_y, orbital p_y
    torch.testing.assert_close(
        raw[16:19, 0, 2:5], torch.eye(3, dtype=torch.float64) * raw[16, 0, 2]
    )
    # A d_z2 auxiliary component distinguishes z from x/y and catches a
    # common simultaneous permutation of both p axes in the raw tensor.
    assert raw[40, 2, 2] > 0
    assert raw[40, 3, 3] < 0
    assert raw[40, 4, 4] < 0
