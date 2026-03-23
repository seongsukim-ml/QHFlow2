"""QHFlow Calculator: MLDFTCalculator subclass for Hamiltonian prediction models.

Extends MLDFTCalculator with QHFlow2-specific features:
- Dual-functional init (separate init_xc, init_density_fit)
- Forced closed-shell init (spin=0)
- Filtering (2-tolerance eigensolve for stability)
- SCF mode (predicted D as SCF initial guess)
- UKS support
- gpu4pyscf support

Usage:
    from qhflow2.md.qhflow_calculator import QHFlowCalculator

    def predict_H(pyscf_mol, S, H_init):
        # model-specific: convention transform, forward, transform back
        return H_pred  # (nao, nao) ndarray

    calc = QHFlowCalculator(
        predict_H, basis='def2-svp', xc='pbe',
        init_xc='pbe', filtering=True,
    )
    atoms.calc = calc
    atoms.get_potential_energy()
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

import sys
sys.path.insert(0, "/home1/irteam/data-vol1/projects/dft-dataset")
from src.calculator import MLDFTCalculator  # noqa: E402

from ase.calculators.calculator import all_changes


class QHFlowCalculator(MLDFTCalculator):
    """MLDFTCalculator extended for QHFlow2 Hamiltonian prediction models.

    Key extensions over MLDFTCalculator:
    - init_xc: separate functional for initial Hamiltonian (model input)
    - init_density_fit: separate DF flag for init
    - spin=0 forced for init mol (closed-shell init regardless of target)
    - filtering: 2-tolerance eigensolve comparison for numerical stability
    - scf mode: run full SCF with predicted density as initial guess
    - UKS support via method parameter
    - gpu4pyscf support
    - timing dict for profiling
    """

    def __init__(
        self,
        predict_hamiltonian: Callable,
        basis: str = "def2-svp",
        xc: str = "pbe",
        density_fit: bool = True,
        init_xc: str | None = None,
        init_density_fit: bool | None = None,
        method: str = "rks",
        filtering: bool = False,
        gt_tol: float = 1e-8,
        pred_tol: float = 1e-3,
        init_grid_level: int | None = None,
        gpu4pyscf: bool = False,
        init_gpu4pyscf: bool | None = None,
        max_cycle: int = 50,
        level_shift: float | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            predict_hamiltonian: Callable(pyscf_mol, S, H_init) → H_pred (nao, nao).
            basis: Gaussian basis set.
            xc: Target XC functional (for energy evaluation).
            density_fit: Use density fitting for target mf.
            init_xc: XC functional for initial Hamiltonian. Defaults to xc.
            init_density_fit: Density fitting for init mf. Defaults to density_fit.
            method: "rks" or "uks".
            filtering: Enable 2-tolerance eigensolve stability check.
            gt_tol: Conservative tolerance for eigensolve (filtering).
            pred_tol: Aggressive tolerance for eigensolve (filtering).
            gpu4pyscf: Use GPU-accelerated PySCF.
            init_grid_level: DFT grid level for init Hamiltonian (0-3).
                Lower = faster but coarser V_xc. None = PySCF default (3).
                For delta-learning models, level=0 gives ~1.9x speedup with ~1e-2 error.
            init_gpu4pyscf: GPU for init mf. Defaults to gpu4pyscf.
            max_cycle: Max SCF cycles.
            level_shift: Level shift for SCF convergence.
        """
        super().__init__(
            predict_hamiltonian=predict_hamiltonian,
            basis=basis, xc=xc, density_fit=density_fit,
            **kwargs,
        )
        self.init_xc = init_xc or xc
        self.init_density_fit = init_density_fit if init_density_fit is not None else density_fit
        self.method = method.lower()
        self.filtering = filtering
        self.gt_tol = gt_tol
        self.pred_tol = pred_tol
        self.init_grid_level = init_grid_level
        self.gpu4pyscf = gpu4pyscf
        self.init_gpu4pyscf = init_gpu4pyscf if init_gpu4pyscf is not None else gpu4pyscf
        self.max_cycle = max_cycle
        self.level_shift = level_shift

        self.time_dict: dict[str, float] = {}
        self.init_hamiltonian: np.ndarray | None = None

    def _build_pyscf(self, atoms, spin=0):
        """Build target PySCF mol + mean-field."""
        from pyscf import gto, dft

        mol = gto.M(
            atom=[(s, p) for s, p in zip(
                atoms.get_chemical_symbols(), atoms.get_positions()
            )],
            basis=self.basis, unit="Ang", verbose=0, spin=spin,
        )

        if self.method == "uks":
            mf = dft.UKS(mol)
        else:
            mf = dft.RKS(mol)

        if self.density_fit:
            mf = mf.density_fit()
        if self.gpu4pyscf:
            mf = mf.to_gpu()
        mf.xc = self.xc
        mf.max_cycle = self.max_cycle
        if self.level_shift is not None:
            mf.level_shift = self.level_shift

        return mol, mf

    def _build_init_pyscf(self, atoms):
        """Build init PySCF mol (spin=0) + mean-field with init_xc."""
        from pyscf import gto, dft

        mol = gto.M(
            atom=[(s, p) for s, p in zip(
                atoms.get_chemical_symbols(), atoms.get_positions()
            )],
            basis=self.basis, unit="Ang", verbose=0, spin=0,
        )
        mf = dft.RKS(mol)
        if self.init_density_fit:
            mf = mf.density_fit()
        if self.init_gpu4pyscf:
            mf = mf.to_gpu()
        mf.xc = self.init_xc
        if self.init_grid_level is not None:
            mf.grids.level = self.init_grid_level
        mf.max_cycle = self.max_cycle
        if self.level_shift is not None:
            mf.level_shift = self.level_shift

        return mol, mf

    @staticmethod
    def _eigensolve(H, S, tol=1e-8):
        """Solve HC=SCε via S^{-1/2} transform (matches QHFlow2's cal_orbital_and_energies)."""
        eigvals_s, U = np.linalg.eigh(S)
        mask = eigvals_s > tol
        X = U[:, mask] / np.sqrt(eigvals_s[mask])
        H_orth = X.T @ H @ X
        energies, C_orth = np.linalg.eigh(H_orth)
        coefficients = X @ C_orth
        return energies, coefficients

    def _eigensolve_with_filtering(self, H_pred, S, n_electrons):
        """Eigensolve with optional 2-tolerance stability check."""
        n_occ = n_electrons // 2

        if not self.filtering:
            return self._eigensolve(H_pred, S, tol=self.gt_tol)

        # Safe (conservative tolerance)
        e_safe, c_safe = self._eigensolve(H_pred, S, tol=self.gt_tol)
        # Aggressive (lower tolerance)
        e_aggr, c_aggr = self._eigensolve(H_pred, S, tol=self.pred_tol)

        # Compare orbital energy stability around HOMO-LUMO
        gap_diff = np.abs(e_safe[:n_occ + 1] - e_aggr[:n_occ + 1])
        if np.max(gap_diff) > 0.03:
            # Large discrepancy → use aggressive (often more accurate)
            return e_aggr, c_aggr
        else:
            return e_safe, c_safe

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
        scf: bool = False,
        spin: int = 0,
    ) -> None:
        from ase.units import Hartree, Bohr
        # Import from dft-dataset (add src/ to path for bare imports)
        _dft_src = "/home1/irteam/data-vol1/projects/dft-dataset/src"
        if _dft_src not in sys.path:
            sys.path.insert(0, _dft_src)
        from solvers import build_density_matrix
        from molecule import Molecule

        if properties is None:
            properties = self.implemented_properties
        # Call Calculator.calculate (skip MLDFTCalculator.calculate)
        from ase.calculators.calculator import Calculator
        Calculator.calculate(self, atoms, properties, system_changes)

        self.time_dict = {}

        # 1. Build target PySCF infrastructure
        t0 = time.time()
        pyscf_mol, mf = self._build_pyscf(atoms, spin=spin)
        self._pyscf_mol = pyscf_mol
        self._mf = mf
        self.time_dict["mf_time"] = time.time() - t0

        # 2. Overlap
        t0 = time.time()
        S = pyscf_mol.intor("int1e_ovlp")
        self.time_dict["ovlp_time"] = time.time() - t0

        # 3. Init Hamiltonian (separate functional, spin=0)
        t0 = time.time()
        _, init_mf = self._build_init_pyscf(atoms)
        init_dm = init_mf.init_guess_by_minao()
        H_init = init_mf.get_fock(dm=init_dm)
        self.init_hamiltonian = H_init
        self.time_dict["init_ham_time"] = time.time() - t0

        # 4. Predict Hamiltonian
        t0 = time.time()
        H_pred = self.predict_hamiltonian(pyscf_mol, S, H_init)
        self.time_dict["pred_ham_time"] = time.time() - t0

        # 5. Eigensolve (with optional filtering)
        t0 = time.time()
        n_electrons = pyscf_mol.nelectron
        mo_energy, mo_coeff = self._eigensolve_with_filtering(H_pred, S, n_electrons)
        n_occ = n_electrons // 2
        D = build_density_matrix(mo_coeff, n_electrons)
        self.time_dict["eigensolve_time"] = time.time() - t0

        # 6. Energy
        t0 = time.time()
        if scf:
            # SCF mode: use predicted D as initial guess, run full SCF
            if self.method == "uks":
                D_init = (D / 2.0, D / 2.0)
            else:
                D_init = D
            energy_ha = mf.kernel(dm0=D_init, dump_chk=False)
            # Update orbitals from SCF result
            mo_energy = mf.mo_energy
            mo_coeff = mf.mo_coeff
            D = mf.make_rdm1()
            if hasattr(D, 'get'):  # cupy → numpy
                D = D.get()
        else:
            if self.method == "uks":
                D_for_energy = (D / 2.0, D / 2.0)
            else:
                D_for_energy = D
            energy_ha = mf.energy_tot(D_for_energy)
        self.time_dict["energy_time"] = time.time() - t0

        # 7. Store as Molecule
        molecule = Molecule(
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            charge=0,
            spin=spin,
            energy_unit="Ha",
            length_unit="Ang",
        )
        molecule.overlap = S
        molecule.hamiltonian = H_pred
        molecule.orbital_energies = np.asarray(mo_energy)
        molecule.orbital_coeffs = np.asarray(mo_coeff)
        molecule.density_matrix = np.asarray(D)
        molecule.energy = float(energy_ha)
        n_occ = n_electrons // 2
        molecule.homo = float(mo_energy[n_occ - 1]) if n_occ > 0 else None
        molecule.lumo = float(mo_energy[n_occ]) if n_occ < len(mo_energy) else None
        molecule.xc = self.xc

        self.molecule = molecule
        self._props = {"homo": molecule.homo, "lumo": molecule.lumo}

        self.results["energy"] = float(energy_ha) * Hartree
        self.results["free_energy"] = float(energy_ha) * Hartree

        # 8. Forces
        if "forces" in properties:
            t0 = time.time()
            mo_occ = mf.get_occ(mo_energy, mo_coeff)
            grad = mf.nuc_grad_method()
            if scf:
                forces = -grad.kernel()
            else:
                grad.base.auxbasis_response = True
                forces = -grad.kernel(
                    mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ,
                )
            self.time_dict["grad_time"] = time.time() - t0
            if hasattr(forces, 'get'):
                forces = forces.get()
            self.results["forces"] = forces * Hartree / Bohr
