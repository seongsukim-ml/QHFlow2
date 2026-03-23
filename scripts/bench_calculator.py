"""Benchmark QHFlowCalculator: energy + force computation throughput.

Measures each pipeline step:
    mf_build → overlap → init_ham → model_forward → eigensolve → energy → forces

Compares:
    1. QHFlowCalculator (PySCF init_ham)
    2. QHFlowCalculator + GPU Fock (get_init_fock_gpu)
    3. Original SCFlowRKSCalculator (baseline, if available)

Usage:
    cd /home1/irteam/data-vol1/projects/QHFlow2/src
    python ../scripts/bench_calculator.py [--device cuda] [--n_runs 5]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from ase import Atoms

sys.path.insert(0, "/home1/irteam/data-vol1/projects/QHFlow2/src")
sys.path.insert(0, "/home1/irteam/data-vol1/projects/dft-dataset/src")


# ── Test molecules ──────────────────────────────────────────────────

MOLECULES = {
    "water": Atoms(
        "H2O",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
    ),
    "ethanol": Atoms(
        "C2H6O",
        positions=[
            [-0.045, 1.398, 0.0], [-0.045, -0.082, 0.0], [1.346, -0.541, 0.0],
            [0.480, 1.812, 0.878], [0.480, 1.812, -0.878], [-1.072, 1.770, 0.0],
            [-0.570, -0.465, 0.878], [-0.570, -0.465, -0.878], [1.830, 0.302, 0.0],
        ],
    ),
    "aspirin": Atoms(
        numbers=[6,6,6,6,6,6,8,6,8,8,6,8,8,1,1,1,1,1,1],
        positions=[
            [2.135,0.845,0],[0.750,0.845,0],[0.058,2.069,0],[0.750,3.293,0],
            [2.135,3.293,0],[2.828,2.069,0],[-0.635,0.845,0],[-1.327,2.069,0],
            [-2.520,2.069,0],[-0.635,3.293,0],[4.281,2.069,0],[4.875,3.200,0],
            [4.875,0.938,0],[2.635,-0.090,0],[0.250,-0.090,0],[0.250,4.228,0],
            [2.635,4.228,0],[-0.635,4.150,0],[5.940,0.938,0],
        ],
    ),
}


def dummy_predict(pyscf_mol, S, H_init):
    """Pass-through: return init_ham as prediction (for benchmarking pipeline)."""
    return H_init


def bench_qhflow(atoms, device, n_runs, use_gpu_fock=False, init_grid_level=None):
    """Benchmark QHFlowCalculator."""
    from md.qhflow_calculator import QHFlowCalculator

    calc = QHFlowCalculator(
        predict_hamiltonian=dummy_predict,
        basis="def2-SVP",
        xc="pbe",
        init_xc="pbe",
        density_fit=True,
        init_density_fit=True,
        init_grid_level=init_grid_level,
    )

    # If GPU Fock, monkey-patch _build_init_pyscf to use get_init_fock_gpu
    if use_gpu_fock:
        from fock import get_init_fock_gpu
        original_calculate = calc.calculate

        def patched_calculate(atoms=None, properties=None, system_changes=None, **kw):
            from ase.calculators.calculator import Calculator, all_changes
            from ase.units import Hartree, Bohr
            from solvers import build_density_matrix
            from molecule import Molecule

            if properties is None:
                properties = calc.implemented_properties
            if system_changes is None:
                system_changes = all_changes
            Calculator.calculate(calc, atoms, properties, system_changes)
            calc.time_dict = {}

            # 1. Build target mf
            t0 = time.time()
            pyscf_mol, mf = calc._build_pyscf(atoms, spin=kw.get("spin", 0))
            calc._pyscf_mol = pyscf_mol
            calc._mf = mf
            calc.time_dict["mf_time"] = time.time() - t0

            # 2. Overlap
            t0 = time.time()
            S = pyscf_mol.intor("int1e_ovlp")
            calc.time_dict["ovlp_time"] = time.time() - t0

            # 3. Init Hamiltonian via GPU Fock
            t0 = time.time()
            _, init_mf = calc._build_init_pyscf(atoms)
            init_dm = init_mf.init_guess_by_minao()
            H_init = get_init_fock_gpu(
                init_mf.mol, init_dm, xc=calc.init_xc,
                mf=init_mf, device=device,
            )
            calc.init_hamiltonian = H_init
            calc.time_dict["init_ham_time"] = time.time() - t0

            # 4. Predict
            t0 = time.time()
            H_pred = calc.predict_hamiltonian(pyscf_mol, S, H_init)
            calc.time_dict["pred_ham_time"] = time.time() - t0

            # 5. Eigensolve
            t0 = time.time()
            n_electrons = pyscf_mol.nelectron
            mo_energy, mo_coeff = calc._eigensolve_with_filtering(H_pred, S, n_electrons)
            D = build_density_matrix(mo_coeff, n_electrons)
            calc.time_dict["eigensolve_time"] = time.time() - t0

            # 6. Energy
            t0 = time.time()
            energy_ha = mf.energy_tot(D)
            calc.time_dict["energy_time"] = time.time() - t0

            # Store
            molecule = Molecule(
                atomic_numbers=atoms.get_atomic_numbers(),
                positions=atoms.get_positions(),
                energy_unit="Ha", length_unit="Ang",
            )
            molecule.overlap = S
            molecule.hamiltonian = H_pred
            molecule.orbital_energies = mo_energy
            molecule.orbital_coeffs = mo_coeff
            molecule.density_matrix = D
            molecule.energy = float(energy_ha)
            n_occ = n_electrons // 2
            molecule.homo = float(mo_energy[n_occ - 1])
            molecule.lumo = float(mo_energy[n_occ])
            calc.molecule = molecule
            calc._props = {"homo": molecule.homo, "lumo": molecule.lumo}
            calc.results["energy"] = float(energy_ha) * Hartree
            calc.results["free_energy"] = float(energy_ha) * Hartree

            # 7. Forces
            if "forces" in properties:
                t0 = time.time()
                mo_occ = mf.get_occ(mo_energy, mo_coeff)
                grad = mf.nuc_grad_method()
                grad.base.auxbasis_response = True
                forces = -grad.kernel(mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
                calc.time_dict["grad_time"] = time.time() - t0
                calc.results["forces"] = forces * Hartree / Bohr

        calc.calculate = patched_calculate

    # Warm up
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    atoms_copy.get_potential_energy()
    atoms_copy.get_forces()

    # Benchmark
    timings = []
    for _ in range(n_runs):
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        t0 = time.time()
        atoms_copy.get_potential_energy()
        atoms_copy.get_forces()
        total = time.time() - t0
        timings.append({**calc.time_dict, "total": total})

    return timings


def bench_pyscf_reference(atoms, n_runs):
    """Benchmark pure PySCF SCF (ground truth speed)."""
    from pyscf import gto, dft
    from ase.units import Hartree, Bohr

    mol = gto.M(
        atom=[(s, p) for s, p in zip(
            atoms.get_chemical_symbols(), atoms.get_positions()
        )],
        basis="def2-SVP", unit="Ang", verbose=0,
    )
    mf = dft.RKS(mol).density_fit()
    mf.xc = "pbe"

    # Warm up
    mf.kernel()
    mf.nuc_grad_method().kernel()

    timings = []
    for _ in range(n_runs):
        t0 = time.time()
        mf2 = dft.RKS(mol).density_fit()
        mf2.xc = "pbe"
        mf2.kernel()
        mf2.nuc_grad_method().kernel()
        timings.append(time.time() - t0)
    return timings


def format_time(ms):
    if ms < 1:
        return f"{ms*1000:.0f}μs"
    return f"{ms:.0f}ms"


def main():
    parser = argparse.ArgumentParser(description="Benchmark QHFlowCalculator")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--molecules", nargs="+", default=list(MOLECULES.keys()))
    args = parser.parse_args()

    print("=" * 90)
    print("QHFlowCalculator Benchmark: Energy + Force Pipeline")
    print(f"Device: {args.device}, Runs: {args.n_runs}")
    print("=" * 90)

    for mol_name in args.molecules:
        atoms = MOLECULES[mol_name]
        from pyscf import gto
        mol = gto.M(
            atom=[(s, p) for s, p in zip(
                atoms.get_chemical_symbols(), atoms.get_positions()
            )],
            basis="def2-SVP", unit="Ang", verbose=0,
        )
        print(f"\n{'─' * 90}")
        print(f"  {mol_name.upper()}: {mol.natm} atoms, {mol.nao} AOs, def2-SVP/PBE")
        print(f"{'─' * 90}")

        # 1. PySCF reference (full SCF)
        scf_times = bench_pyscf_reference(atoms, args.n_runs)
        scf_avg = np.mean(scf_times) * 1000

        # 2. QHFlowCalculator (PySCF init_ham)
        t_cpu = bench_qhflow(atoms, args.device, args.n_runs)
        cpu_avg = {k: np.mean([t[k] for t in t_cpu]) * 1000 for k in t_cpu[0]}

        # 3. QHFlowCalculator + GPU Fock
        t_gpu = bench_qhflow(atoms, args.device, args.n_runs, use_gpu_fock=True)
        gpu_avg = {k: np.mean([t[k] for t in t_gpu]) * 1000 for k in t_gpu[0]}

        # Print step-by-step comparison
        steps = ["mf_time", "ovlp_time", "init_ham_time", "pred_ham_time",
                 "eigensolve_time", "energy_time", "grad_time", "total"]
        labels = {
            "mf_time": "Build mf",
            "ovlp_time": "Overlap S",
            "init_ham_time": "Init H (Fock)",
            "pred_ham_time": "Model forward",
            "eigensolve_time": "Eigensolve + D",
            "energy_time": "E = mf.energy_tot",
            "grad_time": "Forces (grad)",
            "total": "TOTAL",
        }

        print(f"\n  {'Step':<22} {'PySCF init':>12} {'GPU Fock':>12} {'Δ':>10}")
        print(f"  {'─' * 56}")
        for step in steps:
            if step not in cpu_avg:
                continue
            label = labels.get(step, step)
            c, g = cpu_avg[step], gpu_avg[step]
            delta = c - g
            marker = "  ◀" if step == "init_ham_time" else ""
            if step == "total":
                print(f"  {'─' * 56}")
            print(f"  {label:<22} {format_time(c):>12} {format_time(g):>12} {delta:>+9.0f}ms{marker}")

        speedup = cpu_avg["total"] / gpu_avg["total"]
        print(f"\n  Speedup (GPU Fock):  {speedup:.2f}x")
        print(f"  PySCF full SCF:     {format_time(scf_avg)} (reference)")

    print(f"\n{'=' * 90}")
    print("Note: 'Model forward' uses pass-through (H_pred = H_init).")
    print("Real model inference will add to pred_ham_time.")
    print("=" * 90)


if __name__ == "__main__":
    main()
