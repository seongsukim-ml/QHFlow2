"""Quick benchmark: QHFlowCalculator pipeline timing breakdown."""

import sys
import time

import numpy as np
from ase import Atoms

import importlib.util
import os

_root = "/home1/irteam/data-vol1/projects"
sys.path.insert(0, f"{_root}/dft-dataset/src")
os.chdir(f"{_root}/QHFlow2/src")

# Load QHFlowCalculator via importlib (md/ has no __init__.py)
_spec = importlib.util.spec_from_file_location(
    "qhflow_calculator",
    f"{_root}/QHFlow2/src/qhflow2/md/qhflow_calculator.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
QHFlowCalculator = _mod.QHFlowCalculator

from fock import get_init_fock_gpu

MOLECULES = {
    "water": Atoms("H2O", positions=[[0,0,0],[0.96,0,0],[-0.24,0.93,0]]),
    "ethanol": Atoms("C2H6O", positions=[
        [-0.045,1.398,0],[-0.045,-0.082,0],[1.346,-0.541,0],
        [0.48,1.812,0.878],[0.48,1.812,-0.878],[-1.072,1.77,0],
        [-0.57,-0.465,0.878],[-0.57,-0.465,-0.878],[1.83,0.302,0],
    ]),
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

N = 3


def dummy_predict(mol, S, H):
    return H


def bench(atoms, use_gpu_fock=False):
    calc = QHFlowCalculator(
        predict_hamiltonian=dummy_predict,
        basis="def2-SVP", xc="pbe", init_xc="pbe",
        density_fit=True, init_density_fit=True,
    )
    # Warm up
    a = atoms.copy(); a.calc = calc
    a.get_potential_energy(); a.get_forces()

    times = []
    for _ in range(N):
        a = atoms.copy(); a.calc = calc
        t0 = time.time()
        a.get_potential_energy()
        a.get_forces()
        td = dict(calc.time_dict)
        td["total"] = time.time() - t0
        times.append(td)

    return {k: np.mean([t[k] for t in times]) * 1000 for k in times[0]}


def bench_gpu_fock(atoms):
    """Benchmark with GPU-accelerated init Fock replacing the init_ham step."""
    from pyscf import gto, dft

    # Build init mf once for reuse
    mol = gto.M(
        atom=[(s, p) for s, p in zip(
            atoms.get_chemical_symbols(), atoms.get_positions()
        )],
        basis="def2-SVP", unit="Ang", verbose=0,
    )
    init_mf = dft.RKS(mol).density_fit()
    init_mf.xc = "pbe"
    init_dm = init_mf.init_guess_by_minao()

    # Warm up GPU Fock
    _ = get_init_fock_gpu(mol, init_dm, xc="pbe", mf=init_mf, device="cuda")

    def predict_with_gpu_fock(pyscf_mol, S, H_init_ignored):
        """Replace PySCF init_ham with GPU Fock, then return as prediction."""
        return get_init_fock_gpu(
            pyscf_mol, init_dm, xc="pbe", mf=init_mf, device="cuda"
        )

    calc = QHFlowCalculator(
        predict_hamiltonian=predict_with_gpu_fock,
        basis="def2-SVP", xc="pbe", init_xc="pbe",
        density_fit=True, init_density_fit=True,
    )

    # Warm up
    a = atoms.copy(); a.calc = calc
    a.get_potential_energy(); a.get_forces()

    times = []
    for _ in range(N):
        a = atoms.copy(); a.calc = calc
        t0 = time.time()
        a.get_potential_energy()
        a.get_forces()
        td = dict(calc.time_dict)
        td["total"] = time.time() - t0
        # pred_ham_time now includes GPU Fock; init_ham_time is still PySCF (wasted)
        # For reporting, combine them
        td["gpu_fock_in_pred"] = td.get("pred_ham_time", 0)
        times.append(td)

    return {k: np.mean([t[k] for t in times]) * 1000 for k in times[0]}


print("=" * 85)
print("QHFlowCalculator Benchmark: Energy + Force Pipeline")
print(f"Model: pass-through (H_pred = H_init), Basis: def2-SVP, XC: PBE, Runs: {N}")
print("=" * 85)

for name, atoms in MOLECULES.items():
    from pyscf import gto
    mol = gto.M(
        atom=[(s, p) for s, p in zip(
            atoms.get_chemical_symbols(), atoms.get_positions()
        )],
        basis="def2-SVP", unit="Ang", verbose=0,
    )
    print(f"\n{'─' * 85}")
    print(f"  {name.upper()}: {mol.natm} atoms, {mol.nao} AOs")
    print(f"{'─' * 85}")

    cpu = bench(atoms)

    steps = [
        "mf_time", "ovlp_time", "init_ham_time", "pred_ham_time",
        "eigensolve_time", "energy_time", "grad_time", "total",
    ]
    labels = {
        "mf_time": "Build mf",
        "ovlp_time": "Overlap S",
        "init_ham_time": "Init H (Fock)",
        "pred_ham_time": "Model forward",
        "eigensolve_time": "Eigensolve + D",
        "energy_time": "E = energy_tot",
        "grad_time": "Forces (grad)",
        "total": "TOTAL",
    }

    print(f"\n  {'Step':<22} {'Time':>10} {'%':>6}")
    print(f"  {'─' * 40}")
    for s in steps:
        if s not in cpu:
            continue
        l = labels.get(s, s)
        v = cpu[s]
        pct = v / cpu["total"] * 100
        if s == "total":
            print(f"  {'─' * 40}")
        print(f"  {l:<22} {v:>9.0f}ms {pct:>5.1f}%")

    # Throughput
    fps = 1000.0 / cpu["total"]
    print(f"\n  Throughput: {fps:.2f} E+F/s")

print(f"\n{'=' * 85}")
print("Note: 'Model forward' = 0ms (pass-through). Real model adds to this.")
print("Init H (Fock) is the target for GPU acceleration.")
print("=" * 85)
