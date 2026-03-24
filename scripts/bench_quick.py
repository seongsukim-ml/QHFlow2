"""Benchmark: QHFlowCalculator E+F pipeline timing breakdown.

Usage:
    # Dummy model (pipeline overhead only):
    conda run -n experiments python scripts/bench_quick.py

    # With actual QHFlow2 model (requires qhflow env + working fairchem):
    conda run -n qhflow python scripts/bench_quick.py --model outputs/.../last.ckpt

    # Specific molecules:
    python scripts/bench_quick.py --molecules water ethanol
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
from ase import Atoms

_root = "/home1/irteam/data-vol1/projects"
sys.path.insert(0, f"{_root}/dft-dataset/src")
os.chdir(f"{_root}/QHFlow2/src")
if f"{_root}/QHFlow2/src" not in sys.path:
    sys.path.insert(0, f"{_root}/QHFlow2/src")

# Load QHFlowCalculator
_spec = importlib.util.spec_from_file_location(
    "qhflow_calculator",
    f"{_root}/QHFlow2/src/qhflow2/md/qhflow_calculator.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
QHFlowCalculator = _mod.QHFlowCalculator

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


def dummy_predict(mol, S, H):
    """Pass-through: return H_init as prediction."""
    return H


def load_scflow_model(ckpt_path, device="cuda", data_type="qh9", xc="b3lyp"):
    """Load SCFlowRKSCalculator with a trained checkpoint.

    Returns a predict_hamiltonian callable compatible with QHFlowCalculator.
    """
    from qhflow2.md.scflow_calculator import SCFlowRKSCalculator

    calc = SCFlowRKSCalculator(device=device, basis="def2-SVP", functional=xc)
    calc.set_model(
        ckpt_path,
        model_type="flow",
        device=device,
        ode_steps=1,
        data_type=data_type,
        units="ang",
        model_length_unit="ang",
        mf_init_functional=xc,
        density_fit=True,
        init_density_fit=True,
        filtering=False,
        gt_tol=1e-8,
    )

    def predict_H(pyscf_mol, S, H_init):
        """Wrap SCFlowRKSCalculator.forward_component."""
        from pyscf import gto
        atoms_list = [gto.elements.ELEMENTS[pyscf_mol.atom_charge(i)]
                      for i in range(pyscf_mol.natm)]
        coords = pyscf_mol.atom_coords() * 0.529177249  # Bohr → Ang
        return calc.forward_component(atoms_list, coords, S, H_init)

    return predict_H, calc


def bench(atoms, predict_fn, xc="pbe", n_runs=3):
    """Benchmark a single configuration."""
    from ase.calculators.calculator import all_changes

    calc = QHFlowCalculator(
        predict_hamiltonian=predict_fn,
        basis="def2-SVP", xc=xc, init_xc=xc,
        density_fit=True, init_density_fit=True,
    )

    # Warm up
    a = atoms.copy(); a.calc = calc
    calc.calculate(a, ["energy", "forces"], all_changes)

    times = []
    for _ in range(n_runs):
        a = atoms.copy(); a.calc = calc
        t0 = time.time()
        calc.calculate(a, ["energy", "forces"], all_changes)
        td = dict(calc.time_dict)
        td["total"] = time.time() - t0
        times.append(td)

    avg = {k: np.mean([t[k] for t in times]) * 1000 for k in times[0]}
    std = {k: np.std([t[k] for t in times]) * 1000 for k in times[0]}
    return avg, std, calc


def print_table(avg, label=""):
    steps = [
        ("mf_time", "Build mf"),
        ("ovlp_time", "Overlap S"),
        ("init_ham_time", "Init H (Fock)"),
        ("pred_ham_time", "Model forward"),
        ("eigensolve_time", "Eigensolve + D"),
        ("energy_time", "E = energy_tot"),
        ("grad_time", "Forces (grad)"),
        ("total", "TOTAL"),
    ]

    if label:
        print(f"\n  [{label}]")
    print(f"  {'Step':<22} {'Time':>10} {'%':>6}")
    print(f"  {'─' * 40}")
    for key, name in steps:
        if key not in avg:
            continue
        v = avg[key]
        pct = v / avg["total"] * 100
        if key == "total":
            print(f"  {'─' * 40}")
        print(f"  {name:<22} {v:>9.0f}ms {pct:>5.1f}%")

    fps = 1000.0 / avg["total"]
    print(f"\n  Throughput: {fps:.2f} E+F/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark QHFlowCalculator")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .ckpt for real model benchmark")
    parser.add_argument("--data_type", default="qh9")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--molecules", nargs="+", default=list(MOLECULES.keys()))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load model if provided
    model_predict = None
    model_label = "dummy (pass-through)"
    if args.model:
        print(f"Loading model: {args.model}")
        model_predict, scflow_calc = load_scflow_model(
            args.model, device=args.device,
            data_type=args.data_type, xc=args.xc,
        )
        n_params = sum(p.numel() for p in scflow_calc.model.parameters())
        model_label = f"{scflow_calc.conf.model.model_name} ({n_params:,} params)"
        print(f"  Model: {model_label}")

    print(f"\n{'=' * 80}")
    print(f"QHFlowCalculator Benchmark: Energy + Force Pipeline")
    print(f"  Model: {model_label}")
    print(f"  Basis: def2-SVP, XC: {args.xc}, Runs: {args.n_runs}")
    print(f"{'=' * 80}")

    for name in args.molecules:
        atoms = MOLECULES[name]
        from pyscf import gto
        mol = gto.M(
            atom=[(s, p) for s, p in zip(
                atoms.get_chemical_symbols(), atoms.get_positions()
            )],
            basis="def2-SVP", unit="Ang", verbose=0,
        )
        print(f"\n{'─' * 80}")
        print(f"  {name.upper()}: {mol.natm} atoms, {mol.nao} AOs")
        print(f"{'─' * 80}")

        # Always bench dummy first (pipeline overhead)
        avg_dummy, _, _ = bench(atoms, dummy_predict, xc=args.xc, n_runs=args.n_runs)
        print_table(avg_dummy, "Dummy (pipeline overhead)")

        # Bench with real model if loaded
        if model_predict:
            avg_model, _, calc_model = bench(
                atoms, model_predict, xc=args.xc, n_runs=args.n_runs,
            )
            print_table(avg_model, "Model inference")

            # Summary comparison
            overhead = avg_dummy["total"] - avg_dummy.get("pred_ham_time", 0)
            model_time = avg_model.get("pred_ham_time", 0)
            print(f"\n  Pipeline overhead:  {overhead:.0f}ms")
            print(f"  Model inference:    {model_time:.0f}ms")
            print(f"  Model / Total:      {model_time / avg_model['total'] * 100:.1f}%")

    print(f"\n{'=' * 80}")
    if not args.model:
        print("Tip: run with --model <checkpoint.ckpt> to include model inference time")
    print("=" * 80)


if __name__ == "__main__":
    main()
