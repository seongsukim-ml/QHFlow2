#!/usr/bin/env python3
"""
Rebuttal Q1: Analytical Force Consistency from Predicted vs GT Hamiltonian

Compares:
  1. F_analytical(pred H) — forces from non-self-consistent predicted Hamiltonian
  2. F_analytical(GT H)   — forces from self-consistent ground-truth Hamiltonian
  3. F_SCF(GT H)          — forces from full SCF with GT H as initial guess (reference)

If predicted H were perfectly self-consistent, (1) would equal (3).
The gap (1)-(2) quantifies the non-self-consistency effect.

Usage:
    python run_force_comparison.py [--n_mols 20] [--size small]
"""
import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "src"))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HA_PER_BOHR_TO_EV_PER_ANG = 51.42206313  # 1 Ha/Bohr = 51.42 eV/Å


def eigensolve(H, S, tol=1e-8):
    eigvals_s, U = np.linalg.eigh(S)
    mask = eigvals_s > tol
    X = U[:, mask] / np.sqrt(eigvals_s[mask])
    H_orth = X.T @ H @ X
    energies, C_orth = np.linalg.eigh(H_orth)
    return energies, X @ C_orth


def forces_from_H(positions, atoms_z, H, basis="def2-svp", xc="pbe", scf=False):
    """Compute energy + forces from H matrix via PySCF.

    Args:
        scf: If True, run full SCF with H-derived D as initial guess.
             If False, use H directly (non-SCF, Hellmann-Feynman + Pulay).
    Returns:
        energy (Ha), forces (Ha/Bohr), mo_energy, D
    """
    from pyscf import gto, dft
    from pyscf.data.elements import ELEMENTS

    atom_str = [(ELEMENTS[z], pos) for z, pos in zip(atoms_z, positions)]
    mol = gto.M(atom=atom_str, basis=basis, unit="Ang", verbose=0)
    mf = dft.RKS(mol)
    mf = mf.density_fit()
    mf.xc = xc

    S = mf.get_ovlp()
    mo_energy, mo_coeff = eigensolve(H, S)
    n_occ = mol.nelectron // 2
    D = 2.0 * mo_coeff[:, :n_occ] @ mo_coeff[:, :n_occ].T

    if scf:
        energy = mf.kernel(dm0=D, dump_chk=False)
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        D = mf.make_rdm1()
        forces = -mf.nuc_grad_method().kernel()
    else:
        energy = mf.energy_tot(D)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        grad = mf.nuc_grad_method()
        grad.base.auxbasis_response = True
        forces = -grad.kernel(mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)

    return energy, forces, mo_energy, D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_mols", type=int, default=20)
    parser.add_argument("--size", type=str, default="small")
    parser.add_argument("--basis", type=str, default="def2-svp")
    parser.add_argument("--xc", type=str, default="pbe")
    parser.add_argument("--scf", action="store_true", help="Also run full SCF for reference")
    args = parser.parse_args()

    pred_dir = PROJ / f"eval_results/QH9Stable-random/{args.size}/pred_flat"
    gt_dir = PROJ / f"eval_results/QH9Stable-random/{args.size}/gt_flat"

    if not pred_dir.exists():
        print(f"ERROR: {pred_dir} not found. Run eval first.")
        sys.exit(1)

    # Collect molecule indices
    indices = sorted(int(f.stem.split("_")[1]) for f in pred_dir.glob("pred_*.pt"))[:args.n_mols]
    print(f"Evaluating {len(indices)} molecules (size={args.size}, basis={args.basis}, xc={args.xc})")

    all_results = []

    for mol_idx in indices:
        pred = torch.load(str(pred_dir / f"pred_{mol_idx}.pt"), map_location="cpu", weights_only=False)
        gt = torch.load(str(gt_dir / f"gt_{mol_idx}.pt"), map_location="cpu", weights_only=False)

        positions = pred["pos"].double().numpy()
        atoms_z = pred["atoms"].squeeze(-1).numpy()
        H_pred = pred["pred_hamiltonian"].double().numpy()
        H_gt = gt["hamiltonian"].double().numpy()
        n_atoms = len(atoms_z)

        print(f"\nMol {mol_idx}: {n_atoms} atoms, nao={H_pred.shape[0]}")
        t0 = time.time()

        # 1. Non-SCF forces from predicted H
        E_pred, F_pred, mo_e_pred, D_pred = forces_from_H(
            positions, atoms_z, H_pred, args.basis, args.xc, scf=False
        )

        # 2. Non-SCF forces from GT H
        E_gt, F_gt, mo_e_gt, D_gt = forces_from_H(
            positions, atoms_z, H_gt, args.basis, args.xc, scf=False
        )

        # 3. SCF forces (full self-consistency) — optional reference
        E_scf, F_scf = None, None
        if args.scf:
            E_scf, F_scf, _, _ = forces_from_H(
                positions, atoms_z, H_gt, args.basis, args.xc, scf=True
            )

        elapsed = time.time() - t0

        # Metrics (all in Ha/Bohr)
        diff_pred_gt = np.abs(F_pred - F_gt)
        result = {
            "mol_idx": int(mol_idx),
            "n_atoms": int(n_atoms),
            "nao": int(H_pred.shape[0]),
            "E_pred": float(E_pred),
            "E_gt": float(E_gt),
            "F_pred_vs_gt_mae": float(diff_pred_gt.mean()),
            "F_pred_vs_gt_max": float(diff_pred_gt.max()),
            "F_pred_vs_gt_mae_eV_Ang": float(diff_pred_gt.mean() * HA_PER_BOHR_TO_EV_PER_ANG),
            "H_mae_ha": float(np.abs(H_pred - H_gt).mean()),
        }

        if F_scf is not None:
            diff_gt_scf = np.abs(F_gt - F_scf)
            diff_pred_scf = np.abs(F_pred - F_scf)
            result["F_gt_vs_scf_mae"] = float(diff_gt_scf.mean())
            result["F_pred_vs_scf_mae"] = float(diff_pred_scf.mean())
            result["E_scf"] = float(E_scf)

        all_results.append(result)
        print(f"  E(pred)={E_pred:.8f} E(GT)={E_gt:.8f} ΔE={abs(E_pred-E_gt):.2e} Ha")
        print(f"  F(pred) vs F(GT): MAE={result['F_pred_vs_gt_mae']:.6f} Ha/Bohr "
              f"= {result['F_pred_vs_gt_mae_eV_Ang']:.4f} eV/Å")
        if F_scf is not None:
            print(f"  F(GT) vs F(SCF):  MAE={result['F_gt_vs_scf_mae']:.6f} Ha/Bohr [self-consistency gap]")
            print(f"  F(pred) vs F(SCF): MAE={result['F_pred_vs_scf_mae']:.6f} Ha/Bohr")
        print(f"  ({elapsed:.1f}s)")

    # Save
    with open(RESULTS_DIR / "force_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)

    mae_pred_gt = np.mean([r["F_pred_vs_gt_mae"] for r in all_results])
    mae_pred_gt_ev = np.mean([r["F_pred_vs_gt_mae_eV_Ang"] for r in all_results])
    max_pred_gt = np.max([r["F_pred_vs_gt_max"] for r in all_results])
    h_mae = np.mean([r["H_mae_ha"] for r in all_results])
    de_mae = np.mean([abs(r["E_pred"] - r["E_gt"]) for r in all_results])

    print(f"  N molecules:           {len(all_results)}")
    print(f"  H MAE:                 {h_mae:.2e} Ha ({h_mae*1e6:.1f} μHa)")
    print(f"  ΔE (pred vs GT):       {de_mae:.2e} Ha ({de_mae*27211:.2f} meV)")
    print(f"  F(pred) vs F(GT) MAE:  {mae_pred_gt:.6f} Ha/Bohr ({mae_pred_gt_ev:.4f} eV/Å)")
    print(f"  F(pred) vs F(GT) max:  {max_pred_gt:.6f} Ha/Bohr")

    if args.scf and all_results[0].get("F_gt_vs_scf_mae") is not None:
        mae_gt_scf = np.mean([r["F_gt_vs_scf_mae"] for r in all_results])
        mae_pred_scf = np.mean([r["F_pred_vs_scf_mae"] for r in all_results])
        print(f"  F(GT) vs F(SCF) MAE:   {mae_gt_scf:.6f} Ha/Bohr [self-consistency baseline]")
        print(f"  F(pred) vs F(SCF) MAE: {mae_pred_scf:.6f} Ha/Bohr")

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
