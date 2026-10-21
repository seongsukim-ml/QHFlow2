"""
SCF Acceleration benchmark from pre-saved pred_flat/ directories.

Reads pred_flat/*.pt predictions (pred_hamiltonian, overlap, atoms, pos)
and runs DFT with/without predicted init to measure SCF cycle reduction.

By default only runs pred-init DFT. Use --with_scratch to also run baseline.

Usage:
    conda activate util-gpu4pyscf
    CUDA_VISIBLE_DEVICES=5 python scripts/benchmark_scf_from_pred_flat.py \
        --pred_dir rebuttal/.../pred_flat \
        --label QHFlow_260k --num_samples 500

    # With scratch baseline:
    python scripts/benchmark_scf_from_pred_flat.py \
        --pred_dir rebuttal/.../pred_flat \
        --label QHFlow_260k --num_samples 500 --with_scratch

    # Reuse cached scratch baseline:
    python scripts/benchmark_scf_from_pred_flat.py \
        --pred_dir rebuttal/.../pred_flat \
        --label QHNet_residue --num_samples 500 \
        --scratch_cache rebuttal/scf_acceleration/QHFlow_260k/scratch_results.json
"""

import os

_GPU_AVAILABLE = False
try:
    import gpu4pyscf
    from gpu4pyscf.dft import rks as gpu_rks
    _GPU_AVAILABLE = True
except Exception:
    pass

import sys
import time
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy.linalg import eigh
from tqdm import tqdm
from datetime import datetime
from pyscf import gto, dft

ELEMENT_MAP = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


BASIS = "def2svp"
XC = "b3lyp5"
CONV_TOL = 1e-9
MAX_CYCLE = 200


def build_mol(atoms, pos):
    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom=[[ELEMENT_MAP[int(a)], p] for a, p in zip(atoms, pos)],
        basis=BASIS, unit="ang",
    )
    return mol


def ham_to_density(H, S, n_elec):
    eigvals, eigvecs = eigh(H, S)
    n_occ = n_elec // 2
    return 2.0 * eigvecs[:, :n_occ] @ eigvecs[:, :n_occ].T


def run_scf(mol, dm0=None):
    if _GPU_AVAILABLE:
        mf = gpu_rks.RKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = XC
    mf.conv_tol = CONV_TOL
    mf.max_cycle = MAX_CYCLE
    mf.grids.prune = dft.gen_grid.nwchem_prune
    mf.verbose = 0

    t0 = time.perf_counter()
    if dm0 is not None:
        mf.kernel(dm0=dm0)
    else:
        mf.kernel()
    wall = time.perf_counter() - t0

    return {
        "converged": bool(mf.converged),
        "cycles": int(mf.cycles) if hasattr(mf, "cycles") else -1,
        "e_tot": float(mf.e_tot),
        "wall_time": wall,
    }


def load_pred_flat(pred_dir, indices):
    """Load pred_flat/*.pt files for given indices."""
    predictions = []
    for idx in tqdm(indices, desc="  Loading predictions"):
        pt_path = pred_dir / f"pred_{idx}.pt"
        if not pt_path.exists():
            continue
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        entry = {
            "idx": idx,
            "H_pred": d["pred_hamiltonian"].numpy().astype(np.float64),
            "S": d["overlap"].numpy().astype(np.float64),
            "atoms": d["atoms"].numpy().flatten(),
            "pos": d["pos"].numpy(),
        }
        if "pred_density_matrix" in d:
            entry["P_pred"] = d["pred_density_matrix"].numpy().astype(np.float64)
        predictions.append(entry)
    return predictions


def save_checkpoint(output_dir, phase, pred_results=None, scratch_results=None,
                    eigensolve_times=None, predictions=None, config=None):
    """Save intermediate results after each phase completes."""
    ckpt = {"phase": phase, "config": config}
    if eigensolve_times is not None:
        ckpt["eigensolve_times"] = eigensolve_times
    if pred_results is not None:
        ckpt["pred_results"] = pred_results
    if scratch_results is not None:
        ckpt["scratch_results"] = scratch_results
    if predictions is not None:
        # Save only metadata (idx, atoms, n_atoms), not full matrices
        ckpt["pred_meta"] = [{"idx": p["idx"], "n_atoms": len(p["atoms"])} for p in predictions]

    ckpt_path = output_dir / "checkpoint.json"
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f, cls=NumpyEncoder)
    print(f"  [checkpoint saved: {phase}]")


def main():
    parser = argparse.ArgumentParser(description="SCF acceleration from pred_flat/")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Path to pred_flat/ directory")
    parser.add_argument("--label", type=str, required=True,
                        help="Model label for output naming")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with_scratch", action="store_true",
                        help="Also run scratch baseline DFT")
    parser.add_argument("--scratch_cache", type=str, default=None,
                        help="Path to scratch_results.json to reuse")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    assert pred_dir.exists(), f"Not found: {pred_dir}"

    # Output dir (create early for checkpoints)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("rebuttal") / "scf_acceleration" / f"{args.label}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count available predictions
    all_pt = sorted(pred_dir.glob("pred_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    n_available = len(all_pt)
    print(f"{'='*70}")
    print(f"SCF Acceleration from pred_flat/")
    print(f"  Label: {args.label}")
    print(f"  Predictions: {n_available} available, using {min(args.num_samples, n_available)}")
    print(f"  DFT backend: {'gpu4pyscf (GPU)' if _GPU_AVAILABLE else 'PySCF (CPU)'}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    config = {
        "label": args.label,
        "pred_dir": str(pred_dir),
        "num_samples": min(args.num_samples, n_available),
        "seed": args.seed,
        "basis": BASIS, "xc": XC, "conv_tol": CONV_TOL,
        "dft_backend": "gpu4pyscf" if _GPU_AVAILABLE else "pyscf_cpu",
        "timestamp": timestamp,
    }

    # Sample indices
    all_indices = [int(p.stem.split("_")[1]) for p in all_pt]
    rng = np.random.default_rng(args.seed)
    selected = sorted(rng.choice(all_indices, size=min(args.num_samples, n_available), replace=False))

    # ── Phase 1: Load predictions ──
    print("Phase 1: Load predictions")
    predictions = load_pred_flat(pred_dir, selected)
    n_mols = len(predictions)
    print(f"  Loaded {n_mols} molecules\n")

    # ── Phase 2: Eigensolve ──
    print("Phase 2: Build molecules & eigensolve")
    molecules = []
    density_matrices = []
    eigensolve_times = []

    has_P_pred = "P_pred" in predictions[0]
    if has_P_pred:
        print("  Using predicted density matrix (P_pred) directly — skipping eigensolve")

    for pred in tqdm(predictions, desc="  Eigensolve"):
        mol = build_mol(pred["atoms"], pred["pos"])
        molecules.append(mol)

        if has_P_pred:
            dm0 = pred["P_pred"]
            eigensolve_times.append(0.0)
        else:
            n_elec = mol.nelectron
            t0 = time.perf_counter()
            dm0 = ham_to_density(pred["H_pred"], pred["S"], n_elec)
            t1 = time.perf_counter()
            eigensolve_times.append(t1 - t0)
        density_matrices.append(dm0)

    total_eigen = sum(eigensolve_times)
    print(f"  Total: {total_eigen:.3f}s, Per-mol: {total_eigen/n_mols*1000:.1f}ms\n")

    save_checkpoint(output_dir, "eigensolve", eigensolve_times=eigensolve_times,
                    predictions=predictions, config=config)

    # GPU warmup
    print("  DFT warmup...")
    run_scf(molecules[0])
    run_scf(molecules[0])
    print("  Warmup done.\n")

    # ── Phase 3: DFT pred-init ──
    print("Phase 3: DFT with pred-init")
    pred_results = []
    for i in tqdm(range(n_mols), desc="  DFT pred-init"):
        result = run_scf(molecules[i], dm0=density_matrices[i])
        result["n_atoms"] = len(predictions[i]["atoms"])
        result["idx"] = predictions[i]["idx"]
        pred_results.append(result)

        # Periodic checkpoint every 100 molecules
        if (i + 1) % 100 == 0:
            save_checkpoint(output_dir, f"pred_init_{i+1}/{n_mols}",
                            pred_results=pred_results, eigensolve_times=eigensolve_times,
                            predictions=predictions, config=config)

    pred_times = [r["wall_time"] for r in pred_results]
    pred_cycles = [r["cycles"] for r in pred_results]
    total_pred = sum(pred_times)
    print(f"  Total: {total_pred:.3f}s, Per-mol: {total_pred/n_mols*1000:.1f}ms")
    print(f"  SCF cycles: mean={np.mean(pred_cycles):.1f}, median={np.median(pred_cycles):.0f}\n")

    save_checkpoint(output_dir, "pred_init_done", pred_results=pred_results,
                    eigensolve_times=eigensolve_times, predictions=predictions, config=config)

    # ── Phase 4: DFT scratch (optional) ──
    scratch_results = None
    if args.scratch_cache and Path(args.scratch_cache).exists():
        print("Phase 4: Loading cached scratch results")
        with open(args.scratch_cache) as f:
            scratch_data = json.load(f)
        scratch_results = scratch_data["scratch_results"]
        print(f"  Loaded {len(scratch_results)} cached scratch results\n")
    elif args.with_scratch:
        print("Phase 4: DFT from scratch (baseline)")
        scratch_results = []
        for i in tqdm(range(n_mols), desc="  DFT scratch"):
            result = run_scf(molecules[i])
            result["n_atoms"] = len(predictions[i]["atoms"])
            result["idx"] = predictions[i]["idx"]
            scratch_results.append(result)

            if (i + 1) % 100 == 0:
                save_checkpoint(output_dir, f"scratch_{i+1}/{n_mols}",
                                pred_results=pred_results, scratch_results=scratch_results,
                                eigensolve_times=eigensolve_times, predictions=predictions,
                                config=config)

        scratch_times = [r["wall_time"] for r in scratch_results]
        scratch_cycles = [r["cycles"] for r in scratch_results]
        total_scratch = sum(scratch_times)
        print(f"  Total: {total_scratch:.3f}s, Per-mol: {total_scratch/n_mols*1000:.1f}ms")
        print(f"  SCF cycles: mean={np.mean(scratch_cycles):.1f}, median={np.median(scratch_cycles):.0f}\n")
    else:
        print("Phase 4: SKIPPED (use --with_scratch or --scratch_cache to include baseline)\n")

    # ── Summary ──
    print(f"{'='*70}")
    print(f"SUMMARY: {args.label}, {n_mols} molecules")
    print(f"DFT backend: {'gpu4pyscf (GPU)' if _GPU_AVAILABLE else 'PySCF (CPU)'}")
    print(f"{'='*70}")

    if scratch_results is not None:
        scratch_times = [r["wall_time"] for r in scratch_results]
        scratch_cycles = [r["cycles"] for r in scratch_results]
        total_scratch = sum(scratch_times)
        energy_diffs = [abs(pr["e_tot"] - sr["e_tot"]) for pr, sr in zip(pred_results, scratch_results)]

        total_dft_pipeline = total_eigen + total_pred
        cycle_reduction = 100 * (1 - np.mean(pred_cycles) / np.mean(scratch_cycles))

        print()
        print(f"{'Stage':<28s} {'Total (s)':>10s} {'Per-mol (ms)':>13s}")
        print(f"{'-'*55}")
        print(f"{'Eigensolve':<28s} {total_eigen:>10.3f} {total_eigen/n_mols*1000:>13.1f}")
        print(f"{'DFT (pred-init)':<28s} {total_pred:>10.3f} {total_pred/n_mols*1000:>13.1f}")
        print(f"{'-'*55}")
        print(f"{'Total (excl. inference)':<28s} {total_dft_pipeline:>10.3f} {total_dft_pipeline/n_mols*1000:>13.1f}")
        print(f"{'DFT (scratch)':<28s} {total_scratch:>10.3f} {total_scratch/n_mols*1000:>13.1f}")
        print()
        print(f"SCF Cycles: pred={np.mean(pred_cycles):.1f}  scratch={np.mean(scratch_cycles):.1f}  "
              f"reduction={cycle_reduction:.1f}%")
        print(f"Energy consistency: mean |dE| = {np.mean(energy_diffs):.2e} Ha, max = {np.max(energy_diffs):.2e} Ha")
    else:
        energy_diffs = None
        scratch_cycles = None
        total_scratch = None
        print()
        print(f"SCF Cycles (pred-init): mean={np.mean(pred_cycles):.1f}, median={np.median(pred_cycles):.0f}")

    # ── Save final results ──
    per_sample = []
    for i in range(n_mols):
        entry = {
            "idx": predictions[i]["idx"],
            "n_atoms": int(pred_results[i]["n_atoms"]),
            "pred_init_time": pred_results[i]["wall_time"],
            "pred_init_cycles": pred_results[i]["cycles"],
            "pred_init_energy": pred_results[i]["e_tot"],
            "pred_init_converged": pred_results[i]["converged"],
            "eigensolve_time": eigensolve_times[i],
        }
        if scratch_results is not None:
            entry.update({
                "scratch_time": scratch_results[i]["wall_time"],
                "scratch_cycles": scratch_results[i]["cycles"],
                "scratch_energy": scratch_results[i]["e_tot"],
                "scratch_converged": scratch_results[i]["converged"],
                "energy_diff_ha": energy_diffs[i],
            })
        per_sample.append(entry)

    results = {
        "config": config,
        "summary": {
            "eigensolve_total_s": total_eigen,
            "eigensolve_per_mol_ms": total_eigen / n_mols * 1000,
            "dft_pred_init_total_s": total_pred,
            "dft_pred_init_per_mol_ms": total_pred / n_mols * 1000,
            "pred_init_cycles_mean": float(np.mean(pred_cycles)),
            "pred_init_cycles_median": float(np.median(pred_cycles)),
        },
        "per_sample": per_sample,
    }

    if scratch_results is not None:
        results["summary"].update({
            "dft_scratch_total_s": total_scratch,
            "dft_scratch_per_mol_ms": total_scratch / n_mols * 1000,
            "scratch_cycles_mean": float(np.mean(scratch_cycles)),
            "scratch_cycles_median": float(np.median(scratch_cycles)),
            "cycle_reduction_pct": float(100 * (1 - np.mean(pred_cycles) / np.mean(scratch_cycles))),
            "energy_diff_mean_ha": float(np.mean(energy_diffs)),
            "energy_diff_max_ha": float(np.max(energy_diffs)),
        })
        # Save scratch results for reuse by other models
        with open(output_dir / "scratch_results.json", "w") as f:
            json.dump({"scratch_results": scratch_results}, f, cls=NumpyEncoder)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Clean up checkpoint after successful save
    ckpt_path = output_dir / "checkpoint.json"
    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
