#!/usr/bin/env python
"""nablaDFT LMDB → QHFlow2 preprocessed LMDB.

Reads dft-dataset Molecule objects from nablaDFT LMDB, computes:
  - Convention transform: Psi4 → PySCF (d reorder)
  - hcore (T+V) as initial Hamiltonian via PySCF integrals
  - Orbital energies, coefficients via eigensolve
  - Packed upper-triangle matrices

Output: QHFlow2-compatible LMDB with pickled dicts.

Usage:
    python preprocess_nabladft.py --input test_2k_conformers --workers 16
    python preprocess_nabladft.py --input train_100k --workers 32
"""

import argparse
import json
import lmdb
import os
import pickle
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src" / "qhflow2"))
sys.path.insert(0, "/home1/irteam/data-vol1/projects/dft-dataset/src")

NABLA_LMDB_DIR = Path("/home1/irteam/data-vol1/datasets/nablaDFT/lmdb")
OUTPUT_DIR = Path("/home1/irteam/data-vol1/datasets/nablaDFT/qhflow2_processed")

# ── Worker globals (initialized per-process) ──────────────────────

_worker_src_ds = None   # dft-dataset LMDBDataset (read-only)
_worker_conv = None     # Psi4→PySCF convention


def _worker_init(src_lmdb_path):
    """Called once per worker process to open LMDB handle."""
    global _worker_src_ds, _worker_conv
    from dft_dataset.lmdb_dataset import LMDBDataset
    from common.orbital_conventions import get_all_conventions
    _worker_src_ds = LMDBDataset(str(src_lmdb_path))
    _worker_conv = get_all_conventions()["psi4_def2svp_to_pyscf"]


def _reorder_psi4_to_pyscf(matrix, atoms):
    from common.matrix_transforms import matrix_transform_single
    mat = torch.from_numpy(matrix).to(torch.float64)
    at = torch.from_numpy(atoms).long()
    return matrix_transform_single(mat, at, convention=_worker_conv).numpy()


def process_single(idx):
    """Process sample at index `idx`. Worker reads from its own LMDB handle."""
    try:
        mol = _worker_src_ds[idx]
        atoms = np.array(mol.atomic_numbers, dtype=np.int32)
        pos = np.array(mol.positions, dtype=np.float64)
        n_atoms = len(atoms)

        H_psi4 = np.array(mol.hamiltonian, dtype=np.float64)
        S_psi4 = np.array(mol.overlap, dtype=np.float64)
        h_dim = H_psi4.shape[0]

        # Convention: Psi4 → PySCF
        H_pyscf = _reorder_psi4_to_pyscf(H_psi4, atoms)
        S_pyscf = _reorder_psi4_to_pyscf(S_psi4, atoms)

        # hcore (T+V) as initial Hamiltonian — ~0.01s per molecule
        from common.dft_utils import init_pyscf_mol
        pyscf_mol = init_pyscf_mol(atoms, pos.reshape(-1, 3), unit="ang", basis="def2svp")
        init_ham = (pyscf_mol.intor("int1e_kin") + pyscf_mol.intor("int1e_nuc")).astype("float64")

        # Pack matrices (skip eigensolve — not used by NablaDFTDataset)
        from common.matrix_transforms import pack_upper_triangle
        packed_H, _ = pack_upper_triangle(H_pyscf)
        packed_S, _ = pack_upper_triangle(S_pyscf)
        packed_init, _ = pack_upper_triangle(init_ham)

        energy = float(mol.energy) if mol.energy is not None else 0.0
        forces = np.array(mol.forces, dtype=np.float64) if mol.forces is not None else np.zeros((n_atoms, 3))

        result = {
            "id": idx,
            "num_nodes": n_atoms,
            "atoms": atoms.tobytes(),
            "pos": pos.tobytes(),
            "dft_energy": energy,
            "dft_forces": forces.tobytes(),
            "h_dim": h_dim,
            "packed_hamiltonian": packed_H.tobytes() if isinstance(packed_H, np.ndarray) else packed_H.numpy().tobytes(),
            "packed_overlap": packed_S.tobytes() if isinstance(packed_S, np.ndarray) else packed_S.numpy().tobytes(),
            "packed_initial_hamiltonian": packed_init.tobytes() if isinstance(packed_init, np.ndarray) else packed_init.numpy().tobytes(),
            "pos_unit": "Angstrom",
            "init_ham_method": "hcore",
        }
        return idx, result, None

    except Exception as e:
        return idx, None, f"{e}\n{traceback.format_exc()}"


def preprocess_split(split_name, max_workers=8, max_samples=None):
    input_path = NABLA_LMDB_DIR / f"{split_name}.lmdb"
    output_path = OUTPUT_DIR / f"{split_name}.lmdb"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found"); return
    if output_path.exists():
        print(f"SKIP: {output_path} already exists"); return

    # Get total count without loading all data
    from dft_dataset.lmdb_dataset import LMDBDataset
    src = LMDBDataset(str(input_path))
    total = min(len(src), max_samples) if max_samples else len(src)
    del src  # close before forking

    print(f"\n{'='*60}")
    print(f"  Preprocessing: {split_name}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Samples: {total}")
    print(f"  Workers: {max_workers}")
    print(f"  Init ham: hcore (T+V)")
    print(f"{'='*60}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_size = max(total * 2_000_000, 1 << 30)  # ~2MB per sample (378^2 packed matrices × 3)
    env = lmdb.open(str(output_path), map_size=map_size, subdir=False)

    t0 = time.time()
    n_ok, n_err = 0, 0
    indices = list(range(total))

    if max_workers <= 1:
        _worker_init(input_path)
        results = map(process_single, indices)
    else:
        pool = Pool(max_workers, initializer=_worker_init, initargs=(input_path,))
        results = pool.imap_unordered(process_single, indices, chunksize=10)

    with env.begin(write=True) as txn:
        for idx, result, error in results:
            if error:
                n_err += 1
                if n_err <= 5:
                    print(f"  [ERROR] sample {idx}: {error[:200]}")
            else:
                txn.put(str(n_ok).encode(), pickle.dumps(result))
                n_ok += 1

            done = n_ok + n_err
            if done % 200 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] ok={n_ok} err={n_err} "
                      f"rate={rate:.1f}/s ETA={eta/60:.1f}min")

        txn.put(b"__len__", str(n_ok).encode())
        metadata = {
            "source": "nablaDFT",
            "split": split_name,
            "n_samples": n_ok,
            "n_errors": n_err,
            "basis": "def2-svp",
            "xc_functional": "wB97X-D (Psi4 v1.5)",
            "hamiltonian_convention": "pyscf (reordered from psi4)",
            "overlap_source": "nablaDFT (Psi4, reordered to pyscf)",
            "initial_hamiltonian": "hcore (T+V) via PySCF int1e_kin + int1e_nuc",
            "orbital_energies": "generalized eigensolve H@C = S@C@diag(e)",
            "positions_unit": "Angstrom",
            "energy_unit": "Hartree",
            "forces_unit": "Hartree/Bohr",
        }
        txn.put(b"__metadata__", json.dumps(metadata).encode())

    if max_workers > 1:
        pool.close()
        pool.join()

    elapsed = time.time() - t0
    out_size = os.path.getsize(str(output_path)) / (1024**3)
    print(f"\n  Done: {n_ok} ok, {n_err} errors in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {out_size:.2f} GiB")
    print(f"  Throughput: {n_ok/elapsed:.1f} mol/s")


def main():
    parser = argparse.ArgumentParser(description="nablaDFT → QHFlow2 preprocessor")
    parser.add_argument("--input", type=str, required=True,
                        help="Split name (e.g., test_2k_conformers, train_100k)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    preprocess_split(args.input, args.workers, args.max_samples)


if __name__ == "__main__":
    main()
