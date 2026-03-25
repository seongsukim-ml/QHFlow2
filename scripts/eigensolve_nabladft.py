#!/usr/bin/env python
"""2nd pass: add orbital_energies/coefficients via GPU batch eigensolve.

Reads preprocessed nablaDFT LMDB, computes generalized eigenvalue problem
H@C = S@C@diag(e) on GPU in batches, and writes results back.

Usage:
    python eigensolve_nabladft.py --input test_2k_conformers --gpu 7
    python eigensolve_nabladft.py --input train_100k --gpu 7 --batch-size 32
"""

import argparse
import json
import lmdb
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "qhflow2"))
sys.path.insert(0, "/home1/irteam/data-vol1/projects/dft-dataset/src")

from common.matrix_transforms import unpack_upper_triangle
from dft_dataset.solvers import solve_eigenproblem_torch

PROCESSED_DIR = Path("/home1/irteam/data-vol1/datasets/nablaDFT/qhflow2_processed")


def run_eigensolve(split_name, device="cuda", batch_size=64):
    lmdb_path = str(PROCESSED_DIR / f"{split_name}.lmdb")
    # map_size: original file + 100% for eigensolve data (orbital_energies + coefficients are large)
    file_size = os.path.getsize(lmdb_path) if os.path.exists(lmdb_path) else 0
    map_size = max(int(file_size * 3), 20 * (1 << 30))
    env = lmdb.open(lmdb_path, subdir=False, map_size=map_size)

    with env.begin() as txn:
        total = int(txn.get(b"__len__").decode())
        # Check if already done (check LAST sample, not first)
        sample = pickle.loads(txn.get(str(total - 1).encode()))
        if "orbital_energies" in sample:
            print(f"SKIP: {split_name} already has orbital_energies")
            return

    print(f"{'='*60}")
    print(f"  GPU Eigensolve: {split_name}")
    print(f"  Samples: {total}, batch_size: {batch_size}, device: {device}")
    print(f"{'='*60}")

    # Group by h_dim for batched eigensolve
    h_dims = []
    with env.begin() as txn:
        for i in range(total):
            data = pickle.loads(txn.get(str(i).encode()))
            h_dims.append(data["h_dim"])

    groups = defaultdict(list)
    for i, hd in enumerate(h_dims):
        groups[hd].append(i)

    print(f"  Unique h_dims: {len(groups)}, range: {min(h_dims)}-{max(h_dims)}")

    t0 = time.time()
    n_done = 0

    for h_dim, indices in sorted(groups.items()):
        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            B = len(batch_idx)

            H_batch = torch.zeros(B, h_dim, h_dim, dtype=torch.float64, device=device)
            S_batch = torch.zeros(B, h_dim, h_dim, dtype=torch.float64, device=device)

            with env.begin() as txn:
                for bi, idx in enumerate(batch_idx):
                    data = pickle.loads(txn.get(str(idx).encode()))
                    H = unpack_upper_triangle(
                        np.frombuffer(data["packed_hamiltonian"], np.float64).copy(), h_dim)
                    S = unpack_upper_triangle(
                        np.frombuffer(data["packed_overlap"], np.float64).copy(), h_dim)
                    H_batch[bi] = torch.from_numpy(H)
                    S_batch[bi] = torch.from_numpy(S)

            orb_e, orb_c = solve_eigenproblem_torch(H_batch, S_batch)

            with env.begin(write=True) as txn:
                for bi, idx in enumerate(batch_idx):
                    data = pickle.loads(txn.get(str(idx).encode()))
                    data["orbital_energies"] = orb_e[bi].cpu().numpy().tobytes()
                    data["orbital_coefficients"] = orb_c[bi].cpu().numpy().tobytes()
                    txn.put(str(idx).encode(), pickle.dumps(data))

            n_done += B
            if n_done % 200 == 0 or n_done == total:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (total - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{total}] rate={rate:.0f}/s ETA={eta:.0f}s")

    # Update metadata
    with env.begin(write=True) as txn:
        raw = txn.get(b"__metadata__")
        if raw:
            meta = json.loads(raw.decode())
            meta["orbital_energies"] = "generalized eigensolve (GPU, solve_eigenproblem_torch)"
            txn.put(b"__metadata__", json.dumps(meta).encode())

    elapsed = time.time() - t0
    print(f"\n  Done: {total} samples, {elapsed:.1f}s ({elapsed/60:.1f}min), {total/elapsed:.0f} mol/s")


def main():
    parser = argparse.ArgumentParser(description="GPU eigensolve for nablaDFT")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run_eigensolve(args.input, device="cuda", batch_size=args.batch_size)


if __name__ == "__main__":
    main()
