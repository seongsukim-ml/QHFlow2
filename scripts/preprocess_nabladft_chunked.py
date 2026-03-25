#!/usr/bin/env python
"""Chunked parallel nablaDFT preprocessing.

Splits the source LMDB into N chunks, processes each in a separate process,
then merges the results into a single output LMDB.

Usage:
    # Process all chunks in parallel (16 chunks on different CPUs)
    python preprocess_nabladft_chunked.py --input train_100k --n-chunks 16

    # Process a single chunk (for manual parallelism across terminals)
    python preprocess_nabladft_chunked.py --input train_100k --n-chunks 16 --chunk-id 0

    # Merge completed chunks
    python preprocess_nabladft_chunked.py --input train_100k --n-chunks 16 --merge-only
"""

import argparse
import json
import lmdb
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src" / "qhflow2"))
sys.path.insert(0, "/home1/irteam/data-vol1/projects/dft-dataset/src")

NABLA_LMDB_DIR = Path("/home1/irteam/data-vol1/datasets/nablaDFT/lmdb")
OUTPUT_DIR = Path("/home1/irteam/data-vol1/datasets/nablaDFT/qhflow2_processed")


def process_chunk(split_name, chunk_id, n_chunks, gpu_id=None):
    """Process a single chunk of the dataset.

    Args:
        gpu_id: GPU index for convention transform acceleration. None = CPU only.
    """
    from dft_dataset.lmdb_dataset import LMDBDataset
    from common.dft_utils import init_pyscf_mol
    from common.matrix_transforms import pack_upper_triangle, matrix_transform_single
    from common.orbital_conventions import get_all_conventions

    device = f'cuda:{gpu_id}' if gpu_id is not None else 'cpu'

    input_path = str(NABLA_LMDB_DIR / f"{split_name}.lmdb")
    chunk_path = str(OUTPUT_DIR / f"{split_name}_chunk_{chunk_id:03d}.lmdb")

    if os.path.exists(chunk_path):
        # Check if already complete
        try:
            env = lmdb.open(chunk_path, subdir=False, readonly=True)
            with env.begin() as txn:
                n = int(txn.get(b"__len__").decode())
            env.close()
            expected = (total + n_chunks - 1) // n_chunks if 'total' in dir() else n + 1
            if n >= expected - 1:  # chunk is complete (within 1 of expected)
                print(f"  Chunk {chunk_id}: SKIP (complete, {n} samples)")
                return
            elif n > 0:
                print(f"  Chunk {chunk_id}: PARTIAL ({n} samples), restarting...")
                env.close()
                os.remove(chunk_path)
        except Exception:
            os.remove(chunk_path)

    src = LMDBDataset(input_path)
    total = len(src)

    # Calculate chunk range
    chunk_size = (total + n_chunks - 1) // n_chunks
    start_idx = chunk_id * chunk_size
    end_idx = min(start_idx + chunk_size, total)
    n_samples = end_idx - start_idx

    if n_samples <= 0:
        return

    print(f"  Chunk {chunk_id}/{n_chunks}: samples [{start_idx}, {end_idx}) = {n_samples}")

    conv = get_all_conventions()["psi4_def2svp_to_pyscf"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    map_size = max(n_samples * 2_000_000, 1 << 30)
    env = lmdb.open(chunk_path, map_size=map_size, subdir=False)

    t0 = time.time()
    n_ok, n_err = 0, 0

    for idx in range(start_idx, end_idx):
        try:
            mol = src[idx]
            atoms = np.array(mol.atomic_numbers, dtype=np.int32)
            pos = np.array(mol.positions, dtype=np.float64)
            n_atoms = len(atoms)

            H_psi4 = np.array(mol.hamiltonian, dtype=np.float64)
            S_psi4 = np.array(mol.overlap, dtype=np.float64)
            h_dim = H_psi4.shape[0]

            # Convention: Psi4 → PySCF (GPU-accelerated if available)
            at = torch.from_numpy(atoms).long().to(device)
            H_pyscf = matrix_transform_single(
                torch.from_numpy(H_psi4).double().to(device), at, convention=conv).cpu().numpy()
            S_pyscf = matrix_transform_single(
                torch.from_numpy(S_psi4).double().to(device), at, convention=conv).cpu().numpy()

            # hcore (T+V)
            pyscf_mol = init_pyscf_mol(atoms, pos.reshape(-1, 3), unit="ang", basis="def2svp")
            init_ham = (pyscf_mol.intor("int1e_kin") + pyscf_mol.intor("int1e_nuc")).astype("float64")

            # Pack
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

            with env.begin(write=True) as txn:
                txn.put(str(n_ok).encode(), pickle.dumps(result))
            n_ok += 1

        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print(f"    [ERROR] idx={idx}: {str(e)[:100]}")

        done = n_ok + n_err
        # Update __len__ every 100 samples (crash-safe)
        if done % 100 == 0:
            with env.begin(write=True) as txn:
                txn.put(b"__len__", str(n_ok).encode())
        if done % 500 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_samples - done) / rate if rate > 0 else 0
            print(f"    Chunk {chunk_id}: [{done}/{n_samples}] rate={rate:.1f}/s ETA={eta/60:.0f}min",
                  flush=True)

    # Final metadata write
    with env.begin(write=True) as txn:
        txn.put(b"__len__", str(n_ok).encode())
    env.close()

    elapsed = time.time() - t0
    print(f"    Chunk {chunk_id}: done {n_ok} ok, {n_err} err in {elapsed/60:.1f}min ({n_ok/elapsed:.1f}/s)")


def merge_chunks(split_name, n_chunks):
    """Merge all chunk LMDBs into a single output LMDB."""
    output_path = str(OUTPUT_DIR / f"{split_name}.lmdb")
    if os.path.exists(output_path):
        print(f"Output already exists: {output_path}")
        return

    print(f"\nMerging {n_chunks} chunks → {output_path}")

    # Count total samples across chunks
    total = 0
    chunk_info = []
    for i in range(n_chunks):
        chunk_path = str(OUTPUT_DIR / f"{split_name}_chunk_{i:03d}.lmdb")
        if not os.path.exists(chunk_path):
            print(f"  WARNING: chunk {i} missing")
            chunk_info.append((chunk_path, 0))
            continue
        env = lmdb.open(chunk_path, subdir=False, readonly=True)
        with env.begin() as txn:
            n = int(txn.get(b"__len__").decode())
        env.close()
        chunk_info.append((chunk_path, n))
        total += n
        print(f"  Chunk {i}: {n} samples")

    print(f"  Total: {total} samples")

    # Sum actual chunk file sizes for accurate map_size estimate
    total_bytes = sum(os.path.getsize(p) for p, n in chunk_info if n > 0 and os.path.exists(p))
    map_size = max(int(total_bytes * 1.2), total * 2_000_000, 1 << 30)
    out_env = lmdb.open(output_path, map_size=map_size, subdir=False)

    global_idx = 0
    t0 = time.time()

    for chunk_path, n in chunk_info:
        if n == 0:
            continue
        src_env = lmdb.open(chunk_path, subdir=False, readonly=True)
        with src_env.begin() as src_txn:
            with out_env.begin(write=True) as out_txn:
                for i in range(n):
                    data = src_txn.get(str(i).encode())
                    out_txn.put(str(global_idx).encode(), data)
                    global_idx += 1
        src_env.close()

    with out_env.begin(write=True) as txn:
        txn.put(b"__len__", str(global_idx).encode())
        metadata = {
            "source": "nablaDFT",
            "split": split_name,
            "n_samples": global_idx,
            "basis": "def2-svp",
            "xc_functional": "wB97X-D (Psi4 v1.5)",
            "hamiltonian_convention": "pyscf (reordered from psi4)",
            "overlap_source": "nablaDFT (Psi4, reordered to pyscf)",
            "initial_hamiltonian": "hcore (T+V) via PySCF int1e_kin + int1e_nuc",
            "positions_unit": "Angstrom",
            "energy_unit": "Hartree",
            "forces_unit": "Hartree/Bohr",
            "merged_from": f"{n_chunks} chunks",
        }
        txn.put(b"__metadata__", json.dumps(metadata).encode())

    out_env.close()
    elapsed = time.time() - t0
    out_size = os.path.getsize(output_path) / (1024**3)
    print(f"\n  Merged: {global_idx} samples, {out_size:.2f} GiB, {elapsed:.0f}s")


def launch_all_chunks(split_name, n_chunks):
    """Launch all chunks as separate background processes."""
    python = sys.executable
    script = str(Path(__file__).resolve())
    pids = []

    for i in range(n_chunks):
        log = f"/tmp/nabla_chunk_{i:03d}.log"
        cmd = f"nohup {python} -u {script} --input {split_name} --n-chunks {n_chunks} --chunk-id {i} > {log} 2>&1 &"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        # Get PID by checking last background process
        pid_cmd = subprocess.run(
            f"{python} -u {script} --input {split_name} --n-chunks {n_chunks} --chunk-id {i} &\necho $!",
            shell=True, capture_output=True, text=True, executable="/bin/bash"
        )
        print(f"  Chunk {i}: log={log}")

    # Simpler: just use subprocess.Popen
    procs = []
    for i in range(n_chunks):
        log = f"/tmp/nabla_chunk_{i:03d}.log"
        with open(log, "w") as f:
            p = subprocess.Popen(
                [python, "-u", script, "--input", split_name,
                 "--n-chunks", str(n_chunks), "--chunk-id", str(i)],
                stdout=f, stderr=subprocess.STDOUT
            )
            procs.append((i, p))
            print(f"  Chunk {i}: PID={p.pid}, log={log}")

    print(f"\n  Launched {n_chunks} processes. Monitor with:")
    print(f"    tail -f /tmp/nabla_chunk_*.log")
    print(f"  When all done, merge with:")
    print(f"    python {script} --input {split_name} --n-chunks {n_chunks} --merge-only")

    # Wait for all
    print(f"\n  Waiting for all chunks to complete...")
    for i, p in procs:
        p.wait()
        print(f"    Chunk {i}: exit code {p.returncode}")

    print(f"\n  All chunks complete. Merging...")
    merge_chunks(split_name, n_chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--n-chunks", type=int, default=16)
    parser.add_argument("--chunk-id", type=int, default=None,
                        help="Process single chunk (for manual parallelism)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index for convention transform acceleration")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    if args.merge_only:
        merge_chunks(args.input, args.n_chunks)
    elif args.chunk_id is not None:
        process_chunk(args.input, args.chunk_id, args.n_chunks, gpu_id=args.gpu)
    else:
        launch_all_chunks(args.input, args.n_chunks)


if __name__ == "__main__":
    main()
