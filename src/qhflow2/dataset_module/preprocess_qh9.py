#!/usr/bin/env python3
"""QH9 dataset preprocessor — raw shard LMDB → single preprocessed LMDB.

원본 sharded LMDB에서 모든 on-the-fly 연산(matrix transform, block cutting,
Q tensor)을 사전 수행하여 단일 LMDB에 저장한다.

결과 LMDB의 각 샘플에는 raw fields + preprocessed blocks 모두 포함.
  - train 모드: preprocessed blocks만 로드 (~2ms)
  - eval 모드: raw matrices도 함께 로드 (full matrix 복원 가능)

Usage:
    cd /home1/irteam/data-vol1/projects/QHFlow2/src
    python -m dataset_module.preprocess_qh9 \\
        --source dataset/QH9Stable_shard \\
        --output dataset/QH9Stable_preprocessed.lmdb \\
        --num-workers 8 \\
        --verify 50
"""

from __future__ import annotations

import argparse
import json
import lmdb
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from dft_dataset.molecule import BasisInfo, unpack_upper_triangle
from dft_dataset.orbital_matrix import OrbitalMatrix
from dft_dataset.conventions import build_reorder_indices, reorder_matrix

from qhflow2.common.custom_logger import setup_global_logger, get_logger

logger = get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────

BASIS = BasisInfo.from_name("def2-svp", convention="pyscf")
PAD_TO = 14  # max_block_size for def2-svp (C/N/O/F = 14 orbitals)

BOHR2ANG = 0.529177210903


# ── Q tensor utilities ───────────────────────────────────────────

def init_q_infrastructure():
    """Q tensor 계산에 필요한 Q_dict와 convention 로드."""
    import shutil

    # PSI4 needs auxiliary.gbs in cwd
    aux_src = os.path.join(os.path.dirname(__file__), "..", "experiment", "auxiliary.gbs")
    aux_dst = os.path.join(os.getcwd(), "auxiliary.gbs")
    if os.path.exists(aux_src) and not os.path.exists(aux_dst):
        shutil.copy(aux_src, aux_dst)

    from qhflow2.utils import Onsite_3idx_Overlap_Integral
    from qhflow2.common.matrix_transforms import get_convention_dict, matrix_transform_single

    Q_dict = Onsite_3idx_Overlap_Integral(
        atom_list=("H", "C", "N", "O", "F"),
        basis="def2-svp",
    ).Q_table()
    convention_dict = get_convention_dict()
    return Q_dict, convention_dict


def build_q_tensor(
    atoms: np.ndarray,
    Q_dict: dict,
    convention_dict: dict,
) -> np.ndarray:
    """Q tensor를 계산하고 블록으로 분해.

    QHFlow2의 BaseQH9Dataset._build_Q_tensor + _cut_matrix_3d_last 로직을 재현.

    Args:
        atoms: (N,) atomic numbers
        Q_dict: Onsite_3idx_Overlap_Integral.Q_table() 결과
        convention_dict: get_convention_dict() 결과

    Returns:
        Q tensor as numpy array (nao, nao, 60) after convention transform
    """
    from qhflow2.common.matrix_transforms import matrix_transform_single

    atoms_torch = torch.tensor(atoms, dtype=torch.int64)
    blocks = []
    for l in range(60):
        block_diag_components = [Q_dict[int(z)][l] for z in atoms]
        blocks.append(torch.block_diag(*block_diag_components))

    Q = torch.stack(blocks)  # (60, nao, nao)
    Q = matrix_transform_single(
        Q, atoms_torch,
        convention_dict["pyscf_def2svp_to_e3nn"],
    ).permute(1, 2, 0)  # (nao, nao, 60)

    Q_slice = Q[:, :, 16:40]
    Q[:, :, 16:40] = (
        Q_slice.reshape(Q.shape[0], Q.shape[1], -1, 3)[:, :, :, [1, 2, 0]].reshape(
            Q.shape[0], Q.shape[1], 24
        )
    )
    return Q.numpy()


def cut_q_tensor(
    Q: np.ndarray,
    atoms: np.ndarray,
    basis_info: BasisInfo,
) -> Tuple[np.ndarray, np.ndarray]:
    """Q tensor (nao, nao, 60) → per-atom blocks.

    OrbitalMatrix.to_blocks()는 (nao, nao) 2D용이므로,
    Q tensor의 마지막 축은 별도 처리.

    Returns:
        (diagonal_Q, off_diagonal_Q) — padded blocks
    """
    slices = basis_info.orbital_slices(atoms)
    n_atoms = len(atoms)
    n_feat = Q.shape[2]

    # Build orbital indices (shell-aligned, same as OrbitalMatrix)
    from conventions import SHELL_TO_L, shell_size
    unique_z = sorted(set(int(z) for z in atoms))
    max_z = max(unique_z, key=lambda z: basis_info.atom_to_nao[z])
    ref_shells = basis_info.atom_to_shells[max_z]

    ref_shell_starts = {}
    offset = 0
    for label in ref_shells:
        ref_shell_starts.setdefault(label, [])
        ref_shell_starts[label].append(offset)
        offset += shell_size(label, basis_info.angular_type)

    orbital_indices = {}
    for z in unique_z:
        atom_shells = basis_info.atom_to_shells[z]
        indices = []
        shell_counters = {label: 0 for label in ref_shell_starts}
        for label in atom_shells:
            n = shell_size(label, basis_info.angular_type)
            counter = shell_counters[label]
            if counter < len(ref_shell_starts[label]):
                start = ref_shell_starts[label][counter]
                indices.extend(range(start, start + n))
                shell_counters[label] = counter + 1
            else:
                start = max(indices) + 1 if indices else 0
                indices.extend(range(start, start + n))
        orbital_indices[z] = np.array(indices, dtype=np.int64)

    diagonal_list = []
    off_diagonal_list = []

    for src_idx in range(n_atoms):
        sl_src = slices[src_idx]
        src_oidx = orbital_indices[int(atoms[src_idx])]
        for dst_idx in range(n_atoms):
            sl_dst = slices[dst_idx]
            dst_oidx = orbital_indices[int(atoms[dst_idx])]

            block_raw = Q[sl_dst, sl_src, :]  # (n_dst, n_src, 60)
            block = np.zeros((PAD_TO, PAD_TO, n_feat), dtype=Q.dtype)
            block[np.ix_(dst_oidx, src_oidx, np.arange(n_feat))] = block_raw

            if src_idx == dst_idx:
                diagonal_list.append(block)
            else:
                off_diagonal_list.append(block)

    return np.stack(diagonal_list), np.stack(off_diagonal_list)


# ── Sample processing ────────────────────────────────────────────

def process_sample(
    raw_dict: Dict[str, Any],
    idx: int,
    Q_dict: dict = None,
    convention_dict: dict = None,
    q_cache: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = None,
    include_q_tensor: bool = False,
) -> Dict[str, Any]:
    """단일 샘플 전처리.

    Returns:
        dict with raw fields + preprocessed blocks
    """
    atoms = np.frombuffer(raw_dict["atoms"], np.int32).copy()
    num_nodes = raw_dict["num_nodes"]

    # Positions
    pos = np.frombuffer(raw_dict["pos"], np.float64).reshape(-1, 3).copy()
    pos_unit = raw_dict.get("pos_unit", "Angstrom")
    if pos_unit.lower() == "bohr":
        pos = pos * BOHR2ANG

    # Unpack matrices — support both shard format and split format
    if "packed_hamiltonian" in raw_dict:
        # Shard format: packed upper triangle
        packed_H = np.frombuffer(raw_dict["packed_hamiltonian"], np.float64)
        packed_S = np.frombuffer(raw_dict["packed_overlap"], np.float64)
        packed_H_init = np.frombuffer(raw_dict["packed_initial_hamiltonian"], np.float64)
        h_dim = raw_dict["h_dim"]
        H_full = unpack_upper_triangle(packed_H, h_dim)
        S_full = unpack_upper_triangle(packed_S, h_dim)
        H_init_full = unpack_upper_triangle(packed_H_init, h_dim)
    else:
        # Split format (keys: Ham, ovlp, init_ham) — full n×n matrices
        H_raw = np.frombuffer(raw_dict["Ham"], np.float64)
        S_raw = np.frombuffer(raw_dict["ovlp"], np.float64)
        H_init_raw = np.frombuffer(raw_dict["init_ham"], np.float64)
        h_dim = int(round(len(H_raw) ** 0.5))
        H_full = H_raw.reshape(h_dim, h_dim).copy()
        S_full = S_raw.reshape(h_dim, h_dim).copy()
        H_init_full = H_init_raw.reshape(h_dim, h_dim).copy()

    # Convention: pyscf → e3nn via OrbitalMatrix
    basis_pyscf = BasisInfo.from_name("def2-svp", convention="pyscf")
    basis_e3nn = BasisInfo.from_name("def2-svp", convention="e3nn")

    H_om = OrbitalMatrix(data=H_full, atomic_numbers=atoms, basis_info=basis_pyscf)
    S_om = OrbitalMatrix(data=S_full, atomic_numbers=atoms, basis_info=basis_pyscf)
    Hi_om = OrbitalMatrix(data=H_init_full, atomic_numbers=atoms, basis_info=basis_pyscf)

    H_e3nn = H_om.in_convention("e3nn")
    S_e3nn = S_om.in_convention("e3nn")
    Hi_e3nn = Hi_om.in_convention("e3nn")

    # Block decomposition
    H_blocks = H_e3nn.to_blocks(pad_to=PAD_TO, align="shell")
    S_blocks = S_e3nn.to_blocks(pad_to=PAD_TO, align="shell")
    Hi_blocks = Hi_e3nn.to_blocks(pad_to=PAD_TO, align="shell")

    # Q tensor (cached by atom types) — optional, very large (~2.3 MB/sample)
    diag_Q, offdiag_Q = None, None
    if include_q_tensor and Q_dict is not None and q_cache is not None:
        atom_key = tuple(int(z) for z in atoms)
        if atom_key in q_cache:
            diag_Q, offdiag_Q = q_cache[atom_key]
        else:
            Q = build_q_tensor(atoms, Q_dict, convention_dict)
            diag_Q, offdiag_Q = cut_q_tensor(Q, atoms, basis_e3nn)
            q_cache[atom_key] = (diag_Q, offdiag_Q)

    # Energy/forces (optional)
    dft_energy = raw_dict.get("dft_energy", None)
    dft_forces_raw = raw_dict.get("dft_forces", None)
    dft_forces = (
        np.frombuffer(dft_forces_raw, np.float64).reshape(-1, 3).copy()
        if dft_forces_raw is not None else None
    )

    # Pack upper triangle for eval storage
    iu = np.triu_indices(h_dim)
    packed_H = H_full[iu]
    packed_S = S_full[iu]
    packed_H_init = H_init_full[iu]

    # Build output dict
    result = {
        # ── Raw (eval용) ──
        "atoms": atoms,
        "pos": pos,
        "num_nodes": num_nodes,
        "h_dim": h_dim,
        "hamiltonian_packed": packed_H,
        "overlap_packed": packed_S,
        "init_ham_packed": packed_H_init,

        # ── Preprocessed (train용, e3nn convention) ──
        "diagonal_hamiltonian": H_blocks.diagonal,
        "non_diagonal_hamiltonian": H_blocks.off_diagonal,
        "diagonal_overlap": S_blocks.diagonal,
        "non_diagonal_overlap": S_blocks.off_diagonal,
        "diagonal_init_ham": Hi_blocks.diagonal,
        "non_diagonal_init_ham": Hi_blocks.off_diagonal,
        "diagonal_hamiltonian_mask": H_blocks.diagonal_mask,
        "non_diagonal_hamiltonian_mask": H_blocks.off_diagonal_mask,
        "edge_index_full": H_blocks.edge_index,
    }

    if diag_Q is not None:
        result["diagonal_Q"] = diag_Q
        result["non_diagonal_Q"] = offdiag_Q

    if dft_energy is not None:
        result["dft_energy"] = float(dft_energy)
    if dft_forces is not None:
        result["dft_forces"] = dft_forces

    return result


# ── LMDB I/O ─────────────────────────────────────────────────────

def read_source_shards(source_dir: str) -> Tuple[List[str], dict, dict]:
    """원본 shard LMDB들과 인덱스/split 정보를 로드.

    Returns:
        (lmdb_paths, index_info, split_info)
    """
    processed_dir = os.path.join(source_dir, "processed")
    lmdbs_dir = os.path.join(processed_dir, "lmdbs")

    # Find shards
    shard_paths = sorted(
        str(p) for p in Path(lmdbs_dir).glob("shard_*.lmdb")
    )
    if not shard_paths:
        raise FileNotFoundError(f"No shard LMDB files found in {lmdbs_dir}")

    # Load index
    index_path = os.path.join(processed_dir, "index.json")
    with open(index_path) as f:
        index_info = json.load(f)

    # Load splits
    split_info = {}
    for split_file in Path(processed_dir).glob("processed_QH9Stable_*.json"):
        with open(split_file) as f:
            data = json.load(f)
        split_name = split_file.stem  # e.g., "processed_QH9Stable_random_12"
        split_info[split_name] = data

    return shard_paths, index_info, split_info


def read_sample_from_shards(
    idx: int,
    shard_envs: Dict[int, lmdb.Environment],
    shard_paths: List[str],
    index_list: List,
) -> Dict:
    """인덱스로 원본 LMDB에서 샘플 읽기."""
    shard_idx, cur_idx, shard_data_idx = index_list[idx]

    if shard_idx not in shard_envs:
        shard_envs[shard_idx] = lmdb.open(
            shard_paths[shard_idx],
            readonly=True, lock=False, readahead=False,
        )

    env = shard_envs[shard_idx]
    key = int(idx).to_bytes(length=4, byteorder="big")
    with env.begin() as txn:
        raw = txn.get(key)
        if raw is None:
            raise KeyError(f"idx={idx}, shard={shard_idx}, key={key.hex()} not found")
    return pickle.loads(raw)


def write_preprocessed_lmdb(
    output_path: str,
    samples: List[Dict],
    split_info: dict,
    map_size: int = 1 << 38,  # 256 GB
) -> int:
    """전처리된 샘플들을 단일 LMDB로 저장."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(output_path, map_size=map_size, readonly=False, lock=True)

    count = 0
    batch_size = 200
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        with env.begin(write=True) as txn:
            for i, sample in enumerate(batch):
                idx = start + i
                key = idx.to_bytes(4, "big")
                txn.put(key, pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
                count += 1

    # Metadata
    with env.begin(write=True) as txn:
        txn.put(b"__len__", count.to_bytes(4, "big"))
        txn.put(b"__format__", b"pickle")
        if split_info:
            txn.put(b"__splits__", pickle.dumps(split_info))

    env.close()
    return count


# ── Verification ─────────────────────────────────────────────────

def verify_samples(
    source_dir: str,
    output_path: str,
    n_verify: int = 50,
):
    """전처리 결과를 원본 BaseQH9Dataset과 비교 검증."""
    from qhflow2.dataset_module.qh9_datasets_shard import QH9Stable
    from qhflow2.common.matrix_transforms import _cut_matrix_3d

    logger.info(f"Verifying {n_verify} samples against original dataset...")

    # Load original dataset (shard-based)
    dataset = QH9Stable(
        root=os.path.join(source_dir, ".."),
        prefix="_shard",
        split="random",
        compute_q_tensor=True,
        include_initial_hamiltonian=True,
        include_overlap=True,
        include_dft_energy=True,
        include_dft_forces=True,
        include_cut_orbital_coefficients=False,
    )

    # Load preprocessed LMDB
    env = lmdb.open(output_path, readonly=True, lock=False, readahead=False)
    length = int.from_bytes(
        env.begin().get(b"__len__"), "big"
    )

    rng = np.random.RandomState(42)
    indices = rng.choice(min(length, len(dataset)), size=min(n_verify, length), replace=False)

    n_pass = 0
    for idx in indices:
        idx = int(idx)
        # Original
        orig = dataset.get(idx)

        # Preprocessed
        with env.begin() as txn:
            raw = txn.get(idx.to_bytes(4, "big"))
        pre = pickle.loads(raw)

        # Compare diagonal hamiltonian
        orig_dh = orig.diagonal_hamiltonian.numpy()
        pre_dh = pre["diagonal_hamiltonian"]
        if not np.allclose(orig_dh, pre_dh, atol=1e-10):
            logger.error(f"[{idx}] diagonal_hamiltonian mismatch: max_diff={np.abs(orig_dh - pre_dh).max():.2e}")
            continue

        # Compare non-diagonal hamiltonian
        orig_ndh = orig.non_diagonal_hamiltonian.numpy()
        pre_ndh = pre["non_diagonal_hamiltonian"]
        if not np.allclose(orig_ndh, pre_ndh, atol=1e-10):
            logger.error(f"[{idx}] non_diagonal_hamiltonian mismatch: max_diff={np.abs(orig_ndh - pre_ndh).max():.2e}")
            continue

        # Compare edge_index
        orig_ei = orig.edge_index_full.numpy()
        pre_ei = pre["edge_index_full"]
        if not np.array_equal(orig_ei, pre_ei):
            logger.error(f"[{idx}] edge_index_full mismatch")
            continue

        # Compare Q tensor
        if hasattr(orig, "diagonal_Q") and orig.diagonal_Q is not None:
            orig_dq = orig.diagonal_Q.numpy()
            pre_dq = pre["diagonal_Q"]
            if not np.allclose(orig_dq, pre_dq, atol=1e-10):
                logger.error(f"[{idx}] diagonal_Q mismatch: max_diff={np.abs(orig_dq - pre_dq).max():.2e}")
                continue

        n_pass += 1

    env.close()
    dataset._close_db_envs()

    logger.info(f"Verification: {n_pass}/{len(indices)} samples passed")
    if n_pass < len(indices):
        logger.warning("Some samples failed verification!")
    return n_pass == len(indices)


# ── Main ─────────────────────────────────────────────────────────

def read_split_lmdb(lmdb_path: str) -> Tuple[lmdb.Environment, int, dict]:
    """단일 split LMDB를 열고 총 샘플 수와 split 정보를 반환.

    Split LMDB 포맷: key = idx (4 bytes big-endian), value = pickled dict
    with keys: id, num_nodes, atoms, pos, Ham, ovlp, init_ham.

    Returns:
        (env, total_samples, split_info)
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, max_readers=4)
    total = env.stat()["entries"]

    # Load split info from adjacent JSON files
    split_info = {}
    lmdb_dir = os.path.dirname(lmdb_path)
    for fname in os.listdir(lmdb_dir):
        if fname.startswith("processed_QH9Stable") and fname.endswith(".json"):
            with open(os.path.join(lmdb_dir, fname)) as f:
                split_info[fname.replace(".json", "")] = json.load(f)
        elif fname.startswith("processed_QH9Stable") and fname.endswith(".pt"):
            import torch as _torch
            data = _torch.load(os.path.join(lmdb_dir, fname), weights_only=False)
            # .pt format: (train_mask, val_mask, test_mask)
            if isinstance(data, (list, tuple)) and len(data) == 3:
                split_name = fname.replace(".pt", "")
                split_info[split_name] = {
                    "train": data[0].tolist(),
                    "val": data[1].tolist(),
                    "test": data[2].tolist(),
                }

    return env, total, split_info


def main():
    parser = argparse.ArgumentParser(description="Preprocess QH9 dataset")
    parser.add_argument("--source", default=None,
                        help="Source shard dataset dir (e.g., dataset/QH9Stable_shard)")
    parser.add_argument("--source-lmdb", default=None,
                        help="Source split LMDB path (e.g., dataset/QH9Stable/processed/QH9Stable.lmdb)")
    parser.add_argument("--output", default=None,
                        help="Output LMDB path (default: auto-detected)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers (for future use)")
    parser.add_argument("--verify", type=int, default=0,
                        help="Number of samples to verify (0 = skip)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to process (for testing)")
    parser.add_argument("--include-q-tensor", action="store_true",
                        help="Include Q tensor in output (adds ~2.3 MB/sample)")
    args = parser.parse_args()

    setup_global_logger()

    if not args.source and not args.source_lmdb:
        parser.error("Either --source (shard dir) or --source-lmdb (split LMDB) is required")

    use_split_lmdb = args.source_lmdb is not None

    if use_split_lmdb:
        source_lmdb_path = args.source_lmdb
        output_path = args.output or os.path.join(
            os.path.dirname(os.path.dirname(source_lmdb_path)),
            "QH9Stable_preprocessed.lmdb",
        )
        logger.info(f"Source (split LMDB): {source_lmdb_path}")
        logger.info(f"Output: {output_path}")

        src_env, total, split_info = read_split_lmdb(source_lmdb_path)
        if args.max_samples:
            total = min(total, args.max_samples)
        logger.info(f"Total samples: {total}")
    else:
        source_dir = args.source
        output_path = args.output or source_dir.rstrip("/") + "_preprocessed.lmdb"
        logger.info(f"Source (shard): {source_dir}")
        logger.info(f"Output: {output_path}")

        shard_paths, index_info, split_info = read_source_shards(source_dir)
        index_list = index_info["index"]
        total = len(index_list)
        if args.max_samples:
            total = min(total, args.max_samples)
        logger.info(f"Total samples: {total}, Shards: {len(shard_paths)}")

    # Init Q tensor infrastructure (only if needed)
    include_q = args.include_q_tensor
    Q_dict, convention_dict, q_cache = None, None, None
    if include_q:
        logger.info("Initializing Q tensor infrastructure...")
        Q_dict, convention_dict = init_q_infrastructure()
        q_cache = {}
    else:
        logger.info("Skipping Q tensor (use --include-q-tensor to include)")

    # Process + write in streaming fashion (no full dataset in memory)
    logger.info("Processing and writing samples (streaming)...")
    shard_envs: Dict[int, lmdb.Environment] = {}
    t_start = time.time()

    # Open output LMDB directly at target path
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    map_size = 1 << 42 if include_q else 1 << 36  # 4 TB with Q, 64 GB without
    out_env = lmdb.open(output_path, map_size=map_size, readonly=False, lock=True)

    count = 0
    write_batch = []
    write_batch_size = 100

    for idx in range(total):
        if use_split_lmdb:
            key = idx.to_bytes(4, "big")
            with src_env.begin() as txn:
                raw = txn.get(key)
                if raw is None:
                    logger.warning(f"Sample {idx} not found, skipping")
                    continue
            raw_dict = pickle.loads(raw)
        else:
            raw_dict = read_sample_from_shards(idx, shard_envs, shard_paths, index_list)

        result = process_sample(raw_dict, idx, Q_dict, convention_dict, q_cache, include_q_tensor=include_q)
        write_batch.append((idx, result))

        if len(write_batch) >= write_batch_size:
            with out_env.begin(write=True) as txn:
                for widx, sample in write_batch:
                    key = widx.to_bytes(4, "big")
                    txn.put(key, pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
            count += len(write_batch)
            write_batch.clear()

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (total - idx - 1) / rate
            logger.info(
                f"  [{idx + 1}/{total}] {rate:.1f} samples/s, "
                f"Q cache: {len(q_cache)} keys, ETA: {eta / 60:.1f} min"
            )

    # Flush remaining
    if write_batch:
        with out_env.begin(write=True) as txn:
            for widx, sample in write_batch:
                key = widx.to_bytes(4, "big")
                txn.put(key, pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))
        count += len(write_batch)

    # Write metadata
    with out_env.begin(write=True) as txn:
        txn.put(b"__len__", count.to_bytes(4, "big"))
        txn.put(b"__format__", b"pickle")
        if split_info:
            txn.put(b"__splits__", pickle.dumps(split_info))

    out_env.close()

    # Close source envs
    if use_split_lmdb:
        src_env.close()
    else:
        for env in shard_envs.values():
            env.close()

    elapsed = time.time() - t_start
    logger.info(f"Processing done: {count} samples in {elapsed:.1f}s ({count / elapsed:.1f} samples/s)")
    if q_cache is not None:
        logger.info(f"Q tensor cache: {len(q_cache)} unique atom keys")

    logger.info(f"Output: {output_path}")

    # Verify
    if args.verify > 0 and args.source:
        verify_samples(args.source, output_path, n_verify=args.verify)

    logger.info("Done!")


if __name__ == "__main__":
    main()
