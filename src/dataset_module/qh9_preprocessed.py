"""Preprocessed QH9 dataset — single LMDB, minimal __getitem__.

preprocess_qh9.py로 생성된 단일 LMDB를 읽는 Dataset 클래스.
__getitem__에서 연산 없이 deserialize + numpy→torch 변환만 수행.

Train 모드: preprocessed blocks만 로드 (~2ms)
Eval 모드: raw matrices도 포함하여 full matrix 복원 가능

Usage:
    dataset = QH9PreprocessedDataset(
        "dataset/QH9Stable_shard_preprocessed.lmdb",
        split="random",
        mode="train",
    )
    data = dataset[0]  # → torch_geometric.data.Data
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List, Optional, Sequence

import lmdb
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from common.custom_logger import get_logger

logger = get_logger(__name__)


# Fields to load in each mode
_TRAIN_FIELDS = {
    "atoms", "pos", "num_nodes", "h_dim",
    "diagonal_hamiltonian", "non_diagonal_hamiltonian",
    "diagonal_overlap", "non_diagonal_overlap",
    "diagonal_init_ham", "non_diagonal_init_ham",
    "diagonal_hamiltonian_mask", "non_diagonal_hamiltonian_mask",
    "edge_index_full",
    "diagonal_Q", "non_diagonal_Q",
    "dft_energy", "dft_forces",
}

_EVAL_EXTRA_FIELDS = {
    "hamiltonian_packed", "overlap_packed", "init_ham_packed",
}


class QH9PreprocessedDataset(Dataset):
    """Preprocessed QH9 dataset from single LMDB.

    Args:
        lmdb_path: Path to preprocessed LMDB file.
        split: "random" or "size_ood".
        split_path: Path to split JSON (auto-detected if None).
        mode: "train" (fast, blocks only) or "eval" (includes raw matrices).
    """

    def __init__(
        self,
        lmdb_path: str,
        split: str = "random",
        split_path: Optional[str] = None,
        mode: str = "train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.lmdb_path = str(lmdb_path)
        self.split = split
        self.mode = mode
        self._env: Optional[lmdb.Environment] = None

        # Determine which fields to load
        self._fields = set(_TRAIN_FIELDS)
        if mode == "eval":
            self._fields |= _EVAL_EXTRA_FIELDS

        # Read length
        env = self._get_env()
        with env.begin() as txn:
            len_bytes = txn.get(b"__len__")
            self._total_len = int.from_bytes(len_bytes, "big")

            # Try to read embedded splits
            splits_bytes = txn.get(b"__splits__")
            self._embedded_splits = pickle.loads(splits_bytes) if splits_bytes else {}
        self._close_env()

        # Load split info
        self._load_split(split_path)

        logger.info(
            f"QH9PreprocessedDataset: {self._total_len} samples, "
            f"split={split} (train={len(self.train_mask)}, "
            f"val={len(self.val_mask)}, test={len(self.test_mask)}), "
            f"mode={mode}"
        )

        super().__init__(None, transform, pre_transform, pre_filter)

    def _get_env(self) -> lmdb.Environment:
        """Lazy LMDB env — worker fork 시 각 프로세스가 독립 핸들 생성."""
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=256,
            )
        return self._env

    def _close_env(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def _load_split(self, split_path: Optional[str]):
        """Split 정보 로드.

        우선순위:
        1. 명시적 split_path
        2. LMDB 내 embedded splits
        3. 같은 디렉토리의 processed_QH9Stable_*.json
        """
        split_data = None

        # 1. Explicit path
        if split_path and os.path.exists(split_path):
            with open(split_path) as f:
                split_data = json.load(f)

        # 2. Embedded splits
        if split_data is None:
            for key, data in self._embedded_splits.items():
                if self.split in key:
                    split_data = data
                    break

        # 3. Adjacent JSON file
        if split_data is None:
            lmdb_dir = os.path.dirname(self.lmdb_path)
            # Check parent dirs for split files
            for search_dir in [lmdb_dir, os.path.join(lmdb_dir, "..")]:
                for fname in os.listdir(search_dir) if os.path.isdir(search_dir) else []:
                    if fname.startswith("processed_QH9Stable") and self.split in fname and fname.endswith(".json"):
                        with open(os.path.join(search_dir, fname)) as f:
                            split_data = json.load(f)
                        break
                if split_data:
                    break

        if split_data is None:
            raise FileNotFoundError(
                f"Cannot find split info for '{self.split}'. "
                f"Provide split_path or ensure split JSON is embedded in LMDB."
            )

        self.train_mask = torch.tensor(split_data["train"], dtype=torch.int64)
        self.val_mask = torch.tensor(split_data["val"], dtype=torch.int64)
        self.test_mask = torch.tensor(split_data["test"], dtype=torch.int64)

    def len(self) -> int:
        return self._total_len

    def get(self, idx: int) -> Data:
        """Load a single sample from LMDB."""
        env = self._get_env()
        key = int(idx).to_bytes(4, "big")

        with env.begin() as txn:
            raw = txn.get(key)
            if raw is None:
                raise KeyError(f"Sample {idx} not found in LMDB")

        sample = pickle.loads(raw)
        return self._build_data(sample, idx)

    def _build_data(self, sample: Dict, idx: int) -> Data:
        """Deserialize sample dict → torch_geometric Data.

        No computation — just numpy→torch conversion.
        """
        atoms = torch.from_numpy(sample["atoms"]).to(torch.int64).view(-1, 1)
        pos = torch.from_numpy(sample["pos"]).to(torch.float64)
        num_nodes = torch.tensor(sample["num_nodes"], dtype=torch.int64)
        h_dim = torch.tensor(sample["h_dim"], dtype=torch.int64)

        data = Data(
            pos=pos,
            atoms=atoms,
            num_nodes=num_nodes,
            h_dim=h_dim,
            idx=idx,
            # Hamiltonian blocks
            diagonal_hamiltonian=torch.from_numpy(sample["diagonal_hamiltonian"]).to(torch.float64),
            non_diagonal_hamiltonian=torch.from_numpy(sample["non_diagonal_hamiltonian"]).to(torch.float64),
            diagonal_hamiltonian_mask=torch.from_numpy(sample["diagonal_hamiltonian_mask"]).to(torch.float64),
            non_diagonal_hamiltonian_mask=torch.from_numpy(sample["non_diagonal_hamiltonian_mask"]).to(torch.float64),
            edge_index_full=torch.from_numpy(sample["edge_index_full"]).to(torch.int64),
            # Overlap blocks
            diagonal_overlap=torch.from_numpy(sample["diagonal_overlap"]).to(torch.float64),
            non_diagonal_overlap=torch.from_numpy(sample["non_diagonal_overlap"]).to(torch.float64),
            # Init Hamiltonian blocks
            diagonal_init_ham=torch.from_numpy(sample["diagonal_init_ham"]).to(torch.float64),
            non_diagonal_init_ham=torch.from_numpy(sample["non_diagonal_init_ham"]).to(torch.float64),
        )

        # Q tensor
        if "diagonal_Q" in sample:
            data.diagonal_Q = torch.from_numpy(sample["diagonal_Q"]).to(torch.float64)
            data.non_diagonal_Q = torch.from_numpy(sample["non_diagonal_Q"]).to(torch.float64)

        # Energy / forces
        if "dft_energy" in sample:
            data.dft_energy = torch.tensor(sample["dft_energy"], dtype=torch.float64).view(1, 1)
        if "dft_forces" in sample and sample["dft_forces"] is not None:
            data.dft_forces = torch.from_numpy(sample["dft_forces"]).to(torch.float64)

        # Eval mode: raw packed matrices for reconstruction
        if self.mode == "eval":
            if "hamiltonian_packed" in sample:
                data.hamiltonian_packed = torch.from_numpy(sample["hamiltonian_packed"]).to(torch.float64)
            if "overlap_packed" in sample:
                data.overlap_packed = torch.from_numpy(sample["overlap_packed"]).to(torch.float64)
            if "init_ham_packed" in sample:
                data.init_ham_packed = torch.from_numpy(sample["init_ham_packed"]).to(torch.float64)

        return data

    def __del__(self):
        self._close_env()

    def __repr__(self) -> str:
        return (
            f"QH9PreprocessedDataset(path='{os.path.basename(self.lmdb_path)}', "
            f"n={self._total_len}, split='{self.split}', mode='{self.mode}')"
        )
