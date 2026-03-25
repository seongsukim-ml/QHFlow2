"""QH9 Density Matrix Dataset — dft-dataset LMDB를 QHFlow2 학습에 사용.

dft-dataset의 packed LMDB 포맷을 읽어, 기존 BaseQH9Dataset과 동일한 출력
(diagonal/non_diagonal blocks, Q tensor, masks 등)을 반환하면서
density_matrix, initial_density_matrix를 추가 필드로 제공한다.

Usage:
    from qhflow2.dataset_module.qh9_density_dataset import QH9DensityDataset

    dataset = QH9DensityDataset(
        lmdb_path="/home1/irteam/data-vol1/datasets/qh9_b3lyp5_with_density/data.lmdb",
        split_path="/path/to/processed_QH9Stable_random_12.json",
        split="train",
    )
    data = dataset[0]
    # data.diagonal_hamiltonian          — [n_atoms, 14, 14]
    # data.diagonal_density_matrix       — [n_atoms, 14, 14]
    # data.non_diagonal_density_matrix   — [n_edges, 14, 14]
"""

import json
import os
import pickle
from typing import Any, Dict, List, Optional

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from qhflow2.common.matrix_transforms import (
    _cut_matrix_3d,
    _cut_matrix_3d_last,
    get_convention_dict,
    matrix_transform_single,
    unpack_upper_triangle,
)
from qhflow2.models.utils import FUNCTIONAL_REGISTRY, BASIS_REGISTRY, get_orbital_mask
from qhflow2.utils import Onsite_3idx_Overlap_Integral


class QH9DensityDataset(Dataset):
    """dft-dataset LMDB에서 density matrix 포함 Data 객체를 반환하는 dataset.

    기존 BaseQH9Dataset과 동일한 출력 포맷 (diagonal/non_diagonal blocks)을
    사용하면서 density_matrix, initial_density_matrix를 추가 필드로 제공.
    기존 QHFlow2 코드를 수정하지 않고 독립적으로 동작한다.
    """

    full_orbitals: int = 14

    def __init__(
        self,
        lmdb_path: str,
        split_path: Optional[str] = None,
        split: str = "train",
        *,
        include_density_matrix: bool = True,
        include_initial_density_matrix: bool = True,
        include_initial_hamiltonian: bool = True,
        include_overlap: bool = True,
        include_dft_energy: bool = True,
        include_dft_forces: bool = True,
        compute_q_tensor: bool = True,
        functional: str = "b3lyp",
        basis_name: str = "def2-svp",
    ) -> None:
        super().__init__()
        self.lmdb_path = lmdb_path
        self.split = split

        self.include_density_matrix = include_density_matrix
        self.include_initial_density_matrix = include_initial_density_matrix
        self.include_initial_hamiltonian = include_initial_hamiltonian
        self.include_overlap = include_overlap
        self.include_dft_energy = include_dft_energy
        self.include_dft_forces = include_dft_forces
        self.compute_q_tensor = compute_q_tensor

        self._functional_id = FUNCTIONAL_REGISTRY.get(functional, 0)
        self._basis_id = BASIS_REGISTRY.get(basis_name, 0)

        self.orbital_mask = get_orbital_mask(basis_name)
        self.convention_dict = get_convention_dict()
        self._q_tensor_cache: Dict[tuple, torch.Tensor] = {}

        if self.compute_q_tensor:
            atom_list = ("H", "C", "N", "O", "F")
            self.Q_dict = Onsite_3idx_Overlap_Integral(
                atom_list=atom_list,
                basis=basis_name,
            ).Q_table()
        else:
            self.Q_dict = None

        # Open LMDB (read-only)
        self._db_env: Optional[lmdb.Environment] = None
        self._open_lmdb()

        # Detect format and count entries
        with self._db_env.begin() as txn:
            total_keys = txn.stat()["entries"]
            has_schema = txn.get(b"__schema__") is not None

            # Detect split_key format: if first entry has split keys like
            # b'\x00\x00\x00\x00:atomic_numbers', it's split_key format
            first_key = (0).to_bytes(4, "big")
            test_split_key = first_key + b":atomic_numbers"
            self._split_key_format = txn.get(test_split_key) is not None

            if self._split_key_format:
                # Count by probing sequential 4-byte keys
                # Each entry has 1 meta key + N split keys, plus optional __schema__
                # Faster: read meta dict, count split fields, compute
                meta_raw = txn.get(first_key)
                meta = pickle.loads(meta_raw)
                # Count split fields for entry 0
                n_split_fields = 0
                cursor = txn.cursor()
                prefix = first_key + b":"
                cursor.set_range(prefix)
                for k, _ in cursor:
                    if k.startswith(prefix):
                        n_split_fields += 1
                    else:
                        break
                keys_per_entry = 1 + n_split_fields  # meta + split fields
                schema_offset = 1 if has_schema else 0
                self._total_len = (total_keys - schema_offset) // keys_per_entry
                self._n_split_fields = n_split_fields
            else:
                self._total_len = total_keys - (1 if has_schema else 0)

        # Split fields to read (for split_key format)
        self._split_field_names = [
            "atomic_numbers", "positions", "forces",
            "hamiltonian_packed", "overlap_packed",
            "initial_hamiltonian_packed", "density_matrix_packed",
            "initial_density_matrix_packed",
        ]

        # Load split
        if split_path is not None and os.path.exists(split_path):
            with open(split_path, "r") as f:
                split_info = json.load(f)
            self.indices = split_info[split]
        else:
            self.indices = list(range(self._total_len))

    # Shared LMDB environments per path (avoids "already open" error)
    _shared_envs: Dict[str, lmdb.Environment] = {}

    def _open_lmdb(self) -> None:
        if self.lmdb_path in QH9DensityDataset._shared_envs:
            self._db_env = QH9DensityDataset._shared_envs[self.lmdb_path]
            return
        self._db_env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            max_readers=1024,
            readahead=False,
        )
        QH9DensityDataset._shared_envs[self.lmdb_path] = self._db_env

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        raw_idx = self.indices[idx]
        data_dict = self._read_lmdb(raw_idx)
        return self._build_data_object(data_dict, idx=idx)

    # ------------------------------------------------------------------
    # LMDB I/O
    # ------------------------------------------------------------------
    def _read_lmdb(self, idx: int) -> dict:
        if self._db_env is None:
            self._open_lmdb()
        try:
            return self._read_lmdb_inner(idx)
        except lmdb.Error:
            self._open_lmdb()
            return self._read_lmdb_inner(idx)

    def _read_lmdb_inner(self, idx: int) -> dict:
        with self._db_env.begin() as txn:
            key = int(idx).to_bytes(4, byteorder="big")
            raw = txn.get(key)
            if raw is None:
                raise KeyError(f"Index {idx} not found in LMDB")
            data_dict = pickle.loads(raw)

            if self._split_key_format:
                # Merge split key fields into data_dict
                for field in self._split_field_names:
                    split_key = key + b":" + field.encode()
                    val = txn.get(split_key)
                    if val is not None:
                        data_dict[field] = val  # raw bytes

            return data_dict

    # ------------------------------------------------------------------
    # Matrix unpacking (dft-dataset packed format)
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack_matrix(data_dict: dict, name: str) -> Optional[torch.Tensor]:
        """'{name}_packed' (bytes) + '{name}_nao' (int) → (nao, nao) float64 tensor.

        Works with both inline dict format and split_key format where
        packed bytes are stored under data_dict["{name}_packed"].
        """
        packed_key = f"{name}_packed"
        nao_key = f"{name}_nao"
        if packed_key not in data_dict:
            return None
        raw = data_dict[packed_key]
        dtype_str = data_dict.get(f"{name}_dtype", "float64")
        packed = np.frombuffer(raw, dtype=np.dtype(dtype_str)).copy()
        nao = data_dict[nao_key]
        mat = unpack_upper_triangle(packed, nao)
        return torch.from_numpy(mat).to(torch.float64)

    # ------------------------------------------------------------------
    # Build Data object
    # ------------------------------------------------------------------
    def _build_data_object(self, data_dict: dict, idx: int = None) -> Data:
        # --- Atoms & positions ---
        num_atoms = data_dict["num_atoms"]
        atoms_raw = data_dict["atomic_numbers"]  # bytes (from split key or inline)
        atomic_numbers = np.frombuffer(atoms_raw, np.int32).copy()
        atoms = torch.tensor(atomic_numbers, dtype=torch.int64)

        pos_raw = data_dict["positions"]  # bytes
        positions = np.frombuffer(pos_raw, np.float64).copy()
        pos = torch.from_numpy(positions.reshape(num_atoms, 3)).to(torch.float64)

        # --- Unpack matrices ---
        hamiltonian = self._unpack_matrix(data_dict, "hamiltonian")
        overlap = self._unpack_matrix(data_dict, "overlap")
        init_ham = self._unpack_matrix(data_dict, "initial_hamiltonian")
        density_matrix = self._unpack_matrix(data_dict, "density_matrix")
        init_density_matrix = self._unpack_matrix(data_dict, "initial_density_matrix")

        h_dim = hamiltonian.shape[0]

        # --- Stack matrices for batch convention transform + block cut ---
        # Always include H, S, init_H (indices 0, 1, 2)
        matrix_list = [hamiltonian, overlap]
        # Index mapping: 0=H, 1=S, then dynamically append
        idx_init_ham = None
        idx_dm = None
        idx_init_dm = None

        if self.include_initial_hamiltonian and init_ham is not None:
            idx_init_ham = len(matrix_list)
            matrix_list.append(init_ham)

        if self.include_density_matrix and density_matrix is not None:
            idx_dm = len(matrix_list)
            matrix_list.append(density_matrix)

        if self.include_initial_density_matrix and init_density_matrix is not None:
            idx_init_dm = len(matrix_list)
            matrix_list.append(init_density_matrix)

        concat_feat = torch.stack(matrix_list, dim=0)  # [N_mat, nao, nao]

        # --- Convention transform (PySCF → E3NN) ---
        convention = "pyscf_def2svp_to_e3nn"
        transformed = matrix_transform_single(
            concat_feat, atoms, self.convention_dict[convention]
        )

        # --- Block decomposition ---
        (
            concat_diag_blocks,
            concat_nondiag_blocks,
            concat_diag_masks,
            concat_nondiag_masks,
            edge_index_full,
        ) = _cut_matrix_3d(
            transformed,
            atoms,
            self.orbital_mask,
            self.full_orbitals,
        )

        # --- Extract blocks by index ---
        diagonal_hamiltonian = concat_diag_blocks[:, 0]
        non_diagonal_hamiltonian = concat_nondiag_blocks[:, 0]
        diagonal_hamiltonian_mask = concat_diag_masks[:, 0]
        non_diagonal_hamiltonian_mask = concat_nondiag_masks[:, 0]

        diagonal_overlap = concat_diag_blocks[:, 1]
        non_diagonal_overlap = concat_nondiag_blocks[:, 1]

        # --- Q tensor ---
        diagonal_Q = None
        non_diagonal_Q = None
        if self.compute_q_tensor:
            atom_key = tuple(int(z.item()) for z in atoms)
            if atom_key in self._q_tensor_cache:
                cached_q = self._q_tensor_cache[atom_key]
            else:
                cached_q = self._build_Q_tensor(atoms).detach().cpu()
                self._q_tensor_cache[atom_key] = cached_q

            Q = cached_q.to(device=transformed.device, dtype=transformed.dtype)
            (
                diagonal_Q,
                non_diagonal_Q,
                _,
                _,
                _,
            ) = _cut_matrix_3d_last(
                Q, atoms, self.orbital_mask, self.full_orbitals
            )

        # --- Build Data ---
        data_kwargs: Dict[str, Any] = {
            "pos": pos,
            "atoms": atoms.view(-1, 1),
            "diagonal_hamiltonian": diagonal_hamiltonian,
            "non_diagonal_hamiltonian": non_diagonal_hamiltonian,
            "diagonal_hamiltonian_mask": diagonal_hamiltonian_mask,
            "non_diagonal_hamiltonian_mask": non_diagonal_hamiltonian_mask,
            "edge_index_full": edge_index_full,
            "num_nodes": torch.tensor(num_atoms, dtype=torch.int64),
            "h_dim": torch.tensor(h_dim, dtype=torch.int64),
            "functional_id": torch.tensor(self._functional_id, dtype=torch.long),
            "basis_id": torch.tensor(self._basis_id, dtype=torch.long),
        }

        if idx is not None:
            data_kwargs["idx"] = idx

        # Overlap
        if self.include_overlap:
            data_kwargs["diagonal_overlap"] = diagonal_overlap
            data_kwargs["non_diagonal_overlap"] = non_diagonal_overlap

        # Initial Hamiltonian
        if idx_init_ham is not None:
            data_kwargs["diagonal_init_ham"] = concat_diag_blocks[:, idx_init_ham]
            data_kwargs["non_diagonal_init_ham"] = concat_nondiag_blocks[:, idx_init_ham]

        # Density matrix (same basis → same mask as hamiltonian)
        if idx_dm is not None:
            data_kwargs["diagonal_density_matrix"] = concat_diag_blocks[:, idx_dm]
            data_kwargs["non_diagonal_density_matrix"] = concat_nondiag_blocks[:, idx_dm]
            data_kwargs["diagonal_density_matrix_mask"] = diagonal_hamiltonian_mask
            data_kwargs["non_diagonal_density_matrix_mask"] = non_diagonal_hamiltonian_mask

        # Initial density matrix
        if idx_init_dm is not None:
            data_kwargs["diagonal_init_density_matrix"] = concat_diag_blocks[:, idx_init_dm]
            data_kwargs["non_diagonal_init_density_matrix"] = concat_nondiag_blocks[:, idx_init_dm]

        # Energy & forces
        if self.include_dft_energy and "energy" in data_dict:
            data_kwargs["dft_energy"] = torch.tensor(
                data_dict["energy"], dtype=torch.float64
            ).view(1, 1)

        if self.include_dft_forces and "forces" in data_dict and data_dict["forces"] is not None:
            forces_shape = data_dict.get("forces_shape", (num_atoms, 3))
            forces_dtype = np.dtype(data_dict.get("forces_dtype", "float64"))
            forces_raw = data_dict["forces"]  # bytes
            forces = np.frombuffer(forces_raw, dtype=forces_dtype).copy()
            data_kwargs["dft_forces"] = torch.from_numpy(
                forces.reshape(forces_shape)
            ).to(torch.float64)

        data = Data(**data_kwargs)

        if diagonal_Q is not None:
            data.diagonal_Q = diagonal_Q
            data.non_diagonal_Q = non_diagonal_Q

        return data

    # ------------------------------------------------------------------
    # Q tensor (same logic as BaseQH9Dataset)
    # ------------------------------------------------------------------
    def _build_Q_tensor(self, atoms: torch.Tensor) -> torch.Tensor:
        blocks: List[torch.Tensor] = []
        for l in range(60):
            block_diag_components = [self.Q_dict[int(z.item())][l] for z in atoms]
            blocks.append(torch.block_diag(*block_diag_components))

        Q = torch.stack(blocks)
        Q = matrix_transform_single(
            Q, atoms, self.convention_dict["pyscf_def2svp_to_e3nn"]
        ).permute(1, 2, 0)

        Q_slice = Q[:, :, 16:40]
        Q[:, :, 16:40] = (
            Q_slice.reshape(Q.shape[0], Q.shape[1], -1, 3)[:, :, :, [1, 2, 0]]
            .reshape(Q.shape[0], Q.shape[1], 24)
        )
        return Q

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _close(self) -> None:
        # Don't close shared LMDB environments — they are reused across instances
        self._db_env = None

    def __del__(self):
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()
