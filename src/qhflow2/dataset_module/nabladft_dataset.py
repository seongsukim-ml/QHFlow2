"""nablaDFT dataset for QHFlow2.

Loads preprocessed nablaDFT LMDB (from scripts/preprocess_nabladft.py)
and converts to torch_geometric Data objects compatible with QHFlow2 training.

Convention chain: Psi4 → PySCF (done in preprocessing) → e3nn (done here at load time)
"""

import lmdb
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from qhflow2.common.matrix_transforms import (
    _cut_matrix_3d,
    _get_orbital_mask,
    matrix_transform_single,
    unpack_upper_triangle,
)
from qhflow2.common.orbital_conventions import get_all_conventions


class NablaDFTDataset(Dataset):
    """nablaDFT dataset for QHFlow2 Hamiltonian prediction.

    Reads preprocessed LMDB (QHFlow2 format with packed matrices in PySCF convention).
    Applies PySCF→e3nn convention transform at load time and cuts into atomic blocks.

    Args:
        lmdb_path: Path to preprocessed LMDB file
        split: 'train', 'val', or 'test' (for splitting the dataset)
        split_ratio: (train, val, test) ratios (default: 0.8, 0.1, 0.1)
        seed: Random seed for split
        full_orbitals: Maximum orbital block size (32 for nablaDFT with Br)
        basis: Basis identifier for orbital masks ('def2-svp-nabla')
        convention: Convention name for PySCF→e3nn transform
        include_overlap: Include overlap matrix blocks
        include_initial_hamiltonian: Include initial Hamiltonian blocks
        include_dft_energy: Include DFT energy
        include_dft_forces: Include DFT forces
    """

    def __init__(
        self,
        lmdb_path: str,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
        seed: int = 42,
        full_orbitals: int = 32,
        basis: str = "def2-svp-nabla",
        convention: str = "pyscf_def2svp_nabla_to_e3nn",
        include_overlap: bool = True,
        include_initial_hamiltonian: bool = True,
        include_dft_energy: bool = False,
        include_dft_forces: bool = False,
        transform=None,
        pre_transform=None,
    ):
        self.lmdb_path = lmdb_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.full_orbitals = full_orbitals
        self.basis = basis
        self.include_overlap = include_overlap
        self.include_initial_hamiltonian = include_initial_hamiltonian
        self.include_dft_energy = include_dft_energy
        self.include_dft_forces = include_dft_forces

        # Convention for PySCF → e3nn transform
        all_convs = get_all_conventions()
        self.convention = all_convs[convention]

        # Orbital masks
        self.orbital_mask = _get_orbital_mask(basis)

        # Read total length (open+close immediately, don't keep handle for fork safety)
        self._lmdb_path = lmdb_path
        self._env = None  # lazy open per-process
        env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            self._total_len = int(txn.get(b"__len__").decode())
        env.close()

        # Create split indices
        rng = np.random.RandomState(seed)
        perm = rng.permutation(self._total_len)
        n_train = int(self._total_len * split_ratio[0])
        n_val = int(self._total_len * split_ratio[1])

        if split == "train":
            self._split_indices = perm[:n_train]
        elif split == "val":
            self._split_indices = perm[n_train:n_train + n_val]
        elif split == "test":
            self._split_indices = perm[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {split}")

        super().__init__(transform=transform, pre_transform=pre_transform)

    def len(self):
        return len(self._split_indices)

    def _get_env(self):
        """Lazy-open LMDB per process (fork-safe for DataLoader workers)."""
        if self._env is None:
            self._env = lmdb.open(
                self._lmdb_path, subdir=False, readonly=True, lock=False,
                readahead=False, max_readers=256,
            )
        return self._env

    def get(self, idx):
        """Load and process a single sample."""
        real_idx = self._split_indices[idx]

        with self._get_env().begin() as txn:
            raw = txn.get(str(real_idx).encode())
            if raw is None:
                raise IndexError(f"Sample {real_idx} not found in LMDB")
            data_dict = pickle.loads(raw)

        return self._build_data_object(data_dict)

    def _build_data_object(self, data_dict):
        """Convert preprocessed dict to torch_geometric Data."""
        # Unpack basic info
        n_atoms = data_dict["num_nodes"]
        h_dim = data_dict["h_dim"]
        atoms = np.frombuffer(data_dict["atoms"], dtype=np.int32).copy()
        pos = np.frombuffer(data_dict["pos"], dtype=np.float64).reshape(n_atoms, 3).copy()

        # Unpack matrices (PySCF convention)
        hamiltonian = unpack_upper_triangle(
            np.frombuffer(data_dict["packed_hamiltonian"], dtype=np.float64).copy(), h_dim
        )
        overlap = unpack_upper_triangle(
            np.frombuffer(data_dict["packed_overlap"], dtype=np.float64).copy(), h_dim
        )
        init_ham = unpack_upper_triangle(
            np.frombuffer(data_dict["packed_initial_hamiltonian"], dtype=np.float64).copy(), h_dim
        )

        # Convert to tensors
        atoms_tensor = torch.from_numpy(atoms).long()
        pos_tensor = torch.from_numpy(pos).float()

        # Stack matrices for convention transform: [H, S, init_H]
        matrices = [
            torch.from_numpy(hamiltonian).to(torch.float64),
            torch.from_numpy(overlap).to(torch.float64),
            torch.from_numpy(init_ham).to(torch.float64),
        ]
        stacked = torch.stack(matrices, dim=0)  # [3, h_dim, h_dim]

        # Convention transform: PySCF → e3nn (per-matrix)
        transformed = torch.stack([
            matrix_transform_single(m, atoms_tensor, convention=self.convention)
            for m in stacked
        ], dim=0)

        # Cut into atomic blocks
        (
            diagonal_blocks,
            non_diagonal_blocks,
            diagonal_masks,
            non_diagonal_masks,
            edge_index_full,
        ) = _cut_matrix_3d(
            transformed,
            atoms_tensor,
            self.orbital_mask,
            self.full_orbitals,
        )

        # Build Data object
        data = Data(
            pos=pos_tensor,
            atoms=atoms_tensor.unsqueeze(-1),
            # Hamiltonian blocks (index 0)
            diagonal_hamiltonian=diagonal_blocks[:, 0].float(),
            non_diagonal_hamiltonian=non_diagonal_blocks[:, 0].float(),
            diagonal_hamiltonian_mask=diagonal_masks[:, 0].float(),
            non_diagonal_hamiltonian_mask=non_diagonal_masks[:, 0].float(),
            edge_index_full=edge_index_full,
        )

        if self.include_overlap:
            data.diagonal_overlap = diagonal_blocks[:, 1].float()
            data.non_diagonal_overlap = non_diagonal_blocks[:, 1].float()

        if self.include_initial_hamiltonian:
            data.diagonal_init_ham = diagonal_blocks[:, 2].float()
            data.non_diagonal_init_ham = non_diagonal_blocks[:, 2].float()

        if self.include_dft_energy:
            data.dft_energy = torch.tensor(data_dict["dft_energy"], dtype=torch.float64)

        if self.include_dft_forces:
            forces = np.frombuffer(data_dict["dft_forces"], dtype=np.float64).reshape(n_atoms, 3)
            data.dft_forces = torch.from_numpy(forces.copy()).float()

        return data
