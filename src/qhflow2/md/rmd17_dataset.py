#!/usr/bin/env python3
"""
Baseline RMD17 Dataset loader for energy/force prediction.
Simple reader (no extended-precision fallbacks). Keeps get_label_stats for trainers.
"""

from __future__ import annotations

import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import Dataset
from torch_geometric.data import Data


def _to_tensor(data, dtype=torch.float32) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    return torch.from_numpy(np.asarray(data)).to(dtype)


def _safe_read_array(dset, dtype=None):
    """Read an HDF5 dataset defensively, tolerating broken dtypes."""
    if dtype is None:
        dtype = np.float64
    np_dtype = np.dtype(dtype)
    read_order = [
        lambda: np.asarray(dset[()], dtype=np_dtype),
        lambda: np.asarray(dset.astype(np_dtype)[()], dtype=np_dtype),
        lambda: np.asarray(dset[:], dtype=np_dtype),
        lambda: np.array(list(dset), dtype=np_dtype),
    ]
    def _read_direct():
        out = np.empty(dset.shape, dtype=np_dtype)
        try:
            dset.read_direct(out)
            return out
        except Exception:
            dset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, out)
            return out
    read_order.append(_read_direct)

    for reader in read_order:
        try:
            return reader()
        except Exception:
            continue
    return np.zeros(dset.shape, dtype=np_dtype)


def _safe_read_attr(attrs, key, default=0.0):
    """Read an attribute robustly; fall back to default on dtype issues."""
    try:
        val = attrs[key]
        return float(val)
    except Exception:
        try:
            arr = np.array(attrs[key], copy=False)
            return float(arr.astype(np.float64).item())
        except Exception:
            try:
                return float(str(attrs[key]))
            except Exception:
                return float(default)

class RMD17_DFT(Dataset):
    """
    RMD17 Dataset for energy/force prediction.
    Returns PyG Data format compatible with existing trainers.

    config keys:
      src, name, split, split_seed, num_train, num_val
    """

    def __init__(self, config=None, transform=None) -> None:
        super().__init__()
        self.config = config
        self.transform = transform

        root = config.get("src")
        name = config.get("name")
        self.split = config.get("split", "all")
        self.split_seed = config.get("split_seed", 42)

        if not name.startswith("rmd-"):
            name = "rmd-" + name
        self.name = name

        self.folder = os.path.join(root, self.name)
        self.h5_file = os.path.join(self.folder, "data.h5")

        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file}")

        with h5py.File(self.h5_file, "r") as f:
            self.metadata = {}
            if "metadata" in f:
                try:
                    self.metadata = dict(f["metadata"].attrs.items())
                except ValueError:
                    # Some HDF5 files store attrs with unsupported precision; ignore them.
                    for k in f["metadata"].attrs.keys():
                        try:
                            v = f["metadata"].attrs[k]
                            self.metadata[k] = v.tolist() if hasattr(v, "tolist") else v
                        except Exception:
                            self.metadata[k] = None
            total_samples = sum(1 for k in f.keys() if k.startswith("sample_"))

        num_train = config.get("num_train", int(0.8 * total_samples))
        num_val = config.get("num_val", int(0.1 * total_samples))

        generator = torch.Generator().manual_seed(self.split_seed)
        all_indices = torch.randperm(total_samples, generator=generator).tolist()

        train_indices = all_indices[:num_train]
        val_indices = all_indices[num_train:num_train + num_val]
        test_indices = all_indices[num_train + num_val:]

        if self.split == "train":
            self.indices = train_indices
        elif self.split == "val":
            self.indices = val_indices
        elif self.split == "test":
            self.indices = test_indices
        else:
            self.indices = all_indices

        self._cache = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        if idx in self._cache:
            return self._cache[idx]

        file_idx = self.indices[idx]

        with h5py.File(self.h5_file, "r") as f:
            sample = f[f"sample_{file_idx}"]

            try:
                pos_np = sample["positions"][()]
            except Exception:
                pos_np = _safe_read_array(sample["positions"], dtype=np.float64)
            pos = _to_tensor(pos_np)

            try:
                an_np = sample["atomic_numbers"][()]
                an_np = np.asarray(an_np).astype(np.int64, copy=False)
            except Exception:
                try:
                    an_np = _safe_read_array(sample["atomic_numbers"], dtype=np.int64)
                except Exception:
                    fallback_len = pos_np.shape[0]
                    meta_anz = self.metadata.get("atomic_numbers") if hasattr(self, "metadata") else None
                    if meta_anz is not None:
                        an_np = np.asarray(meta_anz, dtype=np.int64)
                    else:
                        an_np = np.zeros(fallback_len, dtype=np.int64)
            atomic_numbers = _to_tensor(an_np, dtype=torch.long)

            try:
                forces_np = sample["forces_ev_ang"][()]
            except Exception:
                forces_np = _safe_read_array(sample["forces_ev_ang"], dtype=np.float64)
            forces = _to_tensor(forces_np)

            energy_val = _safe_read_attr(sample.attrs, "energy_ev", default=0.0)
            energy = torch.tensor([energy_val], dtype=torch.float32)

            n_atoms = len(atomic_numbers)
            large_cell = torch.eye(3, dtype=torch.float32).unsqueeze(0) * 1000.0
            data = Data(
                pos=pos,
                atomic_numbers=atomic_numbers,
                natoms=torch.tensor([n_atoms], dtype=torch.long),
                y=energy,
                force=forces,
                cell=large_cell,
                pbc=torch.zeros(3, dtype=torch.bool),
                cell_offsets=torch.zeros(0, 3),
                fixed=torch.zeros(n_atoms, dtype=torch.long),
                tags=torch.zeros(n_atoms, dtype=torch.long),
                neighbors=torch.tensor([0], dtype=torch.long),
                charge=torch.tensor([0.0], dtype=torch.float32),
                spin=torch.tensor([0.0], dtype=torch.float32),
                dataset="rmd17",
            )

        if self.transform is not None:
            data = self.transform(data)

        self._cache[idx] = data
        return data

    def get_label_stats(self):
        if hasattr(self, "_label_stats"):
            return self._label_stats

        energy_sum = 0.0
        energy_sq_sum = 0.0
        force_sum = 0.0
        force_sq_sum = 0.0
        n_energy = 0
        n_force = 0

        with h5py.File(self.h5_file, "r") as f:
            for idx in self.indices:
                sample = f[f"sample_{idx}"]
                energy = float(sample.attrs["energy_ev"])
                forces = sample["forces_ev_ang"][:]

                energy_sum += energy
                energy_sq_sum += energy * energy
                n_energy += 1

                force_sum += forces.sum()
                force_sq_sum += np.square(forces).sum()
                n_force += forces.size

        e_mean = energy_sum / n_energy
        e_var = energy_sq_sum / n_energy - e_mean ** 2
        e_std = float(np.sqrt(max(e_var, 0.0)))

        if n_force > 0:
            f_mean = force_sum / n_force
            f_var = force_sq_sum / n_force - f_mean ** 2
            f_std = float(np.sqrt(max(f_var, 0.0)))
        else:
            f_mean, f_std = 0.0, 0.0

        self._label_stats = {
            "e_mean": float(e_mean),
            "e_std": e_std,
            "f_mean": float(f_mean),
            "f_std": f_std,
        }
        return self._label_stats

    @property
    def num_samples(self):
        return len(self.indices)

    def close_db(self):
        pass

