# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

"""QHFlow2-native preparation and reading for the QH9 dataset.

This module uses the published QH9 SQLite schema and split protocol as data
format facts. It does not import or copy the upstream QH9 Python loader. The
QH9 data themselves remain under their upstream dataset license.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import lmdb
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from common.dft_utils import calc_dm0, calc_overlap_and_init_hamiltonian
from common.matrix_transforms import (
    _cut_matrix_3d,
    _cut_matrix_3d_last,
    cut_orbital_coefficients,
    matrix_transform_single,
    pack_upper_triangle,
    unpack_upper_triangle,
)
from common.metric import cal_orbital_and_energies
from common.units import BOHR2ANG
from utils import Onsite_3idx_Overlap_Integral


_DOWNLOAD_URLS = {
    ("QH9Stable", ""): "https://drive.google.com/uc?id=1LcEJGhB8VUGkuyb0oQ_9ANJdSkky9xMS",
    ("QH9Dynamic", "100k"): "https://drive.google.com/uc?id=1SNWk0GD6Nt96qNAJJU2uedwWDQ4bbB1w",
    ("QH9Dynamic", "300k"): "https://drive.google.com/uc?id=1sbf-sFhh3ZmhXgTcN2ke_la39MaG0Yho",
}


@dataclass(frozen=True)
class QH9Layout:
    dataset_name: str
    version: str
    split: str

    @property
    def folder_name(self) -> str:
        return "QH9Stable" if self.dataset_name == "QH9Stable" else f"QH9Dynamic_{self.version}"

    @property
    def raw_filename(self) -> str:
        return f"{self.folder_name}.db"

    @property
    def position_unit(self) -> str:
        return "Angstrom" if self.dataset_name == "QH9Stable" else "Bohr"

    @property
    def split_filename(self) -> str:
        return f"processed_{self.dataset_name}_{self.split}.json"

    def decode_row(self, row: Sequence[Any]) -> tuple[int, int, bytes, bytes, bytes]:
        if self.dataset_name == "QH9Stable":
            record_id, num_nodes, atoms, positions, hamiltonian = row[:5]
        else:
            record_id, num_nodes, atoms, positions, hamiltonian = row[0], row[2], row[3], row[4], row[9]
            if isinstance(record_id, (bytes, bytearray, memoryview)):
                record_id = int(np.frombuffer(record_id, dtype=np.int64)[0])
        return int(record_id), int(num_nodes), atoms, positions, hamiltonian


def _layout(dataset_name: str, version: str, split: str) -> QH9Layout:
    if dataset_name == "QH9Stable":
        if split not in {"random", "size_ood"}:
            raise ValueError("QH9Stable split must be 'random' or 'size_ood'")
        return QH9Layout(dataset_name, "", split)
    if dataset_name == "QH9Dynamic":
        if version not in {"100k", "300k"}:
            raise ValueError("QH9Dynamic version must be '100k' or '300k'")
        if split not in {"geometry", "mol"}:
            raise ValueError("QH9Dynamic split must be 'geometry' or 'mol'")
        return QH9Layout(dataset_name, version, split)
    raise ValueError(f"Unsupported QH9 dataset: {dataset_name}")


def parse_shard_idx(value: str | int | Sequence[int]) -> int | list[int]:
    """Parse ``-1``, comma lists, inclusive ranges, or integer shard IDs."""
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return [int(item) for item in value]
    value = value.strip()
    if value == "-1":
        return -1
    if "," in value:
        return [int(item.strip()) for item in value.split(",")]
    if value.count("-") == 1 and not value.startswith("-"):
        start, stop = (int(item) for item in value.split("-"))
        if stop < start:
            raise ValueError("shard range must be increasing")
        return list(range(start, stop + 1))
    return [int(value)]


def _partition(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(len(values) * 0.8)
    val_end = train_end + int(len(values) * 0.1)
    return values[:train_end], values[train_end:val_end], values[val_end:]


def _stable_split(layout: QH9Layout, connection: sqlite3.Connection) -> dict[str, list[int]]:
    total = int(connection.execute("SELECT COUNT(*) FROM data").fetchone()[0])
    if layout.split == "random":
        train, val, test = _partition(np.random.RandomState(43).permutation(total))
    else:
        columns = connection.execute("PRAGMA table_info(data)").fetchall()
        if len(columns) < 2:
            raise ValueError("QH9 Stable data table must contain a molecule-size column")
        size_column = str(columns[1][1]).replace('"', '""')
        sizes = np.fromiter(
            (int(row[0]) for row in connection.execute(f'SELECT "{size_column}" FROM data ORDER BY rowid')),
            dtype=np.int64,
            count=total,
        )
        train = np.flatnonzero(sizes <= 20)
        val = np.flatnonzero((sizes >= 21) & (sizes <= 22))
        test = np.flatnonzero(sizes >= 23)
    return {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}


def _dynamic_split(layout: QH9Layout, connection: sqlite3.Connection) -> dict[str, list[int]]:
    total = int(connection.execute("SELECT COUNT(*) FROM data").fetchone()[0])
    groups = [np.arange(start, min(start + 100, total), dtype=np.int64) for start in range(0, total, 100)]
    if layout.split == "geometry":
        split_groups = [[], [], []]
        for group in groups:
            shuffled = group[np.random.RandomState(int(group[-1])).permutation(len(group))]
            for target, values in zip(split_groups, _partition(shuffled)):
                target.append(values)
    else:
        molecule_ids = list(range(len(groups)))
        random.Random(43).shuffle(molecule_ids)
        split_groups = [[], [], []]
        for target, ids in zip(split_groups, _partition(np.asarray(molecule_ids))):
            target.extend(groups[int(identifier)] for identifier in ids)

    def combine(parts: list[np.ndarray]) -> list[int]:
        if not parts:
            return []
        values = np.concatenate(parts).tolist()
        return values if layout.split == "geometry" else sorted(values)

    return {"train": combine(split_groups[0]), "val": combine(split_groups[1]), "test": combine(split_groups[2])}


class QH9SQLiteShardWriter:
    """Convert one QH9 SQLite database to deterministic indexed LMDB shards."""

    def __init__(
        self,
        *,
        layout: QH9Layout,
        root_path: str,
        shard_num: int,
        save_path: str,
        max_workers: int = 8,
        use_parallel: bool = False,
        include_dft_energy: bool = False,
        include_dft_forces: bool = False,
        processd_dir_name: str = "processed",
        shard_dir_name: str = "lmdbs",
    ) -> None:
        if shard_num <= 0:
            raise ValueError("shard_num must be positive")
        self.layout = layout
        self.raw_path = Path(root_path)
        self.shard_num = shard_num
        self.max_workers = max_workers
        self.use_parallel = use_parallel
        self.include_dft_energy = include_dft_energy
        self.include_dft_forces = include_dft_forces
        self.processed_dir = Path(save_path) / processd_dir_name
        self.shard_dir = self.processed_dir / shard_dir_name
        self.shard_dir.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        if not self.raw_path.is_file():
            raise FileNotFoundError(f"QH9 SQLite file not found: {self.raw_path}")
        return sqlite3.connect(f"file:{self.raw_path}?mode=ro", uri=True)

    def _bounds(self, total: int) -> list[tuple[int, int]]:
        count = min(self.shard_num, total)
        width = (total + count - 1) // count
        return [(start, min(start + width, total)) for start in range(0, total, width)]

    def _write_metadata(self, connection: sqlite3.Connection, bounds: list[tuple[int, int]]) -> None:
        split = _stable_split(self.layout, connection) if self.layout.dataset_name == "QH9Stable" else _dynamic_split(self.layout, connection)
        with (self.processed_dir / self.layout.split_filename).open("w") as handle:
            json.dump(split, handle)
        index = []
        for shard_id, (start, stop) in enumerate(bounds):
            index.extend((shard_id, global_id, local_id) for local_id, global_id in enumerate(range(start, stop)))
        with (self.processed_dir / "index.json").open("w") as handle:
            json.dump({"index": index}, handle)

    def _encode(self, indexed_row: tuple[int, Sequence[Any]]) -> tuple[bytes, bytes]:
        global_id, row = indexed_row
        record_id, num_nodes, atoms_blob, positions_blob, hamiltonian_blob = self.layout.decode_row(row)
        atoms = np.frombuffer(atoms_blob, dtype=np.int32).copy()
        positions = np.frombuffer(positions_blob, dtype=np.float64).reshape(num_nodes, 3).copy()
        positions_angstrom = positions if self.layout.position_unit == "Angstrom" else positions * BOHR2ANG
        include_dft = self.include_dft_energy or self.include_dft_forces
        dft_result = calc_overlap_and_init_hamiltonian(
            atoms,
            positions_angstrom,
            basis="def2svp",
            xc="b3lyp",
            unit="ang",
            out_mf=include_dft,
        )
        if include_dft:
            overlap, initial, mean_field = dft_result
        else:
            overlap, initial = dft_result
        h_dim = int(overlap.shape[0])
        hamiltonian = np.frombuffer(hamiltonian_blob, dtype=np.float64).reshape(h_dim, h_dim).copy()
        record = {
            "schema_version": 1,
            "id": record_id,
            "num_nodes": num_nodes,
            "atoms": bytes(atoms_blob),
            "pos": bytes(positions_blob),
            "pos_unit": self.layout.position_unit,
            "h_dim": h_dim,
            "packed_hamiltonian": pack_upper_triangle(hamiltonian)[0].tobytes(),
            "packed_overlap": pack_upper_triangle(overlap)[0].tobytes(),
            "packed_initial_hamiltonian": pack_upper_triangle(initial)[0].tobytes(),
        }
        if include_dft:
            energies, coefficients = cal_orbital_and_energies(
                torch.from_numpy(overlap).unsqueeze(0),
                torch.from_numpy(hamiltonian).unsqueeze(0),
            )
            coefficients = coefficients.squeeze(0)
            density_matrix = calc_dm0(torch.as_tensor(atoms), coefficients)
            energies_np = energies.squeeze(0).cpu().numpy()
            coefficients_np = coefficients.cpu().numpy()
            if self.include_dft_energy:
                record["dft_energy"] = float(mean_field.energy_tot(density_matrix))
            if self.include_dft_forces:
                occupations = mean_field.get_occ(energies_np, coefficients_np)
                forces = -mean_field.nuc_grad_method().kernel(
                    mo_energy=energies_np,
                    mo_coeff=coefficients_np,
                    mo_occ=occupations,
                )
                record["dft_forces"] = np.asarray(forces, dtype=np.float64).tobytes()
        return global_id.to_bytes(8, "big"), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)

    def _write_shard(self, shard_id: int, start: int, stop: int) -> None:
        target = self.shard_dir / f"shard_{shard_id:03d}.lmdb"
        if target.is_dir():
            return
        temporary = self.shard_dir / f"shard_{shard_id:03d}.lmdb.in-progress"
        if temporary.exists():
            raise RuntimeError(f"Incomplete shard requires inspection: {temporary}")
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM data LIMIT ? OFFSET ?", (stop - start, start)).fetchall()
        indexed_rows = list(zip(range(start, stop), rows))
        map_size = max(1 << 30, int(self.raw_path.stat().st_size * 6 / max(self.shard_num, 1)))
        environment = lmdb.open(str(temporary), map_size=map_size, subdir=True)
        encoded: Iterable[tuple[bytes, bytes]]
        if self.use_parallel and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                encoded = list(executor.map(self._encode, indexed_rows))
        else:
            encoded = map(self._encode, indexed_rows)
        with environment.begin(write=True) as transaction:
            for key, value in encoded:
                transaction.put(key, value)
        environment.sync()
        environment.close()
        temporary.rename(target)

    def process(self, shard_idx: int | Sequence[int] = -1) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM data").fetchone()[0])
            if total == 0:
                raise ValueError("QH9 database contains no rows")
            bounds = self._bounds(total)
            self._write_metadata(connection, bounds)
        if shard_idx == -1 or shard_idx is None:
            selected = range(len(bounds))
        else:
            selected = [shard_idx] if isinstance(shard_idx, int) else shard_idx
        for shard_id in selected:
            if shard_id < 0 or shard_id >= len(bounds):
                raise IndexError(f"shard {shard_id} outside [0, {len(bounds)})")
            self._write_shard(shard_id, *bounds[shard_id])
        if all((self.shard_dir / f"shard_{i:03d}.lmdb").is_dir() for i in range(len(bounds))):
            (self.processed_dir / "ALL_SHARDS_COMPLETED.txt").write_text(f"All {len(bounds)} QH9 shards are present.\n")


@dataclass
class TensorListWrapper:
    tensors: list[torch.Tensor]


class BaseQH9Dataset(Dataset):
    """Read the QHFlow2 indexed QH9 LMDB format."""

    full_orbitals = 14
    orbital_mask = {
        1: torch.tensor([0, 1, 3, 4, 5]),
        6: torch.arange(14),
        7: torch.arange(14),
        8: torch.arange(14),
        9: torch.arange(14),
    }

    def __init__(
        self,
        *,
        dataset_name: str,
        root: str,
        split: str,
        version: str = "",
        shard_num: int = 60,
        shard_idx: int | Sequence[int] = -1,
        prefix: str = "",
        max_workers_preprocess: int = 8,
        use_parallel_preprocess: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        return_orbital_and_energies: bool = False,
        use_ref_hamiltonian_as_init_ham: bool = False,
        compute_q_tensor: bool = True,
        include_initial_hamiltonian: bool = True,
        include_overlap: bool = True,
        include_dft_energy: bool = False,
        include_dft_forces: bool = False,
        include_cut_orbital_coefficients: bool = True,
        cut_coeff_cache_type: str = "both",
    ) -> None:
        self.layout = _layout(dataset_name, version, split)
        self.dataset_name = dataset_name
        self.version = version
        self.split = split
        self.folder = os.path.join(root, self.layout.folder_name + prefix)
        self.shard_num = shard_num
        self.shard_idx = shard_idx
        self.max_workers_preprocess = max_workers_preprocess
        self.use_parallel_preprocess = use_parallel_preprocess
        self.return_orbital_and_energies = return_orbital_and_energies
        self.use_ref_hamiltonian_as_init_ham = use_ref_hamiltonian_as_init_ham
        self.compute_q_tensor = compute_q_tensor
        self.include_initial_hamiltonian = include_initial_hamiltonian
        self.include_overlap = include_overlap
        self.include_dft_energy = include_dft_energy
        self.include_dft_forces = include_dft_forces
        self.include_cut_orbital_coefficients = include_cut_orbital_coefficients
        if cut_coeff_cache_type not in {"transformed", "original", "both"}:
            raise ValueError("cut_coeff_cache_type must be transformed, original, or both")
        self.cut_coeff_cache_type = cut_coeff_cache_type
        self._environments: dict[int, lmdb.Environment] = {}
        self._reference_hamiltonians: dict[int, torch.Tensor] = {}
        self._q_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self.Q_dict = None
        if compute_q_tensor:
            self.Q_dict = Onsite_3idx_Overlap_Integral(
                atom_list=["H", "C", "N", "O", "F"], basis="def2-svp"
            ).Q_table()
        super().__init__(self.folder, transform, pre_transform, pre_filter)
        self._load_metadata()

    @property
    def raw_file_names(self) -> list[str]:
        return [self.layout.raw_filename]

    @property
    def processed_file_names(self) -> list[str]:
        return [self.layout.split_filename, "index.json", "ALL_SHARDS_COMPLETED.txt"]

    def download(self) -> None:
        import gdown
        result = gdown.download(_DOWNLOAD_URLS[(self.dataset_name, self.version)], output=self.raw_paths[0], quiet=False)
        if not result or not os.path.isfile(self.raw_paths[0]):
            raise FileNotFoundError(f"Download QH9 manually to {self.raw_paths[0]}")

    def process(self) -> None:
        QH9SQLiteShardWriter(
            layout=self.layout,
            root_path=self.raw_paths[0],
            shard_num=self.shard_num,
            save_path=self.folder,
            max_workers=self.max_workers_preprocess,
            use_parallel=self.use_parallel_preprocess,
            include_dft_energy=self.include_dft_energy,
            include_dft_forces=self.include_dft_forces,
        ).process(self.shard_idx)

    def _load_metadata(self) -> None:
        with open(os.path.join(self.processed_dir, self.layout.split_filename)) as handle:
            split = json.load(handle)
        self.train_mask = torch.as_tensor(split["train"], dtype=torch.long)
        self.val_mask = torch.as_tensor(split["val"], dtype=torch.long)
        self.test_mask = torch.as_tensor(split["test"], dtype=torch.long)
        with open(os.path.join(self.processed_dir, "index.json")) as handle:
            rows = json.load(handle)["index"]
        self._index = [(int(shard), int(global_id), int(local_id)) for shard, global_id, local_id in rows]
        self.slices = {"id": torch.arange(len(self._index) + 1)}

    def len(self) -> int:
        return len(self._index)

    def _environment(self, shard_id: int) -> lmdb.Environment:
        if shard_id not in self._environments:
            path = os.path.join(self.processed_dir, "lmdbs", f"shard_{shard_id:03d}.lmdb")
            self._environments[shard_id] = lmdb.open(path, readonly=True, lock=False, readahead=False, max_readers=1024)
        return self._environments[shard_id]

    @staticmethod
    def _array(value: Any, dtype: np.dtype) -> np.ndarray:
        return value.astype(dtype, copy=False) if isinstance(value, np.ndarray) else np.frombuffer(value, dtype=dtype)

    def _decode(self, record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        atoms_np = self._array(record["atoms"], np.int32)
        atoms = torch.as_tensor(atoms_np.copy(), dtype=torch.long)
        positions_np = self._array(record["pos"], np.float64).reshape(-1, 3).copy()
        if str(record.get("pos_unit", self.layout.position_unit)).lower() == "bohr":
            positions_np *= BOHR2ANG
        positions = torch.as_tensor(positions_np, dtype=torch.float64)
        h_dim = int(record.get("h_dim", sum(len(self.orbital_mask[int(z)]) for z in atoms)))
        if "packed_hamiltonian" in record:
            hamiltonian = unpack_upper_triangle(self._array(record["packed_hamiltonian"], np.float64), h_dim)
        else:
            hamiltonian = self._array(record["Ham"], np.float64).reshape(h_dim, h_dim)
        if "packed_overlap" in record:
            overlap = unpack_upper_triangle(self._array(record["packed_overlap"], np.float64), h_dim)
            initial = unpack_upper_triangle(self._array(record["packed_initial_hamiltonian"], np.float64), h_dim)
        elif "ovlp" in record and "init_ham" in record:
            overlap = self._array(record["ovlp"], np.float64).reshape(h_dim, h_dim)
            initial = self._array(record["init_ham"], np.float64).reshape(h_dim, h_dim)
        else:
            overlap, initial = calc_overlap_and_init_hamiltonian(
                atoms_np, positions_np, basis="def2svp", xc="b3lyp", unit="ang"
            )
        return tuple(torch.as_tensor(value.copy(), dtype=torch.float64) if isinstance(value, np.ndarray) else value for value in (atoms, positions, hamiltonian, overlap, initial))

    def _reference_hamiltonian(self, atoms: torch.Tensor) -> torch.Tensor:
        from pyscf import dft, gto

        blocks = []
        for atomic_number in atoms.tolist():
            if atomic_number not in self._reference_hamiltonians:
                molecule = gto.Mole()
                molecule.build(
                    atom=[[atomic_number, (0.0, 0.0, 0.0)]],
                    basis="def2svp",
                    unit="bohr",
                    spin=atomic_number % 2,
                    verbose=0,
                )
                mean_field = dft.RKS(molecule)
                mean_field.xc = "pbe"
                density = mean_field.get_init_guess(key="minao")
                matrix = mean_field.get_fock(dm=density)
                self._reference_hamiltonians[atomic_number] = torch.as_tensor(
                    matrix, dtype=torch.float64
                )
            blocks.append(self._reference_hamiltonians[atomic_number])
        return torch.block_diag(*blocks)

    def _q_tensor(self, atoms: torch.Tensor) -> torch.Tensor:
        key = tuple(int(value) for value in atoms.tolist())
        if key not in self._q_cache:
            matrices = [torch.block_diag(*[self.Q_dict[int(z)][order] for z in atoms]) for order in range(60)]
            result = matrix_transform_single(torch.stack(matrices), atoms, convention="pyscf_def2svp_to_e3nn").permute(1, 2, 0)
            window = result[:, :, 16:40]
            result[:, :, 16:40] = window.reshape(*window.shape[:2], -1, 3)[:, :, :, [1, 2, 0]].reshape(*window.shape)
            self._q_cache[key] = result
        return self._q_cache[key]

    def _build_data(self, record: dict[str, Any], idx: int) -> Data:
        atoms, positions, hamiltonian, overlap, initial = self._decode(record)
        if self.use_ref_hamiltonian_as_init_ham:
            initial = self._reference_hamiltonian(atoms)
        transformed = matrix_transform_single(
            torch.stack((hamiltonian, overlap, initial)), atoms, convention="pyscf_def2svp_to_e3nn"
        )
        diagonal, off_diagonal, diagonal_mask, off_diagonal_mask, edge_index = _cut_matrix_3d(
            transformed, atoms, self.orbital_mask, self.full_orbitals
        )
        values: dict[str, Any] = {
            "pos": positions,
            "atoms": atoms.view(-1, 1),
            "diagonal_hamiltonian": diagonal[:, 0],
            "non_diagonal_hamiltonian": off_diagonal[:, 0],
            "diagonal_hamiltonian_mask": diagonal_mask[:, 0],
            "non_diagonal_hamiltonian_mask": off_diagonal_mask[:, 0],
            "edge_index_full": edge_index,
            "num_nodes": torch.tensor(len(atoms), dtype=torch.long),
            "h_dim": torch.tensor(hamiltonian.shape[0], dtype=torch.long),
            "idx": idx,
        }
        if self.include_overlap:
            values.update(diagonal_overlap=diagonal[:, 1], non_diagonal_overlap=off_diagonal[:, 1])
        if self.include_initial_hamiltonian:
            values.update(diagonal_init_ham=diagonal[:, 2], non_diagonal_init_ham=off_diagonal[:, 2])
        if self.include_dft_energy:
            if "dft_energy" not in record:
                raise KeyError("dft_energy is absent; reprocess this dataset with include_dft_energy=True")
            values["dft_energy"] = torch.tensor(record["dft_energy"], dtype=torch.float64).view(1, 1)
        if self.include_dft_forces:
            if "dft_forces" not in record:
                raise KeyError("dft_forces is absent; reprocess this dataset with include_dft_forces=True")
            values["dft_forces"] = torch.as_tensor(self._array(record["dft_forces"], np.float64).reshape(-1, 3).copy())
        data = Data(**values)
        if self.include_cut_orbital_coefficients and self.cut_coeff_cache_type in {"original", "both"}:
            original_diagonal, original_off_diagonal, _, _, _ = _cut_matrix_3d(
                hamiltonian.unsqueeze(0), atoms, self.orbital_mask, self.full_orbitals
            )
            data.diagonal_hamiltonian_untransformed = original_diagonal[:, 0]
            data.non_diagonal_hamiltonian_untransformed = original_off_diagonal[:, 0]
        if self.compute_q_tensor:
            q_diagonal, q_off, _, _, _ = _cut_matrix_3d_last(
                self._q_tensor(atoms), atoms, self.orbital_mask, self.full_orbitals
            )
            data.diagonal_Q, data.non_diagonal_Q = q_diagonal, q_off
        if self.include_cut_orbital_coefficients or self.return_orbital_and_energies:
            energies, coefficients = cal_orbital_and_energies(
                overlap.unsqueeze(0), hamiltonian.unsqueeze(0)
            )
            if self.return_orbital_and_energies:
                data.orbital_energies = energies.squeeze(0)
                data.orbital_coefficients = coefficients.squeeze(0).cpu().numpy()
                initial_energies, initial_coefficients = cal_orbital_and_energies(
                    overlap.unsqueeze(0), initial.unsqueeze(0)
                )
                data.e0 = initial_energies.detach().cpu().numpy()
                data.c0 = initial_coefficients.detach().cpu().numpy()
                data.ef = energies.detach().cpu().numpy()
                data.cf = coefficients.detach().cpu().numpy()
                data.h = hamiltonian.detach().cpu().numpy()
                data.ovlp = overlap.detach().cpu().numpy()
            if self.include_cut_orbital_coefficients:
                original_blocks = None
                transformed_blocks = None
                if self.cut_coeff_cache_type in {"original", "both"}:
                    original_blocks = cut_orbital_coefficients(
                        coefficients.squeeze(0), atoms, self.orbital_mask, self.full_orbitals
                    )
                if self.cut_coeff_cache_type in {"transformed", "both"}:
                    _, transformed_coefficients = cal_orbital_and_energies(
                        transformed[1].unsqueeze(0), transformed[0].unsqueeze(0)
                    )
                    transformed_blocks = cut_orbital_coefficients(
                        transformed_coefficients.squeeze(0),
                        atoms,
                        self.orbital_mask,
                        self.full_orbitals,
                    )
                selected_blocks = (
                    transformed_blocks
                    if self.cut_coeff_cache_type == "transformed"
                    else original_blocks
                )
                if selected_blocks is not None:
                    data.cut_orbital_coefficients = TensorListWrapper(
                        [block.detach().cpu() for block in selected_blocks]
                    )
                if self.cut_coeff_cache_type == "both" and transformed_blocks is not None:
                    data.cut_orbital_coefficients_transformed = TensorListWrapper(
                        [block.detach().cpu() for block in transformed_blocks]
                    )
        return data

    def _payload(self, idx: int) -> tuple[int, bytes]:
        shard_id, global_id, _ = self._index[idx]
        with self._environment(shard_id).begin() as transaction:
            payload = transaction.get(global_id.to_bytes(8, "big")) or transaction.get(global_id.to_bytes(4, "big"))
        if payload is None:
            raise KeyError(f"QH9 index {global_id} is missing from shard {shard_id}")
        return global_id, payload

    def get(self, idx: int) -> Data:
        _, payload = self._payload(idx)
        return self._build_data(pickle.loads(payload), idx)

    def get_gt_outputs(self, idx: int) -> dict[str, Any]:
        _, payload = self._payload(idx)
        atoms, positions, hamiltonian, overlap, initial = self._decode(pickle.loads(payload))
        return {
            "pos": positions,
            "atoms": atoms.view(-1, 1),
            "hamiltonian": hamiltonian,
            "overlap_matrix": overlap,
            "initial_hamiltonian": initial,
            "num_nodes": torch.tensor(len(atoms), dtype=torch.long),
            "h_dim": torch.tensor(hamiltonian.shape[0], dtype=torch.long),
            "idx": idx,
        }

    def get_key_list(self, idx: int) -> list[int]:
        shard_id, _, _ = self._index[idx]
        with self._environment(shard_id).begin() as transaction:
            return [int.from_bytes(key, "big") for key, _ in transaction.cursor()]

    def close(self) -> None:
        environments = getattr(self, "_environments", {})
        for environment in environments.values():
            environment.close()
        environments.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


class QH9Stable(BaseQH9Dataset):
    def __init__(self, root="datasets/", split="random", **kwargs):
        super().__init__(dataset_name="QH9Stable", root=root, split=split, **kwargs)


class QH9Dynamic(BaseQH9Dataset):
    def __init__(self, root="datasets/", split="geometry", version="300k", **kwargs):
        super().__init__(dataset_name="QH9Dynamic", root=root, split=split, version=version, **kwargs)


class QH9StableWriter(QH9SQLiteShardWriter):
    def __init__(self, root_path: str, shard_num: int, save_path: str, split="random", **kwargs):
        super().__init__(layout=_layout("QH9Stable", "", split), root_path=root_path, shard_num=shard_num, save_path=save_path, **kwargs)


class QH9DynamicWriter(QH9SQLiteShardWriter):
    def __init__(self, root_path: str, shard_num: int, save_path: str, split="geometry", version="300k", **kwargs):
        super().__init__(layout=_layout("QH9Dynamic", version, split), root_path=root_path, shard_num=shard_num, save_path=save_path, **kwargs)
