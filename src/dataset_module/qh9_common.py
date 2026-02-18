import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import lmdb
import numpy as np
import torch
from scipy.linalg import block_diag
from torch_geometric.data import Data, InMemoryDataset

from common.dft_utils import calc_dm0, calc_overlap_and_init_hamiltonian
from common.matrix_transforms import (
    _cut_matrix_3d,
    _cut_matrix_3d_last,
    cut_orbital_coefficients,
    get_convention_dict,
    matrix_transform_single,
    pack_upper_triangle,
    unpack_upper_triangle,
    compute_chc_from_blocks,
)
from common.metric import cal_orbital_and_energies
from common.units import BOHR2ANG
from dataset_module.lmdb_shard import LMDBShard_maker_db
from utils import Onsite_3idx_Overlap_Integral


@dataclass
class TensorListWrapper:
    tensors: List[torch.Tensor]


@dataclass
class DFTArtifacts:
    """Container for reusable DFT related quantities."""

    h_dim: int
    dft_energy: float
    dft_forces: np.ndarray
    packed_hamiltonian: np.ndarray
    packed_overlap: np.ndarray
    packed_initial_hamiltonian: np.ndarray
    orbital_energies: np.ndarray
    orbital_coefficients: np.ndarray
    dm0: np.ndarray


def compute_common_artifacts(
    atoms: np.ndarray,
    pos_angstrom: np.ndarray,
    flat_hamiltonian: np.ndarray,
) -> DFTArtifacts:
    """Compute common DFT artifacts shared by QH9 datasets.

    Args:
        atoms: Atomic numbers as integers.
        pos_angstrom: Atomic positions in Angstrom with shape (n_atoms * 3,).
        flat_hamiltonian: Flattened Hamiltonian matrix.

    Returns:
        DFTArtifacts containing packed matrices, dm0, orbital information, and forces.
    """

    ovlp, init_ham, mf = calc_overlap_and_init_hamiltonian(
        atoms,
        pos_angstrom.reshape(-1, 3),
        out_mf=True,
        xc="b3lyp",
    )

    h_dim = ovlp.shape[0]
    hamiltonian = flat_hamiltonian.reshape(h_dim, h_dim)

    overlap_tensor = torch.from_numpy(ovlp).unsqueeze(0).to(torch.float64)
    hamiltonian_tensor = torch.from_numpy(hamiltonian).unsqueeze(0).to(torch.float64)

    orbital_energies, orbital_coefficients = cal_orbital_and_energies(
        overlap_tensor,
        hamiltonian_tensor,
    )
    orbital_coefficients = orbital_coefficients.squeeze()

    dm0 = calc_dm0(atoms, orbital_coefficients)

    orbital_coefficients_np = orbital_coefficients.cpu().numpy()
    orbital_energies_np = orbital_energies.cpu().numpy()

    dft_energy = mf.energy_tot(dm0)
    grad_frame = mf.nuc_grad_method()
    mo_occ = mf.get_occ(
        orbital_energies_np.squeeze(),
        orbital_coefficients_np.squeeze(),
    )
    dft_forces = -grad_frame.kernel(
        mo_energy=orbital_energies_np.squeeze(),
        mo_coeff=orbital_coefficients_np.squeeze(),
        mo_occ=mo_occ,
    )

    packed_hamiltonian, h_dim = pack_upper_triangle(hamiltonian)
    packed_overlap, _ = pack_upper_triangle(ovlp)
    packed_init_ham, _ = pack_upper_triangle(init_ham)

    return DFTArtifacts(
        h_dim=h_dim,
        dft_energy=float(dft_energy),
        dft_forces=dft_forces,
        packed_hamiltonian=packed_hamiltonian,
        packed_overlap=packed_overlap,
        packed_initial_hamiltonian=packed_init_ham,
        orbital_energies=orbital_energies_np,
        orbital_coefficients=orbital_coefficients_np,
        dm0=dm0,
    )


class BaseQH9Dataset(InMemoryDataset):
    """Shared dataset utilities for QH9Stable and QH9Dynamic."""

    atom_list: Sequence[str] = ("H", "C", "N", "O", "F")
    atom_num: Dict[str, int] = {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
    }
    full_orbitals: int = 14
    _orbital_mask_line1 = torch.tensor([0, 1, 3, 4, 5])
    _orbital_mask_line2 = torch.arange(full_orbitals)

    shard_cls: type[LMDBShard_maker_db]

    def __init__(
        self,
        *,
        dataset_name: str,
        root: str,
        split: str,
        shard_num: int,
        shard_idx: int | List[int],
        prefix: str = "",
        max_workers_preprocess: int = 8,
        use_parallel_preprocess: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        return_orbital_and_energies: bool = False,
        use_ref_hamiltonian_as_init_ham: bool = False,
        processd_dir_name: str = "processed",
        shard_dir_name: str = "lmdbs",
        compute_q_tensor: bool = True,
        include_initial_hamiltonian: bool = True,
        include_overlap: bool = True,
        include_dft_energy: bool = True,
        include_dft_forces: bool = True,
        include_cut_orbital_coefficients: bool = True,
        cut_coeff_cache_type: str = "both",
    ) -> None:
        if not hasattr(self, "shard_cls"):
            raise AttributeError("Derived dataset must define shard_cls")

        self.dataset_name = dataset_name
        self.split = split
        self.prefix = prefix

        self.folder = os.path.join(root, dataset_name + prefix)
        self.processd_dir_name = processd_dir_name
        self.shard_dir_name = shard_dir_name
        self._processed_path = os.path.join(self.folder, self.processd_dir_name)

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
        if cut_coeff_cache_type not in ("transformed", "original", "both"):
            raise ValueError(
                f"cut_coeff_cache_type must be 'transformed', 'original', or 'both', "
                f"got '{cut_coeff_cache_type}'"
            )
        self.cut_coeff_cache_type = cut_coeff_cache_type

        self.lmdb_path_list = [
            os.path.join(
                self._processed_path,
                self.shard_dir_name,
                f"shard_{i:03d}.lmdb",
            )
            for i in range(self.shard_num)
        ]

        self.orbital_mask = {
            self.atom_num["H"]: self._orbital_mask_line1,
            self.atom_num["C"]: self._orbital_mask_line2,
            self.atom_num["N"]: self._orbital_mask_line2,
            self.atom_num["O"]: self._orbital_mask_line2,
            self.atom_num["F"]: self._orbital_mask_line2,
        }

        self.convention_dict = get_convention_dict()
        self._q_tensor_cache: Dict[tuple, torch.Tensor] = {}
        if self.compute_q_tensor:
            self.Q_dict = Onsite_3idx_Overlap_Integral(
                atom_list=self.atom_list,
                basis="def2-svp",
            ).Q_table()
        else:
            self.Q_dict = None

        self._db_envs: Dict[int, lmdb.Environment] = {}
        self.shard_idx_list: torch.Tensor | None = None
        self.shard_data_idx_list: List[int] = []

        self.basis = ["def2-svp"]
        self.xc = ["pbe"]
        self.init_guess = ["minao"]
        self.dft_conf_tuple = list(zip(self.basis, self.xc, self.init_guess))

        self._ref_hamiltonian_dict: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}

        self.split_path = self._resolve_split_path()
        self.pos_unit = "Angstrom"
        
        super().__init__(self.folder, transform, pre_transform, pre_filter)

        self._load_ref_hamiltonian()
        self._load_split_info()
        self._load_index_info()

    # ------------------------------------------------------------------
    # Required interface for derived classes
    # ------------------------------------------------------------------
    def _resolve_split_path(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Dataset processing
    # ------------------------------------------------------------------
    def process(self) -> None:
        shard_maker = self.shard_cls(
            root_path=self.raw_paths[0],
            shard_num=self.shard_num,
            save_path=self.folder,
            max_workers=self.max_workers_preprocess,
            processd_dir_name=self.processd_dir_name,
            shard_dir_name=self.shard_dir_name,
            use_parallel=self.use_parallel_preprocess,
            split=self.split,
        )
        if self.shard_idx == -1 or self.shard_idx is None:
            shard_maker.process()
        else:
            shard_maker.process(self.shard_idx)

    # ------------------------------------------------------------------
    # Reference Hamiltonian utilities
    # ------------------------------------------------------------------
    def _load_ref_hamiltonian(self) -> None:
        ref_path = os.path.join(self._processed_path, "ref_hamiltonian.pkl")
        if os.path.exists(ref_path):
            with open(ref_path, "rb") as f:
                self._ref_hamiltonian_dict = pickle.load(f)
            return

        for basis, xc, init_guess in self.dft_conf_tuple:
            self._ref_hamiltonian_dict.setdefault(basis, {})
            self._ref_hamiltonian_dict[basis].setdefault(xc, {})
            for atomic_number in self.atom_num.values():
                self._ref_hamiltonian_dict[basis][xc][
                    atomic_number
                ] = self._get_ref_hamiltonian(
                    atomic_number,
                    basis=basis,
                    xc=xc,
                    init_guess=init_guess,
                )

        with open(ref_path, "wb") as f:
            pickle.dump(self._ref_hamiltonian_dict, f)

    def _get_ref_hamiltonian(
        self,
        atoms: int | str,
        *,
        basis: str = "def2-svp",
        xc: str = "pbe",
        init_guess: str = "minao",
    ) -> np.ndarray:
        if isinstance(atoms, str):
            atoms = self.atom_num[atoms]

        self._ref_hamiltonian_dict.setdefault(basis, {})
        self._ref_hamiltonian_dict[basis].setdefault(xc, {})

        if atoms in self._ref_hamiltonian_dict[basis][xc]:
            return self._ref_hamiltonian_dict[basis][xc][atoms]

        from pyscf import dft, gto

        mol_conf = [[atoms, [0, 0, 0]]]
        spin = 1 if atoms % 2 else 0
        mol = gto.Mole()
        mol.build(verbose=0, atom=mol_conf, basis=basis, unit="bohr", spin=spin)
        mf = dft.RKS(mol)
        mf.xc = xc
        init_dm = mf.get_init_guess(key=init_guess)
        init_ham = mf.get_fock(dm=init_dm)
        self._ref_hamiltonian_dict[basis][xc][atoms] = init_ham
        return init_ham

    def _build_ref_hamiltonian(
        self,
        atoms: torch.Tensor,
        *,
        basis: str = "def2-svp",
        xc: str = "pbe",
        init_guess: str = "minao",
    ) -> np.ndarray:
        init_hams: List[np.ndarray] = []
        for atomic_number in atoms:
            atomic_number_int = int(atomic_number.item())
            init_hams.append(
                self._ref_hamiltonian_dict[basis][xc][atomic_number_int]
            )
        return block_diag(*init_hams).astype(np.float64)

    # ------------------------------------------------------------------
    # Split and index helpers
    # ------------------------------------------------------------------
    def _load_split_info(self) -> None:
        with open(self.processed_paths[0], "r") as f:
            self.split_info = json.load(f)

        self.train_mask = torch.tensor(self.split_info["train"], dtype=torch.int64)
        self.val_mask = torch.tensor(self.split_info["val"], dtype=torch.int64)
        self.test_mask = torch.tensor(self.split_info["test"], dtype=torch.int64)

        self.slices = {
            "id": torch.arange(
                self.train_mask.shape[0]
                + self.val_mask.shape[0]
                + self.test_mask.shape[0]
                + 1
            )
        }

    def _load_index_info(self) -> None:
        with open(os.path.join(self._processed_path, "index.json"), "r") as f:
            index_info = json.load(f)["index"]

        shard_indices: List[int] = []
        shard_data_indices: List[int] = []
        for idx, (shard_idx, cur_idx, shard_data_idx) in enumerate(index_info):
            if cur_idx != idx:
                raise AssertionError(
                    f"Shard index {cur_idx} does not match enumeration index {idx}"
                )
            shard_indices.append(shard_idx)
            shard_data_indices.append(shard_data_idx)

        self.shard_idx_list = torch.tensor(shard_indices, dtype=torch.int64)
        self.shard_data_idx_list = shard_data_indices

        max_shard_idx = torch.max(self.shard_idx_list)
        if max_shard_idx != self.shard_num - 1:
            raise AssertionError(
                f"Max shard index {max_shard_idx} does not match shard_num {self.shard_num}"
            )

    # ------------------------------------------------------------------
    # LMDB helpers
    # ------------------------------------------------------------------
    def _get_shard_db_env(self, idx: int):
        shard_idx = int(self.shard_idx_list[idx].item())
        if shard_idx in self._db_envs:
            try:
                with self._db_envs[shard_idx].begin() as txn:
                    txn.stat()
                return self._db_envs[shard_idx]
            except Exception:
                try:
                    self._db_envs[shard_idx].close()
                except Exception:
                    pass
                del self._db_envs[shard_idx]

        db_env = lmdb.open(
            self.lmdb_path_list[shard_idx],
            readonly=True,
            lock=False,
            max_readers=1024,
            readahead=False,
        )
        self._db_envs[shard_idx] = db_env
        return db_env

    def get(self, idx: int) -> Data:
        try:
            return self._get(idx)
        except Exception:
            shard_idx = int(self.shard_idx_list[idx].item())
            if shard_idx in self._db_envs:
                try:
                    self._db_envs[shard_idx].close()
                except Exception:
                    pass
                del self._db_envs[shard_idx]
            return self._get(idx)

    def _get(self, idx: int) -> Data:
        db_env = self._get_shard_db_env(idx)
        with db_env.begin() as txn:
            key = int(idx).to_bytes(length=4, byteorder="big")
            serialized = txn.get(key)
            if serialized is None:
                raise KeyError(
                    f"Index {idx}, shard_data_idx {self.shard_data_idx_list[idx]} not found"
                )
            data_dict = pickle.loads(serialized)
            data = self._build_data_object(data_dict, idx)
        return data

    def get_gt_outputs(self, idx: int) -> Data:
        db_env = self._get_shard_db_env(idx)
        with db_env.begin() as txn:
            key = int(idx).to_bytes(length=4, byteorder="big")
            serialized = txn.get(key)
            if serialized is None:
                raise KeyError(
                    f"Index {idx}, shard_data_idx {self.shard_data_idx_list[idx]} not found"
                )
            data_dict = pickle.loads(serialized)
            data = self._get_gt_outputs(data_dict, idx)
        return data

    def get_key_list(self, idx: int) -> List[int]:
        db_env = self._get_shard_db_env(idx)
        with db_env.begin() as txn:
            return [int.from_bytes(key, byteorder="big") for key, _ in txn.cursor()]

    def _close_db_envs(self) -> None:
        for shard_idx, db_env in list(self._db_envs.items()):
            try:
                db_env.close()
            except Exception:
                pass
        self._db_envs.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_db_envs()

    def __del__(self):
        self._close_db_envs()

    # ------------------------------------------------------------------
    # Data decoding utilities
    # ------------------------------------------------------------------
    @staticmethod
    def unpack_upper_triangle(packed: np.ndarray, h_dim: int) -> np.ndarray:
        return unpack_upper_triangle(packed, h_dim)

    def matrix_transform(self, hamiltonian: torch.Tensor, atoms: torch.Tensor, *, convention: str) -> torch.Tensor:
        return matrix_transform_single(hamiltonian, atoms, self.convention_dict[convention])

    def _decode_positions(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        pos = np.frombuffer(data_dict["pos"], np.float64).reshape(-1, 3)
        pos_unit = data_dict.get("pos_unit", "Angstrom").lower()
        if pos_unit == "bohr" or self.pos_unit.lower() == "bohr":
            pos = pos * BOHR2ANG
        return torch.from_numpy(pos).to(torch.float64)

    def _get_gt_outputs(self, data_dict: Dict[str, Any], idx: int = None) -> Dict[str, Any]:
        atoms = torch.tensor(
            np.frombuffer(data_dict["atoms"], np.int32),
            dtype=torch.int64,
        )
        num_nodes = torch.tensor(data_dict["num_nodes"], dtype=torch.int64)
        pos = self._decode_positions(data_dict)

        dft_energy = (
            torch.tensor(data_dict["dft_energy"], dtype=torch.float64)
            if self.include_dft_energy
            else None
        )
        dft_forces = (
            torch.tensor(
                np.frombuffer(data_dict["dft_forces"], np.float64).reshape(-1, 3),
                dtype=torch.float64,
            )
            if self.include_dft_forces
            else None
        )

        h_dim = data_dict["h_dim"]
        packed_hamiltonian = np.frombuffer(data_dict["packed_hamiltonian"], np.float64)
        packed_overlap = np.frombuffer(data_dict["packed_overlap"], np.float64)
        hamiltonian = torch.from_numpy(
            self.unpack_upper_triangle(packed_hamiltonian, h_dim)
        ).to(torch.float64)
        overlap_matrix = torch.from_numpy(
            self.unpack_upper_triangle(packed_overlap, h_dim)
        ).to(torch.float64)
        packed_init_ham = np.frombuffer(
            data_dict["packed_initial_hamiltonian"], np.float64
        )
        initial_hamiltonian = torch.from_numpy(
            self.unpack_upper_triangle(packed_init_ham, h_dim)
        ).to(torch.float64)

        data = {
            "pos": pos,
            "atoms": atoms.view(-1, 1),
            "hamiltonian": hamiltonian,
            "overlap_matrix": overlap_matrix,
            "initial_hamiltonian": initial_hamiltonian,
            "num_nodes": num_nodes,
            "h_dim": torch.tensor(h_dim, dtype=torch.int64),
            "idx": idx if idx is not None else None, # For debugging
        }
        return data

    def _build_data_object(self, data_dict: Dict[str, Any], idx: int = None) -> Data:
        atoms = torch.tensor(
            np.frombuffer(data_dict["atoms"], np.int32),
            dtype=torch.int64,
        )
        num_nodes = torch.tensor(data_dict["num_nodes"], dtype=torch.int64)
        pos = self._decode_positions(data_dict)

        dft_energy = (
            torch.tensor(data_dict["dft_energy"], dtype=torch.float64)
            if self.include_dft_energy
            else None
        )
        dft_forces = (
            torch.tensor(
                np.frombuffer(data_dict["dft_forces"], np.float64).reshape(-1, 3),
                dtype=torch.float64,
            )
            if self.include_dft_forces
            else None
        )

        h_dim = data_dict["h_dim"]
        packed_hamiltonian = np.frombuffer(data_dict["packed_hamiltonian"], np.float64)
        packed_overlap = np.frombuffer(data_dict["packed_overlap"], np.float64)

        hamiltonian = torch.from_numpy(
            self.unpack_upper_triangle(packed_hamiltonian, h_dim)
        ).to(torch.float64)
        overlap_matrix = torch.from_numpy(
            self.unpack_upper_triangle(packed_overlap, h_dim)
        ).to(torch.float64)

        if not self.use_ref_hamiltonian_as_init_ham:
            packed_init_ham = np.frombuffer(
                data_dict["packed_initial_hamiltonian"], np.float64
            )
            initial_hamiltonian = torch.from_numpy(
                self.unpack_upper_triangle(packed_init_ham, h_dim)
            ).to(torch.float64)
        else:
            ref_hamiltonian = self._build_ref_hamiltonian(
                atoms,
                basis=self.basis[0],
                xc=self.xc[0],
                init_guess=self.init_guess[0],
            )
            initial_hamiltonian = torch.from_numpy(ref_hamiltonian).to(torch.float64)

        convention = "pyscf_def2svp_to_e3nn"
        concat_feat = torch.stack(
            [hamiltonian, overlap_matrix, initial_hamiltonian], dim=0
        )
        transformed = self.matrix_transform(
            concat_feat,
            atoms,
            convention=convention,
        )

        (
            concat_diagonal_blocks,
            concat_non_diagonal_blocks,
            concat_diagonal_masks,
            concat_non_diagonal_masks,
            edge_index_full,
        ) = _cut_matrix_3d(
            transformed,
            atoms,
            self.orbital_mask,
            self.full_orbitals,
        )

        diagonal_hamiltonian = concat_diagonal_blocks[:, 0]
        non_diagonal_hamiltonian = concat_non_diagonal_blocks[:, 0]
        diagonal_hamiltonian_mask = concat_diagonal_masks[:, 0]
        non_diagonal_hamiltonian_mask = concat_non_diagonal_masks[:, 0]

        diagonal_overlap = concat_diagonal_blocks[:, 1]
        non_diagonal_overlap = concat_non_diagonal_blocks[:, 1]

        diagonal_init_ham = concat_diagonal_blocks[:, 2]
        non_diagonal_init_ham = concat_non_diagonal_blocks[:, 2]

        # If cut_coeff_cache_type is "original" or "both", also create untransformed hamiltonian blocks
        diagonal_hamiltonian_untransformed = None
        non_diagonal_hamiltonian_untransformed = None
        if self.include_cut_orbital_coefficients and self.cut_coeff_cache_type in ("original", "both"):
            (
                concat_diagonal_blocks_untransformed,
                concat_non_diagonal_blocks_untransformed,
                _,
                _,
                _,
            ) = _cut_matrix_3d(
                hamiltonian.unsqueeze(0),
                atoms,
                self.orbital_mask,
                self.full_orbitals,
            )
            diagonal_hamiltonian_untransformed = concat_diagonal_blocks_untransformed[:, 0]
            non_diagonal_hamiltonian_untransformed = concat_non_diagonal_blocks_untransformed[:, 0]

        diagonal_Q = None
        non_diagonal_Q = None
        if self.compute_q_tensor:
            atom_key = tuple(int(z.item()) for z in atoms)
            cache_key = (atom_key, self.pos_unit)
            if cache_key in self._q_tensor_cache:
                cached_q = self._q_tensor_cache[cache_key]
            else:
                cached_q = self._build_Q_tensor(atoms).detach().cpu()
                self._q_tensor_cache[cache_key] = cached_q

            Q = cached_q.to(device=transformed.device, dtype=transformed.dtype)
            (
                diagonal_Q,
                non_diagonal_Q,
                _,
                _,
                _,
            ) = _cut_matrix_3d_last(
                Q,
                atoms,
                self.orbital_mask,
                self.full_orbitals,
            )

        cut_orb_coeff = None
        cut_orb_coeff_transformed = None
        cut_orb_coeff_original = None
        if self.include_cut_orbital_coefficients:
            # Transformed (current convention) cut orbital coefficients
            compute_transformed = self.cut_coeff_cache_type in ("transformed", "both")
            compute_original = self.cut_coeff_cache_type in ("original", "both")

            if compute_transformed:
                transformed_overlap_matrix = transformed[1]
                transformed_hamiltonian = transformed[0]
                _, transformed_coeffs = cal_orbital_and_energies(
                    transformed_overlap_matrix.unsqueeze(0),
                    transformed_hamiltonian.unsqueeze(0),
                )
                cut_orb_coeff_transformed_raw = cut_orbital_coefficients(
                    transformed_coeffs.squeeze(0),
                    atoms.view(-1),
                    self.orbital_mask,
                    self.full_orbitals,
                )
                cut_orb_coeff_transformed = [
                    block.to(device=transformed.device, dtype=transformed.dtype)
                    for block in cut_orb_coeff_transformed_raw
                ]

            if compute_original:
                _, original_coeffs = cal_orbital_and_energies(overlap_matrix.unsqueeze(0),hamiltonian.unsqueeze(0),)
                cut_orb_coeff_original_raw = cut_orbital_coefficients(
                    original_coeffs.squeeze(0),
                    atoms.view(-1),
                    self.orbital_mask,
                    self.full_orbitals,
                )
                cut_orb_coeff_original = [
                    block.to(device=transformed.device, dtype=transformed.dtype)
                    for block in cut_orb_coeff_original_raw
                ]
        
        # Debug code - compute CHC from transformed cut coefficients
        chc_block_trans = None
        if self.include_cut_orbital_coefficients and cut_orb_coeff_transformed is not None:
            chc_block_trans = compute_chc_from_blocks(
                torch.stack(cut_orb_coeff_transformed),
                diagonal_hamiltonian,
                non_diagonal_hamiltonian,
                edge_index_full
            )

        data_kwargs: Dict[str, Any] = {
            "pos": pos,
            "atoms": atoms.view(-1, 1),
            "diagonal_hamiltonian": diagonal_hamiltonian,
            "non_diagonal_hamiltonian": non_diagonal_hamiltonian,
            "diagonal_hamiltonian_mask": diagonal_hamiltonian_mask,
            "non_diagonal_hamiltonian_mask": non_diagonal_hamiltonian_mask,
            "edge_index_full": edge_index_full,
            "num_nodes": num_nodes,
            "h_dim": torch.tensor(h_dim, dtype=torch.int64),
            "idx": idx if idx is not None else None, # For debugging
        }
        
        if chc_block_trans is not None:
            data_kwargs["chc_block_trans"] = chc_block_trans.squeeze(0)
            data_kwargs["coeff_transformed"] = transformed_coeffs.numpy()

        if self.include_initial_hamiltonian:
            data_kwargs["diagonal_init_ham"] = diagonal_init_ham
            data_kwargs["non_diagonal_init_ham"] = non_diagonal_init_ham

        if self.include_overlap:
            data_kwargs["diagonal_overlap"] = diagonal_overlap
            data_kwargs["non_diagonal_overlap"] = non_diagonal_overlap

        if self.include_cut_orbital_coefficients and self.cut_coeff_cache_type in ("original", "both"):
            if diagonal_hamiltonian_untransformed is not None:
                data_kwargs["diagonal_hamiltonian_untransformed"] = diagonal_hamiltonian_untransformed
            if non_diagonal_hamiltonian_untransformed is not None:
                data_kwargs["non_diagonal_hamiltonian_untransformed"] = non_diagonal_hamiltonian_untransformed

        if self.include_dft_energy and dft_energy is not None:
            data_kwargs["dft_energy"] = dft_energy.view(1, 1)

        if self.include_dft_forces and dft_forces is not None:
            data_kwargs["dft_forces"] = dft_forces

        data = Data(**data_kwargs)

        if self.compute_q_tensor and diagonal_Q is not None:
            data.diagonal_Q = diagonal_Q
            data.non_diagonal_Q = non_diagonal_Q

        if self.include_cut_orbital_coefficients:
            # Select which cut_coeff to use based on cut_coeff_cache_type
            cut_orb_coeff_to_use = None
            
            if self.cut_coeff_cache_type == "transformed" and cut_orb_coeff_transformed is not None:
                cut_orb_coeff_to_use = cut_orb_coeff_transformed
            
            elif self.cut_coeff_cache_type == "original" and cut_orb_coeff_original is not None:
                cut_orb_coeff_to_use = cut_orb_coeff_original
            
            elif self.cut_coeff_cache_type == "both":
                if cut_orb_coeff_original is not None:
                    cut_orb_coeff_to_use = cut_orb_coeff_original
                
                if cut_orb_coeff_transformed is not None:
                    data.cut_orbital_coefficients_transformed = TensorListWrapper(
                        [block.detach().cpu() for block in cut_orb_coeff_transformed]
                    )
            
            if cut_orb_coeff_to_use is not None:
                data.cut_orbital_coefficients = TensorListWrapper(
                    [block.detach().cpu() for block in cut_orb_coeff_to_use]
                )

        if self.return_orbital_and_energies:
            self._attach_orbital_data(
                data,
                data_dict,
                overlap_matrix,
                hamiltonian,
                h_dim,
            )

        return data

    def _build_Q_tensor(self, atoms: torch.Tensor) -> torch.Tensor:
        if not self.compute_q_tensor:
            raise RuntimeError("Q tensor computation is disabled for this dataset instance.")
        blocks: List[torch.Tensor] = []
        for l in range(60):
            block_diag_components = [self.Q_dict[int(z.item())][l] for z in atoms]
            blocks.append(torch.block_diag(*block_diag_components))

        Q = torch.stack(blocks)
        Q = self.matrix_transform(
            Q,
            atoms,
            convention="pyscf_def2svp_to_e3nn",
        ).permute(1, 2, 0)

        Q_slice = Q[:, :, 16:40]
        Q[:, :, 16:40] = (
            Q_slice.reshape(Q.shape[0], Q.shape[1], -1, 3)[:, :, :, [1, 2, 0]].reshape(
                Q.shape[0], Q.shape[1], 24
            )
        )
        return Q

    def _attach_orbital_data(
        self,
        data: Data,
        data_dict: Dict[str, Any],
        overlap_matrix: torch.Tensor,
        hamiltonian: torch.Tensor,
        h_dim: int,
    ) -> None:
        orbital_energies = np.frombuffer(data_dict["orbital_energies"], np.float64)
        orbital_energies_tensor = torch.from_numpy(orbital_energies.copy()).to(
            torch.float64
        )
        data.orbital_energies = self._postprocess_orbital_energies(
            orbital_energies_tensor
        )

        if "orbital_coefficients" in data_dict:
            orbital_coefficients = np.frombuffer(
                data_dict["orbital_coefficients"],
                np.float64,
            ).reshape(h_dim, h_dim)
        else:
            recalculated_energies, recalculated_coefficients = cal_orbital_and_energies(
                overlap_matrix.unsqueeze(0),
                hamiltonian.unsqueeze(0),
            )
            orbital_coefficients = recalculated_coefficients.numpy().reshape(
                h_dim, h_dim
            )

        data.orbital_coefficients = self._postprocess_orbital_coefficients(
            orbital_coefficients
        )

    # ------------------------------------------------------------------
    # Hook points for dataset-specific behaviour
    # ------------------------------------------------------------------
    def _postprocess_orbital_energies(self, orbital_energies: torch.Tensor) -> torch.Tensor:
        return orbital_energies

    def _postprocess_orbital_coefficients(
        self, orbital_coefficients: np.ndarray
    ) -> np.ndarray:
        return orbital_coefficients


# ==============================================================================
# Debugging Functions for Eigenvalue and CHC Verification
# ==============================================================================

def debug_eigenvalue_ordering_and_chc(
    dataset: BaseQH9Dataset,
    debug_indices: list[int],
    logger=None,
) -> dict:
    """
    Comprehensive debugging function to verify eigenvalue ordering and CHC calculations.
    
    This function performs multiple checks:
    1. Stored eigenvalue ordering
    2. Eigenvalue recalculation from original matrices
    3. Eigenvalue consistency after transform
    4. Block-wise CHC calculation verification
    5. Flow module CHC calculation verification
    
    Args:
        dataset: QH9 dataset instance
        debug_indices: List of indices to debug
        logger: Logger instance (optional)
        
    Returns:
        Dictionary containing error lists for each check type
    """
    import pickle
    import numpy as np
    import torch
    from common.metric import cal_orbital_and_energies
    from common.matrix_transforms import (
        compute_chc_from_blocks,
        _reconstruct_full_matrix,
        cut_orbital_coefficients,
    )
    from common.custom_logger import get_logger
    
    if logger is None:
        logger = get_logger(__file__)
    
    eigenvalue_ordering_errors = []
    eigenvalue_recalc_errors = []
    chc_eigenvalue_errors = []
    block_chc_errors = []
    
    for debug_idx in debug_indices:
        data = dataset[debug_idx]
        
        try:
            h_dim = int(data.h_dim.item())
            
            # Get raw data from database
            db_env = dataset._get_shard_db_env(debug_idx)
            with db_env.begin() as txn:
                key = int(debug_idx).to_bytes(length=4, byteorder="big")
                serialized = txn.get(key)
                if serialized is None:
                    continue
                    
                data_dict = pickle.loads(serialized)
                
                # Check 1: Stored eigenvalue ordering
                stored_eigenvalues = np.frombuffer(data_dict["orbital_energies"], np.float64)
                if not _check_stored_eigenvalue_ordering(
                    debug_idx, stored_eigenvalues, logger
                ):
                    eigenvalue_ordering_errors.append(debug_idx)
                
                # Check 2: Eigenvalue recalculation
                packed_hamiltonian = np.frombuffer(data_dict["packed_hamiltonian"], np.float64)
                packed_overlap = np.frombuffer(data_dict["packed_overlap"], np.float64)
                
                hamiltonian_full = dataset.unpack_upper_triangle(packed_hamiltonian, h_dim)
                overlap_full = dataset.unpack_upper_triangle(packed_overlap, h_dim)
                
                hamiltonian_tensor = torch.from_numpy(hamiltonian_full).to(torch.float64)
                overlap_tensor = torch.from_numpy(overlap_full).to(torch.float64)
                
                recalc_energies_original, _ = cal_orbital_and_energies(
                    overlap_tensor.unsqueeze(0),
                    hamiltonian_tensor.unsqueeze(0),
                )
                recalc_energies_original = recalc_energies_original.squeeze().detach().cpu().numpy()
                
                if not _check_eigenvalue_recalculation(
                    debug_idx,
                    stored_eigenvalues,
                    recalc_energies_original,
                    logger,
                ):
                    eigenvalue_recalc_errors.append(debug_idx)
                
                # Check 3: Transform eigenvalues
                atoms = data.atoms.view(-1)
                
                if not dataset.use_ref_hamiltonian_as_init_ham:
                    packed_init_ham = np.frombuffer(
                        data_dict["packed_initial_hamiltonian"], np.float64
                    )
                    initial_hamiltonian = torch.from_numpy(
                        dataset.unpack_upper_triangle(packed_init_ham, h_dim)
                    ).to(torch.float64)
                else:
                    ref_hamiltonian = dataset._build_ref_hamiltonian(
                        atoms,
                        basis=dataset.basis[0],
                        xc=dataset.xc[0],
                        init_guess=dataset.init_guess[0],
                    )
                    initial_hamiltonian = torch.from_numpy(ref_hamiltonian).to(torch.float64)
                
                convention = "pyscf_def2svp_to_e3nn"
                concat_feat = torch.stack(
                    [hamiltonian_tensor, overlap_tensor, initial_hamiltonian], dim=0
                )
                transformed = dataset.matrix_transform(
                    concat_feat,
                    atoms,
                    convention=convention,
                )
                transformed_hamiltonian = transformed[0]
                transformed_overlap = transformed[1]
                
                recalc_energies_transformed, recalc_coeffs_transformed = cal_orbital_and_energies(
                    transformed_overlap.unsqueeze(0),
                    transformed_hamiltonian.unsqueeze(0),
                )
                recalc_energies_transformed = recalc_energies_transformed.squeeze().detach().cpu().numpy()
                recalc_coeffs_transformed = recalc_coeffs_transformed.squeeze(0).detach().cpu()
                
                # Also get original coefficients for comparison
                _, recalc_coeffs_original = cal_orbital_and_energies(
                    overlap_tensor.unsqueeze(0),
                    hamiltonian_tensor.unsqueeze(0),
                )
                recalc_coeffs_original = recalc_coeffs_original.squeeze(0).detach().cpu()
                
                if not _check_transform_eigenvalues(
                    debug_idx,
                    recalc_energies_original,
                    recalc_energies_transformed,
                    stored_eigenvalues,
                    logger,
                ):
                    chc_eigenvalue_errors.append(debug_idx)
                
                # Check: Transform된 C와 H에 대한 CHC가 기존 eigenvalue와 같은지 확인
                if not _check_transformed_chc_matches_original_eigenvalues(
                    debug_idx,
                    transformed_hamiltonian,
                    transformed_overlap,
                    recalc_coeffs_transformed,
                    recalc_coeffs_original,
                    hamiltonian_tensor,
                    overlap_tensor,
                    recalc_energies_original,
                    stored_eigenvalues,
                    atoms,
                    dataset,
                    logger,
                ):
                    chc_eigenvalue_errors.append(debug_idx)
                
                # Check 4 & 5: Block-wise CHC and flow_module CHC
                if not _check_block_wise_chc(
                    debug_idx,
                    data,
                    dataset,
                    data_dict,
                    stored_eigenvalues,
                    logger,
                ):
                    block_chc_errors.append(debug_idx)
                    
        except Exception as e:
            logger.error(f"Index {debug_idx}: Error during debugging: {e}")
            import traceback
            traceback.print_exc()
            block_chc_errors.append(debug_idx)
    
    # Summary
    logger.info("=" * 80)
    logger.info("Eigenvalue and CHC Debugging Summary:")
    logger.info(f"  Total samples checked: {len(debug_indices)}")
    logger.info(f"  Eigenvalue ordering errors: {len(eigenvalue_ordering_errors)}")
    if eigenvalue_ordering_errors:
        logger.warning(f"    Indices with ordering errors: {eigenvalue_ordering_errors}")
    logger.info(f"  Eigenvalue recalculation errors: {len(eigenvalue_recalc_errors)}")
    if eigenvalue_recalc_errors:
        logger.warning(f"    Indices with recalculation errors: {eigenvalue_recalc_errors}")
    logger.info(f"  CHC eigenvalue errors: {len(chc_eigenvalue_errors)}")
    if chc_eigenvalue_errors:
        logger.warning(f"    Indices with CHC eigenvalue errors: {chc_eigenvalue_errors}")
    logger.info(f"  Block-wise CHC errors: {len(block_chc_errors)}")
    if block_chc_errors:
        logger.warning(f"    Indices with block-wise CHC errors: {block_chc_errors}")
    logger.info("=" * 80)
    
    return {
        "eigenvalue_ordering_errors": eigenvalue_ordering_errors,
        "eigenvalue_recalc_errors": eigenvalue_recalc_errors,
        "chc_eigenvalue_errors": chc_eigenvalue_errors,
        "block_chc_errors": block_chc_errors,
    }


def _check_stored_eigenvalue_ordering(
    debug_idx: int,
    stored_eigenvalues: np.ndarray,
    logger,
) -> bool:
    """Check if stored eigenvalues are properly ordered."""
    is_sorted = np.all(stored_eigenvalues[:-1] <= stored_eigenvalues[1:])
    if not is_sorted:
        logger.warning(f"Index {debug_idx}: Eigenvalues are NOT properly ordered (ascending)")
        logger.warning(f"  Stored eigenvalues: {stored_eigenvalues[:10]}... (showing first 10)")
        return False
    else:
        logger.debug(f"Index {debug_idx}: Eigenvalues are properly ordered")
        return True


def _check_eigenvalue_recalculation(
    debug_idx: int,
    stored_eigenvalues: np.ndarray,
    recalc_energies: np.ndarray,
    logger,
) -> bool:
    """Check if recalculated eigenvalues match stored ones."""
    stored_sorted = np.sort(stored_eigenvalues)
    recalc_sorted = np.sort(recalc_energies)
    
    if len(stored_sorted) != len(recalc_sorted):
        logger.error(f"Index {debug_idx}: Mismatch in eigenvalue count: stored={len(stored_sorted)}, recalc={len(recalc_sorted)}")
        return False
    
    max_diff = np.abs(stored_sorted - recalc_sorted).max()
    mean_diff = np.abs(stored_sorted - recalc_sorted).mean()
    
    if max_diff > 1e-6:
        logger.warning(f"Index {debug_idx}: Large difference in recalculated eigenvalues")
        logger.warning(f"  Max difference: {max_diff:.2e}, Mean difference: {mean_diff:.2e}")
        return False
    else:
        logger.debug(f"Index {debug_idx}: Recalculated eigenvalues match stored values (max_diff={max_diff:.2e})")
        return True


def _check_transform_eigenvalues(
    debug_idx: int,
    recalc_energies_original: np.ndarray,
    recalc_energies_transformed: np.ndarray,
    stored_eigenvalues: np.ndarray,
    logger,
) -> bool:
    """Check eigenvalue consistency after transform."""
    is_transformed_sorted = np.all(recalc_energies_transformed[:-1] <= recalc_energies_transformed[1:])
    if not is_transformed_sorted:
        logger.warning(f"Index {debug_idx}: Transformed eigenvalues are NOT properly ordered")
        return False
    
    recalc_original_sorted = np.sort(recalc_energies_original)
    recalc_transformed_sorted = np.sort(recalc_energies_transformed)
    
    if len(recalc_original_sorted) != len(recalc_transformed_sorted):
        logger.error(f"Index {debug_idx}: Mismatch in eigenvalue count after transform")
        return False
    
    max_diff_transform = np.abs(recalc_original_sorted - recalc_transformed_sorted).max()
    
    if max_diff_transform > 1e-5:
        logger.warning(f"Index {debug_idx}: Large difference between original and transformed eigenvalues")
        logger.warning(f"  Max difference: {max_diff_transform:.2e}")
        return False
    else:
        logger.debug(f"Index {debug_idx}: Transformed eigenvalues match original (max_diff={max_diff_transform:.2e})")
    
    stored_sorted = np.sort(stored_eigenvalues)
    if len(stored_sorted) == len(recalc_transformed_sorted):
        max_diff_stored_transform = np.abs(stored_sorted - recalc_transformed_sorted).max()
        if max_diff_stored_transform > 1e-5:
            logger.warning(f"Index {debug_idx}: Stored eigenvalues differ from transformed eigenvalues")
            return False
    
    if not np.allclose(recalc_energies_original, recalc_energies_transformed, atol=1e-5):
        if np.allclose(recalc_original_sorted, recalc_transformed_sorted, atol=1e-5):
            logger.warning(f"Index {debug_idx}: Eigenvalues match but ordering differs after transform")
            return False
        else:
            logger.error(f"Index {debug_idx}: Eigenvalues do not match even when sorted after transform")
            return False
    
    return True


def _check_block_wise_chc(
    debug_idx: int,
    data,
    dataset: BaseQH9Dataset,
    data_dict: dict,
    stored_eigenvalues: np.ndarray,
    logger,
) -> bool:
    """Check block-wise CHC calculation and flow_module CHC calculation."""
    import pickle
    import numpy as np
    import torch
    from common.metric import cal_orbital_and_energies
    from common.matrix_transforms import (
        compute_chc_from_blocks,
        _reconstruct_full_matrix,
        cut_orbital_coefficients,
    )
    
    try:
        # Get diagonal and non-diagonal blocks from data
        diag_ham = data.diagonal_hamiltonian.detach().cpu()
        non_diag_ham = data.non_diagonal_hamiltonian.detach().cpu()
        diag_overlap = data.diagonal_overlap.detach().cpu() if hasattr(data, 'diagonal_overlap') else None
        non_diag_overlap = data.non_diagonal_overlap.detach().cpu() if hasattr(data, 'non_diagonal_overlap') else None
        edge_index_full = data.edge_index_full.detach().cpu()
        atoms = data.atoms.view(-1).detach().cpu()
        
        # Get orbital masks
        orbital_mask = dataset.orbital_mask
        full_orbitals = dataset.full_orbitals
        mask_list = [orbital_mask[int(z.item())] for z in atoms]
        
        # Reconstruct full matrices from blocks
        diag_ham_work = diag_ham.to(torch.float64)
        non_diag_ham_work = non_diag_ham.to(torch.float64)
        local_edges = edge_index_full.clone()
        
        full_h = _reconstruct_full_matrix(diag_ham_work, non_diag_ham_work, local_edges, mask_list)
        
        if diag_overlap is not None:
            diag_overlap_work = diag_overlap.to(torch.float64)
            if non_diag_overlap is not None:
                non_diag_overlap_work = non_diag_overlap.to(torch.float64)
            else:
                full_size = diag_ham_work.shape[-1]
                non_diag_overlap_work = diag_ham_work.new_zeros((0, full_size, full_size))
            full_s = _reconstruct_full_matrix(diag_overlap_work, non_diag_overlap_work, local_edges, mask_list)
        else:
            full_s = torch.eye(full_h.shape[-1], dtype=torch.float64)
        
        # Calculate orbital energies from full matrix
        full_h_batch = full_h.unsqueeze(0)
        full_s_batch = full_s.unsqueeze(0)
        orbital_energies_full, orbital_coeffs_full = cal_orbital_and_energies(
            full_s_batch,
            full_h_batch,
        )
        orbital_energies_full = orbital_energies_full.squeeze().detach().cpu().numpy()
        orbital_coeffs_full = orbital_coeffs_full.squeeze(0).detach().cpu()
        
        # Get cut coefficients
        cut_coeff = cut_orbital_coefficients(
            orbital_coeffs_full,
            atoms,
            orbital_mask,
            full_orbitals,
        )
        
        # Calculate CHC from blocks
        chc_from_blocks = compute_chc_from_blocks(
            cut_coeff,
            diag_ham_work,
            non_diag_ham_work,
            local_edges,
        )
        chc_from_blocks = chc_from_blocks.detach().cpu().numpy()
        
        # Check basic CHC calculation
        if not _verify_chc_matches_energies(
            debug_idx,
            chc_from_blocks,
            orbital_energies_full,
            stored_eigenvalues,
            logger,
            label="Block-wise CHC",
        ):
            return False
        
        # Check flow_module-style CHC calculation
        return _check_flow_module_chc(
            debug_idx,
            diag_ham_work,
            non_diag_ham_work,
            diag_overlap,
            non_diag_overlap,
            local_edges,
            mask_list,
            atoms,
            orbital_mask,
            full_orbitals,
            chc_from_blocks,
            logger,
        )
        
    except Exception as e:
        logger.error(f"Index {debug_idx}: Error during block-wise CHC debugging: {e}")
        import traceback
        traceback.print_exc()
        return False


def _verify_chc_matches_energies(
    debug_idx: int,
    chc_values: np.ndarray,
    orbital_energies: np.ndarray,
    stored_eigenvalues: np.ndarray,
    logger,
    label: str = "CHC",
) -> bool:
    """Verify that CHC values match orbital energies."""
    import numpy as np
    
    if len(chc_values) != len(orbital_energies):
        logger.error(f"Index {debug_idx}: Mismatch in {label}/energy count: CHC={len(chc_values)}, energies={len(orbital_energies)}")
        return False
    
    chc_sorted = np.sort(chc_values)
    energies_sorted = np.sort(orbital_energies)
    
    max_diff = np.abs(chc_sorted - energies_sorted).max()
    
    if max_diff > 1e-5:
        logger.warning(f"Index {debug_idx}: {label} does NOT match orbital energies")
        logger.warning(f"  Max difference: {max_diff:.2e}")
        return False
    
    # Check ordering
    if not np.allclose(chc_values, orbital_energies, atol=1e-5):
        if np.allclose(chc_sorted, energies_sorted, atol=1e-5):
            logger.warning(f"Index {debug_idx}: {label} values match but ordering differs")
            return False
        else:
            logger.error(f"Index {debug_idx}: {label} values do not match even when sorted")
            return False
    
    # Compare with stored eigenvalues
    stored_sorted = np.sort(stored_eigenvalues)
    if len(chc_values) == len(stored_eigenvalues):
        chc_stored_diff = np.abs(chc_sorted - stored_sorted).max()
        if chc_stored_diff > 1e-5:
            logger.warning(f"Index {debug_idx}: {label} differs from stored eigenvalues")
            return False
    
    return True


def _check_flow_module_chc(
    debug_idx: int,
    diag_ham_work: torch.Tensor,
    non_diag_ham_work: torch.Tensor,
    diag_overlap: torch.Tensor,
    non_diag_overlap: torch.Tensor,
    local_edges: torch.Tensor,
    mask_list: list,
    atoms: torch.Tensor,
    orbital_mask: dict,
    full_orbitals: int,
    chc_from_blocks: np.ndarray,
    logger,
) -> bool:
    """Check flow_module-style CHC calculation."""
    import numpy as np
    import torch
    from common.metric import cal_orbital_and_energies
    from common.matrix_transforms import (
        compute_chc_from_blocks,
        _reconstruct_full_matrix,
        cut_orbital_coefficients,
    )
    
    try:
        logger.info(f"Index {debug_idx}: Verifying flow_module.py CHC calculation approach...")
        
        # Compute cut_coeff from target Hamiltonian (same as flow_module)
        diag_target_work = diag_ham_work.clone()
        off_target_work = non_diag_ham_work.clone()
        full_h_target = _reconstruct_full_matrix(diag_target_work, off_target_work, local_edges, mask_list)
        
        if diag_overlap is not None:
            full_s_target = _reconstruct_full_matrix(
                diag_overlap.to(torch.float64),
                non_diag_overlap.to(torch.float64) if non_diag_overlap is not None else diag_ham_work.new_zeros((0, diag_ham_work.shape[-1], diag_ham_work.shape[-1])),
                local_edges,
                mask_list,
            )
        else:
            full_s_target = torch.eye(full_h_target.shape[-1], dtype=torch.float64)
        
        full_h_target_batch = full_h_target.unsqueeze(0)
        full_s_target_batch = full_s_target.unsqueeze(0)
        orbital_energies_target, orbital_coeffs_target = cal_orbital_and_energies(
            full_s_target_batch,
            full_h_target_batch,
        )
        orbital_coeffs_target = orbital_coeffs_target.squeeze(0).detach().cpu()
        
        # Get cut coefficients from target
        cut_coeff_target = cut_orbital_coefficients(
            orbital_coeffs_target,
            atoms,
            orbital_mask,
            full_orbitals,
        )
        
        # Compute CHC using target cut_coeff with target Hamiltonian
        chc_target_flow = compute_chc_from_blocks(
            cut_coeff_target,
            diag_target_work,
            off_target_work,
            local_edges,
        )
        chc_target_flow = chc_target_flow.detach().cpu().numpy()
        orbital_energies_target_np = orbital_energies_target.squeeze().detach().cpu().numpy()
        
        # Check if orbital energies are sorted
        is_energies_sorted = np.all(orbital_energies_target_np[:-1] <= orbital_energies_target_np[1:])
        if not is_energies_sorted:
            logger.warning(f"Index {debug_idx}: WARNING - Orbital energies from eigh are NOT sorted!")
            logger.warning(f"  First 10 energies: {orbital_energies_target_np[:10]}")
        
        # Check if CHC is sorted (this is the key issue)
        is_chc_sorted = np.all(chc_target_flow[:-1] <= chc_target_flow[1:])
        if not is_chc_sorted:
            logger.warning(f"Index {debug_idx}: CHC (gt) is NOT sorted - this is the reported issue!")
            logger.warning(f"  First 10 CHC values: {chc_target_flow[:10]}")
            logger.warning(f"  First 10 orbital energies: {orbital_energies_target_np[:10]}")
            
            # Check if CHC matches energies when sorted
            chc_target_flow_sorted = np.sort(chc_target_flow)
            energies_target_sorted = np.sort(orbital_energies_target_np)
            
            if np.allclose(chc_target_flow_sorted, energies_target_sorted, atol=1e-5):
                logger.warning(f"Index {debug_idx}: CHC values match orbital energies but ORDERING is wrong!")
                logger.warning(f"  This suggests cut_coeff column order doesn't match eigenvalue order")
                
                # Find the permutation
                chc_argsort = np.argsort(chc_target_flow)
                energies_argsort = np.argsort(orbital_energies_target_np)
                
                if not np.array_equal(chc_argsort, energies_argsort):
                    logger.warning(f"Index {debug_idx}: Different sorting order detected!")
                    logger.warning(f"  CHC argsort (first 10): {chc_argsort[:10]}")
                    logger.warning(f"  Energies argsort (first 10): {energies_argsort[:10]}")
                    
                    mismatches = np.where(chc_argsort != energies_argsort)[0]
                    if len(mismatches) > 0:
                        logger.warning(f"Index {debug_idx}: Found {len(mismatches)} mismatches in ordering")
                        logger.warning(f"  First 5 mismatches at indices: {mismatches[:5]}")
                
                # Additional debugging: verify cut_coeff column order
                logger.info(f"Index {debug_idx}: Checking cut_coeff column order...")
                for mo_idx in range(min(5, cut_coeff_target.shape[2])):
                    cut_coeff_single = cut_coeff_target[:, :, mo_idx:mo_idx+1]
                    chc_single = compute_chc_from_blocks(
                        cut_coeff_single,
                        diag_target_work,
                        off_target_work,
                        local_edges,
                    )
                    chc_single_val = chc_single.item()
                    energy_val = orbital_energies_target_np[mo_idx]
                    logger.info(f"  MO {mo_idx}: CHC={chc_single_val:.6f}, Energy={energy_val:.6f}, Diff={abs(chc_single_val-energy_val):.2e}")
                
                return False
        
        # Verify CHC matches orbital energies
        if not _verify_chc_matches_energies(
            debug_idx,
            chc_target_flow,
            orbital_energies_target_np,
            np.sort(orbital_energies_target_np),  # stored comparison
            logger,
            label="flow_module-style CHC",
        ):
            return False
        
        # Compare with previous CHC calculation
        chc_prev_sorted = np.sort(chc_from_blocks)
        chc_flow_sorted = np.sort(chc_target_flow)
        
        if len(chc_prev_sorted) == len(chc_flow_sorted):
            max_diff_prev_flow = np.abs(chc_prev_sorted - chc_flow_sorted).max()
            if max_diff_prev_flow > 1e-5:
                logger.warning(f"Index {debug_idx}: Previous CHC calculation differs from flow_module-style CHC")
                logger.warning(f"  Max difference: {max_diff_prev_flow:.2e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Index {debug_idx}: Error during flow_module CHC debugging: {e}")
        import traceback
        traceback.print_exc()
        return False


def _check_transformed_chc_matches_original_eigenvalues(
    debug_idx: int,
    transformed_hamiltonian: torch.Tensor,
    transformed_overlap: torch.Tensor,
    transformed_coeffs: torch.Tensor,
    original_coeffs: torch.Tensor,
    original_hamiltonian: torch.Tensor,
    original_overlap: torch.Tensor,
    original_eigenvalues: np.ndarray,
    stored_eigenvalues: np.ndarray,
    atoms: torch.Tensor,
    dataset: BaseQH9Dataset,
    logger,
) -> bool:
    """
    Check if CHC calculated from transformed C and H matches original eigenvalues.
    
    This verifies that C_transformed^T H_transformed C_transformed gives the same
    eigenvalues as the original Hamiltonian.
    
    Two approaches are compared:
    1. C_transformed (calculated from transformed H/S) with transformed H
    2. C_original_transformed (original C transformed) with transformed H
    """
    import numpy as np
    import torch
    from common.matrix_transforms import (
        compute_chc_from_blocks,
        _cut_matrix_3d,
        cut_orbital_coefficients,
    )
    
    try:
        logger.info(f"Index {debug_idx}: Checking if transformed CHC matches original eigenvalues...")
        
        # First, verify that transformed H and S give the same eigenvalues as original
        # This should be true if transform is unitary
        logger.info(f"Index {debug_idx}: Step 1: Verifying transform preserves eigenvalues (full matrix)...")
        
        # Calculate CHC using full matrices for comparison
        # Method 1: Direct full matrix calculation C^T H C
        chc_full_original = torch.diag(
            torch.matmul(
                torch.matmul(original_coeffs.T, original_hamiltonian),
                original_coeffs
            )
        ).detach().cpu().numpy()
        
        chc_full_transformed = torch.diag(
            torch.matmul(
                torch.matmul(transformed_coeffs.T, transformed_hamiltonian),
                transformed_coeffs
            )
        ).detach().cpu().numpy()
        
        # These should match original eigenvalues
        chc_full_original_sorted = np.sort(chc_full_original)
        chc_full_transformed_sorted = np.sort(chc_full_transformed)
        original_sorted = np.sort(original_eigenvalues)
        
        max_diff_full_original = np.abs(chc_full_original_sorted - original_sorted).max()
        max_diff_full_transformed = np.abs(chc_full_transformed_sorted - original_sorted).max()
        
        logger.info(f"Index {debug_idx}: Full matrix CHC (original): max_diff from eigenvalues = {max_diff_full_original:.2e}")
        logger.info(f"Index {debug_idx}: Full matrix CHC (transformed): max_diff from eigenvalues = {max_diff_full_transformed:.2e}")
        
        if max_diff_full_original > 1e-5:
            logger.warning(f"Index {debug_idx}: Full matrix CHC (original) does NOT match eigenvalues!")
            logger.warning(f"  This suggests an issue with CHC calculation itself")
        
        if max_diff_full_transformed > 1e-5:
            logger.warning(f"Index {debug_idx}: Full matrix CHC (transformed) does NOT match eigenvalues!")
            logger.warning(f"  This suggests transform may not preserve eigenvalues correctly")
        
        # Now check block-wise calculation
        logger.info(f"Index {debug_idx}: Step 2: Verifying block-wise CHC calculation...")
        
        # First, verify block-wise CHC for original matrices
        logger.info(f"Index {debug_idx}: Step 2a: Verifying block-wise CHC for original matrices...")
        from common.matrix_transforms import _reconstruct_full_matrix
        
        # Cut original Hamiltonian into blocks for comparison
        original_hamiltonian_3d = original_hamiltonian.unsqueeze(0)  # [1, n_orb, n_orb]
        atoms_1d = atoms.view(-1) if atoms.dim() > 1 else atoms
        
        (
            concat_diagonal_blocks_orig,
            concat_non_diagonal_blocks_orig,
            _,
            _,
            edge_index_full_orig,
        ) = _cut_matrix_3d(
            original_hamiltonian_3d,
            atoms_1d,
            dataset.orbital_mask,
            dataset.full_orbitals,
        )
        
        # concat_diagonal_blocks_orig shape: [n_atoms, n_features, full_orbitals, full_orbitals]
        # We need [n_atoms, full_orbitals, full_orbitals] for the first feature
        diag_ham_original = concat_diagonal_blocks_orig[:, 0]  # [n_atoms, full_orbitals, full_orbitals]
        non_diag_ham_original = concat_non_diagonal_blocks_orig[:, 0]  # [n_edges, full_orbitals, full_orbitals]
        
        # Cut original coefficients
        cut_coeff_original = cut_orbital_coefficients(
            original_coeffs,
            atoms_1d,
            dataset.orbital_mask,
            dataset.full_orbitals,
        )
        
        # Calculate block-wise CHC for original
        local_edges_orig = edge_index_full_orig.clone()
        chc_original_block = compute_chc_from_blocks(
            cut_coeff_original,
            diag_ham_original.to(torch.float64),
            non_diag_ham_original.to(torch.float64),
            local_edges_orig,
        )
        chc_original_block = chc_original_block.detach().cpu().numpy()
        
        # Compare original block-wise CHC with full matrix CHC
        chc_original_block_sorted = np.sort(chc_original_block)
        max_diff_orig_block_vs_full = np.abs(chc_original_block_sorted - chc_full_original_sorted).max()
        logger.info(f"Index {debug_idx}: Block-wise CHC vs Full matrix CHC (original): max_diff = {max_diff_orig_block_vs_full:.2e}")
        
        if max_diff_orig_block_vs_full > 1e-5:
            logger.warning(f"Index {debug_idx}: Block-wise CHC (original) does NOT match full matrix CHC!")
            logger.warning(f"  This suggests a fundamental issue with block-wise CHC calculation")
            logger.warning(f"  First 10 block-wise CHC (original): {chc_original_block[:10]}")
            logger.warning(f"  First 10 full matrix CHC (original): {chc_full_original[:10]}")
        
        # Now check transformed block-wise calculation
        logger.info(f"Index {debug_idx}: Step 2b: Verifying block-wise CHC for transformed matrices...")
        
        # IMPORTANT: Check if transform commutes with block cutting
        # Method 1: Transform first, then cut into blocks (current approach in _build_data_object)
        logger.info(f"Index {debug_idx}: Method 1: Transform full matrix, then cut into blocks...")
        transformed_hamiltonian_3d = transformed_hamiltonian.unsqueeze(0)  # [1, n_orb, n_orb]
        
        (
            concat_diagonal_blocks,
            concat_non_diagonal_blocks,
            _,
            _,
            edge_index_full,
        ) = _cut_matrix_3d(
            transformed_hamiltonian_3d,
            atoms_1d,
            dataset.orbital_mask,
            dataset.full_orbitals,
        )
        
        # concat_diagonal_blocks shape: [n_atoms, n_features, full_orbitals, full_orbitals]
        # We need [n_atoms, full_orbitals, full_orbitals] for the first feature
        diag_ham_transformed = concat_diagonal_blocks[:, 0]  # [n_atoms, full_orbitals, full_orbitals]
        non_diag_ham_transformed = concat_non_diagonal_blocks[:, 0]  # [n_edges, full_orbitals, full_orbitals]
        
        # Method 2: Cut into blocks first, then transform each block (alternative approach)
        logger.info(f"Index {debug_idx}: Method 2: Cut into blocks first, then transform each block...")
        # Reconstruct full matrix from original blocks to verify
        # Note: _reconstruct_full_matrix expects orbital_masks as List[torch.Tensor]
        mask_list_orig = [dataset.orbital_mask[int(z.item())] for z in atoms_1d]
        # Convert to list of tensors if needed
        mask_list_orig_tensors = [mask if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.long) for mask in mask_list_orig]
        
        # Verify shapes before reconstruction
        logger.info(f"Index {debug_idx}: Shape verification for reconstruction:")
        logger.info(f"  diag_ham_original shape: {diag_ham_original.shape}")
        logger.info(f"  non_diag_ham_original shape: {non_diag_ham_original.shape}")
        logger.info(f"  Number of atoms: {len(atoms_1d)}")
        logger.info(f"  Number of masks: {len(mask_list_orig_tensors)}")
        logger.info(f"  atoms_1d: {atoms_1d}")
        logger.info(f"  concat_diagonal_blocks_orig shape: {concat_diagonal_blocks_orig.shape}")
        logger.info(f"  concat_non_diagonal_blocks_orig shape: {concat_non_diagonal_blocks_orig.shape}")
        
        if diag_ham_original.shape[0] != len(mask_list_orig_tensors):
            logger.error(f"Index {debug_idx}: Shape mismatch detected!")
            logger.error(f"  diag_ham_original.shape[0] = {diag_ham_original.shape[0]}")
            logger.error(f"  len(mask_list_orig_tensors) = {len(mask_list_orig_tensors)}")
            logger.error(f"  This suggests _cut_matrix_3d returned unexpected shape")
            logger.error(f"  Expected: [n_atom, full_orbitals, full_orbitals]")
            logger.error(f"  Got: {diag_ham_original.shape}")
            # Try to understand the issue - maybe we need to check the actual structure
            if len(diag_ham_original.shape) == 3:
                logger.error(f"  diag_ham_original has 3 dimensions, which is correct")
                logger.error(f"  But first dimension ({diag_ham_original.shape[0]}) doesn't match number of atoms ({len(mask_list_orig_tensors)})")
            # Skip this check if shapes don't match
            full_h_reconstructed = None
        else:
            try:
                full_h_reconstructed = _reconstruct_full_matrix(
                    diag_ham_original.to(torch.float64),
                    non_diag_ham_original.to(torch.float64),
                    local_edges_orig,
                    mask_list_orig_tensors,
                )
                logger.info(f"Index {debug_idx}: Successfully reconstructed full matrix from blocks")
                logger.info(f"  Reconstructed shape: {full_h_reconstructed.shape}")
                logger.info(f"  Original hamiltonian shape: {original_hamiltonian.shape}")
            except Exception as e:
                logger.error(f"Index {debug_idx}: Error during reconstruction: {e}")
                import traceback
                traceback.print_exc()
                full_h_reconstructed = None
        
        # Compare the two methods only if reconstruction succeeded
        if full_h_reconstructed is not None:
            # Transform the reconstructed full matrix
            full_h_reconstructed_3d = full_h_reconstructed.unsqueeze(0)  # [1, n_orb, n_orb]
            full_h_transformed_recon = dataset.matrix_transform(
                full_h_reconstructed_3d,
                atoms_1d,
                convention="pyscf_def2svp_to_e3nn",
            ).squeeze(0)
            
            # Cut the transformed reconstructed matrix
            (
                concat_diagonal_blocks_recon,
                concat_non_diagonal_blocks_recon,
                _,
                _,
                edge_index_full_recon,
            ) = _cut_matrix_3d(
                full_h_transformed_recon.unsqueeze(0),
                atoms_1d,
                dataset.orbital_mask,
                dataset.full_orbitals,
            )
            
            # concat_diagonal_blocks_recon shape: [n_atoms, n_features, full_orbitals, full_orbitals]
            diag_ham_transformed_recon = concat_diagonal_blocks_recon[:, 0]  # [n_atoms, full_orbitals, full_orbitals]
            non_diag_ham_transformed_recon = concat_non_diagonal_blocks_recon[:, 0]  # [n_edges, full_orbitals, full_orbitals]
            
            # Compare the two methods
            max_diff_diag = torch.abs(diag_ham_transformed - diag_ham_transformed_recon).max().item()
            max_diff_non_diag = torch.abs(non_diag_ham_transformed - non_diag_ham_transformed_recon).max().item()
            
            logger.info(f"Index {debug_idx}: Transform then cut vs Cut then transform (diagonal blocks): max_diff = {max_diff_diag:.2e}")
            logger.info(f"Index {debug_idx}: Transform then cut vs Cut then transform (non-diagonal blocks): max_diff = {max_diff_non_diag:.2e}")
            
            if max_diff_diag > 1e-5 or max_diff_non_diag > 1e-5:
                logger.warning(f"Index {debug_idx}: Transform does NOT commute with block cutting!")
                logger.warning(f"  This is the root cause - transform and block cutting order matters")
                logger.warning(f"  Diagonal blocks max diff: {max_diff_diag:.2e}")
                logger.warning(f"  Non-diagonal blocks max diff: {max_diff_non_diag:.2e}")
                logger.warning(f"  This means block-wise CHC calculation for transformed matrices may be incorrect")
        else:
            logger.warning(f"Index {debug_idx}: Skipping transform commute check due to shape mismatch")
        
        # Cut transformed coefficients into blocks
        cut_coeff_transformed = cut_orbital_coefficients(
            transformed_coeffs,
            atoms_1d,
            dataset.orbital_mask,
            dataset.full_orbitals,
        )  # [n_atom, full_orbitals, n_mo]
        
        # Calculate CHC from transformed C and H (block-wise)
        local_edges = edge_index_full.clone()
        chc_transformed = compute_chc_from_blocks(
            cut_coeff_transformed,
            diag_ham_transformed.to(torch.float64),
            non_diag_ham_transformed.to(torch.float64),
            local_edges,
        )
        chc_transformed = chc_transformed.detach().cpu().numpy()
        
        # Step 3: Inverse transform the transformed block-wise Hamiltonian back to original
        logger.info(f"Index {debug_idx}: Step 3: Inverse transforming block-wise Hamiltonian...")
        
        # Reconstruct full matrix from transformed blocks
        mask_list_transformed = [dataset.orbital_mask[int(z.item())] for z in atoms_1d]
        mask_list_transformed_tensors = [
            mask if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.long) 
            for mask in mask_list_transformed
        ]
        
        if diag_ham_transformed.shape[0] == len(mask_list_transformed_tensors):
            # Reconstruct full matrix from transformed blocks
            full_h_transformed_from_blocks = _reconstruct_full_matrix(
                diag_ham_transformed.to(torch.float64),
                non_diag_ham_transformed.to(torch.float64),
                local_edges,
                mask_list_transformed_tensors,
            )
            
            # First, verify that reconstructed transformed matrix matches the actual transformed matrix
            max_diff_recon_vs_actual = torch.abs(full_h_transformed_from_blocks - transformed_hamiltonian).max().item()
            logger.info(f"Index {debug_idx}: Reconstructed transformed matrix vs actual transformed matrix: max_diff = {max_diff_recon_vs_actual:.2e}")
            
            if max_diff_recon_vs_actual > 1e-5:
                logger.warning(f"Index {debug_idx}: Reconstructed transformed matrix does NOT match actual transformed matrix!")
                logger.warning(f"  This suggests an issue with block reconstruction")
            else:
                logger.info(f"Index {debug_idx}: Block reconstruction is correct!")
            
            # Apply inverse transform: Since transform is permutation + sign, we need to find inverse
            # Try to find inverse convention or apply transform twice to see if it's self-inverse
            from common.matrix_transforms import get_convention_dict
            convention_dict = get_convention_dict()
            
            # Try different inverse conventions
            inverse_convention = None
            inverse_conventions_to_try = [
                "e3nn_to_pyscf_def2svp",
                "pyscf_def2svp", 
                "back2pyscf",
            ]
            
            for inv_conv in inverse_conventions_to_try:
                if inv_conv in convention_dict:
                    logger.info(f"Index {debug_idx}: Trying inverse convention: {inv_conv}")
                    try:
                        full_h_inverse_transformed = dataset.matrix_transform(
                            full_h_transformed_from_blocks.unsqueeze(0),
                            atoms_1d,
                            convention=inv_conv,
                        ).squeeze(0)
                        
                        # Check if this restores the original matrix
                        max_diff_inv_vs_orig = torch.abs(full_h_inverse_transformed - original_hamiltonian).max().item()
                        logger.info(f"Index {debug_idx}: Inverse transform ({inv_conv}) vs original: max_diff = {max_diff_inv_vs_orig:.2e}")
                        
                        if max_diff_inv_vs_orig < 1e-5:
                            inverse_convention = inv_conv
                            logger.info(f"Index {debug_idx}: Found working inverse convention: {inv_conv}")
                            break
                    except Exception as e:
                        logger.debug(f"Index {debug_idx}: Failed to try {inv_conv}: {e}")
                        continue
            
            if inverse_convention is not None:
                # We already computed full_h_inverse_transformed above
                
                # Cut the inverse transformed matrix into blocks
                (
                    concat_diagonal_blocks_inv,
                    concat_non_diagonal_blocks_inv,
                    _,
                    _,
                    edge_index_full_inv,
                ) = _cut_matrix_3d(
                    full_h_inverse_transformed.unsqueeze(0),
                    atoms_1d,
                    dataset.orbital_mask,
                    dataset.full_orbitals,
                )
                
                # concat_diagonal_blocks_inv shape: [n_atoms, n_features, full_orbitals, full_orbitals]
                diag_ham_inverse = concat_diagonal_blocks_inv[:, 0]  # [n_atoms, full_orbitals, full_orbitals]
                non_diag_ham_inverse = concat_non_diagonal_blocks_inv[:, 0]  # [n_edges, full_orbitals, full_orbitals]
                
                # Compare with original blocks
                max_diff_diag_inv = torch.abs(diag_ham_inverse - diag_ham_original.to(torch.float64)).max().item()
                max_diff_non_diag_inv = torch.abs(non_diag_ham_inverse - non_diag_ham_original.to(torch.float64)).max().item()
                
                logger.info(f"Index {debug_idx}: Inverse transformed blocks vs original blocks (diagonal): max_diff = {max_diff_diag_inv:.2e}")
                logger.info(f"Index {debug_idx}: Inverse transformed blocks vs original blocks (non-diagonal): max_diff = {max_diff_non_diag_inv:.2e}")
                
                if max_diff_diag_inv > 1e-5 or max_diff_non_diag_inv > 1e-5:
                    logger.warning(f"Index {debug_idx}: Inverse transform does NOT fully restore original blocks!")
                    logger.warning(f"  Diagonal blocks max diff: {max_diff_diag_inv:.2e}")
                    logger.warning(f"  Non-diagonal blocks max diff: {max_diff_non_diag_inv:.2e}")
                else:
                    logger.info(f"Index {debug_idx}: Inverse transform successfully restores original blocks!")
                
                # Calculate CHC using inverse transformed blocks with original coefficients
                # Use original coefficients (not transformed) since we're back to original space
                chc_inverse = compute_chc_from_blocks(
                    cut_coeff_original,  # Use original coefficients
                    diag_ham_inverse.to(torch.float64),
                    non_diag_ham_inverse.to(torch.float64),
                    edge_index_full_inv.clone(),
                )
                chc_inverse = chc_inverse.detach().cpu().numpy()
                
                # Compare inverse transformed CHC with original CHC
                chc_inverse_sorted = np.sort(chc_inverse)
                max_diff_inverse_vs_original = np.abs(chc_inverse_sorted - chc_original_block_sorted).max()
                logger.info(f"Index {debug_idx}: CHC from inverse transformed blocks vs original block-wise CHC: max_diff = {max_diff_inverse_vs_original:.2e}")
                
                if max_diff_inverse_vs_original > 1e-5:
                    logger.warning(f"Index {debug_idx}: CHC from inverse transformed blocks does NOT match original block-wise CHC!")
                    logger.warning(f"  This suggests the issue is with block-wise CHC calculation itself")
                    logger.warning(f"  First 10 CHC (inverse transformed): {chc_inverse[:10]}")
                    logger.warning(f"  First 10 CHC (original block-wise): {chc_original_block[:10]}")
                else:
                    logger.info(f"Index {debug_idx}: CHC from inverse transformed blocks matches original block-wise CHC!")
                    logger.info(f"  This confirms that the issue is with transform, not block-wise CHC calculation")
            else:
                logger.warning(f"Index {debug_idx}: Could not find inverse transform convention, skipping inverse transform check")
        else:
            logger.warning(f"Index {debug_idx}: Shape mismatch for inverse transform check, skipping")
        
        # Compare block-wise CHC with full matrix CHC (transformed)
        chc_transformed_sorted = np.sort(chc_transformed)
        max_diff_block_vs_full = np.abs(chc_transformed_sorted - chc_full_transformed_sorted).max()
        logger.info(f"Index {debug_idx}: Block-wise CHC vs Full matrix CHC (transformed): max_diff = {max_diff_block_vs_full:.2e}")
        
        if max_diff_block_vs_full > 1e-5:
            logger.warning(f"Index {debug_idx}: Block-wise CHC (transformed) does NOT match full matrix CHC!")
            logger.warning(f"  This suggests an issue with block-wise CHC calculation for transformed matrices")
            logger.warning(f"  First 10 block-wise CHC (transformed): {chc_transformed[:10]}")
            logger.warning(f"  First 10 full matrix CHC (transformed): {chc_full_transformed[:10]}")
        
        # Compare transformed block-wise CHC with original block-wise CHC
        max_diff_transformed_vs_original_block = np.abs(chc_transformed_sorted - chc_original_block_sorted).max()
        logger.info(f"Index {debug_idx}: Block-wise CHC (transformed) vs Block-wise CHC (original): max_diff = {max_diff_transformed_vs_original_block:.2e}")
        
        if max_diff_transformed_vs_original_block > 1e-5:
            logger.warning(f"Index {debug_idx}: Block-wise CHC (transformed) does NOT match block-wise CHC (original)!")
            logger.warning(f"  This suggests transform affects block-wise CHC calculation")
            logger.warning(f"  First 10 block-wise CHC (transformed): {chc_transformed[:10]}")
            logger.warning(f"  First 10 block-wise CHC (original): {chc_original_block[:10]}")
        
        # Compare block-wise CHC with original eigenvalues
        original_sorted = np.sort(original_eigenvalues)
        chc_transformed_sorted = np.sort(chc_transformed)
        
        if len(original_sorted) != len(chc_transformed_sorted):
            logger.error(f"Index {debug_idx}: Mismatch in transformed CHC/original eigenvalue count: CHC={len(chc_transformed_sorted)}, original={len(original_sorted)}")
            return False
        
        max_diff = np.abs(original_sorted - chc_transformed_sorted).max()
        mean_diff = np.abs(original_sorted - chc_transformed_sorted).mean()
        
        if max_diff > 1e-5:
            logger.warning(f"Index {debug_idx}: Block-wise transformed CHC does NOT match original eigenvalues!")
            logger.warning(f"  Max difference: {max_diff:.2e}, Mean difference: {mean_diff:.2e}")
            logger.warning(f"  First 10 original eigenvalues: {original_eigenvalues[:10]}")
            logger.warning(f"  First 10 transformed CHC (block-wise): {chc_transformed[:10]}")
            logger.warning(f"  First 10 transformed CHC (full matrix): {chc_full_transformed[:10]}")
            
            # Additional debugging: check if the issue is with block-wise calculation
            if max_diff_block_vs_full > 1e-5:
                logger.warning(f"Index {debug_idx}: The issue is likely in block-wise CHC calculation")
                logger.warning(f"  Block-wise and full matrix CHC differ by {max_diff_block_vs_full:.2e}")
            else:
                logger.warning(f"Index {debug_idx}: Block-wise CHC matches full matrix CHC, but both differ from eigenvalues")
                logger.warning(f"  This suggests the issue is with transform or C calculation")
            
            return False
        
        # Check ordering
        if not np.allclose(chc_transformed, original_eigenvalues, atol=1e-5):
            if np.allclose(chc_transformed_sorted, original_sorted, atol=1e-5):
                logger.warning(f"Index {debug_idx}: Transformed CHC values match original eigenvalues but ORDERING differs!")
                logger.warning(f"  This suggests transformed C column order doesn't match eigenvalue order")
                
                # Find the permutation
                chc_argsort = np.argsort(chc_transformed)
                original_argsort = np.argsort(original_eigenvalues)
                
                if not np.array_equal(chc_argsort, original_argsort):
                    logger.warning(f"Index {debug_idx}: Different sorting order detected!")
                    logger.warning(f"  Transformed CHC argsort (first 10): {chc_argsort[:10]}")
                    logger.warning(f"  Original eigenvalues argsort (first 10): {original_argsort[:10]}")
                    
                    mismatches = np.where(chc_argsort != original_argsort)[0]
                    if len(mismatches) > 0:
                        logger.warning(f"Index {debug_idx}: Found {len(mismatches)} mismatches in ordering")
                        logger.warning(f"  First 5 mismatches at indices: {mismatches[:5]}")
                
                return False
            else:
                logger.error(f"Index {debug_idx}: Transformed CHC values do not match original eigenvalues even when sorted")
                return False
        else:
            logger.debug(f"Index {debug_idx}: Transformed CHC correctly matches original eigenvalues in order (max_diff={max_diff:.2e})")
        
        # Also compare with stored eigenvalues
        stored_sorted = np.sort(stored_eigenvalues)
        if len(chc_transformed) == len(stored_eigenvalues):
            chc_stored_diff = np.abs(chc_transformed_sorted - stored_sorted).max()
            if chc_stored_diff > 1e-5:
                logger.warning(f"Index {debug_idx}: Transformed CHC differs from stored eigenvalues")
                logger.warning(f"  Max difference: {chc_stored_diff:.2e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Index {debug_idx}: Error during transformed CHC check: {e}")
        import traceback
        traceback.print_exc()
        return False


__all__ = [
    "BaseQH9Dataset",
    "DFTArtifacts",
    "compute_common_artifacts",
    "debug_eigenvalue_ordering_and_chc",
]
