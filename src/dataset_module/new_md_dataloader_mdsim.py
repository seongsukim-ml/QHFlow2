#!/usr/bin/env python3
"""
In-memory loader for HDF5 MD17 data that mimics the behaviour of the existing
LMDB-based `MD17_DFT_Shard` dataset.  Each sample is returned as an `AOData`
object with the same fields so that downstream code can use the two datasets
interchangeably.
"""

from __future__ import annotations

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from typing import Dict

import h5py
import numpy as np
import torch
# file system
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset

from common.matrix_transforms import get_convention_dict, _matrix_transform_single
from common.units import ANG2BOHR, eV2HA
from ase.data import chemical_symbols
from utils import (
    AOData,
    Onsite_3idx_Overlap_Integral,
    build_AO_index,
    build_molecule,
    construct_orbital_l_index,
)
from pyscf import gto, scf, dft
import pickle
from pathlib import Path
from scipy.linalg import block_diag



def _to_tensor(data) -> torch.Tensor:
    """Utility converting numpy arrays to float64 torch tensors on CPU."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().double()
    arr = np.asarray(data)
    dtype = torch.float64 if arr.dtype.kind == "f" else torch.int64
    return torch.from_numpy(arr).to(dtype)


class MD17_DFT_MDSim(Dataset):
    """
    In-memory loader for HDF5 MD17 data that mimics the behaviour of the existing
    LMDB-based `MD17_DFT_Shard` dataset.

    Each sample is returned as an `AOData` object with the same fields so that
    downstream code can use the two datasets interchangeably. Loads all splits
    (train, val, test) by default.

    Parameters
    ----------
    root:
        Root directory containing the dataset folder.
    name:
        Dataset name (e.g., "water", "aspirin").
    prefix:
        Prefix for the dataset folder name (default: "md-").
    postfix:
        Postfix for the dataset folder name (default: "").
    load_orbitals:
        If ``True`` also include orbital energies / coefficients.
    include_density:
        Attach ``density_matrix`` and ``h_core`` tensors (default ``False``).
    all_features:
        Mimic ``MD17_DFT_Shard(..., all_features=True)`` by also exposing
        ``energy``/``force`` (Hartree / Hartree·Bohr⁻¹) and additional metadata.
    """

    def __init__(
        self,
        root: str,
        name: str,
        prefix: str = "",
        postfix: str = "",
        load_orbitals: bool = False,
        include_density: bool = False,
        all_features: bool = False,
        use_ref_hamiltonian_as_init_ham=False,
    ) -> None:
        super().__init__()

        self.name = name
        assert name in ["mdsim-aspirin", "mdsim-ethanol", "mdsim-naphthalene", "mdsim-salicylic_acid"], f"Dataset name {name} not supported"
        self.prefix = "" if prefix is None else str(prefix)
        self.postfix = "" if postfix is None else str(postfix)
        self.folder = os.path.join(root, self.prefix + self.name + self.postfix)
        self.h5_file = os.path.join(self.folder, "data.h5")

        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file}")

        self.load_orbitals = load_orbitals
        self.include_density = include_density or all_features
        self.all_features = all_features

        # Splits are stored as groups containing samples
        with h5py.File(self.h5_file, "r") as f:
            self.metadata = dict(f["metadata"].attrs.items())
            splits = f["splits"]
            # Get sample keys from each split group
            train_keys = sorted(splits["train"].keys(), key=lambda x: int(x.split("_")[1]))
            val_keys = sorted(splits["val"].keys(), key=lambda x: int(x.split("_")[1]))
            test_keys = sorted(splits["test"].keys(), key=lambda x: int(x.split("_")[1]))
        
        # Store split sizes for external use
        self.num_train = len(train_keys)
        self.num_val = len(val_keys)
        self.num_test = len(test_keys)
        
        # Store split paths: (split_name, sample_key)
        self.sample_paths = []
        self._train_indices = []
        self._val_indices = []
        self._test_indices = []
        
        idx = 0
        for key in train_keys:
            self.sample_paths.append(("train", key))
            self._train_indices.append(idx)
            idx += 1
        for key in val_keys:
            self.sample_paths.append(("val", key))
            self._val_indices.append(idx)
            idx += 1
        for key in test_keys:
            self.sample_paths.append(("test", key))
            self._test_indices.append(idx)
            idx += 1
            
        # Indices for backward compatibility
        self.indices = list(range(len(self.sample_paths)))

        # Orbital reference information for different atomic numbers
        self.full_orbitals = 14
        self.orbital_mask = {}
        self.orbitals_ref = {}
        self.orbitals_ref[1] = np.array([0, 0, 1])  # H: 2s 1p
        self.orbitals_ref[6] = np.array([0, 0, 0, 1, 1, 2])  # C: 3s 2p 1d
        self.orbitals_ref[7] = np.array([0, 0, 0, 1, 1, 2])  # N: 3s 2p 1d
        self.orbitals_ref[8] = np.array([0, 0, 0, 1, 1, 2])  # O: 3s 2p 1d

        if self.name == "mdsim-aspirin":
            self.atoms = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]
            self.atom_list = ["C", "O", "H"]
            self.hamiltonian_size = 222
        elif self.name == "mdsim-ethanol":
            self.atoms = [6, 6, 8, 1, 1, 1, 1, 1, 1]
            self.atom_list = ["C", "O", "H"]
            self.hamiltonian_size = 72
        elif self.name == "mdsim-naphthalene":
            self.atoms = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]
            self.atom_list = ["C", "H"]
            self.hamiltonian_size = 180
        elif self.name == "mdsim-salicylic_acid":
            self.atoms = [6, 6, 6, 8, 6, 6, 6, 6, 8, 8, 1, 1, 1, 1, 1, 1]
            self.atom_list = ["C", "N", "O", "H"]
            self.hamiltonian_size = 170
        else:
            raise NotImplementedError(f"Dataset name {self.name} not supported")

        self._prepare_structure_constants()
        self._cache: Dict[int, AOData] = {}

        self.atom_num = {
            "H": 1,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
        }
        self._ref_hamiltonian_dict = {} # {basis: {xc: {atomic_number: init_ham}}}
        self.use_ref_hamiltonian_as_init_ham = use_ref_hamiltonian_as_init_ham
        print(f"use_ref_hamiltonian_as_init_ham: {self.use_ref_hamiltonian_as_init_ham}")
        if self.use_ref_hamiltonian_as_init_ham:
            self._processed_path = self.folder
            self.dft_conf_tuple = [("def2-svp", "pbe", "minao")]
            self.basis = "def2-svp"
            self.xc = "pbe"
            self._load_ref_hamiltonian()
            self.atom_nums = self.atoms
            self._ref_hamiltonian = torch.tensor(self._build_ref_hamiltonian(self.atom_nums), dtype=torch.float64)
            self._ref_hamiltonian = self._matrix_transform(self._ref_hamiltonian, torch.tensor(self.atom_nums, dtype=torch.int64), convention="pyscf_def2svp").unsqueeze(0)



    # ------------------------------------------------------------------ utils
    def _prepare_structure_constants(self) -> None:
        """Derive molecule-level helpers from the first sample."""
        with h5py.File(self.h5_file, "r") as f:
            split_name, sample_key = self.sample_paths[0]
            sample = f[f"splits/{split_name}/{sample_key}"]
            atomic_numbers = sample["atomic_numbers"][:].astype(int)
            self.h_dim = int(sample.attrs["nbasis"])

        self.atomic_numbers = atomic_numbers.tolist()
        self.atom_tensor = torch.tensor(self.atomic_numbers, dtype=torch.int64)
        self.basis = self.metadata.get("basis", "def2-svp")
        self.functional = self.metadata.get("functional", "pbe")
        self.convention_dict = get_convention_dict()

        orbitals = []
        for Z in self.atoms:
            orbitals.append(tuple((int(Z), int(l)) for l in self.orbitals_ref[Z]))
        self.orbitals = tuple(orbitals)

        # Pre-compute Q table (same logic as MD17_DFT_Shard.setup_Q)
        self.atom_symbols = sorted({chemical_symbols[int(z)] for z in self.atomic_numbers})
        onsite = Onsite_3idx_Overlap_Integral(atom_list=self.atom_symbols, basis=self.basis)
        self.Q_dict = onsite.Q_table()

        blocks = []
        for l in range(60):
            parts = [self.Q_dict[int(Z)][l] for Z in self.atomic_numbers]
            blocks.append(torch.block_diag(*parts))
        Q = torch.stack(blocks)  # [60, h_dim, h_dim]
        Q = self._matrix_transform(Q, self.atom_tensor, "pyscf_def2svp_to_e3nn").permute(1, 2, 0)
        if Q.shape[-1] >= 40:
            Q[:, :, 16:40] = (
                Q[:, :, 16:40]
                .reshape(self.h_dim, self.h_dim, -1, 3)[:, :, :, [1, 2, 0]]
                .reshape(self.h_dim, self.h_dim, 24)
            )
        self.Q = Q.double()



    def _load_ref_hamiltonian(self):
        if os.path.exists(os.path.join(self._processed_path, "ref_hamiltonian.pkl")):
            self._ref_hamiltonian_dict = pickle.load(open(os.path.join(self._processed_path, "ref_hamiltonian.pkl"), "rb"))
        else:        
            for basis, xc, init_guess in self.dft_conf_tuple:
                for _atoms in self.atom_num.values():
                    self._ref_hamiltonian_dict[basis][xc][_atoms] = self._get_ref_hamiltonian(_atoms, basis, xc, init_guess)
            pickle.dump(self._ref_hamiltonian_dict, open(os.path.join(self._processed_path, "ref_hamiltonian.pkl"), "wb"))        

    def _get_ref_hamiltonian(self, atoms, basis="def2-svp", xc="pbe", init_guess="minao"):
        if basis not in self.basis:
            raise ValueError(f"Basis {basis} is not supported")
        if xc not in self.xc:
            raise ValueError(f"XC functional {xc} is not supported")
        
        if basis not in self._ref_hamiltonian_dict.keys():
            self._ref_hamiltonian_dict[basis] = {}
        if xc not in self._ref_hamiltonian_dict[basis].keys():
            self._ref_hamiltonian_dict[basis][xc] = {}

        if type(atoms) == str:
            _atoms = self.atom_num[atoms]
        else:
            _atoms = atoms

        if _atoms in self._ref_hamiltonian_dict[basis][xc].keys():
            return self._ref_hamiltonian_dict[basis][xc][_atoms]
        else:
            mol_conf = [[_atoms, [0,0,0]]]
            spin = 1 if _atoms % 2 else 0

            mol = gto.Mole()
            mol.build(verbose=0, atom=mol_conf, basis=basis, unit="bohr", spin=spin)
            mf = dft.RKS(mol)
            mf.xc = xc
            init_dm = mf.get_init_guess(key=init_guess) 
            init_ham = mf.get_fock(dm=init_dm)
            self._ref_hamiltonian_dict[basis][xc][_atoms] = init_ham
            return init_ham

    def _build_ref_hamiltonian(self, atoms, basis="def2-svp", xc="pbe", init_guess="minao"):
        init_hams = []
        orb_lens = []
        for atom in atoms:
            if type(atom) == str:
                atomic_number = self.atom_num[atom]
            else:
                atomic_number = atom
            # init_hams.append(self._get_ref_hamiltonian(atoms, basis, xc, init_guess))
            init_hams.append(self._ref_hamiltonian_dict[basis][xc][atomic_number])
            orb_len = 5 if atomic_number == 1 else 14
            orb_lens.append(orb_len)
        
        _ref_hamiltonian = block_diag(*init_hams).astype(np.float64)
            
        # _ref_hamiltonian = np.zeros((sum(orb_lens), sum(orb_lens)))
        # cur_idx = 0
        # for i, (ham, orb_len) in enumerate(zip(init_hams, orb_lens)):
        #     slice_idx = slice(cur_idx, cur_idx + orb_len)
        #     _ref_hamiltonian[slice_idx, slice_idx] = ham
        #     cur_idx += orb_len
        return _ref_hamiltonian

    def _matrix_transform(self, matrices: torch.Tensor, atoms: torch.Tensor, convention: str) -> torch.Tensor:
        return _matrix_transform_single(matrices, atoms, self.convention_dict[convention])

    # ----------------------------------------------------------------- dataset
    def __len__(self) -> int:
        return len(self.sample_paths)

    def get_split_indices(self):
        """Return train, val, test indices as stored in the HDF5 file.
        
        Returns:
            tuple: (train_indices, val_indices, test_indices)
        """
        return self._train_indices, self._val_indices, self._test_indices
    
    def get_split_sizes(self):
        """Return train, val, test sizes as stored in the HDF5 file.
        
        Returns:
            tuple: (num_train, num_val, num_test)
        """
        return self.num_train, self.num_val, self.num_test

    def __getitem__(self, idx: int) -> AOData:
        if idx >= len(self.sample_paths):
            raise IndexError(idx)
        if idx in self._cache:
            return self._cache[idx]

        split_name, sample_key = self.sample_paths[idx]
        with h5py.File(self.h5_file, "r") as f:
            sample = f[f"splits/{split_name}/{sample_key}"]
            aodata = self._build_sample(sample, dataset_idx=idx, file_idx=idx)
            self._cache[idx] = aodata
            return aodata

    # ----------------------------------------------------------------- helpers
    def _build_sample(self, sample: h5py.Group, *, dataset_idx: int, file_idx: int) -> AOData:
        pos = _to_tensor(sample["positions"][:])
        atoms = _to_tensor(sample["atomic_numbers"][:]).view(-1, 1)
        energy_ev = sample.attrs["energy_ev"]
        forces_ev = _to_tensor(sample["forces_ev_ang"][:])
        energy_ha = sample.attrs["energy_ha"]
        forces_ha_bohr = _to_tensor(sample["forces_ha_bohr"][:])
        energy_ref_ev = sample.attrs["energy_ref_ev"]
        forces_ref_ev = _to_tensor(sample["forces_ref_ev"][:]) # eV/Angstrom

        energy = torch.tensor([[energy_ev]], dtype=torch.float64)
        forces = forces_ev

        overlap = _to_tensor(sample["overlap"][:]).reshape(self.h_dim, self.h_dim)
        hamiltonian = _to_tensor(sample["hamiltonian"][:]).reshape(self.h_dim, self.h_dim)
        # Try different initial hamiltonian keys (minao or 1e or fallback)
        if "initial_hamiltonian_minao" in sample:
            init_ham = _to_tensor(sample["initial_hamiltonian_minao"][:]).reshape(self.h_dim, self.h_dim)
        elif "initial_hamiltonian_1e" in sample:
            init_ham = _to_tensor(sample["initial_hamiltonian_1e"][:]).reshape(self.h_dim, self.h_dim)
        elif "initial_hamiltonian" in sample:
            init_ham = _to_tensor(sample["initial_hamiltonian"][:]).reshape(self.h_dim, self.h_dim)
        else:
            # Fallback to zeros
            init_ham = torch.zeros(self.h_dim, self.h_dim, dtype=torch.float64)

        overlap = self._matrix_transform(overlap, atoms, "pyscf_def2svp_to_e3nn")
        hamiltonian = self._matrix_transform(hamiltonian, atoms, "pyscf_def2svp_to_e3nn")
        init_ham = self._matrix_transform(init_ham, atoms, "pyscf_def2svp_to_e3nn")

        if self.use_ref_hamiltonian_as_init_ham:
            init_ham = self._ref_hamiltonian

        mol_spec = build_molecule(self.atomic_numbers, pos.detach().cpu().numpy())
        AO_index = build_AO_index(mol_spec, self.basis)
        AO_l_index = construct_orbital_l_index(AO_index[1])
        AO_l_index_len = torch.tensor([[AO_l_index.numel()]], dtype=torch.int64)

        connections = [[i, j] for i in range(pos.shape[0]) for j in range(pos.shape[0]) if i != j]
        full_edge_index = (
            torch.tensor(connections, dtype=torch.int64).t().contiguous()
            if connections
            else torch.zeros((2, 0), dtype=torch.int64)
        )

        data = AOData(
            pos=pos,
            atoms=atoms,
            dft_energy=energy,
            dft_forces=forces,
            energy_ref=energy_ref_ev,
            forces_ref=forces_ref_ev,
            hamiltonian=hamiltonian.view(1, self.h_dim, self.h_dim),
            overlap=overlap.view(1, self.h_dim, self.h_dim),
            init_ham=init_ham.view(1, self.h_dim, self.h_dim),
            AO_index=AO_index,
            AO_l_index=AO_l_index,
            AO_l_index_len=AO_l_index_len,
            num_atoms=torch.tensor([[pos.shape[0]]], dtype=torch.int64),
            Q=self.Q,
            h_dim=torch.tensor([[self.h_dim]], dtype=torch.int64),
            full_edge_index=full_edge_index,
        )

        # Optional extras -----------------------------------------------------
        if self.include_density:
            density = _to_tensor(sample["density_matrix"][:]).reshape(self.h_dim, self.h_dim)
            h_core = _to_tensor(sample["h_core"][:]).reshape(self.h_dim, self.h_dim)
            data.density_matrix = density.view(1, self.h_dim, self.h_dim)
            data.h_core = h_core.view(1, self.h_dim, self.h_dim)

        if self.load_orbitals:
            data.orbital_energies = _to_tensor(sample["orbital_energies"][:]).view(1, -1)
            data.orbital_coefficients = _to_tensor(sample["orbital_coefficients"][:])

        if self.all_features:
            data.energy_ev = torch.tensor([[energy_ev]], dtype=torch.float64)
            data.force_ev = forces_ev
            sample_index = sample.attrs.get("sample_idx", file_idx)
            data.sample_idx = torch.tensor([int(sample_index)], dtype=torch.long)
            data.converged = torch.tensor([bool(sample.attrs.get("converged", True))])
            data.energy = data.dft_energy
            data.force = data.dft_forces

        return data


if __name__ == "__main__":
    # get arguments
    import argparse

    parser = argparse.ArgumentParser(description="RMD17_DFT Dataset Testing")
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset"),
        help="Root directory containing the dataset folder",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="mdsim-salicylic_acid",
        choices=["mdsim-aspirin", "mdsim-ethanol", "mdsim-naphthalene", "mdsim-salicylic_acid"],
        help="Dataset name",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for the dataset folder name",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default="",
        help="Postfix for the dataset folder name",
    )
    parser.add_argument(
        "--load_orbitals",
        action="store_true",
        help="Include orbital energies and coefficients",
    )
    parser.add_argument(
        "--include_density",
        action="store_true",
        help="Include density_matrix and h_core tensors",
    )
    parser.add_argument(
        "--all_features",
        action="store_true",
        help="Include all features (mimics MD17_DFT_Shard with all_features=True)",
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        help="Enable pdb debugging",
    )

    args = parser.parse_args()

    # Set thread environment variables
    torch.set_num_threads(4)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    # Create dataset
    dataset = MD17_DFT_MDSim(
        root=args.root,
        name=args.name,
        prefix=args.prefix,
        postfix=args.postfix,
        load_orbitals=args.load_orbitals,
        include_density=args.include_density,
        all_features=args.all_features,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"First sample:\n{dataset[0]}")
    if len(dataset) > 0:
        print(f"Last sample:\n{dataset[-1]}")

    if args.pdb:
        import pdb

        pdb.set_trace()

    print("Finished")
