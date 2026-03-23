try:
    import gpu4pyscf
    from gpu4pyscf.lib.cupy_helper import (tag_array, pack_tril, asarray)
    IMPORT_GPU4PYSCF = True
    print("gpu4pyscf is installed.")
except:
    IMPORT_GPU4PYSCF = False
    print("gpu4pyscf is not installed. Please install it using 'pip install gpu4pyscf'.")

from pathlib import Path
import copy
import logging
import os
import sys
import yaml
import torch
import json
import types
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from qhflow2.common.matrix_transforms import _cut_matrix_3d
from typing import Any, Dict, List, Sequence

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory

import sys
sys.path.append("/root/limlab01/kaistai/25DFT/QHFlow/src")
from qhflow2.dft_process.dft_process_utils import *
import torch
from torch_geometric.loader import DataLoader
from qhflow2.common.units import (
    BOHR2ANG, ANG2BOHR, HA2eV, HA_BOHR_2_eV_ANG,
    normalize_length_unit, convert_length,
    get_energy_unit_conversion_factor, get_force_unit_conversion_factor
)
import time

# %%
from qhflow2.common.draw_util import *

# %%
from qhflow2.pl_module import get_pl_model
from pyscf import dft, gto
from torch_geometric.data import Data, Batch

USE_GPU4PYSCF = True

def ase_atoms_to_atoms_coords(atoms):
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    return atomic_numbers, positions

def QHData_to_atoms(QHData, unit="ang"):
    numbers = QHData.atoms.squeeze()
    if isinstance(numbers, torch.Tensor):
        numbers = numbers.cpu().numpy()
    positions = QHData.pos.squeeze()
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    positions = convert_length(positions, from_unit=unit, to_unit="ang")
    cell = np.eye(3)
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=[False, False, False])        
    return atoms

def batch_to_atoms(batch):
    raise NotImplementedError("batch_to_atoms is not implemented")

class RKSCalculator(Calculator):
    implemented_properties = ["energy", "forces"]
    
    def __init__(self, basis="def2-SVP", functional="pbe", max_cycle=50, density_fit=True, energy_unit="ev", force_unit="ev/ang", level_shift=None):
        Calculator.__init__(self)
        self.basis = basis
        self.functional = functional
        self.gpu4pyscf = USE_GPU4PYSCF
        self.max_cycle = max_cycle
        self._reset_time_dict()
        self.density_fit = density_fit
        self.energy_unit_str = energy_unit
        self.energy_unit = get_energy_unit_conversion_factor(energy_unit)
        
        self.force_unit_str = force_unit
        self.force_unit = get_force_unit_conversion_factor(force_unit)
        self.level_shift = level_shift

    def _reset_time_dict(self):
        self.time_dict = {
            "mf_time": 0,
            "energy_time": 0,
            "grad_frame_time": 0,
            "init_ham_time": 0,
        }

    @staticmethod
    def ase_atoms_to_mol(ase_atoms, basis="def2-SVP", unit="ang", spin=0):
        atoms, coords = ase_atoms_to_atoms_coords(ase_atoms)
        return RKSCalculator._mol(atoms, coords, basis, unit, spin)

    @staticmethod
    def _mol(atoms, coords, basis, unit="ang", spin=0):
        unit = normalize_length_unit(unit)
        mol = gto.Mole()
        mol_conf = [
            [atoms[atom_idx], coords[atom_idx]] 
            for atom_idx in range(len(atoms))
        ]
        mol.build(verbose=0, atom=mol_conf, basis=basis, unit=unit, spin=spin)
        return mol

    @staticmethod
    def RKS(mol, functional="pbe", basis="def2-SVP", gpu4pyscf=USE_GPU4PYSCF, max_cycle=50, density_fit=True, level_shift=None):
        mf = dft.RKS(mol)
        if density_fit:
            mf = mf.density_fit()
        if gpu4pyscf:
            mf = mf.to_gpu()
        mf.xc = functional
        mf.basis = basis
        mf.max_cycle = max_cycle
        if level_shift is not None:
            mf.level_shift = level_shift
        return mf

    @staticmethod
    def UKS(mol, functional="pbe", basis="def2-SVP", gpu4pyscf=USE_GPU4PYSCF, max_cycle=50, density_fit=True, level_shift=None):
        mf = dft.UKS(mol)
        if density_fit:
            mf = mf.density_fit()
        if gpu4pyscf:
            mf = mf.to_gpu()
        mf.xc = functional
        mf.basis = basis
        mf.max_cycle = max_cycle
        if level_shift is not None:
            mf.level_shift = level_shift
        return mf

    def get_mol(self, atoms, coords, unit="ang", spin=0):
        return self._mol(atoms, coords, self.basis, unit, spin)

    def calculate(self, ase_atoms, properties, system_changes, units="ang", mf=None):
        Calculator.calculate(self, ase_atoms, properties, system_changes)
        # properties, system_changes is not used
        units = normalize_length_unit(units)
        atoms, coords = ase_atoms_to_atoms_coords(ase_atoms)
        self._reset_time_dict()
        start_time = time.time()
        if mf is None:
            mol = self.get_mol(atoms, coords, units)
            self.mf = self.RKS(mol, self.functional, self.basis, gpu4pyscf=self.gpu4pyscf, max_cycle=self.max_cycle, density_fit=self.density_fit, level_shift=self.level_shift)
        else:
            self.mf = mf
        self.time_dict["mf_time"] = time.time() - start_time

        start_time = time.time()
        self.init_dm = self.mf.get_init_guess()
        self.init_ham = self.mf.get_fock(dm=self.init_dm)
        self.time_dict["init_ham_time"] = time.time() - start_time

        start_time = time.time()
        energy = self.mf.kernel(self.init_dm)
        self.time_dict["energy_time"] = time.time() - start_time
        # unconverged
        if not self.mf.converged:
            logging.warning(f"Unconverged energy: {energy}, cycles: {self.mf.cycles}")
        if "forces" in properties:
            start_time = time.time()
            forces = -self.mf.nuc_grad_method().kernel()
            self.time_dict["grad_frame_time"] = time.time() - start_time
        else:
            forces = None
            self.time_dict["grad_frame_time"] = 0
        self.pred_dm0 = self.mf.make_rdm1()
        self.pred_mo_energy = self.mf.mo_energy
        self.pred_mo_coeff = self.mf.mo_coeff
        self.results["energy"] = energy * self.energy_unit
        if forces is not None:
            self.results["forces"] = forces * self.force_unit
        else:
            self.results["forces"] = None

        self.results["cycles"] = self.mf.cycles
        return self.results


class UKSCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, basis="def2-SVP", functional="pbe", max_cycle=50, density_fit=True, energy_unit="ev", force_unit="ev/ang", level_shift=None):
        Calculator.__init__(self)
        self.basis = basis
        self.functional = functional
        self.gpu4pyscf = USE_GPU4PYSCF
        self.max_cycle = max_cycle
        self.density_fit = density_fit
        self.energy_unit_str = energy_unit
        self.energy_unit = get_energy_unit_conversion_factor(energy_unit)
        self.force_unit_str = force_unit
        self.force_unit = get_force_unit_conversion_factor(force_unit)
        self.level_shift = level_shift
        
    @staticmethod
    def _mol(atoms, coords, basis, unit="ang", spin=0):
        unit = normalize_length_unit(unit)
        mol = gto.Mole()
        mol_conf = [
            [atoms[atom_idx], coords[atom_idx]] 
            for atom_idx in range(len(atoms))
        ]
        mol.build(verbose=0, atom=mol_conf, basis=basis, unit=unit, spin=spin)
        return mol

    @staticmethod
    def UKS(mol, functional="pbe", basis="def2-SVP", gpu4pyscf=USE_GPU4PYSCF, max_cycle=50, density_fit=True, level_shift=None):
        mf = dft.UKS(mol)
        if density_fit:
            mf = mf.density_fit()
        if gpu4pyscf:
            mf = mf.to_gpu()
        mf.xc = functional
        mf.basis = basis
        mf.max_cycle = max_cycle
        if level_shift is not None:
            mf.level_shift = level_shift
        return mf

    def get_mol(self, atoms, coords, unit="ang", spin=0):
        return self._mol(atoms, coords, self.basis, unit, spin)

    def calculate(self, ase_atoms, properties, system_changes, units="ang", mf=None, spin=0):
        Calculator.calculate(self, ase_atoms, properties, system_changes)
        # properties, system_changes is not used
        units = normalize_length_unit(units)
        atoms, coords = ase_atoms_to_atoms_coords(ase_atoms)
        if mf is None:
            mol = self.get_mol(atoms, coords, units, spin=spin)
            self.mf = self.UKS(mol, gpu4pyscf=self.gpu4pyscf, max_cycle=self.max_cycle, density_fit=self.density_fit, level_shift=self.level_shift)
        else:
            self.mf = mf
        energy = self.mf.kernel()
        # unconverged
        if "forces" in properties:
            forces = -self.mf.nuc_grad_method().kernel()
        else:
            forces = None
        if not self.mf.converged:
            logging.warning(f"Unconverged energy: {energy}, cycles: {self.mf.cycles}")
        self.pred_dm0 = self.mf.make_rdm1()
        if hasattr(self.pred_dm0[0], 'get'):  # cupy array
            self.pred_dm0[0] = self.pred_dm0[0].get()
            self.pred_dm0[1] = self.pred_dm0[1].get()
        self.pred_mo_energy = self.mf.mo_energy
        self.pred_mo_coeff = self.mf.mo_coeff        
        self.results["energy"] = energy * self.energy_unit
        if forces is not None:
            self.results["forces"] = forces * self.force_unit
        else:
            self.results["forces"] = None

        self.results["cycles"] = self.mf.cycles
        # return self.results

# Single point calculator for SCFlow
class SCFlowRKSCalculator(RKSCalculator):
    default_args = {
        "md17": {
            "basis": "def2-SVP",
            "functional": "pbe",
            "units": "ang",
            "model_length_unit": "bohr",
            "mf_init_functional": "pbe, pbe",
            "init_density_fit": False,
        },
        "rmd17": {
            "basis": "def2-SVP",
            "functional": "pbe",
            "units": "ang",
            "model_length_unit": "ang",
            "mf_init_functional": "pbe",
            "density_fit": True,
            "init_density_fit": True,
        },
        "qh9": {
            "basis": "def2-SVP",
            "functional": "b3lyp",
            "units": "ang",
            "model_length_unit": "ang",
            "mf_init_functional": "b3lyp",
        },
    }

    atom_num: Dict[str, int] = {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
    }
    
    def __init__(self, device="cuda", basis="def2-SVP", functional="pbe", energy_unit="ev", force_unit="ev/ang", max_cycle=50, level_shift=None):
        Calculator.__init__(self)
        RKSCalculator.__init__(self, basis=basis, functional=functional, energy_unit=energy_unit, force_unit=force_unit, max_cycle=max_cycle, level_shift=level_shift)
        self.model = None
        self.conf = None
        self.device = device
        self.gpu4pyscf = USE_GPU4PYSCF
        self._reset_time_dict()

    @staticmethod
    def calc_mo_energy_and_coeff(ham_transformed, calc_overlap, tol=1e-8, pad_eigval=1):
        """Prepare Hamiltonian transformation and compute density matrix
        
        Returns:
            mo_energy: Molecular orbital energies
            mo_coeff: Molecular orbital coefficients
        """
        dtype = torch.float64
        overlap = calc_overlap.to(dtype)

        if overlap.dim() == 2:
            overlap = overlap.unsqueeze(0)
        if ham_transformed.dim() == 2:
            ham_transformed = ham_transformed.unsqueeze(0)
        orbital_energies, orbital_coefficients = cal_orbital_and_energies(
            overlap, ham_transformed, tol=tol, pad_eigval=pad_eigval
        )

        # Extract orbital properties
        mo_energy = orbital_energies.squeeze().numpy()
        mo_coeff = orbital_coefficients.squeeze().numpy()
        
        return mo_energy, mo_coeff

    def set_model(
            self,
            model_path,
            model_type="flow",
            device="cuda",
            ode_steps=1,
            data_type="md17",
            units="ang",
            model_length_unit="bohr",
            density_fit=None,
            filtering=False,
            gt_tol=1e-8,
            pred_tol=1e-3,
            pad_eigval=None,
            mf_init_functional="pbe,pbe",
            init_density_fit=False,
            init_gpu4pyscf=None,
            max_radius=1000,
            max_num_neighbors=200,
            basis="def2-SVP", # un used here
            functional="pbe", # un used here
            ):
        assert data_type.lower() in ["md17", "rmd17", "qm9", "qh9"], f"Invalid data type: {data_type}"
        if data_type.lower() == "qm9":
            print("QM9 is corrected to QH9")
            data_type = "qh9"
        if data_type.lower() == "rmd17":
            print("RMD17 is corrected to MD17")
            data_type = "md17"
        assert model_type in ["flow", "base"], f"Invalid model type: {model_type}"
        self.model_path = model_path
        self.model_type = model_type
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
        self.ckpt = torch.load(model_path)
        self.conf = self.ckpt["hyper_parameters"]["conf"]
        model_cls = get_pl_model(self.conf)
        self.model = model_cls.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.to(device)
        self.data_type = data_type
        print("model trained on dataset: ", self.conf.dataset.dataset_name)
        self.set_ode_steps(ode_steps)

        self.units = normalize_length_unit(units)
        if self.units != self.default_args[self.data_type]["units"]:
            # print warning
            print(f"Units are not consistent with the model. Units: {self.units}, Model units: {self.default_args[self.data_type]['units']}")
        self.model_length_unit = normalize_length_unit(model_length_unit)
        if self.model_length_unit != self.default_args[self.data_type]["model_length_unit"]:
            # print warning
            print(f"Model length unit are not consistent with the model. Model length unit: {self.model_length_unit}, Model length unit: {self.default_args[self.data_type]['model_length_unit']}")

        self.density_fit = density_fit
        self.filtering = filtering
        self.gt_tol = gt_tol
        self.pred_tol = pred_tol
        self.pad_eigval = pad_eigval
        self.mf_init_functional = mf_init_functional
        self.init_density_fit = init_density_fit
        
        if init_gpu4pyscf is None:
            self.init_gpu4pyscf = self.gpu4pyscf
        else:
            self.init_gpu4pyscf = init_gpu4pyscf

        if self.init_gpu4pyscf:
            if not self.gpu4pyscf:
                raise ValueError("init_gpu4pyscf is True, but gpu4pyscf is False")

        if self.data_type == "qh9":
            self.full_orbitals = 14
            self._orbital_mask_line1 = torch.tensor([0, 1, 3, 4, 5])
            self._orbital_mask_line2 = torch.arange(self.full_orbitals)
            self.orbital_mask = {
                self.atom_num["H"]: self._orbital_mask_line1,
                self.atom_num["C"]: self._orbital_mask_line2,
                self.atom_num["N"]: self._orbital_mask_line2,
                self.atom_num["O"]: self._orbital_mask_line2,
                self.atom_num["F"]: self._orbital_mask_line2,
            }
        if max_radius is not None:
            self.max_radius = max_radius
            self.model.model.max_radius = max_radius
        else:
            self.max_radius = self.model.model.max_radius
        if max_num_neighbors is not None:
            self.max_num_neighbors = max_num_neighbors
            self.model.model.max_num_neighbors = max_num_neighbors
        else:
            self.max_num_neighbors = self.model.model.max_num_neighbors
        print(f"max_radius: {self.max_radius}")
        print(f"max_num_neighbors: {self.max_num_neighbors}")
    @property
    def ode_steps(self):
        if self.conf is None:
            raise ValueError("Model not set")
        if "flow" not in self.conf:
            raise ValueError("Model is not a flow model")
        return self.conf["flow"]["num_ode_steps_test"]
   
    def set_ode_steps(self, ode_steps=1):
        assert isinstance(ode_steps, int)
        assert ode_steps > 0
        assert self.conf is not None
        self.conf["flow"]["num_ode_steps_test"] = ode_steps
        print(f"ode_steps: {self.ode_steps}")
    
    @staticmethod
    def _atoms_coords_from_mol(mol):
        # retrun numpy array
        assert isinstance(mol, gto.Mole)
        atom_conf = mol.atom
        atoms = np.array([a[0] for a in atom_conf])
        coords = np.array([a[1] for a in atom_conf])
        return atoms, coords
    
    def _reset_time_dict(self):
        self.time_dict = {
            "mf_time": 0,
            "ovlp_time": 0,
            "init_ham_time": 0,
            "pred_ham_time": 0,
            "pred_dm0_time": 0,
            "energy_time": 0,
            "grad_frame_time": 0
        }

    def forward_component(self, atoms, coords, ovlp, init_ham, input_transform=True, output_transform=True):
        """Forward pass through the model to get predicted Hamiltonian"""
        # Convert atoms to atomic numbers if they are symbols
        if isinstance(atoms, list) and isinstance(atoms[0], str):
            from pyscf import gto
            atom_numbers = [gto.charge(a) for a in atoms]
        else:
            atom_numbers = atoms
        
        # Convert to torch tensors
        
        # Create a simple data object for the model
        pos = torch.tensor(coords, dtype=torch.float32).cuda()
        _atoms = torch.tensor(atom_numbers, dtype=torch.long).squeeze().cuda()
        _init_ham = torch.tensor(init_ham, dtype=torch.float32).squeeze().cuda()
        _ovlp = torch.tensor(ovlp, dtype=torch.float32).squeeze().cuda()
        h_dim = init_ham.shape[0]
        num_atoms = len(_atoms)

        if input_transform:
            convention = "pyscf_def2svp_to_e3nn"
            _init_ham = matrix_transform_single(_init_ham, _atoms, convention=convention)
            _ovlp = matrix_transform_single(_ovlp, _atoms, convention=convention)

        if self.data_type == "md17" or self.data_type == "rmd17":
            edge_index = []
            for i in range(len(_atoms)):
                for j in range(len(_atoms)):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            full_edge_index = edge_index

            one_batch = Data(
                pos=pos,
                atoms=_atoms.view(-1, 1),
                init_ham=_init_ham.unsqueeze(0),
                overlap=_ovlp.unsqueeze(0),
                full_edge_index=full_edge_index,
                h_dim=torch.tensor(h_dim, dtype=torch.long).view(-1, 1),
                num_atoms=torch.tensor(num_atoms, dtype=torch.long).view(-1, 1),
            )

            # batch
            batch = Batch.from_data_list([one_batch])
        
            # Forward pass
            with torch.no_grad():
                batch_gpu = batch.to(self.model.device)
                outputs = self._forward(batch_gpu)
                # 필요한 데이터를 먼저 CPU로 추출
                pred_ham = outputs[0]["hamiltonian"].cpu()

            # GPU batch 명시적으로 삭제
            del batch_gpu
            # torch.cuda.empty_cache()  # 선택사항: GPU 캐시 강제 정리

            # Transform Hamiltonian to PySCF convention
            # pred_ham = outputs["hamiltonian"]
            if output_transform:
                pred_ham_transformed = matrix_transform_single(
                    pred_ham, 
                    batch.atoms, 
                    convention="e3nn_to_pyscf_def2svp"
                )
            else:
                pred_ham_transformed = pred_ham
            pred_ham_transformed = pred_ham_transformed.squeeze(0).detach().cpu().numpy()
            return pred_ham_transformed  # Remove batch dimension
        elif self.data_type == "qh9":
            concat_feat = torch.stack(
                [_ovlp, _init_ham], dim=0
            )

            (
                concat_diagonal_blocks,
                concat_non_diagonal_blocks,
                concat_diagonal_masks,
                concat_non_diagonal_masks,
                edge_index_full,
            ) = _cut_matrix_3d(
                concat_feat,
                _atoms,
                self.orbital_mask,
                self.full_orbitals,
            )

            diagonal_hamiltonian_mask = concat_diagonal_masks[:, 0]
            non_diagonal_hamiltonian_mask = concat_non_diagonal_masks[:, 0]

            diagonal_overlap = concat_diagonal_blocks[:, 0]
            non_diagonal_overlap = concat_non_diagonal_blocks[:, 0]

            diagonal_init_ham = concat_diagonal_blocks[:, 1]
            non_diagonal_init_ham = concat_non_diagonal_blocks[:, 1]

            one_batch = Data(
                pos=pos,
                atoms=_atoms.view(-1, 1),
                diagonal_overlap=diagonal_overlap,
                non_diagonal_overlap=non_diagonal_overlap,
                diagonal_init_ham=diagonal_init_ham,
                non_diagonal_init_ham=non_diagonal_init_ham,
                diagonal_hamiltonian_mask=diagonal_hamiltonian_mask,
                non_diagonal_hamiltonian_mask=non_diagonal_hamiltonian_mask,
                edge_index_full=edge_index_full,
            )
            self.one_batch = one_batch

            batch = Batch.from_data_list([one_batch])

            with torch.no_grad():
                batch_gpu = batch.to(self.model.device)
                outputs = self._forward(batch_gpu)
                self.outputs = outputs
                # 필요한 데이터를 먼저 CPU로 추출
                pred_ham_diagonal = outputs[0]["hamiltonian_diagonal_blocks"]
                pred_ham_non_diagonal = outputs[0]["hamiltonian_non_diagonal_blocks"]
            
                pred_ham_transformed = self.model.build_final_matrix(
                    batch_gpu,
                    pred_ham_diagonal,
                    pred_ham_non_diagonal,
                    transform=output_transform,
                    convention="e3nn_to_pyscf_def2svp",
                )
                pred_ham_transformed = pred_ham_transformed[0]
                # print(pred_ham_transformed.shape)
            # GPU batch 명시적으로 삭제
            del batch_gpu
            # torch.cuda.empty_cache()  # 선택사항: GPU 캐시 강정리
            return pred_ham_transformed.squeeze(0).detach().cpu().numpy().astype(np.float64)
        else:
            raise NotImplementedError("Only MD17 and QH9 are supported for now")

    def _forward(self, x):
        batch = self.model.post_processing(x, self.model.default_type)
        # batch = self.model.corrupt(batch)
        outputs = self.model.sample(batch, num_timesteps=self.ode_steps)
        return outputs

    def calc_mo_energy_and_coeff(self, ham_transformed, calc_overlap, tol=1e-8, pad_eigval=1):
        """Calculate molecular orbital energies and coefficients"""
        # Inline implementation to avoid recursion with module-level function
        dtype = torch.float64
        overlap = calc_overlap.to(dtype) if isinstance(calc_overlap, torch.Tensor) else torch.tensor(calc_overlap, dtype=dtype)

        if overlap.dim() == 2:
            overlap = overlap.unsqueeze(0)
        if isinstance(ham_transformed, torch.Tensor):
            if ham_transformed.dim() == 2:
                ham_transformed = ham_transformed.unsqueeze(0)
        else:
            ham_transformed = torch.tensor(ham_transformed, dtype=dtype).unsqueeze(0)
        
        from common.metric import cal_orbital_and_energies
        orbital_energies, orbital_coefficients = cal_orbital_and_energies(
            overlap, ham_transformed, tol=tol, pad_eigval=pad_eigval
        )

        mo_energy = orbital_energies.squeeze().numpy()
        mo_coeff = orbital_coefficients.squeeze().numpy()
        
        return mo_energy, mo_coeff

    @staticmethod
    def calc_density(mo_coeff, n_occ):
        """Calculate density matrix from molecular orbital energies and coefficients"""
        sliced_mo_coeff = mo_coeff[:, :n_occ]
        density = sliced_mo_coeff @ sliced_mo_coeff.T * 2
        return density


    def calculate(
        self,
        ase_atoms,
        properties,
        system_changes,
        units=None,
        model_length_unit=None,
        density_fit=None,
        filtering=None,
        gt_tol=None,
        pred_tol=None,
        pad_eigval=None,
        init_ham=None,
        init_gpu4pyscf=None,
        scf=False,
        spin=0,
    ):
        if units is None:
            units = self.units
        else:
            units = normalize_length_unit(units)
        if model_length_unit is None:
            model_length_unit = self.model_length_unit
        else:
            model_length_unit = normalize_length_unit(model_length_unit)
        if density_fit is None:
            density_fit = self.density_fit
        if filtering is None:
            filtering = self.filtering
        if gt_tol is None:
            gt_tol = self.gt_tol
        if pred_tol is None:
            pred_tol = self.pred_tol
        if pad_eigval is None:
            pad_eigval = self.pad_eigval
        if init_gpu4pyscf is None:
            init_gpu4pyscf = self.init_gpu4pyscf
        
        if scf is True:
            return self.calculate_scf(ase_atoms, properties, system_changes, units, model_length_unit, density_fit, filtering, gt_tol, pred_tol, pad_eigval, init_ham, init_gpu4pyscf)
        
        Calculator.calculate(self, ase_atoms, properties, system_changes)
        # properties, system_changes is not used
        atoms, coords = ase_atoms_to_atoms_coords(ase_atoms)
        if self.model is None:
            raise ValueError("Model is not set")
        
        if density_fit is None:
            density_fit = True
        
        gpu4pyscf = self.gpu4pyscf
        self._reset_time_dict()
        start_time = time.time()
        mol = self.get_mol(atoms, coords, units, spin=spin)
        self.mf = self.RKS(mol, self.functional, self.basis, gpu4pyscf=gpu4pyscf, density_fit=density_fit, max_cycle=self.max_cycle, level_shift=self.level_shift)
        self.time_dict["mf_time"] = time.time() - start_time
        
        start_time = time.time()
        self.ovlp = mol.intor("int1e_ovlp")
        self.time_dict["ovlp_time"] = time.time() - start_time
        
        start_time = time.time()
        mol_wo_spin = self.get_mol(atoms, coords, units, spin=0)
        self.init_mf = self.RKS(mol_wo_spin,self.mf_init_functional, self.basis, gpu4pyscf=self.init_gpu4pyscf, density_fit=self.init_density_fit, level_shift=self.level_shift)
        self.init_dm = self.init_mf.init_guess_by_minao()
        # self.init_dm = self.mf.get_init_guess(key="vsap")
        self.init_ham = self.init_mf.get_fock(dm=self.init_dm)
        # self.mf.xc = previous_xc
        self.time_dict["init_ham_time"] = time.time() - start_time


        atoms, coords = self._atoms_coords_from_mol(mol)
        input_coords = convert_length(coords, from_unit="ang", to_unit=model_length_unit)
        start_time = time.time()
        current_init_ham = self.init_ham
        if init_ham is not None:
            current_init_ham = init_ham
        self.pred_ham = self.forward_component(atoms, input_coords, self.ovlp, current_init_ham)
        self.time_dict["pred_ham_time"] = time.time() - start_time
        # self.dm0 = self.calc_dm0(atoms, coords, self.pred_ham)
        # self.mf.dm0 = self.dm0
        
        # self.mf.grids.level = 1
        pred_ham_transformed = torch.tensor(self.pred_ham)
        calc_overlap = torch.from_numpy(self.ovlp)
        n_occ = mol.nelectron // 2
        self.n_occ = n_occ
        start_time = time.time()
        if filtering:
            pred_mo_energy_safe, pred_mo_coeff_safe = self.calc_mo_energy_and_coeff(
                pred_ham_transformed, calc_overlap, tol=gt_tol, pad_eigval=pad_eigval
            )
            pred_mo_energy_unsatable, pred_mo_coeff_unsatable = self.calc_mo_energy_and_coeff(
                pred_ham_transformed, calc_overlap, tol=pred_tol, pad_eigval=pad_eigval
            )
            mo_energy_diff = pred_mo_energy_safe - pred_mo_energy_unsatable
            if abs(min(mo_energy_diff[:n_occ+1])) > 0.03:
                pred_mo_energy = pred_mo_energy_unsatable
                pred_mo_coeff = pred_mo_coeff_unsatable
            else:
                pred_mo_energy = pred_mo_energy_safe
                pred_mo_coeff = pred_mo_coeff_safe
        else:
            pred_mo_energy, pred_mo_coeff = self.calc_mo_energy_and_coeff(
                pred_ham_transformed, calc_overlap, tol=gt_tol, pad_eigval=pad_eigval
            )
        self.pred_mo_energy = pred_mo_energy
        self.homo = pred_mo_energy[n_occ-1]
        self.lumo = pred_mo_energy[n_occ]
        self.pred_mo_coeff = pred_mo_coeff
        self.pred_dm0 = self.calc_density(pred_mo_coeff, n_occ)
        self.time_dict["pred_dm0_time"] = time.time() - start_time

        if gpu4pyscf:
            cur_density = asarray(self.pred_dm0)
        else:
            cur_density = self.pred_dm0
        
        start_time = time.time()
        energy = self.mf.energy_tot(cur_density)
        self.time_dict["energy_time"] = time.time() - start_time

        if "forces" in properties:
            start_time = time.time()
            grad_frame = self.mf.nuc_grad_method()

            mo_occ = self.mf.get_occ(pred_mo_energy, pred_mo_coeff)

            if gpu4pyscf:
                cur_mo_energy = pred_mo_energy
                cur_mo_coeff = pred_mo_coeff
                cur_mo_occ = mo_occ
                grad_frame.base.mo_energy = cur_mo_energy
                grad_frame.base.mo_coeff = cur_mo_coeff
                grad_frame.base.mo_occ = cur_mo_occ
            else:
                cur_mo_energy = pred_mo_energy
                cur_mo_coeff = pred_mo_coeff
                cur_mo_occ = mo_occ
            grad_frame.base.auxbasis_response = True
            forces = -grad_frame.kernel(
                mo_energy=cur_mo_energy,
                mo_coeff=-cur_mo_coeff,
                mo_occ=cur_mo_occ)
            self.time_dict["grad_frame_time"] = time.time() - start_time
        else:
            forces = None
            self.time_dict["grad_frame_time"] = 0
        
        self.results["energy"] = energy * self.energy_unit
        if forces is not None:
            self.results["forces"] = forces * self.force_unit
        else:
            self.results["forces"] = None

        return self.results

    def calculate_scf(
            self,
            ase_atoms,
            properties,
            system_changes,
            units=None,
            model_length_unit=None,
            density_fit=None,
            filtering=None,
            gt_tol=None,
            pred_tol=None,
            pad_eigval=None,
            init_ham=None,
            init_gpu4pyscf=None,
            spin=0,
        ):
            if units is None:
                units = self.units
            else:
                units = normalize_length_unit(units)
            if model_length_unit is None:
                model_length_unit = self.model_length_unit
            else:
                model_length_unit = normalize_length_unit(model_length_unit)
            if density_fit is None:
                density_fit = self.density_fit
            if filtering is None:
                filtering = self.filtering
            if gt_tol is None:
                gt_tol = self.gt_tol
            if pred_tol is None:
                pred_tol = self.pred_tol
            if pad_eigval is None:
                pad_eigval = self.pad_eigval
            if init_gpu4pyscf is None:
                init_gpu4pyscf = self.init_gpu4pyscf
            Calculator.calculate(self, ase_atoms, properties, system_changes)
            # properties, system_changes is not used
            atoms, coords = ase_atoms_to_atoms_coords(ase_atoms)
            if self.model is None:
                raise ValueError("Model is not set")
            
            if density_fit is None:
                density_fit = True
            
            gpu4pyscf = self.gpu4pyscf
            self._reset_time_dict()
            start_time = time.time()
            mol = self.get_mol(atoms, coords, units, spin=spin)
            self.mf = self.RKS(mol, self.functional, self.basis, gpu4pyscf=gpu4pyscf, density_fit=density_fit, max_cycle=self.max_cycle, level_shift=self.level_shift)
            self.time_dict["mf_time"] = time.time() - start_time
            
            start_time = time.time()
            self.ovlp = mol.intor("int1e_ovlp")
            self.time_dict["ovlp_time"] = time.time() - start_time
            
            start_time = time.time()
            mol_wo_spin = self.get_mol(atoms, coords, units, spin=0)
            self.init_mf = self.RKS(mol_wo_spin, self.mf_init_functional, self.basis, gpu4pyscf=self.init_gpu4pyscf, density_fit=self.init_density_fit, level_shift=self.level_shift)
            self.init_dm = self.init_mf.init_guess_by_minao()
            # self.init_dm = self.mf.get_init_guess(key="vsap")
            self.init_ham = self.init_mf.get_fock(dm=self.init_dm)
            # self.mf.xc = previous_xc
            self.time_dict["init_ham_time"] = time.time() - start_time


            atoms, coords = self._atoms_coords_from_mol(mol)
            input_coords = convert_length(coords, from_unit="ang", to_unit=model_length_unit)
            start_time = time.time()
            current_init_ham = self.init_ham
            if init_ham is not None:
                current_init_ham = init_ham
            self.pred_ham = self.forward_component(atoms, input_coords, self.ovlp, current_init_ham)
            self.time_dict["pred_ham_time"] = time.time() - start_time
            # self.dm0 = self.calc_dm0(atoms, coords, self.pred_ham)
            # self.mf.dm0 = self.dm0
            
            # self.mf.grids.level = 1
            pred_ham_transformed = torch.tensor(self.pred_ham)
            calc_overlap = torch.from_numpy(self.ovlp)
            n_occ = mol.nelectron // 2
            self.n_occ = n_occ
            start_time = time.time()
            if filtering:
                pred_mo_energy_safe, pred_mo_coeff_safe = self.calc_mo_energy_and_coeff(
                    pred_ham_transformed, calc_overlap, tol=gt_tol, pad_eigval=pad_eigval
                )
                pred_mo_energy_unsatable, pred_mo_coeff_unsatable = self.calc_mo_energy_and_coeff(
                    pred_ham_transformed, calc_overlap, tol=pred_tol, pad_eigval=pad_eigval
                )
                mo_energy_diff = pred_mo_energy_safe - pred_mo_energy_unsatable
                if abs(min(mo_energy_diff[:n_occ+1])) > 0.03:
                    pred_mo_energy = pred_mo_energy_unsatable
                    pred_mo_coeff = pred_mo_coeff_unsatable
                else:
                    pred_mo_energy = pred_mo_energy_safe
                    pred_mo_coeff = pred_mo_coeff_safe
            else:
                pred_mo_energy, pred_mo_coeff = self.calc_mo_energy_and_coeff(
                    pred_ham_transformed, calc_overlap, tol=gt_tol, pad_eigval=pad_eigval
                )
            self.pred_mo_energy = pred_mo_energy
            self.pred_mo_coeff = pred_mo_coeff
            self.pred_dm0 = self.calc_density(pred_mo_coeff, n_occ)
            self.time_dict["pred_dm0_time"] = time.time() - start_time

            if gpu4pyscf:
                cur_density = asarray(self.pred_dm0)
            else:
                cur_density = self.pred_dm0
            
            start_time = time.time()
            # energy = self.mf.energy_tot(cur_density)
            energy = self.mf.kernel(dm0=cur_density, dump_chk=False)
            self.time_dict["energy_time"] = time.time() - start_time

            if "forces" in properties:
                start_time = time.time()
                grad_frame = self.mf.nuc_grad_method()
                forces = -grad_frame.kernel()
                self.time_dict["grad_frame_time"] = time.time() - start_time
            else:
                forces = None
                self.time_dict["grad_frame_time"] = 0
                
            self.results["energy"] = energy * self.energy_unit
            if forces is not None:
                self.results["forces"] = forces * self.force_unit
            else:
                self.results["forces"] = None
            self.results["cycles"] = self.mf.cycles

            return self.results
        
    def calculate_uks(
            self,
            ase_atoms,
            properties,
            system_changes,
            units=None,
            model_length_unit=None,
            density_fit=None,
            filtering=None,
            gt_tol=None,
            pred_tol=None,
            pad_eigval=None,
            init_ham=None,
            init_gpu4pyscf=None,
            scf=False,
            spin=0,
        ):
            if units is None:
                units = self.units
            else:
                units = normalize_length_unit(units)
            if model_length_unit is None:
                model_length_unit = self.model_length_unit
            else:
                model_length_unit = normalize_length_unit(model_length_unit)
            if density_fit is None:
                density_fit = self.density_fit
            if filtering is None:
                filtering = self.filtering
            if gt_tol is None:
                gt_tol = self.gt_tol
            if pred_tol is None:
                pred_tol = self.pred_tol
            if pad_eigval is None:
                pad_eigval = self.pad_eigval
            if init_gpu4pyscf is None:
                init_gpu4pyscf = self.init_gpu4pyscf
            Calculator.calculate(self, ase_atoms, properties, system_changes)
            # properties, system_changes is not used
            atoms, coords = ase_atoms_to_atoms_coords(ase_atoms)
            if self.model is None:
                raise ValueError("Model is not set")
            
            if density_fit is None:
                density_fit = True
            
            gpu4pyscf = self.gpu4pyscf
            self._reset_time_dict()
            start_time = time.time()
            mol = self.get_mol(atoms, coords, units, spin=spin)
            self.mf = self.UKS(mol, self.functional, self.basis, gpu4pyscf=gpu4pyscf, density_fit=density_fit, max_cycle=self.max_cycle, level_shift=self.level_shift)
            self.time_dict["mf_time"] = time.time() - start_time
            
            start_time = time.time()
            self.ovlp = mol.intor("int1e_ovlp")
            self.time_dict["ovlp_time"] = time.time() - start_time
            
            start_time = time.time()
            mol_wo_spin = self.get_mol(atoms, coords, units, spin=0)    
            self.init_mf = self.RKS(mol_wo_spin,self.mf_init_functional, self.basis, gpu4pyscf=self.init_gpu4pyscf, density_fit=self.init_density_fit, level_shift=self.level_shift)
            self.init_dm = self.init_mf.init_guess_by_minao()
            # self.init_dm = self.mf.get_init_guess(key="vsap")
            self.init_ham = self.init_mf.get_fock(dm=self.init_dm)
            # self.mf.xc = previous_xc
            self.time_dict["init_ham_time"] = time.time() - start_time


            atoms, coords = self._atoms_coords_from_mol(mol)
            input_coords = convert_length(coords, from_unit="ang", to_unit=model_length_unit)
            start_time = time.time()
            current_init_ham = self.init_ham
            if init_ham is not None:
                current_init_ham = init_ham
            self.pred_ham = self.forward_component(atoms, input_coords, self.ovlp, current_init_ham)
            self.time_dict["pred_ham_time"] = time.time() - start_time
            # self.dm0 = self.calc_dm0(atoms, coords, self.pred_ham)
            # self.mf.dm0 = self.dm0
            
            # self.mf.grids.level = 1
            pred_ham_transformed = torch.tensor(self.pred_ham)
            calc_overlap = torch.from_numpy(self.ovlp)
            n_occ = mol.nelectron // 2
            self.n_occ = n_occ
            start_time = time.time()
            if filtering:
                pred_mo_energy_safe, pred_mo_coeff_safe = self.calc_mo_energy_and_coeff(
                    pred_ham_transformed, calc_overlap, tol=gt_tol, pad_eigval=pad_eigval
                )
                pred_mo_energy_unsatable, pred_mo_coeff_unsatable = self.calc_mo_energy_and_coeff(
                    pred_ham_transformed, calc_overlap, tol=pred_tol, pad_eigval=pad_eigval
                )
                mo_energy_diff = pred_mo_energy_safe - pred_mo_energy_unsatable
                if abs(min(mo_energy_diff[:n_occ+1])) > 0.03:
                    pred_mo_energy = pred_mo_energy_unsatable
                    pred_mo_coeff = pred_mo_coeff_unsatable
                else:
                    pred_mo_energy = pred_mo_energy_safe
                    pred_mo_coeff = pred_mo_coeff_safe
            else:
                pred_mo_energy, pred_mo_coeff = self.calc_mo_energy_and_coeff(
                    pred_ham_transformed, calc_overlap, tol=gt_tol, pad_eigval=pad_eigval
                )
            self.pred_mo_energy = pred_mo_energy
            self.pred_mo_coeff = pred_mo_coeff
            self.pred_dm0 = self.calc_density(pred_mo_coeff, n_occ)
            self.time_dict["pred_dm0_time"] = time.time() - start_time

            if gpu4pyscf:
                cur_density = asarray(self.pred_dm0)
            else:
                cur_density = self.pred_dm0
                
            cur_density = (cur_density / 2.0, cur_density / 2.0)
            
            start_time = time.time()
            if scf is True:
                energy = self.mf.kernel(dm0=cur_density, dump_chk=False)
            else:
                energy = self.mf.energy_tot(cur_density)
            self.time_dict["energy_time"] = time.time() - start_time

            if "forces" in properties:
                start_time = time.time()
                grad_frame = self.mf.nuc_grad_method()

                mo_occ = self.mf.get_occ(pred_mo_energy, pred_mo_coeff)
                if scf is True:
                    forces = -grad_frame.kernel()
                else:
                    if gpu4pyscf:
                        cur_mo_energy = pred_mo_energy
                        cur_mo_coeff = pred_mo_coeff
                        cur_mo_occ = mo_occ
                        grad_frame.base.mo_energy = cur_mo_energy
                        grad_frame.base.mo_coeff = cur_mo_coeff
                        grad_frame.base.mo_occ = cur_mo_occ
                    else:
                        cur_mo_energy = pred_mo_energy
                        cur_mo_coeff = pred_mo_coeff
                        cur_mo_occ = mo_occ
                    grad_frame.base.auxbasis_response = True
                    forces = -grad_frame.kernel(
                        mo_energy=cur_mo_energy,
                        mo_coeff=-cur_mo_coeff,
                        mo_occ=cur_mo_occ)
                self.time_dict["grad_frame_time"] = time.time() - start_time
            else:
                forces = None
                self.time_dict["grad_frame_time"] = 0
            
            self.results["energy"] = energy * self.energy_unit
            if forces is not None:
                self.results["forces"] = forces * self.force_unit
            else:
                self.results["forces"] = None

            return self.results