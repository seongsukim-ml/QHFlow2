global IMPORT_GPU4PYSCF
IMPORT_GPU4PYSCF = False

try:
    import gpu4pyscf
    from gpu4pyscf.lib.cupy_helper import (tag_array, pack_tril, asarray)
    IMPORT_GPU4PYSCF = True
    print("gpu4pyscf is installed.")
except:
    IMPORT_GPU4PYSCF = False
    print("gpu4pyscf is not installed. Please install it using 'pip install gpu4pyscf'.")


import sys
sys.path.append("/root/limlab01/kaistai/25DFT/QHFlow/src")
# from common.dft_utils import *
import os
# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["MKL_NUM_THREADS"] = "8"
# os.environ["NUMEXPR_NUM_THREADS"] = "8"
# os.environ["OPENBLAS_NUM_THREADS"] = "8"
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dft_process.dft_process_utils import *
# from common.draw_util import *
import torch
from torch_geometric.loader import DataLoader
from common.units import *
import time

from common.draw_util import *

from pl_module import get_pl_model
from pyscf import dft, gto
from torch_geometric.data import Data, Batch
from common.matrix_transforms import (
    _cut_matrix_3d,
    _cut_matrix_3d_last,
    cut_orbital_coefficients,
    get_convention_dict,
    # matrix_transform_single,
    pack_upper_triangle,
    unpack_upper_triangle,
    compute_chc_from_blocks,
)

# from pyscf.df import df_jk
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

class RKS:
    BOHR2ANG = BOHR2ANG
    ANG2BOHR = 1.0 / BOHR2ANG

    def __init__(self, basis="def2-SVP", functional="PBE"):
        self.basis = basis
        self.functional = functional

    @staticmethod
    def _mol(atoms, coords, basis, unit="ang"):
        unit = unit.lower()
        mol = gto.Mole()
        mol_conf = [
            [atoms[atom_idx], coords[atom_idx]] 
            for atom_idx in range(len(atoms))
        ]
        mol.build(verbose=0, atom=mol_conf, basis=basis, unit=unit)
        return mol

    def get_mol(self, atoms, coords, unit="ang"):
        return self._mol(atoms, coords, self.basis, unit)

    @staticmethod
    def _overlap_and_init_ham_from_mol(mol, functional="PBE", basis="def2-SVP", init_type="minao"):
        mf = dft.RKS(mol)
        mf.xc = functional
        mf.basis = basis
        overlap_matrix = mol.intor("int1e_ovlp")
        density_matrix_init = mf.get_init_guess(key=init_type)
        init_hamiltonian = mf.get_fock(dm=density_matrix_init)
        
        return overlap_matrix.astype("float64"), init_hamiltonian.astype("float64")

    @staticmethod
    def RKS(mol, functional="PBE", basis="def2-SVP"):
        mf = dft.RKS(mol)
        mf.xc = functional
        mf.basis = basis
        return mf

class MLRKS(RKS):
    def __init__(self, basis="def2-SVP", functional="PBE", gpu4pyscf=False):
        super().__init__()
        self.model = None
        self.conf = None
        self.gpu4pyscf = gpu4pyscf
        self.time_dict = {
            "mf_time": 0,
            "ovlp_time": 0,
            "init_ham_time": 0,
            "pred_ham_time": 0,
            "pred_dm0_time": 0,
            "energy_time": 0,
            "grad_frame_time": 0
        }

    @staticmethod
    def _atoms_coords_from_mol(mol):
        # retrun numpy array
        assert isinstance(mol, gto.Mole)
        atom_conf = mol.atom
        atoms = np.array([a[0] for a in atom_conf])
        coords = np.array([a[1] for a in atom_conf])
        return atoms, coords

    def set_model(self, model_path, model_type="flow", device="cuda", ode_steps=1, data_type="md17"):
        assert data_type.lower() in ["md17", "qm9"]
        assert model_type in ["flow", "base"]
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

    @property
    def ode_steps(self):
        return self.conf["flow"]["num_ode_steps_test"]
    
    def set_ode_steps(self, ode_steps=1):
        assert isinstance(ode_steps, int)
        assert ode_steps > 0
        assert self.conf is not None
        self.conf["flow"]["num_ode_steps_test"] = ode_steps
        print(f"ode_steps: {self.ode_steps}")


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

        if self.data_type == "md17":
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
            edge_index = []
            for i in range(len(_atoms)):
                for j in range(len(_atoms)):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            full_edge_index = edge_index
            
            raise NotImplementedError("Only MD17 is supported for now")

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

    def MLRKS(self, mol, gpu4pyscf=None, density_fit=None, model_length_unit="ang"):
        if self.model is None:
            raise ValueError("Model is not set")
        
        if gpu4pyscf is None:
            gpu4pyscf = self.gpu4pyscf
        global IMPORT_GPU4PYSCF
        if gpu4pyscf and not IMPORT_GPU4PYSCF:
            raise ImportError("gpu4pyscf is not installed. Please install it using 'pip install gpu4pyscf'.")
        
        if density_fit is None:
            density_fit = True

        # default
        # (1) if gpu4pyscf=False, density_fit=True
        # (2) if gpu4pyscf=True, density_fit=True

        start_time = time.time()
        self.mf = dft.RKS(mol)
        if density_fit:
            self.mf = self.mf.density_fit()
        if gpu4pyscf:
            self.mf = self.mf.to_gpu()
        self.time_dict["mf_time"] = time.time() - start_time

        start_time = time.time()
        self.ovlp = mol.intor("int1e_ovlp")
        self.time_dict["ovlp_time"] = time.time() - start_time
        # previous_xc = self.mf.xc

        start_time = time.time()
        self.mf.xc = "pbe,pbe"
        self.init_dm = self.mf.init_guess_by_minao()
        # self.init_dm = self.mf.get_init_guess(key="vsap")
        self.init_ham = self.mf.get_fock(dm=self.init_dm)
        # self.mf.xc = previous_xc
        self.time_dict["init_ham_time"] = time.time() - start_time

        atoms, coords = self._atoms_coords_from_mol(mol)
        self.atoms = atoms
        self.coords = coords
        if model_length_unit.lower() in ["ang", "angstrom", "a"]:
            input_coords = coords
        elif model_length_unit.lower() in ["bohr", "b"]:
            input_coords = coords * ANG2BOHR
        else:
            raise ValueError(f"Invalid model_length_unit: {model_length_unit}")
        start_time = time.time()
        self.pred_ham = self.forward_component(atoms, input_coords, self.ovlp, self.init_ham)
        self.time_dict["pred_ham_time"] = time.time() - start_time
        # self.dm0 = self.calc_dm0(atoms, coords, self.pred_ham)
        # self.mf.dm0 = self.dm0
        return self.mf

    @staticmethod
    def calc_density(mo_coeff, n_occ):
        """Calculate density matrix from molecular orbital energies and coefficients"""
        sliced_mo_coeff = mo_coeff[:, :n_occ]
        density = sliced_mo_coeff @ sliced_mo_coeff.T * 2
        return density

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

    def direct_energy_force(self, mol, calc_forces=True, mf=None, filtering=True, gt_tol=1e-8, pred_tol=1e-3, pad_eigval=None, gpu4pyscf=None, density_fit=None, model_length_unit="ang"):
        if gpu4pyscf is None:
            gpu4pyscf = self.gpu4pyscf
        if density_fit is None:
            density_fit = True

        if mf is None:
            mf = self.MLRKS(mol, gpu4pyscf=gpu4pyscf, density_fit=density_fit, model_length_unit=model_length_unit)

        mf.grids.level = 1
        pred_ham_transformed = torch.tensor(self.pred_ham)
        calc_overlap = torch.from_numpy(self.ovlp)
        n_occ = mol.nelectron // 2
        atoms = self.atoms
        coords = self.coords
        start_time = time.time()
        if filtering:
            pred_mo_energy_safe, pred_mo_coeff_safe = calc_mo_energy_and_coeff(
                pred_ham_transformed, calc_overlap, tol=gt_tol, pad_eigval=None
            )
            pred_mo_energy_unsatable, pred_mo_coeff_unsatable = calc_mo_energy_and_coeff(
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
            pred_mo_energy, pred_mo_coeff = calc_mo_energy_and_coeff(
                pred_ham_transformed, calc_overlap, tol=gt_tol, pad_eigval=pad_eigval
            )
        self.pred_dm0 = self.calc_density(pred_mo_coeff, n_occ)
        self.time_dict["pred_dm0_time"] = time.time() - start_time

        if gpu4pyscf:
            cur_density = asarray(self.pred_dm0)
        else:
            cur_density = self.pred_dm0
        
        start_time = time.time()
        energy = mf.energy_tot(cur_density)
        self.time_dict["energy_time"] = time.time() - start_time

        if calc_forces:
            start_time = time.time()
            grad_frame = mf.nuc_grad_method()

            mo_occ = mf.get_occ(pred_mo_energy, pred_mo_coeff)

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

        return energy, forces

class MLRKS_qh9(RKS):
    def __init__(self, basis="def2-SVP", functional="PBE", gpu4pyscf=False):
        super().__init__()
        self.model = None
        self.conf = None
        self.gpu4pyscf = gpu4pyscf
        self.time_dict = {
            "mf_time": 0,
            "ovlp_time": 0,
            "init_ham_time": 0,
            "pred_ham_time": 0,
            "pred_dm0_time": 0,
            "energy_time": 0,
            "grad_frame_time": 0
        }

    @staticmethod
    def _atoms_coords_from_mol(mol):
        # retrun numpy array
        assert isinstance(mol, gto.Mole)
        atom_conf = mol.atom
        atoms = np.array([a[0] for a in atom_conf])
        coords = np.array([a[1] for a in atom_conf])
        return atoms, coords

    def set_model(self, model_path, model_type="flow", device="cuda", ode_steps=1, data_type="md17"):
        assert data_type.lower() in ["md17", "qm9"]
        assert model_type in ["flow", "base"]
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
        if self.data_type == "qh9":
            self.orbital_mask = {
                self.atom_num["H"]: self._orbital_mask_line1,
                self.atom_num["C"]: self._orbital_mask_line2,
                self.atom_num["N"]: self._orbital_mask_line2,
                self.atom_num["O"]: self._orbital_mask_line2,
                self.atom_num["F"]: self._orbital_mask_line2,
            }
            self.full_orbitals = 14
    @property
    def ode_steps(self):
        return self.conf["flow"]["num_ode_steps_test"]
    
    def set_ode_steps(self, ode_steps=1):
        assert isinstance(ode_steps, int)
        assert ode_steps > 0
        assert self.conf is not None
        self.conf["flow"]["num_ode_steps_test"] = ode_steps
        print(f"ode_steps: {self.ode_steps}")


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

        if self.data_type == "md17":
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

            batch = Batch.from_data_list([one_batch])

            with torch.no_grad():
                batch_gpu = batch.to(self.model.device)
                outputs = self._forward(batch_gpu)
                # 필요한 데이터를 먼저 CPU로 추출
                pred_ham_diagonal = outputs[0]["hamiltonian_diagonal_blocks"].cpu()
                pred_ham_non_diagonal = outputs[0]["hamiltonian_non_diagonal_blocks"].cpu()
            
            
                pred_ham_transformed = self.model.build_final_matrix(
                    batch_gpu,
                    pred_ham_diagonal,
                    pred_ham_non_diagonal,
                    transform=output_transform,
                    convention="e3nn_to_pyscf_def2svp",
                )

            # GPU batch 명시적으로 삭제
            del batch_gpu
            # torch.cuda.empty_cache()  # 선택사항: GPU 캐시 강정리
            return pred_ham_transformed
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

    def MLRKS(self, mol, gpu4pyscf=None, density_fit=None, model_length_unit="ang"):
        if self.model is None:
            raise ValueError("Model is not set")
        
        if gpu4pyscf is None:
            gpu4pyscf = self.gpu4pyscf
        global IMPORT_GPU4PYSCF
        if gpu4pyscf and not IMPORT_GPU4PYSCF:
            raise ImportError("gpu4pyscf is not installed. Please install it using 'pip install gpu4pyscf'.")
        
        if density_fit is None:
            density_fit = True

        # default
        # (1) if gpu4pyscf=False, density_fit=True
        # (2) if gpu4pyscf=True, density_fit=True

        start_time = time.time()
        self.mf = dft.RKS(mol)
        if density_fit:
            self.mf = self.mf.density_fit()
        if gpu4pyscf:
            self.mf = self.mf.to_gpu()
        self.time_dict["mf_time"] = time.time() - start_time

        start_time = time.time()
        self.ovlp = mol.intor("int1e_ovlp")
        self.time_dict["ovlp_time"] = time.time() - start_time
        # previous_xc = self.mf.xc

        start_time = time.time()
        self.mf.xc = "pbe,pbe"
        self.init_dm = self.mf.init_guess_by_minao()
        # self.init_dm = self.mf.get_init_guess(key="vsap")
        self.init_ham = self.mf.get_fock(dm=self.init_dm)
        # self.mf.xc = previous_xc
        self.time_dict["init_ham_time"] = time.time() - start_time

        atoms, coords = self._atoms_coords_from_mol(mol)
        self.atoms = atoms
        self.coords = coords
        if model_length_unit.lower() in ["ang", "angstrom", "a"]:
            input_coords = coords
        elif model_length_unit.lower() in ["bohr", "b"]:
            input_coords = coords * ANG2BOHR
        else:
            raise ValueError(f"Invalid model_length_unit: {model_length_unit}")
        start_time = time.time()
        self.pred_ham = self.forward_component(atoms, input_coords, self.ovlp, self.init_ham)
        self.time_dict["pred_ham_time"] = time.time() - start_time
        # self.dm0 = self.calc_dm0(atoms, coords, self.pred_ham)
        # self.mf.dm0 = self.dm0
        return self.mf

    @staticmethod
    def calc_density(mo_coeff, n_occ):
        """Calculate density matrix from molecular orbital energies and coefficients"""
        sliced_mo_coeff = mo_coeff[:, :n_occ]
        density = sliced_mo_coeff @ sliced_mo_coeff.T * 2
        return density

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

    def direct_energy_force(self, mol, calc_forces=True, mf=None, filtering=True, gt_tol=1e-8, pred_tol=1e-3, pad_eigval=None, gpu4pyscf=None, density_fit=None, model_length_unit="ang"):
        if gpu4pyscf is None:
            gpu4pyscf = self.gpu4pyscf
        if density_fit is None:
            density_fit = True

        if mf is None:
            mf = self.MLRKS(mol, gpu4pyscf=gpu4pyscf, density_fit=density_fit, model_length_unit=model_length_unit)

        mf.grids.level = 1
        pred_ham_transformed = torch.tensor(self.pred_ham)
        calc_overlap = torch.from_numpy(self.ovlp)
        n_occ = mol.nelectron // 2
        atoms = self.atoms
        coords = self.coords
        start_time = time.time()
        if filtering:
            pred_mo_energy_safe, pred_mo_coeff_safe = calc_mo_energy_and_coeff(
                pred_ham_transformed, calc_overlap, tol=gt_tol, pad_eigval=None
            )
            pred_mo_energy_unsatable, pred_mo_coeff_unsatable = calc_mo_energy_and_coeff(
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
            pred_mo_energy, pred_mo_coeff = calc_mo_energy_and_coeff(
                pred_ham_transformed, calc_overlap, tol=gt_tol, pad_eigval=pad_eigval
            )
        self.pred_dm0 = self.calc_density(pred_mo_coeff, n_occ)
        self.time_dict["pred_dm0_time"] = time.time() - start_time

        if gpu4pyscf:
            cur_density = asarray(self.pred_dm0)
        else:
            cur_density = self.pred_dm0
        
        start_time = time.time()
        energy = mf.energy_tot(cur_density)
        self.time_dict["energy_time"] = time.time() - start_time

        if calc_forces:
            start_time = time.time()
            grad_frame = mf.nuc_grad_method()

            mo_occ = mf.get_occ(pred_mo_energy, pred_mo_coeff)

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

        return energy, forces