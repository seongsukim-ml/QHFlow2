#!/usr/bin/env python3
"""
Final multi-process DFT calculation for MD17 dataset
- Loads data from MD17 DP format (coord.npy, energy.npy, force.npy, box.npy, type.raw)
- Multi-processing with optimal core usage
- Thread control (OMP_NUM_THREADS=1 per process)
- Uses existing train/val/test splits from MD17
- HDF5 storage for all matrices and results
- Complete data: energy, forces, overlap, density matrix, Hamiltonians, orbitals
- Saves each split separately and merges at the end (supports resume)

Usage:
    # Default: process all splits (train/val/test)
    python final_processing_md17.py --molecule benzene
    python final_processing_md17.py --molecule benzene --split all

    # Process only specific split
    python final_processing_md17.py --molecule aspirin --split train
    python final_processing_md17.py --molecule ethanol --split val --num_samples 1000

    # Process multiple splits
    python final_processing_md17.py --molecule toluene --split train val
    python final_processing_md17.py --molecule naphthalene --split train test

    # Custom settings with tag
    python final_processing_md17.py --molecule toluene --num_samples 5000 --n_processes 16 --tag exp1
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python final_processing_md17.py --molecule salicylic_acid --n_processes 63 --tag run1
    
    # GPU usage
    python final_processing_md17.py --molecule benzene --use_gpu --tag gpu_test
    python final_processing_md17.py --molecule aspirin --split train --use_gpu --n_processes 4
    
    # Actual usage with tag
    HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=0 python final_processing_md17.py --molecule aspirin --use_gpu --density_fit --n_processes 4 --tag v1
    HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=1 python final_processing_md17.py --molecule naphthalene --use_gpu --density_fit --n_processes 4 --tag v1
    HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=2 python final_processing_md17.py --molecule salicylic_acid --use_gpu --density_fit --n_processes 4 --tag v1
    HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=3 python final_processing_md17.py --molecule ethanol --use_gpu --density_fit --n_processes 4 --tag v1
    
    # Resume interrupted processing (will skip completed splits)
    python final_processing_md17.py --molecule benzene --tag exp1  # Will resume from where it stopped
    
    # Force reprocess even if files exist
    python final_processing_md17.py --molecule benzene --tag exp1 --force
    
    # Only merge existing split files (skip processing)
    python final_processing_md17.py --molecule benzene --tag exp1 --merge_only
"""

import numpy as np
import h5py
import sys
import argparse
import time
import os
import pickle
import lmdb
import json
import multiprocessing as mp
from multiprocessing import cpu_count
from tqdm import tqdm

# Add the source path
path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(path, 'src'))

from pyscf import gto, dft

# Try to import gpu4pyscf
try:
    import gpu4pyscf
    IMPORT_GPU4PYSCF = True
    print("gpu4pyscf is installed")
except ImportError:
    IMPORT_GPU4PYSCF = False
    print("gpu4pyscf is not installed")

HA2eV = 27.211396641308  
BOHR2ANG = 0.5291772105638411 

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Final DFT processing for MD17 dataset with multi-processing and HDF5 storage')
    parser.add_argument('--molecule', type=str, required=True,
                        help='Molecule name (aspirin, benzene, ethanol, malonaldehyde, naphthalene, salicylic_acid, toluene, uracil)')
    parser.add_argument('--split', type=str, nargs='+', default=['all'],
                        choices=['all', 'train', 'val', 'test'],
                        help='Which split(s) to process. Can specify multiple: --split train val, or use --split all for all splits (default: all)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process per split (default: all available)')
    parser.add_argument('--n_processes', type=int, default=1,
                        help='Number of processes (default: 1)')
    parser.add_argument('--conv_tol', type=float, default=1e-7,
                        help='SCF convergence tolerance (default: 1e-7)')
    parser.add_argument('--grid_level', type=int, default=3,
                        help='DFT grid level (default: 3)')
    parser.add_argument('--basis', type=str, default='def2-SVP',
                        help='Basis set (default: def2-SVP)')
    parser.add_argument('--functional', type=str, default='pbe',
                        help='DFT functional (default: pbe)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Tag for output folder name (default: None)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: current directory)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory for MD17 data (default: auto-detect)')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for DFT calculations (requires gpu4pyscf)')
    parser.add_argument('--density_fit', action='store_true',
                        help='Use density fitting (RI) for DFT calculations')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocess even if split files exist')
    parser.add_argument('--merge_only', action='store_true',
                        help='Only merge existing split files, skip processing')
    return parser.parse_args()

# Conversion constants
KCAL_TO_EV = 1.0 / 23.06054783  # kcal/mol to eV

def load_md17_data(molecule, split, data_root=None, num_samples=None):
    """
    Load MD17 data from LMDB format
    
    Args:
        molecule: molecule name
        split: 'train', 'val', or 'test'
        data_root: root directory for MD17 data
        num_samples: number of samples to load (None = all)
    
    Returns:
        dict with keys: coords, energies, forces, atomic_numbers
    """
    if data_root is None:
        # Auto-detect data root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(script_dir, '..', '..', '..', '..', 'data', 'mdsim_data', 'md17')
        data_root = os.path.abspath(data_root)
    
    # MD17 structure: molecule/10k/{split}/data.lmdb
    # Based on lmdb_dataset.py: use subdir=False for file-based LMDB
    lmdb_path = os.path.join(data_root, molecule, '10k', split, 'data.lmdb')
    
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f"MD17 LMDB file not found: {lmdb_path}")
    
    print(f"Loading MD17 data from: {lmdb_path}")
    
    # Open LMDB database using the same method as lmdb_dataset.py
    # subdir=False is key for file-based LMDB
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    
    # Collect all data from LMDB
    coords_list = []
    energies_list = []
    forces_list = []
    atomic_numbers = None
    
    # Get total number of entries (similar to lmdb_dataset.py)
    with env.begin() as txn:
        stat = env.stat()
        num_entries = stat['entries']
        print(f"  Total entries in LMDB: {num_entries}")
        
        # Read all entries using keys like "0", "1", "2", ... (as in lmdb_dataset.py)
        # Note: num_entries might include metadata, so we try to read until we can't find a key
        for idx in range(num_entries):
            key = f"{idx}".encode("ascii")
            value = txn.get(key)
            
            if value is None:
                # Stop if we can't find a key (might have reached the end)
                break
            
            data_obj = pickle.loads(value)
            
            # Data is a torch_geometric.data.Data object
            # Convert to numpy arrays
            if atomic_numbers is None:
                # Get atomic numbers from first sample
                if hasattr(data_obj, 'atomic_numbers'):
                    atomic_numbers = data_obj.atomic_numbers.cpu().numpy() if hasattr(data_obj.atomic_numbers, 'cpu') else np.array(data_obj.atomic_numbers)
                elif hasattr(data_obj, 'atoms'):
                    atomic_numbers = data_obj.atoms.cpu().numpy() if hasattr(data_obj.atoms, 'cpu') else np.array(data_obj.atoms)
                else:
                    # Try to_dict
                    data_dict = data_obj.to_dict() if hasattr(data_obj, 'to_dict') else {}
                    if 'atomic_numbers' in data_dict:
                        atomic_numbers = data_dict['atomic_numbers']
                        if hasattr(atomic_numbers, 'cpu'):
                            atomic_numbers = atomic_numbers.cpu().numpy()
                        atomic_numbers = np.array(atomic_numbers)
                    elif 'atoms' in data_dict:
                        atoms = data_dict['atoms']
                        if hasattr(atoms, 'cpu'):
                            atoms = atoms.cpu().numpy()
                        atomic_numbers = np.array(atoms)
                
                print(f"  Atomic numbers: {atomic_numbers} (shape: {atomic_numbers.shape})")
            
            # Extract positions
            if hasattr(data_obj, 'pos'):
                pos = data_obj.pos.cpu().numpy() if hasattr(data_obj.pos, 'cpu') else np.array(data_obj.pos)
            else:
                data_dict = data_obj.to_dict() if hasattr(data_obj, 'to_dict') else {}
                pos = data_dict['pos']
                if hasattr(pos, 'cpu'):
                    pos = pos.cpu().numpy()
                pos = np.array(pos)
            coords_list.append(pos)
            
            # Extract energy (stored as 'y' in torch_geometric Data)
            if hasattr(data_obj, 'y'):
                energy = data_obj.y
                if hasattr(energy, 'cpu'):
                    energy = energy.cpu().numpy()
                if hasattr(energy, 'item'):
                    energy = energy.item()
                energy = float(energy)
            elif hasattr(data_obj, 'energy'):
                energy = data_obj.energy
                if hasattr(energy, 'cpu'):
                    energy = energy.cpu().numpy()
                if hasattr(energy, 'item'):
                    energy = energy.item()
                energy = float(energy)
            else:
                data_dict = data_obj.to_dict() if hasattr(data_obj, 'to_dict') else {}
                energy = data_dict.get('y', data_dict.get('energy', None))
                if energy is not None:
                    if hasattr(energy, 'cpu'):
                        energy = energy.cpu().numpy()
                    if hasattr(energy, 'item'):
                        energy = energy.item()
                    energy = float(energy)
                else:
                    raise ValueError("Could not find energy in data object")
            energies_list.append(energy)
            
            # Extract forces
            if hasattr(data_obj, 'force'):
                force = data_obj.force.cpu().numpy() if hasattr(data_obj.force, 'cpu') else np.array(data_obj.force)
            else:
                data_dict = data_obj.to_dict() if hasattr(data_obj, 'to_dict') else {}
                force = data_dict['force']
                if hasattr(force, 'cpu'):
                    force = force.cpu().numpy()
                force = np.array(force)
            forces_list.append(force)
        
        n_frames = len(coords_list)
    
    env.close()
    
    # Convert to numpy arrays
    coords = np.array(coords_list)  # [n_frames, n_atoms, 3]
    energies = np.array(energies_list)  # [n_frames]
    forces = np.array(forces_list)  # [n_frames, n_atoms, 3]
    
    print(f"  Coord shape: {coords.shape}")
    print(f"  Energy shape: {energies.shape}")
    print(f"  Force shape: {forces.shape}")
    
    # Verify shapes
    n_atoms = coords.shape[1]
    assert energies.shape[0] == n_frames, f"Energy frames mismatch: {energies.shape[0]} != {n_frames}"
    assert forces.shape[0] == n_frames, f"Force frames mismatch: {forces.shape[0]} != {n_frames}"
    assert forces.shape[1] == n_atoms, f"Force atoms mismatch: {forces.shape[1]} != {n_atoms}"
    assert len(atomic_numbers) == n_atoms, f"Atomic numbers mismatch: {len(atomic_numbers)} != {n_atoms}"
    
    # Sample if requested
    if num_samples is not None and num_samples < n_frames:
        print(f"  Sampling {num_samples} from {n_frames} frames")
        indices = np.random.choice(n_frames, size=num_samples, replace=False)
        indices = np.sort(indices)
        coords = coords[indices]
        energies = energies[indices]
        forces = forces[indices]
        n_frames = num_samples
    
    print(f"  Loaded {n_frames} frames")
    print(f"  Energy range: [{energies.min():.6f}, {energies.max():.6f}] eV")
    
    return {
        'coords': coords,
        'energies': energies,
        'forces': forces,
        'atomic_numbers': atomic_numbers,
        'n_frames': n_frames,
        'n_atoms': n_atoms
    }

def calculate_single_sample(idx, z, pos, energy_ref, force_ref, settings):
    """
    Calculate DFT for a single sample
    Returns: dictionary with all calculated data
    """
    start_time = time.time()
    
    # Build molecule string
    atom_conf = []
    for i, (zi, ri) in enumerate(zip(z, pos)):
        atom_conf.append([int(zi), float(ri[0]), float(ri[1]), float(ri[2])])
        
    # Create molecule
    mol = gto.Mole()
    mol.build(
        spin=0,
        atom=atom_conf,
        basis=settings['basis'],
        unit='ang',
        verbose=0
    )
    
    # Setup DFT
    mf = dft.RKS(mol)
    
    # Apply density fitting if requested
    density_fit = settings.get('density_fit', False)
    if density_fit:
        mf = mf.density_fit()
    
    # Convert to GPU if requested
    use_gpu = settings.get('use_gpu', False)
    if use_gpu:
        if not IMPORT_GPU4PYSCF:
            raise ImportError("gpu4pyscf is not installed. Please install it using 'pip install gpu4pyscf'.")
        try:
            mf = mf.to_gpu()
        except Exception as e:
            # If GPU initialization fails, fall back to CPU
            print(f"Warning: Failed to initialize GPU for sample {idx}: {e}")
            print("Falling back to CPU mode for this sample")
            use_gpu = False
            settings['use_gpu'] = False
    
    mf.xc = settings['functional']
    mf.verbose = 0
    mf.conv_tol = settings['conv_tol']
    mf.grids.level = settings['grid_level']
    mf.grids.prune = None
    mf.small_rho_cutoff = 1e-12
    
    # Helper function to convert GPU arrays to numpy
    def to_numpy(arr):
        """Convert GPU array (cupy) to numpy array if needed"""
        if hasattr(arr, 'get'):
            return np.array(arr.get())
        return np.array(arr)
    
    # Get initial overlap and Hamiltonian
    ovlp = mf.get_ovlp()
    h_core = mf.get_hcore()
    
    # Get initial density matrices and Hamiltonians
    init_dm_minao = mf.get_init_guess(key='minao')
    init_ham_minao = mf.get_fock(dm=init_dm_minao)
    
    init_dm_1e = mf.get_init_guess(key='1e')
    init_ham_1e = mf.get_fock(dm=init_dm_1e)
    
    # Run SCF
    E_dft = mf.kernel()
    converged = mf.converged
    
    # Get density matrix
    dm = mf.make_rdm1()
    
    # Get Fock matrix (Hamiltonian)
    hamiltonian = mf.get_fock(dm=dm)
    
    # Get orbital energies and coefficients
    orbital_energies = mf.mo_energy
    orbital_coefficients = mf.mo_coeff
    
    # Calculate forces
    grad_obj = mf.nuc_grad_method()
    forces = -grad_obj.kernel()  # Hartree/Bohr
    
    # Get additional energy components
    e_nuc = mf.energy_nuc()  # Nuclear repulsion
    e_tot = mf.e_tot  # Total energy
    
    # Get grid info
    ngrids = mf.grids.weights.size if hasattr(mf.grids, 'weights') else 0
    
    # Convert GPU arrays to numpy arrays
    if use_gpu:
        ovlp = to_numpy(ovlp)
        h_core = to_numpy(h_core)
        init_ham_minao = to_numpy(init_ham_minao)
        init_ham_1e = to_numpy(init_ham_1e)
        dm = to_numpy(dm)
        hamiltonian = to_numpy(hamiltonian)
        orbital_energies = to_numpy(orbital_energies)
        orbital_coefficients = to_numpy(orbital_coefficients)
        forces = to_numpy(forces)
    
    calc_time = time.time() - start_time
    
    # MD17 energies are in eV, forces are in eV/Angstrom
    # Prepare result dictionary
    result = {
        'idx': idx,
        'success': True,
        'converged': converged,
        'calc_time': calc_time,
        
        # Input data
        'positions': pos,  # Angstrom
        'atomic_numbers': z,
        
        # Reference data (MD17 - already in eV)
        'energy_ref_ev': energy_ref,  # eV
        'forces_ref_ev': force_ref,  # eV/Angstrom
        
        # Calculated data
        'energy_ha': E_dft,  # Hartree
        'energy_ev': E_dft * HA2eV,  # eV
        'forces_ha_bohr': forces,  # Hartree/Bohr
        'forces_ev_ang': forces * (HA2eV / BOHR2ANG),  # eV/Angstrom
        
        # Energy components
        'energy_nuclear': e_nuc,  # Hartree
        
        # Matrices
        'overlap': ovlp,
        'initial_hamiltonian_minao': init_ham_minao,
        'initial_hamiltonian_1e': init_ham_1e,
        'h_core': h_core,
        'density_matrix': dm,
        'hamiltonian': hamiltonian,
        
        # Orbitals
        'orbital_energies': orbital_energies,
        'orbital_coefficients': orbital_coefficients,
        
        # Errors
        'error_energy_ev': abs(E_dft * HA2eV - energy_ref),
        'error_energy_mev': abs(E_dft * HA2eV - energy_ref) * 1000,
        'error_forces_mae': np.mean(np.abs(forces * (HA2eV / BOHR2ANG) - force_ref)),
        
        # Metadata
        'ngrids': ngrids,
        'nbasis': ovlp.shape[0],
        'natoms': len(z),
    }
    
    return result

def init_worker_mp(use_gpu_flag):
    """Initialize worker process for multiprocessing
    Sets threading environment variables only.
    GPU initialization is done lazily in calculate_single_sample when needed.
    """
    # Set threading environment variables
    threading_env_vars = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1"
    }
    os.environ.update(threading_env_vars)
    
    # Note: GPU initialization is done lazily in calculate_single_sample
    # to avoid CUDA initialization errors in multiprocessing

def worker_wrapper(idx_and_data):
    """Wrapper for pool.map"""
    idx, z, pos, energy_ref, force_ref, settings = idx_and_data
    return calculate_single_sample(idx, z, pos, energy_ref, force_ref, settings)

def save_first_sample_to_txt(result, split, molecule, output_dir, config):
    """
    Save first sample result to text file
    
    Args:
        result: result dictionary from calculate_single_sample
        split: split name (train/val/test)
        molecule: molecule name
        output_dir: output directory for txt file
        config: configuration dictionary
    """
    if not result['success']:
        return
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate txt filename
    txt_filename = f"first_sample_{split}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"First Sample Result: {molecule} - {split.upper()} Split\n")
        f.write("=" * 80 + "\n\n")
        
        # Metadata
        f.write("Configuration:\n")
        f.write(f"  Molecule:      {molecule}\n")
        f.write(f"  Split:         {split}\n")
        f.write(f"  Functional:    {config['functional'].upper()}\n")
        f.write(f"  Basis:         {config['basis']}\n")
        f.write(f"  Conv_tol:      {config['conv_tol']:.0e}\n")
        f.write(f"  Grid level:    {config['grid_level']}\n")
        f.write(f"  Sample index:  {result['idx']}\n")
        f.write("\n")
        
        # Calculation status
        f.write("Calculation Status:\n")
        f.write(f"  Success:       {result['success']}\n")
        f.write(f"  Converged:     {result['converged']}\n")
        f.write(f"  Calc time:     {result['calc_time']:.4f} seconds\n")
        f.write("\n")
        
        # System information
        f.write("System Information:\n")
        f.write(f"  Number of atoms:  {result['natoms']}\n")
        f.write(f"  Number of basis:  {result['nbasis']}\n")
        f.write(f"  Number of grids:  {result['ngrids']}\n")
        f.write("\n")
        
        # Atomic numbers
        f.write("Atomic Numbers:\n")
        atomic_numbers = result['atomic_numbers']
        f.write(f"  {atomic_numbers}\n")
        f.write("\n")
        
        # Positions
        f.write("Atomic Positions (Angstrom):\n")
        positions = result['positions']
        for i, (z, pos) in enumerate(zip(atomic_numbers, positions)):
            f.write(f"  Atom {i+1:3d} (Z={z:2d}):  {pos[0]:15.10f}  {pos[1]:15.10f}  {pos[2]:15.10f}\n")
        f.write("\n")
        
        # Energies
        f.write("Energies:\n")
        f.write(f"  Reference (MD17):     {result['energy_ref_ev']:20.10f} eV\n")
        f.write(f"  DFT calculated:       {result['energy_ev']:20.10f} eV\n")
        f.write(f"  DFT calculated:       {result['energy_ha']:20.10f} Hartree\n")
        f.write(f"  Nuclear repulsion:    {result['energy_nuclear']:20.10f} Hartree\n")
        f.write(f"  Energy error:         {result['error_energy_ev']:20.10f} eV\n")
        f.write(f"  Energy error:         {result['error_energy_mev']:20.10f} meV\n")
        f.write("\n")
        
        # Forces
        f.write("Forces (eV/Angstrom):\n")
        f.write("  Reference (MD17) vs DFT calculated:\n")
        forces_ref = result['forces_ref_ev']
        forces_calc = result['forces_ev_ang']
        for i in range(len(atomic_numbers)):
            f.write(f"  Atom {i+1:3d} (Z={atomic_numbers[i]:2d}):\n")
            f.write(f"    Ref:  {forces_ref[i,0]:15.10f}  {forces_ref[i,1]:15.10f}  {forces_ref[i,2]:15.10f}\n")
            f.write(f"    DFT:  {forces_calc[i,0]:15.10f}  {forces_calc[i,1]:15.10f}  {forces_calc[i,2]:15.10f}\n")
            diff = forces_calc[i] - forces_ref[i]
            f.write(f"    Diff: {diff[0]:15.10f}  {diff[1]:15.10f}  {diff[2]:15.10f}\n")
        f.write(f"  Force MAE: {result['error_forces_mae']:20.10f} eV/Angstrom\n")
        f.write("\n")
        
        # Orbital energies
        f.write("Orbital Energies (eV):\n")
        orbital_energies = result['orbital_energies'] * HA2eV  # Convert to eV
        n_show = min(20, len(orbital_energies))  # Show first 20
        for i, e in enumerate(orbital_energies[:n_show]):
            f.write(f"  Orbital {i+1:3d}:  {e:15.10f} eV\n")
        if len(orbital_energies) > n_show:
            f.write(f"  ... (showing first {n_show} of {len(orbital_energies)} orbitals)\n")
        f.write("\n")
        
        # Matrices (summary only)
        f.write("Matrix Information:\n")
        overlap = result['overlap']
        hamiltonian = result['hamiltonian']
        density_matrix = result['density_matrix']
        
        f.write(f"  Overlap matrix:                {overlap.shape}, trace = {np.trace(overlap):15.10f}\n")
        f.write(f"  Hamiltonian matrix:            {hamiltonian.shape}, trace = {np.trace(hamiltonian):15.10f} Hartree\n")
        f.write(f"  Density matrix:                {density_matrix.shape}, trace = {np.trace(density_matrix):15.10f}\n")
        f.write(f"  Initial Hamiltonian (minao):   {result['initial_hamiltonian_minao'].shape}, trace = {np.trace(result['initial_hamiltonian_minao']):15.10f} Hartree\n")
        f.write(f"  Initial Hamiltonian (1e):      {result['initial_hamiltonian_1e'].shape}, trace = {np.trace(result['initial_hamiltonian_1e']):15.10f} Hartree\n")
        f.write(f"  H_core matrix:                  {result['h_core'].shape}, trace = {np.trace(result['h_core']):15.10f} Hartree\n")
        f.write("\n")
        
        # Overlap matrix (first 10x10 block)
        f.write("Overlap Matrix (first 10x10 block):\n")
        n_block = min(10, overlap.shape[0])
        f.write("     " + " ".join([f"{i+1:10d}" for i in range(n_block)]) + "\n")
        for i in range(n_block):
            f.write(f"{i+1:4d} " + " ".join([f"{overlap[i,j]:10.6f}" for j in range(n_block)]) + "\n")
        if overlap.shape[0] > n_block:
            f.write(f"  ... (showing first {n_block}x{n_block} of {overlap.shape[0]}x{overlap.shape[0]} matrix)\n")
        f.write("\n")
        
        # Hamiltonian matrix (first 10x10 block)
        f.write("Hamiltonian Matrix (first 10x10 block, Hartree):\n")
        f.write("     " + " ".join([f"{i+1:10d}" for i in range(n_block)]) + "\n")
        for i in range(n_block):
            f.write(f"{i+1:4d} " + " ".join([f"{hamiltonian[i,j]:10.6f}" for j in range(n_block)]) + "\n")
        if hamiltonian.shape[0] > n_block:
            f.write(f"  ... (showing first {n_block}x{n_block} of {hamiltonian.shape[0]}x{hamiltonian.shape[0]} matrix)\n")
        f.write("\n")
        
        # Density matrix (first 10x10 block)
        f.write("Density Matrix (first 10x10 block):\n")
        f.write("     " + " ".join([f"{i+1:10d}" for i in range(n_block)]) + "\n")
        for i in range(n_block):
            f.write(f"{i+1:4d} " + " ".join([f"{density_matrix[i,j]:10.6f}" for j in range(n_block)]) + "\n")
        if density_matrix.shape[0] > n_block:
            f.write(f"  ... (showing first {n_block}x{n_block} of {density_matrix.shape[0]}x{density_matrix.shape[0]} matrix)\n")
        f.write("\n")
        
        # Initial Hamiltonian (minao) (first 10x10 block)
        f.write("Initial Hamiltonian (minao) (first 10x10 block, Hartree):\n")
        init_ham_minao = result['initial_hamiltonian_minao']
        f.write("     " + " ".join([f"{i+1:10d}" for i in range(n_block)]) + "\n")
        for i in range(n_block):
            f.write(f"{i+1:4d} " + " ".join([f"{init_ham_minao[i,j]:10.6f}" for j in range(n_block)]) + "\n")
        if init_ham_minao.shape[0] > n_block:
            f.write(f"  ... (showing first {n_block}x{n_block} of {init_ham_minao.shape[0]}x{init_ham_minao.shape[0]} matrix)\n")
        f.write("\n")
        
        # Initial Hamiltonian (1e) (first 10x10 block)
        f.write("Initial Hamiltonian (1e) (first 10x10 block, Hartree):\n")
        init_ham_1e = result['initial_hamiltonian_1e']
        f.write("     " + " ".join([f"{i+1:10d}" for i in range(n_block)]) + "\n")
        for i in range(n_block):
            f.write(f"{i+1:4d} " + " ".join([f"{init_ham_1e[i,j]:10.6f}" for j in range(n_block)]) + "\n")
        if init_ham_1e.shape[0] > n_block:
            f.write(f"  ... (showing first {n_block}x{n_block} of {init_ham_1e.shape[0]}x{init_ham_1e.shape[0]} matrix)\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")
    
    print(f"  ✓ Saved first sample info to: {txt_path}")

def save_config(config, output_dir):
    """Save configuration to JSON file"""
    config_path = os.path.join(output_dir, 'config.json')
    
    # Make a copy and convert non-serializable types
    config_to_save = config.copy()
    
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"✓ Saved configuration to: {config_path}")

def load_config(output_dir):
    """Load configuration from JSON file"""
    config_path = os.path.join(output_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def check_split_complete(split_h5_path):
    """Check if a split file exists and is complete"""
    if not os.path.exists(split_h5_path):
        return False, 0
    
    try:
        with h5py.File(split_h5_path, 'r') as f:
            if 'metadata' not in f:
                return False, 0
            if 'complete' not in f['metadata'].attrs:
                return False, 0
            if not f['metadata'].attrs['complete']:
                return False, 0
            n_samples = f['metadata'].attrs.get('n_samples', 0)
            return True, n_samples
    except Exception as e:
        print(f"Warning: Error reading {split_h5_path}: {e}")
        return False, 0

def save_split_to_h5(results, stats, split, output_dir, config):
    """Save a single split's results to HDF5 file"""
    split_h5_path = os.path.join(output_dir, f'{split}.h5')
    
    print(f"\n{'='*80}")
    print(f"Saving {split} split to: {split_h5_path}")
    print(f"{'='*80}\n")
    
    with h5py.File(split_h5_path, 'w') as f:
        # Save metadata
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['dataset'] = 'MD17'
        meta_grp.attrs['molecule'] = config['molecule']
        meta_grp.attrs['split'] = split
        meta_grp.attrs['functional'] = config['functional']
        meta_grp.attrs['basis'] = config['basis']
        meta_grp.attrs['conv_tol'] = config['conv_tol']
        meta_grp.attrs['grid_level'] = config['grid_level']
        meta_grp.attrs['n_processes'] = config['n_processes']
        meta_grp.attrs['use_gpu'] = config['use_gpu']
        meta_grp.attrs['density_fit'] = config['density_fit']
        meta_grp.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        meta_grp.attrs['n_samples'] = len(results)
        meta_grp.attrs['n_success'] = stats['n_success']
        meta_grp.attrs['n_converged'] = stats['n_converged']
        meta_grp.attrs['total_time_sec'] = stats['total_time']
        
        if stats['n_success'] > 0:
            meta_grp.attrs['mean_error_ev'] = np.mean(stats['errors_ev'])
            meta_grp.attrs['mean_error_mev'] = np.mean(stats['errors_mev'])
            meta_grp.attrs['mean_force_mae'] = np.mean(stats['force_maes'])
        
        # Save samples
        samples_grp = f.create_group('samples')
        for i, result in enumerate(tqdm(results, desc=f"Saving {split}", unit="sample", ncols=100)):
            sample_grp = samples_grp.create_group(f'sample_{i}')
            
            # Save all data from result dictionary
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    sample_grp.create_dataset(key, data=value, compression='gzip')
                elif isinstance(value, (int, float, bool, str)):
                    sample_grp.attrs[key] = value
                elif value is None:
                    sample_grp.attrs[key] = 'None'
        
        # Mark as complete
        meta_grp.attrs['complete'] = True
    
    file_size_mb = os.path.getsize(split_h5_path) / (1024 * 1024)
    print(f"\n✓ Saved {split} split to {split_h5_path} ({file_size_mb:.2f} MB)")

def process_split(molecule, split, data_root, settings, config, output_dir):
    """Process a single split (train/val/test) and save to individual file"""
    print(f"\n{'='*80}")
    print(f"Processing {split.upper()} split for {molecule}")
    print(f"{'='*80}\n")
    
    # Load data
    data = load_md17_data(molecule, split, data_root, config.get('num_samples'))
    
    coords = data['coords']  # Already in shape [n_frames, n_atoms, 3]
    energies = data['energies']
    forces = data['forces']  # Already in shape [n_frames, n_atoms, 3]
    atomic_numbers = data['atomic_numbers']
    n_frames = data['n_frames']
    
    # Prepare work items
    work_items = [
        (i, atomic_numbers, coords[i], energies[i], forces[i], settings)
        for i in range(n_frames)
    ]
    
    print(f"\n{'='*80}")
    print(f"Starting parallel processing for {split}...")
    print(f"{'='*80}\n")
    
    # Run parallel processing
    start_time = time.time()
    n_processes = config['n_processes']
    
    results = []
    if n_processes == 1:
        # Sequential processing
        with tqdm(total=len(work_items),
                  desc=f"DFT Calculations ({split})",
                  unit="sample",
                  ncols=100,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for item in work_items:
                result = worker_wrapper(item)
                results.append(result)
                if result['success']:
                    error_mev = result.get('error_energy_mev', 0)
                    pbar.set_postfix({'Last_error': f'{error_mev:.1f}meV',
                                      'Converged': result.get('converged', False)})
                else:
                    pbar.set_postfix({'Status': 'FAILED'})
                pbar.update(1)
    else:
        # Parallel processing
        # Use spawn context to avoid CUDA initialization errors (same as measure_energy_force_uma_gpu_v5.py)
        ctx = mp.get_context('spawn')
        initargs = (settings.get('use_gpu', False),) if settings.get('use_gpu', False) else ()
        with ctx.Pool(processes=n_processes, initializer=init_worker_mp, initargs=initargs) as pool:
            with tqdm(total=len(work_items),
                      desc=f"DFT Calculations ({split})",
                      unit="sample",
                      ncols=100,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for result in pool.imap_unordered(worker_wrapper, work_items, chunksize=1):
                    results.append(result)
                    if result['success']:
                        error_mev = result.get('error_energy_mev', 0)
                        pbar.set_postfix({'Last_error': f'{error_mev:.1f}meV',
                                         'Converged': result.get('converged', False)})
                    else:
                        pbar.set_postfix({'Status': 'FAILED'})
                    pbar.update(1)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Processing complete for {split}!")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Rate: {len(work_items)/total_time:.2f} samples/s")
    print(f"{'='*80}\n")
    
    # Analyze results
    n_success = sum(1 for r in results if r['success'])
    n_converged = sum(1 for r in results if r.get('converged', False))
    
    print(f"Results Summary ({split}):")
    print(f"  Successful: {n_success}/{len(results)}")
    print(f"  Converged:  {n_converged}/{len(results)}")
    
    errors_ev = []
    errors_mev = []
    force_maes = []
    
    if n_success > 0:
        success_results = [r for r in results if r['success']]
        errors_ev = [r['error_energy_ev'] for r in success_results]
        errors_mev = [r['error_energy_mev'] for r in success_results]
        force_maes = [r['error_forces_mae'] for r in success_results]
        
        print(f"\nEnergy Errors (eV):")
        print(f"  Mean:   {np.mean(errors_ev):.6f} eV = {np.mean(errors_mev):.2f} meV")
        print(f"  Median: {np.median(errors_ev):.6f} eV = {np.median(errors_mev):.2f} meV")
        print(f"  Min:    {np.min(errors_ev):.6f} eV = {np.min(errors_mev):.2f} meV")
        print(f"  Max:    {np.max(errors_ev):.6f} eV = {np.max(errors_mev):.2f} meV")
        
        print(f"\nForce Errors (MAE in eV/Å):")
        print(f"  Mean:   {np.mean(force_maes):.6f}")
        print(f"  Median: {np.median(force_maes):.6f}")
        print(f"  Min:    {np.min(force_maes):.6f}")
        print(f"  Max:    {np.max(force_maes):.6f}")
    
    stats = {
        'n_success': n_success,
        'n_converged': n_converged,
        'total_time': total_time,
        'errors_ev': errors_ev,
        'errors_mev': errors_mev,
        'force_maes': force_maes,
    }
    
    # Save split to individual file
    save_split_to_h5(results, stats, split, output_dir, config)
    
    # Save first sample to txt
    if len(results) > 0 and results[0]['success']:
        save_first_sample_to_txt(results[0], split, molecule, output_dir, config)
    
    return stats

def merge_splits(output_dir, splits, config):
    """Merge individual split files into a combined HDF5 file"""
    combined_path = os.path.join(output_dir, 'combined.h5')
    
    print(f"\n{'='*80}")
    print(f"Merging splits into: {combined_path}")
    print(f"{'='*80}\n")
    
    with h5py.File(combined_path, 'w') as f_out:
        # Save metadata
        meta_grp = f_out.create_group('metadata')
        meta_grp.attrs['dataset'] = 'MD17'
        meta_grp.attrs['molecule'] = config['molecule']
        meta_grp.attrs['functional'] = config['functional']
        meta_grp.attrs['basis'] = config['basis']
        meta_grp.attrs['conv_tol'] = config['conv_tol']
        meta_grp.attrs['grid_level'] = config['grid_level']
        meta_grp.attrs['n_processes'] = config['n_processes']
        meta_grp.attrs['use_gpu'] = config['use_gpu']
        meta_grp.attrs['density_fit'] = config['density_fit']
        meta_grp.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        if config.get('tag'):
            meta_grp.attrs['tag'] = config['tag']
        
        # Track total samples
        total_samples = 0
        total_success = 0
        total_converged = 0
        
        # Merge each split
        splits_grp = f_out.create_group('splits')
        
        for split in splits:
            split_h5_path = os.path.join(output_dir, f'{split}.h5')
            
            if not os.path.exists(split_h5_path):
                print(f"  ⚠ Skipping {split} (file not found)")
                continue
            
            print(f"  Merging {split}...")
            
            with h5py.File(split_h5_path, 'r') as f_in:
                # Get split metadata
                n_samples = f_in['metadata'].attrs.get('n_samples', 0)
                n_success = f_in['metadata'].attrs.get('n_success', 0)
                n_converged = f_in['metadata'].attrs.get('n_converged', 0)
                total_time = f_in['metadata'].attrs.get('total_time_sec', 0)
                
                total_samples += n_samples
                total_success += n_success
                total_converged += n_converged
                
                # Create split group
                split_grp = splits_grp.create_group(split)
                split_grp.attrs['n_total'] = n_samples
                split_grp.attrs['n_success'] = n_success
                split_grp.attrs['n_converged'] = n_converged
                split_grp.attrs['total_time_sec'] = total_time
                
                # Copy optional attributes
                for attr in ['mean_error_ev', 'mean_error_mev', 'mean_force_mae']:
                    if attr in f_in['metadata'].attrs:
                        split_grp.attrs[attr] = f_in['metadata'].attrs[attr]
                
                # Copy samples with progress bar
                if 'samples' in f_in:
                    for sample_name in tqdm(f_in['samples'].keys(), 
                                           desc=f"  Copying {split}",
                                           unit="sample", ncols=100):
                        f_in.copy(f'samples/{sample_name}', split_grp, name=sample_name)
            
            print(f"    ✓ {split}: {n_samples} samples")
        
        # Update total counts
        meta_grp.attrs['total_samples'] = total_samples
        meta_grp.attrs['total_success'] = total_success
        meta_grp.attrs['total_converged'] = total_converged
    
    file_size_mb = os.path.getsize(combined_path) / (1024 * 1024)
    print(f"\n✓ Created combined file: {combined_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Total samples: {total_samples}")

def main():
    args = parse_arguments()
    
    # Set default number of processes
    if args.n_processes is None:
        args.n_processes = max(1, cpu_count() - 1)
    
    # Process split argument: handle 'all' and multiple splits
    valid_splits = ['train', 'val', 'test']
    if 'all' in args.split:
        splits_to_process = valid_splits.copy()
    else:
        # Remove duplicates and validate
        splits_to_process = list(dict.fromkeys(args.split))  # Preserves order, removes duplicates
        invalid_splits = [s for s in splits_to_process if s not in valid_splits]
        if invalid_splits:
            raise ValueError(f"Invalid split(s): {invalid_splits}. Valid splits are: {valid_splits}")
    
    # Generate output directory name
    if args.output_dir is None:
        args.output_dir = '.'
    
    # Create folder name with tag
    folder_name_parts = ['md17', args.molecule]
    if args.tag:
        folder_name_parts.append(args.tag)
    folder_name_parts.append(f"conv{args.conv_tol:.0e}")
    folder_name_parts.append(f"grid{args.grid_level}")
    if args.num_samples:
        folder_name_parts.append(f"n{args.num_samples}")
    
    folder_name = '_'.join(folder_name_parts)
    output_dir = os.path.join(args.output_dir, folder_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Final DFT Processing for MD17 Dataset")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Molecule:      {args.molecule}")
    print(f"  Split(s):      {', '.join(splits_to_process)}")
    if args.num_samples:
        print(f"  Samples/split: {args.num_samples}")
    print(f"  Functional:    {args.functional.upper()}")
    print(f"  Basis:         {args.basis}")
    print(f"  Conv_tol:      {args.conv_tol:.0e}")
    print(f"  Grid level:    {args.grid_level}")
    print(f"  Use GPU:       {args.use_gpu}")
    print(f"  Density fit:   {args.density_fit}")
    if args.tag:
        print(f"  Tag:           {args.tag}")
    print(f"\nParallel Processing:")
    print(f"  CPU cores:     {cpu_count()}")
    print(f"  Processes:     {args.n_processes}")
    print(f"  Threads/proc:  1 (OMP_NUM_THREADS=1)")
    print(f"\nOutput:")
    print(f"  Directory:     {output_dir}")
    print(f"  Force:         {args.force}")
    print(f"  Merge only:    {args.merge_only}")
    print("=" * 80)
    
    # Check GPU availability
    if args.use_gpu:
        if not IMPORT_GPU4PYSCF:
            print("WARNING: --use_gpu specified but gpu4pyscf is not installed.")
            print("         Falling back to CPU. Install gpu4pyscf to use GPU.")
            args.use_gpu = False
        else:
            print("GPU mode enabled (gpu4pyscf)")
            if args.n_processes > 1:
                print("WARNING: GPU mode with multiple processes may cause CUDA initialization errors.")
                print("         Each process will try to use the same GPU, which can cause conflicts.")
                print("         Consider using --n_processes 1 for GPU calculations.")
                print("         Or use CUDA_VISIBLE_DEVICES to assign different GPUs to different processes.")
                print("         Example: CUDA_VISIBLE_DEVICES=0 python ... --n_processes 1")
    
    # Prepare configuration
    config = {
        'molecule': args.molecule,
        'splits': splits_to_process,
        'num_samples': args.num_samples,
        'n_processes': args.n_processes,
        'basis': args.basis,
        'functional': args.functional,
        'conv_tol': args.conv_tol,
        'grid_level': args.grid_level,
        'use_gpu': args.use_gpu,
        'density_fit': args.density_fit,
        'tag': args.tag,
        'output_dir': output_dir,
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    # Save configuration
    save_config(config, output_dir)
    
    # Prepare settings for DFT calculations
    settings = {
        'basis': args.basis,
        'functional': args.functional,
        'conv_tol': args.conv_tol,
        'grid_level': args.grid_level,
        'use_gpu': args.use_gpu,
        'density_fit': args.density_fit,
    }
    
    # If merge_only, skip processing
    if args.merge_only:
        print("\n*** Merge only mode - skipping DFT calculations ***\n")
    else:
        # Process each split
        for split in splits_to_process:
            split_h5_path = os.path.join(output_dir, f'{split}.h5')
            
            # Check if split is already complete
            is_complete, n_existing = check_split_complete(split_h5_path)
            
            if is_complete and not args.force:
                print(f"\n{'='*80}")
                print(f"✓ {split.upper()} split already complete ({n_existing} samples)")
                print(f"  File: {split_h5_path}")
                print(f"  Use --force to reprocess")
                print(f"{'='*80}\n")
                continue
            
            try:
                process_split(args.molecule, split, args.data_root, settings, config, output_dir)
            except FileNotFoundError as e:
                print(f"ERROR: {e}")
                print(f"Skipping {split} split...")
                continue
    
    # Merge all splits into combined file
    merge_splits(output_dir, splits_to_process, config)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("Processing Complete!")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f}: {size_mb:.2f} MB")
    print(f"\n{'='*80}")

if __name__ == '__main__':
    main()
