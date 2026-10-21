import sys
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
sys.path.append("/root/limlab01/kaistai/25DFT/QHFlow/src")
# import md.scflow_calculator_gpu
from md.scflow_calculator_gpu import SCFlowRKSCalculator, RKSCalculator
from dft_process.dft_process_utils import *
from common.draw_util import *

import time
import json
import numpy as np
from ase.io import read
from tqdm import tqdm

# Setup experiment and calculator
qh9_experiment = QH9Experiment(data_index=1, use_shard=True)
qh9_test_set = qh9_experiment.test_set

small_ckpt = "/root/25DFT/QHFlow/src/outputs/QH9Stable-random/QHFlow_so2_v5_1_small-QH9Stable-random/checkpoints/weights-epoch=78-val_loss=0.0000000.ckpt"
middle_ckpt = "/root/25DFT/QHFlow/src/outputs/QH9Stable-random/QHFlow_so2_v5_1_middle-QH9Stable-random/checkpoints/weights-epoch=78-val_loss=0.0000000.ckpt"
large_ckpt = "/root/25DFT/QHFlow/src/outputs/QH9Stable-random/QHFlow_so2_v5_1_large-QH9Stable-random/checkpoints/weights-epoch=78-val_loss=0.0000000.ckpt"
# extra_large_ckpt = "/root/25DFT/QHFlow/src/outputs/QH9Stable-random/QHFlow_so2_v5_1_extra_large_260k-QH9Stable-random/checkpoints/weights-epoch=78-val_loss=0.0000000.ckpt"

# cur_ckpt = "/root/25DFT/QHFlow/src/outputs/QH9Stable-random/QHFlow_so2_v5_1_extra_large_260k-QH9Stable-random/checkpoints/weights-epoch=78-val_loss=0.0000000.ckpt"
# model_name = "extra_large_260k"

print(f"Loading model from {cur_ckpt}")
# Initialize calculators
sccalculator = SCFlowRKSCalculator(basis="def2-SVP", functional="b3lyp")

sccalculator.set_model(
    cur_ckpt, 
    data_type="qh9", 
    units="ang", 
    model_length_unit="ang", 
    mf_init_functional="b3lyp",
    filtering=True,
    gt_tol=1e-8,
    pred_tol=1e-3,
    pad_eigval=1,
)
print(f"Setting model")
print(f"Setting max radius to 1000")
sccalculator.model.model.max_radius = 1000
print(f"Setting max num neighbors to 200")
sccalculator.model.model.max_num_neighbors = 200
print(f"Setting RKS calculator")
rkscalculator = RKSCalculator(basis="def2-SVP", functional="b3lyp", density_fit=False)

# Load and sort xyz files by atom count
xyz_folder = "/root/25DFT/QHFlow/qhflow_md_yh/yh_dft_test/xyz_files"
file_lists = os.listdir(xyz_folder)

def get_atom_count(xyz_file_path):
    """Extract atom count from the first line of xyz file"""
    with open(xyz_file_path, 'r') as f:
        first_line = f.readline().strip()
        return int(first_line)

# Store filename, atom count, and file path as tuples
file_atom_pairs = []
for filename in file_lists:
    if filename.endswith('.xyz'):
        file_path = os.path.join(xyz_folder, filename)
        try:
            atom_count = get_atom_count(file_path)
            file_atom_pairs.append((filename, atom_count, file_path))
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Sort by atom count (ascending order)
sorted_files = sorted(file_atom_pairs, key=lambda x: x[1])

# Display sorted results
print(f"Total {len(sorted_files)} files sorted by atom count:\n")
for filename, atom_count, file_path in sorted_files:
    print(f"{filename}: {atom_count} atoms")

# Extract sorted file names and paths
sorted_file_names = [filename for filename, _, _ in sorted_files]
sorted_file_paths = [file_path for _, _, file_path in sorted_files]

# Load xyz files using ASE
loaded_xyz_structures = []
for file_path in sorted_file_paths:
    try:
        atoms = read(file_path)
        loaded_xyz_structures.append(atoms)
        print(f"Loaded: {os.path.basename(file_path)} - {len(atoms)} atoms")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

print(f"\nSuccessfully loaded {len(loaded_xyz_structures)} xyz files")

# Helper function to convert numpy arrays to lists for JSON serialization
def to_list(arr):
    """Convert numpy array to list, handling None"""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr

# Helper function to extract atom configuration from ASE Atoms object
def get_atom_configuration(atoms):
    """Extract atomic numbers and positions from ASE Atoms object"""
    return {
        "atomic_numbers": atoms.get_atomic_numbers().tolist(),
        "positions": atoms.get_positions().tolist(),
        "symbols": atoms.get_chemical_symbols()
    }

def calculate_orbital_coefficient_similarity(mo_coeff1, mo_coeff2, overlap_matrix=None):
    """
    Calculate similarity between two sets of molecular orbital coefficients.
    
    If overlap_matrix is provided, calculates overlap: tr(C1^T @ S @ C2)
    Otherwise, calculates cosine similarity for each orbital and averages.
    
    Returns:
        similarity: float between -1 and 1 (or potentially higher for overlap)
    """
    if mo_coeff1 is None or mo_coeff2 is None:
        return None
    
    try:
        mo_coeff1 = np.array(mo_coeff1)
        mo_coeff2 = np.array(mo_coeff2)
        
        if mo_coeff1.shape != mo_coeff2.shape:
            return None
        
        # Calculate overlap if overlap matrix is provided
        if overlap_matrix is not None:
            try:
                S = np.array(overlap_matrix)
                # Calculate overlap: tr(C1^T @ S @ C2)
                overlap = np.trace(mo_coeff1.T @ S @ mo_coeff2)
                # Normalize by the norms
                norm1 = np.sqrt(np.trace(mo_coeff1.T @ S @ mo_coeff1))
                norm2 = np.sqrt(np.trace(mo_coeff2.T @ S @ mo_coeff2))
                if norm1 > 0 and norm2 > 0:
                    similarity = overlap / (norm1 * norm2)
                else:
                    similarity = None
            except:
                similarity = None
        else:
            # Fallback: cosine similarity (flatten and compute)
            coeff1_flat = mo_coeff1.flatten()
            coeff2_flat = mo_coeff2.flatten()
            dot_product = np.dot(coeff1_flat, coeff2_flat)
            norm1 = np.linalg.norm(coeff1_flat)
            norm2 = np.linalg.norm(coeff2_flat)
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
            else:
                similarity = None
        
        return float(similarity) if similarity is not None else None
    except Exception as e:
        print(f"  Warning: Error calculating orbital coefficient similarity: {e}")
        return None

def save_sample_to_txt(sample_result, txt_file):
    """Save a single sample result to a readable text file."""
    with open(txt_file, 'w') as f:
        f.write("=" * 120 + "\n")
        filename = sample_result.get('filename', 'N/A')
        atom_count = sample_result.get('atom_count', 0)
        f.write(f"DFT Calculation Results: {filename} ({atom_count} atoms)\n")
        f.write("=" * 120 + "\n\n")
        
        if 'error' in sample_result:
            f.write(f"Error: {sample_result['error']}\n")
            return
        
        # Extract energies
        scflow_e = sample_result.get('scflow', {}).get('energy', {})
        rks_e = sample_result.get('rks', {}).get('energy', {})
        scf_e = sample_result.get('scf', {}).get('energy', {})
        
        # Extract times
        scflow_time = sample_result.get('scflow', {}).get('time', None)
        rks_time = sample_result.get('rks', {}).get('time', None)
        rks_cycle = sample_result.get('rks', {}).get('cycle', None)
        scf_time = sample_result.get('scf', {}).get('time', None)
        scf_cycle = sample_result.get('scf', {}).get('cycle', None)
        
        # Extract differences
        diff_rks_scflow = sample_result.get('differences', {}).get('rks_vs_scflow', {})
        diff_scf_scflow = sample_result.get('differences', {}).get('scf_vs_scflow', {})
        diff_rks_scf = sample_result.get('differences', {}).get('rks_vs_scf', {})
        
        # Energies
        f.write("Energies:\n")
        if scflow_e.get('eV') is not None:
            f.write(f"  SCFlow:  {scflow_e.get('eV'):>15.6f} eV  ({scflow_e.get('meV'):>15.3f} meV)\n")
        if rks_e.get('eV') is not None:
            f.write(f"  RKS:     {rks_e.get('eV'):>15.6f} eV  ({rks_e.get('meV'):>15.3f} meV)\n")
        if scf_e.get('eV') is not None:
            f.write(f"  SCF:     {scf_e.get('eV'):>15.6f} eV  ({scf_e.get('meV'):>15.3f} meV)\n")
        f.write("\n")
        
        # Energies per atom
        f.write("Energies per atom:\n")
        if scflow_e.get('eV_per_atom') is not None:
            f.write(f"  SCFlow:  {scflow_e.get('eV_per_atom'):>15.6f} eV/atom  ({scflow_e.get('meV_per_atom'):>15.3f} meV/atom)\n")
        if rks_e.get('eV_per_atom') is not None:
            f.write(f"  RKS:     {rks_e.get('eV_per_atom'):>15.6f} eV/atom  ({rks_e.get('meV_per_atom'):>15.3f} meV/atom)\n")
        if scf_e.get('eV_per_atom') is not None:
            f.write(f"  SCF:     {scf_e.get('eV_per_atom'):>15.6f} eV/atom  ({scf_e.get('meV_per_atom'):>15.3f} meV/atom)\n")
        f.write("\n")
        
        # Energy differences
        f.write("Energy Differences:\n")
        rks_scflow_e = diff_rks_scflow.get('energy', {})
        if rks_scflow_e.get('eV') is not None:
            f.write(f"  RKS - SCFlow:   {rks_scflow_e.get('eV'):>15.6f} eV  ({rks_scflow_e.get('meV'):>15.3f} meV)\n")
            f.write(f"                   {rks_scflow_e.get('eV_per_atom'):>15.6f} eV/atom  ({rks_scflow_e.get('meV_per_atom'):>15.3f} meV/atom)\n")
        
        scf_scflow_e = diff_scf_scflow.get('energy', {})
        if scf_scflow_e.get('eV') is not None:
            f.write(f"  SCF - SCFlow:   {scf_scflow_e.get('eV'):>15.6f} eV  ({scf_scflow_e.get('meV'):>15.3f} meV)\n")
            f.write(f"                   {scf_scflow_e.get('eV_per_atom'):>15.6f} eV/atom  ({scf_scflow_e.get('meV_per_atom'):>15.3f} meV/atom)\n")
        
        rks_scf_e = diff_rks_scf.get('energy', {})
        if rks_scf_e.get('eV') is not None:
            f.write(f"  SCF - RKS:      {rks_scf_e.get('eV'):>15.6f} eV  ({rks_scf_e.get('meV'):>15.3f} meV)\n")
            f.write(f"                   {rks_scf_e.get('eV_per_atom'):>15.6f} eV/atom  ({rks_scf_e.get('meV_per_atom'):>15.3f} meV/atom)\n")
        f.write("\n")
        
        # Forces MAE
        f.write("Forces MAE:\n")
        rks_scflow_forces = diff_rks_scflow.get('forces_mae', {})
        if rks_scflow_forces.get('eV_per_ang') is not None:
            f.write(f"  RKS - SCFlow:   {rks_scflow_forces.get('eV_per_ang'):>15.6f} eV/Å  ({rks_scflow_forces.get('meV_per_ang'):>15.3f} meV/Å)\n")
        
        scf_scflow_forces = diff_scf_scflow.get('forces_mae', {})
        if scf_scflow_forces.get('eV_per_ang') is not None:
            f.write(f"  SCF - SCFlow:   {scf_scflow_forces.get('eV_per_ang'):>15.6f} eV/Å  ({scf_scflow_forces.get('meV_per_ang'):>15.3f} meV/Å)\n")
        
        rks_scf_forces = diff_rks_scf.get('forces_mae', {})
        if rks_scf_forces.get('eV_per_ang') is not None:
            f.write(f"  SCF - RKS:      {rks_scf_forces.get('eV_per_ang'):>15.6f} eV/Å  ({rks_scf_forces.get('meV_per_ang'):>15.3f} meV/Å)\n")
        f.write("\n")
        
        # Hamiltonian MAE
        f.write("Hamiltonian MAE:\n")
        rks_scflow_h_mae = diff_rks_scflow.get('hamiltonian_mae', {})
        if rks_scflow_h_mae.get('eV') is not None:
            f.write(f"  RKS - SCFlow:   {rks_scflow_h_mae.get('eV'):>15.6f} eV  ({rks_scflow_h_mae.get('meV'):>15.3f} meV)\n")
        
        scf_scflow_h_mae = diff_scf_scflow.get('hamiltonian_mae', {})
        if scf_scflow_h_mae.get('eV') is not None:
            f.write(f"  SCF - SCFlow:   {scf_scflow_h_mae.get('eV'):>15.6f} eV  ({scf_scflow_h_mae.get('meV'):>15.3f} meV)\n")
        
        rks_scf_h_mae = diff_rks_scf.get('hamiltonian_mae', {})
        if rks_scf_h_mae.get('eV') is not None:
            f.write(f"  SCF - RKS:      {rks_scf_h_mae.get('eV'):>15.6f} eV  ({rks_scf_h_mae.get('meV'):>15.3f} meV)\n")
        f.write("\n")
        
        # Orbital Energy MAE
        f.write("Orbital Energy MAE:\n")
        rks_scflow_oe_mae = diff_rks_scflow.get('orbital_energy_mae', {})
        if rks_scflow_oe_mae.get('eV') is not None:
            f.write(f"  RKS - SCFlow:   {rks_scflow_oe_mae.get('eV'):>15.6f} eV  ({rks_scflow_oe_mae.get('meV'):>15.3f} meV)\n")
        
        scf_scflow_oe_mae = diff_scf_scflow.get('orbital_energy_mae', {})
        if scf_scflow_oe_mae.get('eV') is not None:
            f.write(f"  SCF - SCFlow:   {scf_scflow_oe_mae.get('eV'):>15.6f} eV  ({scf_scflow_oe_mae.get('meV'):>15.3f} meV)\n")
        
        rks_scf_oe_mae = diff_rks_scf.get('orbital_energy_mae', {})
        if rks_scf_oe_mae.get('eV') is not None:
            f.write(f"  SCF - RKS:      {rks_scf_oe_mae.get('eV'):>15.6f} eV  ({rks_scf_oe_mae.get('meV'):>15.3f} meV)\n")
        f.write("\n")
        
        # Orbital Coefficient Similarity
        f.write("Orbital Coefficient Similarity:\n")
        rks_scflow_coeff_sim = diff_rks_scflow.get('orbital_coeff_similarity')
        if rks_scflow_coeff_sim is not None:
            f.write(f"  RKS - SCFlow:   {rks_scflow_coeff_sim:>15.6f}\n")
        
        scf_scflow_coeff_sim = diff_scf_scflow.get('orbital_coeff_similarity')
        if scf_scflow_coeff_sim is not None:
            f.write(f"  SCF - SCFlow:   {scf_scflow_coeff_sim:>15.6f}\n")
        
        rks_scf_coeff_sim = diff_rks_scf.get('orbital_coeff_similarity')
        if rks_scf_coeff_sim is not None:
            f.write(f"  SCF - RKS:      {rks_scf_coeff_sim:>15.6f}\n")
        f.write("\n")
        
        # Computation times
        f.write("Computation Times:\n")
        if scflow_time is not None:
            f.write(f"  SCFlow:  {scflow_time:>15.4f} s\n")
        if rks_time is not None:
            f.write(f"  RKS:     {rks_time:>15.4f} s")
            if rks_cycle is not None:
                f.write(f"  (cycles: {rks_cycle})")
            f.write("\n")
        if scf_time is not None:
            f.write(f"  SCF:     {scf_time:>15.4f} s")
            if scf_cycle is not None:
                f.write(f"  (cycles: {scf_cycle})")
            f.write("\n")
        f.write("\n")

# Conversion constants (from dft_process_utils.py)
ANG2BOHR = 1.8897261258369282      # Angstrom to Bohr conversion
BOHR2ANG = 1.0 / ANG2BOHR         # Bohr to Angstrom conversion
HA2eV = 27.211396641308           # Hartree to eV conversion
HA2meV = 27.211396641308 * 1000   # Hartree to meV conversion
EV_TO_MEV = 1000.0                # 1 eV = 1000 meV

# PySCF returns energy in Hartree and forces in Hartree/Bohr
# Conversion: Hartree/Bohr to eV/Ang = Hartree/Bohr * HA2eV / ANG2BOHR
HA_PER_BOHR_TO_EV_PER_ANG = HA2eV / ANG2BOHR
HA_PER_BOHR_TO_MEV_PER_ANG = HA2meV / ANG2BOHR

# Process each sample and save results
output_dir = f"/root/25DFT/QHFlow/src/md/test_script/dft_test_results_{model_name}"
os.makedirs(output_dir, exist_ok=True)

# sorted_files = sorted_files[1:]
for idx, (filename, atom_count, file_path) in enumerate(tqdm(sorted_files, desc="Processing samples", unit="sample")):
    print(f"\nProcessing sample {idx+1}/{len(sorted_files)}: {filename} ({atom_count} atoms)")
    
    # Create output filename (remove .xyz extension and add .json)
    output_filename = os.path.splitext(filename)[0] + ".json"
    output_file = os.path.join(output_dir, output_filename)
    
    try:
        atoms = loaded_xyz_structures[idx]
        
        # Get atom configuration
        atom_config = get_atom_configuration(atoms)
        
        # Calculate using SCFlow calculator (returns Hartree and Hartree/Bohr)
        start_time = time.time()
        sccalculator.calculate(atoms, properties=["energy", "forces"], system_changes=["positions"])
        scflow_time = time.time() - start_time
        
        scflow_energy_ha = sccalculator.results.get("energy")  # In Hartree
        scflow_forces_ha_bohr = sccalculator.results.get("forces")  # In Hartree/Bohr
        scflow_hamiltonian = None
        scflow_orbital_energy = None
        if hasattr(sccalculator, 'pred_ham') and sccalculator.pred_ham is not None:
            # Convert to numpy array if needed
            if isinstance(sccalculator.pred_ham, np.ndarray):
                scflow_hamiltonian = sccalculator.pred_ham.copy()
            else:
                scflow_hamiltonian = np.array(sccalculator.pred_ham)
        
        # Get orbital energies and coefficients from SCFlow (if available in mf object)
        scflow_orbital_coeff = None
        scflow_overlap = None
        if hasattr(sccalculator, 'mf') and hasattr(sccalculator.mf, 'mo_energy'):
            try:
                mo_energy = sccalculator.mf.mo_energy
                if hasattr(mo_energy, 'get'):
                    scflow_orbital_energy = np.array(mo_energy.get())  # gpu4pyscf
                else:
                    scflow_orbital_energy = np.array(mo_energy)
                
                # Get orbital coefficients
                if hasattr(sccalculator.mf, 'mo_coeff'):
                    mo_coeff = sccalculator.mf.mo_coeff
                    if hasattr(mo_coeff, 'get'):
                        scflow_orbital_coeff = np.array(mo_coeff.get())  # gpu4pyscf
                    else:
                        scflow_orbital_coeff = np.array(mo_coeff)
                
                # Get overlap matrix for similarity calculation
                # Try to get from calculator first, then from mf object
                if hasattr(sccalculator, 'ovlp') and sccalculator.ovlp is not None:
                    scflow_overlap = sccalculator.ovlp
                    if hasattr(scflow_overlap, 'get'):
                        scflow_overlap = np.array(scflow_overlap.get())
                    else:
                        scflow_overlap = np.array(scflow_overlap)
                elif hasattr(sccalculator.mf, 'mol') and hasattr(sccalculator.mf.mol, 'intor'):
                    try:
                        scflow_overlap = sccalculator.mf.mol.intor("int1e_ovlp")
                        if hasattr(scflow_overlap, 'get'):
                            scflow_overlap = np.array(scflow_overlap.get())
                        else:
                            scflow_overlap = np.array(scflow_overlap)
                    except:
                        pass
            except Exception as e:
                print(f"  Warning: Could not get SCFlow orbital energy/coefficients: {e}")
        
        # Convert SCFlow results from Hartree units to eV
        scflow_energy = scflow_energy_ha * HA2eV if scflow_energy_ha is not None else None
        scflow_forces = None
        if scflow_forces_ha_bohr is not None:
            scflow_forces = np.array(scflow_forces_ha_bohr) * HA_PER_BOHR_TO_EV_PER_ANG
        
        # Calculate using RKS calculator (returns Hartree and Hartree/Bohr)
        start_time = time.time()
        rkscalculator.calculate(atoms, properties=["energy", "forces"], system_changes=["positions"])
        rks_time = time.time() - start_time
        
        rks_energy_ha = rkscalculator.results.get("energy")  # In Hartree
        rks_forces_ha_bohr = rkscalculator.results.get("forces")  # In Hartree/Bohr
        rks_cycle = rkscalculator.results.get("cycles")  # Get cycle count if available
        rks_hamiltonian = None
        rks_orbital_energy = None
        if hasattr(rkscalculator, 'mf') and hasattr(rkscalculator.mf, 'get_fock'):
            try:
                fock = rkscalculator.mf.get_fock()
                # Handle gpu4pyscf cupy array
                if hasattr(fock, 'get'):
                    rks_hamiltonian = np.array(fock.get())
                else:
                    rks_hamiltonian = np.array(fock)
            except Exception as e:
                print(f"  Warning: Could not get RKS Hamiltonian: {e}")
        
        # Get orbital energies and coefficients from RKS
        rks_orbital_coeff = None
        rks_overlap = None
        if hasattr(rkscalculator, 'mf') and hasattr(rkscalculator.mf, 'mo_energy'):
            try:
                mo_energy = rkscalculator.mf.mo_energy
                if hasattr(mo_energy, 'get'):
                    rks_orbital_energy = np.array(mo_energy.get())  # gpu4pyscf
                else:
                    rks_orbital_energy = np.array(mo_energy)
                
                # Get orbital coefficients
                if hasattr(rkscalculator.mf, 'mo_coeff'):
                    mo_coeff = rkscalculator.mf.mo_coeff
                    if hasattr(mo_coeff, 'get'):
                        rks_orbital_coeff = np.array(mo_coeff.get())  # gpu4pyscf
                    else:
                        rks_orbital_coeff = np.array(mo_coeff)
                
                # Get overlap matrix
                if hasattr(rkscalculator.mf, 'mol') and hasattr(rkscalculator.mf.mol, 'intor'):
                    try:
                        rks_overlap = rkscalculator.mf.mol.intor("int1e_ovlp")
                        if hasattr(rks_overlap, 'get'):
                            rks_overlap = np.array(rks_overlap.get())
                        else:
                            rks_overlap = np.array(rks_overlap)
                    except Exception as e:
                        print(f"  Warning: Could not get RKS overlap matrix: {e}")
            except Exception as e:
                print(f"  Warning: Could not get RKS orbital energy/coefficients: {e}")
        
        # Convert RKS results from Hartree units to eV
        rks_energy = rks_energy_ha * HA2eV if rks_energy_ha is not None else None
        rks_forces = None
        if rks_forces_ha_bohr is not None:
            rks_forces = np.array(rks_forces_ha_bohr) * HA_PER_BOHR_TO_EV_PER_ANG
        
        # Calculate using SCF (SCFlow SCF)
        scf_energy = None
        scf_energy_ha = None
        scf_forces = None
        scf_forces_ha_bohr = None
        scf_hamiltonian = None
        scf_orbital_energy = None
        scf_time = None
        scf_cycle = None
        try:
            start_time = time.time()
            sccalculator.calculate_scf(atoms, properties=["energy", "forces"], system_changes=["positions"])
            scf_time = time.time() - start_time
            
            scf_energy_ha = sccalculator.results.get("energy")  # In Hartree
            scf_forces_ha_bohr = sccalculator.results.get("forces")  # In Hartree/Bohr
            scf_cycle = sccalculator.results.get("cycles")
            
            # Convert SCF results from Hartree units to eV
            scf_energy = scf_energy_ha * HA2eV if scf_energy_ha is not None else None
            if scf_forces_ha_bohr is not None:
                scf_forces = np.array(scf_forces_ha_bohr) * HA_PER_BOHR_TO_EV_PER_ANG
            
            # Get Hamiltonian if available
            if hasattr(sccalculator, 'pred_ham') and sccalculator.pred_ham is not None:
                if isinstance(sccalculator.pred_ham, np.ndarray):
                    scf_hamiltonian = sccalculator.pred_ham.copy()
                else:
                    scf_hamiltonian = np.array(sccalculator.pred_ham)
            
            # Get orbital energies and coefficients from SCF
            scf_orbital_coeff = None
            scf_overlap = None
            if hasattr(sccalculator, 'mf') and hasattr(sccalculator.mf, 'mo_energy'):
                try:
                    mo_energy = sccalculator.mf.mo_energy
                    if hasattr(mo_energy, 'get'):
                        scf_orbital_energy = np.array(mo_energy.get())  # gpu4pyscf
                    else:
                        scf_orbital_energy = np.array(mo_energy)
                    
                    # Get orbital coefficients
                    if hasattr(sccalculator.mf, 'mo_coeff'):
                        mo_coeff = sccalculator.mf.mo_coeff
                        if hasattr(mo_coeff, 'get'):
                            scf_orbital_coeff = np.array(mo_coeff.get())  # gpu4pyscf
                        else:
                            scf_orbital_coeff = np.array(mo_coeff)
                    
                    # Get overlap matrix
                    if hasattr(sccalculator, 'ovlp') and sccalculator.ovlp is not None:
                        scf_overlap = sccalculator.ovlp
                        if hasattr(scf_overlap, 'get'):
                            scf_overlap = np.array(scf_overlap.get())
                        else:
                            scf_overlap = np.array(scf_overlap)
                    elif hasattr(sccalculator.mf, 'mol') and hasattr(sccalculator.mf.mol, 'intor'):
                        try:
                            scf_overlap = sccalculator.mf.mol.intor("int1e_ovlp")
                            if hasattr(scf_overlap, 'get'):
                                scf_overlap = np.array(scf_overlap.get())
                            else:
                                scf_overlap = np.array(scf_overlap)
                        except:
                            pass
                except Exception as e:
                    print(f"  Warning: Could not get SCF orbital energy/coefficients: {e}")
        except Exception as e:
            print(f"  Warning: Could not calculate SCF: {e}")
        
        # Convert to numpy arrays for difference calculation
        scflow_forces_np = np.array(scflow_forces) if scflow_forces is not None else None
        rks_forces_np = np.array(rks_forces) if rks_forces is not None else None
        scflow_hamiltonian_np = scflow_hamiltonian if scflow_hamiltonian is not None else None
        rks_hamiltonian_np = rks_hamiltonian if rks_hamiltonian is not None else None
        scf_hamiltonian_np = scf_hamiltonian if scf_hamiltonian is not None else None
        
        # Calculate differences
        energy_diff = None
        energy_diff_mev = None
        energy_diff_per_atom = None
        energy_diff_per_atom_mev = None
        if scflow_energy is not None and rks_energy is not None:
            energy_diff = float(rks_energy - scflow_energy)  # RKS - SCFlow
            energy_diff_mev = energy_diff * EV_TO_MEV
            energy_diff_per_atom = energy_diff / atom_count
            energy_diff_per_atom_mev = energy_diff_per_atom * EV_TO_MEV
        
        # Calculate forces MAE (Mean Absolute Error)
        forces_mae = None
        forces_mae_mev = None
        if scflow_forces_np is not None and rks_forces_np is not None:
            forces_diff = rks_forces_np - scflow_forces_np  # RKS - SCFlow
            forces_mae = float(np.mean(np.abs(forces_diff)))  # MAE in eV/Å
            forces_mae_mev = forces_mae * EV_TO_MEV  # MAE in meV/Å
        
        hamiltonian_diff = None
        hamiltonian_mae = None
        hamiltonian_mae_ha = None
        if scflow_hamiltonian_np is not None and rks_hamiltonian_np is not None:
            if scflow_hamiltonian_np.shape == rks_hamiltonian_np.shape:
                hamiltonian_diff = rks_hamiltonian_np - scflow_hamiltonian_np  # RKS - SCFlow
                hamiltonian_mae_ha = float(np.mean(np.abs(hamiltonian_diff)))  # MAE in Hartree
                hamiltonian_mae = hamiltonian_mae_ha * HA2eV  # MAE in eV
            else:
                print(f"  Warning: Hamiltonian shape mismatch (SCFlow: {scflow_hamiltonian_np.shape}, RKS: {rks_hamiltonian_np.shape})")
        
        # Calculate orbital energy differences
        scflow_orbital_energy_np = scflow_orbital_energy if scflow_orbital_energy is not None else None
        rks_orbital_energy_np = rks_orbital_energy if rks_orbital_energy is not None else None
        scf_orbital_energy_np = scf_orbital_energy if scf_orbital_energy is not None else None
        
        orbital_energy_diff = None
        orbital_energy_mae = None
        orbital_energy_mae_ha = None
        if scflow_orbital_energy_np is not None and rks_orbital_energy_np is not None:
            if scflow_orbital_energy_np.shape == rks_orbital_energy_np.shape:
                orbital_energy_diff = rks_orbital_energy_np - scflow_orbital_energy_np  # RKS - SCFlow
                orbital_energy_mae_ha = float(np.mean(np.abs(orbital_energy_diff)))  # MAE in Hartree
                orbital_energy_mae = orbital_energy_mae_ha * HA2eV  # MAE in eV
            else:
                print(f"  Warning: Orbital energy shape mismatch (SCFlow: {scflow_orbital_energy_np.shape}, RKS: {rks_orbital_energy_np.shape})")
        
        # Calculate energy per atom
        scflow_energy_per_atom = None
        scflow_energy_per_atom_mev = None
        if scflow_energy is not None:
            scflow_energy_per_atom = float(scflow_energy / atom_count)
            scflow_energy_per_atom_mev = scflow_energy_per_atom * EV_TO_MEV
        
        rks_energy_per_atom = None
        rks_energy_per_atom_mev = None
        if rks_energy is not None:
            rks_energy_per_atom = float(rks_energy / atom_count)
            rks_energy_per_atom_mev = rks_energy_per_atom * EV_TO_MEV
        
        # Calculate SCF energy per atom
        scf_energy_per_atom = None
        scf_energy_per_atom_mev = None
        if scf_energy is not None:
            scf_energy_per_atom = float(scf_energy / atom_count)
            scf_energy_per_atom_mev = scf_energy_per_atom * EV_TO_MEV
        
        # Convert to numpy arrays for SCF difference calculation
        scf_forces_np = np.array(scf_forces) if scf_forces is not None else None
        scf_hamiltonian_np = scf_hamiltonian if scf_hamiltonian is not None else None
        
        # Calculate differences with SCF
        scf_energy_diff = None
        scf_energy_diff_mev = None
        scf_energy_diff_per_atom = None
        scf_energy_diff_per_atom_mev = None
        if scflow_energy is not None and scf_energy is not None:
            scf_energy_diff = float(scf_energy - scflow_energy)  # SCF - SCFlow
            scf_energy_diff_mev = scf_energy_diff * EV_TO_MEV
            scf_energy_diff_per_atom = scf_energy_diff / atom_count
            scf_energy_diff_per_atom_mev = scf_energy_diff_per_atom * EV_TO_MEV
        
        scf_forces_mae = None
        scf_forces_mae_mev = None
        if scflow_forces_np is not None and scf_forces_np is not None:
            scf_forces_diff = scf_forces_np - scflow_forces_np  # SCF - SCFlow
            scf_forces_mae = float(np.mean(np.abs(scf_forces_diff)))  # MAE in eV/Å
            scf_forces_mae_mev = scf_forces_mae * EV_TO_MEV  # MAE in meV/Å
        
        scf_hamiltonian_diff = None
        scf_hamiltonian_mae = None
        scf_hamiltonian_mae_ha = None
        if scflow_hamiltonian_np is not None and scf_hamiltonian_np is not None:
            if scflow_hamiltonian_np.shape == scf_hamiltonian_np.shape:
                scf_hamiltonian_diff = scf_hamiltonian_np - scflow_hamiltonian_np  # SCF - SCFlow
                scf_hamiltonian_mae_ha = float(np.mean(np.abs(scf_hamiltonian_diff)))  # MAE in Hartree
                scf_hamiltonian_mae = scf_hamiltonian_mae_ha * HA2eV  # MAE in eV
            else:
                print(f"  Warning: SCF Hamiltonian shape mismatch (SCFlow: {scflow_hamiltonian_np.shape}, SCF: {scf_hamiltonian_np.shape})")
        
        # Calculate orbital energy differences with SCF
        scf_orbital_energy_diff = None
        scf_orbital_energy_mae = None
        scf_orbital_energy_mae_ha = None
        if scflow_orbital_energy_np is not None and scf_orbital_energy_np is not None:
            if scflow_orbital_energy_np.shape == scf_orbital_energy_np.shape:
                scf_orbital_energy_diff = scf_orbital_energy_np - scflow_orbital_energy_np  # SCF - SCFlow
                scf_orbital_energy_mae_ha = float(np.mean(np.abs(scf_orbital_energy_diff)))  # MAE in Hartree
                scf_orbital_energy_mae = scf_orbital_energy_mae_ha * HA2eV  # MAE in eV
            else:
                print(f"  Warning: SCF Orbital energy shape mismatch (SCFlow: {scflow_orbital_energy_np.shape}, SCF: {scf_orbital_energy_np.shape})")
        
        # Calculate differences between RKS and SCF
        rks_scf_energy_diff = None
        rks_scf_energy_diff_mev = None
        rks_scf_energy_diff_per_atom = None
        rks_scf_energy_diff_per_atom_mev = None
        if rks_energy is not None and scf_energy is not None:
            rks_scf_energy_diff = float(scf_energy - rks_energy)  # SCF - RKS
            rks_scf_energy_diff_mev = rks_scf_energy_diff * EV_TO_MEV
            rks_scf_energy_diff_per_atom = rks_scf_energy_diff / atom_count
            rks_scf_energy_diff_per_atom_mev = rks_scf_energy_diff_per_atom * EV_TO_MEV
        
        rks_scf_forces_mae = None
        rks_scf_forces_mae_mev = None
        if rks_forces_np is not None and scf_forces_np is not None:
            rks_scf_forces_diff = scf_forces_np - rks_forces_np  # SCF - RKS
            rks_scf_forces_mae = float(np.mean(np.abs(rks_scf_forces_diff)))  # MAE in eV/Å
            rks_scf_forces_mae_mev = rks_scf_forces_mae * EV_TO_MEV  # MAE in meV/Å
        
        rks_scf_hamiltonian_diff = None
        rks_scf_hamiltonian_mae = None
        rks_scf_hamiltonian_mae_ha = None
        if rks_hamiltonian_np is not None and scf_hamiltonian_np is not None:
            if rks_hamiltonian_np.shape == scf_hamiltonian_np.shape:
                rks_scf_hamiltonian_diff = scf_hamiltonian_np - rks_hamiltonian_np  # SCF - RKS
                rks_scf_hamiltonian_mae_ha = float(np.mean(np.abs(rks_scf_hamiltonian_diff)))  # MAE in Hartree
                rks_scf_hamiltonian_mae = rks_scf_hamiltonian_mae_ha * HA2eV  # MAE in eV
            else:
                print(f"  Warning: RKS-SCF Hamiltonian shape mismatch (RKS: {rks_hamiltonian_np.shape}, SCF: {scf_hamiltonian_np.shape})")
        
        # Calculate orbital energy differences between RKS and SCF
        rks_scf_orbital_energy_diff = None
        rks_scf_orbital_energy_mae = None
        rks_scf_orbital_energy_mae_ha = None
        if rks_orbital_energy_np is not None and scf_orbital_energy_np is not None:
            if rks_orbital_energy_np.shape == scf_orbital_energy_np.shape:
                rks_scf_orbital_energy_diff = scf_orbital_energy_np - rks_orbital_energy_np  # SCF - RKS
                rks_scf_orbital_energy_mae_ha = float(np.mean(np.abs(rks_scf_orbital_energy_diff)))  # MAE in Hartree
                rks_scf_orbital_energy_mae = rks_scf_orbital_energy_mae_ha * HA2eV  # MAE in eV
            else:
                print(f"  Warning: RKS-SCF Orbital energy shape mismatch (RKS: {rks_orbital_energy_np.shape}, SCF: {scf_orbital_energy_np.shape})")
        
        # Calculate orbital coefficient similarities
        # Use overlap matrix from RKS as reference (if available), otherwise use cosine similarity
        ref_overlap = rks_overlap if rks_overlap is not None else (scflow_overlap if scflow_overlap is not None else None)
        
        rks_scflow_coeff_similarity = calculate_orbital_coefficient_similarity(
            rks_orbital_coeff, scflow_orbital_coeff, ref_overlap
        )
        scf_scflow_coeff_similarity = calculate_orbital_coefficient_similarity(
            scf_orbital_coeff, scflow_orbital_coeff, ref_overlap
        )
        rks_scf_coeff_similarity = calculate_orbital_coefficient_similarity(
            rks_orbital_coeff, scf_orbital_coeff, ref_overlap
        )
        
        # Store results
        sample_result = {
            "filename": filename,
            "atom_count": atom_count,
            "atom_configuration": atom_config,
            "scflow": {
                "energy": {
                    "hartree": float(scflow_energy_ha) if scflow_energy_ha is not None else None,
                    "eV": float(scflow_energy) if scflow_energy is not None else None,
                    "meV": float(scflow_energy * EV_TO_MEV) if scflow_energy is not None else None,
                    "hartree_per_atom": float(scflow_energy_ha / atom_count) if scflow_energy_ha is not None else None,
                    "eV_per_atom": scflow_energy_per_atom,
                    "meV_per_atom": scflow_energy_per_atom_mev
                },
                "forces": {
                    "hartree_per_bohr": to_list(scflow_forces_ha_bohr),
                    "eV_per_ang": to_list(scflow_forces),
                    "meV_per_ang": to_list(scflow_forces_np * EV_TO_MEV) if scflow_forces_np is not None else None
                },
                "hamiltonian": to_list(scflow_hamiltonian),
                "orbital_energy": to_list(scflow_orbital_energy),
                "orbital_coeff": to_list(scflow_orbital_coeff),
                "time": scflow_time
            },
            "rks": {
                "energy": {
                    "hartree": float(rks_energy_ha) if rks_energy_ha is not None else None,
                    "eV": float(rks_energy) if rks_energy is not None else None,
                    "meV": float(rks_energy * EV_TO_MEV) if rks_energy is not None else None,
                    "hartree_per_atom": float(rks_energy_ha / atom_count) if rks_energy_ha is not None else None,
                    "eV_per_atom": rks_energy_per_atom,
                    "meV_per_atom": rks_energy_per_atom_mev
                },
                "forces": {
                    "hartree_per_bohr": to_list(rks_forces_ha_bohr),
                    "eV_per_ang": to_list(rks_forces),
                    "meV_per_ang": to_list(rks_forces_np * EV_TO_MEV) if rks_forces_np is not None else None
                },
                "hamiltonian": to_list(rks_hamiltonian),
                "orbital_energy": to_list(rks_orbital_energy),
                "orbital_coeff": to_list(rks_orbital_coeff),
                "time": rks_time,
                "cycle": rks_cycle
            },
            "scf": {
                "energy": {
                    "hartree": float(scf_energy_ha) if scf_energy_ha is not None else None,
                    "eV": float(scf_energy) if scf_energy is not None else None,
                    "meV": float(scf_energy * EV_TO_MEV) if scf_energy is not None else None,
                    "hartree_per_atom": float(scf_energy_ha / atom_count) if scf_energy_ha is not None else None,
                    "eV_per_atom": scf_energy_per_atom,
                    "meV_per_atom": scf_energy_per_atom_mev
                },
                "forces": {
                    "hartree_per_bohr": to_list(scf_forces_ha_bohr),
                    "eV_per_ang": to_list(scf_forces),
                    "meV_per_ang": to_list(scf_forces_np * EV_TO_MEV) if scf_forces_np is not None else None
                },
                "hamiltonian": to_list(scf_hamiltonian),
                "orbital_energy": to_list(scf_orbital_energy),
                "orbital_coeff": to_list(scf_orbital_coeff),
                "time": scf_time,
                "cycle": scf_cycle
            },
            "differences": {
                "rks_vs_scflow": {
                    "energy": {
                        "hartree": float((rks_energy_ha - scflow_energy_ha)) if (rks_energy_ha is not None and scflow_energy_ha is not None) else None,
                        "eV": energy_diff,
                        "meV": energy_diff_mev,
                        "hartree_per_atom": float((rks_energy_ha - scflow_energy_ha) / atom_count) if (rks_energy_ha is not None and scflow_energy_ha is not None) else None,
                        "eV_per_atom": energy_diff_per_atom,
                        "meV_per_atom": energy_diff_per_atom_mev
                    },
                    "forces_mae": {
                        "hartree_per_bohr": float(np.mean(np.abs(np.array(rks_forces_ha_bohr) - np.array(scflow_forces_ha_bohr)))) if (rks_forces_ha_bohr is not None and scflow_forces_ha_bohr is not None) else None,
                        "eV_per_ang": forces_mae,
                        "meV_per_ang": forces_mae_mev
                    },
                    "hamiltonian": to_list(hamiltonian_diff),
                    "hamiltonian_mae": {
                        "hartree": hamiltonian_mae_ha,
                        "eV": hamiltonian_mae,
                        "meV": hamiltonian_mae * EV_TO_MEV if hamiltonian_mae is not None else None
                    },
                    "orbital_energy": to_list(orbital_energy_diff),
                    "orbital_energy_mae": {
                        "hartree": orbital_energy_mae_ha,
                        "eV": orbital_energy_mae,
                        "meV": orbital_energy_mae * EV_TO_MEV if orbital_energy_mae is not None else None
                    },
                    "orbital_coeff_similarity": rks_scflow_coeff_similarity
                },
                "scf_vs_scflow": {
                    "energy": {
                        "hartree": float((scf_energy_ha - scflow_energy_ha)) if (scf_energy_ha is not None and scflow_energy_ha is not None) else None,
                        "eV": scf_energy_diff,
                        "meV": scf_energy_diff_mev,
                        "hartree_per_atom": float((scf_energy_ha - scflow_energy_ha) / atom_count) if (scf_energy_ha is not None and scflow_energy_ha is not None) else None,
                        "eV_per_atom": scf_energy_diff_per_atom,
                        "meV_per_atom": scf_energy_diff_per_atom_mev
                    },
                    "forces_mae": {
                        "hartree_per_bohr": float(np.mean(np.abs(np.array(scf_forces_ha_bohr) - np.array(scflow_forces_ha_bohr)))) if (scf_forces_ha_bohr is not None and scflow_forces_ha_bohr is not None) else None,
                        "eV_per_ang": scf_forces_mae,
                        "meV_per_ang": scf_forces_mae_mev
                    },
                    "hamiltonian": to_list(scf_hamiltonian_diff),
                    "hamiltonian_mae": {
                        "hartree": scf_hamiltonian_mae_ha,
                        "eV": scf_hamiltonian_mae,
                        "meV": scf_hamiltonian_mae * EV_TO_MEV if scf_hamiltonian_mae is not None else None
                    },
                    "orbital_energy": to_list(scf_orbital_energy_diff),
                    "orbital_energy_mae": {
                        "hartree": scf_orbital_energy_mae_ha,
                        "eV": scf_orbital_energy_mae,
                        "meV": scf_orbital_energy_mae * EV_TO_MEV if scf_orbital_energy_mae is not None else None
                    },
                    "orbital_coeff_similarity": scf_scflow_coeff_similarity
                },
                "rks_vs_scf": {
                    "energy": {
                        "hartree": float((scf_energy_ha - rks_energy_ha)) if (scf_energy_ha is not None and rks_energy_ha is not None) else None,
                        "eV": rks_scf_energy_diff,
                        "meV": rks_scf_energy_diff_mev,
                        "hartree_per_atom": float((scf_energy_ha - rks_energy_ha) / atom_count) if (scf_energy_ha is not None and rks_energy_ha is not None) else None,
                        "eV_per_atom": rks_scf_energy_diff_per_atom,
                        "meV_per_atom": rks_scf_energy_diff_per_atom_mev
                    },
                    "forces_mae": {
                        "hartree_per_bohr": float(np.mean(np.abs(np.array(scf_forces_ha_bohr) - np.array(rks_forces_ha_bohr)))) if (scf_forces_ha_bohr is not None and rks_forces_ha_bohr is not None) else None,
                        "eV_per_ang": rks_scf_forces_mae,
                        "meV_per_ang": rks_scf_forces_mae_mev
                    },
                    "hamiltonian": to_list(rks_scf_hamiltonian_diff),
                    "hamiltonian_mae": {
                        "hartree": rks_scf_hamiltonian_mae_ha,
                        "eV": rks_scf_hamiltonian_mae,
                        "meV": rks_scf_hamiltonian_mae * EV_TO_MEV if rks_scf_hamiltonian_mae is not None else None
                    },
                    "orbital_energy": to_list(rks_scf_orbital_energy_diff),
                    "orbital_energy_mae": {
                        "hartree": rks_scf_orbital_energy_mae_ha,
                        "eV": rks_scf_orbital_energy_mae,
                        "meV": rks_scf_orbital_energy_mae * EV_TO_MEV if rks_scf_orbital_energy_mae is not None else None
                    },
                    "orbital_coeff_similarity": rks_scf_coeff_similarity
                }
            }
        }
        
        print(f"  SCFlow - Energy: {scflow_energy:.6f} eV ({scflow_energy_per_atom:.6f} eV/atom), Time: {scflow_time:.4f}s")
        print(f"  RKS    - Energy: {rks_energy:.6f} eV ({rks_energy_per_atom:.6f} eV/atom), Time: {rks_time:.4f}s")
        if scf_energy is not None:
            print(f"  SCF    - Energy: {scf_energy:.6f} eV ({scf_energy_per_atom:.6f} eV/atom), Time: {scf_time:.4f}s")
        if energy_diff is not None:
            print(f"  RKS - SCFlow energy difference: {energy_diff:.6f} eV ({energy_diff_mev:.3f} meV)")
            print(f"  RKS - SCFlow energy difference per atom: {energy_diff_per_atom:.6f} eV ({energy_diff_per_atom_mev:.3f} meV)")
        if scf_energy_diff is not None:
            print(f"  SCF - SCFlow energy difference: {scf_energy_diff:.6f} eV ({scf_energy_diff_mev:.3f} meV)")
            print(f"  SCF - SCFlow energy difference per atom: {scf_energy_diff_per_atom:.6f} eV ({scf_energy_diff_per_atom_mev:.3f} meV)")
        if forces_mae is not None:
            print(f"  RKS - SCFlow Forces MAE: {forces_mae:.6f} eV/Å ({forces_mae_mev:.3f} meV/Å)")
        if scf_forces_mae is not None:
            print(f"  SCF - SCFlow Forces MAE: {scf_forces_mae:.6f} eV/Å ({scf_forces_mae_mev:.3f} meV/Å)")
        if rks_scf_energy_diff is not None:
            print(f"  SCF - RKS energy difference: {rks_scf_energy_diff:.6f} eV ({rks_scf_energy_diff_mev:.3f} meV)")
            print(f"  SCF - RKS energy difference per atom: {rks_scf_energy_diff_per_atom:.6f} eV ({rks_scf_energy_diff_per_atom_mev:.3f} meV)")
        if rks_scf_forces_mae is not None:
            print(f"  SCF - RKS Forces MAE: {rks_scf_forces_mae:.6f} eV/Å ({rks_scf_forces_mae_mev:.3f} meV/Å)")
        
    except Exception as e:
        print(f"  Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to get atom configuration even if there's an error
        atom_config = None
        try:
            atoms = loaded_xyz_structures[idx]
            atom_config = get_atom_configuration(atoms)
        except:
            pass
        
        # Store error information
        sample_result = {
            "filename": filename,
            "atom_count": atom_count,
            "atom_configuration": atom_config,
            "error": str(e)
        }
    
    # Save result for this sample to individual JSON file
    with open(output_file, 'w') as f:
        json.dump(sample_result, f, indent=2)
    print(f"  Results saved to {output_file}")
    
    # Also save as readable text file
    txt_file = output_file.replace('.json', '.txt')
    save_sample_to_txt(sample_result, txt_file)
    print(f"  Text summary saved to {txt_file}")

print(f"\n\nAll results saved to {output_dir}")
print(f"Total samples processed: {len(sorted_files)}")
