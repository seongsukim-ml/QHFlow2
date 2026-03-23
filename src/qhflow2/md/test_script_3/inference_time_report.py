import sys
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
sys.path.append("/root/25DFT/QHFlow/src")
# import md.scflow_calculator_gpu
from qhflow2.md.scflow_calculator_gpu import SCFlowRKSCalculator, RKSCalculator
from qhflow2.dft_process.dft_process_utils import *
from qhflow2.common.draw_util import *

import time
import json
import numpy as np
from ase import Atoms
from tqdm import tqdm
import argparse
from collections import defaultdict
import pandas as pd
import torch
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='DFT calculation with different model sizes and splits')
parser.add_argument('--model_size', type=str, choices=['small', 'middle', 'large', 'extra_large_260k'], 
                    default='middle', help='Model size to use (default: middle)')
parser.add_argument('--split', type=str, choices=['random', 'ood', 'geo', 'mol'], 
                    default='random', help='Model split to use: random, ood, geo, or mol (default: random)')
parser.add_argument('--debug', action='store_true', 
                    help='Debug mode: process only 10 molecules for quick testing')
parser.add_argument('--num_samples', type=int, default=None,
                    help='Number of samples to process (default: all samples, or 10 if --debug is used)')
parser.add_argument('--base_path', type=str, default=None,
                    help='Base path for model checkpoints (default: /root/25DFT/QHFlow/src/outputs)')
parser.add_argument('--init_gpu4pyscf', action='store_true',
                    help='Initialize gpu4pyscf (default: False)')
parser.add_argument('--init_density_fit', action='store_true',
                    help='Initialize density fit (default: False)')
parser.add_argument('--use_ref_init_ham', action='store_true',
                    help='Use reference outputs (default: False)')
parser.add_argument('--vasp', action='store_true',
                    help='Use reference outputs (default: False)')
parser.add_argument('--data_index', type=int, default=0,
                    help='Data index (default: 0)')
parser.add_argument('--use_shard', action='store_true',
                    help='Use shard (default: False)')
args = parser.parse_args()

qh9_experiment = QH9Experiment(data_index=args.data_index, use_shard=args.use_shard)
qh9_test_set = qh9_experiment.test_dataset
qh9_test_mask = qh9_experiment.dataset.test_mask

# Setup experiment and calculator
split_name = args.split
base_path = args.base_path
if base_path is None:
    if args.vasp:
        base_path = "/root/25DFT/QHFlow/ckpts/model_ckpt"
    else:
        base_path = "/root/25DFT/QHFlow/src/outputs"

split_dir_dict = {
    'random': 'QH9Stable-random',
    'ood': 'QH9Stable-size_ood',
    'geo': 'QH9Dynamic-300k-geo',
    'mol': 'QH9Dynamic-300k-mol',
}

weight_name = {
    'random': "weights-epoch=78-val_loss=0.0000000.ckpt",
    'ood': "weights-epoch=78-val_loss=0.0000000.ckpt",
    'geo': "weights-epoch=33-val_loss=0.0000000.ckpt",
    'mol': "weights-epoch=33-val_loss=0.0000000.ckpt",
}

# Construct checkpoint path based on split and model size
split_dir = split_dir_dict[split_name]
model_dir = f"QHFlow_so2_v5_1_{args.model_size}-{split_dir}"
if args.vasp:
    checkpoint_path = f"{base_path}/{split_dir}/{model_dir}/{weight_name[split_name]}"
else:
    checkpoint_path = f"{base_path}/{split_dir}/{model_dir}/checkpoints/{weight_name[split_name]}"
cur_ckpt = checkpoint_path

model_name = f"{args.model_size}_{split_name}"
if args.debug:
    model_name = f"{model_name}_debug"
print(f"Using model size: {args.model_size}")
print(f"Using split: {split_name}")
if args.debug:
    print(f"Debug mode: ON (processing 10 molecules)")
if args.num_samples is not None:
    print(f"Number of samples: {args.num_samples}")
print(f"Model checkpoint: {cur_ckpt}")

print(f"Loading model from {cur_ckpt}")
# Initialize calculators
sccalculator = SCFlowRKSCalculator(basis="def2-SVP", functional="b3lyp")

sccalculator.set_model(
    cur_ckpt, 
    data_type="qh9", 
    units="ang", 
    model_length_unit="ang", 
    mf_init_functional="b3lyp",
    init_gpu4pyscf=args.init_gpu4pyscf,
    init_density_fit=args.init_density_fit,
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
rkscalculator = RKSCalculator(basis="def2-SVP", functional="b3lyp", density_fit=True)

# Limit test_set based on arguments
if args.num_samples is not None:
    qh9_test_set = qh9_test_set[:args.num_samples]
    print(f"\nLimited test_set to {len(qh9_test_set)} samples before processing")
elif args.debug:
    qh9_test_set = qh9_test_set[:10]
    print(f"\nDebug mode: Limited test_set to {len(qh9_test_set)} samples before processing")

# Convert test_set to list of ASE Atoms objects
print(f"\nConverting test_set to ASE Atoms objects...")
print(f"Printing first atom structure for verification:")
print(qh9_test_set[0])
test_atoms_list = []
for idx, data in enumerate(tqdm(qh9_test_set, desc="Converting test_set")):
    try:
        # Extract atomic numbers and positions from data
        atomic_numbers = data.atoms.numpy().squeeze()
        positions = data.pos.numpy().squeeze()
        
        # Create ASE Atoms object
        atoms = Atoms(numbers=atomic_numbers, positions=positions)
        test_atoms_list.append((idx, atoms, len(atoms)))
    except Exception as e:
        print(f"Error converting sample {idx}: {e}")
        continue

print(f"Successfully converted {len(test_atoms_list)} samples from test_set")

# Sort by atom count
test_atoms_list.sort(key=lambda x: x[2])

# Group by atom count
atom_count_groups = defaultdict(list)
for idx, atoms, atom_count in test_atoms_list:
    atom_count_groups[atom_count].append((idx, atoms))

print(f"\nFound {len(atom_count_groups)} unique atom counts:")
for atom_count in sorted(atom_count_groups.keys()):
    print(f"  {atom_count} atoms: {len(atom_count_groups[atom_count])} samples")

# Statistics collection
def calculate_statistics(values):
    """Calculate mean, std, min, max, median from a list of values"""
    if not values:
        return None
    values = np.array(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "count": len(values)
    }

# GPU usage monitoring functions
def get_gpu_memory_info():
    """Get GPU memory information using torch.cuda"""
    if not torch.cuda.is_available():
        return None
    try:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "device": device
        }
    except Exception as e:
        print(f"Warning: Could not get GPU memory info: {e}")
        return None


# Process each atom count group
results_by_atom_count = {}
all_samples = []

# Warmup: run calculations on one sample from each atom count group
print(f"\nWarming up GPU with one sample from each atom count group...")
sorted_atom_counts = sorted(atom_count_groups.keys())
for atom_count in sorted_atom_counts:
    samples = atom_count_groups[atom_count]
    if len(samples) > 0:
        sample_idx, atoms = samples[0]  # Use first sample from each atom count group
        print(f"  Warming up with 1 sample from {atom_count} atoms group...")
        try:
            # Get ref_initial_hamiltonian for warmup
            ref_outputs = qh9_experiment.dataset.get_gt_outputs(qh9_test_mask[sample_idx])
            ref_initial_hamiltonian = ref_outputs["initial_hamiltonian"].numpy()
            
            # Warmup SCFlow
            sccalculator.calculate(atoms, properties=["energy", "forces"], system_changes=["positions"], init_ham=ref_initial_hamiltonian)
            # Warmup SCF
            sccalculator.calculate_scf(atoms, properties=["energy", "forces"], system_changes=["positions"], init_ham=ref_initial_hamiltonian)
            # Warmup RKS
            rkscalculator.calculate(atoms, properties=["energy", "forces"], system_changes=["positions"])
            print(f"    Warmup completed for {atom_count} atoms")
        except Exception as e:
            print(f"    Warning: Warmup failed for {atom_count} atoms (sample {sample_idx}): {e}")
print(f"Warmup completed for all atom count groups\n")

print(f"\nProcessing {len(test_atoms_list)} samples from test_set...")
for atom_count in tqdm(sorted_atom_counts, desc="Processing atom groups"):
    samples = atom_count_groups[atom_count]
    
    # Storage for this atom count
    scflow_times = []
    scflow_time_dicts = []
    rks_times = []
    rks_time_dicts = []
    scf_times = []
    scf_time_dicts = []
    
    # Individual time components
    scflow_mf_times = []
    scflow_ovlp_times = []
    scflow_init_ham_times = []
    scflow_pred_ham_times = []
    scflow_pred_dm0_times = []
    scflow_energy_times = []
    scflow_grad_frame_times = []
    
    rks_energy_times = []
    rks_grad_frame_times = []
    rks_init_ham_times = []
    rks_cycles = []
    
    scf_mf_times = []
    scf_ovlp_times = []
    scf_init_ham_times = []
    scf_pred_ham_times = []
    scf_pred_dm0_times = []
    scf_energy_times = []
    scf_grad_frame_times = []
    scf_cycles = []
    
    # Model time = pred_ham_time + pred_dm0_time
    scflow_model_times = []
    scf_model_times = []
    
    # GPU usage tracking
    scflow_gpu_memory = []
    rks_gpu_memory = []
    scf_gpu_memory = []
    
    # Energy and forces storage
    scflow_energies = []
    scflow_forces_list = []
    rks_energies = []
    rks_forces_list = []
    scf_energies = []
    scf_forces_list = []
    
    # Energy and force differences
    rks_vs_scflow_energy_diffs = []
    rks_vs_scflow_forces_mae = []
    rks_vs_scf_energy_diffs = []
    rks_vs_scf_forces_mae = []
    scflow_vs_scf_energy_diffs = []
    scflow_vs_scf_forces_mae = []
    
    for sample_idx, atoms in tqdm(samples, desc=f"  Processing {atom_count} atoms", leave=False):
        try:
            # SCFlow calculation
            # Get GPU info before calculation
            gpu_mem_before = get_gpu_memory_info()
            if args.use_ref_init_ham:
                ref_outputs = qh9_experiment.dataset.get_gt_outputs(qh9_test_mask[sample_idx])
                ref_initial_hamiltonian = ref_outputs["initial_hamiltonian"].numpy()
            else:
                ref_initial_hamiltonian = None

            start_time = time.time()
            sccalculator.calculate(atoms, properties=["energy", "forces"], system_changes=["positions"], init_ham=ref_initial_hamiltonian)
            scflow_time = time.time() - start_time
            scflow_times.append(scflow_time)
            
            # Get energy and forces from SCFlow
            scflow_energy = sccalculator.results.get("energy") if hasattr(sccalculator, 'results') else None
            scflow_forces = sccalculator.results.get("forces") if hasattr(sccalculator, 'results') else None
            if scflow_energy is not None:
                scflow_energies.append(float(scflow_energy))
            if scflow_forces is not None:
                scflow_forces_list.append(np.array(scflow_forces))
            
            # Get GPU info after calculation
            gpu_mem_after = get_gpu_memory_info()
            
            scflow_gpu_memory.append({
                "before": gpu_mem_before,
                "after": gpu_mem_after
            })
            
            # Get time_dict from SCFlow
            scflow_time_dict = None
            if hasattr(sccalculator, 'time_dict') and sccalculator.time_dict:
                scflow_time_dict = sccalculator.time_dict.copy()
                scflow_time_dicts.append(scflow_time_dict)
                
                # Extract individual components
                scflow_mf_times.append(scflow_time_dict.get("mf_time", 0))
                scflow_ovlp_times.append(scflow_time_dict.get("ovlp_time", 0))
                scflow_init_ham_times.append(scflow_time_dict.get("init_ham_time", 0))
                scflow_pred_ham_times.append(scflow_time_dict.get("pred_ham_time", 0))
                scflow_pred_dm0_times.append(scflow_time_dict.get("pred_dm0_time", 0))
                scflow_energy_times.append(scflow_time_dict.get("energy_time", 0))
                scflow_grad_frame_times.append(scflow_time_dict.get("grad_frame_time", 0))
                
                # Model time = pred_ham_time + pred_dm0_time
                model_time = scflow_time_dict.get("pred_ham_time", 0) + scflow_time_dict.get("pred_dm0_time", 0)
                scflow_model_times.append(model_time)
            
            # SCF calculation (SCFlow SCF)
            try:
                gpu_mem_before = get_gpu_memory_info()
                
                start_time = time.time()
                sccalculator.calculate_scf(atoms, properties=["energy", "forces"], system_changes=["positions"], init_ham=ref_initial_hamiltonian)
                scf_time = time.time() - start_time
                scf_times.append(scf_time)
                
                # Get energy and forces from SCF
                scf_energy = sccalculator.results.get("energy") if hasattr(sccalculator, 'results') else None
                scf_forces = sccalculator.results.get("forces") if hasattr(sccalculator, 'results') else None
                if scf_energy is not None:
                    scf_energies.append(float(scf_energy))
                if scf_forces is not None:
                    scf_forces_list.append(np.array(scf_forces))
                
                gpu_mem_after = get_gpu_memory_info()
                
                scf_gpu_memory.append({
                    "before": gpu_mem_before,
                    "after": gpu_mem_after
                })
                
                # Get time_dict from SCF
                scf_time_dict = None
                if hasattr(sccalculator, 'time_dict') and sccalculator.time_dict:
                    scf_time_dict = sccalculator.time_dict.copy()
                    scf_time_dicts.append(scf_time_dict)
                    
                    # Extract individual components
                    scf_mf_times.append(scf_time_dict.get("mf_time", 0))
                    scf_ovlp_times.append(scf_time_dict.get("ovlp_time", 0))
                    scf_init_ham_times.append(scf_time_dict.get("init_ham_time", 0))
                    scf_pred_ham_times.append(scf_time_dict.get("pred_ham_time", 0))
                    scf_pred_dm0_times.append(scf_time_dict.get("pred_dm0_time", 0))
                    scf_energy_times.append(scf_time_dict.get("energy_time", 0))
                    scf_grad_frame_times.append(scf_time_dict.get("grad_frame_time", 0))
                    
                    # Model time = pred_ham_time + pred_dm0_time
                    model_time = scf_time_dict.get("pred_ham_time", 0) + scf_time_dict.get("pred_dm0_time", 0)
                    scf_model_times.append(model_time)
                
                # Get cycle count from SCF
                scf_cycle = sccalculator.results.get("cycles") if hasattr(sccalculator, 'results') else None
                if scf_cycle is not None:
                    scf_cycles.append(scf_cycle)
            except Exception as e:
                print(f"    Warning: SCF calculation failed for sample {sample_idx}: {e}")
                # Append None for missing SCF values
                scf_energies.append(None)
                scf_forces_list.append(None)
            
            # RKS calculation (baseline DFT)
            gpu_mem_before = get_gpu_memory_info()
            
            start_time = time.time()
            rkscalculator.calculate(atoms, properties=["energy", "forces"], system_changes=["positions"])
            rks_time = time.time() - start_time
            rks_times.append(rks_time)
            
            # Get energy and forces from RKS
            rks_energy = rkscalculator.results.get("energy") if hasattr(rkscalculator, 'results') else None
            rks_forces = rkscalculator.results.get("forces") if hasattr(rkscalculator, 'results') else None
            if rks_energy is not None:
                rks_energies.append(float(rks_energy))
            if rks_forces is not None:
                rks_forces_list.append(np.array(rks_forces))
            
            gpu_mem_after = get_gpu_memory_info()
            
            rks_gpu_memory.append({
                "before": gpu_mem_before,
                "after": gpu_mem_after
            })
            
            # Get time_dict and cycles from RKS
            rks_time_dict = None
            if hasattr(rkscalculator, 'time_dict') and rkscalculator.time_dict:
                rks_time_dict = rkscalculator.time_dict.copy()
                rks_time_dicts.append(rks_time_dict)
                
                # Extract individual components
                rks_energy_times.append(rks_time_dict.get("energy_time", 0))
                rks_grad_frame_times.append(rks_time_dict.get("grad_frame_time", 0))
                rks_init_ham_times.append(rks_time_dict.get("init_ham_time", 0))
            
            # Get cycle count from RKS
            rks_cycle = rkscalculator.results.get("cycles") if hasattr(rkscalculator, 'results') else None
            if rks_cycle is not None:
                rks_cycles.append(rks_cycle)
            
            # Calculate energy and force differences for this sample (after all calculations, in meV)
            # RKS vs SCFlow (always calculate if both are available)
            if len(rks_energies) > 0 and len(scflow_energies) > 0:
                if rks_energies[-1] is not None and scflow_energies[-1] is not None:
                    energy_diff = (rks_energies[-1] - scflow_energies[-1]) * 1000  # Convert to meV
                    rks_vs_scflow_energy_diffs.append(energy_diff)
            if len(rks_forces_list) > 0 and len(scflow_forces_list) > 0:
                if rks_forces_list[-1] is not None and scflow_forces_list[-1] is not None:
                    forces_diff = rks_forces_list[-1] - scflow_forces_list[-1]
                    forces_mae = float(np.mean(np.abs(forces_diff))) * 1000  # Convert to meV/Å
                    rks_vs_scflow_forces_mae.append(forces_mae)
            
            # RKS vs SCF (only if SCF succeeded)
            if len(rks_energies) > 0 and len(scf_energies) > 0:
                if rks_energies[-1] is not None and scf_energies[-1] is not None:
                    energy_diff = (rks_energies[-1] - scf_energies[-1]) * 1000  # Convert to meV
                    rks_vs_scf_energy_diffs.append(energy_diff)
            if len(rks_forces_list) > 0 and len(scf_forces_list) > 0:
                if rks_forces_list[-1] is not None and scf_forces_list[-1] is not None:
                    forces_diff = rks_forces_list[-1] - scf_forces_list[-1]
                    forces_mae = float(np.mean(np.abs(forces_diff))) * 1000  # Convert to meV/Å
                    rks_vs_scf_forces_mae.append(forces_mae)
            
            # SCFlow vs SCF (only if SCF succeeded)
            if len(scflow_energies) > 0 and len(scf_energies) > 0:
                if scflow_energies[-1] is not None and scf_energies[-1] is not None:
                    energy_diff = (scflow_energies[-1] - scf_energies[-1]) * 1000  # Convert to meV
                    scflow_vs_scf_energy_diffs.append(energy_diff)
            if len(scflow_forces_list) > 0 and len(scf_forces_list) > 0:
                if scflow_forces_list[-1] is not None and scf_forces_list[-1] is not None:
                    forces_diff = scflow_forces_list[-1] - scf_forces_list[-1]
                    forces_mae = float(np.mean(np.abs(forces_diff))) * 1000  # Convert to meV/Å
                    scflow_vs_scf_forces_mae.append(forces_mae)
        
        except Exception as e:
            print(f"    Error processing sample {sample_idx}: {e}")
            continue
    
    # Calculate statistics for this atom count
    stats = {
        "atom_count": atom_count,
        "num_samples": len(samples),
        "scflow": {
            "total_time": calculate_statistics(scflow_times),
            "mf_time": calculate_statistics(scflow_mf_times),
            "ovlp_time": calculate_statistics(scflow_ovlp_times),
            "init_ham_time": calculate_statistics(scflow_init_ham_times),
            "pred_ham_time": calculate_statistics(scflow_pred_ham_times),
            "pred_dm0_time": calculate_statistics(scflow_pred_dm0_times),
            "energy_time": calculate_statistics(scflow_energy_times),
            "grad_frame_time": calculate_statistics(scflow_grad_frame_times),
            "model_time": calculate_statistics(scflow_model_times),  # pred_ham + pred_dm0
        },
        "rks": {
            "total_time": calculate_statistics(rks_times),
            "energy_time": calculate_statistics(rks_energy_times),
            "grad_frame_time": calculate_statistics(rks_grad_frame_times),
            "init_ham_time": calculate_statistics(rks_init_ham_times),
            "cycles": calculate_statistics(rks_cycles),
        },
        "scf": {
            "total_time": calculate_statistics(scf_times),
            "mf_time": calculate_statistics(scf_mf_times),
            "ovlp_time": calculate_statistics(scf_ovlp_times),
            "init_ham_time": calculate_statistics(scf_init_ham_times),
            "pred_ham_time": calculate_statistics(scf_pred_ham_times),
            "pred_dm0_time": calculate_statistics(scf_pred_dm0_times),
            "energy_time": calculate_statistics(scf_energy_times),
            "grad_frame_time": calculate_statistics(scf_grad_frame_times),
            "model_time": calculate_statistics(scf_model_times),  # pred_ham + pred_dm0
            "cycles": calculate_statistics(scf_cycles),
        },
        "differences": {
            "rks_vs_scflow": {
                "energy_diff": calculate_statistics(rks_vs_scflow_energy_diffs),
                "forces_mae": calculate_statistics(rks_vs_scflow_forces_mae),
            },
            "rks_vs_scf": {
                "energy_diff": calculate_statistics(rks_vs_scf_energy_diffs),
                "forces_mae": calculate_statistics(rks_vs_scf_forces_mae),
            },
            "scflow_vs_scf": {
                "energy_diff": calculate_statistics(scflow_vs_scf_energy_diffs),
                "forces_mae": calculate_statistics(scflow_vs_scf_forces_mae),
            },
        },
        "speedup": {}
    }
    
    # Calculate speedup ratios
    if stats["rks"]["total_time"] and stats["scflow"]["total_time"]:
        rks_mean = stats["rks"]["total_time"]["mean"]
        scflow_mean = stats["scflow"]["total_time"]["mean"]
        if scflow_mean > 0:
            stats["speedup"]["rks_vs_scflow"] = rks_mean / scflow_mean
        else:
            stats["speedup"]["rks_vs_scflow"] = None
    
    if stats["rks"]["total_time"] and stats["scf"]["total_time"]:
        rks_mean = stats["rks"]["total_time"]["mean"]
        scf_mean = stats["scf"]["total_time"]["mean"]
        if scf_mean > 0:
            stats["speedup"]["rks_vs_scf"] = rks_mean / scf_mean
        else:
            stats["speedup"]["rks_vs_scf"] = None
    
    if stats["scflow"]["total_time"] and stats["scf"]["total_time"]:
        scflow_mean = stats["scflow"]["total_time"]["mean"]
        scf_mean = stats["scf"]["total_time"]["mean"]
        if scf_mean > 0:
            stats["speedup"]["scflow_vs_scf"] = scflow_mean / scf_mean
        else:
            stats["speedup"]["scflow_vs_scf"] = None
    
    # Model time speedup
    if stats["scflow"]["model_time"] and stats["scf"]["model_time"]:
        scflow_model_mean = stats["scflow"]["model_time"]["mean"]
        scf_model_mean = stats["scf"]["model_time"]["mean"]
        if scf_model_mean > 0:
            stats["speedup"]["scflow_model_vs_scf_model"] = scflow_model_mean / scf_model_mean
        else:
            stats["speedup"]["scflow_model_vs_scf_model"] = None
    
    results_by_atom_count[atom_count] = stats
    
    # Store individual samples for detailed analysis
    for i, (sample_idx, atoms) in enumerate(samples):
        if i < len(scflow_times):
            # Get cycle counts for this sample
            rks_cycle_val = rks_cycles[i] if i < len(rks_cycles) else None
            scf_cycle_val = scf_cycles[i] if i < len(scf_cycles) else None
            
            # Get energy and forces for this sample
            scflow_energy_val = scflow_energies[i] if i < len(scflow_energies) else None
            scflow_forces_val = scflow_forces_list[i] if i < len(scflow_forces_list) else None
            rks_energy_val = rks_energies[i] if i < len(rks_energies) else None
            rks_forces_val = rks_forces_list[i] if i < len(rks_forces_list) else None
            scf_energy_val = scf_energies[i] if i < len(scf_energies) else None
            scf_forces_val = scf_forces_list[i] if i < len(scf_forces_list) else None
            
            sample_result = {
                "sample_idx": int(sample_idx),
                "atom_count": atom_count,
                "scflow_total_time": scflow_times[i] if i < len(scflow_times) else None,
                "scflow_time_dict": scflow_time_dicts[i] if i < len(scflow_time_dicts) else None,
                "scflow_energy": scflow_energy_val,
                "scflow_forces": scflow_forces_val.tolist() if scflow_forces_val is not None else None,
                "rks_total_time": rks_times[i] if i < len(rks_times) else None,
                "rks_time_dict": rks_time_dicts[i] if i < len(rks_time_dicts) else None,
                "rks_energy": rks_energy_val,
                "rks_forces": rks_forces_val.tolist() if rks_forces_val is not None else None,
                "rks_cycles": rks_cycle_val,
                "scf_total_time": scf_times[i] if i < len(scf_times) else None,
                "scf_time_dict": scf_time_dicts[i] if i < len(scf_time_dicts) else None,
                "scf_energy": scf_energy_val,
                "scf_forces": scf_forces_val.tolist() if scf_forces_val is not None else None,
                "scf_cycles": scf_cycle_val,
                "scflow_gpu_memory": scflow_gpu_memory[i] if i < len(scflow_gpu_memory) else None,
                "rks_gpu_memory": rks_gpu_memory[i] if i < len(rks_gpu_memory) else None,
                "scf_gpu_memory": scf_gpu_memory[i] if i < len(scf_gpu_memory) else None,
            }
            if sample_result["scflow_time_dict"]:
                sample_result["scflow_model_time"] = (
                    sample_result["scflow_time_dict"].get("pred_ham_time", 0) + 
                    sample_result["scflow_time_dict"].get("pred_dm0_time", 0)
                )
            if sample_result["scf_time_dict"]:
                sample_result["scf_model_time"] = (
                    sample_result["scf_time_dict"].get("pred_ham_time", 0) + 
                    sample_result["scf_time_dict"].get("pred_dm0_time", 0)
                )
            
            # Calculate cycle reduction for this sample
            if rks_cycle_val is not None and scf_cycle_val is not None:
                cycle_reduction = ((rks_cycle_val - scf_cycle_val) / rks_cycle_val * 100) if rks_cycle_val > 0 else None
                sample_result["cycle_reduction_percent"] = cycle_reduction
                sample_result["cycle_reduction_ratio"] = (rks_cycle_val / scf_cycle_val) if scf_cycle_val > 0 else None
            else:
                sample_result["cycle_reduction_percent"] = None
                sample_result["cycle_reduction_ratio"] = None
            
            # Calculate energy and force differences for this sample (in meV)
            # RKS vs SCFlow
            if rks_energy_val is not None and scflow_energy_val is not None:
                sample_result["rks_vs_scflow_energy_diff"] = (rks_energy_val - scflow_energy_val) * 1000  # Convert to meV
            else:
                sample_result["rks_vs_scflow_energy_diff"] = None
            
            if rks_forces_val is not None and scflow_forces_val is not None:
                forces_diff = rks_forces_val - scflow_forces_val
                sample_result["rks_vs_scflow_forces_mae"] = float(np.mean(np.abs(forces_diff))) * 1000  # Convert to meV/Å
            else:
                sample_result["rks_vs_scflow_forces_mae"] = None
            
            # RKS vs SCF
            if rks_energy_val is not None and scf_energy_val is not None:
                sample_result["rks_vs_scf_energy_diff"] = (rks_energy_val - scf_energy_val) * 1000  # Convert to meV
            else:
                sample_result["rks_vs_scf_energy_diff"] = None
            
            if rks_forces_val is not None and scf_forces_val is not None:
                forces_diff = rks_forces_val - scf_forces_val
                sample_result["rks_vs_scf_forces_mae"] = float(np.mean(np.abs(forces_diff))) * 1000  # Convert to meV/Å
            else:
                sample_result["rks_vs_scf_forces_mae"] = None
            
            # SCFlow vs SCF
            if scflow_energy_val is not None and scf_energy_val is not None:
                sample_result["scflow_vs_scf_energy_diff"] = (scflow_energy_val - scf_energy_val) * 1000  # Convert to meV
            else:
                sample_result["scflow_vs_scf_energy_diff"] = None
            
            if scflow_forces_val is not None and scf_forces_val is not None:
                forces_diff = scflow_forces_val - scf_forces_val
                sample_result["scflow_vs_scf_forces_mae"] = float(np.mean(np.abs(forces_diff))) * 1000  # Convert to meV/Å
            else:
                sample_result["scflow_vs_scf_forces_mae"] = None
            
            all_samples.append(sample_result)

# Get command line used to run this script
command = " ".join(sys.argv)

# Create summary report
summary = {
    "model_name": model_name,
    "model_checkpoint": cur_ckpt,
    "command": command,
    "total_samples": len(test_atoms_list),
    "atom_count_groups": len(atom_count_groups),
    "results_by_atom_count": results_by_atom_count,
    "all_samples": all_samples
}

# Save results
# Add timestamp to output directory name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/root/25DFT/QHFlow/src/md/test_script_3/inference_time_results_{model_name}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Save JSON results
json_file = os.path.join(output_dir, "inference_time_statistics.json")
with open(json_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {json_file}")

# Create CSV file with all sample data
csv_data = []
for sample in all_samples:
    row = {
        "sample_idx": sample.get("sample_idx"),
        "atom_count": sample.get("atom_count"),
        # SCFlow times
        "scflow_total_time": sample.get("scflow_total_time"),
        "scflow_model_time": sample.get("scflow_model_time"),
        "scflow_mf_time": sample.get("scflow_time_dict", {}).get("mf_time") if sample.get("scflow_time_dict") else None,
        "scflow_ovlp_time": sample.get("scflow_time_dict", {}).get("ovlp_time") if sample.get("scflow_time_dict") else None,
        "scflow_init_ham_time": sample.get("scflow_time_dict", {}).get("init_ham_time") if sample.get("scflow_time_dict") else None,
        "scflow_pred_ham_time": sample.get("scflow_time_dict", {}).get("pred_ham_time") if sample.get("scflow_time_dict") else None,
        "scflow_pred_dm0_time": sample.get("scflow_time_dict", {}).get("pred_dm0_time") if sample.get("scflow_time_dict") else None,
        "scflow_energy_time": sample.get("scflow_time_dict", {}).get("energy_time") if sample.get("scflow_time_dict") else None,
        "scflow_grad_frame_time": sample.get("scflow_time_dict", {}).get("grad_frame_time") if sample.get("scflow_time_dict") else None,
        # RKS times
        "rks_total_time": sample.get("rks_total_time"),
        "rks_energy_time": sample.get("rks_time_dict", {}).get("energy_time") if sample.get("rks_time_dict") else None,
        "rks_grad_frame_time": sample.get("rks_time_dict", {}).get("grad_frame_time") if sample.get("rks_time_dict") else None,
        "rks_init_ham_time": sample.get("rks_time_dict", {}).get("init_ham_time") if sample.get("rks_time_dict") else None,
        "rks_cycles": sample.get("rks_cycles"),
        # SCF times
        "scf_total_time": sample.get("scf_total_time"),
        "scf_model_time": sample.get("scf_model_time"),
        "scf_mf_time": sample.get("scf_time_dict", {}).get("mf_time") if sample.get("scf_time_dict") else None,
        "scf_ovlp_time": sample.get("scf_time_dict", {}).get("ovlp_time") if sample.get("scf_time_dict") else None,
        "scf_init_ham_time": sample.get("scf_time_dict", {}).get("init_ham_time") if sample.get("scf_time_dict") else None,
        "scf_pred_ham_time": sample.get("scf_time_dict", {}).get("pred_ham_time") if sample.get("scf_time_dict") else None,
        "scf_pred_dm0_time": sample.get("scf_time_dict", {}).get("pred_dm0_time") if sample.get("scf_time_dict") else None,
        "scf_energy_time": sample.get("scf_time_dict", {}).get("energy_time") if sample.get("scf_time_dict") else None,
        "scf_grad_frame_time": sample.get("scf_time_dict", {}).get("grad_frame_time") if sample.get("scf_time_dict") else None,
        "scf_cycles": sample.get("scf_cycles"),
        "cycle_reduction_percent": sample.get("cycle_reduction_percent"),
        "cycle_reduction_ratio": sample.get("cycle_reduction_ratio"),
        # Speedup ratios
        "speedup_rks_vs_scflow": sample.get("rks_total_time") / sample.get("scflow_total_time") if (sample.get("rks_total_time") and sample.get("scflow_total_time") and sample.get("scflow_total_time") > 0) else None,
        "speedup_rks_vs_scf": sample.get("rks_total_time") / sample.get("scf_total_time") if (sample.get("rks_total_time") and sample.get("scf_total_time") and sample.get("scf_total_time") > 0) else None,
        "speedup_scflow_vs_scf": sample.get("scflow_total_time") / sample.get("scf_total_time") if (sample.get("scflow_total_time") and sample.get("scf_total_time") and sample.get("scf_total_time") > 0) else None,
        # SCFlow GPU memory
        "scflow_gpu_mem_allocated_before_gb": sample.get("scflow_gpu_memory", {}).get("before", {}).get("allocated_gb") if sample.get("scflow_gpu_memory") else None,
        "scflow_gpu_mem_allocated_after_gb": sample.get("scflow_gpu_memory", {}).get("after", {}).get("allocated_gb") if sample.get("scflow_gpu_memory") else None,
        "scflow_gpu_mem_reserved_before_gb": sample.get("scflow_gpu_memory", {}).get("before", {}).get("reserved_gb") if sample.get("scflow_gpu_memory") else None,
        "scflow_gpu_mem_reserved_after_gb": sample.get("scflow_gpu_memory", {}).get("after", {}).get("reserved_gb") if sample.get("scflow_gpu_memory") else None,
        "scflow_gpu_mem_max_allocated_gb": sample.get("scflow_gpu_memory", {}).get("after", {}).get("max_allocated_gb") if sample.get("scflow_gpu_memory") else None,
        # RKS GPU memory
        "rks_gpu_mem_allocated_before_gb": sample.get("rks_gpu_memory", {}).get("before", {}).get("allocated_gb") if sample.get("rks_gpu_memory") else None,
        "rks_gpu_mem_allocated_after_gb": sample.get("rks_gpu_memory", {}).get("after", {}).get("allocated_gb") if sample.get("rks_gpu_memory") else None,
        "rks_gpu_mem_reserved_before_gb": sample.get("rks_gpu_memory", {}).get("before", {}).get("reserved_gb") if sample.get("rks_gpu_memory") else None,
        "rks_gpu_mem_reserved_after_gb": sample.get("rks_gpu_memory", {}).get("after", {}).get("reserved_gb") if sample.get("rks_gpu_memory") else None,
        "rks_gpu_mem_max_allocated_gb": sample.get("rks_gpu_memory", {}).get("after", {}).get("max_allocated_gb") if sample.get("rks_gpu_memory") else None,
        # SCF GPU memory
        "scf_gpu_mem_allocated_before_gb": sample.get("scf_gpu_memory", {}).get("before", {}).get("allocated_gb") if sample.get("scf_gpu_memory") else None,
        "scf_gpu_mem_allocated_after_gb": sample.get("scf_gpu_memory", {}).get("after", {}).get("allocated_gb") if sample.get("scf_gpu_memory") else None,
        "scf_gpu_mem_reserved_before_gb": sample.get("scf_gpu_memory", {}).get("before", {}).get("reserved_gb") if sample.get("scf_gpu_memory") else None,
        "scf_gpu_mem_reserved_after_gb": sample.get("scf_gpu_memory", {}).get("after", {}).get("reserved_gb") if sample.get("scf_gpu_memory") else None,
        "scf_gpu_mem_max_allocated_gb": sample.get("scf_gpu_memory", {}).get("after", {}).get("max_allocated_gb") if sample.get("scf_gpu_memory") else None,
        # Energy and force differences
        "rks_vs_scflow_energy_diff": sample.get("rks_vs_scflow_energy_diff"),
        "rks_vs_scflow_forces_mae": sample.get("rks_vs_scflow_forces_mae"),
        "rks_vs_scf_energy_diff": sample.get("rks_vs_scf_energy_diff"),
        "rks_vs_scf_forces_mae": sample.get("rks_vs_scf_forces_mae"),
        "scflow_vs_scf_energy_diff": sample.get("scflow_vs_scf_energy_diff"),
        "scflow_vs_scf_forces_mae": sample.get("scflow_vs_scf_forces_mae"),
    }
    csv_data.append(row)

# Save CSV file for individual samples
df = pd.DataFrame(csv_data)
csv_file = os.path.join(output_dir, "inference_time_samples.csv")
df.to_csv(csv_file, index=False)
print(f"CSV results saved to {csv_file}")

# Create CSV file with statistics by atom count
stats_csv_data = []
for atom_count in sorted(results_by_atom_count.keys()):
    stats = results_by_atom_count[atom_count]
    row = {
        "atom_count": atom_count,
        "num_samples": stats["num_samples"],
    }
    
    # Helper function to extract stats values
    def get_stat_value(stat_dict, key):
        if stat_dict and stat_dict.get(key):
            return stat_dict[key]
        return None
    
    # SCFlow statistics
    if stats["scflow"]["total_time"]:
        t = stats["scflow"]["total_time"]
        row["scflow_total_time_mean"] = t["mean"]
        row["scflow_total_time_std"] = t["std"]
        row["scflow_total_time_min"] = t["min"]
        row["scflow_total_time_max"] = t["max"]
        row["scflow_total_time_median"] = t["median"]
    
    if stats["scflow"]["model_time"]:
        t = stats["scflow"]["model_time"]
        row["scflow_model_time_mean"] = t["mean"]
        row["scflow_model_time_std"] = t["std"]
        row["scflow_model_time_min"] = t["min"]
        row["scflow_model_time_max"] = t["max"]
        row["scflow_model_time_median"] = t["median"]
    
    if stats["scflow"]["mf_time"]:
        t = stats["scflow"]["mf_time"]
        row["scflow_mf_time_mean"] = t["mean"]
        row["scflow_mf_time_std"] = t["std"]
    
    if stats["scflow"]["ovlp_time"]:
        t = stats["scflow"]["ovlp_time"]
        row["scflow_ovlp_time_mean"] = t["mean"]
        row["scflow_ovlp_time_std"] = t["std"]
    
    if stats["scflow"]["init_ham_time"]:
        t = stats["scflow"]["init_ham_time"]
        row["scflow_init_ham_time_mean"] = t["mean"]
        row["scflow_init_ham_time_std"] = t["std"]
    
    if stats["scflow"]["pred_ham_time"]:
        t = stats["scflow"]["pred_ham_time"]
        row["scflow_pred_ham_time_mean"] = t["mean"]
        row["scflow_pred_ham_time_std"] = t["std"]
    
    if stats["scflow"]["pred_dm0_time"]:
        t = stats["scflow"]["pred_dm0_time"]
        row["scflow_pred_dm0_time_mean"] = t["mean"]
        row["scflow_pred_dm0_time_std"] = t["std"]
    
    if stats["scflow"]["energy_time"]:
        t = stats["scflow"]["energy_time"]
        row["scflow_energy_time_mean"] = t["mean"]
        row["scflow_energy_time_std"] = t["std"]
    
    if stats["scflow"]["grad_frame_time"]:
        t = stats["scflow"]["grad_frame_time"]
        row["scflow_grad_frame_time_mean"] = t["mean"]
        row["scflow_grad_frame_time_std"] = t["std"]
    
    # RKS statistics
    if stats["rks"]["total_time"]:
        t = stats["rks"]["total_time"]
        row["rks_total_time_mean"] = t["mean"]
        row["rks_total_time_std"] = t["std"]
        row["rks_total_time_min"] = t["min"]
        row["rks_total_time_max"] = t["max"]
        row["rks_total_time_median"] = t["median"]
    
    if stats["rks"]["energy_time"]:
        t = stats["rks"]["energy_time"]
        row["rks_energy_time_mean"] = t["mean"]
        row["rks_energy_time_std"] = t["std"]
    
    if stats["rks"]["grad_frame_time"]:
        t = stats["rks"]["grad_frame_time"]
        row["rks_grad_frame_time_mean"] = t["mean"]
        row["rks_grad_frame_time_std"] = t["std"]
    
    if stats["rks"]["init_ham_time"]:
        t = stats["rks"]["init_ham_time"]
        row["rks_init_ham_time_mean"] = t["mean"]
        row["rks_init_ham_time_std"] = t["std"]
    
    # SCF statistics
    if stats["scf"]["total_time"]:
        t = stats["scf"]["total_time"]
        row["scf_total_time_mean"] = t["mean"]
        row["scf_total_time_std"] = t["std"]
        row["scf_total_time_min"] = t["min"]
        row["scf_total_time_max"] = t["max"]
        row["scf_total_time_median"] = t["median"]
    
    if stats["scf"]["model_time"]:
        t = stats["scf"]["model_time"]
        row["scf_model_time_mean"] = t["mean"]
        row["scf_model_time_std"] = t["std"]
    
    if stats["scf"]["mf_time"]:
        t = stats["scf"]["mf_time"]
        row["scf_mf_time_mean"] = t["mean"]
        row["scf_mf_time_std"] = t["std"]
    
    if stats["scf"]["ovlp_time"]:
        t = stats["scf"]["ovlp_time"]
        row["scf_ovlp_time_mean"] = t["mean"]
        row["scf_ovlp_time_std"] = t["std"]
    
    if stats["scf"]["init_ham_time"]:
        t = stats["scf"]["init_ham_time"]
        row["scf_init_ham_time_mean"] = t["mean"]
        row["scf_init_ham_time_std"] = t["std"]
    
    if stats["scf"]["pred_ham_time"]:
        t = stats["scf"]["pred_ham_time"]
        row["scf_pred_ham_time_mean"] = t["mean"]
        row["scf_pred_ham_time_std"] = t["std"]
    
    if stats["scf"]["pred_dm0_time"]:
        t = stats["scf"]["pred_dm0_time"]
        row["scf_pred_dm0_time_mean"] = t["mean"]
        row["scf_pred_dm0_time_std"] = t["std"]
    
    if stats["scf"]["energy_time"]:
        t = stats["scf"]["energy_time"]
        row["scf_energy_time_mean"] = t["mean"]
        row["scf_energy_time_std"] = t["std"]
    
    if stats["scf"]["grad_frame_time"]:
        t = stats["scf"]["grad_frame_time"]
        row["scf_grad_frame_time_mean"] = t["mean"]
        row["scf_grad_frame_time_std"] = t["std"]
    
    if stats["scf"]["cycles"]:
        t = stats["scf"]["cycles"]
        row["scf_cycles_mean"] = t["mean"]
        row["scf_cycles_std"] = t["std"]
        row["scf_cycles_min"] = t["min"]
        row["scf_cycles_max"] = t["max"]
        row["scf_cycles_median"] = t["median"]
    
    # Speedup ratios
    if stats["speedup"].get("rks_vs_scflow"):
        row["speedup_rks_vs_scflow"] = stats["speedup"]["rks_vs_scflow"]
    if stats["speedup"].get("rks_vs_scf"):
        row["speedup_rks_vs_scf"] = stats["speedup"]["rks_vs_scf"]
    if stats["speedup"].get("scflow_vs_scf"):
        row["speedup_scflow_vs_scf"] = stats["speedup"]["scflow_vs_scf"]
    if stats["speedup"].get("scflow_model_vs_scf_model"):
        row["speedup_scflow_model_vs_scf_model"] = stats["speedup"]["scflow_model_vs_scf_model"]
    if stats["speedup"].get("cycle_reduction_percent") is not None:
        row["cycle_reduction_percent"] = stats["speedup"]["cycle_reduction_percent"]
    if stats["speedup"].get("cycle_reduction_ratio"):
        row["cycle_reduction_ratio"] = stats["speedup"]["cycle_reduction_ratio"]
    
    # Energy and force differences
    if stats["differences"]["rks_vs_scflow"]["energy_diff"]:
        t = stats["differences"]["rks_vs_scflow"]["energy_diff"]
        row["rks_vs_scflow_energy_diff_mean"] = t["mean"]
        row["rks_vs_scflow_energy_diff_std"] = t["std"]
    if stats["differences"]["rks_vs_scflow"]["forces_mae"]:
        t = stats["differences"]["rks_vs_scflow"]["forces_mae"]
        row["rks_vs_scflow_forces_mae_mean"] = t["mean"]
        row["rks_vs_scflow_forces_mae_std"] = t["std"]
    if stats["differences"]["rks_vs_scf"]["energy_diff"]:
        t = stats["differences"]["rks_vs_scf"]["energy_diff"]
        row["rks_vs_scf_energy_diff_mean"] = t["mean"]
        row["rks_vs_scf_energy_diff_std"] = t["std"]
    if stats["differences"]["rks_vs_scf"]["forces_mae"]:
        t = stats["differences"]["rks_vs_scf"]["forces_mae"]
        row["rks_vs_scf_forces_mae_mean"] = t["mean"]
        row["rks_vs_scf_forces_mae_std"] = t["std"]
    if stats["differences"]["scflow_vs_scf"]["energy_diff"]:
        t = stats["differences"]["scflow_vs_scf"]["energy_diff"]
        row["scflow_vs_scf_energy_diff_mean"] = t["mean"]
        row["scflow_vs_scf_energy_diff_std"] = t["std"]
    if stats["differences"]["scflow_vs_scf"]["forces_mae"]:
        t = stats["differences"]["scflow_vs_scf"]["forces_mae"]
        row["scflow_vs_scf_forces_mae_mean"] = t["mean"]
        row["scflow_vs_scf_forces_mae_std"] = t["std"]
    
    stats_csv_data.append(row)

# Save statistics CSV file
stats_df = pd.DataFrame(stats_csv_data)
stats_csv_file = os.path.join(output_dir, "inference_time_statistics_by_atom_count.csv")
stats_df.to_csv(stats_csv_file, index=False)
print(f"Statistics CSV saved to {stats_csv_file}")

# Create human-readable report
report_file = os.path.join(output_dir, "inference_time_report.txt")
with open(report_file, 'w') as f:
    f.write("=" * 120 + "\n")
    f.write(f"Inference Time Report: {model_name}\n")
    f.write("=" * 120 + "\n\n")
    f.write(f"Command: {command}\n")
    f.write(f"Model Checkpoint: {cur_ckpt}\n")
    f.write(f"Total Samples: {len(test_atoms_list)}\n")
    f.write(f"Atom Count Groups: {len(atom_count_groups)}\n\n")
    
    for atom_count in sorted(results_by_atom_count.keys()):
        stats = results_by_atom_count[atom_count]
        f.write("=" * 120 + "\n")
        f.write(f"Atom Count: {atom_count} ({stats['num_samples']} samples)\n")
        f.write("=" * 120 + "\n\n")
        
        # SCFlow statistics
        f.write("SCFlow (Model-based) Statistics:\n")
        f.write("-" * 120 + "\n")
        if stats["scflow"]["total_time"]:
            t = stats["scflow"]["total_time"]
            f.write(f"  Total Time:      {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scflow"]["model_time"]:
            t = stats["scflow"]["model_time"]
            f.write(f"  Model Time:      {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scflow"]["pred_ham_time"]:
            t = stats["scflow"]["pred_ham_time"]
            f.write(f"  Pred Ham Time:   {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scflow"]["pred_dm0_time"]:
            t = stats["scflow"]["pred_dm0_time"]
            f.write(f"  Pred DM0 Time:   {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scflow"]["mf_time"]:
            t = stats["scflow"]["mf_time"]
            f.write(f"  MF Time:         {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scflow"]["ovlp_time"]:
            t = stats["scflow"]["ovlp_time"]
            f.write(f"  Overlap Time:    {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scflow"]["init_ham_time"]:
            t = stats["scflow"]["init_ham_time"]
            f.write(f"  Init Ham Time:   {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scflow"]["energy_time"]:
            t = stats["scflow"]["energy_time"]
            f.write(f"  Energy Time:     {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scflow"]["grad_frame_time"]:
            t = stats["scflow"]["grad_frame_time"]
            f.write(f"  Grad Frame Time: {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        f.write("\n")
        
        # RKS statistics
        f.write("RKS (Baseline DFT) Statistics:\n")
        f.write("-" * 120 + "\n")
        if stats["rks"]["total_time"]:
            t = stats["rks"]["total_time"]
            f.write(f"  Total Time:      {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["rks"]["energy_time"]:
            t = stats["rks"]["energy_time"]
            f.write(f"  Energy Time:     {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["rks"]["grad_frame_time"]:
            t = stats["rks"]["grad_frame_time"]
            f.write(f"  Grad Frame Time: {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["rks"]["init_ham_time"]:
            t = stats["rks"]["init_ham_time"]
            f.write(f"  Init Ham Time:   {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["rks"]["cycles"]:
            t = stats["rks"]["cycles"]
            f.write(f"  SCF Cycles:      {t['mean']:>10.1f} ± {t['std']:>8.1f}  (min: {t['min']:>8.0f}, max: {t['max']:>8.0f}, median: {t['median']:>8.1f})\n")
        f.write("\n")
        
        # SCF statistics
        f.write("SCF (SCFlow SCF) Statistics:\n")
        f.write("-" * 120 + "\n")
        if stats["scf"]["total_time"]:
            t = stats["scf"]["total_time"]
            f.write(f"  Total Time:      {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scf"]["model_time"]:
            t = stats["scf"]["model_time"]
            f.write(f"  Model Time:      {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scf"]["pred_ham_time"]:
            t = stats["scf"]["pred_ham_time"]
            f.write(f"  Pred Ham Time:   {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scf"]["pred_dm0_time"]:
            t = stats["scf"]["pred_dm0_time"]
            f.write(f"  Pred DM0 Time:   {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scf"]["mf_time"]:
            t = stats["scf"]["mf_time"]
            f.write(f"  MF Time:         {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scf"]["ovlp_time"]:
            t = stats["scf"]["ovlp_time"]
            f.write(f"  Overlap Time:    {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scf"]["init_ham_time"]:
            t = stats["scf"]["init_ham_time"]
            f.write(f"  Init Ham Time:   {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scf"]["energy_time"]:
            t = stats["scf"]["energy_time"]
            f.write(f"  Energy Time:     {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scf"]["grad_frame_time"]:
            t = stats["scf"]["grad_frame_time"]
            f.write(f"  Grad Frame Time: {t['mean']:>10.4f} ± {t['std']:>8.4f} s  (min: {t['min']:>8.4f}, max: {t['max']:>8.4f}, median: {t['median']:>8.4f})\n")
        if stats["scf"]["cycles"]:
            t = stats["scf"]["cycles"]
            f.write(f"  SCF Cycles:      {t['mean']:>10.1f} ± {t['std']:>8.1f}  (min: {t['min']:>8.0f}, max: {t['max']:>8.0f}, median: {t['median']:>8.1f})\n")
        f.write("\n")
        
        # Speedup statistics
        f.write("Speedup Ratios:\n")
        f.write("-" * 120 + "\n")
        if stats["speedup"].get("rks_vs_scflow"):
            speedup = stats["speedup"]["rks_vs_scflow"]
            f.write(f"  RKS vs SCFlow:   {speedup:>10.4f}x faster (SCFlow is {speedup:.2f}x faster than RKS)\n")
        if stats["speedup"].get("rks_vs_scf"):
            speedup = stats["speedup"]["rks_vs_scf"]
            f.write(f"  RKS vs SCF:      {speedup:>10.4f}x faster (SCF is {speedup:.2f}x faster than RKS)\n")
        if stats["speedup"].get("scflow_vs_scf"):
            speedup = stats["speedup"]["scflow_vs_scf"]
            f.write(f"  SCFlow vs SCF:   {speedup:>10.4f}x faster (SCF is {speedup:.2f}x faster than SCFlow)\n")
        if stats["speedup"].get("scflow_model_vs_scf_model"):
            speedup = stats["speedup"]["scflow_model_vs_scf_model"]
            f.write(f"  SCFlow Model vs SCF Model: {speedup:>10.4f}x faster (SCF model is {speedup:.2f}x faster than SCFlow model)\n")
        if stats["speedup"].get("cycle_reduction_percent") is not None:
            reduction = stats["speedup"]["cycle_reduction_percent"]
            ratio = stats["speedup"].get("cycle_reduction_ratio")
            f.write(f"  Cycle Reduction: {reduction:>10.2f}% reduction (RKS: {stats['rks']['cycles']['mean']:.1f} cycles -> SCF: {stats['scf']['cycles']['mean']:.1f} cycles")
            if ratio:
                f.write(f", {ratio:.2f}x fewer cycles")
            f.write(")\n")
        
        # Energy and force differences
        f.write("\nEnergy and Force Differences:\n")
        f.write("-" * 120 + "\n")
        if stats["differences"]["rks_vs_scflow"]["energy_diff"]:
            t = stats["differences"]["rks_vs_scflow"]["energy_diff"]
            f.write(f"  RKS - SCFlow Energy: {t['mean']:>10.3f} ± {t['std']:>8.3f} meV  (min: {t['min']:>8.3f}, max: {t['max']:>8.3f}, median: {t['median']:>8.3f})\n")
        if stats["differences"]["rks_vs_scflow"]["forces_mae"]:
            t = stats["differences"]["rks_vs_scflow"]["forces_mae"]
            f.write(f"  RKS - SCFlow Forces MAE: {t['mean']:>10.3f} ± {t['std']:>8.3f} meV/Å  (min: {t['min']:>8.3f}, max: {t['max']:>8.3f}, median: {t['median']:>8.3f})\n")
        if stats["differences"]["rks_vs_scf"]["energy_diff"]:
            t = stats["differences"]["rks_vs_scf"]["energy_diff"]
            f.write(f"  RKS - SCF Energy: {t['mean']:>10.3f} ± {t['std']:>8.3f} meV  (min: {t['min']:>8.3f}, max: {t['max']:>8.3f}, median: {t['median']:>8.3f})\n")
        if stats["differences"]["rks_vs_scf"]["forces_mae"]:
            t = stats["differences"]["rks_vs_scf"]["forces_mae"]
            f.write(f"  RKS - SCF Forces MAE: {t['mean']:>10.3f} ± {t['std']:>8.3f} meV/Å  (min: {t['min']:>8.3f}, max: {t['max']:>8.3f}, median: {t['median']:>8.3f})\n")
        if stats["differences"]["scflow_vs_scf"]["energy_diff"]:
            t = stats["differences"]["scflow_vs_scf"]["energy_diff"]
            f.write(f"  SCFlow - SCF Energy: {t['mean']:>10.3f} ± {t['std']:>8.3f} meV  (min: {t['min']:>8.3f}, max: {t['max']:>8.3f}, median: {t['median']:>8.3f})\n")
        if stats["differences"]["scflow_vs_scf"]["forces_mae"]:
            t = stats["differences"]["scflow_vs_scf"]["forces_mae"]
            f.write(f"  SCFlow - SCF Forces MAE: {t['mean']:>10.3f} ± {t['std']:>8.3f} meV/Å  (min: {t['min']:>8.3f}, max: {t['max']:>8.3f}, median: {t['median']:>8.3f})\n")
        f.write("\n\n")

print(f"\nReport saved to {report_file}")
print(f"\nAll results saved to {output_dir}")
print(f"Total samples processed: {len(test_atoms_list)}")

"""
# Example usage:
# Debug mode (10 molecules):
CUDA_VISIBLE_DEVICES=3 python md/test_script_3/inference_time_report.py --model_size small --split random --debug
CUDA_VISIBLE_DEVICES=3 python md/test_script_3/inference_time_report.py --model_size middle --split random --debug
CUDA_VISIBLE_DEVICES=3 python md/test_script_3/inference_time_report.py --model_size middle --split random --debug --init_gpu4pyscf --init_density_fit --use_ref_init_ham

# Full mode:
CUDA_VISIBLE_DEVICES=2 python md/test_script_3/inference_time_report.py --model_size small --split random 
CUDA_VISIBLE_DEVICES=0 python md/test_script_3/inference_time_report.py --model_size middle --split random --init_gpu4pyscf --init_density_fit --use_ref_init_ham --vasp
CUDA_VISIBLE_DEVICES=2 python md/test_script_3/inference_time_report.py --model_size large --split random --init_gpu4pyscf --init_density_fit --use_ref_init_ham --vasp
CUDA_VISIBLE_DEVICES=3 python md/test_script_3/inference_time_report.py --model_size small --split ood
CUDA_VISIBLE_DEVICES=3 python md/test_script_3/inference_time_report.py --model_size small --split geo
CUDA_VISIBLE_DEVICES=3 python md/test_script_3/inference_time_report.py --model_size small --split mol
"""
