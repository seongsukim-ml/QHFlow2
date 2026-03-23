#!/usr/bin/env python3
"""
Final multi-process DFT calculation with optimized data storage
- Multi-processing with optimal core usage
- Thread control (OMP_NUM_THREADS=1 per process)
- Random sampling (30,000 samples) with fixed seed for reproducibility
- Train/Val/Test splits (25000/500/4500) saved
- HDF5 storage for all matrices and results
- Complete data: energy, forces, overlap, density matrix, Hamiltonians, orbitals

Usage:
    # Default: 30,000 random samples
    python final_processing_rmd_gpu.py --molecule aspirin

    # Custom settings
    python final_processing_rmd_gpu.py --molecule benzene --num_samples 10000 --seed 123
    python final_processing_rmd_gpu.py --molecule ethanol --grid_level 3 --n_processes 16
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python final_processing_rmd_gpu.py --molecule naphthalene --n_processes 4
    CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python final_processing_rmd_gpu.py --molecule salicylic --n_processes 4
    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python final_processing_rmd_gpu.py --molecule aspirin     --n_processes 4
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python final_processing_rmd_gpu.py --molecule ethanol     --n_processes 4
    
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python final_processing_rmd_gpu.py --molecule salicylic   --n_processes 63 &&
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python final_processing_rmd_gpu.py --molecule naphthalene --n_processes 63 &&
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python final_processing_rmd_gpu.py --molecule ethanol     --n_processes 63 &&
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python final_processing_rmd_gpu.py --molecule aspirin     --n_processes 63

    # Sequential sampling (not random)
    python final_processing_rmd_gpu.py --molecule toluene --sequential
"""

import numpy as np
import h5py
import sys
import argparse
import time
import os
import concurrent.futures
import multiprocessing as mp
from multiprocessing import cpu_count
from functools import partial
import json
from tqdm import tqdm

# Set thread count to 1 for each process (optimize multi-processing)
# (Does not work when used in code, try to set in the terminal command instead)
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add the source path
# sys.path.insert(0, '/home/yoonho/QHFlow/QHFlow-mlff/src')
# path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
# sys.path.insert(0, path)

try:
    import gpu4pyscf
    from gpu4pyscf.lib.cupy_helper import (tag_array, pack_tril, asarray)
    IMPORT_GPU4PYSCF = True
    print("gpu4pyscf is installed.")
except:
    IMPORT_GPU4PYSCF = False
    print("gpu4pyscf is not installed. Please install it using 'pip install gpu4pyscf'.")

from pyscf import gto, dft
# from qhflow2.common.units import HA2eV, BOHR2ANG

HA2eV = 27.211396641308  
BOHR2ANG = 0.5291772105638411 

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Final DFT processing with multi-processing and HDF5 storage')
    parser.add_argument('--molecule', type=str, default='aspirin',
                        help='Molecule name (aspirin, benzene, ethanol, etc.)')
    parser.add_argument('--num_samples', type=int, default=30000,
                        help='Number of samples to process (default: 30000)')
    parser.add_argument('--n_processes', type=int, default=1,
                        help='Number of processes (default: 1)')
    parser.add_argument('--conv_tol', type=float, default=1e-7,
                        help='SCF convergence tolerance (default: 1e-7)')
    parser.add_argument('--grid_level', type=int, default=3,
                        help='DFT grid level (default: 3)')
    parser.add_argument('--basis', type=str, default='def2-svp',
                        help='Basis set (default: def2-svp)')
    parser.add_argument('--functional', type=str, default='pbe',
                        help='DFT functional (default: pbe)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HDF5 file (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--sequential', action='store_true',
                        help='Use sequential sampling instead of random (default: random sampling)')
    return parser.parse_args()

# Conversion constants
KCAL_TO_EV = 1.0 / 23.06054783  # kcal/mol to eV


# Helper to move GPU arrays to host numpy for HDF5 compatibility
def to_numpy(arr):
    if IMPORT_GPU4PYSCF:
        try:
            import cupy as cp
            if isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)
        except Exception:
            pass
        if hasattr(arr, "get"):
            try:
                return arr.get()
            except Exception:
                pass
    return np.asarray(arr)


def calculate_single_sample(idx, z, pos, energy_ref, force_ref, settings):
    """
    Calculate DFT for a single sample
    Returns: dictionary with all calculated data
    """
    start_time = time.time()

    # Build molecule string
    atom_str = ""
    for i, (zi, ri) in enumerate(zip(z, pos)):
        atom_str += f"{int(zi)} {ri[0]:.10f} {ri[1]:.10f} {ri[2]:.10f}; "

    try:
        # Create molecule
        mol = gto.M(
            atom=atom_str,
            basis=settings['basis'],
            unit='Angstrom',
            verbose=0
        )

        # Setup DFT
        mf = dft.RKS(mol)
        
        # Convert to GPU if requested (lazy initialization)
        use_gpu = settings.get('use_gpu', False) and IMPORT_GPU4PYSCF
        if use_gpu:
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
        # mf.small_rho_cutoff = 1e-12

        # Get initial overlap and Hamiltonian
        ovlp = mf.get_ovlp()
        h_core = mf.get_hcore()
        
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
        
        # Convert GPU arrays to numpy arrays if using GPU
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

        # Get additional energy components
        e_nuc = mf.energy_nuc()  # Nuclear repulsion
        e_tot = mf.e_tot  # Total energy

        # Get grid info
        ngrids = mf.grids.weights.size if hasattr(mf.grids, 'weights') else 0

        calc_time = time.time() - start_time

        # Prepare result dictionary
        result = {
            'idx': idx,
            'success': True,
            'converged': converged,
            'calc_time': calc_time,

            # Input data
            'positions': pos,  # Angstrom
            'atomic_numbers': z,

            # Reference data (rMD17)
            'energy_ref_kcal': energy_ref,  # kcal/mol
            'energy_ref_ev': energy_ref * KCAL_TO_EV,  # eV
            'forces_ref_kcal': force_ref,  # kcal/mol/Angstrom
            'forces_ref_ev': force_ref * KCAL_TO_EV,  # eV/Angstrom

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
            'error_energy_ev': abs(E_dft * HA2eV - energy_ref * KCAL_TO_EV),
            'error_energy_mev': abs(E_dft * HA2eV - energy_ref * KCAL_TO_EV) * 1000,
            'error_forces_mae': np.mean(np.abs(forces * (HA2eV / BOHR2ANG) - force_ref * KCAL_TO_EV)),

            # Metadata
            'ngrids': ngrids,
            'nbasis': ovlp.shape[0],
            'natoms': len(z),
        }

        return result

    except Exception as e:
        calc_time = time.time() - start_time
        return {
            'idx': idx,
            'success': False,
            'converged': False,
            'calc_time': calc_time,
            'error_message': str(e),
            'positions': pos,
            'atomic_numbers': z,
            'energy_ref_kcal': energy_ref,
            'forces_ref_kcal': force_ref,
        }


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


if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()

    # Set default number of processes
    if args.n_processes is None:
        args.n_processes = max(1, cpu_count() - 1)  # Leave 1 core for system

    # Auto-generate output filename
    if args.output is None:
        gpu_tag = "_gpu4pyscf" if IMPORT_GPU4PYSCF else ""
        args.output = f"final_{args.molecule}_conv{args.conv_tol:.0e}_grid{args.grid_level}_n{args.num_samples}{gpu_tag}.h5"
    elif IMPORT_GPU4PYSCF and "gpu4pyscf" not in args.output:
        root, ext = os.path.splitext(args.output)
        ext = ext if ext else ".h5"
        args.output = f"{root}_gpu4pyscf{ext}"

    print("=" * 80)
    print("Final DFT Processing with Multi-Processing")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Molecule:      {args.molecule}")
    print(f"  Samples:       {args.num_samples}")
    print(f"  Sampling:      {'Sequential' if args.sequential else 'Random'}")
    if not args.sequential:
        print(f"  Random seed:   {args.seed}")
    print(f"  Functional:    {args.functional.upper()}")
    print(f"  Basis:         {args.basis}")
    print(f"  Conv_tol:      {args.conv_tol:.0e}")
    print(f"  Grid level:    {args.grid_level}")
    print(f"\nParallel Processing:")
    print(f"  CPU cores:     {cpu_count()}")
    print(f"  Processes:     {args.n_processes}")
    print(f"  Threads/proc:  1 (OMP_NUM_THREADS=1)")
    print(f"\nOutput:")
    print(f"  HDF5 file:     {args.output}")
    print("=" * 80)

    # Load rMD17 data
    # 임시 제거 rmd17_path = os.path.join(path, 'data', 'rmd17', 'npz_data', f'rmd17_{args.molecule}.npz')
    # rmd17_path = f'/home/yoonho/QHFlow/rmd17/npz_data/rmd17_{args.molecule}.npz'
    rmd17_path = f"/root/25DFT/data/rmd17/npz_data/rmd17_{args.molecule}.npz"
    # rmd17_path = f'/home/yoonho/QHFlow/rmd17/npz_data/rmd17_{args.molecule}.npz'
    if not os.path.exists(rmd17_path):
        print(f"ERROR: rMD17 file not found: {rmd17_path}")
        sys.exit(1)

    print(f"\nLoading rMD17 data: {rmd17_path}")
    data = np.load(rmd17_path)

    nuclear_charges = data['nuclear_charges']
    total_structures = len(data['coords'])

    print(f"  Total available structures: {total_structures:,}")

    # Sample indices
    if args.sequential:
        # Sequential sampling
        if args.num_samples > total_structures:
            print(f"  WARNING: Requested {args.num_samples} samples but only {total_structures} available")
            args.num_samples = total_structures
        sample_indices = np.arange(args.num_samples)
        print(f"  Using sequential indices: 0 to {args.num_samples-1}")
    else:
        # Random sampling with fixed seed
        np.random.seed(args.seed)
        if args.num_samples > total_structures:
            print(f"  WARNING: Requested {args.num_samples} samples but only {total_structures} available")
            args.num_samples = total_structures
        sample_indices = np.random.choice(total_structures, size=args.num_samples, replace=False)
        sample_indices = np.sort(sample_indices)  # Sort for better cache locality
        print(f"  Randomly sampled {args.num_samples} indices (seed={args.seed})")
        print(f"  Index range: [{sample_indices.min()}, {sample_indices.max()}]")

    # Load selected samples
    coords = data['coords'][sample_indices]
    energies_ref = data['energies'][sample_indices]
    forces_ref = data['forces'][sample_indices]

    print(f"\n  Loaded {len(coords)} structures")
    print(f"  Atoms per molecule: {len(nuclear_charges)}")
    print(f"  Energy range: [{energies_ref.min():.2f}, {energies_ref.max():.2f}] kcal/mol")

    # Generate train/val/test splits (seed fixed for reproducibility)
    np.random.seed(args.seed + 1)  # Different seed for split
    n_total = len(sample_indices)
    n_train = 25000 if n_total >= 30000 else int(0.833 * n_total)  # 25000/30000 ≈ 0.833
    n_val = 500 if n_total >= 30000 else int(0.017 * n_total)      # 500/30000 ≈ 0.017
    n_test = n_total - n_train - n_val

    split_indices = np.random.permutation(n_total)
    train_idx = split_indices[:n_train]
    val_idx = split_indices[n_train:n_train+n_val]
    test_idx = split_indices[n_train+n_val:]

    print(f"\n  Train/Val/Test Split (seed={args.seed+1}):")
    print(f"    Train: {n_train:,} samples ({n_train/n_total*100:.1f}%)")
    print(f"    Val:   {n_val:,} samples ({n_val/n_total*100:.1f}%)")
    print(f"    Test:  {n_test:,} samples ({n_test/n_total*100:.1f}%)")

    # Prepare settings
    settings = {
        'basis': args.basis,
        'functional': args.functional,
        'conv_tol': args.conv_tol,
        'grid_level': args.grid_level,
        'use_gpu': IMPORT_GPU4PYSCF,  # Enable GPU if available
    }

    # Prepare work items (use sample_indices as idx for traceability)
    work_items = [
        (sample_indices[i], nuclear_charges, coords[i], energies_ref[i], forces_ref[i], settings)
        for i in range(len(coords))
    ]

    print(f"\n{'='*80}")
    print(f"Starting parallel processing...")
    print(f"{'='*80}\n")

    # Run parallel processing
    start_time = time.time()

    # Calculate optimal chunksize for better load balancing
    chunksize = max(1, len(work_items) // (args.n_processes * 4))
    print(f"\nChunksize: {chunksize}")

    results = []
    print(f"Processing {len(work_items)} samples with {args.n_processes} processes")
    print(f"Note: First sample may take 10-30 seconds to initialize...")
    if args.n_processes == 1:
        # Sequential processing (no Pool)
        with tqdm(total=len(work_items),
                  desc="DFT Calculations",
                  unit="sample",
                  ncols=100,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for item in work_items:
                result = worker_wrapper(item)
                results.append(result)
                # Update tqdm with additional info
                if result['success']:
                    error_mev = result.get('error_energy_mev', 0)
                    pbar.set_postfix({'Last_error': f'{error_mev:.1f}meV',
                                      'Converged': result.get('converged', False)})
                else:
                    pbar.set_postfix({'Status': 'FAILED'})
                pbar.update(1)
    else:
        # Parallel processing
        # Use spawn context to avoid CUDA initialization errors (same as final_processing_md17.py)
        ctx = mp.get_context('spawn')
        initargs = (settings.get('use_gpu', False),) if settings.get('use_gpu', False) else ()
        with ctx.Pool(processes=args.n_processes, initializer=init_worker_mp, initargs=initargs) as pool:
            # Use chunksize=1 for better progress visibility
            results = []
            with tqdm(total=len(work_items),
                      desc="DFT Calculations",
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
    print(f"Processing complete!")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Rate: {len(work_items)/total_time:.2f} samples/s")
    print(f"{'='*80}\n")

    # Analyze results
    n_success = sum(1 for r in results if r['success'])
    n_converged = sum(1 for r in results if r.get('converged', False))

    print("Results Summary:")
    print(f"  Successful: {n_success}/{len(results)}")
    print(f"  Converged:  {n_converged}/{len(results)}")

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

    # Save to HDF5
    print(f"\n{'='*80}")
    print(f"Saving to HDF5: {args.output}")
    print(f"{'='*80}\n")

    with h5py.File(args.output, 'w') as f:
        # Save metadata
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['molecule'] = args.molecule
        meta_grp.attrs['functional'] = args.functional
        meta_grp.attrs['basis'] = args.basis
        meta_grp.attrs['conv_tol'] = args.conv_tol
        meta_grp.attrs['grid_level'] = args.grid_level
        meta_grp.attrs['num_samples'] = args.num_samples
        meta_grp.attrs['sampling_method'] = 'sequential' if args.sequential else 'random'
        meta_grp.attrs['seed'] = args.seed
        meta_grp.attrs['n_processes'] = args.n_processes
        meta_grp.attrs['total_time_sec'] = total_time
        meta_grp.attrs['n_success'] = n_success
        meta_grp.attrs['n_converged'] = n_converged
        meta_grp.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

        # Save sample indices for traceability
        meta_grp.create_dataset('sample_indices', data=sample_indices, compression='gzip')

        # Save train/val/test splits
        split_grp = f.create_group('splits')
        split_grp.attrs['seed'] = args.seed + 1
        split_grp.attrs['n_train'] = n_train
        split_grp.attrs['n_val'] = n_val
        split_grp.attrs['n_test'] = n_test
        split_grp.create_dataset('train_indices', data=train_idx, compression='gzip')
        split_grp.create_dataset('val_indices', data=val_idx, compression='gzip')
        split_grp.create_dataset('test_indices', data=test_idx, compression='gzip')

        # Save summary statistics
        if n_success > 0:
            meta_grp.attrs['mean_error_ev'] = np.mean(errors_ev)
            meta_grp.attrs['mean_error_mev'] = np.mean(errors_mev)
            meta_grp.attrs['mean_force_mae'] = np.mean(force_maes)

        # Save each sample
        print("")  # New line after previous output
        for i, result in tqdm(enumerate(results), total=len(results),
                              desc="Saving to HDF5", unit="sample", ncols=100):
            sample_grp = f.create_group(f'sample_{i}')

            # Save all data from result dictionary
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    sample_grp.create_dataset(key, data=value, compression='gzip')
                elif isinstance(value, (int, float, bool, str)):
                    sample_grp.attrs[key] = value
                elif value is None:
                    sample_grp.attrs[key] = 'None'

    print(f"\n✓ Successfully saved to {args.output}")

    # Print HDF5 structure info
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nFile Info:")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Samples: {len(results)}")
    if n_success > 0:
        avg_size_kb = file_size_mb * 1024 / len(results)
        print(f"  Avg per sample: {avg_size_kb:.2f} KB")

    print(f"\n{'='*80}")
    print("HDF5 Structure:")
    print(f"{'='*80}")
    print("""
final_data.h5
├── metadata/                    # Global settings and statistics
│   ├── Attributes:
│   │   ├── molecule, functional, basis, conv_tol, grid_level
│   │   ├── num_samples, sampling_method, seed, n_processes
│   │   ├── total_time_sec, n_success, n_converged
│   │   └── mean_error_ev, mean_error_mev, mean_force_mae
│   └── Datasets:
│       └── sample_indices [num_samples]  # Original rMD17 indices
│
├── splits/                      # Train/Val/Test splits
│   ├── Attributes: seed, n_train, n_val, n_test
│   └── Datasets:
│       ├── train_indices [n_train]       # Indices into sample_N
│       ├── val_indices [n_val]
│       └── test_indices [n_test]
│
└── sample_N/                    # Per-sample data (N = 0 to num_samples-1)
    ├── Attributes:
    │   ├── idx, success, converged, calc_time
    │   ├── energy_ha, energy_ev, energy_ref_kcal, energy_ref_ev
    │   ├── error_energy_ev, error_energy_mev, error_forces_mae
    │   ├── energy_nuclear, ngrids, nbasis, natoms
    │   └── error_message (if failed)
    │
    └── Datasets (compressed):
        ├── positions [natoms, 3]              # Angstrom
        ├── atomic_numbers [natoms]
        ├── forces_ha_bohr [natoms, 3]         # Hartree/Bohr
        ├── forces_ev_ang [natoms, 3]          # eV/Angstrom
        ├── forces_ref_kcal [natoms, 3]        # kcal/mol/Angstrom
        ├── forces_ref_ev [natoms, 3]          # eV/Angstrom
        ├── overlap [nbasis, nbasis]
        ├── initial_hamiltonian [nbasis, nbasis]
        ├── density_matrix [nbasis, nbasis]
        ├── hamiltonian [nbasis, nbasis]
        ├── orbital_energies [nbasis]
        └── orbital_coefficients [nbasis, nbasis]
""")

    print(f"{'='*80}")
    print("Usage:")
    print(f"{'='*80}")
    print(f"""
# Read the data:
import h5py
import numpy as np

with h5py.File('{args.output}', 'r') as f:
    # Read metadata
    meta = f['metadata']
    print(f"Molecule: {{meta.attrs['molecule']}}")
    print(f"Mean error: {{meta.attrs['mean_error_mev']:.2f}} meV")

    # Get original rMD17 indices
    sample_indices = meta['sample_indices'][:]

    # Get train/val/test splits
    splits = f['splits']
    train_idx = splits['train_indices'][:]  # Indices into sample_N
    val_idx = splits['val_indices'][:]
    test_idx = splits['test_indices'][:]

    print(f"Train: {{len(train_idx)}} samples")

    # Read training sample 0 (first sample in training set)
    first_train_sample_id = train_idx[0]
    sample = f[f'sample_{{first_train_sample_id}}']
    energy = sample.attrs['energy_ev']
    forces = sample['forces_ev_ang'][:]
    overlap = sample['overlap'][:]
    hamiltonian = sample['hamiltonian'][:]
""")

    print(f"{'='*80}")
    print("Complete!")
    print(f"{'='*80}")
