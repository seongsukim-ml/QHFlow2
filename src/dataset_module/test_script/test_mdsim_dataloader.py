#!/usr/bin/env python3
"""
Sample script to test MD17_DFT_MDSim dataset loader.

Usage:
    python test_mdsim_dataloader.py --name mdsim-salicylic_acid
    python test_mdsim_dataloader.py --name mdsim-ethanol --all_features
    python test_mdsim_dataloader.py --name mdsim-aspirin --include_density

Note: The dataset expects a 'data.h5' file in the dataset folder.
      If your file has a different name (e.g., final_md17_xxx.h5), 
      you need to rename or symlink it to 'data.h5'.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

# Set environment variables before importing dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

from dataset_module.new_md_dataloader_mdsim import MD17_DFT_MDSim


def print_sample_info(sample, idx):
    """Print detailed information about a sample."""
    print(f"\n{'='*60}")
    print(f"Sample {idx} Information:")
    print(f"{'='*60}")
    
    print(f"\n[Geometry]")
    print(f"  - Position shape: {sample.pos.shape}")
    print(f"  - Atoms shape: {sample.atoms.shape}")
    print(f"  - Number of atoms: {sample.num_atoms.item()}")
    
    print(f"\n[Energy & Forces]")
    print(f"  - DFT Energy: {sample.dft_energy.item():.6f} eV")
    print(f"  - DFT Forces shape: {sample.dft_forces.shape}")
    print(f"  - Reference Energy: {sample.energy_ref:.6f} eV")
    print(f"  - Reference Forces shape: {sample.forces_ref.shape}")
    
    print(f"\n[Electronic Structure]")
    print(f"  - Hamiltonian shape: {sample.hamiltonian.shape}")
    print(f"  - Overlap shape: {sample.overlap.shape}")
    print(f"  - Initial Hamiltonian shape: {sample.init_ham.shape}")
    print(f"  - H dimension: {sample.h_dim.item()}")
    
    print(f"\n[Atomic Orbital Info]")
    print(f"  - AO_l_index shape: {sample.AO_l_index.shape}")
    print(f"  - Q shape: {sample.Q.shape}")
    print(f"  - Full edge index shape: {sample.full_edge_index.shape}")
    
    # Check for optional attributes
    if hasattr(sample, 'density_matrix') and sample.density_matrix is not None:
        print(f"\n[Density Matrix]")
        print(f"  - Density matrix shape: {sample.density_matrix.shape}")
        print(f"  - H_core shape: {sample.h_core.shape}")
    
    if hasattr(sample, 'orbital_energies') and sample.orbital_energies is not None:
        print(f"\n[Orbital Info]")
        print(f"  - Orbital energies shape: {sample.orbital_energies.shape}")
        print(f"  - Orbital coefficients shape: {sample.orbital_coefficients.shape}")
    
    if hasattr(sample, 'energy_ev') and sample.energy_ev is not None:
        print(f"\n[All Features]")
        print(f"  - Energy (eV): {sample.energy_ev.item():.6f}")
        print(f"  - Converged: {sample.converged.item()}")


def test_dataloader(dataset, batch_size=2):
    """Test the dataset with a DataLoader."""
    print(f"\n{'='*60}")
    print(f"Testing DataLoader with batch_size={batch_size}")
    print(f"{'='*60}")
    
    # Note: AOData might need custom collate function for batching
    try:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0  # Set to 0 for debugging
        )
        
        for i, batch in enumerate(loader):
            print(f"\nBatch {i}:")
            print(f"  - Positions shape: {batch.pos.shape}")
            print(f"  - Atoms shape: {batch.atoms.shape}")
            if i >= 1:  # Only show first 2 batches
                break
        print("DataLoader test passed!")
    except Exception as e:
        print(f"DataLoader test failed (this might be expected for AOData): {e}")
        print("Note: AOData objects may require a custom collate_fn for batching.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test MD17_DFT_MDSim Dataset")
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
        help="Include all features",
    )
    parser.add_argument(
        "--use_ref_hamiltonian",
        action="store_true",
        help="Use reference hamiltonian as initial hamiltonian",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to display",
    )
    parser.add_argument(
        "--test_dataloader",
        action="store_true",
        help="Test DataLoader functionality",
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        help="Enable pdb debugging",
    )

    args = parser.parse_args()

    # Print configuration
    print(f"{'='*60}")
    print("MD17_DFT_MDSim Dataset Test")
    print(f"{'='*60}")
    print(f"Root: {args.root}")
    print(f"Name: {args.name}")
    print(f"Prefix: '{args.prefix}'")
    print(f"Postfix: '{args.postfix}'")
    print(f"Load orbitals: {args.load_orbitals}")
    print(f"Include density: {args.include_density}")
    print(f"All features: {args.all_features}")
    print(f"Use ref hamiltonian: {args.use_ref_hamiltonian}")

    # Check if data file exists
    expected_folder = os.path.join(args.root, args.prefix + args.name + args.postfix)
    expected_file = os.path.join(expected_folder, "data.h5")
    
    if not os.path.exists(expected_file):
        print(f"\n[WARNING] Expected file not found: {expected_file}")
        print(f"Looking for alternative h5 files in {expected_folder}...")
        
        if os.path.exists(expected_folder):
            h5_files = [f for f in os.listdir(expected_folder) if f.endswith('.h5')]
            if h5_files:
                print(f"Found h5 files: {h5_files}")
                print(f"\nCreating symlink automatically...")
                os.symlink(
                    os.path.join(expected_folder, h5_files[0]),
                    expected_file
                )
                print(f"Symlink created: {expected_file} -> {h5_files[0]}")
            else:
                print("No h5 files found in the folder.")
                return
        else:
            print(f"Folder does not exist: {expected_folder}")
            return

    # Create dataset
    print(f"\n{'='*60}")
    print("Loading dataset...")
    print(f"{'='*60}")
    
    try:
        dataset = MD17_DFT_MDSim(
            root=args.root,
            name=args.name,
            prefix=args.prefix,
            postfix=args.postfix,
            load_orbitals=args.load_orbitals,
            include_density=args.include_density,
            all_features=args.all_features,
            use_ref_hamiltonian_as_init_ham=args.use_ref_hamiltonian,
        )
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        if args.pdb:
            import pdb
            pdb.post_mortem()
        return

    print(f"\n[Dataset Info]")
    print(f"  - Dataset length: {len(dataset)}")
    print(f"  - Basis: {dataset.basis}")
    print(f"  - Functional: {dataset.functional}")
    print(f"  - Hamiltonian size: {dataset.hamiltonian_size}")
    print(f"  - Atoms: {dataset.atoms}")
    print(f"  - Atom list: {dataset.atom_list}")

    # Display samples
    num_samples = min(args.num_samples, len(dataset))
    print(f"\nDisplaying {num_samples} samples...")
    
    for i in range(num_samples):
        try:
            sample = dataset[i]
            print_sample_info(sample, i)
        except Exception as e:
            print(f"\nError loading sample {i}: {e}")
            if args.pdb:
                import pdb
                pdb.post_mortem()

    # Test DataLoader if requested
    if args.test_dataloader:
        test_dataloader(dataset)

    # Enter pdb if requested
    if args.pdb:
        print("\nEntering pdb debugger...")
        import pdb
        pdb.set_trace()

    print(f"\n{'='*60}")
    print("Test completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

