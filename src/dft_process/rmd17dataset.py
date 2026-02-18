#!/usr/bin/env python3
"""
RMD17 NPZ Data Loader

This module provides a class for loading and working with the revised MD17 dataset
stored in NPZ format. The dataset contains molecular dynamics trajectories with
energies, forces, and coordinates calculated at the PBE/def2-SVP level of theory.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class RMD17NPZLoader:
    """
    A class for loading and managing the revised MD17 dataset stored in NPZ format.
    
    This class provides easy access to molecular dynamics data including:
    - Nuclear charges (atomic numbers)
    - Atomic coordinates for each conformation
    - Energies for each conformation
    - Forces for each conformation
    - Original MD17 dataset indices and values
    
    The data is from the revised MD17 dataset by Christensen and von Lilienfeld (2020),
    which provides high-quality DFT calculations with tight SCF convergence.
    """
    
    # Available molecules in the dataset
    AVAILABLE_MOLECULES = [
        "benzene", "ethanol", "malonaldehyde", "uracil", "toluene",
        "salicylic", "azobenzene", "naphthalene", "paracetamol", "aspirin"
    ]
    
    # Physical constants and unit conversions
    ANG2BOHR = 1.8897261258369282      # Angstrom to Bohr conversion
    BOHR2ANG = 1.0 / ANG2BOHR         # Bohr to Angstrom conversion
    KCALPM2eV = 0.043364                 # kcal/mol to eV conversion
    eV2KCALPM = 1.0 / KCALPM2eV           # eV to kcal/mol conversion
    KCAL2meV = 43.36                   # kcal/mol to meV conversion
    meV2KCAL = 1.0 / KCAL2meV         # meV to kcal/mol conversion
    
    def __init__(self, data_dir: str = "/root/25DFT/data/rmd17/npz_data"):
        """
        Initialize the RMD17 NPZ data loader.
        
        Args:
            data_dir (str): Directory containing the NPZ files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Check available files
        self._available_files = self._discover_available_files()
        print(f"Found {len(self._available_files)} NPZ files in {data_dir}")
        
        # Data storage
        self._loaded_data: Dict[str, Dict] = {}
        self._current_molecule: Optional[str] = None
        
    def _discover_available_files(self) -> Dict[str, Path]:
        """Discover available NPZ files in the data directory."""
        available = {}
        for npz_file in self.data_dir.glob("rmd17_*.npz"):
            molecule_name = npz_file.stem.replace("rmd17_", "")
            if molecule_name in self.AVAILABLE_MOLECULES:
                available[molecule_name] = npz_file
        return available
    
    @property
    def available_molecules(self) -> List[str]:
        """Get list of available molecule names."""
        return list(self._available_files.keys())
    
    def get_molecule_info(self, molecule_name: str) -> Dict:
        """
        Get information about a specific molecule without loading the full data.
        
        Args:
            molecule_name (str): Name of the molecule
            
        Returns:
            Dict: Dictionary containing molecule information
            
        Raises:
            ValueError: If molecule is not available
        """
        if molecule_name not in self._available_files:
            raise ValueError(f"Molecule '{molecule_name}' not available. "
                           f"Available: {self.available_molecules}")
        
        npz_path = self._available_files[molecule_name]
        
        # Load just the metadata to get info
        with np.load(npz_path) as data:
            info = {
                'molecule_name': molecule_name,
                'file_path': str(npz_path),
                'file_size_mb': npz_path.stat().st_size / (1024 * 1024),
                'num_conformations': len(data['coords']),
                'num_atoms': len(data['nuclear_charges']),
                'atomic_numbers': data['nuclear_charges'].tolist(),
                'available_keys': list(data.keys())
            }
        
        return info
    
    def load_molecule(self, molecule_name: str, 
                     load_old_data: bool = True,
                     convert_to_torch: bool = False) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Load a specific molecule's data from NPZ file.
        
        Args:
            molecule_name (str): Name of the molecule to load
            load_old_data (bool): Whether to load original MD17 data (default: True)
            convert_to_torch (bool): Whether to convert numpy arrays to torch tensors (default: False)
            
        Returns:
            Dict: Dictionary containing the molecule's data
            
        Raises:
            ValueError: If molecule is not available
        """
        if molecule_name not in self._available_files:
            raise ValueError(f"Molecule '{molecule_name}' not available. "
                           f"Available: {self.available_molecules}")
        
        npz_path = self._available_files[molecule_name]
        
        print(f"Loading {molecule_name} from {npz_path}")
        
        # Load the NPZ file
        with np.load(npz_path) as data:
            # Extract core data
            molecule_data = {
                'molecule_name': molecule_name,
                'nuclear_charges': data['nuclear_charges'],
                'coords': data['coords'],
                'energies': data['energies'],
                'forces': data['forces']
            }
            
            # Load original MD17 data if requested
            if load_old_data:
                if 'old_indices' in data:
                    molecule_data['old_indices'] = data['old_indices']
                if 'old_energies' in data:
                    molecule_data['old_energies'] = data['old_energies']
                if 'old_forces' in data:
                    molecule_data['old_forces'] = data['old_forces']
            
            # Convert to torch tensors if requested
            if convert_to_torch:
                for key, value in molecule_data.items():
                    if key != 'molecule_name' and isinstance(value, np.ndarray):
                        molecule_data[key] = torch.from_numpy(value)
        
        # Store in memory and set as current
        self._loaded_data[molecule_name] = molecule_data
        self._current_molecule = molecule_name
        
        print(f"Loaded {molecule_name}: {len(molecule_data['coords'])} conformations, "
              f"{len(molecule_data['nuclear_charges'])} atoms")
        
        return molecule_data
    
    def get_current_molecule_data(self) -> Optional[Dict]:
        """Get data for the currently loaded molecule."""
        if self._current_molecule is None:
            return None
        return self._loaded_data.get(self._current_molecule)
    
    def get_conformation(self, molecule_name: str, conf_idx: int) -> Dict:
        """
        Get a specific conformation from a molecule.
        
        Args:
            molecule_name (str): Name of the molecule
            conf_idx (int): Index of the conformation
            
        Returns:
            Dict: Dictionary containing conformation data
        """
        if molecule_name not in self._loaded_data:
            self.load_molecule(molecule_name)
        
        data = self._loaded_data[molecule_name]
        
        if conf_idx >= len(data['coords']):
            raise IndexError(f"Conformation index {conf_idx} out of range "
                           f"(max: {len(data['coords']) - 1})")
        
        conformation = {
            'molecule_name': molecule_name,
            'conformation_idx': conf_idx,
            'nuclear_charges': data['nuclear_charges'],
            'coords': data['coords'][conf_idx],
            'energy': data['energies'][conf_idx],
            'forces': data['forces'][conf_idx]
        }
        
        # Add original MD17 data if available
        if 'old_indices' in data:
            conformation['old_index'] = data['old_indices'][conf_idx]
        if 'old_energies' in data:
            conformation['old_energy'] = data['old_energies'][conf_idx]
        if 'old_forces' in data:
            conformation['old_forces'] = data['old_forces'][conf_idx]
        
        return conformation
    
    def get_conformation_range(self, molecule_name: str, 
                              start_idx: int, end_idx: int) -> Dict:
        """
        Get a range of conformations from a molecule.
        
        Args:
            molecule_name (str): Name of the molecule
            start_idx (int): Starting conformation index (inclusive)
            end_idx (int): Ending conformation index (exclusive)
            
        Returns:
            Dict: Dictionary containing range of conformations
        """
        if molecule_name not in self._loaded_data:
            self.load_molecule(molecule_name)
        
        data = self._loaded_data[molecule_name]
        
        if start_idx < 0 or end_idx > len(data['coords']):
            raise IndexError(f"Index range [{start_idx}, {end_idx}) out of bounds "
                           f"[0, {len(data['coords'])})")
        
        range_data = {
            'molecule_name': molecule_name,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'nuclear_charges': data['nuclear_charges'],
            'coords': data['coords'][start_idx:end_idx],
            'energies': data['energies'][start_idx:end_idx],
            'forces': data['forces'][start_idx:end_idx]
        }
        
        # Add original MD17 data if available
        if 'old_indices' in data:
            range_data['old_indices'] = data['old_indices'][start_idx:end_idx]
        if 'old_energies' in data:
            range_data['old_energies'] = data['old_energies'][start_idx:end_idx]
        if 'old_forces' in data:
            range_data['old_forces'] = data['old_forces'][start_idx:end_idx]
        
        return range_data
    
    def get_statistics(self, molecule_name: str) -> Dict:
        """
        Get statistical information about a molecule's data.
        
        Args:
            molecule_name (str): Name of the molecule
            
        Returns:
            Dict: Dictionary containing statistical information
        """
        if molecule_name not in self._loaded_data:
            self.load_molecule(molecule_name)
        
        data = self._loaded_data[molecule_name]
        
        stats = {
            'molecule_name': molecule_name,
            'num_conformations': len(data['coords']),
            'num_atoms': len(data['nuclear_charges']),
            'energy_stats': {
                'mean': float(np.mean(data['energies'])),
                'std': float(np.std(data['energies'])),
                'min': float(np.min(data['energies'])),
                'max': float(np.max(data['energies'])),
                'range': float(np.max(data['energies']) - np.min(data['energies']))
            },
            'force_stats': {
                'mean_magnitude': float(np.mean(np.linalg.norm(data['forces'], axis=2))),
                'max_magnitude': float(np.max(np.linalg.norm(data['forces'], axis=2))),
                'std_magnitude': float(np.std(np.linalg.norm(data['forces'], axis=2)))
            },
            'coordinate_stats': {
                'mean_atomic_distance': float(self._calculate_mean_atomic_distance(data['coords']))
            }
        }
        
        return stats
    
    def _calculate_mean_atomic_distance(self, coords: np.ndarray) -> float:
        """Calculate mean atomic distance across all conformations."""
        if len(coords.shape) != 3:  # (n_confs, n_atoms, 3)
            raise ValueError("Coordinates must be 3D array")
        
        total_distance = 0.0
        count = 0
        
        for conf_coords in coords:
            for i in range(len(conf_coords)):
                for j in range(i + 1, len(conf_coords)):
                    dist = np.linalg.norm(conf_coords[i] - conf_coords[j])
                    total_distance += dist
                    count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def convert_units(self, data: Dict, 
                     energy_unit: str = 'kcal/mol',
                     length_unit: str = 'angstrom') -> Dict:
        """
        Convert units in the data dictionary.
        
        Args:
            data (Dict): Data dictionary to convert
            energy_unit (str): Target energy unit ('kcal/mol', 'eV', 'hartree')
            length_unit (str): Target length unit ('angstrom', 'bohr')
            
        Returns:
            Dict: Data dictionary with converted units
        """
        converted_data = data.copy()
        
        # Energy conversions
        if energy_unit == 'kcal/mol' and 'energies' in data:
            pass  # Already in kcal/mol
        elif energy_unit == 'eV' and 'energies' in data:
            converted_data['energies'] = data['energies'] * self.KCAL2EV
        elif energy_unit == 'hartree' and 'energies' in data:
            converted_data['energies'] = data['energies'] * self.KCAL2EV / 27.211396641308
        
        # Length conversions
        if length_unit == 'angstrom' and 'coords' in data:
            pass  # Already in angstrom
        elif length_unit == 'bohr' and 'coords' in data:
            converted_data['coords'] = data['coords'] * self.ANG2BOHR
        
        # Force conversions (energy/length)
        if 'forces' in data:
            if energy_unit == 'kcal/mol' and length_unit == 'angstrom':
                pass  # Already in kcal/mol/angstrom
            elif energy_unit == 'eV' and length_unit == 'bohr':
                converted_data['forces'] = (data['forces'] * self.KCAL2EV / self.ANG2BOHR)
        
        return converted_data
    
    def create_training_split(self, molecule_name: str, 
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.1,
                             test_ratio: float = 0.1,
                             random_seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Create train/validation/test splits for training.
        
        Args:
            molecule_name (str): Name of the molecule
            train_ratio (float): Training set ratio (default: 0.8)
            val_ratio (float): Validation set ratio (default: 0.1)
            test_ratio (float): Test set ratio (default: 0.1)
            random_seed (int): Random seed for reproducibility (default: 42)
            
        Returns:
            Dict: Dictionary containing split indices
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        if molecule_name not in self._loaded_data:
            self.load_molecule(molecule_name)
        
        data = self._loaded_data[molecule_name]
        n_confs = len(data['coords'])
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Create random permutation
        indices = np.random.permutation(n_confs)
        
        # Calculate split points
        train_end = int(n_confs * train_ratio)
        val_end = train_end + int(n_confs * val_ratio)
        
        # Create splits
        splits = {
            'train_indices': indices[:train_end],
            'val_indices': indices[train_end:val_end],
            'test_indices': indices[val_end:],
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'total_conformations': n_confs
        }
        
        return splits
    
    def get_molecule_summary(self) -> str:
        """Get a summary of all available molecules."""
        summary = "RMD17 Dataset Summary:\n"
        summary += "=" * 50 + "\n"
        
        for molecule_name in sorted(self.available_molecules):
            try:
                info = self.get_molecule_info(molecule_name)
                summary += f"\n{molecule_name.upper()}:\n"
                summary += f"  File: {info['file_path'].split('/')[-1]}\n"
                summary += f"  Size: {info['file_size_mb']:.1f} MB\n"
                summary += f"  Conformations: {info['num_conformations']:,}\n"
                summary += f"  Atoms: {info['num_atoms']}\n"
                summary += f"  Composition: {self._format_composition(info['atomic_numbers'])}\n"
            except Exception as e:
                summary += f"\n{molecule_name.upper()}: Error loading - {e}\n"
        
        return summary
    
    def _format_composition(self, atomic_numbers: List[int]) -> str:
        """Format atomic composition as a string."""
        from collections import Counter
        element_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
        
        counter = Counter(atomic_numbers)
        composition = []
        for atomic_num in sorted(counter.keys()):
            symbol = element_symbols.get(atomic_num, f'Z{atomic_num}')
            count = counter[atomic_num]
            composition.append(f"{symbol}{count}")
        
        return " ".join(composition)
    
    def __repr__(self) -> str:
        """String representation of the loader."""
        return (f"RMD17NPZLoader(data_dir='{self.data_dir}', "
                f"available_molecules={len(self.available_molecules)}, "
                f"loaded={list(self._loaded_data.keys())})")
    
    def __len__(self) -> int:
        """Number of available molecules."""
        return len(self.available_molecules)


    # Example usage and testing
    @staticmethod
    def example_usage():
        """Example usage of the RMD17NPZLoader class."""
        print("RMD17 NPZ Data Loader Example")
        print("=" * 40)
        
        # Initialize the loader
        loader = RMD17NPZLoader()
        
        # Print summary
        print(loader.get_molecule_summary())
        
        # Load a specific molecule
        print("\nLoading benzene molecule...")
        benzene_data = loader.load_molecule("benzene")
        
        # Get statistics
        stats = loader.get_statistics("benzene")
        print(f"\nBenzene statistics:")
        print(f"  Conformations: {stats['num_conformations']:,}")
        print(f"  Energy range: {stats['energy_stats']['min']:.2f} to {stats['energy_stats']['max']:.2f} kcal/mol")
        print(f"  Mean force magnitude: {stats['force_stats']['mean_magnitude']:.2f} kcal/mol/angstrom")
        
        # Get a specific conformation
        conf = loader.get_conformation("benzene", 0)
        print(f"\nFirst conformation:")
        print(f"  Energy: {conf['energy']:.4f} kcal/mol")
        print(f"  Coordinates shape: {conf['coords'].shape}")
        
        # Create training split
        splits = loader.create_training_split("benzene", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        print(f"\nTraining splits:")
        print(f"  Train: {len(splits['train_indices']):,} conformations")
        print(f"  Validation: {len(splits['val_indices']):,} conformations")
        print(f"  Test: {len(splits['test_indices']):,} conformations")


if __name__ == "__main__":
    RMD17NPZLoader.example_usage()