"""
Orbital Conventions for Matrix Transformations

This module defines orbital conventions for different quantum chemistry software
and basis sets. Conventions specify how orbitals are ordered and transformed
between different software packages.

Convention Structure:
- atom_to_orbitals_map: Maps atomic numbers to orbital strings (e.g., 'ssspppd')
- orbital_idx_map: Reordering indices for each orbital type
- orbital_sign_map: Sign changes for each orbital type
- orbital_order_map: Ordering within atomic blocks
- max_block_size: Maximum number of orbitals per atom

Auto-detection:
This module can automatically detect orbital configurations from PySCF basis sets,
allowing support for any atom/basis combination without manual configuration.
"""

from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union

# ==============================================================================
# Standard Orbital Reordering Maps (Common across most conventions)
# ==============================================================================

# These maps define how orbitals are reordered when transforming between conventions
STANDARD_ORBITAL_MAPS = {
    # PySCF to E3NN: p orbitals [px,py,pz] -> [py,pz,px]
    'pyscf_to_e3nn': {
        'idx': {
            's': [0],
            'p': [1, 2, 0],  # [py, pz, px]
            'd': [0, 1, 2, 3, 4],
            'f': [0, 1, 2, 3, 4, 5, 6],
        },
        'sign': {
            's': [1],
            'p': [1, 1, 1],
            'd': [1, 1, 1, 1, 1],
            'f': [1, 1, 1, 1, 1, 1, 1],
        }
    },
    
    # E3NN to PySCF: p orbitals [py,pz,px] -> [px,py,pz]
    'e3nn_to_pyscf': {
        'idx': {
            's': [0],
            'p': [2, 0, 1],  # [px, py, pz]
            'd': [0, 1, 2, 3, 4],
            'f': [0, 1, 2, 3, 4, 5, 6],
        },
        'sign': {
            's': [1],
            'p': [1, 1, 1],
            'd': [1, 1, 1, 1, 1],
            'f': [1, 1, 1, 1, 1, 1, 1],
        }
    },
    
    # PSI4 to PySCF: d orbitals have different ordering
    'psi4_to_pyscf': {
        'idx': {
            's': [0],
            'p': [1, 2, 0],
            'd': [4, 2, 0, 1, 3],  # Special PSI4 d-orbital ordering
            'f': [6, 4, 2, 0, 1, 3, 5]  # Special PSI4 f-orbital ordering
        },
        'sign': {
            's': [1],
            'p': [1, 1, 1],
            'd': [1, 1, 1, 1, 1],
            'f': [1, 1, 1, 1, 1, 1, 1],
        }
    }
}


# ==============================================================================
# Basis Set Definitions
# ==============================================================================

class BasisSetTemplate:
    """Template for defining basis sets with atom configurations."""
    
    def __init__(self, name, description, default_orbitals):
        """
        Args:
            name (str): Basis set name
            description (str): Description of the basis set
            default_orbitals (dict): Default orbital configurations for common atoms
        """
        self.name = name
        self.description = description
        self.default_orbitals = default_orbitals
    
    def get_max_block_size(self):
        """Calculate maximum orbital block size for this basis set."""
        # Calculate based on longest orbital string
        max_size = 0
        for orbitals in self.default_orbitals.values():
            size = sum({
                's': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9
            }.get(orb, 0) for orb in orbitals)
            max_size = max(max_size, size)
        return max_size


# Define basis set templates
BASIS_TEMPLATES = {
    '631G': BasisSetTemplate(
        name='6-31G',
        description='6-31G split-valence basis set',
        default_orbitals={
            1: 'ss',        # H: 2 orbitals
            6: 'ssspp',     # C,N,O,F: 9 orbitals
            7: 'ssspp',
            8: 'ssspp',
            9: 'ssspp',
        }
    ),
    
    'def2svp': BasisSetTemplate(
        name='def2-SVP',
        description='def2-SVP basis set with polarization',
        default_orbitals={
            1: 'ssp',       # H: 5 orbitals
            6: 'sssppd',    # C,N,O,F: 14 orbitals
            7: 'sssppd',
            8: 'sssppd',
            9: 'sssppd',
            16: 'sssspppd',     # S: 18 orbitals
            17: 'sssspppd',     # Cl: 18 orbitals
            35: 'sssssppppddd', # Br: 32 orbitals
        }
    ),
    
    'def2tzvp': BasisSetTemplate(
        name='def2-TZVP',
        description='def2-TZVP triple-zeta basis set',
        default_orbitals={
            1:  'sssp',           # H: 6 orbitals
            6:  'ssssspppddf',    # C,N,O,F: 31 orbitals
            7:  'ssssspppddf',
            8:  'ssssspppddf',
            9:  'ssssspppddf',
            15: 'ssssspppppddf',  # P,S,Cl: 37 orbitals
            16: 'ssssspppppddf',
            17: 'ssssspppppddf',
        }
    ),
    
    'def2tzvppd': BasisSetTemplate(
        name='def2-TZVPPD',
        description='def2-TZVPPD with diffuse functions',
        default_orbitals={
            1:  'ssspppd',        # H: 17 orbitals
            6:  'sssssspppdddf',  # C,N: 37 orbitals
            7:  'sssssspppdddf',
            8:  'ssssssppppdddf', # O,F: 40 orbitals
            9:  'ssssssppppdddf',
        }
    ),
}


# ==============================================================================
# Convention Builder
# ==============================================================================

def build_convention(basis_template, orbital_map_type, custom_atoms=None):
    """Build a convention from a basis template and orbital map.
    
    Args:
        basis_template (BasisSetTemplate): Basis set template to use
        orbital_map_type (str): Type of orbital mapping ('pyscf_to_e3nn', 'e3nn_to_pyscf', etc.)
        custom_atoms (dict, optional): Additional atom configurations to add
    
    Returns:
        Namespace: Complete convention configuration
    """
    orbital_maps = STANDARD_ORBITAL_MAPS[orbital_map_type]
    
    # Start with default atoms
    atom_to_orbitals_map = basis_template.default_orbitals.copy()
    
    # Add custom atoms if provided
    if custom_atoms:
        atom_to_orbitals_map.update(custom_atoms)
    
    # Generate orbital order map (default: sequential)
    orbital_order_map = {}
    for atom_num, orbitals in atom_to_orbitals_map.items():
        orbital_order_map[atom_num] = list(range(len(orbitals)))
    
    return Namespace(
        atom_to_orbitals_map=atom_to_orbitals_map,
        orbital_idx_map=orbital_maps['idx'],
        orbital_sign_map=orbital_maps['sign'],
        orbital_order_map=orbital_order_map,
        max_block_size=basis_template.get_max_block_size()
    )


# ==============================================================================
# Pre-defined Conventions
# ==============================================================================

def get_all_conventions():
    """Get all pre-defined orbital conventions.
    
    Returns:
        dict: Dictionary mapping convention names to Namespace configurations
    """
    conventions = {}
    
    # PSI4 conventions
    conventions['psi4_def2-tzvppd_to_pyscf'] = build_convention(
        BASIS_TEMPLATES['def2tzvppd'],
        'psi4_to_pyscf'
    )
    
    # PySCF to E3NN conventions
    conventions['pyscf_def2-tzvppd_to_e3nn'] = build_convention(
        BASIS_TEMPLATES['def2tzvppd'],
        'pyscf_to_e3nn'
    )
    
    conventions['pyscf_def2-tzvp_to_e3nn'] = build_convention(
        BASIS_TEMPLATES['def2tzvp'],
        'pyscf_to_e3nn'
    )
    
    conventions['pyscf_631G_to_e3nn'] = build_convention(
        BASIS_TEMPLATES['631G'],
        'pyscf_to_e3nn'
    )
    
    conventions['pyscf_def2svp_to_e3nn'] = build_convention(
        BASIS_TEMPLATES['def2svp'],
        'pyscf_to_e3nn'
    )
    
    # E3NN to PySCF conventions (reverse)
    conventions['e3nn_def2-tzvppd_to_pyscf'] = build_convention(
        BASIS_TEMPLATES['def2tzvppd'],
        'e3nn_to_pyscf'
    )
    
    conventions['e3nn_to_pyscf_def2-tzvp'] = build_convention(
        BASIS_TEMPLATES['def2tzvp'],
        'e3nn_to_pyscf'
    )
    
    conventions['e3nn_to_pyscf_def2svp'] = build_convention(
        BASIS_TEMPLATES['def2svp'],
        'e3nn_to_pyscf'
    )
    
    # Aliases for convenience
    conventions['back2pyscf'] = conventions['e3nn_to_pyscf_def2svp']
    conventions['pyscf_def2svp'] = conventions['pyscf_def2svp_to_e3nn']
    conventions['pyscf_631G'] = conventions['pyscf_631G_to_e3nn']
    conventions['pyscf_def2-tzvp'] = conventions['pyscf_def2-tzvp_to_e3nn']
    
    # For MD17 dataset (ORCA convention, similar to e3nn_to_pyscf_def2svp)
    conventions['orca_to_e3nn'] = conventions['e3nn_to_pyscf_def2svp']

    # nablaDFT conventions (def2-SVP with S, Cl, Br)
    conventions['psi4_def2svp_to_pyscf'] = build_convention(
        BASIS_TEMPLATES['def2svp'],
        'psi4_to_pyscf'
    )
    conventions['pyscf_def2svp_nabla_to_e3nn'] = build_convention(
        BASIS_TEMPLATES['def2svp'],
        'pyscf_to_e3nn'
    )
    conventions['e3nn_to_pyscf_def2svp_nabla'] = build_convention(
        BASIS_TEMPLATES['def2svp'],
        'e3nn_to_pyscf'
    )

    return conventions


# ==============================================================================
# Custom Convention Creation
# ==============================================================================

def create_custom_convention(basis_set_name, orbital_map_type, atom_configs):
    """Create a custom convention with specific atom configurations.
    
    This is useful for defining conventions for molecules with unusual atoms
    or custom basis sets.
    
    Args:
        basis_set_name (str): Name of the basis set template to use
        orbital_map_type (str): Type of orbital mapping
        atom_configs (dict): Dictionary mapping atomic numbers to orbital strings
    
    Returns:
        Namespace: Custom convention configuration
    
    Example:
        >>> # Create convention for molecule with Bromine
        >>> conv = create_custom_convention(
        ...     'def2svp',
        ...     'pyscf_to_e3nn',
        ...     {35: 'ssssppppppd'}  # Br configuration
        ... )
    """
    if basis_set_name not in BASIS_TEMPLATES:
        raise ValueError(f"Unknown basis set: {basis_set_name}. "
                        f"Available: {list(BASIS_TEMPLATES.keys())}")
    
    if orbital_map_type not in STANDARD_ORBITAL_MAPS:
        raise ValueError(f"Unknown orbital map: {orbital_map_type}. "
                        f"Available: {list(STANDARD_ORBITAL_MAPS.keys())}")
    
    template = BASIS_TEMPLATES[basis_set_name]
    return build_convention(template, orbital_map_type, atom_configs)


def add_basis_set_template(name, description, atom_orbitals):
    """Add a new basis set template to the available templates.
    
    Args:
        name (str): Name of the basis set
        description (str): Description
        atom_orbitals (dict): Default orbital configurations for atoms
    
    Example:
        >>> add_basis_set_template(
        ...     'cc-pVDZ',
        ...     'Correlation consistent double-zeta',
        ...     {1: 'ssp', 6: 'sssppd'}
        ... )
    """
    BASIS_TEMPLATES[name.lower().replace('-', '')] = BasisSetTemplate(
        name, description, atom_orbitals
    )


def add_orbital_map(name, idx_map, sign_map):
    """Add a new orbital reordering map.
    
    Args:
        name (str): Name of the orbital map
        idx_map (dict): Index reordering map
        sign_map (dict): Sign map
    
    Example:
        >>> add_orbital_map(
        ...     'custom_map',
        ...     {'s': [0], 'p': [0, 1, 2], 'd': [0, 1, 2, 3, 4]},
        ...     {'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1]}
        ... )
    """
    STANDARD_ORBITAL_MAPS[name] = {
        'idx': idx_map,
        'sign': sign_map
    }


# ==============================================================================
# PySCF Auto-detection
# ==============================================================================

# Mapping from atomic number to element symbol (1-103)
ATOMIC_SYMBOLS = {
    # Period 1
    1: 'H', 2: 'He',
    # Period 2
    3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    # Period 3
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
    # Period 4
    19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
    26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge',
    33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
    # Period 5
    37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
    44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe',
    # Period 6
    55: 'Cs', 56: 'Ba',
    # Lanthanides
    57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu',
    64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
    # Period 6 continued
    72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
    79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn',
    # Period 7
    87: 'Fr', 88: 'Ra',
    # Actinides
    89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am',
    96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr',
}

# Angular momentum to orbital type
ANGULAR_MOMENTUM_MAP = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h'}


def detect_orbitals_from_pyscf(basis_name: str, atomic_numbers: List[int]) -> Dict[int, str]:
    """Auto-detect orbital configurations from PySCF basis sets.
    
    This function queries PySCF to determine the orbital configuration for each
    atom in the given basis set, eliminating the need for manual configuration.
    
    Args:
        basis_name (str): Name of the basis set (e.g., 'def2-svp', '6-31g', 'cc-pvdz')
        atomic_numbers (List[int]): List of atomic numbers to detect
    
    Returns:
        Dict[int, str]: Mapping from atomic number to orbital string (e.g., 'ssspppd')
        
    Example:
        >>> orbitals = detect_orbitals_from_pyscf('def2-svp', [1, 6, 7, 8])
        >>> print(orbitals)
        {1: 'ssp', 6: 'sssppd', 7: 'sssppd', 8: 'sssppd'}
        
    Note:
        Requires PySCF to be installed. If PySCF is not available, returns empty dict
        and prints a warning.
    """
    try:
        from pyscf import gto
    except ImportError:
        print("Warning: PySCF not installed. Cannot auto-detect orbital configurations.")
        print("Install with: pip install pyscf")
        return {}
    
    orbital_configs = {}
    
    for atomic_num in atomic_numbers:
        if atomic_num not in ATOMIC_SYMBOLS:
            print(f"Warning: Unknown atomic number {atomic_num}, skipping")
            continue
        
        symbol = ATOMIC_SYMBOLS[atomic_num]
        
        try:
            # Load basis set for this atom
            basis_data = gto.basis.load(basis_name, symbol)
            
            # Count orbitals by angular momentum
            angular_momentum_counts = {}
            
            for shell in basis_data:
                l = shell[0]  # Angular momentum quantum number
                # Each shell can have multiple contractions
                # We count each contraction separately
                angular_momentum_counts[l] = angular_momentum_counts.get(l, 0) + 1
            
            # Build orbital string
            orbital_string = ''
            for l in sorted(angular_momentum_counts.keys()):
                if l in ANGULAR_MOMENTUM_MAP:
                    orbital_type = ANGULAR_MOMENTUM_MAP[l]
                    count = angular_momentum_counts[l]
                    orbital_string += orbital_type * count
            
            orbital_configs[atomic_num] = orbital_string
            
        except Exception as e:
            print(f"Warning: Could not load basis '{basis_name}' for {symbol} (Z={atomic_num}): {e}")
            continue
    
    return orbital_configs


def create_convention_from_pyscf(
    basis_name: str,
    atomic_numbers: List[int],
    orbital_map_type: str = 'pyscf_to_e3nn'
) -> Namespace:
    """Create a convention by auto-detecting orbitals from PySCF.
    
    This is the main convenience function for creating conventions for arbitrary
    molecules without manual configuration.
    
    Args:
        basis_name (str): PySCF basis set name (e.g., 'def2-svp', 'cc-pvdz')
        atomic_numbers (List[int]): List of atomic numbers in your molecule
        orbital_map_type (str): Type of orbital reordering to apply
    
    Returns:
        Namespace: Complete convention ready to use with matrix_transform_single
        
    Example:
        >>> # Create convention for a molecule with C, H, Si, Br
        >>> conv = create_convention_from_pyscf(
        ...     'def2-svp',
        ...     [1, 6, 14, 35]
        ... )
        >>> # Use it immediately
        >>> transformed = matrix_transform_single(hamiltonian, atoms, convention=conv)
        
    Note:
        This function automatically handles any atom that PySCF supports for the
        given basis set. No manual configuration needed!
    """
    # Auto-detect orbital configurations
    print(f"Auto-detecting orbitals for basis '{basis_name}'...")
    orbital_configs = detect_orbitals_from_pyscf(basis_name, atomic_numbers)
    
    if not orbital_configs:
        raise ValueError(
            f"Could not detect orbitals for any atoms. "
            f"Make sure PySCF is installed and basis '{basis_name}' is valid."
        )
    
    print(f"Detected configurations for {len(orbital_configs)} atoms:")
    for atom_num in sorted(orbital_configs.keys()):
        symbol = ATOMIC_SYMBOLS.get(atom_num, f"Z={atom_num}")
        orbitals = orbital_configs[atom_num]
        n_orb = sum({'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9}.get(o, 0) for o in orbitals)
        print(f"  {symbol:3s} (Z={atom_num:3d}): '{orbitals}' ({n_orb} orbitals)")
    
    # Get orbital maps
    if orbital_map_type not in STANDARD_ORBITAL_MAPS:
        raise ValueError(
            f"Unknown orbital map type: {orbital_map_type}. "
            f"Available: {list(STANDARD_ORBITAL_MAPS.keys())}"
        )
    
    orbital_maps = STANDARD_ORBITAL_MAPS[orbital_map_type]
    
    # Generate orbital order map (default: sequential)
    orbital_order_map = {}
    for atom_num, orbitals in orbital_configs.items():
        orbital_order_map[atom_num] = list(range(len(orbitals)))
    
    # Calculate max block size
    max_block_size = 0
    for orbitals in orbital_configs.values():
        size = sum({'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9}.get(o, 0) for o in orbitals)
        max_block_size = max(max_block_size, size)
    
    return Namespace(
        atom_to_orbitals_map=orbital_configs,
        orbital_idx_map=orbital_maps['idx'],
        orbital_sign_map=orbital_maps['sign'],
        orbital_order_map=orbital_order_map,
        max_block_size=max_block_size
    )


def extend_convention_with_pyscf(
    base_convention_name: str,
    basis_name: str,
    additional_atoms: List[int]
) -> Namespace:
    """Extend an existing convention with atoms auto-detected from PySCF.
    
    This is useful when you have a pre-defined convention but need to add support
    for a few more atoms.
    
    Args:
        base_convention_name (str): Name of existing convention (e.g., 'pyscf_def2svp')
        basis_name (str): PySCF basis set name (should match the base convention)
        additional_atoms (List[int]): Atomic numbers to add
    
    Returns:
        Namespace: Extended convention with original + new atoms
        
    Example:
        >>> # Extend def2-svp to support Silicon and Bromine
        >>> conv = extend_convention_with_pyscf(
        ...     'pyscf_def2svp',
        ...     'def2-svp',
        ...     [14, 35]  # Si, Br
        ... )
    """
    # Get base convention
    all_convs = get_all_conventions()
    if base_convention_name not in all_convs:
        raise ValueError(f"Unknown convention: {base_convention_name}")
    
    base_conv = all_convs[base_convention_name]
    
    # Auto-detect new atoms
    print(f"Auto-detecting orbitals for additional atoms...")
    new_orbitals = detect_orbitals_from_pyscf(basis_name, additional_atoms)
    
    if not new_orbitals:
        raise ValueError("Could not detect orbitals for any additional atoms")
    
    # Merge with base convention
    import copy
    extended_conv = Namespace(
        atom_to_orbitals_map=copy.deepcopy(base_conv.atom_to_orbitals_map),
        orbital_idx_map=copy.deepcopy(base_conv.orbital_idx_map),
        orbital_sign_map=copy.deepcopy(base_conv.orbital_sign_map),
        orbital_order_map=copy.deepcopy(base_conv.orbital_order_map),
        max_block_size=base_conv.max_block_size
    )
    
    # Add new atoms
    for atom_num, orbitals in new_orbitals.items():
        extended_conv.atom_to_orbitals_map[atom_num] = orbitals
        extended_conv.orbital_order_map[atom_num] = list(range(len(orbitals)))
        
        # Update max block size if needed
        size = sum({'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9}.get(o, 0) for o in orbitals)
        if size > extended_conv.max_block_size:
            extended_conv.max_block_size = size
    
    print(f"Extended convention now supports: {sorted(extended_conv.atom_to_orbitals_map.keys())}")
    
    return extended_conv


# ==============================================================================
# Utility Functions
# ==============================================================================

def print_convention_info(convention_name=None):
    """Print information about conventions.
    
    Args:
        convention_name (str, optional): Specific convention to print. 
            If None, prints all available conventions.
    """
    conventions = get_all_conventions()
    
    if convention_name:
        if convention_name not in conventions:
            print(f"Convention '{convention_name}' not found!")
            print(f"Available: {list(conventions.keys())}")
            return
        
        conv = conventions[convention_name]
        print(f"\n{'='*70}")
        print(f"Convention: {convention_name}")
        print(f"{'='*70}")
        print(f"Supported atoms: {sorted(conv.atom_to_orbitals_map.keys())}")
        print(f"Max block size: {conv.max_block_size}")
        print(f"\nAtom configurations:")
        for atom, orbitals in sorted(conv.atom_to_orbitals_map.items()):
            n_orb = sum({'s': 1, 'p': 3, 'd': 5, 'f': 7}.get(o, 0) for o in orbitals)
            print(f"  Z={atom:3d}: '{orbitals:15s}' ({n_orb:2d} orbitals)")
    else:
        print(f"\n{'='*70}")
        print("Available Conventions")
        print(f"{'='*70}")
        for name, conv in sorted(conventions.items()):
            atoms = sorted(conv.atom_to_orbitals_map.keys())
            print(f"{name:30s}: atoms={atoms}, max_block={conv.max_block_size}")


def list_basis_sets():
    """List all available basis set templates."""
    print(f"\n{'='*70}")
    print("Available Basis Set Templates")
    print(f"{'='*70}")
    for name, template in BASIS_TEMPLATES.items():
        print(f"\n{template.name} ({name}):")
        print(f"  {template.description}")
        print(f"  Supported atoms: {sorted(template.default_orbitals.keys())}")
        print(f"  Max block size: {template.get_max_block_size()}")


# ==============================================================================
# Export
# ==============================================================================

__all__ = [
    # Core functions
    'get_all_conventions',
    'create_custom_convention',
    'add_basis_set_template',
    'add_orbital_map',
    'print_convention_info',
    'list_basis_sets',
    
    # PySCF auto-detection
    'detect_orbitals_from_pyscf',
    'create_convention_from_pyscf',
    'extend_convention_with_pyscf',
    
    # Constants
    'BASIS_TEMPLATES',
    'STANDARD_ORBITAL_MAPS',
    'ATOMIC_SYMBOLS',
]

