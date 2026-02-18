# Physical constants
# ================
# from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal, Angstrom, eV
# """Constants from ase.units"""
# Length unit (Angstrom in pyscf)
# ANG2BOHR = Angstrom / Bohr
# BOHR2ANG = 1.0 / ANG2BOHR

# Energy unit (Eh in pyscf)
# HA2eV = Hartree / eV
# eV2HA = eV / Hartree
# HA2meV = 1000 * HA2eV
# meV2HA = eV2HA / 1000

# KCALPM     = kcal / mol
# KCALPM2eV  = KCALPM / eV
# KCALPM2meV = KCALPM2eV * 1000

# HA2KCALPM  = Hartree / KCALPM     # Hartree to kcal/mol
# KCALPM2HA  = KCALPM / Hartree       # kcal/mol to Hartree

# ================
from ase.units import Bohr, Rydberg, kJ, kB, fs, Hartree, mol, kcal
# from ase.units import Angstrom, eV

# Default units
Angstrom = 1.0
eV = 1.0

# Length unit (Angstrom in pyscf)
ANG2BOHR = 1.8897261258369282     # Angstrom to Bohr conversion
BOHR2ANG = 0.5291772105638411     # Bohr to Angstrom conversion

# Energy unit (Eh in pyscf)
HA2eV    = 27.211396641308        # Hartree to eV conversion
HA2meV   = HA2eV * 1000           # Hartree to meV conversion
eV2HA    = 0.03674932247495664    # eV to Hartree
meV2HA   = eV2HA / 1000           # meV to Hartree

KCALPM2eV  = 0.04336410390059322   # kcal/mol to eV conversion
KCALPM2meV = KCALPM2eV * 1000      # kcal/mol to meV conversion
HA2KCALPM  = 627.5094738898777     # Hartree to kcal/mol
KCALPM2HA  = 0.001593601438080425  # kcal/mol to Hartree

# Force unit (Eh/Bohr in pyscf)
HA_BOHR_2_KCALPM_ANG = HA2KCALPM / BOHR2ANG        # Hartree/Bohr to kcal/mol/Angstrom
KCALPM_ANG_2_HA_BOHR = 1.0 / HA_BOHR_2_KCALPM_ANG  # kcal/mol/Angstrom to Hartree/Bohr
HA_BOHR_2_meV_ANG    = HA2meV / BOHR2ANG           # Hartree/Bohr to meV/Angstrom
HA_BOHR_2_eV_ANG    = HA2eV / BOHR2ANG           # Hartree/Bohr to eV/Angstrom

meV_ANG_2_HA_BOHR    = 1.0 / HA_BOHR_2_meV_ANG     # meV/Angstrom to Hartree/Bohr
eV_ANG_2_HA_BOHR    = 1.0 / HA_BOHR_2_eV_ANG       # eV/Angstrom to Hartree/Bohr
HA_BOHR_2_HA_ANG     = 1.0 / BOHR2ANG              # Hartree/Bohr to Hartree/Angstrom
HA_ANG_2_HA_BOHR     = 1.0 / ANG2BOHR              # Hartree/Angstrom to Hartree/Bohr


def print_unit_conversion():
    print(f"Length unit: {Angstrom} Angstrom")
    print(f"Bohr      : {1*BOHR2ANG} Angstrom")
    
    print(f"Energy unit: {eV} eV")
    print(f"Hartree   : {1*HA2eV} eV")
    print(f"meV       : {1000*eV} meV")
    print(f"kcal/mol  : {1*KCALPM2meV} meV")
    print(f"Hartree   : {1*HA2meV} meV")


# Unit conversion helper functions
# ================================

def normalize_length_unit(unit):
    """Normalize length unit string to standard form.
    
    Args:
        unit: Unit string (case-insensitive)
        
    Returns:
        Normalized unit string: "ang" or "bohr"
        
    Raises:
        ValueError: If unit is not recognized
    """
    unit_lower = unit.lower()
    if unit_lower in ["ang", "angstrom", "a"]:
        return "ang"
    elif unit_lower in ["bohr", "b"]:
        return "bohr"
    else:
        raise ValueError(f"Invalid length unit: {unit}. Must be one of: ang, angstrom, a, bohr, b")


def convert_length(value, from_unit, to_unit):
    """Convert length values between Angstrom and Bohr.
    
    Args:
        value: Length value(s) to convert (scalar or array-like)
        from_unit: Source unit ("ang", "angstrom", "a", "bohr", "b")
        to_unit: Target unit ("ang", "angstrom", "a", "bohr", "b")
        
    Returns:
        Converted length value(s)
        
    Raises:
        ValueError: If units are not recognized
    """
    from_unit = normalize_length_unit(from_unit)
    to_unit = normalize_length_unit(to_unit)
    
    if from_unit == to_unit:
        return value
    
    if from_unit == "ang" and to_unit == "bohr":
        return value * ANG2BOHR
    elif from_unit == "bohr" and to_unit == "ang":
        return value * BOHR2ANG
    else:
        raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")


def get_energy_unit_conversion_factor(unit):
    """Get conversion factor from Hartree to specified energy unit.
    
    Args:
        unit: Energy unit string ("ha", "hartree", "ev", "eV", "mev", "meV")
        
    Returns:
        Conversion factor (multiply Hartree by this to get target unit)
        
    Raises:
        ValueError: If unit is not recognized
    """
    unit_lower = unit.lower()
    if unit_lower in ["ha", "hartree"]:
        return 1.0
    elif unit_lower == "ev":
        return HA2eV
    elif unit_lower == "mev":
        return HA2meV
    else:
        raise ValueError(f"Invalid energy unit: {unit}. Must be one of: ha, hartree, ev, mev")


def get_force_unit_conversion_factor(unit):
    """Get conversion factor from Hartree/Bohr to specified force unit.
    
    Args:
        unit: Force unit string ("ha/bohr", "ev/ang", "eV/Ang", "mev/ang", "meV/Ang")
        
    Returns:
        Conversion factor (multiply Hartree/Bohr by this to get target unit)
        
    Raises:
        ValueError: If unit is not recognized
    """
    unit_lower = unit.lower()
    if unit_lower in ["ha/bohr", "hartree/bohr"]:
        return 1.0
    elif unit_lower in ["ev/ang", "ev/angstrom"]:
        return HA_BOHR_2_eV_ANG
    elif unit_lower in ["mev/ang", "mev/angstrom"]:
        return HA_BOHR_2_meV_ANG
    else:
        raise ValueError(f"Invalid force unit: {unit}. Must be one of: ha/bohr, ev/ang, mev/ang")
