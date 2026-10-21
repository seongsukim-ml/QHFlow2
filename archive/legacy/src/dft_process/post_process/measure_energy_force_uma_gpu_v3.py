"""
Measure Energy and Forces for DFT Predictions with High Precision

This tool compares predicted Hamiltonians/forces/energies against ground truth
and reference PySCF calculations with reproducible caching, custom slicing, and
plot/report generation. All calculations use double precision (float64).

QUICK START
-----------
- Full dataset:   python measure_energy_force_uma_v3.py --dataset_name QH9Stable
- Fraction slice: python measure_energy_force_uma_v3.py --start_frac 0.2 --end_frac 0.25 ...
- Custom files:   python measure_energy_force_uma_v3.py --file_indices 0,12,99 ...
  (use commas/semicolons; fractions are ignored whenever --file_indices is set)

OUTPUT ORGANIZATION
-------------------
- Results live inside `<model_path>/measure_*` folders:
    * `measure_full` for start=0.0/end=1.0
    * `measure_<start>_<end>` for fractional slices (e.g., measure_0p10_0p15)
    * `measure_idx_<...>` for explicit index lists (e.g., measure_idx_0_12_99)
- Each folder contains the CSV, summaries, plots, and a `_settings.json`
  capturing the run configuration (fractions, index list, UMA flag, etc.).

PERFORMANCE OPTIMIZATIONS
-------------------------
1. **Smart caching**: loads gt metadata from `gt_calc` when possible
2. **Lazy PySCF init**: mean-field objects created only when needed
3. **Optimized I/O**: `torch.load(..., map_location='cpu', weights_only=False)`
4. **Efficient dtype conversions**: skip copies when already float64
5. **Parallel processing**: multiprocessing pool with `imap_unordered`
6. **Memory management**: `maxtasksperchild` to avoid worker leaks
7. **Vectorized math**: single-pass error computations
8. **Parallel flattening**: `_flat` directories refresh via up to 8 processes

PERFORMANCE SNAPSHOT (per file)
-------------------------------
- Fully cached:      ~0.02-0.05 s
- Partially cached:  ~0.5-2   s
- First-time PySCF:  ~5-10    s (dominated by SCF)

RECOMMENDED RUNTIME SETTINGS
----------------------------
- num_workers:       8-32 (depending on CPU cores)
- chunksize:         2-5 for balanced progress vs overhead
- maxtasksperchild:  500-1000 to recycle workers
- profile mode:      `--profile` for detailed Timer logs per step
"""
import gpu4pyscf
from gpu4pyscf.lib.cupy_helper import (
    tag_array, pack_tril, asarray)
import cupy as cp
import sys
import os
import shutil
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Tuple, List
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(src_path)

from common.custom_logger import get_logger
logger = get_logger(__file__)

from dft_process.dft_process_utils import *
from argparse import ArgumentParser
import multiprocessing as mp
from tqdm import tqdm
import gc
import time
import traceback
import json
import matplotlib.pyplot as plt
import seaborn as sns
from ase.atoms import Atoms
from fairchem.core import pretrained_mlip, FAIRChemCalculator

# Set matplotlib backend and style
plt.switch_backend('Agg')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Global variables for process-local UMA instance and calculation settings
_process_uma = None
_enable_profiling = False
_enable_filtering = False  # Enable filtering for unstable MO energy calculations
_pred_xc = "b3lyp"  # XC for prediction energy/force calculations
_gt_xc = "b3lyp"    # XC for ground truth energy/force calculations
_calc_xc = "pbe"    # XC for reference calculations
_basis = "def2svp"
_pred_tol = 1e-8     # Tolerance for prediction density matrix calculation
_gt_tol = 1e-8       # Tolerance for ground truth density matrix calculation
_pad_eigval = 1      # Padding value for eigenvalue calculation


# ==================== Profiling Utilities ====================
"""
Profiling helps identify performance bottlenecks.

Usage:
------
1. Context Manager (for code blocks):
    with Timer("My operation"):
        # code here
        pass

2. Decorator (for functions):
    @profile_func("Custom name")
    def my_function():
        pass

Enable profiling:
-----------------
python script.py --profile

Typical Bottlenecks (from slowest to fastest):
-----------------------------------------------
1. PySCF SCF calculation: ~3-5s (70-80% of total time)
2. Force calculation: ~1-2s (10-20%)
3. Matrix transformation: ~0.1-0.5s (2-5%)
4. Density matrix calc: ~0.05-0.2s (1-2%)
5. torch.load: ~0.01-0.05s (<1%)
6. Type conversions: ~0.001-0.01s (<1%)
"""

class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name, enabled=None):
        self.name = name
        self.enabled = enabled if enabled is not None else _enable_profiling
        self.start = None
    
    def __enter__(self):
        if self.enabled:
            self.start = time.time()
        return self
    
    def __exit__(self, *args):
        if self.enabled and self.start is not None:
            elapsed = time.time() - self.start
            logger.debug(f"  {self.name}: {elapsed:.3f}s")


def profile_func(name=None):
    """Decorator to profile function execution time"""
    def decorator(func):
        func_name = name or func.__name__
        
        def wrapper(*args, **kwargs):
            if _enable_profiling:
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.debug(f"  {func_name}: {elapsed:.3f}s")
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def set_calculation_settings(pred_xc="b3lyp", gt_xc="b3lyp", calc_xc="pbe", basis="def2svp", pred_tol=1e-8, gt_tol=1e-8, pad_eigval=1, filtering=False):
    """Set global calculation settings for worker processes"""
    global _pred_xc, _gt_xc, _calc_xc, _basis, _pred_tol, _gt_tol, _pad_eigval, _enable_filtering
    _pred_xc = pred_xc
    _gt_xc = gt_xc
    _calc_xc = calc_xc
    _basis = basis
    _pred_tol = pred_tol
    _gt_tol = gt_tol
    _pad_eigval = pad_eigval
    _enable_filtering = filtering


def enable_profiling():
    """Enable profiling mode"""
    global _enable_profiling
    _enable_profiling = True
    
    # Set logger to DEBUG level to show profiling info
    import logging
    logger.setLevel(logging.DEBUG)
    
    # Also set console handler to DEBUG
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)
    
    logger.info("="*80)
    logger.info("PROFILING ENABLED - Detailed timing information will be shown")
    logger.info("="*80)


# ==================== Input Path Utilities ====================

def _extract_numeric_key(filename: str, prefix: str) -> Tuple[int, ...]:
    """Return tuple of numeric indices extracted from filename for stable sorting"""
    base = os.path.splitext(os.path.basename(filename))[0]
    remainder = base
    prefix_token = f"{prefix}_"
    if base.startswith(prefix_token):
        remainder = base[len(prefix_token):]
    elif "_" in base:
        remainder = base.split("_", 1)[1]
    parts = remainder.split("_")
    indices: List[int] = []
    for part in parts:
        if part.isdigit():
            indices.append(int(part))
        else:
            break
    if not indices:
        indices = [0]
    return tuple(indices)


def _list_sorted_pt_files(directory: str, prefix: str) -> List[str]:
    """List *.pt files in directory sorted by numeric indices"""
    if not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory) if f.endswith(".pt")]
    files.sort(key=lambda name: _extract_numeric_key(name, prefix))
    return files


def _copy_file_task(args: Tuple[int, str, str]) -> int:
    """Worker task to copy a single file"""
    _, src_path, dst_path = args
    shutil.copy2(src_path, dst_path)
    return 1


def _copy_files_to_flat_dir(copy_tasks: List[Tuple[int, str, str]], flat_dir: str) -> int:
    """Copy files into flat directory using multi-processing when beneficial"""
    if not copy_tasks:
        return 0
    
    desc = f"Copying/checking files to {flat_dir}"
    cpu_count = mp.cpu_count() or 1
    max_workers = min(len(copy_tasks), max(1, min(8, cpu_count)))
    
    if max_workers == 1:
        copied = 0
        for _, src_path, dst_path in tqdm(copy_tasks, total=len(copy_tasks), desc=desc):
            shutil.copy2(src_path, dst_path)
            copied += 1
        return copied
    
    copied = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_copy_file_task, task): task for task in copy_tasks
        }
        parallel_desc = f"{desc} (x{max_workers})"
        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc=parallel_desc):
            task = future_to_task[future]
            try:
                future.result()
                copied += 1
            except Exception as exc:
                logger.error(
                    "Failed to copy file | src=%s | dst=%s | error=%s",
                    task[1],
                    task[2],
                    exc,
                )
    return copied


def _ensure_flat_directory(directory: str, prefix: str) -> Tuple[str, List[str], List[str]]:
    """Ensure <directory>_flat exists with flattened file names and return metadata
    
    Returns:
        flat_dir: Absolute path to the flattened directory
        flat_base_names: Sorted base filenames (without .pt) inside flat_dir
        original_base_names: Base filenames inside original directory (or flat if unavailable)
    """
    directory = directory.rstrip(os.sep)
    flat_dir = f"{directory}_flat"
    
    source_files = _list_sorted_pt_files(directory, prefix) if os.path.isdir(directory) else []
    flat_files = _list_sorted_pt_files(flat_dir, prefix) if os.path.isdir(flat_dir) else []
    
    if source_files:
        os.makedirs(flat_dir, exist_ok=True)
        expected_flat_files = [f"{prefix}_{idx}.pt" for idx in range(len(source_files))]
        copied = 0
        if flat_files != expected_flat_files:
            existing_set = set(flat_files)
            copy_tasks: List[Tuple[int, str, str]] = []
            for idx, filename in enumerate(source_files):
                flat_name = expected_flat_files[idx]
                if flat_name in existing_set:
                    continue
                src_path = os.path.join(directory, filename)
                dst_path = os.path.join(flat_dir, flat_name)
                copy_tasks.append((idx, src_path, dst_path))
            
            copied = _copy_files_to_flat_dir(copy_tasks, flat_dir)
            if copied:
                logger.info(f"Created/updated {copied} flat files in: {flat_dir}")
            flat_files = _list_sorted_pt_files(flat_dir, prefix)
        else:
            logger.info(f"Flat directory already up-to-date: {flat_dir}")
        original_base_names = [os.path.splitext(f)[0] for f in source_files]
    else:
        os.makedirs(flat_dir, exist_ok=True)
        original_base_names = [os.path.splitext(f)[0] for f in flat_files]
    
    flat_base_names = [os.path.splitext(f)[0] for f in flat_files]
    if not original_base_names:
        original_base_names = flat_base_names.copy()
    
    return flat_dir, flat_base_names, original_base_names


def _rename_prefix(name: str, new_prefix: str) -> str:
    """Replace prefix before first underscore with new prefix"""
    if "_" not in name:
        return new_prefix
    suffix = name.split("_", 1)[1]
    return f"{new_prefix}_{suffix}"


# ==================== Output Path Helpers ====================

def _format_fraction_label(value: float, precision: int = 3) -> str:
    """Format fraction value for directory names (e.g., 0.125 -> 0p125)"""
    formatted = f"{value:.{precision}f}".rstrip('0').rstrip('.')
    if not formatted:
        formatted = "0"
    return formatted.replace(".", "p")


def _is_full_fraction_range(start_frac: float, end_frac: float, tol: float = 1e-9) -> bool:
    """Return True if requested fraction range covers the full dataset"""
    return math.isclose(start_frac, 0.0, abs_tol=tol) and math.isclose(end_frac, 1.0, abs_tol=tol)


def _format_indices_indicator(indices: List[int], preview_count: int = 5) -> str:
    """Format custom index list into readable indicator for directory names"""
    if not indices:
        return "idx_none"
    indices = sorted(indices)
    preview = "_".join(str(i) for i in indices[:preview_count])
    if len(indices) > preview_count:
        preview = f"{preview}_plus{len(indices) - preview_count}"
    return f"idx_{preview}"


def _prepare_measure_directory(base_model_path: str, start_frac: float, end_frac: float, indices: List[int] = None, tol: float = 1e-8, filtering: bool = False) -> Tuple[str, str]:
    """Create measurement directory based on requested range or index list"""
    tol_str = f"{tol:.0e}".replace("-", "n").replace("+", "")
    adaptive_str = "_adaptive" if filtering else ""
    if indices:
        dir_name = f"measure_{_format_indices_indicator(indices)}{adaptive_str}_tol{tol_str}"
    elif _is_full_fraction_range(start_frac, end_frac):
        dir_name = f"measure_full{adaptive_str}_tol{tol_str}"
    else:
        indicator = f"{_format_fraction_label(start_frac)}_{_format_fraction_label(end_frac)}"
        dir_name = f"measure_{indicator}{adaptive_str}_tol{tol_str}"
    full_path = os.path.join(base_model_path, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path, dir_name


def _normalize_index_list(indices, total_files: int) -> List[int]:
    """Validate and normalize custom index input from CLI (list/tuple/set/string)"""
    if indices is None:
        return []
    
    if isinstance(indices, str):
        sanitized = indices.replace(";", ",").replace(" ", ",")
        tokens = [token for token in sanitized.split(",") if token]
    elif isinstance(indices, (list, tuple, set)):
        tokens = list(indices)
    else:
        raise TypeError("file_indices must be a string or an iterable of integers")
    
    normalized: List[int] = []
    seen = set()
    for token in tokens:
        try:
            value = int(token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid index value: {token}") from exc
        if value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    
    if not normalized:
        raise ValueError("file_indices is empty after parsing")
    
    for idx in normalized:
        if idx < 0 or idx >= total_files:
            raise ValueError(f"file_indices contains out-of-range index: {idx} (total files: {total_files})")
    
    return normalized


# ==================== Helper Functions ====================

def init_pyscf_optimized(atoms, pos, unit="ang", xc="pbe", basis="def2svp", grid_level=None, grid_response=None, gpu=True):
    """Initialize PySCF with optimized settings for energy/force calculations
    
    Args:
        atoms: Atomic numbers
        pos: Atomic positions
        unit: Unit of positions
        xc: Exchange-correlation functional (default: "pbe")
        basis: Basis set (default: "def2svp")
        grid_level: DFT grid level (None=default 3, 2=recommended, 1=fast)
        grid_response: Include grid response (None=default True, False=faster)
    
    Returns:
        calc_mf: PySCF mean-field object
        grad_frame: PySCF gradient object
        settings: dict of settings used
    """
    calc_mf = init_pyscf_mf(atoms, pos, unit=unit, xc=xc, basis=basis, gpu=gpu)
    
    # Apply grid level optimization if specified
    if grid_level is not None:
        calc_mf.grids.level = grid_level
    
    grad_frame = calc_mf.nuc_grad_method()
    
    # Apply grid response optimization if specified
    if grid_response is not None:
        grad_frame.grid_response = grid_response
    
    # Record settings used
    settings = {
        "grid_level": calc_mf.grids.level,
        "grid_response": grad_frame.grid_response if hasattr(grad_frame, 'grid_response') else True,
        "xc": calc_mf.xc,
        "basis": calc_mf.basis,
        "unit": unit
    }
    
    return calc_mf, grad_frame, settings


def to_numpy_float64(data):
    """Convert any data type to numpy float64 array (optimized to avoid unnecessary copies)"""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy().astype(np.float64, copy=False)
    elif isinstance(data, np.ndarray):
        if data.dtype == np.float64:
            return data  # No conversion needed
        return data.astype(np.float64, copy=False)
    else:
        return np.array(data, dtype=np.float64)


def to_scalar_float(value):
    """Convert any value to Python float (double precision)"""
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def init_worker(pred_xc, gt_xc, calc_xc, basis, pred_tol, gt_tol, pad_eigval, filtering):
    """Initialize worker process with proper threading settings and calculation parameters"""
    # Set threading environment variables
    threading_env_vars = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1"
    }
    os.environ.update(threading_env_vars)
    torch.set_num_threads(1)
    
    # Set calculation settings for this worker process
    global _pred_xc, _gt_xc, _calc_xc, _basis, _pred_tol, _gt_tol, _pad_eigval, _enable_filtering
    _pred_xc = pred_xc
    _gt_xc = gt_xc
    _calc_xc = calc_xc
    _basis = basis
    _pred_tol = pred_tol
    _gt_tol = gt_tol
    _pad_eigval = pad_eigval
    _enable_filtering = filtering


def get_uma_instance():
    """Get or create UMA instance for current process"""
    global _process_uma
    if _process_uma is None:
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
        _process_uma = FAIRChemCalculator(predictor, task_name="omol")
    return _process_uma


# ==================== Data Loading and Computation ====================

def load_or_compute_calc_data(calc_path, gt_path, atoms, pos):
    """Load existing calc data or compute it using PySCF with optimizations"""
    if os.path.exists(calc_path):
        try:
            with Timer("Load calc_data"):
                return torch.load(calc_path, map_location='cpu', weights_only=False)
        except Exception:
            pass
    
    # Need to compute - load gt_data first
    with Timer("Load gt_data"):
        gt_data = torch.load(gt_path, map_location='cpu', weights_only=False)
    
    # Compute calc data using PySCF with optimizations
    with Timer("Init PySCF"):
        calc_mf, grad_frame, settings = init_pyscf_optimized(atoms, pos, unit="ang", xc=_calc_xc, basis=_basis)
        
        # Additional SCF optimizations (only for calc_data computation)
        calc_mf.max_cycle = 50
        calc_mf.conv_tol = 1e-9  # Slightly looser convergence for speed
        # calc_mf.direct_scf = True  # Faster for large systems
    
    # SCF calculation
    with Timer("SCF calculation"):
        calc_data = gt_data.copy()
        calc_data["calc_energy"] = calc_mf.kernel()
    
    # Extract results
    with Timer("Extract matrices"):
        calc_data["hamiltonian"] = torch.tensor(calc_mf.get_fock(dm=calc_mf.make_rdm1()), dtype=torch.float64)
        calc_data["overlap"] = torch.tensor(calc_mf.get_ovlp(), dtype=torch.float64)
        calc_data["density_matrix"] = torch.tensor(calc_mf.make_rdm1(), dtype=torch.float64)
        calc_data["mo_energy"] = torch.tensor(calc_mf.mo_energy, dtype=torch.float64)
    
    # Force calculation
    with Timer("Force calculation"):
        calc_data["calc_forces"] = torch.tensor(-grad_frame.kernel(), dtype=torch.float64)
    
    calc_data["method"] = "RKS"
    calc_data["xc"] = settings["xc"]
    calc_data["basis"] = settings["basis"]
    calc_data["grid_level"] = settings["grid_level"]
    calc_data["grid_response"] = settings["grid_response"]
    calc_data["scf_cycles"] = calc_mf.cycles
    
    # Save results
    with Timer("Save calc_data"):
        torch.save(calc_data, calc_path)
    
    return calc_data


def prepare_hamiltonian_and_density(hamiltonian, atoms, calc_overlap, convention=None, tol=1e-8, pad_eigval=1):
    """Prepare Hamiltonian transformation and compute density matrix
    
    Returns:
        density: Density matrix
        mo_energy: Molecular orbital energies
        mo_coeff: Molecular orbital coefficients
        ham_transformed: Transformed Hamiltonian matrix
        res: Full result dictionary from density matrix calculation
    """
    dtype = torch.float64
    overlap = calc_overlap.to(dtype)
    
    with Timer("    Matrix transform"):
        ham_transformed = matrix_transform_single(hamiltonian.unsqueeze(0), atoms, convention=convention).to(dtype)
    
    with Timer("    Calc DM"):
        density, res = calc_dm0_from_ham_(atoms, overlap, ham_transformed, tol=tol, pad_eigval=pad_eigval)
    
    # Extract orbital properties
    mo_energy = res["orbital_energies"].squeeze().numpy()
    mo_coeff = res["orbital_coefficients"].squeeze().numpy()
    
    return density, mo_energy, mo_coeff, ham_transformed, res

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

def calc_density(mo_coeff, n_occ):
    """Calculate density matrix from molecular orbital energies and coefficients"""
    sliced_mo_coeff = mo_coeff[:, :n_occ]
    density = sliced_mo_coeff @ sliced_mo_coeff.T * 2
    return density


def compute_energy_forces(density, mo_energy, mo_coeff, calc_mf, grad_frame, gpu=True):
    """Compute energy and forces from density matrix and orbital properties
    
    Returns:
        energy: Total energy
        forces: Atomic forces
    """
    with Timer("    Energy calc"):
        if gpu:
            cur_density = asarray(density)
        else:
            cur_density = density
        energy = calc_mf.energy_tot(cur_density)
    
    mo_occ = calc_mf.get_occ(mo_energy, mo_coeff)
    
    with Timer("    Force calc"):
        if gpu:
            cur_mo_energy = mo_energy
            cur_mo_coeff = mo_coeff
            cur_mo_occ = mo_occ
            grad_frame.base.mo_energy = cur_mo_energy
            grad_frame.base.mo_coeff = cur_mo_coeff
            grad_frame.base.mo_occ = cur_mo_occ
        else:
            cur_mo_energy = mo_energy
            cur_mo_coeff = mo_coeff
            cur_mo_occ = mo_occ

        forces = -grad_frame.kernel(
            mo_energy=cur_mo_energy,
            mo_coeff=-cur_mo_coeff,
            mo_occ=cur_mo_occ)
    
    return energy, forces


def compute_energy_forces_from_hamiltonian(hamiltonian, atoms, calc_overlap, calc_mf, grad_frame, convention=None, tol=1e-8, pad_eigval=1, gpu=True):
    """Compute energy, forces, and orbital energies from Hamiltonian matrix"""
    
    # Prepare Hamiltonian and density matrix
    density, mo_energy, mo_coeff, ham_transformed, res = prepare_hamiltonian_and_density(
        hamiltonian, atoms, calc_overlap, convention=convention, tol=tol, pad_eigval=pad_eigval
    )
    
    # Compute energy and forces
    energy, forces = compute_energy_forces(density, mo_energy, mo_coeff, calc_mf, grad_frame, gpu=gpu)
    
    return energy, forces, mo_energy, ham_transformed.squeeze(0).numpy()

def load_or_compute_uma_data(omol_path, atoms, pos):
    """Load existing UMA data or compute it"""
    if os.path.exists(omol_path):
        omol_data = torch.load(omol_path, map_location='cpu', weights_only=False)
        return omol_data["energy"], omol_data["forces"]
    
    # Compute UMA data
    start_time = time.time()
    uma = get_uma_instance()
    uma_atom = Atoms(numbers=atoms.squeeze(), positions=pos)
    uma_atom.info["charge"] = 0
    uma_atom.info["spin"] = 0
    uma.calculate(uma_atom, ["energy, forces"], [])
    uma_time = time.time() - start_time
    
    omol_energy = uma.results["energy"]
    omol_forces = uma.results["forces"]
    
    omol_data = {
        "energy": omol_energy,
        "forces": omol_forces,
        "pos": pos,
        "atoms": atoms,
        "uma_time": uma_time
    }
    torch.save(omol_data, omol_path)
    
    return omol_energy, omol_forces


# ==================== Main Processing Function ====================

def process_single_file(args: Tuple) -> Dict:
    """Process a single file tuple
    
    Args:
        args: Tuple of (file_index, pred_path, gt_path, calc_path, omol_path, gt_calc_path)
        
    Returns:
        Dictionary containing computed results or error information
    """
    if len(args) == 6:
        file_index, pred_path, gt_path, calc_path, omol_path, gt_calc_path = args
        file_meta = {}
    else:
        file_index, pred_path, gt_path, calc_path, omol_path, gt_calc_path, file_meta = args
    
    file_meta = file_meta or {}
    pred_alias = file_meta.get("pred_flat_name")
    gt_alias = file_meta.get("gt_flat_name")
    # Use pred_xc in processed directory name (always include for clarity)
    tol_str = f"{_pred_tol:.0e}".replace("-", "n").replace("+", "")
    pad_eigval_str = f"{_pad_eigval:.0e}".replace("-", "n").replace("+", "").replace(".", "p")
    adaptive_str = "_adaptive" if _enable_filtering else ""
    pred_processed_suffix = f"_processed_{_pred_xc}{adaptive_str}_tol{tol_str}_pad{pad_eigval_str}"
    pred_path_processed = pred_path.replace(os.path.dirname(pred_path), f"{os.path.dirname(pred_path)}{pred_processed_suffix}")
    
    try:
        file_start_time = time.time() if _enable_profiling else None
        
        # OPTIMIZATION: Try to load from cached gt_calc first to avoid loading gt_data
        atoms = None
        pos = None
        convention = None
        gt_calc_data = None
        gt_data = None
        
        if os.path.exists(gt_calc_path):
            gt_calc_data = torch.load(gt_calc_path, map_location='cpu', weights_only=False)
            
            # Check if gt_calc has all metadata
            if "atoms" in gt_calc_data and "pos" in gt_calc_data and "convention" in gt_calc_data:
                atoms = gt_calc_data["atoms"]
                pos = gt_calc_data["pos"]
                convention = gt_calc_data["convention"]
                need_gt_data = False
            else:
                need_gt_data = True
        else:
            need_gt_data = True
        
        # Only load gt_data if we don't have metadata in gt_calc
        if need_gt_data:
            gt_data = torch.load(gt_path, map_location='cpu', weights_only=False)
            atoms = gt_data["atoms"]
            pos = to_numpy_float64(gt_data["pos"])
            
            length_unit = gt_data.get("length_unit", "angstrom").lower()
            if length_unit == "bohr":
                pos = pos * np.float64(BOHR2ANG)
            elif length_unit != "angstrom":
                raise ValueError(f"Invalid length unit: {gt_data.get('length_unit')}")
            
            matrix_format = gt_data["format"].lower()
            convention = None if matrix_format == "pyscf_def2svp" else "e3nn_to_pyscf_def2svp"
        else:
            # gt_data not needed for cached processing
            gt_data = None
        
        # Load pred_data
        with Timer("Load pred_data"):
            pred_path_to_load = pred_path_processed if os.path.exists(pred_path_processed) else pred_path
            pred_data = torch.load(pred_path_to_load, map_location='cpu', weights_only=False)
        
        # Load or compute calc_data (reference calculation)
        with Timer("Load/compute calc_data"):
            calc_data = load_or_compute_calc_data(calc_path, gt_path, atoms, pos)
            calc_energy = to_scalar_float(calc_data["calc_energy"])
            calc_forces = to_numpy_float64(calc_data["calc_forces"])
        
        calc_overlap = calc_data["overlap"]
        if isinstance(calc_overlap, torch.Tensor):
            calc_overlap = calc_overlap.to(torch.float64).unsqueeze(0)
        else:
            calc_overlap = torch.tensor(calc_overlap, dtype=torch.float64).unsqueeze(0)
        
        # Load or compute prediction data
        with Timer("PRED processing"):
            has_pred_calc_forces = "calc_forces" in pred_data
            has_pred_mo = "mo_energy" in pred_data
            has_pred_ham = "hamiltonian_transformed" in pred_data
    
            if has_pred_calc_forces and has_pred_mo and has_pred_ham:
                # All data already computed - just extract (FAST PATH)
                pred_energy = pred_data["calc_energy"]
                pred_forces = pred_data["calc_forces"]
                pred_mo_energy = pred_data["mo_energy"]
                pred_ham_transformed = pred_data["hamiltonian_transformed"]
            elif has_pred_calc_forces and not (has_pred_mo and has_pred_ham):
                # Has energy/forces but missing mo_energy or hamiltonian_transformed
                pred_energy = pred_data["calc_energy"]
                pred_forces = pred_data["calc_forces"]
                
                with Timer("  Init PySCF (pred)"):
                    calc_mf, grad_frame, _ = init_pyscf_optimized(atoms, pos, unit="ang", xc=_pred_xc, basis=_basis)
                
                # Recompute to get mo_energy and hamiltonian
                pred_ham_transformed = matrix_transform_single(pred_data["pred_hamiltonian"].unsqueeze(0) if pred_data["pred_hamiltonian"].dim() == 2 else pred_data["pred_hamiltonian"], atoms, convention=convention).to(torch.float64)
                
                if _enable_filtering:
                    pred_mo_energy_safe, pred_mo_coeff_safe = calc_mo_energy_and_coeff(
                        pred_ham_transformed, calc_overlap, tol=_gt_tol, pad_eigval=None
                    )
                    pred_mo_energy_unsatable, pred_mo_coeff_unsatable = calc_mo_energy_and_coeff(
                        pred_ham_transformed, calc_overlap, tol=_pred_tol, pad_eigval=_pad_eigval
                    )
                    n_occ = atoms.sum() // 2
                    
                    # 0.03 Hartree is the threshold for unstable calculation
                    # This is a heuristic threshold based on the results of the test script
                    # safe is default (about 99.8% success rate)
                    # unsatable is used if the difference is greater than 0.03 Hartree
                    mo_energy_diff = pred_mo_energy_safe - pred_mo_energy_unsatable
                    if abs(min(mo_energy_diff[:n_occ+1])) > 0.03:
                        pred_mo_energy = pred_mo_energy_unsatable
                        print("Index:", file_index, "Unstable")
                        tol = _pred_tol
                        pad_eigval = _pad_eigval
                    else:
                        pred_mo_energy = pred_mo_energy_safe
                        tol = _gt_tol
                        pad_eigval = None
                else:
                    pred_mo_energy, _ = calc_mo_energy_and_coeff(
                        pred_ham_transformed, calc_overlap, tol=_pred_tol, pad_eigval=_pad_eigval
                    )
                    tol = _pred_tol
                    pad_eigval = _pad_eigval
                pred_data.update({
                    "mo_energy": pred_mo_energy,
                    "hamiltonian_transformed": pred_ham_transformed,
                    "tol": tol,
                    "pad_eigval": pad_eigval
                })
                torch.save(pred_data, pred_path_processed)
                del calc_mf, grad_frame
            else:
                # Need to compute everything
                with Timer("  Init PySCF (pred)"):
                    calc_mf, grad_frame, _ = init_pyscf_optimized(atoms, pos, unit="ang", xc=_pred_xc, basis=_basis)
                
                pred_ham_transformed = matrix_transform_single(pred_data["pred_hamiltonian"].unsqueeze(0) if pred_data["pred_hamiltonian"].dim() == 2 else pred_data["pred_hamiltonian"], atoms, convention=convention).to(torch.float64)
                
                if _enable_filtering:
                    pred_mo_energy_safe, pred_mo_coeff_safe = calc_mo_energy_and_coeff(
                        pred_ham_transformed, calc_overlap, tol=_gt_tol, pad_eigval=None
                    )
                    pred_mo_energy_unsatable, pred_mo_coeff_unsatable = calc_mo_energy_and_coeff(
                        pred_ham_transformed, calc_overlap, tol=_pred_tol, pad_eigval=_pad_eigval
                    )
                    n_occ = atoms.sum() // 2
                    mo_energy_diff = pred_mo_energy_safe - pred_mo_energy_unsatable
                    if abs(min(mo_energy_diff[:n_occ+1])) > 0.03:
                        print(f"Index: {file_index} Unstable MO energy difference: {max(abs(mo_energy_diff[:n_occ+1]))}")
                        pred_mo_energy = pred_mo_energy_unsatable
                        pred_mo_coeff = pred_mo_coeff_unsatable
                        tol = _pred_tol
                        pad_eigval = _pad_eigval
                    else:
                        pred_mo_energy = pred_mo_energy_safe
                        pred_mo_coeff = pred_mo_coeff_safe
                        tol = _gt_tol
                        pad_eigval = None
                else:
                    pred_mo_energy, pred_mo_coeff = calc_mo_energy_and_coeff(
                        pred_ham_transformed, calc_overlap, tol=_pred_tol, pad_eigval=_pad_eigval
                    )
                    n_occ = atoms.sum() // 2
                    tol = _pred_tol
                    pad_eigval = _pad_eigval
                pred_energy, pred_forces = compute_energy_forces(
                    calc_density(pred_mo_coeff, n_occ), 
                    pred_mo_energy,
                    pred_mo_coeff,
                    calc_mf,
                    grad_frame
                )
                pred_data.update({
                    "calc_energy": pred_energy,
                    "calc_forces": pred_forces,
                    "mo_energy": pred_mo_energy,
                    "hamiltonian_transformed": pred_ham_transformed,
                    "tol": tol,
                    "pad_eigval": pad_eigval
                })
                torch.save(pred_data, pred_path_processed)
                del calc_mf, grad_frame
        
        # Load or compute ground truth data
        with Timer("GT processing"):
            if os.path.exists(gt_calc_path):
                # gt_calc_data already loaded above
                if gt_calc_data is None:
                    gt_calc_data = torch.load(gt_calc_path, map_location='cpu', weights_only=False)
                
                has_gt_energy = "calc_energy" in gt_calc_data
                has_gt_forces = "calc_forces" in gt_calc_data
                has_gt_mo = "mo_energy" in gt_calc_data and gt_calc_data["mo_energy"] is not None
                has_gt_ham = "hamiltonian_transformed" in gt_calc_data and gt_calc_data["hamiltonian_transformed"] is not None
                
                if has_gt_energy and has_gt_forces and has_gt_mo and has_gt_ham:
                    # All data exists - just extract (FAST PATH)
                    gt_energy = gt_calc_data["calc_energy"]
                    gt_forces = gt_calc_data["calc_forces"]
                    gt_mo_energy = gt_calc_data["mo_energy"]
                    gt_ham_transformed = gt_calc_data["hamiltonian_transformed"]
                elif has_gt_energy and has_gt_forces:
                    # Has energy/forces but missing mo_energy or hamiltonian_transformed
                    if gt_data is None:
                        gt_data = torch.load(gt_path, map_location='cpu', weights_only=False)
                    
                    gt_energy = gt_calc_data["calc_energy"]
                    gt_forces = gt_calc_data["calc_forces"]
                    
                    with Timer("  Init PySCF (gt)"):
                        calc_mf, grad_frame, _ = init_pyscf_optimized(atoms, pos, unit="ang", xc=_gt_xc, basis=_basis)
                    
                    _, _, gt_mo_energy, gt_ham_transformed = compute_energy_forces_from_hamiltonian(
                        gt_data["hamiltonian"], atoms, calc_overlap, calc_mf, grad_frame, convention, tol=_gt_tol, pad_eigval=_pad_eigval
                    )
                    
                    gt_calc_data.update({
                        "mo_energy": gt_mo_energy,
                        "hamiltonian_transformed": gt_ham_transformed,
                        "convention": convention
                    })
                    torch.save(gt_calc_data, gt_calc_path)
                    del calc_mf, grad_frame
                else:
                    # Missing energy/forces - need to compute everything
                    if gt_data is None:
                        gt_data = torch.load(gt_path, map_location='cpu', weights_only=False)
                    
                    with Timer("  Init PySCF (gt)"):
                        calc_mf, grad_frame, _ = init_pyscf_optimized(atoms, pos, unit="ang", xc=_gt_xc, basis=_basis)
                    
                    gt_energy, gt_forces, gt_mo_energy, gt_ham_transformed = compute_energy_forces_from_hamiltonian(
                        gt_data["hamiltonian"], atoms, calc_overlap, calc_mf, grad_frame, convention, tol=_gt_tol, pad_eigval=_pad_eigval
                    )
                    gt_calc_data.update({
                        "calc_energy": gt_energy,
                        "calc_forces": gt_forces,
                        "mo_energy": gt_mo_energy,
                        "hamiltonian_transformed": gt_ham_transformed,
                        "convention": convention
                    })
                    torch.save(gt_calc_data, gt_calc_path)
                    del calc_mf, grad_frame
            else:
                # First time - need to compute everything
                if gt_data is None:
                    gt_data = torch.load(gt_path, map_location='cpu', weights_only=False)
                
                with Timer("  Init PySCF (gt)"):
                    calc_mf, grad_frame, _ = init_pyscf_optimized(atoms, pos, unit="ang", xc=_gt_xc, basis=_basis)
                
                gt_energy, gt_forces, gt_mo_energy, gt_ham_transformed = compute_energy_forces_from_hamiltonian(
                    gt_data["hamiltonian"], atoms, calc_overlap, calc_mf, grad_frame, convention, tol=_gt_tol, pad_eigval=_pad_eigval
                )
                gt_calc_data = {
                    "calc_energy": gt_energy,
                    "calc_forces": gt_forces,
                    "mo_energy": gt_mo_energy,
                    "hamiltonian_transformed": gt_ham_transformed,
                    "pos": pos,
                    "atoms": atoms,
                    "convention": convention
                }
                torch.save(gt_calc_data, gt_calc_path)
                del calc_mf, grad_frame
        
        # Convert all data to double precision numpy arrays (optimized - avoid redundant conversions)
        with Timer("Type conversion & error calc"):
            calc_ham = to_numpy_float64(calc_data["hamiltonian"])
            pred_ham_transformed = to_numpy_float64(pred_ham_transformed)
            gt_ham_transformed = to_numpy_float64(gt_ham_transformed)
            pred_mo_energy = to_numpy_float64(pred_mo_energy)
            gt_mo_energy = to_numpy_float64(gt_mo_energy)
            pred_forces = to_numpy_float64(pred_forces)
            gt_forces = to_numpy_float64(gt_forces)
            pred_energy = to_scalar_float(pred_energy)
            gt_energy = to_scalar_float(gt_energy)
            
            # Get calc MO energies from calc_data (always available)
            if "mo_energy" in calc_data:
                calc_mf_mo = to_numpy_float64(calc_data["mo_energy"])
            else:
                # Compute only if not cached (rare case - old calc_data format)
                calc_mf_temp, _, _ = init_pyscf_optimized(atoms, pos, unit="ang", xc=_calc_xc, basis=_basis)
                calc_mf_mo = to_numpy_float64(calc_mf_temp.mo_energy)
                del calc_mf_temp
            
            # Compute errors efficiently (vectorized operations)
            ham_error_pred_gt = float(np.abs(pred_ham_transformed - gt_ham_transformed).mean())
            ham_error_pred_calc = float(np.abs(pred_ham_transformed - calc_ham).mean())
            ham_error_gt_calc = float(np.abs(gt_ham_transformed - calc_ham).mean())
            
            mo_error_pred_gt = float(np.abs(pred_mo_energy - gt_mo_energy).mean())
            mo_error_pred_calc = float(np.abs(pred_mo_energy - calc_mf_mo).mean())
            mo_error_gt_calc = float(np.abs(gt_mo_energy - calc_mf_mo).mean())
            
            # Compute force norms and differences (vectorized operations)
            pred_forces_norm = np.linalg.norm(pred_forces, axis=1)
            gt_forces_norm = np.linalg.norm(gt_forces, axis=1)
            calc_forces_norm = np.linalg.norm(calc_forces, axis=1)
            
            forces_diff_pred_gt = np.abs(pred_forces - gt_forces)
            forces_diff_pred_calc = np.abs(pred_forces - calc_forces)
            forces_diff_gt_calc = np.abs(gt_forces - calc_forces)
            forces_norm_diff_pred_gt = np.abs(pred_forces_norm - gt_forces_norm)
            forces_norm_diff_pred_calc = np.abs(pred_forces_norm - calc_forces_norm)
            forces_norm_diff_gt_calc = np.abs(gt_forces_norm - calc_forces_norm)
        
        # Build results dictionary (using pre-computed values)
        num_atoms = int(len(atoms))
        
        results = {
            "file_index": int(file_index),
            "pred_file_name": f"{pred_alias}.pt" if pred_alias else os.path.basename(pred_path),
            "gt_file_name": f"{gt_alias}.pt" if gt_alias else os.path.basename(gt_path),
            "pred_path": pred_path,
            # Energy metrics
            "pred_energy": pred_energy,
            "gt_energy": gt_energy,
            "calc_energy": calc_energy,
            "energy_diff (pred-gt)": float(pred_energy - gt_energy),
            "energy_diff (pred-calc_energy)": float(pred_energy - calc_energy),
            "energy_diff (gt-calc_energy)": float(gt_energy - calc_energy),
            "energy_diff_per_atom (pred-gt)": float(pred_energy - gt_energy) / num_atoms,
            "energy_diff_per_atom (pred-calc_energy)": float(pred_energy - calc_energy) / num_atoms,
            "energy_diff_per_atom (gt-calc_energy)": float(gt_energy - calc_energy) / num_atoms,
            
            # Force metrics (using pre-computed differences)
            "pred_force": pred_forces,
            "gt_force": gt_forces,
            "calc_force": calc_forces,
            "forces_diff l2 (pred-gt)": float(forces_diff_pred_gt.mean()),
            "forces_diff l2 (pred-calc_forces)": float(forces_diff_pred_calc.mean()),
            "forces_diff l2 (gt-calc_forces)": float(forces_diff_gt_calc.mean()),
            "pred_force_norm": pred_forces_norm,
            "gt_force_norm": gt_forces_norm,
            "calc_force_norm": calc_forces_norm,
            "pred_force_norm_diff (pred-gt)": float(forces_norm_diff_pred_gt.mean()),
            "pred_force_norm_diff (pred-calc_forces)": float(forces_norm_diff_pred_calc.mean()),
            "pred_force_norm_diff (gt-calc_forces)": float(forces_norm_diff_gt_calc.mean()),
            
            # Hamiltonian errors
            "hamiltonian_error (pred-gt)": ham_error_pred_gt,
            "hamiltonian_error (pred-calc)": ham_error_pred_calc,
            "hamiltonian_error (gt-calc)": ham_error_gt_calc,
            
            # Orbital energy errors
            "mo_energy_error (pred-gt)": mo_error_pred_gt,
            "mo_energy_error (pred-calc)": mo_error_pred_calc,
            "mo_energy_error (gt-calc)": mo_error_gt_calc,
            
            # Atom information
            "num_atoms": num_atoms
        }
        
        # Process UMA data if requested
        if omol_path is not None:
            with Timer("UMA processing"):
                omol_energy, omol_forces = load_or_compute_uma_data(omol_path, atoms, pos)
                
                # Convert units: eV -> Hartree, eV/Ang -> Hartree/Bohr
                omol_energy = float(omol_energy * eV2HA)
                omol_forces = to_numpy_float64(omol_forces) * np.float64(eV2HA / ANG2BOHR)
                
                # Compute differences with GT and calc (optimized)
                omol_forces_diff_gt = np.abs(gt_forces - omol_forces)
                omol_forces_diff_calc = np.abs(calc_forces - omol_forces)
                omol_forces_norm = np.linalg.norm(omol_forces, axis=1)
                omol_forces_norm_diff_gt = np.abs(gt_forces_norm - omol_forces_norm)
                omol_forces_norm_diff_calc = np.abs(calc_forces_norm - omol_forces_norm)
            
            results.update({
                "omol_energy": omol_energy,
                "energy_diff (gt-omol_energy)": float(gt_energy - omol_energy),
                "energy_diff (calc-omol_energy)": float(calc_energy - omol_energy),
                "energy_diff_per_atom (gt-omol_energy)": float(gt_energy - omol_energy) / num_atoms,
                "energy_diff_per_atom (calc-omol_energy)": float(calc_energy - omol_energy) / num_atoms,
                "omol_forces": omol_forces,
                "forces_diff l2 (gt-omol_forces)": float(omol_forces_diff_gt.mean()),
                "forces_diff l2 (calc-omol_forces)": float(omol_forces_diff_calc.mean()),
                "omol_force_norm": omol_forces_norm,
                "gt_force_norm_diff (gt-omol_forces)": float(omol_forces_norm_diff_gt.mean()),
                "gt_force_norm_diff (calc-omol_forces)": float(omol_forces_norm_diff_calc.mean())
            })
        
        # Log total file processing time if profiling
        if _enable_profiling and file_start_time is not None:
            total_time = time.time() - file_start_time
            logger.info(f"TOTAL FILE TIME: {total_time:.3f}s - {os.path.basename(pred_path)}")
            logger.info("-"*80)
        
        return results
        
    except Exception as e:
        error_msg = f"Error processing {pred_path}: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        # Log error to file
        error_dir = os.path.dirname(os.path.dirname(pred_path))
        error_file_path = os.path.join(error_dir, "error_files.txt")
        with open(error_file_path, "a") as f:
            f.write(f"{pred_path} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return {
            "error": error_msg,
            "file": pred_path,
            "file_index": int(file_index),
            "pred_file_name": f"{pred_alias}.pt" if pred_alias else os.path.basename(pred_path),
            "line_number": traceback.extract_tb(e.__traceback__)[-1].lineno
        }


# ==================== Parallel Processing ====================

def process_all_files(file_list, pred_xc, gt_xc, calc_xc, basis, num_workers=4, chunksize=1, maxtasksperchild=1000, output_path=None, pred_tol=1e-8, gt_tol=1e-8, pad_eigval=1, filtering=False):
    """Process all files using multiprocessing Pool with imap_unordered
    
    PERFORMANCE TIP:
    - chunksize=1: Better progress tracking but more overhead
    - chunksize=2-5: Good balance for most cases
    - chunksize=10+: Faster processing but coarser progress updates
    
    Args:
        output_path: If provided, saves summary after first result for quick verification
    """
    logger.info(f"Processing {len(file_list)} files with {num_workers} workers")
    logger.info(f"Chunksize: {chunksize}, Max tasks per child: {maxtasksperchild}")
    
    # OPTIMIZATION: Auto-adjust chunksize for better performance
    if chunksize == 1 and len(file_list) > 100:
        recommended_chunksize = max(2, min(10, len(file_list) // (num_workers * 20)))
        logger.info(f"  TIP: Consider using --chunksize {recommended_chunksize} for faster processing")
    
    results = []
    ctx = mp.get_context('spawn')
    
    with ctx.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(pred_xc, gt_xc, calc_xc, basis, pred_tol, gt_tol, pad_eigval, filtering),
        maxtasksperchild=maxtasksperchild
    ) as pool:
        try:
            for idx, result in enumerate(tqdm(
                pool.imap_unordered(process_single_file, file_list, chunksize=chunksize),
                total=len(file_list),
                desc="Processing files",
                smoothing=0.1
            ), start=1):
                results.append(result)
                
                # Save first result immediately for quick verification
                if idx == 1 and output_path and result and "error" not in result:
                    logger.info("\n" + "="*80)
                    logger.info("FIRST FILE PROCESSED - Quick Summary:")
                    logger.info("="*80)
                    first_summary_path = output_path.replace(".csv", "_first_result_summary.txt")
                    with open(first_summary_path, 'w') as f:
                        f.write("="*80 + "\n")
                        f.write("FIRST FILE RESULT (Quick Verification)\n")
                        f.write("="*80 + "\n\n")
                        for key, value in result.items():
                            if key != "error":
                                f.write(f"{key:30s}: {value}\n")
                    logger.info(f"First result saved to: {first_summary_path}")
                    logger.info("="*80 + "\n")
        except KeyboardInterrupt:
            logger.warning("Interrupted by user, terminating pool...")
            pool.terminate()
            pool.join()
            raise
        except Exception as e:
            logger.error(f"Pool execution failed: {str(e)}")
            pool.terminate()
            pool.join()
            raise
    
    # Filter results
    successful_results = [r for r in results if r and "error" not in r]
    successful_results.sort(key=lambda r: r.get("file_index", float("inf")))
    error_results = [r for r in results if r and "error" in r]
    
    logger.info(f"Completed: {len(successful_results)} successful, {len(error_results)} failed")
    
    if error_results:
        logger.info("Failed files:")
        for result in error_results[:5]:
            logger.info(
                "  index=%s | file=%s | name=%s | error=%s",
                result.get("file_index", "Unknown"),
                result.get("file", "Unknown"),
                result.get("pred_file_name", "Unknown"),
                result.get("error", "Unknown error"),
            )
        if len(error_results) > 5:
            logger.info(f"  ... and {len(error_results) - 5} more")
    
    # OPTIMIZATION: gc.collect() only at the end, not in loop
    gc.collect()
    return successful_results


def process_sequential_safe(file_tuples, desc="Processing files"):
    """Process files sequentially with error handling"""
    results = []
    
    for position, file_tuple in enumerate(tqdm(file_tuples, desc=desc), start=1):
        if len(file_tuple) == 6:
            file_index, pred_path, gt_path, calc_path, omol_path, gt_calc_path = file_tuple
        else:
            file_index, pred_path, gt_path, calc_path, omol_path, gt_calc_path, _meta = file_tuple
        try:
            result = process_single_file(file_tuple)
            if result and "error" not in result:
                results.append(result)
            elif result and "error" in result:
                logger.warning(
                    "Skipping file due to error | seq=%s | index=%s | file=%s | name=%s | error=%s",
                    position,
                    result.get("file_index", file_index),
                    result.get("file", pred_path),
                    result.get("pred_file_name", os.path.basename(pred_path)),
                    result.get("error", "Unknown error"),
                )
        except Exception as e:
            logger.error(
                "Error processing file | seq=%s | index=%s | file=%s | error=%s",
                position,
                file_index,
                pred_path,
                str(e),
            )
            continue
    
    results.sort(key=lambda r: r.get("file_index", float("inf")))
    return results


# ==================== Visualization ====================

def generate_plots(results, output_path, uma=False):
    """Generate comprehensive visualization plots"""
    plot_dir = output_path.replace(".csv", "_plots")
    os.makedirs(plot_dir, exist_ok=True)
    logger.info(f"Generating plots in: {plot_dir}")
    
    # 1. Energy Parity Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (x_col, y_col, title) in zip(axes, [
        ("gt_energy", "pred_energy", "Prediction vs Ground Truth"),
        ("calc_energy", "pred_energy", "Prediction vs Calculation"),
        ("calc_energy", "gt_energy", "GT vs Calculation")
    ]):
        x_data = results[x_col] * HA2meV
        y_data = results[y_col] * HA2meV
        ax.scatter(x_data, y_data, alpha=0.5, s=10, edgecolors='none')
        lim = [min(x_data.min(), y_data.min()), max(x_data.max(), y_data.max())]
        ax.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel(f'{x_col.split("_")[0].capitalize()} Energy (meV)', fontsize=12)
        ax.set_ylabel(f'{y_col.split("_")[0].capitalize()} Energy (meV)', fontsize=12)
        ax.set_title(f'Energy: {title}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "energy_parity_plots.png"))
    plt.close()
    
    # 2. Error Distribution Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    error_configs = [
        (axes[0, 0], "energy_diff (pred-gt)", HA2meV, "Energy Error (meV)", "blue"),
        (axes[0, 1], "forces_diff l2 (pred-gt)", HA_BOHR_2_meV_ANG, "Force Error L2 (meV/Å)", "green"),
        (axes[0, 2], "hamiltonian_error (pred-gt)", 1, "Hamiltonian Frobenius Norm Error", "purple"),
        (axes[1, 0], "mo_energy_error (pred-gt)", HA2meV, "MO Energy MAE (meV)", "red"),
    ]
    
    for ax, key, factor, xlabel, color in error_configs:
        ax.hist(results[key] * factor, bins=50, alpha=0.7, color=color, edgecolor='black')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{xlabel} Distribution (Pred - GT)', fontsize=14)
        if "energy_diff" in key:
            ax.axvline(0, color='r', linestyle='--', lw=2)
        ax.grid(True, alpha=0.3)
    
    # Violin plot
    error_data = [
        results["energy_diff (pred-gt)"] * HA2meV,
        results["forces_diff l2 (pred-gt)"] * HA_BOHR_2_meV_ANG,
        results["hamiltonian_error (pred-gt)"],
        results["mo_energy_error (pred-gt)"] * HA2meV
    ]
    axes[1, 1].violinplot(error_data, showmeans=True, showmedians=True)
    axes[1, 1].set_xticks([1, 2, 3, 4])
    axes[1, 1].set_xticklabels(['Energy\n(meV)', 'Force\n(meV/Å)', 'Hamilton.\n(a.u.)', 'MO Energy\n(meV)'], fontsize=10)
    axes[1, 1].set_ylabel('Error Magnitude', fontsize=12)
    axes[1, 1].set_title('Error Comparison (Violin Plot)', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Cumulative distribution
    sorted_energy_err = np.sort(np.abs(results["energy_diff (pred-gt)"] * HA2meV))
    cumulative = np.arange(1, len(sorted_energy_err) + 1) / len(sorted_energy_err) * 100
    axes[1, 2].plot(sorted_energy_err, cumulative, 'b-', lw=2)
    axes[1, 2].set_xlabel('|Energy Error| (meV)', fontsize=12)
    axes[1, 2].set_ylabel('Cumulative Percentage (%)', fontsize=12)
    axes[1, 2].set_title('Cumulative Energy Error Distribution', fontsize=14)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(90, color='r', linestyle='--', lw=1, alpha=0.7, label='90%')
    axes[1, 2].axhline(95, color='g', linestyle='--', lw=1, alpha=0.7, label='95%')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "error_distributions.png"))
    plt.close()
    
    # 3. Bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = {
        'Energy MAE (meV)': results["energy_diff (pred-gt)"].abs().mean() * HA2meV,
        'Force MAE (meV/Å)': results["forces_diff l2 (pred-gt)"].mean() * HA_BOHR_2_meV_ANG,
        'Hamilton. Error': results["hamiltonian_error (pred-gt)"].mean(),
        'MO Energy MAE (meV)': results["mo_energy_error (pred-gt)"].mean() * HA2meV
    }
    
    x = np.arange(len(metrics))
    values = list(metrics.values())
    colors = ['blue', 'green', 'purple', 'red']
    
    bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys(), rotation=15, ha='right', fontsize=11)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Comparison of Different Error Metrics (Pred - GT)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "error_comparison.png"))
    plt.close()
    
    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    error_cols = [
        "energy_diff (pred-gt)",
        "forces_diff l2 (pred-gt)",
        "hamiltonian_error (pred-gt)",
        "mo_energy_error (pred-gt)"
    ]
    
    corr_matrix = results[error_cols].corr()
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(error_cols)))
    ax.set_yticks(np.arange(len(error_cols)))
    labels = ['Energy', 'Force', 'Hamiltonian', 'MO Energy']
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    for i in range(len(error_cols)):
        for j in range(len(error_cols)):
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                   ha="center", va="center", color="black", fontsize=12)
    
    ax.set_title('Error Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "error_correlation.png"))
    plt.close()
    
    logger.info(f"All plots saved to: {plot_dir}")


# ==================== Summary Statistics ====================

def pandas_summary(results, output_path, postfix, keys, unit="meV"):
    """Generate and save summary statistics in multiple formats"""
    unit_conversion = {
        "hartree": 1.0,
        "mev": HA2meV,
        "mu_hartree": 1e6,
        "mev/angstrom": HA_BOHR_2_meV_ANG,
        "mev/ang": HA_BOHR_2_meV_ANG,
        "mu_hartree/bohr": 1e6        
    }
    
    unit_lower = unit.lower()
    if unit_lower not in unit_conversion:
        raise ValueError(f"Invalid unit: {unit}, available: {list(unit_conversion.keys())}")
    
    conversion_factor = unit_conversion[unit_lower]
    base_dir = os.path.dirname(output_path)
    basename = os.path.basename(output_path)
    is_uma = "uma_results" in basename
    summary_dir_name = f"summary_{'uma_' if is_uma else ''}{postfix}"
    summary_dir = os.path.join(base_dir, summary_dir_name)
    os.makedirs(summary_dir, exist_ok=True)
    
    csv_dir = os.path.join(summary_dir, "csv")
    json_dir = os.path.join(summary_dir, "json")
    md_dir = os.path.join(summary_dir, "md")
    txt_dir = os.path.join(summary_dir, "txt")
    
    for path in (csv_dir, json_dir, md_dir, txt_dir):
        os.makedirs(path, exist_ok=True)
    
    # Compute statistics
    stats_data = []
    for key in keys:
        stats_data.append({
            "metric": key,
            "mean (abs)": results[key].abs().mean() * conversion_factor,
            "std (abs)": results[key].abs().std() * conversion_factor,
            "median (abs)": results[key].abs().median() * conversion_factor,
            "min (abs)": results[key].abs().min() * conversion_factor,
            "max (abs)": results[key].abs().max() * conversion_factor,
            "unit": unit
        })
    
    # Save as vertical CSV
    summary_df = pd.DataFrame(stats_data)
    csv_path = os.path.join(csv_dir, "summary.csv")
    summary_df.to_csv(csv_path, index=False, float_format='%.6f')
    
    # Save as JSON
    json_data = {
        "unit": unit,
        "statistics": {item["metric"]: {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                        for k, v in item.items() if k != "metric"}
                      for item in stats_data}
    }
    json_path = os.path.join(json_dir, "summary.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save as Markdown
    md_path = os.path.join(md_dir, "summary.md")
    with open(md_path, 'w') as f:
        f.write(f"# Summary Statistics\n\n**Unit:** {unit}\n\n")
        f.write(f"| Metric | Mean | Std | Median | Min | Max |\n")
        f.write(f"|--------|------|-----|--------|-----|-----|\n")
        for item in stats_data:
            f.write(f"| {item['metric']} | {item['mean (abs)']:.6f} | {item['std (abs)']:.6f} | "
                   f"{item['median (abs)']:.6f} | {item['min (abs)']:.6f} | {item['max (abs)']:.6f} |\n")
    
    # Save as formatted text
    txt_path = os.path.join(txt_dir, "summary.txt")
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"SUMMARY STATISTICS - {postfix}\n")
        f.write("="*80 + "\n")
        f.write(f"Unit: {unit}\n")
        f.write(f"Number of samples: {len(results)}\n")
        f.write("="*80 + "\n\n")
        
        for item in stats_data:
            f.write(f"{item['metric']}:\n")
            f.write(f"  Mean   : {item['mean (abs)']:>12.6f} {unit}\n")
            f.write(f"  Std    : {item['std (abs)']:>12.6f} {unit}\n")
            f.write(f"  Median : {item['median (abs)']:>12.6f} {unit}\n")
            f.write(f"  Min    : {item['min (abs)']:>12.6f} {unit}\n")
            f.write(f"  Max    : {item['max (abs)']:>12.6f} {unit}\n\n")
    
    logger.info(f"Summary saved in: {summary_dir}")


# ==================== Main Function ====================

def main(
    dataset_name,
    model_name,
    model_prefix,
    model_postfix,
    pred_xc="b3lyp",
    gt_xc="b3lyp",
    calc_xc="pbe",
    basis="def2svp",
    chunksize=1,
    maxtasksperchild=1000,
    use_parallel=True,
    num_workers=16,
    max_cpu_cores=None,
    start_frac=0.0,
    end_frac=1.0,
    reverse_order=False,
    file_indices=None,
    uma=False,
    profile=False,
    pred_tol=1e-8,
    gt_tol=1e-8,
    pad_eigval=1,
    filtering=False
):
    """Main processing function
    
    Args:
        pred_xc: XC functional for prediction energy/force calculations (default: b3lyp)
        gt_xc: XC functional for GT energy/force calculations (default: b3lyp)
        calc_xc: XC functional for reference calculations (default: pbe)
        file_indices: Optional list/comma-separated string of specific file indices to process
        pred_tol: Tolerance for prediction density matrix calculation (default: 1e-8)
        gt_tol: Tolerance for ground truth density matrix calculation (default: 1e-8)
        pad_eigval: Padding value for eigenvalue calculation (default: 1)
    """
    # Set calculation settings globally for worker processes
    set_calculation_settings(pred_xc=pred_xc, gt_xc=gt_xc, calc_xc=calc_xc, basis=basis, pred_tol=pred_tol, gt_tol=gt_tol, pad_eigval=pad_eigval, filtering=filtering)
    
    # Enable profiling if requested
    if profile:
        enable_profiling()
    
    logger.info(f"Starting energy measurement for dataset: {dataset_name}")
    logger.info(f"Model: {model_name}, Prefix: {model_prefix}, Postfix: {model_postfix}")
    logger.info(f"DFT Settings:")
    logger.info(f"  Pred functional: {pred_xc.upper()}")
    logger.info(f"  GT functional:   {gt_xc.upper()}")
    logger.info(f"  Calc functional: {calc_xc.upper()}")
    logger.info(f"  Basis set:       {basis.upper()}")
    logger.info(f"Config: chunksize={chunksize}, maxtasksperchild={maxtasksperchild}, "
                f"workers={num_workers}, parallel={use_parallel}, profile={profile}")
    
    # Setup worker configuration
    cpu_count = mp.cpu_count()
    logger.info(f"System: {cpu_count} CPU cores available")
    
    max_cpu_cores = min(max_cpu_cores, cpu_count) if max_cpu_cores else cpu_count
    actual_workers = min(num_workers, max_cpu_cores)
    logger.info(f"Using: {actual_workers} workers (capped at {max_cpu_cores} cores)")
    
    # Initialize paths
    model_path_cls = ModelPath(
        dataset_name=dataset_name,
        model_name=model_name,
        model_prefix=model_prefix,
        model_postfix=model_postfix
    )
    
    # Ensure flat directories exist and gather file lists
    pred_flat_dir, pred_flat_names, pred_original_names = _ensure_flat_directory(model_path_cls.pred, "pred")
    gt_flat_dir, gt_flat_names, gt_original_names = _ensure_flat_directory(model_path_cls.gt, "gt")
    
    logger.info(f"Using prediction directory: {pred_flat_dir}")
    logger.info(f"Using ground-truth directory: {gt_flat_dir}")
    
    pred_list = [os.path.join(pred_flat_dir, f"{name}.pt") for name in pred_flat_names]
    gt_list = [os.path.join(gt_flat_dir, f"{name}.pt") for name in gt_flat_names]
    
    logger.info(f"Prediction files: {len(pred_list)} | Ground-truth files: {len(gt_list)}")
    
    if len(pred_list) != len(gt_list):
        raise ValueError(f"Prediction list ({len(pred_list)}) and GT list ({len(gt_list)}) have different lengths")
    if not pred_list:
        raise ValueError(f"No prediction files found in {pred_flat_dir}")
    
    # Include xc in calc directory names for differentiation
    # Always include XC functional in directory names for clarity
    calc_dir_name = f"_calc_{calc_xc}"
    gt_calc_dir_name = f"_gt_calc_{gt_xc}"
    
    logger.info(f"\nDirectory naming based on XC functionals:")
    logger.info(f"  calc directory:    {calc_dir_name} (using {calc_xc.upper()})")
    logger.info(f"  gt_calc directory: {gt_calc_dir_name} (using {gt_xc.upper()})")
    
    tol_str = f"{pred_tol:.0e}".replace("-", "n").replace("+", "")
    pad_eigval_str = f"{pad_eigval:.0e}".replace("-", "n").replace("+", "").replace(".", "p")
    adaptive_str = "_adaptive" if filtering else ""
    pred_processed_suffix = f"_processed_{pred_xc}{adaptive_str}_tol{tol_str}_pad{pad_eigval_str}"
    logger.info(f"  pred_processed:    {pred_processed_suffix} (using {pred_xc.upper()}, tol={pred_tol}, pad_eigval={pad_eigval})")
    
    calc_flat_dir, _, _ = _ensure_flat_directory(model_path_cls.custom_path(calc_dir_name), "calc")
    gt_calc_flat_dir, _, _ = _ensure_flat_directory(model_path_cls.custom_path(gt_calc_dir_name), "gt")
    logger.info(f"Using calc directory: {calc_flat_dir}")
    logger.info(f"Using gt_calc directory: {gt_calc_flat_dir}")
    if uma:
        omol_flat_dir, _, _ = _ensure_flat_directory(model_path_cls.custom_path("omol_calc"), "omol")
        logger.info(f"Using omol directory: {omol_flat_dir}")
    else:
        omol_flat_dir = None
    
    calc_list = [
        os.path.join(calc_flat_dir, f"{_rename_prefix(name, 'calc')}.pt")
        for name in gt_flat_names
    ]
    gt_calc_list = [
        os.path.join(gt_calc_flat_dir, f"{_rename_prefix(name, 'gt')}.pt")
        for name in gt_flat_names
    ]
    if uma:
        omol_list = [
            os.path.join(omol_flat_dir, f"{_rename_prefix(name, 'omol')}.pt")
            for name in gt_flat_names
        ]
    else:
        omol_list = [None] * len(pred_list)
    
    # Create processed directory (always include pred_xc for clarity)
    pred_processed_dir = f"{os.path.dirname(pred_list[0])}{pred_processed_suffix}"
    os.makedirs(pred_processed_dir, exist_ok=True)
    
    # Show file path structure (example from first file)
    logger.info("\n" + "="*80)
    logger.info("FILE PATH STRUCTURE:")
    logger.info("="*80)
    
    # Show directory structure
    base_dir = os.path.dirname(os.path.dirname(pred_list[0]))
    pred_processed_dir_name = os.path.basename(pred_processed_dir)
    pred_flat_dir_name = os.path.basename(pred_flat_dir)
    gt_flat_dir_name = os.path.basename(gt_flat_dir)
    calc_flat_dir_name = os.path.basename(calc_flat_dir)
    gt_calc_flat_dir_name = os.path.basename(gt_calc_flat_dir)
    omol_flat_dir_name = os.path.basename(omol_flat_dir) if uma else None
    logger.info(f"\nBase directory: {base_dir}/")
    logger.info(f"DFT Functionals: Pred={pred_xc.upper()}, GT={gt_xc.upper()}, Calc={calc_xc.upper()}, Basis={basis.upper()}")
    logger.info("")
    logger.info("├── pred/                    # Prediction results (Hamiltonian)")
    padding = ' ' * max(0, 20 - len(pred_processed_dir_name))
    logger.info(f"├── {pred_processed_dir_name}/{padding} # Pred + computed E/F/MO [CACHE, XC={pred_xc.upper()}]")
    logger.info("├── gt/                      # Ground truth data")
    logger.info(f"├── {gt_flat_dir_name}/           # Flattened ground truth files")
    padding_calc = ' ' * max(0, 20 - len(calc_dir_name))
    logger.info(f"├── {calc_dir_name}/{padding_calc} # Reference PySCF calculations [CACHE, XC={calc_xc.upper()}]")
    logger.info(f"├── {calc_flat_dir_name}/      # Flattened calc data targets")
    padding_gt = ' ' * max(0, 20 - len(gt_calc_dir_name))
    logger.info(f"├── {gt_calc_dir_name}/{padding_gt} # GT + computed E/F/MO [CACHE, XC={gt_xc.upper()}]")
    logger.info(f"├── {gt_calc_flat_dir_name}/   # Flattened GT calc data")
    if uma:
        logger.info("├── omol_calc/               # UMA calculations [CACHE]")
        logger.info(f"└── {omol_flat_dir_name}/          # Flattened UMA calculations [CACHE]")
    else:
        logger.info("└── omol_calc/               # UMA calculations [CACHE - optional]")
    
    # Show actual paths for first file
    logger.info(f"\nExample file paths (file 1/{len(pred_list)}):")
    logger.info(f"  pred:           {pred_list[0]}")
    logger.info(f"  pred_processed: {pred_list[0].replace(os.path.dirname(pred_list[0]), pred_processed_dir)}")
    logger.info(f"  gt:             {gt_list[0]}")
    logger.info(f"  calc:           {calc_list[0]}")
    logger.info(f"  gt_calc:        {gt_calc_list[0]}")
    if uma:
        logger.info(f"  omol_calc:      {omol_list[0]}")
    
    # Verify directory names in actual paths
    logger.info(f"\nVerifying XC functional in paths:")
    logger.info(f"  calc path contains '{calc_dir_name}': {calc_dir_name in calc_list[0]}")
    logger.info(f"  gt_calc path contains '{gt_calc_dir_name}': {gt_calc_dir_name in gt_calc_list[0]}")
    
    # Show cache status
    logger.info(f"\nCache status (first file):")
    logger.info(f"  pred_processed exists: {os.path.exists(pred_list[0].replace(os.path.dirname(pred_list[0]), pred_processed_dir))}")
    logger.info(f"  calc exists:           {os.path.exists(calc_list[0])}")
    logger.info(f"  gt_calc exists:        {os.path.exists(gt_calc_list[0])}")
    if uma:
        logger.info(f"  omol_calc exists:      {os.path.exists(omol_list[0])}")
    logger.info("="*80 + "\n")
    
    total_files = len(pred_list)
    selected_indices: List[int] = []
    custom_indices = False
    if file_indices is not None:
        selected_indices = _normalize_index_list(file_indices, total_files)
        custom_indices = True
    
    if reverse_order:
        if custom_indices:
            logger.warning("reverse_order is ignored when file_indices are provided.")
        else:
            for seq in [
                pred_list,
                gt_list,
                calc_list,
                omol_list,
                gt_calc_list,
                pred_flat_names,
                gt_flat_names,
                pred_original_names,
                gt_original_names
            ]:
                seq.reverse()
    
    # Default path uses fractional slicing; explicit indices completely override it.
    if not custom_indices:
        start_index = int(total_files * start_frac)
        end_index = int(total_files * end_frac)
        sliced_total_files = end_index - start_index
        selected_indices = list(range(start_index, end_index))
        logger.info(f"Total files: {total_files}, Processing: {sliced_total_files} "
                    f"(indices {start_index} to {end_index})")
    else:
        sliced_total_files = len(selected_indices)
        preview = ", ".join(str(i) for i in selected_indices[:10])
        if len(selected_indices) > 10:
            preview += ", ..."
        logger.info(f"Total files: {total_files}, Processing {sliced_total_files} custom indices: {preview}")
    
    if sliced_total_files <= 0:
        raise ValueError("No files selected for processing. Adjust start/end fractions or file_indices.")
    
    indices_for_dir = sorted(selected_indices) if custom_indices else None
    measure_dir_path, measure_dir_name = _prepare_measure_directory(
        model_path_cls.model_path, start_frac, end_frac, indices=indices_for_dir, tol=pred_tol, filtering=filtering
    )
    file_name = f"energy_force_{'uma_' if uma else ''}results_{dataset_name}_{model_name}.csv"
    output_path = os.path.join(measure_dir_path, file_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Measurement output directory: {measure_dir_path}")
    
    def _gather(seq):
        """Helper to pick elements following selected indices while preserving order"""
        return [seq[idx] for idx in selected_indices]
    
    sliced_pred_list = _gather(pred_list)
    sliced_gt_list = _gather(gt_list)
    sliced_calc_list = _gather(calc_list)
    sliced_omol_list = _gather(omol_list)
    sliced_gt_calc_list = _gather(gt_calc_list)
    sliced_pred_flat_names = _gather(pred_flat_names)
    sliced_gt_flat_names = _gather(gt_flat_names)
    sliced_pred_original_names = _gather(pred_original_names)
    sliced_gt_original_names = _gather(gt_original_names)
    
    file_tuples = []
    for idx_value, pred_path, gt_path, calc_path, omol_path, gt_calc_path, sliced_pred_flat, sliced_gt_flat, sliced_pred_orig, sliced_gt_orig in zip(
        selected_indices,
        sliced_pred_list,
        sliced_gt_list,
        sliced_calc_list,
        sliced_omol_list,
        sliced_gt_calc_list,
        sliced_pred_flat_names,
        sliced_gt_flat_names,
        sliced_pred_original_names,
        sliced_gt_original_names
    ):
        file_tuples.append(
            (
                idx_value,
                pred_path,
                gt_path,
                calc_path,
                omol_path,
                gt_calc_path,
                {
                    "pred_flat_name": sliced_pred_flat,
                    "gt_flat_name": sliced_gt_flat,
                    "pred_original_name": sliced_pred_orig,
                    "gt_original_name": sliced_gt_orig
                }
            )
        )
    
    index_to_pred_info = {
        idx: {
            "pred_path": pred_path,
            "pred_file_name": f"{meta.get('pred_flat_name')}.pt" if meta.get("pred_flat_name") else os.path.basename(pred_path)
        }
        for idx, pred_path, _, _, _, _, meta in file_tuples
    }
    
    # Process files
    if os.path.exists(output_path):
        logger.info(f"Loading existing results from: {output_path}")
        results_df = pd.read_csv(output_path)
        results_df = results_df.applymap(lambda x: x.item() if isinstance(x, (torch.Tensor, np.ndarray)) else x)
        
        if "file_index" not in results_df.columns:
            if len(results_df) != len(selected_indices):
                raise ValueError("Existing results file length does not match requested indices.")
            results_df["file_index"] = selected_indices
        results_df["file_index"] = results_df["file_index"].astype(int)
        
        mapped_pred_path = results_df["file_index"].map(lambda idx: index_to_pred_info.get(idx, {}).get("pred_path"))
        if "pred_path" in results_df.columns:
            results_df["pred_path"] = mapped_pred_path.where(mapped_pred_path.notna(), results_df["pred_path"])
        else:
            results_df["pred_path"] = mapped_pred_path
        
        results_df["pred_file_name"] = results_df["pred_path"].map(
            lambda path: os.path.basename(path) if isinstance(path, str) else None
        )
        
        results_df = results_df.sort_values(by="file_index").reset_index(drop=True)
        results_list = results_df.to_dict(orient="records")
    else:
        if use_parallel and actual_workers > 1:
            logger.info("Using parallel processing (no batching)")
            results_list = process_all_files(
                file_tuples, pred_xc, gt_xc, calc_xc, basis, 
                actual_workers, chunksize, maxtasksperchild, 
                output_path=output_path,
                pred_tol=pred_tol,
                gt_tol=gt_tol,
                pad_eigval=pad_eigval,
                filtering=filtering
            )
        else:
            logger.info("Using sequential processing")
            results_list = process_sequential_safe(file_tuples)
    
    # Generate output
    if not results_list:
        logger.info("No results generated. Check for errors in processing.")
        return None
    
    valid_results = [r for r in results_list if "error" not in r]
    error_count = len(results_list) - len(valid_results)
    
    if not valid_results:
        logger.info("No valid results generated. All files failed processing.")
        return None
    
    results = pd.DataFrame(valid_results)
    if "file_index" in results.columns:
        results["file_index"] = results["file_index"].astype(int)
        results = results.sort_values(by="file_index").reset_index(drop=True)
        mapped_pred_path = results["file_index"].map(lambda idx: index_to_pred_info.get(idx, {}).get("pred_path"))
        if "pred_path" in results.columns:
            results["pred_path"] = mapped_pred_path.where(mapped_pred_path.notna(), results["pred_path"])
        else:
            results["pred_path"] = mapped_pred_path
    if "pred_path" in results.columns:
        results["pred_file_name"] = results["pred_path"].map(
            lambda path: os.path.basename(path) if isinstance(path, str) else None
        )
    results.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")
    
    # Save calculation settings/parameters
    settings_path = output_path.replace(".csv", "_settings.json")
    calculation_settings = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "model_prefix": model_prefix,
        "model_postfix": model_postfix,
        "dft_functionals": {
            "pred": pred_xc,
            "gt": gt_xc,
            "calc_reference": calc_xc
        },
        "basis_set": basis,
        "grid_level": "default (3)",
        "grid_response": "default (True)",
        "pred_tol": pred_tol,
        "gt_tol": gt_tol,
        "pad_eigval": pad_eigval,
        "directories": {
            "measurement": measure_dir_name,
            "pred_processed": pred_processed_dir_name,
            "calc": calc_dir_name,
            "gt_calc": gt_calc_dir_name
        },
        "parallel_config": {
            "num_workers": actual_workers,
            "chunksize": chunksize,
            "maxtasksperchild": maxtasksperchild
        },
        "processing_stats": {
            "total_files_processed": len(valid_results),
            "failed_files": error_count,
            "start_frac": start_frac,
            "end_frac": end_frac,
            "selected_indices_count": sliced_total_files,
            "custom_indices_enabled": custom_indices,
            "custom_indices_values": selected_indices if custom_indices else None
        },
        "uma_included": uma,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(settings_path, 'w') as f:
        json.dump(calculation_settings, f, indent=2)
    logger.info(f"Calculation settings saved to: {settings_path}")
    
    # Print summary
    logger.info("\n=== Summary Statistics ===")
    logger.info(f"Successfully processed: {len(valid_results)}/{sliced_total_files} files")
    if error_count > 0:
        logger.info(f"Failed files: {error_count}")
    
    # Generate summaries
    energy_keys = [
        "energy_diff (pred-gt)",
        "energy_diff (pred-calc_energy)",
        "energy_diff (gt-calc_energy)",
        "energy_diff_per_atom (pred-gt)",
        "energy_diff_per_atom (pred-calc_energy)",
        "energy_diff_per_atom (gt-calc_energy)"
    ]
    force_keys = ["forces_diff l2 (pred-gt)", "forces_diff l2 (pred-calc_forces)", "forces_diff l2 (gt-calc_forces)"]
    ham_keys = ["hamiltonian_error (pred-gt)", "hamiltonian_error (pred-calc)", "hamiltonian_error (gt-calc)"]
    mo_keys = ["mo_energy_error (pred-gt)", "mo_energy_error (pred-calc)", "mo_energy_error (gt-calc)"]
    
    if uma:
        energy_keys.extend([
            "energy_diff (gt-omol_energy)",
            "energy_diff (calc-omol_energy)",
            "energy_diff_per_atom (gt-omol_energy)",
            "energy_diff_per_atom (calc-omol_energy)"
        ])
        force_keys.extend(["forces_diff l2 (gt-omol_forces)", "forces_diff l2 (calc-omol_forces)"])
    
    pandas_summary(results, output_path, "energy_diff_meV", energy_keys, "meV")
    pandas_summary(results, output_path, "energy_diff_mu_hartree", energy_keys, "mu_hartree")
    pandas_summary(results, output_path, "force_diff_meV_Ang", force_keys, "meV/Ang")
    pandas_summary(results, output_path, "force_diff_mu_hartree_Bohr", force_keys, "mu_hartree/Bohr")
    pandas_summary(results, output_path, "hamiltonian_error", ham_keys, "mu_hartree")
    pandas_summary(results, output_path, "mo_energy_error_meV", mo_keys, "meV")
    
    # Generate plots
    logger.info("\n=== Generating Plots ===")
    try:
        generate_plots(results, output_path, uma=uma)
        logger.info("Plots generated successfully")
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
        logger.error(traceback.format_exc())
    
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Measure energy and forces for DFT predictions")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--model_name", type=str, default="QHFlow", help="Model name")
    parser.add_argument("--model_prefix", type=str, default="", help="Model prefix")
    parser.add_argument("--model_postfix", type=str, default="", help="Model postfix")
    parser.add_argument("--pred_xc", type=str, default="b3lyp",
                       help="XC functional for Prediction energy/force calculations (default: b3lyp)")
    parser.add_argument("--gt_xc", type=str, default="b3lyp",
                       help="XC functional for GT energy/force calculations (default: b3lyp)")
    parser.add_argument("--calc_xc", type=str, default="b3lyp",
                       help="XC functional for Calc reference (default: b3lyp)")
    parser.add_argument("--basis", type=str, default="def2svp",
                       help="Basis set (default: def2svp)")
    parser.add_argument("--chunksize", type=int, default=1, 
                       help="Number of tasks to submit to each worker at once")
    parser.add_argument("--maxtasksperchild", type=int, default=1000, 
                       help="Maximum tasks per worker before restart")
    parser.add_argument("--num_workers", type=int, default=16,
                       help="Number of parallel workers")
    parser.add_argument("--max_cpu_cores", type=int, default=None,
                       help="Maximum number of CPU cores to use")
    parser.add_argument("--no_parallel", action="store_true",
                       help="Disable parallel processing")
    parser.add_argument("--start_frac", type=float, default=0.0,
                       help="Fraction of files to start from")
    parser.add_argument("--end_frac", type=float, default=1.0,
                       help="Fraction of files to end at")
    parser.add_argument("--reverse_order", action="store_true",
                       help="Reverse the order of files")
    parser.add_argument("--file_indices", type=str, default=None,
                       help="Comma-separated list of specific file indices to process (overrides fractions)")
    parser.add_argument("--uma", action="store_true", 
                       help="Compute UMA results")
    parser.add_argument("--profile", action="store_true",
                       help="Enable detailed profiling (shows timing for each step)")
    parser.add_argument("--pred_tol", type=float, default=1e-3,
                       help="Tolerance for prediction density matrix calculation (default: 1e-3)")
    parser.add_argument("--gt_tol", type=float, default=1e-8,
                       help="Tolerance for ground truth density matrix calculation (default: 1e-8)")
    parser.add_argument("--pad_eigval", type=float, default=1,
                       help="Padding value for eigenvalue calculation (default: 1)")
    parser.add_argument("--filtering", action="store_true",
                       help="Enable filtering for unstable MO energy calculations (adds 'adaptive' to directory/file names)")
    args = parser.parse_args()
    
    main(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        model_prefix=args.model_prefix,
        model_postfix=args.model_postfix,
        pred_xc=args.pred_xc,
        gt_xc=args.gt_xc,
        calc_xc=args.calc_xc,
        basis=args.basis,
        chunksize=args.chunksize,
        maxtasksperchild=args.maxtasksperchild,
        use_parallel=not args.no_parallel,
        num_workers=args.num_workers,
        max_cpu_cores=args.max_cpu_cores,
        start_frac=args.start_frac,
        end_frac=args.end_frac,
        reverse_order=args.reverse_order,
        file_indices=args.file_indices,
        uma=args.uma,
        profile=args.profile,
        pred_tol=args.pred_tol,
        gt_tol=args.gt_tol,
        pad_eigval=args.pad_eigval,
        filtering=args.filtering
    )
