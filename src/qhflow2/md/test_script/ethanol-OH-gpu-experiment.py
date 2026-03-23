import sys
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("/root/limlab01/kaistai/25DFT/QHFlow/src")
import md.scflow_calculator_gpu
from qhflow2.dft_process.dft_process_utils import *

sample_ethanol = [
    (6,  [-0.28630036, -0.47787276,  0.03603168]),  # C
    (6,  [-0.92012881,  0.96245698,  0.23729976]),  # C
    (8,  [ 1.10116376, -0.40535771, -0.20377836]),  # O
    (1,  [-0.49412140, -1.15617614,  0.94202625]),  # H
    (1,  [-0.80417700, -0.91072034, -0.80315132]),  # H
    (1,  [-1.91334152,  0.70271240,  0.63912332]),  # H
    (1,  [-0.36050056,  1.60555901,  0.92692388]),  # H
    (1,  [-0.93001164,  1.37368891, -0.73671183]),  # H
    (1,  [ 1.39790227, -0.95652028, -0.99060288]),  # H
]
sample_ethanol_atoms = [a[0] for a in sample_ethanol]
sample_ethanol_coords = [a[1] for a in sample_ethanol]
 
"""
Compute O–H bond dissociation curves for ethanol using trained models (eSEN, DimeNet, SchNet)
and optionally DFT (psi4). Outputs energies (relative to each curve minimum) and a plot.

This version is adapted for Jupyter notebook:
- No argparse / __main__
- You set input paths and options as variables below.
CUDA_VISIBLE_DEVICES=0 python -m md.test_script.ethanol-OH-gpu-experiment.py

"""
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import torch
import time
import json

# -----------------------------
# User configuration (edit here)
# -----------------------------
# Path to RMD17 ethanol HDF5 file
h5_path = Path("/home/yoonho/rmd_ham_data/rmd-ethanol/data.h5")

# Output directory for .npy and plots (with timestamp)
base_output_dir = Path("/root/25DFT/QHFlow/src/md/test_script")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = base_output_dir / f"outputs_ethanol_OH_{timestamp}"
data_dir = out_dir / "data"
plots_dir = out_dir / "plots"

# Distance range (Å) and number of points
dist_min = 0.01
dist_max = 10.0
# dist_max = 5.3
n_points = 100

# Models to use
# Format: Each model can be:
#   - A string: "Qhflow2-scf" (uses default options)
#   - A tuple: ("Qhflow2-scf", {...options...}) (uses provided options)
#   - A dict: {"name": "Qhflow2-scf", "options": {...}} (uses provided options)
#
# Example:
# models = [
#     ("Qhflow2-scf", {
#         "checkpoint_path": "/path/to/checkpoint.ckpt",
#         "max_cycle": 200,
#         "basis": "def2-SVP",
#         ...
#     }),
#     ("uks-dft", {
#         "max_cycle": 300,
#         "basis": "def2-TZVP",
#     }),
# ]
models = [
    # ("Qhflow2-uks-spin2", {
    #     "checkpoint_path": "/root/25DFT/QHFlow/src/outputs/ethanol/QHFlow_so2_v5_1_middle_b10-ethanol/checkpoints/weights-epoch=79-val_loss=0.0000000.ckpt",
    #     "basis": "def2-SVP",
    #     "functional": "pbe",
    #     "units": "ang",
    #     "model_length_unit": "bohr",
    #     "scf": False,
    #     "uks": True,
    #     "spin": 2,
    # }),
    # ("Qhflow2-uks-scf", {
    #     "checkpoint_path": "/root/25DFT/QHFlow/src/outputs/ethanol/QHFlow_so2_v5_1_middle_b10-ethanol/checkpoints/weights-epoch=79-val_loss=0.0000000.ckpt",
    #     "basis": "def2-SVP",
    #     "functional": "pbe",
    #     "units": "ang",
    #     "model_length_unit": "bohr",
    #     "scf": True,
    #     "uks": True,
    #     "spin": 0,
    #     "max_cycle": 100,
    # }),
    # ("Qhflow2", {
    #     "checkpoint_path": "/root/25DFT/QHFlow/src/outputs/ethanol/QHFlow_so2_v5_1_middle_b10-ethanol/checkpoints/weights-epoch=79-val_loss=0.0000000.ckpt",
    #     "basis": "def2-SVP",
    #     "functional": "pbe",
    #     "units": "ang",
    #     "model_length_unit": "bohr",
    #     "scf": False,
    #     "uks": False,
    #     "spin": 0,
    # }),
    # ("DFT-100", {
    #     "basis": "def2-SVP",
    #     "functional": "pbe",
    #     "max_cycle": 100,
    #     "energy_unit": "ev",
    #     "force_unit": "ev/ang",
    #     "scf": False,
    #     "uks": False,
    # }),
    # ("DFT-100-level-shift-0_5", {
    #     "basis": "def2-SVP",
    #     "functional": "pbe",
    #     "max_cycle": 100,
    #     "energy_unit": "ev",
    #     "force_unit": "ev/ang",
    #     "scf": False,
    #     "uks": True,
    #     # "spin": 2,
    #     "level_shift": 0.5,
    # }),
    ("UKS-DFT-200-spin2", {
        "basis": "def2-SVP",
        "functional": "pbe",
        "max_cycle": 200,
        "energy_unit": "ev",
        "force_unit": "ev/ang",
        "scf": False,
        "uks": True,
        "spin": 2,
    }),
    ("UKS-DFT-200", {
        "basis": "def2-SVP",
        "functional": "pbe",
        "max_cycle": 200,
        "energy_unit": "ev",
        "force_unit": "ev/ang",
        "scf": False,
        "uks": False,
    }),
]

# Default options for each model type (used when options are not provided)
default_model_options = {
    "Qhflow2": {
        "basis": "def2-SVP",
        "functional": "pbe",
        "units": "ang",
        "model_length_unit": "bohr",
        "mf_init_functional": "pbe,pbe",
        "init_density_fit": False,
        "init_gpu4pyscf": False,
        "density_fit": True,
        "filtering": False,
        "gt_tol": 1e-8,
        "pred_tol": 1e-3,
        "pad_eigval": None,
        "max_radius": 1000,
        "max_num_neighbors": 200,
        "ode_steps": 1,
        "device": "cuda",
        "scf": False,
        "spin": 0,
    },
    "uks-dft": {
        "basis": "def2-SVP",
        "functional": "pbe",
        "max_cycle": 50,
        "energy_unit": "ev",
        "force_unit": "ev/ang",
        "spin": 0,
    },
    "dft": {
        "basis": "def2-SVP",
        "functional": "pbe",
        "max_cycle": 50,
        "density_fit": True,
        "energy_unit": "ev",
        "force_unit": "ev/ang",
        "spin": 0,
    },
}

mf_dict = {}
# Whether to compute DFT with psi4
compute_dft = False  # True / False

# Per-model GPU mapping, e.g. {"Qhflow2-scf": "4", "uks-dft": "5"}
gpu_map = None  # or dict, e.g. {"Qhflow2-scf": "4", "uks-dft": "5"}

# Plot prefix
plot_prefix = "ethanol_dissociation_curves_d2"
# -----------------------------


# Ensure local imports work (adapted to your project layout)
# NOTE: In a notebook, __file__ is not defined; use project root directly.
ROOT = Path("/root/25DFT/QHFlow/src").resolve()
sys.path.insert(0, str(ROOT))

# from mdsim.md.ase_utils import OCPCalculator, ESENCalculator  # type: ignore
from ase import Atoms


def load_base_atoms(h5_path: Path) -> Atoms:
    """Load atomic_numbers and positions from sample_0 of RMD17 HDF5."""
    with h5py.File(h5_path, "r") as f:
        idx = 0
        grp = f[f"sample_{idx}"]
        Z = grp["atomic_numbers"][:].astype(int)
        pos = grp["positions"][:]
    atoms = Atoms(numbers=Z, positions=pos)
    return atoms


def find_oh_pair(atoms: Atoms):
    """Return indices (iO, iH) where O is atomic number 8 and H is closest hydrogen to that O."""
    O_indices = [i for i, z in enumerate(atoms.get_atomic_numbers()) if z == 8]
    H_indices = [i for i, z in enumerate(atoms.get_atomic_numbers()) if z == 1]
    assert O_indices and H_indices, "No O/H atoms found"
    iO = O_indices[0]
    distances = [atoms.get_distance(iO, h) for h in H_indices]
    iH = H_indices[int(np.argmin(distances))]
    return iO, iH


def set_oh_distance(atoms: Atoms, iO: int, iH: int, target: float):
    """Move H along the O->H vector to achieve target distance (Å)."""
    atoms = atoms.copy()
    vec = atoms.positions[iH] - atoms.positions[iO]
    curr = np.linalg.norm(vec)
    if curr < 1e-8:
        return atoms
    delta = (vec / curr) * (target - curr)
    atoms.positions[iH] += delta
    return atoms


def run_curve_single(distances, calculator, atoms: Atoms, iO: int, iH: int, scf: bool = False, uks: bool=False, use_qhflow2: bool=False, spin=0):
    vals = []
    mf_list = []
    time_list = []
    dm_list = []
    orb_energy_list = []
    traj_coords_list = []
    for idx,d in enumerate(distances):
        start_time = time.time()
        geom = set_oh_distance(atoms, iO, iH, d)
        if use_qhflow2:
            if uks:
                calculator.calculate_uks(geom, properties=["energy"],system_changes=[], scf=scf, spin=spin)
            else:
                calculator.calculate(geom, properties=["energy"],system_changes=[], scf=scf, spin=spin)
        else:
            calculator.calculate(geom, properties=["energy"],system_changes=[])
        vals.append(calculator.results["energy"])
        print(idx, d, vals[-1])
        mf_list.append([d, calculator.mf])
        if hasattr(calculator.mf, "cycles"):
            print(calculator.mf.cycles)
        time_list.append(time.time() - start_time)
        pred_dm0 = calculator.pred_dm0
        if hasattr(pred_dm0, 'get'):  # cupy array
            pred_dm0 = pred_dm0.get()
        dm_list.append(pred_dm0)
        
        mo_energy = calculator.mf.mo_energy
        if hasattr(mo_energy, 'get'):  # cupy array
            mo_energy = mo_energy.get()
        orb_energy_list.append(mo_energy)
        traj_coords_list.append(np.array(geom.get_positions(), dtype=float))

    return np.array(vals), mf_list, time_list, dm_list, orb_energy_list, traj_coords_list

def shift_to_min(arr):
    return arr - np.nanmin(arr)


# ============================
# Main execution in notebook
# ============================

# Prepare output directories
out_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {out_dir}")
print(f"Data directory: {data_dir}")
print(f"Plots directory: {plots_dir}")

atoms = Atoms(numbers=sample_ethanol_atoms, positions=sample_ethanol_coords, cell=np.eye(3), pbc=False)
iO, iH = find_oh_pair(atoms)
print(f"Using O index {iO}, H index {iH}, initial d={atoms.get_distance(iO, iH):.3f} Å")

distances = np.linspace(dist_min, dist_max, n_points)

# Save original CUDA_VISIBLE_DEVICES
orig_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)


def set_gpu(label):
    if gpu_map is None:
        # restore original if not using mapping
        if orig_cuda_env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda_env
        return

    dev = gpu_map.get(label, None)
    if dev is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
    else:
        # restore original if not using mapping
        if orig_cuda_env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda_env


energies = {}
time_dict = {}
dm_dict = {}
orb_energy_dict = {}
model_options_dict = {}  # Store options used for each model

# Helper function to parse model input
def parse_model_input(model_input):
    """Parse model input which can be string, tuple, or dict.
    
    Returns:
        (model_name, options_dict)
    """
    if isinstance(model_input, str):
        # Just model name, use default options
        return model_input, {}
    elif isinstance(model_input, (tuple, list)) and len(model_input) == 2:
        # (name, options) tuple
        return model_input[0], model_input[1]
    elif isinstance(model_input, dict):
        # {"name": "...", "options": {...}} dict
        return model_input.get("name"), model_input.get("options", {})
    else:
        raise ValueError(f"Invalid model input format: {model_input}")

# Build and run per-model sequentially on mapped GPU
for model_input in models:
    # Parse model input to get name and options
    mdl, user_opts = parse_model_input(model_input)
    set_gpu(mdl)
    
    # Determine model type and get default options
    model_type = None
    mdl_lower = mdl.lower()
    if "qhflow2" in mdl_lower:
        model_type = "Qhflow2"
    elif "uks" in mdl_lower and "dft" in mdl_lower:
        model_type = "uks-dft"
    elif "uks" in mdl_lower:
        model_type = "uks-dft"
    elif "dft" in mdl_lower:
        model_type = "dft"
    else:
        print(f"Unknown model: {mdl}, skipping.")
        continue
    
    # Merge default options with user-provided options (user options take precedence)
    default_opts = default_model_options.get(model_type, {})
    opts = {**default_opts, **user_opts}
    
    # Store options for this model
    model_options_dict[mdl] = {
        "model_type": model_type,
        "options": opts.copy(),
    }
    
    if model_type == "Qhflow2":
        # SCFlowRKSCalculator
        calc = md.scflow_calculator_gpu.SCFlowRKSCalculator(
            device=opts.get("device", "cuda"),
            basis=opts.get("basis", "def2-SVP"),
            functional=opts.get("functional", "pbe"),
            energy_unit=opts.get("energy_unit", "ev"),
            force_unit=opts.get("force_unit", "ev/ang"),
            max_cycle=opts.get("max_cycle", 50),
            level_shift=opts.get("level_shift", None),
        )
        # Check if checkpoint_path is provided
        if "checkpoint_path" not in opts:
            raise ValueError(f"checkpoint_path is required for Qhflow2 model: {mdl}")
        
        calc.set_model(
            opts["checkpoint_path"],
            units=opts.get("units", "ang"),
            model_length_unit=opts.get("model_length_unit", "bohr"),
            mf_init_functional=opts.get("mf_init_functional", "pbe,pbe"),
            init_density_fit=opts.get("init_density_fit", False),
            init_gpu4pyscf=opts.get("init_gpu4pyscf", False),
            density_fit=opts.get("density_fit", True),
            filtering=opts.get("filtering", False),
            gt_tol=opts.get("gt_tol", 1e-8),
            pred_tol=opts.get("pred_tol", 1e-3),
            pad_eigval=opts.get("pad_eigval", None),
            max_radius=opts.get("max_radius", 1000),
            max_num_neighbors=opts.get("max_num_neighbors", 200),
            ode_steps=opts.get("ode_steps", 1),
        )
    elif model_type == "uks-dft":
        # UKSCalculator
        calc = md.scflow_calculator_gpu.UKSCalculator(
            basis=opts.get("basis", "def2-SVP"),
            functional=opts.get("functional", "pbe"),
            max_cycle=opts.get("max_cycle", 50),
            density_fit=opts.get("density_fit", True),
            energy_unit=opts.get("energy_unit", "ev"),
            force_unit=opts.get("force_unit", "ev/ang"),
        )
    elif model_type == "dft":
        # RKSCalculator
        calc = md.scflow_calculator_gpu.RKSCalculator(
            basis=opts.get("basis", "def2-SVP"),
            functional=opts.get("functional", "pbe"),
            max_cycle=opts.get("max_cycle", 50),
            density_fit=opts.get("density_fit", True),
            energy_unit=opts.get("energy_unit", "ev"),
            force_unit=opts.get("force_unit", "ev/ang"),
            level_shift=opts.get("level_shift", None),
        )

    scf = True if ("scf" in mdl.lower() or opts.get("scf", False)) else False
    uks = True if ("uks" in mdl.lower() or opts.get("uks", False)) else False
    use_qhflow2 = True if ("qhflow2" in mdl.lower() or opts.get("qhflow2", False)) else False
    spin = opts.get("spin", 0)
    
    print(f"Running model: {mdl}")
    try:
        energy_list, mf_list, time_list, dm_list, orb_energy_list, traj_coords_list = run_curve_single(
            distances, calc, atoms, iO, iH, scf, uks, use_qhflow2, spin
        )
        energies[mdl] = energy_list
        mf_dict[mdl] = mf_list
        time_dict[mdl] = time_list
        dm_dict[mdl] = dm_list
        orb_energy_dict[mdl] = orb_energy_list
        
        # Create model-specific directories
        # Sanitize model name for filesystem (replace special characters)
        safe_model_name = mdl.replace("/", "_").replace("\\", "_").replace(":", "_")
        model_data_dir = data_dir / safe_model_name
        model_plots_dir = plots_dir / safe_model_name
        model_data_dir.mkdir(parents=True, exist_ok=True)
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results immediately after each model completes
        print(f"Saving results for model: {mdl}")
        
        # Save energy data
        np.save(model_data_dir / f"ethanol_{safe_model_name}_energies.npy", energy_list)
        
        # Save density matrix data
        np.save(model_data_dir / f"ethanol_{safe_model_name}_dm.npy", np.array(dm_list, dtype=object))
        
        # Save orbital energy data
        np.save(model_data_dir / f"ethanol_{safe_model_name}_orb_energy.npy", np.array(orb_energy_list, dtype=object))

        # Save trajectory coordinates (all distances)
        np.save(model_data_dir / f"ethanol_{safe_model_name}_traj_coords.npy", np.array(traj_coords_list, dtype=float))
        
        # Save atomic numbers (per model for convenience)
        np.save(model_data_dir / f"ethanol_{safe_model_name}_atomic_numbers.npy", np.array(atoms.get_atomic_numbers(), dtype=int))
        
        # Save plots
        plt.figure(figsize=(7, 5))
        rel = shift_to_min(energy_list)
        plt.plot(distances, rel, marker="o", ms=3, label=mdl)
        plt.xlabel("O-H distance (Å)")
        plt.ylabel("Energy (eV, relative to min)")
        plt.legend()
        plt.tight_layout()
        out_plot = model_plots_dir / f"{plot_prefix}_{safe_model_name}.png"
        plt.savefig(out_plot, dpi=200)
        plt.close()
        print(f"Saved plot to {out_plot}")
        
        plt.figure(figsize=(7, 5))
        rel = shift_to_min(energy_list)
        plt.plot(distances[2:], rel[2:], marker="o", ms=3, label=mdl)
        plt.xlabel("O-H distance (Å)")
        plt.ylabel("Energy (eV, relative to min)")
        plt.legend()
        plt.tight_layout()
        out_plot = model_plots_dir / f"{plot_prefix}_{safe_model_name}_exclude.png"
        plt.savefig(out_plot, dpi=200)
        plt.close()
        print(f"Saved plot to {out_plot}")
        
        # Save model-specific options
        model_options_file = model_data_dir / "model_options.json"
        with open(model_options_file, "w") as f:
            model_json_options = {
                "model_name": mdl,
                "model_type": model_type,
                "options": {}
            }
            for key, value in opts.items():
                if isinstance(value, (np.integer, np.floating)):
                    model_json_options["options"][key] = value.item()
                elif isinstance(value, np.ndarray):
                    model_json_options["options"][key] = value.tolist()
                elif value is None:
                    model_json_options["options"][key] = None
                elif isinstance(value, (str, int, float, bool)):
                    model_json_options["options"][key] = value
                else:
                    model_json_options["options"][key] = str(value)
            json.dump(model_json_options, f, indent=2)
        print(f"Saved model options to {model_options_file}")
        
        # Save cycles data if available (for DFT and Qhflow2 models with SCF)
        if mf_list:
            cycles_distances = []
            cycles_values = []
            for d, mf in mf_list:
                if hasattr(mf, "cycles"):
                    cycles_distances.append(d)
                    cycles_values.append(mf.cycles)
            
            if cycles_values:
                # Match distances for cycles and energies
                energy_matched = []
                cycles_matched = []
                dist_matched = []
                for i, d in enumerate(cycles_distances):
                    idx = np.argmin(np.abs(distances - d))
                    energy_matched.append(energy_list[idx])
                    cycles_matched.append(cycles_values[i])
                    dist_matched.append(d)
                
                if energy_matched:
                    # Save cycles data
                    np.save(model_data_dir / f"{plot_prefix}_{safe_model_name}_distances.npy", np.array(dist_matched))
                    np.save(model_data_dir / f"{plot_prefix}_{safe_model_name}_energies_matched.npy", np.array(energy_matched))
                    np.save(model_data_dir / f"{plot_prefix}_{safe_model_name}_cycles.npy", np.array(cycles_matched))
                    
                    # Create energy and cycles plot
                    fig, ax1 = plt.subplots(figsize=(7, 5))
                    rel_energy = shift_to_min(np.array(energy_matched))
                    color1 = 'tab:blue'
                    ax1.set_xlabel("O-H distance (Å)")
                    ax1.set_ylabel("Energy (eV, relative to min)", color=color1)
                    line1 = ax1.plot(dist_matched, rel_energy, marker="o", ms=3, 
                                    label=f"{mdl} energy", color=color1)
                    ax1.tick_params(axis='y', labelcolor=color1)
                    ax1.grid(True, alpha=0.3)
                    
                    ax2 = ax1.twinx()
                    color2 = 'tab:red'
                    ax2.set_ylabel("SCF cycles", color=color2)
                    line2 = ax2.plot(dist_matched, cycles_matched, marker="s", ms=3, 
                                     label=f"{mdl} cycles", color=color2, linestyle='--')
                    ax2.tick_params(axis='y', labelcolor=color2)
                    
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax1.legend(lines, labels, loc=7, frameon=False)
                    plt.title(f"Energy and SCF Cycles vs O-H Distance ({mdl})")
                    plt.tight_layout()
                    cycles_plot = model_plots_dir / f"{plot_prefix}_{safe_model_name}_energy_cycles.png"
                    plt.savefig(cycles_plot, dpi=200)
                    plt.close()
                    print(f"Saved energy and cycles plot to {cycles_plot}")
                    
                    # Exclude first 2 points version
                    fig, ax1 = plt.subplots(figsize=(7, 5))
                    rel_energy = shift_to_min(np.array(energy_matched))
                    color1 = 'tab:blue'
                    ax1.set_xlabel("O-H distance (Å)")
                    ax1.set_ylabel("Energy (eV, relative to min)", color=color1)
                    line1 = ax1.plot(dist_matched[2:], rel_energy[2:], marker="o", ms=3, 
                                    label=f"{mdl} energy", color=color1)
                    ax1.tick_params(axis='y', labelcolor=color1)
                    ax1.grid(True, alpha=0.3)
                    
                    ax2 = ax1.twinx()
                    color2 = 'tab:red'
                    ax2.set_ylabel("SCF cycles", color=color2)
                    line2 = ax2.plot(dist_matched[2:], cycles_matched[2:], marker="s", ms=3, 
                                     label=f"{mdl} cycles", color=color2, linestyle='--')
                    ax2.tick_params(axis='y', labelcolor=color2)
                    
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax1.legend(lines, labels, loc=7, frameon=False)
                    plt.title(f"Energy and SCF Cycles vs O-H Distance ({mdl})")
                    plt.tight_layout()
                    cycles_plot = model_plots_dir / f"{plot_prefix}_{safe_model_name}_energy_cycles_exclude.png"
                    plt.savefig(cycles_plot, dpi=200)
                    plt.close()
                    print(f"Saved energy and cycles plot to {cycles_plot}")
        
        # Update global options file (for all models)
        options_file = data_dir / "model_options.json"
        experiment_settings = {
            "dist_min": dist_min,
            "dist_max": dist_max,
            "n_points": n_points,
            "plot_prefix": plot_prefix,
            "timestamp": timestamp,
            "models": [str(m) if isinstance(m, str) else m[0] if isinstance(m, (tuple, list)) else m.get("name", str(m)) for m in models],
        }
        with open(options_file, "w") as f:
            json_options = {
                "experiment_settings": experiment_settings,
                "model_options": {}
            }
            for model_name, model_data in model_options_dict.items():
                json_options["model_options"][model_name] = {
                    "model_type": model_data["model_type"],
                    "options": {}
                }
                for key, value in model_data["options"].items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_options["model_options"][model_name]["options"][key] = value.item()
                    elif isinstance(value, np.ndarray):
                        json_options["model_options"][model_name]["options"][key] = value.tolist()
                    elif value is None:
                        json_options["model_options"][model_name]["options"][key] = None
                    elif isinstance(value, (str, int, float, bool)):
                        json_options["model_options"][model_name]["options"][key] = value
                    else:
                        json_options["model_options"][model_name]["options"][key] = str(value)
            json.dump(json_options, f, indent=2)
        print(f"Updated global model options file: {options_file}")
        
        print(f"Model {mdl} completed and saved successfully")
        
    except Exception as e:
        print(f"Error running model {mdl}: {e}")
        print(f"Previous models' results are already saved.")
        raise
    
    del calc
    torch.cuda.empty_cache()

# restore original env
if orig_cuda_env is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda_env

# All models have been processed and saved individually above
# Now create combined plots for all completed models
print("\nCreating combined plots for all completed models...")

# Combined plot
plt.figure(figsize=(7, 5))
for name, arr in energies.items():
    rel = shift_to_min(arr)
    plt.plot(distances, rel, marker="o", ms=3, label=name)
plt.xlabel("O-H distance (Å)")
plt.ylabel("Energy (eV, relative to min)")
plt.legend()
plt.tight_layout()
combined = plots_dir / f"{plot_prefix}_all.png"
plt.savefig(combined, dpi=200)
plt.show()
print(f"Saved combined plot to {combined}")

 
plt.figure(figsize=(7, 5))
for name, arr in energies.items():
    rel = shift_to_min(arr)
    plt.plot(distances[2:], rel[2:], marker="o", ms=3, label=name)
plt.xlabel("O-H distance (Å)")
plt.ylabel("Energy (eV, relative to min)")
plt.legend()
plt.tight_layout()
combined = plots_dir / f"{plot_prefix}_all_exclude.png"
plt.savefig(combined, dpi=200)
plt.show()
print(f"Saved combined plot to {combined}")


for name in mf_dict.keys():
    if "dft" in name.lower():
        mf_list = mf_dict[name]
        cycles_distances = []
        cycles_values = []
        for d, mf in mf_list:
            if hasattr(mf, "cycles"):
                cycles_distances.append(d)
                cycles_values.append(mf.cycles)
        
        if cycles_values and name in energies:
            # Get energy values for the same distances
            energy_arr = energies[name]
            energy_distances = distances
            
            # Match distances for cycles and energies
            energy_matched = []
            cycles_matched = []
            dist_matched = []
            for i, d in enumerate(cycles_distances):
                # Find closest distance in energy_distances
                idx = np.argmin(np.abs(energy_distances - d))
                energy_matched.append(energy_arr[idx])
                cycles_matched.append(cycles_values[i])
                dist_matched.append(d)
            
            if energy_matched:
                fig, ax1 = plt.subplots(figsize=(7, 5))
                
                # Left y-axis: Energy
                rel_energy = shift_to_min(np.array(energy_matched))
                color1 = 'tab:blue'
                ax1.set_xlabel("O-H distance (Å)")
                ax1.set_ylabel("Energy (eV, relative to min)", color=color1)
                line1 = ax1.plot(dist_matched, rel_energy, marker="o", ms=3, 
                                label=f"{name} energy", color=color1)
                ax1.tick_params(axis='y', labelcolor=color1)
                ax1.grid(True, alpha=0.3)
                
                # Right y-axis: SCF cycles
                ax2 = ax1.twinx()
                color2 = 'tab:red'
                ax2.set_ylabel("SCF cycles", color=color2)
                line2 = ax2.plot(dist_matched, cycles_matched, marker="s", ms=3, 
                                 label=f"{name} cycles", color=color2, linestyle='--')
                ax2.tick_params(axis='y', labelcolor=color2)
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc=7, frameon=False)
                
                plt.title(f"Energy and SCF Cycles vs O-H Distance ({name})")
                plt.tight_layout()
                cycles_plot = plots_dir / f"{plot_prefix}_{name}_energy_cycles.png"
                plt.savefig(cycles_plot, dpi=200)
                plt.show()
                print(f"Saved energy and cycles plot to {cycles_plot}")
                
                # Save original data as npy files
                np.save(data_dir / f"{plot_prefix}_{name}_distances.npy", np.array(dist_matched))
                np.save(data_dir / f"{plot_prefix}_{name}_energies_matched.npy", np.array(energy_matched))
                np.save(data_dir / f"{plot_prefix}_{name}_cycles.npy", np.array(cycles_matched))
                print(f"Saved data files:")
                print(f"  - {data_dir / f'{plot_prefix}_{name}_distances.npy'}")
                print(f"  - {data_dir / f'{plot_prefix}_{name}_energies_matched.npy'}")
                print(f"  - {data_dir / f'{plot_prefix}_{name}_cycles.npy'}")
    
    # Plot energy and max cycles for DFT models (dual y-axis) - exclude first 2 points
    for name in mf_dict.keys():
        if "dft" in name.lower():
            mf_list = mf_dict[name]
            cycles_distances = []
            cycles_values = []
            for d, mf in mf_list:
                if hasattr(mf, "cycles"):
                    cycles_distances.append(d)
                    cycles_values.append(mf.cycles)
            
            if cycles_values and name in energies:
                # Get energy values for the same distances
                energy_arr = energies[name]
                energy_distances = distances
                
                # Match distances for cycles and energies
                energy_matched = []
                cycles_matched = []
                dist_matched = []
                for i, d in enumerate(cycles_distances):
                    # Find closest distance in energy_distances
                    idx = np.argmin(np.abs(energy_distances - d))
                    energy_matched.append(energy_arr[idx])
                    cycles_matched.append(cycles_values[i])
                    dist_matched.append(d)
                
                if energy_matched:
                    fig, ax1 = plt.subplots(figsize=(7, 5))
                    
                    # Left y-axis: Energy
                    rel_energy = shift_to_min(np.array(energy_matched))
                    color1 = 'tab:blue'
                    ax1.set_xlabel("O-H distance (Å)")
                    ax1.set_ylabel("Energy (eV, relative to min)", color=color1)
                    line1 = ax1.plot(dist_matched[2:], rel_energy[2:], marker="o", ms=3, 
                                    label=f"{name} energy", color=color1)
                    ax1.tick_params(axis='y', labelcolor=color1)
                    ax1.grid(True, alpha=0.3)
                    
                    # Right y-axis: SCF cycles
                    ax2 = ax1.twinx()
                    color2 = 'tab:red'
                    ax2.set_ylabel("SCF cycles", color=color2)
                    line2 = ax2.plot(dist_matched[2:], cycles_matched[2:], marker="s", ms=3, 
                                     label=f"{name} cycles", color=color2, linestyle='--')
                    ax2.tick_params(axis='y', labelcolor=color2)
                    
                    # Combine legends
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax1.legend(lines, labels, loc=7, frameon=False)
                    
                    plt.title(f"Energy and SCF Cycles vs O-H Distance ({name})")
                    plt.tight_layout()
                    cycles_plot = plots_dir / f"{plot_prefix}_{name}_energy_cycles_exclude.png"
                    plt.savefig(cycles_plot, dpi=200)
                    plt.show()
                    print(f"Saved energy and cycles plot to {cycles_plot}")