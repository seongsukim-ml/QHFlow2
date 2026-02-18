import sys
sys.path.append("/root/limlab01/kaistai/25DFT/QHFlow/src")
# GPU4PYSCF
from md.scflow_calculator_gpu import SCFlowRKSCalculator, RKSCalculator
import os

import random
import torch
import numpy as np
import subprocess
from pathlib import Path
import json
import yaml

from ase import Atoms, units
from ase.io import Trajectory
from md.ase_utils import Simulator, load_config
from md.rmd17_dataset import RMD17_DFT
import md.integrator as md_integrator
import time
import argparse

def data_to_atoms(data):
    numbers = data.atomic_numbers
    if isinstance(numbers, torch.Tensor):
        numbers = numbers.cpu().numpy()
    positions = data.pos
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    cell = data.cell.squeeze()
    if isinstance(cell, torch.Tensor):
        cell = cell.cpu().numpy()
    atoms = Atoms(numbers=numbers, 
                  positions=positions, 
                  cell=cell,
                  pbc=[False, False, False])
    return atoms

def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def runcmd(cmd_list):
    return subprocess.run(cmd_list, universal_newlines=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
def eval_and_init(config):
    # load model.
    model_dir = config["model_ckpt"]
    if not config['no_evaluate']:
        print('get test metrics.')
        
    if config["sim_type"] == "scflow":
        scflow_config = config.get("scflow_config", {})
        length_units = scflow_config.get("units", "ang")
        model_length_unit = scflow_config.get("model_length_unit", "bohr")
        mf_init_functional = scflow_config.get("mf_init_functional", "pbe")
        init_density_fit = scflow_config.get("init_density_fit", True)
        density_fit = scflow_config.get("density_fit", True)
        # Allow filtering to be overridden from command line
        filtering = scflow_config.get("filtering", False)
        gt_tol = float(scflow_config.get("gt_tol", 1e-8))
        pred_tol = float(scflow_config.get("pred_tol", 1e-3))
        pad_eigval = scflow_config.get("pad_eigval", 1)
        init_gpu4pyscf = scflow_config.get("init_gpu4pyscf", False)
        if pad_eigval is not None:
            pad_eigval = float(pad_eigval)
        print(f"filtering: {filtering}")
        calculator = SCFlowRKSCalculator()
        calculator.set_model(
            model_dir, 
            units=length_units, 
            model_length_unit=model_length_unit, 
            mf_init_functional=mf_init_functional, 
            init_density_fit=init_density_fit,
            density_fit=density_fit,
            filtering=filtering,
            gt_tol=gt_tol,
            pred_tol=pred_tol,
            pad_eigval=pad_eigval,
            init_gpu4pyscf=init_gpu4pyscf
        )
        if config['no_evaluate']:
            test_metrics = {}
        else:
            raise NotImplementedError("SCFlow simulation is not supported yet.")
    elif config["sim_type"].lower() == "dft":
        calculator = RKSCalculator()
        if config['no_evaluate']:
            test_metrics = {}
        else:
            raise NotImplementedError('DFT simulation is not supported yet.')
    else:
        raise NotImplementedError('Invalid simulation type.')
    
    print(test_metrics)

    return calculator, test_metrics

def simulate(config, calculator, test_metrics):
    model_ckpt = Path(config["model_ckpt"])
    model_dir = model_ckpt.parent.parent
    save_name = config.get("save_name", "simulation")
    
    save_dir = model_dir / save_name
    print("save_dir: ", save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = save_dir / 'atoms.traj'
    thermo_log_path = save_dir / 'thermo.log'
    csv_path = save_dir / 'energy_force.csv'
    
    RESTART = False
    simulated_time = 0
    simulated_step = 0
    if trajectory_path.exists():
        try:
            if not thermo_log_path.exists():
                raise ValueError('trajectory exists but thermo.log does not exist.')
            history = Trajectory(trajectory_path)
            if len(history) > 0 and not config['purge']:
                atoms = history[-1]
                with open(thermo_log_path, 'r') as f:
                    lines = f.read().splitlines()
                # Find the last valid data line (skip headers starting with "Time[ps]")
                last_line = None
                for line in reversed(lines):
                    line = line.strip()
                    if line and not line.startswith('Time[ps]'):
                        last_line = line
                        break
                if last_line is None:
                    raise ValueError('No valid data lines found in thermo.log')
                simulated_time = [float(x) for x in last_line.split() if x][0]
                # timestep is in ps, convert to fs for step calculation
                simulated_step = int(simulated_time / config['integrator_config']['timestep'] * 1000)
                RESTART = True
                print(f'Found existing simulation. Simulated time: {simulated_time} ps')
            else:
                # Remove existing files if purge or empty trajectory
                if trajectory_path.exists():
                    os.remove(trajectory_path)
                if thermo_log_path.exists():
                    os.remove(thermo_log_path)
                if csv_path.exists():
                    os.remove(csv_path)
        except Exception as e:
            print(f"Failed to read existing trajectory; starting fresh. ({e})")
            if trajectory_path.exists():
                os.remove(trajectory_path)
            if thermo_log_path.exists():
                os.remove(thermo_log_path)
            if csv_path.exists():
                os.remove(csv_path)
                
    if not RESTART:
        if 'init_dataset_config' in config:
            # RMD17 or other custom dataset
            test_dataset = RMD17_DFT(config['init_dataset_config'])
        else:
            raise ValueError('Invalid dataset configuration.')

        if 'init_idx' in config:
            init_idx = config['init_idx']
        else:
            init_idx = random.randint(0, len(test_dataset) - 1)

        init_data = test_dataset[init_idx]
        atoms = data_to_atoms(init_data)
        simulated_time = 0
        simulated_step = 0
        print('Start simulation from scratch.')
        
        # Override atoms if init_atoms is provided
        if 'init_atoms' in config:
            atoms = config['init_atoms']
        
    # if config['plumed']:
    #     # plumed setting is explicitly for alanine dipeptide.
    #     from ase.calculators.plumed import Plumed
    #     if RESTART:
    #         plumed_meta_cmd = f"metad: METAD ARG=phi,psi PACE=500 HEIGHT=1.2 SIGMA=0.35,0.35 FILE={str(save_dir)}/HILLS RESTART=YES BIASFACTOR=6.0"
    #     else:
    #         plumed_meta_cmd = f"metad: METAD ARG=phi,psi PACE=500 HEIGHT=1.2 SIGMA=0.35,0.35 FILE={str(save_dir)}/HILLS BIASFACTOR=6.0"
    #     setup = [f"UNITS LENGTH=A TIME={1/(1000 * units.fs)} ENERGY={units.mol/units.kJ}",
    #      "MOLINFO STRUCTURE=./alanine_dipeptide_files/ala2.pdb",
    #      "phi: TORSION ATOMS=@phi-2 ",
    #      "psi: TORSION ATOMS=@psi-2 ",
    #      plumed_meta_cmd,
    #      f"PRINT ARG=phi,psi FILE={str(save_dir)}/COLVAR STRIDE=10"]
    #     atoms.set_calculator(
    #         Plumed(calc=calculator,
    #                input=setup,
    #                timestep=config['integrator_config']['timestep'] * units.fs,
    #                atoms=atoms,
    #                kT=1.0))
    # else:
    atoms.set_calculator(calculator)
    
    # Check if simulation is already complete
    if simulated_step >= config['steps']:
        print(f'Simulated step {simulated_step} >= {config["steps"]}. Simulation already complete.')
        return
        
    config["integrator_config"]["timestep"] *= units.fs
    if config["integrator"] in ['NoseHoover', 'NoseHooverChain']:
        config["integrator_config"]["temperature"] *= units.kB
        
    integrator = getattr(md_integrator, config["integrator"])(
        atoms, **config["integrator_config"])
    atoms.pbc = [False, False, False]
    print("atoms: ", atoms)
    print("atoms.positions: ", atoms.positions)
    simulator = Simulator(atoms, integrator, config["T_init"], 
                          restart=RESTART,
                          start_time=simulated_time,
                          save_dir=save_dir, 
                          save_frequency=config["save_freq"])
    
    start_time = time.time()    
    early_stop, step = simulator.run(config["steps"] - simulated_step)
    elapsed = time.time() - start_time
    test_metrics['running_time'] = elapsed
    test_metrics['early_stop'] = early_stop
    test_metrics['simulated_frames'] = step

    with open(save_dir / 'test_metric.json', 'w') as f:
        json.dump(test_metrics, f)
        

def main(config):
    seed_everywhere(config['seed'])
    save_name = 'md'
    if config["identifier"] is not None:
        save_name = 'md_' + config["sim_type"] + '_' + config["identifier"] + '_' + str(config["seed"])
    if 'init_idx' in config:
        save_name = save_name + '_init_' + str(config['init_idx'])
    # Add tag if provided
    if 'tag' in config and config['tag'] is not None:
        save_name = save_name + '_' + str(config['tag'])
    config['save_name'] = save_name
    os.makedirs(Path(config["model_dir"]) / save_name, exist_ok=True)
    with open(os.path.join(Path(config["model_dir"]) / save_name, 'config.yml'), 'w') as yf:
        yaml.dump(config, yf, default_flow_style=False)
        
    calculator, test_metrics = eval_and_init(config)
    simulate(config, calculator, test_metrics)
    

if __name__ == '__main__':       
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yml", required=True, type=Path)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--identifier", type=str)
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--init_idx", type=int, 
                        help='the index of the initial state selected from the init dataset.')
    parser.add_argument("--tag", type=str,
                        help='Tag to append to save folder name (e.g., version, experiment name).')
    parser.add_argument("--filtering", type=str, default=None,
                        help='Enable/Disable filtering for SCFlow calculator. (default: None, use config file). Use "true"/"True"/"1"/"yes" for True, "false"/"False"/"0"/"no" for False.')
    # parser.add_argument("--plumed", action='store_true',
    #                     help='MetaDynamics. only used for alanine dipeptide.')
    parser.add_argument("--evaluate", action='store_true', 
                        help='if <True>, skip force error evaluation and directly start simulation.')
    parser.add_argument("--purge", action='store_true', 
                        help='if <True>, remove the previous run if exists.')
    # Equiformer-specific overrides
    parser.add_argument("--sim_type", type=str, default='scflow',
                        help="Override simulation type (e.g., equiformer, ocp, esen).")
    # parser.add_argument("--equiformer_type", type=str,
    #                     help="Equiformer variant: eqv1 or eqv2.")
    # parser.add_argument("--equiformer_repo", type=str,
    #                     help="Path to Equiformer v1 repo (required for eqv1).")
    
    args, override_args = parser.parse_known_args()
    config, _, _ = load_config(args.config_yml)
    # Exclude filtering from general overrides since it's handled separately
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != 'filtering'}
    config.update(overrides)
    if args.filtering is not None:
        # Initialize scflow_config if it doesn't exist
        if 'scflow_config' not in config:
            config['scflow_config'] = {}
        # Convert string to boolean
        filtering_value = args.filtering.lower() in ['true', '1', 'yes']
        config['scflow_config']['filtering'] = filtering_value
    # if args.deepmd and args.nequip:
    #     raise ValueError('cannot set both <deepmd> and <nequip> to <True>.')
    # elif args.nequip:
    #     config['sim_type'] = 'nequip'
    # elif args.deepmd:
    #     config['sim_type'] = 'dp'
    # config['sim_type'] = 'scflow'
    config['no_evaluate'] = not args.evaluate
    config["plumed"] = False
    main(config)

"""
HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=2 python -m  md.simulate_gpu_v2 --config_yml /root/25DFT/QHFlow/src/md/md_conf/ethanol.yml
HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=2 python -m  md.simulate_gpu_v2 --config_yml /root/25DFT/QHFlow/src/md/md_conf/ethanol.yml --sim_type dft
HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=0 python -m  md.simulate_gpu_v2 --config_yml /root/25DFT/QHFlow/src/md/md_conf/salicylic_acid.yml --sim_type dft --tag v2
HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=2 python -m  md.simulate_gpu_v2 --config_yml /root/25DFT/QHFlow/src/md/md_conf/salicylic_acid.yml --tag v2
HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=3 python -m  md.simulate_gpu_v2 --config_yml /root/25DFT/QHFlow/src/md/md_conf/salicylic_acid.yml --tag v2_no_filtering
HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=2 python -m  md.simulate_gpu_v2 --config_yml /root/25DFT/QHFlow/src/md/md_conf/naphthalene.yml --sim_type dft
HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=1 python -m  md.simulate_gpu_v2 --config_yml /root/25DFT/QHFlow/src/md/md_conf/naphthalene.yml 
HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=3 python -m  md.simulate_gpu_v2 --config_yml /root/25DFT/QHFlow/src/md/md_conf/aspirin.yml --sim_type dft
HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=3 python -m  md.simulate_gpu_v2 --config_yml /root/25DFT/QHFlow/src/md/md_conf/aspirin.yml
"""


"""
ethanol
Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]
0.0000       -4218.8514   -4219.6729       0.8215   706.1
0.0500       -4219.3966   -4219.6719       0.2752   236.6
"""

"""
aspirin
Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]
0.0000      -17676.2191  -17678.0918       1.8727   689.9
0.0500      -17677.5576  -17678.0898       0.5323   196.1
"""

"""
aspirin dft
Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]
0.0000      -17615.5007  -17617.3733       1.8727   689.9
"""