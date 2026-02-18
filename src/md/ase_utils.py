from pathlib import Path
import copy
import logging
import os
import sys
import yaml
import torch
import json
import types
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

# # torch 1.x compatibility: e3nn/ocpmodels expect torch.compiler.disable
# if not hasattr(torch, "compiler"):
#     torch.compiler = types.SimpleNamespace()
#     torch.compiler.disable = lambda *args, **kwargs: (lambda fn: fn)

# # Add fairchem to path BEFORE any other imports
# _fairchem_path = str(Path.home() / 'fairchem' / 'src')
# if _fairchem_path not in sys.path:
#     sys.path.insert(0, _fairchem_path)

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory
import csv

# Logger for MD simulation with energy and force norm output
class NeuralMDLogger(MDLogger):
    def __init__(self,
                 *args,
                 start_time=0,
                 verbose=True,
                 **kwargs):
        if start_time == 0:
            header = True
        else:
            header = False
        super().__init__(header=header, *args, **kwargs)
        """
        Logger uses ps units.
        """
        self.start_time = start_time
        self.verbose = verbose
        self.natoms = self.atoms.get_number_of_atoms()
        
        # Customize header and format to include force norm
        # Set format regardless of header (needed for restart)
        self.hdr = 'Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]   Fnorm_mean[eV/A] Fnorm_max[eV/A]'
        self.fmt = '%10.4f %12.4f %12.4f %12.4f %7.1f %17.6f %17.6f'
        if header:
            if verbose:
                print(self.hdr)

    def __call__(self):
        if self.start_time > 0 and self.dyn.get_time() == 0:
            return
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * self.natoms)
        if self.peratom:
            epot /= self.natoms
            ekin /= self.natoms
        
        # Calculate force norm
        forces = self.atoms.get_forces()
        force_norms = np.linalg.norm(forces, axis=1)
        force_norm_mean = np.mean(force_norms)
        force_norm_max = np.max(force_norms)
        
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000*units.fs) + self.start_time
            dat = (t,)
        else:
            dat = ()
        dat += (epot+ekin, epot, ekin, temp, force_norm_mean, force_norm_max)
        if self.stress:
            dat += tuple(self.atoms.get_stress() / units.GPa)
            # Update format if stress is included
            if not hasattr(self, '_stress_format_set'):
                self.fmt = '%10.4f %12.4f %12.4f %12.4f %7.1f %17.6f %17.6f ' + ' '.join(['%9.4f'] * 6)
                self._stress_format_set = True
        self.logfile.write(self.fmt % dat + '\n')
        self.logfile.flush()

        if self.verbose:
            print(self.fmt % dat)

class CSVLogger:
    """Logger for saving energy and force data to CSV file.
    
    Saves time, energy (potential, kinetic, total), force norm statistics,
    and force components for each atom to a CSV file.
    Force units: eV/A
    Energy units: eV
    Time units: ps
    """
    def __init__(self, integrator, atoms, csv_path, start_time=0):
        self.integrator = integrator
        self.atoms = atoms
        self.csv_path = Path(csv_path)
        self.start_time = start_time
        self.natoms = self.atoms.get_number_of_atoms()
        
        # Check if file exists and has data (for restart)
        file_exists = self.csv_path.exists()
        if file_exists:
            # Read existing file to check if it has data
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) > 1:  # Has header and at least one data row
                    self.has_header = True
                    self.mode = 'a'
                else:
                    self.has_header = False
                    self.mode = 'w'
        else:
            self.has_header = False
            self.mode = 'w'
        
        # Open file in append mode for restart, write mode for new
        self.csvfile = open(self.csv_path, self.mode, newline='')
        self.writer = csv.writer(self.csvfile)
        
        # Write header if new file
        if not self.has_header:
            header = ['Time_ps', 'Epot_eV', 'Ekin_eV', 'Etot_eV', 'Force_norm_mean_eV_A', 'Force_norm_max_eV_A']
            # Add force components for each atom (units: eV/A)
            for i in range(self.natoms):
                header.extend([f'Force_x_{i}_eV_A', f'Force_y_{i}_eV_A', f'Force_z_{i}_eV_A'])
            self.writer.writerow(header)
            self.csvfile.flush()
    
    def __call__(self):
        if self.start_time > 0 and self.integrator.get_time() == 0:
            return
        
        # Get energy
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        etot = epot + ekin
        
        # Get time
        if self.integrator is not None:
            t = self.integrator.get_time() / (1000*units.fs) + self.start_time
        else:
            t = 0.0
        
        # Get forces
        forces = self.atoms.get_forces()
        force_norms = np.linalg.norm(forces, axis=1)
        force_norm_mean = np.mean(force_norms)
        force_norm_max = np.max(force_norms)
        
        # Prepare row data
        row = [t, epot, ekin, etot, force_norm_mean, force_norm_max]
        # Add force components (flatten forces array)
        row.extend(forces.flatten().tolist())
        
        # Write row
        self.writer.writerow(row)
        self.csvfile.flush()
    
    def close(self):
        if self.csvfile:
            self.csvfile.close()

class Simulator:
    def __init__(self, 
                 atoms, 
                 integrator,
                 T_init,
                 start_time=0,
                 save_dir='./log',
                 restart=False,
                 save_frequency=100,
                 min_temp=0.1,
                 max_temp=100000):
        self.atoms = atoms
        self.integrator = integrator
        self.save_dir = Path(save_dir)
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.natoms = self.atoms.get_number_of_atoms()

        # initialize system momentum 
        if not restart:
            assert (self.atoms.get_momenta() == 0).all()
            MaxwellBoltzmannDistribution(self.atoms, T_init * units.kB)
        
        # attach trajectory dump 
        self.traj = Trajectory(self.save_dir / 'atoms.traj', 'a', self.atoms)
        self.integrator.attach(self.traj.write, interval=save_frequency)
        
        # attach log file
        self.integrator.attach(NeuralMDLogger(self.integrator, self.atoms, 
                                        self.save_dir / 'thermo.log', 
                                        start_time=start_time, mode='a'), 
                               interval=save_frequency)
        
        # attach CSV logger for energy and force
        self.csv_logger = CSVLogger(self.integrator, self.atoms,
                                    self.save_dir / 'energy_force.csv',
                                    start_time=start_time)
        self.integrator.attach(self.csv_logger, interval=save_frequency)
        
    def run(self, steps):
        early_stop = False
        step = 0
        for step in tqdm(range(steps)):
            self.integrator.run(1)
            
            ekin = self.atoms.get_kinetic_energy()
            temp = ekin / (1.5 * units.kB * self.natoms)

            prev_atoms_positions = self.atoms.positions
            prev_atoms_momenta = self.atoms.get_momenta()
            prev_atoms_velocities = self.atoms.get_velocities()
            prev_atoms_energy = self.atoms.get_potential_energy()
            prev_atoms_kinetic_energy = self.atoms.get_kinetic_energy()


            if temp < self.min_temp or temp > self.max_temp:
                print(f'Temperature {temp:.2f} is out of range: \
                        [{self.min_temp:.2f}, {self.max_temp:.2f}]. \
                        Early stopping the simulation.')
                early_stop = True
                print(f"prev_atoms_positions: {prev_atoms_positions}")
                print(f"prev_atoms_momenta: {prev_atoms_momenta}")
                print(f"prev_atoms_velocities: {prev_atoms_velocities}")
                print(f"prev_atoms_energy: {prev_atoms_energy}")
                print(f"prev_atoms_kinetic_energy: {prev_atoms_kinetic_energy}")
                print("--------------------------------")
                print(f"atoms.positions: {self.atoms.positions}")
                print(f"atoms.get_potential_energy(): {self.atoms.get_potential_energy()}")
                print(f"atoms.get_kinetic_energy(): {self.atoms.get_kinetic_energy()}")
                break
            
        self.traj.close()
        self.csv_logger.close()
        return early_stop, (step+1)

def merge_dicts(dict1: dict, dict2: dict):
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py

    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.

    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = []

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                duplicates += [f"{k}.{dup}" for dup in duplicates_k]
            else:
                return_dict[k] = dict2[k]
                duplicates.append(k)

    return return_dict, duplicates
    

def load_config(path: str, previous_includes: list = []):
    path = Path(path)
    if path in previous_includes:
        raise ValueError(
            f"Cyclic config include detected. {path} included in sequence {previous_includes}."
        )
    previous_includes = previous_includes + [path]

    direct_config = yaml.safe_load(open(path, "r"))

    # Load config from included files.
    if "includes" in direct_config:
        includes = direct_config.pop("includes")
    else:
        includes = []
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, '{}' provided".format(type(includes))
        )

    config = {}
    duplicates_warning = []
    duplicates_error = []

    for include in includes:
        include_config, inc_dup_warning, inc_dup_error = load_config(
            include, previous_includes
        )
        duplicates_warning += inc_dup_warning
        duplicates_error += inc_dup_error

        # Duplicates between includes causes an error
        config, merge_dup_error = merge_dicts(config, include_config)
        duplicates_error += merge_dup_error

    # Duplicates between included and main file causes warnings
    config, merge_dup_warning = merge_dicts(config, direct_config)
    duplicates_warning += merge_dup_warning

    return config, duplicates_warning, duplicates_error