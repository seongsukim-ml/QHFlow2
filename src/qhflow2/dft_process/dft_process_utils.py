#!/usr/bin/env python3
"""
DFT Utilities Module

This module provides utility functions for Density Functional Theory (DFT) calculations,
including matrix transformations, PySCF integration, and data processing utilities.
"""

import os
import os.path as osp
import sys
import warnings
from argparse import Namespace

import numpy as np
# import cupy as cp

import pandas as pd
import torch
from ase.db import connect
import pyscf
from pyscf import gto, scf, dft

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
from qhflow2.common.custom_logger import get_logger
logger = get_logger()

# Add source path to system path for imports
source = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(source)
logger.info(f"Source path: {source}")

# Import project-specific modules
from qhflow2.pl_module.base_module import LitModel
from qhflow2.experiment import get_test_conf_qh9, get_test_conf_md17
from qhflow2.common.setup import setup_paths, get_root_path, setup_tensor_type_and_seed
from qhflow2.common.qh9_utils import load_qh9_dataset, _create_qh9_data_loaders, _create_qh9_dataset
from qhflow2.common.data_utils import _create_md17_data_loaders, load_md17_dataset, _create_md17_dataset
from qhflow2.common.metric import cal_orbital_and_energies
from qhflow2.common.matrix_transforms import get_convention_dict, _matrix_transform_single
from qhflow2.common.dft_utils import *
from qhflow2.common.units import *
from torch_geometric.loader import DataLoader

# Import necessary functions from ori_dataset
from qhflow2.dataset_module.ori_dataset import (
    MD17_DFT, random_split, get_mask,
    matrix_transform, 
    hamiltonian_transform, 
    construct_orbital_l_index,
    AOData,
    build_molecule,
    build_AO_index,
    Onsite_3idx_Overlap_Integral
)

# from dft_process.rmd17dataset import RMD17NPZLoader

# Set root path
root_path = os.path.dirname(source)
logger.info(f"Root path: {root_path}")

# =============================================================================
# CONSTANTS AND CONFIGURATIONS
# =============================================================================

# Atomic number to element symbol mapping
ATOMS_NAME = {
    1: "H",   # Hydrogen
    6: "C",   # Carbon
    7: "N",   # Nitrogen
    8: "O",   # Oxygen
    9: "F",   # Fluorine
}

# Orbital convention configurations for different basis sets
CONVENTION_DICT = get_convention_dict()

# =============================================================================
# MATRIX TRANSFORMATION FUNCTIONS
# =============================================================================

def matrix_transform_single(hamiltonian, atoms, convention="pyscf_def2svp"):
    """
    Transform matrix between different orbital conventions - CUDA optimized version.
    
    This function reorders and transforms the hamiltonian matrix according to the
    specified orbital convention, handling different basis set orderings.
    
    Args:
        hamiltonian (torch.Tensor): Hamiltonian matrix to transform
        atoms (torch.Tensor): Atomic numbers tensor
        convention (str): Orbital convention to use (default: "pyscf_def2svp")
        
    Returns:
        torch.Tensor: Transformed hamiltonian matrix
        
    Raises:
        AssertionError: If convention is not in CONVENTION_DICT
    """
    if convention is None:
        return hamiltonian
    assert convention in CONVENTION_DICT, f"Invalid convention: {convention}"
    conv = CONVENTION_DICT[convention]
    return _matrix_transform_single(hamiltonian, atoms, conv)

# Alias for backward compatibility
matrix_transform = matrix_transform_single

# =============================================================================
# DATA UTILITY FUNCTIONS
# =============================================================================

def print_data_item(data):
    """
    Print information about a data item.
    
    Args:
        data: Data object with keys attribute
    """
    print("data: ", data)
    for idx, key in enumerate(data.keys):
        print(f"{idx:2d}: {key:15s}, {data[key].shape}")


def pt_list(path):
    """
    Get sorted list of .pt files from a directory.
    
    Args:
        path (str): Directory path to search for .pt files
        
    Returns:
        list: Sorted list of .pt file paths
        
    Note:
        Files are sorted by molecule index and conformation index
    """
    list_pt = os.listdir(path)
    list_pt = [os.path.join(path, pt) for pt in list_pt]
    print(f"len({path.split('/')[-1]}): {len(list_pt)}")
    
    # Sort by molecule index and conformation index
    list_pt.sort(key=lambda x: (int(x.split("_")[1]), int(x.split("_")[2].split(".")[0])))
    
    # Print first item info for debugging
    if list_pt:
        item0 = torch.load(list_pt[0])
        print(f"First item keys: {item0.keys()}")
    
    return list_pt


# =============================================================================
# MD17 DATASET UTILITIES
# =============================================================================

class MD17Experiment:
    """
    MD17 Experiment class for managing dataset setup and testing.
    
    This class provides a structured way to setup and manage MD17 datasets
    for testing and evaluation.
    """
    dataset_names = [
        "water", "ethanol", "malondialdehyde", "uracil",
        "rmd-ethanol", "rmd-salicylic_acid", "rmd-naphthalene", "rmd-aspirin"
        ]
    ANG2BOHR = 1.8897261258369282      # Angstrom to Bohr conversion
    BOHR2ANG = 1.0 / ANG2BOHR         # Bohr to Angstrom conversion
    HA2meV = 27.211396641308 * 1000   # Hartree to meV conversion
    KCALPM2meV = 43.36                # kcal/mol to meV conversion

    def __init__(self, dataset_name=None, data_index=None, use_shard=False):
        """
        Initialize MD17Experiment object.
        
        Args:
            dataset_name (str, optional): Name of the dataset
            data_index (int, optional): Index of the dataset
        """

        if dataset_name is not None:
            assert dataset_name in self.dataset_names, f"dataset_name {dataset_name} not in {self.dataset_names}"
        else:
            assert data_index is not None, "data_index cannot be None if dataset_name is not provided"
            assert data_index < len(self.dataset_names), f"data_index {data_index} is out of range for dataset_names {self.dataset_names}"
            dataset_name = self.dataset_names[data_index]
        self.dataset_name = dataset_name
        # Get configuration and setup
        self.conf = get_test_conf_md17()
        setup_tensor_type_and_seed(self.conf)
        
        if self.dataset_name.startswith("rmd-"):
            self.conf.dataset.dataset_type = "rmd17"
        else:
            self.conf.dataset.dataset_type = "md17"
        self.conf.dataset.dataset_name = dataset_name
        if use_shard:
            self.conf.dataset.use_shard = True
        # Load dataset
        print(self.conf.dataset)
        logger.info(f"[!] conf.dataset.dataset_name: {self.conf.dataset.dataset_name}")
        self.dataset = load_md17_dataset(self.conf, root_path, dataset_type=self.conf.dataset.dataset_type, use_shard=use_shard)
        print(self.conf.dataset.dataset_type)
        print(self.conf.dataset.dataset_name)
        print(self.dataset)

    def check_path_initialized(func):
        """
        Decorator to check if path attribute is initialized before calling method.
        
        Args:
            func: The function to be decorated
            
        Returns:
            wrapper: The wrapped function that checks path initialization
            
        Raises:
            AttributeError: If path is not initialized
        """
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'path'):
                raise AttributeError("Path not initialized. Call set_model_path() first.")
            return func(self, *args, **kwargs)
        return wrapper

    @property
    @check_path_initialized
    def model_path(self):
        return self.path.model_path
    
    @property
    @check_path_initialized
    def pred_path(self):
        return self.path.pred
    
    @property
    @check_path_initialized
    def gt_path(self):
        return self.path.gt
    
    @property
    @check_path_initialized
    def pred_list(self):
        return self.path.pred_list
    
    @property
    @check_path_initialized
    def gt_list(self):
        return self.path.gt_list

    @property
    @check_path_initialized
    def calc_list(self):
        return self.path.calc_list
    
    def set_model_path(self, root_path=None, model_path=None, md17=True,
                 model_name="QHFlow", model_prefix="", model_postfix=""):
        """
        Set model path.
        
        Args:
            model_path (str): Path to model
        """
        self.path = ModelPath(self.dataset_name, root_path, model_path, md17, model_name, model_prefix, model_postfix)


def setup_md17_test(dataset_name=None, data_index=None):
    """
    Setup MD17 dataset for testing.
    
    Args:
        dataset_name (str, optional): Name of the dataset
        data_index (int, optional): Index of the dataset
        
    Returns:
        tuple: (dataset_name, dataset, configuration)
        
    Raises:
        AssertionError: If dataset_name is invalid or data_index is out of range
    """
    dataset_names = ["water", "ethanol", "malondialdehyde", "uracil"]

    if dataset_name is not None:
        assert dataset_name in dataset_names, f"dataset_name {dataset_name} not in {dataset_names}"
    else:
        assert data_index is not None, "data_index cannot be None if dataset_name is not provided"
        assert data_index < len(dataset_names), f"data_index {data_index} is out of range for dataset_names {dataset_names}"
        dataset_name = dataset_names[data_index]
        
    # Get configuration and setup
    conf = get_test_conf_md17()
    setup_tensor_type_and_seed(conf)
    conf.dataset.dataset_name = dataset_name
    logger.info(f"[!] conf.dataset.dataset_name: {conf.dataset.dataset_name}")
    
    # Load dataset
    dataset = load_md17_dataset(conf, root_path)
    
    return dataset_name, dataset, conf


from qhflow2.dataset_module.qh9_datasets_split import QH9Stable, QH9Dynamic
from qhflow2.dataset_module.qh9_datasets_shard import QH9Stable as QH9Stable_shard, QH9Dynamic as QH9Dynamic_shard

class QH9Experiment:
    """
    QH9 Experiment class for managing dataset setup and testing.
    """
    ANG2BOHR = 1.8897261258369282      # Angstrom to Bohr conversion
    BOHR2ANG = 1.0 / ANG2BOHR         # Bohr to Angstrom conversion
    HA2meV = 27.211396641308 * 1000   # Hartree to meV conversion
    KCALPM2meV = 43.36                # kcal/mol to meV conversion

    dataset_list = [
        ("QH9Stable", "random", None),
        ("QH9Stable", "size_ood", None),
        ("QH9Dynamic", "mol", "300k"),
        ("QH9Dynamic", "geometry", "300k"),
    ]

    def __init__(
        self, 
        dataset_name=None, 
        split=None, 
        version=None, 
        data_index=None, 
        use_shard=False,
        root_path="/root/25DFT/QHFlow",
    ):
        """
        Initialize QH9Experiment object.
        """
        if data_index is not None and isinstance(data_index, int):
            dataset_name, split, version = self.dataset_list[data_index]
        else:
            dataset_name = dataset_name
            split = split
            version = version
        
        assert dataset_name in ["QH9Stable", "QH9Dynamic"], f"dataset_name {dataset_name} not in ['QH9Stable', 'QH9Dynamic']"
        if dataset_name == "QH9Stable":
            assert split in ["random", "size_ood"], f"split {split} not in ['random', 'size_ood']"
        elif dataset_name == "QH9Dynamic":
            assert split in ["mol", "geometry"], f"split {split} not in ['mol', 'geometry']"
            assert version in ["300k"], f"version {version} not in ['300k']"
        self.dataset_name = dataset_name
        self.split = split
        self.version = version
        self.use_shard = use_shard

        self.root_path = root_path
        self.dataset_path = os.path.join(self.root_path, "dataset")
        assert os.path.exists(self.dataset_path), f"dataset_path {self.dataset_path} does not exist"

        if self.use_shard:
            if self.dataset_name == "QH9Stable":
                self.dataset = QH9Stable_shard(
                    self.dataset_path,
                    split=self.split,
                    # version=self.version,
                    prefix="_shard",
                    include_dft_energy=True,
                    include_dft_forces=True,
                )
            elif self.dataset_name == "QH9Dynamic":
                self.dataset = QH9Dynamic_shard(
                    self.dataset_path,
                    split=self.split,
                    version=self.version,
                    prefix="_shard",
                    include_dft_energy=True,
                    include_dft_forces=True,

                )
        else:
            if self.dataset_name == "QH9Stable":
                self.dataset = QH9Stable(
                    self.dataset_path,
                    split=self.split,
                    # version=self.version,
                )
            elif self.dataset_name == "QH9Dynamic":
                self.dataset = QH9Dynamic(
                    self.dataset_path,
                    split=self.split,
                    version=self.version,
                )

        self.train_dataset = self.dataset[self.dataset.train_mask]
        self.val_dataset = self.dataset[self.dataset.val_mask]
        self.test_dataset = self.dataset[self.dataset.test_mask]

        self.mask = {
            "train": self.dataset.train_mask,
            "val": self.dataset.val_mask,
            "test": self.dataset.test_mask,
        }

# =============================================================================
# MODEL PATH MANAGEMENT CLASS
# =============================================================================

class ModelPath:
    """
    Manages model paths and file organization for different datasets and models.
    
    This class provides a structured way to organize model outputs, predictions,
    and ground truth data for different datasets and model configurations.
    """
    
    def __init__(self, dataset_name, root_path=None, model_path=None, md17=True,
                 model_name="QHFlow", model_prefix="", model_postfix=""):
        """
        Initialize ModelPath object.
        
        Args:
            dataset_name (str): Name of the dataset
            root_path (str, optional): Root directory path
            model_path (str, optional): Direct model path override
            md17 (bool): Whether this is an MD17 dataset (default: True)
            model_name (str): Base model name (default: "QHFlow")
            model_prefix (str): Prefix for model name (default: "")
            model_postfix (str): Postfix for model name (default: "")
        """
        self.dataset_name = dataset_name
        if self.dataset_name == "malonaldehyde":
            logger.warning(f"Dataset name changed from malonaldehyde to malondialdehyde (for compatibility)")
            self.dataset_name = "malondialdehyde"
        self.root_path = root_path if root_path is not None else "/root/25DFT/QHFlow/src"
        self.model_path = None
        self.model_name = model_name
        self.model_prefix = model_prefix
        self.model_postfix = model_postfix
        
        # Construct full model name
        self.model_full_name = f"{self.model_name}{self.model_prefix}-{self.dataset_name}{self.model_postfix}"
        
        # Set model path
        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = f"{self.root_path}/outputs/{self.dataset_name}/{self.model_full_name}"
        
        # Set prediction and ground truth paths
        self.pred_path = os.path.join(self.model_path, "pred")
        self.gt_path = os.path.join(self.model_path, "gt")
        self.calc_path = os.path.join(os.path.dirname(self.model_path), "calc")
    
    def custom_path(self, path_folder):
        return os.path.join(os.path.dirname(self.model_path), path_folder)
    
    def __str__(self):
        """String representation of the model path."""
        return self.model_path
    
    @property
    def pred(self):
        """Get prediction path."""
        return os.path.join(self.model_path, "pred")
    
    @property
    def gt(self):
        """Get ground truth path."""
        return os.path.join(self.model_path, "gt")
    
    def pt_list(self, path=None, type="pred", print_info=True):
        """
        Get sorted list of .pt files from specified path.
        
        Args:
            path (str, optional): Path to search for files
            type (str): Type of files ("pred" or "gt") (default: "pred")
            print_info (bool): Whether to print information (default: True)
            
        Returns:
            list: Sorted list of .pt file paths
            
        Raises:
            ValueError: If type is not "pred" or "gt"
        """
        if path is None:
            if type == "pred":
                path = self.pred
            elif type == "gt":
                path = self.gt
            else:
                raise ValueError(f"type {type} not in ['pred', 'gt']")

        list_pt = os.listdir(path)
        list_pt = [os.path.join(path, pt) for pt in list_pt]
        
        if print_info:
            print(f"len({path.split('/')[-1]}): {len(list_pt)}, {self.dataset_name}")
            
        # Sort by molecule index and conformation index
        list_pt.sort(key=lambda x: (int(os.path.basename(x).split("_")[1]), int(os.path.basename(x).split("_")[2].split(".")[0])))
        return list_pt
    
    def pred_list(self, print_info=True):
        return self.pt_list(path=self.pred, type="pred", print_info=True)
    
    def gt_list(self, print_info=True):
        return self.pt_list(path=self.gt, type="gt", print_info=True)

    def calc_list(self):
        gt_list = self.gt_list(print_info=False)
        calc_list = []
        for gt in gt_list:
            calc_path = os.path.join(self.calc_path, f"{gt.split('/')[-1].split('.')[0]}.pt")
            calc_list.append(calc_path)
        os.makedirs(self.calc_path, exist_ok=True)
        return calc_list
    
    def custom_list(self, path, path_name="gt"):
        gt_list = self.gt_list(print_info=False)
        custom_list = []
        for gt in gt_list:
            file_name = gt.split('/')[-1].split('.')[0]
            file_name = file_name.replace("gt", path_name)
            custom_path = os.path.join(path, f"{file_name}.pt")
            custom_list.append(custom_path)
        os.makedirs(path, exist_ok=True)
        return custom_list
    
def print_first_item(list_pt):
    """
    Print first item of a list of .pt files.
    """
    item0 = torch.load(list_pt[0])
    print(f"First item keys: {item0.keys()}")
    return item0