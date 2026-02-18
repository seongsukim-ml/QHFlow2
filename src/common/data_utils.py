#!/usr/bin/env python3
"""
Common data utilities for experiments.
"""
import os
from common.custom_logger import get_logger
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
from dataset_module.ori_dataset import MD17_DFT, random_split, get_mask
from dataset_module.md17_datasets_shard import MD17_DFT_Shard
from dataset_module.new_md_dataloader import RMD17_DFT
from dataset_module.new_md_dataloader_mdsim import MD17_DFT_MDSim
from torch.utils.data import Subset

logger = get_logger(__file__)

def load_md17_dataset(conf: DictConfig, root_path: str, dataset_type: str = None):
    if dataset_type is None:
        dataset_type = conf.dataset.get("dataset_type", "md17")
    else:
        dataset_type = dataset_type.lower()
        if dataset_type not in ["md17", "rmd17", "mdsim"]:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
    if dataset_type == "md17":
        if conf.dataset.get("use_shard", False):
            return MD17_DFT_Shard(
                os.path.join(root_path, "dataset"),
                prefix="_shard",
                name=conf.dataset.dataset_name,
                all_features=conf.dataset.get("all_features", False),
                use_in_memory=conf.dataset.get("use_in_memory", False),
                # transform=get_mask, # Not used
            )
        else:
            return MD17_DFT(
                os.path.join(root_path, "dataset"),
                name=conf.dataset.dataset_name,
                transform=get_mask,
                use_ref_hamiltonian_as_init_ham=conf.dataset.get("use_ref_hamiltonian_as_init_ham", False),
            )
    elif dataset_type == "rmd17":
        return RMD17_DFT(
            os.path.join(root_path, "dataset"),
            name=conf.dataset.dataset_name,
            prefix=conf.dataset.get("prefix", ""),
            postfix=conf.dataset.get("postfix", ""),
            load_orbitals=conf.dataset.get("load_orbitals", False),
            include_density=conf.dataset.get("include_density", False),
            all_features=conf.dataset.get("all_features", False),
            use_ref_hamiltonian_as_init_ham=conf.dataset.get("use_ref_hamiltonian_as_init_ham", False),
        )
    elif dataset_type == "mdsim":
        return MD17_DFT_MDSim(
            os.path.join(root_path, "dataset"),
            name=conf.dataset.dataset_name,
            prefix=conf.dataset.get("prefix", ""),
            postfix=conf.dataset.get("postfix", ""),
            load_orbitals=conf.dataset.get("load_orbitals", False),
            include_density=conf.dataset.get("include_density", False),
            all_features=conf.dataset.get("all_features", False),
            use_ref_hamiltonian_as_init_ham=conf.dataset.get("use_ref_hamiltonian_as_init_ham", False),
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

def create_md17_data_loaders(dataset, conf: DictConfig, batch_size=[None, None, None], train_size=None):
    """Create train, validation, and test data loaders.
    
    Args:
        dataset: The full dataset to split.
        conf: Configuration dictionary.
        batch_size: Optional custom batch sizes [train, valid, test].
        train_size: Optional custom train dataset size. If provided, overrides conf.dataset.train_size or num_train.
                    For MDSIM datasets, this slices the pre-defined train split from the front.
    """
    # Get train_size from config if not provided as argument
    if train_size is None:
        train_size = conf.dataset.get("train_size", None)
    
    # Check if dataset is MDSIM (has get_split_indices method)
    if hasattr(dataset, 'get_split_indices'):
        train_dataset, valid_dataset, test_dataset = _create_mdsim_dataset(dataset, conf, train_size=train_size)
    else:
        train_dataset, valid_dataset, test_dataset = _create_md17_dataset(dataset, conf, train_size=train_size)
    
    return _create_md17_data_loaders(train_dataset, valid_dataset, test_dataset, conf, batch_size)

def _create_md17_dataset(dataset, conf: DictConfig, train_size=None):
    """Split dataset into train, validation, and test sets.
    
    Args:
        dataset: The full dataset to split.
        conf: Configuration dictionary.
        train_size: Optional custom train dataset size. If provided, slices train set from the front
                    after splitting with config sizes, ensuring consistency across different train sizes.
    
    Note:
        When train_size is provided, the dataset is first split using config values (num_train, num_valid),
        then the train set is sliced from the front. This ensures that with the same seed,
        a smaller train_size will always be a subset of a larger train_size.
    """
    num_train_config = conf.dataset.num_train
    num_valid = conf.dataset.num_valid
    
    # Ensure we don't exceed dataset size
    total_size = len(dataset)
    if num_train_config + num_valid > total_size:
        logger.warning(f"Requested train({num_train_config}) + valid({num_valid}) exceeds dataset size({total_size}). Adjusting...")
        num_train_config = min(num_train_config, total_size - num_valid - 1)
        if num_train_config < 0:
            num_train_config = int(total_size * 0.8)
            num_valid = int(total_size * 0.1)
    
    num_test = total_size - num_train_config - num_valid
    
    # First, split with config sizes to ensure consistent ordering with same seed
    train_dataset_full, valid_dataset, test_dataset = random_split(
        dataset,
        [num_train_config, num_valid, num_test],
        seed=conf.split_seed,
    )
    
    # Then slice train dataset if custom train_size is provided
    if train_size is not None:
        if train_size > num_train_config:
            logger.warning(f"Requested train_size({train_size}) exceeds config num_train({num_train_config}). Using {num_train_config}.")
            train_size = num_train_config
        logger.info(f"Using custom train size: {train_size} (sliced from {num_train_config} samples)")
        train_dataset = Subset(train_dataset_full, range(train_size))
    else:
        train_dataset = train_dataset_full
    
    if conf.dataset.get("split_1000", False):
        train_dataset = Subset(train_dataset, range(950))
        valid_dataset = Subset(valid_dataset, range(50))
        logger.info(f"Split 1000 dataset: train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset


def _create_mdsim_dataset(dataset, conf: DictConfig, train_size=None):
    """Create train, validation, and test sets using pre-defined splits from MDSIM HDF5 file.
    
    Args:
        dataset: The MDSIM dataset (MD17_DFT_MDSim instance).
        conf: Configuration dictionary.
        train_size: Optional custom train dataset size. If provided, slices train set from the front.
    
    Note:
        MDSIM datasets have pre-defined train/val/test splits stored in the HDF5 file.
        This function uses those splits directly instead of random splitting.
        If train_size is provided, it slices the train set from the front for consistency.
    """
    # Get pre-defined split indices from the dataset
    train_indices, val_indices, test_indices = dataset.get_split_indices()
    num_train_full, num_val, num_test = dataset.get_split_sizes()
    
    logger.info(f"MDSIM dataset using pre-defined splits: train={num_train_full}, val={num_val}, test={num_test}")
    
    # Create Subset datasets using the pre-defined indices
    train_dataset_full = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Slice train dataset if custom train_size is provided
    if train_size is not None:
        if train_size > num_train_full:
            logger.warning(f"Requested train_size({train_size}) exceeds available train samples({num_train_full}). Using {num_train_full}.")
            train_size = num_train_full
        logger.info(f"Using custom train size: {train_size} (sliced from {num_train_full} samples)")
        train_dataset = Subset(train_dataset_full, range(train_size))
    else:
        train_dataset = train_dataset_full
    
    if conf.dataset.get("split_1000", False):
        train_dataset = Subset(train_dataset, range(min(950, len(train_dataset))))
        valid_dataset = Subset(valid_dataset, range(min(50, len(valid_dataset))))
        logger.info(f"Split 1000 dataset: train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size: {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset


def _create_md17_data_loaders(train_dataset, valid_dataset, test_dataset, conf: DictConfig, batch_size=[None, None, None]):
    """Create train, validation, and test data loaders."""
    
    train_batch_size = conf.dataset.train_batch_size
    valid_batch_size = conf.dataset.valid_batch_size
    test_batch_size = conf.dataset.test_batch_size
    
    if batch_size is not None:
        if isinstance(batch_size, int):
            batch_size = [batch_size, batch_size, batch_size]
        if len(batch_size) == 1:
            batch_size = batch_size * 3
        if len(batch_size) == 2:
            batch_size = batch_size + [batch_size[-1]]
        if len(batch_size) != 3:
            raise ValueError(f"Batch size must be a int or a list of 1, 2, 3 elements: {batch_size}")
        if batch_size[0] is not None: 
            train_batch_size = batch_size[0]
            print(f"Using custom train batch size: {train_batch_size} instead of config batch size {conf.dataset.train_batch_size}")
        if batch_size[1] is not None:
            valid_batch_size = batch_size[1]
            print(f"Using custom valid batch size: {valid_batch_size} instead of config batch size {conf.dataset.valid_batch_size}")
        if batch_size[2] is not None:
            test_batch_size = batch_size[2]
            print(f"Using custom test batch size: {test_batch_size} instead of config batch size {conf.dataset.test_batch_size}")
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=conf.dataset.num_workers,
        pin_memory=conf.dataset.pin_memory,
    )
    
    val_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=conf.dataset.num_workers,
        pin_memory=conf.dataset.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=conf.dataset.num_workers,
        pin_memory=conf.dataset.pin_memory,
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_path(root_path: str, dataset_name: str):
    """Get the dataset path."""
    return os.path.join(root_path, "dataset")


def log_dataset_info(dataset, train_dataset, valid_dataset, test_dataset):
    """Log dataset information."""
    logger.info(f"Dataset size:      {len(dataset):>8}")
    logger.info(f"Train size:        {len(train_dataset):>8} ({len(train_dataset) / len(dataset):.2%})")
    logger.info(f"Validation size:   {len(valid_dataset):>8} ({len(valid_dataset) / len(dataset):.2%})")    
    logger.info(f"Test size:         {len(test_dataset):>8} ({len(test_dataset) / len(dataset):.2%})")
