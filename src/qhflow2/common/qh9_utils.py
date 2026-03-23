#!/usr/bin/env python3
"""
Common QH9 utilities for experiments.
"""
import os
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from qhflow2.common.custom_logger import get_logger

logger = get_logger(__file__)


def load_qh9_dataset(conf: DictConfig, root_path: str):
    """Load QH9 dataset based on configuration."""
    from dataset_module.qh9_datasets_split import QH9Stable, QH9Dynamic
    from dataset_module.qh9_datasets_shard import QH9Stable as QH9Stable_shard, QH9Dynamic as QH9Dynamic_shard
    dataset_name = conf.dataset.dataset_name
    logger.info(f"Loading {dataset_name} dataset...")

    # ── Preprocessed single LMDB (fastest) ──
    if conf.dataset.get("use_preprocessed", False):
        from dataset_module.qh9_preprocessed import QH9PreprocessedDataset
        lmdb_path = conf.dataset.get("preprocessed_lmdb", "")
        if not lmdb_path:
            # Auto-detect: look for <dataset_dir>_shard_preprocessed.lmdb
            lmdb_path = os.path.join(
                root_path, "dataset", f"{dataset_name}_shard_preprocessed.lmdb"
            )
        logger.info(f"Using preprocessed LMDB: {lmdb_path}")
        dataset = QH9PreprocessedDataset(
            lmdb_path=lmdb_path,
            split=conf.dataset.split,
            mode=conf.get("mode", "train"),
        )
        return dataset

    if conf.dataset.get("use_shard", False):
        logger.info("Using shard dataset")
        shard_feature_kwargs = dict(
            return_orbital_and_energies=conf.dataset.get("return_orbital_and_energies", False),
            use_ref_hamiltonian_as_init_ham=conf.dataset.get("use_ref_hamiltonian_as_init_ham", False),
            compute_q_tensor=conf.dataset.get("compute_q_tensor", True),
            include_initial_hamiltonian=conf.dataset.get("include_initial_hamiltonian", True),
            include_overlap=conf.dataset.get("include_overlap", True),
            include_dft_energy=conf.dataset.get("include_dft_energy", True),
            include_dft_forces=conf.dataset.get("include_dft_forces", True),
            include_cut_orbital_coefficients=conf.dataset.get("include_cut_orbital_coefficients", True),
        )

        if dataset_name == "QH9Stable":
            dataset = QH9Stable_shard(
                os.path.join(root_path, "dataset"),
                prefix="_shard",
                split=conf.dataset.split,
                **shard_feature_kwargs,
            )
        elif dataset_name == "QH9Dynamic":
            dataset = QH9Dynamic_shard(
                os.path.join(root_path, "dataset"),
                split=conf.dataset.split,
                prefix="_shard",
                version=conf.dataset.version,
                **shard_feature_kwargs,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
        return dataset

    else:
        logger.info("Using split dataset")
        if dataset_name == "QH9Stable":
            dataset = QH9Stable(
                os.path.join(root_path, "dataset"),
                split=conf.dataset.split,
            )
        elif dataset_name == "QH9Dynamic":
            dataset = QH9Dynamic(
                os.path.join(root_path, "dataset"),
                split=conf.dataset.split,
                version=conf.dataset.version,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return dataset

def _create_qh9_dataset(dataset, conf: DictConfig):
    train_dataset = dataset[dataset.train_mask]
    valid_dataset = dataset[dataset.val_mask]
    test_dataset = dataset[dataset.test_mask]
    
    return train_dataset, valid_dataset, test_dataset

def create_qh9_data_loaders(dataset, conf: DictConfig, batch_size=[None, None, None]):
    """Create train, validation, and test data loaders for QH9."""
    train_dataset, valid_dataset, test_dataset = _create_qh9_dataset(dataset, conf)
    return _create_qh9_data_loaders(train_dataset, valid_dataset, test_dataset, conf, batch_size)

def _create_qh9_data_loaders(train_dataset, valid_dataset, test_dataset, conf: DictConfig, batch_size=[None, None, None]):
    # Handle partial validation if specified
    if getattr(conf, "partial_val", None) is not None:
        assert conf.partial_val > 0 and conf.partial_val <= 1
        original_valid_size = len(valid_dataset)
        valid_dataset = valid_dataset[: int(len(valid_dataset) * conf.partial_val)]
        print(f"Using partial validation: {conf.partial_val} ({original_valid_size} -> {len(valid_dataset)}) (for speed up)")
    
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

    use_ddp = "ddp" in str(conf.get("strategy", "None")).lower()
    if not use_ddp:
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
    else:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("DDP strategy requires torch.distributed to be initialized.")

        print("Using DDP")
        print("Num workers: ", conf.dataset.num_workers)

        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            valid_dataset,
            shuffle=False,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            shuffle=False,
            drop_last=False,
        )

        persistent_workers = conf.dataset.num_workers > 0 and conf.dataset.get("persistent_workers", False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
            persistent_workers=persistent_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
            persistent_workers=persistent_workers,
        )
    
    return train_loader, val_loader, test_loader

def setup_warmup_training(conf: DictConfig, lit_model, train_dataset, wandb_logger, callbacks, use_warmup=True):
    """Setup warmup training for Real_QHNet if needed."""
    import pytorch_lightning as pl
    from torch_geometric.loader import DataLoader
    
    # Check if warmup is needed for Real_QHNet
    if (
        use_warmup and
        conf.model.version.lower() == "Real_QHNet".lower()
        and conf.get("warmup_step") is not None
        and conf.get("mode", "train") != "test"
    ):
        logger.info("Warmup training for Real_QHNet")
        
        # Store original learning rate
        real_lr = conf.dataset.learning_rate
        warmup_lr = 1e-3
        conf.dataset.learning_rate = warmup_lr
        
        # Create warmup trainer
        warmup_trainer = pl.Trainer(
            max_steps=conf.warmup_step,
            logger=wandb_logger,
            callbacks=callbacks,
            precision=64 if conf.data_type == "float64" else 32,
            log_every_n_steps=conf.dataset.train_batch_interval,
            accelerator="auto",
            devices=1,
            enable_progress_bar=True,
            gradient_clip_val=5.0,
            num_sanity_val_steps=8,
        )
        
        # Create warmup data loader with smaller batch size
        train_loader_warmup = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )
        
        # Run warmup training
        warmup_trainer.fit(
            lit_model,
            train_dataloaders=train_loader_warmup,
        )
        
        # Restore original learning rate
        conf.dataset.learning_rate = real_lr
        
        return True
    else:
        logger.info("Skip warmup training")
    
    return False


def create_inference_loader(mode: str, test_dataset, conf: DictConfig, test_length=300):
    """Create inference data loader based on mode."""
    if mode == "inference":
        logger.info(f"Using inference loader with {test_length} / {len(test_dataset)}")
        inf_loader = DataLoader(
            test_dataset[:test_length],
            batch_size=1,
            shuffle=False,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )
        
    elif mode == "predict-mul":
        inf_loader = DataLoader(
            test_dataset,
            batch_size=conf.dataset.test_batch_size,
            shuffle=False,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )
    elif mode == "predict":
        inf_loader = DataLoader(
            test_dataset,
            batch_size=conf.dataset.test_batch_size,
            shuffle=False,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return inf_loader

def dataset_abbr(dataset_name: str):
    qh9_abbr = {
        "QH9Stable-random".lower(): "iid",
        "QH9Stable-size_ood".lower(): "ood",
        "QH9Dynamic-300k-geometry".lower(): "geo",
        "QH9Dynamic-300k-mol".lower(): "mol",
    }
    if dataset_name.lower() not in qh9_abbr:
        raise ValueError(f"Unknown dataset for qh9_abbr: {dataset_name} (valid: {qh9_abbr.keys()})")
    return qh9_abbr[dataset_name.lower()]

def dataset_full_name(dataset_abbr: str):
    qh9_full_name = {
        "iid".lower(): "QH9Stable-random",
        "ood".lower(): "QH9Stable-size_ood",
        "geo".lower(): "QH9Dynamic-300k-geometry",
        "mol".lower(): "QH9Dynamic-300k-mol",
    }
    if dataset_abbr.lower() not in qh9_full_name:
        raise ValueError(f"Unknown dataset for qh9_full_name: {dataset_abbr} (valid: {qh9_full_name.keys()})")
    return qh9_full_name[dataset_abbr.lower()]