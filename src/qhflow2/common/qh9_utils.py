#!/usr/bin/env python3
"""
Common QH9 utilities for experiments.
"""
import os
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
from qhflow2.common.custom_logger import get_logger

logger = get_logger(__file__)


def load_qh9_dataset(conf: DictConfig, root_path: str):
    """Load QH9 or nablaDFT dataset based on configuration."""
    from qhflow2.dataset_module.qh9_datasets_split import QH9Stable, QH9Dynamic
    from qhflow2.dataset_module.qh9_datasets_shard import QH9Stable as QH9Stable_shard, QH9Dynamic as QH9Dynamic_shard
    dataset_name = conf.dataset.dataset_name
    logger.info(f"Loading {dataset_name} dataset...")

    # ── QH9 Density dataset ──
    # Handled in create_qh9_data_loaders (split-per-dataset, like NablaDFT)
    if dataset_name == "QH9Density":
        logger.info(f"Loading QH9Density from: {conf.dataset.source_lmdb}")
        return None  # dataset created per-split in create_qh9_data_loaders

    # ── nablaDFT dataset ──
    if dataset_name == "NablaDFT":
        from qhflow2.dataset_module.nabladft_dataset import NablaDFTDataset
        lmdb_path = conf.dataset.source_lmdb
        logger.info(f"Loading NablaDFT from: {lmdb_path}")
        dataset = NablaDFTDataset(
            lmdb_path=lmdb_path,
            split=conf.dataset.split,
            split_ratio=list(conf.dataset.get("split_ratio", [0.8, 0.1, 0.1])),
            seed=conf.dataset.get("split_seed", 42),
            full_orbitals=conf.dataset.get("full_orbitals", 32),
            basis=conf.dataset.get("basis", "def2-svp-nabla"),
            convention=conf.dataset.get("convention", "pyscf_def2svp_nabla_to_e3nn"),
            include_overlap=conf.dataset.get("include_overlap", True),
            include_initial_hamiltonian=conf.dataset.get("include_initial_hamiltonian", True),
            include_dft_energy=conf.dataset.get("include_dft_energy", False),
            include_dft_forces=conf.dataset.get("include_dft_forces", False),
        )
        return dataset

    # ── Preprocessed single LMDB (fastest) ──
    if conf.dataset.get("use_preprocessed", False):
        from qhflow2.dataset_module.qh9_preprocessed import QH9PreprocessedDataset
        lmdb_path = conf.dataset.get("preprocessed_lmdb", "")
        if not lmdb_path:
            # Auto-detect: try <name>_preprocessed.lmdb, then <name>_shard_preprocessed.lmdb
            for suffix in ["_preprocessed.lmdb", "_shard_preprocessed.lmdb"]:
                candidate = os.path.join(root_path, "dataset", f"{dataset_name}{suffix}")
                if os.path.exists(candidate):
                    lmdb_path = candidate
                    break
            if not lmdb_path:
                lmdb_path = os.path.join(root_path, "dataset", f"{dataset_name}_preprocessed.lmdb")
        logger.info(f"Using preprocessed LMDB: {lmdb_path}")
        dataset = QH9PreprocessedDataset(
            lmdb_path=lmdb_path,
            split=conf.dataset.split,
            mode=conf.get("mode", "train"),
            compute_q_tensor=conf.dataset.get("compute_q_tensor", True),
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
    from qhflow2.dataset_module.qh9_preprocessed import QH9PreprocessedDataset
    if isinstance(dataset, QH9PreprocessedDataset):
        # Preprocessed dataset uses torch Subset (not InMemoryDataset slicing)
        from torch.utils.data import Subset
        train_dataset = Subset(dataset, dataset.train_mask.tolist())
        valid_dataset = Subset(dataset, dataset.val_mask.tolist())
        test_dataset = Subset(dataset, dataset.test_mask.tolist())
    else:
        train_dataset = dataset[dataset.train_mask]
        valid_dataset = dataset[dataset.val_mask]
        test_dataset = dataset[dataset.test_mask]

    return train_dataset, valid_dataset, test_dataset

def create_qh9_data_loaders(dataset, conf: DictConfig, batch_size=[None, None, None]):
    """Create train, validation, and test data loaders for QH9 or nablaDFT."""
    # nablaDFT: dataset is already split, load each split separately
    if conf.dataset.dataset_name == "NablaDFT":
        from qhflow2.dataset_module.nabladft_dataset import NablaDFTDataset
        lmdb_path = conf.dataset.source_lmdb
        common_kwargs = dict(
            lmdb_path=lmdb_path,
            split_ratio=list(conf.dataset.get("split_ratio", [0.8, 0.1, 0.1])),
            seed=conf.dataset.get("split_seed", 42),
            full_orbitals=conf.dataset.get("full_orbitals", 32),
            basis=conf.dataset.get("basis", "def2-svp-nabla"),
            convention=conf.dataset.get("convention", "pyscf_def2svp_nabla_to_e3nn"),
            include_overlap=conf.dataset.get("include_overlap", True),
            include_initial_hamiltonian=conf.dataset.get("include_initial_hamiltonian", True),
            include_dft_energy=conf.dataset.get("include_dft_energy", False),
            include_dft_forces=conf.dataset.get("include_dft_forces", False),
        )
        train_dataset = NablaDFTDataset(split="train", **common_kwargs)
        valid_dataset = NablaDFTDataset(split="val", **common_kwargs)
        test_dataset = NablaDFTDataset(split="test", **common_kwargs)
        return _create_qh9_data_loaders(
            train_dataset, valid_dataset, test_dataset, conf, batch_size
        )

    # QH9Density: dataset is already split, load each split separately
    if conf.dataset.dataset_name == "QH9Density":
        from qhflow2.dataset_module.qh9_density_dataset import QH9DensityDataset
        common_kwargs = dict(
            lmdb_path=conf.dataset.source_lmdb,
            split_path=conf.dataset.get("split_path", None),
            include_density_matrix=conf.dataset.get("include_density_matrix", True),
            include_initial_density_matrix=conf.dataset.get("include_initial_density_matrix", True),
            include_initial_hamiltonian=conf.dataset.get("include_initial_hamiltonian", True),
            include_overlap=conf.dataset.get("include_overlap", True),
            include_dft_energy=conf.dataset.get("include_dft_energy", False),
            include_dft_forces=conf.dataset.get("include_dft_forces", False),
            compute_q_tensor=conf.dataset.get("compute_q_tensor", False),
            functional=conf.dataset.get("functional", "b3lyp"),
            basis_name=conf.dataset.get("basis_name", "def2-svp"),
        )
        train_dataset = QH9DensityDataset(split="train", **common_kwargs)
        valid_dataset = QH9DensityDataset(split="val", **common_kwargs)
        test_dataset = QH9DensityDataset(split="test", **common_kwargs)
        return _create_qh9_data_loaders(
            train_dataset, valid_dataset, test_dataset, conf, batch_size
        )

    train_dataset, valid_dataset, test_dataset = _create_qh9_dataset(dataset, conf)
    return _create_qh9_data_loaders(train_dataset, valid_dataset, test_dataset, conf, batch_size)

def _create_qh9_data_loaders(train_dataset, valid_dataset, test_dataset, conf: DictConfig, batch_size=[None, None, None]):
    # Handle partial validation if specified
    if getattr(conf, "partial_val", None) is not None:
        assert conf.partial_val > 0 and conf.partial_val <= 1
        original_valid_size = len(valid_dataset)
        n_val = int(original_valid_size * conf.partial_val)
        if hasattr(valid_dataset, 'index_select'):
            # PyG InMemoryDataset
            valid_dataset = valid_dataset[:n_val]
        else:
            # NablaDFTDataset or other Dataset subclass — use Subset
            from torch.utils.data import Subset
            valid_dataset = Subset(valid_dataset, range(n_val))
        print(f"Using partial validation: {conf.partial_val} ({original_valid_size} -> {n_val}) (for speed up)")
    
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

    # Lightning auto-injects DistributedSampler for DDP — no manual sampler needed.
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
        "nabladft-train".lower(): "nabla",
        "qh9density-random".lower(): "den",
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