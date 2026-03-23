#!/usr/bin/env python3
import hydra
import os
from pathlib import Path
from qhflow2.dataset_module.ori_dataset import MD17_DFT, random_split, get_mask

# Import common utilities
from qhflow2.common.setup import (
    setup_paths, setup_auxiliary_basis, setup_tensor_type_and_seed,
    get_root_path, get_mode
)
from qhflow2.common.data_utils import create_md17_data_loaders, log_dataset_info, load_md17_dataset
from qhflow2.common.training_utils import setup_callbacks, setup_logger, setup_trainer, log_training_config, setup_warmup_trainer

# Setup paths and import models
setup_paths()

import warnings
warnings.filterwarnings("ignore")

from qhflow2.pl_module import get_pl_model
from pytorch_lightning.utilities.model_summary import ModelSummary

import torch    
import sys
from datetime import datetime
from qhflow2.common.custom_logger import get_logger
logger = get_logger(__file__)

NUM_THREADS = 12
logger.info(f"Using torch.set_num_threads({NUM_THREADS})")
torch.set_num_threads(NUM_THREADS)

# Mode descriptions
MODE_DESCRIPTIONS = {
    "train": "Training mode - Trains the model on the training dataset and validates on validation set",
    "finetune": "Training mode with finetuning - Finetunes the model on the training dataset and validates on validation set",
    "test": "Test mode - Evaluates model performance on the test dataset using trained checkpoint",
    "test-mul": "Multiple test mode - Makes multiple test predictions with different random seeds",
    "predict": "Prediction mode - Generates predictions on new data using trained model",
    "inference": "Inference mode - Runs model inference with additional SCF integration",
    "eval": "Evaluation mode - Similar to test but with additional metrics and analysis (Deprecated)",
}

@hydra.main(config_path="../../../configs/md17", config_name="config_flow-lw10")
def main(conf):
    # Setup output directory and auxiliary basis
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    setup_auxiliary_basis(output_dir)

    with open(output_dir / "command_log.txt", "a") as f:
        cmd = " ".join(sys.argv)
        f.write(f"[{datetime.now()}] {cmd}\n")

    # Setup tensor type and seed
    setup_tensor_type_and_seed(conf)

    # Load the dataset
    root_path = get_root_path()
    logger.info(f"Loading {conf.dataset.dataset_name} dataset...")
    dataset = load_md17_dataset(conf, root_path)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_md17_data_loaders(dataset, conf)
    
    # Log dataset information
    log_dataset_info(dataset, train_loader.dataset, val_loader.dataset, test_loader.dataset)

    # Initialize the LightningModule
    pl_model_cls = get_pl_model(conf)
    logger.info(f"Using pl_module: {pl_model_cls}")
    lit_model = pl_model_cls(conf)
    logger.info(f"Using model: {lit_model.model.__class__}")
    
    logger.info(ModelSummary(lit_model, max_depth=2)) # max_depth 2 is enough, 3 is for debugging

    with open(output_dir / "model_summary.txt", "w") as f:
        f.write(str(ModelSummary(lit_model, max_depth=2)))

    # Get and validate mode
    mode = get_mode(conf)
    assert mode in ["train", "test", "eval", "inference", "predict", "test-mul"]
    setattr(lit_model, "test_mode", "test")

    # Import checkpoint utilities
    from common.checkpoint_utils import get_checkpoint_path, setup_wandb_logger
    
    # Load the checkpoint if it exists
    if mode in ["train", "test", "inference", "predict", "test-mul"]:
        # Get checkpoint path
        ckpt_path = get_checkpoint_path(conf, output_dir)
        
        conf.wandb.tags += ["md17", conf.dataset.dataset_name]

        # Setup wandb logger
        wandb_logger = setup_wandb_logger(conf, output_dir)
        wandb_logger.watch(model=lit_model, log_freq=500)

        # Setup callbacks and trainer
        callbacks = setup_callbacks(conf, output_dir)
        trainer = setup_trainer(conf, callbacks, [wandb_logger], output_dir)
        log_training_config(conf)

        if conf.model.version.lower() == "Real_QHNet".lower() and ckpt_path is None and mode == "train":
            from torch_geometric.loader import DataLoader
            # Setup warmup trainer
            logger.info("Warmup training for Real_QHNet")
                        
            warmup_trainer = setup_warmup_trainer(conf, callbacks, [wandb_logger], output_dir)
            
            real_lr = conf.dataset.learning_rate
            warmup_lr = 1e-6
            conf.dataset.learning_rate = warmup_lr
            
            # warmup_dataset = []
            # for i in range(400):
            #     warmup_dataset.append(train_loader.dataset[10000+i])
            warmup_dataset = train_loader.dataset

            train_loader_warmup = DataLoader(
                warmup_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=4,
                pin_memory=False,
            )
            
            warmup_trainer.fit(
                lit_model,
                train_dataloaders=train_loader_warmup
            )
            conf.dataset.learning_rate = real_lr

        if mode in ["predict"]:
            logger.info(f"Mode: {mode}, Setting output_dir to {output_dir}")
            setattr(lit_model, "output_dir", output_dir)        

        # Start training/testing
        _run_training_or_testing(mode, trainer, lit_model, train_loader, val_loader, test_loader, ckpt_path, conf, output_dir)


def _run_training_or_testing(mode, trainer, lit_model, train_loader, val_loader, test_loader, ckpt_path, conf, output_dir):
    """Run training or testing based on mode."""    
    if mode == "train":
        trainer.fit(
            lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )
        logger.info("Testing...")
        trainer.test(lit_model, test_loader, ckpt_path="best")
        
        # predict the model with last checkpoint
        logger.info("Predicting with last checkpoint...")
        setattr(lit_model, "test_mode", "predict")
        logger.info(f"Mode: {mode}, Setting output_dir to {output_dir}")
        setattr(lit_model, "output_dir", output_dir)        
        trainer.test(lit_model, test_loader, ckpt_path="last")

    elif mode == "test":
        # Test the model with the test metric (smaller metrics)
        trainer.test(lit_model, test_loader, ckpt_path=ckpt_path)
    elif mode in ["inference", "predict", "predict-mul"]:
        # predict the model with many metric (larger metrics)
        if mode == "inference":
            from torch_geometric.loader import DataLoader
            inf_loader = DataLoader(
                test_loader.dataset[:300],
                batch_size=1,
                shuffle=False,
                num_workers=conf.dataset.num_workers,
                pin_memory=conf.dataset.pin_memory,
            )
        else:
            inf_loader = test_loader
        setattr(lit_model, "test_mode", mode)
        logger.info(f"{lit_model.test_mode}...")
        trainer.test(lit_model, inf_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
