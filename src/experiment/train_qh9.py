#!/usr/bin/env python3
import hydra
import os
from pathlib import Path
from datetime import datetime
import sys

# Import common utilities
from common.setup import (
    setup_paths, setup_auxiliary_basis, setup_tensor_type_and_seed,
    get_root_path, get_mode
)
from common.qh9_utils import (
    load_qh9_dataset, create_qh9_data_loaders, setup_warmup_training,
    create_inference_loader, dataset_abbr, dataset_full_name
)
from common.checkpoint_utils import get_checkpoint_path, get_checkpoint_path_finetune, setup_wandb_logger
from common.training_utils import setup_callbacks, setup_trainer, log_training_config
from common.data_utils import log_dataset_info

# Setup paths and import models
setup_paths()

import warnings
warnings.filterwarnings("ignore")

from pl_module import get_pl_model, _get_model_by_pl_type
from pytorch_lightning.utilities.model_summary import ModelSummary

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from common.custom_logger import get_logger
logger = get_logger(__file__)

NUM_THREADS = 16
logger.info(f"Using torch.set_num_threads({NUM_THREADS})")
torch.set_num_threads(NUM_THREADS)


# Mode descriptions
MODE_DESCRIPTIONS = {
    "train": "Training mode - Trains the model on the training dataset and validates on validation set",
    "finetune": "Training mode with finetuning - Finetunes the model on the training dataset and validates on validation set",
    "test": "Test mode - Evaluates model performance on the test dataset using trained checkpoint",
    "test-mul": "Multiple test mode - Makes multiple test predictions with different random seeds",
    "predict": "Prediction mode - Generates predictions on new data using trained model",
    "predict_wo_eval": "Prediction mode without evaluation - Generates predictions on new data using trained model without evaluation",
    "inference": "Inference mode - Runs model inference with additional SCF integration",
    "eval": "Evaluation mode - Similar to test but with additional metrics and analysis (Deprecated)",
}

@hydra.main(config_path="../config_qh9", config_name="config_flow_v2_simple")
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
    dataset = load_qh9_dataset(conf, root_path)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_qh9_data_loaders(dataset, conf)
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset
    
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
    use_pretrained_model = conf.get("use_pretrained_model", False)
    POSSIBLE_MODES = ["train", "test", "test-mul", "predict", "inference", "eval", "predict_wo_eval"]
    assert mode in POSSIBLE_MODES

    # Get checkpoint path
    if use_pretrained_model:
        logger.info("Using pretrained model as weights")
        ckpt_path, pretrained_ckpt = get_checkpoint_path_finetune(conf, output_dir)
    else:
        ckpt_path = get_checkpoint_path(conf, output_dir)
        pretrained_ckpt = None

    # Add tags to wandb
    conf.wandb.tags += ["qh9", dataset_abbr(conf.dataset.dataset_full_name)]
    
    # Setup wandb logger
    wandb_logger = setup_wandb_logger(conf, output_dir)
    wandb_logger.watch(model=lit_model, log_freq=500)

    # Setup callbacks and trainer
    callbacks = setup_callbacks(conf, output_dir)
    trainer = setup_trainer(conf, callbacks, [wandb_logger], output_dir)
    log_training_config(conf)

    # Setup warmup training if needed (it is only for Real_QHNet since it sometimes fails to converge)
    setup_warmup_training(conf, lit_model, train_dataset, wandb_logger, callbacks, ckpt_path is None)
    # Start training/testing
    _run_qh9_training_or_testing(mode, trainer, lit_model, train_loader, val_loader, test_loader, ckpt_path, pretrained_ckpt, conf, output_dir)

def _run_qh9_training_or_testing(mode, trainer, lit_model, train_loader, val_loader, test_loader, ckpt_path, pretrained_ckpt, conf, output_dir):
    """Run QH9 training or testing based on mode."""
    
    if mode == "train":
        if ckpt_path is None and conf.get("use_pretrained_model", False):
            logger.info("Using pretrained model as backbone")
            backbone_pl_type = conf.get("backbone_pl_type", "flow")
            backbone_pl_cls = _get_model_by_pl_type(backbone_pl_type)
            backbone_model = backbone_pl_cls.load_from_checkpoint(pretrained_ckpt, conf=conf, strict=False)
            lit_model.model = backbone_model.model
            if conf.get("freeze_backbone", False):
                lit_model.model.eval()

        if ckpt_path is not None:
            trainer.fit(
                lit_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path,
            )
        else:
            trainer.fit(
                lit_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

        logger.info("Testing...")
        trainer.test(lit_model, test_loader, ckpt_path="best")
        
        # predict the model with last checkpoint
        logger.info("Predicting with last checkpoint...")
        
        if conf.get("predict_after_train", True):
            setattr(lit_model, "output_dir", output_dir)
            setattr(lit_model, "test_mode", "predict")
            os.makedirs(output_dir / "sample", exist_ok=True)
            trainer.test(lit_model, test_loader, ckpt_path="last")
        else:
            logger.info("Skipping prediction after training")
    elif mode == "test":
        # Test the model
        trainer.test(lit_model, test_loader, ckpt_path=ckpt_path)
    elif mode in ["inference", "predict", "predict-mul", "predict_wo_eval"]:
        logger.info(f"Mode: {mode}, Setting output_dir to {output_dir}")

        setattr(lit_model, "output_dir", output_dir)
        
        if mode == "inference":
            setattr(lit_model, "test_mode", "inference")
        elif mode == "predict-mul":
            setattr(lit_model, "test_mode", "predict-mul")
        elif mode in ["predict", "predict_wo_eval"]:
            setattr(lit_model, "test_mode", mode)
            os.makedirs(output_dir / "sample", exist_ok=True)
        
        # Create inference loader
        inf_loader = create_inference_loader(mode, test_loader.dataset, conf)
        
        logger.info(f"{lit_model.test_mode}...")
        trainer.test(lit_model, inf_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
