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
import pytorch_lightning as pl
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

    # ── Dry-run: quick pipeline test with throughput measurement ──
    if conf.get("dry_run", False):
        _run_dry_run(conf)
        return

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

def _run_dry_run(conf):
    """Dry-run: load data, run a few steps, measure throughput + TFLOPS, exit.

    Usage:
        python train_qh9.py dry_run=true
        python train_qh9.py dry_run=true dataset.use_preprocessed=true
        python train_qh9.py dry_run=true dry_run_steps=5
    """
    import time

    dry_run_steps = conf.get("dry_run_steps", 3)
    dry_run_val_batches = conf.get("dry_run_val_batches", 2)
    flop_warmup = 2
    flop_measure = 3

    logger.info("=" * 60)
    logger.info("DRY RUN — pipeline + throughput + TFLOPS")
    logger.info("=" * 60)

    # Setup
    setup_tensor_type_and_seed(conf)
    root_path = get_root_path()

    # ── 1. Dataset loading ──
    logger.info("[1/6] Loading dataset...")
    t0 = time.perf_counter()
    dataset = load_qh9_dataset(conf, root_path)
    t_dataset = time.perf_counter() - t0
    logger.info(f"  Dataset loaded in {t_dataset:.1f}s")

    # ── 2. DataLoader creation ──
    logger.info("[2/6] Creating data loaders...")
    t0 = time.perf_counter()
    train_loader, val_loader, test_loader = create_qh9_data_loaders(dataset, conf)
    t_loader = time.perf_counter() - t0
    logger.info(
        f"  Loaders created in {t_loader:.1f}s — "
        f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )

    # ── 3. Model init ──
    logger.info("[3/6] Initializing model...")
    t0 = time.perf_counter()
    pl_model_cls = get_pl_model(conf)
    lit_model = pl_model_cls(conf)
    t_model = time.perf_counter() - t0
    logger.info(f"  Model initialized in {t_model:.1f}s — {lit_model.model.__class__.__name__}")
    param_count = sum(p.numel() for p in lit_model.parameters())
    logger.info(f"  Parameters: {param_count:,}")

    # ── 4. Training steps with throughput ──
    logger.info(f"[4/6] Running {dry_run_steps} training steps + {dry_run_val_batches} val batches...")

    trainer = pl.Trainer(
        accelerator=conf.get("accelerator", "gpu"),
        devices=1,
        precision=64 if conf.get("data_type") == "float64" else conf.get("precision", 32),
        max_steps=dry_run_steps,
        limit_val_batches=dry_run_val_batches,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        gradient_clip_val=conf.get("gradient_clip_val", None),
    )

    t0 = time.perf_counter()
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    t_train = time.perf_counter() - t0

    batch_size = conf.dataset.train_batch_size
    total_samples = dry_run_steps * batch_size
    throughput = total_samples / t_train if t_train > 0 else 0

    # ── 5. TFLOPS measurement ──
    logger.info(f"[5/6] Measuring TFLOPS (warmup={flop_warmup}, measure={flop_measure})...")
    flops_result = _measure_tflops(
        lit_model, train_loader, conf,
        warmup=flop_warmup, measure=flop_measure,
    )

    # ── 6. Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("DRY RUN RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Dataset:        {conf.dataset.dataset_name} ({conf.dataset.split})")
    logger.info(f"  Preprocessed:   {conf.dataset.get('use_preprocessed', False)}")
    logger.info(f"  Batch size:     {batch_size}")
    logger.info(f"  Num workers:    {conf.dataset.num_workers}")
    logger.info(f"  Pin memory:     {conf.dataset.pin_memory}")
    logger.info(f"  Precision:      {trainer.precision}")
    logger.info(f"  Parameters:     {param_count:,}")
    logger.info(f"  ---")
    logger.info(f"  Dataset load:   {t_dataset:.2f}s")
    logger.info(f"  Model init:     {t_model:.2f}s")
    logger.info(f"  {dry_run_steps} train steps:  {t_train:.2f}s")
    logger.info(f"  Throughput:     {throughput:.1f} samples/s")
    logger.info(f"  Per step:       {t_train / dry_run_steps * 1000:.0f} ms/step")
    logger.info(f"  Per sample:     {t_train / total_samples * 1000:.1f} ms/sample")
    logger.info(f"  ---")
    if flops_result:
        logger.info(f"  FLOPs/fwd:      {flops_result['flops_fwd'] / 1e9:.2f} GFLOPs")
        logger.info(f"  FLOPs/fwd+bwd:  {flops_result['flops_fwd_bwd'] / 1e9:.2f} GFLOPs")
        logger.info(f"  GPU time/step:  {flops_result['gpu_ms']:.1f} ms")
        logger.info(f"  TFLOPS (fwd):   {flops_result['tflops_fwd']:.2f}")
        logger.info(f"  TFLOPS (total): {flops_result['tflops_total']:.2f}")
        logger.info(f"  MFU (approx):   {flops_result['mfu_pct']:.1f}%")
    else:
        logger.info(f"  TFLOPS:         (measurement failed)")
    logger.info("=" * 60)


def _measure_tflops(lit_model, train_loader, conf, warmup=2, measure=3):
    """Measure TFLOPS for forward + backward passes.

    Uses torch.utils.flop_counter for FLOP counting and
    torch.cuda.Event for precise GPU timing.

    Returns:
        dict with flops_fwd, flops_fwd_bwd, gpu_ms, tflops_fwd, tflops_total, mfu_pct
        or None if measurement fails.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        logger.warning("torch.utils.flop_counter not available (requires PyTorch 2.1+)")
        return None

    device = next(lit_model.parameters()).device
    model = lit_model.model
    model.train()

    # Get a batch
    batch = next(iter(train_loader))
    batch = batch.to(device)
    batch = lit_model.post_processing(batch, lit_model.default_type)
    batch = lit_model.corrupt(batch, mul=1)

    # ── FLOP counting (forward only) ──
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        with torch.no_grad():
            out = model(batch, batch.init_ham_t, keep_blocks=True)
    flops_fwd = flop_counter.get_total_flops()

    # Estimate fwd+bwd as 3× forward (standard approximation)
    flops_fwd_bwd = flops_fwd * 3

    # ── GPU timing (forward + backward, steady state) ──
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        out = model(batch, batch.init_ham_t, keep_blocks=True)
        loss = out["hamiltonian_diagonal_blocks"].pow(2).mean()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(measure):
        optimizer.zero_grad()
        out = model(batch, batch.init_ham_t, keep_blocks=True)
        loss = out["hamiltonian_diagonal_blocks"].pow(2).mean()
        loss.backward()
        optimizer.step()
    end.record()

    torch.cuda.synchronize()
    gpu_ms = start.elapsed_time(end) / measure  # ms per step

    # ── Compute TFLOPS ──
    gpu_s = gpu_ms / 1000.0
    tflops_fwd = (flops_fwd / 1e12) / gpu_s if gpu_s > 0 else 0
    tflops_total = (flops_fwd_bwd / 1e12) / gpu_s if gpu_s > 0 else 0

    # ── MFU (Model FLOPs Utilization) ──
    # H200 peak: 989.5 TFLOPS (FP16/BF16), 494.7 (TF32), 66.9 (FP64), 3958 (INT8)
    # Use FP32 (TF32 mode) as reference since QHFlow2 defaults to float32
    peak_tflops = 494.7  # H200 TF32 peak
    mfu_pct = (tflops_total / peak_tflops) * 100 if peak_tflops > 0 else 0

    return {
        "flops_fwd": flops_fwd,
        "flops_fwd_bwd": flops_fwd_bwd,
        "gpu_ms": gpu_ms,
        "tflops_fwd": tflops_fwd,
        "tflops_total": tflops_total,
        "mfu_pct": mfu_pct,
    }


if __name__ == "__main__":
    main()
