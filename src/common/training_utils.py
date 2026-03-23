#!/usr/bin/env python3
"""
Common training utilities for experiments.
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from omegaconf import DictConfig
from pathlib import Path
from common.custom_logger import get_logger

logger = get_logger(__file__)


def setup_callbacks(conf: DictConfig, output_dir: Path, loss_format: str = ".7f"):
    """Setup training callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="weights-{epoch:02d}-{val_loss:" + loss_format + "}", # {epoch:02d} is the number of epoch
        monitor=conf.get("monitor", "val/loss"),
        mode=conf.get("monitor_mode", "min"),
        save_top_k=conf.get("save_top_k", 1),
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_JOIN_CHAR = "#"
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Rich progress bar
    rich_progress_bar_theme = RichProgressBarTheme(metrics_format=".7f")
    rich_progress_bar = RichProgressBar(leave=False, theme=rich_progress_bar_theme)
    callbacks.append(rich_progress_bar)

    # TFLOPS measurement (opt-out via log_tflops: false)
    if conf.get("log_tflops", True):
        tflops_interval = conf.get("tflops_log_interval", 50)
        callbacks.append(TFLOPSCallback(log_interval=tflops_interval))

    return callbacks


def setup_logger(conf: DictConfig, output_dir: Path):
    """Setup logging configuration."""
    loggers = []
    
    # Weights & Biases logger
    if conf.get("wandb", {}).get("enabled", False):
        wandb_logger = WandbLogger(
            project=conf.wandb.get("project", "QHFlow"),
            name=conf.wandb.get("name", None),
            log_model=conf.wandb.get("log_model", False),
            save_dir=output_dir,
        )
        loggers.append(wandb_logger)
    
    return loggers


def setup_trainer(conf: DictConfig, callbacks, loggers, output_dir: Path):
    """Setup PyTorch Lightning trainer."""
    
    trainer_kwargs = {
        "accelerator": conf.get("accelerator", "gpu"), # deprecated in v1.5.0
        "devices": conf.get("devices", 1),
        "precision": conf.get("precision", 32),
        "callbacks": callbacks,
        "logger": loggers,
        "enable_progress_bar": conf.get("enable_progress_bar", True),
        "enable_checkpointing": conf.get("enable_checkpointing", True),
        "log_every_n_steps": conf.get("log_every_n_steps", 50),
        "num_nodes": conf.get("num_nodes", 1),
        "num_sanity_val_steps": conf.get("num_sanity_val_steps", 2),
        "profiler": "simple" if conf.get("profiler", "simple") == "simple" else None,
        "accumulate_grad_batches": conf.get("accumulate_grad_batches", 1),
    }
    
    trainer_kwargs["max_steps"] = conf.get("num_training_steps", -1)
    
    if trainer_kwargs["max_steps"] == -1:
        trainer_kwargs["max_epochs"] = conf.get("max_epochs", 1000)   
    # Add gradient clipping if specified
    if conf.get("gradient_clip_val"):
        trainer_kwargs["gradient_clip_val"] = conf.gradient_clip_val
    
    # Add strategy if specified
    if conf.get("strategy"):
        trainer_kwargs["strategy"] = conf.strategy
        
    if conf.get("data_type") == "float64":
        trainer_kwargs["precision"] = 64
    
    trainer = pl.Trainer(**trainer_kwargs)
    return trainer


def setup_warmup_trainer(conf: DictConfig, callbacks, loggers, output_dir: Path):
    """Setup PyTorch Lightning trainer."""
    
    trainer_kwargs = {
        "accelerator": conf.get("accelerator", "gpu"), # deprecated in v1.5.0
        "devices": conf.get("devices", 1),
        "precision": conf.get("precision", 32),
        "callbacks": callbacks,
        "logger": loggers,
        "enable_progress_bar": conf.get("enable_progress_bar", True),
        "enable_checkpointing": conf.get("enable_checkpointing", True),
        "log_every_n_steps": conf.get("log_every_n_steps", 50),
        "num_nodes": conf.get("num_nodes", 1),
        "num_sanity_val_steps": conf.get("num_sanity_val_steps", 2),
        "profiler": "simple" if conf.get("profiler", "simple") == "simple" else None,
        "accumulate_grad_batches": conf.get("accumulate_grad_batches", 1),
    }
    
    trainer_kwargs["max_steps"] = conf.get("warmup_step", 1000)

    # Add gradient clipping if specified
    # if conf.get("gradient_clip_val"):
    trainer_kwargs["gradient_clip_val"] = 10.0
    
    # Add strategy if specified
    if conf.get("strategy"):
        trainer_kwargs["strategy"] = conf.strategy
        
    if conf.get("data_type") == "float64":
        trainer_kwargs["precision"] = 64
    
    trainer = pl.Trainer(**trainer_kwargs)
    return trainer



class TFLOPSCallback(pl.Callback):
    """Measure and log TFLOPS to wandb during training.

    전략:
      1) on_fit_start: FlopCounterMode로 forward FLOPs 1회 측정 (deterministic)
      2) 매 step: cuda.Event로 GPU 시간 측정 → TFLOPS 계산
      3) wandb에 perf/tflops, perf/gpu_ms, perf/mfu 로그

    FLOPs = forward only. TFLOPS = (3 × FLOPs) / GPU시간.
    3× = forward(1) + backward(2)의 standard approximation.

    Args:
        log_interval: 몇 step마다 로깅할지 (default: 50)
        peak_tflops: GPU peak TFLOPS (MFU 계산용, default: H200 TF32 = 494.7)
    """

    def __init__(self, log_interval: int = 50, peak_tflops: float = 494.7):
        super().__init__()
        self.log_interval = log_interval
        self.peak_tflops = peak_tflops
        self._flops_fwd: int | None = None
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self._measured = False

    def on_fit_start(self, trainer, pl_module):
        """모델의 forward FLOPs를 1회 측정."""
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ImportError:
            logger.warning("FlopCounterMode unavailable (PyTorch 2.1+ required). TFLOPS logging disabled.")
            return

        device = next(pl_module.parameters()).device
        model = pl_module.model

        # 첫 번째 배치를 가져와서 FLOPs 측정
        try:
            batch = next(iter(trainer.train_dataloader))
            batch = batch.to(device)
            batch = pl_module.post_processing(batch, pl_module.default_type)
            batch = pl_module.corrupt(batch, mul=1)

            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                with torch.no_grad():
                    model(batch, batch.init_ham_t, keep_blocks=True)
            self._flops_fwd = flop_counter.get_total_flops()
            self._measured = True
            logger.info(f"[TFLOPSCallback] Forward FLOPs: {self._flops_fwd / 1e9:.2f} GFLOPs")
        except Exception as e:
            logger.warning(f"[TFLOPSCallback] FLOP measurement failed: {e}")
            self._measured = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self._measured or batch_idx % self.log_interval != 0:
            return
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._measured or self._start_event is None:
            return
        if batch_idx % self.log_interval != 0:
            return

        self._end_event.record()
        torch.cuda.synchronize()
        gpu_ms = self._start_event.elapsed_time(self._end_event)
        gpu_s = gpu_ms / 1000.0

        # 3× forward = forward + backward approximation
        flops_total = self._flops_fwd * 3
        tflops = (flops_total / 1e12) / gpu_s if gpu_s > 0 else 0
        mfu = (tflops / self.peak_tflops) * 100 if self.peak_tflops > 0 else 0

        pl_module.log("perf/tflops", tflops, on_step=True, on_epoch=False, prog_bar=False)
        pl_module.log("perf/gpu_ms", gpu_ms, on_step=True, on_epoch=False, prog_bar=False)
        pl_module.log("perf/mfu_pct", mfu, on_step=True, on_epoch=False, prog_bar=False)
        pl_module.log("perf/gflops_fwd", self._flops_fwd / 1e9, on_step=False, on_epoch=True, prog_bar=False)

        self._start_event = None


def log_training_config(conf: DictConfig):
    """Log training configuration."""
    logger.info("Training Configuration:")
    logger.info(f"  Max steps: {conf.get('num_training_steps', -1)}")
    if conf.get("num_training_steps", -1) == -1:
        logger.info(f"  Max epochs: {conf.get('max_epochs', 1000)}")
    # logger.info(f"  Accelerator: {conf.get('accelerator', 'auto')}")
    logger.info(f"  Strategy: {conf.get('strategy', "None")}")
    logger.info(f"  Devices: {conf.get('devices', 1)}")
    logger.info(f"  Data type: {conf.get('data_type', 'float32')}")
