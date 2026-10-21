"""v0.2.0-rc.2 training/re-evaluation with exact, unpadded evaluation shards.

The released weights are for inference. For exact continuation, use a full
training checkpoint with optimizer/loop state and the original device layout.
"""
from pathlib import Path
import hydra
from omegaconf import OmegaConf


def evaluation_indices(length, rank, world_size):
    return range(rank, length, world_size)


@hydra.main(version_base=None, config_path="../config_qh9", config_name="config_release_s")
def main(conf):
    from experiment.train_qh9_stable_muon import validate_config
    validate_config(conf)
    if conf.dataset.batch_size * conf.devices * conf.accumulate_grad_batches != 32:
        raise ValueError("Release benchmark uses global batch 32")
    if conf.devices > 1 and conf.flow.error_threshold is not None:
        raise ValueError("DDP evaluation requires fixed ODE steps")
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DistributedSampler, Subset
    from torch_geometric.loader import DataLoader
    from common.qh9_utils import load_qh9_dataset
    from common.setup import setup_tensor_type_and_seed
    from common.compile import compile_from_config
    from pl_module.flow_module import LitModel_flow

    class Data(pl.LightningDataModule):
        def setup(self, stage=None):
            if not hasattr(self, "dataset"):
                self.dataset = load_qh9_dataset(conf, str(Path(__file__).resolve().parents[2]))

        def loader(self, split):
            rank, world = self.trainer.global_rank, self.trainer.world_size
            data = self.dataset[getattr(self.dataset, split + "_mask")]
            options = dict(num_workers=conf.dataset.num_workers, pin_memory=conf.dataset.pin_memory,
                           persistent_workers=conf.dataset.persistent_workers and conf.dataset.num_workers > 0)
            if split == "train":
                sampler = DistributedSampler(data, num_replicas=world, rank=rank, seed=conf.seed, shuffle=True)
                return DataLoader(data, batch_size=conf.dataset.batch_size, sampler=sampler, **options)
            batch_size = conf.dataset.valid_batch_size if split == "val" else conf.dataset.test_batch_size
            # No repeated tail molecules in the published test metrics.
            return DataLoader(Subset(data, evaluation_indices(len(data), rank, world)),
                              batch_size=batch_size, shuffle=False, **options)

        def train_dataloader(self):
            return self.loader("train")

        def val_dataloader(self):
            return self.loader("val")

        def test_dataloader(self):
            return self.loader("test")

    class ReleaseFlow(LitModel_flow):
        def training_step(self, batch, batch_idx):
            self.actual_graphs = batch.num_graphs
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            self.actual_graphs = batch.num_graphs
            return super().validation_step(batch, batch_idx)

        def test_step(self, batch, batch_idx):
            self.actual_graphs = batch.num_graphs
            return super().test_step(batch, batch_idx)

        def log(self, name, value, *args, **kwargs):
            if "batch_size" in kwargs:
                kwargs["batch_size"] = self.actual_graphs
            if name.startswith(("val/", "test/")):
                kwargs.update(on_step=False, on_epoch=True, sync_dist=True)
            return super().log(name, value, *args, **kwargs)

        def _log_sample_metric(self, batch_one, prefix, num_timesteps=1, post_fix="", save_pred=False, log=True, metric_cache=None):
            sample, traj, _ = self.sample(batch_one, num_timesteps=num_timesteps, return_trajectory=save_pred)
            if log:
                metrics = self.metric(sample, batch_one, metric_weights=self.metric_weights, qh9_cache=metric_cache)
                for key, value in metrics.items():
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f"Nonfinite metric: {key}")
                    self.log(f"{prefix}/sample_{key}{post_fix}", value, on_step=False,
                             on_epoch=True, sync_dist=True, batch_size=self.actual_graphs)
            if save_pred:
                return traj, sample

    setup_tensor_type_and_seed(conf)
    torch.set_num_threads(8)
    output = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    module = ReleaseFlow(conf)
    compile_from_config(module, conf)
    data = Data()
    callback = ModelCheckpoint(dirpath=output / "checkpoints", monitor="val/loss", mode="min", save_top_k=1, save_last=True)
    trainer = pl.Trainer(accelerator="gpu", devices=conf.devices, strategy=conf.strategy or "auto",
        max_steps=conf.num_training_steps, precision=conf.precision, inference_mode=False,
        accumulate_grad_batches=conf.accumulate_grad_batches, gradient_clip_val=conf.gradient_clip_val,
        use_distributed_sampler=False, callbacks=[callback], logger=CSVLogger(output, name="metrics"),
        num_sanity_val_steps=0)
    if trainer.is_global_zero:
        OmegaConf.save(conf, output / "resolved-config.yaml", resolve=True)
    if conf.mode == "test":
        if not conf.ckpt:
            raise ValueError("mode=test requires ckpt")
        trainer.test(module, datamodule=data, ckpt_path=conf.ckpt)
    elif conf.mode == "train":
        trainer.fit(module, datamodule=data, ckpt_path=conf.ckpt or None)
        trainer.test(module, datamodule=data, ckpt_path="best")
    else:
        raise ValueError("train_release supports mode=train or mode=test")


if __name__ == "__main__":
    main()
