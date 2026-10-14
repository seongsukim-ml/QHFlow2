# SPDX-FileCopyrightText: 2026 Seongsu Kim
# SPDX-License-Identifier: MIT
"""Resolve all eight configs and optionally exercise one CPU training batch.

This command prepares reports/configs. It never starts a benchmark training run.
"""
import argparse
import gc
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from common.muon import MuonAdamW
from experiment.train_qh9_stable_muon import validate_config
from qhflow2_version import get_code_metadata

MODEL_CONFIGS = [f"QHFlow2{family}_{size}" for family in ("", "_O3") for size in ("s", "m", "l", "xl")]


def prepare(output, models, smoke=False):
    from models import get_model
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("Use an explicit absolute preparation output path")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    report = {"qhflow2": get_code_metadata(), "torch": torch.__version__,
              "models": [], "full_training_launched": False}
    dataset = None
    try:
        with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "config_qh9")):
            for name in models:
                conf = compose(config_name="config_stable_muon", overrides=[f"model={name}"])
                validate_config(conf)
                OmegaConf.save(conf, output / f"{name}.yaml", resolve=True)
                torch.manual_seed(conf.seed)
                if smoke:
                    # The supplied runtime must support CPU execution, even on
                    # hosts with GPUs. This preflight deliberately uses no GPU.
                    if torch.cuda.is_available():
                        raise RuntimeError("Run CPU preflight with CUDA_VISIBLE_DEVICES='' ")
                    from common.qh9_utils import load_qh9_dataset
                    from pl_module.flow_module import LitModel_flow
                    from torch_geometric.data import Batch
                    if dataset is None:
                        dataset = load_qh9_dataset(conf, str(Path(__file__).resolve().parents[2]))
                        report["dataset"] = {
                            "processed_directory": conf.dataset.processed_directory,
                            "split_file": conf.dataset.split_file,
                            "rows": len(dataset), "train": len(dataset.train_mask),
                            "val": len(dataset.val_mask), "test": len(dataset.test_mask),
                        }
                    lit = LitModel_flow(conf)
                    optimizer_list, scheduler_list = lit.configure_optimizers()
                    optimizer = optimizer_list[0]
                    batch = Batch.from_data_list([dataset[int(i)] for i in dataset.train_mask[:2]])
                    # Logging requires a Trainer. The training computation and
                    # corruption/loss path are otherwise the real implementation.
                    lit.log = lambda *args, **kwargs: None
                    loss = lit.training_step(batch, 0)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite training loss for {name}")
                    loss.backward()
                    for parameter in lit.model.parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise RuntimeError(f"Nonfinite training gradient for {name}")
                    rates = [g["lr"] for g in optimizer.param_groups]
                    optimizer.step()
                    scheduler_list[0]["scheduler"].step()
                    model = lit.model
                else:
                    model = get_model(conf.model)
                    optimizer = MuonAdamW(model, **dict(conf.muon))
                record = {
                    "config": name, "model_name": conf.model.model_name,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "optimizer": optimizer.parameter_summary(),
                }
                if smoke:
                    record.update(cpu_training_loss=float(loss.detach()), initial_learning_rates=rates,
                                  smoke="two QH9Stable train samples; backward + optimizer + scheduler step")
                report["models"].append(record)
                (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: v for k, v in record.items() if k != "optimizer"}), flush=True)
                if smoke:
                    del lit, loss, batch, optimizer_list, scheduler_list
                del model, optimizer
                gc.collect()
    finally:
        if dataset is not None:
            dataset.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_CONFIGS, default=MODEL_CONFIGS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare(args.output, args.models, args.smoke)


if __name__ == "__main__":
    main()
