# SPDX-FileCopyrightText: 2026 Seongsu Kim
# SPDX-License-Identifier: MIT
"""Dedicated QH9Stable entrypoint; the legacy entrypoints remain available."""
import hydra


def validate_config(conf):
    if conf.dataset.dataset_name != "QH9Stable":
        raise ValueError("Muon benchmark preparation is restricted to QH9Stable")
    if conf.dataset.split != "random":
        raise ValueError("This benchmark configuration uses the paired QH9Stable random split")
    if conf.model.version not in {"QHFlow2_order_fixed", "QHFlow2_O3"}:
        raise ValueError("Select one of the new QHFlow2 / QHFlow2_O3 model configurations")
    if conf.optimizer != "muon_adamw":
        raise ValueError("This entrypoint requires the split Muon/AdamW optimizer")
    if not conf.dataset.get("processed_directory") or not conf.dataset.get("split_file"):
        raise ValueError("Supply explicit existing QH9Stable processed_directory and split_file")
    if not conf.dataset.get("compute_q_tensor", True) or not conf.dataset.get("include_cut_orbital_coefficients", True):
        active_losses = {name for name, weight in conf.loss_weights.items() if weight}
        if active_losses != {"hamiltonian"}:
            raise ValueError("Lean benchmark loading requires the Hamiltonian-only objective")


@hydra.main(version_base=None, config_path="../config_qh9", config_name="config_stable_muon")
def main(conf):
    validate_config(conf)
    from experiment.train_qh9 import main as train
    train.__wrapped__(conf)


if __name__ == "__main__":
    main()
