# SPDX-FileCopyrightText: 2026 Seongsu Kim
# SPDX-License-Identifier: MIT
"""Native PyTorch Muon and AdamW as one Lightning-compatible optimizer."""

import torch


def partition_parameters(model):
    """Use Muon for 2-D weights except norm/embedding subtrees.

    Packed e3nn weights, biases, scalar gains and other non-2-D tensors use
    AdamW. Shared parameters occur once; any norm/embedding ownership wins.
    """
    excluded = set()
    for name, module in model.named_modules():
        label = f"{name}.{type(module).__name__}".lower()
        if "norm" in label or "embed" in label:
            excluded.update(id(p) for p in module.parameters())
    groups = {"muon": [], "adamw": []}
    names = {"muon": [], "adamw": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        algorithm = "muon" if parameter.ndim == 2 and id(parameter) not in excluded else "adamw"
        groups[algorithm].append(parameter)
        names[algorithm].append(name)
    return groups, names


class MuonAdamW(torch.optim.Optimizer):
    """Share parameter groups and state with native optimizers.

    A single optimizer lets Lightning unscale/clip all gradients and run one
    closure per batch. Scheduler updates and checkpoints include both groups.
    """

    def __init__(self, model, lr=1e-3, adamw_lr=1e-4, weight_decay=1e-2,
                 momentum=0.95, ns_steps=5, adamw_betas=(0.99, 0.999),
                 adjust_lr_fn="original", batched=False):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("Muon training requires a PyTorch build with torch.optim.Muon (>=2.9)")
        groups, names = partition_parameters(model)
        if not groups["muon"] or not groups["adamw"]:
            raise ValueError("Expected both Muon matrix weights and AdamW parameters")
        super().__init__([
            {"params": groups["muon"], "param_names": names["muon"], "algorithm": "muon", "lr": lr},
            {"params": groups["adamw"], "param_names": names["adamw"], "algorithm": "adamw", "lr": adamw_lr},
        ], defaults={"lr": lr})
        self._muon = torch.optim.Muon(
            [self.param_groups[0]], lr=lr, weight_decay=weight_decay,
            momentum=momentum, ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn,
        )
        self._adamw = torch.optim.AdamW(
            [self.param_groups[1]], lr=adamw_lr, weight_decay=weight_decay,
            betas=tuple(adamw_betas),
        )
        self._bind_state()
        self.batched = batched

    def _bind_state(self):
        self._muon.param_groups = [g for g in self.param_groups if g["algorithm"] == "muon"]
        self._adamw.param_groups = [g for g in self.param_groups if g["algorithm"] == "adamw"]
        self._muon.state = self.state
        self._adamw.state = self.state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._bind_state()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.batched:
            from .batched_muon import batched_muon_step
            for group in self._muon.param_groups:
                batched_muon_step(group, self.state)
        else:
            self._muon.step()
        self._adamw.step()
        return loss

    def parameter_summary(self):
        return [{"algorithm": group["algorithm"], "lr": group["lr"],
                 "tensors": len(group["params"]),
                 "parameters": sum(p.numel() for p in group["params"]),
                 "names": list(group["param_names"])} for group in self.param_groups]
