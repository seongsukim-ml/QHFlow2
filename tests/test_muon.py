# SPDX-FileCopyrightText: 2026 Seongsu Kim
# SPDX-License-Identifier: MIT
import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.muon import MuonAdamW, partition_parameters

pytestmark = pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="requires native Muon")


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 4)
        self.embedding = nn.Embedding(3, 4)
        self.norm = nn.LayerNorm(4)
        self.matrix_embedding = nn.Linear(4, 4)
        self.packed = nn.Parameter(torch.randn(12))
        self.frozen = nn.Parameter(torch.ones(2, 2), requires_grad=False)


def test_partition_is_disjoint_complete_and_excludes_norm_embeddings():
    model = Toy()
    groups, names = partition_parameters(model)
    assert names["muon"] == ["hidden.weight"]
    assert set(names["adamw"]) == {n for n, p in model.named_parameters() if p.requires_grad} - {"hidden.weight"}
    assert len({id(p) for g in groups.values() for p in g}) == sum(p.requires_grad for p in model.parameters())


def set_gradients(model):
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.linspace(-0.7, 0.9, p.numel()).reshape_as(p)


def test_updates_match_native_optimizers_and_resume_with_scheduler():
    torch.manual_seed(3)
    model = Toy()
    reference = copy.deepcopy(model)
    optimizer = MuonAdamW(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.8 ** step)
    groups, _ = partition_parameters(reference)
    muon = torch.optim.Muon(groups["muon"], lr=1e-3, weight_decay=1e-2, adjust_lr_fn="original")
    adamw = torch.optim.AdamW(groups["adamw"], lr=1e-4, weight_decay=1e-2, betas=(0.99, 0.999))
    set_gradients(model)
    set_gradients(reference)
    optimizer.step()
    muon.step()
    adamw.step()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    scheduler.step()
    restored_model = copy.deepcopy(model)
    restored = MuonAdamW(restored_model)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored, lambda step: 0.8 ** step)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored_scheduler.load_state_dict(scheduler.state_dict())
    for _ in range(2):
        set_gradients(model)
        set_gradients(restored_model)
        optimizer.step()
        restored.step()
        scheduler.step()
        restored_scheduler.step()
    assert [g["lr"] for g in restored.param_groups] == [g["lr"] for g in optimizer.param_groups]
    for actual, expected in zip(restored_model.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_closure_runs_once_and_both_groups_receive_gradients():
    model = Toy()
    optimizer = MuonAdamW(model)
    calls = []
    def closure():
        calls.append(1)
        optimizer.zero_grad()
        loss = sum(p.square().sum() for p in model.parameters() if p.requires_grad)
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    assert len(calls) == 1 and torch.isfinite(loss)
    assert all(p in optimizer.state for g in optimizer.param_groups for p in g["params"])
