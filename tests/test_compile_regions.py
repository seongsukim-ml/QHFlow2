"""Compiler boundaries preserve model state, gradients and dynamic shapes."""
import copy
import random
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from test_equivariant_models import graph, QHFlow2_order_fixed, QHFlow2_O3
from common.compile import compile_flow


@pytest.mark.parametrize('cls', [QHFlow2_order_fixed, QHFlow2_O3])
def test_fullgraph_regions_preserve_state_and_gradients(cls):
    torch.manual_seed(23)
    reference = cls(hidden_size=4, bottle_hidden_size=2, num_gnn_layers=1,
                    num_ham_gnn_layers=1, use_block_H=True, use_block_S=True).train()
    candidate = copy.deepcopy(reference)
    keys = list(candidate.state_dict())
    compile_flow(SimpleNamespace(model=candidate, expand=nn.Identity()), 'blocks', backend='eager')
    data, h = graph(torch.float32)
    torch.manual_seed(101); random.seed(101)
    expected = reference(data, h, keep_blocks=True)
    torch.manual_seed(101); random.seed(101)
    actual = candidate(data, h, keep_blocks=True)
    assert list(candidate.state_dict()) == keys
    for key in ('hamiltonian_diagonal_blocks', 'hamiltonian_non_diagonal_blocks'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    sum(v.square().mean() for k,v in expected.items() if k.startswith('hamiltonian')).backward()
    sum(v.square().mean() for k,v in actual.items() if k.startswith('hamiltonian')).backward()
    for a,b in zip(reference.parameters(), candidate.parameters()):
        if a.grad is None: assert b.grad is None
        else: torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    reference.load_state_dict(candidate.state_dict(), strict=True)
    layer = candidate.node_attr_backbone.blocks[0].atom_wise
    for nodes in (5, 7, 9):
        x = torch.randn(nodes, 25, 4, requires_grad=True)
        torch.compile(layer, backend='eager', fullgraph=True, dynamic=True)(x).sum().backward()
        assert torch.isfinite(x.grad).all()
