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


@pytest.mark.parametrize('scope', ['heads', 'blocks'])
def test_full_model_regions_do_not_specialize_atom_count(scope):
    from torch_geometric.data import Data
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    model = QHFlow2_order_fixed(hidden_size=4, bottle_hidden_size=2,
        num_gnn_layers=1, num_ham_gnn_layers=1, use_block_H=True, use_block_S=True)
    compile_flow(SimpleNamespace(model=model, expand=nn.Identity()), scope, backend='eager')
    counts=[]
    for atoms in (7, 9, 11, 13):
        h=torch.randn(atoms,14,14)
        data=Data(pos=torch.randn(atoms,3), atoms=torch.full((atoms,1),6),
            batch=torch.zeros(atoms,dtype=torch.long), ptr=torch.tensor([0,atoms]),
            natoms=torch.tensor([atoms]),cell=torch.zeros(1,3,3),t=torch.rand(atoms),
            diagonal_init_ham=torch.randn_like(h),diagonal_overlap=torch.randn_like(h))
        output=model(data,h,keep_blocks=True)
        sum(value.square().mean() for key,value in output.items() if key.startswith('hamiltonian')).backward()
        counts.append(torch._dynamo.utils.counters['stats']['unique_graphs'])
    assert counts[0] > 0
    assert len(set(counts)) == 1, counts
