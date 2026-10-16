"""Numerical and checkpoint contracts for the compute-only changes."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from e3nn import o3
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.matrix_transforms import _build_final_matrix
from models.utils import get_orbital_mask
from models.layers_v2 import PureExpansion, ParamContraction
from models.QHFlow2_equivariant import QHFlow2_order_fixed
from models.QHFlow_so2_v5_1 import QHFlow_escn_v5_1


def test_matrix_assembly_values_and_gradients_with_shuffled_edges():
    ptr = torch.tensor([0, 3, 5, 6])
    edges = torch.tensor([(d, s) for lo, hi in zip(ptr[:-1], ptr[1:])
                          for s in range(lo, hi) for d in range(lo, hi) if d != s]).T
    edges = edges[:, torch.randperm(edges.shape[1])]
    data = Data(ptr=ptr, atoms=torch.tensor([6, 1, 8, 1, 7, 1]), full_edge_index=edges)
    diag = torch.randn(6, 14, 14, dtype=torch.float64, requires_grad=True)
    off = torch.randn(edges.shape[1], 14, 14, dtype=torch.float64, requires_grad=True)
    expected = _build_final_matrix(data, diag, off, get_orbital_mask('def2-svp'), version='legacy')
    actual = _build_final_matrix(data, diag, off, get_orbital_mask('def2-svp'))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    ga = torch.autograd.grad(sum(x.square().sum() for x in actual), (diag, off))
    ge = torch.autograd.grad(sum(x.square().sum() for x in expected), (diag, off))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_reverse_edges_matches_existing_graph_order():
    ptr = torch.tensor([0, 1, 4, 9])
    edges = torch.tensor([(d, s) for lo, hi in zip(ptr[:-1], ptr[1:])
                          for s in range(lo, hi) for d in range(lo, hi) if d != s]).T
    data = Data(ptr=ptr, batch=torch.repeat_interleave(torch.arange(3), ptr.diff()))
    expected = QHFlow_escn_v5_1._calculate_transpose_indices(data, edges)
    actual = QHFlow2_order_fixed._calculate_transpose_indices(data, edges)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(edges[:, actual], edges.flip(0))


def test_couplings_follow_dtype_without_checkpoint_keys_or_forward_rebuild():
    ir = o3.Irreps('3x0e+2x1e+1x2e')
    layer = ParamContraction(ir, ir, o3.Irreps('4x0e')).double()
    state = layer.state_dict()
    assert set(state) == {'path_weights', 'bias_weights'}
    assert all(b.dtype == torch.float64 for b in layer.buffers())
    restored = ParamContraction(ir, ir, o3.Irreps('4x0e')).double()
    restored.load_state_dict(state, strict=True)
    x = torch.randn(3, 14, 14, dtype=torch.float64, requires_grad=True)
    with patch('e3nn.o3.wigner_3j', side_effect=AssertionError('forward rebuilt constant')):
        a = layer(x)
        b = restored(x)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('model', ['QHFlow2_s', 'QHFlow2_O3_s'])
def test_qh9_sampling_optional_latents_and_trajectory(model):
    from hydra import compose, initialize_config_dir
    from pl_module.flow_module import LitModel_flow
    config_dir = str(Path(__file__).resolve().parents[1] / 'src/config_qh9')
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        conf = compose(config_name='config_stable_muon', overrides=[
            f'model={model}', 'model.hidden_size=4', 'model.bottle_hidden_size=2',
            'model.num_gnn_layers=1', 'model.num_ham_gnn_layers=1'])
    module = LitModel_flow(conf).eval()
    n = 3
    data = Data(pos=torch.randn(n, 3), atoms=torch.tensor([[6], [7], [8]]),
                batch=torch.zeros(n, dtype=torch.long), ptr=torch.tensor([0, n]),
                natoms=torch.tensor([n]), cell=torch.zeros(1, 3, 3),
                diagonal_init_ham=torch.randn(n, 14, 14), diagonal_overlap=torch.randn(n, 14, 14),
                non_diagonal_init_ham=torch.randn(n * (n - 1), 14, 14))
    with torch.no_grad():
        torch.manual_seed(71)
        expected, trajectory, predictions = module.sample(data.clone(), num_timesteps=2)
        torch.manual_seed(71)
        actual, no_trajectory, no_predictions = module.sample(
            data.clone(), num_timesteps=2, return_trajectory=False)
    assert len(trajectory) == len(predictions) == 3
    assert no_trajectory == no_predictions == []
    for key in ('hamiltonian_diagonal_blocks', 'hamiltonian_non_diagonal_blocks'):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        assert torch.isfinite(actual[key]).all()
