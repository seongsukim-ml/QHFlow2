import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from torch_geometric.data import Data
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common.inference_cache import inference_cache_scope, cached_tensor_metadata, _CACHE
from common.matrix_transforms import _build_final_matrix
from models.QHFlow2_equivariant import QHFlow2_order_fixed, QHFlow2_O3
from models.utils import get_orbital_mask
from test_equivariant_models import graph, rotation_in_ao_basis
from e3nn import o3


@pytest.fixture(autouse=True)
def double_defaults():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def test_cache_invalidates_mutation_and_cleans_up_exception():
    x = torch.ones(3)
    with pytest.raises(ValueError), inference_cache_scope():
        a, hit = cached_tensor_metadata('test', (x,), lambda: x.square())
        assert not hit
        assert cached_tensor_metadata('test', (x,), lambda: None)[1]
        x.add_(1)
        b, hit = cached_tensor_metadata('test', (x,), lambda: x.square())
        assert not hit
        torch.testing.assert_close(b, 4*a)
        raise ValueError()
    assert _CACHE.get() is None


@pytest.mark.parametrize('cls', [QHFlow2_order_fixed, QHFlow2_O3])
def test_inference_reuses_geometry_preserves_rng_and_invalidates_positions(cls):
    torch.manual_seed(13)
    model = cls(hidden_size=4, bottle_hidden_size=2, num_gnn_layers=1,
                num_ham_gnn_layers=1, use_block_S=True, use_block_H=True).double().eval()
    data, h = graph(torch.float64)
    with torch.no_grad():
        torch.manual_seed(15)
        expected = [model(data, h, keep_blocks=True) for _ in range(2)]
        expected_rng = torch.get_rng_state()
        torch.manual_seed(15)
        import models.QHFlow_so2_v5_1 as parent
        with inference_cache_scope(), patch.object(parent, 'radius_graph', wraps=parent.radius_graph) as radius:
            a = model(data, h, keep_blocks=True)
            calls = radius.call_count
            b = model(data, h, keep_blocks=True)
            assert radius.call_count == calls
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
            for key in ('hamiltonian_diagonal_blocks','hamiltonian_non_diagonal_blocks'):
                torch.testing.assert_close(a[key], expected[0][key], atol=1e-9, rtol=1e-9)
                torch.testing.assert_close(b[key], expected[1][key], atol=1e-9, rtol=1e-9)
            data.pos[0].add_(.1)
            model(data, h, keep_blocks=True)
            assert radius.call_count > calls
    assert _CACHE.get() is None


def test_matrix_indices_reuse_and_different_values_keep_autograd():
    data = Data(ptr=torch.tensor([0,2]), atoms=torch.tensor([6,1]),
                full_edge_index=torch.tensor([[0,1],[1,0]]))
    masks = get_orbital_mask('def2-svp')
    diag = torch.randn(2,14,14,requires_grad=True)
    off = torch.randn(2,14,14,requires_grad=True)
    with inference_cache_scope():
        first = _build_final_matrix(data,diag,off,masks)
        with patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('rebuilt indices')):
            second = _build_final_matrix(data,diag*2,off*2,masks)
        torch.testing.assert_close(second[0],first[0]*2,atol=0,rtol=0)
        second[0].square().sum().backward()
    assert torch.isfinite(diag.grad).all() and torch.isfinite(off.grad).all()


def test_paired_o3_outputs_parameter_gradients_and_equivariance():
    torch.manual_seed(17)
    model=QHFlow2_O3(hidden_size=4,bottle_hidden_size=2,num_gnn_layers=1,
        num_ham_gnn_layers=1,use_block_S=True,use_block_H=True).double().eval()
    data,h=graph(torch.float64)
    keys=list(model.state_dict())
    expected=model(data,h,keep_blocks=True)
    sum(expected[k].square().sum() for k in expected if k.startswith('hamiltonian')).backward()
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True); model.batch_branches=True
    actual=model(data,h,keep_blocks=True)
    sum(actual[k].square().sum() for k in actual if k.startswith('hamiltonian')).backward()
    for key in ('hamiltonian_diagonal_blocks','hamiltonian_non_diagonal_blocks'):
        torch.testing.assert_close(actual[key],expected[key],atol=1e-9,rtol=1e-9)
    for n,p in model.named_parameters():
        if n in grads: torch.testing.assert_close(p.grad,grads[n],atol=1e-9,rtol=1e-8,msg=n)
    assert keys==list(model.state_dict())
    r=-o3.rand_matrix(dtype=torch.float64);d=rotation_in_ao_basis(r)
    rotated=data.clone();rotated.pos=data.pos@r.T
    for k in ('diagonal_init_ham','diagonal_overlap'):rotated[k]=d@data[k]@d.T
    with torch.no_grad(),inference_cache_scope():
        result=model(rotated,d@h@d.T,keep_blocks=True)
    for key in ('hamiltonian_diagonal_blocks','hamiltonian_non_diagonal_blocks'):
        torch.testing.assert_close(result[key],d@actual[key]@d.T,atol=1e-9,rtol=1e-9)


def test_validation_passes_target_spectrum_cache_to_metrics():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from pl_module.flow_module import LitModel_flow
    cache = {}
    batch = object()
    module = SimpleNamespace(sample=lambda *a, **k: ({}, [], []), metric=Mock(return_value={}),
                             metric_weights={}, log=Mock())
    LitModel_flow._log_sample_metric(module, batch, 'val', metric_cache=cache)
    assert module.metric.call_args.kwargs['qh9_cache'] is cache


def test_paired_pyg_batch_without_periodic_cell():
    from torch_geometric.data import Batch
    first, h = graph(torch.float64)
    for key in ('batch', 'ptr', 'cell'):
        del first[key]
    data = Batch.from_data_list([first, first])
    h = torch.cat((h,h))
    model = QHFlow2_O3(hidden_size=4,bottle_hidden_size=2,num_gnn_layers=1,
                      num_ham_gnn_layers=1).eval()
    with torch.no_grad(), inference_cache_scope():
        expected = model(data,h,keep_blocks=True)
        model.batch_branches=True
        actual = model(data,h,keep_blocks=True)
        repeated = model(data,h,keep_blocks=True)
    for key in ('hamiltonian_diagonal_blocks','hamiltonian_non_diagonal_blocks'):
        torch.testing.assert_close(actual[key],expected[key],atol=1e-9,rtol=1e-9)
        torch.testing.assert_close(repeated[key],actual[key],atol=0,rtol=0)
    assert data.num_graphs == 2


@pytest.mark.parametrize('convention', ['back2pyscf','pyscf_def2svp'])
def test_cached_ao_convention_plan_matches_reference_and_gradients(convention):
    from common.matrix_transforms import _matrix_transform_list, _matrix_transform_single, get_convention_dict
    atoms=torch.tensor([6,1,8,1,7]); batch=torch.tensor([0,0,0,1,1])
    data=Data(atoms=atoms,batch=batch,ptr=torch.tensor([0,3,5]))
    conv=get_convention_dict()[convention]
    matrices=[torch.randn(n,n,requires_grad=True) for n in (33,19)]
    expected=[_matrix_transform_single(m,atoms[batch==i],conv) for i,m in enumerate(matrices)]
    from types import SimpleNamespace
    from pl_module.base_module import LitModel
    module=SimpleNamespace(convention_dict=get_convention_dict())
    integrated=LitModel.matrix_transform_list(module,matrices,data,convention)
    for a,b in zip(integrated,expected):torch.testing.assert_close(a,b,atol=0,rtol=0)
    with inference_cache_scope():
        actual=_matrix_transform_list(matrices,data,conv)
        with patch.object(torch.Tensor,'tolist',side_effect=AssertionError('rebuilt convention')):
            repeated=_matrix_transform_list(matrices,data,conv)
    for a,b,c in zip(actual,expected,repeated):
        torch.testing.assert_close(a,b,atol=0,rtol=0);torch.testing.assert_close(a,c,atol=0,rtol=0)
    ga=torch.autograd.grad(sum(m.square().sum() for m in actual),matrices)
    gb=torch.autograd.grad(sum(m.square().sum() for m in expected),matrices)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=0,rtol=0)
