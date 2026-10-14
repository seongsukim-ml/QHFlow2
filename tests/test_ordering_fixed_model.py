# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

"""Integration at both backbone/head boundaries; requires fairchem-core."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from e3nn import o3
from torch_geometric.data import Data

pytest.importorskip("fairchem.core")


def test_separate_registry_and_both_head_paths(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    from models import get_model
    from models.QHFlow_so2_v5_1 import QHFlow_escn_v5_1
    from models.QHFlow_so2_v5_1_ordering_fixed import QHFlow_escn_v5_1_ordering_fixed

    torch.manual_seed(42)
    args = dict(hidden_size=4, bottle_hidden_size=2, num_gnn_layers=1,
                num_ham_gnn_layers=1, use_block_H_fix=False, use_block_S=False)
    legacy = get_model(SimpleNamespace(version="QHFlow_so2_v5_1", **args)).double().eval()
    fixed = get_model(SimpleNamespace(version="QHFlow_so2_v5_1_ordering_fixed", **args)).double().eval()
    assert type(legacy) is QHFlow_escn_v5_1
    assert type(fixed) is QHFlow_escn_v5_1_ordering_fixed
    fixed.load_state_dict(legacy.state_dict(), strict=True)

    n = 4
    pos = torch.randn(n, 3, dtype=torch.float64)
    h = torch.eye(14, dtype=pos.dtype).repeat(n, 1, 1)
    node = torch.randn(n, 25, 4, dtype=pos.dtype)
    pair = torch.randn(n * (n - 1), 25, 4, dtype=pos.dtype)
    rotation = o3.rand_matrix(dtype=pos.dtype)
    native_d = o3.Irreps([(1, (l, 1)) for l in range(5)]).D_from_matrix(rotation)
    packed_d = fixed.hidden_irrep.D_from_matrix(rotation)
    output_d = fixed.output_irrep.D_from_matrix(rotation)

    def graph(positions):
        return Data(pos=positions, atoms=torch.tensor([[6], [7], [8], [6]]),
                    batch=torch.zeros(n, dtype=torch.long), ptr=torch.tensor([0, n]))

    # Prescribe exactly equivariant backbone outputs so grid approximations do
    # not hide a regression in either of the two layout-conversion call sites.
    state = {"node_embedding": node, "xy_embedding": pair}
    for model in (legacy, fixed):
        monkeypatch.setattr(model.node_attr_backbone, "forward", lambda *a, **k: state)

    with torch.no_grad():
        old = legacy(graph(pos), h, keep_blocks=True)
        base = fixed(graph(pos), h, keep_blocks=True)
        torch.testing.assert_close(old["node_attr"], node.flatten(1), rtol=0, atol=0)
        state.update(
            node_embedding=torch.einsum("ij,njc->nic", native_d, node),
            xy_embedding=torch.einsum("ij,njc->nic", native_d, pair),
        )
        actual = fixed(graph(pos @ rotation.T), h, keep_blocks=True)
        torch.testing.assert_close(actual["node_attr"], base["node_attr"] @ packed_d.T,
                                   rtol=1e-9, atol=1e-9)
        for key in ("hamiltonian_diagonal_blocks", "hamiltonian_non_diagonal_blocks"):
            torch.testing.assert_close(actual[key], output_d @ base[key] @ output_d.T,
                                       rtol=1e-5, atol=1e-7)
            assert torch.isfinite(actual[key]).all()
