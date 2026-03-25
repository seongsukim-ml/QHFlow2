"""
Unit experiment: QHNet 초기 forward pass에서 activation scale 추적

e3nn ScriptModule은 hook이 안 되므로, forward를 수동 분해하여
각 stage의 출력을 관찰.
"""

import torch
import torch.nn as nn
import numpy as np
from e3nn import o3
from torch_cluster import radius_graph

import sys
sys.path.insert(0, "src")
from qhflow2.models.Real_QHNet_qh9 import QHNet


def stat(t, name=""):
    """Print tensor statistics."""
    if t.isnan().any():
        nan_frac = t.isnan().float().mean().item() * 100
        return f"{name:<35} std={t[~t.isnan()].std().item() if (~t.isnan()).any() else float('nan'):>10.4f}  absmax={t[~t.isnan()].abs().max().item() if (~t.isnan()).any() else float('nan'):>10.2f}  NaN={nan_frac:.1f}%"
    return f"{name:<35} std={t.std().item():>10.4f}  absmax={t.abs().max().item():>10.2f}"


def test_qhnet_forward(hs, hbs, gnn, label):
    """QHNet forward를 수동으로 실행하면서 각 단계 activation 추적."""
    torch.manual_seed(42)

    model = QHNet(hidden_size=hs, bottle_hidden_size=hbs, num_gnn_layers=gnn)
    model.eval()
    device = torch.device("cpu")
    model.set(device)

    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'=' * 70}")
    print(f"QHNet {label}: hidden={hs}, bottle={hbs}, gnn={gnn}, params={n_params/1e6:.1f}M")
    print(f"{'=' * 70}")

    # Fake batch (4 x CH4-like molecules)
    from torch_geometric.data import Data, Batch
    molecules = []
    for _ in range(4):
        n = 5
        molecules.append(Data(
            atoms=torch.tensor([6, 1, 1, 1, 1]),
            pos=torch.randn(n, 3) * 1.5,
            num_nodes=n,
        ))
    data = Batch.from_data_list(molecules)

    with torch.no_grad():
        # Stage 1: Node embedding
        node_attr = model.node_embedding(data.atoms)
        print(stat(node_attr, "1. node_embedding"))

        # Stage 2: Edge construction
        edge_index = radius_graph(data.pos, r=model.max_radius, batch=data.batch)
        edge_vec = data.pos[edge_index[0]] - data.pos[edge_index[1]]
        edge_len = edge_vec.norm(dim=-1, keepdim=True)
        edge_attr = model.distance_expansion(edge_len)
        edge_sh = o3.spherical_harmonics(
            model.sh_irrep, edge_vec,
            normalize=True, normalization="component"
        )
        print(stat(edge_attr, "2. distance_expansion"))
        print(stat(edge_sh, "2. edge_sh"))

        # Stage 3: GNN layers (ConvNetLayer)
        x = model.input_irrep  # need to build input
        node_features = model.node_embedding(data.atoms)  # [N, hs] -> need to be [N, irrep_dim]
        # Actually QHNet input is just scalar: hs x 0e
        # node_features is already [N, hs]
        print(f"\n  --- ConvNetLayer iterations ---")

        # QHNet forward uses model.forward which does all this internally
        # Let's just call forward and observe the output
        try:
            output = model(data, keep_blocks=False)
            H_pred = output["hamiltonian"]
            print(f"\n  Forward pass: OK")
            print(stat(H_pred, "  Output H_pred"))
            print(f"  H_pred shape: {H_pred.shape}")
            return n_params, H_pred.std().item(), H_pred.abs().max().item(), H_pred.isnan().any().item()
        except Exception as e:
            # Try block output
            try:
                output = model(data, keep_blocks=True)
                diag = output["hamiltonian_diagonal_blocks"]
                off = output["hamiltonian_non_diagonal_blocks"]
                print(f"\n  Forward pass: OK (block output)")
                print(stat(diag, "  Diagonal blocks"))
                print(stat(off, "  Off-diagonal blocks"))
                h_std = diag.std().item()
                h_max = max(diag.abs().max().item(), off.abs().max().item())
                has_nan = diag.isnan().any().item() or off.isnan().any().item()
                return n_params, h_std, h_max, has_nan
            except Exception as e2:
                print(f"\n  Forward FAILED: {e2}")
                return n_params, None, None, True


def main():
    configs = [
        (64, 32, 5, "small"),
        (128, 32, 5, "middle"),
        (256, 64, 7, "large"),
    ]

    results = []
    for hs, hbs, gnn, label in configs:
        result = test_qhnet_forward(hs, hbs, gnn, label)
        results.append((label, *result))

    print(f"\n{'=' * 70}")
    print("Summary: QHNet initial prediction scale vs model size")
    print(f"{'=' * 70}")
    print(f"{'label':<10} {'params':>10} {'H_std':>10} {'H_absmax':>10} {'NaN':>5}")
    print("-" * 50)
    for label, n_params, h_std, h_max, has_nan in results:
        if h_std is not None:
            print(f"{label:<10} {n_params/1e6:>9.1f}M {h_std:>10.4f} {h_max:>10.2f} {str(has_nan):>5}")
        else:
            print(f"{label:<10} {n_params/1e6:>9.1f}M {'FAILED':>10}")

    print(f"""
Interpretation:
- If H_std grows with model size → width-dependent instability
- If NaN appears only for large → initialization problem
- Compare with QHFlow2 pretrained results:
    small  (12.1M):  H MAE = 0.44 meV
    middle (43.3M):  H MAE = 0.24 meV
    large  (183.7M): H MAE = 0.16 meV
    xl     (909.9M): H MAE = 0.14 meV
  QHFlow2 scales smoothly; if QHNet diverges at large size, that's the story.
""")


if __name__ == "__main__":
    main()
