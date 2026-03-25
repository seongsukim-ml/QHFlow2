"""
QHNet NaN 진단: 학습 중 어느 layer에서 activation이 발산하는지 추적.
forward를 수동 분해하여 각 stage 후 statistics를 기록.
"""

import torch
import torch.nn as nn
import time
from torch_geometric.data import Data, Batch
from torch_cluster import radius_graph
from e3nn import o3

import sys
sys.path.insert(0, "src")
from qhflow2.models.Real_QHNet_qh9 import QHNet


def stat_str(t, name=""):
    """Tensor statistics as string."""
    has_nan = t.isnan().any().item()
    has_inf = t.isinf().any().item()
    finite = t[t.isfinite()]
    if len(finite) == 0:
        return f"{name:<40} ALL NaN/Inf"
    s = f"{name:<40} std={finite.std().item():>12.4f}  absmax={finite.abs().max().item():>12.4f}"
    if has_nan:
        nan_pct = t.isnan().float().mean().item() * 100
        s += f"  NaN={nan_pct:.1f}%"
    if has_inf:
        s += "  Inf!"
    return s


def instrumented_forward(model, data):
    """QHNet forward를 단계별로 실행하면서 activation을 기록."""
    stats = {}

    # Stage 1: Build graph + node embedding
    node_attr, edge_index, rbf_new, edge_sh, _ = model.build_graph(data, model.max_radius)
    node_attr = model.node_embedding(node_attr)
    data.node_attr = node_attr
    data.edge_index = edge_index
    data.edge_attr = rbf_new
    data.edge_sh = edge_sh
    stats["node_embedding"] = node_attr.clone()

    _, full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index = (
        model.build_graph(data, max_radius=10000)
    )
    data.full_edge_index = full_edge_index
    data.full_edge_attr = full_edge_attr
    data.full_edge_sh = full_edge_sh

    # Stage 2: GNN layers
    fii = None
    fij = None
    for layer_idx, layer in enumerate(model.e3_gnn_layer):
        node_attr = layer(data, node_attr)
        stats[f"gnn_layer_{layer_idx}"] = node_attr.clone()

        if layer_idx > model.start_layer:
            ham_idx = layer_idx - model.start_layer - 1
            fii = model.e3_gnn_node_layer[ham_idx](data, node_attr, fii)
            fij = model.e3_gnn_node_pair_layer[ham_idx](data, node_attr, fij)
            stats[f"selfnet_{ham_idx}"] = fii.clone()
            stats[f"pairnet_{ham_idx}"] = fij.clone()

    # Stage 3: Output projection
    fii_proj = model.output_ii(fii)
    fij_proj = model.output_ij(fij)
    stats["output_ii"] = fii_proj.clone()
    stats["output_ij"] = fij_proj.clone()

    # Stage 4: Expansion
    full_dst, full_src = data.full_edge_index
    diag = model.expand_ii["hamiltonian"](
        fii_proj,
        model.fc_ii["hamiltonian"](data.node_attr),
        model.fc_ii_bias["hamiltonian"](data.node_attr),
    )
    stats["expand_ii_out"] = diag.clone()

    node_pair_emb = torch.cat([data.node_attr[full_dst], data.node_attr[full_src]], dim=-1)
    off_diag = model.expand_ij["hamiltonian"](
        fij_proj,
        model.fc_ij["hamiltonian"](node_pair_emb),
        model.fc_ij_bias["hamiltonian"](node_pair_emb),
    )
    stats["expand_ij_out"] = off_diag.clone()

    # Stage 5: fc weights
    stats["fc_ii_weights"] = model.fc_ii["hamiltonian"](data.node_attr).clone()
    stats["fc_ij_weights"] = model.fc_ij["hamiltonian"](node_pair_emb).clone()

    return stats


def make_batch(n_mol=4):
    """Fake QH9-like batch."""
    molecules = []
    for _ in range(n_mol):
        n = torch.randint(3, 9, (1,)).item()
        atoms = torch.randint(1, 9, (n,))
        pos = torch.randn(n, 3) * 2.0
        molecules.append(Data(atoms=atoms, pos=pos, num_nodes=n))
    return Batch.from_data_list(molecules)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for hs, hbs, gnn, label in [(128, 32, 5, "middle"), (256, 64, 7, "large")]:
        print(f"\n{'='*70}")
        print(f"QHNet {label}: hidden={hs}, bottle={hbs}, gnn={gnn}")
        print(f"{'='*70}")

        model = QHNet(hidden_size=hs, bottle_hidden_size=hbs, num_gnn_layers=gnn)
        model = model.to(device)
        model.set(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        data = make_batch(4).to(device)

        # Step 0: initial forward
        print(f"\n--- Step 0 (initial, no grad update) ---")
        model.eval()
        with torch.no_grad():
            stats = instrumented_forward(model, data.clone())
        for name, t in stats.items():
            print(f"  {stat_str(t, name)}")

        # Training steps
        model.train()
        for step in range(1, 11):
            data_step = make_batch(4).to(device)

            optimizer.zero_grad()
            try:
                output = model(data_step, keep_blocks=True)
                diag = output["hamiltonian_diagonal_blocks"]
                off = output["hamiltonian_non_diagonal_blocks"]

                # Fake target (zero matrix)
                loss = diag.abs().mean() + off.abs().mean()

                if loss.isnan():
                    print(f"\n--- Step {step}: loss is NaN ---")
                    # Diagnose
                    model.eval()
                    with torch.no_grad():
                        stats = instrumented_forward(model, data_step.clone())
                    for name, t in stats.items():
                        if t.isnan().any() or t.abs().max() > 1e6:
                            print(f"  {stat_str(t, name)}")
                    model.train()
                    # Skip this step
                    continue

                loss.backward()

                # Check gradient norms
                grad_norms = {}
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        gn = p.grad.norm().item()
                        if gn > 1e4 or torch.isnan(p.grad).any():
                            grad_norms[name] = gn

                if grad_norms:
                    print(f"\n--- Step {step}: large/NaN gradients ---")
                    for name, gn in sorted(grad_norms.items(), key=lambda x: -x[1])[:10]:
                        has_nan = torch.isnan(dict(model.named_parameters())[name].grad).any()
                        print(f"  {name:<50} grad_norm={gn:.2e} {'NaN!' if has_nan else ''}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                if step <= 3 or step % 5 == 0:
                    print(f"  Step {step}: loss={loss.item():.6f}")

            except Exception as e:
                print(f"\n--- Step {step}: EXCEPTION: {e} ---")
                # Diagnose after exception
                model.eval()
                with torch.no_grad():
                    try:
                        stats = instrumented_forward(model, data_step.clone())
                        for name, t in stats.items():
                            if t.isnan().any() or t.abs().max() > 1e6:
                                print(f"  {stat_str(t, name)}")
                    except Exception as e2:
                        print(f"  Diagnosis also failed: {e2}")
                break

        del model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
