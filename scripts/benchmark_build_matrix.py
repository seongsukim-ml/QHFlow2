#!/usr/bin/env python3
"""Benchmark build_final_matrix: current vs vectorized implementations.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/benchmark_build_matrix.py
"""

import time
import sys
import os

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Vectorized implementation ──────────────────────────────────────

def build_final_matrix_vectorized(
    data,
    diagonal_matrix,
    non_diagonal_matrix,
    orbital_mask,
):
    """Vectorized: pre-compute gather indices per molecule, single scatter."""
    if hasattr(data, "full_edge_index"):
        dst, src = data.full_edge_index
    else:
        dst, src = data.edge_index_full

    atoms = data.atoms.squeeze().detach().cpu()
    ptr_list = data.ptr.detach().cpu().tolist()

    # Pre-compute active orbital counts per atom type
    mask_sizes = {z: len(m) for z, m in orbital_mask.items()}

    # Determine device
    matrix_device = diagonal_matrix.device if isinstance(diagonal_matrix, torch.Tensor) else diagonal_matrix[0].device

    # Pre-compute masks on device
    masks_on_device = {}
    for z, m in orbital_mask.items():
        masks_on_device[z] = m.to(matrix_device) if m.device != matrix_device else m

    # Build edge lookup: (src, dst) → edge_index (one-time cost)
    src_cpu = src.detach().cpu()
    dst_cpu = dst.detach().cpu()
    # For dense graphs, build a 2D index tensor instead of dict
    n_nodes = len(atoms)
    edge_idx_matrix = torch.full((n_nodes, n_nodes), -1, dtype=torch.long)
    for i in range(len(src_cpu)):
        edge_idx_matrix[src_cpu[i], dst_cpu[i]] = i

    final_matrices = []

    for graph_idx in range(len(ptr_list) - 1):
        g_start = ptr_list[graph_idx]
        g_end = ptr_list[graph_idx + 1]
        n_atoms_g = g_end - g_start
        graph_atoms = atoms[g_start:g_end].tolist()

        # Compute total orbital dimension for this molecule
        active_sizes = [mask_sizes[z] for z in graph_atoms]
        h_dim = sum(active_sizes)

        # Pre-compute cumulative offsets for row/col placement
        offsets = [0]
        for s in active_sizes:
            offsets.append(offsets[-1] + s)

        # Allocate full matrix
        # Handle 2D (H, S) and 3D (Q tensor) cases
        sample_block = diagonal_matrix[g_start]
        if sample_block.dim() == 2:
            mat = torch.zeros(h_dim, h_dim, dtype=sample_block.dtype, device=matrix_device)
        else:
            extra_dim = sample_block.shape[-1]
            mat = torch.zeros(h_dim, h_dim, extra_dim, dtype=sample_block.dtype, device=matrix_device)

        # Fill blocks using pre-computed indices
        for i in range(n_atoms_g):
            src_idx = g_start + i
            src_mask = masks_on_device[graph_atoms[i]]
            r_start, r_end = offsets[i], offsets[i + 1]

            for j in range(n_atoms_g):
                dst_idx = g_start + j
                dst_mask = masks_on_device[graph_atoms[j]]
                c_start, c_end = offsets[j], offsets[j + 1]

                if i == j:
                    block = diagonal_matrix[src_idx]
                else:
                    eidx = edge_idx_matrix[src_idx, dst_idx].item()
                    block = non_diagonal_matrix[eidx]

                # Extract active orbitals and place directly
                if block.dim() == 2:
                    mat[c_start:c_end, r_start:r_end] = block[dst_mask][:, src_mask]
                else:
                    mat[c_start:c_end, r_start:r_end, :] = block[dst_mask][:, src_mask, :]

        final_matrices.append(mat)

    return final_matrices


def build_final_matrix_scatter(
    data,
    diagonal_matrix,
    non_diagonal_matrix,
    orbital_mask,
):
    """Scatter-based: build row/col index tensors, use advanced indexing for all blocks at once."""
    if hasattr(data, "full_edge_index"):
        dst, src = data.full_edge_index
    else:
        dst, src = data.edge_index_full

    atoms = data.atoms.squeeze().detach().cpu()
    ptr_list = data.ptr.detach().cpu().tolist()
    mask_sizes = {z: len(m) for z, m in orbital_mask.items()}

    matrix_device = diagonal_matrix.device if isinstance(diagonal_matrix, torch.Tensor) else diagonal_matrix[0].device

    masks_on_device = {}
    for z, m in orbital_mask.items():
        masks_on_device[z] = m.to(matrix_device) if m.device != matrix_device else m

    # Build edge lookup
    src_cpu = src.detach().cpu()
    dst_cpu = dst.detach().cpu()
    n_nodes = len(atoms)
    edge_idx_matrix = torch.full((n_nodes, n_nodes), -1, dtype=torch.long)
    for i in range(len(src_cpu)):
        edge_idx_matrix[src_cpu[i], dst_cpu[i]] = i

    final_matrices = []

    for graph_idx in range(len(ptr_list) - 1):
        g_start = ptr_list[graph_idx]
        g_end = ptr_list[graph_idx + 1]
        n_atoms_g = g_end - g_start
        graph_atoms = atoms[g_start:g_end].tolist()

        active_sizes = [mask_sizes[z] for z in graph_atoms]
        h_dim = sum(active_sizes)
        offsets = [0]
        for s in active_sizes:
            offsets.append(offsets[-1] + s)

        # Pre-build all row and col indices for the gather
        all_block_indices = []  # (block_tensor_idx, is_diagonal)
        all_row_indices = []
        all_col_indices = []
        all_src_masks = []
        all_dst_masks = []

        for i in range(n_atoms_g):
            src_idx = g_start + i
            src_mask = masks_on_device[graph_atoms[i]]
            for j in range(n_atoms_g):
                dst_idx = g_start + j
                dst_mask = masks_on_device[graph_atoms[j]]

                n_dst = len(dst_mask)
                n_src = len(src_mask)

                # Row indices in final matrix
                rows = torch.arange(offsets[j], offsets[j] + n_dst, device=matrix_device)
                cols = torch.arange(offsets[i], offsets[i] + n_src, device=matrix_device)
                row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')

                all_row_indices.append(row_grid.reshape(-1))
                all_col_indices.append(col_grid.reshape(-1))

                if i == j:
                    block = diagonal_matrix[src_idx]
                else:
                    eidx = edge_idx_matrix[src_idx, dst_idx].item()
                    block = non_diagonal_matrix[eidx]

                vals = block[dst_mask][:, src_mask].reshape(-1)
                all_block_indices.append(vals)

        # Single scatter
        row_idx = torch.cat(all_row_indices)
        col_idx = torch.cat(all_col_indices)
        values = torch.cat(all_block_indices)

        mat = torch.zeros(h_dim, h_dim, dtype=values.dtype, device=matrix_device)
        mat[row_idx, col_idx] = values

        final_matrices.append(mat)

    return final_matrices


def build_final_matrix_cpu(
    data,
    diagonal_matrix,
    non_diagonal_matrix,
    orbital_mask,
):
    """CPU-optimized: move blocks to CPU first, assemble there."""
    if hasattr(data, "full_edge_index"):
        dst, src = data.full_edge_index
    else:
        dst, src = data.edge_index_full

    atoms = data.atoms.squeeze().detach().cpu()
    ptr_list = data.ptr.detach().cpu().tolist()
    mask_sizes = {z: len(m) for z, m in orbital_mask.items()}

    # Move all blocks to CPU once
    if isinstance(diagonal_matrix, torch.Tensor):
        diag_cpu = diagonal_matrix.detach().cpu()
        nondiag_cpu = non_diagonal_matrix.detach().cpu()
    else:
        diag_cpu = torch.stack([d.detach().cpu() for d in diagonal_matrix])
        nondiag_cpu = torch.stack([d.detach().cpu() for d in non_diagonal_matrix])

    masks_cpu = {z: m.cpu() for z, m in orbital_mask.items()}

    # Build edge lookup on CPU
    src_cpu = src.detach().cpu()
    dst_cpu = dst.detach().cpu()
    n_nodes = len(atoms)
    edge_idx_matrix = torch.full((n_nodes, n_nodes), -1, dtype=torch.long)
    for i in range(len(src_cpu)):
        edge_idx_matrix[src_cpu[i], dst_cpu[i]] = i

    final_matrices = []

    for graph_idx in range(len(ptr_list) - 1):
        g_start = ptr_list[graph_idx]
        g_end = ptr_list[graph_idx + 1]
        n_atoms_g = g_end - g_start
        graph_atoms = atoms[g_start:g_end].tolist()

        active_sizes = [mask_sizes[z] for z in graph_atoms]
        h_dim = sum(active_sizes)
        offsets = [0]
        for s in active_sizes:
            offsets.append(offsets[-1] + s)

        mat = torch.zeros(h_dim, h_dim, dtype=diag_cpu.dtype)

        for i in range(n_atoms_g):
            src_idx = g_start + i
            src_mask = masks_cpu[graph_atoms[i]]
            r_start, r_end = offsets[i], offsets[i + 1]
            for j in range(n_atoms_g):
                dst_idx = g_start + j
                dst_mask = masks_cpu[graph_atoms[j]]
                c_start, c_end = offsets[j], offsets[j + 1]

                if i == j:
                    block = diag_cpu[src_idx]
                else:
                    eidx = edge_idx_matrix[src_idx, dst_idx].item()
                    block = nondiag_cpu[eidx]

                mat[c_start:c_end, r_start:r_end] = block[dst_mask][:, src_mask]

        final_matrices.append(mat)

    return final_matrices


def build_final_matrix_numpy(
    data,
    diagonal_matrix,
    non_diagonal_matrix,
    orbital_mask,
):
    """Numpy-based: move everything to CPU numpy, avoid torch overhead for small ops."""
    if hasattr(data, "full_edge_index"):
        dst, src = data.full_edge_index
    else:
        dst, src = data.edge_index_full

    atoms = data.atoms.squeeze().detach().cpu().numpy()
    ptr_list = data.ptr.detach().cpu().numpy()

    # Move blocks to numpy once
    if isinstance(diagonal_matrix, torch.Tensor):
        diag_np = diagonal_matrix.detach().cpu().numpy()
        nondiag_np = non_diagonal_matrix.detach().cpu().numpy()
    else:
        diag_np = torch.stack(list(diagonal_matrix)).detach().cpu().numpy()
        nondiag_np = torch.stack(list(non_diagonal_matrix)).detach().cpu().numpy()

    masks_np = {z: m.detach().cpu().numpy() for z, m in orbital_mask.items()}
    mask_sizes = {z: len(m) for z, m in masks_np.items()}

    # Edge lookup
    src_np = src.detach().cpu().numpy()
    dst_np = dst.detach().cpu().numpy()
    n_nodes = len(atoms)
    edge_idx_matrix = np.full((n_nodes, n_nodes), -1, dtype=np.int64)
    edge_idx_matrix[src_np, dst_np] = np.arange(len(src_np))

    final_matrices = []

    for graph_idx in range(len(ptr_list) - 1):
        g_start = ptr_list[graph_idx]
        g_end = ptr_list[graph_idx + 1]
        n_atoms_g = g_end - g_start
        graph_atoms = atoms[g_start:g_end]

        active_sizes = [mask_sizes[z] for z in graph_atoms]
        h_dim = sum(active_sizes)
        offsets = np.zeros(n_atoms_g + 1, dtype=np.int64)
        np.cumsum(active_sizes, out=offsets[1:])

        mat = np.zeros((h_dim, h_dim), dtype=diag_np.dtype)

        for i in range(n_atoms_g):
            src_idx = g_start + i
            src_mask = masks_np[graph_atoms[i]]
            rs, re = offsets[i], offsets[i + 1]
            for j in range(n_atoms_g):
                dst_idx = g_start + j
                dst_mask = masks_np[graph_atoms[j]]
                cs, ce = offsets[j], offsets[j + 1]

                if i == j:
                    block = diag_np[src_idx]
                else:
                    block = nondiag_np[edge_idx_matrix[src_idx, dst_idx]]

                mat[cs:ce, rs:re] = block[np.ix_(dst_mask, src_mask)]

        final_matrices.append(torch.from_numpy(mat))

    return final_matrices


def build_final_matrix_prealloc_gpu(
    data,
    diagonal_matrix,
    non_diagonal_matrix,
    orbital_mask,
):
    """Pre-allocate on GPU, fill with index_select (same as original but no torch.cat)."""
    if hasattr(data, "full_edge_index"):
        dst, src = data.full_edge_index
    else:
        dst, src = data.edge_index_full

    atoms = data.atoms.squeeze().detach().cpu()
    ptr_list = data.ptr.detach().cpu().tolist()
    mask_sizes = {z: len(m) for z, m in orbital_mask.items()}

    matrix_device = diagonal_matrix.device if isinstance(diagonal_matrix, torch.Tensor) else diagonal_matrix[0].device

    masks_on_device = {z: m.to(matrix_device) for z, m in orbital_mask.items()}

    src_cpu = src.detach().cpu()
    dst_cpu = dst.detach().cpu()
    n_nodes = len(atoms)
    edge_idx_matrix = torch.full((n_nodes, n_nodes), -1, dtype=torch.long)
    for i in range(len(src_cpu)):
        edge_idx_matrix[src_cpu[i], dst_cpu[i]] = i

    final_matrices = []

    for graph_idx in range(len(ptr_list) - 1):
        g_start = ptr_list[graph_idx]
        g_end = ptr_list[graph_idx + 1]
        n_atoms_g = g_end - g_start
        graph_atoms = atoms[g_start:g_end].tolist()

        active_sizes = [mask_sizes[z] for z in graph_atoms]
        h_dim = sum(active_sizes)
        offsets = [0]
        for s in active_sizes:
            offsets.append(offsets[-1] + s)

        mat = torch.zeros(h_dim, h_dim, dtype=diagonal_matrix.dtype, device=matrix_device)

        for i in range(n_atoms_g):
            src_idx = g_start + i
            src_mask = masks_on_device[graph_atoms[i]]
            rs, re = offsets[i], offsets[i + 1]
            for j in range(n_atoms_g):
                dst_idx = g_start + j
                dst_mask = masks_on_device[graph_atoms[j]]
                cs, ce = offsets[j], offsets[j + 1]

                if i == j:
                    block = diagonal_matrix[src_idx].index_select(-2, dst_mask).index_select(-1, src_mask)
                else:
                    eidx = edge_idx_matrix[src_idx, dst_idx].item()
                    block = non_diagonal_matrix[eidx].index_select(-2, dst_mask).index_select(-1, src_mask)

                mat[cs:ce, rs:re] = block

        final_matrices.append(mat)

    return final_matrices


# ── Benchmark ──────────────────────────────────────────────────────

def timer_sync(device):
    class Timer:
        def __enter__(self):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            self.start = time.perf_counter()
            return self
        def __exit__(self, *a):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            self.elapsed_ms = (time.perf_counter() - self.start) * 1000
    return Timer()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    from qhflow2.pl_module.flow_module import LitModel_flow
    ckpt = "src/outputs/QH9Stable-random/QHFlow_so2_v5_1-baseline-QH9Stable-random/checkpoints/last.ckpt"
    model = LitModel_flow.load_from_checkpoint(ckpt, map_location=device)
    model.eval()
    model.to(device)

    # Load dataset
    print("Loading dataset...")
    from qhflow2.dataset_module.qh9_datasets_split import QH9Stable
    from torch_geometric.loader import DataLoader
    dataset = QH9Stable(root="dataset", split="random")
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, dataset.val_mask.tolist()),
        batch_size=32, shuffle=False, num_workers=4,
    )

    # Get orbital mask from model
    orbital_mask = model.orbital_mask

    # Import original
    from qhflow2.common.matrix_transforms import _build_final_matrix

    n_batches = 5
    n_warmup = 1
    times = {
        "original_gpu": [],
        "prealloc_gpu": [],
        "numpy_cpu": [],
        "torch_cpu": [],
    }

    print(f"\nBenchmarking {n_batches} batches (bs=32), {n_warmup} warmup...")
    print()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= n_batches + n_warmup:
                break

            batch = batch.to(device)
            batch = model.post_processing(batch, model.default_type)

            diag = batch.diagonal_hamiltonian
            nondiag = batch.non_diagonal_hamiltonian

            # 1. Original (GPU) — current implementation
            with timer_sync(device) as t:
                result_orig = _build_final_matrix(batch, diag, nondiag, orbital_mask)
            if batch_idx >= n_warmup:
                times["original_gpu"].append(t.elapsed_ms)

            # 2. Pre-alloc GPU — same loop but pre-allocate output, no torch.cat
            with timer_sync(device) as t:
                result_prealloc = build_final_matrix_prealloc_gpu(batch, diag, nondiag, orbital_mask)
            if batch_idx >= n_warmup:
                times["prealloc_gpu"].append(t.elapsed_ms)

            # 3. Numpy CPU — everything in numpy
            with timer_sync(device) as t:
                result_np = build_final_matrix_numpy(batch, diag, nondiag, orbital_mask)
            if batch_idx >= n_warmup:
                times["numpy_cpu"].append(t.elapsed_ms)

            # 4. Torch CPU — torch ops on CPU
            with timer_sync(device) as t:
                result_tcpu = build_final_matrix_cpu(batch, diag, nondiag, orbital_mask)
            if batch_idx >= n_warmup:
                times["torch_cpu"].append(t.elapsed_ms)

            # Verify correctness (first profiled batch)
            if batch_idx == n_warmup:
                print("Verifying correctness...")
                for i in range(min(3, len(result_orig))):
                    orig = result_orig[i].cpu()
                    for name, res in [("prealloc_gpu", result_prealloc), ("numpy_cpu", result_np), ("torch_cpu", result_tcpu)]:
                        r = res[i].cpu() if isinstance(res[i], torch.Tensor) else torch.from_numpy(res[i])
                        ok = torch.allclose(orig, r, atol=1e-10)
                        diff = torch.abs(orig - r).max()
                        print(f"  mol {i} {name}: {'✓' if ok else 'MISMATCH'} max_diff={diff:.2e}")
                print()

            if batch_idx >= n_warmup:
                print(f"  Batch {batch_idx - n_warmup}: orig={times['original_gpu'][-1]:.1f}ms  prealloc={times['prealloc_gpu'][-1]:.1f}ms  numpy={times['numpy_cpu'][-1]:.1f}ms  torch_cpu={times['torch_cpu'][-1]:.1f}ms")

    # Report
    print()
    print("=" * 70)
    print("build_final_matrix Benchmark (bs=32, 5 batches)")
    print("=" * 70)
    print()
    print(f"{'Method':<25} {'Mean (ms)':>10} {'Std':>8} {'Speedup':>8}")
    print("-" * 55)

    baseline = np.mean(times["original_gpu"])
    for name, vals in times.items():
        mean = np.mean(vals)
        std = np.std(vals)
        speedup = baseline / mean if mean > 0 else 0
        print(f"  {name:<23} {mean:>10.1f} {std:>8.1f} {speedup:>7.2f}x")

    # Per-call cost (metric calls build_final_matrix 3 times)
    print()
    print(f"  In metric context (3 calls):")
    for name, vals in times.items():
        print(f"    {name}: {np.mean(vals) * 3:.0f} ms total")


if __name__ == "__main__":
    main()
