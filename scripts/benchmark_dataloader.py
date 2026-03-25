#!/usr/bin/env python3
"""Benchmark dataloader: original vs preprocessed, varying num_workers.

Usage:
    cd /home1/irteam/data-vol1/projects/QHFlow2
    CUDA_VISIBLE_DEVICES=7 python scripts/benchmark_dataloader.py
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch_geometric.loader import DataLoader


def benchmark_loader(dataset, batch_size, num_workers, n_batches=30, warmup=5):
    """Time batch loading + GPU transfer."""
    train_mask = dataset.train_mask if hasattr(dataset, 'train_mask') else None
    n_need = batch_size * (n_batches + warmup)

    if train_mask is not None:
        indices = train_mask.tolist()[:n_need]
    else:
        indices = list(range(min(len(dataset), n_need)))

    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    times = []
    for i, batch in enumerate(loader):
        if i >= n_batches + warmup:
            break
        t0 = time.perf_counter()
        batch = batch.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000)

    del loader
    if not times:
        return {"mean": float('inf'), "std": 0, "samp_s": 0}
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "samp_s": batch_size / (np.mean(times) / 1000),
    }


def benchmark_getitem(dataset, n_samples=20):
    """Benchmark raw __getitem__."""
    n = min(n_samples, len(dataset))
    # warmup
    for i in range(min(3, n)):
        _ = dataset[i]

    times = []
    for i in range(n):
        t0 = time.perf_counter()
        _ = dataset[i]
        times.append((time.perf_counter() - t0) * 1000)

    return {"mean": np.mean(times), "median": np.median(times),
            "p95": np.percentile(times, 95), "std": np.std(times)}


def main():
    BS = 32
    NW_LIST = [0, 4, 8]

    print("=" * 70)
    print("QHFlow2 DataLoader Benchmark: Original vs Preprocessed")
    print("=" * 70)

    # ── Load datasets ──
    print("\nLoading preprocessed dataset...")
    from qhflow2.dataset_module.qh9_preprocessed import QH9PreprocessedDataset
    ds_pre = QH9PreprocessedDataset(
        "dataset/QH9Stable_preprocessed.lmdb", split="random", mode="train",
    )
    print(f"  Preprocessed: {ds_pre.len()} samples")

    print("Loading original dataset...")
    from qhflow2.dataset_module.qh9_datasets_split import QH9Stable
    ds_orig = QH9Stable(root="dataset", split="random")
    print(f"  Original: {len(ds_orig)} samples")

    # ── __getitem__ benchmark ──
    print("\n── __getitem__ (20 samples) ──")
    print(f"{'Dataset':<20} {'Mean (ms)':>10} {'Median':>10} {'P95':>10}")
    print("-" * 55)

    gi_pre = benchmark_getitem(ds_pre, 20)
    print(f"{'preprocessed':<20} {gi_pre['mean']:>10.1f} {gi_pre['median']:>10.1f} {gi_pre['p95']:>10.1f}")

    # Original __getitem__ is very slow due to Q tensor (~16s/sample)
    # Only benchmark 5 samples
    gi_orig = benchmark_getitem(ds_orig, 5)
    print(f"{'original':<20} {gi_orig['mean']:>10.1f} {gi_orig['median']:>10.1f} {gi_orig['p95']:>10.1f}")

    print(f"\n  __getitem__ speedup: {gi_orig['mean'] / gi_pre['mean']:.0f}x")

    # ── DataLoader throughput ──
    print(f"\n── DataLoader throughput (bs={BS}, 30 batches) ──")
    print(f"{'Dataset':<20} {'nw':>4} {'Batch (ms)':>12} {'Std':>8} {'Samp/s':>10}")
    print("-" * 58)

    # Preprocessed: all num_workers
    for nw in NW_LIST:
        r = benchmark_loader(ds_pre, BS, nw)
        print(f"{'preprocessed':<20} {nw:>4} {r['mean']:>12.1f} {r['std']:>8.1f} {r['samp_s']:>10.1f}")

    print()

    # Original: only nw=4 (default) — nw=0 would be too slow
    r_orig_4 = benchmark_loader(ds_orig, BS, 4, n_batches=10, warmup=2)
    print(f"{'original':<20} {4:>4} {r_orig_4['mean']:>12.1f} {r_orig_4['std']:>8.1f} {r_orig_4['samp_s']:>10.1f}")

    r_orig_8 = benchmark_loader(ds_orig, BS, 8, n_batches=10, warmup=2)
    print(f"{'original':<20} {8:>4} {r_orig_8['mean']:>12.1f} {r_orig_8['std']:>8.1f} {r_orig_8['samp_s']:>10.1f}")

    # ── Summary ──
    print("\n── Summary ──")
    r_pre_best = benchmark_loader(ds_pre, BS, 8)
    print(f"  Original (nw=8):       {r_orig_8['samp_s']:>8.0f} samp/s  ({r_orig_8['mean']:.0f} ms/batch)")
    print(f"  Preprocessed (nw=8):   {r_pre_best['samp_s']:>8.0f} samp/s  ({r_pre_best['mean']:.0f} ms/batch)")
    print(f"  Speedup:               {r_pre_best['samp_s'] / r_orig_8['samp_s']:.1f}x")


if __name__ == "__main__":
    main()
