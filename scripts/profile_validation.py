#!/usr/bin/env python3
"""Profile QHFlow2 validation step to identify bottlenecks.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/profile_validation.py \
        --ckpt src/outputs/QH9Stable-random/QHFlow_so2_v5_1-baseline-QH9Stable-random/checkpoints/last.ckpt \
        --n-batches 5
"""

import argparse
import time
import sys
import os

import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def timer(name, times_dict):
    """Context manager for timing code blocks."""
    class Timer:
        def __enter__(self):
            torch.cuda.synchronize()
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - self.start) * 1000
            times_dict.setdefault(name, []).append(elapsed)
    return Timer()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--n-batches", type=int, default=5, help="Number of batches to profile")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model from checkpoint ──
    print("Loading checkpoint...")
    from qhflow2.pl_module.flow_module import LitModel_flow
    model = LitModel_flow.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval()
    model.to(device)
    print(f"Model loaded: {type(model).__name__}")

    # ── Load dataset ──
    print("Loading dataset...")
    from qhflow2.dataset_module.qh9_datasets_split import QH9Stable
    dataset = QH9Stable(root="dataset", split="random")

    from torch_geometric.loader import DataLoader
    val_indices = dataset.val_mask.tolist()
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Validation set: {len(val_indices)} samples, {len(val_loader)} batches")
    print(f"Profiling {args.n_batches} batches...")
    print()

    # ── Profile each component ──
    times = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= args.n_batches:
                break

            # 1. Data transfer + post_processing
            with timer("1_post_processing", times):
                batch = batch.to(device)
                batch = model.post_processing(batch, model.default_type)
                batch_one = batch.clone()

            # 2. Corrupt (noise injection for flow matching)
            with timer("2_corrupt", times):
                batch = model.corrupt(batch, mul=model.batch_mul)

            # 3. Forward pass (regular model)
            with timer("3_forward", times):
                outputs = model(batch, batch.init_ham_t)

            # 4. Loss computation
            with timer("4_criterion", times):
                errors = model.criterion(
                    outputs, batch,
                    loss_weights=model.loss_weights,
                    loss_weights_detail=model.loss_weights_detail,
                    use_t_scale=model.use_t_scale,
                    use_mse_and_mae=model.use_mse_and_mae,
                )
                loss = errors["loss"]

            # 5. ODE sampling (the expensive part)
            with timer("5_sample_ode_3steps", times):
                sample, traj, pred = model.sample(batch_one, num_timesteps=3)

            # 7. ODE sampling with 1 step (before metric — metric may move data to CPU)
            with timer("7_sample_ode_1step", times):
                sample_1, _, _ = model.sample(batch_one, num_timesteps=1)

            # 6. Metric computation (matrix reconstruction + eigensolve)
            # Clone batch_one so metric's CPU transfer doesn't affect other steps
            with timer("6_metric_total", times):
                metrics = model.metric(sample, batch_one.clone(), metric_weights=model.metric_weights)

            # 8. CPU transfer overhead (measure explicit transfer cost)
            with timer("8_cpu_transfer", times):
                for k, v in sample.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        _ = v.cpu()

            print(f"  Batch {batch_idx}: done")

    # ── Report ──
    print()
    print("=" * 75)
    print(f"QHFlow2 Validation Profiling (bs={args.batch_size}, {args.n_batches} batches)")
    print("=" * 75)
    print()
    print(f"{'Component':<35} {'Mean (ms)':>10} {'Std':>8} {'% Total':>8}")
    print("-" * 65)

    # Compute total validation step time
    total_val = 0
    for key in ["1_post_processing", "2_corrupt", "3_forward", "4_criterion",
                 "5_sample_ode_3steps", "6_metric_total", "7_sample_ode_1step"]:
        if key in times:
            total_val += np.mean(times[key])

    for key in sorted(times.keys()):
        vals = times[key]
        mean = np.mean(vals)
        std = np.std(vals)
        pct = mean / total_val * 100 if total_val > 0 else 0
        print(f"  {key:<33} {mean:>10.1f} {std:>8.1f} {pct:>7.1f}%")

    print(f"  {'─' * 33} {'─' * 10} {'─' * 8} {'─' * 7}")
    print(f"  {'TOTAL validation step':<33} {total_val:>10.1f}")

    # Training step estimate (forward + loss only)
    train_est = np.mean(times.get("3_forward", [0])) + np.mean(times.get("4_criterion", [0]))
    print()
    print(f"  Training step (fwd+loss only):     {train_est:.1f} ms")
    print(f"  Validation / Training ratio:       {total_val / train_est:.1f}x")

    # Breakdown of sampling overhead
    print()
    print("── Sampling overhead breakdown ──")
    ode3 = np.mean(times.get("5_sample_ode_3steps", [0]))
    ode1 = np.mean(times.get("7_sample_ode_1step", [0]))
    metric = np.mean(times.get("6_metric_total", [0]))
    fwd = np.mean(times.get("3_forward", [0]))
    print(f"  ODE 3-step sampling:               {ode3:.1f} ms  ({ode3/fwd:.1f}x forward)")
    print(f"  ODE 1-step sampling:               {ode1:.1f} ms  ({ode1/fwd:.1f}x forward)")
    print(f"  Metric (rebuild + eigensolve):      {metric:.1f} ms")
    print(f"  CPU transfer overhead:             {np.mean(times.get('8_cpu_transfer', [0])):.1f} ms")

    # What-if analysis
    print()
    print("── What-if: validation time with optimizations ──")
    base = total_val
    no_ode1 = base - ode1
    no_sampling = np.mean(times.get("1_post_processing", [0])) + np.mean(times.get("2_corrupt", [0])) + \
                  np.mean(times.get("3_forward", [0])) + np.mean(times.get("4_criterion", [0]))
    no_metric = base - metric
    print(f"  Current:                           {base:.0f} ms (1.00x)")
    print(f"  Remove 1-step ODE:                 {no_ode1:.0f} ms ({base/no_ode1:.2f}x)")
    print(f"  Remove all sampling+metric:        {no_sampling:.0f} ms ({base/no_sampling:.2f}x)")
    print(f"  Remove metric only:                {no_metric:.0f} ms ({base/no_metric:.2f}x)")


if __name__ == "__main__":
    main()
