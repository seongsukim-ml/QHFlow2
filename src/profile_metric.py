#!/usr/bin/env python3
"""
Profile the metric computation breakdown for QHFlow2.

Breaks down _metric_qh9() into:
  1. build_final_matrix  (all 3 calls: pred H, target H, overlap S)
  2. cal_orbital_and_energies  (generalized eigenvalue solve, both pred & target)
  3. HOMO/LUMO/GAP extraction + MAE computation (the rest)

Also compares GPU vs CPU metric computation.

Usage:
    CUDA_VISIBLE_DEVICES=7 python profile_metric.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
from collections import defaultdict

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(16)

# Setup paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from qhflow2.common.setup import setup_paths
setup_paths(SRC_DIR)

from qhflow2.pl_module.flow_module import LitModel_flow
from qhflow2.dataset_module.qh9_datasets_split import QH9Stable
from torch_geometric.loader import DataLoader

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
CKPT_PATH = os.path.join(
    SRC_DIR,
    "outputs/QH9Stable-random/QHFlow_so2_v5_1-baseline-QH9Stable-random/checkpoints/last.ckpt",
)
DATASET_ROOT = os.path.join(SRC_DIR, "..", "dataset")
BATCH_SIZE = 32
NUM_PROFILE_BATCHES = 3
NUM_ODE_STEPS = 3   # match val config


def load_model(device):
    """Load model from checkpoint."""
    print(f"[1/4] Loading checkpoint: {os.path.basename(CKPT_PATH)}")
    t0 = time.perf_counter()
    model = LitModel_flow.load_from_checkpoint(CKPT_PATH, map_location=device)
    model.eval()
    model = model.to(device)
    print(f"       Loaded in {time.perf_counter() - t0:.1f}s  (device={device})")
    return model


def load_val_loader():
    """Load the validation split of QH9Stable-random."""
    print(f"[2/4] Loading QH9Stable dataset (split=random) ...")
    t0 = time.perf_counter()
    dataset = QH9Stable(root=DATASET_ROOT, split="random")
    val_dataset = dataset[dataset.val_mask]
    loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"       Loaded {len(val_dataset)} val samples in {time.perf_counter() - t0:.1f}s")
    return loader


# ──────────────────────────────────────────────────────────────
# Simple region timer: each region is independent, no nesting issues
# ──────────────────────────────────────────────────────────────
class RegionTimer:
    """Accumulates timings per named region. Each call is independent."""

    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.wall_totals = defaultdict(float)   # seconds
        self.cuda_totals = defaultdict(float)   # ms
        self.counts = defaultdict(int)

    def time_region(self, name):
        """Context manager that times a named region."""
        return _RegionCtx(self, name)

    def report(self):
        """Return list of (name, wall_ms, cuda_ms, count) sorted by wall descending."""
        rows = []
        for name in self.wall_totals:
            rows.append((
                name,
                self.wall_totals[name] * 1000,
                self.cuda_totals[name],
                self.counts[name],
            ))
        rows.sort(key=lambda r: -r[1])
        return rows


class _RegionCtx:
    __slots__ = ("timer", "name", "t0", "ev_start", "ev_end")

    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        if self.timer.use_cuda:
            torch.cuda.synchronize()
            self.ev_start = torch.cuda.Event(enable_timing=True)
            self.ev_end = torch.cuda.Event(enable_timing=True)
            self.ev_start.record()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.timer.use_cuda:
            self.ev_end.record()
            torch.cuda.synchronize()
            self.timer.cuda_totals[self.name] += self.ev_start.elapsed_time(self.ev_end)
        self.timer.wall_totals[self.name] += time.perf_counter() - self.t0
        self.timer.counts[self.name] += 1
        return False


def _dict_to_device(d, device):
    """Move all tensor values in a dict to device."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ──────────────────────────────────────────────────────────────
# Instrumented metric
# ──────────────────────────────────────────────────────────────
def run_instrumented_metric(model, _outputs, _target, timer):
    """
    Replica of _metric_qh9 with timing instrumentation.
    """
    metric_weights = {
        "hamiltonian": [1.0, "hamiltonian"],
        "diagonal_hamiltonian": [1.0, "diagonal_hamiltonian"],
        "non_diagonal_hamiltonian": [1.0, "non_diagonal_hamiltonian"],
        "orbital_energies": [1.0, "orbital_energies"],
        "orbital_coefficients": [1.0, "orbital_coefficients"],
        "HOMO": [1.0, "HOMO"],
        "LUMO": [1.0, "LUMO"],
        "GAP": [1.0, "GAP"],
    }

    # ── CPU transfer if needed ──
    with timer.time_region("cpu_transfer"):
        if model.metric_cpu:
            outputs = _dict_to_device(_outputs, "cpu")
            batch = _target.to("cpu")
        else:
            outputs = _outputs
            batch = _target

    # ── build_final_matrix (3x: pred H, target H, overlap S) ──
    with timer.time_region("build_final_matrix"):
        outputs["hamiltonian"] = model.build_final_matrix(
            batch,
            outputs["hamiltonian_diagonal_blocks"],
            outputs["hamiltonian_non_diagonal_blocks"],
            transform=True, convention="back2pyscf", dtype=torch.float64,
        )
        batch.hamiltonian = model.build_final_matrix(
            batch,
            batch.diagonal_hamiltonian,
            batch.non_diagonal_hamiltonian,
            transform=True, convention="back2pyscf", dtype=torch.float64,
        )
        overlap = model.build_final_matrix(
            batch,
            batch.diagonal_overlap,
            batch.non_diagonal_overlap,
            transform=True, convention="back2pyscf", dtype=torch.float64,
        )

    # ── cal_orbital_and_energies (2x: pred, target) ──
    with timer.time_region("cal_orbital_and_energies"):
        outputs["orbital_energies"], outputs["orbital_coefficients"] = (
            model.cal_orbital_and_energies(overlap, outputs["hamiltonian"])
        )
        batch.orbital_energies, batch.orbital_coefficients = (
            model.cal_orbital_and_energies(overlap, batch["hamiltonian"])
        )

    # ── HOMO/LUMO/GAP extraction + _metric_calc ──
    with timer.time_region("homo_lumo_mae"):
        num_orb = []
        pred_HOMO, pred_LUMO = [], []
        gt_HOMO, gt_LUMO = [], []
        for i in range(len(batch)):
            num_orb.append(int(batch.atoms[batch.ptr[i]:batch.ptr[i + 1]].sum() / 2))
            pred_HOMO.append(outputs["orbital_energies"][i][num_orb[i] - 1])
            pred_LUMO.append(outputs["orbital_energies"][i][num_orb[i]])
            gt_HOMO.append(batch.orbital_energies[i][num_orb[i] - 1])
            gt_LUMO.append(batch.orbital_energies[i][num_orb[i]])

        pred_HOMO = torch.stack(pred_HOMO)
        pred_LUMO = torch.stack(pred_LUMO)
        gt_HOMO = torch.stack(gt_HOMO)
        gt_LUMO = torch.stack(gt_LUMO)

        outputs["HOMO"] = pred_HOMO
        outputs["LUMO"] = pred_LUMO
        outputs["GAP"] = pred_LUMO - pred_HOMO
        batch["HOMO"] = gt_HOMO
        batch["LUMO"] = gt_LUMO
        batch["GAP"] = gt_LUMO - gt_HOMO

        for i in range(len(batch)):
            outputs["orbital_energies"][i] = outputs["orbital_energies"][i][:num_orb[i]]
            outputs["orbital_coefficients"][i] = outputs["orbital_coefficients"][i][:, :num_orb[i]]
            batch.orbital_energies[i] = batch.orbital_energies[i][:num_orb[i]]
            batch.orbital_coefficients[i] = batch.orbital_coefficients[i][:, :num_orb[i]]

        outputs["diagonal_hamiltonian"] = outputs["hamiltonian_diagonal_blocks"]
        outputs["non_diagonal_hamiltonian"] = outputs["hamiltonian_non_diagonal_blocks"]

        error_dict = model._metric_calc(outputs, batch, metric_weights)

    return error_dict


# ──────────────────────────────────────────────────────────────
# Profile runner
# ──────────────────────────────────────────────────────────────
def run_profile(model, val_loader, device, label, metric_on_cpu):
    """Run profiling over NUM_PROFILE_BATCHES batches."""
    print(f"\n{'='*60}")
    print(f"  Profiling: {label}")
    print(f"  metric_cpu={metric_on_cpu}, device={device}")
    print(f"{'='*60}")

    # Set metric_cpu flag
    orig_metric_cpu = model.metric_cpu
    model.metric_cpu = metric_on_cpu

    use_cuda = (device.type == "cuda")
    # For CPU metric, the computation runs on CPU so CUDA events are not meaningful
    # for the metric sub-components. But sample still runs on GPU.
    timer_sample = RegionTimer(use_cuda=use_cuda)
    timer_metric = RegionTimer(use_cuda=(use_cuda and not metric_on_cpu))

    batch_iter = iter(val_loader)
    for b_idx in range(NUM_PROFILE_BATCHES):
        batch = next(batch_iter)
        batch = batch.to(device)
        batch = model.post_processing(batch, model.default_type)

        # ── Sample (always on GPU) ──
        with timer_sample.time_region("sample"):
            with torch.no_grad():
                sample_out, traj, pred = model.sample(batch, num_timesteps=NUM_ODE_STEPS)

        # ── Metric (instrumented) ──
        with timer_metric.time_region("metric_total"):
            with torch.no_grad():
                _ = run_instrumented_metric(model, sample_out, batch, timer_metric)

        print(f"  batch {b_idx+1}/{NUM_PROFILE_BATCHES} done")

    # Restore
    model.metric_cpu = orig_metric_cpu

    return timer_sample, timer_metric


def print_table(label, timer_sample, timer_metric, n_batches):
    """Pretty-print timing results."""
    print(f"\n{'='*72}")
    print(f"  TIMING RESULTS: {label}  ({n_batches} batches, bs={BATCH_SIZE})")
    print(f"{'='*72}")

    # Sample timing
    s_rows = timer_sample.report()
    s_wall = timer_sample.wall_totals.get("sample", 0) * 1000 / n_batches
    s_cuda = timer_sample.cuda_totals.get("sample", 0) / n_batches
    print(f"\n  sample() per batch:  wall={s_wall:.1f} ms   cuda={s_cuda:.1f} ms")

    # Metric total
    m_wall = timer_metric.wall_totals.get("metric_total", 0) * 1000 / n_batches
    m_cuda = timer_metric.cuda_totals.get("metric_total", 0) / n_batches
    print(f"  metric() per batch:  wall={m_wall:.1f} ms   cuda={m_cuda:.1f} ms")

    # Component breakdown
    has_cuda = any(timer_metric.cuda_totals[n] > 0 for n in timer_metric.cuda_totals)
    if has_cuda:
        print(f"\n  {'Component':<30s} {'Wall (ms)':>10s} {'CUDA (ms)':>10s} {'Calls':>6s} {'% of total':>11s}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*6} {'-'*11}")
    else:
        print(f"\n  {'Component':<30s} {'Wall (ms)':>10s} {'Calls':>6s} {'% of total':>11s}")
        print(f"  {'-'*30} {'-'*10} {'-'*6} {'-'*11}")

    rows = timer_metric.report()
    total_wall = m_wall
    sub_wall = 0.0
    sub_cuda = 0.0
    for name, wall_ms, cuda_ms, count in rows:
        if name == "metric_total":
            continue
        per_batch_wall = wall_ms / n_batches
        per_batch_cuda = cuda_ms / n_batches
        pct = (per_batch_wall / total_wall * 100) if total_wall > 0 else 0
        calls_per_batch = count // n_batches
        sub_wall += per_batch_wall
        sub_cuda += per_batch_cuda
        if has_cuda:
            print(f"  {name:<30s} {per_batch_wall:>10.1f} {per_batch_cuda:>10.1f} {calls_per_batch:>6d} {pct:>10.1f}%")
        else:
            print(f"  {name:<30s} {per_batch_wall:>10.1f} {calls_per_batch:>6d} {pct:>10.1f}%")

    if has_cuda:
        print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*6} {'─'*11}")
        print(f"  {'SUM (sub-components)':<30s} {sub_wall:>10.1f} {sub_cuda:>10.1f}")
        print(f"  {'metric_total (measured)':<30s} {total_wall:>10.1f} {m_cuda:>10.1f}")
    else:
        print(f"  {'─'*30} {'─'*10} {'─'*6} {'─'*11}")
        print(f"  {'SUM (sub-components)':<30s} {sub_wall:>10.1f}")
        print(f"  {'metric_total (measured)':<30s} {total_wall:>10.1f}")
    overhead = total_wall - sub_wall
    print(f"  {'overhead / unaccounted':<30s} {overhead:>10.1f}")
    print()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_model(device)
    val_loader = load_val_loader()

    # ── Warmup: run 1 batch to JIT compile / warm caches ──
    print("\n[3/4] Warmup (1 batch) ...")
    warmup_batch = next(iter(val_loader)).to(device)
    warmup_batch = model.post_processing(warmup_batch, model.default_type)
    with torch.no_grad():
        sample_out, _, _ = model.sample(warmup_batch, num_timesteps=NUM_ODE_STEPS)
        _ = model.metric(sample_out, warmup_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("       Warmup complete.")

    # ── Profile: metric on GPU (metric_cpu=False) ──
    print("\n[4/4] Profiling ...")
    ts_gpu, tm_gpu = run_profile(model, val_loader, device, "metric on GPU", metric_on_cpu=False)
    print_table("metric on GPU (metric_cpu=False)", ts_gpu, tm_gpu, NUM_PROFILE_BATCHES)

    # ── Profile: metric on CPU (metric_cpu=True) ──
    ts_cpu, tm_cpu = run_profile(model, val_loader, device, "metric on CPU", metric_on_cpu=True)
    print_table("metric on CPU (metric_cpu=True)", ts_cpu, tm_cpu, NUM_PROFILE_BATCHES)

    # ── Final comparison ──
    gpu_wall = tm_gpu.wall_totals.get("metric_total", 0) * 1000 / NUM_PROFILE_BATCHES
    cpu_wall = tm_cpu.wall_totals.get("metric_total", 0) * 1000 / NUM_PROFILE_BATCHES
    gpu_eig = tm_gpu.wall_totals.get("cal_orbital_and_energies", 0) * 1000 / NUM_PROFILE_BATCHES
    cpu_eig = tm_cpu.wall_totals.get("cal_orbital_and_energies", 0) * 1000 / NUM_PROFILE_BATCHES
    gpu_build = tm_gpu.wall_totals.get("build_final_matrix", 0) * 1000 / NUM_PROFILE_BATCHES
    cpu_build = tm_cpu.wall_totals.get("build_final_matrix", 0) * 1000 / NUM_PROFILE_BATCHES

    print(f"{'='*72}")
    print(f"  COMPARISON: metric on GPU vs CPU  (wall time, ms/batch)")
    print(f"{'='*72}")
    print(f"  {'Component':<30s} {'GPU':>10s} {'CPU':>10s} {'CPU/GPU':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    for name, g, c in [
        ("build_final_matrix", gpu_build, cpu_build),
        ("cal_orbital_and_energies", gpu_eig, cpu_eig),
        ("metric_total", gpu_wall, cpu_wall),
    ]:
        ratio = f"{c/g:.2f}x" if g > 0 else "n/a"
        print(f"  {name:<30s} {g:>10.1f} {c:>10.1f} {ratio:>10s}")
    print()


if __name__ == "__main__":
    main()
