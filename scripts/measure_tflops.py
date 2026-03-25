#!/usr/bin/env python3
"""Measure TFLOPs for each QHFlow2 model size."""
import os
import sys
import json
import time
import torch
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def measure_tflops(model, batch, device, warmup=3, repeat=10):
    """Measure forward TFLOPs."""
    from torch.utils.flop_counter import FlopCounterMode

    model.eval().to(device)
    H = batch.init_ham_t.to(device) if hasattr(batch, "init_ham_t") else None

    # Move batch to device
    batch = batch.to(device)

    # Count FLOPs
    flop_counter = FlopCounterMode(display=False)
    with flop_counter, torch.no_grad():
        model(batch, H, keep_blocks=True)
    flops_fwd = flop_counter.get_total_flops()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(batch, H, keep_blocks=True)
    torch.cuda.synchronize()

    # Measure time
    times = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            model(batch, H, keep_blocks=True)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    gpu_ms = sum(times) / len(times)
    tflops_fwd = (flops_fwd / 1e12) / (gpu_ms / 1000.0)

    return {
        "gflops_fwd": flops_fwd / 1e9,
        "gpu_ms": round(gpu_ms, 2),
        "tflops_fwd": round(tflops_fwd, 2),
        "tflops_total_est": round(tflops_fwd * 3, 2),
    }


def main():
    device = torch.device("cuda")
    proj = Path(__file__).resolve().parents[1]

    # Load a real batch
    from torch_geometric.loader import DataLoader
    from qhflow2.dataset_module.qh9_datasets_split import QH9Stable

    ds = QH9Stable(
        root=str(proj / "dataset"),
        split="random",
    )
    # Use last 32 samples as test batch
    loader = DataLoader(ds[-32:], batch_size=32, shuffle=False)
    batch = next(iter(loader))
    print(f"Loaded QH9 batch: {batch.num_graphs} graphs, {batch.num_nodes} atoms")

    # Model configs
    sizes = {
        "small": {"hidden_size": 64, "bottle_hidden_size": 32, "num_gnn_layers": 3, "num_ham_gnn_layers": 2},
        "middle": {"hidden_size": 128, "bottle_hidden_size": 64, "num_gnn_layers": 3, "num_ham_gnn_layers": 2},
        "large": {"hidden_size": 256, "bottle_hidden_size": 64, "num_gnn_layers": 4, "num_ham_gnn_layers": 2},
        "extra_large": {"hidden_size": 512, "bottle_hidden_size": 128, "num_gnn_layers": 4, "num_ham_gnn_layers": 3},
    }

    from qhflow2.models import get_model, get_default_model_args

    results = {}
    for name, cfg in sizes.items():
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"{'='*50}")

        args_dict = get_default_model_args("qh9")
        args_dict["version"] = "QHFlow_so2_v5_1"
        args_dict.update(cfg)
        args = SimpleNamespace(**args_dict)

        model = get_model(args)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        try:
            r = measure_tflops(model, batch.clone(), device)
            r["params"] = n_params
            results[name] = r
            print(f"  GFLOPs (fwd):  {r['gflops_fwd']:.1f}")
            print(f"  GPU time:      {r['gpu_ms']:.1f} ms")
            print(f"  TFLOPS (fwd):  {r['tflops_fwd']:.2f}")
            print(f"  TFLOPS (3x):   {r['tflops_total_est']:.2f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"params": n_params, "error": str(e)}

        del model
        torch.cuda.empty_cache()

    # Save
    out = proj / "eval_results" / "tflops.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'Model':<15} {'Params':>12} {'GFLOPs':>10} {'ms/batch':>10} {'TFLOPS(fwd)':>12} {'TFLOPS(3x)':>12}")
    print("-" * 75)
    for name, r in results.items():
        if "error" not in r:
            print(f"{name:<15} {r['params']:>12,} {r['gflops_fwd']:>10.1f} {r['gpu_ms']:>10.1f} {r['tflops_fwd']:>12.2f} {r['tflops_total_est']:>12.2f}")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
