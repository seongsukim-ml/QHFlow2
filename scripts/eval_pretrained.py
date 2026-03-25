#!/usr/bin/env python3
"""Evaluate all pre-trained QH9 checkpoints and measure TFLOPs."""
import subprocess
import sys
import json
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass

PROJ = Path("/home1/irteam/data-vol1/projects/QHFlow2")
PY = "/home1/irteam/data-vol1/conda/envs/qhflow/bin/python"
CKPT_ROOT = PROJ / "ckpt" / "QH9"
RESULTS_DIR = PROJ / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── QH9 evaluation configs ──────────────────────────────────────────
QH9_EVALS = []
for split_dir, (dataset, split) in {
    "QH9Stable-random": ("QH9Stable", "random"),
    "QH9Stable-size_ood": ("QH9Stable", "size_ood"),
    "QH9Dynamic-300k-geometry": ("QH9Dynamic", "geometry"),
    "QH9Dynamic-300k-mol": ("QH9Dynamic", "mol"),
}.items():
    for size, model_cfg in {
        "small": "QHFlow_so2_v5_1_small",
        "middle": "QHFlow_so2_v5_1_middle",
        "large": "QHFlow_so2_v5_1_large",
        "extra_large": "QHFlow_so2_v5_1_extra_large",
    }.items():
        # Find ckpt: ckpt/QH9/{split_dir}/QHFlow_so2_v5_1_{size}-{split_dir}/*.ckpt
        ckpt_dir = CKPT_ROOT / split_dir / f"QHFlow_so2_v5_1_{size}-{split_dir}"
        ckpts = list(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
        if not ckpts:
            continue
        ckpt = ckpts[0]
        QH9_EVALS.append({
            "split_dir": split_dir,
            "size": size,
            "dataset": dataset,
            "split": split,
            "model": model_cfg,
            "ckpt": str(ckpt),
        })


def run_eval(cfg, gpu_id=5):
    """Run a single evaluation and return metrics."""
    split_dir = cfg["split_dir"]
    size = cfg["size"]
    out_dir = RESULTS_DIR / split_dir / size
    out_dir.mkdir(parents=True, exist_ok=True)

    # Symlink ckpt to avoid Hydra = parsing
    link = Path(f"/tmp/qhflow2_eval_{split_dir}_{size}.ckpt")
    link.unlink(missing_ok=True)
    link.symlink_to(cfg["ckpt"])

    print(f"\n{'='*60}")
    print(f"EVAL: {split_dir} / {size}")
    print(f"  ckpt: {cfg['ckpt']}")
    print(f"{'='*60}")

    cmd = [
        PY, "-m", "qhflow2.experiment.train_qh9",
        f"mode=predict",
        f"dataset={cfg['dataset']}",
        f"dataset.split={cfg['split']}",
        f"model={cfg['model']}",
        f"wandb.mode=disabled",
        f"ckpt={link}",
        f"hydra.run.dir={out_dir}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJ), env=env,
        timeout=1800,  # 30 min max per eval
    )
    elapsed = time.time() - t0

    # Save logs
    log_path = out_dir / "eval.log"
    with open(log_path, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    # Parse metrics
    metrics = {"split": split_dir, "size": size, "elapsed_s": elapsed}

    report_path = out_dir / "comparison_report.txt"
    if report_path.exists():
        text = report_path.read_text()
        print(f"  {text.strip()}")
        # Parse MAE from "hamiltonian: MAE=0.000016"
        m = re.search(r"MAE=([\d.]+)", text)
        if m:
            metrics["hamiltonian_mae_ha"] = float(m.group(1))
            metrics["hamiltonian_mae_mev"] = float(m.group(1)) * 27211.386  # Ha to meV

    prop_path = out_dir / "property_metrics.npz"
    if prop_path.exists():
        import numpy as np
        d = np.load(str(prop_path), allow_pickle=True)
        for k in d.files:
            v = d[k]
            if v.ndim == 0:
                metrics[k] = float(v.item())
            else:
                metrics[f"{k}_mean"] = float(v.mean())

    if result.returncode != 0:
        metrics["error"] = result.stderr[-500:] if result.stderr else "unknown"
        print(f"  ERROR (exit={result.returncode})")

    return metrics


def main():
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "5"))
    print(f"GPU: {gpu_id}")
    print(f"Found {len(QH9_EVALS)} QH9 evaluations")

    all_results = []
    for cfg in QH9_EVALS:
        try:
            metrics = run_eval(cfg, gpu_id=gpu_id)
            all_results.append(metrics)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            all_results.append({"split": cfg["split_dir"], "size": cfg["size"], "error": str(e)})

    # Save all results
    results_path = RESULTS_DIR / "qh9_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Split':<30} {'Size':<15} {'H_MAE(meV)':>12} {'HOMO(meV)':>10} {'LUMO(meV)':>10} {'GAP(meV)':>10} {'Time(s)':>8}")
    print("-" * 80)
    for r in all_results:
        if "error" in r:
            print(f"{r['split']:<30} {r['size']:<15} {'ERROR':>12}")
            continue
        h_mae = r.get("hamiltonian_mae_mev", float("nan"))
        homo = r.get("homo_errors_mean", float("nan"))
        lumo = r.get("lumo_errors_mean", float("nan"))
        gap = r.get("gap_errors_mean", float("nan"))
        t = r.get("elapsed_s", 0)
        print(f"{r['split']:<30} {r['size']:<15} {h_mae:>12.2f} {homo:>10.2f} {lumo:>10.2f} {gap:>10.2f} {t:>8.0f}")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
