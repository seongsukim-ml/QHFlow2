"""TDN vs Baseline 중간 출력 비교 분석.

Expansion layer의 입력(fii, fij, weights, bias)과 출력(H blocks)을
baseline과 TDN에서 각각 추출하여 비교한다.

Usage:
    python scripts/analyze_tdn.py

분석 항목:
  1. fc_ii/fc_ij weight 분포 비교 (num_path_weight 차이)
  2. expand 입력 feature (fii, fij) 통계
  3. expand 출력 (H block) 분포 비교
  4. per-l_in weight 기여도 분석 (TDN 병목 진단)
  5. Gradient norm per layer
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project src to path
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

OUTPUTS = PROJECT / "src" / "outputs" / "QH9Stable-random"
BASELINE_DIR = OUTPUTS / "QHFlow_so2_v5_1-baseline-QH9Stable-random"
TDN_DIR = OUTPUTS / "QHFlow_so2_v5_1_tdn-tdn-QH9Stable-random"


def load_model_from_ckpt(ckpt_path, config_dir):
    """Load model from checkpoint + hydra config."""
    from omegaconf import OmegaConf
    from qhflow2.models import get_model

    hydra_cfg = OmegaConf.load(config_dir / ".hydra" / "config.yaml")
    model = get_model(hydra_cfg.model)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # Strip 'model.' prefix if present (Lightning wrapping)
    state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
             for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model, hydra_cfg


def make_synthetic_inputs(model):
    """Create synthetic inputs for expand_ii/ij layers (no dataset needed)."""
    B = 16  # synthetic batch
    expand = model.expand_ii["hamiltonian"]
    x_dim = expand.irrep_in.dim
    hidden = model.hidden_size

    fii = torch.randn(B, x_dim)
    node_attr = torch.randn(B, hidden)
    node_pair_attr = torch.randn(B, hidden * 2)
    return fii, node_attr, node_pair_attr


def stat(name, t):
    """Print statistics of a tensor."""
    if t is None:
        print(f"  {name}: None")
        return
    t = t.float()
    print(f"  {name}: shape={list(t.shape)}, "
          f"mean={t.mean():.6f}, std={t.std():.6f}, "
          f"min={t.min():.6f}, max={t.max():.6f}, "
          f"abs_mean={t.abs().mean():.6f}")


def analyze_fc_weights(model, label):
    """Analyze fc_ii and fc_ij weight distributions."""
    print(f"\n{'='*60}")
    print(f"  FC Weight Analysis: {label}")
    print(f"{'='*60}")
    for name in ["hamiltonian"]:
        fc_ii = model.fc_ii[name]
        fc_ij = model.fc_ij[name]
        fc_ii_bias = model.fc_ii_bias[name]
        fc_ij_bias = model.fc_ij_bias[name]

        # Last linear layer output dim = num_path_weight
        last_ii = list(fc_ii.children())[-1]
        last_ij = list(fc_ij.children())[-1]
        print(f"  fc_ii output dim: {last_ii.out_features}")
        print(f"  fc_ij output dim: {last_ij.out_features}")
        stat("fc_ii last.weight", last_ii.weight.data)
        stat("fc_ij last.weight", last_ij.weight.data)

        last_ii_b = list(fc_ii_bias.children())[-1]
        last_ij_b = list(fc_ij_bias.children())[-1]
        print(f"  fc_ii_bias output dim: {last_ii_b.out_features}")
        print(f"  fc_ij_bias output dim: {last_ij_b.out_features}")


def hook_analysis(model, label):
    """Run expansion layers with synthetic data to capture intermediate values."""
    print(f"\n{'='*60}")
    print(f"  Intermediate Output Analysis: {label}")
    print(f"{'='*60}")

    fii, node_attr, node_pair_attr = make_synthetic_inputs(model)
    model.eval()
    captured = {}

    with torch.no_grad():
        for key in ["hamiltonian"]:
            # Diagonal: expand_ii(fii, fc_ii(node_attr), fc_ii_bias(node_attr))
            w_ii = model.fc_ii[key](node_attr)
            b_ii = model.fc_ii_bias[key](node_attr)
            H_ii = model.expand_ii[key](fii, w_ii, b_ii)

            captured[f"fc_ii_{key}"] = w_ii
            captured[f"fc_ii_bias_{key}"] = b_ii
            captured[f"expand_ii_{key}"] = H_ii

            # Off-diagonal: expand_ij(fii, fc_ij(node_pair), fc_ij_bias(node_pair))
            w_ij = model.fc_ij[key](node_pair_attr)
            b_ij = model.fc_ij_bias[key](node_pair_attr)
            H_ij = model.expand_ij[key](fii, w_ij, b_ij)

            captured[f"fc_ij_{key}"] = w_ij
            captured[f"fc_ij_bias_{key}"] = b_ij
            captured[f"expand_ij_{key}"] = H_ij

    for name, tensor in sorted(captured.items()):
        stat(name, tensor)

    return captured


def compare_weight_utilization(captured_baseline, captured_tdn):
    """Compare how weight capacity is utilized."""
    print(f"\n{'='*60}")
    print(f"  Weight Utilization Comparison")
    print(f"{'='*60}")

    for key in ["fc_ii_hamiltonian", "fc_ij_hamiltonian"]:
        if key in captured_baseline and key in captured_tdn:
            w_base = captured_baseline[key]
            w_tdn = captured_tdn[key]
            print(f"\n  {key}:")
            print(f"    Baseline: dim={w_base.shape[-1]}, "
                  f"effective_rank≈{effective_rank(w_base):.1f}")
            print(f"    TDN:      dim={w_tdn.shape[-1]}, "
                  f"effective_rank≈{effective_rank(w_tdn):.1f}")


def effective_rank(x):
    """Estimate effective rank via singular value entropy."""
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    try:
        s = torch.linalg.svdvals(x.float())
        s = s / s.sum()
        s = s[s > 1e-10]
        entropy = -(s * s.log()).sum()
        return entropy.exp().item()
    except Exception:
        return float('nan')


def analyze_per_l_contribution(model_tdn, captured_tdn):
    """Analyze per-angular-momentum contribution in TDN shared weights."""
    print(f"\n{'='*60}")
    print(f"  Per-l_in Contribution Analysis (TDN)")
    print(f"{'='*60}")

    if "hamiltonian" not in model_tdn.expand_ii:
        print("  No expand_ii found")
        return
    expand = model_tdn.expand_ii["hamiltonian"]

    fc_out = captured_tdn.get("fc_ii_hamiltonian")
    if fc_out is None:
        print("  No fc_ii output captured")
        return

    c = expand.mul_in
    mu1, mu2 = expand.mul_max1, expand.mul_max2
    D_in = expand.D_in

    print(f"  c={c}, mu1={mu1}, mu2={mu2}, D_in={D_in}")
    print(f"  num_path_weight={expand.num_path_weight} (c*mu1*mu2={c*mu1*mu2})")

    # Analyze which (j,k) blocks get most weight energy
    print(f"\n  CG block contributions:")
    for (j_idx, k_idx), grp in expand._groups.items():
        l1, l2 = grp['l1'], grp['l2']
        n_paths = len(grp['paths'])
        l_ins = [p['l_in'] for p in grp['paths']]
        total_P = sum(p['ang_size'] for p in grp['paths'])

        M_name = f'M_{j_idx}_{k_idx}'
        if hasattr(expand, M_name):
            M = getattr(expand, M_name)
            frob = M.norm().item()
        else:
            frob = float('nan')

        print(f"    block ({l1},{l2}): {n_paths} paths, "
              f"l_in={l_ins}, P={total_P}, ||M||={frob:.4f}")


def main():
    # Check checkpoints exist
    baseline_ckpt = BASELINE_DIR / "checkpoints" / "last.ckpt"
    tdn_ckpt = TDN_DIR / "checkpoints" / "last.ckpt"

    for p, name in [(baseline_ckpt, "Baseline"), (tdn_ckpt, "TDN")]:
        if not p.exists():
            print(f"ERROR: {name} checkpoint not found: {p}")
            return

    print("Loading models...")
    model_base, cfg_base = load_model_from_ckpt(baseline_ckpt, BASELINE_DIR)
    model_tdn, cfg_tdn = load_model_from_ckpt(tdn_ckpt, TDN_DIR)

    # Parameter comparison
    n_base = sum(p.numel() for p in model_base.parameters())
    n_tdn = sum(p.numel() for p in model_tdn.parameters())
    print(f"\nBaseline: {n_base/1e6:.2f}M params")
    print(f"TDN:      {n_tdn/1e6:.2f}M params")

    # FC weight analysis
    analyze_fc_weights(model_base, "Baseline")
    analyze_fc_weights(model_tdn, "TDN")

    # Hook analysis with synthetic data
    captured_base = hook_analysis(model_base, "Baseline")
    captured_tdn = hook_analysis(model_tdn, "TDN")

    # Weight utilization
    compare_weight_utilization(captured_base, captured_tdn)

    # Per-l contribution (TDN only)
    analyze_per_l_contribution(model_tdn, captured_tdn)

    print(f"\n{'='*60}")
    print(f"  Analysis Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
