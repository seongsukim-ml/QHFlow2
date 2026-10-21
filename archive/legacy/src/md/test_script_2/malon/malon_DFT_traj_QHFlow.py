#!/usr/bin/env python
"""
Malonaldehyde single-point energy curve from a provided DFT trajectory (no relaxation).
Uses QHFlow (SCFlowRKSCalculator) instead of MLFF.
Outputs: malon_energies_single_point.npy, malon_rOH_single_point.npy,
         malon_scan_omat_single_point.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

# Add QHFlow src to path
sys.path.append("/root/limlab01/kaistai/25DFT/QHFlow/src")
import md.scflow_calculator_gpu


def shift_to_min(arr):
    return arr - np.nanmin(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traj",
        required=True,
        help="Path to DFT trajectory (XYZ/ASE-readable)",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to QHFlow checkpoint",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./result_malon_qhflow_single_point",
        help="Output directory",
    )
    ap.add_argument("--i_Oa", type=int, default=0, help="Acceptor O index")
    ap.add_argument("--i_Od", type=int, default=1, help="Donor O index")
    ap.add_argument("--i_H", type=int, default=5, help="Proton H index")
    ap.add_argument(
        "--plot_prefix",
        type=str,
        default="malon_scan_qhflow_single_point",
        help="Filename prefix for plot",
    )
    ap.add_argument(
        "--data_type",
        type=str,
        default="md17",
        choices=["md17", "rmd17", "qh9"],
        help="Dataset type for QHFlow model (default: md17)",
    )
    ap.add_argument(
        "--basis",
        type=str,
        default="def2-SVP",
        help="Basis set (default: def2-SVP)",
    )
    ap.add_argument(
        "--functional",
        type=str,
        default="pbe",
        help="DFT functional (default: pbe)",
    )
    ap.add_argument(
        "--units",
        type=str,
        default="ang",
        choices=["ang", "bohr"],
        help="Input units (default: ang)",
    )
    ap.add_argument(
        "--model_length_unit",
        type=str,
        default="bohr",
        choices=["ang", "bohr"],
        help="Model length unit (default: bohr for md17)",
    )
    ap.add_argument(
        "--mf_init_functional",
        type=str,
        default="pbe, pbe",
        help="Initial mean-field functional (default: pbe, pbe)",
    )
    ap.add_argument(
        "--init_density_fit",
        type=str,
        default="False",
        choices=["true", "false", "True", "False"],
        help="Initial density fitting (default: False)",
    )
    ap.add_argument(
        "--ode_steps",
        type=int,
        default=1,
        help="Number of ODE steps (default: 1)",
    )
    ap.add_argument(
        "--density_fit",
        type=str,
        default=None,
        choices=["true", "false", "True", "False"],
        help="Use density fitting: 'true' or 'false' (default: None, uses model default)",
    )
    args = ap.parse_args()

    traj_path = Path(args.traj)
    if not traj_path.exists():
        raise SystemExit(f"Trajectory not found: {traj_path}")
    if not Path(args.ckpt).exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    frames = read(str(traj_path), index=":")
    if not frames:
        raise SystemExit(f"No frames read from: {traj_path}")

    # Convert density_fit string to bool if provided
    density_fit = None
    if args.density_fit is not None:
        density_fit = args.density_fit.lower() == 'true'

    init_density_fit = None
    if args.init_density_fit is not None:
        init_density_fit = args.init_density_fit.lower() == 'true'

    # Initialize QHFlow calculator
    print(f"[*] Initializing QHFlow calculator...")
    calc = md.scflow_calculator_gpu.SCFlowRKSCalculator(
        basis=args.basis,
        functional=args.functional,
    )
    
    # Set model
    print(f"[*] Loading QHFlow model from: {args.ckpt}")
    calc.set_model(
        args.ckpt,
        data_type=args.data_type,
        units=args.units,
        model_length_unit=args.model_length_unit,
        mf_init_functional=args.mf_init_functional,
        ode_steps=args.ode_steps,
        density_fit=density_fit,
    )

    r_scan = []
    energies = []
    for idx, atoms in enumerate(frames):
        atoms.calc = calc
        try:
            r_oh = atoms.get_distance(args.i_Od, args.i_H)
            e = atoms.get_potential_energy()
            r_scan.append(r_oh)
            energies.append(e)
            print(f"{idx:4d} | r(Od-H)={r_oh:8.4f} | E={e:12.6f}")
        except Exception as exc:
            r_scan.append(np.nan)
            energies.append(np.nan)
            print(f"{idx:4d} | Fail ({exc})")

    r_scan = np.array(r_scan)
    energies = np.array(energies)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "malon_rOH_single_point.npy", r_scan)
    np.save(out_dir / "malon_energies_single_point.npy", energies)

    plt.figure(figsize=(8, 6))
    rel = shift_to_min(energies)
    plt.plot(r_scan, rel, "o-", color="navy")
    plt.xlabel("r(O-H) [A]")
    plt.ylabel("Rel. Energy [eV]")
    plt.title("Malonaldehyde Proton Transfer (QHFlow single point)")
    plt.grid(True, alpha=0.3)
    out_plot = out_dir / f"{args.plot_prefix}.png"
    plt.savefig(out_plot, dpi=200)
    print(f"[*] Done! Saved {out_plot}")


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=3 \
python /root/25DFT/QHFlow/src/md/test_script_2/malon/malon_DFT_traj_QHFlow.py \
    --traj /root/25DFT/QHFlow/src/md/test_script_2/malon/malon_scan_frames_org.xyz \
    --ckpt /root/25DFT/QHFlow/src/outputs/malondialdehyde/QHFlow_so2_v5_1_middle_b10-malondialdehyde/checkpoints/weights-epoch=79-val_loss=0.0000000.ckpt \
    --data_type md17 \
    --out_dir ./results_malon_qhflow_single_point2
"""