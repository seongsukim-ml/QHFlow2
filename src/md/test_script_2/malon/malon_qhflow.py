#!/usr/bin/env python
"""
Malonaldehyde Proton Transfer (Corrected Cis-Geometry) using QHFlow.
- O–O distance ~2.55 Å (ring-like, H-bonded form).
- Starts from a stable enol; scans r(Od–H) with O–O fixed.
Outputs: malon_scan_corrected.png, malon_energies.npy, malon_rOH.npy
"""
from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.constraints import FixBondLengths
from ase.io import write
from ase.optimize import BFGS

# Add QHFlow src to path
sys.path.append("/root/limlab01/kaistai/25DFT/QHFlow/src")
import md.scflow_calculator_gpu

HARTREE_TO_EV = 27.211386245988


def get_cis_malonaldehyde():
    print("[*] Using CORRECTED Cis-Malonaldehyde Coordinates (O-O ~2.55 Å)...")
    # 수정됨: O3 -> O2 (산소 2개, 탄소 3개, 수소 3개 = 총 8개 + 뒤에 1개 추가)
    atoms = Atoms(
        symbols="O2C3H3",  # <--- 여기가 범인입니다! O3를 O2로 바꿔주세요.
        positions=[
            [1.272, -1.077, 0.000],   # O1 (Acceptor)
            [-1.272, -1.077, 0.000],  # O2 (Donor)
            [1.229, 0.199, 0.000],    # C1
            [0.000, 0.916, 0.000],    # C2
            [-1.229, 0.199, 0.000],   # C3
            [-0.850, -0.900, 0.000],  # H_proton
            [2.168, 0.772, 0.000],    # H (C1)
            [0.000, 2.008, 0.000],    # H (C2)
        ],
    )
    atoms += Atoms("H", positions=[[-2.168, 0.772, 0.000]])  # H on C3
    return atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to QHFlow checkpoint",
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
    ap.add_argument(
        "--filtering",
        type=str,
        default="False",
        choices=["true", "false", "True", "False"],
        help="Use filtering (default: False)",
    )
    args = ap.parse_args()

    if not Path(args.ckpt).exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    # Convert density_fit string to bool if provided
    density_fit = None
    if args.density_fit is not None:
        density_fit = args.density_fit.lower() == 'true'

    init_density_fit = None
    if args.init_density_fit is not None:
        init_density_fit = args.init_density_fit.lower() == 'true'

    filtering = args.filtering.lower() == 'true'

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"malon_qhflow_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"[*] Output directory: {output_dir}")
    
    # Save command used to run the script
    command = " ".join(shlex.quote(arg) for arg in sys.argv)
    with open(output_dir / "command.txt", "w") as f:
        f.write(command + "\n")

    atoms = get_cis_malonaldehyde()
    # Indices: Oa=0, Od=1, H=5 (based on order above)
    i_Oa, i_Od, i_H = 0, 1, 5
    print(f"[*] Indices: Donor(O)={i_Od}, Acceptor(O)={i_Oa}, Proton(H)={i_H}")

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
        init_density_fit=init_density_fit,
        filtering=filtering,
    )
    
    atoms.calc = calc

    print("[*] Optimizing initial structure...")
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.03)

    d_OO = atoms.get_distance(i_Od, i_Oa)
    print(f"[*] Fixed O-O distance: {d_OO:.3f} A (Target ~2.5-2.6 A)")
    if d_OO > 3.0:
        print("[!] Warning: Structure unfolded; check coordinates.")
        return

    # Extend range to 0.90 and make it denser (4x points ~ 60)
    r_scan = np.linspace(0.90, 1.70, 60)
    energies = []
    frames = []

    print(f"{'r(Od-H)':>10} | {'Energy (eV)':>12}")
    print("-" * 25)

    current = atoms.copy()
    for r_target in r_scan:
        geom = current.copy()
        geom.calc = calc
        pos_d = geom.positions[i_Od]
        pos_a = geom.positions[i_Oa]
        vec = pos_a - pos_d
        vec = vec / np.linalg.norm(vec)
        geom.positions[i_H] = pos_d + vec * r_target

        cons = [
            FixBondLengths([(i_Od, i_H)]),
            FixBondLengths([(i_Od, i_Oa)]),
        ]
        geom.set_constraint(cons)

        try:
            opt_scan = BFGS(geom, logfile=None)
            opt_scan.run(fmax=0.05, steps=80)
            e = geom.get_potential_energy()
            energies.append(e)
            current = geom
            frames.append(geom.copy())
            print(f"{r_target:10.3f} | {e:12.4f}")
        except Exception:
            energies.append(np.nan)
            print(f"{r_target:10.3f} | {'Fail':>12}")


    energies = np.array(energies)
    if not np.all(np.isnan(energies)):
        energies -= np.nanmin(energies)

    np.save(output_dir / "malon_energies.npy", energies)
    np.save(output_dir / "malon_rOH.npy", r_scan)
    if frames:
        write(output_dir / "malon_scan_frames_qhflow.xyz", frames)
        write(output_dir / "malon_scan_frames_qhflow.traj", frames)

    plt.figure(figsize=(8, 6))
    plt.plot(r_scan, energies, "o-", color="navy")
    plt.xlabel("r(O-H) [A]")
    plt.ylabel("Rel. Energy [eV]")
    plt.title("Malonaldehyde Proton Transfer (QHFlow)")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "malon_scan_qhflow.png", dpi=200)
    print(f"[*] Done! Saved outputs to {output_dir}/")


if __name__ == "__main__":
    main()

"""
Example usage:
CUDA_VISIBLE_DEVICES=3 \
python /root/25DFT/QHFlow/src/md/test_script_2/malon/malon_qhflow.py \
    --ckpt /root/25DFT/QHFlow/src/outputs/malondialdehyde/QHFlow_so2_v5_1_middle_b10-malondialdehyde/checkpoints/weights-epoch=79-val_loss=0.0000000.ckpt \
    --data_type md17 \
    --basis def2-SVP \
    --functional pbe \
    --units ang \
    --model_length_unit bohr \
    --init_density_fit False \
    --filtering true
"""
