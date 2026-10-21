#!/usr/bin/env python
"""
Malonaldehyde Proton Transfer (Corrected Cis-Geometry).
- O–O distance ~2.55 Å (ring-like, H-bonded form).
- Starts from a stable enol; scans r(Od–H) with O–O fixed.
Outputs: malon_scan_corrected.png, malon_energies.npy, malon_rOH.npy
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixBondLengths
from ase.io import write
from ase.optimize import BFGS

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.52917721092


class PySCFCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, method="dft", basis="def2-svp", xc="PBE", charge=0, spin=0):
        super().__init__()
        self.method = method
        self.basis = basis
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self._last_dm = None
        self._use_gpu = False
        try:
            import gpu4pyscf  # noqa: F401

            self._use_gpu = True
        except Exception:
            self._use_gpu = False

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from pyscf import dft, gto

        Z = [int(z) for z in atoms.get_atomic_numbers()]
        pos = atoms.get_positions()
        mol = gto.M(atom=[(z, tuple(p)) for z, p in zip(Z, pos)], unit="Angstrom", basis=self.basis, verbose=0)

        mf = None
        if self._use_gpu:
            try:
                from gpu4pyscf.dft import RKS

                mf = RKS(mol, xc=self.xc)
            except Exception:
                mf = None
        if mf is None:
            mf = dft.RKS(mol, xc=self.xc)

        if self._last_dm is not None:
            mf.kernel(dm0=self._last_dm)
        else:
            mf.kernel()
        if mf.converged:
            self._last_dm = mf.make_rdm1()

        self.results["energy"] = float(mf.e_tot) * HARTREE_TO_EV
        grad = mf.nuc_grad_method().kernel()
        self.results["forces"] = -grad * (HARTREE_TO_EV / BOHR_TO_ANG)


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
    atoms = get_cis_malonaldehyde()
    # Indices: Oa=0, Od=1, H=5 (based on order above)
    i_Oa, i_Od, i_H = 0, 1, 5
    print(f"[*] Indices: Donor(O)={i_Od}, Acceptor(O)={i_Oa}, Proton(H)={i_H}")

    calc = PySCFCalc(xc="PBE", basis="def2-svp")
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

    np.save("malon_energies.npy", energies)
    np.save("malon_rOH.npy", r_scan)
    if frames:
        write("malon_scan_frames.xyz", frames)
        write("malon_scan_frames.traj", frames)

    plt.figure(figsize=(8, 6))
    plt.plot(r_scan, energies, "o-", color="navy")
    plt.xlabel("r(O-H) [A]")
    plt.ylabel("Rel. Energy [eV]")
    plt.title("Malonaldehyde Proton Transfer (Corrected)")
    plt.grid(True, alpha=0.3)
    plt.savefig("malon_scan_corrected.png")
    print("[*] Done! Saved malon_scan_corrected.png")


if __name__ == "__main__":
    main()
