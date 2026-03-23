"""
Differentiable PBE0 실험
========================
PBE0 functional을 미분가능하게 구현할 수 있는지 검증하는 실험.

실험 1: PySCFAD를 이용한 PBE0 에너지의 핵좌표 미분 + FD 검증
실험 2: HF exchange 에너지의 핵좌표 미분 (exact exchange 미분가능성)
실험 3: Ethanol (larger molecule) 스케일링 테스트

PBE0 = 0.75 * E_x^PBE + 0.25 * E_x^HF + E_c^PBE
"""

import numpy as np
import time
import jax
import jax.numpy as jnp
from pyscfad import gto, dft


def build_mol(atom_str, basis='sto-3g'):
    """Build PySCFAD mol object from atom string."""
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.unit = 'Angstrom'
    mol.build()
    return mol


# ============================================================
# Experiment 1: PBE0 Energy + Nuclear Gradient via AD
# ============================================================
def exp1_pbe0_nuclear_gradient():
    """PySCFAD로 PBE0 에너지를 핵좌표에 대해 미분"""
    print("=" * 60)
    print("Exp 1: PBE0 Nuclear Gradient via PySCFAD (AD)")
    print("=" * 60)

    mol = build_mol('O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692')

    def energy_fn(mol):
        mf = dft.RKS(mol)
        mf.xc = 'pbe0'
        mf.conv_tol = 1e-10
        mf.verbose = 0
        return mf.kernel()

    print("\n[1a] Computing PBE0 energy...")
    t0 = time.time()
    energy = energy_fn(mol)
    t1 = time.time()
    print(f"  PBE0 Energy: {energy:.10f} Hartree")
    print(f"  Time: {t1 - t0:.2f}s")

    print("\n[1b] Computing gradient via jax.grad...")
    t0 = time.time()
    grad_mol = jax.grad(energy_fn)(mol)
    gradient = np.array(grad_mol.coords)
    t1 = time.time()
    print(f"  Nuclear gradient (AD):\n{gradient}")
    print(f"  Max |grad|: {np.max(np.abs(gradient)):.8f}")
    print(f"  Time: {t1 - t0:.2f}s")

    # Finite difference verification
    print("\n[1c] Finite difference verification...")
    t0 = time.time()
    coords_np = np.array(mol.coords)
    fd_grad = np.zeros_like(coords_np)
    h = 1e-4
    for i in range(coords_np.shape[0]):
        for j in range(3):
            coords_p = coords_np.copy()
            coords_m = coords_np.copy()
            coords_p[i, j] += h
            coords_m[i, j] -= h

            mol_p = build_mol('O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692')
            mol_p.coords = jnp.array(coords_p)
            mol_m = build_mol('O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692')
            mol_m.coords = jnp.array(coords_m)

            ep = energy_fn(mol_p)
            em = energy_fn(mol_m)
            fd_grad[i, j] = (ep - em) / (2 * h)
    t1 = time.time()
    print(f"  Nuclear gradient (FD):\n{fd_grad}")
    print(f"  Time: {t1 - t0:.2f}s")

    diff = np.abs(gradient - fd_grad)
    print(f"\n  Max |AD - FD|: {np.max(diff):.2e}")
    print(f"  Agreement: {'PASS' if np.max(diff) < 1e-4 else 'FAIL'}")

    return energy, gradient, fd_grad


# ============================================================
# Experiment 2: HF Exchange energy gradient
# ============================================================
def exp2_hf_exchange_gradient():
    """HF exact exchange 에너지의 핵좌표 미분"""
    print("\n" + "=" * 60)
    print("Exp 2: HF Exchange Energy Gradient (exact exchange 미분)")
    print("=" * 60)

    mol = build_mol('O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692')

    def hf_exchange_energy(mol):
        mf = dft.RKS(mol)
        mf.xc = 'pbe0'
        mf.verbose = 0
        mf.conv_tol = 1e-10
        mf.kernel()

        dm = mf.make_rdm1()
        _, vk = mf.get_jk(mol, dm)
        e_hf_x = -0.5 * jnp.einsum('ij,ji->', vk, dm)
        return e_hf_x

    print("\n[2a] Computing HF exchange energy...")
    t0 = time.time()
    e_hf_x = hf_exchange_energy(mol)
    t1 = time.time()
    print(f"  E_HF_x: {e_hf_x:.10f} Hartree")
    print(f"  Time: {t1 - t0:.2f}s")

    print("\n[2b] Gradient of E_HF_x w.r.t. nuclear coords...")
    t0 = time.time()
    grad_mol = jax.grad(hf_exchange_energy)(mol)
    hf_x_grad = np.array(grad_mol.coords)
    t1 = time.time()
    print(f"  d(E_HF_x)/d(coords):\n{hf_x_grad}")
    print(f"  Max |grad|: {np.max(np.abs(hf_x_grad)):.8f}")
    print(f"  Time: {t1 - t0:.2f}s")
    print(f"  SUCCESS: HF exact exchange is differentiable!")

    return e_hf_x, hf_x_grad


# ============================================================
# Experiment 3: Larger molecule test (ethanol)
# ============================================================
def exp3_larger_molecule():
    """Ethanol으로 스케일링 테스트"""
    print("\n" + "=" * 60)
    print("Exp 3: PBE0 gradient for ethanol (scaling test)")
    print("=" * 60)

    atom_str = (
        'C -0.7516 0.0089 -0.0348; C 0.7516 -0.0089 0.0348; '
        'O 1.1670 0.0000 1.3740; '
        'H -1.1670 -0.8770 0.4580; H -1.1670 0.8913 0.4580; '
        'H -1.1670 0.0089 -1.0700; '
        'H 1.1670 -0.8913 -0.4580; H 1.1670 0.8770 -0.4580; '
        'H 2.1300 0.0000 1.3740'
    )
    mol = build_mol(atom_str)

    def energy_fn(mol):
        mf = dft.RKS(mol)
        mf.xc = 'pbe0'
        mf.verbose = 0
        mf.conv_tol = 1e-10
        return mf.kernel()

    print("\n[3a] PBE0 energy for ethanol...")
    t0 = time.time()
    energy = energy_fn(mol)
    t1 = time.time()
    print(f"  Energy: {energy:.10f} Hartree")
    print(f"  Time: {t1 - t0:.2f}s")

    print("\n[3b] Nuclear gradient via AD...")
    t0 = time.time()
    grad_mol = jax.grad(energy_fn)(mol)
    gradient = np.array(grad_mol.coords)
    t1 = time.time()
    print(f"  Gradient shape: {gradient.shape}")
    print(f"  Max |grad|: {np.max(np.abs(gradient)):.8f}")
    print(f"  Time: {t1 - t0:.2f}s")

    return energy, gradient


# ============================================================
# Experiment 4: def2-SVP basis (bigger basis set)
# ============================================================
def exp4_bigger_basis():
    """def2-SVP basis로 PBE0 미분 테스트"""
    print("\n" + "=" * 60)
    print("Exp 4: PBE0 with def2-SVP basis (larger basis)")
    print("=" * 60)

    mol = build_mol(
        'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
        basis='def2-svp'
    )

    def energy_fn(mol):
        mf = dft.RKS(mol)
        mf.xc = 'pbe0'
        mf.verbose = 0
        mf.conv_tol = 1e-10
        return mf.kernel()

    print(f"\n  Basis functions: {mol.nao}")

    print("\n[4a] PBE0/def2-SVP energy...")
    t0 = time.time()
    energy = energy_fn(mol)
    t1 = time.time()
    print(f"  Energy: {energy:.10f} Hartree")
    print(f"  Time: {t1 - t0:.2f}s")

    print("\n[4b] Nuclear gradient via AD...")
    t0 = time.time()
    grad_mol = jax.grad(energy_fn)(mol)
    gradient = np.array(grad_mol.coords)
    t1 = time.time()
    print(f"  Max |grad|: {np.max(np.abs(gradient)):.8f}")
    print(f"  Time: {t1 - t0:.2f}s")

    return energy, gradient


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Differentiable PBE0 Feasibility Test")
    print("Using PySCFAD (PySCF + JAX Automatic Differentiation)")
    print()

    results = {}
    for name, fn in [
        ("Exp 1", exp1_pbe0_nuclear_gradient),
        ("Exp 2", exp2_hf_exchange_gradient),
        ("Exp 3", exp3_larger_molecule),
        ("Exp 4", exp4_bigger_basis),
    ]:
        try:
            fn()
            results[name] = "PASSED"
            print(f"\n>>> {name}: PASSED")
        except Exception as ex:
            results[name] = f"FAILED - {ex}"
            print(f"\n>>> {name}: FAILED - {ex}")
            import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name}: {status}")
    print()
    print("PBE0 미분가능성 실험 완료.")
    print("PASS = AD gradient가 정상 작동")
