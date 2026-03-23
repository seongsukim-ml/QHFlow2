"""
Equivariant Normalization 검증 실험
====================================

Hamiltonian 블록에 대한 normalization이 SO(3) equivariance를 보존하는지 검증.

검증 조건: normalize(rotate(H)) == rotate(normalize(H))

핵심 원리:
  - rotation invariant quantity(trace, Frobenius norm 등)로 나누면 equivariance 보존
  - rotation invariant를 scalar 부분에서만 빼면 equivariance 보존
  - element-wise mean subtraction은 preferred direction을 도입 → equivariance 깨짐

의존성: torch, numpy, scipy (e3nn/sympy 불필요)
"""

import torch
import numpy as np
from functools import lru_cache
from scipy.spatial.transform import Rotation as SciRotation

torch.set_default_dtype(torch.float64)


# =====================================================================
# Wigner-D for real spherical harmonics
# =====================================================================

@lru_cache(maxsize=32)
def _angular_momentum_generators(l: int):
    """Rotation generators G_x, G_y, G_z for real SH of degree l.

    D(R) = exp(θ n·G) where G are real antisymmetric matrices.
    Eigenvalues of -(Gx²+Gy²+Gz²) = l(l+1).
    """
    dim = 2 * l + 1

    # Complex basis: J_z diagonal, J_± raise/lower
    Jz = torch.diag(torch.arange(-l, l + 1, dtype=torch.float64))
    Jp = torch.zeros(dim, dim, dtype=torch.float64)
    Jm = torch.zeros(dim, dim, dtype=torch.float64)
    for idx in range(dim):
        m = idx - l
        if m + 1 <= l:
            Jp[idx + 1, idx] = np.sqrt(l * (l + 1) - m * (m + 1))
        if m - 1 >= -l:
            Jm[idx - 1, idx] = np.sqrt(l * (l + 1) - m * (m - 1))
    Jx_c = (Jp + Jm) / 2
    Jy_c = (Jp - Jm) / (2j)

    # Complex-to-real unitary
    U = torch.zeros(dim, dim, dtype=torch.complex128)
    for idx in range(dim):
        m = idx - l
        if m > 0:
            U[idx, m + l] = ((-1) ** m) / np.sqrt(2)
            U[idx, -m + l] = 1.0 / np.sqrt(2)
        elif m == 0:
            U[idx, l] = 1.0
        else:
            am = abs(m)
            U[idx, am + l] = 1j * ((-1) ** (am + 1)) / np.sqrt(2)
            U[idx, -am + l] = 1j / np.sqrt(2)

    # G = -iJ in real basis (G are real antisymmetric)
    Gx = (-1j * (U @ Jx_c.to(torch.complex128) @ U.conj().T)).real
    Gy = (-1j * (U @ Jy_c @ U.conj().T)).real
    Gz = (-1j * (U @ Jz.to(torch.complex128) @ U.conj().T)).real
    return Gx, Gy, Gz


def wigner_D(l: int, R: torch.Tensor) -> torch.Tensor:
    """Wigner-D matrix for real SH: D(R) = exp(θ n·G)."""
    if l == 0:
        return torch.ones(1, 1, dtype=torch.float64)

    rotvec = SciRotation.from_matrix(R.numpy()).as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle < 1e-15:
        return torch.eye(2 * l + 1, dtype=torch.float64)

    axis = rotvec / angle
    Gx, Gy, Gz = _angular_momentum_generators(l)
    Gn = axis[0] * Gx + axis[1] * Gy + axis[2] * Gz
    return torch.matrix_exp(angle * Gn)


# =====================================================================
# Block rotation
# =====================================================================

def random_rotation() -> torch.Tensor:
    return torch.tensor(SciRotation.random().as_matrix(), dtype=torch.float64)


def rotate_block(H: torch.Tensor, R: torch.Tensor, l1: int, l2: int) -> torch.Tensor:
    """H' = D^{l1}(R) H D^{l2}(R)^T

    Consistent with the tensor product l1 ⊗ l2:
    vec(H') = (D^{l1} ⊗ D^{l2}) vec(H)
    """
    D1 = wigner_D(l1, R)
    D2 = wigner_D(l2, R)
    return D1 @ H @ D2.T


# =====================================================================
# Rotation-invariant quantities
# =====================================================================

def frobenius_norm(H: torch.Tensor) -> float:
    """||H||_F — rotation invariant: ||D1 H D2^T||_F = ||H||_F."""
    return H.norm().item()


def block_trace(H: torch.Tensor) -> float:
    """Trace of H — invariant only when l1 == l2.
    tr(D H D^T) = tr(D^T D H) = tr(H) since D is orthogonal."""
    return H.trace().item()


def component_rms(H: torch.Tensor) -> float:
    """RMS of all components: sqrt(mean(H^2)) = ||H||_F / sqrt(n)."""
    return H.pow(2).mean().sqrt().item()


# =====================================================================
# Normalization methods
# =====================================================================

def normalize_frobenius(H: torch.Tensor, sigma: float) -> torch.Tensor:
    """Divide by pre-computed Frobenius norm statistic.
    H / σ where σ is a fixed scalar from training set.
    Equivariant: σ is invariant → dividing preserves equivariance.
    """
    return H / sigma


def normalize_trace_shift_and_scale(H: torch.Tensor, mu: float, sigma: float,
                                     l1: int, l2: int) -> torch.Tensor:
    """Shift by trace mean (l1==l2 only), scale by norm.

    For l1==l2: H_norm = (H - μI) / σ
    For l1!=l2: H_norm = H / σ  (no shift — trace is always 0)

    Equivariant because:
    - μI is proportional to identity → D (μI) D^T = μI (invariant under rotation)
    - σ is a pre-computed scalar (invariant)
    """
    if l1 == l2:
        dim = 2 * l1 + 1
        return (H - mu * torch.eye(dim, dtype=torch.float64)) / sigma
    else:
        return H / sigma


def normalize_per_element(H: torch.Tensor, mu: torch.Tensor,
                           sigma: torch.Tensor) -> torch.Tensor:
    """Element-wise normalization — BREAKS equivariance.
    (H - μ_{ij}) / σ_{ij}
    """
    return (H - mu) / sigma


# =====================================================================
# Tests
# =====================================================================

def test_wigner_D_properties(l_max=4, n_rot=5):
    """Verify D matrices are orthogonal and form a representation."""
    print(f"\n[Sanity] Wigner-D properties (l=0..{l_max})")
    print("-" * 60)

    all_pass = True
    for l in range(l_max + 1):
        max_orth_err = 0.0
        max_repr_err = 0.0
        for _ in range(n_rot):
            R = random_rotation()
            D = wigner_D(l, R)
            # Orthogonality
            max_orth_err = max(max_orth_err,
                               (D.T @ D - torch.eye(2*l+1)).abs().max().item())
            # Representation: D(R1) D(R2) = D(R1 R2)
            R2 = random_rotation()
            D2 = wigner_D(l, R2)
            D12 = wigner_D(l, R @ R2)
            max_repr_err = max(max_repr_err, (D @ D2 - D12).abs().max().item())

        passed = max_orth_err < 1e-10 and max_repr_err < 1e-10
        all_pass &= passed
        print(f"  l={l}: orth={max_orth_err:.1e}  repr={max_repr_err:.1e}  "
              f"{'PASS' if passed else 'FAIL'}")
    return all_pass


def test_invariants(n_rot=10):
    """Verify that Frobenius norm and trace are rotation invariants."""
    print("\n[Sanity] Rotation invariants")
    print("-" * 60)

    pairs = [(0, 0, "s-s"), (0, 1, "s-p"), (1, 1, "p-p"),
             (1, 2, "p-d"), (2, 2, "d-d")]

    all_pass = True
    for l1, l2, name in pairs:
        H = torch.randn(2*l1+1, 2*l2+1, dtype=torch.float64)
        norm0 = frobenius_norm(H)
        trace0 = block_trace(H) if l1 == l2 else None

        max_norm_err = 0.0
        max_trace_err = 0.0
        for _ in range(n_rot):
            R = random_rotation()
            H_rot = rotate_block(H, R, l1, l2)
            max_norm_err = max(max_norm_err, abs(frobenius_norm(H_rot) - norm0))
            if l1 == l2:
                max_trace_err = max(max_trace_err,
                                    abs(block_trace(H_rot) - trace0))

        passed = max_norm_err < 1e-10
        if l1 == l2:
            passed &= max_trace_err < 1e-10
        all_pass &= passed

        extra = f"  trace_err={max_trace_err:.1e}" if l1 == l2 else ""
        print(f"  ({l1},{l2}) {name:4s}: norm_err={max_norm_err:.1e}{extra}  "
              f"{'PASS' if passed else 'FAIL'}")

    return all_pass


def test_equivariance(l1, l2, normalize_fn, n_rot=20):
    """Check: normalize(rotate(H)) == rotate(normalize(H))

    Returns max error across all rotations.
    """
    H = torch.randn(2*l1+1, 2*l2+1, dtype=torch.float64)
    max_err = 0.0
    for _ in range(n_rot):
        R = random_rotation()
        # Path A: normalize then rotate
        H_norm = normalize_fn(H)
        H_norm_rot = rotate_block(H_norm, R, l1, l2)
        # Path B: rotate then normalize
        H_rot = rotate_block(H, R, l1, l2)
        H_rot_norm = normalize_fn(H_rot)
        max_err = max(max_err, (H_norm_rot - H_rot_norm).abs().max().item())
    return max_err


# =====================================================================
# Main
# =====================================================================

def main():
    pairs = [(0, 0, "s-s"), (0, 1, "s-p"), (1, 1, "p-p"),
             (1, 2, "p-d"), (2, 2, "d-d")]

    print("=" * 60)
    print("Equivariant Normalization 검증 실험")
    print("=" * 60)

    test_wigner_D_properties()
    test_invariants()

    # ---- Test 1: Frobenius norm scaling ----
    print("\n[Test 1] Frobenius norm scale: H / σ")
    print("  σ = pre-computed scalar (training set statistic)")
    print("-" * 60)
    for l1, l2, name in pairs:
        sigma = 2.5  # fixed scalar from "training set"
        fn = lambda H, s=sigma: normalize_frobenius(H, s)
        err = test_equivariance(l1, l2, fn)
        print(f"  ({l1},{l2}) {name:4s}: {err:.2e}  {'PASS' if err < 1e-10 else 'FAIL'}")

    # ---- Test 2: Trace shift + norm scale ----
    print("\n[Test 2] Trace shift + scale: (H - μI) / σ")
    print("  μ, σ = per-(l1,l2) pre-computed statistics")
    print("  Shift only for l1==l2 (trace is invariant)")
    print("-" * 60)
    for l1, l2, name in pairs:
        mu = 3.7
        sigma = 2.1
        fn = lambda H, m=mu, s=sigma, a=l1, b=l2: \
            normalize_trace_shift_and_scale(H, m, s, a, b)
        err = test_equivariance(l1, l2, fn)
        print(f"  ({l1},{l2}) {name:4s}: {err:.2e}  {'PASS' if err < 1e-10 else 'FAIL'}")

    # ---- Test 3: RMS scaling (per-block) ----
    print("\n[Test 3] Per-block RMS scale: H / rms(H)")
    print("  rms(H) = sqrt(mean(H²)) — rotation invariant")
    print("-" * 60)
    for l1, l2, name in pairs:
        fn = lambda H: H / max(component_rms(H), 1e-15)
        err = test_equivariance(l1, l2, fn)
        print(f"  ({l1},{l2}) {name:4s}: {err:.2e}  {'PASS' if err < 1e-10 else 'FAIL'}")

    # ---- Test 4: Trace shift + DYNAMIC norm scale (instance-wise) ----
    print("\n[Test 4] Instance-wise: (H - μI) / ||H - μI||_F")
    print("  μ = pre-computed, ||·||_F computed per-instance (invariant)")
    print("-" * 60)
    for l1, l2, name in pairs:
        mu = 1.5
        def fn(H, m=mu, a=l1, b=l2):
            if a == b:
                dim = 2 * a + 1
                H_shifted = H - m * torch.eye(dim, dtype=torch.float64)
            else:
                H_shifted = H
            norm = max(frobenius_norm(H_shifted), 1e-15)
            return H_shifted / norm
        err = test_equivariance(l1, l2, fn)
        print(f"  ({l1},{l2}) {name:4s}: {err:.2e}  {'PASS' if err < 1e-10 else 'FAIL'}")

    # ---- Test 5: NEGATIVE CONTROL — element-wise ----
    print("\n[Test 5] Element-wise: (H - μ_ij) / σ_ij  (NEGATIVE CONTROL)")
    print("-" * 60)
    for l1, l2, name in pairs:
        if l1 == 0 and l2 == 0:
            print(f"  ({l1},{l2}) {name:4s}: SKIP  (1×1 scalar, trivially equivariant)")
            continue
        dim1, dim2 = 2*l1+1, 2*l2+1
        mu = torch.randn(dim1, dim2, dtype=torch.float64)
        sigma = torch.rand(dim1, dim2, dtype=torch.float64) + 0.5
        fn = lambda H, m=mu, s=sigma: normalize_per_element(H, m, s)
        err = test_equivariance(l1, l2, fn)
        label = "BROKEN" if err > 0.01 else "UNEXPECTED"
        print(f"  ({l1},{l2}) {name:4s}: {err:.2e}  (equivariance {label})")

    # ---- Test 6: NEGATIVE CONTROL — non-identity shift for l1≠l2 ----
    print("\n[Test 6] Non-identity matrix shift for l1≠l2  (NEGATIVE CONTROL)")
    print("-" * 60)
    for l1, l2, name in pairs:
        if l1 == l2:
            continue
        dim1, dim2 = 2*l1+1, 2*l2+1
        # Subtracting a non-trivial matrix (not proportional to I) breaks equivariance
        bias = torch.randn(dim1, dim2, dtype=torch.float64) * 0.1
        fn = lambda H, b=bias: (H - b)
        err = test_equivariance(l1, l2, fn)
        label = "BROKEN" if err > 0.01 else "UNEXPECTED"
        print(f"  ({l1},{l2}) {name:4s}: {err:.2e}  (equivariance {label})")

    # Summary
    print()
    print("=" * 60)
    print("결론")
    print("=" * 60)
    print()
    print("  Equivariance 보존 조건:")
    print("  ✓ Frobenius norm으로 나누기 (invariant scalar)")
    print("  ✓ RMS로 나누기 (invariant scalar)")
    print("  ✓ Trace mean 빼기 (l1==l2일 때만, μI는 rotation invariant)")
    print("  ✓ Instance-wise norm (||H||_F는 invariant)")
    print()
    print("  Equivariance 깨지는 조건:")
    print("  ✗ Element-wise mean/std (preferred direction 도입)")
    print("  ✗ l1≠l2 블록에 non-identity bias 추가")
    print()
    print("  SLEM 적용 시:")
    print("    • Shift(μ): L=0 성분에만 = trace에 비례 = μI 빼기 → 안전")
    print("    • Scale(σ): per-(Z_i, Z_j, l1, l2, L) invariant scalar → 안전")
    print("    • 어떤 invariant 통계를 쓰든 equivariance 보존")


if __name__ == "__main__":
    main()
