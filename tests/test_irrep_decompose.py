"""
IrrepDecomposer 검증: Roundtrip, Norm Preservation, Rotation Equivariance
=========================================================================

Tests:
  1. Roundtrip:        reconstruct(flatten(H)) == H
  2. Norm preservation: ||H||²_F == Σ_L ||h^L||²
  3. Equivariance:     flatten(rotate(H)) == rotate_irreps(flatten(H))
  4. Batch consistency: batch == single-sample loop
  5. Negative control:  non-CG flatten breaks equivariance
  6. All (l1,l2) pairs: l = 0..4  (s, p, d, f, g)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
from common.irrep_decompose import IrrepDecomposer, wigner_D

torch.set_default_dtype(torch.float64)

N_ROT = 20      # rotations per test
L_MAX = 4       # test up to g-orbitals
TOL = 1e-10      # numerical tolerance


def random_rotation():
    return torch.tensor(SciRotation.random().as_matrix(), dtype=torch.float64)


# =====================================================================
# Tests
# =====================================================================

def test_wigner_D(l_max=L_MAX):
    """Wigner-D: orthogonality + representation property."""
    print(f"\n[1] Wigner-D properties (l=0..{l_max})")
    print("-" * 65)
    all_pass = True
    for l in range(l_max + 1):
        max_orth = max_repr = 0.0
        for _ in range(N_ROT):
            R1, R2 = random_rotation(), random_rotation()
            D1, D2 = wigner_D(l, R1), wigner_D(l, R2)
            D12 = wigner_D(l, R1 @ R2)
            max_orth = max(max_orth, (D1.T @ D1 - torch.eye(2*l+1)).abs().max().item())
            max_repr = max(max_repr, (D1 @ D2 - D12).abs().max().item())
        ok = max_orth < TOL and max_repr < TOL
        all_pass &= ok
        print(f"  l={l}: orth={max_orth:.1e}  repr={max_repr:.1e}  "
              f"{'PASS' if ok else 'FAIL'}")
    return all_pass


def test_roundtrip(l_max=L_MAX):
    """reconstruct(flatten(H)) == H for all (l1, l2)."""
    print(f"\n[2] Roundtrip: reconstruct(flatten(H)) == H")
    print("-" * 65)
    all_pass = True
    for l1 in range(l_max + 1):
        for l2 in range(l1, l_max + 1):
            dec = IrrepDecomposer(l1, l2)
            max_err = 0.0
            for _ in range(5):
                H = torch.randn(dec.d1, dec.d2)
                H_rec = dec.reconstruct(dec.flatten(H))
                max_err = max(max_err, (H - H_rec).abs().max().item())
            ok = max_err < TOL
            all_pass &= ok
            print(f"  ({l1},{l2}): {max_err:.1e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


def test_norm_preservation(l_max=L_MAX):
    """||H||²_F == Σ_L ||h^L||² (Parseval's theorem for CG)."""
    print(f"\n[3] Norm preservation: ||H||²_F == Σ_L ||h^L||²")
    print("-" * 65)
    all_pass = True
    for l1 in range(l_max + 1):
        for l2 in range(l1, l_max + 1):
            dec = IrrepDecomposer(l1, l2)
            max_err = 0.0
            for _ in range(5):
                H = torch.randn(dec.d1, dec.d2)
                norm_sq_matrix = H.pow(2).sum().item()
                coeffs = dec.flatten(H)
                norm_sq_irreps = sum(h.pow(2).sum().item() for h in coeffs.values())
                max_err = max(max_err, abs(norm_sq_matrix - norm_sq_irreps))
            ok = max_err < TOL
            all_pass &= ok
            print(f"  ({l1},{l2}): {max_err:.1e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


def test_equivariance(l_max=L_MAX):
    """flatten(rotate(H)) == rotate_irreps(flatten(H))

    This is the core equivariance proof:
    The CG decomposition commutes with rotation.
    """
    print(f"\n[4] Rotation equivariance: flatten(D·H·D^T) == D^L·flatten(H)")
    print("-" * 65)
    all_pass = True
    for l1 in range(l_max + 1):
        for l2 in range(l1, l_max + 1):
            dec = IrrepDecomposer(l1, l2)
            H = torch.randn(dec.d1, dec.d2)
            max_err = 0.0
            for _ in range(N_ROT):
                R = random_rotation()
                # Path A: flatten → rotate in irrep space
                coeffs = dec.flatten(H)
                coeffs_rot_A = dec.rotate_irreps(coeffs, R)
                # Path B: rotate in matrix space → flatten
                H_rot = dec.rotate_matrix(H, R)
                coeffs_rot_B = dec.flatten(H_rot)
                # Compare
                for L in dec.L_list:
                    err = (coeffs_rot_A[L] - coeffs_rot_B[L]).abs().max().item()
                    max_err = max(max_err, err)
            ok = max_err < TOL
            all_pass &= ok
            print(f"  ({l1},{l2}): {max_err:.1e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


def test_irrep_norms_invariant(l_max=L_MAX):
    """||h^L|| is rotation-invariant for each L."""
    print(f"\n[5] Irrep norms are rotation invariants: ||h^L(R·H)|| == ||h^L(H)||")
    print("-" * 65)
    all_pass = True
    for l1 in range(l_max + 1):
        for l2 in range(l1, l_max + 1):
            dec = IrrepDecomposer(l1, l2)
            H = torch.randn(dec.d1, dec.d2)
            norms_orig = dec.irrep_norms(dec.flatten(H))
            max_err = 0.0
            for _ in range(N_ROT):
                R = random_rotation()
                H_rot = dec.rotate_matrix(H, R)
                norms_rot = dec.irrep_norms(dec.flatten(H_rot))
                for L in dec.L_list:
                    err = abs(norms_orig[L].item() - norms_rot[L].item())
                    max_err = max(max_err, err)
            ok = max_err < TOL
            all_pass &= ok
            print(f"  ({l1},{l2}): {max_err:.1e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


def test_batch_consistency(l_max=3):
    """Batch operations match single-sample loop."""
    print(f"\n[6] Batch consistency (l=0..{l_max})")
    print("-" * 65)
    all_pass = True
    B = 8
    for l1 in range(l_max + 1):
        for l2 in range(l1, l_max + 1):
            dec = IrrepDecomposer(l1, l2)
            H_batch = torch.randn(B, dec.d1, dec.d2)

            # Batch flatten
            coeffs_batch = dec.flatten_batch(H_batch)

            # Single-sample loop
            max_err = 0.0
            for i in range(B):
                coeffs_single = dec.flatten(H_batch[i])
                for L in dec.L_list:
                    err = (coeffs_batch[L][i] - coeffs_single[L]).abs().max().item()
                    max_err = max(max_err, err)

            # Batch reconstruct
            H_rec = dec.reconstruct_batch(coeffs_batch)
            rec_err = (H_batch - H_rec).abs().max().item()
            max_err = max(max_err, rec_err)

            # Batch rotate
            R = random_rotation()
            rot_batch = dec.rotate_irreps(coeffs_batch, R)
            for i in range(B):
                rot_single = dec.rotate_irreps(
                    {L: coeffs_batch[L][i] for L in dec.L_list}, R)
                for L in dec.L_list:
                    err = (rot_batch[L][i] - rot_single[L]).abs().max().item()
                    max_err = max(max_err, err)

            ok = max_err < TOL
            all_pass &= ok
            print(f"  ({l1},{l2}): {max_err:.1e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


def test_negative_control(l_max=3):
    """Non-CG flatten (random orthogonal) breaks equivariance."""
    print(f"\n[7] Negative control: random orthogonal basis breaks equivariance")
    print("-" * 65)
    all_pass = True
    for l1 in range(1, l_max + 1):
        for l2 in range(l1, l_max + 1):
            d = (2 * l1 + 1) * (2 * l2 + 1)
            # Random orthogonal matrix (NOT CG)
            Q, _ = torch.linalg.qr(torch.randn(d, d))
            H = torch.randn(2 * l1 + 1, 2 * l2 + 1)

            max_err = 0.0
            for _ in range(5):
                R = random_rotation()
                D1 = wigner_D(l1, R)
                D2 = wigner_D(l2, R)
                # Path A: flatten → "rotate" in random basis (wrong)
                h = Q.T @ H.reshape(-1)
                Dkron = torch.kron(D1, D2)
                D_rand = Q.T @ Dkron @ Q
                h_rot_A = D_rand @ h
                # Path B: rotate → flatten
                H_rot = D1 @ H @ D2.T
                h_rot_B = Q.T @ H_rot.reshape(-1)
                # These should match (Q is fixed), but D_rand won't be block-diagonal
                # Check if D_rand is block-diagonal (it shouldn't be)
                err = (h_rot_A - h_rot_B).abs().max().item()
                max_err = max(max_err, err)

            # Actually the Q-based decomposition IS equivariant (any orthogonal Q works)
            # The point is: D_rand is NOT block-diagonal, so you can't apply
            # per-block normalization without breaking equivariance.
            # Test: per-block scaling with non-CG basis breaks equivariance
            max_err_norm = 0.0
            scales = torch.rand(d) + 0.5  # per-component scales
            for _ in range(N_ROT):
                R = random_rotation()
                D1, D2 = wigner_D(l1, R), wigner_D(l2, R)
                # Path A: scale in random basis → rotate
                h = Q.T @ H.reshape(-1)
                h_scaled = h * scales
                H_scaled = (Q @ h_scaled).reshape(2*l1+1, 2*l2+1)
                H_scaled_rot = D1 @ H_scaled @ D2.T
                # Path B: rotate → scale in random basis
                H_rot = D1 @ H @ D2.T
                h_rot = Q.T @ H_rot.reshape(-1)
                h_rot_scaled = h_rot * scales
                H_rot_scaled = (Q @ h_rot_scaled).reshape(2*l1+1, 2*l2+1)
                max_err_norm = max(max_err_norm,
                                   (H_scaled_rot - H_rot_scaled).abs().max().item())

            ok = max_err_norm > 0.01  # should be broken
            all_pass &= ok
            label = "BROKEN" if ok else "UNEXPECTED PASS"
            print(f"  ({l1},{l2}): {max_err_norm:.2e}  (equivariance {label})")
    return all_pass


def test_cg_block_diagonal(l_max=L_MAX):
    """Verify C^T (D^l1 ⊗ D^l2) C is actually block-diagonal."""
    print(f"\n[8] CG block-diagonality verification")
    print("-" * 65)
    all_pass = True
    for l1 in range(l_max + 1):
        for l2 in range(l1, l_max + 1):
            dec = IrrepDecomposer(l1, l2)
            C = dec.cg_matrix
            max_offdiag = 0.0
            max_orth_block = 0.0
            for _ in range(5):
                R = random_rotation()
                D1, D2 = wigner_D(l1, R), wigner_D(l2, R)
                D_irrep = C.T @ torch.kron(D1, D2) @ C

                # Check off-diagonal blocks are zero
                for L_i, (si, ei) in dec._offsets.items():
                    for L_j, (sj, ej) in dec._offsets.items():
                        if L_i != L_j:
                            val = D_irrep[si:ei, sj:ej].abs().max().item()
                            max_offdiag = max(max_offdiag, val)
                        else:
                            # Diagonal blocks should be orthogonal
                            block = D_irrep[si:ei, sj:ej]
                            dim = ei - si
                            err = (block.T @ block - torch.eye(dim)).abs().max().item()
                            max_orth_block = max(max_orth_block, err)

            ok = max_offdiag < TOL and max_orth_block < TOL
            all_pass &= ok
            print(f"  ({l1},{l2}): off-diag={max_offdiag:.1e}  "
                  f"block_orth={max_orth_block:.1e}  {'PASS' if ok else 'FAIL'}")
    return all_pass


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 65)
    print("IrrepDecomposer 검증")
    print("Matrix Block ↔ Irreducible Representation Decomposition")
    print(f"Testing all (l1, l2) pairs with l = 0..{L_MAX}")
    print(f"  s(l=0), p(l=1), d(l=2), f(l=3), g(l=4)")
    print("=" * 65)

    results = {}
    results['wigner_D'] = test_wigner_D()
    results['roundtrip'] = test_roundtrip()
    results['norm'] = test_norm_preservation()
    results['equivariance'] = test_equivariance()
    results['invariant_norms'] = test_irrep_norms_invariant()
    results['batch'] = test_batch_consistency()
    results['negative'] = test_negative_control()
    results['block_diag'] = test_cg_block_diagonal()

    print()
    print("=" * 65)
    print("Summary")
    print("=" * 65)
    for name, passed in results.items():
        print(f"  {name:20s}: {'ALL PASS' if passed else 'SOME FAIL'}")

    all_ok = all(results.values())
    print()
    if all_ok:
        print("  ✓ All tests passed.")
        print()
        print("  Proven properties:")
        print("  1. CG decomposition is complete (roundtrip)")
        print("  2. Frobenius norm is preserved (Parseval)")
        print("  3. Decomposition commutes with rotation (equivariant)")
        print("  4. Per-irrep norms are rotation invariants")
        print("  5. Non-CG per-component scaling breaks equivariance")
        print()
        print("  → Per-(l1, l2, L) normalization using irrep norms")
        print("    is provably equivariant for all orbital pairs.")
    else:
        print("  ✗ Some tests failed. Check output above.")


if __name__ == "__main__":
    main()
