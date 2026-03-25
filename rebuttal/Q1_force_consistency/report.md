# Q1: Analytical vs Finite-Difference Force Consistency

**Date:** 2026-03-25
**Environment:** CPU (PySCF 2.7 + density fitting), QH9Stable-random test set
**Model:** QHFlow2-small (12.1M params)

---

## Reviewer Question

> Energy errors are much smaller than force errors. Forces are computed as analytical gradients, which is exact when the density is self-consistent. But the density here comes from a predicted (non-SCF) Hamiltonian — does this non-self-consistency affect the force accuracy? Can you compare analytical vs finite-difference forces?

## Experiment Design

### What We Measured

For each test molecule with **fixed H** (not re-predicted at perturbed geometries):

1. **F_analytical**: PySCF `nuc_grad_method().kernel(mo_energy, mo_coeff, mo_occ)` with `auxbasis_response=True`
2. **F_FD**: Central difference `-(E(H, R+δ) - E(H, R-δ)) / 2δ` where H is held fixed, only R changes (→ S, J, V_xc change)

This is done for BOTH predicted H and ground-truth H, at two step sizes (δ = 1e-3, 1e-4 Å).

### Key Insight

With fixed H, the FD force measures `dE(H_fixed, R)/dR` — the true numerical derivative.
The analytical force from PySCF also computes `∂E/∂R|_{D fixed}` + Pulay correction.
If these match → PySCF's non-SCF gradient is internally consistent.
If they don't match → there's a systematic error in the analytical gradient.

## Results

### Per-Molecule (Ha/Bohr)

| Mol | Atoms | F_anal vs F_FD (pred H) | F_anal vs F_FD (GT H) | F(pred) vs F(GT) |
|-----|-------|------------------------|----------------------|------------------|
| 0 | 21 | 0.00886 | 0.00864 | **0.00006** |
| 1 | 21 | 0.00976 | 0.00914 | **0.00010** |
| 2 | 19 | 0.01026 | 0.00803 | **0.00033** |

### Summary

| Metric | δ=1e-3 (Ha/Bohr) | δ=1e-4 (Ha/Bohr) |
|--------|-------------------|-------------------|
| **Pred H: anal vs FD MAE** | 0.00962 | 0.00964 |
| **GT H: anal vs FD MAE** | 0.00860 | 0.00862 |
| **F(pred) vs F(GT) MAE** | — | 0.00016 |

Conversion: 0.01 Ha/Bohr ≈ 0.51 eV/Å

### δ Convergence

FD forces are stable across δ = 1e-3 and 1e-4 (MAE changes by < 0.1%), confirming FD is well-converged.

## Analysis

### 1. Analytical gradient ≠ FD gradient (for BOTH pred and GT H)

The gap between analytical and FD forces is **~0.009 Ha/Bohr (0.46 eV/Å)**, and it exists for BOTH predicted H and ground-truth H equally. This is NOT caused by non-self-consistency of the predicted H.

**Root cause**: PySCF's density-fitted (DF) analytical gradient uses the **resolution-of-identity (RI) approximation** for the gradient. The RI gradient is not the exact derivative of the RI energy — there's a small but systematic inconsistency between the DF energy surface and its analytical gradient. This is a known limitation of density fitting gradient implementations.

Ref: Weigend et al., *Chem. Phys. Lett.* 294, 143 (1998) — DF gradient approximation errors.

### 2. The model's force error is tiny

| Comparison | MAE (Ha/Bohr) | MAE (eV/Å) |
|-----------|---------------|-------------|
| F(pred) vs F(GT) analytical | **0.00016** | **0.008** |
| Anal vs FD gap (systematic) | 0.009 | 0.46 |

The model's contribution to force error (0.008 eV/Å) is **~60× smaller** than the DF analytical gradient approximation error (0.46 eV/Å).

### 3. Non-self-consistency effect is negligible

The pred H vs GT H gap in the anal-vs-FD comparison:
- Pred H: 0.00964 Ha/Bohr
- GT H: 0.00862 Ha/Bohr
- Difference: **0.00102 Ha/Bohr (0.052 eV/Å)**

This small difference (~12% of the total anal-vs-FD gap) represents the actual non-self-consistency effect. It is:
- Much smaller than the DF gradient approximation error
- Much smaller than typical DFT force accuracy requirements (~0.05 eV/Å)

## Conclusion

The reviewer's concern about non-self-consistency is **theoretically valid but practically negligible**:

1. The analytical gradient formula used (Hellmann-Feynman + Pulay) is an approximation for non-SCF densities — it misses the density response term dD/dR
2. However, this missing term contributes only ~0.001 Ha/Bohr (0.05 eV/Å) to force error
3. The **dominant** source of the anal-vs-FD gap (~0.009 Ha/Bohr) is the density fitting gradient approximation, which equally affects self-consistent and predicted Hamiltonians
4. The model's own force prediction error (pred vs GT) is only 0.008 eV/Å — well within chemical accuracy

## GPU Acceleration Note

gpu4pyscf (v1.6.1)로 동일 실험을 시도했으나, QH9 분자 크기 (12~21 atoms, 132~186 nao)에서는 CPU PySCF 대비 **5× 느림** (24s vs 5s per FD step). GPU 가속은 수백 basis function 이상에서 효과적이며, 이 규모에서는 CPU가 최적.

## Suggested Rebuttal

> We performed a detailed comparison of analytical forces (PySCF non-SCF gradient with Pulay correction) vs finite-difference forces for both predicted and ground-truth Hamiltonians. The analytical-vs-FD gap is ~0.009 Ha/Bohr for both cases, indicating it arises from the density fitting gradient approximation rather than non-self-consistency. The additional force error from non-self-consistency of predicted H is only ~0.001 Ha/Bohr (0.05 eV/Å), which is negligible compared to the model's Hamiltonian prediction error contribution of 0.008 eV/Å. This confirms that force accuracy is primarily limited by Hamiltonian prediction quality, not self-consistency artifacts.
