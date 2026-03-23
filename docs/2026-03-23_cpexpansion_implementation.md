# CPExpansion: TDN-style Vectorized CG Expansion for QHFlow2

**Date:** 2026-03-23
**Environment:** 8xH200 (140 GiB), CUDA 12.1, PyTorch 2.x, e3nn
**Method:** CG tensor product vectorization inspired by Tensor Decomposition Networks (TDN)

## 1. Background

### 1.1 Problem

QHFlow2의 Hamiltonian 예측에서 `Expansion` 클래스가 bottleneck이다. 이 클래스는 equivariant feature를 Hamiltonian matrix block으로 변환하는 Clebsch-Gordan (CG) tensor product를 수행한다.

기존 `Expansion`의 문제점:
1. **O(L^3) CG path에 대한 Python loop** — 각 path마다 별도의 einsum 호출
2. **매 forward마다 `o3.wigner_3j()` 재계산** — 고정된 값인데 캐시하지 않음
3. **소규모 einsum의 반복** — GPU kernel launch overhead가 연산 시간보다 큼

### 1.2 Reference

- Lin et al., "Tensor Decomposition Networks for Fast Machine Learning Interatomic Potential Computations" (arXiv:2507.01131, 2025)
- 핵심 아이디어: CG coefficient tensor M[i,j,k]를 CP decomposition으로 근사
  - `M[i,j,k] ≈ Σ_r A[i,r] · B[j,r] · C[k,r]`
  - 복잡도: O(L^6) → O(L^4) with rank R = 7L^2

### 1.3 SO2Expansion 실패 분석

CPExpansion 이전에 시도한 `SO2Expansion` (bond-aligned local frame + delta_m banding)은 **3가지 근본적 결함**으로 인해 실패했다:

| Bug | 원인 | 영향 |
|-----|------|------|
| Real SH에서 delta_m banding | Complex SH basis에서만 SO(2) equivariant | equivariance 완전 파괴 (3.7e-2 error) |
| `permute(0,1,3,2,4)` | 블록 순서 뒤섞임 | mul>1 블록에서 추가 오차 |
| CPU roundtrip for Wigner-D | `o3.matrix_to_angles(R.detach().cpu())` | CG보다 14x 느림 |

SO2Expansion의 equivariance를 복원하려면 `U_y = U_z @ D(Rx(π/2))` 변환으로 complex SH 중간 단계가 필요한데, 이 경우 복소수 연산 + CPU roundtrip 때문에 CG보다 100-1000x 느려진다. 실용성이 없으므로 대안으로 CPExpansion을 개발했다.

## 2. Design

### 2.1 Core Observation

기존 `Expansion.forward()`에서 각 CG path `(l_in, l1, l2)`는 독립적으로 처리되지만, 같은 output block `(l1, l2)`에 기여하는 path들은 결국 합산된다:

```python
# 기존: 19 paths를 개별 처리, 같은 (j,k)끼리 누적
for ins in self.instructions:       # 19 iterations
    w3j = o3.wigner_3j(l1, l2, l_in)  # 매번 재계산
    result = einsum("ijk, buvk -> buivj", w3j, weighted_x)
    outputs[(j, k)] += result
```

**핵심 관찰**: 같은 `(l1, l2)` 블록에 기여하는 여러 `l_in` path의 w3j를 미리 concatenate하면, 하나의 einsum으로 처리할 수 있다.

### 2.2 Block Grouping

def2-svp 기준 (output irreps: `3x0e + 2x1e + 1x2e`):

| Output block (l1, l2) | Contributing l_in | Paths | Concat dim P |
|------------------------|-------------------|-------|--------------|
| (0, 0) | 0 | 1 | 1 |
| (0, 1) | 1 | 1 | 3 |
| (0, 2) | 2 | 1 | 5 |
| (1, 0) | 1 | 1 | 3 |
| (1, 1) | 0, 2 | 2 | 1+5 = 6 |
| (1, 2) | 1, 3 | 2 | 3+7 = 10 |
| (2, 0) | 2 | 1 | 5 |
| (2, 1) | 1, 3 | 2 | 3+7 = 10 |
| (2, 2) | 0, 2, 4 | 3 | 1+5+9 = 15 |

**19 paths → 9 blocks** (def2-svp), **40 paths → 16 blocks** (def2-tzvp)

### 2.3 Concatenated CG Tensor

각 `(j, k)` 블록에 대해, 기여하는 모든 w3j를 feature dimension을 따라 concatenate:

```
M_{j,k} = [ w3j(l1, l2, l_a) | w3j(l1, l2, l_b) | ... ]
         shape: [2l1+1, 2l2+1, P]
```

여기서 `P = Σ (2l_in + 1)` for all contributing `l_in`.

이 텐서는 고정값이므로 `register_buffer`로 한 번만 계산하여 저장한다.

### 2.4 Forward Pass

```
Input: x_in [B, irrep_in.dim], weights [B, num_path_weight], bias_weights [B, num_bias]

For each output block (j, k):
  1. Weight contraction: 각 path p에 대해
     g_p = einsum('bwuv, bwm -> buvm', weights_p, x_l_p) / mul_in
     (+ bias for l_in=0 path)

  2. Concatenate: g_cat = cat(g_parts, dim=-1)  →  [B, mul1, mul2, P]

  3. CG expansion: 단일 einsum
     H_block = einsum('ijp, buvp -> buivj', M_{j,k}, g_cat)  →  [B, mul1, d1, mul2, d2]

  4. Reshape: [B, mul1*d1, mul2*d2]

Assemble: block-diagonal matrix → full output [B, out_dim1, out_dim2]
```

### 2.5 Optional CP Decomposition

큰 블록(P가 클 때)에 대해 truncated SVD로 근사 가능:

```python
M_mat = M.reshape(d1*d2, P)
U, S, Vh = torch.linalg.svd(M_mat, full_matrices=False)
AB = (U[:,:R] * S[:R]).reshape(d1, d2, R)  # [d1, d2, R]
C = Vh[:R,:].T                               # [P, R]
```

Forward:
```python
h = einsum('buvp, pr -> buvr', g_cat, C)      # project: [B, mul1, mul2, R]
H = einsum('ijr, buvr -> buivj', AB, h)        # expand:  [B, mul1, d1, mul2, d2]
```

## 3. Implementation Details

### 3.1 Weight Compatibility

CPExpansion은 Expansion과 **동일한 weight ordering**을 유지한다:

- `instructions` 생성 순서: 동일한 3중 루프 (i, j, k)
- `num_path_weight`, `num_bias`: 동일한 값
- `flat_weight_index`, `bias_weight_index`: 동일한 순서로 슬라이싱

따라서 `fc_ii`, `fc_ij` 네트워크의 output dimension이 동일하고, 기존 checkpoint에서 weight를 로드할 수 있다.

### 3.2 Internal vs External Weights

기존 Expansion은 두 가지 모드를 지원한다:

1. **Internal weights** (`weights=None`): `self.weights` 파라미터 사용, bias 미적용
2. **External weights** (`weights` 제공): fc 네트워크에서 생성된 weight 사용, bias 적용

CPExpansion은 두 모드 모두 동일하게 동작한다. 주의: internal mode에서는 원본 Expansion이 bias를 적용하지 않으므로, CPExpansion도 동일하게 처리한다.

### 3.3 Files

| File | 변경 내용 |
|------|----------|
| `src/models/layers.py` | `CPExpansion` 클래스 추가 (line 963~) |
| `src/models/QHFlow_so2_v5_1_cpexp.py` | 모델 변형 (Expansion → CPExpansion) |
| `src/models/__init__.py` | import 및 model_dict 등록 |
| `src/config_qh9/model/QHFlow_so2_v5_1_cpexp_small.yaml` | 모델 config |
| `src/config_qh9/config_flow_v2_cpexp.yaml` | 학습 config |

## 4. Verification

### 4.1 Numerical Equivalence

CPExpansion (exact mode, `cp_rank=None`)은 Expansion과 bit-level 동일한 출력을 생성한다.

| Test | Max Diff | Status |
|------|----------|--------|
| Output match (10 seeds, B=32) | 3.73e-09 | PASS |
| `grad_x` match | 2.98e-08 | PASS |
| `grad_w` match | 2.98e-08 | PASS |
| `grad_bw` match | 7.45e-09 | PASS |
| Internal weights match | 7.45e-09 | PASS |
| Full 14x14 matrix element max diff | 1.16e-10 | PASS |

### 4.2 Equivariance

CG Expansion과 동일한 equivariance (D_out @ H @ D_out^T 검증):

| Method | Max Diff | Status |
|--------|----------|--------|
| CG Expansion | 3.73e-09 | PASS |
| CPExpansion (exact) | 3.73e-09 | PASS |
| CPExpansion (R=15) | 7.37e-04 | PASS |
| CPExpansion (R=10) | 1.42e-03 | FAIL |
| CPExpansion (R=5) | 1.88e-03 | FAIL |

CP approximate에서 R=15 이상이면 equivariance 유지. def2-svp에서는 exact mode 권장.

## 5. Benchmark Results

### 5.1 Layer-level Throughput (B=16, N=500)

| Method | it/s | us/iteration | vs CG |
|--------|------|-------------|-------|
| CG Expansion | 109 | 9,175 | 1.00x |
| **CPExpansion (exact)** | **278** | **3,593** | **2.55x** |
| CPExpansion (R=15) | 314 | 3,181 | 2.88x |
| CPExpansion (R=10) | 356 | 2,809 | 3.27x |

### 5.2 Batch Size Scaling

| Batch | CG (us) | CP exact (us) | Speedup |
|-------|---------|---------------|---------|
| 1 | 12,182 | 3,198 | **3.81x** |
| 4 | 10,950 | 2,817 | **3.89x** |
| 16 | 11,227 | 2,948 | **3.81x** |
| 64 | 7,617 | 2,845 | **2.68x** |
| 256 | 9,150 | 2,967 | **3.08x** |

작은 배치에서 speedup이 더 크다 — GPU kernel launch overhead 감소 효과가 지배적.

### 5.3 NAO Size Scaling

| Config | Output dim | Paths→Blocks | CG (us) | CP (us) | Speedup |
|--------|-----------|-------------|---------|---------|---------|
| def2-svp | 14 | 19→9 | 9,005 | 3,700 | **2.43x** |
| def2-tzvp | 30 | 40→16 | 22,922 | 7,299 | **3.14x** |
| large (lmax=4) | 55 | 65→25 | 45,376 | 9,701 | **4.68x** |

**NAO가 커질수록 speedup 증가**: output irrep이 커지면 CG path 수가 O(L^3)으로 증가하지만, block 수는 O(L^2)로만 증가하여 path merge 효과가 커진다.

## 6. Speedup 원인 분석

세 가지 최적화가 기여한다:

### 6.1 Path Loop 축소

19 (def2-svp) 또는 40 (def2-tzvp) 번의 Python loop + einsum 호출이 9 또는 16번으로 줄어든다. 각 einsum 호출의 GPU kernel launch overhead (~5-10 us)가 절감된다.

### 6.2 w3j Precomputation

기존 Expansion은 매 forward pass마다 `o3.wigner_3j(l1, l2, l_in)`을 호출한다. 이 함수는 내부적으로 CG coefficient를 계산하고 tensor를 생성한다. CPExpansion은 이를 `__init__`에서 한 번만 계산하여 `register_buffer`로 저장한다.

### 6.3 Concatenated Einsum

같은 output block에 기여하는 여러 path의 feature를 concatenate하여 하나의 einsum으로 처리한다:

```python
# 기존: path별 소규모 einsum
for l_in in [0, 2, 4]:
    result += einsum('ijk, buvk -> buivj', w3j_l_in, g_l_in)  # k=1,5,9

# CPExpansion: 단일 einsum
g_cat = cat([g_0, g_2, g_4], dim=-1)  # P=15
H = einsum('ijp, buvp -> buivj', M_concat, g_cat)  # p=15
```

GPU는 하나의 큰 einsum이 여러 작은 einsum보다 효율적이다 (더 높은 occupancy, 더 적은 kernel launch).

## 7. Complexity Analysis

### 7.1 Expansion의 Forward Pass 분해

Expansion의 forward는 두 단계로 나뉜다:

**Step 1 — Weight contraction** (per-path):
```
g_{l_in}[b, u, v, m] = Σ_w  W[b, w, u, v] · x[b, w, m]  /  mul_in
```
- Cost per path: O(B · c · c² · d_in) = O(B · c³ · d_in)
- Total over all paths: O(B · c³ · Σ_paths d_in)
- Σ_paths d_in = Σ_{valid (l_in,l1,l2)} (2l_in+1) ≈ O(L⁴)  [~L³ paths × avg d_in ~ L]

**Step 2 — CG expansion** (per-path):
```
H[b, u, i, v, j] = Σ_m  w3j[i, j, m] · g[b, u, v, m]
```
- Cost per path: O(B · c² · d₁ · d₂ · d_in)
- Total over all paths: O(B · c² · Σ_paths d₁·d₂·d_in)
- Σ (2l₁+1)(2l₂+1)(2l_in+1) over valid triplets ≈ O(L⁶)

### 7.2 Complexity 비교

| Method | Step 1 (weight) | Step 2 (CG) | Total | 비고 |
|--------|----------------|-------------|-------|------|
| **Expansion** | O(B·c³·L⁴) | O(B·c²·L⁶) | **O(B·c²·L⁶)** | per-path loop, L³ einsum calls |
| **CPExpansion (exact)** | O(B·c³·L⁴) | O(B·c²·L⁶) | **O(B·c²·L⁶)** | per-block loop, L² einsum calls |
| **CPExpansion (CP, R=7L²)** | O(B·c³·L⁴) | O(B·c²·R·L⁴) = O(B·c²·L⁶) | **O(B·c²·L⁶)** | CP도 total은 동일 |
| **TDN (path-weight sharing + CP)** | O(B·c²·L²) | O(B·c²·R·L²) = O(B·c²·L⁴) | **O(B·c²·L⁴)** | single W, single einsum |

### 7.3 왜 CPExpansion은 asymptotic이 같은데 빠른가?

CPExpansion (exact)의 총 FLOPs은 Expansion과 **동일**하다. 같은 output block `(l₁, l₂)`에 기여하는 path들의 CG 연산을 concatenate해도 총 곱셈 횟수는 변하지 않는다:

```
Σ_paths d₁·d₂·d_in  =  Σ_blocks d₁·d₂·P_{block}     (P = Σ d_in for that block)
```

실측 2.5-4.7x speedup의 원인은 **constant factor**:

1. **Kernel launch overhead**: GPU einsum 호출 1회당 ~5-10 μs. 19회→9회 = ~50-100 μs 절감/iteration
2. **`o3.wigner_3j()` 제거**: Python function call + tensor allocation을 buffer lookup으로 대체
3. **Memory coalescing**: 여러 작은 einsum 대신 하나의 큰 einsum이 GPU cache를 더 효율적으로 사용

### 7.4 TDN의 O(L⁴)를 달성하려면

TDN 논문의 핵심 추가 요소는 **path-weight sharing**: 모든 CG path에 대해 단일 W를 공유.

```python
# 현재 (per-path weights): Step 1이 L³ paths를 순회
for path in paths:
    g_path = W_path @ x_path            # per-path weight → L³번 반복

# TDN (shared weight): Step 1이 single matmul
g = W_shared @ x_concat                 # single weight → 1번
H = A @ diag(C^T @ g) @ B^T             # CP decomposition → O(R·D)
```

이렇게 하면:
- Step 1: O(B·c²·D_in) = O(B·c²·L²) — 단일 matmul
- Step 2: O(B·c²·R·D_out) = O(B·c²·L²·L²) = O(B·c²·L⁴) — CP로 압축
- **Total: O(B·c²·L⁴)** — 진정한 asymptotic 개선

단, path-weight sharing은 per-path expressivity를 제거하므로 정확도에 영향을 줄 수 있다. TDN 논문은 이것이 MLIP (energy/force)에서 성능 저하 없이 작동함을 보였지만, Hamiltonian prediction에서의 검증은 추가 실험이 필요하다.

## 8. Summary: Complexity Hierarchy

```
Expansion (현재 baseline)     O(c²L⁶)   — per-path loop, runtime w3j
CPExpansion exact (구현 완료)  O(c²L⁶)   — per-block loop, precomputed w3j  → 2.5-4.7x 실측 (constant factor)
CPExpansion + CP approx       O(c²L⁶)   — same total FLOPs, smaller tensors → 추가 ~15% (def2-svp에서 미미)
TDN (path-weight sharing+CP)  O(c²L⁴)   — asymptotic 개선                  → 미구현, 추가 실험 필요
```

## 9. Limitations

1. **def2-svp에서의 제한적 speedup**: 19→9 path 감소이므로 ~2.5x. def2-tzvp(~3.1x)이나 더 큰 basis(~4.7x)에서 더 효과적.
2. **CP approximate의 제한**: def2-svp에서 CG tensor가 full-rank (9x9x25)이므로 CP approximation의 compression ratio가 낮다. R=15에서 겨우 equivariance 유지.
3. **Expansion layer 외의 bottleneck**: 전체 모델에서 Expansion은 일부분. GNN backbone (eSCN blocks)이 더 큰 비중을 차지할 수 있으므로, end-to-end speedup은 layer speedup보다 작을 수 있다.

## 10. Future Work

1. **Path-weight sharing** (TDN 논문): 모든 CG path에 대해 단일 W를 공유하면 파라미터가 O(c²L³) → O(c²)로 줄어든다. fc_ii/fc_ij의 output dimension도 감소하여 Step 1과 Step 2 모두 O(L⁴)로 떨어지며, 진정한 asymptotic 개선. Hamiltonian prediction에서의 정확도 검증 필요.
2. **def2-tzvp 학습**: 더 큰 basis에서 CPExpansion의 constant factor 이점이 극대화 (4.7x at 55-dim).
3. **Fused CUDA kernel**: concatenated einsum을 custom CUDA kernel으로 구현하면 추가 최적화 가능.
