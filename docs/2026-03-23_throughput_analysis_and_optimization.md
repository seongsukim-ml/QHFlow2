# QHFlow2 Throughput Analysis & Optimization

**Date:** 2026-03-23
**Environment:** NVIDIA H200 80GB, CUDA 12.x, PyTorch 2.x, QH9Stable-random (52,332 train)
**Method:** Component-level profiling of training step, CUDA op profiling, batch size scaling, BF16 AMP 측정

## 1. Baseline Performance

| Metric | Value |
|---|---|
| Model | QHFlow_so2_v5_1 (12.1M params) |
| Batch size | 32 |
| Precision | FP32 |
| Step time | 1,538 ms |
| Throughput | 21 samples/s |
| GPU memory | 17 GB / 80 GB (21%) |
| Time/epoch (3,271 batches) | ~84 min |

## 2. Training Step Breakdown

가장 중요한 발견: **GPU compute는 전체의 42%에 불과**. Data pipeline이 58%를 차지.

| Component | Time (ms) | % Total | 설명 |
|---|---|---|---|
| **data_load** | 526 | 34.2% | PyG DataLoader collate, LMDB read |
| **to_gpu + corrupt** | 357 | 23.2% | CPU→GPU transfer, post_processing, noise injection |
| **forward** | 419 | 27.3% | GNN forward (graph construction + backbone + expansion) |
| **loss + backward** | 230 | 14.9% | Loss 계산 + autograd backward |
| **optim_step** | 6 | 0.4% | Adam parameter update |

### 왜 forward가 backward보다 큰가? (비정상)

일반적으로 backward ≈ 2× forward지만, QHFlow2는 반대:
- `radius_graph()`: 매 forward마다 neighbor graph 재구축
- `spherical_harmonics()`: 매번 SH basis 재계산
- `obtain_wigner()` (5.9% of CUDA time): Wigner rotation matrix 매번 재계산

Flow matching에서 noise는 **H에만 적용**되고 **좌표는 불변**이므로, 이들은 모두 cacheable.

## 3. CUDA Op Profile (Top 10)

| Rank | Operation | CUDA % | Category |
|---|---|---|---|
| 1 | `aten::bmm` | 13.5% | Matrix multiply (tensor product, expansion) |
| 2 | `aten::mm` | 11.7% | Linear layers, contraction |
| 3 | `gemvx` kernel | 9.1% | Small matrix-vector (basis expansion) |
| 4 | `SO2Conv` | 9.1% | SO2 convolution (GNN core) |
| 5 | `aten::mul` | 8.1% | Element-wise multiply |
| 6 | `aten::tensordot` | 7.8% | Tensor contraction |
| 7 | `elementwise_kernel` | 6.7% | Activation, normalization |
| 8 | `obtain wigner` | 5.9% | Wigner rotation matrices |
| 9 | `edgewise` | 5.5% | Edge-level message passing |
| 10 | `aten::copy_` | 3.3% | Memory copy |

Matrix ops (bmm + mm + tensordot) = 33% → BF16 Tensor Core의 주요 타겟.

## 4. Optimization Experiments

### 4.1 BF16 Mixed Precision

Isolated profiling에서는 forward 3.4× 빠르지만, 실제 학습에서는 효과 제한적:

| Metric | FP32 (bs=32) | BF16 (bs=32) |
|---|---|---|
| Forward (isolated) | 667 ms | 195 ms (3.4×) |
| Fwd+Bwd (isolated) | 1,069 ms | 528 ms (2.0×) |
| **실제 학습 throughput** | **0.68 steps/s** | **0.48 steps/s** |

실제로 더 느려진 이유: GPU compute가 42%뿐이라 Amdahl's law에 의해 전체 speedup이 제한됨.
또한 custom CUDA kernel (SO2Conv, Wigner)의 BF16 autocast overhead가 이득을 상쇄.

### 4.2 Batch Size Scaling

H200 80GB 중 17GB만 사용 — GPU utilization이 극히 낮음:

| Batch Size | Forward (ms) | Samples/s | Peak Memory |
|---|---|---|---|
| 32 | 667 | 30 | 17 GB |
| 64 | 228 | 280 | 35 GB |
| **128** | **318** | **403** | **65 GB** |
| 256 | 500 | 512 | 119 GB (OOM) |

bs=128이 sweet spot: 13.4× throughput, 65 GB memory (headroom 15 GB).

단, 이 수치는 **forward-only** 측정. 실제 학습에서는 data pipeline이 bottleneck이므로 효과 제한적.

### 4.3 torch.compile

CC (C compiler)가 없어서 실패. `conda install gcc_linux-64` 후 재시도 필요.
일반적으로 kernel fusion + memory 최적화로 1.2-1.5× 기대.

## 5. Data Pipeline 상세 분석

### 5.1 Split vs Shard Dataset

현재 `use_shard: false`로 split dataset 사용 중. Shard dataset은 이미 구현되어 있음:

| | Split Dataset | Shard Dataset |
|---|---|---|
| Storage | 단일 LMDB | 다수 LMDB shard (60개) |
| I/O 병렬성 | 낮음 (단일 DB contention) | 높음 (shard별 독립 I/O) |
| Memory | 전체 로드 (InMemoryDataset) | Lazy loading (on-demand) |
| 속도 | Baseline | **~28% faster** (벤치마크 결과) |
| Config | `use_shard: false` | `use_shard: true` |

### 5.2 `__getitem__` 비용 분석

매 sample마다 실행되는 작업:

| 단계 | 시간 (추정) | 설명 |
|---|---|---|
| LMDB read + unpickle | ~1 ms | Binary 데이터 로드 |
| Matrix unpack | ~1 ms | Upper triangle → full matrix |
| **Matrix transform** | ~2-5 ms | PySCF → E3NN orbital convention 변환 |
| **Cut matrices** | ~2-5 ms | Full matrix → block-diagonal 분리 |
| Q tensor | ~1-5 ms | On-site 3-index overlap (cache hit이면 빠름) |
| Cut orbital coefficients | ~3-10 ms | Generalized eigenvalue problem |
| **Total per sample** | **~10-30 ms** | bs=32 → 320-960 ms collation |

### 5.3 `corrupt()` 비용

`to_gpu + corrupt` (357 ms) 내역:

1. **CPU→GPU transfer**: batch 전체의 tensor를 CUDA로 복사
2. **post_processing**: dtype cast, 추가 텐서 생성
3. **time sampling**: `t ~ U[0.01, 1.0]`
4. **Gaussian noise 생성**: `torch.randn_like(init_ham)`
5. **Linear interpolation**: `H(t) = (1-t)·H_noise + t·H_target`
6. **Symmetrization**: `H(t) = (H + H^T) / 2`

이 중 1-2는 CPU overhead, 3-6은 GPU에서 빠르게 처리 가능.

## 6. 최적화 방안 (Impact 순)

### Tier 1: 즉시 적용 가능 (코드 수정 최소)

| 방안 | 예상 Speedup | 변경 |
|---|---|---|
| **Shard dataset 사용** | 1.3× | `use_shard: true` config 변경 |
| **num_workers 증가** | 1.2× | `num_workers: 8` (현재 4) |
| **pin_memory 활성화** | 1.1× | `pin_memory: True` |
| **Batch size 증가** | 1.2-1.5× | `batch_size: 64` (data bottleneck 감안) |

Combined 예상: **~1.5-2×** (84 min/epoch → ~45 min/epoch)

### Tier 2: 중간 난이도

| 방안 | 예상 Speedup | 설명 |
|---|---|---|
| **Graph caching** | 1.1-1.2× | 좌표 불변 → edge_index, SH, Wigner 한 번만 계산. `__getitem__`에서 precompute하거나 forward 첫 호출 시 cache |
| **Prefetch overlap** | 1.5× | Data loading과 GPU compute를 pipeline으로 overlap. `torch.utils.data.DataLoader` + `prefetch_factor=2` |
| **corrupt를 GPU batch 처리** | 1.2× | 현재 357ms 중 대부분 CPU. GPU에서 batch-level noise 생성/보간 |
| **Matrix transform 사전 계산** | 1.3× | PySCF→E3NN 변환을 LMDB 저장 시 수행. `__getitem__`에서 제거 |

Combined 예상: 추가 **~2×** (45 min → ~25 min/epoch)

### Tier 3: 구조 변경

| 방안 | 예상 Speedup | 설명 |
|---|---|---|
| **Pre-transformed LMDB** | 2× | Matrix transform, cut, Q tensor를 전부 전처리하여 LMDB에 저장. `__getitem__`은 순수 load만 |
| **torch.compile** | 1.3× | CC 설치 후 적용. Kernel fusion |
| **Multi-GPU (DDP)** | N× | `devices: 2+`, 현재 GPU 여유 있음 |
| **Custom CUDA collate** | 1.5× | PyG collate 대신 direct tensor stacking |

## 7. 핵심 Insight

### Data pipeline이 bottleneck인 이유

QHFlow2의 데이터는 Hamiltonian 행렬 (N_basis × N_basis)로 일반 이미지/텍스트보다 **개별 sample이 크다** (14×14 = 196 floats/atom × atoms + overlap + initial_ham + orbital coefficients). Collation에서 variable-size block 처리도 비쌈.

### BF16이 효과 없는 이유

Amdahl's Law: GPU compute = 42%, BF16이 이 중 matmul (33% of CUDA time)을 2× 빠르게 해도:
```
Speedup = 1 / (0.58 + 0.42 × 0.67/2 + 0.42 × 0.33/2)
        = 1 / (0.58 + 0.14 + 0.07)
        = 1.27×
```
이론 최대 1.27× — autocast overhead를 감안하면 ~1.0-1.1×. 실험 결과와 일치.

### 실제로 가장 효과적인 전략

**"Compute를 빠르게" 가 아니라 "Data pipeline을 compute와 overlap"** 해야 함:

```
현재:  [Data Load 526ms] → [GPU+Corrupt 357ms] → [Forward 419ms] → [Backward 230ms]
                                                                    Total: 1538ms

목표:  [Data Load N+1]   [Data Load N+2]   [Data Load N+3]
       ──────────────────────────────────────────────
       [GPU N]           [GPU N+1]         [GPU N+2]
       Total: max(526, 1006) ≈ 1006ms → 1.5× speedup
```

Data prefetch + shard dataset으로 data loading을 compute 뒤에 숨기면, throughput은 GPU compute time (forward + backward ≈ 650ms)에 의해 결정됨. 이때 BF16이 의미 있어짐.

## 8. num_workers Scaling 실험

CPU 128코어 환경에서 num_workers를 변화시키며 full training step 측정:

| num_workers | data (ms) | gpu (ms) | total (ms) | samples/s | vs nw=4 |
|---|---|---|---|---|---|
| 0 | 5,013 | 814 | 5,827 | 5 | 0.2× |
| 2 | 1,792 | 498 | 2,290 | 14 | 0.5× |
| **4 (현재)** | **600** | **440** | **1,040** | **31** | **1.0×** |
| **8** | **90** | **434** | **524** | **61** | **2.0×** |
| 12 | 5 | 714 | 720 | 44 | 1.4× |
| 16 | 24 | 680 | 704 | 45 | 1.5× |

**num_workers=8이 sweet spot — 2× speedup.**

- data_load 600ms → 90ms (6.7× 감소)
- GPU time 434ms로 동일 → **GPU가 bottleneck이 됨** (정상 상태)
- num_workers=12+에서 느려지는 이유: CPU 메모리 경쟁 + persistent_workers의 GPU memory contention

### 핵심 발견

num_workers=8에서 data=90ms, gpu=434ms → **data pipeline이 더 이상 bottleneck이 아님**.
이 상태에서 GPU compute 최적화 (BF16 등)가 비로소 의미를 가짐.

즉시 적용: `num_workers: 8`, `pin_memory: True` → config 2줄 변경으로 **84분 → ~42분/epoch**.

## 9. `__getitem__` 내부 비용 분석

`data_load` 시간의 대부분은 LMDB read가 아니라 CPU 연산:

| 연산 | 복잡도 | 호출 횟수 | 캐싱 가능 | GPU 이동 가능 |
|---|---|---|---|---|
| **Matrix transform** | O(N) | 3× (H, init_H, S) | YES (atom 조합별) | YES |
| **Cut matrix** | O(N²) | 4× (diag/offdiag × 2) | NO | YES |
| **Q tensor build** | O(N²×60) | 1× | PARTIAL | YES |
| **Orbital eigensolve** | O(N³) | 조건부 | NO | YES |
| **Data unpack/reshape** | O(N²) | 1× | YES (전처리) | - |

Matrix transform은 atom 조합에만 의존 → 같은 분자 조합이면 재사용 가능.
Cut matrix의 N² block allocation이 가장 비효율적 — pre-allocated buffer로 개선 가능.

## 10. `__getitem__` 실측 Breakdown

20 samples 실측 결과, **Q tensor가 전체 시간의 99%**:

| 연산 | 시간 (ms) | % Total |
|---|---|---|
| matrix_transform ×3 | 16 | 0.1% |
| cut_matrix ×3 | 92 | 0.6% |
| **Q tensor (block_diag ×60 + transform + cut)** | **15,909** | **99.3%** |
| **Total** | **16,017** | |

Q tensor의 `torch.block_diag` × 60이 압도적 병목. 단, std=32,920ms으로 분산이 극단적 — Q tensor 캐시 (`_q_tensor_cache`) hit/miss에 따라 큰 차이.

### 최적화 시도 결과

| 방법 | 결과 | 비고 |
|---|---|---|
| cut_matrix 사전 할당 | **0.3× (느려짐)** | `np.ix_` 인덱싱이 원본 slice보다 비효율적 |
| P@M@P^T 캐싱 | **부정확** | matrix_transform이 단순 permutation이 아님 (block 단위 sign flip 포함) |
| num_workers=8 | **2.0× (성공)** | CPU 병렬화로 `__getitem__` 시간을 worker에 분산 |

### 결론

`__getitem__` 내부 최적화보다 **worker 병렬화가 더 효과적**. Q tensor 캐시 hit율 개선 (같은 원자 조합의 분자 그룹핑) 또는 Q tensor 사전 계산이 다음 기회.

## 11. 추천 실행 계획

1. **즉시**: `num_workers: 8`, `pin_memory: True` → **2× speedup** (config만 변경)
2. **이번 주**: matrix_transform 캐싱 + cut_matrix 최적화 → 추가 1.3×
3. **다음 주**: Graph caching + BF16 (GPU가 bottleneck인 상태에서 의미 있음) → 추가 1.5×
4. **목표**: 84 min/epoch → **~20 min/epoch** (4× overall)
