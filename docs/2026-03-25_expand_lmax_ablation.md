# Tensor Expansion Irreps Order Ablation Analysis

**Date:** 2026-03-25
**Project:** QHFlow2
**Target:** QH9Stable, middle size (hidden=128, bottle=64)

---

## 1. Background

QHFlow2의 readout layer인 `Expansion`은 equivariant node/edge features를 Hamiltonian 행렬 블록으로 변환한다.
이 과정에서 Clebsch-Gordan (CG) coupling을 사용하며, input feature의 최대 angular momentum order (`expand_lmax`)가 출력 행렬의 표현력을 결정한다.

현재 `expand_lmax`는 backbone의 `sh_lmax`(=4)와 동일하게 고정되어 있다.
이 ablation은 **backbone은 sh_lmax=4로 유지하면서 Expansion input의 L만 truncate**하여, readout의 angular resolution이 Hamiltonian 정확도에 미치는 영향을 측정한다.

### 1.1 Expansion의 동작 원리

Expansion은 input irreps `{l_in}` → output matrix `(l₁, l₂)` 블록으로 변환한다:

$$H^{l_1 l_2}_{m_1 m_2} = \sum_{l_{in}} \sum_{m_{in}} C^{l_1 l_2 l_{in}}_{m_1 m_2 m_{in}} \cdot w_{l_{in}} \cdot x^{l_{in}}_{m_{in}}$$

CG coupling rule: `l_in ∈ |l₁ - l₂| ... l₁ + l₂`

**핵심**: l_in=0 (scalar)도 l₁=l₂인 블록에 기여할 수 있다. 하지만 생성하는 패턴이 identity-like으로 제한된다.

### 1.2 Degrees of Freedom (DOF) 분석

`(2l₁+1) × (2l₂+1)` 행렬의 총 자유도 = `Σ_{l=|l₁-l₂|}^{l₁+l₂} (2l+1)`

각 l_in이 기여하는 DOF = `(2l_in + 1)`. 이는 CG decomposition의 완전성 정리에서 온다.

## 2. Feasibility Verification

### 2.1 def2-svp basis (output_irrep = `3x0e + 2x1e + 1x2e`, dim=14)

bottle_hidden_size = 64 (middle size) 기준:

| expand_lmax | input dim | CG paths | fc_weights | bias |
|:-----------:|:---------:|:--------:|:----------:|:----:|
| 0 | 64 | 3 | 896 | 14 |
| 1 | 256 | 9 | 2,240 | 14 |
| 2 | 576 | 15 | 3,200 | 14 |
| 3 | 1,024 | 18 | 3,520 | 14 |
| 4 (baseline) | 1,600 | 19 | 3,584 | 14 |

### 2.2 블록별 DOF 커버리지

**expand_lmax=0:**

| Block (l₁,l₂) | Size | l_in used | DOF covered | Coverage |
|:--------------:|:----:|:---------:|:-----------:|:--------:|
| s-s (0,0) | 1×1 | {0} | 1/1 | 100% |
| s-p (0,1) | 1×3 | — | 0/3 | 0% |
| s-d (0,2) | 1×5 | — | 0/5 | 0% |
| p-p (1,1) | 3×3 | {0} | 1/9 | 11% |
| p-d (1,2) | 3×5 | — | 0/15 | 0% |
| d-d (2,2) | 5×5 | {0} | 1/25 | 4% |

→ 대각 블록만 identity 패턴으로 채움. Off-diagonal 블록 전부 zero.

**expand_lmax=1:**

| Block (l₁,l₂) | l_in used | DOF covered | Coverage |
|:--------------:|:---------:|:-----------:|:--------:|
| s-s (0,0) | {0} | 1/1 | 100% |
| s-p (0,1) | {1} | 3/3 | 100% |
| s-d (0,2) | — | 0/5 | 0% |
| p-p (1,1) | {0,1} | 4/9 | 44% |
| p-d (1,2) | {1} | 3/15 | 20% |
| d-d (2,2) | {0,1} | 4/25 | 16% |

→ s-d 블록 여전히 zero.

**expand_lmax=2:**

| Block (l₁,l₂) | l_in used | DOF covered | Coverage |
|:--------------:|:---------:|:-----------:|:--------:|
| s-s (0,0) | {0} | 1/1 | 100% |
| s-p (0,1) | {1} | 3/3 | 100% |
| s-d (0,2) | {2} | 5/5 | 100% |
| p-p (1,1) | {0,1,2} | 9/9 | **100%** |
| p-d (1,2) | {1,2} | 8/15 | 53% |
| d-d (2,2) | {0,1,2} | 9/25 | 36% |

→ p-p까지 완전. p-d, d-d는 부분적.

**expand_lmax=3:**

| Block (l₁,l₂) | l_in used | DOF covered | Coverage |
|:--------------:|:---------:|:-----------:|:--------:|
| s-s ~ p-p | — | — | 100% |
| p-d (1,2) | {1,2,3} | 15/15 | **100%** |
| d-d (2,2) | {0,1,2,3} | 16/25 | 64% |

→ d-d만 불완전.

**expand_lmax=4 (baseline):**

| Block (l₁,l₂) | l_in used | DOF covered | Coverage |
|:--------------:|:---------:|:-----------:|:--------:|
| all | all | all | **100%** |

→ 모든 블록 완전한 자유도.

### 2.3 검증 결과

- 모든 expand_lmax (0~4)에서 forward pass 정상 동작
- 출력 shape 항상 `[B, 14, 14]` (missing 블록은 zero로 채움)
- Gradient flow 정상 (모든 경우)
- Backbone sh_lmax=4 유지 + Expansion input만 truncate 가능 확인

## 3. 구현 방법

변경 범위가 작음 (모델 파일 1곳 + config):

1. `__init__`에 `expand_lmax` 파라미터 추가 (default = `sh_lmax`)
2. `_create_matrix_prediction_layers`에서 `input_expand_irrep`을 `expand_lmax` 기준으로 생성
3. `output_ii` / `output_ij` Linear의 target irrep을 truncated bottleneck으로 변경

Backbone, SelfNetLayer, PairNetLayer는 변경 없음.

## 4. Ablation Experiment Plan

### 4.1 실험 설계

| Run | expand_lmax | DOF 특성 | 목적 |
|:---:|:-----------:|:--------:|:----:|
| baseline | 4 | 100% 모든 블록 | 기준선 |
| expand_L3 | 3 | d-d 64% | L=4 path 1개의 기여도 |
| expand_L2 | 2 | p-d 53%, d-d 36% | 최소 전블록 활성 |
| expand_L1 | 1 | s-d 0%, p-d 20% | s-p 상호작용만 완전 |

expand_lmax=0은 off-diagonal 블록이 전부 zero이므로 의미 있는 Hamiltonian을 생성할 수 없어 제외.

### 4.2 고정 조건

- **Model**: middle size (hidden=128, bottle=64, sh_lmax=4)
- **Dataset**: QH9Stable, random split
- **Training**: baseline config 그대로 (lr=5e-4, bs=32, 260K steps, warmup=1K)
- **Flow**: sigma=0.1, 3-step ODE, expand init
- **Backbone**: sh_lmax=4, num_gnn_layers=3, num_ham_gnn_layers=2 (변경 없음)

### 4.3 평가 지표

- **H MAE** (Ha, meV) — primary metric
- **HOMO / LUMO / GAP MAE** (meV) — downstream property accuracy
- **블록별 MAE** — s-s, s-p, s-d, p-p, p-d, d-d 각각의 error
- **Training throughput** (samples/s) — Expansion 크기 감소에 따른 속도 변화
- **Parameter count** — Expansion + fc layer의 param 변화

### 4.4 예상 결과

- expand_lmax=4 vs 3: 차이 미미할 가능성 (path 1개, d-d 블록만 영향)
- expand_lmax=2: 핵심 분기점 — p-d/d-d 블록 정확도가 얼마나 떨어지는가
- expand_lmax=1: s-d 블록이 zero → 큰 정확도 하락 예상

블록별 MAE를 측정하면 어떤 블록이 high-L path에 의존하는지 명확히 알 수 있다.

### 4.5 Model Parameters

공통 설정: hidden=128, bottle=64, sh_lmax=4, num_gnn_layers=3, num_ham_gnn_layers=2.

| expand_lmax | Total Params | Expansion input dim | CG paths | fc_weights (per expand) | bias |
|:-----------:|:------------:|:-------------------:|:--------:|:----------------------:|:----:|
| 1 | 42,946,361 (42.9M) | 256 | 9 | 2,240 | 14 |
| 2 | 43,212,345 (43.2M) | 576 | 15 | 3,200 | 14 |
| 3 | 43,311,929 (43.3M) | 1,024 | 18 | 3,520 | 14 |
| **4 (baseline)** | **43,344,953 (43.3M)** | **1,600** | **19** | **3,584** | **14** |

**Parameter 차이 분석:**

expand_lmax에 의해 변하는 component:
- `output_ii` / `output_ij`: Linear(hidden_irrep → expand_bottle_irrep) — target irrep 축소
- `expand_ii` / `expand_ij`: Expansion 내부 path weights — CG path 수에 비례
- `fc_ii` / `fc_ij`: 마지막 layer 출력 dim = Expansion의 `num_path_weight`에 연동

변하지 않는 component (전체의 ~99%):
- Backbone (eSCNMD): message passing, distance expansion
- ParamContraction: input matrix compression
- SelfNetLayer / PairNetLayer: equivariant feature processing
- pre_output_ij TensorProduct

L4 vs L1 param 차이는 약 400K (0.9%) — 매우 작으므로 정확도 차이는 parameter budget이 아닌 **CG 표현력 (DOF coverage)**에서 기인할 것.

### 4.6 Experiment Runs

wandb project: `DFT-25-QH9`, tag: `expand_lmax_ablation`

| Run | expand_lmax | GPU | wandb run_id | Status |
|:---:|:-----------:|:---:|:------------|:------:|
| expandL1 | 1 | GPU 2 | qy40661y | Training |
| expandL2 | 2 | GPU 3 | kaucqj84 | Training |
| expandL3 | 3 | GPU 4 | boj5frn3 | Training |
| expandL4 | 4 | GPU 5 | hhoyay7n | Training |

Config: `config_flow_v2_expandL_ablation.yaml`, 260K steps, 예상 ~16h.
4개 실험 병렬 실행 (2026-03-25 03:27 KST 시작).
