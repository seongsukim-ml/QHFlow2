# QHFlow2 Infrastructure Overhaul — Data Pipeline, Benchmarking, Package Structure

**Date:** 2026-03-23
**Environment:** 8×H200, 2 TiB RAM, Ubuntu 22.04 (K8s), conda (qhflow env)
**Scope:** Data loading 7,300× speedup, TFLOPS 측정, dft-dataset 통합, Python 패키지 정비

## 1. Data Pipeline 개선

### 1.1 문제

`__getitem__`이 training step의 58%를 차지. Q tensor 계산이 99% (15,909ms/sample).

| Component | Time/step | % |
|---|---|---|
| data_load | 526ms | 34.2% |
| to_gpu+corrupt | 357ms | 23.2% |
| forward | 419ms | 27.3% |
| loss+backward | 230ms | 14.9% |
| optim_step | 6ms | 0.4% |
| **Total** | **1,538ms** | |

### 1.2 해결: Preprocessed Single LMDB

모든 on-the-fly 연산을 1회 전처리하여 단일 LMDB에 저장.

**전처리 파이프라인:**
```
raw LMDB (shard) → unpack → matrix_transform (pyscf→e3nn) → cut_matrix → Q tensor → single LMDB
```

**결과:**

| 지표 | 기존 (shard) | Preprocessed |
|---|---|---|
| `__getitem__` | 16,000ms | **2.19ms** |
| 개선 배수 | — | **7,300×** |
| LMDB 형식 | 60 shards | 단일 파일 |
| 저장 크기 | ~85 GB | ~130 GB |

### 1.3 OrbitalMatrix 클래스 (dft-dataset에 추가)

Convention-aware 대칭 행렬 + per-atom block 분해를 통합 API로 제공.

```python
from dft_dataset.orbital_matrix import OrbitalMatrix

H = OrbitalMatrix.from_dense(matrix, atoms, "def2-svp", "pyscf")
H_e3nn = H.in_convention("e3nn")       # convention 변환
blocks = H_e3nn.to_blocks(pad_to=14)   # block 분해
H_recon = blocks.to_dense()            # 복원
```

**검증:** QHFlow2의 `_cut_matrix_3d`와 수치 완벽 일치 (Water, CH4, roundtrip 오차 0.00e+00).

`align="shell"` 모드: H 원자의 5개 orbital을 indices [0,1,3,4,5]에 배치 (QHFlow2 호환).

### 1.4 QH9PreprocessedDataset

| 모드 | 로드 필드 | 용도 |
|---|---|---|
| `train` | Preprocessed blocks only | 학습 (빠름) |
| `eval` | Blocks + raw packed matrices | 평가 (full matrix 복원 가능) |

```python
ds = QH9PreprocessedDataset("QH9Stable_preprocessed.lmdb", split="random", mode="train")
```

## 2. Dry-run 모드

```bash
python train_qh9.py dry_run=true
```

파이프라인 전체를 3 step만 실행하고 throughput + TFLOPS를 측정.

출력:
```
DRY RUN RESULTS
  Dataset load:   X.XXs
  Model init:     X.XXs
  3 train steps:  X.XXs
  Throughput:     XX.X samples/s
  TFLOPS (total): X.XX
  MFU (approx):   X.X%
```

## 3. TFLOPS 측정

### 3.1 방법

| 단계 | 도구 | 설명 |
|---|---|---|
| FLOP 수 | `torch.utils.flop_counter.FlopCounterMode` | PyTorch 공식, ATen op 카운트 |
| GPU 시간 | `torch.cuda.Event` | 커널 실행 시간 정밀 측정 |
| fwd+bwd 추정 | `3 × FLOPs_forward` | PaLM/Chinchilla 표준 근사 |
| MFU | `TFLOPS / peak` | H200 TF32 peak = 494.7 TFLOPS |

### 3.2 E3NN FLOP 커버리지 (실측)

| 연산 | FlopCounterMode | 비고 |
|---|---|---|
| `FullyConnectedTensorProduct` | **COUNTED** | 내부 einsum/mm |
| `nn.Linear`, `bmm`, `addmm` | **COUNTED** | ATen op |
| `spherical_harmonics` | NOT COUNTED | polynomial, FLOP 미미 |
| `scatter_mean/sum` | NOT COUNTED | memory-bound |
| `radius_graph` | NOT COUNTED | custom CUDA |

**Per-module breakdown (small model, batch=4):**

| 모듈 | GFLOPs | 비율 |
|---|---|---|
| `node_attr_backbone` (ESCN) | 31.5 | 79.3% |
| `ham_gnn_backbone` | 7.3 | 18.3% |
| `pre_output_ij` | 0.83 | 2.1% |
| **Total counted** | **39.8** | |

GEMM 연산이 99%, 미카운트 연산은 GPU 시간의 ~5% 미만 → **실질적으로 정확**.

### 3.3 wandb 로깅

`TFLOPSCallback` (PyTorch Lightning callback):
- `on_fit_start`: FLOPs 1회 측정 (deterministic)
- `on_train_batch_end`: 50 step마다 GPU 시간 → TFLOPS 계산
- wandb metrics: `perf/tflops`, `perf/gpu_ms`, `perf/mfu_pct`

## 4. Prediction Benchmarking (dft-dataset 통합)

### 4.1 기존 문제

QHFlow2는 `.pt` 파일로 예측을 저장. 비표준 포맷, 다른 모델과 비교 불가.

### 4.2 해결

`_predict_step`에서 dft-dataset의 `PredictionResult` 포맷으로 동시 저장.

**test epoch 종료 시 자동 생성되는 파일:**

| 파일 | 내용 |
|---|---|
| `predictions.npz` | 전체 H 행렬 (packed), geometry, metadata |
| `comparison_report.txt` | GT 대비 H MAE/RMSE/max |
| `property_metrics.npz` | HOMO/LUMO/gap MAE (meV) |
| `pred_flat/*.pt` | Legacy 호환 |

**Property 추출:** `dft_dataset.pipeline.properties_from_hamiltonian(H, S, n_electrons)` → orbital_energies, HOMO, LUMO, gap_eV, density_matrix.

## 5. Python 패키지 정비

### 5.1 문제

모든 파일에 `sys.path.insert(0, "/절대/경로/...")` 해킹. 30+ 파일, 60+ 인스턴스.

### 5.2 해결

| 프로젝트 | 변경 | Import |
|---|---|---|
| dft-dataset | `src/` → `dft_dataset/` rename, pyproject.toml hatchling | `from dft_dataset.molecule import Molecule` |
| QHFlow2 | `pip install -e .` (이미 pyproject.toml 있었음) | `from qhflow2.common.setup import ...` |
| 크로스 참조 | sys.path 해킹 제거 | `from dft_dataset.xxx import ...` |

**dft-dataset 내부:** 12개 bare import → relative import (`.molecule`, `.conventions`, ...)
**Scripts/tests:** `from molecule import` → `from dft_dataset.molecule import`

### 5.3 검증

```
20/20 tests passed (test_pipeline_integration.py)
```

sys.path.insert 없이 모든 import 정상 동작.

## 6. labutils 공용 패키지

`projects/utils/` — 프로젝트 간 공유 유틸리티.

```python
from labutils.plotting import setup_fonts, style_ax, bold_legend, save, THEMES, SIZES
```

스타일 가이드 기반: DIN 2014 폰트, closed box, bottom+left tick, `quantity / unit` 표기.

## 7. 생성/수정된 파일 목록

### 신규 생성

| 파일 | 위치 |
|---|---|
| `dft_dataset/orbital_matrix.py` | dft-dataset |
| `qhflow2/dataset_module/preprocess_qh9.py` | QHFlow2 |
| `qhflow2/dataset_module/qh9_preprocessed.py` | QHFlow2 |
| `tests/test_pipeline_integration.py` | QHFlow2 |
| `projects/utils/` (전체) | utils |

### 주요 수정

| 파일 | 변경 |
|---|---|
| `dft-dataset/pyproject.toml` | setuptools → hatchling, `dft_dataset` 패키지명 |
| `dft-dataset/dft_dataset/*.py` (12개) | bare import → relative import |
| `QHFlow2/pyproject.toml` | editable install 확인 |
| `qhflow2/common/training_utils.py` | `TFLOPSCallback` 추가 |
| `qhflow2/experiment/train_qh9.py` | `dry_run=true` 모드 + `_measure_tflops()` |
| `qhflow2/pl_module/flow_module.py` | `_predict_step` → PredictionResult NPZ 저장 |
| `qhflow2/common/qh9_utils.py` | `use_preprocessed` 분기 |
| `config_qh9/dataset/QH9Stable.yaml` | `use_preprocessed`, `num_workers: 8`, `pin_memory: true` |
