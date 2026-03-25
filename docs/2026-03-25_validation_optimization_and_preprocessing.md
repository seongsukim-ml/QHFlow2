# QHFlow2 Validation Optimization & Preprocessed Dataset

**Date:** 2026-03-25
**Environment:** NVIDIA H200 (143 GiB), CUDA, PyTorch 2.x, QH9Stable-random (130,831 samples)
**Method:** Component-level profiling of validation step, `build_final_matrix` benchmark (4 implementations), preprocessed LMDB pipeline

## 1. Experiment Throughput Comparison

wandb API에서 수집한 현재 실행 중인 실험들의 throughput.

| Run | Ep/h | vs Baseline | Config |
|-----|------|-------------|--------|
| tdn-mid | 1.30 | **1.93x** | TDN expansion, mid size |
| SO2+eSEN12 (prev) | 0.96 | 1.43x | 이전 아키텍처 |
| eSEN12 (prev) | 0.93 | 1.39x | 이전 아키텍처 |
| tdn-lg | 0.92 | 1.37x | TDN large |
| so2exp-bw1 | 0.82 | 1.22x | SO2 bw=1 |
| **baseline** | **0.67** | **1.00x** | bs=32, nw=4, fp32 |
| cpexp | 0.67 | 1.00x | CP expansion |
| bf16 | 0.64 | 0.96x | bf16-mixed |
| energy-mse | 0.18 | 0.27x | energy loss (λ=0.01) |

**발견:** bf16은 효과 없음 (data pipeline이 58% 병목이라 Amdahl's law). energy-mse는 energy loss 계산이 ~4x 느리게 만듦. tdn-mid가 가장 빠름.

## 2. Validation Step Profiling

bs=32, 5 batches 평균. GPU 7 (H200).

| Component | Time (ms) | % Total |
|-----------|----------|---------|
| ODE 3-step sampling | 2,642 | 62.3% |
| Metric (rebuild + eigensolve) | 975 | 23.0% |
| Post-processing | 282 | 6.5% |
| Forward (1회) | 197 | 4.7% |
| ODE 1-step sampling | 88 | 2.1% |
| Corrupt | 22 | 0.5% |
| Criterion (loss) | 2 | 0.0% |
| **Total** | **4,196** | |

Training step (fwd + loss) = 199 ms. **Validation은 training 대비 21.1x 느림.**

### 2.1 Validation 병목 원인

1. **ODE sampling** (62%): `model.sample(num_timesteps=3)` → forward pass 3회 + ODE integration overhead
2. **Metric** (23%): `build_final_matrix` 3회 (757 ms) + `cal_orbital_and_energies` 2회 (232 ms)
3. Validation은 매 epoch 실행 (`check_val_every_n_epoch: 1`)

### 2.2 적용한 최적화

**`error_threshold: 1e-3 → 1e10`** (14개 config 일괄 변경)

- ODE sampling + metric이 validation에서 완전히 건너뛰어짐
- Validation step: 4,196 ms → **~495 ms** (**8.5x 가속**)
- Validation이 training과 비슷한 속도로 동작

변경 파일: `configs/qh9/config_flow_v2_*.yaml` (14개)

## 3. `build_final_matrix` Benchmark

Per-molecule loop에서 14×14 블록을 orbital mask로 선택 후 조립하는 연산. 4가지 구현 비교.

### 3.1 단독 벤치마크 (bs=32, 5 batches)

| Method | Mean (ms) | Speedup |
|--------|----------|---------|
| **original (GPU)** | **160** | **1.00x** |
| prealloc (GPU) | 327 | 0.49x |
| **numpy (CPU)** | **47** | **3.40x** |
| torch (CPU) | 233 | 0.68x |

numpy가 단독으로 3.4x 빠름: 작은 행렬의 반복 `index_select` + `torch.cat`은 GPU kernel launch overhead가 지배적.

### 3.2 End-to-end Metric 경로

| 경로 | metric_total (ms) | 결과 |
|------|-------------------|------|
| Original (GPU) | 975 | **최적** |
| numpy + force CPU metric | 1,615 | 악화 (batch GPU→CPU transfer ~600ms) |

**결론:** `build_final_matrix` 단독은 numpy가 빠르지만, 전체 metric 파이프라인에서는 PyG Data object 전체를 GPU→CPU 전송하는 비용이 이득을 상쇄. **현재 GPU 구현이 end-to-end 최적.**

### 3.3 Metric 내부 분해 (GPU 경로)

| Component | Time (ms) | % |
|-----------|----------|---|
| build_final_matrix (3 calls) | 757 | 76% |
| cal_orbital_and_energies (2 calls) | 232 | 23% |
| HOMO/LUMO + MAE | 7 | 1% |

### 3.4 코드 변경

- `_build_final_matrix_numpy` 구현 추가 (`matrix_transforms.py`)
- `version="numpy"` 로 명시적 호출 가능, 기본값은 `"optimized"` (GPU) 유지
- `metric_cpu=True` 설정 시 numpy 경로 자동 사용

## 4. Preprocessed Dataset Pipeline

### 4.1 동기

`__getitem__`의 99%가 Q tensor 계산 (15,909 ms, std=32,920 ms). `num_workers=8`로 병렬화하면 data pipeline은 숨겨지지만, 전처리하면 근본적으로 제거 가능.

### 4.2 구현

| 파일 | 역할 |
|------|------|
| `preprocess_qh9.py` | raw LMDB → preprocessed LMDB 변환 스크립트 |
| `qh9_preprocessed.py` | `QH9PreprocessedDataset` — deserialize only, 연산 없음 |
| `qh9_utils.py` | `use_preprocessed: true` config 지원 |

**Split LMDB 포맷 지원 추가:** 원본 shard 포맷 (`packed_hamiltonian`) + split 포맷 (`Ham`, `ovlp`, `init_ham`) 모두 처리.

### 4.3 Q tensor 분리

Q tensor는 샘플당 ~2.3 MB (전체의 93%) → 포함 시 전체 ~3.3 TB.

**결정:** Q tensor는 전처리 LMDB에서 제외, training 시 on-the-fly 계산 (기존과 동일).

```bash
# Q tensor 없이 전처리 (기본)
python -m qhflow2.dataset_module.preprocess_qh9 \
  --source-lmdb dataset/QH9Stable/processed/QH9Stable.lmdb

# Q tensor 포함 (선택)
python -m qhflow2.dataset_module.preprocess_qh9 \
  --source-lmdb dataset/QH9Stable/processed/QH9Stable.lmdb \
  --include-q-tensor
```

### 4.4 전처리 성능

| | Q tensor 포함 | Q tensor 제외 |
|---|---|---|
| 속도 | 14 samp/s | 85 samp/s |
| 전체 시간 | ~2.5시간 | ~26분 |
| 샘플당 크기 | ~2.5 MB | ~1.9 MB |
| 전체 크기 | ~3.3 TB | **~257 GB** |

### 4.5 상태

전처리 진행 중 (Q tensor 제외, 512 GB map size). 완료 후 `use_preprocessed: true` config로 활성화.

## 5. 요약: 적용 가능한 최적화

| 최적화 | 효과 | 상태 | 난이도 |
|--------|------|------|--------|
| `error_threshold: 1e10` (ODE skip) | Val **8.5x** 가속 | **적용 완료** | config |
| `num_workers: 8` | Training **2x** 가속 | 미적용 (다음 실험) | config |
| Preprocessed LMDB | `__getitem__` ~8000x 가속 | **전처리 진행 중** | config + data |
| `build_final_matrix` numpy | 단독 3.4x, e2e 효과 없음 | 코드 추가, 기본값 아님 | — |
| `check_val_every_n_epoch: 5` | Val 빈도 5x 감소 | 미적용 | config |

### 실험 중지 기록 (2026-03-25)

baseline을 제외한 training 실험 중지:
- `config_flow_v2_energy_mse_strong` — 중지
- `config_flow_v2_cpexp` — 중지
- `config_flow_v2_perl` — 중지

baseline만 계속 실행 중. Evaluation (QH9Stable XL random/size_ood, QH9Dynamic) 및 전처리는 유지.

### 다음 단계

1. 전처리 완료 후 `use_preprocessed: true` + `num_workers: 8` 실험
2. 기대 효과: epoch 시간 84분 → ~20분 (4x)
3. 이 상태에서 bf16 재평가 (GPU가 bottleneck이므로 의미 있음)
