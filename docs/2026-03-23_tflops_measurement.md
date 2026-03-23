# TFLOPS Measurement in QHFlow2

## 요약

QHFlow2 학습 파이프라인에 TFLOPS (Tera Floating Point Operations Per Second) 측정 기능을 추가했다.
두 가지 경로로 사용 가능:

1. **Dry-run**: `python train_qh9.py dry_run=true` — 파이프라인 검증 + 일회성 TFLOPS 측정
2. **Training callback**: 학습 중 wandb에 `perf/tflops`, `perf/mfu_pct` 자동 로깅

## 측정 방법

### 1. FLOP 수 계산: `torch.utils.flop_counter.FlopCounterMode`

PyTorch 2.1+에서 제공하는 공식 FLOP counter를 사용한다.

```python
from torch.utils.flop_counter import FlopCounterMode

flop_counter = FlopCounterMode(display=False)
with flop_counter:
    with torch.no_grad():
        model(batch, H_init, keep_blocks=True)
flops_fwd = flop_counter.get_total_flops()
```

**동작 원리:**
- PyTorch의 `__torch_dispatch__` 메커니즘을 사용하여 모든 ATen 연산을 intercept
- 각 연산의 입력 shape에서 FLOPs를 계산 (e.g., `mm(A, B)` → `2 * M * N * K`)
- `aten.mm`, `aten.addmm`, `aten.bmm`, `aten.conv2d` 등 주요 연산을 커버
- Custom CUDA kernel이나 `torch_cluster.radius_graph` 같은 비-ATen 연산은 포함되지 않음

**Forward FLOPs는 deterministic**: 같은 모델 + 같은 배치 크기면 항상 동일한 값.
따라서 학습 시작 시 1회만 측정하면 된다.

### 2. Forward+Backward FLOPs 추정

Backward pass의 FLOPs를 직접 측정하기 어려우므로, 표준 근사를 사용한다:

```
FLOPs_total ≈ 3 × FLOPs_forward
```

**근거:**
- Forward: 1× (각 layer의 연산)
- Backward (gradient 계산): ~2× (loss→input gradient + weight gradient)
- 이 근사는 Transformer/GNN 등 대부분의 아키텍처에서 통용됨
- 참고: Kaplan et al. 2020 "Scaling Laws for Neural Language Models", Hoffmann et al. 2022 "Chinchilla"

### 3. GPU 시간 측정: `torch.cuda.Event`

CUDA event를 사용하여 GPU 커널 실행 시간을 정밀 측정한다.

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... training step (forward + backward + optimizer) ...
end.record()
torch.cuda.synchronize()
gpu_ms = start.elapsed_time(end)
```

**왜 `time.perf_counter()`가 아닌 CUDA event를 사용하는가:**
- CPU 타이머는 CUDA 커널 큐잉만 측정 — 실제 GPU 실행 시간과 다를 수 있음
- CUDA event는 GPU 스트림 내 실행 시간을 직접 측정
- `torch.cuda.synchronize()` 후 `elapsed_time()`으로 정확한 벽시계 시간 획득

### 4. TFLOPS 및 MFU 계산

```python
tflops = (flops_fwd * 3 / 1e12) / gpu_seconds
mfu = tflops / peak_tflops * 100  # Model FLOPs Utilization (%)
```

- **TFLOPS**: 초당 수행하는 teraFLOPs
- **MFU**: GPU의 이론 peak 대비 실제 활용률
  - H200 peak: 989.5 TFLOPS (FP16/BF16), 494.7 (TF32), 66.9 (FP64)
  - QHFlow2는 float32 (TF32 모드) 기준으로 MFU 계산

## 구현 위치

### Dry-run (`experiment/train_qh9.py`)

```python
# _measure_tflops() 함수
# 1) FlopCounterMode로 forward FLOPs 측정
# 2) CUDA event로 fwd+bwd+optim 시간 측정 (warmup 2회, 측정 3회 평균)
# 3) TFLOPS, MFU 계산 및 출력
```

### Training callback (`common/training_utils.py`)

```python
class TFLOPSCallback(pl.Callback):
    """
    on_fit_start: forward FLOPs 1회 측정 (deterministic)
    on_train_batch_start/end: CUDA event로 step 시간 측정
    → wandb에 perf/tflops, perf/gpu_ms, perf/mfu_pct 로그
    """
```

기본 활성화, `log_tflops: false`로 비활성화 가능.
`tflops_log_interval: 50` (default)으로 측정 빈도 조절.

### wandb에 로깅되는 메트릭

| Metric | 설명 | 단위 |
|---|---|---|
| `perf/tflops` | 해당 step의 TFLOPS (fwd+bwd) | TFLOPS |
| `perf/gpu_ms` | 해당 step의 GPU 시간 | ms |
| `perf/mfu_pct` | Model FLOPs Utilization | % |
| `perf/gflops_fwd` | Forward FLOPs (epoch 평균) | GFLOPs |

## 이 방법의 표준성

### 표준적인 부분

1. **`torch.utils.flop_counter`**: PyTorch 공식 FLOP counter. PyTorch 팀이 관리하며, `torchtune`, `torchtitan` 등 공식 학습 프레임워크에서 사용.

2. **3× forward ≈ total FLOPs**: ML 커뮤니티의 de-facto 표준.
   - PaLM (Google, 2022): "We use the convention that... total training FLOPs is approximately 3× forward FLOPs"
   - Chinchilla (DeepMind, 2022): 동일한 근사 사용
   - LLaMA (Meta, 2023): MFU 보고 시 동일한 방법

3. **CUDA event 기반 GPU 시간 측정**: NVIDIA 권장 방식. `torch.cuda.Event`는 `cudaEventElapsedTime` API의 wrapper.

4. **MFU (Model FLOPs Utilization)**: Chowdhery et al. 2022 (PaLM 논문)에서 정의한 표준 효율 지표.

### 주의할 점

1. **`FlopCounterMode`는 ATen 연산만 카운트**: E3NN의 `spherical_harmonics`, `torch_cluster.radius_graph`, custom CUDA 커널 등은 포함되지 않음. 따라서 실제 FLOPs보다 **과소 추정**될 수 있음.

2. **fvcore와의 차이**: Meta의 `fvcore.nn.FlopCounterMode`는 더 많은 custom op를 지원하지만, PyTorch 2.1+ 내장 counter가 더 최신이고 유지보수됨. 둘 다 표준적.

3. **MFU vs HFU**: MFU는 model FLOPs만 고려. HFU (Hardware FLOPs Utilization)는 activation recomputation 등을 포함. QHFlow2는 activation recomputation을 사용하지 않으므로 MFU ≈ HFU.

4. **배치 크기 의존**: FLOPs는 배치 크기에 비례. TFLOPS는 배치 크기가 충분히 크면 포화하고, 너무 작으면 GPU utilization이 낮아 TFLOPS가 떨어짐. 따라서 비교 시 배치 크기를 명시해야 함.

5. **GNN 특성**: GNN의 FLOPs는 그래프 크기(노드 수, 엣지 수)에 따라 달라짐. 배치 내 분자 크기가 다르므로 step 간 TFLOPS 변동이 있을 수 있음. Callback이 주기적으로(50 step마다) 측정하는 이유.

## 대안 및 향후 확장

| 방법 | 장점 | 단점 |
|---|---|---|
| `torch.utils.flop_counter` (현재) | PyTorch 공식, 쉬움 | custom op 미포함 |
| `fvcore.nn.FlopCounterMode` | custom op 지원 | 별도 설치 필요 |
| `torch.profiler` + FLOP 추정 | 세부 op별 분석 가능 | 복잡, overhead |
| NVIDIA Nsight | 가장 정확 | 외부 도구, 자동화 어려움 |
| DeepSpeed FLOPs profiler | ZeRO 통합 | DeepSpeed 의존 |

현재 구현은 가장 표준적이고 간단한 방법. custom op FLOPs가 중요해지면 `fvcore`로 교체하거나 수동 보정할 수 있음.

## 사용법

```bash
# Dry-run으로 TFLOPS 측정
python train_qh9.py dry_run=true

# 학습 중 wandb에 자동 로깅 (기본 활성화)
python train_qh9.py mode=train

# TFLOPS 로깅 비활성화
python train_qh9.py log_tflops=false

# 측정 주기 변경 (10 step마다)
python train_qh9.py tflops_log_interval=10
```
