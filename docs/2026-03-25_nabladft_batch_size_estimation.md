# NablaDFT Batch Size 추정 및 DDP 설정

**Date:** 2026-03-25
**Environment:** 8×H200 (140 GiB each), PyTorch 2.10+cu128, Lightning 2.6.1
**Method:** 실측 VRAM (batch_size=4) + 분자 크기 통계 기반 선형 추정

## 분자 크기 통계 (test_2k_conformers, 500 samples)

| Metric | Orbitals (h_dim) | Matrix elements (h_dim²) |
|--------|-----------------|-------------------------|
| Min | 263 | 69,169 |
| Median | 369 | 136,161 |
| Mean | 373 | ~139,000 |
| Max | 496 | 246,016 |

QH9 대비 약 **27배** 큰 행렬 (QH9 max: 14×14 = 196, NablaDFT max: 496×496 = 246,016).

## 실측 VRAM (Single GPU, batch_size=4)

| 항목 | 값 |
|------|---|
| GPU | H200 (140 GiB) |
| Model | QHFlow_so2_v5_1_nabla_small (13.0M params) |
| batch_size | 4 |
| VRAM 사용 | **~28 GiB** |
| VRAM 여유 | ~112 GiB |

## Batch Size 추정 (Single GPU)

VRAM은 batch 내 가장 큰 분자에 의해 결정됨. 선형 추정 + 안전 마진 20%.

| batch_size | 예상 VRAM (GiB) | H200 140G 가용 여부 | 비고 |
|-----------|----------------|-------------------|------|
| 4 | ~28 | O | 실측 |
| 8 | ~50 | O | |
| 16 | ~95 | O | 권장 (single GPU) |
| 20 | ~115 | △ | 큰 분자 batch에서 OOM 가능 |
| 24 | ~135 | X | OOM 위험 높음 |

> 분자 크기 분산이 크므로 (263-496 orbitals) 최악 batch에서 spike 가능. 안전하게 **batch_size=16** 권장.

## DDP 설정

### 명령어

```bash
# 환경 설정
source /home1/irteam/data-vol1/conda/etc/profile.d/conda.sh
conda activate qhflow
export LD_LIBRARY_PATH=/home1/irteam/data-vol1/conda/envs/qhflow/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home1/irteam/data-vol1/projects/QHFlow2/src:$PYTHONPATH
cd /home1/irteam/data-vol1/projects/QHFlow2

# 2 GPU DDP
CUDA_VISIBLE_DEVICES=0,1 python -m qhflow2.experiment.train_qh9 \
  --config-name config_flow_v2_nabla \
  mode=train \
  devices=2 \
  strategy=ddp \
  dataset.batch_size=8 \
  wandb.mode=online \
  wandb.run_name='nabla-ddp2-bs8'

# 4 GPU DDP
CUDA_VISIBLE_DEVICES=0,1,6,7 python -m qhflow2.experiment.train_qh9 \
  --config-name config_flow_v2_nabla \
  mode=train \
  devices=4 \
  strategy=ddp \
  dataset.batch_size=8 \
  wandb.mode=online \
  wandb.run_name='nabla-ddp4-bs8'
```

### DDP에서의 Effective Batch Size

```
effective_batch_size = batch_size × num_gpus
```

| GPUs | batch_size per GPU | Effective batch | 예상 VRAM/GPU |
|------|-------------------|----------------|--------------|
| 1 | 16 | 16 | ~95 GiB |
| 2 | 8 | 16 | ~50 GiB |
| 4 | 4 | 16 | ~28 GiB |
| 4 | 8 | 32 | ~50 GiB |

### Config 주요 설정

```yaml
# config_flow_v2_nabla.yaml
devices: 1          # override: devices=2 or devices=4
strategy: Null      # override: strategy=ddp

# dataset/NablaDFT.yaml
num_workers: 0      # LMDB + fork 호환 문제로 0 필수
batch_size: 4       # override: dataset.batch_size=8 or 16
```

### DDP 주의사항

1. **`num_workers=0` 필수** — LMDB mmap + DDP fork에서 segfault
2. **`strategy=ddp`** 사용 (`ddp_find_unused_parameters_true` 아님 — QHFlow2는 모든 param 사용)
3. **learning rate scaling** — DDP에서 effective batch가 커지면 lr 비례 조정 고려
   ```bash
   # batch 4배 → lr 2배 (sqrt scaling)
   dataset.batch_size=8 devices=4  # effective=32 vs baseline=4
   # lr: 5e-4 → 1.4e-3 (sqrt(32/4) × 5e-4)
   ```
4. **빈 GPU 확인** — 다른 실험이 GPU를 점유 중일 수 있음. `nvidia-smi`로 확인 후 `CUDA_VISIBLE_DEVICES` 설정

## Overfit Test 실행 예시

```bash
# Single GPU overfit (test_2k, 5000 steps)
CUDA_VISIBLE_DEVICES=0 python -m qhflow2.experiment.train_qh9 \
  --config-name config_flow_v2_nabla \
  mode=train \
  wandb.mode=online \
  wandb.run_name='nabla-overfit-test2k' \
  'wandb.tags=[nabladft,overfit,test2k]' \
  num_training_steps=5000 \
  warmup_step=100 \
  dataset.num_workers=0 \
  dataset.batch_size=4 \
  check_val_every_n_epoch=5 \
  use_last_ckpt=false \
  'ckpt=' \
  seed=42
```
