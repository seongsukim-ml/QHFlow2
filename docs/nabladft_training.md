# NablaDFT Training Guide

QHFlow2로 NablaDFT Hamiltonian을 학습하는 방법.

## Prerequisites

### 환경 설정

```bash
source /home1/irteam/data-vol1/conda/etc/profile.d/conda.sh
conda activate qhflow
export LD_LIBRARY_PATH=/home1/irteam/data-vol1/conda/envs/qhflow/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home1/irteam/data-vol1/projects/QHFlow2/src:$PYTHONPATH
cd /home1/irteam/data-vol1/projects/QHFlow2
```

> `LD_LIBRARY_PATH` 필수 — 없으면 `CXXABI_1.3.15 not found` 에러

### 데이터 전처리

학습 전 3단계 전처리가 필요:

```
SQLite (Psi4 raw) → dft-dataset LMDB → QHFlow2 LMDB
```

```bash
# Step 1: SQLite → dft-dataset LMDB
cd /home1/irteam/data-vol1/projects/dft-dataset
python scripts/convert_nabladft.py --input test_2k_conformers.db

# Step 2: dft-dataset LMDB → QHFlow2 LMDB
cd /home1/irteam/data-vol1/projects/QHFlow2
python scripts/preprocess_nabladft.py --input test_2k_conformers --workers 16
```

전처리 결과: `/home1/irteam/data-vol1/datasets/nablaDFT/qhflow2_processed/<split>.lmdb`

### 전처리 완료 상태

| Split | QHFlow2 LMDB | 상태 |
|-------|-------------|------|
| test_2k_conformers | 8.2 GB | 사용 가능 |
| train_100k | — | 전처리 필요 |
| train_full | — | DB 다운로드 중 |

## Training

### 기본 학습 (overfit test)

```bash
CUDA_VISIBLE_DEVICES=0 python -m qhflow2.experiment.train_qh9 \
  --config-name config_flow_v2_nabla \
  mode=train \
  wandb.mode=online \
  wandb.run_name='nabla-overfit-test2k' \
  num_training_steps=5000 \
  warmup_step=100 \
  seed=42
```

### DDP 멀티 GPU 학습

`ddp_find_unused_parameters_true` strategy 필수 (conditioning 모듈의 unused params).

```bash
# 2 GPU
CUDA_VISIBLE_DEVICES=6,7 python -m qhflow2.experiment.train_qh9 \
  --config-name config_flow_v2_nabla \
  mode=train \
  devices=2 \
  strategy=ddp_find_unused_parameters_true \
  dataset.batch_size=4 \
  wandb.mode=online \
  wandb.run_name='nabla-ddp2-bs4'

# 4 GPU
CUDA_VISIBLE_DEVICES=0,1,6,7 python -m qhflow2.experiment.train_qh9 \
  --config-name config_flow_v2_nabla \
  mode=train \
  devices=4 \
  strategy=ddp_find_unused_parameters_true \
  dataset.batch_size=4 \
  wandb.mode=online \
  wandb.run_name='nabla-ddp4-bs4'
```

DDP 주의사항:

- **`strategy=ddp_find_unused_parameters_true`** 사용 (`ddp`만 쓰면 unused parameter 에러)
- **effective batch size** = `batch_size × num_gpus` (예: bs=4, 4GPU → effective 16)
- **lr scaling** — effective batch가 커지면 lr 비례 조정 고려 (sqrt rule: `lr × sqrt(effective/baseline)`)
- **`num_workers=0`** DDP에서도 필수 (LMDB mmap + fork 비호환)
- Lightning이 `DistributedSampler`를 자동 주입하므로 수동 설정 불필요

### VRAM 추정 (per GPU)

실측 기준 (H200 140 GiB):

| batch_size | 예상 VRAM | 1 GPU | 2 GPU (eff) | 4 GPU (eff) |
|-----------|----------|-------|------------|------------|
| 4 | ~28 GiB | bs=4 | bs=8 | bs=16 |
| 8 | ~50 GiB | bs=8 | bs=16 | bs=32 |
| 16 | ~95 GiB | bs=16 | bs=32 | bs=64 |

### Full training (train_100k)

```bash
CUDA_VISIBLE_DEVICES=0,1,6,7 python -m qhflow2.experiment.train_qh9 \
  --config-name config_flow_v2_nabla \
  mode=train \
  wandb.mode=online \
  dataset.source_lmdb=/home1/irteam/data-vol1/datasets/nablaDFT/qhflow2_processed/train_100k.lmdb \
  devices=4 \
  strategy=ddp_find_unused_parameters_true \
  dataset.batch_size=8 \
  num_training_steps=100000
```

### wandb 비활성화

```bash
python -m qhflow2.experiment.train_qh9 --config-name config_flow_v2_nabla \
  wandb.mode=disabled
```

## Config 구조

```
configs/qh9/
├── config_flow_v2_nabla.yaml          # NablaDFT 학습 config (entry)
├── model/
│   └── QHFlow_so2_v5_1_nabla_small.yaml  # 모델 config
└── dataset/
    └── NablaDFT.yaml                  # 데이터셋 config
```

### 주요 설정

| 항목 | 값 | 비고 |
|------|---|------|
| Model | QHFlow_so2_v5_1 | 13.0M params |
| Basis | def2-svp-nabla | max 32 orbitals (Br) |
| Batch size | 4 | 분자 크기가 큼 (244-496 orbitals) |
| num_workers | 0 | LMDB + fork segfault 방지 |
| Optimizer | AdamW | lr=5e-4, weight_decay=0.01 |
| Scheduler | Polynomial decay | warmup 1000 steps |
| Loss | Hamiltonian MAE + MSE | weight 10.0 |
| Flow | ODE 3 steps | init_p0_type=expand, sigma=0.1 |
| Init Hamiltonian | hcore (T+V) | residual prediction |

## QH9 대비 차이점

| | QH9 | NablaDFT |
|---|-----|---------|
| DFT code | PySCF | Psi4 v1.5 |
| Functional | B3LYP/PBE | ωB97X-D |
| Basis | def2-SVP (14 orb) | def2-SVP (32 orb, +S/Cl/Br) |
| Elements | H,C,N,O,F | H,C,N,O,F,S,Cl,Br |
| Molecules | ~130k | ~2M (12.7M conformers) |
| SCF conv | 1e-9 | 1e-6 |
| Convention transform | PySCF → e3nn | Psi4 → PySCF → e3nn |

## Troubleshooting

### `CXXABI_1.3.15 not found`
```bash
export LD_LIBRARY_PATH=/home1/irteam/data-vol1/conda/envs/qhflow/lib:$LD_LIBRARY_PATH
```

### DataLoader worker segfault
`dataset.num_workers=0` 사용 (config에 이미 설정됨).
원인: LMDB `mmap` + `fork` 비호환.

### `ckpt_path="best"` ValueError
학습 후 test 단계에서 best checkpoint가 없을 때 발생.
validation이 최소 1번 돌아야 checkpoint 저장됨.
`check_val_every_n_epoch`를 적절히 설정.

### pickle UnpicklingError (weights_only)
이전 run의 checkpoint와 호환 문제. `use_last_ckpt=false 'ckpt='`로 새로 시작.
