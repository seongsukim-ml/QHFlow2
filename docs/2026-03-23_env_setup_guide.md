# QHFlow2 Conda Environment 재현 가이드

**Date:** 2026-03-23
**Environment:** NCloud K8s, 8xH200, Lustre persistent volume
**Verified:** 실제 학습이 돌아가는 `qhflow` env에서 확인

## 1. 핵심 스펙

| Component | Version |
|-----------|---------|
| Python | 3.12.13 |
| PyTorch | 2.4.1+cu121 |
| CUDA (torch) | 12.1 |
| cuDNN | 9.1.0.70 |
| e3nn | 0.5.1 |
| torch-geometric | 2.3.0 |
| fairchem-core | 2.6.0 |
| pyscf | 2.10.0 |
| pytorch-lightning | 2.6.1 |
| psi4 | 1.10 (conda-forge) |

## 2. 설치 순서

`psi4`가 conda-forge에서 설치되어야 하며, 나머지는 pip.
순서가 중요 — torch를 먼저 설치한 뒤 torch-geometric 계열 패키지 설치.

```bash
# Step 1: conda env with psi4 (conda-forge only)
conda create -n qhflow python=3.12 psi4 -c conda-forge -y
conda activate qhflow

# Step 2: PyTorch (cu121)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Step 3: PyG (torch-geometric + extensions, cu121 compatible)
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Step 4: fairchem (eSCN backbone dependency)
pip install fairchem-core==2.6.0

# Step 5: pyscf
pip install pyscf==2.10.0

# Step 6: project requirements
cd /home1/irteam/data-vol1/projects/QHFlow2
pip install -r requirements.txt

# Step 7: additional packages (not in requirements.txt but needed)
pip install pytorch-lightning==2.6.1 wandb rich pandas hydra-core omegaconf
pip install matplotlib scikit-learn numba h5py
```

## 3. 검증

```bash
python -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.version.cuda}')
print(f'GPU available: {torch.cuda.is_available()}, count: {torch.cuda.device_count()}')

import e3nn
print(f'e3nn={e3nn.__version__}')

import torch_geometric
print(f'torch_geometric={torch_geometric.__version__}')

import fairchem
print(f'fairchem loaded')

import pyscf
print(f'pyscf={pyscf.__version__}')

import pytorch_lightning
print(f'pytorch_lightning={pytorch_lightning.__version__}')
"
```

Expected output:
```
torch=2.4.1+cu121, cuda=12.1
GPU available: True, count: 8
e3nn=0.5.1
torch_geometric=2.3.0
fairchem loaded
pyscf=2.10.0
pytorch_lightning=2.6.1
```

## 4. 주의사항

1. **`conda run -n qhflow` 사용 금지**: 이 서버에서 `conda run`은 PATH를 오염시켜 system python (3.13)을 사용할 수 있음. 반드시 `conda activate qhflow` 또는 직접 경로 사용:
   ```bash
   /home1/irteam/data-vol1/conda/envs/qhflow/bin/python
   ```

2. **torch-geometric 버전 호환**: PyG extensions (scatter, sparse, cluster 등)는 torch 버전과 정확히 맞아야 함. `torch-2.4.0+cu121` wheel을 사용 (2.4.1과 호환).

3. **psi4는 conda-forge에서만 설치 가능**: pip 미지원. conda env 생성 시 함께 설치.

4. **fairchem-core**: eSCN backbone(`escn_backbone_v4.py`)에서 사용하는 `fairchem.core.models.escn` 모듈 제공. 없으면 model import 실패.

## 5. Exact Reproduction 파일

정확한 패키지 버전 목록:
- `conda-qhflow-explicit.txt`: `conda create --file` 로 재현 가능
- `pip-qhflow-freeze.txt`: `pip install -r` 로 재현 가능

```bash
# 방법 A: exact conda reproduction
conda create -n qhflow --file conda-qhflow-explicit.txt
conda activate qhflow
pip install -r pip-qhflow-freeze.txt

# 방법 B: clean install (위 Step 1-7 순서)
```
