# QHFlow2

<!-- # High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction (QHFlow) -->

By Seongsu Kim, Feb, 2026 [[PDF]](https://arxiv.org/pdf/2602.16897v1)

This repository contains an implementation of QHFlow2 for DFT Hamiltonian prediction.

## Installation

### Quick Install (uv)

```bash
# Core only (inference)
uv pip install -e .

# With training dependencies
uv pip install -e ".[training,data]"

# Full install (all optional dependencies)
uv pip install -e ".[all]"
```

### Dependency Groups

| Extra | Packages | When you need it |
|-------|----------|------------------|
| *(core)* | torch, e3nn, torch-geometric, torch-scatter, torch-cluster | Always (inference & model loading) |
| `training` | pytorch-lightning, hydra-core, wandb, torch-ema, tensorboard | Training models |
| `data` | lmdb, apsw | Building/loading preprocessed datasets |
| `dft` | pyscf | DFT calculations, SCF integration |
| `md` | ase | Molecular dynamics simulations |
| `fairchem` | fairchem-core | eSCN-based model architectures |
| `all` | All of the above | Full development |

### From Scratch (conda + uv)

```bash
conda create -n qhflow2 python=3.12 psi4 -y
conda activate qhflow2

# PyTorch with CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Install qhflow2 with all extras
uv pip install -e ".[all]"
```

For GPU-accelerated DFT (optional):
```bash
conda install cuda=12.4.0 -c nvidia
pip install gpu4pyscf-cuda12x
```

## Project Structure

```
QHFlow2/
├── pyproject.toml              # Package definition & dependency groups
├── src/qhflow2/                # Python package
│   ├── models/                 # QHFlow / QHNet model architectures
│   ├── pl_module/              # PyTorch Lightning training modules
│   ├── common/                 # Shared utilities, metrics, transforms
│   ├── dataset_module/         # Dataset loaders and LMDB sharding
│   ├── dft_process/            # DFT measurement and post-processing
│   ├── md/                     # MD simulation and SCF/MLFF utilities
│   ├── experiment/             # Training entry points (train_qh9, train_md17)
│   └── utils.py                # Core utilities (AOData, Q tensor, etc.)
├── configs/                    # Hydra YAML configs
│   ├── qh9/                    # QH9 dataset configs
│   └── md17/                   # MD17/rMD17 dataset configs
├── scripts/                    # Benchmark, profiling, experiment scripts
├── dataset/                    # Data root (auto or manual download)
├── ckpts/                      # Pretrained checkpoints
└── tests/
```

## Dataset

Datasets are provided via Google Drive. Download and place each dataset folder under `dataset/`.

**MD17:** [md17_processed](https://drive.google.com/drive/folders/1d3HTu0H7gdg54kirWBqN24x-s1QW6OKV?usp=sharing) — Download each folder under `dataset/` (e.g. `dataset/ethanol/`).

**rMD17:** [new_md17 (rMD17)](https://drive.google.com/drive/folders/1yqxWmy2NM84Q_yjBnvQaZLO6vAAc5ghA) — Download each folder under `dataset/` (e.g. `dataset/rmd-aspirin/`).

**QH9:**
```bash
mkdir -p ./dataset/QH9Stable/raw/
gdown https://drive.google.com/uc?id=1LcEJGhB8VUGkuyb0oQ_9ANJdSkky9xMS -O ./dataset/QH9Stable/raw/QH9Stable.db

mkdir -p ./dataset/QH9Dynamic_300k/raw/
gdown https://drive.google.com/uc?id=1sbf-sFhh3ZmhXgTcN2ke_la39MaG0Yho -O ./dataset/QH9Dynamic_300k/raw/QH9Dynamic_300k.db
```

Processing from raw files to torch datasets runs automatically on the first training run.

## Saved Checkpoints

Pre-trained checkpoints: [Google Drive](https://drive.google.com/drive/folders/14My7LuvDJiM9hvG2n4_5T9UN97wp4x9w?usp=sharing). Download and place under `ckpts/`.

## Usage

### Train

```bash
# MD17
python -m qhflow2.experiment.train_md17 dataset=ethanol

# QH9
python -m qhflow2.experiment.train_qh9 dataset=QH9Stable dataset.split=random

# With W&B logging
python -m qhflow2.experiment.train_qh9 dataset=QH9Stable dataset.split=random wandb.mode=online

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m qhflow2.experiment.train_qh9 dataset=QH9Stable strategy=ddp devices=4
```

### Available Datasets & Splits

| Dataset | Names | Splits |
|---------|-------|--------|
| MD17 | `ethanol`, `malondialdehyde`, `uracil`, `water` | — |
| rMD17 | `rmd-aspirin`, `rmd-naphthalene`, `rmd-salicylic_acid` | — |
| QH9Stable | `QH9Stable` | `random`, `size_ood` |
| QH9Dynamic | `QH9Dynamic` | `geometry`, `mol` |

### Available Models

| Model | Config | Requires |
|-------|--------|----------|
| QHFlow (base) | `model=QHFlow` | core |
| QHFlow-SO2 v5.1 | `model=QHFlow_so2_v5_1_small` | `[fairchem]` |
| QHFlow-CPExp | `model=QHFlow_so2_v5_1_cpexp_small` | `[fairchem]` |
| QHFlow-SO2Exp | `model=QHFlow_so2_v5_1_so2exp_small` | `[fairchem]` |
| QHFlow-TDN | `model=QHFlow_so2_v5_1_tdn_small` | `[fairchem]` |

### Predict

Save Hamiltonian predictions for test samples:

```bash
# MD17
python -m qhflow2.experiment.train_md17 mode=predict dataset=water ckpt=ckpts/md17/water/checkpoints/weights.ckpt

# QH9
python -m qhflow2.experiment.train_qh9 mode=predict dataset=QH9Stable dataset.split=random ckpt=ckpts/QH9Stable/random/checkpoints/weights.ckpt
```

### Python API

```python
from qhflow2.models import get_model, get_default_model_args

# Load model
args = get_default_model_args("qh9")
args["version"] = "QHFlow_so2_v5_1"
# model = get_model(args)
```

### Tips

- Training automatically resumes when interrupted
- Use `CUDA_VISIBLE_DEVICES` to select GPUs
- Configs are in `configs/qh9/` and `configs/md17/` — override any value via CLI

## Citation
```
@inproceedings{kim2025high,
  title={High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction},
  author={Kim, Seongsu and Kim, Nayoung and Kim, Dongwoo and Ahn, Sungsoo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgements
This project is based on [AIRS](https://github.com/divelab/AIRS.git) (QHNet).

**MD17 Dataset**: [Revised MD17 dataset (rMD17)](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)

**QH9 Dataset**: [QHBench/QH9](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9)
