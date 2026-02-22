# QHFlow2

<!-- # High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction (QHFlow) -->

By Seongsu Kim, Feb, 2026 [[PDF]]([https://arxiv.org/abs/2505.18817](https://arxiv.org/pdf/2602.16897v1))

🌟 This repository contains an implementation of the QHFlow2 for DFT Hamiltonian prediction. This repository is still updating.

## Packages and Requirements

All codes are tested and confirmed to work with `python 3.12` and `CUDA 12.1`. A similar environment should also work, as this project does not rely on some rapidly changing packages.

```bash
# Example CUDA 12.1 with torch 2.4.1 without gpu-pyscf
conda create -n qhflow2 python=3.12 psi4 -y
conda activate qhflow2

pip install fairchem-core==2.6.0
pip install pyscf==2.10.0
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install -r ../requirements.txt
```

```bash
# Example CUDA 12.1 with torch 2.4.1 with gpu-pyscf
conda create -n qhflow2-gpu-pyscf python=3.12 psi4 -y
conda activate qhflow2-gpu-pyscf
conda install cuda=12.4.0 -c nvidia

pip install nvidia-cuda-runtime-cu12==12.6.37
pip install gpu4pyscf-cuda12x
pip uninstall cupy
conda install cupy=13.4.0

pip install fairchem-core==2.6.0
pip install pyscf==2.10.0
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install -r ../requirements.txt
```

**Note:** Run `pip install -r ../requirements.txt` from the `src/` directory (or `pip install -r requirements.txt` from the repository root).

## Directory and Files
The project follows this directory structure:
```
.
├── src/                       # Source code (run Python from here)
│   ├── experiment/            # Entrypoints: train_md17, train_qh9, train_*_finetune
│   ├── config_md17/           # MD17/rMD17 configs (config_flow-lw10, config_flow-lw10-wa)
│   ├── config_qh9/            # QH9 configs (config_flow_v2_simple, config_flow-cont-ft-wa)
│   ├── common/                # Shared setup, data utils, training utils
│   ├── dataset_module/       # Dataset loaders and LMDB sharding
│   ├── models/                # QHFlow / QHNet
│   ├── pl_module/             # PyTorch Lightning modules
│   ├── dft_process/           # DFT measurement and post-processing scripts
│   ├── md/                    # MD simulation and SCF/MLFF utilities
│   └── utils.py
├── _experiment_scripts/       # Example training/prediction shell scripts
├── _mlff_scripts/             # MLFF/DFT measurement scripts
├── _data_scripts/             # Data processing and DB utilities
├── dataset/                   # Data root (auto or manual download)
├── requirements.txt
├── ckpts/                     # Pretrained/finetuned checkpoints (or download from Drive)
└── README.md
```

## Dataset
Datasets are provided via Google Drive. Download and place each dataset folder under `dataset/`.

**MD17:** Pre-processed MD17 data (ethanol, malondialdehyde, uracil, etc.) is available on Google Drive: [md17_processed](https://drive.google.com/drive/folders/1d3HTu0H7gdg54kirWBqN24x-s1QW6OKV?usp=sharing). Download each folder and place it under `dataset/` (e.g. `dataset/ethanol/`, `dataset/malondialdehyde/`, `dataset/uracil/`).

**rMD17 (Revised MD17):** Pre-processed rMD17 data (rmd-aspirin, rmd-ethanol, rmd-naphthalene, rmd-salicylic_acid) is available on Google Drive: [new_md17 (rMD17)](https://drive.google.com/drive/folders/1yqxWmy2NM84Q_yjBnvQaZLO6vAAc5ghA). Download each folder and place it under `dataset/` (e.g. `dataset/rmd-aspirin/`, `dataset/rmd-ethanol/`, etc.).

**QH9:** Use the commands below:

```bash
mkdir -p ./dataset/QH9Stable/raw/
gdown https://drive.google.com/uc?id=1LcEJGhB8VUGkuyb0oQ_9ANJdSkky9xMS -O ./dataset/QH9Stable/raw/QH9Stable.db

mkdir -p ./dataset/QH9Dynamic_300k/raw/
gdown https://drive.google.com/uc?id=1sbf-sFhh3ZmhXgTcN2ke_la39MaG0Yho -O ./dataset/QH9Dynamic_300k/raw/QH9Dynamic_300k.db
```

Processing from raw files to torch datasets runs automatically on the first training run.
Or, you can process manually with the sharding process:
```bash
python -m dataset_module.qh9_datasets_split \
    --name=${NAME}  \
    --num_chunks=30 --chunk_idx=${DB_IDX} \
    --split=${SPLIT}
```
where NAME is the dataset name (`QH9Stable` / `QH9Dynamic`). Use the following SPLIT options:
- `QH9Stable`: `random`, `size_ood`
- `QH9Dynamic`: `geometry`, `mol`

Data is assembled automatically when the final chunk is processed.

**Note**: The legacy `qh9_datasets_split.py` module will be deprecated. Use `qh9_datasets_shard.py` for all new dataset processing operations.

**Note:** We plan to provide pre-processed datasets for all datasets to facilitate easier setup and usage.

## Saved Checkpoints

Pre-trained checkpoints are available on **Google Drive**: [QHFlow checkpoints](https://drive.google.com/drive/folders/14My7LuvDJiM9hvG2n4_5T9UN97wp4x9w?usp=sharing). Download and place them under `ckpts/` (or set `ckpt` to the path where you saved them).

Dataset and split names used in checkpoint paths:
- **MD17**: `ethanol`, `malondialdehyde`, `uracil`, `water`
- **rMD17**: `rmd-aspirin`, `rmd-naphthalene`, `rmd-salicylic_acid`
- **QH9**: `QH9Stable`, `QH9Dynamic` with splits `random`, `size_ood` (QH9Stable) or `geometry`, `mol` (QH9Dynamic)

When running from `src/`, use the checkpoint path in the `ckpt` argument (e.g. `ckpt=../ckpts/...`).

## Usage

### Prerequisites
All commands should be run from the `src` directory.

### Available Datasets:
- **MD17 DATASET**: `ethanol`, `malondialdehyde`, `uracil`, `water`
- **RMD17 DATASET**: `rmd-aspirin`, `rmd-naphthalene`, `rmd-salicylic_acid`
- **QH9 DATASET**: `QH9Stable`, `QH9Dynamic`
  - **QH9Stable SPLIT (dataset.split)**: `random`, `size_ood`
  - **QH9Dynamic SPLIT (dataset.split)**: `geometry`, `mol`

### Tips

**Training Tips:**
- You can enable Weights & Biases logging with `wandb.mode=online`
- Training automatically resumes when interrupted
- Use `CUDA_VISIBLE_DEVICES` to specify GPU devices: `CUDA_VISIBLE_DEVICES=0,1 python -m experiment.train_md17 dataset=water`

**Performance Tips:**
- For faster training, you can use multiple GPUs. For example, `CUDA_VISIBLE_DEVICES=0,1,2,3` with `strategy=ddp devices=4`
- Monitor GPU memory usage and adjust batch size if needed

**Debugging Tips:**
- Check logs in the `logs/` directory for detailed training information
- Monitor validation metrics to ensure proper training progress

### Train

Detailed train/predict scripts are in `_experiment_scripts/`.

```bash
python -m experiment.train_md17 dataset=${DATASET}
python -m experiment.train_qh9  dataset=${DATASET} dataset.split=${SPLIT}
```

**Examples:**

```bash
# Train MD17 model
python -m experiment.train_md17 dataset=ethanol wandb.mode=online

# Train QH9 model
python -m experiment.train_qh9 dataset=QH9Stable dataset.split=random wandb.mode=online
```

### Prediction (Saving the Output)

This mode is used to predict test files and save individual Hamiltonian matrices for each sample. The predictions are saved to disk for further analysis.

**Output Format:**
- Hamiltonian matrices are saved as individual files
- Each prediction corresponds to a test sample
- Files are organized by dataset and model configuration

```bash
python -m experiment.train_md17 mode=predict dataset=${DATASET} ckpt=${CKPT}
python -m experiment.train_qh9 mode=predict dataset=${DATASET} dataset.split=${SPLIT} ckpt=${CKPT}
```

**Examples:**
```bash
# MD17 prediction
python -m experiment.train_md17 mode=predict dataset=water ckpt=../ckpts/md17/water/checkpoints/weights.ckpt

# QH9 prediction
python -m experiment.train_qh9 mode=predict dataset=QH9Stable dataset.split=random ckpt=../ckpts/QH9Stable/random/checkpoints/weights.ckpt
```

**Output Location:**
- Predictions are typically saved in the `outputs/` directory
- Each run creates timestamped subdirectories for organization

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
This project is based on the repo [AIRS](https://github.com/divelab/AIRS.git) (QHNet).

**MD17 Dataset**: [Revised MD17 dataset (rMD17)](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)

**QH9 Dataset**: [QHBench/QH9](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9)
