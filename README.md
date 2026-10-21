# QHFlow2

<!-- # High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction (QHFlow) -->

By Seongsu Kim, Feb, 2026 [[PDF]](https://arxiv.org/pdf/2602.16897v1)

🌟 This repository contains an implementation of the QHFlow2 for DFT Hamiltonian prediction. This repository is still updating.

[Latest published release](https://github.com/seongsukim-ml/QHFlow2/releases/tag/v0.2.0-rc.1) · [Updated benchmarks](#qh9stable-benchmark-update) · [Stable/Muon training guide](docs/QH9STABLE_MUON.md) · [Version history](#version-history)

## Packages and Requirements

For the corrected SO(3)/O(3) models with Muon, use
[`requirements-stable-muon.txt`](requirements-stable-muon.txt) and the
[tested runtime instructions](docs/QH9STABLE_MUON.md#stable-data-and-runtime).
The installation examples below are retained for the **legacy** models;
their Torch 2.4 runtime does not provide native Muon.

```bash
# Example CUDA 12.1 with torch 2.4.1 without gpu-pyscf
conda create -n qhflow2 python=3.12 -y
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
conda create -n qhflow2-gpu-pyscf python=3.12 -y
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

Q tensor generation uses PySCF (`int3c1e`) with the original 16s/8p/4d auxiliary
basis defined in code. Psi4 and a working-directory `auxiliary.gbs` file are
not required. Existing Q tensor shapes and orbital ordering are preserved.

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

Set `DB_IDX=-1` to process all shards, or use one shard index for a resumable
partial run. Each shard is written directly under `processed/lmdbs/`.
The command above remains as a compatibility CLI, but new Python code should
import from `dataset_module.qh9_dataset`. The following legacy import paths are
deprecated, emit `DeprecationWarning`, and retain their previous exports during
the compatibility period:

- `dataset_module.qh9_common`
- `dataset_module.qh9_datasets_split`
- `dataset_module.qh9_datasets_shard`
- `dataset_module.qh9_datasets_shard_full_process`

**Note:** We plan to provide pre-processed datasets for all datasets to facilitate easier setup and usage.

## Saved Checkpoints

Pre-trained checkpoints are available on **Google Drive**: [QHFlow checkpoints](https://drive.google.com/drive/folders/14My7LuvDJiM9hvG2n4_5T9UN97wp4x9w?usp=sharing). Download and place them under `ckpts/` (or set `ckpt` to the path where you saved them).

Dataset and split names used in checkpoint paths:
- **MD17**: `ethanol`, `malondialdehyde`, `uracil`, `water`
- **rMD17**: `rmd-aspirin`, `rmd-naphthalene`, `rmd-salicylic_acid`
- **QH9**: `QH9Stable`, `QH9Dynamic` with splits `random`, `size_ood` (QH9Stable) or `geometry`, `mol` (QH9Dynamic)

When running from `src/`, use the checkpoint path in the `ckpt` argument (e.g. `ckpt=../ckpts/...`).

## QH9Stable benchmark update

Updated **2026-10-21** · Updated version **[v0.2.0-rc.2](docs/releases/v0.2.0-rc.2-checkpoints.md)** (draft release).

![QH9Stable original-paper and updated-version results](docs/assets/qh9-stable-updated.png)

- **Results and units.** Original-paper metrics, including QHNetV2, reproduce the supplied paper tables. QHFlow2 parameter counts follow the [checkpoint/config audit](docs/MODEL_CONFIGS.md); updated counts are trainable parameters (XL: initialized model). Updated results use the best validation checkpoint, seed 42, all **13,084** official QH9Stable random-split test molecules and **three ODE steps**. H, occupied orbital energies, HOMO, LUMO and GAP are in μEh; Sc is in percent. Original E/F values retain their source units; updated E/F and other unavailable values are shown as dashes.
- **Model update.** Order fix + **gate change (spectral)** + Muon at **1e-3**; norm, embedding and other unsupported parameters use AdamW at **1e-4**. Updated S/M also enable the Hamiltonian-input path. **+ O(3) group averaging** denotes a separately trained QHFlow2-O(3)-s, using shared-weight identity/inversion averaging. These results combine architecture and training changes and are not a single-change ablation.
- **Training cost and speed.** Measurements use H100 80GB GPUs and effective global batch 32. Train GPU hours approximate the retained checkpoint lineage's observed training time × GPU count, including validation/checkpointing within each segment, excluding startup, restart/queue gaps, discarded attempts/steps and final test evaluation. Samples / s is global molecule throughput: total samples / total elapsed time over consecutive 25-step training log windows within each epoch of the final run, including checkpoint I/O in those windows and excluding validation boundaries. O(3) molecules are counted once. Final configurations: S/O(3)-S/M **1 GPU**, L **2 GPUs**, XL **4 GPUs**; L's GPU hours include its earlier 1-GPU phase.
- **XL in progress (\*).** The XL cost/speed values are partial measurements through **step 1,250**, recorded at **2026-10-21 18:46 KST**. Its benchmark metrics and checkpoint are pending. The completed S/O(3)-S/M/L runs each reached 260,000 steps; their packaged weights come from validation-selected step 258,409.

### Updated checkpoints

The four completed checkpoints are attached to the **v0.2.0-rc.2 draft Release**.
The download links below become public when the release is published after code review and push.
Each archive contains `weights.ckpt`, exact configuration, test metrics, source metadata,
a checkpoint license notice and SHA-256 checksums. Weights match the benchmark checkpoints
exactly; optimizer state is excluded, so these files are for inference rather than exact training continuation.

| Model | Checkpoint archive |
| --- | --- |
| QHFlow2-s | [Download](https://github.com/seongsukim-ml/QHFlow2/releases/download/v0.2.0-rc.2/qhflow2-s-qh9stable-random-v0.2.0-rc.2.tar) |
| QHFlow2-s + O(3) group averaging | [Download](https://github.com/seongsukim-ml/QHFlow2/releases/download/v0.2.0-rc.2/qhflow2-o3-s-qh9stable-random-v0.2.0-rc.2.tar) |
| QHFlow2-m | [Download](https://github.com/seongsukim-ml/QHFlow2/releases/download/v0.2.0-rc.2/qhflow2-m-qh9stable-random-v0.2.0-rc.2.tar) |
| QHFlow2-l | [Download](https://github.com/seongsukim-ml/QHFlow2/releases/download/v0.2.0-rc.2/qhflow2-l-qh9stable-random-v0.2.0-rc.2.tar) |

See [checkpoint metadata and loading instructions](docs/releases/v0.2.0-rc.2-checkpoints.md).
The existing Google Drive links above contain the original checkpoint collection.

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

## Version History

### [v0.2.0-rc.2](docs/releases/v0.2.0-rc.2-checkpoints.md) — updated 2026-10-21 (draft)

Completed Stable/Muon benchmarks and checkpoint bundles for QHFlow2-s/m/l and
QHFlow2-s + O(3) group averaging; XL is training on four GPUs. The benchmark
image includes measured training cost and throughput. Validated compute
optimizations and selective torch.compile preserve checkpoint state keys;
see the [compute report](docs/releases/v0.2.0-rc.2.md) and
[geometry follow-up](docs/releases/v0.2.0-rc.2-geometry.md).

The [checkpoint/config audit](docs/MODEL_CONFIGS.md) verifies every size's
width, depth and parameter counts. S/M/L layer counts are unchanged. The XL
preset now matches the published five-layer checkpoint architecture; older
four-layer XL weights require their saved configuration or an explicit
`model.num_gnn_layers=4` override. The `src/qhflow2_version.py --json` command
reports the source version and Git provenance.

### [v0.2.0-rc.1](https://github.com/seongsukim-ml/QHFlow2/releases/tag/v0.2.0-rc.1) — 2026-10-14

Code prerelease with the following updates:

- Replaced Psi4 with PySCF for Q-tensor generation.
- Added corrected SO(3) models and identity/inversion-averaged O(3) models in
  **s/m/l/xl** sizes, with end-to-end equivariance checks.
- Added QH9Stable random-split training configurations: Muon starts at **1e-3**;
  norm, embedding, and other non-matrix parameters use AdamW at **1e-4**.
- Added code-version and model metadata to new checkpoints. All model families
  are available in `main`; release tags pin the source and model configs select
  the architecture.

<details>
<summary>Equivariance correction and checkpoint compatibility</summary>

This release addresses a feature-ordering bug in the legacy eSCN-to-e3nn
interface that breaks rotational equivariance for multi-channel spherical
features. The bug arose in QHFlow2 while adapting feature conventions from prior
models. In follow-up experiments, we observed similar performance after
correcting the ordering issue; the updated Muon benchmarks above combine this correction with
spectral gating and optimizer changes.

The original model names and default configurations retain their behavior for
checkpoint reproducibility. The historical packing-only implementation,
`QHFlow_so2_v5_1_ordering_fixed`, remains available alongside the complete
corrected models `QHFlow2_order_fixed` and `QHFlow2_O3`. Legacy checkpoints
require their original model configuration; the corrected architectures require
new training.

See the [equivariance history and validation scope](docs/EQUIVARIANCE_STATUS.md)
and [release notes](docs/releases/v0.2.0-rc.1.md) for details.

</details>

### [ver.0.1.0](https://github.com/seongsukim-ml/QHFlow2/releases/tag/ver.0.1.0) — 2026-10-07

Historical code prerelease with the original model implementations. The tag and
source snapshot are preserved for reproducibility.

### Planned: v0.2.0

We plan to retrain **QHFlow2-s/m/l/xl (order-fixed)** and
**QHFlow2-O(3)-s/m/l/xl** with Muon on QH9Stable and release updated checkpoints
and detailed benchmark results. The training configurations are available in
`v0.2.0-rc.1`. S/M/L and O(3)-S now have completed benchmarks and draft checkpoint
bundles; XL is in progress. O(3)-M/L/XL benchmarks remain planned.

See the [full changelog](CHANGELOG.md) and
[versioning and release guide](docs/RELEASING.md).

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

## License

QHFlow2 uses file-level licensing; see [LICENSE](LICENSE),
[REUSE.toml](REUSE.toml), and the [legal documentation](docs/legal/README.md).
Original QHFlow2 files are MIT, while identified AIRS/QHNet-derived software
remains GPL-3.0-only. CC-BY-NC-SA-4.0 applies only to QHFlow2 checkpoint weight
files trained on QH9, not to QHFlow2 source code. External datasets retain
their own upstream terms as listed in the
[dataset licensing guide](docs/legal/DATASETS.md).
