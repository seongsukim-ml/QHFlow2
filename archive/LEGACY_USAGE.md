# Archived usage before the 2026-10-21 cleanup

Historical commands and paths; use ../reproduce/README.md for current runs.

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
