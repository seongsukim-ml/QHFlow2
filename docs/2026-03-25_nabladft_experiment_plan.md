# nablaDFT Experiment Plan

**Date:** 2026-03-25
**Dataset:** nablaDFT train_5k (28,005 samples), eval on test_2k (2,774 samples)
**Hardware:** H200 143 GiB per GPU

## Models

| Name | hidden | bottle | gnn_layers | ham_gnn | Params | Config |
|------|--------|--------|------------|---------|--------|--------|
| **Small** | 64 | 32 | 3 | 2 | **13.0M** | `QHFlow_so2_v5_1_nabla_small` |
| **Mid** | 128 | 64 | 3 | 2 | **46.8M** | `QHFlow_so2_v5_1_nabla_middle` |

## Experiments

### Exp 1: nabla-small (GPU 7)

```bash
cd /home1/irteam/data-vol1/projects/QHFlow2
CUDA_VISIBLE_DEVICES=7 python -m qhflow2.experiment.train_qh9 \
  --config-name config_flow_v2_nabla \
  model=QHFlow_so2_v5_1_nabla_small \
  wandb.run_name='nabla-small-train5k' \
  'wandb.tags=[nabladft,small,train5k]'
```

### Exp 2: nabla-mid (GPU 6)

```bash
cd /home1/irteam/data-vol1/projects/QHFlow2
CUDA_VISIBLE_DEVICES=6 python -m qhflow2.experiment.train_qh9 \
  --config-name config_flow_v2_nabla \
  model=QHFlow_so2_v5_1_nabla_middle \
  wandb.run_name='nabla-mid-train5k' \
  'wandb.tags=[nabladft,mid,train5k]'
```

## Common Settings

| Setting | Value | Note |
|---------|-------|------|
| batch_size | 16 | H200 143 GiB, ~80 GiB for mid |
| num_workers | 8 | lazy LMDB, fork-safe |
| total_steps | 100,000 | ~45 epochs over 22,404 train samples |
| warmup_step | 1,000 | |
| lr schedule | cosine decay | end_lr=1e-7 |
| optimizer | AdamW | |
| precision | fp32 | |
| num_ode_steps_val | 1 | fast validation |
| num_ode_steps_test | 3 | full eval at test time |
| loss | hamiltonian MAE+MSE (10.0) | |
| init_ham | hcore (T+V) | residual target |
| basis | def2-svp-nabla | full orbitals (32 slots) |
| eval | test_2k_conformers (2,774) | separate LMDB |

## Data Split

- train_5k LMDB: 28,005 samples
- Split: 80/10/10 = 22,404 train / 2,800 val / 2,801 test
- Eval dataset: test_2k_conformers (2,774 samples, independent)

## Expected Timeline

| Model | Est. time/step | Total (100k steps) |
|-------|---------------|-------------------|
| Small | ~0.5s | ~14h |
| Mid | ~1.5s | ~42h |
