# QH9Stable: corrected SO(3) models, O(3) averaging, and Muon

Code release **`v0.2.0-rc.1`**, also integrated into `main`, provides two separately
named model families. The legacy model names and checkpoints retain their
original behavior. These new architectures require new training runs. Use the
[release/version guide](RELEASING.md) to pin the source tag; a new code release
does not imply that trained benchmark checkpoints are available.

## Models

| Family | Registry version | Hydra model configs |
| --- | --- | --- |
| QHFlow2-s/m/l/xl (order-fixed) | `QHFlow2_order_fixed` | `QHFlow2_s`, `QHFlow2_m`, `QHFlow2_l`, `QHFlow2_xl` |
| QHFlow2-O(3)-s/m/l/xl | `QHFlow2_O3` | `QHFlow2_O3_s`, `QHFlow2_O3_m`, `QHFlow2_O3_l`, `QHFlow2_O3_xl` |

The SO(3) implementation corrects the per-degree component/channel packing,
uses the QH9 loader's `[y,z,x]` AO coordinate frame in the backbone, and replaces
sampled spherical-grid nonlinearities with spectral linear layers and scalar
gates. The earlier `QHFlow_so2_v5_1_ordering_fixed` corrected packing alone;
this new class also addresses the remaining rotation error found in that audit.
Per-atom Hamiltonian, initial-Hamiltonian, overlap, and flow-time conditioning
are indexed by atom, with graph-level times also supported.

The O(3) model shares one set of parameters across two forward passes:

```text
F_O3(x, H, H0, S) = [F(x, H, H0, S) + P F(-x, P H P, P H0 P, P S P) P] / 2
P = diag(+1,+1,+1, -1,-1,-1,-1,-1,-1, +1,+1,+1,+1,+1)
```

Here `P` is the physical AO inversion parity for padded def2-SVP blocks. Both
diagonal and off-diagonal outputs are transformed back and averaged, with
gradients through both evaluations. Combined with the SO(3) base this is the
projection onto O(3) equivariance. It uses two evaluations, not an average over
sampled rotations. The new models currently support QH9 AO blocks only.

Sizes preserve the previous small/middle/large/extra-large widths and depths:

| Size | Hidden / bottleneck | Backbone / head layers | SO(3) microbatch / accumulation | O(3) microbatch / accumulation |
| --- | --- | --- | --- | --- |
| s | 64 / 32 | 3 / 2 | 32 / 1 | 16 / 2 |
| m | 128 / 64 | 3 / 2 | 16 / 2 | 8 / 4 |
| l | 256 / 64 | 4 / 2 | 4 / 8 | 2 / 16 |
| xl | 512 / 128 | 4 / 3 | 1 / 32 | 1 / 32 |

These conservative microbatches give effective batch 32 on one device. They
are preparation defaults, not GPU memory measurements. All models use H0 and S
conditioning. O(3) averaging adds no trainable parameters.

## Optimizer and schedule

- Native `torch.optim.Muon`: eligible two-dimensional weights, starting at
  **1e-3**, momentum 0.95, five Newton–Schulz steps, `adjust_lr_fn=original`.
- AdamW: **all norm and embedding parameters**, plus biases, scalar gains,
  non-2-D tensors, and packed one-dimensional e3nn weights, starting at **1e-4**.
- Weight decay 0.01 for both groups; AdamW betas `(0.99, 0.999)`.
- Cosine decay over 260,000 optimizer steps, **no warmup**, gradient clipping
  at 1.0. Both groups keep their 10:1 learning-rate ratio.

The split follows the [native PyTorch Muon parameter contract](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html).
`MuonAdamW` presents a single optimizer to Lightning while sharing groups and
state with native Muon and AdamW. Scheduler changes, gradient accumulation,
AMP unscaling, and checkpoint state therefore cover both algorithms. The
preparation report lists every parameter's assigned optimizer.

## Stable data and runtime

Only **QH9Stable, random split** is selected: 104,664 training, 13,083 validation,
13,084 test rows. Full validation is enabled. The explicit read-only loader
opens existing shards and their paired split; it does not download, reprocess,
rename, or create dataset files.

```text
processed_directory: /dataset/seongsu/shared-home/data/QH9_shard/QH9Stable_shard/processed
split_file: /dataset/seongsu/shared-home/data/QH9_shard/QH9Stable_shard/processed/processed_QH9Stable_random_12.json
```

The stored record inspected in this preparation uses Angstrom coordinates and
native PySCF AOs: its overlap agrees with a fresh PySCF overlap to about
6e-16. A PySCF orbital-value regression independently checks the loader's p
permutation, d representation, proper rotations, and inversion. The existing
`ao_convention.json` narrative about native p ordering disagrees with this
numerical check; it has been preserved. No dataset or catalog was modified.

The checked CPU runtime uses Python 3.12, PyTorch 2.11, e3nn 0.6,
fairchem-core 2.6, PyG 2.7, PySCF 2.12.1, and Lightning 2.6.1. Native Muon is
required; the old Torch 2.4 installation example is for legacy models.
Use `requirements-stable-muon.txt` for the new dependencies, with matching
PyTorch/CUDA graph-extension wheels. A local cached runtime is prepared at:

```text
/home/gpuuser/dataset/_cache/qhflow2-stable-muon-venv/bin/python
```

That cache installs its additional packages locally and reads the existing
`proj-ml-dft-qhflow3-replay-sail` and `proj-dft-baselines-maloq-sc26` site packages
without modifying either shared environment.

## Prepare or launch

Run from the current code checkout:

```bash
cd /dataset/seongsu/shared-home/workspace/worktrees/qhflow2-equivariance-fix/src

# Inspect the code release and exact source commit before preparing a run.
python3 -m qhflow2_version --json

# Resolve all eight configurations and list optimizer assignments.
/home/gpuuser/dataset/_cache/qhflow2-stable-muon-venv/bin/python \
  -m experiment.prepare_qh9_stable_muon \
  --output /dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-stable-muon-config-preflight

# Optional: two real training samples, backward, optimizer and scheduler step.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
/home/gpuuser/dataset/_cache/qhflow2-stable-muon-venv/bin/python \
  -m experiment.prepare_qh9_stable_muon --smoke --models QHFlow2_s QHFlow2_O3_s \
  --output /dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-stable-muon-cpu-preflight

# These commands START full training when explicitly executed.
/home/gpuuser/dataset/_cache/qhflow2-stable-muon-venv/bin/python \
  -m experiment.train_qh9_stable_muon model=QHFlow2_s
/home/gpuuser/dataset/_cache/qhflow2-stable-muon-venv/bin/python \
  -m experiment.train_qh9_stable_muon model=QHFlow2_O3_s
```

Change the model config to select m/l/xl. Adjust microbatch and accumulation
together when tuning memory. The training config writes timestamped run
directories directly below
`/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/`.
W&B is disabled by default. Preparation refuses to overwrite an existing
report directory. The Stable entrypoint rejects QH9Dynamic.

## Validation scope

Regression tests cover complete-model rotations, translations, atom
permutations, inversion/improper rotations for O(3), input immutability,
backpropagation, AO conventions, optimizer partitioning, native-update
agreement, and optimizer/scheduler resume. New configuration and read-only
loader tests check all eight variants and split coverage.

The CPU preflight exercises the real flow corruption, Hamiltonian loss,
backward, and Muon/AdamW step on QH9Stable training samples. These checks are
implementation validation, not benchmark accuracy or GPU throughput results.
No full training run or new trained benchmark checkpoint is included here.

The preparation completed **58 tests**, CPU optimizer steps for **all eight
models**, and a Lightning automatic-optimization plus validation smoke for
`QHFlow2_O3_s`. The complete-model rotation tests use three transforms in
float32/float64; their tolerances are 5e-5 and 1e-9, respectively. The parameter
counts per size are 12,374,713 / 44,245,817 / 187,993,401 / 930,068,537 for
s/m/l/xl in either family. In particular, xl is a large model and its GPU
memory requirements still need profiling before a full launch.

Local validation artifacts are under
`/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-stable-muon-preparation-20261014/`.
