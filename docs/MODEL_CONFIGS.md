# Checkpoint-aligned QH9Stable model sizes

The 2026-10-21 audit reads the configuration and every state-dict tensor shape
from all four published QH9Stable-random checkpoints at
[ksusu/QHFlow2-QH9](https://huggingface.co/ksusu/QHFlow2-QH9/tree/main/QH9Stable-random).
The pinned revision, checkpoint URLs, publisher-declared file checksums and
locally measured metadata checksums are in [model_parameter_counts.json](model_parameter_counts.json).
Each legacy model has exactly matching state keys and shapes. Only metadata
was fetched remotely; this is an architecture audit, not a new inference run
or a verification of the complete checkpoint file checksums.

| Size | Hidden | Bottleneck | Backbone layers | Hamiltonian layers | Legacy total | Corrected / O(3) total |
|---|---:|---:|---:|---:|---:|---:|
| s | 64 | 32 | 3 | 2 | 12,138,169 | 12,374,713 |
| m | 128 | 64 | 3 | 2 | 43,305,785 | 44,245,817 |
| l | 256 | 64 | 4 | 2 | 183,658,809 | 187,993,401 |
| xl | 512 | 128 | 5 | 3 | 990,091,065 | 1,013,184,313 |

| Size | Legacy trainable | Corrected / O(3) trainable | Added H-input parameters | Added spectral-gate parameters |
|---|---:|---:|---:|---:|
| s | 12,138,105 | 12,374,649 | 9,344 | 227,200 |
| m | 43,305,657 | 44,245,689 | 35,072 | 904,960 |
| l | 183,658,553 | 187,993,145 | 0 | 4,334,592 |
| xl | 990,090,553 | 1,013,183,801 | 0 | 23,093,248 |

Total means `sum(p.numel() for p in model.parameters())`; trainable additionally
requires `p.requires_grad`. The difference is fixed time-embedding frequencies
(64/128/256/512). Buffers are excluded. Counting Adam moment tensors alone is
incorrect: parameters that never received a gradient can lack optimizer state.
O(3) averages identity/inversion predictions with shared weights and adds no
parameters. Correcting packing alone also adds no parameters; the full
corrected model additionally uses spectral gating and the corrected AO frame.

## What differed

- S/M/L widths and layer counts match the published checkpoints.
- The previous XL YAML had four backbone layers, while the original checkpoint
  has five (`node_attr_backbone.blocks.0` through `.4`). This caused unexpected
  fifth-layer keys when using that YAML for original-checkpoint inference.
  All three XL presets now use five layers.
- The legacy factory reads `use_block_H_fix`, not `use_block_H`. Original S/M
  checkpoints omit the former, so the effective setting is false. Their YAMLs
  now state it explicitly. Original L/XL have it true. The corrected and O(3)
  families retain the true setting already used by the completed Muon runs.
- The supplied table's 12M/43M/183M/990M are approximate labels. Use the exact
  counts above; the audited L checkpoint is 183.658809M.

## Checkpoint compatibility and continuation

Use `QHFlow_so2_v5_1_{small,middle,large,extra_large}` for original checkpoints;
use `QHFlow2_{s,m,l,xl}` or `QHFlow2_O3_{s,m,l,xl}` for corrected checkpoints.
The original and corrected families are not interchangeable checkpoint schemas.
Keep strict loading enabled; do not discard unmatched keys to make loading pass.

The completed corrected S/M/L runs remain compatible without architecture
changes. An older corrected XL with four layers has 930,068,537 total and
930,068,025 trainable parameters. Resume it with its embedded configuration or
an explicit `model.num_gnn_layers=4` override. Five-layer XL is fresh training,
not continuation from the four-layer smoke or from an original checkpoint.

The five-layer run keeps QH9Stable random split, seed42, Muon1e-3,
AdamW1e-4 for norm/embedding, effective batch32 and 260000 optimizer steps.
No automatic retraining of S/M/L is required by this config audit.
