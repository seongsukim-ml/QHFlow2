# QHFlow2 reproduction

Updated 2026-10-21 · code/checkpoints v0.2.0-rc.2.

Use the repository root as the working directory and install the
[Stable/Muon runtime](../docs/QH9STABLE_MUON.md#stable-data-and-runtime).
Current source checkout on the shared server:
`/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-checkpoint-configs-20261021`.
The commands below use canonical persistent output paths on that server.
On another machine, supply your own absolute dataset, checkpoint and output paths.

## Training

| Config name | Model | GPUs | Per-GPU batch | Accumulation | Backbone layers |
| --- | --- | ---: | ---: | ---: | ---: |
| `config_release_s` | QHFlow2-s | 1 | 32 | 1 | 3 |
| `config_release_o3_s` | QHFlow2-s + O(3) group averaging | 1 | 32 | 1 | 3 |
| `config_release_m` | QHFlow2-m | 1 | 32 | 1 | 3 |
| `config_release_l` | QHFlow2-l | 2 | 16 | 1 | 4 |
| `config_release_xl` | QHFlow2-xl | 4 | 2 | 4 | 5 |

All have effective batch 32, seed 42, Muon 1e-3 / AdamW 1e-4, no warmup,
260,000 steps, full validation, and block-level compilation. These presets
resolve to the exact saved **final training settings**; [training-runs.json](training-runs.json)
records the source run and configuration hash. O(3)-S uses separate identity and
inversion forwards (`batch_branches=false`). Conservative architecture-only
defaults and original S/M/L/XL checkpoint definitions remain in `src/config_qh9/model`.

```bash
# Inspect only; does not train.
PYTHONPATH=src python -m experiment.train_release --config-name=config_release_xl --cfg job

# Start a fresh S run when intentionally executed.
PYTHONPATH=src python -m experiment.train_release --config-name=config_release_s
```

For L or XL use the corresponding preset with 2 or 4 visible GPUs. The release
entrypoint preserves the recorded no-padding evaluation split and actual
molecule-count metric weighting. Test an exported checkpoint with `mode=test`
and `ckpt=/absolute/path/to/weights.ckpt`. The new entrypoint has configuration/sharding checks; a new multi-GPU training
run was not launched as part of this cleanup. The recorded runs used earlier resume
segments and custom operational launchers; these final presets alone do not
reproduce their exact stochastic training trajectory. A fresh run is a new
experiment. Exact continuation needs a **full** optimizer/loop checkpoint and
the original device layout; the public inference exports omit that state.

Historical MD17, ablation, fine-tuning and optimization settings are explicitly
opt-in through [archive](../archive/README.md). Do not mix legacy and corrected
model definitions when loading checkpoints.

## SCF and DFT energy/forces

### Molecule identities and Figure 4 provenance

[recorded-300.csv](scf/recorded-300.csv) gives the **recovered March 2026 SCF
re-experiment** cohort. Every molecule was matched to the local QH9 raw shard by
atomic numbers and coordinates (all 300 uniquely matched within 2e-6 Å), and
independently checked against the historical test split. Fields include:

- `prediction_index`: historical `pred_<index>.pt` file number / old test position.
- `dataset_index`: zero-based original QH9 dataset row.
- `source_record_id`, `shard`, `lmdb_key`: original stored identity, not an inferred QM9 name.
- `molecule_id`: `QH9Stable:<dataset_index>`; `formula` helps identify the structure but is not unique.
- `current_test_position`: position in the current sorted test split; it differs from the old order.

[recorded-300-cycles.csv](scf/recorded-300-cycles.csv) includes all 1,800 model/molecule
pairs, convergence flags and final energies. Selection was NumPy `default_rng(42)`,
300 without replacement from 13,084 historical test positions, then sorted.
The explicit CSV is authoritative for replay; do not reinterpret file numbers
as global dataset indices or use a different split order.

**Original Figure 4 is not yet fully traced.** The original paper's
[Appendix E.4](https://arxiv.org/html/2602.16897v1#A5.SS4) specifies B3LYP/def2-SVP,
grid 3, `conv_tol=1e-13`, GPU4PySCF and density fitting. Recovered March records
instead specify B3LYP5, `1e-9`, with no density fitting in their source script.
Their dates postdate the February paper. They are not relabeled as Figure 4's
original molecules or settings. [provenance.json](scf/provenance.json) preserves
source paths, hashes and this unresolved attribution. `paper-protocol.yaml`
encodes the documented protocol, while `recorded-protocol.yaml` replays the
recovered experiment. The old runtime versions were not saved in those results;
an exact numerical replay is therefore not certified.

### Checkpoint to traceable predictions

Use `infer` to generate NPZ inputs directly from a checkpoint and explicit cohort:

```bash
PYTHONPATH=src python -m reproduction.qh9 infer \
  --cohort reproduce/scf/recorded-300.csv \
  --checkpoint /absolute/path/to/weights.ckpt \
  --processed-directory /dataset/seongsu/shared-home/data/QH9_shard/QH9Stable_shard/processed \
  --split-file /dataset/seongsu/shared-home/data/QH9_shard/QH9Stable_shard/processed/processed_QH9Stable_random_12.json \
  --output /dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-scf-replay-inputs \
  --device cuda --seed 42 --ode-steps 3
```

This uses one molecule per inference batch, native PySCF AO order, Angstrom
positions and Hartree matrices. It records the checkpoint hash, seed, ODE steps,
cohort, split and each input file hash. It does not overwrite an existing export.
For original-paper models, use the corresponding original checkpoint.
Older checkpoints containing pickled configuration objects additionally require
`--trust-legacy-checkpoint`; use that option only for a checkpoint you trust.

`prepare` accepts a trusted local `pred_flat` directory from an existing run;
it requires the same cohort, checkpoint and dataset arguments plus `--pred-dir`,
and validates molecular geometry and AO overlap before conversion. In that
mode the checkpoint association is supplied by the operator and must be checked
against the original inference log. Prediction filenames must use the cohort's
`prediction_index` column. The old and new test orders must not be mixed.
For a historical energy/force replay, also pass `--gt-dir /absolute/path/to/gt_flat`
to preserve the saved reference matrices and their original precision. Without
it, the reference comes directly from the raw float64 dataset and may differ
from the older float32 prediction exports.

For a fresh cohort, run `select --split-file /absolute/path/to/split.json
--output /absolute/path/to/cohort.csv --count 300 --seed 42`; use `--count 0`
for the full test set. This writes a `new_replay_cohort` provenance sidecar.

### Evaluate

```bash
PYTHONPATH=src python -m reproduction.qh9 evaluate \
  --manifest /dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-scf-replay-inputs/manifest.json \
  --config reproduce/scf/recorded-protocol.yaml \
  --output /dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-scf-replay

PYTHONPATH=src python -m reproduction.qh9 evaluate \
  --manifest /dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-scf-replay-inputs/manifest.json \
  --config reproduce/energy_force/legacy-v4.yaml \
  --output /dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-energy-force-replay
```

The GPU protocol requires a compatible GPU4PySCF/CuPy installation. There is no
silent backend fallback. A CPU check uses a copied YAML with `backend: cpu` in
a new output directory and is reported as a separate runtime.

- SCF compares MinAO, predicted-H and dataset-H initialization with the same solver settings.
  Cycle ratio is **100 × sum(model cycles) / sum(MinAO cycles)**. Reduction is
  100 minus that ratio; neither quantity includes an extra initial-guess cycle.
- Energy/force evaluation extracts the original v4 algorithm without UMA,
  plotting or mutable shared caches. The default preserves its adaptive
  `1e-8` / `1e-3` overlap-eigenvalue **padding** rule and 0.03 Ha switch. Padding
  is not rank truncation. The reference uses dataset H with `1e-8`, padding 1.
- Both predicted and reference E/F are recomputed through the same DFT backend.
  The forces are the analytic DFT gradient at the supplied orbitals; they are
  **not** derivatives through the model. No self-consistency iteration is run
  in this one-shot mode. This is also not the separately converged-SCF reference
  reported by some variants of the archived scripts.
- Raw energy is total Hartree per molecule; force components are Hartree/Bohr.
  Energy MAE averages molecules; force MAE averages all Cartesian components
  over all atoms. Multiply by 27.211386... for eV and by 27.211386.../0.529177...
  for eV/Å. Keep units explicit when reproducing a paper table.
- `protocol.json`, per-molecule JSON and `summary.json` preserve settings,
  package versions, source/input hashes, timings and failures. Resume checks
  those hashes. Changed inputs, settings, code or runtime require a new output.
  A headline mean is withheld if any sample fails or an SCF pair does not converge.

This cleanup has not rerun all original-paper energy/force benchmarks. The
historical evaluators and command lines remain in [archive](../archive/README.md).
