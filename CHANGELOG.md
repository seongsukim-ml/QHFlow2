# Changelog

Code releases are independent of trained checkpoint releases. Git tags pin the
source and configuration files; `model.version` selects an architecture.

## Unreleased — benchmark/checkpoint update and config audit, 2026-10-21

- Display ERIC as a percentage and add original S/M/L/XL training GPU hours
  reconstructed from W&B histories. Preserve timing records,
  resume/duplicate handling, checkpoint batch settings and hardware limitations.
- Separate 1-step and 3-step inference columns, with a reproducible checkpoint
  timing runner; leave unmeasured speed cells empty pending an exclusive GPU.
- Display saved updated H/orbital metrics to three decimals and SCF percentages
  to two; retain unrounded source values and original-paper precision.
- Publish four verified inference checkpoints on Hugging Face while preserving
  all 16 original checkpoints; GitHub release assets remain in a draft.
- Add five release training profiles matching the recorded final configurations,
  and a training entrypoint with evaluation shards that do not repeat molecules.
  Move other training/evaluation experiments into an indexed, hashed archive.
- Recover 300 molecular identities and 1,800 paired SCF cycle records from the
  March 2026 records. The maintainer confirmed on 2026-10-21 that these are the
  original Figure 4 molecules; preserve the recovered solver settings separately.
- Add SCF cycle ratio and post-hoc ERIC columns, with reproducible aggregation
  of all 300 paired molecules for six original models. Updated models retain
  dashes until evaluated; incomplete or nonconverged cohorts have no headline mean.
- Add checkpoint-to-NPZ inference and reproducible SCF/one-shot energy-force
  evaluation with explicit units, solver policies, hashes and failure records.
- Add the README benchmark image, update date, original-paper references,
  spectral gate/O(3) labels, and measured training GPU hours for
  completed S/O(3)-S/M/L runs. Only XL is marked `(on train)`.
- Prepare exact best-validation inference exports for S, O(3)-S, M and L
  with configs, metrics, provenance and checksums; XL is still training.
  See the [draft checkpoint update](docs/releases/v0.2.0-rc.2-checkpoints.md).
- Keep release history below the usage guide and retain the existing
  collapsed explanation of the legacy equivariance issue.

- Verify all four published QH9Stable-random checkpoint state shapes against
  the legacy model, including its effective Hamiltonian-input flag.
- Correct XL backbone depth from four to five in the legacy, corrected and
  O(3) presets. Existing four-layer checkpoints require an explicit
  `model.num_gnn_layers=4` override; S/M/L widths and depths are unchanged.
- Make the disabled H-input flag explicit in legacy S/M presets and record
  exact total/trainable counts for both model families in
  [MODEL_CONFIGS.md](docs/MODEL_CONFIGS.md).

## 0.2.0-rc.2 — development candidate, 2026-10-16

- Cache constant coupling tensors and the unweighted noise expansion; vectorize
  AO block cutting, reconstruction and reverse-edge indexing.
- Avoid cloning unused model inputs and collecting ODE trajectories during
  validation; prepare overlap-factor and target-spectrum reuse within a batch.
  The follow-up below fixes the missing cache forwarding in validation metrics.
- Add a lean QH9Stable Hamiltonian-only loader preset with pinned memory and
  persistent workers. The official split, objective, full validation, effective
  batch 32, native Muon and AdamW learning rates remain unchanged.
- Fix O(3) ODE sampling: the projected model deliberately omits latent features,
  but the former sampler indexed them unconditionally. Training loss was valid;
  sampling metrics raised KeyError and were missing from affected validation.
- Validate selective torch.compile for backbone blocks and tensor heads,
  preserving parameter names and checkpoint keys. Fixed pairwise contractions
  prevent batch-size specialization from einsum path planning.
- On isolated H100 benchmarks at effective batch 32, reduce update time by
  12.9% / 38.5% / 22.1% / 39.3% for s / O(3)-s / m / l, including microbatch
  tuning. Compilation contributes a further 5.6–8.2% over optimized eager code.
- Verify model, optimizer and scheduler restoration for all four runs, including
  Lightning progress migration when changing microbatch size. Original full
  training jobs are stopped at the user's request; no full restart was launched.
  This candidate is not a published release or completed benchmark checkpoint.

See the [validation report](docs/releases/v0.2.0-rc.2.md) for measurement
conditions, memory tradeoffs, startup cost and checkpoint limitations.

Follow-up validation (2026-10-17 KST): reuse geometry, Wigner matrices and AO
index/convention plans across sampling steps; add opt-in paired O(3) execution.
The ordinary QH9 entrypoint now honors the optimized compile preset and uses
no-grad validation with versioned tensors. The legacy presets retain their
defaults. Compared with the already optimized and compiled candidate, O(3)-s
update time falls another 10.4%; validation time falls 56.9% / 50.1% / 46.7% /
29.9% for s / O(3)-s / m / l. Training throughput for s/m/l is essentially
unchanged. See the [samples/s report](docs/releases/v0.2.0-rc.2-geometry.md) for
matched measurements, rejected compiler modes and checkpoint checks.

## 0.2.0-rc.1 — 2026-10-14

- Release the corrected SO(3) and identity/inversion-averaged O(3) families in
  s/m/l/xl sizes, alongside the original model names and defaults.
- Include the PySCF Q-tensor implementation that removes the Psi4 dependency.
- Prepare QH9Stable random-split training with Muon at 1e-3 and AdamW at 1e-4
  for norm, embedding, and other parameters unsupported by native Muon.
- Add a code-version API/CLI and source/model metadata to new checkpoints and
  preparation reports, without changing state-dict keys.
- Consolidate the corrected implementations into `main`; use immutable release
  tags to choose a source snapshot instead of relying on development branches.

This is a **code prerelease**. Full retraining, GPU memory profiling, and new
benchmark checkpoint/results publication are pending. See the
[release notes](docs/releases/v0.2.0-rc.1.md) for validation and compatibility.

## ver.0.1.0 — 2026-10-07

Existing historical prerelease. Its tag and contents are preserved unchanged.
It predates the PySCF dependency replacement and corrected model families,
and contains the [documented legacy equivariance issue](docs/EQUIVARIANCE_STATUS.md).
