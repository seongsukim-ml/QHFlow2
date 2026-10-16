# Changelog

Code releases are independent of trained checkpoint releases. Git tags pin the
source and configuration files; `model.version` selects an architecture.

## 0.2.0-rc.2 — development candidate, 2026-10-16

- Cache constant coupling tensors and the unweighted noise expansion; vectorize
  AO block cutting, reconstruction and reverse-edge indexing.
- Avoid cloning unused model inputs and collecting ODE trajectories during
  validation; reuse overlap factors and target spectra within a validation batch.
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
