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
- Prepare selective torch.compile support using primitive layer metadata,
  without changing parameter names or checkpoint keys. GPU speed and correctness
  measurements determine whether compilation is enabled for each run.
- Continue the four current training runs from their saved model, optimizer and
  scheduler states after CUDA resume checks. This candidate is not a published
  release and does not include completed benchmark weights.

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
