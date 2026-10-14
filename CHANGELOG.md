# Changelog

Code releases are independent of trained checkpoint releases. Git tags pin the
source and configuration files; `model.version` selects an architecture.

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
