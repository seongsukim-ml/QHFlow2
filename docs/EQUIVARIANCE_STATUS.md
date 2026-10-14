# Equivariance issue and separate correction

## Follow-up: SO(3), O(3), and Stable/Muon implementations

The separate [Stable/Muon branch](https://github.com/seongsukim-ml/QHFlow2/tree/codex/qhflow2-stable-muon)
now implements `QHFlow2_order_fixed` and `QHFlow2_O3` with s/m/l/xl
configurations. It combines corrected packing with the QH9 AO frame and
spectral gating, then optionally averages identity and inversion predictions.
Complete-model rotation/reflection tests and CPU training preflights are
described in the [Stable/Muon guide](QH9STABLE_MUON.md). Full training and
benchmark evaluation are still pending. The historical packing-only variant
and its validation limits below remain available separately.

## Confirmed legacy bug

The legacy `QHFlow_so2_v5_1` flattens eSCN node and atom-pair embeddings directly
from `[batch, (lmax + 1)^2, channels]`. Within each angular-degree block, eSCN
stores `[m, channel]`, whereas e3nn's `channels x l` irreps require
`[channel, m]`. Direct flattening mixes magnetic components and channels. For
multiple channels and non-scalar features, it does not commute with rotation.
This bug was present before the Psi4-to-PySCF dependency replacement and is
independent of Q-tensor generation.

The bug arose in QHFlow2 while adapting feature conventions from prior models.
The integration did not reconcile the different component/channel layouts at
the eSCN-to-e3nn boundary.

Both the node and pair paths in `QHFlow_so2_v5_1` are affected. The SO2, SO3,
no-time, and no-time SO2/SO3 variants contain the same direct-flatten pattern;
the experimental correction currently targets only `QHFlow_so2_v5_1`.

## Separate experimental implementation

The [codex/qhflow2-equivariance-fix branch](https://github.com/seongsukim-ml/QHFlow2/tree/codex/qhflow2-equivariance-fix)
provides `QHFlow_so2_v5_1_ordering_fixed`. It transposes the component and channel
axes **within each l block** before packing both node and pair embeddings.
The original model names, default configurations, and legacy packing behavior
are preserved. The variant adds no trainable parameters.

The old and new variants have matching parameter shapes, but those parameters
operate on different feature layouts. Loading a legacy checkpoint into the new
variant is not a behavior-preserving migration. Use a separate run and retrain
or explicitly validate any transfer; keep legacy checkpoints with their
original model configuration.

On the experimental branch, select
`model=QHFlow_so2_v5_1_ordering_fixed_small` in the QH9 or MD17 training entrypoint.
This is an explicit opt-in configuration; selecting the existing model keeps
the existing behavior.

## Performance observations

In follow-up experiments, we observed similar performance after correcting the
feature-ordering issue, compared with the legacy implementation. Detailed
benchmark results will accompany the planned model update below. This empirical
performance observation does not by itself establish full-model equivariance;
the validation scope and remaining rotation checks are described separately.

## Validation and limits

`tests/test_spherical_layout.py` on the experimental branch reproduces the bug
and tests proper rotations at lmax 4 with 1, 2, 64, and 128 channels and three
random seeds. The one-channel case is a control: direct flattening is correct
there. The corrected conversion commutes with the independent e3nn rotation
representation at float64 precision. Tests also cover gradients and the actual
diagonal-head layers, including the model's all-even Expansion convention.

`tests/test_ordering_fixed_model.py` checks separate model registration, strict
state-dict shape compatibility, and rotation of both diagonal and off-diagonal
heads using prescribed equivariant backbone embeddings. This isolates both
conversion call sites from the backbone's numerical approximations. These
tests, together with the Q-tensor and time-embedding regression tests, pass
(30 tests; CPU, Python 3.12, PyTorch 2.11, fairchem-core 2.6.0).

These are component-level regression tests. They do not establish full-model
equivariance, reflection/inversion equivariance, trained-checkpoint accuracy,
or the impact on previously reported benchmark results. An initial untrained
full-model smoke test also completed, but nonzero rotation residuals remained
after fixing the layout. Backbone numerical behavior and the molecular/AO
coordinate convention still require a dedicated end-to-end audit. The
`ordering_fixed` name describes this specific correction, not a certification
of full-model equivariance. Those audits and benchmark evaluation remain
required before a corrected release.

To run the regression checks from the repository root with the model's
dependencies installed:

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_spherical_layout.py tests/test_ordering_fixed_model.py \
  tests/test_onsite_overlap.py tests/test_time_embedding.py
```

The model integration test is skipped if fairchem is unavailable; a skipped
test must not be counted as a successful integration check.

## Release plan

We acknowledge that this bug existed in the legacy implementation. We plan to
retrain the following model families with the **Muon optimizer** and update them
for use as benchmarks:

| Model family | Planned variants |
| --- | --- |
| QHFlow2 (order-fixed) | QHFlow2-s, QHFlow2-m, QHFlow2-l, QHFlow2-xl |
| QHFlow2-O(3) | QHFlow2-O(3)-s, QHFlow2-O(3)-m, QHFlow2-O(3)-l, QHFlow2-O(3)-xl |

These eight trained variants are planned releases. The initial experimental
branch provides the ordering correction for `QHFlow_so2_v5_1`; the Stable/Muon
branch now supplies the new architectures and training configurations.
Updated checkpoints, model
configurations, and detailed benchmark results will be released after
end-to-end equivariance validation, retraining, and benchmark evaluation.
The legacy implementation remains available separately for reproducibility.
