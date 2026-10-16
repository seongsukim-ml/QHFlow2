# QHFlow2 geometry and execution optimization

Status: bounded CPU/CUDA validation; full training stays stopped at user request.
Continues the experiment started on 2026-10-16; later files retain that launcher directory.

Source: `/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-geometry-optimized-20261017`.
Baseline: `/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-compute-v020rc2-20261016`, commit `28d7216`.
Python: `/home/gpuuser/dataset/_cache/qhflow2-stable-muon-venv/bin/python`.
Dataset: `/dataset/seongsu/shared-home/data/QH9_shard/QH9Stable_shard/processed`.
Official random split SHA256: `b6bfadb51aedc894f388ffd786c6c3a682d900aa0ff229a94b678b4c72b898c1`.

The three suites use one queue-selected GPU each. No hard GPU reservations
exist; queue leases and a job-owned autoresearch drain isolate measurements.
All other jobs and workers remain untouched. Outputs are below
`/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-geometry-{models,modes,o3}-20261017/`.
The output spelling denotes three explicit names, not a runtime path resolver.

Validate the runner without CUDA:

```bash
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-10-16/qhflow2_geometry_benchmark.sh validate models 3 /dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-geometry-models-20261017
```

Queue the bounded benchmarks:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python /dataset/seongsu/shared-home/workspace/_global_auto_script/experiment_queue/sc26_queue.py enqueue /dataset/seongsu/shared-home/workspace/worktrees/qhflow2-geometry-optimized-20261017/_my_script/experiment/2026-10-16/queue-geometry.json
```

Each case uses a fresh child interpreter and identical weights and official
training/validation row IDs. Effective batch 32, FP32/TF32-off, native Muon and
AdamW remain unchanged. Twenty-four timed updates use distinct batches after
four warmups. Training time includes loader wait, host-to-device copy, model,
clipping and optimizer updates. Three timed validation batches of 32 force all
existing sampling/orbital metrics. Preparation and first validation are separate.
The protocol checks every initial parameter gradient, variable graph sizes,
rotation/inversion covariance and strict checkpoint reload. Compiler mode and
O3 paired-branch variants must pass these checks before any adoption.

Metadata caches are limited to one no-grad sampling/validation context and
invalidated by tensor identity/version. They never cache learned radial outputs.
Cached Wigner frames preserve subsequent RNG state by consuming the original
per-edge random-roll draw. Graph-gradient calls bypass geometry caches.
