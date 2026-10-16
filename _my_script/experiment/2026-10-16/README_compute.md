# QHFlow2 compute optimization

User-requested optimization of the four QH9Stable Muon models, with
checkpoint continuation support after numerical, equivariance, and CUDA verification.
Data and learning rates retain the official random split and prior settings.

Final status (2026-10-16): all original full training jobs stopped at the user's
request. Benchmarks and bounded resume validations passed; no full training
restart was launched. See [the final report](../../../docs/releases/v0.2.0-rc.2.md).

Source: `/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-compute-v020rc2-20261016`
Baseline: `/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-stable-muon-ml-training-20261016` at `39bce6e`.
Environment: `/home/gpuuser/dataset/_cache/qhflow2-stable-muon-venv/bin/python`.
Data: `/dataset/seongsu/shared-home/data/QH9_shard/QH9Stable_shard/processed`.
Final original/eager output: `/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-final-compute-parity-20261016`.
Final eager/compiled output: `/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-dynamic-compile-benchmark-20261016`.

The paired benchmark uses one queue-allocated H100. A job-owned queue drain
excludes adaptive filler work from that GPU while the benchmark runs. All other
GPU workloads and the existing source/output histories remain intact.

```bash
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-10-16/qhflow2_compute_benchmark.sh validate all 4 /dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-compute-benchmark-20261016
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python /dataset/seongsu/shared-home/workspace/_global_auto_script/experiment_queue/sc26_queue.py enqueue /dataset/seongsu/shared-home/workspace/worktrees/qhflow2-compute-v020rc2-20261016/_my_script/experiment/2026-10-16/queue-benchmark.json
```

Each source is benchmarked in a separate process on identical training and
validation samples. Timing includes forward, backward, clipping and Muon/AdamW
updates at effective batch 32. Validation forces the existing one- and three-step
ODE metrics; skipped O(3) baseline sampling is explicitly marked invalid.
The final benchmark also compares initial losses, parameter names, and all
parameter gradients at identical weights and random seed.

Compilation is opt-in through `common.compile.compile_flow(module, scope="blocks")`
or `QHF_COMPILE_SCOPE=blocks` with the prepared resume launcher at
`/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-10-16/qhflow2_optimized_resume.sh`.
The measured source is `86e2ebb3e612dbf9ebc4bdf989ac557deed8e58d`; later changes
to this candidate only document the completed validation.
