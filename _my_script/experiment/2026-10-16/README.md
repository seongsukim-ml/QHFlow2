# QHFlow2 Stable/Muon training

User-requested runs: QHFlow2-s (order-fixed) and QHFlow2-O(3)-s, each on one
H100 through the SC26 queue. Both use QH9Stable's existing random split,
seed 42, effective batch 32, 260,000 optimizer steps, no warmup, Muon 1e-3,
and AdamW 1e-4 for norm/embedding and non-matrix parameters.

Source starts at release `v0.2.0-rc.1` (`1a49f2d`). Model and optimizer code
are unchanged. This isolated worktree adds the launch runner and manifests.
The queue pins its committed source and hashes the external launcher and split.

- Source: `/dataset/seongsu/shared-home/workspace/worktrees/qhflow2-stable-muon-training-20261016`
- Python: `/home/gpuuser/dataset/_cache/qhflow2-stable-muon-venv/bin/python`
- Data: `/dataset/seongsu/shared-home/data/QH9_shard/QH9Stable_shard/processed`
- Split: `/dataset/seongsu/shared-home/data/QH9_shard/QH9Stable_shard/processed/processed_QH9Stable_random_12.json`
- Launcher: `/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-10-16/qhflow2_stable_muon.sh`
- SO(3) output: `/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-s-order-fixed-stable-muon-s42-20261016`
- O(3) output: `/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-o3-s-stable-muon-s42-20261016`

GPU selection is assigned by the queue on server 1. Each manifest passes the
allocated index to the launcher, which checks live GPU use before starting.
The autoresearch backfill yields through the queue's existing drain mechanism.

Validation (does not train):

```bash
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-10-16/qhflow2_stable_muon.sh validate QHFlow2_s 0 /dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-s-order-fixed-stable-muon-s42-20261016
/dataset/seongsu/shared-home/workspace/project/MALOQ/_my_script/experiment/2026-10-16/qhflow2_stable_muon.sh validate QHFlow2_O3_s 0 /dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/qhflow2-o3-s-stable-muon-s42-20261016
```

Enqueue the CUDA smoke first. Enqueue full training only after both smoke
receipts report `passed`:

```bash
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python /dataset/seongsu/shared-home/workspace/_global_auto_script/experiment_queue/sc26_queue.py enqueue /dataset/seongsu/shared-home/workspace/worktrees/qhflow2-stable-muon-training-20261016/_my_script/experiment/2026-10-16/queue-smoke.json
/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python /dataset/seongsu/shared-home/workspace/_global_auto_script/experiment_queue/sc26_queue.py enqueue /dataset/seongsu/shared-home/workspace/worktrees/qhflow2-stable-muon-training-20261016/_my_script/experiment/2026-10-16/queue-full.json
```

Smoke runs use one 64-sample training epoch and 32 official validation samples,
then save/reload the checkpoint and resume one optimizer step. Smoke outputs
use the full-output name plus `-cuda-smoke/attempt-N`; retain receipts/logs and
checkpoint evidence for reproducibility. They do not change the official split.
Full runs use all 104,664 training and 13,083 validation rows. After training,
the best checkpoint is tested on all 13,084 official test rows.

Progress: each run writes `run.json`, `resolved-config.yaml`, `status.json`,
`progress.jsonl`, `metrics/`, and `console.log`. Best/last epoch checkpoints are
under `training/checkpoints/`, with an additional recovery checkpoint every
1,000 optimizer steps under `recovery/`. W&B remains disabled. The source is
frozen after enqueueing; use queue state and output receipts for live status.
