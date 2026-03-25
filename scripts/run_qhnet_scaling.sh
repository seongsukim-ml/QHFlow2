#!/bin/bash
# QHNet scaling: large (GPU 6) + xl (GPU 7)
# lr=5e-4 (same as QHFlow2), grad_clip=1.0, no warmup
set -euo pipefail

PROJ="/home1/irteam/data-vol1/projects/QHFlow2"
PY="/home1/irteam/data-vol1/conda/envs/qhflow/bin/python"
TRAIN="$PROJ/src/qhflow2/experiment/train_qh9.py"
CONFIG="qhnet_scaling"

# Large: hs=256, hbs=64, gnn=7, ~143.8M
echo "Starting QHNet large on GPU 6 (lr=5e-4, clip=1.0, no warmup)"
CUDA_VISIBLE_DEVICES=6 nohup $PY $TRAIN \
  --config-name $CONFIG \
  model=QHNet_large \
  dataset.learning_rate=5e-4 \
  gradient_clip_val=1.0 \
  > "$PROJ/logs/qhnet_large_gpu6.log" 2>&1 &
echo "  PID: $!"

# XL: hs=512, hbs=128, gnn=7, ~565.3M (bs=8 to avoid OOM)
echo "Starting QHNet xl on GPU 7 (lr=5e-4, clip=1.0, bs=8, no warmup)"
CUDA_VISIBLE_DEVICES=7 nohup $PY $TRAIN \
  --config-name $CONFIG \
  model=QHNet_xl \
  dataset.learning_rate=5e-4 \
  dataset.batch_size=8 \
  gradient_clip_val=1.0 \
  > "$PROJ/logs/qhnet_xl_gpu7.log" 2>&1 &
echo "  PID: $!"

echo ""
echo "QHNet scaling experiments launched. Monitor with:"
echo "  tail -f $PROJ/logs/qhnet_*_gpu*.log"
