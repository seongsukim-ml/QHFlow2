#!/bin/bash
# Expand_lmax ablation: L1,L2,L3,L4 on GPU 2,3,4,5
# Usage: bash scripts/run_expandL_ablation.sh
set -euo pipefail

PROJ="/home1/irteam/data-vol1/projects/QHFlow2"
PY="/home1/irteam/data-vol1/conda/envs/qhflow/bin/python"
TRAIN="$PROJ/src/qhflow2/experiment/train_qh9.py"
CONFIG="config_flow_v2_expandL_ablation"

declare -A GPU_MAP=(
  [1]=2
  [2]=3
  [3]=4
  [4]=5
)

declare -A MODEL_MAP=(
  [1]="QHFlow_so2_v5_1_middle_expandL1"
  [2]="QHFlow_so2_v5_1_middle_expandL2"
  [3]="QHFlow_so2_v5_1_middle_expandL3"
  [4]="QHFlow_so2_v5_1_middle"
)

for LMAX in 1 2 3 4; do
  GPU=${GPU_MAP[$LMAX]}
  MODEL=${MODEL_MAP[$LMAX]}
  echo "Starting expand_lmax=$LMAX on GPU $GPU (model=$MODEL)"
  CUDA_VISIBLE_DEVICES=$GPU nohup $PY $TRAIN \
    --config-name $CONFIG \
    model=$MODEL \
    > "$PROJ/logs/expandL${LMAX}_gpu${GPU}.log" 2>&1 &
  echo "  PID: $!"
done

echo ""
echo "All 4 experiments launched. Monitor with:"
echo "  tail -f $PROJ/logs/expandL*_gpu*.log"
echo "  nvidia-smi"
