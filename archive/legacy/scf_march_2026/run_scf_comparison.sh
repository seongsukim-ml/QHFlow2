#!/bin/bash
# Unified SCF acceleration comparison — 6 models, 300 mols, same GPU
# All on GPU 6 (sequential) for fair wall-clock comparison
set -e

PROJ=/home1/irteam/data-vol1/projects/QHFlow2
PY=/home1/irteam/data-vol1/conda/envs/util-gpu4pyscf/bin/python
SCRIPT=$PROJ/scripts/benchmark_scf_from_pred_flat.py
OUT=$PROJ/rebuttal/scf_comparison
N=300
SEED=42
export CUDA_VISIBLE_DEVICES=6

cd $PROJ
mkdir -p $OUT

# ── Phase A: scratch baseline (once) ──
echo "$(date '+%H:%M:%S') Phase A: scratch baseline (300 mols, b3lyp5, GPU 6)"
$PY $SCRIPT \
    --pred_dir eval_results/QH9Stable-random/small/pred_flat \
    --label scratch_baseline --num_samples $N --seed $SEED \
    --with_scratch \
    --output_dir $OUT/scratch_baseline

CACHE=$OUT/scratch_baseline/scratch_results.json

# ── Phase B: pred-init for each model (sequential, same GPU) ──
declare -A PRED_DIRS
PRED_DIRS[qhflow2_small]="eval_results/QH9Stable-random/small/pred_flat"
PRED_DIRS[qhflow2_middle]="eval_results/QH9Stable-random/middle/pred_flat"
PRED_DIRS[qhflow2_large]="eval_results/QH9Stable-random/large/pred_flat"
PRED_DIRS[qhflow2_xl]="eval_results/QH9Stable-random/extra_large/pred_flat"
PRED_DIRS[qhflow_260k]="rebuttal/QHFlow_260k_p1.0-QH9Stable-random_IG_True-US_True/QHFlow_260k_p1.0-QH9Stable-random_IG_True-US_True/pred_flat"
PRED_DIRS[qhnet_residue]="rebuttal/Real_QHNet_residue-QH9Stable-random-UR_True/Real_QHNet_residue-QH9Stable-random-UR_True/pred_flat"

for label in qhflow2_small qhflow2_middle qhflow2_large qhflow2_xl qhflow_260k qhnet_residue; do
    echo ""
    echo "$(date '+%H:%M:%S') Phase B: $label"
    $PY $SCRIPT \
        --pred_dir "${PRED_DIRS[$label]}" \
        --label $label --num_samples $N --seed $SEED \
        --scratch_cache $CACHE \
        --output_dir $OUT/$label
done

echo ""
echo "$(date '+%H:%M:%S') All done. Results in $OUT/"
