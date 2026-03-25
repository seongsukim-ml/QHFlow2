#!/bin/bash
# Evaluate all pre-trained QH9 checkpoints
# Usage: CUDA_VISIBLE_DEVICES=5 bash scripts/eval_pretrained.sh
set -euo pipefail

PROJ="/home1/irteam/data-vol1/projects/QHFlow2"
PY="/home1/irteam/data-vol1/conda/envs/qhflow/bin/python"
CKPT_ROOT="$PROJ/ckpt/QH9"
RESULTS="$PROJ/eval_results"
mkdir -p "$RESULTS"

# Map: split_dir -> (dataset, dataset.split)
declare -A SPLIT_MAP=(
  ["QH9Stable-random"]="QH9Stable random"
  ["QH9Stable-size_ood"]="QH9Stable size_ood"
  ["QH9Dynamic-300k-geometry"]="QH9Dynamic geometry"
  ["QH9Dynamic-300k-mol"]="QH9Dynamic mol"
)

# Model size -> config name
declare -A MODEL_MAP=(
  ["small"]="QHFlow_so2_v5_1_small"
  ["middle"]="QHFlow_so2_v5_1_middle"
  ["large"]="QHFlow_so2_v5_1_large"
  ["extra_large"]="QHFlow_so2_v5_1_extra_large"
)

for split_dir in "${!SPLIT_MAP[@]}"; do
  read -r dataset split <<< "${SPLIT_MAP[$split_dir]}"

  for size in small middle large extra_large; do
    model_cfg="${MODEL_MAP[$size]}"

    # Find checkpoint
    ckpt_dir="$CKPT_ROOT/$split_dir"
    ckpt_file=$(find "$ckpt_dir" -name "*${size}*" -name "*.ckpt" 2>/dev/null | head -1)

    if [ -z "$ckpt_file" ]; then
      echo "SKIP: $split_dir/$size — no checkpoint found"
      continue
    fi

    # Symlink to avoid Hydra = parsing issues
    link="/tmp/qhflow2_eval_${split_dir}_${size}.ckpt"
    ln -sf "$ckpt_file" "$link"

    out_dir="$RESULTS/${split_dir}/${size}"
    mkdir -p "$out_dir"

    echo "========================================"
    echo "EVAL: $split_dir / $size"
    echo "  ckpt: $ckpt_file"
    echo "  output: $out_dir"
    echo "========================================"

    # Run predict
    cd "$PROJ"
    $PY -m qhflow2.experiment.train_qh9 \
      mode=predict \
      dataset=$dataset \
      dataset.split=$split \
      model=$model_cfg \
      wandb.mode=disabled \
      ckpt="$link" \
      hydra.run.dir="$out_dir" \
      2>&1 | tee "$out_dir/eval.log"

    # Extract key metrics
    if [ -f "$out_dir/comparison_report.txt" ]; then
      echo "--- RESULT: $split_dir / $size ---"
      cat "$out_dir/comparison_report.txt"
    fi

    echo ""
  done
done

echo "All evaluations complete. Results in: $RESULTS"
