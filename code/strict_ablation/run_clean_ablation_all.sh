#!/usr/bin/env bash
set -euo pipefail

# Run this script on a GPU compute node from the code directory.
# It never overwrites the 2026-04-29 formal results.

BASE="${BASE:-/path/to/mcof-workspace}"
CODE_DIR="${CODE_DIR:-$BASE/MCOF_publication_clean_ablation_20260719}"
DATA_ROOT="${DATA_ROOT:-$BASE/MCOF_fixed_v1_run/data}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$BASE/MCOFv2_runs/formal_ablation_clean_${STAMP}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$RESULT_ROOT"

DATASETS=(BRCA STAD ROSMAP SCZ)
# Override with, for example: GPU_LIST="0 1" bash run_clean_ablation_all.sh
read -r -a GPUS <<< "${GPU_LIST:-0 1 2 3}"
MODELS=(mcof_no_se mcof_no_conv)

if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPU_LIST must contain at least one GPU index." >&2
  exit 2
fi

run_one() {
  local dataset="$1"
  local model="$2"
  local gpu="$3"
  local out="$RESULT_ROOT/$dataset/$model"
  mkdir -p "$out"

  {
    echo "project=MCOFv2"
    echo "stage=formal_clean_ablation"
    echo "dataset=$dataset"
    echo "model=$model"
    echo "date=$(date)"
    echo "host=$(hostname)"
    echo "gpu=$gpu"
    echo "code_dir=$CODE_DIR"
    echo "data_dir=$DATA_ROOT/$dataset"
    echo "out_dir=$out"
    echo "repeats=5"
    echo "inner_folds=5"
    echo "random_seed=1"
    echo "batch_size=64"
    echo "learning_rate=0.001"
    echo "max_epochs=300"
    echo "patience=30"
    echo
    "$PYTHON_BIN" -V
    echo
    nvidia-smi || true
    echo
    sha256sum "$CODE_DIR"/*.py
  } > "$out/manifest.txt"

  echo "[$(date)] START $dataset / $model on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$CODE_DIR/main_MCOF_v2.py" \
    --data_dir "$DATA_ROOT/$dataset" \
    --output_dir "$out" \
    --model_type "$model" \
    --test_size 0.2 \
    --repeats 5 \
    --inner_folds 5 \
    --random_seed 1 \
    --hidden_dim 128 \
    --num_heads 4 \
    --conv_channels 64 \
    --conv_kernel_size 5 \
    --dropout 0.30 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --max_epochs 300 \
    --patience 30 \
    --gradient_clip_norm 5.0 \
    --monitor_metric auto \
    --impute_strategy median \
    --scaler standard \
    --max_missing_rate 1.0 \
    --max_zero_rate 1.0 \
    --selector none \
    2>&1 | tee "$out/run.log"
  echo "[$(date)] DONE  $dataset / $model on GPU $gpu"
}

pids=()
for gi in "${!GPUS[@]}"; do
  gpu="${GPUS[$gi]}"
  (
    for i in "${!DATASETS[@]}"; do
      if (( i % ${#GPUS[@]} != gi )); then
        continue
      fi
      dataset="${DATASETS[$i]}"
      for model in "${MODELS[@]}"; do
        run_one "$dataset" "$model" "$gpu"
      done
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

echo "RESULT_ROOT=$RESULT_ROOT"
if [[ "$status" -ne 0 ]]; then
  echo "At least one job failed. Inspect run.log files." >&2
  exit "$status"
fi

echo "All clean ablation jobs completed successfully."
