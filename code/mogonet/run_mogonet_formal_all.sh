#!/usr/bin/env bash
set -euo pipefail
ROOT="/path/to/mcof-workspace/MCOF_publication_work_20260716/MOGONET_formal_20260723"
CODE="$ROOT/code"
DATA="/path/to/mcof-workspace/MCOF_fixed_v1_run/data"
SPLITS="/path/to/mcof-workspace/MCOF_baselines_20260722/splits"
RESULTS="$ROOT/results/formal_main_20260723"
LOGS="$ROOT/logs"
mkdir -p "$RESULTS" "$LOGS"
cd "$CODE"
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate mcof
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
python -u run_mogonet_formal.py \
  --data_root "$DATA" \
  --split_root "$SPLITS" \
  --output_root "$RESULTS" \
  --datasets BRCA STAD ROSMAP SCZ \
  --repeats 1 2 3 4 5 \
  --base_seed 1 \
  --num_epoch_pretrain 500 \
  --num_epoch 2500 \
  --lr_e_pretrain 0.001 \
  --lr_e 0.0005 \
  --lr_c 0.001 \
  --device cuda \
  2>&1 | tee "$LOGS/formal_main_20260723.log"
python summarize_mogonet_formal.py --result_root "$RESULTS" \
  2>&1 | tee "$LOGS/summary_20260723.log"
