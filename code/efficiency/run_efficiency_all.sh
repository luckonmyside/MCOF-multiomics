#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/path/to/mcof-workspace/MCOF_publication_work_20260716"
BENCH="$ROOT/efficiency_benchmark_20260724"
MCOF_CODE="$ROOT"
MCOF_DATA="/path/to/mcof-workspace/MCOF_fixed_v1_run/data"
SPLITS="/path/to/mcof-workspace/MCOF_baselines_20260722/splits"
MCOF_CHECKPOINTS="/path/to/mcof-workspace/MCOFv2_runs/formal_main_20260429_191748"

MOGONET_ROOT="$ROOT/MOGONET_formal_20260723"
MOGONET_CODE="$MOGONET_ROOT/code"
MOGONET_RESULTS="$MOGONET_ROOT/results/formal_main_20260723"

OUTPUT="$BENCH/results"
LOGS="$BENCH/logs"
mkdir -p "$OUTPUT" "$LOGS"

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate mcof

cd "$BENCH"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
PY

python -m py_compile \
  benchmark_mcof_efficiency.py \
  benchmark_mogonet_efficiency.py \
  combine_efficiency_results.py

python -u benchmark_mcof_efficiency.py \
  --mcof_code_root "$MCOF_CODE" \
  --data_root "$MCOF_DATA" \
  --split_root "$SPLITS" \
  --checkpoint_root "$MCOF_CHECKPOINTS" \
  --output_root "$OUTPUT" \
  --batch_size 64 \
  --warmup 10 \
  --timing_repeats 100 \
  2>&1 | tee "$LOGS/mcof_efficiency.log"

python -u benchmark_mogonet_efficiency.py \
  --mogonet_code_root "$MOGONET_CODE" \
  --data_root "$MCOF_DATA" \
  --split_root "$SPLITS" \
  --result_root "$MOGONET_RESULTS" \
  --output_root "$OUTPUT" \
  --warmup 10 \
  --timing_repeats 100 \
  2>&1 | tee "$LOGS/mogonet_efficiency.log"

python combine_efficiency_results.py \
  --mcof_file "$OUTPUT/MCOF_efficiency_by_repeat.csv" \
  --mogonet_file "$OUTPUT/MOGONET_efficiency_by_repeat.csv" \
  --output_root "$OUTPUT" \
  2>&1 | tee "$LOGS/combine_efficiency.log"

echo
echo "Efficiency benchmark completed."
echo "Results: $OUTPUT"
