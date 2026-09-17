#!/usr/bin/env bash
#SBATCH --partition=x86_64_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mcof_rev_post

set -Eeuo pipefail

: "${TOOLKIT_ROOT:?TOOLKIT_ROOT is required}"
: "${REVISION_WORKSPACE:?REVISION_WORKSPACE is required}"

ROOT="/path/to/mcof-workspace"
WORK="$ROOT/MCOF_publication_work_20260716"
DATA="$ROOT/MCOF_fixed_v1_run/data"
SPLITS="$ROOT/MCOF_baselines_20260722/splits"
MAIN="$ROOT/MCOFv2_runs/formal_main_20260429_191748"
ML="$ROOT/MCOF_baselines_20260722/ML"
DIABLO="$ROOT/MCOF_baselines_20260722/DIABLO_final_completed_20260723"
MOGONET="$WORK/MOGONET_formal_20260723/results/formal_main_20260723"
IG="$ROOT/MCOFv2_runs/biomarker_IG_20260720_fixed2"
ALL7="$ROOT/MCOF_baselines_20260722/combined_final_with_MOGONET_20260724/all_7_models_repeat_metrics.csv"
OUT="$REVISION_WORKSPACE/postprocessing"

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate mcof

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$TOOLKIT_ROOT/src:$TOOLKIT_ROOT/scripts:${PYTHONPATH:-}"

mkdir -p "$OUT" "$REVISION_WORKSPACE/status"
cd "$TOOLKIT_ROOT"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
PY

python -m py_compile "$TOOLKIT_ROOT"/scripts/*.py "$TOOLKIT_ROOT"/src/*.py

python -u "$TOOLKIT_ROOT/scripts/export_per_class_metrics.py" \
  --toolkit_root "$TOOLKIT_ROOT" \
  --data_root "$DATA" \
  --split_root "$SPLITS" \
  --mcof_main_root "$MAIN" \
  --ml_root "$ML" \
  --diablo_root "$DIABLO" \
  --mogonet_root "$MOGONET" \
  --output_root "$OUT/per_class"

python -u "$TOOLKIT_ROOT/scripts/batch_independence_test.py" \
  --toolkit_root "$TOOLKIT_ROOT" \
  --data_root "$DATA" \
  --split_root "$SPLITS" \
  --mcof_main_root "$MAIN" \
  --output_root "$OUT/batch_independence" \
  --tolerance 1e-4

python -u "$TOOLKIT_ROOT/scripts/reanalyze_ig_stability.py" \
  --ig_root "$IG" \
  --data_root "$DATA" \
  --output_root "$OUT/ig_stability"

python -u "$TOOLKIT_ROOT/scripts/paired_statistics.py" \
  --repeat_metrics_csv "$ALL7" \
  --output_root "$OUT/statistics"

python -u "$TOOLKIT_ROOT/scripts/ig_convergence_check.py" \
  --toolkit_root "$TOOLKIT_ROOT" \
  --data_root "$DATA" \
  --split_root "$SPLITS" \
  --mcof_main_root "$MAIN" \
  --output_root "$OUT/ig_convergence" \
  --steps 25 50 100 \
  --samples_per_class 8 \
  --batch_size 8 \
  --top_k 100

python -u "$TOOLKIT_ROOT/scripts/discover_full_subtype_data.py" \
  --roots "$ROOT" "/path/to/historical-data" \
  --output_root "$OUT/full_subtype_discovery"

touch "$REVISION_WORKSPACE/status/POSTPROCESSING_COMPLETED"
echo "Postprocessing completed: $OUT"
