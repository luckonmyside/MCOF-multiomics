#!/usr/bin/env bash
#SBATCH --partition=x86_64_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=mcof_rev_final

set -Eeuo pipefail
: "${TOOLKIT_ROOT:?TOOLKIT_ROOT is required}"
: "${REVISION_WORKSPACE:?REVISION_WORKSPACE is required}"

ROOT="/path/to/mcof-workspace"
MAIN="$ROOT/MCOFv2_runs/formal_main_20260429_191748"
OUT="$REVISION_WORKSPACE/final_summaries"

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate mcof
export PYTHONPATH="$TOOLKIT_ROOT/src:$TOOLKIT_ROOT/scripts:${PYTHONPATH:-}"
mkdir -p "$OUT" "$REVISION_WORKSPACE/status"

python -u "$TOOLKIT_ROOT/scripts/summarize_sensitivity.py" \
  --workspace "$REVISION_WORKSPACE" \
  --formal_main_root "$MAIN" \
  --output_root "$OUT"

touch "$REVISION_WORKSPACE/status/FINAL_SUMMARIES_COMPLETED"
echo "Final summaries completed: $OUT"
