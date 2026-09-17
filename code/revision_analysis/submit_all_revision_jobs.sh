#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash submit_all_revision_jobs.sh <workspace> [max_parallel_core=4]" >&2
  exit 2
fi

TOOLKIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(readlink -f "$1")"
MAX_PARALLEL="${2:-4}"
ROOT="/path/to/mcof-workspace"
DATA="$ROOT/MCOF_fixed_v1_run/data"
SPLITS="$ROOT/MCOF_baselines_20260722/splits"

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate mcof
export PYTHONPATH="$TOOLKIT_ROOT/src:$TOOLKIT_ROOT/scripts:${PYTHONPATH:-}"

mkdir -p "$WORKSPACE/logs" "$WORKSPACE/job_ids" "$WORKSPACE/status"

python "$TOOLKIT_ROOT/scripts/prepare_revision_variants.py" \
  --data_root "$DATA" \
  --split_root "$SPLITS" \
  --workspace "$WORKSPACE" \
  --force

count_rows() {
  python - "$1" <<'PY'
import pandas as pd, sys
print(len(pd.read_csv(sys.argv[1])))
PY
}

CORE_MANIFEST="$WORKSPACE/manifests/core_tasks.csv"
BUDGET_MANIFEST="$WORKSPACE/manifests/budget_tasks.csv"
CORE_N=$(count_rows "$CORE_MANIFEST")
BUDGET_N=$(count_rows "$BUDGET_MANIFEST")

CORE_JOB=$(sbatch --parsable \
  --array="0-$((CORE_N-1))%${MAX_PARALLEL}" \
  --output="$WORKSPACE/logs/core_%A_%a.out" \
  --error="$WORKSPACE/logs/core_%A_%a.err" \
  --export="ALL,TOOLKIT_ROOT=$TOOLKIT_ROOT,TASK_MANIFEST=$CORE_MANIFEST" \
  "$TOOLKIT_ROOT/slurm/run_revision_array.slurm")
echo "$CORE_JOB" | tee "$WORKSPACE/job_ids/core_job_id.txt"

POST_JOB=$(sbatch --parsable \
  --output="$WORKSPACE/logs/post_%j.out" \
  --error="$WORKSPACE/logs/post_%j.err" \
  --export="ALL,TOOLKIT_ROOT=$TOOLKIT_ROOT,REVISION_WORKSPACE=$WORKSPACE" \
  "$TOOLKIT_ROOT/slurm/run_postprocessing.slurm")
echo "$POST_JOB" | tee "$WORKSPACE/job_ids/postprocessing_job_id.txt"

BUDGET_JOB=$(sbatch --parsable \
  --dependency="afterok:$CORE_JOB" \
  --array="0-$((BUDGET_N-1))%${MAX_PARALLEL}" \
  --output="$WORKSPACE/logs/budget_%A_%a.out" \
  --error="$WORKSPACE/logs/budget_%A_%a.err" \
  --export="ALL,TOOLKIT_ROOT=$TOOLKIT_ROOT,TASK_MANIFEST=$BUDGET_MANIFEST" \
  "$TOOLKIT_ROOT/slurm/run_revision_array.slurm")
echo "$BUDGET_JOB" | tee "$WORKSPACE/job_ids/budget_job_id.txt"

FINAL_JOB=$(sbatch --parsable \
  --dependency="afterok:$BUDGET_JOB" \
  --output="$WORKSPACE/logs/final_%j.out" \
  --error="$WORKSPACE/logs/final_%j.err" \
  --export="ALL,TOOLKIT_ROOT=$TOOLKIT_ROOT,REVISION_WORKSPACE=$WORKSPACE" \
  "$TOOLKIT_ROOT/slurm/run_finalize.slurm")
echo "$FINAL_JOB" | tee "$WORKSPACE/job_ids/final_job_id.txt"

cat <<EOF
============================================================
Submitted MCOF revision jobs
Core sensitivity array:  $CORE_JOB ($CORE_N tasks)
Postprocessing job:       $POST_JOB
Feature-budget array:     $BUDGET_JOB ($BUDGET_N tasks; after core)
Final summary job:        $FINAL_JOB (after feature-budget)
Workspace:                $WORKSPACE
============================================================
Monitor with:
  squeue -u "$USER"
  tail -f "$WORKSPACE/logs/core_${CORE_JOB}_0.out"
EOF
