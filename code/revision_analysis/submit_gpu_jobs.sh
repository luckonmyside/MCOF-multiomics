#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash submit_gpu_jobs.sh <workspace> <core|budget|all> [max_parallel=4]" >&2
  exit 2
fi

TOOLKIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(readlink -f "$1")"
PHASE="$2"
MAX_PARALLEL="${3:-4}"

submit_one() {
  local manifest="$1"
  local label="$2"
  local count
  count=$(python - "$manifest" <<'PY'
import pandas as pd, sys
print(len(pd.read_csv(sys.argv[1])))
PY
)
  if [[ "$count" -lt 1 ]]; then
    echo "No tasks in $manifest" >&2
    return 1
  fi
  mkdir -p "$WORKSPACE/logs" "$WORKSPACE/job_ids"
  local upper=$((count - 1))
  local jobid
  jobid=$(sbatch --parsable \
    --array="0-${upper}%${MAX_PARALLEL}" \
    --job-name="mcof_${label}" \
    --output="$WORKSPACE/logs/${label}_%A_%a.out" \
    --error="$WORKSPACE/logs/${label}_%A_%a.err" \
    --export="ALL,TOOLKIT_ROOT=$TOOLKIT_ROOT,TASK_MANIFEST=$manifest" \
    "$TOOLKIT_ROOT/slurm/run_revision_array.slurm")
  echo "$jobid" | tee "$WORKSPACE/job_ids/${label}_job_id.txt"
  echo "Submitted $label array: $jobid ($count tasks; max parallel=$MAX_PARALLEL)"
}

case "$PHASE" in
  core)
    submit_one "$WORKSPACE/manifests/core_tasks.csv" core
    ;;
  budget)
    submit_one "$WORKSPACE/manifests/budget_tasks.csv" budget
    ;;
  all)
    submit_one "$WORKSPACE/manifests/core_tasks.csv" core
    submit_one "$WORKSPACE/manifests/budget_tasks.csv" budget
    ;;
  *)
    echo "Unknown phase: $PHASE" >&2
    exit 2
    ;;
esac
