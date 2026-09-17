#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash submit_all_revision_jobs_noarray.sh <workspace> [max_parallel=4]" >&2
  exit 2
fi

TOOLKIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(readlink -f "$1")"
MAX_PARALLEL="${2:-4}"

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: max_parallel must be a positive integer; got: $MAX_PARALLEL" >&2
  exit 2
fi

ROOT="/path/to/mcof-workspace"
DATA="$ROOT/MCOF_fixed_v1_run/data"
SPLITS="$ROOT/MCOF_baselines_20260722/splits"

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate mcof

export PYTHONPATH="$TOOLKIT_ROOT/src:$TOOLKIT_ROOT/scripts:${PYTHONPATH:-}"

mkdir -p "$WORKSPACE/logs" "$WORKSPACE/job_ids" "$WORKSPACE/status"

SUBMISSION_MAP="$WORKSPACE/job_ids/noarray_submission.tsv"
if [[ -s "$SUBMISSION_MAP" ]]; then
  echo "ERROR: $SUBMISSION_MAP already exists and is nonempty." >&2
  echo "This safety check prevents accidental duplicate submission." >&2
  echo "Inspect the listed jobs with: squeue -u \"$USER\"" >&2
  exit 3
fi

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

join_by_colon() {
  local IFS=:
  echo "$*"
}

CORE_MANIFEST="$WORKSPACE/manifests/core_tasks.csv"
BUDGET_MANIFEST="$WORKSPACE/manifests/budget_tasks.csv"
CORE_N="$(count_rows "$CORE_MANIFEST")"
BUDGET_N="$(count_rows "$BUDGET_MANIFEST")"

if (( CORE_N < 1 || BUDGET_N < 1 )); then
  echo "ERROR: empty task manifest: core=$CORE_N budget=$BUDGET_N" >&2
  exit 4
fi

printf "phase\ttask_index\tjob_id\tdependency\n" > "$SUBMISSION_MAP"

declare -a CORE_IDS=()
declare -a BUDGET_IDS=()

on_error() {
  local code=$?
  echo
  echo "Submission stopped with exit code $code." >&2
  echo "Jobs submitted before the failure are recorded in:" >&2
  echo "  $SUBMISSION_MAP" >&2
  echo "Do not rerun blindly; inspect with: squeue -u \"$USER\"" >&2
  exit "$code"
}
trap on_error ERR

echo "Submitting $CORE_N core jobs without Slurm arrays (maximum $MAX_PARALLEL concurrent chains)..."

for ((i=0; i<CORE_N; i++)); do
  dependency=""
  sbatch_args=(
    sbatch --parsable
    --job-name="mcof_core_${i}"
    --output="$WORKSPACE/logs/core_${i}_%j.out"
    --error="$WORKSPACE/logs/core_${i}_%j.err"
    --export="ALL,TOOLKIT_ROOT=$TOOLKIT_ROOT,TASK_MANIFEST=$CORE_MANIFEST,TASK_INDEX=$i"
  )

  # Four dependency lanes preserve the requested concurrency limit without arrays.
  if (( i >= MAX_PARALLEL )); then
    dependency="${CORE_IDS[$((i-MAX_PARALLEL))]}"
    sbatch_args+=(--dependency="afterok:${dependency}")
  fi

  job_id="$("${sbatch_args[@]}" "$TOOLKIT_ROOT/slurm/run_revision_single.slurm")"
  CORE_IDS[$i]="$job_id"
  printf "core\t%s\t%s\t%s\n" "$i" "$job_id" "$dependency" >> "$SUBMISSION_MAP"
  echo "  core[$i] -> $job_id${dependency:+ (afterok:$dependency)}"
done

printf "%s\n" "${CORE_IDS[@]}" > "$WORKSPACE/job_ids/core_job_ids.txt"

echo "Submitting independent postprocessing job..."
POST_JOB="$(sbatch --parsable \
  --job-name="mcof_post" \
  --output="$WORKSPACE/logs/post_%j.out" \
  --error="$WORKSPACE/logs/post_%j.err" \
  --export="ALL,TOOLKIT_ROOT=$TOOLKIT_ROOT,REVISION_WORKSPACE=$WORKSPACE" \
  "$TOOLKIT_ROOT/slurm/run_postprocessing.slurm")"
printf "postprocessing\tNA\t%s\t\n" "$POST_JOB" >> "$SUBMISSION_MAP"
printf "%s\n" "$POST_JOB" > "$WORKSPACE/job_ids/postprocessing_job_id.txt"
echo "  postprocessing -> $POST_JOB"

CORE_DEP="$(join_by_colon "${CORE_IDS[@]}")"

echo "Submitting $BUDGET_N feature-budget jobs without arrays..."
for ((i=0; i<BUDGET_N; i++)); do
  dependency="$CORE_DEP"

  # After all core jobs finish, use the same dependency-lane strategy for budget jobs.
  if (( i >= MAX_PARALLEL )); then
    dependency="${dependency}:${BUDGET_IDS[$((i-MAX_PARALLEL))]}"
  fi

  job_id="$(sbatch --parsable \
    --job-name="mcof_budget_${i}" \
    --dependency="afterok:${dependency}" \
    --output="$WORKSPACE/logs/budget_${i}_%j.out" \
    --error="$WORKSPACE/logs/budget_${i}_%j.err" \
    --export="ALL,TOOLKIT_ROOT=$TOOLKIT_ROOT,TASK_MANIFEST=$BUDGET_MANIFEST,TASK_INDEX=$i" \
    "$TOOLKIT_ROOT/slurm/run_revision_single.slurm")"

  BUDGET_IDS[$i]="$job_id"
  printf "budget\t%s\t%s\t%s\n" "$i" "$job_id" "$dependency" >> "$SUBMISSION_MAP"
  echo "  budget[$i] -> $job_id (after all core jobs${i:+; dependency lane enforced})"
done

printf "%s\n" "${BUDGET_IDS[@]}" > "$WORKSPACE/job_ids/budget_job_ids.txt"

BUDGET_DEP="$(join_by_colon "${BUDGET_IDS[@]}")"
FINAL_DEP="${BUDGET_DEP}:${POST_JOB}"

echo "Submitting final-summary job..."
FINAL_JOB="$(sbatch --parsable \
  --job-name="mcof_final" \
  --dependency="afterok:${FINAL_DEP}" \
  --output="$WORKSPACE/logs/final_%j.out" \
  --error="$WORKSPACE/logs/final_%j.err" \
  --export="ALL,TOOLKIT_ROOT=$TOOLKIT_ROOT,REVISION_WORKSPACE=$WORKSPACE" \
  "$TOOLKIT_ROOT/slurm/run_finalize.slurm")"
printf "final\tNA\t%s\t%s\n" "$FINAL_JOB" "$FINAL_DEP" >> "$SUBMISSION_MAP"
printf "%s\n" "$FINAL_JOB" > "$WORKSPACE/job_ids/final_job_id.txt"

trap - ERR

cat <<EOF

============================================================
Submitted MCOF revision jobs without Slurm arrays
Core jobs:               ${#CORE_IDS[@]}
Postprocessing job:      $POST_JOB
Feature-budget jobs:     ${#BUDGET_IDS[@]} (after all core jobs)
Final summary job:       $FINAL_JOB (after budget + postprocessing)
Maximum concurrent lane: $MAX_PARALLEL
Workspace:               $WORKSPACE
Submission map:          $SUBMISSION_MAP
============================================================

Monitor with:
  squeue -u "$USER"
  bash "$TOOLKIT_ROOT/check_revision_status.sh" "$WORKSPACE"

View the first core log after it starts:
  tail -f "$WORKSPACE/logs/core_0_${CORE_IDS[0]}.out"
EOF
