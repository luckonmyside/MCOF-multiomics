#!/usr/bin/env bash
#SBATCH --partition=x86_64_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mcof_revision

set -Eeuo pipefail

: "${TOOLKIT_ROOT:?TOOLKIT_ROOT is required}"
: "${TASK_MANIFEST:?TASK_MANIFEST is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate mcof

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

cd "$TOOLKIT_ROOT"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the allocated job")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Visible GPU:", torch.cuda.get_device_name(0))
PY

python -m py_compile \
  "$TOOLKIT_ROOT/src/models_v2.py" \
  "$TOOLKIT_ROOT/src/train_utils_v2.py" \
  "$TOOLKIT_ROOT/src/main_MCOF_v2.py" \
  "$TOOLKIT_ROOT/scripts/run_manifest_task.py"

python -u "$TOOLKIT_ROOT/scripts/run_manifest_task.py" \
  --manifest "$TASK_MANIFEST" \
  --task_index "$SLURM_ARRAY_TASK_ID" \
  --toolkit_root "$TOOLKIT_ROOT" \
  --archive_partial
