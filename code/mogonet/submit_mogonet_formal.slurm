#!/usr/bin/env bash
#SBATCH -p x86_64_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=mogonet
#SBATCH --output=/path/to/mcof-workspace/MCOF_publication_work_20260716/MOGONET_formal_20260723/logs/slurm_%j.out
#SBATCH --error=/path/to/mcof-workspace/MCOF_publication_work_20260716/MOGONET_formal_20260723/logs/slurm_%j.err

set -euo pipefail

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate mcof

ROOT=/path/to/mcof-workspace/MCOF_publication_work_20260716/MOGONET_formal_20260723
CODE="$ROOT/code"

cd "$CODE"

echo "===== JOB INFORMATION ====="
date
hostname
nvidia-smi
python -V

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
assert torch.cuda.is_available(), "CUDA is unavailable on the allocated node"
PY

echo "===== SMOKE TEST ====="

python -u run_mogonet_formal.py \
  --data_root /path/to/mcof-workspace/MCOF_fixed_v1_run/data \
  --split_root /path/to/mcof-workspace/MCOF_baselines_20260722/splits \
  --output_root "$ROOT/results/smoke_test" \
  --datasets BRCA \
  --repeats 1 \
  --num_epoch_pretrain 2 \
  --num_epoch 3 \
  --device cuda

echo "===== FORMAL RUN ====="

CUDA_VISIBLE_DEVICES=0 \
bash run_mogonet_formal_all.sh

echo "===== JOB COMPLETED ====="
date
