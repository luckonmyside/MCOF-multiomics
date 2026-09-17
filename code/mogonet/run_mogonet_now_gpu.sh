#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/path/to/mcof-workspace/MCOF_publication_work_20260716/MOGONET_formal_20260723"
CODE="$ROOT/code"
DATA="/path/to/mcof-workspace/MCOF_fixed_v1_run/data"
SPLITS="/path/to/mcof-workspace/MCOF_baselines_20260722/splits"
RESULTS="$ROOT/results/formal_main_20260723"
SMOKE="$ROOT/results/smoke_test_direct"
LOGS="$ROOT/logs"
STATUS="$ROOT/status"

RUN_TAG="direct_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOGS/${RUN_TAG}.log"
RUNNING_FILE="$STATUS/RUNNING_${RUN_TAG}.txt"
FAILED_FILE="$STATUS/FAILED_${RUN_TAG}.txt"
COMPLETED_FILE="$STATUS/COMPLETED_${RUN_TAG}.txt"

mkdir -p "$RESULTS" "$LOGS" "$STATUS"

exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
    rc=$?
    {
        echo "status=FAILED"
        echo "exit_code=$rc"
        echo "failed_at=$(date --iso-8601=seconds)"
        echo "host=$(hostname)"
        echo "log=$LOG_FILE"
    } > "$FAILED_FILE"

    rm -f "$RUNNING_FILE"

    echo
    echo "================ MOGONET FAILED ================"
    echo "Exit code: $rc"
    echo "Log: $LOG_FILE"
    echo "Status: $FAILED_FILE"
    echo "================================================="
    exit "$rc"
}
trap on_error ERR

{
    echo "status=RUNNING"
    echo "started=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "log=$LOG_FILE"
} > "$RUNNING_FILE"

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate mcof

cd "$CODE"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "================================================="
echo "MOGONET DIRECT FORMAL RUN"
echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Slurm job: ${SLURM_JOB_ID:-manual}"
echo "================================================="

echo
echo "===== GPU CHECK ====="
nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() >= 1, "No visible GPU"

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "===== CODE CHECK ====="
python -m py_compile \
  run_mogonet_formal.py \
  summarize_mogonet_formal.py \
  models.py

echo "Code syntax passed."

echo
echo "===== CLEAN SMOKE TEST ====="
rm -rf "$SMOKE"

python -u run_mogonet_formal.py \
  --data_root "$DATA" \
  --split_root "$SPLITS" \
  --output_root "$SMOKE" \
  --datasets BRCA \
  --repeats 1 \
  --base_seed 1 \
  --num_epoch_pretrain 2 \
  --num_epoch 3 \
  --lr_e_pretrain 0.001 \
  --lr_e 0.0005 \
  --lr_c 0.001 \
  --device cuda

test -s \
  "$SMOKE/BRCA/MOGONET/repeat_1/metrics_test.json"

test -s \
  "$SMOKE/BRCA/MOGONET/repeat_1/predictions_test.csv"

echo "Smoke test passed."

echo
echo "===== FORMAL RUN: 4 DATASETS × 5 REPEATS ====="
echo "Completed repeats will be skipped automatically."

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
  --device cuda

echo
echo "===== SUMMARIZATION ====="

python summarize_mogonet_formal.py \
  --result_root "$RESULTS"

echo
echo "===== FINAL INTEGRITY CHECK ====="

python - <<'PY'
from pathlib import Path
import json

import numpy as np
import pandas as pd

root = Path(
    "/path/to/mcof-workspace/"
    "MCOF_publication_work_20260716/"
    "MOGONET_formal_20260723/"
    "results/formal_main_20260723"
)

datasets = ["BRCA", "STAD", "ROSMAP", "SCZ"]

expected_metrics = {
    "BRCA": [
        "acc",
        "f1_macro",
        "f1_weighted",
        "auc_macro_ovr",
        "auc_weighted_ovr",
    ],
    "STAD": [
        "acc",
        "f1_macro",
        "f1_weighted",
        "auc_macro_ovr",
        "auc_weighted_ovr",
    ],
    "ROSMAP": ["acc", "f1", "auc", "mcc"],
    "SCZ": ["acc", "f1", "auc", "mcc"],
}

metric_files = sorted(
    root.glob("*/MOGONET/repeat_*/metrics_test.json")
)
prediction_files = sorted(
    root.glob("*/MOGONET/repeat_*/predictions_test.csv")
)

print("Metric files:", len(metric_files))
print("Prediction files:", len(prediction_files))

assert len(metric_files) == 20, (
    f"Expected 20 metric files, found {len(metric_files)}"
)
assert len(prediction_files) == 20, (
    f"Expected 20 prediction files, found {len(prediction_files)}"
)

rows = []

for dataset in datasets:
    for repeat in range(1, 6):
        run_dir = (
            root
            / dataset
            / "MOGONET"
            / f"repeat_{repeat}"
        )

        with (run_dir / "metrics_test.json").open(
            "r",
            encoding="utf-8",
        ) as handle:
            metrics = json.load(handle)

        assert metrics["dataset"] == dataset
        assert metrics["model"] == "MOGONET"
        assert int(metrics["repeat"]) == repeat

        for metric in expected_metrics[dataset]:
            value = float(metrics[metric])
            assert np.isfinite(value), (
                dataset,
                repeat,
                metric,
                value,
            )
            assert -1.0 <= value <= 1.0, (
                dataset,
                repeat,
                metric,
                value,
            )

        predictions = pd.read_csv(
            run_dir / "predictions_test.csv"
        )

        probability_columns = [
            column
            for column in predictions.columns
            if column.startswith("prob_class_")
        ]

        assert probability_columns, (
            dataset,
            repeat,
            "No probability columns",
        )

        probabilities = predictions[
            probability_columns
        ].to_numpy(dtype=float)

        assert np.isfinite(probabilities).all()
        assert np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-5,
        )

        rows.append(metrics)

table = pd.DataFrame(rows)

counts = table.groupby(["dataset", "model"]).size()

print("\nRuns per dataset:")
print(counts.to_string())

assert len(table) == 20
assert counts.eq(5).all()

required = [
    root / "MOGONET_repeat_metrics.csv",
    root / "MOGONET_summary_long.csv",
    root / "MOGONET_summary_wide.csv",
    root / "MOGONET_runtime_by_repeat.csv",
]

for path in required:
    assert path.is_file() and path.stat().st_size > 0, path

print("\nFormal MOGONET summary:")
summary = pd.read_csv(root / "MOGONET_summary_long.csv")
print(
    summary[
        ["dataset", "metric", "mean_sd_3dp"]
    ].to_string(index=False)
)

print("\nMOGONET integrity check passed.")
PY

find "$RESULTS" \
  -type f \
  ! -name 'SHA256SUMS.txt' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$RESULTS/SHA256SUMS.txt"

rm -f "$RUNNING_FILE"
rm -f "$FAILED_FILE"

{
    echo "status=COMPLETED"
    echo "completed=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "result_root=$RESULTS"
    echo "log=$LOG_FILE"
} > "$COMPLETED_FILE"

echo
echo "================================================="
echo "ALL 20 FORMAL MOGONET RUNS COMPLETED"
echo "Result root: $RESULTS"
echo "Status: $COMPLETED_FILE"
echo "Log: $LOG_FILE"
echo "================================================="
