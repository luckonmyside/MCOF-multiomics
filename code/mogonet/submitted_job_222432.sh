#!/usr/bin/env bash
#SBATCH --partition=x86_64_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mogonet
#SBATCH --output=/path/to/mcof-workspace/MCOF_publication_work_20260716/MOGONET_formal_20260723/logs/slurm_%j.out
#SBATCH --error=/path/to/mcof-workspace/MCOF_publication_work_20260716/MOGONET_formal_20260723/logs/slurm_%j.err

set -Eeuo pipefail

ROOT="/path/to/mcof-workspace/MCOF_publication_work_20260716/MOGONET_formal_20260723"
CODE="$ROOT/code"
LOGS="$ROOT/logs"
STATUS="$ROOT/status"
DATA="/path/to/mcof-workspace/MCOF_fixed_v1_run/data"
SPLITS="/path/to/mcof-workspace/MCOF_baselines_20260722/splits"
RESULTS="$ROOT/results/formal_main_20260723"
SMOKE="$ROOT/results/smoke_test_${SLURM_JOB_ID}"

mkdir -p "$LOGS" "$STATUS" "$RESULTS"

RUNNING_FILE="$STATUS/RUNNING_${SLURM_JOB_ID}.txt"
FAILED_FILE="$STATUS/FAILED_${SLURM_JOB_ID}.txt"
COMPLETED_FILE="$STATUS/COMPLETED_${SLURM_JOB_ID}.txt"

cat > "$RUNNING_FILE" <<RUNINFO
job_id=${SLURM_JOB_ID}
status=RUNNING
started=$(date --iso-8601=seconds)
host=$(hostname)
RUNINFO

on_error() {
    rc=$?
    line=${BASH_LINENO[0]:-unknown}

    {
        echo "job_id=${SLURM_JOB_ID}"
        echo "status=FAILED"
        echo "exit_code=${rc}"
        echo "failed_line=${line}"
        echo "failed_at=$(date --iso-8601=seconds)"
        echo "host=$(hostname)"
    } > "$FAILED_FILE"

    rm -f "$RUNNING_FILE"

    echo
    echo "=================================================="
    echo "MOGONET JOB FAILED"
    echo "Exit code: $rc"
    echo "Line: $line"
    echo "See:"
    echo "$LOGS/slurm_${SLURM_JOB_ID}.out"
    echo "$LOGS/slurm_${SLURM_JOB_ID}.err"
    echo "=================================================="

    exit "$rc"
}

trap on_error ERR

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate mcof

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$CODE"

echo "=================================================="
echo "MOGONET FORMAL BENCHMARK"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date --iso-8601=seconds)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo

echo "===== SLURM ENVIRONMENT ====="
env | grep '^SLURM_' | sort || true
echo

echo "===== GPU INFORMATION ====="
nvidia-smi
echo

echo "===== PYTHON ENVIRONMENT ====="
which python
python -V

python - <<'PY'
import sys
import numpy
import pandas
import sklearn
import torch

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

assert sys.version_info[:2] == (3, 9), (
    f"Unexpected Python version: {sys.version}"
)

assert torch.cuda.is_available(), (
    "CUDA is unavailable on the allocated GPU node."
)

print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("GPU count visible:", torch.cuda.device_count())
PY

echo
echo "===== CODE CHECK ====="

for file in \
  run_mogonet_formal.py \
  summarize_mogonet_formal.py \
  models.py
do
    test -s "$file" || {
        echo "Missing or empty code file: $file"
        exit 1
    }
done

python -m py_compile \
  run_mogonet_formal.py \
  summarize_mogonet_formal.py \
  models.py

sha256sum \
  run_mogonet_formal.py \
  summarize_mogonet_formal.py \
  models.py \
  > "$RESULTS/CODE_SHA256SUMS.txt"

echo "Code syntax and checksums OK."

echo
echo "===== DATA AND SPLIT CHECK ====="

for dataset in BRCA STAD ROSMAP SCZ
do
    test -s "$DATA/$dataset/1_all.csv"
    test -s "$DATA/$dataset/2_all.csv"
    test -s "$DATA/$dataset/labels_all.csv"
    test -s "$SPLITS/${dataset}_outer_splits.csv"

    case "$dataset" in
        ROSMAP)
            test -s "$DATA/$dataset/3_all.csv"
            ;;
    esac

    echo "$dataset data files OK"
done

python - <<'PY'
from pathlib import Path
import pandas as pd

data_root = Path(
    "/path/to/mcof-workspace/MCOF_fixed_v1_run/data"
)
split_root = Path(
    "/path/to/mcof-workspace/MCOF_baselines_20260722/splits"
)

expected = {
    "BRCA": (1002, 2, 4),
    "STAD": (217, 2, 3),
    "ROSMAP": (351, 3, 2),
    "SCZ": (104, 2, 2),
}

for dataset, (expected_n, expected_views, expected_classes) in expected.items():
    labels = pd.read_csv(
        data_root / dataset / "labels_all.csv",
        header=None,
    ).iloc[:, 0]

    assert len(labels) == expected_n, (
        dataset,
        len(labels),
        expected_n,
    )
    assert labels.nunique() == expected_classes, (
        dataset,
        labels.nunique(),
        expected_classes,
    )

    for view in range(1, expected_views + 1):
        matrix = pd.read_csv(
            data_root / dataset / f"{view}_all.csv",
            header=None,
        )
        assert len(matrix) == expected_n, (
            dataset,
            view,
            len(matrix),
            expected_n,
        )

    split = pd.read_csv(
        split_root / f"{dataset}_outer_splits.csv"
    )

    assert sorted(split["repeat"].unique()) == [1, 2, 3, 4, 5]

    for repeat in range(1, 6):
        current = split[split["repeat"].eq(repeat)]
        train = current[current["subset"].eq("train")]
        test = current[current["subset"].eq("test")]

        assert len(train) + len(test) == expected_n
        assert not set(train["sample_index"]).intersection(
            set(test["sample_index"])
        )

    print(
        dataset,
        "samples=", expected_n,
        "views=", expected_views,
        "classes=", expected_classes,
        "splits=5 OK",
    )

print("All data and split checks passed.")
PY

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
  --sample_weight_mode original \
  --device cuda

test -s \
  "$SMOKE/BRCA/MOGONET/repeat_1/metrics_test.json"

test -s \
  "$SMOKE/BRCA/MOGONET/repeat_1/predictions_test.csv"

echo "Smoke test completed successfully."

echo
echo "===== FORMAL MOGONET RUN ====="
echo "Completed repeats will be skipped automatically."
echo

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
  --sample_weight_mode original \
  --device cuda

echo
echo "===== RESULT SUMMARIZATION ====="

python summarize_mogonet_formal.py \
  --result_root "$RESULTS"

echo
echo "===== FINAL INTEGRITY CHECK ====="

python - <<'PY' | tee "$RESULTS/integrity_check.log"
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

print("metrics_test.json files:", len(metric_files))
print("predictions_test.csv files:", len(prediction_files))

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

        metric_path = run_dir / "metrics_test.json"
        prediction_path = run_dir / "predictions_test.csv"
        model_dir = run_dir / "models"

        with metric_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        assert metrics["dataset"] == dataset
        assert int(metrics["repeat"]) == repeat
        assert metrics["model"] == "MOGONET"

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

        predictions = pd.read_csv(prediction_path)

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

        num_views = int(metrics["num_views"])
        expected_checkpoints = 2 * num_views + 1
        checkpoint_files = list(model_dir.glob("*.pt"))

        assert len(checkpoint_files) == expected_checkpoints, (
            dataset,
            repeat,
            len(checkpoint_files),
            expected_checkpoints,
        )

        rows.append(metrics)

repeat_table = pd.DataFrame(rows)

assert len(repeat_table) == 20

counts = (
    repeat_table.groupby(["dataset", "model"])
    .size()
)

assert counts.eq(5).all()

required_outputs = [
    root / "MOGONET_repeat_metrics.csv",
    root / "MOGONET_summary_long.csv",
    root / "MOGONET_summary_wide.csv",
    root / "MOGONET_runtime_by_repeat.csv",
]

for path in required_outputs:
    assert path.is_file() and path.stat().st_size > 0, path

print("\nRuns per dataset:")
print(counts.to_string())

print("\nFormal MOGONET summary:")
summary = pd.read_csv(root / "MOGONET_summary_long.csv")
print(
    summary[
        ["dataset", "metric", "mean_sd_3dp"]
    ].to_string(index=False)
)

print("\nMOGONET integrity check passed.")
PY

echo
echo "===== RESULT CHECKSUMS ====="

CHECKSUM_TMP="$RESULTS/SHA256SUMS.tmp"

find "$RESULTS" \
  -type f \
  ! -name 'SHA256SUMS.txt' \
  ! -name 'SHA256SUMS.tmp' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$CHECKSUM_TMP"

mv "$CHECKSUM_TMP" "$RESULTS/SHA256SUMS.txt"

rm -f "$RUNNING_FILE"
rm -f "$FAILED_FILE"

cat > "$COMPLETED_FILE" <<DONE
job_id=${SLURM_JOB_ID}
status=COMPLETED
completed=$(date --iso-8601=seconds)
host=$(hostname)
result_root=${RESULTS}
DONE

echo
echo "=================================================="
echo "ALL MOGONET RUNS COMPLETED SUCCESSFULLY"
echo "Completed: $(date --iso-8601=seconds)"
echo "Result root: $RESULTS"
echo "Status file: $COMPLETED_FILE"
echo "=================================================="
