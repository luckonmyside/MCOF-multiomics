from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch

from common import ensure_dir, read_fixed_splits, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one sensitivity-analysis task from a CSV manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task_index", required=True, type=int)
    parser.add_argument("--toolkit_root", required=True)
    parser.add_argument("--archive_partial", action="store_true", default=True)
    return parser.parse_args()


def validate_completed_output(output_dir: Path, expected_repeats: int = 5) -> Dict[str, Any]:
    metrics_file = output_dir / "repeat_test_metrics.csv"
    if not metrics_file.is_file():
        return {"complete": False, "reason": f"missing {metrics_file}"}
    frame = pd.read_csv(metrics_file)
    if len(frame) != expected_repeats:
        return {"complete": False, "reason": f"{metrics_file} has {len(frame)} rows"}
    repeats = set(pd.to_numeric(frame["repeat"], errors="raise").astype(int))
    if repeats != set(range(1, expected_repeats + 1)):
        return {"complete": False, "reason": f"unexpected repeat identifiers {sorted(repeats)}"}

    missing = []
    for repeat in range(1, expected_repeats + 1):
        repeat_dir = output_dir / f"repeat_{repeat}"
        for name in [
            "best_model.pt",
            "predictions_test.csv",
            "metrics_test.json",
            "inner_cv_metrics.csv",
            "inner_cv_summary.json",
        ]:
            path = repeat_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        return {"complete": False, "reason": f"missing/empty files: {missing[:5]}"}
    return {"complete": True, "reason": "all expected outputs are present"}


def archive_partial_output(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    completion = validate_completed_output(output_dir)
    if completion["complete"]:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = output_dir.with_name(output_dir.name + f"_partial_{timestamp}")
    if archive.exists():
        raise FileExistsError(archive)
    shutil.move(str(output_dir), str(archive))
    return archive


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    toolkit_root = Path(args.toolkit_root).resolve()
    frame = pd.read_csv(manifest)
    if args.task_index < 0 or args.task_index >= len(frame):
        raise IndexError(f"task_index {args.task_index} is outside 0..{len(frame)-1}")
    row = frame.iloc[args.task_index].to_dict()

    output_dir = Path(str(row["output_dir"]))
    workspace = manifest.resolve().parents[1]
    status_dir = ensure_dir(workspace / "status")
    task_label = f"{row['dataset']}__{row['variant']}"
    status_path = status_dir / f"{task_label}.json"

    existing = validate_completed_output(output_dir)
    if existing["complete"]:
        payload = {
            "task": row,
            "status": "SKIPPED_ALREADY_COMPLETE",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "validation": existing,
        }
        write_json(payload, status_path)
        print(f"SKIP {task_label}: output already complete")
        return

    archived = archive_partial_output(output_dir) if args.archive_partial else None
    ensure_dir(output_dir)

    data_dir = Path(str(row["data_dir"]))
    labels_file = data_dir / "labels_all.csv"
    if not labels_file.is_file():
        raise FileNotFoundError(labels_file)
    n_samples = len(pd.read_csv(labels_file, header=None))
    read_fixed_splits(row["split_file"], n_samples=n_samples, repeats=5)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This task must run inside a Slurm GPU allocation.")

    command = [
        sys.executable,
        str(toolkit_root / "src" / "main_MCOF_v2.py"),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(output_dir),
        "--model_type",
        "mcof_se",
        "--split_file",
        str(row["split_file"]),
        "--repeats",
        "5",
        "--inner_folds",
        "5",
        "--random_seed",
        "1",
        "--hidden_dim",
        "128",
        "--conv_channels",
        "64",
        "--conv_kernel_size",
        "5",
        "--dropout",
        "0.30",
        "--batch_size",
        "64",
        "--learning_rate",
        "0.001",
        "--weight_decay",
        "0.0001",
        "--max_epochs",
        "300",
        "--patience",
        "30",
        "--gradient_clip_norm",
        "5.0",
        "--monitor_metric",
        "auto",
        "--impute_strategy",
        "median",
        "--scaler",
        "standard",
        "--max_missing_rate",
        "1.0",
        "--max_zero_rate",
        "1.0",
    ]

    select_k = row.get("select_k")
    if select_k is not None and not pd.isna(select_k) and str(select_k).strip() != "":
        command.extend(["--select_k", str(int(float(select_k))), "--selector", str(row["selector"])])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(toolkit_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    env.setdefault("OPENBLAS_NUM_THREADS", "8")

    started = time.time()
    task_metadata = {
        "task": row,
        "command": command,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "archived_partial_output": str(archived) if archived else None,
    }
    write_json(task_metadata, output_dir / "revision_task_command.json")
    write_json({**task_metadata, "status": "RUNNING"}, status_path)

    print("=" * 80)
    print(f"TASK {args.task_index}: {task_label}")
    print("COMMAND:", " ".join(command))
    print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 80)

    try:
        subprocess.run(command, check=True, cwd=str(toolkit_root), env=env)
        validation = validate_completed_output(output_dir)
        if not validation["complete"]:
            raise RuntimeError(f"Training returned success but output validation failed: {validation}")
        payload = {
            **task_metadata,
            "status": "COMPLETED",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - started,
            "validation": validation,
        }
        write_json(payload, status_path)
        (output_dir / "SUCCESS").write_text("completed\n", encoding="utf-8")
        print(f"COMPLETED {task_label} in {(time.time()-started)/60:.2f} minutes")
    except Exception as error:
        payload = {
            **task_metadata,
            "status": "FAILED",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - started,
            "error": repr(error),
        }
        write_json(payload, status_path)
        raise


if __name__ == "__main__":
    main()
