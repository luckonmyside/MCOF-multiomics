#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash check_revision_status.sh <workspace>" >&2
  exit 2
fi

WORKSPACE="$(readlink -f "$1")"

echo "===== SLURM QUEUE ====="
squeue -u "$USER" || true

echo
echo "===== JOB IDS ====="
find "$WORKSPACE/job_ids" -maxdepth 1 -type f -print -exec cat {} \; 2>/dev/null || true

echo
echo "===== TASK STATUS FILES ====="
python - "$WORKSPACE" <<'PY'
from pathlib import Path
import json, sys
root=Path(sys.argv[1])
files=sorted((root/'status').glob('*.json'))
counts={}
for p in files:
    try: status=json.loads(p.read_text()).get('status','UNKNOWN')
    except Exception: status='UNREADABLE'
    counts[status]=counts.get(status,0)+1
print('Status counts:', counts)
for p in files:
    try:
        data=json.loads(p.read_text())
        if data.get('status')=='FAILED': print('FAILED:',p,data.get('error'))
    except Exception: pass
PY

echo
echo "===== RECENT ERRORS ====="
find "$WORKSPACE/logs" -type f -name '*.err' -size +0c -print -exec tail -n 30 {} \; 2>/dev/null || true

echo
echo "===== COMPLETION MARKERS ====="
find "$WORKSPACE/status" -maxdepth 1 -type f -name '*COMPLETED*' -print 2>/dev/null || true
