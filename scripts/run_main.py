"""Release-only launcher for the explicit publication recipe; not an original runner."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a=p.parse_args()
    data=a.data_dir.expanduser().resolve(); out=a.output_dir.expanduser().resolve()
    if not data.is_dir(): p.error(f'Data directory not found: {data}')
    for name in ['1_all.csv','labels_all.csv']:
        if not (data/name).is_file(): p.error(f'Required file missing: {data/name}')
    cfg=json.loads((ROOT/'configs/mcof_se.json').read_text())
    if cfg.get('model_type')!='mcof_se' or cfg.get('random_seed')!=1:
        p.error('The explicit publication recipe must select mcof_se and seed 1.')
    cmd=[sys.executable,str(ROOT/'code/main_model/main_MCOF_v2.py'),
         '--data_dir',str(data),'--output_dir',str(out)]
    for key,value in cfg.items():
        if value is not None: cmd.extend(['--'+key,str(value)])
    if a.cpu: cmd.append('--cpu')
    print(shlex.join(cmd),flush=True)
    if a.dry_run: return
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        p.error('Refusing a nonempty output path. Use a new directory.')
    subprocess.run(cmd,check=True,cwd=str(ROOT))

if __name__=='__main__': main()
