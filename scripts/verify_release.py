"""New release integrity check; standard library only; does not execute analysis scripts."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]

def main() -> None:
    sums=ROOT/'SHA256SUMS.txt'
    if not sums.is_file(): raise FileNotFoundError('Release SHA256SUMS.txt is missing.')
    checked=0
    for line in sums.read_text().splitlines():
        if not line: continue
        digest, rel=line.split('  ',1)
        path=ROOT/rel
        if not path.is_file(): raise FileNotFoundError(rel)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Checksum differs: '+rel)
        checked+=1
    py_count=json_count=0
    for path in ROOT.rglob('*.py'):
        if any(x in path.parts for x in ['.venv','.git','runs']):continue
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path.relative_to(ROOT)))
        py_count+=1
    for path in ROOT.rglob('*.json'):
        if any(x in path.parts for x in ['.venv','.git','runs']):continue
        json.loads(path.read_text(encoding='utf-8'));json_count+=1
    counts={'BRCA':1002,'STAD':217,'ROSMAP':351,'SCZ':104}
    for ds,n in counts.items():
        with (ROOT/'splits'/f'{ds}_outer_splits.csv').open(newline='') as f:
            reader=csv.DictReader(f)
            if reader.fieldnames!=['repeat','sample_index','subset']:
                raise ValueError(ds+': unexpected split fields, possibly outcome labels.')
            rows=list(reader)
        for rep in range(1,6):
            subset=[r for r in rows if int(r['repeat'])==rep]
            indices=[int(r['sample_index']) for r in subset]
            if len(indices)!=n or len(set(indices))!=n or set(indices)!=set(range(n)):
                raise ValueError(f'{ds} repeat {rep}: not a unique complete partition.')
            if {r['subset'] for r in subset}!={'train','test'}:raise ValueError('Invalid subset values')
    print(f'PASS: {checked} checksums; {py_count} Python syntax checks; {json_count} JSON files; 20 label-free split partitions.')
    print('This is a package-integrity check, not reproduction of the scientific results.')

if __name__=='__main__': main()
