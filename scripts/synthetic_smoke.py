"""Tiny synthetic CPU training/IG round-trip. These are not study data/results."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def run(cmd,env):
    result=subprocess.run([sys.executable,*map(str,cmd)],cwd=ROOT,env=env,text=True,capture_output=True)
    if result.returncode:
        print(result.stdout[-6000:]);print(result.stderr[-6000:],file=sys.stderr)
        raise RuntimeError('Synthetic check failed: '+str(cmd[0]))
    return result.stdout

def main() -> None:
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONHASHSEED='1')
    rng=np.random.default_rng(314159)
    with tempfile.TemporaryDirectory(prefix='mcof_synthetic_') as tmp:
        base=Path(tmp);data=base/'data';data.mkdir()
        labels=np.tile(np.arange(2),20);rng.shuffle(labels)
        for block,d in enumerate([12,8],1):
            x=rng.normal(size=(len(labels),d)).astype('float32')
            x[:,0]+=labels.astype('float32') # simulated class signal only
            np.savetxt(data/f'{block}_all.csv',x,delimiter=',')
            pd.Series([f'SYNTHETIC_{block}_{i}' for i in range(d)]).to_csv(data/f'{block}_featname.csv',index=False,header=False)
        pd.Series(labels).to_csv(data/'labels_all.csv',index=False,header=False)
        dry=run([ROOT/'scripts/run_main.py','--data-dir',data,'--output-dir',base/'unused','--dry-run','--cpu'],env)
        assert '--model_type mcof_se' in dry and '--random_seed 1' in dry
        assert not (base/'unused').exists()
        output=base/'trained'
        run([ROOT/'code/main_model/main_MCOF_v2.py','--data_dir',data,'--output_dir',output,
             '--model_type','mcof_se','--random_seed','1','--repeats','1','--inner_folds','2',
             '--max_epochs','2','--patience','1','--hidden_dim','32','--conv_channels','8',
             '--batch_size','8','--cpu'],env)
        checkpoint=output/'repeat_1/best_model.pt'
        assert checkpoint.is_file() and (output/'repeat_test_metrics.csv').is_file()
        # Legacy IG uses whichever optional backend is installed; smoke does not certify legacy historical use.
        for name,script in [('frozen',ROOT/'code/main_model/main_biomarker_v2.py'),('revision',ROOT/'code/revision_analysis/src/main_biomarker_v2.py')]:
            ig=base/('ig_'+name)
            run([script,'--checkpoint',checkpoint,'--data_dir',data,'--output_dir',ig,'--top_k','5','--ig_steps','4','--batch_size','20','--cpu'],env)
            f=pd.read_csv(ig/'biomarker_importance_overall.csv')
            assert len(f)==20 and np.isfinite(f['importance']).all()
            cf=pd.read_csv(ig/'biomarker_importance_by_class.csv')
            assert len(cf)==40 and set(cf['class'])=={0,1}
        print('PASS: explicit main launcher dry-run, one tiny synthetic training run, checkpoint save/load, and frozen/revision attribution outputs.')
        print('The synthetic inputs and temporary outputs have no scientific interpretation and are not publication results.')

if __name__=='__main__': main()
