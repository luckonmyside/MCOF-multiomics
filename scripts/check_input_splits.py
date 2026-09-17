"""Check split membership against authorized local labels; no participant-identity validation."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT=Path(__file__).resolve().parents[1]

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    a=p.parse_args()
    for ds in ['BRCA','STAD','ROSMAP','SCZ']:
        f=a.data_root.expanduser()/ds/'labels_all.csv'
        raw=pd.read_csv(f,header=None).iloc[:,0]
        labels,_=pd.factorize(raw,sort=True)
        archived=pd.read_csv(ROOT/'splits'/f'{ds}_outer_splits.csv')
        splitter=StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=1)
        for n,(tr,te) in enumerate(splitter.split(np.zeros_like(labels),labels),1):
            block=archived[archived['repeat']==n]
            for subset,idx in [('train',tr),('test',te)]:
                observed=block.loc[block['subset']==subset,'sample_index'].astype(int).tolist()
                if len(observed)!=len(idx) or set(observed)!=set(idx.tolist()):
                    raise ValueError(f'{ds} repeat {n} {subset}: split membership differs; do not overwrite the stored splits.')
        print(ds+': all five split memberships match (not a patient-identity check).')

if __name__=='__main__': main()
