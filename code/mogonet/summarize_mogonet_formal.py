from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ORDER=["BRCA","STAD","ROSMAP","SCZ"]
METRICS={
 "BRCA":["acc","f1_macro","f1_weighted","auc_macro_ovr","auc_weighted_ovr"],
 "STAD":["acc","f1_macro","f1_weighted","auc_macro_ovr","auc_weighted_ovr"],
 "ROSMAP":["acc","f1","auc","mcc"],
 "SCZ":["acc","f1","auc","mcc"],
}
p=argparse.ArgumentParser(); p.add_argument("--result_root",required=True); a=p.parse_args()
root=Path(a.result_root); rows=[]
for ds in ORDER:
    for r in range(1,6):
        f=root/ds/"MOGONET"/f"repeat_{r}"/"metrics_test.json"
        if not f.is_file(): raise FileNotFoundError(f)
        rows.append(json.loads(f.read_text()))
rep=pd.DataFrame(rows)
assert len(rep)==20 and rep.groupby(["dataset","model"]).size().eq(5).all()
rep.to_csv(root/"MOGONET_repeat_metrics.csv",index=False)
out=[]
for ds in ORDER:
    g=rep[rep.dataset==ds]
    for m in METRICS[ds]:
        x=pd.to_numeric(g[m],errors="coerce")
        assert x.notna().all() and np.isfinite(x).all()
        out.append({"dataset":ds,"model":"MOGONET","metric":m,"n_repeats":5,
                    "mean":x.mean(),"std":x.std(ddof=1),
                    "mean_sd_3dp":f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"})
long=pd.DataFrame(out); long.to_csv(root/"MOGONET_summary_long.csv",index=False)
wide=long.pivot(index=["dataset","model"],columns="metric",values="mean_sd_3dp").reset_index()
wide.columns.name=None; wide.to_csv(root/"MOGONET_summary_wide.csv",index=False)
rep[["dataset","repeat","preprocess_seconds","graph_seconds","train_seconds","predict_seconds","total_seconds"]].to_csv(
    root/"MOGONET_runtime_by_repeat.csv",index=False)
print(long[["dataset","metric","mean_sd_3dp"]].to_string(index=False))
print("Saved summaries under",root)
