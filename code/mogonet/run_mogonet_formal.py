from __future__ import annotations

import argparse, gc, json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, roc_auc_score
)
from models import init_model_dict, init_optim

CONFIG = {
    "BRCA": {"adj": 10, "hidden": [400, 400, 200]},
    "STAD": {"adj": 10, "hidden": [400, 400, 200]},
    "ROSMAP": {"adj": 2, "hidden": [200, 200, 100]},
    "SCZ": {"adj": 2, "hidden": [200, 200, 100]},
}
ORDER = ["BRCA", "STAD", "ROSMAP", "SCZ"]

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--split_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--datasets", nargs="+", default=ORDER, choices=ORDER)
    p.add_argument("--repeats", nargs="+", type=int, default=[1,2,3,4,5])
    p.add_argument("--base_seed", type=int, default=1)
    p.add_argument("--num_epoch_pretrain", type=int, default=500)
    p.add_argument("--num_epoch", type=int, default=2500)
    p.add_argument("--lr_e_pretrain", type=float, default=1e-3)
    p.add_argument("--lr_e", type=float, default=5e-4)
    p.add_argument("--lr_c", type=float, default=1e-3)
    p.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def device_from(x):
    if x == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if x == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def load_data(folder):
    xs, i = [], 1
    while (folder / f"{i}_all.csv").is_file():
        xs.append(pd.read_csv(folder / f"{i}_all.csv", header=None).to_numpy(float))
        i += 1
    if not xs:
        raise FileNotFoundError(f"No omics files in {folder}")
    raw = pd.read_csv(folder / "labels_all.csv", header=None).iloc[:,0]
    levels = sorted(raw.unique().tolist())
    mapping = {v:i for i,v in enumerate(levels)}
    y = raw.map(mapping).to_numpy(np.int64)
    if any(x.shape[0] != len(y) for x in xs):
        raise ValueError(f"Sample count mismatch in {folder}")
    return xs, y, {str(k): int(v) for k,v in mapping.items()}

def preprocess(x, tr, te):
    a = x[tr].astype(float, copy=True)
    b = x[te].astype(float, copy=True)
    med = np.nanmedian(a, axis=0)
    med[~np.isfinite(med)] = 0
    for z in (a,b):
        bad = ~np.isfinite(z)
        if bad.any():
            z[bad] = med[np.where(bad)[1]]
    mean = a.mean(0)
    sd = a.std(0)
    sd[(~np.isfinite(sd)) | (sd < 1e-12)] = 1
    return ((a-mean)/sd).astype("float32"), ((b-mean)/sd).astype("float32")

def cosine(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    n1 = x1.norm(2, dim=1, keepdim=True)
    n2 = n1 if x2 is x1 else x2.norm(2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (n1*n2.t()).clamp(min=eps)

def threshold(edge_per_node, train):
    d = torch.sort(cosine(train).reshape(-1)).values
    idx = min(edge_per_node * train.shape[0], d.numel()-1)
    return float(d[idx].detach().cpu())

def sparse(x):
    idx = x.nonzero(as_tuple=False).t()
    vals = x[idx[0], idx[1]] if idx.numel() else torch.empty(0, device=x.device)
    return torch.sparse_coo_tensor(idx, vals, x.shape, device=x.device).coalesce()

def train_adj(x, th):
    d = cosine(x)
    g = (d <= th).float(); g.fill_diagonal_(0)
    a = (1-d)*g
    a = torch.maximum(a, a.t())
    a = F.normalize(a + torch.eye(len(a), device=a.device), p=1, dim=1)
    return sparse(a)

def test_adj(all_x, ntr, th):
    n = all_x.shape[0]
    a = torch.zeros((n,n), dtype=all_x.dtype, device=all_x.device)
    tr, te = all_x[:ntr], all_x[ntr:]
    d1 = cosine(tr,te); d2 = cosine(te,tr)
    a[:ntr,ntr:] = (1-d1)*(d1 <= th).float()
    a[ntr:,:ntr] = (1-d2)*(d2 <= th).float()
    a = torch.maximum(a, a.t())
    a = F.normalize(a + torch.eye(n, device=a.device), p=1, dim=1)
    return sparse(a)

def original_weights(y, k):
    counts = np.array([(y==i).sum() for i in range(k)], float)
    w = np.zeros(len(y), float)
    for i in range(k):
        w[y==i] = counts[i]/counts.sum()
    return w.astype("float32")

def train_epoch(xs, adjs, y, w, models, opts, vcdn):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    for m in models.values(): m.train()
    losses = {}
    for i in range(len(xs)):
        cn, en = f"C{i+1}", f"E{i+1}"
        opts[cn].zero_grad()
        out = models[cn](models[en](xs[i], adjs[i]))
        loss = torch.mean(criterion(out,y)*w)
        loss.backward(); opts[cn].step()
        losses[cn] = float(loss.detach().cpu())
    if vcdn and len(xs) >= 2:
        opts["C"].zero_grad()
        outs = [models[f"C{i+1}"](models[f"E{i+1}"](xs[i],adjs[i])) for i in range(len(xs))]
        loss = torch.mean(criterion(models["C"](outs),y)*w)
        loss.backward(); opts["C"].step()
        losses["C"] = float(loss.detach().cpu())
    return losses

@torch.no_grad()
def predict(xs, adjs, te_idx, models):
    for m in models.values(): m.eval()
    outs = [models[f"C{i+1}"](models[f"E{i+1}"](xs[i],adjs[i])) for i in range(len(xs))]
    logits = models["C"](outs) if len(xs) >= 2 else outs[0]
    return F.softmax(logits[te_idx], dim=1).cpu().numpy()

def metrics(y, prob, k):
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.clip(prob, 0, None)
    rs = prob.sum(1, keepdims=True)
    zero = rs[:,0] <= 0
    prob[zero] = 1/k
    prob /= prob.sum(1, keepdims=True)
    pred = prob.argmax(1)
    out = {
        "acc": float(accuracy_score(y,pred)),
        "balanced_acc": float(balanced_accuracy_score(y,pred)),
        "confusion_matrix": confusion_matrix(y,pred,labels=np.arange(k)).tolist(),
    }
    if k == 2:
        out.update(
            f1=float(f1_score(y,pred,zero_division=0)),
            auc=float(roc_auc_score(y,prob[:,1])),
            mcc=float(matthews_corrcoef(y,pred)),
        )
    else:
        out.update(
            f1_macro=float(f1_score(y,pred,average="macro",zero_division=0)),
            f1_weighted=float(f1_score(y,pred,average="weighted",zero_division=0)),
            auc_macro_ovr=float(roc_auc_score(y,prob,multi_class="ovr",average="macro")),
            auc_weighted_ovr=float(roc_auc_score(y,prob,multi_class="ovr",average="weighted")),
        )
    return out, pred, prob

def save_models(folder, models):
    folder.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        torch.save(model.state_dict(), folder/f"{name}.pt")

def run_one(a, dataset, repeat, device):
    out = Path(a.output_root)/dataset/"MOGONET"/f"repeat_{repeat}"
    jf = out/"metrics_test.json"
    if jf.is_file() and not a.overwrite:
        print(f"SKIP {dataset} repeat {repeat}")
        return
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seed = a.base_seed + repeat - 1
    set_seed(seed)

    xs, y, label_map = load_data(Path(a.data_root)/dataset)
    s = pd.read_csv(Path(a.split_root)/f"{dataset}_outer_splits.csv")
    cur = s[s["repeat"] == repeat]
    tr = cur.loc[cur["subset"]=="train","sample_index"].astype(int).to_numpy()
    te = cur.loc[cur["subset"]=="test","sample_index"].astype(int).to_numpy()
    if set(tr) & set(te): raise ValueError("train/test overlap")
    if not np.array_equal(cur.set_index("sample_index").loc[np.r_[tr,te],"encoded_label"].to_numpy(int), y[np.r_[tr,te]]):
        raise ValueError("split label mismatch")

    p0 = time.perf_counter()
    train_np, test_np = zip(*(preprocess(x,tr,te) for x in xs))
    preprocess_seconds = time.perf_counter()-p0
    train_x = [torch.from_numpy(x).to(device) for x in train_np]
    all_x = [torch.from_numpy(np.r_[x,z]).to(device) for x,z in zip(train_np,test_np)]

    g0 = time.perf_counter()
    th, atr, ate = [], [], []
    for x,z in zip(train_x,all_x):
        t = threshold(CONFIG[dataset]["adj"],x)
        th.append(t); atr.append(train_adj(x,t)); ate.append(test_adj(z,len(tr),t))
    sync(device); graph_seconds = time.perf_counter()-g0

    k, views = len(np.unique(y)), len(xs)
    dims = [x.shape[1] for x in train_x]
    models = init_model_dict(views,k,dims,CONFIG[dataset]["hidden"],k**views)
    for m in models.values(): m.to(device)
    yt = torch.as_tensor(y[tr],dtype=torch.long,device=device)
    wt = torch.as_tensor(original_weights(y[tr],k),dtype=torch.float32,device=device)

    sync(device); t0 = time.perf_counter()
    opts = init_optim(views,models,a.lr_e_pretrain,a.lr_c)
    for epoch in range(a.num_epoch_pretrain):
        loss = train_epoch(train_x,atr,yt,wt,models,opts,False)
        if epoch == 0 or (epoch+1)%100 == 0 or epoch+1 == a.num_epoch_pretrain:
            print(f"{dataset} repeat {repeat} pretrain {epoch+1}/{a.num_epoch_pretrain}: {loss}")
    opts = init_optim(views,models,a.lr_e,a.lr_c)
    for epoch in range(a.num_epoch):
        loss = train_epoch(train_x,atr,yt,wt,models,opts,True)
        if epoch == 0 or (epoch+1)%250 == 0 or epoch+1 == a.num_epoch:
            print(f"{dataset} repeat {repeat} joint {epoch+1}/{a.num_epoch}: {loss}")
    sync(device); train_seconds = time.perf_counter()-t0

    q0 = time.perf_counter()
    prob = predict(all_x,ate,list(range(len(tr),len(tr)+len(te))),models)
    sync(device); predict_seconds = time.perf_counter()-q0
    met,pred,prob = metrics(y[te],prob,k)
    total_seconds = time.perf_counter()-started

    result = {
        "dataset":dataset,"model":"MOGONET","repeat":repeat,"seed":seed,
        "n_train":len(tr),"n_test":len(te),"num_classes":k,"num_views":views,
        "input_dims":dims,"hidden_dims":CONFIG[dataset]["hidden"],
        "adj_parameter":CONFIG[dataset]["adj"],"adaptive_thresholds":th,
        "num_epoch_pretrain":a.num_epoch_pretrain,"num_epoch":a.num_epoch,
        "lr_e_pretrain":a.lr_e_pretrain,"lr_e":a.lr_e,"lr_c":a.lr_c,
        "device":str(device),"label_map":label_map,
        "preprocess_seconds":preprocess_seconds,"graph_seconds":graph_seconds,
        "train_seconds":train_seconds,"predict_seconds":predict_seconds,
        "total_seconds":total_seconds,
    }
    result.update(met)
    with jf.open("w",encoding="utf-8") as f: json.dump(result,f,indent=2)
    flat = {k:v for k,v in result.items() if not isinstance(v,(list,dict))}
    pd.DataFrame([flat]).to_csv(out/"metrics_test.csv",index=False)
    pred_df = pd.DataFrame({"sample_index":te,"y_true":y[te],"y_pred":pred})
    for j in range(k): pred_df[f"prob_class_{j}"] = prob[:,j]
    pred_df.to_csv(out/"predictions_test.csv",index=False)
    save_models(out/"models",models)
    print(f"COMPLETED {dataset} repeat {repeat}: ACC={met['acc']:.4f}, {total_seconds/60:.2f} min")

    del models,opts,train_x,all_x,atr,ate
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

def main():
    a = args()
    d = device_from(a.device)
    Path(a.output_root).mkdir(parents=True,exist_ok=True)
    print("Device:",d)
    if d.type == "cuda": print("GPU:",torch.cuda.get_device_name(0))
    for dataset in a.datasets:
        for repeat in a.repeats:
            run_one(a,dataset,repeat,d)
    print("All requested MOGONET runs completed.")

if __name__ == "__main__":
    main()
