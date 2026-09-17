from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from models_v2 import ModelConfig, build_model


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEFAULT_DEVICE.type == "cpu":
    # More stable on some CPU runtimes for Conv1d/attention style workloads.
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class TrainingConfig:
    data_dir: str
    output_dir: str
    model_type: str = "mcof_attn"
    test_size: float = 0.20
    repeats: int = 5
    inner_folds: int = 5
    random_seed: int = 42
    hidden_dim: int = 128
    num_heads: int = 4
    conv_channels: int = 64
    conv_kernel_size: int = 5
    dropout: float = 0.30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 30
    gradient_clip_norm: float = 5.0
    monitor_metric: str = "auto"  # auto / loss / mcc / f1_macro
    impute_strategy: str = "median"
    scaler: str = "standard"  # standard / robust / none
    max_missing_rate: float = 1.0
    max_zero_rate: float = 1.0
    select_k: Optional[int] = None
    selector: str = "none"  # none / f_classif / variance
    num_workers: int = 0


class OmicsPreprocessor:
    def __init__(
        self,
        max_missing_rate: float = 1.0,
        max_zero_rate: float = 1.0,
        impute_strategy: str = "median",
        scaler: str = "standard",
        select_k: Optional[int] = None,
        selector: str = "none",
    ) -> None:
        self.max_missing_rate = float(max_missing_rate)
        self.max_zero_rate = float(max_zero_rate)
        self.impute_strategy = impute_strategy
        self.scaler = scaler
        self.select_k = select_k
        self.selector = selector
        self.states: List[Dict[str, Any]] = []

    def _nan_stat(self, arr: np.ndarray, strategy: str) -> np.ndarray:
        if strategy == "mean":
            stat = np.nanmean(arr, axis=0)
        elif strategy == "median":
            stat = np.nanmedian(arr, axis=0)
        else:
            raise ValueError(f"Unsupported impute strategy: {strategy}")
        stat = np.where(np.isnan(stat), 0.0, stat)
        return stat

    def _fit_single(self, x: np.ndarray, y: Optional[np.ndarray]) -> Dict[str, Any]:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x.shape}")

        missing_rate = np.mean(np.isnan(x), axis=0)
        zero_rate = np.mean(np.nan_to_num(x, nan=0.0) == 0.0, axis=0)
        keep_mask = (missing_rate <= self.max_missing_rate) & (zero_rate <= self.max_zero_rate)
        if not np.any(keep_mask):
            keep_mask = np.ones(x.shape[1], dtype=bool)

        x = x[:, keep_mask]
        impute_values = self._nan_stat(x, self.impute_strategy)
        x_imp = np.where(np.isnan(x), impute_values[None, :], x)

        if self.scaler == "standard":
            center = np.mean(x_imp, axis=0)
            scale = np.std(x_imp, axis=0, ddof=0)
            scale[scale == 0.0] = 1.0
        elif self.scaler == "robust":
            center = np.median(x_imp, axis=0)
            q75 = np.percentile(x_imp, 75, axis=0)
            q25 = np.percentile(x_imp, 25, axis=0)
            scale = q75 - q25
            scale[scale == 0.0] = 1.0
        elif self.scaler == "none":
            center = np.zeros(x_imp.shape[1], dtype=np.float32)
            scale = np.ones(x_imp.shape[1], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported scaler: {self.scaler}")

        x_scaled = (x_imp - center[None, :]) / scale[None, :]

        select_idx = np.arange(x_scaled.shape[1])
        if self.select_k is not None and self.select_k < x_scaled.shape[1]:
            if self.selector == "f_classif":
                if y is None:
                    raise ValueError("f_classif selector requires labels.")
                scores, _ = f_classif(x_scaled, y)
                scores = np.where(np.isnan(scores), -np.inf, scores)
                select_idx = np.argsort(scores)[::-1][: self.select_k]
            elif self.selector == "variance":
                variances = np.var(x_scaled, axis=0)
                select_idx = np.argsort(variances)[::-1][: self.select_k]
            elif self.selector == "none":
                select_idx = np.arange(min(self.select_k, x_scaled.shape[1]))
            else:
                raise ValueError(f"Unsupported selector: {self.selector}")
            select_idx = np.sort(select_idx)

        state = {
            "keep_mask": keep_mask.astype(bool),
            "impute_values": impute_values.astype(np.float32),
            "center": center.astype(np.float32),
            "scale": scale.astype(np.float32),
            "select_idx": select_idx.astype(np.int64),
        }
        return state

    def fit(self, omics_list: Sequence[np.ndarray], y: Optional[np.ndarray]) -> "OmicsPreprocessor":
        self.states = [self._fit_single(x, y) for x in omics_list]
        return self

    def _transform_single(self, x: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
        x = x[:, state["keep_mask"]]
        x = np.where(np.isnan(x), state["impute_values"][None, :], x)
        x = (x - state["center"][None, :]) / state["scale"][None, :]
        x = x[:, state["select_idx"]]
        return x.astype(np.float32)

    def transform(self, omics_list: Sequence[np.ndarray]) -> List[np.ndarray]:
        if not self.states:
            raise RuntimeError("Preprocessor must be fit before transform.")
        return [self._transform_single(x, s) for x, s in zip(omics_list, self.states)]

    def transformed_dims(self) -> List[int]:
        return [int(len(state["select_idx"])) for state in self.states]

    def transform_feature_names(self, feature_names: Sequence[Sequence[str]]) -> List[List[str]]:
        if not self.states:
            raise RuntimeError("Preprocessor must be fit before transforming feature names.")
        out: List[List[str]] = []
        for names, state in zip(feature_names, self.states):
            arr = np.asarray(names, dtype=object)
            arr = arr[state["keep_mask"]]
            arr = arr[state["select_idx"]]
            out.append(arr.astype(str).tolist())
        return out

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "max_missing_rate": self.max_missing_rate,
            "max_zero_rate": self.max_zero_rate,
            "impute_strategy": self.impute_strategy,
            "scaler": self.scaler,
            "select_k": self.select_k,
            "selector": self.selector,
            "states": self.states,
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Any]) -> "OmicsPreprocessor":
        obj = cls(
            max_missing_rate=state_dict["max_missing_rate"],
            max_zero_rate=state_dict["max_zero_rate"],
            impute_strategy=state_dict["impute_strategy"],
            scaler=state_dict["scaler"],
            select_k=state_dict.get("select_k"),
            selector=state_dict.get("selector", "none"),
        )
        obj.states = state_dict["states"]
        return obj


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_multiomics_dataset(data_dir: str | Path) -> tuple[List[np.ndarray], np.ndarray, List[List[str]]]:
    data_dir = Path(data_dir)
    omics_list: List[np.ndarray] = []
    feature_names: List[List[str]] = []

    omics_idx = 1
    while True:
        omics_path = data_dir / f"{omics_idx}_all.csv"
        if not omics_path.exists():
            break
        df = pd.read_csv(omics_path, header=None)
        omics_list.append(df.to_numpy(dtype=np.float32))

        feat_path = data_dir / f"{omics_idx}_featname.csv"
        if feat_path.exists():
            feat_df = pd.read_csv(feat_path, header=None)
            feature_names.append(feat_df.iloc[:, 0].astype(str).tolist())
        else:
            feature_names.append([f"omics{omics_idx}_f{j}" for j in range(df.shape[1])])
        omics_idx += 1

    if not omics_list:
        raise FileNotFoundError(f"No *all.csv omics files found in {data_dir}")

    labels_path = data_dir / "labels_all.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels_all.csv not found in {data_dir}")
    labels_df = pd.read_csv(labels_path, header=None)
    labels_series = labels_df.iloc[:, 0]
    labels, _ = pd.factorize(labels_series, sort=True)
    labels = labels.astype(np.int64)

    n_samples = len(labels)
    if any(x.shape[0] != n_samples for x in omics_list):
        raise ValueError("All omics matrices must have the same number of samples as labels.")

    return omics_list, labels, feature_names


def concat_omics(omics_list: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(omics_list, axis=1).astype(np.float32)


def make_tensor_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.from_numpy(weights.astype(np.float32))


def metric_to_monitor(metrics: Dict[str, float], num_classes: int, mode: str) -> float:
    if mode == "loss":
        return -float(metrics["loss"])
    if mode == "mcc":
        return float(metrics.get("mcc", float("-inf")))
    if mode == "f1_macro":
        return float(metrics.get("f1_macro", float("-inf")))
    if mode == "auto":
        return float(metrics.get("mcc") if num_classes == 2 else metrics.get("f1_macro"))
    if mode in metrics:
        return float(metrics[mode])
    raise ValueError(f"Unknown monitor mode: {mode}")


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(prob)
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(probs)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    ys: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
        losses.append(float(loss.item()) * len(xb))
        ys.append(yb.cpu().numpy())
        probs.append(prob)

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    metrics = compute_metrics(y_true, y_prob, num_classes=num_classes)
    metrics["loss"] = float(sum(losses) / len(y_true))
    return metrics


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> Dict[str, Any]:
    y_pred = np.argmax(y_prob, axis=1)
    out: Dict[str, Any] = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(num_classes)).tolist(),
    }

    if num_classes == 2:
        pos_prob = y_prob[:, 1]
        out.update(
            {
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }
        )
        try:
            out["auc"] = float(roc_auc_score(y_true, pos_prob))
        except ValueError:
            out["auc"] = float("nan")
    else:
        out.update(
            {
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            }
        )
        try:
            out["auc_macro_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            out["auc_weighted_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except ValueError:
            out["auc_macro_ovr"] = float("nan")
            out["auc_weighted_ovr"] = float("nan")
    return out


def plot_history(history: Dict[str, List[float]], save_path: str | Path, title: str) -> None:
    save_path = Path(save_path)
    plt.figure(figsize=(8, 5))
    if history.get("train_loss"):
        plt.plot(history["train_loss"], label="train_loss")
    if history.get("val_loss"):
        plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int, save_path: str | Path) -> None:
    save_path = Path(save_path)
    plt.figure(figsize=(6, 6))
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    else:
        for cls in range(num_classes):
            binary_true = (y_true == cls).astype(int)
            try:
                fpr, tpr, _ = roc_curve(binary_true, y_prob[:, cls])
                auc = roc_auc_score(binary_true, y_prob[:, cls])
                plt.plot(fpr, tpr, label=f"class {cls}: AUC={auc:.3f}")
            except ValueError:
                continue
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


class EarlyStopper:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score = float("-inf")
        self.best_state: Optional[Dict[str, Any]] = None
        self.best_epoch = 0
        self.counter = 0

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_with_validation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    monitor_metric: str,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> tuple[nn.Module, Dict[str, List[float]], Dict[str, Any], int]:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, patience // 4), min_lr=1e-6
    )
    early_stopper = EarlyStopper(patience=patience)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_metrics: Dict[str, Any] = {}

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
            running_loss += float(loss.item()) * len(xb)
            n_train += len(xb)

        train_loss = running_loss / max(n_train, 1)
        val_metrics = evaluate_loader(model, val_loader, criterion, device=device, num_classes=num_classes)
        scheduler.step(val_metrics["loss"])
        score = metric_to_monitor(val_metrics, num_classes=num_classes, mode=monitor_metric)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_metrics["loss"]))

        if score >= early_stopper.best_score:
            best_val_metrics = val_metrics
        should_stop = early_stopper.step(score, model, epoch)
        if should_stop:
            break

    if early_stopper.best_state is None:
        raise RuntimeError("Training failed to produce any checkpoint.")
    model.load_state_dict(early_stopper.best_state)
    return model, history, best_val_metrics, early_stopper.best_epoch


def train_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> tuple[nn.Module, Dict[str, List[float]]]:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = {"train_loss": []}
    for _epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
            running_loss += float(loss.item()) * len(xb)
            n_train += len(xb)
        history["train_loss"].append(running_loss / max(n_train, 1))
    return model, history


def build_model_from_training_config(train_cfg: TrainingConfig, input_dims: List[int], num_classes: int) -> nn.Module:
    model_cfg = ModelConfig(
        model_type=train_cfg.model_type,
        input_dims=input_dims,
        num_classes=num_classes,
        hidden_dim=train_cfg.hidden_dim,
        num_heads=train_cfg.num_heads,
        dropout=train_cfg.dropout,
        conv_channels=train_cfg.conv_channels,
        conv_kernel_size=train_cfg.conv_kernel_size,
    )
    return build_model(model_cfg)


def inner_cross_validate(
    omics_train: Sequence[np.ndarray],
    y_train: np.ndarray,
    feature_names: Sequence[Sequence[str]],
    train_cfg: TrainingConfig,
    device: torch.device,
    repeat_dir: Path,
) -> Dict[str, Any]:
    num_classes = int(len(np.unique(y_train)))
    skf = StratifiedKFold(n_splits=train_cfg.inner_folds, shuffle=True, random_state=train_cfg.random_seed)
    fold_records: List[Dict[str, Any]] = []
    best_epochs: List[int] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(np.zeros_like(y_train), y_train), start=1):
        fold_dir = ensure_dir(repeat_dir / f"inner_fold_{fold_idx}")
        x_tr_raw = [x[tr_idx] for x in omics_train]
        x_val_raw = [x[val_idx] for x in omics_train]
        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

        preprocessor = OmicsPreprocessor(
            max_missing_rate=train_cfg.max_missing_rate,
            max_zero_rate=train_cfg.max_zero_rate,
            impute_strategy=train_cfg.impute_strategy,
            scaler=train_cfg.scaler,
            select_k=train_cfg.select_k,
            selector=train_cfg.selector,
        ).fit(x_tr_raw, y_tr)

        x_tr = concat_omics(preprocessor.transform(x_tr_raw))
        x_val = concat_omics(preprocessor.transform(x_val_raw))
        input_dims = preprocessor.transformed_dims()
        model = build_model_from_training_config(train_cfg, input_dims=input_dims, num_classes=num_classes).to(device)

        train_loader = make_tensor_loader(x_tr, y_tr, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
        val_loader = make_tensor_loader(x_val, y_val, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)

        class_weights = compute_class_weights(y_tr, num_classes=num_classes)
        model, history, val_metrics, best_epoch = train_with_validation(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            max_epochs=train_cfg.max_epochs,
            patience=train_cfg.patience,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            gradient_clip_norm=train_cfg.gradient_clip_norm,
            monitor_metric=train_cfg.monitor_metric,
            device=device,
            class_weights=class_weights,
        )
        plot_history(history, fold_dir / "loss_curve.png", title=f"repeat {repeat_dir.name} fold {fold_idx}")
        y_val_true, y_val_prob = predict_proba(model, val_loader, device)
        plot_roc(y_val_true, y_val_prob, num_classes=num_classes, save_path=fold_dir / "roc.png")

        fold_record: Dict[str, Any] = {"fold": fold_idx, "best_epoch": best_epoch}
        fold_record.update({k: v for k, v in val_metrics.items() if k != "confusion_matrix"})
        fold_records.append(fold_record)
        best_epochs.append(best_epoch)

    cv_df = pd.DataFrame(fold_records)
    cv_df.to_csv(repeat_dir / "inner_cv_metrics.csv", index=False)

    summary = {
        "fold_records": fold_records,
        "best_epoch_median": int(np.median(best_epochs)),
        "best_epoch_mean": int(np.round(np.mean(best_epochs))),
        "num_classes": num_classes,
    }
    return summary


def train_final_and_evaluate(
    omics_train: Sequence[np.ndarray],
    y_train: np.ndarray,
    omics_test: Sequence[np.ndarray],
    y_test: np.ndarray,
    feature_names: Sequence[Sequence[str]],
    train_cfg: TrainingConfig,
    best_epochs: int,
    device: torch.device,
    repeat_dir: Path,
    repeat_index: int,
) -> Dict[str, Any]:
    num_classes = int(len(np.unique(np.concatenate([y_train, y_test]))))
    preprocessor = OmicsPreprocessor(
        max_missing_rate=train_cfg.max_missing_rate,
        max_zero_rate=train_cfg.max_zero_rate,
        impute_strategy=train_cfg.impute_strategy,
        scaler=train_cfg.scaler,
        select_k=train_cfg.select_k,
        selector=train_cfg.selector,
    ).fit(omics_train, y_train)

    x_train = concat_omics(preprocessor.transform(omics_train))
    x_test = concat_omics(preprocessor.transform(omics_test))
    transformed_feature_names = preprocessor.transform_feature_names(feature_names)
    input_dims = preprocessor.transformed_dims()

    model = build_model_from_training_config(train_cfg, input_dims=input_dims, num_classes=num_classes).to(device)
    train_loader = make_tensor_loader(x_train, y_train, batch_size=train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    test_loader = make_tensor_loader(x_test, y_test, batch_size=train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)

    class_weights = compute_class_weights(y_train, num_classes=num_classes)
    model, history = train_fixed_epochs(
        model=model,
        train_loader=train_loader,
        epochs=max(1, best_epochs),
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        gradient_clip_norm=train_cfg.gradient_clip_norm,
        device=device,
        class_weights=class_weights,
    )
    plot_history(history, repeat_dir / "final_train_loss.png", title=f"repeat_{repeat_index}_final_train")

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_loader(model, test_loader, criterion=criterion, device=device, num_classes=num_classes)
    y_test_true, y_test_prob = predict_proba(model, test_loader, device=device)
    plot_roc(y_test_true, y_test_prob, num_classes=num_classes, save_path=repeat_dir / "test_roc.png")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "training_config": asdict(train_cfg),
        "model_config": asdict(
            ModelConfig(
                model_type=train_cfg.model_type,
                input_dims=input_dims,
                num_classes=num_classes,
                hidden_dim=train_cfg.hidden_dim,
                num_heads=train_cfg.num_heads,
                dropout=train_cfg.dropout,
                conv_channels=train_cfg.conv_channels,
                conv_kernel_size=train_cfg.conv_kernel_size,
            )
        ),
        "preprocessor_state": preprocessor.to_state_dict(),
        "feature_names": transformed_feature_names,
        "repeat_index": repeat_index,
        "metrics": test_metrics,
        "best_epochs": int(best_epochs),
    }
    ckpt_path = repeat_dir / "best_model.pt"
    torch.save(checkpoint, ckpt_path)

    row: Dict[str, Any] = {"repeat": repeat_index, "checkpoint": str(ckpt_path)}
    row.update({k: v for k, v in test_metrics.items() if k != "confusion_matrix"})
    return row


def summarize_repeat_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    numeric_cols = [
        c for c in df.columns if c not in {"repeat", "checkpoint"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    summary: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        summary[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std(ddof=1)) if len(df) > 1 else 0.0,
        }
    return summary


def run_repeated_holdout_experiment(train_cfg: TrainingConfig, device: torch.device = DEFAULT_DEVICE) -> Dict[str, Any]:
    seed_everything(train_cfg.random_seed)
    output_dir = ensure_dir(train_cfg.output_dir)
    omics_list, labels, feature_names = read_multiomics_dataset(train_cfg.data_dir)
    splitter = StratifiedShuffleSplit(
        n_splits=train_cfg.repeats,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_seed,
    )

    repeat_rows: List[Dict[str, Any]] = []
    for repeat_index, (tr_idx, te_idx) in enumerate(splitter.split(np.zeros_like(labels), labels), start=1):
        repeat_dir = ensure_dir(output_dir / f"repeat_{repeat_index}")
        omics_train = [x[tr_idx] for x in omics_list]
        omics_test = [x[te_idx] for x in omics_list]
        y_train = labels[tr_idx]
        y_test = labels[te_idx]

        cv_summary = inner_cross_validate(
            omics_train=omics_train,
            y_train=y_train,
            feature_names=feature_names,
            train_cfg=train_cfg,
            device=device,
            repeat_dir=repeat_dir,
        )
        with open(repeat_dir / "inner_cv_summary.json", "w", encoding="utf-8") as f:
            json.dump(cv_summary, f, indent=2, cls=NumpyJSONEncoder)

        final_row = train_final_and_evaluate(
            omics_train=omics_train,
            y_train=y_train,
            omics_test=omics_test,
            y_test=y_test,
            feature_names=feature_names,
            train_cfg=train_cfg,
            best_epochs=cv_summary["best_epoch_median"],
            device=device,
            repeat_dir=repeat_dir,
            repeat_index=repeat_index,
        )
        repeat_rows.append(final_row)

    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(output_dir / "repeat_test_metrics.csv", index=False)
    summary = summarize_repeat_metrics(repeat_df)
    result = {
        "repeat_metrics": repeat_rows,
        "summary": summary,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, cls=NumpyJSONEncoder)
    return result


def load_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def restore_model_from_checkpoint(checkpoint: Dict[str, Any], device: torch.device = DEFAULT_DEVICE) -> nn.Module:
    model_cfg = ModelConfig(**checkpoint["model_config"])
    model = build_model(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

