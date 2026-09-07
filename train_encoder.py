from __future__ import annotations

import os
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Set
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
from sklearn.metrics import roc_auc_score


ALL_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
NON_PATHOLOGY = ["No Finding", "Support Devices"]
GLOBAL_POLICIES = {"u_ignore", "u_zeros", "u_ones", "u_ones_lsr"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class Config:
    preprocess_dir: Path = Path("outputs_preprocess")
    pairs_csv: Path = Path("pairs_out/pairs_cv.csv")   # shared fold map (for fold_nested)
    cache_root: Path = Path("mimic-cxr-jpg-320-cache")
    out_dir: Path = Path("encoders/run")

    labels: List[str] = field(default_factory=lambda: list(ALL_LABELS))
    image_size: int = 320

    # Single-image training hyperparameters.
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.03
    min_lr_frac: float = 0.05
    grad_clip_norm: float = 5.0
    pos_weight_cap: float = 10.0
    num_workers: int = min(8, os.cpu_count() or 4)
    channels_last: bool = True

    # Early stopping (patient-disjoint val slice from allowed patients only).
    val_frac: float = 0.10
    early_stopping_patience: int = 3
    seed: int = 42

    # Uncertainty policy for TRAINING targets. A SINGLE global rule applied to all
    # 14 labels (uncertain reads -> soft 0.75): the "U-Ones + label-smoothing
    # regularization" treatment of uncertain CheXpert labels, adopted from
    # Pham et al., Neurocomputing 437 (2021) 186-194,
    # doi:10.1016/j.neucom.2020.03.127. We select this as the global default
    # because it had the best mean validation AUROC in our own policy sweep;
    # override via --uncertainty-policy {u_ignore,u_zeros,u_ones,u_ones_lsr}.
    uncertainty_policy: str = "u_ones_lsr"
    lsr_value: float = 0.75


# Patient selection + leakage safeguards
def load_fold_map(pairs_csv: Path) -> Dict[int, int]:
    """subject_id -> fold, read from the shared pairs table (paired patients only)."""
    df = pd.read_csv(pairs_csv)
    if "fold" not in df.columns or "subject_id" not in df.columns:
        raise ValueError(f"{pairs_csv} must contain 'subject_id' and 'fold' (run build_pairs.py).")
    return {int(s): int(f) for s, f in zip(df["subject_id"], df["fold"])}


def select_encoder_patients(subset: str, fold: int, single_study_subjects: Set[int],
                            fold_map: Dict[int, int]) -> Tuple[Set[int], Set[int], str]:
    """
    Return (allowed_subjects, forbidden_subjects, tag).
      single_study : allowed = single-study patients; forbidden = ALL paired patients
                     (they must be disjoint -- single-study patients are never paired).
      fold_nested  : allowed = paired patients with fold != K;
                     forbidden = paired patients with fold == K (the held-out test fold).
    """
    paired = set(fold_map.keys())
    if subset == "single_study":
        allowed = set(single_study_subjects)
        forbidden = paired
        tag = "disjoint(single_study)"
    elif subset == "fold_nested":
        if fold is None or not (0 <= fold < 1 + max(fold_map.values())):
            raise ValueError("fold_nested requires --fold in the valid range.")
        allowed = {s for s, f in fold_map.items() if f != fold}
        forbidden = {s for s, f in fold_map.items() if f == fold}
        tag = f"fold_nested(fold!={fold})"
    else:
        raise ValueError(f"Unknown patient subset '{subset}'.")
    return allowed, forbidden, tag


def assert_no_leakage(allowed: Set[int], forbidden: Set[int], context: str) -> None:
    """Hard stop: the encoder's patients must not intersect the held-out patients."""
    overlap = allowed & forbidden
    if overlap:
        raise RuntimeError(
            f"LEAKAGE GUARD TRIPPED [{context}]: {len(overlap)} patient(s) appear in BOTH the "
            f"encoder-training set and the held-out set. Refusing to train. Examples: "
            f"{sorted(list(overlap))[:5]}")


def carve_patient_val(subjects: Set[int], frac: float, seed: int) -> Tuple[Set[int], Set[int]]:
    """Split a patient set into (train, val) disjointly BY PATIENT for early stopping."""
    subs = np.array(sorted(subjects))
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n_val = max(1, int(round(len(subs) * frac)))
    val = set(int(s) for s in subs[:n_val])
    train = set(int(s) for s in subs[n_val:])
    return train, val


# Uncertainty policy -> training targets
def resolve_active_policy(label: str, cfg: Config) -> str:
    """A single global uncertainty policy is applied to every label."""
    return cfg.uncertainty_policy


def derive_target_mask_from_raw(raw: np.ndarray, policy: str, lsr_value: float):
    """One label's (target, mask) from raw {1,0,-1,NaN}; blank -> negative."""
    if policy not in GLOBAL_POLICIES:
        raise ValueError(f"Unknown policy '{policy}'.")
    unmentioned = np.isnan(raw)
    uncertain = (raw == -1)
    target = np.where(unmentioned, 0.0, raw).astype("float32")
    mask = np.ones_like(target, dtype="float32")
    if policy == "u_ignore":
        target = np.where(uncertain, 0.0, target); mask = np.where(uncertain, 0.0, 1.0).astype("float32")
    elif policy == "u_zeros":
        target = np.where(uncertain, 0.0, target)
    elif policy == "u_ones":
        target = np.where(uncertain, 1.0, target)
    else:  # u_ones_lsr
        target = np.where(uncertain, float(lsr_value), target)
    return np.clip(target, 0.0, 1.0).astype("float32"), mask


def build_targets_masks(df: pd.DataFrame, cfg: Config):
    labels = cfg.labels
    n = len(df)
    targets = np.zeros((n, len(labels)), dtype="float32")
    masks = np.zeros((n, len(labels)), dtype="float32")
    for i, col in enumerate(labels):
        raw = pd.to_numeric(df[f"raw_{col}"], errors="coerce").to_numpy()
        t, m = derive_target_mask_from_raw(raw, resolve_active_policy(col, cfg), cfg.lsr_value)
        targets[:, i] = t; masks[:, i] = m
    return targets, masks


# Dataset + model (module-level -> picklable for Windows DataLoader workers)
class SingleImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: "Config", train: bool):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.targets, self.masks = build_targets_masks(self.df, cfg)
        self.paths = self.df["cached_image_path"].astype(str).tolist()
        aug = [T.RandomResizedCrop(cfg.image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05),
                                   interpolation=InterpolationMode.BILINEAR),
               T.RandomRotation(7)] if train else []
        self.tf = T.Compose(aug + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def _load(self, path):
        p = Path(path)
        if not p.exists():
            alt = self.cfg.cache_root / p
            if alt.exists():
                p = alt
        return self.tf(Image.open(p).convert("RGB"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {"image": self._load(self.paths[idx]),
                "target": torch.from_numpy(self.targets[idx]),
                "mask": torch.from_numpy(self.masks[idx])}


class DenseNetClassifier(nn.Module):
    """DenseNet-121 backbone + a 14-way multi-label head. Only `features` is transferred."""
    def __init__(self, n_labels: int):
        super().__init__()
        base = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1024, n_labels))

    def forward(self, x):
        f = torch.relu(self.features(x))
        f = self.pool(f).flatten(1)
        return self.classifier(f)


def masked_bce(logits, targets, mask, pos_weight):
    loss = F.binary_cross_entropy_with_logits(logits, targets, weight=None,
                                              pos_weight=pos_weight, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


@torch.no_grad()
def val_macro_auroc(model, loader, device, cfg, use_amp):
    model.eval()
    ys, ps, ms = [], [], []
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        if cfg.channels_last and device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with (torch.amp.autocast("cuda") if use_amp else nullcontext()):
            p = torch.sigmoid(model(x).float())
        ps.append(p.cpu().numpy()); ys.append(b["target"].numpy()); ms.append(b["mask"].numpy())
    y = np.concatenate(ys); p = np.concatenate(ps); m = np.concatenate(ms)
    aurocs = []
    for i, lab in enumerate(cfg.labels):
        keep = m[:, i].astype(bool)
        yt = (y[keep, i] >= 0.5).astype(int)
        if keep.sum() > 0 and len(np.unique(yt)) > 1:
            aurocs.append(roc_auc_score(yt, p[keep, i]))
    return float(np.mean(aurocs)) if aurocs else float("nan")


# Training
def train_encoder(cfg: Config, subset: str, fold: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    # ---- pool all official splits (patient identity, not split, drives selection) ----
    mans = []
    for s in ["train", "validate", "test"]:
        p = cfg.preprocess_dir / f"manifest_{s}_frontal_{cfg.image_size}px_cached.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run preprocess.py first.")
        mans.append(pd.read_csv(p))
    man = pd.concat(mans, ignore_index=True)

    single_study_subjects = set(int(s) for s in man.loc[man["is_single_study"] == True, "subject_id"].unique())

    # ---- select allowed vs forbidden patients from the SHARED fold map ----
    fold_map = load_fold_map(cfg.pairs_csv) if subset == "fold_nested" else {}
    if subset == "single_study":
        # forbidden = all paired patients; we still load the fold map if present to define them
        if cfg.pairs_csv.exists():
            fold_map = load_fold_map(cfg.pairs_csv)
    allowed, forbidden, tag = select_encoder_patients(subset, fold, single_study_subjects, fold_map)

    # ---- HARD leakage assertion BEFORE any training ----
    assert_no_leakage(allowed, forbidden, context=tag)

    # ---- carve patient-disjoint early-stopping val slice from ALLOWED only ----
    tr_subjects, va_subjects = carve_patient_val(allowed, cfg.val_frac, cfg.seed)
    # (val is a subset of allowed -> also disjoint from forbidden by construction)
    assert_no_leakage(va_subjects, forbidden, context=tag + " [val slice]")

    train_df = man[man["subject_id"].isin(tr_subjects)].copy()
    val_df = man[man["subject_id"].isin(va_subjects)].copy()
    print(f"[{tag}] patients: allowed={len(allowed)} (train={len(tr_subjects)}, val={len(va_subjects)}), "
          f"forbidden={len(forbidden)}")
    print(f"[{tag}] images:   train={len(train_df)}, val={len(val_df)}")
    if len(train_df) == 0:
        raise RuntimeError("No training images selected — check subset/fold and the fold map.")

    train_ds = SingleImageDataset(train_df, cfg, train=True)
    val_ds = SingleImageDataset(val_df, cfg, train=False)

    # pos_weight from training targets/masks
    pos = (train_ds.targets * train_ds.masks).sum(0)
    neg = ((1 - train_ds.targets) * train_ds.masks).sum(0)
    pw = np.clip(np.where(pos > 0, neg / np.clip(pos, 1, None), 1.0), 0, cfg.pos_weight_cap)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)

    pin = device.type == "cuda"
    mk = lambda ds, sh: DataLoader(ds, batch_size=cfg.batch_size, shuffle=sh, num_workers=cfg.num_workers,
                                   pin_memory=pin, persistent_workers=cfg.num_workers > 0,
                                   prefetch_factor=4 if cfg.num_workers > 0 else None)
    train_loader, val_loader = mk(train_ds, True), mk(val_ds, False)

    model = DenseNetClassifier(len(cfg.labels)).to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup = max(1, int(total_steps * cfg.warmup_frac))
    def lr_lambda(s):
        if s < warmup: return (s + 1) / warmup
        pr = min(1.0, (s - warmup) / max(1, total_steps - warmup))
        return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * pr))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log = []
    best_auroc, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0; nb = 0
        for b in tqdm(train_loader, desc=f"[{tag}] epoch {epoch}/{cfg.epochs}", leave=False):
            x = b["image"].to(device, non_blocking=True)
            if cfg.channels_last and device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            y = b["target"].to(device, non_blocking=True)
            m = b["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with (torch.amp.autocast("cuda") if use_amp else nullcontext()):
                loss = masked_bce(model(x), y, m, pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item(); nb += 1
        auroc = val_macro_auroc(model, val_loader, device, cfg, use_amp)
        log.append({"epoch": epoch, "train_loss": running / max(nb, 1), "val_macro_auroc": auroc})
        print(f"[{tag}] epoch {epoch}: loss={running/max(nb,1):.4f}  val_macro_AUROC={auroc:.4f}")
        if auroc > best_auroc + 1e-4:
            best_auroc, no_improve = auroc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= cfg.early_stopping_patience:
                print(f"[{tag}] early stop at epoch {epoch}"); break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- save encoder ----
    meta = {
        "subset": subset, "fold": fold, "tag": tag,
        "n_allowed_patients": len(allowed), "n_train_patients": len(tr_subjects),
        "n_val_patients": len(va_subjects), "n_forbidden_patients": len(forbidden),
        "leakage_overlap": 0, "best_val_macro_auroc": best_auroc,
        "labels": cfg.labels, "image_size": cfg.image_size, "seed": cfg.seed,
    }
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, cfg.out_dir / "encoder.pt")
    with open(cfg.out_dir / "encoder_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    pd.DataFrame(log).to_csv(cfg.out_dir / "training_log.csv", index=False)
    print(f"[{tag}] saved encoder.pt (best val macro-AUROC={best_auroc:.4f}) -> {cfg.out_dir.resolve()}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Pretrain the DenseNet-121 encoder (leakage-safe).")
    p.add_argument("--patient-subset", required=True, choices=["single_study", "fold_nested"])
    p.add_argument("--fold", type=int, default=None, help="Required for fold_nested (0..4).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pairs-csv", default=None)
    p.add_argument("--preprocess-dir", default=None)
    p.add_argument("--uncertainty-policy", default=None,
                   choices=["u_ignore", "u_zeros", "u_ones", "u_ones_lsr"],
                   help="Global policy for uncertain labels (default u_ones_lsr).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    args = build_arg_parser().parse_args()
    cfg = Config()
    cfg.out_dir = Path(args.out_dir)
    if args.pairs_csv is not None: cfg.pairs_csv = Path(args.pairs_csv)
    if args.preprocess_dir is not None: cfg.preprocess_dir = Path(args.preprocess_dir)
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.uncertainty_policy is not None: cfg.uncertainty_policy = args.uncertainty_policy
    if args.seed is not None: cfg.seed = args.seed
    if args.patient_subset == "fold_nested" and args.fold is None:
        raise SystemExit("--fold is required when --patient-subset fold_nested")
    train_encoder(cfg, args.patient_subset, args.fold)
