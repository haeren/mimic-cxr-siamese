from __future__ import annotations

import os
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
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
from sklearn.metrics import f1_score, recall_score, roc_auc_score


ALL_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
# Well-supported findings (>= 800 onset AND resolved at 180d) -> selection + headline.
TIER1 = ["Lung Opacity", "Support Devices", "No Finding", "Atelectasis",
         "Cardiomegaly", "Pleural Effusion", "Edema"]
N_TRANS = 4
TRANS_NAME = {0: "0->0", 1: "0->1_onset", 2: "1->0_resolved", 3: "1->1_persist"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class Config:
    pairs_csv: Path = Path("pairs_out/pairs_gap180.csv")
    cache_root: Path = Path("mimic-cxr-jpg-320-cache")
    dataset_root: Path = Path("mimic-cxr-jpg-2.1.0.physionet.org")
    out_dir: Path = Path("runs/run")

    # Encoder arm resolution
    encoder_arm: str = "fold_nested"          # imagenet | disjoint | fold_nested
    encoder_root: Path = Path("encoders")
    disjoint_subdir: str = "disjoint"         # encoders/disjoint/encoder.pt
    nested_prefix: str = "nested_gap180_fold" # encoders/nested_gap180_fold{K}/encoder.pt
    encoder_file: str = "encoder.pt"

    labels: List[str] = field(default_factory=lambda: list(ALL_LABELS))
    image_size: int = 320
    n_folds: int = 5

    mode: str = "siamese"                     # siamese | last_only
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    hidden_dim: int = 512
    grad_clip_norm: float = 5.0
    warmup_frac: float = 0.05
    min_lr_frac: float = 0.05
    num_workers: int = min(8, os.cpu_count() or 4)
    val_frac_of_train: float = 0.1
    early_stopping_patience: int = 4
    max_class_weight: float = 10.0
    seed: int = 42


# Encoder arm -> checkpoint path
def resolve_encoder_path(cfg: Config, fold: int) -> Optional[Path]:
    """
    Which encoder checkpoint to load for a given fold.
      imagenet    -> None (model keeps its ImageNet weights)
      disjoint    -> encoder_root/disjoint_subdir/encoder.pt          (same for all folds)
      fold_nested -> encoder_root/{nested_prefix}{fold}/encoder.pt    (per-fold)
    """
    if cfg.encoder_arm == "imagenet":
        return None
    if cfg.encoder_arm == "disjoint":
        return cfg.encoder_root / cfg.disjoint_subdir / cfg.encoder_file
    if cfg.encoder_arm == "fold_nested":
        return cfg.encoder_root / f"{cfg.nested_prefix}{fold}" / cfg.encoder_file
    raise ValueError(f"Unknown encoder_arm '{cfg.encoder_arm}'.")


def extract_encoder_state(ckpt_state: Dict[str, object]) -> Dict[str, object]:
    """Keep DenseNet `features.*` weights, drop the classifier head. Tolerates a
    leading 'model.' prefix. Used to transfer a train_encoder.py checkpoint."""
    out = {}
    for k, v in ckpt_state.items():
        kk = k[len("model."):] if k.startswith("model.") else k
        if kk.startswith("features."):
            out[kk[len("features."):]] = v
    if not out:
        raise ValueError("No 'features.*' keys found in encoder checkpoint.")
    return out


# Class weights + metrics
def compute_class_weights(trans: np.ndarray, mask: np.ndarray, labels: List[str], max_w: float) -> np.ndarray:
    """Inverse-frequency 4-class weights per finding (train folds only). Empty
    classes -> weight 0; weights normalized to mean ~1 per finding and clipped."""
    L = len(labels)
    W = np.ones((L, N_TRANS), dtype="float64")
    for i in range(L):
        m = mask[:, i].astype(bool)
        y = trans[m, i].astype(int)
        counts = np.array([(y == c).sum() for c in range(N_TRANS)], dtype="float64")
        inv = np.zeros(N_TRANS, dtype="float64")
        nz = counts > 0
        inv[nz] = 1.0 / counts[nz]
        if inv.sum() == 0:
            continue
        w = inv / inv[inv > 0].mean()
        W[i] = np.clip(w, 0.0, max_w)
    return W


def transition_metrics(y_true: np.ndarray, probs: np.ndarray, mask: np.ndarray, labels: List[str]) -> pd.DataFrame:
    """Per finding: per-class recall, macro-F1 (4 classes), onset/resolved 1-vs-rest AUROC."""
    rows = []
    for i, lab in enumerate(labels):
        m = mask[:, i].astype(bool)
        yt = y_true[m, i].astype(int)
        pr = probs[m, i, :]
        if len(yt) == 0:
            continue
        yhat = pr.argmax(axis=1)
        rec = recall_score(yt, yhat, labels=list(range(N_TRANS)), average=None, zero_division=0)
        macro_f1 = f1_score(yt, yhat, labels=list(range(N_TRANS)), average="macro", zero_division=0)

        def ovr(cls):
            yb = (yt == cls).astype(int)
            if yb.sum() == 0 or yb.sum() == len(yb):
                return np.nan
            return roc_auc_score(yb, pr[:, cls])

        rows.append({
            "label": lab, "n": int(len(yt)),
            "recall_0to0": rec[0], "recall_onset": rec[1],
            "recall_resolved": rec[2], "recall_persist": rec[3],
            "macro_F1": macro_f1,
            "AUROC_onset_vs_rest": ovr(1), "AUROC_resolved_vs_rest": ovr(2),
            "support_onset": int((yt == 1).sum()), "support_resolved": int((yt == 2).sum()),
        })
    return pd.DataFrame(rows)


def summarize_selection(metrics_df: pd.DataFrame) -> Dict[str, float]:
    """Selection scalar (tier-1 macro-F1) plus a companion AUROC for printing."""
    t1 = metrics_df[metrics_df["label"].isin(TIER1)]
    return {
        "tier1_macro_F1": float(t1["macro_F1"].mean()),
        "tier1_mean_onset_AUROC": float(t1["AUROC_onset_vs_rest"].mean()),
        "tier1_mean_resolved_AUROC": float(t1["AUROC_resolved_vs_rest"].mean()),
    }


# Dataset + model (module-level -> picklable for Windows DataLoader workers)
class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: "Config", train: bool):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.labels = cfg.labels
        trans = np.stack([pd.to_numeric(self.df[f"trans_{l}"], errors="coerce").to_numpy()
                          for l in self.labels], axis=1)
        self.mask = (~np.isnan(trans)).astype("float32")     # 0 where uncertain-endpoint masked
        self.trans = np.nan_to_num(trans, nan=0.0).astype("int64")
        self.first = self.df["first_image_path"].astype(str).tolist()
        self.last = self.df["last_image_path"].astype(str).tolist()
        aug = [T.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.96, 1.04),
                              interpolation=InterpolationMode.BILINEAR)] if train else []
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
        xf = self._load(self.first[idx])
        xl = self._load(self.last[idx])
        if self.cfg.mode == "last_only":
            xf = xl
        return {"first": xf, "last": xl,
                "trans": torch.from_numpy(self.trans[idx]),
                "mask": torch.from_numpy(self.mask[idx])}


class SiameseChange(nn.Module):
    def __init__(self, cfg: "Config"):
        super().__init__()
        base = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat = 1024
        self.trunk = nn.Sequential(
            nn.Linear(3 * feat, cfg.hidden_dim), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout))
        self.heads = nn.ModuleList([nn.Linear(cfg.hidden_dim, N_TRANS) for _ in cfg.labels])

    def encode(self, x):
        f = torch.relu(self.features(x))
        return self.pool(f).flatten(1)

    def forward(self, x_first, x_last):
        f1 = self.encode(x_first)
        f2 = self.encode(x_last)
        z = torch.cat([f1, f2, f2 - f1], dim=1)
        h = self.trunk(z)
        return torch.stack([head(h) for head in self.heads], dim=1)   # [B, L, 4]

    def load_encoder_transfer(self, ckpt_path: Path, map_location):
        ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        state = ck.get("model_state_dict", ck)
        enc = extract_encoder_state(state)
        missing, unexpected = self.features.load_state_dict(enc, strict=False)
        print(f"  encoder transfer from {ckpt_path}: loaded {len(enc)} tensors "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")


def masked_weighted_ce(logits, trans, mask, class_weights, device):
    """logits [B,L,4], trans [B,L], mask [B,L], class_weights list of [4] tensors."""
    B, L, C = logits.shape
    loss = torch.zeros((), device=device)
    denom = 0.0
    for i in range(L):
        m = mask[:, i].bool()
        if m.sum() == 0:
            continue
        loss = loss + F.cross_entropy(logits[m, i, :], trans[m, i], weight=class_weights[i], reduction="mean")
        denom += 1.0
    return loss / max(denom, 1.0)


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    ys, ps, ms = [], [], []
    for b in loader:
        xf = b["first"].to(device, non_blocking=True)
        xl = b["last"].to(device, non_blocking=True)
        with (torch.amp.autocast("cuda") if use_amp else nullcontext()):
            logits = model(xf, xl)
        ps.append(torch.softmax(logits.float(), dim=2).cpu().numpy())
        ys.append(b["trans"].numpy()); ms.append(b["mask"].numpy())
    return np.concatenate(ys, 0), np.concatenate(ps, 0), np.concatenate(ms, 0)


# Cross-validated training
def run_cv(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    df = pd.read_csv(cfg.pairs_csv)
    if "fold" not in df.columns:
        raise ValueError(f"{cfg.pairs_csv} needs a 'fold' column (run build_pairs.py).")

    print(f"arm={cfg.encoder_arm} | mode={cfg.mode} | seed={cfg.seed} | pairs={cfg.pairs_csv}")
    all_pred = []
    for fold in range(cfg.n_folds):
        print(f"\n===== FOLD {fold} =====")
        test_df = df[df["fold"] == fold]
        trainval = df[df["fold"] != fold]
        val_df = trainval.sample(frac=cfg.val_frac_of_train, random_state=cfg.seed)
        train_df = trainval.drop(val_df.index)

        train_ds = PairDataset(train_df, cfg, train=True)
        val_ds = PairDataset(val_df, cfg, train=False)
        test_ds = PairDataset(test_df, cfg, train=False)

        cw = compute_class_weights(train_ds.trans, train_ds.mask, cfg.labels, cfg.max_class_weight)
        class_weights = [torch.tensor(cw[i], dtype=torch.float32, device=device) for i in range(len(cfg.labels))]

        pin = device.type == "cuda"
        mk = lambda ds, sh: DataLoader(ds, batch_size=cfg.batch_size, shuffle=sh, num_workers=cfg.num_workers,
                                       pin_memory=pin, persistent_workers=cfg.num_workers > 0,
                                       prefetch_factor=4 if cfg.num_workers > 0 else None)
        train_loader, val_loader, test_loader = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

        model = SiameseChange(cfg).to(device)
        enc_path = resolve_encoder_path(cfg, fold)
        if enc_path is None:
            print("  encoder init: ImageNet (arm=imagenet)")
        else:
            if not enc_path.exists():
                raise FileNotFoundError(f"Encoder checkpoint not found: {enc_path}. "
                                        f"Train it first (train_encoder.py) or check --encoder-arm/root.")
            model.load_encoder_transfer(enc_path, device)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        total_steps = max(1, len(train_loader) * cfg.epochs)
        warmup = max(1, int(total_steps * cfg.warmup_frac))
        def lr_lambda(s):
            if s < warmup: return (s + 1) / warmup
            p = min(1.0, (s - warmup) / max(1, total_steps - warmup))
            return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_sel, best_state, no_improve = -1.0, None, 0
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            for b in tqdm(train_loader, desc=f"fold{fold} epoch {epoch}/{cfg.epochs}", leave=False):
                xf = b["first"].to(device, non_blocking=True)
                xl = b["last"].to(device, non_blocking=True)
                tr = b["trans"].to(device, non_blocking=True)
                mk_ = b["mask"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with (torch.amp.autocast("cuda") if use_amp else nullcontext()):
                    loss = masked_weighted_ce(model(xf, xl), tr, mk_, class_weights, device)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(opt); scaler.update(); sched.step()

            vy, vp, vm = predict(model, val_loader, device, use_amp)
            sel = summarize_selection(transition_metrics(vy, vp, vm, cfg.labels))
            print(f"  epoch {epoch:2d}  val tier1 macro-F1={sel['tier1_macro_F1']:.4f}  "
                  f"| onset-AUROC={sel['tier1_mean_onset_AUROC']:.4f}  "
                  f"resolved-AUROC={sel['tier1_mean_resolved_AUROC']:.4f}")
            if sel["tier1_macro_F1"] > best_sel + 1e-4:
                best_sel, no_improve = sel["tier1_macro_F1"], 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= cfg.early_stopping_patience:
                    print("  early stop"); break

        if best_state is not None:
            model.load_state_dict(best_state)
        # save per-fold model (for explain_change.py) + held-out predictions
        torch.save({"model_state_dict": model.state_dict(),
                    "cfg": {k: str(v) for k, v in asdict(cfg).items()},
                    "fold": fold, "labels": cfg.labels},
                   cfg.out_dir / f"fold{fold}_model.pt")
        ty, tp, tm = predict(model, test_loader, device, use_amp)
        np.savez(cfg.out_dir / f"fold{fold}_test.npz", y=ty, probs=tp, mask=tm,
                 subject_id=test_df["subject_id"].to_numpy())
        all_pred.append((ty, tp, tm))

    Y = np.concatenate([a[0] for a in all_pred], 0)
    P = np.concatenate([a[1] for a in all_pred], 0)
    M = np.concatenate([a[2] for a in all_pred], 0)
    pooled = transition_metrics(Y, P, M, cfg.labels)
    pooled.insert(0, "tier", ["Tier1" if l in TIER1 else "Tier2" for l in pooled["label"]])
    pooled.to_csv(cfg.out_dir / "pooled_metrics.csv", index=False)

    with open(cfg.out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"encoder_arm": cfg.encoder_arm, "mode": cfg.mode, "seed": cfg.seed,
                   "pairs_csv": str(cfg.pairs_csv), "n_folds": cfg.n_folds,
                   "encoder_root": str(cfg.encoder_root)}, f, indent=2)

    print(f"\nPooled metrics ({cfg.encoder_arm}/{cfg.mode}) — Tier-1:")
    print(pooled[pooled["tier"] == "Tier1"][
        ["label", "macro_F1", "recall_onset", "recall_resolved",
         "AUROC_onset_vs_rest", "AUROC_resolved_vs_rest"]].to_string(index=False))
    print("\nSaved to", cfg.out_dir.resolve())


def build_arg_parser():
    p = argparse.ArgumentParser(description="Siamese change-detection CV trainer (3 encoder arms).")
    p.add_argument("--encoder-arm", default="fold_nested", choices=["imagenet", "disjoint", "fold_nested"])
    p.add_argument("--mode", default="siamese", choices=["siamese", "last_only"])
    p.add_argument("--pairs-csv", default=None, help="e.g. pairs_out/pairs_gap180.csv")
    p.add_argument("--encoder-root", default=None)
    p.add_argument("--nested-prefix", default=None, help="e.g. nested_gap180_fold")
    p.add_argument("--disjoint-subdir", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    args = build_arg_parser().parse_args()
    cfg = Config()
    cfg.encoder_arm = args.encoder_arm
    cfg.mode = args.mode
    cfg.out_dir = Path(args.out_dir)
    if args.pairs_csv is not None: cfg.pairs_csv = Path(args.pairs_csv)
    if args.encoder_root is not None: cfg.encoder_root = Path(args.encoder_root)
    if args.nested_prefix is not None: cfg.nested_prefix = args.nested_prefix
    if args.disjoint_subdir is not None: cfg.disjoint_subdir = args.disjoint_subdir
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.seed is not None: cfg.seed = args.seed
    run_cv(cfg)
