from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

import torch
import torchvision.transforms as T

# Reuse the exact architecture + constants from the trainer (no duplication).
from train_change_model import (
    SiameseChange, Config, ALL_LABELS, N_TRANS,
    IMAGENET_MEAN, IMAGENET_STD,
)

CLASS_TO_CODE = {"none": 0, "onset": 1, "resolved": 2, "persist": 3}
CODE_TO_NAME = {0: "0to0", 1: "onset", 2: "resolved", 3: "persist"}


# CAM math
def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2-D map to [0, 1]; flat maps -> all zeros."""
    cam = cam.astype("float64")
    lo, hi = float(cam.min()), float(cam.max())
    if hi - lo < 1e-12:
        return np.zeros_like(cam)
    return (cam - lo) / (hi - lo)


def cam_weights_from_grad(activation: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """
    Grad-CAM: alpha_k = spatial mean of gradients for channel k; the map is
    ReLU(sum_k alpha_k * A_k). `activation`/`grad` are [C, H, W]. Returns [H, W].
    """
    alpha = grad.mean(axis=(1, 2))                 # [C]
    cam = np.maximum((alpha[:, None, None] * activation).sum(axis=0), 0.0)
    return cam


def overlay_heatmap(base_img01: np.ndarray, cam01: np.ndarray, alpha: float = 0.45,
                    colormap: str = "jet") -> np.ndarray:
    """
    Blend a [0,1] heatmap over a [0,1] RGB image (both HxWx3 / HxW). cam is
    upsampled to the image size by the caller. Returns HxWx3 uint8.
    """
    cmap = cm.get_cmap(colormap)
    heat = cmap(cam01)[..., :3]                    # HxWx3 in [0,1]
    blended = (1 - alpha) * base_img01 + alpha * heat
    return (np.clip(blended, 0, 1) * 255).astype("uint8")


def denormalize_to_01(chw_tensor: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalization on a [3,H,W] array -> HxWx3 in [0,1]."""
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std = np.array(IMAGENET_STD)[:, None, None]
    img = chw_tensor * std + mean
    return np.clip(np.transpose(img, (1, 2, 0)), 0, 1)


# Grad-CAM on the Siamese model
def _cam_from_feature(feat: "torch.Tensor") -> np.ndarray:
    grad = feat.grad[0].detach().cpu().numpy()      # [C,H,W]
    act = feat[0].detach().cpu().numpy()            # [C,H,W]
    return normalize_cam(cam_weights_from_grad(act, grad))


def gradcam_pair(model, x_first, x_last, finding_idx: int, trans_code: int, device):
    """Standard Grad-CAM through the FULL fusion. Returns (cam_first, cam_last)."""
    model.eval()
    model.zero_grad(set_to_none=True)
    xf = x_first.to(device); xl = x_last.to(device)
    feat_f = model.features(xf); feat_f.retain_grad()
    feat_l = model.features(xl); feat_l.retain_grad()
    f1 = model.pool(torch.relu(feat_f)).flatten(1)
    f2 = model.pool(torch.relu(feat_l)).flatten(1)
    z = torch.cat([f1, f2, f2 - f1], dim=1)
    logit = model.heads[finding_idx](model.trunk(z))[0, trans_code]
    logit.backward()
    return _cam_from_feature(feat_f), _cam_from_feature(feat_l)


def gradcam_change(model, x_first, x_last, finding_idx: int, trans_code: int, device):
    """
    Change-pathway Grad-CAM: gradient flows ONLY through the (f_last - f_first)
    difference block (the two concatenation blocks are detached), so the map
    attributes the prediction to the change signal. Returns (cam_first, cam_last).
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    xf = x_first.to(device); xl = x_last.to(device)
    feat_f = model.features(xf); feat_f.retain_grad()
    feat_l = model.features(xl); feat_l.retain_grad()
    f1 = model.pool(torch.relu(feat_f)).flatten(1)
    f2 = model.pool(torch.relu(feat_l)).flatten(1)
    diff = f2 - f1
    # detach the absolute-state blocks -> gradient enters only via `diff`
    z = torch.cat([f1.detach(), f2.detach(), diff], dim=1)
    logit = model.heads[finding_idx](model.trunk(z))[0, trans_code]
    logit.backward()
    return _cam_from_feature(feat_f), _cam_from_feature(feat_l)


# I/O helpers
def load_change_model(model_path: Path, device):
    ck = torch.load(model_path, map_location=device, weights_only=False)
    cfg = Config()
    # restore architecture-relevant settings if present in the checkpoint
    saved = ck.get("cfg", {})
    for k in ("hidden_dim", "dropout"):
        if k in saved:
            try:
                setattr(cfg, k, type(getattr(cfg, k))(saved[k]))
            except Exception:
                pass
    model = SiameseChange(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    labels = ck.get("labels", list(ALL_LABELS))
    return model, labels


def load_image(path: str, cfg_cache_root: Path, image_size: int) -> Tuple["torch.Tensor", np.ndarray]:
    p = Path(path)
    if not p.exists():
        alt = cfg_cache_root / p
        if alt.exists():
            p = alt
    tf = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    img = Image.open(p).convert("RGB")
    x = tf(img).unsqueeze(0)                        # [1,3,H,W]
    base01 = np.asarray(img.resize((image_size, image_size))).astype("float64") / 255.0
    return x, base01


def upsample_cam(cam01: np.ndarray, size: int) -> np.ndarray:
    """Bilinear upsample a small CAM to size x size using PIL, renormalized."""
    im = Image.fromarray((cam01 * 255).astype("uint8")).resize((size, size), Image.BILINEAR)
    return normalize_cam(np.asarray(im).astype("float64"))


# Column order (left to right); standardized titles (all first-letter capitalized).
COL_TITLES = ["First Raw", "First Grad-CAM", "Last Grad-CAM", "Last Raw",
              "First Change-CAM", "Last Change-CAM"]
# Default rows for the suggested combined figure.
COMBINED_FINDINGS = ["Edema", "Pleural Effusion", "Atelectasis"]
# Tier-1 findings (well-supported onset AND resolved at 180 days): 1x6 panels are
# generated for all of these.
TIER1 = ["No Finding", "Support Devices", "Cardiomegaly", "Lung Opacity",
         "Atelectasis", "Pleural Effusion", "Edema"]


def six_panels(model, pk, i, code, S, device) -> List[np.ndarray]:
    """The six display panels (uint8 HxWx3) in COL_TITLES order for one example."""
    cam_f, cam_l = gradcam_pair(model, pk["xf"], pk["xl"], i, code, device)
    ch_f, ch_l = gradcam_change(model, pk["xf"], pk["xl"], i, code, device)
    bf, bl = pk["base_f"], pk["base_l"]
    return [
        (bf * 255).astype("uint8"),
        overlay_heatmap(bf, upsample_cam(cam_f, S)),
        overlay_heatmap(bl, upsample_cam(cam_l, S)),
        (bl * 255).astype("uint8"),
        overlay_heatmap(bf, upsample_cam(ch_f, S)),
        overlay_heatmap(bl, upsample_cam(ch_l, S)),
    ]


def save_sixpanel(panels: List[np.ndarray], out_path: Path):
    """One example as a 1x6 strip: column titles only, tight spacing, no divider,
    no per-figure text (that lives in explanations_index.csv)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 6, figsize=(11, 2.05),
                             gridspec_kw={"wspace": 0.05})
    for ax, panel, title in zip(axes, panels, COL_TITLES):
        ax.imshow(panel); ax.set_title(title, fontsize=9); ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_combined(rows: List[Tuple[str, List[np.ndarray]]], out_base: str):
    """rows: list of (row_label, panels[6]). N x 6 figure with column titles on the
    top row and row labels on the left. No group headers, no vertical divider."""
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 6, figsize=(11, 1.95 * n),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.08})
    if n == 1:
        axes = axes[None, :]
    for rj, (label, panels) in enumerate(rows):
        for cj, panel in enumerate(panels):
            ax = axes[rj, cj]
            ax.imshow(panel); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if rj == 0:
                ax.set_title(COL_TITLES[cj], fontsize=10)
        axes[rj, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=8)
        axes[rj, 0].yaxis.set_visible(True); axes[rj, 0].set_yticks([])
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.02)
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close()


# Example selection: confident, correct held-out pairs for a (finding, class)
def select_examples(model, fold_df: pd.DataFrame, labels: List[str], finding: str,
                    trans_code: int, n: int, cfg, device) -> List[Dict]:
    """Pick up to n test pairs whose TRUE transition == trans_code, ranked by the
    model's predicted probability for that class (most confident first)."""
    i = labels.index(finding)
    col = f"trans_{finding}"
    cand = fold_df[pd.to_numeric(fold_df[col], errors="coerce") == trans_code]
    picks = []
    for _, row in cand.iterrows():
        xf, base_f = load_image(row["first_image_path"], cfg.cache_root, cfg.image_size)
        xl, base_l = load_image(row["last_image_path"], cfg.cache_root, cfg.image_size)
        with torch.no_grad():
            prob = torch.softmax(model(xf.to(device), xl.to(device)).float(), dim=2)[0, i, trans_code].item()
        picks.append({"row": row, "prob": prob, "xf": xf, "xl": xl, "base_f": base_f, "base_l": base_l})
    picks.sort(key=lambda d: d["prob"], reverse=True)
    return picks[:n]


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM explanations for the Siamese change model.")
    ap.add_argument("--model", required=True, help="fold{K}_model.pt from train_change_model.py")
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--fold", type=int, required=True, help="This model's held-out fold.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--classes", nargs="+", default=["resolved"],
                    choices=list(CLASS_TO_CODE.keys()))
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--combined-findings", nargs="+", default=COMBINED_FINDINGS,
                    help="Findings (rows) for the suggested combined figure.")
    ap.add_argument("--combined-class", default="resolved", choices=list(CLASS_TO_CODE.keys()))
    ap.add_argument("--out-combined", default=None,
                    help="Combined figure basename (default <out-dir>/figure_xai).")
    ap.add_argument("--no-combined", dest="make_combined", action="store_false", default=True)
    ap.add_argument("--cache-root", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model, labels = load_change_model(Path(args.model), device)
    cfg = Config()
    if args.cache_root:
        cfg.cache_root = Path(args.cache_root)
    S = cfg.image_size

    df = pd.read_csv(args.pairs_csv)
    fold_df = df[df["fold"] == args.fold].copy()
    print(f"Explaining fold {args.fold}: {len(fold_df)} held-out pairs "
          f"(this model never trained on them).")

    index_rows = []
    # store every generated example so the combined figure can pick non-duplicate patients
    candidates: Dict[Tuple[str, str], List[Dict]] = {}
    for finding in TIER1:
        if finding not in labels:
            print(f"  skip '{finding}' (not a label)"); continue
        i = labels.index(finding)
        for cls in args.classes:
            code = CLASS_TO_CODE[cls]
            picks = select_examples(model, fold_df, labels, finding, code, args.examples, cfg, device)
            if not picks:
                print(f"  {finding}/{cls}: no examples"); continue
            for n, pk in enumerate(picks, 1):
                panels = six_panels(model, pk, i, code, S, device)
                safe = finding.replace(" ", "_").replace("/", "_")
                fname = f"{safe}_{cls}_ex{n}.png"
                save_sixpanel(panels, out_dir / fname)

                r = pk["row"]
                subj = r.get("subject_id", "")
                index_rows.append({
                    "finding": finding, "class": cls, "example": n,
                    "subject_id": subj, "pred_prob": round(pk["prob"], 4),
                    "delta_days": r.get("delta_days", ""),
                    "png": fname,
                })
                candidates.setdefault((finding, cls), []).append(
                    {"subject": subj, "prob": pk["prob"], "panels": panels})
                print(f"  {finding}/{cls} ex{n}: prob={pk['prob']:.3f} -> {fname}")

    pd.DataFrame(index_rows).to_csv(out_dir / "explanations_index.csv", index=False)
    print(f"\nWrote {len(index_rows)} panel PNG(s) + explanations_index.csv to {out_dir.resolve()}")

    # Suggested combined figure: one row per focal finding. For each finding, take the
    # most-confident example whose patient has not already been used in a previous row,
    # so the same (multi-label) patient never appears in two rows.
    if args.make_combined:
        rows, used = [], set()
        for finding in args.combined_findings:
            cands = candidates.get((finding, args.combined_class), [])
            if not cands:
                continue
            # candidates are already ordered most-confident first (from select_examples)
            pick = next((c for c in cands if c["subject"] not in used), cands[0])
            used.add(pick["subject"])
            rows.append((finding, pick["panels"], pick["subject"], pick["prob"]))
        if rows:
            out_base = args.out_combined or str(out_dir / "figure_xai")
            save_combined([(f, p) for f, p, _, _ in rows], out_base)
            chosen = ", ".join(f"{f} (subj {s}, p={pr:.2f})" for f, _, s, pr in rows)
            n_unique = len({s for _, _, s, _ in rows})
            print(f"Wrote combined figure {out_base}.png/.pdf  -  rows: {chosen}")
            if n_unique < len(rows):
                print("  (note: a patient repeated across rows — too few distinct "
                      "candidates; increase --examples.)")
        else:
            print("Combined figure skipped (no matching examples among --combined-findings).")


if __name__ == "__main__":
    main()
