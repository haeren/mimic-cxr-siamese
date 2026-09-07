from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# Canonical 14-label order (must match train_change_model.ALL_LABELS).
ALL_LABELS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
              "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
              "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
              "Support Devices"]
TIER1 = ["No Finding", "Support Devices", "Cardiomegaly", "Lung Opacity",
         "Atelectasis", "Pleural Effusion", "Edema"]
CLASS = {"onset": 1, "resolved": 2}   # 4-class transition codes


# Discovery + loading
def discover_runs(runs_root: Path):
    """Find runs by run_config.json; return list of dicts {arm, mode, seed, dir}."""
    runs = []
    for cfg_path in runs_root.rglob("run_config.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        arm = cfg.get("encoder_arm") or cfg.get("arm") or cfg.get("--encoder-arm")
        mode = cfg.get("mode") or cfg.get("--mode")
        seed = cfg.get("seed", cfg.get("--seed"))
        if arm is None or mode is None:
            continue
        runs.append({"arm": arm, "mode": mode, "seed": seed, "dir": cfg_path.parent})
    return runs


def load_run_concat(run_dir: Path):
    """Concatenate all fold*_test.npz in a run dir -> y, probs, mask, subj."""
    ys, ps, ms, ss = [], [], [], []
    for npz_path in sorted(run_dir.glob("fold*_test.npz")):
        d = np.load(npz_path, allow_pickle=True)
        ys.append(d["y"]); ps.append(d["probs"]); ms.append(d["mask"])
        ss.append(d["subject_id"])
    if not ys:
        return None
    return (np.concatenate(ys), np.concatenate(ps),
            np.concatenate(ms), np.concatenate(ss))


def seed_average(run_dirs):
    """Average softmax probs across seeds, aligned by subject_id.
    Returns (subj_order, y, mask, mean_probs). y/mask taken from the first run
    (identical across seeds since the data/labels are the same)."""
    per_seed = []
    for rd in run_dirs:
        loaded = load_run_concat(rd)
        if loaded is not None:
            per_seed.append(loaded)
    if not per_seed:
        return None
    # common subject order from the first seed
    y0, p0, m0, s0 = per_seed[0]
    order = list(s0)
    idx_of = {s: i for i, s in enumerate(order)}
    prob_stack = [p0]
    for (y, p, m, s) in per_seed[1:]:
        if set(s) != set(order) or len(s) != len(order):
            raise ValueError(
                f"Seed subject set mismatch: seed has {len(s)} subjects, "
                f"expected {len(order)}. All seeds must cover the same patients."
            )
        remap = np.array([idx_of[sid] for sid in s])   # align this seed to order
        p_aligned = np.empty_like(p0)
        p_aligned[remap] = p
        prob_stack.append(p_aligned)
    mean_probs = np.mean(prob_stack, axis=0)
    return np.array(order), y0, m0, mean_probs


# Metrics
def _ovr(y_col, score, mask_col, positive_class, metric):
    keep = mask_col.astype(bool)
    labels = (y_col[keep] == positive_class).astype(int)
    scores = score[keep]
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    return (roc_auc_score(labels, scores) if metric == "auroc"
            else average_precision_score(labels, scores))


def paired_bootstrap(y, mask, sia_probs, base_probs, i, positive_class, metric,
                     n_boot, rng):
    """Return observed (sia, base, diff) and bootstrap CI + p for the difference.
    Paired: the same resampled patients are used for both models."""
    sia_s = sia_probs[:, i, positive_class]
    base_s = base_probs[:, i, positive_class]
    yc, mc = y[:, i], mask[:, i]
    obs_sia = _ovr(yc, sia_s, mc, positive_class, metric)
    obs_base = _ovr(yc, base_s, mc, positive_class, metric)
    obs_diff = obs_sia - obs_base

    N = len(yc)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        ds = _ovr(yc[idx], sia_s[idx], mc[idx], positive_class, metric)
        db = _ovr(yc[idx], base_s[idx], mc[idx], positive_class, metric)
        diffs[b] = ds - db
    diffs = diffs[~np.isnan(diffs)]

    n_valid = int(mc.astype(bool).sum())
    n_pos = int((yc[mc.astype(bool)] == positive_class).sum())

    if len(diffs) == 0:
        return obs_sia, obs_base, obs_diff, np.nan, np.nan, np.nan, n_valid, n_pos
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = float(min(max(p, 1.0 / (len(diffs) + 1)), 1.0))   # floor so it's never exactly 0
    return obs_sia, obs_base, obs_diff, lo, hi, p, n_valid, n_pos


# Multiple-comparison corrections
def holm(pvals):
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * n / (rank + 1)
        prev = min(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj


def process_arm(arm, runs, n_boot, seed, out_dir):
    sia_dirs = [r["dir"] for r in runs if r["mode"] == "siamese"]
    base_dirs = [r["dir"] for r in runs if r["mode"] == "last_only"]
    if not sia_dirs or not base_dirs:
        print(f"  [{arm}] missing siamese or last_only runs; skipping.")
        return None

    sia = seed_average(sia_dirs)
    base = seed_average(base_dirs)
    if sia is None or base is None:
        print(f"  [{arm}] no npz found; skipping.")
        return None

    subj_s, y_s, mask_s, sia_probs = sia
    subj_b, y_b, mask_b, base_probs = base
    # align baseline to siamese subject order
    idx_of_b = {s: i for i, s in enumerate(subj_b)}
    remap = np.array([idx_of_b[s] for s in subj_s])
    base_probs = base_probs[remap]
    # y/mask must match; use siamese's (identical data)
    y, mask = y_s, mask_s

    rng = np.random.default_rng(seed)
    rows = []
    for cls_name, cls_code in CLASS.items():
        for metric in ("auroc", "auprc"):
            recs = []
            for fam_finding in TIER1:
                i = ALL_LABELS.index(fam_finding)
                res = paired_bootstrap(y, mask, sia_probs, base_probs, i, cls_code,
                                       metric, n_boot, rng)
                osia, obase, odiff, lo, hi, p = res[0], res[1], res[2], res[3], res[4], res[5]
                nv, npos = (res[6], res[7]) if len(res) > 6 else (np.nan, np.nan)
                recs.append({"arm": arm, "class": cls_name, "metric": metric,
                             "finding": fam_finding, "siamese": osia, "baseline": obase,
                             "diff": odiff, "ci_lo": lo, "ci_hi": hi, "p_raw": p,
                             "n_valid": nv, "n_pos": npos})
            # Holm + BH across the 7 Tier-1 findings within this (class, metric) family
            praw = [r["p_raw"] for r in recs]
            ph, pb = holm(praw), benjamini_hochberg(praw)
            for r, a, b2 in zip(recs, ph, pb):
                r["p_holm"], r["p_bh"] = float(a), float(b2)
            rows.extend(recs)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"stats_bootstrap_{arm}.csv", index=False)
    return df


def format_summary(dfs_by_arm, headline="fold_nested"):
    order = [headline] + [a for a in dfs_by_arm if a != headline]
    lines = ["PATIENT-LEVEL BOOTSTRAP — siamese - baseline (paired), 95% CI, "
             "Holm/BH across 7 Tier-1 findings", "=" * 88]
    for arm in order:
        df = dfs_by_arm.get(arm)
        if df is None:
            continue
        tag = " (headline)" if arm == headline else ""
        for cls_name in ("resolved", "onset"):
            for metric in ("auroc", "auprc"):
                sub = df[(df["class"] == cls_name) & (df["metric"] == metric)]
                if sub.empty:
                    continue
                lines.append(f"\n[{arm}{tag}] {cls_name} — {metric.upper()}")
                lines.append(f"  {'finding':<20} {'sia':>6} {'base':>6} {'diff':>7} "
                             f"{'95% CI':>17} {'p_holm':>8} {'p_bh':>8}")
                for _, r in sub.iterrows():
                    ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
                    lines.append(f"  {r['finding']:<20} {r['siamese']:>6.3f} "
                                 f"{r['baseline']:>6.3f} {r['diff']:>+7.3f} {ci:>17} "
                                 f"{r['p_holm']:>8.4f} {r['p_bh']:>8.4f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Patient-level bootstrap + AUPRC + correction.")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out-dir", default="stats_bootstrap_out")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="Restrict to these arms (default: all found).")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(runs_root)
    if not runs:
        raise SystemExit(f"No runs (run_config.json) found under {runs_root.resolve()}")
    by_arm = defaultdict(list)
    for r in runs:
        by_arm[r["arm"]].append(r)
    arms = args.arms or sorted(by_arm)

    print(f"Found arms: {sorted(by_arm)}; processing: {arms}; B={args.n_boot}")
    dfs = {}
    for arm in arms:
        print(f"[{arm}] seed-averaging and bootstrapping...")
        df = process_arm(arm, by_arm[arm], args.n_boot, args.seed, out_dir)
        if df is not None:
            dfs[arm] = df

    if not dfs:
        raise SystemExit("No arm produced results (need both siamese and last_only runs).")

    summary = format_summary(dfs)
    (out_dir / "stats_bootstrap_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nWrote per-arm CSVs and {out_dir / 'stats_bootstrap_summary.txt'}")


if __name__ == "__main__":
    main()
