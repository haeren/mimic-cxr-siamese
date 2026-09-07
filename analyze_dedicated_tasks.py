from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from analyze_stats_bootstrap import (
    discover_runs, seed_average, holm, benjamini_hochberg, ALL_LABELS, TIER1,
)

# task: (positive transition class, negative transition class, score class = positive)
TASKS = {
    "resolved_vs_absent": {"pos": 2, "neg": 0},   # 1->0 vs 0->0
    "onset_vs_persist":   {"pos": 1, "neg": 3},   # 0->1 vs 1->1
}


def _binary_metric(y_col, score, keep_mask, pos, neg, metric):
    """AUROC/AUPRC for pos-vs-neg on the restricted subset (true class in {pos,neg})."""
    sel = keep_mask & ((y_col == pos) | (y_col == neg))
    if sel.sum() == 0:
        return np.nan
    labels = (y_col[sel] == pos).astype(int)
    scores = score[sel]
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    return (roc_auc_score(labels, scores) if metric == "auroc"
            else average_precision_score(labels, scores))


def paired_bootstrap_binary(y, mask, sia_probs, base_probs, i, pos, neg, metric,
                            n_boot, rng):
    """Observed sia/base/diff and bootstrap CI+p for the pos-vs-neg task on finding i.
    Resamples over the restricted subset (patients whose true class is pos or neg)."""
    score_sia = sia_probs[:, i, pos]
    score_base = base_probs[:, i, pos]
    yc = y[:, i]
    keep = mask[:, i].astype(bool)
    sel = np.where(keep & ((yc == pos) | (yc == neg)))[0]     # restricted patient indices
    if len(sel) == 0:
        return dict(sia=np.nan, base=np.nan, diff=np.nan, lo=np.nan, hi=np.nan,
                    p=np.nan, n_pos=0, n_neg=0)

    full_keep = np.zeros(len(yc), bool); full_keep[sel] = True
    obs_sia = _binary_metric(yc, score_sia, full_keep, pos, neg, metric)
    obs_base = _binary_metric(yc, score_base, full_keep, pos, neg, metric)
    obs_diff = obs_sia - obs_base

    ys, ss, sb = yc[sel], score_sia[sel], score_base[sel]
    M = len(sel)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, M, M)
        yb = ys[idx]
        lab = (yb == pos).astype(int)
        if lab.sum() == 0 or lab.sum() == len(lab):
            diffs[b] = np.nan; continue
        if metric == "auroc":
            ds = roc_auc_score(lab, ss[idx]); db = roc_auc_score(lab, sb[idx])
        else:
            ds = average_precision_score(lab, ss[idx]); db = average_precision_score(lab, sb[idx])
        diffs[b] = ds - db
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return dict(sia=obs_sia, base=obs_base, diff=obs_diff, lo=np.nan, hi=np.nan,
                    p=np.nan, n_pos=int((ys == pos).sum()), n_neg=int((ys == neg).sum()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = float(min(max(p, 1.0 / (len(diffs) + 1)), 1.0))
    return dict(sia=obs_sia, base=obs_base, diff=obs_diff, lo=lo, hi=hi, p=p,
                n_pos=int((ys == pos).sum()), n_neg=int((ys == neg).sum()))


def process_arm(arm, runs, n_boot, seed, out_dir):
    sia_dirs = [r["dir"] for r in runs if r["mode"] == "siamese"]
    base_dirs = [r["dir"] for r in runs if r["mode"] == "last_only"]
    if not sia_dirs or not base_dirs:
        print(f"  [{arm}] missing siamese or last_only runs; skipping."); return None
    sia = seed_average(sia_dirs); base = seed_average(base_dirs)
    if sia is None or base is None:
        print(f"  [{arm}] no npz found; skipping."); return None

    subj_s, y, mask, sia_probs = sia
    subj_b, _, _, base_probs = base
    idx_of_b = {s: k for k, s in enumerate(subj_b)}
    base_probs = base_probs[np.array([idx_of_b[s] for s in subj_s])]

    rng = np.random.default_rng(seed)
    rows = []
    for task_name, cfg in TASKS.items():
        for metric in ("auroc", "auprc"):
            recs = []
            for finding in TIER1:
                i = ALL_LABELS.index(finding)
                r = paired_bootstrap_binary(y, mask, sia_probs, base_probs, i,
                                            cfg["pos"], cfg["neg"], metric, n_boot, rng)
                recs.append({"arm": arm, "task": task_name, "metric": metric,
                             "finding": finding, "siamese": r["sia"], "baseline": r["base"],
                             "diff": r["diff"], "ci_lo": r["lo"], "ci_hi": r["hi"],
                             "p_raw": r["p"], "n_pos": r["n_pos"], "n_neg": r["n_neg"]})
            praw = [x["p_raw"] for x in recs]
            for x, a, b2 in zip(recs, holm(praw), benjamini_hochberg(praw)):
                x["p_holm"], x["p_bh"] = float(a), float(b2)
            rows.extend(recs)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"dedicated_tasks_{arm}.csv", index=False)
    return df


def format_summary(dfs_by_arm, headline="fold_nested"):
    order = [headline] + [a for a in dfs_by_arm if a != headline]
    lines = ["DEDICATED BINARY TASKS — siamese vs baseline (paired patient bootstrap, "
             "95% CI, Holm/BH)", "=" * 84,
             "Task A resolved_vs_absent (1->0 vs 0->0): baseline near 0.5 = cannot separate "
             "from final image alone.", "=" * 84]
    for arm in order:
        df = dfs_by_arm.get(arm)
        if df is None:
            continue
        tag = " (headline)" if arm == headline else ""
        for task in TASKS:
            for metric in ("auroc", "auprc"):
                sub = df[(df["task"] == task) & (df["metric"] == metric)]
                if sub.empty:
                    continue
                lines.append(f"\n[{arm}{tag}] {task} — {metric.upper()}")
                lines.append(f"  {'finding':<20} {'sia':>6} {'base':>6} {'diff':>7} "
                             f"{'95% CI':>17} {'p_holm':>8} {'n_pos':>6} {'n_neg':>6}")
                for _, r in sub.iterrows():
                    ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
                    lines.append(f"  {r['finding']:<20} {r['siamese']:>6.3f} "
                                 f"{r['baseline']:>6.3f} {r['diff']:>+7.3f} {ci:>17} "
                                 f"{r['p_holm']:>8.4f} {int(r['n_pos']):>6} {int(r['n_neg']):>6}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Dedicated binary tasks for value-of-prior.")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out-dir", default="dedicated_tasks_out")
    ap.add_argument("--arms", nargs="+", default=None)
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
        print(f"[{arm}] seed-averaging and bootstrapping dedicated tasks...")
        df = process_arm(arm, by_arm[arm], args.n_boot, args.seed, out_dir)
        if df is not None:
            dfs[arm] = df
    if not dfs:
        raise SystemExit("No arm produced results (need both siamese and last_only runs).")

    summary = format_summary(dfs)
    (out_dir / "dedicated_tasks_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nWrote per-arm CSVs and {out_dir / 'dedicated_tasks_summary.txt'}")


if __name__ == "__main__":
    main()
