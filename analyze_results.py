from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score

try:
    from scipy import stats as _stats
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


ALL_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
TIER1 = ["Lung Opacity", "Support Devices", "No Finding", "Atelectasis",
         "Cardiomegaly", "Pleural Effusion", "Edema"]
N_TRANS = 4
METRICS = ["macro_F1", "recall_onset", "recall_resolved",
           "AUROC_onset_vs_rest", "AUROC_resolved_vs_rest"]


# Metrics (torch-free; mirrors train_change_model.transition_metrics)
def pooled_predictions(run_dir: Path):
    """Concatenate fold*_test.npz -> (Y, P, M). Every pair appears once."""
    npzs = sorted(run_dir.glob("fold*_test.npz"))
    if not npzs:
        raise FileNotFoundError(f"No fold*_test.npz in {run_dir}")
    Y, P, M = [], [], []
    for f in npzs:
        d = np.load(f)
        Y.append(d["y"]); P.append(d["probs"]); M.append(d["mask"])
    return np.concatenate(Y), np.concatenate(P), np.concatenate(M)


def per_finding_metrics(Y: np.ndarray, P: np.ndarray, M: np.ndarray, labels: List[str]) -> pd.DataFrame:
    rows = []
    for i, lab in enumerate(labels):
        m = M[:, i].astype(bool)
        yt = Y[m, i].astype(int)
        pr = P[m, i, :]
        if len(yt) == 0:
            continue
        yhat = pr.argmax(axis=1)
        rec = recall_score(yt, yhat, labels=list(range(N_TRANS)), average=None, zero_division=0)
        mf1 = f1_score(yt, yhat, labels=list(range(N_TRANS)), average="macro", zero_division=0)

        def ovr(c):
            yb = (yt == c).astype(int)
            if 0 < yb.sum() < len(yb):
                return roc_auc_score(yb, pr[:, c])
            return np.nan

        rows.append({
            "label": lab, "n": int(len(yt)),
            "macro_F1": mf1, "recall_onset": rec[1], "recall_resolved": rec[2],
            "AUROC_onset_vs_rest": ovr(1), "AUROC_resolved_vs_rest": ovr(2),
            "support_onset": int((yt == 1).sum()), "support_resolved": int((yt == 2).sum()),
        })
    return pd.DataFrame(rows)


# Run discovery
def discover_runs(runs_root: Path) -> List[Dict]:
    runs = []
    for cfgp in sorted(runs_root.rglob("run_config.json")):
        d = cfgp.parent
        if not list(d.glob("fold*_test.npz")):
            continue
        cfg = json.loads(cfgp.read_text())
        runs.append({
            "dir": d,
            "arm": cfg.get("encoder_arm", "?"),
            "mode": cfg.get("mode", "?"),
            "seed": int(cfg.get("seed", -1)),
        })
    if not runs:
        raise FileNotFoundError(f"No runs with run_config.json + fold*_test.npz under {runs_root}")
    return runs


# Aggregation
def build_per_run_table(runs: List[Dict]) -> pd.DataFrame:
    frames = []
    for r in runs:
        Y, P, M = pooled_predictions(r["dir"])
        met = per_finding_metrics(Y, P, M, ALL_LABELS)
        met.insert(0, "arm", r["arm"]); met.insert(1, "mode", r["mode"])
        met.insert(2, "seed", r["seed"])
        met["tier"] = ["Tier1" if l in TIER1 else "Tier2" for l in met["label"]]
        frames.append(met)
    return pd.concat(frames, ignore_index=True)


def _ci95(vals: np.ndarray):
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(vals.mean())
    if n == 1:
        return mean, np.nan, 1
    sd = float(vals.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = _stats.t.ppf(0.975, n - 1) if _HAVE_SCIPY else 1.96
    return mean, tcrit * se, n


def cross_seed_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, mode, label), g in per_run.groupby(["arm", "mode", "label"]):
        for metric in METRICS:
            mean, half_ci, n = _ci95(g[metric].to_numpy(dtype=float))
            rows.append({
                "arm": arm, "mode": mode, "label": label, "metric": metric,
                "mean": mean, "ci95_halfwidth": half_ci, "n_seeds": n,
            })
    return pd.DataFrame(rows)


def paired_ttest(diffs: np.ndarray):
    """Two-sided paired t-test on the per-seed differences. Returns (mean_diff, p, n)."""
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(diffs.mean())
    if n == 1:
        return mean, np.nan, 1
    if _HAVE_SCIPY:
        t, p = _stats.ttest_1samp(diffs, 0.0)
        return mean, float(p), n
    # fallback: report mean and NaN p
    return mean, np.nan, n


def siamese_vs_baseline(per_run: pd.DataFrame) -> pd.DataFrame:
    """Within each arm: paired (siamese - last_only) difference across seeds."""
    rows = []
    for arm in sorted(per_run["arm"].unique()):
        sub = per_run[per_run["arm"] == arm]
        for label in sorted(sub["label"].unique()):
            for metric in METRICS:
                siam = sub[(sub["mode"] == "siamese") & (sub["label"] == label)][["seed", metric]]
                last = sub[(sub["mode"] == "last_only") & (sub["label"] == label)][["seed", metric]]
                merged = siam.merge(last, on="seed", suffixes=("_siam", "_last"))
                if merged.empty:
                    continue
                diffs = (merged[f"{metric}_siam"] - merged[f"{metric}_last"]).to_numpy(dtype=float)
                mean_diff, p, n = paired_ttest(diffs)
                siam_vals = merged[f"{metric}_siam"].to_numpy(dtype=float)
                last_vals = merged[f"{metric}_last"].to_numpy(dtype=float)
                siam_mean = float(np.nanmean(siam_vals)) if not np.all(np.isnan(siam_vals)) else np.nan
                last_mean = float(np.nanmean(last_vals)) if not np.all(np.isnan(last_vals)) else np.nan
                rows.append({
                    "arm": arm, "label": label, "metric": metric,
                    "siamese_mean": siam_mean, "last_only_mean": last_mean,
                    "delta_siam_minus_last": mean_diff, "paired_p": p, "n_pairs": n,
                })
    return pd.DataFrame(rows)


def arm_comparison_siamese(per_run: pd.DataFrame) -> pd.DataFrame:
    """Siamese-only: mean metric per arm, to show the effect across encoder regimes."""
    sub = per_run[per_run["mode"] == "siamese"]
    rows = []
    for label in sorted(sub["label"].unique()):
        for metric in METRICS:
            row = {"label": label, "metric": metric}
            for arm in sorted(sub["arm"].unique()):
                vals = sub[(sub["arm"] == arm) & (sub["label"] == label)][metric].to_numpy(dtype=float)
                row[arm] = float(np.nanmean(vals)) if (len(vals) and not np.all(np.isnan(vals))) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Aggregate change-model runs.")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out-dir", default="analysis_out")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(Path(args.runs_root))
    print(f"Discovered {len(runs)} runs:")
    for r in runs:
        print(f"  {r['arm']:11s} {r['mode']:9s} seed={r['seed']:3d}  ({r['dir']})")

    per_run = build_per_run_table(runs)
    per_run.to_csv(out / "per_run_finding_metrics.csv", index=False)

    summary = cross_seed_summary(per_run)
    summary.to_csv(out / "cross_seed_summary.csv", index=False)

    svb = siamese_vs_baseline(per_run)
    svb.to_csv(out / "siamese_vs_baseline.csv", index=False)

    armc = arm_comparison_siamese(per_run)
    armc.to_csv(out / "arm_comparison_siamese.csv", index=False)

    # ---- readable headline ----
    lines = ["CHANGE-MODEL RESULTS SUMMARY", "=" * 60]
    if not _HAVE_SCIPY:
        lines.append("(scipy not found: CIs use 1.96 and paired p-values are omitted)")
    lines.append("")
    lines.append("SIAMESE vs LAST_ONLY on resolved-vs-rest AUROC (Tier-1 findings):")
    key = svb[(svb["metric"] == "AUROC_resolved_vs_rest") & (svb["label"].isin(TIER1))]
    for arm in sorted(key["arm"].unique()):
        lines.append(f"  [arm = {arm}]")
        for _, r in key[key["arm"] == arm].sort_values("label").iterrows():
            p = "" if np.isnan(r["paired_p"]) else f" p={r['paired_p']:.4f}"
            lines.append(f"    {r['label']:20s} siam {r['siamese_mean']:.3f} vs last "
                         f"{r['last_only_mean']:.3f}  (Δ={r['delta_siam_minus_last']:+.3f}{p})")
        lines.append("")
    lines.append("ARM COMPARISON (siamese, resolved-AUROC, Tier-1) — effect should hold across arms:")
    ac = armc[(armc["metric"] == "AUROC_resolved_vs_rest") & (armc["label"].isin(TIER1))]
    arm_cols = [c for c in ac.columns if c not in ("label", "metric")]
    lines.append("  " + "label".ljust(20) + "  " + "  ".join(a.ljust(12) for a in arm_cols))
    for _, r in ac.sort_values("label").iterrows():
        lines.append("  " + r["label"].ljust(20) + "  " +
                     "  ".join((f"{r[a]:.3f}" if not np.isnan(r[a]) else "  -  ").ljust(12) for a in arm_cols))
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote 4 tables + summary.txt to {out.resolve()}")


if __name__ == "__main__":
    main()
