from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from analyze_stats_bootstrap import (
    discover_runs, seed_average, holm, benjamini_hochberg, ALL_LABELS, TIER1,
)
from analyze_dedicated_tasks import paired_bootstrap_binary, TASKS

BINS = [("0-30", 0, 30), ("31-90", 31, 90), ("91-180", 91, 180)]
BIN_ORDER = [b[0] for b in BINS]


def bin_of(days):
    try:
        d = float(days)
    except (TypeError, ValueError):
        return "excluded"
    for name, lo, hi in BINS:
        if lo <= d <= hi:
            return name
    return "excluded"


def build_gap_map(pairs_csv: Path):
    pairs = pd.read_csv(pairs_csv, usecols=["subject_id", "delta_days"])
    status = {int(s): bin_of(d) for s, d in zip(pairs["subject_id"], pairs["delta_days"])}
    counts = defaultdict(int)
    for v in status.values():
        counts[v] += 1
    return status, counts


def process_arm(arm, runs, gap_status, n_boot, seed, out_dir):
    sia_dirs = [r["dir"] for r in runs if r["mode"] == "siamese"]
    base_dirs = [r["dir"] for r in runs if r["mode"] == "last_only"]
    if not sia_dirs or not base_dirs:
        print(f"  [{arm}] missing siamese or last_only runs; skipping."); return None
    sia = seed_average(sia_dirs); base = seed_average(base_dirs)
    if sia is None or base is None:
        print(f"  [{arm}] no npz found; skipping."); return None

    subj, y, mask, sia_probs = sia
    subj_b, _, _, base_probs = base
    idx_of_b = {s: k for k, s in enumerate(subj_b)}
    base_probs = base_probs[np.array([idx_of_b[s] for s in subj])]

    gap = np.array([gap_status.get(int(s), "excluded") for s in subj])
    rng = np.random.default_rng(seed)
    rows = []
    for task_name, cfg in TASKS.items():
        for bin_name in BIN_ORDER:
            m_bin = mask.copy()
            m_bin[gap != bin_name] = 0.0
            for metric in ("auroc", "auprc"):
                recs = []
                for finding in TIER1:
                    i = ALL_LABELS.index(finding)
                    r = paired_bootstrap_binary(y, m_bin, sia_probs, base_probs, i,
                                                cfg["pos"], cfg["neg"], metric, n_boot, rng)
                    recs.append({"arm": arm, "task": task_name, "bin": bin_name,
                                 "metric": metric, "finding": finding,
                                 "siamese": r["sia"], "baseline": r["base"], "diff": r["diff"],
                                 "ci_lo": r["lo"], "ci_hi": r["hi"], "p_raw": r["p"],
                                 "n_pos": r["n_pos"], "n_neg": r["n_neg"]})
                praw = [x["p_raw"] if not np.isnan(x["p_raw"]) else 1.0 for x in recs]
                for x, a, b2 in zip(recs, holm(praw), benjamini_hochberg(praw)):
                    x["p_holm"], x["p_bh"] = float(a), float(b2)
                rows.extend(recs)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"time_stratified_{arm}.csv", index=False)
    return df


def format_summary(dfs_by_arm, counts, headline="fold_nested"):
    order = [headline] + [a for a in dfs_by_arm if a != headline]
    lines = ["TIME-INTERVAL STRATIFICATION — siamese - baseline diff by gap bin "
             "(paired bootstrap, 95% CI, Holm)", "=" * 92,
             "Stable diff across bins = value of prior imaging does not degrade with the gap.",
             ""]
    lines.append("Pair gap composition: " + ", ".join(f"{k}={counts[k]}" for k in
                 (BIN_ORDER + ["excluded"]) if k in counts))
    for arm in order:
        df = dfs_by_arm.get(arm)
        if df is None:
            continue
        tag = " (headline)" if arm == headline else ""
        for task in TASKS:
            for metric in ("auroc", "auprc"):
                lines.append(f"\n[{arm}{tag}] {task} — {metric.upper()}  (diff [95% CI]; n_pos)")
                header = f"  {'finding':<18}" + "".join(f"{b:>26}" for b in BIN_ORDER)
                lines.append(header)
                for finding in TIER1:
                    cells = []
                    for b in BIN_ORDER:
                        r = df[(df["task"] == task) & (df["metric"] == metric)
                               & (df["bin"] == b) & (df["finding"] == finding)]
                        if r.empty or np.isnan(r["diff"].iloc[0]):
                            cells.append(f"{'n/a':>26}")
                        else:
                            rr = r.iloc[0]
                            s = f"{rr['diff']:+.3f}[{rr['ci_lo']:+.2f},{rr['ci_hi']:+.2f}];{int(rr['n_pos'])}"
                            cells.append(f"{s:>26}")
                    lines.append(f"  {finding:<18}" + "".join(cells))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Time-interval stratification of dedicated tasks.")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--out-dir", default="time_stratified_out")
    ap.add_argument("--arms", nargs="+", default=["fold_nested"],
                    help="Arms to process (default headline fold_nested; pass more if wanted).")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gap_status, counts = build_gap_map(Path(args.pairs_csv))
    print("Gap composition:", dict(counts))

    runs = discover_runs(Path(args.runs_root))
    if not runs:
        raise SystemExit(f"No runs found under {args.runs_root}")
    by_arm = defaultdict(list)
    for r in runs:
        by_arm[r["arm"]].append(r)
    arms = [a for a in args.arms if a in by_arm] or sorted(by_arm)

    dfs = {}
    for arm in arms:
        print(f"[{arm}] time-stratified bootstrap...")
        df = process_arm(arm, by_arm[arm], gap_status, args.n_boot, args.seed, out_dir)
        if df is not None:
            dfs[arm] = df
    if not dfs:
        raise SystemExit("No arm produced results.")

    summary = format_summary(dfs, counts)
    (out_dir / "time_stratified_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nWrote per-arm CSVs and {out_dir / 'time_stratified_summary.txt'}")


if __name__ == "__main__":
    main()
