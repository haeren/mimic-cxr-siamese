from __future__ import annotations

import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from analyze_stats_bootstrap import (
    discover_runs, seed_average, holm, benjamini_hochberg, ALL_LABELS, TIER1,
)
from analyze_dedicated_tasks import paired_bootstrap_binary, TASKS

FRONTAL = {"PA", "AP"}
DICOM_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8})")


def dicom_from_path(p: str):
    """Extract the dicom_id from a cached image path (…/<dicom_id>.jpg)."""
    if not isinstance(p, str):
        return None
    stem = Path(p.replace("\\", "/")).stem   # handle Windows separators
    m = DICOM_RE.search(stem)
    return m.group(1) if m else (stem or None)


def build_projection_map(pairs_csv: Path, metadata_csv: Path):
    """Return subject_id -> 'same' / 'changed' / 'excluded' based on first/last ViewPosition.
    Also return counts. One pair per patient, keyed by subject_id."""
    pairs = pd.read_csv(pairs_csv)
    meta = pd.read_csv(metadata_csv, usecols=["dicom_id", "ViewPosition"])
    view_of = dict(zip(meta["dicom_id"].astype(str), meta["ViewPosition"].astype(str)))

    status = {}
    counts = defaultdict(int)
    for _, r in pairs.iterrows():
        subj = r["subject_id"]
        df_id = dicom_from_path(r.get("first_image_path"))
        dl_id = dicom_from_path(r.get("last_image_path"))
        vf = view_of.get(str(df_id), None)
        vl = view_of.get(str(dl_id), None)
        if vf not in FRONTAL or vl not in FRONTAL:
            status[subj] = "excluded"; counts["excluded"] += 1
            continue
        if vf == vl:
            status[subj] = "same"; counts[f"same_{vf}"] += 1; counts["same"] += 1
        else:
            status[subj] = "changed"; counts["changed"] += 1
            counts[f"{vf}->{vl}"] += 1
    return status, counts


def process_arm(arm, runs, proj_status, n_boot, seed, out_dir):
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

    # boolean masks over the aligned subject order
    proj = np.array([proj_status.get(s, "excluded") for s in subj])
    subsets = {"same_projection": proj == "same",
               "changed_projection": proj == "changed",
               "all_frontal": np.isin(proj, ["same", "changed"])}

    cfg = TASKS["resolved_vs_absent"]      # headline task only
    rng = np.random.default_rng(seed)
    rows = []
    for subset_name, subj_mask in subsets.items():
        # apply the subset by masking out non-subset patients (mask -> 0 there)
        m_sub = mask.copy()
        m_sub[~subj_mask] = 0.0
        for metric in ("auroc", "auprc"):
            recs = []
            for finding in TIER1:
                i = ALL_LABELS.index(finding)
                r = paired_bootstrap_binary(y, m_sub, sia_probs, base_probs, i,
                                            cfg["pos"], cfg["neg"], metric, n_boot, rng)
                recs.append({"arm": arm, "subset": subset_name, "metric": metric,
                             "finding": finding, "siamese": r["sia"], "baseline": r["base"],
                             "diff": r["diff"], "ci_lo": r["lo"], "ci_hi": r["hi"],
                             "p_raw": r["p"], "n_pos": r["n_pos"], "n_neg": r["n_neg"]})
            praw = [x["p_raw"] if not np.isnan(x["p_raw"]) else 1.0 for x in recs]
            for x, a, b2 in zip(recs, holm(praw), benjamini_hochberg(praw)):
                x["p_holm"], x["p_bh"] = float(a), float(b2)
            rows.extend(recs)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"projection_control_{arm}.csv", index=False)
    return df


def format_summary(dfs_by_arm, counts, headline="fold_nested"):
    order = [headline] + [a for a in dfs_by_arm if a != headline]
    lines = ["PROJECTION CONTROL — resolved_vs_absent (1->0 vs 0->0), siamese vs baseline",
             "=" * 80,
             "If the gain persists on same-projection pairs, AP/PA change cannot explain it.",
             ""]
    lines.append("Pair projection composition:")
    for k in sorted(counts):
        lines.append(f"  {k:<16} {counts[k]}")
    for arm in order:
        df = dfs_by_arm.get(arm)
        if df is None:
            continue
        tag = " (headline)" if arm == headline else ""
        for subset in ("same_projection", "changed_projection", "all_frontal"):
            for metric in ("auroc", "auprc"):
                sub = df[(df["subset"] == subset) & (df["metric"] == metric)]
                if sub.empty:
                    continue
                lines.append(f"\n[{arm}{tag}] {subset} — {metric.upper()}")
                lines.append(f"  {'finding':<20} {'sia':>6} {'base':>6} {'diff':>7} "
                             f"{'95% CI':>17} {'p_holm':>8} {'n_pos':>6} {'n_neg':>6}")
                for _, r in sub.iterrows():
                    if np.isnan(r["diff"]):
                        lines.append(f"  {r['finding']:<20} {'n/a':>6} {'n/a':>6} "
                                     f"{'n/a':>7} {'(too few)':>17} {'':>8} "
                                     f"{int(r['n_pos']):>6} {int(r['n_neg']):>6}")
                        continue
                    ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
                    lines.append(f"  {r['finding']:<20} {r['siamese']:>6.3f} "
                                 f"{r['baseline']:>6.3f} {r['diff']:>+7.3f} {ci:>17} "
                                 f"{r['p_holm']:>8.4f} {int(r['n_pos']):>6} {int(r['n_neg']):>6}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="AP/PA projection confound control.")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--metadata-csv", required=True)
    ap.add_argument("--out-dir", default="projection_control_out")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    proj_status, counts = build_projection_map(Path(args.pairs_csv), Path(args.metadata_csv))
    print("Projection composition:", dict(counts))

    runs = discover_runs(Path(args.runs_root))
    if not runs:
        raise SystemExit(f"No runs found under {args.runs_root}")
    by_arm = defaultdict(list)
    for r in runs:
        by_arm[r["arm"]].append(r)
    arms = args.arms or sorted(by_arm)

    dfs = {}
    for arm in arms:
        print(f"[{arm}] projection-controlled bootstrap...")
        df = process_arm(arm, by_arm[arm], proj_status, args.n_boot, args.seed, out_dir)
        if df is not None:
            dfs[arm] = df
    if not dfs:
        raise SystemExit("No arm produced results.")

    summary = format_summary(dfs, counts)
    (out_dir / "projection_control_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nWrote per-arm CSVs and {out_dir / 'projection_control_summary.txt'}")


if __name__ == "__main__":
    main()
