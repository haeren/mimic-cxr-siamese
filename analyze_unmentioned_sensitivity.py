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


def raw_presence(v):
    """1 -> present; 0 -> explicit negative; blank/NaN -> unmentioned; -1 -> uncertain."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unmentioned"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "unmentioned"
    if f == 1.0:
        return "present"
    if f == -1.0:
        return "uncertain"
    return "negative"   # 0.0


def main():
    ap = argparse.ArgumentParser(description="Unmentioned-vs-negative label sensitivity.")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--out-dir", default="unmentioned_sensitivity_out")
    ap.add_argument("--arms", nargs="+", default=["fold_nested"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs_csv)
    pairs["subject_id"] = pairs["subject_id"].astype(int)

    # detect whether blanks are preserved in first_/last_ columns
    any_blank = False
    for finding in TIER1:
        for side in ("first", "last"):
            col = f"{side}_{finding}"
            if col in pairs.columns and pairs[col].isna().any():
                any_blank = True
    if not any_blank:
        print("WARNING: no blank/NaN values found in first_/last_ columns. Either this "
              "cohort genuinely has no unmentioned endpoints, or blanks were already "
              "converted to 0 (in which case unmentioned cannot be distinguished here — "
              "use the raw manifest). Proceeding; label-shift counts may be zero.")

    # ---- Part 1: label shift -------------------------------------------------
    shift_rows = []
    # presence category per subject per finding (first,last)
    pres = {}   # finding -> subject -> (first_cat, last_cat)
    for finding in TIER1:
        d = {}
        for _, r in pairs.iterrows():
            d[int(r["subject_id"])] = (raw_presence(r.get(f"first_{finding}")),
                                       raw_presence(r.get(f"last_{finding}")))
        pres[finding] = d
        # counts over defined transitions (neither endpoint uncertain)
        n_def = n_absent_ep = n_absent_blank = 0
        cls_drop = defaultdict(int); cls_tot = defaultdict(int)
        for (cf, cl) in d.values():
            if cf == "uncertain" or cl == "uncertain":
                continue
            n_def += 1
            pf = 1 if cf == "present" else 0
            pl = 1 if cl == "present" else 0
            code = {(0, 0): "0->0", (0, 1): "0->1", (1, 0): "1->0", (1, 1): "1->1"}[(pf, pl)]
            cls_tot[code] += 1
            # count "absent" endpoints and how many are blank
            blank_absent = False
            if pf == 0:
                n_absent_ep += 1
                if cf == "unmentioned":
                    n_absent_blank += 1; blank_absent = True
            if pl == 0:
                n_absent_ep += 1
                if cl == "unmentioned":
                    n_absent_blank += 1; blank_absent = True
            if blank_absent:
                cls_drop[code] += 1
        shift_rows.append({
            "finding": finding, "n_defined_transitions": n_def,
            "n_absent_endpoints": n_absent_ep,
            "pct_absent_endpoints_unmentioned": (100.0 * n_absent_blank / n_absent_ep) if n_absent_ep else np.nan,
            "drop_0to0": cls_drop["0->0"], "tot_0to0": cls_tot["0->0"],
            "drop_1to0": cls_drop["1->0"], "tot_1to0": cls_tot["1->0"],
            "drop_0to1": cls_drop["0->1"], "tot_0to1": cls_tot["0->1"],
        })
    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(out_dir / "unmentioned_label_shift.csv", index=False)

    # ---- Part 2: stricter re-evaluation -------------------------------------
    runs = discover_runs(Path(args.runs_root))
    by_arm = defaultdict(list)
    for r in runs:
        by_arm[r["arm"]].append(r)
    arms = [a for a in args.arms if a in by_arm] or (sorted(by_arm)[:1] if by_arm else [])

    reeval = {}
    for arm in arms:
        sia_dirs = [r["dir"] for r in by_arm[arm] if r["mode"] == "siamese"]
        base_dirs = [r["dir"] for r in by_arm[arm] if r["mode"] == "last_only"]
        if not sia_dirs or not base_dirs:
            continue
        sia = seed_average(sia_dirs); base = seed_average(base_dirs)
        if sia is None or base is None:
            continue
        subj, y, mask, sia_probs = sia
        subj_b, _, _, base_probs = base
        idx_of_b = {s: k for k, s in enumerate(subj_b)}
        base_probs = base_probs[np.array([idx_of_b[s] for s in subj])]

        # strict mask: drop a (patient,finding) if any ABSENT endpoint was unmentioned
        strict = mask.copy()
        for finding in TIER1:
            i = ALL_LABELS.index(finding)
            d = pres[finding]
            for row_k, s in enumerate(subj):
                cats = d.get(int(s))
                if cats is None:
                    continue
                cf, cl = cats
                pf = 1 if cf == "present" else 0
                pl = 1 if cl == "present" else 0
                if (pf == 0 and cf == "unmentioned") or (pl == 0 and cl == "unmentioned"):
                    strict[row_k, i] = 0.0

        rng = np.random.default_rng(args.seed)
        rows = []
        for task_name, cfg in TASKS.items():
            for metric in ("auroc", "auprc"):
                for mask_name, m in (("original", mask), ("strict", strict)):
                    for finding in TIER1:
                        i = ALL_LABELS.index(finding)
                        r = paired_bootstrap_binary(y, m, sia_probs, base_probs, i,
                                                    cfg["pos"], cfg["neg"], metric,
                                                    args.n_boot, rng)
                        rows.append({"arm": arm, "task": task_name, "metric": metric,
                                     "labels": mask_name, "finding": finding,
                                     "siamese": r["sia"], "baseline": r["base"],
                                     "diff": r["diff"], "ci_lo": r["lo"], "ci_hi": r["hi"],
                                     "n_pos": r["n_pos"], "n_neg": r["n_neg"]})
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"unmentioned_reeval_{arm}.csv", index=False)
        reeval[arm] = df

    # ---- summary ------------------------------------------------------------
    L = ["UNMENTIONED-VS-NEGATIVE SENSITIVITY (evaluation-side; no retraining)", "=" * 70,
         "Scope: checks whether evaluation is inflated by treating blank as absent; does "
         "NOT remove training-side effect (blank->neg was used in training).", ""]
    L.append("PART 1 — label shift (how many absent endpoints are actually unmentioned)")
    L.append(f"  {'finding':<18} {'n_trans':>8} {'%absent_unment':>15} "
             f"{'0->0 drop':>12} {'1->0 drop':>12}")
    for _, r in shift_df.iterrows():
        pct = "n/a" if np.isnan(r["pct_absent_endpoints_unmentioned"]) else f"{r['pct_absent_endpoints_unmentioned']:.1f}%"
        L.append(f"  {r['finding']:<18} {int(r['n_defined_transitions']):>8} {pct:>15} "
                 f"{str(int(r['drop_0to0']))+'/'+str(int(r['tot_0to0'])):>12} "
                 f"{str(int(r['drop_1to0']))+'/'+str(int(r['tot_1to0'])):>12}")

    for arm, df in reeval.items():
        L.append(f"\nPART 2 — stricter re-evaluation [{arm}]  (original vs strict; diff [95% CI]; n_pos)")
        for task in TASKS:
            for metric in ("auroc", "auprc"):
                L.append(f"\n  {task} — {metric.upper()}")
                L.append(f"    {'finding':<18} {'original':>26} {'strict':>26}")
                for finding in TIER1:
                    cells = []
                    for lab in ("original", "strict"):
                        rr = df[(df["task"] == task) & (df["metric"] == metric)
                                & (df["labels"] == lab) & (df["finding"] == finding)]
                        if rr.empty or np.isnan(rr["diff"].iloc[0]):
                            cells.append(f"{'n/a':>26}")
                        else:
                            x = rr.iloc[0]
                            s = f"{x['diff']:+.3f}[{x['ci_lo']:+.2f},{x['ci_hi']:+.2f}];{int(x['n_pos'])}"
                            cells.append(f"{s:>26}")
                    L.append(f"    {finding:<18}" + "".join(cells))

    L += ["", "Read: if 'strict' diffs stay close to 'original' (with smaller n), the "
          "effect does not depend on treating unmentioned as absent."]
    summary = "\n".join(L)
    (out_dir / "unmentioned_sensitivity_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nWrote label-shift + per-arm re-eval CSVs and "
          f"{out_dir / 'unmentioned_sensitivity_summary.txt'}")


if __name__ == "__main__":
    main()
