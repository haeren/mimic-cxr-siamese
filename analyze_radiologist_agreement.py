from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

TIER1 = ["No Finding", "Support Devices", "Cardiomegaly", "Lung Opacity",
         "Atelectasis", "Pleural Effusion", "Edema"]
# reference file uses the old CheXpert name for Lung Opacity
RENAME = {"Airspace Opacity": "Lung Opacity"}


def endpoint_present(label):
    """1 -> present(1); 0 or blank -> absent(0); -1 -> uncertain(None, excluded)."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return 0
    try:
        v = float(label)
    except (TypeError, ValueError):
        return 0
    if v == 1.0:
        return 1
    if v == -1.0:
        return None
    return 0   # 0.0 or anything else -> absent


def transition_code(first, last):
    """4-class transition from two endpoint presences; None if either uncertain."""
    pf, pl = endpoint_present(first), endpoint_present(last)
    if pf is None or pl is None:
        return None
    return {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}[(pf, pl)]


def kappa_safe(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(set(a.tolist()) | set(b.tolist())) < 2:
        return np.nan
    try:
        return cohen_kappa_score(a, b)
    except Exception:
        return np.nan


def load_reference(labeled_csv: Path):
    ref = pd.read_csv(labeled_csv)
    ref = ref.rename(columns=RENAME)
    ref["study_id"] = ref["study_id"].astype(str)
    ref = ref.set_index("study_id")
    return ref


def main():
    ap = argparse.ArgumentParser(description="Radiologist-agreement validation.")
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--labeled-csv", required=True)
    ap.add_argument("--out-dir", default="radiologist_agreement_out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs_csv)
    pairs["first_study_id"] = pairs["first_study_id"].astype(str)
    pairs["last_study_id"] = pairs["last_study_id"].astype(str)
    ref = load_reference(Path(args.labeled_csv))
    ref_ids = set(ref.index)

    # overlap bookkeeping
    endpoint_ids = set(pairs["first_study_id"]) | set(pairs["last_study_id"])
    n_endpoint_overlap = len(endpoint_ids & ref_ids)
    both_mask = pairs["first_study_id"].isin(ref_ids) & pairs["last_study_id"].isin(ref_ids)
    n_pairs_both = int(both_mask.sum())

    print(f"Reference studies: {len(ref_ids)}. Pair-endpoint studies in reference: "
          f"{n_endpoint_overlap}. Pairs with BOTH endpoints in reference: {n_pairs_both}.")

    # ---- (A) endpoint-level agreement -------------------------------------
    endpoint_rows = []
    for finding in TIER1:
        chex, rad = [], []
        for _, r in pairs.iterrows():
            for side in ("first", "last"):
                sid = r[f"{side}_study_id"]
                if sid not in ref_ids:
                    continue
                c = endpoint_present(r.get(f"{side}_{finding}"))
                q = endpoint_present(ref.at[sid, finding]) if finding in ref.columns else None
                if c is None or q is None:
                    continue
                chex.append(c); rad.append(q)
        n = len(chex)
        agree = float(np.mean(np.array(chex) == np.array(rad))) if n else np.nan
        endpoint_rows.append({"finding": finding, "n_endpoints": n,
                              "agreement": agree, "cohen_kappa": kappa_safe(chex, rad)})
    endpoint_df = pd.DataFrame(endpoint_rows)
    endpoint_df.to_csv(out_dir / "radiologist_endpoint_agreement.csv", index=False)

    # ---- (B) transition-level agreement -----------------------------------
    trans_rows = []
    sub = pairs[both_mask]
    for finding in TIER1:
        chex_t, rad_t = [], []
        for _, r in sub.iterrows():
            fid, lid = r["first_study_id"], r["last_study_id"]
            if finding not in ref.columns:
                continue
            rad_code = transition_code(ref.at[fid, finding], ref.at[lid, finding])
            chex_code = r.get(f"trans_{finding}")
            try:
                chex_code = int(chex_code)
            except (TypeError, ValueError):
                chex_code = None
            if rad_code is None or chex_code is None:
                continue
            chex_t.append(chex_code); rad_t.append(rad_code)
        n = len(chex_t)
        chex_t, rad_t = np.array(chex_t), np.array(rad_t)
        agree = float(np.mean(chex_t == rad_t)) if n else np.nan
        # resolved (class 2) and onset (class 1) as one-vs-rest agreement
        res_k = kappa_safe((chex_t == 2).astype(int), (rad_t == 2).astype(int)) if n else np.nan
        ons_k = kappa_safe((chex_t == 1).astype(int), (rad_t == 1).astype(int)) if n else np.nan
        trans_rows.append({"finding": finding, "n_pairs": n, "agreement_4class": agree,
                           "cohen_kappa_4class": kappa_safe(chex_t, rad_t),
                           "kappa_resolved": res_k, "kappa_onset": ons_k,
                           "n_chex_resolved": int((chex_t == 2).sum()),
                           "n_rad_resolved": int((rad_t == 2).sum()),
                           "n_chex_onset": int((chex_t == 1).sum()),
                           "n_rad_onset": int((rad_t == 1).sum())})
    trans_df = pd.DataFrame(trans_rows)
    trans_df.to_csv(out_dir / "radiologist_transition_agreement.csv", index=False)

    # ---- summary ----------------------------------------------------------
    L = ["RADIOLOGIST-AGREEMENT VALIDATION (single-radiologist reference)", "=" * 66,
         f"Column mapping applied: 'Airspace Opacity' -> 'Lung Opacity'.",
         f"Reference studies: {len(ref_ids)} | pair-endpoint studies in reference: "
         f"{n_endpoint_overlap} | pairs with both endpoints: {n_pairs_both}", ""]
    L.append("(A) ENDPOINT-LEVEL  CheXpert vs radiologist (present/absent)")
    L.append(f"  {'finding':<18} {'n':>6} {'agree%':>8} {'kappa':>7}")
    for _, r in endpoint_df.iterrows():
        ag = "n/a" if np.isnan(r["agreement"]) else f"{100*r['agreement']:.1f}"
        kp = "n/a" if np.isnan(r["cohen_kappa"]) else f"{r['cohen_kappa']:.3f}"
        L.append(f"  {r['finding']:<18} {int(r['n_endpoints']):>6} {ag:>8} {kp:>7}")

    L.append("\n(B) TRANSITION-LEVEL  CheXpert vs radiologist-derived transition")
    if n_pairs_both == 0:
        L.append("  No pairs have both endpoints in the reference set -> transition-level "
                 "check not possible; rely on (A).")
    else:
        L.append(f"  {'finding':<18} {'n':>5} {'agr4%':>7} {'k4':>7} {'k_res':>7} "
                 f"{'k_ons':>7} {'res c/r':>9} {'ons c/r':>9}")
        for _, r in trans_df.iterrows():
            def f3(x): return "n/a" if (x is None or np.isnan(x)) else f"{x:.3f}"
            ag = "n/a" if np.isnan(r["agreement_4class"]) else f"{100*r['agreement_4class']:.1f}"
            L.append(f"  {r['finding']:<18} {int(r['n_pairs']):>5} {ag:>7} "
                     f"{f3(r['cohen_kappa_4class']):>7} {f3(r['kappa_resolved']):>7} "
                     f"{f3(r['kappa_onset']):>7} "
                     f"{str(int(r['n_chex_resolved']))+'/'+str(int(r['n_rad_resolved'])):>9} "
                     f"{str(int(r['n_chex_onset']))+'/'+str(int(r['n_rad_onset'])):>9}")
    L += ["", "Honest scope: single radiologist (not consensus); validates label "
          "reliability / direction of change, not the exact 4-class scheme; transition-",
          "level N limited by reference overlap. Full multi-reader adjudication is future work."]
    summary = "\n".join(L)
    (out_dir / "radiologist_agreement_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nWrote CSVs and {out_dir / 'radiologist_agreement_summary.txt'}")


if __name__ == "__main__":
    main()
