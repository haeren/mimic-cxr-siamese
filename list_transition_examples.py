import argparse
from pathlib import Path
import pandas as pd

STATES = [
    (0, "0->0  (absent -> absent)"),
    (1, "0->1  (ONSET)"),
    (2, "1->0  (RESOLVED)"),
    (3, "1->1  (persistent)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", default="pairs_out/pairs_gap180.csv",
                    help="Pairs table to draw from (default: the 180-day window).")
    ap.add_argument("--label", default="Edema",
                    help="Finding to illustrate (default: Edema).")
    ap.add_argument("--n", type=int, default=4, help="Examples per state.")
    ap.add_argument("--max-gap", type=float, default=180.0,
                    help="Maximum follow-up gap in days (default: 180).")
    ap.add_argument("--min-gap", type=float, default=None,
                    help="Optional minimum follow-up gap in days.")
    ap.add_argument("--check-exists", action="store_true",
                    help="Also verify each cached file is present on disk.")
    args = ap.parse_args()

    path = Path(args.pairs_csv)
    if not path.exists():
        raise SystemExit(f"Pairs file not found: {path.resolve()}  (run build_pairs.py first)")
    df = pd.read_csv(path)

    col = f"trans_{args.label}"
    if col not in df.columns:
        raise SystemExit(f"Column '{col}' not found. Available findings: "
                         f"{[c[len('trans_'):] for c in df.columns if c.startswith('trans_')]}")

    # Apply the gap window (the pairs file is already <= its own window, but this
    # makes the 180-day default explicit and lets you tighten it).
    if "delta_days" in df.columns:
        if args.max_gap is not None:
            df = df[df["delta_days"] <= args.max_gap]
        if args.min_gap is not None:
            df = df[df["delta_days"] >= args.min_gap]

    print("=" * 78)
    print(f"### {args.label}   |   pairs: {path.name}   |   gap <= {args.max_gap:g} d")
    print("=" * 78)

    trans = pd.to_numeric(df[col], errors="coerce")
    for code, state_label in STATES:
        sub = df[trans == code].copy()
        n_avail = len(sub)
        if "delta_days" in sub.columns:
            sub = sub.sort_values("delta_days")
        sub = sub.head(args.n)
        print(f"\n-- {state_label}  (n available: {n_avail})")
        if sub.empty:
            print("   (no examples)")
            continue
        for _, r in sub.iterrows():
            gap = r.get("delta_days", "")
            subj = r.get("subject_id", "")
            first = str(r["first_image_path"])
            last = str(r["last_image_path"])
            note = ""
            if args.check_exists:
                note = f"  [first_exists={Path(first).exists()}, last_exists={Path(last).exists()}]"
            print(f"   subject {subj} | gap {gap:>4} d")
            print(f"     first: {first}")
            print(f"     last : {last}{note}")

    print(f"\nDone. Copy the first/last paths for the {args.label} panels you want.")
    print("For onset/resolved, the short-gap examples listed first are usually cleanest.")


if __name__ == "__main__":
    main()
