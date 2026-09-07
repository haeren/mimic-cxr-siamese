from __future__ import annotations

import json
import argparse
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd

# Reuse the EXACT training logic + defaults (no duplication, no drift).
import train_encoder as te


# Strategy list
def build_strategies(fixed_values, rng_range, do_random):
    """The U-Ones+LSR family only: fixed soft targets + an optional sampled variant.
    All share the same validation ground truth (uncertain -> positive), so every
    comparison here is exactly like-for-like."""
    strategies = []
    for v in fixed_values:
        strategies.append({"tag": f"u_ones_lsr_{v:.2f}".replace(".", "p"),
                           "policy": "u_ones_lsr", "kind": "fixed", "value": v})
    if do_random:
        lo, hi = rng_range
        strategies.append({"tag": f"u_ones_lsr_rand_{lo:.2f}_{hi:.2f}".replace(".", "p"),
                           "policy": "u_ones_lsr", "kind": "sampled", "rng_range": (lo, hi)})
    return strategies


# Injected behaviours (monkeypatched into te, then restored)
def make_sampled_derive(lo, hi, seed):
    """Replacement for te.derive_target_mask_from_raw: uncertain labels get a soft
    target sampled uniformly from [lo, hi]; blank->negative, 0/1 kept, mask all ones."""
    rng = np.random.default_rng(seed)

    def derive(raw, policy, lsr_value):
        unmentioned = np.isnan(raw)
        uncertain = (raw == -1)
        target = np.where(unmentioned, 0.0, raw).astype("float32")
        mask = np.ones_like(target, dtype="float32")
        samp = rng.uniform(lo, hi, size=target.shape).astype("float32")
        target = np.where(uncertain, samp, target)
        return np.clip(target, 0.0, 1.0).astype("float32"), mask

    return derive


def make_subsampled_selector(original, frac, base_seed):
    """Wrap te.select_encoder_patients to keep only a seeded fraction of ALLOWED
    patients (forbidden is untouched, so leakage safety is preserved). Seeded by fold
    so every strategy sees the SAME subsample within a fold."""
    def selector(subset, fold, single_study_subjects, fold_map):
        allowed, forbidden, tag = original(subset, fold, single_study_subjects, fold_map)
        rng = np.random.default_rng(base_seed + (fold if fold is not None else 0))
        arr = np.array(sorted(allowed))
        k = max(1, int(round(len(arr) * frac)))
        keep = set(int(s) for s in rng.choice(arr, size=k, replace=False))
        return keep, forbidden, tag + f"+sub{frac:.2f}"
    return selector


def run_one(base_cfg, strat, fold, out_root):
    tag = strat["tag"]
    out_dir = out_root / tag / f"fold{fold}"
    lsr_value = strat.get("value", 0.75)
    cfg = replace(base_cfg, uncertainty_policy=strat["policy"], lsr_value=lsr_value, out_dir=out_dir)

    desc = {"fixed": f"U-Ones+LSR fixed {lsr_value}",
            "sampled": f"U-Ones+LSR random {strat.get('rng_range')}"}[strat["kind"]]
    print(f"\n{'='*70}\n[SWEEP] {tag}  fold {fold}  ({desc})\n{'='*70}")

    if strat["kind"] == "sampled":
        lo, hi = strat["rng_range"]
        original = te.derive_target_mask_from_raw
        te.derive_target_mask_from_raw = make_sampled_derive(lo, hi, cfg.seed + fold)
        try:
            te.train_encoder(cfg, "fold_nested", fold)
        finally:
            te.derive_target_mask_from_raw = original
    else:
        te.train_encoder(cfg, "fold_nested", fold)

    meta = json.loads((out_dir / "encoder_meta.json").read_text())
    return {
        "strategy": tag, "policy": strat["policy"], "kind": strat["kind"],
        "lsr_value": (lsr_value if strat["kind"] == "fixed" else ""),
        "lsr_range": (str(strat.get("rng_range")) if strat["kind"] == "sampled" else ""),
        "fold": fold,
        "best_val_macro_auroc": meta.get("best_val_macro_auroc"),
        "n_train_patients": meta.get("n_train_patients"),
        "n_val_patients": meta.get("n_val_patients"),
    }


def summarize(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag, g in runs_df.groupby("strategy", sort=False):
        vals = pd.to_numeric(g["best_val_macro_auroc"], errors="coerce").dropna().to_numpy()
        n = len(vals)
        rows.append({
            "strategy": tag,
            "mean_val_macro_auroc": float(np.mean(vals)) if n else float("nan"),
            "std_val_macro_auroc": float(np.std(vals, ddof=1)) if n >= 2 else (0.0 if n == 1 else float("nan")),
            "n_folds": n,
            "per_fold": ",".join(f"{v:.4f}" for v in vals),
        })
    out = pd.DataFrame(rows).sort_values("mean_val_macro_auroc", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Uncertainty-strategy sweep over fold_nested folds.")
    ap.add_argument("--out-root", default="policy_sweep_out")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2],
                    help="Which folds to run (default: [0, 1, 2]).")
    ap.add_argument("--subsample-frac", type=float, default=0.25,
                    help="Fraction of allowed patients per encoder (speed lever; default 0.25).")
    ap.add_argument("--subsample-seed", type=int, default=12345)
    ap.add_argument("--fixed-values", nargs="+", type=float, default=[0.55, 0.65, 0.75, 0.85])
    ap.add_argument("--random-range", nargs=2, type=float, default=[0.55, 0.85], metavar=("LO", "HI"))
    ap.add_argument("--no-random", dest="do_random", action="store_false", default=True)
    # overrides mirrored from train_encoder.py
    ap.add_argument("--pairs-csv", default=None)
    ap.add_argument("--preprocess-dir", default=None)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    for v in args.fixed_values:
        if not (0.5 < v < 1.0):
            raise SystemExit(f"--fixed-values must be in (0.5, 1.0); got {v}.")
    lo, hi = args.random_range
    if not (0.5 < lo < hi < 1.0):
        raise SystemExit(f"--random-range must satisfy 0.5 < LO < HI < 1.0; got [{lo}, {hi}].")
    if not (0.0 < args.subsample_frac <= 1.0):
        raise SystemExit("--subsample-frac must be in (0, 1].")

    base_cfg = te.Config()
    if args.pairs_csv is not None: base_cfg = replace(base_cfg, pairs_csv=Path(args.pairs_csv))
    if args.preprocess_dir is not None: base_cfg = replace(base_cfg, preprocess_dir=Path(args.preprocess_dir))
    if args.cache_root is not None: base_cfg = replace(base_cfg, cache_root=Path(args.cache_root))
    if args.epochs is not None: base_cfg = replace(base_cfg, epochs=args.epochs)
    if args.seed is not None: base_cfg = replace(base_cfg, seed=args.seed)

    strategies = build_strategies(args.fixed_values, (lo, hi), args.do_random)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    n_runs = len(strategies) * len(args.folds)
    print(f"Sweep: {len(strategies)} strategies x {len(args.folds)} folds = {n_runs} encoder(s). "
          f"subsample_frac={args.subsample_frac}, seed={base_cfg.seed}, epochs={base_cfg.epochs}.")
    print(f"Strategies: {', '.join(s['tag'] for s in strategies)}")
    print(f"Output -> {out_root.resolve()}")

    # apply the patient subsample globally (seeded per fold) if requested
    restore_selector = None
    if args.subsample_frac < 1.0:
        restore_selector = te.select_encoder_patients
        te.select_encoder_patients = make_subsampled_selector(
            restore_selector, args.subsample_frac, args.subsample_seed)

    runs = []
    try:
        for strat in strategies:
            for fold in args.folds:
                runs.append(run_one(base_cfg, strat, fold, out_root))
                pd.DataFrame(runs).to_csv(out_root / "policy_sweep_runs.csv", index=False)  # incremental
    finally:
        if restore_selector is not None:
            te.select_encoder_patients = restore_selector

    runs_df = pd.DataFrame(runs)
    summary = summarize(runs_df)
    summary.to_csv(out_root / "policy_sweep_summary.csv", index=False)

    lines = ["UNCERTAINTY-STRATEGY SWEEP (fold_nested) — mean +/- std validation macro-AUROC",
             "=" * 74, "",
             f"{'rank':>4}  {'strategy':<26}  {'mean':>8}  {'std':>7}  {'folds':>5}"]
    for _, r in summary.iterrows():
        lines.append(f"{int(r['rank']):>4}  {str(r['strategy']):<26}  "
                     f"{r['mean_val_macro_auroc']:>8.4f}  {r['std_val_macro_auroc']:>7.4f}  "
                     f"{int(r['n_folds']):>5}")
    best = summary.iloc[0]
    lines += ["", f"Best mean: {best['strategy']} "
              f"({best['mean_val_macro_auroc']:.4f} +/- {best['std_val_macro_auroc']:.4f} over {int(best['n_folds'])} folds)."]
    summary_txt = "\n".join(lines)
    (out_root / "policy_sweep_summary.txt").write_text(summary_txt, encoding="utf-8")

    print("\n" + summary_txt)
    print(f"\nWrote:\n  {out_root / 'policy_sweep_runs.csv'}\n  "
          f"{out_root / 'policy_sweep_summary.csv'}\n  {out_root / 'policy_sweep_summary.txt'}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
