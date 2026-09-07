import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tier-1 findings (well-supported onset AND resolved at 180 days).
TIER1 = ["Support Devices", "Edema", "Pleural Effusion", "Atelectasis",
         "No Finding", "Cardiomegaly", "Lung Opacity"]

ONSET_COL = "AUROC_onset_vs_rest"
RESOLVED_COL = "AUROC_resolved_vs_rest"


def paired_delta(df, arm, label, metric):
    """Mean and t-based 95% CI half-width of (siamese - last_only), paired across seeds."""
    s = df[(df.arm == arm) & (df["mode"] == "siamese") & (df.label == label)][["seed", metric]]
    l = df[(df.arm == arm) & (df["mode"] == "last_only") & (df.label == label)][["seed", metric]]
    m = s.merge(l, on="seed", suffixes=("_siam", "_last"))
    d = (m[f"{metric}_siam"] - m[f"{metric}_last"]).to_numpy(dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(d[0]), np.nan
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return float(d.mean()), float(half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="analysis_out/per_run_finding_metrics.csv")
    ap.add_argument("--arm", default="fold_nested",
                    help="Encoder regime to plot (default: fold_nested).")
    ap.add_argument("--out", default="figure_auroc_gain",
                    help="Output basename (writes .png and .pdf).")
    ap.add_argument("--annotate", action="store_true",
                    help="Print the resolved gain above each resolution bar.")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    findings = [f for f in TIER1 if f in set(df.label)]

    # order by resolved gain, descending
    order = sorted(findings, key=lambda L: paired_delta(df, args.arm, L, RESOLVED_COL)[0],
                   reverse=True)
    on = [paired_delta(df, args.arm, L, ONSET_COL) for L in order]
    re = [paired_delta(df, args.arm, L, RESOLVED_COL) for L in order]
    on_m, on_e = [x[0] for x in on], [x[1] for x in on]
    re_m, re_e = [x[0] for x in re], [x[1] for x in re]

    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(order))
    w = 0.38
    c_onset, c_res = "#9ecae1", "#08519c"   # colorblind-safe; separates in grayscale
    ax.bar(x - w / 2, on_m, w, yerr=on_e, capsize=3, color=c_onset, edgecolor="black",
           linewidth=0.6, label="Onset (0\u21921)", error_kw={"elinewidth": 0.8})
    ax.bar(x + w / 2, re_m, w, yerr=re_e, capsize=3, color=c_res, edgecolor="black",
           linewidth=0.6, label="Resolution (1\u21920)", error_kw={"elinewidth": 0.8})
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("AUROC gain from pairing\n(Siamese \u2212 single-image baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ymax = max(re_m) + max(re_e) + 0.05
    ax.set_ylim(min(0, min(on_m) - max([e for e in on_e if not np.isnan(e)] + [0]) - 0.02), ymax)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    if args.annotate:
        for xi, m, e in zip(x, re_m, re_e):
            ax.text(xi + w / 2, m + (e if not np.isnan(e) else 0) + 0.006,
                    f"+{m:.2f}", ha="center", va="bottom", fontsize=8.5, color=c_res)
    plt.tight_layout()
    plt.savefig(f"{args.out}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{args.out}.pdf", bbox_inches="tight")
    print(f"Wrote {args.out}.png and {args.out}.pdf (arm={args.arm})")
    for L, m, e in zip(order, re_m, re_e):
        print(f"  {L:18s} resolved gain {m:+.3f} \u00b1 {e:.3f}")


if __name__ == "__main__":
    main()
