from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


ALL_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
TRANS_CODE = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
TRANS_NAME = {0: "0->0", 1: "0->1(onset)", 2: "1->0(resolved)", 3: "1->1(persist)"}


@dataclass
class Config:
    preprocess_dir: Path = Path("outputs_preprocess")
    dataset_root: Path = Path("mimic-cxr-jpg-2.1.0.physionet.org")
    out_dir: Path = Path("pairs_out")

    image_size: int = 320
    frontal_priority: Tuple[str, ...] = ("PA", "AP")
    labels: List[str] = field(default_factory=lambda: list(ALL_LABELS))

    min_gap_days: int = 1                      # drop same-day pairs
    gap_windows: Tuple[int, ...] = (90, 180)   # build one pairs table per window
    n_folds: int = 5
    cv_seed: int = 42


def resolve_existing_file(root: Path, base: str) -> Path:
    for p in (root / base, root / f"{base}.gz"):
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {base}[.gz] under {root.resolve()}")


def manifest_path(cfg: Config, split: str) -> Path:
    p = cfg.preprocess_dir / f"manifest_{split}_frontal_{cfg.image_size}px_cached.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run preprocess.py first.")
    return p


def load_study_dates(cfg: Config) -> pd.DataFrame:
    """dicom_id -> StudyDate, StudyTime from the metadata CSV."""
    meta_path = resolve_existing_file(cfg.dataset_root, "mimic-cxr-2.0.0-metadata.csv")
    meta = pd.read_csv(meta_path, usecols=lambda c: c in {"dicom_id", "StudyDate", "StudyTime"})
    for c in ("dicom_id", "StudyDate", "StudyTime"):
        if c not in meta.columns:
            raise ValueError(f"metadata CSV missing '{c}'. Columns: {list(meta.columns)}")
    meta["StudyDate"] = pd.to_numeric(meta["StudyDate"], errors="coerce")
    meta["StudyTime"] = pd.to_numeric(meta["StudyTime"], errors="coerce")
    return meta


def pick_frontal_row(group: pd.DataFrame, priority: Tuple[str, ...]) -> pd.Series:
    """One image per study: prefer PA, then AP, else first available."""
    vp = group["ViewPosition"].astype(str).str.upper().str.strip()
    for want in priority:
        hit = group[vp == want]
        if len(hit):
            return hit.iloc[0]
    return group.iloc[0]


def studydate_to_days(date_series: pd.Series) -> pd.Series:
    """Shifted YYYYMMDD -> ordinal days (only within-patient differences are used)."""
    dt = pd.to_datetime(date_series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    return (dt - pd.Timestamp("2100-01-01")).dt.days.astype("float64")


def build_all_pairs(man: pd.DataFrame, dates: pd.DataFrame, cfg: Config):
    """One (first, last) pair per patient with >= 2 studies. Returns (pairs_df, info)."""
    man = man.merge(dates, on="dicom_id", how="left")
    n_missing_date = int(man["StudyDate"].isna().sum())

    rows = []
    eligible = skipped_single = 0
    for subj, g in man.groupby("subject_id"):
        g = g.dropna(subset=["StudyDate"])
        if g["study_id"].nunique() < 2:
            skipped_single += 1
            continue
        reps = pd.DataFrame([pick_frontal_row(sg, cfg.frontal_priority) for _, sg in g.groupby("study_id")])
        reps = reps.sort_values(["StudyDate", "StudyTime"], kind="mergesort")
        first, last = reps.iloc[0], reps.iloc[-1]
        if first["study_id"] == last["study_id"]:
            skipped_single += 1
            continue

        days = studydate_to_days(pd.Series([first["StudyDate"], last["StudyDate"]]))
        delta_days = float(days.iloc[1] - days.iloc[0])

        row = {
            "subject_id": subj,
            "first_study_id": int(first["study_id"]),
            "last_study_id": int(last["study_id"]),
            "first_image_path": first.get("cached_image_path", first["image_path"]),
            "last_image_path": last.get("cached_image_path", last["image_path"]),
            "delta_days": delta_days,
            "n_studies_total": int(reps.shape[0]),
        }
        for L in cfg.labels:
            rf = first.get(f"raw_{L}", np.nan)
            rl = last.get(f"raw_{L}", np.nan)
            row[f"first_{L}"] = rf
            row[f"last_{L}"] = rl
            if rf == -1 or rl == -1:
                row[f"trans_{L}"] = np.nan
            else:
                bf = 1 if rf == 1 else 0
                bl = 1 if rl == 1 else 0
                row[f"trans_{L}"] = TRANS_CODE[(bf, bl)]
        rows.append(row)
        eligible += 1

    pairs = pd.DataFrame(rows)
    info = {"eligible_patients": eligible, "skipped_single_study": skipped_single,
            "images_missing_date": n_missing_date}
    return pairs, info


def transition_audit(pairs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for L in cfg.labels:
        col = pairs[f"trans_{L}"]
        counts = {TRANS_NAME[k]: int((col == k).sum()) for k in range(4)}
        masked = int(col.isna().sum())
        n_lab = sum(counts.values())
        rows.append({
            "label": L, **counts, "masked_uncertain": masked, "labelled_pairs": n_lab,
            "any_change": counts["0->1(onset)"] + counts["1->0(resolved)"],
        })
    return pd.DataFrame(rows)


def assign_folds(subject_ids: np.ndarray, n_folds: int, seed: int) -> Dict[int, int]:
    """Plain random-by-patient, seeded, round-robin -> balanced, patient-disjoint folds."""
    rng = np.random.default_rng(seed)
    subs = np.array(sorted(set(int(s) for s in subject_ids)))
    rng.shuffle(subs)
    return {int(s): int(i % n_folds) for i, s in enumerate(subs)}


def main():
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading study dates...")
    dates = load_study_dates(cfg)

    print("Pooling frontal manifests across official splits...")
    man = pd.concat([pd.read_csv(manifest_path(cfg, s)) for s in ["train", "validate", "test"]],
                    ignore_index=True)
    for L in cfg.labels:
        if f"raw_{L}" in man.columns:
            man[f"raw_{L}"] = pd.to_numeric(man[f"raw_{L}"], errors="coerce")

    all_pairs, info = build_all_pairs(man, dates, cfg)
    n_raw = len(all_pairs)
    print(f"Eligible pairs (>=2 studies, pre-filter): {n_raw}")

    report_lines = ["LONGITUDINAL PAIRING REPORT", "=" * 60,
                    f"eligible pairs (>=2 studies, before gap filter): {n_raw}",
                    f"skipped single-study patients: {info['skipped_single_study']}",
                    f"images missing StudyDate: {info['images_missing_date']}", ""]

    for W in cfg.gap_windows:
        keep = (all_pairs["delta_days"] >= cfg.min_gap_days) & (all_pairs["delta_days"] <= W)
        pairs = all_pairs[keep].reset_index(drop=True)

        fold_map = assign_folds(pairs["subject_id"].to_numpy(), cfg.n_folds, cfg.cv_seed)
        pairs["fold"] = pairs["subject_id"].map(fold_map).astype(int)

        pairs_out = cfg.out_dir / f"pairs_gap{W}.csv"
        pairs.to_csv(pairs_out, index=False)

        # explicit shared fold map (single source of truth for the nested encoder)
        fm = pd.DataFrame({"subject_id": list(fold_map.keys()), "fold": list(fold_map.values())})
        fm.sort_values("subject_id").to_csv(cfg.out_dir / f"fold_map_gap{W}.csv", index=False)

        audit = transition_audit(pairs, cfg)
        audit.to_csv(cfg.out_dir / f"transition_prevalence_gap{W}.csv", index=False)

        fold_sizes = pairs["fold"].value_counts().sort_index()
        gap_desc = pairs["delta_days"].describe(percentiles=[.25, .5, .75])
        report_lines += [
            f"[gap <= {W} days]",
            f"  pairs kept: {len(pairs)} ({len(pairs)/max(n_raw,1)*100:.1f}% of eligible)",
            f"  fold sizes: {dict(fold_sizes)}",
            f"  gap days -- median {gap_desc['50%']:.0f}, p75 {gap_desc['75%']:.0f}, max {pairs['delta_days'].max():.0f}",
            "  effective per-finding support (onset / resolved / persist / none / masked):",
        ]
        for _, r in audit.sort_values("any_change", ascending=False).iterrows():
            report_lines.append(
                f"    {r['label']:28s} {r['0->1(onset)']:5d} / {r['1->0(resolved)']:5d} / "
                f"{r['1->1(persist)']:5d} / {r['0->0']:6d} / {r['masked_uncertain']:5d}")
        report_lines.append("")
        print(f"[gap<={W}] kept {len(pairs)} pairs -> pairs_gap{W}.csv, fold_map_gap{W}.csv")

    report_lines += [
        "Guidance: a finding needs roughly >=100 onset AND >=100 resolved (pooled over CV)",
        "for a per-finding 4-class result with usable CIs. Choose the primary window and",
        "point train_encoder.py/train_change_model.py --pairs-csv at that window's file.",
    ]
    (cfg.out_dir / "pairing_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\nWrote pairs_gap*.csv, fold_map_gap*.csv, transition_prevalence_gap*.csv, pairing_report.txt")
    print("Read pairs_out/pairing_report.txt to compare the 90- and 180-day windows.")


if __name__ == "__main__":
    main()
