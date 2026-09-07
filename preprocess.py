from __future__ import annotations

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# The 14 standard CheXpert / MIMIC-CXR observations.
ALL_CHEXPERT_LABELS: Tuple[str, ...] = (
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
)

# Recognized CheXpert competition subset.
CHEXPERT5_LABELS: Tuple[str, ...] = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion",
)


@dataclass
class Config:
    seed: int = 42

    dataset_root: Path = Path("mimic-cxr-jpg-2.1.0.physionet.org")
    output_dir: Path = Path("outputs_preprocess")

    image_size: int = 320
    cache_root: Path = Path("mimic-cxr-jpg-320-cache")
    cache_jpeg_quality: int = 95
    cache_num_workers: int = min(8, os.cpu_count() or 4)

    all_label_cols: Tuple[str, ...] = ALL_CHEXPERT_LABELS
    chexpert5_label_cols: Tuple[str, ...] = CHEXPERT5_LABELS

    frontal_views: Tuple[str, ...] = ("AP", "PA")


def resolve_existing_file(root: Path, base_name: str) -> Path:
    """Return root/base_name, tolerating a .gz-compressed variant."""
    for p in (root / base_name, root / f"{base_name}.gz"):
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {base_name} or {base_name}.gz under {root.resolve()}")


def square_pad(image: Image.Image) -> Image.Image:
    """Pad to a centered square with black borders (preserves aspect ratio)."""
    w, h = image.size
    m = max(w, h)
    left = (m - w) // 2
    top = (m - h) // 2
    return ImageOps.expand(image, border=(left, top, m - w - left, m - h - top), fill=0)


def bicubic_resample():
    try:
        return Image.Resampling.BICUBIC
    except AttributeError:
        return Image.BICUBIC


def add_raw_label_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Create raw_{label} for every observation (no uncertainty policy applied)."""
    df = df.copy()
    for col in cfg.all_label_cols:
        df[f"raw_{col}"] = pd.to_numeric(df[col], errors="coerce")
    return df


def raw_label_state_audit(chexpert_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Study-level counts of each raw state. Informs, but does not set, policy."""
    rows = []
    for col in cfg.all_label_cols:
        raw = pd.to_numeric(chexpert_df[col], errors="coerce")
        rows.append({
            "label": col,
            "is_chexpert5": col in cfg.chexpert5_label_cols,
            "positive_1": int(raw.eq(1).sum()),
            "negated_0": int(raw.eq(0).sum()),
            "uncertain_neg1": int(raw.eq(-1).sum()),
            "unmentioned_blank": int(raw.isna().sum()),
            "uncertain_fraction": float(raw.eq(-1).mean()),
        })
    return pd.DataFrame(rows)


def add_study_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add n_studies (distinct frontal study_id per subject) and is_single_study.
    `df` must already be filtered to frontal images. These columns drive:
      * pairing eligibility (n_studies >= 2), and
      * the disjoint-encoder patient set (is_single_study == True).
    """
    df = df.copy()
    counts = df.groupby("subject_id")["study_id"].nunique()
    df["n_studies"] = df["subject_id"].map(counts).astype(int)
    df["is_single_study"] = df["n_studies"] == 1
    return df


def cache_one_image(args: Tuple[str, str, str, int, int]) -> Tuple[str, bool, str]:
    """Worker: pad-to-square, resize to image_size, save JPEG. Skips if present."""
    src_rel_path, dataset_root_str, cache_root_str, image_size, jpeg_quality = args
    src_path = Path(dataset_root_str) / src_rel_path
    dst_path = (Path(cache_root_str) / src_rel_path).with_suffix(".jpg")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return str(dst_path), False, ""
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = square_pad(img)
            img = img.resize((image_size, image_size), resample=bicubic_resample())
            img.save(dst_path, quality=jpeg_quality, optimize=False)
        return str(dst_path), True, ""
    except Exception as e:
        return str(dst_path), False, f"{src_path}: {repr(e)}"


def cache_images_for_manifest(df: pd.DataFrame, cfg: Config, split_name: str) -> pd.DataFrame:
    """Cache every image in `df` and attach a cached_image_path column."""
    df = df.copy()
    if "image_path" not in df.columns:
        raise ValueError("Manifest must contain image_path before caching.")

    rel_paths = df["image_path"].astype(str).tolist()
    args_list = [(p, str(cfg.dataset_root), str(cfg.cache_root), cfg.image_size, cfg.cache_jpeg_quality)
                 for p in rel_paths]

    cached_by_rel: Dict[str, str] = {}
    errors: List[str] = []
    created = skipped = 0

    print(f"\nCaching {split_name} images -> {cfg.cache_root} | {len(args_list)} images | workers {cfg.cache_num_workers}")

    if cfg.cache_num_workers <= 1:
        for rel_path, args in tqdm(list(zip(rel_paths, args_list)), desc=f"cache {split_name}"):
            cp, cr, err = cache_one_image(args)
            cached_by_rel[rel_path] = cp
            created += int(cr); skipped += int(not cr and not err)
            if err:
                errors.append(err)
    else:
        with ProcessPoolExecutor(max_workers=cfg.cache_num_workers) as ex:
            fut = {ex.submit(cache_one_image, a): rel_paths[i] for i, a in enumerate(args_list)}
            for f in tqdm(as_completed(fut), total=len(fut), desc=f"cache {split_name}"):
                cp, cr, err = f.result()
                cached_by_rel[fut[f]] = cp
                created += int(cr); skipped += int(not cr and not err)
                if err:
                    errors.append(err)

    if errors:
        ep = cfg.output_dir / f"cache_errors_{split_name}.txt"
        ep.write_text("\n".join(errors), encoding="utf-8")
        raise RuntimeError(f"{len(errors)} image(s) failed to cache for {split_name}. See {ep}")

    df["cached_image_path"] = df["image_path"].astype(str).map(cached_by_rel)
    if df["cached_image_path"].isna().any():
        raise RuntimeError(f"Missing cached paths for split={split_name}.")

    print(f"{split_name}: created={created}, already_existing={skipped}")
    return df


def manifest_name(cfg: Config, split_name: str, cached: bool) -> str:
    suffix = "_cached" if cached else ""
    return f"manifest_{split_name}_frontal_{cfg.image_size}px{suffix}.csv"


def main():
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_root.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("Dataset root:", cfg.dataset_root.resolve())
    print("Output dir:", cfg.output_dir.resolve())
    print("Policy-agnostic manifest: emits raw_{label} for", len(cfg.all_label_cols), "labels.")

    if not cfg.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {cfg.dataset_root.resolve()}")

    required = {
        "image_filenames": cfg.dataset_root / "IMAGE_FILENAMES",
        "metadata": resolve_existing_file(cfg.dataset_root, "mimic-cxr-2.0.0-metadata.csv"),
        "split": resolve_existing_file(cfg.dataset_root, "mimic-cxr-2.0.0-split.csv"),
        "chexpert": resolve_existing_file(cfg.dataset_root, "mimic-cxr-2.0.0-chexpert.csv"),
    }
    for name, path in required.items():
        print(f"{name:16s}: {path} | exists={path.exists()}")

    # 1) IMAGE_FILENAMES -> dicom_id
    with open(required["image_filenames"], "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]
    images_df = pd.DataFrame({"image_path": image_paths})
    images_df["dicom_id"] = images_df["image_path"].apply(lambda x: Path(x).stem)
    print("\nImages in IMAGE_FILENAMES:", len(images_df))

    sample_n = min(1000, len(images_df))
    sample_paths = images_df.sample(n=sample_n, random_state=cfg.seed)["image_path"].tolist()
    missing = [p for p in sample_paths if not (cfg.dataset_root / p).exists()]
    print(f"Checked {sample_n} sampled paths | missing: {len(missing)}")
    if missing:
        print("Examples:", missing[:10])

    # 2) Metadata / split / labels
    metadata_df = pd.read_csv(required["metadata"])
    split_df = pd.read_csv(required["split"])
    chexpert_df = pd.read_csv(required["chexpert"])
    print("\nmetadata:", metadata_df.shape, "| split:", split_df.shape, "| chexpert:", chexpert_df.shape)

    for df_name, df, cols in [
        ("images_df", images_df, ["dicom_id", "image_path"]),
        ("metadata_df", metadata_df, ["dicom_id", "subject_id", "study_id"]),
        ("split_df", split_df, ["dicom_id", "split"]),
        ("chexpert_df", chexpert_df, ["subject_id", "study_id"]),
    ]:
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise ValueError(f"{df_name} missing required columns: {miss}")

    missing_labels = [c for c in cfg.all_label_cols if c not in chexpert_df.columns]
    if missing_labels:
        raise ValueError(f"CheXpert file missing observations: {missing_labels}")

    # 3) Raw-state audit + plot
    raw_state_df = raw_label_state_audit(chexpert_df, cfg)
    raw_state_df.to_csv(cfg.output_dir / "chexpert_raw_label_states.csv", index=False)
    print("\nRaw CheXpert label states (study level):")
    print(raw_state_df.to_string(index=False))

    plt.figure(figsize=(10, 4))
    plt.bar(raw_state_df["label"], raw_state_df["uncertain_fraction"])
    plt.ylabel("Fraction uncertain (-1)")
    plt.title("Per-label uncertain rate")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "uncertain_label_fraction.png", dpi=300)
    plt.close()

    # 4) Merge images + metadata + split + labels
    preferred_metadata_cols = ["dicom_id", "subject_id", "study_id", "ViewPosition",
                               "Rows", "Columns", "StudyDate", "StudyTime"]
    metadata_cols = [c for c in preferred_metadata_cols if c in metadata_df.columns]

    merged_df = images_df.merge(metadata_df[metadata_cols], on="dicom_id", how="left", validate="one_to_one")
    merged_df = merged_df.merge(split_df[["dicom_id", "split"]], on="dicom_id", how="left", validate="one_to_one")
    chexpert_cols = ["subject_id", "study_id"] + list(cfg.all_label_cols)
    merged_df = merged_df.merge(chexpert_df[chexpert_cols], on=["subject_id", "study_id"], how="left", validate="many_to_one")
    print("\nMerged shape:", merged_df.shape)

    has_any = merged_df[list(cfg.all_label_cols)].notna().any(axis=1)
    if int((~has_any).sum()):
        print(f"Dropping {int((~has_any).sum())} images with no CheXpert label row.")
        merged_df = merged_df[has_any].copy()

    # 5) Frontal filter
    if "ViewPosition" not in merged_df.columns:
        raise ValueError("ViewPosition required to filter frontal images.")
    merged_df["ViewPosition"] = merged_df["ViewPosition"].astype(str).str.upper().str.strip()
    frontal_df = merged_df[merged_df["ViewPosition"].isin(cfg.frontal_views)].copy()
    print("All images:", len(merged_df), "| frontal AP/PA:", len(frontal_df))

    frontal_split_counts = frontal_df["split"].value_counts(dropna=False).rename_axis("split").reset_index(name="count")
    frontal_split_counts.to_csv(cfg.output_dir / "split_counts_frontal_images.csv", index=False)
    print("\nFrontal split counts:")
    print(frontal_split_counts)

    # 6) Raw label columns + per-patient frontal study counts
    frontal_df = add_raw_label_columns(frontal_df, cfg)
    frontal_df = add_study_count_columns(frontal_df)

    # 7) Final manifest columns: base + study-count + raw_{label}
    base_cols = ["image_path", "dicom_id", "subject_id", "study_id", "split",
                 "ViewPosition", "n_studies", "is_single_study"]
    raw_cols = [f"raw_{c}" for c in cfg.all_label_cols]
    manifest_cols = [c for c in base_cols if c in frontal_df.columns] + raw_cols
    final_manifest_df = frontal_df[manifest_cols].copy()

    req = ["image_path", "dicom_id", "subject_id", "study_id", "split"]
    before = len(final_manifest_df)
    final_manifest_df = final_manifest_df.dropna(subset=[c for c in req if c in final_manifest_df.columns])
    print("\nRows before/after dropping missing required fields:", before, "->", len(final_manifest_df))

    sort_cols = [c for c in ["subject_id", "study_id", "dicom_id"] if c in final_manifest_df.columns]
    final_manifest_df = final_manifest_df.sort_values(sort_cols).reset_index(drop=True)

    # 8) Patient-disjointness check across official splits
    subj = {s: set(final_manifest_df.loc[final_manifest_df["split"].astype(str).str.lower() == s, "subject_id"])
            for s in ["train", "validate", "test"]}
    for a, b in [("train", "validate"), ("train", "test"), ("validate", "test")]:
        overlap = subj[a] & subj[b]
        if overlap:
            raise ValueError(f"Patient leakage: {len(overlap)} subject_id(s) shared between {a} and {b}.")
    print("Patient-disjointness check passed.")

    # 9) Save patient study-count table (subject-level, one row per patient)
    patient_counts = (final_manifest_df[["subject_id", "n_studies", "is_single_study", "split"]]
                      .drop_duplicates("subject_id").sort_values("subject_id").reset_index(drop=True))
    patient_counts.to_csv(cfg.output_dir / "patient_study_counts.csv", index=False)
    n_single = int(patient_counts["is_single_study"].sum())
    n_multi = int((~patient_counts["is_single_study"]).sum())
    print(f"Patients: {len(patient_counts)} total | single-study {n_single} "
          f"(disjoint-encoder pool) | multi-study {n_multi} (pair-eligible)")

    # 10) Save per-split manifests
    written = []
    for split_value in ["train", "validate", "test"]:
        sd = final_manifest_df[final_manifest_df["split"].astype(str).str.lower() == split_value].copy()
        out_path = cfg.output_dir / manifest_name(cfg, split_value, cached=False)
        sd.to_csv(out_path, index=False)
        written.append(out_path)
        print(f"{split_value:8s}: {len(sd):8d} rows -> {out_path.name}")
        if len(sd) == 0:
            raise ValueError(f"Split '{split_value}' produced an empty manifest.")

    # 11) Cache images + write cached manifests
    cached_written = []
    for split_value in ["train", "validate", "test"]:
        sd_in = pd.read_csv(cfg.output_dir / manifest_name(cfg, split_value, cached=False))
        sd_cached = cache_images_for_manifest(sd_in, cfg, split_value)
        cached_out = cfg.output_dir / manifest_name(cfg, split_value, cached=True)
        sd_cached.to_csv(cached_out, index=False)
        cached_written.append(cached_out)
        print(f"Cached manifest saved: {cached_out.name}")

    # 12) Summary
    summary = {
        "image_size": cfg.image_size,
        "all_label_cols": list(cfg.all_label_cols),
        "manifest_kind": "policy_agnostic_raw_labels_with_study_counts",
        "frontal_images_in_manifest": int(len(final_manifest_df)),
        "n_patients_total": int(len(patient_counts)),
        "n_patients_single_study": n_single,
        "n_patients_multi_study": n_multi,
        "manifest_files": [str(p) for p in written],
        "cached_manifest_files": [str(p) for p in cached_written],
        "patient_study_counts": str(cfg.output_dir / "patient_study_counts.csv"),
        "cache_root": str(cfg.cache_root),
    }
    with open(cfg.output_dir / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(cfg.output_dir / "preprocess_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    print("\nPreprocessing complete.")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
