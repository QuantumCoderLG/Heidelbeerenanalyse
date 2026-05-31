# src/data/metadata.py
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _detect_project_root(start: Optional[Path] = None) -> Path:
    """
    Robustly detect project root to stabilize relative paths across different layouts.

    Strategy:
    - If we're inside .../<root>/berries*/src/, return <root>.
    - Else, if a parent contains 'berries*' directory with a 'src' subdir, return that parent.
    - Else, if a parent contains both 'src' and 'data', return that parent.
    - Fallback: the first directory two levels up from this file (if available).
    """
    here = (start or Path(__file__)).resolve()
    base = here.parent if here.is_file() else here
    for parent in [base] + list(base.parents):
        try:
            # Support both berries and berries2.0 (or any berries* variant)
            if parent.name.startswith("berries") and (parent / "src").is_dir():
                return parent.parent
            # Check for any berries* subdirectory
            for child in parent.iterdir():
                if child.is_dir() and child.name.startswith("berries") and (child / "src").is_dir():
                    return parent
            # Case 3: generic flat layout with src/ and data/
            if (parent / "src").is_dir() and (parent / "data").is_dir():
                return parent
        except (PermissionError, NotADirectoryError):
            continue
    parents = list(base.parents)
    return parents[1] if len(parents) >= 2 else base


def _project_root() -> Path:
    return _detect_project_root()


def _to_rel_str(p: Optional[Path], root: Path) -> Optional[str]:
    if p is None:
        return None
    try:
        return str(Path(p).resolve().relative_to(root).as_posix())
    except Exception:
        return str(Path(p).as_posix())


def _parquet_supported() -> bool:
    return (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )


def write_metadata(
    images_rows: Iterable[dict],
    annotations_rows: Iterable[dict],
    out_dir: Path,
    formats: Iterable[str] = ("csv", "parquet"),
) -> None:
    """
    Write images & annotations metadata tables into CSV and/or Parquet.

    Parameters
    ----------
    images_rows : Iterable[dict]
        Records for images table.
    annotations_rows : Iterable[dict]
        Records for annotations table.
    out_dir : Path
        Processed data root (e.g., berries/data/processed).
    formats : Iterable[str]
        Subset of {"csv", "parquet"}.
    """
    out_dir = Path(out_dir)
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    root = _project_root()

    # Normalize relative paths
    images_records: List[dict] = []
    for r in images_rows:
        r2 = dict(r)
        for key in ("image_path", "xml_path", "instances_mask_path", "overlay_path"):
            if key in r2:
                r2[key] = _to_rel_str(Path(r2[key]) if r2[key] else None, root)
        images_records.append(r2)

    ann_records: List[dict] = []
    for r in annotations_rows:
        r2 = dict(r)
        if "per_ann_mask_path" in r2:
            r2["per_ann_mask_path"] = _to_rel_str(
                Path(r2["per_ann_mask_path"]) if r2["per_ann_mask_path"] else None, root
            )
        ann_records.append(r2)

    df_images = pd.DataFrame.from_records(images_records)
    df_annotations = pd.DataFrame.from_records(ann_records)

    # Column ordering (best effort)
    image_cols = [
        "split",
        "image_id",
        "image_path",
        "xml_path",
        "width",
        "height",
        "n_annotations",
        "instances_mask_path",
        "overlay_path",
    ]
    df_images = df_images[[c for c in image_cols if c in df_images.columns]]

    ann_cols = [
        "split",
        "image_id",
        "annotation_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "area",
        "n_points",
        "per_ann_mask_path",
    ]
    # Keep category_id if present
    if "category_id" in df_annotations.columns and "category_id" not in ann_cols:
        ann_cols.insert(3, "category_id")

    df_annotations = df_annotations[[c for c in ann_cols if c in df_annotations.columns]]

    formats_set = {f.strip().lower() for f in formats}
    if "csv" in formats_set:
        img_csv = meta_dir / "images.csv"
        ann_csv = meta_dir / "annotations.csv"
        df_images.to_csv(img_csv, index=False)
        df_annotations.to_csv(ann_csv, index=False)
        logger.info("Wrote CSV metadata: %s , %s", img_csv, ann_csv)

    if "parquet" in formats_set:
        if not _parquet_supported():
            logger.warning(
                "Parquet output requested but 'pyarrow' or 'fastparquet' is not installed. Skipping Parquet export."
            )
        else:
            img_parq = meta_dir / "images.parquet"
            ann_parq = meta_dir / "annotations.parquet"
            try:
                df_images.to_parquet(img_parq, index=False)
                df_annotations.to_parquet(ann_parq, index=False)
                logger.info("Wrote Parquet metadata: %s , %s", img_parq, ann_parq)
            except Exception as e:
                logger.error("Failed to write Parquet metadata: %s", e)
                raise
