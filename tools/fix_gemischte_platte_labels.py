#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "instance_crops"
META_PATH = DATA_ROOT / "metadata" / "crops.csv"


def _load_folder_mapping() -> Dict[str, str]:
    """
    Build a mapping from basename -> class_label based on the
    sorted folders data/instance_crops/images/{red,yellow,green}.

    If a basename appears in more than one folder, the last one wins,
    but this should not happen in practice for the gemischte_Platte crops.
    """
    mapping: Dict[str, str] = {}
    images_root = DATA_ROOT / "images"
    for cls in ("red", "yellow", "green"):
        folder = images_root / cls
        if not folder.is_dir():
            continue
        for path in folder.rglob("*.png"):
            mapping[path.name] = cls
    return mapping


def _read_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return [], []
    header, data = rows[0], rows[1:]
    return header, data


def _write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _fix_gemischte_platte() -> None:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {META_PATH}")

    header, data = _read_csv(META_PATH)
    if not header:
        print("crops.csv is empty, nothing to do.")
        return

    col_index: Dict[str, int] = {name: i for i, name in enumerate(header)}
    required = ("scene_stem", "crop_path", "class_label")
    missing = [c for c in required if c not in col_index]
    if missing:
        raise RuntimeError(f"Missing required columns in crops.csv: {missing}")

    idx_scene = col_index["scene_stem"]
    idx_crop = col_index["crop_path"]
    idx_label = col_index["class_label"]

    folder_mapping = _load_folder_mapping()
    if not folder_mapping:
        raise RuntimeError("No images found in images/{red,yellow,green}; nothing to map.")

    kept_rows: List[List[str]] = []
    deleted: List[List[str]] = []
    updated_count = 0
    unchanged_count = 0

    for row in data:
        if idx_scene >= len(row) or idx_crop >= len(row) or idx_label >= len(row):
            # Malformed row; keep as-is
            kept_rows.append(row)
            continue

        scene_stem = row[idx_scene]
        if not str(scene_stem).startswith("gemischte_Platte_"):
            kept_rows.append(row)
            continue

        crop_path = row[idx_crop]
        basename = Path(str(crop_path)).name
        new_label: Optional[str] = folder_mapping.get(basename)

        if new_label is None:
            # Not sorted into red/yellow/green yet -> drop this row entirely
            deleted.append(row)
            continue

        old_label = row[idx_label]
        if old_label != new_label:
            row[idx_label] = new_label
            updated_count += 1
        else:
            unchanged_count += 1

        kept_rows.append(row)

    # Backup original file
    backup_path = META_PATH.with_name(META_PATH.stem + "_before_gemischt_fix.csv")
    if not backup_path.exists():
        META_PATH.rename(backup_path)
    else:
        # If backup already exists, just overwrite crops.csv without renaming
        pass

    _write_csv(META_PATH, header, kept_rows)

    print(f"Total rows in original crops.csv: {len(data)}")
    print(f"gemischte_Platte rows updated with folder label: {updated_count}")
    print(f"gemischte_Platte rows unchanged (label already matching): {unchanged_count}")
    print(f"gemischte_Platte rows deleted (not in red/yellow/green folders): {len(deleted)}")
    print(f"Backup written to: {backup_path}")


def main() -> int:
    _fix_gemischte_platte()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

