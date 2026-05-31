#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    # Use repo-aware root detection if available
    from src.config.paths import project_root
except Exception:  # pragma: no cover - fallback
    def project_root() -> Path:  # type: ignore
        return Path(__file__).resolve().parents[1]


ALLOWED_CLASSES = {"red", "yellow", "green", "never"}


def _resolve_repo_path(rel_or_abs: str | Path, root: Path) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (root / p).resolve()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Sync moved unlabeled crops into class folders and update metadata.\n"
            "Use after generating crops into a staging subdirectory via --unlabeled-subdir,\n"
            "then manually dragging PNGs to images/{red,yellow,green,never}."
        )
    )
    p.add_argument("--metadata", type=Path, default=Path("data/instance_crops/metadata/crops.csv"))
    p.add_argument("--images-root", type=Path, default=Path("data/instance_crops/images"))
    p.add_argument("--masks-root", type=Path, default=Path("data/instance_crops/masks"))
    p.add_argument(
        "--staging-subdir",
        type=str,
        required=True,
        help="Subdirectory under images/ where unlabeled crops were initially written (e.g. 'to_sort/batch01').",
    )
    p.add_argument("--move-masks", action="store_true", help="Move corresponding mask PNGs alongside crops.")
    p.add_argument("--dry-run", action="store_true", help="Print planned changes without writing.")
    args = p.parse_args(argv)

    root = project_root().resolve()
    meta_path = _resolve_repo_path(args.metadata, root)
    images_root = _resolve_repo_path(args.images_root, root)
    masks_root = _resolve_repo_path(args.masks_root, root)
    staging_prefix = str((images_root / args.staging_subdir).resolve())

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    df = pd.read_csv(meta_path)
    if "crop_path" not in df.columns:
        raise ValueError("crops.csv missing 'crop_path' column")

    # Select records originating from the staging area
    def _abs_val(x: object) -> str:
        if not isinstance(x, str) or not x:
            return ""
        return str(_resolve_repo_path(x, root))

    df["__abs_crop_path"] = df["crop_path"].map(_abs_val)
    mask_stage = df["__abs_crop_path"].str.startswith(staging_prefix, na=False)
    candidates = df[mask_stage].copy()

    if candidates.empty:
        print("No rows found from the specified staging subdir. Nothing to update.")
        return 0

    updated_rows: List[int] = []
    conflicts: List[Tuple[int, str]] = []
    missing: List[Tuple[int, str]] = []
    mask_moves: List[Tuple[Path, Path]] = []

    for idx, row in candidates.iterrows():
        old_abs = Path(row["__abs_crop_path"]).resolve()
        basename = old_abs.name
        # Find the current location of this crop (user moved it)
        matches = [p for p in images_root.rglob(basename)]
        # Remove staging matches
        matches = [p for p in matches if not str(p).startswith(staging_prefix)]

        if len(matches) == 0:
            # Not moved or still in staging (old_abs may or may not exist)
            if old_abs.exists():
                continue  # still in staging, skip
            missing.append((idx, basename))
            continue
        if len(matches) > 1:
            conflicts.append((idx, basename))
            continue

        new_abs = matches[0].resolve()
        cls = new_abs.parent.name.lower()
        if cls == "green":
            cls = "green"
        if cls not in ALLOWED_CLASSES:
            print(f"Skip {basename}: new folder '{cls}' not in {sorted(ALLOWED_CLASSES)}")
            continue

        # Update metadata
        new_rel = str(new_abs.relative_to(root))
        df.at[idx, "crop_path"] = new_rel
        df.at[idx, "class_label"] = cls

        # Handle mask relocation
        mask_col = "mask_path" if "mask_path" in df.columns else None
        if args.move_masks and mask_col:
            old_mask_val = row.get(mask_col)
            if isinstance(old_mask_val, str) and old_mask_val:
                old_mask_abs = _resolve_repo_path(old_mask_val, root)
            else:
                # default guess: masks/staging_subdir/<basename>
                old_mask_abs = masks_root / args.staging_subdir / basename
            if old_mask_abs.exists():
                new_mask_abs = masks_root / cls / basename
                if not args.dry_run:
                    new_mask_abs.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_mask_abs), str(new_mask_abs))
                mask_moves.append((old_mask_abs, new_mask_abs))
                df.at[idx, mask_col] = str(new_mask_abs.relative_to(root))

        updated_rows.append(idx)

    # Write back
    if not updated_rows:
        print("No updates detected (did you move files out of staging?).")
        return 0

    # Backup
    backup = meta_path.with_suffix(meta_path.suffix + ".bak")
    if not args.dry_run:
        shutil.copy2(meta_path, backup)
        df.drop(columns=["__abs_crop_path"], inplace=True, errors="ignore")
        df.to_csv(meta_path, index=False)
        # Optional parquet if present originally
        try:
            parq = meta_path.with_suffix(".parquet")
            df.to_parquet(parq, index=False)
        except Exception:
            pass

    print(f"Updated {len(updated_rows)} rows. Conflicts: {len(conflicts)}, Missing: {len(missing)}")
    if conflicts:
        print("Conflicts (multiple matches):", conflicts[:5], "...")
    if missing:
        print("Missing (not found anywhere):", missing[:5], "...")
    if mask_moves:
        print(f"Moved {len(mask_moves)} masks.")
    if args.dry_run:
        print("DRY RUN: no files were moved and metadata not saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

