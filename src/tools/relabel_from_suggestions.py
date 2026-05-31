from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# Known class labels (normalized, lowercase)
KNOWN_LABELS = {"yellow", "green", "red", "never"}


def _norm_label(label: str) -> str:
    s = str(label or "").strip().lower()
    if s in {"green", "green"}:
        return "green"
    if s in {"yellow"}:
        return "yellow"
    if s in {"red"}:
        return "red"
    if s in {"never", "nie", "never"}:
        return "never"
    if s in {"ok", "gleich", "same", "unchanged", "lassen"}:
        return "ok"
    return s


def _scene_abbrev(scene_stem: str) -> str:
    """
    Map e.g. "Yellow_2_1_1" -> "YELLOW211" for matching suggestion headers like "YELLOW211".
    """
    s = scene_stem.strip()
    if not s:
        return s
    # Upper-case letters, drop underscores, keep digits
    color = "".join(ch for ch in s if ch.isalpha())
    digits = "".join(ch for ch in s if ch.isdigit())
    return color.upper() + digits


def _candidate_scene_for_token(token: str, all_scenes: Iterable[str]) -> Optional[str]:
    t = token.strip().upper()
    # Also accept variants with spaces or underscores in the token
    t = t.replace(" ", "").replace("_", "")
    # Accept GRÜN vs GREEN
    t = t.replace("GRÜN", "GREEN")
    best: Optional[str] = None
    for scene in all_scenes:
        abbr = _scene_abbrev(scene)
        abbr2 = abbr.replace("GRÜN", "GREEN")
        if abbr == t or abbr2 == t:
            best = scene
            break
    return best


HEADER_RE = re.compile(r"^([A-ZÄÖÜß]+)\s*([0-9_\.-]+)?\s*$", re.IGNORECASE)
NUM_LINE_RE = re.compile(r"^\s*(?:nr\.?|nummer)\s*(\d+)\s*=\s*([\wäöüÄÖÜß\-]+)\s*$", re.IGNORECASE)


@dataclass
class Suggestion:
    scene_stem: str
    # mapping of 1-based Nummer -> target_label
    by_number: Dict[int, str]


def parse_suggestions(text: str, available_scenes: Iterable[str]) -> List[Suggestion]:
    scenes_set = set(available_scenes)
    cur_token: Optional[str] = None
    cur_target_scene: Optional[str] = None
    cur_map: Dict[int, str] = {}
    out: List[Suggestion] = []

    def _flush():
        nonlocal cur_token, cur_target_scene, cur_map, out
        if cur_target_scene and cur_map:
            out.append(Suggestion(scene_stem=cur_target_scene, by_number=dict(cur_map)))
        # reset
        cur_token = None
        cur_target_scene = None
        cur_map = {}

    lines = text.splitlines()
    for raw in lines:
        line = raw.strip()
        if not line:
            # blank line separates blocks
            _flush()
            continue
        m_h = HEADER_RE.match(line)
        if m_h and line.upper().startswith(tuple(["YELLOW", "RED", "GREEN", "GRÜN"])):
            # header: token identifies a scene by shorthand
            _flush()
            token = line.split()[0]
            scene = _candidate_scene_for_token(token, scenes_set)
            cur_token = token
            cur_target_scene = scene
            continue
        # Nummer line
        m_n = NUM_LINE_RE.match(line)
        if m_n and cur_target_scene:
            num = int(m_n.group(1))
            label = _norm_label(m_n.group(2))
            if label in KNOWN_LABELS or label == "ok":
                cur_map[num] = label
            else:
                # ignore unknown target labels gracefully
                pass
            continue
        # tolerate other lines/comments
        # e.g. "Nummer 16: never" or variations
        m = re.match(r"^\s*(?:nr\.?|nummer)\s*(\d+)\s*[:\-]??\s*([\wäöüÄÖÜß\-]+)\s*$", line, re.IGNORECASE)
        if m and cur_target_scene:
            num = int(m.group(1))
            label = _norm_label(m.group(2))
            if label in KNOWN_LABELS or label == "ok":
                cur_map[num] = label
            continue
        # If it's none of the above and we're inside a block, just skip
    # flush last
    _flush()
    # drop blocks without a resolvable scene
    out = [s for s in out if s.scene_stem is not None]
    return out


def load_metadata(crops_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(crops_csv)
    # ensure needed columns exist
    required = {"scene_stem", "crop_path", "mask_path", "class_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in {crops_csv}: {missing}")
    return df


def find_crop_row(df: pd.DataFrame, scene_stem: str, nummer_1_based: int) -> Optional[int]:
    # filename uses idXXX with 1-based numbering (id001 == Nummer 1)
    id_str = f"id{nummer_1_based:03d}"
    mask = (df["scene_stem"] == scene_stem) & df["crop_path"].astype(str).str.contains(fr"_{id_str}\\.png$")
    matches = df.index[mask].to_list()
    if matches:
        return int(matches[0])
    # fallback: check index_within_image column if present (Nummer -> index + 1)
    if "index_within_image" in df.columns:
        mask2 = (df["scene_stem"] == scene_stem) & (df["index_within_image"].astype(int) == (nummer_1_based - 1))
        matches2 = df.index[mask2].to_list()
        if matches2:
            return int(matches2[0])
    return None


def _move_path(src: Path, dst: Path, dry_run: bool) -> None:
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    shutil.move(str(src), str(dst))


def apply_relabel(
    df: pd.DataFrame,
    suggestions: List[Suggestion],
    repo_root: Path,
    dry_run: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    logs: List[str] = []
    moved = 0
    updated = 0
    errors = 0

    for s in suggestions:
        scene = s.scene_stem
        for nummer, label in s.by_number.items():
            if label == "ok":
                continue
            row_idx = find_crop_row(df, scene, nummer)
            if row_idx is None:
                errors += 1
                logs.append(f"WARN: Not found: scene={scene} Nummer={nummer}")
                continue
            row = df.loc[row_idx].copy()
            cur_label = str(row.get("class_label", "")).strip().lower()
            tgt_label = _norm_label(label)
            if tgt_label not in KNOWN_LABELS:
                errors += 1
                logs.append(f"WARN: Unknown label '{label}' for scene={scene} Nummer={nummer}")
                continue
            if cur_label == tgt_label:
                # nothing to do
                continue
            # compute new paths
            crop_path = Path(row["crop_path"]) if row["crop_path"] else None
            mask_path = Path(row["mask_path"]) if row.get("mask_path") else None
            # Resolve to absolute for moving
            abs_crop = (repo_root / crop_path).resolve() if crop_path and not crop_path.is_absolute() else crop_path
            abs_mask = (repo_root / mask_path).resolve() if mask_path and not mask_path.is_absolute() else mask_path

            # filenames remain the same, only parent folder changes
            if abs_crop:
                new_crop = abs_crop.parent.parent / tgt_label / abs_crop.name
            else:
                new_crop = None
            if abs_mask:
                new_mask = abs_mask.parent.parent / tgt_label / abs_mask.name
            else:
                new_mask = None

            # move files
            if abs_crop and abs_crop.exists():
                _move_path(abs_crop, new_crop, dry_run=dry_run)
                moved += 1
            else:
                logs.append(f"WARN: Crop file missing for move: {abs_crop}")
                errors += 1
            if new_mask and abs_mask and abs_mask.exists():
                _move_path(abs_mask, new_mask, dry_run=dry_run)
                moved += 1
            elif abs_mask:
                logs.append(f"WARN: Mask file missing for move: {abs_mask}")
                errors += 1

            # update metadata to new label and relative paths (repo-relative if possible)
            rel_crop = None
            rel_mask = None
            try:
                rel_crop = str(new_crop.relative_to(repo_root)) if new_crop else ""
            except Exception:
                rel_crop = str(new_crop) if new_crop else ""
            try:
                rel_mask = str(new_mask.relative_to(repo_root)) if new_mask else ""
            except Exception:
                rel_mask = str(new_mask) if new_mask else ""

            df.loc[row_idx, "class_label"] = tgt_label
            if rel_crop:
                df.loc[row_idx, "crop_path"] = rel_crop
            if rel_mask:
                df.loc[row_idx, "mask_path"] = rel_mask
            updated += 1
            logs.append(
                f"OK: {scene} Nummer={nummer} {cur_label} -> {tgt_label}"
            )

    logs.append(f"Summary: updated_rows={updated}, files_moved={moved}, errors={errors}")
    return df, logs


def copy_split_sources(df: pd.DataFrame, repo_root: Path, dest_root: Path, dry_run: bool) -> Tuple[int, int]:
    """
    Copy full source images that were used ("geteilt") into an extra subfolder.
    Does not remove originals by default.
    """
    # Column 'source_image_path' is repo-relative path like data/all_images/Ampel/Yellow/Yellow_1_1.JPG
    if "source_image_path" not in df.columns:
        return 0, 0
    src_paths = sorted({str(p) for p in df["source_image_path"].dropna().astype(str).tolist() if p})
    copied = 0
    skipped = 0
    for rel in src_paths:
        src = (repo_root / rel).resolve()
        if not src.exists():
            skipped += 1
            continue
        # re-root under dest_root, preserving tail directories after data/all_images
        try:
            tail = src.relative_to((repo_root / "data" / "all_images").resolve())
        except Exception:
            # fallback: keep filename only
            tail = src.name
        dst = (dest_root / tail)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            copied += 1
            continue
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
            copied += 1
        else:
            skipped += 1
    return copied, skipped


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Relabel/move instance crops from a suggestion file.")
    p.add_argument("suggestions", type=Path, help="Path to the suggestion text file")
    p.add_argument("--crops-csv", type=Path, default=Path("data/instance_crops/metadata/crops.csv"))
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--dry-run", action="store_true", help="Do not move or write files, only print actions")
    p.add_argument("--backup", action="store_true", help="Write a timestamped backup of crops.csv before saving")
    p.add_argument(
        "--split-sources-dir",
        type=Path,
        default=None,
        help="If set, copy full source images used by crops into this extra subfolder",
    )
    args = p.parse_args(argv)

    repo_root = args.repo_root.resolve()
    crops_csv = (repo_root / args.crops_csv).resolve()
    if not crops_csv.exists():
        print(f"ERROR: crops.csv not found: {crops_csv}", file=sys.stderr)
        return 2
    df = pd.read_csv(crops_csv)
    if "scene_stem" not in df.columns:
        print("ERROR: crops.csv missing 'scene_stem' column", file=sys.stderr)
        return 2

    scenes = sorted(set(df["scene_stem"].astype(str).tolist()))
    try:
        txt = args.suggestions.read_text(encoding="utf-8")
    except Exception:
        # try latin-1 if UTF‑8 fails
        txt = args.suggestions.read_text(encoding="latin-1")
    parsed = parse_suggestions(txt, scenes)
    if not parsed:
        print("No valid suggestions found. Nothing to do.")
        return 0

    # Apply
    df2, logs = apply_relabel(df, parsed, repo_root=repo_root, dry_run=args.dry_run)

    # Optionally copy split sources
    if args.split_sources_dir is not None:
        dest = (repo_root / args.split_sources_dir).resolve()
        copied, skipped = copy_split_sources(df2, repo_root=repo_root, dest_root=dest, dry_run=args.dry_run)
        logs.append(f"Split sources: copied={copied}, skipped_existing_or_missing={skipped}")

    # Save
    if not args.dry_run:
        if args.backup:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = crops_csv.with_name(f"crops_{ts}.bak.csv")
            df.to_csv(backup_path, index=False)
        # write updated metadata
        df2.to_csv(crops_csv, index=False)
        # keep parquet in sync if present
        parquet_path = crops_csv.with_suffix(".parquet")
        try:
            df2.to_parquet(parquet_path, index=False)
        except Exception:
            pass

    # Write log
    log_dir = (repo_root / "outputs" / "relabel").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"apply_{ts}.log"
    try:
        log_path.write_text("\n".join(logs), encoding="utf-8")
    except Exception:
        pass

    # Also print summary
    print("\n".join(logs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

