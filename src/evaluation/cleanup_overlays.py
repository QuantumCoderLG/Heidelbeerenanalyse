from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Set

LOGGER = logging.getLogger("cleanup_overlays")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete overlay images that correspond to raw training images.",
    )
    parser.add_argument(
        "--overlays-root",
        type=Path,
        default=Path("outputs/overlays"),
        help="Root directory containing overlay images (per model subdirectories allowed)",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/images"),
        help="Root directory of raw training images",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without removing them",
    )
    return parser.parse_args(argv)


def _normalize_rel_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    key_path = rel.with_suffix("")
    return key_path.as_posix().lower()


def collect_raw_keys(raw_root: Path) -> Set[str]:
    raw_root = raw_root.expanduser().resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw image root not found: {raw_root}")

    keys: Set[str] = set()
    for file_path in raw_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = _normalize_rel_key(file_path, raw_root)
        keys.add(key)
    return keys


def overlay_rel_key(overlay_path: Path, overlays_root: Path) -> Set[str]:
    rel = overlay_path.relative_to(overlays_root)
    candidate_paths = []

    parts = rel.parts
    if len(parts) > 1:
        # Many overlay dumps start with the checkpoint name; drop it for matching.
        candidate_paths.append(Path(*parts[1:]))
    candidate_paths.append(rel)

    keys: Set[str] = set()
    for candidate in candidate_paths:
        stem = candidate.stem
        if stem.endswith("_overlay"):
            stem = stem[: -len("_overlay")]
        keys.add(candidate.with_name(stem).as_posix().lower())
        keys.add(stem.lower())
    return keys


def cleanup_empty_dirs(root: Path) -> None:
    candidates = {p.parent for p in root.rglob("*")}
    for dir_path in sorted(candidates, key=lambda p: len(p.parts), reverse=True):
        if dir_path == root:
            continue
        if not dir_path.exists():
            continue
        iterator = dir_path.iterdir()
        try:
            next(iterator)
        except StopIteration:
            try:
                dir_path.rmdir()
            except OSError:
                pass


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    overlays_root = args.overlays_root.expanduser().resolve()
    if not overlays_root.exists():
        LOGGER.warning("Overlays root %s does not exist; nothing to do", overlays_root)
        return 0

    raw_keys = collect_raw_keys(args.raw_root)
    if not raw_keys:
        LOGGER.warning("No raw image files found under %s; aborting", args.raw_root)
        return 0

    LOGGER.info("Collected %d raw image keys", len(raw_keys))

    matched = 0
    deleted = 0
    skipped = 0

    for overlay_path in overlays_root.rglob("*"):
        if not overlay_path.is_file():
            continue
        if overlay_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        keys = overlay_rel_key(overlay_path, overlays_root)
        if raw_keys.isdisjoint(keys):
            skipped += 1
            continue
        matched += 1
        if args.dry_run:
            LOGGER.info("[dry-run] Would delete %s", overlay_path)
            continue
        try:
            overlay_path.unlink()
            LOGGER.info("Deleted %s", overlay_path)
            deleted += 1
        except OSError as err:
            LOGGER.warning("Failed to delete %s: %s", overlay_path, err)

    if deleted:
        cleanup_empty_dirs(overlays_root)

    LOGGER.info(
        "Done. matched=%d deleted=%d skipped=%d dry_run=%s",
        matched,
        deleted,
        skipped,
        args.dry_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
