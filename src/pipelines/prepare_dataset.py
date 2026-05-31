# src/pipelines/prepare_dataset.py
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from ..data import coco_schema, ids, rasterize, xml_parser
from ..data.matching import discover_pairs
from ..data.metadata import write_metadata
from ..utils import geometry_utils, io_utils


logger = logging.getLogger(__name__)


# ----------------------------- Path helpers -----------------------------


def _detect_project_root(start: Path | None = None) -> Path:
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
    # Walk up from the *directory* to avoid NotADirectoryError on files
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
            # Skip parents we cannot access or that aren't directories
            continue
    # Fallback to a safe ancestor if available
    parents = list(base.parents)
    return parents[1] if len(parents) >= 2 else base


def _project_root() -> Path:
    return _detect_project_root()


def _rel_to_root(p: Path) -> str:
    root = _project_root()
    try:
        return str(Path(p).resolve().relative_to(root).as_posix())
    except Exception:
        return p.as_posix()


# ----------------------------- CLI helpers -----------------------------


def _parse_splits(s: Sequence[str], n_items: int) -> Tuple[int, int, int]:
    """
    Parse either percentages (summing to 100) or absolute counts (summing to n_items).
    Rounding for percentages: floor for train/val; test gets the remainder to ensure total == n_items.
    """
    if len(s) != 3:
        raise ValueError("--splits expects exactly three numbers (e.g., '70 15 15').")
    vals = [float(x) for x in s]
    if any(v < 0 for v in vals):
        raise ValueError("Split values must be non-negative.")
    ssum = sum(vals)
    if abs(ssum - 100.0) < 1e-6:
        # percentages
        train = int((vals[0] / 100.0) * n_items)
        val = int((vals[1] / 100.0) * n_items)
        test = n_items - train - val
    elif abs(ssum - float(n_items)) < 1e-6 or all(float(x).is_integer() for x in vals):
        # absolute counts
        counts = [int(round(v)) for v in vals]
        if sum(counts) != n_items:
            raise ValueError(
                f"Split counts must sum to number of items ({n_items}); got {counts}."
            )
        train, val, test = counts
    else:
        raise ValueError(
            "Invalid --splits. Provide percentages summing to 100 or counts summing to number of items."
        )
    if min(train, val, test) < 0:
        raise ValueError("Split counts must be non-negative.")
    return train, val, test


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------- Core processing -----------------------------


def _process_pair(
    split: str,
    image_path: Path,
    xml_path: Path,
    out_split_dir: Path,
    export_per_ann_masks: bool,
    category_map: Dict[str, int],
) -> Tuple[
    Dict,  # coco image dict
    List[Dict],  # coco annotation dicts
    dict,  # images metadata row
    List[dict],  # annotations metadata rows
]:
    """
    Process a single image/XML pair: build COCO entries, save mask/overlay, and return metadata rows.
    """
    # Load image for size and overlay drawing
    img_np, width, height = io_utils.load_image(image_path)
    image_stem = image_path.stem
    image_id = ids.stable_image_id(image_stem)

    # Parse XML polygons
    xml_text = xml_path.read_text(encoding="utf-8")
    anns_parsed = xml_parser.parse_xml(xml_text)
    polygons: List[List[Tuple[float, float]]] = [ann.polygon for ann in anns_parsed]
    labels: List[str] = [getattr(ann, 'label', 'blueberry') or 'blueberry' for ann in anns_parsed]

    # Defensive validation for coordinates falling within image bounds
    # (Clamp with warning if slightly out due to rounding noise)
    clamped_polygons: List[List[Tuple[float, float]]] = []
    changed = False
    for pi, poly in enumerate(polygons):
        clamped: List[Tuple[float, float]] = []
        for (x, y) in poly:
            x2 = float(min(max(x, 0.0), width - 1))
            y2 = float(min(max(y, 0.0), height - 1))
            if x2 != x or y2 != y:
                changed = True
            clamped.append((x2, y2))
        clamped_polygons.append(clamped)
    if changed:
        logger.debug("Clamped polygon coordinates to image bounds for %s.", image_path)

    # Build COCO image entry
    coco_img = coco_schema.build_coco_image(
        image_id=image_id,
        file_name=_rel_to_root(image_path),
        width=width,
        height=height,
    )

    # Build annotations
    coco_anns: List[Dict] = []
    ann_ids: List[int] = []
    ann_rows: List[dict] = []
    for idx, (poly, label) in enumerate(zip(clamped_polygons, labels)):
        ann_id = ids.stable_annotation_id(image_id, idx)
        ann_ids.append(ann_id)
        area = geometry_utils.polygon_area(poly)
        bbox_x, bbox_y, bbox_w, bbox_h = geometry_utils.polygon_bbox(poly)
        flat_poly: List[float] = []
        for x, y in poly:
            flat_poly.extend([float(x), float(y)])
        cat_id = int(category_map.get(str(label), 1))
        coco_ann = coco_schema.build_coco_annotation(
            ann_id=ann_id,
            image_id=image_id,
            polygon=flat_poly,
            bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
            area=area,
            category_id=cat_id,
        )
        coco_anns.append(coco_ann)
        ann_rows.append(
            {
                "split": split,
                "image_id": image_id,
                "annotation_id": ann_id,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "area": area,
                "n_points": len(poly),
                "category_id": cat_id,
                # "per_ann_mask_path" possibly added below
            }
        )

    # Rasterize instance mask for all polygons
    masks_dir = out_split_dir / "masks" / image_stem
    overlays_dir = out_split_dir / "overlays"
    _ensure_dir(masks_dir)
    _ensure_dir(overlays_dir)

    instance_mask = rasterize.rasterize_instance_mask(
        width=width, height=height, polygons=clamped_polygons, annotation_ids=ann_ids
    )
    instances_png_path = masks_dir / "instances.png"
    rasterize.save_instance_mask_png(instance_mask, instances_png_path)

    # Draw overlay
    overlay_img = rasterize.draw_overlay(img_np, clamped_polygons, ann_ids)
    overlay_path = overlays_dir / f"{image_stem}.png"
    io_utils.atomic_save_pil_image(overlay_img, overlay_path)

    # Optional: per-annotation binary masks
    if export_per_ann_masks and len(ann_ids) > 0:
        per_ann_root = out_split_dir / "masks" / "instances" / image_stem
        _ensure_dir(per_ann_root)
        for ar in ann_rows:
            aid = int(ar["annotation_id"])
            bin_mask = (instance_mask == aid).astype(np.uint8) * 255
            bin_img = Image.fromarray(bin_mask, mode="L")
            per_ann_path = per_ann_root / f"mask_{aid}.png"
            io_utils.atomic_save_pil_image(bin_img, per_ann_path)
            ar["per_ann_mask_path"] = per_ann_path

    # Image metadata row
    img_row = {
        "split": split,
        "image_id": image_id,
        "image_path": image_path,
        "xml_path": xml_path,
        "width": width,
        "height": height,
        "n_annotations": len(ann_ids),
        "instances_mask_path": instances_png_path,
        "overlay_path": overlay_path,
    }

    return coco_img, coco_anns, img_row, ann_rows


def _build_and_write_coco(
    split: str,
    coco_images: List[Dict],
    coco_annotations: List[Dict],
    out_split_dir: Path,
    categories: List[Dict],
) -> Path:
    coco_dict = {
        "info": {
            "description": f"Blueberry dataset ({split})",
            "version": "1.0",
            "year": datetime.utcnow().year,
            "date_created": datetime.utcnow().isoformat() + "Z",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    ann_path = out_split_dir / "annotations.json"
    _ensure_dir(out_split_dir)
    ann_path.write_text(json.dumps(coco_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote COCO annotations: %s", ann_path)
    return ann_path


# ----------------------------- Main CLI -----------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare blueberry dataset: match pairs, split, export COCO, masks, overlays, and metadata."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/raw/images"),
        help="Root directory with raw images (recursive).",
    )
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=Path("data/raw/annotations"),
        help="Root directory with raw XML annotations (recursive).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output root directory for processed artifacts.",
    )
    parser.add_argument(
        "--splits",
        nargs=3,
        default=["70", "15", "15"],
        help="Three numbers (percentages summing to 100, or counts) for train/val/test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic shuffling; omit for random each run.",
    )
    parser.add_argument(
        "--export-per-ann-masks",
        action="store_true",
        help="Additionally export per-annotation binary masks under masks/instances/<image_stem>/",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["csv", "parquet"],
        default=["csv", "parquet"],
        help="Metadata output formats.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        pairs = discover_pairs(args.images_dir, args.ann_dir)
        if not pairs:
            logger.error("No image↔XML pairs found. Exiting.")
            return 2


        # Discover the set of category names across all XMLs (for multi-class support)
        category_names: Set[str] = set()
        try:
            for _img_path, _xml_path in pairs:
                try:
                    _xml_text = Path(_xml_path).read_text(encoding="utf-8")
                    _anns = xml_parser.parse_xml(_xml_text)
                    for _ann in _anns:
                        _label = getattr(_ann, "label", "blueberry") or "blueberry"
                        category_names.add(str(_label))
                except Exception as parse_err:
                    logger.warning("Skipping XML while collecting categories due to parse error: %s", parse_err)
        except Exception as e_collect:
            logger.error("Failed while collecting category names: %s", e_collect)
            return 3

        if not category_names:
            category_names.add("blueberry")

        # Deterministic mapping: sort names alphabetically and assign 1..N
        categories = coco_schema.build_coco_categories(sorted(category_names))
        category_map: Dict[str, int] = {c["name"]: int(c["id"]) for c in categories}

        # Use deterministic shuffle only if a seed is provided; otherwise shuffle randomly each run
        rng = random.Random(args.seed) if args.seed is not None else random.Random()
        if args.seed is not None:
            logger.info("Shuffling pairs with seed=%d (deterministic).", args.seed)
        else:
            logger.info("Shuffling pairs with non-deterministic seed (varies each run).")
        pairs_shuffled = list(pairs)
        rng.shuffle(pairs_shuffled)

        n = len(pairs_shuffled)
        n_train, n_val, n_test = _parse_splits(args.splits, n)
        train_pairs = pairs_shuffled[:n_train]
        val_pairs = pairs_shuffled[n_train : n_train + n_val]
        test_pairs = pairs_shuffled[n_train + n_val :]

        splits = {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs,
        }

        logger.info(
            "Split sizes: train=%d, val=%d, test=%d (total=%d)", n_train, n_val, n_test, n
        )

        # Aggregate metadata across splits
        all_img_rows: List[dict] = []
        all_ann_rows: List[dict] = []

        for split_name, split_pairs in splits.items():
            out_split_dir = args.out_dir / split_name
            _ensure_dir(out_split_dir / "masks")
            _ensure_dir(out_split_dir / "overlays")

            coco_images: List[Dict] = []
            coco_annotations: List[Dict] = []

            for img_path, xml_path in split_pairs:
                coco_img, coco_anns, img_row, ann_rows = _process_pair(
                    split=split_name,
                    image_path=img_path,
                    xml_path=xml_path,
                    out_split_dir=out_split_dir,
                    export_per_ann_masks=args.export_per_ann_masks,
                    category_map=category_map,
                )

                coco_images.append(coco_img)
                coco_annotations.extend(coco_anns)
                all_img_rows.append(img_row)
                all_ann_rows.extend(ann_rows)

            # Verify counts per split
            total_anns_split = sum(r["n_annotations"] for r in all_img_rows if r["split"] == split_name)
            if total_anns_split != sum(1 for a in all_ann_rows if a["split"] == split_name):
                logger.error(
                    "Annotation count mismatch for split '%s': images reported %d, but got %d rows.",
                    split_name,
                    total_anns_split,
                    sum(1 for a in all_ann_rows if a["split"] == split_name),
                )
                return 4

            _build_and_write_coco(
                split=split_name,
                coco_images=coco_images,
                coco_annotations=coco_annotations,
                out_split_dir=out_split_dir,
                categories=categories,
            )

            logger.info(
                "Split '%s': %d images, %d annotations -> %s",
                split_name,
                len(coco_images),
                len(coco_annotations),
                out_split_dir / "annotations.json",
            )

        # Write metadata (across all splits)
        write_metadata(
            images_rows=all_img_rows,
            annotations_rows=all_ann_rows,
            out_dir=args.out_dir,
            formats=args.formats,
        )

        # Summary
        summary = {}
        for split_name in ("train", "val", "test"):
            n_imgs = sum(1 for r in all_img_rows if r["split"] == split_name)
            n_anns = sum(1 for r in all_ann_rows if r["split"] == split_name)
            summary[split_name] = {"images": n_imgs, "annotations": n_anns}
        logger.info("Summary: %s", json.dumps(summary, indent=2))

        return 0

    except Exception as e:
        logger.exception("Fatal error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
