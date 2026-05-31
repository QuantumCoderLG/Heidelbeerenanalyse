from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ..config.paths import project_root
from ..data import ids
from ..utils import image_ops, io_utils
from ..utils.color_norm import gray_world
from .crop_pipeline import _compute_quality_metrics, QAThresholds


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NotBerryConfig:
    image_root: Path = Path("data/all_images/Ampel")
    instance_masks_root: Path = Path("data/instance_crops/instance_masks")
    model_name: Optional[str] = None  # e.g. "fold_1_best"; auto-detect if None
    output_root: Path = Path("data/instance_crops")

    # Output subfolders
    out_class_name: str = "notberry"
    include_mask_channel: bool = True
    background_color: Tuple[int, int, int] = (128, 128, 128)

    # Sampling controls
    n_bg_patches_per_image: int = 2
    n_border_patches_per_image: int = 2
    n_border_mimic_per_image: int = 2
    n_mixup_per_image: int = 1
    min_patch_area_px: int = 1_000
    max_patch_area_px: int = 60_000
    border_dilate_kernel: int = 7  # odd
    margin: float = 0.15

    # Irregular shape parameters
    min_blob_radius: int = 6
    max_blob_radius: int = 40
    min_blobs: int = 1
    max_blobs: int = 4
    max_sampling_tries: int = 60

    # Fold handling
    folds: int = 5

    # IO/Run control
    random_seed: int = 1337
    skip_existing: bool = True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _detect_model_dir(root: Path, preferred: Optional[str]) -> Path:
    root = root.resolve()
    if preferred:
        p = root / preferred
        if not p.exists():
            raise FileNotFoundError(f"Instance mask model directory not found: {p}")
        return p
    # Auto-detect: take first subdirectory
    subs = [d for d in root.iterdir() if d.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No instance mask subdirectories found under {root}")
    subs.sort()
    return subs[0]


def _list_mask_image_pairs(image_root: Path, mask_root: Path) -> List[Tuple[Path, Path]]:
    """
    List tuples of (image_path, instance_mask_png_path) by mirroring directory structure.
    Expects mask files to have same stem and relative parent folders as source JPEGs.
    """
    pairs: List[Tuple[Path, Path]] = []
    for png in mask_root.rglob("*.png"):
        # Derive JPEG path candidates sharing the same stem
        rel = png.relative_to(mask_root)
        stem = png.stem
        candidates = [
            image_root / rel.with_suffix(".JPG"),
            image_root / rel.with_suffix(".JPEG"),
            image_root / rel.with_suffix(".jpg"),
            image_root / rel.with_suffix(".jpeg"),
        ]
        img_path = next((c for c in candidates if c.exists()), None)
        if img_path is None:
            LOGGER.warning("No matching JPEG found for mask %s", png)
            continue
        pairs.append((img_path.resolve(), png.resolve()))
    pairs.sort()
    return pairs


def _load_image_rgb(path: Path) -> np.ndarray:
    # Reuse io_utils for consistency (handles EXIF orientation)
    img_np, _, _ = io_utils.load_image(path)
    return img_np


def _load_instances_mask(path: Path) -> np.ndarray:
    """Load 8-bit instance mask and return boolean foreground mask (instances > 0)."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return (arr.astype(np.uint8) > 0)


def _ensure_dirs(base: Path, class_name: str, include_mask: bool) -> Tuple[Path, Optional[Path]]:
    crops_dir = (base / "images" / class_name).resolve()
    crops_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = None
    if include_mask:
        mask_dir = (base / "masks" / class_name).resolve()
        mask_dir.mkdir(parents=True, exist_ok=True)
    return crops_dir, mask_dir


def _hash_uint32(text: str) -> int:
    # Mirror ids._sha256_uint32 without relying on private symbol
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = 0
    for i in range(0, len(digest), 4):
        chunk = int.from_bytes(digest[i : i + 4], byteorder="big", signed=False)
        value ^= chunk
    return value & 0xFFFFFFFF


def _make_notberry_id(image_id: int, index: int) -> int:
    # Include NB in token to prevent collision with regular annotation ids
    token = f"NB:{image_id}:{index}"
    return _hash_uint32(token)


def _compute_bounds(mask: np.ndarray, margin: float, image_shape: Tuple[int, int]) -> image_ops.CropBounds:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Empty mask for bounds")
    y_min = int(ys.min())
    y_max = int(ys.max())
    x_min = int(xs.min())
    x_max = int(xs.max())
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    pad_y = int(round(h * margin))
    pad_x = int(round(w * margin))
    y0 = max(0, y_min - pad_y)
    y1 = min(image_shape[0], y_max + pad_y + 1)
    x0 = max(0, x_min - pad_x)
    x1 = min(image_shape[1], x_max + pad_x + 1)
    return image_ops.CropBounds(y0=y0, y1=y1, x0=x0, x1=x1)


def _sample_irregular_patch(
    allowed: np.ndarray,
    *,
    min_area: int,
    max_area: int,
    min_radius: int,
    max_radius: int,
    min_blobs: int,
    max_blobs: int,
    max_tries: int,
    target_area: float | None = None,
    area_tolerance: float = 0.4,
) -> Optional[np.ndarray]:
    """
    Sample a random, irregular, blob-like binary patch within the allowed region.
    Returns a binary mask of the same shape as 'allowed', or None if sampling fails.
    """
    if allowed.dtype != np.bool_:
        allowed = allowed.astype(bool)
    h, w = allowed.shape
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tries = 0
    rng = np.random.default_rng()
    allowed_idx = np.flatnonzero(allowed)
    if allowed_idx.size == 0:
        return None
    while tries < max_tries:
        tries += 1
        canvas = np.zeros((h, w), dtype=np.uint8)
        n_blobs = int(rng.integers(min_blobs, max_blobs + 1))
        ok_seed = 0
        for _ in range(n_blobs):
            # Choose a random seed point where allowed is True
            flat = int(allowed_idx[int(rng.integers(0, allowed_idx.size))])
            y = int(flat // w)
            x = int(flat % w)
            # Random ellipse parameters
            r1 = int(rng.integers(min_radius, max_radius + 1))
            r2 = int(rng.integers(min_radius, max_radius + 1))
            angle = float(rng.integers(0, 180))
            # Draw ellipse (clipped by bounds via mask later)
            cv2.ellipse(canvas, (x, y), (r1, r2), angle, 0, 360, 255, -1)
            ok_seed += 1
        if ok_seed == 0:
            continue
        # Smooth/irregularize shape
        iters_d = int(rng.integers(0, 3))
        iters_e = int(rng.integers(0, 2))
        if iters_d:
            canvas = cv2.dilate(canvas, kernel5, iterations=iters_d)
        if iters_e:
            canvas = cv2.erode(canvas, kernel3, iterations=iters_e)
        patch = (canvas > 0) & allowed
        area = int(np.count_nonzero(patch))
        if area < min_area:
            continue
        if target_area is not None:
            lower = int(max(min_area, target_area * (1.0 - area_tolerance)))
            upper = int(min(max_area, target_area * (1.0 + area_tolerance)))
        else:
            lower = min_area
            upper = max_area
        if area > upper:
            # Try to trim a bit with erosion
            patch_u8 = patch.astype(np.uint8) * 255
            patch_u8 = cv2.erode(patch_u8, kernel3, iterations=1)
            patch = patch_u8 > 0
            area = int(np.count_nonzero(patch))
        if area < lower or area > upper:
            continue
        return patch
    return None


def _prepare_for_export(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    path_cols = [
        c
        for c in (
            "crop_path",
            "mask_path",
            "source_image_path",
            "instances_mask_path",
            "source_rel_path",
        )
        if c in df.columns
    ]
    def _to_relative(p: Path | str | None) -> str:
        if not p:
            return ""
        path_obj = Path(p).resolve()
        try:
            return str(path_obj.relative_to(root))
        except ValueError:
            return str(path_obj)

    for col in path_cols:
        df[col] = df[col].apply(_to_relative)
    return df


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


class NotBerryGenerator:
    def __init__(self, cfg: NotBerryConfig) -> None:
        self.cfg = cfg
        # Detect repo root similar to CropPipeline: prefer a directory that contains 'data'
        module_root = Path(__file__).resolve().parents[2]
        candidate_roots = [module_root]
        try:
            candidate_roots.append(project_root().resolve())
        except Exception:
            pass
        chosen_root = None
        for cand in candidate_roots:
            if (cand / "data").exists():
                chosen_root = cand
                break
        self.repo_root = (chosen_root or module_root).resolve()

        self.image_root = (self.repo_root / cfg.image_root).resolve()
        self.masks_root = (self.repo_root / cfg.instance_masks_root).resolve()
        self.model_dir = _detect_model_dir(self.masks_root, cfg.model_name)
        self.output_root = (self.repo_root / cfg.output_root).resolve()
        self.crops_dir, self.mask_dir = _ensure_dirs(self.output_root, cfg.out_class_name, cfg.include_mask_channel)
        self.meta_dir = (self.output_root / "metadata").resolve()
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        random.seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)

        # Pre-load fold mapping from existing crops if available
        self.fold_map: Dict[int, int] = {}
        crops_csv = self.meta_dir / "crops.csv"
        if crops_csv.exists():
            try:
                df = pd.read_csv(crops_csv)
                if "image_id" in df.columns and "fold_id" in df.columns:
                    self.fold_map = {
                        int(r.image_id): int(r.fold_id)
                        for r in df[["image_id", "fold_id"]].dropna().itertuples(index=False)
                    }
                LOGGER.info("Loaded %d image→fold assignments from %s", len(self.fold_map), crops_csv)
            except Exception as exc:
                LOGGER.warning("Failed to load existing fold mapping from %s: %s", crops_csv, exc)

    def _assign_fold(self, image_id: int) -> int:
        if image_id in self.fold_map:
            return int(self.fold_map[image_id])
        # Deterministic fallback: hash modulo
        return int(image_id % max(1, self.cfg.folds))

    def _source_tags(self, image_path: Path) -> Dict[str, object]:
        rel = image_path.resolve().relative_to(self.image_root)
        parts = rel.parts
        source_group = parts[0] if len(parts) >= 1 else ""
        source_subgroup = parts[1] if len(parts) >= 2 else ""
        return {
            "source_group": source_group,
            "source_subgroup": source_subgroup,
            "source_rel_path": rel.as_posix(),
        }

    def _save_mask_png(self, mask: np.ndarray, out_path: Path) -> None:
        mask_u8 = (mask.astype(np.uint8) * 255)
        img = Image.fromarray(mask_u8, mode="L")
        io_utils.atomic_save_pil_image(img, out_path, compress_level=1)

    def _build_record(
        self,
        *,
        image_path: Path,
        mask_path: Path,
        crop_path: Path,
        patch_mask_roi: np.ndarray,
        processed_rgb: np.ndarray,
        image_id: int,
        neg_index: int,
        neg_id: int,
        bounds: image_ops.CropBounds,
        fold_id: int,
        original_area: int,
        neg_type: str,
    ) -> Dict[str, object]:
        metrics = _compute_quality_metrics(
            crop_rgb=processed_rgb,
            crop_mask=patch_mask_roi,
            original_area=int(original_area),
            qa_cfg=QAThresholds(),
        )
        tags = self._source_tags(image_path)
        record: Dict[str, object] = {
            "annotation_id": int(neg_id),
            "image_id": int(image_id),
            "class_label": self.cfg.out_class_name,
            "scene_stem": image_path.stem,
            "crop_path": crop_path,
            "mask_path": (self.mask_dir / f"{neg_id}.png") if self.cfg.include_mask_channel else None,
            "crop_height": int(bounds.height),
            "crop_width": int(bounds.width),
            "qa_reasons": [],
            "source_image_path": image_path,
            "instances_mask_path": mask_path,
            "overlay_path": None,
            "index_within_image": int(neg_index),
            "fold_id": int(fold_id),
            "neg_type": neg_type,
        }
        record.update(metrics)
        record.update(tags)
        return record

    def _process_single(
        self,
        *,
        image_path: Path,
        mask_path: Path,
    ) -> List[Dict[str, object]]:
        image_rgb = _load_image_rgb(image_path)
        mask_inst = _load_instances_mask(mask_path)
        h, w = mask_inst.shape

        k = max(3, int(self.cfg.border_dilate_kernel) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(mask_inst.astype(np.uint8), kernel, iterations=1) > 0
        ring_out = np.logical_and(dilated, ~mask_inst)
        bg_far = np.logical_and(~mask_inst, ~dilated)

        image_id = ids.stable_image_id(image_path.stem)
        records: List[Dict[str, object]] = []
        rng = np.random.default_rng()

        comp_data: List[Dict[str, object]] = []
        num_comp, labels, stats, _ = cv2.connectedComponentsWithStats(mask_inst.astype(np.uint8), connectivity=8)
        for label in range(1, num_comp):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            y1 = min(h, y + height)
            x1 = min(w, x + width)
            bounds = image_ops.CropBounds(y0=y, y1=y1, x0=x, x1=x1)
            comp_mask_full = labels == label
            mask_crop = comp_mask_full[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
            crop_rgb = image_rgb[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
            if crop_rgb.size == 0:
                continue
            metrics = _compute_quality_metrics(
                crop_rgb=crop_rgb,
                crop_mask=mask_crop,
                original_area=crop_rgb.shape[0] * crop_rgb.shape[1],
                qa_cfg=QAThresholds(),
            )
            dilated_local = cv2.dilate(comp_mask_full.astype(np.uint8), kernel, iterations=1) > 0
            ring_local = np.logical_and(dilated_local, ~comp_mask_full)
            ring_local = np.logical_and(ring_local, ~mask_inst)
            comp_data.append(
                {
                    "area": area,
                    "bounds": bounds,
                    "mask_full": comp_mask_full,
                    "mask_crop": mask_crop.astype(np.uint8),
                    "crop_rgb": crop_rgb,
                    "metrics": metrics,
                    "ring_mask": ring_local,
                }
            )

        def _choose_component() -> Optional[Dict[str, object]]:
            if not comp_data:
                return None
            return comp_data[int(rng.integers(0, len(comp_data)))]

        def gen_and_save(
            patch_mask_full: np.ndarray,
            idx_within: int,
            neg_type: str,
            texture_source: Optional[Dict[str, np.ndarray]] = None,
        ) -> Optional[Dict[str, object]]:
            if patch_mask_full is None or not patch_mask_full.any():
                return None
            bounds = _compute_bounds(patch_mask_full, self.cfg.margin, (h, w))
            roi_mask = patch_mask_full[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
            roi_rgb = image_rgb[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
            processed = image_ops.apply_background(roi_rgb.copy(), roi_mask, self.cfg.background_color)
            processed = gray_world(processed, strength=0.7, max_gain=1.6)

            if neg_type == "border_mimic":
                noise = rng.normal(0.0, 7.0, size=processed.shape)
                processed = np.clip(processed.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            if neg_type == "mix" and texture_source is not None:
                src_rgb = texture_source.get("rgb")
                src_mask = texture_source.get("mask")
                if src_rgb is not None and src_mask is not None and src_rgb.size and src_mask.size:
                    height = bounds.height
                    width = bounds.width
                    if height > 0 and width > 0:
                        src_rgb_resized = cv2.resize(src_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
                        src_mask_resized = cv2.resize(src_mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0
                        alpha = float(rng.uniform(0.45, 0.7))
                        blended = processed.astype(np.float32)
                        blended[src_mask_resized] = (
                            alpha * src_rgb_resized[src_mask_resized].astype(np.float32)
                            + (1.0 - alpha) * blended[src_mask_resized]
                        )
                        processed = np.clip(blended, 0, 255).astype(np.uint8)

            neg_id = _make_notberry_id(image_id, idx_within)
            out_img = self.crops_dir / f"{neg_id}.png"
            if self.cfg.include_mask_channel:
                out_mask = self.mask_dir / f"{neg_id}.png" if self.mask_dir else None
            else:
                out_mask = None

            def _make_record() -> Dict[str, object]:
                return self._build_record(
                    image_path=image_path,
                    mask_path=mask_path,
                    crop_path=out_img,
                    patch_mask_roi=roi_mask,
                    processed_rgb=processed,
                    image_id=image_id,
                    neg_index=idx_within,
                    neg_id=neg_id,
                    bounds=bounds,
                    fold_id=self._assign_fold(image_id),
                    original_area=int(image_rgb.shape[0] * image_rgb.shape[1]),
                    neg_type=neg_type,
                )

            if self.cfg.skip_existing and out_img.exists():
                return _make_record()

            img = Image.fromarray(processed)
            io_utils.atomic_save_pil_image(img, out_img, compress_level=1)
            if out_mask is not None:
                self._save_mask_png(roi_mask, out_mask)

            return _make_record()

        idx_counter = 0

        def next_index() -> int:
            nonlocal idx_counter
            value = idx_counter
            idx_counter += 1
            return value

        for _ in range(self.cfg.n_border_mimic_per_image):
            comp = _choose_component()
            if comp is None:
                break
            target_area = comp["area"] * float(rng.uniform(0.7, 1.3))
            patch = _sample_irregular_patch(
                comp["ring_mask"],
                min_area=max(self.cfg.min_patch_area_px, int(target_area * 0.5)),
                max_area=min(self.cfg.max_patch_area_px, int(target_area * 1.6)),
                min_radius=self.cfg.min_blob_radius,
                max_radius=self.cfg.max_blob_radius,
                min_blobs=1,
                max_blobs=2,
                max_tries=self.cfg.max_sampling_tries,
                target_area=target_area,
                area_tolerance=0.45,
            )
            if patch is None:
                continue
            rec = gen_and_save(patch, next_index(), neg_type="border_mimic")
            if rec is not None:
                records.append(rec)

        for _ in range(self.cfg.n_border_patches_per_image):
            comp = _choose_component()
            patch_mask = None
            if comp is not None and comp["ring_mask"].any():
                target_area = comp["area"] * float(rng.uniform(0.4, 0.9))
                patch_mask = _sample_irregular_patch(
                    comp["ring_mask"],
                    min_area=self.cfg.min_patch_area_px,
                    max_area=self.cfg.max_patch_area_px,
                    min_radius=self.cfg.min_blob_radius,
                    max_radius=self.cfg.max_blob_radius,
                    min_blobs=1,
                    max_blobs=self.cfg.max_blobs,
                    max_tries=self.cfg.max_sampling_tries,
                    target_area=target_area,
                    area_tolerance=0.5,
                )
            if patch_mask is None:
                patch_mask = _sample_irregular_patch(
                    ring_out,
                    min_area=self.cfg.min_patch_area_px,
                    max_area=self.cfg.max_patch_area_px,
                    min_radius=self.cfg.min_blob_radius,
                    max_radius=self.cfg.max_blob_radius,
                    min_blobs=self.cfg.min_blobs,
                    max_blobs=self.cfg.max_blobs,
                    max_tries=self.cfg.max_sampling_tries,
                )
            rec = gen_and_save(patch_mask, next_index(), neg_type="ring")
            if rec is not None:
                records.append(rec)

        for _ in range(self.cfg.n_bg_patches_per_image):
            patch = _sample_irregular_patch(
                bg_far,
                min_area=self.cfg.min_patch_area_px,
                max_area=self.cfg.max_patch_area_px,
                min_radius=self.cfg.min_blob_radius,
                max_radius=self.cfg.max_blob_radius,
                min_blobs=self.cfg.min_blobs,
                max_blobs=self.cfg.max_blobs,
                max_tries=self.cfg.max_sampling_tries,
            )
            rec = gen_and_save(patch, next_index(), neg_type="background")
            if rec is not None:
                records.append(rec)

        for _ in range(self.cfg.n_mixup_per_image):
            comp = _choose_component()
            if comp is None:
                break
            target_area = comp["area"] * float(rng.uniform(0.8, 1.4))
            patch = _sample_irregular_patch(
                bg_far,
                min_area=max(self.cfg.min_patch_area_px, int(target_area * 0.6)),
                max_area=min(self.cfg.max_patch_area_px, int(target_area * 1.6)),
                min_radius=self.cfg.min_blob_radius,
                max_radius=self.cfg.max_blob_radius + 10,
                min_blobs=self.cfg.min_blobs,
                max_blobs=self.cfg.max_blobs + 1,
                max_tries=self.cfg.max_sampling_tries,
                target_area=target_area,
                area_tolerance=0.5,
            )
            if patch is None:
                continue
            texture_source = {"rgb": comp["crop_rgb"], "mask": comp["mask_crop"]}
            rec = gen_and_save(patch, next_index(), neg_type="mix", texture_source=texture_source)
            if rec is not None:
                records.append(rec)

        return records

    def run(self) -> Dict[str, object]:
        pairs = _list_mask_image_pairs(self.image_root, self.model_dir)
        if not pairs:
            raise RuntimeError(f"No (image, mask) pairs found under {self.image_root} / {self.model_dir}")
        LOGGER.info("Found %d images with instance masks (model=%s)", len(pairs), self.model_dir.name)

        all_records: List[dict] = []
        for img_path, mask_path in pairs:
            try:
                recs = self._process_single(image_path=img_path, mask_path=mask_path)
                all_records.extend(recs)
            except Exception as exc:
                LOGGER.exception("Failed to generate negatives for %s: %s", img_path, exc)

        # Export metadata
        df = pd.DataFrame.from_records(all_records)
        root = self.repo_root.resolve()
        df = _prepare_for_export(df, root)
        out_csv = self.meta_dir / f"{self.cfg.out_class_name}.csv"
        df.to_csv(out_csv, index=False)
        LOGGER.info("Wrote Not-Berry metadata CSV: %s (%d rows)", out_csv, len(df))
        # Parquet
        if len(df) > 0:
            try:
                out_parq = self.meta_dir / f"{self.cfg.out_class_name}.parquet"
                df.to_parquet(out_parq, index=False)
                LOGGER.info("Wrote Not-Berry metadata Parquet: %s", out_parq)
            except Exception:
                LOGGER.warning("Failed to write Parquet for Not-Berry metadata; continuing with CSV only.")

        return {
            "generated": int(len(df)),
            "images": int(len(pairs)),
            "model": self.model_dir.name,
            "output": str(self.crops_dir),
            "metadata_csv": str(out_csv),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Not-Berry hard negatives from instance masks.")
    p.add_argument("--image-root", type=Path, default=NotBerryConfig.image_root)
    p.add_argument("--instance-masks-root", type=Path, default=NotBerryConfig.instance_masks_root)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--output-root", type=Path, default=NotBerryConfig.output_root)
    p.add_argument("--class-name", type=str, default=NotBerryConfig.out_class_name)
    p.add_argument("--include-mask", dest="include_mask", action="store_true")
    p.add_argument("--no-include-mask", dest="include_mask", action="store_false")
    p.set_defaults(include_mask=NotBerryConfig.include_mask_channel)
    p.add_argument("--bg-color", type=int, nargs=3, default=NotBerryConfig.background_color)
    p.add_argument("--n-bg", type=int, default=NotBerryConfig.n_bg_patches_per_image)
    p.add_argument("--n-border", type=int, default=NotBerryConfig.n_border_patches_per_image)
    p.add_argument("--n-border-mimic", type=int, default=NotBerryConfig.n_border_mimic_per_image)
    p.add_argument("--n-mix", type=int, default=NotBerryConfig.n_mixup_per_image)
    p.add_argument("--min-area", type=int, default=NotBerryConfig.min_patch_area_px)
    p.add_argument("--max-area", type=int, default=NotBerryConfig.max_patch_area_px)
    p.add_argument("--border-kernel", type=int, default=NotBerryConfig.border_dilate_kernel)
    p.add_argument("--margin", type=float, default=NotBerryConfig.margin)
    p.add_argument("--min-radius", type=int, default=NotBerryConfig.min_blob_radius)
    p.add_argument("--max-radius", type=int, default=NotBerryConfig.max_blob_radius)
    p.add_argument("--min-blobs", type=int, default=NotBerryConfig.min_blobs)
    p.add_argument("--max-blobs", type=int, default=NotBerryConfig.max_blobs)
    p.add_argument("--folds", type=int, default=NotBerryConfig.folds)
    p.add_argument("--seed", type=int, default=NotBerryConfig.random_seed)
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.set_defaults(skip_existing=NotBerryConfig.skip_existing)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = NotBerryConfig(
        image_root=args.image_root,
        instance_masks_root=args.instance_masks_root,
        model_name=args.model_name,
        output_root=args.output_root,
        out_class_name=args.class_name,
        include_mask_channel=args.include_mask,
        background_color=tuple(args.bg_color),
        n_bg_patches_per_image=max(0, int(args.n_bg)),
        n_border_patches_per_image=max(0, int(args.n_border)),
        n_border_mimic_per_image=max(0, int(args.n_border_mimic)),
        n_mixup_per_image=max(0, int(args.n_mix)),
        min_patch_area_px=max(1, int(args.min_area)),
        max_patch_area_px=max(1, int(args.max_area)),
        border_dilate_kernel=max(3, int(args.border_kernel) | 1),
        margin=float(args.margin),
        min_blob_radius=max(1, int(args.min_radius)),
        max_blob_radius=max(1, int(args.max_radius)),
        min_blobs=max(1, int(args.min_blobs)),
        max_blobs=max(1, int(args.max_blobs)),
        folds=max(1, int(args.folds)),
        random_seed=int(args.seed),
        skip_existing=bool(args.skip_existing),
    )
    gen = NotBerryGenerator(cfg)
    summary = gen.run()
    LOGGER.info("Summary: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
