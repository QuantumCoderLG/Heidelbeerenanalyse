from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from ..config import load_config
from ..config.paths import project_root
from ..data import ids, rasterize
from ..evaluation.apply_model_to_images import (
    instances_to_overlay,
    list_jpeg_images,
    postprocess_probability,
    preprocess_image,
    resolve_checkpoint,
)
from ..evaluation.postprocessing import apply_postprocessing
from ..training.models import build_model
from ..utils import color_norm, image_ops, io_utils


def _save_array_as_png(
    array: np.ndarray,
    path: Path,
    compress_level: int,
    mode: str | None = None,
) -> None:
    img = Image.fromarray(array, mode=mode) if mode else Image.fromarray(array)
    io_utils.atomic_save_pil_image(img, path, compress_level=compress_level)


def _save_pil_image(image: Image.Image, path: Path, compress_level: int) -> None:
    io_utils.atomic_save_pil_image(image, path, compress_level=compress_level)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QAThresholds:
    min_area_px: int | None = None
    max_area_px: int | None = None
    max_aspect_ratio: float | None = None
    min_circularity: float | None = None
    max_under_exposed_ratio: float | None = None
    max_over_exposed_ratio: float | None = None
    min_focus_measure: float | None = None
    under_exposed_threshold: int = 15
    over_exposed_threshold: int = 240


@dataclass
class ColorNormConfig:
    method: str = "gray_world"
    strength: float = 0.7
    max_gain: float = 1.6


@dataclass
class InstanceSplitConfig:
    enabled: bool = True
    min_circularity: float = 0.78
    min_area_px: int = 200
    min_peak_fraction: float = 0.45
    min_peak_distance_fraction: float = 0.5
    min_child_area_fraction: float = 0.18
    max_seeds: int = 3


@dataclass
class CropPipelineConfig:
    image_root: Path = Path("data/all_images/Ampel")
    checkpoint_path: Path | None = Path("Kanditaten/converted/fold_1_best.safetensors")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    config_path: Path = Path("configs/train.yaml")
    output_root: Path = Path("data/instance_crops")
    margin: float = 0.15
    background_color: Tuple[int, int, int] = (128, 128, 128)
    include_mask_channel: bool = True
    qa: QAThresholds = field(default_factory=QAThresholds)
    color: ColorNormConfig = field(default_factory=ColorNormConfig)
    split: InstanceSplitConfig = field(default_factory=InstanceSplitConfig)
    folds: int = 5
    stratify_on: Sequence[str] = field(
        default_factory=lambda: ("class_label", "lighting", "state")
    )
    random_seed: int = 1337
    max_annotations: int | None = None
    max_images: int | None = None
    skip_existing: bool = True
    write_parquet: bool = True
    device: str | None = None
    amp: bool | None = None
    save_overlays: bool = True
    overlay_max_instances: int = 25
    threshold_override: float | None = None
    border_trim_px: int = 5
    batch_size: int = 5
    num_save_workers: int = 4
    png_compress_level: int = 1
    roi_workers: int = 4
    # When set, write crops into this subfolder under images/masks instead of class-based directories.
    # Example: unlabeled_subdir="to_sort/batch_01" -> crops go to data/instance_crops/images/to_sort/batch_01/...
    unlabeled_subdir: str | None = None


@dataclass
class _PreparedInstanceResult:
    record: dict | None
    rejection: dict | None
    bounds: image_ops.CropBounds
    crop_mask: np.ndarray
    processed_rgb: np.ndarray
    annotation_id: int
    is_rejected: bool


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        parts = p.parts
        if parts and parts[0] == root.name:
            p = Path(*parts[1:])
        p = root / p
    return p.resolve()


def _infer_class_label(stem: str) -> str:
    s = stem.lower()
    if "never" in s:
        return "never"
    if "red" in s:
        return "red"
    if "yellow" in s:
        return "yellow"
    if "green" in s or "green" in s:
        return "green"
    return "unknown"


def _infer_state(stem: str) -> str:
    s = stem.lower()
    if "frisch" in s:
        return "frisch"
    if "aufgetaut" in s:
        return "aufgetaut"
    if "gefroren" in s:
        return "gefroren"
    return "unknown"


def _infer_lighting(stem: str) -> str:
    s = stem.upper()
    if "_WB" in s or s.endswith("WB"):
        return "WB"
    if "_MB" in s or s.endswith("MB"):
        return "MB"
    if "UEB" in s or "ÜB" in s:
        return "UEB"
    return "UNKNOWN"


def _compute_quality_metrics(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    original_area: int,
    qa_cfg: QAThresholds,
) -> Dict[str, float | int]:
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
        raise ValueError("crop_rgb must be RGB.")
    if crop_mask.shape != crop_rgb.shape[:2]:
        raise ValueError("crop_mask must match crop size.")

    mask_bool = crop_mask.astype(bool)
    area_px = int(mask_bool.sum())
    if area_px == 0:
        return {
            "area_px": 0,
            "aspect_ratio": math.inf,
            "circularity": 0.0,
            "relative_size": 0.0,
            "under_ratio": 1.0,
            "over_ratio": 0.0,
            "focus_measure": 0.0,
            "mean_intensity": 0.0,
        }

    h, w = crop_mask.shape
    aspect_ratio = max(h, w) / max(1, min(h, w))

    mask_uint = mask_bool.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0.0
    for cnt in contours:
        perimeter += cv2.arcLength(cnt, True)
    if perimeter <= 0.0:
        circularity = 0.0
    else:
        circularity = float(4.0 * math.pi * area_px / (perimeter ** 2))

    # Brightness stats computed inside the object mask
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    masked_gray = gray[mask_bool]
    under_ratio = float((masked_gray <= qa_cfg.under_exposed_threshold).mean())
    over_ratio = float((masked_gray >= qa_cfg.over_exposed_threshold).mean())
    focus_measure = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_intensity = float(masked_gray.mean())

    relative_size = float(area_px / max(1, original_area))

    return {
        "area_px": area_px,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "relative_size": relative_size,
        "under_ratio": under_ratio,
        "over_ratio": over_ratio,
        "focus_measure": focus_measure,
        "mean_intensity": mean_intensity,
    }


def _qa_fail_reasons(metrics: Dict[str, float | int], cfg: QAThresholds) -> List[str]:
    reasons: List[str] = []
    area = int(metrics["area_px"])
    if area == 0:
        return ["empty_mask"]
    if cfg.min_area_px is not None and area < cfg.min_area_px:
        reasons.append("min_area")
    if cfg.max_area_px is not None and area > cfg.max_area_px:
        reasons.append("max_area")

    if cfg.max_aspect_ratio is not None and float(metrics["aspect_ratio"]) > cfg.max_aspect_ratio:
        reasons.append("aspect_ratio")

    if cfg.min_circularity is not None and float(metrics["circularity"]) < cfg.min_circularity:
        reasons.append("circularity")

    if cfg.max_under_exposed_ratio is not None and float(metrics["under_ratio"]) > cfg.max_under_exposed_ratio:
        reasons.append("under_exposed")

    if cfg.max_over_exposed_ratio is not None and float(metrics["over_ratio"]) > cfg.max_over_exposed_ratio:
        reasons.append("over_exposed")

    if cfg.min_focus_measure is not None and float(metrics["focus_measure"]) < cfg.min_focus_measure:
        reasons.append("focus")

    return reasons


def _apply_color_normalisation(image: np.ndarray, cfg: ColorNormConfig) -> np.ndarray:
    if cfg.method.lower() == "gray_world":
        return color_norm.gray_world(image, strength=cfg.strength, max_gain=cfg.max_gain)
    raise ValueError(f"Unknown color normalisation method: {cfg.method}")


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
            "overlay_path",
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

    if "qa_reasons" in df.columns:
        df["qa_reasons"] = df["qa_reasons"].apply(
            lambda reasons: "|".join(reasons) if isinstance(reasons, (list, tuple)) else (reasons or "")
        )

    return df


# ---------------------------------------------------------------------------
# Mask post-processing helpers
# ---------------------------------------------------------------------------


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    if not mask.size:
        return mask.astype(bool)
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if mask.all() or not mask.any():
        return mask
    mask_uint8 = mask.astype(np.uint8)
    inv = (1 - mask_uint8) * 255
    h, w = inv.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(inv, flood_mask, (0, 0), 0)
    holes = inv == 255
    filled = mask_uint8.copy()
    filled[holes] = 1
    return filled.astype(bool)


def _trim_border(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask
    mask_uint = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_uint, kernel, iterations=iterations)
    return eroded.astype(bool)


def _split_mask_into_subcomponents(
    mask: np.ndarray,
    cfg: InstanceSplitConfig,
    rgb: np.ndarray | None = None,
) -> List[np.ndarray]:
    mask_bool = mask.astype(bool)
    area = int(mask_bool.sum())
    if not cfg.enabled or area == 0 or area < cfg.min_area_px:
        return [mask_bool]

    mask_uint8 = mask_bool.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [mask_bool]

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0:
        return [mask_bool]

    circularity = 4.0 * math.pi * contour_area / (perimeter * perimeter + 1e-6)
    if circularity >= cfg.min_circularity:
        return [mask_bool]

    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    if dist.size == 0:
        return [mask_bool]
    dist = cv2.GaussianBlur(dist, (5, 5), 0)
    max_dist = float(dist.max())
    if max_dist <= 0:
        return [mask_bool]

    kernel = np.ones((3, 3), dtype=np.float32)
    dilated = cv2.dilate(dist, kernel)
    peak_mask = (dist >= (dilated - 1e-6)) & mask_bool
    peak_mask &= dist >= (cfg.min_peak_fraction * max_dist)

    num_peaks, labels = cv2.connectedComponents(peak_mask.astype(np.uint8))
    peaks: List[Tuple[int, int, float]] = []
    for lbl in range(1, num_peaks):
        region = labels == lbl
        if not np.any(region):
            continue
        values = dist[region]
        idx = int(np.argmax(values))
        coords = np.argwhere(region)[idx]
        y, x = int(coords[0]), int(coords[1])
        peaks.append((y, x, float(values[idx])))

    if not peaks:
        return [mask_bool]

    peaks.sort(key=lambda item: item[2], reverse=True)
    filtered: List[Tuple[int, int, float]] = []
    min_center_distance = cfg.min_peak_distance_fraction * max_dist
    min_center_distance_sq = float(min_center_distance * min_center_distance)
    for y, x, val in peaks:
        if any((y - fy) ** 2 + (x - fx) ** 2 < min_center_distance_sq for fy, fx, _ in filtered):
            continue
        filtered.append((y, x, val))
        if len(filtered) >= cfg.max_seeds:
            break

    if len(filtered) < 2:
        return [mask_bool]

    # If RGB available, try color/texture-driven watershed first.
    if rgb is not None and rgb.ndim == 3 and rgb.shape[:2] == mask_bool.shape:
        # Prepare markers: 1=background, 2..K=object seeds
        markers = np.zeros(mask_bool.shape, dtype=np.int32)
        markers[~mask_bool] = 1

        for i, (y, x, _) in enumerate(filtered[: cfg.max_seeds], start=2):
            markers[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = i

        # Smooth and run watershed on BGR image (cv2 expects BGR)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.GaussianBlur(bgr, (5, 5), 0)

        # To confine segmentation inside the mask, set outside pixels to a uniform color
        bgr_masked = bgr.copy()
        bgr_masked[~mask_bool] = (0, 0, 0)

        cv2.watershed(bgr_masked, markers)

        child_masks = []
        child_areas = []
        for lab in range(2, markers.max() + 1):
            child = (markers == lab) & mask_bool
            if not child.any():
                continue
            child_masks.append(child)
            child_areas.append(int(child.sum()))
    else:
        # Fallback: Voronoi (distance to seeds)
        coords = np.column_stack(np.where(mask_bool))
        if coords.size == 0:
            return [mask_bool]

        seeds = np.array([[p[0], p[1]] for p in filtered], dtype=np.float32)
        diffs = coords[:, None, :] - seeds[None, :, :]
        dists = np.einsum("ijk,ijk->ij", diffs, diffs)
        assignments = np.argmin(dists, axis=1)

        child_masks = []
        child_areas = []
        for seed_idx in range(len(filtered)):
            member_coords = coords[assignments == seed_idx]
            if member_coords.size == 0:
                continue
            child_mask = np.zeros_like(mask_bool)
            child_mask[member_coords[:, 0], member_coords[:, 1]] = True
            child_masks.append(child_mask)
            child_areas.append(int(member_coords.shape[0]))

    if len(child_masks) < 2:
        return [mask_bool]

    min_child_area = cfg.min_child_area_fraction * area
    valid_indices = [idx for idx, area_val in enumerate(child_areas) if area_val >= min_child_area]
    if len(valid_indices) < 2:
        return [mask_bool]

    aggregated_masks: List[np.ndarray] = []
    for idx in valid_indices:
        aggregated_masks.append(child_masks[idx])

    if not aggregated_masks:
        return [mask_bool]

    coverage = np.zeros_like(mask_bool)
    for child in aggregated_masks:
        coverage |= child
    if not np.array_equal(coverage, mask_bool):
        residual = mask_bool & ~coverage
        if residual.any():
            largest_idx = int(np.argmax([m.sum() for m in aggregated_masks]))
            aggregated_masks[largest_idx][residual] = True

    return aggregated_masks


def _bounds_from_stats(
    x: int,
    y: int,
    w: int,
    h: int,
    margin: float,
    image_shape: Tuple[int, int],
) -> image_ops.CropBounds:
    pad_y = int(round(h * margin))
    pad_x = int(round(w * margin))
    y0 = max(0, y - pad_y)
    x0 = max(0, x - pad_x)
    y1 = min(image_shape[0], y + h + pad_y)
    x1 = min(image_shape[1], x + w + pad_x)
    if y1 <= y0:
        y1 = min(image_shape[0], y0 + 1)
    if x1 <= x0:
        x1 = min(image_shape[1], x0 + 1)
    return image_ops.CropBounds(y0=y0, y1=y1, x0=x0, x1=x1)


def _resize_instances_to_original(instances: np.ndarray, meta: Dict[str, float]) -> np.ndarray:
    orig_h = int(round(meta.get("original_height", instances.shape[0])))
    orig_w = int(round(meta.get("original_width", instances.shape[1])))
    if instances.shape[0] == orig_h and instances.shape[1] == orig_w:
        return instances.astype(np.int32, copy=False)
    resized = cv2.resize(
        instances.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.int32)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


class CropPipeline:
    def __init__(self, config: CropPipelineConfig) -> None:
        self.cfg = config
        module_root = Path(__file__).resolve().parents[2]
        candidate_roots = [module_root]
        try:
            candidate_roots.append(project_root().resolve())
        except Exception:  # pragma: no cover - project_root should not fail
            pass
        chosen_root = None
        for candidate in candidate_roots:
            if (candidate / "data").exists():
                chosen_root = candidate
                break
        self.repo_root = (chosen_root or module_root).resolve()
        self.image_root = _resolve_path(self.repo_root, self.cfg.image_root)
        self.checkpoint_dir = _resolve_path(self.repo_root, self.cfg.checkpoint_dir)
        self.checkpoint_path = None if self.cfg.checkpoint_path is None else _resolve_path(
            self.repo_root, self.cfg.checkpoint_path
        )
        self.config_path = _resolve_path(self.repo_root, self.cfg.config_path)
        self.output_root = _resolve_path(self.repo_root, self.cfg.output_root)
        self.crops_dir = self.output_root / "images"
        self.crops_mask_dir = self.output_root / "masks"
        self.meta_output_dir = self.output_root / "metadata"
        self.rejects_dir = self.output_root / "rejections"
        self.instance_masks_dir = self.output_root / "instance_masks"
        self.overlays_dir = self.output_root / "overlays"

        self.crops_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.include_mask_channel:
            self.crops_mask_dir.mkdir(parents=True, exist_ok=True)
        self.meta_output_dir.mkdir(parents=True, exist_ok=True)
        self.rejects_dir.mkdir(parents=True, exist_ok=True)
        self.instance_masks_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_overlays:
            self.overlays_dir.mkdir(parents=True, exist_ok=True)

        self._batch_size = max(1, int(self.cfg.batch_size))
        self._png_compress_level = int(min(9, max(0, self.cfg.png_compress_level)))
        max_save_workers = max(1, int(self.cfg.num_save_workers))
        self._save_executor = ThreadPoolExecutor(max_workers=max_save_workers, thread_name_prefix="save")
        self._pending_futures: List[Future] = []
        self._roi_workers = max(1, int(self.cfg.roi_workers))
        self._roi_executor = ThreadPoolExecutor(max_workers=self._roi_workers, thread_name_prefix="roi")

        self._model: nn.Module | None = None
        self._device: torch.device | None = None
        self._cfg_data: Dict[str, object] | None = None
        self._post_cfg: Dict[str, object] = {}
        self._mean: Sequence[float] = (0.485, 0.456, 0.406)
        self._std: Sequence[float] = (0.229, 0.224, 0.225)
        self._target_hw: Tuple[int, int] | None = None
        self._keep_ratio: bool = True
        self._threshold: float = 0.5
        self._channels_last: bool = False
        self._amp_enabled: bool = False
        self._model_name: str = "model"

    def _submit_save(self, func, *args, **kwargs) -> None:
        future = self._save_executor.submit(func, *args, **kwargs)
        self._pending_futures.append(future)

    def _wait_for_saves(self) -> None:
        while self._pending_futures:
            future = self._pending_futures.pop()
            future.result()

    def _start_prefetch(self, images: Sequence[Path]) -> Tuple[queue.Queue, object, threading.Thread]:
        max_queue = max(1, self._batch_size * 2)
        q: queue.Queue = queue.Queue(max_queue)
        stop_token = object()

        def producer() -> None:
            for path in images:
                rel_path = path.relative_to(self.image_root)
                mask_root = self.instance_masks_dir / self._model_name
                mask_path = mask_root / rel_path.with_suffix(".png")

                if self.cfg.skip_existing and mask_path.exists():
                    q.put(
                        {
                            "path": path,
                            "mask_path": mask_path,
                            "skip": True,
                        }
                    )
                    continue

                try:
                    image_np, width, height = io_utils.load_image(path)
                    q.put(
                        {
                            "path": path,
                            "mask_path": mask_path,
                            "image": image_np,
                            "width": width,
                            "height": height,
                            "skip": False,
                        }
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.exception("Failed to load image %s: %s", path, exc)
                    q.put(
                        {
                            "path": path,
                            "mask_path": mask_path,
                            "skip": True,
                            "error": exc,
                        }
                    )
            q.put(stop_token)

        thread = threading.Thread(target=producer, name="prefetch_reader", daemon=True)
        thread.start()
        return q, stop_token, thread

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, object]:
        self._ensure_model()
        images = list_jpeg_images(self.image_root)
        if self.cfg.max_images is not None:
            images = images[: self.cfg.max_images]
        LOGGER.info("Found %d JPEG images under %s", len(images), self.image_root)

        accepted_records: List[dict] = []
        rejected_records: List[dict] = []
        total_annotations = 0
        total_predicted = 0
        processed_images = 0
        skipped_images = 0

        prefetch_queue, stop_token, prefetch_thread = self._start_prefetch(images)
        drain_only = False
        batch_infos: List[dict] = []

        def process_batch(batch_infos_local: List[dict]) -> bool:
            nonlocal total_annotations, total_predicted, processed_images, skipped_images
            if not batch_infos_local:
                return False
            try:
                instances_batch = self._predict_instances_batch([info["image"] for info in batch_infos_local])
            except Exception as exc:  # pragma: no cover - defensive
                for info in batch_infos_local:
                    image_id = ids.stable_image_id(info["path"].stem)
                    LOGGER.exception("Failed to infer image %s: %s", info["path"], exc)
                    rejected_records.append(
                        {
                            "annotation_id": None,
                            "image_id": image_id,
                            "qa_reasons": ["exception"],
                            "exception": str(exc),
                            "source_image_path": info["path"],
                        }
                    )
                batch_infos_local.clear()
                return False

            max_reached_local = False
            for info, instances_raw in zip(batch_infos_local, instances_batch):
                if self.cfg.max_annotations is not None and total_annotations >= self.cfg.max_annotations:
                    max_reached_local = True
                    break

                remaining_local = None
                if self.cfg.max_annotations is not None:
                    remaining_local = max(0, self.cfg.max_annotations - total_annotations)
                    if remaining_local == 0:
                        max_reached_local = True
                        break

                try:
                    result = self._process_single(
                        image_path=info["path"],
                        image_np=info["image"],
                        instances_raw=instances_raw,
                        width=info["width"],
                        height=info["height"],
                        mask_path=info["mask_path"],
                        remaining=remaining_local,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    image_id = ids.stable_image_id(info["path"].stem)
                    LOGGER.exception("Failed to process image %s: %s", info["path"], exc)
                    rejected_records.append(
                        {
                            "annotation_id": None,
                            "image_id": image_id,
                            "qa_reasons": ["exception"],
                            "exception": str(exc),
                            "source_image_path": info["path"],
                        }
                    )
                    continue

                if result.get("skipped"):
                    skipped_images += 1
                    continue

                accepted_records.extend(result["accepted"])
                rejected_records.extend(result["rejected"])
                total_annotations += result["annotations"]
                total_predicted += result.get("predicted", 0)
                processed_images += 1

            batch_infos_local.clear()
            return max_reached_local

        try:
            while True:
                item = prefetch_queue.get()
                if item is stop_token:
                    if not drain_only and batch_infos:
                        if process_batch(batch_infos):
                            drain_only = True
                    break

                if not isinstance(item, dict):
                    continue

                if item.get("error"):
                    if item.get("path") is not None:
                        LOGGER.exception("Failed to load image %s: %s", item["path"], item["error"])
                    else:
                        LOGGER.exception("Prefetch worker error: %s", item["error"])
                    continue

                if item.get("skip"):
                    skipped_images += 1
                    continue

                if drain_only:
                    continue

                batch_infos.append(item)
                if len(batch_infos) >= self._batch_size:
                    if process_batch(batch_infos):
                        drain_only = True

            if not drain_only and batch_infos:
                process_batch(batch_infos)

            self._wait_for_saves()

            accepted_df = pd.DataFrame.from_records(accepted_records)
            rejected_df = pd.DataFrame.from_records(rejected_records)
            if not accepted_df.empty:
                accepted_df = self._assign_folds(accepted_df)

            summary = self._build_summary(
                total_images=processed_images,
                skipped_images=skipped_images,
                total_annotations=total_annotations,
                total_predicted=total_predicted,
                accepted_df=accepted_df,
                rejected_df=rejected_df,
            )

            self._write_outputs(accepted_df, rejected_df)

            summary_path = self.meta_output_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            LOGGER.info("Wrote summary to %s", summary_path)
            return summary

        finally:
            self._wait_for_saves()
            self._save_executor.shutdown(wait=True)
            if prefetch_thread.is_alive():
                prefetch_thread.join()
            # Ensure ROI workers are also shut down
            try:
                self._roi_executor.shutdown(wait=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Model initialisation & inference helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        cfg = load_config(self.config_path)
        self._cfg_data = cfg

        data_cfg = cfg.get("data", {})
        self._mean = tuple(float(x) for x in data_cfg.get("image_mean", self._mean))
        self._std = tuple(float(x) for x in data_cfg.get("image_std", self._std))
        target_size = data_cfg.get("val_size") or data_cfg.get("train_size")
        if target_size and len(target_size) == 2:
            self._target_hw = (int(target_size[0]), int(target_size[1]))
        self._keep_ratio = bool(data_cfg.get("keep_ratio", True))

        train_cfg = cfg.get("train", {})
        device_str = self.cfg.device or "cuda"
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but no CUDA-capable GPU is available.")
        self._device = torch.device(device_str)

        checkpoint_arg = str(self.checkpoint_path) if self.checkpoint_path is not None else None
        if not checkpoint_arg:
            raise ValueError("Crop pipeline requires a checkpoint_path pointing to the trained model.")
        checkpoint_path = resolve_checkpoint(checkpoint_arg, self.checkpoint_dir)

        LOGGER.info("Loading checkpoint %s", checkpoint_path)
        suffix = checkpoint_path.suffix.lower()
        meta_data: Dict[str, object] | None = None

        checkpoint_threshold = None
        if suffix == ".safetensors":
            try:
                from safetensors.torch import load_file as load_safetensors  # type: ignore
            except ImportError as err:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "safetensors checkpoint requested but 'safetensors' package is not installed"
                ) from err
            state_dict = load_safetensors(str(checkpoint_path))
            meta_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_meta.json")
            if meta_path.exists():
                try:
                    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as err:
                    LOGGER.warning("Failed to parse meta file %s: %s", meta_path, err)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            meta_data = checkpoint.get("meta") if isinstance(checkpoint, dict) else None
            state_dict = checkpoint.get("model", checkpoint)
            if isinstance(checkpoint, dict):
                checkpoint_threshold = checkpoint.get("threshold")

        model_cfg = dict(cfg.get("model", {}))
        if meta_data and isinstance(meta_data.get("model"), dict):
            model_cfg.update(meta_data["model"])  # type: ignore[index]

        model = build_model(model_cfg, num_classes=1)
        model.load_state_dict(state_dict)

        post_cfg = cfg.get("postproc", {})
        threshold = post_cfg.get("threshold", 0.5)
        if meta_data and "threshold" in meta_data:
            try:
                threshold = float(meta_data["threshold"])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                LOGGER.warning("Invalid threshold in meta data: %s", meta_data["threshold"])
        if checkpoint_threshold is not None:
            try:
                threshold = float(checkpoint_threshold)
            except (TypeError, ValueError):
                LOGGER.warning("Invalid threshold in checkpoint: %s", checkpoint_threshold)
        if self.cfg.threshold_override is not None:
            threshold = float(self.cfg.threshold_override)
        self._threshold = float(threshold)
        self._post_cfg = post_cfg
        self._model_name = checkpoint_path.stem

        channels_last = bool(train_cfg.get("channels_last", False))
        if channels_last:
            model = model.to(device=self._device, memory_format=torch.channels_last)
        else:
            model = model.to(self._device)
        model.eval()

        default_amp = bool(train_cfg.get("amp", True)) and self._device.type == "cuda"
        self._amp_enabled = bool(self.cfg.amp if self.cfg.amp is not None else default_amp)
        self._channels_last = channels_last
        self._model = model

        LOGGER.info(
            "Model ready on %s | threshold=%.3f | channels_last=%s | amp=%s",
            self._device,
            self._threshold,
            self._channels_last,
            self._amp_enabled,
        )

    def _predict_instances(self, image: np.ndarray) -> np.ndarray:
        return self._predict_instances_batch([image])[0]

    def _predict_instances_batch(self, images: Sequence[np.ndarray]) -> List[np.ndarray]:
        assert self._model is not None and self._device is not None
        if not images:
            return []

        tensors: List[torch.Tensor] = []
        metas: List[Dict[str, float]] = []
        for image in images:
            tensor, meta = preprocess_image(
                image=image,
                mean=self._mean,
                std=self._std,
                target_hw=self._target_hw,
                keep_ratio=self._keep_ratio,
            )
            tensors.append(tensor)
            metas.append(meta)

        batch_input = torch.stack(tensors, dim=0).to(self._device)
        if self._channels_last:
            batch_input = batch_input.to(memory_format=torch.channels_last)

        with torch.no_grad():
            ctx = (
                torch.amp.autocast("cuda", enabled=True)
                if self._amp_enabled and self._device.type == "cuda"
                else nullcontext()
            )
            with ctx:
                outputs = self._model(batch_input)
                logits = outputs["out"] if isinstance(outputs, dict) else outputs

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        instances_full: List[np.ndarray] = []
        for prob_map, meta in zip(probs, metas):
            probs_cropped = postprocess_probability(prob_map, meta, resize_to_original=False)
            _, instances_small = apply_postprocessing(probs_cropped, self._post_cfg, threshold=self._threshold)
            instances_resized = _resize_instances_to_original(instances_small, meta)
            instances_full.append(instances_resized.astype(np.int32))

        return instances_full

    # ------------------------------------------------------------------
    # Image-level processing
    # ------------------------------------------------------------------

    def _process_single(
        self,
        *,
        image_path: Path,
        image_np: np.ndarray,
        instances_raw: np.ndarray,
        width: int,
        height: int,
        mask_path: Path,
        remaining: int | None,
    ) -> Dict[str, object]:
        rel_path = image_path.relative_to(self.image_root)

        # Use instance labels directly (do not merge touching instances)
        instance_labels = [int(x) for x in np.unique(instances_raw) if int(x) != 0]
        overlay_path: Path | None = None
        if self.cfg.save_overlays and instance_labels:
            overlay_root = self.overlays_dir / self._model_name
            overlay_rel = rel_path.with_name(f"{rel_path.stem}_overlay.png")
            overlay_path = overlay_root / overlay_rel
            overlay_img = instances_to_overlay(
                image_np,
                instances_raw,
                max_instances=self.cfg.overlay_max_instances,
            )
            self._submit_save(
                _save_pil_image,
                overlay_img.copy(),
                overlay_path,
                self._png_compress_level,
            )

        ann_mask = np.zeros_like(instances_raw, dtype=np.uint32)
        image_id = ids.stable_image_id(image_path.stem)
        original_area = int(width * height)

        accepted: List[dict] = []
        rejected: List[dict] = []
        accepted_in_image = 0

        image_shape = instances_raw.shape
        component_infos: List[Tuple[float, float, image_ops.CropBounds, np.ndarray]] = []
        for label in instance_labels:
            full_mask = (instances_raw == int(label))
            if not full_mask.any():
                continue
            ys, xs = np.where(full_mask)
            y = int(ys.min())
            x = int(xs.min())
            h = int(ys.max() - y + 1)
            w = int(xs.max() - x + 1)
            bounds = _bounds_from_stats(x, y, w, h, self.cfg.margin, image_shape)
            crop_mask = full_mask[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
            split_components = self._split_component(bounds, crop_mask, image_shape, image_np)
            for sub_bounds, sub_mask in split_components:
                center_y = (sub_bounds.y0 + sub_bounds.y1) / 2.0
                center_x = (sub_bounds.x0 + sub_bounds.x1) / 2.0
                component_infos.append((center_y, center_x, sub_bounds, sub_mask))

        component_infos.sort(key=lambda item: item[0])
        total_components = len(component_infos)
        total_predicted = total_components
        if total_components:
            estimated_cols = max(1, int(round(math.sqrt(total_components))))
            estimated_rows = max(1, int(math.ceil(total_components / estimated_cols)))
            y_values = [item[0] for item in component_infos]
            diffs = np.diff(y_values) if len(y_values) > 1 else np.array([])
            row_breaks: List[int] = []
            if estimated_rows > 1 and diffs.size >= estimated_rows - 1:
                indices = np.argsort(diffs)[-(estimated_rows - 1) :]
                row_breaks = sorted(int(idx) + 1 for idx in np.atleast_1d(indices))
            rows: List[List[Tuple[float, float, image_ops.CropBounds, np.ndarray]]] = []
            start = 0
            for break_idx in row_breaks:
                rows.append(component_infos[start:break_idx])
                start = break_idx
            rows.append(component_infos[start:])
            if estimated_rows > 1 and len(rows) == 1:
                split_arrays = np.array_split(component_infos, estimated_rows)
                rows = [list(chunk) for chunk in split_arrays if len(chunk)]
            ordered_components: List[Tuple[image_ops.CropBounds, np.ndarray]] = []
            for row in rows:
                row.sort(key=lambda item: item[1])
                ordered_components.extend((bounds, crop_mask) for _, _, bounds, crop_mask in row)
        else:
            ordered_components = []

        pending: List[Tuple[int, Future[_PreparedInstanceResult]]] = []

        def _drain_pending(force: bool = False) -> None:
            nonlocal accepted_in_image
            while pending and (force or len(pending) >= self._roi_workers):
                local_idx, future = pending.pop(0)
                try:
                    result = future.result()
                except Exception:
                    LOGGER.exception(
                        "Instance future failed for image %s (index %d)",
                        image_path,
                        local_idx,
                    )
                    continue
                if remaining is not None and accepted_in_image >= remaining:
                    continue
                if result.is_rejected:
                    if result.rejection is not None:
                        rejected.append(result.rejection)
                        self._write_reject(
                            result.annotation_id,
                            result.processed_rgb,
                            result.crop_mask,
                        )
                else:
                    if result.record is None:
                        continue
                    accepted.append(result.record)
                    accepted_in_image += 1
                    bounds = result.bounds
                    view = ann_mask[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
                    view[result.crop_mask] = np.uint32(result.annotation_id)
                    self._write_crop(result.record["crop_path"], result.processed_rgb)
                    mask_path_rec = result.record.get("mask_path")
                    if self.cfg.include_mask_channel and mask_path_rec:
                        self._write_mask(mask_path_rec, result.crop_mask)

        for ordered_index, (bounds, crop_mask) in enumerate(ordered_components):
            if remaining is not None and accepted_in_image >= remaining:
                LOGGER.info(
                    "Reached annotation limit while processing %s; skipping remaining instances.",
                    rel_path,
                )
                break

            annotation_id = ids.stable_annotation_id(image_id, ordered_index)
            future = self._roi_executor.submit(
                self._process_instance,
                image_np=image_np,
                crop_mask=crop_mask,
                bounds=bounds,
                image_path=image_path,
                image_id=image_id,
                annotation_id=annotation_id,
                aggregated_mask_path=mask_path,
                overlay_path=overlay_path,
                original_area=original_area,
                index_within_image=ordered_index,
            )
            pending.append((ordered_index, future))
            _drain_pending()

        _drain_pending(force=True)

        self._write_instance_mask(mask_path, ann_mask.astype(np.uint32, copy=False))

        return {
            "accepted": accepted,
            "rejected": rejected,
            "annotations": len(accepted),
            "predicted": total_predicted,
            "skipped": False,
        }

    # ------------------------------------------------------------------
    # Instance-level processing helpers
    # ------------------------------------------------------------------

    def _split_component(
        self,
        bounds: image_ops.CropBounds,
        crop_mask: np.ndarray,
        image_shape: Tuple[int, int],
        image_rgb_full: np.ndarray,
    ) -> List[Tuple[image_ops.CropBounds, np.ndarray]]:
        # Extract the corresponding RGB crop from the full image for color-driven splitting
        rgb_roi = image_rgb_full[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
        split_masks = _split_mask_into_subcomponents(crop_mask, self.cfg.split, rgb=rgb_roi)
        if len(split_masks) <= 1:
            return [(bounds, crop_mask)]

        components: List[Tuple[image_ops.CropBounds, np.ndarray]] = []
        for local_mask in split_masks:
            if not local_mask.any():
                continue

            local_y, local_x = np.nonzero(local_mask)
            local_y0 = int(local_y.min())
            local_y1 = int(local_y.max()) + 1
            local_x0 = int(local_x.min())
            local_x1 = int(local_x.max()) + 1

            global_y0 = bounds.y0 + local_y0
            global_x0 = bounds.x0 + local_x0
            width = int(local_x1 - local_x0)
            height = int(local_y1 - local_y0)

            sub_bounds = _bounds_from_stats(
                global_x0,
                global_y0,
                width,
                height,
                self.cfg.margin,
                image_shape,
            )

            sub_mask = np.zeros((sub_bounds.height, sub_bounds.width), dtype=bool)
            insert_y0 = (bounds.y0 + local_y0) - sub_bounds.y0
            insert_x0 = (bounds.x0 + local_x0) - sub_bounds.x0
            insert_y1 = insert_y0 + height
            insert_x1 = insert_x0 + width
            sub_mask[
                insert_y0:insert_y1,
                insert_x0:insert_x1,
            ] = local_mask[local_y0:local_y1, local_x0:local_x1]
            components.append((sub_bounds, sub_mask))

        if not components:
            return [(bounds, crop_mask)]
        return components

    def _process_instance(
        self,
        *,
        image_np: np.ndarray,
        crop_mask: np.ndarray,
        bounds: image_ops.CropBounds,
        image_path: Path,
        image_id: int,
        annotation_id: int,
        aggregated_mask_path: Path,
        overlay_path: Path | None,
        original_area: int,
        index_within_image: int,
    ) -> _PreparedInstanceResult:
        if crop_mask.dtype != np.bool_:
            crop_mask = crop_mask.astype(bool)

        crop_rgb = image_np[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1]
        crop_mask_proc = _fill_holes(crop_mask)
        crop_mask_proc = _trim_border(crop_mask_proc, self.cfg.border_trim_px)

        class_label = _infer_class_label(image_path.stem)
        state = _infer_state(image_path.stem)
        lighting = _infer_lighting(image_path.stem)

        processed_rgb = image_ops.apply_background(
            crop_rgb.copy(),
            crop_mask_proc,
            self.cfg.background_color,
        )

        if not crop_mask_proc.any():
            rejection = {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "qa_reasons": ["border_trim_empty"],
                "source_image_path": image_path,
                "class_label": class_label,
                "state": state,
                "lighting": lighting,
                "instances_mask_path": aggregated_mask_path,
                "overlay_path": overlay_path,
                "index_within_image": index_within_image,
                "ordered_id": index_within_image + 1,
            }
            rejection.update(self._source_tags(image_path))
            return _PreparedInstanceResult(
                record=None,
                rejection=rejection,
                bounds=bounds,
                crop_mask=crop_mask_proc,
                processed_rgb=processed_rgb,
                annotation_id=annotation_id,
                is_rejected=True,
            )

        processed_rgb = _apply_color_normalisation(processed_rgb, self.cfg.color)

        metrics = _compute_quality_metrics(
            crop_rgb=processed_rgb,
            crop_mask=crop_mask_proc,
            original_area=original_area,
            qa_cfg=self.cfg.qa,
        )
        reasons = _qa_fail_reasons(metrics, self.cfg.qa)

        if reasons:
            LOGGER.debug("Annotation %s rejected: %s", annotation_id, reasons)
            rejection = {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "qa_reasons": reasons,
                "source_image_path": image_path,
                "class_label": class_label,
                "state": state,
                "lighting": lighting,
                "instances_mask_path": aggregated_mask_path,
                "overlay_path": overlay_path,
                "index_within_image": index_within_image,
                "ordered_id": index_within_image + 1,
            }
            rejection.update(self._source_tags(image_path))
            return _PreparedInstanceResult(
                record=None,
                rejection=rejection,
                bounds=bounds,
                crop_mask=crop_mask_proc,
                processed_rgb=processed_rgb,
                annotation_id=annotation_id,
                is_rejected=True,
            )

        crop_id = index_within_image + 1
        crop_filename = f"{image_path.stem}_id{crop_id:03d}.png"
        # Destination subdir: class label or user-provided unlabeled staging subdir
        dest_dir = Path(self.cfg.unlabeled_subdir) if self.cfg.unlabeled_subdir else Path(class_label)
        crop_rel_path = dest_dir / crop_filename
        crop_abs_path = self.crops_dir / crop_rel_path

        mask_abs_path = None
        if self.cfg.include_mask_channel:
            mask_abs_path = self.crops_mask_dir / crop_rel_path

        record = self._build_metadata_record(
            image_path=image_path,
            annotation_id=annotation_id,
            image_id=image_id,
            crop_path=crop_abs_path,
            mask_path=mask_abs_path,
            bounds=bounds,
            metrics=metrics,
            overlay_path=overlay_path,
            instances_mask_path=aggregated_mask_path,
            index_within_image=index_within_image,
        )
        record["ordered_id"] = crop_id

        return _PreparedInstanceResult(
            record=record,
            rejection=None,
            bounds=bounds,
            crop_mask=crop_mask_proc,
            processed_rgb=processed_rgb,
            annotation_id=annotation_id,
            is_rejected=False,
        )

    def _write_crop(self, path: Path, image: np.ndarray) -> None:
        self._submit_save(
            _save_array_as_png,
            image.copy(),
            path,
            self._png_compress_level,
        )

    def _write_mask(self, path: Path, mask: np.ndarray) -> None:
        mask_uint8 = (mask.astype(np.uint8) * 255)
        self._submit_save(
            _save_array_as_png,
            mask_uint8,
            path,
            self._png_compress_level,
            "L",
        )

    def _write_reject(self, annotation_id: int, image: np.ndarray, mask: np.ndarray) -> None:
        out_img = self.rejects_dir / f"{annotation_id}.png"
        out_mask = self.rejects_dir / f"{annotation_id}_mask.png"
        self._submit_save(
            _save_array_as_png,
            image.copy(),
            out_img,
            self._png_compress_level,
        )
        mask_uint8 = (mask.astype(np.uint8) * 255)
        self._submit_save(
            _save_array_as_png,
            mask_uint8,
            out_mask,
            self._png_compress_level,
            "L",
        )

    def _write_instance_mask(self, path: Path, mask: np.ndarray) -> None:
        self._submit_save(
            rasterize.save_instance_mask_png,
            mask.copy(),
            path,
            compress_level=self._png_compress_level,
        )

    def _source_tags(self, image_path: Path) -> Dict[str, object]:
        rel = image_path.relative_to(self.image_root)
        parts = rel.parts
        source_group = parts[0] if len(parts) >= 1 else ""
        source_subgroup = parts[1] if len(parts) >= 2 else ""
        return {
            "source_group": source_group,
            "source_subgroup": source_subgroup,
            "source_rel_path": rel,
        }

    def _build_metadata_record(
        self,
        *,
        image_path: Path,
        annotation_id: int,
        image_id: int,
        crop_path: Path,
        mask_path: Path | None,
        bounds: image_ops.CropBounds,
        metrics: Dict[str, float | int],
        overlay_path: Path | None,
        instances_mask_path: Path,
        index_within_image: int,
    ) -> dict:
        tags = self._source_tags(image_path)
        record = {
            "annotation_id": int(annotation_id),
            "image_id": int(image_id),
            "class_label": _infer_class_label(image_path.stem),
            "state": _infer_state(image_path.stem),
            "lighting": _infer_lighting(image_path.stem),
            "scene_stem": image_path.stem,
            "crop_path": crop_path,
            "mask_path": mask_path,
            "crop_height": bounds.height,
            "crop_width": bounds.width,
            "qa_reasons": [],
            "source_image_path": image_path,
            "instances_mask_path": instances_mask_path,
            "overlay_path": overlay_path,
            "index_within_image": index_within_image,
        }
        record.update(metrics)
        record.update(tags)
        return record

    # ------------------------------------------------------------------
    # Fold assignment
    # ------------------------------------------------------------------

    def _assign_folds(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n_splits = max(2, int(self.cfg.folds))
        rng = random.Random(self.cfg.random_seed)

        groups = df["image_id"].tolist()
        strat_keys = []
        for _, row in df.iterrows():
            values = []
            for key in self.cfg.stratify_on:
                values.append(str(row.get(key, "unknown")))
            strat_keys.append("|".join(values))

        fold_assignments = self._group_stratified_assign(groups, strat_keys, n_splits, rng)
        df["fold_id"] = fold_assignments
        return df

    def _group_stratified_assign(
        self,
        groups: List[int],
        strat_keys: List[str],
        n_splits: int,
        rng: random.Random,
    ) -> List[int]:
        global_counts = Counter(strat_keys)
        expected_per_fold = {key: count / n_splits for key, count in global_counts.items()}

        group_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, group in enumerate(groups):
            group_indices[group].append(idx)

        grouped_info = []
        for group, idxs in group_indices.items():
            counts = Counter(strat_keys[i] for i in idxs)
            grouped_info.append((group, counts, len(idxs)))

        rng.shuffle(grouped_info)
        grouped_info.sort(key=lambda x: x[2], reverse=True)

        fold_class_counts = [Counter() for _ in range(n_splits)]
        fold_sizes = [0] * n_splits
        assignments: Dict[int, int] = {}

        for group, counts, size in grouped_info:
            best_fold = None
            best_score = None
            for fold_idx in range(n_splits):
                new_counts = fold_class_counts[fold_idx].copy()
                for key, val in counts.items():
                    new_counts[key] += val
                size_score = fold_sizes[fold_idx] + size
                imbalance = 0.0
                for key, expected in expected_per_fold.items():
                    diff = new_counts.get(key, 0.0) - expected
                    imbalance += diff * diff
                score = (size_score, imbalance)
                if best_score is None or score < best_score:
                    best_score = score
                    best_fold = fold_idx
            assert best_fold is not None
            assignments[group] = best_fold
            fold_sizes[best_fold] += size
            for key, val in counts.items():
                fold_class_counts[best_fold][key] += val

        return [assignments[group] for group in groups]

    # ------------------------------------------------------------------
    # Output writers and summary
    # ------------------------------------------------------------------

    def _write_outputs(self, accepted_df: pd.DataFrame, rejected_df: pd.DataFrame) -> None:
        root = self.repo_root
        accepted_out = _prepare_for_export(accepted_df, root)
        rejected_out = _prepare_for_export(rejected_df, root)

        accepted_csv = self.meta_output_dir / "crops.csv"
        accepted_out.to_csv(accepted_csv, index=False)
        LOGGER.info("Wrote accepted crop metadata: %s", accepted_csv)

        if self.cfg.write_parquet and not accepted_out.empty:
            try:
                accepted_parq = self.meta_output_dir / "crops.parquet"
                accepted_out.to_parquet(accepted_parq, index=False)
                LOGGER.info("Wrote accepted crop metadata (Parquet): %s", accepted_parq)
            except Exception:
                LOGGER.exception("Failed to write crops.parquet")

        rejected_csv = self.meta_output_dir / "crops_rejected.csv"
        rejected_out.to_csv(rejected_csv, index=False)
        LOGGER.info("Wrote rejected crop metadata: %s", rejected_csv)

        if self.cfg.write_parquet and not rejected_out.empty:
            try:
                rejected_parq = self.meta_output_dir / "crops_rejected.parquet"
                rejected_out.to_parquet(rejected_parq, index=False)
                LOGGER.info("Wrote rejected crop metadata (Parquet): %s", rejected_parq)
            except Exception:
                LOGGER.exception("Failed to write crops_rejected.parquet")

    def _build_summary(
        self,
        *,
        total_images: int,
        skipped_images: int,
        total_annotations: int,
        total_predicted: int,
        accepted_df: pd.DataFrame,
        rejected_df: pd.DataFrame,
    ) -> Dict[str, object]:
        summary = {
            "total_images_processed": total_images,
            "skipped_images": skipped_images,
            "predicted_instances": total_predicted,
            "accepted_annotations": int(len(accepted_df)),
            "rejected_annotations": int(len(rejected_df)),
            "qa_failures": {},
            "fold_distribution": {},
        }
        if not rejected_df.empty and "qa_reasons" in rejected_df.columns:
            exploded = rejected_df["qa_reasons"].explode().dropna()
            summary["qa_failures"] = exploded.value_counts().to_dict()
        if not accepted_df.empty and "fold_id" in accepted_df.columns:
            summary["fold_distribution"] = (
                accepted_df["fold_id"].value_counts().sort_index().to_dict()
            )
        return summary


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment blueberries with a trained model and build QA-checked crops.")
    parser.add_argument("--image-root", type=Path, default=CropPipelineConfig.image_root)
    parser.add_argument("--checkpoint", type=Path, default=CropPipelineConfig.checkpoint_path)
    parser.add_argument("--checkpoint-dir", type=Path, default=CropPipelineConfig.checkpoint_dir)
    parser.add_argument("--config", type=Path, default=CropPipelineConfig.config_path)
    parser.add_argument("--output-root", type=Path, default=CropPipelineConfig.output_root)
    parser.add_argument("--margin", type=float, default=CropPipelineConfig.margin)
    parser.add_argument("--background-color", type=int, nargs=3, default=CropPipelineConfig.background_color)
    parser.add_argument("--max-annotations", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--folds", type=int, default=CropPipelineConfig.folds)
    parser.add_argument("--random-seed", type=int, default=CropPipelineConfig.random_seed)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Override probability threshold for post-processing")
    parser.add_argument("--qa-min-area", type=int, default=None, help="Minimum instance area (px) to accept")
    parser.add_argument("--qa-max-area", type=int, default=None, help="Maximum instance area (px) to accept")
    parser.add_argument("--qa-max-aspect", type=float, default=None, help="Maximum allowed aspect ratio (>=1)")
    parser.add_argument("--qa-min-circularity", type=float, default=None, help="Minimum circularity (0-1) to accept")
    parser.add_argument("--qa-max-under-ratio", type=float, default=None, help="Maximum fraction of under-exposed pixels inside berry")
    parser.add_argument("--qa-max-over-ratio", type=float, default=None, help="Maximum fraction of over-exposed pixels inside berry")
    parser.add_argument("--qa-min-focus", type=float, default=None, help="Minimum Laplacian variance to accept")
    parser.add_argument("--qa-under-threshold", type=int, default=QAThresholds().under_exposed_threshold, help="Intensity threshold (0-255) for under-exposed detection")
    parser.add_argument("--qa-over-threshold", type=int, default=QAThresholds().over_exposed_threshold, help="Intensity threshold (0-255) for over-exposed detection")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--include-mask-channel", dest="include_mask_channel", action="store_true")
    parser.add_argument("--no-mask-channel", dest="include_mask_channel", action="store_false")
    parser.set_defaults(include_mask_channel=CropPipelineConfig.include_mask_channel)
    parser.add_argument("--save-overlays", dest="save_overlays", action="store_true")
    parser.add_argument("--no-overlays", dest="save_overlays", action="store_false")
    parser.set_defaults(save_overlays=CropPipelineConfig.save_overlays)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=CropPipelineConfig.skip_existing)
    parser.add_argument("--overlay-max-instances", type=int, default=CropPipelineConfig.overlay_max_instances)
    parser.add_argument("--border-trim", type=int, default=CropPipelineConfig.border_trim_px, help="Number of erosion iterations to trim instance borders before cropping")
    parser.add_argument("--batch-size", type=int, default=CropPipelineConfig.batch_size)
    parser.add_argument("--num-save-workers", type=int, default=CropPipelineConfig.num_save_workers)
    parser.add_argument("--png-compress-level", type=int, default=CropPipelineConfig.png_compress_level)
    parser.add_argument("--roi-workers", type=int, default=CropPipelineConfig.roi_workers)
    parser.add_argument(
        "--unlabeled-subdir",
        type=str,
        default=None,
        help=(
            "If set, crops are saved under data/instance_crops/images/<value>/ instead of class subfolders. "
            "Allows manual sorting of unlabeled crops afterwards."
        ),
    )
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    qa_cfg = QAThresholds(
        min_area_px=args.qa_min_area,
        max_area_px=args.qa_max_area,
        max_aspect_ratio=args.qa_max_aspect,
        min_circularity=args.qa_min_circularity,
        max_under_exposed_ratio=args.qa_max_under_ratio,
        max_over_exposed_ratio=args.qa_max_over_ratio,
        min_focus_measure=args.qa_min_focus,
        under_exposed_threshold=args.qa_under_threshold,
        over_exposed_threshold=args.qa_over_threshold,
    )

    config = CropPipelineConfig(
        image_root=args.image_root,
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        config_path=args.config,
        output_root=args.output_root,
        margin=args.margin,
        background_color=tuple(args.background_color),
        max_annotations=args.max_annotations,
        max_images=args.max_images,
        folds=args.folds,
        random_seed=args.random_seed,
        device=args.device,
        threshold_override=args.threshold,
        include_mask_channel=args.include_mask_channel,
        save_overlays=args.save_overlays,
        skip_existing=args.skip_existing,
        overlay_max_instances=args.overlay_max_instances,
        amp=args.amp,
        border_trim_px=max(0, int(args.border_trim)),
        batch_size=max(1, int(args.batch_size)),
        num_save_workers=max(1, int(args.num_save_workers)),
        png_compress_level=args.png_compress_level,
        qa=qa_cfg,
        unlabeled_subdir=(args.unlabeled_subdir if args.unlabeled_subdir not in {None, ""} else None),
    )
    pipeline = CropPipeline(config)
    pipeline.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
