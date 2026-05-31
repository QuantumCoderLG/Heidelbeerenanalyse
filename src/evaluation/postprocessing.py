from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .metrics import MetricsConfig, aggregate_metrics, evaluate_image


@dataclass
class PostProcessConfig:
    threshold: float = 0.5
    min_area: int = 50
    max_area: int | None = None
    circularity_enabled: bool = True
    circularity_min: float = 0.3
    morph_open: int = 3
    morph_close: int = 5
    morph_iterations: int = 1
    watershed_enabled: bool = True
    watershed_rel_thresh: float = 0.4
    watershed_min_distance: int = 5


def apply_postprocessing(
    prob_map: np.ndarray,
    config: Dict[str, any],
    threshold: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = copy.deepcopy(config)
    thresh = float(threshold if threshold is not None else cfg.get("threshold", 0.5))
    binary = (prob_map >= thresh).astype(np.uint8)
    binary = _apply_morphology(binary, cfg.get("morphology", {}))
    if cfg.get("watershed", {}).get("enabled", True):
        instances = _apply_watershed(binary, prob_map, cfg.get("watershed", {}))
    else:
        instances = _connected_components(binary)
    instances = _filter_components(instances, cfg)
    binary = (instances > 0).astype(np.uint8)
    return binary, instances


def _apply_morphology(binary: np.ndarray, morph_cfg: Dict[str, any]) -> np.ndarray:
    kernel_open = int(morph_cfg.get("open_kernel", 0) or 0)
    kernel_close = int(morph_cfg.get("close_kernel", 0) or 0)
    iterations = int(morph_cfg.get("iterations", 1))
    result = binary.copy()
    if kernel_open > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open, kernel_open))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if kernel_close > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return result


def _connected_components(binary: np.ndarray) -> np.ndarray:
    num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))
    if num_labels <= 1:
        return labels.astype(np.int32)
    return labels.astype(np.int32)


def _filter_components(instances: np.ndarray, cfg: Dict[str, any]) -> np.ndarray:
    min_area = int(cfg.get("min_area", 0) or 0)
    max_area_raw = cfg.get("max_area")
    max_area = int(max_area_raw) if max_area_raw else None
    circ_cfg = cfg.get("circularity", {})
    circ_enabled = bool(circ_cfg.get("enabled", False))
    circ_min = float(circ_cfg.get("min", 0.0))

    filtered = np.zeros_like(instances, dtype=np.int32)
    label_indices = [label for label in np.unique(instances) if label > 0]
    next_label = 1
    for label in label_indices:
        mask = (instances == label).astype(np.uint8)
        area = int(mask.sum())
        if area == 0:
            continue
        if min_area and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if circ_enabled:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(contour, closed=True))
            if perimeter == 0:
                continue
            circularity = 4.0 * np.pi * float(area) / (perimeter ** 2)
            if circularity < circ_min:
                continue
        filtered[instances == label] = next_label
        next_label += 1
    return filtered


def _apply_watershed(binary: np.ndarray, prob_map: np.ndarray, cfg: Dict[str, any]) -> np.ndarray:
    if binary.max() == 0:
        return np.zeros_like(binary, dtype=np.int32)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    rel_thresh = float(cfg.get("peak_rel_threshold", 0.4))
    _, sure_fg = cv2.threshold(distance, rel_thresh * distance.max(), 1, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_fg = cv2.erode(sure_fg, np.ones((3, 3), np.uint8), iterations=int(cfg.get("peak_min_distance", 1)))
    unknown = cv2.subtract(binary, sure_fg)
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 1] = 0
    img = cv2.normalize(prob_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    # OpenCV's watershed modifies markers in-place. By convention:
    #   - Boundaries are set to -1
    #   - Background remains label 1 (since we added +1 above)
    # We must convert both to background=0 for downstream processing/visualisation.
    cv2.watershed(img, markers)
    # Drop boundaries
    markers[markers < 0] = 0
    # Map background (label==1) to 0 so it is not treated as an instance
    markers[markers == 1] = 0
    return markers.astype(np.int32)


def search_optimal_threshold(
    prob_maps: Sequence[np.ndarray],
    gt_instances: Sequence[np.ndarray],
    post_cfg: Dict[str, any],
    metrics_cfg: Dict[str, any],
) -> Tuple[float, List[Tuple[float, float]]]:
    sweep_cfg = post_cfg.get("threshold_search", {})
    if not sweep_cfg.get("enabled", False):
        base = float(post_cfg.get("threshold", 0.5))
        return base, [(base, 0.0)]
    low = float(sweep_cfg.get("min", 0.1))
    high = float(sweep_cfg.get("max", 0.9))
    steps = int(sweep_cfg.get("num_steps", 10))
    metric_name = str(sweep_cfg.get("metric", "median_iou")).lower()
    thresholds = np.linspace(low, high, steps)
    history: List[Tuple[float, float]] = []
    best_threshold = float(post_cfg.get("threshold", 0.5))
    best_score = -1.0
    metrics_config = MetricsConfig(
        count_target=int(metrics_cfg.get("count_target", 25)),
        merge_iou_threshold=float(metrics_cfg.get("merge_iou_threshold", 0.1)),
        miss_iou_threshold=float(metrics_cfg.get("miss_iou_threshold", 0.2)),
    )
    key_map = {
        "median_iou": "instance_iou_median",
        "dice": "instance_dice_mean",
        "pixel_iou": "pixel_iou",
    }
    metric_key = key_map.get(metric_name, "instance_iou_median")
    for thr in thresholds:
        metrics_per_image = []
        for prob, gt in zip(prob_maps, gt_instances):
            _, instances = apply_postprocessing(prob, post_cfg, threshold=float(thr))
            metrics_per_image.append(evaluate_image(instances, gt, metrics_config))
        aggregated = aggregate_metrics(metrics_per_image)
        score = float(aggregated.get(metric_key, 0.0))
        history.append((float(thr), score))
        if score > best_score:
            best_score = score
            best_threshold = float(thr)
    return best_threshold, history


def count_guided_threshold(
    prob_map: np.ndarray,
    base_threshold: float,
    target_count: int,
    cfg: Dict[str, any],
) -> float:
    search_cfg = cfg.get("count_guided", {})
    if not search_cfg.get("enabled", False):
        return base_threshold
    window = float(search_cfg.get("window", 0.2))
    steps = int(search_cfg.get("steps", 7))
    tolerance = int(search_cfg.get("tolerance", 1))
    lo = max(0.01, base_threshold - window)
    hi = min(0.99, base_threshold + window)
    thresholds = np.linspace(lo, hi, steps)
    best_thr = base_threshold
    best_gap = float("inf")
    for thr in thresholds:
        _, instances = apply_postprocessing(prob_map, cfg, threshold=float(thr))
        count = int(len(np.unique(instances)) - 1)
        gap = abs(count - target_count)
        if gap < best_gap or (gap == best_gap and abs(thr - base_threshold) < abs(best_thr - base_threshold)):
            best_gap = gap
            best_thr = float(thr)
        if gap <= tolerance:
            break
    return best_thr


__all__ = [
    "apply_postprocessing",
    "search_optimal_threshold",
    "count_guided_threshold",
    "PostProcessConfig",
]
