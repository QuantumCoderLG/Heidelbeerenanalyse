from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class MetricsConfig:
    count_target: int = 25
    merge_iou_threshold: float = 0.1
    miss_iou_threshold: float = 0.2


def _compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    union = pred_sum + gt_sum - intersection
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    return {
        "pixel_iou": float(iou),
        "pixel_dice": float(dice),
        "pred_pixels": float(pred_sum),
        "gt_pixels": float(gt_sum),
    }


def _label_ids(mask: np.ndarray) -> List[int]:
    ids = np.unique(mask)
    ids = ids[ids > 0]
    return ids.tolist()


def _compute_pairwise(pred_instances: np.ndarray, gt_instances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred_ids = _label_ids(pred_instances)
    gt_ids = _label_ids(gt_instances)
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    dice_matrix = np.zeros_like(iou_matrix)
    for gi, gt_id in enumerate(gt_ids):
        gt_mask = gt_instances == gt_id
        gt_area = float(gt_mask.sum())
        for pi, pred_id in enumerate(pred_ids):
            pred_mask = pred_instances == pred_id
            inter = float(np.logical_and(gt_mask, pred_mask).sum())
            if inter == 0:
                continue
            pred_area = float(pred_mask.sum())
            union = gt_area + pred_area - inter
            iou_matrix[gi, pi] = inter / union if union > 0 else 0.0
            denom = gt_area + pred_area
            dice_matrix[gi, pi] = (2.0 * inter / denom) if denom > 0 else 0.0
    return iou_matrix, dice_matrix


def _greedy_match(iou_matrix: np.ndarray, dice_matrix: np.ndarray, gt_ids: Sequence[int], pred_ids: Sequence[int]) -> List[Tuple[int, int, float, float]]:
    if iou_matrix.size == 0:
        return []
    iou_copy = iou_matrix.copy()
    matches: List[Tuple[int, int, float, float]] = []
    while True:
        idx = np.unravel_index(np.argmax(iou_copy), iou_copy.shape)
        best_iou = iou_copy[idx]
        if best_iou <= 0:
            break
        gi, pi = idx
        matches.append((gt_ids[gi], pred_ids[pi], float(best_iou), float(dice_matrix[gi, pi])))
        iou_copy[gi, :] = -1.0
        iou_copy[:, pi] = -1.0
    return matches


def evaluate_instances(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    metrics_cfg: MetricsConfig,
) -> Dict[str, float]:
    pred_ids = _label_ids(pred_instances)
    gt_ids = _label_ids(gt_instances)
    pair_iou, pair_dice = _compute_pairwise(pred_instances, gt_instances)
    matches = _greedy_match(pair_iou, pair_dice, gt_ids, pred_ids)
    matched_ious = [m[2] for m in matches]
    matched_dice = [m[3] for m in matches]
    metrics = {
        "instance_count_gt": float(len(gt_ids)),
        "instance_count_pred": float(len(pred_ids)),
        "instance_iou_median": float(np.median(matched_ious)) if matched_ious else 0.0,
        "instance_iou_p25": float(np.percentile(matched_ious, 25)) if matched_ious else 0.0,
        "instance_iou_p75": float(np.percentile(matched_ious, 75)) if matched_ious else 0.0,
        "instance_dice_mean": float(np.mean(matched_dice)) if matched_dice else 0.0,
        "matched_instances": float(len(matches)),
    }
    miss_threshold = metrics_cfg.miss_iou_threshold
    missed = []
    if gt_ids:
        for gi, gid in enumerate(gt_ids):
            if pair_iou.shape[1] == 0 or float(np.max(pair_iou[gi])) <= miss_threshold:
                missed.append(gid)
    merge_threshold = metrics_cfg.merge_iou_threshold
    merge_counter = 0
    for pi, pred_id in enumerate(pred_ids):
        overlapping = 0
        for gi, gt_id in enumerate(gt_ids):
            if pair_iou[gi, pi] > merge_threshold:
                overlapping += 1
        if overlapping >= 2:
            merge_counter += 1
    count_target = metrics_cfg.count_target
    metrics.update(
        {
            "count_accuracy": 1.0 if len(pred_ids) == count_target else 0.0,
            "miss_rate": float(len(missed)) / float(len(gt_ids)) if gt_ids else 0.0,
            "merge_rate": float(merge_counter) / float(len(gt_ids)) if gt_ids else 0.0,
        }
    )
    return metrics


def evaluate_image(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    metrics_cfg: MetricsConfig,
) -> Dict[str, float]:
    binary_metrics = _compute_binary_metrics(pred_instances > 0, gt_instances > 0)
    instance_metrics = evaluate_instances(pred_instances, gt_instances, metrics_cfg)
    return {**binary_metrics, **instance_metrics}


def aggregate_metrics(per_image: Iterable[Dict[str, float]]) -> Dict[str, float]:
    per_image = list(per_image)
    if not per_image:
        return {}
    keys = per_image[0].keys()
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = [d[key] for d in per_image if key in d]
        if not values:
            continue
        aggregated[key] = float(np.mean(values))
    return aggregated


__all__ = ["MetricsConfig", "evaluate_image", "aggregate_metrics", "evaluate_instances"]
