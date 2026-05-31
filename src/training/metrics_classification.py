from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    auc: float | None
    pr_auc: float | None
    ece: float
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom else 0.0


def binary_classification_metrics(
    *,
    logits: Sequence[float] | None = None,
    probs: Sequence[float] | None = None,
    targets: Sequence[int],
    threshold: float = 0.5,
    num_bins: int = 15,
) -> BinaryMetrics:
    if probs is None:
        if logits is None:
            raise ValueError("Either probs or logits must be provided.")
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    else:
        probs = np.asarray(probs, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.int32)
    preds = (probs >= threshold).astype(np.int32)
    tp = int(((preds == 1) & (targets_arr == 1)).sum())
    fp = int(((preds == 1) & (targets_arr == 0)).sum())
    tn = int(((preds == 0) & (targets_arr == 0)).sum())
    fn = int(((preds == 0) & (targets_arr == 1)).sum())
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # AUC via simple trapezoidal rule over sorted scores
    try:
        auc = roc_auc_score(targets_arr, probs)
    except Exception:
        auc = None
    try:
        pr_auc = average_precision_score(targets_arr, probs)
    except Exception:
        pr_auc = None

    ece = expected_calibration_error(probs, targets_arr, num_bins=num_bins)
    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        auc=auc,
        pr_auc=pr_auc,
        ece=ece,
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def expected_calibration_error(probs: Sequence[float], targets: Sequence[int], num_bins: int = 15) -> float:
    probs_arr = np.asarray(probs, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.int32)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(probs_arr, bins) - 1
    ece = 0.0
    total = len(probs_arr)
    for b in range(num_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_conf = float(probs_arr[mask].mean())
        bin_acc = float(targets_arr[mask].mean())
        ece += abs(bin_conf - bin_acc) * count / total
    return float(ece)


def expected_cost(
    *,
    probs: Sequence[float],
    targets: Sequence[int],
    threshold: float,
    cost: Dict[str, float],
) -> float:
    probs_arr = np.asarray(probs, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.int32)
    preds = (probs_arr >= threshold).astype(np.int32)
    tp = ((preds == 1) & (targets_arr == 1)).sum()
    fp = ((preds == 1) & (targets_arr == 0)).sum()
    tn = ((preds == 0) & (targets_arr == 0)).sum()
    fn = ((preds == 0) & (targets_arr == 1)).sum()
    total = float(len(targets_arr))
    cost_tp = cost.get("tp", 0.0)
    cost_fp = cost.get("fp", 0.0)
    cost_tn = cost.get("tn", 0.0)
    cost_fn = cost.get("fn", 0.0)
    total_cost = tp * cost_tp + fp * cost_fp + tn * cost_tn + fn * cost_fn
    return float(total_cost / total) if total else 0.0


def find_threshold(
    probs: Sequence[float],
    targets: Sequence[int],
    *,
    grid_size: int,
    min_recall_positive: float,
    cost: Dict[str, float] | None = None,
    min_threshold: float = 1e-3,
    max_threshold: float = 1.0 - 1e-3,
    fallback_threshold: float | None = None,
    abstain_margin: float = 0.0,
    grid_mode: str = "linear",
) -> Dict[str, float]:
    probs_arr = np.asarray(probs, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.int32)
    min_threshold = float(np.clip(min_threshold, 1e-6, 0.999))
    max_threshold = float(np.clip(max_threshold, min_threshold + 1e-6, 1.0 - 1e-6))
    # Threshold grid generation: linear or quantile-based
    # Use quantile grid when grid_size < number of unique probabilities to better
    # focus resolution where data lies (helps small validation splits).
    mode = str(grid_mode or "linear").lower()
    if mode == "quantile":
        qs = np.linspace(0.0, 1.0, max(2, grid_size))
        thresholds = np.quantile(probs_arr, qs)
        thresholds = np.clip(thresholds, min_threshold, max_threshold)
        thresholds = np.unique(thresholds)
        if thresholds.size < 2:
            thresholds = np.linspace(min_threshold, max_threshold, max(2, grid_size))
    else:
        thresholds = np.linspace(min_threshold, max_threshold, max(2, grid_size))
    metrics: Dict[str, float] = {}
    best_threshold = thresholds[0]
    best_cost = float("inf")
    best_recall_gap = float("inf")
    positive_total = max(1, int((targets_arr == 1).sum()))
    best_fp = float("inf")
    for th in thresholds:
        preds = (probs_arr >= th).astype(np.int32)
        tp = ((preds == 1) & (targets_arr == 1)).sum()
        fp = ((preds == 1) & (targets_arr == 0)).sum()
        fn = ((preds == 0) & (targets_arr == 1)).sum()
        recall = float(tp / positive_total)
        if recall < min_recall_positive:
            gap = min_recall_positive - recall
            if gap < best_recall_gap:
                best_recall_gap = gap
                best_threshold = float(th)
            continue
        # Among feasible thresholds prioritize minimal cost if provided,
        # otherwise minimize false positives; tie-break by higher threshold (more conservative).
        if cost is not None:
            current_cost = expected_cost(probs=probs_arr, targets=targets_arr, threshold=float(th), cost=cost)
            better = (current_cost < best_cost - 1e-12) or (
                abs(current_cost - best_cost) <= 1e-12 and float(th) > float(best_threshold)
            )
            if better:
                best_cost = current_cost
                best_threshold = float(th)
        else:
            if float(fp) < best_fp - 1e-12 or (abs(float(fp - best_fp)) <= 1e-12 and float(th) > float(best_threshold)):
                best_fp = float(fp)
                best_threshold = float(th)
    recall_satisfied = bool(best_recall_gap == float("inf"))
    raw_threshold = best_threshold
    fallback_applied = False
    if not recall_satisfied:
        fb = fallback_threshold if fallback_threshold is not None else min_threshold
        best_threshold = float(np.clip(fb, min_threshold, max_threshold))
        fallback_applied = True
    metrics["threshold"] = best_threshold
    metrics["raw_threshold"] = raw_threshold
    metrics["min_recall_satisfied"] = recall_satisfied
    metrics["fallback_applied"] = fallback_applied
    metrics["best_cost"] = best_cost if best_cost != float("inf") else float("nan")
    metrics["abstain_margin"] = float(abstain_margin if fallback_applied else 0.0)
    return metrics


def average_precision_score(targets: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(targets, dtype=np.int32)
    scores_arr = np.asarray(scores, dtype=np.float64)
    pos_total = int((y == 1).sum())
    if pos_total == 0 or y.size == 0:
        return float("nan")
    order = np.argsort(-scores_arr, kind="mergesort")
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    precision = np.divide(tp_cum, tp_cum + fp_cum, out=np.zeros_like(tp_cum, dtype=np.float64), where=(tp_cum + fp_cum) > 0)
    recall = tp_cum / pos_total
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return float(ap)


def roc_auc_score(targets: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(targets, dtype=np.int32)
    scores_arr = np.asarray(scores, dtype=np.float64)
    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pos_scores = scores_arr[pos_mask]
    neg_scores = scores_arr[neg_mask]
    # Mann-Whitney U statistic
    combined = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=np.float64)
    sorted_values = combined[order]
    unique_vals, idx_start, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    for start, count in zip(idx_start, counts):
        if count <= 1:
            continue
        end = start + count
        avg_rank = (start + 1 + end) / 2.0
        same_slice = order[start:end]
        ranks[same_slice] = avg_rank
    pos_ranks = ranks[:n_pos]
    sum_pos = pos_ranks.sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


__all__ = [
    "BinaryMetrics",
    "binary_classification_metrics",
    "expected_calibration_error",
    "expected_cost",
    "find_threshold",
    "roc_auc_score",
    "average_precision_score",
]
