import numpy as np

from src.evaluation.metrics import MetricsConfig, evaluate_image


def test_metrics_perfect_prediction():
    gt = np.zeros((32, 32), dtype=np.int32)
    pred = np.zeros_like(gt)
    gt[4:10, 4:10] = 1
    gt[20:26, 20:26] = 2
    pred[4:10, 4:10] = 5
    pred[20:26, 20:26] = 7
    cfg = MetricsConfig(count_target=2, merge_iou_threshold=0.1, miss_iou_threshold=0.2)
    metrics = evaluate_image(pred, gt, cfg)
    assert metrics["instance_iou_median"] == 1.0
    assert metrics["miss_rate"] == 0.0
    assert metrics["merge_rate"] == 0.0


def test_metrics_detects_merge():
    gt = np.zeros((16, 16), dtype=np.int32)
    gt[2:6, 2:6] = 1
    gt[2:6, 10:14] = 2
    pred = np.zeros_like(gt)
    pred[2:6, 2:14] = 3
    cfg = MetricsConfig(count_target=2, merge_iou_threshold=0.1, miss_iou_threshold=0.2)
    metrics = evaluate_image(pred, gt, cfg)
    assert metrics["merge_rate"] > 0
    assert metrics["count_accuracy"] == 0.0
