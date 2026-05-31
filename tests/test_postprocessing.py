import numpy as np

from src.evaluation.postprocessing import apply_postprocessing, count_guided_threshold


def _base_config():
    return {
        "threshold": 0.5,
        "morphology": {"open_kernel": 3, "close_kernel": 3, "iterations": 1},
        "min_area": 10,
        "circularity": {"enabled": True, "min": 0.1},
        "watershed": {"enabled": True, "peak_rel_threshold": 0.3, "peak_min_distance": 1},
        "count_guided": {"enabled": True, "target_count": 2, "tolerance": 0, "window": 0.3, "steps": 5},
    }


def test_apply_postprocessing_separates_regions():
    cfg = _base_config()
    cfg["watershed"]["enabled"] = False
    prob = np.zeros((64, 64), dtype=np.float32)
    cv = np.indices((64, 64))
    rr, cc = cv[0], cv[1]
    mask1 = (rr - 20) ** 2 + (cc - 20) ** 2 <= 6 ** 2
    mask2 = (rr - 44) ** 2 + (cc - 44) ** 2 <= 6 ** 2
    prob[mask1] = 0.95
    prob[mask2] = 0.9
    binary, instances = apply_postprocessing(prob, cfg)
    unique_ids = np.unique(instances)
    assert binary.sum() > 0
    assert set(unique_ids.tolist()) == {0, 1, 2}


def test_count_guided_threshold_moves_towards_target():
    cfg = _base_config()
    prob = np.full((32, 32), 0.2, dtype=np.float32)
    prob[8:16, 8:16] = 0.7
    prob[18:26, 18:26] = 0.65
    chosen = count_guided_threshold(prob, 0.5, 2, cfg)
    assert 0.2 <= chosen <= 0.8
