import random
from pathlib import Path

import numpy as np

from src.pipelines.crop_pipeline import (
    QAThresholds,
    CropPipeline,
    CropPipelineConfig,
    _compute_quality_metrics,
    _qa_fail_reasons,
)
from src.utils.color_norm import gray_world


def test_gray_world_strength_zero():
    image = np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
    corrected = gray_world(image, strength=0.0)
    assert np.array_equal(corrected, image)


def test_quality_metrics_basic():
    image = np.full((10, 10, 3), 128, dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    metrics = _compute_quality_metrics(
        crop_rgb=image,
        crop_mask=mask,
        original_area=100,
        qa_cfg=QAThresholds(),
    )
    assert metrics["area_px"] == 16
    assert metrics["aspect_ratio"] == 1.0
    reasons = _qa_fail_reasons(
        metrics,
        QAThresholds(min_area_px=10, min_focus_measure=0.0),
    )
    assert reasons == []


def test_group_stratified_assign_balanced(tmp_path):
    cfg = CropPipelineConfig(
        output_root=tmp_path / "crops",
        folds=3,
        random_seed=123,
        include_mask_channel=False,
        save_overlays=False,
    )
    pipeline = CropPipeline(cfg)
    groups = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    strat_keys = [
        "never|WB|frisch",
        "never|WB|frisch",
        "red|MB|frisch",
        "red|MB|frisch",
        "red|WB|aufgetaut",
        "red|WB|aufgetaut",
        "green|UEB|frisch",
        "green|UEB|frisch",
        "yellow|WB|aufgetaut",
        "yellow|WB|aufgetaut",
        "never|MB|gefroren",
        "never|MB|gefroren",
    ]
    assignments = pipeline._group_stratified_assign(
        groups=groups,
        strat_keys=strat_keys,
        n_splits=3,
        rng=random.Random(123),
    )
    # Ensure every fold receives at least one sample
    assert sorted(set(assignments)) == [0, 1, 2]
    # Each group must map to a single fold
    per_group = {}
    for g, fold in zip(groups, assignments):
        per_group.setdefault(g, set()).add(fold)
    assert all(len(folds) == 1 for folds in per_group.values())
