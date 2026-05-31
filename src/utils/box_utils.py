from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def h(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return int(self.w * self.h)


def expand_bbox(x: int, y: int, w: int, h: int, margin: float, shape: Tuple[int, int]) -> BBox:
    """Expand a bounding box by a relative margin, clamping to image shape (H, W)."""
    H, W = shape
    size = max(w, h)
    pad = int(round(size * float(margin)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return BBox(x0, y0, x1, y1)


def keep_top_k_instances(instances: np.ndarray, top_k: int) -> np.ndarray:
    """Return a mask containing only the largest ``top_k`` instances."""
    if top_k <= 0:
        return np.zeros_like(instances)

    unique_labels = [int(x) for x in np.unique(instances) if int(x) != 0]
    if not unique_labels:
        return np.zeros_like(instances)

    areas = []
    for label in unique_labels:
        area = int(np.count_nonzero(instances == label))
        if area == 0:
            continue
        areas.append((area, label))

    if not areas:
        return np.zeros_like(instances)

    # Sort by area descending
    areas.sort(key=lambda item: item[0], reverse=True)
    selected = areas[:top_k]

    filtered = np.zeros_like(instances, dtype=np.int32)
    for new_label, (_, orig_label) in enumerate(selected, start=1):
        filtered[instances == orig_label] = new_label

    return filtered
