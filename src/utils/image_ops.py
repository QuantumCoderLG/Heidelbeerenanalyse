from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["CropBounds", "compute_with_margin", "apply_background"]


@dataclass
class CropBounds:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0


def compute_with_margin(
    mask: np.ndarray,
    margin: float = 0.15,
    image_shape: Tuple[int, int] | None = None,
) -> CropBounds:
    """Compute tight bounding box for a binary mask with optional margin."""
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D.")
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Mask is empty; cannot compute bounds.")
    y_min = ys.min()
    y_max = ys.max()
    x_min = xs.min()
    x_max = xs.max()
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    pad_y = int(round(height * margin))
    pad_x = int(round(width * margin))
    y0 = y_min - pad_y
    y1 = y_max + pad_y + 1
    x0 = x_min - pad_x
    x1 = x_max + pad_x + 1
    if image_shape is not None:
        h, w = image_shape
        y0 = max(0, y0)
        x0 = max(0, x0)
        y1 = min(h, y1)
        x1 = min(w, x1)
    return CropBounds(y0=y0, y1=y1, x0=x0, x1=x1)


def apply_background(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Replace background pixels outside ``mask`` by the provided color."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be RGB (H, W, 3)")
    if mask.shape != image.shape[:2]:
        raise ValueError("mask shape must match image height/width")
    mask_bool = mask.astype(bool)
    out = image.copy()
    out[~mask_bool] = color
    return out
