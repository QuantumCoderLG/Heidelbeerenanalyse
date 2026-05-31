from __future__ import annotations

from typing import Tuple, Dict, Sequence, Optional

import cv2
import numpy as np
import torch


def letterbox_resize(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize image to target_hw with padding, preserving aspect ratio (linear interp)."""
    th, tw = target_hw
    h, w = img.shape[:2]
    scale = min(th / max(1, h), tw / max(1, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h = th - nh
    pad_w = tw - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return out


def letterbox_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize mask to target_hw with padding, preserving aspect ratio (nearest neighbor)."""
    th, tw = target_hw
    h, w = mask.shape[:2]
    scale = min(th / max(1, h), tw / max(1, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    pad_h = th - nh
    pad_w = tw - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return out


def resize_with_padding(
    image: np.ndarray,
    target_hw: Tuple[int, int] | None,
    keep_ratio: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Resize image with padding, returning metadata about the transformation."""
    meta: Dict[str, float] = {
        "scale_x": 1.0,
        "scale_y": 1.0,
        "pad_top": 0,
        "pad_bottom": 0,
        "pad_left": 0,
        "pad_right": 0,
    }
    if target_hw is None:
        meta["resized_height"] = image.shape[0]
        meta["resized_width"] = image.shape[1]
        meta["target_height"] = image.shape[0]
        meta["target_width"] = image.shape[1]
        return image, meta

    target_h, target_w = target_hw
    meta["target_height"] = target_h
    meta["target_width"] = target_w
    orig_h, orig_w = image.shape[:2]

    if keep_ratio:
        max_side = max(target_h, target_w)
        scale = max_side / float(max(orig_h, orig_w))
        new_h = max(1, int(round(orig_h * scale)))
        new_w = max(1, int(round(orig_w * scale)))
        meta["scale_x"] = scale
        meta["scale_y"] = scale
    else:
        new_h = target_h
        new_w = target_w
        meta["scale_x"] = new_w / float(orig_w)
        meta["scale_y"] = new_h / float(orig_h)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = max(target_h - new_h, 0)
    pad_w = max(target_w - new_w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0.0,
    )
    meta["pad_top"] = float(pad_top)
    meta["pad_bottom"] = float(pad_bottom)
    meta["pad_left"] = float(pad_left)
    meta["pad_right"] = float(pad_right)
    meta["resized_height"] = float(new_h)
    meta["resized_width"] = float(new_w)
    return padded, meta


def preprocess_image(
    image: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
    target_hw: Tuple[int, int] | None,
    keep_ratio: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Preprocess image: normalize, resize/pad, and convert to tensor."""
    image_float = image.astype(np.float32) / 255.0
    resized, meta = resize_with_padding(image_float, target_hw, keep_ratio)
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    norm = (resized - mean_arr) / std_arr
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).contiguous()
    meta["original_height"] = float(image.shape[0])
    meta["original_width"] = float(image.shape[1])
    return tensor, meta


def postprocess_probability(
    prob: np.ndarray,
    meta: Dict[str, float],
    *,
    resize_to_original: bool = True,
) -> np.ndarray:
    """Crop padding and resize probability map back to original size."""
    pad_top = int(meta.get("pad_top", 0))
    pad_bottom = int(meta.get("pad_bottom", 0))
    pad_left = int(meta.get("pad_left", 0))
    pad_right = int(meta.get("pad_right", 0))
    h, w = prob.shape
    y0 = pad_top
    y1 = h - pad_bottom if pad_bottom > 0 else h
    x0 = pad_left
    x1 = w - pad_right if pad_right > 0 else w
    cropped = prob[y0:y1, x0:x1]
    meta["cropped_height"] = float(cropped.shape[0])
    meta["cropped_width"] = float(cropped.shape[1])
    if not resize_to_original:
        return cropped
    orig_h = int(round(meta.get("original_height", cropped.shape[0])))
    orig_w = int(round(meta.get("original_width", cropped.shape[1])))
    if cropped.shape[0] == orig_h and cropped.shape[1] == orig_w:
        return cropped
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
