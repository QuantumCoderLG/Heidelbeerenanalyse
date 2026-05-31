# src/data/rasterize.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import colorsys

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 is optional
    cv2 = None  # type: ignore

from ..utils import io_utils

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def _poly_to_int(points: Sequence[Point]) -> np.ndarray:
    """
    Convert polygon points to integer numpy array of shape (N, 2), rounded to nearest pixel.
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Polygon must be (N,2).")
    return np.rint(arr).astype(np.int64)


def _roi_from_poly(int_poly: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Compute a clipped integer ROI (x0, y0, x1, y1) for an integer polygon.
    x1, y1 are exclusive bounds; guarantees 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height.
    """
    x_min = int(np.clip(int_poly[:, 0].min(), 0, max(0, width - 1)))
    y_min = int(np.clip(int_poly[:, 1].min(), 0, max(0, height - 1)))
    x_max = int(np.clip(int_poly[:, 0].max(), 0, max(0, width - 1)))
    y_max = int(np.clip(int_poly[:, 1].max(), 0, max(0, height - 1)))
    # exclusive bounds (+1); handle degenerate cases by ensuring at least 1px extent if inside image
    x1 = min(width, x_max + 1)
    y1 = min(height, y_max + 1)
    x0 = max(0, x_min)
    y0 = max(0, y_min)
    return x0, y0, x1, y1


def rasterize_instance_mask(
    width: int,
    height: int,
    polygons: List[List[Point]],
    annotation_ids: List[int],
) -> "np.ndarray":
    """
    Rasterize instance polygons into a 2D integer mask image.

    Memory-safe approach:
    ---------------------
    - Use uint32 for masks (supports up to 4B unique IDs, saves 50% memory vs int64)
    - For each polygon, compute its *local* ROI (bounding box) and rasterize into a
      small temporary uint8 buffer of ROI size only — not a full-frame temporary.
    - Write the annotation id into the corresponding ROI region (last polygon wins).

    Parameters
    ----------
    width, height:
        Output mask width and height in pixels.
    polygons:
        List of polygons (one per instance). Each polygon is a list of (x, y) floats.
    annotation_ids:
        List of instance IDs (must be same length as polygons).

    Returns
    -------
    np.ndarray
        2D array of dtype=np.uint32 with shape (height, width).

    Raises
    ------
    ValueError
        On size mismatches or invalid inputs.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid raster size: width={width}, height={height}")
    if len(polygons) != len(annotation_ids):
        raise ValueError(
            f"polygons and annotation_ids must have same length: "
            f"{len(polygons)} vs {len(annotation_ids)}"
        )

    # Check memory requirements and warn for large images
    pixels = width * height
    memory_mb = (pixels * 4) / (1024 * 1024)  # uint32 = 4 bytes
    if memory_mb > 256:
        logger.warning(
            "Large image mask will use %.1f MB of memory (width=%d, height=%d)",
            memory_mb, width, height
        )

    # Use uint32 to save memory while supporting up to 4 billion unique IDs
    # This is sufficient for SAM 2.1 training and reduces memory by 50% vs int64
    mask = np.zeros((height, width), dtype=np.uint32)

    for idx, (poly, ann_id) in enumerate(zip(polygons, annotation_ids)):
        if len(poly) < 3:
            raise ValueError(f"Polygon at index {idx} has fewer than 3 points.")

        int_poly = _poly_to_int(poly)

        # Compute ROI and shift polygon into ROI coordinates for rasterization.
        x0, y0, x1, y1 = _roi_from_poly(int_poly, width, height)
        if x1 <= x0 or y1 <= y0:
            # Polygon is completely outside image bounds → treat as error
            raise ValueError(
                f"Polygon at index {idx} is out of image bounds after clipping: ROI=({x0},{y0},{x1},{y1})."
            )

        roi_w = x1 - x0
        roi_h = y1 - y0
        roi_poly = int_poly.copy()
        roi_poly[:, 0] -= x0
        roi_poly[:, 1] -= y0

        # Rasterize into small temporary buffer
        temp = np.zeros((roi_h, roi_w), dtype=np.uint8)

        if (cv2 is not None) and hasattr(cv2, "fillPoly"):
            try:
                # OpenCV expects int32 points
                cv2_poly = roi_poly.astype(np.int32)
                cv2.fillPoly(temp, [cv2_poly], 1)
            except Exception as cv2_err:
                # Fallback to PIL if cv2 fails for any reason
                logger.debug("OpenCV fillPoly failed, using PIL fallback: %s", cv2_err)
                temp_img = Image.new("1", (roi_w, roi_h), 0)
                draw = ImageDraw.Draw(temp_img)
                draw.polygon([tuple(p) for p in roi_poly.tolist()], fill=1)
                temp = np.asarray(temp_img, dtype=np.uint8)
        else:
            # PIL fallback — sufficiently fast on ROIs
            temp_img = Image.new("1", (roi_w, roi_h), 0)
            draw = ImageDraw.Draw(temp_img)
            draw.polygon([tuple(p) for p in roi_poly.tolist()], fill=1)
            temp = np.asarray(temp_img, dtype=np.uint8)

        # Validate annotation ID fits in uint32
        if ann_id < 0 or ann_id > 0xFFFFFFFF:
            raise ValueError(
                f"Annotation ID {ann_id} at index {idx} exceeds uint32 range [0, 4294967295]"
            )
        
        # Write into mask (last-wins semantics)
        roi_view = mask[y0:y1, x0:x1]
        roi_view[temp == 1] = np.uint32(ann_id)

    return mask


def save_instance_mask_png(
    mask: "np.ndarray",
    out_path: "Path",
    *,
    compress_level: int = 1,
) -> None:
    """
    Save an instance mask as an 8-bit PNG while preserving full annotation IDs.

    Strategy
    --------
    * Remap unique IDs to 0..255 (0 reserved for background). This satisfies 8-bit pixel storage.
    * Embed a JSON mapping in PNG text metadata: {"index_to_id": {1: 12345, ...}}
      so the original IDs can be recovered exactly.

    Notes
    -----
    * Supports at most 255 distinct non-zero instance IDs per image. Raises if exceeded.
    * Input 'mask' should be 2D integer array. Values may be larger than 255.

    Parameters
    ----------
    mask:
        2D integer array (height, width).
    out_path:
        Destination file path ('.png').
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("mask must have an integer dtype.")

    # Unique IDs (sorted for reproducibility)
    unique_ids = np.unique(mask)
    # Exclude background (0) for mapping
    nonzero_ids = [int(x) for x in unique_ids if int(x) != 0]
    if len(nonzero_ids) > 255:
        raise ValueError(
            f"Exceeded 255 instance IDs for 8-bit PNG: got {len(nonzero_ids)}."
        )

    # Build forward mapping: original id -> 8-bit index (1..N), keep 0 for background
    orig_to_idx = {oid: i + 1 for i, oid in enumerate(nonzero_ids)}
    idx_to_orig = {v: k for k, v in orig_to_idx.items()}

    # Efficient remap without gigantic LUTs (works for sparse, large IDs).
    out8 = np.zeros(mask.shape, dtype=np.uint8)
    temp_bool = np.zeros(mask.shape, dtype=bool)
    for oid, idx in orig_to_idx.items():
        # Reuse a single boolean buffer to avoid repeated allocations.
        np.equal(mask, oid, out=temp_bool)
        out8[temp_bool] = np.uint8(idx)

    img = Image.fromarray(out8, mode="L")

    # Embed mapping metadata
    meta = PngImagePlugin.PngInfo()
    meta.add_text("id_map", json.dumps({"index_to_id": idx_to_orig}, ensure_ascii=False))

    io_utils.ensure_parent_dir(out_path)
    io_utils.atomic_save_pil_image(
        img,
        out_path,
        pnginfo=meta,
        optimize=False,
        compress_level=compress_level,
    )


def _color_for_id(ann_id: int) -> Tuple[int, int, int, int]:
    """
    Deterministic, vivid RGBA color from an integer id.

    Uses HSV with high saturation and value to avoid gray tones; semi-opaque alpha.
    """
    # Mix bits to create a stable hue seed in [0,1)
    x = (ann_id * 2654435761) & 0xFFFFFFFF
    hue = ((x / 4294967296.0) + 0.19) % 1.0  # add offset to avoid clustering at red
    sat = 0.95
    val = 1.00
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    # Slightly higher alpha for crisper overlays
    return int(r * 255), int(g * 255), int(b * 255), 160


def _centroid(points: Sequence[Point]) -> Tuple[float, float]:
    """
    Compute polygon centroid using the shoelace formula. Falls back to mean of points
    if area is near zero.
    """
    n = len(points)
    if n == 0:
        return 0.0, 0.0
    a = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-8:
        # Degenerate polygon → mean of vertices
        xs, ys = zip(*points)
        return float(sum(xs) / n), float(sum(ys) / n)
    a *= 0.5
    cx /= (6.0 * a)
    cy /= (6.0 * a)
    return float(cx), float(cy)


def draw_overlay(
    image: "np.ndarray | Image.Image",
    polygons: List[List[Point]],
    annotation_ids: List[int],
) -> Image.Image:
    """
    Draw semi-transparent overlays and ID labels onto an image.

    Parameters
    ----------
    image:
        Input image as RGB numpy array (H,W,3) or PIL Image.
    polygons:
        List of polygons (one per instance).
    annotation_ids:
        IDs corresponding to each polygon.

    Returns
    -------
    PIL.Image.Image
        Image with overlays drawn.
    """
    # Normalize base image to RGBA PIL image
    if isinstance(image, Image.Image):
        base = image.convert("RGBA")
    else:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("image must be HxWx3/4 numpy array or a PIL.Image.")
        mode = "RGBA" if image.shape[2] == 4 else "RGB"
        base = Image.fromarray(image.astype(np.uint8), mode=mode).convert("RGBA")

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None  # type: ignore

    for poly, ann_id in zip(polygons, annotation_ids):
        if len(poly) < 3:
            raise ValueError("Encountered polygon with < 3 points while drawing overlay.")
        color = _color_for_id(int(ann_id))
        # Fill
        draw.polygon(poly, fill=color)
        # Outline (more opaque)
        outline = (color[0], color[1], color[2], 220)
        draw.line(poly + [poly[0]], fill=outline, width=2)

        # Label
        cx, cy = _centroid(poly)
        label_text = str(int(ann_id))
        # Compute text size without using deprecated/unstubbed draw.textsize
        if font:
            try:
                bbox = draw.textbbox((0, 0), label_text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = (8 * len(label_text), 10)
        else:
            tw, th = (8 * len(label_text), 10)
        # Slightly offset text to center
        draw.rectangle(
            [(cx - tw / 2 - 2, cy - th / 2 - 1), (cx + tw / 2 + 2, cy + th / 2 + 1)],
            fill=(0, 0, 0, 120),
        )
        draw.text((cx - tw / 2, cy - th / 2), label_text, fill=(255, 255, 255, 220), font=font)

    # Composite overlay onto base
    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")
