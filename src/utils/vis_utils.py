from __future__ import annotations

from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..data import rasterize
from .box_utils import BBox


def draw_boxes(
    image: np.ndarray,
    boxes: List[BBox],
    labels: List[str],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> Image.Image:
    """Draw bounding boxes and labels on an image."""
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for box, text in zip(boxes, labels):
        cv2.rectangle(img_bgr, (box.x0, box.y0), (box.x1, box.y1), color, thickness)
        cv2.putText(
            img_bgr,
            text,
            (box.x0, max(0, box.y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def instances_to_overlay(
    image: np.ndarray,
    instances: np.ndarray,
    *,
    max_instances: int = 25,
    alpha: float = 0.5
) -> Image.Image:
    """Create a semi-transparent overlay of instance masks on the image."""
    # Note: We assume instances is already filtered or we filter here.
    # The original code filtered here, so we'll keep it optional or rely on caller.
    # For consistency with original apply_model_to_images, we filter here if needed.
    # But better to let caller filter. We'll just iterate unique labels.
    
    output = image.copy()
    unique_labels = [int(x) for x in np.unique(instances) if int(x) != 0]
    if not unique_labels:
        return Image.fromarray(output)

    # If too many, maybe we should have filtered before? 
    # We will just draw all provided in 'instances' (caller responsibility to filter)
    # unless max_instances is strictly enforced by filtering again?
    # The original code called _keep_top_k_instances inside this function.
    # Let's replicate that behavior for safety if the input isn't already filtered.
    # However, to avoid circular dependency or re-importing box_utils logic if not needed,
    # we'll assume the user might want to pass raw instances.
    # Actually, let's import keep_top_k_instances to be safe and consistent.
    from .box_utils import keep_top_k_instances
    filtered = keep_top_k_instances(instances, max_instances)
    
    unique_labels = [int(x) for x in np.unique(filtered) if int(x) != 0]
    
    overlay_info: List[Tuple[List[np.ndarray], Tuple[int, int, int, int]]] = []
    for label in unique_labels:
        mask = filtered == label
        if not np.any(mask):
            continue
        # rasterize._color_for_id returns (r, g, b, a)
        color_rgba = rasterize._color_for_id(label)  # type: ignore[attr-defined]
        color_rgb = np.array(color_rgba[:3], dtype=np.float32)
        # Use passed alpha or color's alpha? Original used color's alpha.
        # We'll use the logic from original:
        mask_alpha = float(color_rgba[3]) / 255.0
        blended = (1.0 - mask_alpha) * output[mask].astype(np.float32) + mask_alpha * color_rgb
        output[mask] = blended.astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_info.append((contours, color_rgba))

    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    for contours, color_rgba in overlay_info:
        contour_color_bgr = (int(color_rgba[2]), int(color_rgba[1]), int(color_rgba[0]))
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            cv2.drawContours(output_bgr, [contour], -1, contour_color_bgr, thickness=2)

    final_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)


def make_contact_sheet(
    crops: List[np.ndarray],
    labels: List[str],
    cols: int = 5,
    cell: Tuple[int, int] = (160, 160)
) -> Image.Image:
    """Create a grid contact sheet from a list of image crops."""
    if not crops:
        return Image.new("RGB", (cell[1], cell[0]), color=(30, 30, 30))
    rows = (len(crops) + cols - 1) // cols
    W = cols * cell[1]
    H = rows * cell[0]
    sheet = Image.new("RGB", (W, H), color=(30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    for idx, (crop, text) in enumerate(zip(crops, labels)):
        r = idx // cols
        c = idx % cols
        x = c * cell[1]
        y = r * cell[0]
        thumb = cv2.resize(crop, (cell[1], cell[0]), interpolation=cv2.INTER_AREA)
        sheet.paste(Image.fromarray(thumb), (x, y))
        draw.rectangle([(x, y), (x + cell[1] - 1, y + 16)], fill=(0, 0, 0, 128))
        draw.text((x + 2, y + 2), text, fill=(255, 255, 255), font=font)
    return sheet
