# src/data/coco_schema.py
from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Sequence

logger = logging.getLogger(__name__)


def build_coco_image(image_id: int, file_name: str, width: int, height: int) -> Dict:
    """
    Build a minimal COCO 'image' dictionary.

    Parameters
    ----------
    image_id:
        Unique image id (stable).
    file_name:
        Image file name (relative path if needed).
    width:
    height:

    Returns
    -------
    dict
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: width={width}, height={height}")
    if not file_name:
        raise ValueError("file_name must be a non-empty string.")
    d = {
        "id": int(image_id),
        "file_name": str(file_name),
        "width": int(width),
        "height": int(height),
    }
    logger.debug("Built COCO image: %s", d)
    return d


def build_coco_annotation(
    ann_id: int,
    image_id: int,
    polygon: List[float],
    bbox: Tuple[float, float, float, float],
    area: float,
    category_id: int = 1,
) -> Dict:
    """
    Build a COCO 'annotation' dictionary for a single polygon instance.

    Parameters
    ----------
    ann_id:
        Unique annotation id (stable).
    image_id:
        Parent image id.
    polygon:
        Flattened [x1, y1, x2, y2, ...] list of floats (must contain >= 6 elements).
    bbox:
        (x, y, w, h).
    area:
        Polygon area in pixel^2.
    category_id:
        COCO category id. Defaults to 1.

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        If the polygon doesn't contain an even number of coordinates or has < 3 vertices.
    """
    if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
        raise ValueError(
            "polygon must be a flat list of floats with even length >= 6."
        )
    if area < 0:
        raise ValueError("area must be non-negative.")
    x, y, w, h = bbox
    if w < 0 or h < 0:
        raise ValueError("bbox width/height must be non-negative.")
    d = {
        "id": int(ann_id),
        "image_id": int(image_id),
        "category_id": int(category_id),
        "segmentation": [polygon],  # COCO expects list of polygons
        "area": float(area),
        "bbox": [float(x), float(y), float(w), float(h)],
        "iscrowd": 0,
    }
    logger.debug("Built COCO annotation: %s", d)
    return d


def build_coco_categories(names: Sequence[str]) -> List[Dict]:
    """
    Build a COCO categories list from class names.

    Parameters
    ----------
    names : sequence of str
        Category names. Duplicates are removed by the caller.

    Returns
    -------
    list of dict
        [{"id": 1, "name": names[0], "supercategory": "object"}, ...]
    """
    cats: List[Dict] = []
    for i, name in enumerate(names, start=1):
        cats.append({"id": int(i), "name": str(name), "supercategory": "object"})
    logger.debug("COCO dynamic categories: %s", cats)
    return cats


def get_coco_categories() -> List[Dict]:
    """
    Return the list of COCO categories. We provide a single category: blueberry.

    Returns
    -------
    list of dict
    """
    cats = [
        {
            "id": 1,
            "name": "blueberry",
            "supercategory": "fruit",
        }
    ]
    logger.debug("COCO categories: %s", cats)
    return cats
