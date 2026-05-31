from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Tuple
import numpy as np

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def _validate_polygon(points: Sequence[Point]) -> None:
    if points is None:
        raise ValueError("Polygon is None.")
    if len(points) < 3:
        raise ValueError(
            f"Polygon must have at least 3 points; got {len(points)}."
        )
    
    # Check basic point structure and collect coordinates for further validation
    xs = []
    ys = []
    for i, p in enumerate(points):
        if (
            not isinstance(p, (tuple, list))
            or len(p) != 2
            or not isinstance(p[0], (int, float))
            or not isinstance(p[1], (int, float))
        ):
            raise ValueError(
                f"Point at index {i} is not a valid 2D coordinate pair: {p!r}"
            )
        
        # Check for NaN or infinite values
        if not (np.isfinite(p[0]) and np.isfinite(p[1])):
            raise ValueError(
                f"Point at index {i} contains NaN or infinite values: {p!r}"
            )
        
        xs.append(p[0])
        ys.append(p[1])
    
    # Check for degenerate polygons (all points colinear)
    # Using cross product to check if all points are colinear
    if len(points) >= 3:
        # Calculate area using shoelace formula
        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += xs[i] * ys[j] - xs[j] * ys[i]
        area = abs(area) / 2.0
        
        # If area is effectively zero, polygon is degenerate
        if area < 1e-10:
            logger.warning("Polygon is degenerate (all points are colinear or area ~0)")
    
    # Check for duplicate consecutive points
    for i in range(len(points)):
        j = (i + 1) % len(points)
        if abs(points[i][0] - points[j][0]) < 1e-10 and abs(points[i][1] - points[j][1]) < 1e-10:
            logger.warning(
                "Polygon has duplicate consecutive points at indices %d and %d: %s ~= %s",
                i, j, points[i], points[j]
            )


def polygon_area(points: List[Point]) -> float:
    """
    Compute the (absolute) area of a simple polygon via the shoelace formula.

    Parameters
    ----------
    points:
        Sequence of (x, y) vertices. At least 3 points required. The polygon
        does not need to repeat the first vertex at the end.

    Returns
    -------
    float
        Non-negative area.

    Raises
    ------
    ValueError
        For invalid input (less than 3 points or malformed coordinates).
    """
    _validate_polygon(points)
    area2 = 0.0  # Twice the signed area
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    area = abs(area2) * 0.5
    logger.debug("Polygon area computed: %f", area)
    return area


def polygon_bbox(points: List[Point]) -> Tuple[float, float, float, float]:
    """
    Compute the bounding box (x, y, w, h) of a polygon.

    Parameters
    ----------
    points:
        Sequence of (x, y) vertices. At least 3 points required.

    Returns
    -------
    (x, y, w, h): tuple of float
        The minimal axis-aligned bounding box covering all points.

    Raises
    ------
    ValueError
        For invalid input.
    """
    _validate_polygon(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)
    bbox = (float(x_min), float(y_min), float(w), float(h))
    logger.debug("Polygon bbox computed: %s", bbox)
    return bbox
