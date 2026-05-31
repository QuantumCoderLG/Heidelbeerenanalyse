# src/data/xml_parser.py
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import List, NamedTuple, Tuple, Optional

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


class BlueberryAnnotation(NamedTuple):
    """
    Parsed annotation: polygon in image coordinates + optional label.
    """
    polygon: List[Point]
    label: str = "blueberry"


def _parse_points_attr(text: str) -> List[Point]:
    """
    Parse a 'points' attribute string such as:
      "1768.00,3030.00;1741.30,3043.90; ..."
    into a list of (x, y) float tuples.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Missing or invalid 'points' attribute.")
    pts: List[Point] = []
    # Split by semicolon; allow stray spaces
    for token in text.strip().split(";"):
        token = token.strip()
        if not token:
            continue
        # Accept separators like "x,y" or "x , y"
        m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$", token)
        if not m:
            raise ValueError(f"Invalid point token in 'points': {token!r}")
        x = float(m.group(1))
        y = float(m.group(2))
        pts.append((x, y))
    if len(pts) < 3:
        raise ValueError(
            f"'points' must contain at least 3 coordinate pairs; got {len(pts)}."
        )
    return pts


def _to_int_or_none(attr: Optional[str], what: str) -> Optional[int]:
    if attr is None:
        return None
    try:
        return int(attr)
    except ValueError as e:
        raise ValueError(f"Image {what} is not a valid integer: {attr!r}") from e


def parse_xml(xml_text: str) -> List[BlueberryAnnotation]:
    """
    Parse blueberry polygons from XML text into image (pixel) coordinates.

    This parser is structured with a clearly delimited, user-adaptable mapping
    block so that different XML layouts (e.g. CVAT, LabelMe, custom) can be
    supported without changing the core logic.

    Parameters
    ----------
    xml_text:
        XML content as string.

    Returns
    -------
    list[BlueberryAnnotation]
        A list of per-instance polygons.

    Raises
    ------
    ValueError
        If the XML is malformed or required nodes/attributes are missing.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML: {e}") from e

    # Store (polygon, label) tuples explicitly for correct typing
    polygons_with_labels: List[Tuple[List[Point], str]] = []

    # === BEGIN USER-ADAPTABLE MAPPING (based on example XML) ===
    #
    # This block assumes CVAT's XML dump format where polygons live under:
    # <annotations><image width="" height="" ...><polygon points="x,y;..."/></image></annotations>
    #
    # If your XML differs, adjust ONLY the following lines (XPath/tag/attribute names).
    try:
        # 1) Locate images (needed in case of normalized coordinates; also documents the frame size).
        image_elems = list(root.findall(".//image"))
        if not image_elems:
            raise ValueError("No <image> elements found in XML.")
        for img in image_elems:
            width_attr = img.get("width")
            height_attr = img.get("height")
            img_w = _to_int_or_none(width_attr, "width")
            img_h = _to_int_or_none(height_attr, "height")
            if img_w is not None and img_w <= 0:
                raise ValueError("Image width must be a positive integer.")
            if img_h is not None and img_h <= 0:
                raise ValueError("Image height must be a positive integer.")

            # 2) For each polygon under the image, parse its point list
            for poly in img.findall("./polygon"):
                points_attr = poly.get("points")
                if points_attr is None:
                    raise ValueError("Polygon without 'points' attribute encountered.")
                pts = _parse_points_attr(points_attr)
                label_attr = poly.get("label")
                label = str(label_attr) if label_attr else "blueberry"

                # If polygons are normalized (0..1), convert to pixels using image size.
                normalized_flag = poly.get("normalized") or poly.get("is_normalized")
                if normalized_flag in {"1", "true", "True"}:
                    if img_w is None or img_h is None:
                        raise ValueError(
                            "Found normalized polygon but image width/height are missing."
                        )
                    # Validate normalized coordinates are actually in [0, 1] range
                    for i, (x, y) in enumerate(pts):
                        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                            raise ValueError(
                                f"Normalized polygon has coordinates outside [0,1] range: "
                                f"point {i} = ({x}, {y})"
                            )
                    pts = [(x * img_w, y * img_h) for (x, y) in pts]

                polygons_with_labels.append((pts, label))
    except Exception as e:
        # Re-raise with context to make debugging easier for the user-adaptable block.
        raise ValueError(f"XML mapping/parsing failed: {e}") from e
    # === END USER-ADAPTABLE MAPPING ===

    anns = [BlueberryAnnotation(polygon=poly, label=lbl) for (poly, lbl) in polygons_with_labels]
    logger.debug("Parsed %d polygons from XML.", len(anns))
    return anns


# --------------------------------------------------------------------------------------
# Example XML (for mapping/reference only; not used by the parser logic at runtime).
# Paste/keep this here to aid in adapting the mapping block above to your data.
EXAMPLE_CVAT_XML: str = r"""<?xml version="1.0" encoding="utf-8"?> 
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <id>1535020</id>
      <name>Heidelbeere dario</name>
      <size>1</size>
      <mode>annotation</mode>
      <overlap>0</overlap>
      <bugtracker></bugtracker>
      <created>2025-07-29 21:04:16.542816+00:00</created>
      <updated>2025-07-31 08:45:02.981134+00:00</updated>
      <subset>default</subset>
      <start_frame>0</start_frame>
      <stop_frame>0</stop_frame>
      <frame_filter></frame_filter>
      <segments>
        <segment>
          <id>2782978</id>
          <start>0</start>
          <stop>0</stop>
          <url>https://app.cvat.ai/api/jobs/2781090</url>
        </segment>
      </segments>
      <owner>
        <username>dario32</username>
        <email>dario.kubitzek@gmail.com</email>
      </owner>
      <assignee></assignee>
      <labels>
        <label>
          <name>Aufgetaut_UEB_2</name>
          <color>#7d0a4d</color>
          <type>any</type>
          <attributes>
          </attributes>
        </label>
      </labels>
    </task>
    <dumped>2025-07-31 09:54:02.094247+00:00</dumped>
  </meta>
  <image id="0" name="Aufgetaut_UEB_2.JPG" width="6000" height="4000">
    <polygon label="Aufgetaut_UEB_2" source="semi-auto" occluded="0" points="1768.00,3030.00;1741.30,3043.90;1713.60,3068.80;1688.90,3098.70;1677.80,3121.20;1669.50,3143.40;1663.60,3172.00;1665.10,3207.00;1671.60,3235.00;1679.60,3259.30;1688.60,3278.70;1700.50,3295.50;1727.40,3320.50;1769.20,3340.10;1807.50,3350.10;1857.60,3358.10;1889.50,3350.40;1953.70,3309.10;1969.80,3294.40;1981.80,3276.10;1990.30,3260.10;1998.50,3239.90;2006.90,3211.20;2009.70,3173.70;2001.60,3136.50;1989.70,3097.70;1966.40,3056.90;1939.70,3035.20;1911.60,3024.40;1876.50,3017.50;1844.10,3015.10;1817.60,3022.00;1793.50,3023.20" z_order="0">
    </polygon>
    <!-- MANY MORE <polygon> ... </polygon> entries ... -->
  </image>
</annotations>
"""
# --------------------------------------------------------------------------------------
