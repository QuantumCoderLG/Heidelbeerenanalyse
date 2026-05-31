from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
XML_EXT = ".xml"


def _iter_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            files.append(Path(dirpath) / fn)
    return files


def _stem_lower(p: Path) -> str:
    """
    Get normalized stem for case-insensitive matching.
    Uses casefold() for proper Unicode case folding.
    """
    return p.stem.casefold()


def discover_pairs(images_dir: Path, ann_dir: Path) -> list[tuple[Path, Path]]:
    """
    Discover image↔XML pairs by matching stems case-insensitively.

    Parameters
    ----------
    images_dir : Path
        Root directory containing images (searches recursively).
    ann_dir : Path
        Root directory containing XML annotations (searches recursively).

    Returns
    -------
    list[tuple[Path, Path]]
        List of (image_path, xml_path) pairs.

    Notes
    -----
    * If multiple images share the same stem, the first discovered is taken;
      others will be logged as warnings for ambiguity.
    * Unmatched images or XMLs will be logged as warnings.
    """
    images_dir = Path(images_dir)
    ann_dir = Path(ann_dir)

    img_map: Dict[str, Path] = {}
    duplicates: Dict[str, int] = {}
    for p in _iter_files(images_dir):
        if p.suffix.casefold() in IMAGE_EXTS:
            stem = _stem_lower(p)
            if stem in img_map:
                duplicates[stem] = duplicates.get(stem, 1) + 1
            else:
                img_map[stem] = p

    if duplicates:
        for stem, count in duplicates.items():
            logger.warning(
                "Duplicate image stem '%s' encountered %d times. Using first occurrence: %s",
                stem,
                count,
                img_map[stem],
            )

    xml_map: Dict[str, Path] = {}
    for p in _iter_files(ann_dir):
        if p.suffix.casefold() == XML_EXT:
            xml_map[_stem_lower(p)] = p

    pairs: List[Tuple[Path, Path]] = []
    matched_xml = set()
    matched_img = set()
    for stem, img_path in img_map.items():
        xml_path = xml_map.get(stem)
        if xml_path is not None:
            pairs.append((img_path, xml_path))
            matched_xml.add(stem)
            matched_img.add(stem)
        else:
            logger.warning("No XML found for image: %s", img_path)

    for stem, xml_path in xml_map.items():
        if stem not in matched_xml:
            logger.warning("No image found for XML: %s", xml_path)

    pairs.sort(key=lambda t: _stem_lower(t[0]))
    logger.info("Discovered %d image↔XML pairs.", len(pairs))
    return pairs


def infer_image_size(image_path: Path) -> tuple[int, int]:
    """
    Infer image size (width, height) without loading full pixel data.

    Parameters
    ----------
    image_path : Path

    Returns
    -------
    (width, height): tuple[int, int]
    """
    image_path = Path(image_path)
    with Image.open(image_path) as im:
        # Apply EXIF orientation to reflect what annotators/viewers saw
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:
            pass
        width, height = im.size
    return int(width), int(height)
