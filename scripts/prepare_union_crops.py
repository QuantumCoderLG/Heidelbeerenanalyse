#!/usr/bin/env python3
"""Prepare training crops enriched with convex hull metadata for Never berries."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

import visualize_union_boxes as vu


@dataclass
class ComponentInfo:
    label_value: int
    bbox: Tuple[int, int, int, int]  # (left, top, right, bottom)
    index: int

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop individual berries, associate convex hull annotations, and store "
            "metadata for training."
        ),
    )
    parser.add_argument(
        "--annotation",
        required=True,
        type=Path,
        help="Path to a CVAT XML annotation or a directory that contains multiple XML files.",
    )
    parser.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Directory that stores the original source images.",
    )
    parser.add_argument(
        "--mask-root",
        required=True,
        type=Path,
        help="Directory that contains per-image instance masks (PNG).",
    )
    parser.add_argument(
        "--class-label",
        default="never",
        help="Class label used for the generated crops.",
    )
    parser.add_argument(
        "--output-images",
        type=Path,
        default=Path("data/instance_crops/images/never"),
        help="Directory where cropped RGB images will be written.",
    )
    parser.add_argument(
        "--output-masks",
        type=Path,
        default=Path("data/instance_crops/masks/never"),
        help="Directory where cropped instance masks will be written.",
    )
    parser.add_argument(
        "--overlay-dir",
        default="data/instance_crops/overlays/union_crops",
        help="Optional directory for QA overlays (set to '' to disable).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/instance_crops/metadata/never_union_crops.jsonl"),
        help="Destination for the generated metadata (JSON lines).",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=6,
        help="Extra pixels added around each segmentation bounding box before cropping.",
    )
    parser.add_argument(
        "--relative-margin",
        type=float,
        default=0.12,
        help="Additional margin per side relative to the berry size (e.g. 0.15 -> +15% each side).",
    )
    parser.add_argument(
        "--bg-color",
        type=str,
        default="128,128,128",
        help="Background RGB used to fill outside the instance mask inside crops (e.g. '128,128,128').",
    )
    parser.add_argument(
        "--only-images",
        nargs="*",
        help="Optional list of image names (with extension) to process.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing crops whose image+mask files already exist.",
    )
    args = parser.parse_args()
    # Parse background color
    try:
        parts = [int(x) for x in str(args.bg_color).split(",")]
        if len(parts) != 3:
            raise ValueError
        args.bg_color = tuple(max(0, min(255, int(v))) for v in parts)
    except Exception:
        args.bg_color = (128, 128, 128)
    return args


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        if path:
            path.mkdir(parents=True, exist_ok=True)


def load_mask(mask_root: Path, image_name: str) -> Path:
    mask_name = Path(image_name).with_suffix(".png").name
    direct = mask_root / mask_name
    if direct.exists():
        return direct
    matches = list(mask_root.rglob(mask_name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Instance mask for {image_name} not found under {mask_root}")


def extract_components(mask: np.ndarray) -> List[ComponentInfo]:
    labels = [int(v) for v in np.unique(mask) if v != 0]
    components: List[ComponentInfo] = []
    temp: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for value in labels:
        ys, xs = np.where(mask == value)
        if ys.size == 0:
            continue
        top = int(ys.min())
        bottom = int(ys.max() + 1)
        left = int(xs.min())
        right = int(xs.max() + 1)
        temp.append((value, (left, top, right, bottom)))
    temp.sort(key=lambda item: (item[1][1], item[1][0]))  # top -> left ordering
    for idx, (value, bbox) in enumerate(temp, start=1):
        components.append(ComponentInfo(label_value=value, bbox=bbox, index=idx))
    return components


def axis_aligned_intersection(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    return float(right - left) * float(bottom - top)


def clip_and_translate_polygon(
    points: Sequence[Tuple[float, float]],
    crop_left: int,
    crop_top: int,
    crop_right: int,
    crop_bottom: int,
) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return []
    rect = [
        (crop_left, crop_top),
        (crop_right, crop_top),
        (crop_right, crop_bottom),
        (crop_left, crop_bottom),
    ]
    clipped = vu.sutherland_hodgman(points, rect)
    translated = [(float(x - crop_left), float(y - crop_top)) for x, y in clipped]
    return translated


def polygon_centroid(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    twice_area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        cross = x1 * y2 - x2 * y1
        twice_area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(twice_area) < 1e-9:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs) / len(points), sum(ys) / len(points))
    area = twice_area / 2.0
    cx /= (3.0 * twice_area)
    cy /= (3.0 * twice_area)
    return (cx, cy)


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_contains(bbox: Tuple[int, int, int, int], x: float, y: float) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def save_overlay(
    overlay_root: Path | None,
    class_label: str,
    crop_name: str,
    crop_image: Image.Image,
    relative_polygons: Sequence[Sequence[Tuple[float, float]]],
) -> Path | None:
    if not overlay_root:
        return None
    ensure_dirs(overlay_root / class_label)
    overlay = crop_image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    palette = vu.get_color_palette()
    for idx, polygon in enumerate(relative_polygons):
        if len(polygon) < 3:
            continue
        color = vu.ImageColor.getrgb(palette[idx % len(palette)])
        draw.polygon(polygon, outline=(*color, 255))
        vu.draw_polygon_outline(draw, polygon, color=(*color, 255), width=2)
    path = overlay_root / class_label / crop_name
    overlay.convert("RGB").save(path)
    return path


def crop_single_component(
    image: Image.Image,
    mask: np.ndarray,
    component: ComponentInfo,
    pad: int,
    relative_margin: float,
    bg_color: Tuple[int, int, int],
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    img_w, img_h = image.size
    w = component.width
    h = component.height
    side = max(1, max(w, h))
    relative_pad = int(round(side * max(0.0, relative_margin)))
    half_side = side / 2.0 + pad + relative_pad
    cx = (component.bbox[0] + component.bbox[2]) / 2.0
    cy = (component.bbox[1] + component.bbox[3]) / 2.0
    left = int(math.floor(cx - half_side))
    top = int(math.floor(cy - half_side))
    right = int(math.ceil(cx + half_side))
    bottom = int(math.ceil(cy + half_side))
    if right - left < 1:
        right = left + 1
    if bottom - top < 1:
        bottom = top + 1
    if left < 0:
        right = min(img_w, right - left)
        left = 0
    if top < 0:
        bottom = min(img_h, bottom - top)
        top = 0
    if right > img_w:
        shift = right - img_w
        left = max(0, left - shift)
        right = img_w
    if bottom > img_h:
        shift = bottom - img_h
        top = max(0, top - shift)
        bottom = img_h
    crop = image.crop((left, top, right, bottom))
    crop_mask_region = (mask[top:bottom, left:right] == component.label_value).astype(np.uint8) * 255
    # Fill background to a neutral gray to match existing pipeline crops
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    arr = np.array(crop, dtype=np.uint8)
    m = crop_mask_region > 0
    bg = np.array(bg_color, dtype=np.uint8).reshape(1, 1, 3)
    arr[~m] = bg
    crop = Image.fromarray(arr, mode="RGB")
    mask_image = Image.fromarray(crop_mask_region).convert("L")
    return crop, mask_image, (left, top, right, bottom)


def serialize_polygon(points: Sequence[Tuple[float, float]]) -> List[List[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for (x, y) in points]


def main() -> int:
    args = parse_args()
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    ensure_dirs(args.output_images, args.output_masks, overlay_dir.parent if overlay_dir else None)
    only_set = set(args.only_images) if args.only_images else None
    try:
        annotation_files = vu.gather_annotation_files(args.annotation)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    metadata_entries: List[dict] = []
    skipped_existing = 0
    written_crops = 0
    unassigned_regions: List[str] = []

    for annotation_file in annotation_files:
        for image_name, boxes in vu.read_annotation_file(annotation_file):
            if only_set and image_name not in only_set:
                continue
            if not boxes:
                continue
            try:
                image_path = vu.resolve_image_path(args.image_root, image_name)
            except FileNotFoundError as exc:
                print(exc)
                continue
            try:
                mask_path = load_mask(args.mask_root, image_name)
            except FileNotFoundError as exc:
                print(exc)
                continue

            image = Image.open(image_path).convert("RGB")
            mask_array = np.array(Image.open(mask_path))
            unions = vu.build_union_boxes(boxes, vu.MIN_INTERSECTION_RATIO)
            components = extract_components(mask_array)
            if not components:
                print(f"No components found in mask for {image_name}, skipping.")
                continue

            component_centers = {c.index: bbox_center(c.bbox) for c in components}
            assignments: Dict[int, List[Tuple[int, vu.UnionRegion]]] = {c.index: [] for c in components}
            for union_idx, region in enumerate(unions, start=1):
                best = None
                best_overlap = 0.0
                union_bbox = (
                    int(region.bbox.xtl),
                    int(region.bbox.ytl),
                    int(region.bbox.xbr),
                    int(region.bbox.ybr),
                )
                centroid = polygon_centroid(region.polygon)
                for component in components:
                    overlap = axis_aligned_intersection(union_bbox, component.bbox)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best = component
                if best and best_overlap > 0:
                    assignments[best.index].append((union_idx, region))
                    continue
                fallback = None
                if components:
                    for component in components:
                        if bbox_contains(component.bbox, centroid[0], centroid[1]):
                            fallback = component
                            break
                    if fallback is None:
                        fallback = min(
                            components,
                            key=lambda comp: (component_centers[comp.index][0] - centroid[0]) ** 2
                            + (component_centers[comp.index][1] - centroid[1]) ** 2,
                        )
                if fallback:
                    assignments[fallback.index].append((union_idx, region))
                else:
                    unassigned_regions.append(f"{image_name}:U{union_idx}")

            for component in components:
                crop_name = f"{Path(image_name).stem}_union_{component.index:02d}.png"
                image_out = args.output_images / crop_name
                mask_out = args.output_masks / crop_name

                if args.skip_existing and image_out.exists() and mask_out.exists():
                    skipped_existing += 1
                    continue

                ensure_dirs(image_out.parent, mask_out.parent)
                crop_image, crop_mask, crop_bbox = crop_single_component(
                    image=image,
                    mask=mask_array,
                    component=component,
                    pad=args.pad,
                    relative_margin=args.relative_margin,
                    bg_color=args.bg_color,
                )
                crop_image.save(image_out)
                crop_mask.save(mask_out)

                rel_polygons: List[List[Tuple[float, float]]] = []
                unions_for_component = []
                for union_idx, region in assignments.get(component.index, []):
                    relative = clip_and_translate_polygon(
                        region.polygon,
                        crop_bbox[0],
                        crop_bbox[1],
                        crop_bbox[2],
                        crop_bbox[3],
                    )
                    if len(relative) < 3:
                        continue
                    rel_polygons.append(relative)
                    unions_for_component.append(
                        {
                            "union_index": union_idx,
                            "member_boxes": len(region.members),
                            "global_area": round(region.area, 3),
                            "polygon": serialize_polygon(relative),
                        }
                    )

                overlay_path = save_overlay(
                    overlay_dir,
                    args.class_label,
                    crop_name,
                    crop_image,
                    rel_polygons,
                )

                entry = {
                    "class_label": args.class_label,
                    "annotation_file": str(annotation_file),
                    "source_image": str(image_path),
                    "source_mask": str(mask_path),
                    "image_name": image_name,
                    "component_index": component.index,
                    "component_label": component.label_value,
                    "crop_bbox": {
                        "left": crop_bbox[0],
                        "top": crop_bbox[1],
                        "right": crop_bbox[2],
                        "bottom": crop_bbox[3],
                    },
                    "crop_size": {
                        "width": crop_bbox[2] - crop_bbox[0],
                        "height": crop_bbox[3] - crop_bbox[1],
                    },
                    "crop_path": str(image_out),
                    "mask_path": str(mask_out),
                    "overlay_path": str(overlay_path) if overlay_path else None,
                    "union_regions": unions_for_component,
                }
                metadata_entries.append(entry)
                written_crops += 1

    ensure_dirs(args.metadata_path.parent)
    with args.metadata_path.open("w", encoding="utf-8") as fh:
        for entry in metadata_entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"Wrote {written_crops} crops with metadata to {args.metadata_path} "
        f"(skipped {skipped_existing} existing)."
    )
    if unassigned_regions:
        print(
            f"{len(unassigned_regions)} union region(s) could not be matched to a berry: "
            f"{', '.join(unassigned_regions[:10])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
