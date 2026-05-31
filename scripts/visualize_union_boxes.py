#!/usr/bin/env python3
"""Visualize Never Heidelbeer bounding boxes and their unions.

The script draws all annotated bounding boxes together with union boxes that
merge every overlapping set of boxes. The output image highlights what the
training models would effectively see and prints a short summary so humans can
confirm that every relevant region was highlighted (or spot deviations).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import xml.etree.ElementTree as ET

from PIL import Image, ImageColor, ImageDraw, ImageFont
import math

# Union boxes should be as inclusive as possible -> require only the smallest
# overlap threshold (actual intersection > 0).
MIN_INTERSECTION_RATIO = 0.0
EPSILON = 1e-9


@dataclass(frozen=True)
class Box:
    """Represents a (possibly rotated) CVAT bounding box."""

    xtl: float
    ytl: float
    xbr: float
    ybr: float
    label: str
    rotation: float | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.xbr - self.xtl)

    @property
    def height(self) -> float:
        return max(0.0, self.ybr - self.ytl)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.xtl + self.width / 2, self.ytl + self.height / 2)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.xtl, self.ytl, self.xbr, self.ybr)

    @cached_property
    def polygon(self) -> Tuple[Tuple[float, float], ...]:
        """Returns the rectangle corners in CCW order, rotation-aware."""
        cx, cy = self.center
        half_w = self.width / 2
        half_h = self.height / 2
        base_rect = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]
        angle = self.rotation or 0.0  # Follow CVAT's rotation direction.
        if abs(angle) < EPSILON:
            # Shortcut for axis-aligned boxes.
            return (
                (self.xtl, self.ytl),
                (self.xbr, self.ytl),
                (self.xbr, self.ybr),
                (self.xtl, self.ybr),
            )
        theta = math.radians(angle)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rotated = []
        for x, y in base_rect:
            rx = x * cos_t - y * sin_t
            ry = x * sin_t + y * cos_t
            rotated.append((cx + rx, cy + ry))
        return tuple(rotated)


@dataclass(frozen=True)
class UnionRegion:
    bbox: Box
    members: Tuple[Box, ...]
    polygon: Tuple[Tuple[float, float], ...]

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Heidelbeer bounding boxes and their union boxes.",
    )
    parser.add_argument(
        "--annotation",
        required=True,
        type=Path,
        help=(
            "Path to a single CVAT XML file or a directory that contains multiple XMLs "
            "(e.g. data/BBoxes_annotation_data)."
        ),
    )
    parser.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Directory that contains the original images (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/union_box_visualizations"),
        help="Destination directory for the rendered overlay images.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Opacity (0-1) for union box fills.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=4,
        help="Line width for union boxes.",
    )
    parser.add_argument(
        "--individual-width",
        type=int,
        default=2,
        help="Line width for individual annotation boxes (thin context outlines).",
    )
    parser.add_argument(
        "--hide-individual",
        action="store_true",
        help="Skip drawing the thin original bounding boxes.",
    )
    parser.add_argument(
        "--image-rotations",
        nargs="*",
        default=[],
        help=(
            "Optional per-image rotations (clockwise) for extra outputs. "
            "Format: IMAGE_NAME:ANGLE (e.g. Never_1_5.JPG:90)."
        ),
    )
    return parser.parse_args()


def read_annotation_file(annotation_path: Path) -> List[Tuple[str, List[Box]]]:
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    images = []
    for image_el in root.findall(".//image"):
        name = image_el.get("name")
        if not name:
            continue
        boxes: List[Box] = []
        for box_el in image_el.findall("box"):
            boxes.append(
                Box(
                    xtl=float(box_el.get("xtl", 0.0)),
                    ytl=float(box_el.get("ytl", 0.0)),
                    xbr=float(box_el.get("xbr", 0.0)),
                    ybr=float(box_el.get("ybr", 0.0)),
                    label=box_el.get("label", ""),
                    rotation=float(box_el.get("rotation"))
                    if box_el.get("rotation") is not None
                    else None,
                )
            )
        images.append((name, boxes))
    if not images:
        raise ValueError(f"No <image> nodes found in {annotation_path}")
    return images


def resolve_image_path(image_root: Path, image_name: str) -> Path:
    direct = image_root / image_name
    if direct.exists():
        return direct
    matches = list(image_root.rglob(image_name))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not locate {image_name} inside {image_root}. "
        "Use --image-root to point to the directory that stores the original files."
    )


def boxes_overlap(a: Box, b: Box, min_ratio: float) -> bool:
    intersection_area = polygon_intersection_area(a.polygon, b.polygon)
    if intersection_area <= EPSILON:
        return False
    if min_ratio <= 0:
        return True
    min_area = min(a.area, b.area)
    return min_area > 0 and (intersection_area / min_area) >= min_ratio


def build_union_boxes(boxes: Sequence[Box], min_ratio: float) -> List[UnionRegion]:
    if not boxes:
        return []
    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i == root_j:
            return
        parent[root_j] = root_i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_overlap(boxes[i], boxes[j], min_ratio):
                union(i, j)

    grouped: defaultdict[int, List[Box]] = defaultdict(list)
    for idx, box in enumerate(boxes):
        grouped[find(idx)].append(box)

    unions: List[UnionRegion] = []
    for member_boxes in grouped.values():
        points: List[Tuple[float, float]] = []
        for b in member_boxes:
            points.extend(b.polygon)
        hull = convex_hull(points)
        xs = [p[0] for p in hull]
        ys = [p[1] for p in hull]
        xtl = min(xs)
        ytl = min(ys)
        xbr = max(xs)
        ybr = max(ys)
        unions.append(
            UnionRegion(
                bbox=Box(xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr, label="Union"),
                members=tuple(member_boxes),
                polygon=hull,
            )
        )
    unions.sort(key=lambda region: region.area, reverse=True)
    return unions


def get_color_palette() -> List[str]:
    # Hand-picked to stay visible above natural colors.
    return [
        "#ff6b6b",
        "#ffa600",
        "#4ecdc4",
        "#5e60ce",
        "#f94144",
        "#43aa8b",
        "#f3722c",
        "#277da1",
    ]


def draw_visualization(
    image_path: Path,
    boxes: Sequence[Box],
    unions: Sequence[UnionRegion],
    alpha: float,
    union_width: int,
    individual_width: int,
    show_individual: bool,
) -> Image.Image:
    base = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
    palette = get_color_palette()
    font = ImageFont.load_default()

    for idx, region in enumerate(unions):
        color_hex = palette[idx % len(palette)]
        color_rgb = ImageColor.getrgb(color_hex)
        fill = (*color_rgb, max(0, min(255, int(alpha * 255))))
        outline = (*color_rgb, 255)
        overlay_draw.polygon(region.polygon, fill=fill)
        draw_polygon_outline(
            draw=overlay_draw,
            points=region.polygon,
            color=outline,
            width=union_width,
        )
        text = f"U{idx + 1} ({len(region.members)} bx)"
        text_pos = (region.bbox.xtl + 4, region.bbox.ytl + 4)
        overlay_draw.text(text_pos, text, fill=(255, 255, 255, 220), font=font)

    composed = Image.alpha_composite(base, overlay)

    if show_individual and boxes:
        draw = ImageDraw.Draw(composed)
        for box in boxes:
            draw_polygon_outline(
                draw=draw,
                points=box.polygon,
                color=(230, 230, 230),
                width=individual_width,
            )
    return composed.convert("RGB")


def summarize(image_name: str, boxes: Sequence[Box], unions: Sequence[UnionRegion]) -> str:
    total_boxes = len(boxes)
    singleton_unions = [u for u in unions if len(u.members) == 1]
    overlapping_groups = len(unions) - len(singleton_unions)
    message_parts = [
        f"{image_name}: {total_boxes} annotations bundled into {len(unions)} union box(es).",
    ]
    if overlapping_groups:
        message_parts.append(f"{overlapping_groups} group(s) merged multiple overlapping boxes.")
    if singleton_unions:
        message_parts.append(
            f"{len(singleton_unions)} box(es) had no overlap -> visualized as-is for manual review."
        )
    else:
        message_parts.append("All boxes participated in at least one overlap-based union.")
    return " ".join(message_parts)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_rotation_specs(specs: Sequence[str]) -> Dict[str, List[float]]:
    rotations: Dict[str, List[float]] = defaultdict(list)
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid rotation specification '{spec}'. Expected IMAGE:ANGLE.")
        image_name, angle_str = spec.split(":", 1)
        try:
            angle = float(angle_str)
        except ValueError as exc:
            raise ValueError(f"Invalid rotation angle '{angle_str}' in '{spec}'.") from exc
        rotations[image_name].append(angle)
    return rotations


def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_intersection_area(
    subject: Sequence[Tuple[float, float]], clip: Sequence[Tuple[float, float]]
) -> float:
    intersection = sutherland_hodgman(subject, clip)
    if len(intersection) < 3:
        return 0.0
    return polygon_area(intersection)


def sutherland_hodgman(
    subject: Sequence[Tuple[float, float]],
    clip: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    output = list(subject)
    if not output:
        return output
    for i in range(len(clip)):
        edge_start = clip[i]
        edge_end = clip[(i + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break
        S = input_list[-1]
        for E in input_list:
            if is_inside(E, edge_start, edge_end):
                if not is_inside(S, edge_start, edge_end):
                    output.append(intersection_point(S, E, edge_start, edge_end))
                output.append(E)
            elif is_inside(S, edge_start, edge_end):
                output.append(intersection_point(S, E, edge_start, edge_end))
            S = E
    return output


def is_inside(
    point: Tuple[float, float],
    edge_start: Tuple[float, float],
    edge_end: Tuple[float, float],
) -> bool:
    (x, y) = point
    (x1, y1) = edge_start
    (x2, y2) = edge_end
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= -EPSILON


def intersection_point(
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
    edge_start: Tuple[float, float],
    edge_end: Tuple[float, float],
) -> Tuple[float, float]:
    x1, y1 = line_start
    x2, y2 = line_end
    x3, y3 = edge_start
    x4, y4 = edge_end
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < EPSILON:
        return line_end
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def convex_hull(points: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return tuple(unique)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return tuple(hull)


def draw_polygon_outline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[float, float]],
    color: Tuple[int, ...],
    width: int,
) -> None:
    if len(points) < 2:
        return
    for i in range(len(points)):
        start = points[i]
        end = points[(i + 1) % len(points)]
        draw.line([start, end], fill=color, width=width)


def gather_annotation_files(annotation_target: Path) -> List[Path]:
    if annotation_target.is_file():
        return [annotation_target]
    if annotation_target.is_dir():
        files = sorted(annotation_target.rglob("*.xml"))
        if not files:
            raise ValueError(f"No XML annotation files found inside {annotation_target}")
        return files
    raise FileNotFoundError(f"Annotation path {annotation_target} does not exist.")


def main() -> int:
    args = parse_args()
    ensure_output_dir(args.output_dir)
    any_failures = False
    try:
        annotation_files = gather_annotation_files(args.annotation)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        image_rotations = parse_rotation_specs(args.image_rotations)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    for annotation_file in annotation_files:
        try:
            images = read_annotation_file(annotation_file)
        except Exception as exc:  # pragma: no cover - parsing errors need reporting
            print(f"Failed to parse {annotation_file}: {exc}", file=sys.stderr)
            any_failures = True
            continue

        for image_name, boxes in images:
            if not boxes:
                print(f"Skipping {image_name} from {annotation_file}: no boxes.", file=sys.stderr)
                continue
            try:
                image_path = resolve_image_path(args.image_root, image_name)
            except FileNotFoundError as exc:
                print(exc, file=sys.stderr)
                any_failures = True
                continue

            unions = build_union_boxes(boxes, MIN_INTERSECTION_RATIO)
            rendered = draw_visualization(
                image_path=image_path,
                boxes=boxes,
                unions=unions,
                alpha=args.alpha,
                union_width=args.line_width,
                individual_width=args.individual_width,
                show_individual=not args.hide_individual,
            )
            output_path = args.output_dir / f"{Path(image_name).stem}_union_overlay.png"
            rendered.save(output_path)
            print(f"[{annotation_file.name}] {summarize(image_name, boxes, unions)}")
            print(f"Saved visualization to {output_path}")
            for angle in image_rotations.get(image_name, []):
                rotated = rendered.rotate(-angle, expand=True)
                angle_label = format_angle_label(angle)
                rotated_path = (
                    args.output_dir / f"{Path(image_name).stem}_union_overlay_red{angle_label}.png"
                )
                rotated.save(rotated_path)
                print(f"Saved {angle_label}° rotation for {image_name} -> {rotated_path}")

    return 1 if any_failures else 0


def format_angle_label(angle: float) -> str:
    normalized = angle % 360
    if math.isclose(normalized, round(normalized)):
        return str(int(round(normalized)))
    return f"{normalized:.2f}".replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
