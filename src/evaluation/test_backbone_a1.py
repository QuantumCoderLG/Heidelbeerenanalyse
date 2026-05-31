from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from ..evaluation.apply_model_to_images import (
    instances_to_overlay,
    postprocess_probability,
    preprocess_image,
    resolve_checkpoint,
)
from ..evaluation.postprocessing import apply_postprocessing
from ..training.classifier_models import build_classifier
from ..training.models import build_model
from ..utils.io_utils import atomic_save_pil_image, load_image
from ..utils.box_utils import BBox, expand_bbox, keep_top_k_instances
from ..utils.image_utils import letterbox_resize, letterbox_mask
from ..utils.vis_utils import draw_boxes, make_contact_sheet


LOGGER = logging.getLogger("test_backbone_a1")


def _load_yaml(path: Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}














def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment blueberries, classify with Backbone A1, and save visualisations/CSV.")
    p.add_argument("--image", type=Path, required=True, help="Path to the input RGB image")
    p.add_argument("--seg-checkpoint", type=Path, required=True, help="Segmentation checkpoint (.pt or .safetensors)")
    p.add_argument("--seg-config", type=Path, default=Path("configs/train.yaml"), help="Segmentation training config")
    p.add_argument("--clf-checkpoint", type=Path, required=True, help="Backbone A1 checkpoint (best.pt)")
    p.add_argument("--clf-config", type=Path, default=Path("configs/backbone_a.yaml"), help="Backbone A config YAML")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/test_a1"))
    p.add_argument("--top-k", type=int, default=25, help="Keep top-K largest instances")
    p.add_argument("--margin", type=float, default=0.15, help="Relative bbox margin around instances")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--use-mask-channel", dest="use_mask_channel", action="store_true")
    p.add_argument("--no-mask-channel", dest="use_mask_channel", action="store_false")
    p.set_defaults(use_mask_channel=None)
    return p.parse_args(argv)


def load_segmentation(
    cfg_path: Path, checkpoint_path: Path, device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, object], float]:
    cfg = _load_yaml(cfg_path)
    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg, num_classes=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    channels_last = bool(cfg.get("train", {}).get("channels_last", False))
    if channels_last:
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    model.eval()
    # Determine threshold: prefer checkpoint, else config postproc.threshold, else 0.5
    post_cfg = cfg.get("postproc", {}) if isinstance(cfg, dict) else {}
    threshold = float(ckpt.get("threshold", post_cfg.get("threshold", 0.5))) if isinstance(ckpt, dict) else float(post_cfg.get("threshold", 0.5))
    return model, cfg, threshold


def segment_image(
    image_np: np.ndarray,
    seg_model: torch.nn.Module,
    seg_cfg: Dict[str, object],
    device: torch.device,
    *,
    amp: bool,
    threshold_override: float | None = None,
) -> Tuple[np.ndarray, Image.Image, Dict[str, float]]:
    data_cfg = seg_cfg.get("data", {}) if isinstance(seg_cfg, dict) else {}
    mean = data_cfg.get("image_mean", [0.485, 0.456, 0.406])
    std = data_cfg.get("image_std", [0.229, 0.224, 0.225])
    val_size = data_cfg.get("val_size") or data_cfg.get("train_size")
    keep_ratio = bool(data_cfg.get("keep_ratio", True))
    target_hw = None
    if val_size and len(val_size) == 2:
        target_hw = (int(val_size[0]), int(val_size[1]))

    post_cfg = seg_cfg.get("postproc", {}) if isinstance(seg_cfg, dict) else {}

    inputs, meta = preprocess_image(image_np, mean, std, target_hw, keep_ratio)
    inputs = inputs.unsqueeze(0).to(device)
    channels_last = bool(seg_cfg.get("train", {}).get("channels_last", False)) if isinstance(seg_cfg, dict) else False
    if channels_last:
        inputs = inputs.to(memory_format=torch.channels_last)

    ctx = torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")) if amp else nullcontext()
    with torch.no_grad():
        with ctx:
            logits = seg_model(inputs)["out"]
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
    probs = postprocess_probability(probs, meta)
    thr = float(post_cfg.get("threshold", 0.5))
    if threshold_override is not None:
        thr = float(threshold_override)
    _, instances = apply_postprocessing(probs, post_cfg, threshold=thr)
    overlay = instances_to_overlay(image_np, instances, max_instances=100)
    return instances.astype(np.int32), overlay, meta


def load_classifier(clf_cfg_path: Path, checkpoint_path: Path, device: torch.device, use_mask_channel_override: bool | None) -> Tuple[torch.nn.Module, Dict[str, object], float, float]:
    cfg = _load_yaml(clf_cfg_path)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    include_mask_channel = bool(data_cfg.get("include_mask_channel", True))
    mask_usage = data_cfg.get("mask_usage", "auto")
    if use_mask_channel_override is not None:
        include_mask_channel = bool(use_mask_channel_override)
    in_channels = 4 if include_mask_channel and mask_usage != "rgb_only" else 3

    model = build_classifier(
        backbone=model_cfg.get("name", "mobilenet_v3_small"),
        num_classes=1,
        in_channels=in_channels,
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=model_cfg.get("dropout"),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Try to find companion summary.json for temperature/threshold
    temperature = 1.0
    threshold = 0.5
    try:
        sum_path = checkpoint_path.parent / "summary.json"
        if sum_path.exists():
            summary = json.loads(sum_path.read_text(encoding="utf-8"))
            temperature = float(summary.get("temperature", 1.0))
            threshold = float(summary.get("threshold", 0.5))
    except Exception:
        pass

    return model, cfg, temperature, threshold


def classify_instances(
    image_np: np.ndarray,
    instances: np.ndarray,
    boxes: List[BBox],
    comp_labels: List[int],
    *,
    clf_model: torch.nn.Module,
    clf_cfg: Dict[str, object],
    device: torch.device,
    temperature: float,
    use_mask_channel: bool,
) -> Tuple[List[float], List[str], List[np.ndarray]]:
    data_cfg = clf_cfg.get("data", {}) if isinstance(clf_cfg, dict) else {}
    input_size = data_cfg.get("input_size", [320, 320])
    target_hw = (int(input_size[0]), int(input_size[1]))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    crops_vis: List[np.ndarray] = []
    batch_list: List[torch.Tensor] = []
    for box, comp_label in zip(boxes, comp_labels):
        crop_rgb = image_np[box.y0:box.y1, box.x0:box.x1]
        crop_rgb = letterbox_resize(crop_rgb, target_hw)
        crop_float = crop_rgb.astype(np.float32) / 255.0
        crop_norm = (crop_float - mean) / std
        x = torch.from_numpy(crop_norm.transpose(2, 0, 1)).contiguous()  # (3,H,W)

        if use_mask_channel:
            mask_box = (instances[box.y0:box.y1, box.x0:box.x1] == comp_label).astype(np.uint8)
            mask_resized = letterbox_mask(mask_box, target_hw)
            m = torch.from_numpy((mask_resized > 0).astype(np.float32)).unsqueeze(0)
            x = torch.cat([x, m], dim=0)

        batch_list.append(x)
        crops_vis.append(crop_rgb)

    if not batch_list:
        return [], [], []

    batch = torch.stack(batch_list, dim=0).to(device)
    channels_last = bool(clf_cfg.get("training", {}).get("channels_last", True)) if isinstance(clf_cfg, dict) else True
    if channels_last:
        batch = batch.to(memory_format=torch.channels_last)
    with torch.no_grad():
        logits = clf_model(batch).squeeze(1)
        if temperature and temperature > 0:
            logits = logits / float(temperature)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    prob_list = [float(p) for p in probs]
    labels = [f"notberry {p:.2f}" for p in prob_list]
    return prob_list, labels, crops_vis


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Using device: %s", device)

    # Load image
    image_np, _, _ = load_image(args.image)

    # Load segmentation model and run inference
    seg_model, seg_cfg, seg_threshold = load_segmentation(args.seg_config, args.seg_checkpoint, device)
    instances_raw, overlay_seg, meta = segment_image(
        image_np, seg_model, seg_cfg, device, amp=args.amp, threshold_override=seg_threshold
    )
    # Filter to top-K largest instances
    instances = keep_top_k_instances(instances_raw, max(1, int(args.top_k)))

    # Find bounding boxes
    unique_labels = [int(x) for x in np.unique(instances) if int(x) != 0]
    boxes: List[BBox] = []
    H, W = instances.shape
    for label in unique_labels:
        ys, xs = np.where(instances == label)
        if ys.size == 0:
            continue
        x, y, w, h = cv2.boundingRect(np.column_stack((xs, ys)))
        boxes.append(expand_bbox(x, y, w, h, args.margin, (H, W)))

    # Load classifier (A1)
    clf_model, clf_cfg, temperature, threshold = load_classifier(
        args.clf_config, args.clf_checkpoint, device, args.use_mask_channel
    )
    use_mask_channel = False
    if args.use_mask_channel is not None:
        use_mask_channel = bool(args.use_mask_channel)
    else:
        data_cfg = clf_cfg.get("data", {}) if isinstance(clf_cfg, dict) else {}
        include_mask = bool(data_cfg.get("include_mask_channel", True))
        mask_usage = data_cfg.get("mask_usage", "auto")
        use_mask_channel = include_mask and mask_usage != "rgb_only"

    # Track which component label corresponds to each box
    comp_labels: List[int] = []
    for label in unique_labels:
        # same order as boxes construction loop
        comp_labels.append(label)

    probs, labels_text, crops_vis = classify_instances(
        image_np,
        instances,
        boxes,
        comp_labels,
        clf_model=clf_model,
        clf_cfg=clf_cfg,
        device=device,
        temperature=temperature,
        use_mask_channel=use_mask_channel,
    )

    # Derive final decisions
    decisions = ["notberry" if p >= threshold else "berry" for p in probs]
    labels_draw = [f"#{i+1} {d} {p:.2f}" for i, (d, p) in enumerate(zip(decisions, probs))]

    out_root = args.output_dir / args.image.stem
    out_root.mkdir(parents=True, exist_ok=True)

    # Save visualisations
    overlay_seg_path = out_root / "01_seg_overlay.png"
    overlay_boxes = draw_boxes(image_np, boxes, labels_draw)
    overlay_boxes_path = out_root / "02_boxes_classified.png"
    crops_sheet = make_contact_sheet(crops_vis, labels_draw, cols=5)
    crops_sheet_path = out_root / "03_crops_sheet.png"
    mask_vis = Image.fromarray((instances > 0).astype(np.uint8) * 255, mode="L")
    mask_vis_path = out_root / "00_instances_mask.png"

    atomic_save_pil_image(mask_vis, mask_vis_path)
    atomic_save_pil_image(overlay_seg, overlay_seg_path)
    atomic_save_pil_image(overlay_boxes, overlay_boxes_path)
    atomic_save_pil_image(crops_sheet, crops_sheet_path)

    # Save CSV with details
    csv_path = out_root / "predictions.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("id,x0,y0,x1,y1,width,height,area,prob_notberry,decision\n")
        for i, (box, p, d) in enumerate(zip(boxes, probs, decisions), start=1):
            fh.write(f"{i},{box.x0},{box.y0},{box.x1},{box.y1},{box.w},{box.h},{box.area},{p:.6f},{d}\n")

    # Save meta JSON
    meta_out = {
        "image_path": str(args.image),
        "segmentation_checkpoint": str(args.seg_checkpoint),
        "segmentation_config": str(args.seg_config),
        "classifier_checkpoint": str(args.clf_checkpoint),
        "classifier_config": str(args.clf_config),
        "top_k": int(args.top_k),
        "margin": float(args.margin),
        "temperature": float(temperature),
        "threshold": float(threshold),
        "use_mask_channel": bool(use_mask_channel),
        "num_instances": int(len(boxes)),
    }
    (out_root / "meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    LOGGER.info("Saved outputs to %s", out_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
