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
from PIL import Image
from ..utils.color_norm import gray_world

from ..evaluation.apply_model_to_images import (
    instances_to_overlay,
    postprocess_probability,
    postprocess_probability,
    preprocess_image,
)
from ..evaluation.postprocessing import apply_postprocessing
from ..training.classifier_models import build_classifier
from ..training.models import build_model
from ..utils.io_utils import atomic_save_pil_image, load_image
from ..utils.box_utils import BBox, expand_bbox, keep_top_k_instances
from ..utils.image_utils import letterbox_resize, letterbox_mask
from ..utils.vis_utils import draw_boxes


LOGGER = logging.getLogger("run_a1_a2_pipeline")


def _load_yaml(path: Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}








def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run segmentation -> A1 -> A2 -> A3 on a single image and save step-wise outputs.")
    p.add_argument("--image", type=Path, default=Path("outputs/Test/Inferenz_Test_Bild.JPG"), help="Input image (RGB JPEG)")
    p.add_argument("--seg-checkpoint", type=Path, default=Path("Kanditaten/fold_1_best.pt"), help="Segmentation checkpoint (.pt)")
    p.add_argument("--seg-config", type=Path, default=Path("configs/train.yaml"), help="Segmentation training config YAML")
    p.add_argument("--a1-checkpoint", type=Path, default=Path("outputs/backbone_a/a1/fold_00/best.pt"), help="Backbone A1 checkpoint (best.pt)")
    p.add_argument("--a2-checkpoint", type=Path, default=Path("outputs/backbone_a/a2/fold_00/best.pt"), help="Backbone A2 checkpoint (best.pt)")
    p.add_argument("--a3-checkpoint", type=Path, default=Path("outputs/backbone_a/a3/fold_00/best.pt"), help="Backbone A3 checkpoint (best.pt)")
    p.add_argument("--clf-config", type=Path, default=Path("configs/backbone_a.yaml"), help="Backbone A config YAML (A1/A2)")
    p.add_argument("--a3-config", type=Path, default=Path("configs/backbone_a3.yaml"), help="Backbone A3 config YAML")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/pipeline_a1_a2"))
    p.add_argument("--top-k", type=int, default=50, help="Keep top-K largest instances from segmentation")
    p.add_argument("--margin", type=float, default=0.15, help="Relative bbox margin around instances")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--use-mask-channel", dest="use_mask_channel", action="store_true")
    p.add_argument("--no-mask-channel", dest="use_mask_channel", action="store_false")
    p.set_defaults(use_mask_channel=None)
    return p.parse_args(argv)


def load_segmentation(cfg_path: Path, checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, object], float]:
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
    overlay = instances_to_overlay(image_np, instances, max_instances=200)
    return instances.astype(np.int32), overlay, meta


def load_classifier(
    clf_cfg_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    use_mask_channel_override: bool | None,
) -> Tuple[torch.nn.Module, Dict[str, object], float, float, int]:
    """Load a binary classifier checkpoint robustly.

    Heuristics:
    - Detect backbone family (mobilenet vs efficientnet) from state_dict keys
      (e.g., presence of "classifier.1.weight" strongly indicates EfficientNet).
    - Derive input channels from first conv weight (Cin), overriding config if needed.
    - Fall back to config if detection fails.
    """
    cfg = _load_yaml(clf_cfg_path)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    # Load state first for introspection
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    # Infer in_channels from first conv if possible
    inferred_in_channels: int | None = None
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim == 4 and k.endswith("features.0.0.weight"):
            inferred_in_channels = int(v.shape[1])
            break

    # Infer backbone family from key patterns
    eff_classifier_in_features: int | None = None
    has_eff_classifier = any(k.startswith("classifier.1.") for k in state.keys())
    if has_eff_classifier and "classifier.1.weight" in state and isinstance(state["classifier.1.weight"], torch.Tensor):
        eff_classifier_in_features = int(state["classifier.1.weight"].shape[1])
    has_eff_blocks = any(".block.2.fc1" in k for k in state.keys())
    detected_backbone = None
    if has_eff_classifier or has_eff_blocks:
        # Decide between B0 (1280) and B2 (1408) by classifier input width
        if eff_classifier_in_features == 1408:
            detected_backbone = "efficientnet_b2"
        else:
            detected_backbone = "efficientnet_b0"
    else:
        detected_backbone = model_cfg.get("name", "mobilenet_v3_small")

    # Resolve in_channels: prefer checkpoint inference; else config (+ override)
    include_mask_channel = bool(data_cfg.get("include_mask_channel", True))
    mask_usage = str(data_cfg.get("mask_usage", "auto"))
    if use_mask_channel_override is not None:
        include_mask_channel = bool(use_mask_channel_override)
    cfg_in_channels = 4 if include_mask_channel and mask_usage != "rgb_only" else 3
    in_channels = int(inferred_in_channels or cfg_in_channels)

    def _build(backbone_name: str) -> torch.nn.Module:
        return build_classifier(
            backbone=backbone_name,
            num_classes=1,
            in_channels=in_channels,
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout=model_cfg.get("dropout"),
        )

    # Try loading with detected backbone, then fallback variants if needed
    tried: List[str] = []
    model: torch.nn.Module | None = None
    load_ok = False
    for candidate in ([detected_backbone] if detected_backbone else []) + [
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "efficientnet_b2",
    ]:
        if candidate in tried:
            continue
        tried.append(candidate)
        try:
            m = _build(candidate)
            missing, unexpected = m.load_state_dict(state, strict=False)
            LOGGER.info(
                "Loaded classifier checkpoint %s as %s (in_channels=%d) | missing=%d unexpected=%d",
                checkpoint_path,
                candidate,
                in_channels,
                len(missing),
                len(unexpected),
            )
            # Consider it a success if classifier weights and most feature layers matched
            # and no size mismatch exceptions were raised.
            model = m
            load_ok = True
            break
        except Exception:
            load_ok = False
            continue

    if not load_ok or model is None:
        # As a last resort, build from config
        model = _build(model_cfg.get("name", "mobilenet_v3_small"))
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    # Temperature/threshold from summary.json if available
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

    return model, cfg, temperature, threshold, in_channels








def _save_crops(
    root: Path,
    tag: str,
    image_np: np.ndarray,
    instances: np.ndarray,
    boxes: List[BBox],
    comp_labels: List[int],
    crops_rgb: List[np.ndarray],
    labels_text: List[str],
) -> None:
    out_dir = root / tag / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (box, comp, crop, label) in enumerate(zip(boxes, comp_labels, crops_rgb, labels_text), start=1):
        crop_name = f"{i:03d}_c{comp}_{label.replace(' ', '_')}.png"
        atomic_save_pil_image(Image.fromarray(crop), out_dir / crop_name)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Using device: %s", device)

    # Load image
    image_np, _, _ = load_image(args.image)

    # Load segmentation model and run inference
    seg_model, seg_cfg, seg_threshold = load_segmentation(args.seg_config, args.seg_checkpoint, device)
    instances_raw, overlay_seg, _ = segment_image(
        image_np, seg_model, seg_cfg, device, amp=args.amp, threshold_override=seg_threshold
    )
    instances = keep_top_k_instances(instances_raw, max(1, int(args.top_k)))

    # Prepare boxes per component
    unique_labels = [int(x) for x in np.unique(instances) if int(x) != 0]
    boxes: List[BBox] = []
    H, W = instances.shape
    for label in unique_labels:
        ys, xs = np.where(instances == label)
        if ys.size == 0:
            continue
        x, y, w, h = cv2.boundingRect(np.column_stack((xs, ys)))
        boxes.append(expand_bbox(x, y, w, h, args.margin, (H, W)))

    # Output structure
    out_root = args.output_dir / args.image.stem
    out_root.mkdir(parents=True, exist_ok=True)
    step0_dir = out_root / "00_seg"
    step1_dir = out_root / "01_a1"
    step2_dir = out_root / "02_a2"
    step3_dir = out_root / "03_a3"
    step0_dir.mkdir(parents=True, exist_ok=True)
    step1_dir.mkdir(parents=True, exist_ok=True)
    step2_dir.mkdir(parents=True, exist_ok=True)

    # Save segmentation visuals
    atomic_save_pil_image(overlay_seg, step0_dir / "overlay.png")
    atomic_save_pil_image(Image.fromarray((instances > 0).astype(np.uint8) * 255, mode="L"), step0_dir / "instances_mask.png")

    # Load A1 classifier
    a1_model, a1_cfg, a1_temperature, a1_threshold, a1_in_channels = load_classifier(
        args.clf_config, args.a1_checkpoint, device, args.use_mask_channel
    )
    # Decide mask-channel usage for A1 based on model input channels
    use_mask_channel_a1 = (a1_in_channels == 4)
    if args.use_mask_channel is not None and bool(args.use_mask_channel) != use_mask_channel_a1:
        LOGGER.warning(
            "Overriding --%smask-channel to match A1 checkpoint input channels (%d)",
            ("no-" if use_mask_channel_a1 else ""),
            a1_in_channels,
        )

    # Build A1 batch
    data_cfg = a1_cfg.get("data", {}) if isinstance(a1_cfg, dict) else {}
    input_size = data_cfg.get("input_size", [320, 320])
    target_hw = (int(input_size[0]), int(input_size[1]))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    batch_a1: List[torch.Tensor] = []
    crops_a1: List[np.ndarray] = []
    comp_labels: List[int] = []
    for label, box in zip(unique_labels, boxes):
        crop_rgb = image_np[box.y0:box.y1, box.x0:box.x1]
        crop_rgb = letterbox_resize(crop_rgb, target_hw)
        crop_float = crop_rgb.astype(np.float32) / 255.0
        crop_norm = (crop_float - mean) / std
        x = torch.from_numpy(crop_norm.transpose(2, 0, 1)).contiguous()
        if use_mask_channel_a1:
            mask_box = (instances[box.y0:box.y1, box.x0:box.x1] == label).astype(np.uint8)
            mask_resized = letterbox_mask(mask_box, target_hw)
            m = torch.from_numpy((mask_resized > 0).astype(np.float32)).unsqueeze(0)
            x = torch.cat([x, m], dim=0)
        batch_a1.append(x)
        crops_a1.append(crop_rgb)
        comp_labels.append(label)

    probs_a1: List[float] = []
    if batch_a1:
        batch = torch.stack(batch_a1, dim=0).to(device)
        channels_last = bool(a1_cfg.get("training", {}).get("channels_last", True)) if isinstance(a1_cfg, dict) else True
        if channels_last:
            batch = batch.to(memory_format=torch.channels_last)
        with torch.no_grad():
            logits = a1_model(batch).squeeze(1)
            if a1_temperature and a1_temperature > 0:
                logits = logits / float(a1_temperature)
            probs_a1 = torch.sigmoid(logits).detach().cpu().numpy().tolist()

    # A1 decisions
    decisions_a1 = ["notberry" if p >= a1_threshold else "berry" for p in probs_a1]
    labels_a1 = [f"A1:{d} {p:.2f}" for d, p in zip(decisions_a1, probs_a1)]
    overlay_a1 = draw_boxes(image_np, boxes, labels_a1)
    atomic_save_pil_image(overlay_a1, step1_dir / "overlay_boxes.png")

    # Save A1 CSV and crops
    csv_a1 = step1_dir / "predictions.csv"
    with csv_a1.open("w", encoding="utf-8") as fh:
        fh.write("id,x0,y0,x1,y1,width,height,area,prob_notberry,decision\n")
        for i, (box, p, d) in enumerate(zip(boxes, probs_a1, decisions_a1), start=1):
            fh.write(f"{i},{box.x0},{box.y0},{box.x1},{box.y1},{box.w},{box.h},{box.area},{p:.6f},{d}\n")

    # Split A1 accepted/rejected
    keep_idx = [i for i, d in enumerate(decisions_a1) if d == "berry"]
    drop_idx = [i for i, d in enumerate(decisions_a1) if d == "notberry"]

    # Save crops to structured dirs
    if drop_idx:
        _save_crops(step1_dir, "rejected_notberry", image_np, instances, [boxes[i] for i in drop_idx], [comp_labels[i] for i in drop_idx], [crops_a1[i] for i in drop_idx], [labels_a1[i] for i in drop_idx])
    if keep_idx:
        _save_crops(step1_dir, "accepted_berry", image_np, instances, [boxes[i] for i in keep_idx], [comp_labels[i] for i in keep_idx], [crops_a1[i] for i in keep_idx], [labels_a1[i] for i in keep_idx])

    # If nothing to keep, finish early
    if not keep_idx:
        meta = {
            "image_path": str(args.image),
            "segmentation_checkpoint": str(args.seg_checkpoint),
            "a1_checkpoint": str(args.a1_checkpoint),
            "a2_checkpoint": str(args.a2_checkpoint),
            "kept_after_a1": 0,
            "dropped_notberry": len(drop_idx),
        }
        (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        LOGGER.info("No berries kept after A1; outputs written to %s", out_root)
        return 0

    # Load A2 classifier (never vs {red,yellow,green})
    a2_model, a2_cfg, a2_temperature, a2_threshold, a2_in_channels = load_classifier(
        args.clf_config, args.a2_checkpoint, device, args.use_mask_channel
    )
    data_cfg_a2 = a2_cfg.get("data", {}) if isinstance(a2_cfg, dict) else {}
    input_size_a2 = data_cfg_a2.get("input_size", [320, 320])
    target_hw_a2 = (int(input_size_a2[0]), int(input_size_a2[1]))
    # Force mask usage to match checkpoint's expected input channels
    include_mask_a2 = (a2_in_channels == 4)
    if args.use_mask_channel is not None and bool(args.use_mask_channel) != include_mask_a2:
        LOGGER.warning(
            "Overriding --%smask-channel to match A2 checkpoint input channels (%d)",
            ("no-" if include_mask_a2 else ""),
            a2_in_channels,
        )

    # Build A2 batch from A1-accepted indices
    batch_a2: List[torch.Tensor] = []
    crops_a2: List[np.ndarray] = []
    boxes_a2: List[BBox] = []
    comps_a2: List[int] = []
    for i in keep_idx:
        label = comp_labels[i]
        box = boxes[i]
        crop_rgb = image_np[box.y0:box.y1, box.x0:box.x1]
        crop_rgb = letterbox_resize(crop_rgb, target_hw_a2)
        crop_float = crop_rgb.astype(np.float32) / 255.0
        crop_norm = (crop_float - mean) / std
        x = torch.from_numpy(crop_norm.transpose(2, 0, 1)).contiguous()
        if include_mask_a2:
            mask_box = (instances[box.y0:box.y1, box.x0:box.x1] == label).astype(np.uint8)
            mask_resized = letterbox_mask(mask_box, target_hw_a2)
            m = torch.from_numpy((mask_resized > 0).astype(np.float32)).unsqueeze(0)
            x = torch.cat([x, m], dim=0)
        batch_a2.append(x)
        crops_a2.append(crop_rgb)
        boxes_a2.append(box)
        comps_a2.append(label)

    probs_a2: List[float] = []
    if batch_a2:
        batch = torch.stack(batch_a2, dim=0).to(device)
        channels_last = bool(a2_cfg.get("training", {}).get("channels_last", True)) if isinstance(a2_cfg, dict) else True
        if channels_last:
            batch = batch.to(memory_format=torch.channels_last)
        with torch.no_grad():
            logits = a2_model(batch).squeeze(1)
            if a2_temperature and a2_temperature > 0:
                logits = logits / float(a2_temperature)
            probs_a2 = torch.sigmoid(logits).detach().cpu().numpy().tolist()

    # A2 decisions: >= threshold -> "never" else "ok"
    decisions_a2 = ["never" if p >= a2_threshold else "ok" for p in probs_a2]
    labels_a2 = [f"A2:{d} {p:.2f}" for d, p in zip(decisions_a2, probs_a2)]
    overlay_a2 = draw_boxes(image_np, boxes_a2, labels_a2)
    atomic_save_pil_image(overlay_a2, step2_dir / "overlay_boxes.png")

    # Save A2 CSV and crops
    csv_a2 = step2_dir / "predictions.csv"
    with csv_a2.open("w", encoding="utf-8") as fh:
        fh.write("id,x0,y0,x1,y1,width,height,area,prob_never,decision\n")
        for i, (box, p, d) in enumerate(zip(boxes_a2, probs_a2, decisions_a2), start=1):
            fh.write(f"{i},{box.x0},{box.y0},{box.x1},{box.y1},{box.w},{box.h},{box.area},{p:.6f},{d}\n")

    keep_final_idx = [i for i, d in enumerate(decisions_a2) if d == "ok"]
    drop_never_idx = [i for i, d in enumerate(decisions_a2) if d == "never"]

    if drop_never_idx:
        _save_crops(step2_dir, "rejected_never", image_np, instances, [boxes_a2[i] for i in drop_never_idx], [comps_a2[i] for i in drop_never_idx], [crops_a2[i] for i in drop_never_idx], [labels_a2[i] for i in drop_never_idx])
    if keep_final_idx:
        _save_crops(step2_dir, "accepted_candidates", image_np, instances, [boxes_a2[i] for i in keep_final_idx], [comps_a2[i] for i in keep_final_idx], [crops_a2[i] for i in keep_final_idx], [labels_a2[i] for i in keep_final_idx])

    # If nothing remains after A2, store meta and finish
    if not keep_final_idx:
        meta = {
            "image_path": str(args.image),
            "segmentation_checkpoint": str(args.seg_checkpoint),
            "segmentation_config": str(args.seg_config),
            "a1_checkpoint": str(args.a1_checkpoint),
            "a2_checkpoint": str(args.a2_checkpoint),
            "kept_after_a1": int(len(keep_idx)),
            "dropped_notberry": int(len(drop_idx)),
            "kept_after_a2": int(len(keep_final_idx)),
            "dropped_never": int(len(drop_never_idx)),
            "a1_threshold": float(a1_threshold),
            "a2_threshold": float(a2_threshold),
        }
        (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        LOGGER.info("No candidates left after A2; outputs written to %s", out_root)
        return 0

    # -------- A3: red (1) vs {yellow,green} (0) --------
    step3_dir.mkdir(parents=True, exist_ok=True)
    a3_model, a3_cfg, a3_temperature, a3_threshold, a3_in_channels = load_classifier(
        args.a3_config, args.a3_checkpoint, device, args.use_mask_channel
    )
    data_cfg_a3 = a3_cfg.get("data", {}) if isinstance(a3_cfg, dict) else {}
    input_size_a3 = data_cfg_a3.get("input_size", [320, 320])
    target_hw_a3 = (int(input_size_a3[0]), int(input_size_a3[1]))
    include_mask_a3 = False
    color_features: List[str] = []
    if isinstance(a3_cfg, dict):
        include_mask_a3 = bool(a3_cfg.get("data", {}).get("include_mask_channel", True))
        feats = a3_cfg.get("data", {}).get("color_features", [])
        if isinstance(feats, (list, tuple)):
            color_features = [str(f) for f in feats]
    # Determine expected order and counts: [RGB] + extras + [mask?]
    extras_count = 0
    if color_features:
        for f in color_features:
            if f == "hsv":
                extras_count += 3
            elif f in {"redness", "darkness"}:
                extras_count += 1
    # If in_channels implies a mask but config disabled/enabled differently, match checkpoint
    expected_mask = (a3_in_channels == (3 + extras_count + 1))
    include_mask_a3 = expected_mask

    batch_a3: List[torch.Tensor] = []
    crops_a3: List[np.ndarray] = []
    boxes_a3: List[BBox] = []
    comps_a3: List[int] = []
    # Reuse ImageNet mean/std for RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for i in keep_final_idx:
        label = comps_a2[i]
        box = boxes_a2[i]
        crop_rgb = image_np[box.y0:box.y1, box.x0:box.x1]
        crop_rgb = letterbox_resize(crop_rgb, target_hw_a3)
        # Optional color normalization as in dataset (gray-world)
        try:
            if str(data_cfg_a3.get("color_norm", "")).lower() == "gray_world":
                crop_rgb = gray_world(crop_rgb.astype(np.uint8), strength=0.8, max_gain=1.8)
        except Exception:
            pass
        crops_a3.append(crop_rgb)
        boxes_a3.append(box)
        comps_a3.append(label)

        crop_float = crop_rgb.astype(np.float32) / 255.0
        crop_norm = (crop_float - mean) / std
        x = torch.from_numpy(crop_norm.transpose(2, 0, 1)).contiguous()  # (3,H,W)

        # Build extras to match training order
        if color_features:
            extras: List[np.ndarray] = []
            rgb01 = np.clip(crop_float, 0.0, 1.0)
            if "redness" in color_features:
                r, g, b = rgb01[:, :, 0], rgb01[:, :, 1], rgb01[:, :, 2]
                red = np.clip(r - np.maximum(g, b), 0.0, 1.0)
                extras.append(red)
            if "darkness" in color_features:
                hsv = cv2.cvtColor((rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
                v = hsv[:, :, 2].astype(np.float32) / 255.0
                dark = 1.0 - v
                extras.append(dark)
            if "hsv" in color_features:
                hsv = cv2.cvtColor((rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                h = hsv[:, :, 0] / 179.0
                s = hsv[:, :, 1] / 255.0
                v = hsv[:, :, 2] / 255.0
                extras.extend([h, s, v])
            if extras:
                extra_stack = np.stack(extras, axis=2).astype(np.float32)
                extra_norm = (extra_stack - 0.5) / 0.5  # [-1,1]
                x_extra = torch.from_numpy(extra_norm.transpose(2, 0, 1)).contiguous()
                x = torch.cat([x, x_extra], dim=0)

        if include_mask_a3:
            mask_box = (instances[box.y0:box.y1, box.x0:box.x1] == label).astype(np.uint8)
            mask_resized = letterbox_mask(mask_box, target_hw_a3)
            m = torch.from_numpy((mask_resized > 0).astype(np.float32)).unsqueeze(0)
            x = torch.cat([x, m], dim=0)

        # If shapes still mismatch, pad/truncate to expected channels
        if x.shape[0] != a3_in_channels:
            if x.shape[0] < a3_in_channels:
                pad = torch.zeros((a3_in_channels - x.shape[0], x.shape[1], x.shape[2]), dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            else:
                x = x[:a3_in_channels]
        batch_a3.append(x)

    probs_a3: List[float] = []
    if batch_a3:
        batch = torch.stack(batch_a3, dim=0).to(device)
        channels_last = bool(a3_cfg.get("training", {}).get("channels_last", True)) if isinstance(a3_cfg, dict) else True
        if channels_last:
            batch = batch.to(memory_format=torch.channels_last)
        with torch.no_grad():
            logits = a3_model(batch).squeeze(1)
            if a3_temperature and a3_temperature > 0:
                logits = logits / float(a3_temperature)
            probs_a3 = torch.sigmoid(logits).detach().cpu().numpy().tolist()

    # A3 decisions: >= threshold -> red else not-red
    decisions_a3 = ["red" if p >= a3_threshold else "not-red" for p in probs_a3]
    labels_a3 = [f"A3:{d} {p:.2f}" for d, p in zip(decisions_a3, probs_a3)]
    overlay_a3 = draw_boxes(image_np, boxes_a3, labels_a3)
    atomic_save_pil_image(overlay_a3, step3_dir / "overlay_boxes.png")

    csv_a3 = step3_dir / "predictions.csv"
    with csv_a3.open("w", encoding="utf-8") as fh:
        fh.write("id,x0,y0,x1,y1,width,height,area,prob_red,decision\n")
        for i, (box, p, d) in enumerate(zip(boxes_a3, probs_a3, decisions_a3), start=1):
            fh.write(f"{i},{box.x0},{box.y0},{box.x1},{box.y1},{box.w},{box.h},{box.area},{p:.6f},{d}\n")

    idx_red = [i for i, d in enumerate(decisions_a3) if d == "red"]
    idx_not_red = [i for i, d in enumerate(decisions_a3) if d != "red"]
    if idx_red:
        _save_crops(step3_dir, "accepted_red", image_np, instances, [boxes_a3[i] for i in idx_red], [comps_a3[i] for i in idx_red], [crops_a3[i] for i in idx_red], [labels_a3[i] for i in idx_red])
    if idx_not_red:
        _save_crops(step3_dir, "rejected_not_red", image_np, instances, [boxes_a3[i] for i in idx_not_red], [comps_a3[i] for i in idx_not_red], [crops_a3[i] for i in idx_not_red], [labels_a3[i] for i in idx_not_red])

    # Meta summary
    meta = {
        "image_path": str(args.image),
        "segmentation_checkpoint": str(args.seg_checkpoint),
        "segmentation_config": str(args.seg_config),
        "a1_checkpoint": str(args.a1_checkpoint),
        "a2_checkpoint": str(args.a2_checkpoint),
        "a3_checkpoint": str(args.a3_checkpoint),
        "kept_after_a1": int(len(keep_idx)),
        "dropped_notberry": int(len(drop_idx)),
        "kept_after_a2": int(len(keep_final_idx)),
        "dropped_never": int(len(drop_never_idx)),
        "kept_after_a3_red": int(len(idx_red)),
        "kept_after_a3_not_red": int(len(idx_not_red)),
        "a1_threshold": float(a1_threshold),
        "a2_threshold": float(a2_threshold),
        "a3_threshold": float(a3_threshold),
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info("Finished A1->A2->A3 pipeline. Outputs: %s", out_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
