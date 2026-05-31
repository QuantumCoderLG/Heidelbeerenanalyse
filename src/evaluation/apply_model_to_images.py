from __future__ import annotations

import argparse
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ..config import load_config
from ..evaluation.postprocessing import apply_postprocessing
from ..training.models import build_model
from ..utils.io_utils import atomic_save_pil_image, load_image
from ..data import rasterize
from ..utils.image_utils import preprocess_image, postprocess_probability
from ..utils.vis_utils import instances_to_overlay

LOGGER = logging.getLogger("apply_models")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a trained segmentation model to JPEG images and save overlay visualisations.",
    )
    parser.add_argument("model", type=str, help="Model filename (e.g. fold_2_best.pt) or full path")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Directory that contains model checkpoints (ignored when model is a path)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Training config to reuse normalisation and postprocessing settings",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/all_images"),
        help="Root directory containing JPEG images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/overlays"),
        help="Directory where overlays will be written",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device identifier (defaults to CUDA if available)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional probability threshold override (falls back to checkpoint / config)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision during inference",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate overlays even if the output file already exists",
    )
    return parser.parse_args(argv)


def list_jpeg_images(root: Path) -> List[Path]:
    root = root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")
    root = root.resolve()
    patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    paths = sorted({p.resolve() for p in paths})
    if not paths:
        raise FileNotFoundError(f"No JPEG images found under {root}")
    return paths





def resolve_checkpoint(model_arg: str, checkpoint_dir: Path) -> Path:
    candidate = Path(model_arg)
    if candidate.exists():
        return candidate
    candidate = checkpoint_dir / model_arg
    if candidate.exists():
        return candidate
    name = Path(model_arg).name
    matches = sorted(checkpoint_dir.rglob(name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        LOGGER.info(
            "Multiple checkpoints named %s found, using %s",
            name,
            matches[0],
        )
        return matches[0]
    raise FileNotFoundError(
        f"Could not locate checkpoint: {model_arg} (searched {checkpoint_dir} recursively)"
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config(args.config)

    image_root = args.image_root.expanduser().resolve()
    output_root = args.output_dir.expanduser()
    checkpoint_dir = args.checkpoint_dir.expanduser()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Using device: %s", device)

    data_cfg = cfg.get("data", {})
    mean = data_cfg.get("image_mean", [0.485, 0.456, 0.406])
    std = data_cfg.get("image_std", [0.229, 0.224, 0.225])
    val_size = data_cfg.get("val_size") or data_cfg.get("train_size")
    target_hw = None
    if val_size and len(val_size) == 2:
        target_hw = (int(val_size[0]), int(val_size[1]))
    keep_ratio = bool(data_cfg.get("keep_ratio", True))

    post_cfg = cfg.get("postproc", {})
    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg, num_classes=1)
    checkpoint_path = resolve_checkpoint(args.model, checkpoint_dir)
    LOGGER.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)

    threshold = float(
        args.threshold
        if args.threshold is not None
        else checkpoint.get("threshold", post_cfg.get("threshold", 0.5))
    )
    LOGGER.info("Using threshold: %.3f", threshold)

    channels_last = bool(cfg.get("train", {}).get("channels_last", False))
    if channels_last:
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    model.eval()

    amp_enabled = bool(args.amp and device.type == "cuda" and hasattr(torch.amp, "autocast"))

    images = list_jpeg_images(image_root)
    LOGGER.info("Found %d images under %s", len(images), image_root)

    model_name = Path(args.model).stem
    out_root = output_root / model_name

    with torch.no_grad():
        for idx, image_path in enumerate(images, start=1):
            rel_path = image_path.relative_to(image_root)
            overlay_name = f"{rel_path.stem}_overlay.png"
            out_path = out_root / rel_path.with_name(overlay_name)
            if out_path.exists() and not args.overwrite:
                LOGGER.info("[%d/%d] Skip existing %s", idx, len(images), out_path)
                continue

            image_np, _, _ = load_image(image_path)
            tensor, meta = preprocess_image(image_np, mean, std, target_hw, keep_ratio)
            inputs = tensor.unsqueeze(0).to(device)
            if channels_last:
                inputs = inputs.to(memory_format=torch.channels_last)

            ctx = torch.amp.autocast("cuda", enabled=True) if amp_enabled else nullcontext()
            with ctx:
                logits = model(inputs)["out"]
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            probs = postprocess_probability(probs, meta)
            _, instances = apply_postprocessing(probs, post_cfg, threshold=threshold)
            overlay = instances_to_overlay(image_np, instances)
            atomic_save_pil_image(overlay, out_path)
            LOGGER.info("[%d/%d] Wrote %s", idx, len(images), out_path)

    LOGGER.info("Finished processing %d images", len(images))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
