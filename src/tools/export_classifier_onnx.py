from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch

from ..training.classifier_models import build_classifier, ClassifierBackbone, MaskWeightedPoolingWrapper


LOGGER = logging.getLogger("export_classifier_onnx")


BACKBONES: Tuple[ClassifierBackbone, ...] = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b5",
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a trained classifier checkpoint (.pt) to ONNX")
    p.add_argument("--input", type=Path, required=True, help="Path to checkpoint .pt (expects {'model': state_dict})")
    p.add_argument("--out", type=Path, required=True, help="Output ONNX path (.onnx)")
    p.add_argument("--backbone", type=str, default=None, help=f"Backbone name (one of: {', '.join(BACKBONES)})")
    p.add_argument("--in-ch", type=int, default=None, help="Input channels (auto-detected if omitted)")
    p.add_argument("--img-size", type=int, nargs=2, default=[320, 320], help="Export dummy size H W (dynamic axes enabled by default)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--static", action="store_true", help="Export with fixed H/W instead of dynamic axes")
    p.add_argument("--dropout", type=float, default=None, help="Override classifier dropout (optional)")
    return p.parse_args(argv)


def _infer_in_channels(state: Dict[str, torch.Tensor]) -> Optional[int]:
    # Try common first conv keys across torchvision models
    candidates = [
        "features.0.0.weight",  # EfficientNet/MobileNetV3
        "conv_stem.weight",      # EfficientNet (older)
        "layer1.0.conv1.weight", # ResNet-like
    ]
    for key in candidates:
        w = state.get(key)
        if isinstance(w, torch.Tensor) and w.ndim == 4:
            return int(w.shape[1])
    # Fallback: search any 4D weight with spatial dims
    for k, w in state.items():
        if isinstance(w, torch.Tensor) and w.ndim == 4 and min(w.shape[-2:]) in (1, 2, 3, 5, 7):
            return int(w.shape[1])
    return None


def _try_build_and_load(backbones: Iterable[ClassifierBackbone], in_ch: int, state: Dict[str, torch.Tensor], dropout: Optional[float]) -> Tuple[torch.nn.Module, str]:
    best: Optional[Tuple[int, int, str, torch.nn.Module]] = None
    for name in backbones:
        model = build_classifier(backbone=name, num_classes=1, in_channels=in_ch, pretrained=False, dropout=dropout)
        try:
            missing = model.load_state_dict(state, strict=False)
            m = len(missing.missing_keys)
            u = len(missing.unexpected_keys)
            score = m + u
        except Exception:
            # Hard mismatch (e.g., layer shapes), treat as worst score
            m = 9999
            score = 9999
        if best is None or score < best[0]:
            best = (score, m, name, model)
            if score == 0:
                break
    if best is None:
        raise RuntimeError("Could not construct classifier model for provided checkpoint")
    return best[3], best[2]


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ckpt_path = args.input.expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    LOGGER.info("Loading checkpoint: %s", ckpt_path)
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        state: Dict[str, torch.Tensor] = raw["model"]  # type: ignore[assignment]
        epoch = int(raw.get("epoch", -1))
        metric = float(raw.get("metric", float("nan")))
    else:
        state = dict(raw)
        epoch = -1
        metric = float("nan")

    in_ch = args.in_ch if args.in_ch is not None else _infer_in_channels(state) or 3
    LOGGER.info("Inferred input channels: %d", in_ch)

    if args.backbone is not None:
        backbone: ClassifierBackbone = args.backbone  # type: ignore[assignment]
        base_model = build_classifier(backbone=backbone, num_classes=1, in_channels=in_ch, pretrained=False, dropout=args.dropout)
        chosen = backbone
    else:
        base_model, chosen = _try_build_and_load(BACKBONES, in_ch, state, args.dropout)

    # Detect mask-weighted wrapper checkpoints (keys prefixed with "base.")
    use_wrapper = any(k.startswith("base.") for k in state.keys())
    if use_wrapper:
        LOGGER.info("Detected mask-weighted pooling wrapper state dict -> exporting wrapped model")
        model = MaskWeightedPoolingWrapper(base_model, has_mask_channel=True)
    else:
        model = base_model

    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        LOGGER.warning("Missing keys: %s", ", ".join(sorted(missing.missing_keys)))
    if missing.unexpected_keys:
        LOGGER.warning("Unexpected keys: %s", ", ".join(sorted(missing.unexpected_keys)))
    model.eval()
    LOGGER.info("Chosen backbone: %s", chosen)

    H, W = int(args.img_size[0]), int(args.img_size[1])
    dummy = torch.zeros(1, in_ch, H, W, dtype=torch.float32)
    out_path = args.out.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None if args.static else {"input": {2: "height", 3: "width"}, "logits": {}}  # logits is (N,1)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=int(args.opset),
        do_constant_folding=True,
    )
    LOGGER.info("Wrote ONNX: %s", out_path)

    meta = {
        "source_checkpoint": str(ckpt_path),
        "epoch": epoch,
        "metric": metric,
        "backbone": chosen,
        "in_channels": in_ch,
        "img_size": [H, W],
        "onnx": str(out_path),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    LOGGER.info("Wrote metadata: %s", meta_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
