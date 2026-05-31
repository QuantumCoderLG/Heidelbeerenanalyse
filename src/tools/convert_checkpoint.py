from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch

from ..config import load_config
from ..training.models import build_model

LOGGER = logging.getLogger("convert_checkpoint")


def _default_hw(cfg: dict) -> Tuple[int, int]:
    data = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    size = data.get("val_size") or data.get("train_size") or [1024, 1024]
    try:
        h, w = int(size[0]), int(size[1])
    except Exception:
        h, w = 1024, 1024
    return h, w


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a .pt checkpoint to other formats (safetensors, ONNX)",
    )
    p.add_argument("--input", type=Path, default=Path("Kanditaten/fold_1_best.pt"), help="Path to .pt checkpoint")
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="Training config path")
    p.add_argument("--out-dir", type=Path, default=Path("Kanditaten/converted"), help="Output directory for exports")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--no-onnx", dest="export_onnx", action="store_false", help="Disable ONNX export")
    p.add_argument("--no-safetensors", dest="export_safetensors", action="store_false", help="Disable safetensors export")
    p.add_argument("--dynamic", dest="dynamic", action="store_true", help="Use dynamic H/W for ONNX")
    p.add_argument("--static", dest="dynamic", action="store_false", help="Use fixed H/W for ONNX")
    p.set_defaults(export_onnx=True, export_safetensors=True, dynamic=True)
    return p.parse_args(argv)


class OutOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ckpt_path = args.input.expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg, num_classes=1)
    model.eval()

    LOGGER.info("Loading checkpoint: %s", ckpt_path)
    checkpoint: Dict = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        LOGGER.warning("Missing keys: %s", ", ".join(sorted(missing.missing_keys)))
    if missing.unexpected_keys:
        LOGGER.warning("Unexpected keys: %s", ", ".join(sorted(missing.unexpected_keys)))

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "source_checkpoint": str(ckpt_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "score": float(checkpoint.get("score", float("nan"))),
        "threshold": float(checkpoint.get("threshold", cfg.get("postproc", {}).get("threshold", 0.5))),
        "model": {
            "name": str(model_cfg.get("name", "deeplabv3plus")),
            "output_stride": int(model_cfg.get("output_stride", 16)),
            "aux_loss": bool(model_cfg.get("aux_loss", False)),
        },
    }

    # Export safetensors
    if args.export_safetensors:
        try:
            from safetensors.torch import save_file as save_safetensors

            weights_path = out_dir / (ckpt_path.stem + ".safetensors")
            state_dict = model.state_dict()
            # Ensure tensors are on CPU and contiguous
            state_dict = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
            save_safetensors(state_dict, str(weights_path))
            LOGGER.info("Wrote safetensors weights: %s", weights_path)
            meta["safetensors"] = str(weights_path)
        except Exception as err:  # pragma: no cover - optional dependency
            LOGGER.error("Safetensors export failed: %s", err)

    # Export ONNX
    if args.export_onnx:
        h, w = _default_hw(cfg)
        dummy = torch.zeros(1, 3, h, w, dtype=torch.float32)
        wrapper = OutOnly(model).eval()
        onnx_path = out_dir / (ckpt_path.stem + ".onnx")
        dynamic_axes = None
        if args.dynamic:
            dynamic_axes = {"input": {2: "height", 3: "width"}, "out": {2: "height", 3: "width"}}
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["out"],
            dynamic_axes=dynamic_axes,
            opset_version=int(args.opset),
            do_constant_folding=True,
        )
        LOGGER.info("Wrote ONNX model: %s", onnx_path)
        meta["onnx"] = str(onnx_path)

    # Write metadata
    meta_path = out_dir / (ckpt_path.stem + "_meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    LOGGER.info("Wrote metadata: %s", meta_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

