from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from ..config import load_config
from ..data import BlueberrySegmentationDataset, build_transforms, create_dataloader
from .metrics import MetricsConfig, aggregate_metrics, evaluate_image
from ..training.models import build_model
from .postprocessing import apply_postprocessing, count_guided_threshold

LOGGER = logging.getLogger("eval")


def configure_backends(train_cfg: Mapping[str, Any]) -> None:
    backend_cfg = train_cfg.get("backend", {}) if isinstance(train_cfg, Mapping) else {}

    cudnn_benchmark = bool(backend_cfg.get("cudnn_benchmark", False))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = cudnn_benchmark

    allow_tf32 = bool(backend_cfg.get("allow_tf32", False))
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32

    matmul_precision = backend_cfg.get("matmul_precision")
    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    if callable(set_precision) and matmul_precision:
        try:
            set_precision(str(matmul_precision))
        except Exception as err:  # pragma: no cover - defensive
            LOGGER.warning("Failed to set float32 matmul precision (%s): %s", matmul_precision, err)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepLabV3+ blueberry segmentation")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--config", type=Path, default=None, help="Optional config override")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate")
    parser.add_argument("--override", metavar="KEY=VALUE", action="append", default=[], help="Config override entries")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size")
    parser.add_argument("--count-tune", action="store_true", help="Enable count-guided threshold tuning per image")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory to save instance masks")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (default auto)")
    parser.add_argument("--amp", action="store_true", help="Use AMP during inference")
    return parser.parse_args(argv)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.override)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_cfg = cfg.get("train", {})
    configure_backends(train_cfg)

    data_cfg = cfg.get("data", {})
    root = Path(data_cfg.get("root", "data/processed"))
    split = args.split or data_cfg.get("val_split", "val")
    transforms = build_transforms(
        data_cfg.get("image_mean", [0.485, 0.456, 0.406]),
        data_cfg.get("image_std", [0.229, 0.224, 0.225]),
        augment=False,
        size=tuple(data_cfg.get("val_size", [])) if data_cfg.get("val_size") else None,
        keep_ratio=bool(data_cfg.get("keep_ratio", True)),
    )
    dataset = BlueberrySegmentationDataset(
        root=root,
        split=split,
        transforms=transforms,
        cache_images=bool(data_cfg.get("cache_images", False)),
    )
    prefetch_cfg = data_cfg.get("prefetch_factor")
    prefetch_factor = int(prefetch_cfg) if prefetch_cfg is not None else None
    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size or int(data_cfg.get("val_batch_size", data_cfg.get("batch_size", 4))),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=False,
        drop_last=False,
        prefetch_factor=prefetch_factor,
    )

    model = build_model(cfg["model"], num_classes=1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", checkpoint))
    base_threshold = float(checkpoint.get("threshold", cfg.get("postproc", {}).get("threshold", 0.5)))
    channels_last = bool(train_cfg.get("channels_last", False))
    if channels_last:
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    compile_cfg = train_cfg.get("torch_compile", {}) if isinstance(train_cfg, Mapping) else {}
    if compile_cfg.get("enabled", False):
        if hasattr(torch, "compile"):
            compile_mode = compile_cfg.get("mode", "reduce-overhead")
            model = torch.compile(model, mode=compile_mode)
        else:  # pragma: no cover - defensive
            LOGGER.warning("torch.compile requested but not available; skipping compilation")
    model.eval()

    if args.save_dir is not None:
        ensure_dir(args.save_dir)

    post_cfg = cfg.get("postproc", {})
    metrics_cfg = MetricsConfig(
        count_target=int(cfg.get("metrics", {}).get("count_target", 25)),
        merge_iou_threshold=float(cfg.get("metrics", {}).get("merge_iou_threshold", 0.1)),
        miss_iou_threshold=float(cfg.get("metrics", {}).get("miss_iou_threshold", 0.2)),
    )

    results = []
    per_image_metrics = []
    per_image_thresholds = []
    timings = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    amp_enabled = bool(args.amp or train_cfg.get("amp", True))
    use_channels_last = bool(train_cfg.get("channels_last", False))
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, non_blocking=True)
            if use_channels_last:
                images = images.to(memory_format=torch.channels_last)
            instance_masks = batch["instance_mask"].cpu().numpy()
            metas = batch["meta"]
            start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)["out"]
                probs = torch.sigmoid(logits).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            per_image_time = elapsed / len(images)
            timings.extend([per_image_time] * len(images))
            for idx in range(len(probs)):
                prob = probs[idx, 0]
                gt = instance_masks[idx]
                meta = metas[idx]
                thr = base_threshold
                if args.count_tune:
                    thr = count_guided_threshold(prob, base_threshold, metrics_cfg.count_target, post_cfg)
                _, instances = apply_postprocessing(prob, post_cfg, threshold=thr)
                metric_values = evaluate_image(instances, gt, metrics_cfg)
                metric_values.update({
                    "image_id": meta["image_id"],
                    "image_path": meta["image_path"],
                    "threshold": thr,
                    "pred_count": float(len(np.unique(instances)) - 1),
                })
                per_image_metrics.append(metric_values)
                per_image_thresholds.append(thr)
                if args.save_dir is not None:
                    out_path = args.save_dir / f"{Path(meta['image_path']).stem}_instances.npy"
                    np.save(out_path, instances.astype(np.int32))
                results.append(metric_values)
    aggregated = aggregate_metrics(per_image_metrics)
    aggregated["count_accuracy_exact"] = float(np.mean([1.0 if abs(m["pred_count"] - metrics_cfg.count_target) < 0.5 else 0.0 for m in per_image_metrics]))
    aggregated["threshold_mean"] = float(np.mean(per_image_thresholds)) if per_image_thresholds else base_threshold
    aggregated["inference_time_ms"] = float(np.mean(timings) * 1000.0) if timings else 0.0
    aggregated["gpu_mem_mb"] = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda"
        else 0.0
    )

    output = {
        "split": split,
        "checkpoint": str(args.checkpoint),
        "base_threshold": base_threshold,
        "count_tune": args.count_tune,
        "metrics": aggregated,
        "per_image": results,
    }
    out_path = Path(cfg.get("logging", {}).get("metrics_json", "outputs/logs/eval.json"))
    ensure_dir(out_path)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    LOGGER.info("Evaluation complete. Metrics saved to %s", out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
