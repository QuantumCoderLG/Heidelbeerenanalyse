from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ..config import load_config
from ..data import (
    BlueberrySegmentationDataset,
    build_transforms,
    create_dataloader,
    make_subsets,
)
from .losses import build_loss
from ..evaluation.metrics import MetricsConfig, aggregate_metrics, evaluate_image
from .models import build_model
from .models import build_model
from ..evaluation.postprocessing import apply_postprocessing, search_optimal_threshold
from ..utils.training_utils import (
    set_seed,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    ensure_model_real_float,
)


LOGGER = logging.getLogger("train")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ for blueberry instance segmentation")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML configuration file")
    parser.add_argument(
        "--override",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Override configuration entries (dot notation)",
    )
    parser.add_argument(
        "--freeze-blocks",
        nargs="+",
        default=None,
        help="Override model.freeze_blocks list",
    )
    parser.add_argument("--lr-find", dest="lr_find", action="store_true", help="Force LR finder execution")
    parser.add_argument("--no-lr-find", dest="lr_find", action="store_false", help="Disable LR finder")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Force AMP usage")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP usage")
    parser.add_argument("--accum-steps", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fewer iterations)")
    parser.set_defaults(lr_find=None, amp=None)
    return parser.parse_args(argv)





def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)




def load_model_state(model: nn.Module, checkpoint: Mapping[str, Any] | nn.Module, *, strict: bool) -> None:
    """Load weights from checkpoint or raw state dict into model."""

    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a valid state_dict mapping")
    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
    if missing:
        LOGGER.warning("Missing keys while loading checkpoint: %s", ", ".join(sorted(missing)))
    if unexpected:
        LOGGER.warning("Unexpected keys while loading checkpoint: %s", ", ".join(sorted(unexpected)))


def create_folds(n_items: int, k: int, shuffle: bool, seed: int) -> List[Tuple[List[int], List[int]]]:
    indices = list(range(n_items))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    fold_sizes = [n_items // k] * k
    for i in range(n_items % k):
        fold_sizes[i] += 1
    folds: List[Tuple[List[int], List[int]]] = []
    current = 0
    for fold_size in fold_sizes:
        val_ids = indices[current : current + fold_size]
        train_ids = indices[:current] + indices[current + fold_size :]
        folds.append((train_ids, val_ids))
        current += fold_size
    return folds


def _has_c_compiler() -> bool:
    candidates = [os.environ.get("CC"), os.environ.get("CXX"), "cc", "gcc", "clang", "cl"]
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return True
    return False


def maybe_compile_model(model: nn.Module, compile_cfg: Mapping[str, Any]) -> nn.Module:
    if not compile_cfg.get("enabled", False):
        return model
    if not hasattr(torch, "compile"):
        LOGGER.warning("torch.compile requested but not available; skipping compilation")
        return model
    if not _has_c_compiler():
        LOGGER.warning(
            "torch.compile requested but no suitable C compiler found (set CC or install gcc/clang). Skipping compilation."
        )
        return model
    compile_mode = compile_cfg.get("mode", "reduce-overhead")
    try:
        return torch.compile(model, mode=compile_mode)
    except Exception as err:  # pragma: no cover - defensive fallback
        LOGGER.warning("torch.compile failed (%s); running in eager mode instead", err)
        return model


def configure_logging(config: dict) -> None:
    level = getattr(logging, config.get("logging", {}).get("level", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


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

def lr_find(
    model: nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
    amp: bool,
) -> Tuple[float, List[Tuple[float, float]]]:
    ensure_model_real_float(model)
    sweep_cfg = cfg.get("train", {}).get("lr_find", {})
    min_lr = float(sweep_cfg.get("min_lr", 1e-6))
    max_lr = float(sweep_cfg.get("max_lr", 1e-2))
    num_iters = int(sweep_cfg.get("num_iters", 100))
    warmup = int(sweep_cfg.get("warmup_iters", 5))
    try:
        total_batches = len(dataloader)  # type: ignore[arg-type]
    except TypeError:  # pragma: no cover - dataloader without __len__
        total_batches = None
    if total_batches == 0:
        fallback_lr = float(cfg.get("optimizer", {}).get("lr", max_lr))
        LOGGER.warning(
            "LR finder skipped: dataloader returned no batches (perhaps dataset smaller than batch size with drop_last=True). Using lr=%.6f",
            fallback_lr,
        )
        return fallback_lr, []
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = build_loss(cfg["loss"])

    original_state = {
        "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
    }

    lr_values = np.logspace(math.log10(min_lr), math.log10(max_lr), num_iters)
    losses: List[float] = []
    best_lr = min_lr
    best_loss = float("inf")

    stream = iter(dataloader)
    model.train()
    use_channels_last = bool(cfg.get("train", {}).get("channels_last", False))
    for step in range(num_iters):
        try:
            batch = next(stream)
        except StopIteration:
            stream = iter(dataloader)
            batch = next(stream)
        images = batch["image"].to(device=device, non_blocking=True)
        if use_channels_last:
            images = images.to(memory_format=torch.channels_last)
        targets = batch["mask"].to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        lr = float(lr_values[step])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        with torch.amp.autocast("cuda", enabled=amp):
            outputs = model(images)["out"]
            loss = loss_fn(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step >= warmup and loss_value < best_loss:
            best_loss = loss_value
            best_lr = lr
    # restore state
    model.load_state_dict(original_state["model"])
    optimizer.load_state_dict(original_state["optimizer"])
    return best_lr, list(zip(lr_values.tolist(), losses))





def train_fold(
    fold_idx: int,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    cfg: dict,
    device: torch.device,
    writer: Optional[SummaryWriter],
    csv_writer: Optional[csv.writer],
    csv_file,
    metrics_cfg: MetricsConfig,
    checkpoint_dir: Path,
    debug: bool = False,
    scaler_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    loss_fn = build_loss(cfg["loss"])
    amp_enabled = bool(cfg["train"].get("amp", True))
    use_channels_last = bool(cfg["train"].get("channels_last", False))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if scaler_state:
        try:
            scaler.load_state_dict(dict(scaler_state))
        except Exception as err:  # pragma: no cover - defensive
            LOGGER.warning("Failed to restore GradScaler state: %s", err)
    max_epochs = int(cfg["train"].get("max_epochs", 100))
    grad_accum = max(1, int(cfg["train"].get("gradient_accumulation_steps", 1)))
    clip_grad = cfg["train"].get("clip_grad_norm")
    val_interval = max(1, int(cfg["train"].get("val_interval", 1)))
    early_cfg = cfg["train"].get("early_stopping", {})
    patience = int(early_cfg.get("patience", 10))
    min_delta = float(early_cfg.get("min_delta", 1e-4))

    total_batches = len(train_loader)
    effective_total = total_batches if not debug else min(total_batches, 5)
    global_step = 0
    best_metric = -float("inf")
    best_epoch = -1
    patience_counter = 0
    best_threshold = float(cfg["postproc"].get("threshold", 0.5))

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            num_batches += 1
            images = batch["image"].to(device=device, non_blocking=True)
            if use_channels_last:
                images = images.to(memory_format=torch.channels_last)
            targets = batch["mask"].to(device=device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(images)
                logits = outputs["out"]
                base_loss = loss_fn(logits, targets)
                if cfg["model"].get("aux_loss", False) and "aux" in outputs:
                    base_loss = base_loss + 0.4 * loss_fn(outputs["aux"], targets)
            loss = base_loss / grad_accum
            scaler.scale(loss).backward()
            is_last = num_batches == effective_total
            should_step = (num_batches % grad_accum == 0) or is_last
            if should_step:
                if clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
            epoch_loss += float(base_loss.detach().cpu())
            global_step += 1
            if writer is not None and global_step % cfg["train"].get("log_interval", 10) == 0:
                writer.add_scalar(f"fold_{fold_idx}/train_loss", float(base_loss.detach().cpu()), global_step)
            if debug and num_batches >= effective_total:
                break
        epoch_loss = epoch_loss / max(1, num_batches)

        if csv_writer is not None:
            csv_writer.writerow([fold_idx, epoch, "train_loss", epoch_loss])
            csv_file.flush()

        if epoch % val_interval == 0:
            val_metrics, threshold, history = evaluate_model(
                model=model,
                dataloader=val_loader,
                cfg=cfg,
                metrics_cfg=metrics_cfg,
                device=device,
            )
            best_threshold = threshold
            if writer is not None:
                for key, value in val_metrics.items():
                    writer.add_scalar(f"fold_{fold_idx}/val_{key}", value, epoch)
                writer.add_scalar(f"fold_{fold_idx}/val_threshold", threshold, epoch)
            if csv_writer is not None:
                for key, value in val_metrics.items():
                    csv_writer.writerow([fold_idx, epoch, f"val_{key}", value])
                csv_file.flush()
            score = float(val_metrics.get("instance_iou_median", 0.0))
            if score > best_metric + min_delta:
                best_metric = score
                best_epoch = epoch
                patience_counter = 0
                best_path = checkpoint_dir / f"fold_{fold_idx}_best.pt"
                checkpoint_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "score": score,
                    "threshold": threshold,
                    "scaler": scaler.state_dict(),
                }
                if scheduler is not None:
                    checkpoint_state["scheduler"] = scheduler.state_dict()
                save_checkpoint(checkpoint_state, best_path)
            else:
                patience_counter += 1

        last_path = checkpoint_dir / f"fold_{fold_idx}_last.pt"
        last_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "threshold": best_threshold,
            "scaler": scaler.state_dict(),
        }
        if scheduler is not None:
            last_state["scheduler"] = scheduler.state_dict()
        save_checkpoint(last_state, last_path)

        LOGGER.info(
            "Fold %s | Epoch %s | Train loss %.4f | Best metric %.4f @ epoch %s",
            fold_idx,
            epoch,
            epoch_loss,
            best_metric,
            best_epoch,
        )
        if patience_counter >= patience:
            LOGGER.info("Early stopping triggered on fold %s after %s epochs", fold_idx, epoch)
            break

    return {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_threshold": best_threshold,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cfg: dict,
    metrics_cfg: MetricsConfig,
    device: torch.device,
) -> Tuple[Dict[str, float], float, List[Tuple[float, float]]]:
    model.eval()
    post_cfg = cfg.get("postproc", {})
    val_metrics: List[Dict[str, float]] = []
    prob_maps: List[np.ndarray] = []
    gt_instances: List[np.ndarray] = []
    total_loss = 0.0
    loss_fn = build_loss(cfg["loss"])
    timings: List[float] = []
    use_channels_last = bool(cfg.get("train", {}).get("channels_last", False))
    amp_enabled = bool(cfg.get("train", {}).get("amp", True))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device=device, non_blocking=True)
            if use_channels_last:
                images = images.to(memory_format=torch.channels_last)
            targets = batch["mask"].to(device=device, non_blocking=True)
            instances = batch["instance_mask"].cpu().numpy()
            start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            timings.extend([elapsed / len(images)] * len(images))
            logits = outputs["out"]
            loss = loss_fn(logits, targets)
            total_loss += float(loss.detach().cpu())
            probs = torch.sigmoid(logits).cpu().numpy()
            for i in range(len(probs)):
                prob = probs[i, 0]
                _, inst = apply_postprocessing(prob, post_cfg)
                val_metrics.append(evaluate_image(inst, instances[i], metrics_cfg))
                prob_maps.append(prob)
                gt_instances.append(instances[i])
    baseline_metrics = aggregate_metrics(val_metrics)
    baseline_metrics["val_loss"] = total_loss / max(1, len(dataloader))
    baseline_metrics["inference_time_ms"] = float(np.mean(timings) * 1000.0) if timings else 0.0
    baseline_metrics["gpu_mem_mb"] = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda"
        else 0.0
    )
    threshold, history = search_optimal_threshold(prob_maps, gt_instances, post_cfg, cfg.get("metrics", {}))
    if threshold != float(post_cfg.get("threshold", 0.5)):
        tuned_metrics = []
        for prob, gt in zip(prob_maps, gt_instances):
            _, inst = apply_postprocessing(prob, post_cfg, threshold=threshold)
            tuned_metrics.append(evaluate_image(inst, gt, metrics_cfg))
        aggregated = aggregate_metrics(tuned_metrics)
    else:
        aggregated = baseline_metrics
    aggregated.update({
        "baseline_threshold": float(post_cfg.get("threshold", 0.5)),
        "val_loss": baseline_metrics.get("val_loss", baseline_metrics.get("train_loss", 0.0)),
        "inference_time_ms": baseline_metrics.get("inference_time_ms", 0.0),
        "gpu_mem_mb": baseline_metrics.get("gpu_mem_mb", 0.0),
        "optimal_threshold": threshold,
    })
    return aggregated, threshold, history











def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.override)
    if args.freeze_blocks is not None:
        cfg.setdefault("model", {})["freeze_blocks"] = args.freeze_blocks
    if args.amp is not None:
        cfg.setdefault("train", {})["amp"] = bool(args.amp)
    if args.accum_steps is not None:
        cfg.setdefault("train", {})["gradient_accumulation_steps"] = max(1, args.accum_steps)
    if args.lr_find is not None:
        cfg.setdefault("train", {}).setdefault("lr_find", {})["enabled"] = bool(args.lr_find)

    configure_logging(cfg)
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    data_cfg = cfg.get("data", {})
    image_mean = data_cfg.get("image_mean", [0.485, 0.456, 0.406])
    image_std = data_cfg.get("image_std", [0.229, 0.224, 0.225])
    root = Path(data_cfg.get("root", "data/processed"))
    train_split = data_cfg.get("train_split", "train")
    val_split = data_cfg.get("val_split", "val")
    cache_images = bool(data_cfg.get("cache_images", False))
    prefetch_cfg = data_cfg.get("prefetch_factor")
    prefetch_factor = int(prefetch_cfg) if prefetch_cfg is not None else None

    train_transforms = build_transforms(
        image_mean,
        image_std,
        augment=True,
        size=tuple(data_cfg.get("train_size", [])) if data_cfg.get("train_size") else None,
        keep_ratio=bool(data_cfg.get("keep_ratio", True)),
    )
    eval_transforms = build_transforms(
        image_mean,
        image_std,
        augment=False,
        size=tuple(data_cfg.get("val_size", [])) if data_cfg.get("val_size") else None,
        keep_ratio=bool(data_cfg.get("keep_ratio", True)),
    )

    train_cfg = cfg.get("train", {})
    configure_backends(train_cfg)
    resume_path = train_cfg.get("resume_path")
    resume_checkpoint: Dict[str, Any] | None = None
    resume_strict = bool(train_cfg.get("resume_strict", False))
    resume_optimizer = bool(train_cfg.get("resume_optimizer", False))
    resume_scheduler = bool(train_cfg.get("resume_scheduler", False))
    resume_scaler_state: Dict[str, Any] | None = None
    if resume_path:
        resume_path = Path(resume_path).expanduser()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        LOGGER.info("Loading checkpoint weights from %s", resume_path)
        raw_checkpoint = torch.load(resume_path, map_location="cpu")
        if isinstance(raw_checkpoint, Mapping):
            resume_checkpoint = dict(raw_checkpoint)
        else:
            resume_checkpoint = {"model": raw_checkpoint}
        if resume_optimizer and "optimizer" not in resume_checkpoint:
            LOGGER.warning("resume_optimizer=True but checkpoint has no optimizer state")
            resume_optimizer = False
        if resume_scheduler and "scheduler" not in resume_checkpoint:
            LOGGER.warning("resume_scheduler=True but checkpoint has no scheduler state")
            resume_scheduler = False
        if "scaler" in resume_checkpoint:
            resume_scaler_state = resume_checkpoint["scaler"]  # type: ignore[assignment]

    train_dataset = BlueberrySegmentationDataset(
        root=root,
        split=train_split,
        transforms=train_transforms,
        cache_images=cache_images,
    )
    train_dataset_eval = BlueberrySegmentationDataset(
        root=root,
        split=train_split,
        transforms=eval_transforms,
        cache_images=cache_images,
    )
    val_dataset: BlueberrySegmentationDataset
    val_dir = root / val_split if val_split else None
    if val_dir is not None and val_dir.exists():
        val_dataset = BlueberrySegmentationDataset(
            root=root,
            split=val_split,
            transforms=eval_transforms,
            cache_images=cache_images,
        )
    else:
        LOGGER.warning("Validation split '%s' not found. Using training split for evaluation.", val_split)
        val_dataset = train_dataset_eval

    csv_log_path = Path(cfg.get("logging", {}).get("csv_log", "outputs/logs/training.csv"))
    ensure_dir(csv_log_path)
    csv_file = csv_log_path.open("a", newline="")
    csv_writer = csv.writer(csv_file)

    tb_dir = Path(cfg.get("logging", {}).get("tensorboard_dir", "outputs/tensorboard"))
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    model = build_model(cfg["model"], num_classes=1)
    if resume_checkpoint is not None:
        load_model_state(model, resume_checkpoint, strict=resume_strict)
    ensure_model_real_float(model)
    channels_last = bool(train_cfg.get("channels_last", False))
    if channels_last:
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    compile_cfg = train_cfg.get("torch_compile", {}) if isinstance(train_cfg, Mapping) else {}
    model = maybe_compile_model(model, compile_cfg)

    optimizer = build_optimizer(model, cfg)
    if resume_checkpoint is not None and resume_optimizer:
        try:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        except Exception as err:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load optimizer state: %s", err)
    # Build data loaders first to compute total steps for scheduler

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=int(data_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        drop_last=bool(data_cfg.get("drop_last", False)),
        prefetch_factor=prefetch_factor,
    )
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=int(data_cfg.get("val_batch_size", data_cfg.get("batch_size", 4))),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=False,
        drop_last=False,
        prefetch_factor=prefetch_factor,
    )

    # Now that loaders exist, build a safe scheduler based on total optimizer steps
    steps_per_epoch = max(1, int(np.ceil(len(train_loader) / max(1, int(cfg["train"].get("gradient_accumulation_steps", 1))))))
    total_steps = steps_per_epoch * int(cfg["train"].get("max_epochs", 100))
    scheduler = build_scheduler(optimizer, cfg, max_epochs=int(cfg["train"].get("max_epochs", 100)), steps_per_epoch=steps_per_epoch)
    if scheduler is not None and resume_checkpoint is not None and resume_scheduler:
        try:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        except Exception as err:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load scheduler state: %s", err)

    # Determine CV settings before optionally running a global LR finder
    cv_cfg = cfg.get("cv", {})
    cv_enabled = bool(cv_cfg.get("enabled", False))

    # If CV is disabled, we may run a single LR finder here.
    # If CV is enabled, defer LR finding to per-fold (below).
    if (not cv_enabled) and cfg.get("train", {}).get("lr_find", {}).get("enabled", False):
        LOGGER.info("Running LR finder (single-run)...")
        best_lr, history = lr_find(model, train_loader, optimizer, device, cfg, cfg["train"].get("amp", True))
        base_lr = float(cfg.get("optimizer", {}).get("lr", best_lr))
        if best_lr > base_lr:
            LOGGER.warning("LR finder suggested %.6f, clamped to base lr %.6f", best_lr, base_lr)
            best_lr = base_lr
        for group in optimizer.param_groups:
            group["lr"] = best_lr
        LOGGER.info("LR finder selected lr=%.6f", best_lr)
        if writer:
            for idx, (lr, loss) in enumerate(history):
                writer.add_scalar("lr_find/loss", loss, idx)
                writer.add_scalar("lr_find/lr", lr, idx)
    if cv_enabled and resume_optimizer:
        LOGGER.warning("resume_optimizer is ignored when cross-validation is enabled")
        resume_optimizer = False
    if cv_enabled and resume_scheduler:
        LOGGER.warning("resume_scheduler is ignored when cross-validation is enabled")
        resume_scheduler = False
    results: List[Dict[str, float]] = []
    metrics_cfg = MetricsConfig(
        count_target=int(cfg.get("metrics", {}).get("count_target", 25)),
        merge_iou_threshold=float(cfg.get("metrics", {}).get("merge_iou_threshold", 0.1)),
        miss_iou_threshold=float(cfg.get("metrics", {}).get("miss_iou_threshold", 0.2)),
    )
    checkpoint_dir = Path(cfg.get("train", {}).get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if cv_enabled:
        LOGGER.info("Running %s-fold cross-validation", int(cv_cfg.get("num_folds", 5)))
        folds = create_folds(len(train_dataset), int(cv_cfg.get("num_folds", 5)), bool(cv_cfg.get("shuffle", True)), int(cv_cfg.get("seed", seed)))
        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
            fold_train = make_subsets(train_dataset, train_idx)
            fold_val = make_subsets(train_dataset_eval, val_idx)
            fold_train_loader = create_dataloader(
                dataset=fold_train,
                batch_size=int(data_cfg.get("batch_size", 4)),
                shuffle=True,
                num_workers=int(data_cfg.get("num_workers", 4)),
                pin_memory=bool(data_cfg.get("pin_memory", True)),
                persistent_workers=bool(data_cfg.get("persistent_workers", True)),
                drop_last=bool(data_cfg.get("drop_last", False)),
                prefetch_factor=prefetch_factor,
            )
            fold_val_loader = create_dataloader(
                dataset=fold_val,
                batch_size=int(data_cfg.get("val_batch_size", data_cfg.get("batch_size", 4))),
                shuffle=False,
                num_workers=int(data_cfg.get("num_workers", 4)),
                pin_memory=bool(data_cfg.get("pin_memory", True)),
                persistent_workers=False,
                drop_last=False,
                prefetch_factor=prefetch_factor,
            )
            model.apply(lambda m: setattr(m, "training", True))
            model = build_model(cfg["model"], num_classes=1)
            if resume_checkpoint is not None:
                load_model_state(model, resume_checkpoint, strict=resume_strict)
            ensure_model_real_float(model)
            if channels_last:
                model = model.to(device=device, memory_format=torch.channels_last)
            else:
                model = model.to(device)

            model = maybe_compile_model(model, compile_cfg)
            optimizer = build_optimizer(cfg["optimizer"], model)
            if resume_checkpoint is not None and resume_optimizer:
                try:
                    optimizer.load_state_dict(resume_checkpoint["optimizer"])
                except Exception as err:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to load optimizer state: %s", err)
            steps_per_epoch = max(1, int(np.ceil(len(fold_train_loader) / max(1, int(cfg["train"].get("gradient_accumulation_steps", 1))))))
            total_steps = steps_per_epoch * int(cfg["train"].get("max_epochs", 100))
            scheduler = build_scheduler(cfg.get("scheduler", {}), optimizer, total_steps=total_steps)
            if scheduler is not None and resume_checkpoint is not None and resume_scheduler:
                try:
                    scheduler.load_state_dict(resume_checkpoint["scheduler"])
                except Exception as err:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to load scheduler state: %s", err)

            # Per-fold LR finder (if enabled). This runs after optimizer/scheduler are built.
            # It mirrors the single-run logic but tags TensorBoard with the fold index.
            if cfg.get("train", {}).get("lr_find", {}).get("enabled", False):
                LOGGER.info("Fold %s: Running LR finder...", fold_idx)
                best_lr, history = lr_find(
                    model, fold_train_loader, optimizer, device, cfg, cfg["train"].get("amp", True)
                )
                base_lr = float(cfg.get("optimizer", {}).get("lr", best_lr))
                if best_lr > base_lr:
                    LOGGER.warning(
                        "Fold %s: LR finder suggested %.6f, clamped to base lr %.6f",
                        fold_idx,
                        best_lr,
                        base_lr,
                    )
                    best_lr = base_lr
                for group in optimizer.param_groups:
                    group["lr"] = best_lr
                LOGGER.info("Fold %s: LR finder selected lr=%.6f", fold_idx, best_lr)
                if writer:
                    for idx, (lr, loss) in enumerate(history):
                        writer.add_scalar(f"fold_{fold_idx}/lr_find/loss", loss, idx)
                        writer.add_scalar(f"fold_{fold_idx}/lr_find/lr", lr, idx)
            fold_result = train_fold(
                fold_idx=fold_idx,
                model=model,
                train_loader=fold_train_loader,
                val_loader=fold_val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                device=device,
                writer=writer,
                csv_writer=csv_writer,
                csv_file=csv_file,
                metrics_cfg=metrics_cfg,
                checkpoint_dir=checkpoint_dir / f"fold_{fold_idx}",
                debug=args.debug,
                scaler_state=resume_scaler_state if resume_optimizer else None,
            )
            results.append(fold_result)
    else:
        fold_result = train_fold(
            fold_idx=0,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            device=device,
            writer=writer,
            csv_writer=csv_writer,
            csv_file=csv_file,
            metrics_cfg=metrics_cfg,
            checkpoint_dir=checkpoint_dir,
            debug=args.debug,
            scaler_state=resume_scaler_state if resume_optimizer else None,
        )
        results.append(fold_result)

    csv_file.close()
    writer.flush(); writer.close()

    metrics_json = Path(cfg.get("logging", {}).get("metrics_json", "outputs/logs/metrics.json"))
    ensure_dir(metrics_json)
    metrics_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Training completed. Results saved to %s", metrics_json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
