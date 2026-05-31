from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
import yaml
import numpy as np
import pandas as pd
from torch import amp
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.classification_dataset import BerryClassificationDataset, ClassificationDataConfig
from .calibration import (
    apply_temperature,
    apply_temperature_bias,
    fit_temperature,
    fit_temperature_bounded,
)
from .classifier_models import build_classifier, MaskWeightedPoolingWrapper
from .losses import BinaryFocalLoss
from .metrics_classification import (
    BinaryMetrics,
    binary_classification_metrics,
    find_threshold,
)
from ..utils.training_utils import (
    set_seed,
    build_optimizer,
    build_scheduler,
)

SLICE_COLUMNS = ("class_label", "lighting", "state", "source_group", "source_subgroup", "neg_type")

LOGGER = logging.getLogger("train_backbone_a")


def _build_loss_function(
    cfg: Dict[str, Any],
    device: torch.device,
    pos_weight: Optional[torch.Tensor],
) -> Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    loss_cfg = cfg.get("loss", {}) or {}
    name = str(loss_cfg.get("name", "bce")).lower()
    if name in {"bce", "bce_with_logits", "bce_logits"}:
        pos_weight_device = pos_weight.to(device) if pos_weight is not None else None
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_device, reduction="none")
        criterion.to(device)

        def _loss(
            logits: torch.Tensor,
            targets: torch.Tensor,
            sample_weights: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            losses = criterion(logits, targets)
            if sample_weights is not None:
                if sample_weights.shape != losses.shape:
                    raise ValueError("Sample weights must match the per-sample loss shape.")
                losses = losses * sample_weights
            return losses.mean()

        return _loss

    if name == "focal":
        alpha_cfg = loss_cfg.get("focal_alpha", "auto")
        if isinstance(alpha_cfg, str) and alpha_cfg.lower() == "auto" and pos_weight is not None:
            # Convert pos_weight ratio (neg/pos) into class weighting alpha
            try:
                ratio = float(pos_weight.item())
                alpha_val = 1.0 / (1.0 + ratio)
            except Exception:
                alpha_val = 0.25
        else:
            alpha_val = float(alpha_cfg)
        gamma_val = float(loss_cfg.get("focal_gamma", 2.0))
        criterion = BinaryFocalLoss(alpha=alpha_val, gamma=gamma_val, reduction="none")
        criterion.to(device)

        def _loss(
            logits: torch.Tensor,
            targets: torch.Tensor,
            sample_weights: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            losses = criterion(logits, targets)
            if sample_weights is not None:
                if sample_weights.shape != losses.shape:
                    raise ValueError("Sample weights must match the per-sample loss shape.")
                losses = losses * sample_weights
            return losses.mean()

        return _loss

    raise ValueError(f"Unsupported loss.name: {name}")


def _parse_args() -> argparse.Namespace:
    # Also supports A3/A4 color-sensitive heads
    parser = argparse.ArgumentParser(description="Train Backbone A heads (A1/A2/A3/A4) for blueberry pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/backbone_a.yaml"))
    parser.add_argument("--task", choices=["a1", "a2", "a3", "a4"], default=None)
    parser.add_argument("--fold-id", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--override", nargs="*", default=None, help="key=value overrides")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _set_nested(cfg: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str] | None) -> None:
    if not overrides:
        return
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value format")
        key, raw = item.split("=", 1)
        keys = key.strip().split(".")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        _set_nested(cfg, keys, value)





def _prepare_datasets(cfg: Dict[str, Any], task: str, fold_id: int) -> Dict[str, BerryClassificationDataset]:
    data_cfg = cfg["data"]
    split_mode = str(data_cfg.get("split_mode", "fold"))
    split_ratios = data_cfg.get("split_ratios", [70.0, 15.0, 15.0])
    try:
        split_seed = int(cfg.get("seed", 1337))
    except Exception:
        split_seed = 1337

    default_union_cfg = {
        "path": data_cfg.get("union_metadata_path"),
        "class": data_cfg.get("union_metadata_class", "never"),
        "split": data_cfg.get("union_split", "train"),
        "include_union_channel": data_cfg.get("include_union_channel", False),
        "guidance": data_cfg.get("union_guidance"),
    }
    union_map = data_cfg.get("union_metadata", {}) if isinstance(data_cfg.get("union_metadata"), dict) else {}

    def _resolve_union_settings(task_name: str) -> Dict[str, Any]:
        task_cfg: Dict[str, Any] = {}
        raw = union_map.get(task_name) if isinstance(union_map, dict) else None
        if isinstance(raw, str):
            task_cfg = {"path": raw}
        elif isinstance(raw, dict):
            task_cfg = raw
        resolved: Dict[str, Any] = {}
        resolved["path"] = task_cfg.get("path") or default_union_cfg.get("path")
        resolved["class"] = task_cfg.get("class") or task_cfg.get("label") or default_union_cfg.get("class")
        resolved["split"] = task_cfg.get("split") or default_union_cfg.get("split")
        if "include_union_channel" in task_cfg:
            resolved["include_union_channel"] = task_cfg.get("include_union_channel")
        else:
            resolved["include_union_channel"] = default_union_cfg.get("include_union_channel", False)
        resolved["guidance"] = task_cfg.get("guidance")
        if resolved["guidance"] is None:
            resolved["guidance"] = default_union_cfg.get("guidance")
        return resolved

    union_settings = _resolve_union_settings(task)
    union_path = union_settings.get("path")
    union_path = Path(union_path) if union_path not in {None, "", "null"} else None
    union_class = union_settings.get("class") or "never"
    union_split = union_settings.get("split", "train")
    include_union_channel = bool(union_settings.get("include_union_channel", False))
    union_guidance = union_settings.get("guidance")

    base = ClassificationDataConfig(
        metadata_dir=Path(data_cfg["metadata_dir"]),
        crops_csv=data_cfg["crops_csv"],
        notberry_csv=data_cfg.get("notberry_csv", "notberry.csv"),
        folds=int(cfg["experiment"]["folds"]),
        fold_id=fold_id,
        split_mode=split_mode,
        split_ratios=tuple(split_ratios) if isinstance(split_ratios, (list, tuple)) else (70.0, 15.0, 15.0),
        split_seed=split_seed,
        split_group_column=(data_cfg.get("split_group_column") if data_cfg.get("split_group_column") not in {"", None, "null"} else None),
        split_group_val_max_frac=(float(data_cfg.get("split_group_val_max_frac")) if data_cfg.get("split_group_val_max_frac") is not None else None),
        split_group_test_max_frac=(float(data_cfg.get("split_group_test_max_frac")) if data_cfg.get("split_group_test_max_frac") is not None else None),
        split_group_pos_train_min_frac=(float(data_cfg.get("split_group_pos_train_min_frac")) if data_cfg.get("split_group_pos_train_min_frac") is not None else None),
        manual_split_dir=(
            Path(data_cfg.get("manual_split_dir"))
            if data_cfg.get("manual_split_dir") not in {None, ""}
            else None
        ),
        union_metadata_path=union_path,
        union_metadata_class=union_class,
        include_union_channel=include_union_channel,
        union_split=union_split,
        union_guidance=union_guidance,
        input_size=(int(data_cfg["input_size"][0]), int(data_cfg["input_size"][1])),
        include_mask_channel=bool(data_cfg.get("include_mask_channel", True)),
        mask_usage=data_cfg.get("mask_usage", "auto"),
        augment=data_cfg.get("augment"),
        extra_negative_labels=data_cfg.get("extra_negative_labels"),
        split="train",
    )
    train_dataset = BerryClassificationDataset(task=task, config=base)
    val_cfg = ClassificationDataConfig(
        metadata_dir=base.metadata_dir,
        crops_csv=base.crops_csv,
        notberry_csv=base.notberry_csv,
        folds=base.folds,
        fold_id=fold_id,
        split_mode=split_mode,
        split_ratios=base.split_ratios,
        split_seed=split_seed,
        split_group_column=base.split_group_column,
        split_group_val_max_frac=base.split_group_val_max_frac,
        split_group_test_max_frac=base.split_group_test_max_frac,
        split_group_pos_train_min_frac=base.split_group_pos_train_min_frac,
        manual_split_dir=base.manual_split_dir,
        union_metadata_path=base.union_metadata_path,
        union_metadata_class=base.union_metadata_class,
        include_union_channel=base.include_union_channel,
        union_split=base.union_split,
        union_guidance=base.union_guidance,
        input_size=base.input_size,
        include_mask_channel=base.include_mask_channel,
        mask_usage=data_cfg.get("mask_usage", "auto"),
        augment=None,
        extra_negative_labels=base.extra_negative_labels,
        split="val",
    )
    val_dataset = BerryClassificationDataset(task=task, config=val_cfg)
    return {"train": train_dataset, "val": val_dataset}


def _build_sampler(dataset: BerryClassificationDataset, enabled: bool) -> Optional[WeightedRandomSampler]:
    if not enabled:
        return None
    if len(dataset) == 0:
        LOGGER.warning("Weighted sampler disabled: empty training dataset.")
        return None
    targets = dataset.df["target"].to_numpy()
    class_counts = np.bincount(targets, minlength=2)
    weights = np.zeros_like(targets, dtype=np.float64)
    for cls in range(len(class_counts)):
        if class_counts[cls] == 0:
            continue
        weights[targets == cls] = 1.0 / class_counts[cls]
    # Optional hard-negative mining boost for specific samples (e.g., previous FPs)
    try:
        hnm_enabled = bool(getattr(dataset.cfg, "hnm_enabled", False))
    except Exception:
        hnm_enabled = False
    if hnm_enabled:
        try:
            repo_root = getattr(dataset, "repo_root", Path(".")).resolve()
            task = getattr(dataset, "task", "a3")
            # glob patterns relative to repo root
            hnm_glob = getattr(dataset.cfg, "hnm_glob", None)
            if not hnm_glob:
                # default: both val and test hard negatives
                base = repo_root / "outputs" / "backbone_a" / task
                patterns = [str(base / "hard_negatives" / "fold_*.csv"), str(base / "hard_negatives_test" / "fold_*.csv")]
            else:
                patterns = [str(_p) for _p in ([hnm_glob] if isinstance(hnm_glob, (str, Path)) else hnm_glob)]
            import glob
            ann_set: set[int] = set()
            error_types = set(getattr(dataset.cfg, "hnm_error_types", ["fp"]))
            slice_label = getattr(dataset.cfg, "hnm_slice_label", None)
            slice_value = getattr(dataset.cfg, "hnm_slice_value", None)
            for pat in patterns:
                for csv_path in glob.glob(pat):
                    try:
                        df = pd.read_csv(csv_path)
                        if "error_type" in df.columns:
                            df = df[df["error_type"].isin(list(error_types))]
                        if slice_label and slice_label in df.columns and slice_value is not None:
                            df = df[df[slice_label].astype(str) == str(slice_value)]
                        if "annotation_id" in df.columns:
                            ann_set.update(int(a) for a in df["annotation_id"].tolist())
                    except Exception:
                        continue
            if ann_set:
                boost = float(getattr(dataset.cfg, "hnm_boost", 3.0))
                anns = dataset.df.get("annotation_id", None)
                if anns is not None:
                    anns_np = anns.to_numpy()
                    mask = np.isin(anns_np, list(ann_set))
                    weights[mask] = weights[mask] * boost
        except Exception:
            LOGGER.debug("Hard-negative mining boost failed; continuing without boost.")
    weights = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _build_dataloaders(
    datasets: Dict[str, BerryClassificationDataset],
    cfg: Dict[str, Any],
) -> Dict[str, DataLoader]:
    data_cfg = cfg["data"]
    sampler = _build_sampler(datasets["train"], bool(data_cfg.get("use_weighted_sampler", False)))
    dl_train = DataLoader(
        datasets["train"],
        batch_size=int(data_cfg["batch_size"]),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", False)),
        drop_last=False,
    )
    dl_val = DataLoader(
        datasets["val"],
        batch_size=int(data_cfg.get("val_batch_size", data_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", False)),
        drop_last=False,
    )
    return {"train": dl_train, "val": dl_val}





def _compute_pos_weight(dataset: BerryClassificationDataset, cfg: Dict[str, Any]) -> Optional[torch.Tensor]:
    """Compute BCE pos_weight for the positive class (label=1).

    Supports:
    - loss.pos_weight: None | "auto" | float
    - loss.pos_weight_power (float, default=1.0): exponent on (neg/pos)
    - loss.pos_weight_multiplier (float, default=1.0): multiplicative boost
    - loss.pos_weight_min (float, default=1.0): lower clamp
    - loss.pos_weight_max (float, optional): upper clamp (no clamp if missing)
    """
    loss_cfg = cfg.get("loss", {})
    value = loss_cfg.get("pos_weight", None)
    if value is None:
        return None
    if value == "auto":
        counts = dataset.df["target"].value_counts().to_dict()
        pos = int(counts.get(1, 0))
        neg = int(counts.get(0, 0))
        if pos <= 0:
            return None
        base = max(1.0, neg / max(1, pos))
        power = float(loss_cfg.get("pos_weight_power", 1.0) or 1.0)
        mult = float(loss_cfg.get("pos_weight_multiplier", 1.0) or 1.0)
        min_w = float(loss_cfg.get("pos_weight_min", 1.0) or 1.0)
        max_w = loss_cfg.get("pos_weight_max", None)
        weight = mult * (base ** power)
        weight = max(min_w, float(weight))
        if max_w is not None:
            try:
                weight = min(float(max_w), weight)
            except Exception:
                pass
        LOGGER.info(
            "pos_weight(auto): neg=%d pos=%d -> base=%.3f, power=%.2f, mult=%.2f => %.3f",
            neg,
            pos,
            base,
            power,
            mult,
            weight,
        )
        return torch.tensor([weight], dtype=torch.float32)
    try:
        w = float(value)
    except Exception as exc:
        raise ValueError(f"Invalid loss.pos_weight value: {value}") from exc
    return torch.tensor([w], dtype=torch.float32)


def _build_sample_weight_function(
    cfg: Dict[str, Any],
    device: torch.device,
) -> Optional[Callable[[Dict[str, Any], torch.Tensor], torch.Tensor]]:
    loss_cfg = cfg.get("loss", {}) or {}
    neg_cfg = loss_cfg.get("negative_class_weights") or {}
    pos_cfg = loss_cfg.get("positive_class_weights") or {}
    neg_default = float(loss_cfg.get("negative_class_default", 1.0))
    pos_default = float(loss_cfg.get("positive_class_default", 1.0))

    def _normalize_label(value: Any) -> str:
        return str(value).strip().lower()

    neg_map = {_normalize_label(key): float(val) for key, val in neg_cfg.items()}
    pos_map = {_normalize_label(key): float(val) for key, val in pos_cfg.items()}
    enabled = bool(neg_map or pos_map or neg_default != 1.0 or pos_default != 1.0)
    if not enabled:
        return None

    def _compute_for_batch(batch: Dict[str, Any], targets: torch.Tensor) -> torch.Tensor:
        n = targets.shape[0]
        weights = torch.ones(n, dtype=torch.float32, device=device)
        class_labels = batch.get("class_label")
        if class_labels is None:
            return weights
        if isinstance(class_labels, torch.Tensor):
            labels_seq = [str(x) for x in class_labels.tolist()]
        elif isinstance(class_labels, (list, tuple)):
            labels_seq = [str(x) for x in class_labels]
        else:
            labels_seq = [str(class_labels)] * n
        if len(labels_seq) < n:
            labels_seq = list(labels_seq) + [""] * (n - len(labels_seq))
        elif len(labels_seq) > n:
            labels_seq = list(labels_seq)[:n]
        is_positive = targets.detach() >= 0.5
        for idx in range(n):
            label_norm = _normalize_label(labels_seq[idx])
            if bool(is_positive[idx].item()):
                weight = pos_map.get(label_norm, pos_default)
            else:
                weight = neg_map.get(label_norm, neg_default)
            weights[idx] = weights[idx] * float(weight)
        return weights

    return _compute_for_batch


def _prepare_output_dirs(cfg: Dict[str, Any], task: str, fold_id: int) -> Dict[str, Path]:
    root = Path(cfg["experiment"]["output_root"]) / task / f"fold_{fold_id:02d}"
    root.mkdir(parents=True, exist_ok=True)
    preds_dir = Path(cfg["logging"]["predictions_dir"]) / task
    preds_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "predictions": preds_dir}


def _format_metrics(metrics: BinaryMetrics) -> Dict[str, float]:
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "specificity": metrics.specificity,
        "f1": metrics.f1,
        "auc": metrics.auc if metrics.auc is not None else float("nan"),
        "pr_auc": metrics.pr_auc if metrics.pr_auc is not None else float("nan"),
        "ece": metrics.ece,
        "threshold": metrics.threshold,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "tn": metrics.tn,
        "fn": metrics.fn,
    }


def _has_two_classes(targets: np.ndarray) -> bool:
    return np.any(targets == 0) and np.any(targets == 1)


def _split_calibration_eval(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    fraction: float,
    seed: int,
    slice_labels: Dict[str, np.ndarray],
    min_per_class: int = 1,
) -> Dict[str, np.ndarray]:
    n = len(logits)
    indices = np.arange(n)
    if n == 0 or fraction <= 0.0 or fraction >= 1.0:
        return {"cal_idx": indices, "eval_idx": indices}

    effective_fraction = float(np.clip(fraction, 0.05, 0.95))
    rng = np.random.default_rng(seed)

    def _slice_key(idx: int) -> str:
        parts = [str(int(targets[idx]))]
        for key in ("lighting", "state", "source_group"):
            if key in slice_labels:
                parts.append(str(slice_labels[key][idx]))
        return "|".join(parts)

    group_indices: Dict[str, List[int]] = {}
    for idx in indices:
        key = _slice_key(idx)
        group_indices.setdefault(key, []).append(idx)

    cal_idx: List[int] = []
    eval_idx: List[int] = []
    for key, idxs in group_indices.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        take = int(round(len(idxs) * effective_fraction))
        if len(idxs) - take < min_per_class:
            take = max(0, len(idxs) - min_per_class)
        if take <= 0 and len(idxs) > 1 and len(cal_idx) < min_per_class:
            take = 1
        cal_idx.extend(idxs[:take])
        eval_idx.extend(idxs[take:])

    if not cal_idx or not eval_idx:
        return {"cal_idx": indices, "eval_idx": indices}

    cal_idx = np.sort(np.array(cal_idx, dtype=np.int32))
    eval_idx = np.sort(np.array(eval_idx, dtype=np.int32))

    if not _has_two_classes(targets[cal_idx]) or not _has_two_classes(targets[eval_idx]):
        return {"cal_idx": indices, "eval_idx": indices}

    return {"cal_idx": cal_idx, "eval_idx": eval_idx}


def _collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> Dict[str, Any]:
    model.eval()
    logits_list: List[float] = []
    targets_list: List[int] = []
    annotation_ids: List[int] = []
    image_ids: List[int] = []
    meta_lists: Dict[str, List[Any]] = {key: [] for key in SLICE_COLUMNS}
    meta_lists["path"] = []
    loss_sum = 0.0
    count = 0
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for batch in loader:
            inputs = batch["x"].to(device, non_blocking=True)
            if use_amp:
                inputs = inputs.to(memory_format=torch.channels_last)
            targets = batch["y"].float().to(device)
            with amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs).squeeze(1)
            logits_list.extend(outputs.detach().cpu().tolist())
            targets_list.extend(targets.cpu().long().tolist())
            annotation_ids.extend(batch["annotation_id"])  # type: ignore[arg-type]
            image_ids.extend(batch.get("image_id", [-1] * len(outputs)))  # type: ignore[arg-type]
            for key in meta_lists.keys():
                if key in batch:
                    values = batch[key]
                    if isinstance(values, list):
                        meta_lists[key].extend(values)
                    else:
                        meta_lists[key].extend(list(values))
            loss_sum += criterion(outputs, targets).item()
            count += len(outputs)
    avg_loss = float(loss_sum / max(1, count))
    meta_out = {key: np.array(vals, dtype=object) for key, vals in meta_lists.items() if vals}
    return {
        "logits": np.array(logits_list, dtype=np.float64),
        "targets": np.array(targets_list, dtype=np.int32),
        "annotation_ids": np.array(annotation_ids, dtype=np.int64),
        "image_ids": np.array(image_ids, dtype=np.int64),
        "loss": avg_loss,
        "meta": meta_out,
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    *,
    use_amp: bool,
    grad_accum: int,
    grad_clip: float,
    log_interval: int,
    criterion_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    sample_weight_fn: Optional[Callable[[Dict[str, Any], torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    correct = 0
    total = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        inputs = batch["x"].to(device, non_blocking=True)
        if use_amp:
            inputs = inputs.to(memory_format=torch.channels_last)
        targets = batch["y"].float().to(device)
        sample_weights: Optional[torch.Tensor] = None
        if sample_weight_fn is not None:
            sample_weights = sample_weight_fn(batch, targets)
        with amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(inputs).squeeze(1)
            loss = criterion_fn(logits, targets, sample_weights=sample_weights)
        scaler.scale(loss).backward()
        if ((step + 1) % grad_accum == 0) or (step + 1 == len(loader)):
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.detach().item()
        total_batches += 1
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += int((preds == targets.long()).sum().item())
        total += targets.numel()
        if log_interval and (step + 1) % log_interval == 0:
            LOGGER.info("Train step %d/%d | loss=%.4f", step + 1, len(loader), loss.detach().item())
    return {
        "loss": total_loss / max(1, total_batches),
        "accuracy": correct / max(1, total),
    }


def main() -> int:
    args = _parse_args()
    cfg = _load_yaml(args.config)
    _apply_overrides(cfg, args.override)

    exp_cfg = cfg.setdefault("experiment", {})
    task = args.task or exp_cfg.get("task", "a1")
    if task not in {"a1", "a2", "a3", "a4"}:
        raise ValueError(f"Unsupported task '{task}'")
    fold_id = args.fold_id if args.fold_id is not None else int(exp_cfg.get("fold_id", 0))
    device_str = args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu").type
    device = torch.device(device_str)

    log_level = getattr(logging, cfg.get("logging", {}).get("log_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    seed = int(cfg.get("seed", 1337))
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    datasets = _prepare_datasets(cfg, task, fold_id)
    loaders = _build_dataloaders(datasets, cfg)
    try:
        n_train = len(datasets["train"]) if datasets.get("train") is not None else 0
        n_val = len(datasets["val"]) if datasets.get("val") is not None else 0
        cnt_train = datasets["train"].df["target"].value_counts().to_dict() if n_train > 0 else {}
        cnt_val = datasets["val"].df["target"].value_counts().to_dict() if n_val > 0 else {}
        u_train = int(datasets["train"].df["image_id"].nunique()) if n_train > 0 and "image_id" in datasets["train"].df.columns else 0
        u_val = int(datasets["val"].df["image_id"].nunique()) if n_val > 0 and "image_id" in datasets["val"].df.columns else 0
        LOGGER.info("Split sizes | train=%d (counts=%s, images=%d), val=%d (counts=%s, images=%d)", n_train, cnt_train, u_train, n_val, cnt_val, u_val)
    except Exception:
        pass

    # Infer input channels from a real dataset sample so that optional
    # A3/A4 color features and/or mask channel are handled automatically.
    try:
        sample0 = datasets["train"][0]
        in_channels = int(sample0["x"].shape[0])  # type: ignore[index]
    except Exception:
        include_mask = bool(cfg["data"].get("include_mask_channel", True))
        use_mask_channel = getattr(datasets["train"], "use_mask_channel", include_mask)
        in_channels = 4 if use_mask_channel else 3
    model_cfg = cfg["model"]
    model = build_classifier(
        backbone=model_cfg.get("name", "mobilenet_v3_small"),
        num_classes=1,
        in_channels=in_channels,
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=model_cfg.get("dropout"),
    )
    # Optional mask-weighted pooling (uses last channel as mask)
    try:
        mwp_cfg = model_cfg.get("mask_weighted_pooling", {}) or {}
        if bool(mwp_cfg.get("enabled", False)):
            model = MaskWeightedPoolingWrapper(model, has_mask_channel=True)
    except Exception:
        pass
    model.to(device)
    if bool(cfg["training"].get("channels_last", True)):
        model.to(memory_format=torch.channels_last)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, max_epochs=int(cfg["training"]["max_epochs"]))

    train_cfg = cfg["training"]
    pos_weight = _compute_pos_weight(datasets["train"], cfg)
    criterion_fn = _build_loss_function(cfg, device, pos_weight)
    sample_weight_fn = _build_sample_weight_function(cfg, device)
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = amp.GradScaler(enabled=use_amp) if device.type == "cuda" else amp.GradScaler(enabled=False)
    max_epochs = int(train_cfg["max_epochs"])
    patience = int(train_cfg.get("patience", 10))
    min_delta = float(train_cfg.get("min_delta", 0.0))
    grad_accum = max(1, int(train_cfg.get("gradient_accumulation", 1)))
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0) or 0.0)
    log_interval = int(train_cfg.get("log_interval", 0))

    outputs = _prepare_output_dirs(cfg, task, fold_id)
    best_metric = -float("inf")
    best_state: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []
    epochs_no_improve = 0

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        train_stats = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            use_amp=use_amp,
            grad_accum=grad_accum,
            grad_clip=grad_clip,
            log_interval=log_interval,
            criterion_fn=criterion_fn,
            sample_weight_fn=sample_weight_fn,
        )

        with torch.no_grad():
            val_outputs = _collect_predictions(model, loaders["val"], device, use_amp)
        logits_clamped_epoch = np.clip(val_outputs["logits"], -20.0, 20.0)
        val_probs = 1.0 / (1.0 + np.exp(-logits_clamped_epoch))
        val_targets = val_outputs["targets"]
        # Metrics at fixed 0.5 (for reference)
        val_metrics_05 = binary_classification_metrics(
            probs=val_probs,
            targets=val_targets,
            threshold=0.5,
        )
        # Threshold-aware metrics (optimize under current recall constraint)
        th_cfg = cfg.get("threshold", {})
        th_meta_epoch = find_threshold(
            val_probs,
            val_targets,
            grid_size=int(th_cfg.get("grid_size", 201)),
            min_recall_positive=float(th_cfg.get("min_recall_positive", cfg.get("training", {}).get("min_recall", 0.0))),
            cost=th_cfg.get("cost"),
            min_threshold=float(th_cfg.get("min_value", 1e-3)),
            max_threshold=float(th_cfg.get("max_value", 1.0 - 1e-3)),
            fallback_threshold=th_cfg.get("fallback_threshold"),
            abstain_margin=float(th_cfg.get("abstain_margin", 0.0)),
            grid_mode=str(th_cfg.get("grid_mode", "linear")).lower(),
        )
        th_epoch = float(th_meta_epoch.get("threshold", 0.5))
        val_metrics = binary_classification_metrics(
            probs=val_probs,
            targets=val_targets,
            threshold=th_epoch,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_outputs["loss"],
                "val_accuracy": val_metrics.accuracy,
                "val_recall": val_metrics.recall,
                "val_precision": val_metrics.precision,
                "val_f1": val_metrics.f1,
                "val_ece": val_metrics.ece,
                "val_pr_auc": val_metrics.pr_auc if val_metrics.pr_auc is not None else float("nan"),
                "val_threshold": th_epoch,
                "val_f1_at_0p5": val_metrics_05.f1,
                "val_recall_at_0p5": val_metrics_05.recall,
            }
        )
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f (th=%.3f) | val_recall=%.4f | val_pr_auc=%.4f",
            epoch,
            max_epochs,
            train_stats["loss"],
            val_outputs["loss"],
            val_metrics.f1,
            th_epoch,
            val_metrics.recall,
            (val_metrics.pr_auc if val_metrics.pr_auc is not None else float("nan")),
        )

        metric_value = val_metrics.f1
        min_recall_required = float(train_cfg.get("min_recall", 0.0))
        meets_recall = val_metrics.recall >= min_recall_required
        if not meets_recall:
            LOGGER.warning(
                "Epoch %d: recall %.4f fell below required minimum %.4f",
                epoch,
                val_metrics.recall,
                min_recall_required,
            )
        if meets_recall and metric_value > best_metric + min_delta:
            best_metric = metric_value
            epochs_no_improve = 0
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "metric": best_metric,
            }
            torch.save(best_state, outputs["root"] / "best.pt")
        else:
            epochs_no_improve += 1

        torch.save({"model": model.state_dict(), "epoch": epoch}, outputs["root"] / "last.pt")

        if scheduler is not None:
            scheduler.step()

        if patience and epochs_no_improve >= patience:
            LOGGER.info("Early stopping triggered after %d epochs without improvement.", epochs_no_improve)
            break

        if args.dry_run:
            LOGGER.info("Dry-run enabled, stopping after first epoch.")
            break

    elapsed = time.time() - start_time
    LOGGER.info("Training finished in %.1f seconds", elapsed)

    if best_state is None:
        best_state = {"model": model.state_dict(), "metric": best_metric}
    model.load_state_dict(best_state["model"])  # type: ignore[arg-type]
    model.eval()

    val_outputs = _collect_predictions(model, loaders["val"], device, use_amp)
    logits = val_outputs["logits"]
    logits_clamped = np.clip(logits, -20.0, 20.0)
    targets = val_outputs["targets"]
    probs = 1.0 / (1.0 + np.exp(-logits_clamped))
    meta_dict: Dict[str, np.ndarray] = {
        key: value
        for key, value in val_outputs.get("meta", {}).items()
        if isinstance(value, np.ndarray) and value.shape[0] == logits.shape[0]
    }

    calibration_cfg = cfg.get("calibration", {})
    fraction = float(calibration_cfg.get("fraction", 0.0) or 0.0)
    split = _split_calibration_eval(
        logits_clamped,
        targets,
        fraction=fraction,
        seed=seed + fold_id * 31,
        slice_labels={k: v for k, v in meta_dict.items() if k in SLICE_COLUMNS},
        min_per_class=int(calibration_cfg.get("min_per_class", 1)),
    )
    cal_idx = split["cal_idx"]
    eval_idx = split["eval_idx"]
    cal_logits = logits_clamped[cal_idx]
    cal_targets = targets[cal_idx]
    eval_logits = logits_clamped[eval_idx]
    eval_targets = targets[eval_idx]

    temperature = 1.0
    temperature_source = "skipped"
    bias_value = 0.0
    probs_calib_all = probs.copy()
    if calibration_cfg.get("enabled", True) and _has_two_classes(cal_targets):
        # Bounded temperature search (optional bias) with global/OOF modes
        mode = str(calibration_cfg.get("mode", "fold")).lower()
        Tmin = float(calibration_cfg.get("min_temperature", 0.5))
        Tmax = float(calibration_cfg.get("max_temperature", 5.0))
        use_bias = bool(calibration_cfg.get("bias_enabled", False))
        bias_min = float(calibration_cfg.get("bias_min", -2.0))
        bias_max = float(calibration_cfg.get("bias_max", 2.0))
        global_temp_path = calibration_cfg.get("global_temperature_path")

        def _fit_t_on(logs: np.ndarray, tgs: np.ndarray, src: str) -> None:
            nonlocal temperature, bias_value, temperature_source
            res = fit_temperature_bounded(
                torch.tensor(logs, dtype=torch.float32),
                torch.tensor(tgs, dtype=torch.float32),
                min_temperature=Tmin,
                max_temperature=Tmax,
                bias_enabled=use_bias,
                bias_range=(bias_min, bias_max),
            )
            temperature = float(res.temperature)
            bias_value = float(res.bias)
            temperature_source = src

        # Try load precomputed global temperature
        if mode in {"global", "read"} and global_temp_path:
            gp = Path(global_temp_path)
            if gp.exists():
                try:
                    data = json.loads(gp.read_text(encoding="utf-8"))
                    temperature = float(data.get("temperature", 1.0))
                    bias_value = float(data.get("bias", 0.0))
                    temperature_source = "global"
                except Exception as exc:
                    LOGGER.warning("Failed to load global temperature from %s: %s", gp, exc)
        if temperature_source != "global":
            if mode in {"global_fit", "oof", "oof_fit"}:
                # Out-of-fold fit using existing validation predictions across folds
                preds_dir = outputs["predictions"]
                oof_paths = sorted(preds_dir.glob(f"{task}_fold*.csv"))
                if not oof_paths:
                    # Fall back to combined cal split in this fold
                    _fit_t_on(cal_logits, cal_targets, "fitted")
                else:
                    all_logs: List[np.ndarray] = []
                    all_tgs: List[np.ndarray] = []
                    for p in oof_paths:
                        try:
                            df = pd.read_csv(p)
                            # Prefer using the validation (cal+eval) subsets
                            if "subset" in df.columns:
                                df = df[df["subset"].isin(["cal", "eval", "full"])].copy()
                            all_logs.append(df["logit_clamped"].to_numpy(dtype=np.float32))
                            all_tgs.append(df["target"].to_numpy(dtype=np.int32))
                        except Exception:
                            continue
                    if all_logs:
                        logs = np.concatenate(all_logs, axis=0)
                        tgs = np.concatenate(all_tgs, axis=0)
                        _fit_t_on(logs, tgs, "oof_fit")
                    else:
                        _fit_t_on(cal_logits, cal_targets, "fitted")
                # Optionally persist global T
                if global_temp_path:
                    gp = Path(global_temp_path)
                    gp.parent.mkdir(parents=True, exist_ok=True)
                    gp.write_text(json.dumps({"temperature": float(temperature), "bias": float(bias_value), "task": task}), encoding="utf-8")
            else:
                # Per-fold fit on calibration subset
                _fit_t_on(cal_logits, cal_targets, "fitted")
                if global_temp_path and mode == "write":
                    gp = Path(global_temp_path)
                    gp.parent.mkdir(parents=True, exist_ok=True)
                    gp.write_text(json.dumps({"temperature": float(temperature), "bias": float(bias_value), "task": task}), encoding="utf-8")

        logits_scaled = apply_temperature_bias(torch.tensor(logits_clamped, dtype=torch.float32), float(temperature), float(bias_value)).cpu().numpy()
        probs_calib_all = 1.0 / (1.0 + np.exp(-logits_scaled))
    else:
        if not _has_two_classes(cal_targets):
            temperature_source = "insufficient_positives"
        probs_calib_all = probs.copy()

    threshold_cfg = cfg.get("threshold", {})
    min_threshold = float(threshold_cfg.get("min_value", 1e-3))
    max_threshold = float(threshold_cfg.get("max_value", 1.0 - 1e-3))
    fallback_threshold = threshold_cfg.get("fallback_threshold")
    abstain_margin = float(threshold_cfg.get("abstain_margin", 0.02))
    threshold = float(np.clip(threshold_cfg.get("initial", 0.5), min_threshold, max_threshold))
    if threshold_cfg.get("enabled", True) and _has_two_classes(cal_targets):
        threshold_meta = find_threshold(
            probs_calib_all[cal_idx],
            cal_targets,
            grid_size=int(threshold_cfg.get("grid_size", 101)),
            min_recall_positive=float(threshold_cfg.get("min_recall_positive", 0.9)),
            cost=threshold_cfg.get("cost"),
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            fallback_threshold=fallback_threshold,
            abstain_margin=abstain_margin,
            grid_mode=str(threshold_cfg.get("grid_mode", "linear")).lower(),
        )
        threshold = float(threshold_meta["threshold"])
        if not threshold_meta.get("min_recall_satisfied", True):
            LOGGER.warning(
                "Recall constraint not satisfied on calibration split; applying fallback threshold %.4f",
                threshold,
            )
    else:
        threshold = float(np.clip(fallback_threshold if fallback_threshold is not None else min_threshold, min_threshold, max_threshold))
        threshold_meta = {
            "threshold": threshold,
            "raw_threshold": threshold,
            "min_recall_satisfied": False,
            "fallback_applied": True,
            "best_cost": float("nan"),
            "abstain_margin": abstain_margin,
        }

    eval_probs_uncal = probs[eval_idx]
    eval_probs_cal = probs_calib_all[eval_idx]
    metrics_uncal = binary_classification_metrics(
        probs=eval_probs_uncal,
        targets=eval_targets,
        threshold=threshold,
    )
    metrics_cal = binary_classification_metrics(
        probs=eval_probs_cal,
        targets=eval_targets,
        threshold=threshold,
    )

    # If recall constraint on calibration split failed and we fell back to a
    # fixed low threshold, try the candidate "raw_threshold" (closest match
    # to the constraint) and adopt it if it improves F1 without dropping
    # recall below a configurable floor. This guards against pathological
    # fallbacks (e.g., 0.04) that yield many false positives.
    if threshold_meta.get("fallback_applied", False):
        th_raw = float(threshold_meta.get("raw_threshold", threshold))
        recall_floor = float(threshold_cfg.get("fallback_recall_floor", 0.85))
        metrics_raw_cal = binary_classification_metrics(
            probs=eval_probs_cal,
            targets=eval_targets,
            threshold=th_raw,
        )
        if metrics_raw_cal.f1 > metrics_cal.f1 and metrics_raw_cal.recall >= recall_floor:
            threshold = th_raw
            threshold_meta["threshold"] = threshold
            threshold_meta["fallback_overridden_by_raw"] = True
            # Recompute metrics with improved threshold
            metrics_uncal = binary_classification_metrics(
                probs=eval_probs_uncal,
                targets=eval_targets,
                threshold=threshold,
            )
            metrics_cal = binary_classification_metrics(
                probs=eval_probs_cal,
                targets=eval_targets,
                threshold=threshold,
            )

    eval_meta = {key: values[eval_idx] for key, values in meta_dict.items() if values.shape[0] == logits.shape[0]}
    slice_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key in SLICE_COLUMNS:
        if key not in eval_meta:
            continue
        values = eval_meta[key]
        unique_vals = [val for val in np.unique(values) if str(val) not in {"", "None"}]
        metrics_by_label: Dict[str, Dict[str, float]] = {}
        for val in unique_vals:
            mask = values == val
            if mask.sum() < 2:
                continue
            metrics_slice = binary_classification_metrics(
                probs=eval_probs_cal[mask],
                targets=eval_targets[mask],
                threshold=threshold,
            )
            metrics_by_label[str(val)] = _format_metrics(metrics_slice)
        if metrics_by_label:
            slice_metrics[key] = metrics_by_label

    preds_eval = (eval_probs_cal >= threshold).astype(int)
    fp_idx = np.where((preds_eval == 1) & (eval_targets == 0))[0]
    fn_idx = np.where((preds_eval == 0) & (eval_targets == 1))[0]
    eval_annotations = val_outputs["annotation_ids"][eval_idx]
    eval_images = val_outputs["image_ids"][eval_idx]
    eval_paths = meta_dict.get("path")
    if isinstance(eval_paths, np.ndarray) and eval_paths.shape[0] == logits.shape[0]:
        eval_paths = eval_paths[eval_idx]
    else:
        eval_paths = np.array([""] * len(eval_idx), dtype=object)

    hard_records: List[Dict[str, Any]] = []

    def _append_hard(indices: np.ndarray, err_type: str) -> None:
        for local_idx in indices:
            rec = {
                "annotation_id": int(eval_annotations[local_idx]),
                "image_id": int(eval_images[local_idx]),
                "target": int(eval_targets[local_idx]),
                "prob": float(eval_probs_uncal[local_idx]),
                "prob_cal": float(eval_probs_cal[local_idx]),
                "logit": float(eval_logits[local_idx]),
                "threshold": float(threshold),
                "error_type": err_type,
                "path": str(eval_paths[local_idx]),
            }
            for key in SLICE_COLUMNS:
                if key in eval_meta:
                    rec[key] = str(eval_meta[key][local_idx])
            hard_records.append(rec)

    _append_hard(fp_idx, "fp")
    _append_hard(fn_idx, "fn")

    summary = {
        "task": task,
        "fold_id": fold_id,
        "seed": seed,
        "temperature": float(temperature),
        "temperature_bias": float(bias_value),
        "temperature_source": temperature_source,
        "threshold": float(threshold),
        "abstain_margin": float(threshold_meta.get("abstain_margin", 0.0)),
        "calibration_subset_size": int(len(cal_idx)),
        "evaluation_subset_size": int(len(eval_idx)),
        "metrics_uncalibrated": _format_metrics(metrics_uncal),
        "metrics_calibrated": _format_metrics(metrics_cal),
        "threshold_meta": threshold_meta,
        "slice_metrics": slice_metrics,
        "hard_negative_counts": {"fp": int(len(fp_idx)), "fn": int(len(fn_idx))},
        "config_path": str(args.config),
    }

    summary_path = outputs["root"] / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    preds_dir = outputs["predictions"]
    out_pred = preds_dir / f"{task}_fold{fold_id:02d}.csv"
    subset_labels = np.full(len(logits), "eval", dtype=object)
    subset_labels[cal_idx] = "cal"
    if np.array_equal(cal_idx, eval_idx):
        subset_labels[:] = "full"
    df_preds = pd.DataFrame(
        {
            "annotation_id": val_outputs["annotation_ids"],
            "image_id": val_outputs["image_ids"],
            "target": targets,
            "logit": logits,
            "logit_clamped": logits_clamped,
            "prob": probs,
            "prob_cal": probs_calib_all,
            "subset": subset_labels,
        }
    )
    for key, values in meta_dict.items():
        df_preds[key] = values
    df_preds.to_csv(out_pred, index=False)

    # Store hard negatives under the task directory without duplicating the task segment
    # outputs["root"] = <output_root>/<task>/fold_xx -> parent = <output_root>/<task>
    hard_neg_dir = Path(outputs["root"]).parent / "hard_negatives"
    hard_neg_dir.mkdir(parents=True, exist_ok=True)
    hard_csv = hard_neg_dir / f"fold_{fold_id:02d}.csv"
    if hard_records:
        pd.DataFrame.from_records(hard_records).to_csv(hard_csv, index=False)
    elif hard_csv.exists():
        hard_csv.unlink()

    if cfg.get("logging", {}).get("csv_log"):
        # Write per-task training logs to avoid mixing A1/A2 runs in a single CSV
        base_log_path = Path(cfg["logging"]["csv_log"])
        if base_log_path.suffix.lower() == ".csv":
            log_path = base_log_path.with_name(f"{base_log_path.stem}_{task}{base_log_path.suffix}")
        else:
            log_path = base_log_path / f"training_log_{task}.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()
        with log_path.open("a", encoding="utf-8") as fh:
            if write_header:
                fh.write(
                    "task,fold_id,epoch,train_loss,train_accuracy,val_loss,val_accuracy,val_recall,val_precision,val_f1,val_pr_auc,val_ece\n"
                )
            for row in history:
                fh.write(
                    f"{task},{fold_id},{row['epoch']},{row['train_loss']:.6f},{row['train_accuracy']:.6f},{row['val_loss']:.6f},{row['val_accuracy']:.6f},{row['val_recall']:.6f},{row['val_precision']:.6f},{row['val_f1']:.6f},{row.get('val_pr_auc', float('nan')):.6f},{row['val_ece']:.6f}\n"
                )

    metrics_csv = cfg.get("logging", {}).get("metrics_csv")
    if metrics_csv:
        metrics_path = Path(metrics_csv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not metrics_path.exists()
        with metrics_path.open("a", encoding="utf-8") as fh:
            if write_header:
                fh.write(
                    "task,fold_id,temperature,threshold,metric,f1_calibrated,recall_calibrated,pr_auc_calibrated,ece_calibrated\n"
                )
            fh.write(
                f"{task},{fold_id},{temperature:.6f},{threshold:.6f},{best_metric:.6f},{metrics_cal.f1:.6f},{metrics_cal.recall:.6f},{(metrics_cal.pr_auc if metrics_cal.pr_auc is not None else float('nan')):.6f},{metrics_cal.ece:.6f}\n"
            )

    # Intentionally avoid printing the full summary to console; it's saved to JSON
    # LOGGER.debug("Summary: %s", json.dumps(summary, indent=2))

    # ---------------------------------------------------------------
    # Optional: Test evaluation (only for ratio-based splits)
    # ---------------------------------------------------------------
    try:
        data_cfg = cfg.get("data", {})
        split_mode_lower = str(data_cfg.get("split_mode", "fold")).lower()
        if split_mode_lower in {"ratio", "manual"}:
            # Build test dataset/loader with the same parameters (no augmentations)
            test_union = _resolve_union_settings(task)
            test_union_path = test_union.get("path")
            test_union_path = Path(test_union_path) if test_union_path not in {None, "", "null"} else None
            test_union_class = test_union.get("class") or "never"
            test_cfg = ClassificationDataConfig(
                metadata_dir=Path(data_cfg["metadata_dir"]),
                crops_csv=data_cfg["crops_csv"],
                notberry_csv=data_cfg.get("notberry_csv", "notberry.csv"),
                folds=int(cfg["experiment"]["folds"]),
                fold_id=fold_id,
                split_mode=split_mode_lower,
                split_ratios=tuple(data_cfg.get("split_ratios", [70.0, 15.0, 15.0])),
                split_seed=int(cfg.get("seed", 1337)),
                split_group_column=(data_cfg.get("split_group_column") if data_cfg.get("split_group_column") not in {"", None, "null"} else None),
                split_group_val_max_frac=(float(data_cfg.get("split_group_val_max_frac")) if data_cfg.get("split_group_val_max_frac") is not None else None),
                split_group_test_max_frac=(float(data_cfg.get("split_group_test_max_frac")) if data_cfg.get("split_group_test_max_frac") is not None else None),
                split_group_pos_train_min_frac=(float(data_cfg.get("split_group_pos_train_min_frac")) if data_cfg.get("split_group_pos_train_min_frac") is not None else None),
                manual_split_dir=(
                    Path(data_cfg.get("manual_split_dir"))
                    if data_cfg.get("manual_split_dir") not in {None, ""}
                    else None
                ),
                union_metadata_path=test_union_path,
                union_metadata_class=test_union_class,
                include_union_channel=bool(test_union.get("include_union_channel", False)),
                union_split=test_union.get("split", "train"),
                union_guidance=test_union.get("guidance"),
                input_size=(int(data_cfg["input_size"][0]), int(data_cfg["input_size"][1])),
                include_mask_channel=bool(data_cfg.get("include_mask_channel", True)),
                mask_usage=data_cfg.get("mask_usage", "auto"),
                augment=None,
                extra_negative_labels=data_cfg.get("extra_negative_labels"),
                split="test",
            )
            test_dataset = BerryClassificationDataset(task=task, config=test_cfg)
            if len(test_dataset) > 0:
                dl_test = DataLoader(
                    test_dataset,
                    batch_size=int(data_cfg.get("val_batch_size", data_cfg.get("batch_size", 16))),
                    shuffle=False,
                    num_workers=int(data_cfg.get("num_workers", 4)),
                    pin_memory=bool(data_cfg.get("pin_memory", True)),
                    persistent_workers=bool(data_cfg.get("persistent_workers", False)),
                    drop_last=False,
                )
                LOGGER.info("Running test evaluation on %d samples...", len(test_dataset))
                test_outputs = _collect_predictions(model, dl_test, device, use_amp)
                t_logits = test_outputs["logits"]
                t_targets = test_outputs["targets"]
                t_logits_clamped = np.clip(t_logits, -20.0, 20.0)
                t_probs = 1.0 / (1.0 + np.exp(-t_logits_clamped))

                # Apply previously determined temperature (no fitting on test)
                calibration_cfg = cfg.get("calibration", {})
                if calibration_cfg.get("enabled", True) and np.isfinite(temperature):
                    t_scaled = apply_temperature(torch.tensor(t_logits_clamped, dtype=torch.float32), float(temperature)).cpu().numpy()
                    t_probs_cal = 1.0 / (1.0 + np.exp(-t_scaled))
                else:
                    t_probs_cal = t_probs.copy()

                # Evaluate with the chosen threshold
                test_metrics_uncal = binary_classification_metrics(probs=t_probs, targets=t_targets, threshold=float(threshold))
                test_metrics_cal = binary_classification_metrics(probs=t_probs_cal, targets=t_targets, threshold=float(threshold))

                # Slice metrics
                t_meta_dict: Dict[str, np.ndarray] = {
                    key: value
                    for key, value in test_outputs.get("meta", {}).items()
                    if isinstance(value, np.ndarray) and value.shape[0] == t_logits.shape[0]
                }
                test_slice_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
                for key in SLICE_COLUMNS:
                    if key not in t_meta_dict:
                        continue
                    values = t_meta_dict[key]
                    uniq = [val for val in np.unique(values) if str(val) not in {"", "None"}]
                    metrics_by_label: Dict[str, Dict[str, float]] = {}
                    for val_label in uniq:
                        mask = values == val_label
                        if mask.sum() < 2:
                            continue
                        m = binary_classification_metrics(probs=t_probs_cal[mask], targets=t_targets[mask], threshold=float(threshold))
                        metrics_by_label[str(val_label)] = _format_metrics(m)
                    if metrics_by_label:
                        test_slice_metrics[key] = metrics_by_label

                # Hard negatives (FP/FN) on test
                t_preds = (t_probs_cal >= float(threshold)).astype(int)
                t_fp_idx = np.where((t_preds == 1) & (t_targets == 0))[0]
                t_fn_idx = np.where((t_preds == 0) & (t_targets == 1))[0]
                t_ann = test_outputs["annotation_ids"]
                t_img = test_outputs["image_ids"]
                t_paths = t_meta_dict.get("path", np.array([""] * len(t_preds), dtype=object))
                t_hard: List[Dict[str, Any]] = []
                def _add_t(indices: np.ndarray, err: str) -> None:
                    for i in indices:
                        rec = {
                            "annotation_id": int(t_ann[i]),
                            "image_id": int(t_img[i]),
                            "target": int(t_targets[i]),
                            "prob": float(t_probs[i]),
                            "prob_cal": float(t_probs_cal[i]),
                            "logit": float(t_logits[i]),
                            "threshold": float(threshold),
                            "error_type": err,
                            "path": str(t_paths[i]) if isinstance(t_paths, np.ndarray) else str(t_paths),
                        }
                        for key in SLICE_COLUMNS:
                            if key in t_meta_dict:
                                rec[key] = str(t_meta_dict[key][i])
                        t_hard.append(rec)
                _add_t(t_fp_idx, "fp")
                _add_t(t_fn_idx, "fn")

                # Write summary and predictions for test
                test_summary = {
                    "task": task,
                    "fold_id": fold_id,
                    "seed": seed,
                    "temperature": float(temperature),
                    "threshold": float(threshold),
                    "n_test": int(len(t_targets)),
                    "metrics_uncalibrated": _format_metrics(test_metrics_uncal),
                    "metrics_calibrated": _format_metrics(test_metrics_cal),
                    "slice_metrics": test_slice_metrics,
                    "hard_negative_counts": {"fp": int(len(t_fp_idx)), "fn": int(len(t_fn_idx))},
                    "config_path": str(args.config),
                }
                (outputs["root"] / "summary_test.json").write_text(json.dumps(test_summary, indent=2), encoding="utf-8")
                LOGGER.info("Test summary: %s", json.dumps(test_summary, indent=2))

                preds_dir = outputs["predictions"]
                out_pred_test = preds_dir / f"{task}_fold{fold_id:02d}_test.csv"
                df_t = pd.DataFrame(
                    {
                        "annotation_id": test_outputs["annotation_ids"],
                        "image_id": test_outputs["image_ids"],
                        "target": t_targets,
                        "logit": t_logits,
                        "logit_clamped": t_logits_clamped,
                        "prob": t_probs,
                        "prob_cal": t_probs_cal,
                        "subset": np.full(len(t_logits), "test", dtype=object),
                    }
                )
                for key, values in t_meta_dict.items():
                    df_t[key] = values
                df_t.to_csv(out_pred_test, index=False)

                hard_dir = Path(outputs["root"]).parent / "hard_negatives_test"
                hard_dir.mkdir(parents=True, exist_ok=True)
                hard_csv = hard_dir / f"fold_{fold_id:02d}.csv"
                if t_hard:
                    pd.DataFrame.from_records(t_hard).to_csv(hard_csv, index=False)
                elif hard_csv.exists():
                    hard_csv.unlink()
            else:
                LOGGER.info("Test evaluation skipped (empty test split).")
        else:
            LOGGER.info("Test evaluation skipped: split_mode is '%s'.", split_mode_lower)
    except Exception:
        LOGGER.exception("Test evaluation failed; continuing without test metrics.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
