from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 1337,
    "data": {
        "root": "data/processed",
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "train_size": [1024, 1024],
        "val_size": [1024, 1024],
        "keep_ratio": True,
        "batch_size": 4,
        "val_batch_size": 4,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "drop_last": False,
        "cache_images": False,
    },
    "model": {
        "name": "deeplabv3plus",
        "backbone": "resnet50",
        "pretrained": True,
        "aux_loss": False,
        "output_stride": 16,
        "freeze_blocks": [],
        "aspp_dilate": None,
        "dropout": 0.1,
    },
    "loss": {
        "mode": "dice_bce",  # or focal_dice
        "dice_weight": 1.0,
        "bce_weight": 1.0,
        "focal_weight": 1.0,
        "smooth": 1e-5,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
    },
    "optimizer": {
        "name": "adam",
        "lr": 3e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "param_groups": {},
        "momentum": 0.9,  # kept for SGD compatibility
        "foreach": False,
        "fused": False,
    },
    "scheduler": {
        "name": "poly",
        "power": 0.9,
        "min_lr": 1e-6,
    },
    "train": {
        "max_epochs": 100,
        "gradient_accumulation_steps": 1,
        "amp": True,
        "clip_grad_norm": 1.0,
        "log_interval": 10,
        "val_interval": 1,
        "save_every": None,
        "checkpoint_dir": "outputs/checkpoints",
        "best_filename": "best_iou.pt",
        "last_filename": "last.pt",
        "resume_path": None,
        "resume_strict": False,
        "resume_optimizer": False,
        "resume_scheduler": False,
        "channels_last": True,
        "torch_compile": {
            "enabled": False,
            "mode": "reduce-overhead",
        },
        "backend": {
            "cudnn_benchmark": True,
            "allow_tf32": True,
            "matmul_precision": "high",
        },
        "early_stopping": {
            "enabled": True,
            "patience": 12,
            "min_delta": 5e-4,
        },
        "lr_find": {
            "enabled": True,
            "min_lr": 1e-6,
            "max_lr": 1e-2,
            "num_iters": 100,
            "warmup_iters": 5,
        },
    },
    "postproc": {
        "threshold": 0.5,
        "threshold_search": {
            "enabled": True,
            "min": 0.2,
            "max": 0.9,
            "num_steps": 15,
            "metric": "median_iou",  # or dice
        },
        "morphology": {
            "open_kernel": 3,
            "close_kernel": 5,
            "iterations": 1,
        },
        "min_area": 50,
        "max_area": None,
        "circularity": {
            "enabled": True,
            "min": 0.3,
        },
        "watershed": {
            "enabled": True,
            "distance_transform": "L2",
            "peak_rel_threshold": 0.4,
            "peak_min_distance": 5,
        },
        "count_guided": {
            "enabled": False,
            "target_count": 25,
            "tolerance": 1,
            "window": 0.2,
            "steps": 9,
        },
    },
    "cv": {
        "enabled": False,
        "num_folds": 5,
        "seed": 1337,
        "shuffle": True,
    },
    "logging": {
        "tensorboard_dir": "outputs/tensorboard",
        "csv_log": "outputs/logs/training.csv",
        "metrics_json": "outputs/logs/metrics.json",
        "level": "INFO",
    },
    "metrics": {
        "count_target": 25,
        "merge_iou_threshold": 0.1,
        "miss_iou_threshold": 0.2,
    },
}


class ConfigError(RuntimeError):
    """Raised when configuration loading fails."""


def _deep_update(dest: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dest.get(key), MutableMapping):
            _deep_update(dest[key], value)  # type: ignore[index]
        else:
            dest[key] = copy.deepcopy(value)
    return dest


def load_config(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)

    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")
        if yaml is None:
            raise ConfigError("PyYAML is required to load YAML configuration files.")
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, Mapping):
            raise ConfigError("Configuration root must be a mapping/dictionary.")
        _deep_update(config, loaded)

    if overrides:
        override_dict = _parse_overrides(overrides)
        _deep_update(config, override_dict)

    return config


def _parse_overrides(items: Iterable[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"Override '{item}' must be in key=value format.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ConfigError(f"Override '{item}' has empty key.")
        parsed_value: Any
        if raw_value.lower() in {"true", "false"}:
            parsed_value = raw_value.lower() == "true"
        else:
            try:
                parsed_value = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed_value = raw_value
        _assign_override(merged, key.split("."), parsed_value)
    return merged


def _assign_override(target: MutableMapping[str, Any], path: Iterable[str], value: Any) -> None:
    keys = list(path)
    if not keys:
        raise ConfigError("Override path cannot be empty.")
    current = target
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], MutableMapping):
            current[key] = {}
        current = current[key]  # type: ignore[index]
    current[keys[-1]] = value


from . import paths


__all__ = ["load_config", "ConfigError", "DEFAULT_CONFIG", "paths"]
