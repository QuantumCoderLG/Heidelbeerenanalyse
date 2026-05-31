from __future__ import annotations

import logging
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn

LOGGER = logging.getLogger("training_utils")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: deterministic=True can hurt performance, so we often leave it False
    # unless strict reproducibility is required.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_model_real_float(model: nn.Module) -> None:
    """Check that model parameters are real floating point numbers."""
    bad: List[str] = []
    for name, param in model.named_parameters(recurse=True):
        if param.is_complex() or not torch.is_floating_point(param):
            bad.append(f"{name}:{param.dtype}")
    if bad:
        raise RuntimeError("Model has non-real-float parameters: " + ", ".join(bad))


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    """Atomic save of a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    shutil.move(tmp_path, path)


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    # Support both nested "optimizer" key or direct config
    opt_cfg = cfg.get("optimizer", cfg)
    name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    
    # Simple parameter grouping (all params)
    # If advanced parameter grouping is needed (like in train_segmentation), 
    # the caller might need to handle it or we need a more complex builder.
    # For now, we support the common case.
    params = model.parameters()
    
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(opt_cfg.get("nesterov", False)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, 
    cfg: Dict[str, Any], 
    max_epochs: int,
    steps_per_epoch: int = 1
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build LR scheduler from config."""
    sched_cfg = cfg.get("scheduler")
    if not sched_cfg:
        return None
    name = sched_cfg.get("name", "none").lower()
    
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(math.ceil(sched_cfg.get("t_max", max_epochs))),
            eta_min=float(sched_cfg.get("eta_min", 0.0)),
        )
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(sched_cfg.get("max_lr", 1e-3)),
            epochs=max_epochs,
            steps_per_epoch=max(1, steps_per_epoch),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 10)),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )
    if name == "poly":
        power = float(sched_cfg.get("power", 0.9))
        total_steps = max_epochs * steps_per_epoch
        def _lambda(step: int) -> float:
            frac = 1.0 - (step / float(total_steps))
            return float(max(0.0, frac) ** power)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lambda)
        
    return None
