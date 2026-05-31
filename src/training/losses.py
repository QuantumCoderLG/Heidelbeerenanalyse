from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    mode: str = "dice_bce"
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    focal_weight: float = 1.0
    smooth: float = 1e-5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = tuple(range(1, probs.ndim))
        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs, dims) + torch.sum(targets, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction type '{reduction}'")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        pt = pt.clamp_min(1e-6)
        focal_weight = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = focal_weight * alpha_factor
        loss = focal_weight * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, config: dict | LossConfig) -> None:
        super().__init__()
        if isinstance(config, LossConfig):
            cfg = config
        else:
            cfg = LossConfig(
                mode=str(config.get("mode", "dice_bce")),
                dice_weight=float(config.get("dice_weight", 1.0)),
                bce_weight=float(config.get("bce_weight", 1.0)),
                focal_weight=float(config.get("focal_weight", 1.0)),
                smooth=float(config.get("smooth", 1e-5)),
                focal_alpha=float(config.get("focal_alpha", 0.25)),
                focal_gamma=float(config.get("focal_gamma", 2.0)),
            )
        self.cfg = cfg
        self.dice = DiceLoss(smooth=cfg.smooth)
        self.focal = BinaryFocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
        self.mode = cfg.mode.lower()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        dice_term = self.dice(logits, targets)
        if self.mode == "dice_bce":
            bce_term = F.binary_cross_entropy_with_logits(logits, targets)
            loss = self.cfg.dice_weight * dice_term + self.cfg.bce_weight * bce_term
        elif self.mode == "focal_dice":
            focal_term = self.focal(logits, targets)
            loss = self.cfg.dice_weight * dice_term + self.cfg.focal_weight * focal_term
        else:
            raise ValueError(f"Unsupported loss mode: {self.mode}")
        return loss


def build_loss(config: dict) -> CombinedLoss:
    return CombinedLoss(config)


__all__ = ["build_loss", "CombinedLoss", "DiceLoss", "BinaryFocalLoss", "LossConfig"]
