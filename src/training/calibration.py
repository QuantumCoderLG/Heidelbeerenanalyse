from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemperatureScalingResult:
    temperature: float
    loss: float


@dataclass
class TemperatureBiasScalingResult:
    temperature: float
    bias: float
    loss: float


class _LogTemperature(nn.Module):
    def __init__(self, init_log_temp: float = 0.0) -> None:
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([init_log_temp], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temp)
        return logits / temperature.clamp(min=1e-4)


@torch.no_grad()
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = torch.tensor(temperature, dtype=logits.dtype, device=logits.device)
    return logits / temp.clamp(min=1e-4)


@torch.no_grad()
def apply_temperature_bias(logits: torch.Tensor, temperature: float, bias: float = 0.0) -> torch.Tensor:
    """Scale logits by temperature and add optional bias: logits -> logits / T + b.

    The bias allows shifting the decision boundary after calibration to steer
    precision/recall trade-offs in a calibrated score space.
    """
    temp = torch.tensor(temperature, dtype=logits.dtype, device=logits.device)
    b = torch.tensor(bias, dtype=logits.dtype, device=logits.device)
    return logits / temp.clamp(min=1e-4) + b


def fit_temperature(
    logits: Sequence[float] | torch.Tensor,
    targets: Sequence[int] | torch.Tensor,
    *,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> TemperatureScalingResult:
    if not isinstance(logits, torch.Tensor):
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
    else:
        logits_tensor = logits.detach().float()
    if not isinstance(targets, torch.Tensor):
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
    else:
        targets_tensor = targets.detach().float()

    device = logits_tensor.device
    module = _LogTemperature().to(device)
    optimizer = torch.optim.LBFGS(module.parameters(), lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled_logits = module(logits_tensor)
        loss = F.binary_cross_entropy_with_logits(scaled_logits, targets_tensor, reduction="mean")
        loss.backward()
        return loss

    prev_loss = float("inf")
    for _ in range(max_iter):
        loss = optimizer.step(_closure)
        loss_value = float(loss.detach())
        if abs(prev_loss - loss_value) < tol:
            prev_loss = loss_value
            break
        prev_loss = loss_value

    temperature = float(torch.exp(module.log_temp).detach().cpu())
    return TemperatureScalingResult(temperature=temperature, loss=prev_loss)


def _bce_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def _ece_from_probs(probs: torch.Tensor, targets: torch.Tensor, num_bins: int = 15) -> torch.Tensor:
    # Simple ECE computation (torch) for tie-breaking
    bins = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=probs.device)
    bin_ids = torch.bucketize(probs, bins) - 1
    ece = torch.tensor(0.0, device=probs.device)
    total = probs.numel()
    for b in range(num_bins):
        mask = bin_ids == b
        count = int(mask.sum().item())
        if count == 0:
            continue
        conf = probs[mask].mean()
        acc = targets[mask].float().mean()
        ece = ece + (conf - acc).abs() * (count / max(1, total))
    return ece


def fit_temperature_bounded(
    logits: Sequence[float] | torch.Tensor,
    targets: Sequence[int] | torch.Tensor,
    *,
    min_temperature: float = 0.5,
    max_temperature: float = 5.0,
    bias_enabled: bool = False,
    bias_range: Tuple[float, float] = (-2.0, 2.0),
    n_temp: int = 41,
    n_bias: int = 41,
    tie_break_on_ece: bool = True,
) -> TemperatureBiasScalingResult:
    """Grid-search temperature (and optional bias) within bounds minimizing NLL.

    - Clamps temperature to [min_temperature, max_temperature]
    - If bias_enabled, searches bias in [bias_range[0], bias_range[1]]
    - Selects the pair (T, b) minimizing BCE loss; if tie within 1e-6,
      pick lower ECE (on probabilities) as tie-breaker.
    """
    if not isinstance(logits, torch.Tensor):
        logits_t = torch.tensor(logits, dtype=torch.float32)
    else:
        logits_t = logits.detach().float()
    if not isinstance(targets, torch.Tensor):
        targets_t = torch.tensor(targets, dtype=torch.float32)
    else:
        targets_t = targets.detach().float()

    device = logits_t.device
    Tmin = float(max(1e-4, min_temperature))
    Tmax = float(max(Tmin + 1e-4, max_temperature))
    temps = torch.linspace(Tmin, Tmax, steps=max(2, n_temp), device=device)

    if bias_enabled:
        bmin, bmax = bias_range
        bs = torch.linspace(float(bmin), float(bmax), steps=max(2, n_bias), device=device)
    else:
        bs = torch.tensor([0.0], device=device)

    best_T = float(1.0)
    best_b = float(0.0)
    best_loss = float("inf")
    best_ece = float("inf")

    for T in temps:
        scaled = logits_t / T.clamp(min=1e-4)
        if bias_enabled:
            for b in bs:
                loss = _bce_from_logits(scaled + b, targets_t)
                lv = float(loss.detach().cpu())
                ece = _ece_from_probs(torch.sigmoid(scaled + b), targets_t)
                ev = float(ece.detach().cpu())
                if lv < best_loss - 1e-6 or (abs(lv - best_loss) <= 1e-6 and (tie_break_on_ece and ev < best_ece - 1e-6)):
                    best_loss = lv
                    best_ece = ev
                    best_T = float(T.detach().cpu())
                    best_b = float(b.detach().cpu())
        else:
            loss = _bce_from_logits(scaled, targets_t)
            lv = float(loss.detach().cpu())
            ece = _ece_from_probs(torch.sigmoid(scaled), targets_t)
            ev = float(ece.detach().cpu())
            if lv < best_loss - 1e-6 or (abs(lv - best_loss) <= 1e-6 and (tie_break_on_ece and ev < best_ece - 1e-6)):
                best_loss = lv
                best_ece = ev
                best_T = float(T.detach().cpu())
                best_b = 0.0

    return TemperatureBiasScalingResult(temperature=best_T, bias=best_b, loss=best_loss)


__all__ = [
    "TemperatureScalingResult",
    "TemperatureBiasScalingResult",
    "fit_temperature",
    "fit_temperature_bounded",
    "apply_temperature",
    "apply_temperature_bias",
]
