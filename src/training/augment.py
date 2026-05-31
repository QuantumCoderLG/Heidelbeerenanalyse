from __future__ import annotations

from typing import Dict, Any, Sequence, Tuple

import cv2
import numpy as np


def _rand_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    if low == high:
        return float(low)
    return float(rng.uniform(low, high))


def _maybe(val: Any, default: Any) -> Any:
    return default if val is None else val


def _as_range(val: Any, fallback: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return float(val[0]), float(val[1])
    if isinstance(val, (int, float)):
        v = float(val)
        return -v, v
    return fallback


def _unsharp_mask(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    # img expected in [0,1]
    blur = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=max(0.01, sigma))
    sharp = (1.0 + amount) * img - amount * blur
    return np.clip(sharp, 0.0, 1.0)


def _clahe_rgb(img: np.ndarray, clip_limit: float, tile: int = 8) -> np.ndarray:
    # operate on L channel in LAB; img in [0,1]
    u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip_limit)), tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def _jitter_bc(img: np.ndarray, brightness: float, contrast: float, rng: np.random.Generator) -> np.ndarray:
    # brightness: additive range in [-b,b]; contrast: multiplicative in [1-c,1+c]
    b = _rand_uniform(rng, -abs(brightness), abs(brightness))
    c = _rand_uniform(rng, 1.0 - abs(contrast), 1.0 + abs(contrast))
    out = img * c + b
    return np.clip(out, 0.0, 1.0)


def _add_gaussian_noise(img: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0.0:
        return img
    noise = rng.normal(0.0, std, size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def apply_texture_augmentations(img_float: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Apply lightweight texture-focused augmentations to emphasise
    stems, mould (bright speckles) and dents (local contrast changes).

    Expects input RGB float image in [0,1]. Returns image in [0,1].
    Keys in cfg (all optional):
      - enabled (bool)
      - prob (float, 0..1) overall apply probability
      - brightness (float) additive jitter amplitude (default 0.08)
      - contrast (float) multiplicative jitter amplitude (default 0.12)
      - clahe_clip (float or [min,max]) default 1.5..3.0
      - clahe_prob (float) default 0.6
      - unsharp_amount (float or [min,max]) default 0.4..0.9
      - unsharp_sigma (float or [min,max]) default 0.6..1.2
      - unsharp_prob (float) default 0.7
      - noise_std (float or [min,max]) default 0.0..0.01
      - noise_prob (float) default 0.5
    """
    if not _maybe(cfg.get("enabled"), True):
        return np.clip(img_float, 0.0, 1.0)
    rng = np.random.default_rng()
    if rng.random() > float(_maybe(cfg.get("prob"), 1.0)):
        return np.clip(img_float, 0.0, 1.0)

    out = img_float.astype(np.float32)

    # Brightness/contrast jitter first
    out = _jitter_bc(
        out,
        brightness=float(_maybe(cfg.get("brightness"), 0.08)),
        contrast=float(_maybe(cfg.get("contrast"), 0.12)),
        rng=rng,
    )

    # CLAHE to highlight whitish mould and local contrast
    clahe_prob = float(_maybe(cfg.get("clahe_prob"), 0.6))
    if rng.random() < clahe_prob:
        c0, c1 = _as_range(cfg.get("clahe_clip"), (1.5, 3.0))
        clip = _rand_uniform(rng, min(c0, c1), max(c0, c1))
        out = _clahe_rgb(out, clip)

    # Unsharp mask to emphasise edges (stems, dents)
    if rng.random() < float(_maybe(cfg.get("unsharp_prob"), 0.7)):
        a0, a1 = _as_range(cfg.get("unsharp_amount"), (0.4, 0.9))
        s0, s1 = _as_range(cfg.get("unsharp_sigma"), (0.6, 1.2))
        amount = _rand_uniform(rng, min(a0, a1), max(a0, a1))
        sigma = _rand_uniform(rng, min(s0, s1), max(s0, s1))
        out = _unsharp_mask(out, sigma=sigma, amount=amount)

    # Low amplitude Gaussian noise to avoid oversmoothing
    if rng.random() < float(_maybe(cfg.get("noise_prob"), 0.5)):
        n0, n1 = _as_range(cfg.get("noise_std"), (0.0, 0.01))
        std = _rand_uniform(rng, min(n0, n1), max(n0, n1))
        out = _add_gaussian_noise(out, std=std, rng=rng)

    return np.clip(out, 0.0, 1.0)


__all__ = ["apply_texture_augmentations"]

def _gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0.0:
        return img
    inv = 1.0 / gamma
    out = np.power(np.clip(img, 0.0, 1.0), inv)
    return np.clip(out, 0.0, 1.0)


def _hsv_shift(img: np.ndarray, dh: float, ds: float, dv: float) -> np.ndarray:
    u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    # H in [0,179] OpenCV; shift in degrees scaled
    hsv[:, :, 0] = (hsv[:, :, 0] + dh * 179.0) % 179.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + ds), 0.0, 255.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + dv), 0.0, 255.0)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def apply_color_augmentations(img_float: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Color-focused augmentations for A3 (red vs not-red):
    hue/saturation/value shifts, mild gamma, small brightness jitter.
    Keys (optional):
      - prob (float)
      - hue (float or [min,max], in fraction of full hue cycle, default ±0.02)
      - sat (float or [min,max], multiplicative delta, default ±0.15)
      - val (float or [min,max], multiplicative delta, default ±0.08)
      - gamma (float or [min,max], default [0.9, 1.1])
      - brightness (float, additive, default 0.04)
    """
    rng = np.random.default_rng()
    if rng.random() > float(_maybe(cfg.get("prob"), 1.0)):
        return np.clip(img_float, 0.0, 1.0)

    out = img_float.astype(np.float32)
    h0, h1 = _as_range(cfg.get("hue"), (0.02, 0.02))
    s0, s1 = _as_range(cfg.get("sat"), (0.15, 0.15))
    v0, v1 = _as_range(cfg.get("val"), (0.08, 0.08))
    dh = _rand_uniform(rng, -abs(h0), abs(h1))
    ds = _rand_uniform(rng, -abs(s0), abs(s1))
    dv = _rand_uniform(rng, -abs(v0), abs(v1))
    out = _hsv_shift(out, dh=dh, ds=ds, dv=dv)

    g0, g1 = _as_range(cfg.get("gamma"), (0.9, 1.1))
    gamma = _rand_uniform(rng, min(g0, g1), max(g0, g1))
    out = _gamma(out, gamma)

    out = _jitter_bc(out, brightness=float(_maybe(cfg.get("brightness"), 0.04)), contrast=0.0, rng=rng)
    return np.clip(out, 0.0, 1.0)


__all__.extend(["apply_color_augmentations"])
