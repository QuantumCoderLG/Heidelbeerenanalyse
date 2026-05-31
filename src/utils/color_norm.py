from __future__ import annotations

import numpy as np

__all__ = ["gray_world"]


def gray_world(
    image: np.ndarray,
    strength: float = 1.0,
    max_gain: float = 1.8,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply gray-world white balance with optional blending.

    Parameters
    ----------
    image:
        Input RGB image as ``uint8`` array of shape ``(H, W, 3)``.
    strength:
        Blend factor in ``[0, 1]``. ``1`` means full correction, ``0`` leaves
        the image untouched.
    max_gain:
        Clamp channel gains to ``[1 / max_gain, max_gain]`` to avoid extreme
        color shifts.
    eps:
        Numerical stability constant to avoid division by zero.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("gray_world expects an RGB image (H, W, 3).")
    if image.dtype != np.uint8:
        raise ValueError("gray_world expects uint8 input.")
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return image.copy()

    img_float = image.astype(np.float32)
    channel_means = img_float.reshape(-1, 3).mean(axis=0)
    overall_mean = channel_means.mean()
    gains = overall_mean / (channel_means + eps)
    if max_gain is not None and max_gain > 0:
        gains = np.clip(gains, 1.0 / max_gain, max_gain)

    corrected = img_float * gains.reshape(1, 1, 3)
    corrected = np.clip(corrected, 0.0, 255.0)

    if strength < 1.0:
        corrected = strength * corrected + (1.0 - strength) * img_float

    return corrected.round().astype(np.uint8)
