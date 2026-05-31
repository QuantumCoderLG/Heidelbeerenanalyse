from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Optional import guard for onnxruntime to keep a clear error message
try:
    import onnxruntime as ort  # type: ignore
except Exception as exc:  # pragma: no cover
    ort = None  # type: ignore
    _ORT_IMPORT_ERROR = exc
else:
    _ORT_IMPORT_ERROR = None


# ---------------------------- Utils & Dataclasses ----------------------------


@dataclass
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def h(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return int(self.w * self.h)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ensure_ort():
    if ort is None:
        raise RuntimeError(
            f"onnxruntime is required but could not be imported: {_ORT_IMPORT_ERROR}\n"
            "Install with: pip install onnxruntime"
        )


def _load_json(path: Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _letterbox_image(img: np.ndarray, target_hw: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, int]]:
    th, tw = int(target_hw[0]), int(target_hw[1])
    h, w = img.shape[:2]
    scale = min(th / max(1, h), tw / max(1, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h = th - nh
    pad_w = tw - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    meta = {"top": top, "bottom": bottom, "left": left, "right": right, "new_h": nh, "new_w": nw}
    return out, meta


def _letterbox_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = int(target_hw[0]), int(target_hw[1])
    h, w = mask.shape[:2]
    scale = min(th / max(1, h), tw / max(1, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    pad_h = th - nh
    pad_w = tw - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return out


def _normalize(img_float: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    return (img_float - mean_arr) / std_arr


def _to_chw(img_float: np.ndarray) -> np.ndarray:
    return np.transpose(img_float, (2, 0, 1)).astype(np.float32)


def _apply_morphology(binary: np.ndarray, open_k: int = 3, close_k: int = 5, iterations: int = 1) -> np.ndarray:
    out = binary.astype(np.uint8)
    if open_k and open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=iterations)
    if close_k and close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=iterations)
    return out


def _connected_components(binary: np.ndarray) -> np.ndarray:
    num, labels = cv2.connectedComponents(binary.astype(np.uint8))
    if num <= 1:
        return np.zeros_like(binary, dtype=np.int32)
    return labels.astype(np.int32)


def _apply_watershed(binary: np.ndarray, prob_map: np.ndarray, *, rel_thresh: float = 0.35, min_distance: int = 1) -> np.ndarray:
    """Match src watershed behavior using distance transform peaks.

    - Threshold distance map at rel_thresh * max(distance)
    - Erode foreground markers by min_distance (iterations)
    - Run watershed on a 3-channel normalized prob_map
    - Convert boundaries and background to 0
    """
    if binary.max() == 0:
        return np.zeros_like(binary, dtype=np.int32)
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance, float(rel_thresh) * float(distance.max()), 1, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_fg = cv2.erode(sure_fg, np.ones((3, 3), np.uint8), iterations=int(min_distance))
    unknown = cv2.subtract(binary.astype(np.uint8), sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers.astype(np.int32)
    markers += 1
    markers[unknown == 1] = 0
    img = cv2.normalize(prob_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    cv2.watershed(img, markers)
    markers[markers < 0] = 0
    markers[markers == 1] = 0
    return markers.astype(np.int32)


def _filter_components(instances: np.ndarray, *, min_area: int = 30, circ_min: float = 0.25) -> np.ndarray:
    out = np.zeros_like(instances, dtype=np.int32)
    next_id = 1
    for label in np.unique(instances):
        if int(label) <= 0:
            continue
        mask = (instances == int(label)).astype(np.uint8)
        area = int(mask.sum())
        if area < min_area:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        per = float(cv2.arcLength(c, True))
        circularity = (4.0 * math.pi * float(area) / (per ** 2)) if per > 0 else 0.0
        if circularity < circ_min:
            continue
        out[instances == int(label)] = next_id
        next_id += 1
    return out


def _compute_shape_metrics(mask: np.ndarray) -> Tuple[float, float, float]:
    if mask is None or mask.size == 0:
        return float('nan'), float('nan'), float('nan')
    area = int(mask.sum())
    if area <= 0:
        return float('nan'), float('nan'), float('nan')
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return float('nan'), float('nan'), float('nan')
    c = max(contours, key=cv2.contourArea)
    per = float(cv2.arcLength(c, True))
    circularity = (4.0 * math.pi * float(area) / (per ** 2)) if per > 0 else float('nan')
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else float('nan')
    pts = c.reshape(-1, 2).astype(np.float32)
    m = cv2.moments(c)
    if m.get("m00", 0.0) > 0:
        cx = float(m["m10"] / m["m00"])  # type: ignore[index]
        cy = float(m["m01"] / m["m00"])  # type: ignore[index]
    else:
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    rad = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    rad_mean = float(rad.mean()) if rad.size > 0 else 0.0
    rad_std = float(rad.std(ddof=0)) if rad.size > 0 else 0.0
    rough = (rad_std / max(1e-9, rad_mean)) if rad_mean > 0 else float('nan')
    return float(circularity), float(solidity), float(rough)


def _evaluate_classical(features: Dict[str, float], rules: Dict[str, Dict]) -> Tuple[str, Dict[str, int | float | str]]:
    supports: Dict[str, int] = {}
    reasons: List[str] = []
    first_reason: str | None = None
    first_reason_value: float | None = None
    first_reason_feature: str | None = None
    for name, rule in rules.items():
        kind = str(rule.get("kind"))
        low = rule.get("low")
        high = rule.get("high")
        v = float(features.get(name, float("nan")))
        ok = False
        reason: str | None = None
        if not np.isfinite(v):
            ok = False
            reason = f"invalid_{name}"
        elif kind == "range":
            assert low is not None and high is not None
            ok = (v >= float(low)) and (v <= float(high))
            if not ok:
                if name in ("relative_size", "deq"):
                    reason = "too_small" if v < float(low) else "too_large"
                else:
                    reason = f"{name}_{'low' if v < float(low) else 'high'}"
        elif kind == "min":
            assert low is not None
            ok = v >= float(low)
            if not ok:
                reason = f"{name}_low"
        elif kind == "max":
            assert high is not None
            ok = v <= float(high)
            if not ok:
                reason = f"{name}_high"
        else:
            ok = False
            reason = f"invalid_rule_{name}"
        supports[name] = int(ok)
        if reason:
            reasons.append(reason)
            if first_reason is None:
                first_reason = reason
                first_reason_value = float(v) if np.isfinite(v) else None
                first_reason_feature = name
    support_count = int(sum(supports.values()))
    # Neue Logik: Nur Größe + Rundheit zählen. Jeder einzelne Verstoß führt sofort zu "yellow".
    pred = "yellow" if reasons else "green"
    decision_reason = first_reason or ("ok" if pred == "green" else "unknown")
    return pred, {
        **features,
        **{f"support_{k}": int(v) for k, v in supports.items()},
        "support_count": support_count,
        "decision_reason": decision_reason,
        "first_reason": first_reason or "",
        "first_reason_value": first_reason_value if first_reason_value is not None else float("nan"),
        "first_reason_feature": first_reason_feature or "",
    }


# ---------------------------- Inference Pipeline -----------------------------


class OnnxModel:
    def __init__(self, model_path: Path) -> None:
        _ensure_ort()
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        sess_opts = ort.SessionOptions()
        self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])  # type: ignore
        self.input_name = self.session.get_inputs()[0].name
        # Use first output by default
        self.output_name = self.session.get_outputs()[0].name

    def run(self, x: np.ndarray) -> np.ndarray:
        # x must be (N,C,H,W)
        outputs = self.session.run([self.output_name], {self.input_name: x})
        y = outputs[0]
        return y


class InferenceCore:
    def __init__(self, assets_root: Path) -> None:
        self.assets_root = Path(assets_root)
        manifest_path = self.assets_root / "manifest.json"
        # Fallback for PyInstaller one-file builds: assets embedded under sys._MEIPASS/inference_assets
        if not manifest_path.exists():
            try:
                if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
                    base = Path(getattr(sys, "_MEIPASS"))  # type: ignore[attr-defined]
                    candidate = base / "inference_assets"
                    if (candidate / "manifest.json").exists():
                        self.assets_root = candidate
                        manifest_path = self.assets_root / "manifest.json"
            except Exception:
                pass
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found under {self.assets_root}")
        self.manifest = _load_json(manifest_path)
        models = self.manifest.get("models", {})
        self.seg_path = self.assets_root / models.get("segmentation", {}).get("file", "models/segmentation.onnx")

        # Load thresholds (segmentation + classifier heads + classical rules) from dedicated file
        here = Path(__file__).resolve().parent
        thresholds_candidates = [
            here / "Thresholds.json",
            here / "thresholds.json",
            here.parent / "Thresholds.json",
            here.parent / "thresholds.json",
        ]
        self.thresholds_path: Optional[Path] = None
        self.thresholds: Dict[str, Any] = {}
        for cand in thresholds_candidates:
            if cand.exists():
                try:
                    self.thresholds = _load_json(cand)
                    self.thresholds_path = cand
                    break
                except Exception:
                    continue
        if not self.thresholds:
            base_candidates = [
                here / "configs" / "thresholds_base.json",
                here.parent / "configs" / "thresholds_base.json",
            ]
            for base_cfg in base_candidates:
                if not base_cfg.exists():
                    continue
                try:
                    self.thresholds = _load_json(base_cfg)
                except Exception:
                    continue
                self.thresholds_path = base_cfg
                break

        if not self.thresholds:
            # Fallback to legacy manifest entry to remain backwards compatible
            self.thresholds = self.manifest.get("thresholds", {})
            if not self.thresholds:
                raise FileNotFoundError(
                    "Keine thresholds.json gefunden - bitte stelle sicher, dass Thresholds.json gepflegt ist."
                )
        self.seg_threshold = float(self.thresholds.get("segmentation", {}).get("threshold", 0.5))

        # Preprocessing for segmentation
        pre = self.manifest.get("preprocessing", {})
        seg_pre = pre.get("segmentation", {}) if isinstance(pre, dict) else {}
        self.seg_mean = seg_pre.get("mean", [0.485, 0.456, 0.406])
        self.seg_std = seg_pre.get("std", [0.229, 0.224, 0.225])
        self.seg_target_hw = tuple(seg_pre.get("target_hw", [1024, 1024]))  # type: ignore
        self.seg_keep_ratio = bool(seg_pre.get("keep_ratio", True))

        # Build classifier registry dynamically (supports optional A4, future heads)
        self.classifiers: Dict[str, Dict[str, Any]] = {}
        for name in ("a1", "a2", "a3", "a4"):
            model_cfg = models.get(name)
            if not isinstance(model_cfg, dict):
                continue
            model_path = self.assets_root / model_cfg.get("file", f"models/{name}_classifier.onnx")
            pre_cfg = pre.get(name, {}) if isinstance(pre, dict) else {}
            thr_cfg = self.thresholds.get(name, {}) if isinstance(self.thresholds, dict) else {}
            size = tuple(pre_cfg.get("input_size", model_cfg.get("input_size", [320, 320])))
            mean = pre_cfg.get("mean", [0.485, 0.456, 0.406])
            std = pre_cfg.get("std", [0.229, 0.224, 0.225])
            in_ch = int(model_cfg.get("in_channels", len(mean)))
            include_mask = bool(pre_cfg.get("include_mask_channel", True))
            mask_usage = str(pre_cfg.get("mask_usage", "auto")).lower()
            if mask_usage == "rgb_only":
                include_mask = False
            elif mask_usage == "mask_channel":
                include_mask = True
            color_norm = str(pre_cfg.get("color_norm", "")).lower()
            feats = pre_cfg.get("color_features", [])
            if not feats:
                alt_key = f"color_features_{name}"
                feats = pre_cfg.get(alt_key, [])
            if isinstance(feats, list):
                color_features = [str(f) for f in feats]
            else:
                color_features = []
            self.classifiers[name] = {
                "model": OnnxModel(model_path),
                "size": tuple(size),
                "in_ch": in_ch,
                "mean": list(mean),
                "std": list(std),
                "include_mask": include_mask,
                "color_norm": color_norm,
                "color_features": color_features,
                "threshold": float(thr_cfg.get("threshold", 0.5)),
                "temperature": float(thr_cfg.get("temperature", 1.0)),
            }
        self.classifier_order = [name for name in ("a1", "a2", "a3", "a4") if name in self.classifiers]

        # Load classical rules from Thresholds.json (must live im App-Hauptordner)
        classical_cfg: Dict[str, Any] = {}
        if isinstance(self.thresholds, dict):
            maybe_classical = self.thresholds.get("classical_rules")
            if isinstance(maybe_classical, dict):
                classical_cfg = maybe_classical

        if not classical_cfg:
            raise FileNotFoundError(
                "Keine klassischen Thresholds gefunden. Bitte pflege den classical_rules-Block in der Thresholds.json "
                "im App-Hauptordner (z. B. Heidelbeeren-Bewertung-App/Thresholds.json)."
            )

        self.classical = classical_cfg
        self.classical_rules = (self.classical or {}).get("rules", {})
        override_cfg: Dict[str, object] = {}
        if isinstance(self.classical, dict):
            override_cfg = dict(self.classical.get("override", {}) or {})
        self.classical_override_cfg = override_cfg
        raw_deq_override = override_cfg.get("deq_min_for_yellow_override") if override_cfg else None
        try:
            deq_override = float(raw_deq_override) if raw_deq_override is not None else math.inf
        except (TypeError, ValueError):
            deq_override = math.inf
        self.classical_override_deq = deq_override if math.isfinite(deq_override) else math.inf
        self.classical_override_reason = str(override_cfg.get("reason", "too_large")) if override_cfg else "too_large"
        raw_circ_override = override_cfg.get("circularity_min_for_yellow_override") if override_cfg else None
        try:
            circ_override = float(raw_circ_override) if raw_circ_override is not None else math.nan
        except (TypeError, ValueError):
            circ_override = math.nan
        self.classical_override_circ = circ_override if math.isfinite(circ_override) else math.nan
        self.classical_override_circ_reason = str(override_cfg.get("circularity_reason", "too_irregular")) if override_cfg else "too_irregular"
        self.classical_has_override = math.isfinite(self.classical_override_deq) or math.isfinite(self.classical_override_circ)

        # ONNX sessions
        self.seg_model = OnnxModel(self.seg_path)
        # cache for GUI overlay rendering
        self._last: Dict[str, object] | None = None

        # Postprocessing configuration (align with src defaults)
        post = self.manifest.get("postprocessing", {}) if isinstance(self.manifest, dict) else {}
        morph = post.get("morphology", {}) if isinstance(post, dict) else {}
        self.morph_open = int(morph.get("open_kernel", 3))
        self.morph_close = int(morph.get("close_kernel", 5))
        self.morph_iter = int(morph.get("iterations", 1))
        self.min_area = int(post.get("min_area", 30)) if isinstance(post, dict) else 30
        circ_cfg = post.get("circularity", {}) if isinstance(post, dict) else {}
        self.circ_enabled = bool(circ_cfg.get("enabled", True))
        self.circ_min = float(circ_cfg.get("min", 0.25))
        ws = post.get("watershed", {}) if isinstance(post, dict) else {}
        self.watershed_enabled = bool(ws.get("enabled", True))
        self.ws_rel_thresh = float(ws.get("peak_rel_threshold", 0.35))
        self.ws_min_distance = int(ws.get("peak_min_distance", 1))

    # ----------------------------- Segmentation -----------------------------

    def segment(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_float = image_rgb.astype(np.float32) / 255.0
        if self.seg_keep_ratio:
            target_hw = self.seg_target_hw
            inp, meta = _letterbox_image(img_float, target_hw)
        else:
            th, tw = self.seg_target_hw
            inp = cv2.resize(img_float, (int(tw), int(th)), interpolation=cv2.INTER_LINEAR)
            meta = {"top": 0, "bottom": 0, "left": 0, "right": 0, "new_h": int(th), "new_w": int(tw)}
        norm = _normalize(inp, self.seg_mean, self.seg_std)
        x = _to_chw(norm)[None, :, :, :]  # (1,3,H,W)
        y = self.seg_model.run(x)
        # y: (1,1,H,W) logits or probs depending on export; our export is logits
        if y.ndim == 4:
            y = y.squeeze(0).squeeze(0)
        probs = _sigmoid(y.astype(np.float32))
        # Unpad back to resized content, then resize to original
        h, w = probs.shape
        y0 = meta.get("top", 0)
        y1 = h - meta.get("bottom", 0)
        x0 = meta.get("left", 0)
        x1 = w - meta.get("right", 0)
        cropped = probs[int(y0):int(y1), int(x0):int(x1)]
        H, W = image_rgb.shape[:2]
        prob_resized = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)
        # Postproc: threshold + morphology + watershed/cc + filters
        binary = (prob_resized >= float(self.seg_threshold)).astype(np.uint8)
        binary = _apply_morphology(binary, open_k=self.morph_open, close_k=self.morph_close, iterations=self.morph_iter)
        if self.watershed_enabled:
            instances = _apply_watershed(binary, prob_resized, rel_thresh=self.ws_rel_thresh, min_distance=self.ws_min_distance)
        else:
            instances = _connected_components(binary)
        circ_min = float(self.circ_min if self.circ_enabled else 0.0)
        instances = _filter_components(instances, min_area=int(self.min_area), circ_min=circ_min)
        binary = (instances > 0).astype(np.uint8)
        return binary, instances

    # ----------------------------- Overlays (cached) ------------------------

    def get_overlay(self, mode: str = "final") -> Optional[Image.Image]:
        return build_cached_overlay(self._last, mode)

    # ----------------------------- Classification ---------------------------

    def _run_classifier(self, which: str, crop_rgb: np.ndarray, mask_crop: np.ndarray | None) -> float:
        cfg = self.classifiers.get(which)
        if cfg is None:
            raise KeyError(f"Classifier '{which}' is not available in the loaded assets.")
        Ht, Wt = int(cfg["size"][0]), int(cfg["size"][1])
        inp_rgb, _ = _letterbox_image(crop_rgb, (Ht, Wt))
        # Optional gray-world normalization for color-sensitive heads
        if str(cfg.get("color_norm", "")).lower() == "gray_world":
            try:
                img8 = inp_rgb.astype(np.uint8)
                ch_means = img8.reshape(-1, 3).mean(axis=0).astype(np.float32)
                overall = float(ch_means.mean())
                gains = overall / (ch_means + 1e-6)
                gains = np.clip(gains, 1.0 / 1.8, 1.8)
                corrected = np.clip(img8.astype(np.float32) * gains.reshape(1, 1, 3), 0.0, 255.0)
                corrected = 0.8 * corrected + 0.2 * img8.astype(np.float32)
                inp_rgb = corrected.astype(np.uint8)
            except Exception:
                pass
        rgb_float = inp_rgb.astype(np.float32) / 255.0
        mean = cfg.get("mean", [0.485, 0.456, 0.406])
        std = cfg.get("std", [0.229, 0.224, 0.225])
        norm = _normalize(rgb_float, mean, std)
        x = _to_chw(norm)

        # Optional auxiliary color features (match training dataset behaviour)
        feats = cfg.get("color_features", []) or []
        if feats:
            extras: List[np.ndarray] = []
            rgb01 = np.clip(rgb_float, 0.0, 1.0)
            hsv_cache: Optional[np.ndarray] = None
            for feat in feats:
                key = str(feat).lower()
                if key == "redness":
                    r = rgb01[:, :, 0]
                    g = rgb01[:, :, 1]
                    b = rgb01[:, :, 2]
                    redness = np.clip(r - np.maximum(g, b), 0.0, 1.0)
                    extras.append(redness.astype(np.float32))
                elif key == "darkness":
                    if hsv_cache is None:
                        hsv_cache = cv2.cvtColor((rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                    v = hsv_cache[:, :, 2] / 255.0
                    extras.append((1.0 - v).astype(np.float32))
                elif key == "hsv":
                    if hsv_cache is None:
                        hsv_cache = cv2.cvtColor((rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                    h = hsv_cache[:, :, 0] / 179.0
                    s = hsv_cache[:, :, 1] / 255.0
                    v = hsv_cache[:, :, 2] / 255.0
                    extras.extend([
                        h.astype(np.float32),
                        s.astype(np.float32),
                        v.astype(np.float32),
                    ])
                else:
                    continue
            if extras:
                extra_stack = np.stack(extras, axis=2).astype(np.float32)
                extra_norm = (extra_stack - 0.5) / 0.5
                x = np.concatenate([x, np.transpose(extra_norm, (2, 0, 1))], axis=0)

        # Optional mask as extra channel
        if bool(cfg.get("include_mask", False)):
            if mask_crop is None:
                mask_resized = np.zeros((Ht, Wt), dtype=np.float32)
            else:
                mask_resized = _letterbox_mask(mask_crop.astype(np.uint8), (Ht, Wt))
            mask_float = (mask_resized > 0).astype(np.float32)
            x = np.concatenate([x, mask_float[None, :, :]], axis=0)

        # Align channel count with exported model
        in_ch = int(cfg.get("in_ch", x.shape[0]))
        if x.shape[0] > in_ch:
            x = x[:in_ch, :, :]
        elif x.shape[0] < in_ch:
            pad = np.zeros((in_ch - x.shape[0], x.shape[1], x.shape[2]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)

        x = x[None, :, :, :]
        model: OnnxModel = cfg["model"]  # type: ignore[assignment]
        logits = model.run(x)
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        logit = float(logits[0])
        temp = float(cfg.get("temperature", 1.0))
        if temp and np.isfinite(temp) and temp > 0:
            logit = logit / temp
        prob = float(_sigmoid(np.array([logit], dtype=np.float32))[0])
        return prob

    # ----------------------------- Classical Voting -------------------------

    def _classical_predict(
        self,
        crop_rgb: np.ndarray,
        mask_crop: np.ndarray,
        *,
        area_px_full: int,
        image_area: int,
    ) -> Tuple[str, Dict[str, float | int | str]]:
        # area_px_full: Fläche der Instanz im Originalbild (absolute Pixel)
        # image_area: Gesamtfläche des Originalbilds
        deq = 2.0 * float(np.sqrt(area_px_full / math.pi)) if area_px_full > 0 else float("nan")
        circ, _, _ = _compute_shape_metrics(mask_crop)
        feats = {
            "deq": deq,
            "circularity": float(circ),
        }
        return _evaluate_classical(feats, self.classical_rules)

    # ----------------------------- Orchestration ----------------------------

    def run_on_image(
        self,
        image_path: Path,
        *,
        margin: float = 0.15,
        top_k: int = 50,
        progress_cb: Optional[Callable[[str], None]] = None,
        overlay_cb: Optional[Callable[[Image.Image], None]] = None,
    ) -> Tuple[Image.Image, Dict[str, object]]:
        img = Image.open(image_path).convert("RGB")
        image_rgb = np.asarray(img, dtype=np.uint8)

        if progress_cb:
            progress_cb("Segmentiere…")
        binary, instances = self.segment(image_rgb)
        H, W = image_rgb.shape[:2]

        labels = [int(x) for x in np.unique(instances) if int(x) != 0]
        if not labels:
            # Keine Instanzen: kein Gesamtbild-Label, nur Originalbild als Overlay
            overlay = _draw_boxes_overlay(image_rgb, [], title=None)
            meta = {"instances": 0, "results": []}
            return overlay, meta

        if progress_cb:
            progress_cb(f"Instanzen gefunden: {len(labels)} – extrahiere Kandidaten…")

        # Compute bboxes per instance
        areas: List[Tuple[int, int]] = []
        area_map: Dict[int, int] = {}
        for lbl in labels:
            area = int((instances == lbl).sum())
            areas.append((area, lbl))
            area_map[lbl] = area
        areas.sort(key=lambda t: t[0], reverse=True)
        selected = [lbl for _, lbl in areas[: max(1, int(top_k))]]

        results: List[Dict[str, object]] = []
        has_a1 = "a1" in self.classifiers
        has_a2 = "a2" in self.classifiers
        has_a3 = "a3" in self.classifiers
        has_a4 = "a4" in self.classifiers
        thr_a1 = float(self.classifiers["a1"]["threshold"]) if has_a1 else None
        thr_a2 = float(self.classifiers["a2"]["threshold"]) if has_a2 else None
        thr_a3 = float(self.classifiers["a3"]["threshold"]) if has_a3 else None
        thr_a4 = float(self.classifiers["a4"]["threshold"]) if has_a4 else None
        step_a1: List[Dict[str, object]] = []
        step_a2: List[Dict[str, object]] = []
        step_a3: List[Dict[str, object]] = []
        step_a4: List[Dict[str, object]] = []
        first_a1 = has_a1
        first_a2 = has_a2
        first_a3 = has_a3
        first_a4 = has_a4
        use_classical = (not has_a4) and bool(self.classical_rules or self.classical_has_override)
        first_classic = use_classical

        def _snapshot() -> Dict[str, object]:
            cache: Dict[str, object] = {
                "image_rgb": image_rgb,
                "binary": binary,
                "instances": instances,
                "results": list(results),
            }
            if has_a1:
                cache["a1"] = list(step_a1)
            if has_a2:
                cache["a2"] = list(step_a2)
            if has_a3:
                cache["a3"] = list(step_a3)
            if has_a4:
                cache["a4"] = list(step_a4)
            return cache

        def _format_classical_display(info: Optional[Dict[str, object]], rules: Optional[Dict[str, Dict[str, object]]] = None) -> Optional[str]:
            if not info:
                return None
            reason = info.get("first_reason") or info.get("decision_reason")
            val = info.get("first_reason_value")
            feat = str(info.get("first_reason_feature") or "").strip()
            rule = (rules or {}).get(feat, {}) if rules else {}
            if reason:
                # Format Größe und Rundheit verständlicher
                if feat == "relative_size":
                    try:
                        val_f = float(val)
                        if np.isfinite(val_f):
                            max_rel = rule.get("high")
                            min_rel = rule.get("low")
                            parts = [f"{reason} size={val_f * 100:.2f}%"]
                            if np.isfinite(max_rel) or np.isfinite(min_rel):
                                if reason == "too_large" and max_rel is not None:
                                    parts.append(f"(max {float(max_rel) * 100:.2f}%)")
                                if reason == "too_small" and min_rel is not None:
                                    parts.append(f"(min {float(min_rel) * 100:.2f}%)")
                            return " ".join(parts)
                    except Exception:
                        pass
                if feat == "deq":
                    try:
                        val_f = float(val)
                        if np.isfinite(val_f):
                            max_d = rule.get("high")
                            min_d = rule.get("low")
                            parts = [f"{reason} deq={val_f:.1f}px"]
                            if np.isfinite(max_d) or np.isfinite(min_d):
                                if reason == "too_large" and max_d is not None:
                                    parts.append(f"(max {float(max_d):.1f}px)")
                                if reason == "too_small" and min_d is not None:
                                    parts.append(f"(min {float(min_d):.1f}px)")
                            return " ".join(parts)
                    except Exception:
                        pass
                try:
                    val_f = float(val)
                    if np.isfinite(val_f):
                        return f"{reason} {val_f:.3f}"
                except Exception:
                    pass
                return str(reason)
            return None

        for lbl in selected:
            ys, xs = np.where(instances == lbl)
            if ys.size == 0 or xs.size == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
            size = max(bw, bh)
            pad = int(round(size * float(margin)))
            x0p = max(0, x0 - pad)
            y0p = max(0, y0 - pad)
            x1p = min(W, x1 + 1 + pad)
            y1p = min(H, y1 + 1 + pad)
            box = BBox(x0p, y0p, x1p, y1p)
            box_area = int(max(0, box.w) * max(0, box.h))
            crop = image_rgb[box.y0:box.y1, box.x0:box.x1, :]
            mask_crop = (instances[box.y0:box.y1, box.x0:box.x1] == lbl).astype(np.uint8)
            area_px_full = int(area_map.get(lbl, int(mask_crop.sum())))
            image_area = int(H * W)

            # A1: notberry?
            if has_a1:
                if first_a1 and progress_cb:
                    progress_cb("A1: notberry bewerten…")
                    first_a1 = False
                p_a1 = self._run_classifier("a1", crop, mask_crop)
                step_a1.append({
                    "bbox": (box.x0, box.y0, box.x1, box.y1),
                    "prob": p_a1,
                    "decision": ("notberry" if p_a1 >= float(thr_a1) else "berry"),
                })
                if p_a1 >= float(thr_a1):
                    results.append({
                        "label": "unbekannt",
                        "prob": p_a1,
                        "bbox": (box.x0, box.y0, box.x1, box.y1),
                        "bbox_area": box_area,
                        "source": "a1",
                    })
                    self._last = _snapshot()
                    if overlay_cb:
                        try:
                            overlay_cb(_draw_boxes_overlay(image_rgb, results, title=None))
                        except Exception:
                            pass
                    continue

            # A2: never?
            if has_a2:
                if first_a2 and progress_cb:
                    progress_cb("A2: 'Never' bewerten…")
                    first_a2 = False
                p_a2 = self._run_classifier("a2", crop, mask_crop)
                step_a2.append({
                    "bbox": (box.x0, box.y0, box.x1, box.y1),
                    "prob": p_a2,
                    "decision": ("Never" if p_a2 >= float(thr_a2) else "ok"),
                })
                if p_a2 >= float(thr_a2):
                    results.append({
                        "label": "Never",
                        "prob": p_a2,
                        "bbox": (box.x0, box.y0, box.x1, box.y1),
                        "bbox_area": box_area,
                        "source": "a2",
                    })
                    self._last = _snapshot()
                    if overlay_cb:
                        try:
                            overlay_cb(_draw_boxes_overlay(image_rgb, results, title=None))
                        except Exception:
                            pass
                    continue

            # A3: red?
            if has_a3:
                if first_a3 and progress_cb:
                    progress_cb("A3: 'Red' bewerten…")
                    first_a3 = False
                p_a3 = self._run_classifier("a3", crop, mask_crop)
                step_a3.append({
                    "bbox": (box.x0, box.y0, box.x1, box.y1),
                    "prob": p_a3,
                    "decision": ("Red" if p_a3 >= float(thr_a3) else "not-red"),
                })
                if p_a3 >= float(thr_a3):
                    results.append({
                        "label": "Red",
                        "prob": p_a3,
                        "bbox": (box.x0, box.y0, box.x1, box.y1),
                        "bbox_area": box_area,
                        "source": "a3",
                    })
                    self._last = _snapshot()
                    if overlay_cb:
                        try:
                            overlay_cb(_draw_boxes_overlay(image_rgb, results, title=None))
                        except Exception:
                            pass
                    continue

            # A4: green vs yellow?
            if has_a4:
                if first_a4 and progress_cb:
                    progress_cb("A4: Green/Yellow bewerten…")
                    first_a4 = False
                prob_green = self._run_classifier("a4", crop, mask_crop)
                prob_yellow = 1.0 - prob_green
                is_green = prob_green >= float(thr_a4)
                raw_decision = "Green" if is_green else "Yellow"
                decision = raw_decision
                classical_pred: Optional[str] = None
                classical_info: Optional[Dict[str, float | int | str]] = None
                classical_override = False
                deq_value = float("nan")
                circ_value = float("nan")
                classical_reason: Optional[str] = None
                classical_display: Optional[str] = None
                if self.classical_rules or self.classical_has_override:
                    classical_pred, classical_info = self._classical_predict(
                        crop, mask_crop, area_px_full=area_px_full, image_area=image_area
                    )
                    if classical_info:
                        raw_deq = classical_info.get("deq", float("nan"))
                        raw_circ = classical_info.get("circularity", float("nan"))
                        if raw_deq is None:
                            deq_value = float("nan")
                        else:
                            try:
                                deq_value = float(raw_deq)
                            except (TypeError, ValueError):
                                deq_value = float("nan")
                        try:
                            circ_value = float(raw_circ)
                        except (TypeError, ValueError):
                            circ_value = float("nan")
                        reason = classical_info.get("decision_reason")
                        if isinstance(reason, str):
                            classical_reason = reason
                    # Klassik-Regel (sofern vorhanden): jedes Signal -> Yellow
                    if classical_pred == "yellow":
                        decision = "Yellow"
                        classical_override = True
                        classical_display = _format_classical_display(classical_info, self.classical_rules) if classical_info else None
                    # Harte Größe-Override
                    if (
                        decision == "Green"
                        and math.isfinite(self.classical_override_deq)
                        and math.isfinite(deq_value)
                        and deq_value >= self.classical_override_deq
                    ):
                        decision = "Yellow"
                        classical_override = True
                        if not classical_reason:
                            classical_reason = self.classical_override_reason
                        classical_display = (
                            classical_display
                            or f"{self.classical_override_reason} deq={deq_value:.1f}px (max {self.classical_override_deq:.1f}px)"
                        )
                    # Harte Rundheits-Override
                    if (
                        decision == "Green"
                        and math.isfinite(self.classical_override_circ)
                        and math.isfinite(circ_value)
                        and circ_value < self.classical_override_circ
                    ):
                        decision = "Yellow"
                        classical_override = True
                        if not classical_reason:
                            classical_reason = self.classical_override_circ_reason
                        classical_display = (
                            classical_display
                            or f"{self.classical_override_circ_reason} circ={circ_value:.3f} (min {self.classical_override_circ:.3f})"
                        )
                prob_display = prob_green if decision == "Green" else prob_yellow
                display_value: Optional[object] = None
                if classical_display and classical_override:
                    display_value = classical_display
                classical_label = None
                if classical_override:
                    classical_label = "Yellow"
                elif classical_pred is not None:
                    classical_label = "Green" if classical_pred == "green" else "Yellow"
                step_entry: Dict[str, object] = {
                    "bbox": (box.x0, box.y0, box.x1, box.y1),
                    "prob": prob_display,
                    "prob_green": prob_green,
                    "prob_yellow": prob_yellow,
                    "decision": decision,
                    "raw_decision": raw_decision,
                    "override": classical_override,
                }
                if display_value is not None:
                    step_entry["display_value"] = display_value
                if classical_label is not None:
                    step_entry["classical_pred"] = classical_label
                if math.isfinite(deq_value):
                    step_entry["classical_deq"] = deq_value
                if classical_reason:
                    step_entry["classical_reason"] = classical_reason
                step_a4.append(step_entry)
                result_entry: Dict[str, object] = {
                    "label": decision,
                    "prob": prob_display,
                    "bbox": (box.x0, box.y0, box.x1, box.y1),
                    "prob_green": prob_green,
                    "prob_yellow": prob_yellow,
                    "raw_decision": raw_decision,
                    "source": "classical_override" if classical_override else "a4",
                }
                if display_value is not None:
                    result_entry["display_value"] = display_value
                if classical_label is not None:
                    result_entry["classical_pred"] = classical_label
                if math.isfinite(deq_value):
                    result_entry["classical_deq"] = deq_value
                if classical_reason:
                    result_entry["classical_reason"] = classical_reason
                result_entry["bbox_area"] = box_area
                results.append(result_entry)
                self._last = _snapshot()
                if overlay_cb:
                    try:
                        overlay_cb(_draw_boxes_overlay(image_rgb, results, title=None))
                    except Exception:
                        pass
                continue

            # Klassik (Yellow/Green)
            if use_classical:
                if first_classic and progress_cb:
                    progress_cb("Klassik (Yellow/Green) bewerten…")
                    first_classic = False
                pred_c, classical_info = self._classical_predict(
                    crop, mask_crop, area_px_full=area_px_full, image_area=image_area
                )
                classical_override = False
                classical_reason: Optional[str] = None
                deq_value = float("nan")
                circ_value = float("nan")
                if classical_info:
                    try:
                        deq_value = float(classical_info.get("deq", float("nan")))  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        deq_value = float("nan")
                    try:
                        circ_value = float(classical_info.get("circularity", float("nan")))  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        circ_value = float("nan")
                    reason = classical_info.get("decision_reason")
                    if isinstance(reason, str):
                        classical_reason = reason
                display_value = _format_classical_display(classical_info, self.classical_rules) if pred_c == "yellow" else None
                final = "Green" if pred_c == "green" else "Yellow"
                if pred_c == "yellow":
                    classical_override = True
                if (
                    final == "Green"
                    and math.isfinite(self.classical_override_deq)
                    and math.isfinite(deq_value)
                    and deq_value >= self.classical_override_deq
                ):
                    final = "Yellow"
                    classical_override = True
                    if not classical_reason:
                        classical_reason = self.classical_override_reason
                    display_value = display_value or f"{self.classical_override_reason} deq={deq_value:.1f}px (max {self.classical_override_deq:.1f}px)"
                if (
                    final == "Green"
                    and math.isfinite(self.classical_override_circ)
                    and math.isfinite(circ_value)
                    and circ_value < self.classical_override_circ
                ):
                    final = "Yellow"
                    classical_override = True
                    if not classical_reason:
                        classical_reason = self.classical_override_circ_reason
                    display_value = display_value or f"{self.classical_override_circ_reason} circ={circ_value:.3f} (min {self.classical_override_circ:.3f})"
                results.append({
                    "label": final,
                    "prob": float("nan"),
                    "bbox": (box.x0, box.y0, box.x1, box.y1),
                    "bbox_area": box_area,
                    "source": "classical_override" if classical_override else "classical",
                    "display_value": display_value if display_value is not None else box_area,
                    "classical_reason": classical_reason,
                })
                self._last = _snapshot()
                if overlay_cb:
                    try:
                        overlay_cb(_draw_boxes_overlay(image_rgb, results, title=None))
                    except Exception:
                        pass

        # Cache last state for GUI overlay toggles (kein Gesamtbild-Label mehr)
        self._last = _snapshot()
        overlay = _draw_boxes_overlay(image_rgb, results, title=None)
        meta: Dict[str, object] = {
            "instances": int(len(results)),
            "results": results,
        }
        return overlay, meta


def _draw_label_overlay(image_rgb: np.ndarray, text: str) -> Image.Image:
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x0, y0 = 12, 18 + th
    x1, y1 = x0 + tw + 2 * pad, y0 + th + baseline + pad
    cv2.rectangle(img_bgr, (x0 - pad, y0 - th - pad), (x1, y1), (0, 0, 0), thickness=-1)
    cv2.putText(img_bgr, text, (x0, y0), font, scale, (0, 255, 0), thickness, lineType=cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _draw_boxes_overlay(image_rgb: np.ndarray, results: List[Dict[str, object]], *, title: Optional[str] = None) -> Image.Image:
    colors_bgr = {
        "unbekannt": (180, 180, 180),
        "Never": (0, 0, 0),
        "Red": (0, 0, 255),
        "Yellow": (0, 255, 255),
        "Green": (0, 200, 0),
        "ok": (255, 0, 255),
        "not-red": (255, 0, 255),
    }
    text_colors_bgr = {
        "Never": (255, 255, 255),
    }
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    text_items: List[Dict[str, object]] = []

    for r in results:
        bbox = r.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = [int(v) for v in bbox]  # type: ignore[assignment]
        lab = str(r.get("label", ""))
        prob = r.get("prob")
        display_value = r.get("display_value")
        color_box = colors_bgr.get(lab, (0, 255, 0))
        color_text = text_colors_bgr.get(lab, color_box)
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), color_box, 2)

        text = lab
        value_rendered = False
        if display_value is None and r.get("source") in ("classical_override", "classical"):
            display_value = r.get("bbox_area")
        if display_value is not None:
            if isinstance(display_value, str):
                value_rendered = True
                text = f"{lab} {display_value}"
            else:
                try:
                    display_float = float(display_value)
                    if np.isfinite(display_float):
                        value_rendered = True
                        text = f"{lab} {int(round(display_float))}"
                except Exception:
                    pass
        if not value_rendered:
            try:
                if prob is not None and np.isfinite(float(prob)):
                    text = f"{lab} {float(prob):.2f}"
            except Exception:
                pass

        text_items.append({
            "bbox": (x0, y0, x1, y1),
            "text": text,
            "color_rgb": (int(color_text[2]), int(color_text[1]), int(color_text[0])),
        })

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        transliterate = False
    except OSError:
        font = ImageFont.load_default()
        transliterate = True

    def _normalize_text(text: str) -> str:
        if not transliterate:
            return text
        return (
            text.replace("Ä", "Ae")
            .replace("ä", "ae")
            .replace("Ö", "Oe")
            .replace("ö", "oe")
            .replace("Ü", "Ue")
            .replace("ü", "ue")
            .replace("ß", "ss")
        )

    pad_x = 6
    pad_y = 4
    for item in text_items:
        bbox = item["bbox"]
        text = _normalize_text(str(item["text"]))
        x0, y0, _, _ = bbox  # type: ignore[assignment]
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        tx0 = x0
        ty0 = max(0, y0 - th - pad_y - 2)
        draw.rectangle((tx0, ty0, tx0 + tw + pad_x, ty0 + th + pad_y), fill=(0, 0, 0))
        draw.text((tx0 + 3, ty0 + 2), text, font=font, fill=item["color_rgb"])

    if title:
        title_text = _normalize_text(title)
        title_bbox = draw.textbbox((0, 0), title_text, font=font)
        tw = title_bbox[2] - title_bbox[0]
        th = title_bbox[3] - title_bbox[1]
        draw.rectangle((10, 10, 10 + tw + 12, 10 + th + 12), fill=(0, 0, 0))
        draw.text((16, 16), title_text, font=font, fill=(0, 255, 0))

    return pil_img


def _draw_segmentation_overlay(image_rgb: np.ndarray, instances: np.ndarray) -> Image.Image:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    overlay = bgr.copy()
    rng = np.random.default_rng(1337)
    unique = [int(x) for x in np.unique(instances) if int(x) != 0]
    color_map: Dict[int, Tuple[int, int, int]] = {}
    for lbl in unique:
        c = rng.integers(64, 256, size=3)
        color_map[lbl] = (int(c[2]), int(c[1]), int(c[0]))
        m = (instances == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_map[lbl], thickness=-1)
    blended = cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0)
    for lbl in unique:
        m = (instances == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, color_map[lbl], thickness=2)
    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))


def _build_results_overlay(step_list: List[Dict[str, object]], image_rgb: np.ndarray, title: str) -> Image.Image:
    normalized: List[Dict[str, object]] = []
    for r in step_list:
        normalized.append({
            "bbox": r.get("bbox"),
            "label": str(r.get("decision", "")),
            "prob": r.get("prob"),
        })
    return _draw_boxes_overlay(image_rgb, normalized, title=title)


def build_cached_overlay(cache: Optional[Dict[str, object]], mode: str) -> Optional[Image.Image]:
    if cache is None:
        return None
    image_rgb = cache.get("image_rgb") if isinstance(cache, dict) else None  # type: ignore[assignment]
    if image_rgb is None:
        return None
    if mode == "final":
        res = cache.get("results", []) if isinstance(cache, dict) else []  # type: ignore[assignment]
        return _draw_boxes_overlay(image_rgb, res, title=None)
    if mode == "seg":
        instances = cache.get("instances") if isinstance(cache, dict) else None  # type: ignore[assignment]
        if instances is None:
            return None
        return _draw_segmentation_overlay(image_rgb, instances)
    if mode in ("a1", "a2", "a3", "a4"):
        step = cache.get(mode, []) if isinstance(cache, dict) else []  # type: ignore[assignment]
        return _build_results_overlay(step, image_rgb, title=mode.upper())
    return None


# ----------------------------------- CLI ------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-file ONNX Runtime inference: Segmentierung → A1 → A2 → A3 → A4 → Klassik (Fallback)")
    p.add_argument("--assets", type=Path, default=Path("inference_assets"), help="Path to inference_assets root")
    p.add_argument("--image", type=Path, required=True, help="Input image (RGB JPEG/PNG)")
    p.add_argument("--output", type=Path, default=None, help="Optional output image path for overlay")
    p.add_argument("--top-k", type=int, default=50, help="Keep top-K largest instances")
    p.add_argument("--margin", type=float, default=0.15, help="Relative bbox margin")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    core = InferenceCore(args.assets)
    overlay, meta = core.run_on_image(args.image, margin=float(args.margin), top_k=int(args.top_k))
    # Save overlay
    out_path = args.output
    if out_path is None:
        out_path = args.image.with_name(args.image.stem + "_bewertet.png")
    overlay.save(out_path)
    # Meta (ohne Gesamtbild-Label)
    meta_path = out_path.with_suffix(".json")
    meta_out = {"image": str(args.image), "assets": str(args.assets), **meta}
    meta_path.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Overlay: {out_path}")
    print(f"Meta: {meta_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
