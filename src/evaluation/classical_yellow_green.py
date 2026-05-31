from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("classical_yellow_green")


def _normalize_label(label: str) -> str:
    s = str(label or "").strip().lower()
    if s == "green":
        return "green"
    return s


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureRule:
    name: str
    kind: str  # 'range' | 'min' | 'max'
    low: float | None = None
    high: float | None = None
    stats: Dict[str, float] | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "low": self.low,
            "high": self.high,
            "stats": self.stats or {},
        }

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "FeatureRule":
        return FeatureRule(
            name=str(d.get("name")),
            kind=str(d.get("kind")),
            low=(float(d["low"]) if d.get("low") is not None else None),
            high=(float(d["high"]) if d.get("high") is not None else None),
            stats=d.get("stats") or {},
        )


@dataclass
class RuleSet:
    method: str = "quantile"
    q_low: float = 0.05
    q_high: float = 0.95
    mad_k: float = 2.5
    rules: Dict[str, FeatureRule] | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "q_low": self.q_low,
            "q_high": self.q_high,
            "mad_k": self.mad_k,
            "rules": {k: v.to_dict() for k, v in (self.rules or {}).items()},
        }

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "RuleSet":
        rules_raw = d.get("rules") or {}
        rules = {str(k): FeatureRule.from_dict(v) for k, v in rules_raw.items()}
        return RuleSet(
            method=str(d.get("method", "quantile")),
            q_low=float(d.get("q_low", 0.05)),
            q_high=float(d.get("q_high", 0.95)),
            mad_k=float(d.get("mad_k", 2.5)),
            rules=rules,
        )


@dataclass
class LegacyThresholds:
    feature: str = "relative_size"
    method: str = "quantile"
    low: float = 0.0
    high: float = 1.0
    stats: Dict[str, float] | None = None

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "LegacyThresholds":
        return LegacyThresholds(
            feature=str(d.get("feature", "relative_size")),
            method=str(d.get("method", "quantile")),
            low=float(d.get("low", 0.0)),
            high=float(d.get("high", 1.0)),
            stats=d.get("stats") or {},
        )


def compute_quantiles(values: np.ndarray, q_low: float, q_high: float) -> Tuple[float, float, Dict[str, float]]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("No finite values to compute quantiles")
    low = float(np.quantile(v, q_low))
    high = float(np.quantile(v, q_high))
    stats = {
        "count": float(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=0)),
        "median": float(np.median(v)),
        "q01": float(np.quantile(v, 0.01)),
        "q05": float(np.quantile(v, 0.05)),
        "q95": float(np.quantile(v, 0.95)),
        "q99": float(np.quantile(v, 0.99)),
    }
    return low, high, stats


def compute_mad_range(values: np.ndarray, k: float) -> Tuple[float, float, Dict[str, float]]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("No finite values to compute MAD thresholds")
    median = float(np.median(v))
    mad = float(np.median(np.abs(v - median)))
    mad_scaled = 1.4826 * mad
    low = float(median - k * mad_scaled)
    high = float(median + k * mad_scaled)
    return low, high, {"median": median, "mad": mad, "mad_scaled": mad_scaled, "count": float(v.size)}


def _load_mask(mask_path: str) -> np.ndarray | None:
    if not mask_path:
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 0).astype(np.uint8)


def compute_shape_metrics(mask: np.ndarray) -> Tuple[float | float("nan"), float | float("nan"), float | float("nan")]:
    # Returns (circularity, solidity, radial_roughness)
    if mask is None or mask.size == 0 or mask.ndim != 2:
        return float("nan"), float("nan"), float("nan")
    area = int(np.count_nonzero(mask))
    if area <= 0:
        return float("nan"), float("nan"), float("nan")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return float("nan"), float("nan"), float("nan")
    c = max(contours, key=cv2.contourArea)
    per = float(cv2.arcLength(c, True))
    circularity = (4.0 * np.pi * area / (per ** 2)) if per > 0 else float("nan")
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else float("nan")
    # radial roughness: std(rad)/mean(rad)
    pts = c.reshape(-1, 2).astype(np.float32)
    m = cv2.moments(c)
    if m.get("m00", 0.0) > 0:
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
    else:
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    rad = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    rad_mean = float(rad.mean()) if rad.size > 0 else 0.0
    rad_std = float(rad.std(ddof=0)) if rad.size > 0 else 0.0
    rough = (rad_std / rad_mean) if rad_mean > 0 else float("nan")
    return float(circularity), float(solidity), float(rough)


def compute_row_features(row: pd.Series) -> Dict[str, float]:
    # deq from area
    try:
        area_px = float(row.get("area_px", float("nan")))
        deq = 2.0 * float(np.sqrt(area_px / np.pi)) if np.isfinite(area_px) and area_px > 0 else float("nan")
    except Exception:
        deq = float("nan")
    # shape metrics from mask
    mask_path = str(row.get("mask_path", "") or "")
    mask = _load_mask(mask_path)
    circ, soli, rough = compute_shape_metrics(mask) if mask is not None else (float("nan"), float("nan"), float("nan"))
    return {
        "deq": float(deq) if np.isfinite(deq) else float("nan"),
        "circularity": float(circ) if np.isfinite(circ) else float("nan"),
        "solidity": float(soli) if np.isfinite(soli) else float("nan"),
        "radial_roughness": float(rough) if np.isfinite(rough) else float("nan"),
    }


def fit_ruleset(
    df: pd.DataFrame,
    *,
    method: str = "quantile",
    q_low: float = 0.05,
    q_high: float = 0.95,
    mad_k: float = 2.5,
) -> RuleSet:
    green = df[df["class_label_norm"] == "green"].reset_index(drop=True)
    if green.empty:
        raise ValueError("No 'green' samples found for calibration")
    # Compute features for green samples
    feats: Dict[str, List[float]] = {k: [] for k in ["deq", "circularity"]}
    for _, row in green.iterrows():
        vals = compute_row_features(row)
        for k, v in vals.items():
            if np.isfinite(v):
                feats[k].append(float(v))
    rules: Dict[str, FeatureRule] = {}
    # Range features: deq only (size)
    for name in ["deq"]:
        arr = np.asarray(feats[name], dtype=float)
        if arr.size == 0:
            raise ValueError(f"No finite values to calibrate feature '{name}'")
        if method == "quantile":
            low, high, stats = compute_quantiles(arr, q_low, q_high)
        else:
            low, high, stats = compute_mad_range(arr, mad_k)
        rules[name] = FeatureRule(name=name, kind="range", low=float(low), high=float(high), stats=stats)
    # Min feature: circularity (use q_low)
    for name in ["circularity"]:
        arr = np.asarray(feats[name], dtype=float)
        if arr.size == 0:
            raise ValueError(f"No finite values to calibrate feature '{name}'")
        if method == "quantile":
            thr, _, stats = compute_quantiles(arr, q_low, 0.99)
        else:
            thr, _, stats = compute_mad_range(arr, mad_k)
        rules[name] = FeatureRule(name=name, kind="min", low=float(thr), high=None, stats=stats)

    rs = RuleSet(method=method, q_low=q_low, q_high=q_high, mad_k=mad_k, rules=rules)
    LOGGER.info("Fitted rules on %d green samples.", len(green))
    for rn, r in rules.items():
        LOGGER.info("  %s: kind=%s low=%s high=%s", rn, r.kind, f"{r.low:.6f}" if r.low is not None else "-", f"{r.high:.6f}" if r.high is not None else "-")
    return rs


def evaluate_feature(value: float, rule: FeatureRule) -> Tuple[bool, str | None]:
    if not np.isfinite(value):
        return False, f"invalid_{rule.name}"
    if rule.kind == "range":
        assert rule.low is not None and rule.high is not None
        if value < rule.low:
            # specific too small/large for size-like features
            reason = "too_small" if rule.name in ("relative_size", "deq") else f"{rule.name}_low"
            return False, reason
        if value > rule.high:
            reason = "too_large" if rule.name in ("relative_size", "deq") else f"{rule.name}_high"
            return False, reason
        return True, None
    if rule.kind == "min":
        assert rule.low is not None
        return (value >= rule.low), (None if value >= rule.low else f"{rule.name}_low")
    if rule.kind == "max":
        assert rule.high is not None
        return (value <= rule.high), (None if value <= rule.high else f"{rule.name}_high")
    return False, f"invalid_rule_{rule.name}"


def predict_with_rules(row: pd.Series, ruleset: RuleSet) -> Tuple[str, Dict[str, object]]:
    vals = compute_row_features(row)
    supports: Dict[str, int] = {}
    reasons: List[str] = []
    first_reason: str | None = None
    first_reason_value: float | None = None
    first_reason_feature: str | None = None
    for name, rule in (ruleset.rules or {}).items():
        v = vals.get(name, float("nan"))
        ok, reason = evaluate_feature(v, rule)
        supports[name] = int(bool(ok))
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
    details: Dict[str, object] = {
        "support_count": support_count,
        "decision_reason": decision_reason,
        "first_reason": first_reason or "",
        "first_reason_value": first_reason_value if first_reason_value is not None else float("nan"),
        "first_reason_feature": first_reason_feature or "",
        **{f"feature_{k}": float(vals.get(k, float("nan"))) for k in vals.keys()},
        **{f"support_{k}": int(v) for k, v in supports.items()},
    }
    return pred, details


def evaluate_predictions(df: pd.DataFrame) -> Dict[str, object]:
    true = df["class_label_norm"].tolist()
    pred = df["pred_label"].tolist()
    classes = ["green", "yellow"]
    cm = {c: {d: 0 for d in classes} for c in classes}
    for t, p in zip(true, pred):
        if t in classes and p in classes:
            cm[t][p] += 1
    tp = cm["yellow"]["yellow"]
    tn = cm["green"]["green"]
    fp = cm["green"]["yellow"]
    fn = cm["yellow"]["green"]
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision_yellow": prec,
        "recall_yellow": rec,
        "support": {c: sum(cm[c].values()) for c in classes},
    }


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata table not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df = df.copy()
    df["class_label_norm"] = df["class_label"].map(_normalize_label)
    return df


def run_fit(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    df = load_metadata(args.metadata)
    df_rel = df[df["class_label_norm"].isin(["green", "yellow"])].reset_index(drop=True)
    ruleset = fit_ruleset(df_rel, method=args.method, q_low=args.q_low, q_high=args.q_high, mad_k=args.mad_k)
    out = Path(args.output)
    _ensure_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ruleset.to_dict(), f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved thresholds to %s", out)
    return 0


def _load_rules_or_fail(path: Path) -> RuleSet:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict) and "rules" in d:
        return RuleSet.from_dict(d)
    # Legacy single-feature file detected
    legacy = LegacyThresholds.from_dict(d)
    raise RuntimeError(
        "Loaded legacy single-feature thresholds. Please re-run 'fit' to generate multi-feature rules: "
        f"feature={legacy.feature} in [{legacy.low},{legacy.high}]"
    )


def run_predict(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    df = load_metadata(args.metadata)
    df_rel = df[df["class_label_norm"].isin(["green", "yellow"])].reset_index(drop=True)

    cfg_path = Path(args.thresholds)
    ruleset = _load_rules_or_fail(cfg_path)
    LOGGER.info("Loaded %d feature rules (%s)", len(ruleset.rules or {}), ruleset.method)

    preds: List[str] = []
    details_list: List[Dict[str, object]] = []
    for _, row in df_rel.iterrows():
        p, details = predict_with_rules(row, ruleset)
        preds.append(p)
        details_list.append(details)
    df_rel["pred_label"] = preds
    for k in details_list[0].keys():
        df_rel[k] = [d.get(k, None) for d in details_list]

    metrics = evaluate_predictions(df_rel)
    LOGGER.info(
        "Accuracy=%.3f | Precision(yellow)=%.3f | Recall(yellow)=%.3f | cm=%s",
        metrics["accuracy"], metrics["precision_yellow"], metrics["recall_yellow"], json.dumps(metrics["confusion_matrix"]),
    )

    out_csv = Path(args.output_csv)
    _ensure_dir(out_csv)
    df_rel.to_csv(out_csv, index=False)
    LOGGER.info("Saved predictions to %s", out_csv)

    out_json = Path(args.output_metrics)
    _ensure_dir(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved metrics to %s", out_json)
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical yellow/green classifier using 5-feature voting (4-of-5)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="Fit per-feature thresholds from green berries")
    p_fit.add_argument("--metadata", type=Path, default=Path("data/instance_crops/metadata/crops.csv"))
    p_fit.add_argument("--method", type=str, choices=["quantile", "mad"], default="quantile")
    p_fit.add_argument("--q-low", dest="q_low", type=float, default=0.05)
    p_fit.add_argument("--q-high", dest="q_high", type=float, default=0.95)
    p_fit.add_argument("--mad-k", dest="mad_k", type=float, default=2.5, help="k for median±k*MAD_scaled")
    p_fit.add_argument("--output", type=Path, default=Path("outputs/classical/thresholds.json"))
    p_fit.set_defaults(func=run_fit)

    p_pred = sub.add_parser("predict", help="Predict labels with 5-feature voting and evaluate")
    p_pred.add_argument("--metadata", type=Path, default=Path("data/instance_crops/metadata/crops.csv"))
    p_pred.add_argument("--thresholds", type=Path, default=Path("outputs/classical/thresholds.json"))
    p_pred.add_argument("--output-csv", dest="output_csv", type=Path, default=Path("outputs/classical/predictions.csv"))
    p_pred.add_argument("--output-metrics", dest="output_metrics", type=Path, default=Path("outputs/classical/metrics.json"))
    p_pred.set_defaults(func=run_predict)

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
