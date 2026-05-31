from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from ..config.paths import project_root
from ..training.augment import apply_texture_augmentations, apply_color_augmentations
from ..utils.color_norm import gray_world


TaskType = Literal["a1", "a2", "a3", "a4"]


@dataclass
class ClassificationDataConfig:
    metadata_dir: Path = Path("data/instance_crops/metadata")
    crops_csv: str = "crops.csv"
    notberry_csv: str = "notberry.csv"
    folds: int = 5
    fold_id: int = 0
    # Which subset to load
    split: Literal["train", "val", "test"] = "train"
    # Split strategy:
    # - 'fold': use preassigned k-folds via 'fold_id' column (current default behavior)
    # - 'ratio': random, stratified split per class (0/1) with given ratios and seed
    split_mode: Literal["fold", "ratio"] = "fold"
    split_ratios: Tuple[float, float, float] = (70.0, 15.0, 15.0)  # train, val, test (percentages or fractions)
    split_seed: int = 1337
    split_group_column: Optional[str] = None
    # Optional controls for group-based ratio split
    split_group_val_max_frac: Optional[float] = None  # e.g., 0.16 caps val to 16% of total crops
    split_group_test_max_frac: Optional[float] = None  # optional cap for test
    split_group_pos_train_min_frac: Optional[float] = None  # e.g., 0.85 → at least 85% of positives in train
    manual_split_dir: Optional[Path] = None
    union_metadata_path: Optional[Path] = None  # JSONL with convex-hull-aware crops
    union_metadata_class: Optional[str] = "never"
    include_union_channel: bool = False  # append hull mask as extra channel
    union_split: Literal["train", "all"] = "all"  # union-only crops live in train or across all splits
    # Preprocessing
    input_size: Tuple[int, int] = (320, 320)  # H, W (large enough for details)
    imagenet_mean: Sequence[float] = (0.485, 0.456, 0.406)
    imagenet_std: Sequence[float] = (0.229, 0.224, 0.225)
    include_mask_channel: bool = True
    mask_usage: Literal["auto", "rgb_only", "mask_channel"] = "auto"
    union_guidance: Optional[Dict[str, Any]] = None
    # Optional photometric augmentations for training split
    augment: Optional[Dict[str, Any]] = None
    # Hard-negative mining (sampling boost for previously misclassified examples)
    hnm_enabled: bool = False
    hnm_glob: Optional[str] = None  # glob pattern(s) for CSVs with hard negatives
    hnm_boost: float = 3.0
    hnm_error_types: Sequence[str] = ("fp",)
    hnm_slice_label: Optional[str] = None  # e.g., "class_label"
    hnm_slice_value: Optional[str] = None  # e.g., "green"
    extra_negative_labels: Optional[Dict[str, Sequence[str] | str]] = None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet" or path.name.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv" or path.name.endswith(".csv"):
        return pd.read_csv(path)
    # try both
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    if path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv"))
    raise FileNotFoundError(f"Metadata table not found: {path}")


def _resolve_repo_path(rel_or_abs: str | Path, root: Path) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _normalize_label(label: str) -> str:
    s = str(label or "").strip().lower()
    if s == "green":
        return "green"
    return s


def _infer_label_from_path(path: Path) -> str:
    parent = path.parent.name.lower()
    mapping = {
        "yellow": "yellow",
        "green": "green",
        "green": "green",
        "red": "red",
        "never": "never",
        "to_sort": "unknown",
    }
    return mapping.get(parent, parent or "unknown")


def _resize_with_padding_image(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    h, w = img.shape[:2]
    # keep ratio letterbox
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
    return out


def _resize_with_padding_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
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


def _load_union_metadata_table(
    path: Path,
    repo_root: Path,
    class_filter: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], set[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Union metadata file not found: {path}")
    rows: List[Dict[str, Any]] = []
    lookup: Dict[str, Dict[str, Any]] = {}
    normalized_filter = _normalize_label(class_filter) if class_filter else None
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            class_label = _normalize_label(entry.get("class_label", ""))
            if normalized_filter and class_label != normalized_filter:
                continue
            crop_rel = entry.get("crop_path")
            if not crop_rel:
                continue
            crop_abs = str(_resolve_repo_path(crop_rel, repo_root))
            mask_rel = entry.get("mask_path", "")
            mask_abs = str(_resolve_repo_path(mask_rel, repo_root)) if mask_rel else ""
            crop_size = entry.get("crop_size") or {}
            width = int(crop_size.get("width") or 0)
            height = int(crop_size.get("height") or 0)
            if width <= 0 or height <= 0:
                try:
                    with Image.open(crop_abs) as im:
                        width, height = im.size
                except Exception:
                    width = max(1, width)
                    height = max(1, height)
            polygons = []
            for region in entry.get("union_regions", []):
                poly = region.get("polygon") or []
                if len(poly) < 3:
                    continue
                polygons.append(tuple((float(pt[0]), float(pt[1])) for pt in poly))
            lookup[crop_abs] = {
                "size": (int(height), int(width)),
                "polygons": polygons,
            }
            source_group = entry.get("source_group") or entry.get("image_name") or ""
            source_subgroup = entry.get("source_subgroup") or Path(entry.get("source_image", "")).stem
            rows.append(
                {
                    "annotation_id": int(entry.get("annotation_id", -100000 - idx)),
                    "image_id": int(entry.get("component_index", -100000 - idx)),
                    "fold_id": int(entry.get("fold_id", -1)),
                    "class_label": entry.get("class_label", class_filter or "never"),
                    "crop_path": crop_abs,
                    "mask_path": mask_abs,
                    "lighting": entry.get("lighting", ""),
                    "state": entry.get("state", ""),
                    "source_group": source_group,
                    "source_subgroup": source_subgroup,
                    "neg_type": entry.get("neg_type", ""),
                    "__is_union": True,
                }
            )
    df_union = pd.DataFrame(rows)
    norm_paths = {str(Path(p).resolve()) for p in lookup.keys()}
    return df_union, lookup, norm_paths


def _ensure_manual_entries(
    df: pd.DataFrame,
    allowed_abs: Sequence[str],
    repo_root: Path,
) -> pd.DataFrame:
    if not allowed_abs:
        return df
    existing = set(str(_resolve_repo_path(p, repo_root)) for p in df["crop_path"].astype(str))
    rows: List[Dict[str, Any]] = []
    for abs_path in allowed_abs:
        abs_resolved = str(Path(abs_path).resolve())
        if abs_resolved in existing:
            continue
        file_path = Path(abs_resolved)
        if not file_path.exists():
            continue
        rel = str(file_path.relative_to(repo_root))
        label = _infer_label_from_path(file_path)
        mask_abs = repo_root / Path(rel.replace("images/", "masks/"))
        mask_rel = str(mask_abs.relative_to(repo_root)) if mask_abs.exists() else ""
        rows.append(
            {
                "annotation_id": -1,
                "image_id": -1,
                "fold_id": -1,
                "class_label": label,
                "__is_union": False,
                "state": "unknown",
                "lighting": "UNKNOWN",
                "scene_stem": file_path.stem,
                "crop_path": rel,
                "mask_path": mask_rel,
                "source_image_path": "",
                "instances_mask_path": "",
                "overlay_path": "",
                "source_group": "",
                "source_subgroup": "",
                "neg_type": "",
                "target": 1 if label == "never" else 0,
            }
        )
    if not rows:
        return df
    df_new = pd.DataFrame(rows)
    return pd.concat([df, df_new], ignore_index=True, sort=False)


class BerryClassificationDataset(Dataset):
    """
    Classification dataset for Backbone A tasks.

    - A1: notberry (1) vs {yellow, green, red, never} (0)
    - A2: never (1) vs {yellow, green, red} (0) — excludes notberry samples
    - A3: red (1) vs {yellow, green} (0)
    - A4: green (1) vs yellow (0)

    Uses crops from data/instance_crops and metadata tables.
    Preserves aspect ratio via letterbox to a fixed input size.
    Optional 4th channel with the binary mask.
    """

    def __init__(
        self,
        *,
        task: TaskType,
        config: ClassificationDataConfig | None = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.cfg = config or ClassificationDataConfig()
        self.transform = transform
        module_root = Path(__file__).resolve().parents[2]
        candidate_roots = [module_root]
        try:
            candidate_roots.append(project_root().resolve())
        except Exception:
            pass
        chosen_root = None
        for cand in candidate_roots:
            if (cand / "data").exists():
                chosen_root = cand
                break
        self.repo_root = (chosen_root or module_root).resolve()
        meta_root = (self.repo_root / self.cfg.metadata_dir).resolve()

        crops_path = meta_root / self.cfg.crops_csv
        df_crops = _read_table(crops_path)

        if task == "a1":
            nb_path = meta_root / self.cfg.notberry_csv
            if nb_path.exists():
                df_nb = _read_table(nb_path)
                df = pd.concat([df_crops, df_nb], ignore_index=True, sort=False)
            else:
                df = df_crops.copy()
        else:
            df = df_crops.copy()

        df["__is_union"] = False
        self.union_lookup: Dict[str, Dict[str, Any]] = {}
        self.union_norm_paths: set[str] = set()
        union_metadata_path = getattr(self.cfg, "union_metadata_path", None)
        if union_metadata_path:
            union_abs = _resolve_repo_path(union_metadata_path, self.repo_root)
            union_df, union_lookup, union_norm = _load_union_metadata_table(
                union_abs,
                self.repo_root,
                getattr(self.cfg, "union_metadata_class", "never"),
            )
            if not union_df.empty:
                df = pd.concat([df, union_df], ignore_index=True, sort=False)
            self.union_lookup = union_lookup
            self.union_norm_paths = union_norm
        else:
            self.union_lookup = {}
            self.union_norm_paths = set()
        self.union_split_policy = getattr(self.cfg, "union_split", "train")
        guidance_cfg = getattr(self.cfg, "union_guidance", {}) or {}
        self.union_guidance_prob = float(guidance_cfg.get("prob", 0.0) or 0.0)
        self.union_guidance_mode = str(guidance_cfg.get("mode", "blur")).lower()
        self.union_guidance_strength = float(guidance_cfg.get("strength", 0.4) or 0.4)
        self.union_guidance_blur_sigma = float(guidance_cfg.get("blur_sigma", 2.0) or 2.0)
        self.union_guidance_erode = int(guidance_cfg.get("erode_px", 0) or 0)

        extra_negative_cfg = getattr(self.cfg, "extra_negative_labels", None)

        def _get_extra_negatives(task_name: str) -> List[str]:
            if not isinstance(extra_negative_cfg, dict):
                return []
            items = extra_negative_cfg.get(task_name)
            if isinstance(items, str):
                iterable: List[str] = [items]
            elif isinstance(items, (list, tuple, set)):
                iterable = list(items)
            else:
                return []
            return [_normalize_label(x) for x in iterable if isinstance(x, str) and x]

        # Normalize labels and filter per task
        df["class_label"] = df["class_label"].map(_normalize_label)
        if task == "a2":
            # A2: classify 'never' vs {yellow, green, red}
            df = df[df["class_label"].isin(["never", "red", "yellow", "green"])].copy()
        elif task == "a3":
            # A3: classify 'red' vs {yellow, green}
            df = df[df["class_label"].isin(["red", "yellow", "green"])].copy()
        elif task == "a4":
            # A4: classify 'green' vs 'yellow'
            extra_neg = _get_extra_negatives("a4")
            allowed = {"yellow", "green"}
            allowed.update(extra_neg)
            df = df[df["class_label"].isin(list(allowed))].copy()

        # Derive targets first (needed for stratified ratio splitting)
        if task == "a1":
            df["target"] = (df["class_label"] == "notberry").astype(np.int64)
        elif task == "a2":
            df["target"] = (df["class_label"] == "never").astype(np.int64)
        elif task == "a3":
            df["target"] = (df["class_label"] == "red").astype(np.int64)
        elif task == "a4":
            df["target"] = (df["class_label"] == "green").astype(np.int64)
        else:
            raise ValueError(f"Unsupported task '{task}'")

        # Select split
        mode = getattr(self.cfg, "split_mode", "fold")
        if mode == "fold":
            # Original behavior: hold-out one preassigned fold for validation
            if self.cfg.split == "test":
                raise ValueError("'test' split is not supported with split_mode='fold'. Use 'ratio' split.")
            fold_id = int(self.cfg.fold_id)
            if self.cfg.split == "train":
                df = df[df["fold_id"] != fold_id]
            elif self.cfg.split == "val":
                df = df[df["fold_id"] == fold_id]
            else:
                raise ValueError("split must be 'train' or 'val'")
        elif mode == "manual":
            manual_dir = getattr(self.cfg, "manual_split_dir", None)
            if not manual_dir:
                raise ValueError("manual split requires manual_split_dir")
            manual_root = _resolve_repo_path(manual_dir, self.repo_root)
            candidates: List[Path] = []
            if manual_root.is_dir():
                candidates.extend(
                    [
                        manual_root / str(self.task) / f"{self.cfg.split}.txt",
                        manual_root / f"{self.task}_{self.cfg.split}.txt",
                        manual_root / f"{self.cfg.split}.txt",
                    ]
                )
            else:
                raise FileNotFoundError(f"manual_split_dir does not exist: {manual_root}")
            split_file: Optional[Path] = None
            for cand in candidates:
                if cand.exists():
                    split_file = cand
                    break
            if split_file is None:
                raise FileNotFoundError(
                    f"No manual split file found for split='{self.cfg.split}'. Tried: {candidates}"
                )
            allowed: List[str] = []
            with split_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    val = line.strip()
                    if not val or val.startswith("#"):
                        continue
                    allowed.append(str(_resolve_repo_path(val, self.repo_root)))
            df = _ensure_manual_entries(df, allowed, self.repo_root)
            allowed_set = set(allowed)

            def _norm_path(value: object) -> str:
                if not isinstance(value, str) or not value:
                    return ""
                return str(_resolve_repo_path(value, self.repo_root))

            df["__norm_path"] = df["crop_path"].map(_norm_path)
            df = df[df["__norm_path"].isin(allowed_set)].copy()
            df.drop(columns=["__norm_path"], inplace=True)
        elif mode == "ratio":
            # Ratio-based split with optional grouping (e.g. keep all crops from the same image together)
            ratios = list(getattr(self.cfg, "split_ratios", (70.0, 15.0, 15.0)))
            if len(ratios) != 3:
                raise ValueError("split_ratios must have three values: train, val, test")
            rsum = float(sum(ratios))
            if rsum <= 0:
                raise ValueError("split_ratios must sum to a positive value")
            if rsum > 1.0001:
                fracs = [float(r) / rsum for r in ratios]
            else:
                fracs = [float(r) for r in ratios]

            seed = int(getattr(self.cfg, "split_seed", 1337))
            rng = np.random.default_rng(seed)

            union_rows_for_train = pd.DataFrame()
            if self.union_split_policy == "train" and "__is_union" in df.columns:
                union_mask = df["__is_union"] == True
                if union_mask.any():
                    union_rows_for_train = df[union_mask].copy()
                    df = df[~union_mask].copy()

            total = len(df)
            targets = df["target"].to_numpy(dtype=np.int64)
            total_counts = np.bincount(targets, minlength=2).astype(np.float64)

            group_col = getattr(self.cfg, "split_group_column", None)
            # Explicit disable if configured as null/none/empty
            if isinstance(group_col, str) and group_col.strip().lower() in {"", "none", "null"}:
                group_col = None

            selected_indices: Dict[str, np.ndarray]
            used_group_split = False
            if group_col and group_col in df.columns:
                group_values = df[group_col].fillna("__nan__").astype(str)
                df["__split_group"] = group_values
                groups: Dict[str, Dict[str, Any]] = {}
                for key, idxs in df.groupby("__split_group").groups.items():
                    idx_array = np.asarray(sorted(idxs), dtype=np.int64)
                    counts = np.bincount(targets[idx_array], minlength=2).astype(np.float64)
                    groups[key] = {
                        "indices": idx_array,
                        "counts": counts,
                        "size": int(len(idx_array)),
                    }
                if groups:
                    used_group_split = True
                    expected_counts = {
                        "train": total_counts * fracs[0],
                        "val": total_counts * fracs[1],
                        "test": total_counts * fracs[2],
                    }
                    expected_sizes = {
                        "train": fracs[0] * total,
                        "val": fracs[1] * total,
                        "test": fracs[2] * total,
                    }
                    # Optional hard caps on absolute sizes (by fraction of total)
                    val_cap = None
                    if getattr(self.cfg, "split_group_val_max_frac", None):
                        try:
                            val_cap = float(self.cfg.split_group_val_max_frac) * float(total)
                        except Exception:
                            val_cap = None
                    test_cap = None
                    if getattr(self.cfg, "split_group_test_max_frac", None):
                        try:
                            test_cap = float(self.cfg.split_group_test_max_frac) * float(total)
                        except Exception:
                            test_cap = None
                    # Optional minimum share of positives in train
                    pos_total = float(total_counts[1]) if len(total_counts) > 1 else 0.0
                    pos_train_min = None
                    if getattr(self.cfg, "split_group_pos_train_min_frac", None) and pos_total > 0:
                        try:
                            pos_train_min = float(self.cfg.split_group_pos_train_min_frac) * pos_total
                        except Exception:
                            pos_train_min = None
                    assignments: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
                    counts_so_far = {
                        "train": np.zeros_like(total_counts, dtype=np.float64),
                        "val": np.zeros_like(total_counts, dtype=np.float64),
                        "test": np.zeros_like(total_counts, dtype=np.float64),
                    }
                    sizes_so_far = {"train": 0.0, "val": 0.0, "test": 0.0}

                    ordering = [(-info["size"], float(rng.random()), key) for key, info in groups.items()]
                    ordering.sort()
                    for _neg_size, _rand, key in ordering:
                        info = groups[key]
                        best_split = None
                        best_score = None
                        # If we must boost train positives, force positive-heavy groups to train until satisfied
                        if pos_train_min is not None and info["counts"][1] > 0 and counts_so_far["train"][1] < pos_train_min:
                            candidate_splits = ("train", "val", "test")
                        else:
                            candidate_splits = ("train", "val", "test")
                        for split_name, frac in zip(candidate_splits, fracs):
                            new_counts = counts_so_far[split_name] + info["counts"]
                            count_diff = np.sum((new_counts - expected_counts[split_name]) ** 2)
                            new_size = sizes_so_far[split_name] + info["size"]
                            size_diff = (new_size - expected_sizes[split_name]) ** 2
                            # Hard caps: avoid sending more to val/test if caps are set and would be exceeded
                            if split_name == "val" and val_cap is not None and new_size > val_cap:
                                continue
                            if split_name == "test" and test_cap is not None and new_size > test_cap:
                                continue
                            score = count_diff + size_diff / max(1.0, expected_sizes[split_name])
                            # Positive-to-train bias: if we still need positives in train, add penalty to non-train splits
                            if pos_train_min is not None and info["counts"][1] > 0 and counts_so_far["train"][1] < pos_train_min:
                                if split_name != "train":
                                    score += 1e9  # effectively steer to train until min reached
                            if best_score is None or score < best_score:
                                best_score = score
                                best_split = split_name
                        assert best_split is not None
                        assignments[best_split].append(key)
                        counts_so_far[best_split] += info["counts"]
                        sizes_so_far[best_split] += info["size"]

                    selected_indices = {}
                    for split_name, keys in assignments.items():
                        if not keys:
                            selected_indices[split_name] = np.array([], dtype=np.int64)
                            continue
                        selected_indices[split_name] = np.sort(
                            np.concatenate([groups[k]["indices"] for k in keys])
                        )
                else:
                    selected_indices = {}
                df.drop(columns="__split_group", inplace=True)
            else:
                selected_indices = {}

            if not used_group_split:
                idx_all = np.arange(total, dtype=np.int64)
                idx_by_class = {c: idx_all[targets == c] for c in [0, 1]}
                selected_lists: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
                for c in [0, 1]:
                    idxs = idx_by_class.get(c, np.array([], dtype=np.int64))
                    if idxs.size == 0:
                        continue
                    rng.shuffle(idxs)
                    n = int(idxs.size)
                    n_train = int(np.floor(fracs[0] * n))
                    n_val = int(np.floor(fracs[1] * n))
                    n_test = max(0, n - n_train - n_val)
                    selected_lists["train"].extend(idxs[:n_train].tolist())
                    selected_lists["val"].extend(idxs[n_train : n_train + n_val].tolist())
                    selected_lists["test"].extend(idxs[n_train + n_val : n_train + n_val + n_test].tolist())
                selected_indices = {
                    split_name: np.sort(np.array(indices, dtype=np.int64)) if indices else np.array([], dtype=np.int64)
                    for split_name, indices in selected_lists.items()
                }

            # If group-based split produced an empty requested split, fall back to simple per-sample stratified split
            take = selected_indices.get(self.cfg.split, np.array([], dtype=np.int64))
            if take.size == 0:
                idx_all = np.arange(total, dtype=np.int64)
                idx_by_class = {c: idx_all[targets == c] for c in [0, 1]}
                fallback: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
                for c in [0, 1]:
                    idxs = idx_by_class.get(c, np.array([], dtype=np.int64))
                    if idxs.size == 0:
                        continue
                    rng.shuffle(idxs)
                    n = int(idxs.size)
                    n_train = int(np.floor(fracs[0] * n))
                    n_val = int(np.floor(fracs[1] * n))
                    n_test = max(0, n - n_train - n_val)
                    fallback["train"].extend(idxs[:n_train].tolist())
                    fallback["val"].extend(idxs[n_train : n_train + n_val].tolist())
                    fallback["test"].extend(idxs[n_train + n_val : n_train + n_val + n_test].tolist())
                take_fb = np.sort(np.array(fallback.get(self.cfg.split, []), dtype=np.int64))
                if take_fb.size == 0:
                    df = df.iloc[0:0]
                else:
                    df = df.iloc[take_fb]
            else:
                df = df.iloc[take]

            if (
                self.union_split_policy == "train"
                and self.cfg.split == "train"
                and not union_rows_for_train.empty
            ):
                df = pd.concat([df, union_rows_for_train], ignore_index=True, sort=False)
        else:
            raise ValueError(f"Unknown split_mode: {mode}")

        if self.union_split_policy == "train" and self.cfg.split != "train":
            if "__is_union" in df.columns:
                df = df[~df["__is_union"]].copy()

        # Resolve paths to absolute
        def _abs_or_empty(x: object) -> str:
            if isinstance(x, str) and x:
                return str(_resolve_repo_path(x, self.repo_root))
            return ""

        df["crop_path"] = df["crop_path"].map(_abs_or_empty)
        if "mask_path" in df.columns:
            df["mask_path"] = df["mask_path"].map(_abs_or_empty)
        else:
            df["mask_path"] = ""

        # Keep essential columns
        keep_cols = [
            "annotation_id",
            "image_id",
            "class_label",
            "crop_path",
            "mask_path",
            "target",
            "lighting",
            "state",
            "source_group",
            "source_subgroup",
            "neg_type",
        ]
        present_cols = [c for c in keep_cols if c in df.columns]
        self.df = df[present_cols].reset_index(drop=True)

        self.slice_columns = [
            col
            for col in ("class_label", "lighting", "state", "source_group", "source_subgroup", "neg_type")
            if col in self.df.columns
        ]

        # Ensure slice/meta columns are consistently strings (no NaN/float mixups)
        # Mixed types cause PyTorch's default_collate to choose float collation and then fail on strings.
        for col in self.slice_columns:
            # Convert to pandas string dtype and replace missing with empty string
            try:
                self.df[col] = self.df[col].astype("string").fillna("")
            except Exception:
                # Fallback: cast via map(str) if astype fails for any reason
                self.df[col] = self.df[col].map(lambda v: "" if pd.isna(v) else str(v))

        self.mean = np.array(self.cfg.imagenet_mean, dtype=np.float32)
        self.std = np.array(self.cfg.imagenet_std, dtype=np.float32)
        self.target_hw = (int(self.cfg.input_size[0]), int(self.cfg.input_size[1]))
        mask_usage = getattr(self.cfg, "mask_usage", "auto")
        if mask_usage == "auto":
            self.use_mask_channel = bool(self.cfg.include_mask_channel)
        elif mask_usage == "rgb_only":
            self.use_mask_channel = False
        elif mask_usage == "mask_channel":
            self.use_mask_channel = True
        else:
            raise ValueError(f"Unknown mask_usage setting: {mask_usage}")
        include_union_channel = bool(getattr(self.cfg, "include_union_channel", False))
        self.use_union_channel = bool(include_union_channel and self.union_lookup)

    def __len__(self) -> int:
        return int(len(self.df))

    def _load_rgb(self, path: str) -> np.ndarray:
        if not path:
            raise FileNotFoundError("Empty crop_path")
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def _load_mask(self, path: str, shape: Tuple[int, int]) -> np.ndarray:
        # Missing mask → zeros
        if not path:
            return np.zeros(shape, dtype=np.uint8)
        try:
            m = Image.open(path).convert("L")
            arr = np.asarray(m, dtype=np.uint8)
            # Binary masks stored as 0/255
            if arr.ndim == 2:
                return (arr > 0).astype(np.uint8)
            if arr.ndim == 3:
                return (cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) > 0).astype(np.uint8)
            return np.zeros(shape, dtype=np.uint8)
        except Exception:
            return np.zeros(shape, dtype=np.uint8)

    def _render_union_mask(self, crop_path: str, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if not self.union_lookup:
            return None
        info = self.union_lookup.get(str(crop_path))
        if not info or not info.get("polygons"):
            return None
        height, width = info.get("size", shape)
        width = max(1, int(width))
        height = max(1, int(height))
        canvas = Image.new("L", (width, height), color=0)
        draw = ImageDraw.Draw(canvas)
        for poly in info.get("polygons", []):
            if len(poly) < 3:
                continue
            draw.polygon(poly, fill=255, outline=255)
        arr = np.asarray(canvas, dtype=np.uint8)
        if arr.shape != shape:
            arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return arr

    def _apply_union_guidance(self, img: np.ndarray, union_mask: np.ndarray) -> np.ndarray:
        if self.cfg.split != "train":
            return img
        if self.union_guidance_prob <= 0.0 or union_mask is None:
            return img
        if np.random.rand() >= self.union_guidance_prob:
            return img
        if union_mask.max() == 0:
            return img
        mask = union_mask.astype(np.uint8)
        if self.union_guidance_erode > 0:
            k = max(1, int(self.union_guidance_erode))
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
        mask = (mask > 0).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.5) if mask.max() > 0 else mask
        mask = np.clip(mask, 0.0, 1.0)
        keep = mask[:, :, None]
        drop = 1.0 - keep
        if self.union_guidance_mode == "desaturate":
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray3 = np.stack([gray, gray, gray], axis=2)
            guided = keep * img + drop * gray3
        elif self.union_guidance_mode == "darken":
            factor = np.clip(self.union_guidance_strength, 0.0, 1.0)
            guided = keep * img + drop * (img * factor)
        else:  # blur (default)
            sigma = max(0.1, float(self.union_guidance_blur_sigma))
            blurred = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma)
            guided = keep * img + drop * blurred
        return guided.astype(np.uint8)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | str]:
        row = self.df.iloc[idx]
        img = self._load_rgb(row["crop_path"])  # (H, W, 3) uint8
        mask2d = self._load_mask(row.get("mask_path", ""), img.shape[:2]) if self.use_mask_channel else None
        union_raw = self._render_union_mask(row["crop_path"], img.shape[:2])

        # Letterbox resize with preserved aspect ratio
        img_resized = _resize_with_padding_image(img, self.target_hw)
        union_resized = None
        if union_raw is not None:
            union_resized = _resize_with_padding_mask(union_raw.astype(np.uint8), self.target_hw)
            img_resized = self._apply_union_guidance(img_resized, union_resized)
        # Optional color normalization (A2/A3/A4 favoring color consistency)
        c_norm = getattr(self.cfg, "color_norm", None)
        if self.task in {"a2", "a3", "a4"} and c_norm == "gray_world":
            try:
                img_resized = gray_world(img_resized.astype(np.uint8), strength=0.8, max_gain=1.8)
            except Exception:
                pass
        img_float = img_resized.astype(np.float32) / 255.0
        if self.cfg.split == "train" and self.cfg.augment:
            try:
                mode = str(self.cfg.augment.get("mode", "texture")) if isinstance(self.cfg.augment, dict) else "texture"
                if self.task in {"a3", "a4"} and mode == "color":
                    img_float = apply_color_augmentations(img_float, self.cfg.augment)
                else:
                    img_float = apply_texture_augmentations(img_float, self.cfg.augment)
            except Exception:
                img_float = np.clip(img_float, 0.0, 1.0)
        img_norm = (img_float - self.mean) / self.std
        x = torch.from_numpy(img_norm.transpose(2, 0, 1)).contiguous()  # (3,H,W)

        # Optional color feature channels for A3/A4 to accentuate class cues
        if self.task in {"a3", "a4"}:
            feat_key = f"color_features_{self.task}"
            feats = getattr(self.cfg, feat_key, None)
            if feats is None:
                feats = getattr(self.cfg, "color_features", None)
            if feats:
                extras: List[np.ndarray] = []
                rgb01 = np.clip(img_float, 0.0, 1.0)
                if "redness" in feats:
                    r = rgb01[:, :, 0]
                    g = rgb01[:, :, 1]
                    b = rgb01[:, :, 2]
                    red = np.clip(r - np.maximum(g, b), 0.0, 1.0)
                    extras.append(red)
                if "darkness" in feats:
                    hsv = cv2.cvtColor((rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
                    v = hsv[:, :, 2].astype(np.float32) / 255.0
                    dark = 1.0 - v
                    extras.append(dark)
                if "hsv" in feats:
                    hsv = cv2.cvtColor((rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                    h = hsv[:, :, 0] / 179.0
                    s = hsv[:, :, 1] / 255.0
                    v = hsv[:, :, 2] / 255.0
                    extras.extend([h, s, v])
                if extras:
                    extra_stack = np.stack(extras, axis=2).astype(np.float32)
                    extra_norm = (extra_stack - 0.5) / 0.5  # map to [-1,1]
                    x_extra = torch.from_numpy(extra_norm.transpose(2, 0, 1)).contiguous()
                    x = torch.cat([x, x_extra], dim=0)

        if self.use_union_channel and union_resized is not None:
            union_float = (union_resized > 0).astype(np.float32)
            union_tensor = torch.from_numpy(union_float).unsqueeze(0)
            x = torch.cat([x, union_tensor], dim=0)

        if self.use_mask_channel:
            mask_resized = _resize_with_padding_mask(mask2d.astype(np.uint8), self.target_hw)
            mask_float = (mask_resized > 0).astype(np.float32)
            m = torch.from_numpy(mask_float).unsqueeze(0)  # (1,H,W)
            x = torch.cat([x, m], dim=0)

        y = int(row["target"])  # 0/1
        sample: Dict[str, torch.Tensor | int | str] = {
            "x": x,
            "y": torch.tensor(y, dtype=torch.long),
            "annotation_id": int(row["annotation_id"]) if "annotation_id" in row else -1,
            "image_id": int(row["image_id"]) if "image_id" in row else -1,
            "class_label": str(row["class_label"]),
            "path": str(row["crop_path"]),
            "mask_usage": "mask" if self.use_mask_channel else "rgb",
        }
        for col in self.slice_columns:
            sample[col] = row.get(col, "")
        if self.transform is not None:
            sample["x"] = self.transform(sample["x"])  # type: ignore
        return sample


__all__ = ["BerryClassificationDataset", "ClassificationDataConfig"]





print()
