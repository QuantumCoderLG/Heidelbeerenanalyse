#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


def _normalize_name(value: str) -> str:
    mapping = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
        "Ä": "ae",
        "Ö": "oe",
        "Ü": "ue",
    }
    for src, dst in mapping.items():
        value = value.replace(src, dst)
    return value.lower()


def _load_manual_list_file(path: Optional[Path]) -> List[str]:
    if not path:
        return []
    if not path.exists():
        raise FileNotFoundError(f"train-pos list file not found: {path}")
    items: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    return items


def _assign_named(df: pd.DataFrame, names: Sequence[str], split: str, split_map: Dict[str, str]) -> None:
    for name in names:
        matches = df[df["basename"] == name]
        if matches.empty:
            raise ValueError(f"No crop found with name {name!r}")
        for path in matches["crop_path"]:
            prev = split_map.get(path)
            if prev and prev != split:
                raise ValueError(f"Crop {path} already assigned to {prev}, cannot reassign to {split}")
            split_map[path] = split


def _distribute_class(
    df: pd.DataFrame,
    class_label: str,
    split_map: Dict[str, str],
    train_frac: float = 0.65,
    group_key: str = "image_id",
) -> None:
    subset = df[df["class_label"] == class_label].copy()
    subset = subset[~subset["crop_path"].isin(split_map)]
    if subset.empty:
        return

    cls_norm = _normalize_name(class_label)
    subset["norm_basename"] = subset["basename"].apply(_normalize_name)

    # Everything not starting with the normalized class name goes straight to train
    non_original = subset[~subset["norm_basename"].str.startswith(cls_norm)]
    for path in non_original["crop_path"]:
        split_map[path] = "train"

    remaining = subset[subset["norm_basename"].str.startswith(cls_norm)].copy()
    if remaining.empty:
        return
    # Sort deterministically to reduce leakage by grouping entire original images together
    order_cols = []
    if "source_subgroup" in remaining.columns:
        order_cols.append("source_subgroup")
    if "ordered_id" in remaining.columns:
        # Fill NaNs to ensure total ordering
        remaining["ordered_id"] = remaining["ordered_id"].fillna(-1)
        order_cols.append("ordered_id")
    order_cols.append("basename")
    remaining = remaining.sort_values(order_cols)

    # Group by original image identifier (prevents leakage across same scene)
    group_key_use = group_key if group_key in remaining.columns else "base_stem"
    groups = [group for _, group in remaining.groupby(group_key_use, sort=False)]

    total = len(subset)
    already_train = len(non_original)
    rest = total - already_train
    target_train = int(math.ceil(rest * float(train_frac)))
    target_train = min(target_train, rest)

    train_groups: List[pd.DataFrame] = []
    leftover_groups: List[pd.DataFrame] = []
    count_train = 0
    for group in groups:
        if count_train < target_train:
            train_groups.append(group)
            count_train += len(group)
        else:
            leftover_groups.append(group)

    for group in train_groups:
        for path in group["crop_path"]:
            split_map[path] = "train"

    # Leave leftover groups unassigned here; final val/test assignment is done globally below


def _write_split(path: Path, items: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(f"{item}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate manual A3 splits (red vs not-red) with optional fixed train positives "
            "and a controlled, per-class test size."
        )
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/instance_crops/metadata/crops.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/instance_crops/splits/a3"))
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--train-pos-list",
        type=Path,
        default=None,
        help=(
            "Optional path to a text file with one basename per line (e.g., Red_1_7_id020.png) "
            "to force specific 'red' crops into the train split."
        ),
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=2,
        help=(
            "Exact number of crops per class to place into test from the non-train pool. "
            "Classes considered: red, yellow, green."
        ),
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.65,
        help="Target fraction of each class routed to train (remaining split into val/test).",
    )
    parser.add_argument(
        "--print-split",
        action="store_true",
        help="Print the exact split mapping (split,class,crop_path) to stdout after generation.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    df["basename"] = df["crop_path"].apply(lambda p: Path(p).name)
    # Group crops from same original berry together (avoid leakage)
    df["base_stem"] = df["basename"].str.replace(r"_id\d+\.png$", "", regex=True)

    # Limit to A3-relevant classes
    df["class_label"] = df["class_label"].str.lower().replace({"green": "green"})
    df = df[df["class_label"].isin(["red", "yellow", "green"])].copy()

    # Precompute helpers
    df["norm_basename"] = df["basename"].apply(_normalize_name)
    def _class_from_path(p: str) -> str:
        parts = Path(p).parts
        # expected: data/instance_crops/images/<class>/...
        return (parts[3].lower() if len(parts) > 3 else "")
    df["path_class_dir"] = df["crop_path"].apply(_class_from_path)

    split_map: Dict[str, str] = {}

    # Optional: force a manual set of 'red' positives into train
    train_pos = _load_manual_list_file(args.train_pos_list)
    if train_pos:
        _assign_named(df, train_pos, "train", split_map)

    # Always force: items located in 'red' folder but named like 'yellow' or 'green' -> TRAIN
    # (This addresses mislabeled file naming while keeping folder info as a strong cue.)
    mask_red_folder_mismatch = (
        (df["path_class_dir"] == "red")
        & (
            df["norm_basename"].str.startswith("yellow")
            | df["norm_basename"].str.startswith("green")
        )
    )
    forced_count = int(mask_red_folder_mismatch.sum())
    if forced_count:
        for path in df.loc[mask_red_folder_mismatch, "crop_path"].tolist():
            if path not in split_map:
                split_map[path] = "train"
        print(f"Forced red-folder mismatches to train: {forced_count}")

    # Distribute all classes (red positives, yellow/green negatives) similarly to A2 logic
    for cls in ["red", "yellow", "green"]:
        _distribute_class(df, cls, split_map, train_frac=float(args.train_frac))

    # Build lists after initial distribution
    train_list = sorted([path for path, split in split_map.items() if split == "train"])
    all_paths = set(df["crop_path"].tolist())
    non_train = sorted([p for p in all_paths if split_map.get(p) != "train"])

    # Select exactly N per class for test from the non-train pool; remaining go to val
    per_class_pool: Dict[str, List[str]] = {}
    for p in non_train:
        parts = Path(p).parts
        cls = parts[3] if len(parts) > 3 else "unknown"
        per_class_pool.setdefault(cls, []).append(p)

    # Seeded random sampling for test selection (consistent with A2)
    rng = random.Random(args.seed)
    test_list: List[str] = []
    for cls, items in per_class_pool.items():
        if not items:
            continue
        k = min(int(args.test_per_class), len(items))
        chosen = rng.sample(items, k) if k > 0 else []
        test_list.extend(chosen)
        per_class_pool[cls] = [x for x in items if x not in chosen]

    val_list: List[str] = []
    for items in per_class_pool.values():
        val_list.extend(items)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _write_split(args.out_dir / "train.txt", train_list)
    _write_split(args.out_dir / "val.txt", sorted(set(val_list)))
    _write_split(args.out_dir / "test.txt", sorted(set(test_list)))

    def _class_counts(items: Sequence[str]) -> Dict[str, int]:
        if not items:
            return {}
        labels = [Path(item).parts[3] for item in items]
        return dict(pd.Series(labels).value_counts().sort_index())

    summary = {
        "train": {"total": len(train_list), "class_counts": _class_counts(train_list)},
        "val": {"total": len(val_list), "class_counts": _class_counts(val_list)},
        "test": {"total": len(test_list), "class_counts": _class_counts(test_list)},
    }
    print("Split summary:", summary)

    if args.print_split:
        print("split,class,crop_path")
        def _cls(p: str) -> str:
            parts = Path(p).parts
            return parts[3] if len(parts) > 3 else "unknown"
        for p in train_list:
            print(f"train,{_cls(p)},{p}")
        for p in sorted(set(val_list)):
            print(f"val,{_cls(p)},{p}")
        for p in sorted(set(test_list)):
            print(f"test,{_cls(p)},{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
