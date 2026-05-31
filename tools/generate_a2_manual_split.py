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


def _load_manual_list(raw: str) -> List[str]:
    return [line.strip() for line in raw.strip().splitlines() if line.strip()]

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


def _distribute_negatives(df: pd.DataFrame, class_label: str, split_map: Dict[str, str]) -> None:
    subset = df[df["class_label"] == class_label].copy()
    subset = subset[~subset["crop_path"].isin(split_map)]
    if subset.empty:
        return

    cls_norm = _normalize_name(class_label)
    subset["norm_basename"] = subset["basename"].apply(_normalize_name)

    non_original = subset[~subset["norm_basename"].str.startswith(cls_norm)]
    for path in non_original["crop_path"]:
        split_map[path] = "train"

    remaining = subset[subset["norm_basename"].str.startswith(cls_norm)].copy()
    if remaining.empty:
        return
    remaining = remaining.sort_values(["base_stem", "basename"])

    groups = [group for _, group in remaining.groupby("base_stem", sort=False)]

    total = len(subset)
    already_train = len(non_original)
    rest = total - already_train
    target_train = int(math.ceil(rest * 0.65))
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

    if not leftover_groups:
        return

    leftover_total = sum(len(group) for group in leftover_groups)
    val_target = int(math.ceil(leftover_total * 0.7))
    count_val = 0
    for group in leftover_groups:
        target_split = "val" if count_val < val_target else "test"
        for path in group["crop_path"]:
            split_map[path] = target_split
        count_val += len(group)


def _write_split(path: Path, items: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(f"{item}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate manual A2 splits with controlled positives and fixed test size per class.")
    parser.add_argument("--metadata", type=Path, default=Path("data/instance_crops/metadata/crops.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/instance_crops/splits/a2"))
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for deterministic sampling.")
    parser.add_argument("--test-per-class", type=int, default=2, help="Exact number of crops per class to place into test from the non-train pool.")
    parser.add_argument(
        "--train-pos-list",
        type=Path,
        default=None,
        help=(
            "Optional path to a text file with one basename per line (e.g., Red_1_7_id020.png) "
            "to force specific 'never' positives into the train split."
        ),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    df["basename"] = df["crop_path"].apply(lambda p: Path(p).name)
    df["base_stem"] = df["basename"].str.replace(r"_id\d+\.png$", "", regex=True)

    split_map: Dict[str, str] = {}

    # Positives defined by user
    train_pos = _load_manual_list(
        """
Yellow_1_11_id018.png
Green_1_1_id025.png
Red_1_7_id020.png
Red_1_7_id021.png
Red_1_7_id022.png
Red_1_7_id024.png
Red_1_7_id025.png
Never_1_1_id001.png
Never_1_1_id002.png
Never_1_1_id003.png
Never_1_1_id004.png
Never_1_1_id005.png
Never_1_1_id006.png
Never_1_1_id007.png
Never_1_1_id008.png
Never_1_1_id009.png
Never_1_1_id010.png
Never_1_1_id011.png
Never_1_1_id012.png
Never_1_1_id013.png
Never_1_1_id014.png
Never_1_1_id015.png
Never_1_1_id016.png
Never_1_1_id017.png
Never_1_1_id018.png
Never_1_1_id019.png
Never_1_1_id020.png
Never_1_1_id021.png
Never_1_1_id022.png
Never_1_1_id023.png
Never_1_1_id024.png
Never_1_1_id025.png
Never_1_2_id001.png
Never_1_2_id002.png
Never_1_2_id003.png
Never_1_2_id004.png
Never_1_2_id005.png
Never_1_2_id006.png
Never_1_2_id007.png
Never_1_2_id008.png
Never_1_2_id009.png
Never_1_2_id010.png
Never_1_2_id011.png
Never_1_2_id012.png
Never_1_2_id013.png
Never_1_2_id014.png
Never_1_2_id015.png
Never_1_2_id016.png
Never_1_2_id017.png
Never_1_2_id018.png
Never_1_2_id019.png
Never_1_2_id020.png
Never_1_2_id021.png
Never_1_2_id022.png
Never_1_2_id023.png
Never_1_2_id024.png
Never_1_2_id025.png
Never_1_3_id001.png
Never_1_3_id002.png
Never_1_3_id003.png
Never_1_3_id004.png
Never_1_3_id005.png
Never_1_3_id006.png
Never_1_3_id007.png
Never_1_3_id008.png
Never_1_3_id009.png
Never_1_3_id010.png
Never_1_3_id011.png
Never_1_3_id012.png
Never_1_3_id013.png
Never_1_3_id014.png
Never_1_3_id015.png
Never_1_3_id016.png
Never_1_3_id017.png
Never_1_3_id018.png
Never_1_3_id019.png
Never_1_3_id020.png
Never_1_3_id021.png
Never_1_3_id022.png
Never_1_3_id023.png
Never_1_3_id024.png
Never_1_3_id025.png
Never_1_4_id001.png
Never_1_4_id002.png
Never_1_4_id003.png
Never_1_4_id004.png
Never_1_4_id005.png
Never_1_4_id006.png
Never_1_4_id007.png
Never_1_4_id008.png
Never_1_4_id009.png
Never_1_4_id010.png
Never_1_4_id011.png
Never_1_4_id012.png
Never_1_4_id013.png
Never_1_4_id014.png
Never_1_4_id015.png
Never_1_4_id016.png
Never_1_4_id017.png
Never_1_4_id018.png
Never_1_4_id019.png
Never_1_4_id020.png
Never_1_4_id021.png
Never_1_4_id022.png
Never_1_4_id023.png
Never_1_4_id024.png
Never_1_4_id025.png
Never_1_5_id001.png
Never_1_5_id002.png
Never_1_5_id003.png
Never_1_5_id004.png
Never_1_5_id005.png
Never_1_5_id006.png
Never_1_5_id007.png
Never_1_5_id008.png
Never_1_5_id009.png
Never_1_5_id010.png
Never_1_5_id011.png
Never_1_5_id012.png
Never_1_5_id013.png
Never_1_5_id014.png
Never_1_5_id015.png
Never_1_5_id016.png
Never_1_5_id017.png
Never_1_5_id018.png
Never_1_5_id019.png
Never_1_5_id020.png
Never_1_5_id021.png
Never_1_5_id022.png
Never_1_5_id023.png
Never_1_5_id024.png
Never_1_5_id025.png
Never_1_6_id001.png
Never_1_6_id002.png
Never_1_6_id003.png
Never_1_6_id004.png
Never_1_6_id005.png
Never_1_6_id006.png
Never_1_6_id007.png
Never_1_6_id008.png
Never_1_6_id009.png
Never_1_6_id010.png
Never_1_6_id011.png
Never_1_6_id012.png
Never_1_6_id013.png
Never_1_6_id014.png
Never_1_6_id015.png
Never_1_6_id016.png
Never_1_6_id017.png
Never_1_6_id018.png
Never_1_6_id019.png
Never_1_6_id020.png
Never_1_6_id021.png
Never_1_6_id022.png
Never_1_6_id023.png
Never_1_6_id024.png
"""
    )
    # Merge optional external list (one basename per line)
    extra_pos = _load_manual_list_file(args.train_pos_list)
    if extra_pos:
        # Keep order deterministic while removing duplicates
        seen = set()
        merged: List[str] = []
        for name in list(train_pos) + list(extra_pos):
            if name in seen:
                continue
            seen.add(name)
            merged.append(name)
        train_pos = merged
    # Assign only TRAIN positives explicitly; VAL/TEST will be derived to meet constraints
    _assign_named(df, train_pos, "train", split_map)

    for cls in ["green", "yellow", "red"]:
        _distribute_negatives(df, cls, split_map)

    rng = random.Random(args.seed)
    # Build lists after train assignment and negative distribution
    # Train is fixed; everything else is considered for Val/Test construction
    train_list = sorted([path for path, split in split_map.items() if split == "train"])
    all_paths = set(df["crop_path"].tolist())
    non_train = sorted([p for p in all_paths if split_map.get(p) != "train"])
    # Select exactly N per class for test from the non-train pool; remaining go to val
    per_class_pool: Dict[str, List[str]] = {}
    for p in non_train:
        parts = Path(p).parts
        cls = parts[3] if len(parts) > 3 else "unknown"
        per_class_pool.setdefault(cls, []).append(p)
    test_list: List[str] = []
    for cls, items in per_class_pool.items():
        if not items:
            continue
        k = min(args.test_per_class, len(items))
        chosen = rng.sample(items, k)
        test_list.extend(chosen)
        per_class_pool[cls] = [x for x in items if x not in chosen]
    # Everything else (non-train and not in test) goes to val
    val_list = []
    for items in per_class_pool.values():
        val_list.extend(items)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    test_list = sorted(set(test_list))

    _write_split(args.out_dir / "train.txt", train_list)
    _write_split(args.out_dir / "val.txt", val_list)
    _write_split(args.out_dir / "test.txt", test_list)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
