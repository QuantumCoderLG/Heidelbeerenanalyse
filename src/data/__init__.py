from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import albumentations as A
import inspect
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset

from ..config.paths import project_root


LOGGER = logging.getLogger(__name__)


@dataclass
class Sample:
    image_path: Path
    mask_path: Path
    instance_mask_path: Path
    image_id: int
    width: int
    height: int
    num_instances: int


class BlueberrySegmentationDataset(Dataset):
    """Dataset for blueberry instance segmentation using processed COCO exports."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_root_override: str | Path | None = None,
        transforms: Optional[Callable] = None,
        sanity_checks: bool = True,
        cache_images: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        self.transforms = transforms
        self.cache_images = bool(cache_images)
        self.samples = self._load_samples(image_root_override=image_root_override, sanity_checks=sanity_checks)
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        if self.cache_images:
            self._warm_cache()

    def _load_samples(
        self,
        image_root_override: str | Path | None,
        sanity_checks: bool,
    ) -> List[Sample]:
        ann_path = self.split_dir / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"annotations.json not found in {self.split_dir}")
        with ann_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in annotations:
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        repo_root = project_root()
        if image_root_override is not None:
            image_root_override = Path(image_root_override)

        samples: List[Sample] = []
        for img in images:
            image_id = int(img["id"])
            file_name = Path(img["file_name"])
            if file_name.is_absolute():
                image_path = file_name
            else:
                if image_root_override is not None:
                    image_path = image_root_override / file_name.name
                else:
                    image_path = repo_root / file_name
            if not image_path.exists():
                raise FileNotFoundError(f"Image path not found: {image_path}")
            stem = image_path.stem
            mask_dir = self.split_dir / "masks" / stem
            mask_path = mask_dir / "instances.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Instance mask missing: {mask_path}")
            instance_mask_path = mask_path
            anns = anns_by_image.get(image_id, [])
            width = int(img.get("width", 0))
            height = int(img.get("height", 0))
            sample = Sample(
                image_path=image_path,
                mask_path=mask_path,
                instance_mask_path=instance_mask_path,
                image_id=image_id,
                width=width,
                height=height,
                num_instances=len(anns),
            )
            if sanity_checks:
                _sanity_check_mask(sample)
            samples.append(sample)

        samples.sort(key=lambda s: (s.image_path.stem, s.image_id))
        return samples

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        if self.cache_images and idx in self._cache:
            cached = self._cache[idx]
            image = cached["image"].copy()
            instances = cached["instances"].copy()
        else:
            image, instances = self._load_image_and_instances(sample)
            if self.cache_images:
                self._cache[idx] = {
                    "image": image.copy(),
                    "instances": instances.copy(),
                }
        binary_mask = (instances > 0).astype(np.uint8)

        transformed: Dict[str, Any]
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=binary_mask, instance_mask=instances)
            image = transformed["image"]
            binary_mask = transformed["mask"].float()
            instances = transformed["instance_mask"].long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            binary_mask = torch.from_numpy(binary_mask.astype(np.float32)).unsqueeze(0)
            instances = torch.from_numpy(instances.astype(np.int64))
            image = image / 255.0

        if isinstance(instances, np.ndarray):
            instances = torch.from_numpy(instances.astype(np.int64))
        if isinstance(instances, torch.Tensor) and instances.dtype != torch.int64:
            instances = instances.to(dtype=torch.int64)
        if instances.ndim == 3:
            instances = instances.squeeze(0)
        meta = {
            "image_id": sample.image_id,
            "image_path": str(sample.image_path),
            "original_size": (sample.height, sample.width),
            "num_instances": sample.num_instances,
        }
        return {
            "image": image,
            "mask": binary_mask if binary_mask.ndim == 3 else binary_mask.unsqueeze(0),
            "instance_mask": instances,
            "meta": meta,
        }

    def _load_image_and_instances(self, sample: Sample) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {sample.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances = cv2.imread(str(sample.instance_mask_path), cv2.IMREAD_UNCHANGED)
        if instances is None:
            raise RuntimeError(f"Failed to read instance mask: {sample.instance_mask_path}")
        if instances.ndim != 2:
            raise ValueError("Instance mask must be single-channel")
        return image, instances

    def _warm_cache(self) -> None:
        for idx, sample in enumerate(self.samples):
            image, instances = self._load_image_and_instances(sample)
            self._cache[idx] = {
                "image": image,
                "instances": instances,
            }


def _sanity_check_mask(sample: Sample) -> None:
    mask = cv2.imread(str(sample.instance_mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask for sanity check: {sample.instance_mask_path}")
    unique_vals = np.unique(mask)
    if unique_vals.dtype == np.float32:
        raise ValueError(f"Instance mask has float dtype: {sample.instance_mask_path}")
    if unique_vals.size == 0:
        raise ValueError(f"Instance mask empty: {sample.instance_mask_path}")
    if np.any(unique_vals < 0):
        raise ValueError(f"Instance mask contains negative values: {sample.instance_mask_path}")


def build_transforms(
    image_mean: Iterable[float],
    image_std: Iterable[float],
    augment: bool = True,
    size: Optional[Tuple[int, int]] = None,
    keep_ratio: bool = True,
) -> Callable:
    additional_targets = {"instance_mask": "mask"}
    transforms: List[A.BasicTransform] = []
    if augment:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.6),
                A.Perspective(scale=(0.01, 0.03), keep_size=True, p=0.1),
                A.RandomBrightnessContrast(p=0.6),
                A.HueSaturationValue(p=0.4),
                A.RGBShift(p=0.2),
                _gauss_noise_transform(p=0.2),
                A.RandomShadow(shadow_dimension=5, p=0.05),
            ]
        )
    # Memory control: enforce spatial size if configured
    if size is not None and isinstance(size, (tuple, list)):
        h, w = int(size[0]), int(size[1])
        if keep_ratio:
            max_side = max(h, w)
            transforms.append(A.LongestMaxSize(max_size=max_side, p=1.0))
            transforms.append(
                A.PadIfNeeded(
                    min_height=h,
                    min_width=w,
                    border_mode=cv2.BORDER_CONSTANT,
                )
            )
        else:
            transforms.append(A.Resize(height=h, width=w))
    transforms.extend(
        [
            A.Normalize(mean=list(image_mean), std=list(image_std)),
            ToTensorV2(transpose_mask=True),
        ]
    )
    return A.Compose(transforms, additional_targets=additional_targets)


def _gauss_noise_transform(p: float = 0.2) -> A.BasicTransform:
    """Create a GaussNoise/GaussianNoise transform compatible with various A versions."""
    # Prefer GaussianNoise if present, else GaussNoise
    noise_cls = getattr(A, "GaussianNoise", None) or getattr(A, "GaussNoise", None)
    if noise_cls is None:
        return A.NoOp(p=0.0)
    try:
        sig = inspect.signature(noise_cls)
        params = sig.parameters
        if "variance_limit" in params:
            return noise_cls(variance_limit=(5.0, 15.0), p=p)
        if "var_limit" in params:
            return noise_cls(var_limit=(5.0, 15.0), p=p)
    except Exception:
        pass
    # Fallback without kwargs if version is unusual
    try:
        return noise_cls(p=p)
    except Exception:
        return A.NoOp(p=0.0)


def seed_worker(worker_id: int) -> None:  # pragma: no cover - nondeterministic utility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    instances = torch.stack([item["instance_mask"] for item in batch], dim=0)
    metas = [item["meta"] for item in batch]
    return {
        "image": images,
        "mask": masks,
        "instance_mask": instances,
        "meta": metas,
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    drop_last: bool,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    try:
        dataset_len = len(dataset)  # type: ignore[arg-type]
    except TypeError:  # pragma: no cover - exotic dataset
        dataset_len = None

    if drop_last and dataset_len is not None and dataset_len < batch_size:
        LOGGER.warning(
            "DataLoader drop_last=True would drop the only batch (dataset size %s < batch size %s); forcing drop_last=False",
            dataset_len,
            batch_size,
        )
        drop_last = False

    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "worker_init_fn": seed_worker,
        "collate_fn": _collate,
    }
    if num_workers > 0 and prefetch_factor:
        loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return DataLoader(**loader_kwargs)


def make_subsets(dataset: Dataset, indices: List[int]) -> Subset:
    return Subset(dataset, indices)


from . import coco_schema, ids, matching, metadata, rasterize, xml_parser


__all__ = [
    "Sample",
    "BlueberrySegmentationDataset",
    "build_transforms",
    "create_dataloader",
    "make_subsets",
    "seed_worker",
    "coco_schema",
    "ids",
    "matching",
    "metadata",
    "rasterize",
    "xml_parser",
]
