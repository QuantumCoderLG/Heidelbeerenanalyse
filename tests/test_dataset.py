from pathlib import Path

from src.data import BlueberrySegmentationDataset, build_transforms


def test_dataset_sanity():
    root = Path("data/processed")
    transforms = build_transforms([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], augment=False)
    dataset = BlueberrySegmentationDataset(root=root, split="train", transforms=transforms)
    assert len(dataset) > 0
    sample = dataset[0]
    image = sample["image"]
    mask = sample["mask"]
    instances = sample["instance_mask"]
    assert image.shape[0] == 3
    assert mask.shape[0] == 1
    unique_vals = instances.unique().tolist()
    assert 0 in unique_vals
    assert any(v > 0 for v in unique_vals)
    assert mask.max().item() <= 1.0
    assert mask.min().item() >= 0.0
