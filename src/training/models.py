from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.deeplabv3 import ASPP


class DeepLabHeadV3Plus(nn.Module):
    def __init__(
        self,
        in_channels: int,
        low_level_channels: int,
        num_classes: int,
        atrous_rates: Sequence[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.aspp = ASPP(in_channels, out_channels=256, atrous_rates=atrous_rates)
        self.project_low_level = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        classifier_layers = [
            nn.Conv2d(256 + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            classifier_layers.append(nn.Dropout2d(p=dropout))
        classifier_layers.append(nn.Conv2d(256, num_classes, kernel_size=1))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor, low_level: torch.Tensor) -> torch.Tensor:
        high = self.aspp(x)
        low = self.project_low_level(low_level)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([high, low], dim=1)
        return self.classifier(x)


class DeepLabV3Plus(_SimpleSegmentationModel):
    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = features["out"]
        low_level = features["low_level"]
        x = self.classifier(x, low_level)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result = OrderedDict()
        result["out"] = x
        if self.aux_classifier is not None and "aux" in features:
            aux = self.aux_classifier(features["aux"])
            aux = F.interpolate(aux, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = aux
        return result


def build_deeplabv3plus_resnet50(
    num_classes: int,
    pretrained: bool = True,
    aux_loss: bool = False,
    atrous_rates: Sequence[int] | None = None,
    dropout: float = 0.1,
    output_stride: int = 16,
) -> DeepLabV3Plus:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    if output_stride not in {8, 16, 32}:
        raise ValueError(f"Unsupported output_stride: {output_stride}")
    rswd = [False, False, False]
    if output_stride == 8:
        rswd = [False, True, True]
    elif output_stride == 16:
        rswd = [False, True, False]
    elif output_stride == 32:
        rswd = [False, False, False]
    backbone = resnet50(weights=weights, replace_stride_with_dilation=rswd)
    return_layers = {"layer4": "out", "layer1": "low_level"}
    if aux_loss:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    if atrous_rates is None:
        atrous = (12, 24, 36) if output_stride == 8 else (6, 12, 18) if output_stride == 16 else (3, 6, 9)
    else:
        atrous = tuple(atrous_rates)
    classifier = DeepLabHeadV3Plus(2048, 256, num_classes, atrous, dropout=dropout)
    aux_classifier = None
    if aux_loss:
        aux_classifier = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )
    model = DeepLabV3Plus(backbone, classifier, aux_classifier)
    return model


def freeze_backbone_blocks(model: nn.Module, blocks: Iterable[int | str]) -> None:
    mapping = {
        0: ["backbone.conv1", "backbone.bn1"],
        1: ["backbone.layer1"],
        2: ["backbone.layer2"],
        3: ["backbone.layer3"],
        4: ["backbone.layer4"],
    }
    alias = {
        "stem": 0,
        "conv": 0,
        "layer0": 0,
        "layer1": 1,
        "layer2": 2,
        "layer3": 3,
        "layer4": 4,
        "block0": 0,
        "block1": 1,
        "block2": 2,
        "block3": 3,
        "block4": 4,
    }
    normalized: set[int] = set()
    for block in blocks:
        if isinstance(block, str):
            key = block.lower().strip()
            if key.isdigit():
                normalized.add(int(key))
                continue
            if key not in alias:
                raise ValueError(f"Unknown freeze block identifier: {block}")
            normalized.add(alias[key])
        elif isinstance(block, int):
            normalized.add(block)
        else:
            raise TypeError("freeze_backbone_blocks expects ints or strings")
    for idx in normalized:
        targets = mapping.get(idx)
        if not targets:
            continue
        for target in targets:
            module = model
            parts = target.split(".")
            for part in parts:
                module = getattr(module, part)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def build_model(config: dict, num_classes: int = 1) -> nn.Module:
    name = config.get("name", "deeplabv3plus")
    if name.lower() not in {"deeplabv3plus", "deeplabv3+", "deeplabv3"}:
        raise ValueError(f"Unsupported model name: {name}")
    model = build_deeplabv3plus_resnet50(
        num_classes=num_classes,
        pretrained=bool(config.get("pretrained", True)),
        aux_loss=bool(config.get("aux_loss", False)),
        atrous_rates=config.get("aspp_dilate"),
        dropout=float(config.get("dropout", 0.1) or 0.0),
        output_stride=int(config.get("output_stride", 16)),
    )
    freeze = config.get("freeze_blocks", [])
    if freeze:
        freeze_backbone_blocks(model, freeze)
    return model


__all__ = [
    "build_model",
    "build_deeplabv3plus_resnet50",
    "freeze_backbone_blocks",
    "DeepLabV3Plus",
    "DeepLabHeadV3Plus",
]
