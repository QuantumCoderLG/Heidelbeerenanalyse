from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torch.nn.functional as F


ClassifierBackbone = Literal[
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b5",
]


def _adapt_first_conv(module: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if module.in_channels == in_channels:
        return module
    new_conv = nn.Conv2d(
        in_channels,
        module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        bias=module.bias is not None,
    )
    with torch.no_grad():
        if in_channels > module.in_channels:
            new_conv.weight.zero_()
            new_conv.weight[:, : module.in_channels] = module.weight
        else:
            new_conv.weight.copy_(module.weight[:, :in_channels])
        if module.bias is not None:
            new_conv.bias.copy_(module.bias)
    return new_conv


def build_classifier(
    *,
    backbone: ClassifierBackbone,
    num_classes: int = 1,
    in_channels: int = 3,
    pretrained: bool = True,
    dropout: float | None = None,
) -> nn.Module:
    if backbone == "mobilenet_v3_small":
        weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.mobilenet_v3_small(weights=weights)
        feat_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(feat_dim, num_classes)
        if dropout is not None and dropout >= 0:
            # classifier[-2] is the dropout layer in torchvision's implementation
            if isinstance(model.classifier[-2], nn.Dropout):
                model.classifier[-2] = nn.Dropout(p=dropout)
        if in_channels != 3:
            first = model.features[0][0]
            model.features[0][0] = _adapt_first_conv(first, in_channels)
        return model

    if backbone == "mobilenet_v3_large":
        weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.mobilenet_v3_large(weights=weights)
        feat_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(feat_dim, num_classes)
        if dropout is not None and dropout >= 0:
            if isinstance(model.classifier[-2], nn.Dropout):
                model.classifier[-2] = nn.Dropout(p=dropout)
        if in_channels != 3:
            first = model.features[0][0]
            model.features[0][0] = _adapt_first_conv(first, in_channels)
        return model

    if backbone == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.efficientnet_b0(weights=weights)
        feat_dim = model.classifier[-1].in_features
        drop = dropout if dropout is not None else 0.2
        model.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(feat_dim, num_classes),
        )
        if in_channels != 3:
            first = model.features[0][0]
            model.features[0][0] = _adapt_first_conv(first, in_channels)
        return model

    if backbone == "efficientnet_b1":
        try:
            weights = tv_models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
        except AttributeError:
            weights = None if not pretrained else tv_models.EfficientNet_B1_Weights.DEFAULT  # type: ignore[attr-defined]
        model = tv_models.efficientnet_b1(weights=weights)
        feat_dim = model.classifier[-1].in_features
        drop = dropout if dropout is not None else 0.25
        model.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(feat_dim, num_classes),
        )
        if in_channels != 3:
            first = model.features[0][0]
            model.features[0][0] = _adapt_first_conv(first, in_channels)
        return model

    if backbone == "efficientnet_b2":
        try:
            weights = tv_models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        except AttributeError:
            # Fallback if weight enum name differs
            weights = None if not pretrained else tv_models.EfficientNet_B2_Weights.DEFAULT  # type: ignore[attr-defined]
        model = tv_models.efficientnet_b2(weights=weights)
        feat_dim = model.classifier[-1].in_features
        drop = dropout if dropout is not None else 0.3
        model.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(feat_dim, num_classes),
        )
        if in_channels != 3:
            first = model.features[0][0]
            model.features[0][0] = _adapt_first_conv(first, in_channels)
        return model

    if backbone == "efficientnet_b3":
        try:
            weights = tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        except AttributeError:
            weights = None if not pretrained else tv_models.EfficientNet_B3_Weights.DEFAULT  # type: ignore[attr-defined]
        model = tv_models.efficientnet_b3(weights=weights)
        feat_dim = model.classifier[-1].in_features
        drop = dropout if dropout is not None else 0.3
        model.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(feat_dim, num_classes),
        )
        if in_channels != 3:
            first = model.features[0][0]
            model.features[0][0] = _adapt_first_conv(first, in_channels)
        return model

    if backbone == "efficientnet_b5":
        try:
            weights = tv_models.EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
        except AttributeError:
            weights = None if not pretrained else tv_models.EfficientNet_B5_Weights.DEFAULT  # type: ignore[attr-defined]
        model = tv_models.efficientnet_b5(weights=weights)
        feat_dim = model.classifier[-1].in_features
        drop = dropout if dropout is not None else 0.4
        model.classifier = nn.Sequential(
            nn.Dropout(p=drop),
            nn.Linear(feat_dim, num_classes),
        )
        if in_channels != 3:
            first = model.features[0][0]
            model.features[0][0] = _adapt_first_conv(first, in_channels)
        return model

    raise ValueError(f"Unsupported classifier backbone: {backbone}")


class MaskWeightedPoolingWrapper(nn.Module):
    """Wrap a torchvision classifier to use mask-weighted global pooling.

    Assumes the wrapped model exposes `.features` and `.classifier` similar to
    EfficientNet/MobileNet. The input is expected to contain a mask channel as the
    last channel when `has_mask_channel=True`.
    """

    def __init__(self, base: nn.Module, *, has_mask_channel: bool = True) -> None:
        super().__init__()
        self.base = base
        self.has_mask_channel = has_mask_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        mask: Optional[torch.Tensor] = None
        if self.has_mask_channel and x.dim() == 4 and x.size(1) >= 4:
            mask = x[:, -1:, :, :]
        # Forward until features
        if hasattr(self.base, "features"):
            feats = self.base.features(x)
        else:
            # Fallback: try running full model if "features" not available
            return self.base(x)  # type: ignore[misc]
        # Pooling
        if mask is not None:
            m = F.interpolate(mask, size=feats.shape[-2:], mode="bilinear", align_corners=False).clamp(min=0.0, max=1.0)
            # Avoid zero division: add small epsilon to denominator
            denom = m.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            feats_pooled = (feats * m).sum(dim=(2, 3), keepdim=True) / denom
        else:
            if hasattr(self.base, "avgpool") and isinstance(self.base.avgpool, nn.AdaptiveAvgPool2d):  # type: ignore[attr-defined]
                feats_pooled = self.base.avgpool(feats)  # type: ignore[operator]
            else:
                feats_pooled = F.adaptive_avg_pool2d(feats, output_size=(1, 1))
        out = torch.flatten(feats_pooled, 1)
        return self.base.classifier(out)


__all__ = ["build_classifier", "MaskWeightedPoolingWrapper"]
