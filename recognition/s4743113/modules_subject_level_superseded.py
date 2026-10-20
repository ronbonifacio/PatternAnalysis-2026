# SUPERSEDED: retained to reproduce the failed subject-level experiment.
# Prompt:
# Convert the official TorchVision ConvNeXt-Tiny source into a standalone,
# layer-by-layer PyTorch implementation without using the prebuilt model.
# Adapt it to the dataloader by encoding each 2D slice, grouping the slice
# features by subject, averaging them in the same way as the basic CNN
# classifier, and producing one AD/NC prediction per subject. Keep attribution
# to the ConvNeXt paper and TorchVision source.

"""Standalone ConvNeXt modules for subject-level ADNI classification.

The architecture follows ConvNeXt-Tiny from "A ConvNet for the 2020s" and the
TorchVision reference implementation, but is assembled here from PyTorch
layers instead of calling a prebuilt model.

TorchVision's ordinary ConvNeXt forward pass accepts independent images and
produces one prediction per image. ``SubjectConvNeXt`` adds a second level of
pooling for this project: ConvNeXt first spatially pools each MRI slice into a
768-value feature vector, then the wrapper averages those vectors across the
slices belonging to one subject and produces one prediction for that subject.

References:
    https://arxiv.org/abs/2201.03545
    https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import functional as F


__all__ = [
    "CNBlock",
    "CNBlockConfig",
    "ConvNeXt",
    "LayerNorm2d",
    "SubjectConvNeXt",
    "convnext_tiny",
]


class Permute(nn.Module):
    """Permute tensor dimensions inside an ``nn.Sequential`` block."""

    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        self.dims = tuple(dims)

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)


class LayerNorm2d(nn.LayerNorm):
    """Apply LayerNorm over the channel dimension of an NCHW tensor."""

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class StochasticDepth(nn.Module):
    """Randomly drop each input's residual branch during training."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("stochastic-depth probability must be in [0, 1]")
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return x

        survival_probability = 1.0 - self.probability
        if survival_probability == 0.0:
            return torch.zeros_like(x)

        # One decision per image, broadcast across channels and spatial axes.
        noise_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = x.new_empty(noise_shape).bernoulli_(survival_probability)
        noise.div_(survival_probability)
        return x * noise


class CNBlock(nn.Module):
    """One ConvNeXt residual block."""

    def __init__(
        self,
        dim: int,
        layer_scale: float,
        stochastic_depth_probability: float,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            # Spatial mixing: one 7x7 filter per channel.
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            Permute((0, 2, 3, 1)),
            norm_layer(dim),
            # Channel mixing: expand by four, then project back to dim.
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            Permute((0, 3, 1, 2)),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_probability)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.layer_scale * self.block(x)
        return x + self.stochastic_depth(residual)


@dataclass(frozen=True)
class CNBlockConfig:
    """Channel counts and depth for one ConvNeXt stage."""

    input_channels: int
    out_channels: int | None
    num_layers: int


class ConvNeXt(nn.Module):
    """ConvNeXt assembled from explicit PyTorch layers."""

    def __init__(
        self,
        block_setting: Sequence[CNBlockConfig],
        *,
        stochastic_depth_probability: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Callable[..., nn.Module] = CNBlock,
    ) -> None:
        super().__init__()
        if not block_setting:
            raise ValueError("block_setting cannot be empty")
        if not all(isinstance(config, CNBlockConfig) for config in block_setting):
            raise TypeError("block_setting must contain CNBlockConfig objects")

        layers: list[nn.Module] = []
        first_channels = block_setting[0].input_channels

        # Patchify stem: 224x224 becomes 56x56.
        layers.append(
            nn.Sequential(
                nn.Conv2d(3, first_channels, kernel_size=4, stride=4),
                LayerNorm2d(first_channels, eps=1e-6),
            )
        )

        total_blocks = sum(config.num_layers for config in block_setting)
        block_index = 0
        for config in block_setting:
            stage: list[nn.Module] = []
            for _ in range(config.num_layers):
                # ConvNeXt increases drop probability linearly with depth.
                denominator = max(total_blocks - 1, 1)
                drop_probability = (
                    stochastic_depth_probability * block_index / denominator
                )
                stage.append(
                    block(
                        config.input_channels,
                        layer_scale,
                        drop_probability,
                    )
                )
                block_index += 1
            layers.append(nn.Sequential(*stage))

            if config.out_channels is not None:
                layers.append(
                    nn.Sequential(
                        LayerNorm2d(config.input_channels, eps=1e-6),
                        nn.Conv2d(
                            config.input_channels,
                            config.out_channels,
                            kernel_size=2,
                            stride=2,
                        ),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        final_config = block_setting[-1]
        self.feature_dim = final_config.out_channels or final_config.input_channels
        self.norm = nn.LayerNorm(self.feature_dim, eps=1e-6)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        """Return one feature vector per input image.

        This is the same kind of *spatial* pooling used by TorchVision's
        ConvNeXt: every HxW feature map becomes one 768-value slice vector. It
        does not combine different MRI slices or subjects.
        """

        x = self.features(x)
        # Average over the remaining height and width within each slice.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.norm(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x))


def convnext_tiny(
    *,
    num_classes: int = 1000,
    stochastic_depth_probability: float = 0.1,
) -> ConvNeXt:
    """Build ConvNeXt-Tiny without using a prebuilt torchvision model."""

    block_setting = (
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    )
    return ConvNeXt(
        block_setting,
        stochastic_depth_probability=stochastic_depth_probability,
        num_classes=num_classes,
    )


class SubjectConvNeXt(nn.Module):
    """Classify subjects by mean-pooling their ConvNeXt slice features.

    This wrapper is the project-specific difference from TorchVision's image
    classifier. ``packed_slices`` holds the selected slices from every subject
    as one efficient image batch. ``slice_counts`` records where each subject's
    slices end. After ConvNeXt encodes every slice independently, the features
    are split back into subjects and averaged feature-by-feature. The operation
    matches the subject pooling used by the basic CNN classifier.

    For four subjects with 12 slices each, the shapes are::

        packed slices:    [48, 3, 224, 224]
        slice features:   [48, 768]
        subject features: [4, 768]
        output logits:    [4, num_classes]

    The slice mean gives every selected slice equal influence and discards
    slice order. It is separate from ConvNeXt's spatial average pooling.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = convnext_tiny(num_classes=num_classes)

    def forward(
        self,
        packed_slices: Tensor,
        slice_counts: Sequence[int],
    ) -> Tensor:
        if packed_slices.ndim != 4:
            raise ValueError("packed_slices must have shape [slices, C, H, W]")

        counts = tuple(int(count) for count in slice_counts)
        if not counts or any(count <= 0 for count in counts):
            raise ValueError("slice_counts must contain positive integers")
        if sum(counts) != packed_slices.shape[0]:
            raise ValueError("slice_counts must sum to the number of packed slices")

        # ConvNeXt handles the packed tensor as independent 2D images. Unlike
        # the standard TorchVision forward pass, classification is deliberately
        # delayed until after the images have been regrouped into subjects.
        slice_features = self.backbone.forward_features(packed_slices)

        # Example for counts (12, 12, 12, 12): split [48, 768] into four
        # [12, 768] tensors, then mean(dim=0) makes four [768] subject vectors.
        subject_features = torch.stack(
            [features.mean(dim=0) for features in torch.split(slice_features, counts)]
        )
        return self.backbone.classifier(subject_features)
