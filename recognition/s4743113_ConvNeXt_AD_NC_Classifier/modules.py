# Prompt:
# Convert the official TorchVision ConvNeXt-Tiny source into a standalone,
# layer-by-layer PyTorch implementation without using the prebuilt model. Use
# it to classify each 2D ADNI slice independently while keeping subjects intact
# when the data is split. Keep attribution to the ConvNeXt paper and
# TorchVision source.
#
# Superseded prompt from the first experiment:
# Adapt the model by grouping each subject's slice features, averaging them as
# in the basic CNN classifier, and producing one prediction per subject.

"""Standalone ConvNeXt modules for slice-level ADNI classification.

The architecture follows ConvNeXt-Tiny from "A ConvNet for the 2020s" and the
TorchVision reference implementation, but is assembled here from PyTorch
layers instead of calling a prebuilt model.

The active model follows TorchVision's ordinary forward pass. Its three input
channels contain adjacent MRI slices and it produces one AD/NC prediction for
the centre slice. The earlier mean-pooling model is preserved under
``superseded/modules_subject_level_superseded.py``.

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
