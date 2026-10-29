"""The original shallow CNN adapted to one prediction per centre slice.

The convolutional encoder and classifier are unchanged from the baseline
notebook.  Only the old subject-feature averaging has been removed so this
model accepts the same three-channel 2.5D samples and produces the same
slice-level output unit as the final ConvNeXt experiment.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ShallowSliceCNN(nn.Module):
    """Encode one 2.5D MRI sample with the original shallow architecture."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.slice_encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(48, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return one AD/NC logit pair per centre slice."""

        return self.classifier(self.slice_encoder(images))


def shallow_slice_cnn(
    *,
    num_classes: int = 2,
    stochastic_depth_probability: float = 0.0,
) -> ShallowSliceCNN:
    """Build the controlled baseline using the shared training interface."""

    del stochastic_depth_probability  # The shallow CNN has no residual paths.
    return ShallowSliceCNN(num_classes=num_classes)


if __name__ == "__main__":
    model = shallow_slice_cnn()
    output = model(torch.zeros(2, 3, 64, 64))
    print(model)
    print(f"Output shape: {tuple(output.shape)}")
    print(
        "Trainable parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}"
    )
