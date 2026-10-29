"""Train ConvNeXt-Tiny with mild MRI-specific data augmentation."""

from __future__ import annotations

import sys
from pathlib import Path

from train import main


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "convnext_tiny_64px_augmented"


if __name__ == "__main__":
    # Caller-provided arguments appear later and can override these defaults.
    sys.argv[1:1] = [
        "--epochs",
        "100",
        "--batch-size",
        "8",
        "--image-size",
        "64",
        "--input-mode",
        "resize",
        "--learning-rate",
        "0.0001",
        "--weight-decay",
        "0.05",
        "--stochastic-depth",
        "0.1",
        "--label-smoothing",
        "0.1",
        "--amp-dtype",
        "bfloat16",
        "--augment-training",
        "--rotation-degrees",
        "5",
        "--translation-fraction",
        "0.05",
        "--scale-jitter",
        "0.05",
        "--intensity-jitter",
        "0.10",
        "--noise-standard-deviation",
        "0.01",
        "--learning-rate-scheduler",
        "cosine",
        "--disable-early-stopping",
        "--test-checkpoint",
        "accuracy",
        "--training-seed",
        "123",
        "--output-dir",
        str(OUTPUT_DIR),
    ]
    main()
