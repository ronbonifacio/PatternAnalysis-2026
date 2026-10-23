"""Train the regularised ConvNeXt experiment with 96-pixel inputs."""

from __future__ import annotations

import sys
from pathlib import Path

from train import main


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "convnext_tiny_96px_regularized"


if __name__ == "__main__":
    # Caller-provided arguments appear later and can override these defaults.
    sys.argv[1:1] = [
        "--epochs",
        "100",
        "--batch-size",
        "8",
        "--image-size",
        "96",
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
