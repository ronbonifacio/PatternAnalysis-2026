"""Train the 64-pixel ConvNeXt experiment with stronger regularisation."""

from __future__ import annotations

import sys
from pathlib import Path

from train import main


SCRIPT_DIR = Path(__file__).resolve().parent
REGULARIZED_OUTPUT_DIR = SCRIPT_DIR / "output" / "convnext_tiny_64px_regularized"


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
        "--learning-rate-scheduler",
        "cosine",
        "--disable-early-stopping",
        "--test-checkpoint",
        "accuracy",
        "--training-seed",
        "123",
        "--output-dir",
        str(REGULARIZED_OUTPUT_DIR),
    ]
    main()
