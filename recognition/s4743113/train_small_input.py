"""Train ConvNeXt on all slices using 64-pixel inputs and batches of eight."""

from __future__ import annotations

import sys
from pathlib import Path

from train import main


SCRIPT_DIR = Path(__file__).resolve().parent
SMALL_INPUT_OUTPUT_DIR = SCRIPT_DIR / "output" / "convnext_tiny_64px_batch8"


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
        "--learning-rate-scheduler",
        "cosine",
        "--disable-early-stopping",
        "--training-seed",
        "123",
        "--output-dir",
        str(SMALL_INPUT_OUTPUT_DIR),
    ]
    main()
