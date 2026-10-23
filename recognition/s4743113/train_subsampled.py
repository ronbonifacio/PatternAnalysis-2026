"""Run the rotating four-slice-per-scan ConvNeXt pilot experiment."""

from __future__ import annotations

import sys
from pathlib import Path

from train import main


SCRIPT_DIR = Path(__file__).resolve().parent
PILOT_OUTPUT_DIR = SCRIPT_DIR / "output" / "convnext_tiny_slice_subsampling"


if __name__ == "__main__":
    # Caller-provided arguments appear later and can override these defaults.
    sys.argv[1:1] = [
        "--epochs",
        "100",
        "--learning-rate",
        "0.0001",
        "--learning-rate-scheduler",
        "cosine",
        "--disable-early-stopping",
        "--training-seed",
        "123",
        "--slices-per-scan-per-epoch",
        "4",
        "--output-dir",
        str(PILOT_OUTPUT_DIR),
    ]
    main()
