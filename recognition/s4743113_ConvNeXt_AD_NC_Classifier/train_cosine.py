"""Run the fixed-epoch cosine-annealing ConvNeXt experiment."""

from __future__ import annotations

import sys
from pathlib import Path

from train import main


SCRIPT_DIR = Path(__file__).resolve().parent
COSINE_OUTPUT_DIR = SCRIPT_DIR / "output" / "convnext_tiny_slice_level_cosine"


if __name__ == "__main__":
    # Insert experiment defaults before any caller-provided arguments so an
    # explicit command-line value can still override them.
    sys.argv[1:1] = [
        "--epochs",
        "100",
        "--learning-rate",
        "0.0001",
        "--learning-rate-scheduler",
        "cosine",
        "--disable-early-stopping",
        "--output-dir",
        str(COSINE_OUTPUT_DIR),
    ]
    main()
