"""Train the patient-balanced ConvNeXt-Tiny generalisation experiment."""

from __future__ import annotations

import sys
from pathlib import Path

from train import main


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "convnext_tiny_64px_generalized"


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
        "--split-level",
        "patient",
        "--normalization",
        "per-slice",
        "--exclude-scan",
        "1012896",
        "--balance-patients",
        "--augment-training",
        "--rotation-degrees",
        "7",
        "--translation-fraction",
        "0.05",
        "--scale-jitter",
        "0.05",
        "--intensity-jitter",
        "0.10",
        "--noise-standard-deviation",
        "0.01",
        "--horizontal-flip-probability",
        "0.5",
        "--gamma-jitter",
        "0.2",
        "--learning-rate-scheduler",
        "cosine",
        "--early-stopping-start-epoch",
        "10",
        "--test-checkpoint",
        "loss",
        "--calibrate-threshold",
        "--training-seed",
        "123",
        "--output-dir",
        str(OUTPUT_DIR),
    ]
    main()
