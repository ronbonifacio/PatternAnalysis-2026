"""Connect the shallow baseline to the shared controlled training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import train as shared_training

from basic_cnn.modules import shallow_slice_cnn


BASIC_CNN_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASIC_CNN_DIR / "output"


def _has_cli_option(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


def run_shared_pipeline(*, evaluate_only: bool) -> None:
    """Run the final experiment settings with the shallow model factory."""

    if not _has_cli_option("--output-dir"):
        sys.argv.extend(("--output-dir", str(DEFAULT_OUTPUT_DIR)))
    if not _has_cli_option("--stochastic-depth"):
        # Stochastic depth is undefined for this non-residual architecture.
        sys.argv.extend(("--stochastic-depth", "0"))
    if evaluate_only and not _has_cli_option("--evaluate-only"):
        sys.argv.append("--evaluate-only")

    shared_training.main(
        model_factory=shallow_slice_cnn,
        training_plot_title="Basic CNN Training History",
    )
