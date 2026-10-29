"""Evaluate saved shallow-CNN weights and write independent test metrics."""

from basic_cnn.common import run_shared_pipeline


if __name__ == "__main__":
    run_shared_pipeline(evaluate_only=True)
