"""Train the shallow CNN on the final ConvNeXt experiment's data protocol."""

from basic_cnn.common import run_shared_pipeline


if __name__ == "__main__":
    run_shared_pipeline(evaluate_only=False)
