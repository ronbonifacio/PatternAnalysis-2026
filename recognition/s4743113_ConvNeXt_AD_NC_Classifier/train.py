# Prompt:
# "Train the layer-by-layer convnext model using adjacent ADNI slices as 2.5D
# input, with a patient-separated validation split, MRI augmentation, MixUp,
# EMA weights and validation-plateau learning-rate reductions. Save weights,
# plot epoch metrics, and report balanced and high-recall test operating points."

"""Train, validate, test and save the 2.5D ConvNeXt-Tiny classifier."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    ADNISliceDataset,
    FULL_FOREGROUND_CROP_SIZE,
    MRITrainAugmentation,
    create_adni_slice_dataloaders,
    resize_tensor_image,
    set_seed,
)
from modules import convnext_tiny


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "ADNI" / "AD_NC"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "convnext_tiny_64px_2_5d_plateau"

DEFAULT_IMAGE_SIZE = 64
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.0001
DEFAULT_EARLY_STOPPING_START_EPOCH = 10
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_GRADIENT_CLIP = 1.0
PLATEAU_PATIENCE = 5
PLATEAU_FACTOR = 0.5
PLATEAU_MIN_LR = 1e-6


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class TestResult:
    sample_id: str
    slice_index: int
    target: int
    prediction: int
    ad_probability: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 2.5D ConvNeXt-Tiny AD versus NC classifier."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--stochastic-depth", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.4,
        help="Beta-distribution alpha for batch MixUp; zero disables MixUp.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="Exponential moving-average decay for validation and checkpointing.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="CUDA compute precision; float32 disables autocasting.",
    )
    parser.add_argument("--rotation-degrees", type=float, default=7.0)
    parser.add_argument("--translation-fraction", type=float, default=0.05)
    parser.add_argument("--scale-jitter", type=float, default=0.05)
    parser.add_argument("--intensity-jitter", type=float, default=0.10)
    parser.add_argument("--noise-standard-deviation", type=float, default=0.01)
    parser.add_argument("--gamma-jitter", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate the saved best weights without training again.",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=DEFAULT_GRADIENT_CLIP,
    )
    parser.add_argument(
        "--early-stopping-start-epoch",
        type=int,
        default=DEFAULT_EARLY_STOPPING_START_EPOCH,
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument(
        "--exclude-scan",
        action="append",
        default=["1012896"],
        help="Scan ID to omit before splitting; may be supplied more than once.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help=(
            "Also report a sensitivity-oriented operating point whose "
            "threshold is selected to reach at least this AD recall on "
            "validation data."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-seed",
        type=int,
        default=123,
        help="Seed for model initialization, batch order and slice sampling.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Defaults to CUDA when available, otherwise CPU.",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if not 0.0 <= args.stochastic_depth < 1.0:
        parser.error("--stochastic-depth must be in [0, 1)")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0, 1)")
    if args.mixup_alpha < 0:
        parser.error("--mixup-alpha cannot be negative")
    if not 0.0 < args.ema_decay < 1.0:
        parser.error("--ema-decay must be in (0, 1)")
    if args.rotation_degrees < 0:
        parser.error("--rotation-degrees cannot be negative")
    if not 0.0 <= args.translation_fraction < 1.0:
        parser.error("--translation-fraction must be in [0, 1)")
    if not 0.0 <= args.scale_jitter < 1.0:
        parser.error("--scale-jitter must be in [0, 1)")
    if not 0.0 <= args.intensity_jitter < 1.0:
        parser.error("--intensity-jitter must be in [0, 1)")
    if args.noise_standard_deviation < 0:
        parser.error("--noise-standard-deviation cannot be negative")
    if not 0.0 <= args.gamma_jitter < 1.0:
        parser.error("--gamma-jitter must be in [0, 1)")
    if args.target_recall is not None and not 0.0 < args.target_recall <= 1.0:
        parser.error("--target-recall must be in (0, 1]")
    if args.warmup_epochs < 0:
        parser.error("--warmup-epochs cannot be negative")
    if args.gradient_clip <= 0:
        parser.error("--gradient-clip must be positive")
    if args.early_stopping_start_epoch < 0:
        parser.error("--early-stopping-start-epoch cannot be negative")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


class ModelEMA:
    """Maintain an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between zero and one")
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the latest trained parameters into the averaged model."""

        model_parameters = dict(model.named_parameters())
        for name, averaged_parameter in self.module.named_parameters():
            current_parameter = model_parameters[name].detach()
            averaged_parameter.mul_(self.decay).add_(
                current_parameter,
                alpha=1.0 - self.decay,
            )

        model_buffers = dict(model.named_buffers())
        for name, averaged_buffer in self.module.named_buffers():
            averaged_buffer.copy_(model_buffers[name])


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip: float | None = None,
    autocast_dtype: torch.dtype | None = torch.float16,
    mixup_alpha: float = 0.0,
    model_ema: ModelEMA | None = None,
) -> EpochMetrics:
    """Run one training or validation epoch and return slice-level metrics."""

    is_training = optimizer is not None
    model.train(is_training)
    is_cuda = device.type == "cuda"
    use_amp = is_cuda and autocast_dtype is not None

    total_loss = 0.0
    total_correct = 0.0
    total_slices = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=is_cuda)
        labels = batch["label"].to(device, non_blocking=is_cuda)
        paired_labels = None
        mixup_weight = 1.0
        if is_training and mixup_alpha > 0.0 and labels.size(0) > 1:
            mixup_weight = torch.distributions.Beta(
                mixup_alpha,
                mixup_alpha,
            ).sample().item()
            permutation = torch.randperm(labels.size(0), device=device)
            paired_labels = labels[permutation]
            images = (
                mixup_weight * images
                + (1.0 - mixup_weight) * images[permutation]
            )

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype or torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)
                if paired_labels is not None:
                    paired_loss = criterion(logits, paired_labels)
                    loss = (
                        mixup_weight * loss
                        + (1.0 - mixup_weight) * paired_loss
                    )

            if not torch.isfinite(loss).item():
                phase = "training" if is_training else "validation"
                raise FloatingPointError(
                    f"Non-finite {phase} loss at batch {batch_index}; "
                    "stopping before model weights can be corrupted."
                )

            if is_training:
                if scaler is None:
                    loss.backward()
                    if gradient_clip is not None:
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            gradient_clip,
                            error_if_nonfinite=True,
                        )
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    if gradient_clip is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            gradient_clip,
                            error_if_nonfinite=False,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                if model_ema is not None:
                    model_ema.update(model)

        slice_count = labels.size(0)
        total_loss += loss.detach().item() * slice_count
        predictions = logits.argmax(dim=1)
        batch_correct = (predictions == labels).sum().item()
        if paired_labels is not None:
            paired_correct = (predictions == paired_labels).sum().item()
            batch_correct = (
                mixup_weight * batch_correct
                + (1.0 - mixup_weight) * paired_correct
            )
        total_correct += batch_correct
        total_slices += slice_count

    if total_slices == 0:
        raise RuntimeError("The data loader produced no slices")

    return EpochMetrics(
        loss=total_loss / total_slices,
        accuracy=total_correct / total_slices,
    )


def create_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm up linearly to the optimizer's base learning rate."""

    warmup_epochs = min(warmup_epochs, total_epochs)

    def learning_rate_multiplier(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_multiplier,
    )


def infer_test_set(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ad_index: int,
    autocast_dtype: torch.dtype | None = torch.float16,
) -> tuple[float, list[TestResult]]:
    """Run slice-level test inference and retain per-slice results."""

    model.eval()
    is_cuda = device.type == "cuda"
    use_amp = is_cuda and autocast_dtype is not None
    total_loss = 0.0
    total_slices = 0
    results: list[TestResult] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=is_cuda)
            labels = batch["label"].to(device, non_blocking=is_cuda)

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype or torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite test loss at batch {batch_index}; "
                    "the selected checkpoint is invalid."
                )

            probabilities = logits.softmax(dim=1).float().cpu()
            predictions = probabilities.argmax(dim=1)
            slice_count = labels.size(0)
            total_loss += loss.item() * slice_count
            total_slices += slice_count

            for sample_id, slice_index, target, prediction, probability in zip(
                batch["sample_id"],
                batch["slice_index"],
                batch["label"],
                predictions,
                probabilities,
            ):
                results.append(
                    TestResult(
                        sample_id=str(sample_id),
                        slice_index=int(slice_index.item()),
                        target=int(target.item()),
                        prediction=int(prediction.item()),
                        ad_probability=float(probability[ad_index].item()),
                    )
                )

    if total_slices == 0:
        raise RuntimeError("The test loader produced no slices")

    return total_loss / total_slices, results


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def apply_decision_threshold(
    results: list[TestResult],
    *,
    threshold: float,
    ad_index: int,
    nc_index: int,
) -> list[TestResult]:
    """Convert AD probabilities into predictions at a fixed threshold."""

    return [
        replace(
            result,
            prediction=(
                ad_index if result.ad_probability >= threshold else nc_index
            ),
        )
        for result in results
    ]


def calculate_roc_auc(results: list[TestResult], *, ad_index: int) -> float:
    """Calculate binary ROC-AUC with average ranks for tied probabilities."""

    ordered = sorted(results, key=lambda result: result.ad_probability)
    positive_count = sum(result.target == ad_index for result in ordered)
    negative_count = len(ordered) - positive_count
    if not positive_count or not negative_count:
        return 0.0

    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        probability = ordered[start].ad_probability
        while (
            end < len(ordered)
            and ordered[end].ad_probability == probability
        ):
            end += 1
        average_rank = ((start + 1) + end) / 2
        positive_rank_sum += average_rank * sum(
            result.target == ad_index for result in ordered[start:end]
        )
        start = end

    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def calculate_classification_metrics(
    loss: float,
    results: list[TestResult],
    *,
    ad_index: int,
    nc_index: int,
    decision_threshold: float,
) -> dict[str, Any]:
    true_positives = sum(
        result.prediction == ad_index and result.target == ad_index
        for result in results
    )
    false_positives = sum(
        result.prediction == ad_index and result.target == nc_index
        for result in results
    )
    false_negatives = sum(
        result.prediction == nc_index and result.target == ad_index
        for result in results
    )
    true_negatives = sum(
        result.prediction == nc_index and result.target == nc_index
        for result in results
    )
    accuracy = safe_divide(true_positives + true_negatives, len(results))
    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    specificity = safe_divide(true_negatives, true_negatives + false_positives)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "loss": loss,
        "accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": calculate_roc_auc(results, ad_index=ad_index),
        "decision_threshold": decision_threshold,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
    }


def select_validation_threshold(
    results: list[TestResult],
    *,
    ad_index: int,
) -> float:
    """Select the threshold with best validation balanced accuracy."""

    positive_count = sum(result.target == ad_index for result in results)
    negative_count = len(results) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("Threshold calibration requires both AD and NC samples")

    ordered = sorted(
        results,
        key=lambda result: result.ad_probability,
        reverse=True,
    )
    true_positives = 0
    false_positives = 0
    best_score = (-1.0, -1.0, float("-inf"))
    best_threshold = 0.5
    index = 0
    while index < len(ordered):
        threshold = ordered[index].ad_probability
        end = index
        while (
            end < len(ordered)
            and ordered[end].ad_probability == threshold
        ):
            if ordered[end].target == ad_index:
                true_positives += 1
            else:
                false_positives += 1
            end += 1

        recall = true_positives / positive_count
        specificity = (negative_count - false_positives) / negative_count
        balanced_accuracy = (recall + specificity) / 2
        accuracy = (
            true_positives + negative_count - false_positives
        ) / len(ordered)
        score = (balanced_accuracy, accuracy, -abs(threshold - 0.5))
        if score > best_score:
            best_score = score
            best_threshold = threshold
        index = end

    return best_threshold


def select_validation_threshold_for_recall(
    results: list[TestResult],
    *,
    ad_index: int,
    target_recall: float,
) -> float:
    """Select the most specific validation threshold meeting AD recall.

    Probabilities are processed from highest to lowest. The first threshold
    that includes at least the requested proportion of validation AD slices
    is therefore the highest threshold satisfying the recall constraint and
    preserves as much NC specificity as possible.
    """

    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    positive_count = sum(result.target == ad_index for result in results)
    if not positive_count:
        raise ValueError("Recall threshold selection requires AD samples")

    ordered = sorted(
        results,
        key=lambda result: result.ad_probability,
        reverse=True,
    )
    true_positives = 0
    index = 0
    while index < len(ordered):
        threshold = ordered[index].ad_probability
        end = index
        while (
            end < len(ordered)
            and ordered[end].ad_probability == threshold
        ):
            if ordered[end].target == ad_index:
                true_positives += 1
            end += 1
        if true_positives / positive_count >= target_recall:
            return threshold
        index = end

    raise RuntimeError("Could not select a validation recall threshold")


def report_validation_results(
    validation_loss: float,
    results: list[TestResult],
    *,
    ad_index: int,
    nc_index: int,
    decision_threshold: float,
    operating_point_name: str = "balanced-accuracy operating point",
) -> dict[str, Any]:
    results = apply_decision_threshold(
        results,
        threshold=decision_threshold,
        ad_index=ad_index,
        nc_index=nc_index,
    )
    metrics = calculate_classification_metrics(
        validation_loss,
        results,
        ad_index=ad_index,
        nc_index=nc_index,
        decision_threshold=decision_threshold,
    )
    print(
        f"\nSlice-level validation results - {operating_point_name} "
        "(AD is the positive class)"
    )
    print(f"Decision threshold: {decision_threshold:.4f}")
    print(f"Accuracy:           {metrics['accuracy']:.2%}")
    print(f"Balanced accuracy:  {metrics['balanced_accuracy']:.2%}")
    print(f"Precision:          {metrics['precision']:.2%}")
    print(f"Recall:             {metrics['recall']:.2%}")
    print(f"Specificity:        {metrics['specificity']:.2%}")
    print(f"F1 score:           {metrics['f1']:.2%}")
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    return metrics


def print_examples(
    title: str,
    examples: list[TestResult],
    idx_to_class: dict[int, str],
    limit: int = 5,
) -> None:
    print(f"\n{title} ({len(examples)} total):")
    if not examples:
        print("  None")
        return

    for result in examples[:limit]:
        print(
            f"  {result.sample_id}_{result.slice_index}: "
            f"actual={idx_to_class[result.target]}, "
            f"predicted={idx_to_class[result.prediction]}, "
            f"P(AD)={result.ad_probability:.3f}"
        )


def save_slice_predictions(
    results: list[TestResult],
    output_dir: Path,
    *,
    ad_index: int,
    filename: str,
) -> Path:
    """Write per-slice AD probabilities so ROC curves can be plotted later."""

    path = output_dir / filename
    with path.open("w", encoding="utf-8") as predictions_file:
        json.dump(
            {
                "ad_index": ad_index,
                "results": [
                    {
                        "sample_id": result.sample_id,
                        "slice_index": result.slice_index,
                        "target": result.target,
                        "ad_probability": result.ad_probability,
                    }
                    for result in results
                ],
            },
            predictions_file,
        )
    return path


def report_test_results(
    test_loss: float,
    results: list[TestResult],
    *,
    ad_index: int,
    nc_index: int,
    idx_to_class: dict[int, str],
    output_dir: Path,
    decision_threshold: float = 0.5,
    operating_point_name: str = "balanced-accuracy operating point",
    confusion_matrix_filename: str = "test_confusion_matrix.png",
) -> dict[str, Any]:
    """Print test metrics and optionally save a confusion-matrix figure."""

    results = apply_decision_threshold(
        results,
        threshold=decision_threshold,
        ad_index=ad_index,
        nc_index=nc_index,
    )
    metrics = calculate_classification_metrics(
        test_loss,
        results,
        ad_index=ad_index,
        nc_index=nc_index,
        decision_threshold=decision_threshold,
    )
    true_positives = metrics["true_positives"]
    false_positives = metrics["false_positives"]
    false_negatives = metrics["false_negatives"]
    true_negatives = metrics["true_negatives"]

    correct = [result for result in results if result.prediction == result.target]
    false_positive_examples = [
        result
        for result in results
        if result.prediction == ad_index and result.target == nc_index
    ]
    false_negative_examples = [
        result
        for result in results
        if result.prediction == nc_index and result.target == ad_index
    ]

    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]

    print(
        f"\nSlice-level test results - {operating_point_name} "
        "(AD is the positive class)"
    )
    print(f"Decision threshold: {decision_threshold:.4f}")
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {accuracy:.2%} ({len(correct)}/{len(results)})")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"Specificity: {metrics['specificity']:.2%}")
    print(f"F1 score:  {f1:.2%}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print(f"\nSlice-level test confusion matrix - {operating_point_name}")
    print("                    Predicted NC  Predicted AD")
    print(f"Actual NC (negative) {true_negatives:12d} {false_positives:13d}")
    print(f"Actual AD (positive) {false_negatives:12d} {true_positives:13d}")

    print_examples("Correct examples", correct, idx_to_class)
    print_examples(
        "False positives (NC predicted as AD)",
        false_positive_examples,
        idx_to_class,
    )
    print_examples(
        "False negatives (AD predicted as NC)",
        false_negative_examples,
        idx_to_class,
    )

    confusion_matrix = [
        [true_negatives, false_positives],
        [false_negatives, true_positives],
    ]

    # Plotting is optional: test inference, terminal metrics, and metrics.json
    # must still complete on the HPC environment, which has no Matplotlib.
    matplotlib_config_dir = output_dir / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        if error.name is None or not error.name.startswith("matplotlib"):
            raise
        print("Matplotlib is unavailable; skipped the confusion-matrix image.")
        return metrics

    cell_labels = [
        [f"TN\n{true_negatives}", f"FP\n{false_positives}"],
        [f"FN\n{false_negatives}", f"TP\n{true_positives}"],
    ]

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    heatmap = axis.imshow(confusion_matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Predicted NC", "Predicted AD"])
    axis.set_yticks([0, 1], labels=["Actual NC", "Actual AD"])
    axis.set_title(f"AD vs NC Test Confusion Matrix\n{operating_point_name}")

    threshold = max(max(row) for row in confusion_matrix) / 2
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                cell_labels[row][column],
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=(
                    "white"
                    if confusion_matrix[row][column] > threshold
                    else "black"
                ),
            )

    figure.colorbar(heatmap, ax=axis, label="Slices")
    figure.text(
        0.5,
        0.01,
        f"Accuracy: {accuracy:.2%}    Precision: {precision:.2%}    "
        f"Recall: {recall:.2%}",
        ha="center",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    confusion_matrix_path = output_dir / confusion_matrix_filename
    figure.savefig(confusion_matrix_path, dpi=160)
    plt.close(figure)
    print(f"Confusion matrix saved to: {confusion_matrix_path}")

    return metrics


def report_target_recall_operating_point(
    *,
    target_recall: float,
    validation_loss: float,
    validation_results: list[TestResult],
    test_loss: float,
    test_results: list[TestResult],
    ad_index: int,
    nc_index: int,
    idx_to_class: dict[int, str],
    output_dir: Path,
) -> dict[str, Any]:
    """Report a validation-selected high-sensitivity operating point."""

    decision_threshold = select_validation_threshold_for_recall(
        validation_results,
        ad_index=ad_index,
        target_recall=target_recall,
    )
    operating_point_name = (
        f"{target_recall:.0%} validation-recall operating point"
    )
    validation_metrics = report_validation_results(
        validation_loss,
        validation_results,
        ad_index=ad_index,
        nc_index=nc_index,
        decision_threshold=decision_threshold,
        operating_point_name=operating_point_name,
    )
    test_metrics = report_test_results(
        test_loss,
        test_results,
        ad_index=ad_index,
        nc_index=nc_index,
        idx_to_class=idx_to_class,
        output_dir=output_dir,
        decision_threshold=decision_threshold,
        operating_point_name=operating_point_name,
        confusion_matrix_filename=(
            f"test_confusion_matrix_{round(target_recall * 100)}_recall.png"
        ),
    )
    return {
        "target_validation_recall": target_recall,
        "decision_threshold": decision_threshold,
        "validation": validation_metrics,
        "test": test_metrics,
    }


def save_training_curves(
    history: list[dict[str, Any]],
    output_dir: Path,
    *,
    plot_title: str = "ConvNeXt Training History",
) -> Path:
    """Save train/validation loss and accuracy as a dependency-free SVG."""

    if not history:
        raise ValueError("Cannot plot an empty training history")

    width = 1000
    height = 480
    panel_width = 400
    panel_height = 320
    panel_top = 75
    panel_lefts = (70, 570)
    train_colour = "#d95f02"
    validation_colour = "#1b75bc"
    epochs = [int(record["epoch"]) for record in history]

    def x_position(index: int, left: int) -> float:
        if len(history) == 1:
            return left + panel_width / 2
        return left + panel_width * index / (len(history) - 1)

    def series_points(
        values: list[float],
        *,
        left: int,
        minimum: float,
        maximum: float,
    ) -> str:
        value_range = maximum - minimum
        return " ".join(
            f"{x_position(index, left):.1f},"
            f"{panel_top + panel_height * (maximum - value) / value_range:.1f}"
            for index, value in enumerate(values)
        )

    def panel(
        *,
        left: int,
        title: str,
        train_values: list[float],
        validation_values: list[float],
        fixed_bounds: tuple[float, float] | None = None,
    ) -> list[str]:
        if fixed_bounds is None:
            minimum = min(*train_values, *validation_values)
            maximum = max(*train_values, *validation_values)
            padding = max((maximum - minimum) * 0.08, 1e-4)
            minimum -= padding
            maximum += padding
        else:
            minimum, maximum = fixed_bounds

        elements = [
            f'<text x="{left + panel_width / 2:.1f}" y="48" '
            f'text-anchor="middle" font-size="22">{title}</text>',
            f'<rect x="{left}" y="{panel_top}" width="{panel_width}" '
            f'height="{panel_height}" fill="white" stroke="#333"/>',
        ]
        for grid_index in range(5):
            fraction = grid_index / 4
            y = panel_top + panel_height * fraction
            value = maximum - (maximum - minimum) * fraction
            elements.extend(
                [
                    f'<line x1="{left}" y1="{y:.1f}" '
                    f'x2="{left + panel_width}" y2="{y:.1f}" '
                    'stroke="#dddddd"/>',
                    f'<text x="{left - 10}" y="{y + 5:.1f}" '
                    f'text-anchor="end" font-size="13">{value:.3f}</text>',
                ]
            )
        elements.extend(
            [
                f'<polyline points="{series_points(train_values, left=left, minimum=minimum, maximum=maximum)}" '
                f'fill="none" stroke="{train_colour}" stroke-width="2.5"/>',
                f'<polyline points="{series_points(validation_values, left=left, minimum=minimum, maximum=maximum)}" '
                f'fill="none" stroke="{validation_colour}" stroke-width="2.5"/>',
                f'<text x="{left}" y="{panel_top + panel_height + 28}" '
                f'font-size="13">Epoch {epochs[0]}</text>',
                f'<text x="{left + panel_width}" '
                f'y="{panel_top + panel_height + 28}" text-anchor="end" '
                f'font-size="13">Epoch {epochs[-1]}</text>',
            ]
        )
        return elements

    train_losses = [float(record["train"]["loss"]) for record in history]
    validation_losses = [
        float(record["validation"]["loss"]) for record in history
    ]
    train_accuracies = [
        float(record["train"]["accuracy"]) for record in history
    ]
    validation_accuracies = [
        float(record["validation"]["accuracy"]) for record in history
    ]

    svg_elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="500" y="25" text-anchor="middle" font-size="24" '
        f'font-weight="bold">{plot_title}</text>',
    ]
    svg_elements.extend(
        panel(
            left=panel_lefts[0],
            title="Cross-entropy loss",
            train_values=train_losses,
            validation_values=validation_losses,
        )
    )
    svg_elements.extend(
        panel(
            left=panel_lefts[1],
            title="Accuracy",
            train_values=train_accuracies,
            validation_values=validation_accuracies,
            fixed_bounds=(0.0, 1.0),
        )
    )
    svg_elements.extend(
        [
            f'<line x1="375" y1="445" x2="410" y2="445" '
            f'stroke="{train_colour}" stroke-width="3"/>',
            '<text x="418" y="450" font-size="14">Training</text>',
            f'<line x1="520" y1="445" x2="555" y2="445" '
            f'stroke="{validation_colour}" stroke-width="3"/>',
            '<text x="563" y="450" font-size="14">Validation</text>',
            "</svg>",
        ]
    )
    output_path = output_dir / "training_curves.svg"
    output_path.write_text("\n".join(svg_elements), encoding="utf-8")
    return output_path


def main(
    *,
    model_factory: Callable[..., nn.Module] = convnext_tiny,
    training_plot_title: str = "ConvNeXt Training History",
) -> None:
    args = parse_args()
    training_seed = args.training_seed
    set_seed(training_seed)
    image_size = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    evaluation_transform = partial(
        resize_tensor_image,
        size=DEFAULT_IMAGE_SIZE,
    )
    training_transform = MRITrainAugmentation(
        size=DEFAULT_IMAGE_SIZE,
        rotation_degrees=args.rotation_degrees,
        translation_fraction=args.translation_fraction,
        scale_jitter=args.scale_jitter,
        intensity_jitter=args.intensity_jitter,
        noise_standard_deviation=args.noise_standard_deviation,
        gamma_jitter=args.gamma_jitter,
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if (
        device.type == "cuda"
        and args.amp_dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("bfloat16 was requested but this GPU does not support it")

    autocast_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": None,
    }[args.amp_dtype]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = args.output_dir / "best_model_weights.pt"
    last_weights_path = args.output_dir / "last_model_weights.pt"
    last_ema_weights_path = args.output_dir / "last_ema_model_weights.pt"
    metrics_path = args.output_dir / "metrics.json"
    # Fail here rather than after the loaders and model have been built.
    if args.evaluate_only and not best_weights_path.is_file():
        raise FileNotFoundError(
            f"Cannot evaluate because the best weights do not exist: "
            f"{best_weights_path}"
        )

    train_loader, validation_loader, test_loader = create_adni_slice_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        training_seed=training_seed,
        metadata_path=args.metadata_path,
        excluded_sample_ids=args.exclude_scan,
        image_size=FULL_FOREGROUND_CROP_SIZE,
        train_transform=training_transform,
        evaluation_transform=evaluation_transform,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    training_dataset = train_loader.dataset
    if not isinstance(training_dataset, ADNISliceDataset):
        raise TypeError("Expected an ADNISliceDataset")
    class_to_idx = dict(training_dataset.class_to_idx)
    if "AD" not in class_to_idx or "NC" not in class_to_idx:
        raise ValueError("The dataset must contain AD and NC classes")
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    ad_index = class_to_idx["AD"]
    nc_index = class_to_idx["NC"]

    # Dataset construction uses the split seed. Restore the independent
    # training seed before model-weight initialization.
    set_seed(training_seed)
    model = model_factory(
        num_classes=len(class_to_idx),
        stochastic_depth_probability=args.stochastic_depth,
    ).to(device)
    training_criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )
    evaluation_criterion = nn.CrossEntropyLoss()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Device: {device}")
    print(f"Classes: {class_to_idx}")
    if training_dataset.excluded_sample_ids:
        print(
            "Excluded training scans: "
            + ", ".join(training_dataset.excluded_sample_ids)
        )
    print(f"Training scans: {training_dataset.subject_count}")
    print(f"Validation scans: {validation_loader.dataset.subject_count}")
    print(f"Test scans: {test_loader.dataset.subject_count}")
    print(f"Training patients: {training_dataset.patient_count}")
    print(f"Validation patients: {validation_loader.dataset.patient_count}")
    print(f"Test patients: {test_loader.dataset.patient_count}")
    print(f"Training slices available: {len(training_dataset)}")
    print(f"Training slices selected per epoch: {len(train_loader.sampler)}")
    print(f"Validation slices: {len(validation_loader.dataset)}")
    print(f"Test slices: {len(test_loader.dataset)}")
    print(f"Model input size: {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE}")
    print("Input channels: previous, centre and next slice (2.5D)")
    print("Input preprocessing: resize complete 224x224 foreground crop")
    print("Input normalization: per-slice foreground mean and standard deviation")
    print("Validation split level: patient; patient-balanced sampling: enabled")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Split seed: {args.seed}; training seed: {training_seed}")
    print(f"CUDA compute precision: {args.amp_dtype}")
    print(
        f"Training augmentation: rotation +/-{args.rotation_degrees:g} degrees; "
        f"translation {args.translation_fraction:.0%}; scale +/-"
        f"{args.scale_jitter:.0%}; intensity +/-{args.intensity_jitter:.0%}; "
        f"noise standard deviation {args.noise_standard_deviation:g}; "
        f"gamma +/-{args.gamma_jitter:.0%}"
    )
    print(
        f"Label smoothing: {args.label_smoothing:g}; "
        "final test checkpoint: best validation loss"
    )
    print("AD decision threshold: selected on validation balanced accuracy")
    print(
        f"Secondary AD operating point: at least "
        f"{args.target_recall:.0%} recall on validation data"
    )

    def evaluate_selected_model() -> tuple[
        dict[str, float | int],
        dict[str, float | int],
        dict[str, object],
    ]:
        """Evaluate both validation-selected operating points on held-out data."""

        validation_loss, validation_results = infer_test_set(
            model,
            validation_loader,
            evaluation_criterion,
            device,
            ad_index,
            autocast_dtype,
        )
        decision_threshold = select_validation_threshold(
            validation_results,
            ad_index=ad_index,
        )
        validation_report = report_validation_results(
            validation_loss,
            validation_results,
            ad_index=ad_index,
            nc_index=nc_index,
            decision_threshold=decision_threshold,
            operating_point_name="balanced-accuracy operating point",
        )
        test_loss, test_results = infer_test_set(
            model,
            test_loader,
            evaluation_criterion,
            device,
            ad_index,
            autocast_dtype,
        )
        test_metrics = report_test_results(
            test_loss,
            test_results,
            ad_index=ad_index,
            nc_index=nc_index,
            idx_to_class=idx_to_class,
            output_dir=args.output_dir,
            decision_threshold=decision_threshold,
            operating_point_name="balanced-accuracy operating point",
        )
        for results, filename in (
            (validation_results, "validation_slice_predictions.json"),
            (test_results, "test_slice_predictions.json"),
        ):
            path = save_slice_predictions(
                results,
                args.output_dir,
                ad_index=ad_index,
                filename=filename,
            )
            print(f"Slice predictions saved to: {path}")

        target_recall_report = report_target_recall_operating_point(
            target_recall=args.target_recall,
            validation_loss=validation_loss,
            validation_results=validation_results,
            test_loss=test_loss,
            test_results=test_results,
            ad_index=ad_index,
            nc_index=nc_index,
            idx_to_class=idx_to_class,
            output_dir=args.output_dir,
        )
        return validation_report, test_metrics, target_recall_report

    # Re-score the saved checkpoint and return before any optimizer, scheduler
    # or EMA state exists. Results go to test_metrics.json so that re-scoring
    # cannot overwrite the metrics.json written by the run that trained them.
    if args.evaluate_only:
        print(f"Evaluation only; loading weights from: {best_weights_path}")
        model.load_state_dict(
            torch.load(best_weights_path, map_location=device, weights_only=True)
        )
        validation_report, test_metrics, target_recall_report = (
            evaluate_selected_model()
        )
        test_metrics_path = args.output_dir / "test_metrics.json"
        with test_metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(
                {
                    "weights": str(best_weights_path),
                    "batch_size": args.batch_size,
                    "image_size": list(image_size),
                    "input_mode": "resize",
                    "slice_context": 3,
                    "label_smoothing": args.label_smoothing,
                    "mixup_alpha": args.mixup_alpha,
                    "ema_decay": args.ema_decay,
                    "test_checkpoint": "best validation loss",
                    "amp_dtype": args.amp_dtype,
                    "augment_training": True,
                    "split_level": "patient",
                    "normalization": "per-slice",
                    "balance_patients": True,
                    "excluded_scans": args.exclude_scan,
                    "target_recall": args.target_recall,
                    "seed": args.seed,
                    "training_seed": training_seed,
                    "validation": validation_report,
                    "test": test_metrics,
                    "target_recall_operating_point": target_recall_report,
                },
                metrics_file,
                indent=2,
            )
        print(f"Test metrics saved to: {test_metrics_path}")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model_ema = ModelEMA(model, args.ema_decay)
    warmup_scheduler = create_warmup_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=PLATEAU_FACTOR,
        patience=PLATEAU_PATIENCE,
        threshold=EARLY_STOPPING_MIN_DELTA,
        threshold_mode="abs",
        min_lr=PLATEAU_MIN_LR,
    )
    scaler = None
    if device.type == "cuda" and autocast_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")

    print(
        f"Base learning rate: {args.learning_rate:g}; "
        f"warm-up epochs: {args.warmup_epochs}; "
        f"weight decay: {args.weight_decay:g}"
    )
    print(
        f"Plateau scheduler: factor {PLATEAU_FACTOR:g}, patience "
        f"{PLATEAU_PATIENCE}, minimum learning rate {PLATEAU_MIN_LR:g}"
    )
    print(
        f"Stochastic depth: {args.stochastic_depth:g}; "
        f"gradient clipping: {args.gradient_clip:g}"
    )
    print(
        f"MixUp alpha: {args.mixup_alpha:g}; "
        + f"EMA decay: {args.ema_decay:g}"
    )
    print("Validation and best checkpoints use EMA weights")
    print(
        f"Early stopping: start after epoch "
        f"{args.early_stopping_start_epoch}, patience "
        f"{EARLY_STOPPING_PATIENCE}, minimum delta "
        f"{EARLY_STOPPING_MIN_DELTA:g}"
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model,
            train_loader,
            training_criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=args.gradient_clip,
            autocast_dtype=autocast_dtype,
            mixup_alpha=args.mixup_alpha,
            model_ema=model_ema,
        )
        validation_model = model_ema.module
        validation_metrics = run_epoch(
            validation_model,
            validation_loader,
            evaluation_criterion,
            device,
            autocast_dtype=autocast_dtype,
        )

        history.append(
            {
                "epoch": epoch,
                "learning_rate": epoch_learning_rate,
                "train": asdict(train_metrics),
                "validation": asdict(validation_metrics),
            }
        )
        # Refresh the plot every epoch so partial jobs still retain curves.
        training_curves_path = save_training_curves(
            history,
            args.output_dir,
            plot_title=training_plot_title,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_metrics.loss:.4f}, "
            f"accuracy {train_metrics.accuracy:.2%} | "
            f"validation loss {validation_metrics.loss:.4f}, "
            f"accuracy {validation_metrics.accuracy:.2%} | "
            f"lr {epoch_learning_rate:.2e}"
        )

        improvement = best_validation_loss - validation_metrics.loss
        if best_epoch == 0 or improvement >= EARLY_STOPPING_MIN_DELTA:
            best_validation_loss = validation_metrics.loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(validation_model.state_dict(), best_weights_path)
            print(
                f"  Validation loss improved; saved weights to "
                f"{best_weights_path}"
            )
        else:
            if epoch <= args.early_stopping_start_epoch:
                epochs_without_improvement = 0
                print(
                    "  No minimum-delta improvement; early-stopping "
                    f"counting begins after epoch "
                    f"{args.early_stopping_start_epoch}."
                )
            else:
                epochs_without_improvement += 1
                print(
                    f"  No validation-loss improvement of at least "
                    f"{EARLY_STOPPING_MIN_DELTA:g} "
                    f"({epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})"
                )

        learning_rate_before_step = optimizer.param_groups[0]["lr"]
        if epoch < min(args.warmup_epochs, args.epochs):
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(validation_metrics.loss)
        learning_rate_after_step = optimizer.param_groups[0]["lr"]
        if learning_rate_after_step < learning_rate_before_step:
            print(
                f"  Plateau scheduler reduced learning rate from "
                f"{learning_rate_before_step:.2e} to "
                f"{learning_rate_after_step:.2e}."
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping after epoch {epoch}: validation loss did not "
                f"improve by at least {EARLY_STOPPING_MIN_DELTA:g} for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

    print(f"Training curves saved to: {training_curves_path}")
    torch.save(model.state_dict(), last_weights_path)
    torch.save(model_ema.module.state_dict(), last_ema_weights_path)
    print(f"Best epoch: {best_epoch} (validation loss {best_validation_loss:.4f})")
    print(f"Best weights saved to: {best_weights_path}")
    print(f"Final weights saved to: {last_weights_path}")
    print(f"Final EMA weights saved to: {last_ema_weights_path}")

    print(f"Testing best validation-loss weights: {best_weights_path}")
    model.load_state_dict(
        torch.load(best_weights_path, map_location=device, weights_only=True)
    )
    validation_report, test_metrics, target_recall_report = (
        evaluate_selected_model()
    )

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "configuration": {
                    "epochs_requested": args.epochs,
                    "batch_size": args.batch_size,
                    "image_size": list(image_size),
                    "input_mode": "resize",
                    "slice_context": 3,
                    "split_level": "patient",
                    "normalization": "per-slice",
                    "balance_patients": True,
                    "excluded_scans": args.exclude_scan,
                    "calibrate_threshold": True,
                    "decision_threshold": validation_report[
                        "decision_threshold"
                    ],
                    "target_recall": args.target_recall,
                    "learning_rate": args.learning_rate,
                    "learning_rate_scheduler": "plateau",
                    "weight_decay": args.weight_decay,
                    "stochastic_depth": args.stochastic_depth,
                    "label_smoothing": args.label_smoothing,
                    "mixup_alpha": args.mixup_alpha,
                    "ema_decay": args.ema_decay,
                    "test_checkpoint": "loss",
                    "amp_dtype": args.amp_dtype,
                    "augment_training": True,
                    "rotation_degrees": args.rotation_degrees,
                    "translation_fraction": args.translation_fraction,
                    "scale_jitter": args.scale_jitter,
                    "intensity_jitter": args.intensity_jitter,
                    "noise_standard_deviation": args.noise_standard_deviation,
                    "gamma_jitter": args.gamma_jitter,
                    "warmup_epochs": args.warmup_epochs,
                    "plateau_patience": PLATEAU_PATIENCE,
                    "plateau_factor": PLATEAU_FACTOR,
                    "plateau_min_lr": PLATEAU_MIN_LR,
                    "gradient_clip": args.gradient_clip,
                    "early_stopping_enabled": True,
                    "early_stopping_start_epoch": (
                        args.early_stopping_start_epoch
                    ),
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                    "seed": args.seed,
                    "training_seed": training_seed,
                },
                "best_epoch": best_epoch,
                "test_weights": str(best_weights_path),
                "history": history,
                "validation": validation_report,
                "test": test_metrics,
                "target_recall_operating_point": target_recall_report,
            },
            metrics_file,
            indent=2,
        )
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
