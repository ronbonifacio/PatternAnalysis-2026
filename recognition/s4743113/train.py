# Prompt:
# "Train the layer-by-layer convnext model on individual ADNI MRI slices while
# splitting training and validation at subject level. Save model weights and
# report slice-level epoch metrics, test metrics, and a confusion matrix. Use
# early stopping with patience 20 and a minimum validation-loss delta of 0.0001."

"""Train and evaluate the slice-level ConvNeXt-Tiny classifier."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNISliceDataset, create_adni_slice_dataloaders, set_seed
from modules import ConvNeXt, convnext_tiny


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "ADNI" / "AD_NC"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "convnext_tiny_slice_level"

IMAGE_SIZE = (224, 224)
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.0001
DEFAULT_EARLY_STOPPING_START_EPOCH = 10
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_GRADIENT_CLIP = 1.0
PLATEAU_PATIENCE = 5
PLATEAU_FACTOR = 0.5
PLATEAU_MIN_LR = 1e-6
COSINE_MIN_LR = 1e-6


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
        description="Train ConvNeXt-Tiny on individual ADNI MRI slices."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--stochastic-depth", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument(
        "--learning-rate-scheduler",
        choices=("plateau", "cosine"),
        default="plateau",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate the saved best weights without training again.",
    )
    parser.add_argument("--disable-early-stopping", action="store_true")
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Seed for model initialization, batch order and slice sampling.",
    )
    parser.add_argument(
        "--slices-per-scan-per-epoch",
        type=int,
        default=None,
        help="Randomly select this many training slices from each scan per epoch.",
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
    if args.warmup_epochs < 0:
        parser.error("--warmup-epochs cannot be negative")
    if args.gradient_clip <= 0:
        parser.error("--gradient-clip must be positive")
    if args.early_stopping_start_epoch < 0:
        parser.error("--early-stopping-start-epoch cannot be negative")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if (
        args.slices_per_scan_per_epoch is not None
        and args.slices_per_scan_per_epoch <= 0
    ):
        parser.error("--slices-per-scan-per-epoch must be positive")

    return args


def run_epoch(
    model: ConvNeXt,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip: float | None = None,
) -> EpochMetrics:
    """Run one training or validation epoch and return slice-level metrics."""

    is_training = optimizer is not None
    model.train(is_training)
    use_amp = device.type == "cuda"

    total_loss = 0.0
    total_correct = 0
    total_slices = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=use_amp)
        labels = batch["label"].to(device, non_blocking=use_amp)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                if scaler is None:
                    loss.backward()
                    if gradient_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    if gradient_clip is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()

        slice_count = labels.size(0)
        total_loss += loss.detach().item() * slice_count
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
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


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_epochs: int,
    warmup_epochs: int,
    minimum_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm up linearly, then anneal to a fixed minimum learning rate."""

    warmup_epochs = min(warmup_epochs, total_epochs)
    base_learning_rate = optimizer.param_groups[0]["lr"]
    minimum_multiplier = min(minimum_learning_rate / base_learning_rate, 1.0)
    cosine_epochs = max(total_epochs - warmup_epochs, 1)

    def learning_rate_multiplier(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        if total_epochs <= warmup_epochs:
            return 1.0

        progress = min(
            max(epoch_index - warmup_epochs + 1, 0) / cosine_epochs,
            1.0,
        )
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_multiplier + (1.0 - minimum_multiplier) * cosine_value

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_multiplier,
    )


def infer_test_set(
    model: ConvNeXt,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ad_index: int,
) -> tuple[float, list[TestResult]]:
    """Run slice-level test inference and retain per-slice results."""

    model.eval()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_slices = 0
    results: list[TestResult] = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=use_amp)
            labels = batch["label"].to(device, non_blocking=use_amp)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

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


def report_test_results(
    test_loss: float,
    results: list[TestResult],
    *,
    ad_index: int,
    nc_index: int,
    idx_to_class: dict[int, str],
    output_dir: Path,
) -> dict[str, Any]:
    """Print test metrics and optionally save a confusion-matrix figure."""

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

    accuracy = safe_divide(true_positives + true_negatives, len(results))
    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    metrics = {
        "loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
    }

    print("\nSlice-level test results (AD is the positive class)")
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {accuracy:.2%} ({len(correct)}/{len(results)})")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 score:  {f1:.2%}")

    print("\nSlice-level test confusion matrix")
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
    axis.set_title("AD vs NC Slice-Level Test Confusion Matrix")

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
    confusion_matrix_path = output_dir / "test_confusion_matrix.png"
    figure.savefig(confusion_matrix_path, dpi=160)
    plt.close(figure)
    print(f"Confusion matrix saved to: {confusion_matrix_path}")

    return metrics


def main() -> None:
    args = parse_args()
    training_seed = args.seed if args.training_seed is None else args.training_seed
    set_seed(training_seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = args.output_dir / "best_model_weights.pt"
    last_weights_path = args.output_dir / "last_model_weights.pt"
    metrics_path = args.output_dir / "metrics.json"
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
        training_slices_per_scan=args.slices_per_scan_per_epoch,
        image_size=IMAGE_SIZE,
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
    model = convnext_tiny(
        num_classes=len(class_to_idx),
        stochastic_depth_probability=args.stochastic_depth,
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Device: {device}")
    print(f"Classes: {class_to_idx}")
    print(f"Training scans: {training_dataset.subject_count}")
    print(f"Validation scans: {validation_loader.dataset.subject_count}")
    print(f"Test scans: {test_loader.dataset.subject_count}")
    print(f"Training slices available: {len(training_dataset)}")
    print(f"Training slices selected per epoch: {len(train_loader.sampler)}")
    print(f"Validation slices: {len(validation_loader.dataset)}")
    print(f"Test slices: {len(test_loader.dataset)}")
    print(
        f"Training foreground statistics: mean "
        f"{training_dataset.foreground_mean:.4f}, standard deviation "
        f"{training_dataset.foreground_standard_deviation:.4f}"
    )
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Split seed: {args.seed}; training seed: {training_seed}")

    if args.evaluate_only:
        print(f"Evaluation only; loading weights from: {best_weights_path}")
        model.load_state_dict(
            torch.load(best_weights_path, map_location=device, weights_only=True)
        )
        test_loss, test_results = infer_test_set(
            model,
            test_loader,
            criterion,
            device,
            ad_index,
        )
        test_metrics = report_test_results(
            test_loss,
            test_results,
            ad_index=ad_index,
            nc_index=nc_index,
            idx_to_class=idx_to_class,
            output_dir=args.output_dir,
        )
        test_metrics_path = args.output_dir / "test_metrics.json"
        with test_metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(
                {
                    "weights": str(best_weights_path),
                    "batch_size": args.batch_size,
                    "image_size": list(IMAGE_SIZE),
                    "foreground_mean": training_dataset.foreground_mean,
                    "foreground_standard_deviation": (
                        training_dataset.foreground_standard_deviation
                    ),
                    "seed": args.seed,
                    "training_seed": training_seed,
                    "slices_per_scan_per_epoch": args.slices_per_scan_per_epoch,
                    "test": test_metrics,
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
    warmup_scheduler = None
    plateau_scheduler = None
    cosine_scheduler = None
    if args.learning_rate_scheduler == "plateau":
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
    else:
        cosine_scheduler = create_warmup_cosine_scheduler(
            optimizer,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            minimum_learning_rate=COSINE_MIN_LR,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(
        f"Base learning rate: {args.learning_rate:g}; "
        f"warm-up epochs: {args.warmup_epochs}; "
        f"weight decay: {args.weight_decay:g}"
    )
    if args.learning_rate_scheduler == "plateau":
        print(
            f"Plateau scheduler: factor {PLATEAU_FACTOR:g}, patience "
            f"{PLATEAU_PATIENCE}, minimum learning rate {PLATEAU_MIN_LR:g}"
        )
    else:
        print(
            f"Cosine scheduler: anneal after warm-up to minimum learning "
            f"rate {COSINE_MIN_LR:g}"
        )
    print(
        f"Stochastic depth: {args.stochastic_depth:g}; "
        f"gradient clipping: {args.gradient_clip:g}"
    )
    if args.disable_early_stopping:
        print(f"Early stopping: disabled; training all {args.epochs} epochs")
    else:
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
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=args.gradient_clip,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "learning_rate": epoch_learning_rate,
                "train": asdict(train_metrics),
                "validation": asdict(validation_metrics),
            }
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
            torch.save(model.state_dict(), best_weights_path)
            print(
                f"  Validation loss improved; saved weights to "
                f"{best_weights_path}"
            )
        else:
            if args.disable_early_stopping:
                epochs_without_improvement = 0
                print(
                    "  No minimum-delta improvement; fixed-epoch training "
                    "continues."
                )
            elif epoch <= args.early_stopping_start_epoch:
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
        if args.learning_rate_scheduler == "cosine":
            if cosine_scheduler is None:
                raise RuntimeError("Cosine scheduler was not initialized")
            cosine_scheduler.step()
        else:
            if warmup_scheduler is None or plateau_scheduler is None:
                raise RuntimeError("Plateau schedulers were not initialized")
            if epoch < min(args.warmup_epochs, args.epochs):
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(validation_metrics.loss)
        learning_rate_after_step = optimizer.param_groups[0]["lr"]
        if learning_rate_after_step < learning_rate_before_step:
            if args.learning_rate_scheduler == "plateau":
                print(
                    f"  Plateau scheduler reduced learning rate from "
                    f"{learning_rate_before_step:.2e} to "
                    f"{learning_rate_after_step:.2e}."
                )

        if (
            not args.disable_early_stopping
            and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"Early stopping after epoch {epoch}: validation loss did not "
                f"improve by at least {EARLY_STOPPING_MIN_DELTA:g} for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

    torch.save(model.state_dict(), last_weights_path)
    print(f"Best epoch: {best_epoch} (validation loss {best_validation_loss:.4f})")
    print(f"Best weights saved to: {best_weights_path}")
    print(f"Final weights saved to: {last_weights_path}")

    model.load_state_dict(
        torch.load(best_weights_path, map_location=device, weights_only=True)
    )
    test_loss, test_results = infer_test_set(
        model,
        test_loader,
        criterion,
        device,
        ad_index,
    )
    test_metrics = report_test_results(
        test_loss,
        test_results,
        ad_index=ad_index,
        nc_index=nc_index,
        idx_to_class=idx_to_class,
        output_dir=args.output_dir,
    )

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                "configuration": {
                    "epochs_requested": args.epochs,
                    "batch_size": args.batch_size,
                    "image_size": list(IMAGE_SIZE),
                    "foreground_mean": training_dataset.foreground_mean,
                    "foreground_standard_deviation": (
                        training_dataset.foreground_standard_deviation
                    ),
                    "learning_rate": args.learning_rate,
                    "learning_rate_scheduler": args.learning_rate_scheduler,
                    "weight_decay": args.weight_decay,
                    "stochastic_depth": args.stochastic_depth,
                    "warmup_epochs": args.warmup_epochs,
                    "plateau_patience": PLATEAU_PATIENCE,
                    "plateau_factor": PLATEAU_FACTOR,
                    "plateau_min_lr": PLATEAU_MIN_LR,
                    "cosine_min_lr": COSINE_MIN_LR,
                    "gradient_clip": args.gradient_clip,
                    "early_stopping_enabled": (
                        not args.disable_early_stopping
                    ),
                    "early_stopping_start_epoch": (
                        args.early_stopping_start_epoch
                    ),
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                    "seed": args.seed,
                    "training_seed": training_seed,
                    "slices_per_scan_per_epoch": args.slices_per_scan_per_epoch,
                },
                "best_epoch": best_epoch,
                "history": history,
                "test": test_metrics,
            },
            metrics_file,
            indent=2,
        )
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
