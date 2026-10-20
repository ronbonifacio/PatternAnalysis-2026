# Prompt:
# "Create a train.py which implements our modules.py to train our convnext
# model. Save model weights to output. Ensure epochs and metrics are reported
# after each epoch similarly to the basic CNN. Implement early stopping with a
# patience of 5 epochs and a minimum delta of 0.001 on the validation loss. At
# the end of training perform inference on the test set and report metrics and
# confusion matrix similarly to within the basic CNN notebook. Add this prompt
# as a comment to the top of the script. Document our chosen implementation of
# early stopping in the next section of the README.md."

"""Train and evaluate the subject-level ConvNeXt-Tiny classifier."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dataset import create_adni_dataloaders, set_seed
from modules import SubjectConvNeXt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "ADNI" / "AD_NC"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "convnext_tiny"

IMAGE_SIZE = (224, 224)
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class TestResult:
    sample_id: str
    target: int
    prediction: int
    ad_probability: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt-Tiny on grouped ADNI MRI slices."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-slices", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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
    if args.max_slices <= 0:
        parser.error("--max-slices must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    return args


def pack_evenly_spaced_slices(
    images: Tensor,
    slice_mask: Tensor,
    max_slices: int,
) -> tuple[Tensor, list[int]]:
    """Select and pack evenly spaced valid slices from every subject."""

    packed_slices: list[Tensor] = []
    slice_counts: list[int] = []

    for subject_images, subject_mask in zip(images, slice_mask):
        valid_images = subject_images[subject_mask]
        count = min(len(valid_images), max_slices)
        if count == 0:
            raise ValueError("A subject has no valid slices")

        indices = torch.linspace(
            0,
            len(valid_images) - 1,
            steps=count,
            device=valid_images.device,
        ).round().long()
        packed_slices.append(valid_images[indices])
        slice_counts.append(count)

    return torch.cat(packed_slices, dim=0), slice_counts


def run_epoch(
    model: SubjectConvNeXt,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_slices: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> EpochMetrics:
    """Run one training or validation epoch and return subject-level metrics."""

    is_training = optimizer is not None
    model.train(is_training)
    use_amp = device.type == "cuda"

    total_loss = 0.0
    total_correct = 0
    total_subjects = 0

    for batch in loader:
        packed_slices, slice_counts = pack_evenly_spaced_slices(
            batch["images"], batch["slice_mask"], max_slices
        )
        packed_slices = packed_slices.to(device, non_blocking=use_amp)
        labels = batch["labels"].to(device, non_blocking=use_amp)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(packed_slices, slice_counts)
                loss = criterion(logits, labels)

            if is_training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

        subject_count = labels.size(0)
        total_loss += loss.detach().item() * subject_count
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_subjects += subject_count

    if total_subjects == 0:
        raise RuntimeError("The data loader produced no subjects")

    return EpochMetrics(
        loss=total_loss / total_subjects,
        accuracy=total_correct / total_subjects,
    )


def infer_test_set(
    model: SubjectConvNeXt,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_slices: int,
    ad_index: int,
) -> tuple[float, list[TestResult]]:
    """Run subject-level test inference and retain per-subject results."""

    model.eval()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_subjects = 0
    results: list[TestResult] = []

    with torch.inference_mode():
        for batch in loader:
            packed_slices, slice_counts = pack_evenly_spaced_slices(
                batch["images"], batch["slice_mask"], max_slices
            )
            packed_slices = packed_slices.to(device, non_blocking=use_amp)
            labels = batch["labels"].to(device, non_blocking=use_amp)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(packed_slices, slice_counts)
                loss = criterion(logits, labels)

            probabilities = logits.softmax(dim=1).float().cpu()
            predictions = probabilities.argmax(dim=1)
            subject_count = labels.size(0)
            total_loss += loss.item() * subject_count
            total_subjects += subject_count

            for sample_id, target, prediction, probability in zip(
                batch["sample_ids"],
                batch["labels"],
                predictions,
                probabilities,
            ):
                results.append(
                    TestResult(
                        sample_id=str(sample_id),
                        target=int(target.item()),
                        prediction=int(prediction.item()),
                        ad_probability=float(probability[ad_index].item()),
                    )
                )

    if total_subjects == 0:
        raise RuntimeError("The test loader produced no subjects")

    return total_loss / total_subjects, results


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
            f"  {result.sample_id}: "
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
    """Print test metrics and save a labelled confusion-matrix figure."""

    # Use a non-interactive backend and keep Matplotlib's generated cache with
    # the other ignored training outputs rather than in the user's home folder.
    matplotlib_config_dir = output_dir / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    print("\nTest results (AD is the positive class)")
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {accuracy:.2%} ({len(correct)}/{len(results)})")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 score:  {f1:.2%}")

    print("\nTest confusion matrix")
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
    cell_labels = [
        [f"TN\n{true_negatives}", f"FP\n{false_positives}"],
        [f"FN\n{false_negatives}", f"TP\n{true_positives}"],
    ]

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    heatmap = axis.imshow(confusion_matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Predicted NC", "Predicted AD"])
    axis.set_yticks([0, 1], labels=["Actual NC", "Actual AD"])
    axis.set_title("AD vs NC Test Confusion Matrix")

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

    figure.colorbar(heatmap, ax=axis, label="Subjects")
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

    return {
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = args.output_dir / "best_model_weights.pt"
    last_weights_path = args.output_dir / "last_model_weights.pt"
    metrics_path = args.output_dir / "metrics.json"

    train_loader, validation_loader, test_loader = create_adni_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        image_size=IMAGE_SIZE,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    training_dataset = train_loader.dataset.dataset
    class_to_idx = dict(training_dataset.class_to_idx)
    if "AD" not in class_to_idx or "NC" not in class_to_idx:
        raise ValueError("The dataset must contain AD and NC classes")
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    ad_index = class_to_idx["AD"]
    nc_index = class_to_idx["NC"]

    model = SubjectConvNeXt(num_classes=len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Device: {device}")
    print(f"Classes: {class_to_idx}")
    print(f"Training subjects: {len(train_loader.dataset)}")
    print(f"Validation subjects: {len(validation_loader.dataset)}")
    print(f"Test subjects: {len(test_loader.dataset)}")
    print(f"Selected slices per subject: at most {args.max_slices}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.max_slices,
            optimizer=optimizer,
            scaler=scaler,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            args.max_slices,
        )

        history.append(
            {
                "epoch": epoch,
                "train": asdict(train_metrics),
                "validation": asdict(validation_metrics),
            }
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_metrics.loss:.4f}, "
            f"accuracy {train_metrics.accuracy:.2%} | "
            f"validation loss {validation_metrics.loss:.4f}, "
            f"accuracy {validation_metrics.accuracy:.2%}"
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
            epochs_without_improvement += 1
            print(
                f"  No validation-loss improvement of at least "
                f"{EARLY_STOPPING_MIN_DELTA:.3f} "
                f"({epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping after epoch {epoch}: validation loss did not "
                f"improve by at least {EARLY_STOPPING_MIN_DELTA:.3f} for "
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
        args.max_slices,
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
                    "max_slices": args.max_slices,
                    "image_size": list(IMAGE_SIZE),
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
                    "seed": args.seed,
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
