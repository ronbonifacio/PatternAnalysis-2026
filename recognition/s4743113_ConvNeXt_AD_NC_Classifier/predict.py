"""Load trained ConvNeXt weights and visualise example ADNI predictions."""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from dataset import (
    FULL_FOREGROUND_CROP_SIZE,
    ADNISliceDataset,
    create_adni_slice_dataloaders,
    resize_tensor_image,
)
from modules import convnext_tiny


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "ADNI" / "AD_NC"
DEFAULT_METADATA_PATH = SCRIPT_DIR / "ADNI" / "meta_data_with_label.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "convnext_tiny_64px_2_5d_plateau"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run example predictions with the trained 2.5D ConvNeXt."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "best_model_weights.pt",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "metrics.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument(
        "--example-seed",
        type=int,
        default=42,
        help="Seed selecting which test slices are shown.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    if args.examples <= 0:
        parser.error("--examples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def load_operating_points(metrics_path: Path) -> list[tuple[str, float]]:
    """Load validation-selected thresholds, falling back to standard 0.5."""

    if not metrics_path.is_file():
        print(f"Metrics file not found; using threshold 0.5: {metrics_path}")
        return [("balanced", 0.5)]

    with metrics_path.open(encoding="utf-8") as metrics_file:
        metrics: dict[str, Any] = json.load(metrics_file)
    configuration = metrics.get("configuration", metrics)
    operating_points = [
        ("balanced", float(configuration.get("decision_threshold", 0.5)))
    ]
    target_recall = metrics.get("target_recall_operating_point")
    if isinstance(target_recall, dict) and "decision_threshold" in target_recall:
        target = float(target_recall.get("target_validation_recall", 0.95))
        operating_points.append(
            (
                f"{target:.0%} validation recall",
                float(target_recall["decision_threshold"]),
            )
        )
    return operating_points


def display_image(channel: torch.Tensor, *, size: int = 256) -> Image.Image:
    """Convert one normalized MRI channel into a displayable grayscale image."""

    channel = channel.detach().cpu().float()
    foreground = channel.ne(0)
    scaled = torch.zeros_like(channel)
    if foreground.any():
        values = channel.masked_select(foreground)
        minimum = values.min()
        maximum = values.max()
        if maximum > minimum:
            scaled = torch.where(
                foreground,
                (channel - minimum) / (maximum - minimum),
                0.0,
            )
    pixels = scaled.mul(255).clamp(0, 255).to(torch.uint8)
    height, width = pixels.shape
    image = Image.frombytes("L", (width, height), bytes(pixels.flatten().tolist()))
    return image.resize((size, size), Image.Resampling.BILINEAR).convert("RGB")


def save_prediction_grid(
    examples: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Save centre-slice images with labels, probabilities and predictions."""

    columns = 2
    image_size = 256
    caption_height = 62
    rows = math.ceil(len(examples) / columns)
    canvas = Image.new(
        "RGB",
        (columns * image_size, rows * (image_size + caption_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, example in enumerate(examples):
        column = index % columns
        row = index // columns
        left = column * image_size
        top = row * (image_size + caption_height)
        canvas.paste(display_image(example["image"]), (left, top))
        draw.text(
            (left + 5, top + image_size + 4),
            str(example["caption"]),
            fill="black",
            spacing=3,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {args.weights}")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    evaluation_transform = partial(resize_tensor_image, size=64)
    _, _, test_loader = create_adni_slice_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        validation_fraction=0.2,
        seed=42,
        training_seed=123,
        metadata_path=args.metadata_path,
        excluded_sample_ids=["1012896"],
        image_size=FULL_FOREGROUND_CROP_SIZE,
        train_transform=evaluation_transform,
        evaluation_transform=evaluation_transform,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_dataset = test_loader.dataset
    if not isinstance(test_dataset, ADNISliceDataset):
        raise TypeError("Expected an ADNISliceDataset")
    idx_to_class = {
        index: name for name, index in test_dataset.class_to_idx.items()
    }

    model = convnext_tiny(num_classes=len(idx_to_class), stochastic_depth_probability=0.1)
    model.load_state_dict(
        torch.load(args.weights, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    operating_points = load_operating_points(args.metrics)
    print(f"Loaded weights: {args.weights}")
    for name, threshold in operating_points:
        print(f"{name} threshold: {threshold:.4f}")

    # Sample across the whole test set instead of taking the first slices the
    # loader returns, which would all come from one scan.
    example_count = min(args.examples, len(test_dataset))
    selection_generator = torch.Generator().manual_seed(args.example_seed)
    selected_indices = sorted(
        torch.randperm(len(test_dataset), generator=selection_generator)[
            :example_count
        ].tolist()
    )
    example_loader = DataLoader(
        Subset(test_dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Showing {example_count} random test slices "
        f"(example seed {args.example_seed})"
    )

    examples: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in example_loader:
            images = batch["image"]
            logits = model(images.to(device, non_blocking=True))
            ad_index = test_dataset.class_to_idx["AD"]
            probabilities = logits.softmax(dim=1)[:, ad_index].cpu()
            for item_index, probability in enumerate(probabilities.tolist()):
                target = int(batch["label"][item_index])
                sample_id = str(batch["sample_id"][item_index])
                slice_index = int(batch["slice_index"][item_index])
                predictions = [
                    f"{name}={idx_to_class[ad_index if probability >= threshold else test_dataset.class_to_idx['NC']]}"
                    for name, threshold in operating_points
                ]
                prediction_text = ", ".join(predictions)
                print(
                    f"{sample_id}_{slice_index}: actual={idx_to_class[target]}, "
                    f"P(AD)={probability:.3f}, {prediction_text}"
                )
                examples.append(
                    {
                        "image": images[item_index, 1],
                        "caption": (
                            f"{sample_id}_{slice_index}  actual={idx_to_class[target]}\n"
                            f"P(AD)={probability:.3f}  {prediction_text}"
                        ),
                    }
                )

    visualization_path = args.output_dir / "prediction_examples.png"
    save_prediction_grid(examples, visualization_path)
    print(f"Prediction visualisation saved to: {visualization_path}")


if __name__ == "__main__":
    main()
