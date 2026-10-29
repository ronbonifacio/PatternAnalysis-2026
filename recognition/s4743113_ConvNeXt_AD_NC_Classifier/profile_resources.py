"""Profile ConvNeXt-Tiny parameter count, peak CUDA memory and inference time.

The timing covers the model forward pass on an in-memory batch.  It excludes
image decoding, preprocessing and data-loader overhead so repeated runs measure
the model itself rather than shared-filesystem performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from modules import convnext_tiny


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (
    SCRIPT_DIR
    / "output"
    / "convnext_tiny_64px_2_5d_plateau"
    / "best_model_weights.pt"
)
DEFAULT_OUTPUT_DIR = DEFAULT_WEIGHTS.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--precision",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.image_size <= 0:
        parser.error("--image-size must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.warmup_iterations < 0:
        parser.error("--warmup-iterations cannot be negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    return args


def _format_markdown(profile: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# ConvNeXt-Tiny resource profile",
            "",
            (
                "Forward-pass timing uses an in-memory synthetic tensor and "
                "therefore excludes image decoding, preprocessing and data loading."
            ),
            "",
            "| Model | GPU | Precision | Input | Batch | Parameters | Peak allocated VRAM | Peak reserved VRAM | Batch latency | Per-slice latency | Throughput |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| ConvNeXt-Tiny | {profile['device_name']} | "
                f"{profile['precision']} | {profile['image_size']}×"
                f"{profile['image_size']}×3 | {profile['batch_size']} | "
                f"{profile['parameter_count']:,} | "
                f"{profile['peak_memory_allocated_mib']:.2f} MiB | "
                f"{profile['peak_memory_reserved_mib']:.2f} MiB | "
                f"{profile['batch_latency_ms']:.3f} ms | "
                f"{profile['per_slice_latency_ms']:.3f} ms | "
                f"{profile['throughput_slices_per_second']:.2f} slices/s |"
            ),
            "",
            f"Measured over {profile['iterations']} iterations after "
            f"{profile['warmup_iterations']} warm-up iterations.",
        )
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requires a Slurm GPU allocation")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Model weights do not exist: {args.weights}")
    if args.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The allocated GPU does not support bfloat16")

    device = torch.device(args.device)
    autocast_dtype = {
        "float32": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.precision]

    model = convnext_tiny(
        num_classes=2,
        stochastic_depth_probability=0.1,
    )
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict
    model = model.to(device).eval()
    sample = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=device,
    )

    use_autocast = autocast_dtype is not None
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=autocast_dtype or torch.float16,
        enabled=use_autocast,
    ):
        for _ in range(args.warmup_iterations):
            model(sample)
        torch.cuda.synchronize(device)

        # Reset after warm-up. Live model and input allocations remain part of
        # the peak, while one-off kernel initialization allocations do not.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            output = model(sample)
        end.record()
        torch.cuda.synchronize(device)

    if output.shape != (args.batch_size, 2):
        raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")

    elapsed_ms = start.elapsed_time(end)
    batch_latency_ms = elapsed_ms / args.iterations
    per_slice_latency_ms = batch_latency_ms / args.batch_size
    throughput = 1000.0 * args.batch_size / batch_latency_ms
    profile = {
        "model": "ConvNeXt-Tiny",
        "weights": str(args.weights.resolve()),
        "device_name": torch.cuda.get_device_name(device),
        "precision": args.precision,
        "image_size": args.image_size,
        "input_channels": 3,
        "batch_size": args.batch_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "warmup_iterations": args.warmup_iterations,
        "iterations": args.iterations,
        "total_inference_runtime_ms": elapsed_ms,
        "batch_latency_ms": batch_latency_ms,
        "per_slice_latency_ms": per_slice_latency_ms,
        "throughput_slices_per_second": throughput,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    profile["peak_memory_allocated_mib"] = (
        profile["peak_memory_allocated_bytes"] / 1024**2
    )
    profile["peak_memory_reserved_mib"] = (
        profile["peak_memory_reserved_bytes"] / 1024**2
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "resource_profile.json"
    markdown_path = args.output_dir / "resource_profile.md"
    json_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    markdown = _format_markdown(profile)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"JSON profile saved to: {json_path}")
    print(f"Markdown table saved to: {markdown_path}")


if __name__ == "__main__":
    main()
