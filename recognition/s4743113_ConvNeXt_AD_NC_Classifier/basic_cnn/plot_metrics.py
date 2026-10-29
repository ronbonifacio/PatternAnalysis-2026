"""Create dependency-free SVG plots from the shallow-CNN metrics JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train import save_training_curves

from basic_cnn.common import DEFAULT_OUTPUT_DIR


METRIC_KEYS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("specificity", "Specificity"),
    ("f1", "F1"),
    ("roc_auc", "ROC-AUC"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "metrics.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _metric_bar_svg(
    primary: dict[str, float],
    secondary: dict[str, float] | None,
) -> str:
    width, height = 1100, 540
    left, top, chart_width, chart_height = 75, 80, 965, 340
    group_width = chart_width / len(METRIC_KEYS)
    bar_width = 42 if secondary is not None else 70
    colours = ("#3568b8", "#e77b32")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="550" y="35" text-anchor="middle" font-size="25" font-weight="bold">Basic CNN Test Metrics</text>',
    ]
    for tick in range(6):
        score = tick / 5
        y = top + chart_height * (1.0 - score)
        elements.extend(
            (
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#dddddd"/>',
                f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="13">{score:.1f}</text>',
            )
        )

    series = [primary] + ([secondary] if secondary is not None else [])
    for metric_index, (key, label) in enumerate(METRIC_KEYS):
        centre = left + group_width * (metric_index + 0.5)
        for series_index, values in enumerate(series):
            value = float(values[key])
            x = centre - (len(series) * bar_width) / 2 + series_index * bar_width
            bar_height = chart_height * value
            y = top + chart_height - bar_height
            elements.extend(
                (
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 4}" height="{bar_height:.1f}" fill="{colours[series_index]}"/>',
                    f'<text x="{x + (bar_width - 4) / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="11">{value:.3f}</text>',
                )
            )
        elements.append(
            f'<text x="{centre:.1f}" y="450" text-anchor="middle" font-size="13">{label}</text>'
        )

    elements.extend(
        (
            '<rect x="300" y="485" width="20" height="12" fill="#3568b8"/>',
            '<text x="330" y="496" font-size="14">Balanced-accuracy threshold</text>',
        )
    )
    if secondary is not None:
        elements.extend(
            (
                '<rect x="600" y="485" width="20" height="12" fill="#e77b32"/>',
                '<text x="630" y="496" font-size="14">95% validation-recall threshold</text>',
            )
        )
    elements.append("</svg>")
    return "\n".join(elements)


def _confusion_matrix_svg(metrics: dict[str, float], title: str) -> str:
    matrix = (
        (int(metrics["true_negatives"]), int(metrics["false_positives"])),
        (int(metrics["false_negatives"]), int(metrics["true_positives"])),
    )
    maximum = max(value for row in matrix for value in row)
    cells: list[str] = []
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            x = 155 + column_index * 190
            y = 105 + row_index * 190
            opacity = 0.15 + 0.75 * value / maximum
            text_colour = "white" if opacity > 0.55 else "#172033"
            cells.extend(
                (
                    f'<rect x="{x}" y="{y}" width="190" height="190" fill="#1764aa" fill-opacity="{opacity:.3f}"/>',
                    f'<text x="{x + 95}" y="{y + 105}" text-anchor="middle" font-size="29" font-weight="bold" fill="{text_colour}">{value:,}</text>',
                )
            )
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="540" height="560" viewBox="0 0 540 560">',
            '<rect width="100%" height="100%" fill="#fafafa"/>',
            f'<text x="270" y="32" text-anchor="middle" font-size="22" font-weight="bold">{title}</text>',
            f'<text x="270" y="60" text-anchor="middle" font-size="15">Threshold {float(metrics["decision_threshold"]):.4f}</text>',
            '<text x="250" y="525" text-anchor="middle" font-size="17">Predicted class</text>',
            '<text x="60" y="295" text-anchor="middle" font-size="17" transform="rotate(-90 60 295)">Actual class</text>',
            '<text x="250" y="95" text-anchor="middle" font-size="16">NC</text>',
            '<text x="440" y="95" text-anchor="middle" font-size="16">AD</text>',
            '<text x="135" y="205" text-anchor="end" font-size="16">NC</text>',
            '<text x="135" y="395" text-anchor="end" font-size="16">AD</text>',
            *cells,
            '</svg>',
        ]
    )


def main() -> None:
    args = parse_args()
    if not args.metrics.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {args.metrics}")
    output_dir = args.output_dir or args.metrics.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = json.loads(args.metrics.read_text(encoding="utf-8"))

    history = metrics.get("history")
    if history:
        path = save_training_curves(
            history,
            output_dir,
            plot_title="Basic CNN Training History",
        )
        print(f"Training curves saved to: {path}")

    primary = metrics["test"]
    target_report = metrics.get("target_recall_operating_point")
    secondary = target_report.get("test") if target_report else None
    summary_path = output_dir / "test_metric_comparison.svg"
    summary_path.write_text(
        _metric_bar_svg(primary, secondary),
        encoding="utf-8",
    )
    print(f"Metric comparison saved to: {summary_path}")

    primary_path = output_dir / "test_confusion_matrix.svg"
    primary_path.write_text(
        _confusion_matrix_svg(primary, "Basic CNN Test Confusion Matrix"),
        encoding="utf-8",
    )
    print(f"Confusion matrix saved to: {primary_path}")

    if secondary is not None:
        recall_path = output_dir / "test_confusion_matrix_95_recall.svg"
        recall_path.write_text(
            _confusion_matrix_svg(
                secondary,
                "Basic CNN High-Recall Confusion Matrix",
            ),
            encoding="utf-8",
        )
        print(f"High-recall confusion matrix saved to: {recall_path}")


if __name__ == "__main__":
    main()
