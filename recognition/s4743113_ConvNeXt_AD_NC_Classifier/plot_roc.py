"""Draw ROC curves from saved per-slice predictions, one or more runs per plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train import DEFAULT_OUTPUT_DIR


COLOURS = ("#3568b8", "#e77b32", "#3f7d44", "#a63d62")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        metavar="NAME=PATH",
        help="Label and predictions file; repeat to overlay several runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "test_roc_curve.svg",
    )
    parser.add_argument("--title", default="Test ROC Curve")
    args = parser.parse_args()
    if not args.run:
        parser.error("at least one --run NAME=PATH is required")
    return args


def load_predictions(path: Path) -> tuple[list[float], list[int]]:
    """Return AD probabilities and binary AD labels from a predictions file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    ad_index = payload["ad_index"]
    probabilities = [item["ad_probability"] for item in payload["results"]]
    labels = [int(item["target"] == ad_index) for item in payload["results"]]
    return probabilities, labels


def roc_points(
    probabilities: list[float],
    labels: list[int],
) -> tuple[list[tuple[float, float]], float]:
    """Return (FPR, TPR) points and the trapezoidal area under the curve."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("ROC needs both classes present")

    ordered = sorted(zip(probabilities, labels), key=lambda pair: -pair[0])
    points = [(0.0, 0.0)]
    true_positives = 0
    false_positives = 0
    index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        # Tied probabilities share one threshold and so one ROC point.
        while index < len(ordered) and ordered[index][0] == threshold:
            if ordered[index][1]:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        points.append((false_positives / negatives, true_positives / positives))
    if points[-1] != (1.0, 1.0):
        points.append((1.0, 1.0))

    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return points, area


def _svg(curves: list[tuple[str, list[tuple[float, float]], float]], title: str) -> str:
    width, height = 720, 720
    left, top, size = 90, 70, 560
    bottom = top + size

    def x_of(value: float) -> float:
        return left + value * size

    def y_of(value: float) -> float:
        return bottom - value * size

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" font-size="23" '
        f'font-weight="bold">{title}</text>',
    ]

    for step in range(0, 11, 2):
        value = step / 10
        parts.append(
            f'<line x1="{x_of(value):.1f}" y1="{top}" x2="{x_of(value):.1f}" '
            f'y2="{bottom}" stroke="#e2e2e2"/>'
        )
        parts.append(
            f'<line x1="{left}" y1="{y_of(value):.1f}" x2="{left + size}" '
            f'y2="{y_of(value):.1f}" stroke="#e2e2e2"/>'
        )
        parts.append(
            f'<text x="{x_of(value):.1f}" y="{bottom + 24}" text-anchor="middle" '
            f'font-size="13">{value:.1f}</text>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y_of(value) + 5:.1f}" text-anchor="end" '
            f'font-size="13">{value:.1f}</text>'
        )

    parts.append(
        f'<line x1="{left}" y1="{bottom}" x2="{left + size}" y2="{top}" '
        'stroke="#9a9a9a" stroke-dasharray="6 5"/>'
    )
    parts.append(
        f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="none" '
        'stroke="#444"/>'
    )
    parts.append(
        f'<text x="{left + size / 2}" y="{bottom + 52}" text-anchor="middle" '
        'font-size="15">False positive rate (1 - specificity)</text>'
    )
    parts.append(
        f'<text x="26" y="{top + size / 2}" text-anchor="middle" font-size="15" '
        f'transform="rotate(-90 26 {top + size / 2})">True positive rate (recall)</text>'
    )

    for position, (label, points, area) in enumerate(curves):
        colour = COLOURS[position % len(COLOURS)]
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{x_of(x):.2f},{y_of(y):.2f}"
            for index, (x, y) in enumerate(points)
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2.5"/>'
        )
        # Legend sits bottom-right, which stays clear of any useful ROC curve.
        legend_y = bottom - 52 + position * 24
        parts.append(
            f'<line x1="{left + size - 250}" y1="{legend_y}" '
            f'x2="{left + size - 220}" y2="{legend_y}" stroke="{colour}" '
            'stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{left + size - 212}" y="{legend_y + 5}" font-size="14">'
            f"{label} (AUC {area:.4f})</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    curves = []
    for entry in args.run:
        if "=" not in entry:
            raise ValueError(f"Expected NAME=PATH, received: {entry}")
        label, _, raw_path = entry.partition("=")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Predictions file does not exist: {path}")
        points, area = roc_points(*load_predictions(path))
        curves.append((label, points, area))
        print(f"{label}: {len(points)} points, AUC {area:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_svg(curves, args.title), encoding="utf-8")
    print(f"ROC curve saved to: {args.output}")


if __name__ == "__main__":
    main()
