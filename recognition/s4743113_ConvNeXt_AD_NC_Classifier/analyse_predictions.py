"""Report confidence groups, per-scan errors and scan-level accuracy.

Reads the per-slice predictions written during evaluation and the thresholds
recorded in the matching metrics file, so the numbers quoted in the README can
be reproduced without rerunning inference.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from train import DEFAULT_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "test_slice_predictions.json",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "metrics.json",
        help="Metrics file supplying both decision thresholds.",
    )
    parser.add_argument(
        "--scans",
        nargs="*",
        default=(),
        metavar="SCAN_ID",
        help="Also report these scans individually.",
    )
    return parser.parse_args()


def load_thresholds(metrics_path: Path) -> tuple[float, float]:
    """Return the low and high thresholds selected on validation data."""

    metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    configuration = metrics.get("configuration", metrics)
    high = float(configuration["decision_threshold"])
    recall_report = metrics.get("target_recall_operating_point") or {}
    low = float(recall_report.get("decision_threshold", high))
    return low, high


def main() -> None:
    args = parse_args()
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    ad_index = payload["ad_index"]
    results = payload["results"]
    low, high = load_thresholds(args.metrics)

    def is_ad(result: dict[str, Any]) -> bool:
        return result["target"] == ad_index

    total = len(results)
    ad_total = sum(is_ad(result) for result in results)
    print(f"{total} slices, {ad_total} AD ({ad_total / total:.1%}), "
          f"thresholds {low:.4f} and {high:.4f}\n")

    # Confidence groups.
    confident_ad = [r for r in results if r["ad_probability"] >= high]
    confident_nc = [r for r in results if r["ad_probability"] < low]
    unsure = [r for r in results if low <= r["ad_probability"] < high]
    answered = len(confident_ad) + len(confident_nc)
    correct = sum(is_ad(r) for r in confident_ad) + sum(
        not is_ad(r) for r in confident_nc
    )
    overall = sum(
        (r["ad_probability"] >= high) == is_ad(r) for r in results
    ) / total

    print("Confidence groups")
    print(f"  very likely AD   {len(confident_ad):5d} ({len(confident_ad) / total:6.1%})"
          f"  correct {sum(is_ad(r) for r in confident_ad) / len(confident_ad):.1%}")
    print(f"  unsure           {len(unsure):5d} ({len(unsure) / total:6.1%})"
          f"  AD share {sum(is_ad(r) for r in unsure) / len(unsure):.1%}")
    print(f"  very likely NC   {len(confident_nc):5d} ({len(confident_nc) / total:6.1%})"
          f"  correct {sum(not is_ad(r) for r in confident_nc) / len(confident_nc):.1%}")
    print(f"  answered         {answered:5d} ({answered / total:6.1%})"
          f"  accuracy {correct / answered:.1%}")
    print(f"  all slices       {total:5d} (100.0%)  accuracy {overall:.1%}\n")

    # Per-scan behaviour.
    by_scan: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_scan[result["sample_id"]].append(result)

    all_wrong = partly_wrong = clean = 0
    scan_correct = 0
    for slices in by_scan.values():
        label = is_ad(slices[0])
        wrong = sum((s["ad_probability"] >= high) != label for s in slices)
        if wrong == len(slices):
            all_wrong += 1
        elif wrong:
            partly_wrong += 1
        else:
            clean += 1
        mean = sum(s["ad_probability"] for s in slices) / len(slices)
        scan_correct += (mean >= high) == label

    print(f"Scans ({len(by_scan)} total)")
    print(f"  every slice wrong {all_wrong}")
    print(f"  some slices wrong {partly_wrong}")
    print(f"  all slices right  {clean}")
    print(f"  slice-level accuracy      {overall:.2%}")
    print(f"  scan-level accuracy       {scan_correct / len(by_scan):.2%}"
          " (mean probability per scan)\n")

    for sample_id in args.scans:
        slices = sorted(by_scan[sample_id], key=lambda s: s["slice_index"])
        if not slices:
            print(f"Scan {sample_id}: not found")
            continue
        label = is_ad(slices[0])
        wrong = sum((s["ad_probability"] >= high) != label for s in slices)
        probabilities = [s["ad_probability"] for s in slices]
        print(f"Scan {sample_id} ({'AD' if label else 'NC'}): "
              f"{wrong} of {len(slices)} slices wrong, "
              f"mean P(AD) {sum(probabilities) / len(probabilities):.3f}, "
              f"range {min(probabilities):.3f} to {max(probabilities):.3f}")


if __name__ == "__main__":
    main()
