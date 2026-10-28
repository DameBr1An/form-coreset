#!/usr/bin/env python3
"""Evaluate same-source deduplication on the synthetic variant benchmark.

Metrics:
- pairwise precision / recall / F1 over "same source" pairs;
- predicted group purity;
- source coverage (largest predicted group per true source).

Usage:
    uv sync --extra anime
    uv run python benchmarks/run_source_dedup_benchmark.py \
        --benchmark-dir benchmarks/data/anime_same_source_small
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from formcoreset.config import RunConfig
from formcoreset.pipeline import run_same_source_dedup


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("benchmarks/data/anime_same_source_small"),
    )
    parser.add_argument("--method", choices=["sift", "orb"], default="sift")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--max-features", type=int, default=2000)
    parser.add_argument(
        "--linkage",
        choices=["complete", "average", "single"],
        default="complete",
    )
    parser.add_argument("--star-threshold", type=float, default=0.2)
    parser.add_argument("--min-correlation", type=float, default=0.35)
    parser.add_argument("--no-star-merge", action="store_true")
    parser.add_argument(
        "--mask-provider",
        choices=["auto", "anime", "general", "none"],
        default="auto",
    )
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--no-stitch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <benchmark>/dedupe_baseline).",
    )
    args = parser.parse_args()

    benchmark_dir: Path = args.benchmark_dir.resolve()
    out_dir = (args.out or (benchmark_dir / "dedupe_baseline")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(benchmark_dir / "manifest.jsonl")
    truth_by_name = {
        Path(row["path"]).name: row["source_id"] for row in manifest
    }

    config = RunConfig()
    config.source.method = args.method
    config.source.min_score = args.min_score
    config.source.min_inliers = args.min_inliers
    config.source.max_features = args.max_features
    config.source.linkage = args.linkage
    config.source.star_threshold = args.star_threshold
    config.source.star_merge = not args.no_star_merge
    config.source.min_correlation = args.min_correlation
    config.source.mask_provider = "none" if args.no_mask else args.mask_provider
    config.source.use_mask = not args.no_mask and args.mask_provider != "none"
    config.source.verify = not args.no_verify
    config.source.stitch = not args.no_stitch

    scan, groups = run_same_source_dedup(
        benchmark_dir / "images",
        out_dir,
        config,
        progress=True,
    )

    group_by_image: dict[str, str] = {}
    for group in groups:
        for member in group.members:
            group_by_image[member.image_id] = group.source_id

    items = []
    for record in scan.records:
        name = Path(record.rel_path).name
        source_id = truth_by_name.get(name)
        if source_id is None:
            continue
        items.append((record.image_id, source_id, group_by_image.get(record.image_id)))

    true_positive = false_positive = false_negative = 0
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            _, left_truth, left_pred = items[left]
            _, right_truth, right_pred = items[right]
            same_truth = left_truth == right_truth
            same_pred = left_pred == right_pred
            if same_truth and same_pred:
                true_positive += 1
            elif same_pred and not same_truth:
                false_positive += 1
            elif same_truth and not same_pred:
                false_negative += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    group_purity_total = 0
    group_purity_correct = 0
    for group in groups:
        sources = [truth_by_name[Path(member.rel_path).name] for member in group.members]
        if not sources:
            continue
        counts = Counter(sources)
        group_purity_total += len(sources)
        group_purity_correct += max(counts.values())
    group_purity = group_purity_correct / group_purity_total if group_purity_total else 0.0

    source_members: dict[str, list[str]] = defaultdict(list)
    for image_id, source_id, _ in items:
        source_members[source_id].append(image_id)
    predicted_groups: dict[str, list[str]] = defaultdict(list)
    for image_id, _, predicted in items:
        if predicted:
            predicted_groups[predicted].append(image_id)

    source_coverage = []
    for members in source_members.values():
        member_set = set(members)
        best = 0.0
        for predicted_members in predicted_groups.values():
            overlap = len(member_set & set(predicted_members))
            best = max(best, overlap / len(member_set))
        source_coverage.append(best)
    coverage = sum(source_coverage) / len(source_coverage) if source_coverage else 0.0

    summary = {
        "benchmark_dir": benchmark_dir.as_posix(),
        "images": len(items),
        "true_sources": len(source_members),
        "predicted_groups": len(groups),
        "singleton_groups": sum(1 for group in groups if len(group.members) == 1),
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "group_purity": group_purity,
        "source_coverage": coverage,
        "params": {
            "method": args.method,
            "min_score": args.min_score,
            "min_inliers": args.min_inliers,
            "max_features": args.max_features,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Images: {len(items)}; true sources: {len(source_members)}")
    print(f"Predicted groups: {len(groups)}; singletons: {summary['singleton_groups']}")
    print(f"Pairwise precision: {precision:.3f}")
    print(f"Pairwise recall: {recall:.3f}")
    print(f"Pairwise F1: {f1:.3f}")
    print(f"Group purity: {group_purity:.3f}")
    print(f"Source coverage: {coverage:.3f}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
