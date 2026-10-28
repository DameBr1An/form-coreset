#!/usr/bin/env python3
"""Evaluate star-merge rules on labeled same-source datasets.

Rules:
- current:      min(score, corr) >= 0.20
- replacement:  score >= 0.15 and corr >= 0.80
- union:        current OR replacement
- union_085:    current OR (score >= 0.15 and corr >= 0.85)
- union_090:    current OR (score >= 0.15 and corr >= 0.90)

The goal is to justify (or reject) a rule before changing product defaults.

Usage:
    uv run python benchmarks/evaluate_star_rules.py --dataset synthetic
    uv run python benchmarks/evaluate_star_rules.py --dataset jett
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from formcoreset.config import SameSourceConfig
from formcoreset.ingest import scan_directory
from formcoreset.source import (
    _descriptors,
    _mask_for,
    _medoid,
    match_descriptors,
    partition_by_scores,
    verify_homography,
)

JETT_FAMILIES = {
    "004": "A", "008": "A", "017": "A",
    "006": "B", "009": "B", "010": "B", "011": "B", "012": "B", "015": "B",
    "005": "C", "016": "C",
    "007": "D", "024": "D",
}


def _truth_for(dataset: str, benchmark_dir: Path, records: list[Any]) -> dict[str, str]:
    if dataset == "synthetic":
        manifest = [
            json.loads(line)
            for line in (benchmark_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_name = {Path(row["path"]).name: row["source_id"] for row in manifest}
        return {
            record.image_id: by_name.get(Path(record.rel_path).name, record.image_id)
            for record in records
        }

    return {
        record.image_id: JETT_FAMILIES.get(
            Path(record.rel_path).stem.replace("bing_", ""),
            Path(record.rel_path).stem.replace("bing_", ""),
        )
        for record in records
    }


def _build_matrices(
    records: list[Any],
    config: SameSourceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(records)
    scores = np.zeros((count, count), dtype=np.float32)
    correlations = np.zeros((count, count), dtype=np.float32)
    confidence = np.zeros((count, count), dtype=np.float32)

    items: list[tuple[Any, Any, Any, Any]] = []
    for record in records:
        mask = (
            _mask_for(record.path, config.mask_scale, config.mask_provider)
            if config.use_mask
            else None
        )
        keypoints, descriptors = _descriptors(
            record.path,
            config.method,
            config.max_features,
            mask=mask,
        )
        items.append((record, mask, keypoints, descriptors))

    for left in range(count):
        for right in range(left + 1, count):
            left_record, left_mask, left_kp, left_des = items[left]
            right_record, right_mask, right_kp, right_des = items[right]
            if left_des is None or right_des is None:
                continue
            result = match_descriptors(
                left_kp,
                left_des,
                right_kp,
                right_des,
                method=config.method,
                ratio=config.ratio,
            )
            if result.homography is None or result.score < 0.1:
                continue
            if result.inliers < config.min_inliers:
                continue
            corr = verify_homography(
                left_record.path,
                right_record.path,
                result.homography,
                left_mask,
                right_mask,
            )
            scores[left, right] = scores[right, left] = result.score
            correlations[left, right] = correlations[right, left] = corr
            confidence[left, right] = confidence[right, left] = min(result.score, corr)

    return scores, correlations, confidence


def _qualifies(score: float, corr: float, rule: str) -> bool:
    current = min(score, corr) >= 0.20
    if rule == "current":
        return current
    if rule == "replacement":
        return score >= 0.15 and corr >= 0.80
    if rule == "union":
        return current or (score >= 0.15 and corr >= 0.80)
    if rule == "union_085":
        return current or (score >= 0.15 and corr >= 0.85)
    if rule == "union_090":
        return current or (score >= 0.15 and corr >= 0.90)
    raise ValueError(rule)


def _star_merge(
    partitions: list[list[str]],
    ids: list[str],
    scores: np.ndarray,
    correlations: np.ndarray,
    confidence: np.ndarray,
    rule: str,
) -> list[list[str]]:
    index = {image_id: i for i, image_id in enumerate(ids)}
    multi = [list(group) for group in partitions if len(group) >= 2]
    singletons = [group[0] for group in partitions if len(group) == 1]
    if not multi:
        return [list(group) for group in partitions]

    medoids = [_medoid(group, confidence, index) for group in multi]
    merged = [list(group) for group in multi]
    leftover: list[list[str]] = []

    for singleton in singletons:
        i = index[singleton]
        best_group = None
        best_score = -1.0
        for group_index, medoid in enumerate(medoids):
            j = index[medoid]
            if _qualifies(float(scores[i, j]), float(correlations[i, j]), rule):
                value = float(confidence[i, j])
                if value > best_score:
                    best_score = value
                    best_group = group_index
        if best_group is None:
            leftover.append([singleton])
        else:
            merged[best_group].append(singleton)

    groups = merged + leftover
    order = {image_id: i for i, image_id in enumerate(ids)}
    groups.sort(key=lambda group: min(order[item] for item in group))
    return groups


def _metrics(groups: list[list[str]], truth: dict[str, str]) -> dict[str, Any]:
    group_of = {
        image_id: group_index
        for group_index, group in enumerate(groups)
        for image_id in group
    }
    ids = list(group_of)
    tp = fp = fn = 0
    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            a, b = ids[left], ids[right]
            same_truth = truth[a] == truth[b]
            same_pred = group_of[a] == group_of[b]
            if same_truth and same_pred:
                tp += 1
            elif same_pred and not same_truth:
                fp += 1
            elif same_truth and not same_pred:
                fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mixed = sum(
        1
        for group in groups
        if len({truth[image_id] for image_id in group}) > 1
    )
    return {
        "groups": len(groups),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mixed_groups": mixed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "jett"], required=True)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--method", choices=["sift", "orb"], default="sift")
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir or (
        Path("benchmarks/data/anime_same_source_small")
        if args.dataset == "synthetic"
        else Path("benchmarks/data/jett_bing_en")
    )
    config = SameSourceConfig()
    config.method = args.method
    scan = scan_directory(benchmark_dir / "images")
    ids = [record.image_id for record in scan.records]
    truth = _truth_for(args.dataset, benchmark_dir, scan.records)

    print(f"Dataset: {args.dataset}; images={len(ids)}; building pair evidence ...")
    scores, correlations, confidence = _build_matrices(scan.records, config)

    seed = partition_by_scores(
        ids,
        confidence,
        linkage_method="complete",
        threshold=config.min_score,
    )
    print(f"Seed groups: {len(seed)}")

    for rule in ("current", "replacement", "union", "union_085", "union_090"):
        groups = _star_merge(seed, ids, scores, correlations, confidence, rule)
        metrics = _metrics(groups, truth)
        print(
            f"{rule:12s} groups={metrics['groups']:2d} "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
            f"F1={metrics['f1']:.3f} mixed={metrics['mixed_groups']}"
        )


if __name__ == "__main__":
    main()
