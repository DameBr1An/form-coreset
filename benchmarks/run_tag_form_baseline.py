#!/usr/bin/env python3
"""WD14 tag-based form baseline for the pose-balanced anime benchmark.

Why this exists: DWPose / RTMPose HumanArt keypoints are unreliable on
stylized anime (confirmed by keypoint overlays). WD14 tags are a stronger
anime-specific signal for posture/view, though still noisy.

Pipeline:
- WD14 general tags -> fixed posture/view/action vocabulary;
- KMeans (k = number of pose classes) and HDBSCAN;
- compare clusters to the query pose label (weak label).

Usage:
    uv sync --extra anime
    uv run python benchmarks/run_tag_form_baseline.py \
        --benchmark-dir benchmarks/data/anime_miku_pose_small
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from run_pose_form_baseline import _cluster_purity, _write_cluster_sheet  # noqa: I001
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from formcoreset.form.anime_hybrid import build_tag_vector
from formcoreset.form.vocab import POSTURE_TAGS, TAG_VOCAB

try:
    from imgutils.tagging import get_wd14_tags
except ImportError as exc:  # pragma: no cover - runtime dependency hint
    raise SystemExit("dghs-imgutils is required. Run: uv sync --extra anime") from exc


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tag_prediction(tags: dict[str, float]) -> tuple[str, float]:
    best_tag = "unknown"
    best_score = 0.0
    for tag in POSTURE_TAGS:
        score = float(tags.get(tag, 0.0))
        if score > best_score:
            best_tag = tag
            best_score = score
    if best_score < 0.2:
        return "unknown", best_score
    return best_tag, best_score


def _cluster_metrics(
    labels: np.ndarray,
    truth: list[str],
) -> dict[str, Any]:
    mask = labels >= 0
    known = np.asarray([label != "unknown" for label in truth])
    valid = mask & known
    if valid.sum() > 1:
        ami = float(adjusted_mutual_info_score(np.asarray(truth)[valid], labels[valid]))
        nmi = float(normalized_mutual_info_score(np.asarray(truth)[valid], labels[valid]))
    else:
        ami = None
        nmi = None
    return {
        "n_clusters": int(len({int(label) for label in labels if label >= 0})),
        "noise": int((labels == -1).sum()),
        "ami": ami,
        "nmi": nmi,
        "purity": _cluster_purity(labels, truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("benchmarks/data/anime_miku_pose_small"),
    )
    parser.add_argument("--split", default="all")
    parser.add_argument(
        "--pose-label-field",
        default="pose_label",
        help="Manifest field used as the pose label.",
    )
    parser.add_argument("--general-threshold", type=float, default=0.2)
    parser.add_argument("--tag-threshold", type=float, default=0.2)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <benchmark>/tag_form_baseline).",
    )
    args = parser.parse_args()

    benchmark_dir: Path = args.benchmark_dir.resolve()
    out_dir = (args.out or (benchmark_dir / "tag_form_baseline")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in _load_manifest(benchmark_dir / "manifest.jsonl")
        if args.split == "all" or row["split"] == args.split
    ]
    print(f"Running WD14 on {len(rows)} images ...")

    features: list[np.ndarray] = []
    truth: list[str] = []
    tag_predictions: list[dict[str, Any]] = []
    tag_cache: dict[str, dict[str, float]] = {}

    for index, row in enumerate(rows, start=1):
        _, general, _ = get_wd14_tags(
            benchmark_dir / row["path"],
            general_threshold=args.general_threshold,
        )
        tag_cache[row["image_id"]] = general
        features.append(build_tag_vector(general, args.tag_threshold))
        pose_label = str(row.get(args.pose_label_field) or "unknown")
        truth.append(pose_label)
        predicted, score = _tag_prediction(general)
        tag_predictions.append(
            {
                "image_id": row["image_id"],
                "pose_label": pose_label,
                "tag_prediction": predicted,
                "tag_confidence": score,
                "pose_view_tags": {
                    tag: round(float(general[tag]), 4)
                    for tag in TAG_VOCAB
                    if tag in general
                },
            }
        )
        if index % 10 == 0:
            print(f"  {index}/{len(rows)}")

    matrix = np.stack(features)
    n_classes = len({label for label in truth if label != "unknown"})
    kmeans = KMeans(n_clusters=max(2, n_classes), n_init=20, random_state=42)
    kmeans_labels = kmeans.fit_predict(matrix)

    hdbscan = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        copy=True,
    )
    hdbscan_labels = hdbscan.fit_predict(matrix)

    kmeans_metrics = _cluster_metrics(kmeans_labels, truth)
    hdbscan_metrics = _cluster_metrics(hdbscan_labels, truth)

    known = [prediction for prediction in tag_predictions if prediction["pose_label"] != "unknown"]
    tag_correct = sum(
        prediction["tag_prediction"] == prediction["pose_label"]
        for prediction in known
    )
    tag_coverage = sum(
        prediction["tag_prediction"] != "unknown" for prediction in known
    ) / len(known) if known else 0.0
    tag_accuracy = tag_correct / len(known) if known else 0.0

    prediction_by_id = {prediction["image_id"]: prediction for prediction in tag_predictions}
    for row, kmeans_label, hdbscan_label in zip(
        rows, kmeans_labels, hdbscan_labels, strict=True
    ):
        prediction = prediction_by_id[row["image_id"]]
        prediction["kmeans_cluster"] = int(kmeans_label)
        prediction["hdbscan_cluster"] = int(hdbscan_label)

    summary = {
        "benchmark_dir": benchmark_dir.as_posix(),
        "split": args.split,
        "pose_label_field": args.pose_label_field,
        "images": len(rows),
        "pose_classes": n_classes,
        "tag_vocabulary": TAG_VOCAB,
        "tag_direct_accuracy": tag_accuracy,
        "tag_direct_coverage": tag_coverage,
        "kmeans": kmeans_metrics,
        "hdbscan": hdbscan_metrics,
        "pose_label_counts": {
            label: truth.count(label) for label in sorted(set(truth))
        },
        "note": "WD14 tags and query pose labels are weak labels, not human ground truth.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in tag_predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    kmeans_dir = out_dir / "kmeans"
    hdbscan_dir = out_dir / "hdbscan"
    kmeans_dir.mkdir(exist_ok=True)
    hdbscan_dir.mkdir(exist_ok=True)
    for cluster in sorted({int(label) for label in kmeans_labels}):
        _write_cluster_sheet(
            kmeans_dir,
            benchmark_dir,
            rows,
            kmeans_labels,
            cluster=cluster,
            label_field=args.pose_label_field,
        )
    for cluster in sorted({int(label) for label in hdbscan_labels}):
        _write_cluster_sheet(
            hdbscan_dir,
            benchmark_dir,
            rows,
            hdbscan_labels,
            cluster=cluster,
            label_field=args.pose_label_field,
        )

    print()
    print(f"Tag direct accuracy: {tag_accuracy:.3f} (coverage={tag_coverage:.3f})")
    print(
        f"KMeans(k={max(2, n_classes)}): AMI={kmeans_metrics['ami']}, "
        f"purity={kmeans_metrics['purity']:.3f}"
    )
    print(
        f"HDBSCAN: clusters={hdbscan_metrics['n_clusters']}, "
        f"noise={hdbscan_metrics['noise']}, AMI={hdbscan_metrics['ami']}, "
        f"purity={hdbscan_metrics['purity']:.3f}"
    )
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
