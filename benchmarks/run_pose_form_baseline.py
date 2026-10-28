#!/usr/bin/env python3
"""Pose/form baseline for the small anime benchmark.

Pipeline:
- target split only (identity is handled by the CCIP baseline);
- DWPose body keypoints (18 joints);
- torso-centric translation/scale normalization;
- HDBSCAN clustering;
- weak Danbooru pose/view tags as a sanity check, not ground truth.

Usage:
    uv sync --extra anime
    uv run python benchmarks/run_pose_form_baseline.py \
        --benchmark-dir benchmarks/data/anime_miku_small
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

try:
    from imgutils.pose import dwpose_estimate
except ImportError as exc:  # pragma: no cover - runtime dependency hint
    raise SystemExit("dghs-imgutils is required. Run: uv sync --extra anime") from exc

POSTURE_PRIORITY = [
    "sitting",
    "lying",
    "kneeling",
    "squatting",
    "running",
    "walking",
    "jumping",
    "dancing",
    "standing",
]
VIEW_PRIORITY = [
    "from_front",
    "from_side",
    "from_behind",
    "from_above",
    "from_below",
    "profile",
]


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _posture_label(row: dict[str, Any]) -> str:
    tags = set(row.get("weak_pose_tags", []))
    for tag in POSTURE_PRIORITY:
        if tag in tags:
            return tag
    return "unknown"


def _view_label(row: dict[str, Any]) -> str:
    tags = set(row.get("weak_view_tags", []))
    for tag in VIEW_PRIORITY:
        if tag in tags:
            return tag
    for tag in ("full_body", "upper_body", "portrait"):
        if tag in tags:
            return tag
    return "unknown"


def _label_for(row: dict[str, Any], field: str | None) -> str:
    if field:
        return str(row.get(field) or "unknown")
    return _posture_label(row)


def _best_person(people: list[Any]) -> Any:
    """Pick the person with the strongest top-8 body joints."""

    def score(person: Any) -> float:
        values = np.sort(np.asarray(person.body)[:, 2])
        top = values[-8:] if len(values) >= 8 else values
        return float(top.mean())

    return max(people, key=score)


def _pose_feature(
    body: np.ndarray,
    *,
    threshold: float,
    min_valid_joints: int,
) -> tuple[np.ndarray | None, int, float]:
    """Torso-centric normalized 18-joint feature + visibility mask."""

    body = np.asarray(body, dtype=np.float32)
    xy = body[:, :2]
    confidence = body[:, 2]
    valid = confidence >= threshold
    valid_count = int(valid.sum())
    if valid_count < min_valid_joints:
        return None, valid_count, float(confidence.mean())

    neck = 1
    right_hip, left_hip = 8, 11
    right_shoulder, left_shoulder = 2, 5
    hip_mid = (xy[right_hip] + xy[left_hip]) / 2.0

    if valid[right_hip] and valid[left_hip]:
        center = hip_mid
    elif valid[neck]:
        center = xy[neck]
    else:
        center = xy[valid].mean(axis=0)

    if valid[neck] and valid[right_hip] and valid[left_hip]:
        scale = float(np.linalg.norm(xy[neck] - hip_mid))
    elif valid[right_shoulder] and valid[left_shoulder]:
        scale = float(np.linalg.norm(xy[right_shoulder] - xy[left_shoulder]))
    elif valid[right_hip] and valid[left_hip]:
        scale = float(np.linalg.norm(xy[right_hip] - xy[left_hip]))
    elif valid_count >= 2:
        points = xy[valid]
        scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    else:
        scale = 1.0
    scale = max(scale, 1e-6)

    normalized = (xy - center) / scale
    normalized[~valid] = 0.0
    feature = np.concatenate([normalized.reshape(-1), valid.astype(np.float32)])
    return feature, valid_count, float(confidence[valid].mean())


def _cluster(
    features: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        copy=True,
    )
    labels = model.fit_predict(features)
    info = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_clusters": int(len({label for label in labels if label >= 0})),
        "noise": int((labels == -1).sum()),
    }
    return labels, info


def _cluster_purity(labels: np.ndarray, weak_labels: list[str]) -> float:
    """Size-weighted purity on non-noise images with a known weak label."""

    total = 0
    correct = 0
    for cluster in sorted({int(label) for label in labels if label >= 0}):
        indices = [
            index
            for index, label in enumerate(labels)
            if label == cluster and weak_labels[index] != "unknown"
        ]
        if not indices:
            continue
        counts: dict[str, int] = {}
        for index in indices:
            counts[weak_labels[index]] = counts.get(weak_labels[index], 0) + 1
        total += len(indices)
        correct += max(counts.values())
    return correct / total if total else 0.0


def _write_cluster_sheet(
    out_dir: Path,
    benchmark_dir: Path,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    *,
    cluster: int,
    label_field: str | None,
    thumb: int = 200,
    cols: int = 4,
) -> None:
    indices = [index for index, label in enumerate(labels) if label == cluster]
    if not indices:
        return
    cell = thumb + 30
    grid_rows = math.ceil(len(indices) / cols)
    sheet = Image.new("RGB", (cols * cell, grid_rows * cell), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for position, index in enumerate(indices):
        row = rows[index]
        try:
            with Image.open(benchmark_dir / row["path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb, thumb))
                x = (position % cols) * cell + (cell - image.width) // 2
                y = (position // cols) * cell + 24 + (thumb - image.height) // 2
                sheet.paste(image, (x, y))
        except Exception:  # noqa: BLE001 - a missing image should not break the sheet
            pass
        label = f"{row['image_id']} {_label_for(row, label_field)} {_view_label(row)}"
        draw.text(
            ((position % cols) * cell + 4, (position // cols) * cell + 4),
            label,
            fill=(20, 20, 20),
        )
    name = "noise.png" if cluster == -1 else f"cluster_{cluster:02d}.png"
    sheet.save(out_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("benchmarks/data/anime_miku_small"),
    )
    parser.add_argument(
        "--split",
        default="target",
        help="Manifest split to use, or 'all'.",
    )
    parser.add_argument(
        "--pose-label-field",
        default=None,
        help="Optional manifest field used as the pose label (e.g. pose_label).",
    )
    parser.add_argument("--keypoint-threshold", type=float, default=0.1)
    parser.add_argument("--min-valid-joints", type=int, default=6)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <benchmark>/pose_form_baseline).",
    )
    args = parser.parse_args()

    benchmark_dir: Path = args.benchmark_dir.resolve()
    out_dir = (args.out or (benchmark_dir / "pose_form_baseline")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in _load_manifest(benchmark_dir / "manifest.jsonl")
        if args.split == "all" or row["split"] == args.split
    ]
    print(f"Running DWPose on {len(rows)} {args.split} images ...")

    features: list[np.ndarray] = []
    usable_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    pose_failed = 0

    for index, row in enumerate(rows, start=1):
        feature = None
        valid_joints = 0
        mean_score = 0.0
        error = None
        try:
            people = dwpose_estimate(benchmark_dir / row["path"])
            if people:
                person = _best_person(people)
                feature, valid_joints, mean_score = _pose_feature(
                    person.body,
                    threshold=args.keypoint_threshold,
                    min_valid_joints=args.min_valid_joints,
                )
        except Exception as exc:  # noqa: BLE001 - report per-image failures
            error = str(exc)

        if feature is None:
            pose_failed += 1
            predictions.append(
                {
                    "image_id": row["image_id"],
                    "cluster": -1,
                    "pose_failed": True,
                    "valid_joints": valid_joints,
                    "mean_keypoint_score": mean_score,
                    "posture_label": _label_for(row, args.pose_label_field),
                    "view_label": _view_label(row),
                    "error": error,
                }
            )
        else:
            usable_rows.append(row)
            features.append(feature)
            predictions.append(
                {
                    "image_id": row["image_id"],
                    "cluster": None,
                    "pose_failed": False,
                    "valid_joints": valid_joints,
                    "mean_keypoint_score": mean_score,
                    "posture_label": _label_for(row, args.pose_label_field),
                    "view_label": _view_label(row),
                    "error": None,
                }
            )

        if index % 10 == 0:
            print(f"  {index}/{len(rows)}")

    if len(features) < 2:
        raise SystemExit("Not enough usable poses for clustering.")

    matrix = np.stack(features)
    labels, cluster_info = _cluster(
        matrix,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    if cluster_info["n_clusters"] < 2:
        labels, cluster_info = _cluster(
            matrix,
            min_cluster_size=2,
            min_samples=1,
        )
        cluster_info["retry_with_smaller_clusters"] = True

    label_by_id = {
        row["image_id"]: int(label)
        for row, label in zip(usable_rows, labels, strict=True)
    }
    for prediction in predictions:
        if not prediction["pose_failed"]:
            prediction["cluster"] = label_by_id[prediction["image_id"]]

    posture_labels = [_label_for(row, args.pose_label_field) for row in usable_rows]
    view_labels = [_view_label(row) for row in usable_rows]
    mask = labels >= 0
    known_posture = np.asarray([label != "unknown" for label in posture_labels])
    known_view = np.asarray([label != "unknown" for label in view_labels])

    posture_mask = mask & known_posture
    view_mask = mask & known_view
    posture_ami = (
        float(adjusted_mutual_info_score(
            np.asarray(posture_labels)[posture_mask],
            labels[posture_mask],
        ))
        if posture_mask.sum() > 1
        else None
    )
    posture_nmi = (
        float(normalized_mutual_info_score(
            np.asarray(posture_labels)[posture_mask],
            labels[posture_mask],
        ))
        if posture_mask.sum() > 1
        else None
    )
    view_ami = (
        float(adjusted_mutual_info_score(
            np.asarray(view_labels)[view_mask],
            labels[view_mask],
        ))
        if view_mask.sum() > 1
        else None
    )
    purity = _cluster_purity(labels, posture_labels)

    summary = {
        "benchmark_dir": benchmark_dir.as_posix(),
        "split": args.split,
        "images": len(rows),
        "pose_success": len(rows) - pose_failed,
        "pose_success_rate": (len(rows) - pose_failed) / len(rows),
        "pose_failed": pose_failed,
        "keypoint_threshold": args.keypoint_threshold,
        "min_valid_joints": args.min_valid_joints,
        "clustering": cluster_info,
        "posture_ami": posture_ami,
        "posture_nmi": posture_nmi,
        "posture_purity": purity,
        "view_ami": view_ami,
        "posture_label_counts": {
            label: posture_labels.count(label) for label in sorted(set(posture_labels))
        },
        "view_label_counts": {
            label: view_labels.count(label) for label in sorted(set(view_labels))
        },
        "note": "Weak Danbooru tags are not ground truth; treat metrics as a sanity check.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    for cluster in sorted({int(label) for label in labels}):
        _write_cluster_sheet(
            out_dir,
            benchmark_dir,
            usable_rows,
            labels,
            cluster=cluster,
            label_field=args.pose_label_field,
        )
    _write_cluster_sheet(
        out_dir,
        benchmark_dir,
        usable_rows,
        labels,
        cluster=-1,
        label_field=args.pose_label_field,
    )

    print()
    success = len(rows) - pose_failed
    print(f"Pose success: {success}/{len(rows)} ({success / len(rows):.1%})")
    print(
        f"Clusters: {cluster_info['n_clusters']}, noise: {cluster_info['noise']} "
        f"(min_cluster_size={cluster_info['min_cluster_size']}, "
        f"min_samples={cluster_info['min_samples']})"
    )
    print(f"Posture AMI: {posture_ami if posture_ami is not None else 'n/a'}")
    print(f"Posture purity: {purity:.4f}")
    print(f"View AMI: {view_ami if view_ami is not None else 'n/a'}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
