#!/usr/bin/env python3
"""CCIP identity baseline for the small anime benchmark.

This is a baseline, not a product result:
- 3 target images are used as references;
- the remaining target images + all distractor images are evaluated;
- CCIP embeddings are compared to the reference centroid;
- AUC and 5-fold cross-validated threshold F1 are reported.

Usage:
    uv sync --extra anime
    uv run python benchmarks/run_ccip_baseline.py \
        --benchmark-dir benchmarks/data/anime_miku_small
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from imgutils.metrics import ccip_batch_extract_features
except ImportError as exc:  # pragma: no cover - runtime dependency hint
    raise SystemExit(
        "dghs-imgutils is required. Run: uv sync --extra anime"
    ) from exc


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _choose_references(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Prefer full-body, larger target images as references."""

    candidates = [row for row in rows if row["split"] == "target"]
    candidates.sort(
        key=lambda row: (
            "full_body" in row.get("weak_view_tags", []),
            min(row["width"], row["height"]),
        ),
        reverse=True,
    )
    if len(candidates) < count:
        raise ValueError(f"Not enough target images for {count} references")
    return candidates[:count]


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_threshold = float(scores.min())
    best_f1 = -1.0
    for threshold in np.unique(scores):
        predictions = (scores >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("benchmarks/data/anime_miku_small"),
    )
    parser.add_argument("--num-references", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <benchmark>/ccip_baseline).",
    )
    args = parser.parse_args()

    benchmark_dir: Path = args.benchmark_dir.resolve()
    out_dir = (args.out or (benchmark_dir / "ccip_baseline")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest(benchmark_dir / "manifest.jsonl")
    references = _choose_references(rows, args.num_references)
    reference_ids = {row["image_id"] for row in references}
    eval_rows = [row for row in rows if row["image_id"] not in reference_ids]

    paths = [benchmark_dir / row["path"] for row in references]
    paths.extend(benchmark_dir / row["path"] for row in eval_rows)

    print(f"Extracting CCIP features for {len(paths)} images ...")
    features = ccip_batch_extract_features(paths)
    features = features.astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(norms, 1e-12, None)

    reference_features = features[: len(references)]
    eval_features = features[len(references) :]
    centroid = reference_features.mean(axis=0)
    centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)

    scores = eval_features @ centroid
    labels = np.asarray([1 if row["split"] == "target" else 0 for row in eval_rows])

    auc = float(roc_auc_score(labels, scores))
    oracle_threshold, oracle_f1 = _best_threshold(scores, labels)

    # 5-fold cross-validated threshold selection: honest-ish generalization.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_thresholds: list[float] = []
    cv_predictions = np.zeros_like(labels)
    for train_index, test_index in folds.split(scores, labels):
        threshold, _ = _best_threshold(scores[train_index], labels[train_index])
        fold_thresholds.append(threshold)
        cv_predictions[test_index] = (scores[test_index] >= threshold).astype(int)

    cv_precision, cv_recall, cv_f1, _ = precision_recall_fscore_support(
        labels,
        cv_predictions,
        average="binary",
        zero_division=0,
    )

    predictions = []
    for row, score, prediction in zip(eval_rows, scores, cv_predictions, strict=True):
        predictions.append(
            {
                "image_id": row["image_id"],
                "split": row["split"],
                "score": float(score),
                "predicted_target": bool(prediction),
            }
        )
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    summary = {
        "benchmark_dir": benchmark_dir.as_posix(),
        "model": "ccip-caformer-24-randaug-pruned",
        "references": [
            {
                "image_id": row["image_id"],
                "path": row["path"],
                "weak_view_tags": row.get("weak_view_tags", []),
            }
            for row in references
        ],
        "n_eval": len(eval_rows),
        "n_target_eval": int(labels.sum()),
        "n_distractor_eval": int((labels == 0).sum()),
        "auc": auc,
        "oracle_threshold": oracle_threshold,
        "oracle_f1": oracle_f1,
        "cv_f1": float(cv_f1),
        "cv_precision": float(cv_precision),
        "cv_recall": float(cv_recall),
        "cv_thresholds": fold_thresholds,
        "score_mean_target": float(scores[labels == 1].mean()),
        "score_mean_distractor": float(scores[labels == 0].mean()),
        "note": "3 target references; 5-fold CV threshold F1. Not a product benchmark.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"AUC: {auc:.4f}")
    print(f"5-fold CV F1: {cv_f1:.4f} (P={cv_precision:.4f}, R={cv_recall:.4f})")
    print(f"Oracle F1: {oracle_f1:.4f} @ threshold={oracle_threshold:.4f}")
    print(
        f"Mean score: target={scores[labels == 1].mean():.4f}, "
        f"distractor={scores[labels == 0].mean():.4f}"
    )
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
