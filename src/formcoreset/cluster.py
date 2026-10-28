"""Baseline perceptual hashing and grouping.

This module is a scaffold and smoke-test baseline. Real form clustering will
use masked subject embeddings and optional pose features behind the
``FormEncoder`` / ``Clusterer`` interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import HDBSCAN

from .types import ImageRecord


def dhash(path: Path, hash_size: int = 8) -> str:
    """Return a difference hash as a hex string."""

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.int16)

    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def compute_hashes(records: list[ImageRecord]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for record in records:
        try:
            hashes[record.image_id] = dhash(record.path)
        except OSError:
            record.flags.append("hash_failed")
    return hashes


def group_near_duplicates(
    records: list[ImageRecord],
    hashes: dict[str, str],
    threshold: int = 5,
) -> list[list[ImageRecord]]:
    """Greedy connected-component grouping by Hamming distance.

    Deterministic: seeds and candidates are processed in sorted image-id order.
    This is a baseline, not the final form-clustering algorithm.
    """

    by_id = {record.image_id: record for record in records if record.image_id in hashes}
    remaining = set(by_id)
    groups: list[list[ImageRecord]] = []

    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        group = {seed}
        changed = True
        while changed:
            changed = False
            for candidate in sorted(remaining):
                if any(
                    hamming_distance(hashes[candidate], hashes[member]) <= threshold
                    for member in group
                ):
                    remaining.remove(candidate)
                    group.add(candidate)
                    changed = True
        groups.append([by_id[image_id] for image_id in sorted(group)])

    return groups


@dataclass
class FormClusterResult:
    """HDBSCAN labels and metadata for form features."""

    labels: np.ndarray
    method: str
    params: dict[str, Any]
    n_clusters: int
    noise: int


def cluster_features(
    features: np.ndarray,
    *,
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> FormClusterResult:
    """Cluster L2-normalized form features with HDBSCAN.

    Falls back to smaller clusters if the first pass yields fewer than two
    clusters.
    """

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2D array")
    if matrix.shape[0] < 2:
        raise ValueError("at least two feature rows are required")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-12, None)

    def _fit(size: int, samples: int) -> np.ndarray:
        model = HDBSCAN(
            min_cluster_size=size,
            min_samples=samples,
            metric="euclidean",
            copy=True,
        )
        return model.fit_predict(matrix)

    labels = _fit(min_cluster_size, min_samples)
    used_size, used_samples = min_cluster_size, min_samples
    n_clusters = len({int(label) for label in labels if label >= 0})
    if n_clusters < 2 and (min_cluster_size > 2 or min_samples > 1):
        used_size, used_samples = 2, 1
        labels = _fit(used_size, used_samples)
        n_clusters = len({int(label) for label in labels if label >= 0})

    return FormClusterResult(
        labels=labels,
        method="hdbscan",
        params={
            "min_cluster_size": used_size,
            "min_samples": used_samples,
        },
        n_clusters=n_clusters,
        noise=int((labels == -1).sum()),
    )
