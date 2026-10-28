"""Same-source variant detection and grouping.

This is the first-pass deduplication path for posters/artwork variants:
different crops, scales, compression, backgrounds, or effects of the **same
source image**. It uses local feature matching + RANSAC homography, so it does
not need pose labels or a pose vocabulary.

OpenCV is imported lazily so the core CLI does not require it unless same-source
deduplication is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as hierarchy_linkage
from scipy.spatial.distance import squareform

from .config import SameSourceConfig
from .select import pick_canonical
from .types import ImageRecord

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    score: float
    inliers: int
    matches: int
    homography: np.ndarray | None
    correlation: float | None = None


@dataclass
class SourceGroup:
    source_id: str
    members: list[ImageRecord]
    canonical_id: str | None
    min_score: float
    method: str


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency hint
        raise ImportError(
            "OpenCV is required for same-source deduplication. "
            "Install it with: uv sync --extra anime"
        ) from exc
    return cv2


def _detector(cv2: Any, method: str, max_features: int) -> Any:
    if method == "orb":
        return cv2.ORB_create(nfeatures=max_features)
    if not hasattr(cv2, "SIFT_create"):
        return cv2.ORB_create(nfeatures=max_features)
    return cv2.SIFT_create(nfeatures=max_features)


def _mask_for(
    path: Path,
    scale: int,
    provider_name: str = "auto",
) -> np.ndarray | None:
    """Subject mask via the pluggable MaskProvider interface."""

    from .mask import get_mask_provider

    return get_mask_provider(provider_name, scale=scale).mask(path)


def _descriptors(
    path: Path,
    method: str,
    max_features: int,
    *,
    use_mask: bool = True,
    mask_scale: int = 1024,
    mask_provider: str = "auto",
    mask: np.ndarray | None = None,
) -> tuple[Any | None, Any | None]:
    cv2 = _cv2()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    if mask is None and use_mask:
        mask = _mask_for(path, mask_scale, mask_provider)
    keypoints, descriptors = _detector(cv2, method, max_features).detectAndCompute(
        image, mask
    )
    if descriptors is None or len(keypoints) < 4:
        return None, None
    return keypoints, descriptors


def _edge_map(gray: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.GaussianBlur(magnitude, (5, 5), 0)


def verify_homography(
    path_a: Path,
    path_b: Path,
    homography: np.ndarray,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
) -> float:
    """Correlate aligned edge maps to reject false homographies.

    SIFT can match similar characters across different drawings. If the
    homography is real, the aligned edges should correlate; if it is a
    coincidental character/face match, they should not.
    """

    cv2 = _cv2()
    gray_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    gray_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if gray_a is None or gray_b is None:
        return 0.0
    height_b, width_b = gray_b.shape[:2]
    warped_a = cv2.warpPerspective(
        gray_a,
        homography,
        (width_b, height_b),
        flags=cv2.INTER_LINEAR,
    )
    valid = cv2.warpPerspective(
        np.full(gray_a.shape[:2], 255, dtype=np.uint8),
        homography,
        (width_b, height_b),
        flags=cv2.INTER_NEAREST,
    )
    overlap = valid > 0
    if mask_a is not None:
        warped_mask_a = cv2.warpPerspective(
            mask_a,
            homography,
            (width_b, height_b),
            flags=cv2.INTER_NEAREST,
        )
        overlap &= warped_mask_a > 0
    if mask_b is not None:
        overlap &= mask_b > 0
    if int(overlap.sum()) < 100:
        return 0.0

    edges_a = _edge_map(warped_a)[overlap]
    edges_b = _edge_map(gray_b)[overlap]
    if float(edges_a.std()) < 1e-6 or float(edges_b.std()) < 1e-6:
        return 0.0
    correlation = float(np.corrcoef(edges_a, edges_b)[0, 1])
    return correlation if np.isfinite(correlation) else 0.0


def match_descriptors(
    keypoints_a: Any,
    descriptors_a: Any,
    keypoints_b: Any,
    descriptors_b: Any,
    *,
    method: str = "sift",
    ratio: float = 0.75,
    ransac_threshold: float = 4.0,
) -> MatchResult:
    cv2 = _cv2()
    if descriptors_a is None or descriptors_b is None:
        return MatchResult(0.0, 0, 0, None)

    norm = cv2.NORM_HAMMING if method == "orb" else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    try:
        knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    except cv2.error:
        return MatchResult(0.0, 0, 0, None)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good.append(first)

    if len(good) < 4:
        return MatchResult(0.0, len(good), len(good), None)

    source_points = np.float32(
        [keypoints_a[match.queryIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    target_points = np.float32(
        [keypoints_b[match.trainIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    try:
        homography, mask = cv2.findHomography(
            source_points,
            target_points,
            cv2.RANSAC,
            ransac_threshold,
        )
    except cv2.error:
        return MatchResult(0.0, len(good), len(good), None)

    inliers = int(mask.sum()) if mask is not None else 0
    denominator = max(min(len(keypoints_a), len(keypoints_b)), 1)
    return MatchResult(
        score=inliers / denominator,
        inliers=inliers,
        matches=len(good),
        homography=homography,
    )


def match_pair(
    path_a: Path,
    path_b: Path,
    config: SameSourceConfig | None = None,
) -> MatchResult:
    config = config or SameSourceConfig()
    mask_a = (
        _mask_for(path_a, config.mask_scale, config.mask_provider)
        if config.use_mask
        else None
    )
    mask_b = (
        _mask_for(path_b, config.mask_scale, config.mask_provider)
        if config.use_mask
        else None
    )
    keypoints_a, descriptors_a = _descriptors(
        path_a,
        config.method,
        config.max_features,
        use_mask=config.use_mask,
        mask_scale=config.mask_scale,
        mask=mask_a,
    )
    keypoints_b, descriptors_b = _descriptors(
        path_b,
        config.method,
        config.max_features,
        use_mask=config.use_mask,
        mask_scale=config.mask_scale,
        mask=mask_b,
    )
    result = match_descriptors(
        keypoints_a,
        descriptors_a,
        keypoints_b,
        descriptors_b,
        method=config.method,
        ratio=config.ratio,
    )
    if config.verify and result.homography is not None:
        result.correlation = verify_homography(
            path_a,
            path_b,
            result.homography,
            mask_a,
            mask_b,
        )
        if result.correlation < config.min_correlation:
            result.score = 0.0
    return result


def _find(parent: dict[Any, Any], item: Any) -> Any:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union(parent: dict[Any, Any], left: Any, right: Any) -> None:
    root_left = _find(parent, left)
    root_right = _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def partition_by_scores(
    image_ids: list[str],
    scores: np.ndarray,
    *,
    linkage_method: str = "complete",
    threshold: float = 0.3,
) -> list[list[str]]:
    """Partition items from a pairwise same-source confidence matrix.

    ``single`` is the old connected-components behavior and can chain
    A~B, B~C even when A and C do not match. ``complete`` requires every
    cross-cluster pair to be above the threshold, which is the default.
    """

    if linkage_method not in {"complete", "average", "single"}:
        raise ValueError(f"Unsupported linkage method: {linkage_method}")
    if not image_ids:
        return []
    count = len(image_ids)
    if count == 1:
        return [[image_ids[0]]]

    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.shape != (count, count):
        raise ValueError("score matrix shape does not match image_ids")
    matrix = np.clip(matrix, 0.0, 1.0)

    if linkage_method == "single":
        parent = {index: index for index in range(count)}
        for left in range(count):
            for right in range(left + 1, count):
                if matrix[left, right] >= threshold:
                    _union(parent, left, right)
        grouped: dict[int, list[str]] = {}
        for index in range(count):
            grouped.setdefault(_find(parent, index), []).append(image_ids[index])
        groups = list(grouped.values())
    else:
        distance = 1.0 - matrix
        np.fill_diagonal(distance, 0.0)
        condensed = squareform(distance, checks=False)
        tree = hierarchy_linkage(condensed, method=linkage_method)
        labels = fcluster(tree, t=1.0 - threshold, criterion="distance")
        grouped = {}
        for index, label in enumerate(labels):
            grouped.setdefault(int(label), []).append(image_ids[index])
        groups = list(grouped.values())

    order = {image_id: index for index, image_id in enumerate(image_ids)}
    groups.sort(key=lambda ids: min(order[image_id] for image_id in ids))
    return groups


def _medoid(
    member_ids: list[str],
    scores: np.ndarray,
    index_by_id: dict[str, int],
) -> str:
    best_id = member_ids[0]
    best_mean = -1.0
    for candidate in member_ids:
        candidate_index = index_by_id[candidate]
        others = [
            index_by_id[other]
            for other in member_ids
            if other != candidate
        ]
        if not others:
            mean = 0.0
        else:
            mean = sum(
                float(scores[candidate_index, other_index])
                for other_index in others
            ) / len(others)
        if mean > best_mean:
            best_mean = mean
            best_id = candidate
    return best_id


def star_merge_partitions(
    partitions: list[list[str]],
    image_ids: list[str],
    scores: np.ndarray,
    *,
    threshold: float = 0.2,
) -> list[list[str]]:
    """Reattach high-confidence singletons to a group medoid.

    Complete linkage is precision-first and may split a weak variant into a
    singleton. This pass only allows singleton -> group merges, never
    group -> group merges, so it cannot create a multi-group chain. The medoid
    is computed from the seed groups and kept fixed for the pass.
    """

    if not partitions:
        return []
    index_by_id = {
        image_id: index for index, image_id in enumerate(image_ids)
    }
    multi = [list(group) for group in partitions if len(group) >= 2]
    singletons = [group[0] for group in partitions if len(group) == 1]
    if not multi:
        return [list(group) for group in partitions]

    medoids = [_medoid(group, scores, index_by_id) for group in multi]
    merged = [list(group) for group in multi]
    leftover: list[list[str]] = []

    for singleton in singletons:
        singleton_index = index_by_id[singleton]
        best_group: int | None = None
        best_score = -1.0
        for group_index, medoid in enumerate(medoids):
            score = float(scores[singleton_index, index_by_id[medoid]])
            if score > best_score:
                best_score = score
                best_group = group_index
        if best_group is not None and best_score >= threshold:
            merged[best_group].append(singleton)
        else:
            leftover.append([singleton])

    groups = merged + leftover
    order = {image_id: index for index, image_id in enumerate(image_ids)}
    groups.sort(key=lambda ids: min(order[image_id] for image_id in ids))
    return groups


def group_same_source(
    records: list[ImageRecord],
    config: SameSourceConfig | None = None,
    *,
    progress: bool = False,
) -> list[SourceGroup]:
    """Group records by same-source local-feature matches."""

    config = config or SameSourceConfig()
    descriptors: dict[str, tuple[Any, Any]] = {}
    masks: dict[str, np.ndarray | None] = {}
    records_by_id = {record.image_id: record for record in records}

    for index, record in enumerate(records, 1):
        mask = (
            _mask_for(record.path, config.mask_scale, config.mask_provider)
            if config.use_mask
            else None
        )
        masks[record.image_id] = mask
        keypoints, descriptor = _descriptors(
            record.path,
            config.method,
            config.max_features,
            use_mask=config.use_mask,
            mask_scale=config.mask_scale,
            mask=mask,
        )
        if keypoints is not None:
            descriptors[record.image_id] = (keypoints, descriptor)
        if progress and index % 10 == 0:
            print(f"  descriptors {index}/{len(records)}")

    ordered_ids = [record.image_id for record in records]
    index_by_id = {
        image_id: index for index, image_id in enumerate(ordered_ids)
    }
    candidate_threshold = (
        min(config.min_score, config.star_threshold)
        if config.star_merge
        else config.min_score
    )
    score_matrix = np.zeros(
        (len(ordered_ids), len(ordered_ids)),
        dtype=np.float32,
    )
    descriptor_ids = sorted(descriptors)
    total_pairs = len(descriptor_ids) * (len(descriptor_ids) - 1) // 2
    done_pairs = 0

    for left_index, left_id in enumerate(descriptor_ids):
        for right_id in descriptor_ids[left_index + 1 :]:
            left_keypoints, left_descriptor = descriptors[left_id]
            right_keypoints, right_descriptor = descriptors[right_id]
            result = match_descriptors(
                left_keypoints,
                left_descriptor,
                right_keypoints,
                right_descriptor,
                method=config.method,
                ratio=config.ratio,
            )
            if result.score >= candidate_threshold and result.inliers >= config.min_inliers:
                correlation = None
                if config.verify and result.homography is not None:
                    correlation = verify_homography(
                        records_by_id[left_id].path,
                        records_by_id[right_id].path,
                        result.homography,
                        masks.get(left_id),
                        masks.get(right_id),
                    )
                    if correlation < config.min_correlation:
                        continue
                confidence = (
                    result.score
                    if correlation is None
                    else min(result.score, correlation)
                )
                left_pos = index_by_id[left_id]
                right_pos = index_by_id[right_id]
                score_matrix[left_pos, right_pos] = confidence
                score_matrix[right_pos, left_pos] = confidence
            done_pairs += 1
            if progress and done_pairs % 200 == 0:
                print(f"  pairs {done_pairs}/{total_pairs}")

    partitions = partition_by_scores(
        ordered_ids,
        score_matrix,
        linkage_method=config.linkage,
        threshold=config.min_score,
    )
    if config.star_merge:
        partitions = star_merge_partitions(
            partitions,
            ordered_ids,
            score_matrix,
            threshold=min(config.min_score, config.star_threshold),
        )

    groups: list[SourceGroup] = []
    for index, member_ids in enumerate(partitions, start=1):
        members = [records_by_id[image_id] for image_id in member_ids]
        members.sort(key=lambda record: record.image_id)
        canonical = pick_canonical(members)
        member_indices = [index_by_id[image_id] for image_id in member_ids]
        group_scores = [
            float(score_matrix[left, right])
            for left in member_indices
            for right in member_indices
            if left < right and score_matrix[left, right] > 0
        ]
        groups.append(
            SourceGroup(
                source_id=f"source_{index:03d}",
                members=members,
                canonical_id=canonical.image_id if canonical else None,
                min_score=min(group_scores) if group_scores else 0.0,
                method=config.method,
            )
        )
    return groups
