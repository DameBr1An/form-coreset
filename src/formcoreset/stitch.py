"""Canonical stitching for same-source variants.

If variants preserve different regions (e.g., left/right crops), align them to
the canonical reference with homographies and blend the union into a larger,
more complete image. The stitched image is used only when it meaningfully
increases the covered area; otherwise the single canonical is kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import SameSourceConfig
from .source import _cv2, _descriptors, _mask_for, match_descriptors
from .types import ImageRecord


@dataclass
class StitchResult:
    image: Image.Image
    used_members: list[str]
    gain: float
    used: bool


def _load_rgb(path: Path) -> np.ndarray | None:
    cv2 = _cv2()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _corners(image: np.ndarray, homography: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    height, width = image.shape[:2]
    points = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def stitch_group(
    members: list[ImageRecord],
    reference: ImageRecord,
    config: SameSourceConfig,
) -> StitchResult | None:
    """Stitch same-source members into a larger canonical image."""

    cv2 = _cv2()
    reference_image = _load_rgb(reference.path)
    if reference_image is None:
        return None

    reference_keypoints, reference_descriptors = _descriptors(
        reference.path,
        config.method,
        config.max_features,
        use_mask=config.use_mask,
        mask_scale=config.mask_scale,
    )
    if reference_descriptors is None:
        return None

    reference_mask = (
        _mask_for(reference.path, config.mask_scale, config.mask_provider)
        if config.use_mask
        else None
    )
    items: list[tuple[ImageRecord, np.ndarray, np.ndarray, np.ndarray | None, float]] = [
        (reference, reference_image, np.eye(3, dtype=np.float64), reference_mask, 1.0)
    ]

    for member in members:
        if member.image_id == reference.image_id:
            continue
        keypoints, descriptors = _descriptors(
            member.path,
            config.method,
            config.max_features,
            use_mask=config.use_mask,
            mask_scale=config.mask_scale,
        )
        if descriptors is None:
            continue
        match = match_descriptors(
            keypoints,
            descriptors,
            reference_keypoints,
            reference_descriptors,
            method=config.method,
            ratio=config.ratio,
        )
        if (
            match.homography is None
            or match.score < config.min_score
            or match.inliers < config.min_inliers
        ):
            continue
        image = _load_rgb(member.path)
        if image is None:
            continue
        mask = (
            _mask_for(member.path, config.mask_scale, config.mask_provider)
            if config.use_mask
            else None
        )
        items.append((member, image, match.homography, mask, match.score))

    if len(items) < 2:
        return None

    all_corners = np.concatenate(
        [_corners(image, homography) for _, image, homography, _, _ in items],
        axis=0,
    )
    min_xy = np.floor(all_corners.min(axis=0)).astype(int) - 4
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int) + 4
    canvas_width = max(1, int(max_xy[0] - min_xy[0]))
    canvas_height = max(1, int(max_xy[1] - min_xy[1]))
    translation = np.asarray(
        [[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]],
        dtype=np.float64,
    )

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weights = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    used_members: list[str] = []

    for member, image, homography, mask, score in items:
        canvas_homography = translation @ homography
        warped = cv2.warpPerspective(
            image,
            canvas_homography,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
        )
        height, width = image.shape[:2]
        valid = cv2.warpPerspective(
            np.full((height, width), 255, dtype=np.uint8),
            canvas_homography,
            (canvas_width, canvas_height),
            flags=cv2.INTER_NEAREST,
        )
        if mask is not None:
            warped_mask = cv2.warpPerspective(
                mask,
                canvas_homography,
                (canvas_width, canvas_height),
                flags=cv2.INTER_NEAREST,
            )
            weight = ((valid > 0) & (warped_mask > 0)).astype(np.uint8) * 255
        else:
            weight = (valid > 0).astype(np.uint8) * 255

        if not weight.any():
            continue
        distance = cv2.distanceTransform(weight, cv2.DIST_L2, 5)
        distance = distance / max(float(distance.max()), 1.0)
        weight_float = (weight > 0).astype(np.float32) * (0.25 + 0.75 * distance)
        weight_float *= max(score, 0.05)
        canvas += warped.astype(np.float32) * weight_float[..., None]
        weights += weight_float
        used_members.append(member.image_id)

    if len(used_members) < 2:
        return None

    rgb = canvas / np.clip(weights[..., None], 1e-6, None)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = (weights > 0).astype(np.uint8) * 255
    content = alpha > 0
    if not content.any():
        return None
    rows = np.any(content, axis=1)
    cols = np.any(content, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    rgb = rgb[top : bottom + 1, left : right + 1]
    alpha = alpha[top : bottom + 1, left : right + 1]

    if config.use_mask and reference_mask is not None:
        reference_area = int((reference_mask > 0).sum())
    else:
        reference_area = reference_image.shape[0] * reference_image.shape[1]
    stitched_area = int((alpha > 0).sum())
    gain = stitched_area / max(reference_area, 1) - 1.0
    used = gain >= config.stitch_min_gain

    if config.use_mask:
        image_out = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
    else:
        image_out = Image.fromarray(rgb, mode="RGB")
    return StitchResult(
        image=image_out,
        used_members=used_members,
        gain=gain,
        used=used,
    )
