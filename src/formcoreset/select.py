"""Baseline canonical selection.

The real reference-mode scorer will combine quality, completeness, cleanliness,
representativeness, and fidelity. This baseline is intentionally simple so the
CLI and output pipeline can be tested without model backends.
"""

from __future__ import annotations

from .types import ImageRecord


def score_record(record: ImageRecord) -> float:
    """Return a rough 0..1 canonical score."""

    quality = record.quality
    edge = quality.get("edge_variance", 0.0)
    min_side = quality.get("min_side", 0.0)

    sharpness = min(edge / 50.0, 1.0)
    resolution = min(min_side / 1024.0, 1.0)

    penalty = 0.0
    for flag, weight in {
        "blurry": 0.35,
        "low_resolution": 0.25,
        "overexposed": 0.15,
        "underexposed": 0.15,
        "monochrome": 0.05,
        "text_watermark": 0.25,
    }.items():
        if flag in record.flags:
            penalty += weight

    return max(0.0, 0.6 * sharpness + 0.4 * resolution - penalty)


def pick_canonical(members: list[ImageRecord]) -> ImageRecord | None:
    if not members:
        return None
    return max(members, key=lambda record: (score_record(record), record.pixels))
