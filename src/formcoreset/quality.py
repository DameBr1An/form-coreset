"""Cheap, model-free image quality heuristics.

These heuristics are intentionally conservative. The ML stages add richer
signals (pose confidence, segmentation quality, watermark detection). A flag
never deletes an image by itself.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .types import ImageRecord

LOW_RESOLUTION_MIN_SIDE = 256
BLUR_EDGE_VARIANCE = 12.0


def assess_record(record: ImageRecord) -> dict[str, float]:
    """Compute rough quality metrics and append flags to ``record``."""

    try:
        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if max(image.size) > 768:
                image.thumbnail((768, 768))
            gray = image.convert("L")
            array = np.asarray(gray, dtype=np.float32)
            edge_array = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
            rgb = np.asarray(image, dtype=np.float32)
            mean = float(array.mean())
            std = float(array.std())
            edge_variance = float(edge_array.var())
            channel_spread = float(np.abs(rgb - rgb.mean(axis=2, keepdims=True)).mean())
    except OSError:
        record.flags.append("quality_failed")
        return {}

    scores = {
        "width": float(record.width),
        "height": float(record.height),
        "min_side": float(min(record.width, record.height)),
        "mean_luma": mean,
        "std_luma": std,
        "edge_variance": edge_variance,
        "channel_spread": channel_spread,
    }
    record.quality.update(scores)

    if min(record.width, record.height) < LOW_RESOLUTION_MIN_SIDE:
        record.flags.append("low_resolution")
    if edge_variance < BLUR_EDGE_VARIANCE:
        record.flags.append("blurry")
    if mean > 245:
        record.flags.append("overexposed")
    elif mean < 10:
        record.flags.append("underexposed")
    if channel_spread < 1.0:
        record.flags.append("monochrome")

    return scores
