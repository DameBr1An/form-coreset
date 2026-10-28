"""Anime hybrid form encoder.

Why hybrid:
- keypoint models (DWPose / RTMPose HumanArt) are unreliable on stylized anime;
- WD14 tags are a strong anime-specific posture/view signal;
- a masked image embedding adds appearance/geometry context.

The default feature is 70% WD14 tag vector + 30% CCIP embedding. CCIP is an
identity model, so it is only an auxiliary signal. Keypoints are intentionally
not part of the default feature; they can be added later behind a confidence
gate.

Dependencies are imported lazily so the core package (and tests) do not require
the anime extra.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..config import FormConfig
from ..types import ImageRecord
from .base import FormFeatures
from .vocab import TAG_VOCAB

logger = logging.getLogger(__name__)


class FormEncodingError(RuntimeError):
    """Raised when an image cannot be encoded."""


def _get_wd14_tags(path: Any, **kwargs: Any) -> tuple[Any, dict[str, float], Any]:
    from imgutils.tagging import get_wd14_tags

    return get_wd14_tags(path, **kwargs)


def _ccip_extract_feature(path: Any) -> np.ndarray:
    from imgutils.metrics import ccip_extract_feature

    return np.asarray(ccip_extract_feature(path), dtype=np.float32).reshape(-1)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def build_tag_vector(
    tags: dict[str, float],
    threshold: float,
    vocab: Sequence[str] = TAG_VOCAB,
) -> np.ndarray:
    """Confidence vector over the form vocabulary, with a soft threshold."""

    return np.asarray(
        [max(0.0, float(tags.get(tag, 0.0)) - threshold) for tag in vocab],
        dtype=np.float32,
    )


class AnimeHybridFormEncoder:
    """WD14 tags + CCIP embedding form encoder for anime/stylized art."""

    name = "anime-hybrid"
    version = "anime-hybrid-v1"

    def __init__(self, config: FormConfig | None = None) -> None:
        self.config = config or FormConfig()
        if self.config.tag_weight < 0 or self.config.embedding_weight < 0:
            raise ValueError("tag_weight and embedding_weight must be >= 0")
        if self.config.tag_weight + self.config.embedding_weight <= 0:
            raise ValueError("At least one form feature weight must be positive")

    def encode(self, record: ImageRecord) -> FormFeatures:
        try:
            rating, general, characters = _get_wd14_tags(
                record.path,
                general_threshold=self.config.general_threshold,
            )
        except Exception as exc:  # noqa: BLE001 - include image context
            raise FormEncodingError(f"WD14 failed for {record.path}: {exc}") from exc

        tag_vector = build_tag_vector(general, self.config.tag_threshold)
        tag_component = l2_normalize(tag_vector)

        embedding_used = False
        if self.config.embedding_weight > 0:
            try:
                embedding = _ccip_extract_feature(record.path)
                embedding_component = l2_normalize(embedding)
                total = self.config.tag_weight + self.config.embedding_weight
                feature = np.concatenate(
                    [
                        tag_component * (self.config.tag_weight / total),
                        embedding_component * (self.config.embedding_weight / total),
                    ]
                )
                embedding_used = True
            except Exception as exc:  # noqa: BLE001 - fall back to tags
                logger.warning("CCIP embedding failed for %s; tags only: %s", record.path, exc)
                feature = tag_component
        else:
            feature = tag_component

        top_tags = sorted(
            (
                (tag, float(confidence))
                for tag, confidence in general.items()
                if tag in TAG_VOCAB
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:5]

        return FormFeatures(
            embedding=np.asarray(feature, dtype=np.float32),
            confidence=float(np.linalg.norm(tag_vector)),
            method=self.version,
            tags=general,
            tag_vector=tag_vector,
            metadata={
                "encoder": self.version,
                "embedding_used": embedding_used,
                "top_tags": [
                    {"tag": tag, "confidence": round(confidence, 4)}
                    for tag, confidence in top_tags
                ],
                "rating": rating,
                "characters": characters,
            },
        )
