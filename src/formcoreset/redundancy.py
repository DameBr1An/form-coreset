"""Form-caption joint redundancy for training-set curation.

The metric answers three separate questions:

1. Is the *form* repeated? (pose, angle, state, shape)
2. Is the *caption* repeated? (semantic content)
3. Is this pair redundant *relative to a set already selected*?

Raw similarities are cosine similarities clamped to [0, 1]. They are then
thresholded and combined with a soft-AND family:

    form_term    = ft * (form_floor + (1 - form_floor) * ct)
    caption_term = ct * (caption_floor + (1 - caption_floor) * ft)
    joint        = max(form_term, caption_term)

where ``ft`` and ``ct`` are thresholded form/caption similarities.

Interpretation:

- ``form_floor=0, caption_floor=0``: pure AND. A pair is redundant only if
  both form and caption repeat.
- ``form_floor=0.3`` (default): repeated form alone still carries some
  redundancy, even when the caption differs.
- ``form_floor=1``: form-only redundancy, caption ignored.
- ``caption_floor`` plays the symmetric role for repeated captions with
  different forms.
- When captions are absent, the metric degenerates to form-only redundancy.

This module deliberately does not depend on a particular embedding model. The
caller provides form embeddings and caption embeddings (or caption tag sets),
and the metric computes similarities. That keeps the metric testable and
backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

DominantAxis = Literal["form", "caption", "joint", "none"]


@dataclass(frozen=True)
class JointRedundancyConfig:
    """Parameters for the joint redundancy metric.

    Thresholds should be calibrated on a labeled validation set. The defaults
    are conservative starting points, not universal constants.
    """

    form_threshold: float = 0.80
    caption_threshold: float = 0.75
    form_floor: float = 0.30
    caption_floor: float = 0.10
    top_k: int = 1
    quality_weight: float = 1.0
    form_novelty_weight: float = 0.5
    caption_novelty_weight: float = 0.5
    redundancy_weight: float = 0.70

    def __post_init__(self) -> None:
        for name in (
            "form_threshold",
            "caption_threshold",
            "form_floor",
            "caption_floor",
            "redundancy_weight",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.quality_weight < 0:
            raise ValueError("quality_weight must be >= 0")
        if self.form_novelty_weight + self.caption_novelty_weight <= 0:
            raise ValueError("novelty weights must sum to a positive value")


@dataclass(frozen=True)
class RedundancyBreakdown:
    """Full breakdown of redundancy between one pair (or one pair and a set)."""

    form_similarity: float
    caption_similarity: float
    form_redundancy: float
    caption_redundancy: float
    joint_redundancy: float
    dominant: DominantAxis


@dataclass(frozen=True)
class MarginalInfo:
    """How much a candidate adds relative to an already-selected set."""

    form_novelty: float
    caption_novelty: float
    novelty: float
    redundancy: RedundancyBreakdown
    score: float


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def thresholded_similarity(similarity: float, threshold: float) -> float:
    """Map similarity above ``threshold`` to [0, 1]; below it becomes 0."""

    similarity = _clip01(similarity)
    threshold = _clip01(threshold)
    if threshold >= 1.0:
        return 1.0 if similarity >= 1.0 else 0.0
    return _clip01((similarity - threshold) / (1.0 - threshold))


def pair_redundancy(
    form_similarity: float,
    caption_similarity: float | None,
    config: JointRedundancyConfig | None = None,
) -> RedundancyBreakdown:
    """Compute joint redundancy for one pair.

    Pass ``caption_similarity=None`` when captions are absent; the metric then
    becomes form-only redundancy.
    """

    config = config or JointRedundancyConfig()
    form = _clip01(form_similarity)
    form_redundancy = thresholded_similarity(form, config.form_threshold)

    if caption_similarity is None:
        return RedundancyBreakdown(
            form_similarity=form,
            caption_similarity=0.0,
            form_redundancy=form_redundancy,
            caption_redundancy=0.0,
            joint_redundancy=form_redundancy,
            dominant="form" if form_redundancy > 0.0 else "none",
        )

    caption = _clip01(caption_similarity)
    caption_redundancy = thresholded_similarity(caption, config.caption_threshold)

    form_term = form_redundancy * (
        config.form_floor + (1.0 - config.form_floor) * caption_redundancy
    )
    caption_term = caption_redundancy * (
        config.caption_floor + (1.0 - config.caption_floor) * form_redundancy
    )
    joint = _clip01(max(form_term, caption_term))

    if max(form_redundancy, caption_redundancy) < 0.05:
        dominant: DominantAxis = "none"
    elif abs(form_redundancy - caption_redundancy) <= 0.15:
        dominant = "joint"
    elif form_redundancy > caption_redundancy:
        dominant = "form"
    else:
        dominant = "caption"

    return RedundancyBreakdown(
        form_similarity=form,
        caption_similarity=caption,
        form_redundancy=form_redundancy,
        caption_redundancy=caption_redundancy,
        joint_redundancy=joint,
        dominant=dominant,
    )


def normalize_embeddings(embeddings: ArrayLike) -> NDArray[np.float32]:
    """L2-normalize a 1D or 2D array; zero vectors stay zero."""

    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("embeddings must be a 1D or 2D array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def cosine_similarity_matrix(
    left: ArrayLike,
    right: ArrayLike | None = None,
) -> NDArray[np.float32]:
    """Cosine similarity, clamped to [0, 1].

    Negative similarity is treated as no redundancy. This is deliberate: the
    metric is about repetition, not semantic opposition.
    """

    left_normalized = normalize_embeddings(left)
    right_normalized = left_normalized if right is None else normalize_embeddings(right)
    return np.clip(left_normalized @ right_normalized.T, 0.0, 1.0)


def _as_2d(embeddings: ArrayLike, name: str) -> NDArray[np.float32]:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    return array


def redundancy_to_set(
    form_embeddings: ArrayLike,
    caption_embeddings: ArrayLike | None,
    candidate_index: int,
    selected_indices: Sequence[int],
    config: JointRedundancyConfig | None = None,
) -> RedundancyBreakdown:
    """Marginal redundancy of a candidate relative to a selected set.

    ``top_k=1`` uses the most redundant selected neighbor. Larger values average
    the top-k most redundant neighbors, which is more stable for large sets.
    """

    config = config or JointRedundancyConfig()
    if not selected_indices:
        return RedundancyBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, "none")

    form = _as_2d(form_embeddings, "form_embeddings")
    form_similarities = cosine_similarity_matrix(
        form[[candidate_index]], form[list(selected_indices)]
    )[0]

    if caption_embeddings is None:
        caption_similarities: list[float | None] = [None] * len(selected_indices)
    else:
        captions = _as_2d(caption_embeddings, "caption_embeddings")
        caption_similarities = [
            float(value)
            for value in cosine_similarity_matrix(
                captions[[candidate_index]], captions[list(selected_indices)]
            )[0]
        ]

    breakdowns = [
        pair_redundancy(float(form_sim), caption_sim, config)
        for form_sim, caption_sim in zip(
            form_similarities, caption_similarities, strict=True
        )
    ]
    breakdowns.sort(key=lambda item: item.joint_redundancy, reverse=True)
    top = breakdowns[: min(config.top_k, len(breakdowns))]
    count = len(top)

    return RedundancyBreakdown(
        form_similarity=sum(item.form_similarity for item in top) / count,
        caption_similarity=sum(item.caption_similarity for item in top) / count,
        form_redundancy=sum(item.form_redundancy for item in top) / count,
        caption_redundancy=sum(item.caption_redundancy for item in top) / count,
        joint_redundancy=sum(item.joint_redundancy for item in top) / count,
        dominant=top[0].dominant,
    )


def marginal_information(
    form_embeddings: ArrayLike,
    caption_embeddings: ArrayLike | None,
    candidate_index: int,
    selected_indices: Sequence[int],
    quality_score: float,
    config: JointRedundancyConfig | None = None,
) -> MarginalInfo:
    """Score a candidate by novelty minus redundancy, plus image quality.

    Novelty is measured independently on the form and caption axes: a candidate
    can be new because it shows a new form, or because its caption teaches a new
    concept, or both.
    """

    config = config or JointRedundancyConfig()
    form = _as_2d(form_embeddings, "form_embeddings")

    if selected_indices:
        form_similarities = cosine_similarity_matrix(
            form[[candidate_index]], form[list(selected_indices)]
        )[0]
        form_novelty = 1.0 - float(form_similarities.max())
    else:
        form_novelty = 1.0

    if caption_embeddings is None:
        caption_novelty = 0.0
        novelty = form_novelty
    else:
        captions = _as_2d(caption_embeddings, "caption_embeddings")
        if selected_indices:
            caption_similarities = cosine_similarity_matrix(
                captions[[candidate_index]], captions[list(selected_indices)]
            )[0]
            caption_novelty = 1.0 - float(caption_similarities.max())
        else:
            caption_novelty = 1.0
        weight_sum = config.form_novelty_weight + config.caption_novelty_weight
        novelty = (
            config.form_novelty_weight * form_novelty
            + config.caption_novelty_weight * caption_novelty
        ) / weight_sum

    redundancy = redundancy_to_set(
        form,
        caption_embeddings,
        candidate_index,
        selected_indices,
        config,
    )
    score = (
        config.quality_weight * _clip01(quality_score)
        + novelty
        - config.redundancy_weight * redundancy.joint_redundancy
    )
    return MarginalInfo(
        form_novelty=form_novelty,
        caption_novelty=caption_novelty,
        novelty=novelty,
        redundancy=redundancy,
        score=score,
    )


def greedy_select(
    form_embeddings: ArrayLike,
    caption_embeddings: ArrayLike | None,
    quality_scores: Sequence[float],
    *,
    max_items: int | None = None,
    form_ids: Sequence[Hashable] | None = None,
    per_form_cap: int | None = None,
    config: JointRedundancyConfig | None = None,
) -> list[int]:
    """Greedy set selection using marginal information and per-form caps.

    This is a practical baseline, not a certified submodular optimizer. It is
    O(n^2) in the number of candidates, which is fine for 100–10,000 images with
    cached embeddings. Larger sets should switch to a kNN/FAISS-backed variant.
    """

    config = config or JointRedundancyConfig()
    form = _as_2d(form_embeddings, "form_embeddings")
    total = form.shape[0]

    if len(quality_scores) != total:
        raise ValueError("quality_scores length must match form_embeddings")
    if form_ids is not None and len(form_ids) != total:
        raise ValueError("form_ids length must match form_embeddings")
    if caption_embeddings is not None:
        captions = _as_2d(caption_embeddings, "caption_embeddings")
        if captions.shape[0] != total:
            raise ValueError("caption_embeddings length must match form_embeddings")
    if per_form_cap is not None and per_form_cap < 1:
        raise ValueError("per_form_cap must be >= 1")

    budget = total if max_items is None else min(max_items, total)
    selected: list[int] = []
    remaining = set(range(total))
    counts: dict[Hashable, int] = {}

    while remaining and len(selected) < budget:
        best_index: int | None = None
        best_score = float("-inf")

        for index in sorted(remaining):
            if form_ids is not None and per_form_cap is not None:
                form_id = form_ids[index]
                if counts.get(form_id, 0) >= per_form_cap:
                    continue

            info = marginal_information(
                form,
                caption_embeddings,
                index,
                selected,
                quality_scores[index],
                config,
            )
            if info.score > best_score:
                best_score = info.score
                best_index = index

        if best_index is None:
            break
        if selected and best_score <= 0:
            break

        selected.append(best_index)
        remaining.remove(best_index)
        if form_ids is not None:
            form_id = form_ids[best_index]
            counts[form_id] = counts.get(form_id, 0) + 1

    return selected


def parse_tags(caption: str) -> set[str]:
    """Parse a comma-separated tag caption into a normalized set."""

    return {
        tag.strip().lower().replace(" ", "_")
        for tag in caption.split(",")
        if tag.strip()
    }


def jaccard_similarity(
    left: str | set[str],
    right: str | set[str],
    *,
    stop_tags: set[str] | None = None,
) -> float:
    """Jaccard similarity for tag captions.

    ``stop_tags`` should contain ubiquitous tags such as ``1girl`` or ``solo``;
    otherwise every anime caption looks redundant.
    """

    left_tags = parse_tags(left) if isinstance(left, str) else {tag.lower() for tag in left}
    right_tags = parse_tags(right) if isinstance(right, str) else {tag.lower() for tag in right}
    if stop_tags:
        left_tags = left_tags - stop_tags
        right_tags = right_tags - stop_tags
    if not left_tags or not right_tags:
        return 0.0
    return len(left_tags & right_tags) / len(left_tags | right_tags)


def hybrid_caption_similarity(
    embedding_similarity: float,
    tag_similarity: float,
    *,
    tag_weight: float = 0.5,
) -> float:
    """Blend dense caption similarity with tag overlap."""

    tag_weight = _clip01(tag_weight)
    return _clip01((1.0 - tag_weight) * embedding_similarity + tag_weight * tag_similarity)
