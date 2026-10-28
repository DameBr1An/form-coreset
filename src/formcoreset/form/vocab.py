"""Anime form vocabulary used by the hybrid form encoder.

These are Danbooru/WD14-style tags. The vocabulary is intentionally small and
focused on posture, view, and coarse action; it is not a full tag taxonomy.
"""

from __future__ import annotations

from collections.abc import Sequence

POSTURE_TAGS = [
    "standing",
    "sitting",
    "running",
    "jumping",
    "lying",
    "kneeling",
    "squatting",
    "walking",
    "dancing",
    "on_back",
    "on_stomach",
    "all_fours",
    "seiza",
    "crossed_legs",
]

VIEW_TAGS = [
    "from_side",
    "from_behind",
    "from_above",
    "from_below",
    "looking_at_viewer",
    "profile",
    "full_body",
    "upper_body",
    "portrait",
    "cowboy_shot",
]

ACTION_TAGS = [
    "holding",
    "arms_up",
    "hands_up",
    "hand_up",
    "arm_up",
    "crossed_arms",
    "hand_on_hip",
    "outstretched_arm",
    "spread_arms",
]

TAG_VOCAB = POSTURE_TAGS + VIEW_TAGS + ACTION_TAGS


def _top_tag(tags: Sequence[str], scores: dict[str, float]) -> str | None:
    candidates = [(tag, scores[tag]) for tag in tags if tag in scores]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def form_label(scores: dict[str, float]) -> str:
    """Human-readable label: posture first, then view, then action."""

    parts = [
        tag
        for tag in (
            _top_tag(POSTURE_TAGS, scores),
            _top_tag(VIEW_TAGS, scores),
            _top_tag(ACTION_TAGS, scores),
        )
        if tag
    ]
    if parts:
        return " / ".join(parts)
    generic = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
    return " / ".join(tag for tag, _ in generic) or "unknown"
