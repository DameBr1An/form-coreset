"""Form encoder interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..types import ImageRecord


@dataclass
class FormFeatures:
    """Encoded form representation for one image."""

    embedding: np.ndarray
    pose: np.ndarray | None = None
    confidence: float | None = None
    method: str = "embedding"
    tags: dict[str, float] = field(default_factory=dict)
    tag_vector: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FormEncoder(Protocol):
    """Turns an image into a form representation suitable for clustering."""

    name: str
    version: str

    def encode(self, record: ImageRecord) -> FormFeatures:
        ...
