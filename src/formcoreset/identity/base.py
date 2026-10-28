"""Identity matcher interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..types import ImageRecord


class IdentityMatcher(Protocol):
    """Scores how likely an image belongs to the target subject."""

    name: str
    version: str

    def score(
        self,
        record: ImageRecord,
        *,
        references: list[Path],
        text: str | None = None,
    ) -> float:
        """Return a calibrated similarity score in [0, 1]."""
        ...
