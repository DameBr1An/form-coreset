"""Core data types used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ImageStatus(StrEnum):
    """Where an image ended up after identity filtering."""

    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    REJECTED = "rejected"
    UNASSIGNED = "unassigned"


class TargetMode(StrEnum):
    """How the target subject was specified."""

    GUIDED = "guided"
    AUTO = "auto"


@dataclass
class ImageRecord:
    """Everything FormCoreset knows about one input image."""

    image_id: str
    path: Path
    rel_path: str
    sha256: str
    width: int
    height: int
    format: str
    mode: str | None = None
    has_alpha: bool = False
    flags: list[str] = field(default_factory=list)
    quality: dict[str, float] = field(default_factory=dict)
    identity_score: float | None = None
    identity_reasons: list[str] = field(default_factory=list)
    form_id: str | None = None
    form_confidence: float | None = None
    source_id: str | None = None
    canonical_score: float | None = None
    is_canonical: bool = False
    status: ImageStatus = ImageStatus.UNCERTAIN
    same_source_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pixels(self) -> int:
        return self.width * self.height


@dataclass
class FormGroup:
    """A group of images believed to share one form."""

    form_id: str
    member_ids: list[str]
    canonical_id: str | None = None
    confidence: float | None = None
    method: str = "baseline"
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubjectGroup:
    """An identity cluster discovered in auto mode."""

    subject_id: str
    member_ids: list[str]
    label: str | None = None
    confidence: float | None = None
    method: str = "auto"
    forms: list[FormGroup] = field(default_factory=list)


@dataclass
class ScanResult:
    """Output of scanning an input directory."""

    root: Path
    records: list[ImageRecord]
    skipped: list[dict[str, str]] = field(default_factory=list)
