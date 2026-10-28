"""Configuration models and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    description: str | None = None
    references: list[Path] = Field(default_factory=list)
    mode: Literal["guided", "auto"] = "guided"


class IdentityConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_matcher: str = "auto"
    text_matcher: str = "auto"
    alpha: float = 0.75
    confirmed_threshold: float = 0.75
    rejected_threshold: float = 0.45


class FormConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    encoder: str = "subject"
    clusterer: str = "hdbscan"
    min_cluster_size: int = 3
    min_samples: int = 2
    granularity: float = 0.5
    use_keypoints: Literal["auto", "always", "never"] = "auto"
    mask_provider: Literal["auto", "anime", "general", "none"] = "auto"
    mask_scale: int = 1024
    subject_dino_model: str = "vit_small_patch14_dinov2.lvd142m"
    subject_dino_weight: float = 0.7
    subject_silhouette_weight: float = 0.3
    tag_weight: float = 0.7
    embedding_weight: float = 0.3
    general_threshold: float = 0.2
    tag_threshold: float = 0.2


class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: Literal["reference", "training"] = "reference"
    preset: str = "most_standard"
    max_per_form: int = 3
    policy: Literal["strict", "balanced", "diverse"] = "balanced"


class SameSourceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    method: Literal["sift", "orb"] = "sift"
    min_score: float = 0.3
    min_inliers: int = 12
    ratio: float = 0.75
    max_features: int = 2000
    linkage: Literal["complete", "average", "single"] = "complete"
    star_merge: bool = True
    star_threshold: float = 0.2
    use_mask: bool = True
    mask_provider: Literal["auto", "anime", "general", "none"] = "auto"
    mask_scale: int = 1024
    verify: bool = True
    min_correlation: float = 0.35
    stitch: bool = True
    stitch_min_gain: float = 0.02


class NormalizeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False


class PerformanceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 8
    seed: int = 42


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target: TargetConfig = Field(default_factory=TargetConfig)
    adapter: Literal["auto", "person", "object"] = "auto"
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    form: FormConfig = Field(default_factory=FormConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    source: SameSourceConfig = Field(default_factory=SameSourceConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


def load_config(path: Path | None = None) -> RunConfig:
    """Load a run config from YAML, or return defaults."""

    if path is None:
        return RunConfig()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return RunConfig.model_validate(data)
