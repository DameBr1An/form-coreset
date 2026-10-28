"""Pipeline orchestration.

The baseline path is model-free: scan, quality heuristics, dHash grouping.
Model-backed identity and form stages will plug into the same orchestration
without changing the output contract.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from .cluster import cluster_features, compute_hashes, group_near_duplicates
from .config import RunConfig
from .form.base import FormEncoder, FormFeatures
from .ingest import scan_directory
from .organize import organize_form_groups, organize_groups, organize_source_groups
from .quality import assess_record
from .select import pick_canonical
from .source import SourceGroup, group_same_source
from .types import FormGroup, ImageStatus, ScanResult


def scan_and_assess(input_dir: Path) -> ScanResult:
    scan = scan_directory(input_dir)
    for record in scan.records:
        assess_record(record)
    return scan


def run_baseline(
    input_dir: Path,
    out_dir: Path,
    config: RunConfig,
    *,
    threshold: int = 5,
) -> tuple[ScanResult, list[FormGroup]]:
    """Model-free baseline: near-duplicate grouping, canonical, output tree."""

    scan = scan_and_assess(input_dir)
    hashes = compute_hashes(scan.records)
    raw_groups = group_near_duplicates(scan.records, hashes, threshold=threshold)

    groups: list[FormGroup] = []
    for index, members in enumerate(raw_groups, start=1):
        canonical = pick_canonical(members)
        group = FormGroup(
            form_id=f"form_{index:03d}",
            member_ids=[member.image_id for member in members],
            canonical_id=canonical.image_id if canonical else None,
            method="baseline_dhash",
        )
        for member in members:
            member.status = ImageStatus.CONFIRMED
            member.form_id = group.form_id
            member.metadata["baseline"] = True
        groups.append(group)

    organize_groups(
        out_dir=out_dir,
        scan=scan,
        groups=groups,
        config=config,
        mode="auto-baseline",
    )
    return scan, groups


def run_form_extraction(
    input_dir: Path,
    out_dir: Path,
    config: RunConfig,
    *,
    encoder: FormEncoder | None = None,
) -> tuple[ScanResult, list[FormGroup]]:
    """Build a Form Library with the configured form encoder."""

    scan = scan_and_assess(input_dir)

    if encoder is None:
        if config.form.encoder == "anime-hybrid":
            if importlib.util.find_spec("imgutils") is None:
                raise ImportError(
                    "The anime-hybrid encoder requires dghs-imgutils. "
                    "Install it with: uv sync --extra anime"
                )
            from .form.anime_hybrid import AnimeHybridFormEncoder

            encoder = AnimeHybridFormEncoder(config.form)
        elif config.form.encoder == "subject":
            if (
                importlib.util.find_spec("timm") is None
                or importlib.util.find_spec("torch") is None
            ):
                raise ImportError(
                    "The subject encoder requires timm and torch. "
                    "Install them with: uv sync --extra subject"
                )
            from .form.subject import SubjectFormEncoder

            encoder = SubjectFormEncoder(
                config.form,
                device=config.performance.device,
            )
        else:
            raise ValueError(f"Unsupported form encoder: {config.form.encoder}")

    features: list[np.ndarray] = []
    feature_by_id: dict[str, FormFeatures] = {}
    usable_records = []

    for record in scan.records:
        try:
            form_features = encoder.encode(record)
        except Exception as exc:  # noqa: BLE001 - per-image failure
            record.status = ImageStatus.UNCERTAIN
            record.flags.append("form_encoding_failed")
            record.metadata["form_error"] = str(exc)
            continue
        features.append(form_features.embedding)
        feature_by_id[record.image_id] = form_features
        usable_records.append(record)
        record.metadata["form_encoder"] = form_features.method
        record.metadata["form_embedding_used"] = form_features.metadata.get(
            "embedding_used", False
        )

    if len(features) < 2:
        organize_form_groups(
            out_dir=out_dir,
            scan=scan,
            groups=[],
            config=config,
            encoder_version=encoder.version,
        )
        return scan, []

    cluster_result = cluster_features(
        np.stack(features),
        min_cluster_size=config.form.min_cluster_size,
        min_samples=config.form.min_samples,
    )
    label_by_id = {
        record.image_id: int(label)
        for record, label in zip(usable_records, cluster_result.labels, strict=True)
    }

    groups: list[FormGroup] = []
    for cluster_label in sorted(
        {int(label) for label in cluster_result.labels if label >= 0}
    ):
        members = [
            record
            for record in usable_records
            if label_by_id[record.image_id] == cluster_label
        ]
        if not members:
            continue
        canonical = pick_canonical(members)

        confidences = [
            feature_by_id[member.image_id].confidence or 0.0 for member in members
        ]
        group = FormGroup(
            form_id=f"form_{cluster_label + 1:03d}",
            member_ids=[member.image_id for member in members],
            canonical_id=canonical.image_id if canonical else None,
            confidence=float(np.mean(confidences)) if confidences else None,
            method=cluster_result.method,
            metadata={"params": cluster_result.params},
        )
        groups.append(group)

    groups.sort(key=lambda group: group.form_id)
    organize_form_groups(
        out_dir=out_dir,
        scan=scan,
        groups=groups,
        config=config,
        encoder_version=encoder.version,
        mode="form-library",
    )
    return scan, groups


def run_same_source_dedup(
    input_dir: Path,
    out_dir: Path,
    config: RunConfig,
    *,
    progress: bool = True,
) -> tuple[ScanResult, list[SourceGroup]]:
    """Group images that are variants of the same source artwork.

    This is the first-pass deduplication path: crops, resizes, compression,
    background replacement, and effects are grouped without pose labels.
    """

    scan = scan_and_assess(input_dir)
    groups = group_same_source(scan.records, config.source, progress=progress)
    organize_source_groups(
        out_dir=out_dir,
        scan=scan,
        groups=groups,
        config=config,
        mode="same-source",
    )
    return scan, groups
