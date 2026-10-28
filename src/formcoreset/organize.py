"""Create the output tree, copy selected images, and write reports."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .manifest import write_manifest, write_report, write_run_manifest
from .select import pick_canonical, score_record
from .source import SourceGroup
from .stitch import stitch_group
from .types import FormGroup, ImageRecord, ImageStatus, ScanResult

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
logger = logging.getLogger(__name__)


def safe_name(name: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or fallback


def copy_image(record: ImageRecord, destination: Path) -> None:
    """Copy an image, never moving or deleting the source."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(record.path, destination)
    except OSError:
        shutil.copyfile(record.path, destination)


def write_contact_sheet(
    groups: list[FormGroup],
    records_by_id: dict[str, ImageRecord],
    path: Path,
    thumb_size: int = 256,
    max_variants: int = 6,
    canonical_overrides: dict[str, Path] | None = None,
) -> None:
    if not groups:
        return

    rows = len(groups)
    cols = 1 + max_variants
    cell = thumb_size + 12
    sheet = Image.new("RGB", (cols * cell, rows * cell), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for row, group in enumerate(groups):
        members = [records_by_id[i] for i in group.member_ids if i in records_by_id]
        members.sort(
            key=lambda record: (
                record.image_id != group.canonical_id,
                -score_record(record),
            )
        )
        image_paths = [record.path for record in members]
        if canonical_overrides and group.form_id in canonical_overrides:
            image_paths = [canonical_overrides[group.form_id]] + [
                record.path
                for record in members
                if record.image_id != group.canonical_id
            ]
        draw.text((8, row * cell + 4), group.label or group.form_id, fill=(20, 20, 20))
        for col, image_path in enumerate(image_paths[:cols]):
            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((thumb_size, thumb_size))
                    x = col * cell + (cell - image.width) // 2
                    y = row * cell + 20 + (thumb_size - image.height) // 2
                    sheet.paste(image, (x, y))
            except OSError:
                continue

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def organize_groups(
    *,
    out_dir: Path,
    scan: ScanResult,
    groups: list[FormGroup],
    config: Any,
    mode: str,
) -> None:
    """Write the auto-mode output tree for a set of form groups."""

    out_dir.mkdir(parents=True, exist_ok=True)
    records_by_id = {record.image_id: record for record in scan.records}
    forms_root = out_dir / "subjects" / "subject_001" / "forms"
    grouped_ids: set[str] = set()

    for group in groups:
        members = [records_by_id[i] for i in group.member_ids if i in records_by_id]
        if not members:
            continue
        canonical = records_by_id.get(group.canonical_id or "") or pick_canonical(members)
        if canonical is None:
            continue
        group.canonical_id = canonical.image_id

        group_dir = forms_root / group.form_id
        canonical_destination = group_dir / f"canonical{canonical.path.suffix.lower()}"
        copy_image(canonical, canonical_destination)

        canonical.is_canonical = True
        canonical.canonical_score = score_record(canonical)
        canonical.form_id = group.form_id
        canonical.status = ImageStatus.CONFIRMED

        variants_dir = group_dir / "variants"
        for member in members:
            grouped_ids.add(member.image_id)
            member.form_id = group.form_id
            member.status = ImageStatus.CONFIRMED
            member.metadata["baseline"] = True
            if member.image_id != canonical.image_id:
                copy_image(member, variants_dir / safe_name(member.path.name))

        form_payload = {
            "form_id": group.form_id,
            "canonical_id": canonical.image_id,
            "members": group.member_ids,
            "method": group.method,
            "confidence": group.confidence,
        }
        (group_dir / "form.json").write_text(
            json.dumps(form_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Never silently drop an image that could not be hashed or grouped.
    for record in scan.records:
        if record.image_id in grouped_ids:
            continue
        record.status = ImageStatus.UNCERTAIN
        copy_image(record, out_dir / "uncertain" / safe_name(record.path.name))

    write_manifest(scan.records, out_dir / "manifest.jsonl")
    write_report(out_dir / "report.md", scan=scan, groups=groups, mode=mode)
    write_run_manifest(out_dir, config=config, scan=scan, extra={"mode": mode})
    write_contact_sheet(groups, records_by_id, out_dir / "contact_sheet.png")


def organize_form_groups(
    *,
    out_dir: Path,
    scan: ScanResult,
    groups: list[FormGroup],
    config: Any,
    encoder_version: str,
    mode: str = "form-library",
) -> None:
    """Write the Form Library layout for encoder-backed form groups."""

    out_dir.mkdir(parents=True, exist_ok=True)
    records_by_id = {record.image_id: record for record in scan.records}
    forms_root = out_dir / "forms"
    grouped_ids: set[str] = set()

    for group in groups:
        members = [records_by_id[i] for i in group.member_ids if i in records_by_id]
        if not members:
            continue
        canonical = records_by_id.get(group.canonical_id or "") or pick_canonical(members)
        if canonical is None:
            continue
        group.canonical_id = canonical.image_id

        group_dir = forms_root / group.form_id
        copy_image(canonical, group_dir / f"canonical{canonical.path.suffix.lower()}")
        canonical.is_canonical = True
        canonical.canonical_score = score_record(canonical)
        canonical.form_id = group.form_id
        canonical.form_confidence = group.confidence
        canonical.status = ImageStatus.CONFIRMED

        variants_dir = group_dir / "variants"
        for member in members:
            grouped_ids.add(member.image_id)
            member.form_id = group.form_id
            member.form_confidence = group.confidence
            member.status = ImageStatus.CONFIRMED
            member.metadata["encoder"] = encoder_version
            if member.image_id != canonical.image_id:
                copy_image(
                    member,
                    variants_dir / f"{member.image_id}_{safe_name(member.path.name)}",
                )

        payload = {
            "form_id": group.form_id,
            "canonical_id": canonical.image_id,
            "members": group.member_ids,
            "method": group.method,
            "confidence": group.confidence,
            "metadata": group.metadata,
            "encoder": encoder_version,
        }
        (group_dir / "form.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for record in scan.records:
        if record.image_id in grouped_ids:
            continue
        record.status = ImageStatus.UNCERTAIN
        copy_image(
            record,
            out_dir / "uncertain" / f"{record.image_id}_{safe_name(record.path.name)}",
        )

    uncertain_count = sum(
        1 for record in scan.records if record.status == ImageStatus.UNCERTAIN
    )
    write_manifest(scan.records, out_dir / "manifest.jsonl")
    write_report(
        out_dir / "report.md",
        scan=scan,
        groups=groups,
        mode=mode,
        extra_lines=[
            f"- Encoder: `{encoder_version}`",
            f"- Clusters: {len(groups)}",
            f"- Uncertain: {uncertain_count}",
        ],
    )
    write_run_manifest(
        out_dir,
        config=config,
        scan=scan,
        extra={"mode": mode, "encoder": encoder_version, "clusters": len(groups)},
    )
    write_contact_sheet(groups, records_by_id, out_dir / "contact_sheet.png")


def organize_source_groups(
    *,
    out_dir: Path,
    scan: ScanResult,
    groups: list[SourceGroup],
    config: Any,
    mode: str = "same-source",
) -> None:
    """Write same-source groups: canonical + variants, no pose labels."""

    out_dir.mkdir(parents=True, exist_ok=True)
    records_by_id = {record.image_id: record for record in scan.records}
    sources_root = out_dir / "sources"
    grouped_ids: set[str] = set()
    form_groups: list[FormGroup] = []
    canonical_overrides: dict[str, Path] = {}
    stitched_count = 0

    for group in groups:
        members = [
            record for record in group.members if record.image_id in records_by_id
        ]
        if not members:
            continue
        canonical = records_by_id.get(group.canonical_id or "") or pick_canonical(members)
        if canonical is None:
            continue
        group.canonical_id = canonical.image_id

        stitched_result = None
        if config.source.stitch and len(members) >= 2:
            try:
                stitched_result = stitch_group(members, canonical, config.source)
            except Exception as exc:  # noqa: BLE001 - stitching is optional
                logger.warning("Stitching failed for %s: %s", group.source_id, exc)

        source_dir = sources_root / group.source_id
        suffix = canonical.path.suffix.lower()
        copy_image(canonical, source_dir / f"canonical_single{suffix}")
        stitched_used = bool(stitched_result and stitched_result.used)
        if stitched_result is not None and stitched_result.used:
            stitched_path = source_dir / "canonical.png"
            stitched_result.image.save(stitched_path)
            canonical_overrides[group.source_id] = stitched_path
            stitched_count += 1
        else:
            copy_image(canonical, source_dir / f"canonical{suffix}")

        canonical.is_canonical = True
        canonical.canonical_score = score_record(canonical)
        canonical.source_id = group.source_id
        canonical.status = ImageStatus.CONFIRMED
        canonical.metadata["stitched"] = stitched_used

        variants_dir = source_dir / "variants"
        for member in members:
            grouped_ids.add(member.image_id)
            member.source_id = group.source_id
            member.status = ImageStatus.CONFIRMED
            if member.image_id != canonical.image_id:
                copy_image(
                    member,
                    variants_dir / f"{member.image_id}_{safe_name(member.path.name)}",
                )

        payload = {
            "source_id": group.source_id,
            "canonical_id": canonical.image_id,
            "canonical_single": canonical.image_id,
            "members": [member.image_id for member in members],
            "method": group.method,
            "min_score": group.min_score,
            "stitched": stitched_used,
            "stitch_gain": round(stitched_result.gain, 4) if stitched_result else 0.0,
            "stitch_members": (
                len(stitched_result.used_members) if stitched_result else 0
            ),
        }
        (source_dir / "source.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        form_groups.append(
            FormGroup(
                form_id=group.source_id,
                member_ids=[member.image_id for member in members],
                canonical_id=canonical.image_id,
                confidence=group.min_score,
                method=group.method,
                metadata={
                    "min_score": group.min_score,
                    "stitched": stitched_used,
                    "stitch_gain": (
                        round(stitched_result.gain, 4) if stitched_result else 0.0
                    ),
                },
            )
        )

    for record in scan.records:
        if record.image_id in grouped_ids:
            continue
        record.status = ImageStatus.UNCERTAIN
        copy_image(
            record,
            out_dir / "uncertain" / f"{record.image_id}_{safe_name(record.path.name)}",
        )

    singletons = sum(1 for group in groups if len(group.members) == 1)
    variants_removed = len(scan.records) - len(groups)
    method = groups[0].method if groups else "n/a"
    write_manifest(scan.records, out_dir / "manifest.jsonl")
    write_report(
        out_dir / "report.md",
        scan=scan,
        groups=form_groups,
        mode=mode,
        extra_lines=[
            f"- Dedupe method: `{method}`",
            f"- Source groups: {len(groups)}",
            f"- Singletons: {singletons}",
            f"- Variants removed: {variants_removed}",
            f"- Stitched groups: {stitched_count}",
        ],
    )
    write_run_manifest(
        out_dir,
        config=config,
        scan=scan,
        extra={
            "mode": mode,
            "method": method,
            "groups": len(groups),
            "singletons": singletons,
            "variants_removed": variants_removed,
            "stitched_groups": stitched_count,
        },
    )
    write_contact_sheet(
        form_groups,
        records_by_id,
        out_dir / "contact_sheet.png",
        canonical_overrides=canonical_overrides,
    )
