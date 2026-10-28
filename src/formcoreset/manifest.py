"""Manifest, run metadata, and human-readable reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from . import __version__
from .types import ImageRecord, ScanResult


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def record_to_dict(record: ImageRecord) -> dict[str, Any]:
    data = asdict(record)
    data["path"] = record.path.as_posix()
    data["status"] = record.status.value
    return data


def write_manifest(records: list[ImageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record_to_dict(record), ensure_ascii=False, default=_json_default)
            )
            handle.write("\n")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_run_manifest(
    run_dir: Path,
    *,
    config: Any,
    scan: ScanResult,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "app_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "input_root": scan.root.as_posix(),
        "input_count": len(scan.records),
        "skipped": scan.skipped,
        "config": (
            config.model_dump(mode="json") if hasattr(config, "model_dump") else str(config)
        ),
    }
    if extra:
        payload.update(extra)
    path = run_dir / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_report(
    path: Path,
    *,
    scan: ScanResult,
    groups: list[Any],
    mode: str,
    extra_lines: list[str] | None = None,
) -> None:
    lines = [
        "# FormCoreset report",
        "",
        f"- Input: {len(scan.records)} images",
        f"- Skipped: {len(scan.skipped)}",
        f"- Mode: {mode}",
        f"- Groups: {len(groups)}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    lines.append("")

    if scan.skipped:
        lines.append("## Skipped")
        for item in scan.skipped:
            lines.append(f"- `{item['path']}`: {item['reason']}")
        lines.append("")

    lines.append("## Groups")
    for group in groups:
        lines.append(f"### {group.form_id}")
        if getattr(group, "label", None):
            lines.append(f"- Label: {group.label}")
        lines.append(f"- Members: {len(group.member_ids)}")
        if group.canonical_id:
            lines.append(f"- Canonical: `{group.canonical_id}`")
        if getattr(group, "method", None):
            lines.append(f"- Method: `{group.method}`")
        tags = getattr(group, "metadata", {}).get("top_tags", [])
        if tags:
            tag_text = ", ".join(
                f"{item['tag']}:{item['confidence']:.2f}" if isinstance(item, dict) else str(item)
                for item in tags
            )
            lines.append(f"- Top tags: {tag_text}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
