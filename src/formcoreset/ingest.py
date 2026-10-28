"""Scan input directories and build image records."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .types import ImageRecord, ScanResult

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA"} or "transparency" in image.info


def scan_directory(root: Path) -> ScanResult:
    """Walk ``root`` and return one record per supported image."""

    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    records: list[ImageRecord] = []
    skipped: list[dict[str, str]] = []

    for index, path in enumerate(paths):
        relative = path.relative_to(root).as_posix()
        image_id = f"img_{index:04d}"
        try:
            digest = sha256_file(path)
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                image_format = image.format or path.suffix.lstrip(".").upper()
                mode = image.mode
                has_alpha = _has_alpha(image)
            records.append(
                ImageRecord(
                    image_id=image_id,
                    path=path,
                    rel_path=relative,
                    sha256=digest,
                    width=width,
                    height=height,
                    format=image_format,
                    mode=mode,
                    has_alpha=has_alpha,
                )
            )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            skipped.append({"path": relative, "reason": str(exc)})

    return ScanResult(root=root, records=records, skipped=skipped)
