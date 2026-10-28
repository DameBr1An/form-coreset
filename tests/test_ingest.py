from pathlib import Path

import pytest
from PIL import Image

from formcoreset.ingest import scan_directory


def _make_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> None:
    Image.new("RGB", size, color).save(path)


def test_scan_directory(tmp_path: Path) -> None:
    _make_image(tmp_path / "a.png", (255, 0, 0))
    _make_image(tmp_path / "b.jpg", (0, 255, 0))

    result = scan_directory(tmp_path)

    assert len(result.records) == 2
    assert result.skipped == []
    assert {record.rel_path for record in result.records} == {"a.png", "b.jpg"}
    assert all(record.sha256 for record in result.records)


def test_scan_skips_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / "bad.png").write_bytes(b"not an image")

    result = scan_directory(tmp_path)

    assert len(result.records) == 0
    assert len(result.skipped) == 1
    assert result.skipped[0]["path"] == "bad.png"


def test_scan_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_directory(tmp_path / "missing")
