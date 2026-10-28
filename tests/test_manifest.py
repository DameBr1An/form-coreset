from pathlib import Path

from formcoreset.manifest import read_manifest, write_manifest, write_report
from formcoreset.types import FormGroup, ImageRecord, ImageStatus, ScanResult


def _record(tmp_path: Path) -> ImageRecord:
    return ImageRecord(
        image_id="img_0000",
        path=tmp_path / "a.png",
        rel_path="a.png",
        sha256="abc",
        width=64,
        height=64,
        format="PNG",
        status=ImageStatus.CONFIRMED,
        form_id="form_001",
    )


def test_write_and_read_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest([_record(tmp_path)], path)

    rows = read_manifest(path)

    assert len(rows) == 1
    assert rows[0]["status"] == "confirmed"
    assert rows[0]["form_id"] == "form_001"
    assert isinstance(rows[0]["path"], str)


def test_write_report(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    scan = ScanResult(root=tmp_path, records=[_record(tmp_path)])
    group = FormGroup(form_id="form_001", member_ids=["img_0000"], canonical_id="img_0000")

    write_report(path, scan=scan, groups=[group], mode="test")

    text = path.read_text(encoding="utf-8")
    assert "form_001" in text
    assert "Members: 1" in text
    assert "Input: 1 images" in text
