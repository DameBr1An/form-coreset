from pathlib import Path

import numpy as np
from PIL import Image

from formcoreset.config import RunConfig
from formcoreset.form.base import FormFeatures
from formcoreset.pipeline import run_form_extraction
from formcoreset.types import ImageRecord


class FakeEncoder:
    name = "fake"
    version = "fake-v1"

    def encode(self, record: ImageRecord) -> FormFeatures:
        rel_path = record.rel_path
        if rel_path.startswith("bad"):
            raise RuntimeError("fake encoding failure")
        if rel_path.startswith("a"):
            embedding = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            embedding = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return FormFeatures(
            embedding=embedding,
            confidence=0.9,
            method=self.version,
            metadata={
                "top_tags": [{"tag": "fake", "confidence": 0.9}],
                "embedding_used": True,
            },
        )


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (64, 64), color).save(path)


def test_run_form_extraction(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for name, color in [
        ("a1.png", (255, 0, 0)),
        ("a2.png", (255, 50, 0)),
        ("b1.png", (0, 255, 0)),
        ("b2.png", (0, 200, 0)),
        ("bad.png", (0, 0, 255)),
    ]:
        _make_image(input_dir / name, color)

    config = RunConfig()
    config.form.min_cluster_size = 2
    config.form.min_samples = 1
    out_dir = tmp_path / "out"

    scan, groups = run_form_extraction(
        input_dir,
        out_dir,
        config,
        encoder=FakeEncoder(),
    )

    assert len(scan.records) == 5
    assert len(groups) == 2
    assert (out_dir / "forms").exists()
    assert (out_dir / "manifest.jsonl").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "contact_sheet.png").exists()
    assert (out_dir / "uncertain").exists()
    assert any((out_dir / "uncertain").iterdir())
