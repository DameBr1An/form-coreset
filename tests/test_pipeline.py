from pathlib import Path

import numpy as np
from PIL import Image

from formcoreset.config import RunConfig
from formcoreset.ingest import sha256_file
from formcoreset.pipeline import run_baseline


def _gradient(path: Path, horizontal: bool, size: int = 64) -> None:
    array = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            value = int(255 * (x if horizontal else y) / (size - 1))
            array[y, x] = (value, value, value)
    Image.fromarray(array, mode="RGB").save(path)


def test_baseline_pipeline(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    input_dir.mkdir()
    _gradient(input_dir / "a.png", horizontal=True)
    _gradient(input_dir / "a_copy.png", horizontal=True)
    _gradient(input_dir / "b.png", horizontal=False)
    before = {path.name: sha256_file(path) for path in input_dir.glob("*.png")}

    scan, groups = run_baseline(input_dir, out_dir, RunConfig())

    assert len(scan.records) == 3
    assert len(groups) == 2
    assert (out_dir / "manifest.jsonl").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "contact_sheet.png").exists()
    assert (out_dir / "run.json").exists()

    forms_root = out_dir / "subjects" / "subject_001" / "forms"
    assert len(list(forms_root.glob("form_*"))) == 2
    assert any(path.name.startswith("canonical.") for path in forms_root.rglob("*"))

    after = {path.name: sha256_file(path) for path in input_dir.glob("*.png")}
    assert before == after
