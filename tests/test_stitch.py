import importlib.util
import random
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from formcoreset.config import RunConfig, SameSourceConfig
from formcoreset.pipeline import run_same_source_dedup
from formcoreset.stitch import stitch_group
from formcoreset.types import ImageRecord

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None,
    reason="OpenCV is required for stitching",
)


def _textured_image(path: Path, width: int, height: int, seed: int) -> None:
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    for _ in range(60):
        x = rng.randint(0, width - 20)
        y = rng.randint(0, height - 20)
        w = rng.randint(20, 70)
        h = rng.randint(20, 70)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        if rng.random() < 0.5:
            draw.rectangle([x, y, x + w, y + h], fill=color, outline="black")
        else:
            draw.ellipse([x, y, x + w, y + h], fill=color, outline="black")
    image.save(path)


def _record(path: Path, image_id: str) -> ImageRecord:
    with Image.open(path) as image:
        width, height = image.size
    return ImageRecord(
        image_id=image_id,
        path=path,
        rel_path=path.name,
        sha256=image_id,
        width=width,
        height=height,
        format="PNG",
    )


def test_stitch_complementary_crops(tmp_path: Path) -> None:
    base = tmp_path / "base.png"
    _textured_image(base, 400, 400, 7)
    image = Image.open(base).convert("RGB")
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    image.crop((0, 0, 260, 400)).save(left)
    image.crop((140, 0, 400, 400)).save(right)

    records = [_record(left, "left"), _record(right, "right")]
    config = SameSourceConfig(
        min_score=0.1,
        min_inliers=8,
        use_mask=False,
        stitch=True,
    )

    result = stitch_group(records, records[0], config)

    assert result is not None
    assert result.used is True
    assert result.image.width > 300
    assert result.image.height >= 380


def test_run_same_source_dedup_stitches(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    base = tmp_path / "base.png"
    _textured_image(base, 400, 400, 11)
    image = Image.open(base).convert("RGB")
    image.crop((0, 0, 260, 400)).save(input_dir / "left.png")
    image.crop((140, 0, 400, 400)).save(input_dir / "right.png")

    config = RunConfig()
    config.source.min_score = 0.1
    config.source.min_inliers = 8
    config.source.linkage = "single"
    config.source.star_merge = False
    config.source.use_mask = False
    config.source.verify = False
    config.source.stitch = True

    _, groups = run_same_source_dedup(
        input_dir,
        tmp_path / "out",
        config,
        progress=False,
    )

    assert len(groups) == 1
    source_dir = tmp_path / "out" / "sources" / "source_001"
    assert (source_dir / "canonical.png").exists()
    assert (source_dir / "canonical_single.png").exists()
    source = (source_dir / "source.json").read_text(encoding="utf-8")
    assert '"stitched": true' in source
