import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance

from formcoreset.config import RunConfig, SameSourceConfig
from formcoreset.pipeline import run_same_source_dedup
from formcoreset.source import (
    group_same_source,
    match_pair,
    partition_by_scores,
    star_merge_partitions,
)
from formcoreset.types import ImageRecord

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None,
    reason="OpenCV is required for same-source deduplication",
)


def _textured_image(path: Path, seed: int) -> None:
    rng = random.Random(seed)
    image = Image.new("RGB", (320, 320), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    for _ in range(40):
        x = rng.randint(0, 260)
        y = rng.randint(0, 260)
        width = rng.randint(20, 60)
        height = rng.randint(20, 60)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        if rng.random() < 0.5:
            draw.rectangle([x, y, x + width, y + height], fill=color, outline="black")
        else:
            draw.ellipse([x, y, x + width, y + height], fill=color, outline="black")
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
        format=path.suffix.lstrip(".").upper(),
    )


def _make_variants(root: Path) -> list[ImageRecord]:
    root.mkdir(parents=True, exist_ok=True)
    base = root / "base.png"
    _textured_image(base, 42)
    image = Image.open(base).convert("RGB")

    paths = [base]
    image.crop((32, 32, 288, 288)).save(root / "crop.png")
    paths.append(root / "crop.png")
    image.resize((208, 208)).save(root / "resize.png")
    paths.append(root / "resize.png")
    image.save(root / "compressed.jpg", quality=50)
    paths.append(root / "compressed.jpg")
    padded = Image.new("RGB", (420, 420), (20, 80, 160))
    padded.paste(image, (50, 50))
    padded.save(root / "padded.png")
    paths.append(root / "padded.png")
    ImageEnhance.Brightness(image).enhance(0.7).save(root / "bright.png")
    paths.append(root / "bright.png")

    unrelated = root / "unrelated.png"
    _textured_image(unrelated, 99)
    paths.append(unrelated)

    return [_record(path, path.stem) for path in paths]


def test_match_pair_and_grouping(tmp_path: Path) -> None:
    records = _make_variants(tmp_path / "in")
    config = SameSourceConfig(
        min_score=0.1,
        min_inliers=8,
        max_features=2000,
        linkage="single",
        star_merge=False,
        use_mask=False,
        verify=False,
    )

    base = records[0]
    crop = records[1]
    result = match_pair(base.path, crop.path, config)
    assert result.inliers >= 8
    assert result.score >= 0.1

    groups = group_same_source(records, config)
    sizes = sorted((len(group.members) for group in groups), reverse=True)

    assert len(groups) == 2
    assert sizes[0] == 6


def test_partition_complete_linkage_prevents_chaining() -> None:
    scores = np.asarray(
        [
            [0.0, 0.9, 0.1],
            [0.9, 0.0, 0.9],
            [0.1, 0.9, 0.0],
        ],
        dtype=np.float32,
    )

    single = partition_by_scores(
        ["a", "b", "c"],
        scores,
        linkage_method="single",
        threshold=0.5,
    )
    complete = partition_by_scores(
        ["a", "b", "c"],
        scores,
        linkage_method="complete",
        threshold=0.5,
    )

    assert len(single) == 1
    assert len(complete) == 2
    assert sorted(len(group) for group in complete) == [1, 2]
    assert all(len(group) < 3 for group in complete)


def test_star_merge_reattaches_singleton_to_medoid() -> None:
    image_ids = ["a", "b", "d", "c"]
    scores = np.zeros((4, 4), dtype=np.float32)
    for left, right, value in [
        ("a", "b", 0.9),
        ("b", "d", 0.9),
        ("a", "d", 0.1),
        ("c", "b", 0.8),
        ("c", "a", 0.1),
        ("c", "d", 0.1),
    ]:
        i, j = image_ids.index(left), image_ids.index(right)
        scores[i, j] = scores[j, i] = value

    merged = star_merge_partitions(
        [["a", "b", "d"], ["c"]],
        image_ids,
        scores,
        threshold=0.7,
    )

    assert len(merged) == 1
    assert set(merged[0]) == {"a", "b", "c", "d"}


def test_star_merge_keeps_low_confidence_singleton() -> None:
    image_ids = ["a", "b", "c"]
    scores = np.zeros((3, 3), dtype=np.float32)
    scores[0, 1] = scores[1, 0] = 0.9
    scores[1, 2] = scores[2, 1] = 0.5

    groups = star_merge_partitions(
        [["a", "b"], ["c"]],
        image_ids,
        scores,
        threshold=0.7,
    )

    assert len(groups) == 2


def test_star_merge_does_not_merge_groups() -> None:
    image_ids = ["a", "b", "c", "d"]
    scores = np.zeros((4, 4), dtype=np.float32)
    scores[0, 1] = scores[1, 0] = 0.9
    scores[2, 3] = scores[3, 2] = 0.9
    scores[1, 2] = scores[2, 1] = 0.9

    groups = star_merge_partitions(
        [["a", "b"], ["c", "d"]],
        image_ids,
        scores,
        threshold=0.5,
    )

    assert len(groups) == 2


def test_run_same_source_dedup(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    records = _make_variants(input_dir)
    config = RunConfig()
    config.source.min_score = 0.1
    config.source.min_inliers = 8
    config.source.linkage = "single"
    config.source.star_merge = False
    config.source.use_mask = False
    config.source.verify = False
    config.source.stitch = False

    scan, groups = run_same_source_dedup(
        input_dir,
        tmp_path / "out",
        config,
        progress=False,
    )

    assert len(scan.records) == len(records)
    assert len(groups) == 2
    out_dir = tmp_path / "out"
    assert (out_dir / "sources").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "manifest.jsonl").exists()
    assert (out_dir / "contact_sheet.png").exists()
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Label" not in report
    assert "Top tags" not in report
