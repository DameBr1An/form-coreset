from pathlib import Path

import numpy as np
from PIL import Image

from formcoreset.cluster import compute_hashes, group_near_duplicates, hamming_distance
from formcoreset.ingest import scan_directory


def _gradient(path: Path, horizontal: bool, size: int = 64) -> None:
    array = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            value = int(255 * (x if horizontal else y) / (size - 1))
            array[y, x] = (value, value, value)
    Image.fromarray(array, mode="RGB").save(path)


def test_hamming_distance_identity() -> None:
    assert hamming_distance("ffff", "ffff") == 0
    assert hamming_distance("0000", "ffff") == 16


def test_group_near_duplicates(tmp_path: Path) -> None:
    _gradient(tmp_path / "a.png", horizontal=True)
    _gradient(tmp_path / "a_copy.png", horizontal=True)
    _gradient(tmp_path / "b.png", horizontal=False)

    scan = scan_directory(tmp_path)
    hashes = compute_hashes(scan.records)
    groups = group_near_duplicates(scan.records, hashes, threshold=5)

    assert len(groups) == 2
    sizes = sorted(len(group) for group in groups)
    assert sizes == [1, 2]
    assert all(len({record.image_id for record in group}) == len(group) for group in groups)
