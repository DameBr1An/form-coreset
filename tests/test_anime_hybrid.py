from pathlib import Path

import numpy as np
import pytest

import formcoreset.form.anime_hybrid as anime_module
from formcoreset.config import FormConfig
from formcoreset.form.anime_hybrid import (
    AnimeHybridFormEncoder,
    FormEncodingError,
    build_tag_vector,
)
from formcoreset.form.vocab import TAG_VOCAB, form_label
from formcoreset.types import ImageRecord


def _record(tmp_path: Path) -> ImageRecord:
    return ImageRecord(
        image_id="img_0000",
        path=tmp_path / "a.png",
        rel_path="a.png",
        sha256="abc",
        width=512,
        height=512,
        format="PNG",
    )


def test_build_tag_vector() -> None:
    tags = {"standing": 0.9, "sitting": 0.1, "full_body": 0.8}

    vector = build_tag_vector(tags, 0.2)

    assert vector.shape == (len(TAG_VOCAB),)
    assert vector[TAG_VOCAB.index("standing")] == pytest.approx(0.7)
    assert vector[TAG_VOCAB.index("sitting")] == pytest.approx(0.0)
    assert vector[TAG_VOCAB.index("full_body")] == pytest.approx(0.6)


def test_encoder_hybrid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        anime_module,
        "_get_wd14_tags",
        lambda path, **kwargs: (
            {"general": 0.99},
            {"standing": 0.9, "full_body": 0.8},
            {"miku": 0.95},
        ),
    )
    monkeypatch.setattr(
        anime_module,
        "_ccip_extract_feature",
        lambda path: np.ones(8, dtype=np.float32),
    )
    config = FormConfig(
        tag_weight=0.7,
        embedding_weight=0.3,
        general_threshold=0.2,
        tag_threshold=0.2,
    )

    features = AnimeHybridFormEncoder(config).encode(_record(tmp_path))

    assert features.embedding.shape == (len(TAG_VOCAB) + 8,)
    assert features.metadata["embedding_used"] is True
    assert features.tags["standing"] == 0.9
    assert features.metadata["top_tags"][0]["tag"] == "standing"


def test_encoder_falls_back_to_tags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        anime_module,
        "_get_wd14_tags",
        lambda path, **kwargs: ({}, {"standing": 0.9}, {}),
    )

    def _fail(path: Path) -> np.ndarray:
        raise RuntimeError("ccip unavailable")

    monkeypatch.setattr(anime_module, "_ccip_extract_feature", _fail)
    config = FormConfig(tag_weight=0.7, embedding_weight=0.3)

    features = AnimeHybridFormEncoder(config).encode(_record(tmp_path))

    assert features.metadata["embedding_used"] is False
    assert features.embedding.shape == (len(TAG_VOCAB),)


def test_form_label_prioritizes_posture() -> None:
    scores = {
        "holding": 0.95,
        "looking_at_viewer": 0.9,
        "sitting": 0.8,
        "full_body": 0.7,
    }

    assert form_label(scores) == "sitting / looking_at_viewer / holding"


def test_form_label_falls_back_to_generic_tags() -> None:
    assert form_label({"custom_tag": 0.9}) == "custom_tag"


def test_encoder_wd14_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fail(path: Path, **kwargs: object) -> object:
        raise RuntimeError("wd14 unavailable")

    monkeypatch.setattr(anime_module, "_get_wd14_tags", _fail)

    with pytest.raises(FormEncodingError):
        AnimeHybridFormEncoder().encode(_record(tmp_path))
