import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import formcoreset.form.subject as subject_module
from formcoreset.config import FormConfig
from formcoreset.form.subject import SubjectEncodingError, SubjectFormEncoder
from formcoreset.types import ImageRecord

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("timm") is None,
    reason="subject encoder requires torch and timm",
)


class FakeMaskProvider:
    name = "fake"

    def mask(self, path: Path) -> np.ndarray:
        return np.full((64, 64), 255, dtype=np.uint8)


def _record(tmp_path: Path) -> ImageRecord:
    path = tmp_path / "a.png"
    Image.new("RGB", (64, 64), (120, 120, 120)).save(path)
    return ImageRecord(
        image_id="img_0000",
        path=path,
        rel_path="a.png",
        sha256="abc",
        width=64,
        height=64,
        format="PNG",
    )


def test_subject_encoder_with_mask(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        subject_module,
        "_dino_embedding",
        lambda image, model_name, device: np.ones(8, dtype=np.float32),
    )
    monkeypatch.setattr(
        subject_module,
        "_silhouette_features",
        lambda mask, grid=16: np.ones(5, dtype=np.float32),
    )
    config = FormConfig(
        encoder="subject",
        subject_dino_weight=0.7,
        subject_silhouette_weight=0.3,
    )

    encoder = SubjectFormEncoder(config, mask_provider=FakeMaskProvider())
    features = encoder.encode(_record(tmp_path))

    assert features.embedding.shape == (13,)
    assert features.metadata["mask_used"] is True
    assert features.metadata["silhouette_used"] is True
    assert features.metadata["mask_provider"] == "fake"


def test_subject_encoder_without_mask(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        subject_module,
        "_dino_embedding",
        lambda image, model_name, device: np.ones(8, dtype=np.float32),
    )
    config = FormConfig(encoder="subject", mask_provider="none")

    encoder = SubjectFormEncoder(config)
    features = encoder.encode(_record(tmp_path))

    assert features.embedding.shape == (8,)
    assert features.metadata["mask_used"] is False


def test_subject_encoder_dino_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fail(image: Image.Image, model_name: str, device: str) -> np.ndarray:
        raise RuntimeError("dino unavailable")

    monkeypatch.setattr(subject_module, "_dino_embedding", _fail)
    encoder = SubjectFormEncoder(
        FormConfig(encoder="subject", mask_provider="none")
    )

    with pytest.raises(SubjectEncodingError):
        encoder.encode(_record(tmp_path))
