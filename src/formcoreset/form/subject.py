"""Generic subject form encoder.

For non-face subjects (stylized characters, mascots, products, objects), form is
represented by:

    DINOv2(masked subject crop) + silhouette shape features

This is deliberately label-free and domain-agnostic. Anime tag priors can be
added by using the separate ``anime-hybrid`` encoder.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..config import FormConfig
from ..mask import MaskProvider, get_mask_provider
from ..types import ImageRecord
from .base import FormFeatures

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}


class SubjectEncodingError(RuntimeError):
    """Raised when the generic subject encoder cannot process an image."""


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_dino(model_name: str, device: str) -> tuple[Any, Any, Any]:
    import timm
    import torch

    key = (model_name, device)
    if key not in _MODEL_CACHE:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model.eval().to(device)
        config = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**config)
        _MODEL_CACHE[key] = (model, transform, torch)
    return _MODEL_CACHE[key]


def _dino_embedding(image: Image.Image, model_name: str, device: str) -> np.ndarray:
    model, transform, torch = _load_dino(model_name, device)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(tensor)
    return np.asarray(feature.detach().cpu().numpy(), dtype=np.float32).reshape(-1)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _silhouette_features(mask: np.ndarray, *, grid: int = 16) -> np.ndarray:
    """Compact shape descriptor: occupancy grid + area/aspect/centroid."""

    mask_image = Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L")
    bbox = mask_image.getbbox()
    if bbox is None:
        return np.zeros(grid * grid + 5, dtype=np.float32)

    crop = mask_image.crop(bbox).resize(
        (grid, grid),
        Image.Resampling.BILINEAR,
    )
    occupancy = np.asarray(crop, dtype=np.float32).reshape(-1) / 255.0
    binary = np.asarray(mask_image, dtype=np.uint8) > 0
    height, width = mask_image.height, mask_image.width
    bbox_width = max(bbox[2] - bbox[0], 1)
    bbox_height = max(bbox[3] - bbox[1], 1)
    area = float(binary.mean())
    aspect = bbox_width / bbox_height
    extent = float((np.asarray(crop) > 127).mean())
    rows, cols = np.nonzero(binary)
    centroid_x = float(cols.mean() / max(width, 1)) if len(cols) else 0.5
    centroid_y = float(rows.mean() / max(height, 1)) if len(rows) else 0.5
    extras = np.asarray(
        [area, aspect, extent, centroid_x, centroid_y],
        dtype=np.float32,
    )
    return _l2_normalize(np.concatenate([occupancy, extras]))


def _masked_image(path: Path, mask: np.ndarray | None) -> Image.Image:
    with Image.open(path) as raw:
        image = raw.convert("RGB")
    if mask is None:
        return image
    mask_image = Image.fromarray(mask, mode="L")
    bbox = mask_image.getbbox()
    if bbox is None:
        return image
    background = Image.new("RGB", image.size, (255, 255, 255))
    return Image.composite(image, background, mask_image).crop(bbox)


class SubjectFormEncoder:
    """Domain-agnostic encoder for non-face characters and objects."""

    name = "subject"
    version = "subject-dino-silhouette-v1"

    def __init__(
        self,
        config: FormConfig | None = None,
        *,
        mask_provider: MaskProvider | None = None,
        device: str = "auto",
    ) -> None:
        if importlib.util.find_spec("timm") is None or importlib.util.find_spec("torch") is None:
            raise ImportError(
                "The subject encoder requires timm and torch. "
                "Install them with: uv sync --extra subject"
            )
        self.config = config or FormConfig()
        total = (
            self.config.subject_dino_weight
            + self.config.subject_silhouette_weight
        )
        if total <= 0:
            raise ValueError("At least one subject feature weight must be positive")
        self._dino_weight = self.config.subject_dino_weight / total
        self._silhouette_weight = self.config.subject_silhouette_weight / total
        self._device = _resolve_device(device)
        self._mask_provider = mask_provider or get_mask_provider(
            self.config.mask_provider,
            scale=self.config.mask_scale,
        )

    def encode(self, record: ImageRecord) -> FormFeatures:
        mask = self._mask_provider.mask(record.path)
        image = _masked_image(record.path, mask)
        try:
            dino = _l2_normalize(
                _dino_embedding(
                    image,
                    self.config.subject_dino_model,
                    self._device,
                )
            )
        except Exception as exc:  # noqa: BLE001 - include image context
            raise SubjectEncodingError(
                f"DINOv2 failed for {record.path}: {exc}"
            ) from exc

        silhouette_used = mask is not None
        if silhouette_used:
            silhouette = _silhouette_features(mask)
            feature = np.concatenate(
                [
                    dino * self._dino_weight,
                    silhouette * self._silhouette_weight,
                ]
            )
        else:
            feature = dino

        return FormFeatures(
            embedding=np.asarray(feature, dtype=np.float32),
            confidence=None,
            method=self.version,
            metadata={
                "encoder": self.version,
                "mask_provider": self._mask_provider.name,
                "mask_used": silhouette_used,
                "embedding_used": True,
                "dino_model": self.config.subject_dino_model,
                "silhouette_used": silhouette_used,
                "feature_dim": int(feature.shape[0]),
            },
        )
