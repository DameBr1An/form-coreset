"""Pluggable subject mask providers.

The core pipeline should not know whether the subject is an anime character, a
product, or a mascot. Mask backends are adapters:

- ``none``: no mask (full image)
- ``anime``: isnet-anime via imgutils
- ``general``: BiRefNet / rembg
- ``auto``: prefer general, then anime, then none

All backends return a uint8 mask (0/255) at the original image size, or None.
"""

from __future__ import annotations

import importlib.util
import logging
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


class MaskProvider(Protocol):
    name: str

    def mask(self, path: Path) -> np.ndarray | None:
        ...


class NoneMaskProvider:
    name = "none"

    def mask(self, path: Path) -> np.ndarray | None:
        return None


class AnimeMaskProvider:
    name = "anime"

    def __init__(self, scale: int = 1024) -> None:
        self.scale = scale

    def mask(self, path: Path) -> np.ndarray | None:
        try:
            from imgutils.segment import get_isnetis_mask
        except ImportError:
            logger.warning("imgutils is not installed; anime mask unavailable")
            return None

        try:
            mask = np.asarray(
                get_isnetis_mask(path, scale=self.scale),
                dtype=np.float32,
            )
        except Exception as exc:  # noqa: BLE001 - optional backend
            logger.warning("anime segmentation failed for %s: %s", path, exc)
            return None

        if mask.ndim != 2 or mask.size == 0:
            return None
        foreground = float((mask >= 0.5).mean())
        if foreground < 0.005 or foreground > 0.995:
            return None
        binary = (mask >= 0.5).astype(np.uint8) * 255
        return np.asarray(
            Image.fromarray(binary, mode="L").filter(ImageFilter.MaxFilter(5)),
            dtype=np.uint8,
        )


class GeneralMaskProvider:
    name = "general"

    def __init__(self, model_name: str = "birefnet-general") -> None:
        self.model_name = model_name
        self._session = None

    def _get_session(self) -> object | None:
        try:
            from rembg import new_session
        except ImportError:
            logger.warning("rembg is not installed; general mask unavailable")
            return None
        if self._session is None:
            try:
                self._session = new_session(self.model_name)
            except Exception as exc:  # noqa: BLE001 - optional backend
                logger.warning(
                    "rembg session failed for %s: %s",
                    self.model_name,
                    exc,
                )
                return None
        return self._session

    def mask(self, path: Path) -> np.ndarray | None:
        session = self._get_session()
        if session is None:
            return None
        try:
            from rembg import remove

            with Image.open(path) as image:
                result = remove(image.convert("RGBA"), session=session)
        except Exception as exc:  # noqa: BLE001 - optional backend
            logger.warning("general segmentation failed for %s: %s", path, exc)
            return None

        alpha = np.asarray(result)
        if alpha.ndim != 3 or alpha.shape[-1] < 4:
            return None
        mask = alpha[..., 3]
        if mask.max() == 0:
            return None
        return mask


class AutoMaskProvider:
    name = "auto"

    def __init__(self, scale: int = 1024) -> None:
        self.scale = scale
        self._providers: list[MaskProvider] = []
        if importlib.util.find_spec("rembg") is not None:
            self._providers.append(GeneralMaskProvider())
        if importlib.util.find_spec("imgutils") is not None:
            self._providers.append(AnimeMaskProvider(scale=scale))

    def mask(self, path: Path) -> np.ndarray | None:
        for provider in self._providers:
            mask = provider.mask(path)
            if mask is not None:
                return mask
        return None


@lru_cache(maxsize=8)
def get_mask_provider(
    name: str = "auto",
    *,
    scale: int = 1024,
    general_model: str = "birefnet-general",
) -> MaskProvider:
    normalized = name.lower().strip()
    if normalized in {"none", "off", "false"}:
        return NoneMaskProvider()
    if normalized == "anime":
        return AnimeMaskProvider(scale=scale)
    if normalized in {"general", "object", "photo"}:
        return GeneralMaskProvider(model_name=general_model)
    if normalized == "auto":
        return AutoMaskProvider(scale=scale)
    raise ValueError(f"Unsupported mask provider: {name}")
