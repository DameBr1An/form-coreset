from pathlib import Path

import pytest

from formcoreset.mask import (
    AnimeMaskProvider,
    AutoMaskProvider,
    GeneralMaskProvider,
    NoneMaskProvider,
    get_mask_provider,
)


def test_none_mask_provider(tmp_path: Path) -> None:
    provider = NoneMaskProvider()
    assert provider.mask(tmp_path / "missing.png") is None


def test_mask_registry() -> None:
    assert isinstance(get_mask_provider("none"), NoneMaskProvider)
    assert isinstance(get_mask_provider("anime"), AnimeMaskProvider)
    assert isinstance(get_mask_provider("general"), GeneralMaskProvider)
    assert isinstance(get_mask_provider("auto"), AutoMaskProvider)


def test_unknown_mask_provider() -> None:
    with pytest.raises(ValueError):
        get_mask_provider("unknown")
