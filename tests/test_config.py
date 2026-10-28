from pathlib import Path

from formcoreset.config import RunConfig, load_config


def test_defaults() -> None:
    config = RunConfig()

    assert config.adapter == "auto"
    assert config.target.mode == "guided"
    assert config.identity.confirmed_threshold == 0.75
    assert config.selection.max_per_form == 3


def test_load_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "target:\n"
        "  name: test\n"
        "adapter: person\n"
        "selection:\n"
        "  max_per_form: 2\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.target.name == "test"
    assert config.adapter == "person"
    assert config.selection.max_per_form == 2
