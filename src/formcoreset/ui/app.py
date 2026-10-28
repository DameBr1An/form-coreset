"""Minimal local review UI (pre-alpha)."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _gallery_items(run_dir: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for path in sorted(run_dir.rglob("canonical.*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            items.append((str(path), path.parent.name))
    if items:
        return items
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            items.append((str(path), path.parent.name))
    return items


def launch(run_dir: Path) -> None:
    run_dir = Path(run_dir).resolve()
    items = _gallery_items(run_dir)
    with gr.Blocks(title="FormCoreset review") as demo:
        gr.Markdown(f"# FormCoreset review\n\n`{run_dir}`")
        if items:
            gr.Gallery(value=items, label="Forms", columns=4, height=640)
        else:
            gr.Markdown("No images found in this run directory.")
        gr.Markdown(
            "Pre-alpha review UI. Identity corrections, merge/split, and "
            "canonical overrides are not wired yet."
        )
    demo.launch()
