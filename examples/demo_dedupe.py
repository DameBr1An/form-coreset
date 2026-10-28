#!/usr/bin/env python3
"""Generate a synthetic same-source demo and a README image.

No copyrighted images are used. The demo creates three synthetic posters,
their crop/background/effect variants, runs `dedupe`, and writes
`docs/assets/demo.png`.

Usage:
    uv run python examples/demo_dedupe.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance

from formcoreset.config import RunConfig
from formcoreset.pipeline import run_same_source_dedup

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "examples" / "demo_data"
OUT_DIR = ROOT / "examples" / "demo_output"
ASSET = ROOT / "docs" / "assets" / "demo.png"

PALETTES = [
    {"bg": (24, 42, 72), "panel": (44, 92, 150), "body": (70, 170, 220), "accent": (250, 120, 90)},
    {"bg": (50, 32, 24), "panel": (120, 72, 44), "body": (240, 170, 80), "accent": (80, 190, 150)},
    {
        "bg": (32, 48, 36),
        "panel": (60, 120, 80),
        "body": (150, 220, 130),
        "accent": (240, 110, 180),
    },
]


def make_poster(path: Path, seed: int, palette: dict[str, tuple[int, int, int]]) -> None:
    rng = random.Random(seed)
    width, height = 720, 960
    image = Image.new("RGB", (width, height), palette["bg"])
    draw = ImageDraw.Draw(image)

    for _ in range(80):
        x = rng.randint(0, width)
        y = rng.randint(0, height)
        size = rng.randint(10, 70)
        color = tuple(
            max(0, min(255, value + rng.randint(-45, 45)))
            for value in palette["panel"]
        )
        draw.rectangle([x, y, x + size, y + size // 2], fill=color)

    draw.rounded_rectangle(
        [40, 40, width - 40, height - 40],
        radius=24,
        fill=palette["panel"],
        outline=(240, 240, 240),
        width=6,
    )
    draw.text((70, 65), f"SYNTHETIC POSTER {seed}", fill=(245, 245, 245))

    head = (width // 2, 300)
    draw.ellipse(
        [head[0] - 100, head[1] - 100, head[0] + 100, head[1] + 100],
        fill=palette["body"],
        outline=(20, 20, 20),
        width=5,
    )
    draw.ellipse([head[0] - 45, head[1] - 20, head[0] - 15, head[1] + 10], fill=(20, 20, 20))
    draw.ellipse([head[0] + 15, head[1] - 20, head[0] + 45, head[1] + 10], fill=(20, 20, 20))
    draw.line([head[0], head[1] - 100, head[0], head[1] - 150], fill=palette["accent"], width=10)
    draw.ellipse([head[0] - 18, head[1] - 170, head[0] + 18, head[1] - 134], fill=palette["accent"])

    draw.rounded_rectangle(
        [head[0] - 130, head[1] + 105, head[0] + 130, head[1] + 390],
        radius=36,
        fill=palette["body"],
        outline=(20, 20, 20),
        width=5,
    )
    draw.line(
        [head[0] - 130, head[1] + 160, head[0] - 240, head[1] + 300],
        fill=palette["body"],
        width=36,
    )
    draw.line(
        [head[0] + 130, head[1] + 160, head[0] + 240, head[1] + 300],
        fill=palette["body"],
        width=36,
    )
    draw.line(
        [head[0] - 70, head[1] + 390, head[0] - 110, head[1] + 600],
        fill=palette["body"],
        width=42,
    )
    draw.line(
        [head[0] + 70, head[1] + 390, head[0] + 110, head[1] + 600],
        fill=palette["body"],
        width=42,
    )

    for _ in range(35):
        x = rng.randint(80, width - 120)
        y = rng.randint(200, height - 150)
        size = rng.randint(8, 36)
        color = tuple(rng.randint(30, 255) for _ in range(3))
        draw.rectangle([x, y, x + size, y + size], fill=color)

    image.save(path)


def variants(image: Image.Image, rng: random.Random) -> dict[str, Image.Image]:
    width, height = image.size
    result = {"original": image.copy()}
    margin_x, margin_y = int(width * 0.08), int(height * 0.08)
    result["crop"] = image.crop((margin_x, margin_y, width - margin_x, height - margin_y))
    result["resize"] = image.resize((int(width * 0.7), int(height * 0.7)))
    pad = int(max(width, height) * 0.08)
    background = tuple(rng.randint(30, 220) for _ in range(3))
    padded = Image.new(
        "RGB",
        (width + pad * 2, height + pad * 2),
        background,
    )
    padded.paste(image, (pad, pad))
    result["padded"] = padded
    result["bright"] = ImageEnhance.Contrast(
        ImageEnhance.Brightness(image).enhance(0.75)
    ).enhance(1.2)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(14):
        x, y = rng.randint(0, width - 1), rng.randint(0, height - 1)
        draw.ellipse(
            [x, y, min(width, x + rng.randint(30, 160)), min(height, y + rng.randint(30, 160))],
            fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 55),
        )
    result["effect"] = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return result


def make_sheet(
    rows: list[tuple[str, Path]],
    groups: list[tuple[str, list[Path]]],
    path: Path,
) -> None:
    thumb = 170
    cell = thumb + 24
    cols = 6
    top_rows = (len(rows) + cols - 1) // cols
    bottom_rows = len(groups)
    header = 70
    sheet = Image.new(
        "RGB",
        (cols * cell, header + top_rows * cell + 50 + bottom_rows * cell),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "INPUT: same-source variants", fill=(20, 20, 20))

    for index, (label, image_path) in enumerate(rows):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb, thumb))
            x = (index % cols) * cell + (cell - image.width) // 2
            y = header + (index // cols) * cell + (thumb - image.height) // 2
            sheet.paste(image, (x, y))
        draw.text(
            ((index % cols) * cell + 4, header + (index // cols) * cell + 2),
            label,
            fill=(0, 0, 0),
        )

    offset = header + top_rows * cell + 35
    draw.text((10, offset), "OUTPUT: one canonical + variants per source group", fill=(20, 20, 20))
    offset += 25
    for row, (label, image_paths) in enumerate(groups):
        for col, image_path in enumerate(image_paths[:cols]):
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb, thumb))
                x = col * cell + (cell - image.width) // 2
                y = offset + row * cell + (thumb - image.height) // 2
                sheet.paste(image, (x, y))
        draw.text((4, offset + row * cell + 2), label, fill=(0, 0, 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    image_dir = DATA_DIR / "images"
    image_dir.mkdir(parents=True)

    rng = random.Random(7)
    rows: list[tuple[str, Path]] = []
    for source_index, palette in enumerate(PALETTES, start=1):
        source = image_dir / f"src_{source_index:02d}_original.png"
        make_poster(source, source_index, palette)
        with Image.open(source) as raw:
            poster = raw.convert("RGB")
        for name, variant in variants(poster, rng).items():
            destination = image_dir / f"src_{source_index:02d}_{name}.png"
            variant.save(destination)
            rows.append((f"src_{source_index:02d} {name}", destination))

    config = RunConfig()
    config.source.use_mask = False
    config.source.min_score = 0.2
    config.source.linkage = "complete"
    config.source.star_threshold = 0.2
    config.source.stitch = True

    _, groups = run_same_source_dedup(DATA_DIR / "images", OUT_DIR, config, progress=False)

    output_groups: list[tuple[str, list[Path]]] = []
    for source_dir in sorted((OUT_DIR / "sources").glob("source_*")):
        canonical = sorted(source_dir.glob("canonical.*"))
        variants_dir = source_dir / "variants"
        members = sorted(variants_dir.glob("*")) if variants_dir.exists() else []
        output_groups.append((source_dir.name, canonical + members))

    make_sheet(rows, output_groups, ASSET)
    print(f"Input variants: {len(rows)}")
    print(f"Source groups: {len(groups)}")
    print(f"Demo sheet: {ASSET}")


if __name__ == "__main__":
    main()
