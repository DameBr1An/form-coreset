#!/usr/bin/env python3
"""Create a synthetic same-source variant benchmark from public anime images.

Each source image gets several variants:
original, crop, resize, JPEG compression, padded background, brightness/contrast,
and a translucent effect overlay. The ground truth is the source image; no pose
labels are needed.

Images are local-only and must not be committed.

Usage:
    uv run python benchmarks/make_same_source_benchmark.py \
        --source-dir benchmarks/data/anime_miku_small/images/target \
        --sources 10 --out benchmarks/data/anime_same_source_small
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def _variant_images(image: Image.Image, rng: random.Random) -> dict[str, Image.Image]:
    width, height = image.size
    variants: dict[str, Image.Image] = {"original": image.copy()}

    margin_x = int(width * 0.075)
    margin_y = int(height * 0.075)
    variants["crop"] = image.crop((margin_x, margin_y, width - margin_x, height - margin_y))

    variants["resize"] = image.resize(
        (max(64, int(width * 0.7)), max(64, int(height * 0.7))),
        Image.Resampling.LANCZOS,
    )
    variants["jpeg"] = image.copy()

    background = tuple(rng.randint(0, 255) for _ in range(3))
    pad = int(max(width, height) * 0.08)
    padded = Image.new("RGB", (width + pad * 2, height + pad * 2), background)
    padded.paste(image, (pad, pad))
    variants["padded"] = padded

    bright = ImageEnhance.Contrast(
        ImageEnhance.Brightness(image).enhance(0.75)
    ).enhance(1.2)
    variants["bright"] = bright

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(18):
        x1 = rng.randint(0, width - 1)
        y1 = rng.randint(0, height - 1)
        x2 = min(width, x1 + rng.randint(20, max(21, width // 2)))
        y2 = min(height, y1 + rng.randint(20, max(21, height // 2)))
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 60)
        if rng.random() < 0.5:
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=color)
    variants["effect"] = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("benchmarks/data/anime_miku_small/images/target"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/data/anime_same_source_small"),
    )
    parser.add_argument("--sources", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir: Path = args.source_dir
    paths = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED
    )
    if len(paths) < args.sources:
        raise SystemExit(f"Need {args.sources} sources, found {len(paths)} in {source_dir}")

    out_dir: Path = args.out
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Output directory is not empty: {out_dir}\n"
                "Use --overwrite or choose another --out."
            )
        shutil.rmtree(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest: list[dict[str, object]] = []
    for source_index, source_path in enumerate(paths[: args.sources], start=1):
        source_id = f"src_{source_index:02d}"
        with Image.open(source_path) as raw:
            image = raw.convert("RGB")
        variants = _variant_images(image, rng)
        for variant_name, variant_image in variants.items():
            suffix = ".jpg" if variant_name == "jpeg" else ".png"
            filename = f"{source_id}_{variant_name}{suffix}"
            destination = images_dir / filename
            if variant_name == "jpeg":
                variant_image.save(destination, quality=50)
            else:
                variant_image.save(destination)
            manifest.append(
                {
                    "image_id": f"{source_id}_{variant_name}",
                    "source_id": source_id,
                    "variant": variant_name,
                    "path": destination.relative_to(out_dir).as_posix(),
                    "width": variant_image.width,
                    "height": variant_image.height,
                    "source_file": source_path.as_posix(),
                }
            )

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "sources": args.sources,
        "variants": sorted({row["variant"] for row in manifest}),
        "images": len(manifest),
        "seed": args.seed,
        "notes": "Synthetic same-source variants; images are local-only and not redistributed.",
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "# Synthetic same-source benchmark\n\n"
        "Each source image has variants: original, crop, resize, JPEG, padded\n"
        "background, brightness/contrast, and translucent effects. Ground truth is\n"
        "the source image. Images are local-only and must not be committed.\n",
        encoding="utf-8",
    )
    (out_dir / "NOTICE.md").write_text(
        "# Notice\n\n"
        "Derived from Danbooru images for local evaluation only. Not redistributed.\n",
        encoding="utf-8",
    )
    print(f"Created {len(manifest)} images from {args.sources} sources at {out_dir}")


if __name__ == "__main__":
    main()
