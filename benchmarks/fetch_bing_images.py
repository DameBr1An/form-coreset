#!/usr/bin/env python3
"""Fetch the first N Bing image results for a query, for local analysis.

Images are saved under the output directory (gitignored) and are not
redistributed. The script records the original image/page URLs.

Usage:
    uv run python benchmarks/fetch_bing_images.py \
        --query "无畏契约 捷风" --limit 30 \
        --out benchmarks/data/jett_bing
"""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import re
import shutil
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
BING_URL = "https://www.bing.com/images/search"


def _fetch_html(query: str, count: int, market: str) -> str:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "first": 1,
            "count": count,
            "mkt": market,
            "form": "HDRSC3",
        }
    )
    language = "zh-CN,zh;q=0.9,en;q=0.8" if market.lower().startswith("zh") else "en-US,en;q=0.9"
    request = urllib.request.Request(
        f"{BING_URL}?{params}",
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": language,
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="ignore")


def _extract_results(html_text: str, limit: int) -> list[dict[str, str]]:
    decoded = html.unescape(html_text)

    def values(key: str) -> list[str]:
        return [
            match.group(1)
            for match in re.finditer(rf'"{key}":"(.*?)"', decoded)
            if match.group(1)
        ]

    murls = values("murl")
    turls = values("turl")
    purls = values("purl")
    titles = values("t")

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, murl in enumerate(murls):
        if not murl.startswith(("http://", "https://")):
            continue
        if murl in seen:
            continue
        seen.add(murl)
        results.append(
            {
                "murl": murl,
                "turl": turls[index] if index < len(turls) else "",
                "purl": purls[index] if index < len(purls) else "",
                "title": titles[index] if index < len(titles) else "",
            }
        )
        if len(results) >= limit:
            break
    return results


def _download(url: str, destination: Path, *, referer: str) -> tuple[int, int, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": referer,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        image_format = (image.format or "PNG").upper()
        if image_format == "GIF":
            image.seek(0)
            frame = image.convert("RGB")
            frame.save(destination.with_suffix(".png"))
            return width, height, "PNG"
    extension = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "AVIF": ".avif",
    }.get(image_format, ".img")
    destination = destination.with_suffix(extension)
    destination.write_bytes(data)
    return width, height, image_format


def _write_contact_sheet(
    out_dir: Path,
    rows: list[dict[str, Any]],
    path: Path,
    *,
    thumb: int = 220,
    cols: int = 5,
) -> None:
    if not rows:
        return
    cell = thumb + 30
    grid_rows = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * cell, grid_rows * cell), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        try:
            with Image.open(out_dir / row["path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb, thumb))
                x = (index % cols) * cell + (cell - image.width) // 2
                y = (index // cols) * cell + 22 + (thumb - image.height) // 2
                sheet.paste(image, (x, y))
        except Exception:  # noqa: BLE001 - one bad image should not break the sheet
            pass
        draw.text(
            ((index % cols) * cell + 4, (index // cols) * cell + 4),
            f"{row['image_id']} {row['width']}x{row['height']}",
            fill=(20, 20, 20),
        )
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--mkt", default="zh-CN", help="Bing market, e.g. zh-CN or en-US.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

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

    print(f"Fetching Bing images for {args.query!r} ...")
    html_text = _fetch_html(args.query, args.limit + 10, args.mkt)
    (out_dir / "bing.html").write_text(html_text, encoding="utf-8")
    results = _extract_results(html_text, args.limit)
    print(f"Found {len(results)} unique image URLs; downloading ...")

    rows: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        image_id = f"bing_{index:03d}"
        destination = images_dir / image_id
        downloaded_from = "murl"
        try:
            width, height, image_format = _download(
                result["murl"],
                destination,
                referer=result["purl"] or "https://www.bing.com/",
            )
        except Exception as exc:  # noqa: BLE001 - fall back to Bing thumbnail
            print(f"  full image failed {index}: {exc}")
            if not result["turl"]:
                continue
            try:
                width, height, image_format = _download(
                    result["turl"],
                    destination,
                    referer="https://www.bing.com/",
                )
                downloaded_from = "turl"
            except Exception as thumbnail_exc:  # noqa: BLE001 - skip broken result
                print(f"  thumbnail failed {index}: {thumbnail_exc}")
                continue

        saved = next(images_dir.glob(f"{image_id}.*"), None)
        if saved is None:
            continue
        rows.append(
            {
                "image_id": image_id,
                "path": saved.relative_to(out_dir).as_posix(),
                "width": width,
                "height": height,
                "format": image_format,
                "downloaded_from": downloaded_from,
                "murl": result["murl"],
                "turl": result["turl"],
                "purl": result["purl"],
                "title": result["title"],
                "query": args.query,
            }
        )
        print(f"  {index}/{args.limit}: {width}x{height}")
        time.sleep(0.2)

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "query": args.query,
        "requested": args.limit,
        "downloaded": len(rows),
        "source": "bing_images",
        "notes": "Local analysis only; images are not redistributed.",
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_contact_sheet(out_dir, rows[:15], out_dir / "contact_sheet_1.png")
    _write_contact_sheet(out_dir, rows[15:], out_dir / "contact_sheet_2.png")
    (out_dir / "README.md").write_text(
        f"# Bing image sample: {args.query}\n\n"
        f"Downloaded {len(rows)} images for local analysis. Images are not\n"
        "redistributed and remain the property of their respective owners.\n",
        encoding="utf-8",
    )
    (out_dir / "NOTICE.md").write_text(
        "# Notice\n\n"
        "Images were fetched from Bing image search for local, non-redistributed\n"
        "analysis. Respect the original sites and artists' terms. Do not commit or\n"
        "publish the images.\n",
        encoding="utf-8",
    )
    print(f"Done: {len(rows)} images at {out_dir}")


if __name__ == "__main__":
    main()
