#!/usr/bin/env python3
"""Download a small anime identity/form benchmark from Danbooru.

The benchmark has two splits:
- target: images tagged with the target character (default: hatsune_miku)
- distractor: images tagged with another character (default: kagamine_rin)

Images are downloaded locally for evaluation only. They are copyrighted by
their respective artists and must not be committed or redistributed. Only the
script and metadata belong in the repository.

Usage:
    uv run python benchmarks/download_anime_benchmark.py \
        --target-tag hatsune_miku --distractor-tag kagamine_rin \
        --target-limit 50 --distractor-limit 20 \
        --out benchmarks/data/anime_miku_small
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

API_URL = "https://danbooru.donmai.us/posts.json"
USER_AGENT = "FormCoresetBenchmark/0.1 (local evaluation; +https://github.com/your-org/form-coreset)"
MIN_SIDE = 512

POSE_TAGS = {
    "standing",
    "sitting",
    "running",
    "walking",
    "lying",
    "kneeling",
    "squatting",
    "jumping",
    "dancing",
    "arms_up",
    "hand_up",
    "hands_up",
    "holding",
    "crossed_arms",
    "arm_up",
    "legs_crossed",
    "on_back",
    "on_stomach",
}

VIEW_TAGS = {
    "from_front",
    "from_side",
    "from_behind",
    "from_above",
    "from_below",
    "looking_at_viewer",
    "profile",
    "three-quarter_view",
    "full_body",
    "cowboy_shot",
    "upper_body",
    "portrait",
}


def _api_get(params: dict[str, Any], *, timeout: int = 30, retries: int = 3) -> Any:
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - retry API/network errors
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"Danbooru API failed: {last_error}") from last_error


def _fetch_posts(tag: str, wanted: int, *, sleep: float = 1.0) -> list[dict[str, Any]]:
    """Fetch enough candidate posts for a split, with duplicate protection."""

    posts: list[dict[str, Any]] = []
    seen: set[int] = set()
    page = 1
    while len(posts) < wanted * 3 and page <= 4:
        batch = _api_get(
            {
                "tags": f"{tag} solo rating:general",
                "limit": 200,
                "page": page,
            }
        )
        if not batch:
            break
        for post in batch:
            post_id = post.get("id")
            if post_id in seen:
                continue
            seen.add(post_id)
            posts.append(post)
            if len(posts) >= wanted * 3:
                break
        page += 1
        time.sleep(sleep)
    return posts


def _pick_url(post: dict[str, Any], image_field: str) -> str | None:
    if image_field == "file_url":
        return post.get("file_url")
    if image_field == "preview_file_url":
        return post.get("preview_file_url") or post.get("large_file_url")
    return post.get("large_file_url") or post.get("file_url")


def _download_image(
    url: str,
    destination: Path,
    *,
    timeout: int = 45,
    retries: int = 2,
) -> tuple[int, int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                width, height = image.size
                image_format = image.format or "UNKNOWN"
            if min(width, height) < MIN_SIDE:
                raise ValueError(f"image too small: {width}x{height}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            return width, height, image_format
        except Exception as exc:  # noqa: BLE001 - skip bad posts, retry transient
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"download failed: {last_error}") from last_error


def _row_for_post(
    post: dict[str, Any],
    *,
    split: str,
    relative_path: Path,
    downloaded_url: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    tags = set((post.get("tag_string") or "").split())
    return {
        "image_id": f"{split}_{post['id']}",
        "split": split,
        "path": relative_path.as_posix(),
        "source_site": "danbooru",
        "post_id": post["id"],
        "post_url": f"https://danbooru.donmai.us/posts/{post['id']}",
        "md5": post.get("md5"),
        "rating": post.get("rating"),
        "width": width,
        "height": height,
        "downloaded_url": downloaded_url,
        "file_url": post.get("file_url"),
        "artist_tags": (post.get("tag_string_artist") or "").split(),
        "character_tags": (post.get("tag_string_character") or "").split(),
        "copyright_tags": (post.get("tag_string_copyright") or "").split(),
        "weak_pose_tags": sorted(tags & POSE_TAGS),
        "weak_view_tags": sorted(tags & VIEW_TAGS),
        "tags": sorted(tags),
    }


def _download_split(
    posts: list[dict[str, Any]],
    *,
    split: str,
    out_dir: Path,
    limit: int,
    image_field: str,
    seen_md5: set[str],
    sleep: float,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    split_dir = out_dir / "images" / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for post in posts:
        if len(rows) >= limit:
            break
        if post.get("is_deleted") or post.get("is_banned") or post.get("is_flagged"):
            skipped.append({"post_id": str(post.get("id")), "reason": "deleted/banned/flagged"})
            continue
        if post.get("is_pending"):
            skipped.append({"post_id": str(post.get("id")), "reason": "pending"})
            continue
        if post.get("rating") != "g":
            skipped.append({"post_id": str(post.get("id")), "reason": "rating not general"})
            continue
        if (post.get("image_width") or 0) < MIN_SIDE or (post.get("image_height") or 0) < MIN_SIDE:
            skipped.append({"post_id": str(post.get("id")), "reason": "declared size too small"})
            continue
        md5 = post.get("md5")
        if md5 and md5 in seen_md5:
            skipped.append({"post_id": str(post.get("id")), "reason": "duplicate md5"})
            continue

        url = _pick_url(post, image_field)
        if not url:
            skipped.append({"post_id": str(post.get("id")), "reason": "no image URL"})
            continue

        extension = (post.get("file_ext") or "jpg").lower()
        destination = split_dir / f"danbooru_{post['id']}.{extension}"
        if not destination.exists():
            try:
                width, height, _ = _download_image(url, destination)
            except Exception as exc:  # noqa: BLE001 - report and continue
                skipped.append({"post_id": str(post.get("id")), "reason": str(exc)})
                continue
            time.sleep(sleep)
        else:
            try:
                with Image.open(destination) as image:
                    image.load()
                    width, height = image.size
            except Exception as exc:  # noqa: BLE001 - report and continue
                skipped.append({"post_id": str(post.get("id")), "reason": f"cached invalid: {exc}"})
                continue

        if min(width, height) < MIN_SIDE:
            destination.unlink(missing_ok=True)
            skipped.append({"post_id": str(post.get("id")), "reason": "downloaded size too small"})
            continue

        if md5:
            seen_md5.add(md5)
        relative = destination.relative_to(out_dir)
        rows.append(
            _row_for_post(
                post,
                split=split,
                relative_path=relative,
                downloaded_url=url,
                width=width,
                height=height,
            )
        )
        if len(rows) % 10 == 0:
            print(f"  {split}: {len(rows)}/{limit}")

    return rows, skipped


def _write_contact_sheet(
    out_dir: Path,
    rows: list[dict[str, Any]],
    path: Path,
    *,
    thumb: int = 200,
    cols: int = 8,
) -> None:
    if not rows:
        return
    cell = thumb + 28
    grid_rows = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * cell, grid_rows * cell), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        source = out_dir / row["path"]
        try:
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb, thumb))
                x = (index % cols) * cell + (cell - image.width) // 2
                y = (index // cols) * cell + 22 + (thumb - image.height) // 2
                sheet.paste(image, (x, y))
        except Exception:  # noqa: BLE001 - a missing image should not break the sheet
            pass
        label = row["image_id"]
        weak = ",".join(row["weak_pose_tags"][:2])
        if weak:
            label = f"{label} {weak}"
        draw.text(((index % cols) * cell + 4, (index // cols) * cell + 4), label, fill=(20, 20, 20))
    sheet.save(path)


def _write_readme(
    out_dir: Path,
    *,
    target_tag: str,
    distractor_tag: str,
    target_count: int,
    distractor_count: int,
    image_field: str,
    skipped: list[dict[str, str]],
) -> None:
    readme = f"""# Anime small benchmark

Generated: {datetime.now(UTC).isoformat()}

- Target character tag: `{target_tag}`
- Distractor character tag: `{distractor_tag}`
- Target images: {target_count}
- Distractor images: {distractor_count}
- Image field: `{image_field}`
- Skipped posts: {len(skipped)}

## Purpose

A small, reproducible identity/form benchmark for FormCoreset:

- identity filtering: target vs distractor;
- near-duplicate/form grouping: weak pose/view tags from Danbooru;
- 3D input readiness experiments.

## Important

The images are copyrighted by their respective artists. They are downloaded
locally for evaluation only and must not be committed or redistributed. The
repository contains only this download script and the generated metadata.
Respect Danbooru's terms and the artists' rights.

## Weak labels

`weak_pose_tags` and `weak_view_tags` are derived from Danbooru tags. They are
**not** ground-truth form labels; use them for sanity checks and qualitative
analysis, not as a supervised benchmark target.

## Layout

```text
{out_dir.name}/
  images/
    target/
    distractor/
  manifest.jsonl
  dataset.json
  contact_sheet_target.png
  contact_sheet_distractor.png
  README.md
  NOTICE.md
```

## Reproduce

```bash
uv run python benchmarks/download_anime_benchmark.py \\
  --target-tag {target_tag} --distractor-tag {distractor_tag} \\
  --target-limit {target_count} --distractor-limit {distractor_count} \\
  --image-field {image_field} --out {out_dir.as_posix()}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def _write_notice(out_dir: Path) -> None:
    notice = """# Notice

The images in this directory are downloaded from Danbooru for local,
non-redistributed evaluation. Each image remains the property of its respective
artist. Do not commit these files, publish them, or use them commercially
without checking the source post and the artist's terms.

The repository should contain only the download script, manifest, and derived
metadata. The `images/` directory is ignored by git.
"""
    (out_dir / "NOTICE.md").write_text(notice, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tag", default="hatsune_miku")
    parser.add_argument("--distractor-tag", default="kagamine_rin")
    parser.add_argument("--target-limit", type=int, default=50)
    parser.add_argument("--distractor-limit", type=int, default=20)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/data/anime_miku_small"),
    )
    parser.add_argument(
        "--image-field",
        choices=["large_file_url", "file_url", "preview_file_url"],
        default="large_file_url",
        help="Which Danbooru image URL to download (default: 1024px sample).",
    )
    parser.add_argument("--sleep", type=float, default=0.4, help="Delay between downloads.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear the output directory first.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Output directory is not empty: {out_dir}\n"
                "Use --overwrite to clear it, or choose another --out."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching candidates for target={args.target_tag!r} ...")
    target_posts = _fetch_posts(args.target_tag, args.target_limit)
    print(f"Fetching candidates for distractor={args.distractor_tag!r} ...")
    distractor_posts = _fetch_posts(args.distractor_tag, args.distractor_limit)

    seen_md5: set[str] = set()
    print("Downloading target split ...")
    target_rows, target_skipped = _download_split(
        target_posts,
        split="target",
        out_dir=out_dir,
        limit=args.target_limit,
        image_field=args.image_field,
        seen_md5=seen_md5,
        sleep=args.sleep,
    )
    print("Downloading distractor split ...")
    distractor_rows, distractor_skipped = _download_split(
        distractor_posts,
        split="distractor",
        out_dir=out_dir,
        limit=args.distractor_limit,
        image_field=args.image_field,
        seen_md5=seen_md5,
        sleep=args.sleep,
    )

    rows = target_rows + distractor_rows
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dataset = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "source": "danbooru",
        "target_tag": args.target_tag,
        "distractor_tag": args.distractor_tag,
        "target_count": len(target_rows),
        "distractor_count": len(distractor_rows),
        "image_field": args.image_field,
        "skipped": target_skipped + distractor_skipped,
        "notes": "Images are local-only and not redistributed; weak tags are not ground truth.",
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_contact_sheet(out_dir, target_rows, out_dir / "contact_sheet_target.png")
    _write_contact_sheet(
        out_dir,
        distractor_rows,
        out_dir / "contact_sheet_distractor.png",
    )
    _write_readme(
        out_dir,
        target_tag=args.target_tag,
        distractor_tag=args.distractor_tag,
        target_count=len(target_rows),
        distractor_count=len(distractor_rows),
        image_field=args.image_field,
        skipped=target_skipped + distractor_skipped,
    )
    _write_notice(out_dir)

    print(
        f"Done: {len(target_rows)} target + {len(distractor_rows)} distractor images\n"
        f"Output: {out_dir}\n"
        f"Manifest: {out_dir / 'manifest.jsonl'}"
    )


if __name__ == "__main__":
    main()
