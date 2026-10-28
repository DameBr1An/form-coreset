#!/usr/bin/env python3
"""Download a pose-balanced, full-body anime benchmark from Danbooru.

Each pose class is queried separately:
    hatsune_miku solo full_body <pose> rating:general

Images are local-only and must not be committed. The pose label comes from the
query tag, but Danbooru tags are still weak labels, not human ground truth.

Usage:
    uv run python benchmarks/download_anime_pose_benchmark.py \
        --per-pose 8 --out benchmarks/data/anime_miku_pose_small
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from download_anime_benchmark import (
    _api_get,
    _download_split,
    _write_contact_sheet,
)

DEFAULT_POSES = ["standing", "sitting", "running", "jumping", "lying", "kneeling"]


def _fetch_query(query: str, wanted: int, *, sleep: float = 1.0) -> list[dict[str, Any]]:
    """Fetch a 2-tag Danbooru query (anonymous API tag limit), then filter locally."""

    posts: list[dict[str, Any]] = []
    seen: set[int] = set()
    page = 1
    while len(posts) < wanted * 8 and page <= 3:
        batch = _api_get(
            {
                "tags": f"{query} rating:general",
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
        page += 1
        time.sleep(sleep)
    return posts


def _write_readme(out_dir: Path, counts: dict[str, int], per_pose: int) -> None:
    lines = [
        "# Anime pose-balanced small benchmark",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "Target character: `hatsune_miku`",
        "Query form: `solo full_body <pose> rating:general`",
        "",
        "## Counts",
        "",
        "| Pose | Images |",
        "|---|---:|",
    ]
    for pose, count in counts.items():
        lines.append(f"| {pose} | {count} |")
    lines.extend(
        [
            "",
            f"Requested per pose: {per_pose}",
            "",
            "## Important",
            "",
            "The images are copyrighted by their respective artists. They are",
            "downloaded locally for evaluation only and must not be committed or",
            "redistributed. Pose labels come from Danbooru query tags; they are weak",
            "labels, not human ground truth.",
            "",
            "## Layout",
            "",
            "```text",
            f"{out_dir.name}/",
            "  images/<pose>/",
            "  manifest.jsonl",
            "  dataset.json",
            "  contact_sheet_<pose>.png",
            "  README.md",
            "  NOTICE.md",
            "```",
            "",
            "## Reproduce",
            "",
            "```bash",
            "uv run python benchmarks/download_anime_pose_benchmark.py \\",
            f"  --per-pose {per_pose} --out {out_dir.as_posix()}",
            "```",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_notice(out_dir: Path) -> None:
    notice = """# Notice

The images in this directory are downloaded from Danbooru for local,
non-redistributed evaluation. Each image remains the property of its respective
artist. Do not commit these files, publish them, or use them commercially
without checking the source post and the artist's terms.

The `images/` directory is ignored by git.
"""
    (out_dir / "NOTICE.md").write_text(notice, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poses",
        default=",".join(DEFAULT_POSES),
        help="Comma-separated pose tags.",
    )
    parser.add_argument("--per-pose", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/data/anime_miku_pose_small"),
    )
    parser.add_argument(
        "--image-field",
        choices=["large_file_url", "file_url", "preview_file_url"],
        default="large_file_url",
    )
    parser.add_argument("--sleep", type=float, default=0.4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    poses = [pose.strip() for pose in args.poses.split(",") if pose.strip()]
    out_dir: Path = args.out
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Output directory is not empty: {out_dir}\n"
                "Use --overwrite to clear it, or choose another --out."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_md5: set[str] = set()
    all_rows: list[dict[str, object]] = []
    skipped_all: list[dict[str, str]] = []
    counts: dict[str, int] = {}

    for pose in poses:
        print(f"Fetching pose={pose!r} ...")
        posts = _fetch_query(f"hatsune_miku {pose}", args.per_pose)
        full_body_posts = [
            post
            for post in posts
            if "full_body" in (post.get("tag_string") or "").split()
            and "solo" in (post.get("tag_string") or "").split()
        ]
        rows, skipped = _download_split(
            full_body_posts,
            split=pose,
            out_dir=out_dir,
            limit=args.per_pose,
            image_field=args.image_field,
            seen_md5=seen_md5,
            sleep=args.sleep,
        )
        # If full_body is too restrictive, retry with solo-only posts.
        if len(rows) < args.per_pose:
            print(f"  only {len(rows)} with full_body; retrying without full_body ...")
            solo_posts = [
                post
                for post in posts
                if "solo" in (post.get("tag_string") or "").split()
            ]
            extra_rows, extra_skipped = _download_split(
                solo_posts,
                split=pose,
                out_dir=out_dir,
                limit=args.per_pose - len(rows),
                image_field=args.image_field,
                seen_md5=seen_md5,
                sleep=args.sleep,
            )
            rows.extend(extra_rows)
            skipped.extend(extra_skipped)
        for row in rows:
            row["pose_label"] = pose
            row["query"] = f"hatsune_miku solo full_body {pose} rating:general"
        all_rows.extend(rows)
        skipped_all.extend(skipped)
        counts[pose] = len(rows)
        print(f"  {pose}: {len(rows)}")
        _write_contact_sheet(
            out_dir,
            rows,
            out_dir / f"contact_sheet_{pose}.png",
        )

    # Rewrite image_id to include the pose class, keeping post ids unique.
    for row in all_rows:
        row["image_id"] = f"{row['split']}_{row['post_id']}"

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dataset = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "source": "danbooru",
        "character": "hatsune_miku",
        "poses": counts,
        "total": len(all_rows),
        "per_pose": args.per_pose,
        "image_field": args.image_field,
        "skipped": skipped_all,
        "notes": "Pose labels are query tags, not human ground truth; images are local-only.",
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(out_dir, counts, args.per_pose)
    _write_notice(out_dir)
    print(f"Done: {len(all_rows)} images; output={out_dir}")


if __name__ == "__main__":
    main()
