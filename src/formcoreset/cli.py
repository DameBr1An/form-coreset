"""Command-line interface for FormCoreset."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .cluster import compute_hashes, group_near_duplicates
from .config import load_config
from .manifest import read_manifest, write_manifest, write_run_manifest
from .pipeline import (
    run_baseline,
    run_form_extraction,
    run_same_source_dedup,
    scan_and_assess,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Form-aware visual coreset extraction for image collections.",
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"form-coreset {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """FormCoreset: keep the information, drop the duplicates."""


@app.command()
def scan(
    input_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, resolve_path=True, help="Folder to scan."
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", resolve_path=True, help="Write manifest/run metadata here."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", exists=True, dir_okay=False, help="YAML config file."
    ),
) -> None:
    """Scan a folder, compute hashes and quality heuristics."""

    config = load_config(config_path)
    result = scan_and_assess(input_dir)

    console.print(
        f"[bold]Scanned[/bold] {len(result.records)} images; "
        f"skipped {len(result.skipped)}."
    )

    table = Table("id", "file", "size", "flags")
    for record in result.records:
        table.add_row(
            record.image_id,
            record.rel_path,
            f"{record.width}x{record.height}",
            ", ".join(record.flags) or "-",
        )
    if result.records:
        console.print(table)

    if result.skipped:
        skipped_table = Table("skipped", "reason")
        for item in result.skipped:
            skipped_table.add_row(item["path"], item["reason"])
        console.print(skipped_table)

    if out is not None:
        write_manifest(result.records, out / "manifest.jsonl")
        write_run_manifest(out, config=config, scan=result, extra={"mode": "scan"})
        console.print(f"Wrote [cyan]{out / 'manifest.jsonl'}[/cyan]")


@app.command()
def organize(
    input_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, resolve_path=True, help="Folder to organize."
    ),
    out: Path = typer.Option(..., "--out", "-o", resolve_path=True, help="Output folder."),
    auto: bool = typer.Option(
        False, "--auto", help="No target: automatic identity + form clustering."
    ),
    name: str | None = typer.Option(None, "--name", help="Target name (guided mode)."),
    description: str | None = typer.Option(
        None, "--description", help="Target description (guided mode)."
    ),
    reference: list[Path] | None = typer.Option(
        None,
        "--reference",
        "-r",
        exists=True,
        dir_okay=False,
        help="Reference image(s) for the target (guided mode).",
    ),
    adapter: str = typer.Option("auto", "--adapter", help="auto | person | object."),
    threshold: int = typer.Option(
        5, "--threshold", min=0, max=64, help="Baseline dHash Hamming threshold."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", exists=True, dir_okay=False, help="YAML config file."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Scan and report; write nothing."),
) -> None:
    """Organize a folder into form groups.

    The current pre-alpha implements a model-free near-duplicate baseline.
    Guided identity, pose features, and real form clustering are in progress.
    """

    if adapter not in {"auto", "person", "object"}:
        console.print("[red]--adapter must be auto, person, or object.[/red]")
        raise typer.Exit(code=1)

    has_target = bool(name or description or reference)
    if has_target:
        console.print(
            "[yellow]Guided identity backends are not wired in this pre-alpha.[/yellow]\n"
            "Use --auto for the model-free baseline, or watch the repo for M1."
        )
        raise typer.Exit(code=1)

    if not auto:
        console.print(
            "[yellow]This pre-alpha only implements the model-free baseline.[/yellow]\n"
            "Re-run with --auto, or provide a target once guided backends land."
        )
        raise typer.Exit(code=1)

    if adapter != "auto":
        console.print(
            f"[yellow]Adapter '{adapter}' is not wired yet; "
            "the baseline is identity-agnostic.[/yellow]"
        )

    config = load_config(config_path)
    config.adapter = adapter  # type: ignore[assignment]

    if dry_run:
        result = scan_and_assess(input_dir)
        hashes = compute_hashes(result.records)
        groups = group_near_duplicates(result.records, hashes, threshold=threshold)
        console.print(
            f"[bold]Dry run[/bold]: {len(result.records)} images, "
            f"{len(groups)} baseline groups, nothing written."
        )
        return

    result, groups = run_baseline(input_dir, out, config, threshold=threshold)
    console.print(
        f"[bold]Organized[/bold] {len(result.records)} images into "
        f"{len(groups)} baseline groups."
    )
    console.print(f"Output: [cyan]{out}[/cyan]")
    console.print(f"Report: [cyan]{out / 'report.md'}[/cyan]")


@app.command()
def dedupe(
    input_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, resolve_path=True, help="Folder of images."
    ),
    out: Path = typer.Option(
        ..., "--out", "-o", resolve_path=True, help="Deduplicated output folder."
    ),
    method: str = typer.Option(
        "sift", "--method", help="Local feature method: sift or orb."
    ),
    min_score: float = typer.Option(
        0.3, "--min-score", min=0.0, max=1.0, help="Minimum match score."
    ),
    min_inliers: int = typer.Option(
        12, "--min-inliers", min=4, help="Minimum RANSAC inliers."
    ),
    max_features: int = typer.Option(
        2000, "--max-features", min=100, help="Maximum local features per image."
    ),
    mask_provider: str = typer.Option(
        "auto",
        "--mask-provider",
        help="Mask backend: auto, anime, general, none.",
    ),
    linkage: str = typer.Option(
        "complete", "--linkage", help="Grouping linkage: complete, average, single."
    ),
    star_threshold: float = typer.Option(
        0.2,
        "--star-threshold",
        min=0.0,
        max=1.0,
        help="Minimum medoid confidence to reattach a singleton.",
    ),
    min_correlation: float = typer.Option(
        0.35,
        "--min-correlation",
        min=0.0,
        max=1.0,
        help="Minimum aligned-edge correlation.",
    ),
    no_star_merge: bool = typer.Option(
        False, "--no-star-merge", help="Disable singleton medoid reattachment."
    ),
    no_mask: bool = typer.Option(
        False, "--no-mask", help="Disable anime character mask for matching."
    ),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Disable homography edge verification."
    ),
    no_stitch: bool = typer.Option(
        False, "--no-stitch", help="Disable canonical stitching."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", exists=True, dir_okay=False, help="YAML config file."
    ),
) -> None:
    """Group same-source artwork variants (crops, effects, backgrounds)."""

    if method not in {"sift", "orb"}:
        console.print("[red]--method must be sift or orb.[/red]")
        raise typer.Exit(code=1)
    if linkage not in {"complete", "average", "single"}:
        console.print("[red]--linkage must be complete, average, or single.[/red]")
        raise typer.Exit(code=1)
    if mask_provider not in {"auto", "anime", "general", "none"}:
        console.print(
            "[red]--mask-provider must be auto, anime, general, or none.[/red]"
        )
        raise typer.Exit(code=1)

    config = load_config(config_path)
    config.source.method = method
    config.source.min_score = min_score
    config.source.min_inliers = min_inliers
    config.source.max_features = max_features
    config.source.linkage = linkage
    config.source.star_threshold = star_threshold
    config.source.min_correlation = min_correlation
    config.source.star_merge = not no_star_merge
    config.source.mask_provider = "none" if no_mask else mask_provider
    config.source.use_mask = not no_mask and mask_provider != "none"
    config.source.verify = not no_verify
    config.source.stitch = not no_stitch

    try:
        result, groups = run_same_source_dedup(input_dir, out, config)
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    removed = len(result.records) - len(groups)
    console.print(
        f"[bold]Deduped[/bold] {len(result.records)} images into "
        f"{len(groups)} source groups; removed {removed} variants."
    )
    console.print(f"Output: [cyan]{out}[/cyan]")
    console.print(f"Report: [cyan]{out / 'report.md'}[/cyan]")


@app.command()
def extract(
    input_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, resolve_path=True, help="Folder of character images."
    ),
    out: Path = typer.Option(
        ..., "--out", "-o", resolve_path=True, help="Form Library output folder."
    ),
    encoder: str = typer.Option(
        "subject",
        "--encoder",
        help="Form encoder: subject (generic DINOv2+silhouette) or anime-hybrid.",
    ),
    mask_provider: str = typer.Option(
        "auto",
        "--mask-provider",
        help="Mask backend: auto, anime, general, none.",
    ),
    min_cluster_size: int = typer.Option(
        3, "--min-cluster-size", min=2, help="HDBSCAN min cluster size."
    ),
    min_samples: int = typer.Option(
        2, "--min-samples", min=1, help="HDBSCAN min samples."
    ),
    tag_weight: float = typer.Option(
        0.7, "--tag-weight", min=0.0, max=1.0, help="anime-hybrid: WD14 tag weight."
    ),
    embedding_weight: float = typer.Option(
        0.3, "--embedding-weight", min=0.0, max=1.0, help="anime-hybrid: CCIP weight."
    ),
    no_embedding: bool = typer.Option(
        False, "--no-embedding", help="anime-hybrid: use WD14 tags only."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", exists=True, dir_okay=False, help="YAML config file."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Scan only; do not encode or write."
    ),
) -> None:
    """Extract a Form Library from a folder of images."""

    if encoder not in {"subject", "anime-hybrid"}:
        console.print(
            "[red]--encoder must be subject or anime-hybrid.[/red]"
        )
        raise typer.Exit(code=1)
    if mask_provider not in {"auto", "anime", "general", "none"}:
        console.print(
            "[red]--mask-provider must be auto, anime, general, or none.[/red]"
        )
        raise typer.Exit(code=1)

    config = load_config(config_path)
    config.form.encoder = encoder
    config.form.mask_provider = mask_provider
    config.form.min_cluster_size = min_cluster_size
    config.form.min_samples = min_samples
    config.form.tag_weight = tag_weight
    config.form.embedding_weight = 0.0 if no_embedding else embedding_weight

    if dry_run:
        result = scan_and_assess(input_dir)
        console.print(
            f"[bold]Dry run[/bold]: {len(result.records)} images; nothing encoded."
        )
        return

    try:
        result, groups = run_form_extraction(input_dir, out, config)
    except (ImportError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold]Extracted[/bold] {len(result.records)} images into "
        f"{len(groups)} form groups."
    )
    console.print(f"Output: [cyan]{out}[/cyan]")
    console.print(f"Report: [cyan]{out / 'report.md'}[/cyan]")


@app.command()
def report(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, resolve_path=True, help="Output folder to summarize."
    ),
) -> None:
    """Summarize an existing run from its manifest."""

    manifest_path = run_dir / "manifest.jsonl"
    if not manifest_path.exists():
        console.print(f"[red]No manifest found:[/red] {manifest_path}")
        raise typer.Exit(code=1)

    records = read_manifest(manifest_path)
    status_counts: dict[str, int] = {}
    forms: set[str] = set()
    canonicals = 0
    for record in records:
        status = record.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if record.get("form_id"):
            forms.add(record["form_id"])
        if record.get("is_canonical"):
            canonicals += 1

    table = Table("status", "count")
    for status, count in sorted(status_counts.items()):
        table.add_row(status, str(count))
    console.print(table)
    console.print(
        f"Images: [bold]{len(records)}[/bold]; forms: [bold]{len(forms)}[/bold]; "
        f"canonicals: [bold]{canonicals}[/bold]"
    )


@app.command()
def review(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, resolve_path=True, help="Output folder to review."
    ),
) -> None:
    """Launch the local Gradio review UI."""

    try:
        from .ui.app import launch
    except ImportError as exc:
        console.print(
            "[red]Gradio is not installed.[/red] Install it with:\n"
            "  uv sync --extra ui"
        )
        raise typer.Exit(code=1) from exc
    launch(run_dir)


@app.command()
def export(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, resolve_path=True, help="Output folder to export."
    ),
    preset: str = typer.Option("generic", "--preset", help="generic | kohya | diffusers."),
) -> None:
    """Export a run for a downstream tool (not implemented in pre-alpha)."""

    console.print(
        f"[yellow]Export preset '{preset}' is not implemented in this pre-alpha.[/yellow]\n"
        f"Run folder: {run_dir}"
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
