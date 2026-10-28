# Contributing to FormCoreset

Thanks for helping. FormCoreset is alpha; the highest-value contributions right
now are benchmark data, model backends, and critique of the form-clustering
design.

## Current scope (v0.1)

- **Stable:** same-source dedupe, canonical selection, stitching, manifests.
- **Experimental:** form clustering, anime tag prior.
- **Not yet:** automatic identity filtering, full review UI, 3D export,
  large-scale candidate prefilter.

Please keep PRs aligned with this scope. Open an issue before a large change.

## Setup

```bash
git clone https://github.com/DameBr1An/form-coreset.git
cd form-coreset
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Optional extras:

```bash
uv sync --extra subject  # timm + torch (DINOv2 + silhouette)
uv sync --extra mask     # rembg / BiRefNet
uv sync --extra anime    # imgutils / anime-seg / WD14 / CCIP
uv sync --extra ui       # Gradio review UI
uv sync --all-extras     # everything (heavy)
```

## Project layout

```text
src/formcoreset/
  cli.py            # Typer CLI
  config.py         # Pydantic config
  types.py          # Core dataclasses
  ingest.py         # Scan / hash / decode
  quality.py        # Heuristics and flags
  cluster.py        # Baseline dHash + future clustering
  manifest.py       # JSONL / run manifest / report
  organize.py       # Output tree, copying, contact sheet
  pipeline.py       # Orchestration
  identity/         # Identity matchers (CCIP, WD14, embeddings, auto)
  form/             # Form encoders (person, object, auto)
  ui/               # Gradio review app
tests/
benchmarks/
examples/
docs/
```

## Adding a model backend

1. Implement the relevant Protocol in `identity/base.py` or `form/base.py`.
2. Add a registry entry with name, version, source, checksum, and license.
3. Add a license note to `THIRD_PARTY_LICENSES.md`.
4. Do not bundle restricted weights. Download on first use.
5. Add a smoke test using a tiny fixture; mark model-heavy tests with
   `@pytest.mark.slow`.
6. Add a fallback path: a backend failure must not crash the whole run.

## License policy

- Code is Apache-2.0.
- We do not bundle model weights in the wheel.
- Do not add a default dependency with a non-commercial or unclear license
  (e.g. SuperPoint weights). Use SIFT + LightGlue or ALIKED + LightGlue
  instead, or make the backend opt-in with a clear warning.
- Every model must be listed in `THIRD_PARTY_LICENSES.md` with:
  name, version, license, source, commercial-use status, and whether it is
  downloaded or bundled.

## Non-negotiable behavior

- Never modify or delete source images.
- Never upload images or metadata by default.
- Never silently drop images: use `uncertain/`, flags, and report entries.
- Every output must be traceable through `manifest.jsonl`.

## Pull requests

- Keep changes focused.
- Add tests for new behavior.
- Update docs / PRD if the behavior or scope changes.
- Record product/architecture decisions in `docs/decisions.md`.
- Run `uv run ruff check .` and `uv run pytest`.
- Describe user-visible changes and any model/license impact.

## Issue templates

Coming with the first public release:

- Bug report
- Feature request
- Model / license addition
- Benchmark submission
