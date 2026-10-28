# Release checklist

## Before public release

- [x] Repository URLs set to `DameBr1An/form-coreset`.
- [ ] Check GitHub and PyPI name availability (`form-coreset`).
- [ ] Review `LICENSE` (Apache-2.0) and `THIRD_PARTY_LICENSES.md`.
- [ ] Do not bundle restricted model weights; verify each downloaded weight.
- [ ] README states the problem, demo, install, quickstart, scope, and limits.
- [ ] Demo image is synthetic (`examples/demo_dedupe.py`), no copyrighted images.
- [ ] No benchmark images or downloaded data are staged (`git status`).
- [ ] `uv run ruff check .` passes.
- [ ] `uv run pytest` passes (optional extras may skip their tests).
- [ ] CI is green on Python 3.11 and 3.12.
- [ ] `CHANGELOG.md` updated; version in `pyproject.toml`.
- [ ] Issue templates, PR template, Code of Conduct, Security policy present.
- [ ] Model download policy documented (source, checksum, license warning).
- [ ] Benchmark scripts are reproducible; data is gitignored.
- [ ] Non-destructive behavior verified: originals are never modified.

## GitHub setup

- [ ] Create the repository (public/private as decided).
- [ ] Description: "Same-source image variant dedupe for character, mascot, and
      product collections. Local-first, non-destructive."
- [ ] Topics: `image-deduplication`, `dataset-curation`, `computer-vision`,
      `sift`, `image-stitching`, `anime`, `lora`, `local-first`.
- [ ] Enable Issues; Discussions optional.
- [ ] Labels: `bug`, `enhancement`, `model`, `license`, `good first issue`,
      `needs-triage`.
- [ ] Add 3–5 `good first issue` tasks (docs, new mask provider, benchmark).
- [ ] Create release `v0.1.0a0` with release notes from `CHANGELOG.md`.
- [ ] Add repository social preview (demo image).

## After release

- [ ] Monitor issues and label/triage weekly.
- [ ] Keep a support matrix: OS, Python, optional extras.
- [ ] Publish a roadmap milestone for `v0.1.0` / `v0.2.0`.
- [ ] Ask early users for small, redistributable benchmark samples.
