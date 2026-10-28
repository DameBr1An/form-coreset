# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `scan`: directory scan, content hash, image metadata, quality heuristics.
- `dedupe`: same-source variant grouping with SIFT/ORB, optional mask,
  ratio test, RANSAC homography, and aligned-edge verification.
- Grouping strategies: connected components, average linkage, complete linkage,
  and singleton → medoid star merge.
- Canonical selection and optional union-canvas stitching for complementary
  crops.
- Output contracts: `sources/`, `canonical.png`, `canonical_single.*`,
  `variants/`, `report.md`, `contact_sheet.png`, `manifest.jsonl`, `run.json`.
- Pluggable `MaskProvider`: `none`, `anime`, `general`, `auto`.
- Experimental `extract`: generic DINOv2 + silhouette form encoder, optional
  anime-hybrid (WD14 tags + CCIP) encoder, HDBSCAN grouping, no pose labels.
- Tests for ingest, dedupe, stitching, mask providers, subject encoder,
  form-caption redundancy, and CLI pipeline.
- GitHub Actions CI on Python 3.11/3.12.
- Same-source benchmarks and a synthetic variant benchmark.

### Known limitations

- Automatic identity filtering from mixed search results is not implemented.
- Fine-grained form/pose clustering is experimental and needs human review.
- 3D/UE5 export and full review UI are not implemented.
- Pair matching is O(n²); no FAISS/LightGlue prefilter yet.
- `export` is a stub.

## [0.1.0a0] - 2026-10-28

- Initial pre-alpha scaffold: CLI, data model, baseline organizer, docs.
