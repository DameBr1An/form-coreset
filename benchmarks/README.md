# Benchmarks

This folder will hold reproducible M0/M1 evaluation scripts and annotations.

Planned benchmark suites:

- `anime-character/` — search-result cleanup: identity distractors, same-form
  variants with different backgrounds, hard negatives.
- `product-object/` — multi-view / multi-state object images.
- `training-lora/` — image-text pairs for redundancy and overfitting checks.

Image files are not committed when their license is unclear. Prefer scripts +
annotations for non-redistributable sets, and a small redistributable sample for
smoke tests.

Metrics: identity precision/recall, form pairwise precision/recall, ARI/AMI,
coverage@K, redundancy@K, canonical agreement, runtime, downstream LoRA quality.

## Small anime benchmark (local-only)

`download_anime_benchmark.py` downloads a small target-vs-distractor set from
Danbooru for local evaluation. Images are **not** committed; `benchmarks/data/`
is gitignored. The script records source post URLs, artist tags, and weak
pose/view tags in `manifest.jsonl`.

```bash
uv run python benchmarks/download_anime_benchmark.py \
  --target-tag hatsune_miku --distractor-tag kagamine_rin \
  --target-limit 50 --distractor-limit 20
```

The images remain the property of their artists. Respect Danbooru's terms and
do not redistribute them.

### CCIP identity baseline (small)

Setup:

```bash
uv sync --extra anime
```

Run:

```bash
uv run python benchmarks/run_ccip_baseline.py \
  --benchmark-dir benchmarks/data/anime_miku_small
```

Current baseline on 50 target + 20 distractor images, using 3 target
references and evaluating 47 target + 20 distractor:

| Metric | Value |
|---|---|
| AUC | 1.0000 |
| 5-fold CV F1 | 0.9783 |
| 5-fold CV precision | 1.0000 |
| 5-fold CV recall | 0.9574 |
| Mean score: target / distractor | 0.689 / 0.180 |

The two false negatives are portrait/close-up images (`target_12454799`,
`target_12452838`). This is a useful failure mode: CCIP is strong on
full-body/upper-body identity, weaker on very tight face crops.

This is a **small identity sanity check**, not a product benchmark. It uses one
character pair, only weak pose/view tags, and no ground-truth form labels.

### Same-source variant benchmark (primary path)

Logic and threshold rationale: [../docs/same-source-logic.md](../docs/same-source-logic.md).

Generate synthetic variants (crop, resize, JPEG, padded background,
brightness/contrast, effects) from local source images:

```bash
uv run python benchmarks/make_same_source_benchmark.py \
  --source-dir benchmarks/data/anime_miku_small/images/target \
  --sources 8 --out benchmarks/data/anime_same_source_small --overwrite
uv run python benchmarks/run_source_dedup_benchmark.py \
  --benchmark-dir benchmarks/data/anime_same_source_small \
  --method sift --min-score 0.3 --min-inliers 12 --no-stitch
```

Defaults: anime character mask on, aligned-edge verification on, complete
linkage + star merge, stitching on. Current result: 8 sources × 7 variants = 56
images; pairwise precision/recall/F1 = 1.0, group purity = 1.0 at
`min_score=0.3`, `star_threshold=0.2`. Masked SIFT without verification
over-merges similar sources (precision 0.774); unmasked at 0.15 is worse
(precision 0.533). See [RESULTS.md](RESULTS.md).

### Pose/form baseline

A pose-balanced set (6 poses × 8 full-body images) is available via:

```bash
uv run python benchmarks/download_anime_pose_benchmark.py --per-pose 8
uv run python benchmarks/run_pose_form_baseline.py \
  --benchmark-dir benchmarks/data/anime_miku_pose_small \
  --split all --pose-label-field pose_label
uv run python benchmarks/run_tag_form_baseline.py \
  --benchmark-dir benchmarks/data/anime_miku_pose_small
```

Headline results and caveats: [RESULTS.md](RESULTS.md).

Summary: DWPose/RTMPose keypoints are not reliable on stylized anime; WD14 tags
are a stronger form signal (HDBSCAN AMI ~0.5, purity ~0.8 on clustered images),
but running/jumping/action poses remain hard. Treat this as a sanity check, not
a product benchmark.

Status: pre-alpha; scripts land with the M0 spike harness.
