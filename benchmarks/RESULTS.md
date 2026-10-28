# Benchmark results (2026-10-27)

> Small, local-only benchmarks. Images are not committed; see the download
> scripts and `benchmarks/data/` (gitignored). Metrics use weak Danbooru tags,
> not human ground truth.

## Data

| Set | Content | Size |
|---|---|---|
| Identity small | Hatsune Miku target + Kagamine Rin distractors | 50 + 20 = 70 |
| Pose-balanced small | 6 poses × 8 full-body images (`standing`, `sitting`, `running`, `jumping`, `lying`, `kneeling`) | 48 |

## Same-source dedup (primary path)

Data: 8 source images × 7 variants (original, crop, resize, JPEG, padded
background, brightness/contrast, translucent effects) = 56 images. Ground truth
is the source image; no pose labels.

Method: SIFT + ratio test + RANSAC homography + connected components.
Optional anime character mask and post-homography edge verification.

| Configuration | Linkage | Groups | Pairwise precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| no mask, no verify | single | 6 | 0.533 | 1.000 | 0.696 |
| no mask, no verify | single | 8 | 1.000 | 1.000 | 1.000 |
| mask, no verify | single | 7 | 0.774 | 1.000 | 0.873 |
| mask + edge verify | single | 8 | 1.000 | 1.000 | 1.000 |
| mask + edge verify | average | 9 | 1.000 | 0.964 | 0.982 |
| mask + edge verify | complete | 10 | 1.000 | 0.935 | 0.966 |
| **mask + edge verify** | **complete + star merge** | **8** | **1.000** | **1.000** | **1.000** |

At 0.15 unmasked, three similar source images (`src_02`, `src_05`, `src_08`)
were over-merged. Masked SIFT alone also over-merged `src_02`/`src_08` at 0.30;
adding the aligned-edge correlation check rejects that false homography.

**Linkage trade-off:**

- `single` (connected components) has the highest recall but can chain
  A~B, B~C while A≁C. It is recall-oriented.
- `average` is a middle ground.
- `complete` requires **every cross-cluster pair** to match; it is
  precision-first. It split two weak variants into singletons, which a reviewer
  can merge.
- **`complete + star merge`** keeps the complete-linkage seed groups, then
  reattaches a singleton only when it matches a group **medoid** above the
  star threshold (0.20). It never merges group-to-group, so it cannot create a
  chain. It recovers the two split variants and restores recall to 1.0.
- Default is **complete + star merge**, because false merges are more damaging
  than conservative splits. Use `--linkage average` or `single` for
  recall-oriented domains; `--no-star-merge` disables reattachment.

Default: **mask on, edge verification on, complete linkage + star merge,
`min_score=0.30`, `min_inliers=12`, `star_threshold=0.20`**. Stitching is
enabled by default; it is tested separately on complementary crops
(`canonical.png` stitched, `canonical_single.png` kept).

This validates the primary product path: same-source poster/artwork variants can
be deduplicated without pose labels. Synthetic variants are easier than real
posters; thresholds should still be tuned per domain. Outputs:
`benchmarks/data/anime_same_source_small/dedupe_complete_star/`,
`dedupe_complete_verified/`, and `dedupe_average_verified/`.

### Star-merge rule evaluation (2026-10-26)

Comparing the current medoid rule with a proposed dual-axis rule:

- `current`: `min(score, corr) >= 0.20`
- `replacement`: `score >= 0.15 and corr >= 0.80`
- `union`: current OR replacement

| Dataset | Rule | Groups | Precision | Recall | F1 | Mixed groups |
|---|---|---|---:|---:|---:|---:|
| Synthetic | current | 8 | 1.000 | 1.000 | 1.000 | 0 |
| Synthetic | replacement | 8 | 1.000 | 1.000 | 1.000 | 0 |
| Synthetic | union | 8 | 1.000 | 1.000 | 1.000 | 0 |
| Jett | current | 22 | 1.000 | 0.750 | 0.857 | 0 |
| Jett | replacement | 24 | 1.000 | 0.450 | 0.621 | 0 |
| Jett | union | 22 | 1.000 | 0.750 | 0.857 | 0 |

**Decision:** do not adopt the dual-axis rule. It gives no gain on the
synthetic benchmark, no gain on Jett, and is worse when used as a replacement.
The Jett singleton 015 has a high-correlation edge to **non-medoid member 010**,
so a medoid-only dual-axis rule cannot recover it. Recovering it would require
member-level merging or manual review, not a global threshold tweak.

Keep `star_threshold=0.20` and the current medoid rule.

### Canonical stitching

Complementary-crop test: two 260×400 crops from a 400×400 source are grouped
and stitched into a **400×400** `canonical.png` (area gain 0.5385), while
`canonical_single.png` keeps the selected single crop. Unit tests cover both
direct stitching and the dedupe pipeline. Stitching falls back to the single
canonical when the union does not increase coverage.

## Identity baseline (CCIP)

3 target references; evaluate 47 target + 20 distractor.

| Metric | Value |
|---|---:|
| AUC | 1.0000 |
| 5-fold CV F1 | 0.9783 |
| CV precision | 1.0000 |
| CV recall | 0.9574 |
| Mean score target / distractor | 0.689 / 0.180 |

False negatives were portrait/close-up images. CCIP is strong on
full-body/upper-body identity, weaker on tight face crops.

## Pose/form baselines

### Keypoints

| Method | Pose success | Clusters | Noise | Posture AMI | Posture purity | Notes |
|---|---:|---:|---:|---:|---:|---|
| DWPose, identity set target | 46/50 | 3 | 17 | 0.00 | 0.60 | clusters by framing/visibility, not pose |
| DWPose, pose-balanced set | 45/48 | 2 | 14 | -0.01 | 0.258 | not pose-aligned |
| RTMPose + HumanArt, visual check | 6/6 detected | — | — | — | — | keypoints visually unreliable: missing legs, wrong lying/dynamic poses |

**Conclusion:** off-the-shelf 2D keypoints (DWPose or RTMPose HumanArt) are
not reliable enough on stylized anime. Do not build form clustering on
keypoints alone.

### Fully label-free features (negative result)

No hand-defined pose vocabulary; clustering is unsupervised; labels are used
only for evaluation.

| Feature | KMeans AMI / purity | HDBSCAN AMI / purity | Notes |
|---|---:|---:|---|
| WD14 full embedding (1024-d) | 0.181 / 0.417 | 0.073 / 0.306 | 4 clusters, 12 noise |
| WD14 tags + data-driven TF-IDF (254 tags) | 0.204 / 0.458 | 0.121 / 0.378 | best of top-variance/TF-IDF variants |
| DINOv2 ViT-S/14, full image | 0.155 / 0.438 | 0.069 / 0.467 | 2 clusters, 33 noise |
| isnet-anime silhouette, PCA32 | 0.103 / 0.396 | 0.167 / 0.438 | 3 clusters, 32 noise |
| Masked DINOv2, background removed | 0.096 / 0.375 | 0.120 / 0.462 | 3 clusters, 22 noise |
| Masked DINOv2 + PCA32 | 0.097 / 0.396 | 0.142 / 0.500 | 3 clusters, 24 noise |
| Silhouette + masked DINOv2 | 0.156 / 0.417 | 0.050 / 0.462 | 2 clusters, 35 noise |
| Hand-built posture/view/action tag prior | 0.406 / 0.604 | **0.482–0.518 / 0.794–0.800** | internal feature prior, no label output |

**Conclusion:** fully label-free visual/tag embeddings are substantially weaker
on stylized anime, even after background removal and silhouette extraction.
The product keeps a small **internal** posture/view/action tag prior as a
feature, but does **not** output pose names or require user labels. Clustering
itself remains unsupervised (HDBSCAN).

**Important caveat:** this benchmark is **diverse fan art** (different artists,
styles, chibi, line art), not same-source poster variants. If the input is
mostly the same artwork with different backgrounds/effects/crops, **same-source
registration** should be the first pass; it can dedupe those variants without
pose understanding and should perform much better than these numbers suggest.

### WD14 tags with the internal prior

Feature: confidence of posture/view/action tags from WD14 (`SwinV2_v3`).

| Method | Clusters | Noise | AMI | Purity | Notes |
|---|---:|---:|---:|---:|---|
| KMeans (k=6) | 6 | 0 | 0.406 | 0.604 | lying/sitting/standing partly separate |
| HDBSCAN | 8–9 | 14–18 | 0.482–0.518 | 0.794–0.800 | best current baseline |
| Direct tag prediction | — | — | — | — | accuracy 0.604, coverage 0.854 |
| CCIP only | 2 | 32 | 0.313 | 0.500 | identity model, not pose |
| Tags + CCIP (70/30), KMeans | 6 | 0 | 0.465 | 0.625 | small gain over tags |
| Tags + CCIP (70/30), HDBSCAN | 6 | 10 | 0.541 | 0.737 | small AMI gain, lower purity |

Visual inspection of KMeans clusters:

- **lying** cluster is clean (6/6);
- **sitting** cluster is mostly clean (7 sitting + 1 kneeling);
- **standing** cluster is mostly clean (6 standing + 1 lying);
- running/jumping/action poses are mixed with standing/kneeling.

## Findings

1. **Keypoints are the wrong primary signal for anime form clustering.**
   Detection succeeds, but geometry is often wrong on dynamic poses, chibi,
   and lying characters.
2. **WD14 tags are a much stronger anime-specific form signal.** They separate
   lying/sitting/standing reasonably; running/jumping remain hard.
3. **CCIP is an identity model, not a pose model.** It adds little to tags;
   use it for identity, not form.
4. **Weak labels are noisy.** Query tags and WD14 tags disagree on some
   running/jumping images. Treat AMI/purity as sanity checks, not ground truth.
5. **Human-in-the-loop remains necessary.** The UI should show tag evidence and
   let users merge/split form clusters.

## Implications for FormCoreset

- Default form encoder for anime: **WD14 tags + masked image embedding**
  (CCIP/DINOv2), with optional keypoints only when confidence is high.
- Do not promise fully automatic pose clustering on anime yet.
- Running/jumping/action forms need either a learned form embedding (CPIE),
  a VLM-based pose describer, or human review.
- This benchmark is a starting point, not a product-quality evaluation.

## Tool integration (2026-10-27)

`form-coreset extract` now uses the `anime-hybrid` encoder:

- WD14 internal tag prior (0.7) + CCIP embedding (0.3);
- HDBSCAN (`min_cluster_size=3`, `min_samples=2`);
- **no pose labels or pose names in the output**; groups are `form_001`,
  `form_002`, …;
- noise / encoding failures go to `uncertain/`.

Validated on the 48-image pose-balanced set: 7 form groups + 12 uncertain.
The report contains cluster size, canonical, method, and encoder version — no
pose names. Report and contact sheet:
`benchmarks/data/anime_miku_pose_small/form_library_v3/`.

This does not mean the clustering is perfect; it means the end-to-end Form
Library path is now runnable on real anime data without requiring user-defined
pose labels.

## Reproduce

```bash
uv sync --extra anime
uv run python benchmarks/download_anime_benchmark.py
uv run python benchmarks/run_ccip_baseline.py
uv run python benchmarks/download_anime_pose_benchmark.py
uv run python benchmarks/run_pose_form_baseline.py \
  --benchmark-dir benchmarks/data/anime_miku_pose_small \
  --split all --pose-label-field pose_label
uv run python benchmarks/run_tag_form_baseline.py \
  --benchmark-dir benchmarks/data/anime_miku_pose_small
```

RTMPose HumanArt overlays were produced with `rtmlib` (`uv pip install rtmlib`)
as a diagnostic, not a committed baseline. See the keypoint overlay images under
`benchmarks/data/anime_miku_pose_small/pose_form_baseline/`.
