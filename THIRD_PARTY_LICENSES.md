# Third-Party Models and Licenses

> **Status: pre-release audit required.** This table lists planned/default model
> candidates. It is not legal advice. Every weight file must be verified before a
> public release, and the final license text must be included where required.

FormCoreset code is Apache-2.0. Model weights are **not** bundled in the wheel;
they are downloaded on first use (or supplied by the user) with a version,
checksum, and license notice.

| Component | Planned use | License status | Default? | Notes |
|---|---|---|---|---|
| DINOv2 (Meta) | Image / form embeddings | Apache-2.0 (code + weights) | Yes | Commercial use allowed; keep NOTICE |
| timm | Model loading / transforms | Apache-2.0 | Subject extra | Code dependency |
| PyTorch | DINOv2 inference | BSD-3-Clause | Subject extra | Code dependency; install CPU or CUDA build |
| LightGlue (CVG) | Local feature matching | Apache-2.0 | Yes | Use SIFT or ALIKED features by default |
| SuperPoint (Magic Leap) | Local features | **Restrictive weights** | **No** | Do not bundle or default; opt-in only |
| OpenCV SIFT | Local features | Permissive (patent expired) | Yes | Classical fallback |
| CCIP (deepghs/imgutils) | Anime character identity | Verify | Anime extra | Model weights and upstream license need audit |
| WD14 tagger (SmilingWolf) | Anime character/general tags | Verify | Anime extra | Danbooru-derived tags; check model card |
| isnet-anime / anime-seg | Anime subject segmentation | Verify | Anime extra | rembg packages isnet-anime weights |
| rembg | Mask runtime | MIT (wrapper) | Mask extra | Model weights have separate terms |
| BiRefNet | General subject segmentation | MIT (commonly cited; verify mirror) | Mask extra | Check the specific weight mirror / commercial terms |
| RTMPose / MMPose | Human keypoints | Apache-2.0 (verify weights) | Person adapter | HumanArt weights need separate check |
| DWPose | Whole-body keypoints | Apache-2.0 (reported; verify) | Fallback | Real-human training; anime may fail |
| SDPose | Anime/cartoon pose | Verify | Optional | Diffusion-based; check weights |
| MediaPipe | Lightweight pose | Apache-2.0 (verify) | Fallback | Good for quick/CPU runs |
| InsightFace ArcFace / SFace | Real-person identity | Verify | v0.2 | Face models have their own terms |
| sentence-transformers MiniLM | Caption embeddings | Apache-2.0 (verify) | Training mode | Check the specific model card |
| CLIP / SigLIP | Text-image similarity | Verify | Optional | OpenAI CLIP terms differ from open checkpoints |
| OpenCV | Local features / geometry / blending | Apache-2.0 | Dedupe | Code dependency |
| HDBSCAN (scikit-learn) | Clustering | BSD-3-Clause | Yes | Code dependency |
| SciPy | Hierarchical clustering | BSD-3-Clause | Yes | Code dependency |
| Gradio | Review UI | Apache-2.0 | UI extra | Code dependency |

## Adding a model

1. Add a row with name, version, license, source, and commercial-use status.
2. Do not mark a model "default" until the license is verified.
3. If the license is non-commercial or unclear, make it opt-in and print a
   warning at download time.
4. Include the upstream license text in the model cache directory or docs.
