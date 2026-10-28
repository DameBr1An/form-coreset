# FormCoreset · 技术架构

> 目标：本地优先、CPU 可跑、GPU 可选、模型可插拔、非破坏性、可复现。
> 规模：单次 100–10,000 张图；先服务“搜索结果整理”和“LoRA 训练集整理”。

## 1. 设计目标与约束

1. **本地优先**：图片不上传；默认路径不需要云服务。
2. **CPU-first**：ONNX Runtime 默认；GPU 作为可选加速。
3. **模型可插拔**：身份、分割、姿态、embedding、聚类、选择都定义接口；不绑定单一模型。
4. **许可证干净**：代码与权重分离；不默认打包受限权重；启动时下载并提示许可证。
5. **非破坏性**：原图只读；输出复制/硬链接；所有决策写入 manifest。
6. **可复现**：记录模型版本、阈值、配置、git commit、输入哈希。
7. **人类在环**：身份与形态都可审核、合并/拆分、改选；不确定的进 `uncertain/`。
8. **先窄后宽**：先 anime 角色 + 通用 object，再做真人/复杂场景。

## 2. 总体架构

```text
CLI / Gradio UI
      │
      ▼
RunConfig + TargetSpec（text / references / both / auto）
      │
      ▼
Pipeline Orchestrator（stage 缓存、重跑、manifest）
      │
      ├─ Ingest：扫描、内容哈希、解码、EXIF、alpha
      ├─ QC：分辨率、模糊、曝光、NSFW、文字/水印、多主体标记
      │
      ├─ Identity Stage
      │    ├─ Guided：TextMatcher + ImageMatcher → Fusion → confirmed/uncertain/rejected
      │    └─ Auto：IdentityClusterer → subjects/ + unassigned/
      │
      ├─ Segmentation：主体 mask（anime: isnet-anime；通用: BiRefNet）
      ├─ Same-source Registration：SIFT/LightGlue + RANSAC，合并裁剪/换背景变体
      │
      ├─ Form Encoding
      │    ├─ Person：keypoints + masked DINOv2 embedding
      │    ├─ Object：masked DINOv2 embedding（可选视角/状态）
      │    └─ Auto：masked embedding（+ 检测到的 keypoints）
      │
      ├─ Form Clustering：HDBSCAN / Leiden，多粒度 + 稳定性
      │
      ├─ Selection
      │    ├─ Reference mode：每形态 canonical
      │    └─ Training mode：每形态上限 K + caption 去重 + 集合选择
      │
      ├─ Organizer：confirmed/uncertain/rejected、forms/、contact_sheet、report、manifest
      │
      └─ Review UI：身份校正、形态合并/拆分、粒度调整、改选、导出
```

## 3. 模块与接口

```python
class TargetSpec:
    text: str | None
    reference_images: list[Path]
    mode: Literal["guided", "auto"]

class IdentityMatcher(Protocol):
    def score(self, image: ImageRecord, spec: TargetSpec) -> float: ...

class Segmenter(Protocol):
    def mask(self, image: ImageRecord) -> np.ndarray: ...

class FormEncoder(Protocol):
    def encode(self, image: ImageRecord, mask: np.ndarray) -> FormFeatures: ...

class Clusterer(Protocol):
    def cluster(self, features: np.ndarray, **params) -> ClusterResult: ...

class Selector(Protocol):
    def select(self, clusters: list[FormCluster], mode: str, **params) -> SelectionResult: ...
```

核心原则：**身份、形态、选择是三个独立阶段**。身份错了不应污染形态；形态聚类失败可以降级为“按 embedding 分组”，不应阻断输出。

## 4. 关键选型

| 能力 | 默认 | 降级/备选 | 许可证 | 备注 |
|---|---|---|---|---|
| Anime 身份 | **CCIP**（imgutils，ONNX） | WD14 角色标签；DINOv2 | 需核对 | 768 维，余弦相似度；同角色判断 |
| 通用身份 | **DINOv2 ViT-B/14** | CLIP ViT-B/32 | Apache-2.0 | 主体 mask 上取 embedding |
| 真人身份 | **ArcFace/SFace**（InsightFace） | OSNet / DINOv2 | 需核对 | 人脸不可见时退化到人体 ReID |
| Anime 文本标签 | **WD14 tagger** | CLIP 文本相似度 | 需核对 | character threshold 约 0.85；标签名 `name_(series)` |
| Anime form encoder | **WD14 tags + CCIP embedding**（70/30） | keypoints 仅高置信度时辅助；DWPose/RTMPose 仅作诊断 | 需核对 | 基准：keypoints 在动漫上不可靠；tags HDBSCAN AMI ~0.5、purity ~0.8 |
| 通用姿态 | **RTMPose / DWPose** | MediaPipe | Apache-2.0（需核对权重） | 133/26/33 keypoints 按引擎 |
| Anime 分割 | **isnet-anime**（rembg） | anime-seg / BiRefNet | 需核对 | 约 176MB ONNX；扁平面部/线稿友好 |
| 通用分割 | **BiRefNet** | SAM 2 | 需核对 | 照片/发丝/透明材质更强 |
| 同源配准 | **SIFT + LightGlue** 或 SIFT+RANSAC | ALIKED+LightGlue | LightGlue Apache-2.0；SuperPoint 权重受限 | 默认避开 SuperPoint 权重 |
| 形态 embedding | **DINOv2 ViT-B/14** | CLIP；CCIP（仅身份） | Apache-2.0 | 主体 mask + crop 后编码 |
| 聚类 | **HDBSCAN**（L2 归一化 + cosine） | Leiden / 层次聚类 | BSD/MIT | 小数据用 `min_cluster_size=2–5`；UMAP 可选 |
| 集合选择 | **Facility Location / Graph Cut** 贪心 | MMR / k-medoids | — | 代表性 vs 多样性的权衡 |
| Caption embedding | **sentence-transformers MiniLM** | CLIP text；tag overlap | Apache-2.0 | 训练模式可选 |
| 质量评分 | OpenCV Laplacian/Tenengrad + 分辨率 + 曝光 | 美学模型（弱信号） | — | 动漫美学模型迁移有限 |
| UI | **Gradio**（MVP） | FastAPI + React | Apache-2.0/MIT | CLI 是一等公民 |
| 存储 | **SQLite + JSONL manifest** | Parquet | — | 内容哈希缓存 embedding |

> **许可证提醒**：DINOv2、LightGlue 已确认 Apache-2.0；SuperPoint 权重不是宽松许可，默认不要用。CCIP、WD14、isnet-anime、rtmlib、DWPose、BiRefNet 的权重许可证需要在发布前逐一核对，并在 README 中列出。

## 5. 算法细节

### 5.1 身份匹配

**输入模式**

```text
text:        name / description
references:  1–5 张参考图
both:        text + references
auto:        无目标；先身份聚类，再让用户选簇
```

**文本匹配**

- Anime：
  - WD14 character tags → `{name_(series): confidence}`；
  - 角色名做别名/归一化匹配（`hatsune miku` → `hatsune_miku_(vocaloid)`）；
  - 匹配不到时退到 CLIP 文本-图像相似度。
- 通用：
  - CLIP/SigLIP 文本-图像相似度；
  - 描述是弱先验，不作为硬过滤。

**参考图匹配**

- Anime：CCIP embedding，与参考图取平均/最大相似度；
- 真人：ArcFace 人脸 + OSNet 人体；
- 通用：DINOv2 / CLIP；
- 1 张参考图可启动，3–5 张更稳；参考图本身要先做质量检查。

**融合**

```text
identity_score = α * image_similarity + (1 - α) * text_similarity
```

- 有参考图：α = 0.75（可配置）；
- 只有文本：α = 0，阈值放宽；
- 文本与参考图冲突：参考图优先，文本用于消歧/命名；
- 各模型分数先做 z-score / rank 校准，再加权。

**输出**

- `confirmed/`：高置信同主体；
- `uncertain/`：低置信、多主体、半身/头像、文字/水印、低质；
- `rejected/`：明确不同主体；
- 不强行二分类；`uncertain` 是产品的一部分。

### 5.2 同源配准（先做，优先级最高）

目的：合并“同一张图的不同裁剪/缩放/换背景/压缩”变体。

1. 用主体 mask 限制特征提取区域；
2. SIFT/ORB（或 LightGlue）做局部特征匹配；
3. RANSAC 估计单应性；
4. 若 inliers 与重叠率超过阈值，判为同源；
5. 同源组内选主体像素最多、裁剪最少、压缩损伤最小的一张作为 canonical；
6. 保存 homography，供审核 UI 做 diff 叠加。

这一步准确、可解释，能处理大部分“同图不同背景”的情况。AI 重绘/重新生成的变体对不上，交给形态聚类。

### 5.3 形态表示

**Person adapter**

- 姿态：RTMPose + HumanArt（默认）/ DWPose / MediaPipe；
- 归一化：用肩、髋、躯干做 Procrustes（平移/缩放/旋转）；
- 特征：关节角度 + 骨方向向量 + 归一化坐标 + 可见性权重；
- 镜像：与水平翻转比较，取较小距离；
- 外观：主体 mask + crop → DINOv2 embedding；
- 融合：`form_feature = concat(normalize(pose_feat) * w_p, normalize(dino_feat) * w_e)`；
- 关键点置信度低时，自动降低 `w_p`，退化为 embedding-only。

**Object adapter**

- 主体 mask + crop → DINOv2 embedding；
- 视角/状态：优先用“数据驱动的聚类”，不要依赖 CLIP zero-shot 标签；
  - 研究显示 CLIP 对 canonical view 的 zero-shot 识别较弱；
  - 如需标签，用少量样本训练轻量分类器，或让用户在审核 UI 中命名簇。
- 刚体物品的“形态”= 视角/朝向/状态；不同角度是不同形态，不应被配准合并。

**Auto adapter**

- 先用身份聚类分主体；
- 每个主体内用 DINOv2 masked embedding；
- 若检测到人体，附加 keypoints；
- 没有姿态信息时，embedding-only 也能工作，只是粒度更粗。

### 5.4 形态聚类

**预处理**

- 特征 L2 归一化；
- 可选 UMAP 降维（`n_neighbors=10–30`, `min_dist=0.05–0.1`）用于聚类/可视化；
- 用于“选择”的距离仍用原始特征，避免 UMAP 扭曲。

**HDBSCAN 参数（100 张量级）**

- `min_cluster_size=2–5`；
- `min_samples=1–3`；
- `cluster_selection_epsilon` 调小以保留细粒度；
- `metric="cosine"` 或 L2 归一化后欧氏；
- `prediction_data=True`，对高置信噪声做 `approximate_predict` 回收。

**多粒度**

- 对 `min_cluster_size` / `eps` 做网格；
- 用 bootstrap/consensus 计算稳定性；
- 选择稳定切分，或暴露“粒度滑杆”给用户；
- 噪声 → `uncertain/`，不强行归类。

**评估**

- 有标签：pairwise P/R、ARI/AMI、簇纯度；
- 无标签：稳定性、人工审核时间、簇内/簇间距离分布；
- 不用 silhouette 作为主指标（密度聚类 + 噪声不适合）。

### 5.5 集合选择

**Reference mode**

```text
canonical_score =
  w_q * quality
+ w_c * completeness
+ w_b * cleanliness
+ w_r * representativeness (distance to medoid)
+ w_f * fidelity
```

每形态选最高分；保留 `variants/`。

**Training mode**

- 单元是图文对；
- 冗余 = 形态相似 + caption 相似；
- **联合冗余**：`R = max(ft*(γf+(1−γf)ct), ct*(γc+(1−γc)ft))`，用 soft-AND + floors 处理“同形态不同 caption”和“同 caption 不同形态”；详见 [form-caption-redundancy.md](form-caption-redundancy.md)。
- 每形态上限 K；
- 稀有形态优先保留；
- 目标：

```text
maximize  形态覆盖 + caption 概念覆盖 + 样本质量
- λ * 形态内冗余 - μ * caption 冗余
subject to 每个形态 ≤ K 个 pair
```

- 选择算法：Facility Location / Graph Cut 的贪心（子模，1−1/e 保证）；
- 小数据直接贪心即可；不需要 ILP；
- caption 去重：sentence-transformer + tag overlap；可单独关闭；
- 输出 `curated/` + `report.md` + `contact_sheet.png`。

## 6. 数据模型与存储

```text
ImageRecord: id, path, sha256, width, height, format, alpha,
             exif, quality_scores, flags,
             identity_embedding, form_embedding, keypoints,
             subject_id, form_id, status

Subject: id, label, member_ids, confidence, method

FormCluster: id, subject_id, member_ids, canonical_id,
             confidence, method, params

RunManifest: run_id, git_commit, model_versions, thresholds,
             config_hash, input_hash, created_at
```

- v0.1：SQLite（元数据）+ `.npz`（embedding 缓存）+ `manifest.jsonl`（导出）。
- 缓存键：`sha256(image) + model_id + model_version + preprocessing_version`。
- 输出目录：

```text
organized/
  confirmed/
    form_01/
      canonical.png
      variants/
    form_02/...
  uncertain/
  rejected/
  contact_sheet.png
  report.md
  manifest.jsonl
```

Auto 模式：

```text
organized/
  subjects/
    subject_01/forms/...
    subject_02/forms/...
  unassigned/
  contact_sheet.png
  report.md
```

## 7. 性能与部署

- Python 3.11/3.12；`uv` 打包；`typer` CLI；`pydantic` 配置。
- ONNX Runtime CPU 默认；`[gpu]` extra 用 CUDA/TensorRT。
- 模型首次运行下载到 `~/.cache/form-coreset`，带 checksum + 许可证提示；不打包进 wheel。
- 100 张图在 CPU 上应可分钟级完成；具体耗时以 benchmark 为准，不在 README 里拍脑袋承诺。
- 批量 embedding、缓存、可中断/续跑；流水线 stage 可单独重跑。
- 原图只读；输出用复制或硬链接；Windows 硬链接失败时自动复制。

## 8. 评估与 spike 计划

### 必须先做的技术验证

1. **Anime pose spike**
   - 50 张动漫图；对比 RTMPose+HumanArt / DWPose / SDPose / MediaPipe；
   - 指标：关键点置信度、失败率、形态聚类纯度；
   - 结论决定 `person` 适配器的默认引擎。
2. **Identity spike**
   - 100 张搜索结果（含干扰角色）；对比 CCIP / DINOv2 / WD14 tags / 融合；
   - 指标：precision/recall、uncertain 比例、阈值稳定性。
3. **Background invariance**
   - 同形态不同背景 vs 同背景不同形态；
   - 对比 masked / unmasked embedding；配准开关；
   - 结论决定 segmentation 是否是必需步骤（大概率是）。
4. **Auto mode stability**
   - 无目标输入；多次 bootstrap 的簇数/纯度；
   - 评估用户选簇时间。
5. **Training benchmark**
   - 全量 vs 筛选后 LoRA；相同步数/更少步数对比；
   - 指标：验证 loss、prompt adherence、多样性、人工偏好。
6. **License audit**
   - CCIP / WD14 / isnet-anime / rtmlib / DWPose / BiRefNet / SAM2；
   - 输出 `THIRD_PARTY_LICENSES.md`。

### 基准数据集

- 3 个域：anime 角色、真人、商品/物体；
- 每个域 100 张，含干扰项；
- 标注：identity（同/不同）、form（同/不同）、canonical 偏好；
- 公开版本放 Hugging Face / GitHub Release；注意版权，只放可再分发样本。

## 9. 风险与降级路径

| 风险 | 影响 | 降级路径 |
|---|---|---|
| Anime 姿态模型失败 | 形态聚类退化 | embedding-only + WD14 tags + 人工审核 |
| CCIP 混淆相似角色 | 身份过滤错误 | 提高阈值 → `uncertain/`；参考图增多；人工校正 |
| 背景主导 embedding | 同形态不同背景被拆开 | 强制 mask；配准；背景去除后再编码 |
| HDBSCAN 不稳定 | 簇数跳变 | 多粒度 consensus + 粒度滑杆 + 审核 UI |
| CLIP zero-shot 视角弱 | 物体形态标签不可靠 | 不依赖标签；DINOv2 聚类 + 用户命名 |
| 模型许可证受限 | 无法打包/商用 | 权重分离；默认用 Apache-2.0 模型；文档提示 |
| CPU 太慢 | 体验差 | ONNX + 小模型 + 缓存；GPU extra |
| 过度合并 | 丢信息 | 默认非破坏；variants 保留；不确定进 uncertain |

## 10. 里程碑

| 阶段 | 内容 | 产出 |
|---|---|---|
| **M0 技术验证** | 5 个 spike + 基准集 | 选型报告、阈值、失败模式 |
| **M1 可跑通** | ingest、QC、guided identity（参考图）、segmentation、form clustering、输出 | CLI 能处理 100 张搜索图 |
| **M2 自动模式** | auto identity clustering、文本匹配、Gradio 审核 UI | `--auto` 可用 |
| **M3 训练模式** | caption 去重、每形态上限、facility location、report | LoRA 数据集整理 |
| **M4 扩展** | object adapter、真人 adapter、归一化、插件 | 多域可用 |

## 11. 来源

- [DWPose / RTMPose / HumanArt 讨论（ComfyUI-Unblend pose_engine）](https://github.com/LatentDesireAI/ComfyUI-Unblend/blob/main/pose_engine.py)
- [imgutils DWPose API](https://dghs-imgutils.deepghs.org/main/_modules/imgutils/pose/dwpose.html)
- [SDPose / ComfyUI-SDPose-OOD](https://comfy.icu/node/SDPoseOODProcessor)
- [mascot-pose-detect（Hugging Face）](https://huggingface.co/grmchn/mascot-pose-detect)
- [rembg 模型列表 / isnet-anime](https://deepwiki.com/danielgatis/rembg/5.1-available-models)
- [BiRefNet / ComfyUI 讨论](https://comfy.icu/node/Anime-Character-Seg)
- [DeepCore：coreset selection methods](https://github.com/Daidj/DeepCore)
- [CORDS：coreset / data subset selection](https://pypi.org/project/cords/0.0.2/)
- [SubZeroCore（facility location 等子模方法）](https://ar5iv.labs.arxiv.org/html/2509.21748)
- [HDBSCAN + embeddings 实践](https://www.mdpi.com/2078-2489/17/5/422/pdf)
- [Paparazzi：CLIP 视角描述 grounding](https://aclanthology.cn/2023.findings-eacl.62/)
- [Not all Views are Created Equal（ viewpoint instability）](http://export.arxiv.org/pdf/2412.19920v1)
- [WD14 / imgutils tagging](https://deepwiki.com/deepghs/imgutils/4-image-tagging)
- [DINOv2 许可证说明](https://playground.roboflow.com/models/meta/dinov2)
- [LightGlue（Apache-2.0）](https://raw.githubusercontent.com/cvg/LightGlue/ecc01b4d49a70b2fedda2b0b7fc24c49e4d39756/README.md)
- [LightGlue-ONNX](https://github.com/fabio-sim/lightglue-onnx)
