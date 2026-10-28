# 通用流水线架构：提取与去重

> 目标：同一套架构适用于角色、真人、物品、海报；不把 Jett、动漫、姿势标签写进核心逻辑。
> 当前代码是实现这一架构的一个 vertical slice，已有部分 adapter 泄漏，需要继续接口化。

## 1. 一句话

```text
几何同源去重（同一张原图的变体）
  + 语义形态聚类（不同原图，但同一姿势/形态）
  + 集合级选择（canonical / stitching / 每形态上限）
  + 可追溯输出（manifest / provenance）
```

三层解决的是不同问题，不能混为一谈：

| 问题 | 判据 | 方法 | 输出 |
|---|---|---|---|
| 完全重复 | 文件/像素一致 | 哈希 | 去掉 |
| **同源变体** | 同一张原图派生：裁剪/缩放/换背景/特效/压缩 | 局部特征 + 单应性 + 边缘验证 | 每个原图一个 canonical |
| **同形态不同原图** | 不同画师/拍摄/生成，但姿势/视角/状态相同 | 形态 embedding + 聚类 | 每个形态一个代表 |
| 不同主体/不同形态 | 不应合并 | 身份过滤/人工 | 保留或隔离 |

## 2. 通用数据模型

```text
Subject（角色/真人/物品/SKU/品牌）
  └─ Version（装扮/版本/状态；可选，默认单例）
       └─ Form（姿势/动作/视角/开合/形状/版式）
            └─ Variant（裁剪/背景/分辨率/格式/压缩）
```

- `dedupe` 处理 **Variant** 层：同一 Form 或同一原图的不同 Variant。
- `extract` 处理 **Form** 层：同一 Subject 下的不同 Form。
- 两者可以组合：先 Variant 去重，再对每个 source 的 canonical 做 Form 聚类。

## 3. 通用流水线

```text
TargetSpec（可选：名称/描述/参考图/auto）
      ↓
Ingest + QC
      ↓
Identity Filter（可选：只保留目标 Subject）
      ↓
Tier 0: Exact Hash
      ↓
Tier 1: Same-Source Dedupe（几何）
  mask（可选）→ 局部特征 → ratio test → RANSAC → 边缘验证
  → 置信度矩阵 → complete/average/single + star merge
  → 每源 canonical（可选 stitching）
      ↓
Tier 2: Form Discovery（语义，可选）
  FormEncoder（person/object/auto/anime）→ 聚类
  → 每形态 canonical + variants
      ↓
Set Selection（reference / training）
      ↓
Normalize / Stitch（提取式）
      ↓
Organize + contact sheet + report + manifest
```

每个 Tier 都可关闭；不同产品模式只是不同组合：

| 模式 | 执行 |
|---|---|
| `dedupe` | Tier 0 + Tier 1 + selection/stitch |
| `extract` | Tier 0/1（可选）+ Tier 2 + selection |
| `organize --auto` | 仅 Tier 0（dHash baseline，smoke test） |
| `training`（后续） | Tier 1 + Tier 2 + per-form caps + caption dedup |

## 4. 核心层 vs 领域适配器

### 核心层（domain-agnostic）

| 模块 | 职责 | 当前文件 |
|---|---|---|
| Ingest/QC | 扫描、哈希、解码、质量/NSFW/多主体 flag | `ingest.py`, `quality.py` |
| Geometry | SIFT/ORB、ratio、RANSAC、单应性、边缘相关 | `source.py` |
| Grouping | confidence matrix、complete/average/single、star merge | `source.py` |
| Stitching | union canvas、mask-aware feather、gain 检查 | `stitch.py` |
| Selection | canonical 质量/完整度评分 | `select.py` |
| Organization | source/form 目录、contact sheet、report、manifest | `organize.py`, `manifest.py` |
| Orchestration | 阶段组合、缓存、错误隔离 | `pipeline.py` |

这些模块只依赖图像、mask、embedding、置信度矩阵，不知道“Jett”“动漫”“姿势名”。

### 领域适配器（可插拔）

| 适配器 | 当前实现 | 通用接口目标 |
|---|---|---|
| Mask provider | 直接调用 `imgutils.get_isnetis_mask`（anime） | `MaskProvider`：none / anime / BiRefNet / SAM / auto |
| Form encoder | `anime-hybrid`（WD14 tags + CCIP） | `FormEncoder`：person / object / auto / anime |
| Identity matcher | 仅有 `IdentityMatcher` Protocol，无实现 | CCIP / ArcFace / OSNet / DINOv2 |
| Segmenter | 仅 anime-seg | anime-seg / BiRefNet / SAM 2 |
| Quality/aesthetic | 通用启发式 | 可插拔模型（NIMA、CLIP-IQA） |
| 3D backend | 无 | 后续 adapter |

**原则**：领域差异只应出现在 adapter 的输入输出，不应进入 `source.py`、`stitch.py`、`organize.py` 的分支逻辑。

## 5. 当前实现的通用性边界

### 已经通用的部分

- SIFT/ORB + RANSAC + 单应性：不依赖主体类型；
- aligned-edge correlation：不依赖姿势/角色；
- complete/average/single + star merge：纯图论/层次聚类；
- union canvas + feather stitching：纯几何/图像融合；
- canonical 选择、目录/报告/manifest：与领域无关；
- CLI/配置驱动，阈值可调；
- Jett 样本只用于验证，没有写进代码。

### 仍然 domain-specific / 需要接口化

1. **Mask 直接依赖 anime-seg**：`source.py` 的 `_mask_for` 直接 import `imgutils`。  
   → 应改为 `MaskProvider`，`auto` 根据域选择；`none` 用于线稿/无背景图。
2. **Form encoder 只有 anime-hybrid**：默认 `anime-hybrid` 是动漫域 adapter。  
   → 保留为 adapter；补 `person`/`object`/`auto`，由配置选择。
3. **Dedupe 不做身份过滤**：默认输入已经是目标 Subject。  
   → 搜索场景应显式加 `IdentityFilter` 阶段；`dedupe` 本身不应猜。
4. **`extract` 没有先跑 `dedupe`**：同源变体会进入形态聚类。  
   → 通用组合应先 `dedupe`，再对 canonical 做 `extract`。
5. **O(n²) pair matching**：100–1000 可用，更大规模需要候选预筛。  
   → `CandidateGenerator`：exact hash → dHash → embedding kNN → 几何验证。
6. **部分参数是域内经验值**：`min_score=0.3`、`star_threshold=0.2` 来自当前样本。  
   → 应提供域预设 + 校准脚本，而不是硬编码。

## 6. 目标接口

```python
class MaskProvider(Protocol):
    def mask(self, record: ImageRecord) -> np.ndarray | None: ...

class LocalMatcher(Protocol):
    def match(self, a: ImageRecord, b: ImageRecord) -> MatchResult: ...

class GroupingStrategy(Protocol):
    def group(self, ids: list[str], confidence: np.ndarray) -> list[list[str]]: ...

class FormEncoder(Protocol):
    def encode(self, record: ImageRecord) -> FormFeatures: ...

class Selector(Protocol):
    def select(self, group: FormGroup, mode: str) -> Selection: ...

class Stitcher(Protocol):
    def stitch(self, members: list[ImageRecord], reference: ImageRecord) -> StitchResult: ...
```

- `DedupeEngine` 组合 `LocalMatcher` + `GroupingStrategy` + `Stitcher`；
- `FormEngine` 组合 `FormEncoder` + `Clusterer` + `Selector`；
- 每个 adapter 通过配置/注册表选择；
- 领域预设只设置 adapter + 阈值，不改变核心算法。

## 7. 领域预设（目标）

| 预设 | Mask | Identity | Form encoder | 备注 |
|---|---|---|---|---|
| `anime-poster` | anime-seg | CCIP/WD14（可选） | anime-hybrid | 当前验证域 |
| `real-person` | BiRefNet/SAM | ArcFace/OSNet | person（关键点/DINOv2） | 注意隐私 |
| `product-object` | BiRefNet | DINOv2 reference | object（视角/状态） | 刚体更容易 |
| `lineart` | none/auto | 可选 | silhouette/edge | SIFT 特征少，需专门匹配 |
| `generic` | auto | 可选 | auto | 兜底 |

## 8. 组合方式

```bash
# 只做同源变体去重
form-coreset dedupe ./images --out ./sources

# 先同源去重，再对每个 source 的 canonical 做形态聚类
form-coreset dedupe ./images --out ./sources
form-coreset extract ./sources/sources/*/canonical.png --out ./forms
```

当前 `extract` 还没自动消费 `dedupe` 输出；下一步应支持：

```bash
form-coreset extract ./images --out ./forms --dedupe-first
```

## 9. 下一步重构优先级

1. **`MaskProvider` 接口**：把 anime-seg 从 `source.py` 抽出；新增 `none`/`auto`/`general`。
2. **`DedupeEngine` 阶段化**：TargetFilter → Tier0 → Tier1 → Select → Stitch → Organize。
3. **`extract --dedupe-first`**：先同源去重，再形态聚类；避免变体污染 Form。
4. **`CandidateGenerator`**：哈希/embedding 预筛，支持更大规模。
5. **FormEncoderRegistry**：anime/person/object/auto，按预设选择。
6. **IdentityFilter**：搜索图先过滤目标 Subject，再做 dedupe。
7. **域预设 + 校准脚本**：阈值来自数据，不来自单个 case。

## 10. 结论

- 当前 `dedupe` 的算法核心是通用的：几何配准、图聚类、stitching、输出契约都不绑定 Jett/动漫；
- 当前仍是“通用核心 + 动漫 adapter”的 vertical slice，不是完整通用平台；
- Jett 样本的作用是验证和校准，不是实现依据；
- 真正的通用化不是“一个模型通吃”，而是“核心算法稳定 + 领域 adapter 可插拔 + 预设可校准”。

---

## 11. 非人脸角色/物品：统一逻辑

范围：动漫/游戏角色、吉祥物、玩具、商品、物品。**不依赖人脸识别，不做 ArcFace/人脸身份/睁眼检测。**

### 11.1 三个不变量

无论主体是角色还是物品，以下三层都不变：

1. **同源关系 = 几何一致性**
   - 局部特征 + RANSAC 单应性 + aligned-edge 验证；
   - 与主体是角色/物品无关；
   - 输出每个原图一个 canonical，可选 stitching。

2. **同形态关系 = masked subject similarity**
   - 先用 MaskProvider 去掉背景/特效；
   - 再用主体特征比较“形态”；
   - 角色：姿势/动作/视角；物品：视角/朝向/开合/组装/形状；
   - 只换 encoder，不换聚类/选择逻辑。

3. **canonical = quality + completeness + representativeness**
   - 分辨率/主体像素、清晰度、完整度、背景干净度、到簇中心距离；
   - 与主体类型无关；
   - 不看人脸、不看睁眼。

### 11.2 统一流水线

```text
SubjectSpec（单一主体；可选参考图）
      ↓
Ingest + QC
      ↓
MaskProvider（anime / general / none / auto）
      ↓
Same-Source Dedupe（SIFT/ORB + RANSAC + edge verify + complete/star）
      ↓
SubjectFormEncoder（DINOv2 + silhouette + optional pose/tags）
      ↓
HDBSCAN / 图聚类
      ↓
Canonical + variants + stitching
      ↓
Organize + manifest
```

### 11.3 领域差异只放在两个 adapter

| 层 | 动漫/游戏角色 | 物品/商品 | 统一接口 |
|---|---|---|---|
| Mask | anime-seg | BiRefNet | `MaskProvider` |
| Form | DINOv2 + silhouette + optional keypoints / tag prior | DINOv2 + silhouette + optional viewpoint/state | `FormEncoder` |
| Identity | 可选 CCIP/角色标签 | 可选 DINOv2/CLIP 参考图 | `IdentityMatcher` |
| Quality | 主体像素/线条/完整度 | 主体像素/纹理/完整度 | `quality.py` |

**相同部分**：同源配准、edge verification、confidence matrix、complete/average/single、star merge、stitching、canonical、manifest。

### 11.4 统一 form encoder 的建议

```text
FormFeatures =
  DINOv2(masked subject crop)
  + w_s * silhouette embedding
  + w_p * optional pose keypoints
  + w_t * optional tag prior
```

- **DINOv2** 是跨角色/物品的主力：视角、姿势、形状、状态都会影响 embedding；
- **silhouette** 抑制背景/特效/颜色，强调形状；
- **pose keypoints** 可选，只对可检测人体的角色有帮助；物品不用；
- **tag prior** 可选，只对动漫域有帮助；不要求用户定义标签，不输出标签；
- 关键点/标签失败时自动降级到 DINOv2 + silhouette。

### 11.5 与“人脸路线”的区别

| 人脸路线 | 本工具 |
|---|---|
| ArcFace/人脸 embedding 做身份 | 不需要；主体通常已知，或视觉参考 |
| 睁眼/闭眼、表情、人脸质量 | 不做；表情/人脸不是形态主信号 |
| 人脸对齐 | 不做；用主体 mask + 轮廓 |
| 人物 ReID | 仅可选；用于多主体批次 |

**含义**：本工具不处理“这是谁的人脸”，只处理“这个已知主体的形态是否重复、哪张最完整”。

### 11.6 当前实现差距

- `dedupe` 的几何核心已经是通用的；
- `MaskProvider` 还没抽出来，`source.py` 直接依赖 anime-seg；
- `anime-hybrid` 是动漫 adapter，通用 `SubjectFormEncoder`（DINOv2 + silhouette）还没实现；
- `extract` 还没先跑 `dedupe`；
- 多主体批次还没有通用 identity filter（应使用参考图/视觉聚类，不用人脸）。

### 11.7 MVP 建议

1. **先做通用 `dedupe`**：它已经适用于角色和物品，且不依赖人脸；
2. **新增 `--domain subject` 预设**：MaskProvider=auto（anime-seg 或 BiRefNet），FormEncoder=DINOv2+silhouette；
3. **`--domain anime`** 作为可选增强：额外启用 WD14 tag prior / 角色标签；
4. **`--domain object`**：BiRefNet + DINOv2 + silhouette + 可选视角/状态；
5. **`dedupe → extract` 组合**：先去除同源变体，再对每个 source 的 canonical 做形态聚类；
6. 不在核心代码里判断“角色/物品/人脸”，只通过 adapter 和配置选择。

**结论**：所有非人脸角色/物品可以共用同一份逻辑；真正需要变化的只有 mask 和 form encoder，且都可以通过 adapter 替换。

---

## 12. 可直接落地的开源组件（2026-10 调研）

只列能落到当前接口、且不改变核心逻辑的组件。

| 组件 | 许可 | 直接优化 | 接入点 | 决策 |
|---|---|---|---|---|
| **ALIKED + LightGlue（ONNX）** | ALIKED BSD-3；LightGlue Apache-2.0；**避免 SuperPoint 权重** | 低纹理/线稿/大视角变化下比 SIFT 更稳 | `LocalMatcher` 后端；先 SIFT 粗筛，再对困难 pair 用 LightGlue | P1 |
| **BiRefNet via rembg** | MIT（权重镜像需核对） | 通用/物体 mask、发丝/边缘更好 | `GeneralMaskProvider`；必须显式 `birefnet-general` / `birefnet-matting`，rembg 默认仍是 U2Net | 已接入，补 matting 预设 |
| **FAISS / hnswlib** | MIT / Apache-2.0 | 候选预筛，避免 O(n²) 全量 SIFT | 新增 `CandidateGenerator`：hash/embedding kNN → 几何验证 | P1 |
| **OpenCV MultiBandBlender / GraphCutSeamFinder** | Apache-2.0 | 减少 stitching 接缝/ghosting | `Stitcher` 的 blend 模式 | P1 |
| **FiftyOne** | Apache-2.0（需核对） | 数据集 review UI、embedding 可视化、near-dup | 可选 UI/集成；MongoDB 较重，不建议进核心 | P2 |
| **CleanVision** | Apache-2.0（需核对） | 质量/坏图检查 | 可选 QC adapter | P2 |
| **imgraft / cull** | 开源 | coreset/整理思路 | benchmark/竞品参考，不作为依赖 | 观察 |

**不建议作为依赖**：

- **fastdup**：CC BY-NC-ND，非商业限制；可作为 benchmark 参考。
- **SuperPoint 权重**：许可不宽松；默认用 SIFT/ALIKED。
- **FiftyOne 进核心**：MongoDB + 重依赖，违背本地轻量定位。

**下一步优先级**：

1. ALIKED + LightGlue 作为可选 matcher（困难 pair 专用）；
2. FAISS/hnswlib 候选预筛（规模）；
3. OpenCV MultiBandBlender（拼接质量）；
4. BiRefNet matting 预设（发丝/透明物体）；
5. FiftyOne 只做可选 UI 集成。
