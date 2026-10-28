# FormCoreset 产品需求文档（PRD）

> 版本：v0.1 Draft  
> 日期：2026-10-25  
> 状态：待评审  
> 产品代号：FormCoreset（已确认）  
> 文档定位：v0.1 开发的唯一需求源；此前的调研/定位/架构文档作为背景材料

**相关文档**

- [市场调研](market-research.md)
- [开源定位](positioning.md)
- [MVP 设计（部分被本文替代）](mvp-design.md)
- [技术架构](architecture.md)

---

## 0. 决策摘要（TL;DR）

- **产品**：开源、本地优先的“形态感知视觉核心集提取器”。
- **核心场景**：搜索引擎下载 100 张同一主体（角色/人/物）的图 → 过滤非目标 → 合并同形态不同背景/裁剪的变体 → 按形态整理成可复核集合。
- **第二场景**：整理 LoRA/扩散模型训练集，减少同一形态图文对的重复曝光，降低冗余导致的过拟合。
- **输入**：文本（名字/介绍）、参考图、两者、完全不给（`--auto`）。
- **输出**：`confirmed/`、`uncertain/`、`rejected/`、`forms/form_XX/{canonical,variants}`、`contact_sheet.png`、`report.md`、`manifest.jsonl`；auto 模式输出 `subjects/` + `unassigned/`。
- **v0.1**：CLI + Gradio 审核 UI；`person` + `auto` 适配器；`object` 实验性；训练模式基础版；不做生成式、云端、输出归一化。
- **技术**：Python + ONNX Runtime，CPU 优先；身份 CCIP/WD14/DINOv2/ArcFace；形态 RTMPose+HumanArt + DINOv2；分割 isnet-anime/BiRefNet；聚类 HDBSCAN；集合选择 facility location。
- **许可证**：代码 Apache-2.0；模型权重与代码分离，逐一审计。

---

## 1. 背景与问题

### 1.1 问题

用户从搜索引擎、图站、素材库或自己的相册里收集同一主体（角色、真人、商品、物体）的图片时，会得到大量“本质重复”的图：

- 同一张图的不同裁剪、缩放、压缩格式、背景替换；
- 同一形态的不同版本（同姿势不同背景、同角度不同白底/透明底）；
- 同一主体但不同形态（不同姿势、动作、视角、开合状态、形状）；
- 完全不是目标主体的干扰图（相似角色、其他商品、无关图）；
- 训练场景中，同一形态的图文对被反复采样，导致过拟合或浪费算力。

手动整理 100 张图通常需要 1–2 小时，且难以复现。现有工具各有缺口：

- 消费级照片清理工具按场景/时间/人脸分组，不理解“形态”，也不输出可用的标准资产；
- 数据集去重工具（fastdup、SemDeDup、NeMo Curator）按图像 embedding 去重，不建模 caption，也不按形态封顶；
- 角色专用工具（imgutils/waifuc）能做身份聚类和差分过滤，但不做形态发现、集合级选择和整理输出；
- 商品图 API 做单图去背/裁切，不做多图比较与选择。

### 1.2 机会

“形态感知 + 集合级选择 + 可复核整理”目前没有端到端开源工具。`coreset`（核心集）本身就是训练数据选择的术语，适合承载“少而全”的价值主张。

---

## 2. 产品定位与原则

### 2.1 一句话定位

> **FormCoreset 把同一主体的一堆重复图文对，压缩成一组“每个不同形态只保留少量高价值样本”的最小集合——用于参考图库，或用于模型训练，避免同一形态被反复训练导致过拟合。**

主 tagline：**Keep the information. Drop the duplicates.**  
训练场景 tagline：**Train on what's different.**

### 2.2 定位声明

> 面向 AI 画师、模型训练者、游戏/动漫素材整理者、商品素材管理者：FormCoreset 是一个开源的“形态感知数据集整理器”。它识别同一形态的冗余图文对，保留一个信息覆盖完整、样本量最小的子集；本地运行、可复现、不上传、非破坏性。它不是只删除文件的去重工具，而是优化整个集合。

### 2.3 产品原则

1. **Set-level，不是 file-level**：优化“这组图放在一起的信息量”，不是逐张打分。
2. **Form-aware**：按形态（姿势/角度/状态/形状/版式）去重，不按像素或场景。
3. **信息覆盖优先**：宁可多留一张，也不丢一个不同形态。
4. **默认非破坏性**：原图只读；不确定进 `uncertain/`；所有决策可追溯。
5. **本地优先**：默认不上传；CPU 可跑，GPU 可选。
6. **人类在环**：身份与形态可审核、合并/拆分、改选。
7. **可插拔**：模型、聚类、选择、归一化可替换。
8. **先窄后宽**：先做好“anime 角色 + 搜索整理”，再扩展真人/物品/商品。

### 2.4 关键概念

| 概念 | 定义 |
|---|---|
| **主体（Subject）** | 要整理的对象：角色、真人、商品 SKU、物体。 |
| **形态（Form）** | 让两张图“携带不同信息”的视觉状态：姿势/动作/表情/服装/视角；物品的视角/朝向/开合/组装/形状；海报的版式/构图。 |
| **变体（Variant）** | 同形态下的重复版本：裁剪、缩放、背景替换、分辨率、格式、压缩。通常应被去重。 |
| **同源变体** | 同一张原图派生的变体，可用局部特征配准精确对齐。 |
| **标准图（Canonical）** | 一个形态簇中价值最高、最适合代表该形态的图。 |
| **核心集（Coreset）** | 覆盖不同形态、冗余最小、价值最高的一组图。 |
| **不确定桶（Uncertain）** | 低置信、多主体、半身/头像、文字/水印、低质量等，需人工复核。 |

---

## 3. 目标与非目标

### 3.1 产品目标

| 编号 | 目标 | 衡量方式 |
|---|---|---|
| G1 | 把 100 张混合搜索结果整理成可用的角色形态集 | 人工复核 ≤ 10 分钟；身份 confirmed 精确率 ≥ 0.95 |
| G2 | 合并“同形态不同背景/裁剪”的变体 | 同源/同形态召回率 ≥ 0.85；误合并率 ≤ 5% |
| G3 | 发现不同形态，每形态输出 canonical + variants | form pairwise recall ≥ 0.85；人工一致率 ≥ 0.80 |
| G4 | 训练模式减少重复形态曝光 | 用 ≤ 70% 的图文对达到与全量相同或更好的 LoRA 效果 |
| G5 | 开源、本地、可复现、非破坏性 | 5 分钟跑通 demo；无上传；manifest 可复现 |

> 以上为 v0.1 初始目标，需用 benchmark 校准；不达标时先修算法，不靠调报告口径。

### 3.2 非目标（v0.1）

- 不做云端服务、账号、团队协作、权限管理。
- 不做生成式补全、重绘、扩图、统一光照。
- 不做 web-scale（百万级）训练数据去重。
- 不做自动删除原图。
- 不做多角色同图拆分（v0.1 标记为 `uncertain/`）。
- 不做输出归一化（去背、统一画布；v0.1 的分割只用于形态分析）。
- 不做真人身份识别（v0.1 可标记为实验性，正式支持放 v0.2）。

---

## 4. 用户与场景

### 4.1 用户画像

| 画像 | 主要诉求 | 当前替代方案 |
|---|---|---|
| **AI 画师 / LoRA 训练者**（第一优先） | 整理角色参考集；减少训练集重复形态；避免过拟合 | 手动挑图、waifuc、Kohya 数据准备 |
| **游戏/动漫/TRPG 素材整理者** | 立绘差分、表情差分、参考表、pose sheet | 手动分类、Chara Sorter、文件夹管理 |
| **商品/电商素材管理者** | 同 SKU 多角度/多状态整理；每个角度一张标准图 | Photoroom、人工挑选 |
| **摄影选片** | 连拍/同场景中每个姿势/构图选最佳 | PhotoSort、PixTwin、人工 |
| **开源贡献者/插件开发者** | 接自己的模型、聚类、选择器 | 无统一框架 |

### 4.2 核心场景

#### S1：100 张搜索结果整理（P0）

> 我从搜索引擎下载了某角色的前 100 张图。里面有些不是这个角色；是这个角色的图也可能同形态不同背景。我想过滤、去重、按形态整理。

**流程**

1. 用户指定目标：`--name` / `--description` / `--reference` / 两者 / `--auto`。
2. 工具扫描、质量检查、身份过滤，输出 `confirmed/`、`uncertain/`、`rejected/`。
3. 对 confirmed 做主体分割、同源配准、形态聚类。
4. 每个形态输出 canonical + variants。
5. 生成 contact sheet、report、manifest。
6. 用户在 Gradio UI 中复核：改身份、合并/拆分形态、改选 canonical、导出。

**验收**

- 100 张图端到端 ≤ 5 分钟（CPU，不含首次模型下载）。
- 不删除原图；所有结果可追溯到源文件。
- `report.md` 的数量与目录一致。
- 复核后可直接作为参考集使用。

#### S2：LoRA 训练集去重（P0 基础，P1 完整）

> 我有 500 个图文对，同一个姿势/角度重复很多。我希望每个形态最多保留 K 个，且不要丢掉稀有形态。

**流程**

1. 输入图片目录 + Kohya `.txt` caption / JSONL。
2. 形态聚类 + caption 相似度分析。
3. 选择策略 `strict` / `balanced` / `diverse`，每形态上限 K。
4. 输出 `curated/` + `report.md`（每个形态保留/删除数量、caption 冗余、覆盖率）。
5. 可选 Kohya 导出。

**验收**

- 报告准确反映保留/删除数量。
- 稀有形态不被全局 top-K 挤掉。
- 下游 LoRA benchmark 达到 G4。

#### S3：商品/物体固定角度整理（P1 实验性）

> 同一 SKU 的正面/侧面/背面/细节图，按视角整理，每个视角一张标准图。

**流程**：同 S1，但使用 `--adapter object`；形态主要由 DINOv2 embedding + 聚类发现；视角标签可选。

**验收**：不同视角能被分开；同视角不同背景能被合并；不依赖 CLIP zero-shot 视角标签。

#### S4：完全自动聚类（P0）

> 我不确定目标是谁/是什么，先让工具把所有图按主体和形态聚类，我再选。

**流程**：`--auto` → 身份聚类 → 每个主体内形态聚类 → `subjects/subject_XX/forms/` + `unassigned/`。

**验收**：不假设最大簇就是目标；用户可在 UI 中选择目标簇；噪声进 `unassigned/`。

#### S5：人工复核（P0）

> 工具分错了，我要能快速改。

**流程**：Gradio UI 中查看 confirmed/uncertain/rejected；拖拽/勾选改身份；形态卡片合并/拆分；改选 canonical；调整粒度；导出。

**验收**：100 张图复核 ≤ 10 分钟；支持撤销；不破坏原图。

#### S6：导出到训练工具（P1）

> 把整理结果直接给 Kohya / diffusers / ComfyUI 用。

**验收**：Kohya 目录 + caption 可用；manifest 包含来源和选择理由。

---

## 5. 版本范围

### 5.1 v0.1（MVP）

**必须做（P0）**

- 输入：文件夹；文本名字；参考图 1–5 张；文本+参考图；`--auto`。
- 身份过滤：CCIP（anime）、DINOv2（通用）、WD14 角色标签（anime 名字）；三桶输出。
- 自动身份聚类 + 用户选簇。
- 主体分割（isnet-anime / BiRefNet），仅用于形态分析。
- 同源配准（SIFT + LightGlue / RANSAC）。
- 形态表示：`person`（RTMPose+HumanArt + DINOv2）、`auto`（DINOv2）。
- 形态聚类：HDBSCAN，多粒度，噪声处理。
- Reference mode：每形态 canonical + variants。
- Training mode 基础：每形态上限 K、`strict/balanced/diverse`、可选 caption 相似度、报告。
- 输出目录、contact sheet、report、manifest。
- CLI + Gradio 审核 UI。
- 非破坏性、缓存、可复现。

**实验性（P1）**

- `object` 适配器（DINOv2 聚类；视角标签可选）。
- 自由文本介绍（CLIP 弱先验）。
- caption-aware 联合选择（形态 + caption 双维度）。
- 插件接口公开。
- Kohya / diffusers 导出。
- 自动簇命名。

**不做（P2+）**

- 真人 ArcFace/ReID 正式支持。
- 多角色同图拆分。
- 输出归一化（去背、统一画布）。
- ComfyUI 节点、FiftyOne 插件。
- 云端/协作/账号。
- 生成式补全。

### 5.2 优先级定义

- **P0**：v0.1 发布门槛；缺失则不能发布。
- **P1**：v0.1 尽力交付；可推迟到 v0.2，但接口要预留。
- **P2**：明确不在 v0.1；记录在 roadmap。

---

## 6. 功能需求

### FR-01 扫描与质量检查（P0）

**描述**：扫描输入目录，建立图像记录，执行基础质量检查。

**规则**

- 支持 jpg/jpeg/png/webp；GIF/动画 WebP 只取首帧或跳过并警告；HEIC/AVIF 需可选依赖，v0.1 可标记不支持。
- 读取 EXIF 方向并应用；保留 alpha；读取 ICC（v0.1 不做色彩转换）。
- 计算 SHA-256；相同内容只保留一条记录（或标记重复）。
- 质量检查：解码失败、分辨率过低（默认短边 < 256）、模糊（Laplacian 方差）、过曝/欠曝、单色、NSFW（WD14 rating，可选）、文字/水印（OCR 可选）、多主体（人物检测数量）。
- 所有异常写入 flags，不中断整批。

**验收标准**

- 100 张图扫描 ≤ 10 秒（不含模型加载）。
- 损坏文件被跳过并记录；进程不崩溃。
- EXIF 旋转后的可视化方向正确。
- 重复内容被去重或标记。

### FR-02 目标指定（P0）

**描述**：支持四种目标输入方式，并写入 manifest。

**规则**

- `--name "..."`：角色名/主体名。
- `--description "..."`：自由描述（P1 实验性）。
- `--reference img1.png img2.png`：1–5 张参考图；超过 5 张警告但可用。
- 两者可同时提供；参考图权重更高。
- `--auto`：无目标，自动聚类。
- 无参数且非 `--auto`：报错并提示可用方式。
- 参考图本身先过质量检查；不可读则报错。

**验收标准**

- 四种方式均有 CLI 与 UI 入口。
- 非法参数给出可操作错误信息。
- TargetSpec 完整写入 manifest。
- 参考图数量/格式校验通过。

### FR-03 身份过滤（P0）

**描述**：在 guided 模式下，把输入分成 confirmed / uncertain / rejected。

**规则**

- Anime 参考图：CCIP 相似度；通用参考图：DINOv2/CLIP；真人（P2）：ArcFace/OSNet。
- Anime 名字：WD14 character tags；阈值默认 0.85；支持别名归一化（如 `hatsune miku` → `hatsune_miku_(vocaloid)`）。
- 文本+参考图融合：`score = α*image + (1-α)*text`，有参考图时 α 默认 0.75。
- 两个阈值：`confirmed_threshold`、`rejected_threshold`；中间进 `uncertain/`。
- 每条记录保存分数、命中标签、匹配模型、原因。
- 多主体、半身/头像、文字/水印、低质量默认进 `uncertain/`。
- 用户可在 UI 中覆盖。

**验收标准**

- Benchmark：confirmed 精确率 ≥ 0.95；recall ≥ 0.90（含 uncertain 回收）。
- 相似角色不会被大量误判为 confirmed。
- 每个结果有可解释的分数/原因。
- 无参考图/无标签时不会崩溃，进入 uncertain。

### FR-04 自动身份聚类（P0）

**描述**：`--auto` 模式下，按身份聚类所有图片。

**规则**

- Anime：CCIP embedding；通用：DINOv2/CLIP；真人：P2。
- HDBSCAN / Leiden；L2 归一化；小数据用 `min_cluster_size=2–5`。
- 输出 `subjects/subject_XX/`，按大小/质量排序；噪声进 `unassigned/`。
- 不假设最大簇是目标；UI 中由用户选择目标簇。
- 可选簇命名：WD14 角色标签（anime）或 CLIP zero-shot；命名失败则用 `subject_XX`。
- 用户选定目标簇后，后续形态流程只处理该簇；其他簇保留在 `subjects/`。

**验收标准**

- 100 张混合图能分出主要身份簇；纯度和稳定性达到 benchmark 目标。
- 不把 `unassigned/` 强行并入。
- 用户选簇后流程可继续。
- 输出不丢失任何图片（除非明确 rejected）。

### FR-05 主体分割（P0）

**描述**：为形态分析提供主体 mask/bbox。

**规则**

- Anime：isnet-anime；通用：BiRefNet；降级：rembg u2net。
- 输出 mask、bbox、alpha；失败时标记并回退到整图。
- 分割只用于分析，不改变最终输出图（v0.1 不做归一化输出）。
- 支持多主体检测；多主体图标记 `multi_subject`，进 uncertain。

**验收标准**

- 主要 benchmark 图上 mask 可用；失败可感知。
- 分割失败不阻断聚类（回退整图）。
- 不修改原图。

### FR-06 同源变体配准（P0，优先路径）

**描述**：合并同一张原图的不同裁剪、缩放、换背景、特效、压缩版本。用户的主要场景是“同一张海报/原图的变体”，因此这是第一优先路径，而不是姿态聚类。完整逻辑见 [same-source-logic.md](same-source-logic.md)。

**规则**

- CLI：`form-coreset dedupe <input> --out <dir>`。
- 默认：SIFT + ratio test + RANSAC 单应性；ORB 可选。
- **分组默认 complete linkage**：组内任意两张都必须直接匹配，避免 A~B、B~C、A≁C 的链式过度合并；`--linkage average|single` 可切到 recall 优先。
- **star merge 默认开启**：complete 种子组生成后，把与组内 medoid 高置信匹配的 singleton 拉回；只允许 singleton → group，避免组间链式合并；`--no-star-merge` 可关闭，`--star-threshold` 默认 0.20。
- **anime 角色 mask 默认开启**：isnet-anime 去背后再提特征；`--no-mask` 可关闭。
- **aligned-edge 验证默认开启**：用单应性把一张图 warp 到另一张，比较边缘图相关性；`--no-verify` 可关闭。
- **canonical stitching 默认开启**：对齐后拼接 union canvas；若覆盖面积增益 ≥ `stitch_min_gain`，输出 `canonical.png`（拼接版），同时保留 `canonical_single.png`；`--no-stitch` 可关闭。
- 默认阈值：`min_score=0.3`、`min_inliers=12`、`ratio=0.75`、`max_features=2000`、`min_correlation=0.35`。
- 通过阈值的两张图写入置信度矩阵；按 complete linkage 分组（可选 average/single）。
- 组内选择主体像素最多、裁剪最少、压缩损伤最小的一张作为 single canonical；保留 variants。
- 输出 `sources/source_XXX/{canonical.png,canonical_single.*,variants/,source.json}`，**不输出姿势标签**。
- 不同原图不应被误合并；阈值偏保守，可在真实数据上调整。
- 配准失败/低特征图 → singleton 或 `uncertain/`，不阻断。

**验收标准**

- 合成同源基准（8 原图 × 7 变体，含裁剪/缩放/JPEG/换背景/亮度/特效）：
  - no mask, no verify, `min_score=0.3`：pairwise precision / recall / F1 = 1.0。
  - mask only, `min_score=0.3`：precision 0.774（会把相似原图合并）。
  - **mask + edge verify, `min_score=0.3`：**
    - single linkage：8 groups，precision / recall / F1 = 1.0（有链式风险）。
    - average linkage：9 groups，precision 1.0 / recall 0.964 / F1 0.982。
    - complete linkage：10 groups，precision 1.0 / recall 0.935 / F1 0.966；2 个弱变体被保守拆成 singleton。
    - **complete + star merge（默认）：8 groups，precision / recall / F1 = 1.0**。
  - no mask, `min_score=0.15`：precision 0.533，明显过合并。
- 真实数据上精确率 ≥ 0.95；不同原图误合并率 ≤ 5%。
- canonical stitching：互补裁剪能拼出比单张更大的图；否则回退 single canonical。
- 输出不含姿势标签；报告只含 source groups、singletons、variants removed、stitched groups。
- 配准/拼接失败时不影响其他组。

### FR-07 形态表示（P0）

**描述**：把每张图编码成可用于聚类的形态特征。

**规则**

- `anime-hybrid` 适配器（**默认，动漫/风格化**）：
  - **WD14 tags 为主特征**（内置 posture / view / action tag 先验），默认权重 0.7；
  - **CCIP embedding 为辅助**，默认权重 0.3；CCIP 是身份模型，不作为姿势主信号；
  - **keypoints 不作为默认特征**。基准显示 DWPose / RTMPose HumanArt 在动漫上不可靠（见 `benchmarks/RESULTS.md`）；仅在高置信度时作为可选辅助；
  - **输出不包含姿势名称/标签**；分组只用 `form_001` 这样的 cluster ID。不需要用户定义或标注姿势名；
  - 聚类本身是无监督的（HDBSCAN）；内置 tag 先验只作为特征，不参与标签监督；
  - 完全无先验的 WD14 embedding / TF-IDF / DINOv2 在动漫 benchmark 上明显更弱；因此保留内置 tag 先验作为工程折中；
  - 编码失败 → `uncertain/`，不中断整批。
- `person` 适配器（真人）：
  - 姿态：RTMPose + HumanArt 默认；DWPose / MediaPipe 备选；SDPose 可选。
  - Procrustes 归一化（肩/髋/躯干）；特征 = 关节角度 + 骨方向 + 归一化坐标 + 可见性。
  - 镜像感知。
  - 外观：主体 mask + crop → DINOv2 embedding。
  - 关键点失败时降低姿态权重，退化为 embedding-only。
- `object` 适配器（P1 实验性）：
  - 主体 mask + crop → DINOv2 embedding。
  - 视角/状态标签可选；不依赖 CLIP zero-shot（研究显示不可靠）。
  - 如需标签，用少量样本训练轻量分类器或人工命名。
- `auto` 适配器（P0）：
  - 检测到人体则附加 keypoints；否则仅 embedding。
- 特征缓存键：图像哈希 + 模型版本 + 预处理版本。

**验收标准**

- 三种适配器均可输出特征；失败有明确降级日志。
- 特征维度/版本写入 manifest。
- 关键点失败时仍能聚类。
- 同一输入/配置下特征可复现。
- `anime-hybrid` 在姿势均衡基准上可复现；噪声/失败进入 `uncertain/`。
- Form Library 报告包含 encoder 版本、簇数、noise/uncertain 数量；不输出姿势标签。

### FR-08 形态聚类（P0）

**描述**：把同一主体内的图片按形态分组。

**规则**

- 特征 L2 归一化；可选 UMAP 降维（仅用于聚类/可视化；选择用原始特征）。
- HDBSCAN 默认；备选 Leiden / 层次聚类。
- 100 张量级：`min_cluster_size=2–5`，`min_samples=1–3`，`cluster_selection_epsilon` 可调。
- 多粒度：网格参数 + bootstrap/consensus 稳定性；暴露“粒度滑杆”。
- 噪声 → `uncertain/`；高置信噪声可 `approximate_predict` 回收。
- UI 支持合并/拆分；每次修改写入 manifest。

**验收标准**

- Benchmark：form pairwise recall ≥ 0.85；簇纯度 ≥ 0.85。
- 同一配置 + seed 输出稳定。
- 粒度调整可复现。
- 不会因为聚类失败丢失图片。

### FR-09 Canonical 选择（P0）

**描述**：为每个形态选一张标准图。

**规则**

- 评分：质量 + 完整度 + 干净度 + 代表性 + 保真度。
- 质量：有效分辨率、清晰度、压缩、噪点。
- 完整度：主体 bbox、是否截断、全身/半身。
- 干净度：背景复杂度、文字/水印、其他人物。
- 代表性：到簇中心/medoid 的距离。
- 保真度：颜色/线条/无过度放大。
- 预设：最标准 / 最清晰 / 最完整 / 干净背景。
- 展示分数拆解；保留 variants。

**验收标准**

- 每个形态有且仅有一张 canonical。
- 分数拆解可见、可解释。
- Benchmark 人工一致率 ≥ 0.80。
- 不会选到明显截断/水印/低清图。

### FR-10 训练模式（P0 基础 / P1 完整）

**描述**：为 LoRA/扩散微调输出去重后的图文对集合。

**规则**

- 输入：图片 + 可选 caption（Kohya `.txt` / JSONL）。
- 单元是图文对；无 caption 时退化为形态去重。
- 策略：
  - `strict`：每形态 1 个 pair；
  - `balanced`（默认）：每形态 ≤ K；
  - `diverse`：只去近重复，保留背景/光照/风格变化。
- 稀有形态优先保留；主导形态封顶。
- caption 相似度：sentence-transformer + tag overlap；可关闭。
- 联合冗余度量：`R = max(ft*(γf+(1−γf)ct), ct*(γc+(1−γc)ft))`；实现与参数见 [form-caption-redundancy.md](form-caption-redundancy.md)。
- 集合选择：facility location / graph cut 贪心；小数据直接贪心。
- 输出 `curated/` + `report.md` + `contact_sheet.png`。
- 报告：每形态保留/删除数量、caption 冗余、覆盖率、稀有形态。
- 可选 Kohya 导出（P1）。

**验收标准**

- 报告数量与实际目录一致。
- 每个形态不超过 K。
- 稀有形态保留率 ≥ 90%。
- 下游 LoRA benchmark 达到 G4（P1 验证）。
- caption 缺失/格式错误不崩溃。

### FR-11 整理与导出（P0）

**描述**：生成稳定、可读、可追溯的输出。

**规则**

- 目录结构固定；文件名净化；长路径处理。
- `contact_sheet.png`：每个形态一行/网格，标注形态 ID、canonical、数量。
- `report.md`：输入数、confirmed/uncertain/rejected、形态数、每形态数量、稀有形态、运行时间、模型版本。
- `manifest.jsonl`：每张图的来源、哈希、状态、分数、形态 ID、canonical 标记、flags。
- 原图只读；输出用复制/硬链接；失败自动复制。
- 不覆盖已有输出目录，除非 `--overwrite`。
- 导出预设：generic / kohya（P1）/ diffusers（P1）。

**验收标准**

- 输出目录可被人工直接使用。
- report 与目录一致。
- manifest 可被脚本读取。
- 原图未被修改（校验哈希）。
- Windows/macOS/Linux 路径可用。

### FR-12 审核 UI（P0）

**描述**：本地 Gradio UI，支持人工复核。

**规则**

- 身份视图：confirmed/uncertain/rejected 网格；大图预览；批量改状态；显示分数/原因。
- 形态视图：簇卡片；缩略图；合并/拆分；设为 canonical；调整粒度并重算；同源 diff 叠加。
- 训练视图：每形态数量、K 滑块、策略选择、预览保留/删除、报告预览。
- 导出按钮。
- 撤销/重做；所有修改写 manifest。
- 键盘快捷键（至少：上一张/下一张、确认、拒绝、合并）。
- 100 张图复核目标 ≤ 10 分钟。

**验收标准**

- 主要流程可只用 UI 完成。
- 修改可撤销。
- UI 不删除原图。
- 刷新后状态保留。

### FR-13 CLI（P0）

**描述**：一等公民的命令行接口。

**命令草案**

```bash
form-coreset scan    <input> [--out .formcoreset]
form-coreset organize <input> --out <dir> [target options] [options]
form-coreset dedupe  <input> --out <dir> [--method sift|orb] [options]
form-coreset extract <input> --out <dir> [--encoder anime-hybrid] [options]
form-coreset review  <run-dir>
form-coreset export  <run-dir> --preset generic|kohya|diffusers
form-coreset report  <run-dir>
```

**关键选项**

```text
--name TEXT
--description TEXT
--reference PATH [PATH ...]
--auto
--adapter auto|person|object
--method sift|orb
--linkage complete|average|single
--min-score FLOAT
--min-inliers INT
--min-correlation FLOAT
--star-threshold FLOAT
--no-star-merge
--max-features INT
--no-mask
--no-verify
--no-stitch
--encoder anime-hybrid
--min-cluster-size INT
--min-samples INT
--tag-weight FLOAT
--embedding-weight FLOAT
--no-embedding
--mode reference|training
--policy strict|balanced|diverse
--max-per-form INT
--captions
--granularity FLOAT
--device cpu|cuda|auto
--seed INT
--config PATH
--dry-run
--verbose
```

**验收标准**

- `--help` 清晰；错误信息可操作。
- `--dry-run` 不写输出。
- 退出码：0 成功；1 参数/输入错误；2 运行时错误；3 部分成功。
- 日志有 stage、耗时、警告。
- 无网络时（模型已缓存）可运行。

### FR-14 插件与扩展（P1）

**描述**：允许社区接入自己的模型/算法。

**规则**

- 公开接口：`IdentityMatcher`、`Segmenter`、`FormEncoder`、`Clusterer`、`Selector`、`Normalizer`。
- Python entry points 或配置文件注册。
- 模型注册表：名称、版本、许可证、下载地址、校验和。
- 示例插件 + 文档。
- 插件失败不影响主流程（降级/报错）。

**验收标准**

- 第三方可按文档实现一个 encoder 并跑通。
- 插件错误有清晰日志。
- 插件不绕过非破坏性约束。

### FR-15 可复现与清单（P0）

**描述**：同一输入 + 配置 + seed 可复现结果。

**规则**

- manifest 记录：app 版本、git commit、模型名称/版本/许可证、阈值、配置哈希、输入哈希、时间、运行参数。
- 缓存键包含模型/预处理版本。
- 提供 `--seed`；聚类/采样使用固定种子。
- 提供 `form-coreset report` 输出摘要。

**验收标准**

- 同一环境重跑，输出结构/计数一致（允许浮点微差）。
- manifest 可被 JSON Schema 校验。
- 模型版本变化会使缓存失效。

### FR-16 集成（P1）

**描述**：与训练/整理生态衔接。

**规则**

- Kohya：输出图片 + `.txt` caption + 可选 `dataset.toml`。
- diffusers / Hugging Face datasets：导出目录或 JSONL。
- ComfyUI / FiftyOne：P2 插件。
- 不修改上游数据集；只读。

**验收标准**

- Kohya 导出可直接用于训练。
- 示例脚本可运行。

### FR-17 文档与本地化（P0）

**描述**：开源项目的基本文档。

**规则**

- 英文 README（主）+ 中文文档。
- Quickstart：安装、跑 demo、查看输出。
- 模型许可证说明。
- 隐私说明：默认不上传、无遥测。
- 贡献指南、行为准则、issue 模板。
- Benchmark 方法与数据来源。

**验收标准**

- 新用户 5 分钟内跑通示例。
- 所有第三方模型列出版本与许可证。
- 文档与 CLI 行为一致。

---

## 7. 非功能需求

| 编号 | 需求 | 目标 |
|---|---|---|
| NFR-01 | 性能 | 100 张 CPU 端到端 ≤ 5 分钟（不含首次模型下载）；1,000 张 ≤ 30 分钟；聚类阶段 ≤ 30 秒/100 张 |
| NFR-02 | 隐私 | 默认不上传；无遥测；模型下载后可离线运行 |
| NFR-03 | 可复现 | 固定 seed；manifest 记录模型/阈值/版本；同环境重跑一致 |
| NFR-04 | 跨平台 | Windows 10/11、macOS 12+、Ubuntu 20.04+；Python 3.11/3.12 |
| NFR-05 | 健壮性 | 坏文件/坏 caption/坏模型不崩溃；部分成功有报告 |
| NFR-06 | 许可证 | 代码 Apache-2.0；权重分离；`THIRD_PARTY_LICENSES.md`；不默认打包受限权重 |
| NFR-07 | 易用性 | 一条命令跑 demo；错误信息可操作；进度可见 |
| NFR-08 | 可扩展 | 六类插件接口；配置驱动；模型注册表 |
| NFR-09 | 安全 | 模型校验和；路径净化；不执行用户图片中的代码；不任意删除 |
| NFR-10 | 可观测 | 结构化日志；stage 耗时；退出码；run report |
| NFR-11 | 磁盘 | 缓存可配置/清理；输出不复制超大原图时用硬链接 |
| NFR-12 | 可访问性 | CLI 帮助完整；UI 支持键盘操作；颜色不作为唯一状态信号 |

**性能参考硬件**：4 核 CPU / 16GB RAM / 无独显；GPU 作为可选加速。实际数字在 M0 benchmark 后回填，不在 README 中承诺未经验证的耗时。

---

## 8. 数据与输出规范

### 8.1 目录结构（guided 模式）

```text
organized/
  confirmed/
    form_01/
      canonical.png
      variants/
        img_003.png
        img_027.png
      form.json
    form_02/
      canonical.png
      variants/
      form.json
  uncertain/
  rejected/
  contact_sheet.png
  report.md
  manifest.jsonl
  run.json
```

### 8.2 目录结构（auto 模式）

```text
organized/
  subjects/
    subject_01/
      subject.json
      forms/
        form_01/
          canonical.png
          variants/
    subject_02/
  unassigned/
  contact_sheet.png
  report.md
  manifest.jsonl
  run.json
```

### 8.3 `manifest.jsonl` 字段

```json
{
  "schema_version": "1.0",
  "image_id": "img_003",
  "source_path": "downloads/003.jpg",
  "sha256": "...",
  "width": 1024,
  "height": 1536,
  "status": "confirmed",
  "identity_score": 0.87,
  "identity_reasons": ["ccip:0.91", "wd14:hatsune_miku_(vocaloid):0.88"],
  "form_id": "form_01",
  "form_confidence": 0.84,
  "is_canonical": false,
  "canonical_score": 0.72,
  "score_breakdown": {"quality": 0.8, "completeness": 0.7, "cleanliness": 0.9, "representativeness": 0.6},
  "flags": ["watermark"],
  "same_source_group": "src_02",
  "timestamp": "2026-10-25T10:00:00Z"
}
```

### 8.4 `report.md` 必含内容

- 输入数量、成功/跳过/失败数量。
- 目标指定方式（name/description/reference/auto）。
- confirmed / uncertain / rejected 数量与比例。
- 形态数量；每形态成员数、canonical、是否稀有。
- 同源组数量与合并数量。
- 训练模式：每形态保留/删除、caption 冗余、覆盖率。
- 运行时间、模型版本、设备、警告。
- 建议下一步（例如“复核 uncertain 中的 13 张”）。

### 8.5 `contact_sheet.png`

- 每个形态一行（或网格），显示 canonical + 最多 N 个 variants。
- 标注形态 ID、数量、canonical 标记。
- 大图可读；默认长边 1600–2400px。
- 不包含 rejected（可另出 `rejected_sheet.png`）。

---

## 9. 边界情况与错误处理

| 情况 | 行为 |
|---|---|
| 空目录 | 报错，退出码 1，提示路径 |
| 单张图 | 不聚类，输出 1 个形态，警告 |
| 全部 rejected | 输出报告，不崩溃，提示检查目标输入 |
| 全部同一形态 | 输出 1 个形态 + 全部 variants |
| 损坏文件 | 跳过，记录 `decode_failed` |
| GIF/动画 WebP | 取首帧或跳过，记录 |
| HEIC/AVIF | 可选依赖；不支持则跳过并提示 |
| 超大图 | 编码前缩放，输出保留原图 |
| 无 alpha | 正常处理 |
| 多角色同图 | 标记 `multi_subject`，进 uncertain |
| 半身/头像 | 标记 `partial_body`，进 uncertain 或单独形态 |
| 文字/水印 | 标记 `text_watermark`，降低 canonical 分 |
| NSFW | 默认排除并标记；可配置 |
| 无人脸/无关键点 | 退化为 embedding-only |
| 相似角色 | 倾向 uncertain；不硬判 |
| caption 缺失 | 训练模式退化为形态去重 |
| caption 格式错误 | 跳过该 caption，保留图，记录 |
| 模型下载失败 | 明确错误、重试提示、支持离线模型目录 |
| 无网络 | 已缓存模型可运行；否则报错 |
| 路径过长/非法字符 | 净化文件名；必要时截断 + 哈希后缀 |
| 权限错误 | 跳过并警告 |
| 输出目录已存在 | 默认拒绝；`--overwrite` 显式覆盖 |
| 中断 | 保留已完成 stage 缓存；可续跑 |

---

## 10. 评估与质量门槛

### 10.1 Benchmark 组成

| 域 | 数量 | 内容 | 标注 |
|---|---|---|---|
| Anime 角色 | 3 组 × 100 张 | 搜索结果 + 干扰角色 + 同形态不同背景 | identity 同/不同；form 同/不同；canonical 偏好 |
| 商品/物体 | 2 组 × 100 张 | 多角度、多状态、多背景 | 视角/状态；同形态分组 |
| 通用/真人（P2） | 1 组 × 100 张 | 同类主体、不同形态 | identity/form |

数据来源：自建 + 可再分发公开数据；版权敏感的 benchmark 只提供标注/脚本，不直接分发图片。

### 10.2 指标

- 身份：confirmed precision/recall；uncertain 比例；误拒率。
- 形态：pairwise precision/recall；簇纯度；ARI/AMI；噪声比例。
- 同源：合并精确率；误合并率。
- Canonical：人工 top-1 一致率；截断/水印/低清比例。
- 集合：coverage@K、redundancy@K、稀有形态保留率。
- 训练：验证 loss、prompt adherence、多样性、人工偏好；达到相同效果所需 pair 数/步数。
- 性能：端到端耗时、峰值内存、CPU/GPU。
- 稳定性：同 seed 重跑一致性；bootstrap 簇数波动。

### 10.3 v0.1 发布门槛

- 身份 confirmed 精确率 ≥ 0.95；recall ≥ 0.90（含 uncertain）。
- 形态 pairwise recall ≥ 0.85；簇纯度 ≥ 0.85。
- 同源合并精确率 ≥ 0.95；误合并率 ≤ 5%。
- 100 张 CPU 端到端 ≤ 5 分钟。
- 所有 P0 验收标准通过。
- 许可证审计完成；无受限权重被默认打包。
- 无 P0 级崩溃/数据丢失问题。

> 门槛未达标时，先修算法或缩小承诺范围，不通过调整报告口径“达标”。

---

## 11. 里程碑

| 阶段 | 周期（1 人估算） | 内容 | 退出标准 |
|---|---|---|---|
| **M0 技术验证** | 1–2 周 | 5 个 spike + benchmark harness | 选定默认模型/阈值；失败模式清楚 |
| **M1 CLI MVP** | 2–3 周 | 扫描/QC、guided 身份（参考图/名字）、分割、配准、形态聚类、canonical、输出 | 100 张搜索图可整理；report 可用 |
| **M2 自动模式 + UI** | 2 周 | auto 身份聚类、Gradio 审核、导出 | `--auto` 可用；复核 ≤ 10 分钟 |
| **M3 训练模式 + 优化** | 2 周 | caption、每形态上限、facility location、report、Kohya 导出 | 训练集整理可用；报告准确 |
| **M4 扩展** | 持续 | object/真人适配器、归一化、插件、ComfyUI | P1/P2 需求 |

周期为估算，取决于样例数据、模型下载/许可证和 benchmark 标注成本。

### M0 五个 spike（必须先做）

1. Anime pose：RTMPose+HumanArt / DWPose / SDPose / embedding-only。
2. Identity：CCIP / DINOv2 / WD14 tags / 融合。
3. Background invariance：masked vs unmasked；配准开关。
4. Auto mode stability：bootstrap 簇数/纯度；用户选簇时间。
5. Training benchmark：全量 vs 筛选后 LoRA。

---

## 12. 开源策略

### 12.1 仓库与许可

- 仓库名：`form-coreset`（待定名）。
- Python 包：`form-coreset`；import：`formcoreset`。
- 代码许可证：**Apache-2.0**（推荐；专利授权、商业友好）。
- 模型：代码/权重分离；默认不打包受限权重；`THIRD_PARTY_LICENSES.md` 列出每个模型。
- 贡献者协议：DCO 或 CLA（建议 DCO，降低贡献门槛）。

### 12.2 仓库结构

```text
form-coreset/
  pyproject.toml
  README.md
  CONTRIBUTING.md
  CODE_OF_CONDUCT.md
  LICENSE
  THIRD_PARTY_LICENSES.md
  docs/
  src/formcoreset/
    cli.py
    config.py
    pipeline.py
    ingest.py
    quality.py
    identity/
    segmentation.py
    registration.py
    form/
    cluster.py
    select.py
    organize.py
    manifest.py
    ui/
  tests/
  benchmarks/
  examples/
```

### 12.3 社区与发布

- CI：lint、类型检查、单测、CPU smoke test；GPU 测试可选。
- 发布：语义化版本；模型版本固定；CHANGELOG。
- Issue 模板：bug / feature / model-license。
- good-first-issue：新 encoder、新 selector、文档、集成。
- Benchmark 榜单：社区可提交结果；脚本可复现。
- 集成：Kohya、diffusers、ComfyUI、FiftyOne、waifuc。
- 隐私：默认无遥测；如需统计，opt-in + 匿名。
- 文档：英文为主，中文补充；Quickstart 5 分钟。

### 12.4 模型注册表

每个模型记录：

```yaml
name: ccip
version: "1.0"
task: identity
license: "TODO: verify"
source: "https://..."
checksum: "..."
size_mb: 120
commercial_use: true|false|unknown
```

启动时检查许可证；`--accept-model-license` 或交互确认。

---

## 13. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 动漫姿态模型失准 | 形态聚类退化 | M0 spike；embedding-only 降级；WD14 标签辅助；人工审核 |
| CCIP 混淆相似角色 | 身份过滤错误 | 提高 confirmed 阈值；uncertain 桶；参考图增多；人工校正 |
| 背景主导 embedding | 同形态不同背景被拆开 | 强制 mask；同源配准；分割失败回退 |
| HDBSCAN 不稳定 | 簇数跳变 | 多粒度 consensus；粒度滑杆；UI 合并/拆分 |
| CLIP zero-shot 视角弱 | 物体形态标签不可靠 | 不依赖标签；DINOv2 聚类 + 人工命名/小分类器 |
| 过度合并 | 丢信息 | 默认非破坏；variants 保留；不确定进 uncertain；报告提示 |
| CPU 性能不达标 | 体验差 | ONNX、小模型、缓存、批处理；GPU extra |
| 模型许可证受限 | 无法打包/商用 | 权重分离；默认宽松模型；文档提示 |
| 训练集去重过度 | 泛化变差 | `diverse` 策略；caption 维度；下游 benchmark；保守默认 |
| 搜索 benchmark 版权 | 无法分发 | 只分发脚本/标注；使用可再分发样本 |
| 专利风险 | 商业化受阻 | 身份+姿势聚类/canonical 有专利申请；商业化前 FTO |
| “人人可用”导致定位模糊 | 社区不聚焦 | 先打 AI 画师/LoRA beachhead；通用能力后置 |

---

## 14. 决策记录

### 14.1 已确认

| 编号 | 决策 | 结论 | 影响 |
|---|---|---|---|
| D1 | 项目名 | ✅ **FormCoreset** | 仓库 `form-coreset`，Python 包 `form-coreset`，import `formcoreset` |
| D2 | 第一 beachhead | ✅ **AI 绘画 / LoRA 数据集** | README、示例、benchmark 先围绕角色参考集与训练集；商品图作为第二验证域 |
| D3 | 代码许可证 | ✅ **Apache-2.0** | 专利授权、商业友好；模型权重分离并逐一审计 |
| D4 | v0.1 适配器 | ✅ `person` + `auto` 为 P0；`object` 为 P1 实验性 | object 走 `--adapter object`，文档标注 experimental |

### 14.2 待确认

| 编号 | 决策 | 推荐默认 |
|---|---|---|
| D5 | 文本能力 | 名字 + WD14 为 P0；自由描述为 P1 |
| D6 | 自动模式输出 | 输出所有主体簇 + `unassigned/`，由用户选目标 |
| D7 | 簇命名 | 可选；WD14/CLIP 失败则用 `subject_XX/form_XX` |
| D8 | 变体保留 | 默认保留 `variants/`；只高亮 canonical |
| D9 | 输出归一化 | v0.2；v0.1 只用分割做分析 |
| D10 | 训练模式 | v0.1 基础（每形态上限 + 简单 caption 相似度）；完整 caption-aware 放 P1 |
| D11 | 真人身份 | v0.2；v0.1 可用通用 embedding 做实验 |
| D12 | 多角色同图 | v0.1 标 `uncertain/`；拆分放 v0.2 |

---

## 15. 附录

### 15.1 CLI 示例

```bash
# 文本：名字
form-coreset organize ./downloads --name "Hatsune Miku" --out ./organized

# 文本：介绍（P1）
form-coreset organize ./downloads --description "blue twin-tail girl" --out ./organized

# 参考图
form-coreset organize ./downloads --reference ref1.png ref2.png --out ./organized

# 文本 + 参考图
form-coreset organize ./downloads \
  --name "Hatsune Miku" --reference ref1.png --out ./organized

# 完全自动
form-coreset organize ./downloads --auto --out ./organized

# 训练集整理
form-coreset organize ./lora-data --mode training --policy balanced \
  --max-per-form 3 --captions --out ./curated

# 复核与导出
form-coreset review ./organized
form-coreset export ./organized --preset kohya
```

### 15.2 配置示例

```yaml
target:
  name: "Hatsune Miku"
  description: null
  references: []
  mode: guided

adapter: auto

identity:
  image_matcher: ccip
  text_matcher: wd14
  alpha: 0.75
  confirmed_threshold: 0.75
  rejected_threshold: 0.45

form:
  clusterer: hdbscan
  min_cluster_size: 3
  min_samples: 1
  granularity: 0.5
  use_keypoints: auto

selection:
  mode: reference
  preset: most_standard
  max_per_form: 3
  policy: balanced

normalize:
  enabled: false

performance:
  device: auto
  batch_size: 8
  seed: 42
```

### 15.3 术语表

| 术语 | 含义 |
|---|---|
| Coreset | 覆盖不同信息、冗余最小的子集 |
| Canonical | 一个形态簇的代表图 |
| Form | 姿势/角度/状态/形状/版式等视觉信息维度 |
| Variant | 同形态的重复版本 |
| Same-source | 同一原图派生的变体 |
| Guided mode | 用户提供文本/参考图 |
| Auto mode | 不给目标，自动聚类 |
| Uncertain bucket | 需人工复核的中间态 |

### 15.4 参考

- [市场调研](market-research.md)
- [开源定位](positioning.md)
- [技术架构](architecture.md)
- [Apple Photos 合并重复](https://support.apple.com/lt-lt/guide/iphone/iph1978d9c23/17.0)
- [fastdup](https://github.com/visual-layer/fastdup)
- [FiftyOne](https://docs.voxel51.com/)
- [imgutils](https://github.com/deepghs/imgutils)
- [waifuc](https://github.com/deepghs/waifuc)
- [DeepCore（coreset）](https://github.com/Daidj/DeepCore)
- [LightGlue（Apache-2.0）](https://github.com/cvg/LightGlue)
- [DINOv2 许可证](https://playground.roboflow.com/models/meta/dinov2)

---

## 16. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v0.1 Draft | 2026-10-25 | 初版：整合调研、定位、场景、架构与需求 |
