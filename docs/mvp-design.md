# MVP 设计：动漫/游戏角色 · 姿势标准图归一器

> 基于 2026-10-25 市场调研与范围确认
> 范围：动漫/游戏角色；角色已知；姿势未知；提取式归一化（不生成、不重绘）

## 1. 产品定义

**输入**

一个角色目录（或一个角色标签）下的多张图片。它们可能是同一姿势的不同版本，也可能是不同姿势；差异可能来自背景、比例、裁剪、表情、服装、画风、压缩格式。

**输出**

- 按姿势聚类的分组（每组 = 一个“角色 × 姿势”）；
- 每组一张 **canonical 标准图**；
- 可选的归一化版本：去背、统一画布、sRGB、标准命名；
- `manifest.jsonl`：来源、分数、模型版本、阈值、处理参数；
- `quarantine/`：疑似重复/低质/异常图，默认不删除原图。

**非目标（MVP）**

- 不自动识别角色（角色已知；V2 用 CCIP 扩展多角色目录）。
- 不做生成式重绘、补全、扩图、统一光照。
- 不做通用照片清理或大规模训练数据去重。
- 不处理 sprite sheet / 多角色同图（V2 可加人物拆分）。

## 2. 核心流程

```
扫描与预处理
  → 内容哈希、解码、EXIF/ICC、alpha、方向
  → 质量门控：损坏、模糊、过曝、分辨率、单色、NSFW
同源变体配准（高精度优先）
  → 主体 mask → SIFT/ORB/SuperPoint + RANSAC → 单应性/重叠率
  → 同源组：选主体像素最多、裁剪最少、压缩损伤最小的一张
姿势聚类（MVP 核心）
  → 方法 A：关键点 → 归一化骨架 → 角度/骨向量 → HDBSCAN
  → 方法 B：DINOv2 角色 crop embedding → HDBSCAN
  → 方法 C：A + B 混合（推荐先评估）
  → 可选：WD14 姿势标签作为辅助特征/簇命名
Canonical 选择
  → 质量分 + 完整度 + 干净度 + 代表性 + 保真度
  → 预设：最标准 / 最清晰 / 最完整 / 透明背景
归一化
  → anime-seg / isnet-anime 去背
  → alpha-aware bbox + padding + 固定画布 + 锚点
  → sRGB / 8-bit / PNG（可选 WebP）
  → 剥离 EXIF + 写入 manifest
人工审核
  → 簇卡片、并排对比、同源 diff 叠加、改选、合并/拆分、隔离
导出
  → canonical/ + normalized/ + quarantine/ + manifest.jsonl
  → 原图保持不动，所有操作可回滚
```

## 3. 姿势聚类：三种方案，先做技术验证

这是整个 MVP 最大的技术不确定性。动漫角色的人体比例、夸张透视、头发/服装遮挡会让真人关键点模型失准，因此不能直接假设 DWPose/MediaPipe 一定可用。

### 方案 A：关键点 + 归一化骨架

- 模型候选：DWPose、RTMPose、MediaPipe Pose（33 landmarks）。
- 归一化：用肩/髋/躯干做平移、缩放、旋转（Procrustes），再拼接关节角度、骨方向向量。
- 镜像处理：与水平翻转后的骨架比较，取较小距离。
- 缺失关键点：按可见性加权，低置信度关节不参与距离。
- 聚类：L2 归一化后 HDBSCAN；噪声点标为 `-1`，进入人工审核。
- 优点：可解释、真正“姿势感知”、跨角色可用。
- 风险：动漫风格、chibi、极端视角、遮挡下关键点失败。

### 方案 B：DINOv2 角色 crop embedding

- 先用 anime-seg 去背，再用主体 bbox 裁切，送入 DINOv2（ViT-B/14 或 ViT-L/14）。
- 聚类：L2 归一化 + HDBSCAN / 余弦阈值。
- 优点：无需关键点，对画风、比例、背景变化稳健。
- 风险：embedding 会混入服装、表情、视角、光照；容易把“同姿势不同服装”拆开，或把“不同姿势但构图相似”合并。

### 方案 C：混合（推荐默认）

1. 先跑同源配准，精确合并裁剪/缩放/换背景变体；
2. 对剩余图片，优先用高置信度关键点；
3. 关键点不足时回退到 DINOv2；
4. 用两种距离的加权组合做聚类，权重可在配置中调整；
5. 在审核 UI 中显示“聚类依据”，允许人工合并/拆分。

**M0 技术验证**：用真实样例比较 A/B/C 的簇纯度、召回率和人工一致率，再决定默认方案。不要在没有数据的情况下先写死模型。

### 可选的辅助信号

- **WD14 标签**（imgutils）：`standing`、`sitting`、`running`、`arm up` 等标签可作为弱先验，也可用于自动命名簇。
- **CLIP 相似度**：作为粗筛/候选对生成，不作为唯一依据。
- **CCIP**：MVP 角色已知，可不用；V2 多角色目录时用于身份校验和分组。

## 4. Canonical 选择

不要只用一个“美学分”。建议多目标加权，并在 UI 中展示分数拆解：

```text
score =
  w_quality      * 技术质量（分辨率、清晰度、压缩、噪点）
+ w_completeness * 完整度（主体 bbox、是否截断、全身/半身）
+ w_cleanliness  * 干净度（背景复杂度、文字/水印、其他人物）
+ w_represent    * 代表性（到簇中心/medoid 的距离）
+ w_fidelity     * 保真度（颜色、线条、无过度放大）
```

预设：

- **最标准**：代表性权重最高，接近组中心；
- **最清晰**：清晰度 + 有效分辨率；
- **最完整**：主体未被裁切、全身优先；
- **透明背景**：干净度 + alpha 质量优先。

注意：动漫图的“美学模型”多数在真人照片上训练，迁移效果有限；MVP 以清晰度、完整度、背景干净度和代表性为主，美学分只作为弱信号。

## 5. 归一化

| 步骤 | MVP 方案 | 备注 |
|---|---|---|
| 去背 | `isnet-anime`（rembg）或 imgutils `segment_rgba_with_isnetis` / anime-seg | 保留发丝 alpha；输出 RGBA |
| 裁切 | alpha bbox + 可配置 padding（默认 8%） | 不裁掉主体 |
| 画布 | 默认 1024×1024 或 3:4；可配置 | 角色图建议底部锚点 |
| 背景 | 默认透明；可选纯白/品牌色 | 与下游引擎/素材库匹配 |
| 色彩 | 转 sRGB；8-bit | 保留原始文件 |
| 格式 | PNG（无损，含 alpha）；可选 WebP | 命名 `{character}_{pose}_{canonical}.png` |
| 元数据 | 剥离 EXIF；manifest 记录来源/模型/参数 | 可追溯、可回滚 |

## 6. 审核 UI（MVP 用 Gradio，生产可换 FastAPI + React）

必需视图：

- **角色总览**：姿势簇卡片、数量、置信度、缩略图墙；
- **簇详情**：成员网格、分数拆解、canonical 高亮；
- **对比模式**：并排大图；同源组显示 diff 叠加/对齐结果；
- **操作**：设为 canonical、合并/拆分簇、标记“不同服装/表情/忽略”、调整阈值并重算；
- **导出**：canonical、normalized、manifest、quarantine。

关键原则：**默认不删除任何原图**；所有变更写入 manifest，支持撤销。

## 7. 代码结构建议

```text
data_normalization/
  pyproject.toml
  README.md
  configs/
    default.yaml
  src/datanorm/
    cli.py                 # typer CLI
    config.py              # pydantic 配置
    ingest.py              # 扫描、哈希、解码、EXIF
    quality.py             # 模糊/曝光/分辨率/单色/NSFW
    registration.py        # 同源配准
    pose/
      base.py              # PoseEncoder 接口
      keypoints.py         # DWPose/RTMPose/MediaPipe
      embedding.py         # DINOv2
      hybrid.py            # 加权融合
    cluster.py             # HDBSCAN/FAISS
    select.py              # canonical 评分
    normalize.py           # 去背、裁切、画布、色彩
    manifest.py            # JSONL/SQLite
    ui/app.py              # Gradio
  tests/
  data/                    # gitignore
  output/                  # gitignore
```

CLI 草案：

```bash
datanorm scan    <character_dir>          # 扫描 + 质量 + embedding 缓存
datanorm cluster <character_dir> --method hybrid
datanorm select  <character_dir> --preset most_standard
datanorm normalize <character_dir>
datanorm review  <character_dir>          # 启动本地 UI
datanorm export  <character_dir>
```

## 8. 里程碑

| 阶段 | 内容 | 预估 |
|---|---|---|
| **M0 技术验证** | 50–100 张真实样例；比较 A/B/C；确定特征、阈值、失败模式 | 1–2 天 |
| **M1 CLI MVP** | 扫描 → 同源配准 → 姿势聚类 → 评分 → 选 canonical → manifest | 3–5 天 |
| **M2 归一化** | 去背、alpha 裁切、统一画布、sRGB、命名 | 2–3 天 |
| **M3 审核 UI** | Gradio：簇浏览、改选、合并/拆分、导出 | 3–5 天 |
| **M4 评估调优** | 标注集、指标、阈值预设、难例分析 | 2–3 天 |
| **V2** | CCIP 多角色发现、WD14 簇命名、人物拆分、API/插件 | 后续 |

## 9. 评估方法

- **样例**：1–2 个角色，共 50–100 张；覆盖全身/半身、不同背景、不同比例、镜像、相似姿势（站立 vs 轻微侧身）、不同服装/表情、chibi。
- **标注**：人工标出每张图的姿势组（可用“站立/坐下/跑步/挥手/持剑”等名称）。
- **指标**：
  - pairwise precision / recall；
  - 簇纯度、ARI/NMI；
  - canonical top-1 与人工选择一致率；
  - 每 100 张处理时间；
  - 人工改选率（越低越好）。
- **基线**：仅 DINOv2、仅关键点、混合、仅同源配准。

## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| 动漫关键点模型失准 | M0 先验证；失败则回退 DINOv2 / 混合；必要时用动漫数据微调 |
| embedding 混入服装/表情 | 去背 + 主体裁切；审核 UI 允许“同姿势不同服装”合并；权重可调 |
| 过度合并 | 默认 quarantine，不删除；显示聚类依据；人工一键拆分 |
| “同姿势”定义主观 | 提供严格/宽松阈值；按角色保存配置；记录人工决策 |
| 美学模型不适配动漫 | 以清晰度/完整度/干净度为主；美学分仅作弱信号 |
| 版权/隐私 | 本地处理，不上传；manifest 保留来源；不做生成式重绘 |
| 专利风险 | 身份+姿势聚类、canonical 选择有专利申请；商业化前做 FTO |

## 11. 需要你确认 / 提供

1. **样例目录**：能否提供一个角色文件夹（或几个角色）用于 M0？路径即可。
2. **“同姿势”的定义**：是“轮廓几乎一致”，还是“语义动作相同”？镜像姿势算同一组吗？不同服装/表情算同一组吗？
3. **图片构成**：全身/半身/头像/chibi 是否混合？是否有多角色同图或 sprite sheet？
4. **输出偏好**：默认透明背景还是纯白？画布 1:1 还是 3:4？是否需要保留原始分辨率版本？
5. **规模**：单角色几十张、几百张，还是上千张？

拿到样例后，我建议先跑 M0：用同一批图对比“关键点 / DINOv2 / 混合”三种聚类，输出可视化对比和阈值建议，再决定 M1 的默认算法。这样比先写完整应用更稳。

---

## 12. 方向更新：Discovery-first（用户确认）

用户进一步明确：**希望工具自己发现姿势/形态，而不是先定义模板**。目标不是“每个已知姿势选一张”，而是：

> 从一批图片中，去除重复变体，保留“携带不同信息（姿势）”的高价值图片，输出一个最小但覆盖完整的集合。

更准确的叫法是 **pose-aware coreset selection（姿势感知的核心集选择）**：去重 + 覆盖 + 价值排序是集合级问题，不是逐张打分。

### 12.1 集合级目标

```text
maximize  Σ_{c ∈ selected clusters} value(canonical_c)
        + λ * coverage(selected clusters)
        - μ * redundancy(selected set)
subject to  |selected| ≤ K（可选）
```

- `value`：单图质量 × 完整度 × 干净度 × 代表性；
- `coverage`：选中的集合覆盖了多少个不同的姿势簇；
- `redundancy`：选中图之间的相似度总和；
- `K`：可选的“信息预算”；不设 K 时，由稳定簇数量决定。

**关键认知：一张图的价值是相对于集合的。** 再清晰的一张图，如果和另一张姿势重复，它的边际价值也很低。所以不能先给每张图打分再取 top-N，必须先聚类/去重，再在集合层面选择。

### 12.2 推荐算法

1. **主体分组**：如果一批图包含多个角色/人/物，先做身份聚类（CCIP/ArcFace/ReID）；单主体则跳过。
2. **同源变体合并**：局部特征配准，合并裁剪/缩放/换背景变体，保留主体像素最多的一张。
3. **特征提取**：
   - 人物：关键点（DWPose/RTMPose/MediaPipe）→ Procrustes 归一化 → 角度/骨向量；
   - 通用：去背后主体 crop → DINOv2 embedding；
   - 物体：DINOv2/CLIP + 视角/状态分类；
   - 可选：WD14 标签作为弱先验和簇命名。
4. **开放集聚类**：HDBSCAN / 层次聚类 / kNN 图上的 Leiden；多粒度运行，选稳定性最高的切分，或由用户滑杆控制。
5. **簇内 canonical**：medoid + 质量 + 完整度 + 背景干净度。
6. **簇间去冗余与集合选择**：合并相似簇；用 facility location / MMR / top-K 选最终集合；输出每张图的“入选理由”。
7. **人工审核**：簇卡片、合并/拆分、调粒度、改选 canonical；确认后导出。
8. **导出**：`selected/` + `clusters.json` + `contact_sheet.png` + manifest。

### 12.3 必须暴露给用户的控制

- **粒度滑杆**：更激进地合并 vs 更细地拆分（对应 HDBSCAN 的 `min_cluster_size` / 距离阈值）。
- **信息预算 K**：只输出价值最高的 K 张，或输出所有稳定簇。
- **信息维度**：只按姿势，还是姿势+视角+表情+服装。不同选择对应不同特征和权重。
- **价值预设**：最标准 / 最清晰 / 最完整 / 最适合做参考图。

### 12.4 现有工具的位置

| 工具 | 覆盖“去重” | 覆盖“不同姿势” | 覆盖“集合级价值选择” | 缺口 |
|---|---|---|---|---|
| imgraft | ✓ | ~（CLIP 语义，非姿势） | ✓（diverse subset） | 无姿势特征、无 canonical 归一化 |
| SemDeDup | ✓ | ~（语义簇） | ✗（每簇留一个，不看质量） | 无价值排序、无姿势轴 |
| fastdup | ✓（近重复） | ✗ | ✗ | 不做跨簇覆盖选择 |
| PhotoSort / PixTwin | ~（场景组） | ✗ | ~（组内选最佳） | 不发现姿势 |
| **目标工具** | **✓** | **✓（姿势感知）** | **✓（coreset）** | **端到端空白** |

### 12.5 技术风险

- **“不同姿势”没有客观标准**：同一组图，不同人可能给出不同簇数；必须用粒度滑杆 + 人工复核。
- **特征混淆**：embedding 可能按服装/表情/视角分组，而不是姿势；需要用户定义“信息维度”，并用关键点约束。
- **动漫关键点不可靠**：chibi、夸张透视、遮挡会让 DWPose/MediaPipe 失败；先用真实样例验证，再决定是否微调。
- **簇不稳定**：HDBSCAN 对参数敏感；用 bootstrap/consensus 选稳定切分，并显示置信度。
- **多主体**：如果一批图有多个角色/人/物，身份聚类误差会传导到姿势聚类；MVP 建议先做单主体。

### 12.6 验证计划（更新）

1. 选一个主体，30–100 张图；人工标出“不同姿势”组。
2. 对比三种特征：关键点 / DINOv2 / 混合。
3. 指标：pairwise P/R、簇纯度、coverage@K、redundancy@K、人工一致率。
4. 输出：簇卡片 + 每簇 canonical + contact sheet；让用户调粒度，确认是否符合直觉。
5. 通过后再做多主体、自动命名、归一化 UI。
