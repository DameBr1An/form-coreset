# FormCoreset · 开源项目定位

> 工作名 / 仓库名：FormCoreset / `form-coreset`
> 主 tagline：**Keep the information. Drop the duplicates.**
> 训练场景 tagline：**Train on what's different.**
> 副标题：One high-value image per distinct form.
> 状态：定位已确认（名称 / beachhead / license / v0.1 适配器范围）；待社区验证

## 1. 一句话定位

**FormCoreset 是一个开源工具：把同一主体的一堆重复图文对，自动压缩成一组“每个不同形态只保留少量高价值样本”的最小集合——用于构建参考图库，或用于模型训练，避免同一形态被反复训练导致过拟合。**

英文：

> An open-source tool that turns a folder of near-duplicate images (and captions) into a minimal, information-complete set — for reference sheets, or for training without over-representing the same form.

**典型场景**：从搜索引擎下载某角色的前 100 张图 → 过滤掉不是该角色的图 → 合并“同形态、不同背景/裁剪”的变体 → 按形态整理成可复核的集合。

## 2. 定位声明（positioning statement）

**For** AI artists and model trainers **who** fine-tune diffusion/LoRA models on image-text datasets, **FormCoreset is** an open-source **form-aware dataset curator** **that** removes redundant image-text pairs and keeps a minimal, information-complete subset — reducing redundancy-driven overfitting without sacrificing coverage.

**Unlike** generic dataset dedup tools that only compare image embeddings, **FormCoreset models both form** (pose, angle, state, shape) **and caption**, and optimizes the training set as a whole.

中文：

> 面向 AI 画师和模型训练者：FormCoreset 是一个开源的“形态感知数据集整理器”。它识别同一形态的冗余图文对，保留一个信息覆盖完整、样本量最小的子集——在不牺牲覆盖面的前提下，减少因同一形态反复出现而导致的过拟合。与只比较图像 embedding 的通用去重工具不同，FormCoreset 同时建模“形态”和“caption”，并在集合层面优化训练集。

## 2.5 核心场景：从 100 张搜索结果到形态整理

用户从搜索引擎下载同一角色的前 100 张图。里面混着：

- 不是这个角色的图（其他角色、同作品相似角色、无关图）；
- 是这个角色，但形态相同、背景/裁剪/比例不同的变体；
- 是这个角色，但形态不同（不同姿势/动作/视角）；
- 多角色同图、半身/头像、带文字/水印、低质量图。

工具要做三件事：

1. **身份过滤**：判断每张图是不是目标角色；分为 `confirmed` / `uncertain` / `rejected`。
2. **形态去重**：在 confirmed 里，把“同形态、不同背景/裁剪/比例”的图合并；保留每个形态的一组变体 + 一张 canonical。
3. **整理输出**：按形态分文件夹，生成 contact sheet + report + manifest。

### 身份过滤的三种输入方式

| 方式 | 用户提供 | 适用 | 备注 |
|---|---|---|---|
| 参考图 | 1–5 张目标角色图 | 最稳、跨域通用 | anime 用 CCIP；真人用 ArcFace/ReID；物品用 DINOv2/CLIP |
| 角色名 | 角色名/标签 | anime（WD14 角色标签） | 依赖标签库；可能有别名/版本差异 |
| 自动聚类 | 不提供 | 搜索结果以目标角色为主 | 先按身份聚类，用户选目标簇；其余为 rejected/uncertain |

默认建议：**参考图优先**。如果用户没有参考图，可以从搜索结果里选 1–5 张，或让工具聚类后选簇。

### 形态去重的关键

- 先做**主体分割/去背**，再做形态相似度；否则背景会主导 embedding，导致“同形态不同背景”被拆开。
- 先做**同源配准**（SIFT/SuperPoint + RANSAC）：同一张图的不同裁剪/缩放/换背景，直接合并。
- 再做**形态聚类**：人物用关键点 + 主体 embedding；物品用主体 embedding + 视角/状态。
- 多角色同图、半身/头像、文字/水印、低质量图：不要强行归类，进 `uncertain/` 或 `rejected/`，由用户复核。

### 输出布局

```text
organized/
  confirmed/
    form_01/
      canonical.png
      variants/
        img_003.png
        img_027.png
    form_02/
      canonical.png
      variants/...
  uncertain/
  rejected/
  contact_sheet.png
  report.md
  manifest.jsonl
```

`report.md` 示例：

```text
Input: 100 images
Confirmed: 72
Uncertain: 13
Rejected: 15
Distinct forms: 11
Most over-represented form: form_03 (18 images, kept 1 canonical + 17 variants)
Rare forms: form_09, form_11 (2 images each)
```

### 为什么这是最好的开源 demo

- 场景直观：一次搜索 → 100 张下载 → 一条命令 → 干净的角色形态集。
- 痛点明确：手动挑图 1–2 小时；工具几分钟完成，且可复核。
- 结果可展示：contact sheet 一眼看出“整理前 vs 整理后”。
- 与训练场景自然衔接：整理后的 `confirmed/` 就是 LoRA/参考集的候选。

## 2.6 输入方式：文本 / 参考图 / 两者 / 不给

工具应支持四种目标指定方式，优先级和置信度不同：

| 输入 | 例子 | 作用 | 置信度 | 适用 |
|---|---|---|---|---|
| 文本：名字 | `--name "Hatsune Miku"` | 角色身份先验；WD14 角色标签匹配 / CLIP 文本相似度 | 中（依赖标签库/别名） | anime 快捷方式 |
| 文本：介绍 | `--description "blue twin-tail girl with a tie"` | 属性/外观匹配；CLIP 文本-图像相似度 | 低–中 | 无参考图、名字不确定时 |
| 参考图 | `--reference ref1.png ref2.png` | 身份匹配；CCIP / ArcFace / DINOv2 | 高（1 张起，3–5 张更稳） | 跨域最稳 |
| 文本 + 参考图 | 两者都给 | 融合分数；参考图主导身份，文本辅助消歧 | 最高 | 推荐默认 |
| 不给 | `--auto` | 全自动身份聚类 + 形态聚类；用户选目标簇 | 无目标先验 | 探索/批量整理 |

### 融合策略

```text
identity_score = α * image_similarity + (1 - α) * text_similarity
```

- 有参考图时，`α` 默认 0.7–0.8，图像主导；
- 只有文本时，`α = 0`，并放宽阈值，更多进 `uncertain/`；
- 文本 + 参考图冲突时，以参考图为准，文本用于消歧和簇命名；
- 所有分数做校准/归一化，避免不同模型分数尺度不一致。

### 自动模式（不给信息）

- 先按身份聚类（anime: CCIP；真人: ArcFace/ReID；通用: DINOv2/CLIP）。
- 再在每个身份簇内做形态聚类。
- 输出所有身份簇，按大小/质量排序；用户选择目标簇，或直接浏览全部。
- 不假设“最大簇就是目标”；如果给了文本，可用文本相似度给簇排序/命名。
- 噪声/无法归类 → `unassigned/`。

自动模式输出：

```text
organized/
  subjects/
    subject_01/
      form_01/...
      form_02/...
    subject_02/...
  unassigned/
  contact_sheet.png
  report.md
```

### 文本匹配的现实限制

- **anime 名字**：依赖 WD14 角色标签库和别名映射；不同作品/皮肤/版本可能匹配不到。
- **自由描述**：CLIP 对动漫文本-图像匹配能力有限；描述适合作为弱先验，不适合硬过滤。
- **真人名字**：没有参考图时不可靠；真人优先参考图（人脸/人体 ReID）。
- **物品名/描述**：CLIP zero-shot 可用，但细粒度 SKU 仍建议参考图。

## 3. 什么算“形态”？

| 主体 | 形态维度 | 例子 | 通常应被去重的变体 |
|---|---|---|---|
| 角色 / 人物 | 姿势、动作、表情、服装、视角 | 站立 / 坐姿 / 挥手；正面 / 侧面；常服 / 战斗服 | 裁剪、缩放、背景替换、压缩格式 |
| 刚体物品 | 视角、朝向、开合/组装状态 | 正面 / 背面 / 侧面 / 顶视；打开 / 合上；组装 / 拆解 | 背景、阴影、白底/透明底、分辨率 |
| 可变形物品 | 形状、折叠/展开、充气/放气状态 | 折叠 / 展开；空 / 满；平整 / 起皱 | 拍摄距离、裁剪、背景 |
| 海报 / 主视觉 | 版式、构图、文字区块、横竖版 | 横版 / 竖版；标题在上/在下；不同裁剪 | 分辨率、格式、轻微色差 |
| 通用素材 | 由 embedding 自动发现的视觉状态 | 由数据决定 | 背景、比例、裁剪、压缩 |

**关键设计**：形态不是模型预设的，而是由“用户选择的信息维度 + 数据中发现的簇”共同决定。工具要提供粒度控制，允许用户把“同姿势不同服装”合并或拆开。

## 4. 两种输出模式

### Reference mode（参考图库）

- 每个形态选一张 canonical。
- 价值 = 质量 + 完整度 + 代表性。
- 用于角色参考表、商品目录、pose sheet。

### Training mode（训练集整理）

- 每个形态保留 1–K 个样本；K 可配置。
- 价值 = **边际信息增益** + caption 对齐 + 质量；稀有形态权重更高。
- 输出过滤后的数据集 + 冗余报告 + 建议。
- 用于 LoRA / diffusion / 分类模型微调，减少同一形态重复曝光导致的过拟合。

**关键区别**：Reference mode 要“最标准的一张”；Training mode 要“最有信息量的一组”。同一张图在两个模式下的价值可能不同。

## 5. 训练模式：图文对冗余策略

### 5.1 冗余单元是“图文对”，不只是图片

两个 pair 冗余的条件：

- 图像形态相似；
- caption 语义相似（或 caption 几乎相同）。

如果形态相同但 caption 教的是不同属性，可能仍然有价值；如果 caption 相同但形态不同，也可能有价值。是否保留取决于策略。

### 5.2 三种策略

| 策略 | 规则 | 适用 |
|---|---|---|
| **Strict** | 每个形态只保留 1 个 pair | 强去重、快速验证、避免过拟合 |
| **Balanced（默认）** | 每个形态最多 K 个；优先保留 caption 多样、质量高的 pair | 大多数 LoRA / 扩散微调 |
| **Diverse** | 只去近重复；保留背景/光照/风格的合理变化 | 需要泛化、数据量本来就少 |

### 5.3 选择目标

```text
maximize  形态覆盖 + caption 概念覆盖 + 样本质量
- λ * 形态内冗余 - μ * caption 冗余
subject to 每个形态 ≤ K 个 pair
```

- **稀有形态**：保留（不受全局 top-K 挤压）；
- **主导形态**：封顶（避免过拟合）；
- **近重复**：优先删除；
- **caption 冗余**：可单独开关。

### 5.4 输出

- `curated/`：过滤后的图片 + caption；
- `report.md` / `report.json`：每个形态保留/删除数量、caption 冗余、覆盖率；
- `contact_sheet.png`：人工快速检查；
- 可选：Kohya `dataset.toml` / diffusers dataset 配置；
- 可选：对稀有形态给出“建议补充/重复次数”的提示（不生成图片）。

## 6. 目标用户与场景

### 第一优先（beachhead）：AI 绘画 / LoRA 训练与数据集整理

- **痛点**：一个角色几百张图，大量同姿势变体；手动挑选耗时；训练集重复导致过拟合或浪费算力。
- **价值**：几分钟得到“不同形态代表图”集合；可直接作为参考集或训练集骨架。
- **社区**：GitHub、Civitai、ComfyUI、Kohya、Hugging Face；开源友好，传播快。
- **杀手级 demo**：同一套数据，全量训练 vs 用 FormCoreset 筛选后训练；后者用更少样本/更少步数达到相同或更好效果。

### 第二优先：商品 / 电商素材

- **痛点**：同一 SKU 多角度、多状态、多版本；需要每个角度/状态一张标准图，或训练商品模型时避免重复视角。
- **价值**：多图归并 + canonical + manifest；开源替代闭源 API 的一部分。
- **技术优势**：刚体物品通常比人体容易；固定视角是封闭集合，适合验证。

### 第三优先：游戏 / 动漫 / TRPG 素材库

- **痛点**：立绘差分、表情差分、宣传图版本多；需要角色参考表 / 姿势表。
- **价值**：自动生成 pose sheet / reference sheet 的候选集。

### 第四优先：摄影选片

- **痛点**：连拍/同场景照片太多；想要每个姿势/构图一张最佳。
- **价值**：比消费级“相似照片清理”更懂形态与构图；可本地批量。

### 非目标用户（v0.1）

- 需要云端协作/权限管理的企业 DAM 用户。
- 需要生成式修图/扩图的用户。
- 需要 web-scale 训练数据去重的团队（用 fastdup/FiftyOne/NeMo）。

## 7. 核心价值主张

1. **Set-level，不是 file-level**：不是“这张图好不好”，而是“这组图里，哪几张放在一起最有信息量”。
2. **Form-aware**：按形态去重，而不是按像素或场景；形态可以是姿势、角度、状态或形状。
3. **Training-aware**：训练模式下按形态封顶、保留稀有形态、考虑 caption 冗余，直接服务“避免过拟合”。
4. **Extractive & local**：只选现有图，不生成、不上传、可复现。
5. **Open & pluggable**：编码器、聚类、评分、归一化都可替换；CLI + UI + manifest。
6. **Non-destructive**：默认不删除；输出 selected/curated + report；所有决策可追溯。

## 8. 与替代方案的差异

| 替代方案 | 它做什么 | FormCoreset 的不同 |
|---|---|---|
| fastdup / SemDeDup / NeMo Curator | 图像 embedding 语义去重/删除 | 同时建模形态与 caption；按形态封顶；输出训练报告；不做 web-scale |
| Kohya / waifuc / ELSAC | 数据集采集、打标、角色一致性 | 不替代它们；补齐“形态感知去重 + 集合级选择”，可组合使用 |
| Apple/Google Photos、PixTwin、PhotoSort | 相似组 + 选最佳 | 按形态发现；训练模式；输出集合而非清理 |
| Photoroom / Cloudinary | 单图去背/裁切 | 多图比较与选择；开源、本地、可扩展 |
| 手工挑选 | 最灵活 | 自动化 + 可复现 + 可审核；处理几百上千张 |

**开源生态定位**：不是替代 Kohya/waifuc，而是补上它们缺的“形态感知去重”一环：

```text
waifuc 采集 → FormCoreset 去重/选样 → Kohya/diffusers 训练
```

## 9. 产品原则

1. **信息覆盖优先于激进删除**：宁可多留一张，也不要丢掉一个不同形态。
2. **默认非破坏性**：原图不动，所有输出可追溯。
3. **本地优先**：CPU 可跑，GPU 可选；默认不上传任何图片。
4. **人类在环**：聚类与选择可审核、可覆盖、可撤销。
5. **可复现**：manifest 记录模型、阈值、参数、版本。
6. **可插拔**：不绑定单一模型或云服务。
7. **先窄后宽**：先把一个 beachhead 做到最好，再扩展人物/物品/场景。

## 10. v0.1 范围

**做：**

- 一个含目标主体 + 干扰图的文件夹（典型：搜索引擎下载的 100 张图）。
- **身份过滤（四种输入方式）**：
  - 参考图模式（1–5 张，最稳）；
  - 文本模式：角色名 + WD14 标签（anime）/ 自由描述 + CLIP（弱先验）；
  - 文本 + 参考图融合（推荐默认）；
  - 自动身份聚类后由用户选目标簇（`--auto`）；
  - guided 模式输出三桶：`confirmed/`、`uncertain/`、`rejected/`；auto 模式输出 `subjects/` + `unassigned/`。
- **形态去重**：
  - 主体分割/去背仅用于形态分析（不是最终输出归一化）；
  - 同源配准合并裁剪/缩放/换背景变体；
  - 形态聚类：`person`（关键点 + 主体 embedding）/ `object`（embedding + 视角/状态）/ `auto`（仅 embedding）。
- **整理输出**：
  - `confirmed/form_XX/canonical.png` + `variants/`；
  - `uncertain/`、`rejected/`；
  - `contact_sheet.png`、`report.md`、`manifest.jsonl`。
- **训练模式**：`strict` / `balanced` / `diverse`；每形态上限 K。
- **可选 caption 支持**：Kohya `.txt` / JSONL；有 caption 时做形态 + caption 联合去重；没有则仅形态去重。
- CLI + 本地 Gradio 审核 UI。
- CPU fallback；可选 GPU。
- 英文 README + 中文文档。

**不做：**

- 多角色同图拆分（v0.2；v0.1 标为 `uncertain/`）。
- 输出归一化（去背/统一画布，v0.2；v0.1 只用分割做分析）。
- 生成式补全 / 重绘。
- 云端服务 / 账号 / 协作。
- 自动删除原图。

## 11. 开源策略

- **License**：推荐 **Apache-2.0**（专利授权、商业友好、易被采用）。若担心 SaaS 白嫖，可考虑 AGPL-3.0，但会降低采用率。
- **模型许可证**：代码与权重分离；默认不打包受限权重；提供多个 backend（DINOv2、DWPose/RTMPose、MediaPipe、CLIP），文档逐一标注许可证。
- **安装体验**：`pipx install form-coreset` / `uvx form-coreset`；一条命令跑 demo；CPU 也能出结果。
- **可扩展性**：`Encoder` / `Clusterer` / `Selector` / `Normalizer` 插件接口；社区可接自己的模型。
- **基准与样例**：发布一个小型 benchmark + 可视化样例；让贡献者有明确的优化目标。
- **下游验证**：提供 LoRA 训练脚本，对比“全量 vs 筛选后”的效果；这是最有说服力的 demo。
- **集成**：Kohya、diffusers、ComfyUI、FiftyOne、waifuc；后续可做插件。
- **社区**：英文为主、中文补充；issue 模板；good-first-issue；定期 release。
- **非破坏性承诺**：默认只读原图；这是信任的基础。

## 12. 命名候选

| 名称 | 优点 | 风险 |
|---|---|---|
| **FormCoreset** | 准确覆盖人/物/形态；直接表达 coreset（训练数据选择术语） | 略学术；“form”对部分用户较抽象 |
| **VisualCoreset** | 最通用、品类定义清晰 | 较泛，缺少“形态”这个差异化 |
| **SubjectCoreset** | 强调“同一主体、不同形态” | 名字没有点出形态维度 |
| **OnePerForm** | 价值主张极清晰、好记 | 暗示“每形态一张”，但产品可能输出 1–K 张 |
| **Forma** | 简短、有设计感 | 很可能已被占用，描述性弱 |

**推荐**：`FormCoreset` 作为仓库/项目名，**“Keep the information. Drop the duplicates.”** 作为主 tagline，**“Train on what's different.”** 作为训练场景 tagline。发布前查 GitHub / PyPI 占用。

## 13. README 首屏草案

```markdown
# FormCoreset

**Keep the information. Drop the duplicates.**
**Train on what's different.**

Turn a messy image search into a clean, form-organized set — and a better
training set.

[Demo GIF: 100 search results → 72 confirmed → 11 forms]

pipx install form-coreset

# 1) 文本：名字 / 介绍
form-coreset organize ./downloads --name "Hatsune Miku" --out ./organized
form-coreset organize ./downloads --description "blue twin-tail girl" --out ./organized

# 2) 参考图：1–5 张
form-coreset organize ./downloads --reference ref1.png ref2.png --out ./organized

# 3) 文本 + 参考图（推荐）
form-coreset organize ./downloads \
  --name "Hatsune Miku" --reference ref1.png --out ./organized

# 4) 完全自动：身份聚类 + 形态聚类
form-coreset organize ./downloads --auto --out ./organized

# 5) 整理 LoRA 训练集
form-coreset run ./my-dataset --mode training --max-per-form 3 --captions

- Works with a name, a description, reference images, both, or nothing
- Filters non-target images (confirmed / uncertain / rejected)
- Merges same-form variants with different backgrounds/crops
- Groups by form, picks a canonical + keeps variants
- Training-aware: per-form caps, rare-form preservation, caption dedupe
- Works for characters, people, and objects
- Local-first, non-destructive, reproducible
- CLI + review UI + manifest

[Quickstart] [How it works] [Search-result demo] [Training benchmark] [Roadmap] [Contributing]
```

## 14. 成功指标

- 5 分钟内跑通 demo（CPU）。
- **100 张搜索结果 demo**：身份过滤 + 形态分组 + contact sheet，人工复核后可用。
- 下游 LoRA benchmark：用筛选后的 30–70% 数据，达到与全量数据相同或更好的效果。
- 在 3 个公开/自建样例上，form coverage 与人工选择一致率达标。
- 社区能用自己的模型/数据跑通（插件接口）。
- 至少 2 个生态集成（Kohya / diffusers / ComfyUI / FiftyOne）。
- Issue/PR 中出现“我用它整理了 XX 数据集”的真实用例。

## 15. 待确认

1. 项目名选哪个？（FormCoreset / VisualCoreset / SubjectCoreset / OnePerForm / 其他）
2. 四种输入方式是否都作为 v0.1 范围：文本（名字/介绍）/ 参考图 / 文本+参考图 / 不给（`--auto`）？
3. 自动模式下，输出所有身份簇，还是只输出最大簇 / 让用户选？
4. 需要自动给身份簇命名吗（WD14 角色标签 / CLIP zero-shot）？
5. 变体是否保留：`variants/` 子目录，还是只留 canonical？
6. Training mode 是否作为第一优先级，Reference mode 作为第二？
7. caption 格式：Kohya `.txt` / JSONL / 两者都支持？
8. 第一个下游验证用哪类数据：角色 LoRA 还是商品图？
9. License 选 Apache-2.0 还是 AGPL-3.0？
