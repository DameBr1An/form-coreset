# FormCoreset · 决策记录（ADR）

> 记录已确认的产品/技术决策，避免后续反复讨论。  
> 状态：v0.1 已确认项

## ADR-001 项目名使用 FormCoreset

- **日期**：2026-10-25
- **状态**：已确认
- **上下文**：需要同时覆盖人物姿势与物品形态，`PoseCoreset` 太窄。
- **决策**：项目名 **FormCoreset**；仓库 `form-coreset`；Python 包 `form-coreset`；import 名 `formcoreset`。
- **后果**：名称略学术；通过副标题 “One high-value image per distinct form” 降低理解成本。发布前查 GitHub / PyPI 占用。

## ADR-002 第一 beachhead 是 AI 绘画 / LoRA 数据集

- **日期**：2026-10-25
- **状态**：已确认
- **上下文**：需要第一批真实用户和传播路径；AI 画师/LoRA 训练者已经在 GitHub、Civitai、ComfyUI、Kohya 生态中，痛点是“同姿势重复图太多”和“训练集重复导致过拟合”。
- **决策**：README、示例、benchmark 先围绕角色参考集与 LoRA 训练集；商品/物体作为第二验证域。
- **后果**：anime 相关的 CCIP/WD14/anime-seg 优先；object adapter 放 P1 实验性；真人放 v0.2。

## ADR-003 代码许可证使用 Apache-2.0

- **日期**：2026-10-25
- **状态**：已确认
- **上下文**：需要专利授权、商业友好、容易被社区和公司采用。
- **决策**：代码使用 **Apache-2.0**。
- **后果**：允许商业使用与修改；需保留 NOTICE/LICENSE；模型权重不随代码打包，逐一审计许可证。

## ADR-004 v0.1 适配器范围

- **日期**：2026-10-25
- **状态**：已确认
- **决策**：
  - `person`：P0（RTMPose + HumanArt + DINOv2）。
  - `auto`：P0（DINOv2；检测到人体时附加 keypoints）。
  - `object`：P1 实验性（DINOv2 聚类；视角/状态标签可选，不依赖 CLIP zero-shot）。
- **后果**：v0.1 的公开 demo 聚焦动漫/角色；object 文档标注 experimental，不承诺同等精度。

## ADR-005 本地优先、非破坏性

- **日期**：2026-10-25
- **状态**：已确认
- **决策**：
  - 默认不上传图片；无遥测。
  - 原图只读；输出复制/硬链接；默认不删除。
  - 不确定结果进 `uncertain/`，不强行分类。
- **后果**：模型首次下载需要网络，之后可离线；输出目录可能占用额外磁盘。

## ADR-006 v0.1 不做生成式与输出归一化

- **日期**：2026-10-25
- **状态**：已确认
- **决策**：v0.1 只做提取式选择与整理；分割只用于形态分析，不做去背输出、统一画布、生成式补全。
- **后果**：产品价值聚焦在“选得对、分得对”；归一化放 v0.2。

## ADR-007 自动模式输出所有主体簇

- **日期**：2026-10-25
- **状态**：已确认
- **决策**：`--auto` 不假设最大簇就是目标；输出全部 `subjects/` + `unassigned/`，由用户在 UI 中选择目标簇。
- **后果**：避免“自动猜错目标”导致整批错误；UI 需要支持簇选择。

## ADR-008 模型权重与代码分离

- **日期**：2026-10-25
- **状态**：已确认
- **决策**：代码 Apache-2.0；模型首次运行时下载，带 checksum、版本和许可证提示；默认不打包受限权重。
- **后果**：安装包更小；需要模型注册表和离线模型目录；SuperPoint 等受限权重不作为默认依赖。

## ADR-009 Form-caption 联合冗余采用 soft-AND + floors

- **日期**：2026-10-25
- **状态**：已确认
- **上下文**：训练集去重不能只看图像 embedding（会误删“同形态不同 caption”），也不能只看 caption（会误删“同 caption 不同形态”）。需要同时表达“两个轴都重复才强冗余”和“单独一个轴重复也有部分冗余”。
- **决策**：
  - 形态与 caption 分别阈值化，得到 `ft`、`ct`；
  - 联合冗余 `R = max(ft*(γf+(1−γf)ct), ct*(γc+(1−γc)ft))`；
  - `γf=0.30`、`γc=0.10` 为默认 floor；
  - 没有 caption 时退化为 form-only；
  - 相对集合使用 top-k 边际冗余；选择用带每形态上限的贪心。
- **后果**：参数可解释、可校准；不是端到端学习；贪心非全局最优，后续可替换为受约束子模优化。

## ADR-010 主定位切到 3D-ready 输入编译器

- **日期**：2026-10-25
- **状态**：已确认
- **上下文**：通用去重/coreset 已被 imgraft、cull、PixLint、fastdup、FiftyOne、NeMo 等覆盖；3D 生成工具快速增加，但“什么图适合哪个 3D 后端”仍靠人肉试错；用户主要目标是 UE5 角色/物品资产。
- **决策**：
  - 主定位：**3D-ready 输入编译器**（按后端契约编译参考集）。
  - `training` 降为第二模式；`reference` 第三。
  - 默认 **observed-first**；生成补视图只补缺口，标记 `generated`。
  - 不做 3D 生成器；通过 adapter 接 Pixal3D / LocalMesh / UniForge3D / LHM++ / Hunyuan3D-2mv / 云端。
- **后果**：Backend Contract、readiness、camera/view estimation、provenance 成为核心；training 代码保留但不再主导 roadmap。

## ADR-011 目标域确认为动漫/风格化角色与物品，UE5 规范化输出

- **日期**：2026-10-26
- **状态**：已确认
- **决策**：
  - 目标域：**动漫/风格化角色 + 物品**。
  - 输出：可导入 UE5 的规范化 3D 资产（静态网格 + PBR；角色需要绑骨）。
  - **Blender** 作为 AI 生成之后的清理/重拓扑/UV/绑骨/烘焙层。
  - **LHM++ 不作为动漫角色主后端**；仅作实验（真人人体模型，风格化泛化未知）。
  - 动漫角色：CharacterGen / Pixal3D / 云端 Meshy/Tripo + Blender；RCM / UP2You / PIMR 为后续。
  - 物品：UniForge3D-Low-VRAM / LocalMesh + Blender。
- **后果**：选型从“真人 LHM++”转向“动漫可用的单图/多视图生成 + Blender 规范化”；多姿势直接生成仍是研究缺口。

## ADR-012 3D 输出目标改为“所有有效图 → 一个 3D 模型”（combined）

- **日期**：2026-10-26
- **状态**：已确认
- **决策**：
  - 主目标：**combined mode**，参考所有有效图，生成一个 3D 模型；per-form 降为辅助。
  - 当前技术约束：多视图后端不能直接融合不同姿势；动漫没有可靠的本地 8GB multi-pose 模型。
  - 分两级实现：
    1. **texture-combined**（v0.2/v0.3）：几何用一致子集或生成 turnaround；纹理用所有有效图投影/烘焙。
    2. **geometry-combined**（v0.4+）：RCM/UP2You/PIMR 做真正的多姿势几何融合。
  - 所有图标注 `observed/generated`、pose/view、camera；manifest 记录。
- **后果**：FormCoreset 的输出是 combined package（`geometry_views/` + `texture_refs/` + transforms/readiness），而不是 per-form 列表；真正的全姿势几何融合仍是研究目标。

## ADR-013 3D 建模的聚合单位是“装扮/版本”，不是姿势

- **日期**：2026-10-26
- **状态**：已确认
- **决策**：
  - **同一套装扮/版本的不同姿势、不同角度 → 全部作为同一个 3D 模型的参考。**
  - **不同装扮/版本 → 分开建模，每个版本一个 3D 模型。**
  - 数据模型：`Character → Version(outfit) → Form(pose/view) → Images`。
  - Form Library 先按 Version 分组，再展示 Version 内的形态/姿势。
  - texture-combined / PATT 只在同一 Version 内进行；不跨装扮混合纹理。
- **后果**：
  - 新增 **outfit/version clustering**（外观 embedding + 颜色/服装标签 + 人工复核）；
  - 3D 输出按 Version 分组，每个 Version 一个模型；
  - 不同装扮不再混进一个模型，纹理冲突大幅减少。

## ADR-014 Version 聚类降级为可选；v0.2 默认一个角色一个装扮

- **日期**：2026-10-26
- **状态**：已确认
- **决策**：
  - 默认假设：**一个角色只有一个形态/装扮**；所有有效图归入同一个 Version。
  - **v0.2 不做自动 Version/outfit 聚类**；只保留手动 `--version` 分组/标记。
  - 若检测到明显装扮冲突，标记 `possible_version_conflict`，由用户决定是否拆分。
  - 数据模型保留 `Character → Version → Form`，但 Version 默认为单例。
- **后果**：把精力集中到 identity/form dedupe、geometry anchor、texture-combined/PATT、Blender/UE5；自动 outfit clustering 后置到有真实需求时再做。

## ADR-015 动漫 form encoder 采用 WD14 tags + CCIP embedding

- **日期**：2026-10-27
- **状态**：已确认
- **上下文**：基准显示 DWPose / RTMPose HumanArt 在动漫/风格化图上关键点不可靠；WD14 tags 对 posture/view 明显更强，CCIP 是身份模型但可作辅助外观信号。
- **决策**：
  - 默认 `anime-hybrid` encoder：**WD14 内置 tag 先验 0.7 + CCIP embedding 0.3**。
  - **产品输出不包含姿势名称/标签**，分组只用 `form_001` 这样的 cluster ID；不要求用户定义或标注姿势名。
  - 聚类是无监督 HDBSCAN；内置 tag 先验只作为特征，不参与标签监督。
  - **keypoints 不作为默认特征**；仅高置信度时作为可选辅助。
  - 完全无先验的 WD14 embedding / TF-IDF / DINOv2 已测试，明显更弱；不作为默认。
  - noise/编码失败进入 `uncertain/`。
  - 基准结果记录在 `benchmarks/RESULTS.md`。
- **后果**：`form-coreset extract` 可直接产出无标签的 Form Library；running/jumping/action 仍需要人工复核或后续 CPIE/VLM 研究；CCIP 继续用于身份，不用于姿势判断。

## ADR-016 同源变体去重优先于跨画师姿态聚类

- **日期**：2026-10-26
- **状态**：已确认
- **上下文**：用户的主要数据是同一张海报/原图的变体（换背景、特效、裁剪、缩放、压缩），不同画师/不同绘制的同动作图很少。这类问题不需要姿势语义，局部特征配准即可高精度合并；姿态聚类在动漫上反而不可靠。
- **决策**：
  - `form-coreset dedupe` 为第一优先路径：SIFT + ratio test + RANSAC 单应性。
  - **分组默认 complete linkage**：组内任意两张都必须直接匹配，避免 A~B、B~C、A≁C 的链式合并；`--linkage average|single` 可切到 recall 优先。
  - **star merge 默认开启**：complete 种子组生成后，把与组内 medoid 高置信匹配的 singleton 拉回；只允许 singleton → group，不允许 group → group；`--star-threshold` 默认 0.20。
  - 默认 `min_score=0.3`、`min_inliers=12`、`min_correlation=0.35`；ORB 可选。
  - **anime mask 默认开启**：isnet-anime 去背后提特征。
  - **aligned-edge 验证默认开启**：warp 后验证边缘相关性，拒绝“相似角色/巧合匹配”的假 homography。
  - **canonical stitching 默认开启**：互补裁剪对齐后拼接；输出 `canonical.png`（拼接）与 `canonical_single.*`。
  - 输出 `sources/source_XXX/`（canonical + variants），**无姿势标签**。
  - `form-coreset extract` 的形态聚类降为第二路径，只处理不同绘制/动作。
  - 合成同源基准（8 原图 × 7 变体，含裁剪/缩放/JPEG/换背景/亮度/特效）：
    - mask + edge verify，single：precision/recall/F1 = 1.0（有链式风险）。
    - mask + edge verify，average：precision 1.0 / recall 0.964。
    - mask + edge verify，complete：precision 1.0 / recall 0.935；2 个弱变体保守拆成 singleton。
    - **mask + edge verify，complete + star merge（默认）：precision / recall / F1 = 1.0**。
    - mask only，single，0.3：precision 0.774（相似原图会被合并）。
    - no mask，single，0.15：precision 0.533（明显过度合并）。
- **后果**：产品首要价值是“同源海报/原图变体去重”，不是跨画师姿态聚类；默认 complete + star merge 兼顾 precision/recall；真实数据仍需按域调阈值；拼接失败回退 single canonical。

## ADR-017 双轴 star merge 暂不采用（有证据的拒绝）

- **日期**：2026-10-26
- **状态**：已拒绝（当前规则保留）
- **背景**：Jett 样本的 015 与 B 家族 edge correlation 0.943，但 SIFT score 0.181；提出 `score ≥ 0.15 且 corr ≥ 0.80` 的双轴 star merge。
- **评估**：
  - 合成同源基准：current / replacement / union 都是 8 组、precision/recall/F1 = 1.0，无收益。
  - Jett 样本：current 与 union 都是 22 组、P=1.0 / R=0.750；replacement 为 24 组、P=1.0 / R=0.450，更差。
  - 015 的高相关边连接的是组员 010，不是 medoid；medoid-only 双轴规则无法召回它。
- **决策**：不实现双轴规则；保持 `star_threshold=0.20` 的 medoid star merge。
- **后果**：015 仍需审核 UI/人工合并；若继续优化，应先评估 member-level high-corr merge，而不是继续调 medoid 阈值。
