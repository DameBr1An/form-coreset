# FormCoreset 研究议程：补齐 3D 输入工具链的技术缺口

> 目标：当现成模型解决不了“多姿势/多角度 → 单一 canonical 3D 资产”时，用新的技术组件把工具链补完整。  
> 原则：不从头训练 3D 基础模型；优先做输入侧、融合侧、评估侧的高杠杆组件。

## 0. 当前技术解决不了什么

| 缺口 | 现状 | 后果 |
|---|---|---|
| **Multi-pose → canonical static 3D** | LHM++ 只能出 3DGS avatar；Pixal3D/Hunyuan/Meshy/Tripo 要求同姿势多视图；RCM/UP2You 未发布 | 不同姿势的图无法变成 UE5 静态 GLB |
| **Generated view 的可信融合** | 生成图被当成真实观测，或多视图模型直接拒绝 | 幻觉被烘进几何，或干脆不能用 |
| **3D-aware 输入选择** | 选图靠启发式/CLIP/形态聚类 | 不知道哪几张图真正提升 3D 质量 |
| **Cross-pose 身份一致性** | CCIP/DINOv2 对极端姿势/服装漂移 | 选进不同角色/版本，3D 融合失败 |
| **Game-ready 拓扑/UV/绑骨** | AI mesh 需手工重拓扑 | 不能直接进 UE5 |

## 1. 设计原则

1. **输入侧优先**：先做“选对图、标对视角、给对相机”，再考虑生成。
2. **用下游 3D 质量做裁判**：任何组件都必须用 3D 重建/生成质量验证，不看“看起来合理”。
3. **合成数据 + 真实数据双轨**：合成数据给监督和消融，真实数据给泛化。
4. **模块化**：每个组件可独立替换，不绑定某个 3D 后端。
5. **开放 benchmark**：数据、脚本、指标开源；负结果也发布。
6. **8GB 可推理**：训练可以上云，推理必须能在目标硬件跑。

---

## 2. 组件 A：3D-Aware Coreset Selection（3D-Coreset）

**优先级：P0（4–8 周）**

### 问题

给定 N 张同一角色的不同姿势/视角图片，选哪个子集喂给某个 3D 后端，能最大化重建质量？现有方法是“按形态去重 + 选清晰度”，没有 3D 几何概念。

### 形式化

- 图片 \(i=1..N\)；
- 角色表面/身体部件 patch \(p=1..P\)；
- 覆盖 \(c_{ip}\in[0,1]\)：图片 i 能看到 patch p 的程度；
- 质量 \(q_i\)，冗余 \(r_{ij}\)；
- 后端契约约束（数量、同姿势/多姿势、相机）。

选择集合 \(S\)，\(|S|\le K\)：

\[
\max_{S}\ \sum_{p} w_p \min\left(1,\sum_{i\in S} c_{ip}\right)
+ \lambda \sum_{i\in S} q_i
- \mu \sum_{i<j\in S} r_{ij}
\]

- 覆盖项是单调子模；质量是模块化；冗余是惩罚项。
- 用受约束贪心 + lazy evaluation；必要时用 graph cut。
- 对 multi-pose 输入，不要求相机一致，但要求身份一致；对 multi-view 输入，加同姿势/相机约束。

### 覆盖 \(c_{ip}\) 怎么来

1. **几何代理**：关键点 + 视角 + 分割，估计每个身体部件/表面的可见性；
2. **学习可见性**：合成渲染 (pose, view) → 可见 patch，训练一个小模型；
3. **直接效用模型**：用 (输入集合, 3D 质量) 数据训练边际增益预测器。

### 数据与评估

- 合成：Objaverse / VRM / 动漫角色资产，随机姿势、视角、背景、裁剪、压缩；有 GT canonical mesh。
- 真实：用户提供的角色/物品集，人工标注视角/姿势/质量。
- 后端：Pixal3D / LocalMesh / UniForge3D / LHM++。
- 指标：Chamfer / F-score、纹理质量、身份一致性、UE5 导入成功率、人工偏好。
- 对比：random、top-quality、form-diverse、3D-Coreset。

### 新意

- 第一个“3D-aware coreset”用于 image-to-3D；
- 可以推出 **input scaling law**：3D 质量如何随视角/姿势覆盖、一致性、质量变化；
- 直接变成 FormCoreset 的 `readiness` 和 `3d` 选图功能。

### 风险

- 合成-真实 gap；
- 3D 后端评估成本高；
- 需要 GPU 渲染/推理。
- 缓解：先做小规模（20 个资产 × 20 视角），再扩。

---

## 3. 组件 B：Uncertainty-Aware Observed/Generated Fusion（UOGF）

**优先级：P1（2–4 个月）**

### 问题

生成补视图能补背面/侧面，但生成图是“假设”，不是观测。直接融合会把幻觉烘进 3D。

### 方法

- 对每张生成图估计置信度：
  - 多采样一致性（同一 prompt/seed 多次生成）；
  - 与真实观测的身份/特征相似度；
  - 与真实视图的 epipolar/feature 一致性；
  - 分割/掩码质量。
- 在 3DGS/NeRF/网格优化中给每视图权重：
  \[
  L = \sum_{i\in O} L_i + \sum_{j\in G} w_j L_j + \lambda L_{cons}
  \]
  - \(O\)：observed，高权重；\(G\)：generated，低权重/不确定性加权；
  - \(L_{cons}\)：生成视图与观测几何的一致性惩罚。
- 可选：贝叶斯神经场，给每视图噪声项。

### 新意

- **观测-假设概率融合**：把生成图当软约束，而不是硬观测；
- 可以显著降低“生成导致几何崩坏”的风险；
- 与 RCM/Zero123++/SV3D 等生成器解耦。

### 评估

- 合成 GT：已知真实多视图，故意删掉部分视角，用生成补，比较融合策略；
- 真实：用户研究，比较 observed-only vs +generated（naive vs UOGF）。

### 风险

- 需要 3DGS/NeRF 代码与显存；
- 8GB 推理可能紧；
- 生成器质量上限决定收益。

---

## 4. 组件 C：Cross-Pose Identity & Consistency Embedding（CPIE）

**优先级：P1（2–3 个月）**

### 问题

同一角色不同姿势/服装的图像，用 CCIP/DINOv2 可能被当成不同身份；不同角色的相似图可能被混在一起。3D 输入需要“同一角色 + 可融合”的一致性判断。

### 方法

- 输入：关键点 + 主体 mask + 灰度/去色 crop；
- 训练：合成多姿势/多视角渲染，正样本 = 同角色不同姿势，负样本 = 不同角色/相似角色；
- 解耦：pose-conditioned residual / adversarial pose classifier，让 embedding 对姿势不敏感、对身份敏感；
- 输出：身份 embedding + pair-wise consistency + 版本/服装冲突标记。

### 新意

- 把 re-ID 从“同图检索”推进到“跨姿势 3D 输入一致性”；
- 直接服务身份过滤和 3D 选图。

### 风险

- 需要合成多姿势数据；
- 动漫/风格化角色泛化；
- 与 CCIP 的互补性需要验证。

---

## 5. 组件 D：Pose-Invariant Multi-Reference Reconstruction（PIMR）

**优先级：P2（6–12 个月，moonshot）**

### 问题

真正的“多姿势/多角度 → canonical mesh”还没有成熟模型。LHM++ 只做人物 avatar，RCM/UP2You 未发布，DreamBooth3D 太慢。

### 方法

- 建立 **canonical space**：一个模板/隐式 canonical 体积；
- 每张图通过 **pose-conditioned deformation** 反投影到 canonical space；
- **cross-pose attention** 聚合多张图的特征；
- 解码 canonical SDF/mesh + texture；
- 训练目标：多视图渲染损失 + cycle consistency + canonical prior + identity contrastive；
- 对动漫：用 2D 骨架 + 剪影替代 SMPL，学习 deformation field；
- 蒸馏成 feed-forward 模型，目标 8GB 推理。

### 新意

- **canonical-space multi-reference fusion**：把不同姿势的观测统一到一个几何空间；
- 可能成为 FormCoreset 的长期护城河：别人只有多视图模型，我们有“多姿势 → canonical”的模型。

### 风险

- 研究风险高；数据/算力需求大；
- 动漫比例/服装/遮挡难；
- 8GB 推理难；可能需要蒸馏/量化；
- 没有 GT canonical mesh 的真实数据，只能合成训练。

### 里程碑

1. 合成数据 + 人物/物体原型；
2. 单类别（人物）验证；
3. 动漫风格迁移；
4. 蒸馏/8GB；
5. UE5 导出。

---

## 6. 组件 E：Template-Conditioned Game-Ready Reconstruction（TCGR）

**优先级：P1（1–2 个月，工程为主）**

### 问题

AI mesh 拓扑/UV/绑骨不可用，UE5 需要 game-ready 资产。

### 方法

- 用模板网格（humanoid/object）拟合重建形状，继承拓扑/UV/骨骼；
- 或对生成 mesh 做 neural/quad retopo；
- UniRig/RigAnything 自动绑骨；
- b2ue/reference-asset-compiler 做 UE5 导入/重定向/验证；
- PBR 烘焙（Hunyuan3D-Paint / Substance / Blender）。

### 新意

- 不是新理论，是**把“重建”与“game-ready”接起来**；
- 对 UE5 用户价值极高。

### 风险

- 模板适配 vs 角色多样性；
- 自动绑骨质量；
- 重拓扑时间。

---

## 6.5 组件 F：Pose-Aware Texture Transfer（PATT）

**优先级：P1（2–4 个月）**

### 问题

texture-combined 的直觉是“把所有有效图投影到网格上”。但**只有纹理图和网格姿势一致时，直接投影才成立**。不同姿势的图（坐姿、动作、侧面）直接投到 canonical A/T-pose 网格上，会错位：手臂、腿、头发的位置都不对。

### 方法

1. **同姿势 refs**：直接相机投影 + 可见性 mask；
2. **不同姿势 refs**：
   - 用关键点/轮廓/UV 对应，把像素映射到 canonical mesh；
   - 或先用 2D pose transfer / image editing 归一化到 canonical pose，再投影；
   - 对无法对应的区域，用 detail refs 局部补全，或标记 `generated/uncertain`；
3. **置信度**：每张 ref 记录 `pose_match`、`alignment_confidence`、`visibility_mask`；
4. **Blender**：投影/烘焙/补洞；生成 UV/PBR；
5. **失败处理**：对应不上的图不硬投，保留为 detail/readiness。

### 数据与评估

- 合成：已知 canonical mesh + 随机姿势渲染，GT UV；衡量投影误差/纹理一致性；
- 真实：用户研究，比较“直接投影 vs pose-normalized 投影 vs detail-only”；
- 指标：纹理接缝、身份一致性、细节保留、人工偏好。

### 新意

- 把“多姿势参考”变成**可投影的纹理证据**，而不是只能做几何融合；
- 是 texture-combined 能否成立的关键组件；
- 与 RCM/2D pose transfer 解耦，可独立迭代。

### 风险

- 动漫角色没有可靠 UV/关键点对应；
- 姿势差异大时，对应误差大；
- 生成式归一化会漂移；
- 8GB 下 Blender/投影可跑，但生成模型可能吃显存。

---

## 7. Benchmark：InputBench-3D

> 没有 benchmark，就没有“新理论”。

### 数据

- 合成：Objaverse / VRM / 动漫资产，随机姿势/视角/背景/裁剪/压缩；GT canonical mesh + PBR。
- 真实：用户提供；人工标注视角/姿势/质量/身份。
- 规模：先 20–50 个资产 × 20–50 张图；再扩。

### 任务

1. 给定图像集，选 K 张给后端；
2. 预测 3D 质量/失败模式；
3. 比较 observed-only vs +generated；
4. 比较 multi-view vs multi-pose 路由。

### 指标

- 几何：Chamfer、F-score、normal consistency；
- 纹理：CLIP-IQA、LPIPS、人工偏好；
- 身份：DINO/CCIP 相似度；
- UE5：导入成功率、面数、材质完整性、绑骨可用性；
- 成本：显存、时间。

### 产出

- 开源数据脚本 + 标注 + baseline；
- 论文/技术报告；
- FormCoreset 的 readiness/selector 直接用。

---

## 8. 路线图

| 阶段 | 时间 | 内容 | 产出 |
|---|---|---|---|
| **Phase 1** | 4–8 周 | 3D-Coreset + InputBench-3D + readiness | 选图算法、benchmark、产品功能 |
| **Phase 2** | 2–4 个月 | UOGF + CPIE | 生成融合、跨姿势一致性 |
| **Phase 3** | 6–12 个月 | PIMR + TCGR | 多姿势 canonical 重建、game-ready |
| **持续** | — | 后端 adapter、UE5 handoff、开源 | 工具链 |

---

## 9. 风险与止损

| 风险 | 止损 |
|---|---|
| 3D-Coreset 合成-真实 gap 大 | 先小规模真实评测；不达标就退回启发式 |
| UOGF 8GB 跑不动 | 云端/离线；或只做推理侧置信度过滤 |
| CPIE 训练数据不足 | 用 DINOv2 + pose normalization 近似；不做训练 |
| PIMR 研究失败 | 保留为研究项目，不影响产品；先做 RCM/UP2You 集成 |
| Benchmark 版权 | 只开源脚本/标注；样本可再分发 |
| 8GB 限制 | 训练上云，推理蒸馏/量化；不承诺本地训练 |

---

## 10. 与 FormCoreset 的集成

- `BackendContract.input_semantics`: `multi_view | multi_pose | unposed_object | single_image`
- `3D-Coreset` → `3d` 模式的 selector；
- `UOGF` → 可选 `generate` 模式；
- `CPIE` → identity/consistency scorer；
- `PIMR` → 新 backend adapter；
- `TCGR` → UE5 handoff；
- 所有组件输出写入 manifest（provenance、confidence、model/version）。

---

## 11. 不应该做什么

- 不从零训练 3D 基础模型；
- 不重写分割/姿态/embedding 基础模型；
- 不做通用图像生成器；
- 不做 web-scale 数据集去重；
- 不自己写 UE5 导入器/绑骨器（接 b2ue/VERA/UniRig）。

---

## 12. 一句话

> 现成模型解决“同姿势多视图”；我们补的是 **“多姿势输入 → canonical 3D”** 的缺失层：先用 3D-Coreset 把输入选对，再用 UOGF 把生成图安全融合，最后用 PIMR 做 canonical 重建。  
> 这条路线既有短期产品价值，也有长期技术护城河。
