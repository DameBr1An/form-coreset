# FormCoreset 产品战略 v2（2026-10）

> 基于 2026 年 10 月最新开源工具与 3D 管线调研。  
> 结论：从“形态感知去重/coreset”升级为 **“3D-ready 输入编译器（input compiler）”**。

## 0. 一页结论

**旧定位**：形态感知的视觉核心集提取器，用于参考图库和训练集去重。

**新定位**：

> **FormCoreset 把一堆杂乱图片，编译成某个 3D 后端可以直接消费的参考集——同时保留来源、观测/生成标记和可复现清单。**

英文：

> **Compile a messy image collection into a 3D-ready reference set.**

**为什么现在**：

1. 通用去重/coreset 已经拥挤：imgraft、cull、PixLint、fastdup、FiftyOne、NeMo Curator 都在做。
2. 3D 生成在快速进步：Pixal3D、TRELLIS.2、LocalMesh、UniForge3D、Hunyuan3D-2mv、LHM++。
3. 但这些 3D 工具的输入契约各不相同，且都缺少“从网络图到合格输入”的编译器。
4. 相机位姿估计已经可以用 Apache-2.0 的 Depth Anything 3（SMALL/BASE）做，不再必须 COLMAP。
5. UE5 后处理已有 b2ue、reference-asset-compiler、VERA、UniRig/RigAnything；我们应做“输入编译 + handoff”，不重复造轮子。

**三个模式**：

- `training`：LoRA/扩散训练集去重与每形态封顶。
- `reference`：角色/物品参考图库、contact sheet、taxonomy。
- `3d`：按 3D 后端契约编译参考集（Pixal3D MV / Hunyuan3D-2mv / LocalMesh / UniForge3D / LHM++ / 云端）。

**核心护城河**：

- Backend Contract 库；
- 观测/生成 provenance；
- 输入消融 benchmark（证明“输入整理”能提升 3D 质量）；
- UE5 handoff；
- 开源社区与集成。

**不是**：3D 生成器、通用照片清理、web-scale 数据集去重平台。

---

## 1. 2026 市场变化

### 1.1 数据整理层：已经拥挤

| 工具 | 定位 | 对我们的影响 |
|---|---|---|
| **imgraft** | CLIP + HDBSCAN/UMAP，输出 core / diverse / noise | 最接近“coreset”；但没有 3D 契约、没有形态/视角 |
| **cull** | AI 图像数据集策展引擎：抓取 → 去重 → VLM 分类 → 导出 Kohya/HF | 覆盖“采集 + 整理 + 导出”；没有 3D / UE5 |
| **PixLint** | MCP 数据集工具：质量、重复、标签错误、语义搜索 | 非商业许可；更像数据集 CI |
| **NeMo Curator 26.x** | GPU/Ray 语义去重、分类、过滤 | 大规模训练数据；不做 3D 输入 |
| **FiftyOne / fastdup / Lightly / imagededup** | 数据集探索、近重复、主动学习 | 通用平台/库；不做形态契约 |

**结论**：如果 FormCoreset 只做“相似图去重 / coreset”，会直接和它们竞争，且没有明显优势。差异化必须落在 **3D 输入契约**。

### 1.2 3D 输入/生成层：输入准备是空白

| 工具 | 输入 | 输出 | 8GB | 许可 | 备注 |
|---|---|---|---|---|---|
| **Pixal3D** | 单图；多视图 + `transforms.json` | GLB + PBR | 边界（low_vram） | MIT | 目前最接近“带 PBR 的 GLB” |
| **TRELLIS.2** | 单图 | GLB/PBR | 24GB 官方；低显存 fork | MIT | Pixal3D 的基座 |
| **LocalMesh Engine** | 1 张，或 4 张 front/right/left/back | GLB + texture | ✓ | Apache-2.0 | DA3 估角度；Pixal3D FP8；物体为主 |
| **UniForge3D-Low-VRAM** | 2–4 张 unposed 物体照片 | GLB + 2048 PBR | ✓（6.3GB） | Apache-2.0 代码；权重另有条款 | 同一物体；30GB 权重；23GB RAM |
| **Hunyuan3D-2mv** | 正/背/左/右标准视图 | GLB/OBJ；纹理另跑 | 形状 ✓，纹理紧 | 腾讯社区许可 | 许可证限制多 |
| **LHM++** | 1–16 张 pose-free 人物图 | 3DGS avatar | ✓ | 需核对 | 人物专用；非 GLB/PBR |
| **RCM** | 1–4 张任意姿势角色图 | 1024² turnaround | ? | 待发布 | 姿势归一化 + 轨道视图，方向正确 |
| **Meshy / Tripo / Rodin** | 一致多视图 | GLB + rig/动画 | 云端 | 商业 | 多图模式要求同姿势 |

**结论**：3D 生成器在进步，但“什么图适合哪个后端”仍靠人肉试错。这是 FormCoreset 的机会。

### 1.3 相机 / 姿态 / 几何：新工具让“编译器”可行

- **Depth Anything 3（DA3）**：任意数量视图、可有/无相机位姿；SMALL/BASE 是 **Apache-2.0**，LARGE/GIANT 是 CC BY-NC。可用于估计相机位姿，替代部分 COLMAP。
- **VGGT**：代码 Meta VGGT License（商业可用但排除军事）；原始权重非商业，商业 checkpoint 需申请。默认不用。
- **COLMAP**：仍然可靠，但慢、对无纹理/不同姿势不友好。
- **RTMPose + HumanArt / DWPose / SDPose**：姿态关键点；HumanArt 对动漫比例更稳。
- **SAM 2 / SAM 3 / BiRefNet / RMBG-2.0 / isnet-anime**：分割/去背；SAM 3 许可是自定义 SAM License，需核对。

**结论**：用 DA3-SMALL/BASE + 人体关键点，就能做“视角估计 + 相机位姿 + 契约检查”，不必一开始就上重型 SfM。

### 1.4 UE5 交付层：已有轮子，应该接

| 工具 | 作用 | 许可 | 备注 |
|---|---|---|---|
| **b2ue** | Blender → UE5：清理、减面、法线烘焙、自动绑骨（UE5 骨骼名）、FBX 导出 | 需核对 | 非常贴近“AI 资产进 UE5” |
| **reference-asset-compiler** | 单 T-pose 图 → Hunyuan3D → 重拓扑/UV/烘焙 → 绑骨 → UE5 | Apache-2.0 | 可借鉴/集成 |
| **VERA** | UE5 编辑器内 AI agent：IK Rig、Retargeter、批处理 | MIT | 自动化 retarget |
| **UniRig / RigAnything** | 自动绑骨 | 需核对 | 生成后可选 |
| **ComfyUI-Geekatplay-GameAssetMake** | ComfyUI + UE5 bridge | 需核对 | 生成→导入 |

**结论**：FormCoreset 不应该自己写 UE5 导入器/绑骨器；应该输出 contract-compliant 输入，并调用/指导这些工具。

### 1.5 许可证现实

| 模型/工具 | 许可 | 默认可用？ |
|---|---|---|
| Pixal3D | MIT（代码 + HF 权重） | ✓ |
| LocalMesh | Apache-2.0 | ✓ |
| UniForge3D | Apache-2.0 代码；DINOv3/RMBG 权重条款 | ✓ 代码，权重需核对 |
| DA3 SMALL/BASE | Apache-2.0 | ✓ |
| DA3 LARGE/GIANT | CC BY-NC 4.0 | ✗ 商业 |
| VGGT 原始权重 | CC BY-NC 4.0 | ✗ 商业默认 |
| DINOv2 | Apache-2.0（以当前模型卡为准） | ✓ |
| DINOv3 | 自定义 Meta 许可，商业有条件；权重 gated | 可选 |
| SAM 3 | 自定义 SAM License | 需核对；默认 SAM 2/BiRefNet |
| Hunyuan3D-2mv | 腾讯社区许可（MAU/地域/输出限制） | 可选后端，不默认 |
| Meshy/Tripo/Rodin | 商业 API | 用户自带 key |

**原则**：默认后端只选 Apache/MIT；受限模型做可选插件，启动时提示许可证。

---

## 2. 新定位

### 2.1 一句话

> **FormCoreset 把一堆杂乱图片，编译成某个 3D 后端可以直接消费的参考集。**

主 tagline：**Compile images into 3D-ready data.**  
副 tagline：**Keep the information. Drop the duplicates.**

### 2.2 定位声明

> 面向 AI 画师、游戏/UE5 独立开发者、3D 资产管线：FormCoreset 是一个开源的 **3D-ready 输入编译器**。它从搜索/素材库里过滤目标主体、发现不同形态、去背并提取完整角色/物品，然后按照目标 3D 后端的输入契约（多视图 + 相机位姿 / 标准视图 / unposed 多图 / pose-free 人物）打包成可复核、可追溯的参考集。它不生成 3D，也不假装所有图都能直接喂给任何模型。

### 2.3 不是什么

- 不是 3D 生成器；
- 不是通用照片清理；
- 不是 web-scale 数据集去重；
- 不是“任意图 → 3D”的魔法；
- 不是云端 DAM。

### 2.4 三个模式

| 模式 | 输入 | 输出 | 主要用户 |
|---|---|---|---|
| **training** | 图片 + caption | curated/ + report + per-form cap | LoRA/扩散训练者 |
| **reference** | 图片 | forms/ + canonical + contact sheet + taxonomy | 画师/素材库 |
| **3d** | 图片 + 目标后端 | references/ + transforms.json + readiness.md | 3D/UE5 管线 |

### 2.5 目标用户优先级

1. **AI 画师 / 独立游戏开发者**：从搜索图到 UE5/3D 资产，痛点是“输入整理”。
2. **UE5 小团队**：需要 contract-compliant 参考集和 handoff 清单。
3. **3D 工具用户**：已有 Pixal3D/LocalMesh/Meshy/Tripo，缺的是“喂什么图”。
4. **LoRA 训练者**：training 模式仍是社区入口。
5. **商品/电商**：物体多视角，后续扩展。

---

## 3. 核心抽象：Backend Contract

每个 3D 后端不是一个模型名，而是一份 **输入契约**：

| 后端 | 输入类型 | 数量 | 相机位姿 | 姿势一致性 | 背景 | 输出 | 8GB | UE5 |
|---|---|---|---|---|---|---|---|---|
| **Pixal3D MV** | 多视图 | 2–4 | 必需 `transforms.json` | 同姿势/同版本 | alpha mask | GLB+PBR | 边界 | ✓ 需减面 |
| **Hunyuan3D-2mv** | 标准视图 | 1–4 | 固定 front/back/left/right | 标准视图 | 去背 | GLB/OBJ；纹理另跑 | 形状 ✓ | ✓ 需纹理 |
| **LocalMesh** | 物体照片 | 1 或 4 | DA3 估计 | 同一物体 | 去背 | GLB+texture | ✓ | ✓ |
| **UniForge3D** | unposed 物体照片 | 2–4 | 自动 | 同一物体 | 去背 | GLB+PBR | ✓ | ✓ |
| **LHM++** | pose-free 人物 | 1–16 | 不需要 | 同一身份 | 去背 | 3DGS avatar | ✓ | ✗ 需转 mesh |
| **Meshy/Tripo/Rodin** | 一致多视图 | 1–4 | 标准视图 | 同姿势 | 去背 | GLB+rig | 云端 | ✓ |
| **Single-image** | 单图 | 1 | 无 | 无 | 去背 | GLB/PBR/3DGS | 视模型 | 视模型 |

**Compiler 流程**：

```text
TargetSpec
  → Identity Filter
  → Form/View Clustering
  → Contract Check
  → Select observed images
  → (optional) Generate missing views
  → Validate contract
  → Export package + readiness report
```

**关键**：不是“选完图再想模型”，而是“先选后端契约，再编译输入”。

---

### 3.5 三层结构：Character → Version → Form

3D 建模的聚合单位不是姿势，而是**装扮/版本**：

- **Character**：同一角色；
- **Version**：同一套装扮/发型/配饰；
- **Form**：同一 Version 内的姿势/视角。

规则：

- 同一 Version 的不同 Form → **同一个 3D 模型的参考**；
- 不同 Version → **分开建模，每个 Version 一个模型**；
- Form Library 先按 Version 分组，再展示 Version 内的形态。

Form Library 每项包含：

- Version ID / canonical reference；
- Form ID / canonical image；
- members / cluster size；
- quality / completeness；
- pose/view label；
- background removed；
- 3D readiness per backend；
- `recommended_for_3d` 标记。

3D 输出模式：

| 模式 | 含义 | 当前可行性 |
|---|---|---|
| **version-combined** | 同一 Version 的所有有效 Form → 一个 3D 模型 | **v0.2/v0.3 目标**；texture-combined + PATT |
| **per-form** | 每个 Form 单独一个 3D 模型 | 辅助/调试；不是主目标 |

CLI 草案：

```bash
form-coreset extract ./downloads --out ./library
form-coreset versions ./library --review
form-coreset select ./library --version version_001 --out ./selected.json
form-coreset generate3d ./library --selection ./selected.json \
  --mode version-combined --backend character-gen
```

输出：

```text
run/
  library/
    version_001/
      forms/
      version.json
    version_002/
  selected.json
  3d/
    version_001/model.glb
    version_002/model.glb
  report.md
```

**原则**：先按 Version 分组，再在 Version 内用所有姿势/角度；不同 Version 不混进同一个模型。

## 4. 原理升级

### 4.1 现有原理保留

- **Subject × Form × Variant**：主体、形态、变体三层。
- **Set-level selection**：覆盖 + 质量 - 冗余。
- **Form-caption 联合冗余**：训练模式继续使用。
- **非破坏性 + manifest**。

### 4.2 新增原理

1. **Contract compliance**：目标不是“图好看”，而是“满足后端输入契约”。
2. **Observed vs generated**：真实观测图与生成补视图分开标记；默认 observed-first。
3. **Readiness score**：身份、视角/姿势覆盖、完整度、背景、相机、光照、分辨率、契约匹配。
4. **Camera/view estimation**：DA3-SMALL/BASE 估相机；RTMPose/HumanArt + 剪影估视角/姿势。
5. **Provenance**：每张图记录来源、观测/生成、模型、seed、许可证。
6. **Gap report**：缺什么视角/姿势，推荐补图或生成，而不是硬凑。

### 4.3 3D 模式的选图目标

```text
maximize  contract compliance
        + view/pose coverage
        + quality/completeness
        - redundancy
subject to backend count limit
           same identity (and, for multi-view, same pose/version)
```

- 训练模式：多样性优先；
- 3D 多视图模式：一致性 + 覆盖；
- 3D 多参考模式：身份 + 每张新增信息；
- 生成模式：只补缺口，不替代观测。

---

## 5. 开发路线

### v0.2：Contract + Readiness（不接 3D 模型）

- `BackendContract` 数据模型；
- `ReadinessReport`：身份、视角/姿势、完整度、背景、相机、契约匹配；
- `CameraEstimator`：DA3-SMALL/BASE（Apache-2.0）；
- `ViewEstimator`：RTMPose/HumanArt + 剪影；
- `--preset 3d --backend pixal3d-mv|hunyuan3d-2mv|localmesh|uniforge3d|lhmpp`；
- 只做 observed-only 选择，不生成；
- 输出 `references/` + `contract.json` + `readiness.md`。

**验收**：给 100 张图，能明确说“这个后端能不能用、缺什么、选哪几张”。

### v0.3：Pixal3D MV / LocalMesh 适配

- 生成 `transforms.json`（Pixal3D 格式：Z-up、camera-to-world、第一帧正面）；
- 导出 LocalMesh 的 1/4 视图包；
- 调用后端，输出 GLB；
- UE5 导入清单（减面、缩放、材质、Y-up→Z-up）。

**验收**：8GB 上跑通 2–4 视图，输出 UE5 可导入的 GLB。

### v0.4：UniForge3D / LHM++ 适配

- UniForge3D：2–4 张 unposed 物体图；
- LHM++：1–16 张 pose-free 人物图 → 3DGS avatar；
- 明确标注输出类型：GLB/PBR vs 3DGS/avatar；
- 可选 turnaround 生成（RCM/Zero123++/SV3D），只补缺口，标记 generated。

### v0.5：UE5 Handoff + Benchmark

- 集成/参考 b2ue、reference-asset-compiler、VERA、UniRig/RigAnything；
- 输出 `ue5_import.md`、减面/材质/绑骨步骤；
- 发布 **input ablation benchmark**：
  - random vs top-quality vs form-diverse vs contract-compiled；
  - 比较 3D 几何/纹理/UE5 导入成功率；
  - 这是最有说服力的护城河。

---

## 6. 护城河

1. **Contract 库**：每个 3D 后端的输入契约、数量、相机、许可、VRAM、UE5 适配。
2. **Input ablation benchmark**：证明“输入整理”能提升 3D 质量；别人只有模型，没有数据侧证据。
3. **Provenance**：observed/generated 标记；可追溯、可复核、可复现。
4. **UE5 handoff**：从 GLB 到可导入资产的清单/脚本。
5. **社区与集成**：ComfyUI、Blender、UE5、Kohya、LocalMesh/Pixal3D。
6. **不是模型**：模型会变，输入契约和评估不会。

---

## 7. 竞争与风险

### 竞争

- **cull / imgraft / PixLint**：数据整理，但没有 3D 契约；可以互补。
- **LocalMesh / UniForge3D / Pixal3D**：3D 后端，不是输入编译器；可以集成。
- **reference-asset-compiler / b2ue / VERA**：UE5 handoff，不是多图输入选择；可以集成。
- **Meshy / Tripo / Rodin**：端到端 hosted；它们可能内置输入整理，但不会为本地开源后端做契约层。

### 风险

- **许可**：Hunyuan3D-2mv、DINOv3、SAM 3、VGGT、RCM；默认只用 Apache/MIT。
- **8GB 显存**：Pixal3D 边界；Hunyuan 纹理重；LocalMesh/UniForge3D 更稳。
- **生成补视图**：身份漂移、脑补细节；默认 observed-first。
- **UE5 质量**：AI mesh 需重拓扑/绑骨；不要承诺 production-ready。
- **平台集成**：3D 平台可能自己做输入整理；护城河是 benchmark/contract/社区。
- **Benchmark 成本**：需要 GPU 时间与人工评估。

---

## 8. 需要拍板

1. **主定位是否切到“3D-ready 输入编译器”**，training 降为第二模式？
2. **第一个 3D 后端**：Pixal3D MV（PBR GLB）还是 LocalMesh（8GB 更稳）？
3. **是否坚持 observed-first**，生成补视图只作为 P2？
4. **UE5 handoff 范围**：只输出 GLB + 清单，还是集成 b2ue/UniRig 做自动化？
5. **Benchmark 预算**：是否投入 1–2 周做 input ablation？

---

## 9. 一句话总结

> 3D 生成器会越来越多，但“从一堆网络图到合格输入”的编译器不会自动出现。  
> **FormCoreset 的机会不是再造一个 3D 模型，而是成为 3D 管线的数据入口。**

---

## 10. 参考来源（2026-10）

### 数据整理 / coreset
- [imgraft（PyPI）](https://pypi.org/project/imgraft/)
- [cull（GitHub）](https://github.com/tlennon-ie/cull)
- [PixLint（PyPI）](https://pypi.org/project/pixlint/1.1.0/)
- [NVIDIA NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)
- [FiftyOne](https://docs.voxel51.com/)
- [fastdup](https://github.com/visual-layer/fastdup)

### 相机 / 姿态 / 分割
- [Depth Anything 3（GitHub）](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [Depth Anything 3 模型卡 / 许可](https://huggingface.co/depth-anything/DA3-BASE)
- [VGGT（GitHub）](https://github.com/facebookresearch/vggt)
- [VGGT 许可说明](https://hysenlabs.com/projects/facebookresearch-vggt)
- [RTMPose / MMPose](https://github.com/open-mmlab/mmpose)
- [UniRig](https://github.com/VAST-AI-Research/UniRig) · [RigAnything](https://huggingface.co/Isabellaliu/RigAnything)

### 3D 后端 / 输入
- [Pixal3D（GitHub）](https://github.com/TencentARC/Pixal3D) · [模型卡](https://huggingface.co/TencentARC/Pixal3D)
- [LocalMesh Engine](https://github.com/Quentincls/localmesh-engine)
- [UniForge3D-Low-VRAM](https://github.com/HardeNNN/UniForge3D-Low-VRAM)
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) · [2mv 许可](https://huggingface.co/tencent/Hunyuan3D-2mv/blob/main/LICENSE)
- [LHM++](https://github.com/aigc3d/LHM-plusplus)
- [RCM: Rotate Your Character](https://rotateyourcharacter.github.io/)
- [RnG](https://github.com/XiangMochu/RnG)

### UE5 / 后处理
- [b2ue（Blender → UE5）](https://extensions.blender.org/approval-queue/b2ue/)
- [reference-asset-compiler](https://github.com/raydeStar/reference-asset-compiler)
- [VERA（UE5 AI agent）](https://github.com/ezesubu/VERA)
- [AI 3D Character Pipeline 2026](https://www.top3d.ai/zh/learn/ai-3d-character-pipeline-2026)

### 许可
- [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
- [DINOv2 / 模型卡](https://huggingface.co/facebook/dinov2-base)
- [SAM 3 / SAM License 说明](https://github.com/facebookresearch/sam3)
- [腾讯 Hunyuan3D 2.0 Community License](https://huggingface.co/tencent/Hunyuan3D-2mv/blob/main/LICENSE)
