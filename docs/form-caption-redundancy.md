# Form-Caption 联合冗余度量

> 实现：[`src/formcoreset/redundancy.py`](../src/formcoreset/redundancy.py)  
> 测试：[`tests/test_redundancy.py`](../tests/test_redundancy.py)  
> 关联：[PRD FR-10](PRD.md) · [技术架构 5.5](architecture.md)

## 1. 为什么需要联合度量

训练集里重复的不是“图片”，而是“图文对”：

| 情况 | 图像轴 | Caption 轴 | 应该怎么处理 |
|---|---|---|---|
| 同形态 + 同 caption | 重复 | 重复 | 高度冗余，删 |
| 同形态 + 不同 caption | 重复 | 新信息 | 视策略保留少量，避免视觉过拟合 |
| 不同形态 + 同 caption | 新信息 | 重复 | 视策略保留，避免语义过拟合 |
| 不同形态 + 不同 caption | 新信息 | 新信息 | 保留 |

只看图像 embedding 会误删“同形态不同 caption”；只看 caption 会误删“同 caption 不同形态”。联合度量把两个轴分开，再组合。

## 2. 定义

对两个图文对 \(i, j\)：

- \(f_i, c_i\)：L2 归一化的形态 embedding / caption embedding；
- \(s^f_{ij} = \mathrm{clamp}(\cos(f_i, f_j), 0, 1)\)；
- \(s^c_{ij} = \mathrm{clamp}(\cos(c_i, c_j), 0, 1)\)。

阈值化：

\[
\tilde{s}^f_{ij} = \mathrm{clip}\left(\frac{s^f_{ij} - \tau_f}{1 - \tau_f}, 0, 1\right)
\]

\[
\tilde{s}^c_{ij} = \mathrm{clip}\left(\frac{s^c_{ij} - \tau_c}{1 - \tau_c}, 0, 1\right)
\]

联合冗余：

\[
R^{\text{form}}_{ij} = \tilde{s}^f_{ij} \cdot \left(\gamma_f + (1-\gamma_f)\tilde{s}^c_{ij}\right)
\]

\[
R^{\text{caption}}_{ij} = \tilde{s}^c_{ij} \cdot \left(\gamma_c + (1-\gamma_c)\tilde{s}^f_{ij}\right)
\]

\[
R_{ij} = \max\left(R^{\text{form}}_{ij}, R^{\text{caption}}_{ij}\right)
\]

### 参数解释

| 参数 | 默认 | 作用 |
|---|---|---|
| \(\tau_f\) | 0.80 | 形态相似度阈值；低于它视为不重复 |
| \(\tau_c\) | 0.75 | caption 相似度阈值 |
| \(\gamma_f\) | 0.30 | 同形态但 caption 不同时保留的“形态冗余底” |
| \(\gamma_c\) | 0.10 | 同 caption 但形态不同时保留的“语义冗余底” |
| \(k\) | 1 | 相对集合计算时取 top-k 个最冗余邻居 |

**行为**：

- \(\gamma_f=0, \gamma_c=0\)：纯 AND，两个轴都重复才算冗余；
- \(\gamma_f=1\)：退化为 form-only，caption 不影响；
- \(\gamma_c=1\)：退化为 caption-only；
- 没有 caption 时：\(R_{ij} = \tilde{s}^f_{ij}\)（form-only 退化）；
- 默认 \(\gamma_f=0.30, \gamma_c=0.10\)：形态重复是主要风险，caption 重复是次要风险。

### 为什么用 max 而不是平均

`max` 表示“任意一个轴达到高冗余，就认为这个 pair 有冗余”，同时通过两个 floor 保留另一轴的部分影响。平均会把“形态极像 + caption 完全不同”稀释成中等冗余，不符合训练场景的直觉。

## 3. 相对已选集合

选择训练集时，关键不是两两冗余，而是“这个 pair 相对于已选集合是否还有新信息”。

- **边际冗余**：

\[
R_{i|S} = \frac{1}{\min(k,|S|)}\sum_{j \in \text{top-}k(S)} R_{ij}
\]

- **形态新颖度**：\(N^f_{i|S} = 1 - \max_{j\in S} s^f_{ij}\)
- **caption 新颖度**：\(N^c_{i|S} = 1 - \max_{j\in S} s^c_{ij}\)
- **联合新颖度**：\(N_{i|S} = w_f N^f + w_c N^c\)（默认 0.5/0.5）
- **边际价值**：

\[
V_{i|S} = q_i + N_{i|S} - \lambda R_{i|S}
\]

其中 \(q_i\) 是单图质量分，\(\lambda\) 默认 0.70。

## 4. 贪心选择

`greedy_select` 的流程：

1. 从空集合开始；
2. 每一步在未选候选里找边际价值最高的；
3. 跳过已达到“每形态上限 K”的候选；
4. 达到 `max_items`，或最优分数 ≤ 0 时停止。

它是实用基线，不是带理论保证的子模优化器；后续可以把评分函数换成 facility location / graph cut 的受约束贪心。复杂度 \(O(n^2)\)，100–10,000 张图可用；更大规模需要 FAISS/kNN 近似。

## 5. Caption 相似度怎么算

度量本身只接收相似度，不绑定模型。推荐的三种后端：

| Caption 类型 | 方法 | 备注 |
|---|---|---|
| 自然语言 | sentence-transformers MiniLM / CLIP text | 余弦相似度 |
| Danbooru 标签 | tag Jaccard | 去掉 `1girl`、`solo` 等高频标签，否则所有图都“冗余” |
| 混合 | `0.5 * embedding + 0.5 * tag` | 默认 `hybrid_caption_similarity` |

形态相似度来自：

- `person` adapter：RTMPose + HumanArt 关键点特征 + DINOv2 masked embedding；
- `object` / `auto` adapter：DINOv2 masked embedding；
- 必须先去背景/取主体 crop，否则背景会主导形态相似度。

## 6. 用法

```python
import numpy as np

from formcoreset.redundancy import (
    JointRedundancyConfig,
    greedy_select,
    marginal_information,
    pair_redundancy,
)

config = JointRedundancyConfig()

# 两个 pair：同形态、caption 不同
breakdown = pair_redundancy(form_similarity=0.95, caption_similarity=0.2, config=config)
print(breakdown.joint_redundancy, breakdown.dominant)  # ~0.3, "form"

# 相对已选集合
form = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
caption = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

info = marginal_information(form, caption, candidate_index=2, selected_indices=[0], quality_score=0.9)
print(info.form_novelty, info.caption_novelty, info.score)

# 每形态最多 2 个
selected = greedy_select(
    form,
    caption,
    quality_scores=[0.9, 0.8, 0.85],
    max_items=3,
    form_ids=[0, 0, 1],
    per_form_cap=2,
    config=config,
)
print(selected)
```

## 7. 如何校准

默认阈值/floor 只是起点，不是普适常数。

1. 标注一批 pair：`same_form`、`same_caption`、`redundant`。
2. 在验证集上扫 \(\tau_f, \tau_c, \gamma_f, \gamma_c\)，优化 pairwise precision/recall 或 AUC。
3. 用选择结果评估：`redundancy@K`、`coverage@K`、稀有形态保留率。
4. 最终用下游 LoRA 对比：全量 vs 筛选后，比较验证 loss、prompt adherence、多样性、人工偏好。
5. 把选定的参数写入 manifest，保证可复现。

## 8. 局限

- **相似度依赖模型**：不同 form/caption encoder 的余弦分布不同，阈值必须重新校准。
- **Caption 噪声**：自动 caption 的错误会传导到冗余判断；caption 质量差时建议降低 \(w_c\) 或关闭 caption 轴。
- **背景泄漏**：不做主体 mask，形态相似度会被背景支配，导致“同形态不同背景”被误判为新形态。
- **贪心非最优**：`greedy_select` 没有子模保证；大规模/严格预算场景应换 facility location / graph cut。
- **只解决冗余型过拟合**：它不能解决学习率、epoch、caption 质量、数据分布偏差等其他过拟合来源。
- **Tag Jaccard 的高频词问题**：必须使用 stoplist，或用 IDF 加权。

## 9. 与整体管线的关系

```text
FormEncoder ──► form embeddings ─┐
                                 ├─► joint redundancy ──► training selector ──► curated/
Caption encoder ─► caption emb ──┘
                                 └─► manifest 记录配置/版本
```

- Reference mode 仍用 canonical 评分（质量/完整度/代表性）；
- Training mode 用本度量做“每形态上限 + caption 去重 + 稀有形态保护”；
- 度量的参数、模型版本、阈值全部写入 `manifest.jsonl` / `run.json`。
