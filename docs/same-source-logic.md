# 同源变体去重：逻辑与原理

> 目标：同一张海报/原图的变体（裁剪、缩放、换背景、特效、压缩、亮度等）归为一组，每组选一个 canonical；必要时把互补裁剪拼成更完整的一张。无需姿势标签。

## 1. 关系定义

| 关系 | 含义 | 处理 |
|---|---|---|
| **完全重复** | 同一文件/像素一致 | 内容哈希 |
| **同源变体** | 同一张原图派生：裁剪、缩放、换背景、特效、压缩、亮度 | 局部特征配准 + 单应性 |
| **同角色不同绘制** | 不同画师/不同原图，动作可能相同 | 不属于 `dedupe`；交给后续形态聚类/人工 |
| **不同角色/不同原图** | 不应合并 | 拒绝 |

**同源关系应当近似等价关系**：自反、对称、传递。实际匹配有噪声，直接做连通分量会把噪声边传递成错误的大组。这是下面分组算法的核心动机。

## 2. 为什么“连通分量”会过度合并

设 A、B、C 是三张不同原图，但 A~B、B~C 各自有中等匹配，A 与 C 匹配很弱：

```text
A —— B —— C
 \________/
   弱/不匹配
```

连通分量会把 A、B、C 合成一个组。实际数据中，相似角色、相同构图、重复背景都会制造这种“桥边”。

**修正原则**：同源组内应尽量满足“任意两张都能直接匹配”。因此默认使用 **complete-linkage**，而不是连通分量。

## 3. 单对证据

对两张图 \(i,j\)：

1. **Masked SIFT**
   - 用 isnet-anime 得到角色 mask，只在 mask 内检测 SIFT；
   - ratio test + mutual nearest neighbor（后续可加）；
   - RANSAC 估计单应性 \(H_{ij}\)；
   - inliers \(I_{ij}\)，关键点数 \(K_i,K_j\)；

\[
s_{ij} = \frac{I_{ij}}{\min(|K_i|,|K_j|)}
\]

2. **Aligned-edge 验证**
   - 用 \(H_{ij}\) 把图 i warp 到图 j；
   - 在重叠区域 / 角色 mask 内计算 Sobel 边缘图；
   - 相关系数 \(c_{ij}\)。

3. **联合置信度**

\[
\rho_{ij} =
\begin{cases}
\min(s_{ij}, c_{ij}), & \text{verify enabled and } c_{ij} \ge \tau_c \\
s_{ij}, & \text{verify disabled} \\
0, & \text{otherwise}
\end{cases}
\]

- 只有 \(s_{ij}\ge\tau_s\) 且 \(I_{ij}\ge I_{\min}\) 才进入验证；
- \(c_{ij}<\tau_c\) 的边直接丢弃；
- 这样做的直觉：**SIFT 只能证明“局部特征能对上”，edge correlation 证明“整块角色边缘真的对齐”**。相似角色/重复纹理往往能过 SIFT，但过不了全局边缘相关。

当前默认：\(\tau_s=0.30\)、\(I_{\min}=12\)、\(\tau_c=0.35\)。

## 4. 分组

构造置信度矩阵 \(R\)，\(\rho_{ij}\in[0,1]\)，非边为 0。

### 4.1 Single linkage（连通分量）

边 \(\rho_{ij}\ge\tau\) 直接 union。

- 优点：recall 最高；
- 缺点：链式合并（上面的 A-B-C）；
- 适用：数据很干净、想尽量少漏合并。

### 4.2 Complete linkage（默认）

距离 \(d_{ij}=1-\rho_{ij}\)，做层次聚类 complete linkage，在 \(1-\tau\) 处切树。

- 合并两个簇的条件：**所有跨簇 pair 的 \(\rho\) 都 ≥ \(\tau\)**；
- 优点：precision-first，杜绝链式假合并；
- 缺点：某个变体与组内其他成员都弱匹配时，会被保守拆成 singleton；
- 适用：默认；宁可少合并，也不要错误合并。

### 4.3 Average linkage

距离同上，average linkage。

- 介于 single 与 complete 之间；
- 适合变体质量差异较大、但整体仍同源的域。

### 4.4 Star merge（默认后处理）

Complete linkage 是 precision-first，可能把某个弱变体拆成 singleton。Star merge：

1. 用 complete linkage 生成种子组；
2. 对每个组计算 **medoid**（组内平均置信度最高的成员）；
3. 对每个 singleton，若它与某个组 medoid 的置信度 ≥ `star_threshold`，就并入该组；
4. **只允许 singleton → group，不允许 group → group**，因此不会产生跨组合并链；
5. medoid 在种子组上固定，不随合并重算（一轮）。

默认 `star_threshold=0.20`。在合成基准上，complete 单独 recall 0.935；加 star merge 后恢复到 **1.000**，precision 仍为 1.000。

在 Jett Bing 样本上：

- complete 单独：22 个 singleton，移除 4 个变体；
- complete + star merge 0.20：22 组，移除 8 个变体；A 完整，B 到 5/6；
- 降到 0.18 会把 3D 渲染 002 误并入 B，因此默认保持 0.20。

**建议**：默认 complete + star merge；如果真实数据上仍有大量 singleton，先检查是否真的是不同原图，再考虑 `--linkage average` 或按域调 `star_threshold`。

#### 双轴 star merge 的评估结论（暂不采用）

曾提出：不只看 `min(score,corr)`，而是 `score ≥ 0.15 且 corr ≥ 0.80`。评估结果：

| 数据集 | current | replacement | union |
|---|---|---|---|
| 合成基准 | 8 组 / P=R=F1=1.0 | 8 组 / 1.0 | 8 组 / 1.0 |
| Jett 样本 | 22 组 / P=1.0 R=0.75 | 24 组 / P=1.0 R=0.45 | 22 组 / P=1.0 R=0.75 |

结论：

- 合成基准没有收益；
- Jett 上 replacement 更差，union 与 current 完全相同；
- 015 的高相关边连的是组员 010，不是 medoid，medoid-only 双轴规则抓不到；
- 因此不采用，保持 `min(score,corr) ≥ 0.20`。

**原则**：不为了单个 case 改默认规则。若要继续优化，应评估“member-level high-corr merge”（任意组员）或由审核 UI 人工合并，而不是继续调 medoid 阈值。

## 5. Canonical 选择

每组先选 **single canonical**：

```text
score = 分辨率/有效像素 + 清晰度 + 完整度 + 背景干净度 - flags
```

- 选主体像素最多、裁剪最少、压缩损伤最小的一张；
- 不删除其他 variants；
- 输出 `canonical_single.*`。

## 6. Canonical 拼接

互补裁剪可能各自保留不同区域。拼接逻辑：

1. 以 single canonical 为参考帧；
2. 对每个成员估计 \(H_{i\to r}\)；
3. 用所有变换后图像的角点求 union canvas（允许超出参考图边界）；
4. 把每张图 warp 到 canvas；
5. 权重：
   - 角色 mask（若开启）；
   - 有效像素；
   - 距离变换 feather；
   - 匹配置信度；
6. 加权平均，得到 RGB + alpha；
7. 裁剪到内容 bbox；
8. 覆盖面积增益：

\[
g = \frac{A_{\text{stitched}}}{A_{\text{reference}}} - 1
\]

- 若 \(g \ge \gamma\)（默认 0.02）且至少 2 张参与，输出 `canonical.png`（拼接版）；
- 始终保留 `canonical_single.*`；
- 否则回退 single canonical；
- `source.json` 记录 `stitched`、`stitch_gain`、`stitch_members`。

**限制**：拼接受单应性精度、曝光/色彩差异、特效遮挡影响。默认保留 single 版本，拼接失败不破坏结果。

## 7. 阈值与校准

当前默认：

| 参数 | 默认 | 作用 |
|---|---:|---|
| `min_score` | 0.30 | pair 置信度阈值 |
| `min_inliers` | 12 | RANSAC 最少 inliers |
| `min_correlation` | 0.35 | aligned-edge 相关性阈值 |
| `linkage` | complete | 分组策略 |
| `star_merge` | true | singleton → medoid 后处理 |
| `star_threshold` | 0.20 | star merge 的 medoid 置信度阈值 |
| `stitch_min_gain` | 0.02 | 拼接增益阈值 |

**校准原则**：

1. 构造带标签的 pair 集（同源/不同源）；
2. 以 **precision 优先**：在 precision ≥ 0.95 的约束下最大化 recall；
3. 分域校准：线稿、厚涂、强特效、照片的最优阈值不同；
4. 真实数据先跑 `dedupe`，人工抽查 false merge / false split，再调阈值；
5. 记录参数到 manifest，保证可复现。

## 8. 失败模式与对策

| 失败模式 | 表现 | 对策 |
|---|---|---|
| 链式合并 | A~B、B~C 合成一个组 | complete linkage |
| 相似角色同构图 | SIFT 能对上脸/头发 | mask + edge verification |
| 强特效/换背景 | 特征被背景主导 | masked SIFT；必要时扩大 mask / 用边缘 |
| 线稿/低纹理 | SIFT 特征少，singleton | 后续 LightGlue / 线稿专用匹配；人工合并 |
| 无重叠裁剪 | 两张图没有共同区域 | 无法配准；保留 singleton |
| 镜像/翻转 | SIFT 可能错配 | 后续增加 mirror 候选 |
| 文字/水印 | mask 外文字被忽略 | OCR/水印检测 + uncertain |
| 拼接 ghosting | 对齐误差/曝光差异 | 保留 single；后续 color compensation / multi-band |
| 大图性能 | O(n²) pair 匹配 | 先 dHash/embedding 预筛；FAISS |

## 9. 评估

### 合成同源基准

8 原图 × 7 变体 = 56 张（original / crop / resize / JPEG / padded / brightness / effect）。

| 配置 | Linkage | Groups | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| mask + verify | single | 8 | 1.000 | 1.000 | 1.000 |
| mask + verify | average | 9 | 1.000 | 0.964 | 0.982 |
| mask + verify | complete | 10 | 1.000 | 0.935 | 0.966 |
| **mask + verify** | **complete + star merge** | **8** | **1.000** | **1.000** | **1.000** |

Complete 拆出 2 个弱变体 singleton；star merge 把它们按 medoid 置信度拉回，recall 恢复 1.0，precision 仍为 1.0。默认选择 complete + star merge，兼顾问 precision 与 recall。

### 指标

- pairwise precision / recall / F1；
- group purity；
- source coverage；
- false merge / false split 数量；
- 拼接增益、拼接区域视觉质量；
- 运行时间。

### 真实数据计划

- 收集 100–300 张真实海报/原图变体；
- 人工标注同源组；
- 按域调阈值；
- 加入 hard negatives：相似角色、同构图、不同原图；
- 记录失败案例，更新预设。

## 10. 产品原则

1. **`dedupe` 是第一路径**，`extract`（形态聚类）第二；
2. **无姿势标签**；输出只有 source group、canonical、variants；
3. **非破坏性**：原图不动，variants 保留；
4. **可解释**：source.json 记录 min_score、stitch_gain、参数；后续导出 pair evidence；
5. **人机协同**：complete 的保守 singleton 可由 UI 合并；false merge 可拆分；
6. **可复现**：参数、模型版本、运行配置写入 manifest。

## 11. 伪代码

```text
for each image:
    mask = anime_seg(image) if use_mask
    kp, des = SIFT(gray(image), mask)

for each pair (i, j):
    match = ratio_test + RANSAC(kp_i, des_i, kp_j, des_j)
    if match.score < min_score or match.inliers < min_inliers:
        continue
    if verify:
        corr = edge_correlation(warp(i, H), j, mask_i, mask_j)
        if corr < min_correlation:
            continue
        confidence = min(match.score, corr)
    else:
        confidence = match.score
    R[i, j] = R[j, i] = confidence

groups = hierarchical_cluster(R, linkage=complete, threshold=min_score)
groups = star_merge(groups, R, threshold=star_threshold)
for group in groups:
    canonical_single = best_quality(group)
    stitched = stitch(group, reference=canonical_single)
    if stitched.gain >= stitch_min_gain:
        canonical = stitched
    else:
        canonical = canonical_single
```

## 12. 下一步（逻辑层）

1. **Pair evidence 导出**：`matches.jsonl` 记录每对图的 score/inliers/correlation/decision，方便调阈值。
2. **Review UI**：source groups 合并/拆分、stitched 对比、singleton 拉回。
3. **Star merge 后处理**：complete 拆出的 singleton，若与某组 medoid 直接高置信匹配，可安全合并。
4. **Color compensation**：拼接前在重叠区对齐均值/方差，减少曝光接缝。
5. **Multi-band blending**：减少 ghosting。
6. **Mirror handling**：镜像变体作为候选。
7. **LightGlue/线稿匹配**：低纹理/线稿域。
8. **候选预筛**：dHash/embedding/FAISS，支持更大规模。
