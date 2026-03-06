# AR 会是视频生成模型的未来吗？

> 分析日期：2026-03-06

## 结论：不会完全替代 DiT，混合架构是更可能的未来

---

## 当前格局 (2026.03)

| 阵营 | 代表模型 | 占比 | 质量排名 |
|------|---------|------|---------|
| **双向 DiT** | Sora 2, Veo 3.1, Kling 3.0, Seedance 2.0, Wan 2.6, HunyuanVideo 1.5, Hailuo 02, Vidu Q3 | ~70% | 排行榜前列 |
| **纯 AR** | MAGI-1, Grok Imagine | ~15% | 中上 |
| **混合** | Runway Gen-4.5 (A2D), SkyReels-V2 (Diffusion-Forcing) | ~15% | 中上 |

---

## AR 有结构性优势的场景

| 场景 | 为什么 AR 更优 | DiT 的问题 |
|------|--------------|-----------|
| **流式/实时** | 逐 chunk 输出，首帧延迟 1.3s (CausVid) | 必须全部生成完才能输出（210s+） |
| **交互式应用** | 因果结构天然支持条件生成 | 无法根据中途反馈调整后续帧 |
| **长视频 (>30s)** | 内存 O(n)，天然支持无限长度 | 注意力 O(n²)，内存随长度平方增长 |
| **世界模型/机器人** | 因果推理是物理世界的本质 | 双向"看到未来"是不自然的 |

## DiT 有结构性优势的场景

| 场景 | 为什么 DiT 更优 | AR 的问题 |
|------|--------------|-----------|
| **离线高质量生成** | 双向注意力看全局，质量 SOTA | 误差累积，跨 chunk 质量漂移 |
| **短视频 (<10s)** | 一次生成，全局一致性好 | 自回归开销不值得 |
| **GPU 并行计算** | 充分利用 GPU 并行性 | 本质是顺序生成，GPU 利用率低 |
| **全局编辑/修改** | 可以修改任意位置 | 改前面需要重新生成后面所有内容 |

---

## 与 LLM 的类比：为什么视频不会像语言一样 AR 主导

LLM 中 AR 占绝对主导地位，但视频生成有本质区别：

| 维度 | 语言 | 视频 |
|------|------|------|
| **信息密度** | 每个 token 高信息量 | 帧间 90%+ 冗余 |
| **因果性** | 天然因果（前文→后文） | 既有因果（时间流动）也需全局一致（空间） |
| **序列长度** | 数千 token | 数万-数十万 token |
| **并行需求** | 低（token 少） | 高（像素多，需要 GPU 大规模并行） |
| **生成单元** | 离散 token，自然可自回归 | 连续像素/latent，diffusion 更自然 |

语言的因果本质让 AR 成为自然选择。视频**既需要因果（时间流动）也需要全局（空间一致性）**，纯 AR 不是最优解。

---

## 实际趋势：融合，不是替换

### 混合架构的形态

Runway Gen-4.5 的 **A2D（Autoregressive-to-Diffusion）** 可能预示了方向：

```
纯 DiT ──────────→ 混合架构 ←──────────── 纯 AR
(2022-2025)           (2026+)              (2025-)

混合架构的可能形态：

1. A2D: AR 做语义规划 → DiT 做渲染
   (Runway Gen-4.5)
   优势：AR 理解语义因果，DiT 保证视觉质量

2. Diffusion-Forcing: AR 框架内用 diffusion 去噪
   (SkyReels-V2)
   优势：结合 AR 的长序列能力和 diffusion 的生成质量

3. 因果蒸馏: 双向 DiT teacher → 因果 student
   (CausVid, Self-Forcing)
   优势：保留 DiT 质量，获得 AR 的流式能力

4. 分段混合: 关键帧用 DiT → 中间帧用 AR 插值
   优势：关键帧质量高，中间帧生成快
```

### 演进预测

```
2026: DiT 主导 (70%), AR 新兴 (15%), 混合 (15%)     ← 现在
2027: DiT (50%), 混合 (30%), 纯 AR (20%)            ← 混合快速增长
2028+: 混合 (50%+), DiT (30%), 纯 AR (20%)          ← 混合成为主流
```

---

## 对研究机会的启示

### 1. 面向混合架构的加速 > 纯 AR 加速

混合架构同时包含 AR 和 Diffusion 组件，需要**两套加速方法的协同**。这比纯 AR 加速更有长期价值。

### 2. DiT 加速研究不会过时

即使混合架构成为主流，其中的 Diffusion 渲染组件仍需要 DiT 加速技术（缓存/量化/稀疏注意力等）。我们调研的 6 个方向仍然有价值。

### 3. 最有价值的交叉点

```
高价值研究交叉:
├── AR 部分: KV cache 管理（借鉴 LLM）
├── DiT 部分: 特征缓存/稀疏注意力（现有研究）
├── 交叉点: AR 规划如何指导 DiT 渲染的计算分配？
│           （AR 预测下一帧简单→DiT 用更少计算）
└── 服务系统: 同时优化 AR 和 DiT 两阶段的推理服务
```

---

## 参考

- [Runway Gen-4.5: Autoregressive-to-Diffusion VLMs](https://runwayml.com/research/autoregressive-to-diffusion-vlms)
- [CausVid: From Slow Bidirectional to Fast Autoregressive (CVPR'25)](https://causvid.github.io/)
- [Self-Forcing: Bridging Train-Test Gap (NeurIPS'25 Spotlight)](https://arxiv.org/abs/2506.08009)
- [MAGI-1: Autoregressive Video at Scale](https://arxiv.org/abs/2505.13211)
- [SkyReels-V2: Infinite-length Diffusion-Forcing](https://arxiv.org/abs/2504.13074)
- [Artificial Analysis Video Leaderboard (Feb 2026)](https://artificialanalysis.ai/video/leaderboard/text-to-video)
