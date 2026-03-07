# 视频 Diffusion 统一注意力加速框架

> 研究方向启动：2026-03-07

## 一句话目标

设计一个 **IO-aware、视频结构感知** 的统一注意力算法，同时利用视频 3D 注意力的 5 个已知结构性质，实现理论可证明的加速。

## 为什么这个方向

### 核心观察

视频 DiT 中注意力占 **60-90% 计算量**（Wan 76%+，HunyuanVideo 80%+，Mochi 60%）。现有加速方法各自利用了 1-2 个结构性质，但没有统一框架：

| 结构性质 | 量化证据 | 已有方法 | 利用深度 |
|---------|---------|---------|---------|
| ① 块对角线主导（空间局部性） | 对角块权重 2.8x > 非对角 | STA | ✓ 部分 |
| ② 时间局部性衰减 | 15% token 覆盖 70% 注意力 | STA | ✓ 部分 |
| ③ 跨 prompt 模式不变性 | >90% 索引重叠 | STA 校准 | △ 浅层 |
| ④ 去噪步间稳定性（U 形） | 中间 70% 步骤几乎不变 | PAB | △ 仅分解式模型 |
| ⑤ Head 功能特化 | 空间/时间/全局/Sink 头 | SVG | △ 仅二分类 |
| ⑥ 高秩结构化稀疏 | Monarch 分解可捕获 | VMonarch | △ 需微调 |
| ⑦ 能量指数衰减 | R² > 0.985 | Radial Attn | △ 理论阶段 |

### 理论 gap

- **当前最优 kernel**：FPSAttention 7.09x，STA 10.45x
- **理论上界**：O(n²) → O(n log n) 的 gap 意味着可能存在 **n/polylog(n) 倍** 的改进空间
- **IO 下界**：Red-Blue Pebble Game 给出 Ω(nnz · d² / M)，自适应块大小可达到此下界

### 5 个未被统一的维度

```
维度 1: 3D 感知分块          — STA 做了，但固定立方体
维度 2: 每头异构调度          — SVG 做了二分类，不够细
维度 3: 跨步增量计算 ★空白    — 只有输出级缓存，没有注意力矩阵级增量
维度 4: 量化协同              — FPSAttention 做了 FP8+稀疏
维度 5: 离线编译  ★空白       — 模式>90%输入无关，可预编译硬件专用 kernel
```

## 竞争格局

| 方法 | 团队 | 发表 | Kernel 加速 | E2E 加速 |
|------|------|------|-----------|---------|
| STA | Hao AI Lab (UC San Diego) | ICML 2025 | 10.45x | ~2.5x |
| FPSAttention | 清华+NUS | NeurIPS 2025 Spotlight | 7.09x | 4.96x |
| VMonarch | Stanford (Tri Dao 组) | arXiv 2601 | 5x+ | — |
| Radial Attention | — | NeurIPS 2025 | 9x (理论) | — |
| Compact Attention | — | arXiv 2508 | 2.5x | — |
| SVG | Peking U | ICML 2025 | — | 2.33x |

## 研究路径

```
Phase 0: 注意力矩阵采集与分析 ✅ 完成
├── 7 个性质验证: 5/6 通过
├── 关键发现: 块对角比 9x, top 1% 覆盖 57.8%, U 形不成立
└── 决策: GO

Phase 1: 注意力跳过与稀疏复用可行性 ✅ 完成 (CONDITIONAL GO)
├── 1a: 逐步复用容忍度 ✅ 12.7% 安全, 后期步+首尾层最稳定
├── 1b: 实际跳过 + 视频质量 ❌ SSIM < 0.55, 全层跳步不可行
├── 1c: 逐层敏感度 ❌ 0/30 安全层, 首尾层最敏感 (反直觉)
├── 1d: 稀疏掩码跨步复用 ✅ IoU=0.82-0.88, mask 跨步稳定
└── 决策: 放弃步间输出复用, 聚焦稀疏 mask 复用

Phase 2: 稀疏注意力 + Mask 复用验证 ⬜ 下一步
├── 2a: top-k% 稀疏注意力质量验证
├── 2b: mask 跨步复用 + 稀疏计算
└── 2c: 数据驱动 vs STA 固定模式对比

Phase 3: Triton Kernel 实现
├── 4 种单头 kernel + 调度器
├── 步级跳过集成
└── 多模型泛化测试 + 开源 + 论文
```

## 文件结构

```
research/video-attention/
├── 00-overview.md          ← 本文件：方向总览
├── 01-structural-properties.md  ← 7 个结构性质的详细证据
├── 02-existing-methods.md       ← 现有方法的详细分析
├── 03-problem-formulation.md    ← 统一数学框架 ★核心
├── 04-theoretical-analysis.md   ← IO complexity 分析
└── 05-experiment-plan.md        ← 实验验证计划
```

## 关键参考文献

### 结构性质

- [Efficient-vDiT (arXiv:2502.06155)](https://arxiv.org/abs/2502.06155) — 块对角线主导、跨 prompt 不变性
- [Analysis of Attention in VDiTs (arXiv:2504.10317)](https://arxiv.org/abs/2504.10317) — 层敏感性、sink 头、稀疏度容忍度
- [Understanding Attention in Video Diffusion (arXiv:2504.12027)](https://arxiv.org/abs/2504.12027) — 熵分布分析
- [PAB (arXiv:2408.12588)](https://arxiv.org/abs/2408.12588) — U 形时间稳定性
- [Enhance-A-Video (arXiv:2502.07508)](https://arxiv.org/abs/2502.07508) — 非对角线权重的语义含义

### 现有方法

- [STA (arXiv:2502.04507)](https://arxiv.org/abs/2502.04507) — 瓦片滑窗，ICML 2025
- [FPSAttention (arXiv:2506.04648)](https://arxiv.org/abs/2506.04648) — FP8+稀疏，NeurIPS 2025 Spotlight
- [VMonarch (arXiv:2601.22275)](https://arxiv.org/abs/2601.22275) — Monarch 矩阵分解
- [Radial Attention (arXiv:2506.19852)](https://arxiv.org/abs/2506.19852) — O(n log n) 理论，NeurIPS 2025
- [Sparse VideoGen (arXiv:2502.01776)](https://arxiv.org/abs/2502.01776) — 在线 profiling，ICML 2025
- [Compact Attention (arXiv:2508.12969)](https://arxiv.org/abs/2508.12969) — 多模式分类
- [LiteAttention (arXiv:2511.11062)](https://arxiv.org/abs/2511.11062) — 跨步稀疏复用

### 理论基础

- [FlashAttention (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135) — IO-aware 设计范式
- [FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — 并行化优化
- [FlashAttention-3 (arXiv:2407.08608)](https://arxiv.org/abs/2407.08608) — Hopper 硬件特化
- [IO Complexity of Attention (arXiv:2402.07443)](https://arxiv.org/abs/2402.07443) — 理论下界，ICML 2024
- [Red-Blue Pebble Game (Hong & Kung, 1981)](https://dl.acm.org/doi/10.1145/800076.802486) — IO 复杂度框架
