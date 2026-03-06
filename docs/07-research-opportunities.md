# 研究机会分析

> 调研日期：2026-03-06
> 空白验证日期：2026-03-06
> 范式级机会补充：2026-03-06

基于对视频模型推理加速 6 大方向的全面调研 + 跨领域机会搜索，识别出以下研究机会。**所有机会均经过文献验证。**

---

## 机会总览

### 第一梯队：新发现的高价值机会

| 编号 | 方向 | 空白度 | 影响力 | 时效性 |
|------|------|--------|--------|--------|
| **A** | 自回归视频加速（流式/交互场景） | ✅ 极高 (2026.02 刚开始) | ★★ 高 (特定场景) | 紧迫 |
| **B** | 测试时计算 × 加速联合优化 | ✅ 极高 (完全空白) | ★★★ 很高 (改变问题定义) | 紧迫 |
| **C** | 潜空间自适应压缩 | ✅ 高 | ★★ 高 (正交乘法器) | 中 |

### 第二梯队：方向级机会（原有 6 个，经验证）

| 编号 | 方向 | 空白度 | 影响力 | 备注 |
|------|------|--------|--------|------|
| **1** | DuCa 随机性理论推广 | ✅ 确认空白 | 高 (理论) | 低工程量，桥接论文 |
| **2** | 缓存×蒸馏协同设计 | ⚠️ 半空白 | 中 | DisCa(2026.02)已占先 |
| **3** | 长视频加速 Scaling 规律 | ✅ 确认空白 | 中 | 实际需求强 |
| **4** | 自适应组合框架 | ✅ 确认空白 | 高 | 工程量大 |
| **5** | 统一 benchmark | ✅ 确认空白 | 中 | 社区价值 |
| **6** | 消费级 GPU 优化 | ⚠️ 半空白 | 低 | 社区已部分解决 |

### 第三梯队：交叉增强机会

| 编号 | 方向 | 说明 |
|------|------|------|
| **X1** | Reward-Guided Caching | 用 reward 模型替代启发式缓存决策，强化 #1/#2 |
| **X2** | Anytime Video / 渐进预览 | PostDiff"降低每步成本>减少步数"视频验证，强化 #4 |
| **X3** | MoE 路由分析 | 分析 dense DiT 的虚拟专家模式，提供理论洞察 |

---

## A. 自回归视频生成加速 ★★ 面向流式/交互场景

### 背景：架构多元化而非范式替换

> **修正**：此前声称"视频生成正在从双向 DiT → 自回归范式迁移"，经验证**这一判断被夸大了**。

**实际情况**：双向 DiT 仍主导生产部署和质量排行（~70% 主要模型）：

```
双向 DiT (仍然主导)              自回归 (新兴替代)               混合架构
┌─────────────────────┐        ┌─────────────────────┐       ┌──────────────┐
│ Sora 2, Veo 3.1     │        │ MAGI-1              │       │ Runway Gen-4.5│
│ Kling 3.0, Seedance │        │ Grok Imagine        │       │ (A2D)        │
│ Wan 2.6, HunyuanVideo│       │                     │       │ SkyReels-V2  │
│ Hailuo 02, Vidu Q3  │        │                     │       │ (Diff-Forcing)│
│                     │        │                     │       │              │
│ ~9 个主要生产模型     │        │ ~2 个               │       │ ~2 个         │
│ 质量排行榜前列       │        │                     │       │              │
└─────────────────────┘        └─────────────────────┘       └──────────────┘
```

**AR 的真实优势局限于特定场景**：
- **流式/实时生成**：CausVid 9.4 FPS 流式，vs 双向 DiT 生成 10s 视频需 210+ 秒
- **交互式应用**：世界模型、机器人、游戏模拟需要因果/序列生成
- **长视频**：AR 天然支持无限长度（SkyReels-V2），双向注意力随长度平方增长
- **但离线质量仍不如 DiT**：Kling 3.0 / Veo 3.1 / Sora 2 全是 DiT，排行榜前列

**因此这个机会的影响力从 ★★★ 降至 ★★ —— 不是"必须追赶的范式迁移"，而是"面向流式/交互/长视频的增量机会"。**

### 已有工作（极少，2026.01-02 才出现）

| 论文 | 时间 | 做了什么 | 局限 |
|------|------|----------|------|
| **FlowCache** | 2026.02 | 首个 AR 视频缓存框架，MAGI-1 2.38x / SkyReels-V2 6.7x | 仅缓存一种方法 |
| **Light Forcing** | 2026.02 | Self-Forcing 的轻量化改进 | 仅针对特定架构 |
| **PaFu-KV** | 2026.01 | KV cache 的 past-future 分解压缩 | 仅 KV cache 压缩 |
| **Quant VideoGen** | 2026.02 | 2-bit KV cache 量化，~700 帧 | 仅量化一种方法 |
| **CausalWan-MoE** | 2026.02 | FastVideo 的因果推理 + MoE | 预览阶段 |

### 确认的空白

1. **没有 AR 视频的缓存方法系统研究** — FlowCache 是唯一的，且只用了简单策略
2. **没有 AR 视频的稀疏注意力** — STA/VSA 等全部面向双向 DiT
3. **没有 AR 视频的蒸馏方法** — rCM/DCM 等面向双向 DiT
4. **KV cache 管理**：分钟级视频 KV cache >30GB，压缩/淘汰策略几乎空白
5. **双向 DiT 加速方法能否迁移到 AR？** — 无人研究
6. **流式推理优化**：AR 天然支持流式输出，但优化（如 speculative decoding for video）空白
7. **跨 chunk 质量漂移** + 加速的交互：加速是否加剧漂移？

### 研究问题

1. 双向 DiT 的 6 类加速方法（缓存/蒸馏/稀疏注意力/量化/流水线/硬件）哪些能迁移到 AR？
2. AR 视频生成的 KV cache 如何高效管理？LLM 的 PagedAttention 能否直接用？
3. 能否做 AR 视频的 speculative decoding？小模型 draft chunks + 大模型 verify
4. AR 视频的最优缓存粒度是什么？(chunk 级 / frame 级 / token 级)

### 为什么仍值得关注

- 加速论文 2026.01 才开始出现，竞争窗口 **6-12 个月**
- AR 在流式/交互/长视频场景有不可替代的优势
- LLM 推理优化社区（vLLM/SGLang）正在进入 — 他们有工程优势但缺乏视频领域知识
- **注意**：此方向的影响力取决于 AR 视频模型的采用速度，存在不确定性

---

## B. 测试时计算 × 加速联合优化 ★★★ 范式改变

### 核心发现

**EvoSearch (2025.05) 证明：Wan 1.3B + 测试时搜索 > Wan 14B 标准推理** — 用 10x 少的参数达到更好质量。

这根本性地改变了"加速"的问题定义：

```
传统思路: 如何让 14B 模型推理更快？
  → 蒸馏/缓存/量化 → 从 100s 压缩到 1s

新思路: 固定 10s 计算预算，如何最优分配？
  → 模型大小 × 步数 × 搜索迭代 × 缓存程度
  → 可能 1.3B + 搜索 > 14B + 加速
```

### 已有工作

| 论文 | 时间 | 做了什么 |
|------|------|----------|
| **EvoSearch** | 2025.05 | 进化搜索优化去噪轨迹，1.3B 超越 14B |
| **Video-T1** | 2025.03 | 测试时搜索提升视频生成质量 |
| **ImagerySearch** | 2025.10 | 图像生成的推理时搜索 |

### 确认的空白

**完全空白**：没有任何论文研究 **加速技术 × 测试时搜索** 的联合优化。

具体空白：
1. 蒸馏步数 × 搜索迭代次数 的 Pareto 前沿？（4步+搜索 vs 50步无搜索？）
2. 缓存+搜索的交互？（缓存减少每次搜索的成本 → 可以搜索更多次？）
3. 固定延迟预算下的最优 (模型大小, 步数, 搜索次数, 缓存率) 配置？
4. 小模型+搜索 vs 大模型+加速 的系统性对比？

### 为什么影响力大

- **改变问题定义** — 从"加速大模型"到"最优计算分配"
- 直接挑战"更大模型+更多加速 = 最优"的假设
- 连接两个热门领域：视频加速 + 测试时计算 (test-time compute scaling)
- 实践影响：如果 1.3B+搜索真的优于 14B+加速，部署成本可降 10x

---

## C. 潜空间自适应压缩 ★★ 正交乘法器

### 核心发现

**VGDFR (2025.04)**：静态场景用更少 latent token → **3x 加速，零质量损失**。

这是一个与现有 6 个方向**完全正交的新维度**：

```
现有加速维度:                     新维度:
① 减少步数 (蒸馏)                ⑦ 减少 latent token 数量
② 降低每步成本 (缓存/稀疏)         (让问题本身变小)
③ 降低精度 (量化)
④ 系统优化 (compile/并行)
⑤ 解码加速 (轻量 VAE)

理论总加速: 108x × 3x = 324x
```

### 已有工作

| 论文 | 时间 | 做了什么 |
|------|------|----------|
| **VGDFR** | 2025.04 | 动态帧率，静态区域用更少 latent token，3x 加速 |
| **DLFR-VAE** | 2025.02 | 动态潜空间帧率，latent token 减半，DiT 2-6.25x |
| **One-DVA** | 2026.02 | 单步视频生成 + 动态帧率 VAE |

### 确认的空白

1. **潜空间压缩 + 模型加速的联合优化无人研究** — VGDFR 只做了 latent 压缩，没有叠加缓存/量化
2. **场景感知 latent 分配** — 静态/动态区域的自适应 token 分配
3. **与蒸馏的交互** — 更少 latent token 是否让蒸馏更容易？
4. **理论界** — latent token 数量的下界是多少？信息论分析

---

## 1. DuCa "随机 ≈ 重要性" 的理论推广

### 空白验证：✅ 确认空白 — 无统一理论，多个独立发现等待连接

**已有相关工作（独立发现，未统一）**：

| 论文 | 领域 | 发现 |
|------|------|------|
| **DuCa** (2024.12) | 扩散模型 token 缓存 | 随机选 token ≈ 按注意力分数选，多样性 > 重要性 |
| **DART** (EMNLP'25) | 多模态 LLM token 剪枝 | 独立发现 "duplication matters more than importance"，提供 Lipschitz 连续性界 |
| **DivPrune** (CVPR'25) | 视觉 token 剪枝 | 形式化为 Max-Min Diversity Problem，多样性选择优于重要性选择 |
| **IDPruner** (2026) | token 剪枝 | 用 MMR 协调重要性+多样性 |
| **"Unreasonable Effectiveness of Random Pruning"** (ICLR'22) | 网络剪枝 | 随机剪枝在过参数化网络中表现接近精心设计的剪枝 |
| **Hidden Semantic Bottleneck** (2025) | DiT 分析 | DiT embedding 角相似度 >99%，2/3 维度可无损删除 |
| **MAE** (CVPR'22) | 视觉自监督 | 随机 mask 75% patch 仍能有效学习 |

**确认的空白**：
1. **无统一理论**连接以上所有发现
2. **无跨粒度验证**：token/block/layer/step 各级别
3. **无信息论解释**
4. DuCa 和 DART **互不引用** — 两个社区独立发现同一现象

### 实验方案

```
实验 1：跨粒度验证（核心贡献）
  在 3+ 模型上：Token/Block/Step 级 随机 vs 最优选择 Pareto 曲线

实验 2：信息论分析
  PCA/SVD 分析特征空间有效维度
  验证 Hidden Semantic Bottleneck 扩展到视频模型

实验 3：边界条件
  找到随机选择失效的拐点（压缩率/模型规模/蒸馏后）

实验 4：理论证明
  基于 Lipschitz 界 + 浓度不等式建立更紧的界
```

---

## 2. 少步模型的缓存失效问题

### 空白验证：⚠️ 半空白 — DisCa 已部分占先

**关键先行工作**：
- **DisCa** (2026.02, CVPR'26 投稿) — TeaCache 在蒸馏模型上 **-15.5% 语义分**，提出 Restricted MeanFlow + 可学习预测器
- **OmniCache** (ICCV'25) — 蒸馏后的 CogVideoX 上 PAB/Delta-DiT **全部模型崩溃**
- **CacheQuant** (CVPR'25) — 缓存+量化**非正交**，误差乘法耦合

**仍然开放**：
1. 多蒸馏×多缓存的全矩阵对比（DisCa 仅测 2 种缓存 × 1 种蒸馏）
2. 三方法交互（蒸馏+缓存+量化）
3. 通用 cache-aware distillation 框架

---

## 3. 长视频加速 Scaling 规律

### 空白验证：✅ 确认空白

核心定位：**首个加速方法 × 视频长度的系统 scaling 研究（5s-120s）**

- 所有主流方法（TurboDiffusion/TeaCache/STA）在 2-5 秒验证
- AdaSpa 最接近但只到 24 秒
- BLADE 自己承认此 gap

---

## 4. 自适应组合框架

### 空白验证：✅ 确认空白

核心定位：**首个视频扩散模型的自适应多方法加速配置框架**

- DiffBench/DiffAgent (2026.01) 仅图像
- 所有交互研究仅两两组合
- 无 prompt 感知的方法选择

```
输入: prompt + target_quality + hardware + latency_budget
  │
  ▼
[轻量分析器] → prompt 复杂度 / 运动强度估计
  │
  ▼
[配置选择器] → 最优组合
  │
  ▼
输出: 最优配置 + 预估质量/速度
```

---

## 5. 统一 benchmark

### 空白验证：✅ 确认空白

VideoAccelBench：涵盖 6 方向 × top-3 方法 × 组合测试 × 硬件矩阵。DiffBench 仅图像。

---

## 6. 消费级 GPU 优化

### 空白验证：⚠️ 半空白

社区工具丰富（SVDQuant/Wan2GP/LTX Desktop/ComfyUI），但缺乏系统性学术研究。最适合作为 benchmark 论文。

---

## X1. Reward-Guided Caching（交叉增强）

用轻量 reward 模型替代启发式缓存决策（L2/cosine/attention score）。

```
架构:
轻量质量预测器 (1-2 layer MLP on block features)
  │
  ├── 输入: block 特征 + 上一步特征 + timestep
  ├── 输出: 质量增益预测 (重算该 block 的贡献)
  │
  决策: 增益 < 阈值 → 缓存; ≥ 阈值 → 重算
  训练: VBench 分数差异作为监督
```

**已有相关**：ReDiF (2025.12) 用 RL 优化扩散加速，DOLLAR 用 latent reward，但**无人用 reward 指导缓存决策**。

**强化方向**：#1 DuCa 随机性理论 + #2 缓存×蒸馏

---

## X2. Anytime Video / 渐进预览（交叉增强）

**PostDiff (ICCV'25) 关键发现**："降低每步计算成本"比"减少步数"更优 — 但仅在图像上验证。

视频上的研究问题：
1. PostDiff 的结论是否在视频上成立？（时间维度可能改变结论）
2. 渐进预览系统：1 秒低质量预览 → 10 秒渐进精化
3. 可变帧计算量与时间一致性的交互

**强化方向**：#4 自适应组合框架

---

## X3. MoE 路由分析（交叉增强）

不训练 MoE 视频 DiT（太贵），而是分析现有 dense DiT 的"虚拟专家"激活模式：

- 训练 post-hoc router 看哪些区域激活哪些"专家"
- 量化不同时间步/帧位置的有效专家数
- 为缓存/剪枝策略提供理论指导

**已有**：DiT-MoE/EC-DIT/Diff-MoE 全在图像上，**无视频 DiT 的 MoE 分析**。

---

## 最终优先级排序

```
                影响力
                  │
          极高    │  B.测试时计算×加速 ★★★
                  │
          很高    │  A.AR视频加速 ★★    C.潜空间压缩 ★★
                  │  1.DuCa理论 ★       4.自适应框架
                  │
           高     │  X1.Reward缓存      3.长视频Scaling
                  │  X2.Anytime Video
                  │
           中     │  2.缓存×蒸馏        5.统一Benchmark
                  │  X3.MoE分析
                  │
           低     │  6.消费级GPU
                  │
                  └──┬──────┬──────┬──────┬──────→ 紧迫性
                   低      中      高     极高
                                    ↑       ↑
                               1,C,X1    A,B

注：A (AR视频加速) 从 ★★★ 降至 ★★，因为双向 DiT 仍主导生产部署(~70%)，
AR 的优势局限于流式/交互/长视频场景。影响力存在不确定性。
```

**最推荐路径**：

```
路径 1 (追求理论深度): B (测试时计算×加速) — 改变问题定义，完全空白
路径 2 (追求低风险):   1 (DuCa理论) + X1 (Reward缓存) — 空白确认，工程量低
路径 3 (追求特定场景): A (AR视频加速) — 面向流式/交互场景，竞争刚开始但有不确定性
路径 4 (追求工程落地): 4 (自适应框架) + C (潜空间压缩) — 面向视频生成服务
```

---

## 参考

### 本调研文档
- [README](../README.md) | [技术全景图](panorama.md) | [在线可视化](https://video-infer-acc.optima.sh/)
- 各方向深入调研：[01](01-step-distillation.md) | [02](02-feature-caching.md) | [03](03-token-pruning-sparse-attention.md) | [04](04-quantization.md) | [05](05-vae-pipeline-optimization.md) | [06](06-hardware-deployment.md)

### 范式级机会相关论文

**自回归视频加速**：
- [FlowCache: AR Video Caching (2026.02)](https://arxiv.org/abs/2602.10825)
- [Light Forcing (2026.02)](https://arxiv.org/abs/2602.04789)
- [Causal Forcing (2026.02)](https://arxiv.org/abs/2602.02214)
- [PaFu-KV: Past-Future KV Cache (2026.01)](https://arxiv.org/abs/2601.21896)
- [Quant VideoGen: 2-bit KV (2026.02)](https://arxiv.org/abs/2602.02958)
- [MAGI-1 (2025.05)](https://arxiv.org/abs/2505.13211)
- [Self-Forcing (2025.06)](https://arxiv.org/abs/2506.08009)

**测试时计算**：
- [EvoSearch: Evolution for Video Quality (2025.05)](https://arxiv.org/abs/2505.17618)
- [Video-T1: Test-Time Search (2025.03)](https://arxiv.org/abs/2503.18942)
- [ImagerySearch (2025.10)](https://arxiv.org/abs/2510.14847)

**潜空间压缩**：
- [VGDFR: Dynamic Frame Rate (2025.04)](https://arxiv.org/abs/2504.12259)
- [DLFR-VAE (2025.02)](https://arxiv.org/abs/2502.11897)
- [One-DVA (2026.02)](https://arxiv.org/abs/2602.04220)

**交叉增强**：
- [ReDiF: RL for Diffusion Acceleration (2025.12)](https://arxiv.org/abs/2512.22802)
- [PostDiff: Cheaper Per-Step > Fewer Steps (ICCV'25)](https://arxiv.org/abs/2508.06160)
- [Diff-MoE (ICML'25)](https://openreview.net/forum?id=JCUsWrwkKw)
- [EC-DIT: Apple MoE DiT (ICLR'25)](https://arxiv.org/abs/2410.02098)
- [DEER: Speculative Decoding for Diffusion (2025.12)](https://arxiv.org/abs/2512.15176)
- [DFlash: Block Diffusion Drafter (2026.02)](https://arxiv.org/abs/2602.06036)
- [T-STITCH: Small-to-Large Model Switching (ICLR'25)](https://arxiv.org/abs/2402.14167)

### 原有方向验证论文

**缓存×蒸馏**：
- [DisCa (2026.02)](https://arxiv.org/abs/2602.05449) | [OmniCache (ICCV'25)](https://arxiv.org/abs/2508.16212) | [CacheQuant (CVPR'25)](https://arxiv.org/abs/2503.01323)

**随机性理论**：
- [DART (EMNLP'25)](https://arxiv.org/abs/2502.11494) | [DivPrune (CVPR'25)](https://arxiv.org/abs/2503.02175) | [IDPruner (2026)](https://arxiv.org/abs/2602.13315) | [Hidden Semantic Bottleneck (2025)](https://arxiv.org/abs/2602.21596) | [Random Pruning (ICLR'22)](https://arxiv.org/abs/2202.02643)

**自适应框架**：
- [DiffBench/DiffAgent (2026.01)](https://arxiv.org/abs/2601.03178) | [UniCP (2025.02)](https://arxiv.org/abs/2502.04393) | [Q-VDiT (ICML'25)](https://arxiv.org/abs/2505.22167) | [SADA (ICML'25)](https://arxiv.org/abs/2507.17135)

**消费级 GPU**：
- [SVDQuant (ICLR'25)](https://arxiv.org/abs/2411.05007) | [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) | [LTX Desktop](https://github.com/Lightricks/LTX-Desktop)

**长视频**：
- [LongLive (ICLR'26)](https://arxiv.org/abs/2509.22622) | [HiStream (2025.12)](https://arxiv.org/abs/2512.21338) | [BlockVid (2025.11)](https://arxiv.org/abs/2511.22973) | [SANA-Video (ICLR'26)](https://arxiv.org/abs/2509.24695) | [FlowCache (2026.02)](https://arxiv.org/abs/2602.10825) | [AdaSpa (ICCV'25)](https://arxiv.org/abs/2502.21079)
