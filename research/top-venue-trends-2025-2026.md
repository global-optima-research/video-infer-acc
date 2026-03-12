# 顶会趋势分析与新方向探索

> 日期：2026-03-12
> 数据来源：CVPR 2025, NeurIPS 2025, ICML 2025, ICLR 2026 best paper / oral / spotlight
> 目的：寻找视频推理加速之外的新研究方向

## 一、各顶会趋势汇总

### CVPR 2025
- **Best Paper**: VGGT（前馈 3D 几何估计，取代 SfM/SLAM）
- **趋势**：3D/4D-aware 生成、可控视频生成、统一理解+生成、AR 视觉生成

### NeurIPS 2025
- **Best Paper**: Gated Attention（sigmoid 替代 softmax）、1000 层 RL、扩散模型理论
- **趋势**：后 softmax 注意力、一步生成（MeanFlow）、DenseDPO（视频偏好优化）、扩散理论

### ICML 2025
- **Best Paper**: CollabLLM（主动协作）、Masked Diffusion 理论、AR 生成极限
- **趋势**：IMM（无蒸馏快速采样）、MDM、LLM 推理效率栈、测试时计算扩展

### ICLR 2026
- **Oral**: SANA-Video（线性注意力视频生成）、rCM、LPD（并行 AR 解码）、DiffusionNFT（RL 对齐扩散）
- **趋势**：Agentic AI 主导、连续 token AR 生成、离散扩散、全模态统一、数据治理

---

## 二、跨会议高频信号

| 方向 | CVPR | NeurIPS | ICML | ICLR | 频次 |
|------|:----:|:-------:|:----:|:----:|:----:|
| 3D/4D 生成 | ★★★ | ★ | ★ | ★ | 高 |
| 统一理解+生成 | ★★ | ★ | ★ | ★★ | 高 |
| AR 视觉生成（替代扩散） | ★★ | ★ | ★★ | ★★★ | 高 |
| 扩散模型理论 | | ★★★ | ★★ | ★★ | 高 |
| 视频偏好优化/对齐 | | ★★ | | ★★ | 中 |
| 世界模型 | ★★ | | ★ | ★★ | 中 |
| Agentic AI | | | ★ | ★★★ | 中 |
| 物理感知视频生成 | ★ | ★ | | ★ | 中 |

---

## 三、有潜力的新方向（按适配度排序）

### 方向 A：物理感知视频生成（Reward-based Post-training）

**核心思路**：用物理仿真器或视觉基础模型（VJEPA-2 等）作为 reward signal，通过 DPO/RLHF 提升视频 DiT 的物理合理性。

**已有工作**（少量，早期）：
- VideoREPA (NeurIPS 2025) — 视频表征对齐
- PhyGDPO — 物理引导的 DPO
- VLIPP (ICCV 2025) — 物理合理性约束
- WMReward — 世界模型 reward
- DiffPhy — 物理扩散
- DiffusionNFT (ICLR 2026 Oral) — 在线 RL 对齐扩散模型

**适配度**：
- ✅ 直接复用视频 DiT 经验（Wan 2.1 pipeline）
- ✅ 8×H800 足够做 DPO/RLHF post-training
- ✅ 早期阶段，论文少，gap 大
- ✅ 高影响力——物理一致性是视频生成最大痛点之一
- ⚠️ 需要物理 reward 模型的设计（这本身是研究贡献）

**饱和风险**：**低**

### 方向 B：Action-Conditioned 视频世界模型（机器人方向）

**核心思路**：将预训练视频 DiT 适配为动作条件预测模型，用于机器人策略学习。

**已有工作**（增长中）：
- UWM (Unified World Models)
- AVID (Adapting Video Diffusion to World Models)
- PlayWorld, DreamDojo
- NVIDIA Cosmos Policy
- Runway GWM-1

**适配度**：
- ✅ 视频 DiT 经验直接可用
- ✅ 8×H800 适合微调视频基础模型
- ⚠️ 需要机器人数据（可用公开数据集 RoVid-X 等）
- ⚠️ 需要 embodied AI 领域知识
- ⚠️ 竞争在加速（NVIDIA, Runway 等大厂入场）

**饱和风险**：**中低**

### 方向 C：Video-to-3D（视频扩散→3D 高斯泼溅）

**核心思路**：用视频扩散模型生成多视角视频，蒸馏为 3D Gaussian Splatting 表征。

**已有工作**（增长中）：
- Lyra (ICLR 2026, NVIDIA) — 视频扩散→3D 重建
- VideoScene (CVPR 2025) — 视频扩散蒸馏 3D 场景
- V3D, DiffSplat (ICLR 2025)
- CAT4D (CVPR 2025 Oral) — 多视角视频→4D

**适配度**：
- ✅ 视频 DiT 作为 3D prior
- ⚠️ 需要 3D/NeRF/GS 领域知识
- ⚠️ NVIDIA 等大厂已布局
- ⚠️ 需要多视角数据或合成 pipeline

**饱和风险**：**中**

### 方向 D：AR 视频生成（替代扩散范式）

**核心思路**：用自回归模型（而非扩散）做视频生成，连续 token 或离散 token。

**已有工作**：
- CausVid (CVPR 2025) — 因果视频生成
- Infinity (CVPR 2025 Oral) — 位级 AR 图像生成
- NextStep-1 (ICLR 2026 Oral) — 连续 token AR
- LPD (ICLR 2026 Oral) — 并行 AR 解码
- SANA-Video (ICLR 2026 Oral) — 线性注意力视频

**适配度**：
- ⚠️ 需要从扩散转到 AR，技术栈变化大
- ⚠️ 竞争激烈，大厂（NVIDIA, StepFun）主导
- ❌ 8×H800 可能不够训大规模 AR 视频模型

**饱和风险**：**中高**

### 方向 E：DenseDPO 式视频偏好优化

**核心思路**：细粒度时间偏好优化——对视频片段（而非整段视频）做偏好标注和 DPO。

**已有工作**：
- DenseDPO (NeurIPS 2025 Spotlight) — 细粒度时间偏好优化
- VIVA — Edit-GRPO
- Align-A-Video
- DiffusionNFT (ICLR 2026 Oral) — 在线 RL 对齐

**适配度**：
- ✅ 视频 DiT 经验直接可用
- ✅ 8×H800 足够
- ⚠️ DenseDPO 已占据核心位置
- ⚠️ 需要差异化——物理一致性作为偏好目标？

**饱和风险**：**中**

---

## 四、综合推荐

| 排名 | 方向 | 适配度 | 饱和风险 | 影响力 | 综合 |
|------|------|--------|---------|--------|------|
| **1** | **物理感知视频生成** | ★★★★★ | 低 | 高 | **⭐⭐⭐⭐⭐** |
| 2 | Action-Conditioned 世界模型 | ★★★★ | 中低 | 高 | ⭐⭐⭐⭐ |
| 3 | Video-to-3D | ★★★ | 中 | 中高 | ⭐⭐⭐ |
| 4 | DenseDPO 式偏好优化 | ★★★★ | 中 | 中 | ⭐⭐⭐ |
| 5 | AR 视频生成 | ★★ | 中高 | 高 | ⭐⭐ |

**首选方向 A（物理感知视频生成）**理由：
1. 早期阶段，论文少（5-6 篇），远未饱和
2. 完美匹配资源：视频 DiT 经验 + 8×H800 做 DPO/RLHF
3. 高影响力："AI 生成的视频不符合物理规律"是最广为人知的痛点
4. 自然延伸：AutoAccel 的注意力分析数据可辅助理解物理信息在网络中的传播
5. **方向 A + E 可组合**：用物理仿真作为 reward → DenseDPO 式细粒度偏好优化

**但需要先核查**——上次的教训是不要假设 gap 存在。建议对方向 A 做一次深入竞争格局核查。

---

## 参考文献

### CVPR 2025
- [VGGT](https://arxiv.org/abs/2503.11651) — Best Paper, 前馈 3D 几何
- [CAT4D](https://cat-4d.github.io/) — Oral, 多视角视频→4D
- [GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/) — Oral, 3D 一致视频生成
- [Divot](https://arxiv.org/abs/2412.19574) — Oral, 扩散视频 tokenizer
- [Motion Prompting](https://motion-prompting.github.io/) — Oral, 运动轨迹控制
- [Infinity](https://arxiv.org/abs/2412.04431) — Oral, 位级 AR 图像生成

### NeurIPS 2025
- [Gated Attention](https://arxiv.org/abs/2502.21200) — Best Paper, sigmoid 替代 softmax
- [MeanFlow](https://arxiv.org/abs/2504.12904) — Oral, 一步生成
- [DenseDPO](https://neurips.cc/virtual/2025/poster/117435) — Spotlight, 细粒度视频偏好优化
- [Why Diffusion Models Don't Memorize](https://arxiv.org/abs/2502.15470) — Best Paper, 扩散理论

### ICML 2025
- [IMM](https://proceedings.mlr.press/v267/zhou25c.html) — Oral, 无蒸馏快速采样
- [CollabLLM](https://arxiv.org/abs/2502.02823) — Best Paper, 主动协作
- [MAETok](https://arxiv.org/abs/2501.07574) — Spotlight, 高效 tokenizer
- [ConceptAttention](https://arxiv.org/abs/2502.04320) — Spotlight, DiT 可解释性

### ICLR 2026
- [SANA-Video](https://arxiv.org/abs/2502.06527) — Oral, 线性注意力视频生成
- [rCM](https://arxiv.org/abs/2510.08431) — Oral, 一致性蒸馏
- [LPD](https://arxiv.org/abs/2504.20763) — Oral, 并行 AR 解码
- [DiffusionNFT](https://arxiv.org/abs/2502.09668) — Oral, 在线 RL 对齐扩散
- [NextStep-1](https://arxiv.org/abs/2502.04116) — Oral, 连续 token AR

### 新兴方向
- [VideoREPA](https://neurips.cc/virtual/2025/poster/116051) — NeurIPS 2025, 视频表征对齐
- [PhyGDPO](https://huggingface.co/papers/2512.24551) — 物理引导 DPO
- [VLIPP](https://arxiv.org/abs/2506.07173) — ICCV 2025, 物理合理性
- [Lyra](https://arxiv.org/abs/2509.19296) — ICLR 2026, 视频→3D
- [UWM](https://arxiv.org/abs/2504.02792) — 统一世界模型
