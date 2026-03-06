# Token 稀疏化与稀疏注意力加速调研

> 调研日期：2026-03-06

## 目录

- [总览：稀疏化加速的核心思路](#总览稀疏化加速的核心思路)
- [1. Dynamic Diffusion Transformer (DyDiT / DyDiT++)](#1-dynamic-diffusion-transformer-dydit--dydit)
- [2. ViDA: 视频扩散 Transformer 差分近似加速](#2-vida-视频扩散-transformer-差分近似加速)
- [3. Sliding Tile Attention (STA)](#3-sliding-tile-attention-sta)
- [4. Video Sparse Attention (VSA)](#4-video-sparse-attention-vsa)
- [5. Sparse-Linear Attention (SLA) / TurboDiffusion](#5-sparse-linear-attention-sla--turbodiffusion)
- [6. Sparse VideoGen](#6-sparse-videogen)
- [7. AsymRnR: 非对称 Token 裁剪与恢复](#7-asymrnr-非对称-token-裁剪与恢复)
- [8. 视频跨帧 Token 合并](#8-视频跨帧-token-合并)
- [9. Token 稀疏化与特征缓存的关系](#9-token-稀疏化与特征缓存的关系)
- [方法横向对比](#方法横向对比)
- [组合策略建议](#组合策略建议)

---

## 总览：稀疏化加速的核心思路

视频扩散 Transformer (Video DiT) 的推理瓶颈主要来自 **3D 全注意力的二次复杂度**。一个 5 秒 720P 视频在 HunyuanVideo 中产生约 100K+ token，对应的注意力矩阵计算量极大。

稀疏化加速的核心观察是：**并非所有 token 对所有 token 都同等重要**。具体体现为：

| 维度 | 冗余来源 | 对应方法 |
|------|----------|----------|
| **时间维度** | 相邻帧 token 高度相似 | ViDA 差分、VidToMe 跨帧合并 |
| **空间维度** | 背景/平坦区域的 token 可跳过 | DyDiT 空间路由、AsymRnR |
| **注意力维度** | 注意力权重集中在局部窗口 | STA 滑动瓦片、SVG 空间/时间头 |
| **注意力权重分布** | 大部分权重接近零 | SLA Top-K、VSA 粗到精 |
| **时间步维度** | 不同时间步需要不同计算量 | DyDiT 时间步路由 |

---

## 1. Dynamic Diffusion Transformer (DyDiT / DyDiT++)

> **来源**: ICLR 2025 / arXiv 2504.06803 (DyDiT++)
> **机构**: NUS HPC AI Lab
> **核心思想**: 沿时间步和空间两个维度动态分配计算量

### 1.1 Timestep-wise Dynamic Width (TDW) - 时间步动态宽度

标准 DiT 在所有去噪时间步使用相同宽度的网络。DyDiT 观察到：**不同时间步的预测难度不同**。

**机制**:
- 将时间步嵌入 `E_t` 输入到两个轻量路由器 `R_head` 和 `R_channel`
- 每个路由器由一个 **线性层 + Sigmoid** 组成
- 输出每个注意力头和 MLP 通道组的激活概率
- 使用阈值 0.5 将概率转换为二值掩码 `M_head ∈ {0,1}^H` 和 `M_channel ∈ {0,1}^H`
- 推理时可**预计算**所有时间步的掩码，部署前就确定每步的网络架构

**计算分配规律** (从论文 Figure 5):
- 早期时间步（高噪声）：激活较少的头和通道
- 后期时间步（接近干净图像）：激活更多计算资源

### 1.2 Spatial-wise Dynamic Token (SDT) - 空间动态 Token

**机制**:
- 在每个 MLP 层前，Token 路由器 `R_token` 预测每个 token 的处理概率 `S_token ∈ R^N`
- 阈值 0.5 转为二值掩码 `M_token`，`M_token_i = 1` 表示第 i 个 token 需要被处理
- 推理时使用 **gather** 操作收集需处理的 token，送入 MLP
- 完成后使用 **scatter** 操作恢复到原始位置
- **被跳过的 token 直接通过残差连接传递**（不经过 MLP）
- 注意：**MHSA 永远不被跳过**，以保持 token 间的交互

**哪些 token 获得更多计算**:
- **高计算**: 包含详细纹理、色彩丰富的主体对象区域
- **低计算**: 均匀连续的背景区域

### 1.3 训练方法

- 使用 **Straight-Through Estimator (STE)** 和 **Gumbel-Sigmoid** 使二值决策可微分
- 总损失 = DiT 原始损失 + FLOPs 约束损失：`L = L_DiT + L_FLOPs`
- 热身阶段保持完整模型，之后引入动态路由
- 按幅度排序保证每个 block 在所有时间步至少有一个头/通道被激活

### 1.4 性能

| 指标 | 数值 |
|------|------|
| FLOPs 降低 | 51% (DiT-XL) |
| 实际加速 | **1.73x** |
| FID (ImageNet) | 2.07（有竞争力） |
| 额外微调 | <3% 的迭代次数 |

**局限**: 目前主要在图像生成（ImageNet）上验证，尚未直接应用于视频扩散模型。但其空间路由和时间步路由的思想可推广。

---

## 2. ViDA: 视频扩散 Transformer 差分近似加速

> **来源**: ASPDAC 2025 (亚太设计自动化会议)
> **机构**: 上海交通大学
> **核心思想**: 利用相邻帧间的激活相似性，通过差分计算减少冗余

### 2.1 差分近似算法

ViDA 在算法层面提出了差分近似方法，核心思想是利用视频中相邻帧之间激活值的高度相似性。

**两种算子的差分**:

1. **Act-Act 算子** (激活 x 激活，如 Q*K^T):
   - 选择一个激活作为参考激活 X1
   - 对后续帧的激活 X2 计算差分：`ΔX2 = p(|X2 - X1|)`
   - 仅在差分值显著的位置进行全精度计算
   - 观察：帧间激活的余弦相似度 **> 90%**，大量计算可跳过
   - 平均减少 **51.67%** 的 Act-Act 计算量

2. **Act-W 算子** (激活 x 权重，如线性投影):
   - 类似地利用帧间差分减少冗余

**帧分组策略**:
- 通过分析帧间余弦相似度，将每 **8 帧** 设为一组
- 组内第一帧为参考帧，进行完整计算
- 后续 7 帧通过差分近似计算

### 2.2 硬件层面优化

- **列集中 PE (Column-concentrated Processing Element)**: 利用差分计算后产生的列稀疏模式
- **强度自适应数据流**: 平衡不同操作强度的算子执行

### 2.3 性能

| 对比基准 | 加速比 | 面积效率 |
|----------|--------|----------|
| vs NVIDIA A100 GPU | **16.44x** | 18.39x |
| vs SOTA 视觉加速器 | **2.18x** | 2.35x |

**注意**: ViDA 是一个**硬件加速器设计**（ASIC/FPGA 方案），而非纯软件优化。其算法层面的差分近似思想可在 GPU 上实现，但完整的加速比需要定制硬件。

---

## 3. Sliding Tile Attention (STA)

> **来源**: ICML 2025 / arXiv 2502.04507
> **机构**: Hao AI Lab @ UCSD (FastVideo 团队)
> **核心思想**: 基于 3D 局部性观察，用瓦片级滑动窗口替代逐 token 滑动窗口

### 3.1 3D 局部性观察

论文的关键发现：在预训练视频扩散模型中，**仅覆盖 15.52% 总 token 空间的局部窗口就贡献了 70% 的注意力分数**。这一模式在不同注意力头和不同 prompt 间保持一致。

### 3.2 为什么不用标准滑动窗口注意力 (SWA)？

| 问题 | 标准 SWA | STA |
|------|----------|-----|
| 计算模式 | 每个 query 对应不同的 key 集合，产生不规则内存访问 | 同一瓦片内所有 query 对应相同的 key 组 |
| 混合块问题 | 产生"混合块"（部分被 mask），需要显式 mask 计算（~15% 开销） | **没有混合块**，只有全密和全空块 |
| GPU 利用率 | 低（不规则计算模式） | 高（密集计算） |
| MFU | <15% | **58.79%** |

### 3.3 技术细节

**瓦片设计**:
- 将 token 组织为匹配 FlashAttention block 大小的空间-时间立方体
- 滑动步长为 (T, T, T) 而非逐 token 滑动
- 结果：产生的块要么全密集、要么全空 -- 没有 mask 开销

**Kernel 实现** (消费者-生产者范式，借鉴 FlashAttention-3):
- **生产者 warpgroup**: 异步从 HBM 加载相关 KV 块到 SRAM
- **计算 warpgroup**: 对加载的数据执行密集注意力
- **关键**: 块间的稀疏 mask 处理完全在数据加载阶段完成，计算线程对稀疏模式"完全无感知"

### 3.4 性能

| 指标 | 数值 |
|------|------|
| vs FlashAttention-2 | **2.8-17x** 注意力加速 |
| vs FlashAttention-3 | **1.6-10x** 注意力加速 |
| 峰值加速 (注意力 kernel) | **10.45x** (vs FA3) |
| HunyuanVideo 端到端 (无训练) | 945s → 501s (1.89x) |
| HunyuanVideo 端到端 (微调后) | 945s → **268s (3.53x)** |
| 质量 (VBench, 无训练) | 80.58% |
| 质量 (VBench, 微调) | 82.62% (vs 原始 80.xx%) |
| MFU | 58.79% |

**关键优势**:
- Training-free 即可部署，质量无损
- 微调后进一步提速，质量甚至略有提升
- 人类评估中 70% 偏好 STA 微调版本

---

## 4. Video Sparse Attention (VSA)

> **来源**: arXiv 2505.13389
> **机构**: Hao AI Lab @ UCSD (FastVideo 团队)
> **核心思想**: 可训练的粗到精两阶段稀疏注意力，首个可与蒸馏联合训练的稀疏注意力

### 4.1 两阶段架构

#### 粗阶段 (Coarse Stage)

- 将 token 通过 **均值池化** 聚合为 (4,4,4) 的立方体
- 约 100K token 压缩为约 1.5K token
- 在池化表示上计算注意力，识别关键区域
- 计算量 **< 总 FLOPs 的 1%**
- 测试表明均值池化优于最大池化和卷积方法

#### 精阶段 (Fine Stage)

- 对粗阶段通过 **行级 Top-K** 选出的关键立方体，执行 token 级注意力
- 仅在 Top-K 选中的立方体内计算精确注意力

#### 输出融合

两阶段的输出通过**可学习门控**融合：
```
O = O_c * G_c + O_f * G_f
```
其中门控值从输入隐藏状态投影得到。

### 4.2 关键设计选择

- **预测精度**: 粗阶段在大多数层和时间步达到 **60%+** 的关键 token 预测准确率，部分达 **90%**（对比随机 8% 基线）
- **精阶段 kernel**: 保持 FlashAttention3 **85%** 的 MFU
- **硬件友好**: 遵循 block 计算布局，保证硬件效率

### 4.3 与 DMD2 联合训练 (Sparse-Distill)

**这是 VSA 最重要的创新点**：首个与蒸馏兼容的稀疏注意力方法。

- **渐进稀疏度衰减**: 训练开始时使用全注意力，每 50 步逐渐减小 K 值
- **联合训练**: 学生模型同时学习少步生成（DMD2 蒸馏）和稀疏注意力
- 最终学生模型是一个"既快在步数上、又快在每步计算上"的模型

### 4.4 性能

| 模型/场景 | 指标 |
|-----------|------|
| 训练 FLOPs 节省 | **2.53x**（注意力 FLOPs 节省 ~8x） |
| Wan2.1-1.3B 端到端 | 31s → **18s** (1.7x) |
| Wan2.1-14B 端到端 | 1274s → **576s** (2.2x) |
| Sparse-Distill (1.3B) | **50.9x** 端到端加速 |
| 质量 (VBench) | 与原始全注意力模型可比 |
| 人类评估 (200 MovieGen prompts) | 保持生成质量 |

---

## 5. Sparse-Linear Attention (SLA) / TurboDiffusion

> **来源**: arXiv 2509.24006 (SLA) / arXiv 2512.16093 (TurboDiffusion)
> **机构**: 清华大学 ML 组
> **核心思想**: 融合稀疏注意力和线性注意力，突破 90% 稀疏度瓶颈

### 5.1 SLA 三级权重分类

SLA 将注意力权重分为三类，分别用不同复杂度处理：

| 类别 | 占比 (默认配置) | 处理方式 | 复杂度 |
|------|-----------------|----------|--------|
| **Critical (关键)** | ~5% (kh=5%) | 标准 FlashAttention，精确 O(N^2) | O(N^2) |
| **Marginal (边际)** | ~85% | 可学习线性注意力 ϕ(K)^T V | **O(Nd^2)** |
| **Negligible (可忽略)** | ~10% (kl=10%) | 完全跳过 | O(0) |

**工作流程**:
1. 先在池化的 Q/K 上计算压缩注意力矩阵
2. 对矩阵进行 Top-K 选择：前 kh% 标记为 Critical (1)，后 kl% 标记为 Negligible (-1)，其余为 Marginal (0)
3. Critical 块使用 FlashAttention 精确计算
4. Marginal 块使用可学习特征映射 ϕ 做线性注意力：先计算 `ϕ(K)^T V` 和 `rowsum(ϕ(K)^T)`，然后跨所有 query 块复用
5. Negligible 块直接跳过

**关键突破**: 线性注意力作为"可学习补偿"增强了稀疏注意力的效果，而非直接近似边际权重。这需要微调适配。

### 5.2 为什么能突破 90% 稀疏度？

- 纯 Top-K 稀疏注意力在 >90% 稀疏度时质量显著下降
- SLA 在 **95%** 稀疏度时仍能生成与全注意力可比的视频
- 计算复杂度仅为 90% 纯稀疏注意力的约一半

### 5.3 SLA 性能

| 指标 | 数值 |
|------|------|
| 注意力 kernel 加速 | **13.7x** (vs FA2) |
| 注意力延迟 | 97s → 11s (**8.8x**) |
| 注意力 FLOPs 减少 | **20x** |
| 端到端加速 | **2.2x** |
| 稀疏度 | **95%** |

### 5.4 TurboDiffusion 组合加速

TurboDiffusion 将多种加速技术正交组合：

| 组件 | 作用 | 贡献 |
|------|------|------|
| **SLA** | 稀疏注意力 | 注意力稀疏度 90%，Top-K = 0.1 |
| **SageAttention** | 低比特 Tensor Core 加速 | 注意力硬件加速 |
| **SageSLA** | SLA 构建在 SageAttention 之上 | 累积加速 17-20x (注意力) |
| **rCM** | 时间步蒸馏 | 100 步 → 33-44 步 |
| **W8A8 量化** | INT8 参数+激活量化 | 线性层加速 + 模型压缩 |

**关键点**: 稀疏计算与低比特加速是**正交的**，可以累积加速。

**端到端延迟 (RTX 5090)**:

| 模型 | 原始 | TurboDiffusion | 加速比 |
|------|------|----------------|--------|
| Wan2.1-T2V-1.3B-480P | 184s | 1.9s | **~97x** |
| Wan2.1-T2V-14B-480P | 1676s | 9.9s | **~169x** |
| Wan2.1-T2V-14B-720P | 4767s | 24s | **~198x** |
| Wan2.2-I2V-A14B-720P | 4549s | 38s | **~120x** |

---

## 6. Sparse VideoGen

> **来源**: ICML 2025 + NeurIPS 2025 Spotlight / arXiv 2502.01776
> **机构**: NVIDIA EAI
> **核心思想**: 动态分类注意力头为空间头和时间头，分别施加不同稀疏模式

### 6.1 注意力头分类

通过分析 3D 全注意力中各头的注意力模式，发现可动态分为两类：

| 头类型 | 注意力模式 | 物理含义 |
|--------|------------|----------|
| **Spatial Head** | 块状布局，聚焦同帧内的空间局部 token | 维护帧内空间一致性 |
| **Temporal Head** | 斜线布局，等间距，跨帧关注相同空间位置 | 维护时间连贯性 |

### 6.2 在线分析策略

- 不计算完整注意力矩阵，仅**采样 1% 的输入行**
- 分别用空间和时间稀疏模式计算采样 token 的结果
- 选择与全注意力 MSE 更低的模式
- 额外开销仅 **~3%**
- 动态识别是必要的，因为稀疏模式随去噪步骤和输入 prompt 变化

### 6.3 硬件效率解决方案

**时间头的挑战**: 时间注意力的非连续内存布局与 GPU tensor core 要求的 16 元素对齐不兼容。

**解决方案**: **Layout 变换** -- 将 token-major 张量转置为 frame-major 格式，使非连续稀疏模式变为紧凑的硬件友好布局。

### 6.4 性能

| 模型 | 加速比 | PSNR |
|------|--------|------|
| CogVideoX-v1.5-I2V | 2.23x | 28.165 |
| CogVideoX-v1.5-T2V | 2.28x | 29.989 |
| HunyuanVideo-T2V | **2.33x** | 29.546 |

- 与 FP8 量化正交，可额外获得 1.3x 吞吐量提升
- 对比 MInference (均值池化块稀疏)：SVG 在更高加速比下保持更好的 PSNR (29+ vs 22)

---

## 7. AsymRnR: 非对称 Token 裁剪与恢复

> **来源**: ICML 2025 / arXiv 2412.11706
> **核心思想**: 训练-free，对 Q 和 KV token 施加非对称的裁剪策略

### 7.1 核心观察

DiT 中不同特征类型 (Q vs K vs V) 的冗余模式不同：
- **Q 的扰动在浅层 block 中严重降低质量**
- **K 和 V 的扰动影响较小**（因为信息已编码在注意力权重中）

因此应该对 Q 和 KV 采用不同的裁剪力度。

### 7.2 方法

1. **Reduce**: 计算 token 间相似度，迭代合并高相似度的 token 对，直到达到目标长度
2. **Process**: 在裁剪后的序列上执行注意力（O(m^2) 替代 O(n^2)）
3. **Restore**: 根据原始序列的匹配关系复制裁剪后的 token 恢复到原始大小

### 7.3 Matching Cache

由于相邻去噪步之间的匹配相似度变化很小，缓存匹配结果，每 s 步（默认 s=5）才重新计算。

### 7.4 自适应调度

- 基于估计的相似度分布进行自适应阈值判断
- 相似度超过阈值 τ 时才进行裁剪，否则跳过
- 高冗余组件可进行激进裁剪（最高 **80%**）
- 敏感区域保守处理

### 7.5 性能

| 模型 | 加速比 | 质量 |
|------|--------|------|
| CogVideoX-2B | 1.13-1.17x | VBench 持平/略优 |
| CogVideoX-5B | 1.10-1.13x | VBench 持平/略优 |
| HunyuanVideo | **1.24-1.30x** | VBench 持平/略优 |

**特点**: 加速比较为保守，但完全 training-free，模型无关，直接可用。

---

## 8. 视频跨帧 Token 合并

### 8.1 VidToMe (CVPR 2024)

> 零样本视频编辑场景下的跨帧 Token 合并

**核心思想**: 视频帧间 token 在时间域的相关性远高于空间域，可按相关性对齐并压缩。

**机制**:
- **Local Token Merging**: 在视频 chunk 内合并冗余空间 token
- **Global Token Merging**: 组合全局 token 集，实现 chunk 间的 token 共享
- 在合并后的 token 集上执行联合自注意力
- 输出后通过反向合并恢复到原始大小

**分层策略**: 视频分为 chunk，chunk 内做局部合并，chunk 间做全局合并，兼顾短期连续性和长期一致性。

### 8.2 TempMe (ICLR 2025)

> 文本-视频检索中的时间 Token 合并

- **ImgMe Block**: 合并单帧内冗余空间 token
- **ClipMe Block**: 渐进式合并相邻 clip 的 token，跨帧减少时间冗余
  - Cross-clip merging: 聚合相邻 clip 以大幅减少时间 token 数
  - Intra-clip merging: 进一步压缩新形成 clip 内的 token

### 8.3 PruneVid (ACL 2025)

> 视频大语言模型的视觉 Token 裁剪

- 将视频分割为不同场景
- 基于时间变化将 token 分为**静态 token** 和**动态 token**
- 沿时间维度压缩静态 token
- 然后通过空间相似度合并进一步压缩两种 token

### 8.4 TokenTrim

> 自回归长视频生成的推理时 Token 裁剪

- 在条件化上下文中识别不稳定的潜在 token
- 在复用前将其从条件上下文中移除
- 针对长视频自回归生成场景的稳定性优化

---

## 9. Token 稀疏化与特征缓存的关系

### 9.1 它们各自解决什么问题？

| 方法类别 | 减少什么 | 核心维度 |
|----------|----------|----------|
| **特征缓存** | 跨时间步的冗余计算（复用前一步的 block 输出） | **时间步维度** |
| **Token 稀疏化** | 每步内的冗余 token 计算（跳过/合并不重要的 token） | **空间/序列维度** |
| **稀疏注意力** | 注意力矩阵中的冗余计算（只计算重要的 Q-K 对） | **注意力维度** |

### 9.2 它们如何重叠？

1. **都利用了冗余性**: 缓存利用时间步间的特征相似性，稀疏化利用空间/注意力的冗余
2. **都能降低注意力计算量**: 缓存通过跳过整个 block，稀疏化通过减少参与计算的 token 数
3. **部分情况下缓存可视为一种时间步级的 "token 复用"**: 缓存的 token 特征直接被复用

### 9.3 它们如何互补？

**DaTo (Token Pruning for Caching Better)** 论文明确揭示了两者的协同关系：

- **问题**: 特征缓存使 token 间的动态变化（dynamics）被抹平，导致质量下降
- **解决方案**: 选择性剪枝那些**动态被缓存抹平的 token**，保留高动态 token
- **机制**:
  1. Temporal Noise Difference Score (DiffScore): 计算连续时间步输出的绝对差，量化每个 token 的变化程度
  2. 变化小的 token → 与缓存冗余，可安全剪枝
  3. 变化大的 token → 缓存无法覆盖，必须保留参与注意力
  4. NSGA-II 进化算法动态调整每个时间步的缓存深度和裁剪比例
- **效果**: 9x 加速（Stable Diffusion），FID 反而降低 0.33（质量更好）

**ToCa (Token-wise Feature Caching, ICLR 2025)**:
- 在 token 粒度上做缓存决策（而非 block 粒度）
- 不同 token 可以有不同的缓存/重算策略
- FLUX 上实现 3.14x 无损加速

**ClusCa (Cluster-Driven Feature Caching)**:
- 在每个时间步对 token 做空间聚类
- 每个簇仅计算一个 token，信息传播到其他 token
- token 数减少 **>90%**
- 与缓存正交互补

### 9.4 组合范式

```
输入 token 序列 (100K+)
    │
    ├─── Token 稀疏化/合并 ──→ 减少每步内的序列长度 (e.g., 100K → 10K)
    │
    ├─── 稀疏注意力 ──→ 减少注意力矩阵的计算 (e.g., 95% 稀疏)
    │
    ├─── 特征缓存 ──→ 减少需要实际计算的时间步/block 数 (e.g., 50% 步骤跳过)
    │
    └─── 量化 + 蒸馏 ──→ 正交加速
```

---

## 方法横向对比

| 方法 | 类型 | 需要训练 | 加速比 (端到端) | 稀疏度 | 质量影响 | 模型支持 |
|------|------|----------|-----------------|--------|----------|----------|
| **DyDiT** | 动态路由 | 微调 | 1.73x | ~51% FLOPs | FID 2.07 | DiT (图像) |
| **ViDA** | 差分近似 | 否(硬件) | 16.44x (vs GPU) | ~51% Act-Act | 低 | ASIC 方案 |
| **STA** | 瓦片滑动窗口 | 可选微调 | 1.89-3.53x | ~85% 注意力 | 无/极小 | HunyuanVideo |
| **VSA** | 粗到精稀疏 | 训练 | 1.7-2.2x (50.9x w/ distill) | 可变 | 无损 | Wan2.1 |
| **SLA** | 稀疏+线性 | 微调 | 2.2x (注意力 13.7x) | **95%** | 无损 | Wan 系列 |
| **TurboDiffusion** | 组合 (SLA+rCM+量化) | 训练 | **97-198x** | 90% 注意力 | 可比 | Wan 系列 |
| **SVG** | 空间/时间头分类 | 否 | 2.2-2.3x | ~70% | PSNR 29+ | CogVideoX, Hunyuan |
| **AsymRnR** | 非对称 token 裁剪 | 否 | 1.1-1.3x | 最高 80% | 持平/略优 | CogVideoX, Hunyuan |
| **VidToMe** | 跨帧 token 合并 | 否 | 视情况 | - | 改善一致性 | UNet-based |
| **DaTo** | 剪枝+缓存协同 | 否 | **9x** | - | FID 更优 | Stable Diffusion |

---

## 组合策略建议

### 方案 A: Training-Free 快速部署

```
STA (滑动瓦片注意力) + 特征缓存 (TeaCache/FasterCache) + SageAttention (量化)
```
- 预期加速: 3-6x
- 无需训练
- 可立即部署

### 方案 B: 中等投入，显著加速

```
SVG (稀疏注意力头) + AsymRnR (token 裁剪) + 特征缓存 + FP8 量化
```
- 预期加速: 4-8x
- 仅需少量 profiling
- 各组件正交

### 方案 C: 追求极致加速（需要训练）

```
VSA Sparse-Distill (稀疏注意力 + DMD2 蒸馏) + SageAttention + W8A8 量化
```
- 预期加速: 50-100x
- 需要完整训练流程
- FastVideo 框架支持

### 方案 D: TurboDiffusion 全家桶

```
SLA (95%稀疏) + SageAttention + rCM 蒸馏 + W8A8 量化
```
- 预期加速: 100-200x
- 需要训练
- 目前最激进的组合方案
- 已在 RTX 5090 上验证

### 关键组合原则

1. **稀疏注意力 + 低比特量化是正交的** -- 前者减少计算量，后者提升每次计算的效率
2. **稀疏注意力 + 步数蒸馏是正交的** -- 前者加速每步，后者减少步数
3. **Token 裁剪 + 特征缓存可以协同** -- DaTo 证明了裁剪可以改善缓存的质量
4. **Token 裁剪 + 稀疏注意力略有重叠** -- 都在减少注意力计算量，但角度不同（序列长度 vs 稀疏模式）
5. **所有方法都可以叠加量化** -- FP8/INT8 量化是最底层的正交加速

---

## 参考文献

1. DyDiT: Dynamic Diffusion Transformer. ICLR 2025. https://openreview.net/forum?id=taHwqSrbrb
2. DyDiT++: Dynamic Diffusion Transformers for Efficient Visual Generation. https://arxiv.org/abs/2504.06803
3. ViDA: Video Diffusion Transformer Acceleration with Differential Approximation and Adaptive Dataflow. ASPDAC 2025. https://dl.acm.org/doi/10.1145/3658617.3697692
4. STA: Fast Video Generation with Sliding Tile Attention. ICML 2025. https://arxiv.org/abs/2502.04507
5. VSA: Faster Video Diffusion with Trainable Sparse Attention. https://arxiv.org/abs/2505.13389
6. SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention. https://arxiv.org/abs/2509.24006
7. TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times. https://arxiv.org/abs/2512.16093
8. Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity. ICML 2025. https://arxiv.org/abs/2502.01776
9. AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration. ICML 2025. https://arxiv.org/abs/2412.11706
10. VidToMe: Video Token Merging for Zero-Shot Video Editing. CVPR 2024. https://arxiv.org/abs/2312.10656
11. DaTo: Token Pruning for Caching Better. https://arxiv.org/html/2501.00375v1
12. ToCa: Token-wise Feature Caching. ICLR 2025. https://github.com/Shenyi-Z/ToCa
13. FastVideo Framework. https://github.com/hao-ai-lab/FastVideo
