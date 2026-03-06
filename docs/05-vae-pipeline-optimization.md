# 轻量 VAE 解码器与流水线级优化调研

> 调研日期：2026-03-06

## 目录

- [轻量 VAE 解码器](#1-轻量-vae-解码器)
  - [Fast Latent Decoders (arXiv 2503.04871)](#11-fast-latent-decoders-arxiv-250304871)
  - [DLFR-VAE: 动态潜空间帧率](#12-dlfr-vae-动态潜空间帧率)
- [VAE Tiling 策略](#2-vae-tiling-策略)
- [Pipeline Offloading 策略](#3-pipeline-offloading-策略)
- [torch.compile 与算子融合](#4-torchcompile-与算子融合)
- [序列并行 / 上下文并行](#5-序列并行--上下文并行)
- [StreamDiffusion 流水线并行](#6-streamdiffusion-流水线并行)
- [TensorRT 优化](#7-tensorrt-优化)
- [综合对比与适用场景](#8-综合对比与适用场景)

---

## 1. 轻量 VAE 解码器

### 1.1 Fast Latent Decoders (arXiv 2503.04871)

**论文**: "Toward Lightweight and Fast Decoders for Diffusion Models in Image and Video Generation"
**代码**: https://github.com/RedShift51/fast-latent-decoders

#### 核心思想

标准 Stable Diffusion VAE 解码器参数量大 (>=180MB FP16)，在整体推理中占据可观的时间和显存。论文提出用极小的解码器替代，牺牲少量感知质量换取 20x 解码加速。

#### 架构变化

```
标准 SD VAE Decoder (~180MB FP16):
  ├── 4 层 ResNet blocks (通道数 512→512→256→128)
  ├── 中间 attention block
  ├── 多层上采样卷积
  └── GroupNorm + SiLU 激活

TAE-192 轻量解码器 (~25-30MB FP16):
  ├── 4 层 decoder blocks (通道数全部 192)
  ├── 移除 attention block（关键简化）
  ├── 简化上采样路径
  └── 基于 AutoencoderTiny 架构
      AutoencoderTiny(decoder_block_out_channels=(192, 192, 192, 192))
```

**核心简化手段**:
1. **通道数大幅缩减**: 512→192，参数量降至原来的 ~15%
2. **移除 attention 层**: attention 是解码器中计算最密集的部分
3. **统一通道宽度**: 四个 block 都用 192 通道，避免宽→窄的渐变结构

**EfficientViT 变体**: 使用 EfficientViT 的 attention blocks 替代标准卷积 decoder，在保持轻量的同时引入有限的全局感受野。

#### 视频版本: TAE-192 Temporal

在 TAE-192 基础上嵌入时空中间层 (mid spatio-temporal block)，在 spatial attention 之间插入 temporal attention 层，使解码器能处理视频的时间维度。

#### 训练方法

**非知识蒸馏**，而是直接端到端训练。损失函数组合：
- **MSE loss**: 像素级保真度
- **LPIPS loss**: 感知相似度
- **GAN loss**: 对抗监督，保持纹理锐度
- **Temporal alignment error** (仅视频): 时间一致性约束

训练配置: 8x A100 (40GB), 14 epochs, 数据集: LAION-Face + LAION-Aesthetics + UCF101

#### 性能数据

| 配置 | 解码时间 | 加速比 | SSIM | FID |
|------|---------|--------|------|-----|
| **图像 256x256** | | | | |
| 标准 VAE | 0.0100s | 1.0x | 0.7656 | 2.23 |
| TAE-192 | 0.0047s | 2.1x | 0.7034 | 21.48 |
| **图像 1024x1024** | | | | |
| 标准 VAE | - | 1.0x | 0.7729 | 1.10 |
| TAE-192 | - | ~2x | 0.7497 | 2.48 |
| **视频 UCF-101 (8帧)** | | | | |
| 标准 VAE | 0.02169s | 1.0x | - | 8.26 |
| TAE-Temporal | 0.00771s | 2.8x | - | 19.22 |

**解码子模块最高 20x 加速**：在极小分辨率下的单次解码可达 20x；端到端 pipeline 加速约 10-15%。

#### 关键洞察

- 解码质量下降在可接受范围内 (SSIM 下降 <0.05)
- FID 上升较明显，但对于实时/交互式场景可以接受
- **最大价值场景**: 需要快速预览、实时交互的应用（如 ComfyUI 实时预览）
- 解码器在整体 pipeline 中占比有限 (5-10%)，所以 20x 解码加速 → 10-15% 总体加速

---

### 1.2 DLFR-VAE: 动态潜空间帧率

**论文**: "DLFR-VAE: Dynamic Latent Frame Rate VAE for Video Generation" (arXiv 2502.11897)
**代码**: https://github.com/thu-nics/dlfr-vae
**后续工作**: DLFR-Gen (ICCV 2025)

#### 核心思想

现有视频 VAE 使用**固定压缩率** (如 HunyuanVideo 时间维度 4x)，但视频内容的时间复杂度是不均匀的——静态场景帧间差异小、高运动场景帧间差异大。DLFR-VAE 根据内容复杂度**动态调整潜空间帧率**，对静态片段用更低帧率 (更少 latent tokens)，对动态片段用更高帧率。

#### 技术架构

```
输入视频 (T 帧)
    │
    ▼
┌──────────────────────────────┐
│  Dynamic Latent Frame Rate   │
│  Scheduler (DLFRS)           │
│                              │
│  1. 时间分块: 视频分成多个    │
│     temporal chunks          │
│                              │
│  2. 复杂度计算:              │
│     C(Si) = 1/N Σ(1-SSIM(    │
│         x[j], x[j+1]))      │
│     (帧间 SSIM 差异的均值)    │
│                              │
│  3. 帧率分配 (Nyquist):      │
│     F's,i = 2 * f'eff,k     │
│     (Shannon采样定理)         │
│                              │
│  4. 阈值映射: 每个chunk分配  │
│     到N个离散复杂度等级之一   │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Training-Free VAE 适配       │
│                              │
│  Encoder: 插入动态下采样算子  │
│    按帧率调度降低时间分辨率    │
│                              │
│  Decoder: 插入对应上采样算子  │
│    恢复原始帧率               │
│                              │
│  预训练权重基本不变，          │
│  仅新增下/上采样算子          │
└──────────────────────────────┘
    │
    ▼
  动态 Latent Tokens (减少约 50%)
    │
    ▼
  DiT 推理 (token 数减少 → 注意力计算量降低)
```

#### 内容复杂度度量

核心指标：**帧间 SSIM 差异**

```
C(Si) = (1/N) * Σ(1 - SSIM(x[j], x[j+1]))
```

- 静态场景: SSIM 接近 1, C ≈ 0 → 分配低帧率
- 高运动场景: SSIM 偏低, C 较大 → 分配高帧率

这个指标与有效频率和 VAE 重建质量有强相关性，且计算开销极小。

#### 性能数据

| 压缩比 | 延迟降低倍数 | latent token 减少 | SSIM 变化 |
|--------|------------|-------------------|-----------|
| 6x | 2x | ~40% | < 0.03 |
| 8x | 3-4x | ~50% | < 0.03 |
| 12x | 6.25x | ~65% | < 0.03 |

#### 兼容性

已验证的模型：
- **HunyuanVideo VAE** (Kong et al., 2024)
- **Open-Sora 1.2 VAE** (Zheng et al., 2024)
- **DiT 架构**: 通过调整 Rotary Positional Embedding (RoPE) 适配可变序列长度

#### 关键洞察

- **Training-free**: 不需要重新训练 VAE 或 DiT，即插即用
- **加速原因本质**: 减少 latent tokens → attention 计算量二次方降低
- **与步数蒸馏正交**: 可以和 step distillation 叠加使用
- **局限**: 需要额外的复杂度分析步骤；对动态复杂的视频加速有限

---

## 2. VAE Tiling 策略

### 核心问题

视频 VAE 解码/编码时需要处理整个视频张量，对于高分辨率长视频，显存需求可能达到 60GB+ (720p)。Tiling 通过**分块处理**解决这个问题。

### 空间 Tiling (Spatial Tiling)

```
原始 latent (H x W):
┌─────────────────────┐
│                     │
│   整块解码          │  ← 显存需求巨大
│                     │
└─────────────────────┘

Tiled 解码:
┌────────┬───┬────────┐
│ Tile 1 │ O │ Tile 2 │  O = Overlap 区域
│        │ v │        │
├────────┤ e ├────────┤
│ Tile 3 │ r │ Tile 4 │  每个 tile 独立解码
│        │ l │        │  overlap 区域线性插值混合
└────────┴───┴────────┘
```

**混合策略**: overlap 区域使用基于距离的加权混合——越靠近 tile 中心权重越大，靠近边缘权重越小，消除拼接痕迹。

### 时间 Tiling (Temporal Tiling)

```
视频帧序列: [F1, F2, F3, ..., F129]
                    │
            时间分块 (tile_size=32, overlap=4):
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
[F1...F32]    [F29...F60]     [F57...F88]  ...
    │               │               │
  独立解码        独立解码        独立解码
    │               │               │
    └──── 重叠帧混合 ────┘           │
                    └──── 重叠帧混合 ─┘
```

**因果边界处理**: 提取 tile 时使用 `i:i+tsize+1` (+1 账户因果卷积边界)

### HunyuanVideo 3D VAE 的 Tiling 实现

**压缩比**:
- 时间: 4x (129帧 → 33 latent frames)
- 空间: 8x (H, W 各压 8 倍)
- 通道: 3 RGB → 16 latent channels
- **总压缩比**: ~185x

**因果 3D 卷积 (CausalConv3d)**:
```python
# 非对称时间填充: 只向过去看, 不向未来看
temporal_padding = (kernel_size - 1, 0)  # 因果性保证
spatial_padding = standard_padding       # 空间维度标准填充
```

**典型 tiling 配置**:

| 参数 | 含义 | 典型值 |
|------|------|--------|
| tile_size | 空间 tile 边长 (latent) | 48-128 |
| overlap | 空间重叠区域 | 32 |
| temporal_size | 时间 tile 大小 | 32 |
| temporal_overlap | 时间重叠帧数 | 4 |

### 显存影响

| 配置 | 峰值显存 |
|------|---------|
| 无 tiling, 720p | ~60GB |
| 空间 tiling only | ~24GB |
| 空间+时间 tiling | ~14GB |
| + FP8 量化 + attention slicing | ~8GB |

### 关键洞察

- Tiling 是纯工程优化，**不影响最终输出质量** (只要 overlap 足够)
- 时间/空间 tiling 可独立或组合使用
- **代价是延迟增加**: tile 越小，解码操作次数越多
- 对 AutoencoderKLWan (Wan2.1/2.2 的 VAE) 不支持 tiling/slicing

---

## 3. Pipeline Offloading 策略

### 三种 Offloading 层级

```
┌─────────────────────────────────────────────────────┐
│                 Model Offloading                     │
│  整个模型级别: text_encoder → GPU → CPU → GPU (DiT)  │
│  速度: 快 | 显存节省: 中等                            │
│  实现: pipeline.enable_model_cpu_offload()            │
├─────────────────────────────────────────────────────┤
│                 Group Offloading                     │
│  内部层组级别: transformer blocks 分组加载/卸载        │
│  速度: 中等 | 显存节省: 大                            │
│  实现: model.enable_group_offload(                   │
│          offload_type="block_level",                 │
│          num_blocks_per_group=2)                     │
├─────────────────────────────────────────────────────┤
│              Sequential CPU Offloading               │
│  子模块级别: 每个子模块用完立即卸载                    │
│  速度: 极慢 | 显存节省: 最大                          │
│  实现: pipeline.enable_sequential_cpu_offload()       │
└─────────────────────────────────────────────────────┘
```

### Group Offloading 详解 (HunyuanVideo 核心策略)

Group Offloading 是 HunyuanVideo 实现 13.6GB 峰值显存的关键技术。

**Block-level offloading**:
```python
# 40 个 transformer blocks, 每组加载 2 个
pipeline.transformer.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=2  # 每次只有 2 个 block 在 GPU 上
)
```

**Leaf-level offloading** (更激进):
```python
# 最底层模块级别, 类似 sequential offloading 但可配合 CUDA streams
pipeline.transformer.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="leaf_level",
    use_stream=True  # 关键: CUDA stream 重叠传输与计算
)
```

### CUDA Stream 重叠传输 (Overlapped Group Offloading)

```
时间线 →
GPU计算:  [Block N 计算]  [Block N+1 计算]  [Block N+2 计算]
          ───────────────  ───────────────  ───────────────
数据传输:      [Block N+1 传入]  [Block N+2 传入]  [Block N+3 传入]
          ─────↑──────────────↑──────────────────↑─────────
               异步传输        异步传输           异步传输

关键: 当 GPU 在计算 Block N 时, Block N+1 已经在通过
      另一个 CUDA stream 异步传入 GPU, 消除等待时间
```

**配置选项**:
- `use_stream=True`: 启用 CUDA stream 异步传输, CPU 内存需求 ~2x 模型大小
- `record_stream=True`: 进一步加速, 略微增加显存
- `low_cpu_mem_usage=True`: 减少 CPU 内存峰值, 延迟略增

### HunyuanVideo 720p 13.6GB 显存实现

**完整策略组合**:
1. **Text Encoder 提前释放**: CLIP + T5 编码完成后立即卸载到 CPU (文本嵌入只生成一次)
2. **DiT Group Offloading**: transformer blocks 分组在 CPU/GPU 间交换
3. **Overlapped Transfer**: CUDA streams 重叠计算与传输
4. **VAE Tiling**: 解码阶段分块处理
5. **FP8 量化** (可选): 进一步减少 ~10GB 显存

```
峰值显存分布 (13.6GB):
├── 当前活跃的 DiT blocks (~6GB)
├── KV cache / 中间激活 (~4GB)
├── 异步传输缓冲 (~2GB)
├── VAE tile 解码 (~1GB)
└── 其他 (embeddings, etc) (~0.6GB)
```

### 关键洞察

- **Group Offloading >> Model Offloading >> Sequential Offloading** (速度/显存平衡)
- CUDA stream 重叠是让 offloading 不拖慢推理的关键
- CPU 内存需求翻倍 (模型本体 + pinned memory)
- Disk offloading 也已支持，适用于 CPU 内存也不足的场景

---

## 4. torch.compile 与算子融合

### 工作原理

```
PyTorch Eager 执行:
  每个算子独立调度 → 内存读写多 → GPU利用率低

torch.compile 流程:
  Python 代码
    → FX Graph 追踪 (记录算子计算图)
    → 编译器后端 (Inductor)
      → 算子融合 (Kernel Fusion)
      → 乱序执行 (Out-of-Order Execution)
      → 内存规划优化
    → 生成优化 CUDA kernel
    → 缓存复用
```

### 算子融合 (Kernel Fusion) 具体优化

```
融合前: 3次 GPU kernel launch, 3次显存读写
  y = x + bias          # Kernel 1: 读x,bias → 写y → 存到显存
  z = layer_norm(y)     # Kernel 2: 读y ← 从显存 → 写z → 存到显存
  out = gelu(z)         # Kernel 3: 读z ← 从显存 → 写out

融合后: 1次 GPU kernel launch, 1次显存读写
  out = fused_bias_ln_gelu(x, bias)  # 单个 Kernel: 读x,bias → 写out
                                      # 中间结果 y,z 保留在寄存器/shared memory
```

**Diffusion Transformer 中的典型融合**:
- `Linear + LayerNorm + Activation` → 单 kernel
- `QKV 投影 + RoPE` → 单 kernel
- `Attention Score + Softmax + Value Projection` → FlashAttention kernel
- `AdaLN (自适应 LayerNorm) + Modulation` → 单 kernel

### 编译策略对比

| 策略 | 编译时间 | 运行加速 | 适用场景 |
|------|---------|---------|---------|
| **Vanilla (全模型编译)** | 1-3 分钟 | 最大 | 长时间批量推理 |
| **Regional (仅编译 denoiser)** | 10-30 秒 | 接近最大 | 推荐默认策略 |
| **Per-layer (单层编译复用)** | 极少 | 1.5x | 冷启动敏感场景 |

### 实际加速数据

| 模型 | torch.compile 加速 | 编译+量化 加速 |
|------|-------------------|---------------|
| Flux.1-Dev (图像 DiT) | ~2x | 53.88% (vs bf16) |
| CogVideoX-5b (视频 DiT) | ~1.5x | 27.33% (vs bf16) |
| Wan-1.3B (视频 DiT) | ~1.7x (31s→18s) | - |

### 视频 DiT 特有考虑

```python
# 推荐: Regional compilation, 仅编译 transformer
pipeline.transformer = torch.compile(
    pipeline.transformer,
    mode="reduce-overhead"  # 减少显存开销
)

# 处理可变序列长度 (不同视频时长)
torch.compile(model, dynamic=True)  # 启用动态 shape
# 或精确标记动态维度:
torch.compiler.mark_dynamic(input_tensor, dim=1)  # sequence 维度
```

**关键问题: Graph Breaks**
- 视频 DiT 中的动态控制流 (如可变帧数) 可能导致 graph break
- Graph break 会打断优化，严重降低加速效果
- 解决方案: 使用 `fullgraph=True` 开发时检测，重构为静态模式

### 与 Offloading 的兼容性

torch.compile 与 CPU offloading **可以组合使用**，但需注意：
- 编译后的图假设张量在固定设备上
- Group offloading 的设备转移可能触发重编译
- 推荐: 先 compile，再 offload，使用 `use_stream=True`

---

## 5. 序列并行 / 上下文并行

### 为什么需要序列并行

视频 DiT 生成的 token 序列极长:
- 720p, 81帧: ~130K tokens (HunyuanVideo)
- 1080p, 121帧: ~500K tokens
- Attention 复杂度: O(N^2)，单 GPU 无法容纳

### 三种主流方法

#### DeepSpeed-Ulysses 序列并行

```
原始序列: [token_1, token_2, ..., token_N]
           ↓ 沿序列维度切分
GPU 0: [token_1 ... token_N/P]      shape: [B, H, N/P, D]
GPU 1: [token_N/P+1 ... token_2N/P]
GPU 2: [token_2N/P+1 ... token_3N/P]
GPU 3: [token_3N/P+1 ... token_N]

Attention 计算:
1. 每个 GPU 计算本地 Q·K^T (部分 attention scores)
2. All-to-All 通信交换部分 attention scores
3. 聚合得到全局 attention output

特点:
+ 显存效率高，线性降低
+ 模型一致性好
- attention head 数少时并行度受限
- All-to-All 通信开销
```

#### Ring Attention

```
GPU 排列成环形拓扑:

   GPU 0 ──→ GPU 1
     ↑          ↓
   GPU 3 ←── GPU 2

每个 GPU 持有本地 Q, K, V 分片

Step 1: GPU_i 计算 Q_i · K_i^T (本地)
Step 2: K_i, V_i 发送到 GPU_{i+1}; 接收 K_{i-1}, V_{i-1}
Step 3: GPU_i 计算 Q_i · K_{i-1}^T (接收的)
...重复直到所有 K/V blocks 都处理完毕

特点:
+ 与 attention head 数量无关 (head-agnostic)
+ 硬件灵活性好
- 通信密集, 需要同步 K/V 交换
- P 轮通信 (P = GPU 数量)
```

#### Hybrid Parallelism (USP: Unified Sequence Parallelism)

```
8 GPU 组织为 2D 网格 (2×4):

         Ring Attention 方向 →
         ┌──────┬──────┬──────┬──────┐
Ulysses  │GPU 0 │GPU 1 │GPU 2 │GPU 3 │  ← Ring Group 1
方向 ↓   ├──────┼──────┼──────┼──────┤
         │GPU 4 │GPU 5 │GPU 6 │GPU 7 │  ← Ring Group 2
         └──────┴──────┴──────┴──────┘
           ↑             ↑
        Ulysses       Ulysses
        Group 1       Group 2

每 GPU tensor shape: [B, H, N/(2×4), D]
- Ulysses 轴 (2): 切分序列, All-to-All
- Ring 轴 (4): 循环 K/V blocks

结合了:
  Ulysses 的显存效率 + Ring 的 head-agnostic 扩展性
```

### xDiT 推理引擎

xDiT 是 DiT 模型的并行推理引擎，支持 25+ 模型:

**支持的并行策略**:

| 策略 | 描述 | 约束 |
|------|------|------|
| PipeFusion | 序列级 pipeline 并行，利用扩散时间步冗余 | - |
| USP | Ulysses + Ring Attention 统一并行 | - |
| Data Parallel | 多 prompt 并行 | - |
| CFG Parallel | Classifier-Free Guidance 两路并行 (2x) | 固定 2x |
| Tensor Parallel | 模型参数切分 | - |

**组合约束**: `ulysses_degree × pipefusion_degree × cfg_degree = total_devices`

**支持的视频模型**: HunyuanVideo, HunyuanVideo-1.5, CogVideoX, Wan2.1/2.2, Mochi-1, LTX-2, StepVideo

### 实际性能

| 配置 | 基线 | Hybrid Parallelism | 加速比 |
|------|------|-------------------|--------|
| Wan2.2, 8×H100 | 159s | 60s | 2.65x |
| 相比 Ulysses+FSDP | - | - | 62% faster |

### 关键洞察

- **序列并行是多 GPU 视频生成的必需技术**，因为单 GPU 无法容纳长序列
- Ulysses 和 Ring Attention 各有优劣，**Hybrid 是最优解**
- **通信开销是核心挑战**: 高带宽互联 (NVLink/NVSwitch) 对性能至关重要
- xDiT 还集成了 cache 加速 (TeaCache, First-Block-Cache) 减少冗余计算
- **CFG Parallel 是最简单的 2x 加速**，几乎无通信开销

---

## 6. StreamDiffusion 流水线并行

### StreamDiffusion V1 (2023): 图像实时生成

**核心思想: 去噪步骤的流水线化**

```
传统串行去噪:
  Input → [Step 1] → [Step 2] → [Step 3] → [Step 4] → Output
  等待...   等待...   等待...   等待...   完成

StreamDiffusion 流水线:
  时间 →
  [Frame 1 Step 1] [Frame 1 Step 2] [Frame 1 Step 3] [Frame 1 Step 4]
                   [Frame 2 Step 1] [Frame 2 Step 2] [Frame 2 Step 3]
                                    [Frame 3 Step 1] [Frame 3 Step 2]
                                                     [Frame 4 Step 1]

  关键: 不同帧的不同去噪步骤可以 batch 在一起,
       用单次 U-Net forward 同时处理多个去噪步骤
```

**限制**: 基于 image diffusion (U-Net)，帧间无时间建模，存在闪烁和漂移。

### StreamDiffusionV2 (2025): 视频 DiT 实时生成

**三大核心创新**:

#### 1. DiT Block Scheduler (动态 Pipeline 并行)

```
4 GPU Pipeline Parallel:

GPU 0: [DiT Blocks 0-6]  + VAE Encode
GPU 1: [DiT Blocks 7-14]
GPU 2: [DiT Blocks 15-22]
GPU 3: [DiT Blocks 23-30] + VAE Decode

动态调度器:
- 测量每个 stage 的实际执行时间
- 自动搜索最优 block 分配方案
- VAE 编解码集中在首尾 GPU, 调度器补偿这一不均衡
- 最小化 per-stage latency → 最大化 throughput
```

#### 2. Stream-VAE

低延迟视频 VAE 变体:
- 处理短视频 chunk (如 4 帧) 而非长序列
- 缓存 3D 卷积的中间特征，维持时间连贯性
- 专为流式推理设计

#### 3. Rolling KV Cache with Sink Tokens

```
长时间流式生成:
  [历史 KV] [当前 KV]
       │         │
  相似度计算: 更新最不相似的 sink tokens
       │
  周期性重置 RoPE offset: 防止位置编码漂移
```

### 性能数据

| 模型 | GPU | 步数 | FPS | 首帧延迟 |
|------|-----|------|-----|---------|
| Wan-1.3B | 4×H100 | 1-step | 64.52 | <0.5s |
| Wan-1.3B | 4×H100 | 4-step | 61.57 | <0.5s |
| Wan-14B | 4×H100 | 1-step | 58.28 | <0.5s |
| Wan-14B | 4×H100 | 4-step | 31.62 | <0.5s |

**无 TensorRT，无量化**。

### 能否应用于非流式视频生成?

**部分可以**:
- **Pipeline Parallelism 跨 DiT blocks**: 可以应用，但需要多 GPU
- **Stream-VAE**: 可以用于分段解码，减少 VAE 峰值显存
- **去噪步骤 batch 化**: 仅适用于独立去噪路径 (如 DDPM 变体)
- **Rolling KV**: 仅适用于流式/交互场景

**限制**:
- Pipeline 并行需要 GPU 间高带宽连接
- Batch 去噪步骤要求步间独立性，不适用于所有调度器
- 核心设计针对**实时交互**场景

---

## 7. TensorRT 优化

### Adobe Firefly 视频模型案例 (NVIDIA 官方博客)

#### 优化流程

```
PyTorch 模型
    │
    ▼
ONNX Export (框架无关表示)
    │
    ▼
TensorRT 图优化
    ├── 层融合 (Layer Fusion)
    ├── 精度校准 (FP8/BF16 混合精度)
    ├── Kernel Auto-Tuning
    ├── 内存规划优化
    └── 算子替换 (SDPA → Flash-style kernels)
    │
    ▼
TensorRT Engine (部署)
```

#### 图优化细节

**1. 层融合 (Layer/Kernel Fusion)**:
```
融合前:
  Conv → BN → ReLU  (3个 kernel launches)

融合后:
  ConvBNReLU  (1个 kernel launch)

DiT 特有融合:
  - Multi-Head Attention 全流程融合
  - AdaLN-Zero + Linear 融合
  - SwiGLU/GELU + Linear 融合
```

**2. FP8 量化策略**:
- 格式: **E4M3 FP8** (而非 E5M2)
  - E4M3: 4位指数 + 3位尾数，更精细的精度
  - E5M2: 5位指数 + 2位尾数，更大范围但更粗糙
- 量化公式: `q = clip(round(S * x), q_min, q_max)`
  - S: per-Tensor 缩放因子
- 校准: 使用 **NVIDIA TensorRT Model Optimizer PyTorch API** 进行 post-training quantization

**3. 瓶颈识别**:
- 使用 **NVIDIA Nsight Deep Learning Designer** 进行 profiling
- **SDPA (Scaled Dot Product Attention)** 被识别为主要计算瓶颈
- Attention 占 DiT forward 约 60-70% 的时间

#### 性能数据

| 优化级别 | 相对 PyTorch 基线 | 说明 |
|---------|-----------------|------|
| TensorRT + BF16 | 1.6x 加速 | 纯图优化 |
| TensorRT + FP8 | 2.5x 加速 | 图优化 + 量化 |
| 扩散延迟降低 | 60% | DiT 推理部分 |
| TCO (总拥有成本) 降低 | 40% | 更少 GPU 服务更多用户 |

#### 部署架构

```
Pipeline 各阶段的 GPU 分配:

Text Encoder ──→ H100 GPU (可共享)
     │
     ▼
LDM Encoder ───→ H100 GPU
     │
     ▼
Diffusion DiT ─→ H100 GPU (主瓶颈, TensorRT+FP8 优化)
     │
     ▼
Latent Decoder ─→ H100 GPU
     │
     ▼
Upsampling ────→ H100 GPU
     │
     ▼
Postprocessing → CPU

部署在 AWS EC2 P5/P5en (H100 实例)
```

### 关键洞察

- TensorRT 的核心价值在**图优化 + 精度混合**，不仅仅是量化
- **SDPA 是视频 DiT 的关键瓶颈**，TensorRT 用专用 kernel 替代
- E4M3 优于 E5M2 因为推理不需要大动态范围
- **与 torch.compile 对比**: TensorRT 优化更激进但灵活性更差 (静态图)
- 2.5x 加速 vs torch.compile 的 1.5-2x 加速, 但工程成本更高

---

## 8. 综合对比与适用场景

### 技术分层图

```
┌────────────────────────────────────────────────────────────────┐
│                    视频生成加速技术栈                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  算法层 (改变计算量)                                           │
│  ├── 步数蒸馏 (50步→1-4步, 10-50x)                            │
│  ├── 特征缓存 (跳过冗余 DiT 层, 1.5-3x)                      │
│  └── DLFR-VAE (减少 latent tokens, 2-6x DiT 加速)            │
│                                                                │
│  编译/Runtime 层 (不改变计算量, 优化执行效率)                   │
│  ├── torch.compile (算子融合, 1.5-2x)                         │
│  ├── TensorRT (图优化+量化, 2-2.5x)                           │
│  └── 轻量 VAE Decoder (20x 解码, 10-15% e2e)                 │
│                                                                │
│  并行层 (多 GPU 扩展)                                          │
│  ├── 序列并行 Ulysses/Ring (线性扩展)                          │
│  ├── CFG Parallel (2x, 零通信开销)                             │
│  ├── StreamDiffusion Pipeline Parallel (4 GPU → 60 FPS)       │
│  └── Data Parallel (多请求并行)                                │
│                                                                │
│  显存优化层 (不改变速度, 降低显存需求)                          │
│  ├── Group Offloading + CUDA Streams (~13.6GB 720p)           │
│  ├── VAE Tiling (60GB → 8GB)                                  │
│  ├── FP8 Layerwise Casting (显存减半)                          │
│  └── Attention Slicing (削峰)                                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 兼容性矩阵 (能否叠加)

| | 步数蒸馏 | 特征缓存 | DLFR-VAE | torch.compile | TensorRT | 序列并行 | Offloading | VAE Tiling |
|---|---|---|---|---|---|---|---|---|
| 步数蒸馏 | - | 部分 | 是 | 是 | 是 | 是 | 是 | 是 |
| 特征缓存 | 部分 | - | 是 | 部分 | 难 | 是 | 是 | 是 |
| DLFR-VAE | 是 | 是 | - | 需适配 | 需适配 | 是 | 是 | 是 |
| torch.compile | 是 | 部分 | 需适配 | - | 冲突 | 部分 | 需注意 | 是 |
| TensorRT | 是 | 难 | 需适配 | 冲突 | - | 部分 | 不需要 | 是 |
| 序列并行 | 是 | 是 | 是 | 部分 | 部分 | - | 不需要 | 是 |
| Offloading | 是 | 是 | 是 | 需注意 | 不需要 | 不需要 | - | 是 |
| VAE Tiling | 是 | 是 | 是 | 是 | 是 | 是 | 是 | - |

### 场景推荐

| 场景 | 推荐技术组合 | 预期加速 |
|------|------------|---------|
| **单 GPU 消费级 (8-12GB)** | Group Offloading + VAE Tiling + FP8 Casting | 可跑但慢 |
| **单 GPU 专业级 (24GB)** | torch.compile + 步数蒸馏 + VAE Tiling | 5-10x |
| **多 GPU 推理 (4×H100)** | 序列并行 + torch.compile + 步数蒸馏 | 20-50x |
| **实时交互** | StreamDiffusionV2 + 步数蒸馏 | 30-60 FPS |
| **云端大规模部署** | TensorRT + FP8 + 序列并行 | 2.5x + 线性扩展 |
| **预览/草稿** | 轻量 VAE + 步数蒸馏 | 极快但质量有限 |

---

## 参考资料

1. [Toward Lightweight and Fast Decoders for Diffusion Models (arXiv 2503.04871)](https://arxiv.org/abs/2503.04871)
2. [Fast Latent Decoders GitHub](https://github.com/RedShift51/fast-latent-decoders)
3. [DLFR-VAE: Dynamic Latent Frame Rate VAE (arXiv 2502.11897)](https://arxiv.org/abs/2502.11897)
4. [DLFR-VAE GitHub](https://github.com/thu-nics/dlfr-vae)
5. [HunyuanVideo 3D VAE Architecture](https://deepwiki.com/Tencent-Hunyuan/HunyuanVideo/2.1.1-3d-vae)
6. [HunyuanVideo-1.5 GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
7. [Diffusers: Reduce Memory Usage](https://huggingface.co/docs/diffusers/en/optimization/memory)
8. [Group Offloading PR (diffusers #10503)](https://github.com/huggingface/diffusers/pull/10503)
9. [torch.compile and Diffusers Guide](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
10. [TensorRT Video DiT Optimization (NVIDIA Blog)](https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/)
11. [Hybrid Parallelism for Video DiTs (SimpliSmart)](https://simplismart.ai/blog/hybrid-parallelism-video-dit-faster-inference-simplismart)
12. [xDiT: Scalable DiT Inference](https://github.com/xdit-project/xDiT)
13. [StreamDiffusionV2 (arXiv 2511.07399)](https://arxiv.org/abs/2511.07399)
14. [StreamDiffusion (arXiv 2312.12491)](https://arxiv.org/abs/2312.12491)
15. [Unified Sequence Parallelism (USP)](https://arxiv.org/abs/2405.07719)
16. [Ulysses + Ring Attention Technical Principles](https://huggingface.co/blog/exploding-gradients/ulysses-ring-attention)
17. [diffusers-torchao Optimization Recipes](https://github.com/sayakpaul/diffusers-torchao)
