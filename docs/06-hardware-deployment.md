# 硬件加速与部署优化深度调研

> 调研日期：2026-03-06

```
硬件加速与部署优化
├── 1. FlightVGM — FPGA 专用加速器
│   ├── 激活稀疏化 (帧间 + 帧内)
│   ├── 浮点-定点混合精度 DSP58
│   └── 动静态自适应调度
├── 2. NVIDIA GPU 代际演进
│   ├── A100 → H100 → H200 → B200
│   ├── Tensor Core 精度演进 (INT8→FP8→FP4)
│   └── 视频扩散模型受益的硬件特性
├── 3. 消费级 GPU 部署
│   ├── RTX 4090 (24GB, Ada Lovelace)
│   ├── RTX 5090 (32GB, Blackwell)
│   └── HunyuanVideo 1.5 实战
├── 4. AMD GPU 支持
│   ├── ROCm 生态现状
│   ├── FastVideo on MI300X
│   └── 与 NVIDIA 性能对比
├── 5. Apple Silicon / 移动端
│   ├── M5 芯片 + MLX
│   ├── 移动端视频生成 (iPhone 16 Pro Max)
│   └── DiffusionKit / CoreML
├── 6. 推理服务框架
│   ├── SGLang Diffusion
│   ├── vLLM-Omni
│   └── 批处理与调度策略
└── 7. 成本分析
    ├── 云 API 定价
    ├── GPU 云租赁 vs 自建
    └── 不同硬件的单视频成本
```

---

## 1. FlightVGM — 首个视频生成 FPGA 加速器 (FPGA 2025 Best Paper)

> 论文: [FlightVGM: Efficient Video Generation Model Inference with Online Sparsification and Hybrid Precision on FPGAs](https://dl.acm.org/doi/10.1145/3706628.3708864)
> 来源: ACM/SIGDA FPGA 2025, Monterey, CA, 2025年2月27日-3月1日
> 荣誉: **FPGA 2025 最佳论文奖**

### 1.1 核心贡献

FlightVGM 是**首个面向视频生成模型 (VGM) 的 FPGA 加速器**，基于 AMD Versal V80 FPGA 实现，在稀疏 VGM 工作负载上：
- **性能超越 NVIDIA RTX 3090 达 1.30x**
- **能效超越 RTX 3090 达 4.49x**
- 峰值计算能力超过 GPU 21%

### 1.2 激活稀疏化机制

FlightVGM 提出**时空在线激活稀疏化架构**，将计算开销降低 **3.17x**。稀疏化分为两个层级：

#### 帧间稀疏化 (Inter-frame Sparsification)

利用视频相邻帧之间的**时间冗余**。在扩散去噪过程中，连续帧的激活特征高度相似。FlightVGM 通过比较相邻帧的激活差异，对差异低于阈值的帧复用前一帧的计算结果，避免重复推理。

```
帧间稀疏化流程：
Frame t   →  计算特征 F(t)
Frame t+1 →  比较 diff(F(t), F(t+1))
  ├── diff < threshold → 复用 F(t)，跳过计算  ← 节省计算
  └── diff ≥ threshold → 正常计算 F(t+1)
```

#### 帧内稀疏化 (Intra-frame Sparsification)

利用单帧内的**空间冗余**。采用 Token 分块策略：

1. 将每帧的 token 分成大小为 K 的 chunk
2. 每个 chunk 选择一个**参考 token**
3. 计算 chunk 内其余 token 与参考 token 的相似度
4. 相似度超过阈值的 token 直接用参考 token 的结果替代

```
帧内稀疏化 (Token Chunking)：
[t1, t2, t3, t4 | t5, t6, t7, t8 | ...]
 ref           ref
  ├── sim(t2,t1) > θ → 复用 t1 结果
  ├── sim(t3,t1) > θ → 复用 t1 结果
  └── sim(t4,t1) < θ → 正常计算 t4
```

### 1.3 浮点-定点混合精度 DSP58 扩展

FlightVGM 提出了一种创新的**混合精度策略**，在精度和效率之间取得最优平衡：

| 层类型 | 精度选择 | 原因 |
|--------|----------|------|
| 线性层 (FFN) | **定点精度** | 计算量大、对精度不敏感 |
| 注意力层 (MSA/MCA) | **浮点精度** | 对数值范围敏感、需要高动态范围 |

在 AMD V80 FPGA 上，利用 **DSP58 扩展架构**实现浮点-定点混合运算，使 PCP (Peak Computing Power) 提升 **3.26x**。DSP58 是 AMD Versal 系列 FPGA 的高性能 DSP 单元，FlightVGM 通过巧妙的数据格式转换，在同一硬件单元上支持两种精度模式。

### 1.4 动静态结合自适应调度

为了处理在线稀疏化带来的不规则计算模式，FlightVGM 提出**动静态结合自适应调度方法**：

- **静态部分**：对已知计算模式（如全计算帧）使用预定义的数据流
- **动态部分**：对稀疏化跳过的 token/帧进行运行时动态调度
- 计算利用率提升 **2.75x**

### 1.5 FPGA vs GPU 对比

| 指标 | FlightVGM (V80 FPGA) | NVIDIA RTX 3090 |
|------|----------------------|-----------------|
| 性能 | **1.30x** | 1.0x (基准) |
| 能效 | **4.49x** | 1.0x (基准) |
| 适用场景 | 边缘部署、低功耗推理 | 通用 GPU 推理 |
| 灵活性 | 需要定制硬件设计 | 通用编程 |

**FPGA 加速的核心优势**：可以在硬件层面直接实现稀疏计算，不像 GPU 那样即使跳过 token 也需要填充对齐。这使得稀疏化的理论收益能真正转化为实际加速。

---

## 2. NVIDIA GPU 代际演进 — 视频扩散工作负载视角

### 2.1 GPU 硬件规格对比

| 规格 | A100 SXM | H100 SXM | H200 SXM | B200 SXM |
|------|----------|----------|----------|----------|
| 架构 | Ampere | Hopper | Hopper+ | Blackwell |
| HBM 类型 | HBM2e | HBM3 | HBM3e | HBM3e |
| 显存 | 80 GB | 80 GB | 141 GB | 192 GB |
| 带宽 | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s | 8.0 TB/s |
| TDP | 400W | 700W | 700W | 1000W |
| FP16 Tensor | 312 TFLOPS | 989 TFLOPS | 989 TFLOPS | ~2500 TFLOPS |
| FP8 Tensor | -- | 1,979 TFLOPS | 1,979 TFLOPS | ~5000 TFLOPS |
| FP4 Tensor | -- | -- | -- | ~10000 TFLOPS |

### 2.2 Tensor Core 精度演进

```
Volta (V100)  →  FP16, INT8
Ampere (A100) →  FP16, BF16, TF32, INT8, INT4
Hopper (H100) →  FP16, BF16, TF32, FP8 (E4M3/E5M2), INT8
Blackwell (B200) → FP16, BF16, TF32, FP8, FP6, FP4 (NVFP4), INT8
```

**FP8 (Hopper/Ada Lovelace)**：
- E4M3 格式：1位符号 + 4位指数 + 3位尾数
- 视频扩散推理首选格式，NVIDIA TensorRT 在 Adobe Firefly 视频模型上实现扩散骨干 **2.5x 加速**
- 相比 BF16 可将权重/激活内存占用减半

**NVFP4 (Blackwell)**：
- 每个值 4 位 + 每 16 个值共享一个 FP8 scale（平均 4.5 位/值）
- 模型内存占用较 FP16 减少 **~3.5x**，较 FP8 减少 **~1.8x**
- 精度损失极小：在 AIME 2024 基准上，NVFP4 量化后精度损失 <1%，个别任务甚至提升 2%

### 2.3 视频扩散模型受益的关键硬件特性

**1) HBM 带宽 — 最关键指标**

视频扩散 Transformer (DiT) 是**带宽密集型**工作负载：
- 每个去噪步骤需要完整加载模型权重
- 视频的 token 序列极长（720p 121帧 → 数十万 token）
- 注意力计算的 KV Cache 访问量巨大

B200 的 8.0 TB/s 带宽相比 A100 的 2.0 TB/s 提升 **4x**，直接加速权重加载和注意力计算。

**2) FP8 Tensor Core**

Adobe 使用 TensorRT + FP8 在 H100 上优化 Firefly 视频模型：
- 扩散推理延迟降低 **60%**
- 总拥有成本降低 **~40%**
- SDPA (Scaled Dot Product Attention) 被识别为主要计算瓶颈

**3) 大显存容量**

视频生成模型显存需求极大：
- HunyuanVideo (13B 参数): FP16 下需要 ~26 GB 仅存储权重
- 加上 KV Cache、激活值、VAE Decoder，单卡 80GB 可能不够
- B200 的 192 GB 可以无需 offloading 直接跑大模型

**4) NVLink / NVSwitch — 多卡并行**

视频生成天然适合 Sequence Parallelism（序列并行）：
- 将超长视频 token 序列切分到多张卡
- NVLink 带宽决定了 all-to-all 通信效率
- H100 NVLink: 900 GB/s → B200 NVLink: 1,800 GB/s

### 2.4 实测性能参考

HunyuanVideo 在不同 GPU 上的推理时间 (560x368, 73帧, 20步):

| GPU | 无优化 | 全优化 (Sage+FP8+Compile) |
|-----|--------|--------------------------|
| H100 SXM | 36.7s | **18.8s** |
| A100 | 73.2s | ~55s (SageAttn only) |
| RTX 4090 | 77.3s | ~42s |
| L40 | 87.4s | ~52s |
| A40 | 115.9s | N/A |
| A5000 | 139.6s | N/A |

> 数据来源: [InstaSD HunyuanVideo Performance Testing](https://www.instasd.com/post/hunyuanvideo-performance-testing-across-gpus)

---

## 3. 消费级 GPU 边缘部署

### 3.1 RTX 4090 (24GB, Ada Lovelace)

RTX 4090 是当前消费级视频生成的**最佳性价比选择**。

**关键优势**：
- 24 GB GDDR6X，足够运行主流视频模型
- 支持 FP8 Tensor Core (Compute Capability 8.9)
- torch.compile 和 SageAttention 兼容性好

**HunyuanVideo 1.5 在 RTX 4090 上的实战**：

| 配置 | 分辨率 | 帧数 | 生成时间 | 峰值显存 |
|------|--------|------|----------|----------|
| 标准推理 (50步) | 560x368 | 73 | ~77s | ~22 GB |
| 步数蒸馏 (8-12步) | 480p I2V | 73 | **~75s** | ~18 GB |
| Pipeline Offloading + VAE Tiling | 720p | 121 | 8-12 min | **13.6 GB** |

**关键优化组合**（叠加使用）：
1. **SageAttention** — 注意力加速 22-26%
2. **FP8 量化** — 权重/激活减半，需 CC 8.9+
3. **torch.compile + Triton** — kernel 融合优化
4. **Pipeline Offloading** — 将不活跃模块卸载到 CPU
5. **Group Offloading** — 分组管理显存
6. **VAE Tiling** — 分块解码降低显存峰值
7. **特征缓存** (DeepCache/TeaCache/TaylorCache) — 步间复用
8. **步数蒸馏** — 50步→8步，加速 ~75%

> 参考: [HunyuanVideo-1.5 GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)

### 3.2 RTX 5090 (32GB, Blackwell)

RTX 5090 是消费级 Blackwell 架构的旗舰。

| 规格 | RTX 4090 | RTX 5090 |
|------|----------|----------|
| CUDA Cores | 16,384 | 21,760 |
| VRAM | 24 GB GDDR6X | 32 GB GDDR7 |
| 带宽 | 1,008 GB/s | 1,792 GB/s |
| 功耗 | ~350W (峰值 ~235W avg) | ~575W (峰值 ~587W) |
| FP4 Tensor Core | 无 | 有 |
| 价格 | ~$1,599 | ~$1,999 |

**视频生成性能对比** (WAN 2.1 Image-to-Video):

| 指标 | RTX 4090 | RTX 5090 | 提升 |
|------|----------|----------|------|
| 生成时间 | 763s (12.7min) | 419s (7min) | **45%** |
| 峰值 VRAM | 15.6 GB | 24.2 GB | -- |
| 平均功耗 | 235W | 500W+ | 2x+ |

> 数据来源: [Valdi RTX 5090 vs 4090 Benchmark](https://www.valdi.ai/blog/rtx-5090-vs-4090-in-the-real-world-of-image-to-video-inference)

**RTX 5090 的独特价值**：
- **32 GB VRAM** — LTX Video 13B 等大模型在 4090 上吃力，在 5090 上可流畅运行
- **FP4 Tensor Core** — 未来 NVFP4 量化支持可进一步提升吞吐
- **1,792 GB/s 带宽** — 比 4090 高 78%，显著加速权重加载

**劣势**：
- 功耗翻倍 (587W vs 235W)
- 闲置功耗也高 (85W vs 14W)
- 性价比不如 4090 (45% 性能提升 vs 25% 价格提升 + 2x 电费)

### 3.3 消费级部署实践建议

```
消费级 GPU 选型决策树：

预算 < $600  → RTX 3060 12GB (基础体验, 需要大量 offloading)
预算 ~$1,000 → RTX 4080 16GB (中端, 较短视频可用)
预算 ~$1,600 → RTX 4090 24GB ★ 推荐 (最佳性价比)
预算 ~$2,000 → RTX 5090 32GB (追求最大 VRAM 和速度)
多卡预算    → 2x RTX 4090 (比 1x 5090 更划算)
```

---

## 4. AMD GPU 支持 — ROCm 生态

### 4.1 ROCm 现状

ROCm (Radeon Open Compute) 是 AMD 的 GPU 计算平台，与 NVIDIA CUDA 竞争。

**ROCm 6.3+ 支持状况** (2025年)：
- PyTorch 原生支持 ROCm
- Flash Attention 2/3 已有 ROCm 原生实现
- Torch SDPA 支持
- 基本的 diffusers 管线可运行

**尚不支持**：
- Sliding Tile Attention
- SageAttention (AMD 版)
- 部分 CUDA 专用 kernel (如 sgl-kernel)

### 4.2 FastVideo on AMD MI300X

[FastVideo](https://rocm.blogs.amd.com/artificial-intelligence/fastvideo-v1/README.html) 已在 AMD Instinct MI300X 上验证，使用 ROCm 6.3+。

**TeaCache 优化效果** (Wan2.1-T2V-1.3B, MI300X):

| 配置 | 总耗时 | 每步耗时 |
|------|--------|----------|
| 无 TeaCache | 118.19s | 2.02s/step |
| 有 TeaCache | **72.17s** | 1.11s/step |
| 加速比 | **~39%** | -- |

部署方式：Docker (ROCm PyTorch 镜像 v25.6) + FastVideo pip 安装。

### 4.3 AMD vs NVIDIA 性能对比

**数据中心 GPU 对比** (MI300X vs H100):

| 维度 | AMD MI300X | NVIDIA H100 SXM |
|------|-----------|-----------------|
| HBM 容量 | 192 GB (HBM3) | 80 GB (HBM3) |
| 带宽 | 5.3 TB/s | 3.35 TB/s |
| 大批量推理 | 优势 (batch>128) | -- |
| 小批量推理 | -- | 优势 (batch<128) |
| 生态成熟度 | ROCm, 差距在缩小 | CUDA, 完善 |
| 软件兼容性 | 10-30% 性能差距 | 基准 |

**关键差异**：
- MI300X 在**内存密集型任务**上有优势 (192GB HBM, 5.3 TB/s 带宽)
- H100 在**低延迟推理**上更强，尤其在 FP8/TensorRT 优化后
- 视频生成是内存密集型工作负载，MI300X 的大显存和高带宽理论上有利
- 但 **CUDA 生态优势明显**：SageAttention、TensorRT、sgl-kernel 等关键优化都是 CUDA 优先

**消费级 AMD GPU**：
- RX 7900 XTX (24 GB) 理论上可通过 ROCm 运行
- RX 7600 (8 GB) 可通过 DirectML 运行 CogVideoX 等小模型
- 但**设置复杂度显著增加**，且缺乏大多数加速优化

### 4.4 选型建议

| 场景 | 推荐 |
|------|------|
| 快速原型/开发 | NVIDIA (CUDA 生态完善) |
| 预算敏感的大规模推理 | AMD MI300X (价格/VRAM 优势) |
| 消费级部署 | NVIDIA RTX 系列 (生态差距太大) |
| 长期战略 | 关注 MI325X/MI400, ROCm 差距在快速缩小 |

---

## 5. Apple Silicon / 移动端

### 5.1 Apple M 系列芯片

**M5 芯片** (2025年10月发布):
- 统一内存带宽: 153 GB/s (比 M4 提升 ~30%)
- GPU Neural Accelerators: 相比 M4 实现 **最高 4x 加速**
- FLUX-dev-4bit (12B 参数) 1024x1024 图像生成: 比 M4 **快 3.8x**

**视频生成里程碑**：
- M5 iPad (16GB RAM) 可运行 **Wan 2.2 A14B** 模型，生成 **5 秒 480p (448x768)** 视频
- 这是首次在平板设备上实现实用级视频生成

**MLX 生态**：
- Apple 的 MLX 框架针对 Apple Silicon 优化
- DiffusionKit 支持在 CoreML + MLX 上运行扩散模型
- M4 Max (128GB 统一内存) 可运行较大的扩散模型

**局限**：
- 带宽远低于专用 GPU (153 GB/s vs RTX 4090 的 1008 GB/s)
- 缺乏 FP8 Tensor Core 等专用加速单元
- 大型视频模型仍需大量优化才能实用

### 5.2 移动端视频生成

2025-2026 年出现了**首批移动端视频生成模型**：

**Mobile Video DiT** (2025):

| 指标 | Server 模型 | Mobile 模型 |
|------|-------------|-------------|
| 参数量 | 2.0B | **0.9B** |
| 分辨率 | 512x384 | 512x384 |
| 帧数 | 49 帧 (24fps, ~2秒) | 49 帧 |
| 推理速度 | 6.4 FPS | **~15 FPS** |
| 设备 | GPU 服务器 | **iPhone 16 Pro Max** |
| VBench 评分 | 83.09 | 81.45 |

> 来源: [Taming Diffusion Transformer for Efficient Mobile Video Generation in Seconds](https://arxiv.org/html/2507.13343v2)

**关键优化技术**：
1. **高压缩比 VAE** — 压缩率 8x32x32，大幅减少 latent token 数量
2. **知识蒸馏引导的三级剪枝** — Block 级 (28→20)、Head 级 (32→20)、FFN 维度 (8192→~6144)
3. **对抗式步数蒸馏** — 40步→4步，推理加速 **20x**
4. **Tiled GEMM** — 解决 FFN 内存瓶颈，端到端加速 ~10%
5. 部署: FP16 + Apple CoreML, CLIP (替代 T5) 作为文本编码器

**MobileVD (ICCV 2025)**：
- 在移动设备上部署 Image-to-Video 模型
- 不显著牺牲视觉质量

**MoViE**：
- 设备端视频编辑，达到 **12.5 FPS**
- 接近实时的文本引导视频编辑

### 5.3 边缘部署展望

```
设备能力层级 (2026年):

云端 GPU (H100/B200)       → 720p+, 5-10秒视频, 数十秒完成
消费级 GPU (RTX 4090/5090)  → 720p, 5秒视频, 1-12分钟
Apple M5 Pro/Max            → 480p, 5秒视频, 数分钟
Apple M5 (iPad)             → 480p, 5秒视频, 可运行但较慢
iPhone 16 Pro Max           → 384p, 2秒视频, ~3.3秒
普通手机                     → 暂不可用
```

---

## 6. 推理服务框架

### 6.1 SGLang Diffusion

> 来源: [LMSYS Blog - SGLang Diffusion](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)

SGLang Diffusion 将 SGLang 的高性能服务能力扩展到扩散模型，提供端到端的统一推理引擎。

**架构设计**：
```
SGLang Diffusion 架构
├── ComposedPipelineBase
│   ├── DenoisingStage (去噪循环)
│   ├── DecodingStage (VAE 解码)
│   └── 自定义 Stage
├── sgl-kernel (优化算子)
│   ├── CuTeDSL JIT 融合 kernel
│   ├── LayerNorm + Scale-Shift + Residual 融合
│   └── 时间步正弦嵌入专用 kernel
├── 并行策略
│   ├── USP (Ulysses-SP + Ring-Attention 统一)
│   ├── CFG-Parallel (正负提示分卡)
│   └── Tensor Parallelism
└── Cache-DiT 加速 (可选)
    ├── DBCache
    ├── TaylorSeer
    └── SCM
```

**支持模型**：
- 视频: Wan 系列, FastWan, Hunyuan
- 图像: Qwen-Image, Qwen-Image-Edit, FLUX, Z-Image-Turbo, GLM-Image

**性能表现**：
| 对比 | 加速比 |
|------|--------|
| vs HuggingFace Diffusers | **1.2x - 5.9x** |
| 两月迭代后 vs 初版 | **2.5x** |
| Cache-DiT 开启 | 生成速度提升 **169%** (最高 **7.4x**) |
| vs 所有其他方案 | **最高 5x** |

**进阶优化** (2026年2月):
1. **Token 级序列分片** — 取代帧级分片，将时间/高度/宽度维度展平为统一序列后分发
   - 减少/消除 padding，降低通信开销
2. **Parallel Folding** — 文本编码器与 DiT 解耦并行
   - 文本编码器利用 DiT 的序列并行组作为 tensor 并行维度
3. **分布式 VAE** — 高度维分片 + halo 交换（边界像素共享）
   - 消除 VAE 作为高分辨率视频生成瓶颈
4. **视频保存优化** — gpu_worker 直接编码视频，仅返回文件路径
   - 消除序列化/反序列化开销

**LoRA 支持**：
- `/v1/set_lora` — 激活多个 LoRA，支持强度和目标设置
- `/v1/merge_lora_weights` — 权重合并
- 支持 Wan2.2, FLUX, Qwen-Image 等模型的 LoRA 格式

**Layerwise Offload** (层级卸载):
- 预取下一层权重的同时计算当前层
- 实现计算-加载重叠，降低长视频/高分辨率的显存峰值

**硬件支持**：
- NVIDIA: 4090, 5090, H100, H200
- AMD GPU 支持（有专门 benchmark）
- MUSA 架构支持
- ComfyUI 集成（自定义节点替换去噪引擎）

### 6.2 vLLM-Omni

> 来源: [vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models](https://arxiv.org/html/2602.02204v1)

vLLM-Omni 将 vLLM 的服务能力扩展到**全模态模型**（文本、图像、视频、音频），核心创新是**全解耦服务架构**。

**Stage Graph 架构**：
```
vLLM-Omni Stage Graph (以视频生成为例)

[Text Encoder] → [DiT Diffusion Engine] → [VAE Decoder] → [Video Output]
     Stage 1           Stage 2               Stage 3

每个 Stage 独立:
- 独立的批处理调度器
- 独立的 GPU 资源分配
- 独立的扩缩容
```

**解耦三要素**：
1. **Stage 抽象前端** — 用户定义 forward/preprocess/transfer 函数
2. **编排器** — 管理跨 Stage 执行
3. **独立执行引擎** — 每个模型组件独立运行

**Diffusion Engine 优化**：
- Flash Attention / SAGE Attention / TurboAttention
- 迭代去噪缓存: TeaCache, Cache-DiT
- 并行策略: Ring-Attention Context Parallelism, Ulysses Sequence Parallelism

**数据传输**：
- 单节点: 共享内存 (Thinker→Talker 仅 5.49ms)
- 分布式: Mooncake TCP/RDMA
- 支持流式输出: 下游 Stage 可增量接收中间结果

**性能基准**：

| 模型 | 任务 | 加速效果 |
|------|------|----------|
| BAGEL (T2I) | 文生图 | **2.40x** |
| BAGEL (I2I) | 图生图 | **3.72x** |
| Qwen3-Omni | 端到端 | JCT 降低 **91.4%** |
| MiMo-Audio | 音频 | RTF **11.58x** (含图编译) |

**支持的视频生成模型**：
- Wan2.2 系列 (480x640, 80帧)
- GLM-Image (AR LLM + 7B DiT decoder)

### 6.3 批处理与调度策略对比

| 策略 | SGLang Diffusion | vLLM-Omni |
|------|-----------------|-----------|
| 核心理念 | 统一管线优化 | 全解耦 Stage Graph |
| 批处理方式 | Pipeline 级 Batch | Per-Stage 独立 Batch |
| 并行策略 | USP + CFG-Parallel + TP | Ring-Attention + Ulysses SP |
| 缓存加速 | Cache-DiT (可选) | TeaCache / Cache-DiT |
| 多模态 | 图像/视频生成 | 全模态 (文本/图像/视频/音频) |
| 适用场景 | 纯扩散模型推理 | AR+Diffusion 混合模型 |
| 成熟度 | 生产就绪 (8xH100) | 研究+早期生产 |

**调度策略核心差异**：

SGLang Diffusion 的策略是**管线内优化**：通过 Token 级分片、融合 kernel、Cache-DiT 等手段加速单个扩散管线的执行。适合纯扩散模型的高吞吐服务。

vLLM-Omni 的策略是**跨 Stage 解耦**：每个 Stage 独立调度和批处理，支持不同 Stage 使用不同的 GPU 数量和并行策略。适合 AR+Diffusion 混合架构（如 Qwen-Omni 的 Thinker-Talker 设计）。

---

## 7. 成本分析

### 7.1 云 API 定价 — 生成 5 秒 720p 视频

| 服务商 | 模型 | 5秒视频成本 | 说明 |
|--------|------|------------|------|
| Google | Veo 3.1 Fast (720p) | **~$0.75** | $0.15/秒 |
| Google | Veo 3.1 Standard | ~$2.00 | $0.40/秒 |
| OpenAI | Sora 2 (720p) | **~$0.50** | $0.10/秒 |
| OpenAI | Sora 2 Pro | ~$2.50 | $0.50/秒 |
| Kuaishou | Kling 3.0 | **~$0.15** | $0.029/秒, 最便宜 |
| ByteDance | Seedance 2.0 Basic (720p) | ~$0.70 | $0.14/秒 |
| Runway | Gen-3 | ~$0.50-$1.00 | 按 credit |

> 数据来源: [AI Video Generation Models Pricing](https://aifreeforever.com/blog/best-ai-video-generation-models-pricing-benchmarks-api-access), [Veo 3.1 Pricing](https://www.aifreeapi.com/en/posts/veo-3-1-pricing-per-second-gemini-api)

### 7.2 GPU 云租赁自建推理

**按小时租赁成本**：

| GPU | 云商 | 按需价格 | Spot/竞价 |
|-----|------|----------|-----------|
| H100 SXM | AWS P5 | ~$3.90/hr | ~$1.50/hr |
| H100 SXM | GCP A3-High | ~$3.00/hr | ~$2.25/hr |
| H100 SXM | Lambda Labs | ~$2.49/hr | -- |
| H200 | GMI Cloud | ~$3.35/hr | -- |
| A100 80GB | 市场价 | <$1.00/hr | ~$0.50/hr |
| RTX 4090 | RunPod/Vast.ai | ~$0.40/hr | ~$0.25/hr |

**自建推理每视频成本估算** (5秒 720p, 开源模型如 Wan2.2/HunyuanVideo):

| 硬件 | 单视频生成时间 | GPU成本/视频 | 说明 |
|------|---------------|-------------|------|
| H100 SXM (优化后) | ~19s | **~$0.015** | $3.00/hr, 全优化 |
| H100 SXM (无优化) | ~37s | ~$0.031 | $3.00/hr |
| A100 80GB | ~55s | ~$0.015 | $1.00/hr |
| RTX 4090 (优化后) | ~42s | ~$0.005 | $0.40/hr |
| RTX 4090 (步数蒸馏) | ~75s | ~$0.008 | $0.40/hr, 720p 121帧 |
| RTX 4090 (无优化) | ~77s | ~$0.009 | $0.40/hr |
| RTX 5090 | ~7 min (WAN I2V) | ~$0.035 | $0.50/hr 估计 |

**注意**：以上为纯 GPU 时间成本，不含预处理、排队、网络传输等开销。实际生产环境需考虑 GPU 利用率 (通常 60-80%)。

### 7.3 云 API vs 自建 — 决策分析

```
每月视频生成量 vs 推荐方案:

< 1,000 视频/月 → 云 API (Kling 3.0, ~$150/月)
1,000 - 10,000  → 按需 GPU 云租赁 (RTX 4090, ~$50-150/月)
10,000 - 100,000 → 包月 GPU / 多卡 (H100 spot, ~$500-2,000/月)
> 100,000        → 自建 / 长期租赁 (8xH100, ~$3,000-5,000/月)
```

### 7.4 自建 vs 云 — 收支平衡分析

| 方案 | 前期投入 | 月运营成本 | 收支平衡点 |
|------|----------|-----------|-----------|
| 8xH100 DGX 自建 | ~$300,000 | ~$2,000 (电+运维) | ~12 个月 vs 云租赁 |
| 8xA100 DGX 自建 | ~$200,000 | ~$1,500 | ~10 个月 |
| 2x RTX 4090 工作站 | ~$5,000 | ~$100 (电费) | ~2 个月 vs 云 GPU |
| RTX 5090 单卡 | ~$2,500 | ~$50 | ~1 个月 |

> 参考: [Lenovo On-Premise vs Cloud TCO 2025](https://lenovopress.lenovo.com/lp2225), [GMI Cloud H100 Cost Guide](https://www.gmicloud.ai/blog/how-much-does-the-nvidia-h100-gpu-cost-in-2025-buy-vs-rent-analysis)

### 7.5 综合成本对比图

```
每视频成本 (5秒 720p) — 从贵到便宜:

$2.50  ████████████████████████████████████████  Veo 3.1 Standard
$0.75  ████████████                               Veo 3.1 Fast
$0.50  ████████                                   Sora 2
$0.15  ██                                         Kling 3.0
$0.031 █                                          自建 H100 (无优化)
$0.015 █                                          自建 H100 (优化后)
$0.015 █                                          自建 A100
$0.009 █                                          自建 RTX 4090 (无优化)
$0.005 █                                          自建 RTX 4090 (优化后)
```

**结论**: 自建推理比云 API 便宜 **10-100x**，但需要：
- 模型选型和优化工程投入
- 硬件维护和运维
- 初期硬件采购成本
- 承受质量差异（开源模型 vs 商业闭源模型）

---

## 参考资料

### FlightVGM
- [FlightVGM Paper (ACM DL)](https://dl.acm.org/doi/10.1145/3706628.3708864)
- [FlightVGM PDF (SJTU)](https://dai.sjtu.edu.cn/my_file/pdf/83b404ee-15a2-456f-a173-9d260ad2409f.pdf)
- [FPGA Best Paper Awards](https://tcfpga.org/books/community-awards/page/fpga-best-paper-awards)

### NVIDIA GPU
- [NVIDIA TensorRT for Video Diffusion](https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/)
- [NVIDIA Transformer Engine (FP8/FP4)](https://github.com/NVIDIA/TransformerEngine)
- [NVFP4 for Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference)
- [Blackwell vs Hopper Comparison (Exxact)](https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus)
- [B200 vs H100 Benchmarks (Lightly)](https://www.lightly.ai/blog/nvidia-b200-vs-h100)

### 消费级部署
- [HunyuanVideo-1.5 GitHub](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
- [HunyuanVideo Performance Testing (InstaSD)](https://www.instasd.com/post/hunyuanvideo-performance-testing-across-gpus)
- [RTX 5090 vs 4090 Video Inference (Valdi)](https://www.valdi.ai/blog/rtx-5090-vs-4090-in-the-real-world-of-image-to-video-inference)
- [RTX 5090 AI Review (Puget Systems)](https://www.pugetsystems.com/labs/articles/nvidia-geforce-rtx-5090-amp-5080-ai-review/)
- [Consumer GPU Video Generation Guide 2025 (Apatero)](https://www.apatero.com/blog/consumer-gpu-video-generation-complete-guide-2025)

### AMD
- [FastVideo on AMD (ROCm Blog)](https://rocm.blogs.amd.com/artificial-intelligence/fastvideo-v1/README.html)
- [ROCm vs CUDA Comparison](https://www.thundercompute.com/blog/rocm-vs-cuda-gpu-computing)
- [MI300X vs H100 Benchmarks (Clarifai)](https://www.clarifai.com/blog/mi300x-vs-h100)

### Apple Silicon / Mobile
- [Apple M5 Announcement](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)
- [MLX Neural Accelerators on M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Mobile Video DiT](https://arxiv.org/html/2507.13343v2)
- [Mobile Video Diffusion (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Yahia_Mobile_Video_Diffusion_ICCV_2025_paper.pdf)

### 推理框架
- [SGLang Diffusion (LMSYS)](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)
- [SGLang-Diffusion Advanced Optimizations](https://lmsys.org/blog/2026-02-16-sglang-diffusion-advanced-optimizations/)
- [SGLang-Diffusion Two Months In](https://lmsys.org/blog/2026-01-16-sglang-diffusion/)
- [vLLM-Omni Paper](https://arxiv.org/abs/2602.02204)
- [vLLM-Omni Diffusion Docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion_acceleration/)

### 成本分析
- [GPU Cloud Pricing Comparison 2025 (GMI)](https://www.gmicloud.ai/blog/a-guide-to-2025-gpu-cloud-pricing-comparison)
- [On-Premise vs Cloud TCO 2025 (Lenovo)](https://lenovopress.lenovo.com/lp2225)
- [AI Video Generation Models Pricing](https://aifreeforever.com/blog/best-ai-video-generation-models-pricing-benchmarks-api-access)
- [Cloud GPU Pricing (gpuvec)](https://gpuvec.com/)
