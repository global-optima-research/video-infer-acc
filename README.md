# 视频模型推理加速调研

> 调研日期：2026-03-06

## 技术方向总览

```
视频模型推理加速
├── 减少采样步数
│   ├── 一致性蒸馏 (Consistency Distillation)
│   ├── 对抗蒸馏 (Adversarial Distillation)
│   └── 渐进式蒸馏 (Progressive Distillation)
├── 降低每步计算量
│   ├── 特征缓存 (Step/Block/Token 级)
│   ├── Token 剪枝 (时空冗余裁剪)
│   ├── 稀疏注意力 (SageAttention, Block-Sparse)
│   └── 差分近似 (帧间复用)
├── 精度优化
│   ├── 混合精度 (FP8/BF16/NVFP4)
│   └── 训练后量化 (W8A8, INT8)
├── 解码加速
│   ├── 轻量 VAE Decoder
│   └── 动态帧率 VAE
├── 系统级优化
│   ├── 流水线并行 (StreamDiffusion)
│   ├── 算子融合 (torch.compile, TensorRT)
│   └── 显存优化 (Offloading, Tiling)
└── 硬件专用加速
    ├── FPGA (FlightVGM)
    ├── NVIDIA TensorRT
    └── 边缘部署
```

## 1. 步数蒸馏 (Step Distillation) — [深入调研](docs/01-step-distillation.md)

核心思路：将多步扩散过程压缩到少步甚至单步生成。

| 方法 | 效果 | 说明 |
|------|------|------|
| [TurboDiffusion](https://arxiv.org/abs/2512.16093) | **100-200x 加速** | 生数科技+清华，结合 rCM 蒸馏 + SageAttention + W8A8 量化 |
| [DOLLAR](https://arxiv.org/abs/2412.15689) | 4步生成 | 变分分数蒸馏 + 一致性蒸馏 + latent reward 微调 |
| [DCM (Dual-Expert Consistency Model)](https://vchitect.github.io/DCM/) | 少步高质量 | 双专家架构，分别处理语义布局和细节 |
| [Video-BLADE](https://arxiv.org/html/2508.10774v1) | 块稀疏注意力+蒸馏 | 自适应稀疏注意力 + 无数据蒸馏协同设计 |
| NVIDIA FastGen | **10-100x 加速** | 统一蒸馏库，多步模型→单步/少步 |

> 详细技术演进、CD vs CT vs 对抗蒸馏对比、各方法架构图和横向对比见 [docs/01-step-distillation.md](docs/01-step-distillation.md)

## 2. 特征缓存 (Feature Caching) — [深入调研](docs/02-feature-caching.md)

利用相邻去噪步骤之间中间特征的相似性，跳过冗余计算。**当前最活跃的研究方向。**

- **[AdaCache](https://adacache-dit.github.io/)** — 无需训练的即插即用缓存，自适应决定哪些步骤可以复用
- **[FastCache](https://arxiv.org/html/2505.20353v1)** — 时空缓存框架，通过统计相似性测试动态决定复用/重算
- **[DuCa (Dual Feature Caching)](https://arxiv.org/abs/2412.18911)** — 激进+保守双策略交替缓存，在 OpenSora/FLUX 上验证
- **[MixCache](https://arxiv.org/html/2508.12691v1)** — 多粒度混合缓存（step/cfg/block/token 级别）
- **[Token-wise Caching](https://arxiv.org/abs/2409.18523)** — OpenSora 上 **2.36x** 加速，几乎无质量下降

> 缓存粒度分类、8 种方法详解、视频 vs 图像缓存差异、选型决策树见 [docs/02-feature-caching.md](docs/02-feature-caching.md)

## 3. Token 剪枝与稀疏注意力 — [深入调研](docs/03-token-pruning-sparse-attention.md)

视频帧间存在大量冗余 token，通过动态剪枝和稀疏注意力减少计算：

- **[DyDiT (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/a44a70acd5d0abc1a252ada9719dd06d-Paper-Conference.pdf)** — 时间步动态宽度 + 空间动态 token，51% FLOPs 减少
- **[STA (Sliding Tile Attention)](https://arxiv.org/abs/2502.04507)** — 瓦片滑窗注意力，kernel **10.45x** 加速，15% token 覆盖 70% 注意力
- **[VSA (Video Sparse Attention)](https://haoailab.com/blogs/fastvideo_post_training/)** — 粗→细两阶段稀疏 + DMD2 联合蒸馏，总加速 **50.9x**
- **[SLA (Sparse-Linear Attention)](https://arxiv.org/abs/2512.16093)** — 三级注意力分类（5% 全量 / 85% 线性 / 10% 跳过），95% 稀疏度
- **[ViDA](https://dai.sjtu.edu.cn/my_file/pdf/ab524a45-2661-470e-ab57-e762294981e7.pdf)** — 帧间差分近似，帧间余弦相似度 >90%

> DyDiT/STA/VSA/SLA 技术细节、Token Merging (VidToMe/PruneVid)、与缓存的协同关系见 [docs/03-token-pruning-sparse-attention.md](docs/03-token-pruning-sparse-attention.md)

## 4. 量化与精度优化 — [深入调研](docs/04-quantization.md)

| 方案 | 精度 | 效果 |
|------|------|------|
| [FPSAttention (NeurIPS 2025)](https://arxiv.org/abs/2505.20353) | FP8 + 结构化稀疏 | Wan2.1-14B 端到端 **4.96x** 加速 |
| [ViDiT-Q (ICLR 2025)](https://arxiv.org/abs/2401.17510) | W8A8 | **无损**量化，2-2.5x 显存节省 |
| [DVD-Quant](https://arxiv.org/abs/2501.01348) | W4A8 PTQ 免数据 | HunyuanVideo **2.12x** 加速 |
| [QVGen](https://arxiv.org/abs/2505.18391) | W4A4 QAT | CogVideoX/Wan 14B 质量可比全精度 |
| [SageAttention2++](https://arxiv.org/abs/2411.10958) | Q/K INT4, P/V FP8 | FlashAttention2 **3.9x** 加速 |
| NVFP4 (Blackwell) | E2M1 4-bit | FP16 **3.5x** 内存压缩，FLUX 上 B200 **6.3x** 加速 |

> PTQ vs QAT 对比、SageAttention 原理、GGUF 量化、各模型实测数据见 [docs/04-quantization.md](docs/04-quantization.md)

## 5. 轻量解码器与流水线优化 — [深入调研](docs/05-vae-pipeline-optimization.md)

- **[轻量 VAE Decoder (TAE-192)](https://arxiv.org/html/2503.04871v1)** — 解码 **20x** 加速，模型 180MB→25MB
- **[DLFR-VAE](https://dai.sjtu.edu.cn/my_file/pdf/3dbe9923-3576-4ac6-aa42-e67f7bc52485.pdf)** — 动态潜空间帧率，latent token 减少 ~50%，DiT **2-6.25x** 加速
- **torch.compile** — 区域编译去噪器，kernel 融合 **1.5-2x** 加速
- **[Sequence Parallelism (USP)](https://github.com/xdit-project/xDiT)** — Ulysses + Ring 混合并行，8×H100 **2.65x** 加速
- **[StreamDiffusionV2](https://arxiv.org/html/2312.12491v2)** — DiT 流水线并行，14B 模型 4×H100 达 **58 FPS**
- **Pipeline Offloading** — Group Offloading + CUDA Streams，HunyuanVideo 720p 峰值仅 **13.6GB**

> VAE Tiling、TensorRT 优化、多卡并行策略、显存优化详细技术见 [docs/05-vae-pipeline-optimization.md](docs/05-vae-pipeline-optimization.md)

## 6. 硬件加速与部署优化 — [深入调研](docs/06-hardware-deployment.md)

- **[FlightVGM (FPGA 2025 最佳论文)](https://dl.acm.org/doi/10.1145/3706628.3708864)** — FPGA 上性能比 3090 高 **1.3x**，能效高 **4.49x**，帧间+帧内激活稀疏化计算量降低 3.17x
- **NVIDIA GPU 代际演进**：A100→H100→H200→B200，Tensor Core 支持 FP8/FP4，B200 带宽达 8 TB/s
- **消费级部署**：RTX 4090 单卡跑 HunyuanVideo 720p 121帧（峰值 13.6GB），RTX 5090 比 4090 快 **45%**
- **AMD ROCm**：FastVideo 在 MI300X 上验证，TeaCache 加速 39%，ROCm 与 CUDA 差距 10-30%
- **Apple Silicon / 移动端**：M5 iPad 可运行 Wan 2.2 A14B 生成 480p 视频；iPhone 16 Pro Max 上 0.9B DiT 达 ~15 FPS
- **推理服务框架**：SGLang Diffusion (vs Diffusers 1.2-5.9x 加速) 和 vLLM-Omni (全解耦 Stage Graph)
- **成本分析**：自建 RTX 4090 每视频 ~$0.005 vs 云 API $0.15-$2.50，自建便宜 10-100x

> 详细 GPU 对比、FPGA 技术原理、移动端部署、服务框架选型、成本决策分析见 [docs/06-hardware-deployment.md](docs/06-hardware-deployment.md)

## 技术全景图

详细的全栈加速架构图、正交叠加关系、方法图谱、场景选型矩阵、成熟度×加速比矩阵、技术栈组合推荐及时间线见 **[docs/panorama.md](docs/panorama.md)**

## 研究机会

基于全面调研识别出的 6 个研究空白，按可行性和影响力排序：

1. **少步模型缓存失效 + 协同设计** — 4 步蒸馏模型中缓存方法是否仍有效？cache-aware distillation
2. **DuCa "随机 ≈ 重要性" 理论推广** — 随机选择在扩散模型中普遍近似最优？信息论解释
3. **长视频专项加速** — 场景感知缓存，帧间冗余利用，30s+ 视频验证
4. **自适应组合框架** — 根据 prompt/硬件/质量要求自动选择最优加速配置
5. **统一 benchmark** — 跨 6 个方向的公平对比评测框架
6. **消费级 GPU 优化** — RTX 4090/5090 专项优化，一键部署工具

> 详细分析、实验方案设计见 [docs/07-research-opportunities.md](docs/07-research-opportunities.md)

## 框架对比

- [FastGen vs FastVideo 对比分析](docs/fastgen-vs-fastvideo.md) — 工程落地选 FastVideo，蒸馏研究选 FastGen，极致性能两者组合

## 推荐加速组合

最容易出效果的组合（可叠加）：

1. **步数蒸馏**（4-8步）— 最大加速倍数来源
2. **特征缓存**（AdaCache/Token-wise）— 即插即用，无需重训练
3. **量化**（FP8/W8A8）— 与上述正交，可叠加

## 参考资料

- [NVIDIA: Optimizing Transformer-Based Diffusion Models for Video Generation](https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/)
- [NVIDIA: Top 5 AI Model Optimization Techniques](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference)
- [NVIDIA: Accelerating Diffusion Models (FastGen)](https://developer.nvidia.com/blog/accelerating-diffusion-models-with-an-open-plug-and-play-offering)
- [A Survey on Cache Methods in Diffusion Models](https://arxiv.org/pdf/2510.19755)
- [NeurIPS 2024: Streamlined Inference for Video Diffusion](https://arxiv.org/html/2411.01171v1)
- [LLM 推理加速方法 2025 年终总结 (知乎)](https://zhuanlan.zhihu.com/p/1987290155812423513)
- [AI 视频生成研究报告 (量子位)](https://www.qbitai.com/2025/06/296949.html)
