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

## 1. 步数蒸馏 (Step Distillation)

核心思路：将多步扩散过程压缩到少步甚至单步生成。

| 方法 | 效果 | 说明 |
|------|------|------|
| [TurboDiffusion](https://arxiv.org/abs/2512.16093) | **100-200x 加速** | 生数科技+清华，结合 rCM 蒸馏 + SageAttention + W8A8 量化 |
| [DOLLAR](https://arxiv.org/abs/2412.15689) | 4步生成 | 变分分数蒸馏 + 一致性蒸馏 + latent reward 微调 |
| [DCM (Dual-Expert Consistency Model)](https://vchitect.github.io/DCM/) | 少步高质量 | 双专家架构，分别处理语义布局和细节 |
| [Video-BLADE](https://arxiv.org/html/2508.10774v1) | 块稀疏注意力+蒸馏 | 自适应稀疏注意力 + 无数据蒸馏协同设计 |
| NVIDIA FastGen | **10-100x 加速** | 统一蒸馏库，多步模型→单步/少步 |

## 2. 特征缓存 (Feature Caching)

利用相邻去噪步骤之间中间特征的相似性，跳过冗余计算。**当前最活跃的研究方向。**

- **[AdaCache](https://adacache-dit.github.io/)** — 无需训练的即插即用缓存，自适应决定哪些步骤可以复用
- **[FastCache](https://arxiv.org/html/2505.20353v1)** — 时空缓存框架，通过统计相似性测试动态决定复用/重算
- **[DuCa (Dual Feature Caching)](https://arxiv.org/abs/2412.18911)** — 激进+保守双策略交替缓存，在 OpenSora/FLUX 上验证
- **[MixCache](https://arxiv.org/html/2508.12691v1)** — 多粒度混合缓存（step/cfg/block/token 级别）
- **[Token-wise Caching](https://arxiv.org/abs/2409.18523)** — OpenSora 上 **2.36x** 加速，几乎无质量下降

## 3. Token 剪枝 (Token Pruning)

视频帧间存在大量冗余 token，通过动态剪枝减少计算：

- **[Dynamic Diffusion Transformer (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/a44a70acd5d0abc1a252ada9719dd06d-Paper-Conference.pdf)** — 动态计算分配
- **相似性 Token 剪枝** — 无需训练，基于运动和内容变化选择性裁剪冗余空间 token
- **[ViDA](https://dai.sjtu.edu.cn/my_file/pdf/ab524a45-2661-470e-ab57-e762294981e7.pdf)** — 视频扩散 Transformer 差分加速，利用帧间差分近似

## 4. 量化与精度优化

| 方案 | 精度 | 效果 |
|------|------|------|
| NVIDIA TensorRT + FP8/BF16 | 混合精度 | 扩散延迟降低 **60%**，TCO 降低 40% |
| NVFP4 (Blackwell) | 4-bit | 吞吐量较 A100 提升 **2x** |
| W8A8 量化 | INT8 | SkyReels-V1 显存降低 **75%** |
| ComfyUI NVFP4/FP8 | 混合 | 视频生成性能提升 **3x**，显存降低 60% |

## 5. 轻量解码器与流水线优化

- **[轻量 VAE Decoder](https://arxiv.org/html/2503.04871v1)** — 解码速度提升 **20x**，整体提速 15%
- **[DLFR-VAE](https://dai.sjtu.edu.cn/my_file/pdf/3dbe9923-3576-4ac6-aa42-e67f7bc52485.pdf)** — 动态潜空间帧率 VAE
- **[StreamDiffusion](https://arxiv.org/html/2312.12491v2)** — 流水线级方案（Stream Batch + IO 并行），图像生成达 91 fps
- **torch.compile** — kernel 融合 + 算子优化，HunyuanVideo 标配

## 6. 硬件加速与部署优化

- **[FlightVGM (FPGA 2025 最佳论文)](https://fpga.eetrend.com/blog/2025/100589077.html)** — FPGA 上性能比 3090 高 **1.3x**，能效高 **4.49x**
- **HunyuanVideo 1.5 部署实践**：SageAttention + torch.compile + 特征缓存 + pipeline offloading → [单卡 RTX 4090 可跑 720p 121帧，峰值显存仅 13.6GB](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
- **[Open-Sora 2.0](https://arxiv.org/html/2503.09642v3)** — 20万美元训练商业级视频模型，强调推理效率

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
