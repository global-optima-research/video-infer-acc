# Quantization Techniques for Video Diffusion Model Inference Acceleration

## Comprehensive Research Report (March 2026)

---

## Table of Contents

1. [FP8 Quantization for DiT Models](#1-fp8-quantization-for-dit-models)
2. [W8A8 (INT8 Weight and Activation Quantization)](#2-w8a8-int8-weight-and-activation-quantization)
3. [NVFP4 (Blackwell 4-bit)](#3-nvfp4-blackwell-4-bit)
4. [SageAttention / SageAttention2++](#4-sageattention--sageattention2)
5. [PTQ vs QAT for Video Diffusion](#5-ptq-vs-qat-for-video-diffusion)
6. [Key Video Diffusion Quantization Papers](#6-key-video-diffusion-quantization-papers)
7. [GGUF Quantization for Diffusion Models](#7-gguf-quantization-for-diffusion-models)
8. [Practical Deployment Numbers](#8-practical-deployment-numbers)

---

## 1. FP8 Quantization for DiT Models

### How FP8 Works for Diffusion Transformers

FP8 quantization reduces weight and activation precision from 16/32-bit to 8-bit floating point. Two formats exist:

| Format | Exponent | Mantissa | Range | Best For |
|--------|----------|----------|-------|----------|
| **E4M3** | 4 bits | 3 bits | +/-[1.52e-2, 448] | Forward inference (weights + activations) |
| **E5M2** | 5 bits | 2 bits | +/-[1.52e-5, 57344] | Backward pass (gradients) |

**E4M3 is preferred for inference** because it provides more granular precision within a tighter range, which matches the activation distributions in DiT models better than E5M2's wider-but-coarser representation.

### TensorRT FP8/BF16 Mixed Precision

Adobe's Firefly video model optimization (NVIDIA TensorRT) demonstrates the state-of-the-art approach:

- **Mixed precision strategy**: FP8 (E4M3) for compute-intensive linear layers; BF16 for components requiring wider dynamic range (normalization, embeddings)
- **Calibration**: Post-training quantization using TensorRT Model Optimizer PyTorch API with per-tensor scaling: `S = max(|X|) / q_max`
- **No re-training needed**: Leverages existing evaluation pipelines

**Performance results (Adobe Firefly)**:
- 2.5x speedup (FP8+BF16 combined) vs PyTorch BF16 baseline
- 1.6x speedup with BF16 alone
- 60% latency reduction overall
- ~40% reduction in total cost of ownership

### Quality Trade-offs

- FP8 E4M3 is more robust to outliers than INT8 and requires less calibration tuning
- FP8 format covers 92.64% of workloads vs INT8's 65.87%
- Distribution analysis shows negligible accuracy degradation when properly calibrated
- Tile-wise FP8 quantization (as in FPSAttention) coupled with structured sparsity enables nearly 5x acceleration with preserved sample quality

### Key Implementation: FPSAttention (NeurIPS 2025 Spotlight)

FPSAttention co-designs FP8 quantization + sparsity for video diffusion 3D attention:

- **Q, K**: Tile-wise FP8 with per-tile scaling (tiles of shape (Tt, Th, Tw))
- **V**: Channel-wise FP8 quantization
- **Attention weights P**: Tensor-wise FP8 using fixed scalar 1/448
- **Sparsity**: Sliding window locality-based attention (temporal=6, height=6, width=1)
- **Denoising step-aware**: 3 regimes (early/mid/late) with adaptive precision

**Results on Wan2.1-14B at 720p**:
- 7.09x kernel speedup for attention
- 4.96x end-to-end speedup
- Combined with TeaCache: up to further acceleration
- PSNR 25.74, SSIM 0.83, LPIPS 0.076

---

## 2. W8A8 (INT8 Weight and Activation Quantization)

### How W8A8 is Applied to Video Models

W8A8 quantizes both weights and activations to 8-bit integers. For video diffusion transformers, the key challenges are:

1. **Temporal variance**: Activations change significantly across denoising timesteps
2. **Spatial variance**: Large variation across channels in attention layers
3. **Cross-modal interactions**: Text-conditioned attention has different distributions

### ViDiT-Q (ICLR 2025) - The Foundational Video DiT Quantization Method

ViDiT-Q addresses three core challenges:

1. **Fine-grained dynamic quantization parameters**: Uses per-channel or per-group scaling rather than per-tensor, adapting to large within-group data variation
2. **Static-dynamic channel balancing**: Combines rotation-based and scaling-based channel balancing to handle the time-varying channel imbalance unique to DiTs
3. **Metric-decoupled mixed precision**: Different aspects of generation quality (aesthetic, consistency, dynamics) respond differently to quantization of different layers

**Results**:
- **W8A8**: Lossless quantization (no metric degradation)
- **W4A8**: Negligible visual quality loss with mixed precision
- **2-2.5x memory saving**, 1.4-1.7x end-to-end latency speedup
- Supported models: Open-Sora v1.2, Latte, PixArt

### Which Layers Are Quantized?

Based on analysis across multiple papers:

| Layer Type | Quantization Sensitivity | Recommended Precision |
|-----------|-------------------------|----------------------|
| Linear projections (QKV, MLP) | Low-Medium | W8A8 safe, W4A8 possible |
| Attention output projections | Medium | W8A8 safe |
| Cross-attention (text-conditioned) | High | Keep higher precision (BF16) |
| Temporal attention | High | W8A8 with careful calibration |
| Normalization layers | Very High | Keep FP16/BF16 |
| Embeddings | Very High | Keep FP16/BF16 |
| VAE encoder/decoder | High | Keep FP16/FP32 |

### Block-wise Granularity Details

- **Per-tensor**: One scale factor per entire tensor. Fast but inaccurate for DiTs due to outliers.
- **Per-channel**: One scale per output channel. Good balance for weights.
- **Per-token**: One scale per token/spatial position. Good for activations.
- **Per-group (block-wise)**: Groups of 32-128 elements share a scale. Best for activations with outliers.
- **Per-thread** (SageAttention): Groups tokens based on GPU mma instruction layouts. Hardware-optimal.

Q-DiT (CVPR 2025) uses evolutionary search to automatically determine optimal group sizes per layer, minimizing FID/FVD while staying hardware-friendly. Dynamic quantization overhead is only 1.3% of dense GEMM kernel cost.

### SkyReels-V1 Memory Optimization

SkyReels-V1 uses **FP8 weight-only quantization** (not full W8A8) combined with parameter-level offloading:
- Generates 544x960px 97-frame (4s) video on single RTX 4090
- Peak VRAM: 18.5GB
- The "75% memory reduction" claim appears to combine quantization + offloading + VAE tiling, not quantization alone

### TurboDiffusion W8A8

TurboDiffusion (Tsinghua TSAIL, Dec 2025) integrates W8A8 for linear layers:
- Combined with SageAttention (INT4 Q/K) + rCM distillation (3-4 steps) + Sparse-Linear Attention
- **100-200x end-to-end speedup** on single RTX 5090
- Wan2.1-T2V-14B-480P: 1,676s -> 9.9s (~169x)
- `--quant_linear` flag enables W8A8 for <=40GB VRAM GPUs

---

## 3. NVFP4 (Blackwell 4-bit)

### How NVIDIA's FP4 Format Works

**Number format**: E2M1 (1 sign bit, 2 exponent bits, 1 mantissa bit)
- Representable values: {0, 0.5, 1.0, 1.5, 2, 3, 4, 6} and their negatives
- Range: approximately -6 to +6

**Two-level scaling (key innovation)**:
1. **Micro-block scaling**: Every 16 values share an E4M3 FP8 scale factor (fractional scaling, not power-of-2)
2. **Tensor-level scaling**: One FP32 scalar normalizes the overall tensor distribution

**Comparison with MXFP4**:

| Feature | NVFP4 | MXFP4 |
|---------|-------|-------|
| Block size | 16 elements | 32 elements |
| Scale format | E4M3 FP8 | E8M0 (power-of-2 only) |
| Scale granularity | 2x finer | Coarser |
| MSE (quantization error) | 0.08 | 0.72 |
| Effective bits/value | ~4.5 | ~4.25 |

### Memory Footprint

- **3.5x reduction** vs FP16
- **1.8x reduction** vs FP8
- Storage: ~4.5 bits per value (4-bit value + shared FP8 scale per 16 values + per-tensor FP32)

### 2x Throughput vs A100 Claim

The claim is more nuanced:
- Blackwell 5th-gen Tensor Cores natively execute FP4 matrix operations
- NVFP4 models are **2.35x faster than INT4** models on Blackwell
- Combined optimizations on B200: **6.3x speedup over H200 baseline** (FP8+NVFP4+TeaCache+compile)
- Multi-GPU B200: **10.2x speedup** vs H200
- Energy efficiency: up to 25x (Blackwell) to 50x (Blackwell Ultra) vs H100

### When is it Available for Video Models?

**Current status (March 2026)**:
- **Available now**: FLUX.1/FLUX.2 (image diffusion) with NVFP4 on Blackwell GPUs
- **Available**: Pre-quantized checkpoints for FLUX.1-dev on Hugging Face
- **Not yet available**: No published NVFP4 results for video diffusion models specifically
- **Framework support**: TensorRT-LLM, vLLM (early), SGLang (upcoming)
- **Calibration**: TensorRT Model Optimizer and LLM Compressor can quantize to NVFP4

**For video models**: NVFP4 should be applicable to Wan2.1/HunyuanVideo DiT backbones since they use the same linear layer structure as image DiTs. The main challenge is ensuring temporal attention quality preservation at 4-bit.

---

## 4. SageAttention / SageAttention2++

### How Low-Bit Quantized Attention Works

SageAttention replaces standard attention computation with quantized matrix multiplications:

```
Standard:  Attn = softmax(Q @ K^T / sqrt(d)) @ V    [FP16/BF16 throughout]
SageAttn2: Attn = softmax(Q_int4 @ K_int4^T / sqrt(d)) @ V_fp8  [mixed precision]
```

### SageAttention2 Precision for Q/K/V

| Matrix | Precision | Granularity | Notes |
|--------|-----------|-------------|-------|
| **Q** | INT4 | Per-thread | Smoothed first (subtract channel-wise mean) |
| **K** | INT4 | Per-thread | Direct quantization |
| **P (attn weights)** | FP8 E4M3 | Per-block | After softmax |
| **V** | FP8 E4M3 | Per-channel | Channel-wise to handle outliers |

**Q Smoothing Technique**:
1. Compute `mean(Q)` across channel dimension
2. Subtract mean before quantization: `Q_smooth = Q - mean(Q)`
3. Add correction post-computation: `deltaS = mean(Q) @ K^T`
4. This improved MMLU from 72.6% to 80.8% with INT4 quantization

**Per-Thread Quantization** (hardware-aware):
- Groups tokens based on `mma.m16n8k64` PTX instruction requirements
- Each GPU thread uses a single quantization scale
- Achieves 99.45% cosine similarity vs 98.03% for per-block methods
- Zero additional dequantization overhead

### SageAttention2++ Improvements (ICML 2025)

Key upgrade: Uses `mma.f16.f8.f8.f16` instruction (FP16 accumulator for FP8 matmul) which is **2x faster** than `mma.f32.f8.f8.f32` (FP32 accumulator) used in SageAttention2.

**Delayed FP32 Buffering**: Accumulates two consecutive mma results in FP16 before converting to FP32, halving conversion overhead.

**Performance**:
- **3.9x faster** than FlashAttention2 on RTX4090 (INT4 variant)
- **3.0x faster** with INT8 variant
- 481 TOPS peak on RTX4090
- Supports Ampere, Ada, and Hopper GPUs
- `torch.compile` compatible

### Video Model Results

| Model | Metric | SageAttn2 | Full Precision |
|-------|--------|-----------|----------------|
| CogVideoX-2B | VQA-A | 77.368 | 77.605 |
| CogVideoX-2B | VQA-T | 74.906 | 75.360 |
| CogVideoX-1.5-5B | Speedup | 1.8x | baseline |

### Usage in TurboDiffusion

TurboDiffusion uses SageAttention as the base attention acceleration, then adds Sparse-Linear Attention (SLA) on top for an additional 17-20x sparse attention speedup. Combined: up to 200x total end-to-end acceleration.

---

## 5. PTQ vs QAT for Video Diffusion Models

### Post-Training Quantization (PTQ)

**How it works**: Quantize a pre-trained model using a small calibration dataset to determine scale factors. No re-training required.

**Advantages for video diffusion**:
- Fast deployment (minutes to hours vs days of training)
- No GPU-intensive training loop
- Works with any pre-trained checkpoint
- Data-free variants exist (DVD-Quant)

**Limitations**:
- Works well at W8A8, but degrades dramatically below W4A8
- Calibration is timestep-dependent (need samples across denoising steps)
- Struggles with the large activation variance in DiTs

**Key PTQ methods for video diffusion**:

| Method | Venue | Precision | Key Innovation |
|--------|-------|-----------|---------------|
| ViDiT-Q | ICLR 2025 | W8A8, W4A8 | Static-dynamic channel balancing |
| Q-DiT | CVPR 2025 | W6A8, W4A8 | Auto granularity allocation + dynamic activation quant |
| DVD-Quant | 2025 | W4A6, W4A4 | Data-free, online scaling with Hadamard rotation |
| SVDQuant | ICLR 2025 Spotlight | W4A4 | Low-rank branch absorbs outliers |
| PQD | WACV 2025 | Various | Efficient PTQ for diffusion models |

### Quantization-Aware Training (QAT)

**How it works**: Simulate quantized operations during training, allowing the model to adapt to low-precision arithmetic.

**Advantages for video diffusion**:
- Can push to much lower precision (W4A4, even W3A3)
- Higher accuracy than PTQ at equivalent bitwidth
- Can compensate for quantization errors through gradient updates

**Limitations**:
- Requires full training infrastructure (massive GPU cost for 14B+ models)
- Training data requirements
- Longer time-to-deploy
- Less flexible (need to re-train for different precision targets)

**Key QAT methods for video diffusion**:

| Method | Venue | Precision | Key Innovation |
|--------|-------|-----------|---------------|
| QVGen | 2025 | W4A4, W3A3 | Auxiliary modules + rank-decay strategy |
| Mixup-Sign (MSFP) | CVPR 2025 | W4A4 FP | Unsigned FP quant + timestep-aware LoRA |
| FPSAttention | NeurIPS 2025 | FP8+Sparse | Training-aware FP8+sparsity co-design |

### Which is Practical?

**For production deployment (W8A8)**: PTQ is clearly practical and sufficient. ViDiT-Q achieves lossless W8A8 with just calibration.

**For aggressive compression (W4A8)**: PTQ is viable with advanced methods (DVD-Quant, SVDQuant). DVD-Quant's data-free approach is especially practical.

**For ultra-low precision (W4A4 and below)**: QAT is necessary. QVGen shows W4A4 comparable to full precision on VBench, but requires expensive training. Video DMs exhibit "large quantization-induced gradient spikes" absent in image models, making QAT convergence harder.

**Recommendation**: Use PTQ at W8A8 for quick deployment, PTQ+advanced methods at W4A8 for moderate compression, and QAT only when pushing to W4A4 or below where the training cost is justified by massive deployment savings.

---

## 6. Key Video Diffusion Quantization Papers

### Comprehensive Paper List (2024-2025)

| Paper | Venue | Type | Models | Precision | Key Result |
|-------|-------|------|--------|-----------|------------|
| **ViDiT-Q** | ICLR 2025 | PTQ | Open-Sora, Latte | W8A8, W4A8 | Lossless W8A8, 2-2.5x memory saving |
| **Q-DiT** | CVPR 2025 | PTQ | DiT-XL/2 | W6A8, W4A8 | 1.09 FID reduction at W6A8 |
| **SVDQuant** | ICLR 2025 Spotlight | PTQ | FLUX, PixArt-Sigma | W4A4 | 3.5x memory, 3x speedup via Nunchaku engine |
| **DVD-Quant** | 2025 | PTQ (data-free) | HunyuanVideo, Wan2.1 | W4A4-W4A8 | First successful W4A4 PTQ for video |
| **QVGen** | 2025 | QAT | CogVideoX, Wan 14B | W4A4, W3A3 | Full-precision comparable W4A4 |
| **FPSAttention** | NeurIPS 2025 Spotlight | Training-aware | Wan2.1 1.3B/14B | FP8+Sparse | 4.96x e2e speedup at 720p |
| **Mixup-Sign (MSFP)** | CVPR 2025 | QAT | Diffusion models | W4A4 FP | First superior 4-bit FP quantization |
| **SageAttention2** | ICLR 2025, ICML 2025 | Attention-only | CogVideoX, HunyuanVideo | INT4 QK, FP8 PV | 3-5x vs FlashAttention2 |
| **SageAttention2++** | ICML 2025 | Attention-only | CogVideoX, HunyuanVideo, Wan | INT4/INT8 QK, FP8 PV | 3.9x vs FA2 |
| **TurboDiffusion** | Dec 2025 | System | Wan2.1, Wan2.2 | W8A8 + SageAttn | 100-200x e2e speedup |
| **TFMQ-DM** | CVPR 2024, TPAMI 2025 | PTQ | UNet/DiT diffusion | Various | Temporal feature maintenance |
| **AccuQuant** | 2025 | PTQ | Diffusion models | Various | Multi-step denoising simulation |

### Survey Paper

- **"Diffusion Model Quantization: A Review"** (arXiv 2505.05215, May 2025): Comprehensive taxonomy covering PTQ/QAT methods, U-Net vs DiT challenges, evaluation methodologies. Highlights that DiTs exhibit substantial temporal and spatial activation variations making quantization uniquely challenging vs LLMs.

---

## 7. GGUF Quantization for Diffusion Models

### How GGUF Applies to Diffusion Models

GGUF (GPT-Generated Unified Format) was originally designed for LLMs but has been adapted for diffusion transformers:

**Why it works for DiTs but not UNets**: Transformer/DiT architectures are significantly more resilient to quantization than convolutional UNet models. ComfyUI-GGUF notes: "quantization wasn't feasible for regular UNET models (conv2d), transformer/DiT models such as FLUX seem less affected."

**Block-based quantization mechanism**:
1. Weight matrix divided into blocks (typically 32 weights each)
2. Per-block scale and minimum computed independently
3. Each weight quantized relative to its block's statistics
4. Memory-mapped loading + on-demand dequantization minimizes peak VRAM

**Available GGUF quantization levels**:

| Type | Bits/weight | Use Case |
|------|-------------|----------|
| Q3_K | ~3.4 | Minimum VRAM (quality loss) |
| Q4_K_S | ~4.5 | Good balance for consumer GPUs |
| Q5_K_S | ~5.5 | Higher quality |
| Q6_K | ~6.6 | Near full precision |
| Q8_0 | ~8.5 | Minimal quality loss |
| BF16 | 16 | Full precision reference |

### GGUF for Video Models

**Available models**:
- HunyuanVideo GGUF (city96): Q3 through BF16 variants
- FastHunyuan GGUF (city96): Distilled + quantized, 6-step generation
- SkyReels-V2 GGUF (community): I2V 1.3B 540P
- Wan2.1/2.2 GGUF variants (community)

**Practical example**: FastHunyuan Q4_K_S with 6 steps on laptop RTX 4060 8GB: ~2 minutes for video generation (vs ~6 minutes for original Hunyuan Q4_K_S at 20 steps)

### FastVideo and GGUF

FastVideo (Hao AI Lab, UCSD) borrows architectural design from vLLM (acknowledged in their codebase) but **does not currently implement GGUF support**. FastVideo's quantization features are limited to:
- FP8 T5 encoder support
- VAE precision control (FP32 override)
- BitsAndBytes integration for memory-constrained inference
- The `--quant_linear` flag in TurboDiffusion (built on FastVideo patterns)

FastVideo's primary acceleration comes from SageAttention + TeaCache rather than weight quantization. GGUF integration would need to come from community effort or the ComfyUI-GGUF ecosystem.

### How GGUF Differs from Other Quantization

| Aspect | GGUF | TensorRT FP8 | torchao INT8 |
|--------|------|-------------|-------------|
| Runtime | CPU or GPU (mmap) | GPU only (compiled) | GPU (PyTorch) |
| Compilation | None needed | Engine build required | torch.compile optional |
| Flexibility | Any quant level | FP8/INT8 only | INT8/FP8/INT4 |
| Speed | Slower (dequant overhead) | Fastest | Medium |
| VRAM | Lowest (mmap offload) | GPU-bound | GPU-bound |
| Best for | Consumer GPUs (<16GB) | Production serving | Research/prototyping |

---

## 8. Practical Deployment Numbers

### HunyuanVideo

| Configuration | VRAM | Speed (H100) | Quality |
|--------------|------|--------------|---------|
| BF16 (baseline) | ~45GB | 37.6s | Reference |
| FP8 (E4M3) | ~14-15GB | ~25s | Minimal loss |
| FP8 + SageAttn + Compile | ~14GB | 18.8s (50% faster) | Minimal loss |
| GGUF Q4_K_S | ~8GB (mmap) | Slower | Some degradation |
| DVD-Quant W4A8 | ~12GB | 1.75x faster | VBench aesthetic 61.96 |
| DVD-Quant W4A4 | ~10GB | 2.12x faster | VBench aesthetic 61.96 (vs competitors 24-48) |
| DVD-Quant W4A4 + TeaCache | ~10GB | 4.85x faster | Maintained |

### Wan2.1

| Configuration | Model | VRAM | Speed | Quality |
|--------------|-------|------|-------|---------|
| BF16 baseline | 1.3B 480P | 8.19GB | ~4min (RTX4090) | Reference |
| BF16 baseline | 14B 480P | ~45GB | 1,676s | Reference |
| FPSAttention (FP8+sparse) | 14B 720P | ~22GB | 4.96x faster | PSNR 25.74 |
| TurboDiffusion (W8A8+SageAttn+distill) | 1.3B | <24GB | 1.9s (184x) | Maintained |
| TurboDiffusion (W8A8+SageAttn+distill) | 14B 480P | <40GB | 9.9s (169x) | Maintained |
| Attention Surgery (40% FLOPs reduction) | 1.3B | ~5GB | Faster | Negligible VBench drop |

### CogVideoX

| Configuration | Model | Memory Saving | Speedup | Quality |
|--------------|-------|---------------|---------|---------|
| W8A8 (ViDiT-Q style) | 2B | 2-2.5x | 1.4-1.7x | Lossless |
| W4A4 (QVGen) | 2B | ~4x | 1.21-1.44x | +25.28 Dynamic Degree, -4.82 Scene Consistency |
| W4A4 (QVGen) | 1.5-5B | ~4x | Similar | Full-precision comparable |
| INT8 (torchao) | 5B (A100) | ~2x | 27.33% speedup (compiled) | Minimal loss |
| SageAttention2 | 2B | N/A (attn only) | 1.8x | VQA-A 77.37 vs 77.61 |
| SageAttention2 | 1.5-5B | N/A | 99.45% cosine sim | Negligible loss |

### FLUX (Image, for Reference)

| Configuration | GPU | Memory | Speedup vs H200 BF16 |
|--------------|-----|--------|----------------------|
| BF16 baseline | H200 | ~31GB | 1x |
| FP8DQRow + compile | H100 | ~20GB | 2.2x |
| NVFP4 + TeaCache | B200 | ~9GB | 6.3x |
| NVFP4 + TeaCache + 2xB200 | 2xB200 | ~9GB/GPU | 10.2x |
| SVDQuant W4A4 (Nunchaku) | RTX4090 Laptop | ~9GB | 3.0x vs W4A16 |

### Cross-Model Summary: What Works Best

| Goal | Recommended Approach | Expected Gain |
|------|---------------------|---------------|
| Quick win, production | FP8 E4M3 (TensorRT/torchao) | 1.5-2.5x speed, ~2x memory |
| Attention speedup | SageAttention2++ | 2-4x attention speed, plug-and-play |
| Maximum compression | DVD-Quant W4A4 (PTQ, data-free) | 3.7x memory, 2x speed |
| Best quality at 4-bit | QVGen W4A4 (QAT) | 4x memory, maintained quality |
| Extreme e2e speed | TurboDiffusion (W8A8+SageAttn+distill) | 100-200x total |
| Consumer GPU (<16GB) | GGUF Q4_K + offloading | Fits in 8-12GB |
| Blackwell GPUs | NVFP4 | 3.5x memory, 6-10x speed |

---

## Key Findings and Recommendations

### 1. The Quantization Landscape is Rapidly Maturing
Video diffusion model quantization has moved from "barely possible" in 2024 to "production-ready at W8A8" and "research frontier at W4A4" in 2025-2026.

### 2. FP8 is the Safe Default
FP8 E4M3 provides the best quality-performance tradeoff for production deployment. It is supported natively on Hopper/Ada GPUs, requires minimal calibration, and achieves 1.5-2.5x speedup with negligible quality loss.

### 3. Attention Quantization is Complementary
SageAttention2++ is orthogonal to weight quantization and provides additional 2-4x speedup on attention. It should be combined with weight quantization for maximum benefit.

### 4. W4A4 is Now Viable for Video
DVD-Quant (data-free PTQ) and QVGen (QAT) have both demonstrated successful W4A4 quantization for video models with maintained VBench scores. This was not possible a year ago.

### 5. System-Level Optimization Dominates
TurboDiffusion shows that combining quantization (W8A8) + attention acceleration (SageAttention) + step distillation (rCM, 3-4 steps) + sparse attention achieves 100-200x speedup. No single technique achieves this alone.

### 6. NVFP4 is the Next Frontier
Currently validated for image diffusion (FLUX), NVFP4 on Blackwell promises 3.5x memory reduction and massive throughput gains. Video model support is expected soon.

### 7. GGUF Fills the Consumer GPU Niche
For sub-16GB VRAM scenarios, GGUF with memory-mapped loading remains the most practical approach, despite lower throughput than compiled solutions.

---

## Sources

### Papers
- [ViDiT-Q (ICLR 2025)](https://arxiv.org/abs/2406.02540)
- [Q-DiT (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Q-DiT_Accurate_Post-Training_Quantization_for_Diffusion_Transformers_CVPR_2025_paper.pdf)
- [SVDQuant (ICLR 2025 Spotlight)](https://arxiv.org/abs/2411.05007)
- [DVD-Quant (2025)](https://arxiv.org/html/2505.18663)
- [QVGen (2025)](https://arxiv.org/html/2505.11497)
- [FPSAttention (NeurIPS 2025 Spotlight)](https://arxiv.org/abs/2506.04648)
- [SageAttention2 (ICLR 2025)](https://arxiv.org/abs/2411.10958)
- [SageAttention2++ (ICML 2025)](https://arxiv.org/abs/2505.21136)
- [TurboDiffusion (Dec 2025)](https://arxiv.org/abs/2512.16093)
- [Mixup-Sign FP4 (CVPR 2025)](https://arxiv.org/abs/2505.21591)
- [Diffusion Model Quantization Survey (May 2025)](https://arxiv.org/abs/2505.05215)
- [NVFP4 Block Scaling Paper](https://arxiv.org/pdf/2512.02010)

### Blog Posts and Documentation
- [NVIDIA TensorRT for Video Diffusion (Adobe Firefly)](https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/)
- [NVIDIA NVFP4 Introduction](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference)
- [NVIDIA NVFP4 for FLUX.2 on Blackwell](https://developer.nvidia.com/blog/scaling-nvfp4-inference-for-flux-2-on-nvidia-blackwell-data-center-gpus/)
- [diffusers-torchao Benchmarks](https://github.com/sayakpaul/diffusers-torchao)

### Code Repositories
- [ViDiT-Q](https://github.com/thu-nics/ViDiT-Q)
- [SageAttention](https://github.com/thu-ml/SageAttention)
- [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion)
- [SVDQuant / Nunchaku](https://github.com/nunchaku-ai/nunchaku)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [SkyReels-V1](https://github.com/SkyworkAI/SkyReels-V1)
