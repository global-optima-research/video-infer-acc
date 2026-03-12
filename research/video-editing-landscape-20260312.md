# Video Editing with Diffusion Models: Competitive Landscape Analysis
> Date: 2026-03-12
> Goal: Identify genuine gaps for a top-venue paper (NeurIPS 2026 / CVPR 2027 / ICML 2026)

## Executive Summary

Video editing with diffusion models is a **very active** field with 40+ papers in 2025-2026 alone. The field has undergone a major transition from U-Net-based (SD 1.5/SDXL) to DiT-based (Wan 2.1, CogVideoX, HunyuanVideo) architectures. Most sub-directions are crowded, but a few genuine gaps remain -- primarily at the intersection of DiT-native editing with underexplored modalities or in systematic infrastructure that the community lacks.

---

## Paper Inventory (Major Papers, 2024-2026)

### A. Instruction-Based / Text-Guided Video Editing

| Paper | Venue | Date | Base Model | Key Innovation | Training? |
|-------|-------|------|-----------|---------------|-----------|
| **VideoGrain** | ICLR 2025 | Feb 2025 | U-Net (SD) | Multi-grained editing via space-time attention modulation | Training-free |
| **InsViE-1M** | ICCV 2025 | 2025 | U-Net | 1M instruction-editing dataset + model | Trained |
| **Senorita-2M** | NeurIPS 2025 D&B | Feb 2025 | Multiple | 2M video editing pairs, 18 task types | Dataset |
| **VIVA** | arXiv Dec 2025 | Dec 2025 | DiT | VLM-guided encoding + Edit-GRPO reward optimization | Trained |
| **VideoCoF** | CVPR 2026 | Dec 2025 | DiT (Wan-like) | Chain-of-Frames reasoning: see->reason->edit, no masks needed | Trained (50K data) |
| **NOVA** | CVPR 2026 | Mar 2026 | DiT | Sparse keyframe control + dense motion synthesis, pair-free | Trained |
| **AnyEdit** | CVPR 2025 Oral | 2025 | SD | Unified 2.5M dataset, 20+ edit types (images) | Trained |

### B. Training-Free / Inversion-Based Video Editing

| Paper | Venue | Date | Base Model | Key Innovation | DiT-compatible? |
|-------|-------|------|-----------|---------------|----------------|
| **RF-Solver / RF-Edit** | ICML 2025 | Nov 2024 | FLUX, HunyuanVideo | High-order Taylor expansion for rectified flow inversion | Yes (HunyuanVideo) |
| **ContextFlow** | arXiv Sep 2025 | Sep 2025 | DiT | Dual-path KV enrichment for object editing | Yes |
| **IF-V2V** | ICLR 2026 | Sep 2025 | Flow-matching I2V | Inversion-free via vector field rectification | Yes |
| **KV-Lock** | arXiv Mar 2026 | Mar 2026 | DiT | Hallucination detection + dynamic KV locking | Yes (any DiT) |
| **DiTCtrl** | 2025 | 2025 | MM-DiT | Attention control for multi-prompt video gen | Yes |
| **Align-A-Video** | CVPR 2025 | 2025 | Image diffusion | Deterministic reward tuning + feature propagation | U-Net based |
| **Video-3DGS** | 2025 | 2025 | Any | 3D Gaussian Splatting post-refinement for temporal consistency | Model-agnostic |

### C. Video Inpainting / Object Removal

| Paper | Venue | Date | Base Model | Key Innovation |
|-------|-------|------|-----------|---------------|
| **VideoPainter** | SIGGRAPH 2025 | Mar 2025 | Video DiT | Plug-and-play context encoder (6% params), any-length |
| **EraserDiT** | WACV 2026 | Jun 2025 | DiT (3D) | Circular Position-Shift for long-range consistency |
| **DiTPainter** | arXiv 2025 | Apr 2025 | DiT (scratch) | End-to-end DiT inpainting, arbitrary length |
| **DiffuEraser** | arXiv Jan 2025 | Jan 2025 | SD (UNet) | BrushNet auxiliary for masked feature extraction |
| **HomoGen** | CVPR 2025 | 2025 | VDM | Homography propagation for pixel warping priors |

### D. Video Style Transfer

| Paper | Venue | Date | Base Model | Key Innovation |
|-------|-------|------|-----------|---------------|
| **PickStyle** | ICLR 2026 sub | 2025 | Video diffusion | Context-style adapters for V2V style transfer |
| **Sync Multi-Frame Diffusion** | CGF 2025 | 2025 | Image diffusion | Synchronized denoising across frames |
| **CLIPGaussians** | 2025 | 2025 | Gaussian | Unified style across 2D/3D/4D |

### E. Controllable Video Generation/Editing

| Paper | Venue | Date | Key Innovation |
|-------|-------|------|---------------|
| **OmniVDiff** | Apr 2025 | 2025 | Omni-controllable video diffusion |
| **ControlNeXt** | Aug 2024 | 2024 | Efficient control for image/video |
| **Frame Guidance** | Jun 2025 | 2025 | Training-free frame-level control |
| **Motion Prompting** | CVPR 2025 | 2025 | Trajectory control in video diffusion |
| **FreeTraj** | 2024 | 2024 | Tuning-free trajectory via noise + attention |

### F. Efficient / Mobile Video Editing

| Paper | Venue | Date | Key Innovation |
|-------|-------|------|---------------|
| **MoViE** | arXiv Dec 2024 | Dec 2024 | 12 FPS on mobile, CFG distillation + 1-step |
| **AdaFlow** | 2025 | 2025 | Adaptive attention slimming for minute-long videos |

### G. Datasets & Benchmarks

| Resource | Venue | Date | Scale |
|----------|-------|------|-------|
| **Senorita-2M** | NeurIPS 2025 | 2025 | 2M pairs, 18 edit types |
| **InsViE-1M** | ICCV 2025 | 2025 | 1M instruction-editing pairs |
| **VPData/VPBench** | SIGGRAPH 2025 | 2025 | 390K clips with segmentation masks |
| **IVEBench** | arXiv Oct 2025 | 2025 | 600 videos, 8 categories, 35 subcategories |
| **VEditBench** | 2025 | 2025 | 9 evaluation dimensions |
| **VEBench** | 2025 | 2025 | Meta-evaluation of automatic metrics |

---

## Crowdedness Assessment

### CROWDED: Instruction-Based General Video Editing (10+ papers)

**Why crowded:**
- VideoGrain, InsViE-1M, VIVA, VideoCoF, NOVA already cover the core paradigm
- Two massive datasets (Senorita-2M, InsViE-1M) mean data is no longer a bottleneck
- VideoCoF (CVPR 2026) introduced reasoning-based editing -- a strong advance
- NOVA (CVPR 2026) solved pair-free editing with sparse control
- VIVA added reward optimization (Edit-GRPO)
- Multiple benchmarks (IVEBench, VEditBench) exist

**What's left:** Very little room for "just another instruction editor." Any new work must have a fundamentally different angle (e.g., a different modality, a different scale regime, or a different task formulation).

---

### CROWDED: Training-Free / Inversion-Based Editing (8+ papers)

**Why crowded:**
- RF-Solver (ICML 2025) is already the standard for rectified flow inversion on HunyuanVideo/FLUX
- ContextFlow, IF-V2V, KV-Lock all address DiT-based training-free editing
- DiTCtrl handles multi-prompt scenarios
- The space is well-covered for both DDPM and flow-matching paradigms

**What's left:** Minor improvements possible but unlikely to yield a top-venue paper on their own.

---

### CROWDED: Video Inpainting / Object Removal (5+ papers)

**Why crowded:**
- VideoPainter (SIGGRAPH 2025) is very strong: any-length, plug-and-play, DiT-native
- EraserDiT and DiTPainter also cover DiT-based inpainting
- DiffuEraser and HomoGen cover UNet-based approaches
- The VPBench dataset (390K clips) is comprehensive

---

### SEMI-OCCUPIED: Video Style Transfer (3-4 papers, gaps remain)

**Current state:**
- PickStyle (ICLR 2026 submission) covers V2V style with adapters
- Sync Multi-Frame Diffusion handles temporal consistency
- CLIPGaussians does multimodal style

**Remaining gaps:**
- No DiT-native (Wan 2.1 / HunyuanVideo) style transfer method exists yet
- Most work still uses UNet-based or image diffusion backbones
- Reference-image-guided video style (not just text) on DiT models is underexplored
- However, style transfer is generally considered a "low ceiling" topic for top venues

---

### SEMI-OCCUPIED: Controllable Video Editing with Structural Conditions (4-5 papers)

**Current state:**
- ControlNeXt, OmniVDiff, Frame Guidance exist but mostly for generation, not editing
- Motion Prompting and FreeTraj handle trajectory control

**Remaining gaps:**
- ControlNet-style conditioning for DiT-based video *editing* (not generation) is thin
- Multi-condition fusion (depth + edge + pose simultaneously) for DiT video editing is largely unexplored
- But this is dangerously close to "engineering" rather than "research"

---

### SEMI-OCCUPIED: Reward Optimization / Preference Alignment for Video Editing (2-3 papers)

**Current state:**
- VIVA (Edit-GRPO) applies GRPO to video editing
- Align-A-Video applies reward tuning to frame-level editing
- Flow-DPO and Flow-RWR exist for video generation alignment

**Remaining gaps:**
- No systematic reward model specifically for video editing quality (most repurpose generation reward models)
- No DPO/RLHF applied directly to DiT-based video editors
- But VIVA already occupies the core idea

---

### SEMI-OCCUPIED: Long Video Editing (2-3 papers)

**Current state:**
- VideoPainter handles any-length inpainting
- EraserDiT uses Circular Position-Shift for long sequences
- AdaFlow enables minute-long editing via adaptive attention

**Remaining gaps:**
- General-purpose long video editing (not just inpainting) at minute+ scale with DiT models
- Narrative-aware editing (editing that understands scene transitions, plot)
- But long video generation itself is still unsolved, so editing is premature

---

## Genuine Gaps Analysis

### POTENTIAL GAP 1: Video Editing Directly on Modern DiT Models (Wan 2.1 / HunyuanVideo)

**Observation:** Despite 40+ papers, surprisingly few have *actually been validated* on Wan 2.1, CogVideoX, or HunyuanVideo. Most work uses:
- Old U-Net models (SD 1.5, SDXL, AnimateDiff)
- Custom small DiTs trained from scratch

Only RF-Solver explicitly demonstrated HunyuanVideo editing. VIVA and VideoCoF appear to use DiT but with unclear base models. ContextFlow and KV-Lock claim DiT compatibility but it's unclear which specific models.

**Risk assessment:** This is a narrow gap. As the community migrates to DiT models, this will fill rapidly. It's more of a "timing window" than a true research gap. HIGH RISK of being scooped within months.

---

### POTENTIAL GAP 2: Unified Video Editing Framework Across Edit Types on DiT

**Observation:** VideoCoF is the closest -- it does object add/remove/swap/style in one model. But:
- It's trained on only 50K data
- It uses a single DiT architecture
- There's room for a more comprehensive "foundation model for video editing" on top of Wan 2.1/HunyuanVideo that handles all edit types

**Risk assessment:** VideoCoF and NOVA already signal this trend. Any new work would need a clearly superior approach. MEDIUM-HIGH RISK.

---

### POTENTIAL GAP 3: VLM-Grounded Video Editing (Understanding What to Edit)

**Observation:** Most video editors assume the user provides clear instructions or masks. Few leverage VLMs to:
- Automatically understand *what* needs editing from vague instructions
- Ground editing regions via visual reasoning
- Handle complex multi-step or conditional edits ("make it look like it's raining, but only in the outdoor scenes")

**Current competition:** VIVA uses VLM-guided encoding but primarily for instruction parsing, not for spatial/temporal grounding of edits. VideoCoF's "reasoning tokens" are a step in this direction but operate purely in latent space without explicit VLM grounding.

**Risk assessment:** MEDIUM RISK. The intersection of video VLMs (VideoGLaMM, Molmo 2) and video editing is nascent. But the GRAIL-V workshop at CVPR 2026 suggests the community is aware of this direction.

---

### POTENTIAL GAP 4: 3D/Physics-Consistent Video Editing

**Observation:** Current methods edit in 2D pixel/latent space. None ensure:
- 3D geometric consistency of edits (edited object respects perspective, occlusion)
- Physical plausibility (edited object follows gravity, collision)
- Lighting consistency (shadows, reflections update correctly)

**Current competition:** Video-3DGS uses 3DGS for post-hoc temporal consistency but doesn't address 3D-aware editing. GEN3C (CVPR 2025) does 3D-consistent video generation with camera control but not editing.

**Risk assessment:** This is a genuine gap but *extremely hard* technically. Would require combining video diffusion with 3D reconstruction. MEDIUM RISK of competition (because it's hard), but HIGH RISK of not working.

---

### POTENTIAL GAP 5: Interactive / Iterative Video Editing

**Observation:** All current methods are "one-shot": provide instruction, get result. None support:
- Multi-turn editing dialogues
- Progressive refinement based on user feedback
- Undo/redo with selective rollback
- Real-time preview during editing

**Current competition:** Essentially zero papers on interactive video editing with diffusion models.

**Risk assessment:** LOW RISK of competition. But extremely challenging engineering problem. May not yield sufficient "research contribution" for a top venue -- reviewers may see it as a system paper.

---

### POTENTIAL GAP 6: Audio-Synchronized / Multi-Modal Video Editing

**Observation:** No work on editing video content to match audio cues (e.g., "add explosions synced to the bass drops") or vice versa. Wan 2.1 has video-to-audio capability, but editing in the audio-visual joint space is unexplored.

**Risk assessment:** VERY LOW competition. Novel and interesting. But may be seen as too niche for a top venue unless framed carefully. The data collection problem is non-trivial.

---

### POTENTIAL GAP 7: Efficient/Compressed Video Editing for Deployment

**Observation:** MoViE (mobile editing) was the only work on efficiency, and it was withdrawn from ICLR 2026. The gap between research models and deployable editing systems remains wide.

**Risk assessment:** LOW NOVELTY for top venues. Better suited for industry/system venues (MLSys, OSDI) than CV venues.

---

## Honest Assessment: Is There a Genuine Top-Venue Paper Here?

**The uncomfortable truth:** Video editing with diffusion models is a **red ocean** in 2026. The field has:
- 2 dedicated surveys
- 3 large-scale datasets (1M-2M+ pairs)
- 5+ benchmarks
- Papers at every top venue (ICLR, CVPR, ICML, NeurIPS, SIGGRAPH)
- Rapid migration to DiT models already underway

**The strongest remaining angle is Gap 3 (VLM-Grounded Editing)** -- leveraging video understanding models to make editing more intelligent rather than just more capable. This is where the community hasn't fully converged, and it's at the intersection of two hot areas (video VLMs + video editing).

**Gap 4 (3D-Consistent Editing)** is interesting but technically very challenging and may require foundational advances in video 3D understanding first.

**Everything else carries significant scooping risk** because the community is extremely active and the field is mature enough that incremental advances won't clear the bar.

### Recommendation

Before committing to video editing as a research direction, consider:
1. The field is far more crowded than video inference acceleration was
2. Top groups (Tencent ARC, WeChat CV, Alibaba) have massive compute and data advantages
3. The DiT migration window is closing fast -- what's novel today won't be in 3 months
4. Any gap identified here could be filled by a concurrent submission

**If proceeding, the safest bet is Gap 3** with a concrete, well-scoped formulation that clearly differentiates from VIVA and VideoCoF.

---

## Sources

- [InsViE-1M (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_InsViE-1M_Effective_Instruction-based_Video_Editing_with_Elaborate_Dataset_Construction_ICCV_2025_paper.pdf)
- [VIVA: VLM-Guided Video Editing](https://arxiv.org/abs/2512.16906)
- [VideoGrain (ICLR 2025)](https://openreview.net/forum?id=SSslAtcPB6)
- [VideoCoF (CVPR 2026)](https://arxiv.org/abs/2512.07469)
- [NOVA (CVPR 2026)](https://arxiv.org/abs/2603.02802)
- [RF-Solver (ICML 2025)](https://arxiv.org/abs/2411.04746)
- [ContextFlow](https://arxiv.org/abs/2509.17818)
- [IF-V2V (ICLR 2026)](https://openreview.net/forum?id=ITvVX8jaOM)
- [KV-Lock](https://arxiv.org/abs/2603.09657)
- [VideoPainter (SIGGRAPH 2025)](https://arxiv.org/abs/2503.05639)
- [EraserDiT](https://arxiv.org/abs/2506.12853)
- [DiTPainter](https://arxiv.org/pdf/2504.15661)
- [DiffuEraser](https://arxiv.org/html/2501.10018)
- [HomoGen (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Ding_HomoGen_Enhanced_Video_Inpainting_via_Homography_Propagation_and_Diffusion_CVPR_2025_paper.pdf)
- [PickStyle (ICLR 2026)](https://openreview.net/forum?id=NRWI7NRaFD)
- [Align-A-Video (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Align-A-Video_Deterministic_Reward_Tuning_of_Image_Diffusion_Models_for_Consistent_CVPR_2025_paper.pdf)
- [Senorita-2M (NeurIPS 2025)](https://arxiv.org/abs/2502.06734)
- [MoViE](https://arxiv.org/abs/2412.06578)
- [AdaFlow](https://openreview.net/forum?id=yP0iKsinmk)
- [IVEBench](https://arxiv.org/abs/2510.11647)
- [Controllable Video Generation Survey](https://arxiv.org/html/2507.16869v2)
- [Diffusion Model-Based Video Editing Survey](https://arxiv.org/abs/2407.07111)
- [DiTCtrl](https://onevfall.github.io/project_page/ditctrl/)
- [AnyEdit (CVPR 2025 Oral)](https://github.com/DCDmllm/AnyEdit)
- [Analysis of Attention in Video DiTs](https://arxiv.org/abs/2504.10317)
