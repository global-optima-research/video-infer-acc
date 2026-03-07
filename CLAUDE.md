# Video Inference Acceleration Research

## 项目概述

视频模型推理加速技术调研与研究项目。当前研究方向：VideoFlashAttention（统一视频注意力加速框架）。

## GPU 服务器

- 连接方式：`ssh 5090`
- 硬件：RTX 5090
- 用途：所有实验在此机器上运行

## 实验工作流

1. **本地写代码** → commit & push
2. **SSH 到 5090** → git pull → 运行实验
3. **记录结果** → 实验报告写入 `experiments/` 文件夹

实验报告命名格式：`{实验名}_{YYYYMMDD_HHMM}.md`
例如：`attention_profiling_20260307_1430.md`

### 实验报告必须包含的内容

每份实验报告必须包含以下信息以确保**完整可复现**：

1. **代码版本**：git commit hash（`git rev-parse HEAD`）
2. **运行命令**：完整的命令行，包括所有参数
3. **环境信息**：Python 版本、关键依赖版本（torch、diffusers 等）、CUDA 版本、GPU 型号
4. **实验参数**：模型名、分辨率、帧数、prompt 列表、随机种子等
5. **运行时间**：开始时间、结束时间、总耗时
6. **结果数据**：定量结果（表格/数值）、关键可视化
7. **结论**：对结果的分析和下一步建议

报告模板：
```markdown
# 实验：{实验名}
> 日期：{YYYY-MM-DD HH:MM}

## 环境
- Commit: `{hash}`
- GPU: {型号}
- Python: {版本}
- PyTorch: {版本}
- CUDA: {版本}
- 其他依赖: ...

## 运行命令
\`\`\`bash
{完整命令}
\`\`\`

## 参数
{参数表}

## 结果
{数据/图表}

## 分析与结论
{分析}
```

## 项目结构

```
├── CLAUDE.md                  ← 本文件
├── README.md                  ← 调研总览
├── docs/                      ← 6 个方向的调研文档
│   ├── 01-step-distillation.md
│   ├── 02-feature-caching.md
│   ├── 03-token-pruning-sparse-attention.md
│   ├── 04-quantization.md
│   ├── 05-vae-pipeline-optimization.md
│   ├── 06-hardware-deployment.md
│   ├── 07-research-opportunities.md
│   ├── 08-ar-vs-dit-future.md
│   ├── fastgen-vs-fastvideo.md
│   └── panorama.md
├── research/video-attention/  ← 当前研究方向
│   ├── 00-overview.md
│   ├── 01-structural-properties.md
│   ├── 02-existing-methods.md
│   ├── 03-problem-formulation.md
│   └── 04-experiment-plan.md
├── experiments/               ← 实验报告（自动生成）
└── panorama.html              ← 技术全景可视化
```
