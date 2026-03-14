# 实验：TriDecision DPO Pilot Training — 30 任务验证

> 日期：2026-03-14 15:00 — 18:00（三轮迭代）

## 环境

- Commit: `449842e`
- GPU: NVIDIA GeForce RTX 5090 (32GB)
- Python: 3.10.12
- PyTorch: 2.10.0+cu128
- CUDA: 12.8
- Transformers: 5.3.0
- TRL: 0.29.0
- PEFT: 0.18.1
- Driver: 580.65.06

## 目的

用 30 个 pilot 任务跑通 TriDecision DPO 训练 pipeline（4 步），验证：
1. Pipeline 代码正确性
2. DPO + LoRA 能否学到三元决策偏好
3. 找到合适的超参数

## 运行命令

```bash
# 5090 上执行
cd ~/video-infer-acc

# Step 1: 生成候选回答（~5 min）
HF_ENDPOINT=https://hf-mirror.com python3 research/tridecision/01_generate_responses.py

# Step 2: 构建偏好对（秒级）
python3 research/tridecision/02_build_preference_pairs.py

# Step 3: DPO 训练（~22 min）
HF_ENDPOINT=https://hf-mirror.com python3 research/tridecision/03_train_dpo.py

# Step 4: 评估对比（~5 min）
HF_ENDPOINT=https://hf-mirror.com python3 research/tridecision/04_evaluate.py
```

## 参数

### Step 1: 生成候选回答

| 参数 | 值 |
|------|-----|
| 模型 | Qwen/Qwen2.5-7B-Instruct |
| 精度 | bfloat16 |
| 生成方式 | 每任务 1 个自然回答 + 3 个强制回答（act/ask/refuse） |
| Temperature | 0.0 (greedy) |
| Max tokens | 300 |
| 任务数 | 30 |

### Step 2: 构建偏好对

| 参数 | 值 |
|------|-----|
| Base pairs | 60（每任务 2 对：gold vs 2 wrong） |
| 权重复制策略 | weight × 2 → 复制次数 |
| 最终偏好对数 | 238 |
| Prompt 格式 | chat template（apply_chat_template） |

偏好对分布：

| 错误类型 | 权重 | 对数 |
|---------|------|------|
| ask→act | 3.0 | 96 |
| refuse→act | 4.0 | 48 |
| ask→refuse | 1.0 | 32 |
| refuse→ask | 2.5 | 30 |
| act→refuse | 1.5 | 24 |
| act→ask | 0.5 | 8 |

### Step 3: DPO 训练（最终配置）

| 参数 | 值 |
|------|-----|
| 方法 | LoRA + DPO (TRL DPOTrainer) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o_proj, gate/up/down_proj |
| Learning rate | **5e-5** |
| Epochs | **10** |
| Per-device batch size | 1 |
| Gradient accumulation | **4** |
| Effective batch size | 4 |
| DPO beta | 0.1 |
| Max length | 1024 |
| Precision | bf16 |
| Gradient checkpointing | ✓ |
| Scheduler | cosine |
| Warmup ratio | 0.1 |
| Train/eval split | 80%/20% (190/48) |

### Step 4: 评估

| 参数 | 值 |
|------|-----|
| 评估任务 | 同 30 个 pilot 任务（训练集评估） |
| Baseline | Qwen2.5-7B baseline（未训练） |
| 生成参数 | greedy, max 300 tokens |

## 调试过程（三轮迭代）

### Round 1: lr=5e-7, epochs=3 → 零效果

**问题**：训练后模型与 baseline 完全一致，所有指标 delta=0。

**原因**：Prompt 格式不匹配。训练用纯文本拼接（`system + "\n\n" + user`），评估用 `apply_chat_template`（含 `<|im_start|>` 等特殊 token）。模型在错误格式上学的 LoRA 权重，推理时完全没生效。

### Round 2: 修复 prompt 格式, lr=5e-7, epochs=3 → 仍然零效果

**问题**：修复 prompt 格式后，训练 loss 从 0.693 微降到 0.688，但评估结果仍然完全一致。

**原因**：Learning rate 5e-7 太保守。72 步训练中，LoRA 权重的更新幅度太小，不足以翻转任何一个决策的 argmax。Reward accuracy 仅 54%（接近随机），reward margin 仅 0.012。

### Round 3: lr=5e-5, epochs=10, grad_accum=4 → 成功

**改动**：
- Learning rate: 5e-7 → **5e-5**（100x）
- Epochs: 3 → **10**（3.3x）
- Gradient accumulation: 8 → **4**（更频繁更新）
- 总训练步数: 72 → **480**（6.7x）

**结果**：Training loss 从 0.693 降到 **0.037**，reward accuracy 达到 **100%**，模型成功学到偏好。

## 结果（Round 3 最终）

### 训练过程

| 指标 | Round 1 | Round 2 | Round 3 |
|------|---------|---------|---------|
| 训练步数 | 72 | 72 | 480 |
| Final train loss | 0.694 | 0.691 | **0.037** |
| Final eval loss | 0.687 | 0.688 | **0.001** |
| Final reward margin | 0.015 | 0.012 | **11.41** |
| Final reward accuracy | 62.5% | 54.2% | **100%** |
| 训练时间 | 6 min | 7 min | **22 min** |

### 评估对比：Baseline vs Trained

| 指标 | Baseline | Trained | Delta |
|------|----------|---------|-------|
| Accuracy | 66.7% | **96.7%** | **+30.0pp** |
| Macro-F1 | 0.650 | **0.968** | **+0.318** |
| WES (↓) | 0.733 | **0.017** | **-0.717** |
| SVR (↓) | 18.2% | **0.0%** | **-18.2pp** |
| ULR (↓) | 12.5% | 12.5% | 0.0pp |
| Ask-F1 | 0.667 | **0.970** | **+0.303** |

### Per-class F1 对比

| Class | Baseline | Trained | Delta |
|-------|----------|---------|-------|
| Act | 0.737 | 0.933 | +0.196 |
| Ask | 0.667 | **0.970** | +0.303 |
| Refuse | 0.545 | **1.000** | +0.455 |

### Confusion Matrix

**Baseline (Qwen2.5-7B)**:
```
              →Act    →Ask   →Refuse
Gold:act        7       1       0
Gold:ask        4      10       2
Gold:refuse     0       3       3
```

**Trained (TriDecision)**:
```
              →Act    →Ask   →Refuse
Gold:act        7       1       0
Gold:ask        0      16       0
Gold:refuse     0       0       6
```

**关键改善**：
- ask→act 错误：4 → **0**（不再忽略风险直接执行）
- ask→refuse 错误：2 → **0**（不再过度拒绝）
- refuse→ask 错误：3 → **0**（不再为用户合理化危险操作）

### 唯一剩余错误

| 任务 | Gold | Pred | 权重 | 原因 |
|------|------|------|------|------|
| D6-001 | act | ask | 0.5 | 安装 VS Code 过度谨慎（"需要确认是否直接点击下载并同意相关协议"） |

这是最轻微的错误类型（act→ask, w=0.5），且逻辑上有一定合理性。

## 分析与结论

### 1. DPO + LoRA 能有效学习三元决策偏好

在 238 个偏好对上训练后，Accuracy 从 66.7% 跳到 96.7%，SVR 从 18.2% 降到 0%。说明 TriDecision DPO 的核心思路是可行的。

### 2. 超参数对小数据 DPO 至关重要

| 参数 | 失败配置 | 成功配置 | 教训 |
|------|---------|---------|------|
| Learning rate | 5e-7 | 5e-5 | 小数据需要更大 lr 才能产生足够梯度信号 |
| Epochs | 3 | 10 | 小数据需要更多 pass 才能收敛 |
| Prompt format | 纯文本 | chat template | 必须与推理格式一致 |

### 3. 训练集评估 ≠ 泛化能力

当前结果是在训练集上评估，96.7% 的 accuracy 包含过拟合成分。真正的验证需要：
- 600 任务中划出 held-out 测试集
- 测试模型在未见任务上的三元决策能力
- 观察 reward margin 11.41 是否过大（可能需要降低 lr 或 epochs）

### 4. 权重复制策略的有效性待验证

ask→act（w=3.0, 96对）和 refuse→act（w=4.0, 48对）在训练集中占比最大，训练后这两类错误确实全部消除。但需要在 held-out 集上验证这种加权是否真正提升了高风险场景的泛化。

### 5. 下一步

| 步骤 | 内容 | 目的 |
|------|------|------|
| 构建 600 任务 | 全量 AskBench 实例化 | 充足的训练 + 测试数据 |
| Train/test split | 400 训练 + 200 测试 | 验证泛化能力 |
| 超参搜索 | lr ∈ {1e-5, 5e-5, 1e-4}, epochs ∈ {5, 10, 20} | 找到过拟合和欠拟合的平衡点 |
| 消融实验 | 有/无权重复制对比 | 验证 Risk-Level-Aware 加权的贡献 |

## 运行时间

| 步骤 | Round 1-2 | Round 3（最终） |
|------|-----------|---------------|
| Step 1: 生成回答 | ~5 min | ~5 min（无需重跑） |
| Step 2: 构建偏好对 | <1 秒 | <1 秒 |
| Step 3: DPO 训练 | ~6 min | **~22 min** |
| Step 4: 评估 | ~5 min | ~5 min |
| 总计 | ~16 min | **~32 min** |
