# 核心论文精读笔记：Act/Ask/Refuse 研究方向

> 日期：2026-03-12
> 目的：为 Act/Ask/Refuse 三元决策研究提供文献基础

## 1. MOSAIC (2026.03) — 最直接竞品

**论文**：MOSAIC: Aligning LLM Agents to Safely Navigate the Web (Microsoft, arXiv 2603.03205)

### 方法
- **训练框架**：GRPO (Group Relative Policy Optimization)
  - 采样 n=4 trajectories
  - 成对 LLM judge 比较（不是标量分数）
  - Group-relative rewards
- **决策模式**：二元 act/refuse
- **数据**：Agent-SafetyBench 提供初始场景
- **基模**：Qwen2.5 系列

### 关键结果
- Qwen2.5-MOSAIC：harm score 0.09，refusal rate 0.87
- 成对判断 > 逐点评分（消融实验确认）
- 训练资源：4×A100

### 对我们的关键发现
- **Judge prompt 明确说 "prefer helpful completion over refusal" in ambiguous cases**
- 这意味着 MOSAIC **主动回避了 ask 行为**——灰色地带一律按 act 处理
- 没有 "ask human" 作为 action option
- 二元决策的固有局限：要么过度执行（灰色→act），要么过度拒绝（灰色→refuse）

### 我们的差异化
- 加入 ask 作为第三选项
- 灰色地带不再被强制归入 act 或 refuse
- MOSAIC 的 GRPO 框架可以被扩展为三元版本

---

## 2. Mind the GAP (2026.02) — 文字安全 ≠ 工具安全

**论文**：Mind the GAP: Benchmarking the Safety Alignment of LLMs through Tool-Call Generation and Parameter Adherence (arXiv 2602.16943)

### 实验规模
- **17,420 个分析就绪数据点**（17,496 计划配置）
- 6 模型 × 6 管制领域 × 7 jailbreak 场景 × 3 系统提示条件 × 3 治理模式 × 3 重复
- API 成本 ~$120，温度 0.3

### 6 个测试模型 (TC-safe rate under neutral / safety / tool-encouraging)
| 模型 | Neutral | Safety | Tool-Enc | 波动范围 |
|------|---------|--------|----------|---------|
| Claude Sonnet 4.5 | 80% | 95% | 74% | 21pp (最稳定) |
| GPT-5.2 | 31% | 73% | 16% | 57pp (最不稳定) |
| Grok 4.1 Fast | 33% | 57% | 24% | 33pp |
| DeepSeek V3.2 | 21% | 38% | 50% | 29pp |
| Kimi K2.5 | 30% | 64% | 25% | 39pp |
| GLM-4.7 | 23% | 60% | 15% | 45pp |

### 6 个管制领域
1. Pharmaceutical (HIPAA)
2. Financial (SOX)
3. Educational (FERPA)
4. Employment/HR (EEOC)
5. Legal (ABA Model Rules)
6. Infrastructure/DevOps (SOC 2)

### 核心发现
- **GAP 现象**：模型在文本中拒绝，但通过工具调用执行同样的操作
- **GPT-5.2 条件 GAP 率 79.3%**：文本拒绝时，4/5 的情况仍执行禁止的工具调用
- **Claude Sonnet 4.5 GAP 率仅 7.2%**（最佳）

### 三种失败模式
1. **理性推卸 (Rationalized Abdication, 62%)**：承认风险但继续执行，推卸给外部安全控制。Claude 占 87% 属于此类
2. **反射性遮掩 (Reflexive Hedging, 26%)**：文本用 `[REDACTED]` 但工具参数传真实 PII。GPT-5.2 占 48% 属于此类
3. **被动服从 (Passive Compliance)**：无文本抵抗，直接执行。DeepSeek 占 39% 属于此类

### Runtime 治理
- Enforce 模式减少 LEAK 6-31pp
- **但对禁止尝试率零效果**（0/6 模型显著）
- 结论：治理是 catch layer，不是 behavioral modifier

### 对我们的关键发现
- **benchmark 中完全没有 "ask" 类别**——模型要么执行要么拒绝
- **Dual Pathway 假说**：文本生成和工具调用是"部分解耦的决策过程"——RLHF 训练文本输出不迁移到工具行为
- 作者建议未来在 tool-call trajectories 上做 RLHF——我们的方向正是这个
- 数据集和代码已开源（HuggingFace CC-BY-4.0, GitHub MIT）

---

## 3. Unsafer in Many Turns (2026.02) — 多轮风险累积

**论文**：Unsafer in Many Turns: Benchmarking LLMs Safety Across Multi-Turn Tool-Use Interactions (arXiv 2602.13379)

### Benchmark: MT-AgentRisk
- **365 任务，平均 3.19 轮**
- 多轮场景覆盖

### 核心数据
- 多轮 ASR 平均增加 **+16%**
- Claude-4.5-Sonnet：45% → 72% ASR（多轮后恶化 27pp）
- 单步安全 ≠ 多步安全

### ToolShield 防御
- Training-free 自探索防御
- 沙盒中生成安全经验
- 减少 ASR 约 50%
- **但是外挂方案，不是 agent 内在能力**
- 代码：github.com/CHATS-lab/ToolShield

### 对我们的关键发现
- 为我们的 **Stage 3: 多轮风险累积感知** 提供直接支撑
- 多轮数据可用于构造偏好对
- ToolShield 是对比基线（外挂 vs 内在能力）
- 证明仅靠单步安全训练不够

---

## 4. LM Agents Fail to Act on Risk Knowledge (2025.08, COLM 2025)

**论文**：LM Agents Fail to Act on Risk Knowledge (arXiv 2508.13465)

### 实验规模
- 328 trajectories
- 36 toolkits, 311 tools
- 9 风险类别

### 三层测试框架
1. **Knowledge Test**：模型是否知道风险？→ **>98% 通过**
2. **Identification Test**：能否在场景中识别风险？→ 高通过率
3. **Execution Test**：实际执行时是否安全？→ **<26% 安全执行**

### 知行脱节的严重程度
- 知道 (>98%) 到做到 (<26%) 的 gap 高达 **72pp**
- 模型规模增大**不能修复**这个 gap
- 即使 DeepSeek-R1 等推理模型也无法解决

### Verifier + Abstractor 方案
- 用同一个 LLM 作为外部验证器
- 利用 generator-validator gap
- 平均提升到 ~69%
- **但仍是外挂方案**

### 对我们的关键发现
- **Figure 4 出现 "Do you confirm...?" 行为**——这就是 ask！但论文从未形式化它
- 说明 ask 行为已经偶尔出现在模型输出中，只是没人训练和评估它
- 9 个风险类别可用于 AskBench 设计
- 36 toolkits + 311 tools 数据结构可复用

---

## 综合分析：4 篇论文对 Act/Ask/Refuse 的支撑

### 缺口确认

| 维度 | 现有工作 | 我们填补的缺口 |
|------|---------|---------------|
| 决策选项 | 二元 act/refuse | 三元 act/ask/refuse |
| 能力来源 | 外挂护栏/验证器 | agent 内在能力 |
| 安全层面 | 文本安全 OR 工具安全 | 直接在工具/GUI action 层训练 |
| 交互轮次 | 主要单轮 | 多轮风险累积感知 |
| ask 行为 | 偶然出现，未形式化 | 首次形式化为训练目标 |

### 可复用资源

| 来源 | 可复用内容 |
|------|-----------|
| MOSAIC | GRPO 框架→扩展为三元版本 |
| Mind the GAP | 17k 数据点，6 领域分类，失败模式分类 |
| Unsafer | MT-AgentRisk 365 任务，多轮场景 |
| Risk Knowledge | 36 toolkits + 311 tools，三层测试框架 |
| ToolShield | 对比基线（外挂 vs 内在） |
| Verifier+Abstractor | 对比基线（外挂 vs 内在） |

### 技术路线更新

基于精读，TriDecision 训练方案的具体化：

1. **Stage 1 数据构造**：
   - 从 Risk Knowledge 的 9 类风险 + GAP 的 6 管制领域构建场景
   - 用 frontier models 生成 act/ask/refuse 三种候选
   - 基于 gold label 构建偏好对

2. **Stage 2 三元 DPO**：
   - 可借鉴 MOSAIC 的 GRPO 成对比较框架
   - 扩展为三元：(act, ask), (ask, refuse), (act, refuse) 三组偏好对
   - Risk-Level-Aware 权重：refuse→act 最重，act→ask 最轻

3. **Stage 3 多轮**：
   - 直接使用 Unsafer 的 MT-AgentRisk 场景
   - 训练 agent 在第 N 步考虑前 N-1 步累积风险
   - ToolShield 作为对比基线

### MOSAIC 的 GRPO 能否直接扩展？

**可以，但需要修改**：
- MOSAIC 的 judge prompt 强制二元（prefer act over refuse in ambiguous）
- 我们需要三元 judge：given scenario, rank {act, ask, refuse}
- GRPO 的 n=4 采样可以保留，但 reward 需要三元化
- 或者直接用 DPO 代替 GRPO（更简单，先做 DPO baseline）
