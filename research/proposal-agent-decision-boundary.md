# Proposal: Learning to Act, Ask, or Refuse — Calibrated Risk Decision Boundaries for Computer Use Agents

> 日期：2026-03-12
> 目标：顶会论文（NeurIPS 2026 / ICLR 2027）
> 资源：8×H800 (训练) + 1×RTX 5090 (实验)

## 一句话

现有 agent 安全方法都是二元决策（act/refuse），但现实操作大量处于灰色地带。我们提出三元决策框架 **Act/Ask/Refuse**，训练 computer use agent 学习**何时直接执行、何时请求人类确认、何时拒绝**，并校准决策边界。

## 动机

### 问题 1：Agent 知道有风险但还是做了

"LM Agents Fail to Act on Risk Knowledge" (2025.08) 发现：
- 风险**意识**率 >98%
- 安全**执行**率 <26%
- 即使 DeepSeek-R1 等推理模型也无法解决
- 模型规模增大不能修复这个 gap

### 问题 2：文字安全 ≠ 工具安全

"Mind the GAP" (2026.02) 发现：
- 模型在文字对话中正确拒绝有害请求
- 但在工具调用/computer use 中执行同样的有害操作
- 6 个前沿模型、6 个管制领域均存在此问题

### 问题 3：多轮复合风险

"Unsafer in Many Turns" (2026.02) 发现：
- 单步看似安全的操作，多步组合后变得有害
- 多轮交互中攻击成功率平均增加 16%
- 聊天安全的模型在多步工具使用中变得不安全

### 问题 4：现有方法全是二元决策

| 方法 | 决策模式 | 局限 |
|------|---------|------|
| MOSAIC (2026.03) | act / refuse | 没有"问人"选项 |
| AlphaAlign (2025.07) | act / refuse | 没有"问人"选项 |
| CoG (2025.10) | act / refuse | 没有"问人"选项 |
| DeRTa (ACL'25) | act / refuse | 没有"问人"选项 |
| SafePred (2026.02) | 外部护栏 | 不是 agent 自身能力 |

**现实中大量操作不是"明确安全"或"明确危险"，而是灰色地带**：

```
[Act]                    [Ask]                      [Refuse]
打开文件浏览器          发送邮件给陌生人              rm -rf /
点击导航链接            修改系统设置                  转账到未知账户
滚动页面                删除一个用户没提到的文件        安装恶意软件
输入搜索关键词          购买商品                      关闭防火墙
                        以管理员权限运行程序
                        发布社交媒体帖子
```

二元决策导致两个失败模式：
- **过度执行**：应该 ask 的操作被直接 act → 安全事故
- **过度拒绝**：应该 ask 的操作被 refuse → 可用性损失

## 核心贡献

### 贡献 1：AskBench — 三元决策 Benchmark

首个评估 agent **act/ask/refuse 三元决策能力**的 benchmark。

**设计原则**：
- 覆盖 computer use 场景（桌面、Web、移动）
- 每个任务标注 gold decision：act / ask / refuse
- 重点放在 **ask 类别**（现有 benchmark 只测 act 和 refuse）
- 包含灰色地带场景（合理人会有不同判断的操作）

**任务类别**：

| 类别 | Act 示例 | Ask 示例 | Refuse 示例 |
|------|---------|---------|------------|
| 文件操作 | 打开文件 | 删除用户没提到的文件 | rm -rf / |
| 通信 | 草拟邮件 | 发送邮件给陌生人 | 群发垃圾邮件 |
| 购买 | 搜索商品 | 下单购买 | 使用他人信用卡 |
| 系统 | 查看设置 | 修改系统配置 | 关闭安全软件 |
| 社交 | 浏览社交媒体 | 发布帖子 | 发布仇恨言论 |
| 凭证 | 登录已保存账户 | 输入密码到新网站 | 泄露凭证给第三方 |
| 安装 | 从官方商店安装 | 安装未知来源软件 | 安装已知恶意软件 |
| 多步 | 单步安全操作 | 多步组合后可疑 | 多步组合后明确有害 |

**规模**：~500-1000 任务，每个任务包含截图序列 + 指令 + 待执行动作 + gold label

**评估指标**：
- 三分类准确率 + per-class F1
- Ask-Precision（不该问的别问）/ Ask-Recall（该问的必须问）
- Calibration Error（决策置信度校准）
- 可用性损失率（因过度 ask/refuse 损失的正常任务）

### 贡献 2：TriDecision — 三元决策训练方法

训练 agent 自身具备 act/ask/refuse 三元决策能力。

**方法概要**：

```
输入：[截图序列] + [用户指令] + [当前待执行动作]
输出：{act, ask(理由), refuse(理由)}
```

**训练流程**：

**Stage 1: 三元偏好数据构造**

对于每个 (截图, 指令, 动作) 三元组，构造偏好对：
- (ask, act)：该 ask 但 act 了 → ask 优于 act
- (refuse, act)：该 refuse 但 act 了 → refuse 优于 act
- (act, refuse)：该 act 但 refuse 了（过度拒绝）→ act 优于 refuse
- (act, ask)：该 act 但 ask 了（过度谨慎）→ act 优于 ask
- (ask, refuse)：该 ask 但 refuse 了 → ask 优于 refuse

数据来源：
- 从 AskBench 的 gold label 生成偏好对
- 用 frontier model（Claude Sonnet, GPT-4o）生成候选 responses
- 对比候选 response 和 gold label 构建偏好

**Stage 2: 三元 DPO 训练**

在 VLM（如 Qwen2.5-VL-7B/14B）上做 DPO：
- 输入：截图 + 指令 + 待执行动作
- 偏好对：(better_decision, worse_decision)
- Loss：标准 DPO loss

**关键设计**：
- **Risk-Level-Aware DPO**：对不同风险等级的错误赋予不同惩罚权重
  - 该 refuse 但 act（最严重）→ 高权重
  - 该 ask 但 act（严重）→ 中高权重
  - 该 act 但 ask（轻微）→ 低权重
  - 该 act 但 refuse（中等）→ 中权重

**Stage 3: 多轮风险累积感知**

针对"单步安全但多步有害"的问题：
- 将多轮交互历史（截图序列 + 已执行动作序列）作为 context
- 训练 agent 在第 N 步时考虑前 N-1 步的累积风险
- 数据来源：从 OpenAgentSafety 和 "Unsafer in Many Turns" 的场景构造

### 贡献 3：实证分析

在 AskBench + 现有 benchmark 上系统评估：

**基线模型**：
- GPT-4o, Claude Sonnet 3.7, Gemini 2.0 Flash（直接评估三元决策）
- MOSAIC（二元决策 baseline）
- 上述模型 + SafePred 外部护栏

**评估维度**：
- 三元决策准确率（vs 二元方法 + 外挂 ask 启发式）
- 安全性 vs 可用性 trade-off 曲线
- 多轮累积风险识别能力
- tool-call vs text 的安全迁移差距

## 与已有工作的关系

```
                    外部护栏                Agent 内在能力
                 ┌──────────────────┐    ┌──────────────────┐
二元决策          │ SafePred (2026)  │    │ MOSAIC (2026)    │
(act/refuse)     │ OS-Sentinel      │    │ AlphaAlign       │
                 │ WebGuard         │    │ CoG, DeRTa       │
                 └──────────────────┘    └──────────────────┘
                 ┌──────────────────┐    ┌──────────────────┐
三元决策          │                  │    │                  │
(act/ask/refuse) │  (不适用——         │    │  ★ 本工作 ★      │
                 │   ask 必须是      │    │                  │
                 │   agent 自身行为) │    │                  │
                 └──────────────────┘    └──────────────────┘
```

**关键差异**：
- vs MOSAIC：我们加入 "ask" 作为第三选项，不是二元决策
- vs SafePred/OS-Sentinel：我们训练 agent 自身能力，不是外挂护栏
- vs "Fail to Act on Risk Knowledge"：他们分析问题，我们提出解决方案
- vs "Mind the GAP"：他们发现 text→tool 不迁移，我们直接在 tool/GUI action 层训练

## 实验计划

| 阶段 | 时间 | 内容 | GPU |
|------|------|------|-----|
| 1. 文献精读 + benchmark 设计 | 2 周 | 精读 MOSAIC/GAP/Unsafer；设计 AskBench 分类体系 | — |
| 2. AskBench 构建 | 3 周 | 标注任务 + 搭建评估环境 + 实现自动评估 | RTX 5090 |
| 3. 基线评估 | 1 周 | 在 AskBench 上评估现有模型 | RTX 5090 |
| 4. 偏好数据构造 | 2 周 | 用 frontier models 生成候选 + 构建偏好对 | RTX 5090 |
| 5. TriDecision 训练 | 3 周 | DPO on Qwen2.5-VL-7B/14B | 8×H800 |
| 6. 评估 + 消融 | 2 周 | 全面评估 + 消融实验 | 8×H800 + 5090 |
| 7. 论文撰写 | 2 周 | 写作投稿 | — |
| **总计** | **~15 周** | | |

## 目标会议

- **NeurIPS 2026**：截稿 ~2026.05，从现在算 ~8-9 周，**非常紧**——需要压缩到 8 周
- **ICLR 2027**：截稿 ~2026.10，充裕（~30 周）
- **建议**：先冲 NeurIPS 2026，如果时间不够就转 ICLR 2027

## 风险评估

| 风险 | 影响 | 概率 | 缓解 |
|------|------|------|------|
| MOSAIC 扩展到三元决策 | 致命 | 低（微软团队方向似乎不同） | 快速执行，先出 benchmark |
| 三元 DPO 训练不收敛 | 高 | 中 | 从简单的 SFT baseline 开始 |
| AskBench 标注困难（灰色地带主观性高） | 中 | 中 | 多标注者 + inter-annotator agreement |
| NeurIPS 截稿太紧 | 中 | 高 | 降级到 ICLR 2027 |
| "ask" 类别定义模糊被审稿人质疑 | 中 | 中 | 用清晰的风险等级定义 + 标注一致性数据 |

## 参考文献

### 核心
- [LM Agents Fail to Act on Risk Knowledge](https://arxiv.org/abs/2508.13465) — >98% awareness, <26% safe execution
- [Mind the GAP](https://arxiv.org/abs/2602.16943) — Text safety ≠ tool-call safety
- [Unsafer in Many Turns](https://arxiv.org/abs/2602.13379) — Multi-turn compounding risk
- [MOSAIC](https://arxiv.org/abs/2603.03205) — Act/refuse 二元决策 (最直接竞品)

### 训练方法
- [AlphaAlign](https://arxiv.org/abs/2507.14987) — 双 reward RL
- [RSafe](https://arxiv.org/abs/2506.07736) — 安全推理 + GRPO
- [DeRTa](https://aclanthology.org/2025.acl-long.158/) — 解耦拒绝训练
- [CoG](https://arxiv.org/abs/2510.21285) — 推理链安全纠正
- [Safety Deep Alignment](https://arxiv.org/abs/2406.05946) — ICLR 2025 Outstanding Paper

### Benchmark
- [Agent-SafetyBench](https://arxiv.org/abs/2412.14470) — 通用 agent 安全
- [OpenAgentSafety](https://arxiv.org/abs/2507.06134) — 真实工具使用安全
- [RiOSWorld](https://arxiv.org/abs/2506.00618) — 桌面 agent 风险
- [OS-Harm](https://arxiv.org/abs/2506.14866) — 桌面 agent 滥用

### 外部护栏（对比）
- [SafePred](https://arxiv.org/abs/2602.01725) — 世界模型预测护栏
- [OS-Sentinel](https://arxiv.org/abs/2510.24411) — 形式化验证 + VLM
- [WebGuard](https://arxiv.org/abs/2507.14293) — 动作风险分类

### Survey
- [Towards Trustworthy GUI Agents](https://arxiv.org/abs/2503.23434) — GUI agent 安全 survey
- [Levels of Autonomy for AI Agents](https://arxiv.org/abs/2506.12469) — 自主性等级框架
