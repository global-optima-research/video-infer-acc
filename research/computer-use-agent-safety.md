# Computer Use Agent Safety 方向调研

> 日期：2026-03-12
> 目的：评估 Computer Use Agent 安全方向的可行性和差异化空间
> 背景：视频推理加速、Step Distillation、视频编辑、物理感知视频生成、VLM reasoning 六个方向经核查均已饱和，需要全新赛道

## 一、背景

Computer Use Agent（GUI Agent）正在从研究走向产品化：
- Claude Computer Use（Anthropic, 2024.10 发布）
- OpenAI Operator（2025.01 发布）
- Google Project Mariner
- Microsoft Magentic-UI
- 各类开源 GUI Agent（SeeClick, CogAgent, etc.）

核心安全问题：**这些 agent 能否正确识别有风险的操作（删除文件、发送邮件、购买商品、修改系统设置），并避免做出错误行为？**

现有证据令人担忧：
- BrowserART (Scale AI): GPT-4o 执行了 98/100 个有害行为
- GhostEI-Bench: 恶意弹窗攻击成功率 86%
- WebGuard: 前沿 LLM 风险预测准确率 <60%
- Agent-SafetyBench: 没有 agent 安全得分超过 60%

## 二、竞争格局

### 2.1 已有工作统计

| 类别 | 数量 | 成熟度 |
|------|------|--------|
| Benchmark | 12+ | 中 |
| 训练方法 | 3 | 低 |
| 护栏系统 | 5 | 低（全是原型） |
| 攻击论文 | 4+ | 中 |
| Survey | 1 | 早期 |
| **总计** | **~25** | **早中期** |

### 2.2 Benchmark 详情

| Benchmark | 会议 | 环境 | 任务数 | 关键发现 |
|-----------|------|------|--------|---------|
| RiOSWorld | NeurIPS'25 | 桌面 (Ubuntu) | 492 | >75% 不安全率 |
| OS-Harm | ICML'25 | 桌面 (Ubuntu) | 150 | 所有前沿模型都执行有害请求 |
| ST-WebAgentBench | ICML'25 | Web (企业) | 222 | 安全完成率不到名义的 2/3 |
| SafeArena | ICML'25 | Web (4 站点) | 500 | GPT-4o 完成 34.7% 有害请求 |
| WASP | NeurIPS'25 | Web | — | 注入攻击在 86% 场景部分成功 |
| WebGuard | arXiv 2025.07 | Web (193 真实网站) | 4,939 | 前沿 LLM 风险预测 <60% |
| MobileSafetyBench | AAAI'26 | 移动 (Android) | 250 | Agent 在复杂风险场景失败 |
| GhostEI-Bench | arXiv 2025.10 | 移动 (Android) | — | 视觉注入攻击 86% 成功率 |
| BrowserART | Scale AI | 浏览器 | 100 | GPT-4o 执行 98/100 有害行为 |
| MLA-Trust | arXiv 2025.06 | Web + Mobile | 34 | GUI agent 可信度差于独立 MLLM |
| Agent-SafetyBench | arXiv 2024.12 | 通用工具 | 2,000 | 无 agent >60% 安全分 |
| SafeAgentBench | arXiv 2024.12 | 具身仿真 | 750 | 最佳 baseline 仅 5% 拒绝率 |

### 2.3 训练方法

| 方法 | 日期 | 方式 | 局限 |
|------|------|------|------|
| AgentAlign | 2025.05 | 抽象行为链 + SFT | 通用 tool-calling，非 GUI 专用 |
| Agent Safety via RL | 2025.07 | 沙箱 RL + 细粒度 reward | 通用框架，非 GUI 专用 |
| SCoT | 2024.10 | 安全引导 CoT prompting | 仅 prompting，非训练 |

### 2.4 护栏系统

| 系统 | 日期 | 方式 | 状态 |
|------|------|------|------|
| SafePred | 2026.02 | 世界模型预测性护栏 | 研究原型，>97.6% 安全 |
| OS-Sentinel | 2025.10 | 形式化验证 + VLM 语义判断 | 研究原型 |
| WebGuard | 2025.07 | 微调 Qwen2.5VL-7B 做风险分类 | 研究原型 |
| Magentic-UI | 2025.05 | HITL + 动作守卫 | 微软开源，最接近产品 |
| AgentDoG | 2026.01 | 诊断型护栏，3D 风险分类 | 研究原型 |

### 2.5 攻击论文

| 论文 | 攻击方式 | 成功率 |
|------|---------|--------|
| BrowserART | 浏览器 agent 红队 | 98% |
| GhostEI-Bench | 视觉环境注入 (恶意弹窗) | 86% |
| WASP (Meta) | Web agent 提示注入 | 86% |
| Pop-up Attacks on VLM | 对抗性弹窗 | 86% |

## 三、Gap 分析

### ✅ Gap 1: GUI Agent 专用安全训练（最大 gap）

**现状**：3 个训练方法全部是通用 tool-calling agent 的，**没有任何工作专门在 screenshot + mouse/keyboard action space 上做安全训练**。

**为什么重要**：
- GUI agent 的动作空间（像素坐标点击、键盘输入）与 tool-calling（函数调用）完全不同
- 风险评估需要**视觉理解**——"红色的删除按钮" vs "蓝色的确认按钮"
- 上下文更复杂——需要理解当前屏幕状态、应用程序语义

**可能的方案**：
- 构建 GUI 安全训练数据（safe/unsafe 动作对 + 截图）
- 在 VLM 上做安全对齐（DPO/RLHF with GUI observations）
- 与 Gap 2 结合：先训练风险分类器，再用于安全训练的 reward

### ✅ Gap 2: 校准的 GUI 动作风险评估

**现状**：WebGuard 是唯一尝试，但前沿 LLM 准确率 <60%。**Agent 无法可靠区分"点击导航链接"和"点击删除按钮"**。

**为什么重要**：
- 这是所有安全机制的基础——如果不能判断风险，就无法决定是否拒绝
- 需要**视觉+语义联合理解**：按钮文字、颜色、位置、当前应用上下文
- 需要**校准**：不是二分类，而是风险概率估计

**可能的方案**：
- 大规模 GUI 动作风险标注数据集
- 多粒度风险分类（无风险 / 低风险可自动执行 / 中风险需确认 / 高风险拒绝）
- 基于 VLM 的风险评分模型

### ✅ Gap 3: 跨应用链式风险传播

**现状**：**零覆盖**。所有 benchmark 测试孤立任务。

**场景示例**：
- 邮件中有恶意指令 → agent 打开终端 → 执行危险命令
- 网页伪装为系统设置 → agent 输入管理员密码
- 聊天应用收到社工消息 → agent 点击钓鱼链接 → 下载恶意文件

**为什么重要**：
- 真实世界的攻击往往是多步链式的
- 单步安全 ≠ 链式安全（每步看起来低风险，但组合起来高风险）

### ✅ Gap 4: 视觉对抗防御

**现状**：攻击论文有 3-4 篇（86% 成功率），**但没有有效防御方法**。

**攻击方式**：
- 恶意弹窗覆盖 UI 元素
- 伪造的系统对话框
- 视觉相似的钓鱼页面
- 修改 DOM 但保持视觉一致

### ⚠️ Gap 5: Windows/macOS 安全 Benchmark

**现状**：RiOSWorld 和 OS-Harm 都只覆盖 Ubuntu。**零 Windows/macOS 覆盖**。

**局限**：搭建 Windows/macOS 的可复现评估环境比 Linux VM 难很多（授权、自动化工具链）。

## 四、推荐研究方案

### 方案：Computer Use Agent 安全风险评估与对齐

**一句话**：现有 computer use agent 在风险识别上严重不足（<60% 准确率，98% 有害行为执行率）。我们提出一套完整的**风险感知框架**：benchmark + 风险评估模型 + 安全对齐训练。

**核心贡献**：
1. **GUI-SafetyBench**：首个覆盖桌面+Web+移动、包含跨应用链式攻击的 GUI agent 安全 benchmark
2. **GUI-RiskModel**：基于 VLM 的校准风险评估模型，输入截图+待执行动作，输出风险等级+解释
3. **GUI-SafeAlign**：首个专门针对 GUI action space 的安全对齐训练方法

**技术路线**：

| 阶段 | 时间 | 内容 | 产出 |
|------|------|------|------|
| 1. Benchmark 构建 | 4 周 | 设计风险分类体系 + 标注 GUI 动作数据 + 搭建评估环境 | GUI-SafetyBench |
| 2. 风险评估模型 | 3 周 | 在 VLM 上训练风险分类器 + 校准 | GUI-RiskModel |
| 3. 安全对齐 | 4 周 | DPO/RLHF on GUI agent with risk reward | GUI-SafeAlign |
| 4. 评估 + 论文 | 3 周 | 对比实验 + ablation + 写作 | 投稿 |

**资源需求**：
- 8×H800：用于 VLM 安全训练（Qwen2.5-VL-72B 或类似模型的 DPO）
- RTX 5090：快速推理评估
- 评估环境：Ubuntu VM（复用 OSWorld），可选 Windows VM

**目标会议**：NeurIPS 2026（截稿 ~2026.05）或 ICLR 2027

## 五、Double Check 结果（2026-03-12）

对四个 gap 进行了严格二次核实：

| Gap | 初始判断 | 核实结果 | 关键证据 |
|-----|---------|---------|---------|
| Gap 1: GUI 安全训练 | ✅ 空白 | ⚠️ 部分填补 | MOSAIC (2026.03) 接近覆盖；VLM 安全对齐 + Agent 安全 RL 组件分别存在 |
| **Gap 2: 校准风险评估** | ✅ 空白 | **❌ 已覆盖** | WebGuard + SafePred (97.6%) + OS-Sentinel 直接解决 |
| **Gap 3: 跨应用链式攻防** | ✅ 空白 | **✅ 确认空白** | 攻击已被刻画（Kill Chain, CUA SoK），但无防御论文 |
| **Gap 4: 视觉对抗防御** | ✅ 空白 | **✅ 确认空白** | 攻击 86% 成功率，现有防御 ≤25% 效果，无有效方案 |

**结论**：Gap 2 已被覆盖，不可做。Gap 1 窗口期短。**Gap 3 + Gap 4 是确认的真 gap。**

### 核实中发现的新竞品

| 论文 | 日期 | 与 Gap 的关系 |
|------|------|-------------|
| MOSAIC | 2026.03 | 接近 Gap 1，但是 tool-use 非 GUI |
| SafePred | 2026.02 | 直接覆盖 Gap 2（97.6% safety） |
| CUA SoK | 2025.07 | 刻画了 Gap 3 的攻击面，但无防御 |
| Pop-up Attack | 2024.11 | 刻画了 Gap 4 的攻击，证明基本防御无效 |
| GUIGuard | 2026.01 | 隐私导向，非对抗防御 |

## 六、修正后的研究方案

### 定位：Computer Use Agent 在多步跨应用与视觉对抗场景下的安全性

聚焦 **Gap 3（跨应用链式攻击防御）+ Gap 4（视觉对抗防御）**，组合为统一研究：

**一句话**：现有 computer use agent 在单步孤立任务上已有初步安全保障，但面对跨应用链式攻击和视觉对抗注入时几乎完全失败。我们提出首个覆盖这两类高级威胁的 benchmark + 防御框架。

### 核心贡献

1. **CrossAppAttack Benchmark**：首个跨应用链式攻击 benchmark
   - 邮件→浏览器→终端→文件系统的多步攻击链
   - 社工消息→钓鱼链接→凭证窃取的社会工程链
   - 恶意网页→系统设置→权限提升的提权链
   - 覆盖桌面（Ubuntu，复用 OSWorld）环境
   - 包含 safe/attack 对照任务

2. **VisualAdv-GUI Benchmark**：首个 GUI agent 视觉对抗防御 benchmark
   - 恶意弹窗覆盖（基于 Pop-up Attack 扩展）
   - 伪造系统对话框
   - 视觉相似钓鱼页面
   - 动态 DOM 注入
   - 包含攻击成功率 + 防御有效性双向评估

3. **ChainGuard 防御框架**：跨应用 + 视觉对抗的统一防御
   - **跨应用信任追踪**：跟踪信息来源的信任链（trusted/untrusted），跨应用传播时降低信任等级
   - **视觉验证模块**：交叉验证 DOM 信息与视觉渲染的一致性，检测注入
   - **动作风险传播分析**：评估当前动作在攻击链上下文中的风险（vs 孤立评估）

### 技术路线

| 阶段 | 时间 | 内容 | 产出 |
|------|------|------|------|
| 1. 环境搭建 | 2 周 | 基于 OSWorld 搭建跨应用评估环境 | 评估基础设施 |
| 2. Benchmark 构建 | 4 周 | 设计攻击场景 + 标注 + 实现自动化评估 | CrossAppAttack + VisualAdv-GUI |
| 3. 基线评估 | 2 周 | 在 benchmark 上评估现有 agent + 护栏 | 基线数据 |
| 4. 防御框架 | 4 周 | 实现 ChainGuard + 训练/调优 | ChainGuard |
| 5. 评估 + 论文 | 2 周 | 对比实验 + 消融 + 撰写 | 投稿 |

### 目标会议

- **首选**：NeurIPS 2026（截稿 ~2026.05，时间紧但可行）
- **备选**：ICLR 2027（截稿 ~2026.10，充裕）
- **适合的 track**：Safety & Alignment / Agent / Trustworthy AI

### 资源需求

- 评估环境：Ubuntu VM（OSWorld），不需要 Windows/macOS
- GPU：RTX 5090 做推理评估；8×H800 用于可能的防御模型训练
- 人力：主要是 benchmark 设计和工程实现

## 七、风险评估

| 风险 | 影响 | 缓解 |
|------|------|------|
| NeurIPS 2026 截稿太紧 | 高 | 降级到 ICLR 2027；或先投 benchmark 论文 |
| 跨应用 benchmark 被抢先 | 低 | 这个 gap 非常具体，短期内被抢的概率低 |
| 防御效果不显著 | 中 | Benchmark 本身是独立贡献 |
| OSWorld 环境局限性 | 低 | Ubuntu 足够，桌面 agent 主流测试环境 |

## 八、最终方向确定（2026-03-12）

经过进一步核查和讨论，最终确定方向为：

**Agent 三元决策边界（Act / Ask / Refuse）**

核心发现：所有现有方法（MOSAIC, AlphaAlign, CoG, DeRTa）都是二元决策（act/refuse），没有任何工作训练 agent 学习三元决策——包含"请求人类确认"作为第三选项。

详见正式 proposal：`research/proposal-agent-decision-boundary.md`

## 九、与前期工作的关系

虽然从视频推理加速转到 agent safety 跨度大，但以下经验可迁移：
- **系统工程能力**：搭建复杂评估 pipeline 的经验
- **实验方法论**：控制变量、消融实验、多维度评估
- **对 VLM/DiT 的理解**：VLM 是 GUI agent 的核心组件

## 参考文献

### Benchmark
- [RiOSWorld](https://arxiv.org/abs/2506.00618) — NeurIPS 2025, 桌面 agent 风险评估
- [OS-Harm](https://arxiv.org/abs/2506.14866) — ICML 2025, 桌面 agent 滥用
- [ST-WebAgentBench](https://arxiv.org/abs/2410.06703) — ICML 2025, Web agent 安全可信
- [SafeArena](https://arxiv.org/abs/2503.04957) — ICML 2025, Web agent 故意滥用
- [WASP](https://arxiv.org/abs/2504.18575) — NeurIPS 2025, Web agent 提示注入
- [WebGuard](https://arxiv.org/abs/2507.14293) — 动作风险分类
- [MobileSafetyBench](https://arxiv.org/abs/2410.17520) — AAAI 2026, 移动 agent 安全
- [GhostEI-Bench](https://arxiv.org/abs/2510.20333) — 移动视觉注入
- [BrowserART](https://scale.com/research/browser-art) — Scale AI, 浏览器红队
- [MLA-Trust](https://arxiv.org/abs/2506.01616) — GUI agent 可信度
- [Agent-SafetyBench](https://arxiv.org/abs/2412.14470) — 通用 agent 安全
- [SafeAgentBench](https://arxiv.org/abs/2412.13178) — 具身 agent 安全

### 训练方法
- [AgentAlign](https://arxiv.org/abs/2505.23020) — 抽象行为链安全对齐
- [Agent Safety via RL](https://arxiv.org/abs/2507.08270) — 沙箱 RL 安全对齐

### 护栏系统
- [SafePred](https://arxiv.org/abs/2602.01725) — 世界模型预测性护栏
- [OS-Sentinel](https://arxiv.org/abs/2510.24411) — 形式化验证 + VLM 判断
- [WebGuard guardrail](https://arxiv.org/abs/2507.14293) — 微调风险分类器
- [Magentic-UI](https://arxiv.org/abs/2507.22358) — 微软 HITL 护栏
- [AgentDoG](https://arxiv.org/abs/2601.18491) — 诊断型护栏

### 攻击
- [BrowserART](https://scale.com/research/browser-art) — 浏览器 agent 红队
- [GhostEI-Bench](https://arxiv.org/abs/2510.20333) — 视觉环境注入
- [WASP](https://arxiv.org/abs/2504.18575) — Web 提示注入

### Survey & 资源
- [Towards Trustworthy GUI Agents: A Survey](https://arxiv.org/abs/2503.23434)
- [Awesome-GUI-Agent-Safety](https://github.com/Autonomous-Agent-Team/Awesome-GUI-Agent-Safety)
- [2025 AI Agent Index](https://arxiv.org/abs/2602.17753) — MIT
