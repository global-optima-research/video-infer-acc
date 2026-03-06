# 实验验证计划

> 目标：用最少的资源验证核心假设，决定是否值得投入 kernel 开发

## Phase 0: 注意力矩阵采集与分析（1-2 天，单卡）

### 目标
用真实模型采集注意力矩阵，亲自验证 7 个结构性质的量化数据。不依赖他人论文的数据。

### 实验设置
```
模型: Wan 2.1-1.3B (单卡可跑)
分辨率: 480p, 49 帧
Prompt: 10 个不同场景 (静态/运动/多人/自然/室内)
采集: 所有层、所有头、所有去噪步的注意力矩阵
存储: 注意力矩阵太大 → 存统计量（每块的均值/方差/稀疏度/entropy）
```

### 需要验证的量

| 性质 | 测量方法 | 预期结果 |
|------|---------|---------|
| ① 块对角主导 | mean(diag blocks) / mean(off-diag blocks) | >2.5x |
| ② 时间衰减 | attention_mass(window_r) vs r 的关系 | 指数衰减 |
| ③ 跨 prompt 不变性 | top-k 支撑集的 IoU (跨 10 个 prompt) | >0.85 |
| ④ U 形稳定性 | ||A_{s+1} - A_s||_F / ||A_s||_F vs s | U 形 |
| ⑤ 头特化 | 每头注意力模式的聚类 | 3-4 类 |
| ⑥ 层敏感性 | 逐层施加稀疏度，测 FVD/CLIP | 找到关键层 |
| ⑦ 熵分布 | 每头 attention entropy | 双峰分布 |

### 产出
- 7 个性质的**亲自测量**数据（不是引用）
- 每个性质的可视化图表
- 性质间相关性分析（哪些性质在同一个头上同时出现）

---

## Phase 1: 增量计算可行性验证（2-3 天，单卡）

### 核心假设
> 在稳定阶段（中间 70% 步），注意力矩阵的变化足够稀疏，使得增量更新比全量计算更快。

### 实验

**1a. 测量步间变化的稀疏度**
```python
# 伪代码
for step s in [0.15S, 0.85S]:
    delta = A[s+1] - A[s]
    sparsity = (|delta| < threshold).mean()
    # 预期: sparsity > 90% 意味着增量更新只需算 <10% 的项
```

**1b. 增量更新的误差累积**
```python
# 模拟连续 N 步 reuse/delta
A_cached = A[s0]  # 基准步
for s in range(s0+1, s0+N):
    # 方案1: 全 reuse
    error_reuse[s] = ||A_cached - A[s]||_F / ||A[s]||_F

    # 方案2: delta 更新 (只更新变化最大的 top-k% 项)
    delta = A[s] - A_cached
    top_indices = topk(|delta|, k%)
    A_cached[top_indices] = A[s][top_indices]
    error_delta[s] = ||A_cached - A[s]||_F / ||A[s]||_F
```

**1c. 不同 delta 比例 vs 质量**
```
delta 比例: 0% (全 reuse), 1%, 5%, 10%, 20%, 50%, 100% (全算)
测量: 最终视频的 FVD, CLIP-Score, VBench
目标: 找到 sweet spot (最少计算量 + 可接受质量)
```

### 成功标准
- delta 5-10% 时质量损失 <1% → 增量计算可行
- 连续 reuse 5 步质量可接受 → 步间复用有价值

---

## Phase 2: 头异构调度 PyTorch 原型（3-5 天，单卡）

### 核心假设
> 对不同类型的头使用不同计算策略，比统一策略有显著加速。

### 实验

**2a. 头分类器**
```python
# 基于 Phase 0 采集的数据
for each (layer, head):
    pattern = attention_stats[layer][head]
    if is_spatial(pattern):      type = 'spatial'   # 小窗口
    elif is_temporal(pattern):   type = 'temporal'   # stride
    elif is_global(pattern):     type = 'global'     # 低秩/全注意力
    elif is_sink(pattern):       type = 'sink'       # 跳过
```

**2b. PyTorch 级别异构注意力**
```python
# 不是 CUDA kernel，只是 PyTorch 实现，验证正确性
def heterogeneous_attention(Q, K, V, head_types):
    outputs = []
    for h, htype in enumerate(head_types):
        if htype == 'spatial':
            out = window_attention(Q[h], K[h], V[h], window=local)
        elif htype == 'temporal':
            out = strided_attention(Q[h], K[h], V[h], stride=H*W)
        elif htype == 'global':
            out = linear_attention(Q[h], K[h], V[h])  # Random Feature
        elif htype == 'sink':
            out = V[h].mean(dim=0).expand_as(Q[h])    # 常数近似
    return stack(outputs)
```

**2c. 质量测试**
```
对比:
  - 基准: 全注意力 (FlashAttention)
  - 方案A: 统一窗口 (STA 式)
  - 方案B: 异构调度 (本方法)
  - 方案C: 异构 + 步间复用

指标: FVD, CLIP-Score, VBench, SSIM
模型: Wan 2.1-1.3B, 50 个 prompt
```

### 成功标准
- 方案 B 在相同稀疏度下质量 > 方案 A → 异构有价值
- 方案 C 叠加后质量仍可接受 → 维度间可组合

---

## Phase 3: Triton 原型 Kernel（1-2 周，需 GPU）

### 前置条件
Phase 1 和 Phase 2 的成功标准都达到。

### 实现

**3a. 单头 kernel**
```
先实现 4 种单头 kernel:
  - spatial_attention_kernel (3D window)
  - temporal_attention_kernel (strided)
  - global_attention_kernel (linear/low-rank)
  - sink_bypass_kernel (常数输出)

用 Triton 实现，对比 FlashAttention2/3 的速度
```

**3b. 调度器**
```
实现一个 dispatcher:
  输入: head_type_map (离线 profile 结果)
  行为: 对每个头调用对应的 kernel
  目标: dispatcher overhead < 1% 的总计算时间
```

**3c. 增量模式**
```
实现 delta attention kernel:
  输入: Q, K, V, cached_A, delta_mask
  行为: 只计算 delta_mask 标记的位置，其余复用 cached_A
  挑战: softmax 归一化需要全局信息 → 增量 softmax?
```

### 关键技术挑战

**增量 softmax 问题**：
```
标准 softmax: A_ij = exp(s_ij) / Σ_k exp(s_ik)

如果只更新部分 s_ij，分母变了，所有 A_ij 都要变。
可能的解法:
  1. 近似: 假设分母变化小，只更新分子
  2. 块级: 以块为单位全量计算，只跳过整个块
  3. 分段: 只在同一行内做增量（行内分母独立）
```

选项 3 最可行：每行的 softmax 是独立的，如果一行内的变化项少，可以用 online softmax 的增量版本。

---

## Phase 4: 预编译管线（Phase 3 之后）

### 实现
```
输入: 模型 M
流程:
  1. 跑 10-50 个 profile sample
  2. 统计每 (l,h) 的类型、最优窗口大小、精度
  3. 生成模型专用的配置文件
  4. 编译模型专用的 kernel (Triton AOT 编译)

产出: model_config.json + compiled_kernels/
```

---

## 资源估计

| Phase | 时间 | GPU 需求 | 产出 |
|-------|------|---------|------|
| 0: 数据采集 | 1-2 天 | 1× RTX 4090 | 结构性质验证数据 |
| 1: 增量可行性 | 2-3 天 | 1× RTX 4090 | go/no-go 决策 |
| 2: PyTorch 原型 | 3-5 天 | 1× RTX 4090 | 质量验证 |
| 3: Triton kernel | 1-2 周 | 1× H100 | 速度验证 |
| 4: 预编译 | 1 周 | 1× H100 | 完整系统 |

**Phase 0-2 是关键决策点**：如果数据不支持假设，在投入 kernel 开发之前就能止损。

总计：**1 张 4090 跑 1-2 周**即可完成前 3 个 Phase 的验证。

---

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 增量计算误差累积太快 | 中 | 高 | Phase 1 先验证；备选：块级复用代替 token 级 |
| 异构调度 GPU 效率低 | 中 | 中 | Triton 的灵活性可能不够 → 考虑 CUDA |
| 预编译假设不成立（某些模型不满足不变性） | 低 | 中 | 保留在线 fallback |
| 竞争对手先发表 | 中 | 高 | 快速迭代，先发 arXiv |
| Softmax 增量更新不精确 | 中 | 高 | 退化到块级：跳过整个 block 而非 token |
