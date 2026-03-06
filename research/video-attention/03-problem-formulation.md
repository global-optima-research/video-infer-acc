# 统一问题形式化：VideoFlashAttention

> 将 5 个优化维度统一到一个数学框架

## 1. 符号定义

### 视频注意力的基本设定

```
输入视频 latent: X ∈ R^{T × H × W × d}
  T = 帧数 (temporal)
  H, W = 空间分辨率 (height, width)
  d = 隐藏维度

展平为序列: X̄ ∈ R^{n × d}, n = T·H·W

对于注意力头 h ∈ [1..H_heads], 层 l ∈ [1..L]:
  Q^{l,h} = X̄ · W_Q^{l,h}    ∈ R^{n × d_h}
  K^{l,h} = X̄ · W_K^{l,h}    ∈ R^{n × d_h}
  V^{l,h} = X̄ · W_V^{l,h}    ∈ R^{n × d_h}

标准注意力:
  A^{l,h} = softmax(Q^{l,h} · (K^{l,h})^T / √d_h)  ∈ R^{n × n}
  O^{l,h} = A^{l,h} · V^{l,h}                        ∈ R^{n × d_h}
```

### 3D 位置编码

每个 token i 有 3D 坐标 p(i) = (t_i, h_i, w_i)，其中：
- t_i ∈ [0, T-1]：帧索引
- h_i ∈ [0, H-1]：行索引
- w_i ∈ [0, W-1]：列索引

### 去噪过程

```
去噪步: s ∈ {1, ..., S}  (从噪声到干净)
噪声水平: σ(s), 单调递减
每步的注意力: A_s^{l,h}  (随 s 变化)
```

---

## 2. 结构性质的数学建模

### 性质 ①② 统一：能量衰减函数

**定义 (时空距离)**：
```
d(i, j) = √(α_t · (t_i - t_j)² + α_s · ((h_i - h_j)² + (w_i - w_j)²))
```
其中 α_t, α_s 是时间/空间权重（α_t > α_s 表示时间衰减更快）。

**性质 (指数能量衰减)**（Radial Attention 实证，R² > 0.985）：
```
E[A^{l,h}_{ij}] ≈ C^{l,h} · exp(-λ^{l,h} · d(i,j))
```
其中 λ^{l,h} > 0 是层/头特定的衰减率。

**推论**：定义 ε-有效注意力集合
```
S_ε^{l,h} = {(i,j) : d(i,j) ≤ -ln(ε/C^{l,h}) / λ^{l,h}}
```
则 |S_ε^{l,h}| = O(n · r_ε^3) 其中 r_ε 是有效半径。

对于典型参数，|S_ε| = O(n log n)，远小于 n²。

### 性质 ③：输入不变性

**性质 (模式不变性)**：
```
对于同一模型 M，任意两个输入 x₁, x₂:
  IoU(supp(A_{x₁}^{l,h}), supp(A_{x₂}^{l,h})) > 0.9
```
其中 supp(A) 是注意力矩阵的 top-k 支撑集。

**推论**：可以离线计算一个**通用稀疏掩码** M^{l,h} 用于所有输入：
```
M^{l,h} = median_{x~P(x)}[top_k(A_x^{l,h})]
```

### 性质 ④：步间稳定性

**性质 (U 形变化)**：定义步间注意力变化
```
Δ_s^{l,h} = ||A_{s+1}^{l,h} - A_s^{l,h}||_F / ||A_s^{l,h}||_F
```

则：
```
Δ_s^{l,h} =
  大    if s ∈ [1, 0.15S]           (初始化阶段)
  小    if s ∈ [0.15S, 0.85S]       (稳定阶段)
  大    if s ∈ [0.85S, S]           (精修阶段)
```

**关键**：在稳定阶段，A_{s+1} ≈ A_s，可以做增量计算。

### 性质 ⑤：头功能特化

**定义 (头类型函数)**：
```
τ: (l, h) → {spatial, temporal, global, sink}
```

每种类型对应不同的最优注意力模式：
```
spatial  → 块对角 (帧内局部)      → 小窗口, 高精度
temporal → 等间距条纹 (跨帧)      → stride 注意力, 中精度
global   → 近均匀分布             → 低秩近似或低精度全注意力
sink     → 集中在 <s> token       → 跳过 (仅传递 bias)
```

---

## 3. 统一优化问题

### 目标

给定质量约束 ε（允许的最大注意力近似误差），最小化总 IO 成本。

### 决策变量

对于每个 (层 l, 头 h, 去噪步 s)，决定：

1. **稀疏掩码** M_s^{l,h} ∈ {0,1}^{n×n}：哪些注意力项要计算
2. **分块策略** B_s^{l,h}：3D tile 大小和形状
3. **精度** q_s^{l,h} ∈ {FP16, FP8, INT8, INT4}：计算精度
4. **复用标记** r_s^{l,h} ∈ {compute, reuse, delta}：
   - compute：从头计算
   - reuse：直接复用上一步的输出
   - delta：增量更新

### 目标函数

```
minimize  Σ_{l,h,s} IO_cost(M_s^{l,h}, B_s^{l,h}, q_s^{l,h}, r_s^{l,h})
```

其中 IO 成本分解为：

```
IO_cost =
  case r = compute:
    Σ_{blocks b ∈ B} [load_Q(b) + load_K(b) + load_V(b) + store_O(b)] / bandwidth(q)

  case r = reuse:
    0  (零成本，直接用缓存)

  case r = delta:
    IO_cost(ΔM_s^{l,h}, B_s^{l,h}, q_s^{l,h}, compute)
    其中 ΔM 只包含变化的部分
```

### 约束

**质量约束**：
```
||O_approx^{l,h,s} - O_exact^{l,h,s}||_F / ||O_exact^{l,h,s}||_F ≤ ε^{l,h}

其中 ε^{l,h} 是层/头特定的容忍度：
  ε^{l,h} = ε_base           对于普通层
  ε^{l,h} = ε_base / 10      对于关键层 (如 Mochi 44-45)
```

**硬件约束**：
```
每个 block 的 SRAM 用量 ≤ M_SRAM
分块大小 B 必须对齐 GPU warp size
```

---

## 4. 分解为子问题

统一问题太大，直接求解不现实。分解为 3 个可独立优化的子问题：

### 子问题 A：静态配置（离线，一次性）

利用性质 ③（输入不变性）+ ⑤（头特化），离线确定：

```
输入: 模型 M, 少量 profile 数据 (10-50 samples)
输出:
  - 每头类型 τ(l,h)
  - 每头基础掩码 M_base^{l,h}（有效注意力区域）
  - 每头最优 tile 大小 B^{l,h}
  - 关键层列表 L_critical

方法: 在 profile 数据上统计注意力模式，取 median
复杂度: O(1)（只做一次）
```

### 子问题 B：步级调度（在线，每步决策）

利用性质 ④（U 形稳定性），每步决定复用策略：

```
输入: 当前步 s, 缓存的注意力 A_{s-1}^{l,h}
输出: 每 (l,h) 的复用决策 r_s^{l,h}

规则（基于 U 形先验 + 在线校准）：
  if s ∈ 初始化阶段:    r = compute  (每步全算)
  if s ∈ 稳定阶段:
    if (l,h) ∈ L_critical:  r = delta  (增量更新关键层)
    else:                    r = reuse  (直接复用)
  if s ∈ 精修阶段:       r = compute  (每步全算)

增量更新的 delta 掩码:
  ΔM_s^{l,h} = {(i,j) ∈ M_base : |A_s(i,j) - A_{s-1}(i,j)| > δ}

δ 可以通过少量采样 (~1%) 在线估计
```

### 子问题 C：块级计算（Kernel 内部）

利用性质 ①②⑦，在每个要计算的块内做 IO 优化：

```
输入: Q, K, V 子矩阵, 掩码 M, 精度 q
输出: O = masked_softmax(QK^T/√d) · V

策略（按头类型）:
  spatial 头:
    - 3D tile 大小 (1, T_h, T_w), T_h·T_w ≈ √(SRAM/d)
    - 窗口: 帧内 ± r 范围
    - 精度: FP16（关键层）或 FP8（普通层）

  temporal 头:
    - tile 大小 (T_t, 1, 1), stride = H·W
    - 只加载时间对应位置的 K/V
    - 精度: FP8

  global 头:
    - 低秩近似: A ≈ φ(Q) · φ(K)^T (Random Feature)
    - 或 FP8 全注意力 (如果头数少)

  sink 头:
    - 跳过注意力计算
    - 输出 = V_mean · learned_bias
```

---

## 5. 理论分析

### IO 复杂度

**定理（非正式）**：VideoFlashAttention 的总 IO 复杂度为

```
IO_total = Σ_s [
  Σ_{(l,h): r_s=compute} O(|S_ε^{l,h}| · d² / M)
  + Σ_{(l,h): r_s=delta}  O(|ΔS_s^{l,h}| · d² / M)
  + Σ_{(l,h): r_s=reuse}  0
]
```

**对比标准 FlashAttention**：
```
IO_FA = S · L · H_heads · O(n² · d² / M)
```

**加速比（理论）**：
```
Speedup = IO_FA / IO_total
        = S · L · H_heads · n² / Σ effective_entries

分解:
  步数复用 (性质④): ~3.3x (只有 30% 的步需要全算)
  稀疏性 (性质①②): ~6.7x (15% token 覆盖 70% 注意力 → ~15% 有效项)
  头跳过 (性质⑤):   ~1.1x (5% 的 sink 头跳过)
  量化 (性质⑦):     ~2x   (FP8 vs FP16)

理论组合: 3.3 × 6.7 × 1.1 × 2 ≈ 48.6x (注意力 kernel)
```

### 对比现有方法

```
方法          | 利用的乘法因子       | 理论极限
FlashAttn 3   | 1x (基准)           | 1x
SageAttn 2++  | 2x (量化)           | 2x
STA           | 6.7x (稀疏)         | ~10x
FPSAttention  | 6.7 × 2 = 13.4x    | ~13x
PAB (若适用)  | 3.3x (步复用)       | ~3.3x
VideoFlashAttn| 3.3 × 6.7 × 1.1 × 2| ~48x
```

**注意**：这是理论上界。实际 kernel 效率（memory coalescing、warp utilization、bank conflicts）会打折，预计实际 **15-30x** kernel 加速。

---

## 6. 与现有工作的关键区别

| 维度 | 现有最佳 | VideoFlashAttention |
|------|---------|-------------------|
| 算法类型 | 启发式（阈值/窗口大小） | 有理论 bound 的优化 |
| 跨步策略 | 全算或全复用（PAB 式二选一） | 三档：全算/增量/复用 |
| 头处理 | 统一算法，参数不同 | 每头类型一套专用 kernel |
| 精度决策 | 统一精度 | 层/头/步自适应精度 |
| 部署方式 | 运行时动态 | 离线 profile + 预编译 + 运行时微调 |

---

## 7. 核心创新点总结

如果要写成一篇论文，核心贡献是：

1. **统一视角**：首次把视频注意力的 7 个已知结构性质放入一个优化框架，证明现有方法各自只利用了 1-2 个性质
2. **三级复用架构**（compute/delta/reuse）：首次在 full 3D 注意力上实现注意力矩阵级的增量计算（不是输出级缓存）
3. **预编译管线**：利用 >90% 的输入不变性，模型特定的编译优化，零运行时 profiling 开销
4. **理论 IO bound**：证明视频注意力的 IO 复杂度可以从 O(n²d²/M) 降到 O(n·polylog(n)·d²/M)

---

## 8. 待解决的关键问题

### 理论层面
- [ ] 增量计算的误差如何累积？多步 reuse 后误差 bound 是什么？
- [ ] 头类型分类的最优粒度？（二分类 vs 四分类 vs 连续谱）
- [ ] 步级调度的最优策略是否可以 closed-form 求解？

### 实验层面
- [ ] 需要在真实模型上验证 7 个性质的量化数据（而非引用他人论文的数据）
- [ ] 增量计算 (delta mode) 的实际加速比是多少？
- [ ] 预编译的 profile 需要多少样本才稳定？

### 工程层面
- [ ] Triton 原型能否验证理论预测？
- [ ] 异构头调度在 GPU 上的实际效率？（不同 warp 跑不同 kernel）
- [ ] 与 SageAttention 量化的兼容性？
