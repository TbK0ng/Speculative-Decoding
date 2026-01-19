# 02. Speculative Decoding 理论

## 目录

- [1. 核心思想](#1-核心思想)
- [2. 拒绝采样的数学推导](#2-拒绝采样的数学推导)
- [3. 算法流程详解](#3-算法流程详解)
- [4. 加速比分析](#4-加速比分析)
- [5. 与蒙特卡洛方法的关系](#5-与蒙特卡洛方法的关系)
- [6. 小结](#6-小结)

---

## 1. 核心思想

### 1.1 问题陈述

自回归生成的根本问题是：**每步只能生成 1 个 token**。

```
传统方法 (生成 N 个 token):
Step 1: target([x₁, ..., x_t]) → x_{t+1}
Step 2: target([x₁, ..., x_{t+1}]) → x_{t+2}
...
Step N: target([x₁, ..., x_{t+N-1}]) → x_{t+N}

需要: N 次 target 前向传播
```

### 1.2 Speculative Decoding 的洞察

**关键发现 1**：小模型可以快速预测多个候选 token

**关键发现 2**：Transformer 可以并行计算多个位置的概率分布

```
target([x₁, ..., x_t, x̂_{t+1}, x̂_{t+2}, ..., x̂_{t+γ}])
     ↓
输出: P(x_{t+1}, x_{t+2}, ..., x_{t+γ+1} | x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+γ})
      一次性得到 γ+1 个位置的概率分布！
```

### 1.3 核心算法框架

```
给定: 当前序列 [x₁, ..., x_t]

Step 1 (Drafter): 串行生成 γ 个候选
  x̂_{t+1}, x̂_{t+2}, ..., x̂_{t+γ} ~ drafter([x₁, ..., x_t])

Step 2 (Target): 并行验证 γ 个候选
  p = target([x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+γ}])

Step 3 (Rejection Sampling): 决定接受/拒绝
  对于每个候选 x̂_{t+i}:
    以概率 min(1, p(x̂_{t+i}) / q(x̂_{t+i})) 接受
    否则拒绝

Step 4 (Sample): 生成最终 token
  如果全部接受: 从 p(x_{t+γ+1}) 采样
  如果在第 n 个拒绝: 从调整后的分布采样
```

---

## 2. 拒绝采样的数学推导

### 2.1 为什么需要拒绝采样？

**问题**：drafter 的分布 q 与 target 的分布 p 不同，直接使用 q 的输出会改变目标分布。

**目标**：设计一种采样机制，使得最终输出分布**精确等于** p，同时利用 q 加速。

### 2.2 接受概率公式

对于候选 token x̂，接受概率为：

$$P_{\text{accept}}(x̂) = \min\left(1, \frac{p(x̂)}{q(x̂)}\right)$$

**直观解释**：
- 如果 p(x̂) ≥ q(x̂)：drafter 低估了这个 token，总是接受
- 如果 p(x̂) < q(x̂)：drafter 高估了这个 token，以 p/q 的概率接受

### 2.3 证明：输出分布等于目标分布 ⭐

**定理**：Speculative Decoding 的输出分布精确等于 target 的分布 p。

**证明**：

设采样过程为：从 q 中采样 x̂，然后以概率 min(1, p(x̂)/q(x̂)) 接受。

我们需要证明：P[输出 = x] = p(x)

**情况 1：采样被接受**

采样到 x̂ = x 且被接受的概率：
$$P[\text{accept } x] = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))$$

**情况 2：采样被拒绝**

采样到某个 x̂ 被拒绝，然后从调整后的分布重新采样。

总的拒绝概率：
$$P[\text{reject}] = 1 - \sum_{x'} \min(q(x'), p(x')) = 1 - \sum_{x'} \min(q(x'), p(x'))$$

被拒绝后，我们从调整后的分布采样。调整后的分布定义为：

$$p_{\text{adj}}(x) = \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}$$

**最终输出**：

P[输出 = x] = P[第一次采样 x 且接受] + P[第一次拒绝，第二次采样 x]

$$= \min(q(x), p(x)) + P[\text{reject}] \cdot p_{\text{adj}}(x)$$

代入 p_adj：

$$= \min(q(x), p(x)) + \left(1 - \sum_{x'} \min(q(x'), p(x'))\right) \cdot \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}$$

注意到：
- $\min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x)$
- $\sum_{x'} \min(q(x'), p(x')) + \sum_{x'} \max(0, p(x') - q(x')) = \sum_{x'} p(x') = 1$

因此：

$$P[\text{输出} = x] = \min(q(x), p(x)) + \left(\sum_{x'} \max(0, p(x') - q(x'))\right) \cdot \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}$$

$$= \min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x)$$

**Q.E.D.**

### 2.4 期望接受率

期望接受的草稿数量：

$$\mathbb{E}[\text{accepted}] = \sum_{x} \min(p(x), q(x))$$

接受率：

$$\alpha = \frac{\mathbb{E}[\text{accepted}]}{\gamma} = \frac{1}{\gamma} \sum_{x} \min(p(x), q(x))$$

**直观理解**：接受率等于 p 和 q 的重叠程度。

### 2.5 接受率下界 ⭐

**定理**：接受率有下界 $\alpha \geq \frac{1}{1 + D_{KL}(p \| q)}$

**证明**：

由 KL 散度的定义：

$$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \sum_x p(x) \log \frac{p(x)}{\min(p(x), q(x))} \cdot \frac{\min(p(x), q(x))}{q(x)}$$

$$= \underbrace{\sum_x p(x) \log \frac{p(x)}{\min(p(x), q(x))}}_{\text{记为 } A} + \underbrace{\sum_x p(x) \log \frac{\min(p(x), q(x))}{q(x)}}_{\text{记为 } B}$$

对于 A：使用 $\log(1+x) \leq x$，有：

$$A = \sum_x p(x) \log \left(1 + \frac{p(x) - \min(p(x), q(x))}{\min(p(x), q(x))}\right) \leq \sum_x p(x) - \min(p(x), q(x)) = 1 - \sum_x \min(p(x), q(x))$$

对于 B：当 $p(x) \geq q(x)$ 时，$\frac{\min(p(x), q(x))}{q(x)} = 1$，$\log 1 = 0$；当 $p(x) < q(x)$ 时，$\frac{\min(p(x), q(x))}{q(x)} = \frac{p(x)}{q(x)} < 1$，$\log < 0$。

因此 $B \leq 0$。

所以：

$$D_{KL}(p \| q) \leq 1 - \sum_x \min(p(x), q(x)) = 1 - \gamma \alpha$$

$$\alpha \geq \frac{1 - D_{KL}(p \| q)}{\gamma}$$

当 $D_{KL}(p \| q) < 1$ 时，这个界是有效的。

**实用意义**：drafter 与 target 越接近（KL 散度越小），接受率越高。

### 2.6 样本调整的数学原理 ⭐

当第 n 个草稿被拒绝时，我们需要从调整后的分布采样：

$$p_{\text{adj}}(x) = \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}$$

**为什么是这样？**

考虑条件概率：已知前 n-1 个草稿被接受，第 n 个被拒绝。

"第 n 个被拒绝" 意味着：采样到 x̂_n，但拒绝采样拒绝了它。

由 Bayes 公式，x̂_n 的后验分布（在拒绝条件下）：

$$P[x̂_n = x \mid \text{reject}] \propto q(x) \cdot \left(1 - \min\left(1, \frac{p(x)}{q(x)}\right)\right) = q(x) - \min(q(x), p(x)) = \max(0, q(x) - p(x))$$

但是！我们要采样的不是 x̂_n，而是**新的 token**。

关键洞察：如果 q(x) > p(x)，说明 drafter "过度分配"了概率给 x，这些"多余的"概率需要重新分配。

$$p_{\text{adj}}(x) \propto p(x) - \min(p(x), q(x)) = \max(0, p(x) - q(x))$$

归一化后即得到调整后的分布。

**代码实现** (`speculative_decoding.py:10-19`)：

```python
def max_fn(x: torch.Tensor) -> torch.Tensor:
    """实现 max(0, x) 的归一化"""
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum
```

调用 (`speculative_decoding.py:168`)：

```python
p_p = max_fn(p[..., n, :] - q[0, n, :])  # 调整后的分布
```

---

## 3. 算法流程详解

### 3.1 完整算法流程

```
输入: 序列 [x₁, ..., x_t]
参数: γ (草稿数量)

┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Drafter 生成 γ 个草稿                               │
├─────────────────────────────────────────────────────────────┤
│ for k = 1 to γ:                                             │
│   q_k = drafter([x₁, ..., x_{t+k-1}])                      │
│   x̂_{t+k} ~ q_k                                            │
│ end for                                                     │
│                                                             │
│ 输出: [x̂_{t+1}, x̂_{t+2}, ..., x̂_{t+γ}]                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Target 并行验证                                    │
├─────────────────────────────────────────────────────────────┤
│ p = target([x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+γ}])        │
│                                                             │
│ 输出: p([·|x₁,...,x_t,x̂_{t+1},...,x̂_{t+γ}])                │
│       = [p_{t+1}, p_{t+2}, ..., p_{t+γ+1}]                 │
│                                                             │
│ 注意: 一次前向传播得到 γ+1 个位置的概率分布！                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: 拒绝采样                                           │
├─────────────────────────────────────────────────────────────┤
│ n = γ                                                        │
│ for i = 1 to γ:                                             │
│   r ~ Uniform(0, 1)                                         │
│   if r > p_{t+i}(x̂_{t+i}) / q_i(x̂_{t+i}):                │
│     n = i-1   # 找到第一个被拒绝的位置                      │
│     break                                                   │
│   end if                                                    │
│ end for                                                     │
│                                                             │
│ 输出: n (被接受的草稿数量)                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: 采样最终 token                                      │
├─────────────────────────────────────────────────────────────┤
│ if n == γ:  # 全部草稿被接受                                │
│   x_{t+γ+1} ~ p_{t+γ+1}                                    │
│ else:        # 第 n 个草稿被拒绝                            │
│   p_adj(x) = normalize(max(0, p_{t+n}(x) - q_n(x)))       │
│   x_{t+n+1} ~ p_adj                                         │
│   修剪 KV cache，移除被拒绝的草稿                           │
│ end if                                                      │
│                                                             │
│ 输出: [x_{t+1}, ..., x_{t+n+1}] (n+1 个新 token)           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 关键优化：动态 Gamma 调整

代码 (`speculative_decoding.py:106`)：

```python
corrected_gamma = min(gamma, total_len - current_position - 1)
```

**原因**：接近序列末尾时，减少草稿数量避免溢出。

### 3.3 首次 Target Pass（可选）

代码 (`speculative_decoding.py:84-103`)：

```python
if first_target:
    Mp = target(input_ids[..., :current_position], ...)
    p_p = logits_processor(Mp.logits[..., -1, :])
    t = logits_processor.sample(p_p)
    input_ids[0, current_position] = t
    current_position += 1
```

**好处**：
- 预填充 KV Cache
- 避免 drafter 第一个草稿可能的不准确

---

## 4. 加速比分析

### 4.1 理想加速比推导 ⭐

**定义**：
- $T_t$：target 模型单次前向传播时间
- $T_d$：drafter 模型单次前向传播时间
- $\gamma$：草稿数量
- $\alpha$：接受率

**传统自回归生成 N 个 token**：
$$T_{\text{AR}} = N \cdot T_t$$

**Speculative Decoding 生成 N 个 token**：

每次迭代：
1. Drafter 生成 γ 个草稿：$\gamma \cdot T_d$
2. Target 验证 γ 个草稿：$T_t$

每次迭代生成 $1 + \gamma \cdot \alpha$ 个 token（1 个采样 token + 平均接受的草稿）。

需要迭代 $\frac{N}{1 + \gamma \cdot \alpha}$ 次。

$$T_{\text{SD}} = \frac{N}{1 + \gamma \cdot \alpha} (\gamma \cdot T_d + T_t)$$

**加速比**：

$$\text{Speedup} = \frac{T_{\text{AR}}}{T_{\text{SD}}} = \frac{N \cdot T_t}{\frac{N}{1 + \gamma \cdot \alpha} (\gamma \cdot T_d + T_t)}$$

$$= \frac{1 + \gamma \cdot \alpha}{1 + \gamma \cdot \frac{T_d}{T_t}}$$

**理想情况**（$T_d \ll T_t$，即 drafter 远快于 target）：

$$\text{Speedup} \approx 1 + \gamma \cdot \alpha$$

### 4.2 影响加速比的因素

| 因素 | 影响 | 优化方向 |
|------|------|----------|
| Drafter 速度 | $T_d$ 越小，加速越大 | 选择更小、更快的 drafter |
| 接受率 | $\alpha$ 越大，加速越大 | drafter 与 target 分布越接近越好 |
| 草稿数量 | $\gamma$ 过大会降低 $\alpha$ | 权衡选择最优值 |

### 4.3 最优 γ 的理论分析 ⭐

**目标**：最大化加速比 $\text{Speedup}(\gamma) = \frac{1 + \gamma \cdot \alpha(\gamma)}{1 + \gamma \cdot \rho}$

其中 $\rho = T_d / T_t$，$\alpha(\gamma)$ 是 γ 的函数（通常随 γ 增加而减小）。

**启发式分析**：

- γ 太小：加速有限
- γ 太大：接受率下降，抵消加速效果

**经验法则**：

$$\gamma^* \approx \sqrt{\frac{1}{\rho} - 1}$$

当 $\rho = 0.1$（drafter 是 target 时间的 10%）：
$$\gamma^* \approx \sqrt{9} = 3$$

当 $\rho = 0.05$（drafter 是 target 时间的 5%）：
$$\gamma^* \approx \sqrt{19} \approx 4$$

### 4.4 实际加速比示例

假设：$T_t = 100$ ms，$T_d = 10$ ms（$\rho = 0.1$）

| γ | α | 加速比计算 | 加速比 |
|---|---|---------|--------|
| 2 | 0.8 | (1+1.6)/(1+0.2) | 2.17× |
| 4 | 0.7 | (1+2.8)/(1+0.4) | 2.71× |
| 6 | 0.6 | (1+3.6)/(1+0.6) | 2.88× |
| 8 | 0.5 | (1+4.0)/(1+0.8) | 2.78× |

**最优 γ ≈ 6**，加速比约 2.9×。

---

## 5. 与蒙特卡洛方法的关系

### 5.1 Speculative Decoding 作为拒绝采样

**拒绝采样**（Rejection Sampling）是从目标分布采样的经典方法：

```
目标: 从 p(x) 采样
提议分布: q(x)，满足 p(x) ≤ M·q(x)

算法:
1. 从 q(x) 采样 x̂
2. 从 Uniform(0, 1) 采样 u
3. 如果 u < p(x̂)/(M·q(x̂))，接受 x̂；否则拒绝
```

**Speculative Decoding** 是拒绝采样的变体：
- 提议分布：drafter 的分布 q
- 目标分布：target 的分布 p
- M = 1（因为我们使用 min(1, p/q)）

### 5.2 与重要性采样的对比

| 特性 | Speculative Decoding | 重要性采样 |
|------|---------------------|-----------|
| 输出分布 | 精确等于 p | 近似 p |
| 是否需要修正 | 是（样本调整） | 权重修正 |
| 方差 | 无偏差 | 有方差 |
| 适用场景 | 需要精确分布 | 可以接受近似 |

### 5.3 序列决策视角

Speculative Decoding 也可以看作是一个**序列决策问题**：

- **状态**：当前已生成的序列
- **动作**：选择下一个 token
- **奖励**：对数概率
- **策略**：drafter 提供策略，target 评估价值

拒绝采样保证了价值函数的一致性。

---

## 6. 小结

### 关键要点

1. **核心思想**：用小模型快速生成候选，大模型并行验证
2. **数学保证**：拒绝采样确保输出分布精确等于目标分布
3. **加速比**：理想情况下 $\text{Speedup} = 1 + \gamma \cdot \alpha$
4. **权衡**：γ、接受率、drafter 速度之间的平衡

### 关键公式

| 公式 | 含义 |
|------|------|
| $P_{\text{accept}}(x) = \min(1, p(x)/q(x))$ | 接受概率 |
| $\alpha = \frac{1}{\gamma}\sum_x \min(p(x), q(x))$ | 期望接受率 |
| $p_{\text{adj}}(x) = \frac{\max(0, p(x)-q(x))}{\sum \max(0, p-q)}$ | 调整后分布 |
| $\text{Speedup} = \frac{1+\gamma\alpha}{1+\gamma\rho}$ | 加速比 |

### 下一步

阅读 [03. Speculative Decoding 实现](./03-speculative-impl.md) 了解代码实现细节。

### 代码参考

| 功能 | 文件路径 | 行号 |
|------|----------|------|
| 主函数 | `sampling/speculative_decoding.py` | 23-189 |
| max_fn (样本调整) | `sampling/speculative_decoding.py` | 10-19 |
| 草稿生成循环 | `sampling/speculative_decoding.py` | 112-126 |
| 并行验证 | `sampling/speculative_decoding.py` | 129-137 |
| 拒绝采样 | `sampling/speculative_decoding.py` | 139-147 |
| 样本调整 | `sampling/speculative_decoding.py` | 158-171 |
