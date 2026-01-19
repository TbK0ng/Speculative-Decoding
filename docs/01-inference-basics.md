# 01. 大模型推理基础

## 目录

- [1. 自回归生成原理](#1-自回归生成原理)
- [2. KV Cache 的作用和原理](#2-kv-cache-的作用和原理)
- [3. 传统生成方法](#3-传统生成方法)
- [4. 推理瓶颈分析](#4-推理瓶颈分析)
- [5. 小结](#5-小结)

---

## 1. 自回归生成原理

### 1.1 什么是自回归生成？

**自回归生成 (Autoregressive Generation)** 是大语言模型（LLM）生成文本的核心方式。它的基本思想是：**基于已生成的序列，预测下一个 token**。

给定一个上下文序列 $x_1, x_2, ..., x_t$，模型预测下一个 token $x_{t+1}$ 的概率分布：

$$P(x_{t+1} | x_1, x_2, ..., x_t)$$

然后根据这个概率分布采样（或选择最可能的）token，将其加入序列，继续预测下一个 token。

### 1.2 为什么需要逐 token 生成？

这是一个根本性的限制，源于 **Transformer 的因果掩码 (Causal Masking)** 设计：

```
预测 x2 时只能看到:  [x1, <mask>, <mask>, ...]
预测 x3 时只能看到:  [x1, x2, <mask>, ...]
预测 x4 时只能看到:  [x1, x2, x3, ...]
```

每个位置只能依赖之前的位置，因此必须**串行**生成。这是推理速度的主要瓶颈。

### 1.3 生成流程示意

```
输入: "The capital of France is"

Step 1: 模型看到 "The capital of France is"
        → 输出概率分布 P(next_token | context)
        → 选择 " Paris"

Step 2: 模型看到 "The capital of France is Paris"
        → 输出概率分布 P(next_token | context + " Paris")
        → 选择 "."

Step 3: 模型看到 "The capital of France is Paris."
        → 检测到结束符，停止生成

输出: "Paris."
```

---

## 2. KV Cache 的作用和原理

### 2.1 重复计算问题

如果不使用 KV Cache，每次生成新 token 时：

```python
# 第 t 步
model(input_ids[:, :t])      # 计算所有 t 个位置的 attention

# 第 t+1 步
model(input_ids[:, :t+1])    # 重新计算所有 t+1 个位置的 attention！
```

前 t 个位置的 attention 键值对被**重复计算**了，浪费大量计算资源。

### 2.2 KV Cache 原理

**KV Cache** 的核心思想：**缓存每个位置的 Key 和 Value，避免重复计算**。

```python
# 第 t 步：计算并缓存
outputs = model(input_ids[:, :t])
past_kv = outputs.past_key_values  # 保存 K, V

# 第 t+1 步：只计算新位置
outputs = model(
    input_ids[:, t:t+1],           # 只输入新的 token
    past_key_values=past_kv        # 传入缓存
)
# 新的 past_kv 会包含 t+1 个位置的 K, V
```

### 2.3 KV Cache 数据结构

每个 Transformer 层都维护一个 KV Cache：

```
Layer 0: K[0] = [k_1, k_2, ..., k_t], V[0] = [v_1, v_2, ..., v_t]
Layer 1: K[1] = [k_1, k_2, ..., k_t], V[1] = [v_1, v_2, ..., v_t]
...
Layer L: K[L] = [k_1, k_2, ..., k_t], V[L] = [v_1, v_2, ..., v_t]
```

形状通常为：`[batch_size, num_heads, seq_len, head_dim]`

### 2.4 项目中的实现

在 `sampling/base_decoding.py:52-57`：

```python
for curr in range(prompt_len, total_len):
    o = model(input_ids[..., :curr], past_key_values=cache, use_cache=use_cache)
    logits = o.logits[..., -1, :]
    probs = logits_processor(logits)
    x = logits_processor.sample(probs)
    input_ids[0, curr] = x
    cache = o.past_key_values  # 更新缓存
```

- `use_cache=True` 启用 KV Cache
- `past_key_values=cache` 传入历史缓存
- `cache = o.past_key_values` 获取更新后的缓存

### 2.5 KV Cache 的代价

- **内存开销**：每个 token 需要 `2 × num_layers × num_heads × head_dim × bytes_per_param` 字节
- **例如**：Llama-3-8B，32 层，32 头，128 维，float16
  - 每个 token: `2 × 32 × 32 × 128 × 2 = 524,288` 字节 ≈ 0.5 MB
  - 2048 context: ≈ 1 GB

---

## 3. 传统生成方法

### 3.1 Greedy Decoding（贪婪解码）

**核心思想**：每步选择概率最高的 token。

```python
token = argmax(P(x_t | x_1, ..., x_{t-1}))
```

**优点**：
- 简单快速
- 确定性输出

**缺点**：
- 可能陷入重复循环
- 缺乏多样性
- 不是全局最优

**项目实现**：`utils/logits_processor.py` 中的 `GreedyProcessor`

### 3.2 Beam Search（集束搜索）

**核心思想**：维护多个候选序列，选择最终得分最高的。

```
初始: ["The"]

Step 1 (beam_size=2):
  ["The capital"]     (score: 0.3)
  ["The city"]        (score: 0.2)

Step 2:
  ["The capital of"]  (score: 0.25)
  ["The city is"]     (score: 0.18)
  ... 其他低分候选被丢弃
```

**长度惩罚**（`base_decoding.py:106-107`）：

```python
def _length_penalty_fn(length, alpha, min_length):
    return ((min_length + length) / (min_length + 1)) ** alpha
```

避免过度偏向短序列。

**项目实现**：`sampling/base_decoding.py:69-187` 的 `beam_search_generate()`

### 3.3 Nucleus Sampling（Top-p 采样）

**核心思想**：从累积概率达到 p 的最小集合中采样。

```python
# 按概率降序排序
sorted_probs = sort(probs, descending=True)

# 找到最小集合，使 cumulative_prob >= p
cumsum = 0
selected_tokens = []
for prob in sorted_probs:
    selected_tokens.append(prob.token)
    cumsum += prob.value
    if cumsum >= p:
        break

# 从 selected_tokens 中按概率采样
token = sample(selected_tokens)
```

**项目实现**：`utils/logits_processor.py` 中的 `NucleusProcessor`

### 3.4 Top-K Sampling

**核心思想**：只从概率最高的 K 个 token 中采样。

```python
top_k_probs, top_k_tokens = topk(probs, k=K)
token = sample(top_k_probs, top_k_tokens)
```

**项目实现**：`utils/logits_processor.py` 中的 `TopKProcessor`

### 3.5 采样策略对比

| 策略 | 多样性 | 质量 | 速度 | 适用场景 |
|------|--------|------|------|----------|
| Greedy | 低 | 高 | 快 | 确定性任务 |
| Beam Search | 低 | 高 | 慢 | 翻译、摘要 |
| Nucleus | 高 | 中 | 中 | 创意写作 |
| Top-K | 中 | 中 | 快 | 通用 |

---

## 4. 推理瓶颈分析

### 4.1 内存墙问题

**计算 vs 内存访问速度**：

```
GPU 计算:   ~100 TFLOPS
GPU 内存:   ~2 TB/s 带宽
```

每次 forward pass：
- **计算量**：O(seq_len × d_model²)
- **内存访问**：O(seq_len × d_model)

当 seq_len 较小时，**内存传输成为瓶颈**，而非计算能力。

### 4.2 自回归生成的效率

**时间复杂度分析**：

生成 N 个 token 需要：
- N 次 forward pass
- 每次 pass 只产生 1 个新 token
- **利用率**：1/N（极低！）

```
示例：生成 100 个 token

传统方法: 100 次 forward pass
理想情况: 1 次 forward pass（并行）
利用率:   1%
```

### 4.3 小模型 vs 大模型

| 模型大小 | 参数量 | 推理速度 | 质量 |
|----------|--------|----------|------|
| 1B | 小 | 快 | 中 |
| 3B | 中 | 中 | 良 |
| 8B | 大 | 慢 | 高 |
| 70B | 很大 | 很慢 | 很高 |

**权衡**：
- 小模型：快但质量不够
- 大模型：质量高但太慢

### 4.4 加速方向

**Speculative Decoding 的核心思想**：

> 用小模型快速生成候选，大模型并行验证，结合两者优势！

```
小模型 (drafter):  串行生成 γ 个候选  → 快
大模型 (target):   并行验证 γ 个候选  → 质量高
```

下一章将详细介绍 Speculative Decoding 的原理。

---

## 5. 小结

### 关键要点

1. **自回归生成**：LLM 必须逐 token 生成，这是根本性限制
2. **KV Cache**：缓存键值对避免重复计算，但增加内存开销
3. **采样策略**：Greedy、Beam Search、Nucleus、Top-K 各有优劣
4. **推理瓶颈**：内存墙 + 自回归限制 = 低效的串行生成

### 下一步

阅读 [02. Speculative Decoding 理论](./02-speculative-theory.md) 了解如何突破这些限制。

### 代码参考

| 功能 | 文件路径 |
|------|----------|
| 自回归生成实现 | `sampling/base_decoding.py:10-65` |
| Beam Search 实现 | `sampling/base_decoding.py:69-187` |
| Logits Processors | `utils/logits_processor.py` |
| KV Cache 管理 | `utils/caching.py` |
