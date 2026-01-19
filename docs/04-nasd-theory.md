# 04. NASD 理论

## 目录

- [1. 背景与动机](#1-背景与动机)
- [2. Ngram 基础知识](#2-ngram-基础知识)
- [3. NASD 算法原理](#3-nasd-算法原理)
- [4. NASD vs 标准 Speculative Decoding](#4-nasd-vs-标准-speculative-decoding)
- [5. 与 NAPD 的关系](#5-与-napd-的关系)
- [6. 小结](#6-小结)

---

## 1. 背景与动机

### 1.1 Speculative Decoding 的局限性

标准 Speculative Decoding 需要：
- 一个训练好的 drafter 模型
- Drafter 和 target 必须共享 tokenizer
- Drafter 需要与 target 有相似的分布

**问题**：
- 训练 drafter 需要额外成本
- Drafter 选择影响加速效果
- 小模型的 drafter 可能质量不够

### 1.2 关键洞察

**洞察 1**：大语言模型具有**局部模式重复**的特性

```
Prompt: "The capital of France is"
        "The capital of Germany is"
        "The capital of Italy is"

Pattern: "The capital of X is" → [location name]
```

**洞察 2**：统计模型可以捕捉这些局部模式

- N-gram 模型简单、快速
- 无需训练（从数据中学习）
- 计算开销极小

### 1.3 NASD 的核心思想

**Ngram Assisted Speculative Decoding (NASD)**：

> 用统计 N-gram 模型替代神经网络作为 drafter

**优势**：
- ✅ 训练无关（training-free）
- ✅ 任务无关（task-agnostic）
- ✅ 模型无关（model-agnostic）
- ✅ 推理速度极快
- ✅ 内存占用小

---

## 2. Ngram 基础知识

### 2.1 什么是 N-gram？

**N-gram**：连续 N 个 token 的序列

```
Unigram (1-gram):    "The"
Bigram (2-gram):     "The capital"
Trigram (3-gram):    "The capital of"
4-gram:              "The capital of France"
```

### 2.2 N-gram 语言模型

**基本思想**：基于历史 N-1 个 token 预测下一个 token

$$P(x_t | x_{t-N+1}, ..., x_{t-1}) = \frac{\text{count}(x_{t-N+1}, ..., x_t)}{\text{count}(x_{t-N+1}, ..., x_{t-1})}$$

**示例**：

```
训练数据: "The capital of France is Paris. The capital of Germany is Berlin."

Trigram 模型预测:
P("Paris" | "The capital of France is") = 1/1 = 100%
P("Berlin" | "The capital of Germany is") = 1/1 = 100%
P("Paris" | "The capital of Germany is") = 0/1 = 0%
```

### 2.3 回退策略 (Backoff)

当高阶 N-gram 未见时，回退到低阶：

```
5-gram: "The capital of France is" → 未见
4-gram: "capital of France is" → 未见
3-gram: "of France is" → 未见
2-gram: "France is" → 见过，预测 "Paris"
```

**公式**：

$$P_{\text{backoff}}(x_t | \text{context}) = \begin{cases} P_{\text{ngram}}(x_t | \text{context}) & \text{if seen} \\ P_{\text{backoff}}(x_t | \text{context}[1:]) & \text{otherwise} \end{cases}$$

---

## 3. NASD 算法原理

### 3.1 整体框架

```
给定: 当前序列 [x₁, ..., x_t]

Step 1 (Ngram Drafter): 生成 γ 个草稿
  for k = 1 to γ:
    context = [x_{t+k-N+1}, ..., x_{t+k-1}]
    x̂_{t+k} = argmax P(x | context)  # Ngram 预测
  end for

Step 2 (Target): 并行验证 γ 个草稿
  p = target([x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+γ}])

Step 3 (Deterministic Verification): 确定验证
  n = γ
  for i = 1 to γ:
    # 检查 ngram 预测是否与 target 的 top-1 一致
    if x̂_{t+i} ≠ argmax p_{t+i}:
      n = i-1
      break

Step 4 (Sample): 生成最终 token
  if n == γ:  # 全部匹配
    x_{t+γ+1} ~ p_{t+γ+1}
  else:      # 第 n 个不匹配
    x_{t+n+1} ~ p_{t+n}
```

### 3.2 关键差异：确定性验证

**标准 Speculative Decoding**：
```
接受概率: min(1, p(x̂) / q(x̂))
随机性: 需要
```

**NASD**：
```
接受条件: x̂ == argmax p
随机性: 不需要（deterministic）
```

**为什么确定性验证可以工作？**

Ngram 预测的是**最可能的下一个 token**，如果与 target 的 top-1 一致，说明预测正确。

### 3.3 Ngram Storage 设计

**数据结构**：

```python
counts = {
    (context_1): {token_1: count_1, token_2: count_2, ...},
    (context_2): {token_1: count_1, token_2: count_2, ...},
    ...
}

ngrams = {
    (context_1): token_with_highest_count,
    (context_2): token_with_highest_count,
    ...
}
```

**预测**：

```python
def predict(context):
    if context in ngrams:
        return ngrams[context], True  # 已知
    else:
        return random_token(), False  # 未知
```

### 3.4 初始化与更新

**初始化**（从 prompt 中学习）：

```python
# Prompt: "The capital of France is"
# 提取所有 N-gram

N=3 (trigram):
  ("The", "capital") → "of"
  ("capital", "of") → "France"
  ("of", "France") → "is"
  ("France", "is") → ? (未见)
```

**在线更新**（生成过程中学习）：

```python
# 每接受一个草稿
ngramstorage.update(context, accepted_token)

# Top-K Filler: 同时更新多个候选
ngramstorage.update(context, top_k_tokens)
```

### 3.5 Top-K Filler 的作用

**问题**：Ngram 只知道历史中见过的模式，可能错过新的模式。

**Top-K Filler**：用 target 的 top-K 候选更新 ngram

```python
# Target 预测: p = [Paris: 0.6, Berlin: 0.2, London: 0.1, ...]
# Top-K=3: 更新 Paris, Berlin, London

ngramstorage.update(context, [Paris, Berlin, London])
```

**效果**：
- 增加模式的多样性
- 提高草稿接受率
- 加快适应性

---

## 4. NASD vs 标准 Speculative Decoding

### 4.1 算法对比

| 特性 | 标准 Speculative Decoding | NASD |
|------|---------------------------|------|
| Drafter | 神经网络模型 | 统计 N-gram |
| 训练需求 | 需要 | 不需要 |
| 验证方式 | 概率性 (p/q ratio) | 确定性 (top-1 match) |
| 样本调整 | 需要 | 不需要 |
| 内存占用 | 大 (模型权重) | 小 (统计计数) |
| 速度 | 快 | 极快 |

### 4.2 验证机制对比

**标准 Speculative Decoding**：

```python
# 概率性验证
r = torch.rand(1)
if r < p(x̂) / q(x̂):
    accept(x̂)
else:
    reject(x̂)
```

**NASD**：

```python
# 确定性验证
if x̂ == argmax(p):
    accept(x̂)
else:
    reject(x̂)
```

**为什么 NASD 不需要样本调整？**

- 确定性验证保证：接受草稿 = target 的 top-1 预测
- 当拒绝发生时，直接从 target 的分布采样
- 无需复杂的样本调整步骤

### 4.3 接受率对比

| 场景 | 标准 Speculative Decoding | NASD |
|------|---------------------------|------|
| Drafter 与 target 接近 | 高 (0.7-0.9) | 取决于模式重复度 |
| 长重复模式 | 高 | 很高 (接近 1.0) |
| 短重复模式 | 高 | 中等 (0.5-0.7) |
| 无重复模式 | 中等 | 低 (0.3-0.5) |

**关键洞察**：NASD 在**有重复模式**的场景下表现优异。

---

## 5. 与 NAPD 的关系

### 5.1 NAPD 简介

**NAPD** (Adaptive N-gram Parallel Decoding) 是类似的并发方法：
- 同样使用 N-gram 模型
- 同样确定性验证
- 关键区别：只更新被接受的草稿

### 5.2 参数对应

| NASD 参数 | NAPD 参数 | 值 |
|-----------|-----------|---|
| `filler_top_k` | 只更新接受的 | 1 |

### 5.3 等价性

当 `filler_top_k = 1` 时，NASD 等价于 NAPD：

```python
# NASD with filler_top_k=1
ngramstorage.update(context, accepted_token)  # 只更新接受的

# NAPD
ngramstorage.update(context, accepted_token)  # 只更新接受的
```

### 5.4 Top-K Filler 的优势

`filler_top_k > 1` 时：
- 更快适应新模式
- 更高的草稿接受率
- 更好的加速效果

---

## 6. 小结

### 关键要点

1. **NASD 核心思想**：用统计 N-gram 替代神经网络 drafter
2. **优势**：训练无关、任务无关、模型无关
3. **验证方式**：确定性验证（top-1 match）
4. **Top-K Filler**：用 target 的 top-K 更新 ngram，提高适应性和接受率
5. **适用场景**：有重复模式的文本（代码、公式、模板化文本）

### 算法复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| Ngram 预测 | O(1) | 字典查找 |
| Ngram 更新 | O(1) | 字典更新 |
| Target 验证 | O(1) | 单次前向传播 |
| 内存占用 | O(|V|^n × n) | |V| 为词表大小，n 为阶数 |

### 优缺点总结

| 优点 | 缺点 |
|------|------|
| 训练无关 | 首次运行需要从 prompt 学习 |
| 极快速度 | 模式未知时接受率低 |
| 内存占用小 | 需要存储 N-gram 统计 |
| 适应性强 | 适应性依赖于 Top-K Filler |

### 下一步

阅读 [05. NASD 实现](./05-nasd-impl.md) 了解代码实现细节。

### 代码参考

| 功能 | 文件路径 | 行号 |
|------|----------|------|
| NASD 主函数 | `ngram_assisted/ngram_assisted.py` | 11-164 |
| INgramStorage 接口 | `ngram_assisted/ngram_storage.py` | 5-69 |
| OneLevelNGramStorage | `ngram_assisted/ngram_storage.py` | 73-150 |
| NGramStorage (多级) | `ngram_assisted/ngram_storage.py` | 154-249 |
| Ngram 初始化 | `ngram_assisted/ngram_assisted.py` | 71 |
| 草稿生成 | `ngram_assisted/ngram_assisted.py` | 95-101 |
| 确定性验证 | `ngram_assisted/ngram_assisted.py` | 115-120 |
| Ngram 更新 | `ngram_assisted/ngram_assisted.py` | 149-155 |
