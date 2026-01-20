# 06. 技巧与优化

## 目录

- [1. 并行 Token 验证（核心优化）](#1-并行-token-验证核心优化)
- [2. 首次 Target Pass](#2-首次-target-pass)
- [3. 动态 Gamma 调整](#3-动态-gamma-调整)
- [4. 熵自适应 Gamma（EGAG）](#4-熵自适应-gammaegag)
- [5. 多 EOS Token 支持](#5-多-eos-token-支持)
- [6. Ngram 回退策略](#6-ngram-回退策略)
- [7. Top-K Filler](#7-top-k-filler)
- [8. 跳过样本调整](#8-跳过样本调整)
- [9. 其他实现技巧](#9-其他实现技巧)
- [10. 小结](#10-小结)

---

## 1. 并行 Token 验证（核心优化）

### 1.1 核心洞察

**Transformer 可以并行计算多个位置的概率分布！**

这是 Speculative Decoding 加速的关键。

### 1.2 原理说明

```python
# 输入序列
input_ids = [x₁, x₂, ..., x_t, x̂_{t+1}, x̂_{t+2}, ..., x̂_{t+γ}]

# 单次前向传播
outputs = target(input_ids)

# 输出 logits
outputs.logits.shape = [batch, seq_len, vocab_size]
                     = [1, t+γ, vocab_size]

# 关键：outputs.logits[..., t-1:t+γ-1, :] 包含了 γ 个位置的预测
# - logits[..., t-1, :]: 基于 [x₁, ..., x_t] 的预测
# - logits[..., t, :]:   基于 [x₁, ..., x_t, x̂_{t+1}] 的预测
# - ...
# - logits[..., t+γ-2, :]: 基于 [x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+γ-1}] 的预测
```

### 1.3 代码实现

```python
# sampling/speculative_decoding.py:129-136
Mp = target(
    input_ids=input_ids[..., :current_position + corrected_gamma],
    past_key_values=target_cache,
    use_cache=use_cache,
)
target_cache = Mp.past_key_values

# 提取 γ 个草稿位置的概率分布
draft_logits = Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :]
p = logits_processor(draft_logits)  # [1, γ, vocab_size]
```

### 1.4 为什么这是关键？

**没有并行验证**：

```
Step 1: target([x₁, ..., x_t, x̂_{t+1}]) → p₁
Step 2: target([x₁, ..., x_t, x̂_{t+1}, x̂_{t+2}]) → p₂
...
Step γ: target([x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+γ}]) → p_γ

需要: γ 次前向传播
```

**有并行验证**：

```
Step 1: target([x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+γ}]) → [p₁, p₂, ..., p_γ]

需要: 1 次前向传播
```

**加速比**：γ 倍！

---

## 2. 首次 Target Pass

### 2.1 作用

在 Speculative Decoding 主循环之前，先用 target 生成第一个 token。

### 2.2 代码实现

```python
# sampling/speculative_decoding.py:84-103
if first_target:
    Mp = target(
        input_ids=input_ids[..., :current_position],
        past_key_values=target_cache,
        use_cache=use_cache,
    )
    target_cache = Mp.past_key_values
    p_p = logits_processor(Mp.logits[..., -1, :])
    t = logits_processor.sample(p_p)
    input_ids[0, current_position] = t
    current_position += 1

    if torch.isin(t, stop_tokens):
        return input_ids[0, prompt_len:current_position].tolist(), 0
```

### 2.3 优势

1. **预填充 KV Cache**：target 的 KV cache 被初始化
2. **避免初始错误**：drafter 第一个草稿可能不准确
3. **更稳定**：给算法一个良好的起始点

### 2.4 何时使用？

- **推荐**：大多数场景
- **不推荐**：prompt 非常短且 drafter 质量很高

---

## 3. 动态 Gamma 调整

### 3.1 问题

接近序列末尾时，固定的 γ 可能导致溢出。

```python
# total_len = 100, current_position = 98, gamma = 5
# 如果还生成 5 个草稿，会超出 total_len
```

### 3.2 解决方案

```python
# sampling/speculative_decoding.py:106
corrected_gamma = min(gamma, total_len - current_position - 1)
```

### 3.3 效果

```
Position 95: gamma = 5  → corrected_gamma = 5
Position 96: gamma = 5  → corrected_gamma = 4
Position 97: gamma = 5  → corrected_gamma = 3
Position 98: gamma = 5  → corrected_gamma = 2
Position 99: gamma = 5  → corrected_gamma = 1 (最后一个)
```

---

## 4. 熵自适应 Gamma（EGAG）

### 4.1 动机

固定 $\gamma$ 在困难 token 上浪费草稿（高否决率），在简单 token 上又无法充分并行。

### 4.2 核心做法

对 drafter 的当前 logits 计算熵 $H$，映射为动态 $\gamma$：

$$H=-\sum_i p_i \log p_i$$
$$H_{norm}=\frac{H}{\log V}$$
$$\gamma=\lfloor (1-H_{norm})\cdot \gamma_{max} \rfloor,\ \gamma\ge 1$$

### 4.3 实现要点

- 在生成草稿前计算一次熵并更新 $\gamma$。
- 建议使用上下界 $[\gamma_{min}, \gamma_{max}]$。
- 可使用 EMA 平滑避免频繁抖动。

**CLI 参数（infer.py）**
- `/egag`
- `/egag_gamma_min <value>`
- `/egag_gamma_max <value>`
- `/egag_ema <value>`

### 4.4 适用场景

- 生成任务不稳定、接收率波动大时。
- 多样性较高、内容难度起伏较大的文本。

---

## 5. 多 EOS Token 支持

### 4.1 背景

Chat 模型可能有多个结束符：
- `<|eot_id|>`: Chat end token
- `<|end_of_text|>`: General end token
- `<|im_end|>`: Instruction end token

### 4.2 代码实现

```python
# sampling/speculative_decoding.py:68-69
list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)
```

### 4.3 检查结束符

```python
# 检查草稿中是否有结束符
stop_locations = torch.nonzero(torch.eq(input_ids[..., current_position:current_position + n], stop_tokens))
if stop_locations.shape[0] > 0:
    stop_location = stop_locations[0, 1].item()
    return input_ids[0, prompt_len:current_position + stop_location + 1].tolist(), drafts_accepted / drafts_speculated

# 检查最终采样的 token
if torch.isin(x, stop_tokens):
    return input_ids[0, prompt_len:current_position].tolist(), drafts_accepted / drafts_speculated
```

### 4.4 重要性

**多结束符支持**使得 Speculative Decoding 可以正确处理：
- 对话生成
- 指令跟随
- 多轮对话

---

## 6. Ngram 回退策略

### 5.1 问题

高阶 N-gram 可能未见（稀疏性）。

```
5-gram: "The capital of France is" → 未见
4-gram: "capital of France is" → 未见
3-gram: "of France is" → 未见
2-gram: "France is" → 见过！预测 "Paris"
```

### 5.2 实现方式

```python
# ngram_assisted/ngram_storage.py:164-179 (NGramStorage)
def next_token(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.randint(self.vocab_size, size=(input_ids.shape[0],))
    known = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for i, seq in enumerate(input_ids):
        if seq.shape[0] < 1:
            continue

        # 从最长到最短尝试: n-1, n-2, ..., 1
        for j in range(min(self.n - 1, seq.shape[0]), 1, -1):
            gram = tuple(seq[-j:].tolist())

            if gram in self.ngrams[j]:
                out[i] = self.ngrams[j][gram]
                known[i] = True
                break  # 找到就停止

    return out, known
```

### 5.3 优势

- **鲁棒性**：即使高阶未见，仍可使用低阶
- **覆盖率**：更高的上下文覆盖率
- **适应性**：适应不同长度的模式

---

## 7. Top-K Filler

### 6.1 问题

Ngram 只知道历史中见过的模式，可能错过新的模式。

### 6.2 解决方案

用 target 的 top-K 候选更新 ngram。

```python
# ngram_assisted/ngram_assisted.py:149-155
for i in range(n):
    ngramstorage.update(input_ids[..., :current_position + i], input_ids[..., current_position + i].unsqueeze(0))

    # Top-K Filler: 用 target 的 top-K 更新
    if filler_top_k > 1:
        ngramstorage.update(input_ids[..., :current_position + i], p[..., i, :].topk(filler_top_k).indices)
```

### 6.3 效果

```python
# Target 预测: p = [Paris: 0.6, Berlin: 0.2, London: 0.1, ...]
# top_k=3: [Paris, Berlin, London]

# 更新后，下次遇到相同上下文时：
# - 可能预测 Paris（最高频）
# - 可能预测 Berlin（第二高频）
# - 可能预测 London（第三高频）
```

### 6.4 参数选择

| filler_top_k | 接受率提升 | 内存增加 | 推荐场景 |
|-------------|-----------|---------|----------|
| 1 | 基准 | 小 | 模式固定的文本 |
| 3 | +10-20% | 中 | 通用文本 |
| 10 | +20-30% | 大 | 多样化文本 |

---

## 8. 跳过样本调整

### 7.1 作用

跳过样本调整步骤，直接使用 target 的分布。

### 7.2 代码实现

```python
# sampling/speculative_decoding.py:167-170
if not skip_sample_adjustment:
    p_p = max_fn(p[..., n, :] - q[0, n, :])  # 样本调整
else:
    p_p = p[..., n, :]  # 跳过调整
```

### 7.3 用途

**调试**：验证其他部分的正确性

**性能**：在保证分布正确的前提下，可以省略 max_fn 计算

### 7.4 注意事项

⚠️ **跳过样本调整会改变输出分布！**

只在以下情况使用：
- 调试阶段
- p 和 q 非常接近
- 不需要精确分布的场景

---

## 9. 其他实现技巧

### 8.1 KV Cache 一致性

**问题**：草稿被拒绝时，cache 中包含被拒绝的 token。

**解决**：修剪 cache

```python
# sampling/speculative_decoding.py:164-165
if use_cache:
    drafter_cache = prune_cache(drafter_cache, corrected_gamma - n)
    target_cache = prune_cache(target_cache, corrected_gamma - n + 1)
```

### 8.2 设备管理

```python
# sampling/speculative_decoding.py:109, 126
input_ids = input_ids.to(drafter.device)  # 移到 drafter 设备
# ... drafter 生成草稿 ...
input_ids = input_ids.to(target.device)   # 移回 target 设备
```

**原因**：drafter 和 target 可能在不同设备上。

### 8.3 调试输出

```python
# sampling/speculative_decoding.py:173-180
if debug:
    generated = input_ids.clone().detach()

input_ids[0, current_position + n:current_position + corrected_gamma] = pad_token_id
input_ids[0, current_position + n] = x

if debug:
    printing.speculative_step(tokenizer, generated, input_ids, n, ...)
```

**输出示例**：

```
========== Speculative Step ==========
Position: 10, Gamma: 4, Accepted: 3

Drafts:
  [11]: "Paris" (q=0.85) ✓ ACCEPTED
  [12]: "," (q=0.72) ✓ ACCEPTED
  [13]: "France" (q=0.68) ✗ REJECTED

Final:
  [12]: "and" (p=0.45)

Output: "Paris, and..."
=======================================
```

### 8.4 随机数生成

```python
# sampling/speculative_decoding.py:139
r = torch.rand(corrected_gamma, device=target.device)
```

**关键**：使用 `target.device` 确保 GPU 上的随机数生成。

---

## 10. 小结

### 优化技巧总结

| 技巧 | 文件 | 行号 | 加速效果 |
|------|------|------|---------|
| 并行 Token 验证 | `sampling/speculative_decoding.py` | 129-136 | ⭐⭐⭐⭐⭐ |
| 首次 Target Pass | `sampling/speculative_decoding.py` | 84-103 | ⭐⭐ |
| 动态 Gamma 调整 | `sampling/speculative_decoding.py` | 106 | ⭐ |
| EGAG（熵自适应） | `sampling/speculative_decoding.py` | 待实现 | ⭐⭐⭐ |
| Top-K Filler | `ngram_assisted/ngram_assisted.py` | 151-155 | ⭐⭐⭐ |
| Ngram 回退策略 | `ngram_assisted/ngram_storage.py` | 171-177 | ⭐⭐ |

### 实现建议

1. **默认启用**：`first_target=True`
2. **调整 Gamma**：根据接受率动态调整
3. **使用 Top-K Filler**：`filler_top_k=3` 作为起点
4. **自适应策略**：优先实现 EGAG
4. **启用 Cache**：`use_cache=True`（注意已知问题）
5. **多结束符支持**：使用列表指定所有结束符

### 下一步

阅读 [07. 性能基准测试](./07-performance-benchmarks.md) 了解性能对比。

### 代码引用速查

| 技巧 | 文件 | 行号 |
|------|------|------|
| 并行验证 | `sampling/speculative_decoding.py` | 129-136 |
| 首次 Target Pass | `sampling/speculative_decoding.py` | 84-103 |
| 动态 Gamma | `sampling/speculative_decoding.py` | 106 |
| 多 EOS Token | `sampling/speculative_decoding.py` | 68-69 |
| KV Cache 修剪 | `utils/caching.py` | 6-77 |
| Ngram 回退 | `ngram_assisted/ngram_storage.py` | 171-177 |
| Top-K Filler | `ngram_assisted/ngram_assisted.py` | 151-155 |
| 跳过样本调整 | `sampling/speculative_decoding.py` | 167-170 |
