# 03. Speculative Decoding 实现

## 目录

- [1. 代码架构概览](#1-代码架构概览)
- [2. max_fn 函数：样本调整的核心](#2-max_fn-函数样本调整的核心)
- [3. 主函数：speculative_generate](#3-主函数speculative_generate)
- [4. Logits Processor 设计](#4-logits-processor-设计)
- [5. KV Cache 管理](#5-kv-cache-管理)
- [6. 调试与可视化](#6-调试与可视化)
- [7. 扩展：EGAG 集成点](#7-扩展egag-集成点)
- [8. 小结](#8-小结)

---

## 1. 代码架构概览

### 1.1 文件结构

```
sampling/
├── base_decoding.py               # autoregressive_generate, beam_search_generate
└── speculative_decoding.py        # speculative_generate ★

utils/
├── logits_processor.py            # LogitsProcessor 及其子类
├── caching.py                     # prune_cache (KV cache 修剪)
└── printing.py                    # 调试可视化
```

### 1.2 函数签名

```python
# sampling/speculative_decoding.py:23-58
@torch.no_grad()
def speculative_generate(
    inputs: List[int],              # 输入序列
    drafter: Module,                # Drafter 模型
    target: Module,                 # Target 模型
    tokenizer = None,               # Tokenizer（调试用）
    gamma: int = 5,                 # 草稿数量
    use_egag: bool = False,          # 启用 EGAG
    gamma_min: int = 1,              # EGAG 最小 gamma
    gamma_max: int | None = None,    # EGAG 最大 gamma
    egag_ema_beta: float = 0.0,      # EGAG 平滑
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,  # 支持多个结束符
    pad_token_id: int = 0,
    use_cache: bool = False,
    skip_sample_adjustment: bool = False,  # 跳过样本调整（调试用）
    first_target: bool = True,      # 首次用 target 生成
    debug: bool = False,
) -> Tuple[List[int], float]:      # (生成的序列, 接受率)
```

### 1.3 状态变量

```python
# Line 66-71
drafter_cache, target_cache = None, None  # KV 缓存
drafts_accepted, drafts_speculated = .0, .0  # 统计接受率
current_position = prompt_len              # 当前生成位置
```

---

## 2. max_fn 函数：样本调整的核心

### 2.1 函数实现

```python
# sampling/speculative_decoding.py:10-19
def max_fn(x: torch.Tensor) -> torch.Tensor:
    """
    实现 max(0, x) 的归一化

    用于拒绝采样后的样本调整：
    p_adj(x) = max(0, p(x) - q(x)) / Σ max(0, p - q)

    Args:
        x: 输入张量，通常为 p - q
    Returns:
        归一化后的 max(0, x)
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))  # max(0, x)
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)  # Σ max(0, x)
    return x_max / x_max_sum  # 归一化
```

### 2.2 数学对应

| 数学 | 代码 |
|------|------|
| $\max(0, p(x) - q(x))$ | `torch.where(x > 0, x, torch.zeros_like(x))` |
| $\sum_x \max(0, p(x) - q(x))$ | `torch.sum(x_max, dim=-1, keepdim=True)` |
| $\frac{\max(0, p-q)}{\sum \max(0, p-q)}$ | `x_max / x_max_sum` |

### 2.3 使用位置

```python
# Line 168: 当草稿被拒绝时
p_p = max_fn(p[..., n, :] - q[0, n, :])
```

---

## 3. 主函数：speculative_generate

### 3.1 初始化阶段

```python
# Lines 73-82
vocabulary_size = target.config.vocab_size

# 准备输入张量
prompt_len = len(inputs)
max_seq_length = target.config.max_position_embeddings  # 或 max_context_length
total_len = min(max_seq_length, prompt_len + max_gen_len)
input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)
input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=target.device)

current_position = prompt_len
```

**关键点**：
- `max_seq_length`：模型最大序列长度限制
- `total_len`：不超过限制的总长度
- `input_ids`：预分配的张量，填充 `pad_token_id`

### 3.2 首次 Target Pass（可选）

```python
# Lines 84-103
if first_target:
    # 用 target 生成第一个 token
    Mp = target(
        input_ids=input_ids[..., :current_position],
        past_key_values=target_cache,
        use_cache=use_cache,
    )
    target_cache = Mp.past_key_values  # 更新缓存
    p_p = logits_processor(Mp.logits[..., -1, :])
    t = logits_processor.sample(p_p)
    input_ids[0, current_position] = t
    current_position += 1

    # 检查结束符
    if torch.isin(t, stop_tokens):
        return input_ids[0, prompt_len:current_position].tolist(), 0
```

**作用**：
- 预填充 KV Cache
- 确保 drafter 有一个良好的起始点

### 3.3 主循环

```python
# Line 105
while current_position < total_len:
```

#### 3.3.1 动态 Gamma 调整

```python
# Line 106
corrected_gamma = min(gamma, total_len - current_position - 1)
```

**原因**：接近序列末尾时，避免生成超过总长度的草稿。

#### 3.3.2 草稿生成 (Draft Generation)

```python
# Lines 107-126
q = torch.zeros((1, corrected_gamma, vocabulary_size), device=target.device)

input_ids = input_ids.to(drafter.device)

for k in range(corrected_gamma):
    Mq = drafter(
        input_ids=input_ids[..., :current_position + k],
        past_key_values=drafter_cache,
        use_cache=use_cache,
    )
    drafter_cache = Mq.past_key_values

    draft_logits = Mq.logits[..., -1, :]  # [1, vocab_size]
    draft_probs = logits_processor(draft_logits)  # 概率分布
    q[0, k] = draft_probs.to(target.device)  # 保存概率
    xi = logits_processor.sample(draft_probs)  # 采样
    input_ids[0, current_position + k] = xi  # 填入草稿

drafts_speculated += corrected_gamma
input_ids = input_ids.to(target.device)
```

**数据流**：

```
Step k=0: drafter([x₁, ..., x_t]) → q₀, x̂_{t+1}
Step k=1: drafter([x₁, ..., x_t, x̂_{t+1}]) → q₁, x̂_{t+2}
...
Step k=γ-1: drafter([x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+γ-1}]) → q_{γ-1}, x̂_{t+γ}
```

**关键变量**：
- `q`：drafter 的概率分布，shape `[1, γ, vocab_size]`
- `drafter_cache`：drafter 的 KV cache，每次更新
- `input_ids`：填充草稿后的序列

#### 3.3.3 并行验证 (Parallel Verification)

```python
# Lines 129-136
Mp = target(
    input_ids=input_ids[..., :current_position + corrected_gamma],
    past_key_values=target_cache,
    use_cache=use_cache,
)
target_cache = Mp.past_key_values
draft_logits = Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :]
p = logits_processor(draft_logits)  # [1, γ, vocab_size]
```

**关键洞察**：

```
输入: [x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+γ}]
输出: Mp.logits[..., t-1:t+γ-1, :]  # t, t+1, ..., t+γ-1 位置的 logits
     = [logits_t, logits_{t+1}, ..., logits_{t+γ-1}]

     logits_t 对应位置 t 的预测（基于 x₁, ..., x_t）
     logits_{t+1} 对应位置 t+1 的预测（基于 x₁, ..., x_t, x̂_{t+1}）
     ...
     logits_{t+γ-1} 对应位置 t+γ-1 的预测（基于 x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+γ-1}）
```

**一次前向传播得到 γ 个位置的概率分布！**

#### 3.3.4 拒绝采样 (Rejection Sampling)

```python
# Lines 139-147
r = torch.rand(corrected_gamma, device=target.device)  # 随机数
fractions = p / q  # [1, γ, vocab_size]
n = corrected_gamma

for i in range(corrected_gamma):
    # 检查第 i 个草稿是否被接受
    if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
        n = i  # 第 i 个草稿被拒绝
        break

drafts_accepted += n
```

**逻辑解析**：

```
对于第 i 个草稿 x̂_{t+i}:
  接受条件: r[i] ≤ fractions[0, i, x̂_{t+i}] = p(x̂_{t+i}) / q(x̂_{t+i})

  如果 r[i] > p/q: 拒绝，设置 n = i，停止检查后续草稿
  否则: 接受，继续检查下一个
```

**为什么停止？**：一旦某个草稿被拒绝，后续草稿都基于被拒绝的草稿，因此也无效。

#### 3.3.5 样本调整 (Sample Adjustment)

```python
# Lines 158-171
if n == corrected_gamma:
    # 全部草稿被接受：从下一个位置采样
    p_p = Mp.logits[..., current_position + corrected_gamma - 1, :]
    p_p = logits_processor(p_p)
else:
    # 第 n 个草稿被拒绝：修剪 KV cache，从调整后的分布采样
    if use_cache:
        drafter_cache = prune_cache(drafter_cache, corrected_gamma - n)
        target_cache = prune_cache(target_cache, corrected_gamma - n + 1)

    if not skip_sample_adjustment:
        p_p = max_fn(p[..., n, :] - q[0, n, :])  # ★ 样本调整
    else:
        p_p = p[..., n, :]  # 跳过调整（调试用）

x = logits_processor.sample(p_p)
```

**两种情况**：

| 情况 | 采样来源 | 说明 |
|------|----------|------|
| 全部接受 (n=γ) | `Mp.logits[..., t+γ-1, :]` | target 的下一个位置 |
| 部分拒绝 (n<γ) | `max_fn(p[..., n, :] - q[0, n, :])` | 调整后的分布 |

#### 3.3.6 更新和清理

```python
# Lines 173-182
input_ids[0, current_position + n:current_position + corrected_gamma] = pad_token_id
input_ids[0, current_position + n] = x  # 放入采样的 token

current_position += n + 1  # 前进 n 个接受的草稿 + 1 个采样的 token

if torch.isin(x, stop_tokens):
    return input_ids[0, prompt_len:current_position].tolist(), drafts_accepted / drafts_speculated
```

---

## 4. Logits Processor 设计

### 4.1 抽象基类

```python
# utils/logits_processor.py:7-23
class LogitsProcessor(abc.ABC):
    """Logits processors for sampling."""

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, logits: Tensor) -> Tensor:
        proc = self._process(logits)  # 子类实现
        return F.softmax(proc / self.temperature, dim=-1)

    @abc.abstractmethod
    def _process(self, logits: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def sample(self, probs: Tensor) -> Tensor:
        pass
```

**设计模式**：两阶段处理
1. `_process()`: 修改 logits（masking、 filtering）
2. `sample()`: 从概率分布采样

### 4.2 Greedy Processor

```python
# utils/logits_processor.py:26-36
class GreedyProcessor(LogitsProcessor):
    """Greedy: Most probable token."""

    def _process(self, logits: Tensor) -> Tensor:
        return logits  # 不修改

    def sample(self, probs: Tensor) -> Tensor:
        return torch.argmax(probs, dim=-1).unsqueeze(-1)  # argmax
```

**用途**：确定性生成，用于调试和测试。

### 4.3 Top-K Processor

```python
# utils/logits_processor.py:52-63
class TopKProcessor(MultinomialProcessor):
    """Top-k: Top-k sampling."""

    def _process(self, logits: Tensor) -> Tensor:
        top_k = min(self.top_k, logits.size(-1))
        # 找到第 k 大的值
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = -1e20  # mask 掉
        return logits
```

**效果**：只保留 top-k 个 token 的概率，其他设为 -∞。

### 4.4 Nucleus Processor (Top-p)

```python
# utils/logits_processor.py:66-81
class NucleusProcessor(MultinomialProcessor):
    """Nucleus: Top-p sampling."""

    def _process(self, logits: Tensor) -> Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 找到累积概率超过 p 的位置
        sorted_indices_to_remove = cumulative_probs > self.top_p
        # 平滑处理：保留第一个超过阈值的 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_logits[sorted_indices_to_remove] = -1e20
        # 恢复原始顺序
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))
        return logits
```

---

## 5. KV Cache 管理

### 5.1 为什么需要修剪 Cache？

当草稿被拒绝时：

```
接受前: [x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+n}, x̂_{t+n+1}(rejected), ...]
接受后: [x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+n}, x_new]
```

被拒绝的草稿对应的 KV cache 条目必须移除，否则后续计算会出错。

### 5.2 prune_cache 函数

```python
# utils/caching.py:6-24
def prune_cache(cache, num_tokens_to_discard: int):
    """修剪 KV cache，移除末尾的 token"""
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return prune_tuple_cache(cache, num_tokens_to_discard)
    elif isinstance(cache, DynamicCache):
        return prune_dynamic_cache(cache, num_tokens_to_discard)
```

### 5.3 Tuple Cache 实现

```python
# utils/caching.py:27-55
def prune_tuple_cache(cache: Tuple[Tuple[Tensor, Tensor]], num_tokens_to_discard: int):
    """
    Tuple cache 格式: (n_layers,) × (2 or 4) × [batch, heads, seq_len, head_dim]
    """
    if cache is None:
        return None

    new_cache = []
    for layer_cache in cache:
        if layer_cache is None:
            new_cache.append(None)
            continue

        layer = []
        for i in range(len(layer_cache)):
            tensor = layer_cache[i]
            # 移除最后 num_tokens_to_discard 个位置
            new_tensor = tensor[:, :, :-num_tokens_to_discard, :]
            layer.append(new_tensor)
        new_cache.append(tuple(layer))

    return tuple(new_cache)
```

**示例**：

```
原始 cache shape: [batch, num_heads, seq_len=100, head_dim]
prune_cache(cache, 5)
→ [batch, num_heads, seq_len=95, head_dim]
```

### 5.4 Dynamic Cache 实现

```python
# utils/caching.py:58-77
def prune_dynamic_cache(cache: DynamicCache, num_tokens_to_discard: int):
    """HuggingFace 新版 DynamicCache"""
    if cache is None:
        return None

    for layer in range(len(cache)):
        cache.key_cache[layer] = cache.key_cache[layer][:, :, :-num_tokens_to_discard, :]
        cache.value_cache[layer] = cache.value_cache[layer][:, :, :-num_tokens_to_discard, :]
    cache._seen_tokens -= num_tokens_to_discard  # 更新计数器

    return cache
```

### 5.5 调用位置

```python
# speculative_decoding.py:164-165
if use_cache:
    drafter_cache = prune_cache(drafter_cache, corrected_gamma - n)
    target_cache = prune_cache(target_cache, corrected_gamma - n + 1)  # +1 因为 target 多计算了一个位置
```

---

## 6. 调试与可视化

### 6.1 调试输出

```python
# speculative_decoding.py:179-180
if debug:
    printing.speculative_step(tokenizer, generated, input_ids, n, prompt_len, current_position, corrected_gamma)
```

### 6.2 输出示例

```
========== Speculative Step ==========
Position: 10, Gamma: 4, Accepted: 3

Drafts (Drafter):
  [11]: "Paris" (q=0.85) ✓
  [12]: "," (q=0.72) ✓
  [13]: "France" (q=0.68) ✗ (rejected: r=0.75 > p/q=0.62)

Final Sample:
  [12]: "and" (adjusted distribution)

Generated: "Paris, and..."
=======================================
```

---

## 7. 扩展：EGAG 集成点

### 7.1 EGAG（熵自适应 $\gamma$）

**集成位置**：主循环内，在草稿生成前。

- 使用 drafter 的当前 logits 计算熵并更新 $\gamma$。
- 需要保证 $\gamma$ 的上下界与序列末尾的 `corrected_gamma` 同步。
- 建议记录每步 $H_{norm}$ 与 $\gamma$ 以便实验分析。

**涉及模块**：
- `speculative_generate()`
- `LogitsProcessor`（可复用 softmax 或采样逻辑）

**新增参数**：
- `use_egag`
- `gamma_min`, `gamma_max`
- `egag_ema_beta`

## 8. 小结

### 关键实现要点

1. **并行验证**：`target([x₁, ..., x_t, x̂_{t+1}, ..., x̂_{t+γ}])` 一次得到 γ 个位置
2. **拒绝采样**：`r > p/q` 判断接受/拒绝
3. **样本调整**：`max_fn(p - q)` 保证分布正确性
4. **KV Cache 管理**：`prune_cache()` 移除被拒绝的草稿
5. **Logits Processor**：统一的采样策略接口

### 代码引用速查

| 功能 | 文件 | 行号 |
|------|------|------|
| 主函数 | `sampling/speculative_decoding.py` | 23-260 |
| 样本调整 | `sampling/speculative_decoding.py` | 10-19 |
| 草稿生成 | `sampling/speculative_decoding.py` | 112-126 |
| 并行验证 | `sampling/speculative_decoding.py` | 129-137 |
| 拒绝采样 | `sampling/speculative_decoding.py` | 139-147 |
| Logits Processor | `utils/logits_processor.py` | 7-103 |
| KV Cache 修剪 | `utils/caching.py` | 6-77 |

### 下一步

阅读 [04. NASD 理论](./04-nasd-theory.md) 了解 Ngram Assisted Speculative Decoding。
