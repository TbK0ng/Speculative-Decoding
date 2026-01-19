# 05. NASD 实现

## 目录

- [1. 代码架构概览](#1-代码架构概览)
- [2. INgramStorage 抽象接口](#2-ingramstorage-抽象接口)
- [3. OneLevelNGramStorage 实现](#3-onelevelngramstorage-实现)
- [4. NGramStorage 实现](#4-ngramstorage-实现)
- [5. NASD 主函数](#5-nasd-主函数)
- [6. 关键参数说明](#6-关键参数说明)
- [7. 小结](#7-小结)

---

## 1. 代码架构概览

### 1.1 文件结构

```
ngram_assisted/
├── __init__.py                      # 导出接口
├── ngram_assisted.py                # ngram_assisted_speculative_generate ★
└── ngram_storage.py                 # Ngram 存储实现 ★
    ├── INgramStorage                 # 抽象接口
    ├── OneLevelNGramStorage         # 单级实现
    └── NGramStorage                 # 多级实现
```

### 1.2 函数签名

```python
# ngram_assisted/ngram_assisted.py:11-26
@torch.no_grad()
def ngram_assisted_speculative_generate(
    inputs: List[int],                # 输入序列
    ngramstorage: INgramStorage,       # Ngram 存储（drafter）
    target: Module,                   # Target 模型
    tokenizer = None,
    gamma: int = 5,                   # 草稿数量
    filler_top_k: int = 3,            # Top-K Filler
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    first_target: bool = True,
    stop_if_unknown: bool = False,    # 未知上下文时停止
    debug: bool = False,
) -> Tuple[List[int], float]:         # (生成的序列, 接受率)
```

### 1.3 与标准 Speculative Decoding 的区别

| 特性 | 标准版本 | NASD |
|------|----------|------|
| Drafter | `drafter: Module` | `ngramstorage: INgramStorage` |
| 草稿生成 | 模型前向传播 | Ngram 查找 |
| 验证方式 | 概率性 (p/q ratio) | 确定性 (top-1 match) |
| 样本调整 | `max_fn(p - q)` | 直接使用 `p` |

---

## 2. INgramStorage 抽象接口

### 2.1 接口定义

```python
# ngram_assisted/ngram_storage.py:5-69
class INgramStorage(abc.ABC):
    """
    Ngram 存储的抽象接口

    功能:
    - 存储 N-gram 统计
    - 基于上下文预测下一个 token
    """

    def __init__(self, n: int, vocab_size: int):
        assert n > 1, "n should be greater than 1"
        self.n = n  # 最大 N-gram 阶数
        self.vocab_size = vocab_size
```

### 2.2 核心方法

#### 2.2.1 next_token：预测下一个 token

```python
# ngram_assisted/ngram_storage.py:17-27
@abc.abstractmethod
def next_token(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于输入序列预测下一个 token

    Args:
        input_ids: [batch_size, seq_len]

    Returns:
        predicted_tokens: [batch_size, 1] - 预测的 token
        known: [batch_size, 1] - 是否已知（True=见过此上下文，False=随机）
    """
    pass
```

#### 2.2.2 update：更新统计

```python
# ngram_assisted/ngram_storage.py:44-52
@abc.abstractmethod
def update(self, input_ids: torch.Tensor, next_tokens: torch.Tensor):
    """
    用新的观察更新 N-gram 统计

    Args:
        input_ids: [batch_size, seq_len] - 上下文
        next_tokens: [batch_size, k] - k 个后续 token
    """
    pass
```

#### 2.2.3 initialize：批量初始化

```python
# ngram_assisted/ngram_storage.py:55-62
@abc.abstractmethod
def initialize(self, input_ids: torch.Tensor):
    """
    从输入序列中提取所有 N-gram

    Args:
        input_ids: [batch_size, seq_len]
    """
    pass
```

---

## 3. OneLevelNGramStorage 实现

### 3.1 数据结构

```python
# ngram_assisted/ngram_storage.py:78-81
class OneLevelNGramStorage(INgramStorage):
    def __init__(self, n: int, vocab_size: int):
        super().__init__(n, vocab_size)
        self.counts = {}   # gram -> {token -> count}
        self.ngrams = {}   # gram -> token with highest count
```

**示例**：

```python
# 输入: "The capital of France is Paris"
# n=3 (trigram)

counts = {
    ("The", "capital"): {"of": 1},
    ("capital", "of"): {"France": 1},
    ("of", "France"): {"is": 1},
    ("France", "is"): {"Paris": 1},
}

ngrams = {
    ("The", "capital"): "of",
    ("capital", "of"): "France",
    ("of", "France"): "is",
    ("France", "is"): "Paris",
}
```

### 3.2 next_token 实现

```python
# ngram_assisted/ngram_storage.py:83-96
def next_token(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 随机初始化（未知上下文的默认值）
    out = torch.randint(self.vocab_size, size=(input_ids.shape[0],))
    known = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for i, seq in enumerate(input_ids):
        if seq.shape[0] < self.n - 1:
            continue  # 序列太短

        # 提取最后 n-1 个 token 作为上下文
        gram = tuple(seq[-(self.n - 1):].tolist())

        if gram in self.ngrams:
            out[i] = self.ngrams[gram]  # 返回最高频的 token
            known[i] = True  # 标记为已知

    return out, known
```

**示例**：

```python
# n=3, context="The capital of"
# gram = ("The", "capital")  # 只需要 n-1=2 个 token

if ("The", "capital") in ngrams:
    return "of", True  # 预测 "of"
else:
    return random_token(), False
```

### 3.3 update 实现

```python
# ngram_assisted/ngram_storage.py:108-128
def update(self, input_ids: torch.Tensor, next_tokens: torch.Tensor):
    for i, seq in enumerate(input_ids):
        if seq.shape[0] < self.n:
            continue

        gram = tuple(seq[-(self.n - 1):].tolist())
        tokens = next_tokens[i].tolist()

        # 初始化
        if gram not in self.counts:
            self.counts[gram] = {}
        if gram not in self.ngrams:
            self.ngrams[gram] = tokens[0]

        # 更新计数
        for token in tokens:
            if token not in self.counts[gram]:
                self.counts[gram][token] = 1
            else:
                self.counts[gram][token] += 1

            # 更新最高频 token
            if self.counts[gram][token] > self.counts[gram][self.ngrams[gram]]:
                self.ngrams[gram] = token
```

**关键**：每次更新都会检查是否需要更新最高频 token。

---

## 4. NGramStorage 实现

### 4.1 与 OneLevelNGramStorage 的区别

| 特性 | OneLevelNGramStorage | NGramStorage |
|------|---------------------|--------------|
| 存储阶数 | 单一 n 阶 | 多级 (2 到 n 阶) |
| 回退策略 | 不支持 | 支持 |
| 适用场景 | 上下文固定长度 | 上下文长度可变 |

### 4.2 数据结构

```python
# ngram_assisted/ngram_storage.py:159-162
class NGramStorage(INgramStorage):
    def __init__(self, n: int, vocab_size: int):
        super().__init__(n, vocab_size)
        self.counts = {}   # j -> {gram -> {token -> count}}
        self.ngrams = {}   # j -> {gram -> token with highest count}
```

**结构**：

```python
# n=3 时存储 2-gram 和 3-gram

counts = {
    2: {  # bigram
        ("The",): {"capital": 5, "city": 2, ...},
        ("capital",): {"of": 3, "is": 1, ...},
        ...
    },
    3: {  # trigram
        ("The", "capital"): {"of": 2, "is": 1, ...},
        ("capital", "of"): {"France": 1, "Germany": 1, ...},
        ...
    }
}
```

### 4.3 回退策略实现

```python
# ngram_assisted/ngram_storage.py:164-179
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

**示例**：

```python
# n=4, context="The capital of France is"

尝试顺序:
  j=3: ("The", "capital", "of") → 未见
  j=2: ("capital", "of") → 见过，预测 "France" ✓
  j=1: ("of",) → 不再尝试
```

### 4.4 initialize 实现

```python
# ngram_assisted/ngram_storage.py:223-245
def initialize(self, input_ids: torch.Tensor):
    for seq in input_ids:
        for i in range(seq.shape[0]):
            # 提取所有 j-gram (j in [2, n])
            for j in range(min(self.n - 1, i), 1, -1):
                gram = tuple(seq[i - j:i].tolist())
                token = seq[i].item()

                # 初始化
                if j not in self.counts:
                    self.counts[j] = {}
                if j not in self.ngrams:
                    self.ngrams[j] = {}

                if gram not in self.counts[j]:
                    self.counts[j][gram] = {}
                if gram not in self.ngrams[j]:
                    self.ngrams[j][gram] = token

                # 更新计数
                if token not in self.counts[j][gram]:
                    self.counts[j][gram][token] = 1
                else:
                    self.counts[j][gram][token] += 1
                    if self.counts[j][gram][token] > self.counts[j][gram][self.ngrams[j][gram]]:
                        self.ngrams[j][gram] = token
```

**从 prompt 中提取所有 N-gram**：

```python
# Prompt: "The capital of France is"
# n=3

提取:
  i=2, j=2: ("The", "capital") -> "of"
  i=3, j=2: ("capital", "of") -> "France"
  i=3, j=3: ("The", "capital", "of") -> "France"
  i=4, j=2: ("of", "France") -> "is"
  i=4, j=3: ("capital", "of", "France") -> "is"
  ...
```

---

## 5. NASD 主函数

### 5.1 初始化

```python
# ngram_assisted/ngram_assisted.py:62-72
prompt_len = len(inputs)
total_len = min(target.config.max_position_embeddings, prompt_len + max_gen_len)
input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)
input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=target.device)

current_position = prompt_len

# 初始化 Ngram 模型
ngramstorage.initialize(input_ids[..., :prompt_len])
```

### 5.2 首次 Target Pass（可选）

```python
# ngram_assisted/ngram_assisted.py:73-87
if first_target:
    Mp = target(input_ids[..., :current_position], ...)
    target_cache = Mp.past_key_values
    p_p = logits_processor(Mp.logits[..., -1, :])
    t = logits_processor.sample(p_p)
    input_ids[0, prompt_len] = t
    current_position += 1

    # 更新 Ngram 模型
    ngramstorage.update(input_ids[..., :prompt_len], t)
```

**关键**：第一个 token 用 target 生成，然后更新 ngram。

### 5.3 草稿生成 (Ngram Drafter)

```python
# ngram_assisted/ngram_assisted.py:89-101
while current_position < total_len:
    corrected_gamma = min(gamma, total_len - current_position - 1)
    copied_inputs_ids = input_ids.clone()

    # 生成 γ 个草稿
    for k in range(corrected_gamma):
        # Ngram 预测
        copied_inputs_ids[0, current_position + k], known = ngramstorage.next_token(
            copied_inputs_ids[..., :current_position + k]
        )

        # 未知上下文处理
        if not known[0] and stop_if_unknown:
            corrected_gamma = k
            break

    drafts_speculated += corrected_gamma
```

**关键**：`next_token()` 返回 `(predicted_token, known_flag)`

### 5.4 并行验证

```python
# ngram_assisted/ngram_assisted.py:104-112
Mp = target(
    input_ids=copied_inputs_ids[..., :current_position + corrected_gamma],
    past_key_values=target_cache,
    use_cache=use_cache,
)
target_cache = Mp.past_key_values
draft_logits = Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :]
p = logits_processor(draft_logits)  # [1, γ, vocab_size]
```

**与标准版本相同**：一次前向传播验证所有草稿。

### 5.5 确定性验证 ⭐

```python
# ngram_assisted/ngram_assisted.py:114-120
n = corrected_gamma
for i in range(corrected_gamma):
    sample = logits_processor.sample(p[0, i, :])  # Target 的 top-1

    # 检查 ngram 预测是否与 target 的 top-1 一致
    if copied_inputs_ids[0, current_position + i] != sample:
        n = i  # 第 i 个不匹配
        break
```

**逻辑**：

```
对于每个草稿 x̂_{t+i}:
  target_top1 = argmax p_{t+i}

  if x̂_{t+i} == target_top1:
    接受 ✓
  else:
    拒绝 ✗，停止检查后续草稿
```

### 5.6 样本调整（无需）

```python
# ngram_assisted/ngram_assisted.py:131-141
if n == corrected_gamma:
    # 全部接受：从下一个位置采样
    p_p = Mp.logits[..., current_position + corrected_gamma - 1, :]
    p_p = logits_processor(p_p)
else:
    # 部分拒绝：修剪 cache
    if use_cache:
        target_cache = prune_cache(target_cache, corrected_gamma - n + 1)

    # 直接使用 target 的分布（无需样本调整！）
    p_p = p[..., n, :]

x = logits_processor.sample(p_p)
```

**关键差异**：不需要 `max_fn(p - q)`，直接使用 `p`。

### 5.7 Ngram 更新

```python
# ngram_assisted/ngram_assisted.py:148-155
# 更新接受的草稿
for i in range(n):
    ngramstorage.update(input_ids[..., :current_position + i], input_ids[..., current_position + i].unsqueeze(0))

    # Top-K Filler：用 target 的 top-K 更新
    if filler_top_k > 1:
        ngramstorage.update(input_ids[..., :current_position + i], p[..., i, :].topk(filler_top_k).indices)

# 更新最终采样的 token
ngramstorage.update(input_ids[..., :current_position + n], x)
if filler_top_k > 1:
    ngramstorage.update(input_ids[..., :current_position + n], p_p.topk(filler_top_k).indices)
```

**Top-K Filler 的作用**：

```python
# Target 预测 p = [Paris: 0.6, Berlin: 0.2, London: 0.1, ...]
# top_k=3: [Paris, Berlin, London]

ngramstorage.update(context, [Paris, Berlin, London])
# 下次遇到相同上下文时，有机会预测 Berlin 或 London
```

---

## 6. 关键参数说明

### 6.1 n: 最大 N-gram 阶数

```python
ngramstorage = NGramStorage(n=3, vocab_size=...)  # Trigram
```

| n 值 | 上下文长度 | 适用场景 |
|------|-----------|----------|
| 2 | 1 个 token | 简单模式 |
| 3 | 2 个 token | 中等模式（推荐） |
| 4-5 | 3-4 个 token | 复杂模式 |

**权衡**：
- n 太大：稀疏性高，很多上下文未见
- n 太小：捕捉不到复杂模式

### 6.2 filler_top_k: Top-K Filler

```python
filler_top_k = 1   # NAPD 风格（只更新接受的）
filler_top_k = 3   # NASD 风格（更新 top-3）
filler_top_k = 10  # 更高的适应性
```

**效果对比**：

| filler_top_k | 接受率 | 适应性 | 内存占用 |
|-------------|-------|--------|---------|
| 1 | 基准 | 慢 | 小 |
| 3 | +10-20% | 中 | 中 |
| 10 | +20-30% | 快 | 大 |

### 6.3 stop_if_unknown: 未知上下文处理

```python
stop_if_unknown = True   # 遇到未知上下文时停止生成草稿
stop_if_unknown = False  # 继续生成（用随机 token）
```

**建议**：
- `True`：更保守，避免无效草稿
- `False`：更激进，可能在未知上下文中学习

---

## 7. 小结

### 关键实现要点

1. **Ngram 存储**：两种实现
   - `OneLevelNGramStorage`：简单，单阶
   - `NGramStorage`：支持回退，多阶

2. **确定性验证**：`x̂ == argmax(p)` 判断接受/拒绝

3. **无需样本调整**：直接使用 target 的分布

4. **Top-K Filler**：用 target 的 top-K 更新 ngram，提高适应性

5. **在线学习**：生成过程中持续更新 ngram 统计

### 代码引用速查

| 功能 | 文件 | 行号 |
|------|------|------|
| NASD 主函数 | `ngram_assisted/ngram_assisted.py` | 11-164 |
| INgramStorage 接口 | `ngram_assisted/ngram_storage.py` | 5-69 |
| OneLevelNGramStorage | `ngram_assisted/ngram_storage.py` | 73-150 |
| NGramStorage (回退) | `ngram_assisted/ngram_storage.py` | 154-249 |
| Ngram 初始化 | `ngram_assisted/ngram_assisted.py` | 71 |
| 草稿生成 | `ngram_assisted/ngram_assisted.py` | 95-101 |
| 确定性验证 | `ngram_assisted/ngram_assisted.py` | 115-120 |
| Ngram 更新 | `ngram_assisted/ngram_assisted.py` | 149-155 |

### 与标准 Speculative Decoding 代码对比

| 步骤 | 标准 Speculative | NASD |
|------|------------------|------|
| Drafter | `drafter(input_ids)` | `ngramstorage.next_token(input_ids)` |
| 验证 | `r > p/q` | `x̂ != argmax(p)` |
| 样本调整 | `max_fn(p - q)` | `p` (无需调整) |
| 更新 | 不需要 | `ngramstorage.update()` |

### 下一步

阅读 [06. 技巧与优化](./06-tricks-and-optimizations.md) 了解实现中的技巧和优化。
