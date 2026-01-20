# API 参考手册

## 目录

- [1. 生成函数](#1-生成函数)
- [2. Ngram Storage](#2-ngram-storage)
- [3. Logits Processors](#3-logits-processors)
- [4. 工具函数](#4-工具函数)

---

## 1. 生成函数

### 1.1 autoregressive_generate

**文件**: `sampling/base_decoding.py:10-65`

**函数签名**:

```python
@torch.no_grad()
def autoregressive_generate(
    inputs: List[int],
    model: Module,
    max_gen_len: int = 40,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    debug: bool = False,
) -> List[int]
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `inputs` | `List[int]` | 必填 | 输入序列 |
| `model` | `Module` | 必填 | 生成模型 |
| `max_gen_len` | `int` | `40` | 最大生成长度 |
| `logits_processor` | `LogitsProcessor` | `GreedyProcessor()` | Logits 处理器 |
| `eos_tokens_id` | `int or List[int]` | `1` | 结束 token ID |
| `pad_token_id` | `int` | `0` | 填充 token ID |
| `use_cache` | `bool` | `False` | 是否使用 KV Cache |
| `debug` | `bool` | `False` | 是否输出调试信息 |

**返回**:

- `List[int]`: 生成的 token 序列

**示例**:

```python
from sampling import autoregressive_generate
from utils.logits_processor import GreedyProcessor

inputs = [1, 2, 3, 4]  # "Hello world"
output = autoregressive_generate(
    inputs,
    model=target,
    max_gen_len=50,
    logits_processor=GreedyProcessor(),
)
```

---

### 1.2 speculative_generate

**文件**: `sampling/speculative_decoding.py:23-231`

**函数签名**:

```python
@torch.no_grad()
def speculative_generate(
    inputs: List[int],
    drafter: Module,
    target: Module,
    tokenizer = None,
    gamma: int = 5,
    use_egag: bool = False,
    gamma_min: int = 1,
    gamma_max: int | None = None,
    egag_ema_beta: float = 0.0,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    skip_sample_adjustment: bool = False,
    first_target: bool = True,
    debug: bool = False,
) -> Tuple[List[int], float]
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `inputs` | `List[int]` | 必填 | 输入序列 |
| `drafter` | `Module` | 必填 | Drafter 模型 |
| `target` | `Module` | 必填 | Target 模型 |
| `tokenizer` | `Tokenizer` | `None` | Tokenizer（调试用） |
| `gamma` | `int` | `5` | 草稿数量 |
| `use_egag` | `bool` | `False` | 启用 EGAG（熵自适应 gamma） |
| `gamma_min` | `int` | `1` | EGAG 最小 gamma |
| `gamma_max` | `int or None` | `None` | EGAG 最大 gamma（默认使用 `gamma`） |
| `egag_ema_beta` | `float` | `0.0` | EGAG EMA 平滑系数 |
| `logits_processor` | `LogitsProcessor` | `GreedyProcessor()` | Logits 处理器 |
| `max_gen_len` | `int` | `40` | 最大生成长度 |
| `eos_tokens_id` | `int or List[int]` | `1` | 结束 token ID（支持多个） |
| `pad_token_id` | `int` | `0` | 填充 token ID |
| `use_cache` | `bool` | `False` | 是否使用 KV Cache |
| `skip_sample_adjustment` | `bool` | `False` | 是否跳过样本调整 |
| `first_target` | `bool` | `True` | 首次是否用 Target 生成 |
| `debug` | `bool` | `False` | 是否输出调试信息 |

**返回**:

- `List[int]`: 生成的 token 序列
- `float`: 接受率（接受的草稿数 / 总草稿数）

**示例**:

```python
from sampling import speculative_generate
from utils.logits_processor import NucleusProcessor

inputs = [1, 2, 3, 4]
output, acceptance_rate = speculative_generate(
    inputs,
    drafter=drafter,
    target=target,
    gamma=4,
    logits_processor=NucleusProcessor(temperature=0.8, top_p=0.9),
)

print(f"Output: {output}")
print(f"Acceptance rate: {acceptance_rate:.2%}")
```

---

### 1.3 ngram_assisted_speculative_generate

**文件**: `ngram_assisted/ngram_assisted.py:11-164`

**函数签名**:

```python
@torch.no_grad()
def ngram_assisted_speculative_generate(
    inputs: List[int],
    ngramstorage: INgramStorage,
    target: Module,
    tokenizer = None,
    gamma: int = 5,
    filler_top_k: int = 3,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    first_target: bool = True,
    stop_if_unknown: bool = False,
    debug: bool = False,
) -> Tuple[List[int], float]
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `inputs` | `List[int]` | 必填 | 输入序列 |
| `ngramstorage` | `INgramStorage` | 必填 | Ngram 存储（作为 Drafter） |
| `target` | `Module` | 必填 | Target 模型 |
| `tokenizer` | `Tokenizer` | `None` | Tokenizer（调试用） |
| `gamma` | `int` | `5` | 草稿数量 |
| `filler_top_k` | `int` | `3` | Top-K Filler 值 |
| `logits_processor` | `LogitsProcessor` | `GreedyProcessor()` | Logits 处理器 |
| `max_gen_len` | `int` | `40` | 最大生成长度 |
| `eos_tokens_id` | `int or List[int]` | `1` | 结束 token ID |
| `pad_token_id` | `int` | `0` | 填充 token ID |
| `use_cache` | `bool` | `False` | 是否使用 KV Cache |
| `first_target` | `bool` | `True` | 首次是否用 Target 生成 |
| `stop_if_unknown` | `bool` | `False` | 遇到未知上下文时是否停止生成草稿 |
| `debug` | `bool` | `False` | 是否输出调试信息 |

**返回**:

- `List[int]`: 生成的 token 序列
- `float`: 接受率

**示例**:

```python
from ngram_assisted import NGramStorage, ngram_assisted_speculative_generate

inputs = [1, 2, 3, 4]
ngram = NGramStorage(n=3, vocab_size=target.config.vocab_size)

output, acceptance_rate = ngram_assisted_speculative_generate(
    inputs,
    ngramstorage=ngram,
    target=target,
    gamma=4,
    filler_top_k=3,
)
```

---

### 1.4 beam_search_generate

**文件**: `sampling/base_decoding.py:69-187`

**函数签名**:

```python
@torch.no_grad()
def beam_search_generate(
    inputs: List[int],
    model: Module,
    max_gen_len: int = 40,
    num_beams: int = 4,
    top_k: int = 3,
    min_length: float = 5.0,
    alpha: float = 1.2,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    debug: bool = False,
    tokenizer=None,
) -> List[int]
```

**参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `inputs` | `List[int]` | 必填 | 输入序列 |
| `model` | `Module` | 必填 | 生成模型 |
| `max_gen_len` | `int` | `40` | 最大生成长度 |
| `num_beams` | `int` | `4` | Beam 数量 |
| `top_k` | `int` | `3` | 每个分支考虑的 top-k 数量 |
| `min_length` | `float` | `5.0` | 长度惩罚的最小长度 |
| `alpha` | `float` | `1.2` | 长度惩罚的 alpha 参数 |
| `eos_tokens_id` | `int or List[int]` | `1` | 结束 token ID |
| `pad_token_id` | `int` | `0` | 填充 token ID |
| `debug` | `bool` | `False` | 是否输出调试信息 |
| `tokenizer` | `Tokenizer` | `None` | Tokenizer（调试用） |

**返回**:

- `List[int]`: 生成的 token 序列

**示例**:

```python
from sampling import beam_search_generate

inputs = [1, 2, 3, 4]
output = beam_search_generate(
    inputs,
    model=target,
    num_beams=5,
    top_k=3,
    alpha=1.2,
)
```

---

## 2. Ngram Storage

### 2.1 INgramStorage

**文件**: `ngram_assisted/ngram_storage.py:5-69`

**抽象接口**，定义 Ngram 存储的基本操作。

**方法**:

| 方法 | 说明 |
|------|------|
| `next_token(input_ids)` | 预测下一个 token，返回 `(token, known_flag)` |
| `has_gram(ngram)` | 检查 n-gram 是否见过 |
| `update(input_ids, next_tokens)` | 用新观察更新统计 |
| `initialize(input_ids)` | 从输入序列批量初始化 |
| `reset()` | 重置存储 |

---

### 2.2 NGramStorage

**文件**: `ngram_assisted/ngram_storage.py:154-249`

**完整实现**，支持多级 N-gram（2 到 n 阶）和回退策略。

**构造函数**:

```python
NGramStorage(n: int, vocab_size: int)
```

**参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `n` | `int` | 最大 N-gram 阶数（存储 2 到 n 阶） |
| `vocab_size` | `int` | 词表大小 |

**示例**:

```python
from ngram_assisted import NGramStorage

# 存储 2-gram 到 5-gram
ngram = NGramStorage(n=5, vocab_size=32000)

# 初始化
ngram.initialize(input_ids)

# 预测
token, known = ngram.next_token(context)

# 更新
ngram.update(context, next_token)
```

---

### 2.3 OneLevelNGramStorage

**文件**: `ngram_assisted/ngram_storage.py:73-150`

**简化实现**，只存储单一 n 阶 N-gram。

**构造函数**:

```python
OneLevelNGramStorage(n: int, vocab_size: int)
```

**参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `n` | `int` | N-gram 阶数 |
| `vocab_size` | `int` | 词表大小 |

**使用场景**：

- 上下文长度固定
- 内存受限
- 需要更快的查找速度

**示例**:

```python
from ngram_assisted import OneLevelNGramStorage

# 只存储 trigram
ngram = OneLevelNGramStorage(n=3, vocab_size=32000)
```

---

## 3. Logits Processors

### 3.1 LogitsProcessor（抽象基类）

**文件**: `utils/logits_processor.py:7-23`

**所有处理器的基础类**。

**构造函数**:

```python
LogitsProcessor(temperature: float = 1.0)
```

**方法**:

| 方法 | 说明 |
|------|------|
| `__call__(logits)` | 处理 logits，返回概率分布 |
| `_process(logits)` | 子类实现，修改 logits |
| `sample(probs)` | 从概率分布采样 |

---

### 3.2 GreedyProcessor

**文件**: `utils/logits_processor.py:26-36`

**贪婪采样**：总是选择概率最高的 token。

**构造函数**:

```python
GreedyProcessor(temperature: float = 1.0)
```

**示例**:

```python
from utils.logits_processor import GreedyProcessor

processor = GreedyProcessor(temperature=1.0)
probs = processor(logits)  # softmax(logits / temperature)
token = processor.sample(probs)  # argmax(probs)
```

---

### 3.3 MultinomialProcessor

**文件**: `utils/logits_processor.py:39-49`

**多项式采样**：从完整概率分布随机采样。

**构造函数**:

```python
MultinomialProcessor(temperature: float)
```

**示例**:

```python
from utils.logits_processor import MultinomialProcessor

processor = MultinomialProcessor(temperature=0.8)
token = processor.sample(probs)  # multinomial(probs)
```

---

### 3.4 TopKProcessor

**文件**: `utils/logits_processor.py:52-63`

**Top-K 采样**：从概率最高的 K 个 token 中采样。

**构造函数**:

```python
TopKProcessor(temperature: float, top_k: int)
```

**参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `temperature` | `float` | 温度参数 |
| `top_k` | `int` | 保留的 token 数量 |

**示例**:

```python
from utils.logits_processor import TopKProcessor

processor = TopKProcessor(temperature=0.8, top_k=5)
```

---

### 3.5 NucleusProcessor

**文件**: `utils/logits_processor.py:66-81`

**Top-p（Nucleus）采样**：从累积概率达到 p 的最小集合中采样。

**构造函数**:

```python
NucleusProcessor(temperature: float, top_p: float)
```

**参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `temperature` | `float` | 温度参数 |
| `top_p` | `float` | 累积概率阈值（0-1） |

**示例**:

```python
from utils.logits_processor import NucleusProcessor

processor = NucleusProcessor(temperature=0.8, top_p=0.9)
```

---

### 3.6 TopKNucleusProcessor

**文件**: `utils/logits_processor.py:84-103`

**Top-K + Top-p 组合采样**：同时应用 Top-K 和 Top-p 过滤。

**构造函数**:

```python
TopKNucleusProcessor(temperature: float, top_k: int, top_p: float)
```

**参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `temperature` | `float` | 温度参数 |
| `top_k` | `int` | Top-K 阈值 |
| `top_p` | `float` | Top-p 阈值 |

**示例**:

```python
from utils.logits_processor import TopKNucleusProcessor

processor = TopKNucleusProcessor(temperature=0.8, top_k=10, top_p=0.9)
```

---

## 4. 工具函数

### 4.1 max_fn

**文件**: `sampling/speculative_decoding.py:10-19`

**样本调整函数**，用于 Speculative Decoding。

**函数签名**:

```python
def max_fn(x: torch.Tensor) -> torch.Tensor
```

**功能**：实现 `max(0, x)` 的归一化。

$$\text{max_fn}(x) = \frac{\max(0, x)}{\sum \max(0, x)}$$

**用途**：当草稿被拒绝时，调整 target 的分布。

**示例**:

```python
from sampling.speculative_decoding import max_fn

p = torch.tensor([0.3, 0.5, 0.2])  # target 分布
q = torch.tensor([0.4, 0.3, 0.3])  # drafter 分布

adjusted = max_fn(p - q)  # 调整后的分布
```

---

### 4.2 prune_cache

**文件**: `utils/caching.py:6-24`

**KV Cache 修剪函数**，移除被拒绝的草稿。

**函数签名**:

```python
def prune_cache(
    cache: Union[Tuple[Tuple[Tensor]], DynamicCache],
    num_tokens_to_discard: int
) -> Union[Tuple[Tuple[Tensor]], DynamicCache]
```

**参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `cache` | `Tuple or DynamicCache` | KV Cache |
| `num_tokens_to_discard` | `int` | 要移除的 token 数量 |

**返回**:

- 修剪后的 KV Cache

**示例**:

```python
from utils.caching import prune_cache

# 移除最后 3 个 token 的 cache
new_cache = prune_cache(cache, num_tokens_to_discard=3)
```

---

## 5. 完整示例

### 5.1 基础用法

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from sampling import speculative_generate, autoregressive_generate
from utils.logits_processor import GreedyProcessor

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
drafter = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# 准备输入
inputs = "The capital of France is"
input_ids = tokenizer(inputs, return_tensors="pt").input_ids[0].tolist()

# 自回归生成
output_ar = autoregressive_generate(input_ids, target)

# Speculative Decoding
output_sd, acceptance_rate = speculative_generate(
    input_ids,
    drafter=drafter,
    target=target,
    gamma=4,
)

# 解码
print("AR:", tokenizer.decode(output_ar, skip_special_tokens=True))
print("SD:", tokenizer.decode(output_sd, skip_special_tokens=True))
print(f"Acceptance rate: {acceptance_rate:.2%}")
```

### 5.2 NASD 用法

```python
from ngram_assisted import NGramStorage, ngram_assisted_speculative_generate
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# 准备输入
inputs = "def factorial(n):\n    if n == 0:\n        return "
input_ids = tokenizer(inputs, return_tensors="pt").input_ids[0].tolist()

# 创建 Ngram 存储
ngram = NGramStorage(n=3, vocab_size=target.config.vocab_size)

# NASD 生成
output, acceptance_rate = ngram_assisted_speculative_generate(
    input_ids,
    ngramstorage=ngram,
    target=target,
    gamma=4,
    filler_top_k=3,
)

# 解码
print("NASD:", tokenizer.decode(output, skip_special_tokens=True))
```

### 5.3 高级用法

```python
from sampling import speculative_generate
from utils.logits_processor import NucleusProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
drafter = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# 准备输入
inputs = "Write a short poem about AI."
input_ids = tokenizer(inputs, return_tensors="pt").input_ids[0].tolist()

# 使用 Nucleus 采样
processor = NucleusProcessor(temperature=0.8, top_p=0.9)

# Speculative Decoding with cache
output, acceptance_rate = speculative_generate(
    input_ids,
    drafter=drafter,
    target=target,
    gamma=6,
    logits_processor=processor,
    use_cache=True,
    first_target=True,
    debug=True,
)

# 解码
print("Output:", tokenizer.decode(output, skip_special_tokens=True))
print(f"Acceptance rate: {acceptance_rate:.2%}")
```

---

## 附录

### 导入路径

```python
# 生成函数
from sampling import autoregressive_generate
from sampling import speculative_generate
from sampling import beam_search_generate
from ngram_assisted import ngram_assisted_speculative_generate

# Ngram Storage
from ngram_assisted import INgramStorage
from ngram_assisted import NGramStorage
from ngram_assisted import OneLevelNGramStorage

# Logits Processors
from utils.logits_processor import LogitsProcessor
from utils.logits_processor import GreedyProcessor
from utils.logits_processor import MultinomialProcessor
from utils.logits_processor import TopKProcessor
from utils.logits_processor import NucleusProcessor
from utils.logits_processor import TopKNucleusProcessor

# 工具函数
from sampling.speculative_decoding import max_fn
from utils.caching import prune_cache
```

### 类型提示

```python
from typing import List, Tuple, Union
from torch import Tensor
from torch.nn import Module
```

### 相关文档

- [01. 大模型推理基础](./01-inference-basics.md)
- [02. Speculative Decoding 理论](./02-speculative-theory.md)
- [03. Speculative Decoding 实现](./03-speculative-impl.md)
- [04. NASD 理论](./04-nasd-theory.md)
- [05. NASD 实现](./05-nasd-impl.md)
- [06. 技巧与优化](./06-tricks-and-optimizations.md)
- [07. 性能基准测试](./07-performance-benchmarks.md)
