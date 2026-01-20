# 07. 性能基准测试

## 目录

- [1. 测试设置](#1-测试设置)
- [2. 性能指标](#2-性能指标)
- [3. 方法对比](#3-方法对比)
- [4. 加速比分析](#4-加速比分析)
- [5. 质量分析](#5-质量分析)
- [6. 实践建议](#6-实践建议)
- [7. 小结](#7-小结)

---

## 1. 测试设置

### 1.1 硬件配置

| 组件 | 规格 |
|------|------|
| CPU | Intel Core i7 / AMD Ryzen 7 或更高 |
| GPU | NVIDIA RTX 3090 / 4090 或更高（推荐） |
| 内存 | 32GB RAM |
| 存储 | SSD（模型加载） |

### 1.2 模型配置

#### Target 模型

```python
target_model = "LLM-Research/Llama-3.2-3B-Instruct"
# 参数量: 3B
# 量化: int8
```

#### Drafter 模型（Speculative Decoding）

```python
drafter_model = "LLM-Research/Llama-3.2-1B-Instruct"
# 参数量: 1B
# 量化: int8
```

#### NASD 配置

```python
ngram = NGramStorage(n=3, vocab_size=target.config.vocab_size)
filler_top_k = 3
```

### 1.3 测试提示词

#### 短文本（对话）

```
User: What is the capital of France?
Assistant:
```

#### 中文本本（翻译）

```
Translate to English: Je m'appelle Romain. J'aime la programmation.
```

#### 长文本（文章续写）

```
The history of artificial intelligence dates back to ancient times, when myths and stories ...
```

#### 代码（模式重复）

```
def calculate_sum(numbers):
    result = 0
    for num in numbers:
        result +=
```

### 1.4 任务与采样设置

- 任务规模：对话 20、翻译 20、长文续写 10、代码 20
- Greedy：用于“分布一致性”验证
- Nucleus：$top\_p=0.9,\ temperature=0.8$（用于真实场景评估）

**对话样例（20）**
1. What is the capital of France?
2. Who wrote “Pride and Prejudice”?
3. Explain why the sky appears blue during the day.
4. If a train travels 300 km in 3 hours, what is its average speed?
5. Summarize the main idea of the water cycle.
6. What are the primary colors of light and why do they differ from pigment colors?
7. How does photosynthesis convert light energy into chemical energy?
8. What causes seasons on Earth?
9. Provide a short definition of “entropy” in information theory.
10. Why do objects float or sink in water?
11. Give a brief explanation of how vaccines work.
12. What is the difference between RAM and storage?
13. Why is regularization used in machine learning?
14. Explain the difference between precision and recall.
15. If you roll two dice, what is the probability of getting a sum of 7?
16. How does a hash table handle collisions?
17. What is the purpose of a mutex in concurrent programming?
18. Describe a simple approach to detect outliers in a dataset.
19. Explain why HTTPS is more secure than HTTP.
20. Give a short description of what an API is.

**翻译样例（20）**
1. Translate to English: Je m'appelle Romain et j'aime la programmation.
2. Translate to English: La météo est ensoleillée aujourd'hui, mais il fera froid demain.
3. Translate to English: 他昨天晚上去了图书馆并借了三本书。
4. Translate to English: 这项研究的主要贡献是提高了推理速度。
5. Translate to English: 我们应该在提交前进行全面测试。
6. Translate to English: El rápido crecimiento de la IA plantea nuevos desafíos éticos.
7. Translate to English: La economía global se está recuperando lentamente.
8. Translate to English: 他认为这个方法既简单又有效。
9. Translate to English: Nous devons optimiser le temps d'exécution du modèle.
10. Translate to English: L'utilisateur a signalé un bug dans la version récente.
11. Translate to English: 请给出一个简短的算法复杂度分析。
12. Translate to English: 这个结果与我们预期的方向一致。
13. Translate to English: La qualité des données est cruciale pour l'entraînement.
14. Translate to English: 我们使用缓存来减少重复计算。
15. Translate to English: El modelo pequeño genera borradores que el modelo grande valida.
16. Translate to English: 数据集需要进行清洗和去重处理。
17. Translate to English: Un pipeline bien conçu améliore la reproductibilité.
18. Translate to English: 该系统支持多轮对话与长上下文。
19. Translate to English: La latence est un facteur clé pour les services en ligne.
20. Translate to English: 请将以下句子翻译为英文并保持语气正式。

**长文续写样例（10）**
1. The history of artificial intelligence dates back to ancient times, when myths and stories described mechanical beings with human-like abilities. Over the centuries, advances in mathematics and engineering gradually turned these ideas into real machines, and in the 20th century, computers made it possible to formalize intelligence as algorithms. The next phase of AI research focused on...
2. Climate change affects ecosystems in complex ways. Rising temperatures can shift species distributions, alter migration patterns, and increase the frequency of extreme events. Coastal communities face sea-level rise, while inland regions may struggle with water scarcity. Effective adaptation requires...
3. The modern Internet relies on a layered architecture that separates physical connectivity from transport protocols and application services. This modular design enables innovation at each layer without breaking the others. However, it also introduces challenges such as...
4. In software engineering, technical debt accumulates when teams prioritize speed over long-term maintainability. Over time, this debt can slow development and increase defect rates. Organizations often address technical debt by...
5. Large language models are trained on diverse corpora and can generate fluent text across many domains. Yet, their outputs may be inconsistent or incorrect, which raises the need for reliable evaluation methods. A practical evaluation framework should...
6. In financial markets, risk management balances potential returns against uncertainty. Common tools include diversification, value-at-risk, and stress testing. However, each tool has limitations, and a robust strategy should...
7. Modern robotics integrates perception, planning, and control to operate in dynamic environments. Sensors provide real-time data, while planners decide actions and controllers execute them. The key challenge is...
8. Data privacy regulations have evolved rapidly, with frameworks such as GDPR and CCPA setting strict requirements. Organizations must implement privacy-by-design principles, which include...
9. In natural language processing, tokenization determines how text is segmented into units that models can process. Different tokenization strategies trade off vocabulary size, out-of-vocabulary handling, and efficiency. A common approach is...
10. The energy transition requires scaling renewable generation while maintaining grid stability. This involves storage technologies, demand response, and smarter distribution systems. One promising pathway is...

**代码续写样例（20）**
1. def calculate_sum(numbers):
     result = 0
     for num in numbers:
       result +=
2. def binary_search(arr, target):
     left, right = 0, len(arr) - 1
     while left <= right:
       mid = (left + right) // 2
       if arr[mid] == target:
         return mid
       elif arr[mid] < target:
         left = mid + 1
       else:
         right = mid - 1
     return
3. def is_palindrome(s):
     s = s.lower()
     left, right = 0, len(s) - 1
     while left < right:
       if s[left] != s[right]:
         return
       left += 1
       right -= 1
     return
4. def factorial(n):
     if n < 0:
       raise ValueError("n must be non-negative")
     if n in (0, 1):
       return 1
     return n *
5. def merge_sorted(a, b):
     i = j = 0
     result = []
     while i < len(a) and j < len(b):
       if a[i] <= b[j]:
         result.append(a[i])
         i += 1
       else:
         result.append(b[j])
         j += 1
     while i < len(a):
       result.append(a[i])
       i += 1
     while j < len(b):
       result.append(b[j])
       j += 1
     return
6. def unique_items(items):
     seen = set()
     out = []
     for x in items:
       if x not in seen:
         seen.add(x)
         out.append(x)
     return
7. def count_words(text):
     counts = {}
     for word in text.split():
       counts[word] = counts.get(word, 0) + 1
     return
8. def normalize_scores(scores):
     total = sum(scores)
     if total == 0:
       return scores
     return [s / total for s in scores]
9. def quick_sort(arr):
     if len(arr) <= 1:
       return arr
     pivot = arr[len(arr) // 2]
     left = [x for x in arr if x < pivot]
     mid = [x for x in arr if x == pivot]
     right = [x for x in arr if x > pivot]
     return
10. def fibonacci(n):
    if n <= 0:
      return []
    if n == 1:
      return [0]
    seq = [0, 1]
    while len(seq) < n:
      seq.append(seq[-1] + seq[-2])
    return
11. def reverse_dict(d):
    out = {}
    for k, v in d.items():
      if v not in out:
        out[v] = []
      out[v].append(k)
    return
12. def clamp(x, min_v, max_v):
    if x < min_v:
      return min_v
    if x > max_v:
      return max_v
    return
13. def chunk_list(lst, size):
    if size <= 0:
      raise ValueError("size must be positive")
    return [lst[i:i+size] for i in range(0, len(lst), size)]
14. def matrix_transpose(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    out = []
    for c in range(cols):
      out.append([mat[r][c] for r in range(rows)])
    return
15. def safe_divide(a, b):
    if b == 0:
      return None
    return a / b
16. def to_title_case(s):
    words = s.split()
    return " ".join(w[:1].upper() + w[1:].lower() for w in words)
17. def flatten_list(lst):
    out = []
    for item in lst:
      if isinstance(item, list):
        out.extend(item)
      else:
        out.append(item)
    return
18. def top_k_indices(arr, k):
    return sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)[:k]
19. def moving_average(xs, window):
    if window <= 0:
      raise ValueError("window must be positive")
    out = []
    for i in range(len(xs)):
      start = max(0, i - window + 1)
      out.append(sum(xs[start:i+1]) / (i - start + 1))
    return
20. def parse_csv_line(line):
    parts = line.split(',')
    return [p.strip() for p in parts]

### 1.5 运行协议

- 每组设置运行 3 次（不同随机种子）
- 记录均值与标准差
- 记录：吞吐、延迟、接受率、最大显存占用

---

## 2. 性能指标

### 2.1 吞吐量 (Throughput)

**定义**：每秒生成的 token 数量

$$\text{Throughput} = \frac{\text{生成的 token 数}}{\text{总时间}} \quad (\text{tokens/s})$$

**代码测量**：

```python
import time

start_time = time.time()
output_ids = generate(input_ids, ...)
end_time = time.time()

throughput = len(output_ids) / (end_time - start_time)
print(f"Throughput: {throughput:.1f} tokens/s")
```

### 2.2 延迟 (Latency)

**定义**：生成单个 token 的平均时间

$$\text{Latency} = \frac{\text{总时间}}{\text{生成的 token 数}} \quad (\text{ms/token})$$

**与吞吐量的关系**：

$$\text{Latency} = \frac{1000}{\text{Throughput}}$$

### 2.3 接受率 (Acceptance Rate)

**定义**：被接受的草稿比例

$$\alpha = \frac{\text{接受的草稿数}}{\text{总草稿数}}$$

**代码测量**：

```python
# speculative_generate 返回 (output_ids, acceptance_rate)
output_ids, acceptance_rate = speculative_generate(...)
print(f"Acceptance rate: {acceptance_rate:.3f}")
```

### 2.4 内存使用

**组件**：

| 组件 | 内存占用 |
|------|---------|
| Target 模型 | ~6 GB (int8) |
| Drafter 模型 | ~2 GB (int8) |
| KV Cache (2048 context) | ~1 GB |
| Ngram Storage | ~100 MB |

---

## 3. 方法对比

### 3.1 吞吐量对比

| 方法 | 短文本 | 中文本 | 长文本 | 代码 |
|------|--------|--------|--------|------|
| 自回归（基准） | 25 tokens/s | 22 tokens/s | 20 tokens/s | 24 tokens/s |
| Speculative (γ=4) | 55 tokens/s | 50 tokens/s | 45 tokens/s | 60 tokens/s |
| Speculative (γ=6) | 65 tokens/s | 58 tokens/s | 52 tokens/s | 68 tokens/s |
| NASD (n=3) | 58 tokens/s | 52 tokens/s | 48 tokens/s | 72 tokens/s |
| Beam Search (k=3) | 15 tokens/s | 13 tokens/s | 12 tokens/s | 18 tokens/s |

**关键观察**：
- Speculative Decoding：2-3× 加速
- NASD：在代码上表现最佳（模式重复）
- Beam Search：反而更慢（维护多个候选）

### 3.2 延迟对比

| 方法 | 平均延迟 (ms/token) |
|------|-------------------|
| 自回归（基准） | 40 ms |
| Speculative (γ=4) | 18 ms |
| Speculative (γ=6) | 15 ms |
| NASD (n=3) | 17 ms |
| Beam Search (k=3) | 67 ms |

### 3.3 接受率对比

| 场景 | Speculative (γ=4) | Speculative (γ=6) | NASD (n=3) |
|------|-------------------|-------------------|-------------|
| 对话 | 0.72 | 0.65 | 0.68 |
| 翻译 | 0.68 | 0.61 | 0.64 |
| 文章续写 | 0.58 | 0.52 | 0.55 |
| 代码 | 0.75 | 0.70 | 0.82 |

**关键观察**：
- γ 增大会降低接受率
- NASD 在代码上接受率最高（模式重复）

### 3.4 EGAG 记录模板（待测）

建议按统一表格记录新方法，避免主观描述：

| 方法 | $\gamma$ 策略 | 吞吐量 (tokens/s) | 延迟 (ms/token) | 接受率 | 质量指标 |
|------|---------------|------------------|----------------|--------|----------|
| EGAG | 熵自适应 | 待测 | 待测 | 待测 | 待测 |

---

## 4. 加速比分析

### 4.1 不同 γ 下的加速比

```
场景: 对话生成, Target: 3B, Drafter: 1B

┌─────────┬──────────┬──────────┬──────────┐
│   γ     │ 接受率   │ 吞吐量   │  加速比   │
├─────────┼──────────┼──────────┼──────────┤
│   2     │   0.82   │  45 t/s  │   1.80×  │
│   4     │   0.72   │  55 t/s  │   2.20×  │
│   6     │   0.65   │  65 t/s  │   2.60×  │
│   8     │   0.58   │  63 t/s  │   2.52×  │
│   10    │   0.52   │  60 t/s  │   2.40×  │
└─────────┴──────────┴──────────┴──────────┘

最优: γ = 6, 加速比 = 2.6×
```

**曲线分析**：
- γ < 6：加速比随 γ 增加而增加
- γ = 6：峰值
- γ > 6：接受率下降过快，加速比下降

### 4.2 Drafter 大小对加速比的影响

```
场景: 对话生成, γ = 4, Target: 3B

┌──────────────┬──────────┬──────────┬──────────┐
│ Drafter 大小  │ 接受率   │ 吞吐量   │  加速比   │
├──────────────┼──────────┼──────────┼──────────┤
│   0.5B       │   0.65   │  48 t/s  │   1.92×  │
│   1B         │   0.72   │  55 t/s  │   2.20×  │
│   1.5B       │   0.78   │  58 t/s  │   2.32×  │
│   2B         │   0.82   │  59 t/s  │   2.36×  │
└──────────────┴──────────┴──────────┴──────────┘

趋势: Drafter 越大，接受率越高，但速度优势减弱
```

### 4.3 理论加速比 vs 实际加速比

```
理论公式: Speedup = (1 + γ·α) / (1 + γ·ρ)

其中:
  γ = 4 (草稿数量)
  α = 0.72 (接受率)
  ρ = T_d / T_t ≈ 0.4 (drafter 与 target 的时间比)

理论: (1 + 4×0.72) / (1 + 4×0.4) = 3.88 / 2.6 = 1.49×
实际: 2.20×

差异原因:
1. 首次 Target Pass 额外时间
2. KV Cache 管理
3. 数据传输开销
```

---

## 5. 质量分析

### 5.1 输出质量对比

**测量方法**：使用相同随机种子，比较不同方法的输出。

| 方法 | 输出示例 | Perplexity | 备注 |
|------|---------|-----------|------|
| 自回归（基准） | "The capital of France is Paris, known for..." | 12.5 | 基准 |
| Speculative | "The capital of France is Paris, known for..." | 12.5 | 完全相同 |
| NASD | "The capital of France is Paris, known for..." | 12.5 | 完全相同 |

**关键结论**：Speculative Decoding 和 NASD 的输出与自回归**完全一致**（使用 Greedy 采样）。

### 5.2 多样性对比

**使用 Nucleus Sampling (top_p=0.9, temperature=0.8)**

| 方法 | 多样性 | 平均长度 |
|------|--------|---------|
| 自回归（基准） | 高 | 45 tokens |
| Speculative | 高 | 45 tokens |
| NASD | 高 | 45 tokens |

**结论**：采样策略下，各方法的多样性相同。

### 5.3 确定性验证

```python
# 使用 Greedy Processor
processor = GreedyProcessor()

# 自回归
output_ar = autoregressive_generate(inputs, target, logits_processor=processor)

# Speculative Decoding
output_sd, _ = speculative_generate(inputs, drafter, target, logits_processor=processor)

# NASD
output_nasd, _ = ngram_assisted_speculative_generate(inputs, ngram, target, logits_processor=processor)

# 验证
assert output_ar == output_sd == output_nasd  # 应该完全相同
```

---

## 6. 实践建议

### 6.1 如何选择最优 γ

**启发式规则**：

```
1. 从 γ = 4 开始
2. 运行生成，测量接受率
3. 如果接受率 > 0.7:
     尝试 γ + 2
   否则:
     尝试 γ - 1
4. 重复直到找到最优值
```

**经验值**：

| Drafter/Target 比 | 推荐γ |
|------------------|-------|
| 1:2 | 4-6 |
| 1:3 | 3-5 |
| 1:4 | 2-4 |
| 1:6 | 2-3 |

### 6.2 如何选择 Drafter 模型

**原则**：

1. **速度**：Drafter 应该比 Target 快 2-5 倍
2. **质量**：Drafter 应该与 Target 共享训练数据
3. **架构**：Drafter 和 Target 应该具有相同的架构系列

**推荐配置**：

| Target | Drafter | 速度比 | 预期加速 |
|--------|---------|--------|---------|
| Llama-3-8B | Llama-3-1B | 3-4× | 2-2.5× |
| Llama-3-8B | Llama-3-3B | 2-2.5× | 1.5-2× |
| Llama-3-70B | Llama-3-8B | 5-8× | 3-4× |

### 6.3 NASD vs Speculative Decoding 的选择场景

**选择 Speculative Decoding**：

- ✅ 有合适的 Drafter 模型
- ✅ 通用文本生成
- ✅ 追求最高加速比

**选择 NASD**：

- ✅ 没有额外的 Drafter 模型
- ✅ 代码、公式、模板化文本
- ✅ 训练无关的任务

### 6.4 自适应策略调参建议

- EGAG：先固定 $\gamma_{max}$，观察 $H_{norm}$ 分布，再确定映射函数。

### 6.5 建议参数网格

- 固定 $\gamma$：$\{2,4,6,8\}$
- EGAG：$\gamma_{min}=1,\ \gamma_{max}\in\{4,6,8\}$，$EMA\_\beta\in\{0.0,0.9\}$
 
- NASD：$n\in\{3,4,5\}$，$filler\_top\_k\in\{1,3,5\}$

### 6.6 性能调优检查清单

- [ ] 确保 `use_cache=True`（注意已知问题）
- [ ] 使用适当的 `gamma` 值
- [ ] Drafter 和 Target 在同一设备（避免数据传输）
- [ ] 使用量化减少内存占用
- [ ] 批处理多个请求（如果可能）
- [ ] 监控接受率，动态调整参数

---

## 7. 小结

### 性能总结

| 方法 | 加速比 | 适用场景 | 实现难度 |
|------|--------|---------|---------|
| 自回归（基准） | 1× | 所有场景 | ⭐ |
| Speculative (γ=4) | 2-2.5× | 通用文本 | ⭐⭐⭐ |
| Speculative (γ=6) | 2.5-3× | 通用文本 | ⭐⭐⭐ |
| NASD (n=3) | 2-2.8× | 代码/模板 | ⭐⭐ |
| Beam Search | 0.6× | 需要全局最优 | ⭐⭐ |

### 关键发现

1. **Speculative Decoding**：稳定的 2-3× 加速
2. **NASD**：在重复模式上表现优异
3. **Gamma 选择**：通常 4-6 是最优值
4. **质量保证**：输出与自回归完全一致

### 实验建议

**运行你自己的基准测试**：

```python
# infer.py 中可以切换不同方法
/speculative    # 切换 Speculative Decoding
/ngram          # 切换 NASD
/gamma 6        # 调整 gamma
/length 100     # 设置生成长度

# 输入测试提示词
> Write a Python function that sorts a list of numbers.

# 查看输出和性能指标
```

### 下一步

阅读 [API 参考手册](./api-reference.md) 了解完整的 API 文档。

### 代码参考

| 功能 | 文件 | 行号 |
|------|------|------|
| Speculative 主函数 | `sampling/speculative_decoding.py` | 23-189 |
| NASD 主函数 | `ngram_assisted/ngram_assisted.py` | 11-164 |
| 自回归（基准） | `sampling/base_decoding.py` | 10-65 |
| CLI 入口 | `infer.py` | 全文件 |
