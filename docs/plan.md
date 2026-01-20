# EGAG 实施计划

本文档用于规划改进算法（EGAG）的实现与实验验证。

---

## 1. 目标与范围

**目标**
- 在现有 Speculative Decoding 基线之上实现：
   - EGAG（熵自适应 $\gamma$）
- 形成可复现实验结果与消融对比。

**范围**
- 只改动推理逻辑（training‑free）。
- 不引入新的模型训练。

---

## 2. Baseline 设定

**必选**
- 标准自回归解码（AR）
- Speculative Decoding（固定 $\gamma$）
- NASD（若启用 N‑gram）

**可选增强基线**
- OSD / SpecDec++（仅在有开源实现且成本可控时）

---

## 3. 实施步骤

### 3.1 代码改动清单

**核心修改点**
- `sampling/speculative_decoding.py`
   - EGAG：在草稿生成前计算熵并更新 $\gamma$
- `ngram_assisted/ngram_assisted.py`
  - 复用 NASD draft 生成逻辑
- `utils/logits_processor.py`
  - 复用 softmax/采样接口（如需）

### 3.2 EGAG 实现步骤

1. 取 drafter 当前 logits，计算 $H$ 与 $H_{norm}$。
2. 映射得到 $\gamma$，并执行上下界裁剪。
3. 可选：对 $\gamma$ 采用 EMA 平滑。
4. 记录统计（平均 $H_{norm}$、平均 $\gamma$、接受率）。

### 3.3 实验联动（可选）

- 在低熵时增大 $\gamma$，高熵时缩小 $\gamma$。

---

## 4. 实验矩阵（细化版）

### 4.1 任务与输入集合

- 对话：20 条简短问答（事实类/推理类各半）
- 翻译：20 条短句（长度 20–60 tokens）
- 长文续写：10 条长 prompt（200–400 tokens）
- 代码：20 条代码片段续写（含循环/函数模板）

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

### 4.2 采样设置

- Greedy：用于“分布一致性”验证
- Nucleus：$top\_p=0.9,\ temperature=0.8$（用于多样性与真实场景）

### 4.3 关键超参网格

**固定基线**
- $\gamma \in \{2,4,6,8\}$
- NASD：$n \in \{3,4,5\}$，$filler\_top\_k \in \{1,3,5\}$

**EGAG**
- $\gamma_{min}=1$，$\gamma_{max} \in \{4,6,8\}$
- 熵平滑：$EMA\_\beta \in \{0.0, 0.9\}$

 

### 4.4 统计与重复次数

- 每组设置运行 3 次（不同随机种子）
- 记录均值与标准差

### 4.5 方法矩阵

| 方法 | $\gamma$ 策略 | 任务 | 指标 |
|------|---------------|------|------|
| AR | 固定 | 对话/翻译/长文/代码 | 吞吐、延迟、质量 |
| SD | 固定 $\gamma$ | 同上 | 吞吐、延迟、接受率 |
| NASD | 固定 | 同上 | 吞吐、延迟、接受率 |
| EGAG | 熵自适应 | 同上 | 吞吐、延迟、接受率 |

---

## 5. 里程碑

1. **M1：基线复现**
   - 跑通 AR / SD / NASD
   - 记录基线吞吐与接受率

2. **M2：EGAG 完成**
   - 通过单元测试与基本生成
   - 输出 $H_{norm}$ 与 $\gamma$ 统计

3. **M3：消融实验**
   - EGAG vs 固定 $\gamma$

4. **M4：结果整理**
   - 汇总表格与曲线
   - 形成论文图表素材

---

## 6. 评估与验收标准

- **正确性**：输出分布与基线一致（Greedy/固定采样下）
- **性能**：吞吐量有显著提升或延迟下降
- **稳定性**：接受率波动降低，极端输入可回退

---

## 7. 风险与对策

- **熵估计受温度影响** → 固定温度或归一化
- **阈值迁移失效** → 使用分位数/自适应阈值
- **N‑gram 冷启动** → 强制回退 drafter

---

## 8. 交付物清单

- 实现代码（EGAG）
- 实验结果表格与曲线
- 消融实验分析
- 论文方法与实验章节素材
