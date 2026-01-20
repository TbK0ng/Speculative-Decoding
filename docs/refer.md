# 参考文档：推理加速方向与可实现的 Speculative Trick（EGAG）

本文档整理自对话内容，用于后续在本仓库实现推理加速的 EGAG“小 trick”。内容覆盖：近两年方向综述、可复现基线、创新策略、实验设计与论文结构建议。

---

## 1. 近两年（2024–2025）大模型推理加速主要方向

1) Speculative Decoding（推测解码）
- 核心：用小模型或其他机制一次生成多个 draft token，再由大模型并行验证，获得 2–3× 甚至更高加速，几乎不降精度。
- 代表工作：
  - 经典 Speculative Decoding（ICML 2023）
  - Online Speculative Decoding (OSD, ICML 2024)
  - REST：Retrieval‑Based Speculative Decoding（NAACL 2024）
  - SpecDec++（COLM 2025 / ICML 2024 workshop）
  - AdaEDL / AdaSpec / Entropy‑Aware 等自适应策略
  - Gumiho：混合架构、优先高质量前几个 token
- 优点：已有大量开源实现，算法层面改动少，适合作为毕业论文入口。

2) KV Cache 管理与压缩 / Offloading
- 主要用于长上下文：KV Cache 占显存和带宽巨大。
- 代表工作：KV Cache 压缩综述、RocketKV、LMCache 等。
- 偏系统实现，复现门槛较高。

3) Early Exit / Layer Skipping / Self‑Speculative
- 典型：LayerSkip、SpecEE。
- 通常涉及模型结构或训练改动，复杂度更高。

4) 注意力/Kernel 级加速
- 如 FlashAttention / FlashInfer 等。
- 更偏 CUDA/Kernel 工程优化，不太适合算法级小改动场景。

**结论**：在“易复现 + 可加两个小 trick 做算法创新”的需求下，Speculative Decoding 最合适。

---

## 2. 推荐的可复现基线：ROMSTO 的 Speculative-Decoding 实现

**选择理由**
- 代码实现更新于近两年，且为纯 PyTorch + HuggingFace 生态。
- 结构清晰，核心逻辑集中，适合作为算法改动基础。
- 已包含 NASD（N‑gram Assisted Speculative Decoding），便于做混合策略。

**功能概述**
- 经典自回归解码（baseline）
- Speculative Decoding（drafter + target）
- Beam Search
- NASD：用 N‑gram 统计替代小模型生成部分候选

---

## 2.1 Baseline 选择建议（面向两年后毕业）

**是否一定要选 2025 年的 baseline？**

不必。毕业论文更看重“可复现 + 可清晰对比 + 与创新点紧密相关”。
Speculative Decoding 的核心算法在 2023 已成熟，2024–2025 的工作多为增强与工程优化。选择 2023–2024 的稳定基线更容易保证复现与对比公平性。

**建议的 baseline 组合（推荐顺序）**

1. 经典 Speculative Decoding（SD, 2023）
	- 作为“最基础对照”。
2. ROMSTO/Speculative-Decoding 开源实现
	- 作为你实际可改动的代码基线。
3. NASD（若仓库已集成）
	- 作为“非小模型草稿源”的对比基线。
4. 若需 2024–2025 强 baseline（可选）
	- SpecDec++ / OSD 等具备代码的工作（优先选有开源实现者）。

**选择标准**

- 能稳定复现（有开源代码/明确实现细节）
- 与你的创新点高度相关（自适应候选长度、混合 draft 源）
- 实验成本可控（不依赖大型训练或复杂工程）

---

## 3. 可直接实现的“小 trick”算法创新

### Trick 1：基于熵的自适应 gamma（EGAG）

**动机**
- 固定 gamma 在困难 token 上会浪费草稿（高否决率），在简单 token 上又不够快。

**核心思路**
- 计算 drafter 在当前 token 的预测分布熵 $H$，动态调整 gamma：
  - 熵高（不确定）→ gamma 变小
  - 熵低（自信）→ gamma 变大

**关键公式**
$$H=-\sum_i p_i \log p_i$$
$$H_{norm}=\frac{H}{\log V}$$
$$\gamma=\lfloor (1-H_{norm})\cdot \gamma_{max} \rfloor,\ \gamma\ge 1$$

**实现要点**
- 在 `speculative_generate` 内，对 drafter 输出 logits 做 softmax；计算熵并映射到 gamma。
- 仅需几行代码，无需训练。

**改进建议（保证稳定与可写性）**

1. 熵归一化与温度一致性：在计算熵前固定温度或将 temperature 纳入归一化。
2. 映射函数采用分段或平滑映射：避免 gamma 频繁跳动（如 EMA 平滑）。
3. 设定上下界：$\gamma \in [\gamma_{min}, \gamma_{max}]$。
4. 记录统计：输出平均 $H_{norm}$、平均 gamma、接受率曲线，便于论文分析。

**可写进论文的理由**
- 信息论动机清晰；与现有 Entropy‑Aware 方向一致，但实现更简洁。
- 训练‑free，工程改动小，易于复现。

---

## 4. 实验设计建议

**对比方法**
1. 基线 1：纯自回归解码
2. 基线 2：固定 gamma 的 Speculative Decoding
3. 基线 3：纯 NASD（若支持）
4. Method A：EGAG

**评估指标**
- 速度：tokens/s、end‑to‑end latency
- 效率：平均 acceptance tokens、drafter/target 的 FLOPs 估计
- 质量：BLEU/ROUGE/perplexity 或主观评估

**模型与数据**
- 模型：Llama‑3.x 3B（target）+ 1B（drafter）
- 数据：WikiText‑103、C4 子集、翻译数据等
- 输入长度 50–200 tokens，生成长度 50–200 tokens

---

## 5. 论文结构建议

1. 引言：推理成本、加速方向、本文贡献
2. 相关工作：Speculative Decoding、Entropy‑Aware、NASD/REST
3. 方法：
	- 3.1 基础 Speculative Decoding
	- 3.2 EGAG
	- 3.3 EGAG 的实现与消融
4. 实验设置：模型、数据、硬件、超参
5. 实验结果与分析：
	- EGAG vs 固定 gamma
6. 讨论：局限性与 trade‑off
7. 结论与未来工作

---

## 6. 推进节奏建议

**第 1 周**：跑通 baseline 与 NASD，理解 `speculative_generate`

**第 2 周**：实现 EGAG，验证不同 gamma_max 的曲线

**第 3 周**：完善实验与论文撰写

---

## 7. 参考链接

1. Fast inference from transformers via speculative decoding
	- https://arxiv.org/abs/2211.17192
2. Online Speculative Decoding (OSD)
	- https://github.com/LiuXiaoxuanPKU/OSD
3. REST: Retrieval-Based Speculative Decoding
	- https://github.com/FasterDecoding/REST
4. SpecDec++
	- https://github.com/Kaffaljidhmah2/SpecDec_pp
5. AdaEDL
	- https://arxiv.org/abs/2410.18351
6. Entropy-Aware Speculative Decoding
	- https://arxiv.org/abs/2512.23765
7. Gumiho
	- https://openreview.net/forum?id=0ObGn4e1IS
8. KV Cache 加速综述
	- https://arxiv.org/abs/2412.19442
9. RocketKV
	- https://github.com/NVlabs/RocketKV
10. LMCache
	- https://github.com/LMCache/LMCache
11. LayerSkip
	- https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/
12. SpecEE
	- https://dl.acm.org/doi/10.1145/3695053.3730996
