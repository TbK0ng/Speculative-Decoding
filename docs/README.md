# Speculative Decoding 文档

欢迎来到 Speculative Decoding 完整文档！本文档旨在帮助你从零开始理解大模型推理加速技术，特别是 Speculative Decoding（投机解码）及其变体。

## 文档导航

### 📚 基础篇

| 文档 | 描述 |
|------|------|
| [01. 大模型推理基础](./01-inference-basics.md) | 理解 Transformer 自回归生成、KV Cache、传统生成方法和推理瓶颈 |

### 🔬 理论篇

| 文档 | 描述 |
|------|------|
| [02. Speculative Decoding 理论](./02-speculative-theory.md) | 深入数学原理：拒绝采样推导、加速比分析、最优参数 |
| [04. NASD 理论](./04-nasd-theory.md) | Ngram Assisted Speculative Decoding 的创新思想 |

### 💻 实现篇

| 文档 | 描述 |
|------|------|
| [03. Speculative Decoding 实现](./03-speculative-impl.md) | 结合代码理解具体实现细节 |
| [05. NASD 实现](./05-nasd-impl.md) | NASD 的代码实现详解 |

### 🚀 进阶篇

| 文档 | 描述 |
|------|------|
| [06. 技巧与优化](./06-tricks-and-optimizations.md) | 项目中的实现技巧和优化点 |
| [07. 性能基准测试](./07-performance-benchmarks.md) | 不同方法的性能对比和分析 |

### 📖 参考篇

| 文档 | 描述 |
|------|------|
| [API 参考手册](./api-reference.md) | 完整的 API 函数和类参考 |

## 学习路径建议

### 初学者路径

1. **01. 大模型推理基础** → 建立基础概念
2. **02. Speculative Decoding 理论** → 理解核心算法
3. **03. Speculative Decoding 实现** → 看代码实现
4. **06. 技巧与优化** → 了解实践要点

### 进阶路径

5. **04. NASD 理论** → 理解创新方法
6. **05. NASD 实现** → NASD 代码实现
7. **07. 性能基准测试** → 性能对比分析

### 实践者路径

直接跳转到 **API 参考手册** 开始使用代码。

## 核心概念速览

### 什么是 Speculative Decoding？

Speculative Decoding（投机解码）是一种利用小模型加速大模型推理的技术：

- 使用小模型（drafter）快速生成多个候选 token（γ 个）
- 大模型（target）并行验证这些候选
- 通过拒绝采样保证输出分布与目标模型一致
- 典型加速比：1.5x - 3x

### 什么是 NASD？

**Ngram Assisted Speculative Decoding (NASD)** 是本项目的创新：

- 用统计 n-gram 模型替代神经网络 drafter
- 无需训练，完全训练无关（training-free）
- 与标准 Speculative Decoding 相当或更好的加速效果

## 关键参数

| 参数 | 符号 | 说明 | 典型值 |
|------|------|------|--------|
| 草稿数量 | γ | 每步生成的候选 token 数 | 2-8 |
| 接受率 | α | 被接受的草稿比例 | 0.6-0.8 |
| Ngram 阶数 | n | NASD 中最大 n-gram 阶数 | 3-5 |
| Top-K Filler | - | NASD 更新时考虑的候选数 | 1-10 |

## 项目结构

```
Speculative-Decoding/
├── sampling/              # 核心生成算法
│   ├── base_decoding.py              # 自回归 + Beam Search
│   ├── speculative_decoding.py       # Speculative Decoding
│   ├── codec_base_decoding.py        # Encoder-Decoder 版本
│   └── codec_speculative_decoding.py # Encoder-Decoder Speculative
│
├── ngram_assisted/        # NASD 实现
│   ├── ngram_assisted.py              # NASD 生成
│   └── ngram_storage.py              # Ngram 存储
│
├── utils/                 # 工具模块
│   ├── logits_processor.py            # 采样策略
│   ├── caching.py                    # KV Cache 管理
│   └── printing.py                   # 调试可视化
│
├── infer.py              # 交互式 CLI
└── docs/                 # 本文档目录
```

## 快速开始

```bash
# 安装依赖
uv sync

# 运行交互式推理
uv run infer.py
```

在 CLI 中：
- 输入文本让模型生成
- 输入 `/help` 查看所有命令
- 输入 `/quit` 退出

## 参考资料

1. **Leviathan et al., 2023** - Fast Inference from Transformers via Speculative Decoding
2. **Chen et al., 2023** - Accelerating Large Language Model Decoding with Speculative Sampling
3. **Ou et al., 2024** - Lossless Acceleration of Large Language Model via Adaptive N-gram Parallel Decoding (NAPD)

## 贡献

欢迎提交 Issue 和 Pull Request！

---

**项目作者**: [Romain](https://github.com/romain)

**文档版本**: 1.0
