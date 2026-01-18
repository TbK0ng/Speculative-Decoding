# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of Speculative Decoding / Speculative Sampling for transformer models. The project provides three main generation strategies: autoregressive decoding, beam search, and speculative decoding. It also includes **Ngram Assisted Speculative Decoding (NASD)**, a novel approach that replaces the drafter model with statistical n-gram storage.

## Development Commands

### Running Inference

The primary entry point is the interactive CLI:

```bash
python infer.py
# With specific device:
python infer.py --device cuda
```

The CLI supports toggling between different generation modes, adjusting hyperparameters, and comparing outputs in real-time. Type `/help` in the CLI for all available commands.

### Dependency Management

The project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Add a dependency
uv add <package>
```

Python 3.13+ is required (see `.python-version`).

### No Test Suite

This codebase does not have automated tests. Validation is done manually through the `infer.py` CLI interface.

## Architecture Overview

### Core Generation Strategies

The codebase implements four main generation approaches:

1. **Autoregressive Generation** (`sampling/base_decoding.py`): Classic sequential token-by-token generation
2. **Beam Search** (`sampling/base_decoding.py`): Parallel exploration of multiple hypotheses with length penalty
3. **Speculative Decoding** (`sampling/speculative_decoding.py`): Two-model approach using a smaller "drafter" model
4. **Ngram Assisted Speculative Decoding** (`ngram_assisted/ngram_assisted.py`): Uses statistical n-gram models instead of a neural drafter

### Speculative Decoding Data Flow

The core speculative decoding process works as follows:

1. **Draft Generation**: Drafter model generates γ token predictions in parallel
2. **Target Verification**: Target model evaluates all drafts + one additional token
3. **Rejection Sampling**: Accept drafts probabilistically based on ratio p/q (target probability / drafter probability)
4. **Cache Management**: Maintain KV caches for both models with pruning support

Key insight: Transformer models can compute probability distributions for multiple tokens in parallel. This allows verifying γ draft tokens in a single forward pass of the target model.

### Ngram Assisted Speculative Decoding (NASD)

NASD replaces the neural drafter with a statistical n-gram model:

- **NGramStorage** (`ngram_assisted/ngram_storage.py`): Stores k-grams for k ∈ [2, N]
- For a given N-1 token context, returns the most probable next token
- Storage is initialized from the prompt and updated during generation
- Falls back to target model when context is unknown

### Key Abstractions

**LogitsProcessor** (`utils/logits_processor.py`): Abstract base for sampling strategies
- `GreedyProcessor`: Always selects most probable token
- `MultinomialProcessor`: Random sampling
- `TopKProcessor`: Samples from top-k tokens
- `NucleusProcessor`: Top-p (nucleus) sampling
- `TopKNucleusProcessor`: Combined top-k and top-p

**INgramStorage** (`ngram_assisted/ngram_storage.py`): Abstract interface for n-gram storage
- `NGramStorage`: Full multi-level n-gram storage
- `OneLevelNGramStorage`: Simplified single-level variant

### Module Structure

- `sampling/`: Core generation algorithms (decoder-only models)
  - `base_decoding.py`: Autoregressive and beam search
  - `speculative_decoding.py`: Speculative decoding with neural drafter
  - `codec_base_decoding.py`: Encoder-decoder autoregressive
  - `codec_speculative_decoding.py`: Encoder-decoder speculative
- `ngram_assisted/`: NASD implementation and n-gram storage
  - `ngram_assisted.py`: Main NASD generation function
  - `ngram_storage.py`: NGram storage implementations
- `utils/`: KV cache pruning, logits processors, and debug visualization (note: no `__init__.py`, imports use absolute paths)

## Model Requirements

For speculative decoding to work correctly:
- Target and drafter must share the same tokenizer
- Both models must output identical logits shapes
- Target model should be large enough to benefit from acceleration
- Drafter model should be fast enough to create a memory bottleneck

## Known Issues

### KV Cache Inconsistency

The cache feature is inconsistently implemented across HuggingFace models and can lead to incorrect results or errors. To disable caching, set `use_cache=False` in generation methods.

### Encoder-Decoder Models

Separate implementations exist for encoder-decoder models:
- `autoregressive_generate_encoder_decoder`
- `speculative_generate_encoder_decoder`

Use these instead of the decoder-only versions when working with encoder-decoder architectures.

## Working with the Codebase

### Adding a New Sampling Strategy

1. Create a new class inheriting from `LogitsProcessor` in `utils/logits_processor.py`
2. Implement the `__call__` method that takes logits and returns sampled token
3. Add it to the `processors` dict in `infer.py` for CLI access

### Hyperparameters

Key hyperparameters for speculative decoding:
- `gamma`: Number of drafts generated per step (typically 2-8)
- `acceptance_rate`: Fraction of drafts accepted (higher = faster)
- `max_gen_len`: Maximum tokens to generate

For NASD:
- `n`: Maximum n-gram order
- `top_k_filler`: Top-k for unknown context fallback (set to 1 for NAPD-style behavior)
- `stop_if_unknown`: Whether to stop drafting when context is unknown

## Import Patterns

The `utils/` directory has no `__init__.py`. Imports from utils use direct module paths:

```python
from utils.logits_processor import LogitsProcessor, GreedyProcessor
from utils.caching import prune_cache
import utils.printing as printing
```

Main generation functions are imported from `sampling/` and `ngram_assisted/`:

```python
from sampling import autoregressive_generate, speculative_generate
from ngram_assisted import ngram_assisted_speculative_generate, NGramStorage
```
