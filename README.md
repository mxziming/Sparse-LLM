# SparseLLM
## Sparse Attention Inference for Long-Context LLMs

SparseLLM is an experimental implementation of **DeepSeek-style Dynamic Sparse Attention (DSA)** on top of **Qwen2.5**, covering the complete workflow from **Indexer training** to **sparse inference** and **break-even benchmarking**.

The goal of this project is to investigate whether sparse attention can accelerate **long-context autoregressive decoding**, especially for multi-turn Agent applications where conversation history continuously grows.

---

## Features

- ✅ Dynamic Sparse Attention (DSA)
- ✅ Learnable Indexer
- ✅ Two-stage knowledge distillation training
- ✅ Top-K sparse decoding
- ✅ Chunked prefill for long contexts
- ✅ Dense vs Sparse break-even benchmark
- ✅ HuggingFace compatible implementation

---

## Project Structure

```
.
├── model.py          # Modified Qwen2.5 implementation with DSA
├── warmup.py         # Stage 1: Indexer warmup
├── train.py          # Stage 2: Joint training
├── eval.py           # Sparse/Dense benchmark
├── dataset.py
├── data_process.py   # Data pre-process
├── load_config.py
└── README.md
```

---

## Dataset

We use WrtingPrompt (https://huggingface.co/datasets/euclaise/writingprompts) as our training and evaluation dataset. You can choose your own dataset, which won't affect the results.

---

## Method Overview

### 1. Learnable Indexer

Instead of computing attention over every historical token, SparseLLM learns an **Indexer** that predicts which previous tokens are important.

The Indexer consists of three components:

- Shared Key Projection
- Head-wise Importance Scoring
- Top-K Token Selection

During decoding, only the selected Top-K keys and values participate in attention computation.

### 2. Two-stage Training

#### Stage 1 — Warmup

Only the Indexer is trained.

- Freeze the backbone LLM
- Optimize the Indexer
- Full-distribution KL distillation
- Even-layer distillation

This allows the Indexer to learn meaningful scores before joint optimization.

#### Stage 2 — Joint Training

The entire model is fine-tuned.

- Unfreeze all parameters
- Dual learning rates
  - Backbone: small LR
  - Indexer: large LR
- Continue KL distillation
- Language modeling loss + KL loss

### 3. Sparse Decoding

During autoregressive generation,

```
Query
        |
Indexer predicts Top-K tokens
        |
Gather Key/Value
        |
Sparse Attention
```

Instead of attending to all previous tokens, attention is computed only over Top-K selected tokens.

### 4. Chunked Prefill

Long-context prefilling may exceed GPU memory.

SparseLLM supports chunked prefilling by

- constructing KV Cache incrementally
- constructing Indexer cache incrementally
- keeping caches consistent with sparse decoding

This significantly reduces peak GPU memory during prefill.

---

## Benchmark

The repository includes a **break-even benchmark**.

Instead of comparing two different models, a **single model instance** switches between

- Dense Decode
- Sparse Decode

This isolates the computational overhead introduced by sparse attention itself.

The benchmark

- fixes random seeds
- shares identical inputs
- measures only decode latency

---

## Training

Stage 1

```bash
python warmup.py
```

Stage 2

```bash
python train.py
```

## Evaluation

```bash
python eval.py
```

The benchmark reports

- decode latency
- throughput
- dense vs sparse comparison
- theoretical Top-K ratio

---

## Complexity Clarification

Although SparseLLM performs attention over only Top-K selected historical tokens during decoding, the current implementation does **not** achieve strict `O(K)` end-to-end decode complexity because the Indexer still needs to score candidate historical tokens before selecting Top-K. In practice, sparse decoding becomes beneficial only when the saved attention cost is larger than the overhead introduced by Indexer scoring, Top-K selection, and KV gather operations. This is why the benchmark reports a break-even point instead of assuming ideal O(K) speedup.

## Experimental Results

Experiments were conducted on context lengths ranging from

```
1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K
```

Observations:

- Sparse decoding is slower for short contexts due to Indexer and gather overhead.
- Around **32K context**, sparse decoding reaches the break-even point.
- At **64K+**, sparse decoding becomes faster than dense decoding.
- Up to **1.06× decode speedup** is achieved.

---


## Citation

This project is an educational/research implementation inspired by the Dynamic Sparse Attention (DSA) design described by DeepSeek.

It is **not** an official implementation.

---

## License

MIT
