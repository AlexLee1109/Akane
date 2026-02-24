# Akane: AI VTuber Chatbot

Akane is an AI VTuber companion chatbot powered by a custom GPT-style transformer, built from scratch in PyTorch. Akane delivers cheerful, expressive conversations with real-time streaming responses in the terminal. Under the hood, it features Grouped Query Attention (GQA) with QK normalization, shared rotary positional embeddings (RoPE), SwiGLU MLP, and pre-allocated KV caching for fast autoregressive inference.

## Features
- **Streaming Chat Interface**: A rich terminal UI (via `rich`) with live-updating response panels, token-per-second stats, and system info display.
- **Custom Transformer Architecture**: A modular GPT-2-inspired model built for clarity, extensibility, and efficient on-device inference.
- **Grouped Query Attention (GQA) with QK Normalization**: Multi-head attention with fewer KV heads than query heads, plus RMSNorm on Q and K for training stability.
- **Shared Rotary Positional Embeddings (RoPE)**: A single `RotaryEmbedding` instance shared across all layers, applying precomputed cosine/sine rotations within self-attention.
- **Pre-Allocated KV Cache**: A dedicated `KVCache` class pre-allocates contiguous memory for all layers upfront, enabling zero-allocation autoregressive inference via in-place writes and sliced views.
- **SwiGLU MLP**: A gated feed-forward network using SiLU activation (`SiLU(W1·x) * W2·x`) with a ≈2.67x hidden expansion ratio, rounded to the nearest multiple of 256.
- **Zero-Initialized Residual Projections**: Output projections in both attention and MLP blocks are initialized to zero for stable early training.
- **Streaming Text Generation**: Autoregressive generation with KV caching, temperature scaling, top-k sampling, and token-level repetition penalty, yielded as a Python generator for real-time display.

## Table of Contents
- [Overview](#overview)
- [Chat Interface](#chat-interface)
- [Model Architecture](#model-architecture)
- [KV Cache](#kv-cache)
- [Generation](#generation)
- [Training & Fine-Tuning](#training--fine-tuning)
- [Requirements](#requirements)

## Overview
Akane is an AI VTuber chatbot built on a custom transformer trained from scratch on FineWeb and fine-tuned on Hololive VTuber transcript data. It incorporates modern techniques like GQA, RoPE, SwiGLU, and pre-allocated KV caching for responsive, on-device conversation.

### Key Concepts
- **Grouped Query Attention (GQA)**: Uses fewer KV heads than query heads (e.g., 12 query heads, 6 KV heads) for memory-efficient attention via PyTorch's `scaled_dot_product_attention` with `enable_gqa=True`.
- **Pre-Allocated KV Cache**: Eliminates per-step memory allocation during generation by writing into a fixed, pre-allocated tensor.

## Chat Interface
Run `main.py` to start an interactive chat session with Akane:

```bash
python main.py
```

The chat interface features:
- **Live streaming responses** rendered in a bordered panel as tokens are generated.
- **Performance stats** after each response (token count, tokens/sec, elapsed time).
- **System info display** showing device and model parameter count on startup.
- **MPS / CUDA support** with `torch.compile` for optimized inference.

Type `quit`, `exit`, or `q` to end the session.

## Model Architecture
Akane is implemented in Python using **PyTorch**. The model consists of the following components:

1. **Embedding Layer**:
   - Token embedding (`wte`): Maps vocabulary tokens to dense representations (default: 50,304 tokens → 768 dimensions).
   - Embedding normalization (`emb_norm`): RMSNorm applied directly to token embeddings before the transformer stack.
   - No explicit positional embeddings; instead, RoPE is applied within the attention mechanism.

2. **Transformer Blocks**: Each block (default: 12 layers) contains:
   - **Causal Self-Attention**: Separate Q, K, V linear projections (all bias-free). Query heads (default: 12) and KV heads (default: 6) enable Grouped Query Attention. RoPE is applied to Q and K, followed by QK normalization via per-head RMSNorm. The output projection is zero-initialized.
   - **SwiGLU MLP**: A gated feed-forward network with `≈2.67x` hidden expansion (e.g., 768 → 2048, rounded to nearest 256). Uses `SiLU(W1·x) * W2·x` followed by a zero-initialized output projection.
   - **RMS Normalization**: Applied before both the attention and MLP sub-layers (pre-norm architecture).

3. **Language Modeling Head**: A bias-free linear layer maps the final RMSNorm'd transformer outputs to vocabulary logits.

4. **Shared Rotary Positional Embeddings (RoPE)**: A single `RotaryEmbedding` module is instantiated once in the `GPT` class and passed by reference to every attention layer. It precomputes cosine and sine embeddings for the full context length (default: 1,024 positions) in bfloat16 precision with a default theta of 10,000.

## KV Cache
The `KVCache` class provides memory-efficient caching for autoregressive inference:

- **Pre-allocated contiguous memory**: A single tensor of shape `(n_layer, batch, n_kv_head, max_seq_len, head_dim)` is allocated once for both keys and values, eliminating per-step allocations.
- **In-place writes**: New KV pairs are written directly into the pre-allocated buffer via slice assignment — no copies or concatenation.
- **Sliced views**: Returns views (not copies) of the cached KV pairs up to the current sequence position.
- **Sequence tracking**: Automatically tracks the current sequence length and updates it after the final layer processes each step.
- **Factory method**: `KVCache.from_config(config, ...)` constructs a cache directly from a `GPTConfig` instance.
- **Uses `__slots__`** for minimal memory overhead on the Python object itself.

## Generation
The `GPT.generate()` method supports streaming autoregressive generation:

- **KV-cached inference**: Primes the cache with the full prompt in one forward pass, then generates one token at a time.
- **Temperature scaling**: Logits are scaled by `1 / temperature` before sampling.
- **Top-k sampling**: Only the top-k candidate tokens are considered at each step.
- **Repetition penalty**: Tokens that have already been generated are penalized (positive logits divided, negative logits multiplied by the penalty factor).
- **Streaming output**: Tokens are yielded one at a time via a Python generator, enabling real-time display.

## Requirements
To run this project, you need:
- Python 3.10+
- PyTorch 2.0+
- `tiktoken` tokenizer library
- `rich` library (for the chat interface)

Install dependencies:
```bash
pip install torch tiktoken rich
```

## Project Structure
```
Akane/
├── main.py                   # Chat interface entry point
├── Akane/
│   ├── gpt.py                # GPT model, attention, MLP, RoPE
│   └── kv_cache.py           # Pre-allocated KV cache
└── Train/
    ├── train.py              # Pre-training loop
    └── dataloader.py         # FineWeb data loader
```

## Future Implementations
Planned enhancements for Akane include:
- **Reinforcement Learning with Human Feedback (RLHF)**
   - Integrate RLHF to refine model responses using human feedback, improving alignment with user expectations.
- **Proximal Policy Optimization (PPO)**
   - Implement PPO to optimize the model's policy, enhancing decision-making and response quality.
