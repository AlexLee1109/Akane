# Akane: A GPT-2 Inspired Model

Akane is a streamlined, educational GPT-style transformer model built in PyTorch, inspired by GPT-2. It offers a clear and efficient implementation to demonstrate modern transformer techniques, with a focus on performance and modularity. Key features include rotary positional embeddings (RoPE), RMS normalization, key-value (KV) caching for efficient inference, and a custom MLP design.

## Features
- **Compact Transformer Architecture**: A modular GPT-2-inspired model with a focus on clarity and extensibility.
- **RMS Normalization**: Implements a purely functional RMS normalization layer for stable and efficient training.
- **Rotary Positional Embeddings (RoPE)**: Enhances positional encoding within self-attention using precomputed rotary embeddings.
- **Causal Self-Attention with KV Caching**: Supports multi-head self-attention with key-value caching for faster inference.
- **Custom MLP Design**: Features a feed-forward network with ReLU-squared activation for improved non-linearity.
- **Optimized Weight Initialization**: Uses tailored initialization for linear and embedding layers to enhance training stability.
- **Custom Optimizer Configuration**: Employs AdamW with separate parameter groups for weight decay management.
- **Text Generation**: Supports temperature-based and top-k sampling for coherent sequence generation.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Model Comparison](#model-comparison)

## Overview
Akane is designed to help users understand transformer models through a clean, efficient implementation. It incorporates modern techniques like RoPE and KV caching, making it suitable for both educational exploration and practical experimentation.

### Key Concepts
- **Transformer Blocks**: Stacked layers with causal self-attention and MLP components.
- **Multi-Head Self-Attention (MHSA)**: Implements attention with support for Grouped Query Attention (GQA).
- **KV Caching**: Optimizes inference by storing key-value pairs for attention computations.

## Model Architecture
Arcane is implemented in Python using **PyTorch**. The model consists of the following components:

1. **Embedding Layer**:
   - Token embedding (`wte`): Maps vocabulary tokens to dense representations (default: 50,304 tokens to 768 dimensions).
   - No explicit positional embeddings; instead, RoPE is applied within the attention mechanism.

2. **Transformer Blocks**: Each block (default: 12 layers) contains:
   - **Causal Self-Attention**: Multi-head attention (default: 6 heads) with optional Grouped Query Attention, KV caching, and RoPE for positional encoding.
   - **MLP Layer**: A feed-forward network with a 4x expansion (768 to 3072 dimensions) and ReLU-squared activation.
   - **RMS Normalization**: Applied before attention and MLP layers, with no learnable parameters.

3. **Language Modeling Head**: A linear layer maps transformer outputs to token logits, with a soft-capping mechanism (tanh with a cap of 15) for numerical stability.

4. **Rotary Positional Embeddings (RoPE)**: Precomputes cosine and sine embeddings for up to 10x the sequence length (default: 10,240) in bfloat16 precision.

5. **KV Caching**: Supports efficient inference by reusing key-value pairs across sequences, with dynamic attention masking for context-aware generation.

## Requirements
To run this project, you need:
- Python 3.10+
- PyTorch 2.0+
- `tiktoken` tokenizer library (optional, for tokenization)

Install dependencies:
```bash
pip install torch tiktoken
```

## Future Implementations
Planned enhancements for Arcane include:
- **Reinforcement Learning with Human Feedback (RLHF)**
   - Integrate RLHF to refine model responses using human feedback, improving alignment with user expectations.
- **Proximal Policy Optimization (PPO)**
   - Implement PPO to optimize the model's policy, enhancing decision-making and response quality.
