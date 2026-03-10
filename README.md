# Akane: Custom GPT-style Transformer

Akane is a PyTorch-based GPT-style transformer model, currently in the **pre-training phase** on 20B tokens from FineWeb. The project focuses on clean, performant implementation with modern transformer techniques. After pre-training completes, instruction fine-tuning will enable Akane to become a local AI VTuber companion.

## Key Features

- **Grouped Query Attention (GQA)** with QK normalization for training stability
- **Shared Rotary Positional Embeddings (RoPE)** across all layers
- **SwiGLU MLP** with zero-initialized output projections
- **Pre-allocated KV cache** for efficient autoregressive inference
- **Distributed training** support via PyTorch DDP
- **LoRA fine-tuning** infrastructure ready for instruction tuning

## Quick Start

### Pre-training

1. **Download FineWeb data** (20B tokens):
```bash
# Requires `datasets` library
python data/downloaders/fineweb.py
```
This creates shards in `data/processed/fineweb20B/` (~100M tokens each).

2. **Start pre-training**:
```bash
# Single GPU
torchrun train/pretrain.py

# Multi-GPU
torchrun --nproc_per_node=N train/pretrain.py
```

Checkpoints are saved to `logs/` (ignored by git). See `train/pretrain.py` to adjust:
- Model configuration (layers, heads, embedding dim)
- Batch size, learning rate, max steps
- Data directory path if needed

### Inference & Testing

After pre-training produces a checkpoint, convert it for inference:
```bash
python utils/convert_checkpoint.py
# Output: models/akane2.pt (or your chosen name)
```

Then test with interactive chat:
```bash
python -m cli.chat
```
Or run the web server:
```bash
python -m cli.server
```
(Note: `cli/` and `utils/` are local development directories not tracked by git)

## Repository Structure

Only core library and training scripts are version-controlled:

```
Akane/
├── akane/                    # Core model implementation
│   ├── gpt.py               # GPT model with GQA, RoPE, SwiGLU
│   ├── kv_cache.py          # Pre-allocated KV cache for generation
│   └── __init__.py
├── train/                   # Training scripts
│   ├── pretrain.py          # Main pre-training loop (DDP)
│   ├── finetune.py          # LoRA fine-tuning (for later phase)
│   └── dataloaders/
│       ├── fineweb.py       # FineWeb shard loader
│       ├── ultrachat.py     # UltraChat instruction data loader
│       └── __init__.py
├── README.md
└── LICENSE
```

**Local development files** (ignored by git, not tracked):
- `cli/` - Chat CLI and HTTP server
- `utils/` - Checkpoint conversion, plotting, data utilities
- `data/` - Raw data, processed shards, downloaders
- `models/` - Saved checkpoints
- `logs/` - Training logs
- `static/` - Web UI assets

## Model Architecture

### Components

1. **Token Embeddings**: GPT-2 vocabulary (50,304 tokens) with RMSNorm
2. **Transformer Blocks** (configurable count):
   - Causal self-attention with GQA (Grouped Query Attention)
   - Q/K normalization via RMSNorm
   - SwiGLU feed-forward network (≈2.67× expansion)
   - Pre-norm architecture (RMSNorm before each sub-layer)
3. **Output Head**: Linear projection to vocabulary logits

### Configuration

Edit `GPTConfig` in `akane/gpt.py` or pass parameters directly:

```python
config = GPTConfig(
    n_layer=36,      # Number of transformer blocks
    n_head=16,       # Number of query heads
    n_kv_head=4,     # Number of key/value heads (GQA)
    n_embd=1536,     # Embedding dimension
    block_size=1024, # Max sequence length
)
```

### KV Cache

During generation, a pre-allocated `KVCache` stores keys and values for all layers in a single contiguous tensor. This eliminates per-token allocation and enables fast, memory-efficient autoregressive decoding.

## Data Preparation

### FineWeb (Pre-training)

The `data/downloaders/fineweb.py` script streams the FineWeb dataset from HuggingFace and writes tokenized shards:

- Shard size: 100M tokens
- Output: `data/processed/fineweb20B/fineweb_train_*.npy` and `fineweb_val_000000.npy`
- Multiprocessing for fast tokenization
- Target: ~20B training tokens (199 shards)

The `train/dataloaders/fineweb.py` module loads these shards during training.

### UltraChat (Fine-tuning)

UltraChat 200k is loaded directly from HuggingFace by `train/dataloaders/ultrachat.py`. It uses ChatML formatting with special tokens `` and ` to mask non-assistant responses during training.

## Training Details

### Pre-training (`train/pretrain.py`)

- **Optimizer**: AdamW (fused, betas=(0.9, 0.95), weight decay=0.1)
- **Learning rate**: Cosine decay with warmup (adjustable)
- **Gradient accumulation**: Achieves effective batch size of 524,288 tokens
- **Mixed precision**: FP16 via autocast
- **Gradient clipping**: Norm 1.0
- **Checkpointing**: Every 250 steps (configurable), includes optimizer/scheduler state for resume
- **Validation**: Every 250 steps (250 batches)

### Fine-tuning (`train/finetune.py`)

- **Method**: LoRA (rank 32, alpha 64) on attention and MLP projections
- **Base model**: Load from pre-training checkpoint
- **Loss**: Cross-entropy with `IGNORE_INDEX` masking for non-assistant tokens
- **Merging**: LoRA weights merged into base model at end

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Additional libraries: `tiktoken`, `rich`, `datasets`, `tqdm`, `peft`

Install:
```bash
pip install torch tiktoken rich datasets tqdm peft
```

## Notes

- The repository intentionally tracks only core model and training code.
- Data, checkpoints, logs, and development utilities are excluded via `.gitignore` to keep the repo lightweight.
- When cloning, you'll need to recreate the local `cli/`, `utils/`, and `data/downloaders/` directories if you want to use them (they are not tracked).
- Adjust paths in config files if your data directories differ from defaults.

## License

MIT
