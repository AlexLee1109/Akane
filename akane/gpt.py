from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from akane.kv_cache import KVCache

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 6
    n_embd: int = 768
    mlp_ratio: float = 8 / 3
    rope_theta: float = 10000.0
    eps: float = 1e-6


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_pos, theta = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos().to(torch.bfloat16), persistent=False)
        self.register_buffer("sin", freqs.sin().to(torch.bfloat16), persistent=False)

    def rotate(self, x, pos):
        _, _, T, _ = x.shape
        cos = self.cos[pos:pos + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[pos:pos + T].unsqueeze(0).unsqueeze(0)

        x1 = x[..., ::2].clone()
        x2 = x[..., 1::2].clone()
        x[..., ::2] = x1 * cos - x2 * sin
        x[..., 1::2] = x1 * sin + x2 * cos
        return x 


class CausalSelfAttention(nn.Module):
    """Causal self-attention with GQA, shared RoPE, and QK normalization."""

    def __init__(self, config, rotary_emb):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.rotary_emb = rotary_emb  # shared, not owned

        # QKV projections
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # QK normalization for training stability
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.eps)

        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x: torch.Tensor, pos: int, kv_cache=None, layer_idx=None):
        B, T, _ = x.shape

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = self.rotary_emb.rotate(q, pos)
        k = self.rotary_emb.rotate(k, pos)

        q = self.q_norm(q)
        k = self.k_norm(k)

        is_causal = True
        if kv_cache is not None and layer_idx is not None:
            k, v = kv_cache.append(layer_idx, k, v)
            is_causal = T > 1

        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=True)
        return self.c_proj(out.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = int(config.mlp_ratio * config.n_embd)
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(self, config: GPTConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=config.eps)
        self.attn = CausalSelfAttention(config, rotary_emb)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=config.eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, pos: int, kv_cache=None, layer_idx=None):
        x = x + self.attn(self.ln_1(x), pos, kv_cache, layer_idx)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT model with a single shared RoPE embedding across all layers."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # One RoPE instance shared by every attention layer
        head_dim = config.n_embd // config.n_head
        self.rotary_emb = RotaryEmbedding(head_dim, config.block_size, config.rope_theta)

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, self.rotary_emb) for _ in range(config.n_layer)]),
            "ln_f": nn.RMSNorm(config.n_embd, eps=config.eps),
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.emb_norm = nn.RMSNorm(config.n_embd, eps=config.eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, kv_cache=None):
        pos = 0 if kv_cache is None else kv_cache.seq_len
        x = self.emb_norm(self.transformer.wte(idx))
        for i, block in enumerate(self.transformer.h):
            x = block(x, pos, kv_cache, layer_idx=i)
        return self.lm_head(self.transformer.ln_f(x))

    @torch.inference_mode()
    def generate(self, tokens, max_tokens=120, temperature=0.85, top_k=40, repetition_penalty=1.08):
        device = self.transformer.wte.weight.device
        input_ids = torch.as_tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)
        prompt_len = input_ids.size(1)

        kv_cache = KVCache.from_config(
            self.config,
            max_seq_len=prompt_len + max_tokens,
            batch_size=1,
            dtype=torch.bfloat16,
            device=device,
        )

        if prompt_len > 1:
            self(input_ids[:, :-1], kv_cache)   # prime cache

        last_token = input_ids[:, -1:]

        vocab_size = self.config.vocab_size
        k = min(max(top_k, 1), vocab_size)
        inv_temp = 1.0 / temperature
        use_rep_pen = repetition_penalty != 1.0

        seen = torch.zeros(vocab_size, dtype=torch.bool, device=device) if use_rep_pen else None

        for _ in range(max_tokens):
            logits = self(last_token, kv_cache)[0, -1]   # (vocab,)

            # Repetition penalty (only on seen tokens)
            if seen is not None:
                logits = torch.where(
                    seen,
                    torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty),
                    logits
                )

            logits = logits * inv_temp

            # Fast top-k + top-p on candidates only
            if k < vocab_size:
                candidate_logits, candidate_idx = torch.topk(logits, k)
                probs = candidate_logits.softmax(-1)
                next_idx = torch.multinomial(probs, 1)
                token_id = candidate_idx[next_idx].item()
            else:
                probs = logits.softmax(-1)
                token_id = torch.multinomial(probs, 1).item()

            last_token = torch.tensor([[token_id]], device=device, dtype=torch.long)
            yield token_id

            if seen is not None:
                seen[token_id] = True

    def configure_optimizers(self, weight_decay, learning_rate, embedding_lr_scale=0.1):
        decay_params = []
        no_decay_params = []
        embedding_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'wte' in name or 'lm_head' in name:
                embedding_params.append(param)
            elif param.ndim <= 1 or name.endswith('.bias') or 'norm' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
            {'params': embedding_params, 'weight_decay': 0.0, 'lr': learning_rate * embedding_lr_scale}
        ], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)