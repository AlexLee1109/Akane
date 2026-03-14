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
    def __init__(self, head_dim, max_pos, theta=10000.0):
        super().__init__()
        # Compute in float32 for precision, then store as bfloat16 to avoid
        # repeated dtype conversions during forward pass
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos().bfloat16(), persistent=False)
        self.register_buffer("sin", freqs.sin().bfloat16(), persistent=False)
        self._cos_cache = None
        self._sin_cache = None
        self._cache_pos = -1
        self._cache_T = -1

    def _get_cos_sin(self, pos, T, dtype):
        """Get sliced cos/sin, cached to avoid repeated slicing."""
        if self._cache_pos != pos or self._cache_T != T:
            self._cos_cache = self.cos[pos:pos + T].view(1, 1, T, -1)
            self._sin_cache = self.sin[pos:pos + T].view(1, 1, T, -1)
            self._cache_pos = pos
            self._cache_T = T
        # Only cast if caller needs different dtype (rare with bfloat16 model)
        if dtype != self.cos.dtype:
            return self._cos_cache.to(dtype), self._sin_cache.to(dtype)
        return self._cos_cache, self._sin_cache

    def rotate(self, x, pos):
        cos, sin = self._get_cos_sin(pos, x.size(2), x.dtype)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x[..., ::2] = x1 * cos - x2 * sin
        x[..., 1::2] = x1 * sin + x2 * cos
        return x

    def rotate_inline(self, q, k, pos):
        """Rotate both q and k in a single fused operation."""
        cos, sin = self._get_cos_sin(pos, q.size(2), q.dtype)
        q1, q2 = q[..., ::2], q[..., 1::2]
        q[..., ::2] = q1 * cos - q2 * sin
        q[..., 1::2] = q1 * sin + q2 * cos
        k1, k2 = k[..., ::2], k[..., 1::2]
        k[..., ::2] = k1 * cos - k2 * sin
        k[..., 1::2] = k1 * sin + k2 * cos
        return q, k


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

        # Project and reshape: Linear is contiguous, view works directly
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Fused rotary for q and k (single operation instead of two separate calls)
        q, k = self.rotary_emb.rotate_inline(q, k, pos)

        # Apply QK normalization (keep contiguous layout)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # KV cache handling for autoregressive generation
        is_causal = True
        if kv_cache is not None and layer_idx is not None:
            k, v = kv_cache.append(layer_idx, k, v)
            is_causal = T > 1  # Only causal during initial prefilling

        # Flash attention via scaled_dot_product_attention with GQA
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=0.0,  # Set via self.training check if needed
            enable_gqa=True,
        )

        # Reshape output: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.c_proj(out)


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

    def forward(self, idx: torch.Tensor, kv_cache=None, include_lm_head=True):
        pos = 0 if kv_cache is None else kv_cache.seq_len
        x = self.emb_norm(self.transformer.wte(idx))
        for i, block in enumerate(self.transformer.h):
            x = block(x, pos, kv_cache, layer_idx=i)
        if include_lm_head:
            x = self.transformer.ln_f(x)
            return self.lm_head(x)
        return x

    @torch.inference_mode()
    def generate(self, tokens, max_tokens=120, temperature=0.85, top_k=40, top_p=0.95, repetition_penalty=1.08, kv_cache=None):
        device = self.transformer.wte.weight.device
        model_dtype = self.transformer.wte.weight.dtype
        input_ids = torch.as_tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)
        prompt_len = input_ids.size(1)

        if kv_cache is None:
            kv_cache = KVCache.from_config(
                self.config,
                max_seq_len=prompt_len + max_tokens,
                batch_size=1,
                dtype=model_dtype,
                device=device,
            )

        if prompt_len > 1:
            self(input_ids[:, :-1], kv_cache)

        last_token = input_ids[:, -1:].contiguous()
        del input_ids

        k = min(max(top_k, 1), self.config.vocab_size)
        inv_temp = 1.0 / temperature
        use_rep_pen = repetition_penalty != 1.0
        seen = set()

        for _ in range(max_tokens):
            logits = self(last_token, kv_cache)[0, -1]

            # Repetition penalty
            if use_rep_pen and seen:
                idxs = torch.tensor(list(seen), device=device, dtype=torch.long)
                vals = logits[idxs]
                logits[idxs] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

            logits.mul_(inv_temp)

            # Top-k
            cand_logits, cand_idx = torch.topk(logits, k)

            # Top-p (nucleus) filtering
            if top_p is not None and 0 < top_p < 1:
                probs = F.softmax(cand_logits, dim=-1)
                cumprobs = probs.cumsum(dim=-1)
                # Keep tokens until cumulative probability exceeds top_p
                cutoff = torch.searchsorted(cumprobs, top_p, right=False).item()
                keep = min(cutoff + 1, k)
                probs = probs[:keep]
                probs.div_(probs.sum())
                cand_idx = cand_idx[:keep]
            else:
                probs = F.softmax(cand_logits, dim=-1)

            token_id = cand_idx[torch.multinomial(probs, 1)].item()
            last_token[0, 0] = token_id
            yield token_id

            if use_rep_pen:
                seen.add(token_id)

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