import torch

class KVCache:
    """Memory-efficient KV cache with pre-allocation and FP16/FP8 support."""
    
    __slots__ = ('k_cache', 'v_cache', 'seq_len', 'n_layer', 'max_seq_len', 
                 'batch_size', 'n_kv_head', 'head_dim', 'dtype', 'device')
    
    def __init__(self, n_layer, max_seq_len=1024, batch_size=1, n_kv_head=6, 
                 head_dim=64, dtype=torch.float16, device='cuda'):
        self.n_layer = n_layer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.seq_len = 0
        
        # Pre-allocate contiguous memory for all layers
        # Shape: (n_layer, batch, n_kv_head, max_seq_len, head_dim)
        self.k_cache = torch.zeros(
            n_layer, batch_size, n_kv_head, max_seq_len, head_dim,
            dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            n_layer, batch_size, n_kv_head, max_seq_len, head_dim,
            dtype=dtype, device=device
        )
    
    def append(self, layer_idx, k, v):
        """Append new KV pairs to cache (in-place, no allocation)."""
        T = k.size(2)
        end_pos = self.seq_len + T
        
        # Direct write — k, v are already (B, H, T, D)
        self.k_cache[layer_idx, :, :, self.seq_len:end_pos, :] = k
        self.v_cache[layer_idx, :, :, self.seq_len:end_pos, :] = v
        
        # Update seq_len after last layer processes
        if layer_idx == self.n_layer - 1:
            self.seq_len = end_pos
        
        # Return sliced views — no copies, no transposes needed
        return (
            self.k_cache[layer_idx, :, :, :end_pos, :],
            self.v_cache[layer_idx, :, :, :end_pos, :]
        )
    
    def clear(self):
        """Reset cache for new sequence (no deallocation)."""
        self.seq_len = 0
    
    @classmethod
    def from_config(cls, config, max_seq_len=None, batch_size=1, dtype=torch.bfloat16, device='cuda'):
        """Create KVCache from GPTConfig."""
        return cls(
            n_layer=config.n_layer,
            max_seq_len=max_seq_len or config.block_size,
            batch_size=batch_size,
            n_kv_head=config.n_kv_head,
            head_dim=config.n_embd // config.n_head,
            dtype=dtype,
            device=device
        )