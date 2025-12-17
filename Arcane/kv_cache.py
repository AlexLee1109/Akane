class KVCache:
    """
    Per-layer key/value cache for autoregressive decoding.

    Stores concatenated K/V tensors for each transformer layer to
    avoid recomputation during incremental generation.
    """

    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.caches = [None for _ in range(n_layers)]

    def get(self, layer_idx):
        return self.caches[layer_idx]

    def update(self, layer_idx, kv):
        self.caches[layer_idx] = kv

    def reset(self):
        for i in range(self.n_layers):
            self.caches[i] = None

    def is_empty(self):
        return all(cache is None for cache in self.caches)

    def to(self, device):
        """
        Move all cached tensors to the specified device.
        Useful when the model is moved after prefill.
        """
        for i, cache in enumerate(self.caches):
            if cache is not None:
                k, v = cache
                self.caches[i] = (k.to(device), v.to(device))
        return self
