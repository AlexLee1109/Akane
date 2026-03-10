import numpy as np
import os
import torch


def load_tokens(filename: str) -> torch.Tensor:
    npt = np.load(filename).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoader:
    def __init__(self, B: int, T: int, split: str):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        data_root = "/content/drive/Shareddrives/Fineweb20B"
        shards     = sorted(s for s in os.listdir(data_root) if split in s)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, f"no shards found for split {split}"
        print(f"found {len(self.shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard    = 0
        self.current_position = 0
        self.tokens           = load_tokens(self.shards[0])

    def set_state(self, shard: int, position: int):
        """Restore exact dataloader state from checkpoint."""
        self.current_shard    = shard
        self.current_position = position
        self.tokens           = load_tokens(self.shards[shard])

    def next_batch(self):
        B, T  = self.B, self.T
        pos   = self.current_position
        buf   = self.tokens[pos : pos + B * T + 1]
        x     = buf[:-1].view(B, T)
        y     = buf[1:].view(B, T)

        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_shard    = (self.current_shard + 1) % len(self.shards)
            self.tokens           = load_tokens(self.shards[self.current_shard])
            self.current_position = 0

        return x, y