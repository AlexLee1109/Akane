import os
import random
import torch
import tiktoken
from datasets import load_dataset

# ============================================================
# Tokenizer Setup (in-memory only - no disk save/load)
# ============================================================

def build_tokenizer():
    base = tiktoken.get_encoding("gpt2")

    custom_special_tokens = base._special_tokens.copy()
    custom_special_tokens["<|im_start|>"] = base.n_vocab      # 50257
    custom_special_tokens["<|im_end|>"]   = base.n_vocab + 1  # 50258

    enc = tiktoken.Encoding(
        name="gpt2_chatml",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=custom_special_tokens,
    )
    return enc


# Always build in memory - no files required
tokenizer = build_tokenizer()

IM_START      = tokenizer._special_tokens["<|im_start|>"]  # 50257
IM_END        = tokenizer._special_tokens["<|im_end|>"]    # 50258
EOT           = 50256                                       # <|endoftext|>
VOCAB_SIZE    = 50304                                       # model vocab size (padded if needed)
ASSISTANT_IDS = tokenizer.encode("assistant", allowed_special=set())
IGNORE_INDEX  = -100

print(f"Tokenizer ready (in-memory) | vocab size: {tokenizer.n_vocab}")
print(f"<|im_start|> id: {IM_START}")
print(f"<|im_end|> id  : {IM_END}")
print(f"assistant ids  : {ASSISTANT_IDS}")

# ============================================================
# ChatML Conversion
# ============================================================
def convert_ultrachat_to_chatml(example):
    text = ""
    for message in example["messages"]:
        role    = message["role"].strip()
        content = message["content"].strip()
        text   += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return text


def tokenize_and_mask(text):
    input_ids = tokenizer.encode(text, allowed_special={"<|im_start|>", "<|im_end|>"})

    # clean separator between conversations
    input_ids.append(EOT)

    labels = [IGNORE_INDEX] * len(input_ids)

    i = 0
    while i < len(input_ids):
        if input_ids[i] == IM_START:
            role_start = i + 1
            role_end   = role_start + len(ASSISTANT_IDS)
            if input_ids[role_start:role_end] == ASSISTANT_IDS:
                content_start = role_end
                j = content_start
                while j < len(input_ids) and input_ids[j] != IM_END:
                    j += 1
                for k in range(content_start, min(j + 1, len(input_ids))):
                    labels[k] = input_ids[k]
        i += 1

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels,    dtype=torch.long)
    )


# ============================================================
# DataLoader
# ============================================================
class DataLoader:
    def __init__(self, B, T, split, buffer_size=500):
        self.B           = B
        self.T           = T
        self.split       = split
        self.buffer_size = buffer_size
        assert split in {"train", "val"}

        print(f"Loading UltraChat 200k ({split})...")
        hf_split = "train_sft" if split == "train" else "test_sft"
        raw = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split=hf_split,
            cache_dir="./training_data/ultrachat_cache"
        )

        # random shuffle every run so bad examples dont land at same step
        raw = raw.shuffle()
        if split == "val":
            raw = raw.select(range(min(500, len(raw))))

        self.examples = raw
        print(f"  {len(self.examples):,} conversations loaded for split={split}")
        self.reset()

    def reset(self):
        self.example_idx      = 0
        self.current_position = 0
        self.input_ids        = torch.tensor([], dtype=torch.long)
        self.labels           = torch.tensor([], dtype=torch.long)
        self.examples         = self.examples.shuffle()  # re-shuffle on reset
        self._fill_buffer()

    def _fill_buffer(self):
        end         = min(self.example_idx + self.buffer_size, len(self.examples))
        new_ids     = []
        new_lbs     = []
        skipped     = 0

        for i in range(self.example_idx, end):
            text     = convert_ultrachat_to_chatml(self.examples[i])
            ids, lbs = tokenize_and_mask(text)

            # skip out of range token ids
            if ids.max().item() >= VOCAB_SIZE:
                skipped += 1
                continue

            # skip fully masked examples — nothing to learn from
            if (lbs != IGNORE_INDEX).sum().item() == 0:
                skipped += 1
                continue

            new_ids.append(ids)
            new_lbs.append(lbs)

        if skipped > 0:
            print(f"Skipped {skipped} bad examples in buffer")

        if new_ids:
            self.input_ids = torch.cat([self.input_ids, *new_ids])
            self.labels    = torch.cat([self.labels,    *new_lbs])

        self.example_idx = end

    def next_batch(self):
        B, T   = self.B, self.T
        needed = B * T + 1

        while (len(self.input_ids) - self.current_position) < needed:
            if self.example_idx >= len(self.examples):
                print("End of dataset, restarting...")
                self.examples    = self.examples.shuffle()
                self.example_idx = 0

            self.input_ids        = self.input_ids[self.current_position:]
            self.labels           = self.labels[self.current_position:]
            self.current_position = 0
            self._fill_buffer()

        x = self.input_ids[self.current_position     : self.current_position + B*T].view(B, T)
        y = self.labels   [self.current_position + 1 : self.current_position + B*T + 1].view(B, T)

        self.current_position += B * T
        return x, y