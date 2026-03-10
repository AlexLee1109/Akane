import os
import time
import math
import torch
import gc
from finetune_dataloader import DataLoader, tokenizer, IGNORE_INDEX
from gpt import GPT, GPTConfig
from peft import get_peft_model, LoraConfig
from torch.optim.lr_scheduler import LambdaLR

# ============================================================
# Device & Stability
# ============================================================
device = "cuda" if torch.cuda.is_available() else "mps"
print(f"Using device: {device}")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('high')
print("✅ TF32 disabled + FP16 stability enabled")

# ============================================================
# Hyperparameters — tuned for fast early progress
# ============================================================
lr               = 4e-5          # ↑ increased (was 4e-5)
max_steps        = 10000
warmup_steps     = 500           # ↓ shortened dramatically (was 2000)
val_every        = 200           # more frequent early monitoring
val_loss_steps   = 200

total_batch_size = 32768
B, T             = 1, 1024       # ↑ B=4 → ~25% faster on T4, only 8 accum steps
grad_accum_steps = total_batch_size // (B * T)
grad_clip        = 1

print(f"Total batch size: {total_batch_size:,} | Micro-batch: {B}x{T} | Accum: {grad_accum_steps}")
print(f"Training for ~1B tokens → {max_steps:,} steps")

# ============================================================
# Data Loaders
# ============================================================
train_loader = DataLoader(B=B, T=T, split="train", buffer_size=400)
val_loader   = DataLoader(B=B, T=T, split="val",   buffer_size=200)

# ============================================================
# Model
# ============================================================
config = GPTConfig(n_layer=36, n_head=16, n_kv_head=4, n_embd=1536)
model  = GPT(config)

log_dir   = "/content/drive/MyDrive/log"
ckpt_path = os.path.join(log_dir, "modelv2_03500.pt")

print(f"Loading base model: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
model.load_state_dict(checkpoint["model"])
del checkpoint
gc.collect()

model = model.to(device)

# ============================================================
# LoRA
# ============================================================
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["c_q", "c_k", "c_v", "c_proj", "w1", "w2"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = torch.compile(model, dynamic=False)

optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=lr)
scaler    = torch.amp.GradScaler("cuda")

# ============================================================
# Scheduler
# ============================================================
def cosine_warmup_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=cosine_warmup_lambda)

# ============================================================
# Training Loop
# ============================================================
os.makedirs(log_dir, exist_ok=True)
log_file      = os.path.join(log_dir, "log_ft.txt")
start_time    = time.time()
best_val_loss = float('inf')

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # ---------- Validation ----------
    if step % val_every == 0 or last_step:
        model.eval()
        val_loss_accum = 0.0
        valid_batches  = 0

        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if (y != IGNORE_INDEX).sum().item() == 0:
                    continue

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, loss = model(x, y)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    val_loss_accum += loss.item()
                    valid_batches += 1

        if valid_batches > 0:
            avg_val_loss = val_loss_accum / valid_batches
            print(f"{step:05d} | val loss: {avg_val_loss:.4f}")

            with open(log_file, "a") as f:
                f.write(f"{step} val {avg_val_loss:.4f}\n")

        if last_step:
            merged = model.merge_and_unload()
            torch.save({"model": merged.state_dict()}, os.path.join(log_dir, "model_final.pt"))
            save_tokenizer(tokenizer, os.path.join(log_dir, "tokenizer"))
            print(f"✅ Final model + tokenizer saved")

    # ---------- Training ----------
    model.train()
    loss_accum = 0.0
    valid_micro = 0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if (y != IGNORE_INDEX).sum().item() == 0:
            continue   # ← no more noisy print

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _, loss_raw = model(x, y)

            if torch.isnan(loss_raw) or torch.isinf(loss_raw):
                continue

            loss_for_backward = loss_raw / grad_accum_steps
            loss_accum += loss_raw.detach()
            valid_micro += 1

        scaler.scale(loss_for_backward).backward()

    if valid_micro > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    else:
        scaler.update()

    optimizer.zero_grad(set_to_none=True)

    # ---------- Logging ----------
    dt = time.time() - t0
    avg_loss = loss_accum.item() / valid_micro if valid_micro > 0 else float('nan')
    tok_per_sec = (B * T * valid_micro) / dt if valid_micro > 0 else 0

    print(f"{step:05d} | train loss: {avg_loss:.6f} | {dt:.2f}s | {tok_per_sec:.0f} tok/s")
    with open(log_file, "a") as f:
        f.write(f"{step} train {avg_loss:.6f} dt:{dt:.2f}s tok/s:{tok_per_sec:.0f}\n")

    if last_step:
        merged = model.merge_and_unload()
        torch.save({"model": merged.state_dict()}, os.path.join(log_dir, f"model_step_{step:06d}.pt"))
        print(f"   → Checkpoint saved at step {step:,}")

elapsed = (time.time() - start_time) / 60
print(f"✅ Training finished (~1B tokens) in {elapsed:.1f} minutes.")