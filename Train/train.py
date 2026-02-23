import torch
import os
import time
from dataloader import DataLoader
from gpt import GPT, GPTConfig
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import math

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoader(B=B, T=T, split="train")
val_loader = DataLoader(B=B, T=T, split="val")

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

# create model
model = GPT(GPTConfig(n_layer=24, n_head=16, n_kv_head=4, n_embd=1344))
model.to(device)

lr = 3e-4 * 3
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
warmup_steps = 2000

def lr_lambda(step):
    if step < warmup_steps:
        # linear warmup: 0 → 1
        return step / warmup_steps
    else:
        # cosine decay: 1 → 0
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = torch.amp.GradScaler('cuda')

model = torch.compile(model, dynamic=False)

start_step = 0
checkpoint_path = os.path.join("log", "modelv2_0500.pt")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_step = checkpoint['step']
    train_loader.current_shard = checkpoint['current_shard']
    train_loader.current_position = checkpoint['current_position']

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"logv2.txt")
with open(log_file, "a") as f: # open for writing to clear the file
    pass

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 250
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            print(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 250 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"modelv2_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'current_shard': train_loader.current_shard,
                    'current_position': train_loader.current_position
                }
                torch.save(checkpoint, checkpoint_path)

    # do one step of the optimization
    model.train()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
        scaler.scale(loss).backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f} dt: {dt:.2f}s tok/sec: {tokens_per_sec:.2f}\n")
        print(f"{step} train {loss_accum.item():.6f} dt: {dt:.2f}s tok/sec: {tokens_per_sec:.2f}\n")

if ddp:
    destroy_process_group()