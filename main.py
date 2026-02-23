import os
import gc
import time

import torch
import tiktoken
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box

from Akane.gpt import GPT, GPTConfig

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    torch.set_float32_matmul_precision('high')

CHECKPOINT_PATH = "Models/arcanev2.pt"
EXIT_COMMANDS = {"quit", "exit", "q"}

MODEL_CONFIG = GPTConfig(
    n_layer=36,
    n_head=16,
    n_kv_head=4,
    n_embd=1536
)

MAX_TOKENS = 100
TEMPERATURE = 0.85
TOP_K = 40
REPETITION_PENALTY = 1.08

SYSTEM = "You are Akane, a cheerful and expressive AI Vtuber companion.\n"

# ── Helpers ─────────────────────────────────────────────────────────────────
def _free_memory():
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

# ── Model loading for M3 MPS ────────────────────────────────────────────────
def _load_model():
    model = GPT(MODEL_CONFIG)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    gc.collect()

    model = model.to(DEVICE, dtype=torch.bfloat16).eval()
    model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=False)

    _free_memory()
    return model


class ChatBot:
    _PANEL_STYLE = dict(
        title="[bold green]Akane[/bold green]",
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 1),
    )

    def __init__(self, model, console):
        self.model = model
        self.console = console
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def _show_system_info(self):
        param_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        table = Table(title="[bold]System Info[/bold]", box=box.ROUNDED)
        table.add_column("Item", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Device", "MPS")
        table.add_row("Model Size", f"{param_m:.1f}M params")
        self.console.print(table)

    @torch.inference_mode()
    def _stream_tokens(self, prompt):
        tokens = self.tokenizer.encode(SYSTEM + prompt, allowed_special={"<|endoftext|>"})
        tokens = tokens[-MODEL_CONFIG.block_size:]

        byte_buf = b""
        for token_id in self.model.generate(
            tokens=tokens,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
        ):
            byte_buf += self.tokenizer.decode_single_token_bytes(token_id)
            try:
                text = byte_buf.decode("utf-8")
                byte_buf = b""
                yield text
            except UnicodeDecodeError:
                continue

    def _respond(self, prompt):
        t0 = time.perf_counter()
        n_tokens = 0
        chunks = []
        first = True

        with Live(Panel("", **self._PANEL_STYLE), console=self.console, refresh_per_second=30) as live:
            for chunk in self._stream_tokens(prompt):
                n_tokens += 1
                if first:
                    chunk = chunk.lstrip()
                    if not chunk:
                        continue
                    first = False
                chunks.append(chunk)
                live.update(Panel("".join(chunks), **self._PANEL_STYLE))

        elapsed = time.perf_counter() - t0
        tok_s = n_tokens / elapsed if elapsed > 0 else 0.0
        self.console.print(f"[dim]{n_tokens} tokens • {tok_s:.1f} tok/s • {elapsed:.1f}s[/dim]\n")

        _free_memory()

    def run(self):
        self._show_system_info()
        self.console.print("Type [bold]quit, exit or q[/bold] to exit\n", style="dim")

        while True:
            self.console.print("[bold cyan]You[/bold cyan] ", end="")
            user_input = input().strip()

            if user_input.lower() in EXIT_COMMANDS:
                self.console.print("\n[bold cyan]Akane has left the chat.[/bold cyan]\n")
                break
            if not user_input:
                continue

            self._respond(user_input)


def main():
    console = Console()

    with console.status("[bold green]Loading model...", spinner="dots"):
        model = _load_model()

    ChatBot(model, console).run()


if __name__ == "__main__":
    main()