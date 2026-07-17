# Akane

Akane is a local-first AI VTuber runtime built around one persistent character,
one local language model, and multiple interfaces. The web chat, desktop popup,
Discord bot, and optional VS Code bridge all connect to the same bounded prompt,
memory, emotional-state, and generation pipeline.

Akane currently communicates through text. Speech, audio, Live2D control, and
avatar animation are planned directions, not current features.

## Highlights

- **Local inference** using a llama.cpp-compatible GGUF model
- **One shared model runtime** across web, popup, Discord, and developer tools
- **Persistent character identity** defined by `app/soul.md` and `app/identity.md`
- **Per-profile memory** with bounded conversation history and selective long-term recall
- **Emotional continuity** that changes from recent context without requiring a second model pass
- **Deterministic prompt budgeting** that stays inside the configured context window
- **Streaming responses** for the web and popup interfaces
- **Read-only VS Code context** with bounded workspace and file access
- **Raspberry Pi 5 defaults** tuned for low-memory local inference
- **Isolated profiles and conversations** so memory does not leak between users or interfaces

## Requirements

- Python 3.10 or newer
- An instruction-tuned GGUF model with an embedded chat template
- `llama-cpp-python` built for the target machine

## Quick start

Create a virtual environment and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Install the optional popup and Discord dependencies when needed:

```bash
python -m pip install -r requirements-optional.txt
```

The default model path is:

```text
models/gemma-4-E4B-it-Q4_K_M.gguf
```

Override it with `AKANE_MODEL_PATH`, or copy
`app/secrets/local_secrets.py.example` to `app/secrets/local_secrets.py` and set
`MODEL_PATH`. The local secrets file is ignored by Git.

## Running Akane

The main entry point is:

```bash
python -m app [server|popup|discord]
```

### Server and web chat

```bash
python -m app server
```

Open `http://127.0.0.1:8000`.

### Desktop popup

```bash
python -m app popup
```

The popup connects to the configured backend. For local convenience, it starts
an in-process server only when that backend is unavailable.

For a shared popup and Discord setup, start the server first so every interface
uses the same process and model instance.

### Discord

```bash
AKANE_DISCORD_BOT_TOKEN=... python -m app discord
```

The Discord process is a lightweight HTTP adapter and never loads the model.

### Linux services

`start_akane_services.sh` and `stop_akane_services.sh` wrap systemd units named
`akane-server` and `akane-discord`. Configure those units and their environment
before using the scripts.

## Chat commands

Commands apply only to the active profile or conversation:

- `/reset_chat` clears the current conversation history.
- `/clear` clears the current conversation history with a shorter notice.
- `/forget_me` deletes the profile's conversations, emotional context, and long-term memories.
- `/debug_state` shows qualitative state and prompt-size diagnostics without exposing prompt contents.

Discord guild commands require the configured bot prefix or a mention.

## How a response is generated

A normal turn follows one shared pipeline:

1. Validate and normalize the interface input.
2. Reserve a bounded generation slot for the conversation.
3. Retrieve recent history and relevant profile memories.
4. Preview emotional context from the current conversation.
5. Build one deterministic, token-budgeted chat request.
6. Generate one response with the local model.
7. Stream or deliver the completed response through the active interface.
8. Commit memory and emotional state only after successful generation.
9. Atomically persist schema-versioned state files.

There is no auxiliary analysis pass or duplicate-response regeneration. One
process-wide inference lock protects llama.cpp access, while the scheduler
bounds queued work and prevents concurrent turns from mutating the same
conversation.

See [`docs/architecture.md`](docs/architecture.md) for detailed ownership and
request-flow documentation.

## Identity and memory isolation

The popup uses the `local:owner` profile with the `popup:default` conversation.
Discord uses `discord:user:<id>` profiles.

Guild conversations are scoped by guild, channel, and user. Direct messages are
scoped by user. This prevents a public Discord channel from becoming one shared
memory bucket.

Popup and Discord identities are not linked automatically. Each profile owns
its own bounded history, emotional continuity, and long-term memories.

## Configuration

Settings can be supplied through `AKANE_*` environment variables or local-secret
attributes.

| Area | Important settings |
| --- | --- |
| Server | `AKANE_SERVER_HOST`, `AKANE_SERVER_PORT`, `AKANE_SERVER_API_TOKEN` |
| Clients | `AKANE_POPUP_BACKEND_URL`, `AKANE_DISCORD_SERVER_URL` |
| Discord | `AKANE_DISCORD_PREFIX`, `AKANE_DISCORD_ALLOWED_CHANNEL_IDS` |
| Model | `AKANE_MODEL_PATH`, `AKANE_LLAMA_CONTEXT_WINDOW`, `AKANE_LLAMA_GPU_LAYERS` |
| Performance | `AKANE_LLAMA_BATCH_SIZE`, `AKANE_LLAMA_UBATCH_SIZE`, `AKANE_LLAMA_THREADS`, `AKANE_LLAMA_THREADS_BATCH` |
| Attention and cache | `AKANE_LLAMA_FLASH_ATTN`, `AKANE_LLAMA_SWA_FULL`, `AKANE_LLAMA_WARMUP_STATIC_PROMPT` |
| Sampling | `AKANE_MAX_TOKENS`, `AKANE_TEMPERATURE`, `AKANE_TOP_K`, `AKANE_TOP_P`, `AKANE_MIN_P`, `AKANE_REPETITION_PENALTY` |
| Memory | `AKANE_MEMORY_CONTEXT_TOKENS`, `AKANE_MEMORY_MAX_RESULTS`, `AKANE_MEMORY_MIN_RELEVANCE`, `AKANE_MEMORY_MIN_SCORE` |
| Scheduling | `AKANE_MAX_PENDING_GENERATIONS`, `AKANE_GENERATION_QUEUE_TIMEOUT_SECONDS` |
| Diagnostics | `AKANE_PROMPT_DEBUG=1`, `AKANE_TIMING=1` |

Prompt construction reserves space for the reply and chat-template overhead, then
trims optional context to remain inside the configured context window.

## Security

The server binds to `127.0.0.1` by default.

Only bind to `0.0.0.0` on a trusted LAN. When remote local-network clients are
enabled, set a strong `AKANE_SERVER_API_TOKEN` and configure the same token in
the popup, Discord adapter, and VS Code extension.

Do not expose the API directly to the public internet.

For the built-in browser UI on a token-protected server, open:

```text
http://host:port/?api_token=...
```

The browser UI converts the query value into an authorization header for later
API requests.

## Raspberry Pi 5

Akane detects Raspberry Pi hardware and applies speed-oriented CPU defaults:

- 4,096-token context window
- 512-token batch and micro-batch
- Up to four CPU threads
- ARM flash attention
- Full sliding-window cache for Gemma
- Memory-mapped model loading
- Static character-prompt warmup
- 160-token response limit
- No GPU-layer offload by default

The static prompt is evaluated during background loading so later requests can
reuse its KV state.

For lower peak memory, start with:

```bash
AKANE_LLAMA_UBATCH_SIZE=128
```

or:

```bash
AKANE_LLAMA_SWA_FULL=0
```

Both reduce memory use at the cost of some prefill or prompt-reuse performance.
Tune threads and batch sizes before increasing the context window.

## Repository structure

```text
app/
  core/
    character.py      # Reloadable soul and identity definitions
    prompt.py         # Deterministic, token-budgeted prompt construction
    state.py          # Emotional context and persistence
    memory.py         # Conversation history and long-term memory
    session.py        # Input normalization, scheduling, preview, and commit
    model_loader.py   # llama.cpp model lifecycle and inference runtime
    reply_pipeline.py # Generation events and commit-after-success behavior
  integrations/
    discord_bot.py    # Discord filtering, identity mapping, and delivery
  ui/
    popup.py          # Desktop window and HTTP streaming bridge
    static/           # Built-in web and popup interface

integrations/
  akane-vscode-extension/ # Optional read-only VS Code context bridge

evaluations/
  character/          # Repeatable prompt and behavior evaluations
```

## VS Code context bridge

The optional extension in `integrations/akane-vscode-extension` gives Akane
bounded, read-only development context. It sends a limited workspace index and
only the safe file contents requested for the current turn.

See [`docs/vscode-local-model.md`](docs/vscode-local-model.md) for installation,
configuration, and testing instructions.

## Character evaluations

Run prompt-construction and behavior cases without loading the model:

```bash
python -m evaluations.character.run
```

Run the same evaluation harness with the configured GGUF:

```bash
python -m evaluations.character.run --generate
```

## Current scope

Akane currently includes local text generation, character prompting, memory,
emotional continuity, web and popup interfaces, Discord integration, and a
read-only VS Code bridge.

Planned directions include speech, Live2D avatar control, richer autonomous
behavior, and additional integrations. These are not part of the current
runtime.

## License

MIT
