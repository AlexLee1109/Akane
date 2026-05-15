# Akane

Akane is a local-first AI companion app with a FastAPI backend, a built-in web UI, an optional desktop popup, and optional integrations (VS Code editor bridge + Discord bot).

The main entrypoint is:

```bash
python -m app [server|popup|discord]
```

## Quick start

### 1) Install dependencies

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip

# Core runtime
pip install fastapi uvicorn llama-cpp-python

# Popup (optional)
pip install pywebview pillow

# Discord (optional)
pip install discord.py aiohttp
```

Notes:

- The local model backend uses `llama-cpp-python` and expects a GGUF model file.
- On macOS, Akane will use MPS automatically when available.

### 2) Point Akane at a model

By default Akane looks for a GGUF at:

```text
models/Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf
```

Override with either:

- Environment variable: `AKANE_MODEL_PATH=/absolute/or/relative/path/to/model.gguf`
- Local secrets file: copy `app/secrets/local_secrets.py.example` to `app/secrets/local_secrets.py` and set `MODEL_PATH = "..."`

### 3) Run

Backend server + web UI:

```bash
python -m app server
```

Then open:

- `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

Desktop popup (macOS-focused; uses the server locally by default):

```bash
python -m app popup
```

Discord bot (forwards messages to a running server over HTTP):

```bash
python -m app discord
```

## Configuration

Most settings can be provided either via:

- Environment variables prefixed with `AKANE_` (for example: `AKANE_SERVER_PORT=8000`)
- `app/secrets/local_secrets.py` (see `app/secrets/local_secrets.py.example`)

Common settings:

- `AKANE_APP_MODE`: `popup` (default), `server`, or `discord`
- `AKANE_SERVER_HOST`: default `127.0.0.1`
- `AKANE_SERVER_PORT`: default `8000`
- `AKANE_POPUP_BACKEND_URL`: server URL used by the popup when not local
- `AKANE_DISCORD_BOT_TOKEN`: Discord bot token

Local model settings:

- `AKANE_MODEL_PATH`: path to your `.gguf`
- `AKANE_LLAMA_CONTEXT_WINDOW`, `AKANE_LLAMA_BATCH_SIZE`, `AKANE_LLAMA_GPU_LAYERS`: llama.cpp tuning knobs

Optional “coder” backend via OpenRouter:

- `OPENROUTER_API_KEY`
- `OPENROUTER_CODER_MODEL`

## VS Code integration

There is a VS Code extension in `integrations/akane-vscode-extension` that:

- Sends Akane editor context (active file, selection, diagnostics)
- Polls Akane for queued editor actions

Install into your normal VS Code:

```bash
python3 integrations/akane-vscode-extension/install_local.py
```

Then set the extension’s `akane.serverUrl` setting to your server (for local dev: `http://127.0.0.1:8000`).

## Linux service scripts (optional)

`start_akane_services.sh` and `stop_akane_services.sh` are convenience wrappers for systemd units named:

- `akane-server`
- `akane-discord`

They are intended for Linux deployment (they run `systemctl ...`).

## Repository structure

Key folders:

```text
app/                     # Current runtime (server, popup UI, integrations)
    core/                  # Prompt + generation pipeline, model manager
    integrations/          # Discord bot, editor bridge, VS Code launcher
    ui/                    # Popup window + embedded static UI
akane/                   # Older/experimental transformer code
train/                   # Training scripts/dataloaders (optional/experimental)
integrations/            # VS Code extension project
```

## License

MIT
