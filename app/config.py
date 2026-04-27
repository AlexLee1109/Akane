import os
from urllib.parse import urlsplit

try:
    import torch
except ImportError:  # pragma: no cover - optional for API-only usage
    torch = None

try:
    from app import local_secrets as _local_secrets  # type: ignore
except ImportError:  # pragma: no cover - optional local config
    _local_secrets = None


def _local_secret(name: str, default):
    if _local_secrets is None:
        return default
    return getattr(_local_secrets, name, default)


def _secret_or_env(name: str, default: str = "") -> str:
    env_name = f"AKANE_{name}"
    return os.environ.get(env_name, str(_local_secret(name, default) or "")).strip()

DEVICE = "mps" if torch and torch.backends.mps.is_available() else "cpu"

APP_MODE = (_secret_or_env("APP_MODE", "popup") or "popup").lower()
SERVER_HOST = _secret_or_env("SERVER_HOST", "127.0.0.1") or "127.0.0.1"
SERVER_PORT = int((_secret_or_env("SERVER_PORT", "8000") or "8000"))
POPUP_BACKEND_URL = (
    _secret_or_env("POPUP_BACKEND_URL", "http://192.168.1.199:8000")
    or "http://192.168.1.199:8000"
).rstrip("/")


def popup_backend_host() -> str:
    parsed = urlsplit(POPUP_BACKEND_URL)
    return (parsed.hostname or "").strip().lower()


def popup_backend_is_local() -> bool:
    return popup_backend_host() in {"", "127.0.0.1", "localhost", "0.0.0.0", "::1"}

MODEL_PATH = os.environ.get(
    "AKANE_MODEL_PATH",
    _local_secret("MODEL_PATH", "") or "models/Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf",
).strip()
LLAMA_CONTEXT_WINDOW = int(
    os.environ.get(
        "AKANE_LLAMA_CONTEXT_WINDOW",
        str(_local_secret("LLAMA_CONTEXT_WINDOW", 4096)),
    ).strip()
)
LLAMA_BATCH_SIZE = int(
    os.environ.get(
        "AKANE_LLAMA_BATCH_SIZE",
        str(_local_secret("LLAMA_BATCH_SIZE", 64)),
    ).strip()
)
LLAMA_UBATCH_SIZE = int(
    os.environ.get(
        "AKANE_LLAMA_UBATCH_SIZE",
        str(_local_secret("LLAMA_UBATCH_SIZE", 32)),
    ).strip()
)
LLAMA_THREADS = int(
    os.environ.get(
        "AKANE_LLAMA_THREADS",
        str(_local_secret("LLAMA_THREADS", max(1, (os.cpu_count() or 4) - 2))),
    ).strip()
)
LLAMA_FLASH_ATTN = os.environ.get(
    "AKANE_LLAMA_FLASH_ATTN",
    "1" if _local_secret("LLAMA_FLASH_ATTN", True) else "0",
).strip().lower() in {"1", "true", "yes", "on"}
LLAMA_GPU_LAYERS = int(
    os.environ.get(
        "AKANE_LLAMA_GPU_LAYERS",
        str(_local_secret("LLAMA_GPU_LAYERS", -1 if DEVICE == "mps" else 0)),
    ).strip()
)
LLAMA_IDLE_UNLOAD_SECONDS = float(
    os.environ.get(
        "AKANE_LLAMA_IDLE_UNLOAD_SECONDS",
        str(_local_secret("LLAMA_IDLE_UNLOAD_SECONDS", 0)),
    ).strip()
)
CHAT_HISTORY_CONTEXT_TOKENS = int(
    os.environ.get(
        "AKANE_CHAT_HISTORY_CONTEXT_TOKENS",
        str(_local_secret("CHAT_HISTORY_CONTEXT_TOKENS", 3000)),
    ).strip()
)
CODER_MAX_TURNS = int(
    os.environ.get(
        "AKANE_CODER_MAX_TURNS",
        str(_local_secret("CODER_MAX_TURNS", 3)),
    ).strip()
)
CODER_MAX_INITIAL_TARGETS = int(
    os.environ.get(
        "AKANE_CODER_MAX_INITIAL_TARGETS",
        str(_local_secret("CODER_MAX_INITIAL_TARGETS", 2)),
    ).strip()
)
CODER_READ_CHUNK_LINES = int(
    os.environ.get(
        "AKANE_CODER_READ_CHUNK_LINES",
        str(_local_secret("CODER_READ_CHUNK_LINES", 220)),
    ).strip()
)
CODER_INITIAL_CHUNKS_PER_FILE = int(
    os.environ.get(
        "AKANE_CODER_INITIAL_CHUNKS_PER_FILE",
        str(_local_secret("CODER_INITIAL_CHUNKS_PER_FILE", 2)),
    ).strip()
)
CODER_MAX_READ_CHUNKS_PER_FILE = int(
    os.environ.get(
        "AKANE_CODER_MAX_READ_CHUNKS_PER_FILE",
        str(_local_secret("CODER_MAX_READ_CHUNKS_PER_FILE", 8)),
    ).strip()
)
STATIC_DIR = "static"
VSCODE_COMMAND = os.environ.get("AKANE_VSCODE_COMMAND", _local_secret("VSCODE_COMMAND", "")).strip()
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL",
    _local_secret("OPENROUTER_BASE_URL", "") or "https://openrouter.ai/api/v1",
).rstrip("/")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", _local_secret("OPENROUTER_API_KEY", "")).strip()
OPENROUTER_CODER_MODEL = os.environ.get(
    "OPENROUTER_CODER_MODEL",
    _local_secret("OPENROUTER_CODER_MODEL", "") or "",
).strip()
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", _local_secret("OPENROUTER_SITE_URL", "")).strip()
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", _local_secret("OPENROUTER_APP_NAME", "") or "Akane").strip()
OPENROUTER_CA_BUNDLE = os.environ.get("OPENROUTER_CA_BUNDLE", _local_secret("OPENROUTER_CA_BUNDLE", "")).strip()
OPENROUTER_SKIP_SSL_VERIFY = os.environ.get(
    "OPENROUTER_SKIP_SSL_VERIFY",
    "1" if _local_secret("OPENROUTER_SKIP_SSL_VERIFY", False) else "0",
).strip().lower() in {"1", "true", "yes", "on"}
ADVISOR_ONLY = os.environ.get(
    "AKANE_ADVISOR_ONLY",
    "1" if _local_secret("ADVISOR_ONLY", False) else "0",
).strip().lower() in {"1", "true", "yes", "on"}

# Generation parameters
MAX_TOKENS = 768
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.95
REPETITION_PENALTY = 1.1

# Chance (0.0-1.0) that Akane chimes in unprompted after a response
PROACTIVE_CHANCE = 0.05

# Seconds of idle before Akane might speak up on her own (0 = disable)
IDLE_INTERJECT_SECONDS = 120

# Relationship levels (determined dynamically by AI based on interaction quality + time)
# More progression-resistant with higher thresholds and slower accumulation
RELATIONSHIP_LEVELS = {
    "stranger": {"min_score": 0.0, "familiarity": 0.0},
    "acquaintance": {"min_score": 0.4, "familiarity": 0.3},
    "friend": {"min_score": 0.65, "familiarity": 0.6},
    "close_friend": {"min_score": 0.85, "familiarity": 1.0},
}

# Threshold for automatic relationship level check (score evaluation every N interactions)
RELATIONSHIP_CHECK_INTERVAL = 10

# System prompt is now in app/character.py — edit that file to change personality.

# Memory flush cadence (seconds) for background writes.
MEMORY_FLUSH_INTERVAL = float(os.environ.get("AKANE_MEMORY_FLUSH_INTERVAL", "15.0"))
