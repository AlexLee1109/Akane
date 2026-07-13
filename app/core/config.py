"""Small runtime configuration layer."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from urllib.parse import urlsplit

try:
    from app.secrets import local_secrets as _local_secrets  # type: ignore
except ImportError:  # pragma: no cover
    _local_secrets = None


def _local_secret(name: str, default):
    return getattr(_local_secrets, name, default) if _local_secrets is not None else default


def _secret_or_env(name: str, default: str = "") -> str:
    value = os.environ.get(f"AKANE_{name}", os.environ.get(name, _local_secret(name, default)))
    return str(default if value is None else value).strip()


def _coerce_int(value, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float(default)


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value or "").strip().lower()
    if not normalized:
        return bool(default)
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _int_secret(name: str, default: int) -> int:
    return _coerce_int(_secret_or_env(name, str(default)), default)


def _float_secret(name: str, default: float) -> float:
    return _coerce_float(_secret_or_env(name, str(default)), default)


def _bool_secret(name: str, default: bool = False) -> bool:
    return _coerce_bool(_secret_or_env(name, "1" if default else "0"), default)


def _csv_ints(name: str, default: str = "") -> tuple[int, ...]:
    values: list[int] = []
    for part in _secret_or_env(name, default).split(","):
        try:
            values.append(int(part.strip()))
        except ValueError:
            pass
    return tuple(values)


def _csv_strings(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(
        value
        for part in _secret_or_env(name, default).split(",")
        if (value := part.strip())
    )


def _is_raspberry_pi() -> bool:
    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64", "armv7l", "armv8l"}:
        return False
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            model = Path(path).read_text(encoding="utf-8", errors="ignore")
            if "raspberry pi" in model.lower():
                return True
        except OSError:
            pass
    return "raspberry" in platform.node().lower()


IS_RASPBERRY_PI = _bool_secret("RASPBERRY_PI", _is_raspberry_pi())
APP_MODE = (_secret_or_env("APP_MODE", "popup") or "popup").lower()
SERVER_HOST = _secret_or_env("SERVER_HOST", "127.0.0.1") or "127.0.0.1"
SERVER_PORT = _int_secret("SERVER_PORT", 8000)
POPUP_BACKEND_URL = _secret_or_env(
    "POPUP_BACKEND_URL",
    f"http://127.0.0.1:{SERVER_PORT}",
).rstrip("/")
DISCORD_SERVER_URL = _secret_or_env(
    "DISCORD_SERVER_URL",
    f"http://127.0.0.1:{SERVER_PORT}",
).rstrip("/")
SERVER_API_TOKEN = _secret_or_env("SERVER_API_TOKEN", "")
CORS_ALLOWED_ORIGINS = _csv_strings(
    "CORS_ALLOWED_ORIGINS",
    f"http://127.0.0.1:{SERVER_PORT},http://localhost:{SERVER_PORT},null",
)

DISCORD_BOT_TOKEN = _secret_or_env("DISCORD_BOT_TOKEN", "")
DISCORD_PREFIX = _secret_or_env("DISCORD_PREFIX", "!akane")
DISCORD_ALLOWED_CHANNEL_IDS = _csv_ints("DISCORD_ALLOWED_CHANNEL_IDS", "")
DISCORD_REPLY_TO_DMS = _bool_secret("DISCORD_REPLY_TO_DMS", True)


def popup_backend_is_local() -> bool:
    hostname = (urlsplit(POPUP_BACKEND_URL).hostname or "").strip().lower()
    return hostname in {"", "127.0.0.1", "localhost", "0.0.0.0", "::1"}


MODEL_PATH = _secret_or_env("MODEL_PATH", "models/gemma-4-E4B-it-Q4_K_M.gguf")

_CPU_COUNT = os.cpu_count() or 4
_PI_THREADS = min(4, _CPU_COUNT)

LLAMA_CONTEXT_WINDOW = max(512, _int_secret("LLAMA_CONTEXT_WINDOW", 2048))
_DEFAULT_BATCH_SIZE = min(512 if IS_RASPBERRY_PI else 1024, LLAMA_CONTEXT_WINDOW)
LLAMA_BATCH_SIZE = max(
    1,
    min(_int_secret("LLAMA_BATCH_SIZE", _DEFAULT_BATCH_SIZE), LLAMA_CONTEXT_WINDOW),
)
_DEFAULT_UBATCH_SIZE = min(128 if IS_RASPBERRY_PI else 512, LLAMA_BATCH_SIZE)
LLAMA_UBATCH_SIZE = max(
    1,
    min(_int_secret("LLAMA_UBATCH_SIZE", _DEFAULT_UBATCH_SIZE), LLAMA_BATCH_SIZE),
)
LLAMA_THREADS = max(
    1,
    _int_secret(
        "LLAMA_THREADS",
        _PI_THREADS if IS_RASPBERRY_PI else max(1, _CPU_COUNT - 1),
    ),
)
LLAMA_THREADS_BATCH = max(1, _int_secret("LLAMA_THREADS_BATCH", LLAMA_THREADS))
LLAMA_FLASH_ATTN = _bool_secret("LLAMA_FLASH_ATTN", False)
LLAMA_GPU_LAYERS = _int_secret("LLAMA_GPU_LAYERS", 0)
LLAMA_OFFLOAD_KQV = _bool_secret("LLAMA_OFFLOAD_KQV", True)
LLAMA_OP_OFFLOAD = _bool_secret("LLAMA_OP_OFFLOAD", False)
LLAMA_USE_MMAP = _bool_secret("LLAMA_USE_MMAP", True)
LLAMA_USE_MLOCK = _bool_secret("LLAMA_USE_MLOCK", False)
LLAMA_LAST_N_TOKENS_SIZE = _int_secret("LLAMA_LAST_N_TOKENS_SIZE", 64)
CHAT_HISTORY_CONTEXT_TOKENS = max(128, _int_secret("CHAT_HISTORY_CONTEXT_TOKENS", 640))

DATA_DIR = Path(_secret_or_env("DATA_DIR", "data") or "data").expanduser()
MEMORY_PATH = DATA_DIR / "memory.json"
POPUP_USER_PATH = DATA_DIR / "popup_user.json"
EMOTION_STATE_PATH = DATA_DIR / "emotion.json"
MAX_INPUT_CHARS = max(256, _int_secret("MAX_INPUT_CHARS", 8_000))
SUMMARY_CONTEXT_TOKENS = max(32, _int_secret("SUMMARY_CONTEXT_TOKENS", 160))
MAX_CONVERSATIONS = max(8, _int_secret("MAX_CONVERSATIONS", 64))
CONVERSATION_STALE_DAYS = max(1, _int_secret("CONVERSATION_STALE_DAYS", 30))
MAX_PENDING_GENERATIONS = max(0, _int_secret("MAX_PENDING_GENERATIONS", 4))
GENERATION_QUEUE_TIMEOUT_SECONDS = max(
    1.0,
    _float_secret("GENERATION_QUEUE_TIMEOUT_SECONDS", 120.0),
)

ADVISOR_ONLY = _bool_secret("ADVISOR_ONLY", False)

MAX_TOKENS = max(24, min(_int_secret("MAX_TOKENS", 160), LLAMA_CONTEXT_WINDOW - 256))
TEMPERATURE = max(0.0, min(2.0, _float_secret("TEMPERATURE", 0.95)))
TOP_K = max(0, _int_secret("TOP_K", 40))
TOP_P = max(0.05, min(1.0, _float_secret("TOP_P", 0.9)))
MIN_P = max(0.0, min(1.0, _float_secret("MIN_P", 0.05)))
REPETITION_PENALTY = max(0.8, min(1.5, _float_secret("REPETITION_PENALTY", 1.08)))
GENERATION_STOP_SEQUENCES = _csv_strings("GENERATION_STOP_SEQUENCES", "")
STREAM_CHUNK_CHARS = max(1, _int_secret("STREAM_CHUNK_CHARS", 24))
STREAM_FLUSH_SECONDS = max(0.01, _float_secret("STREAM_FLUSH_SECONDS", 0.04))
PROMPT_TOKEN_BUDGET = max(
    256,
    min(
        _int_secret("PROMPT_TOKEN_BUDGET", 1_700),
        max(256, LLAMA_CONTEXT_WINDOW - MAX_TOKENS - 96),
    ),
)
PROMPT_DEBUG = _bool_secret("PROMPT_DEBUG", False)
