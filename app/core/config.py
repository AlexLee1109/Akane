"""Small runtime configuration layer."""

from __future__ import annotations

import os
import platform
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


def _is_raspberry_pi() -> bool:
    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64", "armv7l", "armv8l"}:
        return False
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            if "raspberry pi" in open(path, encoding="utf-8", errors="ignore").read().lower():
                return True
        except OSError:
            pass
    return "raspberry" in platform.node().lower()


IS_RASPBERRY_PI = _bool_secret("RASPBERRY_PI", _is_raspberry_pi())
APP_MODE = (_secret_or_env("APP_MODE", "popup") or "popup").lower()
SERVER_HOST = _secret_or_env("SERVER_HOST", "127.0.0.1") or "127.0.0.1"
SERVER_PORT = _int_secret("SERVER_PORT", 8000)
POPUP_BACKEND_URL = _secret_or_env("POPUP_BACKEND_URL", f"http://127.0.0.1:{SERVER_PORT}").rstrip("/")
DISCORD_SERVER_URL = _secret_or_env("DISCORD_SERVER_URL", f"http://127.0.0.1:{SERVER_PORT}").rstrip("/")

DISCORD_BOT_TOKEN = _secret_or_env("DISCORD_BOT_TOKEN", "")
DISCORD_PREFIX = _secret_or_env("DISCORD_PREFIX", "!akane")
DISCORD_ALLOWED_CHANNEL_IDS = _csv_ints("DISCORD_ALLOWED_CHANNEL_IDS", "")
DISCORD_REPLY_TO_DMS = _bool_secret("DISCORD_REPLY_TO_DMS", True)
DISCORD_UNPROMPTED_IDLE_MINUTES = _float_secret("DISCORD_UNPROMPTED_IDLE_MINUTES", 60.0)


def popup_backend_is_local() -> bool:
    hostname = (urlsplit(POPUP_BACKEND_URL).hostname or "").strip().lower()
    return hostname in {"", "127.0.0.1", "localhost", "0.0.0.0", "::1"}


DEVICE = _secret_or_env("DEVICE", "cpu") or "cpu"
MODEL_PATH = _secret_or_env("MODEL_PATH", "models/Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf")

_CPU_COUNT = os.cpu_count() or 4
_PI_THREADS = min(4, _CPU_COUNT)

LLAMA_CONTEXT_WINDOW = _int_secret("LLAMA_CONTEXT_WINDOW", 2048)
_DEFAULT_BATCH_SIZE = min(512 if IS_RASPBERRY_PI else 1024, LLAMA_CONTEXT_WINDOW)
LLAMA_BATCH_SIZE = _int_secret("LLAMA_BATCH_SIZE", _DEFAULT_BATCH_SIZE)
_DEFAULT_UBATCH_SIZE = min(128 if IS_RASPBERRY_PI else 512, LLAMA_BATCH_SIZE)
LLAMA_UBATCH_SIZE = _int_secret("LLAMA_UBATCH_SIZE", _DEFAULT_UBATCH_SIZE)
LLAMA_THREADS = _int_secret("LLAMA_THREADS", _PI_THREADS if IS_RASPBERRY_PI else max(1, _CPU_COUNT - 1))
LLAMA_THREADS_BATCH = _int_secret("LLAMA_THREADS_BATCH", LLAMA_THREADS)
LLAMA_FLASH_ATTN = _bool_secret("LLAMA_FLASH_ATTN", False)
LLAMA_GPU_LAYERS = _int_secret("LLAMA_GPU_LAYERS", 0)
LLAMA_OFFLOAD_KQV = _bool_secret("LLAMA_OFFLOAD_KQV", True)
LLAMA_OP_OFFLOAD = _bool_secret("LLAMA_OP_OFFLOAD", False)
LLAMA_USE_MMAP = _bool_secret("LLAMA_USE_MMAP", True)
LLAMA_USE_MLOCK = _bool_secret("LLAMA_USE_MLOCK", False)
LLAMA_LAST_N_TOKENS_SIZE = _int_secret("LLAMA_LAST_N_TOKENS_SIZE", 64)
CHAT_HISTORY_CONTEXT_TOKENS = _int_secret("CHAT_HISTORY_CONTEXT_TOKENS", 1024)

STATIC_DIR = "static"
VSCODE_COMMAND = _secret_or_env("VSCODE_COMMAND", "")
ADVISOR_ONLY = _bool_secret("ADVISOR_ONLY", False)

OPENROUTER_BASE_URL = _secret_or_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_API_KEY = _secret_or_env("OPENROUTER_API_KEY", "")
OPENROUTER_CODER_MODEL = _secret_or_env("OPENROUTER_CODER_MODEL", "")
OPENROUTER_SITE_URL = _secret_or_env("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = _secret_or_env("OPENROUTER_APP_NAME", "Akane")
OPENROUTER_CA_BUNDLE = _secret_or_env("OPENROUTER_CA_BUNDLE", "")
OPENROUTER_SKIP_SSL_VERIFY = _bool_secret("OPENROUTER_SKIP_SSL_VERIFY", False)

MAX_TOKENS = _int_secret("MAX_TOKENS", 160)
TEMPERATURE = _float_secret("TEMPERATURE", 0.75)
TOP_K = _int_secret("TOP_K", 30)
TOP_P = _float_secret("TOP_P", 0.9)
REPETITION_PENALTY = _float_secret("REPETITION_PENALTY", 1.08)

CODER_MAX_TOKENS = _int_secret("CODER_MAX_TOKENS", 768)
CODER_TEMPERATURE = _float_secret("CODER_TEMPERATURE", 0.1)
CODER_MAX_TURNS = _int_secret("CODER_MAX_TURNS", 2)
CODER_MAX_INITIAL_TARGETS = _int_secret("CODER_MAX_INITIAL_TARGETS", 2)
CODER_READ_CHUNK_LINES = _int_secret("CODER_READ_CHUNK_LINES", 180)
CODER_INITIAL_CHUNKS_PER_FILE = _int_secret("CODER_INITIAL_CHUNKS_PER_FILE", 1)
CODER_MAX_READ_CHUNKS_PER_FILE = _int_secret("CODER_MAX_READ_CHUNKS_PER_FILE", 3)
CODER_TIMEOUT_SECONDS = _float_secret("CODER_TIMEOUT_SECONDS", 30.0)