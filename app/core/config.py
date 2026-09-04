"""Central, typed runtime configuration for Akane."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

try:
    from app.secrets import local_secrets as _local_secrets  # type: ignore
except ImportError:  # pragma: no cover - optional local file
    _local_secrets = None


def _raw(name: str, default: str = "") -> str:
    local = getattr(_local_secrets, name, default) if _local_secrets is not None else default
    value = os.environ.get(f"AKANE_{name}", os.environ.get(name, local))
    return str(default if value is None else value).strip()


def coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _integer(name: str, default: int) -> int:
    try:
        return int(_raw(name, str(default)))
    except ValueError:
        return default


def _number(name: str, default: float) -> float:
    try:
        return float(_raw(name, str(default)))
    except ValueError:
        return default


def _strings(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(part.strip() for part in _raw(name, default).split(",") if part.strip())


def _integers(name: str, default: str = "") -> tuple[int, ...]:
    values: list[int] = []
    for part in _strings(name, default):
        try:
            values.append(int(part))
        except ValueError:
            continue
    return tuple(values)


_LLAMA_CACHE_TYPES = frozenset({
    "f16", "f32", "q8_0", "q4_0", "q4_1", "iq4_nl",
})


def _llama_cache_type(name: str, default: str = "") -> str:
    value = _raw(name, default).casefold()
    if value and value not in _LLAMA_CACHE_TYPES:
        choices = ", ".join(sorted(_LLAMA_CACHE_TYPES))
        raise ValueError(f"AKANE_{name} must be one of: {choices}.")
    return value


def _is_raspberry_pi() -> bool:
    if platform.machine().casefold() not in {"aarch64", "arm64", "armv7l", "armv8l"}:
        return False
    for path in (Path("/proc/device-tree/model"), Path("/sys/firmware/devicetree/base/model")):
        try:
            if "raspberry pi" in path.read_text(encoding="utf-8", errors="ignore").casefold():
                return True
        except OSError:
            continue
    return "raspberry" in platform.node().casefold()


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    state_path: Path
    app_mode: str
    timezone: str
    server_host: str
    server_port: int
    server_api_token: str
    cors_allowed_origins: tuple[str, ...]
    popup_backend_url: str
    discord_server_url: str
    discord_bot_token: str
    discord_prefix: str
    discord_allowed_channel_ids: tuple[int, ...]
    discord_reply_to_dms: bool
    public_api_enabled: bool
    public_allowed_origins: tuple[str, ...]
    public_guest_idle_seconds: int
    public_guest_max_lifetime_seconds: int
    public_max_guest_sessions: int
    public_max_active: int
    public_max_queue: int
    public_message_limit: int
    public_request_cooldown_seconds: float
    public_generation_timeout_seconds: float
    model_path: str
    llama_context_window: int
    llama_batch_size: int
    llama_ubatch_size: int
    llama_threads: int
    llama_threads_batch: int
    llama_flash_attn: bool
    llama_gpu_layers: int
    llama_offload_kqv: bool
    llama_op_offload: bool
    llama_use_mmap: bool
    llama_use_mlock: bool
    llama_swa_full: bool
    llama_cache_type_k: str
    llama_cache_type_v: str
    llama_enable_thinking: bool
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    repetition_penalty: float
    generation_stop_sequences: tuple[str, ...]
    max_input_chars: int
    max_pending_generations: int
    generation_queue_timeout_seconds: float
    recent_turn_limit: int
    recent_conversation_budget: int
    memory_result_limit: int
    self_result_limit: int
    prompt_debug: bool
    timing_enabled: bool

    def validate(self) -> None:
        if not 1 <= self.server_port <= 65535:
            raise ValueError("AKANE_SERVER_PORT must be between 1 and 65535.")
        if self.llama_context_window < 512:
            raise ValueError("AKANE_LLAMA_CONTEXT_WINDOW must be at least 512.")
        if self.max_tokens >= self.llama_context_window:
            raise ValueError("AKANE_MAX_TOKENS must be smaller than the model context window.")
        if self.max_input_chars < 1:
            raise ValueError("AKANE_MAX_INPUT_CHARS must be positive.")


def load_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    configured_data = Path(_raw("DATA_DIR", "data") or "data").expanduser()
    data_dir = configured_data if configured_data.is_absolute() else root / configured_data
    configured_model = Path(
        _raw("MODEL_PATH", "models/gemma-4-E4B-it-Q4_K_M.gguf")
        or "models/gemma-4-E4B-it-Q4_K_M.gguf"
    ).expanduser()
    model_path = configured_model if configured_model.is_absolute() else root / configured_model
    raspberry_pi = coerce_bool(_raw("RASPBERRY_PI", ""), _is_raspberry_pi())
    cpu_count = os.cpu_count() or 4
    context_window = max(512, _integer("LLAMA_CONTEXT_WINDOW", 4096))
    batch_size = max(1, min(_integer("LLAMA_BATCH_SIZE", 256 if raspberry_pi else 1024), context_window))
    server_port = _integer("SERVER_PORT", 8000)
    settings = Settings(
        project_root=root,
        data_dir=data_dir,
        state_path=data_dir / "akane_state.json",
        app_mode=(_raw("APP_MODE", "popup") or "popup").casefold(),
        timezone=_raw("TIMEZONE", "America/New_York") or "America/New_York",
        server_host=_raw("SERVER_HOST", "127.0.0.1") or "127.0.0.1",
        server_port=server_port,
        server_api_token=_raw("SERVER_API_TOKEN"),
        cors_allowed_origins=_strings("CORS_ALLOWED_ORIGINS", f"http://127.0.0.1:{server_port},http://localhost:{server_port},null"),
        popup_backend_url=_raw("POPUP_BACKEND_URL", f"http://127.0.0.1:{server_port}").rstrip("/"),
        discord_server_url=_raw("DISCORD_SERVER_URL", f"http://127.0.0.1:{server_port}").rstrip("/"),
        discord_bot_token=_raw("DISCORD_BOT_TOKEN"),
        discord_prefix=_raw("DISCORD_PREFIX", "!akane"),
        discord_allowed_channel_ids=_integers("DISCORD_ALLOWED_CHANNEL_IDS"),
        discord_reply_to_dms=coerce_bool(_raw("DISCORD_REPLY_TO_DMS", "1"), True),
        public_api_enabled=coerce_bool(_raw("PUBLIC_API_ENABLED"), False),
        public_allowed_origins=_strings("PUBLIC_ALLOWED_ORIGINS"),
        public_guest_idle_seconds=max(1, _integer("PUBLIC_GUEST_IDLE_SECONDS", 1800)),
        public_guest_max_lifetime_seconds=max(1, _integer("PUBLIC_GUEST_MAX_LIFETIME_SECONDS", 7200)),
        public_max_guest_sessions=max(1, _integer("PUBLIC_MAX_GUEST_SESSIONS", 32)),
        public_max_active=max(1, _integer("PUBLIC_MAX_ACTIVE", 1)),
        public_max_queue=max(0, _integer("PUBLIC_MAX_QUEUE", 2)),
        public_message_limit=max(1, _integer("PUBLIC_MESSAGE_LIMIT", 750)),
        public_request_cooldown_seconds=max(0.0, _number("PUBLIC_REQUEST_COOLDOWN_SECONDS", 8.0)),
        public_generation_timeout_seconds=max(1.0, _number("PUBLIC_GENERATION_TIMEOUT_SECONDS", 90.0)),
        model_path=str(model_path),
        llama_context_window=context_window,
        llama_batch_size=batch_size,
        llama_ubatch_size=max(1, min(_integer("LLAMA_UBATCH_SIZE", min(128 if raspberry_pi else 512, batch_size)), batch_size)),
        llama_threads=max(1, _integer("LLAMA_THREADS", min(4, cpu_count) if raspberry_pi else max(1, cpu_count - 1))),
        llama_threads_batch=max(1, _integer("LLAMA_THREADS_BATCH", min(4, cpu_count) if raspberry_pi else max(1, cpu_count - 1))),
        llama_flash_attn=coerce_bool(_raw("LLAMA_FLASH_ATTN"), raspberry_pi),
        llama_gpu_layers=_integer("LLAMA_GPU_LAYERS", 0),
        llama_offload_kqv=coerce_bool(_raw("LLAMA_OFFLOAD_KQV"), False),
        llama_op_offload=coerce_bool(_raw("LLAMA_OP_OFFLOAD"), False),
        llama_use_mmap=coerce_bool(_raw("LLAMA_USE_MMAP", "1"), True),
        llama_use_mlock=coerce_bool(_raw("LLAMA_USE_MLOCK"), False),
        llama_swa_full=coerce_bool(_raw("LLAMA_SWA_FULL"), raspberry_pi),
        llama_cache_type_k=_llama_cache_type("LLAMA_CACHE_TYPE_K", "q8_0" if raspberry_pi else ""),
        llama_cache_type_v=_llama_cache_type("LLAMA_CACHE_TYPE_V", "q8_0" if raspberry_pi else ""),
        llama_enable_thinking=coerce_bool(_raw("LLAMA_ENABLE_THINKING"), False),
        max_tokens=max(24, min(_integer("MAX_TOKENS", 96), context_window - 256)),
        temperature=max(0.0, min(2.0, _number("TEMPERATURE", 0.98))),
        top_k=max(0, _integer("TOP_K", 64)),
        top_p=max(0.05, min(1.0, _number("TOP_P", 0.93))),
        min_p=max(0.0, min(1.0, _number("MIN_P", 0.04))),
        repetition_penalty=max(
            0.8, min(1.5, _number("REPETITION_PENALTY", 1.10)),
        ),
        generation_stop_sequences=_strings("GENERATION_STOP_SEQUENCES"),
        max_input_chars=max(256, _integer("MAX_INPUT_CHARS", 8000)),
        max_pending_generations=max(0, _integer("MAX_PENDING_GENERATIONS", 4)),
        generation_queue_timeout_seconds=max(1.0, _number("GENERATION_QUEUE_TIMEOUT_SECONDS", 120.0)),
        recent_turn_limit=max(4, _integer("RECENT_TURN_LIMIT", 16)),
        recent_conversation_budget=max(64, _integer("RECENT_CONVERSATION_BUDGET", 360)),
        memory_result_limit=min(2, max(1, _integer("MEMORY_MAX_RESULTS", 2))),
        self_result_limit=min(2, max(1, _integer("SELF_MAX_RESULTS", 2))),
        prompt_debug=coerce_bool(_raw("PROMPT_DEBUG"), False),
        timing_enabled=coerce_bool(_raw("TIMING"), False),
    )
    settings.validate()
    return settings


SETTINGS = load_settings()


def popup_backend_is_local() -> bool:
    hostname = (urlsplit(SETTINGS.popup_backend_url).hostname or "").strip().casefold()
    return hostname in {"", "127.0.0.1", "localhost", "0.0.0.0", "::1"}
