"""Central, typed runtime configuration for Akane."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

from app.ai.background.remote import remote_worker_url, validate_worker_token

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


def _strings(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(part.strip() for part in _raw(name, default).split(",") if part.strip())


def _path(root: Path, value: str) -> Path:
    return root / Path(value).expanduser()


def _context_size(name: str) -> int:
    try:
        return int(_raw(name, '4096'))
    except ValueError as error:
        raise ValueError(f'AKANE_{name} must be an integer.') from error


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    app_mode: str
    server_host: str
    server_port: int
    server_api_token: str
    cors_allowed_origins: tuple[str, ...]
    popup_backend_url: str
    model_path: str
    max_input_chars: int
    timing_enabled: bool
    next_auto_delegation: bool = True
    next_runtime: str = 'shared'
    next_l1_model_path: Path | None = None
    next_l2_model_path: Path | None = None
    next_l1_context_size: int = 4096
    next_l2_context_size: int = 4096
    next_async_delegation: bool = False
    next_l2_lazy_load: bool = False
    next_l2_idle_unload_seconds: float = 60.0
    next_l2_remote_url: str = ''
    next_l2_remote_token: str = field(default='', repr=False)
    next_l2_remote_timeout: float = 20.0

    def validate(self) -> None:
        if not 1 <= self.server_port <= 65535:
            raise ValueError("AKANE_SERVER_PORT must be between 1 and 65535.")
        if self.max_input_chars < 1:
            raise ValueError("AKANE_MAX_INPUT_CHARS must be positive.")
        if self.next_runtime not in ('shared', 'independent'):
            raise ValueError('AKANE_NEXT_RUNTIME must be shared or independent.')
        if self.next_l2_remote_url:
            remote_worker_url(self.next_l2_remote_url)
            validate_worker_token(self.next_l2_remote_token)
            if self.next_runtime != 'independent' or self.next_l2_lazy_load or (
                    self.next_auto_delegation and not self.next_async_delegation):
                raise ValueError('Remote L2 requires independent runtime and asynchronous automatic delegation when enabled.')
            timeout = self.next_l2_remote_timeout
            if type(timeout) not in (int, float) or not isfinite(timeout) or not 0 < timeout <= 120:
                raise ValueError('Remote L2 timeout must be positive and at most 120 seconds.')
        if self.next_runtime == 'independent':
            for role, path, context in (
                ('L1', self.next_l1_model_path, self.next_l1_context_size),
                ('L2', self.next_l2_model_path, self.next_l2_context_size),
            ):
                if path is None and (role != 'L2' or not self.next_l2_remote_url):
                    raise ValueError(f'AKANE_NEXT_{role}_MODEL_PATH is required in independent mode.')
                if type(context) is not int or context < 2048:
                    raise ValueError(f'AKANE_NEXT_{role}_CONTEXT_SIZE must be an integer of at least 2048.')
            if self.next_l2_lazy_load:
                if self.next_auto_delegation and not self.next_async_delegation:
                    raise ValueError('Lazy L2 automatic delegation requires AKANE_NEXT_ASYNC_DELEGATION=true.')
                timeout = self.next_l2_idle_unload_seconds
                if type(timeout) not in (int, float) or not isfinite(timeout) or timeout <= 0:
                    raise ValueError('AKANE_NEXT_L2_IDLE_UNLOAD_SECONDS must be finite and positive.')


def load_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    configured_data = Path(_raw("DATA_DIR", "data") or "data").expanduser()
    data_dir = configured_data if configured_data.is_absolute() else root / configured_data
    configured_model = Path(
        _raw("MODEL_PATH", "models/Qwen3.5-9B-Q4_K_M.gguf")
        or "models/Qwen3.5-9B-Q4_K_M.gguf"
    ).expanduser()
    model_path = configured_model if configured_model.is_absolute() else root / configured_model
    runtime = _raw('NEXT_RUNTIME', 'shared').casefold()
    independent = {}
    remote = _raw('NEXT_L2_REMOTE_URL')
    if remote:
        try:
            timeout = float(_raw('NEXT_L2_REMOTE_TIMEOUT', '20'))
        except ValueError as error:
            raise ValueError('Invalid remote L2 timeout.') from error
        independent.update(next_l2_remote_url=remote, next_l2_remote_token=_raw('NEXT_L2_REMOTE_TOKEN'),
                           next_l2_remote_timeout=timeout)
    if runtime == 'independent':
        l1_path, l2_path = _raw('NEXT_L1_MODEL_PATH'), _raw('NEXT_L2_MODEL_PATH')
        independent.update(next_l1_model_path=_path(root, l1_path) if l1_path else None,
                           next_l2_model_path=_path(root, l2_path) if l2_path else None,
                           next_l1_context_size=_context_size('NEXT_L1_CONTEXT_SIZE'),
                           next_l2_context_size=_context_size('NEXT_L2_CONTEXT_SIZE'))
        if coerce_bool(_raw('NEXT_L2_LAZY_LOAD', 'false')):
            try:
                idle_seconds = float(_raw('NEXT_L2_IDLE_UNLOAD_SECONDS', '60'))
            except ValueError as error:
                raise ValueError('AKANE_NEXT_L2_IDLE_UNLOAD_SECONDS must be finite and positive.') from error
            independent.update(next_l2_lazy_load=True, next_l2_idle_unload_seconds=idle_seconds)
    server_port = _integer("SERVER_PORT", 8000)
    settings = Settings(
        project_root=root,
        data_dir=data_dir,
        app_mode=(_raw("APP_MODE", "popup") or "popup").casefold(),
        server_host=_raw("SERVER_HOST", "127.0.0.1") or "127.0.0.1",
        server_port=server_port,
        server_api_token=_raw("SERVER_API_TOKEN"),
        cors_allowed_origins=_strings("CORS_ALLOWED_ORIGINS", f"http://127.0.0.1:{server_port},http://localhost:{server_port},null"),
        popup_backend_url=_raw("POPUP_BACKEND_URL", f"http://127.0.0.1:{server_port}").rstrip("/"),
        model_path=str(model_path),
        max_input_chars=max(256, _integer("MAX_INPUT_CHARS", 8000)),
        timing_enabled=coerce_bool(_raw("TIMING"), False),
        next_auto_delegation=coerce_bool(_raw("NEXT_AUTO_DELEGATION", "true")),
        next_runtime=runtime,
        next_async_delegation=coerce_bool(_raw('NEXT_ASYNC_DELEGATION', 'false')),
        **independent,
    )
    settings.validate()
    return settings


SETTINGS = load_settings()


def popup_backend_is_local() -> bool:
    hostname = (urlsplit(SETTINGS.popup_backend_url).hostname or "").strip().casefold()
    return hostname in {"", "127.0.0.1", "localhost", "0.0.0.0", "::1"}
