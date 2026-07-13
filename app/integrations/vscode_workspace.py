"""Thread-safe storage for the current read-only VS Code editor snapshot."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any

MAX_REQUEST_BYTES = 32 * 1024
MAX_SELECTION_CHARS = 4_000
MAX_NEARBY_CHARS = 8_000
MAX_DIAGNOSTICS = 5
SNAPSHOT_TTL_SECONDS = 120

_BLOCKED_DIRECTORIES = {
    ".git", "node_modules", ".venv", "venv", "dist", "build", "target",
    "__pycache__", "secrets",
}
_SENSITIVE_NAME = re.compile(
    r"(^|[._-])(secret|token|credential|password|api[_-]?key)([._-]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class EditorDiagnostic:
    severity: str
    message: str
    line: int


@dataclass(frozen=True, slots=True)
class EditorSnapshot:
    filename: str
    language: str
    selection: str
    nearby_code: str
    diagnostics: tuple[EditorDiagnostic, ...]
    updated_at: float


_LOCK = threading.Lock()
_LATEST: EditorSnapshot | None = None


def _text(value: Any, limit: int) -> str:
    return str(value or "").replace("\x00", "").strip()[:limit]


def _safe_filename(value: Any) -> str:
    filename = _text(value, 400).replace("\\", "/").lstrip("/")
    parts = [part.lower() for part in filename.split("/") if part]
    name = parts[-1] if parts else ""
    if (
        not filename
        or filename.startswith("../")
        or any(part in _BLOCKED_DIRECTORIES for part in parts)
        or name == ".env"
        or name.startswith(".env.")
        or name == "local_secrets.py"
        or name.startswith(("id_rsa", "id_ed25519"))
        or name.endswith((".pem", ".key", ".p12", ".pfx"))
        or _SENSITIVE_NAME.search(name)
    ):
        raise ValueError("filename is not eligible for editor context.")
    return filename


def update_editor_context(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    diagnostics: list[EditorDiagnostic] = []
    raw_diagnostics = payload.get("diagnostics")
    if isinstance(raw_diagnostics, list):
        for raw in raw_diagnostics[:MAX_DIAGNOSTICS]:
            if not isinstance(raw, dict):
                continue
            message = _text(raw.get("message"), 500)
            if not message:
                continue
            try:
                line = max(1, int(raw.get("line") or 1))
            except (TypeError, ValueError):
                line = 1
            diagnostics.append(
                EditorDiagnostic(_text(raw.get("severity"), 20), message, line)
            )

    snapshot = EditorSnapshot(
        filename=_safe_filename(payload.get("filename")),
        language=_text(payload.get("language"), 60),
        selection=_text(payload.get("selection"), MAX_SELECTION_CHARS),
        nearby_code=_text(payload.get("nearby_code"), MAX_NEARBY_CHARS),
        diagnostics=tuple(diagnostics),
        updated_at=time.time(),
    )
    global _LATEST
    with _LOCK:
        _LATEST = snapshot
    return workspace_status()


def latest_editor_context() -> EditorSnapshot | None:
    global _LATEST
    with _LOCK:
        if _LATEST and time.time() - _LATEST.updated_at > SNAPSHOT_TTL_SECONDS:
            _LATEST = None
        return _LATEST


def clear_workspace_context() -> None:
    global _LATEST
    with _LOCK:
        _LATEST = None


def workspace_status() -> dict[str, Any]:
    snapshot = latest_editor_context()
    if snapshot is None:
        return {"connected": False}
    return {
        "connected": True,
        "active_file": snapshot.filename,
        "age_seconds": max(0, int(time.time() - snapshot.updated_at)),
    }
