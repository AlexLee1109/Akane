"""Thread-safe storage for the current read-only VS Code editor snapshot."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

MAX_REQUEST_BYTES = 24 * 1024
MAX_SELECTION_CHARS = 1_600
MAX_NEARBY_CHARS = 3_200
MAX_DIFF_CHARS = 2_400
MAX_DIAGNOSTICS = 4
MAX_EVENT_DETAIL_CHARS = 500
MAX_SYMBOL_CHARS = 160
SNAPSHOT_TTL_SECONDS = 120
REVIEW_COOLDOWN_SECONDS = 90
MAX_REVIEW_FINGERPRINTS = 32
MAX_PROJECT_COOLDOWNS = 16

_BLOCKED_DIRECTORIES = {
    ".git", "node_modules", ".venv", "venv", "dist", "build", "target",
    "__pycache__", "secrets",
}
_SENSITIVE_NAME = re.compile(
    r"(^|[._-])(secret|token|credential|password|api[_-]?key)([._-]|$)",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|ghp|github_pat|xox[baprs])[-_][A-Za-z0-9_-]{12,}\b", re.I),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/-]{12,}"),
    re.compile(
        r"(?im)(\b(?:api[_-]?key|secret|token|password|passwd|credential|"
        r"client[_-]?secret)\b\s*[:=]\s*)([^\s,;]+)"
    ),
    re.compile(r"(?i)([?&](?:token|api[_-]?key|secret|password)=)[^&\s]+"),
)
_STYLE_DIAGNOSTIC = re.compile(
    r"\b(style|format(?:ting|ter)?|whitespace|line too long|unused|lint|"
    r"convention|spelling|trailing comma)\b",
    re.I,
)
_BUG_DIAGNOSTIC = re.compile(
    r"\b(not defined|undefined|unresolved|cannot find|could not find|missing import|"
    r"no module named|type mismatch|possibly (?:null|undefined)|null reference|"
    r"syntax error|parse error|circular import|failed|failure|exception)\b",
    re.I,
)
_RISKY_DIFF = re.compile(
    r"(?:\b(?:eval|exec)\s*\(|shell\s*=\s*true|verify\s*=\s*false|"
    r"except(?:\s+exception)?\s*:\s*(?:\n\s*)?pass\b|<<<<<<<|=======|>>>>>>>)",
    re.I,
)
_RISKY_REMOVAL = re.compile(
    r"(?im)^-.*\b(auth|permission|validate|sanitize|escape|lock|rollback|"
    r"signature|checksum)\b"
)
_REVIEW_EVENTS = {"save", "diagnostics", "test_failure", "build_failure", "git_diff"}


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
    recent_diff: str
    current_symbol: str
    event_type: str
    event_detail: str
    project_id: str
    event_sequence: int
    updated_at: float


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    accepted: bool
    reason: str = ""
    snapshot: EditorSnapshot | None = None


_LOCK = threading.RLock()
_LATEST: EditorSnapshot | None = None
_REVIEW_FINGERPRINTS: OrderedDict[str, float] = OrderedDict()
_PROJECT_COOLDOWNS: OrderedDict[str, float] = OrderedDict()


def _text(value: Any, limit: int) -> str:
    text = str(value or "").replace("\x00", "").strip()
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]",
            text,
        )
    return text[:limit]


def _project_id(value: Any) -> str:
    project = re.sub(r"[^a-zA-Z0-9_-]+", "", str(value or ""))[:64]
    return project or "default"


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

    try:
        event_sequence = max(0, int(payload.get("event_sequence") or 0))
    except (TypeError, ValueError):
        event_sequence = 0

    snapshot = EditorSnapshot(
        filename=_safe_filename(payload.get("filename")),
        language=_text(payload.get("language"), 60),
        selection=_text(payload.get("selection"), MAX_SELECTION_CHARS),
        nearby_code=_text(payload.get("nearby_code"), MAX_NEARBY_CHARS),
        diagnostics=tuple(diagnostics),
        recent_diff=_text(payload.get("recent_diff"), MAX_DIFF_CHARS),
        current_symbol=_text(payload.get("current_symbol"), MAX_SYMBOL_CHARS),
        event_type=_text(payload.get("event_type"), 24).lower() or "context",
        event_detail=_text(payload.get("event_detail"), MAX_EVENT_DETAIL_CHARS),
        project_id=_project_id(payload.get("project_id")),
        event_sequence=event_sequence,
        updated_at=time.time(),
    )
    global _LATEST
    with _LOCK:
        if (
            _LATEST is not None
            and _LATEST.project_id == snapshot.project_id
            and snapshot.event_sequence
            and snapshot.event_sequence < _LATEST.event_sequence
        ):
            return workspace_status()
        _LATEST = snapshot
    return workspace_status()


def _review_reason(snapshot: EditorSnapshot) -> str:
    if snapshot.event_type not in _REVIEW_EVENTS:
        return ""
    if snapshot.event_type in {"test_failure", "build_failure"}:
        return snapshot.event_detail or snapshot.event_type.replace("_", " ")
    for diagnostic in snapshot.diagnostics:
        message = diagnostic.message
        if _STYLE_DIAGNOSTIC.search(message):
            continue
        if diagnostic.severity.lower() == "error" or _BUG_DIAGNOSTIC.search(message):
            return f"{diagnostic.severity or 'diagnostic'}: {message}"
    if snapshot.event_type in {"save", "git_diff"} and (
        _RISKY_DIFF.search(snapshot.recent_diff)
        or _RISKY_REMOVAL.search(snapshot.recent_diff)
    ):
        return "the recent diff contains a potentially risky change"
    return ""


def claim_editor_review(now: float | None = None) -> ReviewDecision:
    snapshot = latest_editor_context()
    if snapshot is None:
        return ReviewDecision(False, "no context")
    reason = _review_reason(snapshot)
    if not reason:
        return ReviewDecision(False, "not meaningful", snapshot=snapshot)
    fingerprint = hashlib.sha256(
        "\x1f".join(
            (
                snapshot.project_id,
                snapshot.event_type,
                snapshot.filename,
                snapshot.current_symbol,
                snapshot.event_detail,
                snapshot.recent_diff,
                *(f"{item.severity}:{item.line}:{item.message}" for item in snapshot.diagnostics),
            )
        ).encode("utf-8")
    ).hexdigest()[:24]
    current = time.time() if now is None else float(now)
    with _LOCK:
        if fingerprint in _REVIEW_FINGERPRINTS:
            return ReviewDecision(False, "duplicate", snapshot)
        last_review = _PROJECT_COOLDOWNS.get(snapshot.project_id, 0.0)
        if current - last_review < REVIEW_COOLDOWN_SECONDS:
            return ReviewDecision(False, "cooldown", snapshot)
        _REVIEW_FINGERPRINTS[fingerprint] = current
        _REVIEW_FINGERPRINTS.move_to_end(fingerprint)
        while len(_REVIEW_FINGERPRINTS) > MAX_REVIEW_FINGERPRINTS:
            _REVIEW_FINGERPRINTS.popitem(last=False)
        _PROJECT_COOLDOWNS[snapshot.project_id] = current
        _PROJECT_COOLDOWNS.move_to_end(snapshot.project_id)
        while len(_PROJECT_COOLDOWNS) > MAX_PROJECT_COOLDOWNS:
            _PROJECT_COOLDOWNS.popitem(last=False)
    return ReviewDecision(True, reason, snapshot)


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
        _REVIEW_FINGERPRINTS.clear()
        _PROJECT_COOLDOWNS.clear()


def workspace_status() -> dict[str, Any]:
    snapshot = latest_editor_context()
    if snapshot is None:
        return {"connected": False}
    return {
        "connected": True,
        "active_file": snapshot.filename,
        "age_seconds": max(0, int(time.time() - snapshot.updated_at)),
    }
