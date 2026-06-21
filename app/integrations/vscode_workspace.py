"""In-memory VS Code workspace index and file request transport."""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

MAX_REQUEST_BYTES = 256 * 1024
MAX_INDEX_FILES = 1_200
MAX_OPEN_FILES = 24
MAX_DOCUMENTS = 6
MAX_FILE_CONTENT_CHARS = 12_000
MAX_SELECTION_CHARS = 2_000
FILE_REQUEST_TIMEOUT_SECONDS = 3.0
PROVIDER_STALE_SECONDS = 3.0

_SENSITIVE_NAME = re.compile(
    r"(^|[._-])(?:secret|secrets|token|tokens|credential|credentials|password|"
    r"passwd|api[_-]?key)([._-]|$)",
    re.IGNORECASE,
)
_IGNORED_DIRECTORIES = {
    ".git", "node_modules", ".venv", "venv", "dist", "build", "target",
    "__pycache__", ".pytest_cache", ".mypy_cache", "secrets",
}


@dataclass(frozen=True, slots=True)
class WorkspaceFile:
    path: str
    size: int
    language: str
    mtime: int


@dataclass(frozen=True, slots=True)
class WorkspaceDocument:
    path: str
    language: str
    content: str


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    workspace_name: str
    workspace_folder: str
    files: tuple[WorkspaceFile, ...]
    open_files: tuple[str, ...]
    languages: tuple[str, ...]
    active_file: str
    selection: str
    updated_at: float

    @property
    def file_tree(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files)


@dataclass(slots=True)
class PendingFileRequest:
    request_id: str
    paths: tuple[str, ...]
    created_at: float
    event: threading.Event = field(default_factory=threading.Event)
    delivered_at: float = 0.0
    documents: tuple[WorkspaceDocument, ...] = ()
    error: str = ""


_LOCK = threading.Lock()
_LATEST: WorkspaceSnapshot | None = None
_LAST_PROVIDER_SEEN = 0.0
_FILE_CACHE: dict[str, WorkspaceDocument] = {}
_PENDING: dict[str, PendingFileRequest] = {}


def _text(value: Any, limit: int) -> str:
    return str(value or "").replace("\x00", "").strip()[:limit]


def _safe_path(value: Any) -> str:
    path = _text(value, 400).replace("\\", "/").lstrip("/")
    if not path or path.startswith("../"):
        return ""
    parts = [part.lower() for part in path.split("/") if part]
    if not parts or any(part in _IGNORED_DIRECTORIES for part in parts):
        return ""
    name = parts[-1]
    if (
        name == ".env"
        or name.startswith(".env.")
        or name == "local_secrets.py"
        or name.startswith(("id_rsa", "id_ed25519"))
        or name.endswith((".pem", ".key", ".p12", ".pfx"))
        or _SENSITIVE_NAME.search(name)
    ):
        return ""
    return path


def _folder_name(value: Any) -> str:
    path = _text(value, 200).replace("\\", "/").rstrip("/")
    return path.rsplit("/", 1)[-1] if path else ""


def _parse_files(payload: dict[str, Any]) -> tuple[WorkspaceFile, ...]:
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raw_files = []
    files: list[WorkspaceFile] = []
    seen: set[str] = set()
    for raw in raw_files:
        raw = raw if isinstance(raw, dict) else {"path": raw}
        path = _safe_path(raw.get("path"))
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            size = max(0, int(raw.get("size") or 0))
            mtime = max(0, int(raw.get("mtime") or 0))
        except (TypeError, ValueError):
            size = mtime = 0
        files.append(
            WorkspaceFile(
                path=path,
                size=size,
                language=_text(raw.get("language"), 60),
                mtime=mtime,
            )
        )
        if len(files) >= MAX_INDEX_FILES:
            break
    return tuple(files)


def _parse_documents(payload: dict[str, Any]) -> tuple[WorkspaceDocument, ...]:
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, list):
        return ()
    documents: list[WorkspaceDocument] = []
    for raw in raw_documents[:MAX_DOCUMENTS]:
        if not isinstance(raw, dict):
            continue
        path = _safe_path(raw.get("path"))
        content = _text(raw.get("content"), MAX_FILE_CONTENT_CHARS)
        if path and content:
            documents.append(
                WorkspaceDocument(
                    path=path,
                    language=_text(raw.get("language"), 60),
                    content=content,
                )
            )
    return tuple(documents)


def update_workspace_index(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    workspace = payload.get("workspace")
    workspace = workspace if isinstance(workspace, dict) else {}
    name = _text(workspace.get("name"), 120)
    if not name:
        raise ValueError("workspace.name is required.")

    files = _parse_files(payload)
    file_paths = {item.path for item in files}
    active_file = _safe_path(payload.get("active_file"))
    open_files = tuple(
        path
        for raw in (payload.get("open_files") or [])[:MAX_OPEN_FILES]
        if (path := _safe_path(raw)) and path in file_paths
    )
    languages = tuple(
        dict.fromkeys(
            text
            for value in (payload.get("languages") or [])[:16]
            if (text := _text(value, 60))
        )
    )
    documents = _parse_documents(payload)
    snapshot = WorkspaceSnapshot(
        workspace_name=name,
        workspace_folder=_folder_name(workspace.get("folder")),
        files=files,
        open_files=open_files,
        languages=languages,
        active_file=active_file if active_file in file_paths else "",
        selection=(
            _text(payload.get("selection"), MAX_SELECTION_CHARS)
            if active_file in file_paths
            else ""
        ),
        updated_at=time.time(),
    )

    global _LATEST, _LAST_PROVIDER_SEEN
    with _LOCK:
        old_files = {
            item.path: (item.size, item.mtime)
            for item in (_LATEST.files if _LATEST else ())
        }
        new_files = {item.path: (item.size, item.mtime) for item in files}
        for path in list(_FILE_CACHE):
            if path not in new_files or old_files.get(path) != new_files[path]:
                _FILE_CACHE.pop(path, None)
        for document in documents:
            if document.path in file_paths:
                _FILE_CACHE[document.path] = document
        _LATEST = snapshot
        _LAST_PROVIDER_SEEN = time.time()
    return workspace_status()


def update_active_context(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    documents = _parse_documents(payload)

    global _LATEST, _LAST_PROVIDER_SEEN
    with _LOCK:
        if _LATEST is None:
            raise ValueError("Workspace index is not connected.")
        file_paths = set(_LATEST.file_tree)
        active_file = _safe_path(payload.get("active_file"))
        open_files = tuple(
            path
            for raw in (payload.get("open_files") or [])[:MAX_OPEN_FILES]
            if (path := _safe_path(raw)) and path in file_paths
        )
        languages = tuple(
            dict.fromkeys(
                text
                for value in (payload.get("languages") or [])[:16]
                if (text := _text(value, 60))
            )
        )
        for document in documents:
            if document.path in file_paths:
                _FILE_CACHE[document.path] = document
        _LATEST = replace(
            _LATEST,
            open_files=open_files,
            languages=languages or _LATEST.languages,
            active_file=active_file if active_file in file_paths else "",
            selection=(
                _text(payload.get("selection"), MAX_SELECTION_CHARS)
                if active_file in file_paths
                else ""
            ),
            updated_at=time.time(),
        )
        _LAST_PROVIDER_SEEN = time.time()
    return workspace_status()


def clear_workspace_context() -> None:
    global _LATEST, _LAST_PROVIDER_SEEN
    with _LOCK:
        _LATEST = None
        _LAST_PROVIDER_SEEN = 0.0
        _FILE_CACHE.clear()
        for request in _PENDING.values():
            request.error = "VS Code disconnected."
            request.event.set()
        _PENDING.clear()


def latest_workspace_context() -> WorkspaceSnapshot | None:
    with _LOCK:
        return _LATEST


def workspace_status() -> dict[str, Any]:
    with _LOCK:
        snapshot = _LATEST
        provider_age = time.time() - _LAST_PROVIDER_SEEN if _LAST_PROVIDER_SEEN else 0.0
        pending = sum(1 for request in _PENDING.values() if not request.event.is_set())
    if snapshot is None:
        return {"connected": False, "pending_requests": pending}
    return {
        "connected": provider_age <= PROVIDER_STALE_SECONDS,
        "workspace": snapshot.workspace_name,
        "active_file": snapshot.active_file,
        "file_count": len(snapshot.files),
        "open_file_count": len(snapshot.open_files),
        "pending_requests": pending,
        "age_seconds": max(0, int(time.time() - snapshot.updated_at)),
    }


def _prune_pending(now: float) -> None:
    for request_id, request in list(_PENDING.items()):
        if request.event.is_set() or now - request.created_at > 15:
            _PENDING.pop(request_id, None)


def next_file_request() -> dict[str, Any]:
    global _LAST_PROVIDER_SEEN
    now = time.time()
    with _LOCK:
        _LAST_PROVIDER_SEEN = now
        _prune_pending(now)
        for request in _PENDING.values():
            if request.event.is_set():
                continue
            if request.delivered_at and now - request.delivered_at < 1.5:
                continue
            request.delivered_at = now
            return {
                "request": {
                    "id": request.request_id,
                    "paths": list(request.paths),
                }
            }
    return {"request": None}


def complete_file_request(
    request_id: str,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    global _LAST_PROVIDER_SEEN
    with _LOCK:
        request = _PENDING.get(str(request_id))
        indexed = set(_LATEST.file_tree if _LATEST else ())
    if request is None:
        raise ValueError("File request is no longer pending.")

    documents: list[WorkspaceDocument] = []
    for raw in (payload.get("files") or [])[: len(request.paths)]:
        if not isinstance(raw, dict):
            continue
        path = _safe_path(raw.get("path"))
        content = _text(raw.get("content"), MAX_FILE_CONTENT_CHARS)
        if path in request.paths and path in indexed and content:
            documents.append(
                WorkspaceDocument(
                    path=path,
                    language=_text(raw.get("language"), 60),
                    content=content,
                )
            )
    with _LOCK:
        request = _PENDING.get(str(request_id))
        if request is None:
            raise ValueError("File request is no longer pending.")
        request.documents = tuple(documents)
        request.error = _text(payload.get("error"), 200)
        for document in documents:
            _FILE_CACHE[document.path] = document
        _LAST_PROVIDER_SEEN = time.time()
        request.event.set()
    return {"ok": True, "received": len(documents)}


def request_documents(paths: list[str]) -> tuple[WorkspaceDocument, ...]:
    unique = list(dict.fromkeys(path for path in paths if path))
    with _LOCK:
        cached = [_FILE_CACHE[path] for path in unique if path in _FILE_CACHE]
        missing = [path for path in unique if path not in _FILE_CACHE]
        if not missing:
            return tuple(cached)
        request = PendingFileRequest(
            request_id=uuid.uuid4().hex,
            paths=tuple(missing),
            created_at=time.time(),
        )
        _PENDING[request.request_id] = request

    request.event.wait(FILE_REQUEST_TIMEOUT_SECONDS)
    with _LOCK:
        completed = _PENDING.pop(request.request_id, request)
        fetched = list(completed.documents)
        for path in missing:
            document = _FILE_CACHE.get(path)
            if document and document not in fetched:
                fetched.append(document)
    by_path = {document.path: document for document in [*cached, *fetched]}
    return tuple(by_path[path] for path in unique if path in by_path)
