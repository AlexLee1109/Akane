"""Lean local chat server for Akane."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.core.config import (
    CHAT_HISTORY_CONTEXT_TOKENS,
    MAX_TOKENS,
    REPETITION_PENALTY,
    SERVER_HOST,
    SERVER_PORT,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    _coerce_bool,
    _coerce_int,
)
from app.core.character import build_system_prompt
from app.core.generation import HiddenTagStreamFilter, collapse_hidden_tag_gaps
from app.core.model_loader import ModelManager, content_to_text
from app.integrations.editor_bridge import get_editor_bridge
from app.integrations.vscode_launcher import launch_vscode
from app.memory_store import MEMORY_PATH, format_for_prompt, get_all, reload_from_disk, remember_user_message
from app.ui.assets import resolve_ui_asset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).parent / "ui" / "static"
DEFAULT_SESSION_ID = "popup"

_MAX_HISTORY_MESSAGES = 8
_MAX_HISTORY_CHARS = max(600, min(CHAT_HISTORY_CONTEXT_TOKENS * 4, 1800))
_MAX_MEMORY_CHARS = _coerce_int(os.environ.get("AKANE_MAX_MEMORY_CHARS", 1800), 1800)
_MAX_EDITOR_CHARS = _coerce_int(os.environ.get("AKANE_MAX_EDITOR_CHARS", 1600), 1600)
_FAST_MAX_TOKENS = min(MAX_TOKENS, 160)
_generation_lock = threading.Lock()
_state_lock = threading.Lock()
_SESSIONS: dict[str, "SessionState"] = {}
_LOG_PROMPTS = _coerce_bool(os.environ.get("AKANE_LOG_PROMPTS", "1"), True)
_LOG_PATH = Path(os.environ.get("AKANE_LOG_PATH", "/tmp/akane_server.log"))
_LOG_MAX_CHARS = _coerce_int(os.environ.get("AKANE_LOG_MAX_CHARS", 120_000), 120_000)

_STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
_HIDDEN_TAGS = "MEM|FORGET|PROJECT|EDITOR|ASK_CODER|CODE|READ|WRITE|SHELL|READ_RESULT|THINK"
_POPUP_STRIP_REGEX = re.compile(rf"\[({_HIDDEN_TAGS})\].*?(?:\[/\1\]|$)", re.DOTALL | re.IGNORECASE)
_XML_STRIP_REGEX = re.compile(
    r"<(think|thinking|tool_call|read|write|shell|code|read_result)>.*?(?:</\1>|$)",
    re.DOTALL | re.IGNORECASE,
)
_PATH_RE = re.compile(
    r"(?P<path>(?:\.{0,2}/)?[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)+|[A-Za-z0-9_.@+-]+\.[A-Za-z0-9_+-]+)"
    r"(?::\d+(?::\d+)?)?"
)
_FILE_REF_SUFFIXES = (".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt", ".css", ".html", ".sh")
_NON_SOURCE_FILE_SUFFIXES = {
    ".node", ".so", ".dylib", ".dll", ".exe", ".bin", ".o", ".a", ".obj",
    ".class", ".jar", ".war", ".pyd", ".whl", ".zip", ".tar", ".gz", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
}
_INFRA_PATH_PREFIXES = ("integrations/", "extensions/", ".vscode-server/", ".vscode/", ".idea/", ".zed/", ".github/")
_CODE_HINTS = (
    "code", "file", "bug", "error", "traceback", "stack", "function", "class", "server",
    "python", "javascript", "typescript", "css", "html", ".py", ".js", ".ts", ".tsx", ".json",
)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("akane.server")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            _LOG_PATH,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        print(f"[Akane:log] Could not open log file {_LOG_PATH}: {exc}", flush=True)

    return logger


_LOGGER = _build_logger()


def _log_terminal_stream(label: str, text: str = "") -> None:
    suffix = f" {text}" if text else ""
    print(f"[Akane:{label}]{suffix}", flush=True)


@dataclass
class SessionState:
    messages: deque = field(default_factory=lambda: deque(maxlen=_MAX_HISTORY_MESSAGES))
    streaming_reply_preview: str = ""
    version: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(frozen=True)
class ChatRequestData:
    text: str
    skip_memory: bool = False
    session_id: str = DEFAULT_SESSION_ID


class _StreamVisibleCleaner:
    def __init__(self) -> None:
        self._buffer = ""
        self._previous_blank = False

    def _consume_lines(self, text: str, *, flush: bool = False) -> str:
        if not text and not (flush and self._buffer):
            return ""
        working = self._buffer + text
        if "\n" not in working and not flush:
            self._buffer = working
            return ""

        parts = working.split("\n")
        self._buffer = "" if flush else parts.pop()
        lines: list[str] = []
        for raw in parts:
            line = raw.replace("\t", " ")
            if not line:
                if not self._previous_blank:
                    lines.append("")
                    self._previous_blank = True
                continue
            lines.append(line)
            self._previous_blank = False
        return "\n".join(lines) + ("\n" if lines and (working.endswith("\n") or not flush) else "")

    def feed(self, text: str) -> str:
        return self._consume_lines(str(text or "").replace("\r\n", "\n").replace("`", ""), flush=False)

    def flush(self) -> str:
        return self._consume_lines("", flush=True)


def _normalize_session_id(session_id: str | None) -> str:
    value = str(session_id or "").strip()
    return (value or DEFAULT_SESSION_ID)[:120]


def _session_state(session_id: str | None = None) -> SessionState:
    key = _normalize_session_id(session_id)
    with _state_lock:
        state = _SESSIONS.get(key)
        if state is None:
            state = SessionState()
            _SESSIONS[key] = state
        return state


async def _request_payload(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_chat_request(payload: dict) -> ChatRequestData:
    return ChatRequestData(
        text=str(payload.get("message", "")).strip(),
        skip_memory=_coerce_bool(payload.get("skip_memory", False)),
        session_id=_normalize_session_id(payload.get("session_id")),
    )


def _json_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _log_text(label: str, text: str, *, session_id: str | None = None) -> None:
    if not _LOG_PROMPTS:
        return
    value = str(text or "")
    if len(value) > _LOG_MAX_CHARS:
        value = value[:_LOG_MAX_CHARS].rstrip() + "\n[log trimmed]"
    session = _normalize_session_id(session_id)
    _LOGGER.info("\n===== %s session=%s =====\n%s\n===== end %s =====", label, session, value, label)


def _log_prompt(messages: list[dict], *, session_id: str | None = None, stream: bool = False) -> None:
    payload = json.dumps(messages, ensure_ascii=False, indent=2)
    _log_text("prompt stream" if stream else "prompt", payload, session_id=session_id)


def _log_reply(reply: str, *, session_id: str | None = None, raw: str = "") -> None:
    _log_terminal_stream("done", reply)
    if raw and raw != reply:
        _log_text("raw reply", raw, session_id=session_id)
    _log_text("reply", reply, session_id=session_id)


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[trimmed]"


def _static_response_path(route: str) -> tuple[Path, str] | None:
    if route in _STATIC_ROUTES:
        name, media_type = _STATIC_ROUTES[route]
        return STATIC_DIR / name, media_type
    asset = resolve_ui_asset(route)
    return (asset, "image/png") if asset else None


def _state_messages(session_id: str | None = None) -> list[dict]:
    state = _session_state(session_id)
    with state.lock:
        messages = list(state.messages)
        if state.streaming_reply_preview.strip():
            messages.append({"role": "assistant", "content": state.streaming_reply_preview})
        return messages


def _append_message(role: str, content: str, session_id: str | None = None) -> None:
    state = _session_state(session_id)
    with state.lock:
        state.messages.append({"role": role, "content": content})
        state.streaming_reply_preview = ""
        state.version += 1


def _set_streaming_reply_preview(text: str, session_id: str | None = None) -> None:
    state = _session_state(session_id)
    with state.lock:
        state.streaming_reply_preview = str(text or "")
        state.version += 1


def _clear_messages(session_id: str | None = None) -> None:
    state = _session_state(session_id)
    with state.lock:
        state.messages.clear()
        state.streaming_reply_preview = ""
        state.version += 1


def _clean_reference_path(raw_path: str) -> str:
    path = str(raw_path or "").strip().strip("`'\".,:;()[]{}").replace("\\", "/")
    if not path or "://" in path or path.startswith("~"):
        return ""
    if path.startswith("./"):
        path = path[2:]
    if ":" in path:
        head, _, tail = path.partition(":")
        if tail.strip()[:1].isdigit() or tail.strip().lower().startswith("l"):
            path = head.strip()
    return path


def _project_relative_path(raw_path: str, *, must_be_file: bool = False) -> str:
    path = _clean_reference_path(raw_path)
    if not path:
        return ""
    try:
        candidate = Path(path)
        resolved = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
        rel = resolved.relative_to(PROJECT_ROOT)
    except (OSError, ValueError):
        return ""
    if must_be_file and not resolved.is_file():
        return ""
    return rel.as_posix()


def _explicit_paths_in_text(text: str) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for match in _PATH_RE.finditer(str(text or "")):
        raw = _clean_reference_path(match.group("path"))
        candidates = [raw]
        if raw and not Path(raw).suffix:
            candidates.extend(f"{raw}{suffix}" for suffix in _FILE_REF_SUFFIXES)
        for candidate in candidates:
            rel = _project_relative_path(candidate, must_be_file=True)
            if rel and rel not in seen:
                seen.add(rel)
                paths.append(rel)
                break
    return paths


def _is_infra_path(path: str) -> bool:
    lowered = str(path or "").strip().lower()
    return bool(lowered) and any(lowered.startswith(prefix) for prefix in _INFRA_PATH_PREFIXES)


def _filter_coder_preload_paths(paths: list[str], user_input: str) -> list[str]:
    allow_infra = any(token in str(user_input or "").lower() for token in ("vscode", "extension", "integration"))
    seen: set[str] = set()
    out: list[str] = []
    for raw in paths:
        rel = _project_relative_path(raw, must_be_file=True)
        if not rel or rel in seen:
            continue
        suffix = Path(rel.lower()).suffix
        if suffix in _NON_SOURCE_FILE_SUFFIXES or (_is_infra_path(rel) and not allow_infra):
            continue
        seen.add(rel)
        out.append(rel)
    return out


def _response_uses_tool_calls(raw: str) -> bool:
    text = str(raw or "").lower()
    return bool(text and ("<tool_call>" in text or re.search(r"\[(read|editor|ask_coder)\]", text)))


def _response_uses_disabled_coding_tools(raw: str) -> bool:
    text = str(raw or "").lower()
    return bool(text and ("<tool_call>" in text or re.search(r"\[(editor|ask_coder)\]", text)))


def _strip_popup_tags(text: str) -> str:
    if not text:
        return text
    source = str(text)
    cleaned = _POPUP_STRIP_REGEX.sub("", source)
    cleaned = _XML_STRIP_REGEX.sub("", cleaned)
    if cleaned == source:
        return source
    return "\n".join(line.rstrip() for line in cleaned.splitlines() if line.strip())


def _clean_reply(text: str) -> str:
    cleaned = _strip_popup_tags(text).replace("/no_think", "")
    cleaned = collapse_hidden_tag_gaps(cleaned).replace("\r\n", "\n")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(" `\n\t")


def _extract_message_text(result: dict) -> str:
    choices = result.get("choices") or []
    if not choices:
        return ""
    return content_to_text((choices[0].get("message") or {}).get("content")).strip()


def _looks_codey(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(hint in lowered for hint in _CODE_HINTS) or bool(_explicit_paths_in_text(text))


def _system_prompt(user_text: str, *, skip_memory: bool) -> str:
    sections: list[str] = []
    if not skip_memory:
        memory = format_for_prompt()
        if memory:
            sections.append("Memory:\n" + _clip(memory, _MAX_MEMORY_CHARS))
    if _looks_codey(user_text):
        editor = get_editor_bridge().format_for_prompt()
        if editor:
            sections.append(_clip(editor, _MAX_EDITOR_CHARS))
    return build_system_prompt("\n\n".join(sections), include_memory=not skip_memory)


def _history_messages(session_id: str | None) -> list[dict]:
    state = _session_state(session_id)
    with state.lock:
        messages = list(state.messages)

    total = 0
    kept: list[dict] = []
    for message in reversed(messages[-_MAX_HISTORY_MESSAGES:]):
        role = str(message.get("role", "") or "")
        if role not in {"user", "assistant"}:
            continue
        content = _clip(_clean_reply(str(message.get("content", "") or "")), 700)
        if not content:
            continue
        cost = len(content)
        if kept and total + cost > _MAX_HISTORY_CHARS:
            break
        kept.append({"role": role, "content": content})
        total += cost
    kept.reverse()
    return kept


def _chat_messages(user_input: str, *, skip_memory: bool = False, session_id: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": _system_prompt(user_input, skip_memory=skip_memory)},
        *_history_messages(session_id),
        {"role": "user", "content": user_input},
    ]


def _request_completion(messages: list[dict], *, stream: bool):
    return ModelManager.get_instance().create_chat_completion(
        messages=messages,
        max_tokens=_FAST_MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repeat_penalty=REPETITION_PENALTY,
        stream=stream,
    )


def _start_model_loading() -> None:
    manager = ModelManager.get_instance()
    status = manager.status()
    if status["loading"] or status["loaded"]:
        return

    def load() -> None:
        try:
            manager.ensure_loaded()
        except Exception:
            pass

    threading.Thread(target=load, daemon=True, name="AkaneModelLoader").start()


def _generate_reply(user_input: str, *, skip_memory: bool = False, session_id: str | None = None) -> str:
    with _generation_lock:
        _log_terminal_stream("user", user_input)
        remember_user_message(user_input)
        messages = _chat_messages(user_input, skip_memory=skip_memory, session_id=session_id)
        _log_prompt(messages, session_id=session_id, stream=False)
        _append_message("user", user_input, session_id)
        result = _request_completion(messages, stream=False)
        raw = _extract_message_text(result)
        remember_user_message(raw)
        reply = _clean_reply(raw) or "I lost the thread for a second. Try that again."
        _log_reply(reply, session_id=session_id, raw=raw)
        _append_message("assistant", reply, session_id)
        return reply


def _stream_chat_events(text: str, *, skip_memory: bool = False, session_id: str | None = None):
    try:
        with _generation_lock:
            _log_terminal_stream("user", text)
            _set_streaming_reply_preview("", session_id)
            remember_user_message(text)
            messages = _chat_messages(text, skip_memory=skip_memory, session_id=session_id)
            _log_prompt(messages, session_id=session_id, stream=True)
            _append_message("user", text, session_id)
            yield _json_line({"type": "start", "messages": list(_session_state(session_id).messages)})

            hidden_filter = HiddenTagStreamFilter()
            raw_parts: list[str] = []
            visible_parts: list[str] = []
            stream = _request_completion(messages, stream=True)

            for chunk in stream:
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                token = content_to_text((choices[0].get("delta") or {}).get("content"))
                if not token:
                    continue
                raw_parts.append(token)
                visible = hidden_filter.feed(token)
                if not visible:
                    continue
                visible_parts.append(visible)
                preview = "".join(visible_parts)
                _set_streaming_reply_preview(preview, session_id)
                yield _json_line({"type": "delta", "content": visible, "append": True})

            tail = hidden_filter.flush()
            if tail:
                visible_parts.append(tail)
                preview = "".join(visible_parts)
                _set_streaming_reply_preview(preview, session_id)
                yield _json_line({"type": "delta", "content": tail, "append": True})

            raw = "".join(raw_parts)
            remember_user_message(raw)
            displayed = "".join(visible_parts)
            reply = _clean_reply(raw) or _clean_reply(displayed) or "I lost the thread for a second. Try that again."
            _log_reply(reply, session_id=session_id, raw=raw)
            if reply != displayed.strip():
                _set_streaming_reply_preview(reply, session_id)
                yield _json_line({"type": "delta", "content": reply})

            _append_message("assistant", reply, session_id)
            yield _json_line({"type": "done", "reply": reply, "messages": list(_session_state(session_id).messages)})
    except Exception as exc:
        _set_streaming_reply_preview("", session_id)
        _log_terminal_stream("error", str(exc))
        _LOGGER.exception("Chat generation failed for session=%s", _normalize_session_id(session_id))
        yield _json_line({"type": "error", "error": str(exc)})


def _handle_command(text: str, session_id: str | None = None) -> dict | None:
    if text == "/clear":
        _clear_messages(session_id)
        return {"reply": "", "messages": [], "notice": "Context cleared."}
    if text == "/memory":
        reload_from_disk()
        reply = format_for_prompt() or "No memories yet."
        return {"reply": reply, "messages": [*_state_messages(session_id), {"role": "assistant", "content": reply}], "ephemeral": True}
    if text == "/reset":
        _clear_messages(session_id)
        if MEMORY_PATH.exists():
            MEMORY_PATH.unlink()
        reload_from_disk()
        return {"reply": "", "messages": [], "notice": "Memory wiped and context cleared."}
    if text == "/vscode":
        return {"reply": "", "messages": _state_messages(session_id), "notice": launch_vscode(), "ephemeral": True}
    if text in {"/approve", "/reject"}:
        bridge = get_editor_bridge()
        actions = bridge.approve_all_pending_actions() if text == "/approve" else bridge.reject_all_pending_actions()
        verb = "Approved" if text == "/approve" else "Rejected"
        notice = f"{verb} {len(actions)} pending code change(s)." if actions else "No pending code changes."
        return {"reply": "", "messages": _state_messages(session_id), "notice": notice, "ephemeral": True}
    return None


def _app_state_payload(session_id: str | None = None, *, include_messages: bool = True) -> dict:
    state = _session_state(session_id)
    payload = {
        "model": ModelManager.get_instance().status(),
        "editor": get_editor_bridge().snapshot(),
        "version": state.version,
    }
    if include_messages:
        payload["messages"] = _state_messages(session_id)
    return payload


def create_app() -> FastAPI:
    app = FastAPI(title="Akane API", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    async def root():
        path, media_type = _static_response_path("/") or (None, None)
        if path is None:
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type=media_type)

    @app.get("/api/state")
    async def api_state(session_id: str = DEFAULT_SESSION_ID, include_messages: str = "1"):
        return JSONResponse(_app_state_payload(session_id, include_messages=_coerce_bool(include_messages, True)))

    @app.get("/api/memory")
    async def api_memory():
        return JSONResponse(get_all())

    @app.get("/api/editor/state")
    async def api_editor_state():
        return JSONResponse(get_editor_bridge().snapshot())

    @app.get("/api/editor/actions")
    async def api_editor_actions(after: int = 0):
        return JSONResponse({"actions": get_editor_bridge().actions_after(int(after))})

    @app.post("/api/editor/context")
    async def api_editor_context(request: Request):
        return JSONResponse({"ok": True, "state": get_editor_bridge().update_context(await _request_payload(request))})

    @app.post("/api/editor/action-result")
    async def api_editor_action_result(request: Request):
        payload = await _request_payload(request)
        action_id = _coerce_int(payload.get("id", 0), 0)
        if action_id <= 0:
            return JSONResponse({"error": "Missing action id."}, status_code=400)
        summary = get_editor_bridge().complete_action(
            action_id=action_id,
            ok=_coerce_bool(payload.get("ok", False)),
            result=str(payload.get("result", "")),
            error=str(payload.get("error", "")),
        )
        if summary is None:
            return JSONResponse({"error": "Unknown action id."}, status_code=404)
        return JSONResponse({"ok": True, "summary": summary})

    @app.post("/api/backend")
    async def api_backend(request: Request):
        payload = await _request_payload(request)
        try:
            status = ModelManager.get_instance().switch_backend(
                str(payload.get("backend", "")).strip().lower(),
                local_model_path=str(payload["local_model_path"]) if "local_model_path" in payload else None,
                openrouter_model=str(payload["model_name"]) if "model_name" in payload else None,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "model": status})

    @app.post("/api/open-vscode")
    async def api_open_vscode():
        return JSONResponse({"ok": True, "notice": launch_vscode(), "editor": get_editor_bridge().snapshot()})

    @app.post("/api/quit")
    async def api_quit():
        threading.Thread(target=lambda: (time.sleep(0.15), os._exit(0)), daemon=True, name="AkaneQuit").start()
        return JSONResponse({"ok": True}, status_code=202)

    @app.post("/api/chat")
    async def api_chat(request: Request):
        _start_model_loading()
        chat = _parse_chat_request(await _request_payload(request))
        if not chat.text:
            return JSONResponse({"error": "Message is empty."}, status_code=400)
        command = _handle_command(chat.text, chat.session_id)
        if command is not None:
            return JSONResponse(command)
        try:
            reply = _generate_reply(chat.text, skip_memory=chat.skip_memory, session_id=chat.session_id)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return JSONResponse({"reply": reply, "messages": _state_messages(chat.session_id)})

    @app.post("/api/chat/stream")
    async def api_chat_stream(request: Request):
        _start_model_loading()
        chat = _parse_chat_request(await _request_payload(request))
        if not chat.text:
            return JSONResponse({"error": "Message is empty."}, status_code=400)
        command = _handle_command(chat.text, chat.session_id)
        if command is not None:
            return JSONResponse(command)
        return StreamingResponse(
            _stream_chat_events(chat.text, skip_memory=chat.skip_memory, session_id=chat.session_id),
            media_type="application/x-ndjson; charset=utf-8",
            headers={"Cache-Control": "no-store, no-transform", "X-Accel-Buffering": "no"},
        )

    @app.get("/{asset_path:path}", include_in_schema=False)
    async def static_assets(asset_path: str):
        static = _static_response_path(unquote("/" + asset_path.lstrip("/")))
        if static is None:
            raise HTTPException(status_code=404)
        path, media_type = static
        return FileResponse(path, media_type=media_type)

    return app


APP = create_app()


class BackgroundUvicornServer:
    def __init__(self, host: str, port: int):
        self._config = uvicorn.Config(APP, host=host, port=port, log_level="warning", access_log=False)
        self._server = uvicorn.Server(self._config)
        self._thread: threading.Thread | None = None

    def run(self) -> None:
        self._server.run()

    def shutdown(self) -> None:
        self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


def serve(host: str = SERVER_HOST, port: int = SERVER_PORT) -> None:
    print(f"Akane web chat running at http://{host}:{port}", flush=True)
    _start_model_loading()
    try:
        BackgroundUvicornServer(host, port).run()
    except KeyboardInterrupt:
        print("\nStopping Akane web chat...", flush=True)


def serve_in_thread(host: str = SERVER_HOST, port: int = SERVER_PORT) -> tuple[BackgroundUvicornServer, threading.Thread]:
    _start_model_loading()
    server = BackgroundUvicornServer(host, port)
    thread = threading.Thread(target=server.run, daemon=True, name="AkaneAPIServer")
    server._thread = thread
    thread.start()
    print(f"Akane background API running at http://{host}:{port}", flush=True)
    return server, thread


if __name__ == "__main__":
    serve()
