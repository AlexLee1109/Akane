"""Lean local chat server for Akane."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import uvicorn
import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.core.attention_state import clear_attention
from app.core.config import SERVER_HOST, SERVER_PORT, _coerce_bool, _coerce_int
from app.core.conversation_state import clear_conversation_state
from app.core.model_loader import ModelManager
from app.core.reply_pipeline import (
    debug_state_report,
    generate_reply,
    prepare_reply,
    stream_reply,
    warm_caches,
    warm_prompt_cache,
)
from app.core.emotional_state import snapshot as emotional_snapshot
from app.memory_store import (
    format_for_prompt as format_memory_for_prompt,
    get_all,
    reload_from_disk,
)
from app.integrations.vscode_workspace import (
    MAX_REQUEST_BYTES,
    clear_workspace_context,
    complete_file_request,
    next_file_request,
    update_active_context,
    update_workspace_index,
    workspace_status,
)
from app.ui.assets import resolve_ui_asset

STATIC_DIR = Path(__file__).parent / "ui" / "static"
DEFAULT_SESSION_ID = "popup"
_DEBUG_STATE_COMMAND = "/debug_state"
_MAX_STORED_MESSAGES = max(
    1,
    _coerce_int(
        os.environ.get("AKANE_STORED_MESSAGES", os.environ.get("AKANE_HISTORY_MESSAGES", 16)),
        16,
    ),
)
_MAX_ACTIVE_SESSIONS = 64
_SESSION_STALE_SECONDS = 4 * 3600
_STATE_LOCK = threading.Lock()
_SESSIONS: dict[str, "SessionState"] = {}
_STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


@dataclass(slots=True)
class SessionState:
    messages: deque = field(default_factory=lambda: deque(maxlen=_MAX_STORED_MESSAGES))
    version: int = 0
    updated_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(frozen=True, slots=True)
class ChatRequestData:
    text: str
    skip_memory: bool = False
    session_id: str = DEFAULT_SESSION_ID


def _normalize_session_id(session_id: str | None) -> str:
    value = str(session_id or "").strip()
    return (value or DEFAULT_SESSION_ID)[:120]


def _prune_sessions(now: float) -> None:
    for key, state in list(_SESSIONS.items()):
        if now - state.updated_at > _SESSION_STALE_SECONDS:
            _SESSIONS.pop(key, None)
    if len(_SESSIONS) > _MAX_ACTIVE_SESSIONS:
        oldest = sorted(_SESSIONS.items(), key=lambda item: item[1].updated_at)
        for key, _state in oldest[:len(_SESSIONS) - _MAX_ACTIVE_SESSIONS]:
            _SESSIONS.pop(key, None)


def _session_state(session_id: str | None = None) -> SessionState:
    key = _normalize_session_id(session_id)
    now = time.time()
    with _STATE_LOCK:
        _prune_sessions(now)
        state = _SESSIONS.get(key)
        if state is None:
            state = SessionState()
            _SESSIONS[key] = state
        state.updated_at = now
        return state


def _state_messages(session_id: str | None = None) -> list[dict]:
    state = _session_state(session_id)
    with state.lock:
        return list(state.messages)


def _state_messages_with_user(session_id: str | None, content: str) -> list[dict]:
    messages = _state_messages(session_id)
    messages.append({"role": "user", "content": content})
    return messages


def _append_message(role: str, content: str, session_id: str | None = None) -> None:
    state = _session_state(session_id)
    with state.lock:
        state.messages.append({"role": role, "content": content})
        state.version += 1
        state.updated_at = time.time()


def _clear_messages(session_id: str | None = None) -> None:
    state = _session_state(session_id)
    with state.lock:
        state.messages.clear()
        state.version += 1
        state.updated_at = time.time()


def reset_chat_context(scope: str | None = None) -> None:
    _clear_messages(scope)
    clear_attention(scope)
    clear_conversation_state(scope)


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


def _json_line(payload: dict) -> bytes:
    return orjson.dumps(payload) + b"\n"


def _log(label: str, text: str = "") -> None:
    value = str(text or "")
    if len(value) > 500:
        value = value[:500].rstrip() + " [trimmed]"
    print(f"[Akane:{label}]{' ' + value if value else ''}", flush=True)


def _static_response_path(route: str) -> tuple[Path, str] | None:
    if route in _STATIC_ROUTES:
        name, media_type = _STATIC_ROUTES[route]
        return STATIC_DIR / name, media_type
    asset = resolve_ui_asset(route)
    return (asset, "image/png") if asset else None


def _generate_reply(
    user_input: str,
    *,
    skip_memory: bool = False,
    session_id: str | None = None,
) -> str:
    session = _normalize_session_id(session_id)
    _log("ingress", f"session={session} stream=0 chars={len(user_input)}")
    try:
        _start_model_loading()
        prepared = prepare_reply(user_input, skip_memory=skip_memory, session_id=session_id)
        reply = generate_reply(prepared)
    except Exception as exc:
        _log("error", f"session={session} type={type(exc).__name__}")
        raise
    _append_message("user", user_input, session_id)
    _append_message("assistant", reply, session_id)
    _log("complete", f"session={session} reply_chars={len(reply)}")
    return reply


def _stream_chat_events(
    text: str,
    *,
    skip_memory: bool = False,
    session_id: str | None = None,
):
    session = _normalize_session_id(session_id)
    _log("ingress", f"session={session} stream=1 chars={len(text)}")
    try:
        _start_model_loading()
        prepared = prepare_reply(text, skip_memory=skip_memory, session_id=session_id)
        yield _json_line({"type": "start", "messages": _state_messages_with_user(session_id, text)})
        for event_type, content in stream_reply(prepared):
            if event_type == "delta":
                yield _json_line({"type": "delta", "content": content, "append": True})
                continue
            _append_message("user", text, session_id)
            _append_message("assistant", content, session_id)
            _log("complete", f"session={session} reply_chars={len(content)}")
            yield _json_line({"type": "done", "reply": content, "messages": _state_messages(session_id)})
    except Exception as exc:
        _log("error", f"session={session} type={type(exc).__name__}")
        yield _json_line({"type": "error", "error": str(exc)})


def _debug_state_report(session_id: str | None) -> str:
    return debug_state_report(session_id)


def handle_builtin_command(text: str, scope: str | None = None) -> dict | None:
    if text == _DEBUG_STATE_COMMAND:
        return {"reply": _debug_state_report(scope), "ephemeral": True}
    if text == "/reset_chat":
        reset_chat_context(scope)
        return {"reply": "Chat reset.", "messages": [], "notice": "Chat reset."}
    if text == "/clear":
        reset_chat_context(scope)
        return {"reply": "", "messages": [], "notice": "Context cleared."}
    if text == "/memory":
        reload_from_disk()
        reply = format_memory_for_prompt() or "No memories yet."
        return {
            "reply": reply,
            "messages": [*_state_messages(scope), {"role": "assistant", "content": reply}],
            "ephemeral": True,
        }
    return None


def _handle_command(text: str, session_id: str | None = None) -> dict | None:
    return handle_builtin_command(text, session_id)


def _app_state_payload(session_id: str | None = None, *, include_messages: bool = True) -> dict:
    state = _session_state(session_id)
    with state.lock:
        version = state.version
        messages = list(state.messages) if include_messages else None
    payload = {
        "model": ModelManager.get_instance().status(),
        "internal_state": emotional_snapshot(session_id),
        "version": version,
    }
    if messages is not None:
        payload["messages"] = messages
    return payload


def _start_model_loading(*, warm_character: bool = True) -> None:
    if warm_character:
        warm_prompt_cache()
    manager = ModelManager.get_instance()
    status = manager.status()
    if status["loading"] or status["loaded"]:
        return

    def load() -> None:
        try:
            manager.ensure_loaded()
            if warm_character:
                warm_caches()
        except Exception as exc:
            _log("model-error", str(exc))

    threading.Thread(target=load, daemon=True, name="AkaneModelLoader").start()


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
        static = _static_response_path("/")
        if static is None:
            raise HTTPException(status_code=404)
        path, media_type = static
        return FileResponse(path, media_type=media_type)

    @app.get("/health")
    async def health():
        status = ModelManager.get_instance().status()
        return JSONResponse(
            {
                "status": "ok",
                "model_loaded": bool(status.get("loaded")),
                "model_loading": bool(status.get("loading")),
                "vscode": workspace_status(),
            }
        )

    @app.post("/api/vscode/index")
    async def vscode_index(request: Request):
        try:
            content_length = int(request.headers.get("content-length", "0") or 0)
        except ValueError:
            content_length = 0
        if content_length > MAX_REQUEST_BYTES:
            return JSONResponse({"error": "Workspace context is too large."}, status_code=413)
        try:
            body = await request.body()
            if len(body) > MAX_REQUEST_BYTES:
                return JSONResponse(
                    {"error": "Workspace context is too large."},
                    status_code=413,
                )
            status = update_workspace_index(orjson.loads(body))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception:
            return JSONResponse({"error": "Invalid JSON request body."}, status_code=400)
        return JSONResponse({"ok": True, **status})

    @app.post("/api/vscode/context")
    async def vscode_context(request: Request):
        try:
            body = await request.body()
            if len(body) > MAX_REQUEST_BYTES:
                return JSONResponse({"error": "Editor context is too large."}, status_code=413)
            status = update_active_context(orjson.loads(body))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception:
            return JSONResponse({"error": "Invalid JSON request body."}, status_code=400)
        return JSONResponse({"ok": True, **status})

    @app.get("/api/vscode/status")
    async def vscode_status():
        return JSONResponse(workspace_status())

    @app.get("/api/vscode/requests")
    async def vscode_requests():
        return JSONResponse(next_file_request())

    @app.post("/api/vscode/requests/{request_id}")
    async def vscode_request_response(request_id: str, request: Request):
        try:
            body = await request.body()
            if len(body) > MAX_REQUEST_BYTES:
                return JSONResponse({"error": "File response is too large."}, status_code=413)
            result = complete_file_request(request_id, orjson.loads(body))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception:
            return JSONResponse({"error": "Invalid JSON request body."}, status_code=400)
        return JSONResponse(result)

    @app.delete("/api/vscode/context")
    async def vscode_disconnect():
        clear_workspace_context()
        return JSONResponse({"ok": True, "connected": False})

    @app.get("/api/state")
    async def api_state(session_id: str = DEFAULT_SESSION_ID, include_messages: str = "1"):
        return JSONResponse(
            _app_state_payload(
                session_id,
                include_messages=_coerce_bool(include_messages, True),
            )
        )

    @app.get("/api/memory")
    async def api_memory():
        return JSONResponse(get_all())

    @app.get("/api/emotion")
    async def api_emotion(session_id: str = DEFAULT_SESSION_ID):
        return JSONResponse(emotional_snapshot(session_id))

    @app.post("/api/backend")
    async def api_backend(request: Request):
        payload = await _request_payload(request)
        try:
            status = ModelManager.get_instance().switch_backend(
                str(payload.get("backend", "llama_cpp")).strip().lower(),
                local_model_path=str(payload["local_model_path"]) if "local_model_path" in payload else None,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "model": status})

    @app.post("/api/quit")
    async def api_quit():
        threading.Thread(
            target=lambda: (time.sleep(0.15), os._exit(0)),
            daemon=True,
            name="AkaneQuit",
        ).start()
        return JSONResponse({"ok": True}, status_code=202)

    @app.post("/api/chat")
    async def api_chat(request: Request):
        chat = _parse_chat_request(await _request_payload(request))
        if not chat.text:
            return JSONResponse({"error": "Message is empty."}, status_code=400)
        command = handle_builtin_command(chat.text, chat.session_id)
        if command is not None:
            return JSONResponse(command)
        try:
            reply = await asyncio.to_thread(
                _generate_reply,
                chat.text,
                skip_memory=chat.skip_memory,
                session_id=chat.session_id,
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return JSONResponse({"reply": reply, "messages": _state_messages(chat.session_id)})

    @app.post("/api/chat/stream")
    async def api_chat_stream(request: Request):
        payload = await _request_payload(request)
        chat = _parse_chat_request(payload)
        if not chat.text:
            return JSONResponse({"error": "Message is empty."}, status_code=400)
        command = handle_builtin_command(chat.text, chat.session_id)
        if command is not None:
            return JSONResponse(command)
        return StreamingResponse(
            _stream_chat_events(
                chat.text,
                skip_memory=chat.skip_memory,
                session_id=chat.session_id,
            ),
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
        self._server = uvicorn.Server(
            uvicorn.Config(APP, host=host, port=port, log_level="warning", access_log=False)
        )
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


def serve_in_thread(
    host: str = SERVER_HOST,
    port: int = SERVER_PORT,
) -> tuple[BackgroundUvicornServer, threading.Thread]:
    _start_model_loading()
    server = BackgroundUvicornServer(host, port)
    thread = threading.Thread(target=server.run, daemon=True, name="AkaneAPIServer")
    server._thread = thread
    thread.start()
    print(f"Akane background API running at http://{host}:{port}", flush=True)
    return server, thread


if __name__ == "__main__":
    serve()
