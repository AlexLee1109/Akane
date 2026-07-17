"""FastAPI host for Akane's shared text-chat runtime and local UI."""

from __future__ import annotations

import asyncio
import secrets
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import orjson
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.core.character import load_character_profile
from app.core.config import (
    CORS_ALLOWED_ORIGINS,
    SERVER_API_TOKEN,
    SERVER_HOST,
    SERVER_PORT,
    _coerce_bool,
)
from app.core.memory import get_internal_state_store, get_memory_store
from app.core.model_loader import ModelManager
from app.core.reply_pipeline import (
    GenerationEvent,
    debug_state_report,
    generate_reply,
    prepare_reply,
    stream_reply,
)
from app.core.session import (
    ChatInput,
    GenerationBusyError,
    GenerationCancelled,
    GenerationQueueFullError,
    cancel_all_generations,
    cancel_generation,
    commit_turn,
    finish_turn,
    forget_profile,
    normalize_chat_input,
    reset_conversation,
    session_state_snapshot,
)
from app.integrations.vscode_context import active_file_reply
from app.integrations.vscode_workspace import (
    MAX_REQUEST_BYTES,
    ReviewDecision,
    claim_editor_review,
    clear_workspace_context,
    update_editor_context,
    workspace_status,
)
from app.ui.assets import resolve_ui_asset

STATIC_DIR = Path(__file__).parent / "ui" / "static"
DEFAULT_PROFILE_ID = "local:owner"
DEFAULT_CONVERSATION_ID = "popup:default"
_DEBUG_STATE_COMMAND = "/debug_state"
_MODEL_LOAD_LOCK = threading.Lock()
_MODEL_LOAD_THREAD: threading.Thread | None = None
_STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


@dataclass(frozen=True, slots=True)
class ChatRequestData:
    chat_input: ChatInput
    skip_memory: bool = False
    skip_if_busy: bool = False


def _messages(conversation_id: str, profile_id: str) -> list[dict[str, str]]:
    return get_memory_store().messages(conversation_id, profile_id)


def _messages_with_user(chat_input: ChatInput) -> list[dict[str, str]]:
    return [
        *_messages(chat_input.conversation_id, chat_input.profile_id),
        {"role": "user", "content": chat_input.text},
    ]


async def _request_payload(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_chat_request(payload: dict) -> ChatRequestData:
    source = str(payload.get("source") or "web")
    default_profile = DEFAULT_PROFILE_ID
    default_conversation = DEFAULT_CONVERSATION_ID
    return ChatRequestData(
        chat_input=normalize_chat_input(
            text=payload.get("message", ""),
            profile_id=payload.get("profile_id", default_profile),
            conversation_id=payload.get("conversation_id", payload.get("session_id", default_conversation)),
            source=source,
            timestamp=payload.get("timestamp", 0.0),
            display_name=payload.get("display_name", ""),
            reply_context=payload.get("reply_context", ""),
            group_conversation=_coerce_bool(payload.get("group_conversation", False)),
            autonomous=_coerce_bool(payload.get("autonomous", False)),
        ),
        skip_memory=_coerce_bool(payload.get("skip_memory", False)),
        skip_if_busy=_coerce_bool(payload.get("skip_if_busy", False)),
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


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, (GenerationBusyError, GenerationQueueFullError, GenerationCancelled, ValueError)):
        return str(exc)
    message = str(exc).lower()
    if "model" in message or "llama" in message or "gguf" in message:
        return "Akane's local model is unavailable. Check the configured GGUF and server logs."
    return "Akane could not finish that reply. Check the server logs for details."


def _error_status(exc: Exception) -> int:
    if isinstance(exc, GenerationBusyError):
        return 409
    if isinstance(exc, GenerationQueueFullError):
        return 503
    if isinstance(exc, ValueError):
        return 400
    return 500


def _generate_reply(chat: ChatRequestData) -> str:
    item = chat.chat_input
    direct_reply = active_file_reply(item.text)
    prepared = prepare_reply(
        item,
        skip_memory=chat.skip_memory,
        skip_if_busy=chat.skip_if_busy,
    )
    if direct_reply is not None:
        try:
            commit_turn(prepared, direct_reply)
            return direct_reply
        finally:
            finish_turn(prepared)
    reply = generate_reply(prepared)
    return reply


def _review_vscode_context(decision: ReviewDecision | None = None) -> str:
    decision = decision or claim_editor_review()
    snapshot = decision.snapshot
    if not decision.accepted or snapshot is None:
        return ""
    project = f"vscode:project:{snapshot.project_id}"
    request = ChatRequestData(
        chat_input=normalize_chat_input(
            text=(
                f"Review the current file because {decision.reason}. Give one short, concrete "
                "warning only when the supplied read-only context supports a real bug, broken "
                "build, risky change, or project contradiction. Otherwise reply exactly "
                "[SILENT]. Do not claim to edit files, run commands, or execute tests."
            ),
            profile_id=project,
            conversation_id=f"{project}:reviews",
            source="popup",
        ),
        skip_memory=False,
        skip_if_busy=True,
    )
    reply = _generate_reply(request).strip()
    return "" if reply.upper().rstrip(".! ") == "[SILENT]" else reply


def _stream_chat_events(chat: ChatRequestData):
    item = chat.chat_input
    direct_reply = active_file_reply(item.text)
    if direct_reply is not None:
        prepared = prepare_reply(
            item,
            skip_memory=chat.skip_memory,
            skip_if_busy=chat.skip_if_busy,
        )
        try:
            yield _json_line(
                {
                    "type": "start",
                    "generation_id": prepared.generation_id,
                    "messages": _messages_with_user(item),
                }
            )
            yield _event_json(
                GenerationEvent("delta", prepared.generation_id, text=direct_reply),
                item.conversation_id,
                item.profile_id,
            )
            commit_turn(prepared, direct_reply)
            yield _event_json(
                GenerationEvent("done", prepared.generation_id, reply=direct_reply),
                item.conversation_id,
                item.profile_id,
            )
        finally:
            finish_turn(prepared)
        return
    prepared = None
    try:
        prepared = prepare_reply(
            item,
            skip_memory=chat.skip_memory,
            skip_if_busy=chat.skip_if_busy,
        )
        yield _json_line(
            {
                "type": "start",
                "generation_id": prepared.generation_id,
                "messages": _messages_with_user(item),
            }
        )
        for event in stream_reply(prepared):
            yield _event_json(event, item.conversation_id, item.profile_id)
        prepared = None
    except GenerationCancelled:
        generation_id = prepared.generation_id if prepared else ""
        _log("cancelled", f"conversation={item.conversation_id}")
        yield _json_line({"type": "cancelled", "generation_id": generation_id})
    except Exception as exc:
        _log("error", f"conversation={item.conversation_id} type={type(exc).__name__} detail={exc}")
        yield _json_line(
            {
                "type": "error",
                "generation_id": prepared.generation_id if prepared else "",
                "error": _safe_error(exc),
            }
        )
    finally:
        if prepared is not None:
            finish_turn(prepared)


def _event_json(event: GenerationEvent, conversation_id: str, profile_id: str) -> bytes:
    if event.kind == "delta":
        return _json_line(
            {
                "type": "delta",
                "generation_id": event.generation_id,
                "content": event.text,
            }
        )
    return _json_line(
        {
            "type": "done",
            "generation_id": event.generation_id,
            "reply": event.reply,
            "messages": _messages(conversation_id, profile_id),
            "metadata": event.metadata or {},
        }
    )


def handle_builtin_command(chat_input: ChatInput) -> dict | None:
    text = chat_input.text
    if text == _DEBUG_STATE_COMMAND:
        return {
            "reply": debug_state_report(chat_input.conversation_id, chat_input.profile_id),
            "ephemeral": True,
        }
    if text in {"/reset_chat", "/clear"}:
        reset_conversation(chat_input.conversation_id, chat_input.profile_id)
        notice = "Chat reset." if text == "/reset_chat" else "Context cleared."
        return {"reply": "" if text == "/clear" else notice, "messages": [], "notice": notice}
    if text == "/forget_me":
        cancel_generation(chat_input.conversation_id, chat_input.profile_id)
        forget_profile(chat_input.profile_id)
        return {"reply": "", "messages": [], "notice": "Profile memory cleared."}
    return None


def _app_state_payload(
    conversation_id: str,
    profile_id: str,
    *,
    include_messages: bool = True,
) -> dict:
    conversation = get_memory_store().public_conversation(conversation_id, profile_id)
    payload = {
        "model": ModelManager.get_instance().status(),
        "internal_state": session_state_snapshot(conversation_id, profile_id),
        "version": int(float(conversation.get("updated_at") or 0.0) * 1000),
    }
    if include_messages:
        payload["messages"] = _messages(conversation_id, profile_id)
    return payload


def _start_model_loading() -> None:
    global _MODEL_LOAD_THREAD
    manager = ModelManager.get_instance()
    status = manager.status()
    if status["loading"] or status["loaded"] or status["error"]:
        return
    with _MODEL_LOAD_LOCK:
        if _MODEL_LOAD_THREAD is not None and _MODEL_LOAD_THREAD.is_alive():
            return

        def load() -> None:
            try:
                manager.ensure_loaded()
            except Exception as exc:
                _log("model-error", str(exc))

        _MODEL_LOAD_THREAD = threading.Thread(
            target=load,
            daemon=True,
            name="AkaneModelLoader",
        )
        _MODEL_LOAD_THREAD.start()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    load_character_profile()
    get_memory_store()
    get_internal_state_store()
    _start_model_loading()
    try:
        yield
    finally:
        cancel_all_generations()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Akane API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ALLOWED_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Akane-Token"],
    )

    @app.middleware("http")
    async def authenticate_api(request: Request, call_next):
        if (
            SERVER_API_TOKEN
            and request.method != "OPTIONS"
            and request.url.path.startswith("/api/")
        ):
            bearer = request.headers.get("authorization", "")
            supplied = bearer[7:].strip() if bearer.lower().startswith("bearer ") else ""
            supplied = supplied or request.headers.get("x-akane-token", "")
            if not secrets.compare_digest(supplied, SERVER_API_TOKEN):
                return JSONResponse({"error": "Unauthorized."}, status_code=401)
        return await call_next(request)

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
                "vscode_connected": bool(workspace_status().get("connected")),
            }
        )

    @app.post("/api/vscode/context")
    async def vscode_context(request: Request):
        body = await request.body()
        if len(body) > MAX_REQUEST_BYTES:
            return JSONResponse({"error": "Editor context is too large."}, status_code=413)
        try:
            status = update_editor_context(orjson.loads(body))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception:
            return JSONResponse({"error": "Invalid JSON request body."}, status_code=400)
        decision = claim_editor_review()
        if decision.accepted:
            try:
                suggestion = await asyncio.to_thread(_review_vscode_context, decision)
            except Exception as exc:
                _log("vscode-review-error", f"type={type(exc).__name__} detail={exc}")
            else:
                if suggestion:
                    status["suggestion"] = suggestion
        return JSONResponse({"ok": True, **status})

    @app.get("/api/vscode/status")
    async def vscode_status():
        return JSONResponse(workspace_status())

    @app.delete("/api/vscode/context")
    async def vscode_disconnect():
        clear_workspace_context()
        return JSONResponse({"ok": True, "connected": False})

    @app.get("/api/state")
    async def api_state(
        conversation_id: str = DEFAULT_CONVERSATION_ID,
        profile_id: str = DEFAULT_PROFILE_ID,
        session_id: str = "",
        include_messages: str = "1",
    ):
        conversation = session_id or conversation_id
        return JSONResponse(
            _app_state_payload(
                conversation,
                profile_id,
                include_messages=_coerce_bool(include_messages, True),
            )
        )

    @app.post("/api/chat/cancel")
    async def api_cancel(request: Request):
        payload = await _request_payload(request)
        conversation = str(
            payload.get("conversation_id") or payload.get("session_id") or DEFAULT_CONVERSATION_ID
        )
        profile = str(payload.get("profile_id") or DEFAULT_PROFILE_ID)
        return JSONResponse(
            {"ok": True, "cancelled": cancel_generation(conversation, profile)}
        )

    @app.post("/api/chat")
    async def api_chat(request: Request):
        try:
            chat = _parse_chat_request(await _request_payload(request))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        command = handle_builtin_command(chat.chat_input)
        if command is not None:
            return JSONResponse(command)
        try:
            reply = await asyncio.to_thread(_generate_reply, chat)
        except Exception as exc:
            _log("error", f"type={type(exc).__name__} detail={exc}")
            return JSONResponse({"error": _safe_error(exc)}, status_code=_error_status(exc))
        return JSONResponse(
            {
                "reply": reply,
                "messages": _messages(
                    chat.chat_input.conversation_id,
                    chat.chat_input.profile_id,
                ),
            }
        )

    @app.post("/api/chat/stream")
    async def api_chat_stream(request: Request):
        try:
            chat = _parse_chat_request(await _request_payload(request))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        command = handle_builtin_command(chat.chat_input)
        if command is not None:
            return JSONResponse(command)
        return StreamingResponse(
            _stream_chat_events(chat),
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
        cancel_all_generations()
        self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)


def serve(host: str = SERVER_HOST, port: int = SERVER_PORT) -> None:
    print(f"Akane web chat running at http://{host}:{port}", flush=True)
    try:
        BackgroundUvicornServer(host, port).run()
    except KeyboardInterrupt:
        cancel_all_generations()
        print("\nStopping Akane web chat...", flush=True)


def serve_in_thread(
    host: str = SERVER_HOST,
    port: int = SERVER_PORT,
) -> tuple[BackgroundUvicornServer, threading.Thread]:
    server = BackgroundUvicornServer(host, port)
    thread = threading.Thread(target=server.run, daemon=True, name="AkaneAPIServer")
    server._thread = thread
    thread.start()
    print(f"Akane background API running at http://{host}:{port}", flush=True)
    return server, thread


if __name__ == "__main__":
    serve()
