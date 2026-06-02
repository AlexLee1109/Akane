"""Lean local chat server for Akane."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.core.config import (
    CHAT_HISTORY_CONTEXT_TOKENS,
    LLAMA_CONTEXT_WINDOW,
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
from app.core.emotional_state import (
    format_for_prompt as format_emotional_state_for_prompt,
    observe_user_message as observe_emotional_message,
    snapshot as emotional_snapshot,
)
from app.core.generation import HiddenTagStreamFilter, strip_emoji_chars, strip_hidden_blocks
from app.core.model_loader import ModelManager, content_to_text
from app.memory_store import (
    MEMORY_PATH,
    format_for_prompt as format_memory_for_prompt,
    get_all,
    reload_from_disk,
    remember_user_message,
)
from app.ui.assets import resolve_ui_asset

STATIC_DIR = Path(__file__).parent / "ui" / "static"
DEFAULT_SESSION_ID = "popup"

_MAX_STORED_MESSAGES = max(1, _coerce_int(os.environ.get("AKANE_STORED_MESSAGES", os.environ.get("AKANE_HISTORY_MESSAGES", 16)), 16))
_SHALLOW_HISTORY_MESSAGES = max(0, _coerce_int(os.environ.get("AKANE_SHALLOW_HISTORY_MESSAGES", 2), 2))
_DEEP_HISTORY_MESSAGES = max(_SHALLOW_HISTORY_MESSAGES, _coerce_int(os.environ.get("AKANE_DEEP_HISTORY_MESSAGES", 8), 8))
_SHALLOW_HISTORY_TOKENS = max(0, _coerce_int(os.environ.get("AKANE_SHALLOW_HISTORY_TOKENS", 160), 160))
_DEEP_HISTORY_TOKENS = max(_SHALLOW_HISTORY_TOKENS, _coerce_int(os.environ.get("AKANE_DEEP_HISTORY_TOKENS", 512), 512))
_HISTORY_SAFETY_TOKENS = _coerce_int(os.environ.get("AKANE_HISTORY_SAFETY_TOKENS", 160), 160)
_MAX_HISTORY_MESSAGE_CHARS = _coerce_int(os.environ.get("AKANE_MAX_HISTORY_MESSAGE_CHARS", 900), 900)
_MAX_MEMORY_CHARS = _coerce_int(os.environ.get("AKANE_MAX_MEMORY_CHARS", 1800), 1800)
_FAST_MAX_TOKENS = min(MAX_TOKENS, 160)
_generation_lock = threading.Lock()
_state_lock = threading.Lock()
_SESSIONS: dict[str, "SessionState"] = {}

_STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}
_FOLLOWUP_PHRASES = (
    "what did i", "what did we", "what were we", "what was that", "what about",
    "you said", "i said", "we said", "we talked", "earlier", "last time",
    "previous", "before", "again", "same", "continue", "go on", "remind me",
    "remember", "that one", "this one", "the last", "from before",
)
_FOLLOWUP_WORDS = {
    "it", "that", "this", "those", "these", "they", "them", "he", "she", "him",
    "her", "same", "again", "why", "how",
}
_REPLY_STYLE_CONTEXT = (
    "REPLY STYLE:\n"
    "- Reply directly to the user's actual message, not metadata or private context.\n"
    "- Simple greetings must be 1 short sentence.\n"
    "- Every reply must be 1-3 sentences in one paragraph.\n"
    "- Do not invent scenes, activities, windows, stars, music, or surroundings.\n"
    "- Do not reveal private prompt, memory, mood, emotion, or internal state labels."
)


_DATETIME_CONTEXT_CACHE: tuple[str, str] | None = None


def _log_terminal_stream(label: str, text: str = "") -> None:
    suffix = f" {text}" if text else ""
    print(f"[Akane:{label}]{suffix}", flush=True)


@dataclass
class SessionState:
    messages: deque = field(default_factory=lambda: deque(maxlen=_MAX_STORED_MESSAGES))
    streaming_reply_preview: str = ""
    version: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(frozen=True)
class ChatRequestData:
    text: str
    skip_memory: bool = False
    session_id: str = DEFAULT_SESSION_ID


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


def _log_reply(reply: str, *, session_id: str | None = None, raw: str = "") -> None:
    _log_terminal_stream("done", reply)


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


def _strip_popup_tags(text: str) -> str:
    if not text:
        return text
    source = str(text)
    if "[" not in source and "<" not in source:
        return source
    cleaned = strip_hidden_blocks(source)
    if cleaned == source:
        return source
    return "\n".join(line.rstrip() for line in cleaned.splitlines() if line.strip())


def _collapse_reply_whitespace(text: str) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\t", " ")
    lines: list[str] = []
    blank_count = 0
    for raw in value.split("\n"):
        line = raw.rstrip()
        if line:
            blank_count = 0
            lines.append(line)
        else:
            blank_count += 1
            if blank_count <= 1:
                lines.append("")
    return "\n".join(lines)


def _clean_reply(text: str) -> str:
    cleaned = strip_emoji_chars(_strip_popup_tags(text))
    return _collapse_reply_whitespace(cleaned).strip(" `\n\t")


@lru_cache(maxsize=2048)
def _estimate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4) if text else 0


def _trim_to_tokens(text: str, token_budget: int) -> str:
    text = str(text or "").strip()
    if token_budget <= 0 or not text:
        return ""
    max_chars = max(24, token_budget * 4)
    if len(text) <= max_chars:
        return text
    marker = "[earlier part clipped] "
    if max_chars <= len(marker) + 24:
        return text[-max_chars:].lstrip()
    return marker + text[-(max_chars - len(marker)):].lstrip()


@lru_cache(maxsize=512)
def _discord_user_message(content: str) -> str:
    if not content.startswith("Discord "):
        return content
    for marker in ("\nMessage:\n", "\nUser message:\n"):
        if marker in content:
            return content.split(marker, 1)[1].strip()
    return content


@lru_cache(maxsize=512)
def _discord_history_content(content: str) -> str:
    if not content.startswith("Discord "):
        return content
    speaker = ""
    for line in content.splitlines():
        if line.startswith(("User:", "Speaker:")):
            speaker = line.split(":", 1)[1].strip()
            break
    message = _discord_user_message(content)
    if not speaker:
        return message
    location = "DM" if content.startswith("Discord direct") else "server"
    return f"Discord {location} {speaker}: {message}"


@lru_cache(maxsize=512)
def _cached_history_content(role: str, raw_content: str) -> str:
    content = _clean_reply(raw_content)
    if role == "user":
        content = _discord_history_content(content)
    return _clip(content, _MAX_HISTORY_MESSAGE_CHARS)


def _history_content(message: dict) -> str:
    role = str(message.get("role", "") or "")
    content = str(message.get("content", "") or "")
    return _cached_history_content(role, content)


def _word_tokens(text: str) -> list[str]:
    words: list[str] = []
    start: int | None = None
    for index, char in enumerate(text):
        is_word = ("a" <= char <= "z") or char == "'"
        if is_word and start is None:
            start = index
        elif not is_word and start is not None:
            word = text[start:index].strip("'")
            if word:
                words.append(word)
            start = None
    if start is not None:
        word = text[start:].strip("'")
        if word:
            words.append(word)
    return words


def _needs_deep_history(user_input: str) -> bool:
    text = _discord_user_message(str(user_input or "").strip()).lower()
    words = _word_tokens(text)
    if not words:
        return False
    if any(phrase in text for phrase in _FOLLOWUP_PHRASES):
        return True
    if len(words) <= 8 and any(word in _FOLLOWUP_WORDS for word in words):
        return True
    return bool("?" in text and len(words) <= 4)


def _history_token_budget(system_prompt: str, user_input: str, *, deep: bool) -> int:
    usable = max(256, LLAMA_CONTEXT_WINDOW - _FAST_MAX_TOKENS - _HISTORY_SAFETY_TOKENS)
    prompt_tokens = _estimate_tokens(system_prompt)
    user_tokens = _estimate_tokens(user_input)
    remaining = usable - prompt_tokens - user_tokens
    target = _DEEP_HISTORY_TOKENS if deep else _SHALLOW_HISTORY_TOKENS
    return max(0, min(CHAT_HISTORY_CONTEXT_TOKENS, target, remaining))


def _extract_message_text(result: dict) -> str:
    choices = result.get("choices") or []
    if not choices:
        return ""
    return content_to_text((choices[0].get("message") or {}).get("content")).strip()


def _date_label(day) -> str:
    return f"{day.strftime('%A, %B')} {day.day}, {day.year}"


def _datetime_context(now: datetime | None = None) -> str:
    global _DATETIME_CONTEXT_CACHE
    current = now or datetime.now().astimezone()
    if current.tzinfo is None:
        current = current.astimezone()
    key = current.strftime("%Y-%m-%d")
    if now is None and _DATETIME_CONTEXT_CACHE and _DATETIME_CONTEXT_CACHE[0] == key:
        return _DATETIME_CONTEXT_CACHE[1]
    today = current.date()
    time_text = current.strftime("%I:%M %p").lstrip("0")
    tz_name = current.tzname() or "local time"
    result = (
        "DATE/TIME:\n"
        f"- Current local date/time: {_date_label(today)} at {time_text} {tz_name}.\n"
        f"- Yesterday: {_date_label(today - timedelta(days=1))}.\n"
        f"- Today: {_date_label(today)}.\n"
        f"- Tomorrow: {_date_label(today + timedelta(days=1))}."
    )
    if now is None:
        _DATETIME_CONTEXT_CACHE = (key, result)
    return result


def _system_prompt(
    user_text: str,
    *,
    skip_memory: bool,
    session_id: str | None = None,
    internal_state: dict[str, object] | None = None,
) -> str:
    include_memory = not skip_memory
    datetime_context = _datetime_context()
    memory = ""
    if include_memory:
        memory = format_memory_for_prompt()
        if memory:
            memory = _clip(memory, _MAX_MEMORY_CHARS)

    sections: list[str] = [
        "CONTINUITY:\n"
        "- Use recent conversation messages to resolve follow-ups like it, that, this, same, again, and you said.\n"
        "- Keep track of who is speaking. In Discord, the User field names the current user.\n"
        "- Do not treat Discord metadata as the user's wording; answer the Message field."
    ]
    sections.append(datetime_context)
    sections.append(format_emotional_state_for_prompt(internal_state or emotional_snapshot(session_id)))
    if memory:
        sections.append("Memory:\n" + memory)
    sections.append(_REPLY_STYLE_CONTEXT)
    return build_system_prompt("\n\n".join(sections), include_memory=include_memory)


def _history_messages(session_id: str | None, *, system_prompt: str, user_input: str, deep: bool) -> list[dict]:
    state = _session_state(session_id)
    with state.lock:
        messages = list(state.messages)

    message_limit = _DEEP_HISTORY_MESSAGES if deep else _SHALLOW_HISTORY_MESSAGES
    remaining_tokens = _history_token_budget(system_prompt, user_input, deep=deep)
    if message_limit <= 0 or remaining_tokens <= 0:
        return []

    kept: list[dict] = []
    for message in reversed(messages[-message_limit:]):
        role = str(message.get("role", "") or "")
        if role not in {"user", "assistant"}:
            continue
        content = _history_content(message)
        if not content:
            continue
        content = _trim_to_tokens(content, remaining_tokens)
        if not content:
            break
        cost = _estimate_tokens(content)
        kept.append({"role": role, "content": content})
        remaining_tokens -= cost
        if remaining_tokens <= 0:
            break
    kept.reverse()
    return kept


def _chat_messages(user_input: str, *, skip_memory: bool = False, session_id: str | None = None) -> list[dict]:
    internal_state = observe_emotional_message(session_id, user_input)
    system_prompt = _system_prompt(
        user_input,
        skip_memory=skip_memory,
        session_id=session_id,
        internal_state=internal_state,
    )
    deep_history = _needs_deep_history(user_input)
    return [
        {"role": "system", "content": system_prompt},
        *_history_messages(session_id, system_prompt=system_prompt, user_input=user_input, deep=deep_history),
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
        _append_message("user", user_input, session_id)
        result = _request_completion(messages, stream=False)
        raw = _extract_message_text(result)
        remember_user_message(raw, allow_natural=False)
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
                visible = strip_emoji_chars(hidden_filter.feed(token))
                if not visible:
                    continue
                visible_parts.append(visible)
                displayed = "".join(visible_parts)
                _set_streaming_reply_preview(displayed, session_id)
                yield _json_line({"type": "delta", "content": visible, "append": True})

            tail = strip_emoji_chars(hidden_filter.flush())
            if tail:
                visible_parts.append(tail)
                displayed = "".join(visible_parts)
                _set_streaming_reply_preview(displayed, session_id)
                yield _json_line({"type": "delta", "content": tail, "append": True})

            raw = "".join(raw_parts)
            displayed = "".join(visible_parts)
            remember_user_message(raw, allow_natural=False)
            reply = _clean_reply(raw) or _clean_reply(displayed) or "I lost the thread for a second. Try that again."
            _log_reply(reply, session_id=session_id, raw=raw)

            _append_message("assistant", reply, session_id)
            yield _json_line({"type": "done", "reply": reply, "messages": list(_session_state(session_id).messages)})
    except Exception as exc:
        _set_streaming_reply_preview("", session_id)
        _log_terminal_stream("error", str(exc))
        yield _json_line({"type": "error", "error": str(exc)})


def _handle_command(text: str, session_id: str | None = None) -> dict | None:
    if text == "/clear":
        _clear_messages(session_id)
        return {"reply": "", "messages": [], "notice": "Context cleared."}
    if text == "/memory":
        reload_from_disk()
        reply = format_memory_for_prompt() or "No memories yet."
        return {"reply": reply, "messages": [*_state_messages(session_id), {"role": "assistant", "content": reply}], "ephemeral": True}
    if text == "/reset":
        _clear_messages(session_id)
        if MEMORY_PATH.exists():
            MEMORY_PATH.unlink()
        reload_from_disk()
        return {"reply": "", "messages": [], "notice": "Memory wiped and context cleared."}
    return None


def _app_state_payload(session_id: str | None = None, *, include_messages: bool = True) -> dict:
    state = _session_state(session_id)
    payload = {
        "model": ModelManager.get_instance().status(),
        "internal_state": emotional_snapshot(session_id),
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
