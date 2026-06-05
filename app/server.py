"""Lean local chat server for Akane."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import islice
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
from app.core.character import build_system_prompt, prompt_cache_status, prompt_section_lengths
from app.core.attention_state import (
    clear_attention,
    decay_attention,
    format_attention_for_prompt as format_attention_state_for_prompt,
    get_attention_state,
    observe_attention,
)
from app.core.conversation_state import (
    format_recent_wording_avoidance,
    format_conversation_context,
    get_conversation_state,
    observe_turn as observe_conversation_turn,
    reset_conversation_state,
)
from app.core.emotional_state import (
    format_for_prompt as format_emotional_state_for_prompt,
    observe_user_message as observe_emotional_message,
    persist as persist_emotional_state,
    snapshot as emotional_snapshot,
)
from app.core.generation import HiddenTagStreamFilter, strip_emoji_chars, strip_hidden_blocks
from app.core.model_loader import ModelManager, content_to_text
from app.core.reply_pipeline import build_runtime_context as build_core_runtime_context
from app.core.reply_pipeline import clean_reply_text, content_delta
from app.memory_store import (
    format_for_prompt as format_memory_for_prompt,
    get_all,
    reload_from_disk,
    remember_exchange,
)
from app.ui.assets import resolve_ui_asset

STATIC_DIR = Path(__file__).parent / "ui" / "static"
DEFAULT_SESSION_ID = "popup"

_MAX_STORED_MESSAGES = max(1, _coerce_int(os.environ.get("AKANE_STORED_MESSAGES", os.environ.get("AKANE_HISTORY_MESSAGES", 16)), 16))
_SHALLOW_HISTORY_TOKENS = max(0, _coerce_int(os.environ.get("AKANE_SHALLOW_HISTORY_TOKENS", 160), 160))
_HISTORY_SAFETY_TOKENS = _coerce_int(os.environ.get("AKANE_HISTORY_SAFETY_TOKENS", 160), 160)
_MAX_HISTORY_MESSAGE_CHARS = _coerce_int(os.environ.get("AKANE_MAX_HISTORY_MESSAGE_CHARS", 900), 900)
_MAX_MEMORY_CHARS = _coerce_int(os.environ.get("AKANE_MAX_MEMORY_CHARS", 1800), 1800)
_MEMORY_PROMPT_CHARS = min(_MAX_MEMORY_CHARS, 600)
MAX_RAW_HISTORY_MESSAGES = 4
_MAX_ACTIVE_SESSIONS = 64
_SESSION_STALE_SECONDS = 12 * 3600
_RUNTIME_CONTEXT_CHARS = 2000
_RUNTIME_CONTEXT_PROMPT_CHARS = min(_RUNTIME_CONTEXT_CHARS, 1100)
_DEBUG_STATE_COMMAND = "/debug_state"
_state_lock = threading.Lock()
_SESSIONS: dict[str, "SessionState"] = {}

_STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def _log_terminal_stream(label: str, text: str = "") -> None:
    value = str(text or "")
    if len(value) > 500:
        value = value[:500].rstrip() + " [trimmed]"
    suffix = f" {value}" if value else ""
    print(f"[Akane:{label}]{suffix}", flush=True)


def _timing_enabled() -> bool:
    return str(os.environ.get("AKANE_TIMING", "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class SessionState:
    messages: deque = field(default_factory=lambda: deque(maxlen=_MAX_STORED_MESSAGES))
    version: int = 0
    updated_at: float = field(default_factory=time.time)
    runtime_context_chars: int = 0
    prompt_chars: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(frozen=True)
class ChatRequestData:
    text: str
    skip_memory: bool = False
    session_id: str = DEFAULT_SESSION_ID


def _normalize_session_id(session_id: str | None) -> str:
    value = str(session_id or "").strip()
    return (value or DEFAULT_SESSION_ID)[:120]


def _prune_sessions_locked(now: float) -> None:
    if len(_SESSIONS) <= _MAX_ACTIVE_SESSIONS:
        return
    for key, state in list(_SESSIONS.items()):
        if now - state.updated_at > _SESSION_STALE_SECONDS:
            _SESSIONS.pop(key, None)
    extra = len(_SESSIONS) - _MAX_ACTIVE_SESSIONS
    if extra > 0:
        old = sorted(_SESSIONS.items(), key=lambda item: item[1].updated_at)
        for key, _state in old[:extra]:
            _SESSIONS.pop(key, None)


def _session_state(session_id: str | None = None) -> SessionState:
    key = _normalize_session_id(session_id)
    now = time.time()
    with _state_lock:
        _prune_sessions_locked(now)
        state = _SESSIONS.get(key)
        if state is None:
            state = SessionState()
            _SESSIONS[key] = state
        state.updated_at = now
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


def _log_timing(**values: float | int) -> None:
    if not _timing_enabled():
        return
    parts = []
    for key, value in values.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}s")
        else:
            parts.append(f"{key}={value}")
    print(f"[Akane:timing] {' '.join(parts)}", flush=True)


def _log_ctx(**values: int) -> None:
    if not _timing_enabled():
        return
    print("[Akane:ctx] " + " ".join(f"{key}={value}" for key, value in values.items()), flush=True)


def _log_summary(*, used: bool, chars: int, updated_after_stream: bool) -> None:
    if not _timing_enabled():
        return
    used_text = "true" if used else "false"
    updated_text = "true" if updated_after_stream else "false"
    print(f"[Akane:summary] used={used_text} chars={chars} updated_after_stream={updated_text}", flush=True)


def _log_cache(**values: str) -> None:
    if not _timing_enabled():
        return
    print("[Akane:cache] " + " ".join(f"{key}={value}" for key, value in values.items()), flush=True)


def _log_stream(message: str) -> None:
    if _timing_enabled():
        print(f"[Akane:stream] {message}", flush=True)


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[trimmed]"


def _debug_clip(text: object, limit: int) -> str:
    value = strip_hidden_blocks(str(text or "")).replace("@", "@\u200b")
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or value[:limit]


def _debug_float(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _static_response_path(route: str) -> tuple[Path, str] | None:
    if route in _STATIC_ROUTES:
        name, media_type = _STATIC_ROUTES[route]
        return STATIC_DIR / name, media_type
    asset = resolve_ui_asset(route)
    return (asset, "image/png") if asset else None


def _state_messages(session_id: str | None = None) -> list[dict]:
    state = _session_state(session_id)
    with state.lock:
        return list(state.messages)


def _state_messages_with_user(session_id: str | None, content: str) -> list[dict]:
    state = _session_state(session_id)
    with state.lock:
        return [*state.messages, {"role": "user", "content": content}]


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
    reset_conversation_state(scope)


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


def _clean_reply(text: str) -> str:
    return clean_reply_text(text)


def _estimate_tokens(text: str) -> int:
    return max(1, (len(str(text or "")) + 3) // 4) if text else 0


def _message_token_cost(message: dict) -> int:
    return 6 + _estimate_tokens(str(message.get("role", ""))) + _estimate_tokens(str(message.get("content", "")))


def _messages_token_cost(messages: list[dict]) -> int:
    return 4 + sum(_message_token_cost(message) for message in messages)


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


def _discord_user_message(content: str) -> str:
    if not content.startswith("Discord "):
        return content
    for marker in ("\nMessage:\n", "\nUser message:\n"):
        if marker in content:
            return content.split(marker, 1)[1].strip()
    return content


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


def _clean_history_content(role: str, raw_content: str) -> str:
    content = _clean_reply(raw_content)
    if role == "user":
        content = _discord_history_content(content)
    return _clip(content, _MAX_HISTORY_MESSAGE_CHARS)


def _history_content(message: dict) -> str:
    role = str(message.get("role", "") or "")
    content = str(message.get("content", "") or "")
    return _clean_history_content(role, content)


def _low_history_content(text: str) -> bool:
    value = " ".join(str(text or "").lower().strip(".,!?;:()[]{}\"'`").split())
    return value in {"hello", "hi", "hey", "yo", "lol", "thanks", "thank you", "ok", "okay"}


def _memory_terms(user_text: str) -> set[str]:
    text = _discord_user_message(str(user_text or "")).lower()
    return {part.strip(".,!?;:()[]{}\"'`") for part in text.split() if len(part) >= 4}


def _history_token_budget(system_prompt: str, user_input: str) -> int:
    usable = max(256, LLAMA_CONTEXT_WINDOW - MAX_TOKENS - _HISTORY_SAFETY_TOKENS)
    prompt_tokens = _estimate_tokens(system_prompt)
    user_tokens = _estimate_tokens(user_input)
    remaining = usable - prompt_tokens - user_tokens
    return max(0, min(CHAT_HISTORY_CONTEXT_TOKENS, _SHALLOW_HISTORY_TOKENS, remaining))


def _memory_context(user_text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    memory = format_memory_for_prompt()
    if not memory:
        return ""

    terms = _memory_terms(user_text)
    if terms:
        blocks = []
        for block in memory.split("\n\n"):
            lower = block.lower()
            if block.startswith("User Preferences:") or any(term in lower for term in terms):
                blocks.append(block)
        if blocks:
            memory = "\n\n".join(blocks)
    return _clip(memory, limit)


def _fit_context(messages: list[dict], max_tokens: int, *, reduce_max_tokens: bool = True) -> tuple[list[dict], int, bool]:
    fitted = list(messages)
    max_tokens = max(1, min(int(max_tokens), MAX_TOKENS))
    soft_limit = max(64, LLAMA_CONTEXT_WINDOW - _HISTORY_SAFETY_TOKENS)
    hard_limit = max(2, LLAMA_CONTEXT_WINDOW - 8)
    prompt_tokens = _messages_token_cost(fitted)

    while len(fitted) > 2 and prompt_tokens + max_tokens > soft_limit:
        removed = fitted.pop(1)
        prompt_tokens -= _message_token_cost(removed)

    fits = prompt_tokens + max_tokens <= soft_limit
    if not fits and reduce_max_tokens:
        max_tokens = max(1, min(max_tokens, hard_limit - prompt_tokens))
        fits = prompt_tokens + max_tokens <= hard_limit
    return fitted, max_tokens, fits


def _extract_message_text(result: dict) -> str:
    choices = result.get("choices") or []
    if not choices:
        return ""
    return content_to_text((choices[0].get("message") or {}).get("content")).strip()


def _wording_to_avoid_context(session_id: str | None) -> str:
    return format_recent_wording_avoidance(session_id)


def _build_runtime_context(
    *,
    state_text: str,
    attention_text: str = "",
    conversation_context: str = "",
    wording_to_avoid: str = "",
    memory_text: str = "",
) -> tuple[str, dict[str, int]]:
    result = build_core_runtime_context(
        tone_text=state_text,
        attention_text=attention_text,
        conversation_context=conversation_context,
        wording_to_avoid=wording_to_avoid,
        memory_text=memory_text,
        target_chars=_RUNTIME_CONTEXT_PROMPT_CHARS,
        hard_chars=_RUNTIME_CONTEXT_CHARS,
    )
    lengths = dict(result.lengths)
    lengths["message_context"] = 0
    return result.text, lengths


def _system_prompt(*, include_memory: bool, runtime_context: str) -> str:
    return build_system_prompt(runtime_context, include_memory=include_memory)


def _history_messages(session_id: str | None, *, system_prompt: str, user_input: str) -> list[dict]:
    message_limit = MAX_RAW_HISTORY_MESSAGES
    remaining_tokens = _history_token_budget(system_prompt, user_input)
    if message_limit <= 0 or remaining_tokens <= 0:
        return []

    state = _session_state(session_id)
    with state.lock:
        start = max(0, len(state.messages) - message_limit)
        messages = list(islice(state.messages, start, None))

    kept: list[dict] = []
    for message in reversed(messages[-message_limit:]):
        role = str(message.get("role", "") or "")
        if role not in {"user", "assistant"}:
            continue
        content = _history_content(message)
        if not content:
            continue
        if _low_history_content(content):
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


def _chat_messages(
    user_input: str,
    *,
    skip_memory: bool = False,
    session_id: str | None = None,
) -> tuple[list[dict], int]:
    internal_state = observe_emotional_message(session_id, user_input, persist=False)
    state_text = format_emotional_state_for_prompt(internal_state)
    attention_state = observe_attention(session_id, _discord_user_message(user_input))
    attention_text = format_attention_state_for_prompt(attention_state)
    conversation_context = format_conversation_context(session_id, include_focus=False)
    _log_summary(used=bool(conversation_context), chars=len(conversation_context), updated_after_stream=False)
    wording_to_avoid = _wording_to_avoid_context(session_id)
    max_tokens = MAX_TOKENS
    full_memory = "" if skip_memory else _memory_context(user_input, _MEMORY_PROMPT_CHARS)
    memory_options = [full_memory] if full_memory else [""]
    if full_memory:
        memory_options.append("")

    include_memory = not skip_memory
    for memory_text in memory_options:
        timing = _timing_enabled()
        cache_status = prompt_cache_status(include_memory) if timing else {}
        runtime_context, runtime_lengths = _build_runtime_context(
            state_text=state_text,
            attention_text=attention_text,
            conversation_context=conversation_context,
            wording_to_avoid=wording_to_avoid,
            memory_text=memory_text,
        )
        system_prompt = _system_prompt(include_memory=include_memory, runtime_context=runtime_context)
        history = _history_messages(
            session_id,
            system_prompt=system_prompt,
            user_input=user_input,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_input},
        ]
        fitted, fitted_max_tokens, fits = _fit_context(messages, max_tokens, reduce_max_tokens=not memory_text)
        if fits or not memory_text:
            history_chars = sum(len(str(message.get("content", "") or "")) for message in fitted[1:-1])
            user_chars = len(str(user_input or ""))
            total_chars = sum(len(str(message.get("content", "") or "")) for message in fitted)
            session_state = _session_state(session_id)
            with session_state.lock:
                session_state.runtime_context_chars = runtime_lengths["runtime_context"]
                session_state.prompt_chars = total_chars
            if timing:
                base_lengths = prompt_section_lengths(include_memory)
                _log_cache(
                    **cache_status,
                    runtime_context="built",
                    memory="used" if memory_text else "skip",
                    history="bounded",
                )
                _log_ctx(
                    runtime_rules=base_lengths.get("runtime_rules", 0),
                    soul=base_lengths.get("soul", 0),
                    identity=base_lengths.get("identity", 0),
                    memory_rules=base_lengths.get("memory_rules", 0),
                    summary=runtime_lengths["summary"],
                    avoid=runtime_lengths["avoid"],
                    memory=runtime_lengths["memory"],
                    mood=runtime_lengths["mood"],
                    attention=runtime_lengths["attention"],
                    message_context=runtime_lengths["message_context"],
                    history=history_chars,
                    user=user_chars,
                    runtime_context=runtime_lengths["runtime_context"],
                    total=total_chars,
                )
            return fitted, fitted_max_tokens

    raise RuntimeError("Could not fit prompt into context window.")


def _request_completion(messages: list[dict], *, stream: bool, max_tokens: int):
    return ModelManager.get_instance().create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repeat_penalty=REPETITION_PENALTY,
        stream=stream,
    )


def _debug_state_report(session_id: str | None) -> str:
    mood = emotional_snapshot(session_id)
    focus = get_attention_state(session_id)
    context = get_conversation_state(session_id)
    session = _session_state(session_id)
    with session.lock:
        runtime_context_chars = session.runtime_context_chars
        prompt_chars = session.prompt_chars
    avoidance = format_recent_wording_avoidance(session_id)
    prefix = ModelManager.get_instance().prefix_cache_status()
    variables = mood.get("variables") if isinstance(mood.get("variables"), dict) else mood

    lines = [
        "Akane debug state",
        "",
        "Mood:",
        f"- mood: {_debug_clip(mood.get('mood'), 40)} ({_debug_float(mood.get('mood_intensity'))})",
        f"- emotion: {_debug_clip(mood.get('emotion'), 40)} ({_debug_float(mood.get('emotion_intensity'))})",
        f"- turns: {int(mood.get('emotion_turns') or 0)}",
    ]
    for name in ("energy", "social", "focus", "comfort", "stimulation", "tension", "playfulness", "affection"):
        lines.append(f"- {name}: {_debug_float(variables.get(name) if isinstance(variables, dict) else None)}")

    lines.extend(["", "Focus:"])
    if focus:
        lines.extend([
            f"- topic: {_debug_clip(focus.get('topic'), 100)}",
            f"- summary: {_debug_clip(focus.get('summary'), 220)}",
            f"- strength: {_debug_float(focus.get('strength'))}",
        ])
    else:
        lines.append("- No active focus.")

    lines.extend(["", "Context:"])
    if context:
        turns = context.get("recent_turns")
        recent_count = len(turns) if isinstance(turns, list) else 0
        lines.extend([
            f"- summary: {_debug_clip(context.get('summary'), 220)}",
            f"- last user: {_debug_clip(context.get('last_user_summary'), 180)}",
            f"- last assistant: {_debug_clip(context.get('last_assistant_summary'), 180)}",
            f"- recent turns: {recent_count}",
        ])
    else:
        lines.append("- No recent conversation context.")
    lines.extend([
        "",
        "Prompt:",
        f"- runtime context chars: {runtime_context_chars}",
        f"- prompt chars: {prompt_chars}",
        f"- wording avoidance chars: {len(avoidance)}",
        f"- prefix cache: {'on' if prefix.get('enabled') else 'off'} ({int(prefix.get('tokens') or 0)} tokens)",
    ])
    return "\n".join(lines)


def _commit_reply(user_input: str, raw: str, reply: str, session_id: str | None) -> None:
    remember_exchange(user_input, raw)
    persist_emotional_state()
    focus_state = decay_attention(session_id)
    observe_conversation_turn(session_id, "user", _discord_user_message(user_input), focus_state=focus_state)
    observe_conversation_turn(session_id, "assistant", reply, focus_state=focus_state)
    context = get_conversation_state(session_id)
    summary = str(context.get("summary") or "") if context else ""
    _log_summary(used=bool(summary), chars=len(summary), updated_after_stream=True)
    _append_message("user", user_input, session_id)
    _append_message("assistant", reply, session_id)


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


def _generate_reply(
    user_input: str,
    *,
    skip_memory: bool = False,
    session_id: str | None = None,
) -> str:
    _log_terminal_stream("user", user_input)
    messages, max_tokens = _chat_messages(user_input, skip_memory=skip_memory, session_id=session_id)
    result = _request_completion(messages, stream=False, max_tokens=max_tokens)
    raw = _extract_message_text(result)
    reply = _clean_reply(raw) or "I lost the thread for a second. Try that again."
    _log_reply(reply, session_id=session_id, raw=raw)
    _commit_reply(user_input, raw, reply, session_id)
    return reply


def _stream_chat_events(
    text: str,
    *,
    skip_memory: bool = False,
    session_id: str | None = None,
    request_started_at: float | None = None,
    json_parsed_at: float | None = None,
):
    started_at = request_started_at or time.perf_counter()
    prompt_started_at = first_token_at = first_visible_at = stream_done_at = post_done_at = None
    ctx_chars = 0
    chunk_count = 0
    try:
        _log_terminal_stream("user", text)
        prompt_started_at = time.perf_counter()
        messages, max_tokens = _chat_messages(text, skip_memory=skip_memory, session_id=session_id)
        prompt_done_at = time.perf_counter()
        if _timing_enabled():
            ctx_chars = sum(len(str(message.get("content", "") or "")) for message in messages)
        yield _json_line({"type": "start", "messages": _state_messages_with_user(session_id, text)})

        hidden_filter = HiddenTagStreamFilter()
        raw_parts: list[str] = []
        llama_started_at = time.perf_counter()
        stream = _request_completion(messages, stream=True, max_tokens=max_tokens)
        del messages

        for chunk in stream:
            token = content_delta(chunk, content_to_text)
            if not token:
                continue
            chunk_count += 1
            if first_token_at is None:
                first_token_at = time.perf_counter()
            raw_parts.append(token)
            visible = strip_emoji_chars(hidden_filter.feed(token))
            if chunk_count == 1:
                delay = first_token_at - started_at
                suffix = "; first chunk delayed by model prefill" if delay >= 2.0 else ""
                _log_stream(f"native streaming active; first_chunk={delay:.3f}s{suffix}")
            if not visible:
                continue
            if first_visible_at is None:
                first_visible_at = time.perf_counter()
            yield _json_line({"type": "delta", "content": visible, "append": True})

        tail = strip_emoji_chars(hidden_filter.flush())
        if tail:
            if first_visible_at is None:
                first_visible_at = time.perf_counter()
            yield _json_line({"type": "delta", "content": tail, "append": True})

        stream_done_at = time.perf_counter()
        post_started_at = stream_done_at
        raw = "".join(raw_parts)
        raw_parts.clear()
        reply = _clean_reply(raw) or "I lost the thread for a second. Try that again."
        _log_reply(reply, session_id=session_id, raw=raw)
        _commit_reply(text, raw, reply, session_id)
        post_done_at = time.perf_counter()
        _log_timing(
            json=(json_parsed_at - started_at) if json_parsed_at else 0.0,
            prompt=prompt_done_at - prompt_started_at,
            llama_start=llama_started_at - started_at,
            first_token=(first_token_at or stream_done_at) - started_at,
            first_visible=(first_visible_at or first_token_at or stream_done_at) - started_at,
            stream=stream_done_at - llama_started_at,
            post=post_done_at - post_started_at,
            total=post_done_at - started_at,
            chunks=chunk_count,
            ctx_chars=ctx_chars,
        )
        yield _json_line({"type": "done", "reply": reply, "messages": _state_messages(session_id)})
    except Exception as exc:
        failed_at = time.perf_counter()
        _log_timing(
            json=(json_parsed_at - started_at) if json_parsed_at else 0.0,
            prompt=((failed_at - prompt_started_at) if prompt_started_at else 0.0),
            total=failed_at - started_at,
            chunks=chunk_count,
            ctx_chars=ctx_chars,
        )
        _log_terminal_stream("error", str(exc))
        yield _json_line({"type": "error", "error": str(exc)})


def handle_builtin_command(text: str, scope: str | None = None) -> dict | None:
    if text == _DEBUG_STATE_COMMAND:
        return {"reply": _debug_state_report(scope), "ephemeral": True}
    if text == "/reset_chat":
        try:
            reset_chat_context(scope)
        except Exception:
            return {"error": "Could not reset chat."}
        return {"reply": "Chat reset.", "messages": [], "notice": "Chat reset."}
    if text == "/clear":
        _clear_messages(scope)
        clear_attention(scope)
        reset_conversation_state(scope)
        return {"reply": "", "messages": [], "notice": "Context cleared."}
    if text == "/memory":
        reload_from_disk()
        reply = format_memory_for_prompt() or "No memories yet."
        return {"reply": reply, "messages": [*_state_messages(scope), {"role": "assistant", "content": reply}], "ephemeral": True}
    return None


def _handle_command(text: str, session_id: str | None = None) -> dict | None:
    return handle_builtin_command(text, session_id)


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
        chat = _parse_chat_request(await _request_payload(request))
        if not chat.text:
            return JSONResponse({"error": "Message is empty."}, status_code=400)
        command = handle_builtin_command(chat.text, chat.session_id)
        if command is not None:
            return JSONResponse(command)
        _start_model_loading()
        try:
            reply = _generate_reply(chat.text, skip_memory=chat.skip_memory, session_id=chat.session_id)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return JSONResponse({"reply": reply, "messages": _state_messages(chat.session_id)})

    @app.post("/api/chat/stream")
    async def api_chat_stream(request: Request):
        request_started_at = time.perf_counter()
        payload = await _request_payload(request)
        json_parsed_at = time.perf_counter()
        chat = _parse_chat_request(payload)
        if not chat.text:
            return JSONResponse({"error": "Message is empty."}, status_code=400)
        command = handle_builtin_command(chat.text, chat.session_id)
        if command is not None:
            return JSONResponse(command)
        _start_model_loading()
        return StreamingResponse(
            _stream_chat_events(
                chat.text,
                skip_memory=chat.skip_memory,
                session_id=chat.session_id,
                request_started_at=request_started_at,
                json_parsed_at=json_parsed_at,
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
