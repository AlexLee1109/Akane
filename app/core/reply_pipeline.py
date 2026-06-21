"""Minimal generation path with state updates deferred until after replies."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from app.core.attention_state import format_attention_for_prompt, preview_attention
from app.core.config import MAX_TOKENS
from app.core.emotional_state import tone_line_for_message
from app.core.generation import (
    HiddenTagStreamFilter,
    clean_visible_text,
    completion_text,
    prefix_cache_status,
    stream_completion,
    warm_static_state,
)
from app.integrations.vscode_context import code_context_for_message

_MAX_METRICS = 64
_METRICS: dict[str, dict[str, object]] = {}
_DATE_TIME_MINUTE = -1
_DATE_TIME_TEXT = ""
_TIMING_ENABLED = str(os.environ.get("AKANE_TIMING", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@dataclass(frozen=True, slots=True)
class PreparedReply:
    user_text: str
    model_user_text: str
    actual_user_text: str
    session_id: str
    tone_line: str
    date_time_line: str
    max_tokens: int
    started_at: float
    direct_reply: str = ""
    attention: dict[str, object] | None = None
    code_context_requested: bool = False
    code_context_attached: bool = False


def _actual_message(text: str) -> str:
    if not text.startswith("Discord "):
        return text
    for marker in ("\nMessage:\n", "\nUser message:\n"):
        if marker in text:
            return text.split(marker, 1)[1].strip()
    return text


def _date_time_line() -> str:
    global _DATE_TIME_MINUTE, _DATE_TIME_TEXT
    timestamp = time.time()
    minute = int(timestamp // 60)
    if minute == _DATE_TIME_MINUTE:
        return _DATE_TIME_TEXT

    now = time.localtime(timestamp)
    hour = time.strftime("%I", now).lstrip("0") or "0"
    zone = time.strftime("%Z", now) or "local time"
    _DATE_TIME_TEXT = (
        f"Current date/time: {time.strftime('%A, %B', now)} "
        f"{now.tm_mday}, {now.tm_year} at {hour}:{time.strftime('%M %p', now)} {zone}."
    )
    _DATE_TIME_MINUTE = minute
    return _DATE_TIME_TEXT


def prepare_reply(
    user_text: str,
    *,
    session_id: str | None = None,
    skip_memory: bool = False,
) -> PreparedReply:
    del skip_memory
    text = str(user_text or "").strip()
    if not text:
        raise ValueError("Message is empty.")
    session = (str(session_id or "popup").strip() or "popup")[:120]
    actual_text = _actual_message(text)
    code_context = code_context_for_message(actual_text)
    model_text = text
    direct_reply = ""
    if code_context.requested:
        if code_context.direct_reply:
            direct_reply = code_context.direct_reply
        elif code_context.connected:
            model_text = f"{text}\n\n{code_context.prompt_text}"
        else:
            direct_reply = (
                "VS Code context isn’t connected yet. Open the Akane Local "
                "extension in VS Code and run “Akane Local: Connect to Pi” "
                "or refresh the workspace context."
            )
    context_attached = bool(code_context.prompt_text)
    attention = preview_attention(
        session,
        actual_text,
        code_context_requested=code_context.requested,
        code_context_attached=context_attached,
    )
    tone_line = tone_line_for_message(session, actual_text, attention)
    attention_line = format_attention_for_prompt(attention)
    runtime_line = "\n".join(part for part in (tone_line, attention_line) if part)
    return PreparedReply(
        user_text=text,
        model_user_text=model_text,
        actual_user_text=actual_text,
        session_id=session,
        tone_line=runtime_line,
        date_time_line=_date_time_line(),
        max_tokens=MAX_TOKENS,
        started_at=time.perf_counter() if _TIMING_ENABLED else 0.0,
        direct_reply=direct_reply,
        attention=attention,
        code_context_requested=code_context.requested,
        code_context_attached=context_attached,
    )


def _remember_metrics(prepared: PreparedReply, reply: str) -> None:
    _METRICS[prepared.session_id] = {
        "user_chars": len(prepared.user_text),
        "reply_chars": len(reply),
        "tone_chars": len(prepared.tone_line),
        "tone_included": 1 if prepared.tone_line else 0,
        "date_time_chars": len(prepared.date_time_line),
        "date_time_included": 1 if prepared.date_time_line else 0,
        "code_context_attached": 1 if prepared.code_context_attached else 0,
        "updated_at": time.time(),
    }
    if len(_METRICS) > _MAX_METRICS:
        oldest = min(
            _METRICS,
            key=lambda key: float(_METRICS[key].get("updated_at") or 0.0),
        )
        _METRICS.pop(oldest, None)


def _commit(prepared: PreparedReply, reply: str) -> None:
    try:
        from app.core.attention_state import observe_attention
        from app.core.conversation_state import observe_turn
        from app.core.emotional_state import (
            observe_assistant_reply,
            observe_user_message,
            persist as persist_emotion,
            update_cached_tone_line,
        )
        from app.memory_store import remember_exchange

        focus = observe_attention(
            prepared.session_id,
            prepared.actual_user_text,
            code_context_requested=prepared.code_context_requested,
            code_context_attached=prepared.code_context_attached,
        )
        observe_user_message(
            prepared.session_id,
            prepared.actual_user_text,
            attention=focus,
            persist=False,
        )
        observe_assistant_reply(prepared.session_id, reply, persist=False)
        update_cached_tone_line(prepared.session_id, focus)
        observe_turn(
            prepared.session_id,
            "user",
            prepared.actual_user_text,
            focus_state=focus,
        )
        observe_turn(
            prepared.session_id,
            "assistant",
            reply,
            focus_state=focus,
        )
        remember_exchange(prepared.user_text, reply)
        persist_emotion()
    except Exception as exc:
        print(f"[Akane:post] warning={type(exc).__name__}", flush=True)
    _remember_metrics(prepared, reply)


def _clean_reply_for_delivery(text: str) -> str:
    return clean_visible_text(text).lstrip()


def _timing_log(prepared: PreparedReply, reply: str, *, first_delta: float = 0.0) -> None:
    if not _TIMING_ENABLED:
        return
    done = time.perf_counter()
    cache = prefix_cache_status()
    fields = [
        "[Akane:timing]",
        f"total={done - prepared.started_at:.3f}s",
        f"cache={cache['status']}",
        f"static_prompt_tokens={int(cache['tokens'] or 0)}",
        f"dynamic_prompt_chars={int(cache['dynamic_chars'] or 0)}",
        f"tone_chars={len(prepared.tone_line)}",
        f"date_time_chars={len(prepared.date_time_line)}",
        f"state_load_seconds={float(cache['load_seconds'] or 0):.3f}s",
        f"output_chars={len(reply)}",
    ]
    if first_delta:
        fields.insert(1, f"first_delta={first_delta - prepared.started_at:.3f}s")
    print(" ".join(fields), flush=True)


def generate_reply(prepared: PreparedReply) -> str:
    if prepared.direct_reply:
        reply = prepared.direct_reply
        _commit(prepared, reply)
        return reply
    raw = completion_text(
        prepared.model_user_text,
        max_tokens=prepared.max_tokens,
        tone_line=prepared.tone_line,
        date_time_line=prepared.date_time_line,
    )
    reply = _clean_reply_for_delivery(raw)
    if not reply:
        raise RuntimeError("Model returned no visible reply.")
    _timing_log(prepared, reply)
    _commit(prepared, reply)
    return reply


def stream_reply(prepared: PreparedReply):
    if prepared.direct_reply:
        reply = prepared.direct_reply
        yield "delta", reply
        try:
            yield "done", reply
        finally:
            _commit(prepared, reply)
        return

    raw_parts: list[str] = []
    hidden = HiddenTagStreamFilter()
    started_visible = False
    first_delta = 0.0

    for text in stream_completion(
        prepared.model_user_text,
        max_tokens=prepared.max_tokens,
        tone_line=prepared.tone_line,
        date_time_line=prepared.date_time_line,
    ):
        raw_parts.append(text)
        visible = hidden.feed(text)
        if not started_visible:
            visible = visible.lstrip()
            if not visible:
                continue
            started_visible = True
            first_delta = time.perf_counter()
        yield "delta", visible

    tail = hidden.flush()
    if not started_visible:
        tail = tail.lstrip()
        if tail:
            started_visible = True
            first_delta = time.perf_counter()
    if tail:
        yield "delta", tail

    reply = _clean_reply_for_delivery("".join(raw_parts))
    if not reply:
        raise RuntimeError("Model returned no visible reply.")
    _timing_log(prepared, reply, first_delta=first_delta)
    try:
        yield "done", reply
    finally:
        _commit(prepared, reply)


def _debug_text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or text[:limit]


def debug_state_report(session_id: str | None) -> str:
    from app.core.attention_state import get_attention_state
    from app.core.emotional_state import (
        get_cached_tone_line as cached_tone_line,
        snapshot as emotion_snapshot,
    )

    session = (str(session_id or "popup").strip() or "popup")[:120]
    mood = emotion_snapshot(session)
    tone_line = cached_tone_line(session)
    focus = get_attention_state(session)
    metrics = _METRICS.get(session, {})
    changes = mood.get("top_changes") if isinstance(mood.get("top_changes"), list) else []
    changed_text = ", ".join(
        f"{item.get('name')} {float(item.get('delta') or 0):+.2f}"
        for item in changes
        if isinstance(item, dict) and item.get("name")
    ) or "none"
    lines = [
        "Akane debug state",
        f"- primary mood: {mood.get('primary_mood', 'calm')} ({float(mood.get('mood_intensity') or 0):.2f})",
        f"- secondary: {mood.get('secondary_emotion', 'curious')}",
        f"- top changes: {changed_text}",
        f"- trigger: {_debug_text(mood.get('trigger'), 40) or 'none'}",
        f"- attention: {_debug_text((focus or {}).get('topic'), 80) or 'none'}",
        f"- intent: {_debug_text((focus or {}).get('intent'), 30) or 'none'}",
        f"- tone: {_debug_text(tone_line, 200) or 'none'}",
        f"- code context attached: {'yes' if int(metrics.get('code_context_attached') or 0) else 'no'}",
    ]
    return "\n".join(lines)


def warm_caches() -> None:
    from app.core.character import get_static_system_prompt

    get_static_system_prompt(include_memory=False)
    warm_static_state()
