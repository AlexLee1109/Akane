"""Small bounded in-memory conversation state."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from app.core.generation import strip_hidden_blocks

_MAX_SESSIONS = 64
_STALE_SECONDS = 4 * 3600
_TEXT_CHARS = 160
_MAX_TURNS = 4
_MAX_ASSISTANT_REPLIES = 4
_DEBUG_COMMANDS = {"/debug_state"}
_LOW_CONTENT = {
    "hello", "hi", "hey", "yo", "lol", "ok", "okay", "thanks", "thank you",
    "gm", "gn", "good morning", "good night",
}
_FILLER_PHRASES = (
    "how's it going",
    "checking in",
    "just here",
    "nothing much",
    "still here",
    "waiting for the next message",
    "quiet thoughts",
    "quiet moments",
    "quiet blue",
    "vibes",
    "virtual data streams",
    "that's not how virtual bodies work",
)


@dataclass(slots=True)
class ConversationState:
    summary: str = ""
    last_user_summary: str = ""
    last_assistant_summary: str = ""
    recent_turns: deque = field(default_factory=lambda: deque(maxlen=_MAX_TURNS))
    recent_assistant_replies: deque = field(default_factory=lambda: deque(maxlen=_MAX_ASSISTANT_REPLIES))
    updated_at: float = field(default_factory=time.time)


_STATES: dict[str, ConversationState] = {}
_LOCK = threading.RLock()


def _key(session_id: str | None) -> str:
    return (str(session_id or "default").strip() or "default")[:120]


def _clip(text: object, limit: int = _TEXT_CHARS) -> str:
    value = " ".join(strip_hidden_blocks(str(text or "")).split()).strip()
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or value[:limit]


def _low_content(text: str) -> bool:
    value = _clip(text).lower().strip(".,!?;:()[]{}\"'`")
    return not value or value in _LOW_CONTENT or value in _DEBUG_COMMANDS


def _prune(now: float) -> None:
    for key, state in list(_STATES.items()):
        if now - state.updated_at > _STALE_SECONDS:
            _STATES.pop(key, None)
    if len(_STATES) > _MAX_SESSIONS:
        oldest = sorted(_STATES.items(), key=lambda item: item[1].updated_at)
        for key, _state in oldest[:len(_STATES) - _MAX_SESSIONS]:
            _STATES.pop(key, None)


def _public(state: ConversationState | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "summary": state.summary,
        "last_user_summary": state.last_user_summary,
        "last_assistant_summary": state.last_assistant_summary,
        "recent_turns": list(state.recent_turns),
        "recent_assistant_replies": list(state.recent_assistant_replies),
        "updated_at": state.updated_at,
    }


def get_conversation_state(session_id: str | None, *, now: float | None = None) -> dict[str, object] | None:
    current = time.time() if now is None else float(now)
    with _LOCK:
        _prune(current)
        return _public(_STATES.get(_key(session_id)))


def observe_turn(
    session_id: str | None,
    role: str,
    text: str,
    *,
    focus_state: dict[str, object] | None = None,
    now: float | None = None,
) -> dict[str, object] | None:
    current = time.time() if now is None else float(now)
    content = _clip(text)
    if not content or content in _DEBUG_COMMANDS or content.startswith("Akane debug state"):
        return get_conversation_state(session_id, now=current)

    with _LOCK:
        _prune(current)
        key = _key(session_id)
        state = _STATES.setdefault(key, ConversationState(updated_at=current))
        normalized_role = "assistant" if role == "assistant" else "user"
        state.recent_turns.append({"role": normalized_role, "text": content})
        if normalized_role == "assistant":
            state.last_assistant_summary = content
            state.recent_assistant_replies.append(content)
        elif not _low_content(content):
            state.last_user_summary = content
            focus_summary = _clip((focus_state or {}).get("summary"))
            state.summary = focus_summary or content
        state.updated_at = current
        return _public(state)


def format_conversation_context(
    session_id: str | None,
    *,
    focus_state: dict[str, object] | None = None,
    include_focus: bool = True,
    now: float | None = None,
) -> str:
    state = get_conversation_state(session_id, now=now)
    summary = _clip((state or {}).get("summary"))
    if not summary and include_focus:
        summary = _clip((focus_state or {}).get("summary"))
    return f"Context: {summary}" if summary else ""


def format_recent_wording_avoidance(session_id: str | None, *, now: float | None = None) -> str:
    state = get_conversation_state(session_id, now=now)
    replies = list((state or {}).get("recent_assistant_replies") or [])
    if not replies:
        return ""

    items: list[str] = []
    joined = " ".join(replies).lower().replace("’", "'")
    for phrase in _FILLER_PHRASES:
        if phrase in joined:
            items.append(phrase)
    for reply in reversed(replies):
        clipped = _clip(reply, 100)
        if clipped and clipped.lower() not in {item.lower() for item in items}:
            items.append(clipped)
        if len(items) >= 3:
            break

    text = "Avoid wording: " + " | ".join(items)
    return _clip(text, 240)


def clear_conversation_state(session_id: str | None = None) -> None:
    with _LOCK:
        _STATES.pop(_key(session_id), None)


def reset_conversation_state(session_id: str | None = None) -> None:
    clear_conversation_state(session_id)
