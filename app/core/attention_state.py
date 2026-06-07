"""One short in-memory focus per session."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

_MAX_SESSIONS = 64
_STALE_SECONDS = 60 * 60
_TOPIC_CHARS = 80
_SUMMARY_CHARS = 120
_LOW_CONTENT = {
    "hello", "hi", "hey", "yo", "lol", "ok", "okay", "thanks", "thank you",
    "good morning", "good night", "/debug_state",
}
_STOPWORDS = {
    "a", "an", "and", "are", "can", "could", "do", "does", "doing", "for", "from",
    "how", "i", "is", "it", "like", "me", "more", "of", "on", "please", "the",
    "this", "to", "we", "what", "when", "where", "with", "would", "you", "your",
    "akane",
}


@dataclass(slots=True)
class AttentionState:
    topic: str
    summary: str
    updated_at: float


_STATES: dict[str, AttentionState] = {}
_LOCK = threading.RLock()


def _key(session_id: str | None) -> str:
    return (str(session_id or "default").strip() or "default")[:120]


def _clean(text: object) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _clip(text: object, limit: int) -> str:
    value = _clean(text)
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or value[:limit]


def _terms(text: str) -> list[str]:
    words = []
    for raw in _clean(text).lower().split():
        word = raw.strip(".,!?;:()[]{}\"'`")
        if len(word) >= 4 and word not in _STOPWORDS:
            words.append(word)
    return words[:2]


def _low_content(text: str) -> bool:
    return _clean(text).lower().strip(".,!?;:()[]{}\"'`") in _LOW_CONTENT


def _prune(now: float) -> None:
    for key, state in list(_STATES.items()):
        if now - state.updated_at > _STALE_SECONDS:
            _STATES.pop(key, None)
    if len(_STATES) > _MAX_SESSIONS:
        oldest = sorted(_STATES.items(), key=lambda item: item[1].updated_at)
        for key, _state in oldest[:len(_STATES) - _MAX_SESSIONS]:
            _STATES.pop(key, None)


def _public(state: AttentionState | None, now: float) -> dict[str, object] | None:
    if state is None or now - state.updated_at > _STALE_SECONDS:
        return None
    strength = max(0.0, 1.0 - (now - state.updated_at) / _STALE_SECONDS)
    return {
        "topic": state.topic,
        "summary": state.summary,
        "strength": round(strength, 3),
        "updated_at": state.updated_at,
    }


def observe_attention(session_id: str | None, user_text: str, *, now: float | None = None) -> dict[str, object] | None:
    current = time.time() if now is None else float(now)
    text = _clip(user_text, _SUMMARY_CHARS)
    with _LOCK:
        _prune(current)
        key = _key(session_id)
        if not text or _low_content(text):
            return _public(_STATES.get(key), current)
        terms = _terms(text)
        topic = _clip(" ".join(terms) if terms else text, _TOPIC_CHARS)
        _STATES[key] = AttentionState(topic=topic, summary=text, updated_at=current)
        return _public(_STATES[key], current)


def get_attention_state(session_id: str | None, *, now: float | None = None) -> dict[str, object] | None:
    current = time.time() if now is None else float(now)
    with _LOCK:
        _prune(current)
        return _public(_STATES.get(_key(session_id)), current)


def decay_attention(session_id: str | None, *, now: float | None = None) -> dict[str, object] | None:
    return get_attention_state(session_id, now=now)


def format_attention_for_prompt(state: dict[str, object] | None) -> str:
    topic = _clip((state or {}).get("topic"), _TOPIC_CHARS)
    return f"Focus: {topic}" if topic else ""


def clear_attention(session_id: str | None = None) -> None:
    with _LOCK:
        _STATES.pop(_key(session_id), None)
