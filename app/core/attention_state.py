"""Lightweight in-memory focus tracking for Akane."""

from __future__ import annotations

import time

_ACTIVE: dict[str, dict[str, object]] = {}
_MAX_STATES = 64
_TOPIC_CHARS = 80
_SUMMARY_CHARS = 160
_STALE_SECONDS = 60 * 60
_LOW_CONTENT = {
    "hello", "hi", "hey", "yo", "lol", "thanks", "thank you", "ok", "okay",
    "gm", "gn", "good morning", "good night", "hello there", "hi there", "hey there",
    "/debug_state",
}
_FOCUS_PHRASES = (
    "what do you like doing",
    "who are you",
    "what are you",
    "what do you like",
    "what anime do you like",
)
_STOPWORDS = {
    "about", "again", "akane", "are", "can", "could", "does", "doing", "for", "from",
    "have", "how", "into", "keep", "like", "make", "more", "need", "please", "should",
    "that", "the", "this", "what", "when", "where", "with", "would", "your", "you",
}


def _clamp(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return max(0.0, min(1.0, number))


def _session_id(session_id: str | None) -> str:
    return (str(session_id or "default").strip() or "default")[:120]


def _clean(text: str) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _clip(text: str, limit: int) -> str:
    text = _clean(text)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or text[:limit]


def _words(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for char in str(text or "").lower():
        if char.isalnum() or char in {"_", "-", "."}:
            current.append(char)
        elif current:
            word = "".join(current).strip("._-")
            if word:
                words.append(word)
            current.clear()
    if current:
        word = "".join(current).strip("._-")
        if word:
            words.append(word)
    return words


def _terms(text: str) -> list[str]:
    terms: list[str] = []
    for word in _words(text):
        if word in _STOPWORDS:
            continue
        if len(word) >= 4 or any(mark in word for mark in "_-."):
            terms.append(word)
    return terms


def _is_low_content(text: str) -> bool:
    cleaned = _clean(text).lower().strip(".,!?;:()[]{}\"'`")
    if any(phrase in cleaned for phrase in _FOCUS_PHRASES):
        return False
    return len(cleaned) < 4 or cleaned in _LOW_CONTENT or not _terms(cleaned)


def _focus_from_text(text: str) -> dict[str, object] | None:
    cleaned = _clip(text, _SUMMARY_CHARS)
    if not cleaned or _is_low_content(cleaned):
        return None
    terms = _terms(cleaned)
    topic = _clip(" ".join(terms[:6]) or cleaned, _TOPIC_CHARS)
    strength = 0.34 + min(0.36, len(terms) * 0.045) + min(0.20, len(cleaned) / 900)
    return {
        "topic": topic,
        "summary": cleaned,
        "terms": terms[:12],
        "strength": round(_clamp(strength), 3),
    }


def _fresh(state: dict[str, object] | None, now: float) -> dict[str, object] | None:
    if not state:
        return None
    elapsed = max(0.0, now - float(state.get("updated_at") or now))
    if elapsed > _STALE_SECONDS:
        return None
    strength = _clamp(state.get("strength"), 0.3) - elapsed / _STALE_SECONDS * 0.35
    if strength < 0.08:
        return None
    return {
        "topic": _clip(str(state.get("topic") or ""), _TOPIC_CHARS),
        "summary": _clip(str(state.get("summary") or ""), _SUMMARY_CHARS),
        "strength": round(strength, 3),
        "updated_at": float(state.get("updated_at") or now),
        "terms": list(state.get("terms") or [])[:12],
    }


def _prune(now: float) -> None:
    if len(_ACTIVE) <= _MAX_STATES:
        return
    for key, state in list(_ACTIVE.items()):
        if _fresh(state, now) is None:
            _ACTIVE.pop(key, None)
    extra = len(_ACTIVE) - _MAX_STATES
    if extra > 0:
        old = sorted(_ACTIVE.items(), key=lambda item: float(item[1].get("updated_at") or 0.0))
        for key, _state in old[:extra]:
            _ACTIVE.pop(key, None)


def _public(state: dict[str, object] | None) -> dict[str, object] | None:
    if not state:
        return None
    topic = _clip(str(state.get("topic") or ""), _TOPIC_CHARS)
    summary = _clip(str(state.get("summary") or ""), _SUMMARY_CHARS)
    if not topic or not summary:
        return None
    return {
        "topic": topic,
        "summary": summary,
        "strength": round(_clamp(state.get("strength")), 3),
        "updated_at": float(state.get("updated_at") or time.time()),
    }


def observe_attention(session_id: str | None, user_text: str, *, now: float | None = None) -> dict[str, object] | None:
    current_time = time.time() if now is None else float(now)
    _prune(current_time)
    key = _session_id(session_id)
    current = _fresh(_ACTIVE.get(key), current_time)
    focus = _focus_from_text(user_text)
    if focus is None:
        if current is None:
            _ACTIVE.pop(key, None)
        return _public(current)

    current_terms = set(str(term) for term in (current or {}).get("terms", []))
    new_terms = set(str(term) for term in focus.get("terms", []))
    if current and current_terms & new_terms:
        focus["topic"] = current.get("topic") or focus["topic"]
        focus["terms"] = sorted(current_terms | new_terms)[:12]
        focus["strength"] = round(min(1.0, _clamp(current.get("strength"), 0.3) + 0.08), 3)
    elif current and _clamp(current.get("strength")) > 0.55 and _clamp(focus.get("strength")) < 0.44:
        return _public(current)

    focus["updated_at"] = current_time
    _ACTIVE[key] = focus
    return _public(focus)


def decay_attention(session_id: str | None, *, now: float | None = None) -> dict[str, object] | None:
    current_time = time.time() if now is None else float(now)
    _prune(current_time)
    key = _session_id(session_id)
    state = _fresh(_ACTIVE.get(key), current_time)
    if state is None:
        _ACTIVE.pop(key, None)
        return None
    _ACTIVE[key] = state
    return _public(state)


def get_attention_state(session_id: str | None) -> dict[str, object] | None:
    return decay_attention(session_id)


def format_attention_for_prompt(state: dict[str, object] | None) -> str:
    if not state:
        return ""
    topic = _clip(str(state.get("topic") or ""), _TOPIC_CHARS)
    summary = _clip(str(state.get("summary") or ""), _SUMMARY_CHARS)
    return f"Focus: {topic}. {summary}" if topic and summary else ""


def clear_attention(session_id: str | None = None) -> None:
    _ACTIVE.pop(_session_id(session_id), None)
