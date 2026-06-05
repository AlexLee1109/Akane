"""Compact per-session conversation coherence."""

from __future__ import annotations

import time

from app.core.generation import strip_hidden_blocks

_ACTIVE: dict[str, dict[str, object]] = {}
_MAX_STATES = 64
_MAX_TURNS = 4
_MAX_ASSISTANT_REPLIES = 6
_STALE_SECONDS = 4 * 3600
_TURN_CHARS = 200
_ASSISTANT_REPLY_CHARS = 200
_AVOID_REPLY_CHARS = 160
_AVOID_BLOCK_CHARS = 400
MAX_RECENT_SUMMARY_CHARS = 700
_SUMMARY_CHARS = MAX_RECENT_SUMMARY_CHARS
_LAST_CHARS = 200
_DEBUG_COMMANDS = {"/debug_state"}
_LOW_CONTENT = {
    "hello", "hi", "hey", "yo", "lol", "thanks", "thank you", "ok", "okay",
    "gm", "gn", "good morning", "good night", "hello there", "hi there", "hey there",
}
_FOLLOWUP_WORDS = {
    "again", "continue", "same", "that", "this", "those", "these", "it",
    "previous", "other", "shorter", "longer", "redo", "rewrite",
}
_AVOID_PHRASES = (
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
_AVOID_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "back", "be", "but", "for", "from",
    "i", "if", "in", "is", "it", "just", "like", "me", "my", "of", "on",
    "or", "so", "that", "the", "this", "to", "with", "you", "your",
}


def _session_id(session_id: str | None) -> str:
    return (str(session_id or "default").strip() or "default")[:120]


def _clean(text: str) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def _clip(text: str, limit: int) -> str:
    text = _clean(text)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or text[:limit]


def _assistant_text(text: str) -> str:
    content = _clip(strip_hidden_blocks(str(text or "")), _ASSISTANT_REPLY_CHARS)
    if not content or content.startswith("Akane debug state"):
        return ""
    return content


def _avoid_display(text: object, limit: int = _AVOID_REPLY_CHARS) -> str:
    return _clip(strip_hidden_blocks(str(text or "")), limit)


def _avoid_normalize(text: object) -> str:
    value = _avoid_display(text, 220).lower().replace("’", "'")
    for char in ".,!?;:()[]{}\"`":
        value = value.replace(char, " ")
    return " ".join(value.split())


def _is_low_content(text: str) -> bool:
    value = _clean(text).lower().strip(".,!?;:()[]{}\"'`")
    return len(value) < 4 or value in _LOW_CONTENT or value in _DEBUG_COMMANDS


def _terms(text: str) -> set[str]:
    return {
        word.strip(".,!?;:()[]{}\"'`").lower()
        for word in _clean(text).split()
        if len(word.strip(".,!?;:()[]{}\"'`")) >= 5
    }


def _is_followup(text: str) -> bool:
    words = {word.strip(".,!?;:()[]{}\"'`").lower() for word in _clean(text).split()}
    return bool(words & _FOLLOWUP_WORDS)


def _summary_from_user(text: str) -> str:
    return _clip(text, 260)


def _update_summary(state: dict[str, object], user_text: str, focus_state: dict[str, object] | None) -> None:
    before = str(state.get("summary") or "")
    _apply_focus(state, focus_state)
    if state.get("summary") and state.get("summary") != before:
        return

    if _is_low_content(user_text):
        return
    candidate = _summary_from_user(user_text)
    if not candidate:
        return
    current = str(state.get("summary") or "")
    if not current or _is_low_content(current):
        state["summary"] = _clip(candidate, _SUMMARY_CHARS)
        return
    if _is_followup(user_text) or (_terms(current) & _terms(user_text)):
        return
    state["summary"] = _clip(candidate, _SUMMARY_CHARS)


def _fresh(state: dict[str, object] | None, now: float) -> dict[str, object] | None:
    if not state or now - float(state.get("updated_at") or 0.0) > _STALE_SECONDS:
        return None
    turns = state.get("recent_turns")
    recent_turns = list(turns if isinstance(turns, list) else [])[-_MAX_TURNS:]
    assistant_replies = state.get("recent_assistant_replies")
    if isinstance(assistant_replies, list):
        recent_assistant_replies = [_assistant_text(str(reply)) for reply in assistant_replies]
    else:
        recent_assistant_replies = [
            _assistant_text(str(turn.get("text") or ""))
            for turn in recent_turns
            if isinstance(turn, dict) and turn.get("role") == "assistant"
        ]
    recent_assistant_replies = [reply for reply in recent_assistant_replies if reply][-_MAX_ASSISTANT_REPLIES:]
    return {
        "recent_turns": recent_turns,
        "recent_assistant_replies": recent_assistant_replies,
        "focus": _clip(str(state.get("focus") or ""), 90),
        "summary": _clip(str(state.get("summary") or ""), _SUMMARY_CHARS),
        "last_user_summary": _clip(str(state.get("last_user_summary") or ""), _LAST_CHARS),
        "last_assistant_summary": _assistant_text(str(state.get("last_assistant_summary") or "")),
        "updated_at": float(state.get("updated_at") or now),
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


def _apply_focus(state: dict[str, object], focus_state: dict[str, object] | None) -> None:
    if not focus_state:
        return
    focus = _clip(str(focus_state.get("topic") or ""), 90)
    summary = _clip(str(focus_state.get("summary") or ""), _SUMMARY_CHARS)
    if focus:
        state["focus"] = focus
    if summary:
        state["summary"] = summary


def get_conversation_state(session_id: str | None, *, now: float | None = None) -> dict[str, object] | None:
    current_time = time.time() if now is None else float(now)
    _prune(current_time)
    key = _session_id(session_id)
    state = _fresh(_ACTIVE.get(key), current_time)
    if state is None:
        _ACTIVE.pop(key, None)
    return state


def observe_turn(
    session_id: str | None,
    role: str,
    text: str,
    *,
    focus_state: dict[str, object] | None = None,
    now: float | None = None,
) -> dict[str, object] | None:
    current_time = time.time() if now is None else float(now)
    _prune(current_time)
    key = _session_id(session_id)
    state = _fresh(_ACTIVE.get(key), current_time) or {
        "recent_turns": [],
        "focus": "",
        "summary": "",
        "last_user_summary": "",
        "last_assistant_summary": "",
        "recent_assistant_replies": [],
        "updated_at": current_time,
    }
    role = "assistant" if role == "assistant" else "user"
    content = _assistant_text(text) if role == "assistant" else _clip(text, _TURN_CHARS)
    if content in _DEBUG_COMMANDS:
        return state
    if not content:
        return state

    turns = list(state.get("recent_turns") or [])
    turns.append({"role": role, "text": content, "ts": current_time})
    state["recent_turns"] = turns[-_MAX_TURNS:]
    if role == "user":
        state["last_user_summary"] = _clip(content, _LAST_CHARS)
    else:
        state["last_assistant_summary"] = _clip(content, _LAST_CHARS)
        recent_assistant_replies = list(state.get("recent_assistant_replies") or [])
        recent_assistant_replies.append(_clip(content, _ASSISTANT_REPLY_CHARS))
        state["recent_assistant_replies"] = recent_assistant_replies[-_MAX_ASSISTANT_REPLIES:]
    if role == "user":
        _update_summary(state, content, focus_state)
    elif focus_state:
        _apply_focus(state, focus_state)
    state["updated_at"] = current_time
    _ACTIVE[key] = state
    return state


def _recent_assistant_replies(state: dict[str, object] | None) -> list[str]:
    if not state:
        return []
    replies: list[str] = []
    stored = state.get("recent_assistant_replies")
    if isinstance(stored, list):
        replies.extend(_assistant_text(str(reply)) for reply in stored)
    else:
        for turn in state.get("recent_turns") or []:
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                replies.append(_assistant_text(str(turn.get("text") or "")))
    last = _assistant_text(str(state.get("last_assistant_summary") or ""))
    if last and (not replies or _avoid_normalize(replies[-1]) != _avoid_normalize(last)):
        replies.append(last)
    return [reply for reply in replies if reply][-_MAX_ASSISTANT_REPLIES:]


def _useful_phrase(phrase: str) -> bool:
    words = phrase.split()
    return len(words) >= 2 and len(phrase) >= 8 and any(word not in _AVOID_STOPWORDS for word in words)


def _avoidance_items(replies: list[str]) -> list[str]:
    records: dict[str, dict[str, int | str]] = {}

    def add(phrase: str, index: int, *, common: bool = False) -> None:
        phrase = _avoid_normalize(phrase)
        if not phrase or not _useful_phrase(phrase):
            return
        record = records.setdefault(phrase, {"count": 0, "last": 0, "common": 0, "text": phrase})
        record["count"] = int(record["count"]) + 1
        record["last"] = max(int(record["last"]), index)
        record["common"] = max(int(record["common"]), int(common))

    for index, reply in enumerate(replies):
        normalized = _avoid_normalize(reply)
        for phrase in _AVOID_PHRASES:
            if _avoid_normalize(phrase) in normalized:
                add(phrase, index, common=True)
        words = normalized.split()
        for size in (2, 3, 4):
            seen: set[str] = set()
            for start in range(0, max(0, len(words) - size + 1)):
                phrase = " ".join(words[start:start + size])
                if _useful_phrase(phrase):
                    seen.add(phrase)
            for phrase in seen:
                add(phrase, index)

    phrases = [record for record in records.values() if int(record["common"]) or int(record["count"]) > 1]
    phrases.sort(key=lambda item: (int(item["common"]), int(item["count"]), int(item["last"])), reverse=True)

    items: list[str] = []
    for record in phrases[:4]:
        text = _clip(str(record["text"]), 72)
        if text and text not in items:
            items.append(text)
    for reply in reversed(replies):
        text = _avoid_display(reply, _AVOID_REPLY_CHARS)
        if text and all(_avoid_normalize(text) != _avoid_normalize(item) for item in items):
            items.append(text)
        if len(items) >= 6:
            break
    return items


def format_recent_wording_avoidance(session_id: str | None, *, now: float | None = None) -> str:
    state = get_conversation_state(session_id, now=now)
    items = _avoidance_items(_recent_assistant_replies(state))
    if not items:
        return ""
    header = ["[RECENT WORDING TO AVOID]", "Do not repeat or closely reuse:"]
    footer = "Use different wording while keeping the same intent."
    lines = list(header)
    for item in items:
        candidate = [*lines, f"- {item}", "", footer]
        if len("\n".join(candidate)) <= _AVOID_BLOCK_CHARS:
            lines.append(f"- {item}")
    if len(lines) == len(header):
        return ""
    lines.extend(["", footer])
    return "\n".join(lines)


def format_conversation_context(
    session_id: str | None,
    *,
    focus_state: dict[str, object] | None = None,
    include_focus: bool = True,
    now: float | None = None,
) -> str:
    current_time = time.time() if now is None else float(now)
    _prune(current_time)
    key = _session_id(session_id)
    state = _fresh(_ACTIVE.get(key), current_time)
    if state is None:
        _ACTIVE.pop(key, None)
        state = {}
    if include_focus:
        _apply_focus(state, focus_state)
    summary = _clip(str(state.get("summary") or ""), _SUMMARY_CHARS)
    focus = _clip(str(state.get("focus") or ""), 90) if include_focus else ""
    if not include_focus and focus_state and summary == _clip(str(focus_state.get("summary") or ""), _SUMMARY_CHARS):
        summary = ""
    if not any((summary, focus)):
        return ""

    lines = ["[RECENT CONTEXT]"]
    if summary:
        lines.append(summary)
    if focus:
        lines.append(f"Focus: {focus}")
    lines.append("Use only for follow-ups/continuity. Do not mention this note.")
    return "\n".join(lines)


def reset_conversation_state(session_id: str | None = None) -> None:
    _ACTIVE.pop(_session_id(session_id), None)


def clear_conversation_state(session_id: str | None = None) -> None:
    reset_conversation_state(session_id)
