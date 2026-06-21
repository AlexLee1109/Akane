"""Small, bounded interaction-attention state for each chat session."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass

_MAX_SESSIONS = 64
_STALE_SECONDS = 60 * 60
_TOPIC_CHARS = 80
_SUMMARY_CHARS = 140
_LOW_CONTENT = {
    "hello", "hi", "hey", "yo", "lol", "ok", "okay", "thanks", "thank you",
    "good morning", "good night", "/debug_state",
}
_STOPWORDS = {
    "a", "an", "and", "are", "can", "could", "do", "does", "doing", "for", "from",
    "how", "i", "is", "it", "like", "me", "more", "of", "on", "please", "the",
    "this", "to", "we", "what", "when", "where", "with", "would", "you", "your",
    "akane", "about", "have", "has", "that", "still", "need", "needs",
}
_CODE_TERMS = {
    "api", "bug", "class", "code", "codebase", "config", "debug", "error", "file",
    "function", "implementation", "import", "java", "javascript", "module", "project",
    "python", "refactor", "repository", "rust", "script", "server", "stack",
    "test", "traceback", "typescript", "vscode", "workspace",
}
_CODE_ACTIONS = {
    "check", "debug", "explain", "fix", "implement", "inspect", "optimize",
    "how", "read", "refactor", "review", "rewrite", "test", "use", "write",
}
_PRAISE = (
    "adorable", "amazing", "awesome", "beautiful", "good job", "great job", "love this",
    "nailed it", "nice work", "perfect", "proud of you", "well done",
)
_TEASING = (
    "brat", "cute", "dork", "gremlin", "nerd", "silly", "smug", "tease",
)
_FRUSTRATION = (
    "again", "annoying", "broken", "doesn't work", "does not work", "failed",
    "frustrating", "hate this", "keeps happening", "stuck", "ugh", "wtf",
)
_THANKS = ("appreciate it", "thank you", "thanks")
_IDENTITY = (
    "about yourself", "what are you", "who are you", "your identity",
    "your personality", "yourself",
)
_WORD = re.compile(r"[a-z0-9_+#.-]+", re.IGNORECASE)


@dataclass(slots=True)
class AttentionState:
    topic: str
    summary: str
    topic_confidence: float
    intent: str
    casual_chat: bool
    code_help: bool
    needs_vscode: bool
    teasing: bool
    praise: bool
    frustrated: bool
    stance: str
    emotional_trigger: str
    active_task: str
    code_context_attached: bool
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


def _contains(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _words(text: str) -> set[str]:
    return {word.lower().strip(".") for word in _WORD.findall(text)}


def _topic(text: str) -> tuple[str, float]:
    lower = text.lower()
    known = (
        (
            (
                "personality", "emotion", "mood", "tone", "soul", "identity",
                "prompt rules", "question endings", "fake scenery", "visual-theme",
            ),
            "Akane personality",
        ),
        (("tts", "voice", "audio", "speech"), "TTS voice"),
        (("discord", "idle", "channel"), "Discord behavior"),
        (("vscode", "vs code", "workspace"), "VS Code workspace"),
        (("popup", "streaming"), "popup chat"),
    )
    for terms, label in known:
        if any(term in lower for term in terms):
            return label, 0.78
    terms = [
        word for word in _WORD.findall(lower)
        if len(word) >= 4 and word not in _STOPWORDS
    ]
    unique = list(dict.fromkeys(terms))
    if not unique:
        return _clip(text, _TOPIC_CHARS), 0.42
    return _clip(" ".join(unique[:3]), _TOPIC_CHARS), 0.58


def _low_content(text: str) -> bool:
    return _clean(text).lower().strip(".,!?;:()[]{}\"'`") in _LOW_CONTENT


def _topic_overlap(left: str, right: str) -> float:
    a = _words(left) - _STOPWORDS
    b = _words(right) - _STOPWORDS
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _analyze(
    user_text: str,
    *,
    code_context_requested: bool,
    code_context_attached: bool,
    now: float,
) -> AttentionState:
    summary = _clip(user_text, _SUMMARY_CHARS)
    lower = summary.lower()
    words = _words(lower)
    praise = _contains(lower, _PRAISE)
    teasing = _contains(lower, _TEASING) or any(token in lower for token in ("lol", "haha"))
    frustrated = _contains(lower, _FRUSTRATION)
    gratitude = _contains(lower, _THANKS)
    identity = _contains(lower, _IDENTITY)
    code_help = code_context_requested or bool(words & _CODE_TERMS) and bool(
        words & _CODE_ACTIONS or {"bug", "error", "traceback"} & words
    )

    if code_help:
        intent = "code_help"
        stance = "reassuring" if frustrated else "focused"
        trigger = (
            "repeated_problem"
            if "again" in words or "keeps happening" in lower
            else "code_problem" if frustrated or {"bug", "error", "traceback"} & words
            else "coding_task"
        )
    elif frustrated:
        intent, stance, trigger = "frustration", "reassuring", "user_frustration"
    elif praise:
        intent, stance, trigger = "praise", "playful", "praise"
    elif teasing:
        intent, stance, trigger = "teasing", "playful", "teasing"
    elif gratitude:
        intent, stance, trigger = "gratitude", "warm", "thanks"
    elif identity:
        intent, stance, trigger = "identity", "direct", "identity_interest"
    else:
        intent, stance, trigger = "casual", "warm", ""

    topic, confidence = _topic(summary)
    active_task = topic if code_help or intent in {"frustration", "identity"} else ""
    return AttentionState(
        topic=topic,
        summary=summary,
        topic_confidence=confidence,
        intent=intent,
        casual_chat=not code_help,
        code_help=code_help,
        needs_vscode=code_context_requested,
        teasing=teasing,
        praise=praise,
        frustrated=frustrated,
        stance=stance,
        emotional_trigger=trigger,
        active_task=active_task,
        code_context_attached=code_context_attached,
        updated_at=now,
    )


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
    result = asdict(state)
    result["strength"] = round(
        state.topic_confidence * max(0.0, 1.0 - (now - state.updated_at) / _STALE_SECONDS),
        3,
    )
    result["topic_confidence"] = round(state.topic_confidence, 3)
    return result


def preview_attention(
    session_id: str | None,
    user_text: str,
    *,
    code_context_requested: bool = False,
    code_context_attached: bool = False,
    now: float | None = None,
) -> dict[str, object]:
    current = time.time() if now is None else float(now)
    candidate = _analyze(
        user_text,
        code_context_requested=code_context_requested,
        code_context_attached=code_context_attached,
        now=current,
    )
    with _LOCK:
        previous = _STATES.get(_key(session_id))
        continuation = candidate.intent in {"gratitude", "praise", "teasing"}
        if previous and (
            _low_content(candidate.summary)
            or continuation and bool(previous.active_task)
            or _topic_overlap(previous.topic, candidate.topic) >= 0.5
        ):
            candidate.topic = previous.topic
            candidate.topic_confidence = min(0.94, max(candidate.topic_confidence, previous.topic_confidence + 0.05))
            if not candidate.active_task and previous.active_task:
                candidate.active_task = previous.active_task
    return _public(candidate, current) or {}


def observe_attention(
    session_id: str | None,
    user_text: str,
    *,
    code_context_requested: bool = False,
    code_context_attached: bool = False,
    now: float | None = None,
) -> dict[str, object] | None:
    current = time.time() if now is None else float(now)
    text = _clip(user_text, _SUMMARY_CHARS)
    with _LOCK:
        _prune(current)
        key = _key(session_id)
        previous = _STATES.get(key)
        if not text or _low_content(text):
            if previous is None:
                candidate = _analyze(
                    text,
                    code_context_requested=code_context_requested,
                    code_context_attached=code_context_attached,
                    now=current,
                )
                if candidate.intent == "casual":
                    return None
                _STATES[key] = candidate
                return _public(candidate, current)
            candidate = _analyze(
                text,
                code_context_requested=code_context_requested,
                code_context_attached=code_context_attached,
                now=current,
            )
            candidate.topic = previous.topic
            candidate.topic_confidence = max(0.35, previous.topic_confidence * 0.97)
            candidate.active_task = previous.active_task
        else:
            candidate = _analyze(
                text,
                code_context_requested=code_context_requested,
                code_context_attached=code_context_attached,
                now=current,
            )
            overlap = _topic_overlap(previous.topic, candidate.topic) if previous else 0.0
            weak_tangent = previous and candidate.topic_confidence < 0.65 and len(_words(text)) <= 4
            continuation = (
                previous
                and bool(previous.active_task)
                and candidate.intent in {"gratitude", "praise", "teasing"}
            )
            if previous and (overlap >= 0.5 or weak_tangent or continuation):
                candidate.topic = previous.topic
                candidate.topic_confidence = min(
                    0.94,
                    max(candidate.topic_confidence, previous.topic_confidence + (0.06 if overlap else 0.0)),
                )
                if not candidate.active_task:
                    candidate.active_task = previous.active_task
        _STATES[key] = candidate
        return _public(candidate, current)


def get_attention_state(session_id: str | None, *, now: float | None = None) -> dict[str, object] | None:
    current = time.time() if now is None else float(now)
    with _LOCK:
        _prune(current)
        return _public(_STATES.get(_key(session_id)), current)


def decay_attention(session_id: str | None, *, now: float | None = None) -> dict[str, object] | None:
    return get_attention_state(session_id, now=now)


def format_attention_for_prompt(state: dict[str, object] | None) -> str:
    if not state:
        return ""
    target = _clip(state.get("topic"), _TOPIC_CHARS)
    intent = _clip(state.get("intent"), 24) or "casual"
    stance = _clip(state.get("stance"), 24) or "warm"
    context = "attached" if state.get("code_context_attached") else (
        "needed but unavailable" if state.get("needs_vscode") else "not needed"
    )
    return (
        f"Attention: intent={intent}; target={target or 'current message'}; "
        f"stance={stance}; code context={context}."
    )


def clear_attention(session_id: str | None = None) -> None:
    with _LOCK:
        _STATES.pop(_key(session_id), None)
