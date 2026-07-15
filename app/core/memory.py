"""Bounded conversation history and selective, reliability-aware memories."""

from __future__ import annotations

import copy
import math
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from app.core.config import (
    CHAT_HISTORY_CONTEXT_TOKENS,
    CONVERSATION_STALE_DAYS,
    MAX_CONVERSATIONS,
    MEMORY_CONFIDENCE_WEIGHT,
    MEMORY_CONTEXT_TOKENS,
    MEMORY_CONTINUITY_WEIGHT,
    LONG_TERM_MEMORY_PATH,
    MEMORY_IMPORTANCE_WEIGHT,
    MEMORY_MAX_ENTRIES_PER_PROFILE,
    MEMORY_MAX_RESULTS,
    MEMORY_MIN_RELEVANCE,
    MEMORY_MIN_SCORE,
    MEMORY_PATH,
    MEMORY_RECENCY_WEIGHT,
    MEMORY_RELEVANCE_WEIGHT,
    MEMORY_REPETITION_PENALTY,
    MEMORY_STALENESS_PENALTY,
    POPUP_USER_PATH,
    SUMMARY_CONTEXT_TOKENS,
)
from app.core.persistence import atomic_write_json, read_json
from app.core.signal import (
    EmotionState,
    TurnContext,
    TurnSignal,
    advance_emotion,
    analyze_turn,
    message_similarity,
    normalized_signature,
    topic_overlap,
)
from app.core.utils import compact_text

MEMORY_SCHEMA_VERSION = 1
LONG_TERM_MEMORY_SCHEMA_VERSION = 3
_SUMMARY_BATCH_TURNS = 4
_MAX_TURN_CHARS = 8_000
_MEMORY_CATEGORIES = {
    "stable_fact",
    "episode",
    "tendency",
    "task_outcome",
    "unfinished_topic",
}
_ACTIVE_MEMORY = "active"
_MEMORY_PROMPT_INTRO = "A few past details may matter in this conversation:"
_MEMORY_PROMPT_OUTRO = (
    "Use them only when they genuinely improve the reply, and do not overstate uncertain details."
)
_SENSITIVE_TERMS = (
    "api key",
    "access token",
    "auth token",
    "credit card",
    "password",
    "private key",
    "secret key",
    "social security",
    "ssn",
)
_MEMORY_STOPWORDS = {
    "and",
    "are",
    "about",
    "again",
    "akane",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "its",
    "our",
    "remember",
    "said",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "user",
    "user's",
    "users",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "you",
    "your",
}
_REMEMBER_PATTERN = re.compile(
    r"\b(?:please\s+)?remember(?:\s+that)?\s+(?P<value>[^\n]{3,200})",
    re.IGNORECASE,
)
_PROFILE_NAME_PATTERN = re.compile(
    r"\b(?:my name is|i am called|i'm called|you can call me|call me)\s+"
    r"(?P<name>[A-Za-z][A-Za-z' -]{0,48}?)"
    r"(?=$|[.!?;,]|\s+(?:and|but)\s+I\b)",
    re.IGNORECASE,
)
_PROFILE_PREFERENCE_PATTERN = re.compile(
    r"\bI\s+(?:really\s+)?"
    r"(?P<verb>don't like|do not like|dislike|hate|like|love|prefer)\s+"
    r"(?P<value>[^\n.!?;]{1,160}?)"
    r"(?=$|[.!?;]|\s+(?:but|and)\s+I\b)",
    re.IGNORECASE,
)
_PROFILE_FAVORITE_PATTERN = re.compile(
    r"\bmy\s+favorite\s+(?P<subject>[^\n.!?]{1,48}?)\s+is\s+"
    r"(?P<value>[^\n.!?]{1,100})",
    re.IGNORECASE,
)
_PROFILE_FACT_PATTERNS = (
    ("lives in", re.compile(r"\bI\s+live\s+in\s+(?P<value>[^\n.!?;]{2,100})", re.I)),
    ("works as", re.compile(r"\bI\s+work\s+as\s+(?P<value>[^\n.!?;]{2,100})", re.I)),
    ("birthday", re.compile(r"\bmy\s+birthday\s+is\s+(?P<value>[^\n.!?;]{2,80})", re.I)),
)
_PROJECT_PATTERN = re.compile(
    r"\bI(?:'m| am)\s+(?:still\s+)?working on\s+(?P<value>[^\n.!?;]{3,180})",
    re.I,
)
_GOAL_PATTERN = re.compile(
    r"\bmy\s+(?:long[- ]term\s+)?goal is\s+(?P<value>[^\n.!?;]{3,180})",
    re.I,
)
_EPISODE_PATTERN = re.compile(
    r"\bI\s+(?P<event>(?:finally\s+)?(?:finished|fixed|solved|decided|chose|completed))\s+"
    r"(?P<value>[^\n.!?;]{3,180})",
    re.I,
)
_INTERACTION_PREFERENCE_PATTERN = re.compile(
    r"\b(?:please\s+)?(?P<verb>don't|do not|never)\s+"
    r"(?P<value>(?:ask|call|repeat|summarize|overexplain|use)[^\n.!?;]{1,150})",
    re.I,
)


def estimate_tokens(value: object) -> int:
    """Cheap conservative token estimate that never loads the model tokenizer."""

    text = str(value or "")
    if not text:
        return 0
    byte_estimate = (len(text.encode("utf-8")) + 2) // 3
    word_estimate = (len(text.split()) * 5 + 3) // 4
    return max(1, byte_estimate, word_estimate)


@dataclass(frozen=True, slots=True)
class ChatTurn:
    turn_id: str
    role: str
    content: str
    timestamp: float
    source: str

    @classmethod
    def from_dict(cls, payload: object) -> "ChatTurn | None":
        if not isinstance(payload, dict):
            return None
        role = "assistant" if payload.get("role") == "assistant" else "user"
        content = str(payload.get("content") or payload.get("text") or "").strip()
        if not content:
            return None
        try:
            timestamp = max(0.0, float(payload.get("timestamp") or 0.0))
        except (TypeError, ValueError):
            timestamp = 0.0
        return cls(
            turn_id=compact_text(payload.get("turn_id"), 80) or uuid.uuid4().hex,
            role=role,
            content=content[:_MAX_TURN_CHARS],
            timestamp=timestamp,
            source=compact_text(payload.get("source"), 24) or "unknown",
        )

    def as_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class ConversationRecord:
    conversation_id: str
    profile_id: str
    recent_turns: list[ChatTurn] = field(default_factory=list)
    summary_turns: list[ChatTurn] = field(default_factory=list)
    pending_summary_turns: list[ChatTurn] = field(default_factory=list)
    recent_topic: str = ""
    recent_intent: str = "casual"
    recent_user_tone: str = "neutral"
    current_task: str = ""
    correction: str = ""
    recent_events: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, key: str, payload: object) -> "ConversationRecord | None":
        if not isinstance(payload, dict):
            return None
        conversation_id = compact_text(payload.get("conversation_id") or key, 120)
        profile_id = compact_text(payload.get("profile_id"), 120)
        if not conversation_id or not profile_id:
            return None

        def turns(name: str) -> list[ChatTurn]:
            raw = payload.get(name)
            if not isinstance(raw, list):
                return []
            return [turn for item in raw if (turn := ChatTurn.from_dict(item)) is not None]

        try:
            updated_at = max(0.0, float(payload.get("updated_at") or 0.0))
        except (TypeError, ValueError):
            updated_at = 0.0
        return cls(
            conversation_id=conversation_id,
            profile_id=profile_id,
            recent_turns=turns("recent_turns"),
            summary_turns=turns("summary_turns"),
            pending_summary_turns=turns("pending_summary_turns")[-(_SUMMARY_BATCH_TURNS - 1) :],
            recent_topic=compact_text(payload.get("recent_topic"), 80),
            recent_intent=compact_text(payload.get("recent_intent"), 32) or "casual",
            recent_user_tone=compact_text(payload.get("recent_user_tone"), 32) or "neutral",
            current_task=compact_text(payload.get("current_task"), 100),
            correction=compact_text(payload.get("correction"), 120),
            recent_events=[
                value
                for item in (payload.get("recent_events") or [])[-5:]
                if (value := compact_text(item, 48))
            ],
            updated_at=updated_at,
        )

    def summary_text(self, query: str = "") -> str:
        turns = [*self.summary_turns, *self.pending_summary_turns]
        if query:
            turns = [
                turn
                for turn in turns
                if max(
                    topic_overlap(turn.content, query),
                    message_similarity(turn.content, query),
                )
                >= 0.30
            ][-4:]
        if not turns:
            return ""
        lines = [
            (
                "Prior Akane reply (unverified): "
                if turn.role == "assistant"
                else "User previously said: "
            )
            + compact_text(turn.content, 180)
            for turn in turns
        ]
        return (
            "Relevant earlier dialogue; prior replies are context, not facts:\n"
            + "\n".join(lines)
        )

    def public_state(self) -> dict[str, object]:
        turns = [
            {"role": turn.role, "text": turn.content, "timestamp": turn.timestamp}
            for turn in self.recent_turns
        ]
        users = [turn.content for turn in self.recent_turns if turn.role == "user"]
        assistants = [turn.content for turn in self.recent_turns if turn.role == "assistant"]
        return {
            "summary": self.summary_text(),
            "last_user_summary": compact_text(users[-1], 180) if users else "",
            "last_assistant_summary": compact_text(assistants[-1], 180) if assistants else "",
            "recent_turns": turns,
            "recent_user_messages": users[-4:],
            "recent_assistant_replies": assistants[-3:],
            "recent_events": list(self.recent_events),
            "recent_intent": self.recent_intent,
            "recent_user_tone": self.recent_user_tone,
            "recent_topic": self.recent_topic,
            "current_task": self.current_task,
            "correction": self.correction,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class MemoryContext:
    relationship: str
    earlier_dialogue: str
    recent_turns: tuple[ChatTurn, ...]
    memory_ids: tuple[str, ...] = ()
    memory_contents: tuple[str, ...] = ()


@dataclass(slots=True)
class Memory:
    id: str
    content: str
    category: str
    created_at: float
    last_used_at: float | None = None
    importance: float = 0.5
    confidence: float = 1.0
    source: str = "user"
    access_count: int = 0
    tags: tuple[str, ...] = ()
    status: str = _ACTIVE_MEMORY
    expires_at: float | None = None
    superseded_by: str | None = None

    @classmethod
    def from_dict(cls, payload: object) -> "Memory | None":
        if not isinstance(payload, dict):
            return None
        content = compact_text(payload.get("content"), 240)
        category = compact_text(payload.get("category"), 24).lower()
        if not content or category not in _MEMORY_CATEGORIES:
            return None
        try:
            created_at = max(0.0, float(payload.get("created_at") or 0.0))
            last_used = payload.get("last_used_at")
            last_used_at = max(0.0, float(last_used)) if last_used is not None else None
            expires = payload.get("expires_at")
            expires_at = max(0.0, float(expires)) if expires is not None else None
            access_count = max(0, int(payload.get("access_count") or 0))
        except (TypeError, ValueError):
            return None
        raw_tags = payload.get("tags")
        tags = tuple(
            value
            for item in (raw_tags if isinstance(raw_tags, (list, tuple)) else ())
            if (value := compact_text(item, 48).lower())
        )[:12]
        return cls(
            id=compact_text(payload.get("id"), 80) or uuid.uuid4().hex,
            content=content,
            category=category,
            created_at=created_at,
            last_used_at=last_used_at,
            importance=max(0.0, min(1.0, _number(payload.get("importance"), 0.5))),
            confidence=max(0.0, min(1.0, _number(payload.get("confidence"), 1.0))),
            source=compact_text(payload.get("source"), 48) or "user",
            access_count=access_count,
            tags=tags,
            status=compact_text(payload.get("status"), 24).lower() or _ACTIVE_MEMORY,
            expires_at=expires_at,
            superseded_by=compact_text(payload.get("superseded_by"), 80) or None,
        )

    def is_available(self, now: float) -> bool:
        return self.status == _ACTIVE_MEMORY and not (
            self.expires_at is not None and self.expires_at <= now
        )


@dataclass(frozen=True, slots=True)
class InteractionEvent:
    kind: str
    summary: str
    created_at: float
    resolved: bool = False

    @classmethod
    def from_dict(cls, payload: object) -> "InteractionEvent | None":
        if not isinstance(payload, dict):
            return None
        kind = compact_text(payload.get("kind"), 32).lower()
        summary = compact_text(payload.get("summary"), 160)
        if not kind or not summary:
            return None
        return cls(
            kind=kind,
            summary=summary,
            created_at=max(0.0, _number(payload.get("created_at"), 0.0)),
            resolved=bool(payload.get("resolved")),
        )


@dataclass(frozen=True, slots=True)
class WorkingMemory:
    current_topic: str = ""
    current_task: str = ""
    unresolved_problem: bool = False
    repeated_topic_count: int = 0
    last_outcome: str = ""
    last_user_summary: str = ""
    last_assistant_summary: str = ""
    recent_events: tuple[InteractionEvent, ...] = ()

    @classmethod
    def from_dict(cls, payload: object) -> "WorkingMemory":
        if not isinstance(payload, dict):
            return cls()
        raw_events = payload.get("recent_events")
        events = tuple(
            event
            for item in (raw_events if isinstance(raw_events, list) else [])[-16:]
            if (event := InteractionEvent.from_dict(item)) is not None
        )
        return cls(
            current_topic=compact_text(payload.get("current_topic"), 100),
            current_task=compact_text(payload.get("current_task"), 160),
            unresolved_problem=bool(payload.get("unresolved_problem")),
            repeated_topic_count=max(0, min(20, int(_number(payload.get("repeated_topic_count"), 0)))),
            last_outcome=compact_text(payload.get("last_outcome"), 40).lower(),
            last_user_summary=compact_text(payload.get("last_user_summary"), 180),
            last_assistant_summary=compact_text(payload.get("last_assistant_summary"), 180),
            recent_events=events,
        )


@dataclass(frozen=True, slots=True)
class InternalState:
    emotion: EmotionState
    working: WorkingMemory = WorkingMemory()
    memories: tuple[Memory, ...] = ()
    updated_at: float = 0.0
    version: int = 1


@dataclass(frozen=True, slots=True)
class InternalTurnResult:
    state: InternalState
    signal: TurnSignal
    recalled_memories: tuple[Memory, ...]
    prompt_context: str


@dataclass(frozen=True, slots=True)
class MemoryRetrievalConfig:
    context_tokens: int = MEMORY_CONTEXT_TOKENS
    max_results: int = MEMORY_MAX_RESULTS
    min_relevance: float = MEMORY_MIN_RELEVANCE
    min_score: float = MEMORY_MIN_SCORE
    relevance_weight: float = MEMORY_RELEVANCE_WEIGHT
    importance_weight: float = MEMORY_IMPORTANCE_WEIGHT
    confidence_weight: float = MEMORY_CONFIDENCE_WEIGHT
    recency_weight: float = MEMORY_RECENCY_WEIGHT
    continuity_weight: float = MEMORY_CONTINUITY_WEIGHT
    repetition_penalty: float = MEMORY_REPETITION_PENALTY
    staleness_penalty: float = MEMORY_STALENESS_PENALTY


@dataclass(frozen=True, slots=True)
class _MemoryCandidate:
    content: str
    category: str
    importance: float
    confidence: float
    tags: tuple[str, ...]


class MemoryStore:
    """Own recent turns and rolling excerpts for each conversation."""

    def __init__(self, path: Path = MEMORY_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._conversations: dict[str, ConversationRecord] = {}
        self._load()

    def preview_signal(
        self,
        conversation_id: str,
        user_text: str,
        *,
        code_context_requested: bool = False,
        code_context_attached: bool = False,
    ) -> TurnSignal:
        """Compatibility helper; the runtime uses coordinated ``preview_turn``."""

        candidate = analyze_turn(
            user_text,
            code_context_requested=code_context_requested,
            code_context_attached=code_context_attached,
        )
        with self._lock:
            previous = self._conversations.get(_key(conversation_id, "local:conversation"))
            if previous is None:
                return candidate
            recent_users = [
                turn.content for turn in previous.recent_turns if turn.role == "user"
            ][-4:]
            repeated = [
                item for item in recent_users if message_similarity(item, user_text) >= 0.78
            ]
            if repeated:
                exact = any(
                    normalized_signature(item) == normalized_signature(user_text)
                    for item in repeated
                )
                prior_signal = analyze_turn(repeated[-1])
                candidate = replace(
                    candidate,
                    repetition="exact" if exact else "near",
                    repetition_count=len(repeated),
                    current_activity=candidate.current_activity or prior_signal.current_activity,
                    identity_attribute=(
                        candidate.identity_attribute or prior_signal.identity_attribute
                    ),
                )
            continuation = candidate.low_content or candidate.intent in {
                "gratitude",
                "praise",
                "teasing",
                "correction",
            }
            if previous.recent_topic and (
                continuation or topic_overlap(previous.recent_topic, candidate.topic) >= 0.5
            ):
                return candidate.with_context(
                    topic=previous.recent_topic,
                    task=candidate.task or previous.current_task,
                    confidence=max(candidate.topic_confidence, 0.62),
                )
        return candidate

    def build_context(
        self,
        profile_id: str,
        conversation_id: str,
        *,
        display_name: str = "",
        query: str = "",
        include_memory: bool = True,
    ) -> MemoryContext:
        profile = _key(profile_id, "local:owner")
        conversation_key = _key(conversation_id, "popup:default")
        with self._lock:
            record = self._conversations.get(conversation_key)
            if record is not None and record.profile_id != profile:
                raise ValueError("Conversation belongs to a different profile.")
            recent = tuple(record.recent_turns) if record and include_memory else ()
            summary = record.summary_text(query) if record and include_memory else ""
            relationship = _relationship_context(
                display_name,
                bool(record and record.recent_turns and include_memory),
            )
            return MemoryContext(
                relationship=relationship,
                earlier_dialogue=summary,
                recent_turns=recent,
            )

    def commit_turn(
        self,
        *,
        profile_id: str,
        conversation_id: str,
        source: str,
        user_text: str,
        assistant_text: str,
        signal: TurnSignal,
    ) -> None:
        profile = _key(profile_id, "local:owner")
        conversation_key = _key(conversation_id, "popup:default")
        now = time.time()
        user_turn = ChatTurn(uuid.uuid4().hex, "user", user_text[:_MAX_TURN_CHARS], now, source)
        assistant_turn = ChatTurn(
            uuid.uuid4().hex,
            "assistant",
            assistant_text[:_MAX_TURN_CHARS],
            now,
            source,
        )
        with self._lock:
            previous_conversations = self._conversations
            record = previous_conversations.get(conversation_key)
            if record is None:
                record = ConversationRecord(conversation_key, profile)
            elif record.profile_id != profile:
                raise ValueError("Conversation belongs to a different profile.")
            else:
                record = copy.deepcopy(record)

            self._conversations = previous_conversations.copy()
            self._conversations[conversation_key] = record

            record.recent_turns.extend((user_turn, assistant_turn))
            record.recent_topic = compact_text(signal.topic, 80)
            record.recent_intent = signal.intent
            record.recent_user_tone = signal.tone
            record.current_task = compact_text(signal.task, 100)
            if signal.correction:
                record.correction = compact_text(signal.correction, 120)
            if signal.trigger:
                record.recent_events.append(compact_text(signal.trigger, 48))
                record.recent_events = record.recent_events[-5:]
            record.updated_at = now
            self._trim_conversation(record)

            self._prune(now)
            try:
                self._persist()
            except Exception:
                self._conversations = previous_conversations
                raise

    def messages(
        self,
        conversation_id: str,
        profile_id: str | None = None,
    ) -> list[dict[str, str]]:
        with self._lock:
            record = self._conversations.get(_key(conversation_id, "popup:default"))
            if record and profile_id and record.profile_id != _key(profile_id, "local:owner"):
                return []
            return [turn.as_message() for turn in record.recent_turns] if record else []

    def public_conversation(
        self,
        conversation_id: str,
        profile_id: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            record = self._conversations.get(_key(conversation_id, "popup:default"))
            if record and profile_id and record.profile_id != _key(profile_id, "local:owner"):
                return {}
            return record.public_state() if record else {}

    def clear_conversation(
        self,
        conversation_id: str,
        profile_id: str | None = None,
    ) -> bool:
        with self._lock:
            key = _key(conversation_id, "popup:default")
            record = self._conversations.get(key)
            if record and profile_id and record.profile_id != _key(profile_id, "local:owner"):
                return False
            if record is None:
                return False
            self._conversations.pop(key, None)
            self._persist()
            return True

    def clear_profile(self, profile_id: str) -> None:
        profile = _key(profile_id, "local:owner")
        with self._lock:
            conversations = {
                key: value
                for key, value in self._conversations.items()
                if value.profile_id != profile
            }
            if len(conversations) == len(self._conversations):
                return
            self._conversations = conversations
            self._persist()

    def _trim_conversation(self, record: ConversationRecord) -> None:
        recent_tokens = sum(
            estimate_tokens(turn.content) + 4 for turn in record.recent_turns
        )
        while (
            recent_tokens > CHAT_HISTORY_CONTEXT_TOKENS
            and len(record.recent_turns) > 2
        ):
            remove_count = (
                2
                if len(record.recent_turns) >= 4
                and record.recent_turns[0].role == "user"
                and record.recent_turns[1].role == "assistant"
                else 1
            )
            for _ in range(remove_count):
                removed = record.recent_turns.pop(0)
                recent_tokens -= estimate_tokens(removed.content) + 4
                record.pending_summary_turns.append(removed)
            if len(record.pending_summary_turns) >= _SUMMARY_BATCH_TURNS:
                record.summary_turns.extend(record.pending_summary_turns)
                record.pending_summary_turns.clear()
                summary_tokens = sum(
                    estimate_tokens(turn.content) + 4 for turn in record.summary_turns
                )
                while summary_tokens > SUMMARY_CONTEXT_TOKENS and record.summary_turns:
                    summary_remove_count = (
                        2
                        if len(record.summary_turns) >= 2
                        and record.summary_turns[0].role == "user"
                        and record.summary_turns[1].role == "assistant"
                        else 1
                    )
                    for _ in range(summary_remove_count):
                        removed = record.summary_turns.pop(0)
                        summary_tokens -= estimate_tokens(removed.content) + 4

    def _prune(self, now: float) -> None:
        stale_before = now - CONVERSATION_STALE_DAYS * 86_400
        for key, record in list(self._conversations.items()):
            if record.updated_at and record.updated_at < stale_before:
                self._conversations.pop(key, None)
        if len(self._conversations) > MAX_CONVERSATIONS:
            ordered = sorted(self._conversations.items(), key=lambda item: item[1].updated_at)
            for key, _record in ordered[: len(self._conversations) - MAX_CONVERSATIONS]:
                self._conversations.pop(key, None)

    def _load(self) -> None:
        try:
            payload = read_json(self._path)
            if (
                not isinstance(payload, dict)
                or int(payload.get("schema_version", 0)) != MEMORY_SCHEMA_VERSION
            ):
                raise ValueError("unsupported schema")
            conversations = payload.get("conversations")
            if not isinstance(conversations, dict):
                raise ValueError("invalid memory document")
            self._conversations = {
                record.conversation_id: record
                for key, value in conversations.items()
                if (record := ConversationRecord.from_dict(str(key), value)) is not None
            }
            self._prune(time.time())
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as exc:
            print(f"[Akane:memory] ignored corrupt memory file ({type(exc).__name__})", flush=True)
            self._conversations = {}

    def _persist(self) -> None:
        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "conversations": {
                key: asdict(value) for key, value in self._conversations.items()
            },
        }
        atomic_write_json(self._path, payload)


def _relationship_context(display_name: str, has_history: bool) -> str:
    parts: list[str] = []
    name = compact_text(display_name, 60)
    if name:
        parts.append(f"This person is displayed as {name}; use the name only when natural")
    if has_history:
        parts.append(
            "this is an ongoing relationship, so continuity may be acknowledged "
            "without forced familiarity"
        )
    return "Relationship: " + "; ".join(parts) + "." if parts else ""


def _key(value: object, default: str) -> str:
    return compact_text(value, 120) or default


def new_internal_state(now: float | None = None) -> InternalState:
    current = time.time() if now is None else max(0.0, float(now))
    return InternalState(emotion=EmotionState(updated_at=current), updated_at=current)


def process_internal_turn(
    user_text: str,
    state: InternalState | None = None,
    *,
    now: float | None = None,
    retrieval: MemoryRetrievalConfig | None = None,
    include_memory: bool = True,
    code_context_requested: bool = False,
    code_context_attached: bool = False,
) -> InternalTurnResult:
    """Purely appraise a turn and return the proposed coordinated state."""

    current = time.time() if now is None else max(0.0, float(now))
    previous = state if state is not None else new_internal_state(current)
    current = max(current, previous.updated_at, previous.emotion.updated_at)
    working = previous.working
    context = TurnContext(
        current_topic=working.current_topic,
        current_task=working.current_task,
        unresolved_problem=working.unresolved_problem,
        repeated_topic_count=working.repeated_topic_count,
        last_outcome=working.last_outcome,
    )
    signal = analyze_turn(
        user_text,
        emotion_state=previous.emotion,
        turn_context=context,
        now=current,
        code_context_requested=code_context_requested,
        code_context_attached=code_context_attached,
    )
    continuing = _continues_working_topic(user_text, signal, working)
    topic = working.current_topic if continuing and working.current_topic else signal.topic
    task = working.current_task if continuing and working.current_task else signal.task
    if signal.task_failure and not task:
        task = topic
    similarity = message_similarity(working.last_user_summary, signal.summary)
    if working.last_user_summary and similarity >= 0.78:
        signal = replace(
            signal,
            repetition=(
                "exact"
                if normalized_signature(working.last_user_summary)
                == normalized_signature(signal.summary)
                else "near"
            ),
            repetition_count=max(1, working.repeated_topic_count),
        )
    signal = signal.with_context(
        topic=topic,
        task=task,
        confidence=max(signal.topic_confidence, 0.62) if continuing else None,
    )

    same_topic = bool(
        working.current_topic
        and topic
        and (
            continuing
            or topic_overlap(working.current_topic, topic) >= 0.45
            or message_similarity(working.current_topic, topic) >= 0.78
        )
    )
    repeated_count = min(20, working.repeated_topic_count + 1) if same_topic else 1
    unresolved = working.unresolved_problem if same_topic or continuing else False
    outcome = working.last_outcome if same_topic or continuing else ""
    events = list(working.recent_events)
    if signal.task_failure:
        unresolved = True
        outcome = "technical_failure"
        events.append(InteractionEvent("technical_failure", compact_text(task or topic, 160), current))
    elif signal.task_success and working.unresolved_problem:
        unresolved = False
        outcome = "technical_success"
        events = [
            replace(event, resolved=True)
            if not event.resolved and event.kind in {"technical_failure", "unfinished_task"}
            else event
            for event in events
        ]
        events.append(InteractionEvent("technical_success", compact_text(task or topic, 160), current, True))
    elif signal.correction_requested:
        outcome = "correction"
        events.append(InteractionEvent("correction_received", signal.summary, current))
    for matches, kind in (
        (signal.sadness, "user_distress"),
        (signal.hostility, "conflict"),
        (signal.criticism and not signal.correction_requested, "criticism_received"),
        (signal.praise and not signal.task_success, "praise_received"),
        (signal.teasing, "playful_exchange"),
    ):
        if matches:
            events.append(InteractionEvent(kind, compact_text(signal.summary, 160), current))

    next_working = WorkingMemory(
        current_topic=compact_text(topic, 100),
        current_task=compact_text(task, 160),
        unresolved_problem=unresolved,
        repeated_topic_count=repeated_count,
        last_outcome=outcome,
        last_user_summary=compact_text(signal.summary, 180),
        last_assistant_summary=working.last_assistant_summary,
        recent_events=tuple(events[-16:]),
    )
    memories = copy.deepcopy(list(previous.memories))
    for candidate in _extract_memory_candidates(user_text):
        _insert_into_memories(
            memories,
            candidate,
            source="chat:explicit_user",
            created_at=current,
        )
    task_tag = "task:" + _slot_value(task) if task else ""
    if signal.task_failure and task and not _has_active_tag(memories, task_tag):
        _insert_into_memories(
            memories,
            _MemoryCandidate(
                content=f"An unfinished task remains: {compact_text(task, 150)}.",
                category="unfinished_topic",
                importance=0.72,
                confidence=0.86,
                tags=(task_tag, "unfinished"),
            ),
            source="chat:task_state",
            created_at=current,
        )
    elif signal.task_success and task:
        for memory in memories:
            if memory.status == _ACTIVE_MEMORY and (
                task_tag in memory.tags or memory.category == "unfinished_topic"
                and topic_overlap(memory.content, task) >= 0.45
            ):
                memory.status = "resolved"
        _insert_into_memories(
            memories,
            _MemoryCandidate(
                content=f"The task was resolved: {compact_text(task, 160)}.",
                category="task_outcome",
                importance=0.68,
                confidence=0.82,
                tags=(task_tag, "resolved"),
            ),
            source="chat:task_state",
            created_at=current,
        )

    query = " ".join(part for part in (user_text, next_working.current_task) if part)
    previous_ids = {memory.id for memory in previous.memories}
    recall_pool = tuple(memory for memory in memories if memory.id in previous_ids)
    recalled = (
        _retrieve_memories(
            recall_pool,
            query,
            current,
            retrieval or MemoryRetrievalConfig(),
            next_working,
        )
        if include_memory
        else ()
    )
    _prune_memories(memories)
    next_state = InternalState(
        emotion=signal.emotion_state,
        working=next_working,
        memories=tuple(memories),
        updated_at=current,
        version=1,
    )
    return InternalTurnResult(
        state=next_state,
        signal=signal,
        recalled_memories=recalled,
        prompt_context=_internal_prompt_context(signal, next_working, recalled),
    )


class LongTermMemoryStore:
    """Own the coordinated emotion, working context, and durable profile memory."""

    def __init__(
        self,
        path: Path | None = None,
        retrieval: MemoryRetrievalConfig | None = None,
    ) -> None:
        self._path = Path(path) if path is not None else LONG_TERM_MEMORY_PATH
        self._legacy_path = None if path is not None else POPUP_USER_PATH
        self._retrieval = retrieval or MemoryRetrievalConfig()
        self._lock = threading.RLock()
        self._states: dict[str, InternalState] = {}
        self._load()

    def internal_state(self, profile_id: str = "local:owner") -> InternalState:
        key = _key(profile_id, "local:owner")
        with self._lock:
            state = self._states.get(key)
            return copy.deepcopy(state) if state is not None else new_internal_state()

    def stored_internal_state(self, profile_id: str = "local:owner") -> InternalState | None:
        """Return the persisted profile record, without manufacturing a default."""

        with self._lock:
            return copy.deepcopy(self._states.get(_key(profile_id, "local:owner")))

    def snapshot(self, profile_id: str = "local:owner") -> list[Memory]:
        """Compatibility view of durable memories only."""

        with self._lock:
            state = self._states.get(_key(profile_id, "local:owner"))
            return copy.deepcopy(list(state.memories)) if state else []

    def preview_turn(
        self,
        profile_id: str,
        user_text: str,
        *,
        now: float | None = None,
        include_memory: bool = True,
        code_context_requested: bool = False,
        code_context_attached: bool = False,
    ) -> InternalTurnResult:
        with self._lock:
            state = self._states.get(_key(profile_id, "local:owner"))
        return process_internal_turn(
            user_text,
            state or new_internal_state(now),
            now=now,
            retrieval=self._retrieval,
            include_memory=include_memory,
            code_context_requested=code_context_requested,
            code_context_attached=code_context_attached,
        )

    def commit_turn(
        self,
        profile_id: str,
        result: InternalTurnResult,
        *,
        assistant_text: str = "",
        used_memory_ids: tuple[str, ...] = (),
        now: float | None = None,
    ) -> None:
        key = _key(profile_id, "local:owner")
        current = result.state.updated_at if now is None else max(
            result.state.updated_at,
            0.0,
            float(now),
        )
        with self._lock:
            previous = self._states.get(key)
            state = copy.deepcopy(result.state)
            memories = list(state.memories)
            wanted = set(used_memory_ids)
            for memory in memories:
                if memory.id in wanted and memory.status == _ACTIVE_MEMORY:
                    memory.last_used_at = current
                    memory.access_count += 1
            working = replace(
                state.working,
                last_assistant_summary=compact_text(assistant_text, 180),
            )
            self._states[key] = replace(
                state,
                working=working,
                memories=tuple(memories),
                updated_at=current,
            )
            try:
                self._persist()
            except Exception:
                if previous is None:
                    self._states.pop(key, None)
                else:
                    self._states[key] = previous
                raise

    def retrieve(
        self,
        profile_id: str,
        query: str,
        *,
        now: float | None = None,
    ) -> tuple[Memory, ...]:
        current = time.time() if now is None else float(now)
        with self._lock:
            state = self._states.get(_key(profile_id, "local:owner"))
            if state is None:
                return ()
            return _retrieve_memories(
                state.memories,
                query,
                current,
                self._retrieval,
                state.working,
            )

    def prompt_text(self, query: str = "", profile_id: str = "local:owner") -> str:
        return format_relevant_memories(self.retrieve(profile_id, query))

    def add_memory(
        self,
        profile_id: str,
        content: str,
        *,
        category: str = "stable_fact",
        importance: float = 0.7,
        confidence: float = 1.0,
        source: str = "user",
        tags: tuple[str, ...] = (),
        status: str = _ACTIVE_MEMORY,
        created_at: float | None = None,
    ) -> Memory | None:
        candidate = _MemoryCandidate(
            content=compact_text(content, 240),
            category=compact_text(category, 24).lower(),
            importance=max(0.0, min(1.0, float(importance))),
            confidence=max(0.0, min(1.0, float(confidence))),
            tags=tuple(tags),
        )
        if not candidate.content or candidate.category not in _MEMORY_CATEGORIES:
            return None
        key = _key(profile_id, "local:owner")
        with self._lock:
            previous = self._states.get(key)
            state = copy.deepcopy(previous) if previous else new_internal_state(created_at)
            memories = list(state.memories)
            memory, changed = _insert_into_memories(
                memories,
                candidate,
                source=source,
                created_at=time.time() if created_at is None else float(created_at),
                status=status,
            )
            if not changed:
                return copy.deepcopy(memory)
            self._states[key] = replace(
                state,
                memories=tuple(memories),
                updated_at=time.time() if created_at is None else float(created_at),
            )
            try:
                self._persist()
            except Exception:
                if previous is None:
                    self._states.pop(key, None)
                else:
                    self._states[key] = previous
                raise
            return copy.deepcopy(memory)

    def commit(
        self,
        user_text: str,
        *,
        profile_id: str = "local:owner",
        source: str = "popup",
        used_memory_ids: tuple[str, ...] = (),
        now: float | None = None,
    ) -> None:
        candidates = _extract_memory_candidates(user_text)
        if not candidates and not used_memory_ids:
            return
        key = _key(profile_id, "local:owner")
        with self._lock:
            previous = self._states.get(key)
            changed = False
            current = time.time() if now is None else float(now)
            state = copy.deepcopy(previous) if previous else new_internal_state(current)
            memories = list(state.memories)
            wanted = set(used_memory_ids)
            for memory in memories:
                if memory.id in wanted and memory.status == _ACTIVE_MEMORY:
                    memory.last_used_at = current
                    memory.access_count += 1
                    changed = True
            for candidate in candidates:
                _memory, inserted = _insert_into_memories(
                    memories,
                    candidate,
                    source=f"{compact_text(source, 24) or 'chat'}:explicit_user",
                    created_at=current,
                )
                changed = changed or inserted
            if not changed:
                return
            self._states[key] = replace(state, memories=tuple(memories), updated_at=current)
            try:
                self._persist()
            except Exception:
                if previous is None:
                    self._states.pop(key, None)
                else:
                    self._states[key] = previous
                raise

    def restore(self, profile_id: str, memories: list[Memory]) -> None:
        key = _key(profile_id, "local:owner")
        with self._lock:
            state = self._states.get(key, new_internal_state())
            self._states[key] = replace(state, memories=tuple(copy.deepcopy(memories)))
            self._persist()

    def restore_internal_state(self, profile_id: str, state: InternalState | None) -> None:
        key = _key(profile_id, "local:owner")
        with self._lock:
            if state is None:
                self._states.pop(key, None)
            else:
                self._states[key] = copy.deepcopy(state)
            self._persist()

    def clear(self, profile_id: str = "local:owner") -> None:
        with self._lock:
            key = _key(profile_id, "local:owner")
            if key not in self._states:
                return
            self._states.pop(key)
            self._persist()

    def public_profile(self, profile_id: str = "local:owner") -> dict[str, object]:
        now = time.time()
        with self._lock:
            state = self._states.get(_key(profile_id, "local:owner"))
            active = [
                copy.deepcopy(memory)
                for memory in (state.memories if state else ())
                if memory.is_available(now)
            ]
        user: dict[str, str] = {}
        preferences: list[dict[str, object]] = []
        facts: list[dict[str, object]] = []
        episodes: list[dict[str, object]] = []
        for memory in active:
            item = {"content": memory.content, "category": memory.category}
            if "slot:name" in memory.tags:
                user["name"] = _memory_name(memory.content)
            elif memory.category == "tendency" or any(
                tag.startswith("slot:preference:") for tag in memory.tags
            ):
                preferences.append(item)
            elif memory.category in {"episode", "task_outcome", "unfinished_topic"}:
                episodes.append(item)
            else:
                facts.append(item)
        return {
            "user": user,
            "preferences": preferences,
            "facts": facts,
            "episodes": episodes,
            "activities": {},
        }

    def public_internal_state(self, profile_id: str = "local:owner") -> dict[str, object]:
        state = self.internal_state(profile_id)
        emotion = state.emotion
        return {
            "emotion": {
                "dominant": emotion.dominant,
                "secondary": emotion.secondary,
                "valence": round(emotion.valence, 3),
                "arousal": round(emotion.arousal, 3),
                "frustration": round(emotion.frustration, 3),
                "concern": round(emotion.concern, 3),
                "updated_at": emotion.updated_at,
            },
            "working": asdict(state.working),
            "updated_at": state.updated_at,
        }

    def _load(self) -> None:
        try:
            try:
                payload = read_json(self._path)
            except FileNotFoundError:
                if self._legacy_path is None:
                    return
                payload = read_json(self._legacy_path)
            if not isinstance(payload, dict):
                raise ValueError("invalid long-term memory document")
            schema = int(payload.get("schema_version", 0))
            if schema == MEMORY_SCHEMA_VERSION and isinstance(payload.get("user"), dict):
                migrated = _migrate_legacy_profile(payload["user"])
                if migrated:
                    current = max(memory.created_at for memory in migrated)
                    self._states = {
                        "local:owner": replace(
                            new_internal_state(current),
                            memories=tuple(migrated),
                        )
                    }
                return
            if schema not in {2, LONG_TERM_MEMORY_SCHEMA_VERSION}:
                raise ValueError("unsupported schema")
            profiles = payload.get("profiles")
            if not isinstance(profiles, dict):
                raise ValueError("invalid profiles")
            self._states = {}
            for key, raw_profile in profiles.items():
                profile = _key(key, "")
                if not profile:
                    continue
                if schema == 2 and isinstance(raw_profile, list):
                    loaded = tuple(
                        memory
                        for item in raw_profile[-MEMORY_MAX_ENTRIES_PER_PROFILE:]
                        if (memory := Memory.from_dict(item)) is not None
                    )
                    if loaded:
                        current = max(memory.created_at for memory in loaded)
                        self._states[profile] = replace(
                            new_internal_state(current),
                            memories=loaded,
                        )
                elif schema == LONG_TERM_MEMORY_SCHEMA_VERSION:
                    state = _internal_state_from_dict(raw_profile)
                    if state is not None:
                        self._states[profile] = state
        except FileNotFoundError:
            return
        except (OSError, TypeError, ValueError) as exc:
            print(
                f"[Akane:long-term-memory] ignored corrupt memory ({type(exc).__name__})",
                flush=True,
            )
            self._states = {}

    def _persist(self) -> None:
        atomic_write_json(
            self._path,
            {
                "schema_version": LONG_TERM_MEMORY_SCHEMA_VERSION,
                "profiles": {
                    key: _internal_state_to_dict(state)
                    for key, state in self._states.items()
                },
            },
        )


PopupUserStore = LongTermMemoryStore


def format_relevant_memories(memories: tuple[Memory, ...]) -> str:
    if not memories:
        return ""
    lines = [_memory_prompt_line(memory) for memory in memories]
    return (
        _MEMORY_PROMPT_INTRO
        + "\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n"
        + _MEMORY_PROMPT_OUTRO
    )


def _memory_prompt_line(memory: Memory) -> str:
    if memory.confidence < 0.72:
        return f"A tentative earlier impression: {memory.content}"
    return memory.content


def _continues_working_topic(
    user_text: str,
    signal: TurnSignal,
    working: WorkingMemory,
) -> bool:
    if not working.current_topic:
        return False
    if topic_overlap(signal.topic, working.current_topic) >= 0.45:
        return True
    if message_similarity(signal.summary, working.last_user_summary) >= 0.78:
        return True
    if signal.correction_requested and working.current_task:
        return True
    lower = compact_text(user_text, 240).lower()
    return bool(
        working.unresolved_problem
        and (
            signal.low_content
            or signal.correction_requested
            or re.search(
                r"\b(?:again|same (?:thing|issue)|that|this|it|fixed|solved|worked|works|broke|failed)\b",
                lower,
            )
        )
    )


def _internal_prompt_context(
    signal: TurnSignal,
    working: WorkingMemory,
    memories: tuple[Memory, ...],
) -> str:
    parts = [signal.emotion_prompt()]
    continuity: list[str] = []
    if working.current_task:
        continuity.append(f"Current task: {compact_text(working.current_task, 120)}.")
    elif working.current_topic:
        continuity.append(f"Current topic: {compact_text(working.current_topic, 100)}.")
    if working.unresolved_problem:
        continuity.append("The current problem is still unresolved.")
    elif working.last_outcome == "technical_success":
        continuity.append("The previously unresolved problem is now resolved.")
    if working.repeated_topic_count >= 2 and working.unresolved_problem:
        continuity.append("This is a repeated attempt on the same problem.")
    if continuity:
        parts.append("Continuity: " + " ".join(continuity))
    if memories:
        lines = [compact_text(_memory_prompt_line(memory), 140) for memory in memories]
        parts.append("Relevant memory:\n" + "\n".join(f"- {line}" for line in lines))
    return "\n".join(parts)


def _retrieve_memories(
    memories: tuple[Memory, ...],
    query: str,
    now: float,
    config: MemoryRetrievalConfig,
    working: WorkingMemory,
) -> tuple[Memory, ...]:
    query_text = compact_text(query, 700)
    query_terms = _memory_terms(query_text)
    if not query_text or not query_terms:
        return ()
    ranked: list[tuple[float, Memory]] = []
    for memory in memories:
        if not memory.is_available(now):
            continue
        relevance = _semantic_relevance(query_text, query_terms, memory)
        continuity_bonus = 0.0
        if memory.category == "unfinished_topic" and working.current_task:
            continuity_bonus = max(
                topic_overlap(memory.content, working.current_task),
                _semantic_relevance(working.current_task, _memory_terms(working.current_task), memory),
            ) * 0.30
        if relevance < config.min_relevance and continuity_bonus <= 0.0:
            continue
        age_days = max(0.0, now - memory.created_at) / 86_400.0
        recency = 1.0 / (1.0 + age_days / 90.0)
        continuity = recency if memory.category in {"episode", "task_outcome"} else 0.0
        score = (
            relevance * config.relevance_weight
            + memory.importance * config.importance_weight
            + memory.confidence * config.confidence_weight
            + recency * config.recency_weight
            + continuity * config.continuity_weight
            + continuity_bonus
            - _recent_use(memory, now) * config.repetition_penalty
            - min(1.0, max(0.0, age_days - 365.0) / 365.0) * config.staleness_penalty
        )
        if score >= config.min_score:
            ranked.append((score, memory))
    ranked.sort(key=lambda item: (item[0], item[1].created_at, item[1].id), reverse=True)
    selected: list[Memory] = []
    used_tokens = estimate_tokens(_MEMORY_PROMPT_INTRO) + estimate_tokens(_MEMORY_PROMPT_OUTRO) + 4
    for _score, memory in ranked:
        cost = estimate_tokens(_memory_prompt_line(memory)) + 1
        if used_tokens + cost > config.context_tokens:
            continue
        selected.append(copy.deepcopy(memory))
        used_tokens += cost
        if len(selected) >= config.max_results:
            break
    return tuple(selected)


def _insert_into_memories(
    memories: list[Memory],
    candidate: _MemoryCandidate,
    *,
    source: str,
    created_at: float,
    status: str = _ACTIVE_MEMORY,
) -> tuple[Memory, bool]:
    signature = normalized_signature(candidate.content)
    candidate_tags = _normalized_tags(candidate.tags, candidate.content)
    candidate_slots = {tag for tag in candidate_tags if tag.startswith("slot:")}
    for existing in memories:
        if existing.status != _ACTIVE_MEMORY or existing.category != candidate.category:
            continue
        exact = normalized_signature(existing.content) == signature
        same_slot = bool(candidate_slots.intersection(existing.tags))
        near_duplicate = not same_slot and message_similarity(existing.content, candidate.content) >= 0.94
        if exact or near_duplicate:
            changed = False
            if candidate.importance > existing.importance:
                existing.importance = candidate.importance
                changed = True
            if candidate.confidence > existing.confidence:
                existing.confidence = candidate.confidence
                changed = True
            return existing, changed
    memory = Memory(
        id=uuid.uuid4().hex,
        content=candidate.content,
        category=candidate.category,
        created_at=max(0.0, created_at),
        importance=candidate.importance,
        confidence=candidate.confidence,
        source=compact_text(source, 48) or "user",
        tags=candidate_tags,
        status=compact_text(status, 24).lower() or _ACTIVE_MEMORY,
    )
    if candidate_slots:
        for existing in memories:
            if existing.status == _ACTIVE_MEMORY and candidate_slots.intersection(existing.tags):
                existing.status = "superseded"
                existing.superseded_by = memory.id
    memories.append(memory)
    _prune_memories(memories)
    return memory, True


def _prune_memories(memories: list[Memory]) -> None:
    if len(memories) <= MEMORY_MAX_ENTRIES_PER_PROFILE:
        return
    memories.sort(
        key=lambda item: (item.status == _ACTIVE_MEMORY, item.importance, item.created_at)
    )
    del memories[: len(memories) - MEMORY_MAX_ENTRIES_PER_PROFILE]


def _has_active_tag(memories: list[Memory], tag: str) -> bool:
    return bool(tag) and any(
        memory.status == _ACTIVE_MEMORY and tag in memory.tags for memory in memories
    )


def _internal_state_to_dict(state: InternalState) -> dict[str, object]:
    return {
        "version": state.version,
        "updated_at": state.updated_at,
        "emotion": asdict(state.emotion),
        "working": asdict(state.working),
        "memories": [asdict(memory) for memory in state.memories],
    }


def _internal_state_from_dict(payload: object) -> InternalState | None:
    if not isinstance(payload, dict):
        return None
    updated_at = max(0.0, _number(payload.get("updated_at"), 0.0))
    raw_emotion = payload.get("emotion")
    emotion_payload = raw_emotion if isinstance(raw_emotion, dict) else {}
    emotion_time = max(0.0, _number(emotion_payload.get("updated_at"), updated_at))
    candidate = EmotionState(
        updated_at=emotion_time,
        valence=_number(emotion_payload.get("valence"), 0.05),
        arousal=_number(emotion_payload.get("arousal"), 0.15),
        energy=_number(emotion_payload.get("energy"), 0.65),
        warmth=_number(emotion_payload.get("warmth"), 0.5),
        curiosity=_number(emotion_payload.get("curiosity"), 0.45),
        confidence=_number(emotion_payload.get("confidence"), 0.55),
        stimulation=_number(emotion_payload.get("stimulation"), 0.5),
        amusement=_number(emotion_payload.get("amusement"), 0.0),
        concern=_number(emotion_payload.get("concern"), 0.0),
        frustration=_number(emotion_payload.get("frustration"), 0.0),
        irritation=_number(emotion_payload.get("irritation"), 0.0),
        dominant=compact_text(emotion_payload.get("dominant"), 32) or "relaxed",
        secondary=compact_text(emotion_payload.get("secondary"), 32),
        cause=compact_text(emotion_payload.get("cause"), 100),
    )
    emotion = advance_emotion(candidate, now=emotion_time)
    raw_memories = payload.get("memories")
    memories = tuple(
        memory
        for item in (raw_memories if isinstance(raw_memories, list) else [])[-MEMORY_MAX_ENTRIES_PER_PROFILE:]
        if (memory := Memory.from_dict(item)) is not None
    )
    return InternalState(
        emotion=emotion,
        working=WorkingMemory.from_dict(payload.get("working")),
        memories=memories,
        updated_at=updated_at or emotion.updated_at,
        version=1,
    )


def _extract_memory_candidates(text: str) -> tuple[_MemoryCandidate, ...]:
    value = compact_text(text, 700)
    lower = value.lower()
    if (
        not value
        or any(marker in lower for marker in ("hypothetically", "just kidding", "for example"))
        or any(marker in lower for marker in _SENSITIVE_TERMS)
    ):
        return ()
    candidates: list[_MemoryCandidate] = []

    if match := _PROFILE_NAME_PATTERN.search(value):
        name = compact_text(match.group("name"), 50).strip(" ,;:-")
        if 1 <= len(name.split()) <= 4:
            candidates.append(_candidate(f"The user's name is {name}.", "stable_fact", 0.94, ("slot:name",)))

    if match := _REMEMBER_PATTERN.search(value):
        fact = compact_text(match.group("value"), 180).strip(" ,.;:-")
        if fact:
            candidates.append(_candidate(_as_user_fact(fact), "stable_fact", 0.98, ("explicit-request",)))

    for match in _PROFILE_PREFERENCE_PATTERN.finditer(value):
        preference = compact_text(match.group("value"), 140).strip(" ,;:-")
        if not preference:
            continue
        verb = " ".join(match.group("verb").lower().split())
        negative = verb in {"don't like", "do not like", "dislike", "hate"}
        category = "tendency" if _interaction_preference(preference) else "stable_fact"
        action = "dislikes" if negative else "prefers" if verb == "prefer" else "likes"
        slot = "slot:preference:" + _slot_value(preference)
        candidates.append(
            _candidate(f"The user {action} {preference}.", category, 0.82, (slot,))
        )

    for match in _PROFILE_FAVORITE_PATTERN.finditer(value):
        subject = compact_text(match.group("subject"), 48).strip(" ,;:-")
        favorite = compact_text(match.group("value"), 100).strip(" ,;:-")
        if subject and favorite:
            slot = "slot:favorite:" + _slot_value(subject)
            candidates.append(
                _candidate(
                    f"The user's favorite {subject} is {favorite}.",
                    "stable_fact",
                    0.88,
                    (slot,),
                )
            )

    fact_phrases = {
        "lives in": "The user lives in {value}.",
        "works as": "The user works as {value}.",
        "birthday": "The user's birthday is {value}.",
    }
    for label, pattern in _PROFILE_FACT_PATTERNS:
        if match := pattern.search(value):
            fact = compact_text(match.group("value"), 100).strip(" ,;:-")
            if fact:
                candidates.append(
                    _candidate(
                        fact_phrases[label].format(value=fact),
                        "stable_fact",
                        0.86,
                        ("slot:" + label.replace(" ", "-"),),
                    )
                )

    if match := _PROJECT_PATTERN.search(value):
        project = compact_text(match.group("value"), 180).strip(" ,;:-")
        if project:
            candidates.append(
                _candidate(f"The user is working on {project}.", "stable_fact", 0.78, ("project",))
            )
    if match := _GOAL_PATTERN.search(value):
        goal = compact_text(match.group("value"), 180).strip(" ,;:-")
        if goal:
            candidates.append(
                _candidate(f"The user's long-term goal is {goal}.", "stable_fact", 0.82, ("goal",))
            )
    if match := _EPISODE_PATTERN.search(value):
        event = " ".join(match.group("event").lower().split())
        detail = compact_text(match.group("value"), 180).strip(" ,;:-")
        if detail:
            candidates.append(
                _candidate(f"The user {event} {detail}.", "episode", 0.70, ("event",))
            )
    if match := _INTERACTION_PREFERENCE_PATTERN.search(value):
        instruction = compact_text(match.group("value"), 150).strip(" ,;:-")
        if instruction:
            candidates.append(
                _candidate(
                    f"The user prefers that Akane not {instruction.lower()}.",
                    "tendency",
                    0.90,
                    ("interaction-style",),
                )
            )

    unique: dict[str, _MemoryCandidate] = {}
    for candidate in candidates:
        unique.setdefault(normalized_signature(candidate.content), candidate)
    return tuple(unique.values())


def _candidate(
    content: str,
    category: str,
    importance: float,
    tags: tuple[str, ...],
) -> _MemoryCandidate:
    return _MemoryCandidate(
        content=compact_text(content, 240),
        category=category,
        importance=importance,
        confidence=1.0,
        tags=tags,
    )


def _semantic_relevance(query: str, query_terms: set[str], memory: Memory) -> float:
    memory_terms = _memory_terms(memory.content) | {
        part
        for tag in memory.tags
        for part in tag.replace("slot:", "").replace(":", "-").split("-")
        if len(part) >= 3
    }
    overlap = len(query_terms & memory_terms) / max(1, min(len(query_terms), len(memory_terms)))
    similarity = message_similarity(query, memory.content)
    typo_similarity = similarity * 0.65 if similarity >= 0.72 else 0.0
    return min(
        1.0,
        max(
            overlap,
            topic_overlap(query, memory.content),
            typo_similarity,
        ),
    )


def _memory_terms(value: str) -> set[str]:
    return {
        token
        for token in normalized_signature(value).split()
        if len(token) >= 3 and token not in _MEMORY_STOPWORDS
    }


def _recent_use(memory: Memory, now: float) -> float:
    if memory.last_used_at is None:
        return 0.0
    age_days = max(0.0, now - memory.last_used_at) / 86_400.0
    return min(1.0, 1.0 / (1.0 + age_days * 4.0) + min(memory.access_count, 5) * 0.08)


def _normalized_tags(tags: tuple[str, ...], content: str) -> tuple[str, ...]:
    values = [compact_text(tag, 48).lower() for tag in tags]
    values.extend(sorted(_memory_terms(content))[:8])
    return tuple(dict.fromkeys(value for value in values if value))[:12]


def _interaction_preference(value: str) -> bool:
    lower = value.lower()
    return any(
        marker in lower
        for marker in (
            "answer",
            "concise",
            "detail",
            "explanation",
            "follow-up",
            "question",
            "reply",
            "response",
            "tone",
        )
    )


def _as_user_fact(value: str) -> str:
    fact = compact_text(value, 180).strip(" ,.;:-")
    replacements = (
        (r"^i(?:'m| am)\s+", "The user is "),
        (r"^i like\s+", "The user likes "),
        (r"^i love\s+", "The user loves "),
        (r"^i prefer\s+", "The user prefers "),
        (r"^i (?:don't like|do not like|dislike|hate)\s+", "The user dislikes "),
        (r"^my\s+", "The user's "),
    )
    for pattern, replacement in replacements:
        if re.search(pattern, fact, re.I):
            return re.sub(pattern, replacement, fact, count=1, flags=re.I).rstrip(".") + "."
    return f"The user explicitly stated: {fact}."


def _slot_value(value: str) -> str:
    terms = sorted(_memory_terms(value))[:5]
    return "-".join(terms) or normalized_signature(value).replace(" ", "-")[:32]


def _number(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _memory_name(content: str) -> str:
    match = re.search(r"name is\s+(.+?)[.]?$", content, re.I)
    return match.group(1) if match else content


def _migrate_legacy_profile(payload: dict[str, object]) -> list[Memory]:
    try:
        created_at = max(0.0, float(payload.get("updated_at") or time.time()))
    except (TypeError, ValueError):
        created_at = time.time()
    candidates: list[_MemoryCandidate] = []
    name = compact_text(payload.get("name"), 50)
    if name:
        candidates.append(_candidate(f"The user's name is {name}.", "stable_fact", 0.94, ("slot:name",)))
    for key, action in (("likes", "likes"), ("dislikes", "dislikes")):
        raw = payload.get(key)
        if isinstance(raw, list):
            for item in raw:
                value = compact_text(item, 140)
                if value:
                    candidates.append(
                        _candidate(
                            f"The user {action} {value}.",
                            "stable_fact",
                            0.82,
                            ("slot:preference:" + _slot_value(value),),
                        )
                    )
    raw_facts = payload.get("facts")
    if isinstance(raw_facts, list):
        for item in raw_facts:
            value = compact_text(item, 180)
            if not value:
                continue
            label, separator, detail = value.partition(":")
            if separator and label.lower().startswith("favorite "):
                subject = label[9:].strip()
                candidate = _candidate(
                    f"The user's favorite {subject} is {detail.strip()}.",
                    "stable_fact",
                    0.86,
                    ("slot:favorite:" + _slot_value(subject),),
                )
            elif separator and label.lower() == "works as":
                candidate = _candidate(
                    f"The user works as {detail.strip()}.",
                    "stable_fact",
                    0.84,
                    ("slot:works-as",),
                )
            else:
                candidate = _candidate(_as_user_fact(value), "stable_fact", 0.80, ())
            candidates.append(candidate)
    return [
        Memory(
            id=uuid.uuid4().hex,
            content=candidate.content,
            category=candidate.category,
            created_at=created_at,
            importance=candidate.importance,
            confidence=candidate.confidence,
            source="migrated:popup-user",
            tags=_normalized_tags(candidate.tags, candidate.content),
        )
        for candidate in candidates
    ]


_MEMORY: MemoryStore | None = None
_MEMORY_LOCK = threading.Lock()
_INTERNAL_MEMORY: LongTermMemoryStore | None = None
_INTERNAL_MEMORY_LOCK = threading.Lock()


def get_memory_store() -> MemoryStore:
    global _MEMORY
    if _MEMORY is None:
        with _MEMORY_LOCK:
            if _MEMORY is None:
                _MEMORY = MemoryStore()
    return _MEMORY


def get_internal_state_store() -> LongTermMemoryStore:
    global _INTERNAL_MEMORY
    if _INTERNAL_MEMORY is None:
        with _INTERNAL_MEMORY_LOCK:
            if _INTERNAL_MEMORY is None:
                _INTERNAL_MEMORY = LongTermMemoryStore()
    return _INTERNAL_MEMORY


def get_popup_user_store() -> LongTermMemoryStore:
    """Compatibility alias for the former durable-profile store name."""

    return get_internal_state_store()


def get_session_memory() -> MemoryStore:
    """Return the shared memory owner; retained as the public runtime getter."""

    return get_memory_store()
