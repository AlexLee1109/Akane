"""Bounded conversation history and selective, reliability-aware memories."""

from __future__ import annotations

import copy
import math
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from app.core.config import (
    CONVERSATION_STALE_DAYS,
    LONG_TERM_MEMORY_PATH,
    MAX_CONVERSATIONS,
    MEMORY_CONFIDENCE_WEIGHT,
    MEMORY_CONTINUITY_WEIGHT,
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
)
from app.core.persistence import atomic_write_json, read_json
from app.core.presence import (
    CLAIM_SECONDS,
    RETRY_SECONDS,
    LifeDecision,
    PresenceActivity,
    PresenceState,
    apply_life_decision,
    life_decision_rejection,
    next_decision_time,
    normalize_presence,
    validate_interest_addition,
)
from app.core.signal import (
    AffectTrace,
    EmotionState,
    SemanticEvent,
    TurnContext,
    TurnSignal,
    VALID_EMOTION_LABELS,
    advance_emotion,
    analyze_turn,
    build_affect_trace,
    evolve_emotion,
    message_similarity,
    normalized_signature,
    semantic_event_from_text,
    topic_overlap,
)
from app.core.utils import (
    OWNER_PROFILE_ID,
    canonical_profile_id,
    compact_text,
)

MEMORY_SCHEMA_VERSION = 2
LONG_TERM_MEMORY_SCHEMA_VERSION = 10
_MEMORY_PROMPT_INTRO = "A few past details may matter in this conversation:"
_MEMORY_PROMPT_OUTRO = (
    "Use them only when they genuinely improve the reply, and do not overstate uncertain details."
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
_AKANE_PREFERENCE_TAG = "akane-preference"
_STARTING_INTERESTS = ("anime", "manga", "VTubers")
_PREFERENCE_STANCES = {
    "likes",
    "dislikes",
    "curious",
    "mixed",
    "uncertain",
    "indifferent",
}
_AKANE_PREFERENCE_UPDATE = re.compile(
    r"\b(?:change(?:d)? your mind|different favorite|new favorite|not anymore|"
    r"reconsider|taste(?:s)? changed|prefer now|pick (?:a )?different|"
    r"choose (?:a )?different)\b",
    re.I,
)
_RELATIONSHIP_UPDATE_CATEGORIES = {
    "pattern",
    "shared_context",
    "unresolved_event",
    "resolved_event",
}
_MAX_RELATIONSHIP_PATTERNS = 24
_MAX_RELATIONSHIP_CONTEXTS = 24
_MAX_UNRESOLVED_EVENTS = 16
_EMOTION_CHANGES = {
    "started",
    "intensified",
    "sustained",
    "softened",
    "cleared",
    "replaced",
}
_RELATIONSHIP_RETRIEVAL_STOPWORDS = {
    "akane",
    "arcane",
    "discussion",
    "discussed",
    "relationship",
    "shared",
    "together",
}


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
_MEMORY_KINDS = {
    "working",
    "profile",
    "episode",
    "relationship",
    "self",
    "correction",
    "open_thread",
}
_MEMORY_STATUSES = {
    _ACTIVE_MEMORY,
    "superseded",
    "disputed",
    "resolved",
    "expired",
    "archived",
}
_THREAD_STATUSES = {
    _ACTIVE_MEMORY,
    "planned",
    "blocked",
    "resolved",
    "abandoned",
    "expired",
}
_SOURCE_AUTHORITY = {
    "unknown": 0,
    "speculative_inference": 1,
    "generated_assistant": 2,
    "conversation_summary": 3,
    "deterministic_analysis": 4,
    "trusted_memory": 5,
    "confirmed_action": 6,
    "recorded_offscreen": 7,
    "verified_interface": 8,
    "explicit_user": 9,
}
_ARCANE_ACTIVITY_KEY = "working:arcane-current-activity"
_ARCANE_ACTIVITY_TTL_SECONDS = 6 * 60 * 60
_COMPLETION_IMPORTANCE_TERMS = {
    "api",
    "bot",
    "bug",
    "build",
    "code",
    "coding",
    "compiler",
    "deployment",
    "feature",
    "implementation",
    "issue",
    "model",
    "pipeline",
    "project",
    "repository",
    "server",
    "system",
    "task",
    "test",
    "tests",
}


def _number(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _updated_emotion(
    current: EmotionState,
    update: dict[str, object] | None,
    *,
    now: float,
) -> EmotionState:
    """Accept only a fully validated model-proposed emotion state."""

    if not isinstance(update, dict):
        return current
    primary = compact_text(update.get("primary"), 32).lower()
    cause = compact_text(update.get("cause"), 100)
    change = compact_text(update.get("change"), 24).lower()
    intensity = update.get("intensity")
    if (
        primary not in VALID_EMOTION_LABELS
        or not cause
        or change not in _EMOTION_CHANGES
        or type(intensity) not in {int, float}
        or not math.isfinite(float(intensity))
    ):
        return current
    return EmotionState(
        primary=primary,
        intensity=max(0.0, min(1.0, float(intensity))),
        cause=cause,
        updated_at=now,
    )


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


def _complete_turns(turns: list[ChatTurn]) -> list[ChatTurn]:
    """Keep committed dialogue plus genuine, userless Akane initiatives."""

    complete: list[ChatTurn] = []
    index = 0
    while index < len(turns):
        user_turn = turns[index]
        if user_turn.role == "assistant" and user_turn.source == "initiative":
            complete.append(user_turn)
            index += 1
            continue
        if index + 1 >= len(turns):
            break
        assistant_turn = turns[index + 1]
        if user_turn.role == "user" and assistant_turn.role == "assistant":
            complete.extend((user_turn, assistant_turn))
            index += 2
        else:
            index += 1
    return complete


@dataclass(frozen=True, slots=True)
class InitiativeOpportunity:
    """One grounded reason Akane may choose to begin a conversation."""

    reason: str
    context: str
    importance: float
    created_at: float
    expires_at: float

    @classmethod
    def from_dict(cls, payload: object) -> "InitiativeOpportunity | None":
        if not isinstance(payload, dict):
            return None
        reason = compact_text(payload.get("reason"), 80)
        context = compact_text(payload.get("context"), 360)
        created_at = max(0.0, _number(payload.get("created_at"), 0.0))
        expires_at = max(0.0, _number(payload.get("expires_at"), 0.0))
        importance = max(0.0, min(1.0, _number(payload.get("importance"), 0.0)))
        if not reason or not context or expires_at <= created_at:
            return None
        return cls(reason, context, importance, created_at, expires_at)


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
    unresolved_problem: bool = False
    repeated_topic_count: int = 0
    last_outcome: str = ""
    correction: str = ""
    recent_events: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)
    committed_request_ids: list[str] = field(default_factory=list)
    initiative_opportunity: InitiativeOpportunity | None = None
    initiative_cooldown_until: float = 0.0
    initiative_handled_contexts: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, key: str, payload: object) -> "ConversationRecord | None":
        if not isinstance(payload, dict):
            return None
        conversation_id = compact_text(payload.get("conversation_id") or key, 120)
        raw_profile_id = compact_text(payload.get("profile_id"), 120)
        if not conversation_id or not raw_profile_id:
            return None
        profile_id = canonical_profile_id(raw_profile_id)

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
            unresolved_problem=bool(payload.get("unresolved_problem")),
            repeated_topic_count=max(
                0,
                min(20, int(_number(payload.get("repeated_topic_count"), 0))),
            ),
            last_outcome=compact_text(payload.get("last_outcome"), 40).lower(),
            correction=compact_text(payload.get("correction"), 120),
            recent_events=[
                value
                for item in (payload.get("recent_events") or [])[-5:]
                if (value := compact_text(item, 48))
            ],
            updated_at=updated_at,
            committed_request_ids=[
                value
                for item in (payload.get("committed_request_ids") or [])[-32:]
                if (value := compact_text(item, 160))
            ],
            initiative_opportunity=InitiativeOpportunity.from_dict(
                payload.get("initiative_opportunity")
            ),
            initiative_cooldown_until=max(
                0.0, _number(payload.get("initiative_cooldown_until"), 0.0)
            ),
            initiative_handled_contexts=tuple(
                value
                for item in (payload.get("initiative_handled_contexts") or [])[-5:]
                if (value := compact_text(item, 360))
            ),
        )

    def selected_summary_turns(self, query: str = "") -> tuple[ChatTurn, ...]:
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
        return tuple(turns)

    def summary_text(self, query: str = "") -> str:
        turns = self.selected_summary_turns(query)
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
            "unresolved_problem": self.unresolved_problem,
            "repeated_topic_count": self.repeated_topic_count,
            "last_outcome": self.last_outcome,
            "correction": self.correction,
            "updated_at": self.updated_at,
            "initiative_opportunity": (
                asdict(self.initiative_opportunity)
                if self.initiative_opportunity
                else None
            ),
            "initiative_cooldown_until": self.initiative_cooldown_until,
        }


@dataclass(frozen=True, slots=True)
class MemoryContext:
    relationship: str
    recent_turns: tuple[ChatTurn, ...]
    memory_ids: tuple[str, ...] = ()
    memory_contents: tuple[str, ...] = ()
    earlier_turns: tuple[ChatTurn, ...] = ()
    current_topic: str = ""
    current_task: str = ""
    unresolved_problem: bool = False
    repeated_topic_count: int = 0
    last_outcome: str = ""
    updated_at: float = 0.0


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
    kind: str = ""
    source_type: str = "unknown"
    source_reference: str = ""
    canonical_key: str = ""
    scope: str = "profile"
    updated_at: float = 0.0
    evidence_refs: tuple[str, ...] = ()
    thread_status: str = ""

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
        source = compact_text(payload.get("source"), 48) or "unknown"
        kind = _memory_kind_from_fields(
            compact_text(payload.get("kind"), 24).lower(),
            category,
            tags,
        )
        source_type = _source_type(
            compact_text(payload.get("source_type"), 32).lower() or source
        )
        status = compact_text(payload.get("status"), 24).lower() or _ACTIVE_MEMORY
        if status == "contradicted":
            status = "disputed"
        if status not in _MEMORY_STATUSES:
            status = "archived"
        if kind in {"profile", "self", "relationship", "correction"} and source_type in {
            "conversation_summary",
            "generated_assistant",
            "speculative_inference",
        }:
            status = "archived"
        raw_evidence = payload.get("evidence_refs")
        evidence_refs = tuple(
            value
            for item in (raw_evidence if isinstance(raw_evidence, (list, tuple)) else ())
            if (value := compact_text(item, 100))
        )[:12]
        canonical_key = compact_text(payload.get("canonical_key"), 120).lower()
        if not canonical_key:
            canonical_key = _canonical_key(kind, category, tags, content)
        thread_status = compact_text(payload.get("thread_status"), 24).lower()
        if kind == "open_thread":
            thread_status = thread_status or (
                status if status in _THREAD_STATUSES else _ACTIVE_MEMORY
            )
            if thread_status not in _THREAD_STATUSES:
                thread_status = "expired"
        else:
            thread_status = ""
        return cls(
            id=compact_text(payload.get("id"), 80) or uuid.uuid4().hex,
            content=content,
            category=category,
            created_at=created_at,
            last_used_at=last_used_at,
            importance=max(0.0, min(1.0, _number(payload.get("importance"), 0.5))),
            confidence=max(0.0, min(1.0, _number(payload.get("confidence"), 1.0))),
            source=source,
            access_count=access_count,
            tags=tags,
            status=status,
            expires_at=expires_at,
            superseded_by=compact_text(payload.get("superseded_by"), 80) or None,
            kind=kind,
            source_type=source_type,
            source_reference=compact_text(payload.get("source_reference"), 100),
            canonical_key=canonical_key,
            scope=compact_text(payload.get("scope"), 120).lower() or "profile",
            updated_at=max(
                created_at,
                _number(payload.get("updated_at"), created_at),
            ),
            evidence_refs=evidence_refs,
            thread_status=thread_status,
        )

    def is_available(self, now: float) -> bool:
        return (
            self.status == _ACTIVE_MEMORY
            and self.thread_status not in {"resolved", "abandoned", "expired"}
            and not (
                self.expires_at is not None and self.expires_at <= now
            )
        )


@dataclass(frozen=True, slots=True)
class InteractionEvent:
    kind: str
    summary: str
    created_at: float
    resolved: bool = False

@dataclass(frozen=True, slots=True)
class WorkingMemory:
    current_topic: str = ""
    current_task: str = ""
    unresolved_problem: bool = False
    repeated_topic_count: int = 0
    last_outcome: str = ""
    last_user_summary: str = ""
    recent_events: tuple[InteractionEvent, ...] = ()

@dataclass(frozen=True, slots=True)
class InternalState:
    emotion: EmotionState
    presence: PresenceState
    memories: tuple[Memory, ...] = ()
    interests: tuple[str, ...] = _STARTING_INTERESTS
    preferences: tuple["AkanePreference", ...] = ()
    relationship: "RelationshipState" = field(default_factory=lambda: RelationshipState())
    updated_at: float = 0.0
    version: int = 4


@dataclass(frozen=True, slots=True)
class AkanePreference:
    """One model-proposed personal conclusion, separate from user memories."""

    topic: str
    stance: str
    strength: float
    reason: str
    updated_at: float

    @classmethod
    def from_dict(cls, payload: object) -> "AkanePreference | None":
        if not isinstance(payload, dict):
            return None
        topic = compact_text(payload.get("topic"), 140)
        stance = compact_text(payload.get("stance"), 24).lower()
        reason = compact_text(payload.get("reason"), 240)
        if not topic or stance not in _PREFERENCE_STANCES or not reason:
            return None
        return cls(
            topic=topic,
            stance=stance,
            strength=max(0.0, min(1.0, _number(payload.get("strength"), 0.0))),
            reason=reason,
            updated_at=max(0.0, _number(payload.get("updated_at"), 0.0)),
        )


@dataclass(frozen=True, slots=True)
class RelationshipEntry:
    """One model-proposed observation about Akane and this profile's shared history."""

    summary: str
    importance: float
    confidence: float
    updated_at: float
    status: str = "active"

    @classmethod
    def from_dict(cls, payload: object, *, unresolved: bool = False) -> "RelationshipEntry | None":
        if not isinstance(payload, dict):
            return None
        summary = compact_text(payload.get("summary"), 240)
        if not summary:
            return None
        status = compact_text(payload.get("status"), 24).lower() if unresolved else "active"
        if unresolved and status not in {"unresolved", "resolved"}:
            status = "unresolved"
        return cls(
            summary=summary,
            importance=max(0.0, min(1.0, _number(payload.get("importance"), 0.0))),
            confidence=max(0.0, min(1.0, _number(payload.get("confidence"), 0.0))),
            updated_at=max(0.0, _number(payload.get("updated_at"), 0.0)),
            status=status,
        )


@dataclass(frozen=True, slots=True)
class RelationshipState:
    patterns: tuple[RelationshipEntry, ...] = ()
    shared_context: tuple[RelationshipEntry, ...] = ()
    unresolved_events: tuple[RelationshipEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class InternalTurnResult:
    state: InternalState
    signal: TurnSignal
    recalled_memories: tuple[Memory, ...]
    affect_trace: AffectTrace | None = None
    working_context: WorkingMemory = WorkingMemory()
    grounded_activity_source: str = "none"
    grounded_activity_age_seconds: float = 0.0
    memory_trace: dict[str, object] = field(default_factory=dict)
    memory_uses: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryRetrievalConfig:
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
    supersedes: bool = False
    kind: str = "profile"
    source_type: str = "explicit_user"
    source_reference: str = ""
    canonical_key: str = ""
    scope: str = "profile"
    expires_at: float | None = None
    evidence_refs: tuple[str, ...] = ()
    thread_status: str = ""


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
_EXPLICIT_MEMORY_CORRECTION = re.compile(
    r"(?:\b(?:actually|correction|i meant|instead|changed my mind|no longer|not anymore)\b"
    r"|(?:^|\s)no,\s+(?:my|i)\b|\bthat(?:'s| is) (?:not right|wrong)\b)",
    re.I,
)


def _extract_memory_candidates(
    text: str,
    slot_value: Callable[[str], str],
    *,
    source_reference: str = "",
    scope: str = "profile",
    semantic_event: SemanticEvent | None = None,
    now: float = 0.0,
) -> tuple[_MemoryCandidate, ...]:
    value = compact_text(text, 700)
    lower = value.lower()
    if (
        not value
        or any(marker in lower for marker in ("hypothetically", "just kidding", "for example"))
        or any(marker in lower for marker in _SENSITIVE_TERMS)
    ):
        return ()
    candidates: list[_MemoryCandidate] = []
    event = semantic_event or SemanticEvent()

    if (
        event.event_type == "activity"
        and event.actor in {"Arcane", "shared"}
        and event.temporal_state == "current"
        and event.status in {"ongoing", "started", "switched", "resumed", "paused"}
        and event.subject
    ):
        label = "shared current activity" if event.actor == "shared" else "Arcane current activity"
        candidates.append(
            _MemoryCandidate(
                content=f"{label}: {compact_text(event.subject, 170)} ({event.status}).",
                category="unfinished_topic",
                importance=0.58,
                confidence=event.confidence,
                tags=("arcane-current-activity", f"activity:{slot_value(event.subject)}"),
                supersedes=True,
                kind="working",
                source_type="explicit_user",
                canonical_key=_ARCANE_ACTIVITY_KEY,
                scope=f"conversation:{compact_text(scope, 100).lower()}",
                expires_at=max(0.0, now) + _ARCANE_ACTIVITY_TTL_SECONDS,
            )
        )

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
        slot = "slot:preference:" + slot_value(preference)
        candidates.append(_candidate(f"The user {action} {preference}.", category, 0.82, (slot,)))

    for match in _PROFILE_FAVORITE_PATTERN.finditer(value):
        subject = compact_text(match.group("subject"), 48).strip(" ,;:-")
        favorite = compact_text(match.group("value"), 100).strip(" ,;:-")
        if subject and favorite:
            slot = "slot:favorite:" + slot_value(subject)
            candidates.append(
                _candidate(f"The user's favorite {subject} is {favorite}.", "stable_fact", 0.88, (slot,))
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
                        fact_phrases[label].format(value=fact), "stable_fact", 0.86,
                        ("slot:" + label.replace(" ", "-"),),
                    )
                )

    if event.event_type != "activity" and (match := _PROJECT_PATTERN.search(value)):
        project = compact_text(match.group("value"), 180).strip(" ,;:-")
        if project:
            candidates.append(_candidate(f"The user is working on {project}.", "stable_fact", 0.78, ("project",)))
    if match := _GOAL_PATTERN.search(value):
        goal = compact_text(match.group("value"), 180).strip(" ,;:-")
        if goal:
            candidates.append(_candidate(f"The user's long-term goal is {goal}.", "stable_fact", 0.82, ("goal",)))
    if event.event_type != "completion" and (match := _EPISODE_PATTERN.search(value)):
        event = " ".join(match.group("event").lower().split())
        detail = compact_text(match.group("value"), 180).strip(" ,;:-")
        if detail:
            candidates.append(_candidate(f"The user {event} {detail}.", "episode", 0.70, ("event",)))
    if match := _INTERACTION_PREFERENCE_PATTERN.search(value):
        instruction = compact_text(match.group("value"), 150).strip(" ,;:-")
        if instruction:
            candidates.append(
                replace(
                    _candidate(
                        f"The user prefers that Akane not {instruction.lower()}.",
                        "tendency",
                        0.90,
                        (
                            "interaction-style",
                            "slot:behavior:" + slot_value(instruction),
                        ),
                    ),
                    kind="correction",
                    supersedes=True,
                )
            )

    unique: dict[str, _MemoryCandidate] = {}
    explicit_correction = bool(_EXPLICIT_MEMORY_CORRECTION.search(value))
    for candidate in candidates:
        if explicit_correction and any(tag.startswith("slot:") for tag in candidate.tags):
            candidate = replace(
                candidate,
                confidence=max(0.90, candidate.confidence),
                supersedes=True,
                kind="correction",
            )
        candidate = replace(
            candidate,
            source_reference=compact_text(source_reference, 100),
            scope=scope,
            evidence_refs=(compact_text(source_reference, 100),) if source_reference else (),
            canonical_key=_candidate_key(candidate),
        )
        unique.setdefault(normalized_signature(candidate.content), candidate)
    return tuple(unique.values())


def _candidate(
    content: str,
    category: str,
    importance: float,
    tags: tuple[str, ...],
    confidence: float = 0.78,
) -> _MemoryCandidate:
    kind = {
        "episode": "episode",
        "task_outcome": "episode",
        "unfinished_topic": "open_thread",
    }.get(category, "profile")
    return _MemoryCandidate(
        content=compact_text(content, 240),
        category=category,
        importance=importance,
        confidence=max(0.0, min(1.0, confidence)),
        tags=tags,
        kind=kind,
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


def _source_type(value: str) -> str:
    source = compact_text(value, 48).lower().replace("-", "_")
    if source in _SOURCE_AUTHORITY:
        return source
    if source in {
        "user",
        "explicit",
        "chat:explicit_user",
        "chat:correction",
    } or source.startswith(("owner", "arcane", "user_explicit")):
        return "explicit_user"
    if source.startswith(("workspace", "vscode", "interface", "verified")):
        return "verified_interface"
    if source in {"offscreen_life", "offscreen_schedule"}:
        return "recorded_offscreen"
    if source.startswith(("confirmed", "application_action", "app_action")):
        return "confirmed_action"
    if source.startswith("trusted"):
        return "trusted_memory"
    if source.startswith(("chat:task_state", "deterministic")):
        return "deterministic_analysis"
    if "summary" in source:
        return "generated_assistant" if "assistant" in source else "conversation_summary"
    if source.startswith(("assistant", "generated")):
        return "generated_assistant"
    if source.startswith(("inferred", "speculative")):
        return "speculative_inference"
    return "unknown"


def _memory_kind_from_fields(
    kind: str,
    category: str,
    tags: tuple[str, ...],
) -> str:
    if kind in _MEMORY_KINDS:
        return kind
    if _AKANE_PREFERENCE_TAG in tags or "life-activity" in tags:
        return "self"
    return {
        "episode": "episode",
        "task_outcome": "episode",
        "unfinished_topic": "open_thread",
    }.get(category, "profile")


def _memory_kind(memory: Memory) -> str:
    return _memory_kind_from_fields(memory.kind, memory.category, memory.tags)


def _memory_source_type(memory: Memory) -> str:
    normalized = _source_type(memory.source_type)
    return _source_type(memory.source) if normalized == "unknown" else normalized


def _canonical_key(
    kind: str,
    category: str,
    tags: tuple[str, ...],
    content: str,
) -> str:
    slots = sorted(tag for tag in tags if tag.startswith(("slot:", "task:")))
    if slots:
        return compact_text(slots[0], 120).lower()
    terms = "-".join(sorted(_memory_terms(content))[:8])
    return compact_text(f"{kind or category}:{terms}", 120).lower()


def _candidate_key(candidate: _MemoryCandidate) -> str:
    return candidate.canonical_key or _canonical_key(
        candidate.kind,
        candidate.category,
        candidate.tags,
        candidate.content,
    )


def _authority(value: str) -> int:
    return _SOURCE_AUTHORITY.get(_source_type(value), 0)


def _scope_matches(memory_scope: str, requested_scope: str) -> bool:
    stored = compact_text(memory_scope, 120).lower() or "profile"
    requested = compact_text(requested_scope, 120).lower() or "profile"
    if stored in {"global", "profile"}:
        return True
    return stored == requested or stored == f"conversation:{requested}"


def _candidate_allowed(candidate: _MemoryCandidate) -> tuple[bool, str]:
    source_type = _source_type(candidate.source_type)
    kind = candidate.kind
    if kind not in _MEMORY_KINDS:
        return False, "unsupported_kind"
    if source_type in {"generated_assistant", "speculative_inference"}:
        return False, "untrusted_generated_or_speculative_source"
    if kind in {"profile", "self", "relationship", "correction"} and _authority(
        source_type
    ) < _SOURCE_AUTHORITY["trusted_memory"]:
        return False, "insufficient_source_authority"
    if kind == "relationship" and not candidate.evidence_refs:
        return False, "relationship_requires_evidence"
    if kind == "self" and source_type not in {
        "explicit_user",
        "verified_interface",
        "recorded_offscreen",
        "confirmed_action",
        "trusted_memory",
    }:
        return False, "self_memory_requires_grounded_source"
    if kind == "working" and not candidate.expires_at:
        return False, "working_memory_requires_expiration"
    return True, ""


class MemoryStore:
    """Own recent turns and rolling excerpts for each conversation."""

    def __init__(self, path: Path = MEMORY_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._conversations: dict[str, ConversationRecord] = {}
        self._load()

    def build_context(
        self,
        profile_id: str,
        conversation_id: str,
        *,
        display_name: str = "",
        query: str = "",
        include_memory: bool = True,
    ) -> MemoryContext:
        profile = canonical_profile_id(profile_id)
        conversation_key = _key(conversation_id, "popup:default")
        with self._lock:
            record = self._conversations.get(conversation_key)
            if record is not None and record.profile_id != profile:
                raise ValueError("Conversation belongs to a different profile.")
            recent = (
                tuple(_complete_turns(record.recent_turns)[-12:])
                if record and include_memory
                else ()
            )
            relationship = _relationship_context(
                display_name,
                bool(record and record.recent_turns and include_memory),
            )
            return MemoryContext(
                relationship=relationship,
                recent_turns=recent,
                earlier_turns=(),
                current_topic=record.recent_topic if record and include_memory else "",
                current_task=record.current_task if record and include_memory else "",
                unresolved_problem=(
                    record.unresolved_problem if record and include_memory else False
                ),
                repeated_topic_count=(
                    record.repeated_topic_count if record and include_memory else 0
                ),
                last_outcome=record.last_outcome if record and include_memory else "",
                updated_at=record.updated_at if record and include_memory else 0.0,
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
        request_id: str = "",
    ) -> None:
        profile = canonical_profile_id(profile_id)
        conversation_key = _key(conversation_id, "popup:default")
        now = time.time()
        normalized_request_id = compact_text(request_id, 160)
        user_turn = ChatTurn(
            normalized_request_id or uuid.uuid4().hex,
            "user",
            user_text[:_MAX_TURN_CHARS],
            now,
            source,
        )
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
                record = copy.copy(record)
                record.recent_turns = list(record.recent_turns)
                record.summary_turns = list(record.summary_turns)
                record.pending_summary_turns = list(record.pending_summary_turns)
                record.recent_events = list(record.recent_events)
                record.committed_request_ids = list(record.committed_request_ids)

            self._conversations = previous_conversations.copy()
            self._conversations[conversation_key] = record

            if normalized_request_id and normalized_request_id in record.committed_request_ids:
                return

            record.recent_turns.extend((user_turn, assistant_turn))
            record.initiative_opportunity = None
            previous_topic = record.recent_topic
            record.recent_topic = compact_text(signal.topic, 80)
            record.recent_intent = signal.intent
            record.recent_user_tone = signal.tone
            record.current_task = compact_text(signal.task, 100)
            same_topic = bool(
                previous_topic
                and signal.topic
                and topic_overlap(previous_topic, signal.topic) >= 0.45
            )
            record.repeated_topic_count = (
                min(20, record.repeated_topic_count + 1) if same_topic else 1
            )
            if signal.task_failure:
                record.unresolved_problem = True
                record.last_outcome = "technical_failure"
            elif signal.task_success:
                record.unresolved_problem = False
                record.last_outcome = "technical_success"
            elif signal.correction_requested:
                record.last_outcome = "correction"
            if signal.correction:
                record.correction = compact_text(signal.correction, 120)
            if signal.trigger:
                record.recent_events.append(compact_text(signal.trigger, 48))
                record.recent_events = record.recent_events[-5:]
            record.updated_at = now
            if normalized_request_id:
                record.committed_request_ids.append(normalized_request_id)
                record.committed_request_ids = record.committed_request_ids[-32:]
            self._trim_conversation(record)

            self._prune(now)
            try:
                self._persist()
            except Exception:
                self._conversations = previous_conversations
                raise

    def record_silence(
        self,
        *,
        profile_id: str,
        conversation_id: str,
        signal: TurnSignal,
        request_id: str = "",
    ) -> None:
        """Preserve a model-chosen silence without manufacturing dialogue."""

        profile = canonical_profile_id(profile_id)
        conversation_key = _key(conversation_id, "popup:default")
        now = time.time()
        normalized_request_id = compact_text(request_id, 160)
        with self._lock:
            previous_conversations = self._conversations
            record = previous_conversations.get(conversation_key)
            if record is None:
                record = ConversationRecord(conversation_key, profile)
            elif record.profile_id != profile:
                raise ValueError("Conversation belongs to a different profile.")
            else:
                record = copy.copy(record)
                record.recent_turns = list(record.recent_turns)
                record.summary_turns = list(record.summary_turns)
                record.pending_summary_turns = list(record.pending_summary_turns)
                record.recent_events = list(record.recent_events)
                record.committed_request_ids = list(record.committed_request_ids)

            if normalized_request_id and normalized_request_id in record.committed_request_ids:
                return

            self._conversations = previous_conversations.copy()
            self._conversations[conversation_key] = record
            previous_topic = record.recent_topic
            record.initiative_opportunity = None
            record.recent_topic = compact_text(signal.topic, 80)
            record.recent_intent = signal.intent
            record.recent_user_tone = signal.tone
            record.current_task = compact_text(signal.task, 100)
            record.repeated_topic_count = (
                min(20, record.repeated_topic_count + 1)
                if previous_topic and topic_overlap(previous_topic, signal.topic) >= 0.45
                else 1
            )
            record.last_outcome = "akane_silence"
            record.recent_events.append("akane_silence")
            record.recent_events = record.recent_events[-5:]
            record.updated_at = now
            if normalized_request_id:
                record.committed_request_ids.append(normalized_request_id)
                record.committed_request_ids = record.committed_request_ids[-32:]
            self._prune(now)
            try:
                self._persist()
            except Exception:
                self._conversations = previous_conversations
                raise

    def claim_initiative_opportunity(
        self,
        *,
        profile_id: str,
        conversation_id: str,
        candidates: tuple[InitiativeOpportunity, ...],
        now: float,
        active_window_seconds: float,
    ) -> InitiativeOpportunity | None:
        """Return one persisted opportunity only when the conversation is idle."""

        profile = canonical_profile_id(profile_id)
        conversation_key = _key(conversation_id, "popup:default")
        current = max(0.0, float(now))
        with self._lock:
            previous_conversations = self._conversations
            existing = previous_conversations.get(conversation_key)
            if existing is None or existing.profile_id != profile:
                return None
            record = copy.copy(existing)
            record.recent_turns = list(record.recent_turns)
            record.summary_turns = list(record.summary_turns)
            record.pending_summary_turns = list(record.pending_summary_turns)
            record.recent_events = list(record.recent_events)
            record.committed_request_ids = list(record.committed_request_ids)
            changed = False
            if (
                record.initiative_opportunity is not None
                and record.initiative_opportunity.expires_at <= current
            ):
                record.initiative_opportunity = None
                changed = True
            if current - record.updated_at < active_window_seconds:
                if changed:
                    self._replace_conversation(conversation_key, record, previous_conversations)
                return None
            if record.initiative_opportunity is not None:
                if changed:
                    self._replace_conversation(conversation_key, record, previous_conversations)
                return record.initiative_opportunity
            if record.initiative_cooldown_until > current:
                if changed:
                    self._replace_conversation(conversation_key, record, previous_conversations)
                return None
            eligible = tuple(
                candidate
                for candidate in candidates
                if candidate.importance >= 0.55
                and candidate.expires_at > current
                and candidate.context not in record.initiative_handled_contexts
            )
            if not eligible:
                if changed:
                    self._replace_conversation(conversation_key, record, previous_conversations)
                return None
            opportunity = max(
                eligible,
                key=lambda item: (item.importance, item.created_at, item.context),
            )
            record.initiative_opportunity = opportunity
            self._replace_conversation(conversation_key, record, previous_conversations)
            return opportunity

    def record_initiative_result(
        self,
        *,
        profile_id: str,
        conversation_id: str,
        opportunity: InitiativeOpportunity,
        message: str,
        used: bool,
        cooldown_seconds: float,
        now: float | None = None,
    ) -> None:
        """Record an offered initiative without inventing a user turn."""

        profile = canonical_profile_id(profile_id)
        conversation_key = _key(conversation_id, "popup:default")
        current = time.time() if now is None else max(0.0, float(now))
        with self._lock:
            previous_conversations = self._conversations
            existing = previous_conversations.get(conversation_key)
            if existing is None or existing.profile_id != profile:
                return
            record = copy.copy(existing)
            record.recent_turns = list(record.recent_turns)
            record.summary_turns = list(record.summary_turns)
            record.pending_summary_turns = list(record.pending_summary_turns)
            record.recent_events = list(record.recent_events)
            record.committed_request_ids = list(record.committed_request_ids)
            pending = record.initiative_opportunity
            if pending is None or pending.context != opportunity.context:
                return
            record.initiative_opportunity = None
            record.initiative_cooldown_until = current + max(0.0, cooldown_seconds)
            record.initiative_handled_contexts = (
                *record.initiative_handled_contexts,
                opportunity.context,
            )[-5:]
            record.recent_events.append("initiative_used" if used else "initiative_offered")
            record.recent_events = record.recent_events[-5:]
            record.last_outcome = "initiative_used" if used else "initiative_declined"
            if used and message:
                record.recent_turns.append(
                    ChatTurn(uuid.uuid4().hex, "assistant", message[:_MAX_TURN_CHARS], current, "initiative")
                )
                self._trim_conversation(record)
            record.updated_at = current
            self._replace_conversation(conversation_key, record, previous_conversations)

    def messages(
        self,
        conversation_id: str,
        profile_id: str | None = None,
    ) -> list[dict[str, str]]:
        with self._lock:
            record = self._conversations.get(_key(conversation_id, "popup:default"))
            if record and profile_id and record.profile_id != canonical_profile_id(profile_id):
                return []
            return [turn.as_message() for turn in record.recent_turns] if record else []

    def last_assistant_turn_at(self, profile_id: str) -> float | None:
        """Return the newest persisted, completed assistant turn for a profile."""

        profile = canonical_profile_id(profile_id)
        with self._lock:
            timestamps = [
                turn.timestamp
                for record in self._conversations.values()
                if record.profile_id == profile
                for turn in _complete_turns(
                    [*record.summary_turns, *record.recent_turns]
                )
                if turn.role == "assistant"
            ]
        return max(timestamps) if timestamps else None

    def reply_for_request(
        self,
        conversation_id: str,
        profile_id: str,
        request_id: str,
    ) -> str | None:
        request = compact_text(request_id, 160)
        if not request:
            return None
        with self._lock:
            record = self._conversations.get(_key(conversation_id, "popup:default"))
            if record is None or record.profile_id != canonical_profile_id(profile_id):
                return None
            if request not in record.committed_request_ids:
                return None
            for index in range(len(record.recent_turns) - 2, -1, -1):
                user_turn = record.recent_turns[index]
                if user_turn.turn_id != request or index + 1 >= len(record.recent_turns):
                    continue
                assistant_turn = record.recent_turns[index + 1]
                if assistant_turn.role == "assistant":
                    return assistant_turn.content
            return ""

    def public_conversation(
        self,
        conversation_id: str,
        profile_id: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            record = self._conversations.get(_key(conversation_id, "popup:default"))
            if record and profile_id and record.profile_id != canonical_profile_id(profile_id):
                return {}
            return record.public_state() if record else {}

    def recent_profile_topic(self, profile_id: str) -> str:
        """Return the newest persisted conversation topic for one profile."""

        profile = canonical_profile_id(profile_id)
        with self._lock:
            candidates = (
                record
                for record in self._conversations.values()
                if record.profile_id == profile and record.recent_topic
            )
            latest = max(candidates, key=lambda item: item.updated_at, default=None)
            return latest.recent_topic if latest is not None else ""

    def clear_conversation(
        self,
        conversation_id: str,
        profile_id: str | None = None,
    ) -> bool:
        with self._lock:
            key = _key(conversation_id, "popup:default")
            record = self._conversations.get(key)
            if record and profile_id and record.profile_id != canonical_profile_id(profile_id):
                return False
            if record is None:
                return False
            self._conversations.pop(key, None)
            self._persist()
            return True

    def clear_profile(self, profile_id: str) -> None:
        profile = canonical_profile_id(profile_id)
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
        record.recent_turns = _complete_turns(record.recent_turns)[-12:]
        record.summary_turns.clear()
        record.pending_summary_turns.clear()

    def _replace_conversation(
        self,
        key: str,
        record: ConversationRecord,
        previous_conversations: dict[str, ConversationRecord],
    ) -> None:
        self._conversations = previous_conversations.copy()
        self._conversations[key] = record
        try:
            self._persist()
        except Exception:
            self._conversations = previous_conversations
            raise

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
                or int(payload.get("schema_version", 0)) not in {1, MEMORY_SCHEMA_VERSION}
            ):
                raise ValueError("unsupported schema")
            conversations = payload.get("conversations")
            if not isinstance(conversations, dict):
                raise ValueError("invalid memory document")
            loaded: dict[str, ConversationRecord] = {}
            migrated = False
            for key, value in conversations.items():
                record = ConversationRecord.from_dict(str(key), value)
                if record is None:
                    continue
                raw_profile = (
                    compact_text(value.get("profile_id"), 120)
                    if isinstance(value, dict)
                    else ""
                )
                migrated = migrated or record.profile_id != raw_profile
                loaded[record.conversation_id] = record
            self._conversations = loaded
            self._prune(time.time())
            if migrated:
                self._persist()
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
    current = max(0.0, _number(time.time() if now is None else now, 0.0))
    emotion = EmotionState(updated_at=current)
    return InternalState(
        emotion=emotion,
        presence=PresenceState(),
        updated_at=current,
        version=4,
    )


def _semantic_memory_match(memory: Memory, event: SemanticEvent) -> bool:
    if not event.subject:
        return False
    subject_terms = _memory_terms(event.subject)
    memory_terms = _memory_terms(memory.content)
    shared = subject_terms & memory_terms
    return bool(shared) and (
        len(shared) >= 2
        or len(shared) / max(1, min(len(subject_terms), len(memory_terms))) >= 0.50
        or topic_overlap(memory.content, event.subject) >= 0.38
    )


def _completion_context(
    event: SemanticEvent,
    memories: tuple[Memory, ...],
    working: WorkingMemory,
    *,
    now: float,
    scope: str,
) -> tuple[bool, bool, tuple[str, ...]]:
    if not event.confirmed_completion:
        return False, False, ()
    matches: list[Memory] = []
    for memory in memories:
        if not memory.is_available(now) or not _scope_matches(memory.scope, scope):
            continue
        kind = _memory_kind(memory)
        if kind == "working" and memory.canonical_key == _ARCANE_ACTIVITY_KEY:
            if _semantic_memory_match(memory, event):
                matches.append(memory)
        elif kind == "open_thread" and _semantic_memory_match(memory, event):
            matches.append(memory)
        elif kind in {"profile", "episode"} and _semantic_memory_match(memory, event):
            matches.append(memory)

    subject_terms = _memory_terms(event.subject)
    explicitly_important = bool(subject_terms & _COMPLETION_IMPORTANCE_TERMS)
    working_match = bool(
        working.current_task
        and topic_overlap(
            working.current_task,
            event.subject,
        )
        >= 0.32
    )
    continuity_match = bool(
        (working.current_task or working.current_topic)
        and topic_overlap(
            " ".join((working.current_task, working.current_topic)),
            event.subject,
        )
        >= 0.32
    )
    thread_match = working.unresolved_problem and continuity_match or any(
        _memory_kind(memory) == "open_thread" for memory in matches
    )
    durable_match = any(_memory_kind(memory) != "working" for memory in matches)
    meaningful = bool(
        event.actor == "shared"
        or explicitly_important
        or working_match
        or durable_match
    )
    return meaningful, thread_match, tuple(memory.id for memory in matches)


def process_internal_turn(
    user_text: str,
    state: InternalState | None = None,
    *,
    now: float | None = None,
    retrieval: MemoryRetrievalConfig | None = None,
    include_memory: bool = True,
    code_context_requested: bool = False,
    code_context_attached: bool = False,
    autonomous: bool = False,
    familiar_relationship: bool = False,
    working_context: WorkingMemory | None = None,
    recent_turns: tuple[ChatTurn, ...] = (),
    activity_scope: str = "profile",
    profile_seed: str = "local:owner",
) -> InternalTurnResult:
    """Purely appraise a turn and return the proposed coordinated state."""

    current = max(0.0, _number(time.time() if now is None else now, 0.0))
    previous = state if state is not None else new_internal_state(current)
    current = max(current, previous.updated_at, previous.emotion.updated_at)
    working = working_context or WorkingMemory()
    retrieval_config = retrieval or MemoryRetrievalConfig()
    semantic_event = (
        SemanticEvent() if autonomous else semantic_event_from_text(user_text)
    )
    completion_meaningful, completion_resolves_thread, completion_matches = (
        _completion_context(
            semantic_event,
            previous.memories,
            working,
            now=current,
            scope=activity_scope,
        )
    )
    autonomous_memories: list[Memory] = []
    for memory in previous.memories:
        if autonomous and _AKANE_PREFERENCE_TAG in memory.tags:
            autonomous_memories.append(memory)

    mood_evolution = evolve_emotion(
        previous.emotion,
        now=current,
        profile_seed=profile_seed,
    )
    presence = previous.presence
    event_emotion = mood_evolution.state
    event_effects: dict[str, float] = {}
    latest_activity = presence.current_activity or presence.previous_activity

    memory_decisions: list[str] = []
    source_reference = f"chat:{compact_text(activity_scope, 80)}:{current:.6f}"
    candidates = (
        ()
        if autonomous
        else _extract_memory_candidates(
            user_text,
            _slot_value,
            source_reference=source_reference,
            scope=activity_scope,
            semantic_event=semantic_event,
            now=current,
        )
    )
    needs_memory_mutation = bool(
        candidates
        or semantic_event.event_type in {"completion", "failure"}
    )
    memories = (
        copy.deepcopy(list(previous.memories))
        if needs_memory_mutation
        else list(previous.memories)
    )
    for candidate in candidates:
        _insert_into_memories(
            memories,
            candidate,
            source=(
                "chat:correction"
                if candidate.kind == "correction"
                else candidate.source_type
            ),
            created_at=current,
            trace=memory_decisions,
        )

    # Retrieval represents established state from before this message. Candidate
    # writes remain available for commit, but are not echoed back as remembered
    # history during the turn that created them.
    retrieval_memories = previous.memories
    query_parts = (user_text, working.current_task)
    if autonomous:
        retrieval_memories = tuple(autonomous_memories)
        query_parts = (
            latest_activity.fact() if latest_activity else "",
            *(memory.content for memory in retrieval_memories[-4:]),
        )
    appraisal_query = " ".join(
        part for part in query_parts if part
    )
    recalled = (
        _retrieve_memories(
            retrieval_memories,
            appraisal_query,
            current,
            retrieval_config,
            working,
            scope=activity_scope,
        )
        if include_memory
        else ()
    )
    context = TurnContext(
        current_topic=working.current_topic,
        current_task=working.current_task,
        unresolved_problem=working.unresolved_problem,
        repeated_topic_count=working.repeated_topic_count,
        last_outcome=working.last_outcome,
        memory_relevance=max(
            (memory.importance * memory.confidence for memory in recalled),
            default=0.0,
        ),
        meaningful_memory=any(
            memory.importance >= 0.80
            or memory.category in {"episode", "task_outcome", "unfinished_topic"}
            for memory in recalled
        ),
        familiar_relationship=familiar_relationship,
        completion_meaningful=completion_meaningful,
        completion_resolves_thread=completion_resolves_thread,
        recent_turns=tuple((turn.role, turn.content) for turn in recent_turns),
    )
    signal = analyze_turn(
        "" if autonomous else user_text,
        emotion_state=event_emotion,
        turn_context=context,
        now=current,
        emotion_state_is_current=True,
        code_context_requested=code_context_requested,
        code_context_attached=code_context_attached,
        semantic_event=semantic_event,
    )
    if autonomous:
        signal = replace(signal, emotion_state=event_emotion)
    continuing = bool(
        autonomous and (working.current_topic or working.current_task)
        or signal.current_thought and (working.current_topic or working.current_task)
        or _continues_working_topic(user_text, signal, working)
    )
    topic = working.current_topic if continuing and working.current_topic else signal.topic
    task = working.current_task if continuing and working.current_task else signal.task
    if semantic_event.confirmed_completion and completion_meaningful:
        task = semantic_event.subject or task
    if signal.task_failure and not task:
        task = topic
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
    repeated_count = (
        working.repeated_topic_count
        if autonomous
        else min(20, working.repeated_topic_count + 1) if same_topic else 1
    )
    unresolved = working.unresolved_problem if same_topic or continuing else False
    outcome = working.last_outcome if same_topic or continuing else ""
    events = list(working.recent_events)
    if signal.task_failure:
        unresolved = True
        outcome = "technical_failure"
        events.append(
            InteractionEvent(
                "technical_failure",
                compact_text(task or topic, 160),
                current,
            )
        )
    elif signal.task_success and working.unresolved_problem:
        unresolved = False
        outcome = "technical_success"
        events = [
            replace(event, resolved=True)
            if not event.resolved and event.kind in {"technical_failure", "unfinished_task"}
            else event
            for event in events
        ]
        events.append(
            InteractionEvent(
                "technical_success",
                compact_text(task or topic, 160),
                current,
                True,
            )
        )
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
        last_user_summary=(
            working.last_user_summary if autonomous else compact_text(signal.summary, 180)
        ),
        recent_events=tuple(events[-16:]),
    )
    if (signal.task_failure or signal.task_success) and not needs_memory_mutation:
        memories = copy.deepcopy(list(previous.memories))
    task_tag = "task:" + _slot_value(task) if task else ""
    explicit_completion = semantic_event.confirmed_completion
    if explicit_completion or (
        signal.task_success and task and working.unresolved_problem
    ):
        for memory in memories:
            if memory.status != _ACTIVE_MEMORY:
                continue
            if explicit_completion:
                matches_completion = bool(
                    memory.id in completion_matches
                    and _memory_kind(memory) in {"working", "open_thread"}
                )
            else:
                matches_completion = bool(
                    task_tag in memory.tags
                    or memory.category == "unfinished_topic"
                    and topic_overlap(memory.content, task) >= 0.45
                )
            if not matches_completion:
                continue
            memory.status = "resolved"
            if _memory_kind(memory) == "open_thread":
                memory.thread_status = "resolved"
            memory.updated_at = current
            memory_decisions.append("resolved")
    if signal.task_failure and task and not _has_active_tag(memories, task_tag):
        _insert_into_memories(
            memories,
            _MemoryCandidate(
                content=f"An unfinished task remains: {compact_text(task, 150)}.",
                category="unfinished_topic",
                importance=0.72,
                confidence=0.86,
                tags=(task_tag, "unfinished"),
                kind="open_thread",
                source_type="deterministic_analysis",
                source_reference=source_reference,
                canonical_key=task_tag,
                scope=f"conversation:{compact_text(activity_scope, 100).lower()}",
                evidence_refs=(source_reference,),
                thread_status=_ACTIVE_MEMORY,
            ),
            source="chat:task_state",
            created_at=current,
            trace=memory_decisions,
        )
    elif signal.task_success and task:
        persist_outcome = bool(
            explicit_completion and completion_meaningful
            or not explicit_completion and working.unresolved_problem
        )
        if persist_outcome:
            outcome_subject = semantic_event.subject if explicit_completion else task
            outcome_source = "explicit_user" if explicit_completion else "deterministic_analysis"
            completion_prefix = {
                "Arcane": "Arcane completed",
                "shared": "The shared task was completed",
            }.get(semantic_event.actor, "Completion reported")
            _insert_into_memories(
                memories,
                _MemoryCandidate(
                    content=(
                        f"{completion_prefix}: {compact_text(outcome_subject, 160)}."
                        if explicit_completion
                        else f"The task was resolved: {compact_text(task, 160)}."
                    ),
                    category="task_outcome",
                    importance=0.74 if completion_resolves_thread else 0.68,
                    confidence=(
                        semantic_event.confidence if explicit_completion else 0.82
                    ),
                    tags=(task_tag, "resolved"),
                    kind="episode",
                    source_type=outcome_source,
                    source_reference=source_reference,
                    canonical_key=f"outcome:{_slot_value(outcome_subject)}",
                    scope=f"conversation:{compact_text(activity_scope, 100).lower()}",
                    evidence_refs=(source_reference,),
                ),
                source=outcome_source,
                created_at=current,
                trace=memory_decisions,
            )

    _prune_memories(memories)
    next_state = InternalState(
        emotion=signal.emotion_state,
        presence=presence,
        memories=tuple(memories),
        interests=previous.interests,
        preferences=previous.preferences,
        relationship=previous.relationship,
        updated_at=current,
        version=4,
    )
    memory_uses = _memory_use_decisions(
        recalled,
        signal,
        next_working,
        activity=latest_activity,
        familiar_relationship=familiar_relationship,
    )
    used_memory_ids = {memory_id for memory_id, use in memory_uses if use != "none"}
    used_memories = tuple(memory for memory in recalled if memory.id in used_memory_ids)
    return InternalTurnResult(
        state=next_state,
        signal=signal,
        recalled_memories=used_memories,
        affect_trace=build_affect_trace(
            previous.emotion,
            event_emotion,
            signal,
            evolution=mood_evolution,
            event_delta=tuple(event_effects.items()),
        ),
        working_context=next_working,
        grounded_activity_source="presence" if latest_activity else "none",
        grounded_activity_age_seconds=(
            max(0.0, current - latest_activity.started_at) if latest_activity else 0.0
        ),
        memory_trace=_memory_trace(
            used_memories,
            memory_decisions,
            current,
            memory_uses,
            considered_memories=retrieval_memories,
            scope=activity_scope,
        ),
        memory_uses=memory_uses,
    )


class LongTermMemoryStore:
    """Own durable profile state, including the authoritative presence lifecycle."""

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
        self._presence_wake: Callable[[str], None] | None = None
        self._load()

    def internal_state(self, profile_id: str = "local:owner") -> InternalState:
        key = canonical_profile_id(profile_id)
        with self._lock:
            state = self._states.get(key)
            return copy.deepcopy(state) if state is not None else new_internal_state(0.0)

    def stored_internal_state(self, profile_id: str = "local:owner") -> InternalState | None:
        """Return the persisted profile record, without manufacturing a default."""

        with self._lock:
            return copy.deepcopy(self._states.get(canonical_profile_id(profile_id)))

    def set_presence_wake(
        self,
        callback: Callable[[str], None] | None,
    ) -> None:
        with self._lock:
            self._presence_wake = callback

    def prepare_presence(
        self,
        profile_id: str,
        *,
        now: float,
    ) -> InternalState:
        """Load, validate, and flag due presence without running inference."""

        key = canonical_profile_id(profile_id)
        current = max(0.0, float(now))
        with self._lock:
            previous = self._states.get(key)
            state = previous or new_internal_state(current)
            presence = normalize_presence(
                state.presence,
                now=current,
                initialize_schedule=True,
            )
            refreshed = (
                replace(
                    state,
                    presence=presence,
                    updated_at=max(state.updated_at, current),
                )
                if presence != state.presence
                else state
            )
            if refreshed != previous:
                self._save_state(key, refreshed, previous)
            return copy.deepcopy(refreshed)

    def preview_turn(
        self,
        profile_id: str,
        user_text: str,
        *,
        now: float | None = None,
        include_memory: bool = True,
        code_context_requested: bool = False,
        code_context_attached: bool = False,
        autonomous: bool = False,
        familiar_relationship: bool = False,
        working_context: WorkingMemory | None = None,
        recent_turns: tuple[ChatTurn, ...] = (),
        activity_scope: str = "profile",
    ) -> InternalTurnResult:
        key = canonical_profile_id(profile_id)
        current = time.time() if now is None else max(0.0, float(now))
        state = self.prepare_presence(key, now=current)
        result = process_internal_turn(
            user_text,
            state,
            now=now,
            retrieval=self._retrieval,
            include_memory=include_memory,
            code_context_requested=code_context_requested,
            code_context_attached=code_context_attached,
            autonomous=autonomous,
            familiar_relationship=familiar_relationship,
            working_context=working_context,
            recent_turns=recent_turns,
            activity_scope=activity_scope,
            profile_seed=profile_id,
        )
        return result

    def commit_turn(
        self,
        profile_id: str,
        result: InternalTurnResult,
        *,
        used_memory_ids: tuple[str, ...] = (),
        preference_updates: tuple[dict[str, object], ...] = (),
        interest_additions: tuple[str, ...] = (),
        relationship_updates: tuple[dict[str, object], ...] = (),
        emotion_update: dict[str, object] | None = None,
        now: float | None = None,
    ) -> InternalState | None:
        key = canonical_profile_id(profile_id)
        current = result.state.updated_at if now is None else max(
            result.state.updated_at,
            0.0,
            float(now),
        )
        with self._lock:
            previous = self._states.get(key)
            latest = previous or new_internal_state(current)
            wanted = set(used_memory_ids)
            memories = list(result.state.memories)
            if wanted:
                memories = [
                    copy.copy(memory)
                    if memory.id in wanted and memory.status == _ACTIVE_MEMORY
                    else memory
                    for memory in memories
                ]
                for memory in memories:
                    if memory.id in wanted and memory.status == _ACTIVE_MEMORY:
                        memory.last_used_at = current
                        memory.access_count += 1
            next_state = replace(
                result.state,
                emotion=_updated_emotion(
                    result.state.emotion,
                    emotion_update,
                    now=current,
                ),
                presence=latest.presence,
                memories=tuple(memories),
                interests=_merge_interests(
                    latest.interests,
                    result.state.interests,
                ),
                preferences=result.state.preferences,
                relationship=result.state.relationship,
                updated_at=current,
            )
            next_state = replace(
                next_state,
                interests=_merge_interests(
                    next_state.interests,
                    interest_additions,
                ),
                preferences=_merge_akane_preferences(
                    next_state.preferences,
                    preference_updates,
                    updated_at=current,
                ),
                relationship=_merge_relationship(
                    result.state.relationship,
                    relationship_updates,
                    updated_at=current,
                ),
                updated_at=current,
            )
            self._save_state(key, next_state, previous)
            return previous

    def presence_schedule(
        self,
        *,
        now: float,
    ) -> tuple[tuple[str, ...], float | None]:
        """Return due profiles and the next persisted wake timestamp."""

        current = max(0.0, float(now))
        with self._lock:
            previous_states = self._states
            next_states = previous_states.copy()
            changed = False
            due: list[str] = []
            wake_at: list[float] = []
            for key, state in previous_states.items():
                presence = normalize_presence(
                    state.presence,
                    now=current,
                    initialize_schedule=True,
                )
                if presence != state.presence:
                    next_states[key] = replace(
                        state,
                        presence=presence,
                        updated_at=max(state.updated_at, current),
                    )
                    changed = True
                if presence.claim_token is not None:
                    wake_at.append(presence.claim_expires_at)
                elif presence.retry_at > current:
                    wake_at.append(presence.retry_at)
                elif (
                    presence.decision_pending
                    or presence.next_decision_at <= current
                ):
                    due.append(key)
                else:
                    wake_at.append(presence.next_decision_at)
            if changed:
                self._states = next_states
                try:
                    self._persist()
                except Exception:
                    self._states = previous_states
                    raise
            return tuple(due), min(wake_at, default=None)

    def claim_presence_decision(
        self,
        profile_id: str,
        *,
        now: float,
    ) -> InternalState | None:
        """Atomically claim one due autonomous-life decision."""

        key = canonical_profile_id(profile_id)
        current = max(0.0, float(now))
        with self._lock:
            previous = self._states.get(key)
            if previous is None:
                return None
            presence = normalize_presence(
                previous.presence,
                now=current,
                initialize_schedule=True,
            )
            if presence.claim_token is not None:
                return None
            if presence.retry_at > current:
                return None
            if (
                not presence.decision_pending
                and presence.next_decision_at > current
            ):
                return None
            token = uuid.uuid4().hex
            next_state = replace(
                previous,
                presence=replace(
                    presence,
                    decision_pending=True,
                    claim_token=token,
                    claim_expires_at=current + CLAIM_SECONDS,
                ),
                updated_at=max(previous.updated_at, current),
            )
            self._save_state(key, next_state, previous)
            return copy.deepcopy(next_state)

    def fail_presence_decision(
        self,
        profile_id: str,
        *,
        claim_token: str,
        now: float,
        error: str,
        retryable: bool = True,
    ) -> bool:
        key = canonical_profile_id(profile_id)
        current = max(0.0, float(now))
        with self._lock:
            previous = self._states.get(key)
            if (
                previous is None
                or previous.presence.claim_token != claim_token
            ):
                return False
            second_attempt = previous.presence.retry_at > 0.0
            retry_at = (
                current + RETRY_SECONDS
                if retryable and not second_attempt
                else 0.0
            )
            next_at = (
                previous.presence.next_decision_at
                if retry_at
                else next_decision_time(current)
            )
            current_activity = previous.presence.current_activity
            if current_activity is not None and not retry_at:
                current_activity = replace(
                    current_activity,
                    expected_end_at=next_at,
                )
            self._save_state(
                key,
                replace(
                    previous,
                    presence=replace(
                        previous.presence,
                        current_activity=current_activity,
                        decision_pending=bool(retry_at),
                        claim_token=None,
                        claim_expires_at=0.0,
                        retry_at=retry_at,
                        next_decision_at=next_at,
                        last_error=compact_text(error, 120) or "life decision failed",
                    ),
                    updated_at=max(previous.updated_at, current),
                ),
                previous,
            )
            return True

    def commit_presence_decision(
        self,
        profile_id: str,
        decision: LifeDecision,
        *,
        claim_token: str,
        now: float,
        grounded_context: str,
    ) -> tuple[bool, str]:
        key = canonical_profile_id(profile_id)
        current = max(0.0, float(now))
        with self._lock:
            previous = self._states.get(key)
            if (
                previous is None
                or previous.presence.claim_token != claim_token
            ):
                return False, "life claim is unavailable"
            rejection = life_decision_rejection(previous.presence, decision)
            if rejection:
                self.fail_presence_decision(
                    profile_id,
                    claim_token=claim_token,
                    now=current,
                    error=rejection,
                )
                return False, rejection
            interest = (
                validate_interest_addition(
                    decision.interest_addition,
                    activity=decision.activity,
                    subject=decision.subject,
                    detail=decision.detail,
                    existing_interests=previous.interests,
                    grounded_context=grounded_context,
                )
                if decision.mode == "new"
                else None
            )
            next_state = replace(
                previous,
                presence=apply_life_decision(
                    previous.presence,
                    decision,
                    now=current,
                ),
                interests=_merge_interests(
                    previous.interests,
                    (interest,) if interest else (),
                ),
                updated_at=max(previous.updated_at, current),
            )
            self._save_state(key, next_state, previous)
            return True, ""

    def retrieve(
        self,
        profile_id: str,
        query: str,
        *,
        now: float | None = None,
        scope: str = "profile",
    ) -> tuple[Memory, ...]:
        current = time.time() if now is None else float(now)
        with self._lock:
            state = self._states.get(canonical_profile_id(profile_id))
            if state is None:
                return ()
            return _retrieve_memories(
                state.memories,
                query,
                current,
                self._retrieval,
                WorkingMemory(),
                scope=scope,
            )

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
        kind: str = "",
        source_type: str = "",
        source_reference: str = "",
        canonical_key: str = "",
        scope: str = "profile",
        expires_at: float | None = None,
        evidence_refs: tuple[str, ...] = (),
        thread_status: str = "",
    ) -> Memory | None:
        normalized_category = compact_text(category, 24).lower()
        normalized_tags = tuple(tags)
        normalized_kind = _memory_kind_from_fields(
            compact_text(kind, 24).lower(),
            normalized_category,
            tuple(compact_text(tag, 48).lower() for tag in normalized_tags),
        )
        normalized_source_type = _source_type(source_type or source)
        candidate = _MemoryCandidate(
            content=compact_text(content, 240),
            category=normalized_category,
            importance=max(0.0, min(1.0, float(importance))),
            confidence=max(0.0, min(1.0, float(confidence))),
            tags=normalized_tags,
            supersedes=True,
            kind=normalized_kind,
            source_type=normalized_source_type,
            source_reference=compact_text(source_reference, 100),
            canonical_key=compact_text(canonical_key, 120).lower(),
            scope=compact_text(scope, 120).lower() or "profile",
            expires_at=(max(0.0, float(expires_at)) if expires_at is not None else None),
            evidence_refs=tuple(
                value
                for item in evidence_refs
                if (value := compact_text(item, 100))
            )[:12],
            thread_status=compact_text(thread_status, 24).lower(),
        )
        if not candidate.content or candidate.category not in _MEMORY_CATEGORIES:
            return None
        key = canonical_profile_id(profile_id)
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
            if memory is None:
                return None
            if not changed:
                return copy.deepcopy(memory)
            next_state = replace(
                state,
                memories=tuple(memories),
                updated_at=time.time() if created_at is None else float(created_at),
            )
            self._save_state(key, next_state, previous)
            return copy.deepcopy(memory)

    def restore_internal_state(self, profile_id: str, state: InternalState | None) -> None:
        key = canonical_profile_id(profile_id)
        with self._lock:
            current = self._states.get(key)
            if current is not None:
                state = replace(
                    copy.deepcopy(state)
                    if state is not None
                    else new_internal_state(current.updated_at),
                    presence=current.presence,
                )
            if state is None:
                self._states.pop(key, None)
            else:
                self._states[key] = state
            self._persist()

    def clear(self, profile_id: str = "local:owner") -> None:
        with self._lock:
            key = canonical_profile_id(profile_id)
            if key not in self._states:
                return
            self._states.pop(key)
            self._persist()
            notifier = self._presence_wake
            if notifier is not None:
                notifier(key)

    def public_profile(self, profile_id: str = "local:owner") -> dict[str, object]:
        now = time.time()
        with self._lock:
            state = self._states.get(canonical_profile_id(profile_id))
            active = [
                copy.deepcopy(memory)
                for memory in (state.memories if state else ())
                if memory.is_available(now)
            ]
        user: dict[str, str] = {}
        preferences: list[dict[str, object]] = []
        facts: list[dict[str, object]] = []
        episodes: list[dict[str, object]] = []
        activities: dict[str, object] = {}
        for memory in active:
            item = {"content": memory.content, "category": memory.category}
            if "slot:name" in memory.tags:
                user["name"] = _memory_name(memory.content)
            elif _AKANE_PREFERENCE_TAG in memory.tags:
                continue
            elif _memory_kind(memory) == "working":
                activities["arcane_current"] = {
                    **item,
                    "expires_at": memory.expires_at or 0.0,
                    "source": memory.source_type,
                }
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
            "activities": activities,
        }

    def public_internal_state(self, profile_id: str = "local:owner") -> dict[str, object]:
        state = self.prepare_presence(profile_id, now=time.time())
        return self.public_state_snapshot(state)

    @staticmethod
    def public_state_snapshot(state: InternalState) -> dict[str, object]:
        """Render one stored or read-only candidate state for diagnostics."""

        emotion = state.emotion
        presence = state.presence
        available_memories = [
            memory for memory in state.memories if memory.is_available(state.updated_at)
        ]
        arcane_activity = next(
            (
                memory
                for memory in sorted(
                    available_memories,
                    key=lambda item: item.updated_at or item.created_at,
                    reverse=True,
                )
                if _memory_kind(memory) == "working"
                and memory.canonical_key == _ARCANE_ACTIVITY_KEY
            ),
            None,
        )
        return {
            "emotion": {
                "primary": emotion.primary,
                "intensity": round(emotion.intensity, 3),
                "cause": emotion.cause,
                "updated_at": emotion.updated_at,
            },
            "presence": presence.as_dict(),
            "interests": list(state.interests),
            "preferences": [asdict(preference) for preference in state.preferences],
            "relationship": _relationship_state_to_dict(state.relationship),
            "working": {
                "arcane_current_activity": (
                    {
                        "content": arcane_activity.content,
                        "status": arcane_activity.status,
                        "source": arcane_activity.source_type,
                        "canonical_key": arcane_activity.canonical_key,
                        "expires_at": arcane_activity.expires_at or 0.0,
                    }
                    if arcane_activity is not None
                    else {}
                )
            },
            "memory_core": {
                "active_corrections": sum(
                    _memory_kind(memory) == "correction"
                    for memory in available_memories
                ),
                "active_threads": sum(
                    _memory_kind(memory) == "open_thread"
                    for memory in available_memories
                ),
                "active_working": sum(
                    _memory_kind(memory) == "working"
                    for memory in available_memories
                ),
                "schema_version": LONG_TERM_MEMORY_SCHEMA_VERSION,
            },
            "state_schema_version": state.version,
            "updated_at": state.updated_at,
        }

    def _save_state(
        self,
        key: str,
        state: InternalState,
        previous: InternalState | None,
    ) -> None:
        if state == previous:
            return
        self._states[key] = state
        try:
            self._persist()
        except Exception:
            if previous is None:
                self._states.pop(key, None)
            else:
                self._states[key] = previous
            raise
        notifier = self._presence_wake
        previous_presence = previous.presence if previous is not None else None
        schedule_changed = previous_presence is None or any(
            getattr(state.presence, field) != getattr(previous_presence, field)
            for field in (
                "next_decision_at",
                "decision_pending",
                "retry_at",
                "claim_expires_at",
            )
        )
        if notifier is not None and schedule_changed:
            notifier(key)

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
                migrated = _normalize_loaded_memories(
                    _migrate_legacy_profile(payload["user"])
                )
                if migrated:
                    current = max(memory.created_at for memory in migrated)
                    self._states = {
                        OWNER_PROFILE_ID: replace(
                            new_internal_state(current),
                            memories=tuple(migrated),
                        )
                    }
                self._initialize_loaded_presence()
                return
            if schema not in {
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                LONG_TERM_MEMORY_SCHEMA_VERSION,
            }:
                raise ValueError("unsupported schema")
            profiles = payload.get("profiles")
            if not isinstance(profiles, dict):
                raise ValueError("invalid profiles")
            self._states = {}
            migrate_presence = False
            load_time = time.time()
            for key, raw_profile in profiles.items():
                raw_profile_id = _key(key, "")
                if not raw_profile_id:
                    continue
                profile = canonical_profile_id(raw_profile_id)
                migrate_presence = migrate_presence or profile != raw_profile_id
                if schema == 2 and isinstance(raw_profile, list):
                    loaded = _normalize_loaded_memories(
                        [
                            memory
                            for item in raw_profile
                            if (memory := Memory.from_dict(item)) is not None
                        ]
                    )
                    if loaded:
                        current = max(memory.created_at for memory in loaded)
                        candidate = replace(
                            new_internal_state(current),
                            memories=loaded,
                        )
                        existing = self._states.get(profile)
                        if (
                            existing is None
                            or _state_completeness(candidate)
                            > _state_completeness(existing)
                        ):
                            self._states[profile] = candidate
                elif schema in {
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    LONG_TERM_MEMORY_SCHEMA_VERSION,
                }:
                    raw_presence = (
                        raw_profile.get("presence")
                        if isinstance(raw_profile, dict)
                        else None
                    )
                    canonical_presence_keys = {
                        "current_activity",
                        "previous_activity",
                        "last_decision_at",
                        "next_decision_at",
                        "decision_pending",
                        "claim_token",
                        "claim_expires_at",
                        "retry_at",
                        "last_error",
                        "activity_pattern",
                    }
                    migrate_presence = migrate_presence or (
                        isinstance(raw_presence, dict)
                        and set(raw_presence) != canonical_presence_keys
                    )
                    state = _internal_state_from_dict(
                        raw_profile,
                        now=load_time,
                    )
                    if state is not None:
                        migrate_presence = (
                            migrate_presence
                            or not isinstance(raw_presence, dict)
                            or state.presence.as_dict() != raw_presence
                        )
                        existing = self._states.get(profile)
                        if (
                            existing is None
                            or _state_completeness(state)
                            > _state_completeness(existing)
                        ):
                            self._states[profile] = state
            self._initialize_loaded_presence(
                force_persist=(
                    migrate_presence or schema != LONG_TERM_MEMORY_SCHEMA_VERSION
                )
            )
        except FileNotFoundError:
            return
        except (OSError, TypeError, ValueError) as exc:
            print(
                f"[Akane:long-term-memory] ignored corrupt memory ({type(exc).__name__})",
                flush=True,
            )
            self._states = {}

    def _initialize_loaded_presence(self, *, force_persist: bool = False) -> None:
        """Validate and migrate presence once during startup."""

        current = time.time()
        changed = False
        for key, state in tuple(self._states.items()):
            presence = normalize_presence(
                state.presence,
                now=current,
                initialize_schedule=True,
            )
            if presence != state.presence:
                self._states[key] = replace(
                    state,
                    presence=presence,
                    updated_at=max(state.updated_at, current),
                )
                changed = True
        if changed or force_persist:
            self._persist()

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


def format_relevant_memories(
    memories: tuple[Memory, ...],
    memory_uses: tuple[tuple[str, str], ...] = (),
) -> str:
    if not memories:
        return ""
    used_ids = {memory_id for memory_id, use in memory_uses if use != "none"}
    selected = (
        tuple(memory for memory in memories if memory.id in used_ids)
        if memory_uses
        else memories
    )
    if not selected:
        return ""
    lines = [memory.content for memory in selected]
    return (
        _MEMORY_PROMPT_INTRO
        + "\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n"
        + _MEMORY_PROMPT_OUTRO
    )


def _memory_use_decisions(
    memories: tuple[Memory, ...],
    signal: TurnSignal,
    working: WorkingMemory,
    *,
    activity: PresenceActivity | None = None,
    familiar_relationship: bool = False,
) -> tuple[tuple[str, str], ...]:
    """Assign one compact use to each relevant record without rescoring it."""

    decisions: list[tuple[str, str]] = []
    direct_task = signal.intent in {"technical", "instruction"} or signal.technical
    for memory in memories:
        kind = _memory_kind(memory)
        source = _memory_source_type(memory)
        if source in {
            "conversation_summary",
            "generated_assistant",
            "speculative_inference",
            "unknown",
        }:
            continue
        tags = set(memory.tags)
        emotional = bool(
            tags.intersection(
                {
                    "emotional",
                    "distress",
                    "user-distress",
                    "conflict",
                    "criticism",
                    "praise",
                }
            )
            or any(tag.startswith("emotion:") for tag in tags)
        )
        if kind == "correction":
            use = "correction"
        elif kind == "open_thread":
            use = "thread"
        elif kind == "self":
            use = "self_experience"
        elif kind == "relationship":
            use = "relationship_context"
        elif kind == "episode" and emotional:
            use = "emotional_context"
        elif kind == "episode" and familiar_relationship and not direct_task:
            use = "callback"
        elif kind in {"working", "episode", "profile"}:
            use = "fact"
        else:
            continue
        if use == "self_experience" and activity is not None:
            if topic_overlap(memory.content, activity.fact()) < 0.20 and not signal.current_activity:
                use = "fact"
        decisions.append((memory.id, use))
    return tuple(decisions)


def preference_domain(text: str) -> str:
    """Return the concrete preference area named by a preference question."""

    value = compact_text(text, 300).lower()
    for domain, pattern in (
        ("anime", r"\b(?:anime|manga)\b"),
        ("games", r"\b(?:game|games|gaming)\b"),
        ("music", r"\b(?:music|song|songs|band|artist)\b"),
        ("books", r"\b(?:book|books|novel|novels|reading)\b"),
        ("food", r"\b(?:food|meal|snack|dish|cuisine)\b"),
        ("colors", r"\b(?:color|colors|colour|colours)\b"),
    ):
        if re.search(pattern, value):
            return domain
    return "general"


def preference_update_requested(text: str) -> bool:
    return bool(_AKANE_PREFERENCE_UPDATE.search(str(text or "")))


def established_akane_preference(
    memories: tuple[Memory, ...],
    query: str,
    *,
    now: float | None = None,
) -> Memory | None:
    """Find Akane's latest active preference for the question's domain."""

    current = time.time() if now is None else max(0.0, float(now))
    domain = preference_domain(query)
    candidates = [
        memory
        for memory in memories
        if memory.is_available(current) and _AKANE_PREFERENCE_TAG in memory.tags
    ]
    if domain != "general":
        tag = f"{_AKANE_PREFERENCE_TAG}:{domain}"
        candidates = [memory for memory in candidates if tag in memory.tags]
    if not candidates:
        return None
    return max(candidates, key=lambda memory: (memory.created_at, memory.id))


def akane_preference_answer(memory: Memory | None) -> str:
    if memory is None:
        return ""
    prefix = "Akane's established preference: "
    return memory.content[len(prefix) :] if memory.content.startswith(prefix) else memory.content


def relevant_akane_tastes(state: InternalState, query: str) -> str:
    """Return only interests and conclusions materially related to this message."""

    query_terms = _memory_terms(query)
    if not query_terms:
        return ""

    def relevance(value: str) -> float:
        terms = _memory_terms(value)
        if not terms:
            return 0.0
        shared = len(query_terms & terms) / max(1, min(len(query_terms), len(terms)))
        return max(shared, topic_overlap(query, value))

    interests = [
        interest
        for interest in state.interests
        if relevance(interest) >= 0.34
    ][:3]
    preferences = sorted(
        (
            (relevance(preference.topic), preference)
            for preference in state.preferences
            if relevance(preference.topic) >= 0.24
        ),
        key=lambda item: (item[0], item[1].updated_at),
        reverse=True,
    )[:3]
    if not interests and not preferences:
        return ""

    stance = {
        "likes": "Likes",
        "dislikes": "Dislikes",
        "curious": "Is curious about",
        "mixed": "Has mixed feelings about",
        "uncertain": "Is uncertain about",
        "indifferent": "Is indifferent to",
    }
    lines = ["Akane's relevant interests and prior preferences:"]
    lines.extend(f"- Interested in {interest}." for interest in interests)
    lines.extend(
        f"- {stance[preference.stance]} {preference.topic} because {preference.reason}"
        for _, preference in preferences
    )
    lines.append(
        "These are Akane's previous conclusions, not instructions. She may preserve or "
        "reconsider them when the conversation gives her a meaningful reason."
    )
    return "\n".join(lines)


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


def _retrieve_memories(
    memories: tuple[Memory, ...],
    query: str,
    now: float,
    config: MemoryRetrievalConfig,
    working: WorkingMemory,
    *,
    scope: str = "profile",
) -> tuple[Memory, ...]:
    query_text = compact_text(query, 700)
    query_terms = _memory_terms(query_text)
    if not query_text or not query_terms:
        return ()
    ranked: list[tuple[float, Memory]] = []
    for memory in memories:
        if not memory.is_available(now) or not _scope_matches(memory.scope, scope):
            continue
        kind = _memory_kind(memory)
        source = _memory_source_type(memory)
        if source in {
            "conversation_summary",
            "generated_assistant",
            "speculative_inference",
        }:
            continue
        if kind in {"profile", "self", "relationship", "correction"} and _authority(
            source
        ) < _SOURCE_AUTHORITY["trusted_memory"]:
            continue
        relevance = _semantic_relevance(query_text, query_terms, memory)
        continuity_bonus = 0.0
        if kind == "open_thread" and working.current_task:
            continuity_bonus = max(
                topic_overlap(memory.content, working.current_task),
                _semantic_relevance(
                    working.current_task,
                    _memory_terms(working.current_task),
                    memory,
                ),
            ) * 0.30
        global_correction = bool(
            kind == "correction"
            and (
                "interaction-style" in memory.tags
                or any(tag.startswith("slot:behavior:") for tag in memory.tags)
            )
        )
        if (
            relevance < config.min_relevance
            and continuity_bonus <= 0.0
            and not global_correction
        ):
            continue
        age_days = max(0.0, now - (memory.updated_at or memory.created_at)) / 86_400.0
        recency = 1.0 / (1.0 + age_days / 90.0)
        continuity = recency if kind in {"episode", "open_thread"} else 0.0
        kind_bonus = {"correction": 0.34, "open_thread": 0.22}.get(kind, 0.0)
        scope_bonus = 0.12 if memory.scope not in {"global", "profile"} else 0.0
        authority_bonus = _authority(_memory_source_type(memory)) / 9.0 * 0.12
        score = (
            relevance * config.relevance_weight
            + memory.importance * config.importance_weight
            + memory.confidence * config.confidence_weight
            + recency * config.recency_weight
            + continuity * config.continuity_weight
            + continuity_bonus
            + kind_bonus
            + scope_bonus
            + authority_bonus
            - _recent_use(memory, now) * config.repetition_penalty
            - min(1.0, max(0.0, age_days - 365.0) / 365.0) * config.staleness_penalty
        )
        if score >= config.min_score:
            ranked.append((score, memory))
    ranked.sort(key=lambda item: (item[0], item[1].created_at, item[1].id), reverse=True)
    return tuple(copy.deepcopy(memory) for _score, memory in ranked[: config.max_results])


def _memory_trace(
    recalled: tuple[Memory, ...],
    decisions: list[str],
    now: float,
    memory_uses: tuple[tuple[str, str], ...] = (),
    *,
    considered_memories: tuple[Memory, ...] = (),
    scope: str = "profile",
) -> dict[str, object]:
    retrieved_by_kind: dict[str, int] = {}
    for memory in recalled:
        kind = _memory_kind(memory)
        retrieved_by_kind[kind] = retrieved_by_kind.get(kind, 0) + 1
    return {
        "retrieved_by_kind": retrieved_by_kind,
        "records_considered": sum(
            memory.is_available(now) and _scope_matches(memory.scope, scope)
            for memory in considered_memories
        ),
        "records_used": len(recalled),
        "memory_uses": memory_uses,
        "active_correction": next(
            (memory.id for memory in recalled if _memory_kind(memory) == "correction"),
            "",
        ),
        "active_thread": next(
            (memory.id for memory in recalled if _memory_kind(memory) == "open_thread"),
            "",
        ),
        "grounded_self_event": next(
            (memory.id for memory in recalled if _memory_kind(memory) == "self"),
            "",
        ),
        "candidate_writes": decisions.count("created"),
        "candidate_updates": decisions.count("updated"),
        "migration_version": LONG_TERM_MEMORY_SCHEMA_VERSION,
    }


def _insert_into_memories(
    memories: list[Memory],
    candidate: _MemoryCandidate,
    *,
    source: str,
    created_at: float,
    status: str = _ACTIVE_MEMORY,
    trace: list[str] | None = None,
) -> tuple[Memory | None, bool]:
    candidate_tags = _normalized_tags(candidate.tags, candidate.content)
    source_type = _source_type(candidate.source_type or source)
    canonical_key = _candidate_key(replace(candidate, tags=candidate_tags))
    scope = compact_text(candidate.scope, 120).lower() or "profile"
    candidate = replace(
        candidate,
        tags=candidate_tags,
        source_type=source_type,
        canonical_key=canonical_key,
        scope=scope,
    )
    allowed, _ = _candidate_allowed(candidate)
    if not allowed:
        if trace is not None:
            trace.append("rejected")
        return None, False

    if candidate.kind == "working":
        active = [
            existing
            for existing in memories
            if existing.status == _ACTIVE_MEMORY
            and _memory_kind(existing) == "working"
            and existing.scope == scope
            and existing.canonical_key == canonical_key
        ]
        if active:
            existing = max(active, key=lambda item: item.updated_at or item.created_at)
            existing.content = candidate.content
            existing.category = candidate.category
            existing.importance = candidate.importance
            existing.confidence = candidate.confidence
            existing.source = compact_text(source, 48) or existing.source
            existing.tags = candidate_tags
            existing.status = _ACTIVE_MEMORY
            existing.expires_at = candidate.expires_at
            existing.source_type = source_type
            existing.source_reference = candidate.source_reference
            existing.updated_at = max(existing.updated_at, created_at)
            existing.evidence_refs = tuple(
                dict.fromkeys((*existing.evidence_refs, *candidate.evidence_refs))
            )[:12]
            for duplicate in active:
                if duplicate.id == existing.id:
                    continue
                duplicate.status = "superseded"
                duplicate.superseded_by = existing.id
                duplicate.updated_at = max(duplicate.updated_at, created_at)
            if trace is not None:
                trace.append("updated")
            return existing, True

    for existing in memories:
        if (
            existing.status != _ACTIVE_MEMORY
            or _memory_kind(existing) != candidate.kind
            or existing.scope != scope
            or (
                candidate.kind in {"profile", "relationship", "self", "correction"}
                and existing.canonical_key
                and existing.canonical_key != canonical_key
            )
        ):
            continue
        if _memory_content_matches(existing, candidate.content, candidate_tags):
            changed = False
            new_evidence = tuple(
                ref for ref in candidate.evidence_refs if ref not in existing.evidence_refs
            )
            if candidate.importance > existing.importance:
                existing.importance = candidate.importance
                changed = True
            if (
                new_evidence
                or _authority(source_type) > _authority(_memory_source_type(existing))
            ):
                reinforced = max(existing.confidence, candidate.confidence)
                reinforced = min(0.98, reinforced + (1.0 - reinforced) * 0.20)
                if reinforced > existing.confidence:
                    existing.confidence = reinforced
                    changed = True
                if _authority(source_type) > _authority(_memory_source_type(existing)):
                    existing.source = compact_text(source, 48) or existing.source
                    existing.source_type = source_type
                    existing.source_reference = candidate.source_reference
                    changed = True
            evidence = tuple(
                dict.fromkeys((*existing.evidence_refs, *candidate.evidence_refs))
            )[:12]
            if evidence != existing.evidence_refs:
                existing.evidence_refs = evidence
                changed = True
            if candidate.thread_status and candidate.thread_status != existing.thread_status:
                existing.thread_status = candidate.thread_status
                changed = True
            if changed:
                existing.updated_at = max(existing.updated_at, created_at)
            if trace is not None:
                trace.append("updated" if changed else "duplicate")
            return existing, changed

    conflicts = [
        existing
        for existing in memories
        if existing.status == _ACTIVE_MEMORY
        and existing.scope == scope
        and (existing.canonical_key or _canonical_key(
            _memory_kind(existing), existing.category, existing.tags, existing.content
        )) == canonical_key
        and not _memory_content_matches(existing, candidate.content, candidate_tags)
    ]
    memory_status = compact_text(status, 24).lower() or _ACTIVE_MEMORY
    if memory_status == "contradicted":
        memory_status = "disputed"
    if memory_status not in _MEMORY_STATUSES:
        memory_status = "archived"
    superseded: list[Memory] = []
    if conflicts and memory_status == _ACTIVE_MEMORY:
        strongest = max(
            conflicts,
            key=lambda item: (
                _authority(_memory_source_type(item)),
                item.updated_at or item.created_at,
            ),
        )
        new_authority = _authority(source_type)
        old_authority = _authority(_memory_source_type(strongest))
        if new_authority < old_authority:
            if trace is not None:
                trace.append("rejected")
            return None, False
        if new_authority > old_authority or (
            candidate.supersedes and new_authority >= old_authority
        ):
            superseded = [
                item
                for item in conflicts
                if _authority(_memory_source_type(item)) <= new_authority
            ]
        else:
            memory_status = "disputed"
            for existing in conflicts:
                existing.status = "disputed"
                existing.updated_at = max(existing.updated_at, created_at)
    memory = Memory(
        id=uuid.uuid4().hex,
        content=candidate.content,
        category=candidate.category,
        created_at=max(0.0, created_at),
        importance=candidate.importance,
        confidence=candidate.confidence,
        source=compact_text(source, 48) or "user",
        tags=candidate_tags,
        status=memory_status,
        expires_at=candidate.expires_at,
        kind=candidate.kind,
        source_type=source_type,
        source_reference=candidate.source_reference,
        canonical_key=canonical_key,
        scope=scope,
        updated_at=max(0.0, created_at),
        evidence_refs=candidate.evidence_refs,
        thread_status=(
            candidate.thread_status
            or (memory_status if memory_status in _THREAD_STATUSES else _ACTIVE_MEMORY)
            if candidate.kind == "open_thread"
            else ""
        ),
    )
    if superseded and memory.status == _ACTIVE_MEMORY:
        for existing in superseded:
            existing.status = "superseded"
            existing.superseded_by = memory.id
            existing.updated_at = max(existing.updated_at, created_at)
    memories.append(memory)
    _prune_memories(memories)
    if trace is not None:
        trace.append("created" if memory.status == _ACTIVE_MEMORY else memory.status)
    return memory, True


def _prune_memories(memories: list[Memory]) -> None:
    if len(memories) <= MEMORY_MAX_ENTRIES_PER_PROFILE:
        return
    memories.sort(
        key=lambda item: (item.status == _ACTIVE_MEMORY, item.importance, item.created_at)
    )
    del memories[: len(memories) - MEMORY_MAX_ENTRIES_PER_PROFILE]


def _normalize_loaded_memories(memories: list[Memory]) -> tuple[Memory, ...]:
    """Consolidate persisted duplicates without changing the storage contract."""

    consolidated: list[Memory] = []
    for memory in memories:
        duplicate = next(
            (
                existing
                for existing in consolidated
                if _memory_kind(existing) == _memory_kind(memory)
                and existing.status == memory.status
                and existing.scope == memory.scope
                and existing.canonical_key == memory.canonical_key
                and _memory_content_matches(existing, memory.content, memory.tags)
            ),
            None,
        )
        if duplicate is None:
            consolidated.append(copy.deepcopy(memory))
            continue
        duplicate.created_at = min(duplicate.created_at, memory.created_at)
        duplicate.importance = max(duplicate.importance, memory.importance)
        duplicate.confidence = max(duplicate.confidence, memory.confidence)
        duplicate.tags = tuple(dict.fromkeys((*duplicate.tags, *memory.tags)))[:12]
        duplicate.evidence_refs = tuple(
            dict.fromkeys((*duplicate.evidence_refs, *memory.evidence_refs))
        )[:12]
        duplicate.updated_at = max(duplicate.updated_at, memory.updated_at)
        if memory.last_used_at is not None:
            duplicate.last_used_at = max(
                duplicate.last_used_at or 0.0,
                memory.last_used_at,
            )
        duplicate.access_count += memory.access_count
    _prune_memories(consolidated)
    return tuple(consolidated)


def _memory_content_matches(
    existing: Memory,
    content: str,
    tags: tuple[str, ...],
) -> bool:
    if normalized_signature(existing.content) == normalized_signature(content):
        return True
    slots = {tag for tag in tags if tag.startswith("slot:")}
    if slots.intersection(existing.tags):
        return False
    similarity = message_similarity(existing.content, content)
    existing_terms = _memory_terms(existing.content)
    candidate_terms = _memory_terms(content)
    shared_terms = existing_terms & candidate_terms
    term_overlap = len(shared_terms) / max(
        1,
        min(len(existing_terms), len(candidate_terms)),
    )
    term_jaccard = len(shared_terms) / max(1, len(existing_terms | candidate_terms))
    return similarity >= 0.94 or (
        similarity >= 0.72 and term_overlap >= 0.80 and term_jaccard >= 0.75
    )


def _has_active_tag(memories: list[Memory], tag: str) -> bool:
    return bool(tag) and any(
        memory.status == _ACTIVE_MEMORY and tag in memory.tags for memory in memories
    )


def _internal_state_to_dict(state: InternalState) -> dict[str, object]:
    return {
        "version": state.version,
        "updated_at": state.updated_at,
        "emotion": asdict(state.emotion),
        "presence": state.presence.as_dict(),
        "memories": [asdict(memory) for memory in state.memories],
        "interests": list(state.interests),
        "preferences": [asdict(preference) for preference in state.preferences],
        "relationship": _relationship_state_to_dict(state.relationship),
    }


def _state_completeness(state: InternalState) -> tuple[int, int, float]:
    presence = state.presence
    relationship_count = sum(
        len(values)
        for values in (
            state.relationship.patterns,
            state.relationship.shared_context,
            state.relationship.unresolved_events,
        )
    )
    domain_counts = (
        int(
            presence.current_activity is not None
            and bool(presence.current_activity.detail)
        ),
        int(presence.previous_activity is not None),
        int(bool(state.memories)),
        int(bool(state.interests)),
        int(bool(state.preferences)),
        int(relationship_count > 0),
        int(
            state.emotion.primary != "neutral"
            or state.emotion.intensity > 0.0
            or bool(state.emotion.cause)
        ),
    )
    content_count = (
        len(state.memories)
        + len(state.interests)
        + len(state.preferences)
        + relationship_count
        + sum(domain_counts[:2])
    )
    return sum(domain_counts), content_count, state.updated_at


def _internal_state_from_dict(
    payload: object,
    *,
    now: float,
) -> InternalState | None:
    if not isinstance(payload, dict):
        return None
    updated_at = max(0.0, _number(payload.get("updated_at"), 0.0))
    raw_emotion = payload.get("emotion")
    emotion_payload = raw_emotion if isinstance(raw_emotion, dict) else {}
    emotion_time = max(0.0, _number(emotion_payload.get("updated_at"), updated_at))
    legacy_primary = compact_text(
        emotion_payload.get("dominant") or emotion_payload.get("mood"), 32
    ).lower()
    primary = compact_text(emotion_payload.get("primary"), 32).lower()
    if not primary:
        primary = legacy_primary if legacy_primary not in {"", "steady", "relaxed"} else "neutral"
    legacy_intensity = max(
        _number(emotion_payload.get("irritation"), 0.0),
        _number(emotion_payload.get("frustration"), 0.0),
        _number(emotion_payload.get("concern"), 0.0),
    )
    candidate = EmotionState(
        primary=primary,
        intensity=max(0.0, min(1.0, _number(emotion_payload.get("intensity"), legacy_intensity))),
        cause=compact_text(emotion_payload.get("cause"), 100),
        updated_at=emotion_time,
    )
    emotion = advance_emotion(candidate, now=emotion_time)
    raw_memories = payload.get("memories")
    memories = tuple(
        memory
        for item in (raw_memories if isinstance(raw_memories, list) else [])
        if (memory := Memory.from_dict(item)) is not None
    )
    memories = _normalize_loaded_memories(list(memories))
    interests = _normalize_interests(payload.get("interests"), default_when_missing=True)
    preferences = _normalize_akane_preferences(payload.get("preferences"))
    relationship = _relationship_state_from_dict(payload.get("relationship"))
    presence = PresenceState.from_dict(payload.get("presence"), now=now)
    return InternalState(
        emotion=emotion,
        presence=presence,
        memories=memories,
        interests=interests,
        preferences=preferences,
        relationship=relationship,
        updated_at=updated_at or emotion.updated_at,
        version=4,
    )


def _relationship_state_to_dict(state: RelationshipState) -> dict[str, object]:
    def entry_payload(entry: RelationshipEntry) -> dict[str, object]:
        return {
            "summary": entry.summary,
            "importance": entry.importance,
            "confidence": entry.confidence,
            "updated_at": entry.updated_at,
        }

    return {
        "patterns": [entry_payload(entry) for entry in state.patterns],
        "shared_context": [entry_payload(entry) for entry in state.shared_context],
        "unresolved_events": [
            {**entry_payload(entry), "status": entry.status}
            for entry in state.unresolved_events
        ],
    }


def _relationship_state_from_dict(payload: object) -> RelationshipState:
    if not isinstance(payload, dict):
        return RelationshipState()

    def entries(key: str, *, unresolved: bool = False) -> list[RelationshipEntry]:
        values = payload.get(key)
        if not isinstance(values, list):
            return []
        return [
            entry
            for item in values
            if (entry := RelationshipEntry.from_dict(item, unresolved=unresolved)) is not None
        ]

    return RelationshipState(
        patterns=_cap_relationship_entries(entries("patterns"), _MAX_RELATIONSHIP_PATTERNS),
        shared_context=_cap_relationship_entries(entries("shared_context"), _MAX_RELATIONSHIP_CONTEXTS),
        unresolved_events=_cap_relationship_entries(
            entries("unresolved_events", unresolved=True),
            _MAX_UNRESOLVED_EVENTS,
        ),
    )


def _normalize_interests(
    values: object,
    *,
    default_when_missing: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return _STARTING_INTERESTS if default_when_missing else ()
    interests: list[str] = []
    seen: set[str] = set()
    for item in values:
        interest = compact_text(item, 100)
        signature = normalized_signature(interest)
        if interest and signature and signature not in seen:
            interests.append(interest)
            seen.add(signature)
    return tuple(interests)


def _normalize_akane_preferences(values: object) -> tuple[AkanePreference, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    preferences: list[AkanePreference] = []
    for item in values:
        preference = AkanePreference.from_dict(item)
        if preference is None:
            continue
        duplicate = next(
            (
                index
                for index, existing in enumerate(preferences)
                if normalized_signature(existing.topic)
                == normalized_signature(preference.topic)
            ),
            None,
        )
        if duplicate is None:
            preferences.append(preference)
        else:
            preferences[duplicate] = _stronger_preference(
                preferences[duplicate],
                preference,
            )
    return tuple(preferences)


def _merge_interests(
    current: tuple[str, ...],
    additions: tuple[str, ...],
) -> tuple[str, ...]:
    return _normalize_interests((*current, *additions))


def _merge_akane_preferences(
    current: tuple[AkanePreference, ...],
    updates: tuple[dict[str, object], ...],
    *,
    updated_at: float,
) -> tuple[AkanePreference, ...]:
    merged = list(current)
    for payload in updates:
        candidate = AkanePreference.from_dict({**payload, "updated_at": updated_at})
        if candidate is None:
            continue
        index = next(
            (
                position
                for position, existing in enumerate(merged)
                if normalized_signature(existing.topic)
                == normalized_signature(candidate.topic)
            ),
            None,
        )
        if index is None:
            merged.append(candidate)
        else:
            merged[index] = _stronger_preference(merged[index], candidate)
    return tuple(merged)


def relevant_relationship_context(state: InternalState, query: str) -> str:
    """Render only shared-history observations that overlap the current message."""

    query_text = compact_text(query, 700)
    query_terms = _memory_terms(query_text)
    if not query_terms:
        return ""
    candidates: list[tuple[float, RelationshipEntry]] = []
    for entry in (
        *state.relationship.patterns,
        *state.relationship.shared_context,
        *(event for event in state.relationship.unresolved_events if event.status == "unresolved"),
    ):
        relevance = _relationship_relevance(query_text, query_terms, entry.summary)
        if relevance >= 0.35:
            candidates.append(
                (
                    relevance + entry.importance * 0.12 + entry.confidence * 0.08,
                    entry,
                )
            )
    if not candidates:
        return ""
    selected = sorted(
        candidates,
        key=lambda item: (item[0], item[1].updated_at),
        reverse=True,
    )[:3]
    return (
        "Relevant relationship context:\n"
        + "\n".join(f"- {entry.summary}" for _score, entry in selected)
        + "\nThese are shared-history observations, not response instructions. "
        + "Akane decides what they mean and how she responds."
    )


def _relationship_relevance(query: str, query_terms: set[str], summary: str) -> float:
    specific_query_terms = query_terms - _RELATIONSHIP_RETRIEVAL_STOPWORDS
    summary_terms = _memory_terms(summary) - _RELATIONSHIP_RETRIEVAL_STOPWORDS
    if not specific_query_terms or not summary_terms:
        return 0.0
    overlap = len(specific_query_terms & summary_terms) / max(
        1,
        min(len(specific_query_terms), len(summary_terms)),
    )
    specific_query = " ".join(sorted(specific_query_terms))
    specific_summary = " ".join(sorted(summary_terms))
    return max(
        overlap,
        topic_overlap(specific_query, specific_summary),
        message_similarity(specific_query, specific_summary),
    )


def _merge_relationship(
    current: RelationshipState,
    updates: tuple[dict[str, object], ...],
    *,
    updated_at: float,
) -> RelationshipState:
    patterns = list(current.patterns)
    shared_context = list(current.shared_context)
    unresolved_events = list(current.unresolved_events)
    for update in updates:
        category = compact_text(update.get("category"), 32).lower()
        if category not in _RELATIONSHIP_UPDATE_CATEGORIES:
            continue
        summary = compact_text(update.get("summary"), 240)
        if not summary:
            continue
        candidate = RelationshipEntry(
            summary=summary,
            importance=max(0.0, min(1.0, _number(update.get("importance"), 0.0))),
            confidence=max(0.0, min(1.0, _number(update.get("confidence"), 0.0))),
            updated_at=updated_at,
            status="unresolved" if category == "unresolved_event" else "active",
        )
        if category == "resolved_event":
            _resolve_relationship_event(unresolved_events, candidate, updated_at)
        elif category == "pattern":
            _merge_relationship_entry(patterns, candidate)
        elif category == "shared_context":
            _merge_relationship_entry(shared_context, candidate)
        else:
            _merge_relationship_entry(unresolved_events, candidate)
    return RelationshipState(
        patterns=_cap_relationship_entries(patterns, _MAX_RELATIONSHIP_PATTERNS),
        shared_context=_cap_relationship_entries(shared_context, _MAX_RELATIONSHIP_CONTEXTS),
        unresolved_events=_cap_relationship_entries(unresolved_events, _MAX_UNRESOLVED_EVENTS),
    )


def _merge_relationship_entry(
    entries: list[RelationshipEntry], candidate: RelationshipEntry
) -> None:
    match = next(
        (
            index
            for index, existing in enumerate(entries)
            if _relationship_equivalent(existing.summary, candidate.summary)
        ),
        None,
    )
    if match is None:
        entries.append(candidate)
        return
    existing = entries[match]
    summary = candidate.summary if len(candidate.summary) < len(existing.summary) else existing.summary
    entries[match] = replace(
        existing,
        summary=summary,
        importance=max(existing.importance, candidate.importance),
        confidence=min(1.0, existing.confidence + (1.0 - existing.confidence) * candidate.confidence * 0.5),
        updated_at=max(existing.updated_at, candidate.updated_at),
    )


def _resolve_relationship_event(
    events: list[RelationshipEntry], candidate: RelationshipEntry, updated_at: float
) -> None:
    unresolved = [
        (index, event)
        for index, event in enumerate(events)
        if event.status == "unresolved"
    ]
    if not unresolved:
        return
    index, event = max(
        unresolved,
        key=lambda item: _relationship_similarity(item[1].summary, candidate.summary),
    )
    if _relationship_similarity(event.summary, candidate.summary) < 0.35:
        return
    events[index] = replace(
        event,
        status="resolved",
        importance=max(event.importance, candidate.importance),
        confidence=min(1.0, event.confidence + (1.0 - event.confidence) * candidate.confidence * 0.5),
        updated_at=updated_at,
    )


def _relationship_equivalent(left: str, right: str) -> bool:
    if normalized_signature(left) == normalized_signature(right):
        return True
    return _relationship_similarity(left, right) >= 0.72


def _relationship_similarity(left: str, right: str) -> float:
    return max(message_similarity(left, right), topic_overlap(left, right))


def _cap_relationship_entries(
    entries: list[RelationshipEntry], limit: int
) -> tuple[RelationshipEntry, ...]:
    return tuple(
        sorted(
            entries,
            key=lambda entry: (entry.status == "unresolved", entry.importance, entry.confidence, entry.updated_at),
            reverse=True,
        )[:limit]
    )


def _stronger_preference(
    existing: AkanePreference,
    candidate: AkanePreference,
) -> AkanePreference:
    """Keep a strong conclusion from being replaced by a weaker impression."""

    if candidate.stance != existing.stance and candidate.strength + 0.12 < existing.strength:
        return existing
    if candidate.stance == existing.stance and candidate.strength < existing.strength:
        return existing
    return candidate


def _semantic_relevance(query: str, query_terms: set[str], memory: Memory) -> float:
    memory_terms = _memory_terms(memory.content) | {
        part
        for tag in memory.tags
        for part in tag.replace("slot:", "").replace(":", "-").split("-")
        if len(part) >= 3
    }
    exact = query_terms & memory_terms
    unmatched_memory = memory_terms - exact
    fuzzy_matches = 0
    for query_term in query_terms - exact:
        match = next(
            (
                memory_term
                for memory_term in unmatched_memory
                if query_term[0] == memory_term[0]
                and abs(len(query_term) - len(memory_term)) <= 2
                and message_similarity(query_term, memory_term) >= 0.78
            ),
            None,
        )
        if match is not None:
            fuzzy_matches += 1
            unmatched_memory.remove(match)
    overlap = (len(exact) + fuzzy_matches * 0.8) / max(
        1,
        min(len(query_terms), len(memory_terms)),
    )
    return min(1.0, max(overlap, topic_overlap(query, memory.content)))


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


def _slot_value(value: str) -> str:
    terms = sorted(_memory_terms(value))[:5]
    return "-".join(terms) or normalized_signature(value).replace(" ", "-")[:32]


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
            kind="profile",
            source_type="unknown",
            source_reference="",
            canonical_key=_candidate_key(candidate),
            scope="profile",
            updated_at=created_at,
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
