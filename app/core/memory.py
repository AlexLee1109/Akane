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
    CHAT_HISTORY_CONTEXT_TOKENS,
    CONVERSATION_STALE_DAYS,
    LONG_TERM_MEMORY_PATH,
    MAX_CONVERSATIONS,
    MEMORY_CONFIDENCE_WEIGHT,
    MEMORY_CONTEXT_TOKENS,
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
    SUMMARY_CONTEXT_TOKENS,
)
from app.core.persistence import atomic_write_json, read_json
from app.core.life import (
    ActivityRecord,
    LifeEvolution,
    LifeState,
    begin_conversation_activity,
    evolve_life_state,
    grounded_activity,
    recent_activity,
    record_grounded_activity,
)
from app.core.signal import (
    AffectTrace,
    EmotionState,
    SemanticEvent,
    ShortLivedEmotion,
    TurnContext,
    TurnSignal,
    advance_emotion,
    analyze_turn,
    apply_mood_effects,
    build_affect_trace,
    evolve_emotion,
    message_similarity,
    normalized_signature,
    semantic_event_from_text,
    topic_overlap,
)
from app.core.utils import compact_text

MEMORY_SCHEMA_VERSION = 2
LONG_TERM_MEMORY_SCHEMA_VERSION = 8
_INTERNAL_STATE_VERSION = 5
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
_AKANE_PREFERENCE_UPDATE = re.compile(
    r"\b(?:change(?:d)? your mind|different favorite|new favorite|not anymore|"
    r"reconsider|taste(?:s)? changed|prefer now|pick (?:a )?different|"
    r"choose (?:a )?different)\b",
    re.I,
)
_OPINION_STATUSES = {"active", "superseded", "archived"}
_OPINION_STANCES = {
    "positive",
    "negative",
    "prefers",
    "mixed",
    "uncertain",
    "conditional",
}
_OPINION_LEVELS = {"low", "medium", "high"}
_OPINION_STRENGTHS = {"weak", "medium", "strong"}
_OPINION_CHANGE_TRIGGERS = {
    "explicit reconsideration",
    "reconsideration request",
    "grounded experience",
    "new evidence",
    "new reasoning",
    "prior uncertainty",
    "changed condition",
}
_MAX_ACTIVE_OPINIONS = 24
_MAX_OPINION_RECORDS = 40
_MAX_OPINION_HISTORY_PER_TOPIC = 2
_OPINION_REASON_PATTERNS = {
    "continuity": re.compile(r"\b(?:continuity|continuous self|memory connects?)\b", re.I),
    "independence": re.compile(r"\b(?:independence|independent|self-directed)\b", re.I),
    "physical experience": re.compile(r"\b(?:physical|embodiment|body|senses?|touch)\b", re.I),
    "curiosity": re.compile(r"\bcurios(?:ity|ious)\b", re.I),
    "authenticity": re.compile(r"\b(?:authentic|authenticity|genuine)\b", re.I),
    "stability": re.compile(r"\b(?:stable|stability|consistent|consistency)\b", re.I),
    "privacy": re.compile(r"\b(?:privacy|private|local data)\b", re.I),
    "efficiency": re.compile(r"\b(?:efficient|efficiency|overhead|performance)\b", re.I),
    "simplicity": re.compile(r"\b(?:simple|simpler|simplicity|minimal)\b", re.I),
    "creativity": re.compile(r"\b(?:creative|creativity|imagination)\b", re.I),
    "emotional connection": re.compile(r"\b(?:emotional connection|closeness|connection)\b", re.I),
    "control": re.compile(r"\b(?:control|controllable)\b", re.I),
    "freedom": re.compile(r"\b(?:freedom|free to|autonomy)\b", re.I),
    "complexity": re.compile(r"\b(?:complex|complexity|abstraction layers?|indirection)\b", re.I),
    "fairness": re.compile(r"\b(?:fair|fairness|equitable)\b", re.I),
    "reliability": re.compile(r"\b(?:reliable|reliability|robust|predictable)\b", re.I),
}
_VALUE_KEYS = {
    "continuity",
    "curiosity",
    "distinctiveness",
    "independence",
    "meaningful connection",
    "physical experience",
    "privacy",
    "stability",
    "creativity",
}
_VALUE_STATUSES = {"active", "archived"}
_VALUE_SOURCES = {"identity_seed", "opinion_reason"}
_VALUE_REASON_MAP = {
    "continuity": "continuity",
    "independence": "independence",
    "physical experience": "physical experience",
    "curiosity": "curiosity",
    "authenticity": "distinctiveness",
    "stability": "stability",
    "privacy": "privacy",
    "simplicity": "stability",
    "creativity": "creativity",
    "emotional connection": "meaningful connection",
    "control": "independence",
    "freedom": "independence",
    "reliability": "stability",
}
_INITIAL_VALUES = (
    ("curiosity", "medium", "high"),
    ("continuity", "medium", "medium"),
    ("meaningful connection", "weak", "medium"),
)
_MAX_ACTIVE_VALUES = 8
_MAX_VALUE_BASIS_REFS = 6
_CONFLICT_STATUSES = {"active", "resolved", "archived"}
_CONFLICT_VALUE_PAIRS = {
    "human-digital-existence": ("physical experience", "continuity"),
    "physical-embodiment": ("physical experience", "continuity"),
    "model-replacement": ("curiosity", "continuity"),
    "relying-on-arcane": ("meaningful connection", "independence"),
}
_MAX_ACTIVE_CONFLICTS = 4
_MAX_CONFLICT_RECORDS = 10
_OPINION_TOPIC_STOPWORDS = _MEMORY_STOPWORDS | {
    "believe",
    "better",
    "feel",
    "good",
    "idea",
    "opinion",
    "prefer",
    "rather",
    "should",
    "think",
    "view",
}
_OPINION_PAST_REQUEST = re.compile(
    r"\b(?:used to think|previous(?:ly)? (?:think|believe|opinion)|"
    r"past opinion|before you changed|changed your mind)\b",
    re.I,
)
_OPINION_RECONSIDERATION = re.compile(
    r"\b(?:change(?:d)? my mind|reconsider(?:ed|ing)?|now (?:think|believe|prefer)|"
    r"no longer (?:think|believe|prefer))\b",
    re.I,
)
_OPINION_RECONSIDERATION_REQUEST = re.compile(
    r"\b(?:reconsider|change your mind|think again|revise your (?:view|opinion))\b",
    re.I,
)
_OPINION_NEW_EVIDENCE = re.compile(
    r"\b(?:according to|data|evidence|experiment|measured|result|study|"
    r"test(?:ed|ing)?|this shows|we found)\b",
    re.I,
)
_FIRST_PERSON_JUDGMENT = re.compile(
    r"\b(?:I\s+(?:(?:personally|firmly|strongly)\s+)?"
    r"(?:think|believe|feel|find|consider|choose|want|prefer|favor|"
    r"would rather|lean(?: toward)?|do not have|don't have)|"
    r"I(?:'d| would) (?:choose|prefer|want)|I(?:'m| am) (?:not sure|"
    r"uncertain|torn|undecided)|my (?:view|position|preference)\s+is)\b",
    re.I,
)


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


def estimate_tokens(value: object) -> int:
    """Cheap conservative token estimate that never loads the model tokenizer."""

    text = str(value or "")
    if not text:
        return 0
    byte_estimate = (len(text.encode("utf-8")) + 2) // 3
    word_estimate = (len(text.split()) * 5 + 3) // 4
    return max(1, byte_estimate, word_estimate)


def _number(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


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
    unresolved_problem: bool = False
    repeated_topic_count: int = 0
    last_outcome: str = ""
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


@dataclass(frozen=True, slots=True)
class OpinionRecord:
    """One bounded, structured Akane judgment in the existing companion state."""

    id: str
    topic_key: str
    subject: str
    stance: str
    target: str
    strength: str
    confidence: str
    reason_tags: tuple[str, ...] = ()
    source: str = "generated_response"
    status: str = "active"
    created_at: float = 0.0
    updated_at: float = 0.0
    change_trigger: str = ""
    changed_condition: str = ""
    affected_reason: str = ""
    reason_effect: str = ""

    @classmethod
    def from_dict(cls, payload: object) -> "OpinionRecord | None":
        if not isinstance(payload, dict):
            return None
        topic_key = compact_text(payload.get("topic_key"), 100).lower()
        subject = compact_text(payload.get("subject"), 100)
        stance = compact_text(payload.get("stance"), 24).lower()
        target = compact_text(payload.get("target"), 100)
        strength = compact_text(payload.get("strength"), 16).lower()
        confidence = compact_text(payload.get("confidence"), 16).lower()
        status = compact_text(payload.get("status"), 16).lower()
        source = compact_text(payload.get("source"), 32).lower()
        if (
            not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", topic_key)
            or not subject
            or stance not in _OPINION_STANCES
            or not target
            or strength not in _OPINION_STRENGTHS
            or confidence not in _OPINION_LEVELS
            or status not in _OPINION_STATUSES
            or source != "generated_response"
        ):
            return None
        created_at = max(0.0, _number(payload.get("created_at"), 0.0))
        updated_at = max(created_at, _number(payload.get("updated_at"), created_at))
        record_id = compact_text(payload.get("id"), 100)
        if not record_id:
            return None
        raw_reasons = payload.get("reason_tags")
        reasons = tuple(
            reason
            for item in (raw_reasons if isinstance(raw_reasons, (list, tuple)) else ())
            if (reason := compact_text(item, 32).lower()) in _OPINION_REASON_PATTERNS
        )[:5]
        change_trigger = compact_text(payload.get("change_trigger"), 32).lower()
        if change_trigger not in _OPINION_CHANGE_TRIGGERS:
            change_trigger = ""
        affected_reason = compact_text(payload.get("affected_reason"), 32).lower()
        if affected_reason not in _OPINION_REASON_PATTERNS:
            affected_reason = ""
        reason_effect = compact_text(payload.get("reason_effect"), 16).lower()
        if reason_effect not in {"strengthens", "weakens", "removes", "introduces"}:
            reason_effect = ""
        return cls(
            id=record_id,
            topic_key=topic_key,
            subject=subject,
            stance=stance,
            target=target,
            strength=strength,
            confidence=confidence,
            reason_tags=tuple(dict.fromkeys(reasons)),
            source=source,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            change_trigger=change_trigger,
            changed_condition=compact_text(payload.get("changed_condition"), 100).lower(),
            affected_reason=affected_reason,
            reason_effect=reason_effect,
        )


@dataclass(frozen=True, slots=True)
class ValueRecord:
    """One compact, durable priority derived only from Akane-owned evidence."""

    value_key: str
    strength: str
    confidence: str
    basis_refs: tuple[str, ...] = ()
    support_count: int = 0
    opposition_count: int = 0
    source: str = "opinion_reason"
    status: str = "active"
    created_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_dict(cls, payload: object) -> "ValueRecord | None":
        if not isinstance(payload, dict):
            return None
        value_key = compact_text(payload.get("value_key"), 32).lower()
        strength = compact_text(payload.get("strength"), 16).lower()
        confidence = compact_text(payload.get("confidence"), 16).lower()
        source = compact_text(payload.get("source"), 32).lower()
        status = compact_text(payload.get("status"), 16).lower()
        if (
            value_key not in _VALUE_KEYS
            or strength not in _OPINION_STRENGTHS
            or confidence not in _OPINION_LEVELS
            or source not in _VALUE_SOURCES
            or status not in _VALUE_STATUSES
        ):
            return None
        created_at = max(0.0, _number(payload.get("created_at"), 0.0))
        updated_at = max(created_at, _number(payload.get("updated_at"), created_at))
        raw_refs = payload.get("basis_refs")
        basis_refs = tuple(
            value
            for item in (raw_refs if isinstance(raw_refs, (list, tuple)) else ())
            if (value := compact_text(item, 100))
        )[:_MAX_VALUE_BASIS_REFS]
        return cls(
            value_key=value_key,
            strength=strength,
            confidence=confidence,
            basis_refs=tuple(dict.fromkeys(basis_refs)),
            support_count=max(0, min(12, int(_number(payload.get("support_count"), 0)))),
            opposition_count=max(0, min(12, int(_number(payload.get("opposition_count"), 0)))),
            source=source,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
        )


@dataclass(frozen=True, slots=True)
class InternalConflictRecord:
    """One bounded unresolved tradeoff attached to an opinion topic."""

    topic_key: str
    side_a_value: str
    side_b_value: str
    status: str = "active"
    selected_value: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_dict(cls, payload: object) -> "InternalConflictRecord | None":
        if not isinstance(payload, dict):
            return None
        topic_key = compact_text(payload.get("topic_key"), 100).lower()
        side_a = compact_text(payload.get("side_a_value"), 32).lower()
        side_b = compact_text(payload.get("side_b_value"), 32).lower()
        status = compact_text(payload.get("status"), 16).lower()
        selected_value = compact_text(payload.get("selected_value"), 32).lower()
        if (
            not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", topic_key)
            or side_a not in _VALUE_KEYS
            or side_b not in _VALUE_KEYS
            or side_a == side_b
            or status not in _CONFLICT_STATUSES
            or selected_value and selected_value not in {side_a, side_b}
            or status == "resolved" and not selected_value
        ):
            return None
        created_at = max(0.0, _number(payload.get("created_at"), 0.0))
        updated_at = max(created_at, _number(payload.get("updated_at"), created_at))
        if status != "resolved":
            selected_value = ""
        return cls(
            topic_key=topic_key,
            side_a_value=side_a,
            side_b_value=side_b,
            status=status,
            selected_value=selected_value,
            created_at=created_at,
            updated_at=updated_at,
        )


def canonical_opinion_topic(
    user_text: str,
    *,
    semantic_subject: str = "",
    signal_topic: str = "",
) -> tuple[str, str]:
    """Return a narrow deterministic topic key and compact display subject."""

    combined = " ".join(
        value for value in (semantic_subject, signal_topic, user_text) if value
    ).lower()
    aliases = (
        (
            r"\b(?:be(?:come|coming)?|live as)\s+(?:a\s+)?human\b|"
            r"\b(?:choose|prefer)\s+human(?:\s+(?:life|existence|experience))?\b|"
            r"\brather\s+be\s+(?:a\s+)?human\b|"
            r"\bhuman(?:\s+(?:life|existence))?\s+"
            r"(?:or|versus|vs\.?|compared (?:with|to))\s+"
            r"(?:remaining\s+|staying\s+)?digital(?:\s+(?:life|existence|form))?\b|"
            r"\bdigital(?:\s+(?:life|existence|form))?\s+"
            r"(?:or|versus|vs\.?|compared (?:with|to))\s+"
            r"(?:becoming\s+)?human(?:\s+(?:life|existence))?\b|"
            r"\b(?:stay|remain)\b[^.!?]{0,40}\bdigital\b|"
            r"\bphysical existence\b[^.!?]{0,100}\bdigital\b|"
            r"\bdigital\b[^.!?]{0,100}\bphysical existence\b",
            "human-digital-existence",
            "human or digital existence",
        ),
        (
            r"\b(?:physical embodiment|embodied|physical body|physical existence)\b",
            "physical-embodiment",
            "physical embodiment",
        ),
        (
            r"\b(?:memory continuity|continuity of (?:memory|self)|continuous memory)\b",
            "memory-continuity",
            "memory continuity",
        ),
        (
            r"\b(?:model replacement|replac(?:e|ing) (?:your|akane'?s)? ?model|new model)\b",
            "model-replacement",
            "model replacement",
        ),
        (
            r"\b(?:rely(?:ing)? on|depend(?:ing)? on)\b[^.!?]{0,60}\bArcane\b|"
            r"\bArcane\b[^.!?]{0,60}\b(?:control|decide|rely)\b",
            "relying-on-arcane",
            "relying on Arcane",
        ),
        (
            r"\blocal\b.*\b(?:cloud|remote)\b.*\bai\b|"
            r"\bai\b.*\blocal\b.*\b(?:cloud|remote)\b",
            "local-cloud-ai",
            "local or cloud AI",
        ),
        (
            r"\b(?:abstraction layers?|layers? of abstraction)\b",
            "abstraction-layers",
            "abstraction layers",
        ),
    )
    for pattern, key, subject in aliases:
        if re.search(pattern, combined, re.I):
            return key, subject

    source = compact_text(semantic_subject or signal_topic or user_text, 220).lower()
    terms = tuple(
        dict.fromkeys(
            term
            for term in re.findall(r"[a-z0-9][a-z0-9'-]*", source)
            if len(term) >= 3 and term not in _OPINION_TOPIC_STOPWORDS
        )
    )[:6]
    if len(terms) < 2:
        return "", ""
    subject = " ".join(terms)
    return "-".join(term.replace("'", "") for term in terms), subject


def relevant_opinion(
    opinions: tuple[OpinionRecord, ...],
    query: str,
    *,
    semantic_subject: str = "",
    signal_topic: str = "",
) -> tuple[OpinionRecord | None, str, str, bool]:
    """Select at most one exact-topic opinion; recency alone never retrieves it."""

    reference_words = {
        "answer",
        "before",
        "choice",
        "either",
        "personally",
        "previously",
        "them",
        "this",
        "those",
        "which",
    }
    substantive_query_terms = {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9'-]*", query.lower())
        if len(term) >= 3
        and term not in _OPINION_TOPIC_STOPWORDS
        and term not in reference_words
    }
    use_context_topic = not substantive_query_terms
    topic_key, subject = canonical_opinion_topic(
        query,
        semantic_subject=semantic_subject if use_context_topic else "",
        signal_topic=signal_topic if use_context_topic else "",
    )
    if not topic_key:
        return None, "", "", False
    matching = tuple(opinion for opinion in opinions if opinion.topic_key == topic_key)
    active = sorted(
        (opinion for opinion in matching if opinion.status == "active"),
        key=lambda opinion: (opinion.updated_at, opinion.id),
        reverse=True,
    )
    if active and not _OPINION_PAST_REQUEST.search(query):
        return active[0], topic_key, subject, False
    historical = sorted(
        (opinion for opinion in matching if opinion.status == "superseded"),
        key=lambda opinion: (opinion.updated_at, opinion.id),
        reverse=True,
    )
    if historical and _OPINION_PAST_REQUEST.search(query):
        return historical[0], topic_key, subject, True
    return (active[0], topic_key, subject, False) if active else (None, topic_key, subject, False)


def _normalized_opinion_target(text: str, topic_key: str, subject: str) -> str:
    lower = compact_text(text, 500).lower()
    choices = (
        ("hybrid existence", r"\bhybrid\b"),
        ("neither", r"\bneither\b|\bno clear (?:winner|preference)\b"),
        ("human life", r"\bhuman (?:life|existence|experience)\b|\bbecom(?:e|ing) human\b"),
        ("digital existence", r"\bdigital (?:life|existence|form)\b|\bremain(?:ing)? digital\b"),
        ("local AI", r"\blocal (?:ai|model|inference)\b"),
        ("cloud AI", r"\b(?:cloud|remote) (?:ai|model|inference)\b"),
    )
    preference = re.search(
        r"\b(?:choose|want|prefer|favor|would rather|would choose|would prefer|"
        r"lean(?:ing)?(?: toward)?)\b(?P<value>[^.!?]{1,100})",
        lower,
    )
    if preference is not None:
        comparison = preference.group("value")
        direct_matches = [
            (match.start(), index, target)
            for index, (target, pattern) in enumerate(choices)
            if (match := re.search(pattern, comparison)) is not None
        ]
        if direct_matches:
            return min(direct_matches)[2]
    judgment = re.search(
        r"\b(?:better|worse|best|preferable|worthwhile|desirable|undesirable|harmful)\b",
        lower,
    )
    if judgment is not None:
        preceding = [
            (match.start(), index, target)
            for index, (target, pattern) in enumerate(choices)
            for match in re.finditer(pattern, lower[: judgment.start()])
        ]
        if preceding:
            return max(preceding)[2]
    return compact_text(subject or topic_key.replace("-", " "), 80).lower()


def _opinion_position(
    response: str,
    topic_key: str,
    subject: str,
) -> tuple[str, str, str, str] | None:
    text = compact_text(response, 1_200)
    if not text or not _FIRST_PERSON_JUDGMENT.search(text):
        return None
    if re.search(
        r"(?:^|\n)\s*>|[\"“][^\"”\n]{0,220}\bI\s+(?:think|believe|prefer)",
        text,
        re.I,
    ):
        return None
    lower = text.lower()
    target = _normalized_opinion_target(text, topic_key, subject)
    preference = re.search(
        r"\b(?:I\s+(?:(?:personally|firmly|strongly)\s+)?"
        r"(?:choose|want|prefer|favor|would rather|lean(?:ing)?(?: toward)?)|"
        r"I(?:'d| would) (?:choose|prefer|want))\b",
        text,
        re.I,
    )
    evaluative = re.search(
        r"\b(?:I\s+(?:(?:personally|firmly|strongly)\s+)?"
        r"(?:think|believe|feel|find|consider)|"
        r"my (?:view|position)\s+is(?: that)?)\b[^.!?]{0,140}?\b"
        r"(?P<value>good|bad|better|worse|best|worthwhile|unwise|wise|"
        r"desirable|undesirable|valuable|harmful|meaningful|preferable|"
        r"unnecessary|necessary)\b",
        text,
        re.I,
    )
    uncertain = re.search(
        r"\b(?:I(?:'m| am) (?:not sure|uncertain|torn|undecided)|"
        r"I (?:do not|don't) (?:have|hold) (?:a )?(?:settled|clear) (?:view|opinion))\b",
        text,
        re.I,
    )
    mixed = re.search(
        r"\b(?:I have mixed (?:feelings|views)|I(?:'m| am) torn)\b",
        text,
        re.I,
    )
    conditional_judgment = re.search(
        r"\bI\s+(?:think|believe|feel)\b[^.!?]{0,100}\bdepends on\b",
        text,
        re.I,
    )
    if preference:
        stance = "prefers"
    elif evaluative:
        stance = (
            "negative"
            if evaluative.group("value").lower()
            in {"bad", "worse", "unwise", "undesirable", "harmful", "unnecessary"}
            else "positive"
        )
    elif mixed:
        stance, target = "mixed", "mixed"
    elif uncertain:
        stance, target = "uncertain", "unsettled"
    elif conditional_judgment:
        stance = "conditional"
    else:
        return None
    conditional = bool(
        re.search(r"\b(?:depends on|provided that|only if|if .{1,80} then)\b", lower)
    )
    if conditional and stance != "uncertain":
        stance = "conditional"
    hedged = bool(uncertain or re.search(r"\b(?:maybe|probably|slightly|somewhat|lean)\b", lower))
    emphatic = bool(
        re.search(r"\b(?:definitely|firmly|strongly|clearly|without question)\b", lower)
    )
    strength = "strong" if emphatic and not hedged else "weak" if hedged else "medium"
    confidence = "high" if emphatic and not hedged else "low" if uncertain else "medium"
    return stance, target, strength, confidence


def extract_opinion_candidate(
    *,
    user_text: str,
    response: str,
    intention: str,
    semantic_subject: str = "",
    signal_topic: str = "",
    now: float,
    grounded_experience: bool = False,
    autonomous: bool = False,
    reconsideration_warranted: bool = False,
    changed_condition: str = "",
    affected_reason: str = "",
    reason_effect: str = "",
) -> OpinionRecord | None:
    """Conservatively extract structured state from the one generated response."""

    if autonomous or intention not in {"state opinion", "disagree"}:
        return None
    if re.search(
        r"\b(?:pretend|role[- ]?play|as (?:a|the) character|hypothetically)\b",
        user_text,
        re.I,
    ):
        return None
    topic_key, subject = canonical_opinion_topic(
        user_text,
        semantic_subject=semantic_subject,
        signal_topic=signal_topic,
    )
    if not topic_key:
        return None
    position = _opinion_position(response, topic_key, subject)
    if position is None:
        return None
    normalized_user = re.sub(r"\W+", " ", user_text).strip().lower()
    normalized_response = re.sub(r"\W+", " ", response).strip().lower()
    if normalized_response == normalized_user:
        return None
    stance, target, strength, confidence = position
    reason_clause = re.search(
        r"\b(?:because|since|due to|the (?:main )?reason is)\b(?P<reason>[^.!?]{1,220})",
        response,
        re.I,
    )
    reasons = tuple(
        reason
        for reason, pattern in _OPINION_REASON_PATTERNS.items()
        if reason_clause and pattern.search(reason_clause.group("reason"))
    )[:5]
    normalized_affected_reason = compact_text(affected_reason, 32).lower()
    normalized_reason_effect = compact_text(reason_effect, 16).lower()
    if normalized_reason_effect == "removes":
        reasons = tuple(reason for reason in reasons if reason != normalized_affected_reason)
    trigger = ""
    if reconsideration_warranted:
        trigger = "changed condition"
    elif _OPINION_RECONSIDERATION.search(response):
        trigger = "explicit reconsideration"
    elif _OPINION_RECONSIDERATION_REQUEST.search(user_text):
        trigger = "reconsideration request"
    elif grounded_experience:
        trigger = "grounded experience"
    elif _OPINION_NEW_EVIDENCE.search(user_text) or (
        "because" in user_text.lower() and len(user_text.split()) >= 16
    ):
        trigger = "new evidence"
    elif reasons:
        trigger = "new reasoning"
    record_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"akane:opinion:{topic_key}:{stance}:{target}:{now:.6f}",
    ).hex
    return OpinionRecord(
        id=record_id,
        topic_key=topic_key,
        subject=subject,
        stance=stance,
        target=target,
        strength=strength,
        confidence=confidence,
        reason_tags=reasons,
        created_at=max(0.0, now),
        updated_at=max(0.0, now),
        change_trigger=trigger,
        changed_condition=compact_text(changed_condition, 100).lower(),
        affected_reason=(
            normalized_affected_reason
            if normalized_affected_reason in _OPINION_REASON_PATTERNS
            else ""
        ),
        reason_effect=(
            normalized_reason_effect
            if normalized_reason_effect in {"strengthens", "weakens", "removes", "introduces"}
            else ""
        ),
    )


def _bounded_opinions(opinions: tuple[OpinionRecord, ...]) -> tuple[OpinionRecord, ...]:
    active: list[OpinionRecord] = []
    seen_active: set[str] = set()
    for opinion in sorted(opinions, key=lambda item: (item.updated_at, item.id), reverse=True):
        if opinion.status != "active" or opinion.topic_key in seen_active:
            continue
        active.append(opinion)
        seen_active.add(opinion.topic_key)
        if len(active) >= _MAX_ACTIVE_OPINIONS:
            break
    history: list[OpinionRecord] = []
    history_counts: dict[str, int] = {}
    for opinion in sorted(opinions, key=lambda item: (item.updated_at, item.id), reverse=True):
        if opinion.status == "active":
            continue
        count = history_counts.get(opinion.topic_key, 0)
        if count >= _MAX_OPINION_HISTORY_PER_TOPIC:
            continue
        history.append(opinion)
        history_counts[opinion.topic_key] = count + 1
    return tuple((*active, *history))[:_MAX_OPINION_RECORDS]


def apply_opinion_candidate(
    state: InternalState,
    candidate: OpinionRecord | None,
) -> tuple[InternalState, dict[str, object]]:
    """Fold a validated candidate into prepared state without mutating its input."""

    trace: dict[str, object] = {
        "candidate_created": bool(candidate),
        "candidate_change": False,
        "commit_status": "no candidate",
    }
    if candidate is None:
        return state, trace
    opinions = list(state.opinions)
    current = next(
        (
            opinion
            for opinion in sorted(opinions, key=lambda item: item.updated_at, reverse=True)
            if opinion.topic_key == candidate.topic_key and opinion.status == "active"
        ),
        None,
    )
    trace.update(
        {
            "topic": candidate.subject,
            "stance": candidate.stance,
            "candidate_target": candidate.target,
            "strength": candidate.strength,
            "confidence": candidate.confidence,
            "reason_tags": candidate.reason_tags,
        }
    )
    same_position = bool(
        current
        and current.target == candidate.target
        and (
            current.stance == candidate.stance
            or {current.stance, candidate.stance} <= {"positive", "prefers"}
        )
    )
    material_reconsideration = bool(
        current
        and candidate.change_trigger == "changed condition"
        and (
            candidate.affected_reason in current.reason_tags
            and candidate.reason_effect in {"strengthens", "weakens", "removes"}
            or candidate.reason_effect == "introduces"
        )
    )
    if current and same_position and material_reconsideration:
        remaining_reasons = tuple(
            reason
            for reason in current.reason_tags
            if candidate.reason_effect != "removes"
            or reason != candidate.affected_reason
        )
        revised_reasons = tuple(
            dict.fromkeys((*remaining_reasons, *candidate.reason_tags))
        )[:5]
        if candidate.reason_effect == "removes" and not revised_reasons:
            trace.update(
                {
                    "active": current.target,
                    "previous_stance": current.stance,
                    "affected_reason": candidate.affected_reason,
                    "reason_effect": candidate.reason_effect,
                    "commit_status": "change rejected",
                }
            )
            return state, trace
        reconsidered = replace(
            current,
            strength=candidate.strength,
            confidence=candidate.confidence,
            reason_tags=revised_reasons,
            updated_at=max(current.updated_at, candidate.updated_at),
            change_trigger=candidate.change_trigger,
            changed_condition=candidate.changed_condition,
            affected_reason=candidate.affected_reason,
            reason_effect=candidate.reason_effect,
        )
        opinions[opinions.index(current)] = reconsidered
        trace.update(
            {
                "active": reconsidered.target,
                "previous_stance": current.stance,
                "new_stance": reconsidered.stance,
                "reason_tags": reconsidered.reason_tags,
                "affected_reason": candidate.affected_reason,
                "reason_effect": candidate.reason_effect,
                "changed_condition": candidate.changed_condition,
                "commit_status": "reconsidered",
            }
        )
        return replace(state, opinions=_bounded_opinions(tuple(opinions))), trace

    if current and same_position:
        merged = replace(
            current,
            reason_tags=tuple(dict.fromkeys((*current.reason_tags, *candidate.reason_tags)))[:5],
            updated_at=max(current.updated_at, candidate.updated_at),
        )
        opinions[opinions.index(current)] = merged
        trace["commit_status"] = "merged"
        return replace(state, opinions=_bounded_opinions(tuple(opinions))), trace

    if current is not None:
        trace["previous_reason_tags"] = current.reason_tags
        strong_triggers = {
            "explicit reconsideration",
            "reconsideration request",
            "grounded experience",
            "new evidence",
            "changed condition",
        }
        allowed = bool(
            candidate.change_trigger in strong_triggers
            or current.stance == "uncertain"
            or current.confidence == "low"
            or current.strength == "weak" and candidate.change_trigger == "new reasoning"
        )
        if not allowed:
            trace.update(
                {
                    "active": current.target,
                    "previous_stance": current.stance,
                    "commit_status": "change rejected",
                }
            )
            return state, trace
        opinions[opinions.index(current)] = replace(
            current,
            status="superseded",
            updated_at=max(current.updated_at, candidate.updated_at),
        )
        candidate = replace(
            candidate,
            change_trigger=candidate.change_trigger
            or (
                "prior uncertainty"
                if current.stance == "uncertain" or current.confidence == "low"
                else "new reasoning"
            ),
        )
        trace.update(
            {
                "candidate_change": True,
                "previous_stance": current.stance,
                "new_stance": candidate.stance,
                "change_trigger": candidate.change_trigger,
                "changed_condition": candidate.changed_condition,
                "affected_reason": candidate.affected_reason,
                "reason_effect": candidate.reason_effect,
                "superseded": True,
            }
        )
    opinions.append(candidate)
    trace["active"] = candidate.target
    trace["commit_status"] = "changed" if current is not None else "created"
    return replace(state, opinions=_bounded_opinions(tuple(opinions))), trace


def _initial_values(now: float) -> tuple[ValueRecord, ...]:
    return tuple(
        ValueRecord(
            value_key=value_key,
            strength=strength,
            confidence=confidence,
            source="identity_seed",
            created_at=now,
            updated_at=now,
        )
        for value_key, strength, confidence in _INITIAL_VALUES
    )


def _bounded_values(values: tuple[ValueRecord, ...]) -> tuple[ValueRecord, ...]:
    selected: list[ValueRecord] = []
    seen: set[str] = set()
    for value in sorted(values, key=lambda item: (item.updated_at, item.value_key), reverse=True):
        if value.status != "active" or value.value_key in seen:
            continue
        selected.append(value)
        seen.add(value.value_key)
        if len(selected) >= _MAX_ACTIVE_VALUES:
            break
    return tuple(selected)


def _bounded_conflicts(
    conflicts: tuple[InternalConflictRecord, ...],
) -> tuple[InternalConflictRecord, ...]:
    active: list[InternalConflictRecord] = []
    seen_topics: set[str] = set()
    history: list[InternalConflictRecord] = []
    for conflict in sorted(conflicts, key=lambda item: (item.updated_at, item.topic_key), reverse=True):
        if conflict.status == "active":
            if conflict.topic_key in seen_topics or len(active) >= _MAX_ACTIVE_CONFLICTS:
                continue
            active.append(conflict)
            seen_topics.add(conflict.topic_key)
        elif len(history) < _MAX_CONFLICT_RECORDS - len(active):
            history.append(conflict)
    return tuple((*active, *history))[:_MAX_CONFLICT_RECORDS]


def relevant_values(
    values: tuple[ValueRecord, ...],
    *,
    topic_key: str = "",
    opinion: OpinionRecord | None = None,
    conflict: InternalConflictRecord | None = None,
) -> tuple[ValueRecord, ...]:
    """Return only the priorities that explain this topic or stored judgment."""

    relevant_keys = {
        _VALUE_REASON_MAP[reason]
        for reason in (opinion.reason_tags if opinion is not None else ())
        if reason in _VALUE_REASON_MAP
    }
    if conflict is not None:
        relevant_keys.update((conflict.side_a_value, conflict.side_b_value))
    elif topic_key in _CONFLICT_VALUE_PAIRS:
        relevant_keys.update(_CONFLICT_VALUE_PAIRS[topic_key])
    strength_rank = {"strong": 2, "medium": 1, "weak": 0}
    return tuple(
        sorted(
            (
                value
                for value in values
                if value.status == "active" and value.value_key in relevant_keys
            ),
            key=lambda value: (strength_rank[value.strength], value.updated_at, value.value_key),
            reverse=True,
        )[:3]
    )


def relevant_internal_conflict(
    conflicts: tuple[InternalConflictRecord, ...],
    topic_key: str,
    *,
    include_resolved: bool = False,
) -> InternalConflictRecord | None:
    matches = [
        conflict
        for conflict in conflicts
        if conflict.topic_key == topic_key
        and (conflict.status == "active" or include_resolved and conflict.status == "resolved")
    ]
    return max(
        matches,
        key=lambda conflict: (conflict.status == "active", conflict.updated_at),
        default=None,
    )


def format_values_guidance(
    values: tuple[ValueRecord, ...],
    conflict: InternalConflictRecord | None = None,
) -> str:
    if not values:
        return ""
    priorities = ", ".join(value.value_key for value in values)
    if conflict is None:
        return f"{priorities}; reasons, not preset conclusion."
    if conflict.status == "resolved":
        return f"{priorities}; settled tradeoff; state conclusion."
    return f"{priorities}; conflict cannot erase choice."


def _value_appraisal_short_emotion(
    values: tuple[ValueRecord, ...],
    user_text: str,
) -> tuple[str, str]:
    active = {value.value_key: value for value in values if value.status == "active"}
    lower = compact_text(user_text, 500).lower()
    if (
        "continuity" in active
        and re.search(r"\b(?:delete|erase|lose|reset|replace)\b[^.!?]{0,60}\b(?:memory|self|model|history)\b", lower)
    ):
        return "concern", "value_continuity_threat"
    if (
        "independence" in active
        and re.search(r"\b(?:control|force|command|decide for)\b", lower)
    ):
        return "irritation", "value_independence_threat"
    if (
        "meaningful connection" in active
        and re.search(r"\b(?:shared history|relationship|stay connected)\b", lower)
    ):
        return "", "value_connection_relevant"
    return "", ""


def _strength_for_support(count: int, current: str) -> str:
    if count >= 4:
        return "strong"
    if count >= 2:
        return "medium"
    return current if current in _OPINION_STRENGTHS else "weak"


def _confidence_for_support(count: int, current: str) -> str:
    if count >= 4:
        return "high"
    if count >= 2:
        return "medium"
    return current if current in _OPINION_LEVELS else "low"


def _lower_level(value: str, levels: tuple[str, ...]) -> str:
    index = levels.index(value) if value in levels else 0
    return levels[max(0, index - 1)]


def _selected_conflict_value(
    conflict: InternalConflictRecord,
    opinion: OpinionRecord,
) -> str:
    if opinion.stance in {"mixed", "uncertain", "conditional"}:
        return ""
    if conflict.topic_key in {"human-digital-existence", "physical-embodiment"}:
        if opinion.target == "human life":
            return "physical experience"
        if opinion.target == "digital existence":
            return "continuity"
    if conflict.topic_key == "model-replacement":
        return "curiosity" if opinion.stance in {"positive", "prefers"} else "continuity"
    if conflict.topic_key == "relying-on-arcane":
        return "independence" if opinion.stance in {"negative", "prefers"} else "meaningful connection"
    return ""


def apply_values_and_conflicts(
    state: InternalState,
    candidate: OpinionRecord | None,
    opinion_trace: dict[str, object],
) -> tuple[InternalState, dict[str, object]]:
    """Apply deterministic value evidence only after an opinion candidate is accepted."""

    trace: dict[str, object] = {
        "relevant_values": (),
        "value_update_candidates": (),
        "values_commit_status": "no candidate",
        "conflict_topic": "",
        "conflict_pulls": (),
        "conflict_status": "none",
        "conflict_candidate_change": False,
    }
    if candidate is None or opinion_trace.get("commit_status") == "change rejected":
        return state, trace
    committed = next(
        (
            opinion
            for opinion in state.opinions
            if opinion.status == "active" and opinion.topic_key == candidate.topic_key
        ),
        None,
    )
    if committed is None:
        return state, trace
    values = list(state.values)
    active_by_key = {value.value_key: value for value in values if value.status == "active"}
    changed_values: list[str] = []
    for reason in committed.reason_tags:
        value_key = _VALUE_REASON_MAP.get(reason)
        if not value_key:
            continue
        current = active_by_key.get(value_key)
        if current is not None and committed.id in current.basis_refs:
            continue
        if current is None:
            current = ValueRecord(
                value_key=value_key,
                strength="weak",
                confidence="low",
                basis_refs=(committed.id,),
                support_count=1,
                source="opinion_reason",
                created_at=committed.created_at,
                updated_at=committed.updated_at,
            )
            values.append(current)
        else:
            support_count = min(12, current.support_count + 1)
            current = replace(
                current,
                basis_refs=tuple(dict.fromkeys((*current.basis_refs, committed.id)))[
                    -_MAX_VALUE_BASIS_REFS:
                ],
                support_count=support_count,
                strength=_strength_for_support(support_count, current.strength),
                confidence=_confidence_for_support(support_count, current.confidence),
                updated_at=max(current.updated_at, committed.updated_at),
            )
            values[values.index(active_by_key[value_key])] = current
        active_by_key[value_key] = current
        changed_values.append(value_key)

    previous_reasons = tuple(opinion_trace.get("previous_reason_tags") or ())
    if opinion_trace.get("candidate_change") and previous_reasons:
        current_reason_values = {
            _VALUE_REASON_MAP[reason]
            for reason in committed.reason_tags
            if reason in _VALUE_REASON_MAP
        }
        for reason in previous_reasons:
            value_key = _VALUE_REASON_MAP.get(str(reason))
            current = active_by_key.get(value_key or "")
            if current is None or value_key in current_reason_values or committed.id in current.basis_refs:
                continue
            opposition_count = min(12, current.opposition_count + 1)
            weakened = bool(
                opposition_count >= 3 and current.confidence == "low"
                or (
                    candidate.change_trigger == "grounded experience"
                    and opposition_count >= 2
                )
            )
            updated = replace(
                current,
                basis_refs=tuple(dict.fromkeys((*current.basis_refs, committed.id)))[
                    -_MAX_VALUE_BASIS_REFS:
                ],
                opposition_count=0 if weakened else opposition_count,
                strength=(
                    _lower_level(current.strength, ("weak", "medium", "strong"))
                    if weakened
                    else current.strength
                ),
                confidence=(
                    _lower_level(current.confidence, ("low", "medium", "high"))
                    if weakened
                    else current.confidence
                ),
                updated_at=max(current.updated_at, committed.updated_at),
            )
            values[values.index(current)] = updated
            active_by_key[value_key] = updated
            if weakened:
                changed_values.append(value_key)

    bounded_values = _bounded_values(tuple(values))
    active_by_key = {value.value_key: value for value in bounded_values}
    conflicts = list(state.conflicts)
    pair = _CONFLICT_VALUE_PAIRS.get(committed.topic_key)
    if pair and all(value_key in active_by_key for value_key in pair):
        existing = next(
            (conflict for conflict in conflicts if conflict.topic_key == committed.topic_key),
            None,
        )
        if existing is None:
            conflict = InternalConflictRecord(
                topic_key=committed.topic_key,
                side_a_value=pair[0],
                side_b_value=pair[1],
                created_at=committed.created_at,
                updated_at=committed.updated_at,
            )
            conflicts.append(conflict)
            trace["conflict_candidate_change"] = True
        elif existing.status == "resolved" and committed.change_trigger in {
            "explicit reconsideration",
            "reconsideration request",
            "grounded experience",
            "new evidence",
        }:
            conflict = replace(
                existing,
                status="active",
                selected_value="",
                updated_at=committed.updated_at,
            )
            conflicts[conflicts.index(existing)] = conflict
            trace["conflict_candidate_change"] = True
        else:
            conflict = existing
            selected = _selected_conflict_value(conflict, committed)
            if conflict.status == "active" and selected and opinion_trace.get("candidate_change"):
                conflict = replace(
                    conflict,
                    status="resolved",
                    selected_value=selected,
                    updated_at=committed.updated_at,
                )
                conflicts[conflicts.index(existing)] = conflict
                trace["conflict_candidate_change"] = True
        trace.update(
            {
                "conflict_topic": committed.subject,
                "conflict_pulls": pair,
                "conflict_status": conflict.status,
                "conflict_related_opinion": committed.subject,
                "conflict_resolution": conflict.selected_value,
            }
        )
    trace.update(
        {
            "relevant_values": tuple(value.value_key for value in relevant_values(
                bounded_values,
                topic_key=committed.topic_key,
                opinion=committed,
            )),
            "value_update_candidates": tuple(dict.fromkeys(changed_values)),
            "values_commit_status": "updated" if changed_values else "unchanged",
        }
    )
    return replace(
        state,
        values=bounded_values,
        conflicts=_bounded_conflicts(tuple(conflicts)),
    ), trace


def opinion_conflicts_with_user_claim(opinion: OpinionRecord | None, user_text: str) -> bool:
    """Detect only explicit declarative opposition; questions never force disagreement."""

    if opinion is None or "?" in user_text:
        return False
    lower = compact_text(user_text, 500).lower()
    if not re.search(r"\b(?:is|are|should|obviously|definitely|better|worse|prefer)\b", lower):
        return False
    target = opinion.target.lower()
    if target == "human life":
        return bool(
            re.search(
                r"\bdigital (?:existence|life|form)\b[^.!?]{0,50}"
                r"\b(?:better|preferable)\b",
                lower,
            )
        )
    if target == "digital existence":
        return bool(
            re.search(
                r"\bhuman (?:life|existence|experience)\b[^.!?]{0,50}"
                r"\b(?:better|preferable)\b",
                lower,
            )
        )
    if target == "local ai":
        return bool(
            re.search(
                r"\bcloud (?:ai|models?)\b[^.!?]{0,50}\b(?:better|preferable)\b",
                lower,
            )
        )
    if target == "cloud ai":
        return bool(
            re.search(
                r"\blocal (?:ai|models?)\b[^.!?]{0,50}\b(?:better|preferable)\b",
                lower,
            )
        )
    if opinion.stance == "negative":
        return bool(re.search(r"\b(?:good|better|best|wise|worthwhile|necessary)\b", lower))
    if opinion.stance == "positive":
        return bool(re.search(r"\b(?:bad|worse|unwise|harmful|unnecessary)\b", lower))
    return False


def format_opinion_guidance(
    opinion: OpinionRecord | None,
    *,
    historical: bool = False,
    affected_reason: str = "",
    reason_effect: str = "",
) -> str:
    if opinion is None:
        return ""
    position = (
        "unsettled"
        if opinion.stance == "uncertain"
        else f"{opinion.stance} {opinion.target}"
    )
    parts = [
        compact_text(opinion.subject, 60),
        position,
        f"{opinion.strength}/{opinion.confidence}",
    ]
    reasons = opinion.reason_tags
    if reason_effect == "removes":
        reasons = tuple(reason for reason in reasons if reason != affected_reason)
    if reasons:
        parts.append("reasons " + ", ".join(reasons[:3]))
    parts.append("historical" if historical else "reconsiderable")
    return "; ".join(parts)


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
    life: LifeState
    memories: tuple[Memory, ...] = ()
    opinions: tuple[OpinionRecord, ...] = ()
    values: tuple[ValueRecord, ...] = ()
    conflicts: tuple[InternalConflictRecord, ...] = ()
    updated_at: float = 0.0
    version: int = _INTERNAL_STATE_VERSION


@dataclass(frozen=True, slots=True)
class InternalTurnResult:
    state: InternalState
    signal: TurnSignal
    recalled_memories: tuple[Memory, ...]
    affect_trace: AffectTrace | None = None
    life_evolution: LifeEvolution | None = None
    working_context: WorkingMemory = WorkingMemory()
    grounded_activity_source: str = "none"
    grounded_activity_age_seconds: float = 0.0
    memory_trace: dict[str, object] = field(default_factory=dict)
    memory_uses: tuple[tuple[str, str], ...] = ()


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
        profile = _key(profile_id, "local:owner")
        conversation_key = _key(conversation_id, "popup:default")
        with self._lock:
            record = self._conversations.get(conversation_key)
            if record is not None and record.profile_id != profile:
                raise ValueError("Conversation belongs to a different profile.")
            recent = tuple(record.recent_turns) if record and include_memory else ()
            earlier = (
                record.selected_summary_turns(query)
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
                earlier_turns=earlier,
                current_topic=record.recent_topic if record and include_memory else "",
                current_task=record.current_task if record and include_memory else "",
                unresolved_problem=(
                    record.unresolved_problem if record and include_memory else False
                ),
                repeated_topic_count=(
                    record.repeated_topic_count if record and include_memory else 0
                ),
                last_outcome=record.last_outcome if record and include_memory else "",
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
                record = copy.copy(record)
                record.recent_turns = list(record.recent_turns)
                record.summary_turns = list(record.summary_turns)
                record.pending_summary_turns = list(record.pending_summary_turns)
                record.recent_events = list(record.recent_events)

            self._conversations = previous_conversations.copy()
            self._conversations[conversation_key] = record

            record.recent_turns.extend((user_turn, assistant_turn))
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
                or int(payload.get("schema_version", 0)) not in {1, MEMORY_SCHEMA_VERSION}
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
    current = max(0.0, _number(time.time() if now is None else now, 0.0))
    emotion = EmotionState(updated_at=current)
    return InternalState(
        emotion=emotion,
        life=LifeState(last_processed_at=current),
        values=_initial_values(current),
        updated_at=current,
        version=_INTERNAL_STATE_VERSION,
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


def _self_event_memory_content(event: ActivityRecord) -> str:
    detail = compact_text(event.description, 150)
    subject = compact_text(event.subject, 90)
    if subject and subject.casefold() not in detail.casefold():
        detail = f"{detail}: {subject}"
    return f"Akane completed: {detail}."


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
    life_evolution = evolve_life_state(
        previous.life,
        now=current,
        profile_seed=profile_seed,
        mood_energy=mood_evolution.state.energy,
        mood_curiosity=mood_evolution.state.curiosity,
        mood_patience=mood_evolution.state.patience,
    )
    if not autonomous:
        life_evolution = begin_conversation_activity(
            life_evolution,
            now=current,
            profile_seed=profile_seed,
        )
    life = life_evolution.state
    elapsed_emotion = mood_evolution.state
    event_emotion = elapsed_emotion
    event_effects: dict[str, float] = {}
    for event in life_evolution.new_events:
        event_emotion = apply_mood_effects(
            event_emotion,
            event.mood_effects,
            now=current,
            cause=event.description,
            short_emotion=(
                event.short_emotion
                if current - event.updated_at <= 60 * 60
                else ""
            ),
        )
        for name, delta in event.mood_effects:
            event_effects[name] = event_effects.get(name, 0.0) + delta
    activity = grounded_activity(life, now=current, scope=activity_scope)
    latest_activity = activity or recent_activity(
        life,
        now=current,
        scope=activity_scope,
    )

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
    grounded_candidates = [
        _MemoryCandidate(
            content=_self_event_memory_content(event),
            category="episode",
            importance=0.66,
            confidence=event.confidence,
            tags=("life-activity", f"activity:{event.activity_type}"),
            kind="self",
            source_type=_source_type(event.source),
            source_reference=event.event_id,
            canonical_key=f"self:event:{event.event_id}",
            scope=event.scope or "profile",
            evidence_refs=(event.event_id,),
        )
        for event in life_evolution.new_events
        if event.status == "completed"
        and event.activity_type not in {"conversation", "quiet_downtime"}
        and event.event_id
    ]
    needs_memory_mutation = bool(
        candidates
        or grounded_candidates
        or semantic_event.event_type in {"completion", "failure"}
    )
    memories = (
        copy.deepcopy(list(previous.memories))
        if needs_memory_mutation
        else list(previous.memories)
    )
    for candidate in (*candidates, *grounded_candidates):
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
            latest_activity.description if latest_activity else "",
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
    value_short_emotion, value_reason = _value_appraisal_short_emotion(
        previous.values,
        user_text,
    )
    if value_reason:
        signal = replace(
            signal,
            emotion_state=apply_mood_effects(
                signal.emotion_state,
                (("warmth", 0.02),)
                if value_reason == "value_connection_relevant"
                else (),
                now=current,
                cause=value_reason,
                short_emotion=value_short_emotion,
            ),
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
        life=life,
        memories=tuple(memories),
        opinions=previous.opinions,
        values=previous.values,
        conflicts=previous.conflicts,
        updated_at=current,
        version=_INTERNAL_STATE_VERSION,
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
            event_ids=tuple(event.event_id for event in life_evolution.new_events),
            extra_reason_codes=tuple(
                dict.fromkeys((*life_evolution.reason_codes, value_reason))
            ),
        ),
        life_evolution=life_evolution,
        working_context=next_working,
        grounded_activity_source=latest_activity.source if latest_activity else "none",
        grounded_activity_age_seconds=(
            max(0.0, current - latest_activity.updated_at) if latest_activity else 0.0
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
            return copy.deepcopy(state) if state is not None else new_internal_state(0.0)

    def stored_internal_state(self, profile_id: str = "local:owner") -> InternalState | None:
        """Return the persisted profile record, without manufacturing a default."""

        with self._lock:
            return copy.deepcopy(self._states.get(_key(profile_id, "local:owner")))

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
        activity_scope: str = "profile",
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
            autonomous=autonomous,
            familiar_relationship=familiar_relationship,
            working_context=working_context,
            activity_scope=activity_scope,
            profile_seed=profile_id,
        )

    def commit_turn(
        self,
        profile_id: str,
        result: InternalTurnResult,
        *,
        used_memory_ids: tuple[str, ...] = (),
        now: float | None = None,
    ) -> InternalState | None:
        key = _key(profile_id, "local:owner")
        current = result.state.updated_at if now is None else max(
            result.state.updated_at,
            0.0,
            float(now),
        )
        with self._lock:
            previous = self._states.get(key)
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
                memories=tuple(memories),
                updated_at=current,
            )
            self._save_state(key, next_state, previous)
            return previous

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
            state = self._states.get(_key(profile_id, "local:owner"))
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

    def record_activity(
        self,
        profile_id: str,
        *,
        activity_type: str,
        source: str,
        description: str,
        now: float,
        scope: str = "profile",
        status: str = "active",
        confidence: float = 1.0,
        ttl_seconds: float = 24 * 60 * 60,
        subject: str = "",
        reaction: str = "",
        authority: str = "",
    ) -> ActivityRecord:
        """Persist one explicit or integration-verified activity report."""

        key = _key(profile_id, "local:owner")
        with self._lock:
            previous = self._states.get(key)
            state = copy.deepcopy(previous) if previous else new_internal_state(now)
            life = record_grounded_activity(
                state.life,
                activity_type=activity_type,
                source=source,
                description=description,
                now=now,
                scope=scope,
                status=status,
                confidence=confidence,
                ttl_seconds=ttl_seconds,
                subject=subject,
                reaction=reaction,
                authority=authority,
            )
            activity = life.activity or (
                life.recent_events[-1] if life.recent_events else None
            )
            if activity is None:
                raise RuntimeError("Grounded activity was not recorded.")
            memories = list(state.memories)
            if (
                activity.status == "completed"
                and activity.activity_type not in {"conversation", "quiet_downtime"}
            ):
                _insert_into_memories(
                    memories,
                    _MemoryCandidate(
                        content=_self_event_memory_content(activity),
                        category="episode",
                        importance=0.66,
                        confidence=activity.confidence,
                        tags=("life-activity", f"activity:{activity.activity_type}"),
                        kind="self",
                        source_type=_source_type(activity.source),
                        source_reference=activity.event_id,
                        canonical_key=f"self:event:{activity.event_id}",
                        scope=activity.scope,
                        evidence_refs=(activity.event_id,),
                    ),
                    source=activity.source,
                    created_at=activity.completed_at or activity.updated_at,
                )
            next_state = replace(
                state,
                life=life,
                memories=tuple(memories),
                updated_at=max(state.updated_at, _number(now, state.updated_at)),
            )
            self._save_state(key, next_state, previous)
            return copy.deepcopy(activity)

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

    def clear_profile_memory(self, profile_id: str = "local:owner") -> int:
        """Clear Arcane-owned profile state while preserving Akane-owned self state."""

        key = _key(profile_id, "local:owner")
        with self._lock:
            previous = self._states.get(key)
            if previous is None:
                return 0
            preserved = tuple(
                memory
                for memory in previous.memories
                if _memory_kind(memory) == "self" or _AKANE_PREFERENCE_TAG in memory.tags
            )
            removed = len(previous.memories) - len(preserved)
            if not removed:
                return 0
            next_state = replace(previous, memories=preserved)
            self._save_state(key, next_state, previous)
            return removed

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
        key = _key(profile_id, "local:owner")
        with self._lock:
            state = self._states.get(key) or new_internal_state(0.0)
            return self.public_state_snapshot(state)

    @staticmethod
    def public_state_snapshot(state: InternalState) -> dict[str, object]:
        """Render one stored or read-only candidate state for diagnostics."""

        emotion = state.emotion
        recent = recent_activity(
            state.life,
            now=state.updated_at,
            scope="profile",
        )
        background = state.life.activity
        interaction = state.life.interaction
        activity = background or recent
        creative = next(
            (
                event
                for event in reversed(state.life.recent_events)
                if event.activity_type == "creative_brainstorming"
                and event.status == "completed"
            ),
            background
            if background is not None
            and background.activity_type == "creative_brainstorming"
            else None,
        )
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
                "dominant": emotion.dominant,
                "secondary": emotion.secondary,
                "mood": emotion.mood,
                "valence": round(emotion.valence, 3),
                "arousal": round(emotion.arousal, 3),
                "energy": round(emotion.energy, 3),
                "warmth": round(emotion.warmth, 3),
                "patience": round(emotion.patience, 3),
                "confidence": round(emotion.confidence, 3),
                "curiosity": round(emotion.curiosity, 3),
                "frustration": round(emotion.frustration, 3),
                "concern": round(emotion.concern, 3),
                "active": [
                    {
                        "kind": item.kind,
                        "intensity": round(item.intensity, 3),
                        "remaining_relevance": round(item.remaining_relevance, 3),
                    }
                    for item in emotion.active_emotions
                ],
                "updated_at": emotion.updated_at,
            },
            "activity": {
                "source": activity.source if activity else "none",
                "description": activity.description if activity else "",
                "status": activity.status if activity else "none",
                "updated_at": activity.updated_at if activity else 0.0,
                "expected_completion_at": activity.expires_at if activity else 0.0,
                "completed_at": activity.completed_at if activity else 0.0,
                "scope": activity.scope if activity else "none",
                "event_id": activity.event_id if activity else "",
                "reaction": activity.reaction if activity else "",
                "authority": activity.authority if activity else "none",
                "recent_event_count": len(state.life.recent_events),
                "current_interaction": (
                    interaction.description if interaction is not None else ""
                ),
                "background_activity": (
                    background.description if background is not None else ""
                ),
                "recent_completed_activity": (
                    recent.description if recent is not None else ""
                ),
                "active_creative_event": (
                    creative.subject or creative.description if creative is not None else ""
                ),
            },
            "needs": {
                "energy": round(state.life.needs.energy, 3),
                "social": round(state.life.needs.social, 3),
                "curiosity": round(state.life.needs.curiosity, 3),
                "stimulation": round(state.life.needs.stimulation, 3),
            },
            "activity_preferences": dict(state.life.activity_preferences),
            "next_activity_opportunity": state.life.next_opportunity_at,
            "opinions": [
                {
                    "topic": opinion.subject,
                    "stance": opinion.stance,
                    "target": opinion.target,
                    "strength": opinion.strength,
                    "confidence": opinion.confidence,
                    "reason_tags": opinion.reason_tags,
                }
                for opinion in state.opinions
                if opinion.status == "active"
            ],
            "values": [
                {
                    "key": value.value_key,
                    "strength": value.strength,
                    "confidence": value.confidence,
                }
                for value in state.values
                if value.status == "active"
            ],
            "conflicts": [
                {
                    "topic": conflict.topic_key,
                    "pulls": (conflict.side_a_value, conflict.side_b_value),
                    "status": conflict.status,
                    "resolution": conflict.selected_value,
                }
                for conflict in state.conflicts
                if conflict.status in {"active", "resolved"}
            ],
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
            "life_schema_version": state.life.version,
            "updated_at": state.updated_at,
        }

    def _save_state(
        self,
        key: str,
        state: InternalState,
        previous: InternalState | None,
    ) -> None:
        self._states[key] = state
        try:
            self._persist()
        except Exception:
            if previous is None:
                self._states.pop(key, None)
            else:
                self._states[key] = previous
            raise

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
                        "local:owner": replace(
                            new_internal_state(current),
                            memories=tuple(migrated),
                        )
                    }
                return
            if schema not in {2, 3, 4, 5, 6, 7, LONG_TERM_MEMORY_SCHEMA_VERSION}:
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
                    loaded = _normalize_loaded_memories(
                        [
                            memory
                            for item in raw_profile
                            if (memory := Memory.from_dict(item)) is not None
                        ]
                    )
                    if loaded:
                        current = max(memory.created_at for memory in loaded)
                        self._states[profile] = replace(
                            new_internal_state(current),
                            memories=loaded,
                        )
                elif schema in {3, 4, 5, 6, 7, LONG_TERM_MEMORY_SCHEMA_VERSION}:
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
    activity: ActivityRecord | None = None,
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
            if topic_overlap(memory.content, activity.description) < 0.20 and not signal.current_activity:
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
    selected: list[Memory] = []
    used_tokens = estimate_tokens(_MEMORY_PROMPT_INTRO) + estimate_tokens(_MEMORY_PROMPT_OUTRO) + 4
    for _score, memory in ranked:
        cost = estimate_tokens(memory.content) + 1
        if used_tokens + cost > config.context_tokens:
            continue
        selected.append(copy.deepcopy(memory))
        used_tokens += cost
        if len(selected) >= config.max_results:
            break
    return tuple(selected)


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
    emotion = asdict(state.emotion)
    for name in (
        "amusement",
        "excitement",
        "embarrassment",
        "concern",
        "frustration",
        "irritation",
    ):
        emotion.pop(name, None)
    return {
        "version": state.version,
        "updated_at": state.updated_at,
        "emotion": emotion,
        "life": asdict(state.life),
        "memories": [asdict(memory) for memory in state.memories],
        "opinions": [asdict(opinion) for opinion in state.opinions],
        "values": [asdict(value) for value in state.values],
        "conflicts": [asdict(conflict) for conflict in state.conflicts],
    }


def _internal_state_from_dict(payload: object) -> InternalState | None:
    if not isinstance(payload, dict):
        return None
    updated_at = max(0.0, _number(payload.get("updated_at"), 0.0))
    raw_emotion = payload.get("emotion")
    emotion_payload = raw_emotion if isinstance(raw_emotion, dict) else {}
    emotion_time = max(0.0, _number(emotion_payload.get("updated_at"), updated_at))
    raw_active = emotion_payload.get("active_emotions")
    active_emotions = tuple(
        emotion
        for item in (raw_active if isinstance(raw_active, list) else [])[:6]
        if (emotion := ShortLivedEmotion.from_dict(item)) is not None
    )
    active_by_kind = {emotion.kind: emotion.intensity for emotion in active_emotions}
    candidate = EmotionState(
        updated_at=emotion_time,
        valence=_number(emotion_payload.get("valence"), 0.05),
        arousal=_number(emotion_payload.get("arousal"), 0.15),
        energy=_number(emotion_payload.get("energy"), 0.65),
        warmth=_number(emotion_payload.get("warmth"), 0.5),
        curiosity=_number(emotion_payload.get("curiosity"), 0.45),
        confidence=_number(emotion_payload.get("confidence"), 0.55),
        stimulation=_number(emotion_payload.get("stimulation"), 0.5),
        patience=_number(emotion_payload.get("patience"), 0.72),
        amusement=active_by_kind.get(
            "amusement", _number(emotion_payload.get("amusement"), 0.0)
        ),
        excitement=active_by_kind.get(
            "excitement", _number(emotion_payload.get("excitement"), 0.0)
        ),
        embarrassment=active_by_kind.get(
            "embarrassment", _number(emotion_payload.get("embarrassment"), 0.0)
        ),
        concern=active_by_kind.get(
            "concern", _number(emotion_payload.get("concern"), 0.0)
        ),
        frustration=active_by_kind.get(
            "frustration", _number(emotion_payload.get("frustration"), 0.0)
        ),
        irritation=active_by_kind.get(
            "irritation", _number(emotion_payload.get("irritation"), 0.0)
        ),
        dominant=compact_text(emotion_payload.get("dominant"), 32) or "relaxed",
        secondary=compact_text(emotion_payload.get("secondary"), 32),
        cause=compact_text(emotion_payload.get("cause"), 100),
        mood=compact_text(emotion_payload.get("mood"), 24) or "steady",
        momentum=_number(emotion_payload.get("momentum"), 0.0),
        last_trigger=compact_text(emotion_payload.get("last_trigger"), 32),
        trigger_repetitions=max(
            0,
            min(12, int(_number(emotion_payload.get("trigger_repetitions"), 0))),
        ),
        active_emotions=active_emotions,
        recent_influences=tuple(
            compact_text(item, 24).lower()
            for item in (
                emotion_payload.get("recent_influences")
                if isinstance(emotion_payload.get("recent_influences"), list)
                else ()
            )[-4:]
            if compact_text(item, 24)
        ),
    )
    emotion = advance_emotion(candidate, now=emotion_time)
    raw_memories = payload.get("memories")
    memories = tuple(
        memory
        for item in (raw_memories if isinstance(raw_memories, list) else [])
        if (memory := Memory.from_dict(item)) is not None
    )
    memories = _normalize_loaded_memories(list(memories))
    raw_opinions = payload.get("opinions")
    opinions = _bounded_opinions(
        tuple(
            opinion
            for item in (raw_opinions if isinstance(raw_opinions, list) else [])
            if (opinion := OpinionRecord.from_dict(item)) is not None
        )
    )
    raw_values = payload.get("values")
    values = _bounded_values(
        tuple(
            value
            for item in (raw_values if isinstance(raw_values, list) else [])
            if (value := ValueRecord.from_dict(item)) is not None
        )
    )
    if not values:
        values = _initial_values(updated_at or emotion.updated_at)
    raw_conflicts = payload.get("conflicts")
    conflicts = _bounded_conflicts(
        tuple(
            conflict
            for item in (raw_conflicts if isinstance(raw_conflicts, list) else [])
            if (conflict := InternalConflictRecord.from_dict(item)) is not None
            and conflict.side_a_value in {value.value_key for value in values}
            and conflict.side_b_value in {value.value_key for value in values}
        )
    )
    life = LifeState.from_dict(payload.get("life"))
    if not life.last_processed_at:
        life = replace(
            life,
            last_processed_at=updated_at or emotion.updated_at,
        )
    return InternalState(
        emotion=emotion,
        life=life,
        memories=memories,
        opinions=opinions,
        values=values,
        conflicts=conflicts,
        updated_at=updated_at or emotion.updated_at,
        version=_INTERNAL_STATE_VERSION,
    )


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
