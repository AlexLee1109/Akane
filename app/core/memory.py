"""One state owner, conversation owner, migration path, and atomic commit path."""

from __future__ import annotations

import math
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from app.core.config import (
    CONVERSATION_STALE_DAYS,
    LONG_TERM_MEMORY_PATH,
    MAX_CONVERSATIONS,
    MEMORY_MAX_ENTRIES_PER_PROFILE,
    MEMORY_MAX_RESULTS,
    MEMORY_PATH,
    POPUP_USER_PATH,
)
from app.core.persistence import atomic_write_json, read_json
from app.core.presence import (
    CLAIM_SECONDS,
    EMOTION_UPDATE_FIELDS,
    MOOD_UPDATE_FIELDS,
    RETRY_SECONDS,
    LifeDecision,
    PresenceActivity,
    PresenceState,
    apply_life_decision,
    life_decision_rejection,
    normalize_presence,
    validate_interest_addition,
)
from app.core.time_context import build_time_context
from app.core.utils import OWNER_PROFILE_ID, canonical_profile_id, compact_text, words

STATE_SCHEMA_VERSION = 13
STARTING_INTERESTS = ("anime", "manga", "VTubers")

_MAX_RECENT_TURNS = 28
_MAX_RELATIONSHIP_ENTRIES = 16
_MAX_PREFERENCES = 32
_MAX_OPINIONS = 32
_MAX_INTERESTS = 32
_MAX_RECENT_INITIATIVES = 16
_INITIATIVE_CLAIM_SECONDS = 60.0
ORDINARY_INITIATIVE_COOLDOWN_SECONDS = 4.0 * 3600.0
MAX_ORDINARY_INITIATIVES_PER_LOCAL_DAY = 2
_MEMORY_SUBJECTS = {"user", "akane", "shared"}
_MEMORY_KINDS = {"fact", "event", "commitment", "project", "concern"}
_PREFERENCE_STANCES = {
    "likes", "dislikes", "curious", "mixed", "uncertain", "indifferent",
}
EMOTION_VOCABULARY = frozenset(
    {
        "neutral",
        "calm",
        "content",
        "curious",
        "interested",
        "amused",
        "excited",
        "inspired",
        "affectionate",
        "hopeful",
        "uncertain",
        "concerned",
        "anxious",
        "lonely",
        "tired",
        "disappointed",
        "sad",
        "frustrated",
        "irritated",
        "angry",
    }
)
_EMOTION_SOURCES = {
    "conversation",
    "offscreen_life",
    "memory",
    "relationship",
    "self_reflection",
}
_GENERIC_CAUSES = {
    "her systems are active",
    "her system is active",
    "her internal state changed",
    "she is processing things",
    "she felt something randomly",
    "the model decided",
    "time passed",
}
_UNGROUNDED_CAUSE_LANGUAGE = re.compile(
    r"\b(?:processing|systems?|models?|random(?:ly)?|time (?:passing|passed))\b"
)
_TRIVIAL_EMOTIONAL_INPUTS = {
    "hello", "hi", "hey", "okay", "ok", "thanks", "thank you", "hmm", "hm",
}
_GROUNDING_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "she", "that", "the", "their", "them", "they", "this", "to", "was",
    "we", "were", "with", "you", "your", "user", "arcane", "akane",
}
_TRIVIAL_MEMORY = {
    "hello", "hi", "hey", "good morning", "good night", "thanks",
    "thank you", "how are you", "nice to meet you",
}
_TRIVIAL_MEMORY_TERMS = {
    "greet", "greeted", "greeting", "hello", "hey", "hi", "thank", "thanks",
}
_UNCERTAINTY = re.compile(
    r"\b(?:maybe|might|possibly|probably|not sure|i think|i guess|seems?)\b",
    re.IGNORECASE,
)
_TRANSIENT_STATEMENT = re.compile(
    r"\b(?:am|is|are|i'm|she's|he's|they're|we're|you're)\s+"
    r"(?:currently\s+)?[a-z]+ing\b"
    r"|\b(?:currently|right now)\b",
    re.IGNORECASE,
)
_MEMORY_SLOT_PATTERNS = (
    ("residence", re.compile(r"\b(?:live|lives|reside|resides|based)\b.*\b(?:in|at)\b")),
    ("employment-role", re.compile(r"\b(?:work|works|employed)\b.*\bas\b")),
    ("employment-place", re.compile(r"\b(?:work|works|employed)\b.*\b(?:at|for)\b")),
    ("name", re.compile(r"\b(?:name is|called)\b")),
    ("age", re.compile(r"\b(?:age is|years old)\b")),
)


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _terms(value: object) -> set[str]:
    return {
        term
        for term in words(value)
        if term not in _GROUNDING_STOPWORDS and len(term) > 1
    }


def _grounded(candidate: str, evidence: str) -> bool:
    candidate_terms = _terms(candidate)
    evidence_terms = _terms(evidence)
    if not candidate_terms or not evidence_terms:
        return False
    overlap = candidate_terms & evidence_terms
    coverage = len(overlap) / len(candidate_terms)
    return bool(overlap) and coverage >= 0.5 and (
        len(overlap) >= 2 or len(candidate_terms) <= 2
    )


def _similar(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    lexical = len(left_terms & right_terms) / max(1, len(left_terms | right_terms))
    return max(lexical, SequenceMatcher(None, _key(left), _key(right)).ratio())


def _duplicate_text(left: str, right: str) -> bool:
    if _key(left) == _key(right):
        return True
    left_terms = _terms(left)
    right_terms = _terms(right)
    shared = left_terms & right_terms
    jaccard = len(shared) / max(1, len(left_terms | right_terms))
    containment = len(shared) / max(1, min(len(left_terms), len(right_terms)))
    sequence = SequenceMatcher(None, _key(left), _key(right)).ratio()
    return sequence >= 0.92 or (jaccard >= 0.72 and containment >= 0.80)


def _memory_slot(memory: "Memory") -> str:
    if memory.kind != "fact":
        return ""
    text = _key(memory.text)
    for name, pattern in _MEMORY_SLOT_PATTERNS:
        if pattern.search(text):
            return f"{memory.subject}:{name}"
    return ""


def _dedupe_text(values: tuple[str, ...], *, limit: int) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = compact_text(value, 160)
        signature = _key(text)
        if text and signature and signature not in seen:
            result.append(text)
            seen.add(signature)
    return tuple(result[-limit:])


def _legacy_activity_payload(
    payload: object,
    *,
    fallback_end: float = 0.0,
) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    started = max(0.0, _number(payload.get("started_at")))
    expected = max(
        0.0,
        _number(
            payload.get("expected_end_at"),
            _number(payload.get("ends_at"), fallback_end),
        ),
    )
    migrated = {
        "activity": payload.get("activity"),
        "category": payload.get("category"),
        "subject": payload.get("subject", payload.get("title")),
        "detail": payload.get("detail"),
        "started_at": started,
        "expected_end_at": expected,
        "source": payload.get("source") or "autonomous_life",
    }
    return (
        migrated
        if PresenceActivity.from_dict(migrated) is not None
        else None
    )


def _legacy_presence_payload(payload: object) -> dict[str, object]:
    values = payload if isinstance(payload, dict) else {}
    next_at = max(
        0.0,
        _number(
            values.get("next_decision_at"),
            _number(values.get("life_next_run_at")),
        ),
    )
    current = _legacy_activity_payload(
        values.get("current_activity"),
        fallback_end=next_at,
    )
    recent = values.get("recent_activities")
    historical = [
        activity
        for item in (
            values.get("previous_activity"),
            *(recent if isinstance(recent, (list, tuple)) else ()),
        )
        if (activity := _legacy_activity_payload(item)) is not None
    ]
    historical.sort(
        key=lambda item: (
            _number(item.get("expected_end_at")),
            _number(item.get("started_at")),
        ),
        reverse=True,
    )
    previous = next(
        (activity for activity in historical if activity != current),
        None,
    )
    return {
        "current_activity": current,
        "previous_activity": previous,
        "last_decision_at": max(
            0.0,
            _number(
                values.get("last_decision_at"),
                _number(values.get("life_last_run_at")),
            ),
        ),
        "next_decision_at": next_at,
        "claim_token": values.get("claim_token"),
        "claim_expires_at": values.get("claim_expires_at"),
        "retry_at": values.get("retry_at"),
        "last_error": values.get("last_error"),
        "activity_pattern": values.get("activity_pattern"),
    }


def _legacy_memory_payload(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    status = compact_text(payload.get("status"), 24).casefold()
    thread_status = compact_text(payload.get("thread_status"), 24).casefold()
    if status and status not in {"active", "confirmed"}:
        return None
    if thread_status in {"resolved", "abandoned", "expired"}:
        return None
    old_kind = compact_text(
        payload.get("kind") or payload.get("category"),
        32,
    ).casefold()
    kind = {
        "episode": "event",
        "task_outcome": "event",
        "working": "project",
        "open_thread": "concern",
        "unfinished_topic": "concern",
        "profile": "fact",
        "self": "fact",
        "relationship": "event",
        "correction": "fact",
        "stable_fact": "fact",
        "user_fact": "fact",
    }.get(old_kind, old_kind)
    source_type = compact_text(payload.get("source_type"), 40).casefold()
    source = (
        source_type
        if source_type not in {"", "unknown"}
        else compact_text(payload.get("source"), 40).casefold()
    )
    subject = compact_text(payload.get("subject"), 16).casefold()
    if subject not in _MEMORY_SUBJECTS:
        if old_kind == "self":
            subject = "akane"
        elif old_kind == "relationship":
            subject = "shared"
        elif source in {"user", "explicit_user", "verified_interface"}:
            subject = "user"
        else:
            subject = "shared"
    return {
        "id": payload.get("id"),
        "subject": subject,
        "kind": kind,
        "text": payload.get("text") or payload.get("content") or payload.get("summary"),
        "confidence": payload.get(
            "confidence",
            (
                0.8
                if status == "confirmed"
                or source in {"user", "explicit_user", "verified_interface"}
                else 0.65
            ),
        ),
        "created_at": payload.get("created_at", payload.get("updated_at", 0.0)),
        "updated_at": payload.get("updated_at", payload.get("created_at", 0.0)),
        "source_type": payload.get("source_type"),
    }


def _legacy_profile_payload(payload: object) -> object:
    if isinstance(payload, list):
        return [
            memory
            for item in payload
            if (memory := _legacy_memory_payload(item)) is not None
        ]
    if not isinstance(payload, dict):
        return payload
    migrated = dict(payload)
    migrated["presence"] = _legacy_presence_payload(payload.get("presence"))
    raw_memories = payload.get("memories")
    if isinstance(raw_memories, list):
        migrated["memories"] = [
            memory
            for item in raw_memories
            if (memory := _legacy_memory_payload(item)) is not None
        ]
    return migrated


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
        role = str(payload.get("role") or "").strip().lower()
        content = str(payload.get("content") or payload.get("text") or "").strip()
        if role not in {"user", "assistant"} or not content:
            return None
        return cls(
            compact_text(payload.get("turn_id"), 100) or uuid.uuid4().hex,
            role,
            content[:8_000],
            max(0.0, _number(payload.get("timestamp"))),
            compact_text(payload.get("source"), 32) or "unknown",
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def as_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def _complete_turns(turns: tuple[ChatTurn, ...]) -> tuple[ChatTurn, ...]:
    complete: list[ChatTurn] = []
    index = 0
    while index < len(turns):
        turn = turns[index]
        if turn.role == "assistant" and turn.source == "initiative":
            complete.append(turn)
            index += 1
            continue
        if (
            index + 1 < len(turns)
            and turn.role == "user"
            and turns[index + 1].role == "assistant"
        ):
            complete.extend((turn, turns[index + 1]))
            index += 2
        else:
            index += 1
    return tuple(complete)


@dataclass(frozen=True, slots=True)
class InitiativeOpportunity:
    opportunity_id: str
    reason: str
    source_type: str
    source_id: str
    context: str
    topic_key: str
    created_at: float
    not_before: float
    expires_at: float
    status: str = "pending"
    claim_token: str | None = None
    claim_expires_at: float = 0.0
    message: str | None = None
    evaluated_at: float = 0.0
    generated_at: float = 0.0
    delivery_channel: str | None = None
    delivered_at: float = 0.0
    delivery_message_id: str | None = None
    failed_channels: tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        now: float,
    ) -> "InitiativeOpportunity | None":
        if not isinstance(payload, dict):
            return None
        opportunity_id = compact_text(payload.get("opportunity_id"), 100)
        reason = compact_text(payload.get("reason"), 100)
        source_type = compact_text(payload.get("source_type"), 40).casefold()
        source_id = compact_text(payload.get("source_id"), 160)
        context = compact_text(payload.get("context"), 420)
        topic_key = _key(payload.get("topic_key"))[:120]
        created = max(0.0, _number(payload.get("created_at")))
        not_before = max(created, _number(payload.get("not_before"), created))
        expires = max(0.0, _number(payload.get("expires_at")))
        status = compact_text(payload.get("status"), 24).casefold()
        if (
            not opportunity_id
            or not reason
            or not source_type
            or not source_id
            or not context
            or not topic_key
            or expires <= created
            or status
            not in {"pending", "pending_delivery", "sent", "dismissed", "expired"}
        ):
            return None
        claim_token = compact_text(payload.get("claim_token"), 100) or None
        claim_expires = max(0.0, _number(payload.get("claim_expires_at")))
        delivery_channel = compact_text(
            payload.get("delivery_channel"),
            16,
        ).casefold() or None
        failed_channels = tuple(
            channel
            for item in (payload.get("failed_channels") or ())
            if (channel := compact_text(item, 16).casefold())
            in {"popup", "discord"}
        )
        if claim_token is None or claim_expires <= now:
            if status == "pending_delivery" and delivery_channel:
                failed_channels = tuple(
                    dict.fromkeys((*failed_channels, delivery_channel))
                )
            claim_token = None
            claim_expires = 0.0
            if status != "sent":
                delivery_channel = None
        return cls(
            opportunity_id,
            reason,
            source_type,
            source_id,
            context,
            topic_key,
            created,
            not_before,
            expires,
            status,
            claim_token,
            claim_expires,
            compact_text(payload.get("message"), 500) or None,
            max(0.0, _number(payload.get("evaluated_at"))),
            max(0.0, _number(payload.get("generated_at"))),
            delivery_channel,
            max(0.0, _number(payload.get("delivered_at"))),
            compact_text(payload.get("delivery_message_id"), 160) or None,
            failed_channels,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "failed_channels": list(self.failed_channels),
        }


@dataclass(frozen=True, slots=True)
class SentInitiative:
    topic_key: str
    source_id: str
    source_type: str
    delivered_at: float

    @classmethod
    def from_dict(cls, payload: object) -> "SentInitiative | None":
        if not isinstance(payload, dict):
            return None
        topic = _key(payload.get("topic_key"))[:120]
        source = compact_text(payload.get("source_id"), 160)
        source_type = compact_text(payload.get("source_type"), 40).casefold()
        delivered = max(0.0, _number(payload.get("delivered_at")))
        return (
            cls(topic, source, source_type, delivered)
            if topic and source and source_type and delivered
            else None
        )


@dataclass(frozen=True, slots=True)
class InitiativeState:
    current: InitiativeOpportunity | None = None
    cooldown_until: float = 0.0
    recent: tuple[SentInitiative, ...] = ()
    handled_source_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: object, *, now: float) -> "InitiativeState":
        values = payload if isinstance(payload, dict) else {}
        recent = tuple(
            item
            for raw in (values.get("recent") or ())
            if (item := SentInitiative.from_dict(raw)) is not None
        )[-_MAX_RECENT_INITIATIVES:]
        return cls(
            InitiativeOpportunity.from_dict(values.get("current"), now=now),
            max(0.0, _number(values.get("cooldown_until"))),
            recent,
            tuple(
                source
                for raw in (values.get("handled_source_ids") or ())
                if (source := compact_text(raw, 160))
            )[-_MAX_RECENT_INITIATIVES:],
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "current": self.current.as_dict() if self.current else None,
            "cooldown_until": self.cooldown_until,
            "recent": [asdict(item) for item in self.recent],
            "handled_source_ids": list(self.handled_source_ids),
        }


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    conversation_id: str
    profile_id: str
    recent_turns: tuple[ChatTurn, ...] = ()
    last_user_at: float = 0.0
    last_assistant_at: float = 0.0
    updated_at: float = 0.0
    committed_request_ids: tuple[str, ...] = ()
    request_replies: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_dict(
        cls,
        key: str,
        payload: object,
    ) -> "ConversationRecord | None":
        if not isinstance(payload, dict):
            return None
        conversation_id = compact_text(
            payload.get("conversation_id") or key,
            160,
        )
        if not conversation_id:
            return None
        profile_id = canonical_profile_id(payload.get("profile_id"))
        raw_turns = payload.get("recent_turns")
        turns = tuple(
            turn
            for item in (raw_turns if isinstance(raw_turns, list) else [])
            if (turn := ChatTurn.from_dict(item)) is not None
        )
        turns = _trim_turns(turns)
        ids = tuple(
            value
            for item in (payload.get("committed_request_ids") or [])
            if (value := compact_text(item, 180))
        )[-32:]
        raw_replies = payload.get("request_replies")
        replies: list[tuple[str, str]] = []
        if isinstance(raw_replies, dict):
            replies = [
                (compact_text(key, 180), str(value))
                for key, value in raw_replies.items()
                if compact_text(key, 180) and isinstance(value, str)
            ]
        elif isinstance(raw_replies, list):
            for item in raw_replies:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and compact_text(item[0], 180)
                    and isinstance(item[1], str)
                ):
                    replies.append((compact_text(item[0], 180), item[1]))
        reply_map = dict(replies)
        for request_id in ids:
            if request_id in reply_map:
                continue
            for index, turn in enumerate(turns[:-1]):
                if (
                    turn.role == "user"
                    and turn.turn_id in {request_id, f"{request_id}:user"}
                    and turns[index + 1].role == "assistant"
                ):
                    reply_map[request_id] = turns[index + 1].content
                    break
        replies = list(reply_map.items())
        last_user = max(
            (
                max(0.0, _number(payload.get("last_user_at"))),
                *(turn.timestamp for turn in turns if turn.role == "user"),
            )
        )
        last_assistant = max(
            (
                max(0.0, _number(payload.get("last_assistant_at"))),
                *(turn.timestamp for turn in turns if turn.role == "assistant"),
            )
        )
        return cls(
            conversation_id=conversation_id,
            profile_id=profile_id,
            recent_turns=turns,
            last_user_at=last_user,
            last_assistant_at=last_assistant,
            updated_at=max(
                last_user,
                last_assistant,
                max(0.0, _number(payload.get("updated_at"))),
            ),
            committed_request_ids=ids,
            request_replies=tuple(replies[-32:]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "conversation_id": self.conversation_id,
            "profile_id": self.profile_id,
            "recent_turns": [turn.as_dict() for turn in self.recent_turns],
            "last_user_at": self.last_user_at,
            "last_assistant_at": self.last_assistant_at,
            "updated_at": self.updated_at,
            "committed_request_ids": list(self.committed_request_ids),
            "request_replies": dict(self.request_replies),
        }


def _trim_turns(turns: tuple[ChatTurn, ...]) -> tuple[ChatTurn, ...]:
    kept = list(turns[-_MAX_RECENT_TURNS:])
    while kept and kept[0].role == "assistant" and kept[0].source != "initiative":
        kept.pop(0)
    return tuple(kept)


@dataclass(frozen=True, slots=True)
class MoodState:
    valence: float = 0.0
    energy: float = 0.0
    cause: str = ""
    updated_at: float = 0.0

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        now: float,
    ) -> "MoodState":
        values = payload if isinstance(payload, dict) else {}
        updated = min(now, max(0.0, _number(values.get("updated_at"))))
        return cls(
            max(-1.0, min(1.0, _number(values.get("valence")))),
            max(-1.0, min(1.0, _number(values.get("energy")))),
            compact_text(values.get("cause"), 160),
            updated,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EmotionState:
    primary: str = "neutral"
    intensity: float = 0.0
    cause: str = ""
    source: str | None = None
    source_id: str | None = None
    started_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_dict(
        cls,
        payload: object,
        fallback_time: float = 0.0,
        *,
        now: float,
        migrating: bool,
    ) -> "EmotionState":
        if not isinstance(payload, dict) or not set(payload) & {
            "primary",
            "dominant",
            "mood",
            "intensity",
            "irritation",
            "frustration",
            "concern",
            "cause",
            "updated_at",
        }:
            return cls()
        values = payload
        primary = compact_text(
            values.get("primary") or values.get("dominant") or values.get("mood"),
            32,
        ).casefold()
        if primary not in EMOTION_VOCABULARY:
            primary = "neutral"
        intensity = max(
            0.0,
            min(
                1.0,
                _number(
                    values.get("intensity"),
                    max(
                        _number(values.get("irritation")),
                        _number(values.get("frustration")),
                        _number(values.get("concern")),
                    ),
                ),
            ),
        )
        cause = compact_text(values.get("cause"), 160)
        updated = min(
            now,
            max(0.0, _number(values.get("updated_at"), fallback_time)),
        )
        source = compact_text(values.get("source"), 32).casefold() or None
        if source not in _EMOTION_SOURCES:
            source = "self_reflection" if migrating and primary != "neutral" else None
        source_id = compact_text(values.get("source_id"), 120) or None
        started = min(
            now,
            max(
                0.0,
                _number(values.get("started_at"), updated if migrating else 0.0),
            ),
        )
        if primary == "neutral" or intensity < 0.08 or not cause:
            primary = "neutral"
            intensity = 0.0
            cause = ""
            source = None
            source_id = None
            started = 0.0
        return cls(
            primary,
            intensity,
            cause,
            source,
            source_id,
            started,
            updated,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def effective_mood(mood: MoodState, *, now: float) -> MoodState:
    if mood.updated_at <= 0.0:
        return mood
    elapsed_hours = max(0.0, (now - mood.updated_at) / 3600.0)
    decay = 0.985 ** elapsed_hours
    return replace(
        mood,
        valence=max(-1.0, min(1.0, mood.valence * decay)),
        energy=max(-1.0, min(1.0, mood.energy * decay)),
    )


def effective_emotion(emotion: EmotionState, *, now: float) -> EmotionState:
    if emotion.updated_at <= 0.0 or emotion.primary == "neutral":
        return emotion
    elapsed_hours = max(0.0, (now - emotion.updated_at) / 3600.0)
    intensity = max(0.0, min(1.0, emotion.intensity * (0.88 ** elapsed_hours)))
    if intensity < 0.08:
        return EmotionState(updated_at=emotion.updated_at)
    return replace(emotion, intensity=intensity)


@dataclass(frozen=True, slots=True)
class Memory:
    id: str
    subject: str
    kind: str
    text: str
    confidence: float
    created_at: float
    updated_at: float

    @property
    def content(self) -> str:
        return (
            self.text
            if self.confidence >= 0.75
            else f"Uncertain memory: {self.text}"
        )

    @classmethod
    def from_dict(cls, payload: object) -> "Memory | None":
        if not isinstance(payload, dict):
            return None
        source_type = compact_text(payload.get("source_type"), 40).casefold()
        if source_type in {"generated_assistant", "speculative_inference"}:
            return None
        text = compact_text(
            payload.get("text") or payload.get("content") or payload.get("summary"),
            360,
        )
        if not text:
            return None
        subject = compact_text(payload.get("subject"), 16).casefold()
        if subject not in _MEMORY_SUBJECTS:
            actor = compact_text(payload.get("actor"), 24).casefold()
            subject = "user" if actor in {"arcane", "user"} else (
                "akane" if actor == "akane" else "shared"
            )
        kind = compact_text(
            payload.get("kind") or payload.get("category"),
            32,
        ).casefold()
        aliases = {
            "stable_fact": "fact",
            "user_fact": "fact",
            "episode": "event",
            "working": "project",
            "open_thread": "concern",
            "unfinished_topic": "concern",
        }
        kind = aliases.get(kind, kind)
        if kind not in _MEMORY_KINDS:
            kind = "fact"
        created = max(
            0.0,
            _number(payload.get("created_at"), _number(payload.get("updated_at"))),
        )
        updated = max(created, _number(payload.get("updated_at"), created))
        return cls(
            compact_text(payload.get("id"), 100) or uuid.uuid4().hex,
            subject,
            kind,
            text,
            max(0.0, min(1.0, _number(payload.get("confidence"), 0.7))),
            created,
            updated,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AkanePreference:
    topic: str
    stance: str
    reason: str
    updated_at: float

    @property
    def content(self) -> str:
        return f"{self.topic}: {self.stance} — {self.reason}"

    @classmethod
    def from_dict(cls, payload: object) -> "AkanePreference | None":
        if not isinstance(payload, dict):
            return None
        topic = compact_text(payload.get("topic"), 140)
        stance = compact_text(payload.get("stance"), 24).casefold()
        reason = compact_text(payload.get("reason"), 240)
        if not topic or stance not in _PREFERENCE_STANCES or not reason:
            return None
        return cls(
            topic,
            stance,
            reason,
            max(0.0, _number(payload.get("updated_at"))),
        )


@dataclass(frozen=True, slots=True)
class Opinion:
    topic: str
    position: str
    reason: str
    updated_at: float

    @property
    def content(self) -> str:
        return f"{self.topic}: {self.position} — {self.reason}"

    @classmethod
    def from_dict(cls, payload: object) -> "Opinion | None":
        if not isinstance(payload, dict):
            return None
        topic = compact_text(payload.get("topic"), 140)
        position = compact_text(payload.get("position"), 200)
        reason = compact_text(payload.get("reason"), 240)
        if not topic or not position or not reason:
            return None
        return cls(
            topic,
            position,
            reason,
            max(0.0, _number(payload.get("updated_at"))),
        )


@dataclass(frozen=True, slots=True)
class RelationshipEntry:
    summary: str
    confidence: float
    updated_at: float

    @property
    def content(self) -> str:
        return self.summary

    @classmethod
    def from_dict(cls, payload: object) -> "RelationshipEntry | None":
        if isinstance(payload, str):
            summary = compact_text(payload, 300)
            return cls(summary, 0.7, 0.0) if summary else None
        if not isinstance(payload, dict):
            return None
        summary = compact_text(
            payload.get("summary") or payload.get("content"),
            300,
        )
        if not summary:
            return None
        return cls(
            summary,
            max(0.0, min(1.0, _number(payload.get("confidence"), 0.7))),
            max(0.0, _number(payload.get("updated_at"))),
        )


@dataclass(frozen=True, slots=True)
class RelationshipState:
    patterns: tuple[RelationshipEntry, ...] = ()
    shared_context: tuple[RelationshipEntry, ...] = ()
    unresolved_events: tuple[RelationshipEntry, ...] = ()

    @classmethod
    def from_dict(cls, payload: object) -> "RelationshipState":
        values = payload if isinstance(payload, dict) else {}

        def entries(name: str) -> tuple[RelationshipEntry, ...]:
            raw = values.get(name)
            if not isinstance(raw, (list, tuple)):
                return ()
            return tuple(
                entry
                for item in raw
                if not (
                    name == "unresolved_events"
                    and isinstance(item, dict)
                    and compact_text(item.get("status"), 24).casefold()
                    == "resolved"
                )
                if (entry := RelationshipEntry.from_dict(item)) is not None
            )[-_MAX_RELATIONSHIP_ENTRIES:]

        return cls(entries("patterns"), entries("shared_context"), entries("unresolved_events"))

    def as_dict(self) -> dict[str, object]:
        def payload(values: tuple[RelationshipEntry, ...]) -> list[dict[str, object]]:
            return [asdict(item) for item in values]

        return {
            "patterns": payload(self.patterns),
            "shared_context": payload(self.shared_context),
            "unresolved_events": payload(self.unresolved_events),
        }


@dataclass(frozen=True, slots=True)
class ProfileState:
    mood: MoodState = MoodState()
    emotion: EmotionState = EmotionState()
    presence: PresenceState = PresenceState()
    memories: tuple[Memory, ...] = ()
    interests: tuple[str, ...] = STARTING_INTERESTS
    preferences: tuple[AkanePreference, ...] = ()
    opinions: tuple[Opinion, ...] = ()
    relationship: RelationshipState = RelationshipState()
    initiative: InitiativeState = InitiativeState()
    updated_at: float = 0.0

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        now: float,
        repair_presence: bool = False,
        migrating: bool = False,
    ) -> "ProfileState | None":
        if isinstance(payload, list):
            memories = tuple(
                memory
                for item in payload
                if (memory := Memory.from_dict(item)) is not None
            )
            return cls(
                presence=PresenceState.from_dict(
                    {},
                    now=now,
                    repair_schedule=repair_presence,
                ),
                memories=_merge_memories((), memories),
            )
        if not isinstance(payload, dict):
            return None
        updated = max(0.0, _number(payload.get("updated_at")))
        raw_memories = payload.get("memories")
        memories = tuple(
            memory
            for item in (raw_memories if isinstance(raw_memories, list) else [])
            if (memory := Memory.from_dict(item)) is not None
        )
        interests = _dedupe_text(
            (
                *STARTING_INTERESTS,
                *(
                    tuple(payload.get("interests"))
                    if isinstance(payload.get("interests"), (list, tuple))
                    else ()
                ),
            ),
            limit=_MAX_INTERESTS,
        )
        raw_preferences = payload.get("preferences")
        preferences = tuple(
            preference
            for item in (
                raw_preferences if isinstance(raw_preferences, (list, tuple)) else ()
            )
            if (preference := AkanePreference.from_dict(item)) is not None
        )
        raw_opinions = payload.get("opinions")
        opinions = tuple(
            opinion
            for item in (raw_opinions if isinstance(raw_opinions, (list, tuple)) else ())
            if (opinion := Opinion.from_dict(item)) is not None
        )
        return cls(
            mood=MoodState.from_dict(payload.get("mood"), now=now),
            emotion=EmotionState.from_dict(
                payload.get("emotion"),
                updated,
                now=now,
                migrating=migrating,
            ),
            presence=PresenceState.from_dict(
                payload.get("presence"),
                now=now,
                repair_schedule=repair_presence,
            ),
            memories=_merge_memories((), memories),
            interests=interests,
            preferences=_merge_preferences((), preferences),
            opinions=_merge_opinions((), opinions),
            relationship=RelationshipState.from_dict(payload.get("relationship")),
            initiative=InitiativeState.from_dict(
                payload.get("initiative"),
                now=now,
            ),
            updated_at=updated,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "updated_at": self.updated_at,
            "mood": self.mood.as_dict(),
            "emotion": self.emotion.as_dict(),
            "presence": self.presence.as_dict(),
            "memories": [memory.as_dict() for memory in self.memories],
            "interests": list(self.interests),
            "preferences": [asdict(item) for item in self.preferences],
            "opinions": [asdict(item) for item in self.opinions],
            "relationship": self.relationship.as_dict(),
            "initiative": self.initiative.as_dict(),
        }


def _new_profile(now: float) -> ProfileState:
    return ProfileState(
        presence=PresenceState.from_dict({}, now=now),
        updated_at=0.0,
    )


def _handled_initiative(
    state: InitiativeState,
    opportunity: InitiativeOpportunity,
) -> InitiativeState:
    return replace(
        state,
        handled_source_ids=tuple(
            dict.fromkeys((*state.handled_source_ids, opportunity.source_id))
        )[-_MAX_RECENT_INITIATIVES:],
    )


def _settle_initiative(state: InitiativeState, *, now: float) -> InitiativeState:
    opportunity = state.current
    if opportunity is None:
        return state
    if (
        opportunity.status in {"pending", "pending_delivery"}
        and opportunity.expires_at <= now
    ):
        expired = replace(
            opportunity,
            status="expired",
            evaluated_at=max(opportunity.evaluated_at, now),
            claim_token=None,
            claim_expires_at=0.0,
            delivery_channel=None,
        )
        return _handled_initiative(replace(state, current=expired), expired)
    if opportunity.claim_token and opportunity.claim_expires_at <= now:
        failed = opportunity.failed_channels
        if opportunity.status == "pending_delivery" and opportunity.delivery_channel:
            failed = tuple(
                dict.fromkeys((*failed, opportunity.delivery_channel))
            )
        return replace(
            state,
            current=replace(
                opportunity,
                claim_token=None,
                claim_expires_at=0.0,
                delivery_channel=None,
                failed_channels=failed,
            ),
        )
    return state


def _initiative_source_exists(
    profile: ProfileState,
    opportunity: InitiativeOpportunity,
) -> bool:
    if opportunity.source_type == "offscreen_life":
        return any(
            opportunity.source_id == f"offscreen_life:{activity.started_at:.6f}"
            for activity in (
                profile.presence.current_activity,
                profile.presence.previous_activity,
            )
            if activity is not None
        )
    if opportunity.source_type in {"memory", "reminder"}:
        return any(item.id == opportunity.source_id for item in profile.memories)
    if opportunity.source_type == "realization":
        return any(
            opportunity.source_id
            == f"realization:{item.updated_at:.6f}:{_key(item.topic)[:60]}"
            for item in profile.opinions
        )
    if opportunity.source_type == "relationship":
        return any(
            opportunity.source_id
            == f"relationship:{item.updated_at:.6f}:{_key(item.summary)[:60]}"
            for item in (
                *profile.relationship.unresolved_events,
                *profile.relationship.shared_context,
            )
        )
    return False


def _ordinary_delivery_allowed(
    state: InitiativeState,
    opportunity: InitiativeOpportunity,
    *,
    now: float,
) -> bool:
    if opportunity.source_type == "reminder":
        return True
    if state.cooldown_until > now:
        return False
    local_date = build_time_context(now=now).local_date
    sent_today = sum(
        item.source_type != "reminder"
        and build_time_context(now=item.delivered_at).local_date == local_date
        for item in state.recent
    )
    return sent_today < MAX_ORDINARY_INITIATIVES_PER_LOCAL_DAY


def effective_emotional_state(profile: ProfileState, *, now: float) -> ProfileState:
    """Return lazy mood/emotion decay without mutating or persisting state."""

    return replace(
        profile,
        mood=effective_mood(profile.mood, now=now),
        emotion=effective_emotion(profile.emotion, now=now),
    )


def format_emotional_context(
    profile: ProfileState,
    *,
    now: float,
    include_unappraised: bool = False,
) -> str:
    effective = effective_emotional_state(profile, now=now)
    mood = effective.mood
    emotion = effective.emotion
    lines: list[str] = []
    if mood.updated_at > 0.0 or mood.cause:
        valence = (
            "strongly negative"
            if mood.valence <= -0.6
            else "mildly negative"
            if mood.valence <= -0.15
            else "strongly positive"
            if mood.valence >= 0.6
            else "mildly positive"
            if mood.valence >= 0.15
            else "balanced"
        )
        energy = (
            "very depleted"
            if mood.energy <= -0.6
            else "somewhat depleted"
            if mood.energy <= -0.15
            else "highly energized"
            if mood.energy >= 0.6
            else "moderately energized"
            if mood.energy >= 0.15
            else "steady energy"
        )
        lines.append(f"Akane's mood: {valence} and {energy}.")
        if mood.cause:
            lines.append(f"Ongoing influence: {mood.cause}.")
    elif include_unappraised:
        lines.append("Akane's underlying mood has not yet been appraised.")

    if emotion.primary != "neutral" and emotion.intensity >= 0.08:
        degree = (
            "strong"
            if emotion.intensity >= 0.7
            else "moderate"
            if emotion.intensity >= 0.35
            else "mild"
        )
        lines.append(
            f"Akane's current emotion: {emotion.primary} at {degree} intensity."
        )
        if emotion.cause:
            lines.append(f"Grounded cause: {emotion.cause}.")
    elif include_unappraised and emotion.updated_at <= 0.0:
        lines.append("Akane's immediate emotion has not yet been appraised.")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    profile_id: str
    conversation_id: str
    revision: int
    profile: ProfileState
    conversation: ConversationRecord
    recent_turns: tuple[ChatTurn, ...]
    relevant_memories: tuple[Memory, ...]
    now: float
    last_profile_assistant_at: float = 0.0
    last_profile_user_at: float = 0.0
    last_profile_initiative: ChatTurn | None = None


def _merge_memories(
    current: tuple[Memory, ...],
    additions: tuple[Memory, ...],
) -> tuple[Memory, ...]:
    result = list(current)
    for memory in additions:
        slot = _memory_slot(memory)
        duplicate = next(
            (
                index
                for index, existing in enumerate(result)
                if existing.subject == memory.subject
                and existing.kind == memory.kind
                and (
                    _duplicate_text(existing.text, memory.text)
                    or bool(
                        slot
                        and _memory_slot(existing) == slot
                    )
                )
            ),
            None,
        )
        if duplicate is None:
            result.append(memory)
        else:
            existing = result[duplicate]
            result[duplicate] = (
                memory
                if (memory.updated_at, memory.confidence)
                >= (existing.updated_at, existing.confidence)
                else existing
            )
    return tuple(
        sorted(result, key=lambda item: (item.updated_at, item.created_at))[
            -MEMORY_MAX_ENTRIES_PER_PROFILE:
        ]
    )


def _merge_preferences(
    current: tuple[AkanePreference, ...],
    additions: tuple[AkanePreference, ...],
) -> tuple[AkanePreference, ...]:
    result = list(current)
    for item in additions:
        match = next(
            (index for index, existing in enumerate(result) if _key(existing.topic) == _key(item.topic)),
            None,
        )
        if match is None:
            result.append(item)
        elif item.updated_at >= result[match].updated_at:
            result[match] = item
    return tuple(result[-_MAX_PREFERENCES:])


def _merge_opinions(
    current: tuple[Opinion, ...],
    additions: tuple[Opinion, ...],
) -> tuple[Opinion, ...]:
    result = list(current)
    for item in additions:
        match = next(
            (index for index, existing in enumerate(result) if _key(existing.topic) == _key(item.topic)),
            None,
        )
        if match is None:
            result.append(item)
        elif item.updated_at >= result[match].updated_at:
            result[match] = item
    return tuple(result[-_MAX_OPINIONS:])


def _merge_relationship_entries(
    current: tuple[RelationshipEntry, ...],
    additions: tuple[RelationshipEntry, ...],
) -> tuple[RelationshipEntry, ...]:
    result = list(current)
    for item in additions:
        match = next(
            (
                index
                for index, existing in enumerate(result)
                if _similar(existing.summary, item.summary) >= 0.70
            ),
            None,
        )
        if match is None:
            result.append(item)
        elif item.updated_at >= result[match].updated_at:
            result[match] = item
    return tuple(result[-_MAX_RELATIONSHIP_ENTRIES:])


def _merge_relationship(
    current: RelationshipState,
    addition: RelationshipState,
) -> RelationshipState:
    return RelationshipState(
        _merge_relationship_entries(current.patterns, addition.patterns),
        _merge_relationship_entries(current.shared_context, addition.shared_context),
        _merge_relationship_entries(
            current.unresolved_events,
            addition.unresolved_events,
        ),
    )


def _merge_profiles(left: ProfileState, right: ProfileState, *, now: float) -> ProfileState:
    mood = max(
        (left.mood, right.mood),
        key=lambda item: (
            item.updated_at,
            int(bool(item.cause) or item.valence != 0.0 or item.energy != 0.0),
        ),
    )
    emotion = max(
        (left.emotion, right.emotion),
        key=lambda item: (
            item.updated_at,
            int(item.primary != "neutral" or item.intensity > 0.0 or bool(item.cause)),
        ),
    )
    current_candidates = tuple(
        activity
        for activity in (
            left.presence.current_activity,
            right.presence.current_activity,
        )
        if activity is not None
    )
    current = max(
        current_candidates,
        key=lambda item: (item.started_at, item.expected_end_at),
        default=None,
    )
    historical = (
        *current_candidates,
        left.presence.previous_activity,
        right.presence.previous_activity,
    )
    previous = max(
        (item for item in historical if item is not None and item != current),
        key=lambda item: (item.started_at, item.expected_end_at),
        default=None,
    )
    source_presence = max(
        (left.presence, right.presence),
        key=lambda item: (
            int(current is not None and item.current_activity == current),
            item.last_decision_at,
            item.next_decision_at,
        ),
    )
    presence = normalize_presence(
        replace(
            source_presence,
            current_activity=current,
            previous_activity=previous,
            claim_token=None,
            claim_expires_at=0.0,
        ),
        now=now,
        initialize_schedule=True,
        repair_schedule=True,
    )
    initiative_source = max(
        (left.initiative, right.initiative),
        key=lambda item: (
            item.current.created_at if item.current else 0.0,
            item.cooldown_until,
        ),
    )
    recent = sorted(
        {
            (item.topic_key, item.source_id): item
            for item in (*left.initiative.recent, *right.initiative.recent)
        }.values(),
        key=lambda item: item.delivered_at,
    )[-_MAX_RECENT_INITIATIVES:]
    initiative = replace(
        initiative_source,
        cooldown_until=max(
            left.initiative.cooldown_until,
            right.initiative.cooldown_until,
        ),
        recent=tuple(recent),
        handled_source_ids=tuple(
            dict.fromkeys(
                (
                    *left.initiative.handled_source_ids,
                    *right.initiative.handled_source_ids,
                )
            )
        )[-_MAX_RECENT_INITIATIVES:],
    )
    return ProfileState(
        mood=mood,
        emotion=emotion,
        presence=presence,
        memories=_merge_memories(left.memories, right.memories),
        interests=_dedupe_text(
            (*left.interests, *right.interests),
            limit=_MAX_INTERESTS,
        ),
        preferences=_merge_preferences(left.preferences, right.preferences),
        opinions=_merge_opinions(left.opinions, right.opinions),
        relationship=_merge_relationship(left.relationship, right.relationship),
        initiative=initiative,
        updated_at=max(left.updated_at, right.updated_at),
    )


def _merge_conversations(
    left: ConversationRecord,
    right: ConversationRecord,
) -> ConversationRecord:
    turns_by_id: dict[str, ChatTurn] = {
        turn.turn_id: turn for turn in (*left.recent_turns, *right.recent_turns)
    }
    turns = _trim_turns(
        tuple(
            sorted(
                turns_by_id.values(),
                key=lambda item: (item.timestamp, item.turn_id),
            )
        )
    )
    return ConversationRecord(
        left.conversation_id,
        left.profile_id,
        turns,
        max(left.last_user_at, right.last_user_at),
        max(left.last_assistant_at, right.last_assistant_at),
        max(left.updated_at, right.updated_at),
        tuple(dict.fromkeys((*left.committed_request_ids, *right.committed_request_ids)))[-32:],
        tuple(dict((*left.request_replies, *right.request_replies)).items())[-32:],
    )


def _validate_canonical_profile(
    payload: dict[str, object],
    profile: ProfileState,
    *,
    now: float,
) -> None:
    """Reject current-schema data that parsing would silently discard."""

    normalized = dict(payload)
    raw_presence = payload.get("presence")
    if isinstance(raw_presence, dict):
        normalized_presence = dict(raw_presence)
        token = normalized_presence.get("claim_token")
        expires = _number(normalized_presence.get("claim_expires_at"))
        if isinstance(token, str) and token.strip() and expires <= now:
            normalized_presence["claim_token"] = None
            normalized_presence["claim_expires_at"] = 0.0
        normalized["presence"] = normalized_presence
    raw_initiative = payload.get("initiative")
    if isinstance(raw_initiative, dict):
        normalized_initiative = dict(raw_initiative)
        raw_current = raw_initiative.get("current")
        if isinstance(raw_current, dict):
            normalized_current = dict(raw_current)
            token = normalized_current.get("claim_token")
            expires = _number(normalized_current.get("claim_expires_at"))
            if isinstance(token, str) and token.strip() and expires <= now:
                channel = compact_text(
                    normalized_current.get("delivery_channel"),
                    16,
                ).casefold()
                failed = list(normalized_current.get("failed_channels") or ())
                if (
                    normalized_current.get("status") == "pending_delivery"
                    and channel in {"popup", "discord"}
                    and channel not in failed
                ):
                    failed.append(channel)
                normalized_current["claim_token"] = None
                normalized_current["claim_expires_at"] = 0.0
                if normalized_current.get("status") != "sent":
                    normalized_current["delivery_channel"] = None
                normalized_current["failed_channels"] = failed
            normalized_initiative["current"] = normalized_current
        normalized["initiative"] = normalized_initiative
    raw_mood = payload.get("mood")
    if isinstance(raw_mood, dict):
        normalized_mood = dict(raw_mood)
        if _number(normalized_mood.get("updated_at")) > now:
            normalized_mood["updated_at"] = now
        normalized["mood"] = normalized_mood
    raw_emotion = payload.get("emotion")
    if isinstance(raw_emotion, dict):
        normalized_emotion = dict(raw_emotion)
        for field_name in ("started_at", "updated_at"):
            if _number(normalized_emotion.get(field_name)) > now:
                normalized_emotion[field_name] = now
        normalized["emotion"] = normalized_emotion
    if normalized != profile.as_dict():
        raise ValueError("canonical profile contains malformed state")


def _validate_canonical_conversation(
    payload: dict[str, object],
    conversation: ConversationRecord,
) -> None:
    """Require every current-schema conversation field to survive parsing."""

    normalized = dict(payload)
    normalized["profile_id"] = canonical_profile_id(payload.get("profile_id"))
    if normalized != conversation.as_dict():
        raise ValueError("canonical conversation contains malformed state")


def _profile_emotional_evidence(
    profile: ProfileState,
    *,
    relevance: str,
    broad: bool,
) -> str:
    activities = tuple(
        activity.fact()
        for activity in (
            profile.presence.current_activity,
            profile.presence.previous_activity,
        )
        if activity is not None
    )
    relationship = profile.relationship
    relevance_terms = _terms(relevance)

    def relevant(text: str) -> bool:
        return broad or bool(_terms(text) & relevance_terms)

    return " ".join(
        (
            *activities,
            *(memory.text for memory in profile.memories if relevant(memory.text)),
            *(item for item in profile.interests if relevant(item)),
            *(
                item.content
                for item in profile.preferences
                if relevant(item.content)
            ),
            *(item.content for item in profile.opinions if relevant(item.content)),
            *(
                item.summary
                for item in relationship.patterns
                if relevant(item.summary)
            ),
            *(
                item.summary
                for item in relationship.shared_context
                if relevant(item.summary)
            ),
            *(item.summary for item in relationship.unresolved_events),
            profile.mood.cause,
            profile.emotion.cause,
        )
    )


def _grounded_emotional_cause(cause: str, evidence: str) -> bool:
    cause_key = _key(cause)
    if (
        not cause
        or any(item in cause_key for item in _GENERIC_CAUSES)
        or _UNGROUNDED_CAUSE_LANGUAGE.search(cause_key)
    ):
        return False
    candidate_terms = _terms(cause)
    overlap = candidate_terms & _terms(evidence)
    return bool(candidate_terms and overlap) and (
        len(overlap) >= 2 or len(candidate_terms) <= 3
    )


def _materialize_effective_state(
    profile: ProfileState,
    *,
    now: float,
) -> ProfileState:
    effective = effective_emotional_state(profile, now=now)
    mood = effective.mood
    emotion = effective.emotion
    if mood != profile.mood:
        mood = replace(mood, updated_at=now)
    if emotion != profile.emotion:
        emotion = (
            EmotionState(updated_at=now)
            if emotion.primary == "neutral"
            else replace(emotion, updated_at=now)
        )
    return replace(profile, mood=mood, emotion=emotion)


def _emotion_fields_valid(
    payload: object,
    *,
    require_canonical: bool,
) -> bool:
    if not isinstance(payload, dict):
        return False
    mode = compact_text(payload.get("mode"), 16).casefold()
    if mode in {"keep", "settle"}:
        if not require_canonical and set(payload) == {"mode"}:
            return True
        return all(
            payload.get(name) is None
            for name in ("primary", "intensity", "cause")
        ) and set(payload) == set(EMOTION_UPDATE_FIELDS)
    if mode != "shift" or set(payload) != set(EMOTION_UPDATE_FIELDS):
        return False
    intensity = payload.get("intensity")
    return (
        compact_text(payload.get("primary"), 32).casefold()
        in EMOTION_VOCABULARY - {"neutral"}
        and type(intensity) in {int, float}
        and math.isfinite(float(intensity))
        and bool(compact_text(payload.get("cause"), 160))
    )


def _mood_fields_valid(payload: object) -> bool:
    if not isinstance(payload, dict) or set(payload) != set(MOOD_UPDATE_FIELDS):
        return False
    cause = payload.get("cause")
    return all(
        type(payload.get(name)) in {int, float}
        and math.isfinite(float(payload[name]))
        for name in ("valence_delta", "energy_delta")
    ) and (cause is None or bool(compact_text(cause, 160)))


def _apply_emotional_updates(
    profile: ProfileState,
    *,
    emotion_update: object,
    mood_update: object,
    evidence: str,
    source: str,
    source_id: str | None,
    now: float,
    mood_delta_limit: float,
    expected_emotion_updated_at: float,
    require_complete: bool,
) -> tuple[ProfileState, bool]:
    emotion_valid = _emotion_fields_valid(
        emotion_update,
        require_canonical=require_complete,
    )
    mood_valid = _mood_fields_valid(mood_update)
    if require_complete and (
        not emotion_valid
        or (mood_update is not None and not mood_valid)
    ):
        return profile, False

    next_profile = profile
    effective = effective_emotional_state(profile, now=now)
    profile_evidence = _profile_emotional_evidence(
        profile,
        relevance=evidence,
        broad=source == "offscreen_life",
    )
    emotional_evidence = f"{evidence} {profile_evidence}"

    if emotion_valid:
        values = emotion_update
        mode = compact_text(values.get("mode"), 16).casefold()
        emotion = effective.emotion
        newer_emotion = (
            profile.emotion.updated_at > expected_emotion_updated_at
            and profile.emotion.source_id != source_id
        )
        stale = (
            source == "offscreen_life"
            and newer_emotion
        )
        grounded = True
        if mode == "shift":
            primary = compact_text(values.get("primary"), 32).casefold()
            intensity = max(0.0, min(1.0, float(values["intensity"])))
            cause = compact_text(values.get("cause"), 160)
            grounded = (
                _grounded_emotional_cause(cause, emotional_evidence)
                and (
                    not require_complete
                    or 0.08 <= float(values["intensity"]) <= 1.0
                )
            )
            if require_complete and not grounded:
                return profile, False

        if stale:
            emotion = profile.emotion
        elif mode == "keep":
            emotion = replace(emotion, updated_at=now)
        elif mode == "settle":
            emotion = EmotionState(updated_at=now)
        else:
            related_episode = (
                primary == emotion.primary
                or bool(emotion.cause)
                and _similar(cause, emotion.cause) >= 0.72
            )
            if grounded and related_episode:
                refreshed = (
                    cause
                    if not emotion.cause
                    or (
                        _similar(cause, emotion.cause) < 0.82
                        and bool(_terms(cause) - _terms(emotion.cause))
                    )
                    else emotion.cause
                )
                refresh_source = refreshed != emotion.cause
                emotion = replace(
                    emotion,
                    primary=primary,
                    intensity=max(
                        0.0,
                        min(1.0, emotion.intensity * 0.60 + intensity * 0.40),
                    ),
                    cause=refreshed,
                    source=source if refresh_source else emotion.source,
                    source_id=source_id if refresh_source else emotion.source_id,
                    started_at=emotion.started_at or now,
                    updated_at=now,
                )
            elif (
                grounded
                and (require_complete or intensity >= 0.18)
                and (
                    not newer_emotion
                    or intensity >= emotion.intensity + 0.15
                )
            ):
                emotion = EmotionState(
                    primary,
                    intensity,
                    cause,
                    source,
                    source_id,
                    now,
                    now,
                )
        next_profile = replace(next_profile, emotion=emotion)

    if mood_valid:
        values = mood_update
        cause = compact_text(values.get("cause"), 160)
        valence_delta = max(
            -mood_delta_limit,
            min(mood_delta_limit, float(values["valence_delta"])),
        )
        energy_delta = max(
            -mood_delta_limit,
            min(mood_delta_limit, float(values["energy_delta"])),
        )
        grounded = _grounded_emotional_cause(cause, emotional_evidence)
        changed = valence_delta != 0.0 or energy_delta != 0.0
        if changed and grounded:
            mood = effective.mood
            next_profile = replace(
                next_profile,
                mood=MoodState(
                    max(-1.0, min(1.0, mood.valence + valence_delta)),
                    max(-1.0, min(1.0, mood.energy + energy_delta)),
                    cause,
                    now,
                ),
            )
        elif changed and require_complete:
            return profile, False

    return next_profile, True


class StateStore:
    """Sole production owner of validated state and atomic persistence."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else LONG_TERM_MEMORY_PATH
        self._default_path = path is None
        self._lock = threading.RLock()
        self._profiles: dict[str, ProfileState] = {}
        self._conversations: dict[str, ConversationRecord] = {}
        self._revision = 0
        self._committed_at = 0.0
        self._autonomy_wake: Callable[[str], None] | None = None
        self._load()

    def _document(
        self,
        profiles: dict[str, ProfileState],
        conversations: dict[str, ConversationRecord],
        revision: int,
        committed_at: float,
    ) -> dict[str, object]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "canonical_profile_id": OWNER_PROFILE_ID,
            "revision": revision,
            "committed_at": committed_at,
            "profiles": {
                key: value.as_dict() for key, value in profiles.items()
            },
            "conversations": {
                key: value.as_dict() for key, value in conversations.items()
            },
        }

    def _replace_all(
        self,
        profiles: dict[str, ProfileState],
        conversations: dict[str, ConversationRecord],
        *,
        committed_at: float,
    ) -> bool:
        if profiles == self._profiles and conversations == self._conversations:
            return False
        revision = self._revision + 1
        atomic_write_json(
            self._path,
            self._document(profiles, conversations, revision, committed_at),
        )
        self._profiles = profiles
        self._conversations = conversations
        self._revision = revision
        self._committed_at = committed_at
        return True

    def _load_payload(
        self,
        payload: object,
        *,
        now: float,
    ) -> tuple[
        dict[str, ProfileState],
        dict[str, ConversationRecord],
        int,
        float,
    ]:
        if not isinstance(payload, dict):
            raise ValueError("state document is not an object")
        schema = int(_number(payload.get("schema_version")))
        if schema not in range(0, STATE_SCHEMA_VERSION + 1):
            raise ValueError(f"unsupported state schema {schema}")
        if schema == STATE_SCHEMA_VERSION:
            if (
                set(payload)
                != {
                    "schema_version",
                    "canonical_profile_id",
                    "revision",
                    "committed_at",
                    "profiles",
                    "conversations",
                }
                or type(payload.get("schema_version")) is not int
                or payload.get("schema_version") != STATE_SCHEMA_VERSION
                or payload.get("canonical_profile_id") != OWNER_PROFILE_ID
                or type(payload.get("revision")) is not int
                or int(payload["revision"]) < 0
                or type(payload.get("committed_at")) not in {int, float}
                or not math.isfinite(float(payload["committed_at"]))
                or float(payload["committed_at"]) < 0.0
            ):
                raise ValueError("canonical state header is malformed")
        if schema == 0 and not (
            isinstance(payload.get("user"), dict)
            or any(key in payload for key in ("name", "likes", "dislikes", "facts"))
        ):
            raise ValueError("unrecognized legacy state document")
        profiles: dict[str, ProfileState] = {}
        conversations: dict[str, ConversationRecord] = {}
        raw_profiles = payload.get("profiles")
        if schema == STATE_SCHEMA_VERSION and not isinstance(raw_profiles, dict):
            raise ValueError("canonical state profiles are unavailable")
        if isinstance(raw_profiles, dict):
            for raw_id, raw_profile in raw_profiles.items():
                if schema == STATE_SCHEMA_VERSION and not isinstance(
                    raw_profile,
                    dict,
                ):
                    raise ValueError("canonical profile is malformed")
                profile = ProfileState.from_dict(
                    (
                        raw_profile
                        if schema == STATE_SCHEMA_VERSION
                        else _legacy_profile_payload(raw_profile)
                    ),
                    now=now,
                    repair_presence=schema != STATE_SCHEMA_VERSION,
                    migrating=schema != STATE_SCHEMA_VERSION,
                )
                if profile is None:
                    continue
                if schema == STATE_SCHEMA_VERSION:
                    _validate_canonical_profile(
                        raw_profile,
                        profile,
                        now=now,
                    )
                profile_id = canonical_profile_id(raw_id)
                profiles[profile_id] = (
                    _merge_profiles(profiles[profile_id], profile, now=now)
                    if profile_id in profiles
                    else profile
                )
        elif isinstance(payload.get("user"), dict):
            profiles[OWNER_PROFILE_ID] = self._popup_profile(
                payload["user"],
                now=now,
            )
        elif schema == 0:
            profiles[OWNER_PROFILE_ID] = self._popup_profile(payload, now=now)
        raw_conversations = payload.get("conversations")
        if schema == STATE_SCHEMA_VERSION and not isinstance(
            raw_conversations,
            dict,
        ):
            raise ValueError("canonical state conversations are unavailable")
        if isinstance(raw_conversations, dict):
            for key, raw in raw_conversations.items():
                if schema == STATE_SCHEMA_VERSION and not isinstance(raw, dict):
                    raise ValueError("canonical conversation is malformed")
                record = ConversationRecord.from_dict(str(key), raw)
                if record is None:
                    if schema == STATE_SCHEMA_VERSION:
                        raise ValueError("canonical conversation is malformed")
                    continue
                if schema == STATE_SCHEMA_VERSION:
                    _validate_canonical_conversation(raw, record)
                existing = conversations.get(record.conversation_id)
                conversations[record.conversation_id] = (
                    _merge_conversations(existing, record) if existing else record
                )
        if schema == STATE_SCHEMA_VERSION:
            raw_owner = next(
                (
                    value
                    for key, value in raw_profiles.items()
                    if canonical_profile_id(key) == OWNER_PROFILE_ID
                ),
                None,
            )
            if (
                OWNER_PROFILE_ID not in profiles
                or not isinstance(raw_owner, dict)
                or not set(raw_owner)
                & {
                    "emotion",
                    "mood",
                    "presence",
                    "memories",
                    "interests",
                    "preferences",
                    "opinions",
                    "relationship",
                }
            ):
                raise ValueError("canonical owner profile is unavailable")
        return (
            profiles,
            conversations,
            int(_number(payload.get("revision"))) if schema == STATE_SCHEMA_VERSION else 0,
            max(0.0, _number(payload.get("committed_at"))),
        )

    def _popup_profile(self, payload: dict[str, object], *, now: float) -> ProfileState:
        memories: list[Memory] = []

        def add(text: str, kind: str = "fact") -> None:
            value = compact_text(text, 300)
            if value:
                memories.append(
                    Memory(
                        uuid.uuid4().hex,
                        "user",
                        kind,
                        value,
                        0.8,
                        now,
                        now,
                    )
                )

        name = compact_text(payload.get("name"), 80)
        if name:
            add(f"The user's name is {name}")
        for key, prefix in (("likes", "The user likes"), ("dislikes", "The user dislikes")):
            raw = payload.get(key)
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    if compact_text(item, 120):
                        add(f"{prefix} {compact_text(item, 120)}")
        raw_facts = payload.get("facts")
        if isinstance(raw_facts, (list, tuple)):
            for fact in raw_facts:
                add(str(fact))
        return replace(
            _new_profile(now),
            memories=_merge_memories((), tuple(memories)),
            updated_at=now if memories else 0.0,
        )

    def _read_migration_source(self, path: Path) -> object | None:
        try:
            return read_json(path)
        except FileNotFoundError:
            return None
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Akane state recovery is required for {path}: "
                f"{type(exc).__name__}"
            ) from exc

    def _decode_source(
        self,
        path: Path,
        payload: object,
        *,
        now: float,
    ) -> tuple[
        dict[str, ProfileState],
        dict[str, ConversationRecord],
        int,
        float,
    ]:
        try:
            return self._load_payload(payload, now=now)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Akane state recovery is required for {path}: "
                f"{type(exc).__name__}"
            ) from exc

    def _load(self) -> None:
        now = time.time()
        canonical = self._read_migration_source(self._path)
        migrated = False
        if canonical is not None:
            profiles, conversations, revision, committed_at = self._decode_source(
                self._path,
                canonical,
                now=now,
            )
            schema = int(_number(canonical.get("schema_version"))) if isinstance(canonical, dict) else 0
            migrated = schema != STATE_SCHEMA_VERSION
            self._profiles = profiles
            self._conversations = conversations
            self._revision = revision
            self._committed_at = committed_at
            if schema == STATE_SCHEMA_VERSION:
                raw_profiles = canonical["profiles"]
                raw_conversations = canonical["conversations"]
                migrated = migrated or any(
                    str(profile_id) != canonical_profile_id(profile_id)
                    or bool(
                        isinstance(raw_profile.get("presence"), dict)
                        and (
                            raw_profile["presence"].get("claim_token")
                            or _number(
                                raw_profile["presence"].get("claim_expires_at")
                            )
                            > 0.0
                        )
                    )
                    or bool(
                        isinstance(raw_profile.get("mood"), dict)
                        and _number(raw_profile["mood"].get("updated_at")) > now
                    )
                    or bool(
                        isinstance(raw_profile.get("emotion"), dict)
                        and any(
                            _number(raw_profile["emotion"].get(field_name)) > now
                            for field_name in ("started_at", "updated_at")
                        )
                    )
                    or bool(
                        isinstance(raw_profile.get("initiative"), dict)
                        and isinstance(
                            raw_profile["initiative"].get("current"),
                            dict,
                        )
                        and raw_profile["initiative"]["current"].get(
                            "claim_token"
                        )
                        and _number(
                            raw_profile["initiative"]["current"].get(
                                "claim_expires_at"
                            )
                        )
                        <= now
                    )
                    for profile_id, raw_profile in raw_profiles.items()
                )
                migrated = migrated or set(raw_conversations) != set(conversations)
                migrated = migrated or any(
                    raw.get("profile_id")
                    != canonical_profile_id(raw.get("profile_id"))
                    for raw in raw_conversations.values()
                )
        if self._default_path and (
            canonical is None
            or not isinstance(canonical, dict)
            or int(_number(canonical.get("schema_version"))) != STATE_SCHEMA_VERSION
        ):
            for source in (MEMORY_PATH, POPUP_USER_PATH):
                if source == self._path:
                    continue
                payload = self._read_migration_source(source)
                if payload is None:
                    continue
                profiles, conversations, _revision, _committed = self._decode_source(
                    source,
                    payload,
                    now=now,
                )
                for profile_id, profile in profiles.items():
                    self._profiles[profile_id] = (
                        _merge_profiles(
                            self._profiles[profile_id],
                            profile,
                            now=now,
                        )
                        if profile_id in self._profiles
                        else profile
                    )
                for conversation_id, conversation in conversations.items():
                    existing = self._conversations.get(conversation_id)
                    self._conversations[conversation_id] = (
                        _merge_conversations(existing, conversation)
                        if existing
                        else conversation
                    )
                migrated = True

        if OWNER_PROFILE_ID not in self._profiles:
            self._profiles[OWNER_PROFILE_ID] = _new_profile(now)
            migrated = True
        normalized: dict[str, ProfileState] = {}
        for profile_id, profile in self._profiles.items():
            repair_failed_bootstrap = (
                profile.presence.current_activity is None
                and profile.presence.last_decision_at == 0.0
                and profile.presence.next_decision_at > now
                and bool(profile.presence.last_error)
            )
            presence = normalize_presence(
                profile.presence,
                now=now,
                initialize_schedule=True,
                repair_schedule=migrated,
            )
            if repair_failed_bootstrap:
                presence = replace(
                    presence,
                    next_decision_at=0.0,
                    retry_at=now,
                    claim_token=None,
                    claim_expires_at=0.0,
                )
                migrated = True
            if presence.claim_token is not None or presence.claim_expires_at:
                presence = replace(
                    presence,
                    claim_token=None,
                    claim_expires_at=0.0,
                )
                migrated = True
            normalized[canonical_profile_id(profile_id)] = replace(
                profile,
                presence=presence,
            )
        self._profiles = normalized
        migrated = self._prune_conversations(now) or migrated
        if canonical is None or migrated:
            atomic_write_json(
                self._path,
                self._document(
                    self._profiles,
                    self._conversations,
                    max(1, self._revision + 1),
                    now,
                ),
            )
            self._revision = max(1, self._revision + 1)
            self._committed_at = now

    def _prune_conversations(self, now: float) -> bool:
        before = self._conversations
        cutoff = now - CONVERSATION_STALE_DAYS * 24.0 * 3600.0
        kept = {
            key: value
            for key, value in self._conversations.items()
            if not value.updated_at or value.updated_at >= cutoff
        }
        if len(kept) > MAX_CONVERSATIONS:
            newest = sorted(
                kept.values(),
                key=lambda item: item.updated_at,
                reverse=True,
            )[:MAX_CONVERSATIONS]
            kept = {item.conversation_id: item for item in newest}
        self._conversations = kept
        return kept != before

    def _conversation(
        self,
        profile_id: str,
        conversation_id: str,
    ) -> ConversationRecord:
        current = self._conversations.get(conversation_id)
        if current is not None and current.profile_id == profile_id:
            return current
        return ConversationRecord(conversation_id, profile_id)

    def snapshot(
        self,
        profile_id: str = OWNER_PROFILE_ID,
        conversation_id: str = "popup:default",
        query: str = "",
        now: float | None = None,
        include_memory: bool = True,
    ) -> StateSnapshot:
        profile_key = canonical_profile_id(profile_id)
        conversation_key = compact_text(conversation_id, 160) or "popup:default"
        current = time.time() if now is None else max(0.0, float(now))
        with self._lock:
            profile = self._profiles.get(profile_key) or _new_profile(current)
            conversation = self._conversation(profile_key, conversation_key)
            last_profile_assistant_at = max(
                (
                    record.last_assistant_at
                    for record in self._conversations.values()
                    if record.profile_id == profile_key
                ),
                default=0.0,
            )
            last_profile_user_at = max(
                (
                    record.last_user_at
                    for record in self._conversations.values()
                    if record.profile_id == profile_key
                ),
                default=0.0,
            )
            last_profile_initiative = max(
                (
                    turn
                    for record in self._conversations.values()
                    if record.profile_id == profile_key
                    for turn in record.recent_turns
                    if turn.role == "assistant" and turn.source == "initiative"
                ),
                key=lambda item: item.timestamp,
                default=None,
            )
            recalled = (
                self._retrieve(profile.memories, query, current)
                if include_memory
                else ()
            )
            return StateSnapshot(
                profile_key,
                conversation_key,
                self._revision,
                profile,
                conversation,
                _complete_turns(conversation.recent_turns),
                recalled,
                current,
                last_profile_assistant_at,
                last_profile_user_at,
                last_profile_initiative,
            )

    def _retrieve(
        self,
        memories: tuple[Memory, ...],
        query: str,
        now: float,
    ) -> tuple[Memory, ...]:
        query_terms = _terms(query)
        if not query_terms:
            return ()
        if query_terms & {"remember", "memory", "memories"} and len(query_terms) <= 5:
            return tuple(
                sorted(
                    memories,
                    key=lambda item: (item.confidence, item.updated_at),
                    reverse=True,
                )[:MEMORY_MAX_RESULTS]
            )
        scored: list[tuple[float, Memory]] = []
        for memory in memories:
            memory_terms = _terms(memory.text)
            overlap = len(query_terms & memory_terms) / max(
                1,
                len(query_terms | memory_terms),
            )
            phrase = SequenceMatcher(None, _key(query), _key(memory.text)).ratio()
            if overlap <= 0.0 and phrase < 0.28:
                continue
            age_days = max(0.0, now - memory.updated_at) / (24.0 * 3600.0)
            recency = 1.0 / (1.0 + age_days / 30.0)
            score = 0.62 * max(overlap, phrase * 0.6) + 0.23 * memory.confidence + 0.15 * recency
            if score >= 0.26:
                scored.append((score, memory))
        return tuple(
            item
            for _score, item in sorted(
                scored,
                key=lambda pair: (pair[0], pair[1].updated_at),
                reverse=True,
            )[:MEMORY_MAX_RESULTS]
        )

    def _validated_memory(
        self,
        payload: object,
        *,
        user_text: str,
        assistant_text: str,
        now: float,
    ) -> Memory | None:
        if not isinstance(payload, dict) or set(payload) != {
            "subject", "kind", "text", "confidence",
        }:
            return None
        subject = compact_text(payload.get("subject"), 16).casefold()
        kind = compact_text(payload.get("kind"), 24).casefold()
        text = compact_text(payload.get("text"), 360)
        confidence = payload.get("confidence")
        if (
            subject not in _MEMORY_SUBJECTS
            or kind not in _MEMORY_KINDS
            or not text
            or type(confidence) not in {int, float}
            or not math.isfinite(float(confidence))
            or _key(text) in _TRIVIAL_MEMORY
            or text.rstrip().endswith("?")
            or len(words(text)) < 3
            or len(_terms(text)) < (3 if kind == "event" else 1)
            or (
                len(_terms(text)) <= 4
                and bool(_terms(text) & _TRIVIAL_MEMORY_TERMS)
            )
            or (
                kind in {"fact", "event"}
                and bool(_TRANSIENT_STATEMENT.search(text))
            )
        ):
            return None
        evidence = (
            user_text
            if subject == "user"
            else f"{user_text} {assistant_text}"
        )
        if (
            not _grounded(text, evidence)
            or subject != "user"
            and not (_terms(text) & _terms(user_text))
        ):
            return None
        certainty = max(0.0, min(1.0, float(confidence)))
        if (
            subject == "user"
            and _UNCERTAINTY.search(user_text)
            and not _UNCERTAINTY.search(text)
        ):
            return None
        return Memory(
            uuid.uuid4().hex,
            subject,
            kind,
            text,
            certainty,
            now,
            now,
        )

    def _apply_proposals(
        self,
        profile: ProfileState,
        proposals: object,
        *,
        user_text: str,
        assistant_text: str,
        source: str,
        source_id: str | None,
        expected_emotion_updated_at: float,
        now: float,
    ) -> ProfileState:
        values = proposals if isinstance(proposals, dict) else {}
        emotional_input = _key(user_text)
        if emotional_input in _TRIVIAL_EMOTIONAL_INPUTS:
            next_profile = _materialize_effective_state(profile, now=now)
        else:
            next_profile, _valid = _apply_emotional_updates(
                profile,
                emotion_update=values.get("emotion_update"),
                mood_update=values.get("mood_update"),
                evidence=f"{user_text} {assistant_text}",
                source=source,
                source_id=source_id,
                now=now,
                mood_delta_limit=0.12,
                expected_emotion_updated_at=expected_emotion_updated_at,
                require_complete=False,
            )

        raw_memories = values.get("memories")
        if isinstance(raw_memories, list):
            additions = tuple(
                memory
                for item in raw_memories[:8]
                if (
                    memory := self._validated_memory(
                        item,
                        user_text=user_text,
                        assistant_text=assistant_text,
                        now=now,
                    )
                )
                is not None
            )
            if additions:
                next_profile = replace(
                    next_profile,
                    memories=_merge_memories(next_profile.memories, additions),
                )

        raw_preferences = values.get("preferences")
        if isinstance(raw_preferences, list):
            additions: list[AkanePreference] = []
            for item in raw_preferences[:6]:
                if not isinstance(item, dict) or set(item) != {
                    "topic", "stance", "reason",
                }:
                    continue
                candidate = AkanePreference.from_dict({**item, "updated_at": now})
                if (
                    candidate
                    and _grounded(candidate.topic, user_text)
                    and _grounded(
                        f"{candidate.topic} {candidate.reason}",
                        f"{user_text} {assistant_text}",
                    )
                ):
                    additions.append(candidate)
            if additions:
                next_profile = replace(
                    next_profile,
                    preferences=_merge_preferences(
                        next_profile.preferences,
                        tuple(additions),
                    ),
                )

        raw_interests = values.get("interests")
        if isinstance(raw_interests, list):
            additions = tuple(
                interest
                for item in raw_interests[:6]
                if (
                    interest := compact_text(item, 100)
                )
                and 1 <= len(_terms(interest)) <= 5
                and _grounded(interest, user_text)
            )
            if additions:
                next_profile = replace(
                    next_profile,
                    interests=_dedupe_text(
                        (*next_profile.interests, *additions),
                        limit=_MAX_INTERESTS,
                    ),
                )

        raw_opinions = values.get("opinions")
        if isinstance(raw_opinions, list):
            additions: list[Opinion] = []
            for item in raw_opinions[:6]:
                if not isinstance(item, dict) or set(item) != {
                    "topic", "position", "reason",
                }:
                    continue
                candidate = Opinion.from_dict({**item, "updated_at": now})
                if (
                    candidate
                    and _grounded(candidate.topic, user_text)
                    and _grounded(
                        f"{candidate.topic} {candidate.position} {candidate.reason}",
                        f"{user_text} {assistant_text}",
                    )
                ):
                    additions.append(candidate)
            if additions:
                next_profile = replace(
                    next_profile,
                    opinions=_merge_opinions(
                        next_profile.opinions,
                        tuple(additions),
                    ),
                )

        relationship = values.get("relationship")
        if isinstance(relationship, dict):
            additions: dict[str, tuple[RelationshipEntry, ...]] = {}
            for field_name in ("patterns", "shared_context", "unresolved_events"):
                raw = relationship.get(field_name)
                accepted: list[RelationshipEntry] = []
                if isinstance(raw, list):
                    for item in raw[:6]:
                        if not isinstance(item, dict) or set(item) != {
                            "summary", "confidence",
                        }:
                            continue
                        candidate = RelationshipEntry.from_dict(
                            {**item, "updated_at": now}
                        )
                        if candidate and _grounded(candidate.summary, user_text):
                            accepted.append(candidate)
                additions[field_name] = tuple(accepted)
            current = next_profile.relationship
            merged = RelationshipState(
                _merge_relationship_entries(
                    current.patterns,
                    additions.get("patterns", ()),
                ),
                _merge_relationship_entries(
                    current.shared_context,
                    additions.get("shared_context", ()),
                ),
                _merge_relationship_entries(
                    current.unresolved_events,
                    additions.get("unresolved_events", ()),
                ),
            )
            resolved = relationship.get("resolved_events")
            unresolved = list(merged.unresolved_events)
            if isinstance(resolved, list):
                for item in resolved[:6]:
                    candidate = RelationshipEntry.from_dict(
                        {**item, "updated_at": now}
                        if isinstance(item, dict)
                        else item
                    )
                    if (
                        candidate
                        and _grounded(candidate.summary, user_text)
                    ):
                        match = next(
                            (
                                index
                                for index, existing in enumerate(unresolved)
                                if _similar(existing.summary, candidate.summary) >= 0.55
                            ),
                            None,
                        )
                        if match is not None:
                            unresolved.pop(match)
            merged = replace(merged, unresolved_events=tuple(unresolved))
            if merged != current:
                next_profile = replace(next_profile, relationship=merged)

        if next_profile != profile:
            next_profile = replace(next_profile, updated_at=now)
        return next_profile

    def commit_turn(
        self,
        snapshot: StateSnapshot,
        *,
        user_text: str,
        assistant_text: str,
        source: str,
        request_id: str = "",
        proposals: object = None,
        now: float | None = None,
    ) -> StateSnapshot:
        committed = time.time() if now is None else max(snapshot.now, float(now))
        with self._lock:
            profile = self._profiles.get(snapshot.profile_id) or _new_profile(committed)
            conversation = self._conversation(
                snapshot.profile_id,
                snapshot.conversation_id,
            )
            request = compact_text(request_id, 180)
            if request and request in conversation.committed_request_ids:
                return self.snapshot(
                    snapshot.profile_id,
                    snapshot.conversation_id,
                    query=user_text,
                    now=committed,
                )
            pair_id = request or uuid.uuid4().hex
            next_profile = self._apply_proposals(
                profile,
                proposals,
                user_text=user_text,
                assistant_text=assistant_text,
                source="conversation",
                source_id=pair_id,
                expected_emotion_updated_at=snapshot.profile.emotion.updated_at,
                now=committed,
            )
            turns = list(conversation.recent_turns)
            if assistant_text:
                turns.extend(
                    (
                        ChatTurn(
                            f"{pair_id}:user",
                            "user",
                            str(user_text).strip()[:8_000],
                            snapshot.now,
                            compact_text(source, 32) or "unknown",
                        ),
                        ChatTurn(
                            f"{pair_id}:assistant",
                            "assistant",
                            str(assistant_text).strip()[:8_000],
                            committed,
                            compact_text(source, 32) or "unknown",
                        ),
                    )
                )
            ids = (
                tuple(dict.fromkeys((*conversation.committed_request_ids, request)))[-32:]
                if request
                else conversation.committed_request_ids
            )
            replies = (
                tuple(
                    dict(
                        (
                            *conversation.request_replies,
                            (request, assistant_text),
                        )
                    ).items()
                )[-32:]
                if request
                else conversation.request_replies
            )
            next_conversation = replace(
                conversation,
                recent_turns=_trim_turns(tuple(turns)),
                last_user_at=max(conversation.last_user_at, snapshot.now),
                last_assistant_at=(
                    max(conversation.last_assistant_at, committed)
                    if assistant_text
                    else conversation.last_assistant_at
                ),
                updated_at=committed,
                committed_request_ids=ids,
                request_replies=replies,
            )
            profiles = self._profiles.copy()
            conversations = self._conversations.copy()
            profiles[snapshot.profile_id] = next_profile
            conversations[snapshot.conversation_id] = next_conversation
            self._replace_all(
                profiles,
                conversations,
                committed_at=committed,
            )
            return StateSnapshot(
                snapshot.profile_id,
                snapshot.conversation_id,
                self._revision,
                next_profile,
                next_conversation,
                _complete_turns(next_conversation.recent_turns),
                self._retrieve(next_profile.memories, user_text, committed),
                committed,
                max(
                    snapshot.last_profile_assistant_at,
                    next_conversation.last_assistant_at,
                ),
                max(
                    snapshot.last_profile_user_at,
                    next_conversation.last_user_at,
                ),
                snapshot.last_profile_initiative,
            )

    def messages(
        self,
        conversation_id: str,
        profile_id: str = OWNER_PROFILE_ID,
    ) -> list[dict[str, str]]:
        key = compact_text(conversation_id, 160) or "popup:default"
        profile = canonical_profile_id(profile_id)
        with self._lock:
            record = self._conversation(profile, key)
            return [turn.as_message() for turn in record.recent_turns]

    def reply_for_request(
        self,
        conversation_id: str,
        profile_id: str,
        request_id: str,
    ) -> str | None:
        request = compact_text(request_id, 180)
        if not request:
            return None
        key = compact_text(conversation_id, 160) or "popup:default"
        with self._lock:
            record = self._conversation(canonical_profile_id(profile_id), key)
            return dict(record.request_replies).get(request)

    def public_conversation(
        self,
        conversation_id: str,
        profile_id: str = OWNER_PROFILE_ID,
    ) -> dict[str, object]:
        key = compact_text(conversation_id, 160) or "popup:default"
        with self._lock:
            record = self._conversation(canonical_profile_id(profile_id), key)
            return {
                "conversation_id": record.conversation_id,
                "profile_id": record.profile_id,
                "recent_turns": [
                    {
                        "role": turn.role,
                        "text": turn.content,
                        "timestamp": turn.timestamp,
                    }
                    for turn in record.recent_turns
                ],
                "last_user_at": record.last_user_at,
                "last_assistant_at": record.last_assistant_at,
                "updated_at": record.updated_at,
            }

    def clear_conversation(self, conversation_id: str, profile_id: str) -> None:
        key = compact_text(conversation_id, 160) or "popup:default"
        with self._lock:
            record = self._conversations.get(key)
            if record is None or record.profile_id != canonical_profile_id(profile_id):
                return
            conversations = self._conversations.copy()
            conversations.pop(key, None)
            self._replace_all(
                self._profiles.copy(),
                conversations,
                committed_at=time.time(),
            )

    def clear_profile(self, profile_id: str = OWNER_PROFILE_ID) -> None:
        profile = canonical_profile_id(profile_id)
        current = time.time()
        with self._lock:
            profiles = self._profiles.copy()
            if profile == OWNER_PROFILE_ID:
                profiles[profile] = _new_profile(current)
            else:
                profiles.pop(profile, None)
            conversations = {
                key: value
                for key, value in self._conversations.items()
                if value.profile_id != profile
            }
            self._replace_all(profiles, conversations, committed_at=current)
            callback = self._autonomy_wake if profile == OWNER_PROFILE_ID else None
        if callback is not None:
            callback(profile)

    def public_profile(self, profile_id: str = OWNER_PROFILE_ID) -> dict[str, object]:
        profile = canonical_profile_id(profile_id)
        with self._lock:
            state = self._profiles.get(profile) or _new_profile(time.time())
            return {
                "profile_id": profile,
                "interests": list(state.interests),
                "preferences": [asdict(item) for item in state.preferences],
                "opinions": [asdict(item) for item in state.opinions],
                "updated_at": state.updated_at,
            }

    def public_memory(
        self,
        profile_id: str = OWNER_PROFILE_ID,
    ) -> dict[str, object]:
        profile = canonical_profile_id(profile_id)
        with self._lock:
            state = self._profiles.get(profile) or _new_profile(time.time())
            current = state.presence.current_activity
            activities = {}
            if current is not None:
                activities[current.activity] = {
                    "status": "active",
                    "details": [
                        value
                        for value in (current.subject, current.detail)
                        if value
                    ],
                }
            return {
                "user": {},
                "preferences": [
                    {"content": item.content}
                    for item in state.preferences
                ],
                "activities": activities,
                "facts": [
                    {
                        "content": item.content,
                        "subject": item.subject,
                        "kind": item.kind,
                        "confidence": item.confidence,
                    }
                    for item in state.memories
                ],
                "interests": list(state.interests),
                "opinions": [
                    {"content": item.content}
                    for item in state.opinions
                ],
            }

    def public_internal_state(
        self,
        profile_id: str = OWNER_PROFILE_ID,
    ) -> dict[str, object]:
        profile = canonical_profile_id(profile_id)
        current = time.time()
        with self._lock:
            state = self._profiles.get(profile) or _new_profile(current)
            effective = effective_emotional_state(state, now=current)
            return {
                **effective.as_dict(),
                "profile_id": profile,
                "revision": self._revision,
                "committed_at": self._committed_at,
            }

    def offer_initiative(
        self,
        opportunity: InitiativeOpportunity,
        *,
        profile_id: str = OWNER_PROFILE_ID,
        now: float,
    ) -> bool:
        profile = canonical_profile_id(profile_id)
        if profile != OWNER_PROFILE_ID or opportunity.status != "pending":
            return False
        with self._lock:
            current_profile = self._profiles.get(profile)
            if current_profile is None:
                return False
            initiative = _settle_initiative(current_profile.initiative, now=now)
            active = initiative.current
            if (
                active is not None
                and active.status in {"pending", "pending_delivery"}
                and active.expires_at > now
            ):
                return False
            if (
                opportunity.source_id in initiative.handled_source_ids
                or any(
                    recent.source_id == opportunity.source_id
                    or (
                        opportunity.source_type != "reminder"
                        and recent.source_type != "reminder"
                        and _similar(recent.topic_key, opportunity.topic_key)
                        >= 0.78
                    )
                    for recent in initiative.recent
                )
            ):
                return False
            next_profile = replace(
                current_profile,
                initiative=replace(initiative, current=opportunity),
                updated_at=max(current_profile.updated_at, now),
            )
            profiles = self._profiles.copy()
            profiles[profile] = next_profile
            changed = self._replace_all(
                profiles,
                self._conversations.copy(),
                committed_at=now,
            )
            callback = self._autonomy_wake
        if changed and callback is not None:
            callback(profile)
        return changed

    def initiative_schedule(
        self,
        *,
        now: float,
    ) -> tuple[bool, float | None]:
        current = max(0.0, float(now))
        with self._lock:
            profile = self._profiles.get(OWNER_PROFILE_ID)
            if profile is None:
                return False, None
            initiative = _settle_initiative(profile.initiative, now=current)
            opportunity = initiative.current
            due = bool(
                opportunity is not None
                and opportunity.status == "pending"
                and opportunity.claim_token is None
                and opportunity.not_before <= current
                and opportunity.expires_at > current
            )
            wakes: list[float] = []
            if opportunity is not None and opportunity.status in {
                "pending",
                "pending_delivery",
            }:
                wakes.append(opportunity.expires_at)
                if opportunity.claim_token:
                    wakes.append(opportunity.claim_expires_at)
                elif opportunity.status == "pending":
                    wakes.append(opportunity.not_before)
                elif initiative.cooldown_until > current:
                    wakes.append(initiative.cooldown_until)
            if initiative != profile.initiative:
                profiles = self._profiles.copy()
                profiles[OWNER_PROFILE_ID] = replace(profile, initiative=initiative)
                self._replace_all(
                    profiles,
                    self._conversations.copy(),
                    committed_at=current,
                )
            future = tuple(value for value in wakes if value > current)
            return due, min(future, default=None)

    def claim_initiative_evaluation(
        self,
        *,
        now: float,
    ) -> InitiativeOpportunity | None:
        current = max(0.0, float(now))
        with self._lock:
            profile = self._profiles.get(OWNER_PROFILE_ID)
            if profile is None:
                return None
            initiative = _settle_initiative(profile.initiative, now=current)
            opportunity = initiative.current
            if (
                opportunity is None
                or opportunity.status != "pending"
                or opportunity.claim_token is not None
                or opportunity.not_before > current
                or opportunity.expires_at <= current
            ):
                return None
            if not _initiative_source_exists(profile, opportunity):
                dismissed = replace(
                    opportunity,
                    status="dismissed",
                    evaluated_at=current,
                )
                initiative = _handled_initiative(
                    replace(initiative, current=dismissed),
                    dismissed,
                )
                profiles = self._profiles.copy()
                profiles[OWNER_PROFILE_ID] = replace(profile, initiative=initiative)
                self._replace_all(
                    profiles,
                    self._conversations.copy(),
                    committed_at=current,
                )
                return None
            claimed = replace(
                opportunity,
                claim_token=uuid.uuid4().hex,
                claim_expires_at=current + _INITIATIVE_CLAIM_SECONDS,
            )
            profiles = self._profiles.copy()
            profiles[OWNER_PROFILE_ID] = replace(
                profile,
                initiative=replace(initiative, current=claimed),
            )
            self._replace_all(
                profiles,
                self._conversations.copy(),
                committed_at=current,
            )
            return claimed

    def complete_initiative_evaluation(
        self,
        *,
        claim_token: str,
        decision: str,
        topic: str | None,
        message: str | None,
        now: float,
    ) -> InitiativeOpportunity | None:
        current = max(0.0, float(now))
        with self._lock:
            profile = self._profiles.get(OWNER_PROFILE_ID)
            opportunity = profile.initiative.current if profile else None
            if (
                profile is None
                or opportunity is None
                or opportunity.status != "pending"
                or opportunity.claim_token != claim_token
            ):
                return None
            initiative = profile.initiative
            normalized_topic = _key(topic)[:120]
            speak = (
                decision == "speak"
                and normalized_topic
                and compact_text(message, 500)
            )
            duplicate = bool(
                speak
                and any(
                    _similar(item.topic_key, normalized_topic) >= 0.78
                    for item in initiative.recent
                )
            )
            if not speak or duplicate:
                completed = replace(
                    opportunity,
                    status="dismissed",
                    evaluated_at=current,
                    claim_token=None,
                    claim_expires_at=0.0,
                )
                initiative = _handled_initiative(
                    replace(initiative, current=completed),
                    completed,
                )
            else:
                completed = replace(
                    opportunity,
                    topic_key=normalized_topic,
                    status="pending_delivery",
                    message=compact_text(message, 500),
                    evaluated_at=current,
                    generated_at=current,
                    claim_token=None,
                    claim_expires_at=0.0,
                )
                initiative = replace(initiative, current=completed)
            profiles = self._profiles.copy()
            profiles[OWNER_PROFILE_ID] = replace(
                profile,
                initiative=initiative,
                updated_at=max(profile.updated_at, current),
            )
            self._replace_all(
                profiles,
                self._conversations.copy(),
                committed_at=current,
            )
            callback = self._autonomy_wake
        if callback is not None:
            callback(OWNER_PROFILE_ID)
        return completed

    def fail_initiative_evaluation(
        self,
        *,
        claim_token: str,
        now: float,
    ) -> bool:
        current = max(0.0, float(now))
        with self._lock:
            profile = self._profiles.get(OWNER_PROFILE_ID)
            opportunity = profile.initiative.current if profile else None
            if (
                profile is None
                or opportunity is None
                or opportunity.status != "pending"
                or opportunity.claim_token != claim_token
            ):
                return False
            pending = replace(
                opportunity,
                not_before=min(opportunity.expires_at, current + RETRY_SECONDS),
                claim_token=None,
                claim_expires_at=0.0,
            )
            profiles = self._profiles.copy()
            profiles[OWNER_PROFILE_ID] = replace(
                profile,
                initiative=replace(profile.initiative, current=pending),
            )
            return self._replace_all(
                profiles,
                self._conversations.copy(),
                committed_at=current,
            )

    def claim_initiative_delivery(
        self,
        *,
        adapter: str,
        available_adapters: tuple[str, ...],
        now: float,
    ) -> InitiativeOpportunity | None:
        current = max(0.0, float(now))
        channel = compact_text(adapter, 16).casefold()
        available = tuple(
            value
            for item in available_adapters
            if (value := compact_text(item, 16).casefold())
            in {"popup", "discord"}
        )
        with self._lock:
            profile = self._profiles.get(OWNER_PROFILE_ID)
            if profile is None:
                return None
            initiative = _settle_initiative(profile.initiative, now=current)
            opportunity = initiative.current
            if (
                opportunity is None
                or opportunity.status != "pending_delivery"
                or opportunity.claim_token is not None
                or not opportunity.message
                or not _ordinary_delivery_allowed(
                    initiative,
                    opportunity,
                    now=current,
                )
            ):
                return None
            eligible = tuple(
                value
                for value in available
                if value not in opportunity.failed_channels
            ) or available
            selected = (
                "popup"
                if "popup" in eligible
                else "discord"
                if "discord" in eligible
                else ""
            )
            if channel != selected:
                return None
            claimed = replace(
                opportunity,
                claim_token=uuid.uuid4().hex,
                claim_expires_at=current + _INITIATIVE_CLAIM_SECONDS,
                delivery_channel=channel,
            )
            profiles = self._profiles.copy()
            profiles[OWNER_PROFILE_ID] = replace(
                profile,
                initiative=replace(initiative, current=claimed),
            )
            self._replace_all(
                profiles,
                self._conversations.copy(),
                committed_at=current,
            )
            return claimed

    def acknowledge_initiative_delivery(
        self,
        *,
        opportunity_id: str,
        claim_token: str,
        adapter: str,
        conversation_id: str,
        success: bool,
        message_id: str = "",
        now: float,
    ) -> bool:
        current = max(0.0, float(now))
        channel = compact_text(adapter, 16).casefold()
        conversation_key = compact_text(conversation_id, 160) or "popup:default"
        with self._lock:
            profile = self._profiles.get(OWNER_PROFILE_ID)
            opportunity = profile.initiative.current if profile else None
            if (
                profile is None
                or opportunity is None
                or opportunity.opportunity_id != compact_text(opportunity_id, 100)
            ):
                return False
            if opportunity.status == "sent":
                return (
                    success
                    and opportunity.delivery_channel == channel
                    and opportunity.delivery_message_id
                    == (compact_text(message_id, 160) or opportunity.opportunity_id)
                )
            if (
                opportunity.status != "pending_delivery"
                or opportunity.claim_token != claim_token
                or opportunity.delivery_channel != channel
            ):
                return False
            initiative = profile.initiative
            profiles = self._profiles.copy()
            conversations = self._conversations.copy()
            if not success:
                failed = tuple(
                    dict.fromkeys((*opportunity.failed_channels, channel))
                )
                pending = replace(
                    opportunity,
                    claim_token=None,
                    claim_expires_at=0.0,
                    delivery_channel=None,
                    failed_channels=failed,
                )
                profiles[OWNER_PROFILE_ID] = replace(
                    profile,
                    initiative=replace(initiative, current=pending),
                )
            else:
                delivery_id = (
                    compact_text(message_id, 160) or opportunity.opportunity_id
                )
                sent = replace(
                    opportunity,
                    status="sent",
                    claim_token=None,
                    claim_expires_at=0.0,
                    delivered_at=current,
                    delivery_message_id=delivery_id,
                )
                recent = (
                    *initiative.recent,
                    SentInitiative(
                        sent.topic_key,
                        sent.source_id,
                        sent.source_type,
                        current,
                    ),
                )[-_MAX_RECENT_INITIATIVES:]
                initiative = _handled_initiative(
                    replace(
                        initiative,
                        current=sent,
                        cooldown_until=(
                            initiative.cooldown_until
                            if sent.source_type == "reminder"
                            else current
                            + ORDINARY_INITIATIVE_COOLDOWN_SECONDS
                        ),
                        recent=recent,
                    ),
                    sent,
                )
                profiles[OWNER_PROFILE_ID] = replace(
                    profile,
                    initiative=initiative,
                    updated_at=max(profile.updated_at, current),
                )
                record = self._conversation(OWNER_PROFILE_ID, conversation_key)
                turn_id = f"initiative:{opportunity.opportunity_id}"
                if not any(turn.turn_id == turn_id for turn in record.recent_turns):
                    turn = ChatTurn(
                        turn_id,
                        "assistant",
                        opportunity.message or "",
                        current,
                        "initiative",
                    )
                    record = replace(
                        record,
                        recent_turns=_trim_turns((*record.recent_turns, turn)),
                        last_assistant_at=max(record.last_assistant_at, current),
                        updated_at=max(record.updated_at, current),
                    )
                    conversations[conversation_key] = record
            self._replace_all(
                profiles,
                conversations,
                committed_at=current,
            )
            callback = self._autonomy_wake
        if callback is not None:
            callback(OWNER_PROFILE_ID)
        return True

    def release_initiative_delivery(
        self,
        *,
        adapter: str,
        now: float,
    ) -> bool:
        channel = compact_text(adapter, 16).casefold()
        with self._lock:
            profile = self._profiles.get(OWNER_PROFILE_ID)
            opportunity = profile.initiative.current if profile else None
            if (
                profile is None
                or opportunity is None
                or opportunity.status != "pending_delivery"
                or opportunity.delivery_channel != channel
                or opportunity.claim_token is None
            ):
                return False
            failed = tuple(
                dict.fromkeys((*opportunity.failed_channels, channel))
            )
            pending = replace(
                opportunity,
                claim_token=None,
                claim_expires_at=0.0,
                delivery_channel=None,
                failed_channels=failed,
            )
            profiles = self._profiles.copy()
            profiles[OWNER_PROFILE_ID] = replace(
                profile,
                initiative=replace(profile.initiative, current=pending),
            )
            return self._replace_all(
                profiles,
                self._conversations.copy(),
                committed_at=max(0.0, float(now)),
            )

    def set_autonomy_wake(
        self,
        callback: Callable[[str], None] | None,
    ) -> None:
        with self._lock:
            self._autonomy_wake = callback

    @staticmethod
    def _presence_due_at(presence: PresenceState, now: float) -> float:
        unbootstrapped = (
            presence.current_activity is None
            and presence.last_decision_at == 0.0
        )
        if presence.retry_at > 0.0:
            return presence.retry_at
        return now if unbootstrapped else presence.next_decision_at

    @staticmethod
    def _presence_due(presence: PresenceState, now: float) -> bool:
        return (
            presence.claim_token is None
            and StateStore._presence_due_at(presence, now) <= now
        )

    def wake_presence_if_due(self, profile_id: str, *, now: float) -> bool:
        profile = canonical_profile_id(profile_id)
        with self._lock:
            state = self._profiles.get(profile)
            callback = self._autonomy_wake
            due = bool(
                state is not None
                and self._presence_due(state.presence, now)
            )
        if due and callback is not None:
            callback(profile)
        return due

    def presence_schedule(
        self,
        *,
        now: float,
    ) -> tuple[tuple[str, ...], float | None]:
        current = max(0.0, float(now))
        with self._lock:
            profiles = self._profiles.copy()
            changed = False
            due: list[str] = []
            wakes: list[float] = []
            for profile_id, state in self._profiles.items():
                if profile_id != OWNER_PROFILE_ID:
                    continue
                presence = state.presence
                if (
                    presence.claim_token is not None
                    and presence.claim_expires_at <= current
                ):
                    presence = replace(
                        presence,
                        claim_token=None,
                        claim_expires_at=0.0,
                        last_error="stale claim released",
                    )
                    profiles[profile_id] = replace(state, presence=presence)
                    changed = True
                if presence.claim_token is not None:
                    wakes.append(presence.claim_expires_at)
                else:
                    due_at = self._presence_due_at(presence, current)
                    if due_at <= current:
                        due.append(profile_id)
                    else:
                        wakes.append(due_at)
            if changed:
                self._replace_all(
                    profiles,
                    self._conversations.copy(),
                    committed_at=current,
                )
            return tuple(due), min(wakes, default=None)

    def claim_presence_decision(
        self,
        profile_id: str,
        *,
        now: float,
    ) -> ProfileState | None:
        profile = canonical_profile_id(profile_id)
        if profile != OWNER_PROFILE_ID:
            return None
        current = max(0.0, float(now))
        with self._lock:
            state = self._profiles.get(profile)
            if state is None or not self._presence_due(state.presence, current):
                return None
            presence = replace(
                state.presence,
                claim_token=uuid.uuid4().hex,
                claim_expires_at=current + CLAIM_SECONDS,
            )
            next_state = replace(state, presence=presence)
            profiles = self._profiles.copy()
            profiles[profile] = next_state
            self._replace_all(
                profiles,
                self._conversations.copy(),
                committed_at=current,
            )
            return next_state

    def _failed_presence(
        self,
        presence: PresenceState,
        *,
        now: float,
        error: str,
    ) -> PresenceState:
        return replace(
            presence,
            claim_token=None,
            claim_expires_at=0.0,
            retry_at=now + RETRY_SECONDS,
            last_error=compact_text(error, 120) or "life decision failed",
        )

    def fail_presence_decision(
        self,
        profile_id: str,
        *,
        claim_token: str,
        now: float,
        error: str,
    ) -> bool:
        profile = canonical_profile_id(profile_id)
        current = max(0.0, float(now))
        with self._lock:
            state = self._profiles.get(profile)
            if state is None or state.presence.claim_token != claim_token:
                return False
            next_state = replace(
                state,
                presence=self._failed_presence(
                    state.presence,
                    now=current,
                    error=error,
                ),
            )
            profiles = self._profiles.copy()
            profiles[profile] = next_state
            self._replace_all(
                profiles,
                self._conversations.copy(),
                committed_at=current,
            )
            callback = self._autonomy_wake
        if callback is not None:
            callback(profile)
        return True

    def commit_presence_decision(
        self,
        profile_id: str,
        decision: LifeDecision,
        *,
        claim_token: str,
        now: float,
        grounded_context: str,
        expected_emotion_updated_at: float = 0.0,
    ) -> tuple[bool, str]:
        profile = canonical_profile_id(profile_id)
        current = max(0.0, float(now))
        with self._lock:
            state = self._profiles.get(profile)
            if state is None or state.presence.claim_token != claim_token:
                return False, "life claim is unavailable"
            rejection = life_decision_rejection(
                state.presence,
                decision,
                grounded_context=grounded_context,
            )
            emotionally_updated = state
            if not rejection:
                activity = (
                    state.presence.current_activity.fact()
                    if decision.mode == "continue"
                    and state.presence.current_activity is not None
                    else " ".join(
                        item
                        for item in (
                            decision.activity,
                            decision.subject or "",
                            decision.detail or "",
                        )
                        if item
                    )
                )
                emotionally_updated, appraisal_valid = _apply_emotional_updates(
                    state,
                    emotion_update=decision.emotion_update,
                    mood_update=decision.mood_update,
                    evidence=f"{activity} {decision.continuation_reason or ''}",
                    source="offscreen_life",
                    source_id=claim_token,
                    now=current,
                    mood_delta_limit=0.20,
                    expected_emotion_updated_at=expected_emotion_updated_at,
                    require_complete=True,
                )
                if not appraisal_valid:
                    rejection = "life decision lacks a grounded emotional appraisal"
            if rejection:
                presence = self._failed_presence(
                    state.presence,
                    now=current,
                    error=rejection,
                )
                next_state = replace(state, presence=presence)
            else:
                presence = apply_life_decision(
                    state.presence,
                    decision,
                    now=current,
                )
                interest = validate_interest_addition(
                    decision.interest_addition,
                    activity=decision.activity,
                    subject=decision.subject,
                    detail=decision.detail,
                    existing_interests=emotionally_updated.interests,
                    grounded_context=grounded_context,
                )
                interests = (
                    _dedupe_text(
                        (*emotionally_updated.interests, interest),
                        limit=_MAX_INTERESTS,
                    )
                    if interest
                    else emotionally_updated.interests
                )
                next_state = replace(
                    emotionally_updated,
                    presence=presence,
                    interests=interests,
                    updated_at=current,
                )
            profiles = self._profiles.copy()
            profiles[profile] = next_state
            self._replace_all(
                profiles,
                self._conversations.copy(),
                committed_at=current,
            )
            callback = self._autonomy_wake
        if callback is not None:
            callback(profile)
        return not bool(rejection), rejection


_STORE_LOCK = threading.Lock()
_STATE_STORE: StateStore | None = None


def get_state_store() -> StateStore:
    global _STATE_STORE
    if _STATE_STORE is None:
        with _STORE_LOCK:
            if _STATE_STORE is None:
                _STATE_STORE = StateStore()
    return _STATE_STORE
