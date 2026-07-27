"""Normalized chat input, generation ownership, and successful-turn commits."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from app.core.config import (
    GENERATION_QUEUE_TIMEOUT_SECONDS,
    MAX_INPUT_CHARS,
    MAX_PENDING_GENERATIONS,
    MAX_TOKENS,
    PROMPT_DEBUG,
    TIMEZONE,
)
from app.core.presence import (
    activity_continuity,
    apply_activity_updates,
    advance_presence,
    format_presence_context,
    parse_life_decision,
    validate_activity_update,
    validate_next_activity,
)
from app.core.memory import (
    InternalTurnResult,
    InitiativeOpportunity,
    LongTermMemoryStore,
    Memory,
    MemoryContext,
    MemoryStore,
    WorkingMemory,
    akane_preference_answer,
    established_akane_preference,
    format_relevant_memories,
    get_internal_state_store,
    get_memory_store,
    preference_update_requested,
    relevant_relationship_context,
    relevant_akane_tastes,
)
from app.core.prompt import (
    PromptContext,
    PromptPlan,
    PromptTokenCount,
    build_prompt_plan,
)
from app.core.signal import TurnSignal, VALID_EMOTION_LABELS, topic_overlap
from app.core.utils import compact_text
from app.integrations.vscode_context import CodeContext, code_context_for_message

_TIMING_ENABLED = str(os.environ.get("AKANE_TIMING", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_COMMIT_LOCK = threading.RLock()
_PAUSE_LOCK = threading.RLock()
_PAUSED_UNTIL: dict[tuple[str, str], float] = {}
_COMPANION_DEBUG: dict[tuple[str, str], dict[str, object]] = {}
_LIFE_LOCK = threading.RLock()
_LIFE_ACTIVE: set[str] = set()
_DECISION_BLOCK = re.compile(
    r"<AKANE_DECISION>\s*(.*?)\s*</AKANE_DECISION>",
    re.DOTALL,
)
_STATE_BLOCK = re.compile(
    r"<AKANE_STATE>\s*(.*?)\s*</AKANE_STATE>",
    re.DOTALL,
)
_PREFERENCE_STANCES = {
    "likes",
    "dislikes",
    "curious",
    "mixed",
    "uncertain",
    "indifferent",
}
_EMOTION_CHANGES = {
    "started",
    "intensified",
    "sustained",
    "softened",
    "cleared",
    "replaced",
}


class GenerationBusyError(RuntimeError):
    pass


class GenerationQueueFullError(RuntimeError):
    pass


class GenerationCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CompanionDecision:
    message: str
    should_respond: bool = True
    pause_seconds: int | None = None
    should_initiate: bool = False


@dataclass(frozen=True, slots=True)
class CompanionTurnResult:
    decision: CompanionDecision
    generation_id: str = ""
    suppressed_by_pause: bool = False

    @property
    def message(self) -> str:
        return self.decision.message


@dataclass(frozen=True, slots=True)
class _ParsedCompanionOutput:
    decision: CompanionDecision
    parsed: bool

    @property
    def message(self) -> str:
        """Compatibility view of the visible portion of a model completion."""

        return self.decision.message


@dataclass(frozen=True, slots=True)
class _ParsedAkaneState:
    message: str
    preference_updates: tuple[dict[str, object], ...] = ()
    interest_additions: tuple[str, ...] = ()
    relationship_updates: tuple[dict[str, object], ...] = ()
    activity_update: dict[str, object] | None = None
    next_activity: dict[str, object] | None = None
    emotion_update: dict[str, object] | None = None
    parsed: bool = False


def parse_companion_decision(output: object) -> _ParsedCompanionOutput:
    """Strictly parse model metadata without exposing it in visible text."""

    raw = str(output or "")
    match = _DECISION_BLOCK.search(raw)
    visible_text = _DECISION_BLOCK.sub("", raw)
    # A truncated internal block is metadata too, never user-facing content.
    if "<AKANE_DECISION>" in visible_text:
        visible_text = visible_text.split("<AKANE_DECISION>", 1)[0]
    visible_text = visible_text.replace("</AKANE_DECISION>", "").strip()
    fallback = _ParsedCompanionOutput(
        CompanionDecision(message=visible_text),
        parsed=False,
    )
    if match is None or raw[: match.start()].strip():
        return fallback
    try:
        metadata = json.loads(match.group(1))
    except (TypeError, ValueError):
        return fallback
    if not isinstance(metadata, dict):
        return fallback
    if set(metadata) == {"should_initiate", "message"}:
        should_initiate = metadata["should_initiate"]
        initiative_message = metadata["message"]
        if type(should_initiate) is not bool or not isinstance(initiative_message, str):
            return fallback
        initiative_message = initiative_message.strip()
        if should_initiate and not initiative_message:
            return fallback
        state_tail = visible_text if "<AKANE_STATE>" in visible_text else ""
        return _ParsedCompanionOutput(
            CompanionDecision(
                message=(initiative_message + "\n" + state_tail).strip()
                if should_initiate
                else state_tail,
                should_respond=should_initiate,
                should_initiate=should_initiate,
            ),
            parsed=True,
        )
    if set(metadata) != {"should_respond", "pause_seconds"}:
        return fallback
    should_respond = metadata["should_respond"]
    pause_seconds = metadata["pause_seconds"]
    if type(should_respond) is not bool:
        return fallback
    if pause_seconds is not None and (type(pause_seconds) is not int):
        return fallback
    if pause_seconds is not None:
        pause_seconds = max(10, min(120, pause_seconds))
    return _ParsedCompanionOutput(
        CompanionDecision(
            message=visible_text if should_respond else "",
            should_respond=should_respond,
            pause_seconds=pause_seconds,
        ),
        parsed=True,
    )


def parse_akane_state(output: object) -> _ParsedAkaneState:
    """Strip hidden state metadata and accept only its complete, validated schema."""

    raw = str(output or "")
    matches = tuple(_STATE_BLOCK.finditer(raw))
    visible = _STATE_BLOCK.sub("", raw)
    if "<AKANE_STATE>" in visible:
        visible = visible.split("<AKANE_STATE>", 1)[0]
    visible = visible.strip()
    if len(matches) != 1:
        return _ParsedAkaneState(visible)
    try:
        metadata = json.loads(matches[0].group(1))
    except (TypeError, ValueError):
        return _ParsedAkaneState(visible)
    permitted_keys = {
        "preference_updates", "interest_additions", "relationship_updates",
        "activity_update", "next_activity", "emotion_update",
    }
    if (
        not isinstance(metadata, dict)
        or not metadata
        or not set(metadata) <= permitted_keys
    ):
        return _ParsedAkaneState(visible)
    updates = metadata.get("preference_updates", [])
    additions = metadata.get("interest_additions", [])
    relationship_updates = metadata.get("relationship_updates", [])
    if (
        not isinstance(updates, list)
        or not isinstance(additions, list)
        or not isinstance(relationship_updates, list)
    ):
        return _ParsedAkaneState(visible)
    activity_update = None
    if "activity_update" in metadata:
        activity_update = validate_activity_update(metadata["activity_update"])
        if activity_update is None:
            return _ParsedAkaneState(visible)
    next_activity = None
    if "next_activity" in metadata:
        next_activity = validate_next_activity(metadata["next_activity"])
        if next_activity is None:
            return _ParsedAkaneState(visible)
    emotion_update = None
    if "emotion_update" in metadata:
        candidate = metadata["emotion_update"]
        if not isinstance(candidate, dict) or set(candidate) != {
            "primary", "intensity", "cause", "change"
        }:
            return _ParsedAkaneState(visible)
        primary = compact_text(candidate["primary"], 32).lower()
        intensity = candidate["intensity"]
        cause = compact_text(candidate["cause"], 100)
        change = compact_text(candidate["change"], 24).lower()
        if (
            primary not in VALID_EMOTION_LABELS
            or type(intensity) not in {int, float}
            or not math.isfinite(float(intensity))
            or not cause
            or change not in _EMOTION_CHANGES
        ):
            return _ParsedAkaneState(visible)
        emotion_update = {
            "primary": primary,
            "intensity": max(0.0, min(1.0, float(intensity))),
            "cause": cause,
            "change": change,
        }

    validated_updates: list[dict[str, object]] = []
    for update in updates:
        if not isinstance(update, dict) or set(update) != {
            "topic",
            "stance",
            "strength",
            "reason",
        }:
            return _ParsedAkaneState(visible)
        topic = compact_text(update["topic"], 140)
        stance = compact_text(update["stance"], 24).lower()
        reason = compact_text(update["reason"], 240)
        strength = update["strength"]
        if (
            not topic
            or stance not in _PREFERENCE_STANCES
            or not reason
            or type(strength) not in {int, float}
            or not math.isfinite(float(strength))
        ):
            return _ParsedAkaneState(visible)
        validated_updates.append(
            {
                "topic": topic,
                "stance": stance,
                "strength": max(0.0, min(1.0, float(strength))),
                "reason": reason,
            }
        )
    validated_additions: list[str] = []
    for addition in additions:
        interest = compact_text(addition, 100)
        if not interest:
            return _ParsedAkaneState(visible)
        validated_additions.append(interest)
    validated_relationship_updates: list[dict[str, object]] = []
    for update in relationship_updates:
        if not isinstance(update, dict) or set(update) != {
            "category",
            "summary",
            "importance",
            "confidence",
        }:
            return _ParsedAkaneState(visible)
        category = compact_text(update["category"], 32).lower()
        summary = compact_text(update["summary"], 240)
        importance = update["importance"]
        confidence = update["confidence"]
        if (
            category not in {
                "pattern",
                "shared_context",
                "unresolved_event",
                "resolved_event",
            }
            or not summary
            or type(importance) not in {int, float}
            or type(confidence) not in {int, float}
            or not math.isfinite(float(importance))
            or not math.isfinite(float(confidence))
        ):
            return _ParsedAkaneState(visible)
        validated_relationship_updates.append(
            {
                "category": category,
                "summary": summary,
                "importance": max(0.0, min(1.0, float(importance))),
                "confidence": max(0.0, min(1.0, float(confidence))),
            }
        )
    return _ParsedAkaneState(
        visible,
        tuple(validated_updates),
        tuple(validated_additions),
        tuple(validated_relationship_updates),
        activity_update,
        next_activity,
        emotion_update,
        parsed=True,
    )


_DIRECT_REQUEST = re.compile(
    r"^(?:please\s+)?(?:can|could|would)\s+you\b|"
    r"^(?:please\s+)?(?:answer|check|compare|describe|explain|find|fix|give|help|"
    r"implement|list|read|review|show|summarize|tell|test|write)\b|"
    r"^(?:are|can|did|do|does|has|have|how|is|may|should|was|were|what|when|"
    r"where|which|who|why|will)\b",
    re.IGNORECASE,
)
_OPINION_REQUEST = re.compile(
    r"\b(?:do you (?:like|prefer)|favorite|what do you think|what's your opinion|"
    r"what is your opinion|your take)\b",
    re.IGNORECASE,
)
_REASSURANCE_REQUEST = re.compile(
    r"\b(?:are you sure|is that okay|is it okay|will (?:i|it|this) be okay|"
    r"will (?:i|it|this) be all right)\b",
    re.IGNORECASE,
)
_CLOSING = re.compile(
    r"^(?:bye|goodbye|good night|goodnight|got it|okay,? thanks|ok,? thanks|"
    r"thanks|thank you)[.! ]*$",
    re.IGNORECASE,
)
_SIMPLE_PRAISE = re.compile(
    r"^(?:good job|great job|nice work|well done)[.! ]*$",
    re.IGNORECASE,
)
_ELABORATION_REQUEST = re.compile(
    r"\b(?:detailed|in depth|step by step|thorough|walk me through)\b",
    re.IGNORECASE,
)
_CREATIVE_SELF_REQUEST = re.compile(
    r"\b(?:that|your)\s+(?:(?:creative|story|character)\s+)?(?:idea|concept)\b",
    re.IGNORECASE,
)
@dataclass(frozen=True, slots=True)
class ChatInput:
    profile_id: str
    conversation_id: str
    text: str
    source: str
    timestamp: float
    display_name: str = ""
    reply_context: str = ""
    autonomous: bool = False
    request_id: str = ""
    initiative_opportunity: InitiativeOpportunity | None = None


@dataclass(slots=True)
class GenerationHandle:
    generation_id: str
    conversation_id: str
    profile_id: str
    cancellation: threading.Event
    queue_deadline: float

    def raise_if_cancelled(self) -> None:
        if self.cancellation.is_set():
            raise GenerationCancelled("Generation was cancelled.")


class GenerationScheduler:
    """Bounds model waiters and permits only one active turn per conversation."""

    def __init__(self) -> None:
        self._capacity = threading.BoundedSemaphore(MAX_PENDING_GENERATIONS + 1)
        self._lock = threading.RLock()
        self._active: dict[str, GenerationHandle] = {}

    def begin(
        self,
        conversation_id: str,
        profile_id: str,
        *,
        skip_if_busy: bool = False,
    ) -> GenerationHandle:
        with self._lock:
            if skip_if_busy and self._active:
                raise GenerationBusyError("Akane is busy with another reply.")
            if conversation_id in self._active:
                raise GenerationBusyError("This conversation already has a reply in progress.")
            if any(handle.profile_id == profile_id for handle in self._active.values()):
                raise GenerationBusyError("This profile already has a reply in progress.")
        if not self._capacity.acquire(blocking=False):
            raise GenerationQueueFullError("Akane is busy; the generation queue is full.")
        handle = GenerationHandle(
            generation_id=uuid.uuid4().hex,
            conversation_id=conversation_id,
            profile_id=profile_id,
            cancellation=threading.Event(),
            queue_deadline=time.monotonic() + GENERATION_QUEUE_TIMEOUT_SECONDS,
        )
        with self._lock:
            current = self._active.get(conversation_id)
            profile_active = any(
                active.profile_id == profile_id for active in self._active.values()
            )
            if current is not None or profile_active:
                self._capacity.release()
                raise GenerationBusyError("This conversation or profile already has a reply in progress.")
            self._active[conversation_id] = handle
        return handle

    def finish(self, handle: GenerationHandle) -> None:
        with self._lock:
            if self._active.get(handle.conversation_id) is not handle:
                return
            self._active.pop(handle.conversation_id, None)
        self._capacity.release()

    def cancel(self, conversation_id: str, profile_id: str | None = None) -> bool:
        with self._lock:
            handle = self._active.get(conversation_id)
            if handle is None or (profile_id is not None and handle.profile_id != profile_id):
                return False
            handle.cancellation.set()
            return True

    def cancel_all(self) -> None:
        with self._lock:
            for handle in self._active.values():
                handle.cancellation.set()

    def cancel_profile(self, profile_id: str) -> None:
        with self._lock:
            for handle in self._active.values():
                if handle.profile_id == profile_id:
                    handle.cancellation.set()

    def active_generation_id(self, conversation_id: str) -> str:
        with self._lock:
            handle = self._active.get(conversation_id)
            return handle.generation_id if handle else ""

    def is_active(self, conversation_id: str, profile_id: str) -> bool:
        with self._lock:
            return conversation_id in self._active or any(
                handle.profile_id == profile_id for handle in self._active.values()
            )


@dataclass(frozen=True, slots=True)
class ResponseIntention:
    """One deterministic response goal with at most one companion behavior."""

    primary: str
    optional_behavior: str = "none"
    continuity: str = "none"
    grounding: str = "not required"
    length: str = "concise"
    question_permitted: bool = False
    callback_permitted: bool = False
    grounded_detail_permitted: bool = False
    suppression_reasons: tuple[str, ...] = ()
    direct_request: bool = False
    active_thread: bool = False
    relationship_safe: bool = False
    correction_active: bool = False


@dataclass(frozen=True, slots=True)
class CompiledStyle:
    """Discrete delivery state compiled into bounded, non-dialogue directives."""

    humor: str
    directives: tuple[tuple[str, str], ...]
    validation_limits: tuple[int, int] = (0, 0)
    question_gate: str = "closed"

    def prompt_text(self) -> str:
        return "\n".join(f"{category}: {value}" for category, value in self.directives)


@dataclass(frozen=True, slots=True)
class CoordinatedTurnContext:
    """Immutable, prompt-ready view of one coordinated turn."""

    state_delta: InternalTurnResult
    memory_context: MemoryContext
    behavioral_summary: str
    relationship_context: str = ""
    preference_context: str = ""
    taste_context: str = ""
    relevant_memories: str = ""
    life_context: str = ""
    external_context: str = ""
    reply_context: str = ""
    date_time: str = ""
    activity_continuity: str = "none"
    current_activity_started_at: float | None = None
    last_assistant_turn_at: float | None = None
    activity_detail_grounded: str = "not applicable"
    configured_timezone: str = TIMEZONE
    current_local_time: str = ""
    current_daypart: str = ""
    preference_anchor: str = ""
    initiative_worthwhile: bool = True
    initiative_opportunity: InitiativeOpportunity | None = None
    response_intention: ResponseIntention = ResponseIntention("acknowledge")
    compiled_style: CompiledStyle = CompiledStyle(
        "dry",
        (("Goal", "acknowledge"), ("Length", "concise")),
    )

    @property
    def signal(self) -> TurnSignal:
        return self.state_delta.signal


def _recent_question_pending(recent_turns: tuple[object, ...]) -> bool:
    """Avoid turning consecutive turns into an interview."""

    latest_assistant = next(
        (
            str(getattr(turn, "content", "") or "").strip()
            for turn in reversed(recent_turns)
            if getattr(turn, "role", "") == "assistant"
        ),
        "",
    )
    return latest_assistant.rstrip().endswith("?")


def select_response_intention(
    signal: TurnSignal,
    memories: tuple[Memory, ...] = (),
    memory_uses: tuple[tuple[str, str], ...] = (),
    *,
    user_text: str = "",
    familiar_relationship: bool = False,
    has_grounded_activity: bool = False,
    recent_turns: tuple[object, ...] = (),
) -> ResponseIntention:
    """Select one response purpose from already-computed, structured turn state."""

    text = str(user_text or signal.summary or "").strip()
    request_segments = tuple(
        segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()
    )
    explicit_request = bool(
        "?" in text or any(_DIRECT_REQUEST.search(segment) for segment in request_segments)
    )
    arcane_activity_update = bool(
        signal.semantic_event.event_type == "activity"
        and signal.semantic_event.actor in {"Arcane", "shared"}
        and signal.semantic_event.temporal_state == "current"
    )
    use_by_id = dict(memory_uses)
    used_memories = tuple(memory for memory in memories if memory.id in use_by_id)
    active_thread = "thread" in use_by_id.values()
    trusted_experience = any(
        use_by_id.get(memory.id) == "self_experience"
        and memory.source_type
        in {"explicit_user", "recorded_offscreen", "verified_interface", "trusted_memory"}
        for memory in used_memories
    )
    correction_active = bool(
        signal.correction_requested or "correction" in use_by_id.values()
    )
    opinion_requested = bool(
        signal.identity_attribute == "preferences" or _OPINION_REQUEST.search(text)
    )
    creative_experience_requested = bool(_CREATIVE_SELF_REQUEST.search(text))
    experience_requested = bool(
        signal.current_activity or creative_experience_requested
    )
    reassurance_requested = bool(_REASSURANCE_REQUEST.search(text))
    generic_request = bool(
        not signal.low_content
        and not opinion_requested
        and not reassurance_requested
        and (
            (
                not experience_requested
                and (
                    signal.intent
                    in {"technical", "instruction", "correction", "criticism", "identity"}
                    or signal.technical
                    or signal.code_context_requested
                    or explicit_request
                )
            )
            or (experience_requested and not (trusted_experience or has_grounded_activity))
        )
        and (not arcane_activity_update or explicit_request)
    )
    distress = bool(signal.sadness or signal.intent == "emotional_support")
    substantive_objection = signal.contextual_reaction.kind == "disagreement"
    grounded_experience = experience_requested and (
        trusted_experience or has_grounded_activity
    )
    meaningful_success = bool(
        signal.task_success
        and not generic_request
        and not (signal.praise and _SIMPLE_PRAISE.fullmatch(text))
    )
    supported_reassurance = bool(
        reassurance_requested
        or (
            signal.contextual_reaction.kind == "concerned"
            and not distress
            and not generic_request
        )
    )
    closing = bool(_CLOSING.fullmatch(text))

    candidates = (
        ("comfort", distress),
        ("answer", generic_request),
        ("disagree", substantive_objection),
        ("celebrate", meaningful_success),
        ("share experience", grounded_experience),
        ("continue thread", active_thread and not generic_request),
        ("state opinion", opinion_requested),
        ("reassure", supported_reassurance),
        ("acknowledge", not closing),
        ("remain brief", True),
    )
    eligible = tuple(name for name, matches in candidates if matches)
    primary = eligible[0]

    serious = bool(
        primary in {"comfort", "disagree", "reassure"}
        or signal.intent == "serious"
        or signal.sadness
        or signal.hostility
    )
    affect_kind = signal.contextual_reaction.kind
    explicit_playfulness = bool(signal.teasing or signal.intent == "teasing")
    affect_permits_playfulness = bool(
        explicit_playfulness
        or affect_kind in {"amusement", "playfulness", "lingering amusement"}
    )
    relationship_safe = bool(
        not serious
        and not signal.hostility
        and not signal.sadness
    )
    teasing_allowed = bool(
        relationship_safe
        and familiar_relationship
        and not signal.technical
        and affect_permits_playfulness
        and explicit_playfulness
    )
    callback_allowed = bool(
        relationship_safe
        and familiar_relationship
        and "callback" in use_by_id.values()
        and not signal.technical
    )
    question_allowed = not (
        correction_active
        or serious
        or generic_request
        or signal.technical
        or signal.continued_after_objection
        or _CLOSING.fullmatch(text)
        or _recent_question_pending(recent_turns)
    )

    suppression: list[str] = []
    if serious:
        suppression.append("serious context")
    if correction_active:
        suppression.append("active correction")
    if generic_request or signal.technical:
        suppression.append("direct task")
    if experience_requested and not grounded_experience:
        suppression.append("grounding uncertainty")

    optional_behavior = "none"
    optional_suppressed = bool(
        serious
        or correction_active
        or generic_request
        or signal.technical
        or primary in {"comfort", "disagree", "reassure", "remain brief"}
    )
    if not optional_suppressed:
        if primary == "share experience" and grounded_experience:
            optional_behavior = "grounded personal detail"
        elif callback_allowed and primary in {"acknowledge", "celebrate", "continue thread"}:
            optional_behavior = "brief callback"
        elif primary == "state opinion":
            optional_behavior = "brief opinion"
        elif teasing_allowed or affect_kind in {"amusement", "playfulness", "lingering_amusement"}:
            optional_behavior = "light dry humor"
        elif primary in {"acknowledge", "celebrate", "continue thread"} and not signal.low_content:
            optional_behavior = "relevant observation"

    grounding = "not required"
    if experience_requested:
        if creative_experience_requested and grounded_experience:
            grounding = "stored creative premise"
        elif trusted_experience:
            grounding = "trusted self-memory"
        elif has_grounded_activity:
            grounding = "recorded activity"
        else:
            grounding = "no invented activity"
    elif correction_active:
        grounding = "apply correction silently"

    continuity = (
        "active thread"
        if active_thread
        else "brief callback available"
        if callback_allowed
        else "emotional context"
        if "emotional_context" in use_by_id.values()
        else "none"
    )
    length = (
        "minimal"
        if primary == "remain brief"
        else "task-complete"
        if primary == "answer" and (signal.technical or len(text.split()) >= 24 or "\n" in text)
        else "concise"
    )

    return ResponseIntention(
        primary=primary,
        optional_behavior=optional_behavior,
        continuity=continuity,
        grounding=grounding,
        length=length,
        question_permitted=question_allowed,
        callback_permitted=callback_allowed,
        grounded_detail_permitted=grounded_experience,
        suppression_reasons=tuple(dict.fromkeys(suppression)),
        direct_request=generic_request,
        active_thread=active_thread,
        relationship_safe=relationship_safe,
        correction_active=correction_active,
    )


def compile_akane_style(
    signal: TurnSignal,
    intention: ResponseIntention,
    *,
    user_text: str = "",
    familiar_relationship: bool = False,
) -> CompiledStyle:
    """Compile existing turn state into one bounded model-visible style contract."""

    humor = "dry"
    correction = intention.correction_active
    callback_permission = intention.callback_permitted
    self_experience = intention.grounded_detail_permitted

    grounding: list[str] = []
    avoid = [
        "service posture",
        "forced follow-up",
        "generic validation",
        "internal-state narration",
        "excessive hedging",
        "unnecessary offers",
    ]
    primary = intention.primary

    if signal.identity_attribute:
        grounding.append("stable identity facts")
    if correction:
        grounding.append("apply supplied correction silently")
        avoid.append("correction commentary")
    if intention.grounding == "stored creative premise":
        grounding.extend(
            (
                "stored premise",
                "bounded fictional elaboration",
                "no external-research claim",
            )
        )
        avoid.append("invented past activity")
    elif primary == "share experience" or self_experience:
        grounding.extend(("recorded details only", "uncertainty for gaps"))
        avoid.append("invented specifics")
    elif intention.grounding == "no invented activity":
        grounding.append("no unrecorded activity claim")
    if signal.code_context_requested and not signal.code_context_attached:
        grounding.append("available context only")
        avoid.append("access claims")

    if primary == "acknowledge":
        grounding.append("stated update only")
        avoid.extend(("unsupported assumptions", "invented significance"))
    elif primary == "comfort":
        avoid.extend(("generic validation", "formal support language"))
    elif primary == "disagree":
        avoid.append("excessive hedging")
    elif primary == "state opinion":
        avoid.append("false certainty")
    elif primary == "continue thread":
        avoid.append("full-history recap")
    elif primary == "reassure":
        avoid.extend(("unsupported certainty", "formal support language"))
    if primary in {"answer", "comfort", "disagree", "reassure", "remain brief"}:
        humor = "none"
    if intention.optional_behavior == "light dry humor":
        humor = "dry"

    reaction = signal.contextual_reaction.kind
    if reaction == "concerned":
        humor = "none"
    elif reaction in {"relief", "satisfaction", "pride", "appreciation"}:
        if familiar_relationship and intention.relationship_safe:
            humor = "dry"
    elif reaction in {"amusement", "playfulness"}:
        if intention.relationship_safe and not signal.technical and familiar_relationship:
            humor = "dry"
    elif reaction == "criticism":
        humor = "none"
    elif reaction == "embarrassed":
        avoid.extend(("exaggerated mannerisms", "forced hesitation"))

    serious = bool(
        primary in {"comfort", "disagree", "reassure"}
        or signal.intent == "serious"
        or signal.sadness
        or signal.hostility
        or reaction == "concerned"
    )
    if serious:
        humor = "none"

    tension = bool(signal.hostility)
    if tension:
        humor = "none"
        avoid.append("callbacks")

    final_callback_open = bool(
        callback_permission
        and familiar_relationship
        and not serious
        and not tension
        and not signal.technical
    )
    if not final_callback_open:
        avoid.append("callbacks")

    if serious or tension or signal.technical or primary == "answer":
        humor = "none"

    text = str(user_text or signal.summary or "")
    detailed = bool(_ELABORATION_REQUEST.search(text))
    task_complex = bool(signal.technical or len(text.split()) >= 24 or "\n" in text)
    if primary == "remain brief":
        length = intention.length
        limits = (1, 1)
    elif detailed:
        length = "detailed as requested"
        limits = (0, 0)
    elif intention.length == "task-complete" or primary in {"answer", "continue thread"} and task_complex:
        length = "task-complete"
        limits = (0, 0)
    else:
        length = "concise"
        limits = (1, 4)

    situation = (
        "direct technical request"
        if signal.technical
        else "creative follow-up"
        if intention.grounding == "stored creative premise"
        else "direct request"
        if intention.direct_request
        else "Akane activity question"
        if signal.current_activity
        else "user completion"
        if signal.semantic_event.confirmed_completion
        else "emotional disclosure"
        if signal.sadness
        else "active topic continuation"
        if intention.active_thread
        else "casual update"
    )
    reaction_guidance = (
        "serious and restrained"
        if serious
        else reaction.replace("_", " ")
        if reaction not in {"", "neutral"}
        else "natural companion participation"
    )
    directives: list[tuple[str, str]] = [
        ("Situation", situation),
        ("Reaction", reaction_guidance),
        ("Goal", primary),
    ]
    if intention.optional_behavior != "none":
        directives.append(("Optional", intention.optional_behavior))
    if intention.continuity != "none":
        directives.append(("Continuity", intention.continuity))
    if grounding:
        directives.append(("Grounding", "; ".join(dict.fromkeys(grounding))))
    directives.append(("Length", length))
    directives.append(("Avoid", "; ".join(dict.fromkeys(avoid))))
    directives = [(category, value) for category, value in directives if value]
    return CompiledStyle(
        humor=humor,
        directives=tuple(directives),
        validation_limits=limits,
        question_gate=(
            "open"
            if intention.question_permitted
            else "closed"
        ),
    )


class InternalStateCoordinator:
    """Coordinates existing state domains without taking ownership from them."""

    def __init__(
        self,
        conversation_store: MemoryStore,
        state_store: LongTermMemoryStore,
    ) -> None:
        self._conversation_store = conversation_store
        self._state_store = state_store

    def prepare(
        self,
        chat: ChatInput,
        *,
        skip_memory: bool = False,
    ) -> CoordinatedTurnContext:
        last_assistant_at = self._conversation_store.last_assistant_turn_at(
            chat.profile_id
        )
        code_context = (
            code_context_for_message(chat.text)
            if chat.source in {"popup", "discord"} and not chat.autonomous
            else CodeContext(requested=False, connected=False)
        )
        memory_context = self._conversation_store.build_context(
            chat.profile_id,
            chat.conversation_id,
            display_name=chat.display_name,
            query=chat.text,
            include_memory=not skip_memory
            and (not chat.autonomous or chat.initiative_opportunity is not None),
        )
        familiar_relationship = bool(memory_context.recent_turns)
        conversation_working = WorkingMemory(
            current_topic=memory_context.current_topic,
            current_task=memory_context.current_task,
            unresolved_problem=memory_context.unresolved_problem,
            repeated_topic_count=memory_context.repeated_topic_count,
            last_outcome=memory_context.last_outcome,
        )
        state_delta = self._state_store.preview_turn(
            chat.profile_id,
            chat.text,
            now=chat.timestamp,
            include_memory=not skip_memory,
            code_context_requested=code_context.requested,
            code_context_attached=bool(code_context.prompt_text),
            autonomous=chat.autonomous,
            familiar_relationship=familiar_relationship,
            working_context=conversation_working,
            recent_turns=memory_context.recent_turns,
            activity_scope=chat.conversation_id,
        )
        current_activity = state_delta.state.presence.current_activity
        continuity = activity_continuity(current_activity, last_assistant_at)
        current_local_time, current_daypart = local_time_context(chat.timestamp)
        signal = state_delta.signal
        stable_identity = signal.identity_attribute in {
            "identity",
            "appearance",
            "relationships",
        } and not signal.activity_timeframe
        working = state_delta.working_context
        preference_memory = (
            established_akane_preference(
                state_delta.state.memories,
                chat.text,
                now=chat.timestamp,
            )
            if not skip_memory and signal.identity_attribute == "preferences"
            else None
        )
        preference_change_allowed = (
            signal.identity_attribute == "preferences"
            and preference_update_requested(chat.text)
        )
        preference_anchor = (
            "" if preference_change_allowed else akane_preference_answer(preference_memory)
        )
        selected_memories = (
            []
            if stable_identity
            else list(state_delta.recalled_memories)
        )
        if preference_memory is not None and all(
            memory.id != preference_memory.id for memory in selected_memories
        ):
            selected_memories.insert(0, preference_memory)
        relevant_memories = tuple(
            memory
            for memory in selected_memories
            if preference_memory is None or memory.id != preference_memory.id
        )
        memory_context_text = format_relevant_memories(
            relevant_memories,
            state_delta.memory_uses,
        )
        long_term_relationship_context = (
            ""
            if skip_memory
            else relevant_relationship_context(state_delta.state, chat.text)
        )
        taste_context = (
            ""
            if skip_memory or stable_identity
            else relevant_akane_tastes(state_delta.state, chat.text)
        )
        memory_context = replace(
            memory_context,
            memory_ids=tuple(memory.id for memory in selected_memories),
            memory_contents=tuple(
                compact_text(memory.content, 120) for memory in selected_memories
            ),
        )
        response_intention = select_response_intention(
            signal,
            state_delta.recalled_memories,
            state_delta.memory_uses,
            user_text=chat.text,
            familiar_relationship=familiar_relationship,
            has_grounded_activity=state_delta.grounded_activity_source != "none",
            recent_turns=memory_context.recent_turns,
        )
        compiled_style = compile_akane_style(
            signal,
            response_intention,
            user_text=chat.text,
            familiar_relationship=familiar_relationship,
        )
        presence_planning_available = (
            state_delta.state.presence.current_activity is None
            and state_delta.state.presence.next_activity is None
        )

        behavioral_summary = "\n".join(
            value
            for value in (
                compiled_style.prompt_text(),
                signal.emotion_prompt(
                    emotion_relevant=_emotion_topic_relevant(
                        signal,
                        conversation_working.current_topic,
                        state_delta.state.emotion.cause,
                    )
                ),
                (
                    "Akane chose silence on the previous turn. This is factual "
                    "continuity, not spoken dialogue or a response instruction."
                    if conversation_working.last_outcome == "akane_silence"
                    else ""
                ),
                (
                    "Akane currently has no active or scheduled activity. She may choose "
                    "a grounded activity or quiet downtime based on her interests and "
                    "preferences. Record an activity choice through activity metadata only "
                    "when she genuinely wants one."
                    if presence_planning_available
                    else ""
                ),
                (
                    "A grounded initiative opportunity is available:\n"
                    f"Reason: {chat.initiative_opportunity.reason}\n"
                    f"Context: {chat.initiative_opportunity.context}\n"
                    "Akane decides whether to speak. She may decline. If she initiates, "
                    "use <AKANE_DECISION>{\"should_initiate\":true,\"message\":"
                    "\"her exact message\"}</AKANE_DECISION>; otherwise use false and "
                    "an empty message. This is factual context, not a required question."
                    if chat.initiative_opportunity is not None
                    else ""
                ),
            )
            if value
        )

        editor_context = code_context.prompt_text
        if code_context.requested and not code_context.connected:
            editor_context = (
                "The requested editor context is unavailable. Do not claim to have inspected a file."
            )
        initiative_worthwhile = chat.initiative_opportunity is not None or not chat.autonomous or any(
            (
                bool(selected_memories),
                bool(working.unresolved_problem),
                bool(preference_memory),
            )
        )
        return CoordinatedTurnContext(
            state_delta=state_delta,
            memory_context=memory_context,
            behavioral_summary=behavioral_summary,
            relationship_context="\n\n".join(
                value
                for value in (
                    "" if stable_identity else memory_context.relationship,
                    long_term_relationship_context,
                )
                if value
            ),
            preference_context=(
                ""
                if stable_identity
                else _preference_continuity(
                    preference_memory.content if preference_memory else "",
                    preference_change_allowed,
                )
            ),
            taste_context=taste_context,
            relevant_memories=(
                "" if stable_identity else memory_context_text
            ),
            life_context=(
                format_presence_context(
                    state_delta.state.presence,
                    interests=state_delta.state.interests,
                    emotion=state_delta.state.emotion,
                    timeframe=signal.activity_timeframe,
                    now=chat.timestamp,
                    continuity=continuity,
                    local_time=current_local_time,
                    daypart=current_daypart,
                )
                if not stable_identity
                else ""
            ),
            external_context="" if stable_identity else editor_context,
            reply_context="" if stable_identity else chat.reply_context,
            date_time=(
                date_time_line(chat.timestamp)
                if _time_context_relevant(chat.text)
                else ""
            ),
            activity_continuity=continuity,
            current_activity_started_at=(
                current_activity.started_at if current_activity else None
            ),
            last_assistant_turn_at=last_assistant_at,
            activity_detail_grounded=(
                current_activity.detail or "none recorded"
                if current_activity
                else "not applicable"
            ),
            configured_timezone=TIMEZONE,
            current_local_time=current_local_time,
            current_daypart=current_daypart,
            preference_anchor=preference_anchor,
            initiative_worthwhile=initiative_worthwhile,
            initiative_opportunity=chat.initiative_opportunity,
            response_intention=response_intention,
            compiled_style=compiled_style,
        )

    def commit_completed_turn(
        self,
        chat: ChatInput,
        turn: CoordinatedTurnContext,
        reply: str,
        *,
        preference_updates: tuple[dict[str, object], ...] = (),
        interest_additions: tuple[str, ...] = (),
        relationship_updates: tuple[dict[str, object], ...] = (),
        activity_update: dict[str, object] | None = None,
        next_activity: dict[str, object] | None = None,
        emotion_update: dict[str, object] | None = None,
    ) -> None:
        previous_state = self._state_store.commit_turn(
            chat.profile_id,
            turn.state_delta,
            used_memory_ids=turn.memory_context.memory_ids,
            preference_updates=preference_updates,
            interest_additions=interest_additions,
            relationship_updates=relationship_updates,
            activity_update=activity_update,
            next_activity=next_activity,
            emotion_update=emotion_update,
            now=chat.timestamp,
        )
        if chat.autonomous:
            return
        try:
            self._conversation_store.commit_turn(
                profile_id=chat.profile_id,
                conversation_id=chat.conversation_id,
                source=chat.source,
                user_text=chat.text,
                assistant_text=reply,
                signal=turn.signal,
                request_id=chat.request_id,
            )
        except Exception:
            self._state_store.restore_internal_state(chat.profile_id, previous_state)
            raise

    def commit_silent_turn(
        self,
        chat: ChatInput,
        turn: CoordinatedTurnContext,
        *,
        preference_updates: tuple[dict[str, object], ...] = (),
        interest_additions: tuple[str, ...] = (),
        relationship_updates: tuple[dict[str, object], ...] = (),
        activity_update: dict[str, object] | None = None,
        next_activity: dict[str, object] | None = None,
        emotion_update: dict[str, object] | None = None,
    ) -> None:
        """Commit model-chosen state without creating a fictional chat turn."""

        previous_state = self._state_store.commit_turn(
            chat.profile_id,
            turn.state_delta,
            used_memory_ids=turn.memory_context.memory_ids,
            preference_updates=preference_updates,
            interest_additions=interest_additions,
            relationship_updates=relationship_updates,
            activity_update=activity_update,
            next_activity=next_activity,
            emotion_update=emotion_update,
            now=chat.timestamp,
        )
        if chat.autonomous:
            return
        try:
            self._conversation_store.record_silence(
                profile_id=chat.profile_id,
                conversation_id=chat.conversation_id,
                signal=turn.signal,
                request_id=chat.request_id,
            )
        except Exception:
            self._state_store.restore_internal_state(chat.profile_id, previous_state)
            raise

    def commit_initiative_turn(
        self,
        chat: ChatInput,
        turn: CoordinatedTurnContext,
        *,
        message: str,
        used: bool,
        preference_updates: tuple[dict[str, object], ...] = (),
        interest_additions: tuple[str, ...] = (),
        relationship_updates: tuple[dict[str, object], ...] = (),
        activity_update: dict[str, object] | None = None,
        next_activity: dict[str, object] | None = None,
        emotion_update: dict[str, object] | None = None,
    ) -> None:
        opportunity = chat.initiative_opportunity
        if opportunity is None:
            raise ValueError("Initiative commit requires an opportunity.")
        previous_state = self._state_store.commit_turn(
            chat.profile_id,
            turn.state_delta,
            used_memory_ids=turn.memory_context.memory_ids,
            preference_updates=preference_updates,
            interest_additions=interest_additions,
            relationship_updates=relationship_updates,
            activity_update=activity_update,
            next_activity=next_activity,
            emotion_update=emotion_update,
            now=chat.timestamp,
        )
        try:
            self._conversation_store.record_initiative_result(
                profile_id=chat.profile_id,
                conversation_id=chat.conversation_id,
                opportunity=opportunity,
                message=message,
                used=used,
                cooldown_seconds=24 * 60 * 60,
                now=chat.timestamp,
            )
        except Exception:
            self._state_store.restore_internal_state(chat.profile_id, previous_state)
            raise


@dataclass(frozen=True, slots=True)
class TurnPreparation:
    chat_input: ChatInput
    prompt_plan: PromptPlan
    turn_context: CoordinatedTurnContext
    coordinator: InternalStateCoordinator
    handle: GenerationHandle
    max_tokens: int
    started_at: float
    prompt_seconds: float = 0.0
    preprocess_seconds: float = 0.0
    memory_seconds: float = 0.0
    code_context_attached: bool = False

    @property
    def session_id(self) -> str:
        return self.chat_input.conversation_id

    @property
    def generation_id(self) -> str:
        return self.handle.generation_id

    @property
    def memory_context(self) -> MemoryContext:
        return self.turn_context.memory_context

    @property
    def internal_turn(self) -> InternalTurnResult:
        return self.turn_context.state_delta

def normalize_chat_input(
    *,
    text: object,
    profile_id: object = "local:owner",
    conversation_id: object = "popup:default",
    source: object = "popup",
    timestamp: object = 0.0,
    display_name: object = "",
    reply_context: object = "",
    autonomous: object = False,
    request_id: object = "",
) -> ChatInput:
    message = str(text or "").strip()
    if not message:
        raise ValueError("Message is empty.")
    if len(message) > MAX_INPUT_CHARS:
        raise ValueError(f"Message exceeds the {MAX_INPUT_CHARS}-character limit.")
    try:
        created_at = float(timestamp or time.time())
    except (TypeError, ValueError):
        created_at = time.time()
    normalized_source = compact_text(source, 24).lower() or "popup"
    if normalized_source not in {"popup", "discord", "web"}:
        normalized_source = "web"
    return ChatInput(
        profile_id=compact_text(profile_id, 120) or "local:owner",
        conversation_id=compact_text(conversation_id, 120) or "popup:default",
        text=message,
        source=normalized_source,
        timestamp=created_at,
        display_name=compact_text(display_name, 60),
        reply_context=compact_text(reply_context, 600),
        autonomous=bool(autonomous),
        request_id=compact_text(request_id, 160),
    )


def prepare_turn(
    chat_input: ChatInput | str,
    *,
    session_id: str | None = None,
    skip_memory: bool = False,
    skip_if_busy: bool = False,
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None = None,
) -> TurnPreparation:
    started_at = time.perf_counter()
    chat = (
        normalize_chat_input(
            text=chat_input,
            conversation_id=session_id or "popup:default",
        )
        if isinstance(chat_input, str)
        else chat_input
    )
    handle = _SCHEDULER.begin(
        chat.conversation_id,
        chat.profile_id,
        skip_if_busy=skip_if_busy,
    )
    try:
        memory_started_at = time.perf_counter()
        coordinator = InternalStateCoordinator(
            get_memory_store(),
            get_internal_state_store(),
        )
        turn_context = coordinator.prepare(chat, skip_memory=skip_memory)
        memory_seconds = time.perf_counter() - memory_started_at
        prompt_started_at = time.perf_counter()
        prompt_plan = build_prompt_plan(
            chat.text,
            PromptContext(
                relationship=turn_context.relationship_context,
                preference_continuity=turn_context.preference_context,
                taste_context=turn_context.taste_context,
                relevant_memories=turn_context.relevant_memories,
                durable_memories=tuple(
                    turn_context.memory_context.memory_contents
                ),
                earlier_turns=turn_context.memory_context.earlier_turns,
                recent_turns=turn_context.memory_context.recent_turns,
                behavioral_summary=turn_context.behavioral_summary,
                life_context=turn_context.life_context,
                date_time=turn_context.date_time,
                reply_context=turn_context.reply_context,
                external_context=turn_context.external_context,
            ),
            token_counter=token_counter,
        )
        prompt_seconds = time.perf_counter() - prompt_started_at
        submitted_memory = "\n".join(
            source.content
            for source in prompt_plan.sources
            if source.kind in {"durable_memory", "preference_continuity"}
        )
        included_ids = tuple(
            memory_id
            for memory_id, content in zip(
                turn_context.memory_context.memory_ids,
                turn_context.memory_context.memory_contents,
            )
            if content in submitted_memory
        )
        turn_context = replace(
            turn_context,
            memory_context=replace(
                turn_context.memory_context,
                memory_ids=included_ids,
            ),
        )
        if PROMPT_DEBUG:
            metadata = prompt_plan.debug_metadata()
            print(f"[Akane:prompt] {metadata}", flush=True)
        handle.raise_if_cancelled()
        return TurnPreparation(
            chat_input=chat,
            prompt_plan=prompt_plan,
            turn_context=turn_context,
            coordinator=coordinator,
            handle=handle,
            max_tokens=MAX_TOKENS,
            started_at=started_at,
            prompt_seconds=prompt_seconds,
            preprocess_seconds=time.perf_counter() - started_at,
            memory_seconds=memory_seconds,
            code_context_attached=bool(turn_context.external_context),
        )
    except Exception:
        _SCHEDULER.finish(handle)
        raise


def commit_turn(
    prepared: TurnPreparation,
    reply: str,
    *,
    preference_updates: tuple[dict[str, object], ...] = (),
    interest_additions: tuple[str, ...] = (),
    relationship_updates: tuple[dict[str, object], ...] = (),
    activity_update: dict[str, object] | None = None,
    next_activity: dict[str, object] | None = None,
    emotion_update: dict[str, object] | None = None,
) -> None:
    prepared.handle.raise_if_cancelled()
    with _COMMIT_LOCK:
        prepared.handle.raise_if_cancelled()
        prepared.coordinator.commit_completed_turn(
            prepared.chat_input,
            prepared.turn_context,
            reply,
            preference_updates=preference_updates,
            interest_additions=interest_additions,
            relationship_updates=relationship_updates,
            activity_update=activity_update,
            next_activity=next_activity,
            emotion_update=emotion_update,
        )


def commit_silent_turn(
    prepared: TurnPreparation,
    *,
    preference_updates: tuple[dict[str, object], ...] = (),
    interest_additions: tuple[str, ...] = (),
    relationship_updates: tuple[dict[str, object], ...] = (),
    activity_update: dict[str, object] | None = None,
    next_activity: dict[str, object] | None = None,
    emotion_update: dict[str, object] | None = None,
) -> None:
    prepared.handle.raise_if_cancelled()
    with _COMMIT_LOCK:
        prepared.handle.raise_if_cancelled()
        prepared.coordinator.commit_silent_turn(
            prepared.chat_input,
            prepared.turn_context,
            preference_updates=preference_updates,
            interest_additions=interest_additions,
            relationship_updates=relationship_updates,
            activity_update=activity_update,
            next_activity=next_activity,
            emotion_update=emotion_update,
        )


def commit_initiative_turn(
    prepared: TurnPreparation,
    *,
    message: str,
    used: bool,
    preference_updates: tuple[dict[str, object], ...] = (),
    interest_additions: tuple[str, ...] = (),
    relationship_updates: tuple[dict[str, object], ...] = (),
    activity_update: dict[str, object] | None = None,
    next_activity: dict[str, object] | None = None,
    emotion_update: dict[str, object] | None = None,
) -> None:
    prepared.handle.raise_if_cancelled()
    with _COMMIT_LOCK:
        prepared.handle.raise_if_cancelled()
        prepared.coordinator.commit_initiative_turn(
            prepared.chat_input,
            prepared.turn_context,
            message=message,
            used=used,
            preference_updates=preference_updates,
            interest_additions=interest_additions,
            relationship_updates=relationship_updates,
            activity_update=activity_update,
            next_activity=next_activity,
            emotion_update=emotion_update,
        )


def finish_turn(prepared: TurnPreparation) -> None:
    _SCHEDULER.finish(prepared.handle)


def _emotion_topic_relevant(
    signal: TurnSignal,
    previous_topic: str,
    emotion_cause: str,
) -> bool | None:
    """Keep old emotion factual while lowering its prominence after a topic shift."""

    if signal.continued_after_objection or (
        signal.embodied_action and signal.repetition_count > 1
    ):
        return True
    if not signal.topic:
        return None
    references = (previous_topic, emotion_cause)
    available = tuple(reference for reference in references if reference)
    if not available:
        return None
    return any(topic_overlap(reference, signal.topic) >= 0.40 for reference in available)


def _pause_key(chat: ChatInput) -> tuple[str, str]:
    return (chat.profile_id, chat.conversation_id)


def _pause_remaining(chat: ChatInput, *, now: float | None = None) -> float:
    current = time.time() if now is None else now
    key = _pause_key(chat)
    with _PAUSE_LOCK:
        until = _PAUSED_UNTIL.get(key, 0.0)
        if until <= current:
            _PAUSED_UNTIL.pop(key, None)
            return 0.0
        return until - current


def _apply_pause(chat: ChatInput, pause_seconds: int | None) -> None:
    if pause_seconds is None:
        return
    with _PAUSE_LOCK:
        _PAUSED_UNTIL[_pause_key(chat)] = time.time() + pause_seconds


def _record_companion_debug(
    chat: ChatInput,
    *,
    decision_parsed: bool,
    decision: CompanionDecision,
    message_suppressed: bool,
) -> None:
    remaining = _pause_remaining(chat)
    with _PAUSE_LOCK:
        _COMPANION_DEBUG[_pause_key(chat)] = {
            "decision_parsed": decision_parsed,
            "should_respond": decision.should_respond,
            "pause_seconds": decision.pause_seconds,
            "currently_paused": bool(remaining),
            "message_suppressed": message_suppressed,
        }


def _presence_debug(
    prepared: TurnPreparation,
    raw_reply: str,
    state: _ParsedAkaneState,
) -> dict[str, object]:
    """Expose compact, factual presence diagnostics for a completed turn."""

    turn_context = getattr(prepared, "turn_context", None)
    turn = getattr(turn_context, "state_delta", None)
    before = getattr(getattr(turn, "state", None), "presence", None)
    if before is None:
        return {
            "state_block_found": _STATE_BLOCK.search(raw_reply) is not None,
            "presence_persisted": True,
        }
    state_match = _STATE_BLOCK.search(raw_reply)
    proposed_activity = False
    proposed_next = False
    malformed = False
    if state_match is not None:
        try:
            payload = json.loads(state_match.group(1))
        except (TypeError, ValueError):
            payload = None
            malformed = True
        if isinstance(payload, dict):
            proposed_activity = "activity_update" in payload
            proposed_next = "next_activity" in payload
        elif payload is not None:
            malformed = True
    activity_accepted = bool(state.activity_update) and (
        before.current_activity is None
        or before.current_activity.source == "autonomous_life"
    )
    rejection_reason = ""
    if proposed_activity and not state.activity_update:
        rejection_reason = "invalid activity metadata"
    elif (
        proposed_activity
        and before.current_activity is not None
        and before.current_activity.source != "autonomous_life"
    ):
        rejection_reason = "an active activity already exists"
    elif malformed:
        rejection_reason = "invalid state metadata"
    after = apply_activity_updates(
        before,
        activity_update=state.activity_update if activity_accepted else None,
        next_activity=state.next_activity,
        now=getattr(getattr(prepared, "chat_input", None), "timestamp", time.time()),
    )
    return {
        "current_activity": after.current_activity.fact() if after.current_activity else "",
        "next_activity": after.next_activity.fact() if after.next_activity else "",
        "presence_planning_available": (
            before.current_activity is None and before.next_activity is None
        ),
        "state_block_found": state_match is not None,
        "activity_update_proposed": proposed_activity,
        "next_activity_proposed": proposed_next,
        "activity_update_accepted": activity_accepted,
        "activity_rejection_reason": rejection_reason,
        "activity_activated_this_turn": bool(getattr(turn, "presence_activated", False)),
        "activity_expired_this_turn": bool(getattr(turn, "presence_expired", False)),
        "presence_persisted": True,
        "presence_included_in_prompt": bool(getattr(turn_context, "life_context", "")),
    }


def run_companion_turn(
    chat_input: ChatInput | str,
    *,
    session_id: str | None = None,
    skip_memory: bool = False,
    skip_if_busy: bool = False,
    direct_reply: str | None = None,
) -> CompanionTurnResult:
    """Run one shared companion turn, including decision parsing and persistence."""

    chat = (
        normalize_chat_input(
            text=chat_input,
            conversation_id=session_id or "popup:default",
        )
        if isinstance(chat_input, str)
        else chat_input
    )
    if _pause_remaining(chat):
        decision = CompanionDecision(message="", should_respond=False)
        _record_companion_debug(
            chat,
            decision_parsed=False,
            decision=decision,
            message_suppressed=True,
        )
        return CompanionTurnResult(decision, suppressed_by_pause=True)

    from app.core.model_loader import (
        InferenceCancelled,
        InferenceQueueTimeout,
        InferenceTiming,
        ModelManager,
    )

    prepared = prepare_turn(
        chat,
        skip_memory=skip_memory,
        skip_if_busy=skip_if_busy,
        token_counter=(
            None
            if direct_reply is not None
            else ModelManager.get_instance().tokenize_prompt
        ),
    )
    timing = InferenceTiming(requested_at=time.perf_counter())
    try:
        # Keep the existing debug-state diagnostics in sync without giving the
        # UI a second inference or a separate commit path.
        from app.core.reply_pipeline import _remember_metrics

        _remember_metrics(prepared, committed=False)
        if direct_reply is None:
            manager = ModelManager.get_instance()
            parts: list[str] = []
            try:
                for chunk in manager.stream(
                    prepared.prompt_plan.messages,
                    prompt_tokens=prepared.prompt_plan.token_ids,
                    template_stop_sequences=prepared.prompt_plan.stop_sequences,
                    max_tokens=prepared.max_tokens,
                    cancellation=prepared.handle.cancellation,
                    queue_deadline=prepared.handle.queue_deadline,
                    timing=timing,
                ):
                    parts.append(chunk)
            except InferenceCancelled as exc:
                raise GenerationCancelled(str(exc)) from exc
            except InferenceQueueTimeout as exc:
                raise GenerationQueueFullError(str(exc)) from exc
            raw_reply = "".join(parts).strip()
            if not raw_reply:
                raise RuntimeError("Model returned no completion.")
            parsed = parse_companion_decision(raw_reply)
        else:
            raw_reply = str(direct_reply).strip()
            parsed = _ParsedCompanionOutput(
                CompanionDecision(message=raw_reply),
                parsed=False,
            )

        state = parse_akane_state(parsed.decision.message)
        decision = replace(parsed.decision, message=state.message)

        prepared.handle.raise_if_cancelled()
        _apply_pause(chat, decision.pause_seconds)
        _record_companion_debug(
            chat,
            decision_parsed=parsed.parsed,
            decision=decision,
            message_suppressed=not decision.should_respond,
        )
        commit_kwargs = {
            "preference_updates": state.preference_updates,
            "interest_additions": state.interest_additions,
            "relationship_updates": state.relationship_updates,
            "activity_update": state.activity_update,
            "next_activity": state.next_activity,
            "emotion_update": state.emotion_update,
        }
        if chat.initiative_opportunity is not None:
            initiated = decision.should_initiate and bool(state.message)
            decision = replace(
                decision,
                message=state.message if initiated else "",
                should_respond=initiated,
                should_initiate=initiated,
            )
            commit_initiative_turn(
                prepared,
                message=decision.message,
                used=initiated,
                **commit_kwargs,
            )
        elif decision.should_respond:
            commit_turn(prepared, state.message, **commit_kwargs)
        else:
            commit_silent_turn(prepared, **commit_kwargs)
        _remember_metrics(
            prepared,
            committed=True,
            timing=timing,
            presence_debug=_presence_debug(prepared, raw_reply, state),
        )
        return CompanionTurnResult(decision, prepared.generation_id)
    finally:
        finish_turn(prepared)


_INITIATIVE_ACTIVE_WINDOW_SECONDS = 15 * 60


def _life_prompt(state, *, now: float) -> str:
    presence = state.presence
    activity = presence.current_activity
    current_local_time, daypart = local_time_context(now)
    grounded_memories = sorted(
        (
            memory
            for memory in state.memories
            if memory.is_available(now)
            and memory.source_type not in {
                "generated_assistant",
                "speculative_inference",
                "conversation_summary",
            }
        ),
        key=lambda item: (item.importance, item.updated_at or item.created_at),
        reverse=True,
    )[:4]
    unfinished = tuple(
        memory.content
        for memory in grounded_memories
        if memory.category == "unfinished_topic" or memory.kind == "open_thread"
    )
    grounded = tuple(
        memory.content
        for memory in grounded_memories
        if memory.content not in unfinished
    )
    lines = [
        "Decide Akane's offscreen life state. Return only one <AKANE_LIFE> JSON block.",
        "Allowed decisions: start_activity, schedule_activity, continue_activity, quiet_downtime, do_nothing.",
        f"Current local time: {current_local_time}.",
        f"Current daypart: {daypart}.",
        f"Current activity: {activity.activity if activity else 'none'}.",
        "Current activity category: "
        f"{activity.category if activity and activity.category else 'unavailable'}.",
        "Current activity title: "
        f"{activity.title if activity and activity.title else 'unavailable'}.",
        "Current activity detail: "
        f"{activity.detail if activity and activity.detail else 'none recorded'}.",
        f"Immediately previous activity: {presence.previous_activity.fact() if presence.previous_activity else 'none'}.",
        f"Planned activity: {presence.next_activity.fact() if presence.next_activity else 'none'}.",
        "Activity pattern keys: "
        f"previous={presence.activity_pattern.previous_key or 'none'}; "
        f"prior={presence.activity_pattern.prior_key or 'none'}; "
        f"repeat_count={presence.activity_pattern.repeat_count}.",
        "Interests: " + (", ".join(state.interests) or "none") + ".",
        "Preferences: " + ("; ".join(f"{item.topic}: {item.reason}" for item in state.preferences[-3:]) or "none") + ".",
        "Grounded memories: " + ("; ".join(grounded) or "none") + ".",
        "Unfinished thoughts: " + ("; ".join(unfinished) or "none") + ".",
        f"Emotion: {state.emotion.primary}.",
        f"Pending reason: {presence.life_reason or 'none'}.",
        "Choose what you genuinely want to do next. You may draw from your interests, "
        "recent thoughts, preferences, memories, the time of day, or something new. "
        "You may also choose quiet downtime or no activity. Avoid repeating or "
        "alternating between the same recent activities unless continuing them is "
        "personally meaningful.",
        "Do not invent specific external titles, creators, streams, products, news, "
        "or real-world events. Such details require explicit support in the grounded "
        "context above; assistant-generated dialogue is not evidence.",
        "The JSON object must contain decision, activity, category, subject, detail, "
        "duration_minutes, start_after_minutes, reason, and interest_addition. "
        "Activity may be unrelated to existing interests, and interest_addition may "
        "name a new broad interest. Use null where an optional text field does not apply.",
    ]
    return "\n".join(lines)


def run_life_turn(
    *,
    profile_id: str,
    now: float | None = None,
    direct_reply: str | None = None,
    status_callback: Callable[[str, str, object], None] | None = None,
) -> bool:
    """Run one separately scheduled, non-conversational life inference when due."""

    current = time.time() if now is None else max(0.0, float(now))
    with _LIFE_LOCK:
        if profile_id in _LIFE_ACTIVE:
            return False
        _LIFE_ACTIVE.add(profile_id)
    store = get_internal_state_store()
    try:
        state = store.claim_life_opportunity(profile_id, now=current)
        if state is None:
            return False
        if status_callback is not None:
            status_callback("claimed", profile_id, True)
            status_callback(
                "claim_age",
                profile_id,
                max(0.0, current - state.presence.life_claimed_at),
            )
            status_callback("inference_started", profile_id, True)
        if direct_reply is None:
            from app.core.character import get_hard_constraints_prompt, load_character_profile
            from app.core.model_loader import ModelManager

            profile = load_character_profile()
            life_context = _life_prompt(state, now=current)
            recent_topic = get_memory_store().recent_profile_topic(profile_id)
            if recent_topic:
                life_context += f"\nRecent meaningful conversation topic: {recent_topic}."
            messages = [
                {"role": "system", "content": profile.identity},
                {"role": "system", "content": profile.soul},
                {"role": "system", "content": get_hard_constraints_prompt()},
                {"role": "user", "content": life_context},
            ]
            manager = ModelManager.get_instance()
            tokenized = manager.tokenize_prompt(messages)
            raw = "".join(manager.stream(messages, prompt_tokens=tokenized.tokens, template_stop_sequences=tokenized.stop_sequences, max_tokens=min(MAX_TOKENS, 220))).strip()
        else:
            raw = str(direct_reply)
            life_context = _life_prompt(state, now=current)
        decision = parse_life_decision(raw, grounded_context=life_context)
        if decision is None:
            store.release_life_opportunity(
                profile_id,
                now=current,
                failure_reason="invalid_life_block",
            )
            if status_callback is not None:
                status_callback("rejected", profile_id, "invalid life block")
            return False
        if status_callback is not None:
            status_callback("block_parsed", profile_id, True)
            status_callback("proposal", profile_id, decision.activity or decision.decision)
        accepted, rejection = store.commit_life_decision(
            profile_id,
            decision,
            now=current,
        )
        if not accepted:
            if status_callback is not None:
                status_callback("rejected", profile_id, rejection)
            return False
        if status_callback is not None:
            committed = store.internal_state(profile_id).presence
            status_callback(
                "activity_persisted",
                profile_id,
                bool(committed.current_activity or committed.next_activity),
            )
            status_callback("completed", profile_id, True)
        return True
    except Exception:
        store.release_life_opportunity(profile_id, now=current)
        raise
    finally:
        with _LIFE_LOCK:
            _LIFE_ACTIVE.discard(profile_id)


def _initiative_candidates(
    chat: ChatInput,
    memory_context: MemoryContext,
    *,
    now: float,
) -> tuple[InitiativeOpportunity, ...]:
    """Derive bounded opportunities from persisted facts, never invented events."""

    state_store = get_internal_state_store()
    if hasattr(state_store, "refresh_presence"):
        state = state_store.refresh_presence(chat.profile_id, now=now)
        presence = state.presence
    else:  # Lightweight test doubles may expose only read access.
        state = state_store.internal_state(chat.profile_id)
        presence = advance_presence(state.presence, now=now)
    candidates: list[InitiativeOpportunity] = []
    source_time = memory_context.updated_at
    if memory_context.unresolved_problem and (
        memory_context.current_task or memory_context.current_topic
    ):
        context = compact_text(
            memory_context.current_task or memory_context.current_topic, 360
        )
        if context and source_time:
            candidates.append(
                InitiativeOpportunity(
                    "unfinished meaningful conversation",
                    context,
                    0.85,
                    source_time,
                    source_time + 24 * 60 * 60,
                )
            )
    activity = presence.previous_activity
    if (
        activity is not None
        and activity.ends_at <= now < activity.ends_at + 12 * 60 * 60
    ):
        candidates.append(
            InitiativeOpportunity(
                "recent completed offscreen activity",
                activity.fact(),
                0.62,
                activity.ends_at,
                activity.ends_at + 12 * 60 * 60,
            )
        )
    relationship_entries = (
        *state.relationship.unresolved_events,
        *state.relationship.shared_context,
        *state.relationship.patterns,
    )
    active_entries = tuple(
        entry
        for entry in relationship_entries
        if entry.status != "resolved" and entry.updated_at > 0
    )
    if active_entries:
        entry = max(
            active_entries,
            key=lambda item: (item.importance * item.confidence, item.updated_at),
        )
        candidates.append(
            InitiativeOpportunity(
                "meaningful relationship context",
                entry.summary,
                max(0.60, min(0.90, entry.importance * entry.confidence)),
                entry.updated_at,
                entry.updated_at + 7 * 24 * 60 * 60,
            )
        )
    if state.preferences:
        preference = max(
            state.preferences,
            key=lambda item: (item.strength, item.updated_at),
        )
        if preference.updated_at > 0:
            candidates.append(
                InitiativeOpportunity(
                    "developed personal preference",
                    f"{preference.topic}: {preference.reason}",
                    max(0.55, min(0.75, preference.strength)),
                    preference.updated_at,
                    preference.updated_at + 7 * 24 * 60 * 60,
                )
            )
    return tuple(candidate for candidate in candidates if candidate.expires_at > now)


def run_initiative_turn(
    *,
    profile_id: str,
    conversation_id: str,
    source: str,
    display_name: str = "",
    now: float | None = None,
) -> CompanionTurnResult:
    """Attempt one model-chosen initiative only when a grounded opportunity is due."""

    timestamp = time.time() if now is None else max(0.0, float(now))
    chat = normalize_chat_input(
        text="A grounded initiative opportunity is available.",
        profile_id=profile_id,
        conversation_id=conversation_id,
        source=source,
        timestamp=timestamp,
        display_name=display_name,
        autonomous=True,
    )
    if _pause_remaining(chat) or _SCHEDULER.is_active(chat.conversation_id, chat.profile_id):
        return CompanionTurnResult(CompanionDecision("", should_respond=False))
    memory_store = get_memory_store()
    memory_context = memory_store.build_context(
        chat.profile_id,
        chat.conversation_id,
        display_name=chat.display_name,
        include_memory=True,
    )
    opportunity = memory_store.claim_initiative_opportunity(
        profile_id=chat.profile_id,
        conversation_id=chat.conversation_id,
        candidates=_initiative_candidates(chat, memory_context, now=timestamp),
        now=timestamp,
        active_window_seconds=_INITIATIVE_ACTIVE_WINDOW_SECONDS,
    )
    if opportunity is None:
        return CompanionTurnResult(CompanionDecision("", should_respond=False))
    return run_companion_turn(replace(chat, initiative_opportunity=opportunity), skip_if_busy=True)


def cancel_generation(conversation_id: str, profile_id: str | None = None) -> bool:
    conversation = compact_text(conversation_id, 120) or "popup:default"
    profile = compact_text(profile_id, 120) if profile_id is not None else None
    return _SCHEDULER.cancel(conversation, profile)


def cancel_all_generations() -> None:
    _SCHEDULER.cancel_all()


def reset_conversation(conversation_id: str, profile_id: str) -> None:
    conversation = compact_text(conversation_id, 120) or "popup:default"
    profile = compact_text(profile_id, 120) or "local:owner"
    with _COMMIT_LOCK:
        _SCHEDULER.cancel(conversation, profile)
        with _PAUSE_LOCK:
            _PAUSED_UNTIL.pop((profile, conversation), None)
            _COMPANION_DEBUG.pop((profile, conversation), None)
        get_memory_store().clear_conversation(conversation, profile)


def forget_profile(profile_id: str) -> None:
    profile = compact_text(profile_id, 120) or "local:owner"
    with _COMMIT_LOCK:
        _SCHEDULER.cancel_profile(profile)
        with _PAUSE_LOCK:
            for key in tuple(_PAUSED_UNTIL):
                if key[0] == profile:
                    _PAUSED_UNTIL.pop(key, None)
            for key in tuple(_COMPANION_DEBUG):
                if key[0] == profile:
                    _COMPANION_DEBUG.pop(key, None)
        get_memory_store().clear_profile(profile)
        get_internal_state_store().clear(profile)


def session_state_snapshot(
    conversation_id: str | None = None,
    profile_id: str | None = None,
) -> dict[str, object]:
    conversation = compact_text(conversation_id, 120) or "popup:default"
    profile = compact_text(profile_id, 120) or "local:owner"
    state_store = get_internal_state_store()
    remaining_pause = _pause_remaining(
        ChatInput(profile, conversation, "state", "web", time.time())
    )
    pause_key = (profile, conversation)
    with _PAUSE_LOCK:
        companion_debug = dict(_COMPANION_DEBUG.get(pause_key, {}))
    if companion_debug:
        companion_debug["currently_paused"] = bool(remaining_pause)
    from app.core.life_worker import life_worker_debug

    payload = {
        "akane": state_store.public_internal_state(profile),
        "memory": get_memory_store().public_conversation(conversation, profile),
        "popup_user": state_store.public_profile(profile),
        "active_generation_id": _SCHEDULER.active_generation_id(conversation),
        "pause_remaining_seconds": int(remaining_pause),
        "companion_decision": companion_debug,
        "life_worker": life_worker_debug(),
    }
    return payload


def local_time_context(timestamp: float) -> tuple[str, str]:
    local = datetime.fromtimestamp(timestamp, ZoneInfo(TIMEZONE))
    hour = local.hour
    if 5 <= hour < 12:
        daypart = "morning"
    elif 12 <= hour < 17:
        daypart = "afternoon"
    elif 17 <= hour < 22:
        daypart = "evening"
    else:
        daypart = "night"
    rendered = local.strftime("%I:%M %p").lstrip("0")
    return rendered, daypart


def date_time_line(timestamp: float | None = None) -> str:
    local = datetime.fromtimestamp(
        time.time() if timestamp is None else timestamp,
        ZoneInfo(TIMEZONE),
    )
    hour = local.strftime("%I").lstrip("0") or "0"
    zone = local.tzname() or TIMEZONE
    return (
        f"Current local date and time: {local.strftime('%A, %B')} "
        f"{local.day}, {local.year} at {hour}:{local.strftime('%M %p')} {zone}."
    )


def timing_enabled() -> bool:
    return _TIMING_ENABLED


def _preference_continuity(content: str, change_allowed: bool) -> str:
    value = str(content or "").strip()
    if not value:
        return ""
    if change_allowed:
        return (
            f"{value}\nReconsider it only for a concrete reason stated in the answer; "
            "otherwise preserve the named choice."
        )
    return (
        f"{value}\nPreserve the named choice. Wording and emphasis may vary, but do not "
        "replace it or introduce another favorite."
    )


def _time_context_relevant(text: str) -> bool:
    lower = str(text or "").lower()
    return any(
        marker in lower
        for marker in (
            "today",
            "tonight",
            "tomorrow",
            "yesterday",
            "what time",
            "current time",
            "what day",
            "the date",
            "this morning",
            "this afternoon",
            "this evening",
        )
    )


_SCHEDULER = GenerationScheduler()
