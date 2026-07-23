"""Normalized chat input, generation ownership, and successful-turn commits."""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Callable

from app.core.config import (
    GENERATION_QUEUE_TIMEOUT_SECONDS,
    MAX_INPUT_CHARS,
    MAX_PENDING_GENERATIONS,
    MAX_TOKENS,
    PROMPT_DEBUG,
)
from app.core.life import format_life_state
from app.core.memory import (
    InternalTurnResult,
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
)
from app.core.prompt import (
    PromptContext,
    PromptPlan,
    PromptTokenCount,
    build_prompt_plan,
)
from app.core.signal import TurnSignal
from app.core.utils import compact_text
from app.integrations.vscode_context import CodeContext, code_context_for_message

_TIMING_ENABLED = str(os.environ.get("AKANE_TIMING", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_COMMIT_LOCK = threading.RLock()


class GenerationBusyError(RuntimeError):
    pass


class GenerationQueueFullError(RuntimeError):
    pass


class GenerationCancelled(RuntimeError):
    pass


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
    relevant_memories: str = ""
    life_context: str = ""
    external_context: str = ""
    reply_context: str = ""
    date_time: str = ""
    preference_anchor: str = ""
    initiative_worthwhile: bool = True
    response_intention: ResponseIntention = ResponseIntention("acknowledge")
    compiled_style: CompiledStyle = CompiledStyle(
        "dry",
        (("Goal", "acknowledge"), ("Length", "concise")),
    )

    @property
    def signal(self) -> TurnSignal:
        return self.state_delta.signal


def _question_is_permitted(
    signal: TurnSignal,
    primary: str,
    text: str,
    recent_turns: tuple[object, ...],
    *,
    correction_prohibits: bool,
    serious: bool,
    direct_request: bool,
) -> bool:
    if (
        correction_prohibits
        or serious
        or direct_request
        or primary in {"answer", "comfort", "disagree", "reassure", "set boundary", "remain brief"}
        or signal.intent not in {"casual", "reflection", "teasing"}
        or _CLOSING.fullmatch(text)
    ):
        return False
    assistant_outputs = [
        str(getattr(turn, "content", "") or "").strip()
        for turn in recent_turns
        if getattr(turn, "role", "") == "assistant"
    ]
    if assistant_outputs and assistant_outputs[-1].rstrip().endswith("?"):
        return False
    return sum(output.rstrip().endswith("?") for output in assistant_outputs[-4:]) < 2


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
    boundary_crossed = bool(
        signal.hostility
        or signal.embodied_action and signal.emotion_state.boundary_level >= 2
    )
    distress = bool(signal.sadness or signal.intent == "emotional_support")
    substantive_objection = bool(
        signal.continued_after_objection
        or signal.contextual_reaction.kind == "disagreement"
    )
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
        ("set boundary", boundary_crossed),
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
        primary in {"set boundary", "comfort", "disagree", "reassure"}
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
        and not (
            signal.emotion_state.primary in {"irritated", "angry"}
            and signal.emotion_state.intensity >= 0.35
        )
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
    no_question_correction = any(
        use_by_id.get(memory.id) == "correction"
        and re.search(r"\b(?:ask|question|follow-up)\b", memory.content, re.IGNORECASE)
        for memory in used_memories
    )
    question_allowed = _question_is_permitted(
        signal,
        primary,
        text,
        recent_turns,
        correction_prohibits=no_question_correction,
        serious=serious,
        direct_request=generic_request,
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
        or primary in {"set boundary", "comfort", "disagree", "reassure", "remain brief"}
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
        elif question_allowed and signal.semantic_event.event_type == "activity":
            optional_behavior = "one natural question"
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
        if not intention.question_permitted:
            avoid.append("questions")
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
    elif primary == "set boundary":
        avoid.append("escalation")
    if primary in {"answer", "comfort", "disagree", "reassure", "set boundary", "remain brief"}:
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
    elif reaction in {"irritated", "angry", "criticism"}:
        humor = "none"
    elif reaction == "embarrassed":
        avoid.extend(("exaggerated mannerisms", "forced hesitation"))

    serious = bool(
        primary in {"comfort", "disagree", "reassure", "set boundary"}
        or signal.intent == "serious"
        or signal.sadness
        or signal.hostility
        or reaction == "concerned"
    )
    if serious:
        humor = "none"

    tension = bool(
        signal.hostility
        or signal.emotion_state.primary in {"irritated", "angry"}
        and signal.emotion_state.intensity >= 0.35
    )
    if tension:
        humor = "none"
        avoid.append("callbacks")
    if signal.emotion_state.boundary_level:
        humor = "none"

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
            if intention.optional_behavior == "one natural question" and not serious
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
            include_memory=not skip_memory and not chat.autonomous,
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
        signal = state_delta.signal
        stable_identity = signal.identity_attribute in {
            "identity",
            "appearance",
            "relationships",
        }
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

        behavioral_summary = "\n".join(
            value
            for value in (compiled_style.prompt_text(), signal.emotion_prompt())
            if value
        )

        editor_context = code_context.prompt_text
        if code_context.requested and not code_context.connected:
            editor_context = (
                "The requested editor context is unavailable. Do not claim to have inspected a file."
            )
        initiative_worthwhile = not chat.autonomous or any(
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
            relationship_context="" if stable_identity else memory_context.relationship,
            preference_context=(
                ""
                if stable_identity
                else _preference_continuity(
                    preference_memory.content if preference_memory else "",
                    preference_change_allowed,
                )
            ),
            relevant_memories=(
                "" if stable_identity else memory_context_text
            ),
            life_context=(
                format_life_state(
                    state_delta.state.life,
                    now=chat.timestamp,
                    scope=chat.conversation_id,
                    query=("what are you doing" if chat.autonomous else chat.text),
                )
                if not stable_identity else ""
            ),
            external_context="" if stable_identity else editor_context,
            reply_context="" if stable_identity else chat.reply_context,
            date_time=date_time_line() if _time_context_relevant(chat.text) else "",
            preference_anchor=preference_anchor,
            initiative_worthwhile=initiative_worthwhile,
            response_intention=response_intention,
            compiled_style=compiled_style,
        )

    def commit_completed_turn(
        self,
        chat: ChatInput,
        turn: CoordinatedTurnContext,
        reply: str,
    ) -> None:
        previous_state = self._state_store.commit_turn(
            chat.profile_id,
            turn.state_delta,
            used_memory_ids=turn.memory_context.memory_ids,
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
        normalize_chat_input(text=chat_input, conversation_id=session_id or "popup:default")
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


def commit_turn(prepared: TurnPreparation, reply: str) -> None:
    prepared.handle.raise_if_cancelled()
    with _COMMIT_LOCK:
        prepared.handle.raise_if_cancelled()
        prepared.coordinator.commit_completed_turn(
            prepared.chat_input,
            prepared.turn_context,
            reply,
        )


def finish_turn(prepared: TurnPreparation) -> None:
    _SCHEDULER.finish(prepared.handle)


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
        get_memory_store().clear_conversation(conversation, profile)


def forget_profile(profile_id: str) -> None:
    profile = compact_text(profile_id, 120) or "local:owner"
    with _COMMIT_LOCK:
        _SCHEDULER.cancel_profile(profile)
        get_memory_store().clear_profile(profile)
        get_internal_state_store().clear(profile)


def session_state_snapshot(
    conversation_id: str | None = None,
    profile_id: str | None = None,
) -> dict[str, object]:
    conversation = compact_text(conversation_id, 120) or "popup:default"
    profile = compact_text(profile_id, 120) or "local:owner"
    state_store = get_internal_state_store()
    payload = {
        "akane": state_store.public_internal_state(profile),
        "memory": get_memory_store().public_conversation(conversation, profile),
        "popup_user": state_store.public_profile(profile),
        "active_generation_id": _SCHEDULER.active_generation_id(conversation),
    }
    return payload


def date_time_line() -> str:
    timestamp = time.time()
    now = time.localtime(timestamp)
    hour = time.strftime("%I", now).lstrip("0") or "0"
    zone = time.strftime("%Z", now) or "local time"
    return (
        f"Current local date and time: {time.strftime('%A, %B', now)} "
        f"{now.tm_mday}, {now.tm_year} at {hour}:{time.strftime('%M %p', now)} {zone}."
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
