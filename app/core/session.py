"""One-turn orchestration over the five production authorities."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field

from app.core.config import (
    GENERATION_QUEUE_TIMEOUT_SECONDS,
    MAX_INPUT_CHARS,
    MAX_PENDING_GENERATIONS,
    MAX_TOKENS,
    PROMPT_DEBUG,
)
from app.core.memory import (
    StateSnapshot,
    format_emotional_context,
    get_state_store,
)
from app.core.model_loader import (
    InferenceCancelled,
    InferenceQueueTimeout,
    InferenceTiming,
    ModelManager,
)
from app.core.prompt import (
    PromptContext,
    PromptPlan,
    build_conversation_prompt,
)
from app.core.time_context import (
    build_time_context,
    format_time_context,
)
from app.core.utils import canonical_profile_id, compact_text, words
from app.integrations.vscode_context import CodeContext, code_context_for_message

_STATE_BLOCK = re.compile(
    r"<AKANE_STATE>\s*(.*?)\s*</AKANE_STATE>",
    re.DOTALL | re.IGNORECASE,
)
_PAUSE_LOCK = threading.RLock()
_PAUSED_UNTIL: dict[tuple[str, str], float] = {}
_DEBUG_LOCK = threading.RLock()
_TURN_DEBUG: dict[tuple[str, str], dict[str, object]] = {}
_TIMING_ENABLED = str(os.environ.get("AKANE_TIMING", "")).strip().lower() in {
    "1", "true", "yes", "on",
}
_SECTION_DESCRIPTORS = (
    ("identity", "who you identity self yourself tell describe description person "
     "personality character values becoming changed change matters",
     "Respond from Akane's stable identity."),
    ("activity", "current activity doing occupied occupying occupation working "
     "presence recent experience continuing attention focus mind",
     "Address Akane's recorded current activity."),
    ("emotion", "feel feeling emotion mood emotional bother bothered quieter affect",
     "Address Akane's current emotional state."),
    ("relationship", "relationship between Akane Arcane trust conflict closeness "
     "connection", "Use established relationship evidence."),
    ("user_context", "Arcane user remember remembered facts taste prefer preference "
     "likes dislikes recommend recommendation care cares confirmed",
     "Use confirmed information about Arcane."),
    ("akane_context", "Akane interests likes taste preference preferences prefer "
     "opinion opinions think goals thoughts realizations uncertainties memories "
     "judgment values matters design designs",
     "Use Akane's relevant established preferences or opinions."),
    ("shared_context", "shared together our experience conversation event commitment "
     "disagreement unresolved",
     "Use relevant established shared experiences."),
    ("tool_context", "tool editor file code external technical explain compare "
     "algorithm model quantization",
     "Use relevant technical context."),
)
_DIRECT_ACTIVITY_QUESTION = re.compile(
    r"\b(?:what\s+(?:are\s+you\s+doing|have\s+you\s+been\s+(?:doing|up\s+to)|"
    r"were\s+you\s+thinking\s+about)|how\s+has\s+your\s+day\s+been)\b",
    re.IGNORECASE,
)
_SEMANTIC_STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "but", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "what", "when", "where", "which", "why", "with", "would", "your",
}
_MEMORY_SECTIONS = {"user_context", "akane_context", "shared_context"}
_REFERENCE_ONLY_TERMS = {
    "it", "me", "my", "our", "ours", "that", "them", "this", "us", "you", "your",
}


class GenerationBusyError(RuntimeError):
    pass


class GenerationQueueFullError(RuntimeError):
    pass


class GenerationCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CompanionDecision:
    message: str = ""
    should_respond: bool = True
    pause_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class CompanionTurnResult:
    decision: CompanionDecision
    generation_id: str = ""
    suppressed_by_pause: bool = False

    @property
    def message(self) -> str:
        return self.decision.message


@dataclass(frozen=True, slots=True)
class ParsedStateOutput:
    message: str
    proposals: dict[str, object] = field(default_factory=dict)
    should_respond: bool | None = None
    pause_seconds: int | None = None
    parsed: bool = False


@dataclass(frozen=True, slots=True)
class ChatInput:
    profile_id: str
    conversation_id: str
    text: str
    source: str
    timestamp: float
    display_name: str = ""
    reply_context: str = ""
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
    """Bound visible waiters and prevent overlapping work for one profile."""

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
        cancellation: threading.Event | None = None,
        queue_deadline: float | None = None,
    ) -> GenerationHandle:
        with self._lock:
            if skip_if_busy and self._active:
                raise GenerationBusyError("Akane is busy with another reply.")
            if conversation_id in self._active or any(
                item.profile_id == profile_id for item in self._active.values()
            ):
                raise GenerationBusyError(
                    "This conversation or profile already has a reply in progress."
                )
        if not self._capacity.acquire(blocking=False):
            raise GenerationQueueFullError("Akane's generation queue is full.")
        handle = GenerationHandle(
            uuid.uuid4().hex,
            conversation_id,
            profile_id,
            cancellation or threading.Event(),
            (
                min(
                    queue_deadline,
                    time.monotonic() + GENERATION_QUEUE_TIMEOUT_SECONDS,
                )
                if queue_deadline is not None
                else time.monotonic() + GENERATION_QUEUE_TIMEOUT_SECONDS
            ),
        )
        with self._lock:
            if conversation_id in self._active or any(
                item.profile_id == profile_id for item in self._active.values()
            ):
                self._capacity.release()
                raise GenerationBusyError(
                    "This conversation or profile already has a reply in progress."
                )
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
            if handle is None or (
                profile_id is not None and handle.profile_id != profile_id
            ):
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
                item.profile_id == profile_id for item in self._active.values()
            )


class _VisibleReplyStream:
    """Release dialogue while retaining every possible hidden-block prefix."""

    _MARKERS = ("<AKANE_", "</AKANE_")

    def __init__(self) -> None:
        self._pending = ""
        self._finished = False

    def feed(self, chunk: object) -> str:
        if self._finished:
            return ""
        combined = self._pending + str(chunk or "")
        folded = combined.casefold()
        marker_positions = [
            position
            for marker in self._MARKERS
            if (position := folded.find(marker.casefold())) >= 0
        ]
        if marker_positions:
            self._finished = True
            self._pending = ""
            return combined[: min(marker_positions)]
        held = 0
        for marker in self._MARKERS:
            for size in range(1, min(len(combined), len(marker) - 1) + 1):
                if folded.endswith(marker[:size].casefold()):
                    held = max(held, size)
        if held:
            self._pending = combined[-held:]
            return combined[:-held]
        self._pending = ""
        return combined

    def finish(self) -> str:
        if self._finished:
            return ""
        pending = self._pending
        self._pending = ""
        if any(
            marker.casefold().startswith(pending.casefold())
            for marker in self._MARKERS
        ):
            return ""
        return pending


def _visible_text(raw: str) -> str:
    stream = _VisibleReplyStream()
    return (stream.feed(raw) + stream.finish()).strip()


def parse_akane_state(output: object) -> ParsedStateOutput:
    """Strip metadata and preserve independently valid top-level proposals."""

    raw = str(output or "")
    matches = tuple(_STATE_BLOCK.finditer(raw))
    visible = _visible_text(raw)
    if len(matches) != 1:
        return ParsedStateOutput(visible)
    try:
        payload = json.loads(matches[0].group(1))
    except (TypeError, ValueError):
        return ParsedStateOutput(visible)
    if not isinstance(payload, dict):
        return ParsedStateOutput(visible)

    permitted = {
        "emotion_update", "mood_update", "memories", "preferences",
        "interests", "opinions", "relationship",
    }
    proposals = {
        key: payload[key]
        for key in permitted
        if key in payload and payload[key] is not None
    }
    should_respond: bool | None = None
    pause_seconds: int | None = None
    participation = payload.get("participation")
    if (
        isinstance(participation, dict)
        and set(participation) == {"should_respond", "pause_seconds"}
        and type(participation.get("should_respond")) is bool
    ):
        pause = participation.get("pause_seconds")
        if pause is None or type(pause) is int:
            should_respond = participation["should_respond"]
            if pause is not None:
                pause_seconds = max(10, min(120, pause))
    return ParsedStateOutput(
        visible,
        proposals,
        should_respond,
        pause_seconds,
        parsed=True,
    )


def normalize_chat_input(
    *,
    text: object,
    profile_id: object = "local:owner",
    conversation_id: object = "popup:default",
    source: object = "popup",
    timestamp: object = 0.0,
    display_name: object = "",
    reply_context: object = "",
    request_id: object = "",
) -> ChatInput:
    value = str(text or "").strip()
    if not value:
        raise ValueError("Message cannot be empty.")
    if len(value) > MAX_INPUT_CHARS:
        raise ValueError(f"Message exceeds {MAX_INPUT_CHARS} characters.")
    try:
        sent_at = float(timestamp)
    except (TypeError, ValueError):
        sent_at = 0.0
    if not math.isfinite(sent_at) or sent_at <= 0.0:
        sent_at = time.time()
    return ChatInput(
        profile_id=canonical_profile_id(profile_id),
        conversation_id=compact_text(conversation_id, 160) or "popup:default",
        text=value,
        source=compact_text(source, 24).lower() or "popup",
        timestamp=sent_at,
        display_name=compact_text(display_name, 60),
        reply_context=compact_text(reply_context, 600),
        request_id=compact_text(request_id, 180),
    )


def _item_text(item: object) -> str:
    if isinstance(item, str):
        return compact_text(item, 320)
    for field_name in ("content", "text", "summary"):
        value = compact_text(getattr(item, field_name, ""), 320)
        if value:
            return value
    return ""


def _semantic_terms(value: object) -> set[str]:
    return words(value) - _SEMANTIC_STOPWORDS


def _response_context_plan(
    message: str,
    recent_turns: tuple[object, ...],
    *,
    reply_context: str = "",
    tool_requested: bool = False,
) -> tuple[tuple[str, ...], str, bool]:
    """Select broad context concepts and explicit current-focus questions."""

    direct_text = f"{message} {reply_context}"
    current_terms = _semantic_terms(direct_text)
    direct_activity = bool(_DIRECT_ACTIVITY_QUESTION.search(direct_text))
    recent_terms = _semantic_terms(
        " ".join(_item_text(turn) for turn in recent_turns[-2:])
    )
    recent_weight = 0.65 if len(current_terms) <= 3 else 0.30

    def similarity(left: set[str], descriptor: str) -> float:
        right = _semantic_terms(descriptor)
        if not left or not right:
            return 0.0
        return len(left & right) / math.sqrt(len(left) * len(right))

    scored: list[tuple[float, int, str, str]] = []
    directly_relevant_memory_sections: set[str] = set()
    for index, (name, descriptor, focus) in enumerate(_SECTION_DESCRIPTORS):
        direct_score = similarity(current_terms, descriptor)
        if name == "activity" and direct_activity:
            direct_score = max(direct_score, 1.0)
        score = direct_score
        score += recent_weight * similarity(recent_terms, descriptor)
        if name == "tool_context" and tool_requested:
            score = max(score, 1.0)
        if score >= 0.14:
            scored.append((score, index, name, focus))
        if name in _MEMORY_SECTIONS and direct_score >= 0.14:
            directly_relevant_memory_sections.add(name)
    identity_descriptor = _SECTION_DESCRIPTORS[0][1]
    identity_overlap = current_terms & _semantic_terms(identity_descriptor)
    if identity_overlap == {"you"} and (
        any(item[2] != "identity" for item in scored)
        or ("?" not in str(message) and current_terms - _REFERENCE_ONLY_TERMS)
    ):
        scored = [item for item in scored if item[2] != "identity"]
    strongest = max(scored, default=None, key=lambda item: (item[0], -item[1]))
    chosen = sorted(
        sorted(scored, key=lambda item: (-item[0], item[1]))[:3],
        key=lambda item: item[1],
    )
    memory_relevant = any(
        item[2] in directly_relevant_memory_sections for item in chosen
    ) and bool(current_terms - _REFERENCE_ONLY_TERMS)
    return (
        tuple(item[2] for item in chosen),
        strongest[3] if strongest else "",
        memory_relevant,
    )


def _relevant_items(
    values: tuple[object, ...],
    query: str,
    *,
    limit: int,
    fallback: bool = False,
) -> tuple[str, ...]:
    query_terms = _semantic_terms(query)
    ranked: list[tuple[int, int, str]] = []
    available: list[str] = []
    for index, item in enumerate(values):
        text = _item_text(item)
        if not text:
            continue
        available.append(text)
        overlap = len(query_terms & _semantic_terms(text))
        if overlap:
            ranked.append((overlap, index, text))
    if ranked:
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return tuple(item[2] for item in ranked[:limit])
    return tuple(reversed(available[-limit:])) if fallback else ()


def _prompt_context(
    snapshot: StateSnapshot,
    chat: ChatInput,
    *,
    sections: tuple[str, ...],
    response_focus: str,
    editor,
) -> PromptContext:
    profile = snapshot.profile
    planning_text = chat.text
    query = " ".join(
        (
            planning_text,
            chat.reply_context,
            *(_item_text(turn) for turn in snapshot.recent_turns[-2:]),
        )
    )
    recalled = tuple(snapshot.relevant_memories[:3])

    def relevant(
        name: str, values: tuple[object, ...], limit: int, *, fallback: bool = False,
    ) -> tuple[str, ...]:
        if name not in sections:
            return ()
        return _relevant_items(values, query, limit=limit, fallback=fallback)

    presence_context = ""
    if "activity" in sections:
        current = profile.presence.current_activity
        if current is None:
            presence_context = (
                "Akane's current activity is not recorded. Say so naturally without "
                "inventing one."
            )
        else:
            presence_context = "\n".join(
                (
                    f"Broad activity: {current.summary}",
                    f"Current focus: {current.focus}",
                    "Treat these as first-person factual context. Combine and paraphrase "
                    "them when useful; do not expose the labels, quote either field, or "
                    "speak about Akane in third person. Do not invent settings, physical "
                    "details, prior events, plans, outcomes, titles, people, places, or "
                    "implementation details. Emotion may affect wording only; it must not "
                    "add surroundings.",
                )
            )

    emotion = (
        format_emotional_context(profile, now=snapshot.now)
        if "emotion" in sections else ""
    )
    if "emotion" in sections and not emotion:
        emotion = "Akane's current emotional state is neutral."

    relationship = relevant(
        "relationship",
        (
            *(
                f"Established relationship pattern: {item.summary}"
                for item in profile.relationship.patterns
            ),
            *(
                f"Unresolved relationship evidence: {item.summary}"
                for item in profile.relationship.unresolved_events
            ),
        ),
        2,
        fallback=True,
    )
    user_context = (
        tuple(
            f"Arcane previously stated: {memory.text}"
            for memory in recalled
            if memory.subject == "user" and memory.confidence >= 0.75
        )
        if "user_context" in sections else ()
    )
    akane_context = relevant(
        "akane_context",
        (
            *(f"Akane's established interest: {item}" for item in profile.interests),
            *(
                f"Akane remembers: {memory.content}"
                for memory in recalled
                if memory.subject == "akane"
            ),
            *(
                f"Akane's established preference: {item.content}"
                for item in profile.preferences
            ),
            *(
                f"Akane's established opinion: {item.content}"
                for item in profile.opinions
            ),
        ),
        3,
    )
    shared_context = relevant(
        "shared_context",
        (
            *(
                f"Shared memory: {memory.content}"
                for memory in recalled
                if memory.subject == "shared"
            ),
            *(
                f"Established shared experience: {item.summary}"
                for item in profile.relationship.shared_context
            ),
        ),
        3,
    )
    tool_lines = []
    if editor.requested and editor.connected and editor.prompt_text:
        tool_lines.append(editor.prompt_text)
    elif editor.requested:
        tool_lines.append("No current VS Code editor snapshot is available.")
    current_activity = profile.presence.current_activity
    time_context = format_time_context(
        build_time_context(
            last_user_message_at=snapshot.last_profile_user_at,
            last_akane_message_at=snapshot.last_profile_assistant_at,
            current_activity_started_at=(
                current_activity.started_at if current_activity else None
            ),
        )
    )

    recent_turns = tuple(snapshot.recent_turns)
    initiative = snapshot.last_profile_initiative
    if (
        initiative is not None
        and all(turn.turn_id != initiative.turn_id for turn in recent_turns)
        and (
            not recent_turns
            or initiative.timestamp >= recent_turns[-1].timestamp
        )
    ):
        recent_turns = (*recent_turns, initiative)

    return PromptContext(
        response_focus=response_focus,
        time_context=time_context,
        recent_turns=recent_turns,
        relationship=relationship,
        emotion=emotion,
        presence=presence_context,
        user_context=user_context,
        akane_context=akane_context,
        shared_context=shared_context,
        reply_context=chat.reply_context,
        tool_context="\n".join(tool_lines) if "tool_context" in sections else "",
    )


def _build_prompt(
    snapshot: StateSnapshot,
    chat: ChatInput,
    *,
    context_plan: tuple[tuple[str, ...], str, bool],
    editor,
    token_counter,
    reserved_output_tokens: int = MAX_TOKENS,
) -> PromptPlan:
    sections, response_focus, _memory_relevant = context_plan
    context = _prompt_context(
        snapshot,
        chat,
        sections=sections,
        response_focus=response_focus,
        editor=editor,
    )
    return build_conversation_prompt(
        chat.text,
        context,
        token_counter=token_counter,
        reserved_output_tokens=reserved_output_tokens,
    )


def _decision_from_output(
    parsed: ParsedStateOutput,
) -> CompanionDecision:
    visible = parsed.message.strip()
    # A streamed visible reply cannot be retracted by contradictory trailing metadata.
    should_respond = bool(visible)
    if not visible and parsed.should_respond is False:
        should_respond = False
    pause = parsed.pause_seconds if not should_respond else None
    return CompanionDecision(
        message=visible if should_respond else "",
        should_respond=should_respond,
        pause_seconds=pause,
    )


def run_companion_turn(
    chat: ChatInput,
    *,
    skip_memory: bool = False,
    skip_if_busy: bool = False,
    on_delta=None,
    priority: str = "visible",
    max_tokens: int = MAX_TOKENS,
    cancellation: threading.Event | None = None,
    queue_deadline: float | None = None,
    allow_tool_context: bool = True,
    allow_initiative: bool = True,
) -> CompanionTurnResult:
    if _pause_remaining(chat):
        return CompanionTurnResult(
            CompanionDecision("", should_respond=False),
            suppressed_by_pause=True,
        )
    handle = _SCHEDULER.begin(
        chat.conversation_id,
        chat.profile_id,
        skip_if_busy=skip_if_busy,
        cancellation=cancellation,
        queue_deadline=queue_deadline,
    )
    started_at = time.perf_counter()
    store = get_state_store()
    try:
        timing = InferenceTiming(requested_at=started_at)
        manager = ModelManager.get_instance()
        parts: list[str] = []
        visible_stream = _VisibleReplyStream()
        try:
            with manager.reserve(
                priority=priority,
                cancellation=handle.cancellation,
                queue_deadline=handle.queue_deadline,
            ) as reservation:
                planning_text = chat.text
                snapshot = store.snapshot(
                    chat.profile_id,
                    chat.conversation_id,
                    now=time.time(),
                    include_memory=False,
                )
                editor = (
                    code_context_for_message(planning_text)
                    if allow_tool_context
                    else CodeContext(False, False)
                )
                context_plan = _response_context_plan(
                    planning_text,
                    tuple(snapshot.recent_turns),
                    reply_context=chat.reply_context,
                    tool_requested=editor.requested,
                )
                sections, _focus, memory_relevant = context_plan
                if (
                    not skip_memory
                    and memory_relevant
                    and set(sections) & _MEMORY_SECTIONS
                ):
                    memory_query = " ".join(
                        (
                            planning_text,
                            chat.reply_context,
                            *(_item_text(turn) for turn in snapshot.recent_turns[-2:]),
                        )
                    )
                    snapshot = store.snapshot(
                        chat.profile_id,
                        chat.conversation_id,
                        query=memory_query,
                        now=time.time(),
                        include_memory=True,
                    )
                plan = _build_prompt(
                    snapshot,
                    chat,
                    context_plan=context_plan,
                    editor=editor,
                    token_counter=lambda messages: manager.tokenize_prompt(
                        messages,
                        reservation=reservation,
                    ),
                    reserved_output_tokens=max(1, int(max_tokens)),
                )
                for chunk in manager.stream(
                    prompt_tokens=plan.token_ids,
                    template_stop_sequences=plan.stop_sequences,
                    max_tokens=max(1, int(max_tokens)),
                    cancellation=handle.cancellation,
                    timing=timing,
                    reservation=reservation,
                ):
                    parts.append(chunk)
                    delta = visible_stream.feed(chunk)
                    if on_delta is not None and delta:
                        on_delta(delta)
        except InferenceCancelled as exc:
            raise GenerationCancelled(str(exc)) from exc
        except InferenceQueueTimeout as exc:
            raise GenerationQueueFullError(str(exc)) from exc
        trailing = visible_stream.finish()
        if on_delta is not None and trailing:
            on_delta(trailing)
        raw = "".join(parts).strip()
        if not raw:
            raise RuntimeError("Model returned no completion.")
        handle.raise_if_cancelled()
        parsed = parse_akane_state(raw)
        decision = _decision_from_output(parsed)
        handle.raise_if_cancelled()
        if decision.pause_seconds is not None:
            _apply_pause(chat, decision.pause_seconds)
        committed = store.commit_turn(
            snapshot,
            user_text=chat.text,
            assistant_text=decision.message if decision.should_respond else "",
            source=chat.source,
            request_id=chat.request_id,
            proposals=parsed.proposals,
            now=time.time(),
        )
        if allow_initiative:
            from app.core.life_worker import offer_initiative_from_change

            offer_initiative_from_change(
                store,
                snapshot.profile,
                committed.profile,
                now=committed.now,
                conversation=True,
            )
        _record_debug(chat, snapshot, plan, parsed, decision, timing, started_at)
        return CompanionTurnResult(decision, handle.generation_id)
    finally:
        _SCHEDULER.finish(handle)


def _apply_pause(chat: ChatInput, pause_seconds: int | None) -> None:
    if pause_seconds is None:
        return
    with _PAUSE_LOCK:
        _PAUSED_UNTIL[(chat.profile_id, chat.conversation_id)] = (
            time.time() + max(10, min(120, int(pause_seconds)))
        )


def _pause_remaining(chat: ChatInput, *, now: float | None = None) -> float:
    current = time.time() if now is None else float(now)
    key = (chat.profile_id, chat.conversation_id)
    with _PAUSE_LOCK:
        until = _PAUSED_UNTIL.get(key, 0.0)
        if until <= current:
            _PAUSED_UNTIL.pop(key, None)
            return 0.0
        return until - current


def _record_debug(
    chat: ChatInput,
    snapshot: StateSnapshot,
    plan: PromptPlan,
    parsed: ParsedStateOutput,
    decision: CompanionDecision,
    timing: InferenceTiming,
    started_at: float,
) -> None:
    debug = {
        "revision": snapshot.revision,
        "prompt": plan.debug_metadata(),
        "state_fields": tuple(parsed.proposals),
        "state_block_parsed": parsed.parsed,
        "should_respond": decision.should_respond,
        "pause_seconds": decision.pause_seconds,
        "generation_seconds": time.perf_counter() - started_at,
        "prompt_tokens": timing.prompt_tokens,
        "updated_at": time.time(),
    }
    with _DEBUG_LOCK:
        _TURN_DEBUG[(chat.profile_id, chat.conversation_id)] = debug
    if PROMPT_DEBUG:
        print(f"[Akane:turn] {debug}", flush=True)
    if _TIMING_ENABLED:
        print(
            "[Akane:timing] "
            f"total={debug['generation_seconds']:.3f}s "
            f"prompt_tokens={timing.prompt_tokens}",
            flush=True,
        )


def cancel_generation(
    conversation_id: str,
    profile_id: str | None = None,
) -> bool:
    return _SCHEDULER.cancel(
        compact_text(conversation_id, 160) or "popup:default",
        canonical_profile_id(profile_id) if profile_id is not None else None,
    )


def cancel_all_generations() -> None:
    _SCHEDULER.cancel_all()


def clear_conversation_caches(conversation_id: str, profile_id: str) -> None:
    conversation = compact_text(conversation_id, 160) or "popup:default"
    profile = canonical_profile_id(profile_id)
    _SCHEDULER.cancel(conversation, profile)
    with _PAUSE_LOCK:
        _PAUSED_UNTIL.pop((profile, conversation), None)
    with _DEBUG_LOCK:
        _TURN_DEBUG.pop((profile, conversation), None)


def clear_profile_caches(profile_id: str) -> None:
    profile = canonical_profile_id(profile_id)
    _SCHEDULER.cancel_profile(profile)
    with _PAUSE_LOCK:
        for key in tuple(_PAUSED_UNTIL):
            if key[0] == profile:
                _PAUSED_UNTIL.pop(key, None)
    with _DEBUG_LOCK:
        for key in tuple(_TURN_DEBUG):
            if key[0] == profile:
                _TURN_DEBUG.pop(key, None)


def reset_conversation(conversation_id: str, profile_id: str) -> None:
    conversation = compact_text(conversation_id, 160) or "popup:default"
    profile = canonical_profile_id(profile_id)
    clear_conversation_caches(conversation, profile)
    get_state_store().clear_conversation(conversation, profile)


def forget_profile(profile_id: str) -> None:
    profile = canonical_profile_id(profile_id)
    clear_profile_caches(profile)
    get_state_store().clear_profile(profile)


def session_state_snapshot(
    conversation_id: str | None = None,
    profile_id: str | None = None,
) -> dict[str, object]:
    conversation = compact_text(conversation_id, 160) or "popup:default"
    profile = canonical_profile_id(profile_id)
    store = get_state_store()
    with _DEBUG_LOCK:
        debug = dict(_TURN_DEBUG.get((profile, conversation), {}))
    from app.core.life_worker import presence_worker_debug

    return {
        "akane": store.public_internal_state(profile),
        "memory": store.public_conversation(conversation, profile),
        "popup_user": store.public_profile(profile),
        "active_generation_id": _SCHEDULER.active_generation_id(conversation),
        "pause_remaining_seconds": int(
            _pause_remaining(
                ChatInput(profile, conversation, "state", "web", time.time())
            )
        ),
        "turn": debug,
        "presence_worker": presence_worker_debug(),
    }


def debug_state_report(
    conversation_id: str | None,
    profile_id: str | None = None,
    *,
    verbose: bool = False,
) -> str:
    snapshot = session_state_snapshot(conversation_id, profile_id)
    profile = snapshot["akane"]
    presence = profile.get("presence") or {}
    current = presence.get("current_activity") or {}
    prompt = (snapshot.get("turn") or {}).get("prompt") or {}
    model = ModelManager.get_instance().runtime_report(include_model_hash=verbose)
    return "\n".join(
        (
            "Akane Debug",
            "",
            f"Canonical Profile: {canonical_profile_id(profile_id)}",
            f"State Revision: {profile.get('revision', 'None')}",
            f"Emotion: {(profile.get('emotion') or {}).get('primary', 'neutral')}",
            f"Current Focus: {current.get('summary') or 'None'}",
            f"Next Presence Decision At: {presence.get('next_decision_at') or 'None'}",
            f"Presence Worker: {(snapshot.get('presence_worker') or {}).get('Presence Worker Started', False)}",
            f"Prompt Tokens: {prompt.get('exact_tokens', 'None')}",
            f"Prompt System Characters: {prompt.get('system_characters', 'None')}",
            f"Model Context Window: {model['context_window']}",
        )
    )
_SCHEDULER = GenerationScheduler()
