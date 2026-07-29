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
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.config import (
    GENERATION_QUEUE_TIMEOUT_SECONDS,
    MAX_INPUT_CHARS,
    MAX_PENDING_GENERATIONS,
    MAX_TOKENS,
    PROMPT_DEBUG,
    TIMEZONE,
)
from app.core.memory import InitiativeOpportunity, StateSnapshot, get_state_store
from app.core.model_loader import (
    InferenceCancelled,
    InferenceQueueTimeout,
    InferenceTiming,
    ModelManager,
)
from app.core.presence import activity_continuity, format_presence_context
from app.core.prompt import (
    PromptContext,
    PromptPlan,
    build_conversation_prompt,
    build_initiative_prompt,
)
from app.core.utils import canonical_profile_id, compact_text, words
from app.integrations.vscode_context import code_context_for_message

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
            threading.Event(),
            time.monotonic() + GENERATION_QUEUE_TIMEOUT_SECONDS,
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
        "emotion", "memories", "preferences", "interests", "opinions",
        "relationship",
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
    autonomous: bool = False,
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
        autonomous=bool(autonomous),
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


def _relevant_items(
    values: tuple[object, ...],
    query: str,
    *,
    limit: int,
    broad_request: bool = False,
) -> tuple[str, ...]:
    query_terms = words(query)
    selected: list[str] = []
    for item in reversed(values):
        text = _item_text(item)
        if not text:
            continue
        if broad_request or query_terms & words(text):
            selected.append(text)
        if len(selected) >= limit:
            break
    return tuple(reversed(selected))


def _emotion_context(emotion: object, now: float) -> str:
    primary = compact_text(getattr(emotion, "primary", ""), 32).lower()
    intensity = float(getattr(emotion, "intensity", 0.0) or 0.0)
    cause = compact_text(getattr(emotion, "cause", ""), 140)
    updated_at = float(getattr(emotion, "updated_at", 0.0) or 0.0)
    if updated_at and now > updated_at:
        intensity *= 0.5 ** ((now - updated_at) / (24.0 * 3600.0))
    if not primary or intensity < 0.05:
        return ""
    result = f"{primary}, intensity {max(0.0, min(1.0, intensity)):.2f}"
    return f"{result}; cause: {cause}" if cause else result


def _relationship_context(relationship: object, query: str) -> tuple[str, ...]:
    unresolved = tuple(getattr(relationship, "unresolved_events", ()) or ())
    shared = tuple(getattr(relationship, "shared_context", ()) or ())
    patterns = tuple(getattr(relationship, "patterns", ()) or ())
    relationship_request = bool(
        words(query) & {"relationship", "between", "promise", "promised", "us"}
    )
    return (
        *_relevant_items(unresolved, query, limit=2, broad_request=True),
        *_relevant_items(shared, query, limit=2, broad_request=relationship_request),
        *_relevant_items(patterns, query, limit=1, broad_request=relationship_request),
    )


def _time_relevant(text: str) -> bool:
    return bool(
        words(text)
        & {
            "today", "tonight", "tomorrow", "yesterday", "time", "date",
            "morning", "afternoon", "evening",
        }
    )


def date_time_line(timestamp: float | None = None) -> str:
    local = datetime.fromtimestamp(
        time.time() if timestamp is None else timestamp,
        ZoneInfo(TIMEZONE),
    )
    hour = local.strftime("%I").lstrip("0") or "0"
    zone = local.tzname() or TIMEZONE
    return (
        f"{local.strftime('%A, %B')} {local.day}, {local.year}, "
        f"{hour}:{local.strftime('%M %p')} {zone}"
    )


def _prompt_context(
    snapshot: StateSnapshot,
    chat: ChatInput,
) -> PromptContext:
    profile = snapshot.profile
    editor = code_context_for_message(chat.text)
    editor_context = editor.prompt_text
    if editor.requested and not editor.connected:
        editor_context = "No current VS Code editor snapshot is available."
    continuity = activity_continuity(
        profile.presence.current_activity,
        snapshot.last_profile_assistant_at or None,
    )
    tastes_requested = bool(
        words(chat.text)
        & {"like", "likes", "dislike", "prefer", "preference", "interest", "favorite"}
    )
    opinions_requested = bool(words(chat.text) & {"opinion", "think", "view", "take"})
    return PromptContext(
        recent_turns=tuple(snapshot.recent_turns),
        memories=tuple(
            text
            for item in snapshot.relevant_memories
            if (text := _item_text(item))
        ),
        relationship=_relationship_context(profile.relationship, chat.text),
        preferences=_relevant_items(
            tuple(profile.preferences),
            chat.text,
            limit=3,
            broad_request=tastes_requested,
        ),
        interests=_relevant_items(
            tuple(profile.interests),
            chat.text,
            limit=3,
            broad_request=tastes_requested,
        ),
        opinions=_relevant_items(
            tuple(profile.opinions),
            chat.text,
            limit=3,
            broad_request=opinions_requested,
        ),
        emotion=_emotion_context(profile.emotion, snapshot.now),
        presence=format_presence_context(
            profile.presence,
            now=snapshot.now,
            continuity=continuity,
        ),
        reply_context=chat.reply_context,
        external_context=editor_context,
        date_time=date_time_line(snapshot.now) if _time_relevant(chat.text) else "",
        initiative_opportunity=(
            f"{chat.initiative_opportunity.reason}: "
            f"{chat.initiative_opportunity.context}"
            if chat.initiative_opportunity
            else ""
        ),
    )


def _build_prompt(
    snapshot: StateSnapshot,
    chat: ChatInput,
    *,
    token_counter,
) -> PromptPlan:
    context = _prompt_context(snapshot, chat)
    if chat.initiative_opportunity is not None:
        return build_initiative_prompt(context, token_counter=token_counter)
    return build_conversation_prompt(
        chat.text,
        context,
        token_counter=token_counter,
    )


def _decision_from_output(
    parsed: ParsedStateOutput,
    *,
    initiative: bool,
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
        should_initiate=initiative and should_respond,
    )


def run_companion_turn(
    chat: ChatInput,
    *,
    skip_memory: bool = False,
    skip_if_busy: bool = False,
    on_delta=None,
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
    )
    started_at = time.perf_counter()
    store = get_state_store()
    try:
        snapshot = store.snapshot(
            chat.profile_id,
            chat.conversation_id,
            query=chat.text,
            now=chat.timestamp,
            include_memory=not skip_memory,
        )
        timing = InferenceTiming(requested_at=started_at)
        manager = ModelManager.get_instance()
        priority = "background" if chat.autonomous else "visible"
        parts: list[str] = []
        visible_stream = _VisibleReplyStream()
        try:
            with manager.reserve(
                priority=priority,
                cancellation=handle.cancellation,
                queue_deadline=handle.queue_deadline,
            ) as reservation:
                plan = _build_prompt(
                    snapshot,
                    chat,
                    token_counter=lambda messages: manager.tokenize_prompt(
                        messages,
                        reservation=reservation,
                    ),
                )
                for chunk in manager.stream(
                    prompt_tokens=plan.token_ids,
                    template_stop_sequences=plan.stop_sequences,
                    max_tokens=MAX_TOKENS,
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
        decision = _decision_from_output(
            parsed,
            initiative=chat.initiative_opportunity is not None,
        )
        if decision.pause_seconds is not None:
            _apply_pause(chat, decision.pause_seconds)
        if chat.initiative_opportunity is not None:
            store.commit_initiative(
                snapshot,
                opportunity=chat.initiative_opportunity,
                message=decision.message,
                used=decision.should_initiate,
                proposals=parsed.proposals,
                now=time.time(),
            )
        else:
            store.commit_turn(
                snapshot,
                user_text=chat.text,
                assistant_text=decision.message if decision.should_respond else "",
                source=chat.source,
                request_id=chat.request_id,
                proposals=parsed.proposals,
                now=time.time(),
            )
            store.wake_presence_if_due(chat.profile_id, now=time.time())
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


def autonomous_dialogue_available(
    profile_id: str,
    conversation_id: str,
) -> bool:
    profile = canonical_profile_id(profile_id)
    conversation = compact_text(conversation_id, 160) or "popup:default"
    chat = ChatInput(profile, conversation, "initiative", "autonomous", time.time())
    return not _pause_remaining(chat) and not _SCHEDULER.is_active(
        conversation,
        profile,
    )


def reset_conversation(conversation_id: str, profile_id: str) -> None:
    conversation = compact_text(conversation_id, 160) or "popup:default"
    profile = canonical_profile_id(profile_id)
    _SCHEDULER.cancel(conversation, profile)
    with _PAUSE_LOCK:
        _PAUSED_UNTIL.pop((profile, conversation), None)
    with _DEBUG_LOCK:
        _TURN_DEBUG.pop((profile, conversation), None)
    get_state_store().clear_conversation(conversation, profile)


def forget_profile(profile_id: str) -> None:
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
    from app.core.life_worker import life_worker_debug

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
        "life_worker": life_worker_debug(),
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
            f"Current Activity: {current.get('activity') or 'None'}",
            f"Next Life Decision At: {presence.get('next_decision_at') or 'None'}",
            f"Life Worker: {(snapshot.get('life_worker') or {}).get('Life Worker Started', False)}",
            f"Prompt Tokens: {prompt.get('exact_tokens', 'None')}",
            f"Prompt System Characters: {prompt.get('system_characters', 'None')}",
            f"Model Context Window: {model['context_window']}",
        )
    )
_SCHEDULER = GenerationScheduler()
