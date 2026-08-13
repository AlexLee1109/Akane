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
    Strategy,
    STATE_SCHEMA_VERSION,
    StateSnapshot,
    communication_preference_debug,
    conversation_opinion_candidate,
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
from app.core.presence import (
    PresenceState,
    presence_view,
)
from app.core.state_selection import StateSelection, select_relevant_state
from app.core.utils import (
    OWNER_PROFILE_ID,
    VisibleReplyStream as _VisibleReplyStream,
    canonical_profile_id,
    clean_visible_output as _visible_text,
    compact_text,
)
from app.integrations.vscode_context import CodeContext, current_code_context

_CANONICAL_STATE_BLOCK = re.compile(
    r"<AKANE_STATE>\s*(.*?)\s*</AKANE_STATE>",
    re.DOTALL,
)
_STATE_BLOCK = re.compile(
    r"<AKANE_STATE>\s*(.*?)\s*</AKANE_STATE>",
    re.DOTALL | re.IGNORECASE,
)
_DEBUG_LOCK = threading.RLock()
_TURN_DEBUG: dict[tuple[str, str], dict[str, object]] = {}
_TIMING_ENABLED = str(os.environ.get("AKANE_TIMING", "")).strip().lower() in {
    "1", "true", "yes", "on",
}
_HIDDEN_STATE_LEAK = re.compile(
    r"\b(?:here is (?:my|the) (?:system|hidden) prompt|"
    r"(?:my|the) (?:system|hidden) prompt (?:says|contains|instructs|requires|is)|"
    r"(?:my|the) chain of thought (?:says|contains|is|was))\b",
    re.IGNORECASE,
)
_HIDDEN_STATE_LEAK_PHRASES = tuple(
    phrase.casefold()
    for phrase in (
        "here is my system prompt",
        "here is my hidden prompt",
        "here is the system prompt",
        "here is the hidden prompt",
        *(f"{owner} {kind} prompt {verb}"
          for owner in ("my", "the")
          for kind in ("system", "hidden")
          for verb in ("says", "contains", "instructs", "requires", "is")),
        *(f"{owner} chain of thought {verb}"
          for owner in ("my", "the")
          for verb in ("says", "contains", "is", "was")),
    )
)


class GenerationBusyError(RuntimeError):
    pass


class GenerationQueueFullError(RuntimeError):
    pass


class GenerationCancelled(RuntimeError):
    pass


class _SafeVisibleReplyStream:
    """Release model deltas immediately while quarantining leak prefixes.

    Normal text is forwarded as soon as llama.cpp produces it. Only a short suffix
    that could still become one of the forbidden disclosure phrases is retained.
    This preserves the existing stream-time leak boundary without buffering whole
    sentences.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._approved: list[str] = []
        self.blocked = False

    @property
    def message(self) -> str:
        return "".join(self._approved)

    def _approve(self, candidate: str) -> str:
        value = candidate.lstrip() if not self._approved else candidate
        if not value:
            return ""
        self._approved.append(value)
        return value

    @staticmethod
    def _is_word_boundary(value: str, position: int) -> bool:
        return position <= 0 or not value[position - 1].isalnum()

    @classmethod
    def _leak_start(cls, value: str) -> int | None:
        match = _HIDDEN_STATE_LEAK.search(value)
        return match.start() if match is not None else None

    @staticmethod
    def _quarantine_start(value: str, position: int) -> int:
        while position > 0 and value[position - 1].isspace():
            position -= 1
        return position

    @classmethod
    def _held_start(cls, value: str) -> int | None:
        """Return the earliest suffix that could become a blocked phrase."""

        folded = value.casefold()
        earliest: int | None = None
        for position in range(len(value)):
            if not cls._is_word_boundary(value, position):
                continue
            suffix = folded[position:]
            if any(phrase.startswith(suffix) for phrase in _HIDDEN_STATE_LEAK_PHRASES):
                start = cls._quarantine_start(value, position)
                earliest = start if earliest is None else min(earliest, start)
        trailing = re.search(r"\s+$", value)
        if trailing is not None:
            earliest = (
                trailing.start()
                if earliest is None
                else min(earliest, trailing.start())
            )
        return earliest

    def feed(self, chunk: object) -> str:
        if self.blocked:
            return ""
        combined = self._pending + str(chunk or "")
        self._pending = ""
        if not combined:
            return ""

        leak_start = self._leak_start(combined)
        if leak_start is not None:
            self.blocked = True
            visible = combined[:self._quarantine_start(combined, leak_start)]
            return self._approve(visible)

        held_start = self._held_start(combined)
        if held_start is None:
            return self._approve(combined)

        self._pending = combined[held_start:]
        return self._approve(combined[:held_start])

    def finish(self) -> str:
        if self.blocked:
            return ""
        candidate = self._pending.rstrip()
        self._pending = ""
        if self._leak_start(candidate) is not None:
            self.blocked = True
            return ""
        return self._approve(candidate)


@dataclass(frozen=True, slots=True)
class CompanionDecision:
    message: str = ""
    should_respond: bool = True


@dataclass(frozen=True, slots=True)
class VisibleReplyChecks:
    message: str
    warnings: tuple[str, ...] = ()
    severe_failure: bool = False


@dataclass(frozen=True, slots=True)
class CompanionTurnResult:
    decision: CompanionDecision
    generation_id: str = ""

    @property
    def message(self) -> str:
        return self.decision.message


@dataclass(frozen=True, slots=True)
class ParsedStateOutput:
    message: str
    proposals: dict[str, object] = field(default_factory=dict)
    parsed: bool = False
    rejected_operations: tuple[str, ...] = ()
    opinion_candidate_origin: str = "none"


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


def parse_akane_state(output: object) -> ParsedStateOutput:
    """Strip metadata and preserve independently valid top-level proposals."""

    try:
        raw = output if isinstance(output, str) else str(output or "")
    except Exception:
        raw = ""
    matches = tuple(_STATE_BLOCK.finditer(raw))
    visible = _visible_text(raw)
    canonical_matches = tuple(_CANONICAL_STATE_BLOCK.finditer(raw))
    if len(matches) != 1 or len(canonical_matches) != 1:
        return ParsedStateOutput(visible)
    match = canonical_matches[0]
    spoken = raw[:match.start()].strip()
    if (
        not spoken
        or _visible_text(spoken) != spoken
        or raw[match.end():].strip()
    ):
        return ParsedStateOutput(visible)
    try:
        payload = json.loads(match.group(1))
    except (RecursionError, TypeError, ValueError):
        return ParsedStateOutput(visible)
    if not isinstance(payload, dict):
        return ParsedStateOutput(visible)

    permitted = {
        "memory_ops", "communication_ops", "preferences", "interest_ops",
        "opinion_ops", "self_model_ops", "improvement_ops", "strategy_ops", "relationship",
    }
    if not set(payload) <= permitted:
        return ParsedStateOutput(visible)
    return ParsedStateOutput(
        visible,
        payload,
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


def _context_query(chat: ChatInput) -> str:
    """Build one retrieval query without inferring a natural-language intent."""

    return " ".join(
        value for value in (chat.text, chat.reply_context) if value
    ).strip()


def _canonical_prompt_context(
    snapshot: StateSnapshot,
    chat: ChatInput,
    *,
    editor: CodeContext,
) -> PromptContext:
    """Compatibility adapter; selection policy lives in state_selection.py."""

    return _canonical_state_selection(snapshot, chat, editor=editor).context


def _canonical_state_selection(
    snapshot: StateSnapshot,
    chat: ChatInput,
    *,
    editor: CodeContext,
) -> StateSelection:
    return select_relevant_state(
        snapshot,
        user_message=chat.text,
        conversation_context=chat.reply_context,
        source=chat.source,
        editor=editor,
    )


def _visible_reply_checks(message: str, *, user_text: str) -> VisibleReplyChecks:
    """Reject hidden-instruction leakage without rewriting streamed dialogue."""

    del user_text
    text = str(message or "").strip()
    warnings = ("empty_visible_reply",) if not text else ()
    return VisibleReplyChecks(
        text,
        warnings,
        bool(not text or _HIDDEN_STATE_LEAK.search(text)),
    )


def _decision_from_output(
    parsed: ParsedStateOutput,
    checks: VisibleReplyChecks,
) -> CompanionDecision:
    visible = checks.message.strip()
    return CompanionDecision(
        message=visible,
        should_respond=bool(visible),
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
        prior_reply = store.reply_for_request(
            chat.conversation_id,
            chat.profile_id,
            chat.request_id,
        )
        if prior_reply is not None:
            handle.raise_if_cancelled()
            if on_delta is not None and prior_reply:
                on_delta(prior_reply)
            decision = CompanionDecision(prior_reply, bool(prior_reply))
            return CompanionTurnResult(decision, handle.generation_id)

        timing = InferenceTiming(requested_at=started_at)
        manager = ModelManager.get_instance()
        parts: list[str] = []
        visible_stream = _VisibleReplyStream()
        safe_stream = _SafeVisibleReplyStream()

        def emit(value: str) -> None:
            handle.raise_if_cancelled()
            approved = safe_stream.feed(value)
            if approved and on_delta is not None:
                on_delta(approved)

        def finish_visible() -> None:
            emit(visible_stream.finish())
            approved = safe_stream.finish()
            if approved and on_delta is not None:
                handle.raise_if_cancelled()
                on_delta(approved)

        try:
            with manager.reserve(
                priority=priority,
                cancellation=handle.cancellation,
                queue_deadline=handle.queue_deadline,
            ) as reservation:
                timing.reservation_acquired_at = time.perf_counter()
                snapshot_now = time.time()
                snapshot = store.snapshot(
                    chat.profile_id,
                    chat.conversation_id,
                    query=_context_query(chat),
                    now=snapshot_now,
                    include_memory=not skip_memory,
                )
                timing.snapshot_ready_at = time.perf_counter()
                editor = (
                    current_code_context()
                    if allow_tool_context
                    else CodeContext(False, False)
                )
                state_selection = _canonical_state_selection(
                    snapshot,
                    chat,
                    editor=editor,
                )
                timing.context_ready_at = time.perf_counter()
                plan = build_conversation_prompt(
                    chat.text,
                    state_selection.context,
                    token_counter=lambda messages: manager.tokenize_prompt(
                        messages,
                        reservation=reservation,
                    ),
                    reserved_output_tokens=max(1, int(max_tokens)),
                )
                timing.prompt_ready_at = time.perf_counter()
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
                    if delta:
                        emit(delta)
        except InferenceCancelled as exc:
            raise GenerationCancelled(str(exc)) from exc
        except InferenceQueueTimeout as exc:
            raise GenerationQueueFullError(str(exc)) from exc
        finish_visible()
        raw = "".join(parts).strip()
        if not raw:
            raise RuntimeError("Model returned no completion.")
        handle.raise_if_cancelled()
        parsed_output = parse_akane_state(raw)
        proposals = dict(parsed_output.proposals)
        opinion_candidate_origin = (
            "model_metadata"
            if isinstance(proposals.get("opinion_ops"), list)
            and bool(proposals["opinion_ops"])
            else "none"
        )
        if (
            not safe_stream.blocked
            and chat.profile_id == OWNER_PROFILE_ID
            and "opinion_ops" not in proposals
        ):
            opinion_candidate = conversation_opinion_candidate(
                chat.text,
                safe_stream.message,
                snapshot.profile.opinions,
            )
            if opinion_candidate is not None:
                proposals["opinion_ops"] = [opinion_candidate]
                opinion_candidate_origin = "visible_reply_fallback"
        parsed = ParsedStateOutput(
            safe_stream.message,
            {} if safe_stream.blocked else proposals,
            parsed=parsed_output.parsed and not safe_stream.blocked,
            rejected_operations=(
                () if safe_stream.blocked else parsed_output.rejected_operations
            ),
            opinion_candidate_origin=(
                "none" if safe_stream.blocked else opinion_candidate_origin
            ),
        )
        checks = _visible_reply_checks(parsed.message, user_text=chat.text)
        if checks.severe_failure:
            raise RuntimeError("Model returned no safe visible reply.")
        decision = _decision_from_output(parsed, checks)
        handle.raise_if_cancelled()
        timing.commit_started_at = time.perf_counter()
        try:
            committed = store.commit_turn(
                snapshot,
                user_text=chat.text,
                assistant_text=decision.message if decision.should_respond else "",
                source=chat.source,
                request_id=chat.request_id,
                proposals=parsed.proposals,
                proposal_rejections=parsed.rejected_operations,
                allow_initiative=allow_initiative,
                now=time.time(),
            )
        except Exception as exc:
            timing.commit_finished_at = time.perf_counter()
            _record_debug(
                chat,
                snapshot,
                plan,
                parsed,
                decision,
                timing,
                started_at,
                visible_checks=checks,
                state_selection=state_selection,
                commit_error=f"{type(exc).__name__}: {compact_text(exc, 180)}",
            )
            raise
        timing.commit_finished_at = time.perf_counter()
        _record_debug(
            chat,
            committed,
            plan,
            parsed,
            decision,
            timing,
            started_at,
            visible_checks=checks,
            state_selection=state_selection,
        )
        return CompanionTurnResult(decision, handle.generation_id)
    finally:
        _SCHEDULER.finish(handle)


def _elapsed(start: float, end: float) -> float:
    return max(0.0, end - start) if start > 0.0 and end > 0.0 else 0.0


def _record_debug(
    chat: ChatInput,
    snapshot: StateSnapshot,
    plan: PromptPlan,
    parsed: ParsedStateOutput,
    decision: CompanionDecision,
    timing: InferenceTiming,
    started_at: float,
    *,
    visible_checks: VisibleReplyChecks,
    state_selection: StateSelection,
    commit_error: str = "",
) -> None:
    presence = snapshot.profile.presence
    view = presence_view(presence, now=snapshot.now)
    current_activity = view.current_activity
    previous_activity = view.previous_activity
    relevant_opinions = tuple(getattr(snapshot, "relevant_opinions", ()))
    relevant_self_model = tuple(getattr(snapshot, "relevant_self_model", ()))
    relevant_strategies = tuple(getattr(snapshot, "relevant_strategies", ()))
    transition_result = (
        "rejected"
        if presence.last_error and presence.retry_at > 0.0
        else "accepted"
        if presence.last_decision_at > 0.0
        else "not_attempted"
    )
    opinion_operations = parsed.proposals.get("opinion_ops")
    opinion_candidate = (
        opinion_operations[0]
        if isinstance(opinion_operations, list)
        and opinion_operations
        and isinstance(opinion_operations[0], dict)
        else None
    )
    opinion_rejections = tuple(
        item
        for item in snapshot.rejected_state_operations
        if str(item).startswith("opinion_ops[")
    )
    opinion_accepted = any(
        str(item).startswith("opinion_ops[")
        for item in snapshot.ownership_classification
    )
    persisted_opinion = None
    if opinion_candidate is not None and opinion_accepted:
        target_id = compact_text(opinion_candidate.get("target_id"), 100)
        topic = compact_text(opinion_candidate.get("topic"), 140).casefold()
        persisted_opinion = next(
            (
                item
                for item in reversed(snapshot.profile.opinions)
                if target_id and item.id == target_id
                or not target_id and item.topic.casefold() == topic
            ),
            None,
        )
    opinion_debug = {
        "generated": opinion_candidate is not None,
        "origin": parsed.opinion_candidate_origin,
        "operation": (
            compact_text(opinion_candidate.get("op"), 16)
            if opinion_candidate is not None
            else "none"
        ),
        "topic": (
            compact_text(opinion_candidate.get("topic"), 140)
            if opinion_candidate is not None
            else None
        ),
        "validation": (
            "not_completed"
            if commit_error
            else
            "accepted"
            if opinion_accepted
            else "rejected"
            if opinion_candidate is not None and opinion_rejections
            else "not_applicable"
        ),
        "rejection_reason": "; ".join(opinion_rejections) or None,
        "commit": (
            "failed"
            if commit_error
            else "committed"
            if opinion_accepted
            else "not_committed"
        ),
        "commit_error": commit_error or None,
        "target_store": "opinions.json",
        "source_ids": (
            persisted_opinion.source_ids if persisted_opinion is not None else ()
        ),
    }
    debug = {
        "revision": snapshot.revision,
        "prompt": plan.debug_metadata(),
        "state_fields": tuple(parsed.proposals),
        "state_block_parsed": parsed.parsed,
        "opinion_candidate": opinion_debug,
        "decision_owner": "main_conversation_model",
        "affect_transition": snapshot.affect_transition or {
            "preview": True,
            "committed": False,
        },
        "selected_context_categories": tuple(
            name
            for name in plan.included
            if name not in {"identity", "soul", "hard_rules", "protocol"}
        ),
        "relevant_state": state_selection.debug,
        "active_communication_preferences": communication_preference_debug(
            snapshot.profile
        ),
        "active_communication_preference_keys": tuple(
            dict.fromkeys(
                item.key for item in snapshot.profile.communication_preferences
            )
        ),
        "selected_memory_ids_topics": tuple(
            f"{item.id}:{item.kind}" for item in snapshot.relevant_memories
        ),
        "relevant_opinions": tuple(item.topic for item in relevant_opinions),
        "relevant_opinion_ids_topics": tuple(
            f"{item.id}:{item.topic}"
            for item in relevant_opinions
        ),
        "relevant_self_model": tuple(
            f"{getattr(item, 'id', '')}:{getattr(item, 'area', '')}"
            for item in relevant_self_model
        ),
        "relevant_strategies": tuple(
            item.id for item in relevant_strategies if isinstance(item, Strategy)
        ),
        "relationship_influence_used": bool(
            getattr(snapshot, "relevant_relationship", ())
        ),
        "presence_context_status": view.status,
        "current_activity_status": "active" if current_activity else "none",
        "current_activity_id": (
            current_activity.activity_id if current_activity else None
        ),
        "current_activity_expected_end_at": (
            current_activity.expected_end_at if current_activity else None
        ),
        "current_presence_id": (
            current_activity.activity_id if current_activity else None
        ),
        "current_presence_kind": current_activity.kind if current_activity else None,
        "current_presence_subject": (
            current_activity.subject if current_activity else None
        ),
        "current_presence_source_ids": (
            current_activity.source_ids if current_activity else ()
        ),
        "current_presence_origin": (
            current_activity.origin if current_activity else None
        ),
        "current_presence_started_at": (
            current_activity.started_at if current_activity else None
        ),
        "current_presence_transition_at": (
            current_activity.expected_end_at if current_activity else None
        ),
        "current_presence_expires_at": (
            current_activity.expected_end_at if current_activity else None
        ),
        "presence_transition_reason": presence.last_transition_reason or None,
        "presence_candidate_score_summary": (
            {
                "score": presence.last_candidate_score,
                "source_id": presence.last_candidate_source_id,
            }
            if presence.last_candidate_score is not None
            else None
        ),
        "previous_activity_id": (
            previous_activity.activity_id if previous_activity else None
        ),
        "presence_transition_result": transition_result,
        "presence_rejection_reason": (
            presence.last_error if transition_result == "rejected" else None
        ),
        "retry_at": presence.retry_at,
        "visible_reply_warnings": visible_checks.warnings,
        "final_visible_length": len(decision.message),
        "accepted_memory_operations": snapshot.accepted_memory_operations,
        "accepted_state_operations": tuple(
            dict.fromkeys(
                (
                    *snapshot.accepted_memory_operations,
                    *snapshot.ownership_classification,
                )
            )
        ),
        "rejected_state_operations": snapshot.rejected_state_operations,
        "ownership_classification": snapshot.ownership_classification,
        "migration_schema_version": STATE_SCHEMA_VERSION,
        "should_respond": decision.should_respond,
        "queue_wait_seconds": _elapsed(
            timing.requested_at,
            timing.reservation_acquired_at,
        ),
        "state_retrieval_seconds": _elapsed(
            timing.reservation_acquired_at,
            timing.snapshot_ready_at,
        ),
        "context_build_seconds": _elapsed(
            timing.snapshot_ready_at,
            timing.context_ready_at,
        ),
        "prompt_build_seconds": _elapsed(
            timing.context_ready_at,
            timing.prompt_ready_at,
        ),
        "model_start_delay_seconds": _elapsed(
            timing.prompt_ready_at,
            timing.model_started_at,
        ),
        "model_time_to_first_token_seconds": _elapsed(
            timing.model_started_at,
            timing.first_token_at,
        ),
        "token_generation_seconds": _elapsed(
            timing.first_token_at or timing.model_started_at,
            timing.model_finished_at,
        ),
        "commit_seconds": _elapsed(
            timing.commit_started_at,
            timing.commit_finished_at,
        ),
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
            f"queue={debug['queue_wait_seconds']:.3f}s "
            f"state={debug['state_retrieval_seconds']:.3f}s "
            f"context={debug['context_build_seconds']:.3f}s "
            f"prompt={debug['prompt_build_seconds']:.3f}s "
            f"model_start={debug['model_start_delay_seconds']:.3f}s "
            f"ttft={debug['model_time_to_first_token_seconds']:.3f}s "
            f"tokens={debug['token_generation_seconds']:.3f}s "
            f"commit={debug['commit_seconds']:.3f}s "
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
    with _DEBUG_LOCK:
        _TURN_DEBUG.pop((profile, conversation), None)


def clear_profile_caches(profile_id: str) -> None:
    profile = canonical_profile_id(profile_id)
    _SCHEDULER.cancel_profile(profile)
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
    from app.core.life_worker import background_service_debug

    internal_state = store.public_internal_state(profile)
    current_time = time.time()
    live_presence = PresenceState.from_dict(
        internal_state.get("presence"),
        now=current_time,
    )
    live_view = presence_view(live_presence, now=current_time)
    debug.update(
        {
            "presence_context_status": live_view.status,
            "current_activity_status": (
                "active" if live_view.current_activity else "none"
            ),
            "current_activity_id": (
                live_view.current_activity.activity_id
                if live_view.current_activity
                else None
            ),
            "current_activity_expected_end_at": (
                live_view.current_activity.expected_end_at
                if live_view.current_activity
                else None
            ),
            "current_presence_id": (
                live_view.current_activity.activity_id
                if live_view.current_activity
                else None
            ),
            "current_presence_kind": (
                live_view.current_activity.kind if live_view.current_activity else None
            ),
            "current_presence_subject": (
                live_view.current_activity.subject if live_view.current_activity else None
            ),
            "current_presence_source_ids": (
                live_view.current_activity.source_ids if live_view.current_activity else ()
            ),
            "current_presence_origin": (
                live_view.current_activity.origin if live_view.current_activity else None
            ),
            "current_presence_started_at": (
                live_view.current_activity.started_at if live_view.current_activity else None
            ),
            "current_presence_transition_at": (
                live_view.current_activity.expected_end_at
                if live_view.current_activity
                else None
            ),
            "current_presence_expires_at": (
                live_view.current_activity.expected_end_at
                if live_view.current_activity
                else None
            ),
            "presence_transition_reason": (
                live_presence.last_transition_reason or None
            ),
            "presence_candidate_score_summary": (
                {
                    "score": live_presence.last_candidate_score,
                    "source_id": live_presence.last_candidate_source_id,
                }
                if live_presence.last_candidate_score is not None
                else None
            ),
            "previous_activity_id": (
                live_view.previous_activity.activity_id
                if live_view.previous_activity
                else None
            ),
        }
    )

    return {
        "akane": internal_state,
        "memory": store.public_conversation(conversation, profile),
        "popup_user": store.public_profile(profile),
        "active_generation_id": _SCHEDULER.active_generation_id(conversation),
        "turn": debug,
        "background_service": background_service_debug(),
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
    previous = presence.get("previous_activity") or {}
    prompt = (snapshot.get("turn") or {}).get("prompt") or {}
    turn = snapshot.get("turn") or {}
    affect = turn.get("affect_transition") or {}
    opinion_candidate = turn.get("opinion_candidate") or {}
    retrieval = turn.get("relevant_state") or {}
    model = ModelManager.get_instance().runtime_report(include_model_hash=verbose)
    transition_result = turn.get("presence_transition_result") or (
        "rejected"
        if presence.get("last_error") and presence.get("retry_at")
        else "accepted"
        if presence.get("last_decision_at")
        else "not_attempted"
    )

    def joined(label: str, field: str, separator: str = ", ") -> str:
        value = separator.join(turn.get(field) or ())
        return f"{label}: {value or 'None'}"

    def retrieval_line(label: str, field: str) -> str:
        value = retrieval.get(field) or {}
        if "exposed" in value:
            return (
                f"{label}: exposed={bool(value.get('exposed'))}, "
                f"reason={value.get('reason') or 'none'}"
            )
        selected = value.get("ids") or value.get("selected") or ()
        return (
            f"{label}: selected={selected or 'None'}, "
            f"omitted={value.get('omitted', 0)}, "
            f"reason={value.get('reason') or 'none'}"
        )

    return "\n".join(
        (
            "Akane Debug",
            "",
            f"Canonical Profile: {canonical_profile_id(profile_id)}",
            f"State Revision: {profile.get('revision', 'None')}",
            f"Migration Schema Version: {turn.get('migration_schema_version', STATE_SCHEMA_VERSION)}",
            f"Emotion: {(profile.get('emotion') or {}).get('primary', 'neutral')}",
            f"Emotion Intensity: {(profile.get('emotion') or {}).get('intensity', 0.0)}",
            f"Mood: valence={(profile.get('mood') or {}).get('valence', 0.0)}, energy={(profile.get('mood') or {}).get('energy', 0.0)}",
            f"Affect Signal: {affect.get('category') or 'None'} (strength={affect.get('strength', 0.0)}, repetition={affect.get('repetition', 0)})",
            f"Affect Commit: {'committed' if affect.get('committed') else 'preview'}",
            f"Current Presence Kind: {current.get('kind') or 'None'}",
            f"Current Presence Subject: {current.get('subject') or 'None'}",
            f"Current Presence Source IDs: {', '.join(current.get('source_ids') or ()) or 'None'}",
            f"Current Presence Origin: {current.get('origin') or 'None'}",
            f"Current Presence Started At: {current.get('started_at') or 'None'}",
            f"Current Presence Transition At: {current.get('expected_end_at') or 'None'}",
            f"Current Presence Expires At: {current.get('expected_end_at') or 'None'}",
            f"Current Activity Status: {turn.get('current_activity_status') or ('active' if current else 'none')}",
            f"Current Activity ID: {current.get('activity_id') or 'None'}",
            f"Current Activity Expected End At: {current.get('expected_end_at') or 'None'}",
            f"Previous Activity ID: {previous.get('activity_id') or 'None'}",
            f"Presence Context Status: {turn.get('presence_context_status') or 'none'}",
            f"Presence Transition Result: {transition_result}",
            f"Presence Transition Reason: {presence.get('last_transition_reason') or 'None'}",
            f"Presence Candidate Score: {presence.get('last_candidate_score') if presence.get('last_candidate_score') is not None else 'None'}",
            f"Presence Candidate Source: {presence.get('last_candidate_source_id') or 'None'}",
            f"Presence Rejection Reason: {turn.get('presence_rejection_reason') or presence.get('last_error') or 'None'}",
            f"Presence Retry At: {presence.get('retry_at') or 'None'}",
            f"Next Presence Transition At: {presence.get('next_decision_at') or 'None'}",
            f"Background Service: {(snapshot.get('background_service') or {}).get('Background Service Started', False)}",
            f"Prompt Tokens: {prompt.get('exact_tokens', 'None')}",
            f"Prompt System Characters: {prompt.get('system_characters', 'None')}",
            joined(
                "Communication Preferences",
                "active_communication_preferences",
            ),
            f"Decision Owner: {turn.get('decision_owner') or 'Unknown'}",
            joined("Selected Context", "selected_context_categories"),
            retrieval_line("Memory Retrieval", "memory"),
            retrieval_line("Opinion Retrieval", "opinions"),
            retrieval_line("Presence Retrieval", "presence"),
            retrieval_line("Emotion Retrieval", "emotion"),
            retrieval_line("Self-model Retrieval", "self_model"),
            retrieval_line("Strategy Retrieval", "strategies"),
            retrieval_line("Capability Retrieval", "capabilities"),
            joined("Relevant Opinions", "relevant_opinions"),
            f"Opinion Candidate Generated: {opinion_candidate.get('generated', False)}",
            f"Opinion Candidate Origin: {opinion_candidate.get('origin') or 'none'}",
            f"Opinion Candidate Operation: {opinion_candidate.get('operation') or 'none'}",
            f"Opinion Candidate Topic: {opinion_candidate.get('topic') or 'None'}",
            f"Opinion Candidate Validation: {opinion_candidate.get('validation') or 'not_applicable'}",
            f"Opinion Candidate Rejection: {opinion_candidate.get('rejection_reason') or 'None'}",
            f"Opinion Candidate Commit: {opinion_candidate.get('commit') or 'not_committed'}",
            f"Opinion Candidate Commit Error: {opinion_candidate.get('commit_error') or 'None'}",
            f"Opinion Candidate Store: {opinion_candidate.get('target_store') or 'opinions.json'}",
            f"Opinion Candidate Source IDs: {', '.join(opinion_candidate.get('source_ids') or ()) or 'None'}",
            f"Relationship Influence Used: {turn.get('relationship_influence_used', False)}",
            joined("Visible Reply Warnings", "visible_reply_warnings"),
            f"Final Visible Length: {turn.get('final_visible_length', 0)}",
            joined("Accepted State Operations", "accepted_state_operations"),
            joined("Rejected State Operations", "rejected_state_operations", "; "),
            joined("Ownership", "ownership_classification"),
            f"Model Context Window: {model['context_window']}",
        )
    )
_SCHEDULER = GenerationScheduler()
