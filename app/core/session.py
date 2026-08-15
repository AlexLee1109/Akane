"""Foreground conversation orchestration over the v2 ownership boundaries."""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass

from app.core.config import SETTINGS
from app.core.context import ContextBuilder
from app.core.inference import (
    InferenceCancelled,
    InferenceQueueTimeout,
    InferenceRuntime,
    InferenceTiming,
)
from app.core.state import MoodChange, StateChangeProposal, Turn
from app.core.prompt import PromptPlan, build_dialogue_prompt, build_reasoning_prompt
from app.core.store import get_store
from app.core.utils import (
    OWNER_PROFILE_ID,
    canonical_profile_id,
    compact_text,
    lexical_terms,
    log_performance,
    log_timing,
)

_DELIBERATION_TERMS = {
    "compare", "tradeoff", "plan", "debug", "contradiction", "conflict",
    "decide", "decision", "analyze", "technical", "architecture",
}
_TRIVIAL_EXCHANGES = {
    "hi", "hello", "hey", "hi there", "hello there", "hey there",
    "ok", "okay", "thanks", "thank you", "bye", "goodbye",
}
_GREETING_EXCHANGES = {"hi", "hello", "hey", "hi there", "hello there", "hey there"}


class GenerationBusyError(RuntimeError):
    pass


class GenerationQueueFullError(RuntimeError):
    pass


class GenerationCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CompanionTurnResult:
    message: str
    generation_id: str = ""


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
    profile_id: str
    conversation_id: str
    cancellation: threading.Event
    queue_deadline: float

    def check(self) -> None:
        if self.cancellation.is_set():
            raise GenerationCancelled("Generation was cancelled.")


class GenerationScheduler:
    """Bound foreground work and prevent one profile from racing itself."""

    def __init__(self) -> None:
        self._capacity = threading.BoundedSemaphore(SETTINGS.max_pending_generations + 1)
        self._lock = threading.RLock()
        self._active: dict[tuple[str, str], GenerationHandle] = {}

    def begin(
        self,
        profile_id: str,
        conversation_id: str,
        *,
        skip_if_busy: bool = False,
        cancellation: threading.Event | None = None,
        queue_deadline: float | None = None,
    ) -> GenerationHandle:
        key = (profile_id, conversation_id)
        with self._lock:
            profile_busy = any(existing_profile == profile_id for existing_profile, _ in self._active)
            if key in self._active or profile_busy or (skip_if_busy and self._active):
                raise GenerationBusyError("This profile already has a reply in progress.")
        if not self._capacity.acquire(blocking=False):
            raise GenerationQueueFullError("Akane's generation queue is full.")
        deadline = time.monotonic() + SETTINGS.generation_queue_timeout_seconds
        if queue_deadline is not None:
            deadline = min(deadline, float(queue_deadline))
        handle = GenerationHandle(
            uuid.uuid4().hex,
            profile_id,
            conversation_id,
            cancellation or threading.Event(),
            deadline,
        )
        with self._lock:
            if key in self._active or any(existing_profile == profile_id for existing_profile, _ in self._active):
                self._capacity.release()
                raise GenerationBusyError("This profile already has a reply in progress.")
            self._active[key] = handle
        return handle

    def finish(self, handle: GenerationHandle) -> None:
        key = (handle.profile_id, handle.conversation_id)
        with self._lock:
            if self._active.get(key) is not handle:
                return
            self._active.pop(key, None)
        self._capacity.release()

    def cancel(self, conversation_id: str, profile_id: str) -> bool:
        with self._lock:
            handle = self._active.get((profile_id, conversation_id))
            if handle is None:
                return False
            handle.cancellation.set()
            return True

    def cancel_profile(self, profile_id: str) -> None:
        with self._lock:
            for (active_profile, _), handle in self._active.items():
                if active_profile == profile_id:
                    handle.cancellation.set()

    def cancel_all(self) -> None:
        with self._lock:
            for handle in self._active.values():
                handle.cancellation.set()


_SCHEDULER = GenerationScheduler()
_DEBUG_LOCK = threading.RLock()
_TURN_DEBUG: dict[tuple[str, str], dict[str, object]] = {}


def normalize_chat_input(
    *,
    text: object,
    profile_id: object = OWNER_PROFILE_ID,
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
    if len(value) > SETTINGS.max_input_chars:
        raise ValueError(f"Message exceeds {SETTINGS.max_input_chars} characters.")
    try:
        created_at = float(timestamp)
    except (TypeError, ValueError):
        created_at = 0.0
    if not math.isfinite(created_at) or created_at <= 0:
        created_at = time.time()
    return ChatInput(
        canonical_profile_id(profile_id),
        compact_text(conversation_id, 160) or "popup:default",
        value,
        compact_text(source, 32).casefold() or "popup",
        created_at,
        compact_text(display_name, 60),
        compact_text(reply_context, 600),
        compact_text(request_id, 180),
    )


def _simple_exchange_key(text: str) -> str:
    return " ".join(text.casefold().strip(" \t\r\n!?.,").split())


def _should_reflect(chat: ChatInput, reply: str, *, skip_memory: bool) -> bool:
    if skip_memory:
        return False
    del reply
    return _simple_exchange_key(chat.text) not in _TRIVIAL_EXCHANGES


def _deterministic_mood(chat: ChatInput) -> MoodChange | None:
    if _simple_exchange_key(chat.text) not in _GREETING_EXCHANGES:
        return None
    return MoodChange(0.05, 0.03, "friendly", "greeting")


def _needs_deliberation(message: str, context) -> bool:
    """Use cheap structural signals to protect casual-turn latency."""

    terms = lexical_terms(message)
    if len(message) >= 420 or (message.count("?") >= 2 and len(message) >= 160):
        return True
    reasoning_signals = terms & _DELIBERATION_TERMS
    if reasoning_signals and (len(message) >= 120 or len(terms) >= 16):
        return True
    if len(reasoning_signals) >= 2 and len(message) >= 60:
        return True
    if context.code is not None and context.code.prompt_text:
        return True
    return False


def _fit_prompt(
    runtime: InferenceRuntime,
    reservation,
    context,
    chat: ChatInput,
    max_tokens: int,
    initial_plan: PromptPlan | None = None,
) -> tuple[PromptPlan, int, str, float, float]:
    recent_limit = len(context.state.recent_turns)
    plan = initial_plan
    prompt_build_seconds = 0.0
    tokenization_seconds = 0.0
    while True:
        if plan is None:
            build_started_at = time.perf_counter()
            plan = build_dialogue_prompt(
                context,
                user_message=chat.text,
                reply_context=chat.reply_context,
                recent_limit=recent_limit or None,
            )
            prompt_build_seconds += time.perf_counter() - build_started_at
        tokenization_started_at = time.perf_counter()
        count, method = runtime.count_prompt_tokens(plan.messages, reservation)
        tokenization_seconds += time.perf_counter() - tokenization_started_at
        if count + max_tokens <= SETTINGS.llama_context_window:
            return plan, count, method, prompt_build_seconds, tokenization_seconds
        if recent_limit <= 2:
            raise RuntimeError("The selected prompt does not fit the configured context window.")
        recent_limit = max(2, recent_limit - 2)
        plan = None


def _prompt_token_breakdown(
    runtime, reservation, plan: PromptPlan, prompt_tokens: int,
) -> dict[str, int]:
    if not (SETTINGS.prompt_debug or SETTINGS.timing_enabled):
        return {}
    counter = getattr(runtime, "count_text_tokens", None)
    result: dict[str, int] = {}
    for name, text in plan.token_sections.items():
        if not text:
            result[name] = 0
        elif counter is None:
            result[name] = max(1, len(text) // 4)
        else:
            result[name] = int(counter(text, reservation)[0])
    result["chat_template_overhead"] = max(0, prompt_tokens - sum(result.values()))
    result["other"] = sum(
        result.get(name, 0)
        for name in (
            "code_context", "time", "reply_context",
            "current_message", "chat_template_overhead",
        )
    )
    return result


def _tokens_per_second(tokens: int, seconds: float) -> float:
    return float(tokens) / seconds if tokens > 0 and seconds > 0 else 0.0


def run_companion_turn(
    chat_input: ChatInput,
    *,
    skip_memory: bool = False,
    skip_if_busy: bool = False,
    streaming: bool = False,
    on_delta=None,
    on_stream_end=None,
    priority: str = "owner",
    max_tokens: int | None = None,
    cancellation: threading.Event | None = None,
    queue_deadline: float | None = None,
    allow_tool_context: bool = True,
) -> CompanionTurnResult:
    chat = chat_input
    session_started_at = time.perf_counter()
    if not streaming and (on_delta is not None or on_stream_end is not None):
        raise ValueError("Streaming callbacks require streaming=True.")
    handle = _SCHEDULER.begin(
        chat.profile_id,
        chat.conversation_id,
        skip_if_busy=skip_if_busy,
        cancellation=cancellation,
        queue_deadline=queue_deadline,
    )
    token_limit = SETTINGS.max_tokens if max_tokens is None else max(1, int(max_tokens))
    timing = InferenceTiming(session_started_at)
    reasoning_timing = InferenceTiming(session_started_at)
    store = get_store()
    runtime = InferenceRuntime.get_instance()
    used_reasoning = False
    model_calls = {
        "dialogue": 0, "deliberation": 0, "reflection": 0,
        "inner_life": 0, "autonomy": 0, "other": 0,
    }
    log_performance("foreground_event", stage="request_start", at=time.time())
    try:
        runtime.foreground_started()
        prior_reply = store.reply_for_request(chat.conversation_id, chat.profile_id, chat.request_id)
        if prior_reply is not None:
            return CompanionTurnResult(prior_reply, handle.generation_id)
        context_started_at = time.perf_counter()
        context_builder = ContextBuilder(store)
        context = context_builder.build(
            profile_id=chat.profile_id,
            conversation_id=chat.conversation_id,
            message=chat.text,
            reply_context=chat.reply_context,
            allow_tool_context=allow_tool_context,
            now=chat.timestamp,
        )
        context_finished_at = time.perf_counter()
        handle.check()
        needs_deliberation = _needs_deliberation(chat.text, context)
        plan: PromptPlan | None = None
        prompt_build_seconds = 0.0
        if not needs_deliberation:
            prompt_started_at = time.perf_counter()
            plan = build_dialogue_prompt(
                context,
                user_message=chat.text,
                reply_context=chat.reply_context,
                recent_limit=len(context.state.recent_turns) or None,
            )
            prompt_build_seconds = time.perf_counter() - prompt_started_at
        wait_started_at = time.perf_counter()
        deliberation_started_at = wait_started_at
        deliberation_finished_at = wait_started_at
        last_delta_delivered_at = 0.0
        with runtime.reserve(
            priority=priority,
            cancellation=handle.cancellation,
            queue_deadline=handle.queue_deadline,
        ) as reservation:
            reservation_acquired_at = time.perf_counter()
            log_performance("foreground_event", stage="model_acquired", at=time.time())
            if needs_deliberation:
                used_reasoning = True
                model_calls["deliberation"] += 1
                deliberation_started_at = time.perf_counter()
                deliberation = runtime.complete_messages(
                    build_reasoning_prompt(context, chat.text),
                    max_tokens=SETTINGS.reasoning_tokens,
                    reservation=reservation,
                    cancellation=handle.cancellation,
                    timing=reasoning_timing,
                    temperature=0.2,
                    call_kind="deliberation",
                )
                deliberation_finished_at = time.perf_counter()
                context = context_builder.with_deliberation(context, deliberation)
            else:
                deliberation_started_at = reservation_acquired_at
                deliberation_finished_at = reservation_acquired_at
            plan, prompt_tokens, prompt_method, extra_prompt_build, tokenization_seconds = _fit_prompt(
                runtime, reservation, context, chat, token_limit, initial_plan=plan,
            )
            prompt_build_seconds += extra_prompt_build
            timing.prompt_tokens = prompt_tokens
            timing.prompt_token_method = prompt_method
            token_breakdown = _prompt_token_breakdown(
                runtime, reservation, plan, prompt_tokens,
            )
            static_prefix_hash = getattr(plan, "static_prefix_hash", "")
            cache_key = (
                (chat.profile_id, chat.conversation_id, static_prefix_hash)
                if static_prefix_hash else None
            )
            debug_prompt = getattr(runtime, "debug_dialogue_prompt_once", None)
            if debug_prompt is not None:
                debug_prompt(
                    plan.messages,
                    llm=reservation.llm,
                    token_sections=plan.token_sections,
                    cache_key=cache_key,
                )
            chunks: list[str] = []
            model_calls["dialogue"] += 1
            log_performance("foreground_event", stage="dialogue_start", at=time.time())
            for chunk in runtime.stream_messages(
                plan.messages,
                max_tokens=token_limit,
                reservation=reservation,
                cancellation=handle.cancellation,
                timing=timing,
                call_kind="dialogue",
                cache_key=cache_key,
            ):
                handle.check()
                chunks.append(chunk)
                if streaming and on_delta is not None:
                    on_delta(chunk)
                last_delta_delivered_at = time.perf_counter()
            log_performance("foreground_event", stage="dialogue_end", at=time.time())
        reservation_released_at = time.perf_counter()
        reply = "".join(chunks).strip()
        if not reply:
            raise RuntimeError("The model returned an empty response.")
        handle.check()
        if streaming and on_stream_end is not None:
            on_stream_end()
        committed_at = time.time()
        user_turn = Turn(
            f"turn_{uuid.uuid4().hex}", chat.profile_id, chat.conversation_id,
            "user", chat.text, chat.timestamp, chat.request_id,
        )
        assistant_turn = Turn(
            f"turn_{uuid.uuid4().hex}", chat.profile_id, chat.conversation_id,
            "assistant", reply, max(committed_at, chat.timestamp), chat.request_id,
        )
        enqueue_started_at = time.perf_counter()
        queue_reflection = _should_reflect(chat, reply, skip_memory=skip_memory)
        proposal = StateChangeProposal(
            chat.profile_id,
            turns=(user_turn, assistant_turn),
            mood=_deterministic_mood(chat),
            reflection_turn_ids=(user_turn.id, assistant_turn.id) if not skip_memory else None,
            reflection_ready=queue_reflection,
            origin="conversation",
        )
        enqueue_finished_at = time.perf_counter()
        persist_started_at = enqueue_finished_at
        commit = store.commit(proposal)
        persist_finished_at = time.perf_counter()
        final_token_at = timing.final_token_at or last_delta_delivered_at
        model_finished_at = timing.model_finished_at or reservation_released_at
        log_timing(
            "session",
            store_snapshot=context_builder.last_timing.get("store_snapshot_seconds", 0.0),
            context_build=context_finished_at - context_started_at,
            prompt_build=prompt_build_seconds,
            tokenization=tokenization_seconds,
            model_wait=reservation_acquired_at - wait_started_at,
            prefill=timing.prefill_seconds,
            deliberation=deliberation_finished_at - deliberation_started_at,
            decode=timing.decode_seconds,
            finalize_stream=max(0.0, reservation_released_at - model_finished_at),
            enqueue_reflection=enqueue_finished_at - enqueue_started_at,
            persist_turn=persist_finished_at - persist_started_at,
            final_token_to_return=max(0.0, persist_finished_at - final_token_at),
            total=persist_finished_at - session_started_at,
        )
        metrics = {
            "generation_id": handle.generation_id,
            "prompt_tokens": timing.prompt_tokens,
            "prompt_token_method": timing.prompt_token_method,
            "prompt_token_breakdown": token_breakdown,
            "selected_counts": plan.selected_counts,
            "used_reasoning": used_reasoning,
            "reflection_dirty": not skip_memory,
            "reflection_content_ready": queue_reflection,
            "reflection_inference_eligible": False,
            "reflection_queued": queue_reflection,
            "revision": commit.revision,
            "store_snapshot_ms": context_builder.last_timing.get("store_snapshot_seconds", 0.0) * 1000,
            "context_build_ms": (context_finished_at - context_started_at) * 1000,
            "prompt_build_ms": prompt_build_seconds * 1000,
            "tokenization_ms": tokenization_seconds * 1000,
            "model_wait_ms": (reservation_acquired_at - wait_started_at) * 1000,
            "prefill_ms": timing.prefill_seconds * 1000,
            "prefill_tokens": timing.prefill_tokens,
            "new_prompt_eval_tokens": timing.new_prompt_eval_tokens,
            "reused_prefix_tokens": timing.reused_prefix_tokens,
            "prompt_eval_ms": timing.prefill_seconds * 1000,
            "prefill_tokens_per_second": _tokens_per_second(
                timing.prefill_tokens, timing.prefill_seconds,
            ),
            "time_to_first_token_ms": (
                max(0.0, timing.first_token_at - session_started_at) * 1000
                if timing.first_token_at else None
            ),
            "decode_ms": timing.decode_seconds * 1000,
            "generated_tokens": timing.generated_tokens,
            "generated_token_method": timing.generated_token_method,
            "decode_tokens_per_second": _tokens_per_second(
                timing.generated_tokens, timing.decode_seconds,
            ),
            "store_commit_ms": (persist_finished_at - persist_started_at) * 1000,
            "integration_delivery_ms": None,
            "foreground_total_ms": (persist_finished_at - session_started_at) * 1000,
            "model_calls": model_calls,
            "model_seconds": max(0.0, timing.model_finished_at - timing.model_started_at),
            "first_token_seconds": (
                max(0.0, timing.first_token_at - timing.model_started_at)
                if timing.first_token_at else None
            ),
            "post_generation_seconds": (
                max(0.0, persist_finished_at - final_token_at)
                if final_token_at else None
            ),
            "persist_seconds": persist_finished_at - persist_started_at,
            "model_wait_seconds": reservation_acquired_at - wait_started_at,
        }
        log_performance(
            "foreground",
            context_build_ms=metrics["context_build_ms"],
            prompt_build_ms=metrics["prompt_build_ms"],
            prompt_tokens=metrics["prompt_tokens"],
            reused_prefix_tokens=metrics["reused_prefix_tokens"],
            new_prompt_eval_tokens=metrics["new_prompt_eval_tokens"],
            model_wait_ms=metrics["model_wait_ms"],
            prompt_eval_ms=metrics["prompt_eval_ms"],
            prefill_tokens_per_second=metrics["prefill_tokens_per_second"],
            time_to_first_token_ms=metrics["time_to_first_token_ms"],
            decode_ms=metrics["decode_ms"],
            generated_tokens=metrics["generated_tokens"],
            decode_tokens_per_second=metrics["decode_tokens_per_second"],
            store_commit_ms=metrics["store_commit_ms"],
            foreground_total_ms=metrics["foreground_total_ms"],
            model_calls=metrics["model_calls"],
        )
        with _DEBUG_LOCK:
            _TURN_DEBUG[(chat.profile_id, chat.conversation_id)] = metrics
        log_performance("foreground_event", stage="foreground_complete", at=time.time())
        return CompanionTurnResult(reply, handle.generation_id)
    except (InferenceCancelled, InferenceQueueTimeout) as exc:
        raise GenerationCancelled(str(exc)) from exc
    finally:
        runtime.foreground_finished()
        _SCHEDULER.finish(handle)


def cancel_generation(conversation_id: str, profile_id: str = OWNER_PROFILE_ID) -> bool:
    return _SCHEDULER.cancel(conversation_id, profile_id)


def cancel_all_generations() -> None:
    _SCHEDULER.cancel_all()


def clear_conversation_caches(conversation_id: str, profile_id: str) -> None:
    InferenceRuntime.get_instance().discard_dialogue_cache(profile_id, conversation_id)
    with _DEBUG_LOCK:
        _TURN_DEBUG.pop((profile_id, conversation_id), None)


def clear_profile_caches(profile_id: str) -> None:
    InferenceRuntime.get_instance().discard_dialogue_cache(profile_id)
    with _DEBUG_LOCK:
        for key in tuple(_TURN_DEBUG):
            if key[0] == profile_id:
                _TURN_DEBUG.pop(key, None)


def reset_conversation(conversation_id: str, profile_id: str) -> None:
    cancel_generation(conversation_id, profile_id)
    get_store().clear_conversation(conversation_id, profile_id)
    clear_conversation_caches(conversation_id, profile_id)


def forget_profile(profile_id: str) -> None:
    _SCHEDULER.cancel_profile(profile_id)
    get_store().clear_profile(profile_id)
    clear_profile_caches(profile_id)


def session_state_snapshot(conversation_id: str, profile_id: str) -> dict[str, object]:
    snapshot = get_store().debug_snapshot(profile_id, conversation_id)
    with _DEBUG_LOCK:
        snapshot["last_turn"] = _TURN_DEBUG.get((profile_id, conversation_id), {})
    return snapshot


def debug_state_report(conversation_id: str, profile_id: str, *, verbose: bool = False) -> str:
    snapshot = session_state_snapshot(conversation_id, profile_id)
    last = snapshot.get("last_turn", {})
    self_counts = snapshot["self_counts"]
    lines = [
        f"Akane v2 state: schema {snapshot['schema_version']}, revision {snapshot['revision']}",
        f"Recent turns: {snapshot['recent_turn_count']}",
        "Self: " + ", ".join(f"{kind}={count}" for kind, count in self_counts.items()),
        f"Selected Self: {', '.join(snapshot['selected_self']) or 'none'}",
        f"Selected memories: {len(snapshot['selected_memories'])}",
        f"InnerLife thoughts: {len(snapshot['inner_life'])}",
        f"Mood: {snapshot['mood']}",
        f"Relationship: {snapshot['relationship']}",
        f"Last prompt tokens: {last.get('prompt_tokens', 'not generated')}",
        f"Last reasoning pass: {bool(last.get('used_reasoning'))}",
        f"Last reflection dirty: {bool(last.get('reflection_dirty'))}",
        f"Last reflection content-ready: {bool(last.get('reflection_content_ready'))}",
        f"Last reflection immediately inference-eligible: {bool(last.get('reflection_inference_eligible'))}",
    ]
    if verbose:
        lines.append(f"Last change: {snapshot.get('last_change')}")
        lines.append(f"Last model timing: {last.get('model_seconds', 'n/a')} seconds")
    return "\n".join(lines)
