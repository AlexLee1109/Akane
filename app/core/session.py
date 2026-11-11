"""V2 conversation lifecycle: recent dialogue and Memory → Qwen → atomic history commit."""

from __future__ import annotations

import math
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass

from app.core.config import SETTINGS
from app.core.context import ContextBuilder
from app.core.inference import (
    InferenceCancelled,
    InferenceQueueTimeout,
    InferenceRuntime,
    InferenceTiming,
)
from app.core.mind import derive_memory_change
from app.core.state import StateChangeProposal, Turn
from app.core.prompt import PromptPlan, build_dialogue_prompt
from app.core.store import get_store
from app.core.streaming import SpeakableChunker, SpeakableDispatcher
from app.core.utils import (
    OWNER_PROFILE_ID,
    canonical_profile_id,
    compact_text,
    log_performance,
    log_timing,
)

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
    final_token_at: float = 0.0
    reply_ready_at: float = 0.0
    committed_at: float = 0.0
    handed_off_at: float = 0.0


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


def _fit_prompt(
    runtime: InferenceRuntime,
    reservation,
    context,
    chat: ChatInput,
    max_tokens: int,
    initial_plan: PromptPlan | None = None,
) -> tuple[PromptPlan, int, str, float, float, dict[str, int | bool]]:
    total_recent_turns = len(context.state.recent_turns)
    recent_limit = total_recent_turns
    plan = initial_plan
    prompt_build_seconds = 0.0
    tokenization_seconds = 0.0
    initial_prompt_tokens: int | None = None
    count_text = getattr(runtime, "count_text_tokens", None)
    recent_token_counter = (
        (lambda text: count_text(text, reservation)[0])
        if callable(count_text) else None
    )
    while True:
        if plan is None:
            build_started_at = time.perf_counter()
            plan = build_dialogue_prompt(
                context,
                user_message=chat.text,
                reply_context=chat.reply_context,
                recent_limit=recent_limit,
                recent_token_counter=recent_token_counter,
                output_budget=max_tokens,
            )
            prompt_build_seconds += time.perf_counter() - build_started_at
        tokenization_started_at = time.perf_counter()
        count, method = runtime.count_prompt_tokens(plan.messages, reservation)
        tokenization_seconds += time.perf_counter() - tokenization_started_at
        if initial_prompt_tokens is None:
            initial_prompt_tokens = count
        if count + max_tokens <= SETTINGS.llama_context_window:
            return (
                plan,
                count,
                method,
                prompt_build_seconds,
                tokenization_seconds,
                {
                    "context_trimmed": recent_limit < total_recent_turns,
                    "tokens_removed": max(0, initial_prompt_tokens - count),
                    "turns_removed": max(0, total_recent_turns - recent_limit),
                },
            )
        if recent_limit == 0:
            raise RuntimeError("The selected prompt does not fit the configured context window.")
        recent_limit = max(0, recent_limit - 2)
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
    on_speakable=None,
    on_stream_end=None,
    priority: str = "owner",
    cancellation: threading.Event | None = None,
    queue_deadline: float | None = None,
    allow_tool_context: bool = True,
    runtime_situation=None,
) -> CompanionTurnResult:
    chat = chat_input
    session_started_at = time.perf_counter()
    if not streaming and (
        on_delta is not None or on_speakable is not None or on_stream_end is not None
    ):
        raise ValueError("Streaming callbacks require streaming=True.")
    store = get_store()
    runtime = InferenceRuntime.get_instance()
    handle = _SCHEDULER.begin(
        chat.profile_id,
        chat.conversation_id,
        skip_if_busy=skip_if_busy,
        cancellation=cancellation,
        queue_deadline=queue_deadline,
    )
    token_limit = SETTINGS.max_tokens
    timing = InferenceTiming()
    speech_dispatcher = (
        SpeakableDispatcher(on_speakable)
        if streaming and on_speakable is not None else None
    )
    try:
        log_performance("foreground_event", stage="request_start", at=time.time())
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
        prompt_started_at = time.perf_counter()
        plan = build_dialogue_prompt(
            context,
            user_message=chat.text,
            reply_context=chat.reply_context,
            recent_limit=len(context.state.recent_turns) or None,
            append_only=True,
            output_budget=token_limit,
        )
        prompt_build_seconds = time.perf_counter() - prompt_started_at
        wait_started_at = time.perf_counter()
        last_delta_delivered_at = 0.0
        first_eight_visible_tokens_at = 0.0
        first_speakable_chunk_at = 0.0
        stream_flushed_at = 0.0
        speakable_dispatch_seconds = 0.0
        visible_text = ""
        speakable = SpeakableChunker()
        delta_callback = on_delta
        with runtime.reserve(
            priority=priority,
            cancellation=handle.cancellation,
            queue_deadline=handle.queue_deadline,
        ) as reservation:
            reservation_acquired_at = time.perf_counter()
            log_performance("foreground_event", stage="model_acquired", at=time.time())
            count_text = getattr(runtime, "count_text_tokens", None)
            recent_token_counter = (
                (lambda text: count_text(text, reservation)[0])
                if callable(count_text) else None
            )
            history_identity = tuple(
                (message["role"], message["content"])
                for message in plan.history_messages
            )
            static_prefix_hash = plan.static_prefix_hash
            cache_fingerprint = getattr(runtime, "dialogue_cache_fingerprint", None)
            if static_prefix_hash and callable(cache_fingerprint):
                static_prefix_hash = cache_fingerprint(static_prefix_hash, reservation)
            cache_key = (
                (chat.profile_id, chat.conversation_id, static_prefix_hash)
                if static_prefix_hash else None
            )
            fast_path_check = getattr(runtime, "dialogue_fast_path_available", None)
            fast_path = bool(
                callable(fast_path_check)
                and fast_path_check(
                    cache_key,
                    history_identity,
                    max_tokens=token_limit,
                )
            )
            canonical_plan: PromptPlan | None = None

            def canonical_rebuild() -> PromptPlan:
                nonlocal canonical_plan, prompt_build_seconds
                if canonical_plan is None:
                    build_started_at = time.perf_counter()
                    canonical_plan = build_dialogue_prompt(
                        context,
                        user_message=chat.text,
                        reply_context=chat.reply_context,
                        recent_limit=len(context.state.recent_turns) or None,
                        recent_token_counter=recent_token_counter,
                        output_budget=token_limit,
                    )
                    prompt_build_seconds += time.perf_counter() - build_started_at
                return canonical_plan

            if fast_path:
                prompt_tokens = 0
                prompt_method = "deferred_direct_token_append"
                tokenization_seconds = 0.0
                trim_metrics = {
                    "context_trimmed": False, "tokens_removed": 0, "turns_removed": 0,
                }
            else:
                if not plan.canonical_complete:
                    plan = canonical_rebuild()
                (
                    plan,
                    prompt_tokens,
                    prompt_method,
                    extra_prompt_build,
                    tokenization_seconds,
                    trim_metrics,
                ) = _fit_prompt(
                    runtime, reservation, context, chat, token_limit, initial_plan=plan,
                )
                prompt_build_seconds += extra_prompt_build
            timing.prompt_tokens = prompt_tokens
            timing.prompt_token_method = prompt_method
            token_breakdown = _prompt_token_breakdown(
                runtime, reservation, plan, prompt_tokens,
            )
            static_prefix_tokens = sum(
                token_breakdown.get(name, 0)
                for name in ("identity", "soul", "stable_rules")
            )
            new_user_tokens = (
                int(count_text(chat.text.strip(), reservation)[0])
                if callable(count_text) else token_breakdown.get("current_message", 0)
            )
            debug_prompt = getattr(runtime, "debug_dialogue_prompt_once", None)
            if debug_prompt is not None:
                debug_plan = canonical_rebuild() if SETTINGS.prompt_debug else plan
                debug_prompt(
                    debug_plan.messages,
                    llm=reservation.llm,
                    token_sections=debug_plan.token_sections,
                    cache_key=cache_key,
                )
            chunks: list[str] = []

            def deliver_spoken(chunk: str) -> None:
                nonlocal delta_callback, first_eight_visible_tokens_at
                nonlocal first_speakable_chunk_at, last_delta_delivered_at, visible_text
                if not chunk:
                    return
                chunks.append(chunk)
                if streaming and delta_callback is not None:
                    try:
                        delta_callback(chunk)
                    except Exception as exc:
                        print(
                            "[Akane:streaming:delta-callback-error] "
                            f"type={type(exc).__name__}",
                            flush=True,
                        )
                        delta_callback = None
                last_delta_delivered_at = time.perf_counter()
                visible_text += chunk
                if (
                    not first_eight_visible_tokens_at
                    and len(re.findall(r"\S+", visible_text)) >= 8
                ):
                    first_eight_visible_tokens_at = last_delta_delivered_at
                for speech_chunk in speakable.feed(chunk):
                    if not first_speakable_chunk_at:
                        first_speakable_chunk_at = time.perf_counter()
                    if speech_dispatcher is not None:
                        speech_dispatcher.submit(speech_chunk)

            log_performance("foreground_event", stage="dialogue_start", at=time.time())
            for chunk in runtime.stream_messages(
                plan.messages,
                max_tokens=token_limit,
                reservation=reservation,
                cancellation=handle.cancellation,
                timing=timing,
                call_kind="dialogue",
                cache_key=cache_key,
                history_messages=history_identity,
                canonical_user_content=plan.canonical_user_content,
                turn_user_content=plan.turn_user_content,
                state_revision=plan.state_revision,
                state_sections=plan.state_sections,
                state_items=plan.state_items,
                static_prefix_tokens=static_prefix_tokens,
                new_user_tokens=new_user_tokens,
                context_trimmed=bool(trim_metrics["context_trimmed"]),
                tokens_removed=int(trim_metrics["tokens_removed"]),
                turns_removed=int(trim_metrics["turns_removed"]),
                canonical_rebuild=(canonical_rebuild if fast_path else None),
            ):
                handle.check()
                deliver_spoken(chunk)
            runtime.finalize_dialogue_reply(cache_key, "".join(chunks))
            log_performance("foreground_event", stage="dialogue_end", at=time.time())
        reservation_released_at = time.perf_counter()
        final_token_at = timing.final_token_at or last_delta_delivered_at
        reply_build_started_at = reservation_released_at
        reply = "".join(chunks)
        if not reply.strip():
            raise RuntimeError("The model returned an empty response.")
        reply_ready_at = time.perf_counter()
        speakable_dispatch_started_at = reply_ready_at
        final_speech_chunk = speakable.flush()
        if final_speech_chunk:
            if not first_speakable_chunk_at:
                first_speakable_chunk_at = time.perf_counter()
            if speech_dispatcher is not None:
                speech_dispatcher.submit(final_speech_chunk)
        if speech_dispatcher is not None:
            speech_dispatcher.close()
        speakable_dispatch_seconds = time.perf_counter() - speakable_dispatch_started_at
        handle.check()
        if streaming and on_stream_end is not None:
            try:
                on_stream_end()
            except Exception as exc:
                print(
                    "[Akane:streaming:end-callback-error] "
                    f"type={type(exc).__name__}",
                    flush=True,
                )
        stream_flushed_at = time.perf_counter()
        committed_at = time.time()
        user_turn = Turn(
            f"turn_{uuid.uuid4().hex}", chat.profile_id, chat.conversation_id,
            "user", chat.text, chat.timestamp, chat.request_id,
        )
        assistant_turn = Turn(
            f"turn_{uuid.uuid4().hex}", chat.profile_id, chat.conversation_id,
            "assistant", reply, max(committed_at, chat.timestamp), chat.request_id,
        )
        memory = None if skip_memory else derive_memory_change(
            chat.profile_id, store.memories(chat.profile_id), user_turn, assistant_turn,
            now=committed_at,
        )
        proposal = StateChangeProposal(
            chat.profile_id, turns=(user_turn, assistant_turn),
            memories=(memory,) if memory else (), origin="conversation",
        )
        persist_started_at = time.perf_counter()
        # Reset/forget and cancellation cannot race publication of a completed turn.
        with _SCHEDULER._lock:
            handle.check()
            commit = store.commit(proposal)
        persist_finished_at = time.perf_counter()
        metrics = {
            **asdict(timing),
            "generation_id": handle.generation_id,
            "revision": commit.revision,
            "selected_counts": plan.selected_counts,
            "prompt_token_breakdown": token_breakdown,
            "finish_reason": timing.finish_reason,
            "foreground_total_ms": (persist_finished_at - session_started_at) * 1000,
            "time_to_first_token_ms": ((timing.first_token_at - session_started_at) * 1000
                                       if timing.first_token_at else None),
            "decode_tokens_per_second": _tokens_per_second(timing.generated_tokens, timing.decode_seconds),
            "persist_turn_ms": (persist_finished_at - persist_started_at) * 1000,
            "applied": commit.applied,
            "rejected": commit.rejected,
        }
        with _DEBUG_LOCK:
            _TURN_DEBUG[(chat.profile_id, chat.conversation_id)] = metrics
        handed_off_at = time.perf_counter()
        return CompanionTurnResult(
            reply, handle.generation_id, final_token_at, reply_ready_at,
            persist_finished_at, handed_off_at,
        )
    except (InferenceCancelled, InferenceQueueTimeout) as exc:
        runtime.discard_dialogue_cache(chat.profile_id, chat.conversation_id, reason="turn-failed")
        raise GenerationCancelled(str(exc)) from exc
    except BaseException:
        # A generated reply that failed persistence must never become resident history.
        runtime.discard_dialogue_cache(chat.profile_id, chat.conversation_id, reason="turn-failed")
        raise
    finally:
        if speech_dispatcher is not None:
            speech_dispatcher.close()
        runtime.foreground_finished()
        _SCHEDULER.finish(handle)


def cancel_generation(conversation_id: str, profile_id: str = OWNER_PROFILE_ID) -> bool:
    return _SCHEDULER.cancel(conversation_id, profile_id)


def cancel_all_generations() -> None:
    _SCHEDULER.cancel_all()


def clear_conversation_caches(conversation_id: str, profile_id: str) -> None:
    InferenceRuntime.get_instance().discard_dialogue_cache(
        profile_id, conversation_id, reason="conversation-reset",
    )
    with _DEBUG_LOCK:
        _TURN_DEBUG.pop((profile_id, conversation_id), None)


def clear_profile_caches(profile_id: str) -> None:
    InferenceRuntime.get_instance().discard_dialogue_cache(
        profile_id, reason="profile-forgotten",
    )
    with _DEBUG_LOCK:
        for key in tuple(_TURN_DEBUG):
            if key[0] == profile_id:
                _TURN_DEBUG.pop(key, None)


def reset_conversation(conversation_id: str, profile_id: str, *, store=None) -> None:
    with _SCHEDULER._lock:
        cancel_generation(conversation_id, profile_id)
        (store if store is not None else get_store()).clear_conversation(conversation_id, profile_id)
        clear_conversation_caches(conversation_id, profile_id)


def forget_profile(profile_id: str) -> None:
    with _SCHEDULER._lock:
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
    return "\n".join((
        f"Akane v2: schema {snapshot['schema_version']}, revision {snapshot['revision']}",
        f"Recent turns: {snapshot['recent_turn_count']}",
        f"Memories: {snapshot['memory_count']} (selected {len(snapshot['selected_memories'])})",
        f"Generation: {last.get('finish_reason', 'none')}",
    ))
