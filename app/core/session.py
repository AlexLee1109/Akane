"""Foreground conversation orchestration over the v2 ownership boundaries."""

from __future__ import annotations

import math
import re
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
from app.core.mind import derive_durable_changes, self_development_state
from app.core.state import StateChangeProposal, Turn
from app.core.prompt import PromptPlan, build_dialogue_prompt
from app.core.store import get_store
from app.core.streaming import SemanticSidecarFilter, SpeakableChunker, SpeakableDispatcher
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
                recent_limit=recent_limit or None,
                recent_token_counter=recent_token_counter,
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
    on_speakable=None,
    on_stream_end=None,
    priority: str = "owner",
    cancellation: threading.Event | None = None,
    queue_deadline: float | None = None,
    allow_tool_context: bool = True,
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
            sidecar = SemanticSidecarFilter()

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
                deliver_spoken(sidecar.feed(chunk))
            deliver_spoken(sidecar.finish())
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
        development_started_at = time.perf_counter()
        memory_changes = ()
        self_changes = ()
        experience_changes = ()
        if not skip_memory:
            memory_changes, self_changes, experience_changes = derive_durable_changes(
                chat.profile_id,
                store.self_items(chat.profile_id),
                store.memories(chat.profile_id),
                user_turn,
                assistant_turn,
                experiences=store.experiences(chat.profile_id),
                semantic_evidence=sidecar.semantic_evidence,
                now=committed_at,
            )
        development_finished_at = time.perf_counter()
        proposal = StateChangeProposal(
            chat.profile_id,
            turns=(user_turn, assistant_turn),
            memories=memory_changes,
            experiences=experience_changes,
            self_items=self_changes,
            origin="conversation",
        )
        persist_started_at = time.perf_counter()
        commit = store.commit(proposal)
        persist_finished_at = time.perf_counter()
        new_experience = (
            experience_changes[0]
            if experience_changes and "experience:form" in commit.applied else None
        )
        new_self = (
            self_changes[0]
            if self_changes and f"self:{self_changes[0].action}" in commit.applied else None
        )
        model_finished_at = timing.model_finished_at or reservation_released_at
        log_timing(
            "session",
            store_snapshot=context_builder.last_timing.get("store_snapshot_seconds", 0.0),
            relevance_selection=context_builder.last_timing.get(
                "relevance_selection_seconds", 0.0,
            ),
            context_build=context_finished_at - context_started_at,
            prompt_build=prompt_build_seconds,
            tokenization=tokenization_seconds,
            model_wait=reservation_acquired_at - wait_started_at,
            prefill=timing.prefill_seconds,
            decode=timing.decode_seconds,
            final_eos_or_stop=(
                max(0.0, timing.stop_observed_at - final_token_at)
                if timing.stop_observed_at and final_token_at else 0.0
            ),
            stop_to_model_end=(
                max(0.0, model_finished_at - timing.stop_observed_at)
                if timing.stop_observed_at else 0.0
            ),
            finalize_stream=max(0.0, reservation_released_at - model_finished_at),
            speakable_dispatch=speakable_dispatch_seconds,
            reply_build=reply_ready_at - reply_build_started_at,
            durable_state_derivation=development_finished_at - development_started_at,
            persist_turn=persist_finished_at - persist_started_at,
            final_token_to_return=max(0.0, persist_finished_at - final_token_at),
            total=persist_finished_at - session_started_at,
        )
        metrics = {
            "generation_id": handle.generation_id,
            "prompt_tokens": timing.prompt_tokens,
            "prompt_token_method": timing.prompt_token_method,
            "prompt_token_breakdown": token_breakdown,
            "static_prompt_hash": plan.static_prefix_hash,
            "selected_counts": plan.selected_counts,
            "durable_updates": [
                item for item in commit.applied
                if item.startswith(("self:", "memory:", "experience:"))
            ],
            "new_experience": (
                {
                    "id": new_experience.id,
                    "kind": new_experience.kind,
                    "topic": new_experience.topic,
                    "why": new_experience.reason,
                    "linked_self_item_ids": list(new_experience.self_item_ids),
                }
                if new_experience is not None else None
            ),
            "self_development": (
                {
                    "id": new_self.item.id,
                    "action": new_self.action,
                    "state": self_development_state(new_self.item),
                    "strength": new_self.item.strength,
                    "confidence": new_self.item.confidence,
                    "why": new_self.item.reason,
                    "support_experience_ids": [
                        source for source in new_self.item.source_ids
                        if source.startswith("experience_")
                    ],
                    "contradiction_experience_ids": [
                        source for source in new_self.item.contradiction_ids
                        if source.startswith("experience_")
                    ],
                }
                if new_self is not None and new_self.item is not None else None
            ),
            "revision": commit.revision,
            "store_snapshot_ms": context_builder.last_timing.get("store_snapshot_seconds", 0.0) * 1000,
            "relevance_selection_ms": context_builder.last_timing.get(
                "relevance_selection_seconds", 0.0,
            ) * 1000,
            "memory_candidates": int(context_builder.last_timing.get("memory_candidates", 0)),
            "self_candidates": int(context_builder.last_timing.get("self_candidates", 0)),
            "experience_candidates": int(
                context_builder.last_timing.get("experience_candidates", 0)
            ),
            "context_build_ms": (context_finished_at - context_started_at) * 1000,
            "prompt_build_ms": prompt_build_seconds * 1000,
            "tokenization_ms": tokenization_seconds * 1000,
            "model_wait_ms": (reservation_acquired_at - wait_started_at) * 1000,
            "foreground_queue_wait_ms": timing.foreground_queue_wait_seconds * 1000,
            "model_lock_wait_ms": timing.model_lock_wait_seconds * 1000,
            "prefill_ms": timing.prefill_seconds * 1000,
            "prefill_tokens": timing.prefill_tokens,
            "previous_prompt_tokens": timing.previous_prompt_tokens,
            "current_prompt_tokens": timing.current_prompt_tokens,
            "actual_token_lcp": timing.actual_token_lcp,
            "actual_lcp_percent": timing.actual_lcp_percent,
            "tokens_after_lcp": timing.tokens_after_lcp,
            "backend_prompt_eval_tokens": timing.backend_prompt_eval_tokens,
            "backend_reused_tokens": timing.backend_reused_tokens,
            "backend_compute_graph_reuses": timing.backend_compute_graph_reuses,
            "llama_n_tokens_before": timing.llama_n_tokens_before,
            "llama_n_tokens_after": timing.llama_n_tokens_after,
            "cache_owner": timing.cache_owner,
            "cache_epoch": timing.cache_epoch,
            "cache_invalidated": timing.cache_invalidated,
            "cache_invalidated_reason": timing.cache_invalidated_reason,
            "cache_invalidated_from_token": timing.cache_invalidated_from_token,
            "prompt_architecture": timing.prompt_architecture,
            "static_prefix_tokens": timing.static_prefix_tokens,
            "historical_prefix_tokens": timing.historical_prefix_tokens,
            "dynamic_state_tokens": timing.dynamic_state_tokens,
            "new_user_tokens": timing.new_user_tokens,
            "self_tokens": timing.self_tokens,
            "memory_tokens": timing.memory_tokens,
            "experience_tokens": timing.experience_tokens,
            "time_tokens": timing.time_tokens,
            "code_context_tokens": timing.code_context_tokens,
            "wrapper_tokens": timing.wrapper_tokens,
            "chat_template_overhead_tokens": timing.chat_template_overhead_tokens,
            "state_revision_before": timing.state_revision_before,
            "state_revision_after": timing.state_revision_after,
            "state_payload_tokens": timing.state_payload_tokens,
            "state_payload_mode": timing.state_payload_mode,
            "context_trimmed": timing.context_trimmed,
            "tokens_removed": timing.tokens_removed,
            "turns_removed": timing.turns_removed,
            "cache_rebuild_reason": timing.cache_rebuild_reason,
            "logical_prompt_tokens": timing.logical_prompt_tokens,
            "resident_sequence_tokens": timing.resident_sequence_tokens,
            "appended_tokens": timing.appended_tokens,
            "state_delta_tokens": timing.state_delta_tokens,
            "template_overhead_tokens": timing.template_overhead_tokens,
            "health_before": timing.health_before,
            "health_after_prefill": timing.health_after_prefill,
            "health_after_decode": timing.health_after_decode,
            "sequence_state_bytes": timing.sequence_state_bytes,
            "prompt_eval_ms": timing.prefill_seconds * 1000,
            "prefill_tokens_per_second": _tokens_per_second(
                timing.prefill_tokens, timing.prefill_seconds,
            ),
            "time_to_first_token_ms": (
                max(0.0, timing.first_token_at - session_started_at) * 1000
                if timing.first_token_at else None
            ),
            "decode_ms": timing.decode_seconds * 1000,
            "generation_ms": max(
                0.0, timing.model_finished_at - timing.model_started_at,
            ) * 1000,
            "generated_tokens": timing.generated_tokens,
            "generated_token_method": timing.generated_token_method,
            "finish_reason": timing.finish_reason or "unavailable",
            "decode_tokens_per_second": _tokens_per_second(
                timing.generated_tokens, timing.decode_seconds,
            ),
            "time_to_first_8_visible_tokens_ms": (
                max(0.0, first_eight_visible_tokens_at - session_started_at) * 1000
                if first_eight_visible_tokens_at else None
            ),
            "visible_token_timing_method": "whitespace_estimate",
            "time_to_first_speakable_chunk_ms": (
                max(0.0, first_speakable_chunk_at - session_started_at) * 1000
                if first_speakable_chunk_at else None
            ),
            "final_token_to_reply_ready_ms": (
                max(0.0, reply_ready_at - final_token_at) * 1000
                if final_token_at else None
            ),
            "final_eos_or_stop_ms": (
                max(0.0, timing.stop_observed_at - final_token_at) * 1000
                if timing.stop_observed_at and final_token_at else None
            ),
            "stop_to_model_end_ms": (
                max(0.0, model_finished_at - timing.stop_observed_at) * 1000
                if timing.stop_observed_at else None
            ),
            "reply_persistence_ms": (persist_finished_at - persist_started_at) * 1000,
            "durable_state_derivation_ms": (
                development_finished_at - development_started_at
            ) * 1000,
            "speakable_dispatch_ms": speakable_dispatch_seconds * 1000,
            "kv_finalize_ms": max(0.0, reservation_released_at - model_finished_at) * 1000,
            "store_lock_wait_ms": float(commit.timings.get("lock_wait", 0.0)) * 1000,
            "store_lock_held_ms": float(commit.timings.get("lock_held", 0.0)) * 1000,
            "state_copy_ms": float(commit.timings.get("copy", 0.0)) * 1000,
            "conversation_mutation_ms": float(
                commit.timings.get("conversation_mutation", 0.0)
            ) * 1000,
            "state_validation_ms": float(commit.timings.get("validate", 0.0)) * 1000,
            "state_serialization_ms": float(commit.timings.get("serialize", 0.0)) * 1000,
            "state_json_bytes": int(commit.timings.get("json_bytes", 0)),
            "atomic_write_ms": float(commit.timings.get("write", 0.0)) * 1000,
            "fsync_ms": float(commit.timings.get("fsync", 0.0)) * 1000,
            "atomic_replace_ms": float(commit.timings.get("replace", 0.0)) * 1000,
            "foreground_total_ms": (persist_finished_at - session_started_at) * 1000,
            "visible_response_ms": (
                max(0.0, stream_flushed_at - session_started_at) * 1000
                if streaming else (persist_finished_at - session_started_at) * 1000
            ),
            "last_token_to_visible_complete_ms": (
                max(0.0, stream_flushed_at - final_token_at) * 1000
                if streaming and final_token_at else None
            ),
            "post_generation_ms": (
                max(0.0, persist_finished_at - final_token_at) * 1000
                if final_token_at else None
            ),
            "model_calls": {"dialogue": 1},
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
        runtime_status = getattr(runtime, "status", lambda: {})()
        for name in (
            "foreground_direct_append_count",
            "foreground_canonical_rebuild_count",
        ):
            metrics[name] = runtime_status.get(name)
        log_performance(
            "foreground",
            type="chat",
            input=metrics["prompt_tokens"],
            prefill_eval=metrics["backend_prompt_eval_tokens"],
            prefill_tok_s=metrics["prefill_tokens_per_second"],
            actual_token_lcp=metrics["actual_token_lcp"],
            backend_reused_prompt_tokens=metrics["backend_reused_tokens"],
            generated=metrics["generated_tokens"],
            decode_tok_s=metrics["decode_tokens_per_second"],
            foreground_queue_wait_ms=metrics["foreground_queue_wait_ms"],
            model_lock_wait_ms=metrics["model_lock_wait_ms"],
            ttft_ms=metrics["time_to_first_token_ms"],
            cache_invalidated=metrics["cache_invalidated"],
            cache_invalidated_reason=metrics["cache_invalidated_reason"],
            cache_invalidated_from_token=metrics["cache_invalidated_from_token"],
            prompt_architecture=metrics["prompt_architecture"],
            foreground_direct_append_count=metrics["foreground_direct_append_count"],
            foreground_canonical_rebuild_count=metrics["foreground_canonical_rebuild_count"],
            total_ms=metrics["foreground_total_ms"],
            finish=metrics["finish_reason"],
        )
        with _DEBUG_LOCK:
            _TURN_DEBUG[(chat.profile_id, chat.conversation_id)] = metrics
        log_performance("foreground_event", stage="foreground_complete", at=time.time())
        handed_off_at = time.perf_counter()
        metrics["session_handoff_ms"] = max(0.0, handed_off_at - persist_finished_at) * 1000
        metrics["final_token_to_session_handoff_ms"] = (
            max(0.0, handed_off_at - final_token_at) * 1000
            if final_token_at else None
        )
        metrics["post_generation_ms"] = metrics["final_token_to_session_handoff_ms"]
        metrics["post_generation_seconds"] = (
            max(0.0, handed_off_at - final_token_at)
            if final_token_at else None
        )
        return CompanionTurnResult(
            reply,
            handle.generation_id,
            final_token_at,
            reply_ready_at,
            persist_finished_at,
            handed_off_at,
        )
    except (InferenceCancelled, InferenceQueueTimeout) as exc:
        raise GenerationCancelled(str(exc)) from exc
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
    new_experience = last.get("new_experience")
    self_development = last.get("self_development")
    self_counts = snapshot["self_counts"]
    development_counts = snapshot["self_development_counts"]
    lines = [
        f"Akane v2 state: schema {snapshot['schema_version']}, revision {snapshot['revision']}",
        f"Recent turns: {snapshot['recent_turn_count']}",
        f"Total SelfItems: {snapshot['self_count']}",
        "Self: " + ", ".join(f"{kind}={count}" for kind, count in self_counts.items()),
        "Development: " + ", ".join(
            f"{state}={count}" for state, count in development_counts.items()
        ),
        f"Selected Self: {', '.join(snapshot['selected_self']) or 'none'}",
        f"Selected memories: {len(snapshot['selected_memories'])}",
        f"Experiences: {snapshot['experience_count']}",
        f"Selected experiences: {len(snapshot['selected_experiences'])}",
        (
            f"New experience: {new_experience['id']} because {new_experience['why']}; "
            f"Self links: {', '.join(new_experience['linked_self_item_ids']) or 'none'}"
            if isinstance(new_experience, dict) else "New experience: none"
        ),
        (
            f"Self development: {self_development['action']} {self_development['id']} "
            f"→ {self_development['state']} "
            f"({self_development['strength']:.2f}/{self_development['confidence']:.2f}); "
            f"because {self_development['why']}; "
            f"evidence: {', '.join(self_development['support_experience_ids']) or 'none'}; "
            f"contradictions: "
            f"{', '.join(self_development['contradiction_experience_ids']) or 'none'}"
            if isinstance(self_development, dict) else "Self development: none"
        ),
        f"Last prompt tokens: {last.get('prompt_tokens', 'not generated')}",
        f"Last durable updates: {', '.join(last.get('durable_updates', ())) or 'none'}",
    ]
    if verbose:
        lines.append(f"Last change: {snapshot.get('last_change')}")
        lines.append(f"Last model timing: {last.get('model_seconds', 'n/a')} seconds")
    return "\n".join(lines)
