"""Shared single-generation pipeline with grounding, cancellation, and safe commit."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.core.config import PROMPT_DEBUG, STREAM_CHUNK_CHARS, STREAM_FLUSH_SECONDS
from app.core.model_loader import (
    InferenceCancelled,
    InferenceQueueTimeout,
    InferenceTiming,
    ModelManager,
)
from app.core.memory import estimate_tokens
from app.core.prompt import describe_model_input
from app.core.session import (
    ChatInput,
    GenerationCancelled,
    GenerationQueueFullError,
    TurnPreparation,
    commit_turn,
    finish_turn,
    normalize_chat_input,
    prepare_turn,
    session_state_snapshot,
    timing_enabled,
)
_MAX_METRICS = 64
_METRICS: dict[str, dict[str, object]] = {}
_METRICS_LOCK = threading.Lock()

PreparedReply = TurnPreparation


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    kind: str
    generation_id: str
    text: str = ""
    reply: str = ""
    metadata: dict[str, object] | None = None


def prepare_reply(
    chat_input: ChatInput | str,
    *,
    session_id: str | None = None,
    skip_memory: bool = False,
    skip_if_busy: bool = False,
) -> PreparedReply:
    if isinstance(chat_input, str):
        chat_input = normalize_chat_input(
            text=chat_input,
            conversation_id=session_id or "popup:default",
        )
    return prepare_turn(
        chat_input,
        skip_memory=skip_memory,
        skip_if_busy=skip_if_busy,
    )


def generate_reply(prepared: PreparedReply) -> str:
    for event in _reply_events(prepared, emit_deltas=False):
        if event.kind == "done":
            return event.reply
    raise RuntimeError("Model returned no completion event.")


def stream_reply(prepared: PreparedReply):
    """Yield one grounded reply and one completion event from one model generation."""

    yield from _reply_events(prepared, emit_deltas=True)


def _reply_events(prepared: PreparedReply, *, emit_deltas: bool):
    first_delivery_at = 0.0
    output_chunks = 0
    timing = InferenceTiming(requested_at=time.perf_counter())

    try:
        prepared.handle.raise_if_cancelled()
        manager = ModelManager.get_instance()
        messages = prepared.prompt_plan.messages
        _log_model_input(
            prepared,
            generation_mode="streaming" if emit_deltas else "buffered_stream",
        )
        parts: list[str] = []
        pending: list[str] = []
        pending_chars = 0
        last_flush_at = time.monotonic()
        for text in manager.stream(
            messages,
            max_tokens=prepared.max_tokens,
            cancellation=prepared.handle.cancellation,
            queue_deadline=prepared.handle.queue_deadline,
            timing=timing,
        ):
            output_chunks += 1
            if not timing.first_token_at:
                timing.first_token_at = time.perf_counter()
            parts.append(text)
            if not emit_deltas:
                continue
            pending.append(text)
            pending_chars += len(text)
            now = time.monotonic()
            if (
                pending_chars >= STREAM_CHUNK_CHARS
                or now - last_flush_at >= STREAM_FLUSH_SECONDS
            ):
                prepared.handle.raise_if_cancelled()
                if not first_delivery_at:
                    first_delivery_at = time.perf_counter()
                yield GenerationEvent(
                    "delta",
                    prepared.generation_id,
                    text="".join(pending),
                )
                pending.clear()
                pending_chars = 0
                last_flush_at = now

        if emit_deltas and pending:
            prepared.handle.raise_if_cancelled()
            if not first_delivery_at:
                first_delivery_at = time.perf_counter()
            yield GenerationEvent(
                "delta",
                prepared.generation_id,
                text="".join(pending),
            )
        prepared.handle.raise_if_cancelled()
        if not timing.model_started_at:
            timing.model_started_at = timing.requested_at
        if not timing.model_finished_at:
            timing.model_finished_at = time.perf_counter()

        postprocess_started_at = time.perf_counter()
        reply = "".join(parts).strip()
        if not reply:
            raise RuntimeError("Model returned no visible reply.")
        postprocess_seconds = time.perf_counter() - postprocess_started_at

        prepared.handle.raise_if_cancelled()
        persistence_started_at = time.perf_counter()
        commit_turn(prepared, reply)
        persistence_seconds = time.perf_counter() - persistence_started_at
        _remember_metrics(prepared)
        _timing_log(
            prepared,
            reply,
            timing=timing,
            first_delivery_at=first_delivery_at,
            output_chunks=output_chunks,
            postprocess_seconds=postprocess_seconds,
            persistence_seconds=persistence_seconds,
        )
        yield GenerationEvent(
            "done",
            prepared.generation_id,
            reply=reply,
            metadata={
                "estimated_prompt_tokens": prepared.prompt_plan.estimated_tokens,
                "tokenized_prompt_tokens": timing.prompt_tokens or None,
            },
        )
    except InferenceCancelled as exc:
        raise GenerationCancelled(str(exc)) from exc
    except InferenceQueueTimeout as exc:
        raise GenerationQueueFullError(str(exc)) from exc
    finally:
        finish_turn(prepared)


def debug_state_report(conversation_id: str | None, profile_id: str | None = None) -> str:
    conversation = str(conversation_id or "popup:default")
    snapshot = session_state_snapshot(conversation, profile_id)
    memory = snapshot.get("memory") or {}
    akane = snapshot.get("akane") or {}
    emotion = akane.get("emotion") or {}
    with _METRICS_LOCK:
        metrics = dict(_METRICS.get(conversation, {}))
    lines = [
        "Akane debug state",
        "- personality: stable Akane",
        f"- intent: {_debug_text(memory.get('recent_intent'), 40) or 'casual'}",
        f"- emotional context: {_debug_text(emotion.get('dominant'), 40) or 'neutral'}",
        f"- focus: {_debug_text(memory.get('recent_topic'), 80) or 'none'}",
        f"- prompt tokens (estimated): {int(metrics.get('prompt_tokens') or 0)}",
        f"- code context attached: {'yes' if metrics.get('code_context_attached') else 'no'}",
    ]
    return "\n".join(lines)


def _remember_metrics(
    prepared: PreparedReply,
) -> None:
    with _METRICS_LOCK:
        _METRICS[prepared.session_id] = {
            "prompt_tokens": prepared.prompt_plan.estimated_tokens,
            "code_context_attached": prepared.code_context_attached,
            "updated_at": time.time(),
        }
        if len(_METRICS) > _MAX_METRICS:
            oldest = min(
                _METRICS,
                key=lambda key: float(_METRICS[key].get("updated_at") or 0.0),
            )
            _METRICS.pop(oldest, None)


def _log_model_input(
    prepared: PreparedReply,
    *,
    generation_mode: str,
) -> None:
    if not PROMPT_DEBUG:
        return
    earlier_dialogue = prepared.memory_context.earlier_dialogue
    summary_turns = max(0, len(earlier_dialogue.splitlines()) - 1) if earlier_dialogue else 0
    metadata = describe_model_input(
        prepared.prompt_plan.messages,
        transport=prepared.chat_input.source,
        conversation_id=prepared.chat_input.conversation_id,
        loaded_recent_turns=len(prepared.memory_context.recent_turns),
        summary_turns=summary_turns,
        current_user_text=prepared.chat_input.text,
        generation_mode=generation_mode,
    )
    print(f"[Akane:model-input] {metadata}", flush=True)


def _timing_log(
    prepared: PreparedReply,
    reply: str,
    *,
    timing: InferenceTiming,
    first_delivery_at: float = 0.0,
    output_chunks: int = 0,
    postprocess_seconds: float = 0.0,
    persistence_seconds: float = 0.0,
) -> None:
    if not timing_enabled():
        return
    done = time.perf_counter()
    model_seconds = max(0.0, timing.model_finished_at - timing.model_started_at)
    decode_seconds = max(0.0, timing.model_finished_at - timing.first_token_at)
    prompt_eval_seconds = max(
        0.0,
        timing.first_token_at
        - timing.model_started_at
        - timing.chat_template_seconds
        - timing.prompt_tokenization_seconds,
    )
    output_tokens = estimate_tokens(reply)
    fields = [
        "[Akane:timing]",
        f"total={done - prepared.started_at:.3f}s",
        f"preprocess={getattr(prepared, 'preprocess_seconds', prepared.prompt_seconds):.3f}s",
        f"prompt={prepared.prompt_seconds:.3f}s",
        f"memory={getattr(prepared, 'memory_seconds', 0.0):.3f}s",
        f"subtext={getattr(prepared, 'subtext_seconds', 0.0):.3f}s",
        f"queue={max(0.0, timing.model_started_at - timing.requested_at):.3f}s",
        f"chat_template={timing.chat_template_seconds:.3f}s",
        f"tokenization={timing.prompt_tokenization_seconds:.3f}s",
        f"prompt_eval={prompt_eval_seconds:.3f}s",
        f"model_total={model_seconds:.3f}s",
        f"postprocess={postprocess_seconds:.3f}s",
        f"persistence={persistence_seconds:.3f}s",
        f"prompt_tokens_est={prepared.prompt_plan.estimated_tokens}",
        f"prompt_tokens_backend={timing.prompt_tokens}",
        f"output_tokens_est={output_tokens}",
        f"output_chars={len(reply)}",
    ]
    if decode_seconds > 0.0:
        fields.append(f"tokens_per_second_est={output_tokens / decode_seconds:.2f}")
    if output_chunks:
        fields.append(f"stream_chunks={output_chunks}")
    if timing.first_token_at:
        fields.insert(1, f"first_token={timing.first_token_at - prepared.started_at:.3f}s")
    if first_delivery_at:
        fields.insert(2, f"first_delivery={first_delivery_at - prepared.started_at:.3f}s")
    print(" ".join(fields), flush=True)


def _debug_text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or text[:limit]
