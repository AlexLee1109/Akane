"""One-call generation pipeline with streaming, cancellation, and safe commit."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.core.config import PROMPT_DEBUG, STREAM_CHUNK_CHARS, STREAM_FLUSH_SECONDS
from app.core.model_loader import InferenceCancelled, InferenceQueueTimeout, ModelManager
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
    model_started_at = time.perf_counter()
    try:
        prepared.handle.raise_if_cancelled()
        _log_model_input(prepared, generation_mode="non_streaming")
        raw = ModelManager.get_instance().complete(
            prepared.messages,
            max_tokens=prepared.max_tokens,
            cancellation=prepared.handle.cancellation,
            queue_deadline=prepared.handle.queue_deadline,
        )
        prepared.handle.raise_if_cancelled()
        reply = str(raw or "").strip()
        if not reply:
            raise RuntimeError("Model returned no visible reply.")
        commit_turn(prepared, reply)
        _remember_metrics(prepared, reply)
        _timing_log(prepared, reply, model_started_at=model_started_at)
        return reply
    except InferenceCancelled as exc:
        raise GenerationCancelled(str(exc)) from exc
    except InferenceQueueTimeout as exc:
        raise GenerationQueueFullError(str(exc)) from exc
    finally:
        finish_turn(prepared)


def stream_reply(prepared: PreparedReply):
    """Yield bounded text chunks and one completion event."""

    parts: list[str] = []
    pending: list[str] = []
    pending_chars = 0
    first_delta_at = 0.0
    last_flush_at = time.monotonic()
    model_started_at = time.perf_counter()
    try:
        prepared.handle.raise_if_cancelled()
        _log_model_input(prepared, generation_mode="streaming")
        for text in ModelManager.get_instance().stream(
            prepared.messages,
            max_tokens=prepared.max_tokens,
            cancellation=prepared.handle.cancellation,
            queue_deadline=prepared.handle.queue_deadline,
        ):
            prepared.handle.raise_if_cancelled()
            if not text:
                continue
            if not first_delta_at:
                first_delta_at = time.perf_counter()
            parts.append(text)
            pending.append(text)
            pending_chars += len(text)
            now = time.monotonic()
            if pending_chars >= STREAM_CHUNK_CHARS or now - last_flush_at >= STREAM_FLUSH_SECONDS:
                chunk = "".join(pending)
                pending.clear()
                pending_chars = 0
                last_flush_at = now
                yield GenerationEvent("delta", prepared.generation_id, text=chunk)

        prepared.handle.raise_if_cancelled()
        if pending:
            yield GenerationEvent("delta", prepared.generation_id, text="".join(pending))
        reply = "".join(parts).strip()
        if not reply:
            raise RuntimeError("Model returned no visible reply.")
        commit_turn(prepared, reply)
        _remember_metrics(prepared, reply)
        _timing_log(
            prepared,
            reply,
            model_started_at=model_started_at,
            first_delta_at=first_delta_at,
        )
        yield GenerationEvent(
            "done",
            prepared.generation_id,
            reply=reply,
            metadata={"estimated_prompt_tokens": prepared.prompt_plan.estimated_tokens},
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


def _remember_metrics(prepared: PreparedReply, reply: str) -> None:
    with _METRICS_LOCK:
        _METRICS[prepared.session_id] = {
            "user_chars": len(prepared.user_text),
            "reply_chars": len(reply),
            "prompt_tokens": prepared.prompt_plan.estimated_tokens,
            "prompt_sections": dict(prepared.prompt_plan.section_tokens),
            "code_context_attached": prepared.code_context_attached,
            "updated_at": time.time(),
        }
        if len(_METRICS) > _MAX_METRICS:
            oldest = min(
                _METRICS,
                key=lambda key: float(_METRICS[key].get("updated_at") or 0.0),
            )
            _METRICS.pop(oldest, None)


def _log_model_input(prepared: PreparedReply, *, generation_mode: str) -> None:
    if not PROMPT_DEBUG:
        return
    earlier_dialogue = prepared.memory_context.earlier_dialogue
    summary_turns = max(0, len(earlier_dialogue.splitlines()) - 1) if earlier_dialogue else 0
    metadata = describe_model_input(
        prepared.messages,
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
    model_started_at: float,
    first_delta_at: float = 0.0,
) -> None:
    if not timing_enabled():
        return
    done = time.perf_counter()
    fields = [
        "[Akane:timing]",
        f"total={done - prepared.started_at:.3f}s",
        f"prompt={prepared.prompt_seconds:.3f}s",
        f"queue_to_model={model_started_at - prepared.started_at:.3f}s",
        f"generation={done - model_started_at:.3f}s",
        f"prompt_tokens_est={prepared.prompt_plan.estimated_tokens}",
        f"output_chars={len(reply)}",
    ]
    if first_delta_at:
        fields.insert(1, f"first_token={first_delta_at - prepared.started_at:.3f}s")
    print(" ".join(fields), flush=True)


def _debug_text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or text[:limit]
