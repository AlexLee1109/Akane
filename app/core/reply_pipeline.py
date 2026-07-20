"""Shared single-generation pipeline with grounding, cancellation, and safe commit."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.core.config import PROMPT_DEBUG, STREAM_CHUNK_CHARS, STREAM_FLUSH_SECONDS
from app.core.life import life_evolution_debug
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
_DEBUG_SOURCE_LABELS = {
    "identity": "Identity",
    "soul": "Soul",
    "hard_constraints": "Hard rules",
    "dynamic_guidance": "Dynamic guidance",
    "earlier_user": "Recent conversation",
    "earlier_assistant": "Recent conversation",
    "recent_user": "Recent conversation",
    "recent_assistant": "Recent conversation",
    "relationship": "Relationship context",
    "retrieved_memory": "Memory",
    "memory": "Memory",
    "preference_continuity": "Memory",
    "current_user": "Current message",
    "editor_context": "Code context",
    "code_context": "Code context",
    "reply_quote": "Reply context",
    "life_context": "Activity context",
    "date_time": "Date and time",
}

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
    exact_tokens: bool = True,
) -> TurnPreparation:
    if isinstance(chat_input, str):
        chat_input = normalize_chat_input(
            text=chat_input,
            conversation_id=session_id or "popup:default",
        )
    return prepare_turn(
        chat_input,
        skip_memory=skip_memory,
        skip_if_busy=skip_if_busy,
        token_counter=(
            ModelManager.get_instance().count_prompt_tokens if exact_tokens else None
        ),
    )


def generate_reply(prepared: TurnPreparation) -> str:
    for event in _reply_events(prepared, emit_deltas=False):
        if event.kind == "done":
            return event.reply
    raise RuntimeError("Model returned no completion event.")


def stream_reply(prepared: TurnPreparation):
    """Yield one grounded reply and one completion event from one model generation."""

    yield from _reply_events(prepared, emit_deltas=True)


def commit_reply(
    prepared: TurnPreparation,
    reply: str,
    *,
    timing: InferenceTiming | None = None,
) -> None:
    """Commit a completed deterministic or generated reply and record diagnostics."""

    commit_turn(prepared, reply)
    _remember_metrics(prepared, committed=True, timing=timing)


def _reply_events(prepared: TurnPreparation, *, emit_deltas: bool):
    first_delivery_at = 0.0
    output_chunks = 0
    timing = InferenceTiming(requested_at=time.perf_counter())

    try:
        _remember_metrics(prepared, committed=False)
        prepared.handle.raise_if_cancelled()
        if (
            prepared.chat_input.autonomous
            and not prepared.turn_context.initiative_worthwhile
        ):
            yield GenerationEvent(
                "done",
                prepared.generation_id,
                metadata={"skipped": "no worthwhile initiative"},
            )
            return
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
        commit_reply(prepared, reply, timing=timing)
        persistence_seconds = time.perf_counter() - persistence_started_at
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
                "rendered_prompt_tokens": prepared.prompt_plan.rendered_prompt_tokens,
                "backend_observed_prompt_tokens": timing.prompt_tokens or None,
                "prompt_token_count_is_exact": prepared.prompt_plan.token_count_is_exact,
            },
        )
    except InferenceCancelled as exc:
        raise GenerationCancelled(str(exc)) from exc
    except InferenceQueueTimeout as exc:
        raise GenerationQueueFullError(str(exc)) from exc
    finally:
        finish_turn(prepared)


def debug_state_report(
    conversation_id: str | None,
    profile_id: str | None = None,
    *,
    verbose: bool = False,
) -> str:
    conversation = str(conversation_id or "popup:default")
    snapshot = session_state_snapshot(
        conversation,
        profile_id,
        preview_time=True,
    )
    memory = snapshot.get("memory") or {}
    akane = snapshot.get("akane") or {}
    emotion = akane.get("emotion") or {}
    with _METRICS_LOCK:
        metrics = dict(_METRICS.get(conversation, {}))
    time_trace = snapshot.get("debug_time_trace")
    preview_elapsed = (
        float(time_trace.get("elapsed_seconds") or 0.0)
        if isinstance(time_trace, dict)
        else 0.0
    )
    use_time_preview = not metrics or preview_elapsed >= 30 * 60
    if isinstance(time_trace, dict) and use_time_preview:
        for name in (
            "elapsed_seconds",
            "elapsed_decay_delta",
            "circadian_target",
            "circadian_delta",
            "ambient_delta",
            "event_delta",
            "event_ids",
            "resulting_persistent_mood",
        ):
            metrics[name] = time_trace.get(name, metrics.get(name))
        metrics["debug_time_candidate"] = bool(
            snapshot.get("debug_time_candidate")
        )
    life_trace = snapshot.get("debug_life_trace")
    if isinstance(life_trace, dict) and use_time_preview:
        metrics["life_trace"] = life_trace
    prompt_debug = metrics.get("prompt_debug")
    prompt_debug = prompt_debug if isinstance(prompt_debug, dict) else {}
    versions = prompt_debug.get("persona_versions")
    versions = versions if isinstance(versions, dict) else {}
    category_tokens = prompt_debug.get("sections")
    category_tokens = category_tokens if isinstance(category_tokens, dict) else {}
    source_roles = prompt_debug.get("source_roles")
    source_roles = source_roles if isinstance(source_roles, dict) else {}
    activity = akane.get("activity")
    activity = activity if isinstance(activity, dict) else {}
    activity_source = _debug_text(metrics.get("grounded_activity_source"), 32)
    if use_time_preview or _debug_absent(activity_source):
        activity_source = _debug_text(activity.get("source"), 32)
    sources = prompt_debug.get("sources")
    sources = sources if isinstance(sources, (list, tuple)) else ()
    trimmed = prompt_debug.get("trimmed")
    trimmed = trimmed if isinstance(trimmed, (list, tuple)) else ()
    disposition = _debug_disposition(
        metrics.get("final_disposition") or emotion.get("dominant")
    )
    current_mood = _debug_name(emotion.get("mood") or metrics.get("prior_mood"))
    reaction = _debug_reaction(
        metrics.get("contextual_reaction"),
        metrics.get("detected_signal"),
    )
    short_emotion = _debug_scored_value(metrics.get("active_emotion"))
    mood_changes = _debug_mood_changes(
        metrics.get("state_changes"),
        metrics.get("decay_applied"),
    )
    focus = _debug_text(memory.get("recent_topic"), 80)
    included = _debug_source_names(sources)
    history_messages = sum(
        str(source).startswith(("earlier_", "recent_")) for source in sources
    )
    prompt_tokens, count_label = _debug_prompt_tokens(metrics)
    reserved_value = prompt_debug.get("reserved_output_tokens")
    reserved_tokens = int(reserved_value) if reserved_value is not None else None
    window_value = prompt_debug.get("context_window")
    context_window = int(window_value) if window_value is not None else None
    total_reserved = (
        prompt_tokens + reserved_tokens
        if prompt_tokens is not None and reserved_tokens is not None
        else None
    )
    available = (
        max(0, context_window - total_reserved)
        if context_window is not None and total_reserved is not None
        else None
    )
    interface = _debug_name(metrics.get("interface"))
    activity_present = not _debug_absent(activity_source)
    activity_description = _debug_text(activity.get("description"), 80)
    activity_is_recent = (
        activity_source == "offscreen_schedule"
        or _debug_text(activity.get("status"), 24).lower() == "completed"
    )
    thought = _debug_name(metrics.get("current_thought_source"))
    follow_up = _debug_name(metrics.get("follow_up_tendency"))
    elapsed = float(metrics.get("elapsed_seconds") or 0.0)
    circadian_lines = _debug_delta_lines(metrics.get("circadian_delta"))
    ambient_lines = _debug_delta_lines(metrics.get("ambient_delta"))
    event_count = len(metrics.get("event_ids") or ())
    signal_label, _signal_score = _debug_scored_parts(metrics.get("detected_signal"))
    signal_confidence = float(metrics.get("signal_confidence") or 0.0)
    activation_threshold = float(metrics.get("activation_threshold") or 0.0)
    emotion_applied = bool(
        signal_label
        and signal_label != "neutral"
        and signal_confidence >= activation_threshold
    )
    life_metrics = metrics.get("life_trace")
    life_metrics = life_metrics if isinstance(life_metrics, dict) else {}
    previous_life = life_metrics.get("previous_activity")
    previous_life = previous_life if isinstance(previous_life, dict) else {}
    current_life = life_metrics.get("current_activity")
    current_life = current_life if isinstance(current_life, dict) else {}
    life_need_changes = _debug_delta_lines(life_metrics.get("need_changes"))
    life_mood_effects = _debug_delta_lines(life_metrics.get("mood_effects"))
    lines = [
        "Akane Debug",
        "",
        "Request",
        f"  Interface: {interface}",
        f"  Intent: {_debug_name(memory.get('recent_intent'))}",
        f"  Focus: {_debug_quoted(focus)}",
        f"  State Committed: {_debug_bool(metrics.get('committed'))}",
        "",
        "Response State",
        f"  Mood: {current_mood}",
        f"  Reaction: {reaction}",
        f"  Disposition: {disposition}",
        f"  Short Emotion: {short_emotion}",
    ]
    if mood_changes:
        lines.append("  Mood Changes:")
        lines.extend(f"    - {change}" for change in mood_changes)
    else:
        lines.append("  Mood Changes: None")
    lines.extend(("", "Time Evolution", f"  Elapsed: {_debug_duration(elapsed)}"))
    if metrics.get("debug_time_candidate"):
        lines.append("  Candidate Status: Proposed, not committed")
    meaningful_time_change = bool(circadian_lines or ambient_lines or event_count)
    if meaningful_time_change:
        lines.append(
            "  Circadian Effect: "
            + (", ".join(circadian_lines) if circadian_lines else "None")
        )
        lines.append(
            "  Ambient Drift: "
            + (", ".join(ambient_lines) if ambient_lines else "None")
        )
        lines.append(f"  Offscreen Events: {event_count}")
    else:
        lines.append("  Mood Update: No meaningful elapsed-time change")
    lines.extend(
        (
            "",
            "Emotion Analysis",
            f"  User Signal: {_debug_name(signal_label)}",
            f"  Confidence: {signal_confidence:.2f}",
            f"  Activation Threshold: {activation_threshold:.2f}",
            f"  Reaction: {reaction}",
            f"  Short Emotion: {short_emotion}",
            f"  Applied: {_debug_bool(emotion_applied)}",
        )
    )
    lines.extend(
        (
            "",
            "Offscreen Life",
            "  Elapsed Since Update: "
            f"{_debug_duration(life_metrics.get('elapsed_seconds'))}",
            "  Previous Activity: "
            f"{_debug_activity_description(previous_life)}",
            "  Current Activity: "
            f"{_debug_activity_description(current_life)}",
            "  Completed Activities: "
            f"{int(life_metrics.get('completed_count') or 0)}",
            "  Recent Reaction: "
            f"{_debug_text(life_metrics.get('recent_reaction'), 80) or 'None'}",
            "  Mood Effect: "
            f"{', '.join(life_mood_effects) if life_mood_effects else 'None'}",
        )
    )
    if life_need_changes:
        lines.append("  Need Changes:")
        lines.extend(f"    - {change}" for change in life_need_changes)
    else:
        lines.append("  Life Changes: None")
    lines.extend(
        (
            "",
            "Grounding",
            "  Current Activity: "
            + (
                activity_description or "Grounded activity available"
                if activity_present and not activity_is_recent
                else "None"
            ),
            "  Recent Activity: "
            + (
                activity_description or "Recorded activity available"
                if activity_present and activity_is_recent
                else "None"
            ),
        )
    )
    if activity_present:
        lines.extend(
            (
                f"  Activity Source: {_debug_name(activity_source)}",
                "  Activity Age: "
                f"{_debug_duration(metrics.get('grounded_activity_age_seconds'))}",
            )
        )
    lines.extend(
        (
            f"  Current Thought: {thought}",
            f"  Follow-up: {follow_up}",
            "",
            "Prompt",
            "  Included:",
        )
    )
    lines.extend(f"    - {source}" for source in included or ("None",))
    lines.extend(
        (
            "",
            f"  History Messages: {history_messages}",
            "  Final Message Count: "
            f"{_debug_number(prompt_debug.get('message_count'))}",
        )
    )
    trimmed_names = _debug_trimmed_sources(trimmed)
    if trimmed_names:
        lines.append("  Trimmed:")
        lines.extend(f"    - {source}" for source in trimmed_names)
    else:
        lines.append("  Trimmed: Nothing")
    lines.extend(
        (
            "",
            "Context Usage",
            f"  Prompt Tokens: {_debug_number(prompt_tokens)}",
            f"  Reserved Response Tokens: {_debug_number(reserved_tokens)}",
            f"  Context Window: {_debug_number(context_window)}",
            "  Total Reserved: "
            + (
                f"{total_reserved} / {context_window}"
                if total_reserved is not None and context_window is not None
                else "None"
            ),
            f"  Available Context: {_debug_number(available)}",
            f"  Token Count: {count_label}",
        )
    )
    estimate = int(metrics.get("estimated_prompt_tokens") or 0)
    if _debug_material_estimate_difference(estimate, prompt_tokens):
        lines.extend(
            (
                f"  Preflight Estimate: {estimate}",
                f"  Estimate Difference: {abs(estimate - int(prompt_tokens or 0))}",
            )
        )
    if verbose:
        lines.extend(
            _debug_internal_details(
                metrics,
                prompt_debug,
                versions,
                source_roles,
                category_tokens,
                sources,
                trimmed,
                akane,
            )
        )
    return "\n".join(lines)


def _remember_metrics(
    prepared: TurnPreparation,
    *,
    committed: bool,
    timing: InferenceTiming | None = None,
) -> None:
    turn = prepared.internal_turn
    trace = turn.affect_trace
    life_trace = life_evolution_debug(getattr(turn, "life_evolution", None))
    with _METRICS_LOCK:
        _METRICS[prepared.session_id] = {
            "estimated_prompt_tokens": prepared.prompt_plan.estimated_tokens,
            "rendered_prompt_tokens": prepared.prompt_plan.rendered_prompt_tokens,
            "backend_prompt_tokens": timing.prompt_tokens if timing else 0,
            "prompt_counting_method": prepared.prompt_plan.counting_method,
            "prompt_count_is_exact": prepared.prompt_plan.token_count_is_exact,
            "prompt_debug": prepared.prompt_plan.debug_metadata(),
            "code_context_attached": prepared.code_context_attached,
            "prior_mood": trace.prior_mood if trace else "",
            "baseline_persistent_mood": (
                trace.baseline_persistent_mood if trace else ""
            ),
            "prior_persistent_mood": trace.prior_persistent_mood if trace else "",
            "resulting_persistent_mood": (
                trace.resulting_persistent_mood if trace else ""
            ),
            "detected_signal": (
                f"{trace.detected_signal} {trace.signal_intensity:.2f}" if trace else ""
            ),
            "contextual_reaction": trace.contextual_reaction if trace else "",
            "signal_confidence": trace.signal_confidence if trace else 0.0,
            "activation_threshold": trace.activation_threshold if trace else 0.0,
            "reaction_mapping": trace.reaction_mapping if trace else "none",
            "state_changes": ", ".join(trace.state_changes) if trace else "",
            "candidate_mood_delta": trace.candidate_mood_delta if trace else "none",
            "active_emotion": (
                f"{trace.active_emotion} {trace.active_intensity:.2f}" if trace else ""
            ),
            "decay_applied": trace.decay_applied if trace else 0.0,
            "elapsed_seconds": trace.elapsed_seconds if trace else 0.0,
            "elapsed_decay_delta": trace.elapsed_decay_delta if trace else (),
            "circadian_target": trace.circadian_target if trace else 0.0,
            "circadian_delta": trace.circadian_delta if trace else (),
            "ambient_delta": trace.ambient_delta if trace else (),
            "event_delta": trace.event_delta if trace else (),
            "conversation_delta": trace.conversation_delta if trace else (),
            "short_emotion_change": (
                trace.short_emotion_change if trace else "none"
            ),
            "event_ids": trace.event_ids if trace else (),
            "reason_codes": (
                tuple(
                    dict.fromkeys(
                        (
                            *trace.reason_codes,
                            *(("state_not_committed",) if not committed else ()),
                        )
                    )
                )
                if trace
                else (("state_not_committed",) if not committed else ())
            ),
            "life_trace": life_trace,
            "final_disposition": (
                turn.current_disposition
                or (trace.final_disposition if trace else "")
            ),
            "turn_relevance_adjustment": turn.turn_relevance_adjustment,
            "grounded_activity_source": turn.grounded_activity_source,
            "grounded_activity_age_seconds": turn.grounded_activity_age_seconds,
            "current_thought_source": turn.current_thought_source,
            "follow_up_tendency": turn.follow_up_tendency,
            "dynamic_state_tokens": estimate_tokens(
                prepared.turn_context.behavioral_summary
            ),
            "state_schema_version": turn.state.version,
            "committed": committed,
            "interface": prepared.chat_input.source,
            "updated_at": time.time(),
        }
        if len(_METRICS) > _MAX_METRICS:
            oldest = min(
                _METRICS,
                key=lambda key: float(_METRICS[key].get("updated_at") or 0.0),
            )
            _METRICS.pop(oldest, None)


def _log_model_input(
    prepared: TurnPreparation,
    *,
    generation_mode: str,
) -> None:
    if not PROMPT_DEBUG:
        return
    metadata = describe_model_input(
        prepared.prompt_plan.messages,
        transport=prepared.chat_input.source,
        conversation_id=prepared.chat_input.conversation_id,
        loaded_recent_turns=len(prepared.memory_context.recent_turns),
        summary_turns=len(prepared.memory_context.earlier_turns),
        current_user_text=prepared.chat_input.text,
        generation_mode=generation_mode,
    )
    print(f"[Akane:model-input] {metadata}", flush=True)
    trace = prepared.internal_turn.affect_trace
    if trace is not None:
        print(
            "[Akane:affect] "
            f"interface={prepared.chat_input.source} prior={trace.prior_mood} "
            f"signal={trace.detected_signal}:{trace.signal_intensity:.2f} "
            f"reaction={trace.contextual_reaction} active={trace.active_emotion}:"
            f"{trace.active_intensity:.2f} decay={trace.decay_applied:.3f} "
            f"final={trace.final_disposition}",
            flush=True,
        )


def _timing_log(
    prepared: TurnPreparation,
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
        f"preprocess={prepared.preprocess_seconds:.3f}s",
        f"prompt={prepared.prompt_seconds:.3f}s",
        f"memory={prepared.memory_seconds:.3f}s",
        f"queue={max(0.0, timing.model_started_at - timing.requested_at):.3f}s",
        f"chat_template={timing.chat_template_seconds:.3f}s",
        f"tokenization={timing.prompt_tokenization_seconds:.3f}s",
        f"prompt_eval={prompt_eval_seconds:.3f}s",
        f"model_total={model_seconds:.3f}s",
        f"postprocess={postprocess_seconds:.3f}s",
        f"persistence={persistence_seconds:.3f}s",
        f"prompt_tokens_est={prepared.prompt_plan.estimated_tokens}",
        f"prompt_tokens_rendered={prepared.prompt_plan.rendered_prompt_tokens or 0}",
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


def _debug_absent(value: object) -> bool:
    return str(value or "").strip().lower() in {
        "",
        "none",
        "neutral",
        "null",
        "unknown",
        "unavailable",
    }


def _debug_name(value: object) -> str:
    text = _debug_text(value, 80)
    if _debug_absent(text):
        return "None"
    aliases = {
        "unresolved_conversation": "Unresolved conversation",
        "conversation_topic": "Active conversation topic",
        "grounded_activity": "Grounded activity",
        "relevant_memory": "Relevant memory",
        "active_emotion": "Active emotion",
        "clarification": "Ask for clarification",
        "invited": "Invited continuation",
        "unresolved": "Continue unresolved topic",
    }
    return aliases.get(text.lower(), text.replace("_", " ").capitalize())


def _debug_bool(value: object) -> str:
    return "Yes" if bool(value) else "No"


def _debug_number(value: object) -> str:
    return "None" if value is None else str(int(float(value)))


def _debug_quoted(value: object) -> str:
    text = _debug_text(value, 80)
    return f'"{text.replace(chr(34), chr(39))}"' if text else "None"


def _debug_disposition(value: object) -> str:
    text = _debug_text(value, 80)
    while text.lower().startswith("disposition:"):
        text = text.partition(":")[2].strip()
    text = text.rstrip(". ")
    return text.capitalize() if text else "None"


def _debug_scored_parts(value: object) -> tuple[str, float]:
    text = _debug_text(value, 80)
    if _debug_absent(text):
        return "", 0.0
    label, separator, raw_score = text.rpartition(" ")
    if separator:
        try:
            return label, float(raw_score)
        except ValueError:
            pass
    return text, 0.0


def _debug_scored_value(value: object) -> str:
    label, score = _debug_scored_parts(value)
    if not label or score <= 0.0:
        return "None"
    name = "Concerned" if label == "concern" else _debug_name(label)
    return f"{name} ({score:.2f})"


def _debug_reaction(contextual: object, detected: object) -> str:
    contextual_text = _debug_text(contextual, 48)
    detected_label, score = _debug_scored_parts(detected)
    label = contextual_text if not _debug_absent(contextual_text) else detected_label
    if _debug_absent(label) or score <= 0.0:
        return "None" if _debug_absent(label) else _debug_name(label)
    return f"{_debug_name(label)} ({score:.2f})"


def _debug_mood_changes(changes: object, decay: object) -> tuple[str, ...]:
    if isinstance(changes, (list, tuple)):
        items = [str(item) for item in changes]
    else:
        items = str(changes or "").split(",")
    rendered: list[str] = []
    for item in items:
        name, separator, raw_delta = item.strip().rpartition(" ")
        if not separator:
            continue
        try:
            delta = float(raw_delta)
        except ValueError:
            continue
        if abs(delta) >= 0.005:
            rendered.append(f"{_debug_name(name)}: {delta:+.2f}")
    decay_value = float(decay or 0.0)
    if decay_value >= 0.01:
        rendered.append(f"Decay Applied: {decay_value:.2f}")
    return tuple(rendered)


def _debug_duration(value: object) -> str:
    total = max(0, int(round(float(value or 0.0))))
    if total < 60:
        return f"{total} second{'s' if total != 1 else ''}"
    minutes = total // 60
    if minutes < 60:
        seconds = total % 60
        parts = [f"{minutes} minute{'s' if minutes != 1 else ''}"]
        if seconds:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
        return ", ".join(parts)
    hours, minutes = divmod(minutes, 60)
    parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return ", ".join(parts)


def _debug_delta_lines(value: object, *, minimum: float = 0.005) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    rendered: list[str] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            delta = float(item[1])
        except (TypeError, ValueError):
            continue
        if abs(delta) >= minimum:
            rendered.append(f"{_debug_name(item[0])} {delta:+.2f}")
    return tuple(rendered)


def _debug_delta_text(value: object) -> str:
    return ", ".join(_debug_delta_lines(value, minimum=0.0)) or "None"


def _debug_activity_description(value: object) -> str:
    if not isinstance(value, dict):
        return "None"
    description = _debug_text(value.get("description"), 100)
    return description[:1].upper() + description[1:] if description else "None"


def _debug_pairs_text(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return "None"
    pairs = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append(f"{_debug_name(item[0])}={float(item[1]):.3f}")
    return ", ".join(pairs) or "None"


def _debug_generated_activities(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return "None"
    activities = []
    for item in value:
        if not isinstance(item, dict):
            continue
        description = _debug_text(item.get("description"), 60)
        event_id = _debug_text(item.get("event_id"), 20)
        status = _debug_name(item.get("status"))
        if description:
            started = float(item.get("started_at") or 0.0)
            completed = float(item.get("completed_at") or 0.0)
            activities.append(
                f"{description} [{status}; {event_id or 'no ID'}; "
                f"{started:.0f}->{completed:.0f}]"
            )
    return "; ".join(activities) or "None"


def _debug_source_names(sources: tuple | list) -> tuple[str, ...]:
    names: list[str] = []
    for source in sources:
        raw = str(source or "").strip()
        name = _DEBUG_SOURCE_LABELS.get(raw, _debug_name(raw))
        if name not in names:
            names.append(name)
    return tuple(names)


def _debug_trimmed_sources(trimmed: tuple | list) -> tuple[str, ...]:
    names: list[str] = []
    for item in trimmed:
        raw = str(item or "").strip()
        source, separator, reason = raw.partition(":")
        name = _DEBUG_SOURCE_LABELS.get(source, _debug_name(source))
        if separator and reason:
            name += f" ({_debug_name(reason)})"
        if name not in names:
            names.append(name)
    return tuple(names)


def _debug_prompt_tokens(metrics: dict[str, object]) -> tuple[int | None, str]:
    backend = int(metrics.get("backend_prompt_tokens") or 0)
    rendered = metrics.get("rendered_prompt_tokens")
    estimated = int(metrics.get("estimated_prompt_tokens") or 0)
    if backend > 0:
        prompt_tokens: int | None = backend
    elif rendered is not None:
        prompt_tokens = int(rendered)
    else:
        prompt_tokens = estimated or None
    exact = bool(metrics.get("prompt_count_is_exact")) and (
        backend > 0 or rendered is not None
    )
    return prompt_tokens, "Exact" if exact else "Estimated" if prompt_tokens else "None"


def _debug_material_estimate_difference(estimate: int, actual: int | None) -> bool:
    if not estimate or actual is None or estimate == actual:
        return False
    difference = abs(estimate - actual)
    return difference > max(64, actual * 0.10)


def _debug_internal_details(
    metrics: dict[str, object],
    prompt_debug: dict,
    versions: dict,
    source_roles: dict,
    category_tokens: dict,
    sources: tuple | list,
    trimmed: tuple | list,
    akane: dict,
) -> tuple[str, ...]:
    rendered_tokens = metrics.get("rendered_prompt_tokens")
    backend_tokens = int(metrics.get("backend_prompt_tokens") or 0)
    life = metrics.get("life_trace")
    life = life if isinstance(life, dict) else {}
    previous_life = life.get("previous_activity")
    current_life = life.get("current_activity")
    return (
        "",
        "Internal Details",
        "  Personality: Stable Akane",
        "  Candidate Status: "
        + (
            "Proposed, not committed"
            if metrics.get("debug_time_candidate") or not metrics.get("committed")
            else "Committed"
        ),
        "  Mood Baseline Vector: "
        f"{_debug_text(metrics.get('baseline_persistent_mood'), 160) or 'None'}",
        f"  Prior Mood: {_debug_name(metrics.get('prior_mood'))}",
        "  Previous Persistent Vector: "
        f"{_debug_text(metrics.get('prior_persistent_mood'), 160) or 'None'}",
        "  Persistent Mood Vector: "
        f"{_debug_text(metrics.get('prior_persistent_mood'), 160) or 'None'}",
        "  Resulting Persistent Vector: "
        f"{_debug_text(metrics.get('resulting_persistent_mood'), 160) or 'None'}",
        f"  Elapsed Time: {_debug_duration(metrics.get('elapsed_seconds'))}",
        "  Elapsed Decay Delta: "
        f"{_debug_delta_text(metrics.get('elapsed_decay_delta'))}",
        "  Circadian Target: "
        f"{float(metrics.get('circadian_target') or 0.0):.3f}",
        f"  Circadian Delta: {_debug_delta_text(metrics.get('circadian_delta'))}",
        f"  Ambient Delta: {_debug_delta_text(metrics.get('ambient_delta'))}",
        f"  Event Delta: {_debug_delta_text(metrics.get('event_delta'))}",
        "  Conversation Delta: "
        f"{_debug_delta_text(metrics.get('conversation_delta'))}",
        "  Detected Signal: "
        f"{_debug_text(metrics.get('detected_signal'), 80) or 'None'}",
        f"  Signal Confidence: {float(metrics.get('signal_confidence') or 0.0):.3f}",
        "  Activation Threshold: "
        f"{float(metrics.get('activation_threshold') or 0.0):.3f}",
        "  Contextual Reaction: "
        f"{_debug_name(metrics.get('contextual_reaction'))}",
        f"  Reaction Mapping: {_debug_text(metrics.get('reaction_mapping'), 80) or 'None'}",
        "  Exact Mood Deltas: "
        f"{_debug_text(metrics.get('state_changes'), 160) or 'None'}",
        "  Candidate Mood Delta: "
        f"{_debug_name(metrics.get('candidate_mood_delta'))}",
        f"  Active Short Emotion: {_debug_scored_value(metrics.get('active_emotion'))}",
        "  Short Emotion Change: "
        f"{_debug_name(metrics.get('short_emotion_change'))}",
        f"  Decay Amount: {float(metrics.get('decay_applied') or 0.0):.3f}",
        f"  Event IDs: {_debug_items(metrics.get('event_ids')) or 'None'}",
        f"  Reason Codes: {_debug_items(metrics.get('reason_codes')) or 'None'}",
        "  Relevance Adjustment: "
        f"{float(metrics.get('turn_relevance_adjustment') or 0.0):.3f}",
        "  Dynamic State Tokens: "
        f"{int(metrics.get('dynamic_state_tokens') or 0)}",
        "  Offscreen Previous Activity: "
        f"{_debug_activity_description(previous_life)}",
        "  Offscreen Generated Activities: "
        f"{_debug_generated_activities(life.get('generated_activities'))}",
        "  Offscreen Current Activity: "
        f"{_debug_activity_description(current_life)}",
        "  Offscreen Selection Bucket: "
        f"{_debug_number(life.get('selection_bucket'))}",
        "  Offscreen Selection Candidates: "
        f"{_debug_pairs_text(life.get('selection_candidates'))}",
        "  Offscreen Repetition Penalties: "
        f"{_debug_pairs_text(life.get('repetition_penalties'))}",
        f"  Offscreen Cooldowns: {_debug_items(life.get('cooldowns')) or 'None'}",
        "  Offscreen Needs Before: "
        f"{_debug_mapping(life.get('needs_before')) or 'None'}",
        "  Offscreen Needs After: "
        f"{_debug_mapping(life.get('needs_after')) or 'None'}",
        "  Offscreen Mood Effects: "
        f"{_debug_delta_text(life.get('mood_effects'))}",
        "  Offscreen Preference Effects: "
        f"{_debug_delta_text(life.get('preference_changes'))}",
        "  Offscreen No-Activity Reason: "
        f"{_debug_name(life.get('reason_no_activity'))}",
        f"  Offscreen Authority: {_debug_name(life.get('authority_source'))}",
        "  Offscreen Last Processed: "
        f"{float(life.get('last_processed_at') or 0.0):.3f}",
        "  Offscreen Next Opportunity: "
        f"{float(life.get('next_opportunity_at') or 0.0):.3f}",
        "  State Schema Version: "
        f"{int(metrics.get('state_schema_version') or akane.get('state_schema_version') or 0)}",
        "  Prompt Builder Version: "
        f"{_debug_text(prompt_debug.get('prompt_builder_version'), 16) or 'None'}",
        f"  Identity Hash: {_debug_text(versions.get('identity'), 16) or 'None'}",
        f"  Soul Hash: {_debug_text(versions.get('soul'), 16) or 'None'}",
        "  Hard Rules Hash: "
        f"{_debug_text(versions.get('hard_constraints'), 16) or 'None'}",
        f"  Raw Prompt Sources: {_debug_items(sources) or 'None'}",
        f"  Raw Source Role Mappings: {_debug_mapping(source_roles) or 'None'}",
        f"  Category Token Counts: {_debug_mapping(category_tokens) or 'None'}",
        "  Preflight Token Estimate: "
        f"{int(metrics.get('estimated_prompt_tokens') or 0)}",
        "  Rendered Token Count: "
        f"{rendered_tokens if rendered_tokens is not None else 'None'}",
        "  Backend Observed Token Count: "
        f"{backend_tokens if backend_tokens else 'None'}",
        "  Formatter: "
        f"{_debug_text(metrics.get('prompt_counting_method'), 80) or 'None'}",
        f"  Raw Trimmed Sources: {_debug_items(trimmed) or 'None'}",
        "  Code Context Attached: "
        f"{_debug_bool(metrics.get('code_context_attached'))}",
        f"  Commit Status: {_debug_bool(metrics.get('committed'))}",
    )


def _debug_items(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    return ", ".join(_debug_text(item, 64) for item in value if str(item or "").strip())


def _debug_mapping(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    return ", ".join(
        f"{_debug_text(key, 32)}={_debug_text(item, 32)}"
        for key, item in value.items()
    )
