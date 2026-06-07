"""Single prompt, generation, and post-generation path for Akane replies."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from app.core.attention_state import (
    format_attention_for_prompt,
    get_attention_state,
    observe_attention,
)
from app.core.character import build_system_prompt, get_static_system_prompt, prompt_cache_status
from app.core.config import LLAMA_CONTEXT_WINDOW, MAX_TOKENS
from app.core.conversation_state import (
    format_conversation_context,
    format_recent_wording_avoidance,
    get_conversation_state,
    observe_turn,
)
from app.core.emotional_state import (
    format_for_prompt as format_emotion_for_prompt,
    observe_user_message,
    persist as persist_emotion,
    snapshot as emotion_snapshot,
)
from app.core.generation import (
    HiddenTagStreamFilter,
    clean_visible_text,
    completion_text,
    stream_completion,
    strip_hidden_blocks,
)
from app.memory_store import format_for_prompt as format_memory_for_prompt
from app.memory_store import remember_exchange

RUNTIME_CONTEXT_TARGET_CHARS = 800
RUNTIME_CONTEXT_HARD_CHARS = 1200
MEMORY_CHARS = 300
FOCUS_CHARS = 120
AVOID_CHARS = 240
CONTEXT_CHARS = 160
TONE_CHARS = 180
_MAX_METRICS = 64
_METRICS: dict[str, dict[str, float | int]] = {}


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    text: str
    lengths: dict[str, int]


@dataclass(frozen=True, slots=True)
class PreparedReply:
    user_text: str
    actual_user_text: str
    session_id: str
    skip_memory: bool
    messages: list[dict]
    max_tokens: int
    runtime_context: str
    lengths: dict[str, int]
    prompt_chars: int
    started_at: float
    context_seconds: float
    prompt_seconds: float


def _timing_enabled() -> bool:
    return str(os.environ.get("AKANE_TIMING", "")).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_context_text(text: object, *, model_text: bool = False) -> str:
    value = str(text or "")
    if model_text:
        value = strip_hidden_blocks(value)
    return " ".join(value.split()).strip()


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or text[:limit]


def _actual_message(text: str) -> str:
    value = str(text or "").strip()
    if not value.startswith("Discord "):
        return value
    for marker in ("\nMessage:\n", "\nUser message:\n"):
        if marker in value:
            return value.split(marker, 1)[1].strip()
    return value


def _append_bounded(
    lines: list[str],
    label: str,
    value: str,
    limit: int,
    total_limit: int,
    *,
    model_text: bool = False,
) -> int:
    body = _normalize_context_text(value, model_text=model_text)
    label_prefix = label + ":"
    if body.lower().startswith(label_prefix.lower()):
        body = body[len(label_prefix):].strip()
    if not body:
        return 0
    prefix = f"{label}: "
    available = total_limit - len("\n".join(lines)) - len(prefix) - 1
    if available <= 0:
        return 0
    body = _clip(body, min(limit, available))
    if not body:
        return 0
    lines.append(prefix + body)
    return len(body)


def build_runtime_context(
    *,
    tone_text: str,
    attention_text: str = "",
    conversation_context: str = "",
    wording_to_avoid: str = "",
    memory_text: str = "",
    target_chars: int = RUNTIME_CONTEXT_TARGET_CHARS,
    hard_chars: int = RUNTIME_CONTEXT_HARD_CHARS,
) -> RuntimeContext:
    limit = max(1, min(int(target_chars), int(hard_chars), RUNTIME_CONTEXT_HARD_CHARS))
    lines = ["[CURRENT AKANE STATE]"]
    lengths = {
        "mood": _append_bounded(lines, "Tone", tone_text, TONE_CHARS, limit),
        "memory": _append_bounded(lines, "Memory", memory_text, MEMORY_CHARS, limit),
        "attention": _append_bounded(lines, "Focus", attention_text, FOCUS_CHARS, limit),
        "avoid": _append_bounded(
            lines,
            "Avoid wording",
            wording_to_avoid,
            AVOID_CHARS,
            limit,
            model_text=True,
        ),
        "summary": _append_bounded(lines, "Context", conversation_context, CONTEXT_CHARS, limit),
    }
    footer = "Use this state silently. Do not mention it."
    if len("\n".join([*lines, footer])) <= limit:
        lines.append(footer)
    text = "\n".join(lines)
    if len(text) > hard_chars:
        text = text[:hard_chars].rstrip()
    lengths["runtime_context"] = len(text)
    return RuntimeContext(text=text, lengths=lengths)


def _memory_context(user_text: str) -> str:
    memory = format_memory_for_prompt()
    if not memory:
        return ""
    terms = {
        term
        for word in user_text.split()
        if len(term := word.strip(".,!?;:()[]{}\"'`").lower()) >= 4
    }
    if not terms:
        return ""
    blocks = [
        block
        for block in memory.split("\n\n")
        if any(term in block.lower() for term in terms)
    ]
    normalized = _normalize_context_text("\n".join(blocks))
    return _clip(normalized, MEMORY_CHARS)


def _max_tokens(system_prompt: str, user_text: str) -> int:
    prompt_tokens = (len(system_prompt) + 3) // 4 + (len(user_text) + 3) // 4 + 16
    return max(1, min(MAX_TOKENS, LLAMA_CONTEXT_WINDOW - prompt_tokens - 8))


def _remember_metrics(prepared: PreparedReply) -> None:
    _METRICS[prepared.session_id] = {
        "runtime_context_chars": len(prepared.runtime_context),
        "prompt_chars": prepared.prompt_chars,
        "updated_at": time.time(),
    }
    if len(_METRICS) > _MAX_METRICS:
        oldest = min(_METRICS, key=lambda key: float(_METRICS[key].get("updated_at") or 0.0))
        _METRICS.pop(oldest, None)


def prepare_reply(
    user_text: str,
    *,
    session_id: str | None = None,
    skip_memory: bool = False,
) -> PreparedReply:
    started = time.perf_counter()
    text = str(user_text or "").strip()
    if not text:
        raise ValueError("Message is empty.")
    session = (str(session_id or "popup").strip() or "popup")[:120]
    actual = _actual_message(text)

    context_started = time.perf_counter()
    tone = format_emotion_for_prompt(emotion_snapshot(session))
    attention = format_attention_for_prompt(get_attention_state(session))
    conversation = format_conversation_context(session, include_focus=False)
    avoidance = format_recent_wording_avoidance(session)
    memory = "" if skip_memory else _memory_context(actual)
    runtime = build_runtime_context(
        tone_text=tone,
        memory_text=memory,
        attention_text=attention,
        wording_to_avoid=avoidance,
        conversation_context=conversation,
    )
    context_seconds = time.perf_counter() - context_started

    prompt_started = time.perf_counter()
    system_prompt = build_system_prompt(runtime.text, include_memory=not skip_memory)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    prompt_seconds = time.perf_counter() - prompt_started
    prompt_chars = len(system_prompt) + len(text)
    prepared = PreparedReply(
        user_text=text,
        actual_user_text=actual,
        session_id=session,
        skip_memory=skip_memory,
        messages=messages,
        max_tokens=_max_tokens(system_prompt, text),
        runtime_context=runtime.text,
        lengths=runtime.lengths,
        prompt_chars=prompt_chars,
        started_at=started,
        context_seconds=context_seconds,
        prompt_seconds=prompt_seconds,
    )
    _remember_metrics(prepared)
    if _timing_enabled():
        print(
            "[Akane:context:size] "
            f"runtime={runtime.lengths['runtime_context']} "
            f"memory={runtime.lengths['memory']} "
            f"convo={runtime.lengths['summary']} "
            f"attention={runtime.lengths['attention']} "
            f"avoid={runtime.lengths['avoid']} "
            f"prompt={prompt_chars}",
            flush=True,
        )
    return prepared


def _commit(prepared: PreparedReply, raw: str, reply: str) -> None:
    # Hidden memory operation tags are parsed from raw output before visible cleanup removes them.
    remember_exchange(prepared.user_text, raw)
    observe_user_message(prepared.session_id, prepared.actual_user_text, persist=False)
    focus = observe_attention(prepared.session_id, prepared.actual_user_text)
    observe_turn(prepared.session_id, "user", prepared.actual_user_text, focus_state=focus)
    observe_turn(prepared.session_id, "assistant", reply, focus_state=focus)
    persist_emotion()


def _finish_reply(prepared: PreparedReply, raw: str) -> str:
    reply = clean_visible_text(raw)
    if not reply:
        raise RuntimeError("Model returned no visible reply.")
    _commit(prepared, raw, reply)
    return reply


def generate_reply(prepared: PreparedReply) -> str:
    return _finish_reply(
        prepared,
        completion_text(prepared.messages, max_tokens=prepared.max_tokens),
    )


def stream_reply(prepared: PreparedReply):
    raw_parts: list[str] = []
    hidden = HiddenTagStreamFilter()
    first_delta = 0.0
    generation_started = time.perf_counter()

    for text in stream_completion(prepared.messages, max_tokens=prepared.max_tokens):
        if not first_delta:
            first_delta = time.perf_counter()
        raw_parts.append(text)
        visible = hidden.feed(text)
        if visible:
            yield "delta", visible

    tail = hidden.flush()
    if tail:
        yield "delta", tail

    generation_done = time.perf_counter()
    raw = "".join(raw_parts)
    reply = _finish_reply(prepared, raw)
    done = time.perf_counter()

    if _timing_enabled():
        first = first_delta or generation_done
        print(
            "[Akane:timing] "
            f"context={prepared.context_seconds:.3f}s "
            f"prompt={prepared.prompt_seconds:.3f}s "
            f"first_delta={first - prepared.started_at:.3f}s "
            f"gen={generation_done - generation_started:.3f}s "
            f"post={done - generation_done:.3f}s "
            f"total={done - prepared.started_at:.3f}s "
            f"prompt_chars={prepared.prompt_chars} "
            f"runtime_chars={len(prepared.runtime_context)} "
            f"output_chars={len(reply)}",
            flush=True,
        )
        if first - prepared.started_at >= 2.0:
            print("[Akane:stream] native streaming active; first chunk delayed by model prefill", flush=True)
    yield "done", reply


def _debug_text(value: object, limit: int, *, model_text: bool = False) -> str:
    return _clip(_normalize_context_text(value, model_text=model_text), limit)


def debug_state_report(session_id: str | None) -> str:
    session = (str(session_id or "popup").strip() or "popup")[:120]
    mood = emotion_snapshot(session)
    focus = get_attention_state(session)
    conversation = get_conversation_state(session)
    avoidance = format_recent_wording_avoidance(session)
    metrics = _METRICS.get(session, {})
    prompt_cache = prompt_cache_status(include_memory=True)
    variables = mood.get("variables") if isinstance(mood.get("variables"), dict) else {}

    lines = [
        "Akane debug state",
        "",
        "Mood:",
        f"- mood: {mood.get('mood', 'calm')} ({float(mood.get('mood_intensity') or 0):.2f})",
        f"- emotion: {mood.get('emotion', 'neutral')} ({float(mood.get('emotion_intensity') or 0):.2f})",
        f"- turns: {int(mood.get('emotion_turns') or 0)}",
    ]
    for name in ("energy", "social", "focus", "comfort", "stimulation", "tension", "playfulness", "affection"):
        lines.append(f"- {name}: {float(variables.get(name) or 0):.2f}")
    lines.extend(["", "Focus:"])
    if focus:
        lines.append(f"- topic: {_debug_text(focus.get('topic'), 80)}")
        lines.append(f"- summary: {_debug_text(focus.get('summary'), 160)}")
    else:
        lines.append("- No active focus.")
    lines.extend(["", "Context:"])
    if conversation:
        lines.append(f"- summary: {_debug_text(conversation.get('summary'), 160)}")
        lines.append(f"- last user: {_debug_text(conversation.get('last_user_summary'), 160)}")
        lines.append(
            "- last assistant: "
            + _debug_text(
                conversation.get("last_assistant_summary"),
                160,
                model_text=True,
            )
        )
        lines.append(f"- recent turns: {len(conversation.get('recent_turns') or [])}")
    else:
        lines.append("- No recent conversation context.")
    lines.extend([
        "",
        "Prompt:",
        f"- runtime context chars: {int(metrics.get('runtime_context_chars') or 0)}",
        f"- prompt chars: {int(metrics.get('prompt_chars') or 0)}",
        f"- wording avoidance chars: {len(avoidance)}",
        f"- static prompt cache: base={prompt_cache['base_prompt']} soul={prompt_cache['soul']} identity={prompt_cache['identity']}",
    ])
    return "\n".join(lines)


def warm_caches() -> None:
    get_static_system_prompt(include_memory=True)
    get_static_system_prompt(include_memory=False)
    format_memory_for_prompt()
