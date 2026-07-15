"""Deterministic, bounded chat-message construction independent of inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import CHAT_HISTORY_CONTEXT_TOKENS, LLAMA_CONTEXT_WINDOW, MAX_TOKENS
from app.core.memory import ChatTurn, estimate_tokens


@dataclass(frozen=True, slots=True)
class PromptContext:
    relationship: str = ""
    emotion: str = ""
    relevant_memories: str = ""
    earlier_dialogue: str = ""
    recent_turns: tuple[ChatTurn, ...] = ()
    constraints: str = ""
    date_time: str = ""
    reply_context: str = ""
    external_context: str = ""
    internal_context: str = ""


@dataclass(frozen=True, slots=True)
class PromptPlan:
    messages: tuple[dict[str, str], ...]
    section_tokens: tuple[tuple[str, int], ...]
    estimated_tokens: int

    def debug_metadata(self) -> dict[str, object]:
        return {
            "estimated_tokens": self.estimated_tokens,
            "message_count": len(self.messages),
            "sections": dict(self.section_tokens),
        }


def describe_model_input(
    messages: list[dict[str, str]],
    *,
    transport: str,
    conversation_id: str,
    loaded_recent_turns: int = 0,
    summary_turns: int = 0,
    current_user_text: str = "",
    generation_mode: str = "",
) -> dict[str, object]:
    """Return content-free metadata for the exact structured model input."""

    safe_transport = str(transport or "unknown").strip().lower() or "unknown"
    conversation_digest = hashlib.sha256(
        str(conversation_id or "").encode("utf-8")
    ).hexdigest()[:12]
    recent_messages = messages[1:-1] if len(messages) >= 2 else []
    current_content = messages[-1].get("content", "") if messages else ""
    return {
        "transport": safe_transport,
        "conversation_id": f"{safe_transport}:{conversation_digest}",
        "loaded_recent_turns": max(0, int(loaded_recent_turns)),
        "selected_recent_turns": len(recent_messages),
        "selected_user_turns": sum(item.get("role") == "user" for item in recent_messages),
        "selected_assistant_turns": sum(
            item.get("role") == "assistant" for item in recent_messages
        ),
        "summary_turns": max(0, int(summary_turns)),
        "final_message_roles": ",".join(item.get("role", "unknown") for item in messages),
        "memory_chars": sum(len(item.get("content", "")) for item in recent_messages),
        "current_user_occurrences": (
            current_content.count(current_user_text) if current_user_text else 0
        ),
        "generation_mode": str(generation_mode or "unknown"),
    }


def format_runtime_context(*parts: object) -> str:
    cleaned = [text for part in parts if (text := str(part or "").strip())]
    return _section("VERIFIED RUNTIME STATE", *cleaned)


@lru_cache(maxsize=4)
def _static_prompt_tokens(static_prompt: str) -> int:
    return estimate_tokens(static_prompt) + 4


def build_prompt_plan(user_text: str, context: PromptContext) -> PromptPlan:
    """Build bounded messages in stable priority and role order."""

    from app.core.character import get_static_system_prompt

    context_limit = max(256, LLAMA_CONTEXT_WINDOW - MAX_TOKENS - 96)
    static_prompt = get_static_system_prompt()
    static_tokens = _static_prompt_tokens(static_prompt)
    if static_tokens + 64 >= context_limit:
        raise RuntimeError(
            "Akane's canonical character prompt exceeds the available model context."
        )
    # Reserve a small margin for chat-template and section-label overhead that
    # the byte estimator cannot see precisely.
    available = max(48, context_limit - static_tokens - 40)

    # The current message is never displaced by memory. Editor and reply data
    # receive only what remains after preserving the user's own words.
    current_budget = max(96, int(available * 0.48))
    current_text = _clip_tokens(str(user_text or "").strip(), current_budget)
    used = estimate_tokens(current_text) + 4
    remaining = max(0, available - used)

    extra_parts: list[tuple[str, str]] = []
    extra_cap = min(max(0, remaining // 2), 520)
    extra_used = 0
    for name, value in (
        ("reply_context", _quoted_context("REPLY CONTEXT", context.reply_context)),
        ("external_context", _quoted_context("READ-ONLY EDITOR CONTEXT", context.external_context)),
    ):
        text = value
        if not text or extra_used >= extra_cap:
            continue
        clipped = _clip_tokens(text, extra_cap - extra_used)
        if clipped:
            extra_parts.append((name, clipped))
            extra_used += estimate_tokens(clipped) + 1
    remaining = max(0, remaining - extra_used)

    runtime_text = _clip_tokens(
        format_runtime_context(
            context.constraints,
            context.relationship,
            context.date_time,
        ),
        min(150, max(0, remaining // 3)),
    )
    runtime_used = estimate_tokens(runtime_text)
    remaining = max(0, remaining - runtime_used)

    internal_text = _clip_tokens(
        _section("CURRENT INTERNAL CONTEXT", context.internal_context),
        min(210, max(0, remaining // 3)),
    )
    internal_used = estimate_tokens(internal_text)
    remaining = max(0, remaining - internal_used)

    memory_text = _clip_tokens(
        _section("RELEVANT MEMORY", context.relevant_memories, context.earlier_dialogue),
        min(180, max(0, remaining // 3)),
    )
    memory_used = estimate_tokens(memory_text)
    remaining = max(0, remaining - memory_used)

    emotion_text = _clip_tokens(
        _section("CURRENT EMOTIONAL CONTEXT", context.emotion),
        min(90, max(0, remaining // 4)),
    ) if not internal_text else ""
    emotion_used = estimate_tokens(emotion_text)
    remaining = max(0, remaining - emotion_used)

    recent_budget = min(CHAT_HISTORY_CONTEXT_TOKENS, max(0, remaining - 8))
    recent_turns = _select_recent_turns(context.recent_turns, recent_budget)
    recent_used = sum(estimate_tokens(turn.content) + 4 for turn in recent_turns)

    system_parts = [static_prompt, runtime_text, internal_text, memory_text, emotion_text]
    system_prompt = "\n\n".join(part for part in system_parts if part)

    current_parts = [*(text for _name, text in extra_parts), current_text]
    current_message = "\n\n".join(part for part in current_parts if part)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(turn.as_message() for turn in recent_turns)
    messages.append({"role": "user", "content": current_message})

    section_tokens = [
        ("hard_rules_soul_identity", static_tokens),
        ("runtime_state", runtime_used),
        ("internal_context", internal_used),
        ("memory", memory_used),
        ("emotion", emotion_used),
        ("recent_turns", recent_used),
        *((name, estimate_tokens(text)) for name, text in extra_parts),
        ("current_message", estimate_tokens(current_text)),
    ]
    estimated = sum(estimate_tokens(message["content"]) + 4 for message in messages)
    return PromptPlan(
        messages=tuple(messages),
        section_tokens=tuple(section_tokens),
        estimated_tokens=estimated,
    )


def _section(label: str, *parts: object) -> str:
    cleaned = [text for part in parts if (text := str(part or "").strip())]
    return f"[{label}]\n" + "\n".join(cleaned) if cleaned else ""


def _select_recent_turns(turns: tuple[ChatTurn, ...], budget: int) -> tuple[ChatTurn, ...]:
    if budget <= 8:
        return ()
    if len(turns) >= 2 and turns[-1].role == "assistant" and turns[-2].role == "user":
        pair = turns[-2:]
        pair_cost = sum(estimate_tokens(turn.content) + 4 for turn in pair)
        if pair_cost > budget:
            per_turn = max(12, (budget - 8) // 2)
            return tuple(
                ChatTurn(
                    turn.turn_id,
                    turn.role,
                    _clip_tokens(turn.content, per_turn),
                    turn.timestamp,
                    turn.source,
                )
                for turn in pair
            )
    selected: list[ChatTurn] = []
    used = 0
    for turn in reversed(turns):
        cost = estimate_tokens(turn.content) + 4
        if selected and used + cost > budget:
            break
        if cost > budget:
            clipped = _clip_tokens(turn.content, max(16, budget - 4))
            if clipped and not selected:
                selected.append(
                    ChatTurn(turn.turn_id, turn.role, clipped, turn.timestamp, turn.source)
                )
            break
        selected.append(turn)
        used += cost
    selected.reverse()
    if selected and selected[0].role == "assistant":
        selected.pop(0)
    return tuple(selected)


def _quoted_context(label: str, value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f"[{label}]\n{text}"


def _clip_tokens(value: str, budget: int) -> str:
    text = str(value or "").strip()
    if budget <= 0 or not text:
        return ""
    if estimate_tokens(text) <= budget:
        return text
    marker = "\n...[context trimmed]...\n"
    usable = max(1, budget * 3 - len(marker.encode("utf-8")))
    while True:
        head = max(1, int(usable * 0.68))
        tail = max(0, usable - head)
        clipped = text[:head].rstrip() + marker + (text[-tail:].lstrip() if tail else "")
        estimate = estimate_tokens(clipped)
        if estimate <= budget or usable <= 1:
            return clipped
        usable = max(1, int(usable * budget / estimate) - 1)
