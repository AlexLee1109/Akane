"""Deterministic, bounded chat-message construction independent of inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import CHAT_HISTORY_CONTEXT_TOKENS, LLAMA_CONTEXT_WINDOW, MAX_TOKENS
from app.core.memory import ChatTurn, estimate_tokens
from app.core.signal import SubtextAppraisal


@dataclass(frozen=True, slots=True)
class TurnGuidance:
    """Structured per-turn behavior selected before prompt rendering."""

    autonomous: bool = False
    identity_attribute: str = ""
    current_activity: bool = False
    group_conversation: bool = False
    criticism: bool = False
    correction_requested: bool = False
    repetition_level: int = 0


@dataclass(frozen=True, slots=True)
class PromptContext:
    relationship: str = ""
    preference_continuity: str = ""
    relevant_memories: str = ""
    earlier_dialogue: str = ""
    recent_turns: tuple[ChatTurn, ...] = ()
    subtext: SubtextAppraisal | None = None
    constraints: str = ""
    response_pressure: str = ""
    date_time: str = ""
    reply_context: str = ""
    external_context: str = ""
    internal_context: str = ""
    life_context: str = ""
    turn_guidance: TurnGuidance | None = None


@dataclass(frozen=True, slots=True)
class PromptPlan:
    messages: list[dict[str, str]]
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


@lru_cache(maxsize=4)
def _static_prompt_tokens(static_prompt: str) -> int:
    return estimate_tokens(static_prompt) + 4


def build_prompt_plan(user_text: str, context: PromptContext) -> PromptPlan:
    """Build bounded messages in stable priority and role order."""

    from app.core.character import get_hard_constraints_prompt, get_static_system_prompt

    context_limit = max(256, LLAMA_CONTEXT_WINDOW - MAX_TOKENS - 96)
    static_prompt = get_static_system_prompt()
    static_tokens = _static_prompt_tokens(static_prompt)
    guidance_text = _turn_guidance_text(context.turn_guidance)
    hard_text = get_hard_constraints_prompt(
        " ".join(
            text
            for value in (context.constraints, guidance_text)
            if (text := str(value or "").strip())
        )
    )
    hard_tokens = estimate_tokens(hard_text)
    current_text = str(user_text or "")
    current_tokens = estimate_tokens(current_text) + 4
    fixed_tokens = static_tokens + hard_tokens + current_tokens
    if fixed_tokens >= context_limit:
        raise RuntimeError(
            "Akane's stable character prompt and current message exceed the available model context."
        )
    remaining = max(0, context_limit - fixed_tokens)
    recent_reserve = min(CHAT_HISTORY_CONTEXT_TOKENS, remaining // 3)
    dynamic_remaining = max(0, remaining - recent_reserve)

    continuity_text = _clip_tokens(
        _section(
            "4. RELEVANT LONG-TERM CONTINUITY",
            context.relationship,
            context.preference_continuity,
            context.relevant_memories,
            context.earlier_dialogue,
        ),
        min(320, dynamic_remaining),
    )
    continuity_used = estimate_tokens(continuity_text)
    dynamic_remaining = max(0, dynamic_remaining - continuity_used)

    internal_text = _clip_tokens(
        _section(
            "5. CURRENT INTERNAL STATE",
            context.internal_context,
            context.life_context,
            context.date_time,
        ),
        min(240, dynamic_remaining),
    )
    internal_used = estimate_tokens(internal_text)
    dynamic_remaining = max(0, dynamic_remaining - internal_used)

    pressure_text = _clip_tokens(
        _section(
            "6. CURRENT SUBTEXT AND RESPONSE PRESSURE",
            context.response_pressure,
            _subtext_text(context.subtext),
            _labeled_context("Referenced message", context.reply_context),
            _labeled_context("Read-only editor context", context.external_context),
        ),
        min(600, dynamic_remaining),
    )
    pressure_used = estimate_tokens(pressure_text)

    dynamic_used = continuity_used + internal_used + pressure_used
    recent_budget = min(
        CHAT_HISTORY_CONTEXT_TOKENS,
        max(0, remaining - dynamic_used),
    )
    recent_turns = _select_recent_turns(context.recent_turns, recent_budget)
    recent_used = sum(estimate_tokens(turn.content) + 4 for turn in recent_turns)

    system_parts = [
        static_prompt,
        hard_text,
        continuity_text,
        internal_text,
        pressure_text,
    ]
    system_prompt = "\n\n".join(part for part in system_parts if part)

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(turn.as_message() for turn in recent_turns)
    messages.append({"role": "user", "content": current_text})

    section_tokens = [
        ("stable_identity_and_personality", static_tokens),
        ("hard_constraints", hard_tokens),
        ("long_term_continuity", continuity_used),
        ("internal_state", internal_used),
        ("subtext_response_pressure", pressure_used),
        ("recent_turns", recent_used),
        ("current_message", current_tokens - 4),
    ]
    estimated = sum(estimate_tokens(message["content"]) + 4 for message in messages)
    if estimated > context_limit:
        raise RuntimeError("The constructed prompt exceeds the available model context.")
    return PromptPlan(
        messages=messages,
        section_tokens=tuple(section_tokens),
        estimated_tokens=estimated,
    )


def _section(label: str, *parts: object) -> str:
    cleaned = [text for part in parts if (text := str(part or "").strip())]
    return f"[{label}]\n" + "\n".join(cleaned) if cleaned else ""


def _subtext_text(appraisal: SubtextAppraisal | None) -> str:
    if appraisal is None:
        return ""
    return "\n".join(
        (
            f"Possible interpretation, not a fact: {appraisal.summary}",
            f"Response adjustment: {appraisal.behavioral_effect}",
            "Use this silently; do not announce, explain, or attribute the interpretation to the user.",
        )
    )


def _turn_guidance_text(guidance: TurnGuidance | None) -> str:
    if guidance is None:
        return ""
    if guidance.autonomous:
        return (
            "Unprompted thought: use the supplied internal and life state plus Akane-owned "
            "memory; do not address anyone, invent an activity, expose user context, or "
            "mention implementation."
        )

    parts: list[str] = []
    if guidance.identity_attribute == "preferences":
        parts.append(
            "Preference answer: casual, direct, and concise; name one or two concrete "
            "interests with a simple reason. For anime, name titles first. Do not claim "
            "anything Akane watched, played, did, or pursued recently or offscreen."
        )
    elif guidance.identity_attribute:
        parts.append(
            f"Identity question ({guidance.identity_attribute}): use only canonical identity facts."
        )
    if guidance.current_activity:
        parts.append(
            "Current activity: use the supplied life state; do not substitute an activity "
            "or mention implementation."
        )
    if guidance.group_conversation:
        parts.append(
            "Group chat: address this user; do not treat channel context as personal memory."
        )
    if guidance.criticism:
        parts.append(
            "Correct the answer directly; do not analyze the criticism, do not defend the "
            "prior answer, redirect, or promise future improvement."
        )
    elif guidance.correction_requested:
        parts.append(
            "Apply the correction in the answer without defending the prior framing."
        )
    if guidance.repetition_level == 1:
        parts.append(
            "Repeated request: answer fully with different wording; preserve the answer "
            "and do not mention repetition."
        )
    elif guidance.repetition_level == 2:
        parts.append(
            "Repeated request: preserve the answer while varying wording, reason, or emphasis; "
            "acknowledge the pattern only if natural."
        )
    elif guidance.repetition_level >= 3:
        parts.append(
            "Repeated request: still answer fully and preserve facts and preferences; mild "
            "exasperation is allowed, never hostility or refusal."
        )
    return " ".join(parts)


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


def _labeled_context(label: str, value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f"{label}:\n{text}"


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
