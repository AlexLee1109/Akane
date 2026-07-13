"""Deterministic, budgeted chat-message construction independent of inference."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import CHAT_HISTORY_CONTEXT_TOKENS, PROMPT_TOKEN_BUDGET
from app.core.memory import ChatTurn, estimate_tokens


@dataclass(frozen=True, slots=True)
class PromptContext:
    relationship: str = ""
    emotion: str = ""
    user_profile: str = ""
    rolling_summary: str = ""
    recent_turns: tuple[ChatTurn, ...] = ()
    constraints: str = ""
    date_time: str = ""
    reply_context: str = ""
    external_context: str = ""


@dataclass(frozen=True, slots=True)
class PromptPlan:
    messages: tuple[dict[str, str], ...]
    section_tokens: tuple[tuple[str, int], ...]
    estimated_tokens: int

    def debug_metadata(self) -> dict[str, object]:
        return {
            "estimated_tokens": self.estimated_tokens,
            "budget_tokens": PROMPT_TOKEN_BUDGET,
            "message_count": len(self.messages),
            "sections": dict(self.section_tokens),
        }


def format_runtime_context(*parts: object) -> str:
    cleaned = [str(part or "").strip() for part in parts if str(part or "").strip()]
    return "[LIVE CONVERSATION CONTEXT]\n" + "\n".join(cleaned) if cleaned else ""


def build_prompt_plan(user_text: str, context: PromptContext) -> PromptPlan:
    """Build bounded messages in stable priority and role order."""

    from app.core.character import get_static_system_prompt

    static_prompt = _clip_tokens(
        get_static_system_prompt(),
        max(256, int(PROMPT_TOKEN_BUDGET * 0.78)),
    )
    static_tokens = estimate_tokens(static_prompt) + 4
    # Reserve a small margin for chat-template and section-label overhead that
    # the byte estimator cannot see precisely.
    available = max(48, PROMPT_TOKEN_BUDGET - static_tokens - 24)

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
        text = str(value or "").strip()
        if not text or extra_used >= extra_cap:
            continue
        clipped = _clip_tokens(text, extra_cap - extra_used)
        if clipped:
            extra_parts.append((name, clipped))
            extra_used += estimate_tokens(clipped) + 1
    remaining = max(0, remaining - extra_used)

    live_candidates = (
        ("constraints", context.constraints),
        ("relationship", context.relationship),
        ("emotion", context.emotion),
        ("date_time", context.date_time),
    )
    live_parts: list[tuple[str, str]] = []
    live_cap = min(160, max(0, remaining // 3))
    live_used = 0
    for name, value in live_candidates:
        text = str(value or "").strip()
        if not text or live_used >= live_cap:
            continue
        clipped = _clip_tokens(text, live_cap - live_used)
        if clipped:
            live_parts.append((name, clipped))
            live_used += estimate_tokens(clipped) + 1
    remaining = max(0, remaining - live_used)

    recent_budget = min(CHAT_HISTORY_CONTEXT_TOKENS, max(0, int(remaining * 0.68)))
    recent_turns = _select_recent_turns(context.recent_turns, recent_budget)
    recent_used = sum(estimate_tokens(turn.content) + 4 for turn in recent_turns)
    remaining = max(0, remaining - recent_used)

    memory_parts: list[tuple[str, str]] = []
    for name, value in (
        ("user_profile", context.user_profile),
        ("rolling_summary", context.rolling_summary),
    ):
        text = str(value or "").strip()
        if not text or remaining <= 0:
            continue
        clipped = _clip_tokens(text, remaining)
        if clipped:
            memory_parts.append((name, clipped))
            remaining -= estimate_tokens(clipped) + 1

    dynamic_parts = [*live_parts, *memory_parts]
    dynamic_text = format_runtime_context(*(text for _name, text in dynamic_parts))
    system_prompt = f"{static_prompt}\n\n{dynamic_text}" if dynamic_text else static_prompt

    current_parts = [*(text for _name, text in extra_parts), current_text]
    current_message = "\n\n".join(part for part in current_parts if part)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(turn.as_message() for turn in recent_turns)
    messages.append({"role": "user", "content": current_message})

    section_tokens = [
        ("hard_rules_soul_identity", static_tokens),
        *((name, estimate_tokens(text)) for name, text in dynamic_parts),
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
