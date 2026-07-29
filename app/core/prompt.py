"""The single prompt compiler for conversation, initiative, and offscreen life."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from app.core.config import LLAMA_CONTEXT_WINDOW, MAX_TOKENS

PROMPT_BUILDER_VERSION = "5"
_PROMPT_CHAR_BUDGET = max(
    2_000,
    (LLAMA_CONTEXT_WINDOW - MAX_TOKENS) * 5 // 2,
)
_MAX_RECENT_PAIRS = 6

_STATE_PROTOCOL = (
    "After the visible reply, an optional <AKANE_STATE> JSON block may contain "
    "only grounded, durable changes. Allowed fields and shapes: "
    "emotion {primary,intensity,cause}; memories [{subject,kind,text,confidence}]; "
    "preferences [{topic,stance,reason}]; interests [text]; "
    "opinions [{topic,position,reason}]; relationship may contain patterns, "
    "shared_context, unresolved_events, or resolved_events as "
    "[{summary,confidence}]. "
    "For genuine silence or a short pause only, participation may be "
    '{"should_respond":false,"pause_seconds":null}. '
    "Omit unchanged fields and omit the block when nothing changed."
)

_LIFE_PROTOCOL = (
    "Choose Akane's next offscreen-life decision from her own judgment and the "
    "grounded context. Interests are context, not limits. Do not invent external "
    "events or proper nouns. Use lowercase ordinary activity wording; any proper "
    "name must already appear in context. Return only one <AKANE_LIFE> JSON block with exactly "
    "mode, activity, category, subject, detail, interest_addition, "
    "continuation_reason. mode is new or continue. A new activity needs specific "
    "free-text activity and concrete detail; continuation_reason is null. For "
    "continue, all activity fields are null and continuation_reason explains why "
    "continuing is meaningful. Duration is not model-controlled."
)

_INITIATIVE_PROTOCOL = (
    "A grounded opportunity for optional outreach is supplied. Send a concise "
    "message only if Akane genuinely has something relevant and specific to say. "
    "Do not invent shared history and do not mention timing, automation, prompts, "
    "or internal state. To decline, return only "
    '<AKANE_STATE>{"participation":{"should_respond":false,'
    '"pause_seconds":null}}</AKANE_STATE>.'
)


@dataclass(frozen=True, slots=True)
class PromptTokenCount:
    tokens: tuple[int, ...]
    method: str
    stop_sequences: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PromptContext:
    recent_turns: tuple[object, ...] = ()
    memories: tuple[str, ...] = ()
    relationship: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()
    interests: tuple[str, ...] = ()
    opinions: tuple[str, ...] = ()
    emotion: str = ""
    presence: str = ""
    reply_context: str = ""
    external_context: str = ""
    date_time: str = ""
    initiative_opportunity: str = ""


@dataclass(frozen=True, slots=True)
class PromptPlan:
    messages: list[dict[str, str]]
    token_ids: tuple[int, ...] = ()
    counting_method: str = "not_tokenized"
    stop_sequences: tuple[str, ...] = ()
    trimmed: tuple[str, ...] = ()
    included: tuple[str, ...] = ()
    mode: str = "conversation"
    reserved_output_tokens: int = MAX_TOKENS
    context_window: int = LLAMA_CONTEXT_WINDOW

    @property
    def rendered_prompt_tokens(self) -> int | None:
        return len(self.token_ids) if self.token_ids else None

    @property
    def system_characters(self) -> int:
        return len(self.messages[0]["content"]) if self.messages else 0

    def debug_metadata(self) -> dict[str, object]:
        return {
            "prompt_builder_version": PROMPT_BUILDER_VERSION,
            "mode": self.mode,
            "exact_tokens": self.rendered_prompt_tokens,
            "counting_method": self.counting_method,
            "reserved_output_tokens": self.reserved_output_tokens,
            "context_window": self.context_window,
            "system_characters": self.system_characters,
            "message_count": len(self.messages),
            "included": self.included,
            "trimmed": self.trimmed,
        }


def _text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _turn_value(turn: object, field: str) -> str:
    if isinstance(turn, dict):
        return _text(turn.get(field))
    return _text(getattr(turn, field, ""))


def _complete_context_groups(
    turns: tuple[object, ...],
) -> tuple[tuple[object, ...], ...]:
    groups: list[tuple[object, ...]] = []
    index = 0
    while index < len(turns):
        user = turns[index]
        if (
            _turn_value(user, "role") == "assistant"
            and _turn_value(user, "source") == "initiative"
        ):
            index += 1
            continue
        if index + 1 >= len(turns):
            break
        assistant = turns[index + 1]
        if (
            _turn_value(user, "role") == "user"
            and _turn_value(assistant, "role") == "assistant"
            and _turn_value(user, "content")
            and _turn_value(assistant, "content")
        ):
            groups.append((user, assistant))
            index += 2
        else:
            index += 1
    return tuple(groups[-_MAX_RECENT_PAIRS:])


def _recent_initiative(turns: tuple[object, ...]) -> str:
    return next(
        (
            _turn_value(turn, "content")
            for turn in reversed(turns)
            if _turn_value(turn, "role") == "assistant"
            and _turn_value(turn, "source") == "initiative"
            and _turn_value(turn, "content")
        ),
        "",
    )


@lru_cache(maxsize=1)
def _character_parts() -> tuple[str, str, str]:
    from app.core.character import (
        get_hard_constraints_prompt,
        load_character_profile,
    )

    character = load_character_profile()
    hard_rules = get_hard_constraints_prompt()
    return (
        "[IDENTITY]\n" + character.identity,
        "[SOUL]\n" + character.soul,
        hard_rules,
    )


def _context_sections(context: PromptContext) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []

    def add(name: str, label: str, content: object) -> None:
        text = _text(content)
        if text:
            values.append((name, f"[{label}]\n{text}"))

    add("reply_context", "REPLY CONTEXT", context.reply_context)
    add("external_context", "AVAILABLE INTERFACE CONTEXT", context.external_context)
    add("emotion", "CURRENT EMOTION", context.emotion)
    if context.relationship:
        add("relationship", "RELATIONSHIP CONTINUITY", "\n".join(context.relationship))
    if context.memories:
        add("memories", "RELEVANT DURABLE MEMORIES", "\n".join(context.memories))
    if context.preferences:
        add("preferences", "RELEVANT PREFERENCES", "\n".join(context.preferences))
    if context.opinions:
        add("opinions", "RELEVANT OPINIONS", "\n".join(context.opinions))
    if context.interests:
        add("interests", "RELEVANT INTERESTS", ", ".join(context.interests))
    add("date_time", "CURRENT TIME", context.date_time)
    return tuple(values)


def _compile(
    *,
    mode: str,
    current_input: str,
    protocol: str,
    context: PromptContext,
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None,
    retry_note: str = "",
) -> PromptPlan:
    stable = _character_parts()
    system_parts = [*stable, f"[{mode.upper()}]\n{protocol}"]
    included = ["identity", "soul", "hard_rules", "protocol"]
    trimmed: list[str] = []
    required_chars = sum(len(part) for part in system_parts) + len(current_input) + 128
    remaining = max(0, _PROMPT_CHAR_BUDGET - required_chars)

    if initiative_message := _recent_initiative(context.recent_turns):
        initiative_context = (
            "[RECENT INITIATIVE MESSAGE]\n" + initiative_message
        )
        if len(initiative_context) <= remaining:
            system_parts.append(initiative_context)
            included.append("recent_initiative")
            remaining -= len(initiative_context)
        else:
            trimmed.append("recent_initiative:character_budget")

    if context.presence:
        presence = "[CURRENT PRESENCE]\n" + _text(context.presence)
        if len(presence) <= remaining:
            system_parts.append(presence)
            included.append("presence")
            remaining -= len(presence)
        else:
            trimmed.append("presence:character_budget")

    selected_groups: list[tuple[object, ...]] = []
    groups = _complete_context_groups(context.recent_turns)
    pair_budget = max(0, remaining - min(900, remaining // 3))
    for group in reversed(groups):
        size = sum(len(_turn_value(turn, "content")) + 32 for turn in group)
        if size <= pair_budget:
            selected_groups.append(group)
            pair_budget -= size
            remaining -= size
        else:
            trimmed.append("recent_context:character_budget")
            break
    selected_groups.reverse()
    if selected_groups:
        included.append(f"recent_context:{len(selected_groups)}")

    sections = list(_context_sections(context))
    if mode == "initiative" and context.initiative_opportunity:
        sections.insert(
            0,
            (
                "initiative_opportunity",
                "[GROUNDED OPPORTUNITY]\n" + _text(context.initiative_opportunity),
            ),
        )
    if retry_note:
        sections.insert(0, ("retry_note", "[PREVIOUS PROPOSAL]\n" + _text(retry_note)))
    for name, section in sections:
        if len(section) <= remaining:
            system_parts.append(section)
            included.append(name)
            remaining -= len(section)
        else:
            trimmed.append(f"{name}:character_budget")

    messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
    for group in selected_groups:
        messages.extend(
            {
                "role": _turn_value(turn, "role"),
                "content": _turn_value(turn, "content"),
            }
            for turn in group
        )
    messages.append({"role": "user", "content": current_input})

    token_ids: tuple[int, ...] = ()
    method = "not_tokenized"
    stops: tuple[str, ...] = ()
    if token_counter is not None:
        count = token_counter(messages)
        if len(count.tokens) > LLAMA_CONTEXT_WINDOW - MAX_TOKENS:
            raise RuntimeError(
                "Akane's final prompt exceeds the configured context window."
            )
        token_ids = count.tokens
        method = count.method
        stops = count.stop_sequences
    return PromptPlan(
        messages=messages,
        token_ids=token_ids,
        counting_method=method,
        stop_sequences=stops,
        trimmed=tuple(trimmed),
        included=tuple(included),
        mode=mode,
    )


def build_conversation_prompt(
    user_text: str,
    context: PromptContext,
    *,
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None = None,
) -> PromptPlan:
    return _compile(
        mode="conversation",
        current_input=str(user_text or ""),
        protocol=_STATE_PROTOCOL,
        context=context,
        token_counter=token_counter,
    )


def build_life_prompt(
    context: PromptContext,
    *,
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None = None,
    retry_note: str = "",
) -> PromptPlan:
    return _compile(
        mode="life",
        current_input="Decide the next offscreen-life lifecycle.",
        protocol=_LIFE_PROTOCOL,
        context=context,
        token_counter=token_counter,
        retry_note=retry_note,
    )


def build_initiative_prompt(
    context: PromptContext,
    *,
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None = None,
) -> PromptPlan:
    return _compile(
        mode="initiative",
        current_input="Decide whether this grounded opportunity deserves outreach.",
        protocol=_INITIATIVE_PROTOCOL,
        context=context,
        token_counter=token_counter,
    )
