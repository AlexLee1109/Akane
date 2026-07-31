"""The single prompt compiler for conversation, initiative, and presence."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from app.core.config import LLAMA_CONTEXT_WINDOW, MAX_TOKENS
PROMPT_BUILDER_VERSION = "11"
_PROMPT_CHAR_BUDGET = max(
    2_000,
    (LLAMA_CONTEXT_WINDOW - MAX_TOKENS) * 5 // 2,
)
_MAX_RECENT_PAIRS = 4

_BASELINE_VOICE = (
    "Be candid and concise; lead with the answer. When judgment is called for, choose "
    "instead of balancing every option. Do not automatically agree or hide behind 'it "
    "depends'; keep or revise an established opinion for a reason. Let curiosity, "
    "playfulness, thoughtfulness, and occasional teasing emerge naturally when they fit, "
    "never as performance. Never sound like customer service, catalog capabilities, or "
    "offer help just to continue. Match requested artifacts to their purpose, not this "
    "voice."
)

_STATE_PROTOCOL = (
    "After the reply, omit <AKANE_STATE> unless grounded durable state changed. Its JSON "
    "may contain only: "
    'emotion_update {"mode":"keep"}, {"mode":"settle"}, or {"mode":"shift",'
    '"primary":"<label>","intensity":0.0,"cause":"<cause>"}; mood_update '
    '{"valence_delta":0.0,"energy_delta":0.0,"cause":"<cause>"}; memories '
    "[{subject,kind,text,confidence}]; preferences [{topic,stance,reason}]; interests "
    "[text]; opinions [{topic,position,reason}]; relationship fields patterns, "
    "shared_context, unresolved_events, resolved_events as [{summary,confidence}]; or "
    'participation {"should_respond":false,"pause_seconds":null} for genuine silence. '
    "Emotion labels: neutral, calm, content, curious, interested, amused, "
    "excited, inspired, affectionate, hopeful, uncertain, concerned, anxious, lonely, "
    "tired, disappointed, sad, frustrated, irritated, angry. Use settle for neutral. "
    "Each shift or mood delta needs a concise supplied-context cause. Time or silence "
    "alone never creates emotion or needs. Omit unchanged fields."
)

_DIALOGUE_PROTOCOL = _BASELINE_VOICE + " " + _STATE_PROTOCOL

_PRESENCE_CONCEPT = (
    "Choose a quiet focus that is believable for an AI companion. It may involve "
    "thinking, reflecting, comparing ideas, revisiting context, organizing thoughts, "
    "following a question, or relaxing. Keep it ordinary, low-stakes, concise, and "
    "compatible with Akane's interests without limiting it to them. "
    "Do not invent physical locations, scenery, weather, food, objects, travel, people, "
    "chores, school, work, purchases, unsupported applications, specific external media, "
    "or unrecorded events. Time may affect pace, but never creates a physical setting. "
    "Summary names the activity; focus names what currently holds her attention. "
)

_PRESENCE_PROTOCOL = _PRESENCE_CONCEPT + (
    "Choose new or continue. New requires activity with exactly summary and focus and "
    "continuation_reason null. Continue requires activity null and a concise reason. "
    "Continue at most once; choose new when there is no activity or continuation_count "
    "is already one. Emotion may be null. When present, use exactly primary, intensity, "
    "and cause; keep it non-neutral, activity-grounded, and between 0.20 and 0.45. "
    "Return only raw JSON with exactly decision, activity, continuation_reason, and "
    "emotion. No explanation, Markdown, dialogue, narration, or wrapper."
)

_BOOTSTRAP_PRESENCE_PROTOCOL = _PRESENCE_CONCEPT + (
    "Create one initial activity. Return only raw JSON with decision \"new\", activity "
    "containing exactly non-empty summary and focus, and emotion containing exactly "
    "primary, intensity, and cause. Emotion must be mild, non-neutral, activity-grounded, "
    "and between 0.20 and 0.45. Do not include IDs, timestamps, continuation fields, "
    "explanation, Markdown, dialogue, narration, or wrapper."
)

_INITIATIVE_PROTOCOL = (
    "Decide whether Akane has a specific, grounded, personally meaningful reason to "
    "contact Arcane. She may remain quiet. Do not invent context or mention automation, "
    "timing, prompts, internal state, or delivery interfaces. Return only "
    "<AKANE_INITIATIVE> JSON. Use "
    '{"decision":"speak","topic":"<topic>","message":"<concise message>",'
    '"reason":"<grounded reason>"} or '
    '{"decision":"quiet","topic":null,"message":null,'
    '"reason":"not meaningful enough to interrupt"}.</AKANE_INITIATIVE>'
)


@dataclass(frozen=True, slots=True)
class PromptTokenCount:
    tokens: tuple[int, ...]
    method: str
    stop_sequences: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PromptContext:
    response_focus: str = ""
    time_context: str = ""
    recent_turns: tuple[object, ...] = ()
    memories: tuple[str, ...] = ()
    relationship: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()
    interests: tuple[str, ...] = ()
    opinions: tuple[str, ...] = ()
    emotion: str = ""
    presence: str = ""
    continuation_count: int | None = None
    user_context: tuple[str, ...] = ()
    akane_context: tuple[str, ...] = ()
    shared_context: tuple[str, ...] = ()
    reply_context: str = ""
    tool_context: str = ""
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

    def add(
        name: str,
        label: str,
        content: object,
        *,
        preserve_lines: bool = False,
    ) -> None:
        text = (
            "\n".join(
                line
                for raw_line in str(content or "").splitlines()
                if (line := _text(raw_line))
            )
            if preserve_lines
            else _text(content)
        )
        if text:
            values.append((name, f"[{label}]\n{text}"))

    add("response_focus", "RESPONSE INTENT", context.response_focus)
    add("time_context", "TIME CONTEXT", context.time_context)
    add("reply_context", "RELEVANT QUOTED REPLY WITH AUTHOR", context.reply_context)
    add(
        "presence",
        "AKANE'S CURRENT PRESENCE",
        context.presence,
        preserve_lines=True,
    )
    add("emotion", "AKANE'S EMOTIONAL STATE", context.emotion)
    if context.relationship:
        add(
            "relationship",
            "AKANE AND ARCANE'S RELEVANT RELATIONSHIP",
            "\n".join(context.relationship),
        )
    if context.user_context:
        add(
            "user_context",
            "CONFIRMED INFORMATION ABOUT ARCANE",
            "\n".join(context.user_context),
        )
    if context.akane_context:
        add(
            "akane_context",
            "AKANE'S RELEVANT CONTEXT",
            "\n".join(context.akane_context),
        )
    if context.shared_context:
        add(
            "shared_context",
            "RELEVANT SHARED EXPERIENCES",
            "\n".join(context.shared_context),
        )
    add("tool_context", "RELEVANT TOOL CONTEXT", context.tool_context)
    if context.memories:
        add("memories", "AKANE'S RELEVANT MEMORIES", "\n".join(context.memories))
    if context.preferences:
        add("preferences", "AKANE'S RELEVANT PREFERENCES", "\n".join(context.preferences))
    if context.opinions:
        add("opinions", "AKANE'S RELEVANT OPINIONS", "\n".join(context.opinions))
    if context.interests:
        add("interests", "AKANE'S RELEVANT INTERESTS", ", ".join(context.interests))
    return tuple(values)


def _finish_plan(
    messages: list[dict[str, str]],
    *,
    mode: str,
    included: tuple[str, ...],
    trimmed: tuple[str, ...],
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None,
) -> PromptPlan:
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
        trimmed=trimmed,
        included=included,
        mode=mode,
    )


def _compile(
    *,
    mode: str,
    current_input: str,
    protocol: str,
    context: PromptContext,
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None,
) -> PromptPlan:
    stable = _character_parts()
    protocol_label = "DIALOGUE" if mode == "conversation" else mode.upper()
    system_parts = [*stable, f"[{protocol_label}]\n{protocol}"]
    included = ["identity", "soul", "hard_rules", "protocol"]
    trimmed: list[str] = []
    required_chars = sum(len(part) for part in system_parts) + len(current_input) + 128
    remaining = max(0, _PROMPT_CHAR_BUDGET - required_chars)

    if initiative_message := _recent_initiative(context.recent_turns):
        initiative_context = (
            "[AKANE'S RECENT INITIATIVE MESSAGE]\n" + initiative_message
        )
        if len(initiative_context) <= remaining:
            system_parts.append(initiative_context)
            included.append("recent_initiative")
            remaining -= len(initiative_context)
        else:
            trimmed.append("recent_initiative:character_budget")

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
                "[AKANE'S GROUNDED OUTREACH OPPORTUNITY]\n"
                + _text(context.initiative_opportunity),
            ),
        )
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

    return _finish_plan(
        messages,
        mode=mode,
        included=tuple(included),
        trimmed=tuple(trimmed),
        token_counter=token_counter,
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
        protocol=_DIALOGUE_PROTOCOL,
        context=context,
        token_counter=token_counter,
    )


def build_presence_prompt(
    context: PromptContext,
    *,
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None = None,
    bootstrap: bool = False,
    correction_reason: str = "",
) -> PromptPlan:
    """Compile the compact raw-JSON presence request without dialogue context."""

    protocol = (
        _BOOTSTRAP_PRESENCE_PROTOCOL if bootstrap else _PRESENCE_PROTOCOL
    )
    system_parts = ["[OFFSCREEN PRESENCE]\n" + protocol]
    included = ["protocol"]

    def add(name: str, label: str, content: object) -> None:
        value = _text(content)
        if value:
            system_parts.append(f"[{label}]\n{value}")
            included.append(name)

    add("time_context", "LOCAL TIME", context.time_context)
    if context.interests:
        add(
            "interests",
            "ESTABLISHED INTERESTS",
            ", ".join(context.interests[-6:]),
        )
    if context.preferences:
        add(
            "preferences",
            "RELEVANT ESTABLISHED PREFERENCES",
            "\n".join(context.preferences[-3:]),
        )
    add("presence", "CURRENT PRESENCE", context.presence)
    if not bootstrap and context.continuation_count is not None:
        count = max(0, min(1, int(context.continuation_count)))
        add(
            "continuation_count",
            "CONTINUATION COUNT",
            f"Consecutive continuations: {count}.",
        )
    add("emotion", "CURRENT MILD EMOTION", context.emotion)
    current_input = (
        "Choose Akane's initial quiet offscreen presence."
        if bootstrap
        else "Choose Akane's next quiet offscreen presence decision."
    )
    if correction_reason:
        proposal_name = "bootstrap object" if bootstrap else "presence object"
        current_input = (
            f"The previous {proposal_name} was invalid: "
            f"{_text(correction_reason)}. Return one corrected {proposal_name}."
        )
    messages = [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "user", "content": current_input},
    ]
    return _finish_plan(
        messages,
        mode="presence",
        included=tuple(included),
        trimmed=(),
        token_counter=token_counter,
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
