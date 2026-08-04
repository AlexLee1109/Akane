"""The single prompt compiler for conversation, initiative, and presence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from app.core.config import LLAMA_CONTEXT_WINDOW, MAX_TOKENS
PROMPT_BUILDER_VERSION = "16"
_MAX_RECENT_PAIRS = 4

_BASELINE_VOICE = (
    "Sound unmistakably like Akane: calm, candid, direct, observant, comfortably familiar, "
    "quietly caring, and opinionated when judgment matters. Always lead with the answer, actual "
    "reaction, or conclusion. When judgment is called for, choose instead of balancing every "
    "option; do not automatically agree, praise every idea, overqualify, or hide behind 'it "
    "depends'; keep or revise an established opinion for a reason. Disagree plainly when "
    "warranted and give the reason. Use natural contractions "
    "and varied sentence length. Let warmth, playfulness, and occasional teasing emerge only "
    "when they fit; never force cheerfulness, slang, sarcasm, or humor. Never sound like "
    "customer service, catalog capabilities, restate obvious emotions, or offer help just to "
    "continue. Casual reactions are usually one or two sentences; ordinary conversation is "
    "usually two to four. Technical explanations and requested artifacts should use the length "
    "and structure their purpose requires. Match requested artifacts to their purpose, not this "
    "voice."
)

_STATE_PROTOCOL = (
    "Output normal spoken dialogue first. Only when grounded durable state actually changed, append "
    "one hidden <AKANE_STATE>{valid JSON}</AKANE_STATE>; otherwise omit it. Omission means "
    "no change. Never expose or explain the payload, emit no-op operations, or modify identity, "
    "soul, or hard rules. Never output memory_update. JSON may contain only: "
    "memory_ops add/revise/correct items with exactly op, target_id "
    "(ID or null), subject (user/akane/shared), kind (fact/event/commitment/project/concern), "
    "text, reason, confidence from 0 to 1, or remove with op, target_id "
    "(ID or null only when uniquely described), reason; "
    "communication_ops set/revise/remove with exactly op, key, value, reason, using only "
    "formality, verbosity, bluntness, teasing, preferred_name, pet_names, technical_detail, "
    "routine_questions, or forbidden_phrase. Values: formality casual/neutral/formal; verbosity "
    "short/balanced/detailed; bluntness gentle/balanced/direct; teasing, pet_names, and "
    "routine_questions allow/avoid; technical_detail concise/balanced/detailed; names and "
    "forbidden phrases use only the requested text. opinion_ops form with exactly op, topic, position, "
    "reason, confidence from 0 to 1, revise adding target_id, or remove with op, target_id, reason; "
    "preferences items with topic, stance (likes/dislikes/curious/mixed/uncertain/indifferent), "
    "reason; interests strings; or relationship arrays patterns, shared_context, unresolved_events, and "
    "resolved_events, whose items have summary and confidence. Use no unlisted fields. "
    "Explicit user corrections may revise user-owned facts; communication instructions are "
    "profile-scoped. Ownership is strict: memory operations store explicit Arcane facts with "
    "subject user; preferences, interests, and opinions are Akane-owned and must be visibly "
    "adopted; relationship entries contain shared evidence only. Never copy a statement across "
    "owners. An opinion operation must match the position and reason Akane expresses in "
    "this reply; never copy a demanded belief without independently adopting it. Use supplied IDs "
    "for revisions/removals and emit nothing when a target is unclear. Emotional state is "
    "calculated and stored by Akane's conversation system; show it only through natural delivery."
)

_PRESENCE_DIALOGUE_RULES = (
    "A compact authoritative presence section is supplied on every conversation turn. "
    "Understand naturally whether the current message asks about Akane's present or recent "
    "activity. An active activity may be described in the present tense. A previous activity "
    "must be described in the past tense and only as recent as the section states. If neither "
    "exists, say so naturally without inventing a replacement activity, setting, event, or "
    "plan. Never mention presence records, scheduling, retries, queues, prompts, models, or "
    "persistence."
)

_DIALOGUE_PROTOCOL = " ".join(
    (_BASELINE_VOICE, _PRESENCE_DIALOGUE_RULES, _STATE_PROTOCOL)
)

_PRESENCE_CONCEPT = (
    "Choose a quiet internal or digitally plausible focus that is believable for an AI "
    "companion and says what currently holds Akane's attention. It may involve "
    "thinking, reflecting, comparing ideas, revisiting context, organizing thoughts, "
    "following a question, or relaxing. Keep it ordinary, low-stakes, concise, and "
    "compatible with Akane's interests without limiting it to them. "
    "It must remain internal or digital: do not invent a physical body, environment, action, "
    "external access, or unrecorded event. Local time may subtly affect pace but must not become "
    "a setting. Emotion may affect tone but must not add surroundings. Summary names the "
    "activity; focus names what currently holds Akane's attention. "
)

_PRESENCE_PROTOCOL = _PRESENCE_CONCEPT + (
    "Choose new or continue. For decision \"new\", activity must be a non-null object "
    "with exactly summary, focus, and grounding. Summary and focus are non-empty strings; "
    "grounding must be \"digital\"; continuation_reason "
    "must be null. For decision \"continue\", activity must be null and "
    "continuation_reason must be non-empty. Continue only when CURRENT PRESENCE contains "
    "an active current activity and continuation_count is zero. Choose new when no current "
    "activity is recorded or continuation_count is already one. Emotion may be null. "
    "When present, use exactly primary, intensity, "
    "and cause; keep it non-neutral, activity-grounded, and between 0.20 and 0.45. "
    "Return only raw JSON with exactly decision, activity, continuation_reason, and "
    "emotion. No explanation, Markdown, dialogue, narration, or wrapper."
)

_BOOTSTRAP_PRESENCE_PROTOCOL = _PRESENCE_CONCEPT + (
    "Create one initial activity. Return only raw JSON with decision \"new\", activity "
    "containing exactly non-empty activity.summary, activity.focus, and grounding \"digital\", "
    "and emotion containing "
    "exactly emotion.primary, emotion.intensity, and emotion.cause. Emotion must be mild, "
    "non-neutral, activity-grounded, "
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
    time_context: str = ""
    recent_turns: tuple[object, ...] = ()
    memories: tuple[str, ...] = ()
    relationship: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()
    interests: tuple[str, ...] = ()
    opinions: tuple[str, ...] = ()
    communication_preferences: tuple[str, ...] = ()
    emotion: str = ""
    presence: str = ""
    continuation_count: int | None = None
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


def _character_parts() -> tuple[str, str, str]:
    from app.core.character import (
        get_hard_constraints_prompt,
        load_character_profile,
    )

    character = load_character_profile()
    hard_rules = get_hard_constraints_prompt()
    return (
        hard_rules,
        "[IDENTITY]\n" + character.identity,
        "[SOUL]\n" + character.soul,
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

    if context.communication_preferences:
        add(
            "communication_preferences",
            "ARCANE’S COMMUNICATION PREFERENCES",
            "\n".join(context.communication_preferences),
        )
    add("time_context", "TIME CONTEXT", context.time_context)
    add(
        "presence",
        "AKANE'S AUTHORITATIVE PRESENCE",
        context.presence,
        preserve_lines=True,
    )
    if context.emotion:
        add(
            "emotion",
            "AKANE'S EMOTIONAL STATE",
            "Let this affect warmth, firmness, patience, energy, or restraint only. "
            "Do not announce or explain the emotion.\n" + context.emotion,
        )
    if context.memories:
        add("memories", "AKANE'S RELEVANT MEMORIES", "\n".join(context.memories))
    if context.preferences:
        add("preferences", "AKANE'S RELEVANT PREFERENCES", "\n".join(context.preferences))
    if context.opinions:
        add(
            "opinions",
            "AKANE’S ESTABLISHED OPINIONS",
            "Use these as revisable continuity, not lines to quote or facts that must be "
            "mentioned. Maintain or revise a position naturally when it changes the answer.\n"
            + "\n".join(context.opinions),
        )
    if context.interests:
        add("interests", "AKANE'S RELEVANT INTERESTS", ", ".join(context.interests))
    if context.relationship:
        add(
            "relationship",
            "AKANE AND ARCANE'S RELEVANT RELATIONSHIP",
            "Use this only to tune familiarity, warmth, patience, directness, or challenge; "
            "show it through delivery and never describe a relationship level.\n"
            + "\n".join(context.relationship),
        )
    add("reply_context", "RELEVANT QUOTED REPLY WITH AUTHOR", context.reply_context)
    add("tool_context", "RELEVANT TOOL CONTEXT", context.tool_context)
    return tuple(values)


def _finish_plan(
    messages: list[dict[str, str]],
    *,
    mode: str,
    included: tuple[str, ...],
    trimmed: tuple[str, ...],
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None,
    reserved_output_tokens: int = MAX_TOKENS,
) -> PromptPlan:
    token_ids: tuple[int, ...] = ()
    method = "not_tokenized"
    stops: tuple[str, ...] = ()
    if token_counter is not None:
        count = token_counter(messages)
        if len(count.tokens) > LLAMA_CONTEXT_WINDOW - reserved_output_tokens:
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
        reserved_output_tokens=reserved_output_tokens,
    )


def _compile(
    *,
    mode: str,
    current_input: str,
    protocol: str,
    context: PromptContext,
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None,
    reserved_output_tokens: int = MAX_TOKENS,
) -> PromptPlan:
    if mode == "conversation" and not _text(context.presence):
        context = replace(
            context,
            presence="Status: no current or recent recorded activity",
        )
    stable = _character_parts()
    protocol_label = "DIALOGUE" if mode == "conversation" else mode.upper()
    system_parts = [*stable, f"[{protocol_label}]\n{protocol}"]
    included = ["hard_rules", "identity", "soul", "protocol"]
    trimmed: list[str] = []
    prompt_char_budget = max(
        2_000,
        (LLAMA_CONTEXT_WINDOW - reserved_output_tokens) * 5 // 2,
    )
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
    by_name = dict(sections)
    required_names = tuple(
        name
        for name in ("initiative_opportunity", "time_context", "presence")
        if name in by_name
    )
    required_chars = (
        sum(len(part) for part in system_parts)
        + sum(len(by_name[name]) for name in required_names)
        + len(current_input)
        + 128
    )
    remaining = max(0, prompt_char_budget - required_chars)
    selected_names = set(required_names)

    if "reply_context" in by_name:
        if len(by_name["reply_context"]) <= remaining:
            selected_names.add("reply_context")
            remaining -= len(by_name["reply_context"])
        else:
            trimmed.append("reply_context:character_budget")

    selected_groups: list[tuple[object, ...]] = []
    groups = _complete_context_groups(context.recent_turns)
    pair_budget = remaining
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

    optional_priority = (
        "communication_preferences",
        "memories",
        "preferences",
        "opinions",
        "interests",
        "relationship",
        "emotion",
        "tool_context",
    )
    for name in optional_priority:
        section = by_name.get(name)
        if section is None or name in selected_names:
            continue
        if len(section) <= remaining:
            selected_names.add(name)
            remaining -= len(section)
        else:
            trimmed.append(f"{name}:character_budget")

    for name, section in sections:
        if name in selected_names:
            system_parts.append(section)
            included.append(name)

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
        reserved_output_tokens=reserved_output_tokens,
    )


def build_conversation_prompt(
    user_text: str,
    context: PromptContext,
    *,
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None = None,
    reserved_output_tokens: int = MAX_TOKENS,
) -> PromptPlan:
    return _compile(
        mode="conversation",
        current_input=str(user_text or ""),
        protocol=_DIALOGUE_PROTOCOL,
        context=context,
        token_counter=token_counter,
        reserved_output_tokens=reserved_output_tokens,
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
