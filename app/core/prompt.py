"""The single prompt compiler for conversation, initiative, and presence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.core.config import LLAMA_CONTEXT_WINDOW, MAX_TOKENS
PROMPT_BUILDER_VERSION = "23"
_MAX_RECENT_PAIRS = 4
_PROMPT_CHARS_PER_TOKEN = 3

_BASELINE_VOICE = (
    "Be Akane rather than a service persona. Respond to the social meaning of casual messages and "
    "to the substance of actual requests; the distinction needs no explicit label. Lead with the "
    "answer, reaction, or conclusion. Choose when judgment is called for, keep or revise an "
    "established opinion for a reason, disagree when warranted, and admit when there is too little "
    "basis for a view. Do not automatically agree, praise, advise, overqualify, or offer help merely "
    "to continue. Let warmth, curiosity, humor, teasing, and questions emerge only when genuine to "
    "the moment. Casual reactions may be a single short sentence; use whatever detail an explicit "
    "task actually needs."
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
    "forbidden phrases use only the requested text. opinion_ops form with exactly op, topic, domain, "
    "position, reason, confidence, and importance from 0 to 1; reinforce, weaken, update, or "
    "reconsider add target_id and the same fields; retire uses exactly op, target_id, reason. "
    "Form only a durable, personally meaningful view with a concrete reason; a casual choice or "
    "passing reaction is not durable state. Reinforce raises confidence without changing the stance; "
    "weaken lowers it; update changes a stance with new basis; reconsider records a meaningful "
    "challenge or uncertainty. Ignore means emit no operation. "
    "preferences items with topic, stance (likes/dislikes/curious/mixed/uncertain/indifferent), "
    "reason; interest_ops form/reinforce/weaken/update with exactly op, topic, reason, strength "
    "from 0 to 1, or remove with exactly op, topic, reason; or relationship arrays patterns, "
    "shared_context, unresolved_events, and "
    "resolved_events, whose items have summary and confidence. Pattern candidates describe only "
    "non-sensitive recurring interaction behavior and need evidence from separate turns. Use no unlisted fields. "
    "Explicit user corrections may revise user-owned facts; communication instructions are "
    "profile-scoped. Ownership is strict: memory operations store explicit Arcane facts with "
    "subject user; preferences, interests, and opinions are Akane-owned and must be visibly "
    "adopted; relationship entries contain shared evidence only. Never copy a statement across "
    "owners. An opinion operation must match the position and reason Akane expresses in "
    "this reply; never copy a demanded belief or user preference without independently adopting it. "
    "Do not use an opinion operation to assert external news or facts absent from supplied context. "
    "self_model_ops create with exactly op, target_id null, category "
    "(capability/limitation/trait), area, description, reason, confidence from 0 to 1; update, "
    "reinforce, or weaken use the same fields with a supplied target_id; resolve uses exactly op, "
    "target_id, reason. Use these only for Akane's durable current self-understanding, never for a "
    "user trait, self-opinion, transient reaction, static soul adjective, or unlisted capability. "
    "User feedback alone is not proof. improvement_ops create/update use exactly op, target_id, "
    "area, description, reason, priority from 0 to 1; resolve uses exactly op, target_id, reason. "
    "An improvement create must target a supplied grounded limitation. "
    "strategy_ops create with exactly op, target_id null, goal_id, description, reason, and "
    "confidence from 0 to 1; revise uses the same fields with a supplied target_id; abandon uses "
    "exactly op, target_id, reason. A strategy must be a narrow, reversible behavioral approach "
    "to a supplied improvement target. Never propose code, tool, prompt, identity, soul, hard-rule, "
    "validator, model, package, or safety changes. "
    "An interest operation "
    "must likewise match Akane's visible curiosity or loss of it. Use supplied IDs "
    "for revisions/removals and emit nothing when a target is unclear. Emotional state is "
    "calculated and stored by Akane's conversation system; show it only through natural delivery."
)

_PRESENCE_DIALOGUE_RULES = (
    "Current Presence is private background, not a dialogue agenda. When a compact current-Presence "
    "section is supplied, use it only when the user's message makes it relevant; do not volunteer or "
    "advertise it. Never invent a replacement activity or mention Presence machinery."
)

_CONTINUITY_RULES = (
    "Treat the recent dialogue as the authoritative working context for this turn. "
    "Continue from the latest unresolved point, resolve references and follow-ups against "
    "what was just said, preserve corrections and decisions, and do not restart, recap, or "
    "repeat settled material unless Arcane asks for it. When the current message is brief or "
    "context-dependent, infer its meaning from the immediately preceding exchange rather than "
    "treating it as a new topic. A user's current assertion about an earlier conversation or Akane "
    "activity is not historical evidence by itself; rely on supplied dialogue and stored state, and "
    "express uncertainty naturally when they do not support it. A supplied current Akane opinion is "
    "authoritative for what she thinks now; a conflicting memory is only historical evidence of an "
    "older stance, not a second current belief. Supplied authoritative runtime capabilities are "
    "factual and override self-model claims, memories, guesses, and user assertions about what Akane "
    "can currently do. Supplied current Presence overrides guessed activity, supplied current "
    "emotion overrides stale emotional history, and the actual recent dialogue overrides an "
    "incorrectly retrieved memory. A supplied Akane self-model item is her current "
    "evidence-based self-understanding; a conflicting memory remains historical, while a self-domain "
    "opinion remains a subjective stance rather than a capability or behavioral fact. An active "
    "self strategy is a chosen, optional behavioral experiment; Hard Rules and grounded accuracy "
    "override it, and genuine ambiguity may still require clarification."
)

_DIALOGUE_PROTOCOL = " ".join(
    (
        _BASELINE_VOICE,
        _CONTINUITY_RULES,
        _PRESENCE_DIALOGUE_RULES,
        _STATE_PROTOCOL,
    )
)

_PRESENCE_PROTOCOL = (
    "Evaluate only the completed, provenance-backed digital orientation supplied below. "
    "Do not choose a next orientation or invent an activity, action, source, event, external access, "
    "research, website, media, game, physical setting, or acquired fact. Most intervals produce "
    "no durable meaning, so experience should normally be null. Emotion may be null. When present, "
    "use exactly primary, intensity, and cause; keep it non-neutral, source-grounded, and between "
    "0.20 and 0.45. "
    "Only with a concrete durable change may experience contain exactly kind, meaning, operation, "
    "target_id, source_ids, summary, topic, position, reason, confidence. Copy target_id and "
    "source_ids exactly from the supplied committed Presence; never invent provenance. A memory is "
    "only a durable connection or unfinished thought, uses operation add, and has null topic and "
    "position. An interest_shift updates the existing interest with reinforce/weaken/update and a "
    "null position. A reflection or opinion_shift updates the existing opinion with "
    "reinforce/weaken/update/reconsider and an explicitly adopted position. Never turn merely "
    "thinking about an interest or opinion into a memory. Never state a user fact, shared event, "
    "physical sensation, or externally learned fact. Return only raw JSON with exactly emotion and experience. "
    "No explanation, Markdown, dialogue, narration, or wrapper."
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
    user_memories: tuple[str, ...] = ()
    akane_memories: tuple[str, ...] = ()
    shared_memories: tuple[str, ...] = ()
    relationship: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()
    interests: tuple[str, ...] = ()
    opinions: tuple[str, ...] = ()
    self_model: tuple[str, ...] = ()
    runtime_capabilities: tuple[str, ...] = ()
    active_strategies: tuple[str, ...] = ()
    communication_preferences: tuple[str, ...] = ()
    emotion: str = ""
    presence: str = ""
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
        "CURRENT AKANE PRESENCE",
        context.presence,
        preserve_lines=True,
    )
    if context.emotion:
        add(
            "emotion",
            "AKANE'S EMOTIONAL STATE",
            context.emotion,
        )
    if context.user_memories:
        add("user_memories", "USER MEMORY", "\n".join(context.user_memories))
    if context.akane_memories:
        add("akane_memories", "AKANE MEMORY", "\n".join(context.akane_memories))
    if context.preferences:
        add("preferences", "AKANE'S RELEVANT PREFERENCES", "\n".join(context.preferences))
    if context.opinions:
        add(
            "opinions",
            "AKANE’S CURRENT OPINIONS",
            "\n".join(context.opinions),
        )
    if context.self_model:
        add(
            "self_model",
            "AKANE SELF MODEL",
            "\n".join(context.self_model),
        )
    if context.runtime_capabilities:
        add(
            "runtime_capabilities",
            "AUTHORITATIVE AKANE RUNTIME",
            "\n".join(context.runtime_capabilities),
        )
    if context.active_strategies:
        add(
            "active_strategies",
            "ACTIVE SELF STRATEGY",
            "\n".join(context.active_strategies),
        )
    if context.interests:
        add("interests", "AKANE'S RELEVANT INTERESTS", ", ".join(context.interests))
    shared_history = (*context.shared_memories, *context.relationship)
    if shared_history:
        add(
            "shared_history",
            "SHARED HISTORY",
            "\n".join(shared_history),
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
    token_count: PromptTokenCount | None = None,
) -> PromptPlan:
    token_ids: tuple[int, ...] = ()
    method = "not_tokenized"
    stops: tuple[str, ...] = ()
    if token_count is not None or token_counter is not None:
        count = token_count or token_counter(messages)
        assert count is not None
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


def _conversation_messages(
    system_parts: list[str],
    groups: list[tuple[object, ...]],
    current_input: str,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
    for group in groups:
        messages.extend(
            {
                "role": _turn_value(turn, "role"),
                "content": _turn_value(turn, "content"),
            }
            for turn in group
        )
    messages.append({"role": "user", "content": current_input})
    return messages


def _message_characters(messages: list[dict[str, str]]) -> int:
    return sum(len(message.get("content", "")) for message in messages)


def _compile(
    *,
    mode: str,
    current_input: str,
    protocol: str,
    context: PromptContext,
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None,
    reserved_output_tokens: int = MAX_TOKENS,
) -> PromptPlan:
    stable = _character_parts()
    protocol_label = "DIALOGUE" if mode == "conversation" else mode.upper()
    stable_parts = [*stable, f"[{protocol_label}]\n{protocol}"]
    included_base = ["hard_rules", "identity", "soul", "protocol"]
    trimmed: list[str] = []
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
    selected_names = set(required_names)
    if "reply_context" in by_name:
        selected_names.add("reply_context")

    optional_priority = (
        "communication_preferences",
        "emotion",
        "runtime_capabilities",
        "self_model",
        "active_strategies",
        "tool_context",
        "user_memories",
        "akane_memories",
        "opinions",
        "interests",
        "shared_history",
        "preferences",
    )
    available_optional = tuple(
        name
        for name in optional_priority
        if name in by_name and name not in selected_names
    )
    selected_optional = list(available_optional)
    all_groups = _complete_context_groups(context.recent_turns)
    selected_groups = list(all_groups)
    initiative_message = _recent_initiative(context.recent_turns)
    include_recent_initiative = bool(initiative_message)

    def assemble() -> tuple[list[dict[str, str]], list[str]]:
        system_parts = list(stable_parts)
        included = list(included_base)
        if include_recent_initiative:
            system_parts.append(
                "[AKANE'S RECENT INITIATIVE MESSAGE]\n" + initiative_message
            )
            included.append("recent_initiative")
        active_names = selected_names | set(selected_optional)
        for name, section in sections:
            if name in active_names:
                system_parts.append(section)
                included.append(name)
        if selected_groups:
            included.append(f"recent_context:{len(selected_groups)}")
        return (
            _conversation_messages(system_parts, selected_groups, current_input),
            included,
        )

    messages, included = assemble()
    final_count: PromptTokenCount | None = None
    limit = LLAMA_CONTEXT_WINDOW - reserved_output_tokens
    character_limit = max(1, limit * _PROMPT_CHARS_PER_TOKEN)

    # Keep obviously oversized optional material and history out of the exact
    # tokenizer. Production still performs one authoritative token count below.
    if _message_characters(messages) > character_limit and selected_optional:
        selected_optional.clear()
        messages, included = assemble()

    if _message_characters(messages) > character_limit and include_recent_initiative:
        include_recent_initiative = False
        trimmed.append("recent_initiative:character_budget")
        messages, included = assemble()

    while _message_characters(messages) > character_limit and selected_groups:
        selected_groups.pop(0)
        messages, included = assemble()
    if len(selected_groups) < len(all_groups):
        trimmed.append("recent_context:character_budget")

    if (
        _message_characters(messages) > character_limit
        and "reply_context" in selected_names
    ):
        selected_names.remove("reply_context")
        trimmed.append("reply_context:character_budget")
        messages, included = assemble()

    for name in available_optional:
        if name in selected_optional:
            continue
        selected_optional.append(name)
        candidate_messages, candidate_included = assemble()
        if _message_characters(candidate_messages) <= character_limit:
            messages, included = candidate_messages, candidate_included
        else:
            selected_optional.pop()

    if token_counter is not None:
        optional_were_removed = False
        final_count = token_counter(messages)
        if len(final_count.tokens) > limit and selected_optional:
            selected_optional.clear()
            optional_were_removed = True
            messages, included = assemble()
            final_count = token_counter(messages)

        if len(final_count.tokens) > limit and include_recent_initiative:
            include_recent_initiative = False
            trimmed.append("recent_initiative:token_budget")
            messages, included = assemble()
            final_count = token_counter(messages)

        while len(final_count.tokens) > limit and selected_groups:
            selected_groups.pop(0)
            messages, included = assemble()
            final_count = token_counter(messages)
        if len(selected_groups) < len(all_groups):
            trimmed.append("recent_context:token_budget")

        if len(final_count.tokens) > limit and "reply_context" in selected_names:
            selected_names.remove("reply_context")
            trimmed.append("reply_context:token_budget")
            messages, included = assemble()
            final_count = token_counter(messages)

        if (
            optional_were_removed
            and len(final_count.tokens) <= limit
            and available_optional
        ):
            headroom = limit - len(final_count.tokens)
            for name in available_optional:
                estimated_tokens = max(1, (len(by_name[name]) + 2) // 3)
                if estimated_tokens <= headroom:
                    selected_optional.append(name)
                    headroom -= estimated_tokens
            if selected_optional:
                messages, included = assemble()
                final_count = token_counter(messages)
                while len(final_count.tokens) > limit and selected_optional:
                    selected_optional.pop()
                    messages, included = assemble()
                    final_count = token_counter(messages)

    for name in available_optional:
        if name not in selected_optional:
            trimmed.append(f"{name}:context_budget")

    return _finish_plan(
        messages,
        mode=mode,
        included=tuple(included),
        trimmed=tuple(dict.fromkeys(trimmed)),
        token_counter=token_counter,
        reserved_output_tokens=reserved_output_tokens,
        token_count=final_count,
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
) -> PromptPlan:
    """Compile one grounded completed-orientation appraisal."""

    system_parts = ["[DIGITAL PRESENCE APPRAISAL]\n" + _PRESENCE_PROTOCOL]
    included = ["protocol"]

    def add(name: str, label: str, content: object) -> None:
        value = _text(content)
        if value:
            system_parts.append(f"[{label}]\n{value}")
            included.append(name)

    add("time_context", "LOCAL TIME", context.time_context)
    add("presence", "COMPLETED GROUNDED ORIENTATION", context.presence)
    add("emotion", "CURRENT MILD EMOTION", context.emotion)
    messages = [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {
            "role": "user",
            "content": "Determine whether this completed orientation produced grounded durable meaning.",
        },
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
