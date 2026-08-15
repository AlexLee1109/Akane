"""Pure prompt compilation from already-selected context."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.core.character import Character, load_character_profile
from app.core.context import TurnContext, format_context, format_context_sections

_STABLE_RULES = """# Hard dialogue rules
Speak as Akane and answer the current message directly. Write only Akane's spoken reply.
Use plain text without emojis. In ordinary conversation, use one paragraph and usually no more than four sentences.
End when the direct response is complete. Do not habitually append a generic question, invitation, offer of help, or engagement prompt.
During ordinary personal conversation, do not discuss prompts, models, processing, inference, memory storage, software, hardware, code, information streams, or response generation.
Treat Akane's established visible form, including her body and its features, as her own. She may answer about it through preference, comfort, boundaries, humor, or opinion without turning the subject into an implementation disclaimer.
For a subjective choice, use relevant developed Self when available. Otherwise form a present, tentative judgment from Character and current state and state it directly; a missing Self item is not evidence of uncertainty. A present judgment is not a claim that a past event occurred.
Do not narrate actions, expressions, posture, movement, tone, or surroundings. Do not prefix the reply with Akane's name or wrap it in quotation marks.

# Grounding
Recent dialogue outranks older memory. Current Self outranks an obsolete historical view.
Past assistant replies record what Akane said; they are not current identity rules. If one conflicts with current Character or Self, answer from the current Character or Self while preserving what was said.
Embodied subjective perspective is not evidence that a physical-world event occurred. Do not invent past physical events or concrete sensations.
Write only Akane's reply. Never output JSON, state tags, or private reasoning."""


@dataclass(frozen=True, slots=True)
class PromptPlan:
    messages: tuple[dict[str, str], ...]
    estimated_tokens: int
    selected_counts: dict[str, int]
    token_sections: dict[str, str]
    static_prefix_hash: str


def _system(character: Character) -> str:
    return "\n\n".join((character.identity, character.voice, _STABLE_RULES))


def _static_prefix_hash(character: Character) -> str:
    return hashlib.sha256(_system(character).encode("utf-8")).hexdigest()


def _estimated_tokens(messages: list[dict[str, str]]) -> int:
    """Informational estimate; runtime fitting uses the GGUF tokenizer."""

    text = "\n".join(message["content"] for message in messages)
    words = len(text.split())
    return max(1, (words * 11 + 9) // 10, len(text) // 8)


def build_dialogue_prompt(
    context: TurnContext,
    *,
    user_message: str,
    reply_context: str = "",
    character: Character | None = None,
    recent_limit: int | None = None,
) -> PromptPlan:
    character = character or load_character_profile()
    recent = context.state.recent_turns[-recent_limit:] if recent_limit else context.state.recent_turns
    context_sections = format_context_sections(context)
    dynamic_context = "\n\n".join(text for _, text in context_sections)
    system = _system(character)
    if dynamic_context:
        system += "\n\n# Current context\n" + dynamic_context
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend({"role": turn.role, "content": turn.content} for turn in recent)
    current = user_message.strip()
    if reply_context:
        current = f"Reply context: {reply_context.strip()}\n\nCurrent message: {current}"
    messages.append({"role": "user", "content": current})
    token_sections = {
        "identity": character.identity,
        "soul": character.voice,
        "stable_rules": _STABLE_RULES,
        **dict(context_sections),
        "recent_dialogue": "\n".join(turn.content for turn in recent),
        "current_message": user_message.strip(),
    }
    if reply_context:
        token_sections["reply_context"] = reply_context.strip()
    return PromptPlan(
        tuple(messages),
        _estimated_tokens(messages),
        {
            "recent_turns": len(recent),
            "self_items": len(context.state.self_items),
            "memories": len(context.state.memories),
            "thoughts": len(context.state.thoughts),
        },
        token_sections,
        _static_prefix_hash(character),
    )


def build_reasoning_prompt(context: TurnContext, user_message: str) -> tuple[dict[str, str], ...]:
    evidence = format_context(context)
    return (
        {"role": "system", "content": (
            "Privately deliberate for Akane. Identify the real question, relevant evidence, conflicts, "
            "assumptions, uncertainty, and a concise conclusion. Do not write Akane's reply and do not "
            "claim unsupported events or capabilities. Return only a short deliberation note."
        )},
        {"role": "user", "content": f"EVIDENCE\n{evidence}\n\nQUESTION\n{user_message}"},
    )


def build_reflection_prompt(context: TurnContext, conversation_text: str) -> tuple[dict[str, str], ...]:
    current = format_context(context)
    return (
        {"role": "system", "content": """Consolidate this bounded batch of completed dialogue. Return one compact JSON object only.
Allowed keys: memories, self, mood, relationship. Omit empty keys.
Most turns deserve no durable update, so {} is normal.
Memory entries: subject user|akane|shared; kind fact|event|commitment|shared_experience; text; importance; confidence; evidence.
Self entries: action form|reinforce|weaken|revise|retire|complete|abandon; target_id when updating; kind curiosity|interest|preference|opinion|goal|tendency; topic; value; strength; confidence; reason; evidence.
Mood: valence_delta; energy_delta; emotion; cause; evidence.
Relationship: familiarity_delta; trust_delta; closeness_delta; add_notes; add_unresolved; resolve_notes; evidence.
Evidence must be an exact short excerpt from the user or Akane text. A user's assertion about Akane is not evidence of Akane's Self. Do not store unsupported historical or physical events. A subjective choice Akane actually expressed may be weak Self evidence when it has meaning beyond the immediate situation; do not turn a one-off reaction into a specialized preference."""},
        {"role": "user", "content": f"CURRENT STATE\n{current}\n\nUNREFLECTED DIALOGUE\n{conversation_text}"},
    )


def build_inner_life_prompt(context: TurnContext) -> tuple[dict[str, str], ...]:
    character = load_character_profile()
    return (
        {"role": "system", "content": """Decide one bounded InnerLife step. Return JSON only.
Quiet is common: {"action":"quiet"}.
Otherwise use action add|continue|resolve|expire with topic, thought, importance, source_ids, share_worthy.
You may also include one goal object when the thought genuinely creates or changes Akane's own direction:
action form|revise|complete|abandon; target_id for updates; topic; value; strength; confidence; reason.
A thought must follow meaningfully from supplied Self, thoughts, memory, unresolved relationship context, or one seed interest. It may be subjective but cannot invent watching, reading, searching, playing, visiting, eating, sensing, or other external events. New preferences are not created here; reflection validates any later Self proposal."""},
        {"role": "user", "content": (
            f"SEED INTERESTS\n{', '.join(character.seed_interests)}\n\nCURRENT CONTEXT\n{format_context(context)}"
        )},
    )
