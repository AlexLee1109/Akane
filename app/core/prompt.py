"""Pure prompt compilation from an already-selected context snapshot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

from app.core.character import Character, load_character_profile
from app.core.config import SETTINGS
from app.core.context import TurnContext, format_context_sections
from app.core.streaming import SEMANTIC_SIDECAR_END, SEMANTIC_SIDECAR_START


_STABLE_RULES = f"""A=Akane;U=user;STATE=context;no labels/reasoning.
{SEMANTIC_SIDECAR_START}{{"k":"p","t":"topic","s":"+","d":"c","e":"exact span"}}{SEMANTIC_SIDECAR_END}
p/o/i/g=self pref/opinion/interest/goal;c=correct;f/t±=feedback/task result;x±=prediction;u±=open/resolved curiosity;j±=goal commit/drop;n=none. s=+/-/0/cmp;d=c/tmp/hyp/task/n. Self:e=A. Outcome:e=U,a=prior act,b/f=behavior/effect,r=procedure,j=goal. Predict:d=task,e/a=expect/action,q=h/l. Curiosity:w=focus,e=A/U for u+/-;not a question mandate. Goal:j=intention,e=A;context-only,no autonomous action. v=choice;omit unsupported."""

STATE_MARKER = "STATE"
USER_MARKER = "U"
TRANSIENT_STATE_SECTIONS = frozenset({"time", "code_context"})


@dataclass(frozen=True, slots=True)
class ModelStateItem:
    key: str
    section: str
    wire: str
    version: str
    clear_wire: str
    persistent: bool


@dataclass(frozen=True, slots=True)
class PromptPlan:
    messages: tuple[dict[str, str], ...]
    selected_counts: dict[str, int]
    token_sections: dict[str, str]
    static_prefix_hash: str
    history_messages: tuple[dict[str, str], ...]
    canonical_user_content: str
    turn_user_content: str
    state_revision: int
    state_sections: tuple[tuple[str, str], ...]
    state_items: tuple[ModelStateItem, ...]
    canonical_complete: bool


@lru_cache(maxsize=4)
def _stable_prompt_parts(
    _content_sha256: str,
    identity: str,
    voice: str,
    rules: str,
) -> tuple[str, str]:
    prompt = "\n\n".join(part for part in (identity, voice, rules) if part)
    return prompt, hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def stable_system_prompt(character: Character | None = None) -> str:
    character = character or load_character_profile()
    return _stable_prompt_parts(
        character.content_sha256, character.identity, character.voice, _STABLE_RULES,
    )[0]


def stable_prompt_hash(character: Character | None = None) -> str:
    character = character or load_character_profile()
    return _stable_prompt_parts(
        character.content_sha256, character.identity, character.voice, _STABLE_RULES,
    )[1]


def _recent_turns_within_budget(turns, budget: int, token_counter=None):
    """Keep newest complete exchanges under one deterministic token budget."""

    selected = []
    used = 0
    pairs = [turns[index:index + 2] for index in range(0, len(turns), 2)]
    for pair in reversed(pairs):
        cost = sum(
            (
                max(1, int(token_counter(turn.content)))
                if token_counter is not None else max(1, len(turn.content) // 4)
            ) + 4
            for turn in pair
        )
        if selected and used + cost > budget:
            break
        selected[:0] = pair
        used += cost
    return tuple(selected)


def encode_dialogue_user(content: str) -> str:
    return f"{USER_MARKER} {json.dumps(content, ensure_ascii=False)}"


def compose_dialogue_update(
    state_sections: tuple[tuple[str, str], ...],
    encoded_user_content: str,
    *,
    cleared: tuple[str, ...] = (),
) -> str:
    parts: list[str] = []
    if state_sections or cleared:
        parts.append(STATE_MARKER)
        parts.extend(f"{name}-" for name in cleared)
        parts.extend(text for _, text in state_sections)
    parts.append(encoded_user_content)
    return "\n".join(parts)


def _model_state_items(
    context: TurnContext,
    sections: tuple[tuple[str, str], ...],
) -> tuple[ModelStateItem, ...]:
    section_map = dict(sections)
    items: list[ModelStateItem] = []
    persistent_sections = (
        (
            "self",
            context.state.self_items,
            lambda item: f"{item.revision_count}:{item.updated_at:.6f}:{item.status}",
            lambda item: f"S- {item.kind} {item.topic}",
        ),
        (
            "behavioral_tendency",
            context.state.behavioral_tendencies,
            lambda item: f"{item.revision_count}:{item.updated_at:.6f}:{item.status}",
            lambda item: f"B- {item.context} {item.behavior}",
        ),
        (
            "strategy",
            context.state.strategies,
            lambda item: f"{item.revision_count}:{item.updated_at:.6f}:{item.status}",
            lambda item: f"R- {item.context} {item.procedure}",
        ),
        (
            "curiosity",
            context.state.curiosities,
            lambda item: f"{item.updated_at:.6f}:{item.status}",
            lambda item: f"A- {item.topic} {item.focus}",
        ),
        (
            "developmental_goal",
            context.state.developmental_goals,
            lambda item: f"{item.updated_at:.6f}:{item.status}",
            lambda item: f"G- {item.topic} {item.goal}",
        ),
        (
            "memory",
            context.state.memories,
            lambda item: f"{item.updated_at:.6f}",
            lambda item: f"M- {item.text}",
        ),
        (
            "experience",
            context.state.experiences,
            lambda item: f"{item.created_at:.6f}",
            lambda item: f"E- {item.kind} {item.topic}",
        ),
    )
    for section, state_items, version, clear_wire in persistent_sections:
        for item, wire in zip(
            state_items, section_map.get(section, "").splitlines(),
        ):
            items.append(ModelStateItem(
                key=f"{section}:{item.id}",
                section=section,
                wire=wire,
                version=version(item),
                clear_wire=clear_wire(item),
                persistent=True,
            ))
    persistent_section_names = {section for section, *_ in persistent_sections}
    for section, wire in sections:
        if section in persistent_section_names:
            continue
        items.append(ModelStateItem(
            key=section,
            section=section,
            wire=wire,
            version=hashlib.sha256(wire.encode("utf-8")).hexdigest(),
            clear_wire="",
            persistent=section not in TRANSIENT_STATE_SECTIONS,
        ))
    return tuple(items)


def build_dialogue_prompt(
    context: TurnContext,
    *,
    user_message: str,
    reply_context: str = "",
    character: Character | None = None,
    recent_limit: int | None = None,
    recent_token_counter=None,
    append_only: bool = False,
) -> PromptPlan:
    character = character or load_character_profile()
    recent = context.state.recent_turns[-recent_limit:] if recent_limit else context.state.recent_turns
    recent = _recent_turns_within_budget(
        recent, SETTINGS.recent_conversation_budget, recent_token_counter,
    )
    context_sections = format_context_sections(context, compact=True)
    state_items = _model_state_items(context, context_sections)
    history_messages = tuple(
        {"role": turn.role, "content": turn.content} for turn in recent
    )
    canonical_user_content = user_message.strip()
    current = canonical_user_content
    if reply_context:
        current = f"Reply context: {reply_context.strip()}\nCurrent message: {current}"
    turn_user_content = encode_dialogue_user(current)
    current = compose_dialogue_update(context_sections, turn_user_content)
    messages: list[dict[str, str]] = [{"role": "user", "content": current}]
    if not append_only:
        messages = [
            {"role": "system", "content": stable_system_prompt(character)},
            *history_messages,
            *messages,
        ]
    token_sections = {
        "identity": character.identity,
        "soul": character.voice,
        "stable_rules": _STABLE_RULES,
        "state_wrapper": f"{STATE_MARKER}\n{USER_MARKER}" if context_sections else USER_MARKER,
        **dict(context_sections),
        "recent_dialogue": "" if append_only else "\n".join(turn.content for turn in recent),
        "current_message": canonical_user_content,
    }
    if reply_context:
        token_sections["reply_context"] = reply_context.strip()
    return PromptPlan(
        messages=tuple(messages),
        selected_counts={
            "recent_turns": len(recent),
            "self_items": len(context.state.self_items),
            "memories": len(context.state.memories),
            "experiences": len(context.state.experiences),
            "behavioral_tendencies": len(context.state.behavioral_tendencies),
            "strategies": len(context.state.strategies),
            "curiosities": len(context.state.curiosities),
            "developmental_goals": len(context.state.developmental_goals),
        },
        token_sections=token_sections,
        static_prefix_hash=stable_prompt_hash(character),
        history_messages=history_messages,
        canonical_user_content=canonical_user_content,
        turn_user_content=turn_user_content,
        state_revision=context.state.revision,
        state_sections=context_sections,
        state_items=state_items,
        canonical_complete=not append_only,
    )
