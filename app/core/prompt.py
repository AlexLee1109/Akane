"""Pure prompt compilation from an already-selected context snapshot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

from app.core.character import Character, load_character_profile
from app.core.config import SETTINGS
from app.core.context import TurnContext, format_context_sections
from app.core.situation import SITUATION_RULES
from app.core.retrieval import RETRIEVAL_RULES
from app.core.streaming import SEMANTIC_SIDECAR_END, SEMANTIC_SIDECAR_START


_STABLE_RULES = f"""A=Akane;U=user;STATE=context;no labels/reasoning.
When asked for your view or choice, form a present judgment naturally, even with empty Self. This is a present stance, not proof of past experience.
STEP 1 — choose domain:
WORLD first: the CURRENT U turn directly establishes external entities/state/events/relations. U's direct report takes priority over A's reaction; no sensory/physical/tool access needed.
SELF otherwise: A expresses a meaningful preference/opinion/interest/etc., or eligible Development evidence. d=c is a weak candidate; no prior history needed.
NONE: neither domain has eligible evidence. Recall questions, hypotheticals/imagined statements and neutral greetings: n. They are not new World evidence. Never replay old evidence.
STEP 2 — if WORLD, choose kind:
ws: one entity has a mutable current state/property; d="na".
wr: two distinct entities relate/link/connect/belong/point to each other; use wr, NOT wf; d="na".
wf: durable proposition about one entity, NOT an entity-to-entity relation; d="da".
wv: newly reported historical occurrence; k,t,a,d="pa"; no v.
we: entity existence; k,t,d="na". wr-: remove relation; k,t,a,v,d="nn".
STEP 3 — emit exactly one existing shape; never mix Self and World fields or invent kinds/scope codes.
SELF p/o/i/g=preference/opinion/interest/goal: k,t,s,d. NEVER e or a; v only for comparative s=cmp.
{{"k":"p","t":"topic","s":"+","d":"c"}}
WORLD STATE / FACT / RELATION: k,t,a,v,d. NEVER e or s for current reports.
{{"k":"ws","t":"entity","a":"attribute","v":"value","d":"na"}}
{{"k":"wr","t":"subject","a":"relation","v":"target","d":"na"}}
NONE: {{"k":"n"}}
Substitute actual subjects/values. Runtime binds Self to A's complete visible reply and direct World to the complete current U turn, with their source IDs. Do not emit alternate sources. World v must occur in U; reuse entity labels/IDs and attribute keys. Uncertainty, third-party reports and generated claims are not World evidence. Ambiguity means n; never infer opposite values from negation. ws records transition history. wr:b=old target replaces an edge. Only prior-source wf/wv add z=existing Memory/Experience ID and e=its complete original U quote.
Existing Development witness grammar (separate from current Self/World):
k: c=correction;f+/f-=feedback;t+/t-=task result;x+/x-=prediction;u+/u-=open/resolved curiosity;j+/j-=goal commit/drop. Required k,t,s,d,e. s=+/-/0/cmp;d=c/tmp/hyp/task/n. Events:e=U,a=prior act,b/f=behavior/effect,r=procedure,j=goal;generated event claims alone are not evidence. Predict:d=task,e/a=expect/action,q=h/l. Curiosity:w=focus,e=A/U for u+/u-;not a question mandate. Goal:j=intention,e=A;context-only,no autonomous action. Omit unused fields.
After one short visible sentence, append {SEMANTIC_SIDECAR_START}JSON{SEMANTIC_SIDECAR_END}, replacing JSON with the chosen object. Always emit the literal {SEMANTIC_SIDECAR_END}, including NONE; nothing after it."""

STATE_MARKER = "STATE"
USER_MARKER = "U"
TRANSIENT_STATE_SECTIONS = frozenset({"time", "code_context", "situation", "world"})


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
        character.content_sha256, character.identity, character.voice, _STABLE_RULES + "\n" + SITUATION_RULES + "\n" + RETRIEVAL_RULES,
    )[0]


def stable_prompt_hash(character: Character | None = None) -> str:
    character = character or load_character_profile()
    return _stable_prompt_parts(
        character.content_sha256, character.identity, character.voice, _STABLE_RULES + "\n" + SITUATION_RULES + "\n" + RETRIEVAL_RULES,
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
        "stable_rules": _STABLE_RULES + "\n" + SITUATION_RULES + "\n" + RETRIEVAL_RULES,
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
