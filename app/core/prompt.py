"""Role-correct prompt construction and exact chat-template tokenization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from app.core.config import LLAMA_CONTEXT_WINDOW, MAX_TOKENS
from app.core.memory import ChatTurn

PROMPT_BUILDER_VERSION = "3"


class PromptAuthority(str, Enum):
    STABLE_POLICY = "stable_policy"
    DYNAMIC_GUIDANCE = "dynamic_guidance"
    HISTORICAL_EVIDENCE = "historical_evidence"
    REFERENCE_CONTEXT = "reference_context"
    CURRENT_INPUT = "current_input"


@dataclass(frozen=True, slots=True)
class PromptSource:
    kind: str
    content: str
    role: str
    authority: PromptAuthority
    required: bool
    trim_group: int
    origin: str
    pair_index: int = -1


@dataclass(frozen=True, slots=True)
class PromptTokenCount:
    tokens: tuple[int, ...]
    method: str
    stop_sequences: tuple[str, ...] = ()

    @property
    def exact(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class TrimmedPromptSource:
    kind: str
    origin: str
    reason: str


@dataclass(frozen=True, slots=True)
class PromptContext:
    behavioral_summary: str = ""
    relationship: str = ""
    preference_continuity: str = ""
    relevant_memories: str = ""
    durable_memories: tuple[str, ...] = ()
    earlier_turns: tuple[ChatTurn, ...] = ()
    recent_turns: tuple[ChatTurn, ...] = ()
    life_context: str = ""
    date_time: str = ""
    reply_context: str = ""
    external_context: str = ""


@dataclass(frozen=True, slots=True)
class PromptPlan:
    messages: list[dict[str, str]]
    sources: tuple[PromptSource, ...]
    trimmed_sources: tuple[TrimmedPromptSource, ...]
    persona_versions: tuple[tuple[str, str], ...]
    token_ids: tuple[int, ...] = ()
    counting_method: str = "not_tokenized"
    stop_sequences: tuple[str, ...] = ()
    reserved_output_tokens: int = MAX_TOKENS
    context_window: int = LLAMA_CONTEXT_WINDOW

    @property
    def rendered_prompt_tokens(self) -> int | None:
        return len(self.token_ids) if self.token_ids else None

    @property
    def token_count_is_exact(self) -> bool:
        return bool(self.token_ids)

    def debug_metadata(self) -> dict[str, object]:
        return {
            "prompt_builder_version": PROMPT_BUILDER_VERSION,
            "exact_tokens": self.rendered_prompt_tokens,
            "counting_method": self.counting_method,
            "reserved_output_tokens": self.reserved_output_tokens,
            "context_window": self.context_window,
            "message_count": len(self.messages),
            "sources": tuple(source.kind for source in self.sources),
            "persona_versions": dict(self.persona_versions),
            "trimmed": tuple(
                f"{item.kind}:{item.reason}" for item in self.trimmed_sources
            ),
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
    safe_transport = str(transport or "unknown").strip().lower() or "unknown"
    conversation_digest = hashlib.sha256(
        str(conversation_id or "").encode("utf-8")
    ).hexdigest()[:12]
    history = messages[1:-1] if len(messages) >= 2 else []
    return {
        "transport": safe_transport,
        "conversation_id": f"{safe_transport}:{conversation_digest}",
        "loaded_recent_turns": max(0, int(loaded_recent_turns)),
        "complete_pairs": sum(item.get("role") == "assistant" for item in history),
        "role_sequence": ",".join(item.get("role", "unknown") for item in messages),
        "summary_turns": max(0, int(summary_turns)),
        "current_user_occurrences": int(
            bool(messages)
            and messages[-1].get("role") == "user"
            and messages[-1].get("content") == current_user_text
        ),
        "generation_mode": str(generation_mode or "unknown"),
    }


def build_prompt_plan(
    user_text: str,
    context: PromptContext,
    *,
    token_counter: Callable[[list[dict[str, str]]], PromptTokenCount] | None = None,
) -> PromptPlan:
    """Build one system message, six native pairs, and the current user message."""

    from app.core.character import (
        get_hard_constraints_prompt,
        get_persona_versions,
        load_character_profile,
    )

    profile = load_character_profile()
    hard_text = get_hard_constraints_prompt()
    versions = tuple(get_persona_versions(profile, hard_text).items())
    sources = _prompt_sources(
        profile.identity,
        profile.soul,
        hard_text,
        str(user_text or ""),
        context,
    )
    selected = list(sources)
    trimmed: list[TrimmedPromptSource] = []
    messages = _render_messages(selected)
    token_ids: tuple[int, ...] = ()
    method = "not_tokenized"
    stop_sequences: tuple[str, ...] = ()

    if token_counter is not None:
        count = token_counter(messages)
        limit = LLAMA_CONTEXT_WINDOW - MAX_TOKENS
        while len(count.tokens) > limit:
            removed = _remove_next(selected)
            if not removed:
                raise RuntimeError(
                    "Akane's required persona and current message exceed the model context."
                )
            trimmed.extend(
                TrimmedPromptSource(item.kind, item.origin, "exact_context_pressure")
                for item in removed
            )
            messages = _render_messages(selected)
            count = token_counter(messages)
        token_ids = count.tokens
        method = count.method
        stop_sequences = count.stop_sequences

    return PromptPlan(
        messages=messages,
        sources=tuple(selected),
        trimmed_sources=tuple(trimmed),
        persona_versions=versions,
        token_ids=token_ids,
        counting_method=method,
        stop_sequences=stop_sequences,
    )


def _prompt_sources(
    identity: str,
    soul: str,
    hard_text: str,
    current_text: str,
    context: PromptContext,
) -> list[PromptSource]:
    sources = [
        PromptSource("identity", identity, "system", PromptAuthority.STABLE_POLICY, True, 99, "app/identity.md"),
        PromptSource("soul", soul, "system", PromptAuthority.STABLE_POLICY, True, 99, "app/soul.md"),
        PromptSource("hard_constraints", hard_text, "system", PromptAuthority.STABLE_POLICY, True, 99, "app/core/character.py"),
    ]
    if context.behavioral_summary:
        sources.append(
            PromptSource(
                "turn_guidance",
                context.behavioral_summary,
                "system",
                PromptAuthority.DYNAMIC_GUIDANCE,
                True,
                99,
                "InternalStateCoordinator",
            )
        )

    optional_details = (
        ("editor_context", context.external_context, "VSCodeSnapshot"),
        ("reply_context", context.reply_context, "ChatInput.reply_context"),
        ("relationship", context.relationship, "MemoryStore.relationship"),
        ("life_context", context.life_context, "LifeState"),
        ("date_time", context.date_time, "server_clock"),
        ("preference_continuity", context.preference_continuity, "LongTermMemoryStore.preference"),
    )
    for kind, content, origin in optional_details:
        if content:
            sources.append(
                PromptSource(kind, content, "system", PromptAuthority.REFERENCE_CONTEXT, False, 0, origin)
            )

    memories = context.durable_memories or ((context.relevant_memories,) if context.relevant_memories else ())
    for index, content in enumerate(memories):
        if content:
            sources.append(
                PromptSource(
                    "durable_memory",
                    content,
                    "system",
                    PromptAuthority.REFERENCE_CONTEXT,
                    False,
                    1,
                    f"LongTermMemoryStore:{index}",
                )
            )

    turns = _complete_recent_turns(context.recent_turns)[-12:]
    for index, turn in enumerate(turns):
        pair_index = index // 2
        sources.append(
            PromptSource(
                f"history_{turn.role}",
                turn.content,
                turn.role,
                PromptAuthority.HISTORICAL_EVIDENCE,
                False,
                2,
                f"MemoryStore.pair:{pair_index}",
                pair_index,
            )
        )
    sources.append(
        PromptSource("current_user", current_text, "user", PromptAuthority.CURRENT_INPUT, True, 99, "ChatInput.text")
    )
    return sources


def _complete_recent_turns(turns: tuple[ChatTurn, ...]) -> tuple[ChatTurn, ...]:
    complete: list[ChatTurn] = []
    index = 0
    while index + 1 < len(turns):
        if turns[index].role == "user" and turns[index + 1].role == "assistant":
            complete.extend((turns[index], turns[index + 1]))
            index += 2
        else:
            index += 1
    return tuple(complete)


def _remove_next(sources: list[PromptSource]) -> tuple[PromptSource, ...]:
    optional = next((item for item in sources if not item.required and item.trim_group == 0), None)
    if optional is not None:
        sources.remove(optional)
        return (optional,)

    memories = [item for item in sources if not item.required and item.trim_group == 1]
    if memories:
        removed = memories[-1]
        sources.remove(removed)
        return (removed,)

    pair_indexes = [item.pair_index for item in sources if item.trim_group == 2]
    if pair_indexes:
        oldest = min(pair_indexes)
        removed = tuple(item for item in sources if item.pair_index == oldest)
        sources[:] = [item for item in sources if item.pair_index != oldest]
        return removed
    return ()


def _render_messages(sources: list[PromptSource]) -> list[dict[str, str]]:
    system_parts = [
        source.content.strip()
        for source in sources
        if source.authority in {
            PromptAuthority.STABLE_POLICY,
            PromptAuthority.DYNAMIC_GUIDANCE,
            PromptAuthority.REFERENCE_CONTEXT,
        }
        and source.content.strip()
    ]
    messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
    messages.extend(
        {"role": source.role, "content": source.content}
        for source in sources
        if source.authority == PromptAuthority.HISTORICAL_EVIDENCE
    )
    current = next(source for source in sources if source.authority == PromptAuthority.CURRENT_INPUT)
    messages.append({"role": "user", "content": current.content})
    return messages
