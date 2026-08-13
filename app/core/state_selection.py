"""Central policy for state exposed to Akane's normal conversation prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.capabilities import (
    CAPABILITY_REGISTRY,
    CapabilityRuntime,
)
from app.core.memory import (
    ImprovementTarget,
    SelfModelItem,
    StateSnapshot,
    Strategy,
    communication_directives,
    effective_emotion,
    format_emotional_context,
    lightweight_relevance_score,
)
from app.core.presence import format_presence_prompt_context
from app.core.prompt import PromptContext
from app.core.time_context import build_time_context, format_time_context
from app.core.utils import words
from app.integrations.vscode_context import CodeContext


SECTION_LIMITS = {
    "memories_total": 3,
    "memories_per_owner": 2,
    "opinions": 2,
    "preferences": 2,
    "interests": 2,
    "relationship": 2,
    "self_model": 2,
    "strategies": 1,
    "capabilities": 3,
    "communication_preferences": 8,
}

_PRESENCE_QUESTION = re.compile(
    r"\b(?:what\s+(?:are|were)\s+you\s+(?:doing|up\s+to|thinking\s+about)|"
    r"what\s+(?:have|had)\s+you\s+been\s+(?:doing|up\s+to|thinking\s+about)|"
    r"what\s+did\s+you\s+do|"
    r"what(?:['’](?:re|ve))\s+you\s+(?:doing|been\s+doing|up\s+to)|"
    r"what\s+you\s+(?:doing|up\s+to)|what['’]s\s+been\s+occupying\s+your\s+attention|"
    r"what['’]s\s+on\s+your\s+mind|how\s+(?:have|had)\s+you\s+been\s+spending\s+your\s+time|"
    r"how\s+long\s+have\s+you\s+been\s+doing\s+that|"
    r"(?:are|were)\s+you\s+(?:busy|doing\s+anything)|"
    r"(?:have|had)\s+you\s+been\s+up\s+to\s+anything|doing\s+anything)\b",
    re.IGNORECASE,
)
_EMOTION_QUESTION = re.compile(
    r"\b(?:how are you|how do you feel|what are you feeling|your mood|"
    r"are you (?:okay|upset|angry|sad|happy|excited|worried))\b",
    re.IGNORECASE,
)
_RELATIONSHIP_QUESTION = re.compile(
    r"\b(?:our relationship|between us|how well do you know me|how do you see me|"
    r"what are we|our history|we(?:['’]ve| have) been through|remember us)\b",
    re.IGNORECASE,
)
_HISTORICAL_OPINION_QUERY = re.compile(
    r"\b(?:used to|before|previously|changed your mind|change over time|why did you change|history)\b",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^\s*(?:hi|hello|hey|yo|good (?:morning|afternoon|evening)|howdy)[!.?\s]*$",
    re.IGNORECASE,
)
_CODE_QUERY = re.compile(
    r"\b(?:code|file|function|class|method|bug|error|diagnostic|editor|repository|"
    r"workspace|selection|cursor|python|javascript|typescript|css|html|git)\b",
    re.IGNORECASE,
)
_PRESENCE_RELEVANCE_THRESHOLD = 0.42
_EMOTION_RELEVANCE_THRESHOLD = 0.26


@dataclass(frozen=True, slots=True)
class StateSelection:
    context: PromptContext
    debug: dict[str, object]


def _reasoned_count(
    *,
    selected: int,
    available: int,
    reason: str,
    ids: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "selected": selected,
        "omitted": max(0, available - selected),
        "reason": reason,
        "ids": ids,
    }


def _presence_context(
    snapshot: StateSnapshot,
    query: str,
) -> tuple[str, str]:
    direct = bool(_PRESENCE_QUESTION.search(query))
    current = snapshot.profile.presence.current_activity
    if direct:
        return (
            format_presence_prompt_context(
                snapshot.profile.presence,
                now=snapshot.now,
                include_previous=True,
            ),
            "direct_query",
        )
    if current is None or current.kind in {"quiet", "legacy"} or not current.subject:
        return "", "quiet_or_unavailable"
    score = lightweight_relevance_score(
        query,
        current.subject,
        now=snapshot.now,
        updated_at=current.started_at,
        confidence=current.grounding_confidence,
    )
    if score < _PRESENCE_RELEVANCE_THRESHOLD:
        return "", "unrelated"
    return (
        format_presence_prompt_context(snapshot.profile.presence, now=snapshot.now),
        "topic_match",
    )


def _emotion_context(
    snapshot: StateSnapshot,
    query: str,
    presence_context: str,
) -> tuple[str, str]:
    formatted = format_emotional_context(snapshot.profile, now=snapshot.now)
    if not formatted:
        return "", "neutral_or_expired"
    emotion = effective_emotion(snapshot.profile.emotion, now=snapshot.now)
    if _EMOTION_QUESTION.search(query):
        return formatted, "direct_query"
    current = snapshot.profile.presence.current_activity
    if (
        presence_context
        and current is not None
        and emotion.source_id == current.activity_id
    ):
        return formatted, "current_presence"
    cause_score = lightweight_relevance_score(
        query,
        emotion.cause,
        now=snapshot.now,
        updated_at=emotion.updated_at,
        confidence=emotion.intensity,
    )
    if words(query) & words(emotion.cause) and cause_score >= _EMOTION_RELEVANCE_THRESHOLD:
        return formatted, "topic_match"
    if emotion.intensity >= 0.35 and not _GREETING.fullmatch(query):
        return formatted, "current_authoritative_state"
    return "", "not_material_to_turn"


def _memory_conflicts_with_current_opinion(memory: object, opinions: tuple[object, ...]) -> bool:
    content = getattr(memory, "content", "")
    if _HISTORICAL_OPINION_QUERY.search(content):
        return False
    memory_terms = words(content)
    for opinion in opinions:
        topic_terms = words(getattr(opinion, "topic", ""))
        if topic_terms and len(memory_terms & topic_terms) / len(topic_terms) >= 0.6:
            return True
    return False


def _deduplicate_opinions(opinions: tuple[object, ...]) -> tuple[object, ...]:
    selected: list[object] = []
    for opinion in opinions:
        topic_terms = words(getattr(opinion, "topic", ""))
        if any(
            topic_terms
            and len(topic_terms & words(getattr(existing, "topic", "")))
            / max(1, len(topic_terms | words(getattr(existing, "topic", ""))))
            >= 0.72
            for existing in selected
        ):
            continue
        selected.append(opinion)
        if len(selected) >= SECTION_LIMITS["opinions"]:
            break
    return tuple(selected)


def select_relevant_state(
    snapshot: StateSnapshot,
    *,
    user_message: str,
    conversation_context: str = "",
    source: str = "popup",
    editor: CodeContext = CodeContext(False, False),
) -> StateSelection:
    """Select and format the only persistent/runtime state exposed this turn."""

    query = " ".join(
        value.strip() for value in (user_message, conversation_context) if value.strip()
    )
    profile = snapshot.profile
    recent_turns = tuple(snapshot.recent_turns)
    initiative = snapshot.last_profile_initiative
    if (
        initiative is not None
        and all(turn.turn_id != initiative.turn_id for turn in recent_turns)
        and (not recent_turns or initiative.timestamp >= recent_turns[-1].timestamp)
    ):
        recent_turns = (*recent_turns, initiative)

    opinions = _deduplicate_opinions(
        tuple(getattr(snapshot, "relevant_opinions", ()))
    )
    preferences = tuple(getattr(snapshot, "relevant_preferences", ()))[:
        SECTION_LIMITS["preferences"]
    ]
    interests = tuple(getattr(snapshot, "relevant_interests", ()))[:
        SECTION_LIMITS["interests"]
    ]
    relationship_items = tuple(getattr(snapshot, "relevant_relationship", ()))[:
        SECTION_LIMITS["relationship"]
    ]

    recent_pair_ids = {
        turn.turn_id.rsplit(":", 1)[0]
        for turn in recent_turns[-8:]
        if turn.turn_id.endswith((":user", ":assistant"))
    }
    memories = tuple(
        memory
        for memory in snapshot.relevant_memories
        if memory.source_id not in recent_pair_ids
        and (
            _HISTORICAL_OPINION_QUERY.search(query)
            or not _memory_conflicts_with_current_opinion(memory, opinions)
        )
    )[: SECTION_LIMITS["memories_total"]]
    memory_by_owner = {
        owner: tuple(memory for memory in memories if memory.subject == owner)[
            : SECTION_LIMITS["memories_per_owner"]
        ]
        for owner in ("user", "akane", "shared")
    }

    self_model_candidates = tuple(getattr(snapshot, "relevant_self_model", ()))
    self_model = tuple(
        item
        for item in self_model_candidates
        if not isinstance(item, SelfModelItem) or item.category != "capability"
    )[: SECTION_LIMITS["self_model"]]
    strategies = tuple(
        item
        for item in getattr(snapshot, "relevant_strategies", ())
        if isinstance(item, Strategy) and item.status == "active"
    )[: SECTION_LIMITS["strategies"]]

    capability_runtime = CapabilityRuntime(
        profile_id=snapshot.profile_id,
        source=source,
        editor_connected=editor.connected,
    )
    capabilities = CAPABILITY_REGISTRY.relevant(
        user_message,
        capability_runtime,
        limit=SECTION_LIMITS["capabilities"],
    )
    presence, presence_reason = _presence_context(snapshot, query)
    emotion, emotion_reason = _emotion_context(snapshot, query, presence)
    relationship = tuple(item.content for item in relationship_items)
    if snapshot.familiarity and (
        relationship_items or _RELATIONSHIP_QUESTION.search(query)
    ):
        relationship = (snapshot.familiarity, *relationship)
    relationship = relationship[: SECTION_LIMITS["relationship"]]

    tool_context = (
        editor.prompt_text
        if editor.connected and (_CODE_QUERY.search(query) or source == "vscode")
        else ""
    )
    current = profile.presence.current_activity
    context = PromptContext(
        time_context=format_time_context(
            build_time_context(
                now=snapshot.now,
                last_user_message_at=snapshot.last_profile_user_at,
                last_akane_message_at=snapshot.last_profile_assistant_at,
                current_activity_started_at=(
                    current.started_at if current is not None and presence else None
                ),
            )
        ),
        recent_turns=recent_turns,
        user_memories=tuple(
            f"[id={memory.id}; owner=user; kind={memory.kind}; source={memory.source_type}] "
            f"{memory.content}"
            for memory in memory_by_owner["user"]
        ),
        akane_memories=tuple(
            f"[id={memory.id}; owner=akane; kind={memory.kind}; source={memory.source_type}] "
            f"{memory.content}"
            for memory in memory_by_owner["akane"]
        ),
        shared_memories=tuple(
            f"[id={memory.id}; owner=shared; kind={memory.kind}; source={memory.source_type}] "
            f"{memory.content}"
            for memory in memory_by_owner["shared"]
        ),
        preferences=tuple(item.content for item in preferences),
        opinions=tuple(
            f"- {item.topic} ({item.domain}): {item.position} "
            f"Reason: {item.reason} Confidence: {item.confidence:.2f}. "
            f"[target_id={item.id}]"
            for item in opinions
        ),
        self_model=tuple(
            (
                f"- improvement ({item.area}): {item.description} "
                f"Reason: {item.reason}. [target_id={item.id}]"
                if isinstance(item, ImprovementTarget)
                else f"- {item.category} ({item.area}): {item.description} "
                f"[target_id={item.id}]"
            )
            for item in self_model
            if isinstance(item, (SelfModelItem, ImprovementTarget))
        ),
        runtime_capabilities=tuple(
            f"- {fact.key}={'true' if fact.available else 'false'} — {fact.description}"
            for fact in capabilities
        ),
        active_strategies=tuple(f"- {item.description}" for item in strategies),
        interests=tuple(str(item) for item in interests),
        communication_preferences=communication_directives(profile)[
            : SECTION_LIMITS["communication_preferences"]
        ],
        relationship=relationship,
        emotion=emotion,
        presence=presence,
        reply_context=conversation_context,
        tool_context=tool_context,
    )

    selected_memory_ids = tuple(
        memory.id for owner in ("user", "akane", "shared") for memory in memory_by_owner[owner]
    )
    debug = {
        "memory": _reasoned_count(
            selected=len(selected_memory_ids),
            available=len(profile.memories),
            reason="topic_match" if selected_memory_ids else "no_relevant_match",
            ids=selected_memory_ids,
        ),
        "opinions": _reasoned_count(
            selected=len(opinions),
            available=len(profile.opinions),
            reason="current_authoritative_state" if opinions else "no_relevant_match",
            ids=tuple(item.id for item in opinions),
        ),
        "preferences": _reasoned_count(
            selected=len(preferences),
            available=len(profile.preferences),
            reason="topic_match" if preferences else "no_relevant_match",
        ),
        "interests": _reasoned_count(
            selected=len(interests),
            available=len(profile.interests),
            reason="topic_match" if interests else "no_relevant_match",
        ),
        "relationship": _reasoned_count(
            selected=len(relationship_items),
            available=(
                len(profile.relationship.patterns)
                + len(profile.relationship.shared_context)
                + len(profile.relationship.unresolved_events)
            ),
            reason="topic_match" if relationship else "unrelated",
        ),
        "self_model": _reasoned_count(
            selected=len(self_model),
            available=len(snapshot.self_model.items) + len(snapshot.self_model.improvement_targets),
            reason="direct_or_topic_match" if self_model else "unrelated",
            ids=tuple(item.id for item in self_model),
        ),
        "strategies": _reasoned_count(
            selected=len(strategies),
            available=len(snapshot.strategies),
            reason="applies_to_turn" if strategies else "does_not_apply",
            ids=tuple(item.id for item in strategies),
        ),
        "presence": {
            "exposed": bool(presence),
            "reason": presence_reason,
        },
        "emotion": {
            "exposed": bool(emotion),
            "reason": emotion_reason,
        },
        "capabilities": _reasoned_count(
            selected=len(capabilities),
            available=CAPABILITY_REGISTRY.count,
            reason=(
                capabilities[0].reason
                if capabilities
                else "not_a_capability_query"
            ),
            ids=tuple(fact.key for fact in capabilities),
        ),
        "communication_preferences": _reasoned_count(
            selected=len(context.communication_preferences),
            available=len(profile.communication_preferences),
            reason="applies_to_every_turn" if context.communication_preferences else "none",
        ),
        "tool_context": {
            "exposed": bool(tool_context),
            "reason": (
                "topic_match"
                if tool_context
                else "disconnected"
                if not editor.connected
                else "unrelated"
            ),
        },
        "section_limits": dict(SECTION_LIMITS),
    }
    return StateSelection(context, debug)
