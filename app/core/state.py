"""Canonical value objects shared by Akane's state domains."""

from __future__ import annotations

from dataclasses import dataclass, field

SELF_GROUP_BY_KIND = {
    "opinion": "opinions",
    "preference": "preferences",
    "interest": "interests",
    "curiosity": "curiosities",
    "goal": "goals",
    "tendency": "tendencies",
}
SELF_KINDS = frozenset(SELF_GROUP_BY_KIND)
SELF_STATUSES = frozenset({"active", "uncertain", "completed", "abandoned", "retired"})
MEMORY_SUBJECTS = frozenset({"user", "akane", "shared"})
MEMORY_KINDS = frozenset({"fact", "event", "commitment", "shared_experience"})
THOUGHT_STATUSES = frozenset({"active", "resolved", "expired"})


@dataclass(frozen=True, slots=True)
class Turn:
    id: str
    profile_id: str
    conversation_id: str
    role: str
    content: str
    created_at: float
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class Memory:
    id: str
    profile_id: str
    subject: str
    kind: str
    text: str
    importance: float
    confidence: float
    created_at: float
    updated_at: float
    source_turn_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SelfItem:
    id: str
    profile_id: str
    kind: str
    topic: str
    value: str
    strength: float
    confidence: float
    reason: str
    status: str
    created_at: float
    updated_at: float
    source_ids: tuple[str, ...] = ()
    revision_count: int = 0


@dataclass(frozen=True, slots=True)
class SelfRevision:
    id: str
    self_item_id: str
    profile_id: str
    value: str
    strength: float
    confidence: float
    reason: str
    status: str
    source_ids: tuple[str, ...]
    changed_at: float


@dataclass(frozen=True, slots=True)
class Mood:
    profile_id: str
    valence: float = 0.0
    energy: float = 0.0
    emotion: str = "calm"
    cause: str = ""
    updated_at: float = 0.0


@dataclass(frozen=True, slots=True)
class Relationship:
    profile_id: str
    familiarity: float = 0.0
    trust: float = 0.0
    closeness: float = 0.0
    interaction_notes: tuple[str, ...] = ()
    unresolved_events: tuple[str, ...] = ()
    updated_at: float = 0.0


@dataclass(frozen=True, slots=True)
class Thought:
    id: str
    profile_id: str
    topic: str
    text: str
    importance: float
    source_ids: tuple[str, ...]
    started_at: float
    updated_at: float
    status: str = "active"
    share_worthy: bool = False


@dataclass(frozen=True, slots=True)
class ProactiveMessage:
    id: str
    profile_id: str
    thought_id: str
    text: str
    importance: float
    created_at: float
    status: str = "pending"


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    profile_id: str
    conversation_id: str
    recent_turns: tuple[Turn, ...]
    memories: tuple[Memory, ...]
    self_items: tuple[SelfItem, ...]
    self_revisions: tuple[SelfRevision, ...]
    mood: Mood
    relationship: Relationship
    thoughts: tuple[Thought, ...]
    schema_version: int
    revision: int


@dataclass(frozen=True, slots=True)
class MemoryChange:
    action: str
    memory: Memory | None = None
    target_id: str = ""


@dataclass(frozen=True, slots=True)
class SelfChange:
    action: str
    item: SelfItem | None = None
    target_id: str = ""


@dataclass(frozen=True, slots=True)
class MoodChange:
    valence_delta: float = 0.0
    energy_delta: float = 0.0
    emotion: str = ""
    cause: str = ""


@dataclass(frozen=True, slots=True)
class RelationshipChange:
    familiarity_delta: float = 0.0
    trust_delta: float = 0.0
    closeness_delta: float = 0.0
    add_notes: tuple[str, ...] = ()
    resolve_notes: tuple[str, ...] = ()
    add_unresolved: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ThoughtChange:
    action: str
    thought: Thought | None = None
    target_id: str = ""


@dataclass(frozen=True, slots=True)
class StateChangeProposal:
    profile_id: str
    turns: tuple[Turn, ...] = ()
    memories: tuple[MemoryChange, ...] = ()
    self_items: tuple[SelfChange, ...] = ()
    mood: MoodChange | None = None
    relationship: RelationshipChange | None = None
    thoughts: tuple[ThoughtChange, ...] = ()
    proactive_messages: tuple[ProactiveMessage, ...] = ()
    reflection_turn_ids: tuple[str, str] | None = None
    # Content readiness is not temporal inference eligibility; Store.available_at
    # and InferenceRuntime's foreground-idle gate decide when work may start.
    reflection_ready: bool = True
    reflection_job_id: str = ""
    reflected_through_turn_id: str = ""
    reflected_turn_count: int = 0
    origin: str = "conversation"
    rejected: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CommitResult:
    revision: int
    applied: tuple[str, ...]
    rejected: tuple[str, ...]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
