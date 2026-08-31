"""Canonical value objects for Akane's durable schema-8 state."""

from __future__ import annotations

from dataclasses import dataclass, field


SELF_GROUP_BY_KIND = {
    "opinion": "opinions",
    "preference": "preferences",
    "interest": "interests",
    "goal": "goals",
}
SELF_KINDS = frozenset(SELF_GROUP_BY_KIND)
SELF_ACTIONS = ("form", "reinforce", "weaken", "revise", "retire", "complete", "abandon")
SELF_GOAL_TERMINAL_ACTIONS = frozenset({"complete", "abandon"})
SELF_STATUSES = frozenset({"active", "uncertain", "completed", "abandoned", "retired"})
MEMORY_SUBJECTS = frozenset({"user", "akane", "shared"})
MEMORY_KINDS = frozenset({"fact", "event", "commitment", "shared_experience"})
EXPERIENCE_SUBJECTS = frozenset({"akane", "shared"})
EXPERIENCE_KINDS = frozenset({
    "preference", "opinion", "interest", "goal", "mind_change", "correction",
    "positive_feedback", "negative_feedback", "task_success", "task_failure",
})


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
class Experience:
    id: str
    profile_id: str
    kind: str
    subject: str
    topic: str
    what_happened: str
    akane_response: str
    outcome: str
    salience: float
    reason: str
    created_at: float
    self_item_ids: tuple[str, ...] = ()
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
    contradiction_ids: tuple[str, ...] = ()
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
    contradiction_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    profile_id: str
    conversation_id: str
    recent_turns: tuple[Turn, ...]
    memories: tuple[Memory, ...]
    experiences: tuple[Experience, ...]
    self_items: tuple[SelfItem, ...]
    self_revisions: tuple[SelfRevision, ...]
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
class StateChangeProposal:
    profile_id: str
    turns: tuple[Turn, ...] = ()
    memories: tuple[MemoryChange, ...] = ()
    experiences: tuple[Experience, ...] = ()
    self_items: tuple[SelfChange, ...] = ()
    origin: str = "conversation"
    rejected: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CommitResult:
    revision: int
    applied: tuple[str, ...]
    rejected: tuple[str, ...]
    timings: dict[str, float | int] = field(default_factory=dict)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
