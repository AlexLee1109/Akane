"""Canonical value objects for Akane's durable schema-14 state."""

from __future__ import annotations

from dataclasses import dataclass, field


SELF_GROUP_BY_KIND = {
    "opinion": "opinions",
    "preference": "preferences",
    "interest": "interests",
    "goal": "goals",
}
SELF_KINDS = frozenset(SELF_GROUP_BY_KIND)
SELF_STATUSES = frozenset({"active", "uncertain", "completed", "abandoned", "retired"})
MEMORY_SUBJECTS = frozenset({"user", "akane", "shared"})
MEMORY_KINDS = frozenset({"fact", "event", "commitment", "shared_experience"})
EXPERIENCE_SUBJECTS = frozenset({"akane", "shared"})
EXPERIENCE_KINDS = frozenset({
    "preference", "opinion", "interest", "goal", "mind_change", "correction",
    "positive_feedback", "negative_feedback", "task_success", "task_failure",
    "unresolved_curiosity", "resolved_curiosity",
    "developmental_goal", "developmental_goal_release",
})
OUTCOME_RESULTS = frozenset({
    "correction", "positive_feedback", "negative_feedback",
    "task_success", "task_failure",
})
PREDICTION_RESULTS = frozenset({"success", "failure"})
PREDICTION_STATUSES = frozenset({"unresolved", "resolved", "expired"})
PREDICTION_ERROR_CATEGORIES = frozenset({
    "none", "negative_error", "positive_surprise",
})
PREDICTION_TTL_SECONDS = 7 * 24 * 60 * 60
UNRESOLVED_PREDICTION_LIMIT = 16
BEHAVIORAL_TENDENCY_STATUSES = frozenset({"active", "uncertain"})
STRATEGY_STATUSES = frozenset({"active", "uncertain", "retired"})
CURIOSITY_STATUSES = frozenset({"active", "uncertain", "resolved"})
DEVELOPMENTAL_GOAL_STATUSES = frozenset({
    "candidate", "active", "satisfied", "retired",
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
class Outcome:
    id: str
    profile_id: str
    result: str
    description: str
    action: str
    action_turn_id: str
    confidence: float
    reason: str
    created_at: float
    source_turn_ids: tuple[str, ...] = ()
    experience_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Prediction:
    id: str
    profile_id: str
    action: str
    action_turn_id: str
    expectation: str
    expected_result: str
    confidence: float
    status: str
    created_at: float
    expires_at: float
    outcome_id: str = ""
    actual_result: str = ""
    error_category: str = ""
    error_value: float = 0.0
    resolved_at: float = 0.0
    experience_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BehavioralTendency:
    id: str
    profile_id: str
    context: str
    behavior: str
    expected_effect: str
    expected_result: str
    strength: float
    confidence: float
    status: str
    created_at: float
    updated_at: float
    supporting_outcome_ids: tuple[str, ...] = ()
    contradiction_outcome_ids: tuple[str, ...] = ()
    revision_count: int = 0


@dataclass(frozen=True, slots=True)
class Strategy:
    id: str
    profile_id: str
    context: str
    procedure: str
    expected_result: str
    strength: float
    confidence: float
    status: str
    created_at: float
    updated_at: float
    supporting_outcome_ids: tuple[str, ...] = ()
    contradiction_outcome_ids: tuple[str, ...] = ()
    revision_count: int = 0


@dataclass(frozen=True, slots=True)
class Curiosity:
    id: str
    profile_id: str
    topic: str
    focus: str
    strength: float
    confidence: float
    status: str
    created_at: float
    updated_at: float
    source_ids: tuple[str, ...] = ()
    resolution_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DevelopmentalGoal:
    id: str
    profile_id: str
    topic: str
    goal: str
    strength: float
    confidence: float
    status: str
    created_at: float
    updated_at: float
    source_ids: tuple[str, ...] = ()
    progress_outcome_ids: tuple[str, ...] = ()
    contradiction_ids: tuple[str, ...] = ()


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
    behavioral_tendencies: tuple[BehavioralTendency, ...]
    strategies: tuple[Strategy, ...]
    curiosities: tuple[Curiosity, ...]
    developmental_goals: tuple[DevelopmentalGoal, ...]
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
class PredictionChange:
    action: str
    prediction: Prediction
    target_id: str = ""


@dataclass(frozen=True, slots=True)
class BehavioralTendencyChange:
    action: str
    tendency: BehavioralTendency
    target_id: str = ""


@dataclass(frozen=True, slots=True)
class StrategyChange:
    action: str
    strategy: Strategy
    target_id: str = ""


@dataclass(frozen=True, slots=True)
class CuriosityChange:
    action: str
    curiosity: Curiosity
    target_id: str = ""


@dataclass(frozen=True, slots=True)
class DevelopmentalGoalChange:
    action: str
    goal: DevelopmentalGoal
    target_id: str = ""


@dataclass(frozen=True, slots=True)
class StateChangeProposal:
    profile_id: str
    turns: tuple[Turn, ...] = ()
    memories: tuple[MemoryChange, ...] = ()
    experiences: tuple[Experience, ...] = ()
    outcomes: tuple[Outcome, ...] = ()
    predictions: tuple[PredictionChange, ...] = ()
    behavioral_tendencies: tuple[BehavioralTendencyChange, ...] = ()
    strategies: tuple[StrategyChange, ...] = ()
    curiosities: tuple[CuriosityChange, ...] = ()
    developmental_goals: tuple[DevelopmentalGoalChange, ...] = ()
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
