"""Immutable, compact L3-to-L1 data and host-only source mappings."""
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from .contracts import HistoryEntry
from .continuity.compaction_contracts import SummaryEntry
from .provenance import SourceAttribution


@dataclass(frozen=True, slots=True)
class ProjectedMemory:
    reference: str
    quote: str
    correction: bool


@dataclass(frozen=True, slots=True)
class ContextReference:
    alias: str
    session_id: UUID
    turn_id: UUID
    role: Literal['user', 'assistant']
    source: SourceAttribution
    outcome: str = "completed"


@dataclass(frozen=True, slots=True)
class ProjectedSegment:
    reference: str
    text: str | None
    source: SourceAttribution
    entries: tuple[SummaryEntry, ...] = ()
    incomplete: bool = False


@dataclass(frozen=True, slots=True)
class L3Projection:
    profile_id: UUID
    session_id: UUID
    session_created_at: datetime
    revision: int
    summary_id: UUID | None
    summary: tuple[SummaryEntry, ...]
    covered_turn_ids: tuple[UUID, ...]
    memories: tuple[ProjectedMemory, ...]
    references: tuple[ContextReference, ...]
    baseline_dialogue: tuple[HistoryEntry, ...]
    automatic: bool = False
    memory_token_budget: int = 256
    protected_exchanges: int = 2
    incomplete_turns: tuple[tuple[UUID, str], ...] = ()
    segment_mode: bool = False
    segments: tuple[ProjectedSegment, ...] = ()
    max_summary_segments: int = 4
    summary_token_budget: int = 768


@dataclass(frozen=True, slots=True)
class ContextTokenBudget:
    # All counterfactuals use the same identity/current input/output reserve and
    # L2 carrier. Baseline uses original dialogue without L3 audit metadata.
    baseline_tokens: int
    summary_augmented_tokens: int
    memory_augmented_tokens: int
    combined_tokens: int
    summary_contribution: int
    memory_contribution: int
    combined_memory_contribution: int
    summary_replacement_savings: int
    net_replacement_savings: int
    l2_contribution: int
    output_reservation: int
    context_size: int


@dataclass(frozen=True, slots=True)
class ContextSelectionStatus:
    profile_id: UUID
    session_id: UUID
    summary_enabled: bool
    memory_ids: tuple[UUID, ...]
    last_summary_state: Literal['disabled', 'missing', 'selected'] | None
    last_token_budget: ContextTokenBudget | None
