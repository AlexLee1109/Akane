"""Bounded automatic maintenance policy and immutable diagnostic observations."""
from dataclasses import dataclass, field
from math import isfinite
from uuid import UUID

from .continuity.compaction_contracts import CompactionLimits


@dataclass(frozen=True, slots=True)
class AutomaticContextPolicy:
    high_water: float = .78
    low_water: float = .60
    headroom_tokens: int = 128
    future_input_tokens: int = 128
    memory_tokens: int = 256
    max_memories: int = 4
    protected_exchanges: int = 2
    retry_advance_exchanges: int = 2
    segment_exchanges: int | None = None  # Legacy library policy; durable terminal opts into four.
    max_summary_segments: int = 4
    summary_tokens: int = 768
    compaction: CompactionLimits = field(default_factory=CompactionLimits)
    periodic_interval_seconds: float | None = 300.0

    def __post_init__(self):
        if not (isfinite(self.low_water) and isfinite(self.high_water)
                and 0 < self.low_water < self.high_water < 1):
            raise ValueError('Pressure thresholds must satisfy 0 < low < high < 1.')
        if any(type(value) is not int or value < 1 for value in (
            self.headroom_tokens, self.future_input_tokens, self.memory_tokens,
            self.max_memories, self.protected_exchanges, self.retry_advance_exchanges,
            self.max_summary_segments, self.summary_tokens,
        )):
            raise ValueError('Automatic context bounds must be positive integers.')
        if self.segment_exchanges is not None and (type(self.segment_exchanges) is not int or not 1 <= self.segment_exchanges <= 8):
            raise ValueError('Segment bound must be between one and eight exchanges.')
        interval = self.periodic_interval_seconds
        if interval is not None and (type(interval) not in (int, float)
                                     or not isfinite(interval) or interval <= 0):
            raise ValueError('Periodic interval must be finite and positive or disabled with None.')


@dataclass(frozen=True, slots=True)
class AutomaticPromptInfo:
    requested_prompt_tokens: int
    prompt_tokens: int
    output_tokens: int
    memory_tokens: int
    included_memories: tuple[str, ...]
    excluded_memories: tuple[str, ...]
    fallback: bool
    summary_used: bool
    retained_turn_ids: tuple[UUID, ...]
    prepare_seconds: float
    error: str | None = None
    summary_tokens: int = 0
    included_segments: tuple[str, ...] = ()
    omitted_segments: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MaintenanceStatus:
    session_id: UUID
    state: str = 'idle'
    job_id: UUID | None = None
    source_revision: int | None = None
    boundary_exchanges: int = 0
    attempts: int = 0
    pressure: float | None = None
    error: str | None = None
    trigger: str | None = None
