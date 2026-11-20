"""Original-range coverage, aggregate model prose, and immutable segment records."""
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from ..contracts import ConversationSession, GenerationInfo, HistoryEntry
from .compaction_contracts import CompactionLimits, SummarySource

SEGMENT_LABEL = 'Model-generated segment summary; aggregate source provenance, not verified per-claim citations or user memory.'


@dataclass(frozen=True, slots=True)
class SegmentCapture:
    session: ConversationSession
    start: int  # Zero-based inclusive exchange ordinal.
    end: int    # Exclusive, before any exact-prompt admission shortening.
    entries: tuple[HistoryEntry, ...]
    digest: str
    frontier_id: UUID | None
    base_checkpoint_id: UUID | None
    base_checkpoint_digest: str | None

    @property
    def through_turn_id(self):
        return self.entries[-2].turn_id


@dataclass(frozen=True, slots=True)
class SegmentCandidate:
    text: str
    exchanges: int
    generated_at: datetime
    model: str
    template_sha256: str | None
    source_tokens: int
    summary_tokens: int
    generation: GenerationInfo


@dataclass(frozen=True, slots=True)
class SummarySegment:
    segment_id: UUID
    start: int
    end: int
    source_revision: int
    covered_turn_ids: tuple[UUID, ...]
    source_digest: str
    sources: tuple[SummarySource, ...]
    limits: CompactionLimits
    candidate: SegmentCandidate


@dataclass(frozen=True, slots=True)
class SegmentManifest:
    profile_id: UUID
    session_id: UUID
    session_created_at: datetime
    base_checkpoint_id: UUID | None
    base_checkpoint_digest: str | None
    base_exchanges: int
    segments: tuple[SummarySegment, ...] = ()

    @property
    def frontier(self):
        return self.segments[-1].end if self.segments else self.base_exchanges

    @property
    def frontier_id(self):
        return self.segments[-1].segment_id if self.segments else None
