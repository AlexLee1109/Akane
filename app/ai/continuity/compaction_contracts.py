"""Immutable compaction data, separate from dialogue and approved user memory."""
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Literal
from uuid import UUID

from ..contracts import ConversationSession, GenerationInfo, HistoryEntry
from ..provenance import SourceAttribution


@dataclass(frozen=True, slots=True)
class CompactionLimits:
    keep_recent_exchanges: int = 2
    max_prompt_tokens: int = 2048
    max_output_tokens: int = 384
    max_characters: int = 4096
    max_seconds: float = 40.0
    min_saved_tokens: int = 32
    max_summary_ratio: float = 0.8

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in (
            self.keep_recent_exchanges, self.max_prompt_tokens, self.max_output_tokens,
            self.max_characters, self.min_saved_tokens,
        )) or not isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("Compaction bounds must be positive.")
        if not isfinite(self.max_summary_ratio) or not 0 < self.max_summary_ratio < 1:
            raise ValueError("Compaction ratio must be between zero and one.")


@dataclass(frozen=True, slots=True)
class SummarySource:
    alias: str
    turn_id: UUID
    role: Literal["user", "assistant"]
    created_at: datetime
    attribution: SourceAttribution
    outcome: str = "completed"


@dataclass(frozen=True, slots=True)
class CompactionPrefix:
    session: ConversationSession
    through_turn_id: UUID
    entries: tuple[HistoryEntry, ...]
    digest: str
    prior_checkpoint_id: UUID | None
    include_diagnostics: bool = False


@dataclass(frozen=True, slots=True)
class SummaryEntry:
    speaker: Literal["user", "assistant", "mixed"]
    text: str
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SummaryCandidate:
    entries: tuple[SummaryEntry, ...]
    generated_at: datetime
    model: str
    template_sha256: str | None
    source_tokens: int
    summary_tokens: int
    generation: GenerationInfo


@dataclass(frozen=True, slots=True)
class SummaryCheckpoint:
    checkpoint_id: UUID
    profile_id: UUID
    session_id: UUID
    session_created_at: datetime
    source_revision: int
    through_turn_id: UUID
    covered_turn_ids: tuple[UUID, ...]
    source_digest: str
    sources: tuple[SummarySource, ...]
    limits: CompactionLimits
    candidate: SummaryCandidate
    schema_version: int = 1


class SummaryNotFound(LookupError):
    """No valid current checkpoint exists for this source session/profile."""
