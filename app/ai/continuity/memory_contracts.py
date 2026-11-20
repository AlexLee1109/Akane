"""Immutable L3 memory records; no model, prompt, or automatic admission path."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from ..provenance import SourceAttribution


@dataclass(frozen=True, slots=True)
class MemorySource:
    session_id: UUID
    session_created_at: datetime
    turn_id: UUID
    turn_created_at: datetime
    start: int
    end: int
    attribution: SourceAttribution
    role: Literal["user"] = "user"


@dataclass(frozen=True, slots=True)
class MemoryProposal:
    """Ephemeral proposal. Content is the exact selected user quote, not a paraphrase."""

    memory_id: UUID
    profile_id: UUID
    content: str
    source: MemorySource


@dataclass(frozen=True, slots=True)
class MemoryApproval:
    """Caller's explicit human approval assertion, bound to profile, ID and action.

    This local API is not an authentication service or proof of human presence.
    The caller must obtain approval; no layer constructs approval automatically.
    """

    profile_id: UUID
    memory_id: UUID
    action: Literal["admit", "supersede", "delete"]
    approved_at: datetime
    approved: bool
    actor: Literal["user"] = "user"


@dataclass(frozen=True, slots=True)
class MemoryDeletion:
    reason: Literal["manual", "source_unavailable", "source_deletion"]
    deleted_at: datetime
    approval: MemoryApproval | None = None


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: UUID
    profile_id: UUID
    content: str | None
    source: MemorySource | None
    created_at: datetime
    approval: MemoryApproval
    status: Literal["active", "superseded", "deleted"] = "active"
    supersedes: UUID | None = None
    superseded_by: UUID | None = None
    deletion: MemoryDeletion | None = None


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    profile_id: UUID
    revision: int
    records: tuple[MemoryRecord, ...]


class MemoryNotFound(LookupError):
    """No requested memory is available in the selected profile/status view."""


class InvalidMemorySource(ValueError):
    """The selected evidence is not an exact durable user-authored source."""
