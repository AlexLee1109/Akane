"""Immutable messages crossing layer boundaries; no runtime or backend imports."""

from dataclasses import dataclass, field
from hashlib import sha256
from datetime import datetime
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from .conversation.cancellation import CancellationToken
from .conversation.events import InteractionEvent
from .provenance import SourceAttribution

if TYPE_CHECKING:
    from .context_contracts import L3Projection


__all__ = [
    "InteractionRequest", "InteractionEvent", "DelegationRequest",
    "DelegationResult", "ConversationContext", "ContextSnapshot",
    "CancellationToken", "HistoryEntry", "ContextRecord", "SourceAttribution",
    "Profile", "ConversationSession",
    "GenerationInfo",
    "AttachedResult",
]


@dataclass(frozen=True, slots=True)
class Profile:
    profile_id: UUID
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationSession:
    conversation_id: UUID
    profile_id: UUID
    created_at: datetime
    revision: int = 0


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    turn_id: UUID
    role: Literal["user", "assistant"]
    text: str
    outcome: Literal["completed", "cancelled", "failed"] = "completed"
    created_at: datetime | None = None
    profile_id: UUID | None = None
    conversation_id: UUID | None = None
    attribution: SourceAttribution | None = None

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError("History roles must be user or assistant.")
        if self.outcome not in ("completed", "cancelled", "failed"):
            raise ValueError("Unknown history outcome.")
        if self.role == "user" and self.outcome != "completed":
            raise ValueError("Only assistant replies have generation outcomes.")
        if self.created_at is not None and self.created_at.utcoffset() is None:
            raise ValueError("History timestamps must include a timezone.")
        if self.attribution is not None:
            expected_owner = "user" if self.role == "user" else "l1"
            if self.attribution.owner != expected_owner or not self.attribution.source_id:
                raise ValueError("History attribution must preserve the speaker's ownership.")

    @property
    def source(self) -> SourceAttribution:
        if self.attribution is not None:
            return self.attribution
        source_id = f"{self.turn_id}:{self.role}"
        if self.profile_id is not None and self.conversation_id is not None:
            source_id = f"{self.profile_id}:{self.conversation_id}:{source_id}"
        return SourceAttribution(
            "user" if self.role == "user" else "l1",
            source_id,
        )


@dataclass(frozen=True, slots=True)
class ContextRecord:
    """A memory, summary, or finding with its supporting sources retained."""

    text: str
    sources: tuple[SourceAttribution, ...]

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("Context records require source attribution.")


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """L3's bounded read view; never contains the pending interaction turn."""

    conversation_id: UUID
    revision: int = 0
    recent: tuple[HistoryEntry, ...] = ()
    memories: tuple[ContextRecord, ...] = ()
    summary: ContextRecord | None = None
    summary_turn_ids: tuple[UUID, ...] = ()
    owner: Literal["l3"] = field(default="l3", init=False)
    profile_id: UUID | None = None

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("Snapshot revision cannot be negative.")
        if (self.summary is None) != (not self.summary_turn_ids):
            raise ValueError("A summary and its covered turn IDs must appear together.")
        if set(self.summary_turn_ids) & {entry.turn_id for entry in self.recent}:
            raise ValueError("Summary coverage must not overlap recent history.")
        for entry in self.recent:
            if entry.conversation_id not in (None, self.conversation_id):
                raise ValueError("Snapshot history belongs to another session.")
            if self.profile_id is not None and entry.profile_id not in (None, self.profile_id):
                raise ValueError("Snapshot history belongs to another profile.")


@dataclass(frozen=True, slots=True)
class DetachedBinding:
    runtime_epoch: UUID
    source_sha256: str


@dataclass(frozen=True, slots=True)
class TaskReference:
    task_id: UUID
    runtime_epoch: UUID


@dataclass(frozen=True, slots=True)
class DeliveryClaim:
    task: TaskReference
    pin_id: UUID
    profile_id: UUID
    conversation_id: UUID
    session_created_at: datetime
    target_turn_id: UUID
    expected_revision: int
    source_sha256: str
    finding_sha256: str


@dataclass(frozen=True, slots=True)
class DelegationRequest:
    """An explicit L1 task; L2 cannot create independent goals."""

    request_id: UUID
    conversation_id: UUID
    turn_id: UUID
    task: str
    snapshot: ContextSnapshot | None = None
    owner: Literal["l1"] = field(default="l1", init=False)
    source_context: tuple[ContextRecord, ...] = ()
    session_created_at: datetime | None = None
    source_turn_id: UUID | None = None
    delivery_turn_id: UUID | None = None
    detached: DetachedBinding | None = None

    def __post_init__(self) -> None:
        if self.snapshot is not None:
            if self.snapshot.conversation_id != self.conversation_id:
                raise ValueError("Delegation snapshot belongs to another conversation.")
        if self.detached is not None:
            expected = ContextRecord(self.task, (SourceAttribution('user', f'{self.turn_id}:user'),))
            if (self.delivery_turn_id is not None or self.source_turn_id != self.turn_id
                    or self.request_id != self.turn_id or self.session_created_at is None
                    or self.snapshot is None or self.snapshot.profile_id is None
                    or self.snapshot != ContextSnapshot(self.conversation_id, self.snapshot.revision,
                                                        profile_id=self.snapshot.profile_id)
                    or self.source_context != (expected,)
                    or self.detached.source_sha256 != sha256(self.task.encode()).hexdigest()):
                raise ValueError('Detached work requires the exact authorized user source and a metadata-only anchor.')

    @property
    def automatic(self) -> bool:
        return self.delivery_turn_id is not None or self.detached is not None


@dataclass(frozen=True, slots=True)
class GenerationInfo:
    """Measured generation metadata, not evidence that model claims are true."""

    prompt_tokens: int
    sampled_tokens: int | None
    completion_tokens: int | None
    started_monotonic: float
    finished_monotonic: float
    first_token_monotonic: float | None = None
    last_checkpoint_monotonic: float | None = None
    generator_exit_monotonic: float | None = None
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class DelegationResult:
    """L2 work data, never an assistant message or an identity instruction."""

    request_id: UUID
    conversation_id: UUID
    turn_id: UUID
    status: Literal["completed", "cancelled", "failed"]
    findings: tuple[ContextRecord, ...] = ()
    error: str | None = None
    owner: Literal["l2"] = field(default="l2", init=False)
    generation: GenerationInfo | None = None

    def __post_init__(self) -> None:
        if self.status != "completed" and self.findings:
            raise ValueError("Cancelled or failed work cannot publish findings.")
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("Only failed work must carry an error.")

    @property
    def source(self) -> SourceAttribution:
        return SourceAttribution("l2", str(self.request_id))


@dataclass(frozen=True, slots=True)
class TaskStatus:
    state: str
    label: str
    cancellation_stage: Literal['while queued', 'after worker dispatch began'] | None = None


@dataclass(frozen=True, slots=True)
class ConversationContext:
    """Optional L3 history and L2 data; an empty value is fully valid."""

    snapshot: ContextSnapshot | None = None
    delegation_results: tuple[DelegationResult, ...] = ()
    attached_results: tuple["AttachedResult", ...] = ()
    l3_projection: "L3Projection | None" = None
    delegation_failure: ContextRecord | None = None
    task_statuses: tuple[TaskStatus, ...] = ()


@dataclass(frozen=True, slots=True)
class AttachedResult:
    """An explicit one-turn delivery envelope, never a dialogue message."""

    job_id: UUID
    profile_id: UUID
    request: DelegationRequest
    result: DelegationResult
    created_at: datetime
    completed_at: datetime
    target_turn_id: UUID
    claim: DeliveryClaim | None = None

    def __post_init__(self) -> None:
        if (self.request.detached is None) != (self.claim is None):
            raise ValueError('A detached attachment requires its delivery claim.')
        if self.claim is not None and (
                self.claim.task != TaskReference(self.job_id, self.request.detached.runtime_epoch)
                or self.claim.profile_id != self.profile_id or self.claim.conversation_id != self.request.conversation_id
                or self.claim.session_created_at != self.request.session_created_at
                or self.claim.target_turn_id != self.target_turn_id
                or self.claim.source_sha256 != self.request.detached.source_sha256):
            raise ValueError('Attachment does not match its delivery claim.')
        if self.request.delivery_turn_id not in (None, self.target_turn_id):
            raise ValueError("Automatic findings are bound to one originating interaction.")
        if self.result.status != "completed" or not self.result.findings:
            raise ValueError("Only nonempty completed findings may be attached.")
        if (self.result.request_id, self.result.conversation_id, self.result.turn_id) != (
            self.request.request_id, self.request.conversation_id, self.request.turn_id,
        ):
            raise ValueError("Attachment result does not match its delegated request.")
        if self.created_at.utcoffset() is None or self.completed_at.utcoffset() is None:
            raise ValueError("Attachment timestamps must include a timezone.")


@dataclass(frozen=True, slots=True)
class InteractionRequest:
    conversation_id: UUID
    turn_id: UUID
    text: str
    context: ConversationContext = field(default_factory=ConversationContext)
    control_mode: Literal["plain", "decision", "final"] = "plain"
    acknowledge_work: bool = False

    def __post_init__(self):
        if self.control_mode not in ("plain", "decision", "final"):
            raise ValueError("Unknown interaction control mode.")
        if self.acknowledge_work and (self.control_mode != 'final' or self.context.attached_results
                                     or self.context.delegation_results or self.context.delegation_failure):
            raise ValueError('Acknowledgment requires a final-only pass without findings.')

    @property
    def source(self) -> SourceAttribution:
        return SourceAttribution("user", f"{self.turn_id}:user")
