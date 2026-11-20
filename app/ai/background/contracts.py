"""Process-local task envelopes. Payloads reuse the existing delegation contracts."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID

from ..contracts import DelegationRequest, DelegationResult, GenerationInfo
from ..conversation.cancellation import CancellationToken


class TaskState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class BackgroundJob:
    job_id: UUID
    profile_id: UUID
    request: DelegationRequest
    created_at: datetime
    budget_seconds: float
    cancellation: CancellationToken
    status: TaskState = TaskState.PENDING
    acquired_monotonic: float | None = None
    cancellation_requested_monotonic: float | None = None
    invalidated: bool = False
    delivered_turn_id: UUID | None = None

    @property
    def conversation_id(self) -> UUID:
        return self.request.conversation_id

    @property
    def request_id(self) -> UUID:
        return self.request.request_id


@dataclass(frozen=True, slots=True)
class BackgroundResult:
    job_id: UUID
    profile_id: UUID
    conversation_id: UUID
    status: TaskState
    payload: DelegationResult | None
    completed_at: datetime
    error: str | None = None
    generation: GenerationInfo | None = None
    released_monotonic: float | None = None

    def __post_init__(self) -> None:
        if self.status not in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
            raise ValueError("Results must have a terminal state.")
        if (self.status == TaskState.COMPLETED) != (self.payload is not None):
            raise ValueError("Only completed work may publish a payload.")
        if (self.status == TaskState.FAILED) != (self.error is not None):
            raise ValueError("Only failed work must carry an error.")
        if self.payload is not None and (
            self.payload.status != "completed" or self.payload.conversation_id != self.conversation_id
        ):
            raise ValueError("Result payload must be completed work from the same conversation.")
