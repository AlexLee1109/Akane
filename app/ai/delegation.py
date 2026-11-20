"""Small host policy/audit contracts; no worker, scheduler or model loader."""
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from .contracts import DelegationRequest

# Flip only after the declared real-model automatic/streaming acceptance passes.
DEFAULT_AUTODELEGATION = False


@dataclass(frozen=True, slots=True)
class AutoDelegationLimits:
    task_tokens: int = 256
    source_tokens: int = 1024
    source_characters: int = 4096
    output_tokens: int = 128
    prompt_tokens: int = 2048
    result_characters: int = 2048
    seconds: float = 20.0
    wait_seconds: float | None = None
    drain_seconds: float = 5.0
    min_task_tokens: int = 0

    def __post_init__(self):
        from math import isfinite
        if type(self.min_task_tokens) is not int or not 0 <= self.min_task_tokens <= self.task_tokens:
            raise ValueError('Invalid automatic task admission floor.')
        if any(type(value) is not int or value<1 for value in (
            self.task_tokens,self.source_tokens,self.source_characters,self.output_tokens,
            self.prompt_tokens,self.result_characters)) or any(
                not isfinite(value) or value <= 0 for value in (self.seconds, self.drain_seconds)
            ) or (self.wait_seconds is not None and (
                not isfinite(self.wait_seconds) or self.wait_seconds < self.seconds
            )):
            raise ValueError('Automatic task bounds must be positive.')


PRODUCTION_AUTODELEGATION_LIMITS = AutoDelegationLimits(
    task_tokens=1024, output_tokens=192, wait_seconds=24.0, min_task_tokens=40,
)


@dataclass(frozen=True, slots=True)
class AuthorizedTask:
    profile_id: UUID
    session_id: UUID
    session_created_at: datetime
    turn_id: UUID
    revision: int | None
    request: DelegationRequest
    task_tokens: int
    source_tokens: int
