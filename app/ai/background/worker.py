"""A bounded work boundary; no scheduler, autonomous loop, or model loader."""

from collections.abc import Callable
from typing import Protocol

from ..contracts import ContextRecord, DelegationRequest, DelegationResult
from ..conversation.cancellation import CancellationToken, GenerationCancelled


class BackgroundIntelligence(Protocol):
    def execute(
        self, request: DelegationRequest, *, cancellation: CancellationToken
    ) -> DelegationResult:
        """Return attributed work data for L1 to consider."""
        ...


DelegationHandler = Callable[
    [DelegationRequest, CancellationToken], tuple[ContextRecord, ...]
]


class BackgroundWorker:
    """Runs only a supplied task handler; executing it is an explicit action.

    A handler must observe cancellation. It may not call the dialogue model
    concurrently, emit interaction events, mutate identity, or persist memory.
    """

    def __init__(self, handler: DelegationHandler) -> None:
        self._handler = handler

    def execute(
        self, request: DelegationRequest, *, cancellation: CancellationToken
    ) -> DelegationResult:
        if cancellation.turn_id != request.turn_id:
            raise ValueError("Cancellation token belongs to another turn.")
        try:
            cancellation.raise_if_cancelled()
            findings = self._handler(request, cancellation)
            cancellation.raise_if_cancelled()
            return DelegationResult(
                request.request_id, request.conversation_id, request.turn_id,
                "completed", findings,
            )
        except GenerationCancelled:
            return DelegationResult(
                request.request_id, request.conversation_id, request.turn_id, "cancelled"
            )
        except Exception as error:
            return DelegationResult(
                request.request_id, request.conversation_id, request.turn_id,
                "failed", error=str(error),
            )
