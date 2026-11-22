"""Operation owners signal cancellation; work receives read-only observation."""

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event, Lock, RLock
from time import monotonic
from uuid import UUID


class GenerationCancelled(Exception):
    """Generation stopped in response to its owner's cancellation request."""

    def __init__(self, message: str, *, cause: "CancellationCause | None" = None) -> None:
        super().__init__(message)
        self.cause = cause


@dataclass(frozen=True, slots=True)
class CancellationCause:
    origin: str
    requested_monotonic: float | None = None
    deadline_monotonic: float | None = None


class CancellationSignal(Event):
    """Preserve the first request when owners race to stop an operation."""

    def __init__(self, origin: str = "explicit_signal") -> None:
        super().__init__()
        self.origin = origin
        self.cause: CancellationCause | None = None
        self._request_lock = RLock()

    def set(self, origin: str | None = None, *, deadline: float | None = None) -> None:
        with self._request_lock:
            if not self.is_set():
                self.cause = CancellationCause(origin or self.origin, monotonic(), deadline)
                super().set()


@dataclass(frozen=True, slots=True)
class CancellationToken:
    turn_id: UUID
    _signal: Event = field(repr=False, compare=False)
    _parent: "CancellationToken | None" = field(default=None, repr=False, compare=False)
    _deadline: float | None = field(default=None, repr=False, compare=False)
    _clock: Callable[[], float] = field(default=monotonic, repr=False, compare=False)
    _deadline_origin: str = field(default="token_deadline", repr=False, compare=False)

    @property
    def cancelled(self) -> bool:
        return (
            self._signal.is_set()
            or (self._parent is not None and self._parent.cancelled)
            or (self._deadline is not None and self._clock() >= self._deadline)
        )

    def raise_if_cancelled(self) -> None:
        if self.cancelled and (cause := self.cause):
            raise GenerationCancelled(f"Turn {self.turn_id} was cancelled by {cause.origin}.", cause=cause)

    @property
    def cause(self) -> CancellationCause | None:
        causes = []
        if self._signal.is_set():
            if not isinstance(self._signal, CancellationSignal):
                return CancellationCause("explicit_signal")
            causes.append(self._signal.cause)
        if self._parent is not None and (cause := self._parent.cause):
            causes.append(cause)
        if self._deadline is not None and self._clock() >= self._deadline:
            causes.append(CancellationCause(self._deadline_origin, deadline_monotonic=self._deadline))
        return min(causes, key=_cancelled_at, default=None)


def _cancelled_at(cause: CancellationCause) -> float:
    if cause.deadline_monotonic is not None:
        return cause.deadline_monotonic
    if cause.requested_monotonic is not None:
        return cause.requested_monotonic
    return float("-inf")


class TurnCancellationOwner:
    """Reserve one generation slot until its owning turn has fully unwound.

    The future Layer 1 coordinator alone calls begin, cancel, and finish. Each
    begin uses a fresh UUID. A cancellation request does not release the slot;
    finish is called only after the worker and stream have stopped.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._active: CancellationToken | None = None

    def begin(self, turn_id: UUID) -> CancellationToken:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("The active turn must finish before another can begin.")
            token = CancellationToken(turn_id, CancellationSignal("turn_owner"))
            self._active = token
            return token

    def cancel(self, turn_id: UUID, *, origin: str = "user_cancel") -> bool:
        """Request cancellation only if this is still the owning turn."""
        with self._lock:
            if self._active is None or self._active.turn_id != turn_id:
                return False
            self._active._signal.set(origin)
            return True

    def finish(self, turn_id: UUID) -> bool:
        """Release a stopped turn; stale callbacks cannot release a newer one."""
        with self._lock:
            if self._active is None or self._active.turn_id != turn_id:
                return False
            self._active = None
            return True

