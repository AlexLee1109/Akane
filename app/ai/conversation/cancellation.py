"""Operation owners signal cancellation; work receives read-only observation."""

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event, Lock
from time import monotonic
from uuid import UUID


class GenerationCancelled(Exception):
    """Generation stopped in response to its owner's cancellation request."""


@dataclass(frozen=True, slots=True)
class CancellationToken:
    turn_id: UUID
    _signal: Event = field(repr=False, compare=False)
    _parent: "CancellationToken | None" = field(default=None, repr=False, compare=False)
    _deadline: float | None = field(default=None, repr=False, compare=False)
    _clock: Callable[[], float] = field(default=monotonic, repr=False, compare=False)

    @property
    def cancelled(self) -> bool:
        return (
            self._signal.is_set()
            or (self._parent is not None and self._parent.cancelled)
            or (self._deadline is not None and self._clock() >= self._deadline)
        )

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise GenerationCancelled(f"Turn {self.turn_id} was cancelled.")


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
            token = CancellationToken(turn_id, Event())
            self._active = token
            return token

    def cancel(self, turn_id: UUID) -> bool:
        """Request cancellation only if this is still the owning turn."""
        with self._lock:
            if self._active is None or self._active.turn_id != turn_id:
                return False
            self._active._signal.set()
            return True

    def finish(self, turn_id: UUID) -> bool:
        """Release a stopped turn; stale callbacks cannot release a newer one."""
        with self._lock:
            if self._active is None or self._active.turn_id != turn_id:
                return False
            self._active = None
            return True

