"""One cooperative inference gate, foreground leases, and an explicit L2 queue."""

from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import isfinite
from threading import Condition, Event, Thread, current_thread, get_ident
from time import monotonic
from uuid import UUID

from ..contracts import DelegationRequest, DelegationResult
from ..conversation.cancellation import CancellationToken, GenerationCancelled
from .contracts import BackgroundJob, BackgroundResult, TaskState
from .worker import BackgroundIntelligence


class QueueFull(RuntimeError):
    """The bounded pending queue has no free slot."""


class JobNotFound(LookupError):
    """No job is available to this profile."""


class JobConflict(ValueError):
    """A stable task identity was reused for different work."""


class SchedulerClosed(RuntimeError):
    """Shutdown has begun; new work is rejected."""


@dataclass(slots=True)
class _Background:
    job: BackgroundJob
    worker: BackgroundIntelligence
    signal: Event
    maintenance: bool
    result: BackgroundResult | None = None
    pin_id: UUID | None = None


@dataclass(slots=True)
class _Foreground:
    profile_id: UUID
    request_id: UUID
    token: CancellationToken
    signal: Event
    thread_id: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class InferenceScheduler:
    """Cooperating operations share one resource; this class never loads a model.

    Foreground work executes in its caller's thread. One lazily started thread
    executes explicitly submitted background jobs. Neither path writes storage.
    """

    def __init__(
        self, *, max_pending_background: int = 4, max_pending_foreground: int = 8,
        max_retained_terminal_jobs: int = 32, prioritize_tasks: bool = False,
        clock: Callable[[], float] = monotonic, utc_clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if any(type(limit) is not int or limit < 1 for limit in (
            max_pending_background, max_pending_foreground, max_retained_terminal_jobs,
        )):
            raise ValueError("Queue bounds must be positive integers.")
        self._background_limit = max_pending_background
        self._foreground_limit = max_pending_foreground
        self._retained_limit = max_retained_terminal_jobs
        self._prioritize_tasks = prioritize_tasks
        self._clock = clock
        self._utc_clock = utc_clock
        self._condition = Condition()
        self._background: deque[_Background] = deque()
        self._foreground: deque[_Foreground] = deque()
        self._jobs: dict[UUID, _Background] = {}
        self._requests: dict[tuple[UUID, UUID], UUID] = {}
        self._active: _Background | _Foreground | None = None
        self._background_pauses = 0
        self._thread: Thread | None = None
        self._stopping = False

    def _timestamp(self) -> datetime:
        value = self._utc_clock()
        if value.utcoffset() is None:
            raise ValueError("Task timestamps must include a timezone.")
        return value

    def _available(self) -> None:
        if self._stopping:
            raise SchedulerClosed("Inference scheduler is shutting down.")

    def _entry(self, profile_id: UUID, job_id: UUID) -> _Background:
        entry = self._jobs.get(job_id)
        if entry is None or entry.job.profile_id != profile_id:
            raise JobNotFound("Job is not retained for this profile (unknown or evicted).")
        return entry

    def submit_background(
        self, profile_id: UUID, request: DelegationRequest, worker: BackgroundIntelligence, *,
        job_id: UUID | None = None, budget_seconds: float = 30.0,
        cancellation: CancellationToken | None = None, maintenance: bool = False,
    ) -> BackgroundJob:
        job_id = request.request_id if job_id is None else job_id
        if not all(isinstance(value, UUID) for value in (
            profile_id, job_id, request.request_id, request.conversation_id, request.turn_id,
        )):
            raise ValueError("Task identifiers must be UUIDs.")
        if not isfinite(budget_seconds) or budget_seconds <= 0:
            raise ValueError("Execution budget must be finite and positive.")
        if not request.task.strip() or not callable(getattr(worker, "execute", None)):
            raise ValueError("A task and an explicit work executor are required.")
        if cancellation is not None and cancellation.turn_id != request.turn_id:
            raise ValueError("Cancellation token belongs to another turn.")
        if request.snapshot is not None and request.snapshot.profile_id not in (None, profile_id):
            raise ValueError("Task snapshot belongs to another profile.")
        with self._condition:
            self._available()
            if job_id in self._jobs:
                entry = self._entry(profile_id, job_id)
                if (entry.job.request != request or entry.job.budget_seconds != budget_seconds
                        or entry.worker is not worker or entry.maintenance != maintenance):
                    raise JobConflict("Job ID was already submitted with different work.")
                return entry.job
            if (profile_id, request.request_id) in self._requests:
                raise JobConflict("Delegation request already has a different job ID.")
            if len(self._background) >= self._background_limit:
                raise QueueFull("Background pending queue is full.")
            # Reserve retention capacity before starting work. Never evict an
            # active/pending job or a result pinned for pending/in-flight delivery.
            while len(self._jobs) >= self._retained_limit:
                candidates = [item for item in self._jobs.values()
                              if item.result is not None and item.pin_id is None and item is not self._active]
                if not candidates:
                    raise QueueFull("Result retention is full of active or pinned work.")
                victim = min(candidates, key=lambda item: item.result.completed_at)
                del self._jobs[victim.job.job_id]
                del self._requests[victim.job.profile_id, victim.job.request_id]
            signal = Event()
            token = CancellationToken(request.turn_id, signal, _parent=cancellation)
            job = BackgroundJob(job_id, profile_id, request, self._timestamp(), budget_seconds, token)
            entry = _Background(job, worker, signal, maintenance)
            self._jobs[job_id] = entry
            self._requests[profile_id, request.request_id] = job_id
            if token.cancelled:
                self._finish(entry, None, None)
            else:
                if self._prioritize_tasks and not maintenance:
                    position = next((i for i, queued in enumerate(self._background) if queued.maintenance),
                                    len(self._background))
                    self._background.insert(position, entry)
                    if isinstance(self._active, _Background) and self._active.maintenance:
                        self._request_cancel(self._active)
                else:
                    self._background.append(entry)
                if self._thread is None:
                    self._thread = Thread(target=self._run, name="akane-next-background", daemon=True)
                    self._thread.start()
            self._condition.notify_all()
            return entry.job

    def get_job(self, profile_id: UUID, job_id: UUID) -> BackgroundJob:
        with self._condition:
            return self._entry(profile_id, job_id).job

    def get_result(self, profile_id: UUID, job_id: UUID) -> BackgroundResult | None:
        with self._condition:
            return self._entry(profile_id, job_id).result

    def pending_background_count(self) -> int:
        with self._condition:
            return len(self._background)

    def pin_result(self, profile_id: UUID, job_id: UUID, pin_id: UUID) -> tuple[BackgroundJob, BackgroundResult]:
        with self._condition:
            entry = self._entry(profile_id, job_id)
            if (entry.result is None or entry.job.status != TaskState.COMPLETED
                    or entry.job.invalidated or entry.job.delivered_turn_id is not None):
                raise JobConflict("Result is incomplete, invalidated, or already delivered.")
            if entry.pin_id not in (None, pin_id):
                raise JobConflict("Result is already reserved for another delivery.")
            entry.pin_id = pin_id
            return entry.job, entry.result

    def unpin_result(self, profile_id: UUID, job_id: UUID, pin_id: UUID) -> None:
        with self._condition:
            entry = self._entry(profile_id, job_id)
            if entry.pin_id != pin_id:
                raise JobConflict("Delivery does not own this result pin.")
            entry.pin_id = None

    def consume_result(self, profile_id: UUID, job_id: UUID, *, pin_id: UUID, turn_id: UUID) -> None:
        with self._condition:
            entry = self._entry(profile_id, job_id)
            if entry.pin_id != pin_id or entry.job.invalidated:
                raise JobConflict("Delivery is not owned or was invalidated.")
            if entry.job.delivered_turn_id not in (None, turn_id):
                raise JobConflict("Result was delivered to another turn.")
            entry.job = replace(entry.job, delivered_turn_id=turn_id)

    def invalidate_result(self, profile_id: UUID, job_id: UUID) -> None:
        """Explicitly withdraw completed findings without rewriting their outcome."""
        with self._condition:
            entry = self._entry(profile_id, job_id)
            if entry.pin_id is not None:
                raise JobConflict("Remove the reserved attachment before invalidating its result.")
            entry.job = replace(entry.job, invalidated=True)

    def wait_result(self, profile_id: UUID, job_id: UUID, *, timeout: float | None = None) -> BackgroundResult:
        with self._condition:
            entry = self._entry(profile_id, job_id)
            if not self._condition.wait_for(lambda: entry.result is not None, timeout):
                raise TimeoutError("Background work has not stopped.")
            return entry.result

    def cancel_background(self, profile_id: UUID, job_id: UUID) -> bool:
        with self._condition:
            entry = self._entry(profile_id, job_id)
            if entry.result is not None:
                return False
            self._request_cancel(entry)
            if entry.job.status == TaskState.PENDING:
                self._background.remove(entry)
                self._finish(entry, None, None)
            self._condition.notify_all()
            return True

    def _request_cancel(self, entry: _Background) -> None:
        if entry.job.cancellation_requested_monotonic is None:
            entry.job = replace(entry.job, cancellation_requested_monotonic=self._clock())
        entry.signal.set()

    def require_background_owner(
        self, profile_id: UUID, request_id: UUID, *, cancellation: CancellationToken,
    ) -> None:
        """A model-backed executor must already be running in its granted slot."""
        with self._condition:
            if (not isinstance(self._active, _Background) or current_thread() is not self._thread
                    or self._active.job.profile_id != profile_id
                    or self._active.job.request_id != request_id
                    or self._active.job.cancellation is not cancellation):
                raise RuntimeError("Native background execution requires its scheduler-owned lease.")

    @contextmanager
    def foreground(
        self, profile_id: UUID, request_id: UUID, *, cancellation: CancellationToken,
    ) -> Iterator[CancellationToken]:
        if not isinstance(profile_id, UUID) or not isinstance(request_id, UUID):
            raise ValueError("Foreground identifiers must be UUIDs.")
        signal = Event()
        ticket = _Foreground(
            profile_id, request_id,
            CancellationToken(cancellation.turn_id, signal, _parent=cancellation), signal, get_ident(),
        )
        with self._condition:
            self._available()
            ticket.token.raise_if_cancelled()
            tickets = (*self._foreground, *((self._active,) if isinstance(self._active, _Foreground) else ()))
            if any(item.profile_id == profile_id and item.request_id == request_id for item in tickets):
                raise JobConflict("Foreground request already owns or is waiting for inference.")
            if len(self._foreground) >= self._foreground_limit:
                raise QueueFull("Foreground pending queue is full.")
            self._foreground.append(ticket)
            if isinstance(self._active, _Background):
                self._request_cancel(self._active)
            self._condition.notify_all()
            try:
                self._condition.wait_for(lambda: (
                    self._stopping or ticket.token.cancelled
                    or (self._active is None and self._foreground[0] is ticket)
                ))
                ticket.token.raise_if_cancelled()
                self._available()
                self._foreground.popleft()
                self._active = ticket
            finally:
                if ticket in self._foreground:
                    self._foreground.remove(ticket)
                    self._condition.notify_all()
        try:
            yield ticket.token
        finally:
            with self._condition:
                if self._active is not ticket:
                    raise RuntimeError("Stale foreground resource owner.")
                self._active = None
                self._condition.notify_all()

    def cancel_foreground(self, profile_id: UUID, request_id: UUID) -> bool:
        """Signal and wake a queued/running request; host calls this on interruption."""
        with self._condition:
            tickets = (*self._foreground, *((self._active,) if isinstance(self._active, _Foreground) else ()))
            for ticket in tickets:
                if ticket.profile_id == profile_id and ticket.request_id == request_id:
                    ticket.signal.set()
                    self._condition.notify_all()
                    return True
            return False

    @contextmanager
    def pause_background(self, *, cancellation: CancellationToken) -> Iterator[None]:
        """Hold background dispatch during an independent foreground pass on Metal."""
        with self._condition:
            self._available()
            cancellation.raise_if_cancelled()
            self._background_pauses += 1
            if isinstance(self._active, _Background) and self._active.maintenance:
                self._request_cancel(self._active)
        try:
            with self._condition:
                while isinstance(self._active, _Background):
                    self._available()
                    cancellation.raise_if_cancelled()
                    # The other scheduler owns cancellation of this L1 token.
                    self._condition.wait(0.05)
                self._available()
                cancellation.raise_if_cancelled()
            yield
        finally:
            with self._condition:
                self._background_pauses -= 1
                self._condition.notify_all()

    def _finish(
        self, entry: _Background, payload: DelegationResult | None, error: str | None,
    ) -> None:
        """Only the owning worker or cancellation of a pending job may finalize."""
        if entry.result is not None or (
            entry.job.status == TaskState.RUNNING and self._active is not entry
        ) or (
            entry.job.status == TaskState.PENDING and not entry.job.cancellation.cancelled
        ):
            raise RuntimeError("Duplicate or stale background completion.")
        generation = payload.generation if payload is not None else None
        if entry.job.cancellation.cancelled or (payload is not None and payload.status == "cancelled"):
            status, payload, error = TaskState.CANCELLED, None, None
            entry.signal.set()
        elif error is not None:
            status, payload = TaskState.FAILED, None
        elif payload is None:
            raise RuntimeError("Uncancelled work returned no result.")
        elif (payload.request_id, payload.conversation_id, payload.turn_id) != (
            entry.job.request_id, entry.job.conversation_id, entry.job.request.turn_id,
        ):
            status, payload, error = TaskState.FAILED, None, "Work returned stale or mismatched delegation IDs."
        elif payload.status == "failed":
            status, error, payload = TaskState.FAILED, payload.error, None
        else:
            status = TaskState.COMPLETED
        entry.job = replace(entry.job, status=status)
        entry.result = BackgroundResult(
            entry.job.job_id, entry.job.profile_id, entry.job.conversation_id,
            status, payload, self._timestamp(), error, generation,
        )
        self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopping or (
                    self._background and self._active is None and not self._foreground and not self._background_pauses
                ))
                if self._stopping:
                    return
                entry = self._background.popleft()
                if entry.job.cancellation.cancelled:
                    self._finish(entry, None, None)
                    continue
                token = CancellationToken(
                    entry.job.request.turn_id, entry.signal, _parent=entry.job.cancellation,
                    _deadline=self._clock() + entry.job.budget_seconds, _clock=self._clock,
                )
                entry.job = replace(entry.job, cancellation=token, status=TaskState.RUNNING,
                                    acquired_monotonic=self._clock())
                self._active = entry
            payload, error = None, None
            try:
                token.raise_if_cancelled()
                payload = entry.worker.execute(entry.job.request, cancellation=token)
                if not isinstance(payload, DelegationResult) or payload.status not in ("completed", "failed", "cancelled"):
                    payload, error = None, "Executor returned an invalid DelegationResult."
            except GenerationCancelled:
                entry.signal.set()
            except BaseException as exception:
                error = f"{type(exception).__name__}: {exception}"
            finally:
                with self._condition:
                    try:
                        self._finish(entry, payload, error)
                    finally:
                        self._active = None
                        if entry.result is not None:
                            entry.result = replace(entry.result, released_monotonic=self._clock())
                        self._condition.notify_all()

    def request_shutdown(self) -> None:
        """Close admission and signal queued and active operations without joining."""
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            while self._background:
                entry = self._background.popleft()
                self._request_cancel(entry)
                self._finish(entry, None, None)
            for ticket in self._foreground:
                ticket.signal.set()
            if self._active is not None:
                if isinstance(self._active, _Background):
                    self._request_cancel(self._active)
                else:
                    self._active.signal.set()
            self._condition.notify_all()

    def shutdown(self, *, wait: bool = True, timeout: float | None = None) -> None:
        with self._condition:
            if wait and (current_thread() is self._thread or (
                isinstance(self._active, _Foreground) and self._active.thread_id == get_ident()
            )):
                raise RuntimeError("Release your inference lease or use shutdown(wait=False).")
            self.request_shutdown()
            deadline = monotonic() + timeout if timeout is not None else None
            if wait and not self._condition.wait_for(lambda: self._active is None, timeout):
                raise TimeoutError("Work has not reached a safe cancellation checkpoint.")
        if wait and self._thread is not None:
            self._thread.join(None if deadline is None else max(0, deadline - monotonic()))
            if self._thread.is_alive():
                raise TimeoutError("Background worker has not stopped.")
