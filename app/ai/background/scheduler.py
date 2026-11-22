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
from weakref import ReferenceType, ref

from ..contracts import DelegationRequest, DelegationResult
from ..conversation.cancellation import CancellationSignal, CancellationToken, GenerationCancelled
from .contracts import BackgroundJob, BackgroundResult, TaskState
from .residency import LazyL2
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
    worker: BackgroundIntelligence | None
    signal: Event
    maintenance: bool
    result: BackgroundResult | None = None
    pin_id: UUID | None = None
    queue_deadline: float | None = None
    expires: float | None = None
    retired_worker: ReferenceType[BackgroundIntelligence] | None = None


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
        detached_retention_seconds: float = 900.0,
        clock: Callable[[], float] = monotonic, utc_clock: Callable[[], datetime] = _utc_now,
        lazy_l2: LazyL2 | None = None,
    ) -> None:
        if any(type(limit) is not int or limit < 1 for limit in (
            max_pending_background, max_pending_foreground, max_retained_terminal_jobs,
        )):
            raise ValueError("Queue bounds must be positive integers.")
        self._background_limit = max_pending_background
        self._foreground_limit = max_pending_foreground
        self._retained_limit = max_retained_terminal_jobs
        if not isfinite(detached_retention_seconds) or detached_retention_seconds <= 0:
            raise ValueError('Detached retention must be finite and positive.')
        self._detached_retention = detached_retention_seconds
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
        self._lazy_l2 = lazy_l2
        self._idle_until = None
        self._evicting = False

    def _timestamp(self) -> datetime:
        value = self._utc_clock()
        if value.utcoffset() is None:
            raise ValueError("Task timestamps must include a timezone.")
        return value

    def _available(self) -> None:
        if self._stopping:
            raise SchedulerClosed("Inference scheduler is shutting down.")

    def _entry(self, profile_id: UUID, job_id: UUID) -> _Background:
        self._expire_detached()
        entry = self._jobs.get(job_id)
        if entry is None or entry.job.profile_id != profile_id:
            raise JobNotFound("Job is not retained for this profile (unknown or evicted).")
        return entry

    def _remove(self, entry):
        del self._jobs[entry.job.job_id]
        del self._requests[entry.job.profile_id, entry.job.request_id]

    def _expire_detached(self):
        now = self._clock()
        for entry in tuple(self._jobs.values()):
            if entry.expires is not None and now >= entry.expires and entry.pin_id is None:
                self._remove(entry)
            elif (entry.job.status == TaskState.PENDING and entry.queue_deadline is not None
                    and now >= entry.queue_deadline):
                self._background.remove(entry)
                self._finish(entry, None, 'Task expired before an L2 slot became available.')

    def _start_worker(self):
        if self._thread is None:
            worker = Thread(target=self._run, name='akane-next-background', daemon=True)
            worker.start()
            self._thread = worker

    def _enqueue(self, entry):
        if self._prioritize_tasks and not entry.maintenance:
            position = next((i for i, queued in enumerate(self._background) if queued.maintenance), len(self._background))
            self._background.insert(position, entry)
            if isinstance(self._active, _Background) and self._active.maintenance:
                self._request_cancel(self._active, 'task_priority')
        else:
            self._background.append(entry)
        self._condition.notify_all()

    def submit_background(
        self, profile_id: UUID, request: DelegationRequest, worker: BackgroundIntelligence, *,
        job_id: UUID | None = None, budget_seconds: float = 30.0,
        cancellation: CancellationToken | None = None, maintenance: bool = False,
        reserve: bool = False,
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
        if reserve != (request.detached is not None) or reserve and (cancellation is not None or maintenance):
            raise ValueError('Only detached work can reserve a slot, with independent cancellation.')
        with self._condition:
            self._available()
            self._expire_detached()
            if job_id in self._jobs:
                entry = self._entry(profile_id, job_id)
                original_worker = entry.retired_worker() if entry.retired_worker is not None else entry.worker
                if (entry.job.request != request or entry.job.budget_seconds != budget_seconds
                        or original_worker is not worker or entry.maintenance != maintenance):
                    raise JobConflict("Job ID was already submitted with different work.")
                return entry.job
            if (profile_id, request.request_id) in self._requests:
                raise JobConflict("Delegation request already has a different job ID.")
            if self.pending_background_count() >= self._background_limit:
                raise QueueFull("Background pending queue is full.")
            # Reserve retention capacity before starting work. Never evict an
            # active/pending job or a result pinned for pending/in-flight delivery.
            while len(self._jobs) >= self._retained_limit:
                candidates = [item for item in self._jobs.values()
                              if item.result is not None and item.pin_id is None and item is not self._active]
                if not candidates:
                    raise QueueFull("Result retention is full of active or pinned work.")
                victim = min(candidates, key=lambda item: item.result.completed_at)
                self._remove(victim)
            signal = CancellationSignal('background_scheduler')
            token = CancellationToken(request.turn_id, signal, _parent=cancellation)
            job = BackgroundJob(job_id, profile_id, request, self._timestamp(), budget_seconds, token)
            if reserve:
                job = replace(job, status=TaskState.RESERVED)
            entry = _Background(job, worker, signal, maintenance)
            if not token.cancelled:
                self._start_worker()
            self._jobs[job_id] = entry
            self._requests[profile_id, request.request_id] = job_id
            if token.cancelled:
                self._finish(entry, None, None)
            elif not reserve:
                self._enqueue(entry)
            self._condition.notify_all()
            return entry.job

    def publish_reserved(self, profile_id, job_id, *, queue_seconds):
        if not isfinite(queue_seconds) or queue_seconds <= 0:
            raise ValueError('Queue wait must be finite and positive.')
        with self._condition:
            entry = self._entry(profile_id, job_id)
            if entry.job.request.detached is None:
                raise JobConflict('Only detached work can be published from a reservation.')
            if entry.job.status == TaskState.RESERVED:
                entry.job = replace(entry.job, status=TaskState.PENDING, status_version=entry.job.status_version + 1)
                if self._stopping:
                    self._request_cancel(entry, 'shutdown')
                    self._finish(entry, None, None)
                else:
                    entry.queue_deadline = self._clock() + queue_seconds
                    self._enqueue(entry)
            return entry.job

    def discard_reserved(self, profile_id, job_id):
        with self._condition:
            entry = self._jobs.get(job_id)
            if entry is not None and entry.job.profile_id == profile_id and entry.job.status == TaskState.RESERVED:
                self._remove(entry)
                self._condition.notify_all()

    def get_job(self, profile_id: UUID, job_id: UUID) -> BackgroundJob:
        with self._condition:
            return self._entry(profile_id, job_id).job

    def get_result(self, profile_id: UUID, job_id: UUID) -> BackgroundResult | None:
        with self._condition:
            return self._entry(profile_id, job_id).result

    def detached_jobs(self, profile_id, conversation_id, created_at):
        with self._condition:
            self._expire_detached()
            return tuple(entry.job for entry in self._jobs.values()
                         if entry.job.profile_id == profile_id and entry.job.conversation_id == conversation_id
                         and entry.job.request.session_created_at == created_at and entry.job.request.detached is not None
                         and entry.job.status != TaskState.RESERVED)

    def pending_background_count(self) -> int:
        with self._condition:
            self._expire_detached()
            return len(self._background) + sum(entry.job.status == TaskState.RESERVED for entry in self._jobs.values())

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

    def consume_result(self, profile_id: UUID, job_id: UUID, *, pin_id: UUID, turn_id: UUID,
                       release_pin: bool = False) -> BackgroundJob:
        with self._condition:
            entry = self._entry(profile_id, job_id)
            if entry.pin_id != pin_id or entry.job.invalidated:
                raise JobConflict("Delivery is not owned or was invalidated.")
            if entry.job.delivered_turn_id not in (None, turn_id):
                raise JobConflict("Result was delivered to another turn.")
            if entry.job.delivered_turn_id is None:
                entry.job = replace(entry.job, delivered_turn_id=turn_id, status_version=entry.job.status_version + 1)
            if release_pin:
                entry.pin_id = None
            return entry.job

    def invalidate_result(self, profile_id: UUID, job_id: UUID) -> None:
        """Explicitly withdraw completed findings without rewriting their outcome."""
        with self._condition:
            entry = self._entry(profile_id, job_id)
            if entry.pin_id is not None:
                raise JobConflict("Remove the reserved attachment before invalidating its result.")
            if not entry.job.invalidated:
                entry.job = replace(entry.job, invalidated=True, status_version=entry.job.status_version + 1)

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
            if entry.job.status == TaskState.RESERVED:
                return False
            self._request_cancel(entry, 'background_cancel')
            if entry.job.status == TaskState.PENDING:
                self._background.remove(entry)
                self._finish(entry, None, None)
            self._condition.notify_all()
            return True

    def _request_cancel(self, entry: _Background, origin: str) -> None:
        if entry.job.cancellation_requested_monotonic is None:
            entry.job = replace(entry.job, cancellation_requested_monotonic=self._clock(),
                                status_version=entry.job.status_version + 1)
        entry.signal.set(origin)

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

    def borrow_background_model(self, model, cancellation):
        """Called by an authorized executor after validating its granted job."""
        if self._lazy_l2 is None:
            return model, cancellation
        with self._condition:
            entry = self._active
            if (not isinstance(entry, _Background) or current_thread() is not self._thread
                    or entry.job.cancellation is not cancellation or model is not self._lazy_l2):
                raise RuntimeError('Lazy L2 access requires its active executor lease.')

        def mark_aborting():
            with self._condition:
                if self._active is entry and entry.job.status == TaskState.LOADING:
                    entry.job = replace(entry.job, load_aborting=True, status_version=entry.job.status_version + 1)
                    self._condition.notify_all()

        model = self._lazy_l2.acquire(cancellation, mark_aborting)
        with self._condition:
            cancellation.raise_if_cancelled()
            if entry.job.status == TaskState.LOADING:
                entry.job = replace(entry.job, status=TaskState.RUNNING, status_version=entry.job.status_version + 1)
                self._condition.notify_all()
            return model, CancellationToken(cancellation.turn_id, entry.signal, _parent=cancellation,
                _deadline=self._clock() + entry.job.budget_seconds, _clock=self._clock,
                _deadline_origin='background_lease')

    def background_activity(self):
        with self._condition:
            return {'active_jobs': int(isinstance(self._active, _Background)),
                    'pending_jobs': self.pending_background_count()}

    @contextmanager
    def foreground(
        self, profile_id: UUID, request_id: UUID, *, cancellation: CancellationToken,
    ) -> Iterator[CancellationToken]:
        if not isinstance(profile_id, UUID) or not isinstance(request_id, UUID):
            raise ValueError("Foreground identifiers must be UUIDs.")
        signal = CancellationSignal('foreground_scheduler')
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
                self._request_cancel(self._active, 'foreground_preemption')
            self._condition.notify_all()
            try:
                self._condition.wait_for(lambda: (
                    self._stopping or ticket.token.cancelled
                    or (self._active is None and not self._evicting and self._foreground[0] is ticket)
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
                    ticket.signal.set('foreground_cancel')
                    self._condition.notify_all()
                    return True
            return False

    @contextmanager
    def pause_background(self, *, cancellation: CancellationToken, drain_active: bool = True,
                         drain_loading: bool = False) -> Iterator[None]:
        """Hold background dispatch during an independent foreground pass on Metal."""
        with self._condition:
            self._available()
            cancellation.raise_if_cancelled()
            self._background_pauses += 1
            if isinstance(self._active, _Background) and self._active.maintenance:
                self._request_cancel(self._active, 'foreground_preemption')
        try:
            with self._condition:
                while self._evicting or (isinstance(self._active, _Background)
                        and (drain_active or drain_loading and self._active.job.status == TaskState.LOADING)):
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
            entry.job.status in (TaskState.LOADING, TaskState.RUNNING) and self._active is not entry
        ) or (
            entry.job.status == TaskState.PENDING and not entry.job.cancellation.cancelled and error is None
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
        entry.job = replace(entry.job, status=status, status_version=entry.job.status_version + 1)
        entry.result = BackgroundResult(
            entry.job.job_id, entry.job.profile_id, entry.job.conversation_id,
            status, payload, self._timestamp(), error, generation,
        )
        # Results outlive execution. Keep the original identity for replay
        # without retaining a compaction executor's capture and prepared prompt.
        try:
            entry.retired_worker = ref(entry.worker)
        except TypeError:
            pass  # Some slotted executors need a strong reference for replay.
        else:
            entry.worker = None
        if entry.job.request.detached is not None:
            entry.expires = self._clock() + self._detached_retention
        self._condition.notify_all()

    def _run(self) -> None:
        try:
            self._serve_background()
        finally:
            if self._lazy_l2 is not None:
                try:
                    self._lazy_l2.unload('shutdown')
                except Exception:
                    pass  # The awaiting shutdown caller checks for an unreleased owner.

    def _wait_for_dispatch(self):
        """Wait under the condition. True reserves an idle eviction on this thread."""
        if self._lazy_l2 is None:
            self._condition.wait_for(lambda: self._stopping or (
                self._background and self._active is None and not self._foreground and not self._background_pauses
            ))
            self._expire_detached()
            return False
        while not self._stopping:
            pending = self.pending_background_count()
            available = self._active is None and not self._foreground and not self._background_pauses
            if self._background and available:
                self._idle_until = None
                return False
            timeout = None
            if pending or self._active is not None or not self._lazy_l2.ready:
                self._idle_until = None
            else:
                now = self._clock()
                if self._idle_until is None:
                    self._idle_until = now + self._lazy_l2.idle_seconds
                remaining = self._idle_until - now
                if remaining <= 0 and available:
                    self._evicting = True
                    return True
                if remaining > 0:
                    timeout = min(remaining, 60)
            # Wake for queued expiry even when a foreground pause prevents dispatch.
            deadlines = [entry.queue_deadline for entry in self._background if entry.queue_deadline is not None]
            if deadlines:
                until_expiry = max(0, min(deadlines) - self._clock())
                timeout = until_expiry if timeout is None else min(timeout, until_expiry)
            self._condition.wait(timeout)
        return False

    def _serve_background(self) -> None:
        while True:
            with self._condition:
                evict = self._wait_for_dispatch()
                if self._stopping:
                    return
                if not evict:
                    if not self._background:
                        continue
                    entry = self._background.popleft()
                    if entry.job.cancellation.cancelled:
                        self._finish(entry, None, None)
                        del entry
                        continue
                    now = self._clock()
                    token = CancellationToken(entry.job.request.turn_id, entry.signal, _parent=entry.job.cancellation,
                        _deadline=now + entry.job.budget_seconds if self._lazy_l2 is None else None, _clock=self._clock,
                        _deadline_origin='background_lease')
                    status = (TaskState.LOADING if self._lazy_l2 is not None and not self._lazy_l2.ready
                              else TaskState.RUNNING)
                    entry.job = replace(entry.job, cancellation=token, status=status,
                                        acquired_monotonic=now, status_version=entry.job.status_version + 1)
                    self._active = entry
            if evict:
                try:
                    self._lazy_l2.unload('idle')
                except Exception:
                    pass  # The owner exposes the failure and refuses reuse.
                finally:
                    with self._condition:
                        self._evicting = False
                        self._idle_until = None
                        self._condition.notify_all()
                continue
            payload, error = None, None
            try:
                token.raise_if_cancelled()
                if entry.worker is None:
                    raise RuntimeError('Running work has no executor.')
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
                        # The worker may sleep indefinitely after the retention table evicts this job.
                        del entry, payload, token
                        self._condition.notify_all()

    def request_shutdown(self) -> None:
        """Close admission and signal queued and active operations without joining."""
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            while self._background:
                entry = self._background.popleft()
                self._request_cancel(entry, 'shutdown')
                self._finish(entry, None, None)
            for ticket in self._foreground:
                ticket.signal.set('shutdown')
            if self._active is not None:
                if isinstance(self._active, _Background):
                    self._request_cancel(self._active, 'shutdown')
                else:
                    self._active.signal.set('shutdown')
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
            if self._lazy_l2 is not None and self._lazy_l2.snapshot()['loaded']:
                raise RuntimeError('Qwen L2 shutdown left an unreleased model.')
