"""Host-owned context maintenance with one admission slot and completion thread."""
from collections import OrderedDict
from dataclasses import replace
from threading import Condition, Thread, TIMEOUT_MAX
from time import monotonic, perf_counter
from uuid import uuid4

from .automatic_contracts import AutomaticContextPolicy, MaintenanceStatus
from .compaction import Compactor
from .continuity.compaction import JSONSummaryStore
from .continuity.store import ContinuityConflict, SessionNotFound
from .conversation.cancellation import GenerationCancelled, TurnCancellationOwner


class AutomaticContextManager:
    def __init__(self, store, model, scheduler, policy: AutomaticContextPolicy, *, clock=None):
        self.store, self.scheduler, self.policy = store, scheduler, policy
        self.compactor = Compactor(JSONSummaryStore(store), model, scheduler, limits=policy.compaction)
        self._condition = Condition()
        self._thread = None
        self._operation = None
        self._owner = TurnCancellationOwner()
        self._stopping = False
        self.enabled = True
        self._attempts = 0
        self._suppression = OrderedDict()  # Bounded process-local session/incarnation history.
        self._status = None
        self.last_operation = None
        self._clock = monotonic if clock is None else clock
        self._active_session = None
        self._deadline = None
        self._periodic_seen = OrderedDict()
        self._checks = self._periodic_jobs = 0
        self._last_check = None

    def _start_thread(self):
        if self._thread is None:
            self._thread = Thread(target=self._run, name='akane-next-maintenance-commit', daemon=True)
            self._thread.start()

    def _arm_deadline(self):
        interval = self.policy.periodic_interval_seconds
        self._deadline = None
        if (self.enabled and self._active_session is not None and interval is not None
                and self.policy.segment_exchanges is not None and not self._stopping):
            self._deadline = self._clock() + interval
            self._start_thread()
        self._condition.notify_all()

    def activate(self, session):
        """Bind the timer to the explicitly selected session incarnation."""
        if session.profile_id != self.store.profile_id:
            raise ValueError('Maintenance session belongs to another profile.')
        with self._condition:
            self.cancel()
            self._active_session = session
            self._arm_deadline()

    def periodic_status(self):
        with self._condition:
            return {
                'interval_seconds': self.policy.periodic_interval_seconds,
                'next_check_monotonic': self._deadline,
                'checks': self._checks,
                'scheduled': self._periodic_jobs,
                'last_check': dict(self._last_check) if self._last_check is not None else None,
            }

    def check_periodic(self):
        """Evaluate one due window. The timed worker calls this same gate."""
        with self._condition:
            now = self._clock()
            if self._stopping or self._deadline is None or now < self._deadline:
                return None
            self._deadline = now + self.policy.periodic_interval_seconds
            active = self._active_session
            started = perf_counter()
            outcome = 'busy' if self._operation is not None else self._admit(
                active.conversation_id, trigger='periodic')
            self._checks += 1
            self._periodic_jobs += outcome == 'scheduled'
            self._last_check = {'session_id': active.conversation_id,
                                'outcome': outcome, 'seconds': perf_counter() - started}
            self._condition.notify_all()
            return outcome

    @staticmethod
    def _remember(records, key, value):
        records[key] = value
        records.move_to_end(key)
        if len(records) > 32:
            records.popitem(last=False)

    def status(self, session_id):
        with self._condition:
            status = self._status
            if status is None or status.session_id != session_id:
                return MaintenanceStatus(session_id, attempts=self._attempts)
            if self._operation is not None:
                job = self.scheduler.get_job(self.store.profile_id, self._operation.job_id)
                if job.status.name in ('PENDING', 'RUNNING'):
                    status = replace(status, state='queued' if job.status.name == 'PENDING' else 'running')
            return status

    def observe(self, session_id, info, *, context_size):
        if info is None:
            return
        # The prepared prompt precedes this reply. Reserve its maximum size, the
        # next reply, a future input allowance and headroom. No future-text claim.
        pressure = (max(info.requested_prompt_tokens, info.prompt_tokens)
                    + 2*info.output_tokens + self.policy.future_input_tokens
                    + self.policy.headroom_tokens) / context_size
        with self._condition:
            if self._stopping:
                return
            if self._operation is None and (self._status is None or self._status.session_id != session_id):
                self._status = MaintenanceStatus(session_id, attempts=self._attempts, pressure=pressure)
            if self._status is not None and self._status.session_id == session_id:
                self._status = replace(self._status, pressure=pressure)
            if self._operation is not None or not self.enabled:
                return
            self._admit(session_id, trigger='pressure', pressure=pressure)

    def _admit(self, session_id, *, trigger, pressure=None):
        """Run the shared admission path with the manager condition held."""
        periodic = trigger == 'periodic'
        if periodic and self._status is not None and self._status.session_id == session_id:
            pressure = self._status.pressure
        try:
            session = self.store.get_session(session_id)
            if periodic and session.created_at != self._active_session.created_at:
                self._deadline = None
                return 'session_replaced'
            incarnation = (session_id, session.created_at)
            checked_revision, tried_boundary = self._periodic_seen.get(incarnation, (-1, 0))
            if periodic:
                if session.revision == checked_revision:
                    return 'unchanged_revision'
                self._remember(self._periodic_seen, incarnation, (session.revision, tried_boundary))
            entries = self.store.transcript(session_id)
            # Preserve the latest completed pairs and cover intervening diagnostics.
            remaining = self.policy.compaction.keep_recent_exchanges
            for index in range(len(entries) - 2, -1, -2):
                if entries[index + 1].outcome == 'completed':
                    remaining -= 1
                if remaining == 0:
                    break
            if remaining or index == 0:
                return 'no_eligible_range'
            eligible = count = index // 2
            key = incarnation
            if self.policy.segment_exchanges is not None:
                manifest = self.compactor.summaries.get_segments(session_id)
                if count <= manifest.frontier:
                    return 'no_eligible_range'
                count = min(count, manifest.frontier + self.policy.segment_exchanges)
                key = (*key, manifest.frontier)

            prior_count, armed = self._suppression.get(key, (0, True))
            if periodic:
                # A single pass over a static backlog. After a failed attempt at
                # this frontier, require meaningful new eligible originals.
                if eligible <= tried_boundary:
                    return 'no_new_eligible_history'
                if prior_count and eligible - tried_boundary < self.policy.retry_advance_exchanges:
                    return 'retry_held'
            else:
                if pressure <= self.policy.low_water:
                    armed = True
                    if key in self._suppression:
                        self._suppression[key] = (prior_count, armed)
                if (pressure < self.policy.high_water or count <= prior_count
                        or (not armed and count - prior_count < self.policy.retry_advance_exchanges)):
                    return 'pressure_held'
            self._remember(self._suppression, key, (count, False))
            if self.policy.segment_exchanges is not None:
                self._remember(self._periodic_seen, incarnation, (session.revision, eligible))
            self._attempts += 1
            self._status = MaintenanceStatus(session_id, 'queued', source_revision=session.revision,
                                             boundary_exchanges=count, attempts=self._attempts,
                                             pressure=pressure, trigger=trigger)
            token = self._owner.begin(uuid4())
            try:
                if self.policy.segment_exchanges is not None:
                    operation = self.compactor.start_segment(session_id, cancellation=token,
                        max_exchanges=self.policy.segment_exchanges, expected_session=session,
                        expected_start=manifest.frontier)
                else:
                    operation = self.compactor.start(session_id, entries[index - 2].turn_id,
                                                     cancellation=token, include_diagnostics=True)
            except Exception:
                self._owner.finish(token.turn_id)
                raise
            self._operation = self.last_operation = operation
            self._status = replace(self._status, job_id=operation.job_id)
            self._start_thread()
            self._condition.notify_all()
            return 'scheduled'
        except SessionNotFound as error:
            if periodic:
                self._deadline = None
                return 'session_deleted'
            self._status = MaintenanceStatus(session_id, 'scheduling_failed', attempts=self._attempts,
                                             pressure=pressure, error=str(error), trigger=trigger)
        except Exception as error:
            self._status = MaintenanceStatus(session_id, 'scheduling_failed', attempts=self._attempts,
                                             pressure=pressure, error=f'{type(error).__name__}: {error}',
                                             trigger=trigger)
        return 'scheduling_failed'

    def _run(self):
        while True:
            with self._condition:
                while self._operation is None and not self._stopping:
                    self.check_periodic()
                    if self._operation is None:
                        timeout = None if self._deadline is None else min(
                            TIMEOUT_MAX, max(0, self._deadline - self._clock()))
                        self._condition.wait(timeout)
                if self._operation is None:
                    return
                operation = self._operation
            state, error = 'completed', None
            published = None
            try:
                published = self.compactor.finish(operation)
            except GenerationCancelled as exception:
                state, error = 'cancelled', str(exception)
            except ContinuityConflict as exception:
                state, error = 'revision_conflict', str(exception)
            except SessionNotFound as exception:
                state, error = 'source_deleted', str(exception)
            except OSError as exception:
                state, error = 'publication_failed', f'{type(exception).__name__}: {exception}'
            except Exception as exception:
                state, error = 'invalid_result', f'{type(exception).__name__}: {exception}'
            finally:
                with self._condition:
                    self._status = replace(self._status, state=state, error=error,
                                           boundary_exchanges=getattr(published, "end", self._status.boundary_exchanges))
                    self._owner.finish(operation.cancellation.turn_id)
                    self._operation = None
                    self._condition.notify_all()

    def request_retry(self, session_id):
        """An explicit development action, never called by failed model output."""
        with self._condition:
            if self._operation is not None or self._stopping:
                raise RuntimeError('Wait for maintenance to stop before authorizing a retry.')
            session = self.store.get_session(session_id)
            key = (session_id, session.created_at)
            if self.policy.segment_exchanges is not None:
                key = (*key, self.compactor.summaries.get_segments(session_id).frontier)
            self._suppression.pop(key, None)

    def set_enabled(self, enabled: bool):
        """Explicit development pause; summary selection remains automatic."""
        if type(enabled) is not bool:
            raise ValueError('Maintenance enablement must be Boolean.')
        with self._condition:
            if self.enabled == enabled:
                return
            self.enabled = enabled
            if not enabled:
                self.cancel()
            self._arm_deadline()

    def cancel(self):
        with self._condition:
            if self._operation is not None:
                self._owner.cancel(self._operation.cancellation.turn_id)
                self.scheduler.cancel_background(self.store.profile_id, self._operation.job_id)

    def wait_idle(self, *, timeout=None):
        """Diagnostic/test completion barrier; normal input never calls this."""
        with self._condition:
            if not self._condition.wait_for(lambda: self._operation is None, timeout):
                raise TimeoutError('Maintenance has not reached a safe completion checkpoint.')

    def close(self):
        with self._condition:
            self._stopping = True
            self._deadline = None
            self.cancel()
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join()  # Native unwind must precede model/store close.
