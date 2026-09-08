"""Sparse runtime coordination over Phase 4 APIs; no model or executor access."""

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, replace

from app.core.perception import _text
from app.core.config import SETTINGS
from app.core.situation import current_situation
from app.core.temporal import temporal_context, time_relevant


PENDING_PROFILE_LIMIT = 16
PENDING_TOTAL_LIMIT = 64
CYCLE_PROFILE_LIMIT = 16
CYCLE_TOTAL_LIMIT = 64
CYCLE_TTL = 300
# No autonomous generation path exists: the enforceable budget is zero.
AUTONOMOUS_MODEL_CALLS_PER_CYCLE = 0
AUTONOMOUS_MODEL_CALLS_PER_WINDOW = 0


@dataclass(frozen=True, slots=True)
class CognitiveCycle:
    id: str
    profile_id: str
    percept_id: str
    trigger_kind: str
    started_at: float
    finished_at: float
    status: str
    attention_reference: str = ''
    frame_reference: str = ''
    assessment_reference: str = ''
    mode: str = ''
    active_goal_id: str = ''
    selected_action: str = ''
    strategy_id: str = ''
    episode_id: str = ''
    model_reasoning_requested: bool = False
    communication_requested: bool = False


@dataclass(frozen=True, slots=True)
class _Trigger:
    percept: object
    result: str = ''
    completed_goal_id: str = ''


@dataclass(frozen=True, slots=True)
class _Deadline:
    profile_id: str
    id: str
    at: float
    subject: str
    goal_id: str


class CognitionCoordinator:
    """One sleeping worker, bounded inputs, no polling or outbound side effects.

    Lock order is Store → condition. The worker never waits for Store while
    holding condition. Producers only enqueue; cognition never runs on admission.
    """

    def __init__(self, store, *, clock=time.time, timezone=SETTINGS.timezone):
        self.store = store
        self.clock = clock
        self.timezone = timezone
        temporal_context(now=clock(), timezone=timezone)  # trusted configuration
        self._condition = threading.Condition()
        self._pending = []
        self._cycles = []
        self._deadlines = []
        self._foreground = 0
        self._preempt = threading.Event()
        self._closed = False
        self._running = None
        self._worker = threading.Thread(target=self._run, name='akane-cognition', daemon=True)

    def start(self):
        self._worker.start()

    def _notify(self, percept):
        with self._condition:
            if self._closed:
                return True
            if any(t.percept.id == percept.id for t in self._pending):
                return True
            # Only identical runtime observations coalesce. World transitions,
            # user turns and tool/task results keep their occurrence identities.
            if percept.source == 'runtime' and percept.producer != 'temporal':
                for old in self._pending:
                    p = old.percept
                    if not old.result and (p.profile_id, p.source, p.producer, p.subject, p.content) == (
                            percept.profile_id, percept.source, percept.producer, percept.subject, percept.content):
                        # 4B retains the first identical observation as its
                        # salience representative. Keep that eligible occurrence.
                        self._condition.notify_all()
                        return True
            if (len(self._pending) >= PENDING_TOTAL_LIMIT or
                    sum(t.percept.profile_id == percept.profile_id for t in self._pending) >= PENDING_PROFILE_LIMIT):
                return False  # tool/task admission reports retryable backpressure
            self._pending.append(_Trigger(percept))
            self._condition.notify_all()
            return True

    def foreground_started(self):
        self._preempt.set()  # no Store/model lock acquisition on foreground ingress
        with self._condition:
            self._foreground += 1
            self._preempt.set()
            self._condition.notify_all()

    def foreground_finished(self):
        with self._condition:
            self._foreground = max(0, self._foreground - 1)
            if not self._foreground:
                self._preempt.clear()
            self._condition.notify_all()

    def cycles(self, profile_id):
        with self._condition:
            now = time.monotonic()
            self._cycles = [(at, c) for at, c in self._cycles if now - at < CYCLE_TTL]
            return tuple(c for _, c in self._cycles if c.profile_id == profile_id)

    def pending_count(self, profile_id=None):
        with self._condition:
            return sum(profile_id is None or t.percept.profile_id == profile_id for t in self._pending)

    def clear(self, profile_id):
        # Store lock prevents a cleared profile's in-flight cycle publishing later.
        with self._condition:
            self._pending = [t for t in self._pending if t.percept.profile_id != profile_id]
            self._cycles = [(at, c) for at, c in self._cycles if c.profile_id != profile_id]
            self._deadlines = [d for d in self._deadlines if d.profile_id != profile_id]
            self._condition.notify_all()

    def cancel(self, profile_id):
        self.foreground_started()
        try:
            with self.store._lock:
                self.clear(profile_id)
        finally:
            self.foreground_finished()

    def close(self):
        with self._condition:
            self._closed = True
            self._preempt.set()
            self._pending.clear()
            self._deadlines.clear()
            self._condition.notify_all()
        if threading.current_thread() is not self._worker:
            self._worker.join()
        with self._condition:
            self._cycles.clear()

    def wait_idle(self, timeout=5):
        """Synchronization for adapters/tests; production foreground never waits."""
        with self._condition:
            return self._condition.wait_for(lambda: not self._pending and self._running is None, timeout)

    def wake(self):
        """Trusted clock-change notification; does not synthesize a clock event."""
        with self._condition:
            self._condition.notify_all()

    def schedule(self, profile_id, deadline_id, *, at, subject, goal_id=''):
        """Trusted explicit deadline only. No periodic schedule or text parsing."""
        now = self.clock()
        temporal_context(now=at, timezone=self.timezone)
        if (not all(_text(v) for v in (profile_id, deadline_id, subject))
                or not _text(goal_id, optional=True) or at < now or at - now > 86400):
            raise ValueError('Invalid explicit deadline.')
        with self._condition:
            if self._closed:
                return False
            if any((d.profile_id, d.id) == (profile_id, deadline_id) for d in self._deadlines):
                return False
            if (len(self._deadlines) >= PENDING_TOTAL_LIMIT or
                    sum(d.profile_id == profile_id for d in self._deadlines) >= PENDING_PROFILE_LIMIT):
                return False
            self._deadlines.append(_Deadline(profile_id, deadline_id, at, subject, goal_id))
            self._condition.notify_all()
            return True

    def receive_result(self, profile_id, episode_id, *, source, producer, occurrence_id,
                       content, timestamp, result, completed_goal_id=''):
        """Trusted observed result code; never inferred from receipt text.

        Admission and attachment are atomic with respect to the worker. Full
        queues reject before admission; the producer must retry, never assume success.
        """
        if result not in ('success', 'failure', 'inconclusive') or not _text(completed_goal_id, optional=True):
            raise ValueError('Invalid observed result.')
        with self.store._lock:
            with self._condition:
                if self._closed:
                    raise RuntimeError('Cognition is closed.')
            p = self.store.record_percept_receipt(profile_id, source=source, producer=producer,
                source_id=occurrence_id, subject=episode_id, content=content, timestamp=timestamp)
            if p is None:
                p = next((p for p in self.store.percepts(profile_id)
                          if (p.source, p.producer, p.source_id, p.subject) ==
                          (source, producer, occurrence_id, episode_id)), None)
                episode = next((e for e in self.store.action_episodes(profile_id)
                                if e.id == episode_id and e.status == 'pending'), None)
                if p is None or episode is None:
                    return None
                with self._condition:
                    if not any(t.percept.id == p.id for t in self._pending):
                        if not self._notify(p):
                            raise BufferError('Cognition queue full; retry this receipt later.')
            with self._condition:
                self._pending = [replace(t, result=result, completed_goal_id=completed_goal_id)
                                 if t.percept.id == p.id else t for t in self._pending]
            return p

    def awareness(self, profile_id):
        clock = temporal_context(now=self.clock(), timezone=self.timezone)
        with self.store._lock:
            situation = current_situation(self.store.world(profile_id), profile_id, clock.now)
            goals = self.store.active_goals(profile_id)
            episodes = tuple(e for e in self.store.action_episodes(profile_id)
                             if e.status == 'pending' and clock.now < e.expires_at)
            percepts = self.store.percepts(profile_id)
            cycles = self.cycles(profile_id)
            meaningful = next((c for c in reversed(cycles) if c.attention_reference), None)
            waiting_cycle = next((c for c in reversed(cycles)
                                  if any(e.id == c.episode_id for e in episodes)), None)
            with self._condition:
                running = self._running
            current = running if running and running.profile_id == profile_id else None
            return dict(
                temporal=clock, situation=tuple(situation.values()),
                active_goals=tuple(g for g in goals if clock.now < g.expires_at),
                awaiting_result=bool(episodes), episodes=episodes,
                current_percept_id=current.id if current else '',
                last_meaningful_percept_id=meaningful.percept_id if meaningful else '',
                mode=waiting_cycle.mode if waiting_cycle else '',
                last_interaction_age=clock.age(self.store.last_interaction_at(profile_id)),
                percept_ages=tuple((p.id, clock.age(p.timestamp)) for p in percepts),
                goal_ages=tuple((g.id, clock.age(g.created_at)) for g in goals if clock.now < g.expires_at),
                awaiting_result_ages=tuple((e.id, clock.age(e.created_at)) for e in episodes),
                situation_ages=tuple((slot, clock.age(s.updated_at)) for slot, s in situation.items()),
            )

    def _publish(self, cycle):
        with self._condition:
            if self._closed:
                return
            own = [c for _, c in self._cycles if c.profile_id == cycle.profile_id]
            if len(own) >= CYCLE_PROFILE_LIMIT:
                self._cycles = [(at, c) for at, c in self._cycles if c.id != own[0].id]
            self._cycles.append((time.monotonic(), cycle))
            self._cycles = self._cycles[-CYCLE_TOTAL_LIMIT:]

    def _cycle(self, trigger):
        p = trigger.percept
        now = self.clock()
        cycle = CognitiveCycle('cycle_' + uuid.uuid4().hex, p.profile_id, p.id,
                               'deadline' if p.producer == 'temporal' else p.source,
                               now, now, 'ignored')
        if trigger.result:
            resolved = self.store.resolve_action_episode(p.profile_id, p.subject, p.id,
                result=trigger.result, completed_goal_id=trigger.completed_goal_id, now=now)
            return replace(cycle, status='completed' if resolved else 'ignored',
                           episode_id=resolved.id if resolved else '', finished_at=self.clock())
        if not any(a.percept_id == p.id for a in self.store.attention(p.profile_id, now=now)):
            return cycle
        if self._preempt.is_set():
            return None
        temporal = ()
        if p.producer == 'temporal' or time_relevant(p.content):
            clock = temporal_context(now=now, timezone=self.timezone)
            temporal = (*clock.values(), ('event_age_seconds', str(clock.age(p.timestamp))),
                        ('last_interaction_age_seconds', str(clock.age(self.store.last_interaction_at(p.profile_id)))))
            temporal += tuple(('goal_age:' + g.id, str(clock.age(g.created_at)))
                              for g in self.store.active_goals(p.profile_id) if now < g.expires_at)
        step = self.store.cognition_step(p.profile_id, p.id, now=now, temporal=temporal)
        if step is None:
            return cycle
        frame, assessment, decision = step
        if decision is None:
            return replace(cycle, status='attended', attention_reference=p.id)
        if self._preempt.is_set():
            return None
        episode = None
        # User turns already have a delivery path. Never duplicate it or turn a
        # foreground response into background execution, even if VERIFY wins.
        if p.source != 'user' and decision.selected_action in {'inspect', 'verify', 'continue'}:
            episode = self.store._track_action_decision(decision,
                source='task', producer='cognition-handoff', occurrence_id=cycle.id)
        return replace(cycle,
            status='awaiting_result' if episode else 'completed',
            attention_reference=p.id, frame_reference=assessment.frame_reference,
            assessment_reference=hashlib.sha256(assessment.to_json().encode()).hexdigest(),
            mode=assessment.mode, active_goal_id=decision.active_goal_id,
            selected_action=decision.selected_action, strategy_id=decision.strategy_id,
            episode_id=episode.id if episode else '', finished_at=self.clock(),
            model_reasoning_requested=(p.source != 'user' and bool(decision.active_goal_id)
                and assessment.mode in {'deliberate', 'verify'} and decision.selected_action != 'wait'),
            communication_requested=p.source != 'user' and decision.selected_action == 'respond')

    def _run(self):
        while True:
            with self._condition:
                while not self._closed:
                    now = self.clock()
                    due = next((d for d in self._deadlines if d.at <= now), None)
                    if not self._foreground and (self._pending or due):
                        break
                    delay = None if self._foreground or not self._deadlines else max(0, min(d.at for d in self._deadlines) - now)
                    self._condition.wait(delay)
                if self._closed:
                    return
            # Store is acquired without holding condition; foreground may enter
            # immediately, and the flag is checked again before any work starts.
            with self.store._lock:
                with self._condition:
                    if self._closed:
                        return
                    if self._foreground:
                        continue
                    now = self.clock()
                    due = next((d for d in self._deadlines if d.at <= now), None)
                    if due and len(self._pending) < PENDING_TOTAL_LIMIT and sum(t.percept.profile_id == due.profile_id for t in self._pending) < PENDING_PROFILE_LIMIT:
                        self._deadlines.remove(due)
                        if not due.goal_id or any(g.id == due.goal_id and now < g.expires_at for g in self.store.active_goals(due.profile_id)):
                            self.store.record_percept_receipt(due.profile_id, source='runtime', producer='temporal',
                                source_id=due.id, subject=due.subject, content='Deadline reached: ' + due.subject, timestamp=now)
                    if not self._pending:
                        continue
                    trigger = self._pending[0]
                    self._running = trigger.percept
                try:
                    cycle = self._cycle(trigger)
                    if cycle is not None:
                        self._publish(cycle)
                        with self._condition:
                            self._pending = [t for t in self._pending if t.percept.id != trigger.percept.id]
                except Exception:
                    # Fail closed: no automatic retries, model calls or fabricated
                    # outcome. A failed result can be explicitly retried by adapter.
                    self._publish(CognitiveCycle('cycle_' + uuid.uuid4().hex,
                        trigger.percept.profile_id, trigger.percept.id, trigger.percept.source,
                        now, self.clock(), 'cancelled'))
                    with self._condition:
                        self._pending = [t for t in self._pending if t.percept.id != trigger.percept.id]
                finally:
                    with self._condition:
                        self._running = None
                        self._condition.notify_all()
