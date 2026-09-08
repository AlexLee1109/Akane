"""Transient goal activation and action selection. Nothing here executes actions."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace


ACTIVE_GOAL_LIMIT = 4
ACTIVE_GOAL_TOTAL_LIMIT = 16
ACTIVE_GOAL_TTL = 300
ACTIVE_GOAL_BYTES = 2048
ACTION_CANDIDATE_LIMIT = 5
DECISION_MAX_BYTES = 4096
_ACTIONS = ('verify', 'inspect', 'continue', 'respond', 'wait')
_STRATEGY_VERBS = {
    'inspect': 'inspect', 'check': 'inspect', 'review': 'inspect', 'measure': 'inspect',
    'verify': 'verify', 'validate': 'verify', 'confirm': 'verify',
    'continue': 'continue', 'resume': 'continue',
}
_INTENTIONS = {
    'verify': 'verify_evidence', 'inspect': 'inspect_context',
    'continue': 'pursue_objective', 'respond': 'respond_to_user', 'wait': 'await_input',
}


def _json(value):
    return json.dumps(asdict(value), ensure_ascii=False, allow_nan=False, separators=(',', ':'))


@dataclass(frozen=True, slots=True)
class ActiveGoal:
    id: str
    profile_id: str
    source_percept_id: str
    objective_kind: str
    subject: str
    source_goal_id: str
    source_curiosity_id: str
    source_situation_id: str
    status: str
    priority: str
    created_at: float
    expires_at: float


@dataclass(frozen=True, slots=True)
class ActionDecision:
    profile_id: str
    percept_id: str
    frame_reference: str
    active_goal_id: str
    intention: str
    selected_action: str
    candidate_actions: tuple[str, ...]
    strategy_id: str
    reason_codes: tuple[str, ...]
    created_at: float

    def to_json(self):
        return _json(self)


def _objective(frame, assessment):
    if 'subject' in assessment.missing_requirements:
        return None
    # Original record references identify the source, not generated prose goals.
    goals = sorted(frame.relevant_goals, key=lambda r: 'goal' not in dict(r.values))
    if goals:
        goal = goals[0]
        return ('continue', dict(goal.values)['topic'], goal.id, '', '')
    curiosity = next((r for r in frame.relevant_self if r.kind == 'curiosity'), None)
    if curiosity:
        return ('understand', dict(curiosity.values)['topic'], '', curiosity.id, '')
    task = next((r for r in frame.situation if frame.objective and r.id == frame.objective.id), None)
    if task:
        return ('continue', dict(task.values)['value'], '', '', task.id)
    if assessment.verification_needed:
        ids = {dict(r.values).get('entity_id') for r in frame.relevant_world
               if r.kind == 'world_current'} - {None, ''}
        if frame.percept.subject in ids:
            return ('verify', frame.percept.subject, '', '', '')
    return None


class ActionSelection:
    """Store lock owns access. Terminal records also occupy the bounded window."""

    def __init__(self):
        self._goals: list[tuple[float, ActiveGoal]] = []

    def _expire(self):
        now = time.monotonic()
        self._goals = [(at, goal) for at, goal in self._goals if now - at < ACTIVE_GOAL_TTL]

    def active(self, profile_id):
        self._expire()
        return tuple(g for _, g in self._goals if g.profile_id == profile_id and g.status == 'active')

    def clear(self, profile_id):
        self._goals = [(at, g) for at, g in self._goals if g.profile_id != profile_id]

    def finish(self, profile_id, goal_id, status):
        if status not in ('completed', 'dropped'):
            raise ValueError('Active goals may only be completed or dropped.')
        self._expire()
        for i, (at, goal) in enumerate(self._goals):
            if goal.profile_id == profile_id and goal.id == goal_id and goal.status == 'active':
                finished = replace(goal, status=status)
                self._goals[i] = (at, finished)
                return finished
        return None

    def _activate(self, frame, assessment):
        objective = _objective(frame, assessment)
        if objective is None:
            return None
        kind, subject, source_goal, source_curiosity, source_situation = objective
        for _, goal in self._goals:
            if (goal.profile_id, goal.objective_kind, goal.subject, goal.source_goal_id,
                goal.source_curiosity_id, goal.source_situation_id) == (frame.profile_id, *objective):
                return goal if goal.status == 'active' else None
        if (len(self._goals) >= ACTIVE_GOAL_TOTAL_LIMIT or
                sum(g.profile_id == frame.profile_id for _, g in self._goals) >= ACTIVE_GOAL_LIMIT):
            return None
        identity = json.dumps([frame.profile_id, objective, frame.percept.id, frame.created_at])
        goal = ActiveGoal(
            'active_' + hashlib.sha256(identity.encode()).hexdigest(),
            frame.profile_id, frame.percept.id, kind, subject, source_goal,
            source_curiosity, source_situation, 'active', frame.attention.priority,
            frame.created_at, frame.created_at + ACTIVE_GOAL_TTL,
        )
        if len(_json(goal).encode('utf-8')) > ACTIVE_GOAL_BYTES:
            return None
        self._goals.append((time.monotonic(), goal))
        return goal

    def decide(self, frame, assessment):
        self._expire()
        goal = self._activate(frame, assessment)
        available = {'wait'}
        if frame.percept.source == 'user':
            available.add('respond')
        if assessment.verification_needed:
            available.update(('verify', 'inspect'))
        elif assessment.mode == 'deliberate' or (goal and goal.objective_kind == 'understand'):
            available.add('inspect')
        if goal and not assessment.verification_needed and not assessment.missing_requirements:
            available.add('continue')

        strategy_id, preferred = '', ''
        for strategy in frame.relevant_strategies:
            values = dict(strategy.values)
            words = values.get('procedure', '').casefold().split()
            action = _STRATEGY_VERBS.get(words[0], '') if words else ''
            if (action in available and values.get('expected_result') == 'success'
                    and len({s.id for s in strategy.sources if s.kind == 'outcome'}) >= 2):
                strategy_id, preferred = strategy.id, action
                break
        reasons = []
        if assessment.verification_needed:
            selected, strategy_id = 'verify', ''
            reasons.append('verify_required')
        elif preferred:
            selected = preferred
            reasons.append('strategy')
        elif 'respond' in available:
            selected = 'respond'
            reasons.append('user_event')
        elif goal:
            selected = 'inspect' if goal.objective_kind == 'understand' else 'continue'
        else:
            selected = 'wait'
            reasons.append('no_action_needed')
        if goal:
            reasons.append('active_goal')
            if goal.source_situation_id:
                reasons.append('task')
        if assessment.missing_requirements:
            reasons.append('missing_state')
        decision = ActionDecision(
            frame.profile_id, frame.percept.id, assessment.frame_reference,
            goal.id if goal else '', _INTENTIONS[selected], selected,
            tuple(a for a in _ACTIONS if a in available), strategy_id,
            tuple(reasons), frame.created_at,
        )
        return decision if len(decision.to_json().encode('utf-8')) <= DECISION_MAX_BYTES else None
