"""Transient goal activation and action selection. Nothing here executes actions."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from app.core.environment import EnvironmentChoice


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
    source_interest_id: str = ''


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

    environment: EnvironmentChoice | None = None
    suggested_choice_id: str = ''

    def to_json(self):
        return _json(self)


def _objective(frame, assessment):
    if 'subject' in assessment.missing_requirements:
        return None
    # Original record references identify the source, not generated prose goals.
    goals = sorted(frame.relevant_goals, key=lambda r: 'goal' not in dict(r.values))
    if goals:
        goal = goals[0]
        return ('continue', dict(goal.values)['topic'], goal.id, '', '', '')
    curiosity = next((r for r in frame.relevant_self if r.kind == 'curiosity'), None)
    if curiosity:
        return ('understand', dict(curiosity.values)['topic'], '', curiosity.id, '', '')
    task = next((r for r in frame.situation if frame.objective and r.id == frame.objective.id), None)
    if task:
        return ('continue', dict(task.values)['value'], '', '', task.id, '')
    # Developed positive interest can motivate a bounded investigation, only
    # where there is currently a useful information-bearing legal action.
    if any(e.eligible and e.information_gain > 0 for e in assessment.action_evaluations):
        interest = next((r for r in frame.relevant_self if r.kind == 'interest'
            and dict(r.values).get('stance') == 'positive'
            and dict(r.values).get('status') == 'active' and len({s.id for s in r.sources}) >= 2), None)
        if interest:
            return ('understand', dict(interest.values)['topic'], '', '', '', interest.id)
    if assessment.verification_needed:
        ids = {dict(r.values).get('entity_id') for r in frame.relevant_world
               if r.kind == 'world_current'} - {None, ''}
        if frame.percept.subject in ids:
            return ('verify', frame.percept.subject, '', '', '', '')
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
        kind, subject, source_goal, source_curiosity, source_situation, source_interest = objective
        for _, goal in self._goals:
            if (goal.profile_id, goal.objective_kind, goal.subject, goal.source_goal_id,
                goal.source_curiosity_id, goal.source_situation_id, goal.source_interest_id) == (frame.profile_id, *objective):
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
            source_interest_id=source_interest,
        )
        if len(_json(goal).encode('utf-8')) > ACTIVE_GOAL_BYTES:
            return None
        self._goals.append((time.monotonic(), goal))
        return goal

    def decide(self, frame, assessment, *, choice_id=None, suggested_choice_id=''):
        self._expire()
        goal = self._activate(frame, assessment)
        if frame.environment is not None:
            return self._environment_decision(frame, assessment, goal, choice_id, suggested_choice_id)
        if choice_id is not None:
            raise ValueError('Choice IDs apply only to a current virtual observation.')
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

    def _environment_decision(self, frame, assessment, goal, choice_id, suggestion):
        """Bind parameters to existing actions, with the same objective gate.

        Failed result evidence removes an automatic retry candidate. A trusted
        caller may explicitly retry a still-legal choice; WAIT always remains.
        """
        choices = frame.environment.choices
        if choice_id is not None and choice_id != 'wait' and choice_id not in {c.id for c in choices}:
            raise ValueError('Unknown or currently illegal virtual choice.')
        if not isinstance(suggestion, str) or choice_id is not None and suggestion:
            raise ValueError('Use an explicit choice or an advisory suggestion, not both.')
        selected, strategy_id, reasons = None, '', ['no_action_needed']
        evaluations = {e.choice_id: e for e in assessment.action_evaluations}
        if goal and choice_id != 'wait':
            if choice_id is not None:
                selected = next(c for c in choices if c.id == choice_id)
                reasons = ['explicit_choice', 'active_goal']
            else:
                eligible = [c for c in choices if c.id in evaluations and evaluations[c.id].eligible
                            and evaluations[c.id].rank > tuple(0 for _ in evaluations[c.id].rank)
                            and (goal.objective_kind != 'understand' or evaluations[c.id].information_gain > 0)]
                if eligible:
                    best = max(evaluations[c.id].rank for c in eligible)
                    tied = [c for c in eligible if evaluations[c.id].rank == best]
                    selected = next((c for c in tied if c.id == suggestion), tied[0])
                    assessment_row = evaluations[selected.id]
                    strategy_id = assessment_row.strategy_id
                    reasons = ['strategy' if r == 'learned_strategy' else r for r in assessment_row.reason_codes]
                    reasons.extend(('legal_choice', 'active_goal'))
        if suggestion and (selected is None or selected.id != suggestion):
            reasons.append('evidence_over_suggestion' if goal else 'no_objective')
        action = selected.action if selected else 'wait'
        decision = ActionDecision(frame.profile_id, frame.percept.id, assessment.frame_reference,
            goal.id if goal else '', _INTENTIONS[action], action,
            tuple(a for a in _ACTIONS if a == 'wait' or any(c.action == a for c in choices)),
            strategy_id, tuple(reasons), frame.created_at, selected, suggestion)
        return decision if len(decision.to_json().encode('utf-8')) <= DECISION_MAX_BYTES else None
