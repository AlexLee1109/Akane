"""Bounded action/result linkage feeding the existing Phase 2 learning rules."""

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, replace

from app.core.mind import (
    SemanticEvidence, _resolve_prediction, _strategy_change_from_outcome,
    _tendency_change_from_outcome, find_behavioral_tendency,
)
from app.core.perception import _text
from app.core.state import Experience, Outcome, Prediction, PredictionChange, StateChangeProposal
from app.core.environment import EnvironmentChoice, action_occurrence
from app.core.utils import compact_text


EPISODE_PROFILE_LIMIT = 8
EPISODE_TOTAL_LIMIT = 32
EPISODE_TTL = 300
EPISODE_MAX_BYTES = 4096


@dataclass(frozen=True, slots=True)
class ActionEpisode:
    id: str
    profile_id: str
    decision_reference: str
    active_goal_id: str
    selected_action: str
    strategy_id: str
    source_percept_id: str
    source: str
    producer: str
    occurrence_id: str
    context: str
    procedure: str
    status: str
    created_at: float
    expires_at: float
    prediction: Prediction | None = None
    result_percept_id: str = ''
    outcome_id: str = ''
    environment: EnvironmentChoice | None = None


class EpisodeBook:
    """Store lock serializes access; terminal entries also bound replay state."""

    def __init__(self):
        self._items: list[tuple[float, ActionEpisode]] = []

    def _expire(self):
        now = time.monotonic()
        self._items = [(at, e) for at, e in self._items if now - at < EPISODE_TTL]

    def snapshot(self, profile_id):
        self._expire()
        return tuple(e for _, e in self._items if e.profile_id == profile_id)

    def clear(self, profile_id):
        self._items = [(at, e) for at, e in self._items if e.profile_id != profile_id]

    def cancel(self, profile_id, episode_ids):
        """Cancel only bound background attempts, without fabricating a result."""
        self._items = [(at, replace(e, status='cancelled')
                        if e.profile_id == profile_id and e.id in episode_ids and e.status == 'pending' else e)
                       for at, e in self._items]

    def start(self, decision, *, goal, strategy, source, producer, occurrence_id):
        self._expire()
        if (decision.selected_action not in {'inspect', 'verify', 'continue'}
                or not (goal or strategy)):
            return None
        if decision.environment is not None:
            occurrence_id = action_occurrence(decision.environment)
        key = (decision.profile_id, decision.percept_id, decision.selected_action,
               decision.active_goal_id, decision.strategy_id, decision.environment)
        for _, episode in self._items:
            same_decision = (episode.profile_id, episode.source_percept_id, episode.selected_action,
                             episode.active_goal_id, episode.strategy_id, episode.environment) == key
            same_occurrence = (episode.profile_id, episode.source, episode.producer, episode.occurrence_id) == (
                decision.profile_id, source, producer, occurrence_id)
            if same_decision or same_occurrence:
                return episode if same_decision and same_occurrence else None
        if (len(self._items) >= EPISODE_TOTAL_LIMIT or
                sum(e.profile_id == decision.profile_id for _, e in self._items) >= EPISODE_PROFILE_LIMIT):
            return None
        id = 'episode_' + uuid.uuid4().hex
        context = strategy.context if strategy else compact_text(goal.subject, 120)
        procedure = strategy.procedure if strategy else decision.selected_action
        if decision.environment is not None:
            context = 'virtual ' + decision.environment.activity
            procedure = decision.environment.procedure
        prediction = None
        if strategy:
            # Reuse the learned expectation/confidence; no invented probability
            # for actions without an existing learned expectation.
            prediction = Prediction(
                'prediction_' + id, decision.profile_id, procedure if decision.environment else decision.selected_action, id,
                strategy.expected_result, strategy.expected_result, strategy.confidence,
                'unresolved', decision.created_at, decision.created_at + EPISODE_TTL,
            )
        episode = ActionEpisode(
            id, decision.profile_id,
            hashlib.sha256(decision.to_json().encode()).hexdigest(), decision.active_goal_id,
            decision.selected_action, decision.strategy_id, decision.percept_id,
            source, producer, occurrence_id, context, procedure, 'pending',
            decision.created_at, decision.created_at + EPISODE_TTL, prediction,
            environment=decision.environment,
        )
        # Reserve space for bounded terminal IDs and prediction resolution fields.
        if len(json.dumps(asdict(episode), ensure_ascii=False).encode()) > EPISODE_MAX_BYTES - 512:
            return None
        self._items.append((time.monotonic(), episode))
        return episode

    def match(self, profile_id, episode_id, receipt, now):
        for episode in self.snapshot(profile_id):
            if (episode.id == episode_id and episode.status == 'pending'
                    and receipt.profile_id == profile_id and receipt.subject == episode.id
                    and (receipt.source, receipt.producer, receipt.source_id) == (
                        episode.source, episode.producer, episode.occurrence_id)
                    and episode.created_at <= receipt.timestamp <= now
                    and receipt.timestamp < episode.expires_at):
                return episode
        return None

    def finish(self, episode, receipt, *, outcome_id='', prediction=None):
        finished = replace(episode, status='resolved' if outcome_id else 'inconclusive',
                           result_percept_id=receipt.id, outcome_id=outcome_id,
                           prediction=prediction or episode.prediction)
        self._items = [(at, finished if e.id == episode.id else e) for at, e in self._items]
        return finished


def validate_producer(source, producer, occurrence_id):
    if (not isinstance(source, str) or source not in {'runtime', 'tool', 'task'}
            or not _text(producer) or not _text(occurrence_id)):
        raise ValueError('A trusted local source, producer and occurrence ID are required.')


def outcome_id(receipt):
    # Phase 4A ID already binds profile, source, kind, producer and occurrence.
    return 'outcome_' + receipt.id


def result_proposal(episode, receipt, result, *, strategies, tendencies, outcomes, now):
    """Adapt trusted linkage, then reuse Phase 2 mutations without changing rules.

    Schema-15's legacy *_turn_id fields hold namespaced action/feedback source
    references here. No synthetic conversation turns or assistant claims exist.
    """
    virtual = episode.environment is not None
    action = episode.procedure if virtual else episode.selected_action
    experience = Experience(
        'experience_' + receipt.id, episode.profile_id, 'task_' + result, 'akane',
        episode.context, f'In the virtual {episode.environment.activity}, Akane executed {action}.'[:200],
        action, receipt.content[:160], .7, 'grounded:virtual_environment', now,
        source_turn_ids=(episode.id, receipt.id),
    ) if virtual else None
    outcome = Outcome(
        outcome_id(receipt), episode.profile_id, 'task_' + result, result,
        action, episode.id, 1.0, 'grounded:virtual_environment' if virtual else 'grounded:action_episode', now,
        source_turn_ids=(episode.id, receipt.id),
    )
    if virtual:
        outcome = replace(outcome, description=receipt.content[:280])
    semantic = SemanticEvidence(
        outcome.result, episode.context, 'positive' if result == 'success' else 'negative',
        'task-local', result, action=action, action_turn_id=episode.id,
        strategy=episode.procedure,
        behavior=action if virtual else '', effect='verified virtual transition' if virtual else '',
    )
    # A revised/removed selected strategy must not transfer credit to another
    # procedure. New procedures can form only through the normal success rule.
    selected = next((s for s in strategies if s.id == episode.strategy_id), None)
    change = None
    if not episode.strategy_id or (selected and (selected.context, selected.procedure) == (
            episode.context, episode.procedure)):
        change = _strategy_change_from_outcome(
            episode.profile_id, strategies, outcomes, outcome, semantic, now=now,
        )
    tendency = find_behavioral_tendency(tendencies, episode.context, action)
    tendency_change = None
    if tendency or virtual:
        tendency_change = _tendency_change_from_outcome(
            episode.profile_id, tendencies, outcomes, outcome,
            replace(semantic, behavior=tendency.behavior, effect=tendency.expected_effect) if tendency else semantic, now=now,
        )
    predictions = ()
    resolved = None
    if episode.prediction:
        resolved = _resolve_prediction(episode.prediction, outcome, now=now)
        predictions = (PredictionChange('form', episode.prediction), PredictionChange('resolve', resolved))
    return StateChangeProposal(
        episode.profile_id, outcomes=(outcome,), predictions=predictions,
        experiences=(experience,) if experience else (),
        strategies=(change,) if change else (),
        behavioral_tendencies=(tendency_change,) if tendency_change else (),
        origin='action_episode',
    ), resolved
