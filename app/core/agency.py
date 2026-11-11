"""Grounded agency: relevant events, goals, legal choices and verified consequences.

One event context feeds one evidence check and one action choice. Communication
admission shares that decision; it cannot create a second objective or execute an
unsupported action. Store owns locking and atomic persistence; the scheduler owns
waiting/preemption, and environment adapters own observable truth.
"""

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, replace

from app.core.learning import (
    SELF_ESTABLISHED_EVIDENCE, SemanticEvidence, _reason_semantics,
    _resolve_prediction, _strategy_change_from_outcome, _tendency_change_from_outcome,
    behavioral_tendency_state, curiosity_state, find_behavioral_tendency,
    self_development_state, self_topic_terms,
)
from app.core.perception import Percept, _text, PERCEPT_CONTENT_BYTES
from app.core.knowledge import world_records
from app.core.situation import current_situation
from app.core.state import (
    Experience, Outcome, Prediction, PredictionChange, StateChangeProposal,
    WorldEvent, WorldSnapshot, WorldSource,
)
from app.core.environment import (
    PRODUCER as ENVIRONMENT_PRODUCER, EnvironmentChoice, EnvironmentObservation,
    action_occurrence, goal_targets, information_gain, learning_context, task_progress,
)
from app.core.utils import compact_text, lexical_terms


ATTENTION_LIMIT = 16
# Grammar and generic judgment/action words cannot establish topic relevance.
_STOP = self_topic_terms(
    'the this that these those is are was were be been have has had with from '
    'into about your their our you they it its my me we us and but for not '
    'like love prefer want think interest goal task result event status '
    'do does did doing make improve complete completed talking engaged'
)
_REASONS = ('goal', 'focus', 'self', 'novelty', 'surprise', 'curiosity', 'task', 'deadline')


@dataclass(frozen=True, slots=True)
class AttentionCandidate:
    percept_id: str
    profile_id: str
    priority: str
    salience: int
    reason_codes: tuple[str, ...]
    created_at: float


def _terms(text):
    return self_topic_terms(text.replace('_', ' ')) - _STOP


def _relevant(anchor, terms, entity_terms):
    overlap = anchor & terms
    # A lone term from a multiword topic needs a known World entity match.
    return bool(overlap) and (
        anchor <= terms or len(overlap) >= 2
        or any(label and label <= overlap for label in entity_terms)
    )


def _evaluate_attention(profile_id, percepts, *, world, self_items, tendencies,
                        curiosities, goals, predictions, outcomes, now, deadline_ids=()):
    """Internal projection. Inputs come exclusively from the locked Store.

    The public boundary accepts a profile, never supplied Percepts or state.
    Equal-priority candidates keep percept admission order. No evaluation history
    is retained; repeated reads are stable views, not consume/ack operations.
    """
    entities = {e.id: e.label for e in world.entities if e.profile_id == profile_id}
    events = {e.id: e for e in world.events if e.profile_id == profile_id}
    own = lambda rows: (r for r in rows if r.profile_id == profile_id)
    anchors = {code: [] for code in _REASONS}

    def add(code, *texts):
        anchors[code].extend(terms for text in texts if (terms := _terms(text)))

    for item in own(self_items):
        if item.status != 'active':
            continue
        if item.kind == 'goal':
            add('goal', item.topic, item.value)
        elif self_development_state(item) in {'reinforced', 'established'}:
            add('self', item.topic, item.value)
    for item in own(tendencies):
        if item.status == 'active' and behavioral_tendency_state(item) == 'reinforced':
            add('self', item.context)
    for item in own(curiosities):
        if item.status == 'active' and curiosity_state(item) == 'reinforced':
            add('curiosity', item.topic, item.focus)
    for item in own(goals):
        if item.status == 'active':
            add('goal', item.topic, item.goal)
    for slot, item in current_situation(world, profile_id, now).items():
        if slot == 'focus':
            add('focus', item.value)
        elif slot in {'activity', 'intention'}:
            add('task', item.value)

    # Surprise is linked to the actual outcome's source user turn. A matching
    # topic, tool name or receipt ID alone cannot impersonate an outcome.
    outcome_map = {o.id: o for o in own(outcomes)}
    surprise_turns = set()
    for prediction in own(predictions):
        outcome = outcome_map.get(prediction.outcome_id)
        if (prediction.status == 'resolved' and prediction.resolved_at <= now
                and prediction.error_category in {'negative_error', 'positive_surprise'}
                and outcome is not None and outcome.created_at <= now
                and outcome.action_turn_id == prediction.action_turn_id):
            surprise_turns.update(outcome.source_turn_ids)

    entity_terms = [_terms(label) for label in entities.values()]

    candidates = []
    previous = {}
    for percept in percepts:
        if percept.profile_id != profile_id or percept.timestamp > now:
            continue
        event = events.get(percept.event_id) if percept.source == 'world' else None
        novelty = bool(event and event.created_at <= now
                       and event.kind in {'state_transition', 'relation_transition'}
                       and event.before != event.after)
        if percept.source == 'runtime' and percept.producer == ENVIRONMENT_PRODUCER:
            novelty = True  # A trusted version-bound offer can still produce WAIT.
        surprise = percept.source == 'user' and percept.source_id in surprise_turns
        key = (percept.source, percept.producer,
               percept.world_id or percept.subject)
        signature = ' '.join(percept.content.casefold().split())
        repeated = previous.get(key) == signature
        previous[key] = signature
        if repeated and not (novelty or surprise):
            continue
        labels = [entities.get(percept.subject, '')]
        if event:
            labels.extend(entities.get(id, '') for id in event.entity_ids)
        terms = _terms(' '.join((percept.content, percept.subject, *labels)))
        reasons = {code for code, values in anchors.items()
                   if any(_relevant(anchor, terms, entity_terms) for anchor in values)}
        if percept.source == 'runtime' and percept.producer == 'temporal' and percept.source_id in deadline_ids:
            reasons.add('deadline')
        if novelty:
            reasons.add('novelty')
        if surprise:
            reasons.add('surprise')
        if not reasons:
            continue
        salience = min(3, len(reasons))
        candidates.append(AttentionCandidate(
            percept.id, profile_id, ('low', 'medium', 'high')[salience - 1],
            salience, tuple(code for code in _REASONS if code in reasons),
            percept.timestamp,
        ))
    # Store filters these to current revisions. At most three explicit sandbox
    # offers reserve admission before old projections; this makes a purpose-free
    # WAIT decision available even when historical events fill the window.
    offers = {p.id for p in percepts if p.source == 'runtime' and p.producer == ENVIRONMENT_PRODUCER}
    return tuple(sorted(candidates, key=lambda c: (c.percept_id not in offers, -c.salience))[:ATTENTION_LIMIT])


FRAME_MAX_BYTES = 6144
FRAME_RECORD_BYTES = 1024
FRAME_LIMITS = {
    'situation': 4, 'relevant_world': 4, 'relevant_self': 3,
    'relevant_experience': 1,
    'relevant_strategies': 2, 'relevant_goals': 2, 'linked_evidence': 2,
}


@dataclass(frozen=True, slots=True)
class FrameRecord:
    kind: str
    id: str
    values: tuple[tuple[str, str], ...]
    timestamp: float
    sources: tuple[WorldSource, ...]


@dataclass(frozen=True, slots=True)
class ActionHistory:
    """Read-only summary of distinct verified executions in comparable conditions."""
    choice_id: str
    successes: int = 0
    failures: int = 0
    latest_result: str = ''
    strategy_id: str = ''
    tendency_result: str = ''
    sources: tuple[WorldSource, ...] = ()


def action_histories(observation, row, outcomes, strategies, tendencies):
    receipts = {r['episode_id']: r for r in row['receipts']}
    histories = []
    for choice in observation.choices:
        related, actions = [], set()
        context = learning_context(choice, row['state'])
        for outcome in reversed(outcomes):
            proof = receipts.get(outcome.action_turn_id)
            if (proof is None or outcome.action_turn_id in actions
                    or outcome.reason != 'grounded:virtual_environment' or outcome.action != choice.procedure
                    or outcome.result != 'task_' + proof['result']
                    or outcome.source_turn_ids[-1] != proof['percept_id']
                    or learning_context(choice, proof['before']) != context):
                continue
            related.append(outcome)
            actions.add(outcome.action_turn_id)
        confirmed = {o.id for o in related}
        strategy = next((s for s in strategies if s.status == 'active' and s.procedure == choice.procedure
                         and len(confirmed.intersection(s.supporting_outcome_ids)) >= 2), None)
        tendency = next((t for t in tendencies if t.status == 'active' and t.behavior == choice.procedure
                         and len(confirmed.intersection(t.supporting_outcome_ids)) >= 2), None)
        histories.append(ActionHistory(choice.id,
            sum(o.result == 'task_success' for o in related),
            sum(o.result == 'task_failure' for o in related),
            related[0].result if related else '', strategy.id if strategy else '',
            tendency.expected_result if tendency else '',
            tuple(WorldSource('outcome', o.id) for o in related[:2])))
    return tuple(histories)


@dataclass(frozen=True, slots=True)
class EventContext:
    profile_id: str
    percept: Percept
    attention: AttentionCandidate
    created_at: float
    objective: WorldSource | None = None
    situation: tuple[FrameRecord, ...] = ()
    relevant_world: tuple[FrameRecord, ...] = ()
    relevant_self: tuple[FrameRecord, ...] = ()
    relevant_experience: tuple[FrameRecord, ...] = ()
    relevant_strategies: tuple[FrameRecord, ...] = ()
    relevant_goals: tuple[FrameRecord, ...] = ()
    linked_evidence: tuple[FrameRecord, ...] = ()
    temporal: tuple[tuple[str, str], ...] = ()
    environment: EnvironmentObservation | None = None
    action_history: tuple[ActionHistory, ...] = ()

    def to_json(self):
        """Canonical wire representation used for the hard UTF-8 byte bound."""
        return _json(self)


def _json(value):
    return json.dumps(asdict(value), ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'))


def _percept_query(percept, world):
    ids = {percept.subject}
    for event in world.events:
        if event.id == percept.event_id:
            ids.update(event.entity_ids)
    # Labels enrich IDs; never replace observed content with current World values.
    labels = [e.label for e in world.entities if e.id in ids]
    return ' '.join((percept.content, percept.subject, *labels))


def _record(item, kind, fields, *, sources=None, extra=()):
    return FrameRecord(
        kind, item.id, tuple((field, getattr(item, field)) for field in fields) + extra,
        getattr(item, 'updated_at', item.created_at),
        item.sources if sources is None else sources,
    )


def _build_frame(percept, attention, *, query, state, world, predictions, outcomes, now,
                 environment=None, linked_evidence=(), action_history=()):
    """Only Store calls this, after validating current attention and ownership."""
    frame = EventContext(percept.profile_id, percept, attention, now,
                           environment=environment, linked_evidence=linked_evidence, action_history=action_history)
    frame_bytes = len(frame.to_json().encode('utf-8'))
    if frame_bytes > FRAME_MAX_BYTES:
        return None  # Required provenance is never truncated to make it fit.
    terms = _terms(query)
    entities = {e.id: e.label for e in world.entities if e.profile_id == percept.profile_id}
    entity_terms = [_terms(label) for label in entities.values()]

    def related(*texts):
        return any(_relevant(_terms(text), terms, entity_terms) for text in texts)

    def add(category, record, *, objective=False):
        nonlocal frame, frame_bytes
        records = getattr(frame, category)
        if len(records) >= FRAME_LIMITS[category]:
            return
        record_bytes = len(_json(record).encode('utf-8'))
        if record_bytes > FRAME_RECORD_BYTES:
            return
        # Canonical JSON arrays add only a comma and the new record. Account
        # for this exactly instead of serializing the growing frame per item.
        added = record_bytes + bool(records)
        objective_source = WorldSource(record.kind, record.id) if objective and frame.objective is None else None
        if objective_source is not None:
            added += len(_json(objective_source).encode('utf-8')) - len('null')
        if frame_bytes + added > FRAME_MAX_BYTES:
            return
        frame = replace(frame, **{category: (*records, record)})
        if objective_source is not None:
            frame = replace(frame, objective=objective_source)
        frame_bytes += added

    # Snapshot ranking remains authoritative; direct relevance removes broad
    # preference fallbacks intended for foreground conversation, not events.
    for item in (*state.developmental_goals, *(s for s in state.self_items if s.kind == 'goal')):
        value_field = 'goal' if hasattr(item, 'goal') else 'value'
        if item.status == 'active' and related(item.topic, getattr(item, value_field)):
            add('relevant_goals', _record(item, 'goal', ('topic', value_field),
                sources=tuple(WorldSource('evidence', id) for id in item.source_ids)),
                objective='goal' in attention.reason_codes)

    situation = current_situation(world, percept.profile_id, now)
    for slot in ('intention', 'activity', 'focus', 'status'):
        item = situation.get(slot)
        if item and related(item.value):
            add('situation', _record(item, 'situation', ('attribute', 'value')),
                objective=slot in {'intention', 'activity'})
    if frame.objective is None and frame.relevant_goals:
        goal = frame.relevant_goals[0]
        source = WorldSource(goal.kind, goal.id)
        added = len(_json(source).encode('utf-8')) - len('null')
        if frame_bytes + added <= FRAME_MAX_BYTES:
            frame = replace(frame, objective=source)
            frame_bytes += added

    # Legal event choices consume current state and transitions, not unparsed
    # fact/relation prose. Keep the richer knowledge graph for foreground Qwen.
    event_world = WorldSnapshot(entities=world.entities, states=world.states, events=world.events)
    for item in world_records(event_world, percept.profile_id, query, now=now):
        if isinstance(item, WorldEvent):
            fields = ('kind', 'attribute', 'before', 'after')
            labels = ', '.join(entities.get(id, id) for id in item.entity_ids)
            kind = 'world_history'
        else:
            fields = tuple(f for f in ('entity_id', 'subject_id', 'attribute',
                'predicate', 'value', 'relation', 'object_id') if hasattr(item, f))
            labels = entities.get(getattr(item, 'entity_id', getattr(item, 'subject_id', '')), '')
            kind = 'world_current'
        add('relevant_world', _record(item, kind, fields, extra=(('entity', labels),)))

    for item in state.self_items:
        if (item.kind != 'goal' and item.status in {'active', 'uncertain'}
                and related(item.topic, item.value)):
            stance, preferred = _reason_semantics(item.reason)
            add('relevant_self', _record(item, item.kind, ('topic', 'value'),
                sources=tuple(WorldSource('evidence', id) for id in item.source_ids),
                extra=(('stance', stance), ('preferred', preferred), ('status', item.status),
                       ('strength', str(item.strength)), ('confidence', str(item.confidence)))))
    for collection, kind, fields, refs in (
        (state.curiosities, 'curiosity', ('topic', 'focus'), 'source_ids'),
        (state.behavioral_tendencies, 'tendency', ('context', 'behavior', 'expected_effect', 'expected_result'), 'supporting_outcome_ids'),
    ):
        for item in collection:
            if item.status == 'active' and related(*(getattr(item, f) for f in fields[:2])):
                add('relevant_self', _record(item, kind, fields,
                    sources=tuple(WorldSource('evidence', id) for id in getattr(item, refs))))
    for item in state.strategies:
        if item.status == 'active' and related(item.context, item.procedure):
            add('relevant_strategies', _record(item, 'strategy', ('context', 'procedure', 'expected_result'),
                sources=tuple(WorldSource('outcome', id) for id in item.supporting_outcome_ids)))
    for item in state.experiences:
        if related(item.topic, item.what_happened):
            add('relevant_experience', _record(item, 'experience', ('topic', 'what_happened', 'outcome'),
                sources=tuple(WorldSource('turn', id) for id in item.source_turn_ids)))
    if 'surprise' in attention.reason_codes:
        for prediction in predictions:
            outcome = next((o for o in outcomes if o.id == prediction.outcome_id
                            and percept.source_id in o.source_turn_ids), None)
            if (prediction.status == 'resolved' and prediction.resolved_at <= now
                    and prediction.error_category in {'negative_error', 'positive_surprise'}
                    and outcome is not None and outcome.created_at <= now):
                add('linked_evidence', _record(prediction, 'prediction',
                    ('action', 'expected_result', 'actual_result', 'error_category'),
                    sources=(WorldSource('outcome', outcome.id),)))
                add('linked_evidence', _record(outcome, 'outcome', ('action', 'result', 'description'),
                    sources=tuple(WorldSource('turn', id) for id in outcome.source_turn_ids)))
                break
    return frame


ASSESSMENT_MAX_BYTES = 4096


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    choice_id: str
    eligible: bool
    goal_progress: float
    self_alignment: float
    competence: float
    information_gain: float
    uncertainty_reduction: float
    novelty: float
    strategy_id: str = ''
    evidence: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @property
    def rank(self):
        # Explicit objectives, then owned judgments, then demonstrated skill,
        # then useful information. No tuned reward weights or personality prior.
        return (self.goal_progress, self.self_alignment, self.competence,
                self.information_gain, self.uncertainty_reduction, self.novelty)


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    profile_id: str
    percept_id: str
    frame_reference: str
    mode: str
    confidence: str
    uncertainty_codes: tuple[str, ...]
    conflict_codes: tuple[str, ...]
    missing_requirements: tuple[str, ...]
    verification_needed: bool
    reason_codes: tuple[str, ...]
    created_at: float
    action_evaluations: tuple[ActionEvaluation, ...] = ()

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False,
                          separators=(',', ':'))


def _satisfied_goal(frame):
    """Only explicit target/value goals with a matching current observation.

    No inference of completion for mastery, open-ended growth or missing slots.
    A later state change may make the same ongoing goal actionable again.
    """
    if not frame.relevant_goals:
        return False
    facts = dict(frame.environment.facts)
    choices = frame.environment.choices
    targets = set(facts) | {choice.target for choice in choices if choice.target}
    for goal in frame.relevant_goals:
        terms = lexical_terms(' '.join(value for _, value in goal.values))
        referenced = targets & terms
        if not referenced:
            return False
        for target in referenced:
            values = {facts.get(target, ''), *(choice.value for choice in choices
                                             if choice.target == target)} - {''}
            desired = values & terms
            if len(desired) != 1 or facts.get(target) not in desired:
                return False
    return True


def _evaluate_actions(frame, verification=False):
    if frame.environment is None:
        return ()
    observation = frame.environment
    context = _terms('virtual ' + observation.activity)
    histories = {h.choice_id: h for h in frame.action_history}
    goal_anchors = [_terms(' '.join(value for _, value in r.values)) - context
                    for r in frame.relevant_goals]
    complete = _satisfied_goal(frame)
    evaluations = []
    for choice in observation.choices:
        history = histories.get(choice.id)
        successes, failures = (history.successes, history.failures) if history else (0, 0)
        n = successes + failures
        reasons, evidence = [], [s.id for s in history.sources] if history else []
        failed = bool(history and (history.latest_result == 'task_failure' or history.tendency_result == 'failure'))
        eligible = (not failed and choice.operation != 'next' and not complete
                    and (not verification or choice.action in {'inspect', 'verify'}))
        if failed:
            reasons.append('observed_failure')
        if complete:
            reasons.append('goal_already_observed')
        targets = _terms(goal_targets(choice)) | lexical_terms(choice.value)
        progress = max((len(anchor & targets) / len(anchor) for anchor in goal_anchors if anchor), default=None)
        if progress is None:
            progress = task_progress(observation, choice)
        if progress > 0:
            evidence.extend(r.id for r in frame.relevant_goals)
            reasons.append('goal_progress')
        affinities = []
        for item in frame.relevant_self:
            if item.kind not in {'preference', 'opinion', 'interest'}:
                continue
            values = dict(item.values)
            if values.get('status', 'active') != 'active':
                continue  # Contradicted Self is uncertainty, not a durable veto.
            stance = values.get('stance', '')
            if stance not in {'positive', 'negative', 'comparative'}:
                continue
            anchor = _terms(values.get('topic', '')) - context
            if stance == 'comparative':
                anchor = lexical_terms(values.get('preferred', '')) - lexical_terms('virtual ' + observation.activity)
            features = _terms(choice.target + ' ' + choice.operation) | lexical_terms(choice.value)
            overlap = len(anchor & features) / len(anchor) if anchor else 0.0
            if not overlap:
                continue
            # Reuse existing strength/confidence and three-source establishment;
            # an initial own judgment is weak and is never silently established.
            support = min(1.0, len({s.id for s in item.sources}) / SELF_ESTABLISHED_EVIDENCE)
            weight = float(values.get('strength', 0)) * float(values.get('confidence', 0)) * support
            affinities.append((-1 if stance == 'negative' else 1) * weight * overlap)
            evidence.append(item.id)
            reasons.append(item.kind)
        affinity = sum(affinities) / len(affinities) if affinities else 0.0
        strategy_id = history.strategy_id if history else ''
        # One neutral pseudo-observation at each pole. Positive competence needs
        # an existing reinforced strategy backed by >=2 comparable executions.
        competence = (successes - failures) / (n + 2) if strategy_id or failures else 0.0
        if strategy_id:
            evidence.append(strategy_id)
            reasons.append('learned_strategy')
        information = information_gain(observation, choice)
        # Both signals are gated by a real information-bearing affordance.
        # Revision changes, repeated switching and noise get no novelty bonus.
        uncertainty = information / (1 + n)
        novelty = information / (1 + successes)
        if information:
            reasons.append('information')
        evaluations.append(ActionEvaluation(choice.id, eligible, progress, affinity, competence,
            information, uncertainty, novelty, strategy_id,
            tuple(dict.fromkeys(evidence))[:3], tuple(dict.fromkeys(reasons))))
    return tuple(evaluations)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate observation field.')
        result[key] = value
    return result


def _state_report(percept):
    """Optional exact data shape in a local receipt, not a control protocol.

    No free-text assertion extraction. Subject must agree with the receipt's
    provenance envelope. Extra fields have no effect; duplicate keys fail closed.
    """
    if percept.source not in {'runtime', 'tool'}:
        return None
    try:
        row = json.loads(percept.content, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        return None
    if (not isinstance(row, dict) or not {'entity_id', 'attribute', 'value'} <= row.keys()
            or any(not isinstance(row[k], str) or not row[k] or row[k] != row[k].strip()
                   for k in ('entity_id', 'attribute', 'value'))
            or row['entity_id'] != percept.subject
            or row['attribute'] != row['attribute'].casefold()):
        return None
    return row


def _assess_frame(frame):
    """Internal classifier. Store reconstructs a current frame before calling."""
    current = []
    history = []
    for item in frame.relevant_world:
        row = dict(item.values)
        if item.kind == 'world_history':
            history.append(item)
        elif item.kind == 'world_current':
            if {'entity_id', 'attribute', 'value'} <= row.keys():
                current.append((item, row))

    conflicts, missing, uncertainty, reasons = [], [], [], []
    values_by_key = {}
    for item, row in current:
        if item.sources:
            key = (row['entity_id'], row['attribute'])
            values_by_key.setdefault(key, set()).add(row['value'])
    if any(len(values) > 1 for values in values_by_key.values()):
        conflicts.append('incompatible_current_state')

    report = _state_report(frame.percept)
    corroborated = False
    if report:
        matches = [(item, row) for item, row in current
                   if (row['entity_id'], row['attribute']) == (report['entity_id'], report['attribute'])]
        if not matches:
            uncertainty.append('observation_unconfirmed')
        elif any(frame.percept.timestamp < item.timestamp for item, _ in matches):
            uncertainty.append('observation_older_than_state')
        elif any(row['value'] != report['value'] for _, row in matches):
            conflicts.append('observation_world')
        else:
            corroborated = True

    # A failure word is only an observed problem cue. It grants no authority;
    # both an existing objective and missing structured evidence are required.
    problem = bool(frame.objective and (
        report or _terms(frame.percept.content) & {'failed', 'failure', 'broke', 'broken', 'error'}
    ))
    outcomes = [r for r in frame.linked_evidence if r.kind == 'outcome' and r.sources]
    required_current = current
    if report:
        required_current = [(item, row) for item, row in current
                            if (row['entity_id'], row['attribute']) == (report['entity_id'], report['attribute'])]
    elif frame.percept.subject and frame.percept.subject != 'user':
        required_current = [(item, row) for item, row in current
                            if frame.percept.subject in {row['entity_id'], row.get('entity', '')}]
    if problem and not required_current and not outcomes:
        if not frame.percept.subject or frame.percept.subject == 'user':
            missing.append('subject')
        missing.append('current_state_or_result')

    grounded_current = any(any(s.kind == 'user_turn' for s in item.sources)
                           for item, _ in required_current)
    learned = any(dict(r.values).get('expected_result') == 'success'
                  and len({s.id for s in r.sources if s.kind == 'outcome'}) >= 2
                  for r in frame.relevant_strategies)
    developed = any(r.kind in {'preference', 'opinion', 'interest'}
                    and dict(r.values).get('status', 'active') == 'active'
                    and len({s.id for s in r.sources}) >= 2 for r in frame.relevant_self)
    if any(r.kind in {'preference', 'opinion', 'interest'} and (
            len({s.id for s in r.sources}) < 2 or dict(r.values).get('status') == 'uncertain')
           for r in frame.relevant_self):
        uncertainty.append('provisional_self')
    surprise = 'surprise' in frame.attention.reason_codes
    if grounded_current:
        reasons.append('grounded_current')
    if corroborated:
        reasons.append('corroborated_observation')
    if learned:
        reasons.append('learned_strategy')
    if developed:
        reasons.append('developed_self')
    if history:
        reasons.append('relevant_history')
        if not current:
            uncertainty.append('historical_only')
    if outcomes:
        reasons.append('linked_outcome')
    if frame.objective:
        reasons.append('active_objective')
    if surprise:
        uncertainty.append('prediction_surprise')
        reasons.append('prediction_surprise')
    if missing:
        uncertainty.append('required_evidence_missing')
        reasons.append('required_evidence_missing')
    if conflicts:
        uncertainty.append('conflicting_evidence')
        reasons.append('structured_conflict')

    confidence = 'medium'
    if (grounded_current and (corroborated or learned)) or (developed and not problem):
        confidence = 'high'
    if missing or conflicts or (report and not corroborated) or (history and not current):
        confidence = 'low'
    elif surprise:
        confidence = 'medium'

    # Only evidence that changes the legal decision affects the mode. Counting
    # records/relations added no capability and made broad context look complex.
    verification = bool(conflicts or missing or (surprise and frame.objective))
    mode = ('verify' if verification else 'deliberate' if history or outcomes or learned
            else 'normal' if current or frame.objective or developed else 'fast')
    if not reasons:
        reasons.append('simple_context')
    result = EvidenceAssessment(
        frame.profile_id, frame.percept.id,
        'frame_' + hashlib.sha256(frame.to_json().encode('utf-8')).hexdigest(),
        mode, confidence, tuple(uncertainty), tuple(conflicts),
        tuple(missing), verification, tuple(reasons), frame.created_at, _evaluate_actions(frame, verification),
    )
    return result if len(result.to_json().encode('utf-8')) <= ASSESSMENT_MAX_BYTES else None


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


JOURNAL_LIMIT = 64
SCHEDULE_LIMIT = 64


def empty_initiative():
    return dict(preferences=dict(enabled=True, availability='unknown', quiet_start=22, quiet_end=8),
                journal=[], schedules=[])


def validate_preferences(row):
    if (not isinstance(row, dict) or set(row) != {'enabled', 'availability', 'quiet_start', 'quiet_end'}
            or type(row['enabled']) is not bool
            or row['availability'] not in ('unknown', 'available', 'busy', 'away')
            or any(type(row[k]) is not int or not 0 <= row[k] <= 23 for k in ('quiet_start', 'quiet_end'))):
        raise ValueError('Invalid initiative preferences.')


def validate_initiative(row):
    """Strict persistence validation; corrupt safety state never resets silently."""
    if not isinstance(row, dict) or set(row) != {'preferences', 'journal', 'schedules'}:
        raise ValueError('Invalid initiative state.')
    validate_preferences(row['preferences'])
    if isinstance(row['schedules'], list) and sum(r.get('status') == 'pending' for r in row['schedules']
                                                if isinstance(r, dict)) > 16:
        raise ValueError('Initiative schedule capacity exceeded.')
    for group, limit in (('journal', JOURNAL_LIMIT), ('schedules', SCHEDULE_LIMIT)):
        rows = row[group]
        if not isinstance(rows, list) or len(rows) > limit:
            raise ValueError('Initiative capacity exceeded.')
        ids = set()
        for item in rows:
            keys = ({'id', 'fingerprint', 'subject', 'content', 'created_at', 'expires_at', 'action',
                     'status', 'episode_id', 'outcome_id', 'generation_at', 'offered_turn_id'}
                    if group == 'journal' else {'id', 'at', 'subject', 'goal_id', 'status'})
            if not isinstance(item, dict) or set(item) != keys:
                raise ValueError('Invalid initiative record.')
            if not _text(item['id']) or item['id'] in ids or not _text(item['subject'], optional=True):
                raise ValueError('Invalid initiative identity.')
            ids.add(item['id'])
            times = ('created_at', 'expires_at', 'generation_at') if group == 'journal' else ('at',)
            if any(type(item[k]) not in (int, float) or not math.isfinite(item[k]) or item[k] < 0 for k in times):
                raise ValueError('Invalid initiative time.')
            if group == 'journal':
                if (item['action'] not in ('act', 'follow_up')
                        or item['status'] not in ('reserved', 'pending', 'offered', 'completed',
                                                  'awaiting_result', 'cancelled', 'expired')
                        or item['expires_at'] < item['created_at']
                        or not _text(item['fingerprint'])
                        or not _text(item['content'], PERCEPT_CONTENT_BYTES, optional=True)
                        or any(not _text(item[k], optional=True) for k in
                               ('episode_id', 'outcome_id', 'offered_turn_id'))):
                    raise ValueError('Invalid initiative outcome.')
            elif item['status'] not in ('pending', 'fired', 'cancelled') or not _text(item['goal_id'], optional=True):
                raise ValueError('Invalid initiative schedule.')
