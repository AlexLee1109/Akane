"""Bounded working context for one attended observation; no reasoning or actions."""

import json
from dataclasses import asdict, dataclass, replace

from app.core.attention import AttentionCandidate, _relevant, _terms
from app.core.mind import _reason_semantics
from app.core.perception import Percept
from app.core.retrieval import world_records
from app.core.situation import current_situation
from app.core.state import WorldEvent, WorldSource
from app.core.environment import EnvironmentObservation, learning_context


FRAME_MAX_BYTES = 6144
FRAME_RECORD_BYTES = 1024
FRAME_LIMITS = {
    'situation': 4, 'relevant_world': 4, 'relevant_self': 3,
    'relevant_memory': 2, 'relevant_experience': 1,
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
class CognitiveFrame:
    profile_id: str
    percept: Percept
    attention: AttentionCandidate
    created_at: float
    objective: WorldSource | None = None
    situation: tuple[FrameRecord, ...] = ()
    relevant_world: tuple[FrameRecord, ...] = ()
    relevant_self: tuple[FrameRecord, ...] = ()
    relevant_memory: tuple[FrameRecord, ...] = ()
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
    frame = CognitiveFrame(percept.profile_id, percept, attention, now,
                           environment=environment, linked_evidence=linked_evidence, action_history=action_history)
    if len(frame.to_json().encode('utf-8')) > FRAME_MAX_BYTES:
        return None  # Required provenance is never truncated to make it fit.
    terms = _terms(query)
    entities = {e.id: e.label for e in world.entities if e.profile_id == percept.profile_id}
    entity_terms = [_terms(label) for label in entities.values()]

    def related(*texts):
        return any(_relevant(_terms(text), terms, entity_terms) for text in texts)

    def add(category, record, *, objective=False):
        nonlocal frame
        if len(getattr(frame, category)) >= FRAME_LIMITS[category]:
            return
        if len(_json(record).encode('utf-8')) > FRAME_RECORD_BYTES:
            return
        candidate = replace(frame, **{category: (*getattr(frame, category), record)})
        if objective and candidate.objective is None:
            candidate = replace(candidate, objective=WorldSource(record.kind, record.id))
        if len(candidate.to_json().encode('utf-8')) <= FRAME_MAX_BYTES:
            frame = candidate

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
        frame = replace(frame, objective=WorldSource(goal.kind, goal.id))
        if len(frame.to_json().encode('utf-8')) > FRAME_MAX_BYTES:
            frame = replace(frame, objective=None)

    for item in world_records(world, percept.profile_id, query, now=now):
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
    for item in state.memories:
        # Store already gates hybrid recall, confidence, time and provenance.
        # A second lexical gate would discard indirectly relevant evidence.
        add('relevant_memory', _record(item, 'memory', ('subject', 'kind', 'text'),
            sources=tuple(WorldSource('turn', id) for id in item.source_turn_ids)))
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
