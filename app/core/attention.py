"""Model-free eligibility over Store-owned percepts; no cognitive consumer."""

from dataclasses import dataclass

from app.core.mind import (
    behavioral_tendency_state, curiosity_state, self_development_state,
    self_topic_terms,
)
from app.core.situation import current_situation


ATTENTION_LIMIT = 16
# Grammar and generic judgment/action words cannot establish topic relevance.
_STOP = self_topic_terms(
    'the this that these those is are was were be been have has had with from '
    'into about your their our you they it its my me we us and but for not '
    'like love prefer want think interest goal task result event status '
    'do does did doing make improve complete completed talking engaged'
)
_REASONS = ('goal', 'focus', 'self', 'novelty', 'surprise', 'curiosity', 'task')


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
                        curiosities, goals, predictions, outcomes, now):
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
    return tuple(sorted(candidates, key=lambda c: -c.salience)[:ATTENTION_LIMIT])
