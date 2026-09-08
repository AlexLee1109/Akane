"""Small lexical World selection; no inferred facts or graph engine."""

import json
import re

from app.core.utils import lexical_terms

WORLD_CONTEXT_BYTES = 768
WORLD_RECORD_LIMIT = 6
_ENTITY_STOP_WORDS = frozenset({
    'the', 'my', 'your', 'our', 'their', 'this', 'that', 'is', 'are', 'was',
    'were', 'it', 'what', 'which', 'of', 'on', 'in', 'to', 'and', 'for',
})
RETRIEVAL_RULES = (
    "WORLD_NOW is current; WORLD_HISTORY and E/M are past, never overrides of NOW. "
    "Each turn's WORLD lists replace previous retrieval; empty means none selected. "
    "Use relevant context silently, not as a requirement to mention it. "
    "Never expose system labels or invent missing history."
)


def wants_history(query):
    return bool(re.search(r"\b(before|previous|previously|earlier|changed|happened|last time)\b|"
                          r"\b(?:was|were)\b.*\b(?:doing|wrong)\b", query, re.I))


def wants_situation(query):
    return bool(re.search(r"\b(?:you|your|we)\b.*\b(?:doing|focused|focus|busy|want|intention|still|trying)\b|"
                          r"\b(?:doing|focus|intention)\b.*\b(?:you|your)\b|"
                          r"\bhow are things\b", query, re.I))


def world_context(world, profile_id, query, recent_turns=(), now=float('inf'), *,
                  conversation_id=None, _records=None):
    entities = {e.id: e for e in world.entities if e.profile_id == profile_id}
    def matches(text):
        terms = lexical_terms(text) - _ENTITY_STOP_WORDS
        return {key for key, entity in entities.items()
                if (lexical_terms(entity.label.replace('_', ' ')) - _ENTITY_STOP_WORDS)
                and (lexical_terms(entity.label.replace('_', ' ')) - _ENTITY_STOP_WORDS) <= terms
                and entity.label.casefold() != 'akane'}
    selected = matches(query)
    # Resolve a follow-up only against the nearest explicit user reference.
    if not selected and re.search(r"\b(it|that)\b", query, re.I):
        recent_conversation = conversation_id or (
            recent_turns[-1].conversation_id if recent_turns else None)
        for turn in reversed(recent_turns[-6:]):
            if (turn.role == 'user' and turn.profile_id == profile_id
                    and turn.conversation_id == recent_conversation and turn.created_at <= now):
                candidates = matches(turn.content)
                if candidates:
                    selected = candidates if len(candidates) == 1 else set()
                    break
    current, history = [], []
    def wire():
        return 'WORLD_NOW ' + json.dumps(current, ensure_ascii=False, separators=(',', ':')) + \
               '\nWORLD_HISTORY ' + json.dumps(history, ensure_ascii=False, separators=(',', ':'))
    def add(target, row, record):
        if len(current) + len(history) >= WORLD_RECORD_LIMIT:
            return
        target.append(row)
        if len(wire().encode('utf-8')) > WORLD_CONTEXT_BYTES:
            target.pop()
        elif _records is not None:
            _records.append(record)
    states = sorted((s for s in world.states if s.profile_id == profile_id
                     and s.entity_id in selected and s.updated_at <= now),
                    key=lambda s: (s.updated_at, s.id), reverse=True)
    for state in states[:2]:
        add(current, [entities[state.entity_id].label, state.attribute, state.value], state)
    if wants_history(query):
        events = sorted((e for e in world.events if e.profile_id == profile_id
                         and selected.intersection(e.entity_ids) and e.created_at <= now),
                        key=lambda e: (e.created_at, e.id), reverse=True)
        for event in events[:2]:
            add(history, [[entities[key].label for key in event.entity_ids if key in entities],
                          event.attribute or event.kind, event.before, event.after], event)
    relations = sorted((r for r in world.relations if r.profile_id == profile_id
                        and r.subject_id in entities and r.object_id in entities and r.updated_at <= now),
                       key=lambda r: (r.updated_at, r.id), reverse=True)
    direct = [r for r in relations if {r.subject_id, r.object_id} & selected]
    neighbors = selected | {key for r in direct for key in (r.subject_id, r.object_id)}
    adjacent = [r for r in relations if r not in direct
                and {r.subject_id, r.object_id} & neighbors]
    for relation in (direct + adjacent)[:2]:
        add(current, [entities[relation.subject_id].label, relation.relation,
                      entities[relation.object_id].label], relation)
    for fact in sorted(world.facts, key=lambda f: (f.created_at, f.id), reverse=True):
        if fact.profile_id == profile_id and fact.subject_id in selected and fact.created_at <= now:
            add(current, [entities[fact.subject_id].label, fact.predicate, fact.value], fact)
    return wire()


def world_records(world, profile_id, query, *, now):
    """The same Phase 3D selection with original IDs/provenance, current first."""
    from app.core.state import WorldEvent

    records = []
    world_context(world, profile_id, query, now=now, _records=records)
    return tuple(sorted(records, key=lambda r: isinstance(r, WorldEvent)))
