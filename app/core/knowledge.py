"""Legacy state selection for explicit environment tools and saved-data inspection.

Ordinary text conversation uses context.py and retrieval.py exclusively.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.learning import (
    behavioral_tendency_state, curiosity_state, developmental_goal_state,
    self_development_state, self_topic_terms, strategy_state,
)
from app.core.state import SELF_KINDS
from app.core.utils import lexical_terms, text_key
from app.core.retrieval import RecallQuery, recall_terms, semantic_terms, same_memory_fact, evidence_rank, wants_history



def judgment_relevance(query: RecallQuery, topic: str, statement: str) -> float:
    """Recall an owned judgment for its subject, not one incidental reply word."""
    if query.score(topic) == 0 and len(query.terms & recall_terms(statement)) < 2:
        return 0.0
    return query.score(f'{topic} {statement}')


def memory_relevance(query: RecallQuery, text: str, subject: str, *, cues=(), role=''):
    """Use source-linked semantic labels for recall, never as factual context.

    The same foreground model names the report's concepts when it is formed.
    A resource/consequence link needs a shared concept and a personal query;
    its role alone cannot make an unrelated record relevant.
    """
    score = query.score(text, subject=subject)
    if not cues or not query.personal:
        return score
    cue_score = query.score(' '.join(cues), subject=subject)
    if cue_score >= .25:
        score = max(score, cue_score * .8)
        if role in {'constraint', 'consequence', 'commitment'}:
            score = max(score, .3)
    return score




def consolidate_experiences(selected, experiences, outcomes=(), self_items=(), revisions=()):
    """Connect retained originals without writing summaries or reinforcing Self.

    Exact action AND context separate task procedures; Self's own provenance
    determines support/contradiction and prior interpretations. We never infer
    a positive outcome from response wording. Overlapping or recursively cited
    source sets form one evidence component, including reworded summaries.
    Work is bounded by the existing 64-episode retention cap, selected <=1.
    """
    from app.core.state import ExperienceGroup
    profiles = {e.profile_id for e in selected}
    experiences = tuple(e for e in experiences if e.profile_id in profiles)
    by_id = {(e.profile_id, e.id): e for e in experiences}

    def observations(e):
        # Keep original observations separate: a summary citing A and B must
        # neither count as C nor collapse independent A/B into one observation.
        pending, visited, originals = [e], set(), []
        while pending:
            item = pending.pop()
            key = (item.profile_id, item.id)
            if key in visited:
                continue
            visited.add(key)
            references = [source for source in item.source_turn_ids
                          if source.startswith('experience_') or (item.profile_id, source) in by_id]
            if references:
                pending.extend(by_id[(item.profile_id, source)] for source in references
                               if (item.profile_id, source) in by_id)
            elif item.source_turn_ids:
                originals.append(item)
        return originals

    def key(e):
        if e.self_item_ids:
            return 'self', e.subject, tuple(sorted(e.self_item_ids))
        family = ('task' if e.kind in {'task_success', 'task_failure'} else
                  'feedback' if e.kind in {'positive_feedback', 'negative_feedback'} else e.kind)
        # An outcome supplies the procedure, not a guess from the narration.
        linked = [o for o in outcomes if o.profile_id == e.profile_id and
                  (e.id in o.experience_ids or set(o.source_turn_ids) & set(e.source_turn_ids))]
        actions = {o.action for o in linked}
        action = next(iter(actions)) if len(actions) == 1 else e.akane_response
        context = text_key(e.topic) if family == 'task' else tuple(sorted(semantic_terms(e.topic)))
        return family, e.subject, context, action if family == 'task' else ''

    groups = []
    for anchor in selected:
        anchor_key = key(anchor)
        members = [e for e in experiences if e.profile_id == anchor.profile_id and key(e) == anchor_key]
        members.sort(key=lambda e: (e.created_at, e.id))
        components, original_ids = [], set()
        for e in members:
            for original in observations(e):
                if original.id in original_ids or key(original) != anchor_key:
                    continue
                original_ids.add(original.id)
                sources = set(original.source_turn_ids)
                overlapping = [c for c in components if c & sources]
                for component in overlapping:
                    sources.update(component)
                    components.remove(component)
                components.append(sources)
        ids = {e.id for e in members}
        supporting, contradicting, superseded = set(), set(), set()
        for item in self_items:
            if item.profile_id == anchor.profile_id and item.id in anchor.self_item_ids:
                supporting.update(ids & set(item.source_ids))
                contradicting.update(ids & set(item.contradiction_ids))
                superseded.update(ids & {source for revision in revisions
                    if revision.profile_id == anchor.profile_id and revision.self_item_id == item.id
                    for source in revision.source_ids}
                    - set(item.source_ids) - set(item.contradiction_ids))
        if anchor.kind in {'task_success', 'task_failure', 'positive_feedback', 'negative_feedback'}:
            supporting.update(e.id for e in members if e.id in original_ids
                              and e.kind in {'task_success', 'positive_feedback'})
            contradicting.update(e.id for e in members if e.id in original_ids
                                 and e.kind in {'task_failure', 'negative_feedback'})
        groups.append(ExperienceGroup(anchor.id, tuple(e.id for e in members),
            tuple(sorted(set().union(*components))), len(components),
            tuple(sorted(supporting)), tuple(sorted(contradicting)), tuple(sorted(superseded))))
    return tuple(groups)

WORLD_CONTEXT_BYTES = 768
WORLD_RECORD_LIMIT = 6
_ENTITY_STOP_WORDS = frozenset({
    'the', 'my', 'your', 'our', 'their', 'this', 'that', 'is', 'are', 'was',
    'were', 'it', 'what', 'which', 'of', 'on', 'in', 'to', 'and', 'for',
})


def wants_situation(query):
    return bool(re.search(r"\b(?:you|your|we)\b.*\b(?:doing|focused|focus|busy|want|intention|still|trying)\b|"
                          r"\b(?:doing|focus|intention)\b.*\b(?:you|your)\b|"
                          r"\bhow are things\b", query, re.I))


def world_selection(world, profile_id, query, recent_turns=(), now=float('inf'), *,
                    conversation_id=None):
    """Resolve explicit entities and same-conversation references conservatively."""
    entities = {e.id: e for e in world.entities if e.profile_id == profile_id}
    entity_terms = {
        key: terms for key, entity in entities.items()
        if entity.label.casefold() != 'akane'
        and (terms := lexical_terms(entity.label.replace('_', ' ')) - _ENTITY_STOP_WORDS)
    }
    def matches(text):
        terms = lexical_terms(text) - _ENTITY_STOP_WORDS
        return {key for key, anchor in entity_terms.items() if anchor <= terms}
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
    return entities, selected, entity_terms


def _world_report_is_context(text):
    """Allow qualified claims, but never crop fiction/questions into reports."""
    from app.core.learning import (_FICTIONAL_FRAMING, _HYPOTHETICAL_MARKERS,
                                   _NONLITERAL_FRAMING, _normalized_evidence)

    normalized = _normalized_evidence(text)
    return bool(normalized) and not (
        len(text.encode('utf-8')) > 320 or '?' in text
        or _FICTIONAL_FRAMING.search(normalized)
        or _NONLITERAL_FRAMING.search(normalized)
        or any(marker in normalized for marker in _HYPOTHETICAL_MARKERS)
        or re.search(r'\b(?:if|imagine|imagined|pretend|suppose|hypothetical|hypothetically)\b', normalized)
    )


def _world_timestamp(timestamp):
    """Project an absolute UTC date without altering the durable numeric time."""
    try:
        return datetime.fromtimestamp(timestamp, ZoneInfo('UTC')).isoformat().replace('+00:00', 'Z')
    except (OverflowError, OSError, ValueError):
        # Legacy numeric timestamps may exceed the supported calendar range.
        return None


def world_context(world, profile_id, query, recent_turns=(), now=float('inf'), *,
                  conversation_id=None, reports=(), _records=None, _selection=None):
    entities, selected, entity_terms = (_selection if _selection is not None else
        world_selection(world, profile_id, query, recent_turns, now,
                        conversation_id=conversation_id))
    current, history, reported = [], [], []
    def wire():
        return 'WORLD_NOW ' + json.dumps(current, ensure_ascii=False, separators=(',', ':')) + \
               '\nWORLD_HISTORY ' + json.dumps(history, ensure_ascii=False, separators=(',', ':')) + \
               ('\nWORLD_REPORTS ' + json.dumps(reported, ensure_ascii=False, separators=(',', ':'))
                if reported else '')
    if not selected:
        return wire()
    def add(target, row, record):
        if len(current) + len(history) + len(reported) >= WORLD_RECORD_LIMIT:
            return False
        target.append(row)
        if len(wire().encode('utf-8')) > WORLD_CONTEXT_BYTES:
            target.pop()
            return False
        if _records is not None and record is not None:
            _records.append(record)
        return True
    states = sorted((s for s in world.states if s.profile_id == profile_id
                     and s.entity_id in selected and s.updated_at <= now),
                    key=lambda s: (s.updated_at, s.id), reverse=True)
    mutable_rows = []
    for state in states[:2]:
        row = [entities[state.entity_id].label, state.attribute, state.value,
            {'at': _world_timestamp(state.updated_at),
             'basis': ('user_report' if any(source.kind == 'user_turn' for source in state.sources)
                       else 'verified_observation' if any(source.kind == 'environment' for source in state.sources)
                       else 'recorded_evidence'),
             'verified_now': False}]
        if add(current, row, state):
            mutable_rows.append((row[3], state))
    # Keep a newer, unparsed report available without changing validated World.
    # Exact source text preserves uncertainty, attribution and dates for Qwen;
    # it is not a replacement state or evidence of an executed action.
    # A newer observation of another attribute cannot settle an older slot.
    oldest = {key: min((s.updated_at for s in states if s.entity_id == key), default=0)
              for key in selected}
    accepted = {source.id for state in states for source in state.sources if source.kind == 'user_turn'}
    labels = {key: (lexical_terms(entities[key].label) if '_' in entities[key].label
                    else entity_terms[key]) for key in selected}
    for turn in sorted(reports, key=lambda t: (t.created_at, t.id), reverse=True):
        mentioned = {key for key, terms in labels.items()
                     if terms <= lexical_terms(turn.content) - _ENTITY_STOP_WORDS}
        if (turn.role != 'user' or turn.profile_id != profile_id or turn.created_at > now
                or turn.id in accepted
                or not mentioned or not any(turn.created_at > oldest[key] for key in mentioned)
                or not _world_report_is_context(turn.content)):
            continue
        # Receipt order is explicit; the quoted event may still be historical.
        # Count only included reports and include these flags in the byte bound.
        marked = [metadata for metadata, state in mutable_rows
                  if state.entity_id in mentioned and turn.created_at > state.updated_at
                  and 'newer_report_received' not in metadata]
        for metadata in marked:
            metadata['newer_report_received'] = True
        if not add(reported, {'text': turn.content, 'at': _world_timestamp(turn.created_at),
                              'source': turn.id, 'unverified': True}, None):
            for metadata in marked:
                del metadata['newer_report_received']
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
    adjacent = [r for r in relations if r.subject_id not in selected and r.object_id not in selected
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


_GENERIC_QUERY_TERMS = frozenset({
    "about", "are", "do", "feel", "have", "like", "now", "still", "think",
    "want", "what", "which", "why", "would", "you", "your",
})


def _term_relevance(query_terms: set[str], candidate: object) -> float:
    candidate_terms = lexical_terms(candidate) - _GENERIC_QUERY_TERMS
    shared = len(query_terms & candidate_terms)
    return max(
        shared / max(1, len(query_terms) + len(candidate_terms) - shared),
        shared / max(1, min(len(query_terms), 4)),
    )


def _broad_self_kinds(query: str) -> frozenset[str]:
    normalized = text_key(query)
    terms = lexical_terms(query)
    if "yourself" in terms or normalized in {"who are you", "what are you like"}:
        return frozenset(SELF_KINDS)
    if normalized == "what have you changed your mind about":
        return frozenset(SELF_KINDS)
    if "your interests" in normalized or "your hobbies" in normalized:
        return frozenset({"interest"})
    if "your opinion" in normalized:
        return frozenset({"opinion"})
    if "favorite" in terms or "favourite" in terms:
        return frozenset({"preference"})
    if normalized == "what do you want":
        return frozenset({"goal", "preference"})
    if "your goals" in normalized:
        return frozenset({"goal"})
    if normalized in {"what do you like", "what do you prefer"}:
        return frozenset({"interest", "preference"})
    return frozenset()


def _memory_query_topic(query: str) -> str:
    normalized = text_key(query)
    if "where" in normalized and any(term in normalized for term in ("i live", "i am from", "i m from")):
        return "residence"
    if "my name" in normalized or "what is my name" in normalized:
        return "name"
    if "my birthday" in normalized:
        return "birthday"
    if "my job" in normalized or "where do i work" in normalized:
        return "work"
    if "my project" in normalized or "working on" in normalized:
        return "project"
    return ""


def _rank_developmental_rows(
    rows,
    query_terms: set[str],
    *,
    text_fields: tuple[str, ...],
    provenance_fields: tuple[str, ...],
    minimum_provenance: int,
    minimum_strength: float,
    minimum_confidence: float,
    now: float = float('inf'),
) -> list[tuple[float, float, float, float, str, dict[str, object]]]:
    ranked = []
    for row in rows:
        provenance_count = sum(len(row[field]) for field in provenance_fields)
        if (
            row["status"] != "active"
            or float(row['updated_at']) > now
            or provenance_count < minimum_provenance
            or float(row["strength"]) < minimum_strength
            or float(row["confidence"]) < minimum_confidence
        ):
            continue
        evidence = " ".join(str(row[field]) for field in text_fields)
        # Learned guidance belongs to its context/topic. Shared procedure or
        # effect words alone cannot make a different target relevant.
        anchor = semantic_terms(' '.join(self_topic_terms(row[text_fields[0]])))
        if not anchor or not anchor <= query_terms:
            continue
        score = _term_relevance(query_terms, evidence) if query_terms else 0.0
        if score >= 0.24:
            ranked.append((
                score,
                float(row["strength"]),
                float(row["confidence"]),
                float(row["updated_at"]),
                str(row["id"]),
                row,
            ))
    return ranked


from app.core.context import ContextBuilder, MEMORY_LINE_BYTES, format_context, format_context_sections

# Historical inspection tools use this bound for persisted experience excerpts.
EXPERIENCE_LINE_BYTES = 1024
