"""Bounded hybrid evidence selection and lexical World relations; no model."""

from dataclasses import dataclass
import json
import re

from app.core.utils import lexical_terms, text_key


# Function words carry no autobiographical evidence. Unlike stemming or fuzzy
# character matching, these explicit equivalences never invent a stored fact.
RECALL_STOP_WORDS = frozenset('''a an and are as at be been by can could did do
does for from had has have how in is it me my of on or our should that the their
them there these they this to was we were what when where which who why will
with would you your about am please tell remind suggest consider explain
remember recall report reported own owned some another additional more'''.split())
_CONCEPT_WORDS = {
    'fatigue': 'exhausted exhaustion tired fatigue drained depleted burnout overworked',
    'work': 'work job employment office workplace shift shifts manager employer career',
    'responsibility': 'responsibility responsibilities duty duties workload assignment assignments',
    'food': 'food eat eating hungry meal meals snack snacks sandwich sandwiches lunch dinner breakfast diet dietary',
    'allergy': 'allergy allergies allergic allergen allergens',
    'restriction': 'restriction restrictions intolerance intolerant avoid avoidance',
    'money': 'money cost costs price prices budget budgets afford affordable expensive purchase purchases buy buying',
    'schedule': 'schedule calendar meeting meetings appointment appointments availability available busy',
    'home': 'home residence reside live living city',
    'collection': 'collection collect collecting collected',
    'map': 'map maps cartography cartographic',
}
_CONCEPTS = {term: concept for concept, words in _CONCEPT_WORDS.items()
             for term in lexical_terms(words)}
_FOOD_ALLERGENS = lexical_terms('peanut peanuts nut nuts dairy milk egg eggs wheat soy shellfish sesame fish gluten')
_DAYS = lexical_terms('Monday Tuesday Wednesday Thursday Friday Saturday Sunday')
_MUTABLE = lexical_terms(_CONCEPT_WORDS['fatigue'] + ' meeting appointment schedule this month')
_FATIGUE_PHRASES = re.compile(r'\b(?:worn out|(?:running|run) on empty|out of (?:energy|steam)|'
                              r'(?:no|little) energy|burn(?:ed|t) out)\b', re.I)
_DEPICTIONS = lexical_terms('book novel fiction story painting picture film movie article')
_BUDGET_CONSTRAINTS = lexical_terms('budget afford allowance saving savings limit funds')
_SPENDING_DECISIONS = lexical_terms('afford budget expensive buy buying')


def recall_terms(text):
    return lexical_terms(text) - RECALL_STOP_WORDS


def semantic_terms(text):
    return _semantic_concepts(text, recall_terms(text))


def _semantic_concepts(text, terms):
    concepts = {_CONCEPTS.get(term, term) for term in terms}
    if _FATIGUE_PHRASES.search(text):
        concepts.add('fatigue')
    return concepts


def same_memory_fact(left, right):
    """Deduplicate normalized reports, not facts sharing sentence grammar.

    Mutable slots (residence/name/etc.) are handled by Memory's existing update
    rule. A high overlap ratio cannot establish that changed entities, amounts,
    negation or relations mean the same fact. Keep both originals when unsure.
    """
    return text_key(left) == text_key(right)


@dataclass(frozen=True, slots=True)
class RecallQuery:
    text: str
    terms: frozenset[str]
    concepts: frozenset[str]
    personal: bool
    historical: bool

    @classmethod
    def build(cls, text, recent=(), *, profile_id='', conversation_id='', now=float('inf')):
        # Only a genuinely ambiguous follow-up borrows the nearest user context.
        # Assistant prose is never a source of facts about the user's life.
        if re.search(r'\b(that|it|those)\b', text, re.I) and len(recall_terms(text)) <= 3:
            for turn in reversed(recent[-6:]):
                if (turn.role == 'user' and turn.profile_id == profile_id
                        and turn.conversation_id == conversation_id and turn.created_at <= now):
                    if len(recall_terms(turn.content)) >= 2:
                        text += ' ' + turn.content
                    break
        personal = bool(re.search(r'\b(I|me|my|we|our|remember|recall|remind|suggest|recommend)\b', text, re.I))
        return cls(text, frozenset(recall_terms(text)), frozenset(semantic_terms(text)),
                   personal, wants_history(text) or bool(re.search(
                       r'\b(change|ago|historical)\b|\blast (?:winter|summer|year|month|week)\b|\bused to\b', text, re.I)))

    def score(self, text, *, subject='akane'):
        """Lexical/semantic coverage plus four guarded constraint relations.

        Broad topic overlap is insufficient for recalling a user's constraint.
        A relation needs both an applicable constraint and a matching decision.
        Scores are bounded [0,1]; .25 is one of four content anchors.
        """
        if not self.terms:
            return 0.0
        terms = recall_terms(text)
        concepts = _semantic_concepts(text, terms)
        shared = len(self.terms & terms)
        lexical = shared / max(1, min(4, len(self.terms)))
        semantic = len(self.concepts & concepts) / max(1, min(4, len(self.concepts)))
        if subject == 'user' and not self.personal:
            # A specific nonpersonal question may still recall an exact object;
            # general explanatory questions should not pull personal biography.
            if shared < 2 or re.match(r'\s*(explain|how|why)\b', self.text, re.I):
                return 0.0
        relation = 0.0
        capacity_query = 'work' in self.concepts and bool(self.concepts & {'fatigue', 'responsibility'})
        food_decision = 'food' in self.concepts and not self.terms & _DEPICTIONS
        # A report about a story/painting is not a report of the user's capacity.
        # Media questions can still retrieve that exact remembered object.
        if capacity_query and terms & _DEPICTIONS and not self.terms & _DEPICTIONS:
            return 0.0
        if self.personal:
            if ({'fatigue', 'work'} <= concepts and 'work' in self.concepts
                    and 'responsibility' in self.concepts):
                relation = .75
            if ({'allergy', 'restriction'} & concepts and terms & _FOOD_ALLERGENS
                    and food_decision):
                relation = .75
            if ('money' in concepts and 'money' in self.concepts
                    and terms & _BUDGET_CONSTRAINTS and self.terms & _SPENDING_DECISIONS):
                relation = .75
            if ('schedule' in concepts and 'schedule' in self.concepts
                    and terms & self.terms & _DAYS):
                relation = .75
        if (capacity_query
                and 'work' in concepts and not concepts & {'fatigue', 'responsibility'}):
            return 0.0
        if 'work' in self.concepts and 'fatigue' in concepts and 'work' not in concepts:
            return 0.0
        if food_decision and not ('food' in concepts or
                concepts & {'allergy', 'restriction'} and terms & _FOOD_ALLERGENS):
            return 0.0
        # Do not treat food safety as evidence of a non-dietary allergy, nor
        # transfer a dated appointment to a different day.
        if (food_decision and 'allergy' in concepts
                and not terms & _FOOD_ALLERGENS):
            return 0.0
        if self.terms & _DAYS and terms & _DAYS and not self.terms & terms & _DAYS:
            return 0.0
        return min(1.0, max(lexical, semantic, relation))


def evidence_rank(score, row, now, *, historical=False, affinity=0.0):
    """Relevance gates admission; confidence, age and context only rank hits.

    Confidence scales evidence quality directly. Source-less legacy rows get
    half weight. Mutable reports have a 30-day half-life with a 25% floor: old
    evidence stays recallable as history, never silently declared resolved.
    Stable facts and explicit historical queries do not decay.
    """
    at = float(row.get('updated_at', row.get('created_at', 0)))
    if at > now:
        return 0.0, 0.0, 0.0
    weight = max(0.0, min(1.0, float(row.get('confidence', 1.0))))
    if not row.get('source_turn_ids', row.get('source_ids', ())):
        weight *= .5
    if not historical and (recall_terms(row.get('text', '')) & _MUTABLE
                           or _FATIGUE_PHRASES.search(row.get('text', ''))):
        weight *= max(.25, 2 ** (-max(0.0, now - at) / (30 * 86400)))
    # Context may resolve equally relevant evidence; it cannot admit a miss.
    return score * weight, affinity, float(row.get('importance', row.get('salience', 0)))


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
    current, history = [], []
    def wire():
        return 'WORLD_NOW ' + json.dumps(current, ensure_ascii=False, separators=(',', ':')) + \
               '\nWORLD_HISTORY ' + json.dumps(history, ensure_ascii=False, separators=(',', ':'))
    if not selected:
        return wire()
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
