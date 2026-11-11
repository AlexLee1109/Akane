"""Bounded grounded memory recall; no model, personality learning, or state writes."""

from dataclasses import dataclass
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
    'strain': 'stress stressful stressed overwhelm overwhelmed overload overloaded',
    'work': 'work job employment office workplace shift shifts manager employer career',
    'responsibility': 'responsibility responsibilities duty duties workload assignment assignments commitment commitments volunteer volunteering',
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
_MUTABLE = lexical_terms(_CONCEPT_WORDS['fatigue'] + ' ' + _CONCEPT_WORDS['strain']
                        + ' meeting appointment schedule this month')
_FATIGUE_PHRASES = re.compile(r'\b(?:worn out|(?:running|run) on empty|out of (?:energy|steam)|'
                              r'(?:no|little) energy|burn(?:ed|t) out)\b', re.I)
_DEPICTIONS = lexical_terms('book novel fiction story painting picture film movie article')
_BUDGET_CONSTRAINTS = lexical_terms('budget afford allowance saving savings limit funds')
_SPENDING_DECISIONS = lexical_terms('afford budget expensive buy buying')


def recall_terms(text):
    terms = lexical_terms(text)
    if '-' in text:
        # Preserve exact identifiers while also matching ordinary compound
        # words written with spaces. Numbers and stored evidence stay literal.
        terms |= lexical_terms(re.sub(r'(?<=[a-zA-Z])-(?=[a-zA-Z])', ' ', text))
    return terms - RECALL_STOP_WORDS


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
        capacity_query = ('responsibility' in self.concepts
                          or 'work' in self.concepts and bool(self.concepts & {'fatigue', 'strain'}))
        food_decision = 'food' in self.concepts and not self.terms & _DEPICTIONS
        # A report about a story/painting is not a report of the user's capacity.
        # Media questions can still retrieve that exact remembered object.
        if capacity_query and terms & _DEPICTIONS and not self.terms & _DEPICTIONS:
            return 0.0
        if self.personal:
            if ('responsibility' in self.concepts and not self.terms & _DEPICTIONS
                    and concepts & {'fatigue', 'strain'}
                    and ('work' not in self.concepts or 'work' in concepts or 'strain' in concepts)):
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
        if ('work' in self.concepts and 'fatigue' in concepts
                and 'work' not in concepts and 'responsibility' not in self.concepts):
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


def wants_history(query):
    return bool(re.search(r"\b(before|previous|previously|earlier|change|changed|changes|changing|happened|last time)\b|"
                          r"\b(?:was|were)\b.*\b(?:doing|wrong)\b", query, re.I))
