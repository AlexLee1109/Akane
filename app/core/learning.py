"""Legacy evidence validation and explicit environment result helpers.

No foreground personality learning or automatic post-turn derivation runs here.
Existing persisted developmental records remain readable and validated.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from app.core.state import BehavioralTendency, BehavioralTendencyChange, Curiosity, DevelopmentalGoal, Experience, MEMORY_RECALL_ROLES, Outcome, Prediction, SelfItem, Strategy, StrategyChange, Turn, clamp
from app.core.utils import compact_text, lexical_terms, relevance, text_key


_GENERIC_TOPIC_TERMS = frozenset({
    "about", "akane", "and", "are", "as", "at", "because", "choose", "do",
    "feel", "favorite", "for", "from", "have", "interest", "into", "is",
    "it", "like", "me", "my", "of", "on", "opinion", "prefer", "that",
    "the", "them", "think", "this", "to", "view", "what", "which", "while",
    "versus", "vs", "with", "would", "you", "your",
})

# These cues guard Memory extraction and structured Self candidate admission.
_TEMPORARY_MARKERS = (
    "at the moment", "for this one", "for now", "in this situation",
    "just this once", "pretend", "right now", "roleplay", "temporary",
    "temporarily", "today only", "tonight",
)
_HYPOTHETICAL_MARKERS = (
    "forced to choose", "hypothetical", "hypothetically", "if i had to",
    "if you had to", "imagine that", "in character", "speaking as",
    "suppose that",
)
_HYPOTHETICAL_OPENERS = ("if ", "imagine ", "maybe ", "pretend ", "suppose ")
_CURRENT_WANT = re.compile(
    r"\b(?:i(?:'m| am) craving|i crave|for this task|for this meal)\b",
    re.IGNORECASE,
)
_OTHER_PERSON_CLAIM = re.compile(
    r"\b(?:my\s+(?:brother|coworker|dad|father|friend|mother|mom|sister)|"
    r"he|she|someone|somebody|they)\s+"
    r"(?:believes?|feels?|has|hates?|is|likes?|lives?|loves?|prefers?|said|says?|thinks?|told|works?)\b",
    re.IGNORECASE,
)
_NAMED_PERSON_CLAIM = re.compile(
    r"\b[A-Z][A-Za-z'-]{1,40}(?:'s)?\s+"
    r"(?:favorite\b|believes?|hates?|likes?|lives?|loves?|prefers?|said|says?|thinks?|told|works?)\b",
)
_QUOTED_FIRST_PERSON = re.compile(r'["“][^"”]{0,200}\b(?:i|my)\b', re.IGNORECASE)
_OTHER_PERSON_FRAMING = re.compile(r"\b(?:according to|quoting|to quote)\b", re.IGNORECASE)
_FICTIONAL_FRAMING = re.compile(
    r"\b(?:in|for|during)\s+(?:(?:a|an|the|my|our|this|that)\s+)?"
    r"(?:(?:fictional|imaginary|invented|made-up)\s+)?"
    r"(?:novel|story|screenplay|script|scenario|role[ -]?play|dream)\b|"
    r"\b(?:my|our|this|that|the)\s+(?:fictional\s+)?"
    r"(?:character|narrator|protagonist|persona)\b|"
    r"\b(?:this is|that is|it is|it's)\s+(?:a\s+)?"
    r"(?:fiction|fictional|an? imaginary|invented)\b|"
    r"\bi\s+(?:dreamed|dreamt)\b",
    re.IGNORECASE,
)
_NONLITERAL_FRAMING = re.compile(
    r"\b(?:i(?:'m| am) (?:just )?joking|(?:this|that|it)(?:'s| is| was) "
    r'(?:just |only )?a joke|just a joke|only a joke|not literally|being sarcastic)\b', re.I,
)

_SELF_EVIDENCE_KINDS = frozenset({"preference", "opinion", "interest", "goal"})
_EVENT_EVIDENCE_KINDS = frozenset({
    "correction", "positive_feedback", "negative_feedback",
    "task_success", "task_failure",
})
_PREDICTION_EVIDENCE_KINDS = frozenset({
    "prediction_success", "prediction_failure",
})
_CURIOSITY_EVIDENCE_KINDS = frozenset({
    "unresolved_curiosity", "resolved_curiosity",
})
_DEVELOPMENTAL_GOAL_EVIDENCE_KINDS = frozenset({
    "developmental_goal", "developmental_goal_release",
})
_SEMANTIC_KINDS = (
    _SELF_EVIDENCE_KINDS | _EVENT_EVIDENCE_KINDS
    | _PREDICTION_EVIDENCE_KINDS | _CURIOSITY_EVIDENCE_KINDS
    | _DEVELOPMENTAL_GOAL_EVIDENCE_KINDS | {"none"}
)
_SEMANTIC_STANCES = frozenset({"positive", "negative", "neutral", "comparative"})
_SEMANTIC_DURABILITY = frozenset({
    "candidate", "temporary", "hypothetical", "task-local", "none",
})
_SEMANTIC_KIND_CODES = {
    "p": "preference", "o": "opinion", "i": "interest", "g": "goal",
    "c": "correction", "f+": "positive_feedback", "f-": "negative_feedback",
    "t+": "task_success", "t-": "task_failure", "n": "none",
    "x+": "prediction_success", "x-": "prediction_failure",
    "u+": "unresolved_curiosity", "u-": "resolved_curiosity",
    "j+": "developmental_goal", "j-": "developmental_goal_release",
}
_SEMANTIC_STANCE_CODES = {
    "+": "positive", "-": "negative", "0": "neutral", "cmp": "comparative",
}
_SEMANTIC_DURABILITY_CODES = {
    "c": "candidate", "tmp": "temporary", "hyp": "hypothetical",
    "task": "task-local", "n": "none",
}
_SEMANTIC_REQUIRED_KEYS = {"k", "t", "s", "d", "e"}
_SEMANTIC_OPTIONAL_KEY_SETS = frozenset({
    frozenset(),
    frozenset({"v"}),
    frozenset({"a"}),
    frozenset({"a", "b", "f"}),
    frozenset({"a", "r"}),
    frozenset({"a", "b", "f", "r"}),
    frozenset({"a", "q"}),
    frozenset({"w"}),
    frozenset({"j"}),
    frozenset({"a", "j"}),
    frozenset({"a", "b", "f", "j"}),
    frozenset({"a", "r", "j"}),
    frozenset({"a", "b", "f", "r", "j"}),
})
_PREDICTION_CONFIDENCE_CODES = {"h": "high", "l": "low"}


@dataclass(frozen=True, slots=True)
class SemanticEvidence:
    kind: str
    topic: str
    stance: str
    durability: str
    evidence: str
    value: str = ""
    action: str = ""
    action_turn_id: str = ""
    confidence_band: str = ""
    behavior: str = ""
    effect: str = ""
    strategy: str = ""
    curiosity_focus: str = ""
    developmental_goal: str = ""


@dataclass(frozen=True, slots=True)
class SemanticSidecar:
    """One syntactic admission shared by development and grounded world learning."""

    raw: str
    records: tuple[dict, ...]

    @classmethod
    def parse(cls, raw: str) -> SemanticSidecar:
        return cls(raw, _semantic_records(raw))

    def world_evidence(self) -> SemanticSidecar:
        """Search annotations never become World facts or situation evidence."""
        records = tuple({key:value for key,value in row.items() if key not in {'r','c'}}
                        if row['k'] in _WORLD_EVIDENCE_KINDS else row for row in self.records)
        return SemanticSidecar(self.raw, records)


SELF_ESTABLISHED_EVIDENCE = 3
SELF_ESTABLISHED_STRENGTH = 0.55
SELF_ESTABLISHED_CONFIDENCE = 0.75
TENDENCY_FORM_STRENGTH = 0.35
TENDENCY_FORM_CONFIDENCE = 0.65
TENDENCY_REINFORCE_STRENGTH = 0.10
TENDENCY_REINFORCE_CONFIDENCE = 0.05
TENDENCY_WEAKEN_STRENGTH = 0.15
TENDENCY_WEAKEN_CONFIDENCE = 0.10
STRATEGY_FORM_STRENGTH = 0.35
STRATEGY_FORM_CONFIDENCE = 0.65


def self_topic_terms(value: object) -> frozenset[str]:
    terms = lexical_terms(value) - _GENERIC_TOPIC_TERMS
    return frozenset(
        term[:-3] if len(term) > 5 and term.endswith("ing") else term
        for term in terms
    )


def self_topic_similarity(left: object, right: object) -> float:
    left_key = text_key(left)
    right_key = text_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    left_terms = self_topic_terms(left)
    right_terms = self_topic_terms(right)
    shared = len(left_terms & right_terms)
    coverage = shared / max(1, min(len(left_terms), len(right_terms)))
    return coverage


def _best_match(items, score, threshold: float = 0.67):
    best = max(
        ((score(item), item) for item in items),
        key=lambda pair: pair[0],
        default=None,
    )
    return best[1] if best is not None and best[0] >= threshold else None


def _stable_id(prefix: str, *parts: str) -> str:
    fingerprint = "\0".join(parts)
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _adjust_strength_confidence(
    strength: float,
    confidence: float,
    strength_delta: float,
    confidence_delta: float,
) -> tuple[float, float]:
    return (
        round(clamp(strength + strength_delta, 0.0, 1.0), 6),
        round(clamp(confidence + confidence_delta, 0.0, 1.0), 6),
    )


def _same_self_topic(kind: str, left: object, right: object) -> bool:
    left_key = text_key(left)
    right_key = text_key(right)
    comparison = r"\b(?:vs|versus|over|rather than|more than|better than)\b"
    left_is_comparison = bool(re.search(comparison, left_key))
    right_is_comparison = bool(re.search(comparison, right_key))
    if kind == "preference" and left_is_comparison != right_is_comparison:
        return False
    if kind == "preference":
        left_terms = self_topic_terms(left)
        right_terms = self_topic_terms(right)
        return left_key == right_key or (
            len(left_terms & right_terms) / max(1, len(left_terms | right_terms)) >= 0.67
        )
    return self_topic_similarity(left, right) >= 0.67


def find_self_item(
    items: tuple[SelfItem, ...], kind: str, topic: str,
) -> SelfItem | None:
    eligible = tuple(
        item for item in items
        if item.kind == kind and item.status in {"active", "uncertain"}
    )
    if kind == "preference":
        return next((
            item for item in eligible
            if _same_self_topic(kind, topic, item.topic)
        ), None)
    return _best_match(
        eligible,
        lambda item: self_topic_similarity(topic, item.topic),
    )


def self_development_state(item: SelfItem) -> str:
    if item.status == "uncertain":
        return "uncertain"
    if (
        len(item.source_ids) >= SELF_ESTABLISHED_EVIDENCE
        and item.strength >= SELF_ESTABLISHED_STRENGTH
        and item.confidence >= SELF_ESTABLISHED_CONFIDENCE
    ):
        return "established"
    if len(item.source_ids) >= 2:
        return "reinforced"
    return "weak"


def _bounded_ids(*groups: tuple[str, ...], limit: int = 6) -> tuple[str, ...]:
    return tuple(dict.fromkeys(identifier for group in groups for identifier in group))[-limit:]


def _reason_semantics(reason: str) -> tuple[str, str]:
    marker = "|semantic:"
    if marker not in reason:
        return "", ""
    payload = reason.rpartition(marker)[2]
    stance, separator, value = payload.partition(":")
    if not separator or stance not in _SEMANTIC_STANCES:
        return "", ""
    return stance, value


def _normalized_evidence(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"',
    }))
    return " ".join(normalized.split()).casefold()


def _valid_semantic_phrase(value: str) -> bool:
    return bool(self_topic_terms(value)) and "\n" not in value and "\r" not in value


_POSITIVE_RESULT = re.compile(
    r"\b(?:worked|works|succeeded|succeeds|passed|helped|fixed|resolved|recovered|restored|"
    r"(?:reveal(?:ed)?|identif(?:ied|y)|found|find) (?:the )?(?:root )?cause|"
    r"(?:was|is|were|are|proved) (?:a )?(?:success|successful|effective|useful|"
    r"helpful|correct|accurate|clear)|(?:completed|finished) successfully)\b",
)
_NEGATIVE_RESULT = re.compile(
    r"\b(?:failed|fails|failure|unsuccessful|broke|broken|"
    r"(?:was|is|were|are|proved) (?:incorrect|wrong|inaccurate|unclear|ineffective|unhelpful)|"
    r"(?:was|is|were|are|proved) not (?:successful|effective|useful|helpful|correct|accurate|clear)|"
    r"(?:not|never|didn't|doesn't|hasn't|haven't|couldn't|wouldn't) "
    r"(?:work|succeed|pass|help|fix|resolve|recover|restore))\b",
)
_RESULT_NEGATION = re.compile(
    r"(?:\b(?:not|never|no)\b|n't)(?:\W+\w+){0,3}\W*$",
)


def _feedback_result_signals(text: str, action: str = "") -> set[str]:
    text = _normalized_evidence(text)
    if action:
        text = re.sub(r"(?<!\w)" + re.escape(_normalized_evidence(action)) + r"(?!\w)", "", text)
    results = ({'ambiguous'} if re.search(r"\b(?:hardly|barely|partly|partially|almost)\b", text)
               else set())
    for match in _POSITIVE_RESULT.finditer(text):
        prefix = re.split(r"[.!?;,\n]|\b(?:but|however|yet)\b", text[:match.start()])[-1]
        results.add("failure" if _RESULT_NEGATION.search(prefix) else "success")
    for match in _NEGATIVE_RESULT.finditer(text):
        prefix = re.split(r"[.!?;,\n]|\b(?:but|however|yet)\b", text[:match.start()])[-1]
        if not _RESULT_NEGATION.search(prefix):
            results.add("failure")
    return results


def verified_feedback_result(text: str, action: str = "") -> str:
    """Read explicit result witnesses, without treating a model label as proof.

    Unknown wording, negated success and mixed outcomes cannot grant positive
    credit. The whole user report is checked as well as the selected quote, so
    a sidecar cannot crop away an adverse result.
    """
    results = _feedback_result_signals(text, action)
    return next(iter(results)) if len(results) == 1 and 'ambiguous' not in results else ""


def feedback_result_is_contradicted(text: str, action: str, expected: str) -> bool:
    opposite = 'failure' if expected == 'success' else 'success'
    return opposite in _feedback_result_signals(text, action)


def _feedback_kind_is_grounded(kind: str, evidence: str, user_text: str, action: str = "") -> bool:
    if not _world_source_is_asserted(user_text) or _normalized_evidence(user_text).startswith(('"', '>')):
        return False
    if kind == "correction" and not action:
        return True
    expected = "success" if kind in {"task_success", "positive_feedback"} else "failure"
    return (
        verified_feedback_result(evidence, action) == expected
        and verified_feedback_result(user_text, action) == expected
    )


def _feedback_action_turn(
    action: str, evidence: str, user_turn: Turn, recent_turns: tuple[Turn, ...],
) -> Turn | None:
    action_key = _normalized_evidence(action)
    action_terms = self_topic_terms(action)
    if not action_terms:
        return None
    matches = tuple(
        turn for turn in recent_turns
        if turn.role == "assistant"
        and turn.profile_id == user_turn.profile_id
        and turn.conversation_id == user_turn.conversation_id
        and turn.created_at <= user_turn.created_at
        and action_key in _normalized_evidence(turn.content)
    )
    if len(matches) != 1:
        return None
    immediate_feedback = bool(re.fullmatch(
        r"(?:it|that) (?:worked|failed|helped)[.!]?", _normalized_evidence(user_turn.content),
    ))
    # A result and a quoted action in separate reports do not establish credit.
    if not immediate_feedback and not (
        _world_span(action, evidence) or action_terms <= self_topic_terms(evidence)
    ):
        return None
    if immediate_feedback:
        latest = next((turn for turn in reversed(recent_turns)
                       if turn.role == "assistant"
                       and turn.profile_id == user_turn.profile_id
                       and turn.conversation_id == user_turn.conversation_id
                       and turn.created_at <= user_turn.created_at), None)
        if latest is None or latest.id != matches[0].id:
            return None
    return matches[0]


def dialogue_outcome_is_grounded(outcome: Outcome, turns: tuple[Turn, ...]) -> bool:
    """Recheck conversational Outcome provenance at the canonical Store boundary."""
    if (
        outcome.result not in _EVENT_EVIDENCE_KINDS
        or outcome.reason not in {
            f"grounded:user-{outcome.result.replace('_', '-')}",
            f"grounded:user-{outcome.result.replace('_', '-')}:partial-report",
        }
        or len(outcome.source_turn_ids) != 2
        or outcome.source_turn_ids[0] != outcome.action_turn_id
    ):
        return False
    feedback = next((turn for turn in turns
                     if turn.id == outcome.source_turn_ids[1]
                     and turn.role == "user" and turn.profile_id == outcome.profile_id), None)
    if feedback is None or not _world_span(outcome.description, feedback.content):
        return False
    if outcome.reason.endswith(':partial-report') and (
        outcome.result not in {'task_success', 'positive_feedback'}
        or len(' '.join(feedback.content.split())) <= 280
    ):
        return False
    prior_turns = turns[:turns.index(feedback)]
    action = _feedback_action_turn(outcome.action, outcome.description, feedback, prior_turns)
    return bool(
        action is not None and action.id == outcome.action_turn_id
        and _feedback_kind_is_grounded(
            outcome.result, outcome.description, feedback.content, outcome.action,
        )
    )


_WORLD_SCOPE_BY_KIND = {
    "we": "na", "ws": "na", "wf": "da", "wr": "na", "wr-": "nn", "wv": "pa",
}
_WORLD_EVIDENCE_KINDS = frozenset(_WORLD_SCOPE_BY_KIND)


def _semantic_records(raw: str | SemanticSidecar) -> tuple[dict, ...]:
    """One object, or one Development + one World object in the same suffix."""
    if isinstance(raw, SemanticSidecar):
        return raw.records
    if not isinstance(raw, str) or not raw or len(raw) > 1024:
        return ()

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate semantic key")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=unique_keys)
    except (TypeError, ValueError):
        return ()
    records = (payload,) if isinstance(payload, dict) else payload
    if not isinstance(records, (list, tuple)) or not 1 <= len(records) <= 2:
        return ()
    if any(not isinstance(item, dict) or not isinstance(item.get("k"), str) for item in records):
        return ()
    if len(records) == 2 and not (
        sum(item["k"] in _WORLD_EVIDENCE_KINDS for item in records) == 1
        and sum(item["k"] in _SEMANTIC_KIND_CODES for item in records) == 1
    ):
        return ()
    # The kind already specifies World temporal scope. Omission is compact
    # syntax; an explicit incompatible scope must still reach rejection.
    return tuple(
        {**item, "d": _WORLD_SCOPE_BY_KIND[item["k"]]}
        if item["k"] in _WORLD_SCOPE_BY_KIND and "d" not in item else item
        for item in records
    )


_SELF_HISTORY = re.compile(
    r"\b(?:i|we)(?:\s+(?:earlier|once|always|never|also|then|often|just|[a-z]+ly))*\s+"
    r"(?:was|were|had|did|used to|[a-z]{3,}ed|drank|ate|went|grew|saw|sang|"
    r"took|made|met|felt|found|thought|knew|bought|brought|chose|came|drove|"
    r"flew|gave|got|heard|kept|left|lost|ran|rode|sat|slept|spent|stood|"
    r"swam|taught|told|wore|won|wrote|said)\b|"
    r"\b(?:i|we)(?: have|'ve|'d)\s+(?:(?:already|always|never|once|previously)\s+)*"
    r"(?:been|[a-z]{3,}ed|drunk|eaten|gone|seen|taken|made|met|felt|found|"
    r"known|bought|chosen|come|given|heard|kept|left|lost|run|ridden|slept|"
    r"spent|stood|swum|taught|told|worn|won|written|said)\b|"
    r"\bi(?: am|'m)\s+(?:originally\s+)?(?:from|born|raised)\b|"
    r"\b(?:my|our)\s+(?:childhood|upbringing|birthplace|hometown|schooldays)\b",
    re.IGNORECASE,
)
_SELF_BIOGRAPHY = re.compile(
    r"\b(?:my|our)\s+(?:(?:first|last|past|early|own|family)\s+)*"
    r"(?:trip|trips|visit|visits|journey|travels|upbringing|memories)\b|"
    r"\b(?:my|our)\s+(?:\w+\s+)?(?:years|days)\s+(?:in|at)\b|"
    r"\b(?:i|we)(?: have|'ve)\s+(?:\w+\s+)?memories\b|"
    r"\b(?:i|we)(?:\s+(?:still|vividly|clearly|fondly))*\s+(?:remember|recall)\s+"
    r"(?!you\b|your\b|(?:this|our|the) (?:conversation|chat)\b)\S|"
    r"\bi(?: am|'m)\s+(?:a|an)\s+(?:\w+\s+)?native\b|"
    r"\b(?:my|our)\s+(?:\w+\s+){1,3}"
    r"(?:was|were|had|became|taught|brought|gave|took|[a-z]{3,}ed)\b",
    re.IGNORECASE,
)
_SELF_PAST_FRAMING = re.compile(
    r"(?:^|,\s*|\b(?:from|after|since)\s+)growing up\b|"
    r"(?:^|,\s*)having\s+(?:been|[a-z]{3,}ed)\b|"
    r"\b[a-z]+ing\b[^.!?;]*\b(?:last (?:year|month|week|night)|"
    r"(?:years?|months?|weeks?|days?) ago|in (?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_SELF_SPEECH_HISTORY = re.compile(
    r"\bi\s+(?:(?:previously|earlier|once)\s+)?(?:said|wrote)\s+"
    r"[\"'](?P<expression>.+)[\"'][.!]?\s*$", re.IGNORECASE,
)
_SELF_JUDGMENT_HISTORY = re.compile(
    r"i\s+(?:(?:used to)\s+(?P<present>like|prefer|believe|think|want|intend)|"
    r"(?P<past>liked|preferred|believed|thought|wanted|intended))\s+(?P<view>.+)",
    re.IGNORECASE,
)
_PRESENT_SELF_JUDGMENT = dict(zip(
    ('liked', 'preferred', 'believed', 'thought', 'wanted', 'intended'),
    ('like', 'prefer', 'believe', 'think', 'want', 'intend'),
))


def self_history_is_grounded(
    text: str, *, prior_expressions: tuple[str, ...] = (),
    observed_events: tuple[str, ...] = (),
) -> bool:
    """Past speech proves speech; only an observed event can prove an action.

    This is a conservative lexical admission guard, not a semantic classifier.
    Current opinions need no historical witness. Reject mixed candidates rather
    than manufacture a cleaned preference or infer an event from topic overlap.
    """
    normalized = _normalized_evidence(text)
    events = {_normalized_evidence(event).strip(' .!') for event in observed_events}
    expressions = tuple(_normalized_evidence(value) for value in prior_expressions
                        if _normalized_evidence(value) != normalized)
    # Clause boundaries let a present judgment cite a separately witnessed past
    # event. Do not crop an event down to a favorable substring of its witness.
    for clause in re.split(r'[;\n]|(?<=[.!?])\s+|\bbecause\b|\bbut\b', normalized):
        speech = _SELF_SPEECH_HISTORY.fullmatch(clause.strip())
        unquoted = re.sub(r'(?<!\w)([\"\']).*?\1(?!\w)', '', clause)
        # Quotes alone do not turn a claimed personal event into fiction. Keep
        # an unattributed quoted assertion subject to the same history checks.
        framed_quote = re.search(r'\b(?:fiction|fictional|story|novel|line|phrase|'
                                 r'quote|quotation|character|poem|passage|lyrics?)\b', unquoted)
        personal = re.sub(r'\bakane\b', 'i', unquoted if framed_quote else clause)
        if not (
            _SELF_HISTORY.search(personal)
            or _SELF_BIOGRAPHY.search(personal)
            or (_SELF_PAST_FRAMING.search(personal)
                and re.search(r'\b(?:i|we|me|my|us|our)\b', personal))
        ):
            continue
        if speech:
            expression = speech['expression'].strip(' .!')
            if expression and any(_world_span(expression, source) for source in expressions):
                continue
        judgment = _SELF_JUDGMENT_HISTORY.fullmatch(clause.strip(' ,.!'))
        if judgment:
            view = judgment['view']
            verb = judgment['present'] or _PRESENT_SELF_JUDGMENT[judgment['past']]
            # An exact prior expressed view establishes that view's history,
            # never an additional action or autobiographical clause inside it.
            if self_history_is_grounded(view) and any(
                _world_span(f'i {verb} {view}', source) for source in expressions
            ):
                continue
        if clause.strip(' ,.!') in events:
            continue
        return False
    return True


def observed_self_history(
    experiences: tuple[Experience, ...], outcomes: tuple[Outcome, ...], *,
    profile_id: str, before: float,
) -> tuple[str, ...]:
    """Exact trusted witnesses, retaining the virtual environment's scope."""
    events = [item.what_happened.replace('Akane', 'I') for item in experiences
              if item.profile_id == profile_id and item.created_at <= before
              and item.reason == 'grounded:virtual_environment']
    # Action-episode descriptions are trusted result receipts, not model prose.
    events.extend(item.description for item in outcomes
                  if item.profile_id == profile_id and item.created_at <= before
                  and item.reason == 'grounded:action_episode')
    return tuple(events)


def _unquoted_self_text(text: str) -> str:
    # Keep offsets when selecting a bounded witness, including an unfinished
    # quoted span in a stored prefix. Apostrophes inside words are not quotes.
    return re.sub(r'''"[^"]*(?:"|$)|“[^”]*(?:”|$)|(?<!\w)['‘](?:[^'’]|(?<=\w)['’](?=\w))*(?:['’]|$)''',
                  lambda match: ' ' * len(match.group()), text)


def self_expression_is_owned(kind: str, text: str) -> bool:
    """Advice and quoted speakers do not establish Akane's own disposition.

    Opinions may be implicit judgments. Preferences, interests and intentions
    need a personal subject or beneficiary beyond a rhetorical advice preface.
    This checks ownership, not whether the model's stance/topic is correct.
    """
    if kind not in {'preference', 'interest', 'goal'}:
        return True
    plain = _unquoted_self_text(_normalized_evidence(text))
    plain = re.sub(r"\bi(?:(?:'d|'ll)|\s+(?:do|don't|would|will|can|might|not|really|still|also))*"
                   r"\s+(?:think|believe|suggest|recommend|advise)\b", '', plain)
    plain = re.sub(r'\bmy (?:advice|suggestion|recommendation)\b', '', plain)
    if kind == 'interest':
        # Mentioning a subject, a denied visit, or an ability is not interest.
        # Negation remains eligible: "doesn't interest me" expresses disinterest.
        return bool(re.search(
            r"\bi(?:'d|'m)?\s+(?:(?:am|remain|feel|become|do|not|don't|would|really|still|also)\s+){0,3}"
            r'(?:curious|interested|intrigued|fascinated|drawn|wonder|like|love|enjoy|prefer|favor|oppose)\b|'
            r"\bi(?:'d)?\s+(?:want|hope|intend|aim)(?:\s+\w+){0,2}\s+"
            r'(?:learn|understand|explore|investigate|know)\b|'
            r'\b(?:interests?|fascinates?|intrigues?|appeals?)(?:\s+\w+){0,4}\s+me\b|'
            r'\bmy\s+(?:curiosity|interest|attention)\b|'
            r'\b(?:makes?|keeps?|leaves?)\s+me\s+(?:curious|interested|intrigued)\b', plain))
    return bool(re.search(r'\b(?:i|me|my|mine|myself)\b', plain))


def validate_semantic_evidence(
    raw: str | SemanticSidecar,
    user_turn: Turn,
    assistant_turn: Turn,
    recent_turns: tuple[Turn, ...] = (),
    *, observed_history: tuple[str, ...] = (),
    historical_expressions: tuple[str, ...] = (),
) -> SemanticEvidence | None:
    """Validate one model sidecar against canonical source turns, failing closed."""

    source = raw.raw if isinstance(raw, SemanticSidecar) else raw
    if (
        not source
        or len(source) > 1024
        or not user_turn.id
        or not assistant_turn.id
        or user_turn.id == assistant_turn.id
        or user_turn.role != "user"
        or assistant_turn.role != "assistant"
        or user_turn.profile_id != assistant_turn.profile_id
        or user_turn.conversation_id != assistant_turn.conversation_id
    ):
        return None
    payload = next((item for item in _semantic_records(raw)
                    if item["k"] in _SEMANTIC_KIND_CODES), None)
    if payload is None:
        return None
    kind = _SEMANTIC_KIND_CODES.get(payload["k"], "")
    current_self = kind in _SELF_EVIDENCE_KINDS
    source_reference = type(payload.get("e")) is int
    if source_reference:
        expected = (
            0 if kind in _EVENT_EVIDENCE_KINDS | {"resolved_curiosity"} else
            1 if kind in (_PREDICTION_EVIDENCE_KINDS
                | _DEVELOPMENTAL_GOAL_EVIDENCE_KINDS | {"unresolved_curiosity"}) else None
        )
        if payload["e"] != expected:
            return None
        # The model selects a canonical current source, never replacement text.
        # Validate the complete source below before any bounded storage excerpt.
        payload = {**payload, "e": (user_turn if expected == 0 else assistant_turn).content}
    if current_self:
        # Current Self has exactly one source; the model cannot replace it.
        if "e" in payload:
            return None
        payload = {**payload, "e": assistant_turn.content}
    keys = set(payload)
    if (
        not _SEMANTIC_REQUIRED_KEYS <= keys
        or frozenset(keys - _SEMANTIC_REQUIRED_KEYS)
        not in _SEMANTIC_OPTIONAL_KEY_SETS
    ):
        return None
    if any(not isinstance(payload[key], str) for key in keys):
        return None
    topic = payload["t"].strip()
    stance = _SEMANTIC_STANCE_CODES.get(payload["s"], "")
    durability = _SEMANTIC_DURABILITY_CODES.get(payload["d"], "")
    evidence = payload["e"] if current_self else payload["e"].strip()
    value = payload.get("v", "").strip()
    action = payload.get("a", "").strip()
    behavior = payload.get("b", "").strip()
    effect = payload.get("f", "").strip()
    strategy = payload.get("r", "").strip()
    curiosity_focus = payload.get("w", "").strip()
    developmental_goal = payload.get("j", "").strip()
    confidence_band = _PREDICTION_CONFIDENCE_CODES.get(payload.get("q", ""), "")
    if (
        kind not in _SEMANTIC_KINDS
        or stance not in _SEMANTIC_STANCES
        or durability not in _SEMANTIC_DURABILITY
        or len(topic) > 120
        or (not current_self and not source_reference and len(evidence) > 280)
        or len(value) > 80
        or len(action) > 280
        or len(behavior) > 120
        or len(effect) > 160
        or len(strategy) > 160
        or len(curiosity_focus) > 160
        or len(developmental_goal) > 160
    ):
        return None
    if kind == "none":
        return None
    if kind in _PREDICTION_EVIDENCE_KINDS:
        if durability != "task-local" or not confidence_band:
            return None
    elif "q" in payload:
        return None
    elif durability != "candidate":
        return None
    if not topic or not evidence or not self_topic_terms(topic):
        return None
    if action and kind not in (_EVENT_EVIDENCE_KINDS | _PREDICTION_EVIDENCE_KINDS):
        return None
    if (behavior or effect) and kind not in _EVENT_EVIDENCE_KINDS:
        return None
    if strategy and kind not in _EVENT_EVIDENCE_KINDS:
        return None
    if curiosity_focus and kind not in _CURIOSITY_EVIDENCE_KINDS:
        return None
    if developmental_goal and kind not in (
        _DEVELOPMENTAL_GOAL_EVIDENCE_KINDS | _EVENT_EVIDENCE_KINDS
    ):
        return None
    if bool(behavior) != bool(effect):
        return None
    if behavior and (not self_topic_terms(behavior) or not self_topic_terms(effect)):
        return None
    if strategy and not _valid_semantic_phrase(strategy):
        return None
    if curiosity_focus and not _valid_semantic_phrase(curiosity_focus):
        return None
    if developmental_goal and not _valid_semantic_phrase(developmental_goal):
        return None
    evidence_key = _normalized_evidence(evidence)
    reply_key = _normalized_evidence(assistant_turn.content)
    user_key = _normalized_evidence(user_turn.content)
    if not evidence_key:
        return None
    if kind in _SELF_EVIDENCE_KINDS and (
        self_source_is_recall(user_turn.content)
        or _has_non_durable_framing(user_key)
        or _has_non_durable_framing(evidence_key)
        or _CURRENT_WANT.search(user_key)
        or _CURRENT_WANT.search(evidence_key)
    ):
        return None
    if current_self and not self_expression_is_owned(kind, evidence):
        return None
    if current_self and not self_history_is_grounded(
        evidence,
        prior_expressions=historical_expressions + tuple(turn.content for turn in recent_turns
            if turn.role == 'assistant' and turn.id != assistant_turn.id
            and turn.profile_id == user_turn.profile_id
            and turn.conversation_id == user_turn.conversation_id
            and turn.created_at < assistant_turn.created_at),
        observed_events=observed_history,
    ):
        return None
    if kind in _CURIOSITY_EVIDENCE_KINDS:
        if (
            not curiosity_focus
            or action or value or behavior or effect or strategy
            or confidence_band
        ):
            return None
        if kind == "unresolved_curiosity":
            if evidence_key not in reply_key or evidence_key in user_key:
                return None
        elif evidence_key not in user_key:
            return None
        return SemanticEvidence(
            kind, topic, stance, durability, evidence,
            curiosity_focus=curiosity_focus,
        )
    if kind in _DEVELOPMENTAL_GOAL_EVIDENCE_KINDS:
        if (
            _has_non_durable_framing(user_key)
            or _has_non_durable_framing(evidence_key)
            or _CURRENT_WANT.search(user_key)
            or _CURRENT_WANT.search(evidence_key)
        ):
            return None
        expected_stance = (
            "positive" if kind == "developmental_goal" else "negative"
        )
        if (
            not developmental_goal
            or action or value or behavior or effect or strategy
            or curiosity_focus or confidence_band
            or stance != expected_stance
            or evidence_key not in reply_key
            or evidence_key in user_key
        ):
            return None
        return SemanticEvidence(
            kind, topic, stance, durability, evidence,
            developmental_goal=developmental_goal,
        )
    if kind in _PREDICTION_EVIDENCE_KINDS:
        action_key = _normalized_evidence(action)
        expected_stance = "positive" if kind == "prediction_success" else "negative"
        if (
            not action_key
            or not self_topic_terms(action)
            or action_key not in reply_key
            or evidence_key not in reply_key
            or evidence_key in user_key
            or stance != expected_stance
            or value
        ):
            return None
        return SemanticEvidence(
            kind, topic, stance, durability, evidence,
            action=action,
            action_turn_id=assistant_turn.id,
            confidence_band=confidence_band,
        )
    if kind in _EVENT_EVIDENCE_KINDS:
        if (
            evidence_key not in user_key or value
            or not _feedback_kind_is_grounded(kind, evidence, user_turn.content, action)
            or (kind in {"task_success", "positive_feedback"} and stance != "positive")
            or (kind in {"task_failure", "negative_feedback"} and stance != "negative")
        ):
            return None
        if not action:
            if behavior or effect or strategy or developmental_goal:
                return None
            return SemanticEvidence(kind, topic, stance, durability, evidence)
        action_turn = _feedback_action_turn(action, evidence, user_turn, tuple(
            turn for turn in recent_turns if turn.id not in {user_turn.id, assistant_turn.id}
        ))
        if action_turn is None:
            return None
        return SemanticEvidence(
            kind, topic, stance, durability, evidence,
            action=action, action_turn_id=action_turn.id,
            behavior=behavior, effect=effect,
            strategy=strategy,
            developmental_goal=developmental_goal,
        )
    if developmental_goal:
        return None
    if evidence_key not in reply_key or evidence_key in user_key:
        return None
    if stance == "comparative":
        value_key = _normalized_evidence(value)
        if not value_key or value_key not in evidence_key:
            return None
    elif value:
        return None
    return SemanticEvidence(kind, topic, stance, durability, evidence, value)


def _world_span(span: str, text: str) -> bool:
    """Exact normalized lexical witness, never a language/meaning classifier."""
    span, text = _normalized_evidence(span), _normalized_evidence(text)
    start = text.find(span)
    while span and start >= 0:
        end = start + len(span)
        if (start == 0 or not text[start - 1].isalnum()) and (
            end == len(text) or not text[end].isalnum()
        ):
            return True
        start = text.find(span, start + 1)
    return False


def _world_source_is_asserted(text: str) -> bool:
    """Conservative source rejection, never inference of a World kind/value."""
    normalized = _normalized_evidence(text)
    return bool(normalized) and not (
        normalized.startswith(_HYPOTHETICAL_OPENERS)
        or bool(_FICTIONAL_FRAMING.search(normalized))
        or bool(_NONLITERAL_FRAMING.search(normalized))
        or any(marker in normalized for marker in _HYPOTHETICAL_MARKERS)
        or re.search(r"\b(?:if|whether|maybe|perhaps|possibly|probably|may|might|could|"
                     r"hypothetical|hypothetically|imagine|imagined|pretend|suppose|"
                     r"uncertain|unsure|apparently|reportedly)\b", normalized)
        or re.search(r"\b(?:according to|said|says|think|thinks|believe|believes)\b", normalized)
        or re.match(r"(?:who|what|when|where|why|how|is|are|was|were|does|do|did|"
                    r"can|could|would|should|will|has|have)\b", normalized)
        or re.match(r"[^.!?\n]*\?", normalized)
    )


def self_source_is_recall(text: str) -> bool:
    """An ownership lookup supplies no new personal judgment evidence.

    The reply may still answer freely. This conservative admission guard keeps
    a requested attribution from becoming a newly invented disposition; an
    explicit request for Akane's current evaluation remains eligible.
    """
    if re.search(r"\b(?:do|would) you\b|\byour (?:own|present|current)\b|\bto you\b", text, re.I):
        return False
    return bool(re.match(
        r"\s*whose\s+(?:taste|preference|opinion|interest|goal)s?\b", text, re.I,
    ))


def _has_non_durable_framing(normalized: str) -> bool:
    return (
        normalized.startswith(_HYPOTHETICAL_OPENERS)
        or any(marker in normalized for marker in _HYPOTHETICAL_MARKERS)
        or any(marker in normalized for marker in _TEMPORARY_MARKERS)
        or bool(_FICTIONAL_FRAMING.search(normalized))
        or bool(_NONLITERAL_FRAMING.search(normalized))
    )


def _is_other_person_claim(text: str) -> bool:
    return bool(
        _OTHER_PERSON_CLAIM.search(text)
        or _NAMED_PERSON_CLAIM.search(text)
        or _OTHER_PERSON_FRAMING.search(text)
    )


def memory_report_is_supported(source: str, *, personal: bool = False) -> bool:
    normalized = _normalized_evidence(source)
    return bool(source.strip()) and not (
        source.rstrip().endswith('?') or _has_non_durable_framing(normalized)
        or _CURRENT_WANT.search(normalized) or _is_other_person_claim(source)
        or _QUOTED_FIRST_PERSON.search(source)
    ) and (not personal or bool(re.search(r'\b(?:i|my|we|our)\b', source, re.I)))


def memory_recall_evidence(raw: str | SemanticSidecar, user_turn: Turn):
    """Bind model-written search labels to one actual personal user report.

    Labels affect retrieval only. They are not asserted facts, experiences or
    evidence of an action, and are never substituted for the source text.
    """
    record = next((row for row in _semantic_records(raw)
                   if row['k'] in _WORLD_EVIDENCE_KINDS and 'r' in row and 'c' in row), None)
    if (record is None or not isinstance(record['r'], str) or not isinstance(record['c'], str)
            or record['r'] not in MEMORY_RECALL_ROLES
            or user_turn.role != 'user'
            or not memory_report_is_supported(user_turn.content, personal=True)):
        return None
    cues = tuple(' '.join(cue.split()).casefold() for cue in record['c'].split(','))
    if (not 1 <= len(cues) <= 6 or any(not cue or len(cue.encode()) > 48 for cue in cues)
            or sum(len(cue.encode()) for cue in cues) > 160
            or any(not all(c.isalnum() or c in " -'" for c in cue) for cue in cues)):
        return None
    return record['r'], tuple(dict.fromkeys(cues))


def same_experience_source(previous: Experience, candidate: Experience) -> bool:
    """A summary of one source cannot become independent reinforcement."""
    if (
        previous.profile_id != candidate.profile_id
        or not (
            previous.kind == candidate.kind
            or {previous.kind, candidate.kind} <= (_SELF_EVIDENCE_KINDS | {"mind_change"})
        )
        or previous.subject != candidate.subject
        or not candidate.source_turn_ids
        or set(previous.source_turn_ids) != set(candidate.source_turn_ids)
    ):
        return False
    return (
        text_key(previous.topic) == text_key(candidate.topic)
        or relevance(previous.topic, candidate.topic) >= 0.85
    )


def same_experience(previous: Experience, candidate: Experience) -> bool:
    if not same_experience_source(previous, candidate):
        return False
    previous_response = text_key(previous.akane_response)
    candidate_response = text_key(candidate.akane_response)
    previous_terms = lexical_terms(previous.akane_response)
    candidate_terms = lexical_terms(candidate.akane_response)
    response_jaccard = len(previous_terms & candidate_terms) / max(
        1, len(previous_terms | candidate_terms),
    )
    return (
        previous_response == candidate_response
        or response_jaccard >= 0.72
        or SequenceMatcher(None, previous_response, candidate_response).ratio() >= 0.82
    )


def behavioral_tendency_state(item: BehavioralTendency) -> str:
    if item.status == "uncertain":
        return "uncertain"
    if len(item.supporting_outcome_ids) < 2:
        return "weak"
    return "reinforced"


def _development_similarity(left: str, right: str) -> float:
    """Require overlap across both learning targets, not just the shorter one."""
    if text_key(left) == text_key(right):
        return 1.0
    left_terms, right_terms = self_topic_terms(left), self_topic_terms(right)
    return len(left_terms & right_terms) / max(1, len(left_terms | right_terms))


def _procedure_similarity(context: str, left: str, right: str) -> float:
    """Structured virtual procedures include authoritative parameters.

    Word overlap must not merge on/off, object IDs or numeric answers. Legacy
    natural-language strategies retain their existing semantic matching.
    """
    prefix = tuple(verb + ' ' + context + ' ' for verb in ('inspect', 'verify', 'continue'))
    if context.startswith('virtual ') and (left.startswith(prefix) or right.startswith(prefix)):
        return float(text_key(left) == text_key(right))
    return _development_similarity(left, right)


def find_behavioral_tendency(
    tendencies: tuple[BehavioralTendency, ...],
    context: str,
    behavior: str,
) -> BehavioralTendency | None:
    return _best_match(
        tendencies,
        lambda item: min(
            _development_similarity(context, item.context),
            _procedure_similarity(context, behavior, item.behavior),
        ),
    )


def _outcome_result(result: str) -> str:
    return (
        "success"
        if result in {"task_success", "positive_feedback"}
        else "failure"
    )


def _independent_outcome(item: BehavioralTendency | Strategy, outcome: Outcome,
                         prior_outcomes: tuple[Outcome, ...]) -> bool:
    evidence = {*item.supporting_outcome_ids, *item.contradiction_outcome_ids}
    return not any(previous.id in evidence and previous.action_turn_id == outcome.action_turn_id
                   for previous in prior_outcomes)


def _update_procedure_evidence(
    current: BehavioralTendency | Strategy, outcome: Outcome, *, now: float,
    effect: str = "",
) -> tuple[str, BehavioralTendency | Strategy]:
    """One evidence lifecycle; a strategy targets success, a tendency predicts effects."""
    actual = _outcome_result(outcome.result)
    supported = actual == current.expected_result
    strength, confidence = _adjust_strength_confidence(
        current.strength, current.confidence,
        TENDENCY_REINFORCE_STRENGTH if supported else -TENDENCY_WEAKEN_STRENGTH,
        TENDENCY_REINFORCE_CONFIDENCE if supported else -TENDENCY_WEAKEN_CONFIDENCE,
    )
    if supported:
        supports = _bounded_ids(current.supporting_outcome_ids, (outcome.id,))
        return "reinforce", replace(
            current, strength=strength, confidence=confidence,
            status="active" if len(supports) > len(current.contradiction_outcome_ids) else current.status,
            updated_at=now, supporting_outcome_ids=supports,
        )
    contradictions = _bounded_ids(current.contradiction_outcome_ids, (outcome.id,))
    if not current.contradiction_outcome_ids:
        return "weaken", replace(
            current, strength=strength, confidence=confidence, status="uncertain",
            updated_at=now, contradiction_outcome_ids=contradictions,
        )
    if isinstance(current, BehavioralTendency):
        return "revise", replace(
            current, expected_effect=compact_text(effect, 160), expected_result=actual,
            strength=0.45, confidence=0.70, status="active", updated_at=now,
            supporting_outcome_ids=contradictions,
            contradiction_outcome_ids=current.supporting_outcome_ids[-6:],
            revision_count=current.revision_count + 1,
        )
    return "retire", replace(
        current, strength=strength, confidence=confidence, status="retired",
        updated_at=now, contradiction_outcome_ids=contradictions,
        revision_count=current.revision_count + 1,
    )


def _tendency_change_from_outcome(
    profile_id: str,
    tendencies: tuple[BehavioralTendency, ...],
    prior_outcomes: tuple[Outcome, ...],
    outcome: Outcome,
    semantic: SemanticEvidence,
    *,
    now: float,
) -> BehavioralTendencyChange | None:
    if not semantic.behavior or not semantic.effect:
        return None
    current = find_behavioral_tendency(
        tendencies, semantic.topic, semantic.behavior,
    )
    if current is not None and not _independent_outcome(current, outcome, prior_outcomes):
        return None
    actual = _outcome_result(outcome.result)
    if current is None:
        item = BehavioralTendency(
            id=_stable_id(
                "tendency", profile_id, text_key(semantic.topic),
                text_key(semantic.behavior),
            ),
            profile_id=profile_id,
            context=compact_text(semantic.topic, 120),
            behavior=compact_text(semantic.behavior, 120),
            expected_effect=compact_text(semantic.effect, 160),
            expected_result=actual,
            strength=TENDENCY_FORM_STRENGTH,
            confidence=TENDENCY_FORM_CONFIDENCE,
            status="active",
            created_at=now,
            updated_at=now,
            supporting_outcome_ids=(outcome.id,),
        )
        return BehavioralTendencyChange("form", item)
    action, item = _update_procedure_evidence(current, outcome, now=now, effect=semantic.effect)
    return BehavioralTendencyChange(action, item, current.id)


def strategy_state(item: Strategy) -> str:
    if item.status == "retired":
        return "retired"
    if item.status == "uncertain":
        return "uncertain"
    if len(item.supporting_outcome_ids) < 2:
        return "weak"
    return "reinforced"


def find_strategy(
    strategies: tuple[Strategy, ...], context: str, procedure: str,
) -> Strategy | None:
    return _best_match(
        strategies,
        lambda item: min(
            _development_similarity(context, item.context),
            _procedure_similarity(context, procedure, item.procedure),
        ),
    )


def _strategy_change_from_outcome(
    profile_id: str,
    strategies: tuple[Strategy, ...],
    prior_outcomes: tuple[Outcome, ...],
    outcome: Outcome,
    semantic: SemanticEvidence,
    *,
    now: float,
) -> StrategyChange | None:
    if not semantic.strategy:
        return None
    current = find_strategy(strategies, semantic.topic, semantic.strategy)
    if current is not None:
        if not _independent_outcome(current, outcome, prior_outcomes):
            return None
        action, item = _update_procedure_evidence(current, outcome, now=now)
        return StrategyChange(action, item, current.id)
    actual = _outcome_result(outcome.result)
    if actual != "success":
        return None
    replacement_candidates = tuple(
        item for item in strategies
        if item.status in {"uncertain", "retired"}
        and _development_similarity(semantic.topic, item.context) >= 0.67
        and (outcome.reason != 'grounded:virtual_environment' or item.procedure == semantic.strategy)
    )
    if len(replacement_candidates) == 1:
        replaced = replacement_candidates[0]
        if not _independent_outcome(replaced, outcome, prior_outcomes):
            return None
        item = replace(
            replaced,
            procedure=compact_text(semantic.strategy, 160),
            expected_result="success",
            strength=STRATEGY_FORM_STRENGTH,
            confidence=STRATEGY_FORM_CONFIDENCE,
            status="active",
            updated_at=now,
            supporting_outcome_ids=(outcome.id,),
            contradiction_outcome_ids=(
                *replaced.supporting_outcome_ids,
                *replaced.contradiction_outcome_ids,
            )[-6:],
            revision_count=replaced.revision_count + 1,
        )
        return StrategyChange("revise", item, replaced.id)
    if len(replacement_candidates) > 1:
        return None
    item = Strategy(
        id=_stable_id(
            "strategy", profile_id, text_key(semantic.topic),
            text_key(semantic.strategy),
        ),
        profile_id=profile_id,
        context=compact_text(semantic.topic, 120),
        procedure=compact_text(semantic.strategy, 160),
        expected_result="success",
        strength=STRATEGY_FORM_STRENGTH,
        confidence=STRATEGY_FORM_CONFIDENCE,
        status="active",
        created_at=now,
        updated_at=now,
        supporting_outcome_ids=(outcome.id,),
    )
    return StrategyChange("form", item)


def curiosity_state(item: Curiosity) -> str:
    if item.status == "resolved":
        return "resolved"
    if item.status == "uncertain":
        return "uncertain"
    if len(item.source_ids) < 3:
        return "weak"
    return "reinforced"


def find_curiosity(
    curiosities: tuple[Curiosity, ...], topic: str,
) -> Curiosity | None:
    return _best_match(
        curiosities,
        lambda item: _development_similarity(topic, item.topic),
    )


def developmental_goal_state(item: DevelopmentalGoal) -> str:
    if item.status in {"candidate", "satisfied", "retired"}:
        return item.status
    if item.progress_outcome_ids:
        return "progressing"
    return "active"


def find_developmental_goal(
    goals: tuple[DevelopmentalGoal, ...], topic: str, goal: str,
) -> DevelopmentalGoal | None:
    return _best_match(
        goals,
        lambda item: min(
            _development_similarity(topic, item.topic),
            _development_similarity(goal, item.goal),
        ),
    )


def _grounded_outcome_relevant_to_goal(
    goal: DevelopmentalGoal, outcome: Outcome,
) -> bool:
    goal_topic_terms = self_topic_terms(goal.topic)
    grounded_terms = self_topic_terms(
        f"{outcome.action} {outcome.description}",
    )
    return bool(goal_topic_terms) and goal_topic_terms <= grounded_terms


def developmental_goal_completed_by_outcome(
    goal: DevelopmentalGoal, outcome: Outcome,
    *, feedback_text: str | None = None,
) -> bool:
    """A successful sub-action is progress, not completion of a whole goal.

    Completion requires a trusted result explicitly naming the entire owned
    goal. The goal's content is unrestricted; neither categories nor a number
    of successful trials can substitute for that completion witness.
    """
    if (
        goal.profile_id != outcome.profile_id
        or _outcome_result(outcome.result) != "success"
        or outcome.reason not in {
            "grounded:user-task-success", "grounded:user-positive-feedback",
            "grounded:action_episode", "grounded:virtual_environment",
        }
        or not _grounded_outcome_relevant_to_goal(goal, outcome)
        or "?" in outcome.description
        or outcome.description.lstrip().startswith(('"', "'", "“", ">"))
        or (feedback_text is not None
            and _normalized_evidence(feedback_text) != _normalized_evidence(outcome.description))
    ):
        return False
    goal_text = re.escape(text_key(goal.goal))
    completion = (rf"(?:your|akane s) goal {goal_text} (?:is|has been) "
                  r"(?:complete|completed|fulfilled|achieved|met)")
    action_result = re.escape(text_key(outcome.action)) + r" (?:worked|succeeded)"
    return bool(goal_text and re.fullmatch(
        rf"(?:{completion}(?: it (?:worked|succeeded))?|"
        rf"{action_result} {completion}|{completion} {action_result})",
        text_key(outcome.description)))


def _resolve_prediction(
    prediction: Prediction,
    outcome: Outcome,
    *,
    now: float,
) -> Prediction:
    actual = _outcome_result(outcome.result)
    if prediction.expected_result == actual:
        category = "none"
        error = 0.0
    elif prediction.expected_result == "success":
        category = "negative_error"
        error = -prediction.confidence
    else:
        category = "positive_surprise"
        error = prediction.confidence
    return replace(
        prediction,
        status="resolved",
        outcome_id=outcome.id,
        actual_result=actual,
        error_category=category,
        error_value=round(error, 6),
        resolved_at=now,
    )


def independent_self_outcome(
    profile_id: str, kind: str, topic: str,
    experiences: tuple[Experience, ...], outcomes: tuple[Outcome, ...], *,
    before: float, self_item_id: str = "", source_ids: tuple[str, ...] | None = None,
) -> Outcome | None:
    """Find a relevant consequence not already credited to this judgment's history.

    Shared with Store's mutation boundary so a forged reinforcement proposal
    cannot substitute another model utterance for independent experience.
    """
    topic_terms = self_topic_terms(topic)
    prior_judgments = tuple(item for item in experiences
        if item.profile_id == profile_id and item.subject == 'akane' and (item.kind == kind or (
            item.kind == 'mind_change' and self_item_id in item.self_item_ids))
        and _same_self_topic(kind, item.topic, topic))
    used_actions = {item.action_turn_id for item in outcomes
        if item.profile_id == profile_id and item.source_turn_ids and any(
            set(item.source_turn_ids) <= set(previous.source_turn_ids)
            for previous in prior_judgments)}
    return next((item for item in reversed(outcomes)
        if item.profile_id == profile_id and item.created_at <= before
        and item.reason in {'grounded:virtual_environment', 'grounded:action_episode',
                            'grounded:user-task-success', 'grounded:user-task-failure',
                            'grounded:user-positive-feedback', 'grounded:user-negative-feedback'}
        and topic_terms and topic_terms <= self_topic_terms(f'{item.action} {item.description}')
        and item.source_turn_ids and item.action_turn_id not in used_actions
        and (source_ids is None or set(item.source_turn_ids) <= set(source_ids))), None)
