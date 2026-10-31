"""Model-free post-turn derivation of grounded developmental state."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from app.core.state import (
    BehavioralTendency,
    BehavioralTendencyChange,
    Curiosity,
    CuriosityChange,
    DevelopmentalGoal,
    DevelopmentalGoalChange,
    Experience,
    Memory,
    MemoryChange,
    Outcome,
    PREDICTION_TTL_SECONDS,
    UNRESOLVED_PREDICTION_LIMIT,
    Prediction,
    PredictionChange,
    SelfChange,
    SelfItem,
    Strategy,
    StrategyChange,
    Turn,
    WorldChange,
    WorldEntity,
    WorldEvent,
    WorldFact,
    WorldRelation,
    WorldSnapshot,
    WorldSource,
    WorldState,
    clamp,
)
from app.core.utils import compact_text, lexical_terms, relevance, text_key
from app.core.retrieval import same_memory_fact


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
_MEMORY_CUES = re.compile(
    r"\b(?:call me|i am|i'm|i have|i live|i love|i hate|i like|i own|i prefer|"
    r"i study|i work|i(?:'m| am) working on|my birthday|my favorite|my job|"
    r"my name|please remember|remember that|i moved|i promise|i will|i need to|let'?s)\b",
    re.IGNORECASE,
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

SELF_FORM_STRENGTH = 0.35
SELF_FORM_CONFIDENCE = 0.65
SELF_REINFORCE_STRENGTH = 0.10
SELF_REINFORCE_CONFIDENCE = 0.05
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
STRATEGY_REINFORCE_STRENGTH = 0.10
STRATEGY_REINFORCE_CONFIDENCE = 0.05
STRATEGY_WEAKEN_STRENGTH = 0.15
STRATEGY_WEAKEN_CONFIDENCE = 0.10
CURIOSITY_FORM_STRENGTH = 0.35
CURIOSITY_FORM_CONFIDENCE = 0.65
CURIOSITY_REINFORCE_STRENGTH = 0.10
CURIOSITY_REINFORCE_CONFIDENCE = 0.05
CURIOSITY_WEAKEN_STRENGTH = 0.15
CURIOSITY_WEAKEN_CONFIDENCE = 0.10
DEVELOPMENTAL_GOAL_FORM_STRENGTH = 0.25
DEVELOPMENTAL_GOAL_FORM_CONFIDENCE = 0.55
DEVELOPMENTAL_GOAL_REINFORCE_STRENGTH = 0.15
DEVELOPMENTAL_GOAL_REINFORCE_CONFIDENCE = 0.10
DEVELOPMENTAL_GOAL_PROGRESS_STRENGTH = 0.10
DEVELOPMENTAL_GOAL_PROGRESS_CONFIDENCE = 0.05
DEVELOPMENTAL_GOAL_WEAKEN_STRENGTH = 0.15
DEVELOPMENTAL_GOAL_WEAKEN_CONFIDENCE = 0.10


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


def form_self_item(
    profile_id: str,
    kind: str,
    topic: str,
    value: str,
    *,
    source_id: str,
    item_id: str = "",
    stance: str = "",
    semantic_value: str = "",
    now: float | None = None,
) -> SelfItem:
    current = time.time() if now is None else float(now)
    evidence = compact_text(value, 280)
    return SelfItem(
        id=item_id or f"self_{uuid.uuid4().hex}",
        profile_id=profile_id,
        kind=kind,
        topic=compact_text(topic, 120),
        value=evidence,
        strength=SELF_FORM_STRENGTH,
        confidence=SELF_FORM_CONFIDENCE,
        reason=_semantic_reason("form", source_id, stance, semantic_value),
        status="active",
        created_at=current,
        updated_at=current,
        source_ids=(source_id,),
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


def reinforce_self_item(
    item: SelfItem,
    *,
    source_id: str,
    value: str = "",
    stance: str = "",
    semantic_value: str = "",
    now: float | None = None,
) -> SelfItem:
    current = time.time() if now is None else float(now)
    sources = _bounded_ids(item.source_ids, (source_id,))
    status = item.status
    contradiction_ids = item.contradiction_ids
    if status == "uncertain" and len(sources) >= 2:
        status = "active"
        contradiction_ids = ()
    strength, confidence = _adjust_strength_confidence(
        item.strength,
        item.confidence,
        SELF_REINFORCE_STRENGTH,
        SELF_REINFORCE_CONFIDENCE,
    )
    return replace(
        item,
        value=(
            compact_text(value, 280)
            if item.status == "uncertain" and value else item.value
        ),
        strength=strength,
        confidence=confidence,
        reason=_semantic_reason(
            "reinforce", source_id, stance, semantic_value,
        ),
        status=status,
        updated_at=current,
        source_ids=sources,
        contradiction_ids=contradiction_ids,
    )


def refine_self_item(
    item: SelfItem,
    value: str,
    *,
    source_id: str,
    stance: str = "",
    semantic_value: str = "",
    now: float | None = None,
) -> SelfItem:
    refined = reinforce_self_item(
        item,
        source_id=source_id,
        value=value,
        stance=stance,
        semantic_value=semantic_value,
        now=now,
    )
    return replace(
        refined,
        value=compact_text(value, 280),
        reason=_semantic_reason(
            "refine", source_id, stance, semantic_value,
        ),
        revision_count=item.revision_count + 1,
    )


def weaken_self_item(
    item: SelfItem,
    value: str,
    *,
    source_id: str,
    stance: str = "",
    semantic_value: str = "",
    now: float | None = None,
) -> SelfItem:
    current = time.time() if now is None else float(now)
    return replace(
        item,
        value=compact_text(value, 280),
        strength=SELF_FORM_STRENGTH,
        confidence=SELF_FORM_CONFIDENCE,
        reason=_semantic_reason("weaken", source_id, stance, semantic_value),
        status="uncertain",
        updated_at=current,
        source_ids=(source_id,),
        contradiction_ids=_bounded_ids(item.source_ids, item.contradiction_ids),
        revision_count=item.revision_count + 1,
    )


def abandon_self_goal(
    item: SelfItem,
    value: str,
    *,
    source_id: str,
    stance: str,
    semantic_value: str = "",
    now: float | None = None,
) -> SelfItem:
    current = time.time() if now is None else float(now)
    return replace(
        item,
        value=compact_text(value, 280),
        reason=_semantic_reason("abandon", source_id, stance, semantic_value),
        status="abandoned",
        updated_at=current,
        source_ids=(source_id,),
        contradiction_ids=_bounded_ids(item.source_ids, item.contradiction_ids),
        revision_count=item.revision_count + 1,
    )


def revise_self_item(
    item: SelfItem,
    value: str,
    *,
    source_id: str,
    stance: str = "",
    semantic_value: str = "",
    now: float | None = None,
) -> SelfItem:
    current = time.time() if now is None else float(now)
    evidence = compact_text(value, 280)
    sources = _bounded_ids(item.contradiction_ids, (source_id,))
    prior_support = max(0, len(sources) - 1)
    strength, confidence = _adjust_strength_confidence(
        SELF_FORM_STRENGTH,
        SELF_FORM_CONFIDENCE,
        prior_support * SELF_REINFORCE_STRENGTH,
        prior_support * SELF_REINFORCE_CONFIDENCE,
    )
    return replace(
        item,
        value=evidence,
        strength=strength,
        confidence=confidence,
        reason=_semantic_reason("revise", source_id, stance, semantic_value),
        status="active",
        updated_at=current,
        source_ids=sources,
        contradiction_ids=(),
        revision_count=item.revision_count + 1,
    )


def _semantic_reason(
    action: str,
    source_id: str,
    stance: str = "",
    semantic_value: str = "",
) -> str:
    reason = f"experience:{action}:{source_id}"
    if stance in _SEMANTIC_STANCES:
        reason += f"|semantic:{stance}:{text_key(semantic_value)[:80]}"
    return reason


def _reason_semantics(reason: str) -> tuple[str, str]:
    marker = "|semantic:"
    if marker not in reason:
        return "", ""
    payload = reason.rpartition(marker)[2]
    stance, separator, value = payload.partition(":")
    if not separator or stance not in _SEMANTIC_STANCES:
        return "", ""
    return stance, value


def _legacy_stance(value: str) -> str:
    """Best-effort compatibility for Self formed before semantic sidecars."""

    normalized = text_key(value)
    terms = lexical_terms(value)
    if "not like" in normalized or terms & {
        "dislike", "harmful", "hate", "oppose", "unfair", "worse", "wrong",
    }:
        return "negative"
    if terms & {
        "adore", "better", "enjoy", "favorite", "fair", "good", "love",
        "meaningful", "prefer", "support", "valuable", "worth",
    }:
        return "positive"
    return ""


def _same_judgment(current: SelfItem, semantic: SemanticEvidence) -> bool:
    old_stance, old_value = _reason_semantics(current.reason)
    if not old_stance:
        old_stance = _legacy_stance(current.value)
    new_stance = semantic.stance
    new_value = text_key(semantic.value)
    if old_stance == "comparative" or new_stance == "comparative":
        return (
            old_stance == new_stance == "comparative"
            and bool(old_value)
            and old_value == new_value
        )
    if old_stance in {"positive", "negative"} and new_stance in {
        "positive", "negative",
    }:
        return old_stance == new_stance
    if old_stance and old_stance == new_stance:
        return True
    return (
        relevance(current.value, semantic.evidence) >= 0.42
        or self_topic_similarity(current.value, semantic.evidence) >= 0.5
    )


def _is_refinement(current: SelfItem, semantic: SemanticEvidence) -> bool:
    if current.kind != semantic.kind or current.kind != "preference":
        return False
    current_terms = self_topic_terms(current.topic)
    new_terms = self_topic_terms(semantic.topic)
    if not current_terms or not current_terms < new_terms:
        return False
    old_stance, _ = _reason_semantics(current.reason)
    if not old_stance:
        old_stance = _legacy_stance(current.value)
    return (
        old_stance == semantic.stance
        or old_stance == "positive" and semantic.stance == "comparative"
    )


def _normalized_evidence(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"',
    }))
    return " ".join(normalized.split()).casefold()


def _valid_semantic_phrase(value: str) -> bool:
    return bool(self_topic_terms(value)) and "\n" not in value and "\r" not in value


_POSITIVE_RESULT = re.compile(
    r"\b(?:worked|works|succeeded|succeeds|passed|helped|fixed|resolved|recovered|"
    r"(?:reveal(?:ed)?|identif(?:ied|y)|found|find) (?:the )?(?:root )?cause|"
    r"(?:was|is|were|are|proved) (?:a )?(?:success|successful|effective|useful|"
    r"helpful|correct|accurate|clear)|(?:completed|finished) successfully)\b",
)
_NEGATIVE_RESULT = re.compile(
    r"\b(?:failed|fails|failure|unsuccessful|broke|broken|"
    r"(?:was|is|were|are|proved) (?:incorrect|wrong|inaccurate|unclear|ineffective|unhelpful)|"
    r"(?:was|is|were|are|proved) not (?:successful|effective|useful|helpful|correct|accurate|clear)|"
    r"(?:not|never|didn't|doesn't|hasn't|haven't|couldn't|wouldn't) "
    r"(?:work|succeed|pass|help|fix|resolve|recover))\b",
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


_WORLD_EVIDENCE_KINDS = frozenset({"we", "ws", "wf", "wr", "wr-", "wv"})


def _semantic_records(raw: str) -> tuple[dict, ...]:
    """One object, or one Development + one World object in the same suffix."""
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
    return tuple(records)


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


def validate_semantic_evidence(
    raw: str,
    user_turn: Turn,
    assistant_turn: Turn,
    recent_turns: tuple[Turn, ...] = (),
    *, observed_history: tuple[str, ...] = (),
    historical_expressions: tuple[str, ...] = (),
) -> SemanticEvidence | None:
    """Validate one model sidecar against canonical source turns, failing closed."""

    if (
        not raw
        or len(raw) > 1024
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
        or (not current_self and len(evidence) > 280)
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
        _has_non_durable_framing(user_key)
        or _has_non_durable_framing(evidence_key)
        or _CURRENT_WANT.search(user_key)
        or _CURRENT_WANT.search(evidence_key)
    ):
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
        or any(marker in normalized for marker in _HYPOTHETICAL_MARKERS)
        or re.search(r"\b(?:if|whether|maybe|perhaps|possibly|probably|may|might|could|"
                     r"hypothetical|hypothetically|imagine|imagined|pretend|suppose|"
                     r"uncertain|unsure|apparently|reportedly)\b", normalized)
        or re.search(r"\b(?:according to|said|says|think|thinks|believe|believes)\b", normalized)
        or re.match(r"(?:who|what|when|where|why|how|is|are|was|were|does|do|did|"
                    r"can|could|would|should|will|has|have)\b", normalized)
        or re.match(r"[^.!?\n]*\?", normalized)
    )


def derive_world_changes(
    raw: str, user_turn: Turn, assistant_turn: Turn, *,
    world: WorldSnapshot,
    recent_turns: tuple[Turn, ...] = (),
    memories: tuple[Memory, ...] = (),
    experiences: tuple[Experience, ...] = (),
    now: float | None = None,
) -> tuple[WorldChange, ...]:
    """Validate semantic source witnesses before proposing Phase 3A mutations.

    The current user turn is the source, never model-supplied text. Conservative
    framing guards reject nonassertions; values and identities need source spans.
    This does not independently prove arbitrary natural-language entailment.
    """
    payload = next((item for item in _semantic_records(raw)
                    if item["k"] in _WORLD_EVIDENCE_KINDS), None)
    if payload is None or (
        user_turn.role != "user" or assistant_turn.role != "assistant"
        or not user_turn.id or not assistant_turn.id or user_turn.id == assistant_turn.id
        or user_turn.profile_id != assistant_turn.profile_id
        or user_turn.conversation_id != assistant_turn.conversation_id
    ):
        return ()
    kind = payload["k"]
    required = {"k", "t", "d"}
    if "z" in payload:
        required.add("e")
    if kind != "we":
        required.add("a")
    if kind in {"ws", "wf", "wr", "wr-"}:
        required.add("v")
    optional = {"z", "b"} if kind == "wr" else {"z"}
    if not required <= set(payload) or set(payload) - required - optional:
        return ()
    if any(not isinstance(value, str) or not value.strip() or len(value) > 280
           for value in payload.values()):
        return ()
    expected_scope = {"we": "na", "ws": "na", "wf": "da", "wr": "na",
                      "wr-": "nn", "wv": "pa"}
    if payload["d"] != expected_scope[kind]:
        return ()
    profile_id = user_turn.profile_id
    prior_users = tuple(turn for turn in recent_turns if (
        turn.role == "user" and turn.profile_id == profile_id
        and turn.conversation_id == user_turn.conversation_id
        and turn.id != user_turn.id and turn.created_at <= user_turn.created_at
    ))
    source_turn = user_turn
    sources = (WorldSource("user_turn", user_turn.id),)
    if "z" in payload:
        # Historical evidence can establish a durable fact/occurrence, never
        # silently revive an old mutable value as current.
        if kind not in {"wf", "wv"}:
            return ()
        matches = [item for item in (*memories, *experiences)
                   if item.id == payload["z"] and item.profile_id == profile_id]
        if len(matches) != 1:
            return ()
        source = matches[0]
        if isinstance(source, Experience) and source.kind not in _EVENT_EVIDENCE_KINDS:
            return ()
        if isinstance(source, Memory) and source.subject != "user":
            return ()
        quotes = [turn for turn in prior_users if turn.id in source.source_turn_ids
                  and _normalized_evidence(turn.content) == _normalized_evidence(payload["e"])]
        if len(quotes) != 1:
            return ()
        source_turn = quotes[0]
        sources = (WorldSource("memory" if isinstance(source, Memory) else "experience", source.id),
                   WorldSource("user_turn", source_turn.id))
    # Prior sources still require their complete explicit witness. Current
    # sources are bound directly, so qualifiers cannot be cropped or replaced.
    evidence = source_turn.content
    if not _world_source_is_asserted(evidence):
        return ()
    if re.search(r'["“”]', evidence) or evidence.lstrip().startswith('>'):
        return ()
    if kind != "wr-" and re.search(r"\b(?:not|never|no longer)\b",
                                  _normalized_evidence(evidence)):
        return ()
    current_time = time.time() if now is None else float(now)
    if source_turn.created_at > current_time:
        return ()
    timestamp = source_turn.created_at
    entities = tuple(item for item in world.entities if item.profile_id == profile_id)
    formed: list[WorldChange] = []

    def resolve(label: str, *, discourse: bool = False) -> WorldEntity | None:
        reference = label.strip()
        label = _normalized_evidence(label)
        if not label or len(label) > 200:
            return None
        matches = [item for item in entities
                   if item.id == reference or _normalized_evidence(item.label) == label]
        if len(matches) > 1:
            return None
        if matches:
            item = matches[0]
            if _world_span(item.label, evidence) or _world_span(item.id, evidence):
                return item
            if (discourse and source_turn == user_turn and prior_users
                    and re.search(r"\b(?:it|that)\b", evidence, re.I)):
                previous = max(prior_users, key=lambda turn: turn.created_at)
                mentioned = [item for item in entities if _world_span(item.label, previous.content)]
                if len(mentioned) == 1 and mentioned[0].id == item.id:
                    return item
            return None
        if not _world_span(label, evidence):
            return None
        item = WorldEntity(_stable_id("world_entity", profile_id, label), profile_id,
                           label, timestamp, timestamp, sources)
        if all(change.record.id != item.id for change in formed):
            formed.append(WorldChange("upsert", item))
        return item

    subject = resolve(payload["t"], discourse=True)
    if subject is None:
        return ()
    if kind == "we":
        return tuple(formed)
    attribute = _normalized_evidence(payload["a"])
    if not attribute or len(attribute) > 200:
        return ()
    if kind == "wf" and attribute in {"status", "activity", "focus", "intention"}:
        return ()
    value = payload.get("v", "").strip()
    if kind in {"ws", "wf"} and not _world_span(value, evidence):
        return ()
    if kind == "ws":
        current = world.current_state(subject.id, attribute)
        if current:
            sources += (WorldSource("world", current.id),)
        item = WorldState(_stable_id("world_state", subject.id, attribute), profile_id,
                          subject.id, attribute, value, 0.85, timestamp, timestamp, sources)
    elif kind == "wf":
        item = WorldFact(_stable_id("world_fact", subject.id, attribute, value), profile_id,
                         subject.id, attribute, value, 0.85, source_turn.created_at, sources)
    elif kind == "wv":
        if not _world_span(attribute, evidence):
            return ()
        item = WorldEvent(_stable_id("world_event", source_turn.id, subject.id, attribute),
                          profile_id, (subject.id,), attribute, 0.85,
                          source_turn.created_at, sources)
    else:
        target = resolve(value)
        if target is None:
            return ()
        current = None
        if kind == "wr-" or "b" in payload:
            old = resolve(payload.get("b", value))
            if old is None:
                return ()
            matches = [item for item in world.relations if item.profile_id == profile_id
                       and (item.subject_id, item.relation, item.object_id)
                       == (subject.id, attribute, old.id)]
            if len(matches) != 1:
                return ()
            current = matches[0]
            sources += (WorldSource("world", current.id),)
        item = WorldRelation(
            current.id if current else _stable_id("world_relation", subject.id, attribute, target.id),
            profile_id, subject.id, attribute, target.id, 0.85, timestamp, timestamp, sources,
        )
    return (*formed, WorldChange("remove" if kind == "wr-" else "upsert", item))


def memory_topic(text: str) -> str:
    normalized = text_key(text)
    slots = (
        ("name", ("my name", "call me")),
        ("birthday", ("my birthday",)),
        ("residence", ("i live", "i moved", "i am from", "i m from")),
        ("work", ("i work", "my job")),
        ("study", ("i study",)),
        ("project", ("working on", "my project")),
    )
    for slot, phrases in slots:
        if any(phrase in normalized for phrase in phrases):
            return slot
    if "my favorite" in normalized:
        tail = normalized.partition("my favorite")[2]
        return "favorite:" + " ".join(tail.split()[:2])
    return ""


def _find_memory(
    memories: tuple[Memory, ...], text: str, subject: str,
) -> Memory | None:
    owned = tuple(memory for memory in memories if memory.subject == subject)
    slot = memory_topic(text)
    if slot:
        for memory in reversed(owned):
            if memory_topic(memory.text) == slot:
                return memory
    return next((memory for memory in reversed(owned)
                 if same_memory_fact(text, memory.text)), None)


def _has_non_durable_framing(normalized: str) -> bool:
    return (
        normalized.startswith(_HYPOTHETICAL_OPENERS)
        or any(marker in normalized for marker in _HYPOTHETICAL_MARKERS)
        or any(marker in normalized for marker in _TEMPORARY_MARKERS)
    )


def _is_other_person_claim(text: str) -> bool:
    return bool(
        _OTHER_PERSON_CLAIM.search(text)
        or _NAMED_PERSON_CLAIM.search(text)
        or _OTHER_PERSON_FRAMING.search(text)
    )


def _memory_change(
    profile_id: str,
    memories: tuple[Memory, ...],
    user_turn: Turn,
    assistant_turn: Turn,
    *,
    now: float,
) -> MemoryChange | None:
    if (user_turn.profile_id != profile_id or assistant_turn.profile_id != profile_id
            or user_turn.role != "user" or assistant_turn.role != "assistant"
            or user_turn.conversation_id != assistant_turn.conversation_id
            or not user_turn.id or not assistant_turn.id or user_turn.id == assistant_turn.id
            or user_turn.created_at > now):
        return None
    # Eligibility uses the complete canonical source, before storage compaction.
    source = user_turn.content
    normalized = _normalized_evidence(source)
    if (
        not source.strip()
        or source.rstrip().endswith("?")
        or _has_non_durable_framing(normalized)
        or _CURRENT_WANT.search(normalized)
        or _is_other_person_claim(source)
        or _QUOTED_FIRST_PERSON.search(source)
        or not _MEMORY_CUES.search(source)
    ):
        return None
    text = compact_text(source, 360)
    normalized = text_key(text)
    now = user_turn.created_at
    subject = "user"
    kind = "fact"
    if any(cue in normalized for cue in ("i promise", "i will", "i need to")):
        kind = "commitment"
    elif any(cue in normalized for cue in ("i just", "i started", "i finished", "i moved")):
        kind = "event"
    if normalized.startswith(("let s ", "lets ")) and re.search(
        r"\b(?:agreed|deal|let's|lets|yes|we will)\b", assistant_turn.content, re.IGNORECASE,
    ):
        subject = "shared"
        kind = "commitment"
    current = _find_memory(memories, text, subject)
    source_ids = (user_turn.id,) if subject == "user" else (user_turn.id, assistant_turn.id)
    if current is not None and (
        now <= current.updated_at or user_turn.id in current.source_turn_ids
    ):
        return None
    if current is None:
        memory = Memory(
            id=f"memory_{uuid.uuid4().hex}",
            profile_id=profile_id,
            subject=subject,
            kind=kind,
            text=text,
            importance=0.55,
            confidence=0.75,
            created_at=now,
            updated_at=now,
            source_turn_ids=source_ids,
        )
        return MemoryChange("upsert", memory)
    memory = replace(
        current,
        subject=subject,
        kind=kind,
        text=text,
        importance=clamp(max(current.importance, 0.55), 0.0, 1.0),
        confidence=clamp(max(current.confidence, 0.75), 0.0, 1.0),
        updated_at=now,
        source_turn_ids=_bounded_ids(current.source_turn_ids, source_ids),
    )
    return MemoryChange("upsert", memory, current.id)


def _experience(
    profile_id: str,
    user_turn: Turn,
    assistant_turn: Turn,
    *,
    kind: str,
    subject: str,
    topic: str,
    what_happened: str,
    outcome: str,
    salience: float,
    reason: str,
    akane_response: str = "",
    self_item_ids: tuple[str, ...] = (),
    now: float,
) -> Experience:
    return Experience(
        id=f"experience_{uuid.uuid4().hex}",
        profile_id=profile_id,
        kind=kind,
        subject=subject,
        topic=compact_text(topic, 120),
        what_happened=compact_text(what_happened, 200),
        akane_response=compact_text(akane_response or assistant_turn.content, 280),
        outcome=compact_text(outcome, 160),
        salience=clamp(salience, 0.0, 1.0),
        reason=reason,
        created_at=now,
        self_item_ids=self_item_ids[:6],
        source_turn_ids=(user_turn.id, assistant_turn.id),
    )


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


def _event_evidence_experience(
    profile_id: str,
    user_turn: Turn,
    assistant_turn: Turn,
    semantic: SemanticEvidence,
    *,
    now: float,
) -> Experience:
    descriptions = {
        "correction": "The user corrected Akane and Akane responded to it.",
        "task_success": "The user reported that Akane's contribution succeeded.",
        "task_failure": "The user reported that Akane's contribution failed.",
        "positive_feedback": "The user gave explicit positive feedback about Akane's response.",
        "negative_feedback": "The user gave explicit negative feedback about Akane's response.",
    }
    return _experience(
        profile_id,
        user_turn,
        assistant_turn,
        kind=semantic.kind,
        subject="akane" if semantic.kind == "correction" else "shared",
        topic=semantic.topic,
        what_happened=descriptions[semantic.kind],
        outcome=(
            "Akane acknowledged the correction."
            if semantic.kind == "correction"
            else compact_text(user_turn.content, 160)
        ),
        salience=0.8 if semantic.kind == "correction" else 0.75,
        reason=(
            "user:correction"
            if semantic.kind == "correction"
            else f"user:{semantic.kind.replace('_', '-')}"
        ),
        akane_response=assistant_turn.content,
        now=now,
    )


def _grounded_outcome(
    profile_id: str,
    user_turn: Turn,
    semantic: SemanticEvidence,
    experiences: tuple[Experience, ...],
    *,
    now: float,
) -> Outcome:
    linked_experiences = tuple(
        item.id for item in experiences
        if semantic.action_turn_id in item.source_turn_ids
    )[-6:]
    report = ' '.join(user_turn.content.split())
    goal_result = bool(semantic.developmental_goal) and semantic.kind in {
        'task_success', 'positive_feedback',
    }
    partial_report = goal_result and len(report) > 280
    return Outcome(
        id=_stable_id(
            "outcome", profile_id, semantic.kind, semantic.action_turn_id,
        ),
        profile_id=profile_id,
        result=semantic.kind,
        description=(report if goal_result and not partial_report
                     else compact_text(semantic.evidence, 280)),
        action=compact_text(semantic.action, 280),
        action_turn_id=semantic.action_turn_id,
        confidence=(
            0.9
            if semantic.kind in {"correction", "task_success", "task_failure"}
            else 0.8
        ),
        reason=(f"grounded:user-{semantic.kind.replace('_', '-')}"
                + (':partial-report' if partial_report else '')),
        created_at=now,
        source_turn_ids=(semantic.action_turn_id, user_turn.id),
        experience_ids=linked_experiences,
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
    if current is not None:
        evidence = {
            item.id: item for item in prior_outcomes
            if item.id in {
                *current.supporting_outcome_ids,
                *current.contradiction_outcome_ids,
            }
        }
        if any(
            item.action_turn_id == outcome.action_turn_id
            for item in evidence.values()
        ):
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
    if current.expected_result == actual:
        supports = (*current.supporting_outcome_ids, outcome.id)[-6:]
        strength, confidence = _adjust_strength_confidence(
            current.strength,
            current.confidence,
            TENDENCY_REINFORCE_STRENGTH,
            TENDENCY_REINFORCE_CONFIDENCE,
        )
        item = replace(
            current,
            strength=strength,
            confidence=confidence,
            status=(
                "active"
                if len(supports) > len(current.contradiction_outcome_ids)
                else current.status
            ),
            updated_at=now,
            supporting_outcome_ids=supports,
        )
        return BehavioralTendencyChange("reinforce", item, current.id)
    contradictions = (*current.contradiction_outcome_ids, outcome.id)[-6:]
    if not current.contradiction_outcome_ids:
        strength, confidence = _adjust_strength_confidence(
            current.strength,
            current.confidence,
            -TENDENCY_WEAKEN_STRENGTH,
            -TENDENCY_WEAKEN_CONFIDENCE,
        )
        item = replace(
            current,
            strength=strength,
            confidence=confidence,
            status="uncertain",
            updated_at=now,
            contradiction_outcome_ids=contradictions,
        )
        return BehavioralTendencyChange("weaken", item, current.id)
    item = replace(
        current,
        expected_effect=compact_text(semantic.effect, 160),
        expected_result=actual,
        strength=0.45,
        confidence=0.70,
        status="active",
        updated_at=now,
        supporting_outcome_ids=contradictions,
        contradiction_outcome_ids=current.supporting_outcome_ids[-6:],
        revision_count=current.revision_count + 1,
    )
    return BehavioralTendencyChange("revise", item, current.id)


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
    evidence_by_id = {item.id: item for item in prior_outcomes}

    def repeats_action(item: Strategy) -> bool:
        evidence_ids = {
            *item.supporting_outcome_ids,
            *item.contradiction_outcome_ids,
        }
        return any(
            evidence_by_id[evidence_id].action_turn_id == outcome.action_turn_id
            for evidence_id in evidence_ids
            if evidence_id in evidence_by_id
        )

    if current is not None and repeats_action(current):
        return None
    actual = _outcome_result(outcome.result)
    if current is not None:
        if actual == "success":
            supports = (*current.supporting_outcome_ids, outcome.id)[-6:]
            strength, confidence = _adjust_strength_confidence(
                current.strength,
                current.confidence,
                STRATEGY_REINFORCE_STRENGTH,
                STRATEGY_REINFORCE_CONFIDENCE,
            )
            item = replace(
                current,
                strength=strength,
                confidence=confidence,
                status=(
                    "active"
                    if len(supports) > len(current.contradiction_outcome_ids)
                    else current.status
                ),
                updated_at=now,
                supporting_outcome_ids=supports,
            )
            return StrategyChange("reinforce", item, current.id)
        contradictions = (*current.contradiction_outcome_ids, outcome.id)[-6:]
        strength, confidence = _adjust_strength_confidence(
            current.strength,
            current.confidence,
            -STRATEGY_WEAKEN_STRENGTH,
            -STRATEGY_WEAKEN_CONFIDENCE,
        )
        if not current.contradiction_outcome_ids:
            item = replace(
                current,
                strength=strength,
                confidence=confidence,
                status="uncertain",
                updated_at=now,
                contradiction_outcome_ids=contradictions,
            )
            return StrategyChange("weaken", item, current.id)
        item = replace(
            current,
            strength=strength,
            confidence=confidence,
            status="retired",
            updated_at=now,
            contradiction_outcome_ids=contradictions,
            revision_count=current.revision_count + 1,
        )
        return StrategyChange("retire", item, current.id)
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
        if repeats_action(replaced):
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


def _form_curiosity(
    profile_id: str,
    topic: str,
    focus: str,
    source_ids: tuple[str, ...],
    *,
    now: float,
) -> CuriosityChange | None:
    unique_sources = tuple(dict.fromkeys(source_ids))[-6:]
    if len(unique_sources) < 2:
        return None
    item = Curiosity(
        id=_stable_id("curiosity", profile_id, text_key(topic)),
        profile_id=profile_id,
        topic=compact_text(topic, 120),
        focus=compact_text(focus, 160),
        strength=CURIOSITY_FORM_STRENGTH,
        confidence=CURIOSITY_FORM_CONFIDENCE,
        status="active",
        created_at=now,
        updated_at=now,
        source_ids=unique_sources,
    )
    return CuriosityChange("form", item)


def _reinforce_curiosity(
    current: Curiosity,
    source_ids: tuple[str, ...],
    *,
    now: float,
) -> CuriosityChange | None:
    new_ids = tuple(
        source_id for source_id in source_ids
        if source_id not in current.source_ids
        and source_id not in current.resolution_ids
    )
    if not new_ids:
        return None
    sources = (*current.source_ids, *new_ids)[-6:]
    status = current.status
    action = "reinforce"
    if current.status == "resolved":
        status = (
            "active" if len(sources) > len(current.resolution_ids)
            else "uncertain"
        )
        action = "reactivate"
    elif current.status == "uncertain" and len(sources) > len(current.resolution_ids):
        status = "active"
    strength, confidence = _adjust_strength_confidence(
        current.strength,
        current.confidence,
        CURIOSITY_REINFORCE_STRENGTH,
        CURIOSITY_REINFORCE_CONFIDENCE,
    )
    item = replace(
        current,
        strength=strength,
        confidence=confidence,
        status=status,
        updated_at=now,
        source_ids=sources,
    )
    return CuriosityChange(action, item, current.id)


def _resolve_curiosity(
    current: Curiosity,
    resolution_ids: tuple[str, ...],
    *,
    strong: bool,
    now: float,
) -> CuriosityChange | None:
    new_ids = tuple(
        source_id for source_id in resolution_ids
        if source_id not in current.source_ids
        and source_id not in current.resolution_ids
    )
    if not new_ids:
        return None
    resolutions = (*current.resolution_ids, *new_ids)[-6:]
    resolved = strong or bool(current.resolution_ids)
    strength, confidence = _adjust_strength_confidence(
        current.strength,
        current.confidence,
        -CURIOSITY_WEAKEN_STRENGTH,
        -CURIOSITY_WEAKEN_CONFIDENCE,
    )
    item = replace(
        current,
        strength=strength,
        confidence=confidence,
        status="resolved" if resolved else "uncertain",
        updated_at=now,
        resolution_ids=resolutions,
    )
    return CuriosityChange("resolve" if resolved else "weaken", item, current.id)


def _curiosity_evidence_experience(
    profile_id: str,
    user_turn: Turn,
    assistant_turn: Turn,
    semantic: SemanticEvidence,
    *,
    now: float,
) -> Experience:
    unresolved = semantic.kind == "unresolved_curiosity"
    return _experience(
        profile_id,
        user_turn,
        assistant_turn,
        kind=semantic.kind,
        subject="akane" if unresolved else "shared",
        topic=semantic.topic,
        what_happened=(
            f"Akane expressed unresolved attention about {semantic.topic}."
            if unresolved else
            f"The user supplied grounded resolution about {semantic.topic}."
        ),
        outcome="" if unresolved else semantic.evidence,
        salience=0.65 if unresolved else 0.75,
        reason=(
            "akane:unresolved-curiosity"
            if unresolved else "user:curiosity-resolution"
        ),
        akane_response=semantic.evidence if unresolved else assistant_turn.content,
        now=now,
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


def _form_developmental_goal(
    profile_id: str,
    topic: str,
    goal: str,
    source_ids: tuple[str, ...],
    *,
    now: float,
) -> DevelopmentalGoalChange | None:
    unique_sources = tuple(dict.fromkeys(source_ids))[-6:]
    if not unique_sources:
        return None
    established = len(unique_sources) >= 2
    item = DevelopmentalGoal(
        id=_stable_id(
            "developmental_goal", profile_id, text_key(topic), text_key(goal),
        ),
        profile_id=profile_id,
        topic=compact_text(topic, 120),
        goal=compact_text(goal, 160),
        strength=(
            DEVELOPMENTAL_GOAL_FORM_STRENGTH
            + (DEVELOPMENTAL_GOAL_REINFORCE_STRENGTH if established else 0.0)
        ),
        confidence=(
            DEVELOPMENTAL_GOAL_FORM_CONFIDENCE
            + (DEVELOPMENTAL_GOAL_REINFORCE_CONFIDENCE if established else 0.0)
        ),
        status="active" if established else "candidate",
        created_at=now,
        updated_at=now,
        source_ids=unique_sources,
    )
    return DevelopmentalGoalChange("form", item)


def _reinforce_developmental_goal(
    current: DevelopmentalGoal,
    source_ids: tuple[str, ...],
    *,
    now: float,
) -> DevelopmentalGoalChange | None:
    if current.status in {"satisfied", "retired"}:
        return None
    new_ids = tuple(
        source_id for source_id in source_ids
        if source_id not in current.source_ids
        and source_id not in current.progress_outcome_ids
        and source_id not in current.contradiction_ids
    )
    if not new_ids:
        return None
    sources = (*current.source_ids, *new_ids)[-6:]
    strength, confidence = _adjust_strength_confidence(
        current.strength,
        current.confidence,
        DEVELOPMENTAL_GOAL_REINFORCE_STRENGTH,
        DEVELOPMENTAL_GOAL_REINFORCE_CONFIDENCE,
    )
    item = replace(
        current,
        strength=strength,
        confidence=confidence,
        status="active" if len(sources) >= 2 else current.status,
        updated_at=now,
        source_ids=sources,
    )
    action = "activate" if current.status == "candidate" else "reinforce"
    return DevelopmentalGoalChange(action, item, current.id)


def _progress_developmental_goal(
    current: DevelopmentalGoal,
    outcome_id: str,
    *,
    now: float,
    completed: bool = False,
) -> DevelopmentalGoalChange | None:
    if (
        current.status in {"satisfied", "retired"}
        or outcome_id in current.source_ids
        or outcome_id in current.progress_outcome_ids
        or outcome_id in current.contradiction_ids
    ):
        return None
    progress = (*current.progress_outcome_ids, outcome_id)[-6:]
    strength, confidence = _adjust_strength_confidence(
        current.strength,
        current.confidence,
        DEVELOPMENTAL_GOAL_PROGRESS_STRENGTH,
        DEVELOPMENTAL_GOAL_PROGRESS_CONFIDENCE,
    )
    item = replace(
        current,
        strength=strength,
        confidence=confidence,
        status=(
            "satisfied" if completed else
            "active" if current.source_ids else current.status
        ),
        updated_at=now,
        progress_outcome_ids=progress,
    )
    return DevelopmentalGoalChange(
        "satisfy" if completed else "progress", item, current.id,
    )


def _contradict_developmental_goal(
    current: DevelopmentalGoal,
    evidence_id: str,
    *,
    now: float,
) -> DevelopmentalGoalChange | None:
    if (
        current.status in {"satisfied", "retired"}
        or evidence_id in current.source_ids
        or evidence_id in current.progress_outcome_ids
        or evidence_id in current.contradiction_ids
    ):
        return None
    contradictions = (*current.contradiction_ids, evidence_id)[-6:]
    retired = current.status == "candidate" or len(contradictions) >= 2
    strength, confidence = _adjust_strength_confidence(
        current.strength,
        current.confidence,
        -DEVELOPMENTAL_GOAL_WEAKEN_STRENGTH,
        -DEVELOPMENTAL_GOAL_WEAKEN_CONFIDENCE,
    )
    item = replace(
        current,
        strength=strength,
        confidence=confidence,
        status="retired" if retired else current.status,
        updated_at=now,
        contradiction_ids=contradictions,
    )
    return DevelopmentalGoalChange(
        "retire" if retired else "weaken", item, current.id,
    )


def _developmental_goal_evidence_experience(
    profile_id: str,
    user_turn: Turn,
    assistant_turn: Turn,
    semantic: SemanticEvidence,
    *,
    now: float,
) -> Experience:
    commitment = semantic.kind == "developmental_goal"
    return _experience(
        profile_id,
        user_turn,
        assistant_turn,
        kind=semantic.kind,
        subject="akane",
        topic=semantic.topic,
        what_happened=(
            f"Akane expressed a developmental intention about {semantic.topic}."
            if commitment else
            f"Akane withdrew a developmental intention about {semantic.topic}."
        ),
        outcome="",
        salience=0.75,
        reason=(
            "akane:developmental-goal"
            if commitment else "akane:developmental-goal-release"
        ),
        akane_response=semantic.evidence,
        now=now,
    )


def _expire_predictions(
    predictions: tuple[Prediction, ...], now: float,
) -> tuple[list[PredictionChange], tuple[Prediction, ...]]:
    changes: list[PredictionChange] = []
    active: list[Prediction] = []
    for item in predictions:
        if item.status == "unresolved" and item.expires_at <= now:
            expired = replace(item, status="expired")
            changes.append(PredictionChange("expire", expired, item.id))
            active.append(expired)
        else:
            active.append(item)
    return changes, tuple(active)


def _grounded_prediction(
    profile_id: str,
    assistant_turn: Turn,
    semantic: SemanticEvidence,
    experiences: tuple[Experience, ...],
    *,
    now: float,
) -> Prediction:
    expected = "success" if semantic.kind == "prediction_success" else "failure"
    return Prediction(
        id=_stable_id("prediction", profile_id, assistant_turn.id, expected),
        profile_id=profile_id,
        action=compact_text(semantic.action, 280),
        action_turn_id=assistant_turn.id,
        expectation=compact_text(semantic.evidence, 280),
        expected_result=expected,
        confidence=0.9 if semantic.confidence_band == "high" else 0.4,
        status="unresolved",
        created_at=now,
        expires_at=now + PREDICTION_TTL_SECONDS,
        experience_ids=tuple(
            item.id for item in experiences
            if assistant_turn.id in item.source_turn_ids
        )[-6:],
    )


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


def _self_evidence_experience(
    profile_id: str,
    user_turn: Turn,
    assistant_turn: Turn,
    *,
    semantic: SemanticEvidence,
    self_item_id: str,
    existing: SelfItem | None,
    now: float,
) -> Experience:
    changed = existing is not None and not _same_judgment(existing, semantic)
    return _experience(
        profile_id, user_turn, assistant_turn,
        kind="mind_change" if changed else semantic.kind,
        subject="akane",
        topic=semantic.topic,
        what_happened=(
            f"Akane expressed contrary grounded evidence about {semantic.topic}."
            if changed else
            f"Akane expressed grounded first-person {semantic.kind} evidence "
            f"about {semantic.topic}."
        ),
        outcome="",
        salience=0.85 if changed else 0.65,
        reason=(
            "akane:mind-change-evidence"
            if changed else f"akane:{semantic.kind}-evidence"
        ),
        akane_response=semantic.evidence,
        self_item_ids=(self_item_id,),
        now=now,
    )


def _self_change_from_experience(
    profile_id: str,
    existing: SelfItem | None,
    *,
    semantic: SemanticEvidence,
    self_item_id: str,
    experience_id: str,
    now: float,
) -> SelfChange:
    if existing is None:
        item = form_self_item(
            profile_id, semantic.kind, semantic.topic, semantic.evidence,
            source_id=experience_id,
            item_id=self_item_id,
            stance=semantic.stance,
            semantic_value=semantic.value,
            now=now,
        )
        return SelfChange("form", item)
    if existing.kind == "goal" and semantic.stance == "negative":
        item = abandon_self_goal(
            existing,
            semantic.evidence,
            source_id=experience_id,
            stance=semantic.stance,
            semantic_value=semantic.value,
            now=now,
        )
        return SelfChange("abandon", item, existing.id)
    if _is_refinement(existing, semantic):
        item = refine_self_item(
            existing,
            semantic.evidence,
            source_id=experience_id,
            stance=semantic.stance,
            semantic_value=semantic.value,
            now=now,
        )
        return SelfChange("refine", item, existing.id)
    if _same_judgment(existing, semantic):
        item = reinforce_self_item(
            existing,
            source_id=experience_id,
            value=semantic.evidence,
            stance=semantic.stance,
            semantic_value=semantic.value,
            now=now,
        )
        return SelfChange("reinforce", item, existing.id)
    if (
        self_development_state(existing) == "established"
        and not existing.contradiction_ids
    ):
        item = weaken_self_item(
            existing,
            semantic.evidence,
            source_id=experience_id,
            stance=semantic.stance,
            semantic_value=semantic.value,
            now=now,
        )
        return SelfChange("weaken", item, existing.id)
    item = revise_self_item(
        existing,
        semantic.evidence,
        source_id=experience_id,
        stance=semantic.stance,
        semantic_value=semantic.value,
        now=now,
    )
    return SelfChange("revise", item, existing.id)


def derive_curiosity_changes(
    profile_id: str,
    self_items: tuple[SelfItem, ...],
    memories: tuple[Memory, ...],
    user_turn: Turn,
    assistant_turn: Turn,
    *,
    experiences: tuple[Experience, ...] = (),
    outcomes: tuple[Outcome, ...] = (),
    predictions: tuple[Prediction, ...] = (),
    behavioral_tendencies: tuple[BehavioralTendency, ...] = (),
    strategies: tuple[Strategy, ...] = (),
    curiosities: tuple[Curiosity, ...] = (),
    recent_turns: tuple[Turn, ...] = (),
    semantic_evidence: str = "",
    now: float | None = None,
) -> tuple[
    tuple[MemoryChange, ...],
    tuple[SelfChange, ...],
    tuple[Experience, ...],
    tuple[Outcome, ...],
    tuple[PredictionChange, ...],
    tuple[BehavioralTendencyChange, ...],
    tuple[StrategyChange, ...],
    tuple[CuriosityChange, ...],
]:
    """Derive grounded evidence and one bounded post-turn state proposal."""

    current = time.time() if now is None else float(now)
    memory = _memory_change(
        profile_id, memories, user_turn, assistant_turn, now=current,
    )
    prediction_changes, current_predictions = _expire_predictions(
        predictions, current,
    )
    semantic = (
        validate_semantic_evidence(
            semantic_evidence, user_turn, assistant_turn, recent_turns,
            observed_history=observed_self_history(experiences, outcomes,
                profile_id=profile_id, before=user_turn.created_at),
            historical_expressions=tuple(item.akane_response for item in experiences
                if item.profile_id == profile_id and item.created_at <= user_turn.created_at),
        )
        if user_turn.profile_id == profile_id == assistant_turn.profile_id
        else None
    )
    if semantic is None:
        return (
            (memory,) if memory else (), (), (), (),
            tuple(prediction_changes), (), (), (),
        )
    if semantic.kind in _DEVELOPMENTAL_GOAL_EVIDENCE_KINDS:
        experience = _developmental_goal_evidence_experience(
            profile_id, user_turn, assistant_turn, semantic, now=current,
        )
        if any(same_experience(item, experience) for item in experiences):
            return ((memory,) if memory else (), (), (), (), tuple(prediction_changes), (), (), ())
        return (
            (memory,) if memory else (), (), (experience,), (),
            tuple(prediction_changes), (), (), (),
        )
    if semantic.kind in _CURIOSITY_EVIDENCE_KINDS:
        experience = _curiosity_evidence_experience(
            profile_id, user_turn, assistant_turn, semantic, now=current,
        )
        if any(same_experience(item, experience) for item in experiences):
            return ((memory,) if memory else (), (), (), (), tuple(prediction_changes), (), (), ())
        existing_curiosity = find_curiosity(curiosities, semantic.topic)
        curiosity_change = None
        if semantic.kind == "unresolved_curiosity":
            if existing_curiosity is not None:
                curiosity_change = _reinforce_curiosity(
                    existing_curiosity, (experience.id,), now=current,
                )
            else:
                prior = tuple(
                    item for item in experiences
                    if item.kind == "unresolved_curiosity"
                    and _development_similarity(item.topic, semantic.topic) >= 0.67
                )
                if prior:
                    curiosity_change = _form_curiosity(
                        profile_id,
                        semantic.topic,
                        semantic.curiosity_focus,
                        (prior[-1].id, experience.id),
                        now=current,
                    )
        elif (existing_curiosity is not None
              and _development_similarity(existing_curiosity.focus, semantic.curiosity_focus) >= 0.67):
            curiosity_change = _resolve_curiosity(
                existing_curiosity, (experience.id,), strong=False, now=current,
            )
        return (
            (memory,) if memory else (), (), (experience,), (),
            tuple(prediction_changes), (), (),
            (curiosity_change,) if curiosity_change else (),
        )
    if semantic.kind in _EVENT_EVIDENCE_KINDS:
        if semantic.action_turn_id:
            duplicate = any(
                (item.result == semantic.kind and item.action_turn_id == semantic.action_turn_id)
                or user_turn.id in item.source_turn_ids
                for item in outcomes
            )
            outcome = None if duplicate else _grounded_outcome(
                profile_id, user_turn, semantic, experiences, now=current,
            )
            resolved_prediction = None
            if outcome is not None:
                matches = tuple(
                    item for item in current_predictions
                    if item.status == "unresolved"
                    and item.expires_at > current
                    and item.action_turn_id == outcome.action_turn_id
                    and _normalized_evidence(item.action) == _normalized_evidence(outcome.action)
                )
                if len(matches) == 1:
                    resolved = _resolve_prediction(matches[0], outcome, now=current)
                    resolved_prediction = resolved
                    prediction_changes.append(PredictionChange(
                        "resolve", resolved, matches[0].id,
                    ))
            tendency_change = (
                _tendency_change_from_outcome(
                    profile_id, behavioral_tendencies, outcomes,
                    outcome, semantic, now=current,
                )
                if outcome is not None else None
            )
            strategy_change = (
                _strategy_change_from_outcome(
                    profile_id, strategies, outcomes,
                    outcome, semantic, now=current,
                )
                if outcome is not None else None
            )
            curiosity_change = None
            if outcome is not None:
                existing_curiosity = find_curiosity(curiosities, semantic.topic)
                resolution_ids: tuple[str, ...] = ()
                support_ids: tuple[str, ...] = ()
                focus = compact_text(
                    f"Understand unresolved results in {semantic.topic}", 160,
                )
                if strategy_change is not None and strategy_change.action == "revise":
                    resolution_ids = (outcome.id,)
                elif (
                    resolved_prediction is not None
                    and resolved_prediction.error_category != "none"
                    and abs(resolved_prediction.error_value) >= 0.75
                ):
                    support_ids = (resolved_prediction.id, outcome.id)
                elif strategy_change is not None and strategy_change.action == "retire":
                    support_ids = strategy_change.strategy.contradiction_outcome_ids[-2:]
                    focus = compact_text(
                        f"Understand why {strategy_change.strategy.context} remains unresolved",
                        160,
                    )
                elif tendency_change is not None and tendency_change.action == "revise":
                    support_ids = (
                        tendency_change.tendency.contradiction_outcome_ids[-2:]
                    )
                if resolution_ids and existing_curiosity is not None:
                    curiosity_change = _resolve_curiosity(
                        existing_curiosity,
                        resolution_ids,
                        strong=True,
                        now=current,
                    )
                elif support_ids:
                    curiosity_change = (
                        _reinforce_curiosity(
                            existing_curiosity, support_ids, now=current,
                        )
                        if existing_curiosity is not None else
                        _form_curiosity(
                            profile_id,
                            semantic.topic,
                            focus,
                            support_ids,
                            now=current,
                        )
                    )
            return (
                (memory,) if memory else (),
                (),
                (),
                (outcome,) if outcome else (),
                tuple(prediction_changes),
                (tendency_change,) if tendency_change else (),
                (strategy_change,) if strategy_change else (),
                (curiosity_change,) if curiosity_change else (),
            )
        event = _event_evidence_experience(
            profile_id, user_turn, assistant_turn, semantic, now=current,
        )
        if experiences and same_experience(experiences[-1], event):
            event = None
        return (
            (memory,) if memory else (),
            (),
            (event,) if event else (),
            (),
            tuple(prediction_changes),
            (),
            (),
            (),
        )
    if semantic.kind in _PREDICTION_EVIDENCE_KINDS:
        if not any(
            item.action_turn_id == assistant_turn.id for item in current_predictions
        ):
            unresolved = tuple(
                item for item in current_predictions if item.status == "unresolved"
            )
            if len(unresolved) >= UNRESOLVED_PREDICTION_LIMIT:
                oldest = min(unresolved, key=lambda item: (item.created_at, item.id))
                prediction_changes.append(PredictionChange(
                    "expire", replace(oldest, status="expired"), oldest.id,
                ))
            prediction = _grounded_prediction(
                profile_id, assistant_turn, semantic, experiences, now=current,
            )
            prediction_changes.append(PredictionChange("form", prediction))
        return (
            (memory,) if memory else (), (), (), (),
            tuple(prediction_changes), (), (), (),
        )
    existing = find_self_item(self_items, semantic.kind, semantic.topic)
    if existing is None and semantic.kind == "goal" and semantic.stance == "negative":
        return (
            (memory,) if memory else (), (), (), (),
            tuple(prediction_changes), (), (), (),
        )
    self_item_id = existing.id if existing is not None else f"self_{uuid.uuid4().hex}"
    experience = _self_evidence_experience(
        profile_id, user_turn, assistant_turn,
        semantic=semantic,
        self_item_id=self_item_id,
        existing=existing, now=current,
    )
    # Identical restatements need new relevant executions to count as support.
    # Attach that execution's real source pair without inferring a preference
    # from task success; an independently emitted own judgment is still required.
    topic_terms = self_topic_terms(semantic.topic)
    prior_judgments = tuple(item for item in experiences
        if item.subject == 'akane' and (item.kind == semantic.kind or (
            item.kind == 'mind_change' and existing is not None and existing.id in item.self_item_ids))
        and _same_self_topic(semantic.kind, item.topic, semantic.topic))
    observed = next((item for item in reversed(outcomes)
        if item.reason == 'grounded:virtual_environment' and item.created_at <= user_turn.created_at
        and topic_terms and topic_terms <= self_topic_terms(item.action)), None)
    new_observation = observed is not None and not any(
        set(observed.source_turn_ids) <= set(item.source_turn_ids) for item in prior_judgments)
    if (not new_observation and any(_normalized_evidence(item.akane_response)
            == _normalized_evidence(semantic.evidence) for item in prior_judgments)):
        return ((memory,) if memory else (), (), (), (), tuple(prediction_changes), (), (), ())
    if new_observation:
        experience = replace(experience, source_turn_ids=_bounded_ids(
            experience.source_turn_ids, observed.source_turn_ids))
    if any(same_experience(previous, experience) for previous in experiences):
        return (
            (memory,) if memory else (), (), (), (),
            tuple(prediction_changes), (), (), (),
        )
    self_change = _self_change_from_experience(
        profile_id, existing,
        semantic=semantic,
        self_item_id=self_item_id,
        experience_id=experience.id, now=current,
    )
    return (
        (memory,) if memory else (),
        (self_change,),
        (experience,),
        (),
        tuple(prediction_changes),
        (),
        (),
        (),
    )


def derive_developmental_goal_changes(
    profile_id: str,
    self_items: tuple[SelfItem, ...],
    memories: tuple[Memory, ...],
    user_turn: Turn,
    assistant_turn: Turn,
    *,
    experiences: tuple[Experience, ...] = (),
    outcomes: tuple[Outcome, ...] = (),
    predictions: tuple[Prediction, ...] = (),
    behavioral_tendencies: tuple[BehavioralTendency, ...] = (),
    strategies: tuple[Strategy, ...] = (),
    curiosities: tuple[Curiosity, ...] = (),
    developmental_goals: tuple[DevelopmentalGoal, ...] = (),
    recent_turns: tuple[Turn, ...] = (),
    semantic_evidence: str = "",
    now: float | None = None,
) -> tuple[
    tuple[MemoryChange, ...],
    tuple[SelfChange, ...],
    tuple[Experience, ...],
    tuple[Outcome, ...],
    tuple[PredictionChange, ...],
    tuple[BehavioralTendencyChange, ...],
    tuple[StrategyChange, ...],
    tuple[CuriosityChange, ...],
    tuple[DevelopmentalGoalChange, ...],
]:
    """Add grounded developmental Goal changes to the post-turn proposal."""

    current = time.time() if now is None else float(now)
    base = derive_curiosity_changes(
        profile_id,
        self_items,
        memories,
        user_turn,
        assistant_turn,
        experiences=experiences,
        outcomes=outcomes,
        predictions=predictions,
        behavioral_tendencies=behavioral_tendencies,
        strategies=strategies,
        curiosities=curiosities,
        recent_turns=recent_turns,
        semantic_evidence=semantic_evidence,
        now=current,
    )
    semantic = (
        validate_semantic_evidence(
            semantic_evidence, user_turn, assistant_turn, recent_turns,
        )
        if user_turn.profile_id == profile_id == assistant_turn.profile_id
        else None
    )
    if semantic is None:
        return (*base, ())
    goal_change = None
    if semantic.kind in _DEVELOPMENTAL_GOAL_EVIDENCE_KINDS and base[2]:
        experience = base[2][0]
        current_goal = find_developmental_goal(
            developmental_goals, semantic.topic, semantic.developmental_goal,
        )
        if semantic.kind == "developmental_goal":
            if current_goal is not None:
                goal_change = _reinforce_developmental_goal(
                    current_goal, (experience.id,), now=current,
                )
            else:
                source_ids = [experience.id]
                curiosity = find_curiosity(curiosities, semantic.topic)
                if (
                    curiosity is not None
                    and curiosity.status == "active"
                    and len(curiosity.source_ids) >= 3
                    and curiosity.strength >= 0.45
                    and curiosity.confidence >= 0.70
                ):
                    source_ids.append(curiosity.id)
                related_interests = tuple(
                    item for item in self_items
                    if item.kind == "interest"
                    and self_development_state(item) == "established"
                    and self_topic_similarity(item.topic, semantic.topic) >= 0.67
                )
                if len(related_interests) == 1:
                    source_ids.append(related_interests[0].id)
                goal_change = _form_developmental_goal(
                    profile_id,
                    semantic.topic,
                    semantic.developmental_goal,
                    tuple(source_ids),
                    now=current,
                )
        elif current_goal is not None:
            goal_change = _contradict_developmental_goal(
                current_goal, experience.id, now=current,
            )
    elif (
        semantic.kind in _EVENT_EVIDENCE_KINDS
        and semantic.developmental_goal
        and base[3]
    ):
        outcome = base[3][0]
        current_goal = find_developmental_goal(
            developmental_goals, semantic.topic, semantic.developmental_goal,
        )
        if (
            current_goal is not None
            and _grounded_outcome_relevant_to_goal(current_goal, outcome)
            and not any(
                item.action_turn_id == outcome.action_turn_id
                and item.id in (*current_goal.progress_outcome_ids, *current_goal.contradiction_ids)
                for item in outcomes
            )
        ):
            if _outcome_result(outcome.result) == "success":
                goal_change = _progress_developmental_goal(
                    current_goal, outcome.id, now=current,
                    completed=developmental_goal_completed_by_outcome(
                        current_goal, outcome, feedback_text=user_turn.content,
                    ),
                )
            else:
                goal_change = _contradict_developmental_goal(
                    current_goal, outcome.id, now=current,
                )
    return (*base, (goal_change,) if goal_change else ())


def derive_procedural_changes(
    profile_id: str,
    self_items: tuple[SelfItem, ...],
    memories: tuple[Memory, ...],
    user_turn: Turn,
    assistant_turn: Turn,
    *,
    experiences: tuple[Experience, ...] = (),
    outcomes: tuple[Outcome, ...] = (),
    predictions: tuple[Prediction, ...] = (),
    behavioral_tendencies: tuple[BehavioralTendency, ...] = (),
    strategies: tuple[Strategy, ...] = (),
    recent_turns: tuple[Turn, ...] = (),
    semantic_evidence: str = "",
    now: float | None = None,
) -> tuple[
    tuple[MemoryChange, ...],
    tuple[SelfChange, ...],
    tuple[Experience, ...],
    tuple[Outcome, ...],
    tuple[PredictionChange, ...],
    tuple[BehavioralTendencyChange, ...],
    tuple[StrategyChange, ...],
]:
    """Compatibility wrapper for the Phase 2.4 procedural interface."""

    memory, self_changes, experience, outcome, prediction, tendency, strategy, _ = (
        derive_curiosity_changes(
            profile_id,
            self_items,
            memories,
            user_turn,
            assistant_turn,
            experiences=experiences,
            outcomes=outcomes,
            predictions=predictions,
            behavioral_tendencies=behavioral_tendencies,
            strategies=strategies,
            recent_turns=recent_turns,
            semantic_evidence=semantic_evidence,
            now=now,
        )
    )
    return (
        memory, self_changes, experience, outcome, prediction, tendency, strategy,
    )


def derive_learning_changes(
    profile_id: str,
    self_items: tuple[SelfItem, ...],
    memories: tuple[Memory, ...],
    user_turn: Turn,
    assistant_turn: Turn,
    *,
    experiences: tuple[Experience, ...] = (),
    outcomes: tuple[Outcome, ...] = (),
    predictions: tuple[Prediction, ...] = (),
    behavioral_tendencies: tuple[BehavioralTendency, ...] = (),
    recent_turns: tuple[Turn, ...] = (),
    semantic_evidence: str = "",
    now: float | None = None,
) -> tuple[
    tuple[MemoryChange, ...],
    tuple[SelfChange, ...],
    tuple[Experience, ...],
    tuple[Outcome, ...],
    tuple[PredictionChange, ...],
    tuple[BehavioralTendencyChange, ...],
]:
    """Compatibility wrapper for the Phase 2.3 learning interface."""

    memory, self_changes, experience, outcome, prediction, tendency, _ = (
        derive_procedural_changes(
            profile_id,
            self_items,
            memories,
            user_turn,
            assistant_turn,
            experiences=experiences,
            outcomes=outcomes,
            predictions=predictions,
            behavioral_tendencies=behavioral_tendencies,
            recent_turns=recent_turns,
            semantic_evidence=semantic_evidence,
            now=now,
        )
    )
    return memory, self_changes, experience, outcome, prediction, tendency


def derive_state_changes(
    profile_id: str,
    self_items: tuple[SelfItem, ...],
    memories: tuple[Memory, ...],
    user_turn: Turn,
    assistant_turn: Turn,
    *,
    experiences: tuple[Experience, ...] = (),
    outcomes: tuple[Outcome, ...] = (),
    predictions: tuple[Prediction, ...] = (),
    recent_turns: tuple[Turn, ...] = (),
    semantic_evidence: str = "",
    now: float | None = None,
) -> tuple[
    tuple[MemoryChange, ...],
    tuple[SelfChange, ...],
    tuple[Experience, ...],
    tuple[Outcome, ...],
    tuple[PredictionChange, ...],
]:
    """Compatibility wrapper for the Phase 2.2 state-change interface."""

    memory, self_changes, experience, outcome, prediction, _ = (
        derive_learning_changes(
            profile_id,
            self_items,
            memories,
            user_turn,
            assistant_turn,
            experiences=experiences,
            outcomes=outcomes,
            predictions=predictions,
            recent_turns=recent_turns,
            semantic_evidence=semantic_evidence,
            now=now,
        )
    )
    return memory, self_changes, experience, outcome, prediction


def derive_turn_changes(
    profile_id: str,
    self_items: tuple[SelfItem, ...],
    memories: tuple[Memory, ...],
    user_turn: Turn,
    assistant_turn: Turn,
    *,
    experiences: tuple[Experience, ...] = (),
    outcomes: tuple[Outcome, ...] = (),
    recent_turns: tuple[Turn, ...] = (),
    semantic_evidence: str = "",
    now: float | None = None,
) -> tuple[
    tuple[MemoryChange, ...],
    tuple[SelfChange, ...],
    tuple[Experience, ...],
    tuple[Outcome, ...],
]:
    """Compatibility wrapper for the Phase 2.1 turn-change interface."""

    memory, self_changes, experience, outcome, _ = derive_state_changes(
        profile_id,
        self_items,
        memories,
        user_turn,
        assistant_turn,
        experiences=experiences,
        outcomes=outcomes,
        recent_turns=recent_turns,
        semantic_evidence=semantic_evidence,
        now=now,
    )
    return memory, self_changes, experience, outcome


def derive_durable_changes(
    profile_id: str,
    self_items: tuple[SelfItem, ...],
    memories: tuple[Memory, ...],
    user_turn: Turn,
    assistant_turn: Turn,
    *,
    experiences: tuple[Experience, ...] = (),
    semantic_evidence: str = "",
    now: float | None = None,
) -> tuple[tuple[MemoryChange, ...], tuple[SelfChange, ...], tuple[Experience, ...]]:
    """Compatibility wrapper for the Phase 1 durable-change interface."""

    memory, self_changes, experience, _ = derive_turn_changes(
        profile_id,
        self_items,
        memories,
        user_turn,
        assistant_turn,
        experiences=experiences,
        semantic_evidence=semantic_evidence,
        now=now,
    )
    return memory, self_changes, experience
