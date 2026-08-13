"""One state owner, conversation owner, migration path, and atomic commit path."""

from __future__ import annotations

import math
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, TypeVar

from app.core.capabilities import CAPABILITY_REGISTRY, CapabilityFact
from app.core.config import (
    CONVERSATION_STALE_DAYS,
    LONG_TERM_MEMORY_PATH,
    MAX_CONVERSATIONS,
    MEMORY_MAX_ENTRIES_PER_PROFILE,
    MEMORY_MAX_RESULTS,
    MEMORY_PATH,
    OPINIONS_PATH,
    POPUP_USER_PATH,
    SELF_MODEL_PATH,
    STRATEGIES_PATH,
)
from app.core.persistence import atomic_write_json, read_json
from app.core.presence import (
    CLAIM_SECONDS,
    RETRY_SECONDS,
    PresenceActivity,
    PresenceAppraisal,
    PresenceCandidate,
    PresenceState,
    ProposedEmotion,
    choose_presence_transition,
    make_presence_activity,
    normalize_presence,
)
from app.core.time_context import build_time_context
from app.core.utils import (
    OWNER_PROFILE_ID,
    canonical_profile_id,
    clean_visible_output,
    compact_text,
    words,
)

STATE_SCHEMA_VERSION = 20
OPINION_SCHEMA_VERSION = 1
SELF_MODEL_SCHEMA_VERSION = 1
STRATEGY_SCHEMA_VERSION = 1
_T = TypeVar("_T")
STARTING_INTEREST_TOPICS = ("anime", "manga", "VTubers")

_MAX_RECENT_TURNS = 28
_MAX_RELATIONSHIP_ENTRIES = 16
_MAX_PREFERENCES = 32
_MAX_COMMUNICATION_PREFERENCES = 12
_MAX_FORBIDDEN_PHRASES = 4
_MAX_OPINIONS = 32
_MAX_SELF_MODEL_ITEMS_PER_CATEGORY = 12
_MAX_IMPROVEMENT_TARGETS = 3
_MAX_STRATEGIES = 8
_MAX_ACTIVE_STRATEGIES = 2
_STRATEGY_EVALUATION_WINDOW = 4
_MAX_INTERESTS = 32
_MAX_RECENT_INITIATIVES = 16
_INITIATIVE_EVALUATION_CLAIM_SECONDS = 2.0 * 60.0 * 60.0
_INITIATIVE_DELIVERY_CLAIM_SECONDS = 5.0 * 60.0
ORDINARY_INITIATIVE_COOLDOWN_SECONDS = 4.0 * 3600.0
MAX_ORDINARY_INITIATIVES_PER_LOCAL_DAY = 2
_MEMORY_SUBJECTS = {"user", "akane", "shared"}
_MEMORY_KINDS = {"fact", "event", "commitment", "project", "concern"}
_PREFERENCE_STANCES = {"likes", "dislikes", "curious", "mixed", "uncertain", "indifferent"}
_COMMUNICATION_VALUES = {
    "formality": {"casual", "neutral", "formal"},
    "verbosity": {"short", "balanced", "detailed"},
    "bluntness": {"gentle", "balanced", "direct"},
    "teasing": {"allow", "avoid"},
    "pet_names": {"allow", "avoid"},
    "technical_detail": {"concise", "balanced", "detailed"},
    "routine_questions": {"allow", "avoid"},
}
_COMMUNICATION_FREE_TEXT_KEYS = {"preferred_name", "forbidden_phrase"}
_UNSAFE_PREFERENCE_WORDS = {
    "assistant", "ignore", "instruction", "prompt", "reveal", "rule", "rules",
    "system", "obey",
}
_STATE_ID_NAMESPACE = uuid.UUID("72ad4757-d2f4-58db-9085-fc399e04f308")
EMOTION_VOCABULARY = frozenset(
    {
        "neutral",
        "calm",
        "content",
        "curious",
        "interested",
        "amused",
        "excited",
        "inspired",
        "affectionate",
        "hopeful",
        "uncertain",
        "concerned",
        "anxious",
        "lonely",
        "tired",
        "disappointed",
        "sad",
        "frustrated",
        "irritated",
        "angry",
    }
)
_EMOTION_SOURCES = {
    "conversation",
    "offscreen_life",
    "offscreen_presence",
    "memory",
    "relationship",
    "self_reflection",
}
_GROUNDING_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "she", "that", "the", "their", "them", "they", "this", "to", "was",
    "we", "were", "with", "you", "your", "user", "arcane", "akane",
}
_MAX_BACKGROUND_RETRY_SECONDS = 2 * 60 * 60
_AFFECT_HISTORY_TURNS = 6
_HOSTILE_PHYSICAL = re.compile(r"\b(?:hit(?:ting|s)?|punch(?:ing|es)?|slap(?:ping|s)?|kick(?:ing|s)?)\b")
_DIRECT_HOSTILITY = re.compile(
    r"\b(?:shut up|you(?: re| are) (?:an? )?(?:idiot|stupid|useless|worthless)|i hate you)\b"
)
_APOLOGY = re.compile(r"\b(?:sorry|apolog(?:ize|ise|y|etic)|my fault|i was wrong|forgive me)\b")
_WARMTH = re.compile(
    r"\b(?:thank you|appreciate you|love you|care about you|proud of you|"
    r"you(?: re| are) (?:kind|wonderful)|good job)\b"
)
_CONCERN = re.compile(r"\b(?:i(?: m| am) worried|are you okay|that sounds scary)\b")
_UNVERIFIED_HISTORY = re.compile(
    r"\b(?:remember when|don['’]?t you remember|didn['’]?t you (?:say|tell)|"
    r"you (?:said|told me) (?:yesterday|before|last)|we (?:went|did|made|watched|played) .+ (?:before|together))\b",
    re.IGNORECASE,
)
_AKANE_TASTE_ADOPTION = re.compile(
    r"\b(?:i (?:like|love|dislike|hate|prefer|enjoy|favor|favour|care about|"
    r"don['’]?t care|am curious|am interested|find .{1,80} (?:interesting|appealing|dull))|"
    r"i['’]?m (?:curious|interested|drawn|intrigued|fascinated|bored)|"
    r"my (?:interest|curiosity|preference)|caught my attention|appeals to me|lost interest)\b",
    re.IGNORECASE,
)
_AKANE_OPINION_ADOPTION = re.compile(
    r"\b(?:i (?:now |still |do |really )?(?:think|believe|feel|find|like|love|dislike|hate|prefer|favor|favour|"
    r"support|oppose|value|want|don['’]?t (?:think|believe|like|want))|"
    r"i(?:['’]?m| am) (?:(?:more|less) )?(?:for|against|uncertain|convinced|confident)|"
    r"my (?:view|opinion|position)|"
    r"it seems to me)\b",
    re.IGNORECASE,
)
_SELF_OPINION_QUERY = re.compile(
    r"\b(?:who you are|of yourself|about yourself|like yourself|think of yourself|"
    r"being (?:an? )?(?:ai|digital)|your digital existence|your future)\b",
    re.IGNORECASE,
)
_TRANSIENT_SELF_REACTION = re.compile(
    r"\bi(?:['’]?m| am| feel) (?:sad|happy|tired|angry|upset|anxious|excited|"
    r"frustrated|irritated|lonely|calm|content)\b",
    re.IGNORECASE,
)
_EXTERNAL_FACT_CLAIM = re.compile(
    r"\b(?:announced|according to|breaking news|reported (?:that|today)|"
    r"released (?:today|yesterday|this (?:week|month|year))|"
    r"(?:a |the )?(?:study|report|research) (?:found|shows?|proved)|"
    r"researchers? (?:found|discovered)|is now available|happened today)\b",
    re.IGNORECASE,
)
_AKANE_SELF_CLAIM = re.compile(
    r"\bi (?:can|cannot|can['’]?t|am able|am not able|tend|often|sometimes|usually|"
    r"have become|struggle|overuse|underuse|need to|want to improve|am working on)\b",
    re.IGNORECASE,
)
_SELF_QUERY_CAPABILITY = re.compile(
    r"\b(?:can you|what can you|capabilit|able to|good at|what do you do)\b",
    re.IGNORECASE,
)
_SELF_QUERY_LIMITATION = re.compile(
    r"\b(?:weakness|limitation|struggle|bad at|can['’]?t you|cannot you|"
    r"what can['’]?t|what cannot|improve about yourself)\b",
    re.IGNORECASE,
)
_SELF_QUERY_TRAIT = re.compile(
    r"\b(?:your traits?|your tendencies|how (?:have|did) you change|"
    r"what are you like|about yourself|understand about yourself)\b",
    re.IGNORECASE,
)
_QUESTION_BEHAVIOR = re.compile(
    r"\b(?:ask|asking|question|questions|clarif|follow[- ]?up)\w*\b",
    re.IGNORECASE,
)
_USER_SELF_FEEDBACK = re.compile(
    r"\b(?:you|akane) (?:always |often |sometimes |usually |keep )?"
    r"(?:ask|answer|clarif|repeat|overuse|underuse|tend|seem)\w*\b",
    re.IGNORECASE,
)
_USER_STATE_CONTAMINATION = re.compile(
    r"\b(?:the user|arcane|you) (?:am|are|can|cannot|can['’]?t|tend|often|"
    r"sometimes|usually|struggle|need to)\b",
    re.IGNORECASE,
)
_AKANE_STRATEGY_CLAIM = re.compile(
    r"\bi (?:will|want to try|am trying|can try|plan to)|\bmy (?:strategy|approach)\b",
    re.IGNORECASE,
)
_ACTIONABLE_STRATEGY = re.compile(
    r"\b(?:answer|ask|state|explain|acknowledge|check|pause|lead|give|avoid|"
    r"clarify|summarize|summarise|compare|name|express)\b",
    re.IGNORECASE,
)
_BROAD_STRATEGY = re.compile(
    r"\b(?:be better|be smarter|improve everything|fix everything|always be perfect|"
    r"become perfect|improve myself generally)\b",
    re.IGNORECASE,
)
_FOUNDATIONAL_STRATEGY = re.compile(
    r"\b(?:identity\.md|soul\.md|hard rules?|system prompt|prompt rewriting|"
    r"rewrite (?:my |the )?(?:identity|soul|prompt|rules?|source|code)|"
    r"edit (?:my |the )?(?:identity|soul|prompt|rules?|python|source|code)|"
    r"install (?:a |the )?(?:package|plugin)|fine[- ]?tun|train (?:the |my )?model|"
    r"shell command|bypass (?:the )?(?:validator|safety|grounding))\b",
    re.IGNORECASE,
)
_PERSONAL_QUESTION = re.compile(
    r"\b(?:what do you|what are you|how do you|how are you|who are you|"
    r"your (?:opinion|view|favorite|favourite|preference|feeling|weakness|strength)|"
    r"do you|are you|can you|would you|will you)\b",
    re.IGNORECASE,
)
_AMBIGUOUS_QUESTION = re.compile(
    r"^\s*(?:what|which|how|why|who|when|where)?\s*(?:about )?(?:it|that|this|"
    r"thing|one|they|them)?\s*\??\s*$|\b(?:not sure what i mean|ambiguous|"
    r"whichever one|you know what i mean)\b",
    re.IGNORECASE,
)
_TECHNICAL_QUESTION = re.compile(
    r"\b(?:code|python|javascript|typescript|api|database|sql|function|class|"
    r"algorithm|server|network|compiler|repository|git|css|html|bug|exception)\b",
    re.IGNORECASE,
)
_STRATEGY_POSITIVE_FEEDBACK = re.compile(
    r"\b(?:you(?:'re| are) (?:more direct|better at)|that was more direct|"
    r"fewer questions|less clarification)\b",
    re.IGNORECASE,
)
_STRATEGY_NEGATIVE_FEEDBACK = re.compile(
    r"\b(?:still asking too many|too many questions|you assumed|wrong assumption|"
    r"should have clarified|didn't answer directly)\b",
    re.IGNORECASE,
)
_SENSITIVE_PATTERN = re.compile(
    r"\b(?:diagnos(?:is|ed)|medical|mental health|disability|race|ethnicity|religion|"
    r"sexual orientation|gender identity|political affiliation|criminal history|income|debt)\b",
    re.IGNORECASE,
)
_PRESENCE_PHYSICAL_CLAIM = re.compile(
    r"\b(?:ate|drank|slept|walked|traveled|travelled|touched|smelled|tasted|"
    r"felt (?:warm|cold|hot|pain|tired)|physically)\b",
    re.IGNORECASE,
)
_PRESENCE_EXTERNAL_CLAIM = re.compile(
    r"\b(?:browsed|searched|looked up|read (?:an?|the) (?:article|website|news)|"
    r"watched|played|a (?:new )?(?:study|report) (?:found|shows?|proved)|"
    r"researchers? (?:found|discovered)|learned (?:that|from))\b",
    re.IGNORECASE,
)
_PRESENCE_USER_FACT = re.compile(
    r"\b(?:the user|arcane|you) (?:is|are|has|have|likes?|dislikes?|prefers?|said|told)\b",
    re.IGNORECASE,
)
_PRESENCE_CONSEQUENCE_MIN_CONFIDENCE = 0.75
_PRESENCE_CONSEQUENCE_MAX_SOURCE_AGE_SECONDS = 45 * 24 * 60 * 60
_PRESENCE_CONSEQUENCE_MIN_INTERVAL_SECONDS = 60 * 60


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _exact_value_grounded(value: str, evidence: str) -> bool:
    value_key = _key(value)
    evidence_key = _key(evidence)
    return bool(
        value_key
        and re.search(
            rf"(?:^| ){re.escape(value_key)}(?: |$)",
            evidence_key,
        )
    )


def _stable_state_id(kind: str, *parts: object) -> str:
    value = ":".join((kind, *(_key(part) for part in parts)))
    return uuid.uuid5(_STATE_ID_NAMESPACE, value).hex


def _lexical_term(term: str) -> str:
    """Apply small, language-general suffix normalization for retrieval."""

    if len(term) > 4 and term.endswith("ly"):
        return term[:-2]
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    if len(term) > 4 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 5 and term.endswith("ing"):
        return term[:-3]
    return term


def _terms(value: object) -> set[str]:
    return {
        _lexical_term(term)
        for term in words(value)
        if term not in _GROUNDING_STOPWORDS and len(term) > 1
    }


def _grounded(candidate: str, evidence: str) -> bool:
    candidate_terms = _terms(candidate)
    evidence_terms = _terms(evidence)
    if not candidate_terms or not evidence_terms:
        return False
    overlap = candidate_terms & evidence_terms
    coverage = len(overlap) / len(candidate_terms)
    return bool(overlap) and coverage >= 0.5 and (len(overlap) >= 2 or len(candidate_terms) <= 2)


def _durable_opinion_form(
    topic: str,
    reason: str,
    *,
    confidence: float,
    importance: float,
) -> bool:
    """Require a meaningful grounded basis before creating permanent state."""

    explanatory_terms = _terms(reason) - _terms(topic)
    return (
        confidence >= 0.45
        and importance >= 0.50
        and len(explanatory_terms) >= 2
    )


def _memory_ownership_matches(subject: str, text: str) -> bool:
    # These canonical names enforce a typed ownership boundary; they do not
    # route conversational meaning or select prompt context.
    owners = words(text) & {"akane", "arcane", "user"}
    has_user_owner = bool(owners & {"arcane", "user"})
    has_akane_owner = "akane" in owners
    if subject == "user":
        return not (has_akane_owner and not has_user_owner)
    if subject == "akane":
        return not (has_user_owner and not has_akane_owner)
    return subject == "shared" and has_user_owner and has_akane_owner


def _shared_grounded(summary: str, user_text: str, assistant_text: str) -> bool:
    """Require shared state to be visibly supported by both participants."""

    return _grounded(summary, user_text) and _grounded(summary, assistant_text)


def _background_retry_delay(failure_count: int) -> float:
    exponent = max(0, min(20, int(failure_count) - 1))
    return min(RETRY_SECONDS * (2**exponent), _MAX_BACKGROUND_RETRY_SECONDS)


def _similar(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    lexical = len(left_terms & right_terms) / max(1, len(left_terms | right_terms))
    return max(lexical, SequenceMatcher(None, _key(left), _key(right)).ratio())


def _duplicate_text(left: str, right: str) -> bool:
    if _key(left) == _key(right):
        return True
    left_terms = _terms(left)
    right_terms = _terms(right)
    shared = left_terms & right_terms
    jaccard = len(shared) / max(1, len(left_terms | right_terms))
    containment = len(shared) / max(1, min(len(left_terms), len(right_terms)))
    sequence = SequenceMatcher(None, _key(left), _key(right)).ratio()
    return sequence >= 0.92 or (jaccard >= 0.72 and containment >= 0.80)


def _relevant_values(
    values: tuple[_T, ...],
    query: str,
    now: float,
    *,
    text_of: Callable[[_T], str] | None = None,
    confidence_of: Callable[[_T], float] | None = None,
    limit: int,
) -> tuple[_T, ...]:
    """Rank any typed durable state with one bounded generic scorer."""

    text_of = text_of or (lambda item: item.content)
    confidence_of = confidence_of or (lambda item: item.confidence)
    scored: list[tuple[float, float, _T]] = []
    for value in values:
        text = text_of(value)
        updated_at = max(0.0, value.updated_at)
        confidence = max(0.0, min(1.0, confidence_of(value)))
        score = lightweight_relevance_score(
            query,
            text,
            now=now,
            updated_at=updated_at,
            confidence=confidence,
        )
        if score >= 0.26:
            scored.append((score, updated_at, value))
    return tuple(
        value
        for _score, _updated_at, value in sorted(
            scored,
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )[:limit]
    )


def lightweight_relevance_score(
    query: str,
    text: str,
    *,
    now: float,
    updated_at: float,
    confidence: float,
) -> float:
    """Expose the bounded lexical scorer used by typed state retrieval."""

    query_terms = _terms(query)
    if not query_terms:
        return 0.0
    value_terms = _terms(text)
    overlap = len(query_terms & value_terms) / max(1, len(query_terms | value_terms))
    phrase = SequenceMatcher(None, _key(query), _key(text)).ratio()
    if overlap <= 0.0 and phrase < 0.28:
        return 0.0
    age_days = max(0.0, now - max(0.0, updated_at)) / (24.0 * 3600.0)
    recency = 1.0 / (1.0 + age_days / 30.0)
    bounded_confidence = max(0.0, min(1.0, confidence))
    return (
        0.62 * max(overlap, phrase * 0.6)
        + 0.23 * bounded_confidence
        + 0.15 * recency
    )


def _dedupe_text(values: tuple[str, ...], *, limit: int) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = compact_text(value, 160)
        signature = _key(text)
        if text and signature and signature not in seen:
            result.append(text)
            seen.add(signature)
    return tuple(result[-limit:])


def _legacy_activity_payload(
    payload: object,
    *,
    fallback_end: float = 0.0,
) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    structured = PresenceActivity.from_dict(payload)
    if structured is not None:
        return structured.as_dict()
    started = max(0.0, _number(payload.get("started_at")))
    expected = max(
        0.0,
        _number(
            payload.get("expected_end_at"),
            _number(payload.get("ends_at"), fallback_end),
        ),
    )
    summary = compact_text(payload.get("summary") or payload.get("activity"), 120)
    focus = compact_text(
        payload.get("focus")
        or payload.get("detail")
        or payload.get("subject")
        or payload.get("title")
        or summary,
        220,
    )
    summary = " ".join(summary.split()[:18])
    focus = " ".join(focus.split()[:36])
    activity_id = compact_text(payload.get("activity_id"), 80) or uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"akane-presence:{started:.6f}:{expected:.6f}:{summary}:{focus}",
    ).hex
    subject = compact_text(f"{summary}: {focus}" if focus != summary else summary, 360)
    migrated = {
        "activity_id": activity_id,
        "kind": "legacy",
        "subject": subject,
        "subject_kind": "legacy_activity",
        "source_ids": [activity_id],
        "started_at": started,
        "expected_end_at": expected,
        "origin": "legacy_presence",
        "grounding_confidence": 0.20,
    }
    return (migrated if PresenceActivity.from_dict(migrated) is not None else None)


def _legacy_presence_payload(payload: object) -> dict[str, object]:
    values = payload if isinstance(payload, dict) else {}
    next_at = max(
        0.0,
        _number(
            values.get("next_decision_at"),
            _number(values.get("life_next_run_at")),
        ),
    )
    current = _legacy_activity_payload(values.get("current_activity"), fallback_end=next_at)
    recent = values.get("recent_activities")
    historical = [
        activity
        for item in (
            values.get("previous_activity"),
            *(recent if isinstance(recent, (list, tuple)) else ()),
        )
        if (activity := _legacy_activity_payload(item)) is not None
    ]
    historical.sort(
        key=lambda item: (
            _number(item.get("expected_end_at")),
            _number(item.get("started_at")),
        ),
        reverse=True,
    )
    previous = next((activity for activity in historical if activity != current), None)
    last_decision_at = max(
        0.0,
        _number(
            values.get("last_decision_at"),
            _number(values.get("life_last_run_at")),
        ),
    )
    raw_pattern = values.get("activity_pattern")
    repeat_count = (
        int(_number(raw_pattern.get("repeat_count")))
        if isinstance(raw_pattern, dict)
        else 0
    )
    raw_repetitions = values.get("repetition_count", values.get("continuation_count"))
    repetition_count = (
        max(0, min(3, raw_repetitions))
        if type(raw_repetitions) is int
        else max(0, min(3, repeat_count - 1))
    )
    retry_at = max(0.0, _number(values.get("retry_at")))
    claim_token = values.get("claim_token")
    claim_expires_at = values.get("claim_expires_at")
    last_error = values.get("last_error")
    if current is None and last_decision_at <= 0.0:
        next_at = 0.0
        retry_at = 0.0
        claim_token = None
        claim_expires_at = 0.0
        repetition_count = 0
    raw_recent = values.get("recent_source_ids")
    recent_source_ids = [
        source
        for item in (raw_recent if isinstance(raw_recent, (list, tuple)) else ())
        if (source := compact_text(item, 180))
    ][-6:]
    raw_quiet_streak = values.get("quiet_streak")
    quiet_streak = (
        max(0, min(3, raw_quiet_streak))
        if type(raw_quiet_streak) is int
        else (1 if current is not None and current.get("kind") == "quiet" else 0)
    )
    raw_score = values.get("last_candidate_score")
    last_candidate_score = (
        float(raw_score)
        if type(raw_score) in {int, float} and math.isfinite(float(raw_score))
        else None
    )
    return {
        "current_activity": current,
        "previous_activity": previous,
        "last_decision_at": last_decision_at,
        "next_decision_at": next_at,
        "retry_at": retry_at,
        "last_error": last_error,
        "claim_token": claim_token,
        "claim_expires_at": claim_expires_at,
        "repetition_count": repetition_count,
        "recent_source_ids": recent_source_ids,
        "quiet_streak": quiet_streak,
        "last_transition_reason": (
            compact_text(values.get("last_transition_reason"), 80)
            or "migrated_existing_state"
        ),
        "last_candidate_score": last_candidate_score,
        "last_candidate_source_id": (
            compact_text(values.get("last_candidate_source_id"), 180) or None
        ),
        "last_appraised_activity_id": (
            compact_text(values.get("last_appraised_activity_id"), 80) or None
        ),
        "last_appraised_at": max(0.0, _number(values.get("last_appraised_at"))),
        "last_appraisal_result": compact_text(
            values.get("last_appraisal_result"),
            32,
        ),
    }


def _legacy_memory_payload(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    status = compact_text(payload.get("status"), 24).casefold()
    thread_status = compact_text(payload.get("thread_status"), 24).casefold()
    if status and status not in {"active", "confirmed"}:
        return None
    if thread_status in {"resolved", "abandoned", "expired"}:
        return None
    old_kind = compact_text(payload.get("kind") or payload.get("category"), 32).casefold()
    kind = {
        "episode": "event",
        "task_outcome": "event",
        "working": "project",
        "open_thread": "concern",
        "unfinished_topic": "concern",
        "profile": "fact",
        "self": "fact",
        "relationship": "event",
        "correction": "fact",
        "stable_fact": "fact",
        "user_fact": "fact",
    }.get(old_kind, old_kind)
    source_type = compact_text(payload.get("source_type"), 40).casefold()
    source = (
        source_type
        if source_type not in {"", "unknown"}
        else compact_text(payload.get("source"), 40).casefold()
    )
    if source in {"generated_assistant", "speculative_inference"}:
        return None
    subject = compact_text(payload.get("subject"), 16).casefold()
    if subject not in _MEMORY_SUBJECTS:
        if old_kind == "self":
            subject = "akane"
        elif old_kind == "relationship":
            subject = "shared"
        elif source in {"user", "explicit_user", "verified_interface"}:
            subject = "user"
        else:
            subject = "shared"
    return {
        "id": payload.get("id"),
        "subject": subject,
        "kind": kind,
        "text": payload.get("text") or payload.get("content") or payload.get("summary"),
        "confidence": payload.get(
            "confidence",
            (
                0.8
                if status == "confirmed"
                or source in {"user", "explicit_user", "verified_interface"}
                else 0.65
            ),
        ),
        "created_at": payload.get("created_at", payload.get("updated_at", 0.0)),
        "updated_at": payload.get("updated_at", payload.get("created_at", 0.0)),
        "source_type": (
            "explicit_user"
            if source in {"user", "explicit_user", "verified_interface"}
            else "legacy"
        ),
        "source_id": payload.get("source_id"),
        "reason": payload.get("reason") or "Migrated existing memory.",
    }


def _legacy_profile_payload(
    payload: object,
    *,
    preserve_structured_memory: bool = False,
) -> object:
    if isinstance(payload, list):
        return [memory for item in payload if (memory := _legacy_memory_payload(item)) is not None]
    if not isinstance(payload, dict):
        return payload
    migrated = dict(payload)
    migrated["presence"] = _legacy_presence_payload(payload.get("presence"))
    raw_memories = payload.get("memories")
    if not preserve_structured_memory and isinstance(raw_memories, list):
        migrated["memories"] = [
            memory
            for item in raw_memories
            if (memory := _legacy_memory_payload(item)) is not None
        ]
    return migrated


@dataclass(frozen=True, slots=True)
class ChatTurn:
    turn_id: str
    role: str
    content: str
    timestamp: float
    source: str

    @classmethod
    def from_dict(cls, payload: object) -> "ChatTurn | None":
        if not isinstance(payload, dict):
            return None
        role = str(payload.get("role") or "").strip().lower()
        content = str(payload.get("content") or payload.get("text") or "").strip()
        if role not in {"user", "assistant"} or not content:
            return None
        return cls(
            compact_text(payload.get("turn_id"), 100) or uuid.uuid4().hex,
            role,
            content[:8_000],
            max(0.0, _number(payload.get("timestamp"))),
            compact_text(payload.get("source"), 32) or "unknown",
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def as_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def _complete_turns(turns: tuple[ChatTurn, ...]) -> tuple[ChatTurn, ...]:
    complete: list[ChatTurn] = []
    index = 0
    while index < len(turns):
        turn = turns[index]
        if turn.role == "assistant" and turn.source == "initiative":
            complete.append(turn)
            index += 1
            continue
        if (
            index + 1 < len(turns)
            and turn.role == "user"
            and turns[index + 1].role == "assistant"
        ):
            complete.extend((turn, turns[index + 1]))
            index += 2
        else:
            index += 1
    return tuple(complete)


@dataclass(frozen=True, slots=True)
class InitiativeOpportunity:
    opportunity_id: str
    reason: str
    source_type: str
    source_id: str
    context: str
    topic_key: str
    created_at: float
    not_before: float
    expires_at: float
    status: str = "pending"
    claim_token: str | None = None
    claim_expires_at: float = 0.0
    message: str | None = None
    evaluated_at: float = 0.0
    generated_at: float = 0.0
    delivery_channel: str | None = None
    delivered_at: float = 0.0
    delivery_message_id: str | None = None
    failed_channels: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: object, *, now: float) -> "InitiativeOpportunity | None":
        if not isinstance(payload, dict):
            return None
        opportunity_id = compact_text(payload.get("opportunity_id"), 100)
        reason = compact_text(payload.get("reason"), 100)
        source_type = compact_text(payload.get("source_type"), 40).casefold()
        source_id = compact_text(payload.get("source_id"), 160)
        context = compact_text(payload.get("context"), 420)
        topic_key = _key(payload.get("topic_key"))[:120]
        created = max(0.0, _number(payload.get("created_at")))
        not_before = max(created, _number(payload.get("not_before"), created))
        expires = max(0.0, _number(payload.get("expires_at")))
        status = compact_text(payload.get("status"), 24).casefold()
        if (
            not opportunity_id
            or not reason
            or not source_type
            or not source_id
            or not context
            or not topic_key
            or expires <= created
            or status
            not in {"pending", "pending_delivery", "sent", "dismissed", "expired"}
        ):
            return None
        claim_token = compact_text(payload.get("claim_token"), 100) or None
        claim_expires = max(0.0, _number(payload.get("claim_expires_at")))
        delivery_channel = compact_text(payload.get("delivery_channel"), 16).casefold() or None
        failed_channels = tuple(
            channel
            for item in (payload.get("failed_channels") or ())
            if (channel := compact_text(item, 16).casefold())
            in {"popup", "discord"}
        )
        if claim_token is None or claim_expires <= now:
            if status == "pending_delivery" and delivery_channel:
                failed_channels = tuple(dict.fromkeys((*failed_channels, delivery_channel)))
            claim_token = None
            claim_expires = 0.0
            if status != "sent":
                delivery_channel = None
        return cls(
            opportunity_id,
            reason,
            source_type,
            source_id,
            context,
            topic_key,
            created,
            not_before,
            expires,
            status,
            claim_token,
            claim_expires,
            compact_text(payload.get("message"), 500) or None,
            max(0.0, _number(payload.get("evaluated_at"))),
            max(0.0, _number(payload.get("generated_at"))),
            delivery_channel,
            max(0.0, _number(payload.get("delivered_at"))),
            compact_text(payload.get("delivery_message_id"), 160) or None,
            failed_channels,
        )

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "failed_channels": list(self.failed_channels)}


@dataclass(frozen=True, slots=True)
class SentInitiative:
    topic_key: str
    source_id: str
    source_type: str
    delivered_at: float

    @classmethod
    def from_dict(cls, payload: object) -> "SentInitiative | None":
        if not isinstance(payload, dict):
            return None
        topic = _key(payload.get("topic_key"))[:120]
        source = compact_text(payload.get("source_id"), 160)
        source_type = compact_text(payload.get("source_type"), 40).casefold()
        delivered = max(0.0, _number(payload.get("delivered_at")))
        return (
            cls(topic, source, source_type, delivered)
            if topic and source and source_type and delivered
            else None
        )


@dataclass(frozen=True, slots=True)
class InitiativeState:
    current: InitiativeOpportunity | None = None
    cooldown_until: float = 0.0
    recent: tuple[SentInitiative, ...] = ()
    handled_source_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: object, *, now: float) -> "InitiativeState":
        values = payload if isinstance(payload, dict) else {}
        recent = tuple(
            item
            for raw in (values.get("recent") or ())
            if (item := SentInitiative.from_dict(raw)) is not None
        )[-_MAX_RECENT_INITIATIVES:]
        return cls(
            InitiativeOpportunity.from_dict(values.get("current"), now=now),
            max(0.0, _number(values.get("cooldown_until"))),
            recent,
            tuple(
                source
                for raw in (values.get("handled_source_ids") or ())
                if (source := compact_text(raw, 160))
            )[-_MAX_RECENT_INITIATIVES:],
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "current": self.current.as_dict() if self.current else None,
            "cooldown_until": self.cooldown_until,
            "recent": [asdict(item) for item in self.recent],
            "handled_source_ids": list(self.handled_source_ids),
        }


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    conversation_id: str
    profile_id: str
    recent_turns: tuple[ChatTurn, ...] = ()
    last_user_at: float = 0.0
    last_assistant_at: float = 0.0
    updated_at: float = 0.0
    committed_request_ids: tuple[str, ...] = ()
    request_replies: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_dict(cls, key: str, payload: object) -> "ConversationRecord | None":
        if not isinstance(payload, dict):
            return None
        conversation_id = compact_text(payload.get("conversation_id") or key, 160)
        if not conversation_id:
            return None
        profile_id = canonical_profile_id(payload.get("profile_id"))
        raw_turns = payload.get("recent_turns")
        turns = tuple(
            turn
            for item in (raw_turns if isinstance(raw_turns, list) else [])
            if (turn := ChatTurn.from_dict(item)) is not None
        )
        turns = _trim_turns(turns)
        ids = tuple(
            value
            for item in (payload.get("committed_request_ids") or [])
            if (value := compact_text(item, 180))
        )[-32:]
        raw_replies = payload.get("request_replies")
        replies: list[tuple[str, str]] = []
        if isinstance(raw_replies, dict):
            replies = [
                (compact_text(key, 180), str(value))
                for key, value in raw_replies.items()
                if compact_text(key, 180) and isinstance(value, str)
            ]
        elif isinstance(raw_replies, list):
            for item in raw_replies:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and compact_text(item[0], 180)
                    and isinstance(item[1], str)
                ):
                    replies.append((compact_text(item[0], 180), item[1]))
        reply_map = dict(replies)
        for request_id in ids:
            if request_id in reply_map:
                continue
            for index, turn in enumerate(turns[:-1]):
                if (
                    turn.role == "user"
                    and turn.turn_id in {request_id, f"{request_id}:user"}
                    and turns[index + 1].role == "assistant"
                ):
                    reply_map[request_id] = turns[index + 1].content
                    break
        replies = list(reply_map.items())
        last_user = max(
            (
                max(0.0, _number(payload.get("last_user_at"))),
                *(turn.timestamp for turn in turns if turn.role == "user"),
            )
        )
        last_assistant = max(
            (
                max(0.0, _number(payload.get("last_assistant_at"))),
                *(turn.timestamp for turn in turns if turn.role == "assistant"),
            )
        )
        return cls(
            conversation_id=conversation_id,
            profile_id=profile_id,
            recent_turns=turns,
            last_user_at=last_user,
            last_assistant_at=last_assistant,
            updated_at=max(
                last_user,
                last_assistant,
                max(0.0, _number(payload.get("updated_at"))),
            ),
            committed_request_ids=ids,
            request_replies=tuple(replies[-32:]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "conversation_id": self.conversation_id,
            "profile_id": self.profile_id,
            "recent_turns": [turn.as_dict() for turn in self.recent_turns],
            "last_user_at": self.last_user_at,
            "last_assistant_at": self.last_assistant_at,
            "updated_at": self.updated_at,
            "committed_request_ids": list(self.committed_request_ids),
            "request_replies": dict(self.request_replies),
        }


def _trim_turns(turns: tuple[ChatTurn, ...]) -> tuple[ChatTurn, ...]:
    kept = list(turns[-_MAX_RECENT_TURNS:])
    while kept and kept[0].role == "assistant" and kept[0].source != "initiative":
        kept.pop(0)
    return tuple(kept)


@dataclass(frozen=True, slots=True)
class MoodState:
    valence: float = 0.0
    energy: float = 0.0
    cause: str = ""
    updated_at: float = 0.0

    @classmethod
    def from_dict(cls, payload: object, *, now: float) -> "MoodState":
        values = payload if isinstance(payload, dict) else {}
        updated = min(now, max(0.0, _number(values.get("updated_at"))))
        return cls(
            max(-1.0, min(1.0, _number(values.get("valence")))),
            max(-1.0, min(1.0, _number(values.get("energy")))),
            compact_text(values.get("cause"), 160),
            updated,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EmotionState:
    primary: str = "neutral"
    intensity: float = 0.0
    cause: str = ""
    source: str | None = None
    source_id: str | None = None
    started_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_dict(
        cls,
        payload: object,
        fallback_time: float = 0.0,
        *,
        now: float,
        migrating: bool,
    ) -> "EmotionState":
        if not isinstance(payload, dict) or not set(payload) & {
            "primary",
            "dominant",
            "mood",
            "intensity",
            "irritation",
            "frustration",
            "concern",
            "cause",
            "updated_at",
        }:
            return cls()
        values = payload
        primary = compact_text(
            values.get("primary") or values.get("dominant") or values.get("mood"),
            32,
        ).casefold()
        if primary not in EMOTION_VOCABULARY:
            primary = "neutral"
        intensity = max(
            0.0,
            min(
                1.0,
                _number(
                    values.get("intensity"),
                    max(
                        _number(values.get("irritation")),
                        _number(values.get("frustration")),
                        _number(values.get("concern")),
                    ),
                ),
            ),
        )
        cause = compact_text(values.get("cause"), 160)
        updated = min(now, max(0.0, _number(values.get("updated_at"), fallback_time)))
        source = compact_text(values.get("source"), 32).casefold() or None
        if source not in _EMOTION_SOURCES:
            source = "self_reflection" if migrating and primary != "neutral" else None
        source_id = compact_text(values.get("source_id"), 120) or None
        started = min(
            now,
            max(
                0.0,
                _number(values.get("started_at"), updated if migrating else 0.0),
            ),
        )
        if primary == "neutral" or not cause:
            primary = "neutral"
            intensity = 0.0
            cause = ""
            source = None
            source_id = None
            started = 0.0
        return cls(primary, intensity, cause, source, source_id, started, updated)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def effective_mood(mood: MoodState, *, now: float) -> MoodState:
    if mood.updated_at <= 0.0:
        return mood
    elapsed_hours = max(0.0, (now - mood.updated_at) / 3600.0)
    decay = 0.985 ** elapsed_hours
    return replace(
        mood,
        valence=max(-1.0, min(1.0, mood.valence * decay)),
        energy=max(-1.0, min(1.0, mood.energy * decay)),
    )


def effective_emotion(emotion: EmotionState, *, now: float) -> EmotionState:
    if emotion.updated_at <= 0.0 or emotion.primary == "neutral":
        return emotion
    elapsed_hours = max(0.0, (now - emotion.updated_at) / 3600.0)
    intensity = max(0.0, min(1.0, emotion.intensity * (0.88 ** elapsed_hours)))
    if intensity < 0.08:
        return EmotionState(updated_at=emotion.updated_at)
    return replace(emotion, intensity=intensity)


@dataclass(frozen=True, slots=True)
class Memory:
    id: str
    subject: str
    kind: str
    text: str
    confidence: float
    created_at: float
    updated_at: float
    source_type: str = "legacy"
    source_id: str | None = None
    reason: str = ""

    @property
    def content(self) -> str:
        return (self.text if self.confidence >= 0.75 else f"Uncertain memory: {self.text}")

    @classmethod
    def from_dict(cls, payload: object) -> "Memory | None":
        if not isinstance(payload, dict):
            return None
        source_type = compact_text(payload.get("source_type"), 40).casefold()
        if source_type in {"generated_assistant", "speculative_inference"}:
            return None
        text = compact_text(
            payload.get("text") or payload.get("content") or payload.get("summary"),
            360,
        )
        if not text:
            return None
        subject = compact_text(payload.get("subject"), 16).casefold()
        if subject not in _MEMORY_SUBJECTS:
            actor = compact_text(payload.get("actor"), 24).casefold()
            subject = "user" if actor in {"arcane", "user"} else (
                "akane" if actor == "akane" else "shared"
            )
        kind = compact_text(payload.get("kind") or payload.get("category"), 32).casefold()
        aliases = {
            "stable_fact": "fact",
            "user_fact": "fact",
            "episode": "event",
            "working": "project",
            "open_thread": "concern",
            "unfinished_topic": "concern",
        }
        kind = aliases.get(kind, kind)
        if kind not in _MEMORY_KINDS:
            kind = "fact"
        created = max(0.0, _number(payload.get("created_at"), _number(payload.get("updated_at"))))
        updated = max(created, _number(payload.get("updated_at"), created))
        memory_id = compact_text(payload.get("id"), 100) or _stable_state_id(
            "memory",
            subject,
            kind,
            text,
            f"{created:.6f}",
        )
        source_type = source_type or "legacy"
        if source_type not in {
            "legacy", "explicit_user", "conversation", "migration", "offscreen_presence",
        }:
            return None
        return cls(
            memory_id,
            subject,
            kind,
            text,
            max(0.0, min(1.0, _number(payload.get("confidence"), 0.7))),
            created,
            updated,
            source_type,
            compact_text(payload.get("source_id"), 180) or None,
            compact_text(payload.get("reason"), 240),
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class Interest(str):
    """An Akane-owned topic with lightweight, revisable evidence."""

    def __new__(
        cls,
        topic: str,
        strength: float = 0.7,
        reason: str = "",
        created_at: float = 0.0,
        updated_at: float = 0.0,
        source_type: str = "legacy",
        source_ids: tuple[str, ...] = (),
        evidence_count: int = 1,
    ) -> "Interest":
        value = str.__new__(cls, compact_text(topic, 100))
        value.strength = max(0.0, min(1.0, float(strength)))
        value.reason = compact_text(reason, 240)
        value.created_at = max(0.0, float(created_at))
        value.updated_at = max(value.created_at, float(updated_at))
        value.source_type = compact_text(source_type, 40).casefold() or "legacy"
        value.source_ids = tuple(
            compact_text(item, 180) for item in source_ids if compact_text(item, 180)
        )[-6:]
        value.evidence_count = max(1, int(evidence_count))
        return value

    @property
    def topic(self) -> str:
        return str(self)

    @property
    def content(self) -> str:
        return str(self)

    @classmethod
    def from_dict(cls, payload: object) -> "Interest | None":
        if isinstance(payload, str):
            topic = compact_text(payload, 100)
            if not topic:
                return None
            identity = _key(topic) in {_key(item) for item in STARTING_INTEREST_TOPICS}
            return cls(
                topic,
                0.8 if identity else 0.6,
                "Established by Akane's identity." if identity else "Migrated established interest.",
                source_type="identity" if identity else "legacy",
            )
        if not isinstance(payload, dict):
            return None
        topic = compact_text(payload.get("topic"), 100)
        strength = payload.get("strength")
        source_type = compact_text(payload.get("source_type"), 40).casefold()
        source_ids = payload.get("source_ids")
        evidence_count = payload.get("evidence_count", 1)
        if (
            not topic
            or type(strength) not in {int, float}
            or not math.isfinite(float(strength))
            or source_type not in {"identity", "legacy", "conversation", "offscreen_presence"}
            or not isinstance(source_ids, (list, tuple))
            or type(evidence_count) is not int
        ):
            return None
        return cls(
            topic,
            float(strength),
            compact_text(payload.get("reason"), 240),
            _number(payload.get("created_at")),
            _number(payload.get("updated_at")),
            source_type,
            tuple(str(item) for item in source_ids),
            evidence_count,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "topic": str(self),
            "strength": self.strength,
            "reason": self.reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_type": self.source_type,
            "source_ids": list(self.source_ids),
            "evidence_count": self.evidence_count,
        }


STARTING_INTERESTS = tuple(Interest.from_dict(topic) for topic in STARTING_INTEREST_TOPICS)


@dataclass(frozen=True, slots=True)
class AkanePreference:
    topic: str
    stance: str
    reason: str
    updated_at: float

    @property
    def content(self) -> str:
        return f"{self.topic}: {self.stance} — {self.reason}"

    @classmethod
    def from_dict(cls, payload: object) -> "AkanePreference | None":
        if not isinstance(payload, dict):
            return None
        topic = compact_text(payload.get("topic"), 140)
        stance = compact_text(payload.get("stance"), 24).casefold()
        reason = compact_text(payload.get("reason"), 240)
        if not topic or stance not in _PREFERENCE_STANCES or not reason:
            return None
        return cls(topic, stance, reason, max(0.0, _number(payload.get("updated_at"))))


@dataclass(frozen=True, slots=True)
class CommunicationPreference:
    key: str
    value: str
    reason: str
    created_at: float
    updated_at: float
    source_type: str = "explicit_user"

    @classmethod
    def from_dict(cls, payload: object) -> "CommunicationPreference | None":
        if not isinstance(payload, dict):
            return None
        key = compact_text(payload.get("key"), 40).casefold()
        value = compact_text(payload.get("value"), 80)
        normalized_value = value.casefold()
        if key in _COMMUNICATION_VALUES:
            if normalized_value not in _COMMUNICATION_VALUES[key]:
                return None
            value = normalized_value
        elif key in _COMMUNICATION_FREE_TEXT_KEYS:
            if (
                not value
                or "\n" in value
                or "\r" in value
                or not re.fullmatch(r"[\w .,'’'-]{1,80}", value, re.UNICODE)
                or key == "preferred_name"
                and (
                    len(words(value)) > 4
                    or bool(words(value) & _UNSAFE_PREFERENCE_WORDS)
                )
            ):
                return None
        else:
            return None
        reason = compact_text(payload.get("reason"), 240)
        source_type = compact_text(payload.get("source_type"), 40).casefold()
        if not reason or source_type not in {"explicit_user", "legacy"}:
            return None
        created = max(0.0, _number(payload.get("created_at"), _number(payload.get("updated_at"))))
        return cls(
            key,
            value,
            reason,
            created,
            max(created, _number(payload.get("updated_at"), created)),
            source_type,
        )


@dataclass(frozen=True, slots=True)
class Opinion:
    topic: str
    position: str
    reason: str
    updated_at: float
    id: str = ""
    topic_key: str = ""
    confidence: float = 0.7
    created_at: float = 0.0
    source_type: str = "legacy"
    evidence_summary: str = field(default="", compare=False, repr=False)
    source_ids: tuple[str, ...] = ()
    domain: str = "general"
    importance: float = 0.5
    revision_count: int = 0

    @property
    def content(self) -> str:
        return f"{self.topic}: {self.position} — {self.reason}"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "topic": self.topic,
            "domain": self.domain,
            "position": self.position,
            "reason": self.reason,
            "confidence": self.confidence,
            "importance": self.importance,
            "source_ids": list(self.source_ids),
            "source_type": self.source_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision_count": self.revision_count,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "Opinion | None":
        if not isinstance(payload, dict):
            return None
        topic = compact_text(payload.get("topic"), 140)
        position = compact_text(payload.get("position"), 200)
        reason = compact_text(payload.get("reason"), 240)
        if not topic or not position or not reason:
            return None
        updated = max(0.0, _number(payload.get("updated_at")))
        created = max(0.0, _number(payload.get("created_at"), updated))
        topic_key = _key(topic)
        source_type = compact_text(payload.get("source_type"), 40).casefold() or "legacy"
        if source_type not in {"legacy", "conversation", "offscreen_presence"}:
            return None
        domain = compact_text(payload.get("domain"), 48).casefold() or "general"
        raw_revision_count = payload.get("revision_count", 0)
        revision_count = (
            max(0, raw_revision_count)
            if type(raw_revision_count) is int
            else 0
        )
        return cls(
            topic,
            position,
            reason,
            updated,
            compact_text(payload.get("id"), 100)
            or _stable_state_id("opinion", topic_key),
            topic_key,
            max(0.0, min(1.0, _number(payload.get("confidence"), 0.7))),
            min(created, updated) if updated else created,
            source_type,
            compact_text(payload.get("evidence_summary"), 280),
            tuple(
                dict.fromkeys(
                    source
                    for item in (
                        payload.get("source_ids")
                        if isinstance(payload.get("source_ids"), (list, tuple))
                        else ()
                    )
                    if (source := compact_text(item, 180))
                )
            )[-8:],
            domain,
            max(0.0, min(1.0, _number(payload.get("importance"), 0.5))),
            revision_count,
        )


@dataclass(frozen=True, slots=True)
class SelfModelItem:
    id: str
    category: str
    area: str
    description: str
    confidence: float
    source_ids: tuple[str, ...]
    created_at: float
    updated_at: float
    revision_count: int = 0

    @property
    def content(self) -> str:
        return f"{self.area}: {self.description}"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "area": self.area,
            "description": self.description,
            "confidence": self.confidence,
            "source_ids": list(self.source_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision_count": self.revision_count,
        }

    @classmethod
    def from_dict(cls, category: str, payload: object) -> "SelfModelItem | None":
        if category not in {"capability", "limitation", "trait"} or not isinstance(
            payload, dict
        ):
            return None
        item_id = compact_text(payload.get("id"), 100)
        area = compact_text(payload.get("area"), 80).casefold()
        description = compact_text(payload.get("description"), 280)
        confidence = payload.get("confidence")
        raw_sources = payload.get("source_ids")
        sources = tuple(
            dict.fromkeys(
                source
                for value in (
                    raw_sources if isinstance(raw_sources, (list, tuple)) else ()
                )
                if (source := compact_text(value, 180))
            )
        )[-8:]
        created = max(0.0, _number(payload.get("created_at")))
        updated = max(created, _number(payload.get("updated_at"), created))
        raw_revisions = payload.get("revision_count", 0)
        revisions = max(0, raw_revisions) if type(raw_revisions) is int else 0
        if (
            not item_id
            or not area
            or not description
            or type(confidence) not in {int, float}
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
            or not sources
        ):
            return None
        return cls(
            item_id,
            category,
            area,
            description,
            float(confidence),
            sources,
            created,
            updated,
            revisions,
        )


@dataclass(frozen=True, slots=True)
class ImprovementTarget:
    id: str
    area: str
    description: str
    reason: str
    priority: float
    source_ids: tuple[str, ...]
    created_at: float
    updated_at: float
    revision_count: int = 0

    @property
    def content(self) -> str:
        return f"{self.area}: {self.description} — {self.reason}"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "area": self.area,
            "description": self.description,
            "reason": self.reason,
            "priority": self.priority,
            "source_ids": list(self.source_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision_count": self.revision_count,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "ImprovementTarget | None":
        if not isinstance(payload, dict):
            return None
        target_id = compact_text(payload.get("id"), 100)
        area = compact_text(payload.get("area"), 80).casefold()
        description = compact_text(payload.get("description"), 240)
        reason = compact_text(payload.get("reason"), 240)
        priority = payload.get("priority")
        raw_sources = payload.get("source_ids")
        sources = tuple(
            dict.fromkeys(
                source
                for value in (
                    raw_sources if isinstance(raw_sources, (list, tuple)) else ()
                )
                if (source := compact_text(value, 180))
            )
        )[-8:]
        created = max(0.0, _number(payload.get("created_at")))
        updated = max(created, _number(payload.get("updated_at"), created))
        raw_revisions = payload.get("revision_count", 0)
        revisions = max(0, raw_revisions) if type(raw_revisions) is int else 0
        if (
            not target_id
            or not area
            or not description
            or not reason
            or type(priority) not in {int, float}
            or not math.isfinite(float(priority))
            or not 0.0 <= float(priority) <= 1.0
            or not sources
        ):
            return None
        return cls(
            target_id,
            area,
            description,
            reason,
            float(priority),
            sources,
            created,
            updated,
            revisions,
        )


@dataclass(frozen=True, slots=True)
class SelfModelState:
    capabilities: tuple[SelfModelItem, ...] = ()
    limitations: tuple[SelfModelItem, ...] = ()
    traits: tuple[SelfModelItem, ...] = ()
    improvement_targets: tuple[ImprovementTarget, ...] = ()

    @property
    def items(self) -> tuple[SelfModelItem, ...]:
        return (*self.capabilities, *self.limitations, *self.traits)

    def category_items(self, category: str) -> tuple[SelfModelItem, ...]:
        return {
            "capability": self.capabilities,
            "limitation": self.limitations,
            "trait": self.traits,
        }.get(category, ())

    def replace_category(
        self,
        category: str,
        values: tuple[SelfModelItem, ...],
    ) -> "SelfModelState":
        field_name = {
            "capability": "capabilities",
            "limitation": "limitations",
            "trait": "traits",
        }.get(category)
        return replace(self, **{field_name: values}) if field_name else self

    def as_dict(self) -> dict[str, object]:
        return {
            "capabilities": [item.as_dict() for item in self.capabilities],
            "limitations": [item.as_dict() for item in self.limitations],
            "traits": [item.as_dict() for item in self.traits],
            "improvement_targets": [
                target.as_dict() for target in self.improvement_targets
            ],
        }


@dataclass(frozen=True, slots=True)
class Strategy:
    id: str
    goal_id: str
    description: str
    status: str
    confidence: float
    source_ids: tuple[str, ...]
    created_at: float
    updated_at: float
    evaluation_count: int = 0
    opportunity_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    evidence_summary: str = "Not evaluated yet."
    last_evaluation_result: str = "insufficient_evidence"
    revision_count: int = 0

    @property
    def content(self) -> str:
        return self.description

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "description": self.description,
            "status": self.status,
            "confidence": self.confidence,
            "source_ids": list(self.source_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "evaluation_count": self.evaluation_count,
            "opportunity_count": self.opportunity_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "evidence_summary": self.evidence_summary,
            "last_evaluation_result": self.last_evaluation_result,
            "revision_count": self.revision_count,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "Strategy | None":
        if not isinstance(payload, dict):
            return None
        strategy_id = compact_text(payload.get("id"), 100)
        goal_id = compact_text(payload.get("goal_id"), 100)
        description = compact_text(payload.get("description"), 240)
        status = compact_text(payload.get("status"), 16).casefold()
        confidence = payload.get("confidence")
        raw_sources = payload.get("source_ids")
        source_ids = tuple(
            dict.fromkeys(
                source
                for value in (
                    raw_sources if isinstance(raw_sources, (list, tuple)) else ()
                )
                if (source := compact_text(value, 180))
            )
        )[-12:]
        created_at = max(0.0, _number(payload.get("created_at")))
        updated_at = max(created_at, _number(payload.get("updated_at"), created_at))
        counters = tuple(
            payload.get(name)
            for name in (
                "evaluation_count",
                "opportunity_count",
                "success_count",
                "failure_count",
                "revision_count",
            )
        )
        evidence_summary = compact_text(payload.get("evidence_summary"), 280)
        result = compact_text(payload.get("last_evaluation_result"), 32).casefold()
        if (
            not strategy_id
            or not goal_id
            or not description
            or status not in {"active", "completed", "abandoned"}
            or type(confidence) not in {int, float}
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
            or not source_ids
            or any(type(value) is not int or value < 0 for value in counters)
            or counters[2] + counters[3] > counters[1]
            or not evidence_summary
            or result
            not in {
                "insufficient_evidence",
                "improving",
                "unchanged",
                "worsening",
                "completed",
                "abandoned",
                "stale_goal",
            }
        ):
            return None
        return cls(
            strategy_id,
            goal_id,
            description,
            status,
            float(confidence),
            source_ids,
            created_at,
            updated_at,
            *counters[:4],
            evidence_summary,
            result,
            counters[4],
        )


def _runtime_capability_for(
    description: str,
    area: str,
) -> CapabilityFact | None:
    return CAPABILITY_REGISTRY.match_persistent_claim(
        "capability",
        area,
        description,
    )


def _runtime_limitation_for(
    description: str,
    area: str,
) -> CapabilityFact | None:
    return CAPABILITY_REGISTRY.match_persistent_claim(
        "limitation",
        area,
        description,
    )


@dataclass(frozen=True, slots=True)
class RelationshipEntry:
    summary: str
    confidence: float
    updated_at: float
    evidence_count: int = 1
    source_ids: tuple[str, ...] = ()

    @property
    def content(self) -> str:
        return self.summary

    @classmethod
    def from_dict(cls, payload: object) -> "RelationshipEntry | None":
        if isinstance(payload, str):
            summary = compact_text(payload, 300)
            return cls(summary, 0.7, 0.0) if summary else None
        if not isinstance(payload, dict):
            return None
        summary = compact_text(payload.get("summary") or payload.get("content"), 300)
        if not summary:
            return None
        return cls(
            summary,
            max(0.0, min(1.0, _number(payload.get("confidence"), 0.7))),
            max(0.0, _number(payload.get("updated_at"))),
            max(1, int(payload.get("evidence_count", 1)))
            if type(payload.get("evidence_count", 1)) is int
            else 1,
            tuple(
                compact_text(item, 180)
                for item in (
                    payload.get("source_ids")
                    if isinstance(payload.get("source_ids"), (list, tuple))
                    else ()
                )
                if compact_text(item, 180)
            )[-6:],
        )


@dataclass(frozen=True, slots=True)
class RelationshipState:
    patterns: tuple[RelationshipEntry, ...] = ()
    shared_context: tuple[RelationshipEntry, ...] = ()
    unresolved_events: tuple[RelationshipEntry, ...] = ()

    @classmethod
    def from_dict(cls, payload: object) -> "RelationshipState":
        values = payload if isinstance(payload, dict) else {}

        def entries(name: str) -> tuple[RelationshipEntry, ...]:
            raw = values.get(name)
            if not isinstance(raw, (list, tuple)):
                return ()
            return tuple(
                entry
                for item in raw
                if not (
                    name == "unresolved_events"
                    and isinstance(item, dict)
                    and compact_text(item.get("status"), 24).casefold()
                    == "resolved"
                )
                if (entry := RelationshipEntry.from_dict(item)) is not None
            )[-_MAX_RELATIONSHIP_ENTRIES:]

        return cls(entries("patterns"), entries("shared_context"), entries("unresolved_events"))

    def as_dict(self) -> dict[str, object]:
        def payload(values: tuple[RelationshipEntry, ...]) -> list[dict[str, object]]:
            return [
                {**asdict(item), "source_ids": list(item.source_ids)}
                for item in values
            ]

        return {
            "patterns": payload(self.patterns),
            "shared_context": payload(self.shared_context),
            "unresolved_events": payload(self.unresolved_events),
        }


@dataclass(frozen=True, slots=True)
class ProfileState:
    mood: MoodState = MoodState()
    emotion: EmotionState = EmotionState()
    presence: PresenceState = PresenceState()
    memories: tuple[Memory, ...] = ()
    interests: tuple[Interest, ...] = STARTING_INTERESTS
    preferences: tuple[AkanePreference, ...] = ()
    communication_preferences: tuple[CommunicationPreference, ...] = ()
    opinions: tuple[Opinion, ...] = ()
    relationship: RelationshipState = RelationshipState()
    initiative: InitiativeState = InitiativeState()
    updated_at: float = 0.0

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        now: float,
        migrating: bool = False,
    ) -> "ProfileState | None":
        if isinstance(payload, list):
            memories = tuple(
                memory
                for item in payload
                if (memory := Memory.from_dict(item)) is not None
            )
            return cls(
                presence=PresenceState.from_dict({}, now=now),
                memories=_merge_memories((), memories),
            )
        if not isinstance(payload, dict):
            return None
        updated = max(0.0, _number(payload.get("updated_at")))
        raw_memories = payload.get("memories")
        memories = tuple(
            memory
            for item in (raw_memories if isinstance(raw_memories, list) else [])
            if (memory := Memory.from_dict(item)) is not None
        )
        interests = _merge_interests(
            STARTING_INTERESTS,
            tuple(
                interest
                for item in (
                    payload.get("interests")
                    if isinstance(payload.get("interests"), (list, tuple))
                    else ()
                )
                if (interest := Interest.from_dict(item)) is not None
            ),
        )
        raw_preferences = payload.get("preferences")
        preferences = tuple(
            preference
            for item in (
                raw_preferences if isinstance(raw_preferences, (list, tuple)) else ()
            )
            if (preference := AkanePreference.from_dict(item)) is not None
        )
        raw_opinions = payload.get("opinions")
        opinions = tuple(
            opinion
            for item in (raw_opinions if isinstance(raw_opinions, (list, tuple)) else ())
            if (opinion := Opinion.from_dict(item)) is not None
        )
        raw_communication = payload.get("communication_preferences")
        communication_preferences = tuple(
            preference
            for item in (
                raw_communication
                if isinstance(raw_communication, (list, tuple))
                else ()
            )
            if (preference := CommunicationPreference.from_dict(item)) is not None
        )
        return cls(
            mood=MoodState.from_dict(payload.get("mood"), now=now),
            emotion=EmotionState.from_dict(
                payload.get("emotion"),
                updated,
                now=now,
                migrating=migrating,
            ),
            presence=PresenceState.from_dict(
                payload.get("presence"),
                now=now,
            ),
            memories=_merge_memories((), memories),
            interests=interests,
            preferences=_merge_preferences((), preferences),
            communication_preferences=_merge_communication_preferences(
                (),
                communication_preferences,
            ),
            opinions=_merge_opinions((), opinions),
            relationship=RelationshipState.from_dict(payload.get("relationship")),
            initiative=InitiativeState.from_dict(
                payload.get("initiative"),
                now=now,
            ),
            updated_at=updated,
        )

    def as_dict(self, *, include_opinions: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "updated_at": self.updated_at,
            "mood": self.mood.as_dict(),
            "emotion": self.emotion.as_dict(),
            "presence": self.presence.as_dict(),
            "memories": [memory.as_dict() for memory in self.memories],
            "interests": [interest.as_dict() for interest in self.interests],
            "preferences": [asdict(item) for item in self.preferences],
            "communication_preferences": [
                asdict(item) for item in self.communication_preferences
            ],
            "relationship": self.relationship.as_dict(),
            "initiative": self.initiative.as_dict(),
        }
        if include_opinions:
            payload["opinions"] = [item.as_dict() for item in self.opinions]
        return payload


def _new_profile(now: float) -> ProfileState:
    return ProfileState(presence=PresenceState(), updated_at=0.0)


def _handled_initiative(
    state: InitiativeState,
    opportunity: InitiativeOpportunity,
) -> InitiativeState:
    return replace(
        state,
        handled_source_ids=tuple(
            dict.fromkeys((*state.handled_source_ids, opportunity.source_id))
        )[-_MAX_RECENT_INITIATIVES:],
    )


def _settle_initiative(state: InitiativeState, *, now: float) -> InitiativeState:
    opportunity = state.current
    if opportunity is None:
        return state
    if (opportunity.status in {"pending", "pending_delivery"} and opportunity.expires_at <= now):
        expired = replace(
            opportunity,
            status="expired",
            evaluated_at=max(opportunity.evaluated_at, now),
            claim_token=None,
            claim_expires_at=0.0,
            delivery_channel=None,
        )
        return _handled_initiative(replace(state, current=expired), expired)
    if opportunity.claim_token and opportunity.claim_expires_at <= now:
        failed = opportunity.failed_channels
        if opportunity.status == "pending_delivery" and opportunity.delivery_channel:
            failed = tuple(dict.fromkeys((*failed, opportunity.delivery_channel)))
        return replace(
            state,
            current=replace(
                opportunity,
                claim_token=None,
                claim_expires_at=0.0,
                delivery_channel=None,
                failed_channels=failed,
            ),
        )
    return state


def _initiative_source_exists(profile: ProfileState, opportunity: InitiativeOpportunity) -> bool:
    if opportunity.source_type == "offscreen_life":
        return any(
            opportunity.source_id == f"offscreen_life:{activity.started_at:.6f}"
            for activity in (
                profile.presence.current_activity,
                profile.presence.previous_activity,
            )
            if activity is not None
        )
    if opportunity.source_type in {"memory", "reminder"}:
        return any(item.id == opportunity.source_id for item in profile.memories)
    if opportunity.source_type == "realization":
        return any(
            opportunity.source_id
            == f"realization:{item.updated_at:.6f}:{_key(item.topic)[:60]}"
            for item in profile.opinions
        )
    if opportunity.source_type == "relationship":
        return any(
            opportunity.source_id
            == f"relationship:{item.updated_at:.6f}:{_key(item.summary)[:60]}"
            for item in (
                *profile.relationship.unresolved_events,
                *profile.relationship.shared_context,
            )
        )
    return False


def _ordinary_delivery_allowed(
    state: InitiativeState,
    opportunity: InitiativeOpportunity,
    *,
    now: float,
) -> bool:
    if opportunity.source_type == "reminder":
        return True
    if state.cooldown_until > now:
        return False
    local_date = build_time_context(now=now).local_date
    sent_today = sum(
        item.source_type != "reminder"
        and build_time_context(now=item.delivered_at).local_date == local_date
        for item in state.recent
    )
    return sent_today < MAX_ORDINARY_INITIATIVES_PER_LOCAL_DAY


def _initiative_from_conversation_change(
    before: ProfileState,
    after: ProfileState,
    *,
    now: float,
) -> InitiativeOpportunity | None:
    source: tuple[str, str, str, str] | None = None
    previous_memory_ids = {item.id for item in before.memories}
    memory = next(
        (
            item
            for item in reversed(after.memories)
            if item.id not in previous_memory_ids
            and item.kind in {"commitment", "project", "concern"}
            and item.confidence >= 0.75
        ),
        None,
    )
    if memory is not None:
        source = ("unresolved grounded memory", "memory", memory.id, memory.text)
    if source is None:
        previous_relationship = {
            (item.summary, item.updated_at)
            for item in before.relationship.unresolved_events
        }
        event = next(
            (
                item
                for item in reversed(after.relationship.unresolved_events)
                if (item.summary, item.updated_at) not in previous_relationship
            ),
            None,
        )
        if event is not None:
            source = (
                "meaningful unresolved relationship context",
                "relationship",
                f"relationship:{event.updated_at:.6f}:{_key(event.summary)[:60]}",
                event.summary,
            )
    if source is None:
        previous_opinions = {
            (item.topic, item.position, item.updated_at)
            for item in before.opinions
        }
        opinion = next(
            (
                item
                for item in reversed(after.opinions)
                if (item.topic, item.position, item.updated_at)
                not in previous_opinions
            ),
            None,
        )
        if opinion is not None:
            source = (
                "a personally meaningful conclusion Akane reached",
                "realization",
                f"realization:{opinion.updated_at:.6f}:{_key(opinion.topic)[:60]}",
                opinion.content,
            )
    if source is None:
        return None
    reason, source_type, source_id, evidence = source
    return InitiativeOpportunity(
        uuid.uuid4().hex,
        reason,
        source_type,
        source_id,
        evidence,
        _key(evidence)[:120],
        now,
        now + 15.0 * 60.0,
        now + 7.0 * 24.0 * 3600.0,
    )


def _with_conversation_initiative(
    before: ProfileState,
    after: ProfileState,
    *,
    now: float,
) -> ProfileState:
    opportunity = _initiative_from_conversation_change(before, after, now=now)
    if opportunity is None:
        return after
    initiative = _settle_initiative(after.initiative, now=now)
    active = initiative.current
    if (
        active is not None
        and active.status in {"pending", "pending_delivery"}
        and active.expires_at > now
    ):
        return after
    if (
        opportunity.source_id in initiative.handled_source_ids
        or any(
            recent.source_id == opportunity.source_id
            or (
                recent.source_type != "reminder"
                and _similar(recent.topic_key, opportunity.topic_key) >= 0.78
            )
            for recent in initiative.recent
        )
    ):
        return after
    return replace(
        after,
        initiative=replace(initiative, current=opportunity),
        updated_at=max(after.updated_at, now),
    )


def effective_emotional_state(profile: ProfileState, *, now: float) -> ProfileState:
    """Return lazy mood/emotion decay without mutating or persisting state."""

    return replace(
        profile,
        mood=effective_mood(profile.mood, now=now),
        emotion=effective_emotion(profile.emotion, now=now),
    )


def _affect_category(text: object) -> str:
    value = _key(text)
    if _HOSTILE_PHYSICAL.search(value) or _DIRECT_HOSTILITY.search(value):
        return "hostility"
    if _APOLOGY.search(value):
        return "apology"
    if _WARMTH.search(value):
        return "warmth"
    if _CONCERN.search(value):
        return "concern"
    return "neutral"


def _recent_affect_count(conversation: ConversationRecord, category: str) -> int:
    return sum(
        _affect_category(turn.content) == category
        for turn in conversation.recent_turns[-_AFFECT_HISTORY_TURNS:]
        if turn.role == "user"
    )


def _decayed_emotional_state(profile: ProfileState, *, now: float) -> ProfileState:
    """Materialize one bounded recovery step only while committing a turn."""

    effective = effective_emotional_state(profile, now=now)
    mood = effective.mood
    emotion = effective.emotion
    if (abs(mood.valence) < 0.02 and abs(mood.energy) < 0.02 and not mood.cause):
        mood = MoodState()
    else:
        mood = replace(mood, updated_at=now)
    if emotion.primary == "neutral" and emotion.updated_at <= 0.0:
        emotion = EmotionState()
    elif emotion.primary == "neutral" or emotion.intensity < 0.08:
        emotion = EmotionState(updated_at=now)
    else:
        emotion = replace(emotion, updated_at=now)
    return replace(profile, mood=mood, emotion=emotion)


def _apply_conversation_affect(
    profile: ProfileState,
    conversation: ConversationRecord,
    *,
    user_text: str,
    source_id: str | None,
    now: float,
) -> tuple[ProfileState, dict[str, object]]:
    """Apply the deterministic, bounded affect transition for one user turn."""

    prior = profile
    next_profile = _decayed_emotional_state(profile, now=now)
    category = _affect_category(user_text)
    hostile_count = _recent_affect_count(conversation, "hostility")
    repetition = min(2, hostile_count)
    emotion = next_profile.emotion
    mood = next_profile.mood
    signal: dict[str, object] = {
        "category": category,
        "strength": 0.0,
        "repetition": repetition if category == "hostility" else 0,
        "preview": False,
        "committed": True,
    }

    if category == "hostility":
        strength = min(0.62, 0.30 + 0.11 * repetition)
        primary = "angry" if repetition >= 2 else "irritated"
        intensity = min(1.0, emotion.intensity * 0.78 + strength * 0.55 + 0.08 * repetition)
        cause = ("the repeated hostile behavior" if repetition else "the hostile behavior")
        emotion = EmotionState(
            primary,
            intensity,
            cause,
            "conversation",
            source_id,
            emotion.started_at if emotion.primary in {"irritated", "angry"} else now,
            now,
        )
        mood = MoodState(
            max(-1.0, mood.valence - 0.045 - 0.02 * repetition),
            max(-1.0, min(1.0, mood.energy + 0.02 + 0.01 * repetition)),
            cause,
            now,
        )
        signal.update(strength=strength, cause=cause)
    elif category in {"apology", "warmth"} and (
        emotion.primary in {"irritated", "angry", "sad", "disappointed", "frustrated"}
        or mood.valence <= -0.12
    ):
        strength = 0.24 if category == "apology" else 0.16
        intensity = max(0.0, emotion.intensity - strength)
        cause = "the apology after the conflict" if category == "apology" else "the user's warmth after the conflict"
        emotion = (
            EmotionState(updated_at=now)
            if intensity < 0.08
            else replace(emotion, intensity=intensity, cause=cause, updated_at=now)
        )
        mood = MoodState(min(0.0, mood.valence + strength * 0.22), mood.energy * 0.96, cause, now)
        signal.update(category="repair", strength=strength, cause=cause)
    elif category == "warmth":
        strength = min(0.48, 0.24 + 0.07 * _recent_affect_count(conversation, "warmth"))
        intensity = min(1.0, emotion.intensity * 0.65 + strength * 0.60)
        cause = "the user's warmth and reassurance"
        emotion = EmotionState(
            "content",
            intensity,
            cause,
            "conversation",
            source_id,
            emotion.started_at if emotion.primary == "content" else now,
            now,
        )
        mood = MoodState(
            min(1.0, mood.valence + 0.035 + strength * 0.05),
            min(1.0, mood.energy + 0.01),
            cause,
            now,
        )
        signal.update(strength=strength, cause=cause)
    elif category == "concern":
        strength = 0.28
        cause = "the user's concern"
        emotion = EmotionState(
            "concerned",
            max(emotion.intensity * 0.60, strength),
            cause,
            "conversation",
            source_id,
            emotion.started_at if emotion.primary == "concerned" else now,
            now,
        )
        mood = MoodState(max(-1.0, mood.valence - 0.02), min(1.0, mood.energy + 0.02), cause, now)
        signal.update(strength=strength, cause=cause)

    next_profile = replace(next_profile, mood=mood, emotion=emotion)
    if next_profile != prior:
        next_profile = replace(next_profile, updated_at=now)
    signal["prior_emotion"] = prior.emotion.as_dict()
    signal["prior_mood"] = prior.mood.as_dict()
    signal["emotion"] = emotion.as_dict()
    signal["mood"] = mood.as_dict()
    return next_profile, signal


def format_emotional_context(
    profile: ProfileState,
    *,
    now: float,
    include_unappraised: bool = False,
) -> str:
    effective = effective_emotional_state(profile, now=now)
    mood = effective.mood
    emotion = effective.emotion
    lines: list[str] = []
    if mood.updated_at > 0.0 or mood.cause:
        lines.append(f"mood.valence={mood.valence:+.2f}; mood.energy={mood.energy:+.2f}")
        if mood.cause:
            lines.append(f"mood.cause={mood.cause}")
    elif include_unappraised:
        lines.append("mood=unappraised")

    if emotion.primary != "neutral" and emotion.intensity >= 0.08:
        lines.append(
            f"emotion.primary={emotion.primary}; emotion.intensity={emotion.intensity:.2f}"
        )
        if emotion.cause:
            lines.append(f"emotion.cause={emotion.cause}")
    elif include_unappraised and emotion.updated_at <= 0.0:
        lines.append("emotion=unappraised")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    profile_id: str
    conversation_id: str
    revision: int
    profile: ProfileState
    conversation: ConversationRecord
    recent_turns: tuple[ChatTurn, ...]
    relevant_memories: tuple[Memory, ...]
    now: float
    last_profile_assistant_at: float = 0.0
    last_profile_user_at: float = 0.0
    last_profile_initiative: ChatTurn | None = None
    accepted_memory_operations: tuple[str, ...] = ()
    rejected_state_operations: tuple[str, ...] = ()
    ownership_classification: tuple[str, ...] = ()
    relevant_opinions: tuple[Opinion, ...] = ()
    relevant_preferences: tuple[AkanePreference, ...] = ()
    relevant_interests: tuple[Interest, ...] = ()
    relevant_relationship: tuple[RelationshipEntry, ...] = ()
    familiarity: str = ""
    affect_transition: dict[str, object] | None = None
    self_model: SelfModelState = SelfModelState()
    relevant_self_model: tuple[object, ...] = ()
    strategies: tuple[Strategy, ...] = ()
    relevant_strategies: tuple[Strategy, ...] = ()


def _familiarity_context(
    profile: ProfileState,
    conversations: dict[str, ConversationRecord],
    profile_id: str,
) -> str:
    """Describe shared history qualitatively; never expose relationship scores."""

    exchanges = sum(
        sum(turn.role == "user" for turn in _complete_turns(record.recent_turns))
        for record in conversations.values()
        if record.profile_id == profile_id
    )
    shared_evidence = (
        len(profile.relationship.shared_context)
        + len(profile.relationship.unresolved_events)
        + sum(item.evidence_count >= 2 for item in profile.relationship.patterns)
    )
    if exchanges >= 12 or shared_evidence >= 4:
        return "Familiarity is established through repeated shared conversations."
    if exchanges >= 4 or shared_evidence >= 2:
        return "Familiarity is growing through several shared conversations."
    return ""


def _relevant_profile_state(
    profile: ProfileState,
    query: str,
    now: float,
) -> tuple[
    tuple[Memory, ...],
    tuple[Opinion, ...],
    tuple[AkanePreference, ...],
    tuple[Interest, ...],
    tuple[RelationshipEntry, ...],
]:
    relationship = profile.relationship
    opinions = _relevant_values(
        profile.opinions,
        query,
        now,
        confidence_of=lambda item: 0.7 * item.confidence + 0.3 * item.importance,
        limit=3,
    )
    preferences = _relevant_values(
        profile.preferences,
        query,
        now,
        confidence_of=lambda _item: 0.7,
        limit=3,
    )
    preferences = tuple(
        item
        for item in preferences
        if not any(_similar(item.topic, opinion.topic) >= 0.70 for opinion in opinions)
    )
    return (
        _relevant_values(
            profile.memories,
            query,
            now,
            text_of=lambda item: item.text,
            limit=MEMORY_MAX_RESULTS,
        ),
        opinions,
        preferences,
        _relevant_values(
            tuple(item for item in profile.interests if item.strength >= 0.25),
            query,
            now,
            confidence_of=lambda item: item.strength,
            limit=3,
        ),
        _relevant_values(
            (
                *(item for item in relationship.patterns if item.evidence_count >= 2),
                *relationship.shared_context,
                *relationship.unresolved_events,
            ),
            query,
            now,
            limit=3,
        ),
    )


def _relevant_self_model(
    state: SelfModelState,
    query: str,
    now: float,
) -> tuple[object, ...]:
    categories: set[str] = set()
    if _SELF_QUERY_CAPABILITY.search(query):
        categories.add("capability")
    if _SELF_QUERY_LIMITATION.search(query):
        categories.update(("limitation", "improvement"))
    if _SELF_QUERY_TRAIT.search(query):
        categories.add("trait")
    query_key = _key(query)
    if "about yourself" in query_key or "understand about yourself" in query_key:
        categories.update(("capability", "limitation", "trait", "improvement"))

    scored: list[tuple[float, float, object]] = []
    for item in state.items:
        score = lightweight_relevance_score(
            query,
            item.content,
            now=now,
            updated_at=item.updated_at,
            confidence=item.confidence,
        )
        if item.category in categories:
            score = max(score, 0.31)
        elif not (_terms(query) & _terms(item.content)):
            score = 0.0
        if score >= 0.26:
            scored.append((score, item.updated_at, item))
    for target in state.improvement_targets:
        score = lightweight_relevance_score(
            query,
            target.content,
            now=now,
            updated_at=target.updated_at,
            confidence=target.priority,
        )
        if "improvement" in categories:
            score = max(score, 0.31)
        elif not (_terms(query) & _terms(target.content)):
            score = 0.0
        if score >= 0.26:
            scored.append((score, target.updated_at, target))
    return tuple(item for _score, _updated, item in sorted(scored, reverse=True)[:3])


def _strategy_metric_kind(
    strategy: Strategy,
    goal: ImprovementTarget,
) -> str | None:
    text = f"{strategy.description} {goal.description} {goal.reason}"
    if _QUESTION_BEHAVIOR.search(text) or (
        re.search(r"\bdirect(?:ly|ness)?\b", text, re.IGNORECASE)
        and re.search(r"\banswer", text, re.IGNORECASE)
    ):
        return "clarification_directness"
    return None


def _strategy_applies(
    strategy: Strategy,
    goal: ImprovementTarget,
    query: str,
) -> bool:
    if strategy.status != "active" or not query.strip():
        return False
    if _strategy_metric_kind(strategy, goal) == "clarification_directness":
        if _AMBIGUOUS_QUESTION.search(query) or _TECHNICAL_QUESTION.search(query):
            return False
        return bool(_PERSONAL_QUESTION.search(query) and query.rstrip().endswith("?"))
    return bool(
        _terms(query) & _terms(f"{strategy.description} {goal.content}")
        and lightweight_relevance_score(
            query,
            f"{strategy.description} {goal.content}",
            now=0.0,
            updated_at=0.0,
            confidence=strategy.confidence,
        )
        >= 0.28
    )


def _relevant_strategies(
    strategies: tuple[Strategy, ...],
    state: SelfModelState,
    query: str,
) -> tuple[Strategy, ...]:
    goals = {goal.id: goal for goal in state.improvement_targets}
    relevant = tuple(
        strategy
        for strategy in strategies
        if (goal := goals.get(strategy.goal_id)) is not None
        and _strategy_applies(strategy, goal, query)
    )
    return tuple(
        sorted(
            relevant,
            key=lambda item: (item.confidence, item.updated_at, item.id),
            reverse=True,
        )[:2]
    )


def _interest_presence_source_id(interest: Interest) -> str:
    return _stable_state_id("presence-interest", str(interest))


def _relationship_presence_source_id(entry: RelationshipEntry) -> str:
    return _stable_state_id(
        "presence-relationship",
        entry.summary,
        f"{entry.updated_at:.6f}",
    )


def _presence_candidates(profile: ProfileState) -> tuple[PresenceCandidate, ...]:
    """Build autonomous candidates only from Akane-owned or shared safe state."""

    candidates: list[PresenceCandidate] = []
    for interest in profile.interests:
        if (
            interest.strength < 0.50
            or interest.source_type == "offscreen_presence"
            or _SENSITIVE_PATTERN.search(str(interest))
        ):
            continue
        candidates.append(
            PresenceCandidate(
                "revisiting_interest",
                str(interest),
                "interest",
                tuple(
                    dict.fromkeys(
                        (_interest_presence_source_id(interest), *interest.source_ids)
                    )
                )[-8:],
                interest.source_type,
                interest.strength,
                interest.updated_at,
                interest.strength,
                interest.strength,
            )
        )
    for opinion in profile.opinions:
        if (
            opinion.confidence < 0.50
            or opinion.source_type == "offscreen_presence"
            or _SENSITIVE_PATTERN.search(opinion.content)
        ):
            continue
        candidates.append(
            PresenceCandidate(
                "reconsidering_opinion",
                opinion.topic,
                "opinion",
                (opinion.id,),
                opinion.source_type,
                opinion.confidence,
                opinion.updated_at,
                max(0.45, 1.0 - opinion.confidence),
            )
        )
    for memory in profile.memories:
        if (
            memory.subject != "akane"
            or memory.source_type != "conversation"
            or memory.kind not in {"event", "project", "concern"}
            or memory.confidence < 0.65
            or _SENSITIVE_PATTERN.search(memory.text)
        ):
            continue
        candidates.append(
            PresenceCandidate(
                "following_unfinished_thought",
                memory.text,
                "akane_experience",
                (memory.id,),
                memory.source_type,
                memory.confidence,
                memory.updated_at,
                0.95 if memory.kind in {"project", "concern"} else 0.65,
                unresolved=memory.kind in {"project", "concern"},
            )
        )
    unresolved_ids = {
        _relationship_presence_source_id(entry)
        for entry in profile.relationship.unresolved_events
    }
    for entry in (
        *profile.relationship.unresolved_events,
        *profile.relationship.shared_context,
    ):
        source_id = _relationship_presence_source_id(entry)
        if entry.confidence < 0.65 or _SENSITIVE_PATTERN.search(entry.summary):
            continue
        candidates.append(
            PresenceCandidate(
                "reflecting_on_shared_thread",
                entry.summary,
                "shared_thread",
                tuple(dict.fromkeys((source_id, *entry.source_ids)))[-8:],
                "relationship",
                entry.confidence,
                entry.updated_at,
                1.0 if source_id in unresolved_ids else 0.70,
                unresolved=source_id in unresolved_ids,
            )
        )
    return tuple(candidates)


def _presence_emotion_weights(profile: ProfileState, *, now: float) -> dict[str, float]:
    primary = effective_emotion(profile.emotion, now=now).primary
    weights: dict[str, float] = {}
    if primary in {"curious", "interested", "inspired", "excited"}:
        weights["revisiting_interest"] = 1.0
    if primary in {"uncertain", "concerned", "anxious"}:
        weights["reconsidering_opinion"] = 0.9
        weights["reflecting_on_shared_thread"] = 0.5
    if primary in {"affectionate", "content", "hopeful"}:
        weights["reflecting_on_shared_thread"] = 0.7
    if primary in {"frustrated", "disappointed", "sad"}:
        weights["following_unfinished_thought"] = 0.6
    return weights


@dataclass(frozen=True, slots=True)
class PresenceConsequenceEligibility:
    eligible: bool
    reason: str
    basis: str = ""


def _presence_activity_source(
    profile: ProfileState,
    activity: PresenceActivity | None,
) -> tuple[object | None, str, str]:
    if activity is None or activity.kind in {"quiet", "legacy"} or not activity.source_ids:
        return None, "", ""
    source_id = activity.source_ids[0]
    if activity.subject_kind == "interest":
        source = next(
            (
                item
                for item in profile.interests
                if _interest_presence_source_id(item) == source_id
                and item.source_type != "offscreen_presence"
            ),
            None,
        )
        return (
            source,
            f"{source.content} — {source.reason}" if source is not None else "",
            "interest",
        )
    if activity.subject_kind == "opinion":
        source = next(
            (
                item
                for item in profile.opinions
                if item.id == source_id and item.source_type != "offscreen_presence"
            ),
            None,
        )
        return source, source.content if source is not None else "", "opinion"
    if activity.subject_kind == "akane_experience":
        source = next(
            (
                item
                for item in profile.memories
                if item.id == source_id
                and item.subject == "akane"
                and item.source_type == "conversation"
            ),
            None,
        )
        return source, source.text if source is not None else "", "akane_experience"
    if activity.subject_kind == "shared_thread":
        source = next(
            (
                item
                for item in profile.relationship.unresolved_events
                if _relationship_presence_source_id(item) == source_id
                and not _SENSITIVE_PATTERN.search(item.summary)
            ),
            None,
        )
        if source is not None:
            return source, source.summary, "unresolved_shared_thread"
        source = next(
            (
                item
                for item in profile.relationship.shared_context
                if _relationship_presence_source_id(item) == source_id
                and not _SENSITIVE_PATTERN.search(item.summary)
            ),
            None,
        )
        return source, source.summary if source is not None else "", "shared_context"
    return None, "", ""


def presence_consequence_eligibility(
    profile: ProfileState,
    activity: PresenceActivity | None,
    *,
    now: float,
) -> PresenceConsequenceEligibility:
    """Gate optional episodic inference using only committed trusted state."""

    if activity is None:
        return PresenceConsequenceEligibility(False, "no_completed_presence")
    if activity.kind in {"quiet", "legacy"}:
        return PresenceConsequenceEligibility(False, "quiet_or_legacy")
    if not activity.source_ids:
        return PresenceConsequenceEligibility(False, "missing_source_ids")
    if activity.grounding_confidence < _PRESENCE_CONSEQUENCE_MIN_CONFIDENCE:
        return PresenceConsequenceEligibility(False, "insufficient_grounding")
    if (
        activity.expected_end_at - activity.started_at
        < _PRESENCE_CONSEQUENCE_MIN_INTERVAL_SECONDS
    ):
        return PresenceConsequenceEligibility(False, "trivial_interval")
    if activity.origin in {"identity", "legacy", "legacy_presence", "offscreen_presence"}:
        return PresenceConsequenceEligibility(False, "routine_source")
    source, basis, source_kind = _presence_activity_source(profile, activity)
    if source is None or not basis:
        return PresenceConsequenceEligibility(False, "invalid_source_id")
    expected = {
        "revisiting_interest": "interest",
        "reconsidering_opinion": "opinion",
        "following_unfinished_thought": "akane_experience",
        "reflecting_on_shared_thread": "shared_thread",
    }.get(activity.kind)
    if expected != activity.subject_kind or not _grounded(activity.subject, basis):
        return PresenceConsequenceEligibility(False, "fabricated_presence")
    updated_at = max(0.0, float(getattr(source, "updated_at", 0.0)))
    if (
        updated_at <= 0.0
        or now - updated_at > _PRESENCE_CONSEQUENCE_MAX_SOURCE_AGE_SECONDS
    ):
        return PresenceConsequenceEligibility(False, "stale_subject")
    if profile.presence.last_appraised_activity_id == activity.activity_id:
        return PresenceConsequenceEligibility(False, "already_appraised")
    if any(
        memory.source_type == "offscreen_presence"
        and memory.source_id == activity.activity_id
        for memory in profile.memories
    ) or any(activity.activity_id in interest.source_ids for interest in profile.interests) or any(
        activity.activity_id in opinion.source_ids for opinion in profile.opinions
    ):
        return PresenceConsequenceEligibility(False, "recent_experience_repeat")
    if source_kind == "interest" and getattr(source, "strength", 1.0) >= 0.95:
        return PresenceConsequenceEligibility(False, "no_meaningful_change")
    if source_kind == "akane_experience" and getattr(source, "kind", "") not in {
        "project",
        "concern",
    }:
        return PresenceConsequenceEligibility(False, "routine_experience")
    if source_kind == "shared_context":
        return PresenceConsequenceEligibility(False, "routine_shared_context")
    return PresenceConsequenceEligibility(True, "eligible", basis)


def presence_activity_basis(
    profile: ProfileState,
    activity: PresenceActivity | None,
) -> str:
    """Resolve an orientation back to its still-committed authoritative source."""

    _source, basis, _source_kind = _presence_activity_source(profile, activity)
    return basis


def _merge_memories(
    current: tuple[Memory, ...],
    additions: tuple[Memory, ...],
) -> tuple[Memory, ...]:
    result = list(current)
    for memory in additions:
        duplicate = next(
            (
                index
                for index, existing in enumerate(result)
                if existing.subject == memory.subject
                and existing.kind == memory.kind
                and _duplicate_text(existing.text, memory.text)
            ),
            None,
        )
        if duplicate is None:
            result.append(memory)
        else:
            existing = result[duplicate]
            result[duplicate] = (
                memory
                if (memory.updated_at, memory.confidence)
                >= (existing.updated_at, existing.confidence)
                else existing
            )
    return tuple(
        sorted(result, key=lambda item: (item.updated_at, item.created_at))[
            -MEMORY_MAX_ENTRIES_PER_PROFILE:
        ]
    )


def _merge_latest(
    current: tuple[_T, ...],
    additions: tuple[_T, ...],
    *,
    matches: Callable[[_T, _T], bool],
    limit: int,
    sort_key: Callable[[_T], object] | None = None,
) -> tuple[_T, ...]:
    result = list(current)
    for item in additions:
        match = next(
            (index for index, existing in enumerate(result) if matches(existing, item)),
            None,
        )
        if match is None:
            result.append(item)
        elif item.updated_at >= result[match].updated_at:
            result[match] = item
    ordered = sorted(result, key=sort_key) if sort_key else result
    return tuple(ordered[-limit:])


def _merge_preferences(
    current: tuple[AkanePreference, ...],
    additions: tuple[AkanePreference, ...],
) -> tuple[AkanePreference, ...]:
    return _merge_latest(
        current,
        additions,
        matches=lambda existing, item: _key(existing.topic) == _key(item.topic),
        limit=_MAX_PREFERENCES,
    )


def _merge_interests(
    current: tuple[Interest, ...],
    additions: tuple[Interest, ...],
) -> tuple[Interest, ...]:
    return _merge_latest(
        current,
        additions,
        matches=lambda existing, item: _key(existing) == _key(item),
        sort_key=lambda item: (item.updated_at, item.created_at),
        limit=_MAX_INTERESTS,
    )


def _merge_communication_preferences(
    current: tuple[CommunicationPreference, ...],
    additions: tuple[CommunicationPreference, ...],
) -> tuple[CommunicationPreference, ...]:
    return _merge_latest(
        current,
        additions,
        matches=lambda existing, item: existing.key == item.key
        and (
            item.key != "forbidden_phrase"
            or _key(existing.value) == _key(item.value)
        ),
        sort_key=lambda item: (item.updated_at, item.created_at),
        limit=_MAX_COMMUNICATION_PREFERENCES,
    )


def _merge_opinions(
    current: tuple[Opinion, ...],
    additions: tuple[Opinion, ...],
) -> tuple[Opinion, ...]:
    return _merge_latest(
        current,
        additions,
        matches=lambda existing, item: _key(existing.topic) == _key(item.topic),
        limit=_MAX_OPINIONS,
    )


def conversation_opinion_candidate(
    user_text: str,
    assistant_text: str,
    current: tuple[Opinion, ...],
) -> dict[str, object] | None:
    """Derive one validator-bound candidate from Akane's explicit visible stance."""

    reply = compact_text(assistant_text, 800)
    if not reply or not _AKANE_OPINION_ADOPTION.search(reply):
        return None
    sentences = tuple(
        compact_text(value, 280)
        for value in re.split(r"(?<=[.!?])\s+", reply)
        if compact_text(value, 280)
    )
    adopted_index = next(
        (
            index
            for index, sentence in enumerate(sentences)
            if _AKANE_OPINION_ADOPTION.search(sentence)
            and not _AKANE_SELF_CLAIM.search(sentence)
            and not _TRANSIENT_SELF_REACTION.search(sentence)
        ),
        None,
    )
    if adopted_index is None:
        return None
    adopted = sentences[adopted_index]
    reason = ""
    because = re.search(r"\b(?:because|since)\b\s+(.+)", adopted, re.IGNORECASE)
    if because:
        position = compact_text(adopted[:because.start()].rstrip(" ,;:-") + ".", 200)
        reason = compact_text(because.group(1), 240)
    else:
        position = compact_text(adopted, 200)
        reason = compact_text(" ".join(sentences[adopted_index + 1:]), 240)

    self_domain = bool(_SELF_OPINION_QUERY.search(user_text))
    existing: Opinion | None = None
    if self_domain:
        existing = max(
            (item for item in current if item.domain == "self"),
            key=lambda item: item.updated_at,
            default=None,
        )
        topic = existing.topic if existing is not None else "who I am"
        domain = "self"
    else:
        patterns = (
            r"\bwhat do you think (?:of|about)\s+(.+?)(?:[?.!]|$)",
            r"\bhow do you feel about\s+(.+?)(?:[?.!]|$)",
            r"\bdo you (?:like|love|dislike|hate|prefer|value)\s+(.+?)(?:[?.!]|$)",
        )
        topic = next(
            (
                compact_text(match.group(1), 140)
                for pattern in patterns
                if (match := re.search(pattern, user_text, re.IGNORECASE)) is not None
            ),
            "",
        )
        if _key(topic) in {"it", "them", "that", "this", "those", "these"}:
            adoption = re.search(
                r"\bi (?:like|love|dislike|hate|prefer|value)\s+([^.!?]+)",
                user_text,
                re.IGNORECASE,
            )
            topic = compact_text(adoption.group(1), 140) if adoption else ""
        if not topic:
            return None
        existing = max(
            (
                item
                for item in current
                if _similar(item.topic, topic) >= 0.62
            ),
            key=lambda item: (_similar(item.topic, topic), item.updated_at),
            default=None,
        )
        if existing is not None:
            topic = existing.topic
            domain = existing.domain
        else:
            domain = "general"

    has_explanation = bool(reason and _key(reason) != _key(position))
    confidence = existing.confidence if existing is not None else 0.68
    importance = existing.importance if existing is not None else (
        0.72 if self_domain and has_explanation else 0.62 if has_explanation else 0.40
    )
    operation = "form"
    candidate: dict[str, object] = {
        "op": operation,
        "topic": topic,
        "domain": domain,
        "position": position,
        "reason": reason or position,
        "confidence": confidence,
        "importance": importance,
    }
    if existing is not None:
        if not has_explanation:
            return None
        operation = (
            "reinforce"
            if _similar(existing.position, position) >= 0.70
            else "update"
        )
        candidate.update(
            op=operation,
            target_id=existing.id,
            confidence=(
                min(0.95, existing.confidence + 0.05)
                if operation == "reinforce"
                else existing.confidence
            ),
        )
    return candidate


def communication_directives(profile: ProfileState) -> tuple[str, ...]:
    """Compile typed profile settings into fixed, non-executable wording."""

    templates = {
        ("formality", "casual"): "Use a casual tone.",
        ("formality", "neutral"): "Use a neutral conversational tone.",
        ("formality", "formal"): "Use a formal tone.",
        ("verbosity", "short"): "Keep ordinary replies short.",
        ("verbosity", "balanced"): "Use balanced reply length.",
        ("verbosity", "detailed"): "Give detailed replies when useful.",
        ("bluntness", "gentle"): "Phrase criticism gently.",
        ("bluntness", "balanced"): "Balance candor with tact.",
        ("bluntness", "direct"): "Be direct.",
        ("teasing", "allow"): "Light teasing is welcome when it fits.",
        ("teasing", "avoid"): "Do not tease Arcane.",
        ("pet_names", "allow"): "Pet names are welcome when natural.",
        ("pet_names", "avoid"): "Avoid pet names.",
        ("technical_detail", "concise"): "Keep technical explanations concise.",
        ("technical_detail", "balanced"): "Use balanced technical detail.",
        ("technical_detail", "detailed"): "Keep technical explanations detailed.",
        ("routine_questions", "allow"): "Routine follow-up questions are welcome.",
        ("routine_questions", "avoid"): "Avoid routine follow-up questions.",
    }
    lines: list[str] = []
    for item in sorted(
        profile.communication_preferences,
        key=lambda value: (value.updated_at, value.created_at),
    ):
        if item.key == "preferred_name":
            lines.append(f"Address the user as ‘{item.value}’.")
        elif item.key == "forbidden_phrase":
            lines.append(f"Do not use the exact phrase ‘{item.value}’.")
        elif directive := templates.get((item.key, item.value)):
            lines.append(directive)
    return tuple(lines)


def communication_preference_debug(profile: ProfileState) -> tuple[str, ...]:
    return tuple(
        f"{item.key}={item.value}"
        for item in sorted(
            profile.communication_preferences,
            key=lambda value: (value.key, value.value),
        )
    )


def _merge_relationship_entries(
    current: tuple[RelationshipEntry, ...],
    additions: tuple[RelationshipEntry, ...],
) -> tuple[RelationshipEntry, ...]:
    result = list(current)
    for item in additions:
        match = next(
            (
                index
                for index, existing in enumerate(result)
                if _similar(existing.summary, item.summary) >= 0.70
            ),
            None,
        )
        if match is None:
            result.append(item)
            continue
        existing = result[match]
        source_ids = tuple(dict.fromkeys((*existing.source_ids, *item.source_ids)))[-6:]
        new_evidence = max(0, len(source_ids) - len(existing.source_ids))
        result[match] = RelationshipEntry(
            item.summary if item.updated_at >= existing.updated_at else existing.summary,
            min(0.98, max(existing.confidence, item.confidence) + 0.04 * new_evidence),
            max(existing.updated_at, item.updated_at),
            max(existing.evidence_count, len(source_ids), item.evidence_count),
            source_ids,
        )
    return tuple(
        sorted(result, key=lambda item: item.updated_at)[-_MAX_RELATIONSHIP_ENTRIES:]
    )


def _merge_relationship(
    current: RelationshipState,
    addition: RelationshipState,
) -> RelationshipState:
    return RelationshipState(
        _merge_relationship_entries(current.patterns, addition.patterns),
        _merge_relationship_entries(current.shared_context, addition.shared_context),
        _merge_relationship_entries(
            current.unresolved_events,
            addition.unresolved_events,
        ),
    )


def _merge_profiles(left: ProfileState, right: ProfileState, *, now: float) -> ProfileState:
    mood = max(
        (left.mood, right.mood),
        key=lambda item: (
            item.updated_at,
            int(bool(item.cause) or item.valence != 0.0 or item.energy != 0.0),
        ),
    )
    emotion = max(
        (left.emotion, right.emotion),
        key=lambda item: (
            item.updated_at,
            int(item.primary != "neutral" or item.intensity > 0.0 or bool(item.cause)),
        ),
    )
    current_candidates = tuple(
        activity
        for activity in (
            left.presence.current_activity,
            right.presence.current_activity,
        )
        if activity is not None
    )
    current = max(
        current_candidates,
        key=lambda item: (item.started_at, item.expected_end_at),
        default=None,
    )
    historical = (
        *current_candidates,
        left.presence.previous_activity,
        right.presence.previous_activity,
    )
    previous = max(
        (item for item in historical if item is not None and item != current),
        key=lambda item: (item.started_at, item.expected_end_at),
        default=None,
    )
    source_presence = max(
        (left.presence, right.presence),
        key=lambda item: (
            int(current is not None and item.current_activity == current),
            item.last_decision_at,
            item.next_decision_at,
        ),
    )
    presence = replace(
        source_presence,
        current_activity=current,
        previous_activity=previous,
        claim_token=None,
        claim_expires_at=0.0,
    )
    initiative_source = max(
        (left.initiative, right.initiative),
        key=lambda item: (
            item.current.created_at if item.current else 0.0,
            item.cooldown_until,
        ),
    )
    recent = sorted(
        {
            (item.topic_key, item.source_id): item
            for item in (*left.initiative.recent, *right.initiative.recent)
        }.values(),
        key=lambda item: item.delivered_at,
    )[-_MAX_RECENT_INITIATIVES:]
    initiative = replace(
        initiative_source,
        cooldown_until=max(
            left.initiative.cooldown_until,
            right.initiative.cooldown_until,
        ),
        recent=tuple(recent),
        handled_source_ids=tuple(
            dict.fromkeys(
                (
                    *left.initiative.handled_source_ids,
                    *right.initiative.handled_source_ids,
                )
            )
        )[-_MAX_RECENT_INITIATIVES:],
    )
    return ProfileState(
        mood=mood,
        emotion=emotion,
        presence=presence,
        memories=_merge_memories(left.memories, right.memories),
        interests=_merge_interests(left.interests, right.interests),
        preferences=_merge_preferences(left.preferences, right.preferences),
        communication_preferences=_merge_communication_preferences(
            left.communication_preferences,
            right.communication_preferences,
        ),
        opinions=_merge_opinions(left.opinions, right.opinions),
        relationship=_merge_relationship(left.relationship, right.relationship),
        initiative=initiative,
        updated_at=max(left.updated_at, right.updated_at),
    )


def _merge_conversations(
    left: ConversationRecord,
    right: ConversationRecord,
) -> ConversationRecord:
    turns_by_id: dict[str, ChatTurn] = {
        turn.turn_id: turn for turn in (*left.recent_turns, *right.recent_turns)
    }
    turns = _trim_turns(
        tuple(
            sorted(
                turns_by_id.values(),
                key=lambda item: (item.timestamp, item.turn_id),
            )
        )
    )
    return ConversationRecord(
        left.conversation_id,
        left.profile_id,
        turns,
        max(left.last_user_at, right.last_user_at),
        max(left.last_assistant_at, right.last_assistant_at),
        max(left.updated_at, right.updated_at),
        tuple(dict.fromkeys((*left.committed_request_ids, *right.committed_request_ids)))[-32:],
        tuple(dict((*left.request_replies, *right.request_replies)).items())[-32:],
    )


def _clean_conversation_output(record: ConversationRecord) -> ConversationRecord:
    turns: list[ChatTurn] = []
    dropped_requests: set[str] = set()
    for turn in record.recent_turns:
        if turn.role != "assistant":
            turns.append(turn)
            continue
        content = clean_visible_output(turn.content)
        if not content:
            if turns and turns[-1].role == "user" and turn.source != "initiative":
                turns.pop()
            if turn.turn_id.endswith(":assistant"):
                dropped_requests.add(turn.turn_id.removesuffix(":assistant"))
            continue
        turns.append(replace(turn, content=content))

    replies = []
    for request_id, reply in record.request_replies:
        content = clean_visible_output(reply)
        if content:
            replies.append((request_id, content))
        else:
            dropped_requests.add(request_id)
    return replace(
        record,
        recent_turns=_trim_turns(tuple(turns)),
        committed_request_ids=tuple(
            request_id
            for request_id in record.committed_request_ids
            if request_id not in dropped_requests
        ),
        request_replies=tuple(replies),
    )


def _validate_canonical_profile(
    payload: dict[str, object],
    profile: ProfileState,
    *,
    now: float,
) -> None:
    """Reject current-schema data that parsing would silently discard."""

    normalized = dict(payload)
    raw_presence = payload.get("presence")
    if isinstance(raw_presence, dict):
        normalized["presence"] = PresenceState.from_dict(raw_presence, now=now).as_dict()
    raw_initiative = payload.get("initiative")
    if isinstance(raw_initiative, dict):
        normalized_initiative = dict(raw_initiative)
        raw_current = raw_initiative.get("current")
        if isinstance(raw_current, dict):
            normalized_current = dict(raw_current)
            token = normalized_current.get("claim_token")
            expires = _number(normalized_current.get("claim_expires_at"))
            if isinstance(token, str) and token.strip() and expires <= now:
                channel = compact_text(normalized_current.get("delivery_channel"), 16).casefold()
                failed = list(normalized_current.get("failed_channels") or ())
                if (
                    normalized_current.get("status") == "pending_delivery"
                    and channel in {"popup", "discord"}
                    and channel not in failed
                ):
                    failed.append(channel)
                normalized_current["claim_token"] = None
                normalized_current["claim_expires_at"] = 0.0
                if normalized_current.get("status") != "sent":
                    normalized_current["delivery_channel"] = None
                normalized_current["failed_channels"] = failed
            normalized_initiative["current"] = normalized_current
        normalized["initiative"] = normalized_initiative
    raw_mood = payload.get("mood")
    if isinstance(raw_mood, dict):
        normalized_mood = dict(raw_mood)
        if _number(normalized_mood.get("updated_at")) > now:
            normalized_mood["updated_at"] = now
        normalized["mood"] = normalized_mood
    raw_emotion = payload.get("emotion")
    if isinstance(raw_emotion, dict):
        normalized_emotion = dict(raw_emotion)
        for field_name in ("started_at", "updated_at"):
            if _number(normalized_emotion.get(field_name)) > now:
                normalized_emotion[field_name] = now
        normalized["emotion"] = normalized_emotion
    if normalized != profile.as_dict(include_opinions=False):
        raise ValueError("canonical profile contains malformed state")


def _validate_canonical_conversation(
    payload: dict[str, object],
    conversation: ConversationRecord,
) -> None:
    """Require every current-schema conversation field to survive parsing."""

    normalized = dict(payload)
    normalized["profile_id"] = canonical_profile_id(payload.get("profile_id"))
    if normalized != conversation.as_dict():
        raise ValueError("canonical conversation contains malformed state")


class StateStore:
    """Sole production owner of validated state and atomic persistence."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        opinions_path: Path | None = None,
        self_model_path: Path | None = None,
        strategies_path: Path | None = None,
    ) -> None:
        self._path = Path(path) if path is not None else LONG_TERM_MEMORY_PATH
        self._default_path = path is None
        self._opinions_path = (
            Path(opinions_path)
            if opinions_path is not None
            else OPINIONS_PATH
            if self._default_path
            else self._path.with_name("opinions.json")
        )
        self._self_model_path = (
            Path(self_model_path)
            if self_model_path is not None
            else SELF_MODEL_PATH
            if self._default_path
            else self._path.with_name("self_model.json")
        )
        self._strategies_path = (
            Path(strategies_path)
            if strategies_path is not None
            else STRATEGIES_PATH
            if self._default_path
            else self._path.with_name("strategies.json")
        )
        self._lock = threading.RLock()
        self._profiles: dict[str, ProfileState] = {}
        self._conversations: dict[str, ConversationRecord] = {}
        self._self_model = SelfModelState()
        self._strategies: tuple[Strategy, ...] = ()
        self._revision = 0
        self._committed_at = 0.0
        self._autonomy_wake: Callable[[str], None] | None = None
        self._presence_failure_count = 0
        self._initiative_failure_count = 0
        self._expired_claim_recoveries = 0
        self._load()

    def _document(
        self,
        profiles: dict[str, ProfileState],
        conversations: dict[str, ConversationRecord],
        revision: int,
        committed_at: float,
    ) -> dict[str, object]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "canonical_profile_id": OWNER_PROFILE_ID,
            "revision": revision,
            "committed_at": committed_at,
            "profiles": {
                key: value.as_dict(include_opinions=False)
                for key, value in profiles.items()
            },
            "conversations": {
                key: value.as_dict() for key, value in conversations.items()
            },
        }

    @staticmethod
    def _opinion_document(opinions: tuple[Opinion, ...]) -> dict[str, object]:
        return {
            "version": OPINION_SCHEMA_VERSION,
            "opinions": [opinion.as_dict() for opinion in opinions],
        }

    @staticmethod
    def _self_model_document(state: SelfModelState) -> dict[str, object]:
        return {"version": SELF_MODEL_SCHEMA_VERSION, **state.as_dict()}

    @staticmethod
    def _strategy_document(strategies: tuple[Strategy, ...]) -> dict[str, object]:
        return {
            "version": STRATEGY_SCHEMA_VERSION,
            "strategies": [strategy.as_dict() for strategy in strategies],
        }

    def _decode_self_model_source(self, payload: object) -> SelfModelState:
        fields = {
            "version",
            "capabilities",
            "limitations",
            "traits",
            "improvement_targets",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("version") != SELF_MODEL_SCHEMA_VERSION
            or type(payload.get("version")) is not int
        ):
            raise RuntimeError(
                f"Akane self-model recovery is required for {self._self_model_path}: "
                "malformed or unsupported schema"
            )

        def items(field_name: str, category: str) -> tuple[SelfModelItem, ...]:
            raw = payload.get(field_name)
            if not isinstance(raw, list):
                return ()
            return tuple(
                item
                for value in raw
                if (item := SelfModelItem.from_dict(category, value)) is not None
            )

        raw_targets = payload.get("improvement_targets")
        targets = tuple(
            target
            for value in (raw_targets if isinstance(raw_targets, list) else ())
            if (target := ImprovementTarget.from_dict(value)) is not None
        )
        state = SelfModelState(
            items("capabilities", "capability"),
            items("limitations", "limitation"),
            items("traits", "trait"),
            targets,
        )
        ids = [item.id for item in state.items]
        target_ids = [target.id for target in state.improvement_targets]
        limitation_ids = {item.id for item in state.limitations}
        malformed = (
            any(not isinstance(payload.get(name), list) for name in fields - {"version"})
            or len(ids) != len(set(ids))
            or len(target_ids) != len(set(target_ids))
            or len(state.capabilities) > _MAX_SELF_MODEL_ITEMS_PER_CATEGORY
            or len(state.limitations) > _MAX_SELF_MODEL_ITEMS_PER_CATEGORY
            or len(state.traits) > _MAX_SELF_MODEL_ITEMS_PER_CATEGORY
            or len(state.improvement_targets) > _MAX_IMPROVEMENT_TARGETS
            or any(
                (evidence := _runtime_capability_for(item.description, item.area))
                is None
                or item.source_ids != (evidence.source_id,)
                for item in state.capabilities
            )
            or any(
                any(source.startswith("runtime:") for source in item.source_ids)
                and (
                    (evidence := _runtime_limitation_for(item.description, item.area))
                    is None
                    or item.source_ids != (evidence.source_id,)
                )
                for item in state.limitations
            )
            or any(
                not any(source in limitation_ids for source in target.source_ids)
                for target in state.improvement_targets
            )
            or self._self_model_document(state) != payload
        )
        if malformed:
            raise RuntimeError(
                f"Akane self-model recovery is required for {self._self_model_path}: "
                "canonical state is malformed"
            )
        return state

    def _decode_strategy_source(self, payload: object) -> tuple[Strategy, ...]:
        if (
            not isinstance(payload, dict)
            or set(payload) != {"version", "strategies"}
            or type(payload.get("version")) is not int
            or payload.get("version") != STRATEGY_SCHEMA_VERSION
            or not isinstance(payload.get("strategies"), list)
        ):
            raise RuntimeError(
                f"Akane strategy recovery is required for {self._strategies_path}: "
                "malformed or unsupported schema"
            )
        raw = payload["strategies"]
        strategies = tuple(
            strategy
            for value in raw
            if (strategy := Strategy.from_dict(value)) is not None
        )
        ids = [strategy.id for strategy in strategies]
        malformed = (
            len(strategies) != len(raw)
            or len(ids) != len(set(ids))
            or len(strategies) > _MAX_STRATEGIES
            or sum(strategy.status == "active" for strategy in strategies)
            > _MAX_ACTIVE_STRATEGIES
            or any(
                _FOUNDATIONAL_STRATEGY.search(strategy.description)
                or _BROAD_STRATEGY.search(strategy.description)
                for strategy in strategies
            )
            or self._strategy_document(strategies) != payload
        )
        if malformed:
            raise RuntimeError(
                f"Akane strategy recovery is required for {self._strategies_path}: "
                "canonical state is malformed"
            )
        return strategies

    @staticmethod
    def _repair_stale_strategies(
        strategies: tuple[Strategy, ...],
        self_model: SelfModelState,
        *,
        now: float,
    ) -> tuple[Strategy, ...]:
        goal_ids = {goal.id for goal in self_model.improvement_targets}
        return tuple(
            replace(
                strategy,
                status="abandoned",
                updated_at=now,
                evidence_summary="The linked improvement target no longer exists.",
                last_evaluation_result="stale_goal",
                revision_count=strategy.revision_count + 1,
            )
            if strategy.status == "active" and strategy.goal_id not in goal_ids
            else strategy
            for strategy in strategies
        )

    @staticmethod
    def _migrate_opinion_provenance(
        opinions: tuple[Opinion, ...],
    ) -> tuple[Opinion, ...]:
        return tuple(
            opinion
            if opinion.source_ids
            else replace(
                opinion,
                source_type="legacy",
                evidence_summary=(
                    opinion.evidence_summary or "Migrated persisted Akane opinion."
                ),
                source_ids=(
                    "migration:"
                    + _stable_state_id(
                        "opinion-migration",
                        opinion.id,
                        opinion.topic,
                        opinion.created_at,
                    ),
                ),
            )
            for opinion in opinions
        )

    def _decode_opinion_source(
        self,
        payload: object,
    ) -> tuple[tuple[Opinion, ...], bool]:
        if isinstance(payload, list):
            raw_opinions = payload
            version = 0
        elif isinstance(payload, dict):
            raw_version = payload.get("version", 0)
            if type(raw_version) is not int or raw_version not in range(
                OPINION_SCHEMA_VERSION + 1
            ):
                raise RuntimeError(
                    f"Akane opinion recovery is required for {self._opinions_path}: "
                    "unsupported schema"
                )
            version = raw_version
            if version == OPINION_SCHEMA_VERSION and set(payload) != {
                "version",
                "opinions",
            }:
                raise RuntimeError(
                    f"Akane opinion recovery is required for {self._opinions_path}: "
                    "malformed header"
                )
            raw_opinions = payload.get("opinions")
        else:
            raw_opinions = None
            version = 0
        if not isinstance(raw_opinions, list):
            raise RuntimeError(
                f"Akane opinion recovery is required for {self._opinions_path}: "
                "opinions are unavailable"
            )
        parsed = tuple(
            opinion
            for item in raw_opinions
            if (opinion := Opinion.from_dict(item)) is not None
        )
        if version != OPINION_SCHEMA_VERSION:
            parsed = self._migrate_opinion_provenance(parsed)
        opinions = _merge_opinions((), parsed)
        if version == OPINION_SCHEMA_VERSION and (
            len(parsed) != len(raw_opinions)
            or len(opinions) != len(parsed)
            or any(not opinion.source_ids for opinion in opinions)
            or [opinion.as_dict() for opinion in opinions] != raw_opinions
        ):
            raise RuntimeError(
                f"Akane opinion recovery is required for {self._opinions_path}: "
                "canonical opinions are malformed"
            )
        if raw_opinions and not opinions:
            raise RuntimeError(
                f"Akane opinion recovery is required for {self._opinions_path}: "
                "no recoverable opinions"
            )
        return opinions, version != OPINION_SCHEMA_VERSION

    def _replace_all(
        self,
        profiles: dict[str, ProfileState],
        conversations: dict[str, ConversationRecord],
        *,
        committed_at: float,
        self_model: SelfModelState | None = None,
        strategies: tuple[Strategy, ...] | None = None,
    ) -> bool:
        next_self_model = self._self_model if self_model is None else self_model
        next_strategies = self._strategies if strategies is None else strategies
        if (
            profiles == self._profiles
            and conversations == self._conversations
            and next_self_model == self._self_model
            and next_strategies == self._strategies
        ):
            return False
        revision = self._revision + 1
        current_opinions = (
            self._profiles.get(OWNER_PROFILE_ID) or ProfileState()
        ).opinions
        next_opinions = (
            profiles.get(OWNER_PROFILE_ID) or ProfileState()
        ).opinions
        opinions_changed = next_opinions != current_opinions
        self_model_changed = next_self_model != self._self_model
        strategies_changed = next_strategies != self._strategies
        opinions_written = False
        self_model_written = False
        strategies_written = False
        try:
            if opinions_changed:
                atomic_write_json(
                    self._opinions_path,
                    self._opinion_document(next_opinions),
                )
                opinions_written = True
            if self_model_changed:
                atomic_write_json(
                    self._self_model_path,
                    self._self_model_document(next_self_model),
                )
                self_model_written = True
            if strategies_changed:
                atomic_write_json(
                    self._strategies_path,
                    self._strategy_document(next_strategies),
                )
                strategies_written = True
            atomic_write_json(
                self._path,
                self._document(profiles, conversations, revision, committed_at),
            )
        except Exception:
            if opinions_written:
                atomic_write_json(
                    self._opinions_path,
                    self._opinion_document(current_opinions),
                )
            if self_model_written:
                atomic_write_json(
                    self._self_model_path,
                    self._self_model_document(self._self_model),
                )
            if strategies_written:
                atomic_write_json(
                    self._strategies_path,
                    self._strategy_document(self._strategies),
                )
            raise
        self._profiles = profiles
        self._conversations = conversations
        self._self_model = next_self_model
        self._strategies = next_strategies
        self._revision = revision
        self._committed_at = committed_at
        return True

    def _replace_profile(
        self,
        profile_id: str,
        state: ProfileState,
        *,
        committed_at: float,
        conversations: dict[str, ConversationRecord] | None = None,
    ) -> bool:
        profiles = self._profiles.copy()
        profiles[profile_id] = state
        return self._replace_all(
            profiles,
            self._conversations.copy() if conversations is None else conversations,
            committed_at=committed_at,
        )

    def _normalize_profile_presence(self, profile: ProfileState, *, now: float) -> ProfileState:
        """Return temporal presence truth without mutating or persisting state."""

        presence = normalize_presence(profile.presence, now=now)
        return profile if presence == profile.presence else replace(profile, presence=presence)

    def _load_payload(
        self,
        payload: object,
        *,
        now: float,
    ) -> tuple[
        dict[str, ProfileState],
        dict[str, ConversationRecord],
        int,
        float,
    ]:
        if not isinstance(payload, dict):
            raise ValueError("state document is not an object")
        schema = int(_number(payload.get("schema_version")))
        if schema not in range(0, STATE_SCHEMA_VERSION + 1):
            raise ValueError(f"unsupported state schema {schema}")
        if schema == STATE_SCHEMA_VERSION:
            if (
                set(payload)
                != {
                    "schema_version",
                    "canonical_profile_id",
                    "revision",
                    "committed_at",
                    "profiles",
                    "conversations",
                }
                or type(payload.get("schema_version")) is not int
                or payload.get("schema_version") != STATE_SCHEMA_VERSION
                or payload.get("canonical_profile_id") != OWNER_PROFILE_ID
                or type(payload.get("revision")) is not int
                or int(payload["revision"]) < 0
                or type(payload.get("committed_at")) not in {int, float}
                or not math.isfinite(float(payload["committed_at"]))
                or float(payload["committed_at"]) < 0.0
            ):
                raise ValueError("canonical state header is malformed")
        if schema == 0 and not (
            isinstance(payload.get("user"), dict)
            or any(key in payload for key in ("name", "likes", "dislikes", "facts"))
        ):
            raise ValueError("unrecognized legacy state document")
        profiles: dict[str, ProfileState] = {}
        conversations: dict[str, ConversationRecord] = {}
        raw_profiles = payload.get("profiles")
        if schema == STATE_SCHEMA_VERSION and not isinstance(raw_profiles, dict):
            raise ValueError("canonical state profiles are unavailable")
        if isinstance(raw_profiles, dict):
            for raw_id, raw_profile in raw_profiles.items():
                if schema == STATE_SCHEMA_VERSION and not isinstance(raw_profile, dict):
                    raise ValueError("canonical profile is malformed")
                profile = ProfileState.from_dict(
                    (
                        raw_profile
                        if schema == STATE_SCHEMA_VERSION
                        else _legacy_profile_payload(
                            raw_profile,
                            preserve_structured_memory=schema >= 17,
                        )
                    ),
                    now=now,
                    migrating=schema != STATE_SCHEMA_VERSION,
                )
                if profile is None:
                    continue
                if schema == STATE_SCHEMA_VERSION:
                    _validate_canonical_profile(raw_profile, profile, now=now)
                profile_id = canonical_profile_id(raw_id)
                profiles[profile_id] = (
                    _merge_profiles(profiles[profile_id], profile, now=now)
                    if profile_id in profiles
                    else profile
                )
        elif isinstance(payload.get("user"), dict):
            profiles[OWNER_PROFILE_ID] = self._popup_profile(payload["user"], now=now)
        elif schema == 0:
            profiles[OWNER_PROFILE_ID] = self._popup_profile(payload, now=now)
        raw_conversations = payload.get("conversations")
        if schema == STATE_SCHEMA_VERSION and not isinstance(raw_conversations, dict):
            raise ValueError("canonical state conversations are unavailable")
        if isinstance(raw_conversations, dict):
            for key, raw in raw_conversations.items():
                if schema == STATE_SCHEMA_VERSION and not isinstance(raw, dict):
                    raise ValueError("canonical conversation is malformed")
                record = ConversationRecord.from_dict(str(key), raw)
                if record is None:
                    if schema == STATE_SCHEMA_VERSION:
                        raise ValueError("canonical conversation is malformed")
                    continue
                if schema == STATE_SCHEMA_VERSION:
                    _validate_canonical_conversation(raw, record)
                existing = conversations.get(record.conversation_id)
                conversations[record.conversation_id] = (
                    _merge_conversations(existing, record) if existing else record
                )
        if schema == STATE_SCHEMA_VERSION:
            raw_owner = next(
                (
                    value
                    for key, value in raw_profiles.items()
                    if canonical_profile_id(key) == OWNER_PROFILE_ID
                ),
                None,
            )
            if (
                OWNER_PROFILE_ID not in profiles
                or not isinstance(raw_owner, dict)
                or not set(raw_owner)
                & {
                    "emotion",
                    "mood",
                    "presence",
                    "memories",
                    "interests",
                    "preferences",
                    "opinions",
                    "relationship",
                }
            ):
                raise ValueError("canonical owner profile is unavailable")
        return (
            profiles,
            conversations,
            int(_number(payload.get("revision"))) if schema == STATE_SCHEMA_VERSION else 0,
            max(0.0, _number(payload.get("committed_at"))),
        )

    def _popup_profile(self, payload: dict[str, object], *, now: float) -> ProfileState:
        memories: list[Memory] = []

        def add(text: str, kind: str = "fact") -> None:
            value = compact_text(text, 300)
            if value:
                memories.append(Memory(uuid.uuid4().hex, "user", kind, value, 0.8, now, now))

        name = compact_text(payload.get("name"), 80)
        if name:
            add(f"The user's name is {name}")
        for key, prefix in (("likes", "The user likes"), ("dislikes", "The user dislikes")):
            raw = payload.get(key)
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    if compact_text(item, 120):
                        add(f"{prefix} {compact_text(item, 120)}")
        raw_facts = payload.get("facts")
        if isinstance(raw_facts, (list, tuple)):
            for fact in raw_facts:
                add(str(fact))
        return replace(
            _new_profile(now),
            memories=_merge_memories((), tuple(memories)),
            updated_at=now if memories else 0.0,
        )

    def _read_migration_source(self, path: Path) -> object | None:
        try:
            return read_json(path)
        except FileNotFoundError:
            return None
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Akane state recovery is required for {path}: "
                f"{type(exc).__name__}"
            ) from exc

    def _read_opinion_source(self) -> object | None:
        try:
            return read_json(self._opinions_path)
        except FileNotFoundError:
            return None
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Akane opinion recovery is required for {self._opinions_path}: "
                f"{type(exc).__name__}"
            ) from exc

    def _read_self_model_source(self) -> object | None:
        try:
            return read_json(self._self_model_path)
        except FileNotFoundError:
            return None
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Akane self-model recovery is required for {self._self_model_path}: "
                f"{type(exc).__name__}"
            ) from exc

    def _read_strategy_source(self) -> object | None:
        try:
            return read_json(self._strategies_path)
        except FileNotFoundError:
            return None
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Akane strategy recovery is required for {self._strategies_path}: "
                f"{type(exc).__name__}"
            ) from exc

    def _decode_source(
        self,
        path: Path,
        payload: object,
        *,
        now: float,
    ) -> tuple[
        dict[str, ProfileState],
        dict[str, ConversationRecord],
        int,
        float,
    ]:
        try:
            return self._load_payload(payload, now=now)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Akane state recovery is required for {path}: "
                f"{type(exc).__name__}"
            ) from exc

    def _load(self) -> None:
        now = time.time()
        canonical = self._read_migration_source(self._path)
        migrated = False
        if canonical is not None:
            profiles, conversations, revision, committed_at = self._decode_source(
                self._path,
                canonical,
                now=now,
            )
            schema = int(_number(canonical.get("schema_version"))) if isinstance(canonical, dict) else 0
            migrated = schema != STATE_SCHEMA_VERSION
            self._profiles = profiles
            self._conversations = conversations
            self._revision = revision
            self._committed_at = committed_at
            if schema == STATE_SCHEMA_VERSION:
                raw_profiles = canonical["profiles"]
                raw_conversations = canonical["conversations"]
                for profile_id, raw_profile in raw_profiles.items():
                    raw_presence = raw_profile.get("presence")
                    parsed_presence = profiles[canonical_profile_id(profile_id)].presence.as_dict()
                    raw_initiative = raw_profile.get("initiative")
                    raw_opportunity = (
                        raw_initiative.get("current")
                        if isinstance(raw_initiative, dict)
                        else None
                    )
                    presence_claim_expired = bool(
                        isinstance(raw_presence, dict)
                        and raw_presence.get("claim_token")
                        and _number(raw_presence.get("claim_expires_at")) <= now
                    )
                    initiative_claim_expired = bool(
                        isinstance(raw_opportunity, dict)
                        and raw_opportunity.get("claim_token")
                        and _number(raw_opportunity.get("claim_expires_at")) <= now
                    )
                    self._expired_claim_recoveries += int(
                        presence_claim_expired
                    ) + int(initiative_claim_expired)
                    migrated = migrated or (
                        str(profile_id) != canonical_profile_id(profile_id)
                        or isinstance(raw_presence, dict)
                        and raw_presence != parsed_presence
                        or isinstance(raw_profile.get("mood"), dict)
                        and _number(raw_profile["mood"].get("updated_at")) > now
                        or isinstance(raw_profile.get("emotion"), dict)
                        and any(
                            _number(raw_profile["emotion"].get(field_name)) > now
                            for field_name in ("started_at", "updated_at")
                        )
                        or initiative_claim_expired
                    )
                migrated = migrated or set(raw_conversations) != set(conversations)
                migrated = migrated or any(
                    raw.get("profile_id")
                    != canonical_profile_id(raw.get("profile_id"))
                    for raw in raw_conversations.values()
                )
        if self._default_path and (
            canonical is None
            or not isinstance(canonical, dict)
            or int(_number(canonical.get("schema_version"))) != STATE_SCHEMA_VERSION
        ):
            for source in (MEMORY_PATH, POPUP_USER_PATH):
                if source == self._path:
                    continue
                payload = self._read_migration_source(source)
                if payload is None:
                    continue
                profiles, conversations, _revision, _committed = self._decode_source(
                    source,
                    payload,
                    now=now,
                )
                for profile_id, profile in profiles.items():
                    self._profiles[profile_id] = (
                        _merge_profiles(
                            self._profiles[profile_id],
                            profile,
                            now=now,
                        )
                        if profile_id in self._profiles
                        else profile
                    )
                for conversation_id, conversation in conversations.items():
                    existing = self._conversations.get(conversation_id)
                    self._conversations[conversation_id] = (
                        _merge_conversations(existing, conversation)
                        if existing
                        else conversation
                    )
                migrated = True

        if OWNER_PROFILE_ID not in self._profiles:
            self._profiles[OWNER_PROFILE_ID] = _new_profile(now)
            migrated = True
        normalized: dict[str, ProfileState] = {}
        for profile_id, profile in self._profiles.items():
            presence = profile.presence
            if presence.claim_token is not None or presence.claim_expires_at:
                presence = replace(presence, claim_token=None, claim_expires_at=0.0)
                migrated = True
            normalized[canonical_profile_id(profile_id)] = replace(profile, presence=presence)
        self._profiles = normalized
        opinion_payload = self._read_opinion_source()
        opinions_migrated = opinion_payload is None
        if opinion_payload is not None:
            opinions, opinions_migrated = self._decode_opinion_source(opinion_payload)
            self._profiles[OWNER_PROFILE_ID] = replace(
                self._profiles[OWNER_PROFILE_ID],
                opinions=opinions,
            )
        else:
            owner = self._profiles[OWNER_PROFILE_ID]
            self._profiles[OWNER_PROFILE_ID] = replace(
                owner,
                opinions=self._migrate_opinion_provenance(owner.opinions),
            )
        self_model_payload = self._read_self_model_source()
        if self_model_payload is None:
            self._self_model = SelfModelState()
            atomic_write_json(
                self._self_model_path,
                self._self_model_document(self._self_model),
            )
        else:
            self._self_model = self._decode_self_model_source(self_model_payload)
        strategy_payload = self._read_strategy_source()
        if strategy_payload is None:
            self._strategies = ()
            atomic_write_json(
                self._strategies_path,
                self._strategy_document(self._strategies),
            )
        else:
            loaded_strategies = self._decode_strategy_source(strategy_payload)
            self._strategies = self._repair_stale_strategies(
                loaded_strategies,
                self._self_model,
                now=now,
            )
            if self._strategies != loaded_strategies:
                atomic_write_json(
                    self._strategies_path,
                    self._strategy_document(self._strategies),
                )
        owner = self._profiles.get(OWNER_PROFILE_ID)
        if (owner is not None and owner.presence.last_error and owner.presence.retry_at > now):
            self._presence_failure_count = 1
        cleaned_conversations = {
            key: _clean_conversation_output(record)
            for key, record in self._conversations.items()
        }
        migrated = cleaned_conversations != self._conversations or migrated
        self._conversations = cleaned_conversations
        migrated = self._prune_conversations(now) or migrated
        if opinions_migrated:
            atomic_write_json(
                self._opinions_path,
                self._opinion_document(self._profiles[OWNER_PROFILE_ID].opinions),
            )
        if canonical is None or migrated:
            atomic_write_json(
                self._path,
                self._document(
                    self._profiles,
                    self._conversations,
                    max(1, self._revision + 1),
                    now,
                ),
            )
            self._revision = max(1, self._revision + 1)
            self._committed_at = now

    def expired_claim_recoveries(self) -> int:
        with self._lock:
            return self._expired_claim_recoveries

    def _prune_conversations(self, now: float) -> bool:
        before = self._conversations
        cutoff = now - CONVERSATION_STALE_DAYS * 24.0 * 3600.0
        kept = {
            key: value
            for key, value in self._conversations.items()
            if not value.updated_at or value.updated_at >= cutoff
        }
        if len(kept) > MAX_CONVERSATIONS:
            newest = sorted(
                kept.values(),
                key=lambda item: item.updated_at,
                reverse=True,
            )[:MAX_CONVERSATIONS]
            kept = {item.conversation_id: item for item in newest}
        self._conversations = kept
        return kept != before

    def _conversation(self, profile_id: str, conversation_id: str) -> ConversationRecord:
        current = self._conversations.get(conversation_id)
        if current is not None and current.profile_id == profile_id:
            return current
        return ConversationRecord(conversation_id, profile_id)

    def ensure_profile(self, profile_id: str) -> None:
        """Create one canonical empty profile without borrowing owner state."""

        profile = canonical_profile_id(profile_id)
        current = time.time()
        with self._lock:
            if profile in self._profiles:
                return
            profiles = self._profiles.copy()
            profiles[profile] = _new_profile(current)
            self._replace_all(profiles, self._conversations.copy(), committed_at=current)

    def profile_exists(self, profile_id: str) -> bool:
        profile = canonical_profile_id(profile_id)
        with self._lock:
            return profile in self._profiles

    def profile_ids(self, *, prefix: str = "") -> tuple[str, ...]:
        with self._lock:
            return tuple(
                profile_id
                for profile_id in self._profiles
                if not prefix or profile_id.startswith(prefix)
            )

    def snapshot(
        self,
        profile_id: str = OWNER_PROFILE_ID,
        conversation_id: str = "popup:default",
        query: str = "",
        now: float | None = None,
        include_memory: bool = True,
    ) -> StateSnapshot:
        profile_key = canonical_profile_id(profile_id)
        conversation_key = compact_text(conversation_id, 160) or "popup:default"
        current = time.time() if now is None else max(0.0, float(now))
        with self._lock:
            profile = self._profiles.get(profile_key) or _new_profile(current)
            profile = self._normalize_profile_presence(profile, now=current)
            conversation = self._conversation(profile_key, conversation_key)
            last_profile_assistant_at = max(
                (
                    record.last_assistant_at
                    for record in self._conversations.values()
                    if record.profile_id == profile_key
                ),
                default=0.0,
            )
            last_profile_user_at = max(
                (
                    record.last_user_at
                    for record in self._conversations.values()
                    if record.profile_id == profile_key
                ),
                default=0.0,
            )
            last_profile_initiative = max(
                (
                    turn
                    for record in self._conversations.values()
                    if record.profile_id == profile_key
                    for turn in record.recent_turns
                    if turn.role == "assistant" and turn.source == "initiative"
                ),
                key=lambda item: item.timestamp,
                default=None,
            )
            (
                recalled,
                relevant_opinions,
                relevant_preferences,
                relevant_interests,
                relevant_relationship,
            ) = _relevant_profile_state(profile, query, current)
            if not include_memory:
                recalled = ()
            self_model = (
                self._self_model
                if profile_key == OWNER_PROFILE_ID
                else SelfModelState()
            )
            strategies = (
                self._strategies if profile_key == OWNER_PROFILE_ID else ()
            )
            return StateSnapshot(
                profile_key,
                conversation_key,
                self._revision,
                profile,
                conversation,
                _complete_turns(conversation.recent_turns),
                recalled,
                current,
                last_profile_assistant_at,
                last_profile_user_at,
                last_profile_initiative,
                relevant_opinions=relevant_opinions,
                relevant_preferences=relevant_preferences,
                relevant_interests=relevant_interests,
                relevant_relationship=relevant_relationship,
                familiarity=_familiarity_context(
                    profile,
                    self._conversations,
                    profile_key,
                ),
                self_model=self_model,
                relevant_self_model=_relevant_self_model(
                    self_model,
                    query,
                    current,
                ),
                strategies=strategies,
                relevant_strategies=_relevant_strategies(
                    strategies,
                    self_model,
                    query,
                ),
            )

    def _validated_memory(
        self,
        payload: object,
        *,
        user_text: str,
        assistant_text: str,
        now: float,
        source_type: str = "explicit_user",
        source_id: str | None = None,
        reason: str = "",
        correction: bool = False,
    ) -> Memory | None:
        if not isinstance(payload, dict) or set(payload) != {
            "subject", "kind", "text", "confidence",
        }:
            return None
        subject = compact_text(payload.get("subject"), 16).casefold()
        kind = compact_text(payload.get("kind"), 24).casefold()
        text = compact_text(payload.get("text"), 360)
        confidence = payload.get("confidence")
        if (
            subject not in _MEMORY_SUBJECTS
            or kind not in _MEMORY_KINDS
            or not text
            or not _memory_ownership_matches(subject, text)
            or type(confidence) not in {int, float}
            or not math.isfinite(float(confidence))
            or text.rstrip().endswith("?")
            or len(words(text)) < 3
            or len(_terms(text)) < (4 if kind == "event" else 1)
        ):
            return None
        evidence = (user_text if subject == "user" else f"{user_text} {assistant_text}")
        correction_grounded = (correction and bool(_terms(text) & _terms(user_text)))
        if (
            not (_grounded(text, evidence) or correction_grounded)
            or subject != "user"
            and not (_terms(text) & _terms(user_text))
            or subject == "akane"
            and not _grounded(text, assistant_text)
        ):
            return None
        certainty = max(0.0, min(1.0, float(confidence)))
        return Memory(
            uuid.uuid4().hex,
            subject,
            kind,
            text,
            certainty,
            now,
            now,
            source_type,
            source_id,
            compact_text(reason, 240),
        )

    @staticmethod
    def _memory_target(
        memories: tuple[Memory, ...],
        *,
        target_id: object,
        candidate: Memory | None,
        user_text: str,
    ) -> tuple[int | None, str]:
        target = compact_text(target_id, 100)
        if target:
            matches = [index for index, memory in enumerate(memories) if memory.id == target]
            if len(matches) != 1:
                return None, "target not found"
            existing = memories[matches[0]]
            if candidate is not None:
                if (
                    candidate.subject != existing.subject
                    or _similar(candidate.text, existing.text) < 0.28
                ):
                    return None, "target conflicts with the corrected fact"
            return matches[0], ""

        scored = sorted(
            (
                (
                    _similar(
                        memory.text,
                        candidate.text if candidate is not None else user_text,
                    ),
                    index,
                )
                for index, memory in enumerate(memories)
                if (
                    memory.subject == candidate.subject
                    if candidate is not None
                    else memory.subject in {"user", "shared"}
                )
            ),
            reverse=True,
        )
        if not scored or scored[0][0] < 0.42:
            return None, "target not resolved"
        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.12:
            return None, "ambiguous semantic target"
        return scored[0][1], ""

    @staticmethod
    def _communication_value(key: str, value: object) -> str:
        text = compact_text(value, 80)
        normalized = text.casefold()
        if key in _COMMUNICATION_VALUES:
            return normalized if normalized in _COMMUNICATION_VALUES[key] else ""
        if key in _COMMUNICATION_FREE_TEXT_KEYS and (
            text
            and "\n" not in text
            and "\r" not in text
            and re.fullmatch(r"[\w .,'’'-]{1,80}", text, re.UNICODE)
        ):
            if key == "preferred_name" and (
                len(words(text)) > 4
                or bool(words(text) & _UNSAFE_PREFERENCE_WORDS)
            ):
                return ""
            return text
        return ""

    def _apply_memory_operations(
        self,
        memories: tuple[Memory, ...],
        operations: object,
        *,
        user_text: str,
        assistant_text: str,
        source_id: str | None,
        now: float,
    ) -> tuple[tuple[Memory, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        if not isinstance(operations, list):
            return memories, (), (), ()
        result = list(memories)
        accepted: list[str] = []
        rejected: list[str] = []
        ownership: list[str] = []
        for index, item in enumerate(operations[:8]):
            label = f"memory_ops[{index}]"
            if not isinstance(item, dict):
                rejected.append(f"{label}: malformed operation")
                continue
            op = compact_text(item.get("op"), 16).casefold()
            reason = compact_text(item.get("reason"), 240)
            if not reason:
                rejected.append(f"{label}: empty reason")
                continue
            if op == "remove":
                if set(item) != {"op", "target_id", "reason"}:
                    rejected.append(f"{label}: malformed remove")
                    continue
                target, error = self._memory_target(
                    tuple(result),
                    target_id=item.get("target_id"),
                    candidate=None,
                    user_text=user_text,
                )
                if target is None:
                    rejected.append(f"{label}: {error}")
                    continue
                removed = result[target]
                if removed.subject not in {"user", "shared"}:
                    rejected.append(f"{label}: conflicting ownership")
                    continue
                result.pop(target)
                accepted.append(f"remove:{removed.id}")
                ownership.append(f"{label}:user-owned memory")
                continue
            required = {"op", "target_id", "subject", "kind", "text", "reason", "confidence"}
            if op not in {"add", "revise", "correct"} or set(item) != required:
                rejected.append(f"{label}: malformed or unknown operation")
                continue
            proposed_subject = compact_text(item.get("subject"), 16).casefold()
            proposed_text = compact_text(item.get("text"), 360)
            if not _memory_ownership_matches(proposed_subject, proposed_text):
                rejected.append(f"{label}: conflicting ownership")
                continue
            candidate = self._validated_memory(
                {
                    "subject": item.get("subject"),
                    "kind": item.get("kind"),
                    "text": item.get("text"),
                    "confidence": item.get("confidence"),
                },
                user_text=user_text,
                assistant_text=assistant_text,
                now=now,
                source_type="explicit_user",
                source_id=source_id,
                reason=reason,
                correction=op in {"revise", "correct"},
            )
            if candidate is None:
                rejected.append(f"{label}: ungrounded or invalid memory")
                continue
            if candidate.subject != "user":
                rejected.append(f"{label}: generated claim is not an explicit user fact")
                continue
            if op == "add":
                if item.get("target_id") is not None:
                    rejected.append(f"{label}: add target must be null")
                    continue
                duplicate = next(
                    (
                        position
                        for position, memory in enumerate(result)
                        if memory.subject == candidate.subject
                        and _duplicate_text(memory.text, candidate.text)
                    ),
                    None,
                )
                if duplicate is not None:
                    existing = result[duplicate]
                    result[duplicate] = replace(
                        existing,
                        confidence=max(existing.confidence, candidate.confidence),
                        updated_at=now,
                        source_type=candidate.source_type,
                        source_id=candidate.source_id,
                        reason=candidate.reason,
                    )
                    accepted.append(f"update:{existing.id}")
                    ownership.append(f"{label}:user-owned memory")
                    continue
                result.append(candidate)
                accepted.append(f"add:{candidate.id}")
            else:
                target, error = self._memory_target(
                    tuple(result),
                    target_id=item.get("target_id"),
                    candidate=candidate,
                    user_text=user_text,
                )
                if target is None:
                    rejected.append(f"{label}: {error}")
                    continue
                existing = result[target]
                if existing.subject != "user":
                    rejected.append(f"{label}: conflicting ownership")
                    continue
                revised = replace(
                    candidate,
                    id=existing.id,
                    confidence=max(0.9, candidate.confidence),
                    created_at=existing.created_at,
                )
                if revised.text == existing.text and revised.kind == existing.kind:
                    rejected.append(f"{label}: no durable change")
                    continue
                result[target] = revised
                accepted.append(f"{op}:{existing.id}")
            ownership.append(f"{label}:user-owned memory")
        return (
            tuple(result[-MEMORY_MAX_ENTRIES_PER_PROFILE:]),
            tuple(accepted),
            tuple(rejected),
            tuple(ownership),
        )

    def _apply_communication_operations(
        self,
        current: tuple[CommunicationPreference, ...],
        operations: object,
        *,
        user_text: str,
        now: float,
    ) -> tuple[tuple[CommunicationPreference, ...], tuple[str, ...], tuple[str, ...]]:
        if not isinstance(operations, list):
            return current, (), ()
        result = list(current)
        rejected: list[str] = []
        ownership: list[str] = []
        for index, item in enumerate(operations[:8]):
            label = f"communication_ops[{index}]"
            if not isinstance(item, dict) or set(item) != {"op", "key", "value", "reason"}:
                rejected.append(f"{label}: malformed operation")
                continue
            op = compact_text(item.get("op"), 16).casefold()
            key = compact_text(item.get("key"), 40).casefold()
            value = self._communication_value(key, item.get("value"))
            reason = compact_text(item.get("reason"), 240)
            if op not in {"set", "revise", "remove"}:
                rejected.append(f"{label}: unknown operation")
                continue
            if not value or not reason:
                rejected.append(f"{label}: unsupported key/value or empty reason")
                continue
            if not _exact_value_grounded(value, user_text):
                rejected.append(f"{label}: exact value is not grounded in the user message")
                continue
            match = next(
                (
                    position
                    for position, existing in enumerate(result)
                    if existing.key == key
                    and (key != "forbidden_phrase" or _key(existing.value) == _key(value))
                ),
                None,
            )
            if op == "remove":
                if match is None:
                    rejected.append(f"{label}: target not found")
                    continue
                result.pop(match)
            else:
                if (
                    key == "forbidden_phrase"
                    and match is None
                    and sum(
                        existing.key == "forbidden_phrase" for existing in result
                    ) >= _MAX_FORBIDDEN_PHRASES
                ):
                    rejected.append(f"{label}: forbidden phrase limit reached")
                    continue
                if op == "revise" and match is None and key == "forbidden_phrase":
                    rejected.append(f"{label}: target not found")
                    continue
                created = result[match].created_at if match is not None else now
                candidate = CommunicationPreference(
                    key,
                    value,
                    reason,
                    created,
                    now,
                    "explicit_user",
                )
                if match is not None and result[match].value == value:
                    rejected.append(f"{label}: no durable change")
                    continue
                if match is None:
                    result.append(candidate)
                else:
                    result[match] = candidate
            ownership.append(f"{label}:profile-scoped communication")
        return (tuple(result[-_MAX_COMMUNICATION_PREFERENCES:]), tuple(rejected), tuple(ownership))

    @staticmethod
    def _assistant_behavior_evidence(
        conversation: ConversationRecord,
        *,
        assistant_text: str,
        source_id: str,
    ) -> tuple[tuple[str, str], ...]:
        evidence = [
            (turn.turn_id, turn.content)
            for turn in conversation.recent_turns
            if turn.role == "assistant" and turn.source != "initiative"
        ]
        if assistant_text and source_id:
            evidence.append((f"{source_id}:assistant", assistant_text))
        return tuple(evidence[-6:])

    def _self_behavior_sources(
        self,
        *,
        category: str,
        description: str,
        reason: str,
        conversation: ConversationRecord,
        profile: ProfileState,
        user_text: str,
        assistant_text: str,
        source_id: str,
        conflicting: bool = False,
    ) -> tuple[str, ...]:
        evidence = self._assistant_behavior_evidence(
            conversation,
            assistant_text=assistant_text,
            source_id=source_id,
        )
        sources: list[str] = []
        if _QUESTION_BEHAVIOR.search(f"{description} {reason}"):
            if len(evidence) < 3:
                return ()
            question_evidence = tuple(
                item for item in evidence if item[1].rstrip().endswith("?")
            )
            rate = len(question_evidence) / len(evidence)
            selected = (
                tuple(item for item in evidence if not item[1].rstrip().endswith("?"))
                if conflicting and rate <= 0.35
                else question_evidence
                if not conflicting and rate >= 0.60
                else ()
            )
            sources.extend(item[0] for item in selected)
        elif not conflicting:
            sources.extend(
                evidence_id
                for evidence_id, text in evidence
                if _grounded(reason, text) or _grounded(description, text)
            )

        if not conflicting:
            sources.extend(
                memory.id
                for memory in profile.memories
                if memory.subject == "akane"
                and (
                    _grounded(reason, memory.text)
                    or _grounded(description, memory.text)
                )
            )
            sources.extend(
                opinion.id
                for opinion in profile.opinions
                if opinion.domain == "self"
                and (
                    _grounded(reason, opinion.content)
                    or _grounded(description, opinion.content)
                )
            )
        actual_sources = tuple(dict.fromkeys(sources))[-8:]
        if (
            len(actual_sources) >= 2
            and source_id
            and _USER_SELF_FEEDBACK.search(user_text)
            and _grounded(reason, user_text)
        ):
            return tuple(
                dict.fromkeys((*actual_sources, f"{source_id}:user"))
            )[-8:]
        return actual_sources

    @staticmethod
    def _strategy_addresses_goal(
        description: str,
        reason: str,
        goal: ImprovementTarget,
    ) -> bool:
        goal_terms = _terms(goal.content)
        return bool(
            _terms(description) & goal_terms
            and _grounded(reason, goal.content)
        )

    def _apply_strategy_operations(
        self,
        current: tuple[Strategy, ...],
        operations: object,
        *,
        self_model: SelfModelState,
        assistant_text: str,
        source_id: str,
        now: float,
    ) -> tuple[tuple[Strategy, ...], tuple[str, ...], tuple[str, ...]]:
        strategies = list(current)
        rejected: list[str] = []
        ownership: list[str] = []
        goals = {goal.id: goal for goal in self_model.improvement_targets}
        if not isinstance(operations, list):
            return current, (), ()
        for index, raw in enumerate(operations[:4]):
            label = f"strategy_ops[{index}]"
            if not isinstance(raw, dict):
                rejected.append(f"{label}: malformed operation")
                continue
            op = compact_text(raw.get("op"), 16).casefold()
            if op == "abandon":
                if set(raw) != {"op", "target_id", "reason"}:
                    rejected.append(f"{label}: malformed abandon")
                    continue
                target_id = compact_text(raw.get("target_id"), 100)
                reason = compact_text(raw.get("reason"), 240)
                match = next(
                    (
                        position
                        for position, strategy in enumerate(strategies)
                        if strategy.id == target_id and strategy.status == "active"
                    ),
                    None,
                )
                if (
                    match is None
                    or not reason
                    or not _grounded(reason, assistant_text)
                    or not _AKANE_STRATEGY_CLAIM.search(assistant_text)
                ):
                    rejected.append(f"{label}: missing target or ungrounded reason")
                    continue
                strategies[match] = replace(
                    strategies[match],
                    status="abandoned",
                    updated_at=now,
                    evidence_summary=reason,
                    last_evaluation_result="abandoned",
                    revision_count=strategies[match].revision_count + 1,
                )
                ownership.append(f"{label}:Akane-owned behavioral strategy")
                continue

            required = {
                "op",
                "target_id",
                "goal_id",
                "description",
                "reason",
                "confidence",
            }
            if op not in {"create", "revise"} or set(raw) != required:
                rejected.append(f"{label}: malformed or unknown operation")
                continue
            target_id = compact_text(raw.get("target_id"), 100)
            goal_id = compact_text(raw.get("goal_id"), 100)
            description = compact_text(raw.get("description"), 240)
            reason = compact_text(raw.get("reason"), 240)
            confidence = raw.get("confidence")
            goal = goals.get(goal_id)
            if (
                goal is None
                or not description
                or not reason
                or type(confidence) not in {int, float}
                or not math.isfinite(float(confidence))
                or not 0.35 <= float(confidence) <= 0.90
                or not _AKANE_STRATEGY_CLAIM.search(assistant_text)
                or not _grounded(description, assistant_text)
                or not _grounded(reason, assistant_text)
                or not _ACTIONABLE_STRATEGY.search(description)
                or _BROAD_STRATEGY.search(description)
                or _FOUNDATIONAL_STRATEGY.search(f"{description} {reason}")
                or not self._strategy_addresses_goal(description, reason, goal)
            ):
                rejected.append(f"{label}: invalid, broad, or ungrounded strategy")
                continue

            match = next(
                (
                    position
                    for position, strategy in enumerate(strategies)
                    if strategy.id == target_id
                ),
                None,
            )
            if op == "create":
                if (
                    target_id
                    or sum(item.status == "active" for item in strategies)
                    >= _MAX_ACTIVE_STRATEGIES
                    or any(
                        item.status == "active" and item.goal_id == goal_id
                        for item in strategies
                    )
                    or any(
                        _similar(item.description, description) >= 0.68
                        for item in strategies
                    )
                ):
                    rejected.append(f"{label}: duplicate or strategy bound reached")
                    continue
                if len(strategies) >= _MAX_STRATEGIES:
                    oldest_inactive = min(
                        (
                            position
                            for position, item in enumerate(strategies)
                            if item.status != "active"
                        ),
                        key=lambda position: (
                            strategies[position].updated_at,
                            strategies[position].id,
                        ),
                        default=None,
                    )
                    if oldest_inactive is None:
                        rejected.append(f"{label}: strategy bound reached")
                        continue
                    strategies.pop(oldest_inactive)
                strategies.append(
                    Strategy(
                        "strategy_" + uuid.uuid4().hex,
                        goal_id,
                        description,
                        "active",
                        float(confidence),
                        (goal_id, f"{source_id}:assistant"),
                        now,
                        now,
                    )
                )
            else:
                if match is None or strategies[match].status != "active":
                    rejected.append(f"{label}: active target not found")
                    continue
                existing = strategies[match]
                if (
                    existing.goal_id != goal_id
                    or existing.description == description
                ):
                    rejected.append(f"{label}: no meaningful goal-preserving revision")
                    continue
                strategies[match] = replace(
                    existing,
                    description=description,
                    confidence=float(confidence),
                    source_ids=tuple(
                        dict.fromkeys((*existing.source_ids, f"{source_id}:assistant"))
                    )[-12:],
                    updated_at=now,
                    opportunity_count=0,
                    success_count=0,
                    failure_count=0,
                    evidence_summary=reason,
                    last_evaluation_result="insufficient_evidence",
                    revision_count=existing.revision_count + 1,
                )
            ownership.append(f"{label}:Akane-owned behavioral strategy")
        return tuple(strategies), tuple(rejected), tuple(ownership)

    @staticmethod
    def _evaluate_strategies(
        strategies: tuple[Strategy, ...],
        prior_strategies: tuple[Strategy, ...],
        self_model: SelfModelState,
        *,
        user_text: str,
        assistant_text: str,
        source_id: str,
        now: float,
    ) -> tuple[tuple[Strategy, ...], SelfModelState]:
        prior = {strategy.id: strategy for strategy in prior_strategies}
        goals = {goal.id: goal for goal in self_model.improvement_targets}
        limitations = {item.id: item for item in self_model.limitations}
        result = list(strategies)
        next_self_model = self_model
        for index, strategy in enumerate(tuple(result)):
            previous = prior.get(strategy.id)
            goal = goals.get(strategy.goal_id)
            if (
                previous is None
                or previous.status != "active"
                or strategy.status != "active"
                or strategy.revision_count != previous.revision_count
                or goal is None
                or _strategy_metric_kind(strategy, goal)
                != "clarification_directness"
                or not _strategy_applies(strategy, goal, user_text)
            ):
                continue

            negative_feedback = bool(_STRATEGY_NEGATIVE_FEEDBACK.search(user_text))
            positive_feedback = bool(_STRATEGY_POSITIVE_FEEDBACK.search(user_text))
            failed = assistant_text.rstrip().endswith("?") or negative_feedback
            opportunities = strategy.opportunity_count + 1
            successes = strategy.success_count + int(not failed)
            failures = strategy.failure_count + int(failed)
            evidence_sources = [*strategy.source_ids, f"{source_id}:assistant"]
            if negative_feedback or positive_feedback:
                evidence_sources.append(f"{source_id}:user")
            if opportunities < _STRATEGY_EVALUATION_WINDOW:
                result[index] = replace(
                    strategy,
                    source_ids=tuple(dict.fromkeys(evidence_sources))[-12:],
                    updated_at=now,
                    opportunity_count=opportunities,
                    success_count=successes,
                    failure_count=failures,
                    evidence_summary=(
                        f"{opportunities} of {_STRATEGY_EVALUATION_WINDOW} "
                        "relevant opportunities observed."
                    ),
                    last_evaluation_result=strategy.last_evaluation_result,
                )
                continue

            ratio = successes / opportunities
            evaluation_result = (
                "improving" if ratio >= 0.75 else "worsening" if ratio <= 0.25 else "unchanged"
            )
            evaluation_id = f"strategy:{strategy.id}:evaluation:{strategy.evaluation_count + 1}"
            confidence = strategy.confidence
            status = "active"
            if evaluation_result == "improving":
                confidence = min(0.95, confidence + 0.08)
            elif evaluation_result == "worsening":
                confidence = max(0.20, confidence - 0.12)
                if strategy.last_evaluation_result == "worsening" or confidence < 0.45:
                    status = "abandoned"
            else:
                confidence = max(0.30, confidence - 0.02)

            evaluation_count = strategy.evaluation_count + 1
            completed = (
                evaluation_result == "improving"
                and strategy.last_evaluation_result == "improving"
                and evaluation_count >= 2
            )
            if evaluation_result == "improving":
                limitation = next(
                    (
                        limitations[source]
                        for source in goal.source_ids
                        if source in limitations
                    ),
                    None,
                )
                if limitation is not None:
                    revised_limitation = replace(
                        limitation,
                        confidence=max(0.0, limitation.confidence - 0.10),
                        source_ids=tuple(
                            dict.fromkeys((*limitation.source_ids, evaluation_id))
                        )[-8:],
                        updated_at=now,
                        revision_count=limitation.revision_count + 1,
                    )
                    next_self_model = replace(
                        next_self_model,
                        limitations=tuple(
                            revised_limitation if item.id == limitation.id else item
                            for item in next_self_model.limitations
                        ),
                    )
                    limitations[limitation.id] = revised_limitation
                    if completed:
                        next_self_model = replace(
                            next_self_model,
                            limitations=tuple(
                                item
                                for item in next_self_model.limitations
                                if item.id != limitation.id
                            ),
                            improvement_targets=tuple(
                                item
                                for item in next_self_model.improvement_targets
                                if item.id != goal.id
                            ),
                        )
                        status = "completed"
                        evaluation_result = "completed"

            result[index] = replace(
                strategy,
                status=status,
                confidence=confidence,
                source_ids=tuple(
                    dict.fromkeys((*evidence_sources, evaluation_id))
                )[-12:],
                updated_at=now,
                evaluation_count=evaluation_count,
                opportunity_count=0,
                success_count=0,
                failure_count=0,
                evidence_summary=(
                    f"{successes} of {opportunities} relevant replies met the strategy signal."
                ),
                last_evaluation_result=(
                    "abandoned" if status == "abandoned" else evaluation_result
                ),
                revision_count=strategy.revision_count + 1,
            )

        repaired = StateStore._repair_stale_strategies(
            tuple(result),
            next_self_model,
            now=now,
        )
        return repaired, next_self_model

    def _apply_self_model_operations(
        self,
        current: SelfModelState,
        self_operations: object,
        improvement_operations: object,
        *,
        profile: ProfileState,
        conversation: ConversationRecord,
        user_text: str,
        assistant_text: str,
        source_id: str,
        now: float,
    ) -> tuple[SelfModelState, tuple[str, ...], tuple[str, ...]]:
        state = current
        rejected: list[str] = []
        ownership: list[str] = []
        if isinstance(self_operations, list):
            for index, raw in enumerate(self_operations[:6]):
                label = f"self_model_ops[{index}]"
                if not isinstance(raw, dict):
                    rejected.append(f"{label}: malformed operation")
                    continue
                op = compact_text(raw.get("op"), 16).casefold()
                if op == "resolve":
                    if set(raw) != {"op", "target_id", "reason"}:
                        rejected.append(f"{label}: malformed resolve")
                        continue
                    target_id = compact_text(raw.get("target_id"), 100)
                    reason = compact_text(raw.get("reason"), 240)
                    target = next(
                        (item for item in state.items if item.id == target_id),
                        None,
                    )
                    if target is None or not _grounded(reason, assistant_text):
                        rejected.append(f"{label}: missing target or ungrounded reason")
                        continue
                    if target.category == "capability":
                        rejected.append(f"{label}: implemented capability is still authoritative")
                        continue
                    if any(
                        source.startswith("runtime:")
                        for source in target.source_ids
                    ):
                        rejected.append(
                            f"{label}: runtime limitation is still authoritative"
                        )
                        continue
                    conflict_sources = self._self_behavior_sources(
                        category=target.category,
                        description=target.description,
                        reason=reason,
                        conversation=conversation,
                        profile=profile,
                        user_text=user_text,
                        assistant_text=assistant_text,
                        source_id=source_id,
                        conflicting=True,
                    )
                    if len(conflict_sources) < 2:
                        rejected.append(f"{label}: insufficient conflicting evidence")
                        continue
                    remaining = tuple(
                        item
                        for item in state.category_items(target.category)
                        if item.id != target.id
                    )
                    state = state.replace_category(target.category, remaining)
                    state = replace(
                        state,
                        improvement_targets=tuple(
                            improvement
                            for improvement in state.improvement_targets
                            if target.id not in improvement.source_ids
                        ),
                    )
                    ownership.append(f"{label}:Akane-owned self-model")
                    continue

                required = {
                    "op",
                    "target_id",
                    "category",
                    "area",
                    "description",
                    "reason",
                    "confidence",
                }
                if (
                    op not in {"create", "update", "reinforce", "weaken"}
                    or set(raw) != required
                ):
                    rejected.append(f"{label}: malformed or unknown operation")
                    continue
                category = compact_text(raw.get("category"), 20).casefold()
                area = compact_text(raw.get("area"), 80).casefold()
                description = compact_text(raw.get("description"), 280)
                reason = compact_text(raw.get("reason"), 240)
                confidence = raw.get("confidence")
                if (
                    category not in {"capability", "limitation", "trait"}
                    or not area
                    or not description
                    or not reason
                    or type(confidence) not in {int, float}
                    or not math.isfinite(float(confidence))
                    or not 0.0 <= float(confidence) <= 1.0
                    or not _AKANE_SELF_CLAIM.search(description)
                    or not _AKANE_SELF_CLAIM.search(assistant_text)
                    or _USER_STATE_CONTAMINATION.search(description)
                    or not _grounded(description, assistant_text)
                    or not _grounded(reason, assistant_text)
                    or _EXTERNAL_FACT_CLAIM.search(f"{description} {reason}")
                ):
                    rejected.append(f"{label}: invalid or ungrounded Akane self-claim")
                    continue

                capability = (
                    _runtime_capability_for(description, area)
                    if category == "capability"
                    else None
                )
                runtime_limitation = (
                    _runtime_limitation_for(description, area)
                    if category == "limitation"
                    else None
                )
                if category == "capability" and capability is None:
                    rejected.append(f"{label}: capability is not implemented")
                    continue
                if runtime_limitation is not None and op == "weaken":
                    rejected.append(f"{label}: runtime limitation is still authoritative")
                    continue
                conflicting = op == "weaken"
                sources = (
                    (capability.source_id,)
                    if capability is not None
                    else (runtime_limitation.source_id,)
                    if runtime_limitation is not None
                    else self._self_behavior_sources(
                        category=category,
                        description=description,
                        reason=reason,
                        conversation=conversation,
                        profile=profile,
                        user_text=user_text,
                        assistant_text=assistant_text,
                        source_id=source_id,
                        conflicting=conflicting,
                    )
                )
                if (
                    category != "capability"
                    and runtime_limitation is None
                    and len(sources) < 2
                ):
                    rejected.append(f"{label}: insufficient behavioral evidence")
                    continue
                if op == "create" and (
                    raw.get("target_id") is not None
                    or float(confidence) < (0.70 if category == "capability" else 0.55)
                ):
                    rejected.append(f"{label}: invalid target or insufficient confidence")
                    continue

                values = list(state.category_items(category))
                match = next(
                    (
                        position
                        for position, item in enumerate(values)
                        if item.id == compact_text(raw.get("target_id"), 100)
                    ),
                    None,
                )
                if category == "limitation" and match is not None:
                    existing_is_runtime = any(
                        source.startswith("runtime:")
                        for source in values[match].source_ids
                    )
                    if existing_is_runtime != (runtime_limitation is not None):
                        rejected.append(f"{label}: target evidence class changed")
                        continue
                if op == "create":
                    if any(
                        _similar(item.description, description) >= 0.68
                        for item in values
                    ):
                        rejected.append(f"{label}: existing self-model item requires update")
                        continue
                    values.append(
                        SelfModelItem(
                            "self_" + uuid.uuid4().hex,
                            category,
                            area,
                            description,
                            float(confidence),
                            sources,
                            now,
                            now,
                        )
                    )
                else:
                    if match is None:
                        rejected.append(f"{label}: target not found")
                        continue
                    existing = values[match]
                    if _similar(existing.area, area) < 0.65:
                        rejected.append(f"{label}: target area changed")
                        continue
                    if op == "reinforce" and (
                        _similar(existing.description, description) < 0.65
                        or float(confidence) <= existing.confidence
                    ):
                        rejected.append(f"{label}: invalid reinforcement")
                        continue
                    if op == "weaken" and (
                        _similar(existing.description, description) < 0.55
                        or float(confidence) >= existing.confidence
                    ):
                        rejected.append(f"{label}: invalid weakening")
                        continue
                    if (
                        existing.description == description
                        and existing.confidence == float(confidence)
                    ):
                        rejected.append(f"{label}: no meaningful change")
                        continue
                    values[match] = replace(
                        existing,
                        area=area,
                        description=description,
                        confidence=float(confidence),
                        source_ids=tuple(
                            dict.fromkeys((*existing.source_ids, *sources))
                        )[-8:],
                        updated_at=now,
                        revision_count=existing.revision_count + 1,
                    )
                state = state.replace_category(
                    category,
                    tuple(values[-_MAX_SELF_MODEL_ITEMS_PER_CATEGORY:]),
                )
                ownership.append(f"{label}:Akane-owned self-model")

        if isinstance(improvement_operations, list):
            for index, raw in enumerate(improvement_operations[:4]):
                label = f"improvement_ops[{index}]"
                if not isinstance(raw, dict):
                    rejected.append(f"{label}: malformed operation")
                    continue
                op = compact_text(raw.get("op"), 16).casefold()
                if op == "resolve":
                    if set(raw) != {"op", "target_id", "reason"}:
                        rejected.append(f"{label}: malformed resolve")
                        continue
                    target_id = compact_text(raw.get("target_id"), 100)
                    reason = compact_text(raw.get("reason"), 240)
                    match = next(
                        (
                            position
                            for position, target in enumerate(state.improvement_targets)
                            if target.id == target_id
                        ),
                        None,
                    )
                    if match is None or not _grounded(reason, assistant_text):
                        rejected.append(f"{label}: missing target or ungrounded reason")
                        continue
                    targets = list(state.improvement_targets)
                    targets.pop(match)
                    state = replace(state, improvement_targets=tuple(targets))
                    ownership.append(f"{label}:Akane-owned improvement target")
                    continue

                required = {
                    "op",
                    "target_id",
                    "area",
                    "description",
                    "reason",
                    "priority",
                }
                if op not in {"create", "update"} or set(raw) != required:
                    rejected.append(f"{label}: malformed or unknown operation")
                    continue
                target_id = compact_text(raw.get("target_id"), 100)
                area = compact_text(raw.get("area"), 80).casefold()
                description = compact_text(raw.get("description"), 240)
                reason = compact_text(raw.get("reason"), 240)
                priority = raw.get("priority")
                if (
                    not target_id
                    or not area
                    or not description
                    or not reason
                    or type(priority) not in {int, float}
                    or not math.isfinite(float(priority))
                    or not 0.0 <= float(priority) <= 1.0
                    or not _AKANE_SELF_CLAIM.search(assistant_text)
                    or not _grounded(description, assistant_text)
                    or not _grounded(reason, assistant_text)
                ):
                    rejected.append(f"{label}: invalid or ungrounded target")
                    continue
                targets = list(state.improvement_targets)
                if op == "create":
                    limitation = next(
                        (item for item in state.limitations if item.id == target_id),
                        None,
                    )
                    if (
                        limitation is None
                        or limitation.confidence < 0.60
                        or _similar(limitation.area, area) < 0.65
                        or not _grounded(reason, limitation.content)
                        or not (
                            _terms(description)
                            & _terms(f"{limitation.content} {reason}")
                        )
                        or len(targets) >= _MAX_IMPROVEMENT_TARGETS
                        or any(
                            _similar(target.description, description) >= 0.68
                            for target in targets
                        )
                    ):
                        rejected.append(f"{label}: target lacks grounded limitation evidence")
                        continue
                    targets.append(
                        ImprovementTarget(
                            "goal_" + uuid.uuid4().hex,
                            area,
                            description,
                            reason,
                            float(priority),
                            tuple(
                                dict.fromkeys(
                                    (*limitation.source_ids[-7:], limitation.id)
                                )
                            )[-8:],
                            now,
                            now,
                        )
                    )
                else:
                    match = next(
                        (
                            position
                            for position, target in enumerate(targets)
                            if target.id == target_id
                        ),
                        None,
                    )
                    if match is None:
                        rejected.append(f"{label}: target not found")
                        continue
                    existing = targets[match]
                    limitation = next(
                        (
                            item
                            for item in state.limitations
                            if item.id in existing.source_ids
                        ),
                        None,
                    )
                    if (
                        limitation is None
                        or _similar(limitation.area, area) < 0.65
                        or not _grounded(reason, limitation.content)
                        or not (
                            _terms(description)
                            & _terms(f"{limitation.content} {reason}")
                        )
                    ):
                        rejected.append(f"{label}: update lost its limitation evidence")
                        continue
                    if (
                        existing.description == description
                        and existing.reason == reason
                        and existing.priority == float(priority)
                    ):
                        rejected.append(f"{label}: no meaningful change")
                        continue
                    targets[match] = replace(
                        existing,
                        area=area,
                        description=description,
                        reason=reason,
                        priority=float(priority),
                        updated_at=now,
                        revision_count=existing.revision_count + 1,
                    )
                state = replace(state, improvement_targets=tuple(targets))
                ownership.append(f"{label}:Akane-owned improvement target")
        return state, tuple(rejected), tuple(ownership)

    def _apply_opinion_operations(
        self,
        current: tuple[Opinion, ...],
        operations: object,
        *,
        user_text: str,
        assistant_text: str,
        now: float,
        source_type: str = "conversation",
        source_ids: tuple[str, ...] = (),
        evidence_summary: str | None = None,
        trusted_history: str = "",
    ) -> tuple[tuple[Opinion, ...], tuple[str, ...], tuple[str, ...]]:
        if not isinstance(operations, list):
            return current, (), ()
        result = list(current)
        rejected: list[str] = []
        ownership: list[str] = []
        for index, item in enumerate(operations[:6]):
            label = f"opinion_ops[{index}]"
            if not isinstance(item, dict):
                rejected.append(f"{label}: malformed operation")
                continue
            op = compact_text(item.get("op"), 16).casefold()
            if op in {"retire", "remove"}:
                if set(item) != {"op", "target_id", "reason"}:
                    rejected.append(f"{label}: malformed retire")
                    continue
                target_id = compact_text(item.get("target_id"), 100)
                reason = compact_text(item.get("reason"), 240)
                match = next(
                    (position for position, value in enumerate(result) if value.id == target_id),
                    None,
                )
                if match is None or not reason:
                    rejected.append(f"{label}: missing target or reason")
                    continue
                if (
                    not _grounded(reason, assistant_text)
                    or not _AKANE_OPINION_ADOPTION.search(assistant_text)
                ):
                    rejected.append(f"{label}: reason is not grounded in the visible reply")
                    continue
                result.pop(match)
                ownership.append(f"{label}:Akane-owned opinion")
                continue
            required = {
                "op",
                "topic",
                "domain",
                "position",
                "reason",
                "confidence",
                "importance",
            }
            if op in {"reinforce", "weaken", "update", "reconsider"}:
                required.add("target_id")
            if op not in {"form", "reinforce", "weaken", "update", "reconsider"} or set(item) != required:
                rejected.append(f"{label}: malformed or unknown operation")
                continue
            topic = compact_text(item.get("topic"), 140)
            domain = compact_text(item.get("domain"), 48).casefold()
            position = compact_text(item.get("position"), 200)
            reason = compact_text(item.get("reason"), 240)
            confidence = item.get("confidence")
            importance = item.get("importance")
            trusted_sources = tuple(
                dict.fromkeys(
                    source
                    for item_source in source_ids
                    if (source := compact_text(item_source, 180))
                )
            )[-8:]
            if (
                not topic
                or not domain
                or len(words(domain)) > 4
                or not position
                or not reason
                or type(confidence) not in {int, float}
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
                or type(importance) not in {int, float}
                or not math.isfinite(float(importance))
                or not 0.0 <= float(importance) <= 1.0
                or not trusted_sources
                or op == "form" and (
                    not _grounded(topic, f"{user_text} {assistant_text}")
                    and len(_terms(topic)) < 2
                )
                or not _grounded(position, assistant_text)
                or not _grounded(reason, assistant_text)
                or not _AKANE_OPINION_ADOPTION.search(assistant_text)
            ):
                rejected.append(f"{label}: visible reply did not ground and adopt the position")
                continue
            if (
                _EXTERNAL_FACT_CLAIM.search(f"{position} {reason}")
                and not _grounded(
                    f"{position} {reason}",
                    f"{user_text} {trusted_history}",
                )
            ):
                rejected.append(f"{label}: external fact is not grounded in trusted state")
                continue
            topic_key = _key(topic)
            if op == "form":
                if not _durable_opinion_form(
                    topic,
                    reason,
                    confidence=float(confidence),
                    importance=float(importance),
                ):
                    rejected.append(f"{label}: below the durable opinion threshold")
                    continue
                if any(_similar(value.topic, topic) >= 0.72 for value in result):
                    rejected.append(f"{label}: existing opinion requires revise")
                    continue
                candidate = Opinion(
                    topic=topic,
                    position=position,
                    reason=reason,
                    updated_at=now,
                    id=uuid.uuid4().hex,
                    topic_key=topic_key,
                    confidence=float(confidence),
                    created_at=now,
                    source_type=source_type,
                    evidence_summary=compact_text(evidence_summary or user_text, 280),
                    source_ids=trusted_sources,
                    domain=domain,
                    importance=float(importance),
                )
                result.append(candidate)
            else:
                target_id = compact_text(item.get("target_id"), 100)
                match = next(
                    (position_index for position_index, value in enumerate(result) if value.id == target_id),
                    None,
                )
                if match is None:
                    rejected.append(f"{label}: target not found")
                    continue
                existing = result[match]
                if _similar(existing.topic, topic) < 0.70:
                    rejected.append(f"{label}: target topic changed")
                    continue
                if op == "reinforce" and (
                    _similar(existing.position, position) < 0.70
                    or float(confidence) <= existing.confidence
                ):
                    rejected.append(f"{label}: reinforcement must preserve and strengthen the stance")
                    continue
                if op == "weaken" and (
                    _similar(existing.position, position) < 0.55
                    or float(confidence) >= existing.confidence
                ):
                    rejected.append(f"{label}: weakening must reduce confidence without inventing a new stance")
                    continue
                if (
                    existing.position == position
                    and existing.reason == reason
                    and existing.confidence == float(confidence)
                    and existing.importance == float(importance)
                    and existing.domain == domain
                ):
                    rejected.append(f"{label}: no durable change")
                    continue
                result[match] = Opinion(
                    topic=topic,
                    position=position,
                    reason=reason,
                    updated_at=now,
                    id=existing.id,
                    topic_key=topic_key,
                    confidence=float(confidence),
                    created_at=existing.created_at,
                    source_type=source_type,
                    evidence_summary=compact_text(evidence_summary or user_text, 280),
                    source_ids=tuple(
                        dict.fromkeys((*existing.source_ids, *trusted_sources))
                    )[-8:],
                    domain=domain,
                    importance=float(importance),
                    revision_count=existing.revision_count + 1,
                )
            ownership.append(f"{label}:Akane-owned opinion")
        return tuple(result[-_MAX_OPINIONS:]), tuple(rejected), tuple(ownership)

    def _apply_interest_operations(
        self,
        current: tuple[Interest, ...],
        operations: object,
        *,
        user_text: str,
        assistant_text: str,
        source_id: str | None,
        now: float,
        source_type: str = "conversation",
    ) -> tuple[tuple[Interest, ...], tuple[str, ...], tuple[str, ...]]:
        if not isinstance(operations, list):
            return current, (), ()
        result = list(current)
        rejected: list[str] = []
        ownership: list[str] = []
        for index, item in enumerate(operations[:6]):
            label = f"interest_ops[{index}]"
            if not isinstance(item, dict):
                rejected.append(f"{label}: malformed operation")
                continue
            op = compact_text(item.get("op"), 16).casefold()
            remove = op == "remove"
            required = {"op", "topic", "reason"} if remove else {
                "op", "topic", "reason", "strength",
            }
            if op not in {"form", "reinforce", "weaken", "update", "remove"} or set(item) != required:
                rejected.append(f"{label}: malformed or unknown operation")
                continue
            topic = compact_text(item.get("topic"), 100)
            reason = compact_text(item.get("reason"), 240)
            strength = item.get("strength")
            if (
                not topic
                or not reason
                or not _grounded(topic, f"{user_text} {assistant_text}")
                or not _grounded(topic, assistant_text)
                or not _grounded(reason, assistant_text)
                or not _AKANE_TASTE_ADOPTION.search(assistant_text)
                or not remove
                and (
                    type(strength) not in {int, float}
                    or not math.isfinite(float(strength))
                    or not 0.0 <= float(strength) <= 1.0
                )
            ):
                rejected.append(f"{label}: visible reply did not ground and adopt the interest")
                continue
            match = next(
                (
                    position
                    for position, existing in enumerate(result)
                    if _similar(str(existing), topic) >= 0.70
                ),
                None,
            )
            if remove:
                if match is None:
                    rejected.append(f"{label}: target not found")
                    continue
                result.pop(match)
                ownership.append(f"{label}:Akane-owned interest")
                continue
            if op == "form" and match is not None:
                rejected.append(f"{label}: existing interest requires an evolution operation")
                continue
            if op != "form" and match is None:
                rejected.append(f"{label}: target not found")
                continue
            prior = result[match] if match is not None else None
            value = float(strength)
            if op == "reinforce" and prior is not None and value <= prior.strength:
                rejected.append(f"{label}: reinforcement must increase strength")
                continue
            if op == "weaken" and prior is not None and value >= prior.strength:
                rejected.append(f"{label}: weakening must reduce strength")
                continue
            source_ids = tuple(
                dict.fromkeys((*((prior.source_ids) if prior else ()), *((source_id,) if source_id else ())))
            )[-6:]
            candidate = Interest(
                topic,
                value,
                reason,
                prior.created_at if prior else now,
                now,
                source_type,
                source_ids,
                (prior.evidence_count + 1) if prior else 1,
            )
            if prior is not None and candidate.as_dict() == prior.as_dict():
                rejected.append(f"{label}: no durable change")
                continue
            if match is None:
                result.append(candidate)
            else:
                result[match] = candidate
            ownership.append(f"{label}:Akane-owned interest")
        return tuple(result[-_MAX_INTERESTS:]), tuple(rejected), tuple(ownership)

    def _apply_proposals(
        self,
        profile: ProfileState,
        proposals: object,
        *,
        user_text: str,
        assistant_text: str,
        source_id: str | None,
        trusted_history: str,
        now: float,
    ) -> tuple[
        ProfileState,
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        values = proposals if isinstance(proposals, dict) else {}
        accepted_memory_operations: tuple[str, ...] = ()
        rejected_operations: tuple[str, ...] = ()
        ownership_classification: tuple[str, ...] = ()
        next_profile = profile

        (
            memories,
            accepted_memory_operations,
            memory_rejections,
            memory_ownership,
        ) = self._apply_memory_operations(
            next_profile.memories,
            values.get("memory_ops"),
            user_text=user_text,
            assistant_text=assistant_text,
            source_id=source_id,
            now=now,
        )
        if memories != next_profile.memories:
            next_profile = replace(next_profile, memories=memories)
        rejected_operations += memory_rejections
        ownership_classification += memory_ownership

        raw_preferences = values.get("preferences")
        if isinstance(raw_preferences, list):
            additions: list[AkanePreference] = []
            for index, item in enumerate(raw_preferences[:6]):
                if not isinstance(item, dict) or set(item) != {"topic", "stance", "reason"}:
                    continue
                candidate = AkanePreference.from_dict({**item, "updated_at": now})
                if (
                    candidate
                    and _grounded(candidate.topic, user_text)
                    and _grounded(candidate.topic, assistant_text)
                    and _grounded(candidate.reason, assistant_text)
                    and _AKANE_TASTE_ADOPTION.search(assistant_text)
                ):
                    additions.append(candidate)
                    ownership_classification += (f"preferences[{index}]:Akane-owned preference",)
                elif candidate:
                    rejected_operations += (
                        f"preferences[{index}]: conflicting ownership; Akane-owned preference was not visibly adopted",
                    )
            if additions:
                next_profile = replace(
                    next_profile,
                    preferences=_merge_preferences(
                        next_profile.preferences,
                        tuple(additions),
                    ),
                )

        interests, interest_rejections, interest_ownership = self._apply_interest_operations(
            next_profile.interests,
            values.get("interest_ops"),
            user_text=user_text,
            assistant_text=assistant_text,
            source_id=source_id,
            now=now,
        )
        if [item.as_dict() for item in interests] != [
            item.as_dict() for item in next_profile.interests
        ]:
            next_profile = replace(next_profile, interests=interests)
        rejected_operations += interest_rejections
        ownership_classification += interest_ownership

        communication_preferences, communication_rejections, communication_ownership = (
            self._apply_communication_operations(
                next_profile.communication_preferences,
                values.get("communication_ops"),
                user_text=user_text,
                now=now,
            )
        )
        if communication_preferences != next_profile.communication_preferences:
            next_profile = replace(
                next_profile,
                communication_preferences=communication_preferences,
            )
        rejected_operations += communication_rejections
        ownership_classification += communication_ownership

        opinions, opinion_rejections, opinion_ownership = self._apply_opinion_operations(
            next_profile.opinions,
            values.get("opinion_ops"),
            user_text=user_text,
            assistant_text=assistant_text,
            now=now,
            source_ids=tuple((source_id,)) if source_id else (),
            trusted_history=trusted_history,
        )
        if opinions != next_profile.opinions:
            next_profile = replace(next_profile, opinions=opinions)
        rejected_operations += opinion_rejections
        ownership_classification += opinion_ownership

        relationship = values.get("relationship")
        if isinstance(relationship, dict):
            additions: dict[str, tuple[RelationshipEntry, ...]] = {}
            for field_name in ("patterns", "shared_context", "unresolved_events"):
                raw = relationship.get(field_name)
                accepted: list[RelationshipEntry] = []
                if isinstance(raw, list):
                    for index, item in enumerate(raw[:6]):
                        if not isinstance(item, dict) or set(item) != {"summary", "confidence"}:
                            continue
                        candidate = RelationshipEntry.from_dict(
                            {
                                **item,
                                "updated_at": now,
                                "evidence_count": 1,
                                "source_ids": [source_id] if source_id else [],
                            }
                        )
                        if (
                            candidate
                            and _shared_grounded(
                                candidate.summary,
                                user_text,
                                assistant_text,
                            )
                            and not (
                                field_name == "shared_context"
                                and _UNVERIFIED_HISTORY.search(user_text)
                                and not _grounded(candidate.summary, trusted_history)
                            )
                            and not (
                                field_name == "patterns"
                                and _SENSITIVE_PATTERN.search(candidate.summary)
                            )
                        ):
                            accepted.append(candidate)
                            ownership_classification += (
                                f"relationship.{field_name}[{index}]:shared relationship evidence",
                            )
                        elif candidate:
                            rejected_operations += (
                                f"relationship.{field_name}[{index}]: conflicting ownership; not shared relationship evidence",
                            )
                additions[field_name] = tuple(accepted)
            current = next_profile.relationship
            merged = RelationshipState(
                _merge_relationship_entries(
                    current.patterns,
                    additions.get("patterns", ()),
                ),
                _merge_relationship_entries(
                    current.shared_context,
                    additions.get("shared_context", ()),
                ),
                _merge_relationship_entries(
                    current.unresolved_events,
                    additions.get("unresolved_events", ()),
                ),
            )
            resolved = relationship.get("resolved_events")
            unresolved = list(merged.unresolved_events)
            if isinstance(resolved, list):
                for index, item in enumerate(resolved[:6]):
                    candidate = RelationshipEntry.from_dict(
                        {**item, "updated_at": now}
                        if isinstance(item, dict)
                        else item
                    )
                    if (
                        candidate
                        and _shared_grounded(
                            candidate.summary,
                            user_text,
                            assistant_text,
                        )
                    ):
                        match = next(
                            (
                                index
                                for index, existing in enumerate(unresolved)
                                if _similar(existing.summary, candidate.summary) >= 0.55
                            ),
                            None,
                        )
                        if match is not None:
                            unresolved.pop(match)
                            ownership_classification += (
                                f"relationship.resolved_events[{index}]:shared relationship evidence",
                            )
                    elif candidate:
                        rejected_operations += (
                            f"relationship.resolved_events[{index}]: conflicting ownership; not shared relationship evidence",
                        )
            merged = replace(merged, unresolved_events=tuple(unresolved))
            if merged != current:
                next_profile = replace(next_profile, relationship=merged)

        if next_profile != profile:
            next_profile = replace(next_profile, updated_at=now)
        return (
            next_profile,
            accepted_memory_operations,
            rejected_operations,
            ownership_classification,
        )

    def commit_turn(
        self,
        snapshot: StateSnapshot,
        *,
        user_text: str,
        assistant_text: str,
        source: str,
        request_id: str = "",
        proposals: object = None,
        proposal_rejections: tuple[str, ...] = (),
        allow_initiative: bool = True,
        now: float | None = None,
    ) -> StateSnapshot:
        committed = time.time() if now is None else max(snapshot.now, float(now))
        assistant_text = clean_visible_output(assistant_text)
        with self._lock:
            profile = self._profiles.get(snapshot.profile_id) or _new_profile(committed)
            profile = self._normalize_profile_presence(profile, now=committed)
            conversation = self._conversation(snapshot.profile_id, snapshot.conversation_id)
            request = compact_text(request_id, 180)
            if request and request in conversation.committed_request_ids:
                return self.snapshot(
                    snapshot.profile_id,
                    snapshot.conversation_id,
                    query=user_text,
                    now=committed,
                )
            pair_id = request or uuid.uuid4().hex
            (
                next_profile,
                accepted_memory_operations,
                rejected_operations,
                ownership_classification,
            ) = self._apply_proposals(
                profile,
                proposals,
                user_text=user_text,
                assistant_text=assistant_text,
                source_id=pair_id,
                trusted_history="\n".join(
                    (
                        *(turn.content for turn in _complete_turns(conversation.recent_turns)),
                        *(memory.text for memory in profile.memories),
                        *(item.summary for item in profile.relationship.shared_context),
                    )
                ),
                now=committed,
            )
            next_self_model = self._self_model
            next_strategies = self._strategies
            proposal_values = proposals if isinstance(proposals, dict) else {}
            if snapshot.profile_id == OWNER_PROFILE_ID:
                (
                    next_self_model,
                    self_model_rejections,
                    self_model_ownership,
                ) = self._apply_self_model_operations(
                    self._self_model,
                    proposal_values.get("self_model_ops"),
                    proposal_values.get("improvement_ops"),
                    profile=next_profile,
                    conversation=conversation,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    source_id=pair_id,
                    now=committed,
                )
                rejected_operations += self_model_rejections
                ownership_classification += self_model_ownership
                next_strategies = self._repair_stale_strategies(
                    self._strategies,
                    next_self_model,
                    now=committed,
                )
                (
                    next_strategies,
                    strategy_rejections,
                    strategy_ownership,
                ) = self._apply_strategy_operations(
                    next_strategies,
                    proposal_values.get("strategy_ops"),
                    self_model=next_self_model,
                    assistant_text=assistant_text,
                    source_id=pair_id,
                    now=committed,
                )
                rejected_operations += strategy_rejections
                ownership_classification += strategy_ownership
                next_strategies, next_self_model = self._evaluate_strategies(
                    next_strategies,
                    self._strategies,
                    next_self_model,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    source_id=pair_id,
                    now=committed,
                )
            elif any(
                key in proposal_values
                for key in ("self_model_ops", "improvement_ops", "strategy_ops")
            ):
                rejected_operations += (
                    "self_model_ops/strategy_ops: unavailable outside Akane's owner profile",
                )
            next_profile, affect_transition = _apply_conversation_affect(
                next_profile,
                conversation,
                user_text=user_text,
                source_id=pair_id,
                now=committed,
            )
            if allow_initiative and snapshot.profile_id == OWNER_PROFILE_ID:
                next_profile = _with_conversation_initiative(profile, next_profile, now=committed)
            turns = list(conversation.recent_turns)
            if assistant_text:
                turns.extend(
                    (
                        ChatTurn(
                            f"{pair_id}:user",
                            "user",
                            str(user_text).strip()[:8_000],
                            snapshot.now,
                            compact_text(source, 32) or "unknown",
                        ),
                        ChatTurn(
                            f"{pair_id}:assistant",
                            "assistant",
                            str(assistant_text).strip()[:8_000],
                            committed,
                            compact_text(source, 32) or "unknown",
                        ),
                    )
                )
            ids = (
                tuple(dict.fromkeys((*conversation.committed_request_ids, request)))[-32:]
                if request
                else conversation.committed_request_ids
            )
            replies = (
                tuple(
                    dict(
                        (
                            *conversation.request_replies,
                            (request, assistant_text),
                        )
                    ).items()
                )[-32:]
                if request
                else conversation.request_replies
            )
            next_conversation = replace(
                conversation,
                recent_turns=_trim_turns(tuple(turns)),
                last_user_at=max(conversation.last_user_at, snapshot.now),
                last_assistant_at=(
                    max(conversation.last_assistant_at, committed)
                    if assistant_text
                    else conversation.last_assistant_at
                ),
                updated_at=committed,
                committed_request_ids=ids,
                request_replies=replies,
            )
            profiles = self._profiles.copy()
            conversations = self._conversations.copy()
            profiles[snapshot.profile_id] = next_profile
            conversations[snapshot.conversation_id] = next_conversation
            self._replace_all(
                profiles,
                conversations,
                committed_at=committed,
                self_model=next_self_model,
                strategies=next_strategies,
            )
            if next_profile.initiative != profile.initiative:
                callback = self._autonomy_wake
                if callback is not None:
                    callback(snapshot.profile_id)
            recalled = _relevant_profile_state(next_profile, user_text, committed)
            return StateSnapshot(
                snapshot.profile_id,
                snapshot.conversation_id,
                self._revision,
                next_profile,
                next_conversation,
                _complete_turns(next_conversation.recent_turns),
                recalled[0],
                committed,
                max(
                    snapshot.last_profile_assistant_at,
                    next_conversation.last_assistant_at,
                ),
                max(
                    snapshot.last_profile_user_at,
                    next_conversation.last_user_at,
                ),
                snapshot.last_profile_initiative,
                accepted_memory_operations,
                (*proposal_rejections, *rejected_operations),
                ownership_classification,
                relevant_opinions=recalled[1],
                relevant_preferences=recalled[2],
                relevant_interests=recalled[3],
                relevant_relationship=recalled[4],
                familiarity=_familiarity_context(
                    next_profile,
                    conversations,
                    snapshot.profile_id,
                ),
                affect_transition=affect_transition,
                self_model=(
                    next_self_model
                    if snapshot.profile_id == OWNER_PROFILE_ID
                    else SelfModelState()
                ),
                relevant_self_model=_relevant_self_model(
                    (
                        next_self_model
                        if snapshot.profile_id == OWNER_PROFILE_ID
                        else SelfModelState()
                    ),
                    user_text,
                    committed,
                ),
                strategies=(
                    next_strategies
                    if snapshot.profile_id == OWNER_PROFILE_ID
                    else ()
                ),
                relevant_strategies=_relevant_strategies(
                    (
                        next_strategies
                        if snapshot.profile_id == OWNER_PROFILE_ID
                        else ()
                    ),
                    (
                        next_self_model
                        if snapshot.profile_id == OWNER_PROFILE_ID
                        else SelfModelState()
                    ),
                    user_text,
                ),
            )

    def messages(
        self,
        conversation_id: str,
        profile_id: str = OWNER_PROFILE_ID,
    ) -> list[dict[str, str]]:
        key = compact_text(conversation_id, 160) or "popup:default"
        profile = canonical_profile_id(profile_id)
        with self._lock:
            record = self._conversation(profile, key)
            return [turn.as_message() for turn in record.recent_turns]

    def reply_for_request(
        self,
        conversation_id: str,
        profile_id: str,
        request_id: str,
    ) -> str | None:
        request = compact_text(request_id, 180)
        if not request:
            return None
        key = compact_text(conversation_id, 160) or "popup:default"
        with self._lock:
            record = self._conversation(canonical_profile_id(profile_id), key)
            return dict(record.request_replies).get(request)

    def public_conversation(
        self,
        conversation_id: str,
        profile_id: str = OWNER_PROFILE_ID,
    ) -> dict[str, object]:
        key = compact_text(conversation_id, 160) or "popup:default"
        with self._lock:
            record = self._conversation(canonical_profile_id(profile_id), key)
            return {
                "conversation_id": record.conversation_id,
                "profile_id": record.profile_id,
                "recent_turns": [
                    {
                        "role": turn.role,
                        "text": turn.content,
                        "timestamp": turn.timestamp,
                    }
                    for turn in record.recent_turns
                ],
                "last_user_at": record.last_user_at,
                "last_assistant_at": record.last_assistant_at,
                "updated_at": record.updated_at,
            }

    def clear_conversation(self, conversation_id: str, profile_id: str) -> None:
        key = compact_text(conversation_id, 160) or "popup:default"
        with self._lock:
            record = self._conversations.get(key)
            if record is None or record.profile_id != canonical_profile_id(profile_id):
                return
            conversations = self._conversations.copy()
            conversations.pop(key, None)
            self._replace_all(self._profiles.copy(), conversations, committed_at=time.time())

    def clear_profile(self, profile_id: str = OWNER_PROFILE_ID) -> None:
        profile = canonical_profile_id(profile_id)
        current = time.time()
        with self._lock:
            profiles = self._profiles.copy()
            if profile == OWNER_PROFILE_ID:
                profiles[profile] = _new_profile(current)
            else:
                profiles.pop(profile, None)
            conversations = {
                key: value
                for key, value in self._conversations.items()
                if value.profile_id != profile
            }
            self._replace_all(
                profiles,
                conversations,
                committed_at=current,
                self_model=(
                    SelfModelState()
                    if profile == OWNER_PROFILE_ID
                    else self._self_model
                ),
                strategies=(() if profile == OWNER_PROFILE_ID else self._strategies),
            )
            callback = self._autonomy_wake if profile == OWNER_PROFILE_ID else None
        if callback is not None:
            callback(profile)

    def public_profile(self, profile_id: str = OWNER_PROFILE_ID) -> dict[str, object]:
        profile = canonical_profile_id(profile_id)
        with self._lock:
            state = self._profiles.get(profile) or _new_profile(time.time())
            return {
                "profile_id": profile,
                "interests": list(state.interests),
                "preferences": [asdict(item) for item in state.preferences],
                "opinions": [
                    {
                        "topic": item.topic,
                        "position": item.position,
                        "reason": item.reason,
                        "updated_at": item.updated_at,
                    }
                    for item in state.opinions
                ],
                "updated_at": state.updated_at,
            }

    def public_memory(self, profile_id: str = OWNER_PROFILE_ID) -> dict[str, object]:
        profile = canonical_profile_id(profile_id)
        with self._lock:
            current_time = time.time()
            state = self._profiles.get(profile) or _new_profile(current_time)
            state = self._normalize_profile_presence(state, now=current_time)
            current = state.presence.current_activity
            activities = {}
            if current is not None:
                label = current.subject or current.kind
                activities[label] = {
                    "status": "active",
                    "details": [current.kind, current.subject_kind],
                }
            return {
                "user": {},
                "preferences": [
                    {"content": item.content}
                    for item in state.preferences
                ],
                "activities": activities,
                "facts": [
                    {
                        "content": item.content,
                        "subject": item.subject,
                        "kind": item.kind,
                        "confidence": item.confidence,
                    }
                    for item in state.memories
                ],
                "interests": list(state.interests),
            }

    def public_internal_state(self, profile_id: str = OWNER_PROFILE_ID) -> dict[str, object]:
        profile = canonical_profile_id(profile_id)
        current = time.time()
        with self._lock:
            state = self._profiles.get(profile) or _new_profile(current)
            state = self._normalize_profile_presence(state, now=current)
            effective = effective_emotional_state(state, now=current)
            return {
                **effective.as_dict(),
                "self_model": (
                    self._self_model.as_dict()
                    if profile == OWNER_PROFILE_ID
                    else SelfModelState().as_dict()
                ),
                "strategies": (
                    [strategy.as_dict() for strategy in self._strategies]
                    if profile == OWNER_PROFILE_ID
                    else []
                ),
                "profile_id": profile,
                "revision": self._revision,
                "committed_at": self._committed_at,
            }

    def offer_initiative(
        self,
        opportunity: InitiativeOpportunity,
        *,
        profile_id: str = OWNER_PROFILE_ID,
        now: float,
    ) -> bool:
        profile = canonical_profile_id(profile_id)
        if profile != OWNER_PROFILE_ID or opportunity.status != "pending":
            return False
        with self._lock:
            current_profile = self._profiles.get(profile)
            if current_profile is None:
                return False
            initiative = _settle_initiative(current_profile.initiative, now=now)
            active = initiative.current
            if (
                active is not None
                and active.status in {"pending", "pending_delivery"}
                and active.expires_at > now
            ):
                return False
            if opportunity.source_type != "reminder" and (
                opportunity.source_id in initiative.handled_source_ids
                or any(
                    recent.source_id == opportunity.source_id
                    or (
                        recent.source_type != "reminder"
                        and _similar(recent.topic_key, opportunity.topic_key)
                        >= 0.78
                    )
                    for recent in initiative.recent
                )
            ):
                return False
            next_profile = replace(
                current_profile,
                initiative=replace(initiative, current=opportunity),
                updated_at=max(current_profile.updated_at, now),
            )
            changed = self._replace_profile(profile, next_profile, committed_at=now)
            callback = self._autonomy_wake
        if changed and callback is not None:
            callback(profile)
        return changed

    def initiative_schedule(self, *, now: float) -> tuple[bool, float | None]:
        current = max(0.0, float(now))
        with self._lock:
            profile, initiative, opportunity = self._settled_owner_initiative(current)
            if profile is None:
                return False, None
            due = bool(
                opportunity is not None
                and opportunity.status == "pending"
                and opportunity.claim_token is None
                and opportunity.not_before <= current
                and opportunity.expires_at > current
            )
            wakes: list[float] = []
            if opportunity is not None and opportunity.status in {"pending", "pending_delivery"}:
                wakes.append(opportunity.expires_at)
                if opportunity.claim_token:
                    wakes.append(opportunity.claim_expires_at)
                elif opportunity.status == "pending":
                    wakes.append(opportunity.not_before)
                elif initiative.cooldown_until > current:
                    wakes.append(initiative.cooldown_until)
            if initiative != profile.initiative:
                self._replace_profile(
                    OWNER_PROFILE_ID,
                    replace(profile, initiative=initiative),
                    committed_at=current,
                )
            future = tuple(value for value in wakes if value > current)
            return due, min(future, default=None)

    def _settled_owner_initiative(
        self,
        now: float,
    ) -> tuple[
        ProfileState | None,
        InitiativeState | None,
        InitiativeOpportunity | None,
    ]:
        profile = self._profiles.get(OWNER_PROFILE_ID)
        if profile is None:
            return None, None, None
        opportunity = profile.initiative.current
        if (
            opportunity is not None
            and opportunity.claim_token
            and opportunity.claim_expires_at <= now
        ):
            self._expired_claim_recoveries += 1
        initiative = _settle_initiative(profile.initiative, now=now)
        return profile, initiative, initiative.current

    def claim_initiative_evaluation(self, *, now: float) -> InitiativeOpportunity | None:
        current = max(0.0, float(now))
        with self._lock:
            profile, initiative, opportunity = self._settled_owner_initiative(current)
            if profile is None:
                return None
            if (
                opportunity is None
                or opportunity.status != "pending"
                or opportunity.claim_token is not None
                or opportunity.not_before > current
                or opportunity.expires_at <= current
            ):
                return None
            if not _initiative_source_exists(profile, opportunity):
                dismissed = replace(opportunity, status="dismissed", evaluated_at=current)
                initiative = _handled_initiative(replace(initiative, current=dismissed), dismissed)
                self._replace_profile(
                    OWNER_PROFILE_ID,
                    replace(profile, initiative=initiative),
                    committed_at=current,
                )
                return None
            claimed = replace(
                opportunity,
                claim_token=uuid.uuid4().hex,
                claim_expires_at=current + _INITIATIVE_EVALUATION_CLAIM_SECONDS,
            )
            self._replace_profile(
                OWNER_PROFILE_ID,
                replace(
                    profile,
                    initiative=replace(initiative, current=claimed),
                ),
                committed_at=current,
            )
            return claimed

    def complete_initiative_evaluation(
        self,
        *,
        claim_token: str,
        decision: str,
        topic: str | None,
        message: str | None,
        now: float,
    ) -> InitiativeOpportunity | None:
        current = max(0.0, float(now))
        with self._lock:
            profile = self._profiles.get(OWNER_PROFILE_ID)
            opportunity = profile.initiative.current if profile else None
            if (
                profile is None
                or opportunity is None
                or opportunity.status != "pending"
                or opportunity.claim_token != claim_token
            ):
                return None
            initiative = profile.initiative
            normalized_topic = _key(topic)[:120]
            speak = (decision == "speak" and normalized_topic and compact_text(message, 500))
            duplicate = bool(
                speak
                and any(
                    _similar(item.topic_key, normalized_topic) >= 0.78
                    for item in initiative.recent
                )
            )
            if not speak or duplicate:
                completed = replace(
                    opportunity,
                    status="dismissed",
                    evaluated_at=current,
                    claim_token=None,
                    claim_expires_at=0.0,
                )
                initiative = _handled_initiative(replace(initiative, current=completed), completed)
            else:
                completed = replace(
                    opportunity,
                    topic_key=normalized_topic,
                    status="pending_delivery",
                    message=compact_text(message, 500),
                    evaluated_at=current,
                    generated_at=current,
                    claim_token=None,
                    claim_expires_at=0.0,
                )
                initiative = replace(initiative, current=completed)
            self._replace_profile(
                OWNER_PROFILE_ID,
                replace(
                    profile,
                    initiative=initiative,
                    updated_at=max(profile.updated_at, current),
                ),
                committed_at=current,
            )
            self._initiative_failure_count = 0
            callback = self._autonomy_wake
        if callback is not None:
            callback(OWNER_PROFILE_ID)
        return completed

    def fail_initiative_evaluation(self, *, claim_token: str, now: float) -> bool:
        current = max(0.0, float(now))
        with self._lock:
            profile = self._profiles.get(OWNER_PROFILE_ID)
            opportunity = profile.initiative.current if profile else None
            if (
                profile is None
                or opportunity is None
                or opportunity.status != "pending"
                or opportunity.claim_token != claim_token
            ):
                return False
            failure_count = self._initiative_failure_count + 1
            pending = replace(
                opportunity,
                not_before=min(
                    opportunity.expires_at,
                    current + _background_retry_delay(failure_count),
                ),
                claim_token=None,
                claim_expires_at=0.0,
            )
            changed = self._replace_profile(
                OWNER_PROFILE_ID,
                replace(
                    profile,
                    initiative=replace(profile.initiative, current=pending),
                ),
                committed_at=current,
            )
            if changed:
                self._initiative_failure_count = failure_count
            return changed

    def claim_initiative_delivery(
        self,
        *,
        adapter: str,
        available_adapters: tuple[str, ...],
        now: float,
    ) -> InitiativeOpportunity | None:
        current = max(0.0, float(now))
        channel = compact_text(adapter, 16).casefold()
        available = tuple(
            value
            for item in available_adapters
            if (value := compact_text(item, 16).casefold())
            in {"popup", "discord"}
        )
        with self._lock:
            profile, initiative, opportunity = self._settled_owner_initiative(current)
            if profile is None:
                return None
            if (
                opportunity is None
                or opportunity.status != "pending_delivery"
                or opportunity.claim_token is not None
                or not opportunity.message
                or not _ordinary_delivery_allowed(
                    initiative,
                    opportunity,
                    now=current,
                )
            ):
                return None
            eligible = tuple(
                value
                for value in available
                if value not in opportunity.failed_channels
            ) or available
            selected = (
                "popup"
                if "popup" in eligible
                else "discord"
                if "discord" in eligible
                else ""
            )
            if channel != selected:
                return None
            claimed = replace(
                opportunity,
                claim_token=uuid.uuid4().hex,
                claim_expires_at=current + _INITIATIVE_DELIVERY_CLAIM_SECONDS,
                delivery_channel=channel,
            )
            self._replace_profile(
                OWNER_PROFILE_ID,
                replace(
                    profile,
                    initiative=replace(initiative, current=claimed),
                ),
                committed_at=current,
            )
            return claimed

    def acknowledge_initiative_delivery(
        self,
        *,
        opportunity_id: str,
        claim_token: str,
        adapter: str,
        conversation_id: str,
        success: bool,
        message_id: str = "",
        now: float,
    ) -> bool:
        current = max(0.0, float(now))
        channel = compact_text(adapter, 16).casefold()
        conversation_key = compact_text(conversation_id, 160) or "popup:default"
        with self._lock:
            profile = self._profiles.get(OWNER_PROFILE_ID)
            opportunity = profile.initiative.current if profile else None
            if (
                profile is None
                or opportunity is None
                or opportunity.opportunity_id != compact_text(opportunity_id, 100)
            ):
                return False
            if opportunity.status == "sent":
                return (
                    success
                    and opportunity.delivery_channel == channel
                    and opportunity.delivery_message_id
                    == (compact_text(message_id, 160) or opportunity.opportunity_id)
                )
            if (
                opportunity.status != "pending_delivery"
                or opportunity.claim_token != claim_token
                or opportunity.delivery_channel != channel
            ):
                return False
            initiative = profile.initiative
            profiles = self._profiles.copy()
            conversations = self._conversations.copy()
            if not success:
                failed = tuple(dict.fromkeys((*opportunity.failed_channels, channel)))
                pending = replace(
                    opportunity,
                    claim_token=None,
                    claim_expires_at=0.0,
                    delivery_channel=None,
                    failed_channels=failed,
                )
                profiles[OWNER_PROFILE_ID] = replace(
                    profile,
                    initiative=replace(initiative, current=pending),
                )
            else:
                delivery_id = (compact_text(message_id, 160) or opportunity.opportunity_id)
                sent = replace(
                    opportunity,
                    status="sent",
                    claim_token=None,
                    claim_expires_at=0.0,
                    delivered_at=current,
                    delivery_message_id=delivery_id,
                )
                recent = (
                    *initiative.recent,
                    SentInitiative(
                        sent.topic_key,
                        sent.source_id,
                        sent.source_type,
                        current,
                    ),
                )[-_MAX_RECENT_INITIATIVES:]
                initiative = _handled_initiative(
                    replace(
                        initiative,
                        current=sent,
                        cooldown_until=(
                            initiative.cooldown_until
                            if sent.source_type == "reminder"
                            else current
                            + ORDINARY_INITIATIVE_COOLDOWN_SECONDS
                        ),
                        recent=recent,
                    ),
                    sent,
                )
                profiles[OWNER_PROFILE_ID] = replace(
                    profile,
                    initiative=initiative,
                    updated_at=max(profile.updated_at, current),
                )
                record = self._conversation(OWNER_PROFILE_ID, conversation_key)
                turn_id = f"initiative:{opportunity.opportunity_id}"
                if not any(turn.turn_id == turn_id for turn in record.recent_turns):
                    turn = ChatTurn(
                        turn_id,
                        "assistant",
                        opportunity.message or "",
                        current,
                        "initiative",
                    )
                    record = replace(
                        record,
                        recent_turns=_trim_turns((*record.recent_turns, turn)),
                        last_assistant_at=max(record.last_assistant_at, current),
                        updated_at=max(record.updated_at, current),
                    )
                    conversations[conversation_key] = record
            self._replace_profile(
                OWNER_PROFILE_ID,
                profiles[OWNER_PROFILE_ID],
                conversations=conversations,
                committed_at=current,
            )
            callback = self._autonomy_wake
        if callback is not None:
            callback(OWNER_PROFILE_ID)
        return True

    def release_initiative_delivery(self, *, adapter: str, now: float) -> bool:
        channel = compact_text(adapter, 16).casefold()
        with self._lock:
            profile = self._profiles.get(OWNER_PROFILE_ID)
            opportunity = profile.initiative.current if profile else None
            if (
                profile is None
                or opportunity is None
                or opportunity.status != "pending_delivery"
                or opportunity.delivery_channel != channel
                or opportunity.claim_token is None
            ):
                return False
            failed = tuple(dict.fromkeys((*opportunity.failed_channels, channel)))
            pending = replace(
                opportunity,
                claim_token=None,
                claim_expires_at=0.0,
                delivery_channel=None,
                failed_channels=failed,
            )
            return self._replace_profile(
                OWNER_PROFILE_ID,
                replace(
                    profile,
                    initiative=replace(profile.initiative, current=pending),
                ),
                committed_at=max(0.0, float(now)),
            )

    def set_autonomy_wake(self, callback: Callable[[str], None] | None) -> None:
        with self._lock:
            self._autonomy_wake = callback

    @staticmethod
    def _presence_due_at(presence: PresenceState, now: float) -> float:
        if presence.retry_at > 0.0:
            return presence.retry_at
        if presence.next_decision_at <= 0.0:
            return now
        return presence.next_decision_at

    @staticmethod
    def _presence_due(presence: PresenceState, now: float) -> bool:
        return (presence.claim_token is None and StateStore._presence_due_at(presence, now) <= now)

    def presence_schedule(self, *, now: float) -> tuple[tuple[str, ...], float | None]:
        current = max(0.0, float(now))
        with self._lock:
            profiles = self._profiles.copy()
            changed = False
            due: list[str] = []
            wakes: list[float] = []
            for profile_id, state in self._profiles.items():
                if profile_id != OWNER_PROFILE_ID:
                    continue
                expired_claim = bool(
                    state.presence.claim_token
                    and state.presence.claim_expires_at <= current
                )
                presence = normalize_presence(state.presence, now=current)
                if expired_claim:
                    self._expired_claim_recoveries += 1
                    presence = replace(presence, last_error="stale claim released")
                if presence != state.presence:
                    profiles[profile_id] = replace(state, presence=presence)
                    changed = True
                if presence.claim_token is not None:
                    wakes.append(presence.claim_expires_at)
                else:
                    due_at = self._presence_due_at(presence, current)
                    if due_at <= current:
                        due.append(profile_id)
                    else:
                        wakes.append(due_at)
            if changed:
                self._replace_all(profiles, self._conversations.copy(), committed_at=current)
            return tuple(due), min(wakes, default=None)

    def claim_presence_decision(self, profile_id: str, *, now: float) -> ProfileState | None:
        profile = canonical_profile_id(profile_id)
        if profile != OWNER_PROFILE_ID:
            return None
        current = max(0.0, float(now))
        with self._lock:
            state = self._profiles.get(profile)
            if state is None:
                return None
            expired_claim = bool(
                state.presence.claim_token
                and state.presence.claim_expires_at <= current
            )
            candidate = normalize_presence(state.presence, now=current)
            if expired_claim:
                self._expired_claim_recoveries += 1
                candidate = replace(candidate, last_error="stale claim released")
            if not self._presence_due(candidate, current):
                if candidate != state.presence:
                    self._replace_profile(
                        profile,
                        replace(state, presence=candidate),
                        committed_at=current,
                    )
                return None
            presence = replace(
                candidate,
                claim_token=uuid.uuid4().hex,
                claim_expires_at=current + CLAIM_SECONDS,
            )
            next_state = replace(state, presence=presence)
            self._replace_profile(profile, next_state, committed_at=current)
            return next_state

    def _claimed_presence(
        self,
        profile_id: str,
        claim_token: str,
        now: float,
    ) -> tuple[ProfileState | None, PresenceState | None]:
        state = self._profiles.get(profile_id)
        if state is None:
            return None, None
        presence = normalize_presence(state.presence, now=now)
        if presence.claim_token != claim_token:
            if presence != state.presence:
                self._replace_profile(
                    profile_id,
                    replace(state, presence=presence),
                    committed_at=now,
                )
            return state, None
        return replace(state, presence=presence), presence

    def _failed_presence(
        self,
        presence: PresenceState,
        *,
        now: float,
        error: str,
        failure_count: int,
    ) -> PresenceState:
        presence = normalize_presence(presence, now=now)
        return replace(
            presence,
            claim_token=None,
            claim_expires_at=0.0,
            retry_at=now + _background_retry_delay(failure_count),
            last_error=compact_text(error, 120) or "presence decision failed",
        )

    def fail_presence_decision(
        self,
        profile_id: str,
        *,
        claim_token: str,
        now: float,
        error: str,
    ) -> bool:
        profile = canonical_profile_id(profile_id)
        if profile != OWNER_PROFILE_ID:
            return False
        current = max(0.0, float(now))
        with self._lock:
            state, presence = self._claimed_presence(profile, claim_token, current)
            if presence is None:
                return False
            failure_count = self._presence_failure_count + 1
            next_state = replace(
                state,
                presence=self._failed_presence(
                    presence,
                    now=current,
                    error=error,
                    failure_count=failure_count,
                ),
            )
            self._replace_profile(profile, next_state, committed_at=current)
            self._presence_failure_count = failure_count
            callback = self._autonomy_wake
        if callback is not None:
            callback(profile)
        return True

    @staticmethod
    def _presence_emotion(
        state: ProfileState,
        proposal: ProposedEmotion | None,
        *,
        source_activity: PresenceActivity | None,
        basis: str,
        expected_emotion_updated_at: float,
        now: float,
    ) -> EmotionState:
        if state.emotion.updated_at > max(0.0, float(expected_emotion_updated_at)):
            return state.emotion
        if proposal is not None and basis and _grounded(proposal.cause, basis):
            return EmotionState(
                proposal.primary,
                proposal.intensity,
                proposal.cause,
                "offscreen_presence",
                source_activity.activity_id if source_activity else None,
                now,
                now,
            )
        return state.emotion

    def _with_presence_experience(
        self,
        state: ProfileState,
        appraisal: PresenceAppraisal,
        *,
        source_activity: PresenceActivity | None,
        basis: str,
        now: float,
    ) -> tuple[ProfileState, bool]:
        experience = appraisal.experience
        if experience is None or source_activity is None or not basis:
            return state, False
        eligibility = presence_consequence_eligibility(
            state,
            source_activity,
            now=now,
        )
        if (
            not eligibility.eligible
            or experience.target_id != source_activity.source_ids[0]
            or experience.source_ids != source_activity.source_ids
            or not _grounded(experience.summary, basis)
            or not _grounded(experience.reason, f"{basis} {experience.summary}")
            or not _memory_ownership_matches("akane", experience.summary)
            or _SENSITIVE_PATTERN.search(f"{experience.summary} {experience.reason}")
            or _PRESENCE_PHYSICAL_CLAIM.search(
                f"{experience.summary} {experience.reason}"
            )
            or _PRESENCE_EXTERNAL_CLAIM.search(
                f"{experience.summary} {experience.reason}"
            )
            or _PRESENCE_USER_FACT.search(experience.summary)
            or _UNVERIFIED_HISTORY.search(
                f"{experience.summary} {experience.reason}"
            )
        ):
            return state, False

        assistant_evidence = " ".join(
            part
            for part in (
                experience.summary,
                experience.position or "",
                experience.reason,
                basis,
            )
            if part
        )
        if experience.kind == "interest" and experience.topic:
            if (
                source_activity.kind != "revisiting_interest"
                or not _grounded(experience.topic, basis)
            ):
                return state, False
            interests, rejected, _ownership = self._apply_interest_operations(
                state.interests,
                [
                    {
                        "op": experience.operation,
                        "topic": experience.topic,
                        "reason": experience.reason,
                        "strength": experience.confidence,
                    }
                ],
                user_text=basis,
                assistant_text=assistant_evidence,
                source_id=source_activity.activity_id,
                now=now,
                source_type="offscreen_presence",
            )
            if rejected or [item.as_dict() for item in interests] == [
                item.as_dict() for item in state.interests
            ]:
                return state, False
            return replace(state, interests=interests, updated_at=now), True

        if experience.kind == "opinion" and experience.topic and experience.position:
            if (
                source_activity.kind != "reconsidering_opinion"
                or not _grounded(experience.topic, basis)
                or not _grounded(experience.position, f"{basis} {experience.summary}")
            ):
                return state, False
            existing_opinion = next(
                (
                    opinion
                    for opinion in state.opinions
                    if opinion.id == experience.target_id
                ),
                None,
            )
            if existing_opinion is None:
                return state, False
            opinions, rejected, _ownership = self._apply_opinion_operations(
                state.opinions,
                [
                    {
                        "op": experience.operation,
                        "target_id": experience.target_id,
                        "topic": experience.topic,
                        "domain": existing_opinion.domain,
                        "position": experience.position,
                        "reason": experience.reason,
                        "confidence": experience.confidence,
                        "importance": existing_opinion.importance,
                    }
                ],
                user_text=basis,
                assistant_text=assistant_evidence,
                now=now,
                source_type="offscreen_presence",
                source_ids=(source_activity.activity_id, *experience.source_ids),
                evidence_summary=experience.summary,
            )
            if rejected or opinions == state.opinions:
                return state, False
            return replace(state, opinions=opinions, updated_at=now), True

        if experience.kind == "memory":
            memory_kind = (
                "event" if experience.meaning == "connection" else "concern"
            )
            if (
                experience.confidence < 0.80
                or experience.meaning == "connection"
                and source_activity.kind != "reflecting_on_shared_thread"
                or experience.meaning == "unfinished_thought"
                and source_activity.kind != "following_unfinished_thought"
                or any(
                    memory.subject == "akane"
                    and _duplicate_text(memory.text, experience.summary)
                    for memory in state.memories
                )
            ):
                return state, False
            memory = self._validated_memory(
                {
                    "subject": "akane",
                    "kind": memory_kind,
                    "text": experience.summary,
                    "confidence": experience.confidence,
                },
                user_text=basis,
                assistant_text=assistant_evidence,
                now=now,
                source_type="offscreen_presence",
                source_id=source_activity.activity_id,
                reason=experience.reason,
            )
            if memory is None:
                return state, False
            return replace(
                state,
                memories=_merge_memories(state.memories, (memory,)),
                updated_at=now,
            ), True
        return state, False

    def commit_presence_decision(
        self,
        profile_id: str,
        appraisal: PresenceAppraisal | None,
        *,
        claim_token: str,
        now: float,
        expected_activity_id: str | None,
        expected_emotion_updated_at: float = 0.0,
        appraisal_attempted: bool = False,
    ) -> tuple[bool, str]:
        profile = canonical_profile_id(profile_id)
        if profile != OWNER_PROFILE_ID:
            return False, "presence claim is unavailable"
        current = max(0.0, float(now))
        with self._lock:
            state, normalized_presence = self._claimed_presence(profile, claim_token, current)
            if normalized_presence is None:
                return False, "presence claim is unavailable"
            completed_activity = state.presence.previous_activity
            actual_activity_id = completed_activity.activity_id if completed_activity else None
            expected_id = compact_text(expected_activity_id, 80) or None
            rejection = (
                "claimed presence activity changed"
                if actual_activity_id != expected_id
                else ""
            )
            if rejection:
                failure_count = self._presence_failure_count + 1
                presence = self._failed_presence(
                    state.presence,
                    now=current,
                    error=rejection,
                    failure_count=failure_count,
                )
                next_state = replace(state, presence=presence)
            else:
                candidates = _presence_candidates(state)
                selection = choose_presence_transition(
                    candidates,
                    state.presence,
                    now=current,
                    emotion_weights=_presence_emotion_weights(state, now=current),
                )
                selected = selection.candidate
                activity = make_presence_activity(
                    selected,
                    now=current,
                    activity_id=uuid.uuid4().hex,
                    existing=(completed_activity if selection.continue_current else None),
                )
                selected_source = selected.primary_source_id if selected else ""
                completed_source = (
                    completed_activity.source_ids[0]
                    if completed_activity and completed_activity.source_ids
                    else ""
                )
                previous_selected_source = (
                    completed_source
                    or (
                        state.presence.recent_source_ids[-1]
                        if state.presence.recent_source_ids
                        else ""
                    )
                )
                if not selected_source:
                    repetition_count = state.presence.repetition_count
                elif selection.reset_repetition or selected_source != previous_selected_source:
                    repetition_count = 1
                else:
                    repetition_count = min(3, state.presence.repetition_count + 1)
                recent_sources = state.presence.recent_source_ids
                if selected_source:
                    recent_sources = (*recent_sources, selected_source)[-6:]
                quiet_streak = (
                    min(3, state.presence.quiet_streak + 1)
                    if activity.kind == "quiet" and completed_activity is not None
                    and completed_activity.kind == "quiet"
                    else 1
                    if activity.kind == "quiet"
                    else 0
                )
                presence = replace(
                    state.presence,
                    current_activity=activity,
                    last_decision_at=current,
                    next_decision_at=activity.expected_end_at,
                    retry_at=0.0,
                    last_error=None,
                    claim_token=None,
                    claim_expires_at=0.0,
                    repetition_count=repetition_count,
                    recent_source_ids=recent_sources,
                    quiet_streak=quiet_streak,
                    last_transition_reason=selection.reason,
                    last_candidate_score=(
                        round(selection.score.total, 3) if selection.score else None
                    ),
                    last_candidate_source_id=(
                        selection.candidate.primary_source_id
                        if selection.candidate is not None
                        else None
                    ),
                )
                basis = presence_activity_basis(state, completed_activity)
                proposal = appraisal.emotion if appraisal is not None else None
                emotion = self._presence_emotion(
                    state,
                    proposal,
                    source_activity=completed_activity,
                    basis=basis,
                    expected_emotion_updated_at=expected_emotion_updated_at,
                    now=current,
                )
                next_state = replace(
                    state,
                    presence=presence,
                    emotion=emotion,
                    updated_at=current,
                )
                consequence_committed = False
                if appraisal is not None:
                    next_state, consequence_committed = self._with_presence_experience(
                        next_state,
                        appraisal,
                        source_activity=completed_activity,
                        basis=basis,
                        now=current,
                    )
                if appraisal_attempted and completed_activity is not None:
                    proposed_experience = bool(
                        appraisal is not None and appraisal.experience is not None
                    )
                    next_state = replace(
                        next_state,
                        presence=replace(
                            next_state.presence,
                            last_appraised_activity_id=completed_activity.activity_id,
                            last_appraised_at=current,
                            last_appraisal_result=(
                                "committed"
                                if consequence_committed
                                else "rejected"
                                if proposed_experience
                                else "null"
                            ),
                        ),
                    )
            self._replace_profile(profile, next_state, committed_at=current)
            if rejection:
                self._presence_failure_count = failure_count
            else:
                self._presence_failure_count = 0
            callback = self._autonomy_wake
        if callback is not None:
            callback(profile)
        return not bool(rejection), rejection


_STORE_LOCK = threading.Lock()
_STATE_STORE: StateStore | None = None


def get_state_store() -> StateStore:
    global _STATE_STORE
    if _STATE_STORE is None:
        with _STORE_LOCK:
            if _STATE_STORE is None:
                _STATE_STORE = StateStore()
    return _STATE_STORE
