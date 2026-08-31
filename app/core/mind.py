"""Cheap post-turn formation of grounded Experience, Self, and Memory."""

from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from app.core.state import Experience, Memory, MemoryChange, SelfChange, SelfItem, Turn, clamp
from app.core.utils import compact_text, lexical_terms, relevance, text_key


_GENERIC_TOPIC_TERMS = frozenset({
    "about", "akane", "and", "are", "as", "at", "because", "choose", "do",
    "feel", "favorite", "for", "from", "have", "interest", "into", "is",
    "it", "like", "me", "my", "of", "on", "opinion", "prefer", "that",
    "the", "them", "think", "this", "to", "view", "what", "which", "while",
    "versus", "vs", "with", "would", "you", "your",
})

# These remaining language cues serve user Memory extraction only. Developed Self
# evidence arrives through the structured sidecar validated below.
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
_SEMANTIC_KINDS = _SELF_EVIDENCE_KINDS | _EVENT_EVIDENCE_KINDS | {"none"}
_SEMANTIC_STANCES = frozenset({"positive", "negative", "neutral", "comparative"})
_SEMANTIC_DURABILITY = frozenset({
    "candidate", "temporary", "hypothetical", "task-local", "none",
})
_SEMANTIC_KIND_CODES = {
    "p": "preference", "o": "opinion", "i": "interest", "g": "goal",
    "c": "correction", "f+": "positive_feedback", "f-": "negative_feedback",
    "t+": "task_success", "t-": "task_failure", "n": "none",
}
_SEMANTIC_STANCE_CODES = {
    "+": "positive", "-": "negative", "0": "neutral", "cmp": "comparative",
}
_SEMANTIC_DURABILITY_CODES = {
    "c": "candidate", "tmp": "temporary", "hyp": "hypothetical",
    "task": "task-local", "n": "none",
}
_SEMANTIC_REQUIRED_KEYS = {"k", "t", "s", "d", "e"}


@dataclass(frozen=True, slots=True)
class SemanticEvidence:
    kind: str
    topic: str
    stance: str
    durability: str
    evidence: str
    value: str = ""

SELF_FORM_STRENGTH = 0.35
SELF_FORM_CONFIDENCE = 0.65
SELF_REINFORCE_STRENGTH = 0.10
SELF_REINFORCE_CONFIDENCE = 0.05
SELF_ESTABLISHED_EVIDENCE = 3
SELF_ESTABLISHED_STRENGTH = 0.55
SELF_ESTABLISHED_CONFIDENCE = 0.75
SELF_WEAKEN_STRENGTH = 0.15
SELF_WEAKEN_CONFIDENCE = 0.10


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


def _same_self_topic(kind: str, left: object, right: object) -> bool:
    if kind == "preference":
        left_terms = self_topic_terms(left)
        return bool(left_terms) and left_terms == self_topic_terms(right)
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
    ranked = sorted(
        ((self_topic_similarity(topic, item.topic), item) for item in eligible),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return ranked[0][1] if ranked and ranked[0][0] >= 0.67 else None


def _related_judgment(
    items: tuple[SelfItem, ...], kind: str, topic: str,
) -> SelfItem | None:
    if kind not in {"opinion", "preference"}:
        return None
    ranked = sorted(
        (
            (
                1.0
                if "preference" in {kind, item.kind}
                and _same_self_topic("preference", topic, item.topic)
                else self_topic_similarity(topic, item.topic)
                if "preference" not in {kind, item.kind}
                else 0.0,
                item,
            )
            for item in items
            if item.kind in {"opinion", "preference"}
            and item.status in {"active", "uncertain"}
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return ranked[0][1] if ranked and ranked[0][0] >= 0.82 else None


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
    stance: str = "",
    semantic_value: str = "",
    now: float | None = None,
) -> SelfItem:
    current = time.time() if now is None else float(now)
    sources = _bounded_ids(item.source_ids, (source_id,))
    status = item.status
    if status == "uncertain" and len(sources) >= len(item.contradiction_ids) + 2:
        status = "active"
    return replace(
        item,
        strength=round(clamp(item.strength + SELF_REINFORCE_STRENGTH, 0.0, 1.0), 6),
        confidence=round(clamp(
            item.confidence + SELF_REINFORCE_CONFIDENCE, 0.0, 1.0,
        ), 6),
        reason=_semantic_reason(
            "reinforce", source_id, stance, semantic_value,
        ),
        status=status,
        updated_at=current,
        source_ids=sources,
    )


def weaken_self_item(
    item: SelfItem,
    *,
    source_id: str,
    now: float | None = None,
) -> SelfItem:
    current = time.time() if now is None else float(now)
    stance, semantic_value = _reason_semantics(item.reason)
    return replace(
        item,
        strength=round(clamp(item.strength - SELF_WEAKEN_STRENGTH, 0.0, 1.0), 6),
        confidence=round(clamp(
            item.confidence - SELF_WEAKEN_CONFIDENCE, 0.0, 1.0,
        ), 6),
        reason=_semantic_reason("weaken", source_id, stance, semantic_value),
        status="uncertain",
        updated_at=current,
        contradiction_ids=_bounded_ids(item.contradiction_ids, (source_id,)),
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
    return replace(
        item,
        value=evidence,
        strength=round(clamp(
            SELF_FORM_STRENGTH + prior_support * SELF_REINFORCE_STRENGTH, 0.0, 1.0,
        ), 6),
        confidence=round(clamp(
            SELF_FORM_CONFIDENCE + prior_support * SELF_REINFORCE_CONFIDENCE, 0.0, 1.0,
        ), 6),
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


def _normalized_evidence(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"',
    }))
    return " ".join(normalized.split()).casefold()


def validate_semantic_evidence(
    raw: str,
    user_turn: Turn,
    assistant_turn: Turn,
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
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    keys = set(payload)
    if keys not in (_SEMANTIC_REQUIRED_KEYS, _SEMANTIC_REQUIRED_KEYS | {"v"}):
        return None
    if any(not isinstance(payload[key], str) for key in keys):
        return None
    kind = _SEMANTIC_KIND_CODES.get(payload["k"], "")
    topic = payload["t"].strip()
    stance = _SEMANTIC_STANCE_CODES.get(payload["s"], "")
    durability = _SEMANTIC_DURABILITY_CODES.get(payload["d"], "")
    evidence = payload["e"].strip()
    value = payload.get("v", "").strip()
    if (
        kind not in _SEMANTIC_KINDS
        or stance not in _SEMANTIC_STANCES
        or durability not in _SEMANTIC_DURABILITY
        or len(topic) > 120
        or len(evidence) > 280
        or len(value) > 80
    ):
        return None
    if kind == "none" or durability != "candidate":
        return None
    if not topic or not evidence or not self_topic_terms(topic):
        return None
    evidence_key = _normalized_evidence(evidence)
    reply_key = _normalized_evidence(assistant_turn.content)
    user_key = _normalized_evidence(user_turn.content)
    if (
        not evidence_key
        or evidence_key not in reply_key
        or evidence_key in user_key
    ):
        return None
    if stance == "comparative":
        value_key = _normalized_evidence(value)
        if not value_key or value_key not in evidence_key:
            return None
    elif value:
        return None
    return SemanticEvidence(kind, topic, stance, durability, evidence, value)


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
    ranked = sorted(
        ((relevance(text, memory.text), memory) for memory in owned),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return ranked[0][1] if ranked and ranked[0][0] >= 0.62 else None


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
    text = compact_text(user_turn.content, 360)
    normalized = text_key(text)
    if (
        not text
        or text.rstrip().endswith("?")
        or _has_non_durable_framing(normalized)
        or _is_other_person_claim(text)
        or _QUOTED_FIRST_PERSON.search(text)
        or not _MEMORY_CUES.search(text)
    ):
        return None
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
        source_turn_ids=(*current.source_turn_ids, *source_ids)[-6:],
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


def same_experience(previous: Experience, candidate: Experience) -> bool:
    same_topic = (
        text_key(previous.topic) == text_key(candidate.topic)
        or relevance(previous.topic, candidate.topic) >= 0.85
    )
    previous_response = text_key(previous.akane_response)
    candidate_response = text_key(candidate.akane_response)
    previous_terms = lexical_terms(previous.akane_response)
    candidate_terms = lexical_terms(candidate.akane_response)
    response_jaccard = len(previous_terms & candidate_terms) / max(
        1, len(previous_terms | candidate_terms),
    )
    same_response = (
        previous_response == candidate_response
        or response_jaccard >= 0.72
        or SequenceMatcher(
            None, previous_response, candidate_response,
        ).ratio() >= 0.82
    )
    return (
        previous.kind == candidate.kind
        and previous.subject == candidate.subject
        and same_topic
        and same_response
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
        akane_response=semantic.evidence,
        now=now,
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
    if _same_judgment(existing, semantic):
        item = reinforce_self_item(
            existing,
            source_id=experience_id,
            stance=semantic.stance,
            semantic_value=semantic.value,
            now=now,
        )
        return SelfChange("reinforce", item, existing.id)
    if (
        self_development_state(existing) == "established"
        and not existing.contradiction_ids
    ):
        item = weaken_self_item(existing, source_id=experience_id, now=now)
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
    """Derive grounded evidence, then at most one Experience-backed Self change."""

    current = time.time() if now is None else float(now)
    memory = _memory_change(
        profile_id, memories, user_turn, assistant_turn, now=current,
    )
    semantic = (
        validate_semantic_evidence(semantic_evidence, user_turn, assistant_turn)
        if user_turn.profile_id == profile_id == assistant_turn.profile_id
        else None
    )
    if semantic is None:
        return ((memory,) if memory else (), (), ())
    if semantic.kind in _EVENT_EVIDENCE_KINDS:
        event = _event_evidence_experience(
            profile_id, user_turn, assistant_turn, semantic, now=current,
        )
        if experiences and same_experience(experiences[-1], event):
            event = None
        return (
            (memory,) if memory else (),
            (),
            (event,) if event else (),
        )
    existing = find_self_item(self_items, semantic.kind, semantic.topic)
    if existing is None:
        existing = _related_judgment(
            self_items, semantic.kind, semantic.topic,
        )
    self_item_id = existing.id if existing is not None else f"self_{uuid.uuid4().hex}"
    experience = _self_evidence_experience(
        profile_id, user_turn, assistant_turn,
        semantic=semantic,
        self_item_id=self_item_id,
        existing=existing, now=current,
    )
    if experiences and same_experience(experiences[-1], experience):
        return ((memory,) if memory else (), (), ())
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
    )
