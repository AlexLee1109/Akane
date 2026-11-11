"""V2 post-turn memory admission from actual user reports; no personality learning."""

from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import replace

from app.core.state import Memory, MemoryChange, Turn, clamp
from app.core.utils import compact_text, text_key


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
    r"\b(?:call me|i am|i'm|i've|i have|i feel|i live|i love|i hate|i like|i own|i prefer|"
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


def _bounded_ids(*groups: tuple[str, ...], limit: int = 6) -> tuple[str, ...]:
    return tuple(dict.fromkeys(identifier for group in groups for identifier in group))[-limit:]


def _normalized_evidence(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"',
    }))
    return " ".join(normalized.split()).casefold()


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
        if any(re.search(r'\b' + re.escape(phrase) + r'\b', normalized) for phrase in phrases):
            return slot
    if "my favorite" in normalized:
        tail = normalized.partition("my favorite")[2]
        return "favorite:" + " ".join(tail.split()[:2])
    return ""


def _find_memory(
    memories: tuple[Memory, ...], text: str, subject: str,
) -> Memory | None:
    from app.core.retrieval import same_memory_fact

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


def derive_memory_change(
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
    if not memory_report_is_supported(source, personal=True) or not (
        _MEMORY_CUES.search(normalized)
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
        recall_role='',
        recall_cues=(),
    )
    return MemoryChange("upsert", memory, current.id)
