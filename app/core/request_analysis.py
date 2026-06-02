"""Tiny chat request analyzer kept for reply post-processing tests."""

from __future__ import annotations

from dataclasses import dataclass

_DETAIL_WORDS = {"detail", "details", "explain", "walkthrough", "deep", "full"}
_SUGGEST_WORDS = {"suggest", "suggestion", "improve", "recommend"}
_FOLLOWUP_WORDS = {"it", "this", "that", "those", "them", "same", "still", "again"}
_AFFIRMATIVE = {"yes", "yeah", "yep", "sure", "ok", "okay", "please", "do it"}
_SMALLTALK = {"hi", "hello", "hey", "thanks", "thank", "bye", "cool", "nice"}


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for char in str(text or "").lower():
        if char.isalnum() or char in {"'", "_"}:
            current.append(char)
        elif current:
            out.append("".join(current).strip("'_"))
            current.clear()
    if current:
        out.append("".join(current).strip("'_"))
    return [token for token in out if token]


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    last_user: str = ""
    last_assistant: str = ""
    recent_text: str = ""


@dataclass(frozen=True, slots=True)
class RequestAnalysis:
    query_tokens: tuple[str, ...] = ()
    referential_followup: bool = False
    followup_reference: bool = False
    affirmative_followup: bool = False
    assistant_invited_continuation: bool = False
    topic_overlap: bool = False
    topic_shift: bool = False
    should_carry_last_reply: bool = False
    wants_detail: bool = False
    wants_single_suggestion: bool = False
    wants_brainstorm: bool = False
    smalltalk: bool = False


class RequestAnalyzer:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def extract_query_tokens(self, text: str) -> tuple[str, ...]:
        return tuple(token for token in _tokens(text) if len(token) > 2)

    def analyze(self, user_input: str, snapshot: RequestSnapshot | None = None) -> RequestAnalysis:
        snapshot = snapshot or RequestSnapshot()
        toks = _tokens(user_input)
        token_set = set(toks)
        query_tokens = self.extract_query_tokens(user_input)
        recent_tokens = set(self.extract_query_tokens(snapshot.recent_text or snapshot.last_user or snapshot.last_assistant))
        followup_reference = bool(token_set & _FOLLOWUP_WORDS)
        referential_followup = bool(token_set & {"it", "this", "that", "those", "them", "same"})
        affirmative_followup = " ".join(toks) in _AFFIRMATIVE
        topic_shift = bool(toks[:1] and toks[0] in {"also", "instead", "unrelated", "separately", "different"})
        topic_overlap = bool(set(query_tokens) & recent_tokens)
        assistant_invited = "?" in str(snapshot.last_assistant or "") and any(
            phrase in str(snapshot.last_assistant).lower()
            for phrase in ("want me", "should i", "i can", "would you like")
        )
        wants_detail = bool(token_set & _DETAIL_WORDS)
        wants_single_suggestion = bool(token_set & _SUGGEST_WORDS and not wants_detail)
        wants_brainstorm = bool("?" in str(user_input) and token_set & {"idea", "ideas", "feature", "build", "make"})
        smalltalk = bool(token_set and token_set <= _SMALLTALK)
        should_carry = bool(
            (snapshot.last_user or snapshot.last_assistant or snapshot.recent_text)
            and not topic_shift
            and (followup_reference or affirmative_followup or topic_overlap or assistant_invited)
        )
        return RequestAnalysis(
            query_tokens=query_tokens,
            referential_followup=referential_followup,
            followup_reference=followup_reference,
            affirmative_followup=affirmative_followup,
            assistant_invited_continuation=assistant_invited,
            topic_overlap=topic_overlap,
            topic_shift=topic_shift,
            should_carry_last_reply=should_carry,
            wants_detail=wants_detail,
            wants_single_suggestion=wants_single_suggestion,
            wants_brainstorm=wants_brainstorm,
            smalltalk=smalltalk,
        )
