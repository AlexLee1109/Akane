"""Small text and identity helpers; no psychological state lives here."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.core.config import SETTINGS

OWNER_PROFILE_ID = "local:owner"
_WORD = re.compile(r"[a-z0-9_+#./-]+", re.IGNORECASE)


def log_timing(scope: str, **durations: float) -> None:
    if not SETTINGS.timing_enabled:
        return
    values = " ".join(
        f"{name}_ms={max(0.0, float(duration)) * 1000:.3f}"
        for name, duration in durations.items()
    )
    print(f"[Akane:timing:{scope}] {values}", flush=True)


def log_performance(scope: str, **metrics: object) -> None:
    if not SETTINGS.timing_enabled:
        return
    values = " ".join(
        f"{name}={value:.3f}" if isinstance(value, float) else f"{name}={value}"
        for name, value in metrics.items()
    )
    print(f"[Akane:performance:{scope}] {values}", flush=True)


def compact_text(value: object, limit: int = 180) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or text[:limit]


def canonical_profile_id(value: object) -> str:
    profile = compact_text(value, 120) or OWNER_PROFILE_ID
    folded = profile.casefold()
    if folded in {OWNER_PROFILE_ID, "local", "popup", "discord:owner"}:
        return OWNER_PROFILE_ID
    if folded.startswith(("local:", "popup:", "discord:user:")):
        return OWNER_PROFILE_ID
    return profile


def lexical_terms(value: object) -> set[str]:
    terms: set[str] = set()
    for raw in _WORD.findall(str(value or "").casefold()):
        term = raw.strip(".")
        if len(term) <= 1:
            continue
        if len(term) > 5 and term.endswith("ing"):
            term = term[:-3]
        elif len(term) > 4 and term.endswith("ies"):
            term = term[:-3] + "y"
        elif len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
            term = term[:-1]
        terms.add(term)
    return terms


def text_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def relevance(query: object, candidate: object) -> float:
    query_terms = lexical_terms(query)
    candidate_terms = lexical_terms(candidate)
    shared = len(query_terms & candidate_terms)
    jaccard = shared / max(1, len(query_terms | candidate_terms))
    query_coverage = shared / max(1, min(len(query_terms), 4))
    return max(jaccard, query_coverage, SequenceMatcher(None, text_key(query), text_key(candidate)).ratio() * 0.35)
