"""Small reply cleanup helpers."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
import string

from app.core.generation import clean_model_text, collapse_hidden_tag_gaps, finalize_model_response
from app.core.request_analysis import RequestAnalysis

_EMPTY_REPLY_SENTINEL = "__AKANE_EMPTY_REPLY__"
_PUNCT = str.maketrans({ch: " " for ch in string.punctuation})
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_FOLLOWUP_STARTS = {"how", "what", "why", "when", "where", "who", "can", "could", "would", "should", "do", "does", "is", "are", "want"}
_FILLER = ("mmm", "mm", "hmm", "hm", "ah", "oh", "heh")


def _collapse_visible(text: str) -> str:
    return collapse_hidden_tag_gaps(str(text or "")).strip()


def clean_reply_text(raw: str) -> str:
    return _collapse_visible(clean_model_text(raw)) if raw else ""


def finalize_reply(raw: str, visible_fallback: str = "") -> str:
    cleaned = _collapse_visible(finalize_model_response(raw)) if raw else ""
    if cleaned:
        return cleaned
    fallback = _collapse_visible(visible_fallback)
    return fallback or _EMPTY_REPLY_SENTINEL


def normalize_final_reply(reply: str) -> str:
    value = str(reply or "").replace(_EMPTY_REPLY_SENTINEL, "").strip()
    return value or "I hit a snag on that. Try me one more time."


def _split_sentences(text: str, max_parts: int | None = None) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_RE.split(str(text or "").strip()) if part.strip()]
    return parts[:max_parts] if max_parts is not None else parts


def _norm(text: str) -> str:
    return " ".join(collapse_hidden_tag_gaps(str(text or "")).lower().translate(_PUNCT).split())


def _similar(left: str, right: str) -> float:
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if (2 * min(len(a), len(b))) / (len(a) + len(b)) < 0.82:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _strip_repeated_sentences_against_previous(reply: str, previous_reply: str) -> str:
    text = str(reply or "").strip()
    previous = str(previous_reply or "").strip()
    if not text or not previous:
        return text
    previous_parts = _split_sentences(previous)
    kept = [part for part in _split_sentences(text) if not any(_similar(part, old) >= 0.9 for old in previous_parts)]
    return " ".join(kept).strip() if kept and len(kept) != len(_split_sentences(text)) else text


def _strip_filler(reply: str) -> str:
    text = str(reply or "").strip()
    lower = text.lower()
    for opener in _FILLER:
        if lower == opener or lower.startswith(opener + " ") or lower.startswith(opener + "."):
            text = text[len(opener):].lstrip(" .,!?:;")
            return text[:1].upper() + text[1:] if text else ""
    return text


def _strip_trailing_followup_question(reply: str) -> str:
    parts = _split_sentences(reply)
    if not parts or not parts[-1].endswith("?"):
        return str(reply or "").strip()
    first = parts[-1].lower().lstrip().split(" ", 1)[0]
    if first not in _FOLLOWUP_STARTS:
        return str(reply or "").strip()
    if len(parts) == 1:
        return parts[-1].rstrip("?").rstrip() + "."
    return " ".join(parts[:-1]).strip()


def _compact(text: str, *, sentences: int = 3) -> str:
    parts = _split_sentences(text, max_parts=sentences + 1)
    if len(parts) > sentences:
        text = " ".join(parts[:sentences]).strip()
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if len(lines) > sentences:
        text = " ".join(lines[:sentences])
    return _strip_trailing_followup_question(text)


def _strip_robotic_review_prefix(reply: str) -> str:
    text = str(reply or "").strip()
    lowered = text.lower()
    for prefix in ("after reviewing ", "after inspecting ", "after looking at ", "based on "):
        if lowered.startswith(prefix):
            cut = len(prefix)
            return text[cut:cut + 1].upper() + text[cut + 1:].lstrip()
    return text


def _strip_coding_preface(reply: str) -> str:
    lines = str(reply or "").splitlines()
    if not lines:
        return ""
    lowered = lines[0].lower()
    if "direct coding response" in lowered or "no file operations needed" in lowered or "no editor operations needed" in lowered:
        return "\n".join(lines[1:]).strip()
    return reply


def _normalize_reply_seed(reply: str) -> str:
    return _strip_coding_preface(_strip_robotic_review_prefix(_strip_filler(_collapse_visible(reply))))


def postprocess_reply(
    reply: str,
    *,
    analysis: RequestAnalysis,
    compact_mode: bool = True,
    previous_assistant_reply: str = "",
) -> str:
    cleaned = _strip_repeated_sentences_against_previous(_normalize_reply_seed(reply), previous_assistant_reply)
    if not compact_mode:
        return cleaned
    if getattr(analysis, "wants_single_suggestion", False):
        return _compact(cleaned, sentences=1)
    if getattr(analysis, "wants_detail", False):
        return cleaned
    if len(cleaned) > 280 or cleaned.count("\n") > 4:
        return _compact(cleaned, sentences=3)
    return _strip_trailing_followup_question(cleaned)
