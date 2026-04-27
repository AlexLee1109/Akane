"""Reply cleanup and shaping helpers for Akane."""

from __future__ import annotations

import re
from collections.abc import Callable

from app.generation import clean_model_text, collapse_hidden_tag_gaps, finalize_model_response
from app.request_analysis import RequestAnalysis

_INCOMPLETE_REPLY_PATTERN = re.compile(
    r"(?:[`([{:/\\-]\s*$|\b(?:examine|inspect|open|read|look at|check)\s+`?\s*$)",
    re.IGNORECASE,
)
_TRAILING_CONTINUATION_PATTERN = re.compile(
    r"\b(?:to|for|with|by|and|or|if|when|while|because|so|then|into|from|that|which|where|using|like)\s*$",
    re.IGNORECASE,
)
_CODE_DUMP_HINT_PATTERN = re.compile(
    r"^\s*(?:from |import |def |class |return |if |elif |else:|for |while |try:|except |with |const |let |var |function |\{|\}|\]|\)|<\?|\#include)\b",
    re.IGNORECASE,
)
_FOLLOWUP_QUESTION_START_PATTERN = re.compile(
    r"^(?:how|what|why|when|where|who|anything|any|did|do|does|is|are|would|could|should|want|wanna|need)\b",
    re.IGNORECASE,
)
_FOLLOWUP_QUESTION_PHRASE_PATTERN = re.compile(
    r"\b(?:how(?:'s| is)|what(?:'s| is) new|anything exciting|how (?:is|was) your day|what have you been up to|how are you doing)\b",
    re.IGNORECASE,
)
_FILLER_OPENER_PATTERN = re.compile(
    r"^\s*(?:m+m+(?:[.!?…]+)?|mm+(?:[.!?…]+)?|h+m+(?:[.!?…]+)?|hmm+(?:[.!?…]+)?|ah+(?:[.!?…]+)?|oh+(?:[.!?…]+)?|heh+(?:[.!?…]+)?)\s*",
    re.IGNORECASE,
)
_EMPTY_REPLY_SENTINEL = "__AKANE_EMPTY_REPLY__"


def clean_reply_text(raw: str) -> str:
    cleaned = clean_model_text(raw) if raw else ""
    return collapse_hidden_tag_gaps(cleaned).strip() if cleaned else ""


def finalize_reply(raw: str, visible_fallback: str = "") -> str:
    cleaned = finalize_model_response(raw) if raw else ""
    cleaned = collapse_hidden_tag_gaps(cleaned).strip() if cleaned else ""
    if cleaned:
        return cleaned

    visible_fallback = collapse_hidden_tag_gaps(visible_fallback).strip() if visible_fallback else ""
    if visible_fallback:
        return visible_fallback

    return _EMPTY_REPLY_SENTINEL


def normalize_final_reply(reply: str) -> str:
    reply = str(reply or "").replace(_EMPTY_REPLY_SENTINEL, "").strip()
    if not reply:
        return "I hit a snag on that. Try me one more time."
    return reply


def merge_visible_reply(prefix: str, continuation: str) -> str:
    prefix_clean = collapse_hidden_tag_gaps(str(prefix or "").replace(_EMPTY_REPLY_SENTINEL, "")).strip()
    continuation_clean = collapse_hidden_tag_gaps(
        str(continuation or "").replace(_EMPTY_REPLY_SENTINEL, "")
    ).strip()
    if not prefix_clean:
        return continuation_clean
    if not continuation_clean:
        return prefix_clean

    prefix_norm = " ".join(prefix_clean.split())
    continuation_norm = " ".join(continuation_clean.split())
    if continuation_norm == prefix_norm:
        return prefix_clean
    if continuation_norm.startswith(prefix_norm):
        return continuation_clean
    if prefix_norm.startswith(continuation_norm):
        return prefix_clean
    if continuation_clean[0] in ",.;:!?)":
        return f"{prefix_clean}{continuation_clean}"
    return f"{prefix_clean} {continuation_clean}"


def _looks_incomplete_reply(text: str) -> bool:
    stripped = str(text or "").rstrip()
    if not stripped:
        return True
    if stripped.endswith("```"):
        return True
    if stripped.count("```") % 2 == 1:
        return True
    if stripped.count("`") % 2 == 1:
        return True
    if _TRAILING_CONTINUATION_PATTERN.search(stripped):
        return True
    if stripped[-1] not in ".!?)]}\"'" and len(stripped.split()) >= 4:
        return True
    if len(stripped) > 120 and stripped[-1] not in ".!?)]}\"'":
        return True
    return bool(_INCOMPLETE_REPLY_PATTERN.search(stripped))


def _needs_visible_retry(reply: str, finish_reason: str = "") -> bool:
    return (
        finish_reason == "length"
        or reply == _EMPTY_REPLY_SENTINEL
        or _looks_incomplete_reply(reply)
    )


def _retry_messages_for_visible_reply(chat_messages: list[dict], partial_reply: str = "") -> list[dict]:
    if partial_reply:
        retry_instruction = (
            "Your last visible reply was cut off. Continue from where you left off with one short complete continuation. "
            "Do not restart from the beginning. Do not emit only tags, metadata, or an empty message."
        )
        return [
            *chat_messages,
            {"role": "assistant", "content": partial_reply},
            {"role": "system", "content": retry_instruction},
        ]

    retry_instruction = (
        "Please answer the user with at least one short complete visible sentence. "
        "Do not cut off mid-sentence. Do not reply with only tags, metadata, or an empty message."
    )
    return [*chat_messages, {"role": "system", "content": retry_instruction}]


def ensure_complete_visible_reply(
    base_messages: list[dict],
    reply: str,
    *,
    finish_reason: str = "",
    request_completion: Callable[[list[dict]], dict],
    extract_message_text: Callable[[dict], str],
    extract_finish_reason: Callable[[dict], str],
) -> str:
    cleaned = str(reply or "").strip()
    current_finish_reason = str(finish_reason or "").strip().lower()
    attempts = 0

    while attempts < 2 and _needs_visible_retry(cleaned, current_finish_reason):
        retry_result = request_completion(_retry_messages_for_visible_reply(base_messages, cleaned))
        retry_raw = extract_message_text(retry_result)
        current_finish_reason = extract_finish_reason(retry_result)
        cleaned = merge_visible_reply(cleaned, finalize_reply(retry_raw))
        attempts += 1

    return cleaned


def _looks_like_raw_tool_dump(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if "[READ_RESULT]" in stripped or "[/READ_RESULT]" in stripped:
        return True
    lines = [line.rstrip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 10:
        code_like = sum(1 for line in lines if _CODE_DUMP_HINT_PATTERN.match(line))
        if code_like >= 3:
            return True
    return len(stripped) > 1400


def _needs_compact_summary(analysis: RequestAnalysis, reply: str) -> bool:
    if analysis.wants_detail:
        return False
    if _looks_like_raw_tool_dump(reply):
        return True
    return len(str(reply or "").strip()) > 420 or str(reply or "").count("\n") > 5


def _looks_like_single_suggestion_reply(reply: str) -> bool:
    text = str(reply or "").strip()
    if not text:
        return False
    if len(text) > 340 or text.count("\n") > 3:
        return False
    if re.search(r"^\s*[-*]\s+", text, re.MULTILINE):
        return False
    if len(re.findall(r"(?:^|\n)\s*\d+[.)]\s+", text)) > 2:
        return False
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return 1 <= len(sentences) <= 2


def _looks_like_companion_sized_reply(reply: str) -> bool:
    text = collapse_hidden_tag_gaps(str(reply or "")).strip()
    if not text:
        return False
    if len(text) > 260 or text.count("\n") > 2:
        return False
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return 1 <= len(sentences) <= 2


def _strip_trailing_followup_question(reply: str) -> str:
    text = collapse_hidden_tag_gaps(str(reply or "")).strip()
    if not text:
        return ""

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        return text

    last_sentence = sentences[-1]
    if not last_sentence.endswith("?"):
        return text
    if not (
        _FOLLOWUP_QUESTION_START_PATTERN.match(last_sentence)
        or _FOLLOWUP_QUESTION_PHRASE_PATTERN.search(last_sentence)
    ):
        return text
    if len(sentences) == 1:
        return re.sub(r"\?\s*$", ".", last_sentence).strip()

    kept = " ".join(sentences[:-1]).strip()
    return kept or text


def _strip_filler_opener(reply: str) -> str:
    text = collapse_hidden_tag_gaps(str(reply or "")).strip()
    if not text:
        return ""
    return _FILLER_OPENER_PATTERN.sub("", text, count=1).strip()


def _clamp_suggestion_reply(reply: str) -> str:
    text = _strip_filler_opener(reply)
    if not text:
        return ""

    numbered = re.findall(r"(?:^|\n)\s*(\d+[.)]\s+.*?)(?=(?:\n\s*\d+[.)]\s+)|\Z)", text, re.DOTALL)
    if numbered:
        return "\n".join(item.strip() for item in numbered[:2])

    bulleted = re.findall(r"(?:^|\n)\s*[-*]\s+(.*?)(?=(?:\n\s*[-*]\s+)|\Z)", text, re.DOTALL)
    if bulleted:
        return "\n".join(f"- {item.strip()}" for item in bulleted[:2])

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) > 2:
        text = " ".join(sentences[:2]).strip()
    return _strip_trailing_followup_question(text)


def _clamp_companion_reply(reply: str) -> str:
    text = _strip_filler_opener(reply)
    if not text:
        return ""

    numbered = re.findall(r"(?:^|\n)\s*(\d+[.)]\s+.*?)(?=(?:\n\s*\d+[.)]\s+)|\Z)", text, re.DOTALL)
    if numbered:
        text = " ".join(item.strip() for item in numbered[:2]).strip()
    else:
        bulleted = re.findall(r"(?:^|\n)\s*[-*]\s+(.*?)(?=(?:\n\s*[-*]\s+)|\Z)", text, re.DOTALL)
        if bulleted:
            text = " ".join(item.strip() for item in bulleted[:2]).strip()

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) > 2:
        text = " ".join(sentences[:2]).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 2:
        text = " ".join(lines[:2]).strip()

    return _strip_trailing_followup_question(text)


def _locally_compact_reply(reply: str) -> str:
    text = _strip_filler_opener(reply)
    if not text:
        return ""
    if _looks_like_raw_tool_dump(text):
        clamped = _clamp_suggestion_reply(text)
        if clamped:
            return clamped
    return _clamp_companion_reply(text)


def _retry_messages_for_concise_reply(chat_messages: list[dict], verbose_reply: str) -> list[dict]:
    retry_instruction = (
        "Rewrite your last reply as a compact desktop-companion response. "
        "Use 1-2 short sentences only. "
        "Prefer a single short sentence unless a second sentence is required to finish the thought. "
        "Do not print raw code, raw file contents, READ_RESULT blocks, or long bullet lists. "
        "Keep only the most important findings in plain conversational language. "
        "Stop after the second sentence."
    )
    return [
        *chat_messages,
        {"role": "assistant", "content": verbose_reply},
        {"role": "system", "content": retry_instruction},
    ]


def _retry_messages_for_single_suggestion(chat_messages: list[dict], verbose_reply: str) -> list[dict]:
    retry_instruction = (
        "Rewrite your last reply for a tiny desktop companion bubble. "
        "Give one or two best suggestions, not a long list. "
        "Use 1-2 short sentences only. "
        "Say what to change and why it matters in plain language. "
        "Do not include raw code, file dumps, bullet lists, or more than two recommendations. "
        "Stop after the second sentence."
    )
    return [
        *chat_messages,
        {"role": "assistant", "content": verbose_reply},
        {"role": "system", "content": retry_instruction},
    ]


def _rewrite_reply(
    working_messages: list[dict],
    retry_messages_builder: Callable[[list[dict], str], list[dict]],
    reply: str,
    *,
    request_completion: Callable[[list[dict]], dict],
    extract_message_text: Callable[[dict], str],
    extract_finish_reason: Callable[[dict], str],
) -> str:
    rewrite_result = request_completion(retry_messages_builder(working_messages, reply))
    rewrite_raw = extract_message_text(rewrite_result)
    rewrite_finish_reason = extract_finish_reason(rewrite_result)
    rewrite_cleaned = finalize_reply(rewrite_raw)
    return ensure_complete_visible_reply(
        working_messages,
        rewrite_cleaned,
        finish_reason=rewrite_finish_reason,
        request_completion=request_completion,
        extract_message_text=extract_message_text,
        extract_finish_reason=extract_finish_reason,
    )


def postprocess_reply(
    reply: str,
    *,
    analysis: RequestAnalysis,
    working_messages: list[dict],
    request_completion: Callable[[list[dict]], dict],
    extract_message_text: Callable[[dict], str],
    extract_finish_reason: Callable[[dict], str],
) -> str:
    cleaned = _strip_filler_opener(reply)

    if analysis.wants_single_suggestion:
        locally_clamped = _clamp_suggestion_reply(cleaned) or cleaned
        if _looks_like_single_suggestion_reply(locally_clamped):
            return locally_clamped
        rewritten = _rewrite_reply(
            working_messages,
            _retry_messages_for_single_suggestion,
            cleaned,
            request_completion=request_completion,
            extract_message_text=extract_message_text,
            extract_finish_reason=extract_finish_reason,
        )
        return _clamp_companion_reply(_clamp_suggestion_reply(rewritten) or rewritten or locally_clamped)

    if _needs_compact_summary(analysis, cleaned):
        locally_compacted = _locally_compact_reply(cleaned)
        if locally_compacted and not _needs_compact_summary(analysis, locally_compacted):
            return _clamp_companion_reply(locally_compacted)
        rewritten = _rewrite_reply(
            working_messages,
            _retry_messages_for_concise_reply,
            cleaned,
            request_completion=request_completion,
            extract_message_text=extract_message_text,
            extract_finish_reason=extract_finish_reason,
        )
        if rewritten:
            return _clamp_companion_reply(rewritten)
        if locally_compacted:
            return _clamp_companion_reply(locally_compacted)

    if not analysis.wants_detail and _looks_like_companion_sized_reply(cleaned):
        return _clamp_companion_reply(cleaned)
    return _clamp_companion_reply(cleaned)
