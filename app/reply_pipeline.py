"""Reply cleanup and shaping helpers for Akane."""

from __future__ import annotations

from collections.abc import Callable

from app.generation import clean_model_text, collapse_hidden_tag_gaps, finalize_model_response
from app.request_analysis import RequestAnalysis

_EMPTY_REPLY_SENTINEL = "__AKANE_EMPTY_REPLY__"
_TRAILING_CONTINUATIONS = {
    "to", "for", "with", "by", "and", "or", "if", "when", "while", "because", "so", "then",
    "into", "from", "that", "which", "where", "using", "like",
}
_CODE_DUMP_PREFIXES = (
    "from ", "import ", "def ", "class ", "return ", "if ", "elif ", "else:", "for ", "while ",
    "try:", "except ", "with ", "const ", "let ", "var ", "function ", "{", "}", "]", ")", "<?", "#include",
)
_FOLLOWUP_QUESTION_STARTERS = {
    "how", "what", "why", "when", "where", "who", "anything", "any", "did", "do", "does",
    "is", "are", "would", "could", "should", "want", "wanna", "need",
}
_FOLLOWUP_QUESTION_PHRASES = (
    "how's", "how is", "what's new", "what is new", "anything exciting",
    "how is your day", "how was your day", "what have you been up to", "how are you doing",
)
_FILLER_OPENERS = ("mmm", "mm", "hmm", "hm", "ah", "oh", "heh")
_ROBOTIC_REVIEW_PREFIXES = (
    "after reviewing ", "after inspecting ", "after looking at ", "inspected ", "reviewed ", "based on ",
)
_CODING_PREFACE_MARKERS = (
    "direct coding response",
    "no editor or file operations needed",
    "no file operations needed",
    "no editor operations needed",
)


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


def _split_sentences(text: str) -> list[str]:
    stripped = str(text or "").strip()
    if not stripped:
        return []
    parts: list[str] = []
    start = 0
    for idx, ch in enumerate(stripped):
        if ch not in ".!?":
            continue
        end = idx + 1
        while end < len(stripped) and stripped[end].isspace():
            end += 1
        part = stripped[start:end].strip()
        if part:
            parts.append(part)
        start = end
    tail = stripped[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _extract_numbered_items(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        head = line.split(" ", 1)[0] if " " in line else line
        numbered = head[:-1].isdigit() and head.endswith((".", ")"))
        if numbered:
            if current:
                items.append(" ".join(current).strip())
            current = [line]
        elif current:
            current.append(line)
    if current:
        items.append(" ".join(current).strip())
    return items


def _extract_bulleted_items(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bulleted = line.startswith("- ") or line.startswith("* ")
        if bulleted:
            if current:
                items.append(" ".join(current).strip())
            current = [line[2:].strip()]
        elif current:
            current.append(line)
    if current:
        items.append(" ".join(current).strip())
    return items


def _normalize_reply_seed(reply: str) -> str:
    return _strip_coding_preface(_strip_robotic_review_prefix(_strip_filler_opener(reply)))


def _ends_with_incomplete_marker(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    trailing_chars = "`([{:/\\-"
    if stripped[-1] in trailing_chars:
        return True
    lowered = stripped.lower().rstrip("` ").rstrip()
    for phrase in ("examine", "inspect", "open", "read", "look at", "check"):
        if lowered.endswith(f" {phrase}") or lowered == phrase:
            return True
    return False


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
    last_word = stripped.rstrip(".,;:!?)]}\"' ").split()[-1].lower() if stripped.split() else ""
    if last_word in _TRAILING_CONTINUATIONS:
        return True
    if stripped[-1] not in ".!?)]}\"'" and len(stripped.split()) >= 4:
        return True
    if len(stripped) > 120 and stripped[-1] not in ".!?)]}\"'":
        return True
    return _ends_with_incomplete_marker(stripped)


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
        code_like = sum(1 for line in lines if line.lstrip().lower().startswith(_CODE_DUMP_PREFIXES))
        if code_like >= 3:
            return True
    return len(stripped) > 1400


def _needs_compact_summary(analysis: RequestAnalysis, reply: str) -> bool:
    if analysis.wants_detail:
        return False
    if _looks_like_raw_tool_dump(reply):
        return True
    text = str(reply or "").strip()
    if analysis.coding_like or analysis.codebase_context:
        return len(text) > 240 or text.count("\n") > 3
    return len(text) > 280 or text.count("\n") > 4


def _looks_like_single_suggestion_reply(reply: str) -> bool:
    text = str(reply or "").strip()
    if not text:
        return False
    if len(text) > 340 or text.count("\n") > 3:
        return False
    if any(line.lstrip().startswith(("- ", "* ")) for line in text.splitlines()):
        return False
    if len(_extract_numbered_items(text)) > 1:
        return False
    sentences = _split_sentences(text)
    return 1 <= len(sentences) <= 3


def _looks_like_companion_sized_reply(reply: str) -> bool:
    text = collapse_hidden_tag_gaps(str(reply or "")).strip()
    if not text:
        return False
    if len(text) > 240 or text.count("\n") > 2:
        return False
    sentences = _split_sentences(text)
    return 1 <= len(sentences) <= 3


def _strip_trailing_followup_question(reply: str) -> str:
    text = collapse_hidden_tag_gaps(str(reply or "")).strip()
    if not text:
        return ""

    sentences = _split_sentences(text)
    if not sentences:
        return text

    last_sentence = sentences[-1]
    if not last_sentence.endswith("?"):
        return text
    last_lower = last_sentence.lower()
    last_start = last_lower.lstrip().split(" ", 1)[0]
    if not (
        last_start in _FOLLOWUP_QUESTION_STARTERS
        or any(phrase in last_lower for phrase in _FOLLOWUP_QUESTION_PHRASES)
    ):
        return text
    if len(sentences) == 1:
        return last_sentence.rstrip().rstrip("?").rstrip() + "."

    kept = " ".join(sentences[:-1]).strip()
    return kept or text


def _strip_filler_opener(reply: str) -> str:
    text = collapse_hidden_tag_gaps(str(reply or "")).strip()
    if not text:
        return ""
    lowered = text.lower().lstrip()
    leading_ws_len = len(text) - len(text.lstrip())
    for opener in _FILLER_OPENERS:
        if not lowered.startswith(opener):
            continue
        idx = leading_ws_len + len(opener)
        while idx < len(text) and text[idx] in ".!?… ":
            idx += 1
        return text[idx:].strip()
    return text


def _strip_robotic_review_prefix(reply: str) -> str:
    text = collapse_hidden_tag_gaps(str(reply or "")).strip()
    if not text:
        return ""
    lines = text.splitlines()
    if lines:
        first_line = lines[0].strip().lower()
        if any(first_line.startswith(prefix) for prefix in _ROBOTIC_REVIEW_PREFIXES):
            if "." in first_line or first_line.endswith("suggestions:") or first_line.endswith("suggestion:"):
                text = "\n".join(lines[1:]).strip()

    lowered = text.lower().lstrip()
    stripped = text
    if lowered.startswith("noted unused imports"):
        first_period = stripped.find(".")
        stripped = stripped[first_period + 1 :].strip() if first_period != -1 else ""
    elif lowered.startswith("(") and ") and " in lowered:
        _, _, rest = stripped.partition(") and ")
        stripped = rest.strip()
    lowered = stripped.lower().lstrip()
    for prefix in _ROBOTIC_REVIEW_PREFIXES:
        if not lowered.startswith(prefix):
            continue
        cut = len(stripped) - len(stripped.lstrip()) + len(prefix)
        stripped = stripped[cut:].lstrip()
        while stripped[:1] in ".,:":
            stripped = stripped[1:].lstrip()
        if stripped.lower().startswith("suggestions:") or stripped.lower().startswith("suggestion:"):
            stripped = stripped.split(":", 1)[1].lstrip()
        break
    if stripped and stripped != text:
        return stripped[0].upper() + stripped[1:] if len(stripped) > 1 else stripped.upper()
    return text


def _strip_coding_preface(reply: str) -> str:
    text = collapse_hidden_tag_gaps(str(reply or "")).strip()
    if not text:
        return ""
    lines = text.splitlines()
    if not lines:
        return text
    first_line = lines[0].strip()
    lowered = first_line.lower()
    if not any(marker in lowered for marker in _CODING_PREFACE_MARKERS):
        return text
    rest = "\n".join(lines[1:]).strip()
    if rest:
        return rest
    for marker in _CODING_PREFACE_MARKERS:
        idx = lowered.find(marker)
        if idx != -1:
            candidate = first_line[idx + len(marker):].lstrip(" -:.")
            if candidate:
                return candidate
    return ""


def _clamp_suggestion_reply(reply: str) -> str:
    text = _normalize_reply_seed(reply)
    if not text:
        return ""

    numbered = _extract_numbered_items(text)
    if numbered:
        return numbered[0].strip()

    bulleted = _extract_bulleted_items(text)
    if bulleted:
        return f"- {bulleted[0].strip()}"

    sentences = _split_sentences(text)
    if len(sentences) > 3:
        text = " ".join(sentences[:3]).strip()
    return _strip_trailing_followup_question(text)


def _clamp_companion_reply(reply: str) -> str:
    text = _normalize_reply_seed(reply)
    if not text:
        return ""

    numbered = _extract_numbered_items(text)
    if numbered:
        text = " ".join(item.strip() for item in numbered[:3]).strip()
    else:
        bulleted = _extract_bulleted_items(text)
        if bulleted:
            text = " ".join(item.strip() for item in bulleted[:3]).strip()

    sentences = _split_sentences(text)
    if len(sentences) > 3:
        text = " ".join(sentences[:3]).strip()

    lines = _nonempty_lines(text)
    if len(lines) > 3:
        text = " ".join(lines[:3]).strip()

    return _strip_trailing_followup_question(text)


def _locally_compact_reply(reply: str) -> str:
    text = _normalize_reply_seed(reply)
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
        "Use 2-3 short sentences max, and prefer 1-2 when possible. "
        "Keep the total reply short, ideally under about 220 characters when possible. "
        "Shorten overly long sentences instead of packing too much into one sentence. "
        "Do not print raw code, raw file contents, READ_RESULT blocks, or long bullet lists. "
        "Keep only the most important findings in plain conversational language. "
        "Stop after the third sentence."
    )
    return [
        *chat_messages,
        {"role": "assistant", "content": verbose_reply},
        {"role": "system", "content": retry_instruction},
    ]


def _retry_messages_for_single_suggestion(chat_messages: list[dict], verbose_reply: str) -> list[dict]:
    retry_instruction = (
        "Rewrite your last reply for a tiny desktop companion bubble. "
        "Give only the single best suggestion, not a list. "
        "Use 2-3 short sentences max, and prefer 1-2 when possible. "
        "Keep the total reply short, ideally under about 220 characters when possible. "
        "Shorten overly long sentences instead of packing too much into one sentence. "
        "Say what to change and why it matters in plain language. "
        "Do not include raw code, file dumps, bullet lists, or more than one recommendation. "
        "Stop after the third sentence."
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
    cleaned = _normalize_reply_seed(reply)

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
