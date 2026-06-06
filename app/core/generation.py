"""Small streaming cleanup helpers used by the chat server."""

from __future__ import annotations

from app.core.config import MAX_TOKENS, REPETITION_PENALTY, TEMPERATURE, TOP_K, TOP_P
from app.core.model_loader import ModelManager, content_to_text

STREAM_HIDDEN_TAGS = {
    "MEM": "[/MEM]",
    "FORGET": "[/FORGET]",
    "PROJECT": "[/PROJECT]",
    "EDITOR": "[/EDITOR]",
    "ASK_CODER": "[/ASK_CODER]",
    "CODE": "[/CODE]",
    "READ": "[/READ]",
    "WRITE": "[/WRITE]",
    "SHELL": "[/SHELL]",
    "READ_RESULT": "[/READ_RESULT]",
    "THINK": "[/THINK]",
}
STREAM_HIDDEN_XML_TAGS = {
    "think": "</think>",
    "thinking": "</thinking>",
    "tool_call": "</tool_call>",
    "read": "</read>",
    "write": "</write>",
    "shell": "</shell>",
    "code": "</code>",
    "read_result": "</read_result>",
}
_OPENERS = tuple(
    sorted(
        [(f"[{name}]", close) for name, close in STREAM_HIDDEN_TAGS.items()]
        + [(f"<{name}>", close) for name, close in STREAM_HIDDEN_XML_TAGS.items()],
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)
_OPENERS_LOWER = tuple((opener.lower(), close.lower(), opener, close) for opener, close in _OPENERS)
_OPENER_PREFIXES = frozenset(opener[:i].lower() for opener, _ in _OPENERS for i in range(1, len(opener) + 1))
_MAX_OPENER_LEN = max(len(opener) for opener, _ in _OPENERS)


def collapse_hidden_tag_gaps(text: str) -> str:
    if not text:
        return text or ""
    value = str(text).replace("\r\n", "\n").replace("\t", " ")
    lines: list[str] = []
    previous_blank = False
    for raw in value.split("\n"):
        line = raw.rstrip()
        if line:
            lines.append(line)
            previous_blank = False
        elif not previous_blank:
            lines.append("")
            previous_blank = True
    return "\n".join(lines)


def strip_emoji_chars(text: str) -> str:
    if not text:
        return text or ""
    value = str(text)
    if value.isascii():
        return value
    out: list[str] = []
    for char in value:
        code = ord(char)
        if (
            0x1F000 <= code <= 0x1FAFF
            or 0x2600 <= code <= 0x27BF
            or 0xFE00 <= code <= 0xFE0F
            or code in {0x200D, 0x20E3, 0x00A9, 0x00AE, 0x2122, 0x3030, 0x303D, 0x3297, 0x3299}
        ):
            continue
        out.append(char)
    return "".join(out)


def strip_hidden_blocks(text: str) -> str:
    if not text:
        return text or ""
    source = str(text)
    if "[" not in source and "<" not in source:
        return source
    source_lower = source.lower()
    pieces: list[str] = []
    start = 0
    while start < len(source):
        match: tuple[int, str, str] | None = None
        for opener_lower, close_lower, opener, _close in _OPENERS_LOWER:
            idx = source_lower.find(opener_lower, start)
            if idx >= 0 and (match is None or idx < match[0]):
                match = (idx, opener, close_lower)
        if match is None:
            pieces.append(source[start:])
            break
        idx, opener, close = match
        pieces.append(source[start:idx])
        end = source_lower.find(close, idx + len(opener))
        if end < 0:
            break
        start = end + len(close)
    return "".join(pieces)


class HiddenTagStreamFilter:
    """Remove hidden/tool tags from token streams."""

    def __init__(self) -> None:
        self._buffer = ""
        self._hidden_close = ""
        self._hidden_close_lower = ""

    @staticmethod
    def _hold_prefix(buffer: str) -> int:
        lower = buffer.lower()
        best = 0
        for i in range(1, min(len(lower), _MAX_OPENER_LEN) + 1):
            if lower[-i:] in _OPENER_PREFIXES:
                best = i
        return best

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._buffer += str(text)
        if not self._hidden_close and "[" not in self._buffer and "<" not in self._buffer:
            out = self._buffer
            self._buffer = ""
            return out
        out: list[str] = []

        while self._buffer:
            lower = self._buffer.lower()
            if self._hidden_close:
                close = self._hidden_close_lower
                idx = lower.find(close)
                if idx == -1:
                    keep = max(len(close) - 1, 0)
                    self._buffer = self._buffer[-keep:] if keep else ""
                    return "".join(out)
                self._buffer = self._buffer[idx + len(self._hidden_close):]
                self._hidden_close = ""
                self._hidden_close_lower = ""
                continue

            match: tuple[int, str, str] | None = None
            for opener_lower, _close_lower, opener, close in _OPENERS_LOWER:
                idx = lower.find(opener_lower)
                if idx < 0:
                    continue
                if match is None or idx < match[0]:
                    match = (idx, opener, close)

            if match is None:
                hold = self._hold_prefix(self._buffer)
                if hold:
                    out.append(self._buffer[:-hold])
                    self._buffer = self._buffer[-hold:]
                else:
                    out.append(self._buffer)
                    self._buffer = ""
                break

            idx, opener, close = match
            if idx:
                out.append(self._buffer[:idx])
            self._buffer = self._buffer[idx + len(opener):]
            self._hidden_close = close
            self._hidden_close_lower = close.lower()

        return "".join(out)

    def flush(self) -> str:
        if self._hidden_close:
            self._buffer = ""
            self._hidden_close = ""
            self._hidden_close_lower = ""
            return ""
        out = self._buffer
        self._buffer = ""
        return out


def _request(messages: list[dict], *, max_tokens: int, stream: bool):
    return ModelManager.get_instance().create_chat_completion(
        messages=messages,
        max_tokens=max(1, min(int(max_tokens), MAX_TOKENS)),
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repeat_penalty=REPETITION_PENALTY,
        stream=stream,
    )


def completion_text(messages: list[dict], *, max_tokens: int) -> str:
    result = _request(messages, max_tokens=max_tokens, stream=False)
    choices = result.get("choices") or []
    if not choices:
        return ""
    return content_to_text((choices[0].get("message") or {}).get("content"))


def stream_completion(messages: list[dict], *, max_tokens: int):
    for chunk in _request(messages, max_tokens=max_tokens, stream=True):
        choices = chunk.get("choices") or []
        if not choices:
            continue
        text = content_to_text((choices[0].get("delta") or {}).get("content"))
        if text:
            yield text


def clean_visible_text(text: str) -> str:
    source = str(text or "")
    cleaned = strip_hidden_blocks(source)
    if cleaned != source:
        cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines() if line.strip())
    cleaned = strip_emoji_chars(cleaned)
    return collapse_hidden_tag_gaps(cleaned).strip(" `\n\t")
