"""Small streaming cleanup helpers used by the chat server."""

from __future__ import annotations

import hashlib

from app.core.character import get_static_system_prompt
from app.core.config import MAX_TOKENS, REPETITION_PENALTY, TEMPERATURE, TOP_K, TOP_P
from app.core.model_loader import ModelManager, content_to_text

_NO_THINK = "/no_think"
_PREFIX = {
    "model": None,
    "prompt": None,
    "hash": "",
    "tokens": [],
    "status": "empty",
    "disabled": False,
    "reused": False,
    "load_seconds": 0.0,
    "dynamic_chars": 0,
}
_PREFIX_WARNING = ""

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


def clear_static_state() -> None:
    global _PREFIX_WARNING
    _PREFIX.update({
        "model": None,
        "prompt": None,
        "hash": "",
        "tokens": [],
        "status": "empty",
        "disabled": False,
        "reused": False,
        "load_seconds": 0.0,
        "dynamic_chars": 0,
    })
    _PREFIX_WARNING = ""


def prefix_cache_status() -> dict[str, object]:
    return {
        "status": _PREFIX["status"],
        "reused": _PREFIX["reused"],
        "tokens": len(_PREFIX["tokens"]),
        "hash": str(_PREFIX["hash"])[:12],
        "dynamic_chars": _PREFIX["dynamic_chars"],
        "load_seconds": _PREFIX["load_seconds"],
    }


def _warn_once(reason: str) -> None:
    global _PREFIX_WARNING
    reason = str(reason or "unknown")[:80]
    if reason == _PREFIX_WARNING:
        return
    _PREFIX_WARNING = reason
    print(f"[Akane:prefix_cache] fallback reason={reason}", flush=True)


def _disable_cache(exc: Exception) -> None:
    _PREFIX.update({
        "tokens": [],
        "status": "disabled",
        "disabled": True,
        "reused": False,
        "load_seconds": 0.0,
    })
    _warn_once(type(exc).__name__ + ":" + str(exc))


def _ensure_static_state(llm) -> bool:
    static_prompt = get_static_system_prompt(include_memory=False)
    if (
        _PREFIX["model"] == id(llm)
        and _PREFIX["prompt"] is static_prompt
        and _PREFIX["tokens"]
    ):
        return True

    metadata = getattr(llm, "metadata", {}) or {}
    metadata_values = metadata.values() if isinstance(metadata, dict) else ()
    model_name = " ".join([
        str(getattr(llm, "model_path", "")),
        *map(str, metadata_values),
    ]).lower()
    if "qwen" not in model_name:
        raise RuntimeError("non_qwen_model")

    fingerprint = hashlib.sha256(static_prompt.encode("utf-8")).hexdigest()
    for name in ("reset", "eval", "tokenize"):
        if not hasattr(llm, name):
            raise RuntimeError(f"{name}_unavailable")
    tokens = llm.tokenize(
        f"<|im_start|>system\n{static_prompt}".encode("utf-8"),
        add_bos=False,
        special=True,
    )
    llm.reset()
    llm.eval(tokens)
    _PREFIX.update({
        "model": id(llm),
        "prompt": static_prompt,
        "hash": fingerprint,
        "tokens": tokens,
        "status": "built",
        "disabled": False,
        "reused": False,
        "load_seconds": 0.0,
    })
    return False


def _completion_kwargs(max_tokens: int, stream: bool) -> dict[str, object]:
    return {
        "max_tokens": max(1, min(int(max_tokens), MAX_TOKENS)),
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "repeat_penalty": REPETITION_PENALTY,
        "stream": stream,
    }


def _dynamic_system_text(tone_line: str, date_time_line: str) -> str:
    return "\n".join(
        line
        for value in (tone_line, date_time_line)
        if (line := str(value or "").strip())
    )


def _cached_request(
    llm,
    user_text: str,
    *,
    max_tokens: int,
    stream: bool,
    tone_line: str = "",
    date_time_line: str = "",
):
    reused = _ensure_static_state(llm)
    dynamic_system = _dynamic_system_text(tone_line, date_time_line)
    suffix = (
        "<|im_end|>\n"
        + (
            f"<|im_start|>system\n{dynamic_system}<|im_end|>\n"
            if dynamic_system
            else ""
        )
        + "<|im_start|>user\n"
        f"{user_text.rstrip()}\n\n{_NO_THINK}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    dynamic_tokens = llm.tokenize(
        suffix.encode("utf-8"),
        add_bos=False,
        special=True,
    )
    _PREFIX.update({
        "status": "hit" if reused else "built",
        "reused": reused,
        "load_seconds": 0.0,
        "dynamic_chars": len(suffix),
    })
    return llm.create_completion(
        prompt=_PREFIX["tokens"] + dynamic_tokens,
        stop=["<|im_end|>"],
        **_completion_kwargs(max_tokens, stream),
    )


def _fallback_request(
    user_text: str,
    *,
    max_tokens: int,
    stream: bool,
    tone_line: str = "",
    date_time_line: str = "",
):
    static_prompt = get_static_system_prompt(include_memory=False)
    dynamic_system = _dynamic_system_text(tone_line, date_time_line)
    if dynamic_system:
        static_prompt = f"{static_prompt}\n\n{dynamic_system}"
    return ModelManager.get_instance().create_chat_completion(
        messages=[
            {"role": "system", "content": static_prompt},
            {"role": "user", "content": user_text},
        ],
        **_completion_kwargs(max_tokens, stream),
    )


def _request(
    user_text: str,
    *,
    max_tokens: int,
    stream: bool,
    tone_line: str = "",
    date_time_line: str = "",
):
    manager = ModelManager.get_instance()
    if _PREFIX["disabled"]:
        return _fallback_request(
            user_text,
            max_tokens=max_tokens,
            stream=stream,
            tone_line=tone_line,
            date_time_line=date_time_line,
        )
    if not stream:
        try:
            with manager.inference() as llm:
                return _cached_request(
                    llm,
                    user_text,
                    max_tokens=max_tokens,
                    stream=False,
                    tone_line=tone_line,
                    date_time_line=date_time_line,
                )
        except Exception as exc:
            _disable_cache(exc)
            return _fallback_request(
                user_text,
                max_tokens=max_tokens,
                stream=False,
                tone_line=tone_line,
                date_time_line=date_time_line,
            )

    def wrapped():
        emitted = False
        try:
            with manager.inference() as llm:
                cached = _cached_request(
                    llm,
                    user_text,
                    max_tokens=max_tokens,
                    stream=True,
                    tone_line=tone_line,
                    date_time_line=date_time_line,
                )
                for chunk in cached:
                    emitted = True
                    yield chunk
                return
        except Exception as exc:
            if emitted:
                raise
            _disable_cache(exc)
        yield from _fallback_request(
            user_text,
            max_tokens=max_tokens,
            stream=True,
            tone_line=tone_line,
            date_time_line=date_time_line,
        )

    return wrapped()


def warm_static_state() -> None:
    if _PREFIX["disabled"]:
        return
    try:
        with ModelManager.get_instance().inference() as llm:
            _ensure_static_state(llm)
    except Exception as exc:
        _disable_cache(exc)


def completion_text(
    user_text: str,
    *,
    max_tokens: int,
    tone_line: str = "",
    date_time_line: str = "",
) -> str:
    result = _request(
        user_text,
        max_tokens=max_tokens,
        stream=False,
        tone_line=tone_line,
        date_time_line=date_time_line,
    )
    choices = result.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    return content_to_text(choice.get("text") or (choice.get("message") or {}).get("content"))


def stream_completion(
    user_text: str,
    *,
    max_tokens: int,
    tone_line: str = "",
    date_time_line: str = "",
):
    for chunk in _request(
        user_text,
        max_tokens=max_tokens,
        stream=True,
        tone_line=tone_line,
        date_time_line=date_time_line,
    ):
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        text = content_to_text(choice.get("text") or (choice.get("delta") or {}).get("content"))
        if text:
            yield text


def clean_visible_text(text: str) -> str:
    source = str(text or "")
    cleaned = strip_hidden_blocks(source)
    if cleaned != source:
        cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines() if line.strip())
    return collapse_hidden_tag_gaps(cleaned).strip(" `\n\t")
