"""Lean text, tag, and local-read helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re

from app.core.config import ADVISOR_ONLY, CHAT_HISTORY_CONTEXT_TOKENS
from app.core.character import build_system_prompt, prompt_revision
from app.integrations.editor_bridge import get_editor_bridge
from app.integrations.vscode_launcher import launch_vscode
from app.memory_store import apply_tag_operations, format_for_prompt, remember_user_message

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
_OPENER_PREFIXES = frozenset(opener[:i].lower() for opener, _ in _OPENERS for i in range(1, len(opener) + 1))
_TAG_NAMES = "MEM|FORGET|PROJECT|EDITOR|ASK_CODER|CODE|READ|WRITE|SHELL|READ_RESULT|THINK"
_XML_NAMES = "think|thinking|tool_call|read|write|shell|code|read_result"
_SQUARE_TAG_RE = re.compile(rf"\[({_TAG_NAMES})\].*?(?:\[/\1\]|$)", re.DOTALL | re.IGNORECASE)
_XML_TAG_RE = re.compile(rf"<({_XML_NAMES})>.*?(?:</\1>|$)", re.DOTALL | re.IGNORECASE)
_MULTI_BLANK_RE = re.compile(r"[ \t]*\n([ \t]*\n)+")
_FUNCTION_RE = re.compile(r"<function\s*=\s*['\"]?([^'\">]+)['\"]?>(.*?)</function>", re.DOTALL | re.IGNORECASE)
_PARAM_RE = re.compile(r"<parameter\s*=\s*['\"]?([^'\">]+)['\"]?>(.*?)</parameter>", re.DOTALL | re.IGNORECASE)
_PAIR_RE = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_READ_MAX_LINES_PER_FILE = 420
_READ_MAX_CHARS_PER_FILE = 16_000


def _extract_tag_contents(text: str, opener: str, closer: str, *, case_insensitive: bool = False) -> list[str]:
    source = str(text or "")
    if not source:
        return []
    haystack = source.lower() if case_insensitive else source
    open_tag = opener.lower() if case_insensitive else opener
    close_tag = closer.lower() if case_insensitive else closer
    out: list[str] = []
    start = 0
    while True:
        i = haystack.find(open_tag, start)
        if i == -1:
            return out
        j = haystack.find(close_tag, i + len(open_tag))
        if j == -1:
            return out
        content = source[i + len(open_tag):j].strip()
        if content:
            out.append(content)
        start = j + len(close_tag)


def _extract_named_xml_sections(text: str, tag_name: str) -> list[str]:
    return _extract_tag_contents(text, f"<{tag_name}>", f"</{tag_name}>", case_insensitive=True)


def _extract_assignment_blocks(text: str, prefix: str, closing_tag: str) -> list[tuple[str, str]]:
    attr_name = prefix.lstrip("<").rstrip("=").lower()
    pattern = re.compile(
        rf"<{re.escape(attr_name)}\s*=\s*['\"]?([^'\">]+)['\"]?>(.*?)</{re.escape(attr_name)}>",
        re.DOTALL | re.IGNORECASE,
    )
    return [(m.group(1).strip(), m.group(2).strip()) for m in pattern.finditer(str(text or ""))]


def _extract_simple_xml_pairs(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).strip()) for m in _PAIR_RE.finditer(str(text or ""))]


def collapse_hidden_tag_gaps(text: str) -> str:
    if not text:
        return text or ""
    value = str(text).replace("\r\n", "\n").replace("\t", " ")
    return _MULTI_BLANK_RE.sub("\n\n", value)


class HiddenTagStreamFilter:
    """Remove hidden/tool tags from token streams."""

    def __init__(self) -> None:
        self._buffer = ""
        self._hidden_close = ""

    @staticmethod
    def _hold_prefix(buffer: str) -> int:
        lower = buffer.lower()
        best = 0
        for i in range(1, min(len(lower), 24) + 1):
            if lower[-i:] in _OPENER_PREFIXES:
                best = i
        return best

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._buffer += str(text)
        out: list[str] = []

        while self._buffer:
            lower = self._buffer.lower()
            if self._hidden_close:
                close = self._hidden_close.lower()
                idx = lower.find(close)
                if idx == -1:
                    keep = max(len(close) - 1, 0)
                    self._buffer = self._buffer[-keep:] if keep else ""
                    return "".join(out)
                self._buffer = self._buffer[idx + len(self._hidden_close):]
                self._hidden_close = ""
                continue

            match = min(
                ((lower.find(op.lower()), op, close) for op, close in _OPENERS if lower.find(op.lower()) != -1),
                default=None,
                key=lambda item: item[0],
            )
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

        return collapse_hidden_tag_gaps("".join(out))

    def flush(self) -> str:
        if self._hidden_close:
            self._buffer = ""
            self._hidden_close = ""
            return ""
        out = self._buffer
        self._buffer = ""
        return collapse_hidden_tag_gaps(out)


def capture_explicit_user_memories(user_text: str) -> bool:
    return remember_user_message(user_text)


def clean_model_text(text: str) -> str:
    source = str(text or "")
    if not source:
        return ""
    source = _SQUARE_TAG_RE.sub("", source)
    source = _XML_TAG_RE.sub("", source)
    return collapse_hidden_tag_gaps(source).replace("`", "").strip()


def _parse_mem(raw: str) -> tuple[str, str] | None:
    if ":" not in raw:
        return None
    category, content = raw.split(":", 1)
    key = category.strip().lower()
    if key in {"fact", "facts"}:
        return "facts", content.strip()
    if key in {"preference", "preferences"}:
        return "preferences", content.strip()
    if key in {"user", "name"}:
        return "user", content.strip() if key == "user" else f"name: {content.strip()}"
    return None


def apply_response_side_effects(text: str) -> None:
    source = str(text or "")
    if not source:
        return
    lowered = source.lower()
    mem_ops = [op for raw in _extract_tag_contents(source, "[MEM]", "[/MEM]", case_insensitive=True) if (op := _parse_mem(raw))]
    forget_queries = _extract_tag_contents(source, "[FORGET]", "[/FORGET]", case_insensitive=True)
    project_ops: list[tuple[str, str]] = []
    for raw in _extract_tag_contents(source, "[PROJECT]", "[/PROJECT]", case_insensitive=True):
        name, _, detail = raw.partition(":")
        if name.strip():
            project_ops.append((name.strip().lower(), detail.strip().lower()))
    if mem_ops or forget_queries or project_ops:
        apply_tag_operations(mem_ops=mem_ops, forget_queries=forget_queries, project_ops=project_ops)

    if ADVISOR_ONLY or ("[editor]" not in lowered and "<tool_call>" not in lowered):
        return
    bridge = get_editor_bridge()
    for raw in extract_editor_commands(source):
        command, argument = _parse_editor_tool(raw)
        if not command:
            continue
        if command in {"open_vscode", "open_project", "open_workspace"}:
            try:
                launch_vscode()
            except RuntimeError:
                pass
        else:
            bridge.queue_action(command, argument)


def finalize_model_response(text: str) -> str:
    apply_response_side_effects(text)
    return clean_model_text(text)


def extract_coder_requests(text: str) -> list[str]:
    return _extract_tag_contents(text, "[ASK_CODER]", "[/ASK_CODER]", case_insensitive=True)


def _function_blocks(text: str) -> list[tuple[str, str]]:
    return [(m.group(1).strip().strip("'\"").lower(), m.group(2).strip()) for m in _FUNCTION_RE.finditer(str(text or ""))]


def _parameters(block: str) -> dict[str, str]:
    return {m.group(1).strip().strip("'\"").lower(): m.group(2).strip() for m in _PARAM_RE.finditer(block)}


def _first(parameters: dict[str, str], *names: str) -> str:
    for name in names:
        if parameters.get(name):
            return parameters[name]
    return ""


def extract_read_requests(text: str) -> list[str]:
    source = str(text or "")
    out: list[str] = []
    seen: set[str] = set()
    for value in _extract_tag_contents(source, "[READ]", "[/READ]", case_insensitive=True):
        if value not in seen:
            seen.add(value)
            out.append(value)
    for value in _extract_named_xml_sections(source, "read") + _extract_named_xml_sections(source, "READ"):
        if value not in seen:
            seen.add(value)
            out.append(value)
    for block in _extract_named_xml_sections(source, "tool_call"):
        for name, inner in _function_blocks(block):
            if name != "read":
                continue
            path = _first(_parameters(inner), "path", "file", "value", "argument")
            if path and path not in seen:
                seen.add(path)
                out.append(path)
    return out


def _parse_editor_tool(raw: str) -> tuple[str, str]:
    command, sep, argument = str(raw or "").strip().partition(":")
    return command.strip().lower(), argument.strip() if sep else ""


def extract_editor_commands(text: str) -> list[str]:
    source = str(text or "")
    out = _extract_tag_contents(source, "[EDITOR]", "[/EDITOR]", case_insensitive=True)
    for block in _extract_named_xml_sections(source, "tool_call"):
        for name, inner in _function_blocks(block):
            if name == "editor":
                params = _parameters(inner)
                command = _first(params, "command", "action", "name")
                argument = _first(params, "argument", "value", "path", "file", "text")
                if command:
                    out.append(command if not argument else f"{command}: {argument}")
            elif name in {"open_vscode", "open_project", "open_workspace"}:
                out.append(name)
    return [item for item in out if item.strip()]


def _normalize_read_path(raw_path: str) -> str:
    path = str(raw_path or "").strip().strip("`'\".,;()[]{}").replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    if ":" in path:
        head, _, suffix = path.partition(":")
        if suffix.strip()[:1].isdigit() or suffix.strip().lower().startswith("l"):
            path = head.strip()
    return path


def _read_numbered_preview(path: Path) -> str:
    lines: list[str] = []
    chars = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for number, raw in enumerate(handle, start=1):
            if number > _READ_MAX_LINES_PER_FILE or chars >= _READ_MAX_CHARS_PER_FILE:
                lines.append("...[trimmed]")
                break
            line = raw.rstrip("\n")
            lines.append(f"{number:>4}| {line}")
            chars += len(line) + 1
    return "\n".join(lines)


def execute_read_requests(filepaths: list[str]) -> list[str]:
    root = Path.cwd().resolve()
    seen: set[str] = set()
    out: list[str] = []
    for raw in filepaths:
        filepath = _normalize_read_path(raw)
        if not filepath or filepath in seen:
            continue
        seen.add(filepath)
        resolved = (root / filepath).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            out.append(f"[READ_RESULT]Access denied: {filepath} is outside project directory[/READ_RESULT]")
            continue
        if not resolved.exists():
            out.append(f"[READ_RESULT]Error reading {filepath}: file not found[/READ_RESULT]")
            continue
        if not resolved.is_file():
            out.append(f"[READ_RESULT]Error reading {filepath}: not a file[/READ_RESULT]")
            continue
        try:
            out.append(f"[READ_RESULT]{filepath} (line-numbered):\n{_read_numbered_preview(resolved)}[/READ_RESULT]")
        except OSError as exc:
            out.append(f"[READ_RESULT]Error reading {filepath}: {exc}[/READ_RESULT]")
    return out


_system_prompt_cache: tuple[tuple[str, bool, tuple[int, int, int]], str] | None = None
_runtime_context_cache: tuple[tuple[bool, bool, str, str], str] | None = None
_datetime_context_cache: tuple[str, str] | None = None


def _estimate_text_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4) if text else 0


def _current_datetime_context() -> str:
    global _datetime_context_cache
    now = datetime.now()
    key = now.strftime("%Y-%m-%d %H:%M")
    if _datetime_context_cache and _datetime_context_cache[0] == key:
        return _datetime_context_cache[1]
    today = now.date()
    lines = [f"Current date/time: {now.strftime('%A, %B %-d, %Y at %-I:%M %p')}"]
    for offset, label in ((-1, "Yesterday"), (0, "Today"), (1, "Tomorrow")):
        day = today + timedelta(days=offset)
        lines.append(f"{label}: {day.strftime('%A, %B ')}{day.day}")
    result = "\n".join(lines)
    _datetime_context_cache = (key, result)
    return result


def build_runtime_context(*, include_memory: bool = True, include_editor: bool = True) -> str:
    global _runtime_context_cache
    memory = format_for_prompt() if include_memory else ""
    editor = get_editor_bridge().format_for_prompt() if include_editor else ""
    key = (include_memory, include_editor, memory, editor)
    if _runtime_context_cache and _runtime_context_cache[0] == key:
        return _runtime_context_cache[1]
    result = "\n\n".join(part for part in (memory, editor) if part)
    _runtime_context_cache = (key, result)
    return result


def _cached_system_prompt(runtime_context: str, *, include_memory: bool = True) -> str:
    global _system_prompt_cache
    key = (runtime_context, include_memory, prompt_revision())
    if _system_prompt_cache and _system_prompt_cache[0] == key:
        return _system_prompt_cache[1]
    result = build_system_prompt(runtime_context, include_memory=include_memory)
    _system_prompt_cache = (key, result)
    return result


def truncate_messages(
    messages: list[dict],
    system_prompt: str,
    user_message: str,
    max_context_tokens: int = CHAT_HISTORY_CONTEXT_TOKENS,
) -> list[dict]:
    budget = max(0, max_context_tokens - _estimate_text_tokens(system_prompt) - _estimate_text_tokens(user_message) - 64)
    kept: list[dict] = []
    used = 0
    for message in reversed(list(messages or [])):
        content = str(message.get("content", "") or "")
        cost = _estimate_text_tokens(content)
        if kept and used + cost > budget:
            break
        if cost > budget and not kept:
            content = content[-max(0, budget * 4):].lstrip()
        kept.append({**message, "content": content})
        used += min(cost, budget)
    kept.reverse()
    return [{"role": "system", "content": system_prompt}, *kept, {"role": "user", "content": user_message}]
