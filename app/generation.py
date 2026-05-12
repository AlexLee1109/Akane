import re
import threading

from app.config import ADVISOR_ONLY, CHAT_HISTORY_CONTEXT_TOKENS
from app.character import build_system_prompt
from app.editor_bridge import get_editor_bridge
from app.memory_store import apply_tag_operations, format_for_prompt, get_store, remember_user_message
from app.vscode_launcher import launch_vscode

_generation_lock = threading.Lock()

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
    "tool_call": "/tool",
    "READ": "</READ>",
    "WRITE": "</WRITE>",
    "SHELL": "</SHELL>",
    "CODE": "</CODE>",
    "READ_RESULT": "</READ_RESULT>",
}
_STREAM_OPENERS = tuple(
    sorted(
        [
            *((f"[{name}]", closer) for name, closer in STREAM_HIDDEN_TAGS.items()),
            *((f"<{name}>", closer) for name, closer in STREAM_HIDDEN_XML_TAGS.items()),
        ],
        key=lambda item: len(item[0]),
        reverse=True,
    )
)

_HIDDEN_SQUARE_TAG_NAMES = (
    "MEM", "FORGET", "PROJECT", "EDITOR", "ASK_CODER", "CODE", "READ", "WRITE", "SHELL", "READ_RESULT", "THINK",
)
_HIDDEN_XML_SIMPLE_TAG_NAMES = ("READ", "WRITE", "SHELL", "CODE", "READ_RESULT", "THINK")

# ── Pre-compiled regex patterns (compiled once at import time) ──────────────
_SQ_NAMES = "MEM|FORGET|PROJECT|EDITOR|ASK_CODER|CODE|READ|WRITE|SHELL|READ_RESULT|THINK"
_XML_NAMES = "think|thinking|tool_call|READ|WRITE|SHELL|CODE|READ_RESULT|THINK"

# Complete tag pairs (non-greedy)
_HIDDEN_SQUARE_TAG_RE = re.compile(
    rf"\[({_SQ_NAMES})\].*?\[/\1\]", re.DOTALL | re.IGNORECASE
)
_HIDDEN_XML_TAG_RE = re.compile(
    rf"<({_XML_NAMES})>.*?</\1>", re.DOTALL | re.IGNORECASE
)
# Unclosed tags — consume to end of string (mirrors original _consume_hidden_tag behaviour)
_UNCLOSED_SQUARE_TAG_RE = re.compile(
    rf"\[({_SQ_NAMES})\].*", re.DOTALL | re.IGNORECASE
)
_UNCLOSED_XML_TAG_RE = re.compile(
    rf"<({_XML_NAMES})>.*", re.DOTALL | re.IGNORECASE
)
# Consecutive blank lines (2+ blank lines → 1 blank line)
_MULTI_BLANK_LINE_RE = re.compile(r"[ \t]*\n([ \t]*\n)+")
# Inline backtick pairs (no newlines inside)
_INLINE_BACKTICK_RE = re.compile(r"`([^`\n]*)`")


from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=64)
def _make_section_re(opener: str, closer: str, case_insensitive: bool) -> re.Pattern:
    flags = re.DOTALL | (re.IGNORECASE if case_insensitive else 0)
    return re.compile(re.escape(opener) + r"(.*?)" + re.escape(closer), flags)


def _extract_wrapped_sections(text: str, opener: str, closer: str, *, case_insensitive: bool = False) -> list[str]:
    source = str(text or "")
    if not source:
        return []
    return [m.group(1).strip() for m in _make_section_re(opener, closer, case_insensitive).finditer(source)]

def _extract_named_xml_sections(text: str, tag_name: str) -> list[str]:
    return _extract_wrapped_sections(
        text,
        f"<{tag_name}>",
        f"</{tag_name}>",
        case_insensitive=True,
    )


@_lru_cache(maxsize=32)
def _make_assignment_re(prefix: str, closing_tag: str) -> re.Pattern:
    # Matches e.g. <function=name>content</function>
    return re.compile(
        re.escape(prefix) + r"([^>]*?)>" + r"(.*?)" + re.escape(closing_tag),
        re.DOTALL | re.IGNORECASE,
    )


def _extract_assignment_blocks(text: str, prefix: str, closing_tag: str) -> list[tuple[str, str]]:
    source = str(text or "")
    if not source:
        return []
    return [(m.group(1).strip(), m.group(2).strip()) for m in _make_assignment_re(prefix, closing_tag).finditer(source)]


# Pre-compiled for _extract_simple_xml_pairs — matches <tagname>content</tagname>
_SIMPLE_XML_PAIR_RE = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)


def _extract_simple_xml_pairs(text: str) -> list[tuple[str, str]]:
    source = str(text or "")
    if not source:
        return []
    return [(m.group(1), m.group(2).strip()) for m in _SIMPLE_XML_PAIR_RE.finditer(source)]


def _strip_hidden_markup(text: str) -> str:
    source = str(text or "")
    if not source:
        return ""
    if "[" not in source and "<" not in source:
        return collapse_hidden_tag_gaps(source).strip()
    # Remove complete tag pairs first (non-greedy match)
    source = _HIDDEN_SQUARE_TAG_RE.sub("", source)
    source = _HIDDEN_XML_TAG_RE.sub("", source)
    # Remove unclosed tags — consume everything from the opener to end of string,
    # matching the original _consume_hidden_tag behaviour for truncated model output.
    source = _UNCLOSED_SQUARE_TAG_RE.sub("", source)
    source = _UNCLOSED_XML_TAG_RE.sub("", source)
    return collapse_hidden_tag_gaps(source).strip()


def _strip_inline_markdown_code(text: str) -> str:
    if "`" not in text:
        return text
    # Strip paired inline backticks (no newlines inside)
    result = _INLINE_BACKTICK_RE.sub(r"\1", text)
    # Remove any remaining unpaired backticks
    return result.replace("`", "") if "`" in result else result

CATEGORY_MAP = {
    "name": "user",
    "fact": "facts",
    "facts": "facts",
    "preference": "preferences",
    "preferences": "preferences",
    "user": "user",
}


def _looks_like_name(value: str) -> bool:
    candidate = value.strip()
    if not candidate or len(candidate) > 40:
        return False
    parts = candidate.replace("-", " ").split()
    if not 1 <= len(parts) <= 3:
        return False
    return all(part.isalpha() for part in parts)


def capture_explicit_user_memories(user_text: str) -> bool:
    """Store important explicit user details directly from the raw user message."""
    return remember_user_message(user_text)


def collapse_hidden_tag_gaps(text: str) -> str:
    """Remove blank-line artifacts left behind when inline tags are hidden."""
    if not text:
        return text or ""
    if "\n" not in text and "\t" not in text and "\r" not in text:
        return text
    text = text.replace("\r\n", "\n").replace("\t", " ")
    return _MULTI_BLANK_LINE_RE.sub("\n\n", text)


# ── Fast lookup tables for HiddenTagStreamFilter ────────────────────────────
# Dict: full opener string → closer string (O(1) full-match lookup)
_OPENER_DICT: dict[str, str] = {op: cl for op, cl in _STREAM_OPENERS}
# Group openers by their first two characters (usually narrows candidates to 1-2)
_OPENERS_BY_PREFIX2: dict[str, list[tuple[str, str]]] = {}
for _op, _cl in _STREAM_OPENERS:
    _OPENERS_BY_PREFIX2.setdefault(_op[:2], []).append((_op, _cl))
# All possible prefixes of all openers — used for O(1) partial-match detection
_OPENER_PREFIXES: frozenset[str] = frozenset(
    _op[:i] for _op, _ in _STREAM_OPENERS for i in range(1, len(_op) + 1)
)


class HiddenTagStreamFilter:
    """Hide inline tags during streaming while preserving surrounding text."""

    def __init__(self):
        self._visible_buffer = ""
        self._pending_visible = ""
        self._in_tag = False
        self._closing = ""

    def _drain_pending(self, *, flush: bool = False) -> str:
        """Emit only stable visible text while holding ambiguous trailing whitespace."""
        if not self._pending_visible:
            return ""

        pv = self._pending_visible
        normalized = collapse_hidden_tag_gaps(pv) if ("\n" in pv or "\t" in pv or "\r" in pv) else pv
        if flush:
            self._pending_visible = ""
            return normalized

        stable = normalized.rstrip(" \t\n")
        trailing = normalized[len(stable):]
        self._pending_visible = trailing
        return stable

    def feed(self, text: str) -> str:
        self._visible_buffer += text

        while self._visible_buffer:
            if self._in_tag:
                close_idx = self._visible_buffer.find(self._closing)
                if close_idx == -1:
                    keep = max(len(self._closing) - 1, 0)
                    self._visible_buffer = self._visible_buffer[-keep:] if keep else ""
                    break
                self._visible_buffer = self._visible_buffer[close_idx + len(self._closing):]
                self._in_tag = False
                self._closing = ""
                continue

            bracket_idx = self._visible_buffer.find("[")
            angle_idx = self._visible_buffer.find("<")
            if bracket_idx == -1 and angle_idx == -1:
                self._pending_visible += self._visible_buffer
                self._visible_buffer = ""
                break
            elif bracket_idx == -1:
                tag_idx = angle_idx
            elif angle_idx == -1:
                tag_idx = bracket_idx
            else:
                tag_idx = min(bracket_idx, angle_idx)

            if tag_idx > 0:
                self._pending_visible += self._visible_buffer[:tag_idx]
                self._visible_buffer = self._visible_buffer[tag_idx:]

            buf = self._visible_buffer
            # Fast path: group by first two chars to narrow candidates, then startswith check
            matched = False
            for opener, closer in _OPENERS_BY_PREFIX2.get(buf[:2], ()):
                if buf.startswith(opener):
                    self._visible_buffer = buf[len(opener):]
                    self._in_tag = True
                    self._closing = closer
                    matched = True
                    break

            if matched:
                continue

            # O(1) partial-match check: is buf a possible prefix of any opener?
            if buf in _OPENER_PREFIXES:
                break  # wait for more data to arrive

            # Not an opener — emit the bracket character and advance
            self._pending_visible += buf[0]
            self._visible_buffer = buf[1:]

        return self._drain_pending()

    def flush(self) -> str:
        if self._in_tag:
            return ""
        self._pending_visible += self._visible_buffer
        self._visible_buffer = ""
        return self._drain_pending(flush=True)


def _collect_memory_and_tool_ops(text: str) -> tuple[
    list[tuple[str, str]],
    list[str],
    list[tuple[str, str]],
    list[str],
]:
    mem_ops: list[tuple[str, str]] = []
    forget_queries: list[str] = []
    project_ops: list[tuple[str, str]] = []
    editor_ops = extract_editor_commands(text)

    if "[MEM]" in text:
        for raw in _extract_wrapped_sections(text, "[MEM]", "[/MEM]"):
            if ":" in raw:
                cat, content = raw.split(":", 1)
                cat = cat.strip().lower()
                content = content.strip()
                mem_cat = CATEGORY_MAP.get(cat)
                if mem_cat and content:
                    mem_ops.append((mem_cat, content))

    if "[FORGET]" in text:
        for query in _extract_wrapped_sections(text, "[FORGET]", "[/FORGET]"):
            if query:
                forget_queries.append(query)

    if "[PROJECT]" in text:
        for raw in _extract_wrapped_sections(text, "[PROJECT]", "[/PROJECT]"):
            if ":" in raw:
                name, detail = raw.split(":", 1)
                name = name.strip().lower()
                detail = detail.strip().lower()
                project_ops.append((name, detail))
            elif raw:
                project_ops.append((raw.strip().lower(), ""))

    return mem_ops, forget_queries, project_ops, editor_ops


def apply_response_side_effects(text: str) -> None:
    """Apply memory and editor side effects without mutating visible text."""
    mem_ops, forget_queries, project_ops, editor_ops = _collect_memory_and_tool_ops(text)

    filtered_mem_ops = []
    for mem_cat, content in mem_ops:
        if mem_cat != "user":
            filtered_mem_ops.append((mem_cat, content))
            continue
        if ":" in content:
            filtered_mem_ops.append((mem_cat, content))
        elif _looks_like_name(content):
            filtered_mem_ops.append((mem_cat, f"name: {content.strip()}"))

    if filtered_mem_ops or forget_queries or project_ops:
        apply_tag_operations(
            mem_ops=filtered_mem_ops,
            forget_queries=forget_queries,
            project_ops=project_ops,
        )

    if editor_ops and not ADVISOR_ONLY:
        bridge = get_editor_bridge()
        for raw in editor_ops:
            command, argument = _parse_editor_tool(raw)
            if command:
                if command in {"open_vscode", "open_project", "open_workspace"}:
                    try:
                        launch_vscode()
                    except RuntimeError:
                        pass
                else:
                    bridge.queue_action(command, argument)


def clean_model_text(text: str) -> str:
    """Return only the visible conversational text from a model reply."""
    return _strip_inline_markdown_code(_strip_hidden_markup(text))


def extract_coder_requests(text: str) -> list[str]:
    if "[ASK_CODER]" not in str(text or ""):
        return []
    requests: list[str] = []
    for content in _extract_wrapped_sections(text, "[ASK_CODER]", "[/ASK_CODER]"):
        if content:
            requests.append(content)
    return requests


def finalize_model_response(text: str) -> str:
    """Apply response side effects, then return the visible text."""
    apply_response_side_effects(text)
    return clean_model_text(text)


def _parse_editor_tool(raw: str) -> tuple[str, str]:
    raw = raw.strip()
    if not raw:
        return "", ""
    if ":" not in raw:
        return raw.strip().lower(), ""
    command, argument = raw.split(":", 1)
    return command.strip().lower(), argument.strip()


def extract_editor_commands(text: str) -> list[str]:
    source = str(text or "")
    if "[EDITOR]" not in source and "<tool_call>" not in source.lower():
        return []
    editor_ops: list[str] = []

    for raw in _extract_wrapped_sections(source, "[EDITOR]", "[/EDITOR]"):
        if raw:
            editor_ops.append(raw)

    for block in _extract_tool_call_blocks(source):
        editor_ops.extend(_extract_editor_commands_from_tool_block(block))

    return editor_ops


def extract_read_requests(text: str) -> list[str]:
    source = str(text or "")
    source_lower = source.lower()
    if "[READ]" not in source and "<tool_call>" not in source_lower and "<read>" not in source_lower:
        return []
    read_requests: list[str] = []
    seen: set[str] = set()

    for filepath in _extract_wrapped_sections(source, "[READ]", "[/READ]"):
        if filepath and filepath not in seen:
            seen.add(filepath)
            read_requests.append(filepath)

    for block in _extract_tool_call_blocks(source):
        function_name, inner = _extract_function_tool_block(block)
        if function_name != "read":
            continue
        parameters = _extract_tool_parameters(inner)
        filepath = _first_parameter(parameters, "file", "path", "value", "argument")
        if filepath and filepath not in seen:
            seen.add(filepath)
            read_requests.append(filepath)

    for filepath in _extract_named_xml_sections(source, "READ"):
        if filepath and filepath not in seen:
            seen.add(filepath)
            read_requests.append(filepath)

    return read_requests


def execute_read_requests(filepaths: list[str]) -> list[str]:
    from pathlib import Path

    read_results: list[str] = []
    project_root = Path.cwd().resolve()
    seen: set[str] = set()

    def format_with_line_numbers(text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return ""
        width = max(3, len(str(len(lines))))
        return "\n".join(
            f"{idx:>{width}}| {line}"
            for idx, line in enumerate(lines, start=1)
        )

    for raw_path in filepaths:
        filepath = str(raw_path or "").strip()
        if not filepath or filepath in seen:
            continue
        seen.add(filepath)
        resolved = (project_root / filepath).resolve()
        if not str(resolved).startswith(str(project_root)):
            read_results.append(f"[READ_RESULT]Access denied: {filepath} is outside project directory[/READ_RESULT]")
            continue
        if not resolved.exists():
            read_results.append(f"[READ_RESULT]Error reading {filepath}: file not found[/READ_RESULT]")
            continue
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                content = f.read()
            numbered = format_with_line_numbers(content)
            read_results.append(
                f"[READ_RESULT]{filepath} (line-numbered):\n{numbered}[/READ_RESULT]"
            )
        except Exception as e:
            read_results.append(f"[READ_RESULT]Error reading {filepath}: {e}[/READ_RESULT]")

    return read_results


def _extract_tool_call_blocks(text: str) -> list[str]:
    return _extract_named_xml_sections(text, "tool_call")


def _extract_function_tool_block(block: str) -> tuple[str, str]:
    matches = _extract_assignment_blocks(block, "<function=", "</function>")
    if not matches:
        return "", ""
    name, content = matches[0]
    return name.strip().lower(), content


def _extract_tool_parameters(block: str) -> dict[str, str]:
    return {
        name.strip().lower(): content.strip()
        for name, content in _extract_assignment_blocks(block, "<parameter=", "</parameter>")
        if name.strip()
    }


def _first_parameter(parameters: dict[str, str], *names: str) -> str:
    for name in names:
        value = parameters.get(name, "")
        if value:
            return value
    return ""


def _editor_ops_from_parameter_block(inner: str) -> list[str]:
    parameters = _extract_tool_parameters(inner)
    command = _first_parameter(parameters, "action", "command", "name")
    argument = _first_parameter(parameters, "value", "argument", "file", "path", "text")
    if command:
        lowered = command.lower()
        return [lowered if not argument else f"{lowered}: {argument}"]
    return [
        command if not argument else f"{command}: {argument}"
        for command, argument in (
            (name.strip().lower(), content.strip())
            for name, content in _extract_simple_xml_pairs(inner)
        )
        if command
    ]


def _extract_editor_commands_from_tool_block(block: str) -> list[str]:
    editor_blocks = _extract_named_xml_sections(block, "editor")
    if editor_blocks:
        inner = editor_blocks[0]
        return _editor_ops_from_parameter_block(inner)

    function_name, inner = _extract_function_tool_block(block)
    if function_name == "editor":
        return _editor_ops_from_parameter_block(inner)

    if function_name in {"open_vscode", "open_project", "open_workspace"}:
        return [function_name]
    return []


# Cache for system prompt to avoid repeated formatting
_system_prompt_cache = None
_system_prompt_gen = -1
_runtime_context_cache = ""
_include_memory_cache: bool | None = None


def _nth_weekday(year: int, month: int, weekday: int, n: int):
    """nth occurrence of weekday (0=Mon,6=Sun) in month. n=-1 = last."""
    from datetime import date, timedelta
    if n > 0:
        d = date(year, month, 1)
        count = 0
        while True:
            if d.weekday() == weekday:
                count += 1
                if count == n:
                    return d
            d += timedelta(days=1)
    else:
        nxt = date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)
        d = nxt - timedelta(days=1)
        while d.weekday() != weekday:
            d -= timedelta(days=1)
        return d


def _current_datetime_context() -> str:
    """Return current date/time plus yesterday/today/tomorrow holiday facts."""
    from datetime import datetime as _datetime, timedelta
    now = _datetime.now()
    today = now.date()
    weekday = now.strftime("%A")
    month = now.strftime("%B")
    year = now.year
    hour = now.hour % 12 or 12
    minute = now.strftime("%M")
    am_pm = "AM" if now.hour < 12 else "PM"
    date_str = f"{weekday}, {month} {today.day}, {year} at {hour}:{minute} {am_pm}"

    lines: list[str] = [f"CURRENT DATE AND TIME: {date_str}"]
    for offset, label in ((-1, "Yesterday"), (0, "Today"), (1, "Tomorrow")):
        d = today + timedelta(days=offset)
        day_str = d.strftime("%A, %B ") + str(d.day)
        lines.append(f"{label}: {day_str}")
    lines.append(
        "Use this to answer any date, day-of-week, or holiday questions. "
        "Never invent or guess the date."
    )
    return "\n".join(lines)

def build_runtime_context(*, include_memory: bool = True, include_editor: bool = True) -> str:
    memory_context = format_for_prompt() if include_memory else ""
    editor_context = get_editor_bridge().format_for_prompt() if include_editor else ""
    parts = [_current_datetime_context()]
    if memory_context:
        parts.append(memory_context)
    if editor_context:
        parts.append(editor_context)
    return "\n\n".join(parts)


def _cached_system_prompt(runtime_context: str, *, include_memory: bool = True) -> str:
    global _system_prompt_cache, _system_prompt_gen, _runtime_context_cache, _include_memory_cache

    store = get_store()
    current_gen = store._prompt_cache_gen
    if (
        _system_prompt_cache is not None
        and _system_prompt_gen == current_gen
        and _runtime_context_cache == runtime_context
        and _include_memory_cache == include_memory
    ):
        return _system_prompt_cache

    system_prompt = build_system_prompt(runtime_context, include_memory=include_memory)
    _system_prompt_cache = system_prompt
    _system_prompt_gen = current_gen
    _runtime_context_cache = runtime_context
    _include_memory_cache = include_memory
    return system_prompt


def truncate_messages(messages: list[dict], system_prompt: str, user_message: str, max_context_tokens: int = CHAT_HISTORY_CONTEXT_TOKENS) -> list[dict]:
    """
    Truncate message history to fit within context window.

    Uses rough token estimation: ~4 chars per token for English.
    Prioritizes keeping recent messages and the most recent complete exchanges.
    """
    # Estimate tokens (rough: 4 chars = 1 token)
    system_tokens = len(system_prompt) // 4
    user_msg_tokens = len(user_message) // 4
    available_for_history = max_context_tokens - system_tokens - user_msg_tokens - 100  # safety buffer

    if available_for_history <= 0:
        # System prompt itself is too large - this is a configuration error
        # Return just the system prompt and user message, drop all history
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    # Convert deque to list if needed
    messages_list = list(messages)
    if not messages_list:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    # Calculate current history tokens
    history_tokens = sum(len(m.get("content", "")) // 4 for m in messages_list)

    if history_tokens <= available_for_history:
        chat_messages = [{"role": "system", "content": system_prompt}]
        chat_messages.extend(messages_list)
        chat_messages.append({"role": "user", "content": user_message})
        return chat_messages

    # Need to truncate: keep the most recent conversation first, while trying
    # to preserve complete recent exchanges instead of a random cutoff.
    trimmed: list[dict] = []
    current_tokens = 0
    min_recent_messages = min(len(messages_list), 6)

    for index in range(len(messages_list) - 1, -1, -1):
        msg = messages_list[index]
        msg_tokens = len(msg.get("content", "")) // 4
        if current_tokens + msg_tokens <= available_for_history:
            trimmed.append(msg)
            current_tokens += msg_tokens
            continue
        if len(trimmed) < min_recent_messages:
            trimmed.append(msg)
            current_tokens += msg_tokens
            continue
        break
    trimmed.reverse()

    # Build final message list
    chat_messages = [{"role": "system", "content": system_prompt}]
    chat_messages.extend(trimmed)
    chat_messages.append({"role": "user", "content": user_message})

    return chat_messages