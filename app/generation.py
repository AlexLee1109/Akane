import re
import threading

from app.config import (
    CHAT_HISTORY_CONTEXT_TOKENS,
    MAX_TOKENS,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from app.character import build_system_prompt
from app.config import ADVISOR_ONLY
from app.editor_bridge import get_editor_bridge
from app.memory_store import NEGATIVE, NEUTRAL, POSITIVE, apply_tag_operations, format_for_prompt, get_store
from app.model_loader import LLM
from app.vscode_launcher import launch_vscode

_generation_lock = threading.Lock()

# Tag patterns
MEM_PATTERN = re.compile(r"\[MEM\](.*?)\[/MEM\]", re.DOTALL)
OBSERVE_PATTERN = re.compile(r"\[OBSERVE\](.*?)\[/OBSERVE\]", re.DOTALL)
FORGET_PATTERN = re.compile(r"\[FORGET\](.*?)\[/FORGET\]", re.DOTALL)
PROJECT_PATTERN = re.compile(r"\[PROJECT\](.*?)\[/PROJECT\]", re.DOTALL)
EDITOR_PATTERN = re.compile(r"\[EDITOR\](.*?)\[/EDITOR\]", re.DOTALL)
ASK_CODER_PATTERN = re.compile(r"\[ASK_CODER\](.*?)\[/ASK_CODER\]", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
EDITOR_TOOL_BLOCK_PATTERN = re.compile(r"<editor>\s*(.*?)\s*</editor>", re.DOTALL | re.IGNORECASE)
FUNCTION_TOOL_BLOCK_PATTERN = re.compile(
    r"<function=([A-Z_][A-Z0-9_]*)>\s*(.*?)\s*</function>",
    re.DOTALL | re.IGNORECASE,
)
FUNCTION_PARAMETER_PATTERN = re.compile(
    r"<parameter=([a-z_][a-z0-9_]*)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)
NESTED_TOOL_ACTION_PATTERN = re.compile(
    r"<([a-z_][a-z0-9_]*)>\s*(.*?)\s*</\1>",
    re.DOTALL | re.IGNORECASE,
)
XML_SIMPLE_TOOL_PATTERN = re.compile(
    r"<([A-Z_][A-Z0-9_]*)>\s*(.*?)\s*</\1>",
    re.DOTALL | re.IGNORECASE,
)
XML_READ_PATTERN = re.compile(r"<READ>\s*(.*?)\s*</READ>", re.DOTALL | re.IGNORECASE)

# Tool tags (for future implementation)
TOOL_PATTERNS = {
    "CODE": re.compile(r"\[CODE\](.*?)\[/CODE\]", re.DOTALL),
    "READ": re.compile(r"\[READ\](.*?)\[/READ\]", re.DOTALL),
    "WRITE": re.compile(r"\[WRITE\](.*?)\[/WRITE\]", re.DOTALL),
    "SHELL": re.compile(r"\[SHELL\](.*?)\[/SHELL\]", re.DOTALL),
}

# Strip all special tags from displayed text
ALL_TAGS_PATTERN = re.compile(
    r"\[(?:MEM|OBSERVE|FORGET|PROJECT|EDITOR|ASK_CODER|CODE|READ|WRITE|SHELL|READ_RESULT)\].*?\[/(?:MEM|OBSERVE|FORGET|PROJECT|EDITOR|ASK_CODER|CODE|READ|WRITE|SHELL|READ_RESULT)\]",
    re.DOTALL,
)
EDITOR_TOOL_CALL_STRIP_PATTERN = re.compile(
    r"<tool_call>.*?(?:</tool_call>|$)",
    re.DOTALL | re.IGNORECASE,
)
XML_SIMPLE_TOOL_STRIP_PATTERN = re.compile(
    r"<(?:READ|WRITE|SHELL|CODE|READ_RESULT)>\s*.*?\s*</(?:READ|WRITE|SHELL|CODE|READ_RESULT)>",
    re.DOTALL | re.IGNORECASE,
)

STREAM_HIDDEN_TAGS = {
    "MEM": "[/MEM]",
    "OBSERVE": "[/OBSERVE]",
    "FORGET": "[/FORGET]",
    "PROJECT": "[/PROJECT]",
    "EDITOR": "[/EDITOR]",
    "ASK_CODER": "[/ASK_CODER]",
    "CODE": "[/CODE]",
    "READ": "[/READ]",
    "WRITE": "[/WRITE]",
    "SHELL": "[/SHELL]",
    "READ_RESULT": "[/READ_RESULT]",
}
STREAM_HIDDEN_XML_TAGS = {
    "tool_call": "</tool_call>",
    "READ": "</READ>",
    "WRITE": "</WRITE>",
    "SHELL": "</SHELL>",
    "CODE": "</CODE>",
    "READ_RESULT": "</READ_RESULT>",
}

_HIDDEN_SQUARE_TAG_NAMES = (
    "MEM", "OBSERVE", "FORGET", "PROJECT", "EDITOR", "ASK_CODER", "CODE", "READ", "WRITE", "SHELL", "READ_RESULT",
)
_HIDDEN_XML_SIMPLE_TAG_NAMES = ("READ", "WRITE", "SHELL", "CODE", "READ_RESULT")


def _strip_wrapped_tag_ranges(text: str, opener: str, closer: str) -> str:
    source = str(text or "")
    if opener not in source:
        return source
    parts: list[str] = []
    index = 0
    opener_len = len(opener)
    closer_len = len(closer)
    while True:
        start = source.find(opener, index)
        if start == -1:
            parts.append(source[index:])
            break
        parts.append(source[index:start])
        end = source.find(closer, start + opener_len)
        if end == -1:
            parts.append(source[start:])
            break
        index = end + closer_len
    return "".join(parts)


def _strip_hidden_square_tags(text: str) -> str:
    cleaned = str(text or "")
    for tag_name in _HIDDEN_SQUARE_TAG_NAMES:
        cleaned = _strip_wrapped_tag_ranges(cleaned, f"[{tag_name}]", f"[/{tag_name}]")
    return cleaned


def _strip_hidden_xml_tags(text: str) -> str:
    cleaned = str(text or "")
    cleaned = _strip_wrapped_tag_ranges(cleaned, "<tool_call>", "</tool_call>")
    for tag_name in _HIDDEN_XML_SIMPLE_TAG_NAMES:
        cleaned = _strip_wrapped_tag_ranges(cleaned, f"<{tag_name}>", f"</{tag_name}>")
    return cleaned


def _strip_hidden_markup(text: str) -> str:
    cleaned = _strip_hidden_square_tags(text)
    cleaned = _strip_hidden_xml_tags(cleaned)
    return collapse_hidden_tag_gaps(cleaned).strip()


def _strip_inline_markdown_code(text: str) -> str:
    cleaned = str(text or "")
    if "`" not in cleaned:
        return cleaned
    parts: list[str] = []
    index = 0
    while index < len(cleaned):
        start = cleaned.find("`", index)
        if start == -1:
            parts.append(cleaned[index:])
            break
        parts.append(cleaned[index:start])
        end = cleaned.find("`", start + 1)
        if end == -1 or "\n" in cleaned[start + 1 : end]:
            parts.append(cleaned[start:].replace("`", ""))
            break
        parts.append(cleaned[start + 1 : end])
        index = end + 1
    return "".join(parts)

CATEGORY_MAP = {
    "name": "user",
    "fact": "facts",
    "facts": "facts",
    "preference": "preferences",
    "preferences": "preferences",
    "user": "user",
}


def _detect_weight(text: str) -> str:
    """Detect emotional weight using explicit markers instead of keyword lists."""
    lowered = str(text or "").strip().lower()
    if lowered.startswith(("negative:", "bad:", "- ")):
        return NEGATIVE
    if lowered.startswith(("positive:", "good:", "+ ")):
        return POSITIVE
    return NEUTRAL


def _looks_like_name(value: str) -> bool:
    candidate = value.strip()
    if not candidate or len(candidate) > 40:
        return False
    parts = candidate.replace("-", " ").split()
    if not 1 <= len(parts) <= 3:
        return False
    return all(part.isalpha() for part in parts)


def _normalize_memory_text(value: str) -> str:
    value = " ".join(value.strip().split())
    return value.strip(",:;- ")


def _replace_ci_word(text: str, source: str, target: str) -> str:
    lowered = text.lower()
    source_lower = source.lower()
    index = 0
    parts: list[str] = []
    source_len = len(source)
    while True:
        found = lowered.find(source_lower, index)
        if found == -1:
            parts.append(text[index:])
            break
        before_ok = found == 0 or not (lowered[found - 1].isalnum() or lowered[found - 1] == "_")
        after_index = found + source_len
        after_ok = after_index >= len(text) or not (lowered[after_index].isalnum() or lowered[after_index] == "_")
        if not (before_ok and after_ok):
            parts.append(text[index : found + 1])
            index = found + 1
            continue
        parts.append(text[index:found])
        parts.append(target)
        index = after_index
    return "".join(parts)


def _canonicalize_clause(clause: str) -> str:
    clause = clause.replace("’", "'")
    clause = _replace_ci_word(clause, "i'm", "i am")
    clause = _replace_ci_word(clause, "im", "i am")
    clause = _replace_ci_word(clause, "don't", "do not")
    clause = _replace_ci_word(clause, "can't", "can not")
    return " ".join(clause.split())


def _split_on_punctuation(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for ch in str(text or ""):
        if ch in ".!?;\n":
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _starts_self_clause(lowered: str) -> bool:
    return lowered.startswith(("i ", "i'm ", "im ", "my name", "name's ", "call me ", "i go by "))


def _split_self_subclauses(text: str) -> list[str]:
    lowered = str(text or "").lower().replace("’", "'")
    original = str(text or "")
    parts: list[str] = []
    start = 0
    index = 0
    for connector in (" and ", " but "):
        pass
    while index < len(lowered):
        split_at = -1
        split_len = 0
        for connector in (" and ", " but "):
            if lowered.startswith(connector, index):
                remainder = lowered[index + len(connector) :].lstrip()
                if _starts_self_clause(remainder):
                    split_at = index
                    split_len = len(connector)
                    break
        if split_at == -1:
            index += 1
            continue
        part = original[start:split_at].strip()
        if part:
            parts.append(part)
        start = split_at + split_len
        index = start
    tail = original[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_self_clauses(text: str) -> list[str]:
    clauses: list[str] = []
    for chunk in _split_on_punctuation(text):
        chunk = _normalize_memory_text(chunk)
        if not chunk:
            continue
        for clause in _split_self_subclauses(chunk):
            clause = _normalize_memory_text(clause)
            if not clause:
                continue
            normalized = _canonicalize_clause(clause).lower()
            if normalized.startswith(("i ", "my name", "name's ", "call me ", "i go by ")):
                clauses.append(clause)
    return clauses


def _normalize_project_phrase(value: str) -> tuple[str, str]:
    """Split a project phrase into a short name plus optional detail."""
    value = _normalize_memory_text(value)
    lowered = value.lower()
    for prefix in ("a ", "an ", "the ", "my "):
        if lowered.startswith(prefix):
            value = value[len(prefix) :]
            lowered = value.lower()
            break
    for prefix in ("project called ", "app called ", "tool called "):
        if lowered.startswith(prefix):
            value = value[len(prefix) :]
            lowered = value.lower()
            break
    if not value:
        return "", ""

    split_index = -1
    split_len = 0
    lowered = value.lower()
    for connector in (" using ", " with ", " for ", " on ", " at "):
        found = lowered.find(connector)
        if found != -1 and (split_index == -1 or found < split_index):
            split_index = found
            split_len = len(connector)
    if split_index == -1:
        name_part = value
        detail_part = ""
    else:
        name_part = value[:split_index]
        detail_part = value[split_index + split_len :]
    name = _normalize_memory_text(name_part).lower()
    detail = _normalize_memory_text(detail_part).lower()
    if len(name) > 60:
        name = name[:60].rsplit(" ", 1)[0] or name[:60]
    return name, detail


def _looks_like_durable_fact(value: str) -> bool:
    lowered = value.lower().strip()
    if not lowered:
        return False
    if len(lowered) <= 6:
        return False
    meaningful_tokens = [token for token in lowered.split() if len(token) > 2]
    return len(meaningful_tokens) >= 3 or any(char.isdigit() for char in value)


def _extract_name_from_clause(clause: str) -> str | None:
    lowered = clause.lower()
    for prefix in ("my name is ", "name is ", "name's ", "call me ", "i go by "):
        if lowered.startswith(prefix):
            candidate = _normalize_memory_text(clause[len(prefix):])
            return candidate if _looks_like_name(candidate) else None
    return None


def _extract_preference_memory(clause: str) -> str | None:
    lowered = _canonicalize_clause(clause).lower()
    for prefix, memory_prefix in (
        ("i prefer ", "prefers "),
        ("i like ", "likes "),
        ("i love ", "loves "),
        ("i enjoy ", "enjoys "),
        ("i want ", "wants "),
        ("i need ", "needs "),
        ("i hate ", "doesn't like "),
        ("i do not like ", "doesn't like "),
        ("i don't like ", "doesn't like "),
    ):
        if lowered.startswith(prefix):
            remainder = _normalize_memory_text(clause[len(prefix):]).lower()
            return f"{memory_prefix}{remainder}" if remainder else None
    return None


def _extract_project_memory(clause: str) -> tuple[str, str] | None:
    lowered = _canonicalize_clause(clause).lower()
    for prefix in (
        "i am working on ",
        "i'm working on ",
        "im working on ",
        "i am building ",
        "i'm building ",
        "im building ",
        "i am making ",
        "i'm making ",
        "im making ",
        "i am creating ",
        "i'm creating ",
        "im creating ",
    ):
        if lowered.startswith(prefix):
            value = _normalize_memory_text(clause[len(prefix):])
            name, detail = _normalize_project_phrase(value)
            return (name, detail) if name else None
    return None


def _extract_fact_memory(clause: str) -> str | None:
    lowered = _canonicalize_clause(clause).lower()
    for prefix, fact_prefix in (
        ("i use ", "uses "),
        ("i'm using ", "uses "),
        ("im using ", "uses "),
        ("i have ", "has "),
        ("i'm on ", "uses "),
        ("im on ", "uses "),
        ("i run ", "runs "),
    ):
        if lowered.startswith(prefix):
            value = _normalize_memory_text(clause[len(prefix):]).lower()
            if not _looks_like_durable_fact(value):
                return None
            return f"{fact_prefix}{value}" if value else None
    return None


def capture_explicit_user_memories(user_text: str) -> bool:
    """Store important explicit user details directly from the raw user message."""
    lowered = str(user_text or "").lower()
    if "i" not in lowered and "name" not in lowered and "call me" not in lowered:
        return False
    mem_ops: list[tuple[str, str]] = []
    project_ops: list[tuple[str, str]] = []
    changed = False
    store = get_store().get()
    known_name = store.get("user", {}).get("name", "").strip().lower()

    for clause in _extract_self_clauses(user_text):
        name = _extract_name_from_clause(clause)
        if name:
            if known_name != name.lower():
                mem_ops.append(("user", f"name: {name}"))
                changed = True
            continue

        project = _extract_project_memory(clause)
        if project:
            project_ops.append(project)
            changed = True
            continue

        preference = _extract_preference_memory(clause)
        if preference:
            mem_ops.append(("preferences", preference))
            changed = True
            continue

        fact = _extract_fact_memory(clause)
        if fact:
            mem_ops.append(("facts", fact))
            changed = True

    if not changed:
        return False

    apply_tag_operations(
        mem_ops=mem_ops,
        observe_ops=[],
        forget_queries=[],
        project_ops=project_ops,
    )
    return True


def collapse_hidden_tag_gaps(text: str) -> str:
    """Remove blank-line artifacts left behind when inline tags are hidden."""
    text = str(text or "").replace("\r\n", "\n")
    if not text:
        return text

    normalized_lines: list[str] = []
    previous_blank = False
    for raw_line in text.split("\n"):
        line = raw_line.replace("\t", " ")
        if not line:
            if previous_blank:
                continue
            normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line)
        previous_blank = False
    return "\n".join(normalized_lines)


class HiddenTagStreamFilter:
    """Hide inline tags during streaming while preserving surrounding text."""

    def __init__(self):
        self._openers = sorted(
            [
                *((f"[{name}]", closer) for name, closer in STREAM_HIDDEN_TAGS.items()),
                *((f"<{name}>", closer) for name, closer in STREAM_HIDDEN_XML_TAGS.items()),
            ],
            key=lambda item: len(item[0]),
            reverse=True,
        )
        self._visible_buffer = ""
        self._pending_visible = ""
        self._in_tag = False
        self._closing = ""

    def _drain_pending(self, *, flush: bool = False) -> str:
        """Emit only stable visible text while holding ambiguous trailing whitespace."""
        if not self._pending_visible:
            return ""

        normalized = collapse_hidden_tag_gaps(self._pending_visible)
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
                    if keep:
                        self._visible_buffer = self._visible_buffer[-keep:]
                    else:
                        self._visible_buffer = ""
                    break

                self._visible_buffer = self._visible_buffer[close_idx + len(self._closing):]
                self._in_tag = False
                self._closing = ""
                continue

            bracket_idx = self._visible_buffer.find("[")
            angle_idx = self._visible_buffer.find("<")
            indices = [idx for idx in (bracket_idx, angle_idx) if idx != -1]
            if not indices:
                self._pending_visible += self._visible_buffer
                self._visible_buffer = ""
                break
            tag_idx = min(indices)

            if tag_idx > 0:
                self._pending_visible += self._visible_buffer[:tag_idx]
                self._visible_buffer = self._visible_buffer[tag_idx:]

            partial_match = False
            for opener, closer in self._openers:
                if self._visible_buffer.startswith(opener):
                    self._visible_buffer = self._visible_buffer[len(opener):]
                    self._in_tag = True
                    self._closing = closer
                    partial_match = False
                    break
                if opener.startswith(self._visible_buffer):
                    partial_match = True
                    break

            if self._in_tag:
                continue

            if partial_match:
                break

            self._pending_visible += "["
            self._visible_buffer = self._visible_buffer[1:]

        return self._drain_pending()

    def flush(self) -> str:
        if self._in_tag:
            return ""
        self._pending_visible += self._visible_buffer
        self._visible_buffer = ""
        return self._drain_pending(flush=True)


def _collect_memory_and_tool_ops(text: str) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[str],
    list[tuple[str, str]],
    list[str],
]:
    mem_ops: list[tuple[str, str]] = []
    observe_ops: list[tuple[str, str]] = []
    forget_queries: list[str] = []
    project_ops: list[tuple[str, str]] = []
    editor_ops = extract_editor_commands(text)

    for match in MEM_PATTERN.finditer(text):
        raw = match.group(1).strip()
        if ":" in raw:
            cat, content = raw.split(":", 1)
            cat = cat.strip().lower()
            content = content.strip()
            mem_cat = CATEGORY_MAP.get(cat)
            if mem_cat and content:
                mem_ops.append((mem_cat, content))

    for match in OBSERVE_PATTERN.finditer(text):
        content = match.group(1).strip()
        if content:
            observe_ops.append((content, _detect_weight(content)))

    for match in FORGET_PATTERN.finditer(text):
        query = match.group(1).strip()
        if query:
            forget_queries.append(query)

    for match in PROJECT_PATTERN.finditer(text):
        raw = match.group(1).strip()
        if ":" in raw:
            name, detail = raw.split(":", 1)
            name = name.strip().lower()
            detail = detail.strip().lower()
            project_ops.append((name, detail))
        elif raw:
            project_ops.append((raw.strip().lower(), ""))

    return mem_ops, observe_ops, forget_queries, project_ops, editor_ops


def apply_response_side_effects(text: str) -> None:
    """Apply memory and editor side effects without mutating visible text."""
    mem_ops, observe_ops, forget_queries, project_ops, editor_ops = _collect_memory_and_tool_ops(text)

    filtered_mem_ops = []
    for mem_cat, content in mem_ops:
        if mem_cat != "user":
            filtered_mem_ops.append((mem_cat, content))
            continue
        if ":" in content:
            filtered_mem_ops.append((mem_cat, content))
        elif _looks_like_name(content):
            filtered_mem_ops.append((mem_cat, f"name: {content.strip()}"))

    if filtered_mem_ops or observe_ops or forget_queries or project_ops:
        apply_tag_operations(
            mem_ops=filtered_mem_ops,
            observe_ops=observe_ops,
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
    requests: list[str] = []
    for match in ASK_CODER_PATTERN.finditer(str(text or "")):
        content = (match.group(1) or "").strip()
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
    editor_ops: list[str] = []

    for match in EDITOR_PATTERN.finditer(text):
        raw = (match.group(1) or "").strip()
        if raw:
            editor_ops.append(raw)

    for block in _extract_tool_call_blocks(text):
        editor_ops.extend(_extract_editor_commands_from_tool_block(block))

    return editor_ops


def extract_read_requests(text: str) -> list[str]:
    read_requests: list[str] = []
    seen: set[str] = set()

    for match in TOOL_PATTERNS["READ"].finditer(text):
        filepath = (match.group(1) or "").strip()
        if filepath and filepath not in seen:
            seen.add(filepath)
            read_requests.append(filepath)

    for block in _extract_tool_call_blocks(text):
        function_name, inner = _extract_function_tool_block(block)
        if function_name != "read":
            continue
        parameters = _extract_tool_parameters(inner)
        filepath = _first_parameter(parameters, "file", "path", "value", "argument")
        if filepath and filepath not in seen:
            seen.add(filepath)
            read_requests.append(filepath)

    for match in XML_READ_PATTERN.finditer(text):
        filepath = (match.group(1) or "").strip()
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
    return [(match.group(1) or "").strip() for match in TOOL_CALL_PATTERN.finditer(text)]


def _extract_function_tool_block(block: str) -> tuple[str, str]:
    match = FUNCTION_TOOL_BLOCK_PATTERN.search(block)
    if not match:
        return "", ""
    return (match.group(1) or "").strip().lower(), (match.group(2) or "")


def _extract_tool_parameters(block: str) -> dict[str, str]:
    return {
        (match.group(1) or "").strip().lower(): (match.group(2) or "").strip()
        for match in FUNCTION_PARAMETER_PATTERN.finditer(block)
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
            (
                (nested_match.group(1) or "").strip().lower(),
                (nested_match.group(2) or "").strip(),
            )
            for nested_match in NESTED_TOOL_ACTION_PATTERN.finditer(inner)
        )
        if command
    ]


def _extract_editor_commands_from_tool_block(block: str) -> list[str]:
    editor_match = EDITOR_TOOL_BLOCK_PATTERN.search(block)
    if editor_match:
        inner = (editor_match.group(1) or "").strip()
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


def build_runtime_context(*, include_memory: bool = True, include_editor: bool = True) -> str:
    memory_context = format_for_prompt() if include_memory else ""
    editor_context = get_editor_bridge().format_for_prompt() if include_editor else ""
    parts = []
    if memory_context:
        parts.append(memory_context)
    if editor_context:
        parts.append(editor_context)
    return "\n\n".join(parts)


def _cached_system_prompt(runtime_context: str) -> str:
    global _system_prompt_cache, _system_prompt_gen, _runtime_context_cache

    store = get_store()
    current_gen = store._prompt_cache_gen
    if (
        _system_prompt_cache is not None
        and _system_prompt_gen == current_gen
        and _runtime_context_cache == runtime_context
    ):
        return _system_prompt_cache

    system_prompt = build_system_prompt(runtime_context)
    _system_prompt_cache = system_prompt
    _system_prompt_gen = current_gen
    _runtime_context_cache = runtime_context
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
            trimmed.insert(0, msg)
            current_tokens += msg_tokens
            continue
        if len(trimmed) < min_recent_messages:
            trimmed.insert(0, msg)
            current_tokens += msg_tokens
            continue
        break

    # Build final message list
    chat_messages = [{"role": "system", "content": system_prompt}]
    chat_messages.extend(trimmed)
    chat_messages.append({"role": "user", "content": user_message})

    return chat_messages
