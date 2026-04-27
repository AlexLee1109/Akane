import queue
import random
import re
import threading
from datetime import datetime

from app.config import (
    CHAT_HISTORY_CONTEXT_TOKENS,
    MAX_TOKENS,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from app.character import build_system_prompt
from app import memory
from app.config import ADVISOR_ONLY
from app.editor_bridge import get_editor_bridge
from app.memory import format_for_prompt, get_store
from app.memory_store import apply_tag_operations
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

INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")


def _strip_hidden_markup(text: str) -> str:
    cleaned = str(text or "")
    for pattern in (ALL_TAGS_PATTERN, EDITOR_TOOL_CALL_STRIP_PATTERN, XML_SIMPLE_TOOL_STRIP_PATTERN):
        cleaned = pattern.sub("", cleaned)
    return collapse_hidden_tag_gaps(cleaned).strip()


def _strip_inline_markdown_code(text: str) -> str:
    cleaned = str(text or "")
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = INLINE_CODE_PATTERN.sub(r"\1", cleaned)
    return cleaned.replace("`", "")

CATEGORY_MAP = {
    "name": "user",
    "fact": "facts",
    "facts": "facts",
    "preference": "preferences",
    "preferences": "preferences",
    "relationship": "relationships",
    "relationships": "relationships",
    "user": "user",
}

POSITIVE_WORDS = {
    "happy", "excited", "loves", "enjoy", "great", "awesome", "fantastic", "wonderful",
    "amazing", "fun", "passionate", " thrilled", "delighted", "pleased", "satisfied",
    "grateful", "thankful", "blessed", "joy", "love it", "perfect", "nailed it",
    "proud", "confident", "optimistic", "hopeful", "energized", "motivated",
    "inspired", "impressed", "good news", "achievement", "success"
}

NEGATIVE_WORDS = {
    "frustrated", "stressed", "hate", "struggling", "stuck", "annoyed", "worried",
    "anxious", "overwhelmed", "exhausted", "tired", "disappointed", "sad", "upset",
    "angry", "mad", "fuming", "annoying", "difficult", "hard", "impossible",
    "terrible", "awful", "horrible", "bad news", "failure", "mistake", "regret",
    "confused", "lost", "helpless", "hopeless", "burned out", "dread", "dreadful",
    "pain", "hurting", "sick", "ill", "unhappy", "miserable", "depressed"
}

NAME_PREFIXES = (
    "my name is ",
    "name is ",
    "name's ",
    "call me ",
    "i go by ",
)

CLAUSE_SPLIT_PATTERN = re.compile(r"[.!?;\n]+")
SELF_CLAUSE_SPLIT_PATTERN = re.compile(
    r"\s+(?:and|but)\s+(?=(?:i\b|i['’]?m\b|im\b|my name\b|name['’]?s\b|call me\b))",
    re.IGNORECASE,
)

PREFERENCE_HEADS = {"prefer", "like", "love", "enjoy", "want", "need", "hate"}
PROJECT_HEADS = {
    "work", "working", "build", "building", "make", "making", "create", "creating",
    "train", "training", "debug", "debugging", "fix", "fixing", "write", "writing",
    "learn", "learning", "study", "studying", "develop", "developing", "prepare",
    "preparing", "plan", "planning", "ship", "shipping", "deploy", "deploying",
}
FACT_HEADS = {"use", "using", "have", "having", "run", "running", "on", "with"}
AUXILIARY_HEADS = {"am", "are", "was", "were", "be", "been", "do", "does", "did"}

TECH_FACT_HINTS = {
    "mac", "macos", "windows", "linux", "colab", "cuda", "mps", "bf16",
    "bfloat16", "fp16", "gpu", "a100", "h100", "rtx", "vscode", "python",
    "pytorch", "llama", "gguf",
}

NON_DURABLE_FACT_HINTS = {
    "question", "questions", "problem", "problems", "issue", "issues",
    "idea", "ideas", "thing", "things", "minute", "minutes",
}


def _detect_weight(text: str) -> str:
    """Detect emotional weight of observation content."""
    lower = text.lower()
    # Count positive and negative word matches
    pos_count = sum(1 for w in POSITIVE_WORDS if w in lower)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in lower)

    # Stronger signals: multiple hits or very strong words
    if neg_count > 0 and neg_count >= pos_count:
        return memory.NEGATIVE
    if pos_count > 0 and pos_count > neg_count:
        return memory.POSITIVE
    return memory.NEUTRAL


def _looks_like_name(value: str) -> bool:
    """Heuristic guard so random user facts do not overwrite the name field."""
    candidate = value.strip()
    if not candidate or len(candidate) > 40:
        return False
    lowered = candidate.lower()
    forbidden_fragments = {
        "working on",
        "debugging",
        "memory",
        "project",
        "prefer",
        "likes",
        "loves",
        "using",
        "building",
        "coding",
        "issue",
        "bug",
    }
    if any(fragment in lowered for fragment in forbidden_fragments):
        return False
    parts = candidate.replace("-", " ").split()
    if not 1 <= len(parts) <= 3:
        return False
    return all(part.isalpha() for part in parts)


def _normalize_memory_text(value: str) -> str:
    value = " ".join(value.strip().split())
    value = re.sub(r"^[,:;\-\s]+", "", value)
    value = re.sub(r"[,:;\-\s]+$", "", value)
    return value


def _canonicalize_clause(clause: str) -> str:
    clause = clause.replace("’", "'")
    clause = re.sub(r"\bi'm\b", "i am", clause, flags=re.IGNORECASE)
    clause = re.sub(r"\bim\b", "i am", clause, flags=re.IGNORECASE)
    clause = re.sub(r"\bdon't\b", "do not", clause, flags=re.IGNORECASE)
    clause = re.sub(r"\bcan't\b", "can not", clause, flags=re.IGNORECASE)
    return " ".join(clause.split())


def _extract_self_clauses(text: str) -> list[str]:
    clauses: list[str] = []
    for chunk in CLAUSE_SPLIT_PATTERN.split(text):
        chunk = _normalize_memory_text(chunk)
        if not chunk:
            continue
        for clause in SELF_CLAUSE_SPLIT_PATTERN.split(chunk):
            clause = _normalize_memory_text(clause)
            if not clause:
                continue
            normalized = _canonicalize_clause(clause).lower()
            if normalized.startswith("i ") or normalized.startswith(("my name", "name's ", "call me ")):
                clauses.append(clause)
                continue
            first_word = normalized.split(" ", 1)[0]
            if first_word in PROJECT_HEADS or _stem_head(first_word) in PROJECT_HEADS:
                clauses.append(clause)
    return clauses


def _normalize_project_phrase(value: str) -> tuple[str, str]:
    """Split a project phrase into a short name plus optional detail."""
    value = _normalize_memory_text(value)
    value = re.sub(r"^(?:a|an|the|my)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:project|app|tool)\s+called\s+", "", value, flags=re.IGNORECASE)
    if not value:
        return "", ""

    parts = re.split(r"\b(?:using|with|for|on|at)\b", value, maxsplit=1, flags=re.IGNORECASE)
    name = _normalize_memory_text(parts[0]).lower()
    detail = _normalize_memory_text(parts[1]).lower() if len(parts) > 1 else ""
    if len(name) > 60:
        name = name[:60].rsplit(" ", 1)[0] or name[:60]
    return name, detail


def _looks_like_technical_fact(value: str) -> bool:
    lowered = value.lower().strip()
    if lowered in NON_DURABLE_FACT_HINTS:
        return False
    if any(lowered.startswith(f"{hint} ") or lowered.endswith(f" {hint}") for hint in NON_DURABLE_FACT_HINTS):
        return False
    if any(hint in lowered for hint in TECH_FACT_HINTS):
        return True
    return any(char.isdigit() for char in value)


def _extract_name_from_clause(clause: str) -> str | None:
    lowered = clause.lower()
    for prefix in NAME_PREFIXES:
        if lowered.startswith(prefix):
            candidate = _normalize_memory_text(clause[len(prefix):])
            return candidate if _looks_like_name(candidate) else None
    return None


def _stem_head(word: str) -> str:
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def _parse_statement(clause: str) -> tuple[str, str, bool] | None:
    normalized = _canonicalize_clause(clause).lower()
    tokens = normalized.split()
    if not tokens:
        return None

    idx = 0
    negated = False

    if tokens[0] == "i":
        idx = 1
        while idx < len(tokens) and tokens[idx] in AUXILIARY_HEADS:
            idx += 1

    if idx < len(tokens) and tokens[idx] == "not":
        negated = True
        idx += 1
    if idx >= len(tokens):
        return None

    head = tokens[idx]
    remainder = _normalize_memory_text(" ".join(tokens[idx + 1:]))
    return head, remainder, negated


def _preference_memory(head: str, remainder: str, negated: bool) -> str | None:
    if not remainder:
        return None
    value = remainder.lower()
    base = _stem_head(head)
    if base == "want":
        prefix = "doesn't want" if negated else "wants"
        return f"{prefix} {value}"
    if base == "need":
        prefix = "doesn't need" if negated else "needs"
        return f"{prefix} {value}"
    if base in {"like", "love", "enjoy"} and not negated:
        return f"{base}s {value}"
    if base == "hate" or negated:
        return f"doesn't like {value}"
    if base == "prefer":
        return f"prefers {value}"
    return None


def _project_memory(head: str, remainder: str) -> tuple[str, str] | None:
    if not remainder:
        return None
    value = remainder
    if _stem_head(head) == "work" and value.lower().startswith("on "):
        value = value[3:]
    name, detail = _normalize_project_phrase(value)
    if not name:
        return None
    return name, detail


def _fact_memory(head: str, remainder: str) -> str | None:
    if not remainder:
        return None
    value = _normalize_memory_text(remainder)
    if not _looks_like_technical_fact(value):
        return None
    base = _stem_head(head)
    if base == "use":
        return f"uses {value.lower()}"
    if base == "have":
        return f"has {value.lower()}"
    return f"{base} {value.lower()}"


def capture_explicit_user_memories(user_text: str) -> bool:
    """Store important explicit user details directly from the raw user message."""
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

        parsed = _parse_statement(clause)
        if not parsed:
            continue
        head, remainder, negated = parsed
        base = _stem_head(head)

        if head in PROJECT_HEADS or base in PROJECT_HEADS:
            project = _project_memory(head, remainder)
            if project:
                project_ops.append(project)
                changed = True
            continue

        if head in PREFERENCE_HEADS or base in PREFERENCE_HEADS:
            preference = _preference_memory(head, remainder, negated)
            if preference:
                mem_ops.append(("preferences", preference))
                changed = True
            continue

        if head in FACT_HEADS or base in FACT_HEADS:
            fact = _fact_memory(head, remainder)
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
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n(?:[ \t]*\n)+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


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


def parse_and_store_tags(text: str) -> str:
    """Backward-compatible wrapper for finalizing a model response."""
    return finalize_model_response(text)


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


def _extract_editor_commands_from_tool_block(block: str) -> list[str]:
    editor_ops: list[str] = []

    editor_match = EDITOR_TOOL_BLOCK_PATTERN.search(block)
    if editor_match:
        inner = (editor_match.group(1) or "").strip()
        parameters = _extract_tool_parameters(inner)
        command = _first_parameter(parameters, "action", "command", "name")
        argument = _first_parameter(parameters, "value", "argument", "file", "path", "text")
        if command:
            editor_ops.append(command.lower() if not argument else f"{command.lower()}: {argument}")
            return editor_ops
        for nested_match in NESTED_TOOL_ACTION_PATTERN.finditer(inner):
            command = (nested_match.group(1) or "").strip().lower()
            argument = (nested_match.group(2) or "").strip()
            if command:
                editor_ops.append(command if not argument else f"{command}: {argument}")
        return editor_ops

    function_name, inner = _extract_function_tool_block(block)
    if function_name == "editor":
        parameters = _extract_tool_parameters(inner)
        command = _first_parameter(parameters, "action", "command", "name")
        argument = _first_parameter(parameters, "value", "argument", "file", "path", "text")
        if command:
            editor_ops.append(command.lower() if not argument else f"{command.lower()}: {argument}")
            return editor_ops
        for nested_match in NESTED_TOOL_ACTION_PATTERN.finditer(inner):
            command = (nested_match.group(1) or "").strip().lower()
            argument = (nested_match.group(2) or "").strip()
            if command:
                editor_ops.append(command if not argument else f"{command}: {argument}")
        return editor_ops

    if function_name in {"open_vscode", "open_project", "open_workspace"}:
        editor_ops.append(function_name)

    return editor_ops


# Cache for system prompt to avoid repeated formatting
_system_prompt_cache = None
_system_prompt_gen = -1
_runtime_context_cache = ""


def build_runtime_context() -> str:
    now = datetime.now().astimezone()
    time_context = "\n".join(
        [
            "CURRENT TIME:",
            f"- Local date: {now.strftime('%A, %B %d, %Y')}",
            f"- Local time: {now.strftime('%I:%M %p').lstrip('0')}",
            f"- Timezone: {now.tzname() or 'local time'}",
        ]
    )
    memory_context = format_for_prompt()
    editor_context = get_editor_bridge().format_for_prompt()
    parts = [time_context]
    if memory_context:
        parts.append(memory_context)
    if editor_context:
        parts.append(editor_context)
    return "\n\n".join(parts)


def build_messages(user_message: str) -> list[dict]:
    """Build chat messages with cached system prompt when memory hasn't changed."""
    global _system_prompt_cache, _system_prompt_gen, _runtime_context_cache

    runtime_context = build_runtime_context()
    store = memory.get_store()
    current_gen = store._prompt_cache_gen

    if (
        _system_prompt_cache is not None
        and _system_prompt_gen == current_gen
        and _runtime_context_cache == runtime_context
    ):
        system_prompt = _system_prompt_cache
    else:
        system_prompt = build_system_prompt(runtime_context)
        _system_prompt_cache = system_prompt
        _system_prompt_gen = current_gen
        _runtime_context_cache = runtime_context

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def generate_stream(user_message: str, token_queue: queue.Queue) -> str:
    messages = build_messages(user_message)

    stream = LLM.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repeat_penalty=REPETITION_PENALTY,
        stream=True,
    )

    full_response = []
    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta and delta["content"]:
            token_queue.put(delta["content"])
            full_response.append(delta["content"])

    token_queue.put(None)
    return "".join(full_response)


def run_generation(user_message: str, token_queue: queue.Queue) -> str:
    with _generation_lock:
        raw_response = generate_stream(user_message, token_queue)

    cleaned = parse_and_store_tags(raw_response)
    return cleaned


def generate_proactive(messages_history: list[dict]) -> str | None:
    """Generate an unprompted comment based on memory/context and relationship."""
    if not messages_history:
        return None

    from app.memory import get_relationship_context

    runtime_context = build_runtime_context()
    store = memory.get_store()

    # Use cached system prompt if available
    global _system_prompt_cache, _system_prompt_gen, _runtime_context_cache
    current_gen = store._prompt_cache_gen
    if (
        _system_prompt_cache is not None
        and _system_prompt_gen == current_gen
        and _runtime_context_cache == runtime_context
    ):
        system_prompt = _system_prompt_cache
    else:
        system_prompt = build_system_prompt(runtime_context)
        _system_prompt_cache = system_prompt
        _system_prompt_gen = current_gen
        _runtime_context_cache = runtime_context

    rel_ctx = get_relationship_context()
    familiarity = rel_ctx.get("familiarity", 0.0)

    # Context-aware proactive styles based on relationship and recent conversation
    if familiarity < 0.3:
        # Stranger/early acquaintance - cautious, helpful
        styles = [
            "You have a gentle follow-up question or helpful thought. Keep it warm but brief. No tags.",
            "Offer a small piece of encouragement or acknowledgement based on what they said.",
            "Ask a simple clarifying question to show you're engaged.",
        ]
    elif familiarity < 0.6:
        # Friend - more personal, occasional humor
        styles = [
            "Share a brief thought that just occurred to you based on the conversation.",
            "Offer encouragement or a gentle observation. Be friendly.",
            "Ask a follow-up question that shows you're actually listening.",
            "Mention something you remember slightly, if relevant.",
        ]
    else:
        # Close friend - casual, authentic, can tease lightly
        styles = [
            "Mutter a passing thought out loud. Be natural, like you're talking to a friend.",
            "React naturally to what they said — agreement, surprise, concern, etc.",
            "Ask a casual follow-up question, like you're genuinely curious.",
            "Reference something from your memories if it's relevant. Feel like talking to someone you know well.",
            "If they seem stuck or uncertain, offer a gentle nudge or encouraging word.",
        ]

    proactive_prompt = random.choice(styles)

    # Use truncation to ensure we fit in context window
    chat_messages = truncate_messages(
        list(messages_history) if messages_history else [],
        system_prompt,
        proactive_prompt,
        max_context_tokens=CHAT_HISTORY_CONTEXT_TOKENS
    )

    with _generation_lock:
        result = LLM.create_chat_completion(
            messages=chat_messages,
            max_tokens=96,  # Slightly longer for better quality
            temperature=0.8,  # More creative/less rigid
            top_k=TOP_K,
            top_p=TOP_P,
            repeat_penalty=REPETITION_PENALTY,
        )

    raw = result["choices"][0]["message"]["content"].strip()
    if not raw:
        return None

    cleaned = parse_and_store_tags(raw)
    return cleaned if cleaned else None


def flush_memories() -> None:
    memory.flush_now()


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

    # Need to truncate: remove oldest messages until we fit
    # Keep messages in pairs (user + assistant) to maintain conversation structure
    trimmed = []
    current_tokens = 0

    for msg in reversed(messages_list):
        msg_tokens = len(msg.get("content", "")) // 4
        if current_tokens + msg_tokens <= available_for_history:
            trimmed.insert(0, msg)  # Insert at beginning to maintain order
            current_tokens += msg_tokens
        else:
            break

    # Build final message list
    chat_messages = [{"role": "system", "content": system_prompt}]
    chat_messages.extend(trimmed)
    chat_messages.append({"role": "user", "content": user_message})

    return chat_messages
