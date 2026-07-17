"""Attach the current VS Code editor snapshot to code-related prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.integrations.vscode_workspace import EditorSnapshot, latest_editor_context

MAX_PROMPT_CONTEXT_CHARS = 5_500
EDITOR_CONTEXT_UNAVAILABLE_REPLY = "I don't have a current VS Code snapshot."
_CODE_REQUEST = re.compile(
    r"\b(code|debug|error|exception|traceback|file|document|function|class|method|module|"
    r"script|test|bug|vscode|vs code|editor|implementation|refactor)\b",
    re.IGNORECASE,
)
_CODE_ACTION = re.compile(
    r"\b(look|inspect|review|check|explain|fix|debug|improve|change|refactor|"
    r"implement|simplify|why|how|help|current|selected|open|opened|active|editing|"
    r"wrong|broken|failing)\b",
    re.IGNORECASE,
)
_ACTIVE_FILE_STATE = re.compile(r"\b(open|opened|active|editing|editor)\b", re.IGNORECASE)
_FILE_OR_DOCUMENT = re.compile(r"\b(file|document)\b", re.IGNORECASE)
_DIRECT_QUESTION = re.compile(r"\b(what|which|name|identify)\b", re.IGNORECASE)
_BROADER_CODE_REQUEST = re.compile(
    r"\b(review|inspect|check|explain|fix|debug|improve|change|refactor|implement|"
    r"simplify|summarize|analyze|why|how|wrong|broken|failing|error|issue|"
    r"contents?|contains?|purpose|changes?|code|think|what\s+(?:does|can|should|would))\b",
    re.IGNORECASE,
)
_FILE_REFERENCE = re.compile(
    r"\b[\w./-]+\.(?:py|js|jsx|ts|tsx|java|c|cc|cpp|h|hpp|cs|go|rs|rb|"
    r"php|swift|kt|sh|html|css|vue|svelte|json|ya?ml|toml|md|sql)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CodeContext:
    requested: bool
    connected: bool
    prompt_text: str = ""


def is_direct_active_file_question(message: str) -> bool:
    text = str(message or "").strip()
    if (
        not text
        or _BROADER_CODE_REQUEST.search(text)
        or not _DIRECT_QUESTION.search(text)
    ):
        return False
    return bool(
        (_FILE_OR_DOCUMENT.search(text) and _ACTIVE_FILE_STATE.search(text))
        or re.search(r"\bediting\b", text, re.IGNORECASE)
        or (
            re.search(r"\beditor\b", text, re.IGNORECASE)
            and re.search(r"\b(open|opened|active)\b", text, re.IGNORECASE)
        )
    )


def active_file_reply(message: str) -> str | None:
    if not is_direct_active_file_question(message):
        return None
    snapshot = latest_editor_context()
    return snapshot.filename if snapshot is not None else EDITOR_CONTEXT_UNAVAILABLE_REPLY


def requests_code_context(message: str) -> bool:
    text = str(message or "").strip()
    if not text or text.lower().startswith("[autonomous discord post]"):
        return False
    return (
        is_direct_active_file_question(text)
        or bool(_FILE_REFERENCE.search(text))
        or bool(_CODE_REQUEST.search(text) and _CODE_ACTION.search(text))
        or bool(_ACTIVE_FILE_STATE.search(text) and _BROADER_CODE_REQUEST.search(text))
    )


def _format_context(snapshot: EditorSnapshot) -> str:
    lines = [
        "[VS CODE EDITOR CONTEXT — READ ONLY]",
        f"File: {snapshot.filename}",
        (
            "The File value above is the authoritative active filename; do not replace "
            "it with identity, memory, or code text."
        ),
        f"Language: {snapshot.language or 'unknown'}",
    ]
    if snapshot.current_symbol:
        lines.append(f"Current function or class: {snapshot.current_symbol}")
    if snapshot.event_detail:
        lines.append(f"Editor event: {snapshot.event_type} — {snapshot.event_detail}")
    if snapshot.diagnostics:
        lines.append("Diagnostics:")
        lines.extend(
            f"- {item.severity or 'diagnostic'} line {item.line}: {item.message}"
            for item in snapshot.diagnostics
        )
    if snapshot.recent_diff:
        lines.extend(("Recent diff (preferred evidence):", snapshot.recent_diff))
    if snapshot.selection:
        lines.extend(("Selected code:", snapshot.selection))
    elif snapshot.nearby_code:
        lines.extend(("Code near cursor:", snapshot.nearby_code))
    lines.append("Use only this supplied code; do not claim access to other files.")
    return "\n".join(lines)[:MAX_PROMPT_CONTEXT_CHARS]


def code_context_for_message(message: str) -> CodeContext:
    if not requests_code_context(message):
        return CodeContext(False, False)
    snapshot = latest_editor_context()
    if snapshot is None:
        return CodeContext(True, False)
    return CodeContext(True, True, _format_context(snapshot))
