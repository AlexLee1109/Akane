"""Attach the current VS Code editor snapshot to code-related prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.integrations.vscode_workspace import EditorSnapshot, latest_editor_context

MAX_PROMPT_CONTEXT_CHARS = 7_000
_CODE_REQUEST = re.compile(
    r"\b(code|debug|error|exception|traceback|file|function|class|method|module|"
    r"script|test|bug|vscode|vs code|editor|implementation|refactor)\b",
    re.IGNORECASE,
)
_CODE_ACTION = re.compile(
    r"\b(look|inspect|review|check|explain|fix|debug|improve|change|refactor|"
    r"implement|simplify|why|how|help|current|selected|wrong|broken|failing)\b",
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


def requests_code_context(message: str) -> bool:
    text = str(message or "").strip()
    if not text or text.lower().startswith("[autonomous discord post]"):
        return False
    return bool(_FILE_REFERENCE.search(text)) or bool(
        _CODE_REQUEST.search(text) and _CODE_ACTION.search(text)
    )


def _format_context(snapshot: EditorSnapshot) -> str:
    lines = [
        "[VS CODE EDITOR CONTEXT — READ ONLY]",
        f"File: {snapshot.filename}",
        f"Language: {snapshot.language or 'unknown'}",
    ]
    if snapshot.selection:
        lines.extend(("Selected code:", snapshot.selection))
    elif snapshot.nearby_code:
        lines.extend(("Code near cursor:", snapshot.nearby_code))
    if snapshot.diagnostics:
        lines.append("Diagnostics:")
        lines.extend(
            f"- {item.severity or 'diagnostic'} line {item.line}: {item.message}"
            for item in snapshot.diagnostics
        )
    lines.append("Use only this supplied code; do not claim access to other files.")
    return "\n".join(lines)[:MAX_PROMPT_CONTEXT_CHARS]


def code_context_for_message(message: str) -> CodeContext:
    if not requests_code_context(message):
        return CodeContext(False, False)
    snapshot = latest_editor_context()
    if snapshot is None:
        return CodeContext(True, False)
    return CodeContext(True, True, _format_context(snapshot))
