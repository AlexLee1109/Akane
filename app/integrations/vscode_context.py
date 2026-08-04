"""Format the current bounded VS Code editor snapshot for a prompt."""

from __future__ import annotations

from dataclasses import dataclass

from app.integrations.vscode_workspace import EditorSnapshot, latest_editor_context

MAX_PROMPT_CONTEXT_CHARS = 5_500


@dataclass(frozen=True, slots=True)
class CodeContext:
    requested: bool
    connected: bool
    prompt_text: str = ""


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


def current_code_context() -> CodeContext:
    snapshot = latest_editor_context()
    if snapshot is None:
        return CodeContext(False, False)
    return CodeContext(True, True, _format_context(snapshot))
