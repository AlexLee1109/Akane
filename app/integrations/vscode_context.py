"""Resolve chat requests into bounded VS Code file context."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from app.integrations.vscode_workspace import (
    WorkspaceDocument,
    WorkspaceSnapshot,
    latest_workspace_context,
    request_documents,
    workspace_status,
)

MAX_PROMPT_CONTEXT_CHARS = 3_000
MAX_RELATED_FILES = 2

_CODE_NOUNS = (
    "code", "codebase", "workspace", "implementation", "function", "class",
    "module", "script", "bug", "error", "traceback", "refactor", "vscode",
    "vs code",
)
_CODE_ACTIONS = (
    "look", "inspect", "review", "check", "explain", "fix", "debug", "improve",
    "suggest", "refactor", "implement", "why", "what", "how", "help", "current",
    "selected", "structure",
)
_FILE_ACTIONS = (
    "look", "inspect", "review", "check", "explain", "fix", "debug", "improve",
    "suggest", "refactor", "why", "current", "selected", "structure",
    "not working", "broken",
)
_DIRECT_CODE_PHRASES = (
    "project structure", "file tree", "workspace structure", "what files",
    "this file", "current file", "selected code", "vs code project",
    "vscode project",
)
_FILE_REFERENCE = re.compile(
    r"\b[\w./-]+\.(?:py|js|jsx|ts|tsx|java|c|cc|cpp|h|hpp|cs|go|rs|rb|php|"
    r"swift|kt|kts|scala|sh|bash|zsh|fish|html|css|scss|sass|vue|svelte|json|"
    r"yaml|yml|toml|ini|cfg|md|sql|graphql|proto)\b",
    re.IGNORECASE,
)
_PATH_REFERENCE = re.compile(r"\b(?:[\w.-]+/)+[\w.-]+\b")
_NAMED_FILE = re.compile(r"\b([\w.-]+)\s+(?:file|module)\b", re.IGNORECASE)
_PY_IMPORT = re.compile(r"^\s*import\s+(.+)$", re.MULTILINE)
_PY_FROM_IMPORT = re.compile(r"^\s*from\s+([.\w]+)\s+import\s+", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class CodeContext:
    requested: bool
    connected: bool
    prompt_text: str = ""
    direct_reply: str = ""


def _clip_middle(text: str, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    marker = "\n...[truncated]...\n"
    remaining = max(0, limit - len(marker))
    start = (remaining * 2) // 3
    return value[:start].rstrip() + marker + value[-(remaining - start):].lstrip()


def requests_code_context(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text or text.startswith("[autonomous discord post]"):
        return False
    if _FILE_REFERENCE.search(text) or _PATH_REFERENCE.search(text) or any(
        phrase in text for phrase in _DIRECT_CODE_PHRASES
    ):
        return True
    if any(noun in text for noun in _CODE_NOUNS):
        return any(action in text for action in _CODE_ACTIONS)
    return any(noun in text for noun in ("file", "project")) and any(
        action in text for action in _FILE_ACTIONS
    )


def _references(message: str) -> list[str]:
    references = [match.group(0) for match in _FILE_REFERENCE.finditer(message)]
    references.extend(match.group(0) for match in _PATH_REFERENCE.finditer(message))
    references.extend(match.group(1) for match in _NAMED_FILE.finditer(message))
    result: list[str] = []
    for reference in references:
        value = reference.strip("`'\".,:;!?()[]{}").replace("\\", "/")
        if value and value.lower() not in {"this", "current", "project"} and value not in result:
            result.append(value)
    return result


def _path_score(reference: str, path: str) -> float:
    ref = reference.lower().strip("/")
    candidate = path.lower()
    ref_no_ext = ref.rsplit(".", 1)[0] if "." in ref.rsplit("/", 1)[-1] else ref
    path_no_ext = candidate.rsplit(".", 1)[0] if "." in candidate.rsplit("/", 1)[-1] else candidate
    basename = candidate.rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0]
    if ref == candidate:
        return 1.0
    if ref_no_ext == path_no_ext:
        return 0.98
    if ref == basename:
        return 0.97
    if ref_no_ext == stem:
        return 0.95
    if candidate.endswith("/" + ref) or path_no_ext.endswith("/" + ref_no_ext):
        return 0.93
    score = max(
        difflib.SequenceMatcher(None, ref, candidate).ratio(),
        difflib.SequenceMatcher(None, ref_no_ext, stem).ratio() * 0.9,
    )
    ref_name = ref.rsplit("/", 1)[-1]
    candidate_name = candidate.rsplit("/", 1)[-1]
    if "." in ref_name and ref_name.rsplit(".", 1)[-1] != candidate_name.rsplit(".", 1)[-1]:
        score = min(score, 0.65)
    return score


def _resolve_paths(
    snapshot: WorkspaceSnapshot,
    message: str,
) -> tuple[list[str], list[str], bool]:
    references = _references(message)
    resolved: list[str] = []
    suggestions: list[str] = []
    for reference in references:
        ranked = sorted(
            ((_path_score(reference, path), path) for path in snapshot.file_tree),
            key=lambda item: (-item[0], len(item[1]), item[1]),
        )
        if ranked and ranked[0][0] >= 0.70:
            if ranked[0][1] not in resolved:
                resolved.append(ranked[0][1])
        else:
            for score, path in ranked[:3]:
                if score >= 0.35 and path not in suggestions:
                    suggestions.append(path)

    lower = message.lower()
    current_reference = any(
        phrase in lower for phrase in ("this file", "current file", "this module")
    )
    if not resolved and (current_reference or not references) and snapshot.active_file:
        resolved.append(snapshot.active_file)
    return resolved, suggestions[:3], bool(references)


def _module_candidates(path: str, module: str) -> tuple[str, ...]:
    current = PurePosixPath(path)
    dots = len(module) - len(module.lstrip("."))
    name = module.lstrip(".")
    base = current.parent
    for _ in range(max(0, dots - 1)):
        base = base.parent
    parts = tuple(part for part in name.split(".") if part)
    target = base.joinpath(*parts) if dots else PurePosixPath(*parts)
    value = target.as_posix()
    return (f"{value}.py", f"{value}/__init__.py")


def _python_import_paths(
    document: WorkspaceDocument,
    indexed_paths: tuple[str, ...],
) -> list[str]:
    if not document.path.endswith(".py"):
        return []
    modules: list[str] = []
    for match in _PY_IMPORT.finditer(document.content):
        modules.extend(
            part.strip().split(" as ", 1)[0].strip()
            for part in match.group(1).split(",")
            if part.strip()
        )
    modules.extend(match.group(1) for match in _PY_FROM_IMPORT.finditer(document.content))
    indexed = set(indexed_paths)
    related: list[str] = []
    for module in modules:
        for candidate in _module_candidates(document.path, module):
            if candidate in indexed and candidate != document.path and candidate not in related:
                related.append(candidate)
                break
        if len(related) >= MAX_RELATED_FILES:
            break
    return related


def _format_context(
    snapshot: WorkspaceSnapshot,
    documents: tuple[WorkspaceDocument, ...],
    message: str,
) -> str:
    lower = message.lower()
    structure_request = any(
        phrase in lower
        for phrase in ("project structure", "file tree", "workspace structure", "what files")
    )
    lines = [
        "[VS CODE WORKSPACE CONTEXT — READ ONLY]",
        f"Workspace: {snapshot.workspace_name}",
    ]
    if snapshot.languages:
        lines.append("Languages: " + ", ".join(snapshot.languages))
    if snapshot.active_file:
        lines.append(f"Active file: {snapshot.active_file}")
    if structure_request or not documents:
        lines.append("File tree: " + _clip_middle(", ".join(snapshot.file_tree), 900))
    if snapshot.selection and snapshot.active_file in {doc.path for doc in documents}:
        lines.extend(["Selected text:", _clip_middle(snapshot.selection, 500)])

    remaining = MAX_PROMPT_CONTEXT_CHARS - len("\n".join(lines)) - 120
    for index, document in enumerate(documents):
        if remaining <= 120:
            break
        budget = min(1_350 if index == 0 else 650, remaining)
        lines.extend(
            [
                f"File ({document.path}, {document.language or 'text'}):",
                _clip_middle(document.content, budget),
            ]
        )
        remaining -= budget
    lines.append("Use only these supplied files; do not claim access to other contents.")
    return _clip_middle("\n".join(lines), MAX_PROMPT_CONTEXT_CHARS)


def code_context_for_message(message: str) -> CodeContext:
    if not requests_code_context(message):
        return CodeContext(requested=False, connected=False)
    snapshot = latest_workspace_context()
    if snapshot is None or not workspace_status()["connected"]:
        return CodeContext(requested=True, connected=False)

    lower = message.lower()
    structure_request = any(
        phrase in lower
        for phrase in ("project structure", "file tree", "workspace structure", "what files")
    )
    if structure_request and not _references(message):
        return CodeContext(True, True, _format_context(snapshot, (), message))

    resolved, suggestions, had_reference = _resolve_paths(snapshot, message)
    if had_reference and not resolved:
        detail = (
            " Closest matches: " + ", ".join(suggestions) + "."
            if suggestions
            else ""
        )
        return CodeContext(
            True,
            True,
            direct_reply="I couldn’t find that file in the VS Code workspace index." + detail,
        )
    if not resolved:
        return CodeContext(True, True, _format_context(snapshot, (), message))

    documents = request_documents(resolved[:1])
    if not documents:
        return CodeContext(
            True,
            True,
            direct_reply=(
                f"VS Code found {resolved[0]}, but its contents were unavailable. "
                "Check that the Akane Local extension is still connected."
            ),
        )
    primary = documents[0]
    related_paths = _python_import_paths(primary, snapshot.file_tree)
    if related_paths:
        related = request_documents(related_paths)
        documents = tuple([primary, *[doc for doc in related if doc.path != primary.path]])
    return CodeContext(True, True, _format_context(snapshot, documents, message))
