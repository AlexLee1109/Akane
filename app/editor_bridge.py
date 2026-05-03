"""Structured localhost editor bridge for VS Code integration."""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

_APPROVAL_REQUIRED_ACTIONS = {
    "create_file",
    "write_file",
    "append_file",
    "replace_file_range",
    "replace_selection",
    "insert_text",
    "format_document",
    "save_file",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim_text(value: str, limit: int = 500) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def _project_root() -> Path:
    return Path.cwd().resolve()


def _resolve_project_path(path_text: str) -> Path | None:
    raw = str(path_text or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (_project_root() / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not str(candidate).startswith(str(_project_root())):
        return None
    return candidate


def _read_local_text(path_text: str) -> str:
    resolved = _resolve_project_path(path_text)
    if not resolved or not resolved.exists() or not resolved.is_file():
        return ""
    try:
        return resolved.read_text(encoding="utf-8")
    except Exception:
        return ""


def _line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def _slice_line_range(text: str, start_line: int, end_line: int) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    start_index = max(start_line - 1, 0)
    end_index = max(end_line, start_index)
    return "\n".join(lines[start_index:end_index])


def _parse_file_write_argument(argument: str) -> tuple[str, str]:
    header, _, content = str(argument or "").partition("\n")
    return header.strip(), content


def _parse_range_write_argument(argument: str) -> tuple[str, int | None, int | None, str]:
    header, _, content = str(argument or "").partition("\n")
    parts = header.split(":")
    if len(parts) < 3:
        return header.strip(), None, None, content
    file_path = ":".join(parts[:-2]).strip()
    try:
        start_line = int(parts[-2])
        end_line = int(parts[-1])
    except ValueError:
        return file_path, None, None, content
    return file_path, start_line, end_line, content


def _preview_text_block(text: str, *, max_lines: int = 14, max_chars: int = 900) -> str:
    value = str(text or "").replace("\r\n", "\n").strip("\n")
    if not value:
        return "(empty)"
    lines = value.splitlines()
    clipped_lines = lines[:max_lines]
    clipped = "\n".join(clipped_lines)
    if len(clipped) > max_chars:
        clipped = clipped[: max_chars - 1].rstrip() + "..."
    if len(lines) > max_lines or len(value) > len(clipped):
        return clipped.rstrip() + "\n..."
    return clipped


class EditorBridge:
    """Thread-safe store for editor context and queued actions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._next_action_id = 1
        self._prompt_cache = ""
        self._prompt_cache_dirty = True
        self._context = {
            "connected": False,
            "client": "",
            "workspace_name": "",
            "workspace_folders": [],
            "active_file": "",
            "language_id": "",
            "open_tabs": [],
            "file_excerpt": {
                "text": "",
                "start_line": None,
                "end_line": None,
            },
            "selection": {
                "text": "",
                "start_line": None,
                "start_character": None,
                "end_line": None,
                "end_character": None,
            },
            "diagnostics": [],
            "updated_at": None,
        }
        self._actions: list[dict] = []
        self._results = deque(maxlen=12)

    def _mark_prompt_dirty(self) -> None:
        self._prompt_cache_dirty = True

    def _build_preview_meta(self, command: str, argument: str) -> dict:
        preview: dict = {}
        active_file = str(self._context.get("active_file") or "").strip()
        selection = self._context.get("selection") or {}

        if command == "replace_file_range":
            file_path, start_line, end_line, content = _parse_range_write_argument(argument)
            if file_path and start_line is not None and end_line is not None:
                before_text = _slice_line_range(_read_local_text(file_path), start_line, end_line)
                preview.update(
                    {
                        "path": file_path,
                        "label": f"{file_path} lines {start_line}-{end_line}",
                        "start_line": start_line,
                        "end_line": end_line,
                        "before": _preview_text_block(before_text),
                        "after": _preview_text_block(content),
                    }
                )
            return preview

        if command in {"write_file", "append_file"}:
            file_path, content = _parse_file_write_argument(argument)
            if file_path:
                existing_text = _read_local_text(file_path)
                existing_lines = _line_count(existing_text)
                if command == "write_file":
                    preview.update(
                        {
                            "path": file_path,
                            "label": f"{file_path} full file overwrite",
                            "start_line": 1,
                            "end_line": max(existing_lines, 1),
                            "before": _preview_text_block(existing_text),
                            "after": _preview_text_block(content),
                        }
                    )
                else:
                    preview.update(
                        {
                            "path": file_path,
                            "label": f"{file_path} append after line {existing_lines}",
                            "start_line": existing_lines + 1,
                            "end_line": existing_lines,
                            "before": "",
                            "after": _preview_text_block(content),
                        }
                    )
            return preview

        if command == "create_file":
            file_path = str(argument or "").strip()
            if file_path:
                preview.update(
                    {
                        "path": file_path,
                        "label": f"{file_path} create empty file",
                        "start_line": 1,
                        "end_line": 0,
                        "before": "",
                        "after": "",
                    }
                )
            return preview

        if command == "replace_selection":
            preview.update(
                {
                    "path": active_file,
                    "label": (
                        f"{active_file} lines {int(selection.get('start_line', 0)) + 1}-{int(selection.get('end_line', selection.get('start_line', 0))) + 1}"
                        if active_file and selection.get("start_line") is not None
                        else active_file or "active selection"
                    ),
                    "start_line": int(selection.get("start_line", 0)) + 1 if selection.get("start_line") is not None else None,
                    "end_line": int(selection.get("end_line", selection.get("start_line", 0))) + 1 if selection.get("start_line") is not None else None,
                    "before": _preview_text_block(str(selection.get("text", "") or "")),
                    "after": _preview_text_block(argument),
                }
            )
            return preview

        if command == "insert_text":
            line = selection.get("start_line")
            preview.update(
                {
                    "path": active_file,
                    "label": f"{active_file} at line {int(line) + 1}" if active_file and line is not None else active_file or "active cursor",
                    "start_line": int(line) + 1 if line is not None else None,
                    "end_line": int(line) + 1 if line is not None else None,
                    "before": "",
                    "after": _preview_text_block(argument),
                }
            )
            return preview

        return preview

    def _snapshot_unlocked(self) -> dict:
        return {
            "context": {
                **self._context,
                "file_excerpt": dict(self._context["file_excerpt"]),
                "selection": dict(self._context["selection"]),
                "open_tabs": list(self._context["open_tabs"]),
                "workspace_folders": list(self._context["workspace_folders"]),
                "diagnostics": list(self._context["diagnostics"]),
            },
            "recent_results": list(self._results),
            "pending_actions": [
                action.copy() for action in self._actions if action["status"] == "pending"
            ],
            "pending_approvals": [
                action.copy() for action in self._actions if action["status"] == "pending_approval"
            ],
        }

    def update_context(self, payload: dict) -> dict:
        with self._lock:
            selection = payload.get("selection") or {}
            self._context.update(
                {
                    "connected": True,
                    "client": str(payload.get("client", "vscode")),
                    "workspace_name": str(payload.get("workspace_name", "")),
                    "workspace_folders": list(payload.get("workspace_folders") or []),
                    "active_file": str(payload.get("active_file", "")),
                    "language_id": str(payload.get("language_id", "")),
                    "open_tabs": list(payload.get("open_tabs") or []),
                    "file_excerpt": {
                        "text": str((payload.get("file_excerpt") or {}).get("text", "")),
                        "start_line": (payload.get("file_excerpt") or {}).get("start_line"),
                        "end_line": (payload.get("file_excerpt") or {}).get("end_line"),
                    },
                    "selection": {
                        "text": str(selection.get("text", "")),
                        "start_line": selection.get("start_line"),
                        "start_character": selection.get("start_character"),
                        "end_line": selection.get("end_line"),
                        "end_character": selection.get("end_character"),
                    },
                    "diagnostics": list(payload.get("diagnostics") or []),
                    "updated_at": _now_iso(),
                }
            )
            self._mark_prompt_dirty()
            return self._snapshot_unlocked()

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_unlocked()

    def queue_action(self, command: str, argument: str = "", meta: dict | None = None) -> dict:
        with self._lock:
            status = "pending_approval" if command in _APPROVAL_REQUIRED_ACTIONS else "pending"
            action_meta = dict(meta or {})
            if status == "pending_approval":
                preview = self._build_preview_meta(command, argument)
                if preview:
                    action_meta["preview"] = preview
            action = {
                "id": self._next_action_id,
                "command": command,
                "argument": argument,
                "meta": action_meta,
                "status": status,
                "created_at": _now_iso(),
            }
            self._next_action_id += 1
            self._actions.append(action)
            self._mark_prompt_dirty()
            self._condition.notify_all()
            return action.copy()

    def next_action_id(self) -> int:
        with self._lock:
            return self._next_action_id

    def actions_after(self, after_id: int) -> list[dict]:
        with self._lock:
            return [
                action.copy()
                for action in self._actions
                if action["status"] == "pending" and action["id"] > after_id
            ]

    def pending_approval_actions(self) -> list[dict]:
        with self._lock:
            return [action.copy() for action in self._actions if action["status"] == "pending_approval"]

    def approve_all_pending_actions(self) -> list[dict]:
        with self._lock:
            approved: list[dict] = []
            for action in self._actions:
                if action["status"] != "pending_approval":
                    continue
                action["status"] = "pending"
                action["approved_at"] = _now_iso()
                approved.append(action.copy())
            if approved:
                self._mark_prompt_dirty()
                self._condition.notify_all()
            return approved

    def reject_all_pending_actions(self, reason: str = "User rejected the proposed code change.") -> list[dict]:
        with self._lock:
            rejected: list[dict] = []
            for action in self._actions:
                if action["status"] != "pending_approval":
                    continue
                action["status"] = "error"
                action["completed_at"] = _now_iso()
                action["error"] = reason
                summary = {
                    "id": action["id"],
                    "command": action["command"],
                    "ok": False,
                    "result": "",
                    "error": reason,
                    "completed_at": action["completed_at"],
                }
                self._results.appendleft(summary)
                rejected.append(action.copy())
            if rejected:
                self._mark_prompt_dirty()
                self._condition.notify_all()
            return rejected

    def complete_action(self, action_id: int, ok: bool, result: str = "", error: str = "") -> dict | None:
        with self._lock:
            for action in self._actions:
                if action["id"] != action_id:
                    continue
                action["status"] = "done" if ok else "error"
                action["completed_at"] = _now_iso()
                if result:
                    action["result"] = result
                if error:
                    action["error"] = error
                summary = {
                    "id": action["id"],
                    "command": action["command"],
                    "ok": ok,
                    "result": result,
                    "error": error,
                    "completed_at": action["completed_at"],
                }
                self._results.appendleft(summary)
                self._mark_prompt_dirty()
                self._condition.notify_all()
                return summary
            return None

    def wait_for_action_results_after(
        self, after_id: int, expected_count: int, timeout: float = 4.0
    ) -> list[dict]:
        if expected_count <= 0:
            return []
        with self._condition:
            deadline = datetime.now(timezone.utc).timestamp() + timeout
            while True:
                completed = [
                    {
                        "id": action["id"],
                        "command": action["command"],
                        "ok": action["status"] == "done",
                        "result": action.get("result", ""),
                        "error": action.get("error", ""),
                        "completed_at": action.get("completed_at"),
                    }
                    for action in self._actions
                    if action["id"] > after_id and action["status"] in {"done", "error"}
                ]
                if len(completed) >= expected_count:
                    return completed
                remaining = deadline - datetime.now(timezone.utc).timestamp()
                if remaining <= 0:
                    return completed
                self._condition.wait(timeout=remaining)

    def format_for_prompt(self) -> str:
        with self._lock:
            if not self._context["connected"]:
                return ""
            if not self._prompt_cache_dirty:
                return self._prompt_cache

            lines = [
                "System Context:",
                "- VS Code is open and connected through the local editor bridge.",
            ]

            if self._context["workspace_name"]:
                lines.append(f"- Workspace: {self._context['workspace_name']}")

            if self._context["active_file"]:
                file_line = f"- Active file: {self._context['active_file']}"
                if self._context["language_id"]:
                    file_line += f" ({self._context['language_id']})"
                lines.append(file_line)

            if self._context["open_tabs"]:
                lines.append(f"- Open tabs: {', '.join(self._context['open_tabs'][:6])}")

            selection = self._context["selection"]
            if selection.get("text"):
                lines.append(f"- Current selection: {_trim_text(selection['text'], 350)}")
            elif selection.get("start_line") is not None:
                line = int(selection["start_line"]) + 1
                lines.append(f"- Cursor line: {line}")

            excerpt = self._context["file_excerpt"]
            if excerpt.get("text"):
                start_line = excerpt.get("start_line")
                end_line = excerpt.get("end_line")
                if start_line is not None and end_line is not None:
                    lines.append(f"- Active file excerpt (lines {int(start_line) + 1}-{int(end_line) + 1}):")
                else:
                    lines.append("- Active file excerpt:")
                lines.append(_trim_text(str(excerpt["text"]), 1800))

            diagnostics = self._context["diagnostics"][:5]
            if diagnostics:
                lines.append("- Diagnostics:")
                for item in diagnostics:
                    path = item.get("path") or "unknown file"
                    line = item.get("line")
                    message = _trim_text(str(item.get("message", "")), 180)
                    severity = str(item.get("severity", "info")).lower()
                    if line is None:
                        lines.append(f"  • {severity} in {path}: {message}")
                    else:
                        lines.append(f"  • {severity} in {path}:{line}: {message}")

            results = list(self._results)[:3]
            if results:
                lines.append("- Recent editor actions:")
                for item in results:
                    limit = 1200 if item["command"] in {"read_file", "read_current_file", "list_files"} else 160
                    outcome = item["result"] if item["ok"] else item["error"] or "failed"
                    lines.append(
                        f"  • {item['command']}: {_trim_text(str(outcome), limit)}"
                    )

            approvals = [action for action in self._actions if action["status"] == "pending_approval"]
            if approvals:
                lines.append("- Pending approval for editor changes:")
                for action in approvals[:5]:
                    preview = ((action.get("meta") or {}).get("preview") or {})
                    label = str(preview.get("label") or "").strip()
                    lines.append(
                        f"  • {action['command']}: {_trim_text(label or str(action.get('argument', '')), 180)}"
                    )

            self._prompt_cache = "\n".join(lines)
            self._prompt_cache_dirty = False
            return self._prompt_cache


_BRIDGE = EditorBridge()


def get_editor_bridge() -> EditorBridge:
    return _BRIDGE