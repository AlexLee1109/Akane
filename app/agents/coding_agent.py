"""Deterministic coding-model workflow for fresh reads and VS Code edits."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from app.core.config import (
    CODER_INITIAL_CHUNKS_PER_FILE,
    CODER_MAX_INITIAL_TARGETS,
    CODER_MAX_READ_CHUNKS_PER_FILE,
    CODER_MAX_TOKENS,
    CODER_MAX_TURNS,
    CODER_READ_CHUNK_LINES,
    CODER_TEMPERATURE,
    CODER_TIMEOUT_SECONDS,
    REPETITION_PENALTY,
    TOP_K,
    TOP_P,
)
from app.integrations.editor_bridge import get_editor_bridge
from app.core.model_loader import ModelManager

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

_EDITOR_LAUNCH_COMMANDS = {"open_vscode", "open_project", "open_workspace"}
_READ_ACTION_COMMANDS = {
    "read_file", "read_current_file",
    "read_file_chunk", "read_current_file_chunk",
    "list_files",
}
_EDIT_ACTION_COMMANDS = {
    "create_file", "write_file", "append_file", "replace_file_range",
    "replace_selection", "insert_text", "save_file", "format_document", "open_file",
}
_ALLOWED_ACTIONS = _READ_ACTION_COMMANDS | _EDIT_ACTION_COMMANDS | _EDITOR_LAUNCH_COMMANDS
_ACTIVE_EDITOR_COMMANDS = {"replace_selection", "insert_text", "format_document", "save_file"}
_ARG_REQUIRED_ACTIONS = {
    "create_file", "write_file", "append_file", "replace_file_range",
    "replace_selection", "insert_text", "open_file",
    "read_file", "read_file_chunk", "list_files", "save_file",
}
_ACTION_WAIT_SECONDS = 5.5
_READ_MAX_LINES_PER_FILE = 420
_READ_MAX_CHARS_PER_FILE = 16_000


def collapse_hidden_tag_gaps(text: str) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\t", " ")
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


def clean_model_text(text: str) -> str:
    return collapse_hidden_tag_gaps(str(text or "")).replace("`", "").strip()


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
        filepath = _sanitize_path_argument(str(raw or ""), preserve_suffix=False)
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


# ------------------------------------------------------------------ #
# Public types                                                         #
# ------------------------------------------------------------------ #

@dataclass
class CoderOutcome:
    summary: str
    action_results: list[dict]
    actions_applied: bool
    tool_used: bool
    approval_required: bool = False
    proposed_actions: list[dict] | None = None
    timed_out: bool = False
    error: str = ""


# ------------------------------------------------------------------ #
# LLM helpers                                                          #
# ------------------------------------------------------------------ #

def _extract_message_text(result: dict) -> str:
    choices = result.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    return _content_to_text(content).strip()


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return str(content)


def _remaining_seconds(deadline: float | None) -> float:
    if deadline is None:
        return CODER_TIMEOUT_SECONDS
    return max(0.0, deadline - time.monotonic())


def _budget_exhausted(deadline: float | None, *, reserve: float = 0.0) -> bool:
    return _remaining_seconds(deadline) <= reserve


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    lowered = str(value or "").strip().lower()
    if not lowered:
        return bool(default)
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _request_coder_completion(messages: list[dict], *, deadline: float | None = None, json_mode: bool = True) -> dict:
    timeout = max(4.0, min(30.0, _remaining_seconds(deadline) - 1.0))
    if timeout <= 4.0 and _budget_exhausted(deadline, reserve=4.5):
        raise TimeoutError("Coder model budget exhausted before request.")
    return ModelManager.get_instance().create_chat_completion(
        messages=messages,
        max_tokens=CODER_MAX_TOKENS,
        temperature=CODER_TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repeat_penalty=REPETITION_PENALTY,
        stream=False,
        role="coder",
        response_format={"type": "json_object"} if json_mode else None,
        timeout=timeout,
    )


def _request_coder_completion_with_json_retry(messages: list[dict], *, deadline: float | None = None) -> dict:
    try:
        return _request_coder_completion(messages, deadline=deadline, json_mode=True)
    except Exception as exc:
        message = str(exc).lower()
        if "response_format" not in message and "json" not in message:
            raise
        return _request_coder_completion(messages, deadline=deadline, json_mode=False)


def _extract_json_payload(raw: str) -> dict:
    text = str(raw or "").strip()
    if not text or "{" not in text:
        return {}
    # Strip markdown fences
    if text.startswith("```"):
        nl = text.find("\n")
        text = text[nl + 1:] if nl != -1 else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()
    # Direct parse
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    # Extract first {...} block
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _request_payload_with_repair(
    messages: list[dict], *, progress_callback=None, deadline: float | None = None
) -> tuple[str, dict]:
    """Call the coder LLM and attempt one JSON-repair pass on failure."""
    if progress_callback:
        progress_callback("Using coding model...")
    try:
        raw = _extract_message_text(_request_coder_completion_with_json_retry(messages, deadline=deadline))
    except Exception as exc:
        return "", {
            "summary": "",
            "done": True,
            "reason": f"Coder model request failed: {exc}",
            "_error": str(exc),
        }
    payload = _extract_json_payload(raw)
    if payload:
        return raw, payload
    if _budget_exhausted(deadline, reserve=8.0):
        return raw, {
            "summary": "",
            "done": True,
            "reason": "Coder model did not return valid JSON before the time budget expired.",
            "_error": "invalid_json",
        }

    repair_instruction = (
        "Your reply was empty. Reply with a complete JSON object using the required schema."
        if not raw.strip()
        else "Your reply was not valid JSON. Reply with JSON only — no markdown, no prose outside the JSON."
    )
    if progress_callback:
        progress_callback("Using coding model...")
    try:
        repair_raw = _extract_message_text(
            _request_coder_completion_with_json_retry([
                *messages,
                {"role": "assistant", "content": raw or "{}"},
                {"role": "system",    "content": repair_instruction},
            ], deadline=deadline)
        )
    except Exception as exc:
        return raw, {
            "summary": "",
            "done": True,
            "reason": f"Coder JSON repair failed: {exc}",
            "_error": str(exc),
        }
    return repair_raw, _extract_json_payload(repair_raw)


# ------------------------------------------------------------------ #
# Payload sanitisation                                                 #
# ------------------------------------------------------------------ #

def _split_path_suffix(path: str) -> tuple[str, str]:
    raw = str(path or "").strip().replace("\\", "/")
    if raw.startswith("./"):
        raw = raw[2:]
    if ":" not in raw:
        return raw, ""
    path_part, _, suffix = raw.partition(":")
    suffix = suffix.strip()
    if path_part and suffix and (suffix[0].isdigit() or suffix.lower().startswith("l")):
        return path_part.strip(), f":{suffix}"
    return raw, ""


def _sanitize_relative_path(path: str) -> str:
    raw, _ = _split_path_suffix(path)
    raw = raw.strip().strip("`'\".,;()[]{}").replace("\\", "/")
    if raw.startswith("./"):
        raw = raw[2:]
    if not raw or raw.startswith("/") or raw.startswith("~") or "://" in raw or ":" in raw:
        return ""
    parts = PurePosixPath(raw).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return ""
    return "/".join(parts)


def _sanitize_path_argument(argument: str, *, preserve_suffix: bool = True) -> str:
    path, suffix = _split_path_suffix(argument)
    sanitized = _sanitize_relative_path(path)
    if not sanitized:
        return ""
    return sanitized + (suffix if preserve_suffix else "")


def _sanitize_path_first_line_argument(argument: str) -> str:
    first_line, sep, rest = str(argument or "").partition("\n")
    sanitized = _sanitize_path_argument(first_line)
    if not sanitized:
        return ""
    return sanitized + (sep + rest if sep else "")


def _sanitize_range_argument(argument: str) -> str:
    header, sep, rest = str(argument or "").partition("\n")
    path_part, colon, range_part = header.replace("\\", "/").partition(":")
    if not colon or not range_part.strip():
        return ""
    sanitized = _sanitize_path_argument(path_part, preserve_suffix=False)
    if not sanitized:
        return ""
    return f"{sanitized}:{range_part.strip()}" + (sep + rest if sep else "")


def _sanitize_read_requests(payload: dict) -> list[str]:
    raw = payload.get("read_requests") or []
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        path = _sanitize_path_argument(str(item or ""), preserve_suffix=False)
        if not path or path in seen:
            continue
        seen.add(path)
        result.append(path)
        if len(result) >= CODER_MAX_INITIAL_TARGETS:
            break
    return result


def _sanitize_actions(payload: dict) -> list[dict]:
    raw = payload.get("actions") or []
    if not isinstance(raw, list):
        return []
    actions: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        command = str(item.get("command") or "").strip().lower()
        argument = str(item.get("argument") or item.get("value") or "").strip()
        if not command or command not in _ALLOWED_ACTIONS:
            continue
        if command in _ARG_REQUIRED_ACTIONS and not argument:
            continue
        if command in {"open_file", "read_file", "read_file_chunk"}:
            argument = _sanitize_path_argument(argument)
            if not argument:
                continue
        elif command in {"list_files", "create_file", "save_file"}:
            argument = _sanitize_path_argument(argument, preserve_suffix=False)
            if not argument:
                continue
        elif command in {"write_file", "append_file"}:
            argument = _sanitize_path_first_line_argument(argument)
            if not argument:
                continue
        elif command == "replace_file_range":
            argument = _sanitize_range_argument(argument)
            if not argument:
                continue
        actions.append({"command": command, "argument": argument})
        if len(actions) >= 8:
            break
    return actions


# ------------------------------------------------------------------ #
# Rendering helpers                                                    #
# ------------------------------------------------------------------ #

def _render_read_results(results: list[str]) -> str:
    if not results:
        return ""
    return "Fresh file reads:\n\n" + "\n\n".join(r.strip() for r in results if r.strip())


def _render_action_results(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["Editor action results:"]
    for item in results:
        outcome = item.get("result") if item.get("ok") else (item.get("error") or "failed")
        lines.append(f"- {item['command']}: {outcome}")
    return "\n".join(lines)


def _has_successful_edit(results: list[dict]) -> bool:
    return any(item.get("ok") and item.get("command") in _EDIT_ACTION_COMMANDS for item in results)


# ------------------------------------------------------------------ #
# Bridge read helpers                                                  #
# ------------------------------------------------------------------ #

def _parse_read_result_header(text: str) -> tuple[str, int | None, int | None, int | None]:
    first = (str(text or "").splitlines() or [""])[0]
    if not first.lower().startswith("file "):
        return "", None, None, None
    rest = first[5:]
    lower = rest.lower()
    marker = " (line-numbered"
    path = rest[: lower.index(marker)].strip() if marker in lower else rest.strip()
    start_line = end_line = total_lines = None
    if "lines " in lower and " of " in lower:
        after = lower.split("lines ", 1)[1]
        range_part, _, total_part = after.partition(" of ")
        def _int(s: str) -> int | None:
            digits = "".join(c for c in s if c.isdigit())
            return int(digits) if digits else None
        if "-" in range_part:
            sl, el = range_part.split("-", 1)
            start_line, end_line = _int(sl), _int(el)
        total_lines = _int(total_part)
    return path, start_line, end_line, total_lines


def _run_single_read(command: str, argument: str, *, logical_path: str, deadline: float | None = None) -> str:
    timeout = min(_ACTION_WAIT_SECONDS, max(0.0, _remaining_seconds(deadline) - 1.0))
    if timeout <= 0:
        return ""
    bridge = get_editor_bridge()
    start_id = bridge.next_action_id()
    bridge.queue_action(command, argument)
    for item in bridge.wait_for_action_results_after(
        after_id=start_id - 1, expected_count=1, timeout=timeout
    ):
        if item.get("command") != command or not item.get("ok"):
            continue
        text = str(item.get("result") or "").strip()
        if not text:
            continue
        path, _, _, _ = _parse_read_result_header(text)
        if path and logical_path and path != logical_path:
            continue
        return text
    return ""


def _read_file_via_bridge(
    path: str, *, use_current: bool, max_chunks: int, progress_callback=None, deadline: float | None = None
) -> list[str]:
    results: list[str] = []
    start_line = 1
    for _ in range(max_chunks):
        if _budget_exhausted(deadline, reserve=2.0):
            break
        end_line = start_line + CODER_READ_CHUNK_LINES - 1
        if use_current:
            cmd, arg = "read_current_file_chunk", f"{start_line}:{end_line}"
        else:
            cmd, arg = "read_file_chunk", f"{path}:{start_line}:{end_line}"
        if progress_callback:
            progress_callback("Reading files...")
        text = _run_single_read(cmd, arg, logical_path=path, deadline=deadline)
        if not text:
            break
        results.append(text)
        _, _, chunk_end, total = _parse_read_result_header(text)
        if chunk_end is None or total is None or chunk_end >= total:
            break
        next_start = chunk_end + 1
        if next_start <= start_line:
            break
        start_line = next_start
    return results


def _fresh_reads(
    targets: list[str],
    *,
    progress_callback=None,
    max_chunks: int = CODER_MAX_READ_CHUNKS_PER_FILE,
    deadline: float | None = None,
) -> tuple[list[str], list[str]]:
    """Read files, preferring the editor bridge when connected.

    Returns (read_result_strings, consulted_paths).
    """
    bridge = get_editor_bridge()
    snapshot = bridge.snapshot().get("context", {})
    connected = bool(snapshot.get("connected"))
    active_file = _sanitize_relative_path(str(snapshot.get("active_file") or ""))

    # Deduplicate while preserving order
    seen: set[str] = set()
    normalized: list[str] = []
    for p in targets:
        p = _sanitize_path_argument(str(p or ""), preserve_suffix=False)
        if p and p not in seen:
            seen.add(p)
            normalized.append(p)
    normalized = normalized[:CODER_MAX_INITIAL_TARGETS]

    if not connected:
        if _budget_exhausted(deadline, reserve=1.0):
            return [], normalized or []
        results = execute_read_requests(normalized)
        return results, normalized or []

    results: list[str] = []
    consulted: list[str] = []
    done: set[str] = set()

    # Read active file first when it is relevant
    read_active = active_file and (not normalized or active_file in normalized)
    if read_active:
        chunks = _read_file_via_bridge(
            active_file, use_current=True,
            max_chunks=max_chunks, progress_callback=progress_callback, deadline=deadline,
        )
        if chunks:
            results.extend(chunks)
            consulted.append(active_file)
            done.add(active_file)

    # Read remaining explicit targets via bridge
    for path in normalized:
        if path in done:
            continue
        chunks = _read_file_via_bridge(
            path, use_current=False,
            max_chunks=max_chunks, progress_callback=progress_callback, deadline=deadline,
        )
        if chunks:
            results.extend(chunks)
            consulted.append(path)
            done.add(path)

    # Filesystem fallback for anything the bridge did not serve
    missing = [p for p in normalized if p not in done]
    if missing:
        if _budget_exhausted(deadline, reserve=1.0):
            return results, consulted or normalized
        results.extend(execute_read_requests(missing))
        consulted.extend(missing)

    return results, consulted or normalized


# ------------------------------------------------------------------ #
# Editor action helpers                                                #
# ------------------------------------------------------------------ #

def _action_target_path(action: dict, active_file: str) -> str:
    command = str(action.get("command") or "").lower()
    argument = str(action.get("argument") or "")
    if command in {"read_file", "open_file"}:
        return _sanitize_path_argument(argument, preserve_suffix=False)
    if command in {"create_file", "save_file"}:
        return _sanitize_path_argument(argument, preserve_suffix=False) or _sanitize_relative_path(active_file)
    if command in {"write_file", "append_file"}:
        return _sanitize_path_argument(argument.split("\n", 1)[0], preserve_suffix=False)
    if command == "replace_file_range":
        return _sanitize_path_argument(argument.split("\n", 1)[0].split(":", 1)[0], preserve_suffix=False)
    if command in {"replace_selection", "insert_text", "format_document"}:
        return _sanitize_relative_path(active_file)
    return ""


def _prepare_editor_actions(actions: list[dict], active_file: str) -> list[dict]:
    queued = list(actions)
    needs_active = any(
        str(a.get("command") or "").lower() in _ACTIVE_EDITOR_COMMANDS for a in queued
    )
    preferred = next(
        (p for a in queued if (p := _action_target_path(a, ""))),
        active_file,
    )
    if (
        needs_active and preferred and preferred != active_file
        and not any(
            str(a.get("command") or "").lower() == "open_file"
            and _sanitize_path_argument(str(a.get("argument") or ""), preserve_suffix=False) == preferred
            for a in queued
        )
    ):
        queued = [{"command": "open_file", "argument": preferred}, *queued]
    return queued


def _queue_editor_actions(actions: list[dict], *, progress_callback=None, deadline: float | None = None) -> list[dict]:
    if not actions:
        return []
    if _budget_exhausted(deadline, reserve=2.0):
        return [
            {
                "command": str(action.get("command") or ""),
                "argument": str(action.get("argument") or ""),
                "ok": False,
                "result": "",
                "error": "Coder time budget expired before editor action could run.",
            }
            for action in actions
        ]
    bridge = get_editor_bridge()
    snapshot = bridge.snapshot().get("context", {})
    active_file = _sanitize_relative_path(str(snapshot.get("active_file") or ""))
    queued = _prepare_editor_actions(actions, active_file)

    # Auto-append save after in-place edits
    if any(a["command"] in {"replace_file_range", "replace_selection", "insert_text", "format_document"} for a in queued):
        if not any(a["command"] == "save_file" for a in queued):
            save_target = next(
                (p for a in queued if (p := _action_target_path(a, ""))),
                active_file,
            )
            queued.append({"command": "save_file", "argument": save_target})

    start_id = bridge.next_action_id()
    queued_actions = [bridge.queue_action(a["command"], a["argument"]) for a in queued]

    if any(a.get("status") == "pending_approval" for a in queued_actions):
        # Open the target file so the user can see the proposed diff
        target = next(
            (p for a in queued if (p := _action_target_path(a, ""))
             and str(a.get("command") or "").lower() != "open_file"),
            "",
        )
        if target and not any(str(a.get("command") or "") == "open_file" for a in queued_actions):
            oid = bridge.next_action_id()
            bridge.queue_action("open_file", target)
            bridge.wait_for_action_results_after(
                after_id=oid - 1,
                expected_count=1,
                timeout=min(_ACTION_WAIT_SECONDS, max(0.1, _remaining_seconds(deadline) - 1.0)),
            )
        return [
            {**a, "ok": False, "result": "", "error": "Awaiting user approval."}
            for a in queued_actions
        ]

    if progress_callback:
        progress_callback("Working in VS Code...")
    return bridge.wait_for_action_results_after(
        after_id=start_id - 1,
        expected_count=len(queued),
        timeout=min(_ACTION_WAIT_SECONDS, max(0.1, _remaining_seconds(deadline) - 1.0)),
    )


# ------------------------------------------------------------------ #
# System prompt                                                        #
# ------------------------------------------------------------------ #

def _build_system_prompt(*, apply_now: bool) -> str:
    mode = (
        "Apply the change now. Request reads if context is missing, then emit editor actions. "
        "If blocked, set done=true and explain the exact blocker in reason."
        if apply_now else
        "Answer accurately based on the freshest code reads you have. "
        "For new standalone scripts, emit create_file + write_file directly without reading first."
    )
    return (
        "You are a hidden coding assistant. Return JSON only — no prose outside the JSON object.\n\n"
        "Schema:\n"
        "{\n"
        '  "summary": "...",\n'
        '  "done": true,\n'
        '  "read_requests": ["path/to/file.py"],\n'
        '  "actions": [{"command": "replace_file_range", "argument": "path:start:end\\ncode"}],\n'
        '  "reason": "optional blocker"\n'
        "}\n\n"
        "Rules:\n"
        "- Prefer a final grounded summary over asking for more reads when the provided context is enough.\n"
        "- If you cannot find the requested symbol/file, set done=true and explain exactly what you checked in reason.\n"
        "- Use read_requests before answering about code you have not seen.\n"
        "- Keep read_requests focused: request at most 2 likely files at a time.\n"
        "- Set done=true only when summary contains a complete, grounded answer.\n"
        "- Use replace_file_range only when you have exact line numbers from fresh reads.\n"
        "- Use write_file only when replacing an entire small file with complete content.\n"
        "- For debugging, name the likely bug and the smallest fix; do not punt to another model.\n"
        "- Never claim a change succeeded unless editor actions ran and were confirmed.\n"
        "- summary must be a short, natural sentence in Akane's voice — direct, calm, not robotic.\n"
        "- Never start with a heading, label, or self-description.\n"
        "- Write like you noticed something, not like you're filing a report.\n"
        f"- {mode}"
    )


# ------------------------------------------------------------------ #
# Main entry point                                                     #
# ------------------------------------------------------------------ #

def run_coder_specialist(
    task: str,
    *,
    preload_paths: list[str] | None = None,
    apply_now: bool = False,
    progress_callback=None,
) -> CoderOutcome:
    deadline = time.monotonic() + max(8.0, CODER_TIMEOUT_SECONDS)
    bridge = get_editor_bridge()
    initial_targets = [
        str(p or "").strip() for p in (preload_paths or []) if str(p or "").strip()
    ][:CODER_MAX_INITIAL_TARGETS]

    # ── Initial reads ─────────────────────────────────────────────
    fresh_reads, consulted = _fresh_reads(
        initial_targets,
        progress_callback=progress_callback,
        max_chunks=CODER_INITIAL_CHUNKS_PER_FILE,
        deadline=deadline,
    )
    print(
        f"[Akane][coder] start apply_now={apply_now} targets={initial_targets} reads={len(fresh_reads)}",
        flush=True,
    )

    # ── Build initial message list ─────────────────────────────────
    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(apply_now=apply_now)},
    ]
    bridge_prompt = bridge.format_for_prompt()
    if bridge_prompt:
        messages.append({"role": "system", "content": bridge_prompt})
    if consulted:
        messages.append({
            "role": "system",
            "content": f"Likely relevant files: {', '.join(consulted[:CODER_MAX_INITIAL_TARGETS])}.",
        })
    if fresh_reads:
        messages.append({
            "role": "system",
            "content": (
                f"{_render_read_results(fresh_reads)}\n\n"
                "Use those reads as the source of truth before requesting more."
            ),
        })
    messages.append({"role": "user", "content": task})

    # ── Agentic loop ──────────────────────────────────────────────
    all_action_results: list[dict] = []
    tool_used = bool(fresh_reads)
    used_read_sigs: set[tuple[str, ...]] = set()
    used_action_sigs: set[tuple[tuple[str, str], ...]] = set()
    last_summary = ""
    last_reason = ""
    last_error = ""

    for turn in range(CODER_MAX_TURNS):
        if _budget_exhausted(deadline, reserve=4.0):
            last_reason = last_reason or "The coding model reached its time budget before finishing."
            break
        raw, payload = _request_payload_with_repair(
            messages,
            progress_callback=progress_callback,
            deadline=deadline,
        )
        summary      = clean_model_text(str(payload.get("summary") or "")).strip()
        reason       = clean_model_text(str(payload.get("reason")  or "")).strip()
        done         = _coerce_bool(payload.get("done"))
        read_requests = _sanitize_read_requests(payload)
        actions       = _sanitize_actions(payload)
        last_error = str(payload.get("_error") or last_error or "").strip()

        print(
            f"[Akane][coder] turn={turn} summary={len(summary)} "
            f"reads={len(read_requests)} actions={len(actions)} done={done}",
            flush=True,
        )

        if summary:
            last_summary = summary
        if reason:
            last_reason = reason

        # ── Read phase ────────────────────────────────────────────
        if read_requests:
            sig = tuple(read_requests)
            if sig in used_read_sigs:
                messages.append({"role": "system", "content":
                    "You already requested those reads. Use the results already provided "
                    "or request different files."})
                continue
            used_read_sigs.add(sig)
            results, _ = _fresh_reads(
                read_requests,
                progress_callback=progress_callback,
                max_chunks=CODER_MAX_READ_CHUNKS_PER_FILE,
                deadline=deadline,
            )
            tool_used = tool_used or bool(results)
            if not results:
                messages.extend([
                    {"role": "assistant", "content": raw or "{}"},
                    {"role": "system", "content": (
                        "Those read requests returned no content or timed out. "
                        "Do not repeat the same reads. Use the existing context to give the best grounded answer, "
                        "or set done=true with a precise reason describing what could not be found."
                    )},
                ])
                continue
            messages.extend([
                {"role": "assistant", "content": raw or "{}"},
                {"role": "system",    "content": (
                    f"{_render_read_results(results)}\n\n"
                    "Those are now the authoritative reads. Analyse them and continue."
                )},
            ])
            continue

        # ── Action phase ──────────────────────────────────────────
        if actions:
            sig = tuple((a["command"], a["argument"]) for a in actions)
            if sig in used_action_sigs:
                messages.append({"role": "system", "content":
                    "You already emitted those actions. "
                    "Summarise what changed or explain the blocker."})
                continue
            used_action_sigs.add(sig)

            results = _queue_editor_actions(actions, progress_callback=progress_callback, deadline=deadline)
            all_action_results.extend(results)
            tool_used = True

            if any(r.get("status") == "pending_approval" for r in results):
                return CoderOutcome(
                    summary="",
                    action_results=results,
                    actions_applied=False,
                    tool_used=True,
                    approval_required=True,
                    proposed_actions=results,
                )

            # Verify the touched files
            snap = bridge.snapshot().get("context", {})
            active_file = _sanitize_relative_path(str(snap.get("active_file") or ""))
            touched = list(dict.fromkeys(
                p for a in actions if (p := _action_target_path(a, active_file))
            ))
            verify_reads, _ = _fresh_reads(
                touched[:CODER_MAX_INITIAL_TARGETS],
                progress_callback=progress_callback,
                max_chunks=CODER_INITIAL_CHUNKS_PER_FILE,
                deadline=deadline,
            )
            messages.extend([
                {"role": "assistant", "content": raw or "{}"},
                {"role": "system",    "content": (
                    f"{_render_action_results(results)}\n\n"
                    f"{_render_read_results(verify_reads)}\n\n"
                    "Those are the latest action results and post-edit reads. "
                    "Summarise what changed with done=true."
                )},
            ])
            continue

        # ── Terminal conditions ───────────────────────────────────
        if done and (summary or reason):
            break
        if summary and not apply_now:
            break
        if summary and apply_now and _has_successful_edit(all_action_results):
            break
        if reason and apply_now:
            break

        # Empty payload — nudge the model
        if not payload:
            messages.append({"role": "system", "content":
                "Reply with a complete JSON object using the required schema."})
            continue

        # No useful output and no tool calls — prompt for progress
        if not summary and not read_requests and not actions:
            messages.append({"role": "system", "content":
                "Request file reads, emit editor actions, or return a grounded summary with done=true."})

    # ── Build final outcome ───────────────────────────────────────
    actions_applied = _has_successful_edit(all_action_results)

    if apply_now and not actions_applied:
        final_summary = last_reason or last_summary or ""
    elif last_summary:
        final_summary = last_summary
    else:
        final_summary = last_reason or ""

    if apply_now and actions_applied and not final_summary:
        touched = list(dict.fromkeys(
            p
            for item in all_action_results
            if item.get("ok")
            for p in [_sanitize_path_argument(str(item.get("argument", "")).split("\n")[0].split(":")[0], preserve_suffix=False)]
            if p
        ))
        final_summary = (
            f"Applied changes to {', '.join(touched[:3])}."
            if touched else
            "Applied the requested code changes."
        )

    timed_out = _budget_exhausted(deadline)
    if not final_summary:
        if consulted:
            final_summary = "I checked the likely code context, but the coding model did not find a reliable result before its budget ended."
        elif last_error:
            final_summary = f"The coding model could not complete reliably: {last_error}"
        else:
            final_summary = "The coding model could not produce a reliable result for this request."

    print(
        f"[Akane][coder] done summary={len(final_summary)} "
        f"applied={actions_applied} tool_used={tool_used}",
        flush=True,
    )
    return CoderOutcome(
        summary=final_summary.strip(),
        action_results=all_action_results,
        actions_applied=actions_applied,
        tool_used=tool_used,
        timed_out=timed_out,
        error=last_error,
    )
