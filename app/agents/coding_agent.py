"""Deterministic coding-model workflow for fresh reads and VS Code edits."""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.core.config import (
    CODER_INITIAL_CHUNKS_PER_FILE,
    CODER_MAX_INITIAL_TARGETS,
    CODER_MAX_READ_CHUNKS_PER_FILE,
    CODER_MAX_TOKENS,
    CODER_MAX_TURNS,
    CODER_READ_CHUNK_LINES,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from app.integrations.editor_bridge import get_editor_bridge
from app.core.generation import clean_model_text, collapse_hidden_tag_gaps, execute_read_requests
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


# ------------------------------------------------------------------ #
# LLM helpers                                                          #
# ------------------------------------------------------------------ #

def _extract_message_text(result: dict) -> str:
    choices = result.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts).strip()
    return str(content).strip()


def _request_coder_completion(messages: list[dict]) -> dict:
    return ModelManager.get_instance().create_chat_completion(
        messages=messages,
        max_tokens=CODER_MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repeat_penalty=REPETITION_PENALTY,
        stream=False,
        role="coder",
    )


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
    messages: list[dict], *, progress_callback=None
) -> tuple[str, dict]:
    """Call the coder LLM and attempt one JSON-repair pass on failure."""
    if progress_callback:
        progress_callback("Using coding model...")
    raw = _extract_message_text(_request_coder_completion(messages))
    payload = _extract_json_payload(raw)
    if payload:
        return raw, payload

    repair_instruction = (
        "Your reply was empty. Reply with a complete JSON object using the required schema."
        if not raw.strip()
        else "Your reply was not valid JSON. Reply with JSON only — no markdown, no prose outside the JSON."
    )
    if progress_callback:
        progress_callback("Using coding model...")
    repair_raw = _extract_message_text(
        _request_coder_completion([
            *messages,
            {"role": "assistant", "content": raw or "{}"},
            {"role": "system",    "content": repair_instruction},
        ])
    )
    return repair_raw, _extract_json_payload(repair_raw)


# ------------------------------------------------------------------ #
# Payload sanitisation                                                 #
# ------------------------------------------------------------------ #

def _sanitize_read_requests(payload: dict) -> list[str]:
    raw = payload.get("read_requests") or []
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        path = str(item or "").strip().replace("\\", "/")
        if ":" in path and "/" in path:
            path = path.split(":", 1)[0].strip()
        if path.startswith("./"):
            path = path[2:]
        if not path or path in seen or path.startswith("/") or path.startswith(".."):
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
            argument = argument.replace("\\", "/")
            if argument.startswith("./"):
                argument = argument[2:]
            if argument.startswith("/") or argument.startswith(".."):
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


def _run_single_read(command: str, argument: str, *, logical_path: str) -> str:
    bridge = get_editor_bridge()
    start_id = bridge.next_action_id()
    bridge.queue_action(command, argument)
    for item in bridge.wait_for_action_results_after(
        after_id=start_id - 1, expected_count=1, timeout=_ACTION_WAIT_SECONDS
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
    path: str, *, use_current: bool, max_chunks: int, progress_callback=None
) -> list[str]:
    results: list[str] = []
    start_line = 1
    for _ in range(max_chunks):
        end_line = start_line + CODER_READ_CHUNK_LINES - 1
        if use_current:
            cmd, arg = "read_current_file_chunk", f"{start_line}:{end_line}"
        else:
            cmd, arg = "read_file_chunk", f"{path}:{start_line}:{end_line}"
        if progress_callback:
            progress_callback("Reading files...")
        text = _run_single_read(cmd, arg, logical_path=path)
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
) -> tuple[list[str], list[str]]:
    """Read files, preferring the editor bridge when connected.

    Returns (read_result_strings, consulted_paths).
    """
    bridge = get_editor_bridge()
    snapshot = bridge.snapshot().get("context", {})
    connected = bool(snapshot.get("connected"))
    active_file = str(snapshot.get("active_file") or "").strip()

    # Deduplicate while preserving order
    seen: set[str] = set()
    normalized: list[str] = []
    for p in targets:
        p = str(p or "").strip()
        if p and p not in seen:
            seen.add(p)
            normalized.append(p)
    normalized = normalized[:CODER_MAX_INITIAL_TARGETS]

    if not connected:
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
            max_chunks=max_chunks, progress_callback=progress_callback,
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
            max_chunks=max_chunks, progress_callback=progress_callback,
        )
        if chunks:
            results.extend(chunks)
            consulted.append(path)
            done.add(path)

    # Filesystem fallback for anything the bridge did not serve
    missing = [p for p in normalized if p not in done]
    if missing:
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
        return argument.split(":", 1)[0].strip()
    if command in {"create_file", "save_file"}:
        return argument.strip() or active_file
    if command in {"write_file", "append_file"}:
        return argument.split("\n", 1)[0].strip()
    if command == "replace_file_range":
        return argument.split("\n", 1)[0].split(":", 1)[0].strip()
    if command in {"replace_selection", "insert_text", "format_document"}:
        return active_file
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
            and str(a.get("argument") or "").split(":", 1)[0].strip() == preferred
            for a in queued
        )
    ):
        queued = [{"command": "open_file", "argument": preferred}, *queued]
    return queued


def _queue_editor_actions(actions: list[dict], *, progress_callback=None) -> list[dict]:
    if not actions:
        return []
    bridge = get_editor_bridge()
    snapshot = bridge.snapshot().get("context", {})
    active_file = str(snapshot.get("active_file") or "").strip()
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
                after_id=oid - 1, expected_count=1, timeout=_ACTION_WAIT_SECONDS
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
        timeout=_ACTION_WAIT_SECONDS,
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
        "- Use read_requests before answering about code you have not seen.\n"
        "- Set done=true only when summary contains a complete, grounded answer.\n"
        "- Use replace_file_range only when you have exact line numbers from fresh reads.\n"
        "- Use write_file only when replacing an entire small file with complete content.\n"
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
    bridge = get_editor_bridge()
    initial_targets = [
        str(p or "").strip() for p in (preload_paths or []) if str(p or "").strip()
    ][:CODER_MAX_INITIAL_TARGETS]

    # ── Initial reads ─────────────────────────────────────────────
    fresh_reads, consulted = _fresh_reads(
        initial_targets,
        progress_callback=progress_callback,
        max_chunks=CODER_INITIAL_CHUNKS_PER_FILE,
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

    for turn in range(CODER_MAX_TURNS):
        raw, payload = _request_payload_with_repair(messages, progress_callback=progress_callback)
        summary      = clean_model_text(str(payload.get("summary") or "")).strip()
        reason       = clean_model_text(str(payload.get("reason")  or "")).strip()
        done         = bool(payload.get("done"))
        read_requests = _sanitize_read_requests(payload)
        actions       = _sanitize_actions(payload)

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
            )
            tool_used = tool_used or bool(results)
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

            results = _queue_editor_actions(actions, progress_callback=progress_callback)
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
            active_file = str(snap.get("active_file") or "").strip()
            touched = list(dict.fromkeys(
                p for a in actions if (p := _action_target_path(a, active_file))
            ))
            verify_reads, _ = _fresh_reads(
                touched[:CODER_MAX_INITIAL_TARGETS],
                progress_callback=progress_callback,
                max_chunks=CODER_INITIAL_CHUNKS_PER_FILE,
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
            for p in [str(item.get("argument", "")).split("\n")[0].split(":")[0].strip()]
            if p
        ))
        final_summary = (
            f"Applied changes to {', '.join(touched[:3])}."
            if touched else
            "Applied the requested code changes."
        )

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
    )