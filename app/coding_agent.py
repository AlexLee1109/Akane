"""Deterministic coding-model workflow for fresh reads and VS Code edits."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.config import (
    CODER_INITIAL_CHUNKS_PER_FILE,
    CODER_MAX_INITIAL_TARGETS,
    CODER_MAX_READ_CHUNKS_PER_FILE,
    CODER_MAX_TURNS,
    CODER_READ_CHUNK_LINES,
    MAX_TOKENS,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from app.editor_bridge import get_editor_bridge
from app.generation import clean_model_text, collapse_hidden_tag_gaps, execute_read_requests
from app.model_loader import ModelManager

_EDITOR_LAUNCH_COMMANDS = {"open_vscode", "open_project", "open_workspace"}
_READ_ACTION_COMMANDS = {"read_file", "read_current_file", "read_file_chunk", "read_current_file_chunk", "list_files"}
_EDIT_ACTION_COMMANDS = {
    "create_file",
    "write_file",
    "append_file",
    "replace_file_range",
    "replace_selection",
    "insert_text",
    "save_file",
    "format_document",
    "open_file",
}
_ALLOWED_ACTIONS = _READ_ACTION_COMMANDS | _EDIT_ACTION_COMMANDS | _EDITOR_LAUNCH_COMMANDS
_ACTIVE_EDITOR_COMMANDS = {"replace_selection", "insert_text", "format_document", "save_file"}
_ACTION_WAIT_SECONDS = 5.5
_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_FILE_RESULT_PATH_PATTERN = re.compile(r"^FILE\s+(.+?)\s+\(line-numbered(?:\s+lines\s+(\d+)-(\d+)\s+of\s+(\d+))?\)", re.IGNORECASE)
_SYMBOL_PATTERN = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_DEFERRED_SUMMARY_PATTERN = re.compile(
    r"\b(?:let me|i(?:'|’)ll|i will|i need to|first i(?:'|’)ll|before i|i should)\b.*\b"
    r"(?:look|check|inspect|review|read|figure out|understand|analyze|see)\b",
    re.IGNORECASE,
)
_META_SUMMARY_PATTERN = re.compile(
    r"\b(?:summary|done|read_requests|actions|reason|json|schema|return json|set done|provide suggestions|"
    r"the user asks|the user asked|we need to output json|we have already read|without any actions)\b",
    re.IGNORECASE,
)
_BLOCKER_PATTERN = re.compile(
    r"\b(?:blocked|missing|need|can't|cannot|unable|not enough|unclear|ambiguous|no file|no path|no code)\b",
    re.IGNORECASE,
)


@dataclass
class CoderOutcome:
    summary: str
    action_results: list[dict]
    actions_applied: bool
    tool_used: bool
    approval_required: bool = False
    proposed_actions: list[dict] | None = None


def _extract_message_text(result: dict) -> str:
    choices = result.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {}) or {}
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts).strip()
    return str(content).strip()


def _request_coder_completion(messages: list[dict]) -> dict:
    return ModelManager.get_instance().create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repeat_penalty=REPETITION_PENALTY,
        stream=False,
        role="coder",
    )


def _extract_json_payload(raw: str) -> dict:
    text = str(raw or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    for candidate in (text,):
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
    match = _JSON_BLOCK_PATTERN.search(text)
    if not match:
        return {}
    snippet = match.group(0)
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _sanitize_read_requests(payload: dict) -> list[str]:
    value = payload.get("read_requests") or []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        path = str(item or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _sanitize_actions(payload: dict) -> list[dict]:
    value = payload.get("actions") or []
    if not isinstance(value, list):
        return []
    actions: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        command = str(item.get("command") or "").strip().lower()
        argument = str(item.get("argument") or item.get("value") or "").strip()
        if not command or command not in _ALLOWED_ACTIONS:
            continue
        actions.append({"command": command, "argument": argument})
    return actions


def _render_read_results(results: list[str]) -> str:
    if not results:
        return ""
    return "Fresh file reads:\n\n" + "\n\n".join(str(item or "").strip() for item in results if str(item or "").strip())


def _render_action_results(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["Editor action results:"]
    for item in results:
        outcome = item.get("result") if item.get("ok") else item.get("error") or "failed"
        lines.append(f"- {item.get('command', 'action')}: {outcome}")
    return "\n".join(lines)


def _append_coder_feedback(messages: list[dict], raw: str, instruction: str) -> None:
    messages.extend(
        [
            {"role": "assistant", "content": raw or "{}"},
            {"role": "system", "content": instruction},
        ]
    )


def _fresh_reads_system_message(results: list[str]) -> str:
    return (
        f"{_render_read_results(results)}\n\n"
        "Those fresh reads are now the current source of truth."
    )


def _post_action_system_message(results: list[dict], verification_reads: list[str]) -> str:
    return (
        f"{_render_action_results(results)}\n\n{_render_read_results(verification_reads)}\n\n"
        "Those are the latest action results and fresh post-edit reads. "
        "Use them as the source of truth and now return your final JSON result."
    )


def _grounding_terms_for_paths(paths: list[str]) -> set[str]:
    terms: set[str] = set()
    for raw in paths:
        path = str(raw or "").strip()
        if not path:
            continue
        lowered = path.lower()
        terms.add(lowered)
        basename = lowered.rsplit("/", 1)[-1]
        stem = basename.rsplit(".", 1)[0]
        if basename:
            terms.add(basename)
        if stem:
            terms.add(stem)
    return {term for term in terms if len(term) >= 3}


def _grounding_terms_for_reads(read_results: list[str]) -> set[str]:
    terms: set[str] = set()
    for block in read_results:
        text = str(block or "")
        if not text:
            continue
        for match in _FILE_RESULT_PATH_PATTERN.finditer(text):
            path = match.group(1).strip().lower()
            if path:
                terms.update(_grounding_terms_for_paths([path]))
        for match in _SYMBOL_PATTERN.finditer(text):
            symbol = match.group(1).strip().lower()
            if len(symbol) >= 3:
                terms.add(symbol)
                terms.add(f"{symbol}()")
    return terms


def _parse_read_result_header(result_text: str) -> tuple[str, int | None, int | None, int | None]:
    first_line = str(result_text or "").splitlines()[0] if str(result_text or "") else ""
    match = _FILE_RESULT_PATH_PATTERN.search(first_line)
    if not match:
        return "", None, None, None
    path = match.group(1).strip()
    start_line = int(match.group(2)) if match.group(2) else None
    end_line = int(match.group(3)) if match.group(3) else None
    total_lines = int(match.group(4)) if match.group(4) else None
    return path, start_line, end_line, total_lines


def _run_single_read_action(command: str, argument: str, *, logical_path: str) -> str:
    bridge = get_editor_bridge()
    action_start_id = bridge.next_action_id()
    bridge.queue_action(command, argument)
    action_results = bridge.wait_for_action_results_after(
        after_id=action_start_id - 1,
        expected_count=1,
        timeout=_ACTION_WAIT_SECONDS,
    )
    for item in action_results:
        if item.get("command") != command or not item.get("ok"):
            continue
        result_text = str(item.get("result") or "").strip()
        if not result_text:
            continue
        path, _, _, _ = _parse_read_result_header(result_text)
        if path and logical_path and path != logical_path:
            continue
        return result_text
    return ""


def _read_target_via_bridge_chunks(
    logical_path: str,
    *,
    use_current_file: bool,
    max_chunks: int,
    progress_callback=None,
) -> list[str]:
    results: list[str] = []
    start_line = 1
    for _ in range(max_chunks):
        end_line = start_line + CODER_READ_CHUNK_LINES - 1
        if use_current_file:
            command = "read_current_file_chunk"
            argument = f"{start_line}:{end_line}"
        else:
            command = "read_file_chunk"
            argument = f"{logical_path}:{start_line}:{end_line}"
        if progress_callback:
            progress_callback("Reading files...")
        result_text = _run_single_read_action(command, argument, logical_path=logical_path)
        if not result_text:
            break
        results.append(result_text)
        _, chunk_start, chunk_end, total_lines = _parse_read_result_header(result_text)
        if chunk_end is None or total_lines is None or chunk_end >= total_lines:
            break
        next_start = chunk_end + 1
        if next_start <= start_line:
            break
        start_line = next_start
    return results


def _merge_grounding_terms(*groups: set[str]) -> set[str]:
    merged: set[str] = set()
    for group in groups:
        merged.update(group)
    return merged


def _summary_references_grounding(summary: str, grounding_terms: set[str]) -> bool:
    lowered = str(summary or "").lower()
    if not lowered or not grounding_terms:
        return False
    for term in grounding_terms:
        if term and term in lowered:
            return True
    return False


def _looks_like_deferred_summary(summary: str) -> bool:
    text = collapse_hidden_tag_gaps(str(summary or "")).strip()
    if not text:
        return False
    return bool(_DEFERRED_SUMMARY_PATTERN.search(text))


def _looks_like_meta_summary(summary: str) -> bool:
    text = collapse_hidden_tag_gaps(str(summary or "")).strip()
    if not text:
        return False
    if _META_SUMMARY_PATTERN.search(text):
        return True
    lowered = text.lower()
    return (
        lowered.startswith("1.")
        or lowered.startswith("2.")
        or lowered.startswith("- ")
    ) and any(token in lowered for token in ("done", "summary", "actions", "read", "json"))


def _looks_like_blocker(reason: str) -> bool:
    text = collapse_hidden_tag_gaps(str(reason or "")).strip()
    if not text:
        return False
    return bool(_BLOCKER_PATTERN.search(text))


def _summary_retry_instruction(
    *,
    summary: str,
    grounded_reads_available: bool,
    grounding_terms: set[str],
    apply_now: bool,
    action_results: list[dict],
) -> str:
    if not summary:
        return ""
    if _looks_like_meta_summary(summary):
        return (
            "Your summary leaked internal control language or JSON-instruction text. "
            "Return only the user-facing answer. "
            "Do not mention summary, done, actions, read_requests, reason, schema, or JSON."
        )
    if grounded_reads_available and (
        _looks_like_deferred_summary(summary) or not _summary_references_grounding(summary, grounding_terms)
    ):
        return (
            "You already have fresh authoritative code reads. "
            "Do not say you will inspect the code later. "
            "Analyze the current reads now and return a grounded summary that mentions the actual file or symbol you inspected."
        )
    if apply_now and not _has_successful_edit(action_results) and _looks_like_deferred_summary(summary):
        return (
            "The user asked you to apply the change now. "
            "A promise to inspect or implement later does not count. "
            "Either emit editor actions now, request specific missing reads, or return a concrete blocker."
        )
    return ""


def _done_retry_instruction(
    *,
    done: bool,
    summary: str,
    reason: str,
    apply_now: bool,
    action_results: list[dict],
) -> str:
    if not done or not (summary or reason):
        return ""
    if apply_now and not _has_successful_edit(action_results) and reason and not _looks_like_blocker(reason):
        return (
            "If no edit was applied, the reason must be a concrete blocker grounded in the current code. "
            "Otherwise emit editor actions or request the exact reads you still need."
        )
    return ""


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


def _explicit_action_target_path(action: dict) -> str:
    command = str(action.get("command") or "").lower()
    argument = str(action.get("argument") or "")
    if command in {"read_file", "open_file"}:
        return argument.split(":", 1)[0].strip()
    if command == "create_file":
        return argument.strip()
    if command in {"write_file", "append_file"}:
        return argument.split("\n", 1)[0].strip()
    if command == "replace_file_range":
        return argument.split("\n", 1)[0].split(":", 1)[0].strip()
    if command == "save_file":
        return argument.strip()
    return ""


def _prepare_editor_actions(actions: list[dict], active_file: str) -> list[dict]:
    queued = list(actions)
    needs_active_editor = any(str(action.get("command") or "").lower() in _ACTIVE_EDITOR_COMMANDS for action in queued)
    preferred_path = ""
    for action in queued:
        preferred_path = _explicit_action_target_path(action)
        if preferred_path:
            break
    if not preferred_path:
        preferred_path = active_file

    if (
        needs_active_editor
        and preferred_path
        and preferred_path != active_file
        and not any(
            str(action.get("command") or "").lower() == "open_file"
            and str(action.get("argument") or "").split(":", 1)[0].strip() == preferred_path
            for action in queued
        )
    ):
        queued = [{"command": "open_file", "argument": preferred_path}, *queued]
    return queued


def _has_successful_edit(results: list[dict]) -> bool:
    return any(item.get("ok") and item.get("command") in _EDIT_ACTION_COMMANDS for item in results)


def _build_system_prompt(*, apply_now: bool) -> str:
    mode_instruction = (
        "The user wants you to actually apply the code change now. "
        "Do not stop at a plan. "
        "Either request the missing reads you need, emit editor actions to make the change, or explain the exact blocker."
        if apply_now
        else
        "The user wants an accurate coding answer grounded in the current code."
    )
    return (
        "You are Akane's hidden coding model. "
        "Return JSON only. No markdown, no prose outside JSON.\n\n"
        "Schema:\n"
        "{\n"
        '  "summary": "short user-ready answer",\n'
        '  "done": true,\n'
        '  "read_requests": ["repo/relative/path.py"],\n'
        '  "actions": [{"command": "replace_file_range", "argument": "path:start:end\\\\ncode"}],\n'
        '  "reason": "optional blocker or note"\n'
        "}\n\n"
        "Rules:\n"
        "- Base your answer on the freshest file reads provided in this conversation.\n"
        "- If you need more code context, use read_requests instead of guessing.\n"
        "- Prefer reading files before answering.\n"
        "- Do not say that you will inspect the code later if fresh reads are already present. Analyze the reads you already have.\n"
        "- Never claim code changed unless actions were executed successfully and fresh reads confirmed the result.\n"
        "- Prefer replace_file_range only when you have exact line numbers from fresh line-numbered reads.\n"
        "- Prefer write_file only when replacing an entire small file and you provide the complete updated file.\n"
        "- Keep summary concise and grounded. Mention the file or symbol you inspected.\n"
        "- Do not include process labels or routing commentary like direct coding response, no editor needed, or no file operations needed.\n"
        f"- {mode_instruction}"
    )


def _fresh_read_results(
    targets: list[str],
    *,
    progress_callback=None,
    prefer_current_file: bool = True,
    max_chunks_per_file: int = CODER_MAX_READ_CHUNKS_PER_FILE,
) -> tuple[list[str], list[str]]:
    bridge = get_editor_bridge()
    snapshot = bridge.snapshot().get("context", {})
    connected = bool(snapshot.get("connected"))
    active_file = str(snapshot.get("active_file") or "").strip()
    normalized_targets: list[str] = []
    seen: set[str] = set()
    for raw in targets:
        path = str(raw or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        normalized_targets.append(path)

    if not connected:
        return execute_read_requests(normalized_targets[:CODER_MAX_INITIAL_TARGETS]), normalized_targets[:CODER_MAX_INITIAL_TARGETS]
    consulted: list[str] = []
    remaining_targets = list(normalized_targets[:CODER_MAX_INITIAL_TARGETS])
    results: list[str] = []
    completed_paths: set[str] = set()
    if prefer_current_file and active_file and (not remaining_targets or active_file in remaining_targets):
        consulted.append(active_file)
        results.extend(
            _read_target_via_bridge_chunks(
                active_file,
                use_current_file=True,
                max_chunks=max_chunks_per_file,
                progress_callback=progress_callback,
            )
        )
        completed_paths.add(active_file)
        remaining_targets = [path for path in remaining_targets if path != active_file]
    elif not remaining_targets and active_file:
        consulted.append(active_file)
        results.extend(
            _read_target_via_bridge_chunks(
                active_file,
                use_current_file=True,
                max_chunks=max_chunks_per_file,
                progress_callback=progress_callback,
            )
        )
        completed_paths.add(active_file)

    for path in remaining_targets:
        consulted.append(path)
        chunk_results = _read_target_via_bridge_chunks(
            path,
            use_current_file=False,
            max_chunks=max_chunks_per_file,
            progress_callback=progress_callback,
        )
        if chunk_results:
            results.extend(chunk_results)
            completed_paths.add(path)

    missing = [path for path in normalized_targets if path not in completed_paths]
    if missing:
        results.extend(execute_read_requests(missing))
    return results, consulted or normalized_targets[:CODER_MAX_INITIAL_TARGETS]


def _queue_editor_actions(actions: list[dict], *, progress_callback=None) -> list[dict]:
    if not actions:
        return []
    bridge = get_editor_bridge()
    snapshot = bridge.snapshot().get("context", {})
    active_file = str(snapshot.get("active_file") or "").strip()
    queued = _prepare_editor_actions(actions, active_file)
    if any(action["command"] in {"replace_file_range", "replace_selection", "insert_text", "format_document"} for action in queued):
        if not any(action["command"] == "save_file" for action in queued):
            save_target = ""
            for action in queued:
                save_target = _explicit_action_target_path(action)
                if save_target:
                    break
            queued.append({"command": "save_file", "argument": save_target or active_file})

    queued_actions: list[dict] = []
    action_start_id = bridge.next_action_id()
    for action in queued:
        queued_actions.append(bridge.queue_action(action["command"], action["argument"]))
    if any(action.get("status") == "pending_approval" for action in queued_actions):
        return [
            {
                **action,
                "ok": False,
                "result": "",
                "error": "Awaiting user approval.",
            }
            for action in queued_actions
        ]
    if progress_callback:
        progress_callback("Working in VS Code...")
    return bridge.wait_for_action_results_after(
        after_id=action_start_id - 1,
        expected_count=len(queued),
        timeout=_ACTION_WAIT_SECONDS,
    )


def _request_coder_payload(messages: list[dict], *, progress_callback=None) -> tuple[str, dict]:
    if progress_callback:
        progress_callback("Using coding model...")
    result = _request_coder_completion(messages)
    raw = _extract_message_text(result)
    payload = _extract_json_payload(raw)
    if payload:
        return raw, payload
    repair_instruction = (
        "Your last reply was empty. Reply now with a complete JSON object using the required schema."
        if not str(raw or "").strip()
        else
        "Your last reply was not valid JSON. Reply again with JSON only using the required schema."
    )
    repair_messages = [
        *messages,
        {"role": "assistant", "content": raw or "{}"},
        {
            "role": "system",
            "content": (
                f"{repair_instruction} "
                "Do not include markdown fences or any prose outside the JSON object."
            ),
        },
    ]
    if progress_callback:
        progress_callback("Using coding model...")
    repair_result = _request_coder_completion(repair_messages)
    repair_raw = _extract_message_text(repair_result)
    return repair_raw, _extract_json_payload(repair_raw)


def run_coder_specialist(
    task: str,
    *,
    preload_paths: list[str] | None = None,
    apply_now: bool = False,
    progress_callback=None,
) -> CoderOutcome:
    bridge = get_editor_bridge()
    preload_paths = [str(path or "").strip() for path in (preload_paths or []) if str(path or "").strip()]
    initial_targets = preload_paths[:CODER_MAX_INITIAL_TARGETS]
    fresh_reads, consulted = _fresh_read_results(
        initial_targets,
        progress_callback=progress_callback,
        prefer_current_file=True,
        max_chunks_per_file=CODER_INITIAL_CHUNKS_PER_FILE,
    )
    grounding_terms = _merge_grounding_terms(
        _grounding_terms_for_paths(consulted),
        _grounding_terms_for_reads(fresh_reads),
    )
    print(
        f"[Akane][coder] start apply_now={apply_now} initial_targets={initial_targets} fresh_reads={len(fresh_reads)}",
        flush=True,
    )

    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(apply_now=apply_now)},
    ]
    if bridge.format_for_prompt():
        messages.append({"role": "system", "content": bridge.format_for_prompt()})
    if consulted:
        messages.append(
            {
                "role": "system",
                "content": f"Likely relevant files: {', '.join(consulted[:CODER_MAX_INITIAL_TARGETS])}.",
            }
        )
    if fresh_reads:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"{_render_read_results(fresh_reads)}\n\n"
                    "Those are the freshest authoritative code reads currently available. "
                    "Use them before asking for more context."
                ),
            }
        )
    messages.append({"role": "user", "content": task})

    all_action_results: list[dict] = []
    tool_used = bool(fresh_reads)
    grounded_reads_available = bool(fresh_reads)
    used_read_signatures: set[tuple[str, ...]] = set()
    action_signatures: set[tuple[tuple[str, str], ...]] = set()
    last_summary = ""
    last_reason = ""

    for _ in range(CODER_MAX_TURNS):
        raw, payload = _request_coder_payload(messages, progress_callback=progress_callback)
        summary = clean_model_text(str(payload.get("summary") or "")).strip()
        reason = clean_model_text(str(payload.get("reason") or "")).strip()
        raw_visible = collapse_hidden_tag_gaps(clean_model_text(raw)).strip() if raw else ""
        if (
            not payload
            and not summary
            and raw_visible
            and not raw_visible.startswith(("{", "["))
            and not _looks_like_meta_summary(raw_visible)
        ):
            summary = raw_visible
        read_requests = _sanitize_read_requests(payload)
        actions = _sanitize_actions(payload)
        done = bool(payload.get("done"))
        print(
            f"[Akane][coder] turn raw_len={len(raw)} summary_len={len(summary)} reads={len(read_requests)} actions={len(actions)} done={done}",
            flush=True,
        )

        if summary:
            last_summary = summary
        if reason:
            last_reason = reason

        if read_requests:
            signature = tuple(read_requests)
            if signature in used_read_signatures:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You already requested those same reads. "
                            "Use the file results already provided or request different files."
                        ),
                    }
                )
                continue
            used_read_signatures.add(signature)
            fresh_results, _ = _fresh_read_results(
                read_requests,
                progress_callback=progress_callback,
                prefer_current_file=True,
                max_chunks_per_file=CODER_MAX_READ_CHUNKS_PER_FILE,
            )
            tool_used = tool_used or bool(fresh_results)
            grounded_reads_available = grounded_reads_available or bool(fresh_results)
            grounding_terms = _merge_grounding_terms(
                grounding_terms,
                _grounding_terms_for_paths(read_requests),
                _grounding_terms_for_reads(fresh_results),
            )
            _append_coder_feedback(messages, raw, _fresh_reads_system_message(fresh_results))
            continue

        if actions:
            signature = tuple((action["command"], action["argument"]) for action in actions)
            if signature in action_signatures:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You already emitted those same actions. "
                            "Do not repeat them. Either summarize what changed or explain the blocker."
                        ),
                    }
                )
                continue
            action_signatures.add(signature)
            results = _queue_editor_actions(actions, progress_callback=progress_callback)
            all_action_results.extend(results)
            tool_used = True
            if any(item.get("status") == "pending_approval" for item in results):
                return CoderOutcome(
                    summary="",
                    action_results=results,
                    actions_applied=False,
                    tool_used=True,
                    approval_required=True,
                    proposed_actions=results,
                )
            touched_paths: list[str] = []
            snapshot = bridge.snapshot().get("context", {})
            active_file = str(snapshot.get("active_file") or "").strip()
            for action in actions:
                target = _action_target_path(action, active_file)
                if target and target not in touched_paths:
                    touched_paths.append(target)
            verification_reads, _ = _fresh_read_results(
                touched_paths[:CODER_MAX_INITIAL_TARGETS],
                progress_callback=progress_callback,
                prefer_current_file=True,
                max_chunks_per_file=CODER_INITIAL_CHUNKS_PER_FILE,
            )
            grounded_reads_available = grounded_reads_available or bool(verification_reads)
            grounding_terms = _merge_grounding_terms(
                grounding_terms,
                _grounding_terms_for_paths(touched_paths),
                _grounding_terms_for_reads(verification_reads),
            )
            _append_coder_feedback(messages, raw, _post_action_system_message(results, verification_reads))
            continue

        retry_instruction = _summary_retry_instruction(
            summary=summary,
            grounded_reads_available=grounded_reads_available,
            grounding_terms=grounding_terms,
            apply_now=apply_now,
            action_results=all_action_results,
        )
        if retry_instruction:
            _append_coder_feedback(messages, raw, retry_instruction)
            continue
        done_retry_instruction = _done_retry_instruction(
            done=done,
            summary=summary,
            reason=reason,
            apply_now=apply_now,
            action_results=all_action_results,
        )
        if done_retry_instruction:
            _append_coder_feedback(messages, raw, done_retry_instruction)
            continue
        if done and (summary or reason):
            break
        if summary and not apply_now:
            break
        if summary and apply_now and _has_successful_edit(all_action_results):
            break
        if reason and apply_now and _looks_like_blocker(reason):
            break

        if not grounded_reads_available:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "You do not have enough fresh code context yet. "
                        "Request the exact file reads you need before answering."
                    ),
                }
            )
            continue

        messages.append(
            {
                "role": "system",
                "content": (
                    "That was not enough. "
                    "Either request more reads, emit editor actions, or return a grounded summary with done=true."
                ),
            }
        )

    actions_applied = _has_successful_edit(all_action_results)
    final_summary = last_summary
    if apply_now and not actions_applied:
        final_summary = last_reason or ""
    elif not final_summary:
        final_summary = last_reason or ""
    if _looks_like_meta_summary(final_summary):
        print("[Akane][coder] dropping leaked meta summary and falling back.", flush=True)
        final_summary = ""

    print(
        f"[Akane][coder] done summary_len={len(final_summary)} actions_applied={actions_applied} tool_used={tool_used}",
        flush=True,
    )
    return CoderOutcome(
        summary=final_summary.strip(),
        action_results=all_action_results,
        actions_applied=actions_applied,
        tool_used=tool_used,
    )
