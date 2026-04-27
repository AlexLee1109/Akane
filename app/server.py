"""Browser-based chat interface for Akane."""

import json
import os
import re
import threading
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.character import build_system_prompt
from app.codebase_search import CodebaseSearch
from app.config import ADVISOR_ONLY, CHAT_HISTORY_CONTEXT_TOKENS, LLAMA_CONTEXT_WINDOW, MAX_TOKENS, REPETITION_PENALTY, SERVER_HOST, SERVER_PORT, TEMPERATURE, TOP_K, TOP_P
from app.editor_bridge import get_editor_bridge
from app.generation import (
    HiddenTagStreamFilter,
    _generation_lock,
    build_runtime_context,
    capture_explicit_user_memories,
    collapse_hidden_tag_gaps,
    execute_read_requests,
    extract_coder_requests,
    extract_editor_commands,
    extract_read_requests,
    truncate_messages,
)
from app.memory import MEMORY_PATH, analyze_conversation_context, format_for_prompt, record_interaction, reload_from_disk
from app.model_loader import LLM, ModelManager
from app.reply_pipeline import (
    _clamp_companion_reply,
    _looks_incomplete_reply,
    clean_reply_text,
    ensure_complete_visible_reply,
    finalize_reply,
    normalize_final_reply,
    postprocess_reply,
)
from app.request_analysis import FILE_REF_PATTERN as REQUEST_FILE_REF_PATTERN, RequestAnalyzer, RequestSnapshot
from app.specialists import CoderOutcome, run_coder_specialist
from app.vscode_launcher import launch_vscode

HOST = SERVER_HOST
PORT = SERVER_PORT
STATIC_DIR = Path(__file__).parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_messages = deque(maxlen=40)
_state_lock = threading.Lock()
_streaming_reply_preview = ""
_EDITOR_ACTION_PATTERN = re.compile(
    r"\[EDITOR\].*?(?:\[/EDITOR\]|$)|<tool_call>.*?(?:</tool_call>|$)",
    re.DOTALL | re.IGNORECASE,
)
_FENCED_CODE_PATTERN = re.compile(r"```")
_FILE_REF_PATTERN = REQUEST_FILE_REF_PATTERN
_EDITOR_LAUNCH_COMMANDS = {"open_vscode", "open_project", "open_workspace"}
_EDITOR_AUTONOMY_MAX_TURNS = 2
_EDITOR_ACTION_WAIT_SECONDS = 4.5
_LLAMA_CONTEXT_SAFETY_TOKENS = 512
_MAX_INLINE_READ_RESULT_CHARS = 1800
_MAX_TOTAL_READ_RESULT_CHARS = 5200
_MAX_EXTRA_SYSTEM_CHARS = 3200
_DETAIL_REQUEST_PATTERN = re.compile(
    r"\b(detail|details|deeper|deep dive|full|full breakdown|walk through|walkthrough|show me|explain more|longer|full analysis)\b",
    re.IGNORECASE,
)
_IMPLEMENTATION_WALKTHROUGH_PATTERN = re.compile(
    r"\b(?:show me how|how (?:do|would|can) (?:i|we)\b.*\bimplement|how to implement|walk me through|implementation steps|wire (?:it|this|that) up|where should i (?:change|put|add)|what should i change)\b",
    re.IGNORECASE,
)
_SUGGESTION_REQUEST_PATTERN = re.compile(
    r"\b(suggest|suggestion|suggestions|improve|improvement|improvements|recommend|recommendation|recommendations|what should i change|how should i change|how can i improve)\b",
    re.IGNORECASE,
)
_CODER_SPECIALIST_REQUEST_PATTERN = re.compile(
    r"\b(?:debug|trace|root cause|why (?:is|does|did)|failing|failure|broken|bug|regression|refactor|rewrite|migrate|investigate|diagnose|test failure|stack trace|exception|error)\b",
    re.IGNORECASE,
)
_LAST_CODEBASE_TARGETS: list[str] = []
_ACTIVE_EDITOR_COMMANDS = {"replace_selection", "insert_text", "format_document", "save_file"}
_DEFERRED_ACTION_REPLY_PATTERN = re.compile(
    r"\b(?:i(?:'|’)ll|i will|let me|i can)\b.*\b(?:check|look at|inspect|review|read|implement|change|update|fix|patch|edit)\b",
    re.IGNORECASE,
)
_CODEBASE_SEARCH = CodebaseSearch(PROJECT_ROOT)
_REQUEST_ANALYZER = RequestAnalyzer(_CODEBASE_SEARCH)

# Compatibility aliases for older code paths that may still reference
# the pre-refactor helper names during a running session.
_finalize_reply = finalize_reply
_clean_reply_text = clean_reply_text
_normalize_final_reply = normalize_final_reply


def _model_status() -> dict[str, object]:
    return ModelManager.get_instance().status()


def _start_model_loading() -> None:
    manager = ModelManager.get_instance()
    status = manager.status()
    if status["loading"] or status["loaded"]:
        return

    def loader() -> None:
        try:
            manager.ensure_loaded()
        except Exception:
            pass

    threading.Thread(target=loader, daemon=True, name="AkaneModelLoader").start()


def _snapshot_messages() -> list[dict]:
    with _state_lock:
        return list(_messages)


def _state_messages() -> list[dict]:
    with _state_lock:
        messages = list(_messages)
        preview = _streaming_reply_preview.strip()
    if preview:
        messages.append({"role": "assistant", "content": preview})
    return messages


def _set_streaming_reply_preview(text: str) -> None:
    with _state_lock:
        global _streaming_reply_preview
        _streaming_reply_preview = str(text or "")


def _clear_streaming_reply_preview() -> None:
    _set_streaming_reply_preview("")


def _log_terminal_stream(label: str, text: str = "") -> None:
    message = f"[Akane:{label}]"
    if text:
        message = f"{message} {text}"
    print(message, flush=True)


def _explicit_paths_in_text(text: str) -> list[str]:
    seen: set[str] = set()
    matches: list[str] = []
    for match in _FILE_REF_PATTERN.finditer(str(text or "")):
        raw = match.group(1).strip("`'\".,:;()[]{}")
        if not raw:
            continue
        if "/" in raw:
            resolved = (PROJECT_ROOT / raw).resolve()
            if resolved.is_file() and str(resolved).startswith(str(PROJECT_ROOT)):
                rel = str(resolved.relative_to(PROJECT_ROOT))
                if rel not in seen:
                    seen.add(rel)
                    matches.append(rel)
            continue

        for rel in _CODEBASE_SEARCH.get_project_file_index().get(raw.lower(), []):
            if rel not in seen:
                seen.add(rel)
                matches.append(rel)
    return matches


def _remember_codebase_targets(paths: list[str]) -> None:
    remembered = [str(path or "").strip() for path in paths if str(path or "").strip()]
    if not remembered:
        return
    with _state_lock:
        _LAST_CODEBASE_TARGETS[:] = remembered[:5]


def _recent_codebase_targets() -> list[str]:
    with _state_lock:
        return list(_LAST_CODEBASE_TARGETS)


def _last_assistant_message() -> str:
    for message in reversed(_snapshot_messages()):
        if message.get("role") == "assistant":
            return str(message.get("content", "") or "").strip()
    return ""


def _last_user_message() -> str:
    for message in reversed(_snapshot_messages()):
        if message.get("role") == "user":
            return str(message.get("content", "") or "").strip()
    return ""


def _recent_exchange_context(limit: int = 4) -> str:
    messages = _snapshot_messages()
    if not messages:
        return ""
    recent = messages[-limit:]
    parts: list[str] = []
    for message in recent:
        role = str(message.get("role", "") or "").strip()
        content = collapse_hidden_tag_gaps(str(message.get("content", "") or "")).strip()
        if not content:
            continue
        label = "User" if role == "user" else "Akane" if role == "assistant" else role.title()
        parts.append(f"{label}: {content}")
    return "\n".join(parts)


def _recent_exchange_note(limit: int = 4) -> str:
    messages = _snapshot_messages()
    if not messages:
        return ""
    recent = messages[-limit:]
    last_user = ""
    last_assistant = ""
    for message in reversed(recent):
        role = str(message.get("role", "") or "").strip()
        content = collapse_hidden_tag_gaps(str(message.get("content", "") or "")).strip()
        if not content:
            continue
        if role == "assistant" and not last_assistant:
            last_assistant = content
        elif role == "user" and not last_user:
            last_user = content
        if last_user and last_assistant:
            break
    parts: list[str] = []
    if last_user:
        parts.append(f"Previous user message: {last_user[:500]}")
    if last_assistant:
        parts.append(f"Previous Akane reply: {last_assistant[:700]}")
    return "\n".join(parts)


def _request_snapshot(*, last_assistant_override: str | None = None) -> RequestSnapshot:
    context = _editor_snapshot()
    return RequestSnapshot(
        last_user=_last_user_message(),
        last_assistant=str(last_assistant_override if last_assistant_override is not None else _last_assistant_message()),
        recent_code_targets=tuple(_recent_codebase_targets()),
        active_file=str(context.get("active_file", "") or "").strip(),
        open_tabs=tuple(str(path or "").strip() for path in (context.get("open_tabs") or []) if str(path or "").strip()),
        recent_text=_recent_conversation_text(),
        editor_connected=bool(context.get("connected")),
    )


def _analyze_request(user_input: str, *, last_assistant_override: str | None = None):
    return _REQUEST_ANALYZER.analyze(user_input, _request_snapshot(last_assistant_override=last_assistant_override))


def _record_recent_interaction() -> None:
    context = analyze_conversation_context(_snapshot_messages()[-20:])
    record_interaction(conversation_context=context)


def _append_message(role: str, content: str) -> None:
    with _state_lock:
        _messages.append({"role": role, "content": content})
        if role == "assistant":
            global _streaming_reply_preview
            _streaming_reply_preview = ""
    if role == "assistant":
        _remember_codebase_targets(_explicit_paths_in_text(content))


def _clear_messages() -> None:
    with _state_lock:
        _messages.clear()
        global _streaming_reply_preview
        _streaming_reply_preview = ""


def _handle_command(text: str) -> dict | None:
    bridge = get_editor_bridge()
    if text == "/vscode":
        notice = launch_vscode()
        return {"reply": "", "messages": _snapshot_messages(), "notice": notice, "ephemeral": True}

    if text == "/approve":
        approved = bridge.approve_all_pending_actions()
        if not approved:
            return {"reply": "", "messages": _snapshot_messages(), "notice": "No pending code changes to approve.", "ephemeral": True}
        return {
            "reply": "",
            "messages": _snapshot_messages(),
            "notice": f"Approved {len(approved)} pending code change(s).",
            "ephemeral": True,
        }

    if text == "/reject":
        rejected = bridge.reject_all_pending_actions()
        if not rejected:
            return {"reply": "", "messages": _snapshot_messages(), "notice": "No pending code changes to reject.", "ephemeral": True}
        return {
            "reply": "",
            "messages": _snapshot_messages(),
            "notice": f"Rejected {len(rejected)} pending code change(s).",
            "ephemeral": True,
        }

    if text == "/memory":
        reload_from_disk()
        reply = format_for_prompt() or "No memories yet."
        preview_messages = _snapshot_messages() + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": reply},
        ]
        return {"reply": reply, "messages": preview_messages, "ephemeral": True}

    if text == "/clear":
        _clear_messages()
        return {"reply": "", "messages": [], "notice": "Context cleared."}

    if text == "/reset":
        _clear_messages()
        if MEMORY_PATH.exists():
            os.remove(MEMORY_PATH)
        reload_from_disk()
        return {"reply": "", "messages": [], "notice": "Memory wiped and context cleared."}

    return None


def _editor_snapshot() -> dict:
    return get_editor_bridge().snapshot().get("context", {})


def _editor_connected() -> bool:
    return bool(_editor_snapshot().get("connected"))


def _preview_action_change(action: dict) -> str:
    preview = ((action.get("meta") or {}).get("preview") or {})
    if preview:
        label = str(preview.get("label") or action.get("command") or "").strip()
        before = collapse_hidden_tag_gaps(str(preview.get("before") or "")).strip()
        after = collapse_hidden_tag_gaps(str(preview.get("after") or "")).strip()
        parts = [label]
        if before or after:
            parts.append(f"Current:\n{before or '(empty)'}")
            parts.append(f"Proposed:\n{after or '(empty)'}")
        return "\n".join(parts)
    command = str(action.get("command", "") or "").lower()
    argument = str(action.get("argument", "") or "")
    if command == "replace_file_range":
        header, _, content = argument.partition("\n")
        return f"{header} -> {collapse_hidden_tag_gaps(content).strip()[:180]}"
    if command in {"write_file", "append_file"}:
        header, _, content = argument.partition("\n")
        return f"{header} -> {collapse_hidden_tag_gaps(content).strip()[:180]}"
    if command in {"replace_selection", "insert_text"}:
        return collapse_hidden_tag_gaps(argument).strip()[:180]
    return collapse_hidden_tag_gaps(argument).strip()[:180]


def _format_pending_approval_reply(actions: list[dict]) -> str:
    if not actions:
        return "I staged a code change for approval. Use /approve to apply it or /reject to cancel it."
    lines = ["I staged these code changes for approval:"]
    visible_actions = [
        action
        for action in actions
        if str(action.get("command", "") or "").lower()
        not in {"save_file", "format_document", "open_file"}
    ] or list(actions)
    for index, action in enumerate(visible_actions[:3], start=1):
        preview = _preview_action_change(action)
        if preview:
            lines.append(f"{index}. {preview}")
        else:
            lines.append(f"{index}. {action.get('command')}")
    lines.append("Use /approve to apply them or /reject to cancel them.")
    return "\n".join(lines)


def _is_open_vscode_request(user_input: str) -> bool:
    return _analyze_request(user_input).open_vscode


def _is_codebase_context_request(user_input: str) -> bool:
    return _analyze_request(user_input).codebase_context


def _wants_code_execution(user_input: str) -> bool:
    return _analyze_request(user_input).wants_execution


def _looks_like_deferred_action_reply(reply: str) -> bool:
    text = collapse_hidden_tag_gaps(str(reply or "")).strip()
    if not text:
        return False
    if _response_uses_tool_calls(text):
        return False
    return bool(_DEFERRED_ACTION_REPLY_PATTERN.search(text))


def _recent_conversation_text(limit: int = 4) -> str:
    parts: list[str] = []
    for message in _snapshot_messages()[-limit:]:
        content = str(message.get("content", "") or "").strip()
        if content:
            parts.append(content)
    return " ".join(parts)


def _candidate_codebase_paths(user_input: str, limit: int = 3) -> list[str]:
    return _REQUEST_ANALYZER.candidate_paths(user_input, _request_snapshot(), limit=limit)


def _enforce_vscode_for_coding(chat_messages: list[dict]) -> list[dict]:
    instruction = (
        "The user is asking for coding or editor help. "
        "You must inspect the relevant code before replying with only plain advice or fenced code. "
        "If VS Code is connected, prefer VS Code editor actions. If not, use [READ] to inspect the relevant files. "
        "If the user wants the project opened in VS Code, emit [EDITOR]open_vscode[/EDITOR]. "
        "If code needs to be created or changed, emit [EDITOR] tags to read files, open files, write files, replace ranges, save files, or format files. "
        "For implementation or codebase questions, read the relevant file(s) first, then continue in the SAME reply with a short summary of what you found. "
        "Use only the square-bracket tag forms like [EDITOR]...[/EDITOR] and [READ]...[/READ]. "
        "Do not emit XML-style tool markup like <tool_call>, <editor>, <READ>, or <function=...>. "
        "Do not tell the user to copy, save, or paste code manually."
    )
    return [*chat_messages[:-1], {"role": "system", "content": instruction}, chat_messages[-1]]


def _should_force_vscode(user_input: str) -> bool:
    if ADVISOR_ONLY:
        return False
    return _analyze_request(user_input).should_force_vscode


def _should_use_coder_specialist(user_input: str, analysis=None) -> bool:
    analysis = analysis or _analyze_request(user_input)
    coding_like = analysis.coding_like
    if analysis.wants_brainstorm and not analysis.codebase_context and not analysis.wants_execution:
        return False
    if ADVISOR_ONLY:
        if coding_like:
            print("[Akane] Skipping coding model: advisor-only mode enabled.", flush=True)
        return False
    if analysis.open_vscode:
        if coding_like:
            print("[Akane] Skipping coding model: open-vscode request.", flush=True)
        return False
    manager = ModelManager.get_instance()
    status = manager.status()
    if status.get("coder_backend") != "openrouter":
        if coding_like:
            print(
                "[Akane] Skipping coding model: no OpenRouter coder backend configured.",
                flush=True,
            )
        return False
    coder_model = str(status.get("openrouter_coder_model") or "")
    if not coder_model:
        if coding_like:
            print("[Akane] Skipping coding model: no coder model configured.", flush=True)
        return False
    if not coding_like:
        return False
    lowered = str(user_input or "").lower()
    if analysis.wants_execution:
        specialist_reason = "execution request"
    elif analysis.codebase_context:
        specialist_reason = "codebase request"
    elif analysis.wants_single_suggestion:
        specialist_reason = "code review request"
    elif analysis.wants_detail:
        specialist_reason = "detailed coding request"
    elif _CODER_SPECIALIST_REQUEST_PATTERN.search(lowered):
        specialist_reason = "complex coding request"
    else:
        specialist_reason = "general coding request"
    print(
        f"[Akane] Coding model eligible ({specialist_reason}). main={status.get('model_name')} coder={coder_model}",
        flush=True,
    )
    return True


def _response_uses_editor_tools(raw: str) -> bool:
    return bool(_EDITOR_ACTION_PATTERN.search(raw))


def _response_uses_tool_calls(raw: str) -> bool:
    if ADVISOR_ONLY:
        return bool(extract_read_requests(raw))
    return _response_uses_editor_tools(raw) or bool(extract_read_requests(raw)) or bool(extract_coder_requests(raw))


def _response_uses_disabled_coding_tools(raw: str) -> bool:
    text = str(raw or "")
    if not text:
        return False
    return bool(extract_coder_requests(text) or _response_uses_editor_tools(text))


def _needs_editor_retry(user_input: str, raw: str) -> bool:
    if not _should_force_vscode(user_input):
        return False
    if _response_uses_tool_calls(raw):
        return False
    if _looks_like_deferred_action_reply(raw):
        return True
    if _FENCED_CODE_PATTERN.search(raw):
        return True
    return True


def _build_chat_messages(user_input: str) -> list[dict]:
    capture_explicit_user_memories(user_input)
    ModelManager.get_instance().ensure_loaded()
    messages = _snapshot_messages()
    system_prompt = build_system_prompt(build_runtime_context())
    chat_messages = truncate_messages(messages, system_prompt, user_input, max_context_tokens=CHAT_HISTORY_CONTEXT_TOKENS)
    last_assistant = _last_assistant_message()
    analysis = _analyze_request(user_input, last_assistant_override=last_assistant)

    if analysis.should_carry_last_reply:
        exchange = _recent_exchange_context()
        exchange_note = _recent_exchange_note()
        if exchange or exchange_note:
            chat_messages = [
                *chat_messages[:-1],
                {
                    "role": "system",
                    "content": (
                        "Treat the recent exchange as the default context for this turn unless the user clearly switches topics. "
                        "Resolve short references like it/that/this/the first one against the previous exchange, not in isolation. "
                        "If the user asks a short follow-up question, assume they mean the main subject, files, functions, parameters, or thresholds from the previous exchange. "
                        "Only drop that context if the user explicitly introduces a different topic.\n\n"
                        f"{exchange_note[:1400]}\n\n"
                        f"Recent exchange:\n{exchange[:1200]}"
                    ),
                },
                chat_messages[-1],
            ]
    if (
        not analysis.wants_detail
        and not analysis.coding_like
        and not analysis.wants_execution
    ):
        chat_messages = [
            *chat_messages[:-1],
            {
                "role": "system",
                "content": (
                    "For this turn, reply like a compact desktop companion message. "
                    "Use exactly one short sentence when possible, or two short declarative sentences at most if needed. "
                    "Stop immediately after the first complete thought unless a second short sentence is genuinely needed. "
                    "Stay coherent with the immediately previous exchange and continue the same conversational thread unless the user clearly changes topics. "
                    "Do not start with filler like Mm..., Mmm..., Hmm..., Ah..., or Oh.... "
                    "Do not add a follow-up question or end with a question mark unless the user explicitly needs clarification or emotional support. "
                    "Do not use routine check-in questions like 'How was your day so far?' or 'How's your day going so far?'. "
                    "Prefer a simple reaction, answer, or observation, then stop. "
                    "Do not add extra elaboration, examples, or sign-off sentences."
                ),
            },
            chat_messages[-1],
        ]
    if analysis.codebase_followup and analysis.assistant_invited_continuation:
        exchange = _recent_exchange_context()
        chat_messages = [
            *chat_messages[:-1],
            {
                "role": "system",
                "content": (
                    "The user is following up on your last question or offer. "
                    "Continue the same topic and do the thing you just offered. "
                    "Do not ask them to repeat or restate what they want. "
                    "If your last message offered an example, rewrite, sketch, or implementation step, provide it now.\n\n"
                    f"Recent exchange:\n{exchange[:1200]}"
                ),
                },
                chat_messages[-1],
            ]
    elif analysis.codebase_followup and analysis.referential_followup:
        exchange = _recent_exchange_context()
        if exchange:
            chat_messages = [
                *chat_messages[:-1],
                {
                    "role": "system",
                    "content": (
                        "The user is referring to the main subject of your last reply. "
                        "Resolve pronouns like it/that/this using the recent exchange and continue the same topic. "
                        "Do not ask what the referent is unless your previous answer truly discussed multiple competing things. "
                        f"Recent exchange:\n{exchange[:1200]}"
                    ),
                },
                chat_messages[-1],
            ]
    recent_targets = _recent_codebase_targets()
    if recent_targets and analysis.coding and not analysis.explicit_file_reference:
        chat_messages = [
            *chat_messages[:-1],
            {
                "role": "system",
                "content": (
                    "Recent code context: unless the user clearly switches topics, "
                    f"their follow-up coding request likely refers to these files: {', '.join(recent_targets[:3])}. "
                    "Prefer to keep working on those files before assuming a new code target."
                ),
                },
                chat_messages[-1],
            ]
    if _editor_connected() and analysis.wants_execution and not ADVISOR_ONLY:
        chat_messages = [
            *chat_messages[:-1],
            {
                "role": "system",
                "content": (
                    "The user wants you to apply the code change now, not just describe it. "
                    "Use VS Code editor actions to make the change if you have enough context. "
                    "Before editing, prefer reading the freshest editor state from VS Code for the target file."
                ),
                },
                chat_messages[-1],
            ]
    if ADVISOR_ONLY and analysis.coding_like:
        chat_messages = [
            *chat_messages[:-1],
            {
                "role": "system",
                "content": (
                    "Advisor-only mode is enabled. "
                    "Do not offer to apply edits, save files, patch code, or use hidden coding/tool execution. "
                    "You may inspect code with [READ] tags when needed, then explain the best updates or improvements in plain language."
                ),
            },
            chat_messages[-1],
        ]
    if _should_force_vscode(user_input):
        chat_messages = _enforce_vscode_for_coding(chat_messages)
    return chat_messages


def _run_coder_handoff(
    user_input: str,
    chat_messages: list[dict],
    initial_raw: str,
    *,
    progress_callback=None,
) -> str:
    def _fallback_from_coder_result(coder_text: str, fallback_hint: str = "") -> str:
        base = collapse_hidden_tag_gaps(str(coder_text or "").strip()).strip()
        if base:
            return ensure_complete_visible_reply(
                chat_messages,
                finalize_reply(base, fallback_hint),
                request_completion=lambda messages: _request_completion(messages, stream=False),
                extract_message_text=_extract_message_text,
                extract_finish_reason=_extract_finish_reason,
            )
        fallback_messages = _build_codebase_fallback_messages(
            chat_messages,
            user_input,
            coder_failed=True,
        )
        if fallback_messages:
            fallback_response = _request_completion(fallback_messages, stream=False, role="main")
            fallback_raw = _extract_message_text(fallback_response)
            fallback_cleaned = finalize_reply(fallback_raw, fallback_hint)
            return ensure_complete_visible_reply(
                fallback_messages,
                fallback_cleaned,
                finish_reason=_extract_finish_reason(fallback_response),
                request_completion=lambda messages: _request_completion(messages, stream=False),
                extract_message_text=_extract_message_text,
                extract_finish_reason=_extract_finish_reason,
            )
        return finalize_reply(fallback_hint or initial_raw)

    if progress_callback:
        progress_callback("Using coding model...")
    print("[Akane] Using coding model for coding specialist handoff.", flush=True)
    requests = extract_coder_requests(initial_raw)
    task = "\n\n".join(requests).strip() or user_input
    analysis = _analyze_request(user_input)
    if analysis.should_carry_last_reply:
        exchange = _recent_exchange_context()
        if exchange:
            task = (
                f"Recent exchange context:\n{exchange[:1400]}\n\n"
                f"Current user request:\n{task}"
            )
    preload_query = task if task and task != user_input else user_input
    preload_paths = _candidate_codebase_paths(preload_query, limit=3)
    apply_now = _wants_code_execution(user_input)
    coder_outcome = run_coder_specialist(
        task,
        preload_paths=preload_paths,
        apply_now=apply_now,
        progress_callback=progress_callback,
    )
    if coder_outcome.approval_required:
        return _format_pending_approval_reply(coder_outcome.proposed_actions or [])
    coder_result = coder_outcome.summary
    if not coder_result:
        print("[Akane] Coding model returned no usable result. Falling back to direct main-model answer.", flush=True)
        return _fallback_from_coder_result("", clean_reply_text(initial_raw))
    cleaned = ensure_complete_visible_reply(
        chat_messages,
        finalize_reply(coder_result, clean_reply_text(initial_raw)),
        request_completion=lambda messages: _request_completion(messages, stream=False),
        extract_message_text=_extract_message_text,
        extract_finish_reason=_extract_finish_reason,
    )
    return postprocess_reply(
        cleaned,
        analysis=_analyze_request(user_input),
        working_messages=chat_messages,
        request_completion=lambda messages: _request_completion(messages, stream=False),
        extract_message_text=_extract_message_text,
        extract_finish_reason=_extract_finish_reason,
    )


def _retry_messages_for_editor_actions(chat_messages: list[dict]) -> list[dict]:
    retry_instruction = (
        "You are handling a coding request and VS Code is connected. "
        "Retry this answer using VS Code editor actions. "
        "Use only square-bracket tool tags such as [EDITOR]...[/EDITOR] or [READ]...[/READ]. "
        "Do not emit XML-style tool markup. "
        "Do not answer with fenced code, copy-paste instructions, or a plain explanation alone."
    )
    return [*chat_messages, {"role": "system", "content": retry_instruction}]


def _retry_messages_for_advisor_only_tools(chat_messages: list[dict], raw: str) -> list[dict]:
    retry_instruction = (
        "Advisor-only mode is enabled. "
        "Do not use [ASK_CODER] or [EDITOR] tags. "
        "Do not offer to directly change files. "
        "If you need code context, use only [READ]path/to/file[/READ]. "
        "Otherwise answer directly with suggested updates and improvements."
    )
    return [
        *chat_messages,
        {"role": "assistant", "content": raw},
        {"role": "system", "content": retry_instruction},
    ]


def _wants_implementation_walkthrough(user_input: str) -> bool:
    text = str(user_input or "").strip()
    if not text:
        return False
    if _IMPLEMENTATION_WALKTHROUGH_PATTERN.search(text):
        return True
    analysis = _analyze_request(user_input)
    return analysis.coding_like and analysis.wants_detail


def _parse_editor_command(raw: str) -> tuple[str, str]:
    raw = str(raw or "").strip()
    if not raw:
        return "", ""
    if ":" not in raw:
        return raw.lower(), ""
    command, argument = raw.split(":", 1)
    return command.strip().lower(), argument.strip()


def _editor_command_target_path(command: str, argument: str, active_file: str) -> str:
    if command in {"read_file", "open_file"}:
        return argument.split(":", 1)[0].strip()
    if command == "create_file":
        return argument.strip()
    if command in {"write_file", "append_file"}:
        return argument.split("\n", 1)[0].strip()
    if command == "replace_file_range":
        return argument.split("\n", 1)[0].split(":", 1)[0].strip()
    if command == "save_file":
        return argument.strip() or active_file
    if command in {"replace_selection", "insert_text", "format_document"}:
        return active_file
    return ""


def _prepare_editor_commands(actions: list[tuple[str, str]], active_file: str) -> list[tuple[str, str]]:
    queued = list(actions)
    needs_active_editor = any(command in _ACTIVE_EDITOR_COMMANDS for command, _ in queued)
    preferred_path = ""
    for command, argument in queued:
        if command in {"replace_selection", "insert_text", "format_document"}:
            continue
        preferred_path = _editor_command_target_path(command, argument, "")
        if preferred_path:
            break
    if not preferred_path:
        preferred_path = active_file

    if (
        needs_active_editor
        and preferred_path
        and preferred_path != active_file
        and not any(command == "open_file" and argument.split(":", 1)[0].strip() == preferred_path for command, argument in queued)
    ):
        queued = [("open_file", preferred_path), *queued]
    return queued


def _render_editor_results(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["VS Code editor action results:"]
    for item in results:
        outcome = item.get("result") if item.get("ok") else item.get("error") or "failed"
        lines.append(f"- {item.get('command', 'action')}: {outcome}")
    return "\n".join(lines)


def _render_read_results(results: list[str]) -> str:
    if not results:
        return ""
    rendered: list[str] = ["Local file reads (private tool results, do not print verbatim):"]
    pattern = re.compile(r"^\[READ_RESULT\](.*?)(?:\[/READ_RESULT\])?$", re.DOTALL)
    total_chars = 0
    for item in results:
        text = str(item or "").strip()
        match = pattern.match(text)
        if match:
            text = match.group(1).strip()
        remaining = _MAX_TOTAL_READ_RESULT_CHARS - total_chars
        if remaining <= 0:
            rendered.append("[Additional file context trimmed for fit.]")
            break
        clipped = text[: min(_MAX_INLINE_READ_RESULT_CHARS, remaining)].strip()
        if len(text) > len(clipped):
            clipped = clipped.rstrip() + "\n...[trimmed]"
        rendered.append(clipped)
        total_chars += len(clipped)
    return "\n\n".join(rendered)


def _estimate_prompt_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        content = str(message.get("content", "") or "")
        total += max(1, len(content) // 4)
    return total


def _fit_messages_to_context(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages

    budget = max(1024, LLAMA_CONTEXT_WINDOW - MAX_TOKENS - _LLAMA_CONTEXT_SAFETY_TOKENS)
    fitted = [dict(message) for message in messages]

    for index, message in enumerate(fitted[1:-1], start=1):
        if message.get("role") != "system":
            continue
        content = str(message.get("content", "") or "")
        if len(content) <= _MAX_EXTRA_SYSTEM_CHARS:
            continue
        fitted[index]["content"] = content[:_MAX_EXTRA_SYSTEM_CHARS].rstrip() + "\n\n[Context trimmed for fit.]"

    while len(fitted) > 2 and _estimate_prompt_tokens(fitted) > budget:
        removable_index = next(
            (i for i, message in enumerate(fitted[1:-1], start=1) if message.get("role") != "system"),
            None,
        )
        if removable_index is None:
            removable_index = next(
                (i for i, message in enumerate(fitted[1:-1], start=1) if message.get("role") == "system"),
                None,
            )
        if removable_index is None:
            break
        del fitted[removable_index]

    if _estimate_prompt_tokens(fitted) > budget and fitted:
        system_content = str(fitted[0].get("content", "") or "")
        max_system_chars = max(1200, budget * 4 // 2)
        if len(system_content) > max_system_chars:
            fitted[0]["content"] = system_content[:max_system_chars].rstrip() + "\n\n[System context trimmed for fit.]"

    return fitted


def _build_codebase_fallback_messages(
    chat_messages: list[dict],
    user_input: str,
    *,
    coder_failed: bool = False,
) -> list[dict]:
    targets = _candidate_codebase_paths(user_input)
    if not targets:
        active_file = str(_editor_snapshot().get("active_file", "") or "").strip()
        if active_file:
            targets = [active_file]
    if not targets:
        return []
    _remember_codebase_targets(targets)

    read_results = execute_read_requests(targets)
    if not read_results:
        return []

    if _wants_implementation_walkthrough(user_input):
        answer_instruction = (
            "Answer the user's exact implementation question now. "
            "Give a concrete walkthrough grounded in the fresh reads. "
            "Explain which file or symbol to change first, what to add or update, and the order of the steps. "
            "A short numbered list is okay when it makes the implementation clearer. "
            "Do not print raw code, READ_RESULT blocks, or a file dump unless the user explicitly asked for code."
        )
    else:
        answer_instruction = (
            "Answer the user now in 1-3 short sentences by default. "
            "Do not print raw code, READ_RESULT blocks, or a file dump. "
            "Summarize the important findings and give the most useful suggestion first."
        )
    if coder_failed:
        answer_instruction = (
            "The hidden coding handoff did not produce a usable answer. "
            "Answer the user directly now without mentioning the failed handoff. "
            + answer_instruction
        )

    instruction = (
        f"{_render_read_results(read_results)}\n\n"
        "I already inspected the most relevant file(s) for this question. "
        "Treat those freshly read file contents as the current source of truth. "
        "If anything in earlier conversation history conflicts with the fresh file contents, trust the fresh file contents and correct yourself. "
        "Do not ask to read the files again unless absolutely necessary. "
        "Do not emit tool tags unless you truly still need another tool step. "
        f"{answer_instruction}"
    )
    return [*chat_messages, {"role": "system", "content": instruction}]


def _maybe_get_codebase_messages(
    base_messages: list[dict],
    working_messages: list[dict],
    user_input: str,
) -> list[dict]:
    if working_messages is not base_messages:
        return working_messages
    return _build_codebase_fallback_messages(base_messages, user_input)


def _should_preload_codebase_context(user_input: str, analysis=None) -> bool:
    analysis = analysis or _analyze_request(user_input)
    if _is_open_vscode_request(user_input):
        return False
    if _should_use_coder_specialist(user_input, analysis):
        return False
    return analysis.codebase_context


def _run_editor_followup_loop(
    user_input: str,
    chat_messages: list[dict],
    initial_raw: str,
    initial_visible: str = "",
    initial_finish_reason: str = "",
    progress_callback=None,
) -> str:
    bridge = get_editor_bridge()
    loop_messages = list(chat_messages)
    raw = initial_raw
    seen_read_signatures: set[tuple[str, ...]] = set()
    last_finish_reason = initial_finish_reason
    last_cleaned = ""
    saw_tool_work = False

    del initial_visible

    for _ in range(_EDITOR_AUTONOMY_MAX_TURNS):
        snapshot = bridge.snapshot().get("context", {})
        active_file = str(snapshot.get("active_file") or "").strip()
        editor_ops = extract_editor_commands(raw)
        actionable: list[tuple[str, str]] = []
        for op in editor_ops:
            command, argument = _parse_editor_command(op)
            if not command or command in _EDITOR_LAUNCH_COMMANDS:
                continue
            actionable.append((command, argument))
        actionable = _prepare_editor_commands(actionable, active_file)
        read_requests = extract_read_requests(raw)
        read_signature = tuple(read_requests)
        read_results: list[str] = []
        if read_requests and read_signature not in seen_read_signatures:
            if progress_callback:
                progress_callback("Reading files...")
            seen_read_signatures.add(read_signature)
            _remember_codebase_targets(read_requests)
            read_results = execute_read_requests(read_requests)
        queued_actions: list[dict] = []
        action_start_id = bridge.next_action_id()
        for command, argument in actionable:
            queued_actions.append(bridge.queue_action(command, argument))
        cleaned = clean_reply_text(raw)
        if cleaned:
            last_cleaned = cleaned

        if any(action.get("status") == "pending_approval" for action in queued_actions):
            return _format_pending_approval_reply(queued_actions)

        if not actionable and not read_results:
            if _wants_code_execution(user_input) and _looks_like_deferred_action_reply(cleaned):
                forced_result = _request_completion(
                    [
                        *loop_messages,
                        {"role": "assistant", "content": cleaned or "Applying the change."},
                        {
                            "role": "system",
                            "content": (
                                "The user wants the code changed now. "
                                "Do not explain or promise future work. "
                                "Emit only the [READ] and [EDITOR] tags needed to inspect the latest file and apply the edit. "
                                "If you already have enough context, emit the exact [EDITOR] actions now."
                            ),
                        },
                    ],
                    stream=False,
                )
                raw = _extract_message_text(forced_result)
                last_finish_reason = _extract_finish_reason(forced_result)
                forced_cleaned = clean_reply_text(raw)
                if forced_cleaned:
                    last_cleaned = forced_cleaned
                if _response_uses_tool_calls(raw):
                    continue
            break
        saw_tool_work = True

        results: list[dict] = []
        if actionable:
            if progress_callback:
                progress_callback("Working in VS Code...")
            results = bridge.wait_for_action_results_after(
                after_id=action_start_id - 1,
                expected_count=len(actionable),
                timeout=_EDITOR_ACTION_WAIT_SECONDS,
            )
        if actionable and not results and not read_results:
            break

        tool_context_parts = [
            part for part in (_render_editor_results(results), _render_read_results(read_results)) if part
        ]
        loop_messages = [
            *loop_messages,
            {"role": "assistant", "content": cleaned or "Let me check that in VS Code."},
            {
                "role": "system",
                "content": (
                    f"{'\n\n'.join(tool_context_parts)}\n\n"
                    "Continue helping the user using those results. "
                    "Do not print raw READ_RESULT blocks, raw file dumps, or verbatim tool output to the user. "
                    "Use the file contents privately and summarize the important findings in your own words. "
                    "If you already have enough information, answer briefly and concretely. "
                    "If you still need another tool call, emit it now."
                ),
            },
        ]
        if progress_callback:
            progress_callback("Thinking...")
        response = _request_completion(loop_messages, stream=False)
        raw = _extract_message_text(response)
        last_finish_reason = _extract_finish_reason(response)
        cleaned = clean_reply_text(raw)
        if cleaned:
            last_cleaned = cleaned

    final_reply = last_cleaned or finalize_reply(raw)
    if (saw_tool_work or _response_uses_tool_calls(initial_raw)) and (
        _looks_incomplete_reply(final_reply) or last_finish_reason == "length"
    ):
        repair_raw = _extract_message_text(
            _request_completion(
                [
                    *loop_messages,
                    {"role": "assistant", "content": final_reply},
                    {
                        "role": "system",
                        "content": (
                            "Your last visible reply was cut off or too partial. "
                            "Reply again with only the final short summary for the user. "
                            "Do not include the earlier 'reading' preamble. "
                            "Do not emit tool tags unless you still truly need them."
                        ),
                    },
                ],
                stream=False,
            )
        )
        repaired = clean_reply_text(repair_raw)
        if repaired:
            return repaired
    return finalize_reply(raw, final_reply)


def _request_completion(messages: list[dict], *, stream: bool, role: str = "main"):
    if role == "main":
        messages = _fit_messages_to_context(messages)
    return LLM.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repeat_penalty=REPETITION_PENALTY,
        stream=stream,
        role=role,
    )


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


def _extract_finish_reason(result: dict) -> str:
    choices = result.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("finish_reason", "") or "").strip().lower()


def _complete_reply(
    user_input: str,
    base_messages: list[dict],
    working_messages: list[dict],
    raw: str,
    *,
    finish_reason: str = "",
    visible_fallback: str = "",
    progress_callback=None,
) -> str:
    current_raw = raw
    current_finish_reason = finish_reason
    analysis = _analyze_request(user_input)

    if ADVISOR_ONLY and _response_uses_disabled_coding_tools(current_raw):
        advisor_retry = _request_completion(
            _retry_messages_for_advisor_only_tools(working_messages, current_raw),
            stream=False,
        )
        current_raw = _extract_message_text(advisor_retry)
        current_finish_reason = _extract_finish_reason(advisor_retry)

    coder_requested = bool(
        not ADVISOR_ONLY
        and analysis.coding_like
        and extract_coder_requests(current_raw)
    )

    if coder_requested or _should_use_coder_specialist(user_input, analysis):
        cleaned = _run_coder_handoff(
            user_input,
            working_messages,
            current_raw,
            progress_callback=progress_callback,
        )
    else:
        if _needs_editor_retry(user_input, current_raw):
            retry_result = _request_completion(_retry_messages_for_editor_actions(working_messages), stream=False)
            current_raw = _extract_message_text(retry_result)
            current_finish_reason = _extract_finish_reason(retry_result)

        if _response_uses_tool_calls(current_raw):
            cleaned = _run_editor_followup_loop(
                user_input,
                working_messages,
                current_raw,
                visible_fallback,
                initial_finish_reason=current_finish_reason,
                progress_callback=progress_callback,
            )
        elif _is_codebase_context_request(user_input):
            codebase_messages = _maybe_get_codebase_messages(base_messages, working_messages, user_input)
            if codebase_messages is working_messages and working_messages is not base_messages:
                if _response_uses_tool_calls(current_raw):
                    cleaned = _run_editor_followup_loop(
                        user_input,
                        working_messages,
                        current_raw,
                        initial_finish_reason=current_finish_reason,
                        progress_callback=progress_callback,
                    )
                else:
                    cleaned = finalize_reply(current_raw, visible_fallback)
            elif codebase_messages:
                fallback_result = _request_completion(codebase_messages, stream=False)
                fallback_raw = _extract_message_text(fallback_result)
                current_finish_reason = _extract_finish_reason(fallback_result)
                if _response_uses_tool_calls(fallback_raw):
                    cleaned = _run_editor_followup_loop(
                        user_input,
                        codebase_messages,
                        fallback_raw,
                        initial_finish_reason=current_finish_reason,
                        progress_callback=progress_callback,
                    )
                else:
                    cleaned = finalize_reply(fallback_raw, visible_fallback)
            else:
                cleaned = finalize_reply(current_raw, visible_fallback)
        else:
            cleaned = finalize_reply(current_raw, visible_fallback)

    cleaned = ensure_complete_visible_reply(
        working_messages,
        cleaned,
        finish_reason=current_finish_reason,
        request_completion=lambda messages: _request_completion(messages, stream=False),
        extract_message_text=_extract_message_text,
        extract_finish_reason=_extract_finish_reason,
    )
    cleaned = postprocess_reply(
        cleaned,
        analysis=analysis,
        working_messages=working_messages,
        request_completion=lambda messages: _request_completion(messages, stream=False),
        extract_message_text=_extract_message_text,
        extract_finish_reason=_extract_finish_reason,
    )
    return normalize_final_reply(cleaned)


def _generate_reply(user_input: str) -> str:
    with _generation_lock:
        chat_messages = _build_chat_messages(user_input)
        _append_message("user", user_input)
        analysis = _analyze_request(user_input)
        if _should_use_coder_specialist(user_input, analysis):
            cleaned = _run_coder_handoff(user_input, chat_messages, user_input)
        else:
            working_messages = chat_messages
            if _should_preload_codebase_context(user_input, analysis):
                fallback_messages = _build_codebase_fallback_messages(chat_messages, user_input)
                if fallback_messages:
                    working_messages = fallback_messages

            result = _request_completion(working_messages, stream=False)
            cleaned = _complete_reply(
                user_input,
                chat_messages,
                working_messages,
                _extract_message_text(result),
                finish_reason=_extract_finish_reason(result),
            )

    _append_message("assistant", cleaned)
    _record_recent_interaction()
    return cleaned


class AkaneHandler(BaseHTTPRequestHandler):
    server_version = "AkaneHTTP/1.0"
    protocol_version = "HTTP/1.1"

    def _send_cors_headers(self) -> None:
        origin = self.headers.get("Origin", "*")
        self.send_header("Access-Control-Allow-Origin", origin if origin else "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Vary", "Origin")

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_stream_headers(self) -> None:
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _write_stream_event(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        self.wfile.write(line.encode("utf-8"))
        self.wfile.flush()

    def _handle_stream_chat(self, text: str) -> None:
        self._send_stream_headers()
        try:
            with _generation_lock:
                _clear_streaming_reply_preview()
                _log_terminal_stream("user", text)
                chat_messages = _build_chat_messages(text)
                _append_message("user", text)
                self._write_stream_event({"type": "start", "messages": _snapshot_messages()})
                if _should_use_coder_specialist(text):
                    cleaned = _run_coder_handoff(
                        text,
                        chat_messages,
                        text,
                        progress_callback=lambda label: self._write_stream_event(
                            {"type": "status", "label": label}
                        ),
                    )
                    self._write_stream_event({"type": "delta", "content": cleaned})
                    _set_streaming_reply_preview(cleaned)
                    _log_terminal_stream("assistant", cleaned)
                else:
                    suppress_intermediate_text = _should_force_vscode(text) or _is_codebase_context_request(text)
                    working_messages = chat_messages
                    if _should_preload_codebase_context(text):
                        _log_terminal_stream("status", "Reading files...")
                        self._write_stream_event({"type": "status", "label": "Reading files..."})
                        fallback_messages = _build_codebase_fallback_messages(chat_messages, text)
                        if fallback_messages:
                            working_messages = fallback_messages

                    stream = _request_completion(working_messages, stream=True)

                    hidden_filter = HiddenTagStreamFilter()
                    full_response: list[str] = []
                    displayed_text = ""
                    last_logged_text = ""
                    stream_finish_reason = ""

                    for chunk in stream:
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})
                        token = delta.get("content")
                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            stream_finish_reason = str(finish_reason).strip().lower()
                        if not token:
                            continue

                        full_response.append(token)
                        visible = hidden_filter.feed(token)
                        if not visible:
                            continue

                        displayed_text = collapse_hidden_tag_gaps(displayed_text + visible)
                        if not suppress_intermediate_text:
                            _set_streaming_reply_preview(displayed_text)
                            self._write_stream_event({"type": "delta", "content": displayed_text})
                            if displayed_text.startswith(last_logged_text):
                                delta = displayed_text[len(last_logged_text):]
                            else:
                                delta = displayed_text
                            if delta:
                                print(delta, end="", flush=True)
                                last_logged_text = displayed_text

                    visible = hidden_filter.flush()
                    if visible:
                        displayed_text = collapse_hidden_tag_gaps(displayed_text + visible)
                        if not suppress_intermediate_text:
                            _set_streaming_reply_preview(displayed_text)
                            self._write_stream_event({"type": "delta", "content": displayed_text})
                            if displayed_text.startswith(last_logged_text):
                                delta = displayed_text[len(last_logged_text):]
                            else:
                                delta = displayed_text
                            if delta:
                                print(delta, end="", flush=True)
                                last_logged_text = displayed_text

                    raw = "".join(full_response).strip()
                    streamed_visible = collapse_hidden_tag_gaps(displayed_text).strip()
                    if not suppress_intermediate_text and streamed_visible:
                        cleaned = _clamp_companion_reply(streamed_visible)
                    else:
                        cleaned = _complete_reply(
                            text,
                            chat_messages,
                            working_messages,
                            raw,
                            finish_reason=stream_finish_reason,
                            visible_fallback=displayed_text,
                            progress_callback=lambda label: self._write_stream_event(
                                {"type": "status", "label": label}
                            ),
                        )
                    if suppress_intermediate_text or cleaned != displayed_text:
                        _set_streaming_reply_preview(cleaned)
                        self._write_stream_event({"type": "delta", "content": cleaned})
                        if last_logged_text:
                            print("", flush=True)
                        _log_terminal_stream("assistant", cleaned)
                    elif last_logged_text:
                        print("", flush=True)
                _append_message("assistant", cleaned)
                _record_recent_interaction()
                _log_terminal_stream("done", cleaned)
                self._write_stream_event(
                    {"type": "done", "reply": cleaned, "messages": _snapshot_messages()}
                )
        except Exception as exc:  # pragma: no cover - surfaced in browser
            _clear_streaming_reply_preview()
            _log_terminal_stream("error", str(exc))
            self._write_stream_event({"type": "error", "error": str(exc)})

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if route == "/app.js":
            return self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
        if route == "/styles.css":
            return self._send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
        if route == "/popup_icon.png":
            return self._send_file(PROJECT_ROOT / "popup_icon.png", "image/png")
        if route == "/input_bar.png":
            return self._send_file(PROJECT_ROOT / "input_bar.png", "image/png")
        if route == "/send_button.png":
            return self._send_file(PROJECT_ROOT / "send_button.png", "image/png")
        if route == "/api/state":
            payload = {
                "messages": _state_messages(),
                "model": _model_status(),
                "editor": _editor_snapshot(),
            }
            return self._send_json(payload)
        if route == "/api/editor/state":
            return self._send_json(get_editor_bridge().snapshot())
        if route == "/api/editor/actions":
            query = parse_qs(urlparse(self.path).query)
            try:
                after_id = int((query.get("after") or ["0"])[0])
            except ValueError:
                after_id = 0
            bridge = get_editor_bridge()
            return self._send_json({"actions": bridge.actions_after(after_id)})

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route not in {
            "/api/chat",
            "/api/chat/stream",
            "/api/backend",
            "/api/quit",
            "/api/open-vscode",
            "/api/editor/context",
            "/api/editor/action-result",
        }:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        if route in {"/api/chat", "/api/chat/stream"}:
            _start_model_loading()

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            return self._send_json({"error": "Invalid JSON."}, HTTPStatus.BAD_REQUEST)

        if route == "/api/editor/context":
            snapshot = get_editor_bridge().update_context(payload)
            return self._send_json({"ok": True, "state": snapshot})

        if route == "/api/editor/action-result":
            action_id = int(payload.get("id", 0) or 0)
            if action_id <= 0:
                return self._send_json({"error": "Missing action id."}, HTTPStatus.BAD_REQUEST)
            summary = get_editor_bridge().complete_action(
                action_id=action_id,
                ok=bool(payload.get("ok", False)),
                result=str(payload.get("result", "")),
                error=str(payload.get("error", "")),
            )
            if summary is None:
                return self._send_json({"error": "Unknown action id."}, HTTPStatus.NOT_FOUND)
            return self._send_json({"ok": True, "summary": summary})

        if route == "/api/backend":
            backend = str(payload.get("backend", "")).strip().lower()
            model_name = payload.get("model_name")
            local_model_path = payload.get("local_model_path")
            try:
                status = ModelManager.get_instance().switch_backend(
                    backend,
                    local_model_path=str(local_model_path) if local_model_path is not None else None,
                    openrouter_model=str(model_name) if model_name is not None else None,
                )
            except ValueError as exc:
                return self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return self._send_json({"ok": True, "model": status})

        if route == "/api/open-vscode":
            try:
                notice = launch_vscode()
            except RuntimeError as exc:
                return self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return self._send_json({"ok": True, "notice": notice, "editor": _editor_snapshot()})

        if route == "/api/quit":
            def _quit_soon() -> None:
                import time
                time.sleep(0.15)
                os._exit(0)

            threading.Thread(target=_quit_soon, daemon=True, name="AkaneQuit").start()
            return self._send_json({"ok": True}, HTTPStatus.ACCEPTED)

        text = str(payload.get("message", "")).strip()
        if not text:
            return self._send_json({"error": "Message is empty."}, HTTPStatus.BAD_REQUEST)

        command_result = _handle_command(text)
        if command_result is not None:
            command_result.setdefault("messages", _snapshot_messages())
            return self._send_json(command_result)

        if route == "/api/chat/stream":
            return self._handle_stream_chat(text)

        model_status = _model_status()
        if model_status["error"]:
            return self._send_json({"error": model_status["error"]}, HTTPStatus.INTERNAL_SERVER_ERROR)

        try:
            reply = _generate_reply(text)
        except Exception as exc:  # pragma: no cover - surfaced in browser
            return self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        return self._send_json({"reply": reply, "messages": _snapshot_messages()})


def serve(host: str = HOST, port: int = PORT) -> None:
    try:
        httpd = create_http_server(host=host, port=port)
    except OSError as exc:
        print(f"Failed to start Akane web chat on http://{host}:{port} - {exc}", flush=True)
        raise

    print(f"Akane web chat running at http://{host}:{port}", flush=True)
    print("Model loading is deferred until the first chat request.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Akane web chat...")
    finally:
        httpd.server_close()


def create_http_server(host: str = HOST, port: int = PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), AkaneHandler)


def serve_in_thread(host: str = HOST, port: int = PORT) -> tuple[ThreadingHTTPServer, threading.Thread]:
    httpd = create_http_server(host=host, port=port)
    thread = threading.Thread(
        target=httpd.serve_forever,
        daemon=True,
        name="AkaneHTTPServer",
    )
    thread.start()
    print(f"Akane background API running at http://{host}:{port}", flush=True)
    print("Model loading is deferred until the first chat request.", flush=True)
    return httpd, thread


if __name__ == "__main__":
    serve()
