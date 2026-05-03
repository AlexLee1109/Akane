"""Browser-based chat interface for Akane."""

import json
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.codebase_search import CodebaseSearch
from app.config import ADVISOR_ONLY, CHAT_HISTORY_CONTEXT_TOKENS, LLAMA_CONTEXT_WINDOW, MAX_TOKENS, REPETITION_PENALTY, SERVER_HOST, SERVER_PORT, TEMPERATURE, TOP_K, TOP_P
from app.coding_agent import run_coder_specialist
from app.editor_bridge import get_editor_bridge
from app.generation import (
    HiddenTagStreamFilter,
    _cached_system_prompt,
    _generation_lock,
    build_runtime_context,
    capture_explicit_user_memories,
    clean_model_text,
    collapse_hidden_tag_gaps,
    execute_read_requests,
    extract_coder_requests,
    extract_editor_commands,
    extract_read_requests,
    truncate_messages,
)
from app.memory_store import MEMORY_PATH, analyze_conversation_context, format_for_prompt, record_interaction, reload_from_disk
from app.model_loader import LLM, ModelManager
from app.reply_pipeline import (
    clean_reply_text,
    ensure_complete_visible_reply,
    finalize_reply,
    normalize_final_reply,
    postprocess_reply,
)
from app.request_analysis import RequestAnalyzer, RequestSnapshot
from app.ui_assets import resolve_ui_asset
from app.vscode_launcher import launch_vscode

HOST = SERVER_HOST
PORT = SERVER_PORT
STATIC_DIR = Path(__file__).parent / "static"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_state_lock = threading.Lock()
_EDITOR_LAUNCH_COMMANDS = {"open_vscode", "open_project", "open_workspace"}
_EDITOR_AUTONOMY_MAX_TURNS = 2
_EDITOR_ACTION_WAIT_SECONDS = 4.5
_LLAMA_CONTEXT_SAFETY_TOKENS = 512
_MAX_INLINE_READ_RESULT_CHARS = 1800
_MAX_TOTAL_READ_RESULT_CHARS = 5200
_MAX_EXTRA_SYSTEM_CHARS = 3200
_FAST_CHAT_MAX_TOKENS = min(MAX_TOKENS, 160)
_FACTUAL_CHAT_MAX_TOKENS = min(MAX_TOKENS, 96)
_INFRA_PATH_PREFIXES = (
    "integrations/",
    "extensions/",
    ".vscode/",
    ".idea/",
    ".zed/",
    ".github/",
    ".gitlab/",
)
DEFAULT_SESSION_ID = "popup"


@dataclass
class SessionState:
    messages: deque = field(default_factory=lambda: deque(maxlen=40))
    streaming_reply_preview: str = ""
    recent_code_targets: list[str] = field(default_factory=list)
    version: int = 0
    history_cache_version: int = -1
    cached_last_user: str = ""
    cached_last_assistant: str = ""
    cached_recent_text: str = ""
    cached_recent_exchange_context: str = ""
    cached_recent_exchange_note: str = ""


@dataclass(frozen=True)
class ChatRequestData:
    text: str
    skip_memory: bool = False
    session_id: str = DEFAULT_SESSION_ID
_ACTIVE_EDITOR_COMMANDS = {"replace_selection", "insert_text", "format_document", "save_file"}
_CODEBASE_SEARCH = CodebaseSearch(PROJECT_ROOT)
_REQUEST_ANALYZER = RequestAnalyzer(_CODEBASE_SEARCH)
_SESSIONS: dict[str, SessionState] = {}
_IMPLEMENTATION_WALKTHROUGH_PHRASES = (
    "show me how", "how to implement", "walk me through", "implementation steps",
    "wire it up", "wire this up", "wire that up", "where should i change",
    "where should i put", "where should i add", "what should i change",
)
_CODER_SPECIALIST_HINTS = (
    "debug", "trace", "root cause", "failing", "failure", "broken", "bug", "regression",
    "refactor", "rewrite", "migrate", "investigate", "diagnose", "test failure",
    "stack trace", "exception", "error",
)
_CODER_SPECIALIST_WHY_PREFIXES = ("why is", "why does", "why did")
_INFRA_CONTEXT_KEYWORDS = (
    "vscode", "vs code", "extension", "extensions", "integration", "integrations",
    "bridge", "launch.json", "settings.json", "workflow", "workflows", "github actions", "ci", "pipeline",
)
_DEFERRED_ACTION_OPENERS = ("i'll", "i’ll", "i will", "let me", "i can")
_DEFERRED_ACTION_VERBS = (
    "check", "look at", "inspect", "review", "read", "implement", "change", "update",
    "fix", "patch", "edit", "outline", "sketch", "draft", "split", "refactor", "rewrite",
    "restructure", "reorganize",
)

_COMPACT_REPLY_RULES = (
    "Use one to three short declarative sentences max, and prefer one or two when possible. "
    "Stop immediately after the first complete thought unless a second or third short sentence is genuinely needed. "
    "Start immediately with the real reply content. "
    "Answer the user's literal question first before discussing nearby or related topics. "
    "Do not replace the asked topic with an adjacent one. "
    "If they ask whether they should do something, answer that recommendation directly in the first sentence. "
    "Do not begin with filler like Mm..., Mmm..., Hmm..., Ah..., Oh..., Well..., or Heh.... "
    "Do not generate a filler opener and then continue into the real answer. "
    "Do not add a follow-up question or end with a question mark unless the user explicitly needs clarification or emotional support. "
    "Do not use routine check-in questions like 'How was your day so far?' or 'How's your day going so far?'."
)
_TECHNICAL_FOCUS_RULES = (
    "Stay focused on the inspected code and the user's actual engineering question. "
    "Do not switch into emotional support, life advice, productivity coaching, or generic encouragement. "
    "Do not restate the user's request. "
    "Do not print raw code, READ_RESULT blocks, or a file dump."
)


def _compact_companion_instruction() -> str:
    return (
        "For this turn, reply like a compact desktop companion message. "
        f"{_COMPACT_REPLY_RULES} "
        "If the user is asking for suggestions or improvements, give only the single best recommendation first, not a list. "
        "Stay coherent with the immediately previous exchange and continue the same conversational thread unless the user clearly changes topics. "
        "Prefer a simple reaction, answer, or observation, then stop. "
        "Do not add extra elaboration, examples, or sign-off sentences."
    )


def _technical_answer_instruction(*, single_suggestion: bool = False) -> str:
    suggestion_rule = (
        "Give only the single best technical suggestion first, grounded in the fresh reads. "
        if single_suggestion
        else "Summarize the important findings and give the most useful suggestion first. "
    )
    return (
        f"Answer the user now in 1-3 short technical sentences by default. {_COMPACT_REPLY_RULES} "
        f"{_TECHNICAL_FOCUS_RULES} "
        f"{suggestion_rule}"
    )


def _static_response_path(route: str) -> tuple[Path, str] | None:
    if route == "/":
        return STATIC_DIR / "index.html", "text/html; charset=utf-8"
    if route == "/app.js":
        return STATIC_DIR / "app.js", "application/javascript; charset=utf-8"
    if route == "/styles.css":
        return STATIC_DIR / "styles.css", "text/css; charset=utf-8"

    asset_path = resolve_ui_asset(route)
    if asset_path is not None:
        return asset_path, "image/png"

    return None


def _normalize_session_id(session_id: str | None) -> str:
    cleaned = str(session_id or "").strip()
    return cleaned[:120] if cleaned else DEFAULT_SESSION_ID


def _session_state(session_id: str | None = None) -> SessionState:
    key = _normalize_session_id(session_id)
    with _state_lock:
        state = _SESSIONS.get(key)
        if state is None:
            state = SessionState()
            _SESSIONS[key] = state
        return state


def _model_status() -> dict[str, object]:
    return ModelManager.get_instance().status()


async def _request_payload(request: Request) -> dict:
    if not request.headers.get("content-length"):
        return {}
    payload = await request.json()
    return payload if isinstance(payload, dict) else {}


def _parse_chat_request(payload: dict) -> ChatRequestData:
    return ChatRequestData(
        text=str(payload.get("message", "")).strip(),
        skip_memory=_coerce_bool(payload.get("skip_memory", False)),
        session_id=_normalize_session_id(payload.get("session_id")),
    )


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


def _warm_model_now() -> None:
    manager = ModelManager.get_instance()
    status = manager.status()
    if status["loading"] or status["loaded"]:
        return
    print("Loading model now...", flush=True)
    manager.ensure_loaded()
    print("Model warmup complete.", flush=True)


def _snapshot_messages(session_id: str | None = None) -> list[dict]:
    return list(_session_state(session_id).messages)


def _refresh_session_history_cache(state: SessionState) -> None:
    if state.history_cache_version == state.version:
        return
    messages = list(state.messages)
    last_user = ""
    last_assistant = ""
    recent_text_parts: list[str] = []
    recent_exchange_parts: list[str] = []
    for message in messages[-8:]:
        role = str(message.get("role", "") or "").strip()
        content = collapse_hidden_tag_gaps(str(message.get("content", "") or "")).strip()
        if not content:
            continue
        recent_text_parts.append(content)
        label = "User" if role == "user" else "Akane" if role == "assistant" else role.title()
        recent_exchange_parts.append(f"{label}: {content}")
    for message in reversed(messages):
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
    note_parts: list[str] = []
    if last_user:
        note_parts.append(f"Previous user message: {last_user[:500]}")
    if last_assistant:
        note_parts.append(f"Previous Akane reply: {last_assistant[:700]}")
    state.cached_last_user = last_user
    state.cached_last_assistant = last_assistant
    state.cached_recent_text = " ".join(recent_text_parts)
    state.cached_recent_exchange_context = "\n".join(recent_exchange_parts)
    state.cached_recent_exchange_note = "\n".join(note_parts)
    state.history_cache_version = state.version


def _state_messages(session_id: str | None = None) -> list[dict]:
    state = _session_state(session_id)
    messages = list(state.messages)
    preview = state.streaming_reply_preview.strip()
    if preview:
        messages.append({"role": "assistant", "content": preview})
    return messages


def _set_streaming_reply_preview(text: str, session_id: str | None = None) -> None:
    state = _session_state(session_id)
    next_text = str(text or "")
    if state.streaming_reply_preview == next_text:
        return
    state.streaming_reply_preview = next_text
    state.version += 1


def _clear_streaming_reply_preview(session_id: str | None = None) -> None:
    _set_streaming_reply_preview("", session_id)


def _log_terminal_stream(label: str, text: str = "") -> None:
    message = f"[Akane:{label}]"
    if text:
        message = f"{message} {text}"
    print(message, flush=True)


def _explicit_paths_in_text(text: str) -> list[str]:
    seen: set[str] = set()
    matches: list[str] = []
    for raw_value in _REQUEST_ANALYZER.extract_file_refs(text):
        raw = str(raw_value or "").strip("`'\".,:;()[]{}")
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


def _remember_codebase_targets(paths: list[str], session_id: str | None = None) -> None:
    remembered = [
        str(path or "").strip()
        for path in paths
        if str(path or "").strip() and not _is_infra_path(str(path or "").strip())
    ]
    if not remembered:
        return
    state = _session_state(session_id)
    next_targets = remembered[:5]
    if state.recent_code_targets == next_targets:
        return
    state.recent_code_targets = next_targets
    state.version += 1


def _recent_codebase_targets(session_id: str | None = None) -> list[str]:
    return list(_session_state(session_id).recent_code_targets)


def _last_message(role: str, session_id: str | None = None) -> str:
    state = _session_state(session_id)
    _refresh_session_history_cache(state)
    if role == "user":
        return state.cached_last_user
    if role == "assistant":
        return state.cached_last_assistant
    return ""


def _recent_exchange_context(limit: int = 8, session_id: str | None = None) -> str:
    del limit
    state = _session_state(session_id)
    _refresh_session_history_cache(state)
    return state.cached_recent_exchange_context


def _recent_exchange_note(limit: int = 8, session_id: str | None = None) -> str:
    del limit
    state = _session_state(session_id)
    _refresh_session_history_cache(state)
    return state.cached_recent_exchange_note


def _request_snapshot(*, session_id: str | None = None, last_assistant_override: str | None = None) -> RequestSnapshot:
    state = _session_state(session_id)
    _refresh_session_history_cache(state)
    context = _editor_snapshot()
    return RequestSnapshot(
        last_user=state.cached_last_user,
        last_assistant=str(last_assistant_override if last_assistant_override is not None else state.cached_last_assistant),
        recent_code_targets=tuple(_recent_codebase_targets(session_id)),
        active_file=str(context.get("active_file", "") or "").strip(),
        open_tabs=tuple(str(path or "").strip() for path in (context.get("open_tabs") or []) if str(path or "").strip()),
        recent_text=state.cached_recent_text,
        editor_connected=bool(context.get("connected")),
    )


def _analyze_request(user_input: str, *, session_id: str | None = None, last_assistant_override: str | None = None):
    return _REQUEST_ANALYZER.analyze(
        user_input,
        _request_snapshot(session_id=session_id, last_assistant_override=last_assistant_override),
    )


def _record_recent_interaction(session_id: str | None = None) -> None:
    context = analyze_conversation_context(_snapshot_messages(session_id)[-12:])
    record_interaction(conversation_context=context)


def _runtime_context_for_request(*, skip_memory: bool = False, include_editor: bool = True) -> str:
    if not skip_memory:
        return build_runtime_context(include_editor=include_editor)
    try:
        return build_runtime_context(include_memory=False, include_editor=include_editor)
    except TypeError:
        return get_editor_bridge().format_for_prompt() if include_editor else ""


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _append_message(role: str, content: str, session_id: str | None = None) -> None:
    state = _session_state(session_id)
    state.messages.append({"role": role, "content": content})
    state.version += 1
    if role == "assistant":
        state.streaming_reply_preview = ""
    if role == "assistant":
        _remember_codebase_targets(_explicit_paths_in_text(content), session_id)


def _clear_messages(session_id: str | None = None) -> None:
    state = _session_state(session_id)
    if not state.messages and not state.streaming_reply_preview and not state.recent_code_targets:
        return
    state.messages.clear()
    state.streaming_reply_preview = ""
    state.recent_code_targets.clear()
    state.version += 1


def _handle_command(text: str, session_id: str | None = None) -> dict | None:
    bridge = get_editor_bridge()
    if text == "/vscode":
        notice = launch_vscode()
        return {"reply": "", "messages": _snapshot_messages(session_id), "notice": notice, "ephemeral": True}

    if text == "/approve":
        approved = bridge.approve_all_pending_actions()
        if not approved:
            return {"reply": "", "messages": _snapshot_messages(session_id), "notice": "No pending code changes to approve.", "ephemeral": True}
        return {
            "reply": "",
            "messages": _snapshot_messages(session_id),
            "notice": f"Approved {len(approved)} pending code change(s).",
            "ephemeral": True,
        }

    if text == "/reject":
        rejected = bridge.reject_all_pending_actions()
        if not rejected:
            return {"reply": "", "messages": _snapshot_messages(session_id), "notice": "No pending code changes to reject.", "ephemeral": True}
        return {
            "reply": "",
            "messages": _snapshot_messages(session_id),
            "notice": f"Rejected {len(rejected)} pending code change(s).",
            "ephemeral": True,
        }

    if text == "/memory":
        reload_from_disk()
        reply = format_for_prompt() or "No memories yet."
        preview_messages = _snapshot_messages(session_id) + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": reply},
        ]
        return {"reply": reply, "messages": preview_messages, "ephemeral": True}

    if text == "/clear":
        _clear_messages(session_id)
        return {"reply": "", "messages": [], "notice": "Context cleared."}

    if text == "/reset":
        _clear_messages(session_id)
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


def _is_open_vscode_request(user_input: str, analysis=None, *, session_id: str | None = None) -> bool:
    analysis = analysis or _analyze_request(user_input, session_id=session_id)
    return analysis.open_vscode


def _is_codebase_context_request(user_input: str, analysis=None, *, session_id: str | None = None) -> bool:
    analysis = analysis or _analyze_request(user_input, session_id=session_id)
    return analysis.codebase_context


def _wants_code_execution(user_input: str, analysis=None, *, session_id: str | None = None) -> bool:
    analysis = analysis or _analyze_request(user_input, session_id=session_id)
    return analysis.wants_execution


def _looks_like_deferred_action_reply(reply: str) -> bool:
    text = collapse_hidden_tag_gaps(str(reply or "")).strip()
    if not text:
        return False
    if _response_uses_tool_calls(text):
        return False
    lowered = text.lower()
    return any(opener in lowered for opener in _DEFERRED_ACTION_OPENERS) and any(
        verb in lowered for verb in _DEFERRED_ACTION_VERBS
    )


def _recent_conversation_text(limit: int = 8, session_id: str | None = None) -> str:
    del limit
    state = _session_state(session_id)
    _refresh_session_history_cache(state)
    return state.cached_recent_text


def _candidate_codebase_paths(user_input: str, limit: int = 3, session_id: str | None = None) -> list[str]:
    return _REQUEST_ANALYZER.candidate_paths(user_input, _request_snapshot(session_id=session_id), limit=limit)


def _is_infra_path(path: str) -> bool:
    lowered = str(path or "").strip().lower()
    if not lowered:
        return False
    if any(lowered.startswith(prefix) for prefix in _INFRA_PATH_PREFIXES):
        return True
    return any(
        marker in lowered
        for marker in (
            "/.vscode/",
            "/.idea/",
            "/.zed/",
            "/.github/",
            "/.gitlab/",
        )
    )


def _filter_coder_preload_paths(paths: list[str], user_input: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    lowered_input = str(user_input or "").lower()
    allow_infra = any(token in lowered_input for token in _INFRA_CONTEXT_KEYWORDS)
    for raw in paths:
        rel = str(raw or "").strip()
        if not rel or rel in seen:
            continue
        lowered = rel.lower()
        if not allow_infra and _is_infra_path(lowered):
            continue
        seen.add(rel)
        cleaned.append(rel)
    return cleaned


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


def _should_force_vscode(user_input: str, analysis=None, *, session_id: str | None = None) -> bool:
    if ADVISOR_ONLY:
        return False
    analysis = analysis or _analyze_request(user_input, session_id=session_id)
    return analysis.should_force_vscode


def _should_use_coder_specialist(user_input: str, analysis=None, *, session_id: str | None = None) -> bool:
    analysis = analysis or _analyze_request(user_input, session_id=session_id)
    coding_like = analysis.coding_like
    current_turn_explicitly_non_code = bool(
        not analysis.coding
        and not analysis.codebase_direct
        and not analysis.explicit_file_reference
        and not analysis.wants_execution
        and not analysis.wants_single_suggestion
        and not analysis.wants_detail
    )
    if current_turn_explicitly_non_code:
        return False
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
    explicit_codebase_followup = bool(
        analysis.codebase_followup
        and (
            analysis.followup_reference
            or analysis.explanation_followup
            or analysis.affirmative_followup
            or analysis.assistant_invited_continuation
        )
    )
    explicit_coder_turn = bool(
        analysis.wants_execution
        or analysis.codebase_direct
        or explicit_codebase_followup
        or analysis.explicit_file_reference
        or analysis.wants_single_suggestion
        or analysis.wants_detail
        or any(hint in lowered for hint in _CODER_SPECIALIST_HINTS)
        or any(lowered.startswith(prefix) for prefix in _CODER_SPECIALIST_WHY_PREFIXES)
        or (analysis.coding and not analysis.short_followup and len(analysis.query_tokens) >= 3)
    )
    if not explicit_coder_turn:
        print("[Akane] Skipping coding model: current turn is a lightweight follow-up.", flush=True)
        return False
    if analysis.wants_execution:
        specialist_reason = "execution request"
    elif analysis.codebase_context:
        specialist_reason = "codebase request"
    elif analysis.wants_single_suggestion:
        specialist_reason = "code review request"
    elif analysis.wants_detail:
        specialist_reason = "detailed coding request"
    elif any(hint in lowered for hint in _CODER_SPECIALIST_HINTS) or any(lowered.startswith(prefix) for prefix in _CODER_SPECIALIST_WHY_PREFIXES):
        specialist_reason = "complex coding request"
    else:
        specialist_reason = "general coding request"
    print(
        f"[Akane] Coding model eligible ({specialist_reason}). main={status.get('model_name')} coder={coder_model}",
        flush=True,
    )
    return True


def _response_uses_editor_tools(raw: str) -> bool:
    text = str(raw or "")
    lowered = text.lower()
    return "[EDITOR]" in text or "<tool_call>" in lowered


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
    if "```" in str(raw or ""):
        return True
    return True


def _needs_codebase_action_retry(analysis, raw: str) -> bool:
    if not (analysis.codebase_context or analysis.coding_like):
        return False
    if _response_uses_tool_calls(raw):
        return False
    return _looks_like_deferred_action_reply(raw)


def _is_light_chat_turn(analysis) -> bool:
    return bool(
        not analysis.coding_like
        and not analysis.wants_execution
        and not analysis.open_vscode
        and not analysis.explicit_file_reference
        and not analysis.wants_detail
    )


def _is_simple_factual_turn(user_input: str, analysis) -> bool:
    if not _is_light_chat_turn(analysis):
        return False
    if analysis.wants_single_suggestion or analysis.wants_brainstorm or analysis.should_carry_last_reply:
        return False
    text = collapse_hidden_tag_gaps(str(user_input or "")).strip()
    if not text or "?" not in text:
        return False
    return len(analysis.query_tokens) <= 8


def _completion_overrides_for_turn(user_input: str, analysis) -> dict[str, float | int]:
    if _is_simple_factual_turn(user_input, analysis):
        return {
            "max_tokens_override": _FACTUAL_CHAT_MAX_TOKENS,
            "temperature_override": min(TEMPERATURE, 0.45),
        }
    if _is_light_chat_turn(analysis):
        return {"max_tokens_override": _FAST_CHAT_MAX_TOKENS}
    return {}


def _should_attach_recent_exchange(analysis, user_input: str) -> bool:
    if not analysis.should_carry_last_reply:
        return False
    if analysis.coding_like or analysis.wants_execution or analysis.wants_detail:
        return True

    text = collapse_hidden_tag_gaps(str(user_input or "")).strip()
    if not text:
        return False

    lowered = text.lower()
    if len(lowered) <= 12:
        return lowered not in {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "cool", "nice", "sure", "yep", "yes", "no"}
    if len(lowered) <= 40 and " " not in lowered:
        return False
    return len(lowered.split()) >= 3


def _prepare_chat_turn(user_input: str, *, skip_memory: bool = False, session_id: str | None = None) -> tuple[list[dict], object]:
    last_assistant = _last_message("assistant", session_id)
    analysis = _analyze_request(user_input, session_id=session_id, last_assistant_override=last_assistant)
    if not skip_memory:
        capture_explicit_user_memories(user_input)
    ModelManager.get_instance().ensure_loaded()
    messages = _snapshot_messages(session_id)
    light_chat_turn = _is_light_chat_turn(analysis)
    runtime_context = _runtime_context_for_request(
        skip_memory=skip_memory,
        include_editor=not light_chat_turn,
    )
    system_prompt = _cached_system_prompt(runtime_context)
    chat_messages = truncate_messages(messages, system_prompt, user_input, max_context_tokens=CHAT_HISTORY_CONTEXT_TOKENS)

    if _should_attach_recent_exchange(analysis, user_input):
        exchange = _recent_exchange_context(session_id=session_id)
        exchange_note = _recent_exchange_note(session_id=session_id)
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
                        f"{exchange_note[:500]}\n\n"
                        f"Recent exchange:\n{exchange[:500]}"
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
                "content": _compact_companion_instruction(),
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
                    f"Recent exchange:\n{exchange[:700]}"
                ),
                },
                chat_messages[-1],
            ]
    elif analysis.codebase_followup and analysis.referential_followup:
        exchange = _recent_exchange_context(session_id=session_id)
        if exchange:
            chat_messages = [
                *chat_messages[:-1],
                {
                    "role": "system",
                    "content": (
                        "The user is referring to the main subject of your last reply. "
                        "Resolve pronouns like it/that/this using the recent exchange and continue the same topic. "
                        "Do not ask what the referent is unless your previous answer truly discussed multiple competing things. "
                        f"Recent exchange:\n{exchange[:700]}"
                    ),
                },
                chat_messages[-1],
            ]
    recent_targets = _recent_codebase_targets(session_id)
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
    if _should_force_vscode(user_input, analysis, session_id=session_id):
        chat_messages = _enforce_vscode_for_coding(chat_messages)
    return chat_messages, analysis


def _build_chat_messages(user_input: str, *, skip_memory: bool = False, session_id: str | None = None) -> list[dict]:
    chat_messages, _ = _prepare_chat_turn(user_input, skip_memory=skip_memory, session_id=session_id)
    return chat_messages


def _run_coder_handoff(
    user_input: str,
    chat_messages: list[dict],
    initial_raw: str,
    *,
    session_id: str | None = None,
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
    analysis = _analyze_request(user_input, session_id=session_id)
    if analysis.should_carry_last_reply:
        exchange = _recent_exchange_context(session_id=session_id)
        if exchange:
            task = (
                f"Recent exchange context:\n{exchange[:1400]}\n\n"
                f"Current user request:\n{task}"
            )
    preload_query = task if task and task != user_input else user_input
    preload_paths: list[str] = []
    if analysis.codebase_context or analysis.explicit_file_reference:
        preload_paths = _filter_coder_preload_paths(
            _candidate_codebase_paths(preload_query, limit=6, session_id=session_id),
            preload_query,
        )[:3]
    apply_now = _wants_code_execution(user_input, analysis, session_id=session_id)
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
        analysis=_analyze_request(user_input, session_id=session_id),
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
        "Do not say you will inspect, outline, split, refactor, or update the code later. "
        "Do the work now in this same reply. "
        "Do not answer with fenced code, copy-paste instructions, or a plain explanation alone."
    )
    return [*chat_messages, {"role": "system", "content": retry_instruction}]


def _retry_messages_for_codebase_action(
    chat_messages: list[dict],
    raw: str,
    user_input: str,
) -> list[dict]:
    retry_instruction = (
        "The user asked for real codebase help, not a promise of future work. "
        "Your previous reply sounded like a plan or intention instead of doing the task. "
        "Do the work now in this same reply. "
        "If you need file context, use [READ]path/to/file[/READ] now. "
        "If VS Code is connected and files need to change, emit [EDITOR] actions now. "
        "If you already have enough context, answer concretely now. "
        "Do not say you will outline, inspect, split, refactor, rewrite, review, or update something later. "
        "Do not narrate intent. "
        "For requests like reorganizing, splitting, or refactoring a file, either inspect the file now or give the actual grounded split/refactor answer now."
    )
    return [
        *chat_messages,
        {"role": "assistant", "content": raw or user_input},
        {"role": "system", "content": retry_instruction},
    ]


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


def _wants_implementation_walkthrough(user_input: str, *, session_id: str | None = None) -> bool:
    text = str(user_input or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if any(phrase in lowered for phrase in _IMPLEMENTATION_WALKTHROUGH_PHRASES):
        return True
    if ("how do i" in lowered or "how would i" in lowered or "how can i" in lowered) and "implement" in lowered:
        return True
    analysis = _analyze_request(user_input, session_id=session_id)
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
    total_chars = 0
    for item in results:
        text = str(item or "").strip()
        if text.startswith("[READ_RESULT]"):
            text = text[len("[READ_RESULT]") :]
            if text.endswith("[/READ_RESULT]"):
                text = text[: -len("[/READ_RESULT]")]
            text = text.strip()
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
    session_id: str | None = None,
    coder_failed: bool = False,
) -> list[dict]:
    targets = _candidate_codebase_paths(user_input, session_id=session_id)
    if not targets:
        active_file = str(_editor_snapshot().get("active_file", "") or "").strip()
        if active_file:
            targets = [active_file]
    if not targets:
        return []
    _remember_codebase_targets(targets, session_id)

    read_results = execute_read_requests(targets)
    if not read_results:
        return []

    if _wants_implementation_walkthrough(user_input, session_id=session_id):
        answer_instruction = (
            "Answer the user's exact implementation question now. "
            "Give a concrete walkthrough grounded in the fresh reads. "
            "Explain which file or symbol to change first, what to add or update, and the order of the steps. "
            "A short numbered list is okay when it makes the implementation clearer. "
            "Do not print raw code, READ_RESULT blocks, or a file dump unless the user explicitly asked for code."
        )
    elif _analyze_request(user_input, session_id=session_id).wants_single_suggestion:
        answer_instruction = (
            "Answer the user's technical code-review or improvement request now. "
            "Stay strictly focused on the inspected code, files, functions, structure, performance, or maintainability. "
            + _technical_answer_instruction(single_suggestion=True)
        )
    else:
        answer_instruction = _technical_answer_instruction()
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
    *,
    session_id: str | None = None,
) -> list[dict]:
    if working_messages is not base_messages:
        return working_messages
    return _build_codebase_fallback_messages(base_messages, user_input, session_id=session_id)


def _should_preload_codebase_context(user_input: str, analysis=None, *, session_id: str | None = None, use_coder_specialist: bool | None = None) -> bool:
    analysis = analysis or _analyze_request(user_input, session_id=session_id)
    if _is_open_vscode_request(user_input, analysis, session_id=session_id):
        return False
    if use_coder_specialist is None:
        use_coder_specialist = _should_use_coder_specialist(user_input, analysis, session_id=session_id)
    if use_coder_specialist:
        return False
    return analysis.codebase_context


def _run_editor_followup_loop(
    user_input: str,
    chat_messages: list[dict],
    initial_raw: str,
    initial_visible: str = "",
    initial_finish_reason: str = "",
    session_id: str | None = None,
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
            if _wants_code_execution(user_input, session_id=session_id) and _looks_like_deferred_action_reply(cleaned):
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
    if (saw_tool_work or _response_uses_tool_calls(initial_raw)) and last_finish_reason == "length":
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


def _request_completion(
    messages: list[dict],
    *,
    stream: bool,
    role: str = "main",
    max_tokens_override: int | None = None,
    temperature_override: float | None = None,
):
    if role == "main":
        messages = _fit_messages_to_context(messages)
    return LLM.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens_override if max_tokens_override is not None else MAX_TOKENS,
        temperature=temperature_override if temperature_override is not None else TEMPERATURE,
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
    session_id: str | None = None,
    analysis=None,
    use_coder_specialist: bool | None = None,
    progress_callback=None,
) -> str:
    current_raw = raw
    current_finish_reason = finish_reason
    analysis = analysis or _analyze_request(user_input, session_id=session_id)
    light_chat_turn = _is_light_chat_turn(analysis)

    if light_chat_turn:
        cleaned = finalize_reply(current_raw, visible_fallback)
        return normalize_final_reply(cleaned)

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

    if use_coder_specialist is None:
        use_coder_specialist = _should_use_coder_specialist(user_input, analysis, session_id=session_id)

    if coder_requested or use_coder_specialist:
        cleaned = _run_coder_handoff(
            user_input,
            working_messages,
            current_raw,
            session_id=session_id,
            progress_callback=progress_callback,
        )
    else:
        if _needs_codebase_action_retry(analysis, current_raw):
            retry_result = _request_completion(
                _retry_messages_for_codebase_action(working_messages, current_raw, user_input),
                stream=False,
            )
            current_raw = _extract_message_text(retry_result)
            current_finish_reason = _extract_finish_reason(retry_result)

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
                session_id=session_id,
                progress_callback=progress_callback,
            )
        elif _is_codebase_context_request(user_input, analysis, session_id=session_id):
            codebase_messages = _maybe_get_codebase_messages(
                base_messages,
                working_messages,
                user_input,
                session_id=session_id,
            )
            if codebase_messages is working_messages and working_messages is not base_messages:
                if _response_uses_tool_calls(current_raw):
                    cleaned = _run_editor_followup_loop(
                        user_input,
                        working_messages,
                        current_raw,
                        initial_finish_reason=current_finish_reason,
                        session_id=session_id,
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
                        session_id=session_id,
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


def _generate_reply(user_input: str, *, skip_memory: bool = False, session_id: str | None = None) -> str:
    with _generation_lock:
        chat_messages, analysis = _prepare_chat_turn(user_input, skip_memory=skip_memory, session_id=session_id)
        _append_message("user", user_input, session_id)
        light_chat_turn = _is_light_chat_turn(analysis)
        completion_overrides = _completion_overrides_for_turn(user_input, analysis)
        use_coder_specialist = False if light_chat_turn else _should_use_coder_specialist(user_input, analysis, session_id=session_id)
        if use_coder_specialist:
            cleaned = _run_coder_handoff(user_input, chat_messages, user_input, session_id=session_id)
        else:
            working_messages = chat_messages
            if not light_chat_turn and _should_preload_codebase_context(user_input, analysis, session_id=session_id, use_coder_specialist=use_coder_specialist):
                fallback_messages = _build_codebase_fallback_messages(chat_messages, user_input, session_id=session_id)
                if fallback_messages:
                    working_messages = fallback_messages

            result = _request_completion(working_messages, stream=False, **completion_overrides)
            cleaned = _complete_reply(
                user_input,
                chat_messages,
                working_messages,
                _extract_message_text(result),
                finish_reason=_extract_finish_reason(result),
                session_id=session_id,
                analysis=analysis,
                use_coder_specialist=use_coder_specialist,
            )

    _append_message("assistant", cleaned, session_id)
    if not skip_memory:
        _record_recent_interaction(session_id)
    return cleaned


def _json_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _stream_chat_events(text: str, *, skip_memory: bool = False, session_id: str | None = None):
    try:
        with _generation_lock:
            _clear_streaming_reply_preview(session_id)
            _log_terminal_stream("user", text)
            chat_messages, analysis = _prepare_chat_turn(text, skip_memory=skip_memory, session_id=session_id)
            _append_message("user", text, session_id)
            yield _json_line({"type": "start", "messages": _snapshot_messages(session_id)})
            light_chat_turn = _is_light_chat_turn(analysis)
            completion_overrides = _completion_overrides_for_turn(text, analysis)
            use_coder_specialist = False if light_chat_turn else _should_use_coder_specialist(text, analysis, session_id=session_id)
            if use_coder_specialist:
                cleaned = _run_coder_handoff(text, chat_messages, text, session_id=session_id)
                yield _json_line({"type": "delta", "content": cleaned})
                _set_streaming_reply_preview(cleaned, session_id)
                _log_terminal_stream("assistant", cleaned)
            else:
                suppress_intermediate_text = False if light_chat_turn else (
                    _should_force_vscode(text, analysis, session_id=session_id)
                    or _is_codebase_context_request(text, analysis, session_id=session_id)
                )
                working_messages = chat_messages
                if not light_chat_turn and _should_preload_codebase_context(text, analysis, session_id=session_id, use_coder_specialist=use_coder_specialist):
                    _log_terminal_stream("status", "Reading files...")
                    yield _json_line({"type": "status", "label": "Reading files..."})
                    fallback_messages = _build_codebase_fallback_messages(chat_messages, text, session_id=session_id)
                    if fallback_messages:
                        working_messages = fallback_messages

                stream = _request_completion(working_messages, stream=True, **completion_overrides)
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
                    displayed_text = clean_model_text(collapse_hidden_tag_gaps(displayed_text + visible))
                    if not suppress_intermediate_text:
                        _set_streaming_reply_preview(displayed_text, session_id)
                        yield _json_line({"type": "delta", "content": displayed_text})
                        logged_delta = displayed_text[len(last_logged_text):] if displayed_text.startswith(last_logged_text) else displayed_text
                        if logged_delta:
                            print(logged_delta, end="", flush=True)
                            last_logged_text = displayed_text

                visible = hidden_filter.flush()
                if visible:
                    displayed_text = clean_model_text(collapse_hidden_tag_gaps(displayed_text + visible))
                    if not suppress_intermediate_text:
                        _set_streaming_reply_preview(displayed_text, session_id)
                        yield _json_line({"type": "delta", "content": displayed_text})
                        logged_delta = displayed_text[len(last_logged_text):] if displayed_text.startswith(last_logged_text) else displayed_text
                        if logged_delta:
                            print(logged_delta, end="", flush=True)
                            last_logged_text = displayed_text

                raw = "".join(full_response).strip()
                streamed_visible = clean_model_text(displayed_text).strip()
                if not suppress_intermediate_text and streamed_visible:
                    cleaned = streamed_visible
                else:
                    cleaned = _complete_reply(
                        text,
                        chat_messages,
                        working_messages,
                        raw,
                        finish_reason=stream_finish_reason,
                        visible_fallback=displayed_text,
                        session_id=session_id,
                        analysis=analysis,
                        use_coder_specialist=use_coder_specialist,
                    )
                if suppress_intermediate_text or cleaned != displayed_text:
                    _set_streaming_reply_preview(cleaned, session_id)
                    yield _json_line({"type": "delta", "content": cleaned})
                    if last_logged_text:
                        print("", flush=True)
                    _log_terminal_stream("assistant", cleaned)
                elif last_logged_text:
                    print("", flush=True)
            _append_message("assistant", cleaned, session_id)
            if not skip_memory:
                _record_recent_interaction(session_id)
            _log_terminal_stream("done", cleaned)
            yield _json_line({"type": "done", "reply": cleaned, "messages": _snapshot_messages(session_id)})
    except Exception as exc:  # pragma: no cover - surfaced in browser
        _clear_streaming_reply_preview(session_id)
        _log_terminal_stream("error", str(exc))
        yield _json_line({"type": "error", "error": str(exc)})


def _app_state_payload(session_id: str | None = None, *, include_messages: bool = True) -> dict:
    state = _session_state(session_id)
    payload = {
        "model": _model_status(),
        "editor": _editor_snapshot(),
        "version": state.version,
    }
    if include_messages:
        payload["messages"] = _state_messages(session_id)
    return payload


def create_app() -> FastAPI:
    app = FastAPI(title="Akane API", docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    async def root():
        static_response = _static_response_path("/")
        if static_response is None:
            raise HTTPException(status_code=404)
        path, content_type = static_response
        return FileResponse(path, media_type=content_type)

    @app.get("/api/state")
    async def api_state(session_id: str = DEFAULT_SESSION_ID, include_messages: str = "1"):
        return JSONResponse(
            _app_state_payload(
                session_id,
                include_messages=_coerce_bool(include_messages),
            )
        )

    @app.get("/api/editor/state")
    async def api_editor_state():
        return JSONResponse(get_editor_bridge().snapshot())

    @app.get("/api/editor/actions")
    async def api_editor_actions(after: int = 0):
        return JSONResponse({"actions": get_editor_bridge().actions_after(int(after))})

    @app.post("/api/editor/context")
    async def api_editor_context(request: Request):
        payload = await _request_payload(request)
        snapshot = get_editor_bridge().update_context(payload)
        return JSONResponse({"ok": True, "state": snapshot})

    @app.post("/api/editor/action-result")
    async def api_editor_action_result(request: Request):
        payload = await _request_payload(request)
        action_id = int(payload.get("id", 0) or 0)
        if action_id <= 0:
            return JSONResponse({"error": "Missing action id."}, status_code=400)
        summary = get_editor_bridge().complete_action(
            action_id=action_id,
            ok=bool(payload.get("ok", False)),
            result=str(payload.get("result", "")),
            error=str(payload.get("error", "")),
        )
        if summary is None:
            return JSONResponse({"error": "Unknown action id."}, status_code=404)
        return JSONResponse({"ok": True, "summary": summary})

    @app.post("/api/backend")
    async def api_backend(request: Request):
        payload = await _request_payload(request)
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
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, "model": status})

    @app.post("/api/open-vscode")
    async def api_open_vscode():
        try:
            notice = launch_vscode()
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return JSONResponse({"ok": True, "notice": notice, "editor": _editor_snapshot()})

    @app.post("/api/quit")
    async def api_quit():
        def _quit_soon() -> None:
            import time
            time.sleep(0.15)
            os._exit(0)

        threading.Thread(target=_quit_soon, daemon=True, name="AkaneQuit").start()
        return JSONResponse({"ok": True}, status_code=202)

    @app.post("/api/chat")
    async def api_chat(request: Request):
        _start_model_loading()
        chat_request = _parse_chat_request(await _request_payload(request))
        if not chat_request.text:
            return JSONResponse({"error": "Message is empty."}, status_code=400)

        command_result = _handle_command(chat_request.text, chat_request.session_id)
        if command_result is not None:
            command_result.setdefault("messages", _snapshot_messages(chat_request.session_id))
            return JSONResponse(command_result)

        model_status = _model_status()
        if model_status["error"]:
            return JSONResponse({"error": model_status["error"]}, status_code=500)

        try:
            reply = _generate_reply(
                chat_request.text,
                skip_memory=chat_request.skip_memory,
                session_id=chat_request.session_id,
            )
        except Exception as exc:  # pragma: no cover - surfaced in browser
            return JSONResponse({"error": str(exc)}, status_code=500)

        return JSONResponse({"reply": reply, "messages": _snapshot_messages(chat_request.session_id)})

    @app.post("/api/chat/stream")
    async def api_chat_stream(request: Request):
        _start_model_loading()
        chat_request = _parse_chat_request(await _request_payload(request))
        if not chat_request.text:
            return JSONResponse({"error": "Message is empty."}, status_code=400)

        command_result = _handle_command(chat_request.text, chat_request.session_id)
        if command_result is not None:
            command_result.setdefault("messages", _snapshot_messages(chat_request.session_id))
            return JSONResponse(command_result)

        model_status = _model_status()
        if model_status["error"]:
            return JSONResponse({"error": model_status["error"]}, status_code=500)

        return StreamingResponse(
            _stream_chat_events(
                chat_request.text,
                skip_memory=chat_request.skip_memory,
                session_id=chat_request.session_id,
            ),
            media_type="application/x-ndjson; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/{asset_path:path}", include_in_schema=False)
    async def static_assets(asset_path: str):
        route = "/" + asset_path.lstrip("/")
        static_response = _static_response_path(unquote(route))
        if static_response is None:
            raise HTTPException(status_code=404)
        path, content_type = static_response
        return FileResponse(path, media_type=content_type)

    return app


APP = create_app()


class BackgroundUvicornServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._config = uvicorn.Config(
            APP,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(self._config)
        self._thread: threading.Thread | None = None

    def run(self) -> None:
        self._server.run()

    def shutdown(self) -> None:
        self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


def serve(host: str = HOST, port: int = PORT) -> None:
    print(f"Akane web chat running at http://{host}:{port}", flush=True)
    _warm_model_now()
    server = BackgroundUvicornServer(host, port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nStopping Akane web chat...", flush=True)


def serve_in_thread(host: str = HOST, port: int = PORT) -> tuple[BackgroundUvicornServer, threading.Thread]:
    _start_model_loading()
    server = BackgroundUvicornServer(host, port)
    thread = threading.Thread(
        target=server.run,
        daemon=True,
        name="AkaneAPIServer",
    )
    server._thread = thread
    thread.start()
    print(f"Akane background API running at http://{host}:{port}", flush=True)
    return server, thread


if __name__ == "__main__":
    serve()