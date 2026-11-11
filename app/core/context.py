"""Bounded, model-free state selection for one foreground turn."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.config import SETTINGS
from app.core.state import StateSnapshot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.store import Store
from app.core.utils import OWNER_PROFILE_ID, lexical_terms, text_key
from app.integrations.vscode_workspace import current_code_context


MEMORY_LINE_BYTES = 512


def _evidence_excerpt(text, limit):
    text = ' '.join(text.split())
    raw = text.encode('utf-8')
    if len(raw) <= limit:
        return text
    marker = ' [excerpt]'
    return raw[:limit - len(marker)].decode('utf-8', errors='ignore').rstrip() + marker


_CLOCK_TERMS = frozenset({
    "afternoon", "date", "evening", "midnight", "morning", "night", "noon",
    "time", "today", "tomorrow", "tonight", "when", "yesterday",
})
_CODE_TERMS = frozenset({
    "bug", "class", "code", "editor", "error", "file", "function", "line",
    "method", "traceback", "vscode",
})


@dataclass(frozen=True, slots=True)
class TimeContext:
    local_iso: str
    daypart: str
    seconds_since_user_message: float | None
    seconds_since_akane_message: float | None


@dataclass(frozen=True, slots=True)
class TurnContext:
    state: StateSnapshot
    time: TimeContext
    code: str | None = None
    include_time: bool = False
    include_clock_time: bool = False
    include_elapsed_time: bool = False
    environment: str = ""


def _elapsed(now: float, timestamp: float | None) -> float | None:
    if timestamp is None or not math.isfinite(timestamp) or timestamp <= 0 or timestamp > now:
        return None
    return now - timestamp


def _build_time_context(
    *,
    now: float | None = None,
    timezone: str = SETTINGS.timezone,
    last_user_message_at: float | None = None,
    last_akane_message_at: float | None = None,
) -> TimeContext:
    current = time.time() if now is None else float(now)
    local = datetime.fromtimestamp(current, ZoneInfo(timezone))
    hour = local.hour
    daypart = (
        "early morning" if 5 <= hour < 9 else
        "morning" if 9 <= hour < 12 else
        "afternoon" if 12 <= hour < 17 else
        "evening" if 17 <= hour < 22 else
        "late night"
    )
    return TimeContext(
        local.isoformat(timespec="seconds"),
        daypart,
        _elapsed(current, last_user_message_at),
        _elapsed(current, last_akane_message_at),
    )


def _duration(seconds: float) -> str:
    minutes = int(seconds) // 60
    if minutes < 1:
        return "less than a minute"
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''}"


def _format_time_context(
    context: TimeContext,
    *,
    include_clock: bool,
    include_elapsed: bool,
) -> str:
    local = datetime.fromisoformat(context.local_iso)
    lines = []
    if include_clock:
        lines.append(
            f"It is {local.strftime('%A, %B')} {local.day}, {local.year}, "
            f"{local.strftime('%I:%M %p').lstrip('0')} ({context.daypart})."
        )
    if include_elapsed and context.seconds_since_user_message is not None:
        lines.append(f"The user's previous message was {_duration(context.seconds_since_user_message)} ago.")
    if include_elapsed and context.seconds_since_akane_message is not None:
        lines.append(f"Akane last spoke {_duration(context.seconds_since_akane_message)} ago.")
    return " ".join(lines)


def _append_section(
    sections: list[tuple[str, str]],
    name: str,
    heading: str,
    lines: list[str],
    compact: bool,
) -> None:
    text = "\n".join(lines)
    sections.append((name, text if compact else f"{heading}\n{text}"))


def format_context_sections(context: TurnContext, *, compact: bool = False) -> tuple[tuple[str, str], ...]:
    sections = []
    if context.state.memories:
        _append_section(
            sections, "memory", "RELEVANT MEMORY (dated reports)",
            [_evidence_excerpt(
                f"M {item.subject} report ({datetime.fromtimestamp(item.updated_at, ZoneInfo('UTC')).date()}): "
                f"{item.text}", MEMORY_LINE_BYTES)
             for item in context.state.memories], compact,
        )
    if context.code:
        sections.append(("code_context", "EDITOR CONTEXT (read only)\n" + context.code))
    if context.environment:
        sections.append(("environment", "VERIFIED VIRTUAL ENVIRONMENT\n" + context.environment))
    if context.include_time:
        text = _format_time_context(context.time, include_clock=context.include_clock_time,
                                    include_elapsed=context.include_elapsed_time)
        if text:
            sections.append(("time", "TIME\n" + text))
    return tuple(sections)


def format_context(context: TurnContext) -> str:
    return "\n\n".join(text for _, text in format_context_sections(context))


class ContextBuilder:
    def __init__(self, store: Store):
        self.store = store
        self.last_timing = {}

    def build(self, *, profile_id: str, conversation_id: str, message: str,
              reply_context: str = "", allow_tool_context: bool = True,
              now: float | None = None) -> TurnContext:
        started = time.perf_counter()
        state = self.store.conversation_snapshot(profile_id, conversation_id, query=message, now=now)
        last_user = next((t.created_at for t in reversed(state.recent_turns) if t.role == "user"), None)
        last_akane = next((t.created_at for t in reversed(state.recent_turns) if t.role == "assistant"), None)
        terms, normalized = lexical_terms(message), text_key(message)
        include_clock = bool(terms & _CLOCK_TERMS)
        include_elapsed = any(phrase in normalized for phrase in (
            "been a while", "how long", "last spoke", "last talked",
        ))
        code = None
        if allow_tool_context and profile_id == OWNER_PROFILE_ID and (
            terms & _CODE_TERMS or reply_context
            or normalized.startswith(("look at this", "what is wrong with this", "why is this"))
        ):
            code = current_code_context()
        self.last_timing = {**self.store.snapshot_timing(),
                            "context_build_seconds": time.perf_counter() - started}
        return TurnContext(
            state, _build_time_context(now=now, last_user_message_at=last_user,
                                       last_akane_message_at=last_akane), code,
            include_clock or include_elapsed, include_clock, include_elapsed,
            self.store.environment_context(profile_id, message),
        )
