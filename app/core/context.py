"""Bounded, model-free state selection for one foreground turn."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.config import SETTINGS
from app.core.mind import (
    behavioral_tendency_state,
    curiosity_state,
    developmental_goal_state,
    self_development_state,
    strategy_state,
)
from app.core.state import StateSnapshot
from app.core.store import Store
from app.core.utils import lexical_terms, text_key
from app.integrations.vscode_context import CodeContext, current_code_context


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
    code: CodeContext | None = None
    include_time: bool = False
    include_clock_time: bool = False
    include_elapsed_time: bool = False
    include_self_history: bool = False


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


def format_context_sections(
    context: TurnContext,
    *,
    compact: bool = False,
) -> tuple[tuple[str, str], ...]:
    state = context.state
    sections: list[tuple[str, str]] = []
    if state.self_items:
        history = {row.self_item_id: row for row in reversed(state.self_revisions)}
        lines = []
        for item in state.self_items:
            line = (
                f"S {item.kind} {item.topic}: {item.value} "
                f"| {self_development_state(item)}"
            )
            previous = history.get(item.id) if context.include_self_history else None
            if previous is not None:
                line += f"; previously: {previous.value}"
            lines.append(line)
        _append_section(sections, "self", "AKANE SELF", lines, compact)
    if state.behavioral_tendencies:
        lines = [
            f"B {item.context}: {item.behavior} -> {item.expected_effect} "
            f"| {behavioral_tendency_state(item)}"
            for item in state.behavioral_tendencies
        ]
        _append_section(
            sections, "behavioral_tendency", "LEARNED BEHAVIOR", lines, compact,
        )
    if state.strategies:
        lines = [
            f"R {item.context}: {item.procedure} | {strategy_state(item)}"
            for item in state.strategies
        ]
        _append_section(
            sections, "strategy", "REUSABLE STRATEGY", lines, compact,
        )
    if state.curiosities:
        lines = [
            f"A {item.topic}: {item.focus} | {curiosity_state(item)}"
            for item in state.curiosities
        ]
        _append_section(
            sections, "curiosity", "DEVELOPMENTAL ATTENTION", lines, compact,
        )
    if state.developmental_goals:
        lines = [
            f"G {item.topic}: {item.goal} | {developmental_goal_state(item)}"
            for item in state.developmental_goals
        ]
        _append_section(
            sections, "developmental_goal", "DEVELOPMENTAL GOAL", lines, compact,
        )
    if state.experiences:
        lines = []
        for item in state.experiences:
            line = (
                f"E {item.kind} {item.topic}: {item.what_happened} "
                f"| Akane: {item.akane_response}"
            )
            if item.outcome:
                line += f" | Outcome: {item.outcome}"
            lines.append(line)
        _append_section(
            sections, "experience", "RELEVANT EXPERIENCE", lines, compact,
        )
    if state.memories:
        _append_section(
            sections,
            "memory",
            "RELEVANT MEMORY",
            [f"M {item.text}" for item in state.memories],
            compact,
        )
    if context.code and context.code.prompt_text:
        text = "C " + context.code.prompt_text
        sections.append(("code_context", text if compact else "EDITOR CONTEXT\n" + text))
    if context.include_time:
        text = "T " + _format_time_context(
            context.time,
            include_clock=context.include_clock_time,
            include_elapsed=context.include_elapsed_time,
        )
        if text.strip() != "T":
            sections.append(("time", text if compact else "TIME\n" + text))
    return tuple(sections)


def format_context(context: TurnContext) -> str:
    return "\n\n".join(text for _, text in format_context_sections(context))


class ContextBuilder:
    def __init__(self, store: Store):
        self.store = store
        self.last_timing: dict[str, float | int] = {}

    def build(
        self,
        *,
        profile_id: str,
        conversation_id: str,
        message: str,
        reply_context: str = "",
        allow_tool_context: bool = True,
        now: float | None = None,
    ) -> TurnContext:
        started_at = time.perf_counter()
        query = " ".join(part for part in (message, reply_context) if part).strip()
        snapshot_started_at = time.perf_counter()
        state = self.store.snapshot(profile_id, conversation_id, query=query, now=now)
        snapshot_finished_at = time.perf_counter()
        snapshot_timing = self.store.snapshot_timing()
        last_user = next(
            (turn.created_at for turn in reversed(state.recent_turns) if turn.role == "user"), None,
        )
        last_akane = next(
            (turn.created_at for turn in reversed(state.recent_turns) if turn.role == "assistant"), None,
        )
        time_context = _build_time_context(
            now=now,
            last_user_message_at=last_user,
            last_akane_message_at=last_akane,
        )
        terms = lexical_terms(query)
        normalized = text_key(query)
        include_clock = bool(terms & _CLOCK_TERMS)
        include_elapsed = any(phrase in normalized for phrase in (
            "been a while", "how long", "last spoke", "last talked",
        ))
        code = None
        if allow_tool_context and (
            bool(terms & _CODE_TERMS)
            or bool(reply_context)
            or normalized.startswith(("look at this", "what is wrong with this", "why is this"))
        ):
            candidate = current_code_context()
            code = candidate if candidate.connected else None
        result = TurnContext(
            state=state,
            time=time_context,
            code=code,
            include_time=include_clock or include_elapsed,
            include_clock_time=include_clock,
            include_elapsed_time=include_elapsed,
            include_self_history=bool(terms & {"before", "change", "changed", "mind", "used"}),
        )
        self.last_timing = {
            "store_snapshot_seconds": snapshot_finished_at - snapshot_started_at,
            "relevance_selection_seconds": snapshot_timing.get("selection_seconds", 0.0),
            "context_build_seconds": time.perf_counter() - started_at,
            "memory_candidates": snapshot_timing.get("memory_candidates", 0),
            "self_candidates": snapshot_timing.get("self_candidates", 0),
            "experience_candidates": snapshot_timing.get("experience_candidates", 0),
            "behavioral_tendency_candidates": snapshot_timing.get(
                "behavioral_tendency_candidates", 0,
            ),
            "strategy_candidates": snapshot_timing.get("strategy_candidates", 0),
            "curiosity_candidates": snapshot_timing.get("curiosity_candidates", 0),
            "developmental_goal_candidates": snapshot_timing.get(
                "developmental_goal_candidates", 0,
            ),
        }
        return result
