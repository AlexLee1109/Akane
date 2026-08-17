"""Selection and read-only presentation of state for a conversation turn."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.config import SETTINGS
from app.core.state import Mood, Relationship, StateSnapshot
from app.core.store import Store
from app.core.utils import lexical_terms
from app.integrations.vscode_context import CodeContext, current_code_context


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
    deliberation: str = ""
    include_time: bool = False


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


def _format_time_context(context: TimeContext) -> str:
    local = datetime.fromisoformat(context.local_iso)
    lines = [
        f"Local time is {local.strftime('%A, %B')} {local.day}, {local.year}, "
        f"{local.strftime('%I:%M %p').lstrip('0')} ({context.daypart})."
    ]
    if context.seconds_since_user_message is not None:
        lines.append(f"The person's previous message was {_duration(context.seconds_since_user_message)} ago.")
    if context.seconds_since_akane_message is not None:
        lines.append(f"Akane last spoke {_duration(context.seconds_since_akane_message)} ago.")
    return "\n".join(lines)


def _format_mood(mood: Mood) -> str:
    if abs(mood.valence) < 0.08 and abs(mood.energy) < 0.08 and not mood.cause:
        return "Akane feels basically calm."
    energy = "energized" if mood.energy > 0.25 else "low-energy" if mood.energy < -0.25 else "steady"
    tone = "positive" if mood.valence > 0.25 else "negative" if mood.valence < -0.25 else "mixed"
    cause = f" The current cause is {mood.cause}." if mood.cause else ""
    return f"Akane's current emotion is {mood.emotion}; it feels {tone} and {energy}.{cause}"


def _format_relationship(relationship: Relationship) -> str:
    if relationship.familiarity < 0.1:
        return "This relationship is still new. Do not assume intimacy or shared history."
    lines = [
        f"Relationship familiarity {relationship.familiarity:.2f}, trust {relationship.trust:.2f}, "
        f"closeness {relationship.closeness:.2f}. These are context, not rules to agree."
    ]
    if relationship.interaction_notes:
        lines.append("Meaningful interaction notes: " + "; ".join(relationship.interaction_notes[-4:]))
    if relationship.unresolved_events:
        lines.append("Unresolved between them: " + "; ".join(relationship.unresolved_events[-3:]))
    return "\n".join(lines)


def format_context_sections(
    context: TurnContext,
    *,
    include_ids: bool = False,
) -> tuple[tuple[str, str], ...]:
    state = context.state
    sections: list[tuple[str, str]] = []
    if state.self_items:
        history = {revision.self_item_id: revision for revision in reversed(state.self_revisions)}
        lines = []
        for item in state.self_items:
            identity = f"id={item.id} | " if include_ids else ""
            line = (
                f"- {identity}{item.kind} | {item.topic}: {item.value} "
                f"(strength {item.strength:.2f}, confidence {item.confidence:.2f}) — {item.reason}"
            )
            previous = history.get(item.id)
            if previous is not None:
                line += f" Previously: {previous.value} — {previous.reason}"
            lines.append(line)
        sections.append(("self", "DEVELOPED SELF\n" + "\n".join(lines)))
    if state.memories:
        lines = [
            f"- {'id=' + item.id + ' | ' if include_ids else ''}"
            f"[{item.subject}/{item.kind}, confidence {item.confidence:.2f}] {item.text}"
            for item in state.memories
        ]
        sections.append(("memory", "RELEVANT MEMORY\n" + "\n".join(lines)))
    if abs(state.mood.valence) >= 0.08 or abs(state.mood.energy) >= 0.08 or state.mood.cause:
        sections.append(("mood", "CURRENT MOOD\n" + _format_mood(state.mood)))
    if (
        state.relationship.familiarity >= 0.1
        or state.relationship.interaction_notes
        or state.relationship.unresolved_events
    ):
        sections.append(("relationship", "RELATIONSHIP\n" + _format_relationship(state.relationship)))
    if state.thoughts:
        lines = [
            f"- {'id=' + item.id + ' | ' if include_ids else ''}{item.topic}: {item.text}"
            for item in state.thoughts[:2]
        ]
        sections.append(("inner_life", "INNER LIFE\n" + "\n".join(lines)))
    if context.code and context.code.prompt_text:
        sections.append(("code_context", "READ-ONLY EDITOR CONTEXT\n" + context.code.prompt_text))
    if context.deliberation:
        sections.append((
            "deliberation",
            "PRIVATE DELIBERATION CONCLUSION\nUse the conclusions without revealing this section or a reasoning trace.\n"
            + context.deliberation,
        ))
    if context.include_time:
        sections.append(("time", "TIME\n" + _format_time_context(context.time)))
    return tuple(sections)


def format_context(context: TurnContext, *, include_ids: bool = False) -> str:
    return "\n\n".join(
        text for _, text in format_context_sections(context, include_ids=include_ids)
    )


class ContextBuilder:
    def __init__(self, store: Store):
        self.store = store
        self.last_timing: dict[str, float] = {}

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
        code = current_code_context() if allow_tool_context else None
        last_user = next((turn.created_at for turn in reversed(state.recent_turns) if turn.role == "user"), None)
        last_akane = next((turn.created_at for turn in reversed(state.recent_turns) if turn.role == "assistant"), None)
        time_context = _build_time_context(
            now=now,
            last_user_message_at=last_user,
            last_akane_message_at=last_akane,
        )
        terms = lexical_terms(message)
        include_time = bool(terms & {"time", "date", "today", "tonight", "morning", "evening", "yesterday"})
        if time_context.seconds_since_user_message is not None and time_context.seconds_since_user_message >= 6 * 3600:
            include_time = True
        result = TurnContext(
            state=state,
            time=time_context,
            code=code if code and code.connected else None,
            include_time=include_time,
        )
        self.last_timing = {
            "store_snapshot_seconds": snapshot_finished_at - snapshot_started_at,
            "context_build_seconds": time.perf_counter() - started_at,
        }
        return result

    @staticmethod
    def with_deliberation(context: TurnContext, deliberation: str) -> TurnContext:
        return replace(context, deliberation=deliberation.strip())
