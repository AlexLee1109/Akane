"""Fresh, read-only conversational time context."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.config import TIMEZONE


@dataclass(frozen=True, slots=True)
class TimeContext:
    local_iso: str
    local_date: str
    local_time: str
    weekday: str
    daypart: str
    seconds_since_user_message: float | None
    seconds_since_akane_message: float | None
    seconds_in_current_activity: float | None


def _daypart(hour: int) -> str:
    if 5 <= hour < 9:
        return "early morning"
    if 9 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "late night"


def _elapsed(now: float, timestamp: float | None) -> float | None:
    if timestamp is None:
        return None
    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0 or value > now:
        return None
    return now - value


def build_time_context(
    *,
    now: float | None = None,
    timezone: str = TIMEZONE,
    last_user_message_at: float | None = None,
    last_akane_message_at: float | None = None,
    current_activity_started_at: float | None = None,
) -> TimeContext:
    """Build current local and elapsed values without reading or writing state."""

    current = time.time() if now is None else float(now)
    if not math.isfinite(current) or current < 0.0:
        raise ValueError("Current time must be a finite non-negative timestamp.")
    local = datetime.fromtimestamp(current, ZoneInfo(timezone))
    return TimeContext(
        local_iso=local.isoformat(timespec="seconds"),
        local_date=local.date().isoformat(),
        local_time=local.strftime("%H:%M:%S"),
        weekday=local.strftime("%A"),
        daypart=_daypart(local.hour),
        seconds_since_user_message=_elapsed(current, last_user_message_at),
        seconds_since_akane_message=_elapsed(current, last_akane_message_at),
        seconds_in_current_activity=_elapsed(current, current_activity_started_at),
    )


def _duration(seconds: float) -> str:
    total_minutes = int(seconds) // 60
    if total_minutes < 1:
        return "less than a minute"
    units = (
        ("day", total_minutes // (24 * 60)),
        ("hour", (total_minutes % (24 * 60)) // 60),
        ("minute", total_minutes % 60),
    )
    parts = [
        f"{value} {name}{'' if value == 1 else 's'}"
        for name, value in units
        if value
    ][:2]
    return " and ".join(parts)


def format_time_context(context: TimeContext) -> str:
    """Format prompt-safe facts without raw timestamps or emotional inference."""

    local = datetime.fromisoformat(context.local_iso)
    clock = local.strftime("%I:%M %p").lstrip("0")
    lines = [
        f"Local date: {context.weekday}, {local.strftime('%B')} "
        f"{local.day}, {local.year}.",
        f"Local time: {context.weekday} {context.daypart}, {clock}.",
    ]
    if context.seconds_since_user_message is not None:
        lines.append(
            "Time since Arcane's last message: "
            f"{_duration(context.seconds_since_user_message)}."
        )
    if context.seconds_since_akane_message is not None:
        lines.append(
            "Time since Akane last spoke: "
            f"{_duration(context.seconds_since_akane_message)}."
        )
    if context.seconds_in_current_activity is not None:
        lines.append(
            "Akane has been in her current activity for "
            f"{_duration(context.seconds_in_current_activity)}."
        )
    return "\n".join(lines)
