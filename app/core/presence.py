"""Persisted offscreen presence; activity choices belong to Akane's life turn."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace

from app.core.signal import EmotionState
from app.core.utils import compact_text

_MAX_ACTIVITY_CHARS = 160
_MAX_DETAIL_CHARS = 240
_MAX_DURATION_MINUTES = 120
_MAX_START_AFTER_MINUTES = 7 * 24 * 60
_LIFE_DECISIONS = {"start_activity", "schedule_activity", "continue_activity", "quiet_downtime", "do_nothing"}
_ACTIVITY_KEY_STOPWORDS = {
    "a", "about", "an", "and", "at", "her", "in", "of", "on", "some",
    "something", "the", "to", "with",
}


def _timestamp(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and number >= 0 else default


def _activity_text(value: object) -> str:
    activity = compact_text(value, _MAX_ACTIVITY_CHARS)
    words = re.findall(r"[a-z]+", activity.casefold())
    if not 1 <= len(words) <= 16:
        return ""
    return activity


def activity_key(value: object) -> str:
    """Return compact comparison metadata, not a displayable activity history."""

    tokens = []
    for token in re.findall(
        r"[a-z0-9]+",
        compact_text(value, _MAX_ACTIVITY_CHARS).casefold(),
    ):
        if token in _ACTIVITY_KEY_STOPWORDS:
            continue
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        if len(token) > 4 and token.endswith("e"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)[:96]


@dataclass(frozen=True, slots=True)
class ActivityPattern:
    previous_key: str = ""
    prior_key: str = ""
    repeat_count: int = 0

    @classmethod
    def from_dict(cls, payload: object) -> "ActivityPattern":
        values = payload if isinstance(payload, dict) else {}
        try:
            repeat_count = max(0, min(100, int(values.get("repeat_count", 0))))
        except (TypeError, ValueError):
            repeat_count = 0
        return cls(
            activity_key(values.get("previous_key")),
            activity_key(values.get("prior_key")),
            repeat_count,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "previous_key": self.previous_key,
            "prior_key": self.prior_key,
            "repeat_count": self.repeat_count,
        }


@dataclass(frozen=True, slots=True)
class PresenceActivity:
    activity: str
    detail: str | None
    started_at: float
    ends_at: float
    source: str = ""
    category: str = ""
    title: str | None = None

    @classmethod
    def from_dict(cls, payload: object) -> "PresenceActivity | None":
        if not isinstance(payload, dict):
            return None
        activity = compact_text(payload.get("activity"), _MAX_ACTIVITY_CHARS)
        detail_value = payload.get("detail")
        detail = compact_text(detail_value, _MAX_DETAIL_CHARS) if isinstance(detail_value, str) else None
        started_at, ends_at = _timestamp(payload.get("started_at")), _timestamp(payload.get("ends_at"))
        if not activity or (detail_value is not None and not detail) or ends_at <= started_at:
            return None
        ends_at = min(ends_at, started_at + _MAX_DURATION_MINUTES * 60)
        title_value = payload.get("title")
        title = compact_text(title_value, 160) if isinstance(title_value, str) else None
        return cls(
            activity,
            detail,
            started_at,
            ends_at,
            compact_text(payload.get("source"), 48),
            compact_text(payload.get("category"), 48),
            title,
        )

    def as_dict(self) -> dict[str, object]:
        return {"activity": self.activity, "category": self.category or None, "title": self.title, "detail": self.detail, "started_at": self.started_at, "ends_at": self.ends_at, "source": self.source}

    def fact(self) -> str:
        return f"{self.activity} — {self.detail}" if self.detail else self.activity


@dataclass(frozen=True, slots=True)
class PresenceState:
    current_activity: PresenceActivity | None = None
    previous_activity: PresenceActivity | None = None
    next_activity: PresenceActivity | None = None
    life_pending: bool = False
    life_reason: str = ""
    life_claimed_at: float = 0.0
    life_last_run_at: float = 0.0
    life_next_run_at: float = 0.0
    activity_pattern: ActivityPattern = ActivityPattern()

    @classmethod
    def from_dict(cls, payload: object) -> "PresenceState":
        values = payload if isinstance(payload, dict) else {}
        previous = PresenceActivity.from_dict(values.get("previous_activity"))
        if previous is None:
            raw_recent = (
                values.get("recent_activities")
                if isinstance(values.get("recent_activities"), (list, tuple))
                else ()
            )
            migrated = tuple(
                (
                    activity,
                    _timestamp(item.get("ends_at"), _timestamp(item.get("started_at"))),
                    _timestamp(item.get("started_at")),
                )
                for item in raw_recent
                if isinstance(item, dict)
                and (activity := PresenceActivity.from_dict(item)) is not None
            )
            if migrated:
                previous = max(
                    migrated,
                    key=lambda item: (item[1], item[2]),
                )[0]
        current = PresenceActivity.from_dict(values.get("current_activity"))
        scheduled = PresenceActivity.from_dict(values.get("next_activity"))
        next_run = _timestamp(values.get("life_next_run_at"))
        if not next_run:
            next_run = _next_lifecycle_at(current, scheduled)
        pattern = ActivityPattern.from_dict(values.get("activity_pattern"))
        if not pattern.previous_key and (current is not None or previous is not None):
            latest = current if current is not None else previous
            assert latest is not None
            pattern = ActivityPattern(
                previous_key=activity_key(latest.activity),
                prior_key=activity_key(previous.activity) if current and previous else "",
                repeat_count=1,
            )
        return cls(
            current,
            previous,
            scheduled,
            bool(values.get("life_pending")),
            compact_text(values.get("life_reason"), 80),
            _timestamp(values.get("life_claimed_at")),
            _timestamp(values.get("life_last_run_at")),
            next_run,
            pattern,
        )
    def as_dict(self) -> dict[str, object]:
        return {
            "current_activity": self.current_activity.as_dict() if self.current_activity else None,
            "previous_activity": self.previous_activity.as_dict() if self.previous_activity else None,
            "next_activity": self.next_activity.as_dict() if self.next_activity else None,
            "life_pending": self.life_pending,
            "life_reason": self.life_reason,
            "life_claimed_at": self.life_claimed_at,
            "life_last_run_at": self.life_last_run_at,
            "life_next_run_at": self.life_next_run_at,
            "activity_pattern": self.activity_pattern.as_dict(),
        }


def activity_continuity(
    current: PresenceActivity | None,
    last_assistant_at: float | None,
) -> str:
    """Classify an active activity against committed assistant history."""

    if current is None:
        return "none"
    if last_assistant_at is None:
        return "new"
    if current.started_at > last_assistant_at:
        return "new"
    return "ongoing"


def _next_lifecycle_at(
    current: PresenceActivity | None,
    scheduled: PresenceActivity | None,
) -> float:
    deadlines = []
    if current is not None:
        deadlines.append(current.ends_at)
    if scheduled is not None:
        deadlines.append(scheduled.started_at)
    return min(deadlines, default=0.0)


def request_life_decision(state: PresenceState, *, reason: str) -> PresenceState:
    if state.life_pending:
        return state
    return replace(state, life_pending=True, life_reason=compact_text(reason, 80), life_claimed_at=0.0)


def advance_presence(state: PresenceState, *, now: float) -> PresenceState:
    current = state.current_activity
    previous = state.previous_activity
    scheduled = state.next_activity
    reason = ""
    if current is not None and current.ends_at <= now:
        previous, current, reason = current, None, "activity_expired"
    if scheduled is not None and scheduled.started_at <= now:
        if current is not None:
            replacement_end = max(current.started_at + 1.0, scheduled.started_at)
            previous = replace(current, ends_at=replacement_end)
        if scheduled.ends_at <= now:
            previous, current, reason = scheduled, None, "scheduled_expired"
        else:
            current, reason = scheduled, "scheduled_due"
        scheduled = None
    advanced = replace(
        state,
        current_activity=current,
        previous_activity=previous,
        next_activity=scheduled,
        life_next_run_at=_next_lifecycle_at(current, scheduled),
    )
    if reason:
        if reason in {"activity_expired", "scheduled_expired"}:
            return replace(
                advanced,
                life_pending=True,
                life_reason=reason,
                life_claimed_at=0.0,
            )
        return advanced
    if advanced.current_activity is None and advanced.next_activity is None:
        return request_life_decision(advanced, reason="no_activity")
    return advanced


@dataclass(frozen=True, slots=True)
class LifeDecision:
    decision: str
    activity: str = ""
    category: str = ""
    subject: str | None = None
    detail: str | None = None
    duration_minutes: int = 45
    start_after_minutes: int = 0
    reason: str = ""
    interest_addition: str | None = None


def _unsupported_proper_nouns(value: str, grounded_context: str) -> bool:
    grounded = grounded_context.casefold()
    words = re.finditer(r"\b[A-Z][A-Za-z0-9'-]*\b", value)
    return any(
        match.group(0) != "I"
        and not (
            match.start() == 0
            and match.group(0).casefold().endswith(("ing", "ed"))
        )
        and match.group(0).casefold() not in grounded
        for match in words
    )


def parse_life_decision(
    output: object,
    *,
    grounded_context: str = "",
) -> LifeDecision | None:
    text = str(output or "")
    matches = re.findall(r"<AKANE_LIFE>\s*(.*?)\s*</AKANE_LIFE>", text, re.DOTALL)
    if len(matches) != 1:
        return None
    try:
        payload = json.loads(matches[0])
    except (TypeError, ValueError):
        return None
    required = {"decision", "activity", "subject", "detail", "duration_minutes", "reason", "interest_addition"}
    allowed = required | {"category", "start_after_minutes"}
    if not isinstance(payload, dict) or not required <= set(payload) <= allowed:
        return None
    decision = compact_text(payload.get("decision"), 32).casefold()
    activity = _activity_text(payload.get("activity"))
    category = compact_text(payload.get("category"), 48).casefold()
    detail = compact_text(payload.get("detail"), _MAX_DETAIL_CHARS) if isinstance(payload.get("detail"), str) else None
    subject = compact_text(payload.get("subject"), 160) if isinstance(payload.get("subject"), str) else None
    reason = compact_text(payload.get("reason"), 160)
    addition = compact_text(payload.get("interest_addition"), 100) if isinstance(payload.get("interest_addition"), str) else None
    duration, delay = payload.get("duration_minutes"), payload.get("start_after_minutes", 0)
    if decision not in _LIFE_DECISIONS or type(duration) is not int or type(delay) is not int or not reason:
        return None
    if decision in {"start_activity", "schedule_activity"} and not activity:
        return None
    if decision == "quiet_downtime":
        activity, detail = "quiet downtime", detail
    grounded_text = " ".join(value for value in (activity, subject or "", detail or "") if value)
    if (
        _unsupported_proper_nouns(grounded_text, grounded_context)
        or (
            subject is not None
            and subject.casefold() not in grounded_context.casefold()
        )
    ):
        return None
    return LifeDecision(
        decision,
        activity,
        category,
        subject,
        detail,
        max(15, min(_MAX_DURATION_MINUTES, duration)),
        max(0, min(_MAX_START_AFTER_MINUTES, delay)),
        reason,
        addition,
    )


def life_decision_rejection(state: PresenceState, decision: LifeDecision) -> str:
    advanced = advance_presence(state, now=max(state.life_claimed_at, state.life_last_run_at))
    current = advanced.current_activity
    if decision.decision == "continue_activity":
        if current is None:
            return "no current activity to continue"
        if len(decision.reason.split()) < 3:
            return "continuation reason is not meaningful"
        return ""
    if decision.decision in {"do_nothing"}:
        return ""
    if current is not None:
        return "current activity is still active"
    key = activity_key(decision.activity)
    pattern = advanced.activity_pattern
    if key and key == pattern.previous_key:
        return "proposal repeats the immediately previous activity"
    if key and key == pattern.prior_key and pattern.repeat_count >= 2:
        return "proposal would complete a two-activity loop"
    return ""


def _updated_pattern(pattern: ActivityPattern, activity: str) -> ActivityPattern:
    key = activity_key(activity)
    if not key:
        return pattern
    repeating = key == pattern.previous_key or (
        bool(pattern.prior_key) and key == pattern.prior_key
    )
    return ActivityPattern(
        previous_key=key,
        prior_key=pattern.previous_key,
        repeat_count=pattern.repeat_count + 1 if repeating else 1,
    )


def apply_life_decision(state: PresenceState, decision: LifeDecision, *, now: float) -> PresenceState:
    advanced = advance_presence(state, now=now)
    current = advanced.current_activity
    previous = advanced.previous_activity
    scheduled = advanced.next_activity
    pattern = advanced.activity_pattern
    if decision.decision in {"start_activity", "quiet_downtime"}:
        if current is not None:
            previous = current
        current = PresenceActivity(decision.activity, decision.detail, now, now + decision.duration_minutes * 60, "autonomous_life", decision.category, decision.subject)
        pattern = _updated_pattern(pattern, decision.activity)
    elif decision.decision == "schedule_activity":
        scheduled = PresenceActivity(decision.activity, decision.detail, now + decision.start_after_minutes * 60, now + (decision.start_after_minutes + decision.duration_minutes) * 60, "autonomous_life", decision.category, decision.subject)
        pattern = _updated_pattern(pattern, decision.activity)
    elif decision.decision == "continue_activity" and current is not None:
        current = replace(current, ends_at=max(current.ends_at, now + decision.duration_minutes * 60))
    return replace(
        advanced,
        current_activity=current,
        previous_activity=previous,
        next_activity=scheduled,
        life_pending=False,
        life_reason="",
        life_claimed_at=0.0,
        life_last_run_at=now,
        life_next_run_at=_next_lifecycle_at(current, scheduled),
        activity_pattern=pattern,
    )


def apply_activity_updates(state: PresenceState, *, activity_update: dict[str, object] | None, next_activity: dict[str, object] | None, now: float) -> PresenceState:
    advanced = advance_presence(state, now=now)
    current, scheduled = advanced.current_activity, advanced.next_activity
    pattern = advanced.activity_pattern
    if activity_update is not None and current is None:
        current = PresenceActivity(str(activity_update["activity"]), activity_update["detail"], now, now + int(activity_update["duration_minutes"]) * 60, "gemma")
        pattern = _updated_pattern(pattern, current.activity)
    if next_activity is not None:
        scheduled = PresenceActivity(str(next_activity["activity"]), None, now + int(next_activity["start_after_minutes"]) * 60, now + (int(next_activity["start_after_minutes"]) + int(next_activity["duration_minutes"])) * 60, "gemma")
        pattern = _updated_pattern(pattern, scheduled.activity)
    return replace(
        advanced,
        current_activity=current,
        next_activity=scheduled,
        life_next_run_at=_next_lifecycle_at(current, scheduled),
        activity_pattern=pattern,
    )


def validate_activity_update(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict) or set(payload) != {"activity", "detail", "duration_minutes"}:
        return None
    activity, duration = _activity_text(payload.get("activity")), payload.get("duration_minutes")
    detail = compact_text(payload.get("detail"), _MAX_DETAIL_CHARS) if isinstance(payload.get("detail"), str) else None
    grounding_text = " ".join(value for value in (activity, detail or "") if value)
    if (
        not activity
        or (payload.get("detail") is not None and not detail)
        or type(duration) is not int
        or _unsupported_proper_nouns(grounding_text, "")
    ):
        return None
    return {"activity": activity, "detail": detail, "duration_minutes": max(15, min(_MAX_DURATION_MINUTES, duration))}


def validate_next_activity(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict) or set(payload) != {"activity", "start_after_minutes", "duration_minutes"}:
        return None
    activity, delay, duration = _activity_text(payload.get("activity")), payload.get("start_after_minutes"), payload.get("duration_minutes")
    if (
        not activity
        or type(delay) is not int
        or type(duration) is not int
        or _unsupported_proper_nouns(activity, "")
    ):
        return None
    return {"activity": activity, "start_after_minutes": max(0, min(_MAX_START_AFTER_MINUTES, delay)), "duration_minutes": max(15, min(_MAX_DURATION_MINUTES, duration))}


def format_presence_context(
    state: PresenceState,
    *,
    interests: tuple[str, ...],
    emotion: EmotionState,
    timeframe: str,
    now: float,
    continuity: str = "none",
    local_time: str = "",
    daypart: str = "",
) -> str:
    del interests, emotion, timeframe
    if state.current_activity is None:
        parts = ["Current activity: none.", "Activity continuity: none."]
        if state.previous_activity is not None:
            parts.append(f"Previous activity: {state.previous_activity.fact()}.")
        if state.life_pending:
            parts.append("An autonomous life decision is pending.")
    else:
        activity = state.current_activity
        parts = [
            f"Current activity: {activity.activity}.",
            f"Activity continuity: {continuity}.",
            f"Activity category: {activity.category or 'unavailable'}.",
            f"Activity title: {activity.title or 'unavailable'}.",
            f"Activity detail: {activity.detail or 'none recorded'}.",
            f"Started {max(0, int((now - activity.started_at) // 60))} minutes ago.",
        ]
    if local_time:
        parts.append(f"Current local time: {local_time}.")
    if daypart:
        parts.append(f"Current daypart: {daypart}.")
    parts.extend(
        (
            "Use temporal wording consistent with the supplied activity continuity. "
            "Do not describe a new activity as continuing from an earlier conversation.",
            "Do not add activity details that are not recorded.",
            "Do not repeat internal missing-value markers unless they are directly relevant.",
        )
    )
    return "Presence facts:\n" + "\n".join(parts)
