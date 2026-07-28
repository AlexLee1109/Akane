"""Canonical schema and validation for Akane's persisted offscreen life."""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from app.core.utils import compact_text

MIN_DECISION_MINUTES = 150
MAX_DECISION_MINUTES = 210
CLAIM_SECONDS = 10 * 60
RETRY_SECONDS = 5 * 60

_MAX_ACTIVITY_CHARS = 160
_MAX_DETAIL_CHARS = 240
_LIFE_MODES = {"new", "continue"}
_ACTIVITY_KEY_STOPWORDS = {
    "a",
    "about",
    "after",
    "an",
    "and",
    "as",
    "at",
    "because",
    "by",
    "for",
    "from",
    "her",
    "herself",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "some",
    "something",
    "the",
    "that",
    "this",
    "to",
    "up",
    "while",
    "with",
}


def _timestamp(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and number >= 0.0 else default


def _optional_text(value: object, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return compact_text(value, limit) or None


def _activity_text(value: object) -> str:
    activity = compact_text(value, _MAX_ACTIVITY_CHARS)
    words = re.findall(r"[a-z0-9]+", activity.casefold())
    return activity if 1 <= len(words) <= 24 else ""


def minutes_to_seconds(minutes: float) -> float:
    return float(minutes) * 60.0


def next_decision_time(now: float) -> float:
    minutes = random.uniform(MIN_DECISION_MINUTES, MAX_DECISION_MINUTES)
    return max(0.0, float(now)) + minutes_to_seconds(minutes)


def activity_key(value: object) -> str:
    """Create small comparison metadata without retaining activity history."""

    tokens: list[str] = []
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


def _near_key(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    shared = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return (
        bool(shared)
        and len(shared) / max(1, len(union)) >= 0.6
    ) or SequenceMatcher(None, left, right).ratio() >= 0.86


def _activity_semantic_key(
    activity: object,
    subject: object = None,
) -> str:
    return activity_key(" ".join(value for value in (
        compact_text(activity, _MAX_ACTIVITY_CHARS),
        compact_text(subject, 160),
    ) if value))


def _record_key(activity: "PresenceActivity | None") -> str:
    if activity is None:
        return ""
    return _activity_semantic_key(activity.activity, activity.subject)


def _meaningful_detail(detail: str | None, *, activity: str) -> bool:
    detail_tokens = set(activity_key(detail).split())
    activity_tokens = set(activity_key(activity).split())
    return len(detail_tokens) >= 3 and len(detail_tokens - activity_tokens) >= 2


@dataclass(frozen=True, slots=True)
class ActivityPattern:
    current_key: str = ""
    previous_key: str = ""
    repeat_count: int = 0

    @classmethod
    def from_dict(cls, payload: object) -> "ActivityPattern":
        values = payload if isinstance(payload, dict) else {}
        try:
            repeat_count = max(0, min(100, int(values.get("repeat_count", 0))))
        except (TypeError, ValueError):
            repeat_count = 0
        return cls(
            current_key=activity_key(
                values.get("current_key") or values.get("previous_key")
            ),
            previous_key=activity_key(
                values.get("previous_key")
                if "current_key" in values
                else values.get("prior_key")
            ),
            repeat_count=repeat_count,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "current_key": self.current_key,
            "previous_key": self.previous_key,
            "repeat_count": self.repeat_count,
        }


@dataclass(frozen=True, slots=True)
class PresenceActivity:
    activity: str
    category: str | None
    subject: str | None
    detail: str | None
    started_at: float
    expected_end_at: float
    source: str = "autonomous_life"

    @classmethod
    def from_dict(cls, payload: object) -> "PresenceActivity | None":
        if not isinstance(payload, dict):
            return None
        activity = _activity_text(payload.get("activity"))
        started_at = _timestamp(payload.get("started_at"))
        expected_end_at = _timestamp(
            payload.get("expected_end_at"),
            _timestamp(payload.get("ends_at")),
        )
        if not activity or expected_end_at <= started_at:
            return None
        detail_value = payload.get("detail")
        detail = _optional_text(detail_value, _MAX_DETAIL_CHARS)
        if detail_value is not None and detail is None:
            return None
        return cls(
            activity=activity,
            category=_optional_text(payload.get("category"), 48),
            subject=_optional_text(
                payload.get("subject", payload.get("title")),
                160,
            ),
            detail=detail,
            started_at=started_at,
            expected_end_at=expected_end_at,
            source=compact_text(payload.get("source"), 48) or "autonomous_life",
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "activity": self.activity,
            "category": self.category,
            "subject": self.subject,
            "detail": self.detail,
            "started_at": self.started_at,
            "expected_end_at": self.expected_end_at,
            "source": self.source,
        }

    def fact(self) -> str:
        return f"{self.activity} — {self.detail}" if self.detail else self.activity


def _newest_historical_activity(values: dict[str, object]) -> PresenceActivity | None:
    candidates: list[PresenceActivity] = []
    previous = PresenceActivity.from_dict(values.get("previous_activity"))
    if previous is not None:
        candidates.append(previous)
    recent = values.get("recent_activities")
    if isinstance(recent, (list, tuple)):
        candidates.extend(
            activity
            for item in recent
            if (activity := PresenceActivity.from_dict(item)) is not None
        )
    return max(
        candidates,
        key=lambda item: (item.expected_end_at, item.started_at),
        default=None,
    )


@dataclass(frozen=True, slots=True)
class PresenceState:
    current_activity: PresenceActivity | None = None
    previous_activity: PresenceActivity | None = None
    last_decision_at: float = 0.0
    next_decision_at: float = 0.0
    decision_pending: bool = False
    claim_token: str | None = None
    claim_expires_at: float = 0.0
    retry_at: float = 0.0
    last_error: str | None = None
    activity_pattern: ActivityPattern = ActivityPattern()

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        now: float | None = None,
    ) -> "PresenceState":
        current_time = time.time() if now is None else max(0.0, float(now))
        values = payload if isinstance(payload, dict) else {}
        current_activity = PresenceActivity.from_dict(
            values.get("current_activity")
        )
        previous = _newest_historical_activity(values)
        next_decision_at = _timestamp(values.get("next_decision_at"))
        if not next_decision_at:
            next_decision_at = _timestamp(values.get("life_next_run_at"))
        claim_token = _optional_text(values.get("claim_token"), 80)
        claim_expires_at = _timestamp(values.get("claim_expires_at"))
        if claim_token is None or claim_expires_at <= current_time:
            claim_token = None
            claim_expires_at = 0.0
        state = cls(
            current_activity=current_activity,
            previous_activity=previous,
            last_decision_at=_timestamp(
                values.get("last_decision_at"),
                _timestamp(values.get("life_last_run_at")),
            ),
            next_decision_at=next_decision_at,
            decision_pending=bool(values.get("decision_pending", False)),
            claim_token=claim_token,
            claim_expires_at=claim_expires_at,
            retry_at=_timestamp(values.get("retry_at")),
            last_error=_optional_text(values.get("last_error"), 120),
            activity_pattern=ActivityPattern.from_dict(
                values.get("activity_pattern")
            ),
        )
        return normalize_presence(
            state,
            now=current_time,
            initialize_schedule=True,
            repair_schedule=True,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "current_activity": (
                self.current_activity.as_dict()
                if self.current_activity is not None
                else None
            ),
            "previous_activity": (
                self.previous_activity.as_dict()
                if self.previous_activity is not None
                else None
            ),
            "last_decision_at": self.last_decision_at,
            "next_decision_at": self.next_decision_at,
            "decision_pending": self.decision_pending,
            "claim_token": self.claim_token,
            "claim_expires_at": self.claim_expires_at,
            "retry_at": self.retry_at,
            "last_error": self.last_error,
            "activity_pattern": self.activity_pattern.as_dict(),
        }


def normalize_presence(
    state: PresenceState,
    *,
    now: float,
    initialize_schedule: bool,
    repair_schedule: bool = False,
) -> PresenceState:
    """Validate lifecycle metadata without expiring the current activity."""

    current = state.current_activity
    previous = state.previous_activity
    pending = state.decision_pending
    next_at = state.next_decision_at
    token = state.claim_token
    claim_expires_at = state.claim_expires_at
    last_error = state.last_error

    if token is not None and claim_expires_at <= now:
        token = None
        claim_expires_at = 0.0
        pending = True
        last_error = "stale claim released"

    schedule_invalid = next_at <= 0.0
    if repair_schedule and not state.retry_at and not state.last_error:
        lifecycle_seconds = next_at - state.last_decision_at
        schedule_invalid = schedule_invalid or (
            state.last_decision_at > 0.0
            and (
                lifecycle_seconds
                < minutes_to_seconds(MIN_DECISION_MINUTES) - 1.0
                or lifecycle_seconds
                > minutes_to_seconds(MAX_DECISION_MINUTES) + 1.0
            )
        )
        schedule_invalid = schedule_invalid or (
            state.last_decision_at <= 0.0
            and next_at > now + minutes_to_seconds(MAX_DECISION_MINUTES)
        )
        schedule_invalid = schedule_invalid or (
            state.last_decision_at > 0.0 and current is None
        )
        schedule_invalid = schedule_invalid or (
            current is not None
            and abs(current.expected_end_at - next_at) > 1.0
        )
        schedule_invalid = schedule_invalid or (
            current is not None
            and not _meaningful_detail(current.detail, activity=current.activity)
        )
    if schedule_invalid and initialize_schedule:
        next_at = now
        pending = True
        token = None
        claim_expires_at = 0.0
        last_error = None
    if next_at > 0.0 and next_at <= now:
        pending = True

    pattern = state.activity_pattern
    current_key = _record_key(current)
    previous_key = _record_key(previous)
    if (
        pattern.current_key != current_key
        or pattern.previous_key != previous_key
    ):
        pattern = ActivityPattern(
            current_key=current_key,
            previous_key=previous_key,
            repeat_count=pattern.repeat_count or int(bool(current_key)),
        )

    return replace(
        state,
        current_activity=current,
        previous_activity=previous,
        next_decision_at=next_at,
        decision_pending=pending,
        claim_token=token,
        claim_expires_at=claim_expires_at,
        last_error=last_error,
        activity_pattern=pattern,
    )


def activity_continuity(
    current: PresenceActivity | None,
    last_assistant_at: float | None,
) -> str:
    if current is None:
        return "none"
    if last_assistant_at is None or current.started_at > last_assistant_at:
        return "new"
    return "ongoing"


@dataclass(frozen=True, slots=True)
class LifeDecision:
    mode: str
    activity: str = ""
    category: str | None = None
    subject: str | None = None
    detail: str | None = None
    interest_addition: str | None = None
    continuation_reason: str | None = None


def _unsupported_proper_nouns(value: str, grounded_context: str) -> bool:
    grounded = grounded_context.casefold()
    return any(
        match.group(0) != "I"
        and match.start() != 0
        and match.group(0).casefold() not in grounded
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9'-]*\b", value)
    )


def parse_life_decision(
    output: object,
    *,
    grounded_context: str = "",
) -> LifeDecision | None:
    text = str(output or "")
    matches = re.findall(
        r"<AKANE_LIFE>\s*(.*?)\s*</AKANE_LIFE>",
        text,
        re.DOTALL,
    )
    if len(matches) != 1:
        return None
    try:
        payload = json.loads(matches[0])
    except (TypeError, ValueError):
        return None
    required = {
        "mode",
        "activity",
        "category",
        "subject",
        "detail",
        "interest_addition",
        "continuation_reason",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        return None

    mode = compact_text(payload.get("mode"), 16).casefold()
    activity = _activity_text(payload.get("activity"))
    category = _optional_text(payload.get("category"), 48)
    subject = _optional_text(payload.get("subject"), 160)
    detail = _optional_text(payload.get("detail"), _MAX_DETAIL_CHARS)
    addition = _optional_text(payload.get("interest_addition"), 100)
    continuation_reason = _optional_text(
        payload.get("continuation_reason"),
        _MAX_DETAIL_CHARS,
    )
    optional_values = (
        ("category", category),
        ("subject", subject),
        ("detail", detail),
        ("interest_addition", addition),
        ("continuation_reason", continuation_reason),
    )
    if (
        not isinstance(payload.get("mode"), str)
        or mode not in _LIFE_MODES
        or any(
            payload[name] is not None and value is None
            for name, value in optional_values
        )
        or (
            mode == "new"
            and (
                not isinstance(payload.get("activity"), str)
                or not activity
                or not _meaningful_detail(detail, activity=activity)
                or continuation_reason is not None
            )
        )
        or (
            mode == "continue"
            and (
                payload.get("activity") is not None
                or category is not None
                or subject is not None
                or detail is not None
                or addition is not None
                or continuation_reason is None
            )
        )
    ):
        return None

    grounded_text = " ".join(
        value
        for value in (activity, category or "", subject or "", detail or "")
        if value
    )
    if (
        _unsupported_proper_nouns(grounded_text, grounded_context)
        or (
            subject is not None
            and _unsupported_proper_nouns(
                f"about {subject}",
                grounded_context,
            )
        )
    ):
        return None

    return LifeDecision(
        mode=mode,
        activity=activity,
        category=category,
        subject=subject,
        detail=detail,
        interest_addition=addition,
        continuation_reason=continuation_reason,
    )


def life_decision_rejection(
    state: PresenceState,
    decision: LifeDecision,
) -> str:
    if decision.mode == "continue":
        current = state.current_activity
        if (
            current is None
            or not _meaningful_detail(current.detail, activity=current.activity)
        ):
            return "no valid current activity to continue"
        reason_tokens = set(activity_key(decision.continuation_reason).split())
        current_tokens = set(
            activity_key(
                f"{current.activity} {current.subject or ''} {current.detail or ''}"
            ).split()
        )
        minimum_reason_tokens = (
            min(8, 3 + state.activity_pattern.repeat_count)
            if state.activity_pattern.repeat_count >= 2
            else 2
        )
        if (
            len(reason_tokens) < minimum_reason_tokens
            or not reason_tokens - current_tokens
        ):
            return "continuation needs a stronger meaningful reason"
        return ""

    key = _activity_semantic_key(decision.activity, decision.subject)
    pattern = state.activity_pattern
    if _near_key(key, pattern.current_key):
        return "proposal repeats the current activity without choosing continuation"
    if _near_key(key, pattern.previous_key):
        return "proposal repeats a recent activity pattern"
    return ""


def _updated_pattern(
    pattern: ActivityPattern,
    *,
    new_activity: PresenceActivity,
    old_current: PresenceActivity | None,
) -> ActivityPattern:
    return ActivityPattern(
        current_key=_record_key(new_activity),
        previous_key=_record_key(old_current) or pattern.previous_key,
        repeat_count=1,
    )


def apply_life_decision(
    state: PresenceState,
    decision: LifeDecision,
    *,
    now: float,
) -> PresenceState:
    current = state.current_activity
    previous = state.previous_activity
    pattern = state.activity_pattern
    next_at = next_decision_time(now)

    if decision.mode == "new":
        old_current = current
        previous = current if current is not None else previous
        current = PresenceActivity(
            activity=decision.activity,
            category=decision.category,
            subject=decision.subject,
            detail=decision.detail,
            started_at=now,
            expected_end_at=next_at,
        )
        pattern = _updated_pattern(
            pattern,
            new_activity=current,
            old_current=old_current,
        )
    elif decision.mode == "continue" and current is not None:
        current = replace(
            current,
            expected_end_at=next_at,
        )
        pattern = ActivityPattern(
            current_key=_record_key(current),
            previous_key=pattern.previous_key,
            repeat_count=pattern.repeat_count + 1,
        )

    return replace(
        state,
        current_activity=current,
        previous_activity=previous,
        last_decision_at=now,
        next_decision_at=next_at,
        decision_pending=False,
        claim_token=None,
        claim_expires_at=0.0,
        retry_at=0.0,
        last_error=None,
        activity_pattern=pattern,
    )


def validate_interest_addition(
    addition: str | None,
    *,
    activity: str,
    subject: str | None,
    detail: str | None,
    existing_interests: tuple[str, ...],
    grounded_context: str,
) -> str | None:
    candidate = compact_text(addition, 100)
    if not candidate:
        return None
    key = activity_key(candidate)
    words = key.split()
    if (
        not 1 <= len(words) <= 5
        or key == activity_key(activity)
        or any(key == activity_key(item) for item in existing_interests)
        or _unsupported_proper_nouns(
            f"interest in {candidate}",
            grounded_context,
        )
    ):
        return None
    connection = set(words) & set(
        activity_key(f"{activity} {subject or ''} {detail or ''}").split()
    )
    return candidate if connection else None


def format_presence_context(
    state: PresenceState,
    *,
    now: float,
    continuity: str = "none",
    local_time: str = "",
    daypart: str = "",
) -> str:
    activity = state.current_activity
    if activity is None:
        parts = [
            "Current activity: none.",
            "Activity status: none.",
        ]
        if state.previous_activity is not None:
            parts.append(
                f"Previous activity: {state.previous_activity.activity}."
            )
    else:
        status = (
            "new since the previous conversation"
            if continuity == "new"
            else "ongoing"
        )
        parts = [
            f"Current activity: {activity.activity}.",
            f"Activity status: {status}.",
        ]
        if activity.subject is not None:
            parts.append(f"Recorded subject: {activity.subject}.")
        if activity.category is not None:
            parts.append(f"Recorded category: {activity.category}.")
        if activity.detail is not None:
            parts.append(f"Recorded detail: {activity.detail}.")
        else:
            parts.append("No additional activity detail is recorded.")
        minutes = max(0, int((now - activity.started_at) // 60))
        parts.append(f"Started about {minutes} minutes ago.")
    if local_time:
        parts.append(f"Current local time: {local_time}.")
    if daypart:
        parts.append(f"Current daypart: {daypart}.")
    parts.append(
        "Use only the supplied activity details. Do not invent missing titles, "
        "subjects, sources, progress, surroundings, or content."
    )
    if continuity == "new":
        parts.append(
            "Do not describe this new activity as still happening, continuing, "
            "happening again, or as before."
        )
    return "Presence facts:\n" + "\n".join(parts)
