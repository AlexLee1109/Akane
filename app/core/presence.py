"""Authoritative schema and validation for Akane's persisted offscreen life."""

from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from app.core.utils import compact_text

MIN_DECISION_MINUTES = 150.0
MAX_DECISION_MINUTES = 210.0
CLAIM_SECONDS = 10.0 * 60.0
RETRY_SECONDS = 5.0 * 60.0

_MAX_ACTIVITY_CHARS = 160
_MAX_DETAIL_CHARS = 240
_LIFE_MODES = {"new", "continue"}
_KEY_STOPWORDS = {
    "a", "about", "an", "and", "as", "at", "by", "for", "from", "her",
    "herself", "in", "into", "it", "of", "on", "or", "some", "the", "to",
    "up", "while", "with",
}
_GENERIC_ACTIVITY_TERMS = {
    "brows", "listen", "read", "relax", "unwind", "view", "watch",
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
    if not isinstance(value, str):
        return ""
    activity = compact_text(value, _MAX_ACTIVITY_CHARS)
    words = re.findall(r"[a-z0-9]+", activity.casefold())
    return activity if 1 <= len(words) <= 24 else ""


def minutes_to_seconds(minutes: float) -> float:
    return float(minutes) * 60.0


def next_decision_time(now: float) -> float:
    interval = random.uniform(MIN_DECISION_MINUTES, MAX_DECISION_MINUTES)
    return max(0.0, float(now)) + minutes_to_seconds(interval)


def activity_key(value: object) -> str:
    """Small comparison metadata; it is not an activity catalogue."""

    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+", compact_text(value, 360).casefold()):
        if token in _KEY_STOPWORDS:
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
    return " ".join(tokens)[:120]


def _near_key(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    overlap = len(left_tokens & right_tokens) / max(1, len(union))
    if overlap >= 0.58 or SequenceMatcher(None, left, right).ratio() >= 0.84:
        return True
    left_focus = left_tokens - _GENERIC_ACTIVITY_TERMS
    right_focus = right_tokens - _GENERIC_ACTIVITY_TERMS
    return bool(
        (left_tokens | right_tokens) & _GENERIC_ACTIVITY_TERMS
        and left_focus
        and left_focus == right_focus
        and len(left_focus) <= 2
    )


def _meaningful_detail(detail: str | None, activity: str) -> bool:
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
            repeats = max(0, min(100, int(values.get("repeat_count", 0))))
        except (TypeError, ValueError):
            repeats = 0
        return cls(
            activity_key(values.get("current_key")),
            activity_key(values.get("previous_key") or values.get("prior_key")),
            repeats,
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
        started = _timestamp(payload.get("started_at"))
        expected = _timestamp(
            payload.get("expected_end_at"),
        )
        if not activity or expected <= started:
            return None
        raw_detail = payload.get("detail")
        detail = _optional_text(raw_detail, _MAX_DETAIL_CHARS)
        if raw_detail is not None and detail is None:
            return None
        return cls(
            activity=activity,
            category=_optional_text(payload.get("category"), 48),
            subject=_optional_text(payload.get("subject"), 160),
            detail=detail,
            started_at=started,
            expected_end_at=expected,
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


@dataclass(frozen=True, slots=True)
class PresenceState:
    current_activity: PresenceActivity | None = None
    previous_activity: PresenceActivity | None = None
    last_decision_at: float = 0.0
    next_decision_at: float = 0.0
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
        repair_schedule: bool = False,
    ) -> "PresenceState":
        current_time = time.time() if now is None else max(0.0, float(now))
        values = payload if isinstance(payload, dict) else {}
        token = _optional_text(values.get("claim_token"), 80)
        claim_expires = _timestamp(values.get("claim_expires_at"))
        if token is None or claim_expires <= current_time:
            token = None
            claim_expires = 0.0
        state = cls(
            current_activity=PresenceActivity.from_dict(values.get("current_activity")),
            previous_activity=PresenceActivity.from_dict(
                values.get("previous_activity")
            ),
            last_decision_at=_timestamp(values.get("last_decision_at")),
            next_decision_at=_timestamp(values.get("next_decision_at")),
            claim_token=token,
            claim_expires_at=claim_expires,
            retry_at=_timestamp(values.get("retry_at")),
            last_error=_optional_text(values.get("last_error"), 120),
            activity_pattern=ActivityPattern.from_dict(values.get("activity_pattern")),
        )
        return normalize_presence(
            state,
            now=current_time,
            initialize_schedule=True,
            repair_schedule=repair_schedule,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "current_activity": (
                self.current_activity.as_dict() if self.current_activity else None
            ),
            "previous_activity": (
                self.previous_activity.as_dict() if self.previous_activity else None
            ),
            "last_decision_at": self.last_decision_at,
            "next_decision_at": self.next_decision_at,
            "claim_token": self.claim_token,
            "claim_expires_at": self.claim_expires_at,
            "retry_at": self.retry_at,
            "last_error": self.last_error,
            "activity_pattern": self.activity_pattern.as_dict(),
        }


def _repaired_boundary(
    current: PresenceActivity,
    next_at: float,
    last_decision_at: float,
    now: float,
) -> tuple[PresenceActivity, float]:
    wrong_minute_min = minutes_to_seconds(MIN_DECISION_MINUTES) * 10.0
    wrong_minute_max = minutes_to_seconds(MAX_DECISION_MINUTES) * 10.0
    valid_min = minutes_to_seconds(MIN_DECISION_MINUTES) - 1.0
    valid_max = minutes_to_seconds(MAX_DECISION_MINUTES) + 1.0

    lifecycle_start = (
        last_decision_at
        if last_decision_at >= current.started_at
        else current.started_at
    )

    def repair(candidate: float) -> float:
        duration = candidate - lifecycle_start
        if wrong_minute_min - 1.0 <= duration <= wrong_minute_max + 1.0:
            return lifecycle_start + duration / 10.0
        return candidate

    scheduled = repair(next_at)
    expected = repair(current.expected_end_at)
    if valid_min <= scheduled - lifecycle_start <= valid_max:
        boundary = scheduled
    elif valid_min <= expected - lifecycle_start <= valid_max:
        boundary = expected
    else:
        boundary = max(current.started_at + 1.0, now)
    return replace(current, expected_end_at=boundary), boundary


def normalize_presence(
    state: PresenceState,
    *,
    now: float,
    initialize_schedule: bool,
    repair_schedule: bool = False,
) -> PresenceState:
    """Repair lifecycle metadata without clearing the last valid activity."""

    current = state.current_activity
    token = state.claim_token
    claim_expires = state.claim_expires_at
    error = state.last_error
    if token is not None and claim_expires <= now:
        token = None
        claim_expires = 0.0
        error = "stale claim released"

    next_at = state.next_decision_at
    if current is None:
        if initialize_schedule and (next_at <= 0.0 or repair_schedule):
            next_at = now
    elif repair_schedule:
        current, next_at = _repaired_boundary(
            current,
            next_at,
            state.last_decision_at,
            now,
        )
    elif initialize_schedule and next_at <= 0.0:
        next_at = now

    current_key = activity_key(
        f"{current.activity} {current.subject or ''}" if current else ""
    )
    previous = state.previous_activity
    previous_key = activity_key(
        f"{previous.activity} {previous.subject or ''}" if previous else ""
    )
    pattern = state.activity_pattern
    if (
        pattern.current_key != current_key
        or pattern.previous_key != previous_key
    ):
        pattern = ActivityPattern(
            current_key,
            previous_key,
            pattern.repeat_count or int(bool(current_key)),
        )
    return replace(
        state,
        current_activity=current,
        next_decision_at=next_at,
        claim_token=token,
        claim_expires_at=claim_expires,
        last_error=error,
        activity_pattern=pattern,
    )


def activity_continuity(
    current: PresenceActivity | None,
    last_assistant_at: float | None,
) -> str:
    if current is None:
        return "none"
    if last_assistant_at is None or current.started_at >= last_assistant_at:
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
    grounded = {
        match.group(0).casefold()
        for match in re.finditer(r"\b[A-Za-z0-9'-]+\b", grounded_context)
    }
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9'-]*\b", value):
        word = match.group(0)
        if word == "I" or word.casefold() in grounded:
            continue
        prefix = value[: match.start()].rstrip()
        sentence_initial_action = (
            (not prefix or prefix.endswith((".", "!", "?")))
            and len(word) > 4
            and word.casefold().endswith(("ed", "ing"))
        )
        if not sentence_initial_action:
            return True
    return False


def parse_life_decision(
    output: object,
    *,
    grounded_context: str = "",
) -> LifeDecision | None:
    matches = re.findall(
        r"<AKANE_LIFE>\s*(.*?)\s*</AKANE_LIFE>",
        str(output or ""),
        re.DOTALL,
    )
    if len(matches) != 1:
        return None
    try:
        payload = json.loads(matches[0])
    except (TypeError, ValueError):
        return None
    keys = {
        "mode", "activity", "category", "subject", "detail",
        "interest_addition", "continuation_reason",
    }
    if not isinstance(payload, dict) or set(payload) != keys:
        return None
    mode = compact_text(payload.get("mode"), 16).casefold()
    activity = _activity_text(payload.get("activity"))
    category = _optional_text(payload.get("category"), 48)
    subject = _optional_text(payload.get("subject"), 160)
    detail = _optional_text(payload.get("detail"), _MAX_DETAIL_CHARS)
    addition = _optional_text(payload.get("interest_addition"), 100)
    reason = _optional_text(payload.get("continuation_reason"), _MAX_DETAIL_CHARS)
    optional = {
        "category": category,
        "subject": subject,
        "detail": detail,
        "interest_addition": addition,
        "continuation_reason": reason,
    }
    if (
        mode not in _LIFE_MODES
        or any(payload[name] is not None and value is None for name, value in optional.items())
    ):
        return None
    if mode == "new":
        if (
            not activity
            or not _meaningful_detail(detail, activity)
            or reason is not None
        ):
            return None
    elif (
        payload.get("activity") is not None
        or any(value is not None for name, value in optional.items() if name != "continuation_reason")
        or reason is None
    ):
        return None
    grounded = ". ".join(
        item for item in (activity, category or "", subject or "", detail or "") if item
    )
    if _unsupported_proper_nouns(grounded, grounded_context):
        return None
    return LifeDecision(mode, activity, category, subject, detail, addition, reason)


def life_decision_rejection(
    state: PresenceState,
    decision: LifeDecision,
) -> str:
    if decision.mode == "continue":
        current = state.current_activity
        if current is None or not _meaningful_detail(current.detail, current.activity):
            return "no valid current activity to continue"
        reason = set(activity_key(decision.continuation_reason).split())
        existing = set(activity_key(current.fact()).split())
        if len(reason) < 2 or not reason - existing:
            return "continuation needs a meaningful reason"
        return ""
    proposed = activity_key(f"{decision.activity} {decision.subject or ''}")
    pattern = state.activity_pattern
    if _near_key(proposed, pattern.current_key):
        return "proposal repeats the current activity"
    if _near_key(proposed, pattern.previous_key):
        return "proposal repeats the previous activity"
    if not _meaningful_detail(decision.detail, decision.activity):
        return "proposal lacks concrete detail"
    return ""


def apply_life_decision(
    state: PresenceState,
    decision: LifeDecision,
    *,
    now: float,
) -> PresenceState:
    next_at = next_decision_time(now)
    current = state.current_activity
    previous = state.previous_activity
    pattern = state.activity_pattern
    if decision.mode == "new":
        previous = current or previous
        current = PresenceActivity(
            decision.activity,
            decision.category,
            decision.subject,
            decision.detail,
            now,
            next_at,
        )
        pattern = ActivityPattern(
            activity_key(f"{current.activity} {current.subject or ''}"),
            activity_key(
                f"{previous.activity} {previous.subject or ''}" if previous else ""
            ),
            1,
        )
    elif current is not None:
        current = replace(current, expected_end_at=next_at)
        pattern = replace(
            pattern,
            current_key=activity_key(f"{current.activity} {current.subject or ''}"),
            repeat_count=pattern.repeat_count + 1,
        )
    return PresenceState(
        current_activity=current,
        previous_activity=previous,
        last_decision_at=now,
        next_decision_at=next_at,
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
    key = activity_key(candidate)
    if (
        not candidate
        or not 1 <= len(key.split()) <= 5
        or any(key == activity_key(item) for item in existing_interests)
        or _unsupported_proper_nouns(candidate, grounded_context)
    ):
        return None
    activity_terms = set(
        activity_key(f"{activity} {subject or ''} {detail or ''}").split()
    )
    return candidate if set(key.split()) & activity_terms else None


def format_presence_context(
    state: PresenceState,
    *,
    now: float,
    continuity: str = "none",
    include_previous: bool = False,
) -> str:
    del now
    activity = state.current_activity
    lines: list[str] = []
    if activity is None:
        lines.append("Current activity: none recorded.")
        if state.previous_activity is not None:
            lines.append(
                f"Previous activity: {state.previous_activity.activity}."
            )
    else:
        lines.extend(
            (
                f"Current activity: {activity.activity}.",
                f"Activity continuity: {continuity}.",
            )
        )
        if activity.subject:
            lines.append(f"Recorded subject: {activity.subject}.")
        if activity.detail:
            lines.append(f"Recorded detail: {activity.detail}.")
        if include_previous and state.previous_activity is not None:
            lines.append(
                f"Previous activity: {state.previous_activity.fact()}."
            )
    lines.append("Use only recorded activity details.")
    if continuity == "new":
        lines.append("This activity is new; do not describe it as still continuing.")
    return "\n".join(lines)
