"""Canonical state and proposal contract for Akane's offscreen presence."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, replace

from app.core.utils import compact_text

PRESENCE_INTERVAL_SECONDS = 4 * 60 * 60
CLAIM_SECONDS = 10 * 60
RETRY_SECONDS = 5 * 60

PRESENCE_PROPOSAL_FIELDS = (
    "decision",
    "activity",
    "continuation_reason",
    "emotion",
)
BOOTSTRAP_PRESENCE_FIELDS = ("decision", "activity", "emotion")
PRESENCE_ACTIVITY_FIELDS = ("summary", "focus")
PRESENCE_EMOTION_FIELDS = ("primary", "intensity", "cause")
_DECISIONS = {"new", "continue"}
_EMOTIONS = {
    "calm",
    "content",
    "curious",
    "interested",
    "amused",
    "excited",
    "inspired",
    "affectionate",
    "hopeful",
    "uncertain",
    "concerned",
    "anxious",
    "lonely",
    "tired",
    "disappointed",
    "sad",
    "frustrated",
    "irritated",
    "angry",
}
UNSUPPORTED_PHYSICAL_ACTIVITY_REASON = (
    "presence activity must be plausible for a digital AI companion and must not "
    "invent a physical environment"
)
_DIRECT_UNSUPPORTED_ACTIVITY = re.compile(
    r"\b(?:eat(?:s|ing)?|dr(?:ink|inks|inking|ank)|sip(?:s|ping)?|"
    r"travel(?:s|ed|led|ing)|commut(?:e|es|ed|ing)|shop(?:s|ped|ping)?|"
    r"purchas(?:e|es|ed|ing)|buy(?:s|ing)?|bought|cook(?:s|ed|ing)?|"
    r"wash(?:es|ed|ing)?|vacuum(?:s|ed|ing)?|"
    r"meet(?:s|ing)?\s+(?:someone|people|other\s+people|friends?|"
    r"a\s+friends?|a\s+person)|"
    r"do(?:es|ing)?\s+(?:laundry|dishes|chores)|breaking\s+news|"
    r"new\s+release|chapter\s+\d+|episode\s+\d+)\b",
    re.IGNORECASE,
)
_PHYSICAL_CONTEXT = re.compile(
    r"\b(?:room|bed|sofa|couch|chair|desk|window|balcony|porch|garden|"
    r"park|cafe|store|school|office|place|outside|indoors|weather|rain|"
    r"snow|storm|clouds?|"
    r"sun(?:light|rise|set)?|daylight|sky|scenery|landscape|tea|coffee|"
    r"food|meal|snack|drink|cup|mug|glass|book|phone|pen|package|blinds|"
    r"physical\s+object)\b",
    re.IGNORECASE,
)
_PHYSICAL_CONTEXT_ACTION = re.compile(
    r"\b(?:sit(?:s|ting)?|stand(?:s|ing)?|lie|lies|lying|sleep(?:s|ing)?|"
    r"rest(?:s|ing)?|relax(?:es|ing)?|look(?:s|ing)?|watch(?:es|ing)?|"
    r"observ(?:e|es|ing)|hold(?:s|ing)?|carr(?:y|ies|ying)|"
    r"handl(?:e|es|ing)|hav(?:e|ing)|visit(?:s|ed|ing)?|"
    r"clean(?:s|ed|ing)?|attend(?:s|ed|ing)?|taking\s+a\s+break)\b|"
    r"\b(?:akane|she)\s+is\s+(?:in|at|by|outside)\b",
    re.IGNORECASE,
)
_OBSOLETE_ERRORS = (
    "life decision lacks a grounded emotional appraisal",
    "invalid life block",
    "invalid emotional appraisal",
)


def _nullable(kind: str) -> dict[str, object]:
    return {"type": [kind, "null"]}


PRESENCE_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": sorted(_DECISIONS)},
            "activity": {
                "type": ["object", "null"],
                "properties": {
                    "summary": {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": list(PRESENCE_ACTIVITY_FIELDS),
                "additionalProperties": False,
            },
            "continuation_reason": _nullable("string"),
            "emotion": {
                "type": ["object", "null"],
                "properties": {
                    "primary": {"type": "string", "enum": sorted(_EMOTIONS)},
                    "intensity": {"type": "number"},
                    "cause": {"type": "string"},
                },
                "required": list(PRESENCE_EMOTION_FIELDS),
                "additionalProperties": False,
            },
        },
        "required": list(PRESENCE_PROPOSAL_FIELDS),
        "additionalProperties": False,
    },
    separators=(",", ":"),
)

BOOTSTRAP_PRESENCE_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["new"]},
            "activity": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "minLength": 1},
                    "focus": {"type": "string", "minLength": 1},
                },
                "required": list(PRESENCE_ACTIVITY_FIELDS),
                "additionalProperties": False,
            },
            "emotion": {
                "type": ["object", "null"],
                "properties": {
                    "primary": {"type": "string", "enum": sorted(_EMOTIONS)},
                    "intensity": {"type": "number"},
                    "cause": {"type": "string", "minLength": 1},
                },
                "required": list(PRESENCE_EMOTION_FIELDS),
                "additionalProperties": False,
            },
        },
        "required": ["decision", "activity"],
        "additionalProperties": False,
    },
    separators=(",", ":"),
)


def _timestamp(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number >= 0.0 else 0.0


def _text(value: object, limit: int) -> str:
    return compact_text(value, limit) if isinstance(value, str) else ""


def _activity_text(value: object, limit: int, maximum_words: int) -> str:
    text = _text(value, limit)
    count = len(re.findall(r"[a-z0-9]+", text.casefold()))
    return text if 1 <= count <= maximum_words else ""


@dataclass(frozen=True, slots=True)
class PresenceActivity:
    activity_id: str
    summary: str
    focus: str
    started_at: float
    expected_end_at: float

    @classmethod
    def from_dict(cls, payload: object) -> "PresenceActivity | None":
        if not isinstance(payload, dict):
            return None
        activity_id = _text(payload.get("activity_id"), 80)
        summary = _activity_text(payload.get("summary"), 120, 18)
        focus = _activity_text(payload.get("focus"), 220, 36)
        started_at = _timestamp(payload.get("started_at"))
        expected_end_at = _timestamp(payload.get("expected_end_at"))
        if not activity_id or not summary or not focus or expected_end_at <= started_at:
            return None
        return cls(activity_id, summary, focus, started_at, expected_end_at)

    def as_dict(self) -> dict[str, object]:
        return {
            "activity_id": self.activity_id,
            "summary": self.summary,
            "focus": self.focus,
            "started_at": self.started_at,
            "expected_end_at": self.expected_end_at,
        }


@dataclass(frozen=True, slots=True)
class PresenceState:
    current_activity: PresenceActivity | None = None
    previous_activity: PresenceActivity | None = None
    last_decision_at: float = 0.0
    next_decision_at: float = 0.0
    retry_at: float = 0.0
    last_error: str | None = None
    claim_token: str | None = None
    claim_expires_at: float = 0.0
    continuation_count: int = 0

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
        token = _text(values.get("claim_token"), 80) or None
        claim_expires_at = _timestamp(values.get("claim_expires_at"))
        if token is None or claim_expires_at <= current_time:
            token = None
            claim_expires_at = 0.0
        raw_count = values.get("continuation_count", 0)
        continuation_count = (
            max(0, min(1, raw_count)) if type(raw_count) is int else 0
        )
        state = cls(
            current_activity=PresenceActivity.from_dict(values.get("current_activity")),
            previous_activity=PresenceActivity.from_dict(values.get("previous_activity")),
            last_decision_at=_timestamp(values.get("last_decision_at")),
            next_decision_at=_timestamp(values.get("next_decision_at")),
            retry_at=_timestamp(values.get("retry_at")),
            last_error=_text(values.get("last_error"), 120) or None,
            claim_token=token,
            claim_expires_at=claim_expires_at,
            continuation_count=continuation_count,
        )
        return normalize_presence(
            state,
            now=current_time,
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
            "retry_at": self.retry_at,
            "last_error": self.last_error,
            "claim_token": self.claim_token,
            "claim_expires_at": self.claim_expires_at,
            "continuation_count": self.continuation_count,
        }


def needs_bootstrap(state: PresenceState) -> bool:
    return state.current_activity is None or state.last_decision_at <= 0.0


def normalize_presence(
    state: PresenceState,
    *,
    now: float,
    initialize_schedule: bool = False,
    repair_schedule: bool = False,
) -> PresenceState:
    """Normalize claims and perform the one schema-migration bootstrap repair."""

    del initialize_schedule
    token = state.claim_token
    claim_expires_at = state.claim_expires_at
    if token is not None and claim_expires_at <= now:
        token = None
        claim_expires_at = 0.0
    next_decision_at = state.next_decision_at
    retry_at = state.retry_at
    last_error = state.last_error
    if repair_schedule and needs_bootstrap(state):
        next_decision_at = 0.0
        retry_at = 0.0
        token = None
        claim_expires_at = 0.0
    if last_error and any(error in last_error.casefold() for error in _OBSOLETE_ERRORS):
        last_error = None
    return replace(
        state,
        next_decision_at=next_decision_at,
        retry_at=retry_at,
        last_error=last_error,
        claim_token=token,
        claim_expires_at=claim_expires_at,
    )


@dataclass(frozen=True, slots=True)
class ProposedActivity:
    summary: str
    focus: str


@dataclass(frozen=True, slots=True)
class ProposedEmotion:
    primary: str
    intensity: float
    cause: str


@dataclass(frozen=True, slots=True)
class PresenceProposal:
    decision: str
    activity: ProposedActivity | None
    continuation_reason: str | None
    emotion: ProposedEmotion | None


class PresenceParseError(ValueError):
    """A concise failure in a dedicated raw-JSON presence completion."""

    def __init__(self, message: str, *, decoded: object = None) -> None:
        super().__init__(message)
        self.decoded = decoded


def _normalize_activity(
    payload: object,
    *,
    bootstrap: bool,
) -> ProposedActivity | None:
    if not isinstance(payload, dict):
        return None
    summary = (
        _text(payload.get("summary"), 120)
        if bootstrap
        else _activity_text(payload.get("summary"), 120, 18)
    )
    focus = (
        _text(payload.get("focus"), 220)
        if bootstrap
        else _activity_text(payload.get("focus"), 220, 36)
    )
    return ProposedActivity(summary, focus) if summary and focus else None


def _normalize_emotion(payload: object) -> ProposedEmotion | None:
    if not isinstance(payload, dict) or set(payload) != set(PRESENCE_EMOTION_FIELDS):
        return None
    primary = _text(payload.get("primary"), 32).casefold()
    intensity = payload.get("intensity")
    cause = _text(payload.get("cause"), 160)
    if (
        primary not in _EMOTIONS
        or type(intensity) not in {int, float}
        or not math.isfinite(float(intensity))
        or not 0.20 <= float(intensity) <= 0.45
        or not cause
    ):
        return None
    return ProposedEmotion(primary, float(intensity), cause)


def presence_activity_rejection(activity: ProposedActivity | None) -> str:
    if activity is None:
        return ""
    text = f"{activity.summary} {activity.focus}"
    unsupported = _DIRECT_UNSUPPORTED_ACTIVITY.search(text) or (
        _PHYSICAL_CONTEXT.search(text)
        and _PHYSICAL_CONTEXT_ACTION.search(text)
    )
    return UNSUPPORTED_PHYSICAL_ACTIVITY_REASON if unsupported else ""


def parse_presence_proposal(
    output: object,
    *,
    bootstrap: bool = False,
) -> PresenceProposal:
    """Decode one complete raw proposal and normalize its optional emotion once."""

    text = str(output or "").strip()
    if not text:
        raise PresenceParseError(
            "presence JSON decode failed" if bootstrap else "empty presence output"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PresenceParseError(
            (
                "presence JSON decode failed"
                if bootstrap
                else f"presence JSON decode failed at line {exc.lineno}, column {exc.colno}"
            )
        ) from exc
    if not isinstance(payload, dict):
        raise PresenceParseError(
            (
                "presence JSON decode failed"
                if bootstrap
                else "presence output must be a JSON object"
            ),
            decoded=payload,
        )
    allowed = set(
        BOOTSTRAP_PRESENCE_FIELDS if bootstrap else PRESENCE_PROPOSAL_FIELDS
    )
    unexpected = set(payload) - allowed
    if unexpected:
        raise PresenceParseError("unexpected presence fields", decoded=payload)
    if bootstrap:
        if payload.get("decision") != "new":
            raise PresenceParseError(
                "bootstrap requires decision=new",
                decoded=payload,
            )
        raw_activity = payload.get("activity")
        if not isinstance(raw_activity, dict):
            raise PresenceParseError(
                "new decision requires activity object",
                decoded=payload,
            )
    else:
        missing = set(PRESENCE_PROPOSAL_FIELDS[:-1]) - set(payload)
        if missing:
            raise PresenceParseError(
                f"missing presence fields: {', '.join(sorted(missing))}",
                decoded=payload,
            )
        raw_activity = payload.get("activity")
    if isinstance(raw_activity, dict):
        summary = raw_activity.get("summary")
        focus = raw_activity.get("focus")
        if not isinstance(summary, str) or not summary.strip():
            raise PresenceParseError(
                "presence activity requires summary",
                decoded=payload,
            )
        if not isinstance(focus, str) or not focus.strip():
            raise PresenceParseError(
                "presence activity requires focus",
                decoded=payload,
            )
        if set(raw_activity) != set(PRESENCE_ACTIVITY_FIELDS):
            raise PresenceParseError(
                "unexpected presence activity fields",
                decoded=payload,
            )
    reason = None
    if not bootstrap and payload.get("continuation_reason") is not None:
        reason = _text(payload.get("continuation_reason"), 180) or None
    return PresenceProposal(
        _text(payload.get("decision"), 16).casefold(),
        _normalize_activity(raw_activity, bootstrap=bootstrap),
        reason,
        _normalize_emotion(payload.get("emotion")),
    )


def presence_proposal_rejection(
    state: PresenceState,
    proposal: PresenceProposal,
    *,
    bootstrap: bool | None = None,
) -> str:
    """Validate one normalized activity decision without appraising emotion."""

    first_decision = needs_bootstrap(state) if bootstrap is None else bootstrap
    if proposal.decision not in _DECISIONS:
        return "invalid presence decision"
    if first_decision and proposal.decision != "new":
        return "bootstrap requires decision=new"
    if proposal.decision == "new":
        if proposal.activity is None:
            return "new decision requires activity object"
        if not first_decision and proposal.continuation_reason is not None:
            return "new presence activity cannot have a continuation reason"
        return presence_activity_rejection(proposal.activity)
    if proposal.activity is not None:
        return "continued presence activity must be null"
    if state.current_activity is None:
        return "no current presence activity to continue"
    if state.continuation_count >= 1:
        return "presence activity already continued once"
    reason_words = re.findall(
        r"[a-z0-9]+",
        (proposal.continuation_reason or "").casefold(),
    )
    if len(reason_words) < 3:
        return "continuation needs a meaningful reason"
    return ""


def apply_presence_proposal(
    state: PresenceState,
    proposal: PresenceProposal,
    *,
    now: float,
    activity_id: str,
) -> PresenceState:
    next_decision_at = max(0.0, float(now)) + PRESENCE_INTERVAL_SECONDS
    current = state.current_activity
    previous = state.previous_activity
    continuation_count = state.continuation_count
    if proposal.decision == "new" and proposal.activity is not None:
        previous = current or previous
        current = PresenceActivity(
            activity_id,
            proposal.activity.summary,
            proposal.activity.focus,
            now,
            next_decision_at,
        )
        continuation_count = 0
    elif current is not None:
        current = replace(current, expected_end_at=next_decision_at)
        continuation_count = 1
    return PresenceState(
        current_activity=current,
        previous_activity=previous,
        last_decision_at=now,
        next_decision_at=next_decision_at,
        retry_at=0.0,
        last_error=None,
        claim_token=None,
        claim_expires_at=0.0,
        continuation_count=continuation_count,
    )


def format_presence_context(state: PresenceState) -> str:
    activity = state.current_activity
    if activity is None:
        return "Current activity: none."
    return "\n".join(
        (
            f"Current activity: {activity.summary}.",
            f"Current focus: {activity.focus}.",
        )
    )
