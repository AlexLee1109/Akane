"""Canonical Ambient Presence state and grounded appraisal contract."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from typing import Literal

from app.core.utils import compact_text

PRESENCE_DURATION_SECONDS = {
    "quiet": 6 * 60 * 60,
    "reflecting_on_shared_thread": 4 * 60 * 60,
    "revisiting_interest": 3 * 60 * 60,
    "reconsidering_opinion": 3 * 60 * 60,
    "following_unfinished_thought": 4 * 60 * 60,
}
MIN_QUIET_INTERVALS = 2
MAX_SOURCE_SELECTIONS = 3
CLAIM_SECONDS = 2 * 60 * 60
RETRY_SECONDS = 5 * 60
RECENT_PREVIOUS_ACTIVITY_SECONDS = 24 * 60 * 60

PRESENCE_KINDS = frozenset(
    {
        "quiet",
        "reflecting_on_shared_thread",
        "revisiting_interest",
        "reconsidering_opinion",
        "following_unfinished_thought",
        "legacy",
    }
)
PRESENCE_SUBJECT_KINDS = frozenset(
    {"none", "interest", "opinion", "akane_experience", "shared_thread", "legacy_activity"}
)
PRESENCE_APPRAISAL_FIELDS = ("emotion", "experience")
PRESENCE_EMOTION_FIELDS = ("primary", "intensity", "cause")
PRESENCE_EXPERIENCE_FIELDS = (
    "kind",
    "meaning",
    "operation",
    "target_id",
    "source_ids",
    "summary",
    "topic",
    "position",
    "reason",
    "confidence",
)
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


def _nullable(kind: str) -> dict[str, object]:
    return {"type": [kind, "null"]}


PRESENCE_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
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
            "experience": {
                "type": ["object", "null"],
                "properties": {
                    "kind": {"type": "string", "enum": ["memory", "interest", "opinion"]},
                    "meaning": {
                        "type": "string",
                        "enum": [
                            "reflection",
                            "interest_shift",
                            "opinion_shift",
                            "connection",
                            "unfinished_thought",
                        ],
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["add", "reinforce", "weaken", "update", "reconsider"],
                    },
                    "target_id": {"type": "string"},
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 8,
                    },
                    "summary": {"type": "string"},
                    "topic": _nullable("string"),
                    "position": _nullable("string"),
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": list(PRESENCE_EXPERIENCE_FIELDS),
                "additionalProperties": False,
            },
        },
        "required": list(PRESENCE_APPRAISAL_FIELDS),
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


@dataclass(frozen=True, slots=True)
class PresenceActivity:
    """One provenance-backed digital orientation interval."""

    activity_id: str
    kind: str
    subject: str
    subject_kind: str
    source_ids: tuple[str, ...]
    started_at: float
    expected_end_at: float
    origin: str
    grounding_confidence: float

    @classmethod
    def from_dict(cls, payload: object) -> "PresenceActivity | None":
        if not isinstance(payload, dict):
            return None
        activity_id = _text(payload.get("activity_id"), 80)
        kind = _text(payload.get("kind"), 48).casefold()
        subject = _text(payload.get("subject"), 360)
        subject_kind = _text(payload.get("subject_kind"), 40).casefold()
        raw_sources = payload.get("source_ids")
        source_ids = tuple(
            dict.fromkeys(
                source
                for item in (raw_sources if isinstance(raw_sources, (list, tuple)) else ())
                if (source := _text(item, 180))
            )
        )[-8:]
        started_at = _timestamp(payload.get("started_at"))
        expected_end_at = _timestamp(payload.get("expected_end_at"))
        origin = _text(payload.get("origin"), 48).casefold()
        confidence = payload.get("grounding_confidence")
        if (
            not activity_id
            or kind not in PRESENCE_KINDS
            or subject_kind not in PRESENCE_SUBJECT_KINDS
            or expected_end_at <= started_at
            or not origin
            or type(confidence) not in {int, float}
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            return None
        if kind == "quiet":
            if subject or subject_kind != "none" or source_ids:
                return None
        elif not subject or not source_ids or subject_kind == "none":
            return None
        if kind == "legacy" and origin != "legacy_presence":
            return None
        return cls(
            activity_id,
            kind,
            subject,
            subject_kind,
            source_ids,
            started_at,
            expected_end_at,
            origin,
            float(confidence),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "activity_id": self.activity_id,
            "kind": self.kind,
            "subject": self.subject,
            "subject_kind": self.subject_kind,
            "source_ids": list(self.source_ids),
            "started_at": self.started_at,
            "expected_end_at": self.expected_end_at,
            "origin": self.origin,
            "grounding_confidence": self.grounding_confidence,
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
    repetition_count: int = 0
    recent_source_ids: tuple[str, ...] = ()
    quiet_streak: int = 0
    last_transition_reason: str = ""
    last_candidate_score: float | None = None
    last_candidate_source_id: str | None = None
    last_appraised_activity_id: str | None = None
    last_appraised_at: float = 0.0
    last_appraisal_result: str = ""

    @classmethod
    def from_dict(cls, payload: object, *, now: float | None = None) -> "PresenceState":
        current_time = time.time() if now is None else max(0.0, float(now))
        values = payload if isinstance(payload, dict) else {}
        token = _text(values.get("claim_token"), 80) or None
        claim_expires_at = _timestamp(values.get("claim_expires_at"))
        if token is None or claim_expires_at <= current_time:
            token = None
            claim_expires_at = 0.0
        raw_count = values.get("repetition_count", 0)
        repetition_count = max(0, min(3, raw_count)) if type(raw_count) is int else 0
        raw_recent = values.get("recent_source_ids")
        recent_source_ids = tuple(
            source
            for item in (raw_recent if isinstance(raw_recent, (list, tuple)) else ())
            if (source := _text(item, 180))
        )[-6:]
        raw_quiet_streak = values.get("quiet_streak", 0)
        quiet_streak = (
            max(0, min(3, raw_quiet_streak))
            if type(raw_quiet_streak) is int
            else 0
        )
        raw_score = values.get("last_candidate_score")
        last_candidate_score = (
            float(raw_score)
            if type(raw_score) in {int, float} and math.isfinite(float(raw_score))
            else None
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
            repetition_count=repetition_count,
            recent_source_ids=recent_source_ids,
            quiet_streak=quiet_streak,
            last_transition_reason=_text(values.get("last_transition_reason"), 80),
            last_candidate_score=last_candidate_score,
            last_candidate_source_id=(
                _text(values.get("last_candidate_source_id"), 180) or None
            ),
            last_appraised_activity_id=(
                _text(values.get("last_appraised_activity_id"), 80) or None
            ),
            last_appraised_at=_timestamp(values.get("last_appraised_at")),
            last_appraisal_result=_text(values.get("last_appraisal_result"), 32),
        )
        return normalize_presence(state, now=current_time)

    def as_dict(self) -> dict[str, object]:
        return {
            "current_activity": self.current_activity.as_dict() if self.current_activity else None,
            "previous_activity": self.previous_activity.as_dict() if self.previous_activity else None,
            "last_decision_at": self.last_decision_at,
            "next_decision_at": self.next_decision_at,
            "retry_at": self.retry_at,
            "last_error": self.last_error,
            "claim_token": self.claim_token,
            "claim_expires_at": self.claim_expires_at,
            "repetition_count": self.repetition_count,
            "recent_source_ids": list(self.recent_source_ids),
            "quiet_streak": self.quiet_streak,
            "last_transition_reason": self.last_transition_reason,
            "last_candidate_score": self.last_candidate_score,
            "last_candidate_source_id": self.last_candidate_source_id,
            "last_appraised_activity_id": self.last_appraised_activity_id,
            "last_appraised_at": self.last_appraised_at,
            "last_appraisal_result": self.last_appraisal_result,
        }


def activity_status(activity: PresenceActivity | None, *, now: float) -> str:
    if activity is None:
        return "none"
    if activity.expected_end_at <= now:
        return "expired"
    if activity.started_at <= now < activity.expected_end_at:
        return "active"
    return "none"


def is_activity_active(activity: PresenceActivity | None, *, now: float) -> bool:
    return activity_status(activity, now=now) == "active"


def normalize_presence(state: PresenceState, *, now: float) -> PresenceState:
    """Return the sole temporal interpretation of persisted Presence state."""

    current = state.current_activity
    previous = state.previous_activity
    next_decision_at = state.next_decision_at
    status = activity_status(current, now=now)
    if status != "active":
        if status == "expired" and current is not None and (
            previous is None or previous.activity_id != current.activity_id
        ):
            previous = current
        current = None
        next_decision_at = 0.0
    elif current is not None and (
        next_decision_at <= 0.0 or next_decision_at != current.expected_end_at
    ):
        next_decision_at = current.expected_end_at
    token = state.claim_token
    claim_expires_at = state.claim_expires_at
    if token is not None and claim_expires_at <= now:
        token = None
        claim_expires_at = 0.0
    return replace(
        state,
        current_activity=current,
        previous_activity=previous,
        next_decision_at=next_decision_at,
        claim_token=token,
        claim_expires_at=claim_expires_at,
    )


@dataclass(frozen=True, slots=True)
class PresenceCandidate:
    kind: str
    subject: str
    subject_kind: str
    source_ids: tuple[str, ...]
    origin: str
    grounding_confidence: float
    updated_at: float
    salience: float
    interest: float = 0.0
    unresolved: bool = False

    @property
    def primary_source_id(self) -> str:
        return self.source_ids[0] if self.source_ids else ""


@dataclass(frozen=True, slots=True)
class PresenceCandidateScore:
    total: float
    grounding: float
    continuity: float
    salience: float
    interest: float
    recency: float
    novelty: float
    emotion: float
    repetition_penalty: float
    staleness_penalty: float


@dataclass(frozen=True, slots=True)
class PresenceSelection:
    candidate: PresenceCandidate | None
    score: PresenceCandidateScore | None
    reason: str
    continue_current: bool = False
    reset_repetition: bool = False


def score_presence_candidate(
    candidate: PresenceCandidate,
    state: PresenceState,
    *,
    now: float,
    emotion_weight: float = 0.0,
) -> PresenceCandidateScore:
    """Score one already-grounded candidate; no randomness or topic invention."""

    previous = state.current_activity or state.previous_activity
    same_source = bool(
        previous
        and previous.kind not in {"quiet", "legacy"}
        and previous.source_ids
        and previous.source_ids[0] == candidate.primary_source_id
    )
    continuity = 1.0 if same_source and candidate.unresolved else 0.65 if same_source else 0.0
    age_days = max(0.0, now - max(0.0, candidate.updated_at)) / 86_400.0
    recency = 0.0 if candidate.updated_at <= 0.0 else 1.0 / (1.0 + age_days / 30.0)
    novelty = 0.0 if same_source else 1.0
    repeats = sum(
        source_id == candidate.primary_source_id for source_id in state.recent_source_ids
    )
    repetition_penalty = 12.0 * min(3, repeats)
    staleness = 0.65 if candidate.updated_at <= 0.0 else 1.0 - recency
    staleness_penalty = 10.0 * staleness
    grounding = max(0.0, min(1.0, candidate.grounding_confidence))
    salience = max(0.0, min(1.0, candidate.salience))
    interest = max(0.0, min(1.0, candidate.interest))
    emotion = max(0.0, min(1.0, emotion_weight))
    total = (
        100.0 * grounding
        + 24.0 * continuity
        + 12.0 * salience
        + 8.0 * interest
        + 4.0 * recency
        + 2.0 * novelty
        + emotion
        - repetition_penalty
        - staleness_penalty
    )
    return PresenceCandidateScore(
        total,
        grounding,
        continuity,
        salience,
        interest,
        recency,
        novelty,
        emotion,
        repetition_penalty,
        staleness_penalty,
    )


def _rank_presence_candidates(
    candidates: tuple[PresenceCandidate, ...],
    state: PresenceState,
    *,
    now: float,
    emotion_weights: dict[str, float] | None = None,
) -> list[tuple[PresenceCandidateScore, PresenceCandidate]]:
    weights = emotion_weights or {}
    scored = [
        (
            score_presence_candidate(
                candidate,
                state,
                now=now,
                emotion_weight=weights.get(candidate.kind, 0.0),
            ),
            candidate,
        )
        for candidate in candidates
        if candidate.kind in PRESENCE_KINDS - {"quiet", "legacy"}
        and candidate.subject
        and candidate.source_ids
        and candidate.grounding_confidence >= 0.50
    ]
    scored.sort(
        key=lambda item: (
            -item[0].total,
            item[1].kind,
            item[1].subject_kind,
            item[1].subject.casefold(),
            item[1].primary_source_id,
        )
    )
    return [item for item in scored if item[0].total >= 60.0]


def choose_presence_transition(
    candidates: tuple[PresenceCandidate, ...],
    state: PresenceState,
    *,
    now: float,
    emotion_weights: dict[str, float] | None = None,
) -> PresenceSelection:
    """Choose the due transition without inventing a topic or replacing valid state."""

    completed = state.current_activity or state.previous_activity
    ranked = _rank_presence_candidates(
        candidates,
        state,
        now=now,
        emotion_weights=emotion_weights,
    )
    top_score, top_candidate = ranked[0] if ranked else (None, None)
    if completed is None:
        return PresenceSelection(None, top_score, "initial_quiet")
    if completed.kind == "legacy":
        return PresenceSelection(None, top_score, "legacy_expired")

    if completed.kind != "quiet":
        continuing = next(
            (
                (score, candidate)
                for score, candidate in ranked
                if candidate.unresolved
                and completed.source_ids
                and candidate.primary_source_id == completed.source_ids[0]
            ),
            None,
        )
        if continuing is not None:
            score, candidate = continuing
            refreshed = candidate.updated_at > state.last_decision_at
            if state.repetition_count < MAX_SOURCE_SELECTIONS or refreshed:
                return PresenceSelection(
                    candidate,
                    score,
                    "unresolved_continuity",
                    continue_current=True,
                    reset_repetition=refreshed,
                )
        reason = "stale_subject_to_quiet" if continuing is not None else "active_completed_to_quiet"
        return PresenceSelection(None, top_score, reason)

    eligible: tuple[PresenceCandidateScore, PresenceCandidate] | None = None
    saw_stale = False
    for score, candidate in ranked:
        same_recent_source = bool(
            state.recent_source_ids
            and state.recent_source_ids[-1] == candidate.primary_source_id
        )
        refreshed = candidate.updated_at > state.last_decision_at
        if (
            same_recent_source
            and state.repetition_count >= MAX_SOURCE_SELECTIONS
            and not refreshed
        ):
            saw_stale = True
            continue
        eligible = (score, candidate)
        break
    if eligible is None:
        return PresenceSelection(
            None,
            top_score,
            "quiet_stale_subject" if saw_stale else "quiet_no_candidate",
            continue_current=True,
        )
    score, candidate = eligible
    refreshed = candidate.updated_at > state.last_decision_at
    if state.quiet_streak < MIN_QUIET_INTERVALS and not refreshed:
        return PresenceSelection(
            None,
            score,
            "quiet_minimum",
            continue_current=True,
        )
    return PresenceSelection(
        candidate,
        score,
        "refreshed_candidate" if refreshed else "grounded_candidate",
        reset_repetition=refreshed,
    )


def select_presence_candidate(
    candidates: tuple[PresenceCandidate, ...],
    state: PresenceState,
    *,
    now: float,
    emotion_weights: dict[str, float] | None = None,
) -> PresenceCandidate | None:
    """Compatibility view of the deterministic due transition."""

    return choose_presence_transition(
        candidates,
        state,
        now=now,
        emotion_weights=emotion_weights,
    ).candidate


def presence_duration_seconds(kind: str) -> float:
    return float(PRESENCE_DURATION_SECONDS.get(kind, 4 * 60 * 60))


def make_presence_activity(
    candidate: PresenceCandidate | None,
    *,
    now: float,
    activity_id: str,
    existing: PresenceActivity | None = None,
) -> PresenceActivity:
    kind = candidate.kind if candidate is not None else "quiet"
    expected_end_at = now + presence_duration_seconds(kind)
    if existing is not None and (
        (candidate is None and existing.kind == "quiet")
        or (
            candidate is not None
            and existing.kind == candidate.kind
            and existing.source_ids
            and existing.source_ids[0] == candidate.primary_source_id
        )
    ):
        return replace(
            existing,
            subject=candidate.subject if candidate is not None else "",
            subject_kind=candidate.subject_kind if candidate is not None else "none",
            source_ids=candidate.source_ids if candidate is not None else (),
            expected_end_at=expected_end_at,
            origin=candidate.origin if candidate is not None else "deterministic",
            grounding_confidence=(
                candidate.grounding_confidence if candidate is not None else 1.0
            ),
        )
    if candidate is None:
        return PresenceActivity(
            activity_id,
            "quiet",
            "",
            "none",
            (),
            now,
            expected_end_at,
            "deterministic",
            1.0,
        )
    return PresenceActivity(
        activity_id,
        candidate.kind,
        candidate.subject,
        candidate.subject_kind,
        candidate.source_ids,
        now,
        expected_end_at,
        candidate.origin,
        candidate.grounding_confidence,
    )


@dataclass(frozen=True, slots=True)
class ProposedEmotion:
    primary: str
    intensity: float
    cause: str


@dataclass(frozen=True, slots=True)
class PresenceExperience:
    kind: str
    meaning: str
    operation: str
    target_id: str
    source_ids: tuple[str, ...]
    summary: str
    topic: str | None
    position: str | None
    reason: str
    confidence: float


@dataclass(frozen=True, slots=True)
class PresenceAppraisal:
    emotion: ProposedEmotion | None
    experience: PresenceExperience | None


class PresenceParseError(ValueError):
    def __init__(self, message: str, *, decoded: object = None) -> None:
        super().__init__(message)
        self.decoded = decoded


def _normalize_emotion(payload: object) -> ProposedEmotion | None:
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != set(PRESENCE_EMOTION_FIELDS):
        raise PresenceParseError("unexpected presence emotion fields", decoded=payload)
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
        raise PresenceParseError("presence emotion is invalid", decoded=payload)
    return ProposedEmotion(primary, float(intensity), cause)


def _normalize_experience(payload: object) -> PresenceExperience | None:
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != set(PRESENCE_EXPERIENCE_FIELDS):
        raise PresenceParseError("unexpected presence experience fields", decoded=payload)
    kind = _text(payload.get("kind"), 16).casefold()
    meaning = _text(payload.get("meaning"), 32).casefold()
    operation = _text(payload.get("operation"), 20).casefold()
    target_id = _text(payload.get("target_id"), 180)
    raw_sources = payload.get("source_ids")
    source_ids = tuple(
        dict.fromkeys(
            source
            for item in (raw_sources if isinstance(raw_sources, list) else ())
            if (source := _text(item, 180))
        )
    )
    summary = _text(payload.get("summary"), 360)
    topic = _text(payload.get("topic"), 100) or None
    position = _text(payload.get("position"), 200) or None
    reason = _text(payload.get("reason"), 240)
    confidence = payload.get("confidence")
    if (
        kind not in {"memory", "interest", "opinion"}
        or not target_id
        or not source_ids
        or not summary
        or not reason
        or type(confidence) not in {int, float}
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
        or kind == "memory"
        and (
            meaning not in {"connection", "unfinished_thought"}
            or operation != "add"
            or topic is not None
            or position is not None
        )
        or kind == "interest"
        and (
            meaning != "interest_shift"
            or operation not in {"reinforce", "weaken", "update"}
            or topic is None
            or position is not None
        )
        or kind == "opinion"
        and (
            meaning not in {"reflection", "opinion_shift"}
            or operation not in {"reinforce", "weaken", "update", "reconsider"}
            or topic is None
            or position is None
        )
    ):
        raise PresenceParseError("presence experience is invalid", decoded=payload)
    return PresenceExperience(
        kind,
        meaning,
        operation,
        target_id,
        source_ids,
        summary,
        topic,
        position,
        reason,
        float(confidence),
    )


def parse_presence_appraisal(output: object) -> PresenceAppraisal:
    text = str(output or "").strip()
    if not text:
        raise PresenceParseError("empty presence appraisal")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PresenceParseError(
            f"presence JSON decode failed at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(payload, dict):
        raise PresenceParseError("presence appraisal must be a JSON object", decoded=payload)
    if set(payload) != set(PRESENCE_APPRAISAL_FIELDS):
        raise PresenceParseError("unexpected presence appraisal fields", decoded=payload)
    return PresenceAppraisal(
        _normalize_emotion(payload.get("emotion")),
        _normalize_experience(payload.get("experience")),
    )


@dataclass(frozen=True, slots=True)
class PresenceView:
    status: Literal["active", "previous", "none"]
    current_activity: PresenceActivity | None = None
    previous_activity: PresenceActivity | None = None


def presence_view(state: PresenceState, *, now: float) -> PresenceView:
    normalized = normalize_presence(state, now=now)
    if normalized.current_activity is not None:
        return PresenceView("active", current_activity=normalized.current_activity)
    previous = normalized.previous_activity
    if previous is not None and 0.0 <= now - previous.expected_end_at <= RECENT_PREVIOUS_ACTIVITY_SECONDS:
        return PresenceView("previous", previous_activity=previous)
    return PresenceView("none")


def _orientation_lines(activity: PresenceActivity, *, previous: bool = False) -> tuple[str, ...]:
    prefix = "Previous " if previous else ""
    if activity.kind == "quiet":
        return (f"{prefix}Orientation: quiet",)
    return (
        f"{prefix}Orientation kind: {activity.kind}",
        f"{prefix}Subject: {activity.subject}",
        f"{prefix}Subject kind: {activity.subject_kind}",
    )


def format_presence_context(state: PresenceState, *, now: float) -> str:
    view = presence_view(state, now=now)
    if view.current_activity is not None:
        activity = view.current_activity
        status = "Status: quiet and available" if activity.kind == "quiet" else "Status: active"
        return "\n".join((status, *_orientation_lines(activity)))
    if view.previous_activity is not None:
        previous = view.previous_activity
        return "\n".join(
            (
                "Status: no current orientation",
                *_orientation_lines(previous, previous=True),
                "Previous orientation completed earlier.",
            )
        )
    return "Status: quiet and available; no recent recorded orientation"


def format_presence_prompt_context(
    state: PresenceState,
    *,
    now: float,
    include_previous: bool = False,
) -> str:
    """Render only compact private background state for normal dialogue."""

    view = presence_view(state, now=now)
    current = view.current_activity
    if current is None or current.kind == "legacy":
        lines = ["kind=quiet"]
    else:
        lines = [f"kind={current.kind}"]
        if current.subject:
            lines.append(f"subject={current.subject}")
        lines.append(f"started_at={current.started_at:.3f}")
    previous = view.previous_activity
    if include_previous and previous is not None and previous.kind not in {"quiet", "legacy"}:
        lines.extend(
            (
                f"previous_kind={previous.kind}",
                f"previous_subject={previous.subject}",
                f"previous_ended_at={previous.expected_end_at:.3f}",
            )
        )
    return "\n".join(lines)
