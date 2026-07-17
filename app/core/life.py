"""Deterministic, bounded continuity for Akane's simulated internal life."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass

_MIN_ACTIVITY_SECONDS = 2 * 60 * 60
_ACTIVITIES = (
    (
        "quietly reflecting",
        "turning over a familiar idea",
        "thoughtful",
        "came away with a clearer view of the idea",
    ),
    (
        "organizing remembered details",
        "putting a few memories into a clearer order",
        "focused",
        "left a few remembered details feeling more settled",
    ),
    (
        "following a line of curiosity",
        "considering one question from a few angles",
        "curious",
        "found a more interesting angle on the question",
    ),
    (
        "letting her attention settle",
        "holding a quiet, unhurried focus",
        "calm",
        "finished with steadier attention",
    ),
    (
        "revisiting an unresolved idea",
        "giving an unfinished question another quiet pass",
        "thoughtful",
        "made the unfinished idea a little clearer",
    ),
    (
        "shaping an opinion",
        "deciding what she actually thinks about a familiar subject",
        "confident",
        "came away with a more definite opinion",
    ),
)
_ACTIVITY_BY_NAME = {spec[0]: spec for spec in _ACTIVITIES}


@dataclass(frozen=True, slots=True)
class LifeState:
    current_activity: str
    activity_detail: str
    mood: str
    started_time: float
    previous_activity: str
    previous_outcome: str
    last_update_time: float

    @classmethod
    def from_dict(cls, payload: object) -> "LifeState | None":
        if not isinstance(payload, dict):
            return None
        activity = str(payload.get("current_activity") or "").strip()
        spec = _ACTIVITY_BY_NAME.get(activity)
        previous = str(payload.get("previous_activity") or "").strip()
        if spec is None or previous and previous not in _ACTIVITY_BY_NAME:
            return None
        try:
            started = max(0.0, float(payload.get("started_time") or 0.0))
            updated = max(0.0, float(payload.get("last_update_time") or 0.0))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(started) or not math.isfinite(updated):
            return None
        _name, detail, mood, _outcome = spec
        return cls(
            activity,
            detail,
            mood,
            started,
            previous,
            _ACTIVITY_BY_NAME[previous][3] if previous else "",
            updated,
        )


def advance_life_state(
    state: LifeState | None,
    *,
    now: float,
    preferences: tuple[str, ...] = (),
    emotion_mood: str = "steady",
    energy: float = 0.65,
    unresolved_interests: tuple[str, ...] = (),
) -> LifeState:
    """Advance at most once after a varied, believable activity duration."""

    current_time = max(0.0, float(now))
    preference_key = "\n".join(
        sorted({str(item).strip() for item in preferences if str(item).strip()})
    )
    unresolved_key = "\n".join(
        sorted({str(item).strip() for item in unresolved_interests if str(item).strip()})
    )
    context_key = f"{preference_key}|{unresolved_key}|{emotion_mood}"

    if state is None:
        chosen = _choose_activity(
            context_key,
            current_time,
            emotion_mood,
            energy,
            bool(unresolved_key),
            excluded=(),
        )
        activity, detail, mood, _outcome = chosen
        return LifeState(activity, detail, mood, current_time, "", "", current_time)

    seed = _seed(f"{state.current_activity}|{state.started_time:.0f}|{context_key}")
    duration = _activity_duration(state, seed, energy)
    if current_time - state.started_time < duration:
        return state

    chosen = _choose_activity(
        context_key,
        current_time,
        emotion_mood,
        energy,
        bool(unresolved_key),
        excluded=(state.current_activity, state.previous_activity),
    )
    activity, detail, mood, _outcome = chosen
    previous_spec = _ACTIVITY_BY_NAME[state.current_activity]
    return LifeState(
        activity,
        detail,
        mood,
        current_time,
        state.current_activity,
        previous_spec[3],
        current_time,
    )


def format_life_state(state: LifeState) -> str:
    """Format only natural activity continuity, never persistence metadata."""

    parts = [
        f"Before this message, Akane was {state.current_activity}, {state.activity_detail}."
    ]
    if state.previous_activity:
        previous = f"Earlier, she was {state.previous_activity}"
        if state.previous_outcome:
            previous += f" and {state.previous_outcome}"
        parts.append(previous + ".")
    parts.append(
        "Only these supplied activities or an activity explicitly established in memory may "
        "be mentioned. Treat this as silent continuity and do not add another activity."
    )
    return " ".join(parts)


def _activity_duration(state: LifeState, seed: int, energy: float) -> float:
    varied = _MIN_ACTIVITY_SECONDS + (seed % 7) * 25 * 60
    if energy < 0.42 or state.mood == "calm":
        varied += 45 * 60
    elif energy > 0.76:
        varied -= 20 * 60
    return max(_MIN_ACTIVITY_SECONDS, varied)


def _choose_activity(
    context_key: str,
    now: float,
    emotion_mood: str,
    energy: float,
    has_unresolved_interest: bool,
    *,
    excluded: tuple[str, ...],
) -> tuple[str, str, str, str]:
    hour = time.localtime(now).tm_hour
    available = [spec for spec in _ACTIVITIES if spec[0] not in excluded]
    if not available:
        available = list(_ACTIVITIES)

    def score(spec: tuple[str, str, str, str]) -> float:
        activity, _detail, mood, _outcome = spec
        value = (_seed(f"{context_key}|{int(now // 3600)}|{activity}") % 10_000) / 10_000
        if has_unresolved_interest and activity == "revisiting an unresolved idea":
            value += 0.34
        if emotion_mood in {"strained", "weary"} and mood == "calm":
            value += 0.26
        if emotion_mood in {"bright", "pensive"} and mood in {"curious", "thoughtful"}:
            value += 0.18
        if energy >= 0.72 and mood in {"curious", "confident", "focused"}:
            value += 0.18
        if energy <= 0.44 and mood in {"calm", "thoughtful"}:
            value += 0.18
        if (hour < 7 or hour >= 23) and mood == "calm":
            value += 0.14
        return value

    return max(available, key=lambda spec: (score(spec), spec[0]))


def _seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
