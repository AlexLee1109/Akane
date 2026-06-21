"""Pure numeric rules for Akane's bounded emotion model."""

from __future__ import annotations

import math

MOODS = (
    "calm", "amused", "pleased", "focused", "curious", "smug",
    "pouty", "flustered", "proud", "concerned", "tired", "uncertain",
)
EMOTIONS = MOODS
BASELINES = {
    "warmth": 0.56,
    "comfort": 0.54,
    "playfulness": 0.32,
    "affection": 0.38,
    "curiosity": 0.46,
    "confidence": 0.58,
    "focus": 0.42,
    "energy": 0.56,
    "tension": 0.20,
    "irritation": 0.12,
    "uncertainty": 0.24,
    "fatigue": 0.18,
    "pride": 0.30,
    "fluster": 0.10,
    "smugness": 0.18,
    "concern": 0.22,
}
TONE_COLOR = {
    "calm": "grounded and natural",
    "amused": "quick, playful, with a lightly smug edge",
    "pleased": "warm and a little brighter",
    "focused": "precise, invested, and familiar",
    "curious": "alert, interested, and mentally present",
    "smug": "confident with restrained teasing",
    "pouty": "mildly resistant without becoming rude",
    "flustered": "softly hesitant with one small protest",
    "proud": "satisfied and slightly showy",
    "concerned": "careful, grounded, and less playful",
    "tired": "soft, dry, and lower-energy",
    "uncertain": "candid and careful without hedging excessively",
}
_HALF_LIFE = {
    "warmth": 6 * 3600,
    "comfort": 8 * 3600,
    "affection": 10 * 3600,
    "confidence": 4 * 3600,
    "energy": 3 * 3600,
    "fatigue": 3 * 3600,
}
_DEFAULT_HALF_LIFE = 75 * 60
_MAX_VALUE = 0.94
_PROBLEM_TERMS = (
    "bug", "broken", "doesn't work", "does not work", "error", "failed",
    "keeps happening", "not working", "stuck", "traceback",
)
_SUCCESS_TERMS = ("fixed", "finally works", "it works", "passed", "solved", "working now")
_DISTRESS_TERMS = ("awful day", "hurt", "lonely", "rough day", "sad", "scared", "upset")


def clamp(value: object, default: float = 0.0, high: float = _MAX_VALUE) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(high, number))


def valid(value: object, allowed: tuple[str, ...], default: str) -> str:
    name = str(value or "").strip().lower()
    return name if name in allowed else default


def decay(values: dict[str, float], elapsed: float) -> dict[str, float]:
    if elapsed <= 0:
        return dict(values)
    result: dict[str, float] = {}
    for name, baseline in BASELINES.items():
        half_life = _HALF_LIFE.get(name, _DEFAULT_HALF_LIFE)
        retained = math.pow(0.5, elapsed / half_life)
        result[name] = baseline + (values.get(name, baseline) - baseline) * retained
    return result


def apply_oppositions(values: dict[str, float]) -> None:
    values["energy"] = min(values["energy"], 0.96 - values["fatigue"] * 0.56)
    values["playfulness"] *= 1.0 - max(0.0, values["tension"] - 0.24) * 0.72
    values["confidence"] *= (
        1.0 - max(0.0, values["uncertainty"] - BASELINES["uncertainty"]) * 0.46
    )
    values["smugness"] = min(
        values["smugness"],
        0.12 + values["comfort"] * 0.48 + values["confidence"] * 0.28,
    )
    if values["irritation"] > 0.28:
        values["tension"] += (values["irritation"] - 0.28) * 0.28
        values["warmth"] -= (values["irritation"] - 0.28) * 0.12
    if values["concern"] > 0.35:
        values["focus"] += (values["concern"] - 0.35) * 0.18
        values["playfulness"] -= (values["concern"] - 0.35) * 0.20
    for name in BASELINES:
        values[name] = clamp(values[name])


def _mood_scores(values: dict[str, float]) -> dict[str, float]:
    return {
        "calm": 0.51 - values["tension"] * 0.10,
        "amused": values["playfulness"] + values["comfort"] * 0.20 - values["tension"] * 0.10,
        "pleased": values["warmth"] * 0.55 + values["pride"] * 0.25 + values["affection"] * 0.20,
        "focused": values["focus"] + values["concern"] * 0.20 - values["fatigue"] * 0.10,
        "curious": values["curiosity"] * 0.82 + values["energy"] * 0.10,
        "smug": values["smugness"] * 0.76 + values["confidence"] * 0.20,
        "pouty": values["irritation"] * 0.76 + values["tension"] * 0.20,
        "flustered": values["fluster"] * 0.84 + values["warmth"] * 0.10,
        "proud": values["pride"] * 0.78 + values["confidence"] * 0.16,
        "concerned": values["concern"] * 0.70 + values["tension"] * 0.28,
        "tired": values["fatigue"] * 0.72 + (1.0 - values["energy"]) * 0.28,
        "uncertain": values["uncertainty"] * 0.72 + (1.0 - values["confidence"]) * 0.24,
    }


def derive_moods(values: dict[str, float]) -> tuple[str, str, float]:
    scores = _mood_scores(values)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary, score = ranked[0]
    baseline_scores = _mood_scores(BASELINES)
    deviations = sorted(
        (
            (name, mood_score - baseline_scores[name])
            for name, mood_score in scores.items()
            if name not in {primary, "calm"}
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    fallback = "pleased" if primary == "curious" else "curious"
    secondary = deviations[0][0] if deviations and deviations[0][1] > 0.015 else fallback
    return primary, secondary, clamp((score - 0.36) / 0.44, high=1.0)


def event_deltas(
    text: str,
    attention: dict[str, object],
    failure_streak: int,
    comfort: float,
) -> tuple[str, dict[str, float], int]:
    lower = str(text or "").lower()
    deltas: dict[str, float] = {}

    def add(name: str, amount: float) -> None:
        deltas[name] = deltas.get(name, 0.0) + amount

    praise = bool(attention.get("praise"))
    teasing = bool(attention.get("teasing"))
    frustrated = bool(attention.get("frustrated")) or any(
        phrase in lower for phrase in _PROBLEM_TERMS
    )
    code_help = bool(attention.get("code_help"))
    gratitude = attention.get("intent") == "gratitude"
    success = any(phrase in lower for phrase in _SUCCESS_TERMS)
    distress = any(phrase in lower for phrase in _DISTRESS_TERMS)
    trigger = str(attention.get("emotional_trigger") or "")

    if praise:
        trigger = "praise"
        for name, amount in (
            ("warmth", 0.08), ("pride", 0.13), ("fluster", 0.09),
            ("affection", 0.04), ("smugness", 0.04),
        ):
            add(name, amount)
    if teasing:
        trigger = trigger or "teasing"
        add("playfulness", 0.11)
        add("smugness", 0.04 + max(0.0, comfort - 0.40) * 0.16)
        add("fluster", 0.07 - min(0.04, max(0.0, comfort - 0.40) * 0.12))
        if comfort < 0.30:
            add("irritation", 0.04)
    if gratitude:
        trigger = "thanks"
        for name, amount in (
            ("warmth", 0.05), ("comfort", 0.04), ("affection", 0.03), ("tension", -0.03),
        ):
            add(name, amount)
    if success:
        trigger = "success"
        for name, amount in (
            ("pride", 0.12), ("confidence", 0.08), ("warmth", 0.04),
            ("tension", -0.09), ("concern", -0.07), ("fatigue", -0.03),
        ):
            add(name, amount)
        failure_streak = 0
    elif frustrated:
        failure_streak = min(6, failure_streak + 1)
        trigger = "repeated_problem" if failure_streak > 1 else "code_problem"
        for name, amount in (
            ("focus", 0.13), ("concern", 0.11), ("irritation", 0.025),
            ("fatigue", 0.012 * failure_streak),
            ("tension", 0.07 + max(0, failure_streak - 1) * 0.025),
        ):
            add(name, amount)
        if failure_streak > 1:
            add("confidence", -0.035 * min(3, failure_streak - 1))
            add("uncertainty", 0.045 * min(3, failure_streak - 1))
    elif code_help:
        trigger = trigger or "coding_task"
        add("focus", 0.12)
        add("curiosity", 0.04)
        add("playfulness", -0.025)
        failure_streak = max(0, failure_streak - 1)
    else:
        failure_streak = max(0, failure_streak - 1)
    if distress:
        trigger = "user_distress"
        for name, amount in (
            ("concern", 0.13), ("warmth", 0.06), ("affection", 0.04), ("playfulness", -0.08),
        ):
            add(name, amount)
    if attention.get("intent") == "identity":
        trigger = trigger or "identity_interest"
        add("curiosity", 0.07)
        add("confidence", 0.04)
    return trigger, deltas, failure_streak
