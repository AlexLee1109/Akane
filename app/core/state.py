"""Deterministic, profile-scoped emotional continuity for Akane."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from app.core.config import EMOTION_STATE_PATH
from app.core.persistence import atomic_write_json, read_json
from app.core.signal import TurnSignal, topic_overlap
from app.core.utils import clamp, compact_text

STATE_SCHEMA_VERSION = 1
_MAX_PROFILES = 128
_BASELINE = {
    "valence": 0.56,
    "arousal": 0.44,
    "warmth": 0.58,
    "social_ease": 0.62,
    "irritation": 0.08,
    "curiosity": 0.58,
    "familiarity": 0.22,
}
_CONTINUOUS_FIELDS = tuple(_BASELINE)


@dataclass(slots=True)
class AkaneState:
    schema_version: int = STATE_SCHEMA_VERSION
    valence: float = _BASELINE["valence"]
    arousal: float = _BASELINE["arousal"]
    warmth: float = _BASELINE["warmth"]
    social_ease: float = _BASELINE["social_ease"]
    irritation: float = _BASELINE["irritation"]
    curiosity: float = _BASELINE["curiosity"]
    familiarity: float = _BASELINE["familiarity"]
    recent_intent: str = "casual"
    recent_tone: str = "neutral"
    recent_topic: str = ""
    transition_reason: str = "baseline"
    inner_impulse: str = "remain attentive and contribute a perspective of her own"
    updated_at: float = 0.0

    @classmethod
    def from_dict(cls, payload: object) -> "AkaneState":
        if not isinstance(payload, dict):
            return cls()
        valid_names = {item.name for item in fields(cls)}
        values = {key: value for key, value in payload.items() if key in valid_names}
        try:
            state = cls(**values)
        except (TypeError, ValueError):
            return cls()
        state.schema_version = STATE_SCHEMA_VERSION
        for name in _CONTINUOUS_FIELDS:
            setattr(state, name, clamp(getattr(state, name), _BASELINE[name]))
        state.recent_intent = compact_text(state.recent_intent, 32) or "casual"
        state.recent_tone = compact_text(state.recent_tone, 32) or "neutral"
        state.recent_topic = compact_text(state.recent_topic, 80)
        state.transition_reason = compact_text(state.transition_reason, 80) or "loaded"
        state.inner_impulse = (
            compact_text(state.inner_impulse, 140)
            or "remain attentive and contribute a perspective of her own"
        )
        try:
            state.updated_at = max(0.0, float(state.updated_at))
        except (TypeError, ValueError):
            state.updated_at = 0.0
        return state

    def preview(self, signal: TurnSignal, *, now: float | None = None) -> "AkaneState":
        candidate = replace(self)
        candidate.update(signal, now=now)
        return candidate

    def update(self, signal: TurnSignal, *, now: float | None = None) -> None:
        current = time.time() if now is None else float(now)
        previous_topic = self.recent_topic
        topic_continuity = topic_overlap(previous_topic, signal.topic)
        topic_is_new = bool(
            signal.topic
            and not signal.low_content
            and (not previous_topic or topic_continuity < 0.35)
        )
        self._decay(current)
        self.recent_intent = signal.intent
        self.recent_tone = signal.tone
        self.recent_topic = compact_text(signal.topic, 80)

        if topic_is_new:
            self._nudge("curiosity", 0.032)
            self._nudge("arousal", 0.012)
            self._nudge("valence", 0.008)
        elif topic_continuity >= 0.50 and not signal.low_content:
            self._nudge("warmth", 0.008)
            self._nudge("social_ease", 0.008)
            self._nudge("familiarity", 0.006)

        if signal.technical:
            self._nudge("arousal", 0.025)
            self._nudge("curiosity", 0.035)
            self._toward("warmth", 0.54, 0.08)
        if signal.sadness:
            self._nudge("warmth", 0.025)
            self._nudge("arousal", -0.025)
            self._nudge("social_ease", -0.01)
        if signal.praise:
            self._nudge("valence", 0.045)
            self._nudge("warmth", 0.035)
            self._nudge("social_ease", 0.025)
            self._nudge("familiarity", 0.012)
        elif signal.friendliness:
            self._nudge("valence", 0.018)
            self._nudge("warmth", 0.018)
            self._nudge("familiarity", 0.008)
        if signal.teasing:
            self._nudge("arousal", 0.03)
            self._nudge("curiosity", 0.015)
            self._nudge("warmth", 0.01)
        if signal.criticism or signal.correction_requested:
            self._nudge("valence", -0.025)
            self._nudge("social_ease", -0.025)
            self._nudge("curiosity", 0.015)
        if signal.frustration and not signal.hostility:
            self._nudge("irritation", 0.012)
            self._nudge("arousal", 0.012)
        if signal.hostility:
            self._nudge("valence", -0.055)
            self._nudge("warmth", -0.045)
            self._nudge("social_ease", -0.055)
            self._nudge("irritation", 0.065)
            self._nudge("arousal", 0.035)

        self.transition_reason = _transition_reason(
            signal,
            topic_is_new=topic_is_new,
            topic_continuity=topic_continuity,
        )
        self.inner_impulse = _inner_impulse(
            self,
            signal,
            topic_is_new=topic_is_new,
            topic_continuity=topic_continuity,
        )
        self.updated_at = current

    def _decay(self, now: float) -> None:
        elapsed_hours = (
            max(0.0, now - self.updated_at) / 3600.0 if self.updated_at else 0.0
        )
        amount = min(0.28, 0.035 + elapsed_hours * 0.025)
        for name, baseline in _BASELINE.items():
            self._toward(name, baseline, amount)

    def _nudge(self, name: str, delta: float) -> None:
        setattr(self, name, clamp(getattr(self, name) + delta))

    def _toward(self, name: str, target: float, amount: float) -> None:
        current = float(getattr(self, name))
        setattr(self, name, clamp(current + (target - current) * amount))


def emotion_label(state: AkaneState) -> str:
    if state.irritation >= 0.30:
        return "mildly annoyed"
    if state.irritation >= 0.20 or (
        state.recent_intent == "hostility" and state.irritation >= 0.14
    ):
        return "guarded"
    if state.recent_intent == "teasing" and state.valence >= 0.52:
        return "playful"
    if state.recent_intent == "praise" and state.valence >= 0.58:
        return "quietly pleased"
    if state.recent_intent == "emotional_support":
        return "concerned"
    if state.arousal >= 0.58 and state.valence >= 0.58:
        return "bright"
    if state.curiosity >= 0.60:
        return "curious"
    if state.warmth >= 0.65 and state.valence >= 0.58:
        return "warm"
    if state.valence <= 0.43:
        return "subdued"
    if state.arousal <= 0.36:
        return "reflective"
    return "attentive"


def summarize_state(state: AkaneState, signal: TurnSignal | None = None) -> str:
    intent = signal.intent if signal else state.recent_intent
    parts: list[str] = [emotion_label(state)]
    if intent == "emotional_support":
        parts.append("steady rather than clinical")
    elif intent == "teasing":
        parts.append("ready to tease back lightly")
    elif intent in {"criticism", "correction", "hostility"}:
        parts.append("more direct and less eager to agree")

    if state.irritation >= 0.18:
        parts.append("some irritation is carrying over")
    elif state.warmth >= 0.66 and state.familiarity >= 0.28:
        parts.append("familiar warmth is carrying over")
    bearing = "; ".join(dict.fromkeys(parts))
    return (
        f"Current emotional bearing: {bearing}. "
        f"Private inclination: {state.inner_impulse}. Keep this implicit; express it "
        "through emphasis, rhythm, and what Akane chooses to add."
    )


def public_state(state: AkaneState) -> dict[str, object]:
    payload = {
        name: round(float(getattr(state, name)), 3)
        for name in _CONTINUOUS_FIELDS
    }
    payload.update(
        {
            "schema_version": state.schema_version,
            "mood": emotion_label(state),
            "recent_intent": state.recent_intent,
            "recent_tone": state.recent_tone,
            "recent_topic": state.recent_topic,
            "transition_reason": state.transition_reason,
            "inner_impulse": state.inner_impulse,
            "updated_at": state.updated_at,
        }
    )
    return payload


def _transition_reason(
    signal: TurnSignal,
    *,
    topic_is_new: bool,
    topic_continuity: float,
) -> str:
    topic = compact_text(signal.topic, 54) or "the conversation"
    if signal.hostility:
        return "guardedness rose after hostility"
    if signal.sadness:
        return "concern deepened around what the user shared"
    if signal.praise:
        return "pleasure and ease rose after earned praise"
    if signal.teasing:
        return "playful energy surfaced"
    if signal.criticism:
        return "social ease tightened after criticism"
    if signal.correction_requested:
        return "attention shifted to the user's correction"
    if signal.frustration:
        return "tension rose around the user's frustration"
    if signal.technical:
        return f"interest sharpened around {topic}"
    if topic_is_new:
        return f"curiosity sparked around {topic}"
    if topic_continuity >= 0.50 and not signal.low_content:
        return f"familiarity grew while staying with {topic}"
    if signal.low_content:
        return "emotion carried forward without forcing a new reaction"
    return "settled into the exchange on her own terms"


def _inner_impulse(
    state: AkaneState,
    signal: TurnSignal,
    *,
    topic_is_new: bool,
    topic_continuity: float,
) -> str:
    topic = compact_text(signal.topic, 54) or "the current thought"
    if signal.hostility:
        return "hold her ground without escalating"
    if signal.correction_requested or signal.criticism:
        return "reconsider the detail honestly instead of defending the old answer"
    if signal.sadness:
        return "stay close to what was actually said and respond with steady care"
    if signal.teasing:
        return "tease back lightly while keeping the real point intact"
    if signal.praise:
        return "enjoy the praise without becoming overly agreeable"
    if signal.technical:
        return "get absorbed in the concrete problem and offer a clear point of view"
    if state.irritation >= 0.20:
        return "answer honestly with a slightly sharper edge"
    if topic_is_new:
        return f"follow her curiosity about {topic} and add a perspective of her own"
    if topic_continuity >= 0.50 and not signal.low_content:
        return f"stay with {topic} and build on the shared thread"
    if signal.low_content:
        return state.inner_impulse
    return "respond from her own perspective instead of merely matching the user"


class EmotionStore:
    """Owns profile-scoped state and persists it only after successful turns."""

    def __init__(self, path: Path = EMOTION_STATE_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._states: dict[str, AkaneState] = {}
        self._load()

    def preview(self, profile_id: str, signal: TurnSignal) -> AkaneState:
        with self._lock:
            state = self._state(profile_id)
            self._prune()
            return state.preview(signal)

    def commit(self, profile_id: str, signal: TurnSignal) -> AkaneState:
        with self._lock:
            key = compact_text(profile_id, 120) or "local:owner"
            previous = replace(self._state(key))
            candidate = previous.preview(signal)
            self._states[key] = candidate
            self._prune()
            try:
                self._persist()
            except Exception:
                self._states[key] = previous
                raise
            return replace(candidate)

    def snapshot(self, profile_id: str) -> AkaneState:
        with self._lock:
            state = self._state(profile_id)
            self._prune()
            return replace(state)

    def clear(self, profile_id: str) -> None:
        with self._lock:
            self._states.pop(compact_text(profile_id, 120) or "local:owner", None)
            self._persist()

    def restore(self, profile_id: str, state: AkaneState) -> None:
        """Restore a prior snapshot when a later commit stage fails."""

        key = compact_text(profile_id, 120) or "local:owner"
        with self._lock:
            self._states[key] = replace(state)
            self._persist()

    def _state(self, profile_id: str) -> AkaneState:
        key = compact_text(profile_id, 120) or "local:owner"
        state = self._states.get(key)
        if state is None:
            state = AkaneState(updated_at=time.time())
            self._states[key] = state
        return state

    def _prune(self) -> None:
        if len(self._states) <= _MAX_PROFILES:
            return
        ordered = sorted(self._states.items(), key=lambda item: item[1].updated_at)
        for key, _state in ordered[: len(self._states) - _MAX_PROFILES]:
            self._states.pop(key, None)

    def _load(self) -> None:
        try:
            payload = read_json(self._path)
            if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != STATE_SCHEMA_VERSION:
                raise ValueError("unsupported schema")
            profiles = payload.get("profiles")
            if not isinstance(profiles, dict):
                raise ValueError("invalid profiles")
            self._states = {
                compact_text(key, 120): AkaneState.from_dict(value)
                for key, value in profiles.items()
                if compact_text(key, 120)
            }
            self._prune()
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as exc:
            print(f"[Akane:state] ignored corrupt state file ({type(exc).__name__})", flush=True)
            self._states = {}

    def _persist(self) -> None:
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "profiles": {key: asdict(value) for key, value in self._states.items()},
        }
        atomic_write_json(self._path, payload)


_EMOTIONS: EmotionStore | None = None
_EMOTIONS_LOCK = threading.Lock()


def get_emotion_store() -> EmotionStore:
    global _EMOTIONS
    if _EMOTIONS is None:
        with _EMOTIONS_LOCK:
            if _EMOTIONS is None:
                _EMOTIONS = EmotionStore()
    return _EMOTIONS
