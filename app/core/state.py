"""Legacy emotional-state API retained for compatibility outside the chat runtime.

The active runtime persists ``signal.EmotionState`` through ``LongTermMemoryStore``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Iterable

from app.core.config import EMOTION_STATE_PATH
from app.core.persistence import atomic_write_json, read_json
from app.core.signal import TurnSignal, topic_overlap
from app.core.utils import clamp, compact_text

STATE_SCHEMA_VERSION = 3
_MAX_PROFILES = 128
_NEUTRAL_ENERGY = 0.5


@dataclass(frozen=True, slots=True)
class _Reaction:
    primary: str
    secondary: str | None
    intensity: float
    valence: float
    energy: float
    confidence: float
    cause: str


@dataclass(slots=True)
class AkaneState:
    """A reaction with continuity, not a personality mode or response template."""

    schema_version: int = STATE_SCHEMA_VERSION
    primary: str | None = None
    secondary: str | None = None
    intensity: float = 0.0
    valence: float = 0.0
    energy: float = _NEUTRAL_ENERGY
    confidence: float = 0.0
    cause: str | None = None
    momentum: float = 0.0
    recent_topic: str = ""
    turns_since_cause: int = 0
    updated_at: float = 0.0

    @classmethod
    def from_dict(cls, payload: object) -> "AkaneState":
        if not isinstance(payload, dict):
            return cls()
        if any(name in payload for name in ("warmth", "playfulness", "concern", "irritation")):
            return _migrate_delivery_state(payload)
        valid_names = {item.name for item in fields(cls)}
        values = {key: value for key, value in payload.items() if key in valid_names}
        try:
            state = cls(**values)
        except (TypeError, ValueError):
            return cls()
        state.schema_version = STATE_SCHEMA_VERSION
        state.primary = compact_text(state.primary, 24).lower() or None
        state.secondary = compact_text(state.secondary, 24).lower() or None
        if state.secondary == state.primary:
            state.secondary = None
        state.intensity = clamp(state.intensity)
        state.valence = clamp(state.valence, 0.0, -1.0, 1.0)
        state.energy = clamp(state.energy, _NEUTRAL_ENERGY)
        state.confidence = clamp(state.confidence)
        state.cause = compact_text(state.cause, 100) or None
        state.momentum = clamp(state.momentum)
        state.recent_topic = compact_text(state.recent_topic, 80)
        try:
            state.turns_since_cause = max(0, int(state.turns_since_cause))
            state.updated_at = max(0.0, float(state.updated_at))
        except (TypeError, ValueError):
            state.turns_since_cause = 0
            state.updated_at = 0.0
        state._clear_if_faint()
        return state

    def preview(
        self,
        signal: TurnSignal,
        *,
        recent_turns: Iterable[object] = (),
        now: float | None = None,
    ) -> "AkaneState":
        candidate = replace(self)
        candidate.update(signal, recent_turns=recent_turns, now=now)
        return candidate

    def decayed(self, *, now: float | None = None) -> "AkaneState":
        candidate = replace(self)
        candidate._decay_for_time(time.time() if now is None else float(now))
        candidate._clear_if_faint()
        return candidate

    def update(
        self,
        signal: TurnSignal,
        *,
        recent_turns: Iterable[object] = (),
        now: float | None = None,
    ) -> None:
        current = time.time() if now is None else float(now)
        self._decay_for_time(current)
        continuity = _conversation_continuity(signal, recent_turns, self.recent_topic)
        reaction = _reaction_from_signal(signal)
        if reaction is None:
            self._soften(0.84 if continuity else 0.62)
            self.turns_since_cause += 1
        else:
            self._apply_reaction(reaction, continuity=continuity)
            self.turns_since_cause = 0
        self.recent_topic = compact_text(signal.topic, 80)
        self.updated_at = current
        self._clear_if_faint()

    def _apply_reaction(self, reaction: _Reaction, *, continuity: bool) -> None:
        previous_primary = self.primary
        previous_intensity = self.intensity
        carries = bool(previous_primary and previous_intensity >= 0.14 and self.momentum >= 0.16)

        if previous_primary == reaction.primary:
            primary = reaction.primary
            secondary = reaction.secondary or self.secondary
            intensity = min(1.0, previous_intensity * 0.62 + reaction.intensity * 0.58)
        elif carries and previous_intensity > reaction.intensity * (1.12 if continuity else 1.35):
            primary = previous_primary
            secondary = reaction.primary
            intensity = min(1.0, previous_intensity * 0.78 + reaction.intensity * 0.28)
        else:
            primary = reaction.primary
            secondary = previous_primary if carries else reaction.secondary
            intensity = min(1.0, reaction.intensity + previous_intensity * (0.24 if carries else 0.08))

        self.primary = primary
        self.secondary = secondary if secondary != primary else None
        self.intensity = intensity
        blend = 0.62 if carries else 0.82
        self.valence = clamp(
            self.valence * (1.0 - blend) + reaction.valence * blend,
            0.0,
            -1.0,
            1.0,
        )
        self.energy = clamp(self.energy * (1.0 - blend) + reaction.energy * blend)
        self.confidence = clamp(max(reaction.confidence, self.confidence * 0.72))
        self.cause = reaction.cause
        self.momentum = clamp(self.momentum * 0.45 + reaction.intensity * 0.75)

    def _soften(self, factor: float) -> None:
        self.intensity *= factor
        self.momentum *= factor * 0.9
        self.confidence *= 0.88
        self.valence *= 0.82
        self.energy += (_NEUTRAL_ENERGY - self.energy) * 0.24

    def _decay_for_time(self, now: float) -> None:
        if not self.updated_at:
            return
        elapsed_hours = max(0.0, now - self.updated_at) / 3600.0
        if elapsed_hours <= 0.0:
            return
        retention = 0.5 ** (elapsed_hours / 12.0)
        self.intensity *= retention
        self.momentum *= retention
        self.confidence *= 0.75 + retention * 0.25
        self.valence *= retention
        self.energy = _NEUTRAL_ENERGY + (self.energy - _NEUTRAL_ENERGY) * retention

    def _clear_if_faint(self) -> None:
        if self.intensity >= 0.11 and self.confidence >= 0.22:
            return
        self.primary = None
        self.secondary = None
        self.intensity = 0.0
        self.valence = 0.0
        self.confidence = 0.0
        self.cause = None
        self.momentum = 0.0


def summarize_state(state: AkaneState, signal: TurnSignal | None = None) -> str:
    """Describe optional context permissively without prescribing response wording."""

    del signal  # The preview already incorporates the current turn and recent context.
    if not state.primary or state.intensity < 0.11 or state.confidence < 0.22:
        return ""
    degree = "slightly" if state.intensity < 0.34 else "noticeably" if state.intensity < 0.66 else "strongly"
    uncertainty = "may be" if state.confidence < 0.58 else "is"
    mixed = f" and {state.secondary}" if state.secondary else ""
    because = f" because of {state.cause}" if state.cause else ""
    return (
        f"Akane {uncertainty} {degree} {state.primary}{mixed}{because}. "
        "This is subtle context, not something she needs to name; let it affect the reply only if useful."
    )


def public_state(state: AkaneState) -> dict[str, object]:
    """Expose qualitative debug information without leaking internal scores."""

    return {
        "schema_version": state.schema_version,
        "emotional_context": "active" if state.primary else "neutral",
        "recent_topic": state.recent_topic,
        "updated_at": state.updated_at,
    }


def _reaction_from_signal(signal: TurnSignal) -> _Reaction | None:
    """Convert strong conversational cues into Akane's possible reaction, not the user's mood."""

    if signal.sadness:
        return _Reaction("concerned", "tender", 0.58, -0.28, 0.36, 0.82, "what the user shared")
    if signal.hostility:
        return _Reaction("irritated", "guarded", 0.56, -0.48, 0.58, 0.86, "the hostility toward her")
    if signal.criticism or signal.correction_requested:
        return _Reaction("concerned", "irritated", 0.34, -0.18, 0.48, 0.68, "the correction to her response")
    if signal.teasing:
        return _Reaction("amused", "competitive", 0.46, 0.42, 0.62, 0.67, "the user's playful challenge")
    if signal.praise:
        return _Reaction("pleased", "embarrassed", 0.40, 0.52, 0.58, 0.70, "the user's praise")
    if signal.frustration:
        return _Reaction("concerned", "curious", 0.38, -0.18, 0.46, 0.62, "the problem frustrating the user")
    if signal.friendliness:
        return _Reaction("affectionate", None, 0.24, 0.34, 0.52, 0.48, "the warmth in the exchange")
    if signal.debugging:
        return _Reaction("curious", "concerned", 0.28, -0.05, 0.56, 0.48, "the unresolved technical problem")
    return None


def _conversation_continuity(
    signal: TurnSignal,
    recent_turns: Iterable[object],
    previous_topic: str,
) -> bool:
    if signal.low_content:
        return bool(previous_topic)
    candidates = [
        compact_text(getattr(turn, "content", ""), 180)
        for turn in recent_turns
        if getattr(turn, "role", "") == "user"
    ]
    prior = candidates[-1] if candidates else previous_topic
    return bool(prior and max(topic_overlap(prior, signal.topic), topic_overlap(previous_topic, signal.topic)) >= 0.34)


def _migrate_delivery_state(payload: dict[str, object]) -> AkaneState:
    candidates = [
        (clamp(payload.get("concern"), 0.1) - 0.1, "concerned", -0.2),
        (clamp(payload.get("irritation"), 0.06) - 0.06, "irritated", -0.4),
        (clamp(payload.get("playfulness"), 0.24) - 0.24, "amused", 0.3),
        (clamp(payload.get("warmth"), 0.58) - 0.58, "affectionate", 0.35),
    ]
    ranked = sorted((item for item in candidates if item[0] >= 0.08), reverse=True)
    try:
        updated_at = max(0.0, float(payload.get("updated_at") or 0.0))
    except (TypeError, ValueError):
        updated_at = 0.0
    if not ranked:
        return AkaneState(
            recent_topic=compact_text(payload.get("recent_topic"), 80),
            updated_at=updated_at,
        )
    strength, primary, valence = ranked[0]
    secondary = ranked[1][1] if len(ranked) > 1 else None
    return AkaneState(
        primary=primary,
        secondary=secondary,
        intensity=clamp(0.18 + strength),
        valence=valence,
        energy=clamp(payload.get("energy"), _NEUTRAL_ENERGY),
        confidence=0.55,
        cause=compact_text(payload.get("transition_reason"), 100) or "earlier conversation context",
        momentum=0.28,
        recent_topic=compact_text(payload.get("recent_topic"), 80),
        updated_at=updated_at,
    )


class EmotionStore:
    """Own profile-scoped context and persist only successful turns."""

    def __init__(self, path: Path = EMOTION_STATE_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._states: dict[str, AkaneState] = {}
        self._load()

    def preview(
        self,
        profile_id: str,
        signal: TurnSignal,
        *,
        recent_turns: Iterable[object] = (),
        now: float | None = None,
    ) -> AkaneState:
        with self._lock:
            state = self._state(profile_id)
            self._prune()
            return state.preview(signal, recent_turns=recent_turns, now=now)

    def commit(
        self,
        profile_id: str,
        signal: TurnSignal,
        *,
        recent_turns: Iterable[object] = (),
        now: float | None = None,
    ) -> AkaneState:
        with self._lock:
            key = compact_text(profile_id, 120) or "local:owner"
            previous = replace(self._state(key))
            candidate = previous.preview(signal, recent_turns=recent_turns, now=now)
            self._states[key] = candidate
            self._prune()
            try:
                self._persist()
            except Exception:
                self._states[key] = previous
                raise
            return replace(candidate)

    def snapshot(self, profile_id: str, *, now: float | None = None) -> AkaneState:
        with self._lock:
            state = self._state(profile_id)
            self._prune()
            return state.decayed(now=now)

    def clear(self, profile_id: str) -> None:
        with self._lock:
            key = compact_text(profile_id, 120) or "local:owner"
            if key not in self._states:
                return
            self._states.pop(key)
            self._persist()

    def restore(self, profile_id: str, state: AkaneState) -> None:
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
            if not isinstance(payload, dict):
                raise ValueError("invalid state document")
            schema = int(payload.get("schema_version", 0))
            if schema not in {2, STATE_SCHEMA_VERSION}:
                raise ValueError("unsupported schema")
            profiles = payload.get("profiles")
            if not isinstance(profiles, dict):
                raise ValueError("invalid profiles")
            self._states = {
                profile: AkaneState.from_dict(value)
                for key, value in profiles.items()
                if (profile := compact_text(key, 120))
            }
            self._prune()
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as exc:
            print(f"[Akane:state] ignored corrupt state file ({type(exc).__name__})", flush=True)
            self._states = {}

    def _persist(self) -> None:
        atomic_write_json(
            self._path,
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "profiles": {key: asdict(value) for key, value in self._states.items()},
            },
        )


_EMOTIONS: EmotionStore | None = None
_EMOTIONS_LOCK = threading.Lock()


def get_emotion_store() -> EmotionStore:
    global _EMOTIONS
    if _EMOTIONS is None:
        with _EMOTIONS_LOCK:
            if _EMOTIONS is None:
                _EMOTIONS = EmotionStore()
    return _EMOTIONS
