"""Lightweight internal mood and emotion state for Akane."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

MOODS = ("calm", "happy", "sleepy", "focused", "lonely")
EMOTIONS = ("neutral", "curious", "amused", "embarrassed", "concerned", "excited")

STATE_PATH = Path(os.environ.get(
    "AKANE_EMOTIONAL_STATE_PATH",
    Path(__file__).resolve().parent.parent / "emotional_state.json",
))

_EMOTION_TURNS = 3
_EMOTION_IDLE_SECONDS = 45 * 60
_MOOD_HALFLIFE_SECONDS = 36 * 60 * 60
_SAVE_INTERVAL_SECONDS = 5 * 60

_CONCERNED = ("error", "broken", "crash", "problem", "worried", "anxious", "sad", "hurt", "scared", "not working", "too slow")
_EMBARRASSED = ("sorry", "my bad", "embarrassing", "awkward", "shy", "oops")
_EXCITED = ("!!!", "let's go", "lets go", "awesome", "amazing", "finally", "yay", "i did it", "love this")
_AMUSED = ("lol", "haha", "lmao", "funny", "joke")
_CURIOUS = ("?", "why ", "how ", "what ", "can you", "could you", "tell me", "wonder")
_FOCUSED = ("code", "bug", "fix", "file", "server", "python", "implement", "refactor", "test", "model", "prompt")
_SLEEPY = ("tired", "sleepy", "sleep", "nap", "exhausted", "late night")
_LONELY = ("lonely", "alone", "miss you", "quiet here")
_HAPPY = ("thanks", "thank you", "nice", "great", "good job", "cute", "happy")


def _valid(value: object, allowed: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _detect_emotion(text: str) -> str | None:
    lowered = f" {str(text or '').lower()} "
    if _contains_any(lowered, _CONCERNED):
        return "concerned"
    if _contains_any(lowered, _EMBARRASSED):
        return "embarrassed"
    if _contains_any(lowered, _EXCITED):
        return "excited"
    if _contains_any(lowered, _AMUSED):
        return "amused"
    if _contains_any(lowered, _CURIOUS):
        return "curious"
    return None


class EmotionalState:
    def __init__(self, path: Path = STATE_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._mood = "calm"
        self._scores = {mood: (1.0 if mood == "calm" else 0.0) for mood in MOODS}
        self._updated_at = time.time()
        self._last_save = 0.0
        self._session_emotions: dict[str, dict[str, object]] = {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            if not isinstance(data, dict):
                return
            self._mood = _valid(data.get("mood"), MOODS, "calm")
            raw_scores = data.get("mood_scores")
            if isinstance(raw_scores, dict):
                for mood in MOODS:
                    try:
                        self._scores[mood] = max(0.0, min(3.0, float(raw_scores.get(mood, self._scores[mood]))))
                    except (TypeError, ValueError):
                        pass
            try:
                self._updated_at = float(data.get("updated_at", self._updated_at))
            except (TypeError, ValueError):
                pass

    def observe_user_message(self, session_id: str | None, text: str, *, now: float | None = None) -> dict[str, object]:
        current = time.time() if now is None else float(now)
        session = str(session_id or "default")[:120]
        with self._lock:
            old_mood = self._mood
            self._decay_mood_locked(current)
            emotion_state = self._session_emotions.setdefault(
                session,
                {"emotion": "neutral", "turns": 0, "updated_at": current},
            )
            self._decay_emotion_locked(emotion_state, current, count_turn=True)

            detected = _detect_emotion(text)
            if detected:
                emotion_state["emotion"] = detected
                emotion_state["turns"] = _EMOTION_TURNS
            emotion_state["updated_at"] = current

            self._nudge_mood_locked(str(text or ""), str(emotion_state["emotion"]), current)
            self._refresh_mood_locked()
            self._save_locked(current, force=self._mood != old_mood)
            return self._snapshot_locked(session, current)

    def snapshot(self, session_id: str | None = None, *, now: float | None = None) -> dict[str, object]:
        current = time.time() if now is None else float(now)
        session = str(session_id or "default")[:120]
        with self._lock:
            self._decay_mood_locked(current)
            self._refresh_mood_locked()
            return self._snapshot_locked(session, current)

    def _decay_mood_locked(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated_at)
        if elapsed <= 0.5:
            return
        decay = 0.5 ** (elapsed / _MOOD_HALFLIFE_SECONDS)
        for mood in MOODS:
            if mood == "calm":
                self._scores[mood] = max(0.5, self._scores[mood] * decay + (1.0 - decay) * 1.0)
            else:
                self._scores[mood] = max(0.0, self._scores[mood] * decay)
        self._updated_at = now

    def _decay_emotion_locked(self, state: dict[str, object], now: float, *, count_turn: bool) -> None:
        try:
            updated_at = float(state.get("updated_at", now))
        except (TypeError, ValueError):
            updated_at = now
        if now - updated_at > _EMOTION_IDLE_SECONDS:
            state["emotion"] = "neutral"
            state["turns"] = 0
            return
        if not count_turn:
            return
        turns = int(state.get("turns", 0) or 0)
        if turns > 0:
            turns -= 1
        state["turns"] = turns
        if turns <= 0:
            state["emotion"] = "neutral"

    def _nudge_mood_locked(self, text: str, emotion: str, now: float) -> None:
        lowered = f" {text.lower()} "
        if _contains_any(lowered, _FOCUSED) or emotion in {"concerned", "curious"}:
            self._nudge_locked("focused", 0.18)
        if _contains_any(lowered, _SLEEPY) or time.localtime(now).tm_hour in {0, 1, 2, 3, 4, 5}:
            self._nudge_locked("sleepy", 0.08)
        if _contains_any(lowered, _LONELY):
            self._nudge_locked("lonely", 0.28)
        if _contains_any(lowered, _HAPPY) or emotion in {"amused", "excited"}:
            self._nudge_locked("happy", 0.22)

    def _nudge_locked(self, mood: str, amount: float) -> None:
        self._scores[mood] = max(0.0, min(3.0, self._scores[mood] + amount))
        if mood != "calm":
            self._scores["calm"] = max(0.2, self._scores["calm"] - amount * 0.04)

    def _refresh_mood_locked(self) -> None:
        current_score = self._scores.get(self._mood, 0.0)
        best = max(MOODS, key=lambda mood: self._scores.get(mood, 0.0))
        if best != self._mood and self._scores[best] >= current_score + 0.12:
            self._mood = best
        if self._mood != "calm" and self._scores.get(self._mood, 0.0) < 0.2:
            self._mood = "calm"

    def _snapshot_locked(self, session_id: str, now: float) -> dict[str, object]:
        state = self._session_emotions.get(session_id)
        if state:
            self._decay_emotion_locked(state, now, count_turn=False)
            emotion = _valid(state.get("emotion"), EMOTIONS, "neutral")
            turns = int(state.get("turns", 0) or 0)
        else:
            emotion = "neutral"
            turns = 0
        return {
            "mood": self._mood,
            "emotion": emotion,
            "emotion_turns": turns,
            "updated_at": self._updated_at,
            "allowed_moods": list(MOODS),
            "allowed_emotions": list(EMOTIONS),
        }

    def _save_locked(self, now: float, *, force: bool = False) -> None:
        if not force and now - self._last_save < _SAVE_INTERVAL_SECONDS:
            return
        payload = {
            "mood": self._mood,
            "mood_scores": {mood: round(self._scores[mood], 4) for mood in MOODS},
            "updated_at": self._updated_at,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
            self._last_save = now
        except OSError:
            pass


_STATE = EmotionalState()


def get_emotional_state() -> EmotionalState:
    return _STATE


def observe_user_message(session_id: str | None, text: str) -> dict[str, object]:
    return _STATE.observe_user_message(session_id, text)


def snapshot(session_id: str | None = None) -> dict[str, object]:
    return _STATE.snapshot(session_id)


def format_for_prompt(state: dict[str, object]) -> str:
    mood = _valid(state.get("mood"), MOODS, "calm")
    emotion = _valid(state.get("emotion"), EMOTIONS, "neutral")
    return (
        "Private internal state:\n"
        f"* Mood: {mood}\n"
        f"* Emotion: {emotion}\n\n"
        "Do not report these labels or values to the user. "
        "Use them as subtle behavioral guidance; mood should guide tone more than emotion."
    )
