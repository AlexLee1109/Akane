"""Lightweight mood state with compact prompt guidance."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

MOODS = ("calm", "sleepy", "focused", "playful", "warm", "lonely", "concerned", "restless")
EMOTIONS = ("neutral", "amused", "smug", "embarrassed", "concerned", "pleased", "annoyed", "soft", "focused")
STATE_PATH = Path(os.environ.get(
    "AKANE_EMOTIONAL_STATE_PATH",
    Path(__file__).resolve().parent.parent / "emotional_state.json",
))

DEFAULT_VARS = {
    "energy": 0.62,
    "social": 0.50,
    "focus": 0.40,
    "comfort": 0.58,
    "stimulation": 0.24,
    "tension": 0.16,
    "playfulness": 0.18,
    "affection": 0.34,
}
_MAX_SESSIONS = 64
_STALE_SECONDS = 4 * 3600
_EMOTION_TURNS = 3
_MOOD_TONE = {
    "calm": "natural, grounded, compact",
    "sleepy": "softer, lower-energy, shorter",
    "focused": "direct, practical, compact",
    "playful": "casual, lightly teasing, lightly amused",
    "warm": "gentler, familiar, compact",
    "lonely": "soft, restrained, compact",
    "concerned": "careful, grounded, less playful",
    "restless": "slightly sharper, shorter, controlled",
}
_EMOTION_TONE = {
    "amused": "lightly humorous",
    "smug": "dryly amused",
    "embarrassed": "warmer",
    "concerned": "careful",
    "pleased": "lightly satisfied",
    "annoyed": "blunt but not mean",
    "soft": "gentler",
    "focused": "direct",
}


def _clamp(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _valid(value: object, allowed: tuple[str, ...], default: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in allowed else default


class EmotionalState:
    def __init__(self, path: Path = STATE_PATH) -> None:
        now = time.time()
        self.path = Path(path)
        self._lock = threading.RLock()
        self._vars = dict(DEFAULT_VARS)
        self._mood = "calm"
        self._mood_intensity = 0.6
        self._updated_at = now
        self._last_interaction_at = now
        self._sessions: dict[str, dict[str, object]] = {}
        self._dirty = False
        self.reload()

    def reload(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        with self._lock:
            mood = "warm" if data.get("mood") == "happy" else data.get("mood")
            self._mood = _valid(mood, MOODS, "calm")
            self._mood_intensity = _clamp(data.get("mood_intensity"), 0.6)
            variables = data.get("variables")
            if isinstance(variables, dict):
                self._vars = {
                    name: _clamp(variables.get(name), default)
                    for name, default in DEFAULT_VARS.items()
                }
            elif self._mood == "focused":
                self._vars["focus"] = 0.70
            elif self._mood == "sleepy":
                self._vars["energy"] = 0.30
            elif self._mood == "playful":
                self._vars["playfulness"] = 0.65
            elif self._mood == "warm":
                self._vars["comfort"] = 0.72
                self._vars["affection"] = 0.52
            elif self._mood == "concerned":
                self._vars["tension"] = 0.60
            elif self._mood == "restless":
                self._vars["stimulation"] = 0.68
            self._updated_at = float(data.get("updated_at") or self._updated_at)
            self._last_interaction_at = float(data.get("last_interaction_at") or self._updated_at)

    def _prune(self, now: float) -> None:
        for key, state in list(self._sessions.items()):
            if now - float(state.get("updated_at") or 0.0) > _STALE_SECONDS:
                self._sessions.pop(key, None)
        if len(self._sessions) > _MAX_SESSIONS:
            oldest = sorted(self._sessions.items(), key=lambda item: float(item[1].get("updated_at") or 0.0))
            for key, _state in oldest[:len(self._sessions) - _MAX_SESSIONS]:
                self._sessions.pop(key, None)

    def _drift(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated_at)
        if elapsed < 60:
            return
        amount = min(0.08, elapsed / (24 * 3600) * 0.08)
        for name, target in DEFAULT_VARS.items():
            self._vars[name] += (target - self._vars[name]) * amount
        self._updated_at = now

    def _choose_mood(self, now: float) -> None:
        values = self._vars
        silence = now - self._last_interaction_at
        if silence > 24 * 3600 and values["social"] < 0.32:
            mood = "lonely"
        elif values["energy"] < 0.36:
            mood = "sleepy"
        elif values["tension"] > 0.58:
            mood = "concerned"
        elif values["focus"] > 0.66:
            mood = "focused"
        elif values["playfulness"] > 0.60:
            mood = "playful"
        elif values["comfort"] > 0.70 and values["affection"] > 0.48:
            mood = "warm"
        elif values["stimulation"] > 0.66:
            mood = "restless"
        else:
            mood = "calm"
        self._mood = mood
        self._mood_intensity = 0.6 if mood == "calm" else 0.45

    def observe_user_message(
        self,
        session_id: str | None,
        text: str,
        *,
        now: float | None = None,
        persist: bool = True,
    ) -> dict[str, object]:
        current = time.time() if now is None else float(now)
        key = (str(session_id or "default").strip() or "default")[:120]
        lower = str(text or "").lower()
        with self._lock:
            self._drift(current)
            self._prune(current)
            session = self._sessions.setdefault(
                key,
                {"emotion": "neutral", "intensity": 0.0, "turns": 0, "updated_at": current},
            )
            turns = max(0, int(session.get("turns") or 0) - 1)
            session.update({"emotion": "neutral", "intensity": 0.0, "turns": turns})

            emotion = "neutral"
            if any(word in lower for word in ("error", "failed", "broken", "stuck", "annoying")):
                emotion = "concerned" if self._vars["tension"] < 0.45 else "annoyed"
                self._vars["tension"] = _clamp(self._vars["tension"] + 0.05)
            elif any(word in lower for word in ("fixed", "worked", "success", "finally", "great")):
                emotion = "pleased"
                self._vars["comfort"] = _clamp(self._vars["comfort"] + 0.04)
            elif any(word in lower for word in ("lol", "haha", "funny")):
                emotion = "amused"
                self._vars["playfulness"] = _clamp(self._vars["playfulness"] + 0.05)
            elif any(word in lower for word in ("thanks", "thank you", "cute", "hug")):
                emotion = "soft"
                self._vars["affection"] = _clamp(self._vars["affection"] + 0.03)
            elif any(word in lower for word in ("code", "model", "prompt", "latency", "server", "discord")):
                emotion = "focused"
                self._vars["focus"] = _clamp(self._vars["focus"] + 0.04)

            if emotion != "neutral":
                session.update({"emotion": emotion, "intensity": 0.36, "turns": _EMOTION_TURNS})
            self._vars["social"] = _clamp(self._vars["social"] + 0.01)
            self._vars["stimulation"] = _clamp(self._vars["stimulation"] + min(0.03, len(lower) / 10000))
            session["updated_at"] = current
            self._last_interaction_at = current
            self._updated_at = current
            self._choose_mood(current)
            self._dirty = True
            if persist:
                self._save()
            return self._snapshot(key, current)

    def _snapshot(self, key: str, now: float) -> dict[str, object]:
        session = self._sessions.get(key) or {}
        variables = {name: round(value, 3) for name, value in self._vars.items()}
        return {
            "mood": self._mood,
            "mood_intensity": round(self._mood_intensity, 3),
            "emotion": _valid(session.get("emotion"), EMOTIONS, "neutral"),
            "emotion_intensity": round(_clamp(session.get("intensity")), 3),
            "emotion_turns": int(session.get("turns") or 0),
            "variables": variables,
            **variables,
            "updated_at": self._updated_at,
            "last_interaction_at": self._last_interaction_at,
            "allowed_moods": list(MOODS),
            "allowed_emotions": list(EMOTIONS),
        }

    def snapshot(self, session_id: str | None = None, *, now: float | None = None) -> dict[str, object]:
        current = time.time() if now is None else float(now)
        key = (str(session_id or "default").strip() or "default")[:120]
        with self._lock:
            self._drift(current)
            self._prune(current)
            return self._snapshot(key, current)

    def _save(self) -> None:
        payload = {
            "version": 3,
            "mood": self._mood,
            "mood_intensity": round(self._mood_intensity, 4),
            "variables": {name: round(value, 4) for name, value in self._vars.items()},
            "updated_at": self._updated_at,
            "last_interaction_at": self._last_interaction_at,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            tmp.replace(self.path)
            self._dirty = False
        except OSError:
            pass

    def persist(self) -> None:
        with self._lock:
            if self._dirty:
                self._save()


_STATE = EmotionalState()


def get_emotional_state() -> EmotionalState:
    return _STATE


def observe_user_message(session_id: str | None, text: str, *, persist: bool = True) -> dict[str, object]:
    return _STATE.observe_user_message(session_id, text, persist=persist)


def snapshot(session_id: str | None = None) -> dict[str, object]:
    return _STATE.snapshot(session_id)


def persist() -> None:
    _STATE.persist()


def tone_guidance(state: dict[str, object]) -> str:
    mood = _valid(state.get("mood"), MOODS, "calm")
    emotion = _valid(state.get("emotion"), EMOTIONS, "neutral")
    tone = _MOOD_TONE[mood]
    extra = _EMOTION_TONE.get(emotion)
    return f"{tone}, {extra}" if extra and extra not in tone else tone


def format_for_prompt(state: dict[str, object]) -> str:
    return f"Tone: {tone_guidance(state)}. Do not mention internal state."