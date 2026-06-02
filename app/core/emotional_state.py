"""Small autonomous mood and emotion state for Akane."""

from __future__ import annotations

import json
import math
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

DEFAULT_VARS = {
    "energy": 0.62,
    "social": 0.52,
    "focus": 0.40,
    "comfort": 0.58,
    "stimulation": 0.24,
}
EMOTION_VARS = {
    "stimulation": 0.24,
    "comfort": 0.58,
    "tension": 0.16,
    "playfulness": 0.12,
}

_CALM_INTENSITY = 0.6
_MOOD_SHIFT_SECONDS = 60 * 60
_EMOTION_IDLE_SECONDS = 45 * 60
_EMOTION_TURNS = 3
_EMOTION_DECAY = 0.58

def _clamp(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return max(0.0, min(1.0, number))


def _valid(value: object, allowed: tuple[str, ...], default: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in allowed else default


def _float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mix(value: float, target: float, amount: float) -> float:
    return _clamp(value + (target - value) * _clamp(amount))


def _alpha(seconds: float, half_life: float) -> float:
    return 0.0 if seconds <= 0 else 1.0 - (0.5 ** (seconds / half_life))


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in str(text or "").split():
        word = raw.strip(".,!?;:()[]{}\"'`").lower()
        if len(word) >= 4:
            tokens.add(word)
    return tokens


def _signals(text: str) -> dict[str, float | bool]:
    value = str(text or "").strip()
    question_marks = value.count("?")
    exclaims = value.count("!")
    return {
        "activity": min(1.0, len(value) / 700.0),
        "exclaim": min(1.0, exclaims / 4.0),
        "question_pressure": min(1.0, max(0, question_marks - 1) / 3.0),
        "short": len(value) < 80,
    }


def _energy_target(now: float) -> float:
    hour = time.localtime(now).tm_hour
    if hour < 5:
        return 0.30
    if hour < 8:
        return 0.48
    if hour < 18:
        return 0.66
    if hour < 23:
        return 0.56
    return 0.40


class EmotionalState:
    def __init__(self, path: Path = STATE_PATH) -> None:
        now = time.time()
        self.path = Path(path)
        self._lock = threading.RLock()
        self._vars = dict(DEFAULT_VARS)
        self._mood = "calm"
        self._mood_intensity = _CALM_INTENSITY
        self._mood_candidate = "calm"
        self._candidate_at = now
        self._updated_at = now
        self._last_interaction_at = now
        self._started_at = now
        self._dirty = False
        self._sessions: dict[str, dict[str, object]] = {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            if not isinstance(data, dict):
                return

            mood = _valid(data.get("mood"), MOODS, "calm")
            default_intensity = _CALM_INTENSITY if mood == "calm" else 0.45
            old_scores = data.get("mood_scores")
            if isinstance(old_scores, dict):
                default_intensity = _clamp(old_scores.get(mood), default_intensity)

            variables = data.get("variables")
            if isinstance(variables, dict):
                self._vars = {name: _clamp(variables.get(name), default) for name, default in DEFAULT_VARS.items()}
            else:
                self._migrate_old_mood(mood, _clamp(data.get("mood_intensity"), default_intensity))

            self._mood = mood
            self._mood_intensity = _clamp(data.get("mood_intensity"), default_intensity)
            self._mood_candidate = mood
            self._updated_at = _float(data.get("updated_at"), self._updated_at)
            self._candidate_at = self._updated_at
            self._last_interaction_at = _float(data.get("last_interaction_at"), self._updated_at)

    def observe_user_message(
        self,
        session_id: str | None,
        text: str,
        *,
        now: float | None = None,
        persist: bool = True,
    ) -> dict[str, object]:
        current = time.time() if now is None else float(now)
        session_id = str(session_id or "default")[:120]
        with self._lock:
            elapsed = self._advance(current)
            session = self._sessions.setdefault(session_id, self._new_session(current))
            self._decay_emotion(session, current, count_turn=True)

            signals = _signals(text)
            new_topic = self._update_session(session, text, signals, current)
            self._nudge(signals, new_topic, current)
            self._choose_emotion(session, signals, current)
            self._update_mood(current, elapsed)

            self._last_interaction_at = current
            session["updated_at"] = current
            self._dirty = True
            if persist:
                self._save()
            return self._snapshot(session_id, current)

    def snapshot(self, session_id: str | None = None, *, now: float | None = None) -> dict[str, object]:
        current = time.time() if now is None else float(now)
        session_id = str(session_id or "default")[:120]
        with self._lock:
            elapsed = self._advance(current)
            self._update_mood(current, elapsed)
            self._dirty = self._dirty or elapsed > 0.5
            return self._snapshot(session_id, current)

    def persist(self) -> None:
        with self._lock:
            if self._dirty:
                self._save()

    def _new_session(self, now: float) -> dict[str, object]:
        return {
            "emotion": "neutral",
            "intensity": 0.0,
            "turns": 0,
            "updated_at": now,
            "topic": set(),
            "rare_at": 0.0,
            "problem_run": 0,
            "vars": dict(EMOTION_VARS),
        }

    def _migrate_old_mood(self, mood: str, intensity: float) -> None:
        amount = _clamp(intensity, 0.45) * 0.25
        if mood == "happy":
            self._add("comfort", amount)
            self._add("social", amount * 0.7)
        elif mood == "sleepy":
            self._add("energy", -amount)
        elif mood == "focused":
            self._add("focus", amount)
        elif mood == "lonely":
            self._add("social", -amount)

    def _advance(self, now: float) -> float:
        if now < self._updated_at:
            self._updated_at = self._last_interaction_at = self._started_at = now
            return 0.0
        if self._started_at > now:
            self._started_at = now

        elapsed = now - self._updated_at
        if elapsed <= 0.5:
            return 0.0

        silence = now - self._last_interaction_at
        uptime_hours = max(0.0, now - self._started_at) / 3600.0
        social_target = 0.54
        if silence > 24 * 60 * 60:
            social_target = 0.16
        elif silence > 6 * 60 * 60:
            social_target = 0.24
        elif silence > 2 * 60 * 60:
            social_target = 0.34
        elif silence > 45 * 60:
            social_target = 0.44

        targets = {
            "energy": _energy_target(now) - min(0.14, max(0.0, uptime_hours - 3.0) * 0.02),
            "social": social_target,
            "focus": 0.36,
            "comfort": 0.58,
            "stimulation": 0.20,
        }
        half_lives = {"energy": 7 * 3600, "social": 5 * 3600, "focus": 4 * 3600, "comfort": 12 * 3600, "stimulation": 70 * 60}
        for index, name in enumerate(DEFAULT_VARS):
            drift = math.sin((now / 3600.0) + index) * 0.002 * min(1.0, elapsed / (6 * 3600))
            self._vars[name] = _mix(self._vars[name], targets[name], _alpha(elapsed, half_lives[name]))
            self._add(name, drift)

        self._updated_at = now
        return elapsed

    def _emotion_vars(self, session: dict[str, object]) -> dict[str, float]:
        values = session.get("vars")
        if not isinstance(values, dict):
            values = dict(EMOTION_VARS)
            session["vars"] = values
        return values

    def _add_emotion_var(self, session: dict[str, object], name: str, amount: float) -> None:
        values = self._emotion_vars(session)
        values[name] = _clamp(values.get(name, EMOTION_VARS[name]) + amount)

    def _update_session(self, session: dict[str, object], text: str, signals: dict[str, float | bool], now: float) -> bool:
        tokens = _tokens(text)
        old_tokens = session.get("topic")
        old_tokens = old_tokens if isinstance(old_tokens, set) else set()
        new_topic = len(tokens) >= 2 and not bool(tokens & old_tokens)
        session["topic"] = tokens or old_tokens

        if now - _float(session.get("updated_at"), now) > _EMOTION_IDLE_SECONDS:
            self._add_emotion_var(session, "comfort", -0.03)
            self._add_emotion_var(session, "stimulation", -0.02)

        activity = float(signals["activity"])
        exclaim = float(signals["exclaim"])
        pressure = float(signals["question_pressure"])
        self._add_emotion_var(session, "stimulation", activity * 0.025 + (0.018 if new_topic else 0.0))
        self._add_emotion_var(session, "comfort", exclaim * 0.010)
        self._add_emotion_var(session, "playfulness", exclaim * 0.012 if signals["short"] else 0.0)

        problem_run = int(session.get("problem_run", 0) or 0)
        if pressure > 0.0:
            problem_run += 1
        else:
            problem_run = max(0, problem_run - 1)
        session["problem_run"] = min(4, problem_run)
        if problem_run >= 3:
            self._add_emotion_var(session, "tension", 0.05)
        return new_topic

    def _nudge(self, signals: dict[str, float | bool], new_topic: bool, now: float) -> None:
        gap = now - self._last_interaction_at
        self._add("social", 0.024 if gap < 10 * 60 else 0.016 if gap < 45 * 60 else 0.008)
        if gap < 10 * 60:
            self._add("energy", -0.006)

        activity = float(signals["activity"])
        exclaim = float(signals["exclaim"])
        self._add("focus", activity * 0.006 + (0.006 if new_topic else 0.0))
        self._add("comfort", exclaim * 0.006 - float(signals["question_pressure"]) * 0.004)
        self._add(
            "stimulation",
            activity * 0.012
            + exclaim * 0.014
            + (0.018 if new_topic else 0.0)
        )

    def _update_mood(self, now: float, elapsed: float) -> None:
        mood, intensity = self._mood_target(now)
        if mood == self._mood:
            self._mood_intensity = _mix(self._mood_intensity, intensity, _alpha(elapsed, 3 * 3600) or 0.08)
            self._mood_candidate = mood
            self._candidate_at = now
            return

        if mood != self._mood_candidate:
            self._mood_candidate = mood
            self._candidate_at = now - min(elapsed, _MOOD_SHIFT_SECONDS)

        if mood == "calm":
            self._mood_intensity = _mix(self._mood_intensity, 0.26, _alpha(elapsed, 4 * 3600) or 0.05)
            if self._mood_intensity <= 0.32:
                self._mood = "calm"
                self._mood_intensity = _CALM_INTENSITY
            return

        if intensity >= 0.54 and now - self._candidate_at >= _MOOD_SHIFT_SECONDS:
            self._mood = mood
            self._mood_intensity = min(0.58, intensity)

    def _mood_target(self, now: float) -> tuple[str, float]:
        energy = self._vars["energy"]
        social = self._vars["social"]
        focus = self._vars["focus"]
        comfort = self._vars["comfort"]
        stimulation = self._vars["stimulation"]
        late = time.localtime(now).tm_hour >= 23 or time.localtime(now).tm_hour <= 5
        silence = now - self._last_interaction_at
        uptime_hours = max(0.0, now - self._started_at) / 3600.0
        happy = 0.0
        if comfort > 0.62 and social > 0.57 and energy > 0.44:
            happy = (comfort - 0.58) * 0.9 + (social - 0.54) * 0.8 + (energy - 0.44) * 0.25

        scores = {
            "sleepy": max(0.0, 0.46 - energy) * 1.35 + (0.16 if late else 0.0) + min(0.18, max(0.0, uptime_hours - 4.0) * 0.025),
            "focused": max(0.0, focus - 0.58) * 1.75 + max(0.0, stimulation - 0.45) * 0.30,
            "happy": happy,
            "lonely": max(0.0, 0.40 - social) * 1.55 + (min(0.20, (silence - 2 * 3600) / (10 * 3600) * 0.20) if silence > 2 * 3600 else 0.0),
        }
        mood, score = max(scores.items(), key=lambda item: item[1])
        return ("calm", _CALM_INTENSITY) if score < 0.08 else (mood, _clamp(0.36 + score * 0.72))

    def _decay_emotion(self, session: dict[str, object], now: float, *, count_turn: bool) -> None:
        elapsed = now - _float(session.get("updated_at"), now)
        values = self._emotion_vars(session)
        if elapsed > 0:
            drift = _alpha(elapsed, 35 * 60)
            values["stimulation"] = _mix(values["stimulation"], 0.18, drift)
            values["playfulness"] = _mix(values["playfulness"], 0.10, drift)
            values["tension"] = _mix(values["tension"], 0.14, drift)
            values["comfort"] = _mix(values["comfort"], self._vars["comfort"], _alpha(elapsed, 90 * 60))

        if elapsed > _EMOTION_IDLE_SECONDS:
            session.update({"emotion": "neutral", "intensity": 0.0, "turns": 0})
            return
        if not count_turn:
            return
        turns = int(session.get("turns", 0) or 0)
        if turns > 0:
            session["turns"] = turns - 1
            session["intensity"] = _clamp(session.get("intensity")) * _EMOTION_DECAY
        if int(session.get("turns", 0) or 0) <= 0 or _clamp(session.get("intensity")) < 0.08:
            session.update({"emotion": "neutral", "intensity": 0.0, "turns": 0})

    def _choose_emotion(
        self,
        session: dict[str, object],
        signals: dict[str, float | bool],
        now: float,
    ) -> None:
        values = self._emotion_vars(session)
        stimulation = values["stimulation"]
        comfort = values["comfort"]
        tension = values["tension"]
        playfulness = values["playfulness"]
        last_rare = _float(session.get("rare_at"), 0.0)
        if float(signals["exclaim"]) > 0.6 and playfulness > 0.35 and comfort > 0.60 and now - last_rare > 20 * 60:
            self._set_emotion(session, "embarrassed", 0.36)
            session["rare_at"] = now
            return

        scores = {
            "curious": stimulation * (1.0 - tension),
            "amused": playfulness * 0.65 + comfort * 0.35,
            "concerned": tension * 0.75 + max(0.0, 0.50 - comfort) * 0.9,
            "excited": stimulation * 0.75 + comfort * 0.25,
        }
        thresholds = {"curious": 0.30, "amused": 0.42, "concerned": 0.31, "excited": 0.62}
        eligible = [(emotion, score) for emotion, score in scores.items() if score >= thresholds[emotion]]
        if eligible:
            emotion, score = max(eligible, key=lambda item: item[1] - thresholds[item[0]])
            self._set_emotion(session, emotion, _clamp(0.20 + (score - thresholds[emotion]) * 0.9))

    def _set_emotion(self, session: dict[str, object], emotion: str, intensity: float) -> None:
        intensity = min(0.72, _clamp(intensity))
        current = _valid(session.get("emotion"), EMOTIONS, "neutral")
        current_intensity = _clamp(session.get("intensity"))
        if current != "neutral" and emotion != current and intensity < current_intensity + 0.12:
            return
        session.update({"emotion": emotion, "intensity": max(0.20, intensity), "turns": _EMOTION_TURNS})

    def _snapshot(self, session_id: str, now: float) -> dict[str, object]:
        session = self._sessions.get(session_id)
        if session:
            self._decay_emotion(session, now, count_turn=False)
        emotion_values = self._emotion_vars(session) if session else EMOTION_VARS
        emotion = _valid(session.get("emotion") if session else None, EMOTIONS, "neutral")
        return {
            "mood": self._mood,
            "mood_intensity": round(self._mood_intensity, 3),
            "emotion": emotion,
            "emotion_intensity": round(_clamp(session.get("intensity") if session else 0.0), 3),
            "emotion_turns": int(session.get("turns", 0) if session else 0),
            "energy": round(self._vars["energy"], 3),
            "social": round(self._vars["social"], 3),
            "focus": round(self._vars["focus"], 3),
            "comfort": round(emotion_values["comfort"], 3),
            "stimulation": round(emotion_values["stimulation"], 3),
            "tension": round(emotion_values["tension"], 3),
            "playfulness": round(emotion_values["playfulness"], 3),
            "updated_at": self._updated_at,
            "last_interaction_at": self._last_interaction_at,
            "allowed_moods": list(MOODS),
            "allowed_emotions": list(EMOTIONS),
        }

    def _save(self) -> None:
        payload = {
            "version": 2,
            "mood": self._mood,
            "mood_intensity": round(self._mood_intensity, 4),
            "variables": {name: round(self._vars[name], 4) for name in DEFAULT_VARS},
            "updated_at": self._updated_at,
            "last_interaction_at": self._last_interaction_at,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
            self._dirty = False
        except OSError:
            pass

    def _add(self, name: str, amount: float) -> None:
        self._vars[name] = _clamp(self._vars[name] + amount)


_STATE = EmotionalState()


def get_emotional_state() -> EmotionalState:
    return _STATE


def observe_user_message(session_id: str | None, text: str, *, persist: bool = True) -> dict[str, object]:
    return _STATE.observe_user_message(session_id, text, persist=persist)


def snapshot(session_id: str | None = None) -> dict[str, object]:
    return _STATE.snapshot(session_id)


def persist() -> None:
    _STATE.persist()


def format_for_prompt(state: dict[str, object]) -> str:
    mood = _valid(state.get("mood"), MOODS, "calm")
    emotion = _valid(state.get("emotion"), EMOTIONS, "neutral")
    mood_intensity = _clamp(state.get("mood_intensity"), _CALM_INTENSITY if mood == "calm" else 0.4)
    emotion_intensity = _clamp(state.get("emotion_intensity"), 0.0 if emotion == "neutral" else 0.4)
    return (
        "Current internal state:\n"
        f"Mood: {mood} ({mood_intensity:.1f})\n"
        f"Emotion: {emotion} ({emotion_intensity:.1f})\n\n"
        "Use this as subtle behavioral guidance only. Do not mention labels or values."
    )
