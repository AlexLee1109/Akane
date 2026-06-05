"""Small autonomous mood and emotion state for Akane."""

from __future__ import annotations

import json
import math
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
EMOTION_VARS = {
    "stimulation": 0.24,
    "comfort": 0.58,
    "tension": 0.16,
    "playfulness": 0.18,
    "affection": 0.34,
}

_CALM_INTENSITY = 0.6
_MOOD_SHIFT_SECONDS = 60 * 60
_EMOTION_IDLE_SECONDS = 45 * 60
_SESSION_STALE_SECONDS = 4 * 3600
_MAX_SESSIONS = 64
_EMOTION_TURNS = 3
_EMOTION_DECAY = 0.58
_MOOD_TONE = {
    "calm": ("natural", "grounded", "compact"),
    "sleepy": ("softer", "lower-energy", "shorter"),
    "focused": ("direct", "practical", "less playful"),
    "playful": ("casual", "lightly teasing", "a little amused"),
    "warm": ("gentler", "familiar", "not dramatic"),
    "lonely": ("quieter", "a little more attached", "not needy"),
    "concerned": ("grounded", "careful", "less teasing"),
    "restless": ("slightly sharper", "shorter", "controlled"),
}
_EMOTION_TONE = {
    "amused": ("lightly humorous", "relaxed"),
    "smug": ("dryly amused", "lightly teasing"),
    "embarrassed": ("warmer", "compact"),
    "concerned": ("grounded", "careful", "less joking"),
    "pleased": ("warmer", "lightly satisfied"),
    "annoyed": ("blunt", "lower patience", "not mean"),
    "soft": ("gentler", "warm", "compact"),
    "focused": ("direct", "compact"),
}

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
    lower = value.lower()
    question_marks = value.count("?")
    exclaims = value.count("!")
    has = lambda *terms: any(term in lower for term in terms)
    return {
        "activity": min(1.0, len(value) / 700.0),
        "exclaim": min(1.0, exclaims / 4.0),
        "question_pressure": min(1.0, max(0, question_marks - 1) / 3.0),
        "short": len(value) < 80,
        "tired": has("tired", "sleepy", "exhausted", "no energy"),
        "positive": has("fixed", "worked", "success", "finally", "nice", "great"),
        "friction": has("bug", "error", "broke", "failed", "annoying", "stuck", "problem"),
        "play": has("lol", "haha", "funny"),
        "affection": has("cute", "good girl", "proud", "pat", "hug"),
        "thanks": has("thanks", "thank you"),
        "tease": has("suspicious", "allegations"),
        "technical": has("code", "model", "prompt", "latency", "discord"),
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

            raw_mood = str(data.get("mood") or "").strip().lower()
            mood = "warm" if raw_mood == "happy" else _valid(raw_mood, MOODS, "calm")
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
            self._prune_sessions(current)
            session = self._sessions.setdefault(session_id, self._new_session(current))
            self._decay_emotion(session, current, count_turn=True)

            signals = _signals(text)
            new_topic = self._update_session(session, text, signals, current)
            self._nudge(signals, new_topic, current)
            self._choose_emotion(session, signals, current)
            self._last_interaction_at = current
            self._update_mood(current, elapsed)
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
            self._prune_sessions(current)
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

    def _prune_sessions(self, now: float) -> None:
        if len(self._sessions) <= _MAX_SESSIONS:
            return
        for key, session in list(self._sessions.items()):
            if now - _float(session.get("updated_at"), now) > _SESSION_STALE_SECONDS:
                self._sessions.pop(key, None)
        extra = len(self._sessions) - _MAX_SESSIONS
        if extra > 0:
            old = sorted(self._sessions.items(), key=lambda item: _float(item[1].get("updated_at"), 0.0))
            for key, _session in old[:extra]:
                self._sessions.pop(key, None)

    def _migrate_old_mood(self, mood: str, intensity: float) -> None:
        amount = _clamp(intensity, 0.45) * 0.25
        if mood in {"happy", "warm"}:
            self._add("comfort", amount)
            self._add("affection", amount * 0.8)
            self._add("social", amount * 0.3)
        elif mood == "sleepy":
            self._add("energy", -amount)
        elif mood == "focused":
            self._add("focus", amount)
        elif mood == "lonely":
            self._add("social", -amount)
        elif mood == "concerned":
            self._add("tension", amount)
        elif mood == "playful":
            self._add("playfulness", amount)
        elif mood == "restless":
            self._add("stimulation", amount)
            self._add("tension", amount * 0.6)

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
        social_target = 0.50
        if silence > 24 * 60 * 60:
            social_target = 0.16
        elif silence > 6 * 60 * 60:
            social_target = 0.26
        elif silence > 2 * 60 * 60:
            social_target = 0.36
        elif silence > 45 * 60:
            social_target = 0.44

        low_energy_tension = max(0.0, 0.34 - self._vars["energy"]) * 0.45
        uptime_tension = min(0.14, max(0.0, uptime_hours - 5.0) * 0.018)
        targets = {
            "energy": _energy_target(now) - min(0.14, max(0.0, uptime_hours - 3.0) * 0.02),
            "social": social_target,
            "focus": 0.42 if silence < 30 * 60 else 0.36,
            "comfort": 0.58,
            "stimulation": 0.28 if silence < 15 * 60 else 0.18,
            "tension": 0.16 + low_energy_tension + uptime_tension,
            "playfulness": _clamp(self._vars["comfort"] * 0.45 + self._vars["energy"] * 0.35 - self._vars["tension"] * 0.25 + 0.10),
            "affection": 0.34 if silence < 12 * 60 * 60 else 0.28 if silence < 48 * 60 * 60 else 0.22,
        }
        half_lives = {
            "energy": 7 * 3600,
            "social": 7 * 3600,
            "focus": 4 * 3600,
            "comfort": 12 * 3600,
            "stimulation": 90 * 60,
            "tension": 5 * 3600,
            "playfulness": 4 * 3600,
            "affection": 36 * 3600,
        }
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
        else:
            for name, default in EMOTION_VARS.items():
                values[name] = _clamp(values.get(name), default)
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
        self._add_emotion_var(session, "stimulation", activity * 0.020 + (0.015 if new_topic else 0.0) + (0.020 if signals["technical"] else 0.0))
        self._add_emotion_var(session, "comfort", exclaim * 0.008 + (0.020 if signals["positive"] or signals["affection"] or signals["thanks"] else 0.0) - (0.020 if signals["friction"] else 0.0))
        self._add_emotion_var(session, "playfulness", 0.040 if signals["play"] or signals["tease"] else exclaim * 0.010 if signals["short"] else 0.0)
        self._add_emotion_var(session, "tension", pressure * 0.020 + (0.035 if signals["friction"] else 0.0) + (0.015 if signals["tired"] else 0.0))
        self._add_emotion_var(session, "affection", 0.025 if signals["affection"] else 0.010 if signals["thanks"] else 0.0)

        problem_run = int(session.get("problem_run", 0) or 0)
        if pressure > 0.0 or signals["friction"]:
            problem_run += 1
        else:
            problem_run = max(0, problem_run - 1)
        session["problem_run"] = min(4, problem_run)
        if problem_run >= 3:
            self._add_emotion_var(session, "tension", 0.05)
        return new_topic

    def _nudge(self, signals: dict[str, float | bool], new_topic: bool, now: float) -> None:
        gap = now - self._last_interaction_at
        self._add("social", 0.006 if gap < 10 * 60 else 0.010 if gap < 2 * 3600 else 0.015)
        if gap < 10 * 60:
            self._add("energy", -0.002)

        activity = float(signals["activity"])
        exclaim = float(signals["exclaim"])
        self._add("stimulation", min(0.015, activity * 0.010 + exclaim * 0.004 + (0.003 if new_topic else 0.0)))
        self._add("focus", 0.005 if signals["technical"] else 0.0)
        self._add("comfort", 0.005 if signals["positive"] or signals["affection"] or signals["thanks"] else 0.0)
        self._add("tension", 0.005 if signals["friction"] or float(signals["question_pressure"]) > 0 else 0.0)
        self._add("playfulness", 0.004 if signals["play"] or signals["tease"] else 0.0)
        self._add("affection", 0.003 if signals["affection"] else 0.001 if signals["thanks"] else 0.0)

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

        if intensity >= 0.78 or (intensity >= 0.54 and now - self._candidate_at >= _MOOD_SHIFT_SECONDS):
            self._mood = mood
            self._mood_intensity = _mix(self._mood_intensity, min(0.82, intensity), 0.35)

    def _mood_target(self, now: float) -> tuple[str, float]:
        energy = self._vars["energy"]
        social = self._vars["social"]
        focus = self._vars["focus"]
        comfort = self._vars["comfort"]
        stimulation = self._vars["stimulation"]
        tension = self._vars["tension"]
        playfulness = self._vars["playfulness"]
        affection = self._vars["affection"]
        late = time.localtime(now).tm_hour >= 23 or time.localtime(now).tm_hour <= 5
        silence = now - self._last_interaction_at
        uptime_hours = max(0.0, now - self._started_at) / 3600.0

        scores = {
            "sleepy": max(0.0, 0.46 - energy) * 1.35 + (0.16 if late else 0.0) + min(0.18, max(0.0, uptime_hours - 4.0) * 0.025),
            "focused": max(0.0, focus - 0.54) * 1.45 + max(0.0, stimulation - 0.42) * 0.70 - tension * 0.20,
            "playful": max(0.0, playfulness - 0.54) * 1.35 + max(0.0, energy - 0.44) * 0.28 + max(0.0, 0.34 - tension) * 0.25,
            "warm": max(0.0, comfort - 0.58) * 1.0 + max(0.0, affection - 0.44) * 1.35,
            "lonely": max(0.0, 0.40 - social) * 1.55 + (min(0.20, (silence - 2 * 3600) / (10 * 3600) * 0.20) if silence > 2 * 3600 else 0.0),
            "concerned": max(0.0, tension - 0.46) * 1.45 + max(0.0, 0.43 - comfort) * 1.1,
            "restless": max(0.0, stimulation - 0.52) * 1.0 + max(0.0, tension - 0.40) * 1.0 + max(0.0, energy - 0.42) * 0.25 + max(0.0, 0.46 - focus) * 0.35,
        }
        mood, score = max(scores.items(), key=lambda item: item[1])
        return ("calm", _CALM_INTENSITY) if score < 0.10 else (mood, _clamp(0.34 + score * 0.72))

    def _decay_emotion(self, session: dict[str, object], now: float, *, count_turn: bool) -> None:
        elapsed = now - _float(session.get("updated_at"), now)
        values = self._emotion_vars(session)
        if elapsed > 0:
            drift = _alpha(elapsed, 35 * 60)
            values["stimulation"] = _mix(values["stimulation"], self._vars["stimulation"], drift)
            values["playfulness"] = _mix(values["playfulness"], self._vars["playfulness"], drift)
            values["tension"] = _mix(values["tension"], self._vars["tension"], drift)
            values["comfort"] = _mix(values["comfort"], self._vars["comfort"], _alpha(elapsed, 90 * 60))
            values["affection"] = _mix(values["affection"], self._vars["affection"], _alpha(elapsed, 90 * 60))

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
        affection = values["affection"]
        last_rare = _float(session.get("rare_at"), 0.0)
        if signals["affection"] and comfort > 0.58 and now - last_rare > 20 * 60:
            self._set_emotion(session, "embarrassed", 0.36)
            session["rare_at"] = now
            return

        choices: list[tuple[str, float]] = []
        if signals["tired"]:
            choices.append(("concerned" if tension > 0.36 else "soft", 0.34 + tension * 0.35))
        if signals["positive"]:
            choices.append(("pleased", 0.34 + comfort * 0.25))
        if signals["friction"]:
            choices.append(("annoyed" if tension > 0.34 or int(session.get("problem_run", 0) or 0) >= 2 else "concerned", 0.34 + tension * 0.45))
        if signals["play"]:
            choices.append(("amused", 0.34 + playfulness * 0.35))
        if signals["tease"]:
            choices.append(("smug", 0.34 + playfulness * 0.30))
        if signals["affection"]:
            choices.append(("soft", 0.30 + affection * 0.28))
        if signals["thanks"]:
            choices.append(("soft", 0.32 + comfort * 0.22))
        if signals["technical"]:
            choices.append(("focused", 0.30 + stimulation * 0.25))
        if choices:
            emotion, score = max(choices, key=lambda item: item[1])
            self._set_emotion(session, emotion, score)
            return

        scores = {
            "amused": playfulness * 0.65 + comfort * 0.25,
            "concerned": tension * 0.75 + max(0.0, 0.50 - comfort) * 0.9,
            "soft": affection * 0.45 + comfort * 0.30,
            "focused": stimulation * 0.55 + self._vars["focus"] * 0.30 - tension * 0.20,
        }
        thresholds = {"amused": 0.46, "concerned": 0.36, "soft": 0.48, "focused": 0.48}
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
        emotion = _valid(session.get("emotion") if session else None, EMOTIONS, "neutral")
        variables = {name: round(self._vars[name], 3) for name in DEFAULT_VARS}
        return {
            "mood": self._mood,
            "mood_intensity": round(self._mood_intensity, 3),
            "emotion": emotion,
            "emotion_intensity": round(_clamp(session.get("intensity") if session else 0.0), 3),
            "emotion_turns": int(session.get("turns", 0) if session else 0),
            "variables": variables,
            **variables,
            "updated_at": self._updated_at,
            "last_interaction_at": self._last_interaction_at,
            "allowed_moods": list(MOODS),
            "allowed_emotions": list(EMOTIONS),
        }

    def _save(self) -> None:
        payload = {
            "version": 3,
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


def tone_guidance(state: dict[str, object]) -> str:
    mood = _valid(state.get("mood"), MOODS, "calm")
    emotion = _valid(state.get("emotion"), EMOTIONS, "neutral")
    words: list[str] = []
    for word in (*_MOOD_TONE.get(mood, _MOOD_TONE["calm"]), *_EMOTION_TONE.get(emotion, ())):
        if word not in words:
            words.append(word)
    if emotion == "neutral" and mood == "calm":
        words.append("normal Akane tone")
    if "compact" not in words:
        words.append("compact")
    return ", ".join(words[:5])


def format_for_prompt(state: dict[str, object]) -> str:
    tone = tone_guidance(state)
    return (
        "[AKANE EMOTION STATE]\n"
        f"Tone: {tone}. Do not mention internal state. "
        "Tone must not create facts, activities, scenery, questions, or fake memories."
    )
