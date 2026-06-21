"""Lightweight mood state with compact prompt guidance."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

MOODS = ("calm", "sleepy", "focused", "playful", "warm", "lonely", "concerned", "restless")
EMOTIONS = (
    "neutral",
    "amused",
    "smug",
    "flustered",
    "embarrassed",
    "concerned",
    "sympathetic",
    "pleased",
    "excited",
    "surprised",
    "curious",
    "annoyed",
    "soft",
    "focused",
)
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
    "sleepy": "shorter, drier, lower-energy",
    "focused": "direct, practical, compact",
    "playful": "lightly teasing, a little smug",
    "warm": "warmer, more familiar, compact",
    "lonely": "soft, restrained, compact",
    "concerned": "careful, grounded, less playful",
    "restless": "slightly sharper, shorter, controlled",
}
_EMOTION_TONE = {
    "amused": "lightly humorous",
    "smug": "a little smug",
    "flustered": "briefly flustered but warm",
    "embarrassed": "warmer",
    "concerned": "careful",
    "sympathetic": "gently sympathetic",
    "pleased": "lightly satisfied",
    "excited": "brighter and more animated",
    "surprised": "visibly surprised",
    "curious": "genuinely intrigued",
    "annoyed": "mildly blunt, not rude",
    "soft": "gentler",
    "focused": "direct",
}
_EMOTION_EXPRESSION = {
    "amused": "Let a quick dry laugh or teasing beat show.",
    "smug": "Let a tiny smug edge show.",
    "flustered": "Let brief flustered warmth leak into the wording.",
    "embarrassed": "Let a small embarrassed stumble show.",
    "concerned": "Lead with grounded concern.",
    "sympathetic": "Acknowledge the weight of it with gentle warmth.",
    "pleased": "Treat it like a real little victory.",
    "excited": "Give the reply visible momentum and delight.",
    "surprised": "Open with natural surprise, then answer.",
    "curious": "Sound genuinely interested and mentally engaged.",
    "annoyed": "Let mild irritation sharpen the phrasing without being mean.",
    "soft": "Let the warmth be noticeable but restrained.",
    "focused": "Sound invested and decisive.",
}
_NEGATIVE_WORDS = (
    "annoying", "frustrating", "irritating", "hate", "angry", "mad",
    "upset", "stop", "wtf", "ugh",
)
_PROBLEM_WORDS = ("error", "failed", "broken", "stuck", "bug", "issue", "problem")
_SUCCESS_WORDS = ("fixed", "worked", "success", "finally", "great", "nice", "good")
_EXCITED_WORDS = (
    "awesome", "amazing", "perfect", "excited", "hyped", "lets go", "let's go",
)
_AFFECTION_WORDS = (
    "cute", "adorable", "pretty", "beautiful", "love you", "best girl",
)
_DISTRESS_WORDS = (
    "sad", "lonely", "scared", "hurt", "crying", "rough day", "awful day",
)
_SURPRISE_WORDS = ("wow", "whoa", "no way", "wait what", "seriously")
_CURIOUS_WORDS = ("interesting", "curious", "wondering", "what if")
_PLAYFUL_WORDS = (
    "lol", "haha", "funny", "joke", "tease", "silly", "what you doing",
    "what are you doing",
)
_WARM_WORDS = ("thanks", "thank you", "appreciate", "hug")
_FOCUS_WORDS = (
    "code", "model", "prompt", "latency", "server", "discord", "personality",
    "design", "feature", "fix", "debug", "ttft", "cache", "generation",
)
_WORD_PUNCT = str.maketrans({char: " " for char in ".,!?;:()[]{}\"'`"})


def _clamp(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _valid(value: object, allowed: tuple[str, ...], default: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in allowed else default


def _has_term(lower: str, words: set[str], terms: tuple[str, ...]) -> bool:
    return any(term in lower if " " in term else term in words for term in terms)


def _detect_reaction(lower: str, words: set[str]) -> tuple[str, float]:
    if _has_term(lower, words, _NEGATIVE_WORDS):
        return "annoyed", 0.58
    if _has_term(lower, words, _DISTRESS_WORDS):
        return "sympathetic", 0.52
    if _has_term(lower, words, _PROBLEM_WORDS):
        return "concerned", 0.48
    if _has_term(lower, words, _EXCITED_WORDS):
        return "excited", 0.58
    if _has_term(lower, words, _AFFECTION_WORDS):
        return "flustered", 0.48
    if _has_term(lower, words, _SUCCESS_WORDS):
        return "pleased", 0.46
    if _has_term(lower, words, _PLAYFUL_WORDS):
        return "amused", 0.46
    if lower.strip() in {"hi", "hello", "hey"} or _has_term(lower, words, _WARM_WORDS):
        return "soft", 0.32
    if _has_term(lower, words, _SURPRISE_WORDS):
        return "surprised", 0.46
    if _has_term(lower, words, _CURIOUS_WORDS):
        return "curious", 0.42
    if _has_term(lower, words, _FOCUS_WORDS):
        return "focused", 0.44
    return "neutral", 0.0


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
            sessions = data.get("sessions")
            if isinstance(sessions, dict):
                self._sessions = {}
                for raw_key, raw_state in list(sessions.items())[:_MAX_SESSIONS]:
                    if not isinstance(raw_state, dict):
                        continue
                    key = self._session_key(raw_key)
                    self._sessions[key] = {
                        "emotion": _valid(raw_state.get("emotion"), EMOTIONS, "neutral"),
                        "intensity": _clamp(raw_state.get("intensity")),
                        "turns": max(0, int(raw_state.get("turns") or 0)),
                        "focus_hint": str(raw_state.get("focus_hint") or "")[:80],
                        "tone_line": str(raw_state.get("tone_line") or "")[:240],
                        "tone_updated_at": float(raw_state.get("tone_updated_at") or 0.0),
                        "updated_at": float(raw_state.get("updated_at") or self._updated_at),
                    }
            self._updated_at = float(data.get("updated_at") or self._updated_at)
            self._last_interaction_at = float(data.get("last_interaction_at") or self._updated_at)

    def _prune(self, now: float) -> None:
        for key, state in list(self._sessions.items()):
            if now - float(state.get("updated_at") or 0.0) > _STALE_SECONDS:
                self._sessions.pop(key, None)
        if len(self._sessions) > _MAX_SESSIONS:
            oldest = sorted(
                self._sessions.items(),
                key=lambda item: float(item[1].get("updated_at") or 0.0),
            )
            for key, _state in oldest[:len(self._sessions) - _MAX_SESSIONS]:
                self._sessions.pop(key, None)

    @staticmethod
    def _session_key(session_id: str | None) -> str:
        return (str(session_id or "default").strip() or "default")[:120]

    def _session(self, key: str, now: float) -> dict[str, object]:
        return self._sessions.setdefault(
            key,
            {
                "emotion": "neutral",
                "intensity": 0.0,
                "turns": 0,
                "focus_hint": "",
                "tone_line": "",
                "tone_updated_at": 0.0,
                "updated_at": now,
            },
        )

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
        elif values["tension"] > 0.52:
            mood = "concerned"
        elif values["focus"] > 0.52:
            mood = "focused"
        elif values["playfulness"] > 0.38 and values["tension"] < 0.40:
            mood = "playful"
        elif values["comfort"] > 0.64 and values["affection"] > 0.42:
            mood = "warm"
        elif values["stimulation"] > 0.66:
            mood = "restless"
        else:
            mood = "calm"
        self._mood = mood
        self._mood_intensity = 0.6 if mood == "calm" else 0.45

    def _focus_hint(self, lower: str, words: set[str]) -> str:
        if _has_term(lower, words, ("personality", "design", "appearance", "live2d")):
            return "focused on Akane's design"
        if _has_term(
            lower,
            words,
            ("code", "server", "debug", "ttft", "cache", "latency", "model", "prompt"),
        ):
            return "focused on the build"
        if _has_term(lower, words, ("joke", "funny", "lol", "haha", "tease")):
            return "open to playful banter"
        return ""

    def _build_tone_line(
        self,
        key: str,
        *,
        emotion_override: str | None = None,
        intensity_override: float | None = None,
        focus_override: str | None = None,
    ) -> str:
        session = self._sessions.get(key) or {}
        mood = self._mood
        emotion = _valid(
            session.get("emotion") if emotion_override is None else emotion_override,
            EMOTIONS,
            "neutral",
        )
        intensity = _clamp(
            session.get("intensity") if intensity_override is None else intensity_override
        )
        if emotion == "annoyed" and intensity >= 0.15:
            parts = [_EMOTION_TONE["annoyed"]]
        elif emotion == "concerned" and intensity >= 0.15:
            parts = ["careful, grounded"]
        else:
            parts = [_MOOD_TONE[mood]]

        extra = _EMOTION_TONE.get(emotion)
        if (
            emotion != "neutral"
            and intensity >= 0.15
            and extra
            and extra not in parts[0]
        ):
            parts.append(extra)
        if self._vars["stimulation"] > 0.42 and mood != "focused":
            parts.append("mentally engaged")
        if (
            self._vars["comfort"] > 0.62
            and mood != "warm"
            and emotion not in {"annoyed", "concerned"}
        ):
            parts.append("comfortable")

        focus_hint = str(
            session.get("focus_hint") if focus_override is None else focus_override
        ).strip()
        if focus_hint:
            parts.append(focus_hint)

        deduped: list[str] = []
        seen: set[str] = set()
        for part in parts:
            clean = " ".join(str(part or "").split()).strip(" ,.")
            key_part = clean.lower()
            if clean and key_part not in seen:
                seen.add(key_part)
                deduped.append(clean)
        line = "Tone: " + ", ".join(deduped[:4]) + "."
        expression = _EMOTION_EXPRESSION.get(emotion) if intensity >= 0.15 else ""
        if expression:
            line += f" {expression} Do not name the emotion."
        if len(line) <= 240:
            return line
        return line[:240].rsplit(" ", 1)[0].rstrip(" ,.;:") + "."

    def tone_line_for_message(self, session_id: str | None, text: str) -> str:
        key = self._session_key(session_id)
        now = time.time()
        lower = str(text or "").lower()
        words = set(lower.translate(_WORD_PUNCT).split())
        emotion, intensity = _detect_reaction(lower, words)
        with self._lock:
            self._prune(now)
            focus_hint = self._focus_hint(lower, words)
            return self._build_tone_line(
                key,
                emotion_override=emotion if emotion != "neutral" else None,
                intensity_override=intensity if emotion != "neutral" else None,
                focus_override=focus_hint or None,
            )

    def get_cached_tone_line(self, session_id: str | None) -> str:
        key = self._session_key(session_id)
        now = time.time()
        with self._lock:
            self._prune(now)
            session = self._session(key, now)
            line = str(session.get("tone_line") or "").strip()
            if not line:
                line = self._build_tone_line(key)
                session["tone_line"] = line
                session["tone_updated_at"] = now
            return line

    def update_cached_tone_line(self, session_id: str | None) -> str:
        key = self._session_key(session_id)
        now = time.time()
        with self._lock:
            self._prune(now)
            self._session(key, now)
            line = self._build_tone_line(key)
            self._sessions[key]["tone_line"] = line
            self._sessions[key]["tone_updated_at"] = now
            self._dirty = True
            return line

    def observe_user_message(
        self,
        session_id: str | None,
        text: str,
        *,
        now: float | None = None,
        persist: bool = True,
    ) -> dict[str, object]:
        current = time.time() if now is None else float(now)
        key = self._session_key(session_id)
        lower = str(text or "").lower()
        words = set(lower.translate(_WORD_PUNCT).split())
        with self._lock:
            self._drift(current)
            self._prune(current)
            session = self._session(key, current)
            previous_emotion = _valid(session.get("emotion"), EMOTIONS, "neutral")
            previous_intensity = _clamp(session.get("intensity"))
            turns = max(0, int(session.get("turns") or 0) - 1)

            emotion, intensity = _detect_reaction(lower, words)
            focus_hint = self._focus_hint(lower, words)
            if emotion == "annoyed":
                self._vars["tension"] = _clamp(self._vars["tension"] + 0.12)
                self._vars["comfort"] = _clamp(self._vars["comfort"] - 0.03)
            elif emotion == "sympathetic":
                self._vars["tension"] = _clamp(self._vars["tension"] + 0.03)
                self._vars["affection"] = _clamp(self._vars["affection"] + 0.04)
                self._vars["comfort"] = _clamp(self._vars["comfort"] + 0.02)
            elif emotion == "concerned":
                self._vars["tension"] = _clamp(self._vars["tension"] + 0.06)
                self._vars["focus"] = _clamp(self._vars["focus"] + 0.05)
            elif emotion == "excited":
                self._vars["energy"] = _clamp(self._vars["energy"] + 0.04)
                self._vars["playfulness"] = _clamp(self._vars["playfulness"] + 0.12)
                self._vars["comfort"] = _clamp(self._vars["comfort"] + 0.05)
            elif emotion == "flustered":
                self._vars["playfulness"] = _clamp(self._vars["playfulness"] + 0.06)
                self._vars["comfort"] = _clamp(self._vars["comfort"] + 0.04)
                self._vars["affection"] = _clamp(self._vars["affection"] + 0.05)
            elif emotion == "pleased":
                self._vars["comfort"] = _clamp(self._vars["comfort"] + 0.05)
                self._vars["affection"] = _clamp(self._vars["affection"] + 0.02)
            elif emotion == "amused":
                self._vars["playfulness"] = _clamp(self._vars["playfulness"] + 0.14)
                self._vars["comfort"] = _clamp(self._vars["comfort"] + 0.02)
            elif emotion == "soft":
                self._vars["comfort"] = _clamp(self._vars["comfort"] + 0.025)
                self._vars["affection"] = _clamp(self._vars["affection"] + 0.02)
            elif emotion == "surprised":
                self._vars["stimulation"] = _clamp(self._vars["stimulation"] + 0.12)
                self._vars["playfulness"] = _clamp(self._vars["playfulness"] + 0.04)
            elif emotion == "curious":
                self._vars["focus"] = _clamp(self._vars["focus"] + 0.05)
                self._vars["stimulation"] = _clamp(self._vars["stimulation"] + 0.08)
            elif emotion == "focused":
                self._vars["focus"] = _clamp(self._vars["focus"] + 0.10)
                self._vars["stimulation"] = _clamp(self._vars["stimulation"] + 0.08)
                if _has_term(lower, words, ("personality", "design")):
                    self._vars["playfulness"] = _clamp(self._vars["playfulness"] + 0.04)

            if emotion != "neutral":
                session.update({"emotion": emotion, "intensity": intensity, "turns": _EMOTION_TURNS})
            elif turns > 0 and previous_emotion != "neutral":
                session.update({
                    "emotion": previous_emotion,
                    "intensity": max(0.16, previous_intensity * 0.85),
                    "turns": turns,
                })
            else:
                session.update({"emotion": "neutral", "intensity": 0.0, "turns": 0})
            if focus_hint or emotion in {"annoyed", "concerned"}:
                session["focus_hint"] = focus_hint
            self._vars["social"] = _clamp(self._vars["social"] + 0.01)
            friendly = _has_term(
                lower,
                words,
                (*_WARM_WORDS, *_PLAYFUL_WORDS, *_SUCCESS_WORDS),
            )
            if not _has_term(lower, words, _NEGATIVE_WORDS):
                self._vars["comfort"] = _clamp(
                    self._vars["comfort"] + (0.010 if friendly else 0.003)
                )
                self._vars["affection"] = _clamp(
                    self._vars["affection"] + (0.007 if friendly else 0.002)
                )
            self._vars["stimulation"] = _clamp(self._vars["stimulation"] + min(0.04, len(lower) / 6000))
            self._vars["tension"] = _clamp(self._vars["tension"] * 0.985)
            session["updated_at"] = current
            self._last_interaction_at = current
            self._updated_at = current
            self._choose_mood(current)
            self._dirty = True
            if persist:
                self._save()
            return self._snapshot(key, current)

    def observe_assistant_reply(
        self,
        session_id: str | None,
        text: str,
        *,
        now: float | None = None,
        persist: bool = True,
    ) -> dict[str, object]:
        current = time.time() if now is None else float(now)
        key = self._session_key(session_id)
        lower = str(text or "").lower()
        with self._lock:
            self._drift(current)
            self._prune(current)
            session = self._session(key, current)
            self._vars["comfort"] = _clamp(self._vars["comfort"] + 0.006)
            self._vars["affection"] = _clamp(self._vars["affection"] + 0.004)
            self._vars["stimulation"] = _clamp(self._vars["stimulation"] + min(0.015, len(lower) / 12000))
            if any(word in lower for word in ("sorry", "careful", "fix", "debug")):
                self._vars["focus"] = _clamp(self._vars["focus"] + 0.01)
            session["updated_at"] = current
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
            "cached_tone_line": str(session.get("tone_line") or "").strip(),
            "tone_line_updated_at": float(session.get("tone_updated_at") or 0.0),
            "last_emotion_update_at": float(session.get("updated_at") or 0.0),
            "variables": variables,
            **variables,
            "updated_at": self._updated_at,
            "last_interaction_at": self._last_interaction_at,
            "allowed_moods": list(MOODS),
            "allowed_emotions": list(EMOTIONS),
        }

    def snapshot(self, session_id: str | None = None, *, now: float | None = None) -> dict[str, object]:
        current = time.time() if now is None else float(now)
        key = self._session_key(session_id)
        with self._lock:
            self._drift(current)
            self._prune(current)
            return self._snapshot(key, current)

    def _save(self) -> None:
        payload = {
            "version": 4,
            "mood": self._mood,
            "mood_intensity": round(self._mood_intensity, 4),
            "variables": {name: round(value, 4) for name, value in self._vars.items()},
            "sessions": self._serializable_sessions(),
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

    def _serializable_sessions(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for key, state in self._sessions.items():
            out[key] = {
                "emotion": _valid(state.get("emotion"), EMOTIONS, "neutral"),
                "intensity": round(_clamp(state.get("intensity")), 4),
                "turns": max(0, int(state.get("turns") or 0)),
                "focus_hint": str(state.get("focus_hint") or "")[:80],
                "tone_line": str(state.get("tone_line") or "")[:240],
                "tone_updated_at": float(state.get("tone_updated_at") or 0.0),
                "updated_at": float(state.get("updated_at") or self._updated_at),
            }
        return out


_STATE = EmotionalState()


def get_emotional_state() -> EmotionalState:
    return _STATE


def observe_user_message(session_id: str | None, text: str, *, persist: bool = True) -> dict[str, object]:
    return _STATE.observe_user_message(session_id, text, persist=persist)


def observe_assistant_reply(session_id: str | None, text: str, *, persist: bool = True) -> dict[str, object]:
    return _STATE.observe_assistant_reply(session_id, text, persist=persist)


def get_cached_tone_line(session_id: str | None) -> str:
    return _STATE.get_cached_tone_line(session_id)


def tone_line_for_message(session_id: str | None, text: str) -> str:
    return _STATE.tone_line_for_message(session_id, text)


def update_cached_tone_line(session_id: str | None) -> str:
    return _STATE.update_cached_tone_line(session_id)


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
