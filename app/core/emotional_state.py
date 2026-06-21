"""Bounded, decaying emotional state used to shape Akane's live tone."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from app.core.emotion_dynamics import (
    BASELINES,
    EMOTIONS,
    MOODS,
    TONE_COLOR,
    apply_oppositions,
    clamp,
    decay,
    derive_moods,
    event_deltas,
    valid,
)

STATE_PATH = Path(os.environ.get(
    "AKANE_EMOTIONAL_STATE_PATH",
    Path(__file__).resolve().parent.parent / "emotional_state.json",
))

_MAX_SESSIONS = 64
_STALE_SECONDS = 6 * 3600
def _attention_for(
    session_id: str | None,
    text: str,
    attention: dict[str, object] | None,
    now: float,
) -> dict[str, object]:
    if attention is not None:
        return attention
    try:
        from app.core.attention_state import preview_attention

        return preview_attention(session_id, text, now=now)
    except Exception:
        return {}


class EmotionalState:
    def __init__(self, path: Path = STATE_PATH) -> None:
        now = time.time()
        self.path = Path(path)
        self._lock = threading.RLock()
        self._vars = dict(BASELINES)
        self._mood = "calm"
        self._secondary = "curious"
        self._mood_intensity = 0.27
        self._updated_at = now
        self._last_interaction_at = now
        self._sessions: dict[str, dict[str, object]] = {}
        self._dirty = False
        self.reload()

    @staticmethod
    def _session_key(session_id: str | None) -> str:
        return (str(session_id or "default").strip() or "default")[:120]

    def _session(self, key: str, now: float) -> dict[str, object]:
        return self._sessions.setdefault(
            key,
            {
                "trigger": "",
                "failure_streak": 0,
                "top_changes": [],
                "tone_line": "",
                "tone_updated_at": 0.0,
                "updated_at": now,
            },
        )

    def reload(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        with self._lock:
            variables = data.get("variables")
            if isinstance(variables, dict):
                migrated = dict(BASELINES)
                for name, baseline in BASELINES.items():
                    migrated[name] = clamp(variables.get(name), baseline)
                if "warmth" not in variables:
                    migrated["warmth"] = clamp(variables.get("comfort"), BASELINES["warmth"])
                    migrated["curiosity"] = clamp(variables.get("stimulation"), BASELINES["curiosity"])
                self._vars = migrated
            old_mood = {
                "sleepy": "tired",
                "playful": "amused",
                "warm": "pleased",
                "restless": "uncertain",
                "lonely": "calm",
            }.get(str(data.get("mood") or ""), data.get("mood"))
            self._mood = valid(old_mood, MOODS, "calm")
            self._secondary = valid(data.get("secondary"), MOODS, "curious")
            self._mood_intensity = clamp(data.get("mood_intensity"), 0.27, high=1.0)
            sessions = data.get("sessions")
            if isinstance(sessions, dict):
                for raw_key, raw_state in list(sessions.items())[:_MAX_SESSIONS]:
                    if not isinstance(raw_state, dict):
                        continue
                    self._sessions[self._session_key(raw_key)] = {
                        "trigger": str(raw_state.get("trigger") or "")[:40],
                        "failure_streak": min(6, max(0, int(raw_state.get("failure_streak") or 0))),
                        "top_changes": list(raw_state.get("top_changes") or [])[:3],
                        "tone_line": str(raw_state.get("tone_line") or "")[:420],
                        "tone_updated_at": float(raw_state.get("tone_updated_at") or 0.0),
                        "updated_at": float(raw_state.get("updated_at") or self._updated_at),
                    }
            self._updated_at = float(data.get("updated_at") or self._updated_at)
            self._last_interaction_at = float(data.get("last_interaction_at") or self._updated_at)
            apply_oppositions(self._vars)

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

    def _drift(self, now: float) -> None:
        self._vars = decay(self._vars, max(0.0, now - self._updated_at))
        apply_oppositions(self._vars)
        self._updated_at = now
        self._mood, self._secondary, self._mood_intensity = derive_moods(self._vars)

    def _preview_values(
        self,
        key: str,
        text: str,
        attention: dict[str, object],
        now: float,
    ) -> tuple[dict[str, float], str, list[tuple[str, float]]]:
        values = decay(self._vars, max(0.0, now - self._updated_at))
        session = self._sessions.get(key) or {}
        trigger, deltas, _streak = event_deltas(
            text,
            attention,
            int(session.get("failure_streak") or 0),
            values["comfort"],
        )
        for name, amount in deltas.items():
            values[name] = clamp(values[name] + amount)
        apply_oppositions(values)
        changes = sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        return values, trigger, changes

    def _build_tone_line(
        self,
        values: dict[str, float],
        attention: dict[str, object],
    ) -> str:
        primary, secondary, _intensity = derive_moods(values)
        stance = str(attention.get("stance") or "warm")
        rhythm = "compact and conversational"
        if primary in {"focused", "concerned"}:
            rhythm = "compact, concrete, and decisive"
        elif primary == "tired":
            rhythm = "short, unhurried, and dry"
        elif primary in {"amused", "smug", "flustered"}:
            rhythm = "quick and naturally reactive"
        parts = [
            f"Tone: stance={stance}; rhythm={rhythm}; color={TONE_COLOR[primary]}",
        ]
        if secondary != primary and secondary != "calm":
            parts.append(f"secondary color={TONE_COLOR[secondary]}")
        if attention.get("casual_chat"):
            parts.append("use statements, not questions; no question marks")
        if attention.get("code_help"):
            parts.append("solve the task precisely without slipping into customer-support voice")
        parts.append("show the feeling through wording; never name internal state")
        return "; ".join(parts)[:420].rstrip(" ,;") + "."

    def tone_line_for_message(
        self,
        session_id: str | None,
        text: str,
        attention: dict[str, object] | None = None,
    ) -> str:
        key = self._session_key(session_id)
        now = time.time()
        current_attention = _attention_for(session_id, text, attention, now)
        with self._lock:
            self._prune(now)
            values, _trigger, _changes = self._preview_values(key, text, current_attention, now)
            return self._build_tone_line(values, current_attention)

    def get_cached_tone_line(self, session_id: str | None) -> str:
        key = self._session_key(session_id)
        now = time.time()
        with self._lock:
            self._prune(now)
            session = self._session(key, now)
            line = str(session.get("tone_line") or "").strip()
            if not line:
                line = self._build_tone_line(self._vars, {})
                session["tone_line"] = line
                session["tone_updated_at"] = now
            return line

    def update_cached_tone_line(
        self,
        session_id: str | None,
        attention: dict[str, object] | None = None,
    ) -> str:
        key = self._session_key(session_id)
        now = time.time()
        with self._lock:
            self._prune(now)
            session = self._session(key, now)
            line = self._build_tone_line(self._vars, attention or {})
            session["tone_line"] = line
            session["tone_updated_at"] = now
            self._dirty = True
            return line

    def observe_user_message(
        self,
        session_id: str | None,
        text: str,
        *,
        attention: dict[str, object] | None = None,
        now: float | None = None,
        persist: bool = True,
    ) -> dict[str, object]:
        current = time.time() if now is None else float(now)
        key = self._session_key(session_id)
        current_attention = _attention_for(session_id, text, attention, current)
        with self._lock:
            self._drift(current)
            self._prune(current)
            session = self._session(key, current)
            trigger, deltas, failure_streak = event_deltas(
                text,
                current_attention,
                int(session.get("failure_streak") or 0),
                self._vars["comfort"],
            )
            for name, amount in deltas.items():
                self._vars[name] = clamp(self._vars[name] + amount)
            apply_oppositions(self._vars)
            self._mood, self._secondary, self._mood_intensity = derive_moods(self._vars)
            session["trigger"] = trigger
            session["failure_streak"] = failure_streak
            session["top_changes"] = [
                {"name": name, "delta": round(delta, 3)}
                for name, delta in sorted(
                    deltas.items(),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[:3]
            ]
            session["updated_at"] = current
            self._last_interaction_at = current
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
        with self._lock:
            self._drift(current)
            self._prune(current)
            session = self._session(key, current)
            if text:
                self._vars["tension"] = clamp(self._vars["tension"] - 0.008)
            apply_oppositions(self._vars)
            self._mood, self._secondary, self._mood_intensity = derive_moods(self._vars)
            session["updated_at"] = current
            self._dirty = True
            if persist:
                self._save()
            return self._snapshot(key, current)

    def _snapshot(self, key: str, now: float) -> dict[str, object]:
        session = self._sessions.get(key) or {}
        variables = {name: round(value, 3) for name, value in self._vars.items()}
        return {
            "mood": self._mood,
            "primary_mood": self._mood,
            "mood_intensity": round(self._mood_intensity, 3),
            "emotion": self._secondary,
            "secondary_emotion": self._secondary,
            "emotion_intensity": round(self._mood_intensity * 0.75, 3),
            "emotion_turns": 0,
            "trigger": str(session.get("trigger") or ""),
            "top_changes": list(session.get("top_changes") or [])[:3],
            "failure_streak": int(session.get("failure_streak") or 0),
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
            "version": 5,
            "mood": self._mood,
            "secondary": self._secondary,
            "mood_intensity": round(self._mood_intensity, 4),
            "variables": {name: round(value, 4) for name, value in self._vars.items()},
            "sessions": self._sessions,
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


def observe_user_message(
    session_id: str | None,
    text: str,
    *,
    attention: dict[str, object] | None = None,
    persist: bool = True,
) -> dict[str, object]:
    return _STATE.observe_user_message(session_id, text, attention=attention, persist=persist)


def observe_assistant_reply(session_id: str | None, text: str, *, persist: bool = True) -> dict[str, object]:
    return _STATE.observe_assistant_reply(session_id, text, persist=persist)


def get_cached_tone_line(session_id: str | None) -> str:
    return _STATE.get_cached_tone_line(session_id)


def tone_line_for_message(
    session_id: str | None,
    text: str,
    attention: dict[str, object] | None = None,
) -> str:
    return _STATE.tone_line_for_message(session_id, text, attention)


def update_cached_tone_line(
    session_id: str | None,
    attention: dict[str, object] | None = None,
) -> str:
    return _STATE.update_cached_tone_line(session_id, attention)


def snapshot(session_id: str | None = None) -> dict[str, object]:
    return _STATE.snapshot(session_id)


def persist() -> None:
    _STATE.persist()


def tone_guidance(state: dict[str, object]) -> str:
    mood = valid(state.get("mood"), MOODS, "calm")
    return TONE_COLOR[mood]


def format_for_prompt(state: dict[str, object]) -> str:
    mood = valid(state.get("mood"), MOODS, "calm")
    secondary = valid(state.get("secondary_emotion") or state.get("emotion"), MOODS, "curious")
    return (
        f"Tone: {TONE_COLOR[mood]}; secondary color={TONE_COLOR[secondary]}. "
        "Use this subtly and never name internal state."
    )
