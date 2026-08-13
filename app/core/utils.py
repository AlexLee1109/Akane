"""Shared low-level helpers for Akane's hot path."""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[a-z0-9_+#./-]+", re.IGNORECASE)
OWNER_PROFILE_ID = "local:owner"
_RESERVED_STATE_NAMES = (
    "memory_update",
    "emotion_update",
    "mood_update",
    "relationship_update",
    "opinion_update",
    "preference_update",
    "memory_ops",
    "communication_ops",
    "opinion_ops",
    "self_model_ops",
    "improvement_ops",
    "strategy_ops",
    "preferences",
    "interests",
    "relationship",
)
_RESERVED_STATE_PATTERN = "|".join(
    re.escape(name) for name in _RESERVED_STATE_NAMES
)
_MALFORMED_STATE_PREFIX = re.compile(
    rf'(?:{_RESERVED_STATE_PATTERN})[ \t]*\{{|'
    rf'"(?:{_RESERVED_STATE_PATTERN})"[ \t]*:',
    re.IGNORECASE,
)


class VisibleReplyStream:
    """Release dialogue while quarantining canonical and reserved state payloads."""

    _MARKERS = ("<AKANE_", "</AKANE_")

    def __init__(self) -> None:
        self._pending = ""
        self._finished = False
        self._released_any = False
        self._released_boundary_char = ""

    def _record_released(self, value: str) -> None:
        if not value:
            return
        self._released_any = True
        for character in reversed(value):
            if character not in " \t":
                self._released_boundary_char = character
                return

    def _is_payload_boundary(self, value: str, position: int) -> bool:
        cursor = position - 1
        while cursor >= 0 and value[cursor] in " \t":
            cursor -= 1
        if cursor >= 0:
            return value[cursor] in ".!?\r\n"
        return (
            not self._released_any
            or self._released_boundary_char in ".!?\r\n"
        )

    def _metadata_start(self, value: str) -> int | None:
        folded = value.casefold()
        positions = [
            position
            for marker in self._MARKERS
            if (position := folded.find(marker.casefold())) >= 0
        ]
        for match in _MALFORMED_STATE_PREFIX.finditer(value):
            if self._is_payload_boundary(value, match.start()):
                positions.append(match.start())
        return min(positions) if positions else None

    @staticmethod
    def _could_be_reserved_prefix(value: str) -> bool:
        folded = value.casefold()
        for name in _RESERVED_STATE_NAMES:
            for prefix in (name, f'"{name}"'):
                if prefix.casefold().startswith(folded):
                    return True
                if folded.startswith(prefix.casefold()):
                    remainder = value[len(prefix):]
                    if remainder and remainder.strip(" \t"):
                        continue
                    return True
        return False

    @staticmethod
    def _quarantine_start(value: str, position: int) -> int:
        while position > 0 and value[position - 1] in " \t\r\n":
            position -= 1
        return position

    def _held_start(self, value: str) -> int | None:
        folded = value.casefold()
        positions: list[int] = []
        for position in range(len(value)):
            suffix = folded[position:]
            if any(marker.casefold().startswith(suffix) for marker in self._MARKERS):
                positions.append(self._quarantine_start(value, position))
            if (
                self._is_payload_boundary(value, position)
                and self._could_be_reserved_prefix(value[position:])
            ):
                positions.append(self._quarantine_start(value, position))

        whitespace = re.search(r"[ \t\r\n]+$", value)
        if (
            whitespace is not None
            and self._is_payload_boundary(value, whitespace.start())
        ):
            positions.append(whitespace.start())
        return min(positions) if positions else None

    def feed(self, chunk: object) -> str:
        if self._finished:
            return ""
        combined = self._pending + str(chunk or "")
        metadata_start = self._metadata_start(combined)
        if metadata_start is not None:
            self._finished = True
            self._pending = ""
            visible = combined[:metadata_start].rstrip()
            self._record_released(visible)
            return visible
        held_start = self._held_start(combined)
        if held_start is None:
            self._pending = ""
            self._record_released(combined)
            return combined
        self._pending = combined[held_start:]
        visible = combined[:held_start]
        self._record_released(visible)
        return visible

    def finish(self) -> str:
        if self._finished:
            return ""
        pending = self._pending
        self._pending = ""
        folded = pending.casefold()
        partial_positions = [
            position
            for position in range(len(pending))
            if any(
                marker.casefold().startswith(folded[position:])
                for marker in self._MARKERS
            )
        ]
        if partial_positions:
            visible = pending[:min(partial_positions)].rstrip()
            self._record_released(visible)
            return visible
        self._record_released(pending)
        return pending


def clean_visible_output(raw: object) -> str:
    """Return only spoken dialogue from one complete model output."""

    try:
        value = raw if isinstance(raw, str) else str(raw or "")
    except Exception:
        value = ""
    stream = VisibleReplyStream()
    return (stream.feed(value) + stream.finish()).strip()


def compact_text(value: object, limit: int = 180) -> str:
    text = " ".join(
        str(value or "").replace("\r", " ").replace("\n", " ").split()
    ).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or text[:limit]


def canonical_profile_id(value: object) -> str:
    profile = compact_text(value, 120) or OWNER_PROFILE_ID
    normalized = profile.casefold()
    return (
        OWNER_PROFILE_ID
        if normalized in {OWNER_PROFILE_ID, "local", "popup", "discord:owner"}
        or normalized.startswith(("local:", "popup:", "discord:user:"))
        else profile
    )


def words(value: object) -> set[str]:
    return {item.lower().strip(".") for item in WORD_RE.findall(str(value or "").lower())}
