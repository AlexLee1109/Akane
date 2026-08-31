"""Small text-stream helpers for downstream speech consumers."""

from __future__ import annotations

import queue
import re
import threading


_SENTENCE_BOUNDARY = re.compile(r"[.!?][\"'\u2019\u201d)\]]*\s+")
_CLAUSE_BOUNDARY = re.compile(r"[,;:\u2014][\"'\u2019\u201d)\]]*\s+")
_ABBREVIATIONS = frozenset({
    "dr", "e.g", "etc", "i.e", "jr", "mr", "mrs", "ms", "prof", "sr", "st", "vs",
})
SEMANTIC_SIDECAR_START = "\n<akane:evidence>"
SEMANTIC_SIDECAR_END = "</akane:evidence>"
_SEMANTIC_SIDECAR_BARE_START = SEMANTIC_SIDECAR_START.lstrip()


class SemanticSidecarFilter:
    """Forward spoken text while withholding an optional metadata suffix."""

    def __init__(self) -> None:
        self._pending = ""
        self._metadata = ""
        self._trailing = ""
        self._collecting = False
        self._closed = False
        self._complete = False

    @property
    def semantic_evidence(self) -> str:
        return self._metadata if self._complete else ""

    def feed(self, text: str) -> str:
        chunk = str(text or "")
        if not chunk:
            return ""
        if self._closed:
            self._trailing += chunk
            return ""
        if self._collecting:
            self._metadata += chunk
            self._close_metadata()
            return ""
        self._pending += chunk
        markers = tuple(
            (position, marker)
            for marker in (SEMANTIC_SIDECAR_START, _SEMANTIC_SIDECAR_BARE_START)
            if (position := self._pending.find(marker)) >= 0
        )
        if markers:
            marker, marker_text = min(markers, key=lambda item: item[0])
            spoken = self._pending[:marker]
            self._metadata = self._pending[marker + len(marker_text):]
            self._pending = ""
            self._collecting = True
            self._close_metadata()
            return spoken
        held = 0
        starts = (SEMANTIC_SIDECAR_START, _SEMANTIC_SIDECAR_BARE_START)
        maximum = min(len(self._pending), max(map(len, starts)) - 1)
        for length in range(maximum, 0, -1):
            if any(
                length < len(marker) and self._pending.endswith(marker[:length])
                for marker in starts
            ):
                held = length
                break
        spoken = self._pending[:-held] if held else self._pending
        self._pending = self._pending[-held:] if held else ""
        return spoken

    def finish(self) -> str:
        if self._collecting:
            self._complete = self._closed and not self._trailing.strip()
            self._metadata = self._metadata if self._complete else ""
            return ""
        spoken = self._pending
        self._pending = ""
        return spoken

    def _close_metadata(self) -> None:
        marker = self._metadata.find(SEMANTIC_SIDECAR_END)
        if marker < 0:
            return
        self._trailing = self._metadata[marker + len(SEMANTIC_SIDECAR_END):]
        self._metadata = self._metadata[:marker]
        self._closed = True


class SpeakableChunker:
    """Emit unchanged text at conservative sentence or long-clause boundaries."""

    def __init__(self, *, minimum_clause_chars: int = 48) -> None:
        self._minimum_clause_chars = max(1, int(minimum_clause_chars))
        self._buffer = ""

    @staticmethod
    def _safe_period(text: str, position: int) -> bool:
        if position and position + 1 < len(text):
            if text[position - 1].isdigit() and text[position + 1].isdigit():
                return False
        prefix = text[:position + 1].rstrip()
        token = prefix.rsplit(maxsplit=1)[-1].strip("\"'\u2018\u2019\u201c\u201d([{")
        lowered = token.casefold()
        bare = lowered.rstrip(".")
        if bare in _ABBREVIATIONS or len(bare) == 1:
            return False
        if re.fullmatch(r"(?:[a-z]\.){2,}", lowered):
            return False
        if "://" in lowered or lowered.startswith("www.") or "@" in lowered:
            return False
        return token.count(".") <= 1

    def _boundary(self) -> int | None:
        candidates: list[int] = []
        for match in _SENTENCE_BOUNDARY.finditer(self._buffer):
            punctuation = match.start()
            if self._buffer[punctuation] != "." or self._safe_period(self._buffer, punctuation):
                candidates.append(match.end())
        for match in _CLAUSE_BOUNDARY.finditer(self._buffer):
            if len(self._buffer[:match.end()].strip()) >= self._minimum_clause_chars:
                candidates.append(match.end())
        return min(candidates) if candidates else None

    def feed(self, text: str) -> tuple[str, ...]:
        self._buffer += str(text)
        chunks: list[str] = []
        while (boundary := self._boundary()) is not None:
            chunks.append(self._buffer[:boundary])
            self._buffer = self._buffer[boundary:]
        return tuple(chunks)

    def flush(self) -> str:
        chunk = self._buffer
        self._buffer = ""
        return chunk


class SpeakableDispatcher:
    """Deliver ordered speech chunks without holding up model generation."""

    def __init__(self, callback) -> None:
        self._callback = callback
        self._queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="AkaneSpeakableCallback",
        )
        self._thread.start()

    def submit(self, text: str) -> None:
        if text and not self._closed:
            self._queue.put(text)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)

    def _run(self) -> None:
        error_logged = False
        while (chunk := self._queue.get()) is not None:
            try:
                self._callback(chunk)
            except Exception as exc:
                if not error_logged:
                    print(
                        "[Akane:streaming:speakable-callback-error] "
                        f"type={type(exc).__name__}",
                        flush=True,
                    )
                    error_logged = True
