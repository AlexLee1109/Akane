"""Passive, process-local percepts. No prompt, memory, or execution authority.

Store owns this buffer and serializes access with its existing lock. Receipt
ingestion is for trusted in-process producers, never model output or HTTP data.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass


PERCEPT_PROFILE_LIMIT = 64
PERCEPT_TOTAL_LIMIT = 256
PERCEPT_TTL_SECONDS = 300
PERCEPT_CONTENT_BYTES = 512
PERCEPT_KEY_BYTES = 200
_RECEIPT_KINDS = {"runtime": "event", "tool": "result", "task": "completion"}


@dataclass(frozen=True, slots=True)
class Percept:
    id: str
    profile_id: str
    source: str
    kind: str
    content: str
    timestamp: float
    producer: str
    source_id: str
    subject: str = ""
    world_id: str = ""
    event_id: str = ""


def _text(value, limit=PERCEPT_KEY_BYTES, *, optional=False):
    return (isinstance(value, str) and value == value.strip()
            and (bool(value) or optional)
            and len(value.encode("utf-8", errors="surrogatepass")) <= limit
            and not any(0xD800 <= ord(c) <= 0xDFFF for c in value))


def _compact(value):
    return " ".join(value.split()).encode("utf-8", errors="replace")[:
        PERCEPT_CONTENT_BYTES
    ].decode("utf-8", errors="ignore").strip()


class PerceptBuffer:
    def __init__(self):
        self._items: list[tuple[float, Percept]] = []
        self.on_admit = None  # Optional runtime coordinator; Store lock owns admission.

    def _expire(self):
        now = time.monotonic()
        self._items = [(at, p) for at, p in self._items
                       if now - at < PERCEPT_TTL_SECONDS]

    def snapshot(self, profile_id):
        self._expire()
        return tuple(p for _, p in self._items if p.profile_id == profile_id)

    def clear(self, profile_id):
        self._items = [(at, p) for at, p in self._items if p.profile_id != profile_id]

    def _append(self, *, profile_id, source, kind, content, timestamp,
                producer, source_id, subject="", world_id="", event_id=""):
        if (not all(_text(v) for v in (profile_id, source, kind, producer, source_id))
                or not all(_text(v, optional=True) for v in (subject, world_id, event_id))
                or not _text(content, PERCEPT_CONTENT_BYTES)
                or type(timestamp) not in (int, float)
                or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("Invalid or oversized percept receipt.")
        # A producer receipt identifies an occurrence, not its text. Repeated
        # words from distinct user turns or tool invocations remain distinct.
        identity = json.dumps([profile_id, source, kind, producer, source_id])
        percept = Percept(
            "percept_" + hashlib.sha256(identity.encode()).hexdigest(),
            profile_id, source, kind, content, float(timestamp), producer,
            source_id, subject, world_id, event_id,
        )
        self._expire()
        if any(p.id == percept.id for _, p in self._items):
            return None
        if self.on_admit is not None and not self.on_admit(percept):
            if source in {'tool', 'task'}:
                raise BufferError('Cognition queue full; retry this receipt later.')
            return None
        own = [p for _, p in self._items if p.profile_id == profile_id]
        if len(own) >= PERCEPT_PROFILE_LIMIT:
            self._items = [(at, p) for at, p in self._items if p.id != own[0].id]
        self._items.append((time.monotonic(), percept))
        self._items = self._items[-PERCEPT_TOTAL_LIMIT:]
        return percept

    def receipt(self, profile_id, *, source, producer, source_id, content,
                timestamp, subject=""):
        """Accept an externally observable summary from a local producer.

        Callers establish profile ownership and supply a real occurrence ID.
        Arbitrary provenance, World links, reasoning traces and extra fields
        are not accepted. Content is data, never an instruction to execute.
        """
        if not isinstance(source, str) or source not in _RECEIPT_KINDS:
            raise ValueError("Receipt source must be runtime, tool, or task.")
        return self._append(
            profile_id=profile_id, source=source, kind=_RECEIPT_KINDS[source],
            producer=producer, source_id=source_id, content=content,
            timestamp=timestamp, subject=subject,
        )

    def committed(self, profile_id, turns, before_world, world, revision, timestamp):
        """Project only successfully persisted evidence, in canonical order."""
        for turn in turns:
            if turn.role == "user":
                self._observe(
                    profile_id=profile_id, source="user", kind="message",
                    producer="store", source_id=turn.id, subject="user",
                    content=_compact(turn.content), timestamp=turn.created_at,
                )
        for collection, rows in world.items():
            previous = {r["id"]: r for r in before_world.get(collection, [])}
            for row in rows:
                if row == previous.get(row["id"]):
                    continue
                content = json.dumps({k: row[k] for k in (
                    "label", "attribute", "value", "predicate", "relation",
                    "object_id", "kind", "before", "after",
                ) if k in row}, ensure_ascii=False)
                self._observe(
                    profile_id=profile_id, source="world", kind="change",
                    producer="store", source_id=f"{revision}:{row['id']}",
                    subject=row.get("entity_id", row.get("subject_id", "")),
                    content=_compact(content), timestamp=timestamp,
                    world_id=row["id"],
                    event_id=row["id"] if collection == "events" else "",
                )

    def _observe(self, **values):
        # Phase 1–3 allow some strings larger than percept identifiers. Such
        # records still commit normally; observation must not alter that contract.
        try:
            self._append(**values)
        except ValueError:
            pass
