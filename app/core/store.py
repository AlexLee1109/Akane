"""Canonical in-memory state and atomic JSON persistence."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path

from app.core.config import SETTINGS
from app.core.mind import (
    find_self_item,
    memory_topic,
    same_experience,
    self_development_state,
    self_topic_terms,
)
from app.core.state import (
    CommitResult,
    EXPERIENCE_KINDS,
    EXPERIENCE_SUBJECTS,
    MEMORY_KINDS,
    MEMORY_SUBJECTS,
    SELF_GROUP_BY_KIND,
    SELF_KINDS,
    SELF_STATUSES,
    Experience,
    Memory,
    MemoryChange,
    SelfChange,
    SelfItem,
    SelfRevision,
    StateChangeProposal,
    StateSnapshot,
    Turn,
)
from app.core.utils import OWNER_PROFILE_ID, lexical_terms, relevance, text_key


SCHEMA_VERSION = 8
CONVERSATION_TURN_LIMIT = max(32, SETTINGS.recent_turn_limit * 2)
MEMORY_LIMIT = 128
EXPERIENCE_LIMIT = 64
EXPERIENCE_RESULT_LIMIT = 1
SELF_ITEM_LIMIT = 64
SELF_REVISION_LIMIT = 4

_ROOT_KEYS = {"schema_version", "revision", "updated_at", "profiles"}
_PROFILE_KEYS = {
    "created_at", "updated_at", "self", "memories", "experiences", "conversations",
}
_SELF_KEYS = {*SELF_GROUP_BY_KIND.values(), "revisions"}
_CONVERSATION_KEYS = {"created_at", "updated_at", "turns"}
_TURN_KEYS = {"id", "role", "content", "created_at", "request_id"}
_MEMORY_KEYS = {
    "id", "subject", "kind", "text", "importance", "confidence",
    "created_at", "updated_at", "source_turn_ids",
}
_EXPERIENCE_KEYS = {
    "id", "kind", "subject", "topic", "what_happened", "akane_response",
    "outcome", "salience", "reason", "created_at", "self_item_ids",
    "source_turn_ids",
}
_SELF_ITEM_KEYS = {
    "id", "kind", "topic", "value", "strength", "confidence", "reason",
    "status", "created_at", "updated_at", "source_ids", "contradiction_ids",
    "revision_count",
}
_SELF_REVISION_KEYS = {
    "id", "self_item_id", "value", "strength", "confidence", "reason",
    "status", "source_ids", "contradiction_ids", "changed_at",
}
_GENERIC_QUERY_TERMS = frozenset({
    "about", "are", "do", "feel", "have", "like", "now", "still", "think",
    "want", "what", "which", "why", "would", "you", "your",
})


def _term_relevance(query_terms: set[str], candidate: object) -> float:
    candidate_terms = lexical_terms(candidate) - _GENERIC_QUERY_TERMS
    shared = len(query_terms & candidate_terms)
    return max(
        shared / max(1, len(query_terms | candidate_terms)),
        shared / max(1, min(len(query_terms), 4)),
    )


class StateIntegrityError(RuntimeError):
    """Canonical state could not be loaded or committed safely."""


def _empty_state() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": 0,
        "updated_at": time.time(),
        "profiles": {},
    }


def _new_profile(now: float) -> dict[str, object]:
    return {
        "created_at": now,
        "updated_at": now,
        "self": {
            "opinions": [],
            "preferences": [],
            "interests": [],
            "goals": [],
            "revisions": [],
        },
        "memories": [],
        "experiences": [],
        "conversations": {},
    }


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise StateIntegrityError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StateIntegrityError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number):
        raise StateIntegrityError(f"{label} must be a finite number.")
    return number


def _identifier(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 200:
        raise StateIntegrityError(f"{label} is invalid.")
    return text


def _mapping(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise StateIntegrityError(f"{label} has an invalid structure.")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise StateIntegrityError(f"{label} must be an array.")
    result = tuple(_identifier(item, label) for item in value)
    if len(result) > 6:
        raise StateIntegrityError(f"{label} exceeds its bound.")
    return result


def _turn_row(turn: Turn) -> dict[str, object]:
    row = asdict(turn)
    row.pop("profile_id")
    row.pop("conversation_id")
    return row


def _memory_row(memory: Memory) -> dict[str, object]:
    row = asdict(memory)
    row.pop("profile_id")
    row["source_turn_ids"] = list(memory.source_turn_ids[-6:])
    return row


def _experience_row(experience: Experience) -> dict[str, object]:
    row = asdict(experience)
    row.pop("profile_id")
    row["self_item_ids"] = list(experience.self_item_ids[-6:])
    row["source_turn_ids"] = list(experience.source_turn_ids[-6:])
    return row


def _self_row(item: SelfItem) -> dict[str, object]:
    row = asdict(item)
    row.pop("profile_id")
    row["source_ids"] = list(item.source_ids[-6:])
    return row


def _revision_row(revision: SelfRevision) -> dict[str, object]:
    row = asdict(revision)
    row.pop("profile_id")
    row["source_ids"] = list(revision.source_ids[-6:])
    return row


def _turn(row: object, profile_id: str, conversation_id: str) -> Turn:
    data = _mapping(row, _TURN_KEYS, "turn")
    role = str(data["role"])
    if role not in {"user", "assistant"}:
        raise StateIntegrityError("Turn role must be user or assistant.")
    content = str(data["content"])
    if not content.strip():
        raise StateIntegrityError("Turn content cannot be empty.")
    return Turn(
        id=_identifier(data["id"], "turn ID"),
        profile_id=profile_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        created_at=_finite(data["created_at"], "turn timestamp"),
        request_id=str(data["request_id"] or "")[:200],
    )


def _memory(row: object, profile_id: str) -> Memory:
    data = _mapping(row, _MEMORY_KEYS, "memory")
    subject = str(data["subject"])
    kind = str(data["kind"])
    if subject not in MEMORY_SUBJECTS or kind not in MEMORY_KINDS:
        raise StateIntegrityError("Memory ownership or kind is invalid.")
    text = str(data["text"])
    if not text.strip():
        raise StateIntegrityError("Memory text cannot be empty.")
    return Memory(
        id=_identifier(data["id"], "memory ID"),
        profile_id=profile_id,
        subject=subject,
        kind=kind,
        text=text,
        importance=_finite(data["importance"], "memory importance"),
        confidence=_finite(data["confidence"], "memory confidence"),
        created_at=_finite(data["created_at"], "memory creation time"),
        updated_at=_finite(data["updated_at"], "memory update time"),
        source_turn_ids=_strings(data["source_turn_ids"], "memory source ID"),
    )


def _experience(row: object, profile_id: str) -> Experience:
    data = _mapping(row, _EXPERIENCE_KEYS, "experience")
    kind = str(data["kind"])
    subject = str(data["subject"])
    if kind not in EXPERIENCE_KINDS or subject not in EXPERIENCE_SUBJECTS:
        raise StateIntegrityError("Experience kind or subject is invalid.")
    topic = str(data["topic"])
    happened = str(data["what_happened"])
    response = str(data["akane_response"])
    outcome = str(data["outcome"])
    if (
        not topic.strip() or not happened.strip() or not response.strip()
        or len(topic) > 120 or len(happened) > 200
        or len(response) > 280 or len(outcome) > 160
    ):
        raise StateIntegrityError("Experience evidence is empty or exceeds its bound.")
    source_turn_ids = _strings(data["source_turn_ids"], "experience source turn ID")
    if not source_turn_ids:
        raise StateIntegrityError("Experience requires source-turn provenance.")
    return Experience(
        id=_identifier(data["id"], "experience ID"),
        profile_id=profile_id,
        kind=kind,
        subject=subject,
        topic=topic,
        what_happened=happened,
        akane_response=response,
        outcome=outcome,
        salience=_finite(data["salience"], "experience salience"),
        reason=_identifier(data["reason"], "experience reason"),
        created_at=_finite(data["created_at"], "experience time"),
        self_item_ids=_strings(data["self_item_ids"], "experience Self ID"),
        source_turn_ids=source_turn_ids,
    )


def _self_item(row: object, profile_id: str) -> SelfItem:
    data = _mapping(row, _SELF_ITEM_KEYS, "Self item")
    kind = str(data["kind"])
    status = str(data["status"])
    if kind not in SELF_KINDS or status not in SELF_STATUSES:
        raise StateIntegrityError("Self kind or status is invalid.")
    topic = str(data["topic"])
    value = str(data["value"])
    if not topic.strip() or not value.strip():
        raise StateIntegrityError("Self topic and value cannot be empty.")
    revision_count = data["revision_count"]
    if isinstance(revision_count, bool) or not isinstance(revision_count, int) or revision_count < 0:
        raise StateIntegrityError("Self revision count is invalid.")
    return SelfItem(
        id=_identifier(data["id"], "Self ID"),
        profile_id=profile_id,
        kind=kind,
        topic=topic,
        value=value,
        strength=_finite(data["strength"], "Self strength"),
        confidence=_finite(data["confidence"], "Self confidence"),
        reason=str(data["reason"]),
        status=status,
        created_at=_finite(data["created_at"], "Self creation time"),
        updated_at=_finite(data["updated_at"], "Self update time"),
        source_ids=_strings(data["source_ids"], "Self source ID"),
        contradiction_ids=_strings(
            data["contradiction_ids"], "Self contradiction ID",
        ),
        revision_count=revision_count,
    )


def _self_revision(row: object, profile_id: str) -> SelfRevision:
    data = _mapping(row, _SELF_REVISION_KEYS, "Self revision")
    status = str(data["status"])
    if status not in SELF_STATUSES:
        raise StateIntegrityError("Self revision status is invalid.")
    return SelfRevision(
        id=_identifier(data["id"], "Self revision ID"),
        self_item_id=_identifier(data["self_item_id"], "Self revision target"),
        profile_id=profile_id,
        value=str(data["value"]),
        strength=_finite(data["strength"], "Self revision strength"),
        confidence=_finite(data["confidence"], "Self revision confidence"),
        reason=str(data["reason"]),
        status=status,
        source_ids=_strings(data["source_ids"], "Self revision source ID"),
        changed_at=_finite(data["changed_at"], "Self revision time"),
        contradiction_ids=_strings(
            data["contradiction_ids"], "Self revision contradiction ID",
        ),
    )


def _self_rows(profile: dict[str, object]):
    state = profile["self"]
    for group in SELF_GROUP_BY_KIND.values():
        yield from state[group]


def _validate_state(value: object) -> None:
    root = _mapping(value, _ROOT_KEYS, "state")
    if root["schema_version"] != SCHEMA_VERSION:
        raise StateIntegrityError(f"State schema must be {SCHEMA_VERSION}.")
    revision = root["revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise StateIntegrityError("State revision is invalid.")
    _finite(root["updated_at"], "state update time")
    profiles = root["profiles"]
    if not isinstance(profiles, dict):
        raise StateIntegrityError("State profiles must be an object.")
    all_turn_ids: set[str] = set()
    for profile_id, raw_profile in profiles.items():
        profile_id = _identifier(profile_id, "profile ID")
        profile = _mapping(raw_profile, _PROFILE_KEYS, "profile")
        _finite(profile["created_at"], "profile creation time")
        _finite(profile["updated_at"], "profile update time")
        self_state = _mapping(profile["self"], _SELF_KEYS, "Self")
        self_ids: set[str] = set()
        active_topics: list[tuple[str, str, frozenset[str]]] = []
        for kind, group in SELF_GROUP_BY_KIND.items():
            rows = self_state[group]
            if not isinstance(rows, list) or len(rows) > SELF_ITEM_LIMIT:
                raise StateIntegrityError("Self group is invalid or exceeds its bound.")
            for row in rows:
                item = _self_item(row, profile_id)
                if item.kind != kind or item.id in self_ids:
                    raise StateIntegrityError("Self grouping or ID uniqueness is invalid.")
                if not 0.0 <= item.strength <= 1.0 or not 0.0 <= item.confidence <= 1.0:
                    raise StateIntegrityError("Self numeric values must be in [0, 1].")
                self_ids.add(item.id)
                if item.status in {"active", "uncertain"}:
                    topic_key = text_key(item.topic)
                    topic_terms = self_topic_terms(item.topic)
                    for existing_kind, existing_key, existing_terms in active_topics:
                        shared = len(topic_terms & existing_terms)
                        coverage = shared / max(1, min(len(topic_terms), len(existing_terms)))
                        if item.kind == existing_kind and (
                            topic_key == existing_key
                            or (item.kind != "preference" and coverage >= 0.67)
                        ):
                            raise StateIntegrityError("Active Self contains a semantic duplicate.")
                    active_topics.append((item.kind, topic_key, topic_terms))
        revisions = self_state["revisions"]
        if not isinstance(revisions, list):
            raise StateIntegrityError("Self revisions must be an array.")
        revision_counts: dict[str, int] = {}
        for row in revisions:
            item = _self_revision(row, profile_id)
            if item.self_item_id not in self_ids:
                raise StateIntegrityError("Self revision target does not exist.")
            revision_counts[item.self_item_id] = revision_counts.get(item.self_item_id, 0) + 1
            if revision_counts[item.self_item_id] > SELF_REVISION_LIMIT:
                raise StateIntegrityError("Self revision history exceeds its bound.")
        memories = profile["memories"]
        if not isinstance(memories, list) or len(memories) > MEMORY_LIMIT:
            raise StateIntegrityError("Memories are invalid or exceed their bound.")
        memory_ids: set[str] = set()
        for row in memories:
            item = _memory(row, profile_id)
            if item.id in memory_ids:
                raise StateIntegrityError("Memory IDs must be unique.")
            if not 0.0 <= item.importance <= 1.0 or not 0.0 <= item.confidence <= 1.0:
                raise StateIntegrityError("Memory numeric values must be in [0, 1].")
            memory_ids.add(item.id)
        experiences = profile["experiences"]
        if not isinstance(experiences, list) or len(experiences) > EXPERIENCE_LIMIT:
            raise StateIntegrityError("Experiences are invalid or exceed their bound.")
        experience_ids: set[str] = set()
        for row in experiences:
            item = _experience(row, profile_id)
            if item.id in experience_ids:
                raise StateIntegrityError("Experience IDs must be unique.")
            if not 0.0 <= item.salience <= 1.0:
                raise StateIntegrityError("Experience salience must be in [0, 1].")
            experience_ids.add(item.id)
        conversations = profile["conversations"]
        if not isinstance(conversations, dict):
            raise StateIntegrityError("Conversations must be an object.")
        for conversation_id, raw_conversation in conversations.items():
            conversation_id = _identifier(conversation_id, "conversation ID")
            conversation = _mapping(raw_conversation, _CONVERSATION_KEYS, "conversation")
            _finite(conversation["created_at"], "conversation creation time")
            _finite(conversation["updated_at"], "conversation update time")
            rows = conversation["turns"]
            if not isinstance(rows, list) or len(rows) > CONVERSATION_TURN_LIMIT:
                raise StateIntegrityError("Conversation turns are invalid or exceed their bound.")
            for row in rows:
                item = _turn(row, profile_id, conversation_id)
                if item.id in all_turn_ids:
                    raise StateIntegrityError("Turn IDs must be globally unique.")
                all_turn_ids.add(item.id)


def _legacy_record(row: object, allowed: set[str]) -> dict[str, object]:
    if not isinstance(row, dict):
        raise StateIntegrityError("Legacy state contains an invalid record.")
    return {key: copy.deepcopy(value) for key, value in row.items() if key in allowed}


def _migrate_legacy(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not isinstance(value.get("profiles"), dict):
        raise StateIntegrityError("Legacy state has no valid profile map.")
    version = value.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or not 1 <= version < SCHEMA_VERSION:
        raise StateIntegrityError(f"Unsupported state schema: {version!r}.")
    migrated = _empty_state()
    migrated["revision"] = int(value.get("revision") or 0)
    migrated["updated_at"] = _finite(value.get("updated_at") or time.time(), "state update time")
    profiles: dict[str, object] = {}
    for profile_id, raw_profile in value["profiles"].items():
        profile_id = _identifier(profile_id, "profile ID")
        if not isinstance(raw_profile, dict):
            raise StateIntegrityError("Legacy profile is invalid.")
        created = _finite(raw_profile.get("created_at") or time.time(), "profile creation time")
        profile = _new_profile(created)
        profile["updated_at"] = _finite(raw_profile.get("updated_at") or created, "profile update time")
        raw_self = raw_profile.get("self")
        items: list[dict[str, object]] = []
        revisions: list[dict[str, object]] = []
        if isinstance(raw_self, dict):
            if set(raw_self) - _SELF_KEYS or "self_items" in raw_profile or "self_revisions" in raw_profile:
                raise StateIntegrityError("Legacy Self structure cannot be migrated losslessly.")
            for group in ("opinions", "preferences", "interests", "goals"):
                rows = raw_self.get(group, [])
                if not isinstance(rows, list):
                    raise StateIntegrityError(f"Legacy Self {group} are invalid.")
                items.extend(rows)
            raw_revisions = raw_self.get("revisions", [])
            if not isinstance(raw_revisions, list):
                raise StateIntegrityError("Legacy Self revisions are invalid.")
            revisions = list(raw_revisions)
        elif "self_items" in raw_profile:
            if not isinstance(raw_profile["self_items"], list):
                raise StateIntegrityError("Legacy Self items are invalid.")
            items = list(raw_profile["self_items"])
            raw_revisions = raw_profile.get("self_revisions", [])
            if not isinstance(raw_revisions, list):
                raise StateIntegrityError("Legacy Self revisions are invalid.")
            revisions = list(raw_revisions)
        elif raw_self is not None:
            raise StateIntegrityError("Legacy Self is invalid.")
        elif "self_revisions" in raw_profile:
            raise StateIntegrityError("Legacy Self revisions have no Self items.")
        accepted: list[SelfItem] = []
        for raw in items:
            if not isinstance(raw, dict):
                raise StateIntegrityError("Legacy Self item is invalid.")
            kind = str(raw.get("kind") or "")
            if kind == "curiosity":
                kind = "interest"
            if kind not in SELF_KINDS:
                raise StateIntegrityError(
                    f"Legacy Self item has unsupported kind: {kind or '<empty>'}."
                )
            normalized = {
                "id": raw.get("id"), "kind": kind, "topic": raw.get("topic"),
                "value": raw.get("value"), "strength": raw.get("strength", 0.4),
                "confidence": raw.get("confidence", 0.6), "reason": raw.get("reason", ""),
                "status": raw.get("status", "active"), "created_at": raw.get("created_at", created),
                "updated_at": raw.get("updated_at", created), "source_ids": raw.get("source_ids", []),
                "contradiction_ids": raw.get("contradiction_ids", []),
                "revision_count": raw.get("revision_count", 0),
            }
            item = _self_item(normalized, profile_id)
            duplicate = find_self_item(tuple(accepted), item.kind, item.topic)
            if duplicate is None:
                accepted.append(item)
            else:
                raise StateIntegrityError(
                    "Legacy Self contains duplicate topics that cannot be migrated losslessly."
                )
        for item in accepted:
            profile["self"][SELF_GROUP_BY_KIND[item.kind]].append(_self_row(item))
        accepted_ids = {item.id for item in accepted}
        migrated_revisions = []
        for raw in revisions:
            if not isinstance(raw, dict):
                raise StateIntegrityError("Legacy Self revision is invalid.")
            if str(raw.get("self_item_id") or "") not in accepted_ids:
                raise StateIntegrityError("Legacy Self revision target cannot be migrated losslessly.")
            normalized = {
                "id": raw.get("id"), "self_item_id": raw.get("self_item_id"),
                "value": raw.get("value", ""), "strength": raw.get("strength", 0.0),
                "confidence": raw.get("confidence", 0.0), "reason": raw.get("reason", ""),
                "status": raw.get("status", "retired"), "source_ids": raw.get("source_ids", []),
                "contradiction_ids": raw.get("contradiction_ids", []),
                "changed_at": raw.get("changed_at", created),
            }
            migrated_revisions.append(_self_revision(normalized, profile_id))
        for item_id in accepted_ids:
            selected = sorted(
                (row for row in migrated_revisions if row.self_item_id == item_id),
                key=lambda row: row.changed_at,
            )
            if len(selected) > SELF_REVISION_LIMIT:
                raise StateIntegrityError("Legacy Self revision history exceeds its bound.")
            profile["self"]["revisions"].extend(_revision_row(row) for row in selected)
        raw_memories = raw_profile.get("memories", [])
        if not isinstance(raw_memories, list):
            raise StateIntegrityError("Legacy memories are invalid.")
        if len(raw_memories) > MEMORY_LIMIT:
            raise StateIntegrityError("Legacy memories exceed the canonical bound.")
        for raw in raw_memories:
            if not isinstance(raw, dict):
                raise StateIntegrityError("Legacy memory is invalid.")
            normalized = {
                "id": raw.get("id"), "subject": raw.get("subject", "user"),
                "kind": raw.get("kind", "fact"), "text": raw.get("text", ""),
                "importance": raw.get("importance", 0.5), "confidence": raw.get("confidence", 0.7),
                "created_at": raw.get("created_at", created), "updated_at": raw.get("updated_at", created),
                "source_turn_ids": raw.get("source_turn_ids", []),
            }
            profile["memories"].append(_memory_row(_memory(normalized, profile_id)))
        raw_experiences = raw_profile.get("experiences", [])
        if not isinstance(raw_experiences, list) or len(raw_experiences) > EXPERIENCE_LIMIT:
            raise StateIntegrityError("Legacy experiences are invalid or exceed their bound.")
        for raw in raw_experiences:
            if not isinstance(raw, dict):
                raise StateIntegrityError("Legacy experience is invalid.")
            normalized = {
                "id": raw.get("id"), "kind": raw.get("kind"),
                "subject": raw.get("subject"), "topic": raw.get("topic"),
                "what_happened": raw.get("what_happened"),
                "akane_response": raw.get("akane_response"),
                "outcome": raw.get("outcome", ""),
                "salience": raw.get("salience", 0.5),
                "reason": raw.get("reason", "legacy:experience"),
                "created_at": raw.get("created_at", created),
                "self_item_ids": raw.get("self_item_ids", []),
                "source_turn_ids": raw.get("source_turn_ids", []),
            }
            profile["experiences"].append(
                _experience_row(_experience(normalized, profile_id))
            )
        raw_conversations = raw_profile.get("conversations", {})
        if not isinstance(raw_conversations, dict):
            raise StateIntegrityError("Legacy conversations are invalid.")
        for conversation_id, raw_conversation in raw_conversations.items():
            conversation_id = _identifier(conversation_id, "conversation ID")
            if not isinstance(raw_conversation, dict) or not isinstance(raw_conversation.get("turns"), list):
                raise StateIntegrityError("Legacy conversation is invalid.")
            rows = []
            for raw in raw_conversation["turns"][-CONVERSATION_TURN_LIMIT:]:
                if not isinstance(raw, dict):
                    raise StateIntegrityError("Legacy turn is invalid.")
                normalized = {
                    "id": raw.get("id"), "role": raw.get("role"),
                    "content": raw.get("content"), "created_at": raw.get("created_at", created),
                    "request_id": raw.get("request_id", ""),
                }
                rows.append(_turn_row(_turn(normalized, profile_id, conversation_id)))
            profile["conversations"][conversation_id] = {
                "created_at": _finite(raw_conversation.get("created_at") or created, "conversation creation time"),
                "updated_at": _finite(raw_conversation.get("updated_at") or created, "conversation update time"),
                "turns": rows,
            }
        profiles[profile_id] = profile
    migrated["profiles"] = profiles
    _validate_state(migrated)
    return migrated


def _broad_self_kinds(query: str) -> frozenset[str]:
    normalized = text_key(query)
    terms = lexical_terms(query)
    if "yourself" in terms or normalized == "who are you":
        return frozenset(SELF_KINDS)
    if normalized == "what have you changed your mind about":
        return frozenset(SELF_KINDS)
    if "your interests" in normalized or "your hobbies" in normalized:
        return frozenset({"interest"})
    if "your opinions" in normalized:
        return frozenset({"opinion"})
    if "your goals" in normalized or normalized == "what do you want":
        return frozenset({"goal"})
    if normalized in {"what do you like", "what do you prefer"}:
        return frozenset({"interest", "preference"})
    return frozenset()


def _memory_query_topic(query: str) -> str:
    normalized = text_key(query)
    if "where" in normalized and any(term in normalized for term in ("i live", "i am from", "i m from")):
        return "residence"
    if "my name" in normalized or "what is my name" in normalized:
        return "name"
    if "my birthday" in normalized:
        return "birthday"
    if "my job" in normalized or "where do i work" in normalized:
        return "work"
    if "my project" in normalized or "working on" in normalized:
        return "project"
    return ""


class Store:
    """One state authority; derived retrieval is computed directly on demand."""

    def __init__(self, path: str | Path = SETTINGS.state_path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._timing = threading.local()
        self._state = self._load_state()

    def _load_state(self) -> dict[str, object]:
        if not self.path.exists():
            return _empty_state()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StateIntegrityError(f"Could not load canonical state: {exc}") from exc
        if isinstance(value, dict) and value.get("schema_version") != SCHEMA_VERSION:
            value = _migrate_legacy(value)
            self._write_state(value)
        _validate_state(value)
        return value

    def _write_state(self, state: dict[str, object]) -> dict[str, float | int]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized_at = time.perf_counter()
        payload = (
            json.dumps(state, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8")
        serialized = time.perf_counter() - serialized_at
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent,
        )
        write_at = time.perf_counter()
        fsync_seconds = 0.0
        replace_seconds = 0.0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                synced_at = time.perf_counter()
                os.fsync(handle.fileno())
                fsync_seconds += time.perf_counter() - synced_at
            replaced_at = time.perf_counter()
            os.replace(temporary, self.path)
            replace_seconds = time.perf_counter() - replaced_at
            try:
                directory = os.open(self.path.parent, os.O_RDONLY)
                try:
                    synced_at = time.perf_counter()
                    os.fsync(directory)
                    fsync_seconds += time.perf_counter() - synced_at
                finally:
                    os.close(directory)
            except OSError:
                pass
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return {
            "serialize": serialized,
            "write": time.perf_counter() - write_at,
            "fsync": fsync_seconds,
            "replace": replace_seconds,
            "json_bytes": len(payload),
        }

    def _candidate(self, profile_id: str, now: float) -> tuple[dict[str, object], dict[str, object]]:
        candidate = dict(self._state)
        candidate["profiles"] = dict(self._state["profiles"])
        existing = candidate["profiles"].get(profile_id)
        profile = copy.deepcopy(existing) if existing is not None else _new_profile(now)
        candidate["profiles"][profile_id] = profile
        return candidate, profile

    def ensure_profile(self, profile_id: str) -> None:
        with self._lock:
            if profile_id in self._state["profiles"]:
                return
            now = time.time()
            candidate, _ = self._candidate(profile_id, now)
            candidate["revision"] = int(self._state["revision"]) + 1
            candidate["updated_at"] = now
            _validate_state(candidate)
            self._write_state(candidate)
            self._state = candidate

    def profile_exists(self, profile_id: str) -> bool:
        with self._lock:
            return profile_id in self._state["profiles"]

    def profile_ids(self, prefix: str = "") -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(
                profile_id for profile_id in self._state["profiles"]
                if not prefix or profile_id.startswith(prefix)
            ))

    @staticmethod
    def _append_turn(profile: dict[str, object], turn: Turn, now: float) -> bool:
        conversations = profile["conversations"]
        conversation = conversations.get(turn.conversation_id)
        if conversation is None:
            conversation = {"created_at": now, "updated_at": now, "turns": []}
            conversations[turn.conversation_id] = conversation
        if any(row["id"] == turn.id for row in conversation["turns"]):
            return False
        conversation["turns"].append(_turn_row(turn))
        if len(conversation["turns"]) > CONVERSATION_TURN_LIMIT:
            conversation["turns"] = conversation["turns"][-CONVERSATION_TURN_LIMIT:]
            while conversation["turns"] and conversation["turns"][0]["role"] == "assistant":
                conversation["turns"].pop(0)
        conversation["updated_at"] = max(now, turn.created_at)
        return True

    @staticmethod
    def _apply_memory(profile: dict[str, object], change: MemoryChange) -> str | None:
        rows = profile["memories"]
        if change.action == "delete":
            target = change.target_id
            before = len(rows)
            rows[:] = [row for row in rows if row["id"] != target]
            return "memory:delete" if len(rows) != before else None
        if change.action != "upsert" or change.memory is None:
            return None
        item = change.memory
        target = change.target_id or item.id
        index = next((i for i, row in enumerate(rows) if row["id"] == target), None)
        if index is not None and rows[index]["subject"] != item.subject:
            index = None
        if index is None:
            topic = memory_topic(item.text)
            duplicate = next((
                i for i, row in enumerate(rows)
                if row["subject"] == item.subject
                and (
                    (topic and memory_topic(row["text"]) == topic)
                    or relevance(row["text"], item.text) >= 0.72
                )
            ), None)
            index = duplicate
        if index is None:
            rows.append(_memory_row(item))
            rows[:] = sorted(
                rows, key=lambda row: (float(row["importance"]), float(row["updated_at"])),
                reverse=True,
            )[:MEMORY_LIMIT]
            return "memory:form"
        current = _memory(rows[index], item.profile_id)
        updated = Memory(
            id=current.id,
            profile_id=item.profile_id,
            subject=item.subject,
            kind=item.kind,
            text=item.text,
            importance=item.importance,
            confidence=item.confidence,
            created_at=current.created_at,
            updated_at=item.updated_at,
            source_turn_ids=item.source_turn_ids[-6:],
        )
        rows[index] = _memory_row(updated)
        return "memory:update"

    @staticmethod
    def _apply_self(profile: dict[str, object], change: SelfChange, now: float) -> str | None:
        if change.item is None or change.action not in {
            "form", "reinforce", "weaken", "revise", "retire", "complete", "abandon",
        }:
            return None
        item = change.item
        all_items = tuple(_self_item(row, item.profile_id) for row in _self_rows(profile))
        target = next((current for current in all_items if current.id == change.target_id), None)
        if target is None:
            target = find_self_item(all_items, item.kind, item.topic)
        if target is None:
            group = profile["self"][SELF_GROUP_BY_KIND[item.kind]]
            group.append(_self_row(item))
            if len(group) > SELF_ITEM_LIMIT:
                group.sort(key=lambda row: (
                    float(row["strength"]),
                    float(row["confidence"]),
                    len(row["source_ids"]),
                    str(row["status"]) == "active",
                    float(row["updated_at"]),
                ), reverse=True)
                del group[SELF_ITEM_LIMIT:]
            return "self:form"
        old_group = profile["self"][SELF_GROUP_BY_KIND[target.kind]]
        old_index = next(i for i, row in enumerate(old_group) if row["id"] == target.id)
        updated = SelfItem(
            id=target.id,
            profile_id=item.profile_id,
            kind=target.kind,
            topic=target.topic,
            value=item.value,
            strength=item.strength,
            confidence=item.confidence,
            reason=item.reason,
            status=item.status,
            created_at=target.created_at,
            updated_at=item.updated_at,
            source_ids=item.source_ids[-6:],
            contradiction_ids=item.contradiction_ids[-6:],
            revision_count=item.revision_count,
        )
        changed_judgment = (
            change.action in {"weaken", "revise", "retire", "complete", "abandon"}
            and (
                target.value != updated.value
                or target.strength != updated.strength
                or target.confidence != updated.confidence
                or target.status != updated.status
            )
        )
        if changed_judgment:
            revision = SelfRevision(
                id=f"revision_{target.id}_{target.revision_count + 1}",
                self_item_id=target.id,
                profile_id=item.profile_id,
                value=target.value,
                strength=target.strength,
                confidence=target.confidence,
                reason=target.reason,
                status=target.status,
                source_ids=target.source_ids[-6:],
                changed_at=now,
                contradiction_ids=target.contradiction_ids[-6:],
            )
            revisions = profile["self"]["revisions"]
            revisions.append(_revision_row(revision))
            matching = [row for row in revisions if row["self_item_id"] == target.id]
            for stale in matching[:-SELF_REVISION_LIMIT]:
                revisions.remove(stale)
        old_group[old_index] = _self_row(updated)
        return f"self:{change.action}"

    @staticmethod
    def _apply_experience(profile: dict[str, object], item: Experience) -> str:
        rows = profile["experiences"]
        if rows:
            previous = _experience(rows[-1], item.profile_id)
            if same_experience(previous, item):
                return "experience:duplicate"
        rows.append(_experience_row(item))
        protected = {
            source_id
            for self_row in _self_rows(profile)
            for source_id in (
                *self_row["source_ids"], *self_row["contradiction_ids"],
            )
        }
        if item.self_item_ids:
            protected.add(item.id)
        while len(rows) > EXPERIENCE_LIMIT:
            removable = next((
                index for index, row in enumerate(rows)
                if row["id"] not in protected
            ), 0)
            rows.pop(removable)
        return "experience:form"

    def commit(self, proposal: StateChangeProposal, *, now: float | None = None) -> CommitResult:
        current = time.time() if now is None else float(now)
        lock_started = time.perf_counter()
        with self._lock:
            lock_acquired = time.perf_counter()
            candidate, profile = self._candidate(proposal.profile_id, current)
            applied: list[str] = []
            rejected = list(proposal.rejected)
            mutation_started = time.perf_counter()
            for turn in proposal.turns:
                if turn.profile_id != proposal.profile_id:
                    rejected.append("turn:profile-mismatch")
                    continue
                if self._append_turn(profile, turn, current):
                    applied.append(f"turn:{turn.role}")
                else:
                    rejected.append("turn:duplicate")
            for change in proposal.memories:
                if change.memory is not None and change.memory.profile_id != proposal.profile_id:
                    rejected.append("memory:profile-mismatch")
                    continue
                result = self._apply_memory(profile, change)
                (applied if result else rejected).append(result or "memory:invalid")
            proposed_experience_ids = {item.id for item in proposal.experiences}
            accepted_experience_ids: set[str] = set()
            for experience in proposal.experiences:
                if experience.profile_id != proposal.profile_id:
                    rejected.append("experience:profile-mismatch")
                    continue
                result = self._apply_experience(profile, experience)
                if result == "experience:form":
                    applied.append(result)
                    accepted_experience_ids.add(experience.id)
                else:
                    rejected.append(result)
            for change in proposal.self_items:
                if change.item is not None and change.item.profile_id != proposal.profile_id:
                    rejected.append("self:profile-mismatch")
                    continue
                evidence_ids = set(
                    (*change.item.source_ids, *change.item.contradiction_ids)
                    if change.item is not None else ()
                )
                linked_proposals = evidence_ids & proposed_experience_ids
                if linked_proposals - accepted_experience_ids:
                    rejected.append("self:evidence-rejected")
                    continue
                result = self._apply_self(profile, change, current)
                (applied if result else rejected).append(result or "self:invalid")
            mutation_seconds = time.perf_counter() - mutation_started
            if not applied:
                return CommitResult(
                    int(self._state["revision"]), (), tuple(rejected),
                    {"lock_wait": lock_acquired - lock_started, "lock_held": time.perf_counter() - lock_acquired},
                )
            profile["updated_at"] = current
            candidate["schema_version"] = SCHEMA_VERSION
            candidate["revision"] = int(self._state["revision"]) + 1
            candidate["updated_at"] = current
            validation_started = time.perf_counter()
            _validate_state(candidate)
            validation_seconds = time.perf_counter() - validation_started
            write_metrics = self._write_state(candidate)
            self._state = candidate
            lock_held = time.perf_counter() - lock_acquired
        return CommitResult(
            int(candidate["revision"]), tuple(applied), tuple(rejected),
            {
                "lock_wait": lock_acquired - lock_started,
                "lock_held": lock_held,
                "conversation_mutation": mutation_seconds,
                "validate": validation_seconds,
                **write_metrics,
            },
        )

    def snapshot(
        self,
        profile_id: str,
        conversation_id: str,
        *,
        query: str = "",
        now: float | None = None,
    ) -> StateSnapshot:
        del now
        started = time.perf_counter()
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            revision = int(self._state["revision"])
            if profile is None:
                self._timing.value = {
                    "selection_seconds": time.perf_counter() - started,
                    "memory_candidates": 0,
                    "self_candidates": 0,
                    "experience_candidates": 0,
                }
                return StateSnapshot(
                    profile_id, conversation_id, (), (), (), (), (), SCHEMA_VERSION, revision,
                )
            conversation = profile["conversations"].get(conversation_id)
            recent = tuple(
                _turn(row, profile_id, conversation_id)
                for row in (conversation["turns"][-SETTINGS.recent_turn_limit:] if conversation else [])
            )
            self_rows = tuple(
                row for row in _self_rows(profile)
                if row["status"] in {"active", "uncertain"}
            )
            memory_rows = tuple(profile["memories"])
            experience_rows = tuple(profile["experiences"])
            query_terms = lexical_terms(query) - _GENERIC_QUERY_TERMS
            broad = _broad_self_kinds(query)
            memory_topic_query = _memory_query_topic(query)
            self_ranked = []
            for row in self_rows:
                score = _term_relevance(
                    query_terms, f"{row['topic']} {row['value']}",
                ) if query_terms else 0.0
                if row["kind"] in broad:
                    score = max(score, 0.2 + float(row["strength"]) * 0.1)
                if score >= 0.16:
                    self_ranked.append((
                        score, float(row["strength"]), float(row["confidence"]),
                        float(row["updated_at"]), str(row["id"]), row,
                    ))
            memory_ranked = []
            for row in memory_rows:
                score = _term_relevance(query_terms, row["text"]) if query_terms else 0.0
                if memory_topic_query and memory_topic(str(row["text"])) == memory_topic_query:
                    score = max(score, 0.3)
                if score >= 0.16:
                    memory_ranked.append((score, float(row["updated_at"]), str(row["id"]), row))
            experience_ranked = []
            for row in experience_rows:
                evidence = " ".join((
                    str(row["topic"]), str(row["what_happened"]),
                    str(row["akane_response"]), str(row["outcome"]),
                ))
                score = _term_relevance(query_terms, evidence) if query_terms else 0.0
                if score >= 0.24:
                    experience_ranked.append((
                        score, float(row["salience"]), float(row["created_at"]),
                        str(row["id"]), row,
                    ))
            selected_self = tuple(
                _self_item(row, profile_id)
                for _, _, _, _, _, row in sorted(self_ranked, reverse=True)[
                    :SETTINGS.self_result_limit
                ]
            )
            selected_memories = tuple(
                _memory(row, profile_id)
                for _, _, _, row in sorted(memory_ranked, reverse=True)[:SETTINGS.memory_result_limit]
            )
            selected_experiences = tuple(
                _experience(row, profile_id)
                for _, _, _, _, row in sorted(experience_ranked, reverse=True)[
                    :EXPERIENCE_RESULT_LIMIT
                ]
            )
            include_history = bool(lexical_terms(query) & {"before", "change", "changed", "mind", "used"})
            selected_ids = {item.id for item in selected_self}
            revisions = tuple(
                _self_revision(row, profile_id)
                for row in profile["self"]["revisions"]
                if include_history and row["self_item_id"] in selected_ids
            )
        self._timing.value = {
            "selection_seconds": time.perf_counter() - started,
            "memory_candidates": len(memory_rows),
            "self_candidates": len(self_rows),
            "experience_candidates": len(experience_rows),
        }
        return StateSnapshot(
            profile_id, conversation_id, recent, selected_memories,
            selected_experiences, selected_self, revisions, SCHEMA_VERSION, revision,
        )

    def snapshot_timing(self) -> dict[str, float | int]:
        return dict(getattr(self._timing, "value", {}))

    def messages(self, conversation_id: str, profile_id: str) -> list[dict[str, str]]:
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            conversation = profile["conversations"].get(conversation_id) if profile else None
            return [
                {"role": str(row["role"]), "content": str(row["content"])}
                for row in (conversation["turns"] if conversation else [])
            ]

    def self_items(self, profile_id: str) -> tuple[SelfItem, ...]:
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            if profile is None:
                return ()
            return tuple(
                _self_item(row, profile_id) for row in _self_rows(profile)
                if row["status"] in {"active", "uncertain"}
            )

    def memories(self, profile_id: str) -> tuple[Memory, ...]:
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            return tuple(_memory(row, profile_id) for row in profile["memories"]) if profile else ()

    def experiences(self, profile_id: str) -> tuple[Experience, ...]:
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            return (
                tuple(_experience(row, profile_id) for row in profile["experiences"])
                if profile else ()
            )

    def latest_turn_at(self, profile_id: str, *, role: str = "") -> float | None:
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            if profile is None:
                return None
            values = (
                float(turn["created_at"])
                for conversation in profile["conversations"].values()
                for turn in conversation["turns"]
                if not role or turn["role"] == role
            )
            return max(values, default=None)

    def latest_conversation_id(self, profile_id: str) -> str | None:
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            if profile is None or not profile["conversations"]:
                return None
            return str(max(
                profile["conversations"].items(),
                key=lambda pair: float(pair[1]["updated_at"]),
            )[0])

    def reply_for_request(self, conversation_id: str, profile_id: str, request_id: str) -> str | None:
        if not request_id:
            return None
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            conversation = profile["conversations"].get(conversation_id) if profile else None
            matches = [
                row for row in (conversation["turns"] if conversation else [])
                if row["role"] == "assistant" and row["request_id"] == request_id
            ]
            return str(matches[-1]["content"]) if matches else None

    def public_conversation(self, conversation_id: str, profile_id: str) -> dict[str, object]:
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            conversation = profile["conversations"].get(conversation_id) if profile else None
            return {
                "id": conversation_id,
                "profile_id": profile_id,
                "created_at": float(conversation["created_at"]) if conversation else 0.0,
                "updated_at": float(conversation["updated_at"]) if conversation else 0.0,
            }

    def public_memory(self, profile_id: str) -> dict[str, object]:
        memories = self.memories(profile_id)
        self_items = self.self_items(profile_id)
        return {
            "user": {},
            "preferences": [
                {"content": item.text, "kind": item.kind}
                for item in memories if item.subject == "user"
            ],
            "facts": [
                {"content": item.text, "subject": item.subject, "kind": item.kind}
                for item in memories
            ],
            "self": [
                {"kind": item.kind, "topic": item.topic, "value": item.value}
                for item in self_items
            ],
        }

    def clear_conversation(self, conversation_id: str, profile_id: str) -> None:
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            if profile is None or conversation_id not in profile["conversations"]:
                return
            now = time.time()
            candidate, copied = self._candidate(profile_id, now)
            copied["conversations"].pop(conversation_id)
            copied["updated_at"] = now
            candidate["revision"] = int(self._state["revision"]) + 1
            candidate["updated_at"] = now
            _validate_state(candidate)
            self._write_state(candidate)
            self._state = candidate

    def clear_profile(self, profile_id: str) -> None:
        with self._lock:
            if profile_id not in self._state["profiles"]:
                return
            now = time.time()
            candidate = dict(self._state)
            candidate["profiles"] = dict(self._state["profiles"])
            if profile_id == OWNER_PROFILE_ID:
                candidate["profiles"][profile_id] = _new_profile(now)
            else:
                candidate["profiles"].pop(profile_id)
            candidate["revision"] = int(self._state["revision"]) + 1
            candidate["updated_at"] = now
            _validate_state(candidate)
            self._write_state(candidate)
            self._state = candidate

    def debug_snapshot(self, profile_id: str, conversation_id: str, query: str = "") -> dict[str, object]:
        state = self.snapshot(profile_id, conversation_id, query=query)
        all_self = self.self_items(profile_id)
        development = tuple(self_development_state(item) for item in all_self)
        return {
            "schema_version": state.schema_version,
            "revision": state.revision,
            "recent_turn_count": len(state.recent_turns),
            "selected_self": [f"{item.kind}:{item.topic}" for item in state.self_items],
            "self_counts": {
                group: sum(1 for item in all_self if SELF_GROUP_BY_KIND[item.kind] == group)
                for group in SELF_GROUP_BY_KIND.values()
            },
            "self_count": len(all_self),
            "self_development_counts": {
                label: development.count(label)
                for label in ("weak", "reinforced", "established", "uncertain")
            },
            "selected_memories": [item.id for item in state.memories],
            "memory_count": len(self.memories(profile_id)),
            "selected_experiences": [item.id for item in state.experiences],
            "experience_count": len(self.experiences(profile_id)),
        }

    def state_bytes(self) -> int:
        with self._lock:
            return len(json.dumps(
                self._state, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8"))


_STORE: Store | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = Store()
    return _STORE
