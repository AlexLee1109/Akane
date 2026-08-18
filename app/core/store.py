"""The only canonical persistence authority for Akane v2."""

from __future__ import annotations

import copy
import heapq
import json
import math
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, fields
from pathlib import Path

from app.core.character import load_character_profile
from app.core.config import SETTINGS
from app.core.state import (
    CommitResult,
    MEMORY_KINDS,
    MEMORY_SUBJECTS,
    SELF_GROUP_BY_KIND,
    SELF_KINDS,
    SELF_STATUSES,
    THOUGHT_STATUSES,
    Memory,
    Mood,
    ProactiveMessage,
    Relationship,
    SelfItem,
    SelfRevision,
    StateChangeProposal,
    StateSnapshot,
    Thought,
    Turn,
    clamp,
)
from app.core.utils import lexical_terms, log_timing, relevance, text_key

SCHEMA_VERSION = 3
CHANGE_LOG_LIMIT = 200
_REFLECTION_RETRY_BASE_SECONDS = 120
_REFLECTION_RETRY_MAX_SECONDS = 1800
_ROOT_KEYS = {
    "schema_version",
    "revision",
    "updated_at",
    "profiles",
    "reflection_jobs",
    "proactive_queue",
    "change_log",
}
_PROFILE_KEYS = {
    "created_at",
    "updated_at",
    "self",
    "mood",
    "relationship",
    "memories",
    "inner_life",
    "conversations",
}
_SELF_KEYS = {*SELF_GROUP_BY_KIND.values(), "revisions"}
_INNER_LIFE_KEYS = {"thoughts"}
_V2_PROFILE_KEYS = {
    "created_at",
    "updated_at",
    "conversations",
    "memories",
    "self_items",
    "self_revisions",
    "mood",
    "relationship",
    "thoughts",
}
_CONVERSATION_KEYS = {"created_at", "updated_at", "turns", "reflection_empty_streak"}
_REFLECTION_JOB_KEYS = {
    "id",
    "profile_id",
    "conversation_id",
    "first_turn_id",
    "latest_turn_id",
    "turn_count",
    "ready",
    "status",
    "attempts",
    "available_at",
    "error",
    "created_at",
    "updated_at",
}
_REFLECTION_JOB_ORDER = (
    "id", "profile_id", "conversation_id", "status", "ready", "turn_count",
    "first_turn_id", "latest_turn_id", "attempts", "available_at", "created_at",
    "updated_at", "error",
)
_PROACTIVE_KEYS = {
    "id",
    "profile_id",
    "thought_id",
    "text",
    "importance",
    "created_at",
    "status",
    "claim_token",
    "claimed_at",
    "adapter",
    "conversation_id",
    "delivered_at",
    "message_id",
}
_PROACTIVE_ORDER = (
    "id", "profile_id", "thought_id", "text", "importance", "created_at", "status",
    "claim_token", "claimed_at", "adapter", "conversation_id", "delivered_at", "message_id",
)
_CHANGE_LOG_KEYS = {"id", "profile_id", "origin", "applied", "rejected", "created_at"}
_CHANGE_LOG_ORDER = ("id", "profile_id", "origin", "applied", "rejected", "created_at")
_RECORD_FIELD_ORDER = {
    Turn: ("id", "role", "content", "created_at", "request_id", "profile_id", "conversation_id"),
    Memory: (
        "id", "subject", "kind", "text", "importance", "confidence", "created_at",
        "updated_at", "source_turn_ids", "profile_id",
    ),
    SelfItem: (
        "id", "topic", "value", "strength", "confidence", "reason", "created_at",
        "updated_at", "source_ids", "kind", "status", "revision_count", "profile_id",
    ),
    SelfRevision: (
        "id", "self_item_id", "value", "strength", "confidence", "reason", "status",
        "source_ids", "changed_at", "profile_id",
    ),
    Thought: (
        "id", "topic", "text", "importance", "status", "started_at", "updated_at",
        "source_ids", "share_worthy", "profile_id",
    ),
    Mood: ("emotion", "valence", "energy", "cause", "updated_at", "profile_id"),
    Relationship: (
        "familiarity", "trust", "closeness", "interaction_notes", "unresolved_events",
        "updated_at", "profile_id",
    ),
}


class StateIntegrityError(RuntimeError):
    """A canonical state document could not be safely loaded or committed."""


def _empty_state() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": 0,
        "updated_at": time.time(),
        "profiles": {},
        "reflection_jobs": [],
        "proactive_queue": [],
        "change_log": [],
    }


def _migrate_v1_state(state: dict[str, object]) -> dict[str, object]:
    """Coalesce v1 per-pair jobs while preserving all durable state."""

    migrated = copy.deepcopy(state)
    profiles = migrated.get("profiles")
    old_jobs = migrated.get("reflection_jobs")
    if not isinstance(profiles, dict) or not isinstance(old_jobs, list):
        raise StateIntegrityError("State schema 1 cannot be migrated safely.")

    turn_locations: dict[str, tuple[str, str, int]] = {}
    for profile_id, profile in profiles.items():
        if not isinstance(profile, dict) or not isinstance(profile.get("conversations"), dict):
            raise StateIntegrityError("State schema 1 has invalid conversations.")
        for conversation_id, conversation in profile["conversations"].items():
            if not isinstance(conversation, dict) or not isinstance(conversation.get("turns"), list):
                raise StateIntegrityError("State schema 1 has invalid turns.")
            conversation["reflection_empty_streak"] = 0
            for index, turn in enumerate(conversation["turns"]):
                if isinstance(turn, dict):
                    turn_locations[str(turn.get("id") or "")] = (
                        str(profile_id), str(conversation_id), index,
                    )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for raw in old_jobs:
        if not isinstance(raw, dict) or raw.get("status") not in {"pending", "running", "failed"}:
            continue
        user_location = turn_locations.get(str(raw.get("user_turn_id") or ""))
        assistant_location = turn_locations.get(str(raw.get("assistant_turn_id") or ""))
        if user_location is None or assistant_location is None or user_location[:2] != assistant_location[:2]:
            continue
        grouped.setdefault(user_location[:2], []).append(raw)

    pending = []
    for (profile_id, conversation_id), jobs in grouped.items():
        conversation = profiles[profile_id]["conversations"][conversation_id]
        turns = conversation["turns"]
        indices = []
        for job in jobs:
            indices.extend((
                turn_locations[str(job["user_turn_id"])][2],
                turn_locations[str(job["assistant_turn_id"])][2],
            ))
        first_index, latest_index = min(indices), max(indices)
        created_at = min(float(job.get("created_at") or 0.0) for job in jobs)
        updated_job = max(jobs, key=lambda item: float(item.get("updated_at") or 0.0))
        pending.append({
            "id": str(jobs[0].get("id") or uuid.uuid4().hex),
            "profile_id": profile_id,
            "conversation_id": conversation_id,
            "first_turn_id": str(turns[first_index]["id"]),
            "latest_turn_id": str(turns[latest_index]["id"]),
            "turn_count": latest_index - first_index + 1,
            "ready": True,
            "status": "pending",
            "attempts": max(int(job.get("attempts") or 0) for job in jobs),
            "available_at": max(float(job.get("available_at") or 0.0) for job in jobs),
            "error": str(updated_job.get("error") or "")[:500],
            "created_at": created_at,
            "updated_at": float(updated_job.get("updated_at") or created_at),
        })

    migrated["reflection_jobs"] = pending
    change_log = migrated.get("change_log")
    if isinstance(change_log, list):
        migrated["change_log"] = change_log[-CHANGE_LOG_LIMIT:]
    migrated["schema_version"] = 2
    return migrated


def _migrate_v2_state(state: dict[str, object]) -> dict[str, object]:
    """Group Self and InnerLife without changing any durable values."""

    if set(state) != _ROOT_KEYS or not isinstance(state.get("profiles"), dict):
        raise StateIntegrityError("State schema 2 cannot be migrated safely.")
    for key in ("reflection_jobs", "proactive_queue", "change_log"):
        if not isinstance(state[key], list):
            raise StateIntegrityError(f"State schema 2 has invalid {key}.")
    profiles: dict[str, object] = {}
    for profile_id, raw_profile in state["profiles"].items():
        if not isinstance(raw_profile, dict) or set(raw_profile) != _V2_PROFILE_KEYS:
            raise StateIntegrityError("State schema 2 has an invalid profile structure.")
        raw_items = raw_profile["self_items"]
        if not isinstance(raw_items, list):
            raise StateIntegrityError("State schema 2 has invalid Self items.")
        for key in ("self_revisions", "memories", "thoughts"):
            if not isinstance(raw_profile[key], list):
                raise StateIntegrityError(f"State schema 2 has invalid {key}.")
        grouped = {group: [] for group in SELF_GROUP_BY_KIND.values()}
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                raise StateIntegrityError("State schema 2 has an invalid Self item.")
            kind = str(raw_item.get("kind") or "")
            group = SELF_GROUP_BY_KIND.get(kind)
            if group is None:
                raise StateIntegrityError(
                    f"State schema 2 has unsupported Self item kind: {kind or '<empty>'}."
                )
            grouped[group].append(_reorder_record(raw_item, SelfItem, "Self item"))
        conversations = raw_profile["conversations"]
        if not isinstance(conversations, dict):
            raise StateIntegrityError("State schema 2 has invalid conversations.")
        ordered_conversations = {}
        for conversation_id, raw_conversation in conversations.items():
            if not isinstance(raw_conversation, dict) or set(raw_conversation) != _CONVERSATION_KEYS:
                raise StateIntegrityError("State schema 2 has an invalid conversation structure.")
            if not isinstance(raw_conversation["turns"], list):
                raise StateIntegrityError("State schema 2 has invalid turns.")
            ordered_conversations[conversation_id] = {
                "created_at": copy.deepcopy(raw_conversation["created_at"]),
                "updated_at": copy.deepcopy(raw_conversation["updated_at"]),
                "reflection_empty_streak": copy.deepcopy(raw_conversation["reflection_empty_streak"]),
                "turns": [
                    _reorder_record(turn, Turn, "turn")
                    for turn in raw_conversation["turns"]
                ],
            }
        profiles[profile_id] = {
            "created_at": copy.deepcopy(raw_profile["created_at"]),
            "updated_at": copy.deepcopy(raw_profile["updated_at"]),
            "self": {
                **grouped,
                "revisions": [
                    _reorder_record(revision, SelfRevision, "Self revision")
                    for revision in raw_profile["self_revisions"]
                ],
            },
            "mood": _reorder_record(raw_profile["mood"], Mood, "mood"),
            "relationship": _reorder_record(
                raw_profile["relationship"], Relationship, "relationship"
            ),
            "memories": [
                _reorder_record(memory, Memory, "memory")
                for memory in raw_profile["memories"]
            ],
            "inner_life": {
                "thoughts": [
                    _reorder_record(thought, Thought, "thought")
                    for thought in raw_profile["thoughts"]
                ]
            },
            "conversations": ordered_conversations,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": copy.deepcopy(state["revision"]),
        "updated_at": copy.deepcopy(state["updated_at"]),
        "profiles": profiles,
        "reflection_jobs": [
            _reorder_mapping(job, _REFLECTION_JOB_KEYS, _REFLECTION_JOB_ORDER, "reflection job")
            for job in state["reflection_jobs"]
        ],
        "proactive_queue": [
            _reorder_mapping(item, _PROACTIVE_KEYS, _PROACTIVE_ORDER, "proactive message")
            for item in state["proactive_queue"]
        ],
        "change_log": [
            _reorder_mapping(item, _CHANGE_LOG_KEYS, _CHANGE_LOG_ORDER, "change log entry")
            for item in state["change_log"][-CHANGE_LOG_LIMIT:]
        ],
    }


def _record(value) -> dict[str, object]:
    result = asdict(value)
    converted = {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in result.items()
    }
    field_order = _RECORD_FIELD_ORDER.get(type(value), ())
    return {key: converted[key] for key in field_order or converted}


def _reorder_record(value: object, model_type, label: str) -> dict[str, object]:
    row = _expect_model(value, model_type, label)
    return {key: copy.deepcopy(row[key]) for key in _RECORD_FIELD_ORDER[model_type]}


def _self_rows(profile: dict[str, object]):
    self_state = profile["self"]
    for group in SELF_GROUP_BY_KIND.values():
        yield from self_state[group]


def _tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise StateIntegrityError(f"{label} must be finite.")
    return result


def _identifier(value: object, label: str) -> str:
    result = str(value or "").strip()
    if not result or len(result) > 180 or any(char in result for char in "\r\n\x00"):
        raise StateIntegrityError(f"Invalid {label}.")
    return result


def _expect_keys(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise StateIntegrityError(f"Invalid {label} structure.")
    return value


def _expect_model(value: object, model_type, label: str) -> dict[str, object]:
    return _expect_keys(value, {field.name for field in fields(model_type)}, label)


def _reorder_mapping(
    value: object,
    keys: set[str],
    order: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    row = _expect_keys(value, keys, label)
    return {key: copy.deepcopy(row[key]) for key in order}


def _turn(value: object) -> Turn:
    row = _expect_model(value, Turn, "turn")
    return Turn(
        str(row["id"]),
        str(row["profile_id"]),
        str(row["conversation_id"]),
        str(row["role"]),
        str(row["content"]),
        float(row["created_at"]),
        str(row["request_id"]),
    )


def _memory(value: object) -> Memory:
    row = _expect_model(value, Memory, "memory")
    return Memory(
        str(row["id"]),
        str(row["profile_id"]),
        str(row["subject"]),
        str(row["kind"]),
        str(row["text"]),
        float(row["importance"]),
        float(row["confidence"]),
        float(row["created_at"]),
        float(row["updated_at"]),
        _tuple(row["source_turn_ids"]),
    )


def _self_item(value: object) -> SelfItem:
    row = _expect_model(value, SelfItem, "Self item")
    return SelfItem(
        str(row["id"]),
        str(row["profile_id"]),
        str(row["kind"]),
        str(row["topic"]),
        str(row["value"]),
        float(row["strength"]),
        float(row["confidence"]),
        str(row["reason"]),
        str(row["status"]),
        float(row["created_at"]),
        float(row["updated_at"]),
        _tuple(row["source_ids"]),
        int(row["revision_count"]),
    )


def _self_revision(value: object) -> SelfRevision:
    row = _expect_model(value, SelfRevision, "Self revision")
    return SelfRevision(
        str(row["id"]),
        str(row["self_item_id"]),
        str(row["profile_id"]),
        str(row["value"]),
        float(row["strength"]),
        float(row["confidence"]),
        str(row["reason"]),
        str(row["status"]),
        _tuple(row["source_ids"]),
        float(row["changed_at"]),
    )


def _mood(value: object) -> Mood:
    row = _expect_model(value, Mood, "mood")
    return Mood(
        str(row["profile_id"]),
        float(row["valence"]),
        float(row["energy"]),
        str(row["emotion"]),
        str(row["cause"]),
        float(row["updated_at"]),
    )


def _relationship(value: object) -> Relationship:
    row = _expect_model(value, Relationship, "relationship")
    return Relationship(
        str(row["profile_id"]),
        float(row["familiarity"]),
        float(row["trust"]),
        float(row["closeness"]),
        _tuple(row["interaction_notes"]),
        _tuple(row["unresolved_events"]),
        float(row["updated_at"]),
    )


def _thought(value: object) -> Thought:
    row = _expect_model(value, Thought, "thought")
    return Thought(
        str(row["id"]),
        str(row["profile_id"]),
        str(row["topic"]),
        str(row["text"]),
        float(row["importance"]),
        _tuple(row["source_ids"]),
        float(row["started_at"]),
        float(row["updated_at"]),
        str(row["status"]),
        bool(row["share_worthy"]),
    )


def _validate_state(value: object) -> None:
    try:
        state = _expect_keys(value, _ROOT_KEYS, "state document")
        if state["schema_version"] != SCHEMA_VERSION:
            raise StateIntegrityError(
                f"State schema {state['schema_version']} is not supported; expected {SCHEMA_VERSION}."
            )
        if not isinstance(state["revision"], int) or isinstance(state["revision"], bool) or state["revision"] < 0:
            raise StateIntegrityError("State revision must be a non-negative integer.")
        _finite(state["updated_at"], "state timestamp")
        if not isinstance(state["profiles"], dict):
            raise StateIntegrityError("State profiles must be an object.")
        for key in ("reflection_jobs", "proactive_queue", "change_log"):
            if not isinstance(state[key], list):
                raise StateIntegrityError(f"State {key} must be a list.")
        if len(state["change_log"]) > CHANGE_LOG_LIMIT:
            raise StateIntegrityError("State change log exceeds its retention limit.")

        turn_ids: set[str] = set()
        self_ids: set[str] = set()
        thought_ids: set[str] = set()
        thought_owners: dict[str, str] = {}
        memory_ids: set[str] = set()
        for profile_id, raw_profile in state["profiles"].items():
            profile_id = _identifier(profile_id, "profile ID")
            profile = _expect_keys(raw_profile, _PROFILE_KEYS, "profile")
            _finite(profile["created_at"], "profile creation timestamp")
            _finite(profile["updated_at"], "profile update timestamp")
            if not isinstance(profile["conversations"], dict):
                raise StateIntegrityError("Profile conversations must be an object.")
            for conversation_id, raw_conversation in profile["conversations"].items():
                conversation_id = _identifier(conversation_id, "conversation ID")
                conversation = _expect_keys(raw_conversation, _CONVERSATION_KEYS, "conversation")
                _finite(conversation["created_at"], "conversation creation timestamp")
                _finite(conversation["updated_at"], "conversation update timestamp")
                if (
                    not isinstance(conversation["reflection_empty_streak"], int)
                    or conversation["reflection_empty_streak"] < 0
                ):
                    raise StateIntegrityError("Reflection empty streak is invalid.")
                if not isinstance(conversation["turns"], list):
                    raise StateIntegrityError("Conversation turns must be a list.")
                for raw_turn in conversation["turns"]:
                    turn = _turn(raw_turn)
                    turn_id = _identifier(turn.id, "turn ID")
                    if turn_id in turn_ids:
                        raise StateIntegrityError("Turn IDs must be unique.")
                    turn_ids.add(turn_id)
                    if turn.profile_id != profile_id or turn.conversation_id != conversation_id:
                        raise StateIntegrityError("Turn ownership is inconsistent.")
                    if turn.role not in {"user", "assistant"} or not turn.content.strip():
                        raise StateIntegrityError("Turn role and content are required.")
                    _finite(turn.created_at, "turn timestamp")

            if not isinstance(profile["memories"], list):
                raise StateIntegrityError("Profile memories must be a list.")
            self_state = _expect_keys(profile["self"], _SELF_KEYS, "Self state")
            for group in SELF_GROUP_BY_KIND.values():
                if not isinstance(self_state[group], list):
                    raise StateIntegrityError(f"Self {group} must be a list.")
            if not isinstance(self_state["revisions"], list):
                raise StateIntegrityError("Self revisions must be a list.")
            inner_life = _expect_keys(profile["inner_life"], _INNER_LIFE_KEYS, "InnerLife state")
            if not isinstance(inner_life["thoughts"], list):
                raise StateIntegrityError("InnerLife thoughts must be a list.")
            for raw_memory in profile["memories"]:
                memory = _memory(raw_memory)
                memory_id = _identifier(memory.id, "memory ID")
                if memory_id in memory_ids:
                    raise StateIntegrityError("Memory IDs must be unique.")
                memory_ids.add(memory_id)
                if memory.profile_id != profile_id or memory.subject not in MEMORY_SUBJECTS or memory.kind not in MEMORY_KINDS:
                    raise StateIntegrityError("Invalid memory ownership or kind.")
                if not memory.text.strip() or len(memory.text) > 1000:
                    raise StateIntegrityError("Memory text is invalid.")
                if not 0 <= _finite(memory.importance, "memory importance") <= 1:
                    raise StateIntegrityError("Memory importance is out of range.")
                if not 0 <= _finite(memory.confidence, "memory confidence") <= 1:
                    raise StateIntegrityError("Memory confidence is out of range.")
                _finite(memory.created_at, "memory creation timestamp")
                _finite(memory.updated_at, "memory update timestamp")

            profile_self_ids: set[str] = set()
            for kind, group in SELF_GROUP_BY_KIND.items():
                for raw_item in self_state[group]:
                    item = _self_item(raw_item)
                    item_id = _identifier(item.id, "Self item ID")
                    if item_id in self_ids:
                        raise StateIntegrityError("Self item IDs must be unique.")
                    self_ids.add(item_id)
                    profile_self_ids.add(item_id)
                    if (
                        item.profile_id != profile_id
                        or item.kind != kind
                        or item.status not in SELF_STATUSES
                    ):
                        raise StateIntegrityError("Invalid Self item ownership, group, kind, or status.")
                    if not item.topic.strip() or not item.value.strip() or not item.reason.strip():
                        raise StateIntegrityError("Self items require topic, value, and reason.")
                    if not -1 <= _finite(item.strength, "Self strength") <= 1:
                        raise StateIntegrityError("Self strength is out of range.")
                    if not 0 <= _finite(item.confidence, "Self confidence") <= 1:
                        raise StateIntegrityError("Self confidence is out of range.")
                    if item.revision_count < 0:
                        raise StateIntegrityError("Self revision count is invalid.")
                    _finite(item.created_at, "Self creation timestamp")
                    _finite(item.updated_at, "Self update timestamp")

            for raw_revision in self_state["revisions"]:
                revision = _self_revision(raw_revision)
                _identifier(revision.id, "Self revision ID")
                if revision.profile_id != profile_id or revision.self_item_id not in profile_self_ids:
                    raise StateIntegrityError("Self revision ownership is inconsistent.")
                if revision.status not in SELF_STATUSES:
                    raise StateIntegrityError("Self revision status is invalid.")
                _finite(revision.changed_at, "Self revision timestamp")

            mood = _mood(profile["mood"])
            if mood.profile_id != profile_id:
                raise StateIntegrityError("Mood ownership is inconsistent.")
            if not -1 <= _finite(mood.valence, "mood valence") <= 1:
                raise StateIntegrityError("Mood valence is out of range.")
            if not -1 <= _finite(mood.energy, "mood energy") <= 1:
                raise StateIntegrityError("Mood energy is out of range.")
            _finite(mood.updated_at, "mood timestamp")

            relationship = _relationship(profile["relationship"])
            if relationship.profile_id != profile_id:
                raise StateIntegrityError("Relationship ownership is inconsistent.")
            for number in (relationship.familiarity, relationship.trust, relationship.closeness):
                if not 0 <= _finite(number, "relationship value") <= 1:
                    raise StateIntegrityError("Relationship value is out of range.")
            _finite(relationship.updated_at, "relationship timestamp")

            for raw_thought in inner_life["thoughts"]:
                thought = _thought(raw_thought)
                thought_id = _identifier(thought.id, "thought ID")
                if thought_id in thought_ids:
                    raise StateIntegrityError("Thought IDs must be unique.")
                thought_ids.add(thought_id)
                thought_owners[thought_id] = profile_id
                if thought.profile_id != profile_id or thought.status not in THOUGHT_STATUSES:
                    raise StateIntegrityError("Invalid thought ownership or status.")
                if not thought.topic.strip() or not thought.text.strip():
                    raise StateIntegrityError("Thought topic and text are required.")
                if not 0 <= _finite(thought.importance, "thought importance") <= 1:
                    raise StateIntegrityError("Thought importance is out of range.")
                _finite(thought.started_at, "thought start timestamp")
                _finite(thought.updated_at, "thought update timestamp")

        job_ids: set[str] = set()
        conversation_jobs: set[tuple[str, str]] = set()
        for raw_job in state["reflection_jobs"]:
            job = _expect_keys(raw_job, _REFLECTION_JOB_KEYS, "reflection job")
            job_id = _identifier(job["id"], "reflection job ID")
            if job_id in job_ids:
                raise StateIntegrityError("Reflection job IDs must be unique.")
            job_ids.add(job_id)
            profile_id = _identifier(job["profile_id"], "reflection profile ID")
            conversation_id = _identifier(job["conversation_id"], "reflection conversation ID")
            first_turn_id = _identifier(job["first_turn_id"], "reflection first turn ID")
            latest_turn_id = _identifier(job["latest_turn_id"], "reflection latest turn ID")
            if first_turn_id not in turn_ids or latest_turn_id not in turn_ids:
                raise StateIntegrityError("Reflection job turns do not exist.")
            owner = (profile_id, conversation_id)
            if owner in conversation_jobs:
                raise StateIntegrityError("A conversation may have only one pending reflection range.")
            conversation_jobs.add(owner)
            profile = state["profiles"].get(profile_id)
            conversation = profile.get("conversations", {}).get(conversation_id) if profile else None
            if conversation is None:
                raise StateIntegrityError("Reflection conversation does not exist.")
            conversation_turn_ids = [str(turn["id"]) for turn in conversation["turns"]]
            try:
                first_index = conversation_turn_ids.index(first_turn_id)
                latest_index = conversation_turn_ids.index(latest_turn_id)
            except ValueError as exc:
                raise StateIntegrityError("Reflection range ownership is inconsistent.") from exc
            if first_index > latest_index or int(job["turn_count"]) != latest_index - first_index + 1:
                raise StateIntegrityError("Reflection range is invalid.")
            if job["status"] not in {"pending", "running"}:
                raise StateIntegrityError("Reflection job status is invalid.")
            if not isinstance(job["attempts"], int) or job["attempts"] < 0:
                raise StateIntegrityError("Reflection attempts are invalid.")
            if not isinstance(job["turn_count"], int) or job["turn_count"] < 1:
                raise StateIntegrityError("Reflection turn count is invalid.")
            if not isinstance(job["ready"], bool):
                raise StateIntegrityError("Reflection readiness is invalid.")
            for key in ("available_at", "created_at", "updated_at"):
                _finite(job[key], f"reflection {key}")

        proactive_ids: set[str] = set()
        for raw_message in state["proactive_queue"]:
            message = _expect_keys(raw_message, _PROACTIVE_KEYS, "proactive message")
            message_id = _identifier(message["id"], "proactive message ID")
            if message_id in proactive_ids:
                raise StateIntegrityError("Proactive message IDs must be unique.")
            proactive_ids.add(message_id)
            profile_id = _identifier(message["profile_id"], "proactive profile ID")
            thought_id = _identifier(message["thought_id"], "proactive thought ID")
            if thought_id not in thought_ids or thought_owners[thought_id] != profile_id:
                raise StateIntegrityError("Proactive thought does not exist.")
            if message["status"] not in {"pending", "claimed", "delivered"}:
                raise StateIntegrityError("Proactive status is invalid.")
            if not str(message["text"]).strip():
                raise StateIntegrityError("Proactive text is required.")
            if not 0 <= _finite(message["importance"], "proactive importance") <= 1:
                raise StateIntegrityError("Proactive importance is out of range.")
            for key in ("created_at", "claimed_at", "delivered_at"):
                _finite(message[key], f"proactive {key}")

        change_ids: set[int] = set()
        for raw_change in state["change_log"]:
            change = _expect_keys(raw_change, _CHANGE_LOG_KEYS, "change log entry")
            if not isinstance(change["id"], int) or change["id"] <= 0 or change["id"] in change_ids:
                raise StateIntegrityError("Change log ID is invalid.")
            change_ids.add(change["id"])
            _identifier(change["profile_id"], "change profile ID")
            if not isinstance(change["applied"], list) or not isinstance(change["rejected"], list):
                raise StateIntegrityError("Change log values must be lists.")
            _finite(change["created_at"], "change timestamp")
    except StateIntegrityError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise StateIntegrityError(f"Canonical state validation failed: {exc}") from exc


class Store:
    """Atomic JSON ownership boundary for conversation and psychological state."""

    def __init__(self, path: str | Path = SETTINGS.state_path):
        self.path = str(path)
        self._path = None if self.path == ":memory:" else Path(self.path).expanduser()
        self._lock = threading.RLock()
        self._proactive_condition = threading.Condition(self._lock)
        if self._path is None:
            self._state = _empty_state()
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            self._state = self._load_state()
            self._recover_interrupted_reflections()
        else:
            self._state = _empty_state()
            _validate_state(self._state)
            self._write_state(self._state)

    def _recover_interrupted_reflections(self) -> None:
        current = time.time()
        with self._transaction() as state:
            for job in state["reflection_jobs"]:
                if job["status"] != "running":
                    continue
                job["status"] = "pending"
                job["available_at"] = current + min(
                    _REFLECTION_RETRY_MAX_SECONDS,
                    _REFLECTION_RETRY_BASE_SECONDS * (2 ** min(max(0, int(job["attempts"]) - 1), 4)),
                )
                job["error"] = "Reflection was interrupted by process shutdown."
                job["updated_at"] = current
                state["updated_at"] = current

    def _load_state(self) -> dict[str, object]:
        assert self._path is not None
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except json.JSONDecodeError as exc:
            raise StateIntegrityError(f"State file is not valid JSON: {self._path}") from exc
        except OSError as exc:
            raise StateIntegrityError(f"State file could not be read: {self._path}") from exc
        migrated = False
        if isinstance(state, dict) and state.get("schema_version") == 1:
            state = _migrate_v1_state(state)
            migrated = True
        if isinstance(state, dict) and state.get("schema_version") == 2:
            state = _migrate_v2_state(state)
            migrated = True
        if migrated:
            _validate_state(state)
            self._write_state(state)
            return state
        _validate_state(state)
        return state

    def _write_state(self, state: dict[str, object]) -> dict[str, float]:
        if self._path is None:
            return {"serialize": 0.0, "write": 0.0, "fsync": 0.0, "replace": 0.0}
        temporary = ""
        try:
            serialize_started = time.perf_counter()
            serialized = json.dumps(
                state,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            ) + "\n"
            serialized_at = time.perf_counter()
            descriptor, temporary = tempfile.mkstemp(
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                written_at = time.perf_counter()
                os.fsync(handle.fileno())
                synced_at = time.perf_counter()
            os.replace(temporary, self._path)
            replaced_at = time.perf_counter()
            temporary = ""
            return {
                "serialize": serialized_at - serialize_started,
                "write": written_at - serialized_at,
                "fsync": synced_at - written_at,
                "replace": replaced_at - synced_at,
            }
        except (OSError, TypeError, ValueError) as exc:
            raise StateIntegrityError(f"Atomic JSON state write failed: {exc}") from exc
        finally:
            if temporary:
                try:
                    Path(temporary).unlink()
                except OSError:
                    pass

    def _publish(self, candidate: dict[str, object], expected_revision: int) -> dict[str, float]:
        if self._state["revision"] != expected_revision:
            raise StateIntegrityError("State changed while a candidate update was being prepared.")
        validation_started = time.perf_counter()
        _validate_state(candidate)
        validated_at = time.perf_counter()
        try:
            timings = self._write_state(candidate)
        except OSError as exc:
            raise StateIntegrityError(f"Atomic JSON state write failed: {exc}") from exc
        self._state = candidate
        return {"validate": validated_at - validation_started, **timings}

    @contextmanager
    def _transaction(self):
        with self._lock:
            started_at = time.perf_counter()
            expected_revision = int(self._state["revision"])
            candidate = copy.deepcopy(self._state)
            copied_at = time.perf_counter()
            yield candidate
            if candidate != self._state:
                timings = self._publish(candidate, expected_revision)
                log_timing(
                    "store",
                    copy=copied_at - started_at,
                    **timings,
                    total=time.perf_counter() - started_at,
                )

    @staticmethod
    def _new_profile(profile_id: str, now: float) -> dict[str, object]:
        self_state = {group: [] for group in SELF_GROUP_BY_KIND.values()}
        self_state["interests"] = [
            _record(SelfItem(
                id=(
                    "self_seed_"
                    + uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"akane:self-seed:{profile_id}:{interest.casefold()}",
                    ).hex
                ),
                profile_id=profile_id,
                kind="interest",
                topic=interest,
                value=f"Interested in {interest}.",
                strength=0.42,
                confidence=0.68,
                reason="Starting interest.",
                status="active",
                created_at=now,
                updated_at=now,
            ))
            for interest in load_character_profile().seed_interests
        ]
        return {
            "created_at": now,
            "updated_at": now,
            "self": {**self_state, "revisions": []},
            "mood": _record(Mood(profile_id, updated_at=now)),
            "relationship": _record(Relationship(profile_id, updated_at=now)),
            "memories": [],
            "inner_life": {"thoughts": []},
            "conversations": {},
        }

    @staticmethod
    def _ensure_profile(state: dict[str, object], profile_id: str, now: float) -> bool:
        profiles = state["profiles"]
        if profile_id in profiles:
            return False
        profiles[profile_id] = Store._new_profile(profile_id, now)
        state["updated_at"] = now
        return True

    @staticmethod
    def _turn_lookup(state: dict[str, object]) -> dict[str, tuple[dict[str, object], str]]:
        turns: dict[str, tuple[dict[str, object], str]] = {}
        for profile in state["profiles"].values():
            for conversation_id, conversation in profile["conversations"].items():
                for turn in conversation["turns"]:
                    turns[str(turn["id"])] = (turn, conversation_id)
        return turns

    def ensure_profile(self, profile_id: str) -> None:
        profile = _identifier(profile_id, "profile ID")
        now = time.time()
        with self._transaction() as state:
            self._ensure_profile(state, profile, now)

    def profile_exists(self, profile_id: str) -> bool:
        with self._lock:
            return profile_id in self._state["profiles"]

    def profile_ids(self, prefix: str = "") -> tuple[str, ...]:
        with self._lock:
            values = sorted(str(item) for item in self._state["profiles"])
        return tuple(item for item in values if not prefix or item.startswith(prefix))

    def commit(self, proposal: StateChangeProposal, *, now: float | None = None) -> CommitResult:
        timestamp = time.time() if now is None else _finite(now, "commit timestamp")
        profile_id = _identifier(proposal.profile_id, "profile ID")
        applied: list[str] = []
        rejected = list(proposal.rejected)
        try:
            with self._transaction() as state:
                guard = None
                if proposal.reflection_job_id:
                    guard = next(
                        (
                            job for job in state["reflection_jobs"]
                            if job["id"] == proposal.reflection_job_id
                            and job["profile_id"] == profile_id
                            and job["status"] == "running"
                        ),
                        None,
                    )
                    if guard is None:
                        raise StateIntegrityError("The originating reflection job is no longer active.")
                self._ensure_profile(state, profile_id, timestamp)
                for turn in proposal.turns:
                    self._insert_turn(state, profile_id, turn)
                    applied.append(f"turn:{turn.id}")
                for change in proposal.memories:
                    if self._apply_memory(state, profile_id, change.action, change.target_id, change.memory):
                        applied.append(f"memory:{change.target_id or change.memory.id}")
                for change in proposal.self_items:
                    if self._apply_self(state, profile_id, change.action, change.target_id, change.item):
                        applied.append(f"self:{change.target_id or change.item.id}")
                    else:
                        item_id = change.target_id or (change.item.id if change.item is not None else "unknown")
                        rejected.append(f"self:not-applied:{change.action}:{item_id}")
                if proposal.mood is not None:
                    self._apply_mood(state, profile_id, proposal.mood, timestamp)
                    applied.append("mood")
                if proposal.relationship is not None:
                    self._apply_relationship(state, profile_id, proposal.relationship, timestamp)
                    applied.append("relationship")
                for change in proposal.thoughts:
                    if self._apply_thought(state, profile_id, change.action, change.target_id, change.thought):
                        applied.append(f"thought:{change.target_id or change.thought.id}")
                for message in proposal.proactive_messages:
                    if self._insert_proactive(state, profile_id, message):
                        applied.append(f"proactive:{message.id}")
                if proposal.reflection_turn_ids is not None:
                    first_turn_id, latest_turn_id = proposal.reflection_turn_ids
                    turns = self._turn_lookup(state)
                    first = turns.get(first_turn_id)
                    latest = turns.get(latest_turn_id)
                    if first is None or latest is None or first[1] != latest[1]:
                        raise StateIntegrityError("Reflection ranges require committed conversation turns.")
                    conversation_id = first[1]
                    existing = next((
                        job for job in state["reflection_jobs"]
                        if job["profile_id"] == profile_id
                        and job["conversation_id"] == conversation_id
                    ), None)
                    if existing is None:
                        state["reflection_jobs"].append({
                            "id": uuid.uuid4().hex,
                            "profile_id": profile_id,
                            "conversation_id": conversation_id,
                            "status": "pending",
                            "ready": bool(proposal.reflection_ready),
                            "turn_count": 2,
                            "first_turn_id": first_turn_id,
                            "latest_turn_id": latest_turn_id,
                            "attempts": 0,
                            "available_at": timestamp + SETTINGS.background_idle_grace_seconds,
                            "created_at": timestamp,
                            "updated_at": timestamp,
                            "error": "",
                        })
                    else:
                        existing["latest_turn_id"] = latest_turn_id
                        conversation = state["profiles"][profile_id]["conversations"][conversation_id]
                        turn_ids = [str(turn["id"]) for turn in conversation["turns"]]
                        existing["turn_count"] = (
                            turn_ids.index(latest_turn_id)
                            - turn_ids.index(str(existing["first_turn_id"]))
                            + 1
                        )
                        if (
                            proposal.reflection_ready
                            and not existing["ready"]
                            and str(existing["error"]).startswith("reflection:parse-")
                        ):
                            existing["attempts"] = 0
                            existing["error"] = ""
                        existing["ready"] = bool(existing["ready"] or proposal.reflection_ready)
                        existing["available_at"] = max(
                            float(existing["available_at"]),
                            timestamp + SETTINGS.background_idle_grace_seconds,
                        )
                        existing["updated_at"] = timestamp
                    applied.append("reflection:dirty")
                revision = self._record_change(
                    state, profile_id, proposal.origin, applied, rejected, timestamp
                )
                if proposal.reflection_job_id:
                    assert guard is not None
                    claimed_latest = proposal.reflected_through_turn_id
                    if not claimed_latest or claimed_latest != guard["latest_turn_id"]:
                        if claimed_latest not in self._turn_lookup(state):
                            raise StateIntegrityError("The reflected turn range is invalid.")
                        conversation = state["profiles"][profile_id]["conversations"][guard["conversation_id"]]
                        turn_ids = [str(turn["id"]) for turn in conversation["turns"]]
                        claimed_index = turn_ids.index(claimed_latest)
                        latest_index = turn_ids.index(str(guard["latest_turn_id"]))
                        if claimed_index >= latest_index:
                            raise StateIntegrityError("The reflected turn range is inconsistent.")
                        guard["first_turn_id"] = turn_ids[claimed_index + 1]
                        guard["turn_count"] = max(
                            1, int(guard["turn_count"]) - int(proposal.reflected_turn_count),
                        )
                        guard["status"] = "pending"
                        guard["attempts"] = 0
                        guard["error"] = ""
                        guard["updated_at"] = timestamp
                    else:
                        state["reflection_jobs"].remove(guard)
                    conversation = state["profiles"][profile_id]["conversations"][guard["conversation_id"]]
                    if any(item.startswith(("memory:", "self:", "mood", "relationship")) for item in applied):
                        conversation["reflection_empty_streak"] = 0
                    else:
                        conversation["reflection_empty_streak"] = min(
                            4, int(conversation["reflection_empty_streak"]) + 1,
                        )
        except StateIntegrityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise StateIntegrityError(f"State transaction failed: {exc}") from exc
        if proposal.proactive_messages:
            with self._proactive_condition:
                self._proactive_condition.notify_all()
        return CommitResult(revision, tuple(applied), tuple(rejected))

    @staticmethod
    def _record_change(
        state: dict[str, object],
        profile_id: str,
        origin: str,
        applied: list[str],
        rejected: list[str],
        now: float,
    ) -> int:
        profile = state["profiles"][profile_id]
        profile["updated_at"] = now
        revision = int(state["revision"]) + 1
        state["revision"] = revision
        state["updated_at"] = now
        log = state["change_log"]
        log.append({
            "id": max((int(item["id"]) for item in log), default=0) + 1,
            "profile_id": profile_id,
            "origin": str(origin),
            "applied": list(applied),
            "rejected": list(rejected),
            "created_at": now,
        })
        del log[:-CHANGE_LOG_LIMIT]
        return revision

    @staticmethod
    def _insert_turn(state: dict[str, object], profile_id: str, turn: Turn) -> None:
        if turn.profile_id != profile_id:
            raise StateIntegrityError("Turn belongs to a different profile.")
        if turn.role not in {"user", "assistant"} or not str(turn.content).strip():
            raise StateIntegrityError("Turn role and content are required.")
        _finite(turn.created_at, "turn timestamp")
        turn_id = _identifier(turn.id, "turn ID")
        conversation_id = _identifier(turn.conversation_id, "conversation ID")
        if turn_id in Store._turn_lookup(state):
            raise StateIntegrityError("Turn ID already exists.")
        profile = state["profiles"][profile_id]
        conversations = profile["conversations"]
        conversation = conversations.get(conversation_id)
        if conversation is None:
            conversation = {
                "created_at": turn.created_at,
                "updated_at": turn.created_at,
                "reflection_empty_streak": 0,
                "turns": [],
            }
            conversations[conversation_id] = conversation
        conversation["turns"].append(_record(turn))
        conversation["updated_at"] = turn.created_at

    @staticmethod
    def _apply_memory(state, profile_id, action, target_id, memory) -> bool:
        values = state["profiles"][profile_id]["memories"]
        if action == "delete":
            original = len(values)
            values[:] = [item for item in values if item["id"] != target_id]
            return len(values) != original
        if action != "add" or memory is None or memory.profile_id != profile_id:
            raise StateIntegrityError("Invalid memory change.")
        if memory.subject not in MEMORY_SUBJECTS or memory.kind not in MEMORY_KINDS:
            raise StateIntegrityError("Invalid memory ownership or kind.")
        if not memory.text.strip() or len(memory.text) > 1000:
            raise StateIntegrityError("Memory text is invalid.")
        key = text_key(memory.text)
        if any(
            item["id"] == memory.id
            for profile in state["profiles"].values()
            for item in profile["memories"]
        ) or any(
            item["subject"] == memory.subject
            and item["kind"] == memory.kind
            and text_key(item["text"]) == key
            for item in values
        ):
            return False
        values.append(_record(Memory(
            memory.id,
            profile_id,
            memory.subject,
            memory.kind,
            memory.text.strip(),
            clamp(memory.importance, 0, 1),
            clamp(memory.confidence, 0, 1),
            memory.created_at,
            memory.updated_at,
            memory.source_turn_ids,
        )))
        return True

    @staticmethod
    def _apply_self(state, profile_id, action, target_id, item) -> bool:
        profile = state["profiles"][profile_id]
        current = next((value for value in _self_rows(profile) if value["id"] == target_id), None)
        if action == "retire":
            if current is None:
                return False
            Store._record_self_revision(profile, current)
            if item is None:
                current["status"] = "retired"
                current["updated_at"] = time.time()
                current["revision_count"] = int(current["revision_count"]) + 1
                return True
            if item.profile_id != profile_id or item.kind != current["kind"]:
                raise StateIntegrityError("Invalid retired Self item ownership or kind.")
            current.update(_record(SelfItem(
                str(current["id"]),
                profile_id,
                str(current["kind"]),
                str(current["topic"]),
                item.value.strip() or str(current["value"]),
                clamp(item.strength, -1, 1),
                clamp(item.confidence, 0, 1),
                item.reason.strip() or str(current["reason"]),
                "retired",
                float(current["created_at"]),
                item.updated_at,
                item.source_ids,
                int(current["revision_count"]) + 1,
            )))
            return True
        if action not in {"form", "reinforce", "weaken", "revise", "complete", "abandon"} or item is None:
            raise StateIntegrityError("Invalid Self change.")
        if item.profile_id != profile_id or item.kind not in SELF_KINDS or item.status not in SELF_STATUSES:
            raise StateIntegrityError("Invalid Self item ownership, kind, or status.")
        if not item.topic.strip() or not item.value.strip() or not item.reason.strip():
            raise StateIntegrityError("Self items require topic, value, and reason.")
        values = profile["self"][SELF_GROUP_BY_KIND[item.kind]]
        if action == "form":
            topic = text_key(item.topic)
            if any(
                existing["id"] == item.id
                for owner in state["profiles"].values()
                for existing in _self_rows(owner)
            ) or any(existing["kind"] == item.kind and text_key(existing["topic"]) == topic for existing in values):
                return False
            values.append(_record(SelfItem(
                item.id,
                profile_id,
                item.kind,
                item.topic.strip(),
                item.value.strip(),
                clamp(item.strength, -1, 1),
                clamp(item.confidence, 0, 1),
                item.reason.strip(),
                item.status,
                item.created_at,
                item.updated_at,
                item.source_ids,
                max(0, item.revision_count),
            )))
            return True
        target = target_id or item.id
        current = next((value for value in _self_rows(profile) if value["id"] == target), None)
        if current is None:
            return False
        if current["kind"] != item.kind:
            raise StateIntegrityError("A Self lifecycle update cannot change item kind.")
        Store._record_self_revision(profile, current)
        status = "completed" if action == "complete" else "abandoned" if action == "abandon" else item.status
        current.update(_record(SelfItem(
            target,
            profile_id,
            item.kind,
            item.topic.strip(),
            item.value.strip(),
            clamp(item.strength, -1, 1),
            clamp(item.confidence, 0, 1),
            item.reason.strip(),
            status,
            float(current["created_at"]),
            item.updated_at,
            item.source_ids,
            int(current["revision_count"]) + 1,
        )))
        return True

    @staticmethod
    def _record_self_revision(profile: dict[str, object], current: dict[str, object]) -> None:
        profile["self"]["revisions"].append(_record(SelfRevision(
            f"revision_{uuid.uuid4().hex}",
            str(current["id"]),
            str(current["profile_id"]),
            str(current["value"]),
            float(current["strength"]),
            float(current["confidence"]),
            str(current["reason"]),
            str(current["status"]),
            _tuple(current["source_ids"]),
            time.time(),
        )))

    @staticmethod
    def _apply_mood(state, profile_id, change, now) -> None:
        profile = state["profiles"][profile_id]
        current = _mood(profile["mood"])
        emotion = str(change.emotion or current.emotion).strip()[:80] or "calm"
        cause = str(change.cause or current.cause).strip()[:500]
        profile["mood"] = _record(Mood(
            profile_id,
            clamp(current.valence + change.valence_delta, -1, 1),
            clamp(current.energy + change.energy_delta, -1, 1),
            emotion,
            cause,
            now,
        ))

    @staticmethod
    def _apply_relationship(state, profile_id, change, now) -> None:
        profile = state["profiles"][profile_id]
        current = _relationship(profile["relationship"])
        notes = list(current.interaction_notes)
        unresolved = list(current.unresolved_events)
        for note in change.add_notes:
            value = str(note).strip()[:300]
            if value and value not in notes:
                notes.append(value)
        resolved_keys = {text_key(item) for item in change.resolve_notes}
        unresolved = [item for item in unresolved if text_key(item) not in resolved_keys]
        for item in change.add_unresolved:
            value = str(item).strip()[:300]
            if value and value not in unresolved:
                unresolved.append(value)
        profile["relationship"] = _record(Relationship(
            profile_id,
            clamp(current.familiarity + change.familiarity_delta, 0, 1),
            clamp(current.trust + change.trust_delta, 0, 1),
            clamp(current.closeness + change.closeness_delta, 0, 1),
            tuple(notes[-16:]),
            tuple(unresolved[-8:]),
            now,
        ))

    @staticmethod
    def _apply_thought(state, profile_id, action, target_id, thought) -> bool:
        values = state["profiles"][profile_id]["inner_life"]["thoughts"]
        if action in {"resolve", "expire"}:
            current = next((value for value in values if value["id"] == target_id), None)
            if current is None:
                return False
            current["status"] = "resolved" if action == "resolve" else "expired"
            current["updated_at"] = time.time()
            return True
        if action not in {"add", "continue"} or thought is None or thought.profile_id != profile_id:
            raise StateIntegrityError("Invalid InnerLife thought change.")
        if thought.status not in THOUGHT_STATUSES or not thought.topic.strip() or not thought.text.strip():
            raise StateIntegrityError("Invalid InnerLife thought.")
        if action == "add":
            if any(
                existing["id"] == thought.id
                for profile in state["profiles"].values()
                for existing in profile["inner_life"]["thoughts"]
            ):
                raise StateIntegrityError("Thought ID already exists.")
            values.append(_record(Thought(
                thought.id,
                profile_id,
                thought.topic.strip(),
                thought.text.strip(),
                clamp(thought.importance, 0, 1),
                thought.source_ids,
                thought.started_at,
                thought.updated_at,
                thought.status,
                thought.share_worthy,
            )))
            return True
        target = target_id or thought.id
        current = next((value for value in values if value["id"] == target), None)
        if current is None:
            return False
        current.update(_record(Thought(
            target,
            profile_id,
            thought.topic.strip(),
            thought.text.strip(),
            clamp(thought.importance, 0, 1),
            thought.source_ids,
            float(current["started_at"]),
            thought.updated_at,
            thought.status,
            thought.share_worthy,
        )))
        return True

    @staticmethod
    def _insert_proactive(state, profile_id, message: ProactiveMessage) -> bool:
        if message.profile_id != profile_id or message.status != "pending":
            raise StateIntegrityError("Invalid proactive message.")
        profile = state["profiles"][profile_id]
        thought = next((
            item for item in profile["inner_life"]["thoughts"]
            if item["id"] == message.thought_id
        ), None)
        if (
            thought is None
            or thought["status"] != "active"
            or not thought["share_worthy"]
            or float(thought["importance"]) < 0.7
        ):
            raise StateIntegrityError("Proactive messages require an important share-worthy active thought.")
        for existing in state["proactive_queue"]:
            if existing["id"] == message.id:
                raise StateIntegrityError("Proactive message ID already exists.")
            if (
                existing["profile_id"] == profile_id
                and existing["thought_id"] == message.thought_id
                and (
                    existing["status"] in {"pending", "claimed"}
                    or float(existing["delivered_at"]) > message.created_at - 21600
                )
            ):
                return False
        record = _record(message)
        record.update({
            "claim_token": "",
            "claimed_at": 0.0,
            "adapter": "",
            "conversation_id": "",
            "delivered_at": 0.0,
            "message_id": "",
        })
        state["proactive_queue"].append(record)
        return True

    def snapshot(
        self,
        profile_id: str,
        conversation_id: str,
        *,
        query: str = "",
        now: float | None = None,
    ) -> StateSnapshot:
        current = time.time() if now is None else float(now)
        with self._lock:
            stored_profile = self._state["profiles"].get(profile_id)
            revision = int(self._state["revision"])
            profile = stored_profile or self._new_profile(profile_id, current)
            conversation = profile["conversations"].get(conversation_id, {"turns": []})
            recent_pairs = heapq.nlargest(
                SETTINGS.recent_turn_limit,
                enumerate(conversation["turns"]),
                key=lambda pair: (float(pair[1]["created_at"]), pair[0]),
            )
            recent_pairs.sort(key=lambda pair: (float(pair[1]["created_at"]), pair[0]))
            recent_rows = [item for _, item in recent_pairs]
            memory_rows = heapq.nlargest(
                128, profile["memories"], key=lambda item: float(item["updated_at"]),
            )
            self_rows = heapq.nlargest(
                128,
                (item for item in _self_rows(profile) if item["status"] != "retired"),
                key=lambda item: float(item["updated_at"]),
            )
            revision_rows = heapq.nlargest(
                24, profile["self"]["revisions"], key=lambda item: float(item["changed_at"]),
            )
            thought_rows = heapq.nlargest(
                32,
                (
                    item for item in profile["inner_life"]["thoughts"]
                    if item["status"] == "active"
                ),
                key=lambda item: (float(item["importance"]), float(item["updated_at"])),
            )
            recent_rows, memory_rows, self_rows, revision_rows, thought_rows, mood_row, relationship_row = copy.deepcopy((
                recent_rows,
                memory_rows,
                self_rows,
                revision_rows,
                thought_rows,
                profile["mood"],
                profile["relationship"],
            ))
        memories = [_memory(row) for row in memory_rows]
        self_items = [_self_item(row) for row in self_rows]
        if query:
            query_terms = lexical_terms(query)
            normalized_query = text_key(query)
            memory_relevance = {item.id: relevance(query, item.text) for item in memories}
            self_relevance = {
                item.id: relevance(query, f"{item.topic} {item.value} {item.reason}")
                for item in self_items
            }
            memory_scores = {
                item.id: memory_relevance[item.id] * 0.65 + item.importance * 0.25 + item.confidence * 0.1
                for item in memories
            }
            self_scores = {
                item.id: self_relevance[item.id] * 0.7 + abs(item.strength) * 0.15 + item.confidence * 0.15
                for item in self_items
            }
            broad_memory = normalized_query in {
                "what do you remember", "what do you remember about me", "what is our history",
            }
            broad_self = normalized_query in {
                "who are you", "what do you like", "what do you like to do", "what do you want",
                "what are your interests", "what are your hobbies", "what hobbies do you have",
                "what are your opinions", "what are your goals",
                "what have you changed your mind about", "what are you uncertain about",
            }
            memories = [item for item in memories if broad_memory or memory_relevance[item.id] >= 0.12]
            self_items = [item for item in self_items if broad_self or self_relevance[item.id] >= 0.12]
            memories.sort(key=lambda item: memory_scores[item.id], reverse=True)
            self_items.sort(key=lambda item: self_scores[item.id], reverse=True)
            relevant_thoughts = [
                row for row in thought_rows
                if relevance(query, f"{row['topic']} {row['text']}") >= 0.12
            ]
            thought_rows = relevant_thoughts or thought_rows[:1]
        mood = _mood(mood_row)
        elapsed_hours = max(0.0, current - mood.updated_at) / 3600.0
        decay = 0.5 ** (elapsed_hours / 6.0) if elapsed_hours else 1.0
        effective_mood = Mood(
            mood.profile_id,
            mood.valence * decay,
            mood.energy * decay,
            mood.emotion if decay >= 0.2 else "calm",
            mood.cause if decay >= 0.2 else "",
            current,
        )
        selected_self_ids = {item.id for item in self_items[: SETTINGS.self_result_limit]}
        return StateSnapshot(
            profile_id,
            conversation_id,
            tuple(_turn(row) for row in recent_rows),
            tuple(memories[: SETTINGS.memory_result_limit]),
            tuple(self_items[: SETTINGS.self_result_limit]),
            tuple(
                _self_revision(row)
                for row in revision_rows
                if row["self_item_id"] in selected_self_ids
            ),
            effective_mood,
            _relationship(relationship_row),
            tuple(_thought(row) for row in thought_rows[:3]),
            SCHEMA_VERSION,
            revision,
        )

    def messages(self, conversation_id: str, profile_id: str) -> list[dict[str, str]]:
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            conversation = profile["conversations"].get(conversation_id) if profile else None
            rows = copy.deepcopy(conversation["turns"]) if conversation else []
        rows = [
            item for _, item in sorted(
                enumerate(rows),
                key=lambda pair: (float(pair[1]["created_at"]), pair[0]),
            )
        ]
        return [{"role": str(row["role"]), "content": str(row["content"])} for row in rows]

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

    def reply_for_request(self, conversation_id: str, profile_id: str, request_id: str) -> str | None:
        if not request_id:
            return None
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            conversation = profile["conversations"].get(conversation_id) if profile else None
            rows = list(conversation["turns"]) if conversation else []
        matches = [
            (index, row) for index, row in enumerate(rows)
            if row["request_id"] == request_id and row["role"] == "assistant"
        ]
        if not matches:
            return None
        _, row = max(matches, key=lambda pair: (float(pair[1]["created_at"]), pair[0]))
        return str(row["content"])

    def public_conversation(self, conversation_id: str, profile_id: str) -> dict[str, object]:
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            conversation = profile["conversations"].get(conversation_id) if profile else None
            conversation = copy.deepcopy(conversation)
        return {
            "id": conversation_id,
            "profile_id": profile_id,
            "created_at": float(conversation["created_at"]) if conversation else 0.0,
            "updated_at": float(conversation["updated_at"]) if conversation else 0.0,
        }

    def public_memory(self, profile_id: str) -> dict[str, object]:
        snapshot = self.snapshot(profile_id, "", query="")
        return {
            "user": {},
            "preferences": [
                {"content": item.text, "kind": item.kind}
                for item in snapshot.memories if item.subject == "user"
            ],
            "facts": [
                {"content": item.text, "subject": item.subject, "kind": item.kind}
                for item in snapshot.memories
            ],
            "self": [
                {
                    "kind": item.kind,
                    "topic": item.topic,
                    "value": item.value,
                    "strength": item.strength,
                    "confidence": item.confidence,
                    "reason": item.reason,
                }
                for item in snapshot.self_items
            ],
            "mood": {
                "emotion": snapshot.mood.emotion,
                "valence": snapshot.mood.valence,
                "energy": snapshot.mood.energy,
                "cause": snapshot.mood.cause,
            },
            "inner_life": [
                {"topic": thought.topic, "thought": thought.text, "importance": thought.importance}
                for thought in snapshot.thoughts
            ],
        }

    def clear_conversation(self, conversation_id: str, profile_id: str) -> None:
        with self._transaction() as state:
            profile = state["profiles"].get(profile_id)
            if profile is None or conversation_id not in profile["conversations"]:
                return
            conversation = profile["conversations"].pop(conversation_id)
            state["reflection_jobs"][:] = [
                job for job in state["reflection_jobs"]
                if not (
                    job["profile_id"] == profile_id
                    and job["conversation_id"] == conversation_id
                )
            ]
            state["updated_at"] = time.time()

    def clear_profile(self, profile_id: str) -> None:
        with self._transaction() as state:
            if profile_id not in state["profiles"]:
                return
            state["profiles"].pop(profile_id)
            state["reflection_jobs"][:] = [
                job for job in state["reflection_jobs"] if job["profile_id"] != profile_id
            ]
            state["proactive_queue"][:] = [
                item for item in state["proactive_queue"] if item["profile_id"] != profile_id
            ]
            state["updated_at"] = time.time()

    def claim_reflection_job(self, *, now: float | None = None) -> dict[str, object] | None:
        current = time.time() if now is None else float(now)
        result = None

        def due(job, state) -> bool:
            content_ready = bool(job["ready"])
            if job["status"] != "pending" or not content_ready or float(job["available_at"]) > current:
                return False
            conversation = state["profiles"][job["profile_id"]]["conversations"][job["conversation_id"]]
            required_turns = 2 * (1 + min(3, int(conversation["reflection_empty_streak"])))
            return int(job["turn_count"]) >= required_turns

        with self._lock:
            if not any(due(job, self._state) for job in self._state["reflection_jobs"]):
                return None
            with self._transaction() as state:
                candidates = [job for job in state["reflection_jobs"] if due(job, state)]
                job = min(candidates, key=lambda item: float(item["created_at"]))
                conversation = state["profiles"][job["profile_id"]]["conversations"][job["conversation_id"]]
                turns = conversation["turns"]
                turn_ids = [str(turn["id"]) for turn in turns]
                first_index = turn_ids.index(str(job["first_turn_id"]))
                latest_index = turn_ids.index(str(job["latest_turn_id"]))
                limit = SETTINGS.reflection_turn_limit
                if limit % 2:
                    limit -= 1
                available = turns[first_index:min(latest_index + 1, first_index + max(2, limit))]
                selected = []
                last_complete = 0
                rendered_chars = 0
                for turn in available:
                    label = "USER" if turn["role"] == "user" else "AKANE"
                    line_chars = len(label) + 2 + len(str(turn["content"])) + 1
                    if (
                        selected
                        and last_complete
                        and rendered_chars + line_chars > SETTINGS.reflection_input_chars
                    ):
                        break
                    selected.append(turn)
                    rendered_chars += line_chars
                    if turn["role"] == "assistant":
                        last_complete = len(selected)
                selected = selected[:last_complete]
                if not selected:
                    return None
                result = copy.deepcopy(job)
                result.update({
                    "user_text": "\n".join(
                        str(turn["content"]) for turn in selected if turn["role"] == "user"
                    ),
                    "assistant_text": "\n".join(
                        str(turn["content"]) for turn in selected if turn["role"] == "assistant"
                    ),
                    "conversation_text": "\n".join(
                        f"{'USER' if turn['role'] == 'user' else 'AKANE'}: {turn['content']}"
                        for turn in selected
                    ),
                    "selected_turns": tuple(_turn(turn) for turn in selected),
                    "first_turn_id": str(selected[0]["id"]),
                    "latest_turn_id": str(selected[-1]["id"]),
                    "turn_count": len(selected),
                    "claimed_monotonic_at": time.perf_counter(),
                })
                job["status"] = "running"
                job["attempts"] = int(job["attempts"]) + 1
                job["updated_at"] = current
                state["updated_at"] = current
        return result

    def defer_reflection_job(self, job_id: str, *, now: float | None = None) -> None:
        current = time.time() if now is None else float(now)
        with self._transaction() as state:
            job = next((item for item in state["reflection_jobs"] if item["id"] == job_id), None)
            if job is None or job["status"] != "running":
                return
            job["status"] = "pending"
            job["attempts"] = max(0, int(job["attempts"]) - 1)
            job["available_at"] = current + SETTINGS.background_idle_grace_seconds
            job["error"] = ""
            job["updated_at"] = current
            state["updated_at"] = current

    def finish_reflection_job(self, job_id: str, *, error: str = "", now: float | None = None) -> None:
        current = time.time() if now is None else float(now)
        with self._transaction() as state:
            job = next((item for item in state["reflection_jobs"] if item["id"] == job_id), None)
            if job is None:
                return
            if error:
                attempts = int(job["attempts"])
                message = str(error)[:500]
                job["status"] = "pending"
                if message == "reflection:parse-truncated-output" or (
                    message.startswith("reflection:parse-") and attempts >= 2
                ):
                    # Preserve the range, but do not retry the same malformed
                    # output forever. New reflectable dialogue reactivates it.
                    job["ready"] = False
                    job["available_at"] = current + SETTINGS.background_idle_grace_seconds
                else:
                    job["available_at"] = current + min(
                        _REFLECTION_RETRY_MAX_SECONDS,
                        _REFLECTION_RETRY_BASE_SECONDS * (2 ** min(max(0, attempts - 1), 4)),
                    )
                job["error"] = message
            else:
                state["reflection_jobs"].remove(job)
            job["updated_at"] = current
            state["updated_at"] = current

    def claim_proactive(
        self,
        profile_id: str,
        *,
        adapter: str,
        conversation_id: str,
        now: float | None = None,
    ) -> dict[str, object] | None:
        current = time.time() if now is None else float(now)
        result = None
        with self._lock:
            if not any(
                item["profile_id"] == profile_id
                and (
                    item["status"] == "pending"
                    or (
                        item["status"] == "claimed"
                        and float(item["claimed_at"]) < current - 120
                    )
                )
                for item in self._state["proactive_queue"]
            ):
                return None
            with self._transaction() as state:
                changed = False
                for item in state["proactive_queue"]:
                    if (
                        item["profile_id"] == profile_id
                        and item["status"] == "claimed"
                        and float(item["claimed_at"]) < current - 120
                    ):
                        item.update({
                            "status": "pending",
                            "claim_token": "",
                            "claimed_at": 0.0,
                            "adapter": "",
                            "conversation_id": "",
                        })
                        changed = True
                pending = [
                    item for item in state["proactive_queue"]
                    if item["profile_id"] == profile_id and item["status"] == "pending"
                ]
                if pending:
                    message = min(
                        pending,
                        key=lambda item: (-float(item["importance"]), float(item["created_at"])),
                    )
                    token = uuid.uuid4().hex
                    message.update({
                        "status": "claimed",
                        "claim_token": token,
                        "claimed_at": current,
                        "adapter": adapter,
                        "conversation_id": conversation_id,
                    })
                    result = {
                        "opportunity_id": message["id"],
                        "claim_token": token,
                        "message": message["text"],
                        "thought_id": message["thought_id"],
                        "created_at": message["created_at"],
                    }
                    changed = True
                if changed:
                    state["updated_at"] = current
        return result

    def wait_for_proactive(self, profile_id: str, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._proactive_condition:
            while not any(
                item["profile_id"] == profile_id and item["status"] == "pending"
                for item in self._state["proactive_queue"]
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._proactive_condition.wait(remaining)
            return True

    def acknowledge_proactive(
        self,
        message_id: str,
        claim_token: str,
        *,
        success: bool,
        delivery_message_id: str = "",
        now: float | None = None,
    ) -> bool:
        current = time.time() if now is None else float(now)
        accepted = False
        with self._transaction() as state:
            message = next(
                (
                    item for item in state["proactive_queue"]
                    if item["id"] == message_id
                    and item["claim_token"] == claim_token
                    and item["status"] == "claimed"
                ),
                None,
            )
            if message is None:
                return False
            accepted = True
            if success:
                message["status"] = "delivered"
                message["delivered_at"] = current
                message["message_id"] = delivery_message_id
                if message["conversation_id"]:
                    self._insert_turn(state, message["profile_id"], Turn(
                        f"turn_proactive_{uuid.uuid4().hex}",
                        message["profile_id"],
                        message["conversation_id"],
                        "assistant",
                        message["text"],
                        current,
                        f"proactive:{message_id}",
                    ))
                self._record_change(
                    state,
                    message["profile_id"],
                    "proactive_delivery",
                    [f"proactive:{message_id}:delivered", "turn:proactive"],
                    [],
                    current,
                )
            else:
                message.update({
                    "status": "pending",
                    "claim_token": "",
                    "claimed_at": 0.0,
                    "adapter": "",
                    "conversation_id": "",
                })
                state["updated_at"] = current
        if accepted and not success:
            with self._proactive_condition:
                self._proactive_condition.notify_all()
        return accepted

    def debug_snapshot(self, profile_id: str, conversation_id: str, query: str = "") -> dict[str, object]:
        state = self.snapshot(profile_id, conversation_id, query=query)
        with self._lock:
            profile = self._state["profiles"].get(profile_id)
            self_counts = {
                group: len(profile["self"][group]) if profile else 0
                for group in SELF_GROUP_BY_KIND.values()
            }
            last_change = next(
                (
                    copy.deepcopy(item) for item in reversed(self._state["change_log"])
                    if item["profile_id"] == profile_id
                ),
                None,
            )
        return {
            "schema_version": state.schema_version,
            "revision": state.revision,
            "recent_turn_count": len(state.recent_turns),
            "selected_self": [f"{item.kind}:{item.topic}" for item in state.self_items],
            "self_counts": self_counts,
            "selected_memories": [item.id for item in state.memories],
            "mood": {
                "emotion": state.mood.emotion,
                "valence": round(state.mood.valence, 3),
                "energy": round(state.mood.energy, 3),
            },
            "relationship": {
                "familiarity": round(state.relationship.familiarity, 3),
                "trust": round(state.relationship.trust, 3),
                "closeness": round(state.relationship.closeness, 3),
            },
            "inner_life": [thought.id for thought in state.thoughts],
            "last_change": last_change,
        }


_STORE: Store | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = Store()
    return _STORE
