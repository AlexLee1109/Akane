"""Bounded World mutations and validation; persistence belongs to Store."""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, fields, replace

from app.core.state import (
    WorldChange, WorldEntity, WorldEvent, WorldFact, WorldRelation,
    WorldSnapshot, WorldSource, WorldState,
)


WORLD_LIMITS = {
    "entities": 64, "states": 128, "facts": 128, "relations": 128, "events": 64,
}
WORLD_SOURCE_LIMIT = 6
_RECORD_TYPES = {
    "entities": WorldEntity, "states": WorldState, "facts": WorldFact,
    "relations": WorldRelation, "events": WorldEvent,
}
_SOURCE_KINDS = frozenset({"user_turn", "assistant_turn", "memory", "experience", "world"})
_KEY_FIELDS = frozenset({"attribute", "predicate", "relation"})


def empty_world() -> dict[str, list]:
    return {name: [] for name in _RECORD_TYPES}


def world_row(item) -> dict:
    row = asdict(item)
    row.pop("profile_id")
    return row


def _text(value: object, *, optional: bool = False, limit: int = 200) -> bool:
    return (
        isinstance(value, str) and value == value.strip()
        and (bool(value) or optional) and len(value) <= limit
    )


def world_record(row: object, record_type: type, profile_id: str):
    expected = {field.name for field in fields(record_type)} - {"profile_id"}
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError("World record has an invalid structure.")
    data = dict(row)
    for key, value in data.items():
        if key in {"sources", "entity_ids"}:
            continue
        if key in {"created_at", "updated_at", "confidence"}:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (key == "confidence" and value > 1)):
                raise ValueError("World numeric value is invalid.")
        else:
            optional = key in {"before", "after"} or (
                record_type is WorldEvent and key == "attribute"
            ) or (record_type is WorldEntity and key == "kind")
            if not _text(value, optional=optional, limit=280 if key in {
                "value", "before", "after",
            } else 200):
                raise ValueError("World text is invalid or exceeds its bound.")
            if key in _KEY_FIELDS and value != value.casefold():
                raise ValueError("World keys must be canonical lowercase text.")
    if data.get("updated_at", data["created_at"]) < data["created_at"]:
        raise ValueError("World update precedes creation.")
    sources = data["sources"]
    if not isinstance(sources, (list, tuple)) or not 1 <= len(sources) <= WORLD_SOURCE_LIMIT:
        raise ValueError("World provenance is required and bounded.")
    parsed = []
    for source in sources:
        if (not isinstance(source, dict) or set(source) != {"kind", "id"}
                or not isinstance(source["kind"], str)
                or source["kind"] not in _SOURCE_KINDS or not _text(source["id"])):
            raise ValueError("World source is invalid.")
        parsed.append(WorldSource(**source))
    if len(set(parsed)) != len(parsed):
        raise ValueError("World sources must be unique.")
    data["sources"] = tuple(parsed)
    if record_type is WorldEvent:
        ids = data["entity_ids"]
        if (not isinstance(ids, (list, tuple)) or not 1 <= len(ids) <= 6
                or any(not _text(item) for item in ids) or len(set(ids)) != len(ids)):
            raise ValueError("World event entities are invalid or exceed their bound.")
        data["entity_ids"] = tuple(ids)
    return record_type(profile_id=profile_id, **data)


def _key(item) -> tuple:
    if isinstance(item, WorldState):
        return item.entity_id, item.attribute
    if isinstance(item, WorldFact):
        return item.subject_id, item.predicate, item.value
    if isinstance(item, WorldRelation):
        return item.subject_id, item.relation, item.object_id
    return (item.id,)


def _entity_ids(item) -> tuple[str, ...]:
    if isinstance(item, WorldState):
        return (item.entity_id,)
    if isinstance(item, WorldFact):
        return (item.subject_id,)
    if isinstance(item, WorldRelation):
        return item.subject_id, item.object_id
    if isinstance(item, WorldEvent):
        return item.entity_ids
    return ()


def world_snapshot(world: dict, profile_id: str) -> WorldSnapshot:
    return WorldSnapshot(**{
        name: tuple(world_record(row, kind, profile_id) for row in world[name])
        for name, kind in _RECORD_TYPES.items()
    })


def validate_world(world: object, profile_id: str) -> None:
    if not isinstance(world, dict) or set(world) != set(_RECORD_TYPES):
        raise ValueError("World has an invalid structure.")
    all_ids = set()
    entity_ids = set()
    for name, kind in _RECORD_TYPES.items():
        rows = world[name]
        if not isinstance(rows, list) or len(rows) > WORLD_LIMITS[name]:
            raise ValueError("World collection is invalid or exceeds its bound.")
        keys = set()
        for row in rows:
            item = world_record(row, kind, profile_id)
            key = _key(item)
            if item.id in all_ids or key in keys:
                raise ValueError("World IDs and current keys must be unique.")
            if set(_entity_ids(item)) - entity_ids:
                raise ValueError("World references an unknown entity.")
            if name == "entities":
                entity_ids.add(item.id)
            all_ids.add(item.id)
            keys.add(key)
    # Source references are checked when admitted. Their original records may
    # later be pruned; retaining the typed IDs preserves the provenance trail.


def known_world_sources(profile: dict) -> set[WorldSource]:
    sources = {
        WorldSource(row["role"] + "_turn", row["id"])
        for conversation in profile["conversations"].values()
        for row in conversation["turns"] if row["role"] in {"user", "assistant"}
    }
    for kind, collection in (("memory", "memories"), ("experience", "experiences")):
        sources.update(WorldSource(kind, row["id"]) for row in profile[collection])
    sources.update(WorldSource("world", row["id"])
                   for rows in profile["world"].values() for row in rows)
    return sources


def apply_world_change(
    world: dict, change: WorldChange, profile_id: str,
    known_sources: set[WorldSource],
) -> tuple[bool, str]:
    """Mutate a Store candidate only after validating this entire change."""
    item = change.record
    name = next((name for name, kind in _RECORD_TYPES.items() if type(item) is kind), "")
    if not name or change.action not in {"upsert", "remove"}:
        return False, "world:invalid"
    label = f"world:{name}"
    if item.profile_id != profile_id:
        return False, f"{label}:profile-mismatch"
    if change.action == "remove" and not isinstance(item, WorldRelation):
        return False, f"{label}:invalid-action"
    # The proposal API canonicalizes keys; persisted rows must already be canonical.
    item = replace(item, **{
        field.name: getattr(item, field.name).strip().casefold()
        for field in fields(item) if field.name in _KEY_FIELDS
        and isinstance(getattr(item, field.name), str)
    })
    try:
        item = world_record(world_row(item), type(item), profile_id)
    except (ValueError, TypeError):
        return False, f"{label}:invalid"
    if set(item.sources) - known_sources:
        return False, f"{label}:evidence-unknown"
    if set(_entity_ids(item)) - {row["id"] for row in world["entities"]}:
        return False, f"{label}:entity-unknown"
    rows = world[name]
    records = [world_record(row, type(item), profile_id) for row in rows]
    by_id = next((record for record in records if record.id == item.id), None)
    current = by_id or next((record for record in records if _key(record) == _key(item)), None)
    if any(row["id"] == item.id for group, values in world.items()
           if group != name for row in values):
        return False, f"{label}:id-conflict"
    if by_id and (
        isinstance(item, WorldState) and _key(item) != _key(by_id)
        or isinstance(item, WorldRelation)
        and (item.subject_id, item.relation) != (by_id.subject_id, by_id.relation)
    ):
        return False, f"{label}:identity-conflict"
    if current and isinstance(item, (WorldFact, WorldEvent)):
        duplicate = item == current or (
            isinstance(item, WorldFact) and _key(item) == _key(current)
        )
        return False, f"{label}:duplicate" if duplicate else f"{label}:immutable"
    if change.action == "remove" and (by_id is None or _key(item) != _key(by_id)):
        return False, f"{label}:target-mismatch"
    if current is None and isinstance(item, WorldRelation) and any(
        event["kind"] == "relation_transition"
        and event["attribute"] == item.relation
        and item.subject_id == event["entity_ids"][0]
        and item.object_id == event["before"]
        and item.updated_at <= event["created_at"]
        for event in world["events"]
    ):
        return False, f"{label}:stale"
    if current:
        item = replace(item, id=current.id, created_at=current.created_at)
        if item == current:
            return False, f"{label}:duplicate"
        if item.updated_at <= current.updated_at:
            return False, f"{label}:stale"
        if any(record.id != current.id and _key(record) == _key(item) for record in records):
            return False, f"{label}:key-conflict"
    elif name != "events" and len(rows) >= WORLD_LIMITS[name]:
        return False, f"{label}:limit"

    event = None
    if current and isinstance(item, (WorldState, WorldRelation)):
        before = current.value if name == "states" else current.object_id
        after = item.value if name == "states" else item.object_id
        if change.action == "remove":
            after = ""
        if before != after:
            event = WorldEvent(
                id=f"world-event:{uuid.uuid4().hex}", profile_id=profile_id,
                entity_ids=tuple(dict.fromkeys((*_entity_ids(current), *_entity_ids(item)))),
                kind="state_transition" if name == "states" else "relation_transition",
                attribute=item.attribute if name == "states" else item.relation,
                before=before, after=after, confidence=item.confidence,
                created_at=item.updated_at,
                sources=tuple(dict.fromkeys((*current.sources, *item.sources)))[-WORLD_SOURCE_LIMIT:],
            )
    if current:
        rows[:] = [row for row in rows if row["id"] != current.id]
    if change.action != "remove":
        rows.append(world_row(item))
    if event:
        world["events"].append(world_row(event))
    world["events"][:] = sorted(
        world["events"], key=lambda row: row["created_at"],
    )[-WORLD_LIMITS["events"]:]
    action = "remove" if change.action == "remove" else "update" if current else "form"
    return True, f"{label}:{action}"
