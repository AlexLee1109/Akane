"""Bounded World mutations and validation; persistence belongs to Store."""

from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import asdict, fields, replace

from app.core.state import (
    Experience, Memory, Turn, WorldChange, WorldEntity, WorldEvent, WorldFact, WorldRelation,
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
_SOURCE_KINDS = frozenset({"user_turn", "assistant_turn", "memory", "experience", "world", "environment"})
_KEY_FIELDS = frozenset({"attribute", "predicate", "relation"})
_NONCURRENT_STATE_SOURCE = re.compile(
    r"\b(?:was|were|had been|used to be|yesterday|previously|formerly|earlier|ago|"
    r"last (?:night|week|month|year)|in (?:19|20)\d{2}|"
    r"will be|going to be|tomorrow|next (?:week|month|year))\b", re.IGNORECASE,
)


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
    for row in profile.get('environment', {}).values():
        sources.add(WorldSource('environment', row['instance_id']))
        sources.update(WorldSource('environment', receipt['id']) for receipt in row['receipts'])
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


def derive_world_changes(
    raw: str, user_turn: Turn, assistant_turn: Turn, *,
    world: WorldSnapshot,
    recent_turns: tuple[Turn, ...] = (),
    memories: tuple[Memory, ...] = (),
    experiences: tuple[Experience, ...] = (),
    now: float | None = None,
) -> tuple[WorldChange, ...]:
    """Validate semantic source witnesses before proposing Phase 3A mutations.

    The current user turn is the source, never model-supplied text. Conservative
    framing guards reject nonassertions; values and identities need source spans.
    This does not independently prove arbitrary natural-language entailment.
    """
    from app.core.learning import (
        _EVENT_EVIDENCE_KINDS, _WORLD_EVIDENCE_KINDS, _normalized_evidence,
        _semantic_records, _stable_id, _world_source_is_asserted, _world_span,
    )

    payload = next((item for item in _semantic_records(raw)
                    if item["k"] in _WORLD_EVIDENCE_KINDS), None)
    if payload is None or (
        user_turn.role != "user" or assistant_turn.role != "assistant"
        or not user_turn.id or not assistant_turn.id or user_turn.id == assistant_turn.id
        or user_turn.profile_id != assistant_turn.profile_id
        or user_turn.conversation_id != assistant_turn.conversation_id
    ):
        return ()
    kind = payload["k"]
    required = {"k", "t", "d"}
    if "z" in payload:
        required.add("e")
    if kind != "we":
        required.add("a")
    if kind in {"ws", "wf", "wr", "wr-"}:
        required.add("v")
    optional = {"z", "b"} if kind == "wr" else {"z"}
    if not required <= set(payload) or set(payload) - required - optional:
        return ()
    if any(not isinstance(value, str) or not value.strip() or len(value) > 280
           for value in payload.values()):
        return ()
    expected_scope = {"we": "na", "ws": "na", "wf": "da", "wr": "na",
                      "wr-": "nn", "wv": "pa"}
    if payload["d"] != expected_scope[kind]:
        return ()
    profile_id = user_turn.profile_id
    prior_users = tuple(turn for turn in recent_turns if (
        turn.role == "user" and turn.profile_id == profile_id
        and turn.conversation_id == user_turn.conversation_id
        and turn.id != user_turn.id and turn.created_at <= user_turn.created_at
    ))
    source_turn = user_turn
    sources = (WorldSource("user_turn", user_turn.id),)
    if "z" in payload:
        # Historical evidence can establish a durable fact/occurrence, never
        # silently revive an old mutable value as current.
        if kind not in {"wf", "wv"}:
            return ()
        matches = [item for item in (*memories, *experiences)
                   if item.id == payload["z"] and item.profile_id == profile_id]
        if len(matches) != 1:
            return ()
        source = matches[0]
        if isinstance(source, Experience) and source.kind not in _EVENT_EVIDENCE_KINDS:
            return ()
        if isinstance(source, Memory) and source.subject != "user":
            return ()
        quotes = [turn for turn in prior_users if turn.id in source.source_turn_ids
                  and _normalized_evidence(turn.content) == _normalized_evidence(payload["e"])]
        if len(quotes) != 1:
            return ()
        source_turn = quotes[0]
        sources = (WorldSource("memory" if isinstance(source, Memory) else "experience", source.id),
                   WorldSource("user_turn", source_turn.id))
    # Prior sources still require their complete explicit witness. Current
    # sources are bound directly, so qualifiers cannot be cropped or replaced.
    evidence = source_turn.content
    if not _world_source_is_asserted(evidence):
        return ()
    # A model's mutable-state label cannot turn an explicitly historical or
    # future source into a current observation. Mixed-time sources remain
    # available as qualified conversational text rather than guessed updates.
    if kind == "ws" and _NONCURRENT_STATE_SOURCE.search(evidence):
        return ()
    if re.search(r'["“”]', evidence) or evidence.lstrip().startswith('>'):
        return ()
    if kind != "wr-" and re.search(r"\b(?:not|never|no longer)\b",
                                  _normalized_evidence(evidence)):
        return ()
    current_time = time.time() if now is None else float(now)
    if source_turn.created_at > current_time:
        return ()
    timestamp = source_turn.created_at
    entities = tuple(item for item in world.entities if item.profile_id == profile_id)
    formed: list[WorldChange] = []

    def resolve(label: str, *, discourse: bool = False) -> WorldEntity | None:
        reference = label.strip()
        label = _normalized_evidence(label)
        if not label or len(label) > 200:
            return None
        matches = [item for item in entities
                   if item.id == reference or _normalized_evidence(item.label) == label]
        if len(matches) > 1:
            return None
        if matches:
            item = matches[0]
            if _world_span(item.label, evidence) or _world_span(item.id, evidence):
                return item
            if (discourse and source_turn == user_turn and prior_users
                    and re.search(r"\b(?:it|that)\b", evidence, re.I)):
                previous = max(prior_users, key=lambda turn: turn.created_at)
                mentioned = [item for item in entities if _world_span(item.label, previous.content)]
                if len(mentioned) == 1 and mentioned[0].id == item.id:
                    return item
            return None
        if not _world_span(label, evidence):
            return None
        item = WorldEntity(_stable_id("world_entity", profile_id, label), profile_id,
                           label, timestamp, timestamp, sources)
        if all(change.record.id != item.id for change in formed):
            formed.append(WorldChange("upsert", item))
        return item

    subject = resolve(payload["t"], discourse=True)
    if subject is None:
        return ()
    if kind == "we":
        return tuple(formed)
    attribute = _normalized_evidence(payload["a"])
    if not attribute or len(attribute) > 200:
        return ()
    if kind == "wf" and attribute in {"status", "activity", "focus", "intention"}:
        return ()
    value = payload.get("v", "").strip()
    if kind in {"ws", "wf"} and not _world_span(value, evidence):
        return ()
    if kind == "ws":
        current = world.current_state(subject.id, attribute)
        if current:
            sources += (WorldSource("world", current.id),)
        item = WorldState(_stable_id("world_state", subject.id, attribute), profile_id,
                          subject.id, attribute, value, 0.85, timestamp, timestamp, sources)
    elif kind == "wf":
        item = WorldFact(_stable_id("world_fact", subject.id, attribute, value), profile_id,
                         subject.id, attribute, value, 0.85, source_turn.created_at, sources)
    elif kind == "wv":
        if not _world_span(attribute, evidence):
            return ()
        item = WorldEvent(_stable_id("world_event", source_turn.id, subject.id, attribute),
                          profile_id, (subject.id,), attribute, 0.85,
                          source_turn.created_at, sources)
    else:
        target = resolve(value)
        if target is None:
            return ()
        current = None
        if kind == "wr-" or "b" in payload:
            old = resolve(payload.get("b", value))
            if old is None:
                return ()
            matches = [item for item in world.relations if item.profile_id == profile_id
                       and (item.subject_id, item.relation, item.object_id)
                       == (subject.id, attribute, old.id)]
            if len(matches) != 1:
                return ()
            current = matches[0]
            sources += (WorldSource("world", current.id),)
        item = WorldRelation(
            current.id if current else _stable_id("world_relation", subject.id, attribute, target.id),
            profile_id, subject.id, attribute, target.id, 0.85, timestamp, timestamp, sources,
        )
    return (*formed, WorldChange("remove" if kind == "wr-" else "upsert", item))
