"""Fresh current situation over ordinary World slots; no background activity."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.core.mind import _semantic_records, _stable_id, _world_source_is_asserted, _world_span
from app.core.state import WorldChange, WorldEntity, WorldSource, WorldState


SITUATION_TTL = 300.0
SLOTS = ("activity", "focus", "status", "intention")
VALUE_BYTES = 48
# This scoped exception uses the existing ws grammar, without changing 3B rules.
SITUATION_RULES = (
    "AKANE_NOW replaces earlier situation; omitted slots are unknown. Use silently; "
    "no invented offscreen life. AKANE_BEFORE is history only. "
    "For a grounded present Akane focus/intention/activity, existing ws may use "
    "t=Akane,a=slot,v=short exact A span,d=na; runtime binds A. "
    "Physical activity still needs dialogue/runtime support. Recall needs n."
)


@dataclass(frozen=True, slots=True)
class RuntimeSituation:
    """Trusted caller observation of work actually active during this turn.

    This is not model or user input. No tool integration is inferred from context.
    """

    activity: str = "talking"
    focus: str = ""
    status: str = "engaged"

    def __post_init__(self):
        for value in (self.activity, self.focus, self.status):
            if value and not _value(value):
                raise ValueError("Runtime situation must be a compact single-line value.")


def _value(value):
    return (isinstance(value, str) and bool(value.strip()) and value == value.strip()
            and len(value.encode("utf-8")) <= VALUE_BYTES
            and not any(ord(c) < 32 for c in value))


def akane_entity(world, profile_id):
    candidates = [e for e in world.entities
                  if e.profile_id == profile_id and e.label.casefold() == "akane"]
    return candidates[0] if len(candidates) == 1 else None


def current_situation(world, profile_id, now):
    entity = akane_entity(world, profile_id)
    if entity is None:
        return {}
    return {s.attribute: s for s in world.states
            if s.profile_id == profile_id and s.entity_id == entity.id
            and s.attribute in SLOTS and 0 <= now - s.updated_at < SITUATION_TTL}


def situation_context(world, profile_id, now, query="", runtime=None):
    runtime = runtime or RuntimeSituation()
    current = {key: s.value for key, s in current_situation(world, profile_id, now).items()}
    current.update(activity=runtime.activity, status=runtime.status)
    if runtime.focus:
        current["focus"] = runtime.focus
    # Fixed byte bounds also bound native tokenizer input, independent of language.
    current = {k: v.encode("utf-8")[:VALUE_BYTES].decode("utf-8", errors="ignore")
               for k, v in current.items() if v}
    text = "AKANE_NOW " + json.dumps(current, ensure_ascii=False, separators=(",", ":"))
    if re.search(r"\b(before|previous|previously|earlier)\b", query.casefold()):
        entity = akane_entity(world, profile_id)
        previous = "unknown"
        if entity:
            activity = world.current_state(entity.id, "activity")
            if activity and activity.updated_at <= now and (
                activity.value != runtime.activity or now - activity.updated_at >= SITUATION_TTL
            ):
                previous = activity.value
            else:
                events = [e for e in world.events if e.profile_id == profile_id
                          and entity.id in e.entity_ids and e.attribute == "activity"
                          and e.kind == "state_transition" and e.created_at <= now]
                if events:
                    previous = max(events, key=lambda e: e.created_at).before
        previous = previous.encode("utf-8")[:VALUE_BYTES].decode("utf-8", errors="ignore")
        text += "\nAKANE_BEFORE " + json.dumps(previous, ensure_ascii=False)
    return text


def _own(record, world):
    target = record.get("t")
    if isinstance(target, str):
        target = target.strip()
    return isinstance(target, str) and (
        target.casefold() == "akane" or any(
            e.id == target and e.label.casefold() == "akane" for e in world.entities
        )
    ) and record.get("a") in SLOTS


def external_evidence(raw, world):
    """Reserved situation slots cannot fall through to external-World admission."""
    records = _semantic_records(raw)
    if any(_own(r, world) for r in records):
        return json.dumps([r for r in records if not _own(r, world)])
    return raw


def derive_situation_changes(raw, user_turn, assistant_turn, *, world, now, runtime=None):
    if (user_turn.role != "user" or assistant_turn.role != "assistant"
            or user_turn.profile_id != assistant_turn.profile_id
            or user_turn.conversation_id != assistant_turn.conversation_id
            or not user_turn.id or not assistant_turn.id or user_turn.id == assistant_turn.id
            or user_turn.created_at > now or assistant_turn.created_at > now):
        return ()
    now = assistant_turn.created_at
    profile_id = user_turn.profile_id
    entity = akane_entity(world, profile_id)
    if entity is None and any(e.label.casefold() == "akane" for e in world.entities):
        return ()  # Ambiguous identity or foreign snapshot: fail closed.
    runtime = runtime or RuntimeSituation()
    sources = (WorldSource("user_turn", user_turn.id),)
    changes = []
    if entity is None:
        entity = WorldEntity(_stable_id("world_entity", profile_id, "akane"),
                             profile_id, "Akane", now, now, sources)
        changes.append(WorldChange("upsert", entity))
    values = {"activity": (runtime.activity, sources), "status": (runtime.status, sources)}
    if runtime.focus:
        values["focus"] = runtime.focus, sources
    current = current_situation(world, profile_id, now)
    for record in _semantic_records(raw):
        if not _own(record, world):
            continue
        if (set(record) != {"k", "t", "a", "v", "d"}
                or record["k"] != "ws" or record["d"] != "na"
                or not _value(record["v"])):
            continue
        slot, value = record["a"], record["v"]
        reply = assistant_turn.content
        # Present declarations only. Entire reply is retained by its canonical ID.
        if re.search(r"\b(if|suppose|imagine|pretend|hypothetical)\b", user_turn.content, re.I):
            continue
        if (value not in reply or re.search(
                r"\b(if|might|maybe|perhaps|pretend|imagine|always|usually|earlier|yesterday|before|ago|was|went|ate|watched|not|never|no|said|says)\b|n't\b|\?",
                reply, re.I)):
            continue
        # The value must belong to the first-person declaration itself, not
        # another sentence, a quoted speaker, or a coincidental substring.
        prefixes = {
            "activity": r"I(?:'m| am)\s+",
            "status": r"I(?:'m| am)\s+",
            "focus": r"I(?:'m| am)\s+(?:focused on|focusing on|thinking about)\s+",
            "intention": r"I(?:'ll| will| intend to| plan to| am going to)\s+",
        }
        declaration = re.search(
            r"(?:^|[.!]\s+|^I think )" + prefixes[slot]
            + re.escape(value) + r"(?!\w)", reply, re.I,
        )
        if declaration is None:
            continue
        if slot == "intention":
            grounded = True
        else:
            support = (
                (_world_source_is_asserted(user_turn.content)
                 and _world_span(value, user_turn.content)
                 and not re.search(
                     r"\b(not|never|no|earlier|yesterday|before|ago|was|were|went)\b|n't\b",
                     user_turn.content, re.I)
                 and (slot != "activity" or re.search(
                     r"\b(?:we(?:'re| are)|you(?:'re| are)|Akane is)\s+"
                     + re.escape(value) + r"(?!\w)", user_turn.content, re.I)))
                or (slot in current and current[slot].value == value)
                or (slot in values and values[slot][0] == value)
            )
            grounded = support
        if grounded:
            values[slot] = value, (WorldSource("assistant_turn", assistant_turn.id),
                                  WorldSource("user_turn", user_turn.id))
    for slot, (value, source) in values.items():
        if not value:
            continue
        old = world.current_state(entity.id, slot)
        changes.append(WorldChange("upsert", WorldState(
            old.id if old else _stable_id("world_state", entity.id, slot),
            profile_id, entity.id, slot, value, 1.0,
            old.created_at if old else now, now, source,
        )))
    return tuple(changes)
