"""Deterministic operations over Akane's psychological state."""

from __future__ import annotations

import json
import time
import uuid

from app.core.character import load_character_profile
from app.core.state import (
    SELF_KINDS,
    ProactiveMessage,
    SelfChange,
    SelfItem,
    StateChangeProposal,
    Thought,
    ThoughtChange,
    clamp,
)
from app.core.utils import text_key

_UNSUPPORTED_EVENT_TERMS = {
    "ate", "drank", "visited", "traveled", "watched", "played", "touched", "felt",
    "smelled", "tasted", "heard", "saw", "searched", "browsed", "read",
}


def form_self_item(
    profile_id: str,
    *,
    kind: str,
    topic: str,
    value: str,
    strength: float,
    confidence: float,
    reason: str,
    source_ids: tuple[str, ...],
    now: float | None = None,
) -> SelfChange:
    if kind not in SELF_KINDS:
        raise ValueError(f"Unknown Self item kind: {kind}")
    timestamp = time.time() if now is None else float(now)
    item = SelfItem(
        id=f"self_{uuid.uuid4().hex}",
        profile_id=profile_id,
        kind=kind,
        topic=topic.strip(),
        value=value.strip(),
        strength=clamp(strength, -1, 1),
        confidence=clamp(confidence, 0, 1),
        reason=reason.strip(),
        status="uncertain" if confidence < 0.5 else "active",
        created_at=timestamp,
        updated_at=timestamp,
        source_ids=source_ids,
    )
    return SelfChange("form", item=item)


def revise_self_item(
    current: SelfItem,
    *,
    action: str,
    value: str,
    strength: float,
    confidence: float,
    reason: str,
    source_ids: tuple[str, ...],
    now: float | None = None,
) -> SelfChange:
    if action not in {"reinforce", "weaken", "revise", "complete", "abandon"}:
        raise ValueError(f"Invalid Self lifecycle action: {action}")
    timestamp = time.time() if now is None else float(now)
    sources = tuple(dict.fromkeys((*current.source_ids, *source_ids)))[-12:]
    status = current.status
    if status in {"active", "uncertain"}:
        status = "uncertain" if confidence < 0.5 else "active"
    item = SelfItem(
        id=current.id,
        profile_id=current.profile_id,
        kind=current.kind,
        topic=current.topic,
        value=value.strip(),
        strength=clamp(strength, -1, 1),
        confidence=clamp(confidence, 0, 1),
        reason=reason.strip(),
        status=status,
        created_at=current.created_at,
        updated_at=timestamp,
        source_ids=sources,
        revision_count=current.revision_count + 1,
    )
    return SelfChange(action, item=item, target_id=current.id)


def _payload(output: object) -> dict[str, object]:
    text = str(output or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return {"action": "quiet"}
    return value if isinstance(value, dict) else {"action": "quiet"}


def validate_inner_life(output: object, *, context, now: float | None = None) -> StateChangeProposal:
    current = time.time() if now is None else float(now)
    raw = _payload(output)
    action = str(raw.get("action") or "quiet")
    profile_id = context.state.profile_id
    if action == "quiet":
        return StateChangeProposal(profile_id, origin="inner_life")
    existing = {item.id for item in (*context.state.self_items, *context.state.memories, *context.state.thoughts)}
    seeds = {f"seed:{value.casefold()}" for value in load_character_profile().seed_interests}
    source_ids = tuple(
        str(item) for item in raw.get("source_ids", [])
        if str(item) in existing or str(item) in seeds
    ) if isinstance(raw.get("source_ids"), list) else ()
    topic = " ".join(str(raw.get("topic") or "").split())[:160]
    thought_text = " ".join(str(raw.get("thought") or "").split())[:800]
    if not source_ids or not topic or not thought_text:
        return StateChangeProposal(profile_id, origin="inner_life", rejected=("thought:missing-source",))
    if set(text_key(thought_text).split()) & _UNSUPPORTED_EVENT_TERMS:
        return StateChangeProposal(profile_id, origin="inner_life", rejected=("thought:unsupported-event",))
    importance = max(0.0, min(1.0, float(raw.get("importance") or 0.3)))
    last_user_at = next(
        (turn.created_at for turn in reversed(context.state.recent_turns) if turn.role == "user"),
        None,
    )
    interruption_window_clear = last_user_at is None or current - last_user_at >= 600
    share_worthy = (
        bool(raw.get("share_worthy"))
        and importance >= 0.75
        and interruption_window_clear
    )
    active = {thought.id: thought for thought in context.state.thoughts}
    target_id = str(raw.get("target_id") or "")
    thought_id = target_id
    changes: list[ThoughtChange] = []
    proactive: list[ProactiveMessage] = []
    self_changes = []
    if action in {"resolve", "expire"} and target_id in active:
        changes.append(ThoughtChange(action, target_id=target_id))
    elif action in {"add", "continue"}:
        if action == "continue" and target_id not in active:
            return StateChangeProposal(profile_id, origin="inner_life", rejected=("thought:unknown-target",))
        thought_id = target_id if action == "continue" else f"thought_{uuid.uuid4().hex}"
        started_at = active[target_id].started_at if action == "continue" else current
        thought = Thought(
            thought_id, profile_id, topic, thought_text, importance, source_ids,
            started_at, current, "active", share_worthy,
        )
        changes.append(ThoughtChange(action, thought=thought, target_id=target_id))
        if share_worthy:
            proactive.append(ProactiveMessage(
                f"proactive_{uuid.uuid4().hex}", profile_id, thought_id,
                thought_text, importance, current,
            ))
    else:
        return StateChangeProposal(profile_id, origin="inner_life", rejected=("thought:invalid-action",))

    raw_goal = raw.get("goal")
    if isinstance(raw_goal, dict):
        goal_action = str(raw_goal.get("action") or "")
        goal_topic = " ".join(str(raw_goal.get("topic") or "").split())[:160]
        goal_value = " ".join(str(raw_goal.get("value") or "").split())[:500]
        goal_reason = " ".join(str(raw_goal.get("reason") or "").split())[:500]
        try:
            if goal_action == "form" and goal_topic and goal_value and goal_reason:
                self_changes.append(form_self_item(
                    profile_id,
                    kind="goal",
                    topic=goal_topic,
                    value=goal_value,
                    strength=max(0.1, min(0.55, float(raw_goal.get("strength") or 0.3))),
                    confidence=max(0.1, min(0.4, float(raw_goal.get("confidence") or 0.25))),
                    reason=goal_reason,
                    source_ids=(thought_id,),
                    now=current,
                ))
            else:
                goal_id = str(raw_goal.get("target_id") or "")
                current_goal = next(
                    (item for item in context.state.self_items if item.id == goal_id and item.kind == "goal"),
                    None,
                )
                if current_goal is not None and goal_action in {"revise", "complete", "abandon"}:
                    self_changes.append(revise_self_item(
                        current_goal,
                        action=goal_action,
                        value=goal_value or current_goal.value,
                        strength=float(raw_goal.get("strength") or current_goal.strength),
                        confidence=float(raw_goal.get("confidence") or current_goal.confidence),
                        reason=goal_reason or current_goal.reason,
                        source_ids=(thought_id,),
                        now=current,
                    ))
        except (TypeError, ValueError):
            pass
    return StateChangeProposal(
        profile_id,
        self_items=tuple(self_changes),
        thoughts=tuple(changes),
        proactive_messages=tuple(proactive),
        origin="inner_life",
    )
