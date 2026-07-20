"""Deterministic persistent offscreen life for Akane's shared profile state."""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, replace

from app.core.utils import compact_text

LIFE_STATE_VERSION = 4
_DEFAULT_ACTIVITY_TTL_SECONDS = 24 * 60 * 60
_OFFSCREEN_SOURCE = "offscreen_life"
_LEGACY_OFFSCREEN_SOURCES = {"offscreen_schedule"}
_UNTRUSTED_ACTIVITY_SOURCES = {"generated", "legacy_synthetic", "synthetic"}
_ACTIVITY_STATUSES = {"active", "completed", "cancelled"}
_MAX_RECENT_ACTIVITIES = 20
_MAX_COMPLETED_PER_ADVANCE = 4
_MAX_SELECTION_ATTEMPTS = 8
_MIN_OPPORTUNITY_DELAY = 30 * 60
_MAX_OPPORTUNITY_DELAY = 2 * 60 * 60
_MAX_NEED_ELAPSED = 48 * 60 * 60
_CONVERSATION_TTL_SECONDS = 30 * 60
_AUTHORITY_RANKS = {
    "none": 0,
    "quiet": 1,
    "simulated": 2,
    "conversation": 3,
    "verified": 4,
    "explicit": 5,
}
_ACTIVITY_ALIASES = {
    "watched_anime": "watching_anime",
    "read_manga": "reading_manga",
    "played_game": "playing_game",
    "listened_music": "listening_music",
    "browsed_interest": "browsing_interests",
    "rested": "resting",
    "bored_downtime": "quiet_downtime",
}


def _timestamp(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return max(0.0, default)
    return max(0.0, number) if math.isfinite(number) else max(0.0, default)


def _signed_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, float(value)))


def _pairs(
    value: object,
    *,
    limit: int,
    delta_limit: float,
) -> tuple[tuple[str, float], ...]:
    pairs: list[tuple[str, float]] = []
    for item in value if isinstance(value, (list, tuple)) else ():
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        name = compact_text(item[0], 32).lower()
        number = _signed_number(item[1])
        if name and number is not None:
            pairs.append((name, max(-delta_limit, min(delta_limit, number))))
    return tuple(pairs[:limit])


def _authority_for(source: str, activity_type: str, supplied: str = "") -> str:
    authority = compact_text(supplied, 24).lower()
    if authority in _AUTHORITY_RANKS:
        return authority
    if source in {_OFFSCREEN_SOURCE, *_LEGACY_OFFSCREEN_SOURCES}:
        return "quiet" if activity_type == "quiet_downtime" else "simulated"
    if source == "conversation":
        return "conversation"
    if source.startswith(("arcane", "owner", "user_explicit")):
        return "explicit"
    return "verified"


@dataclass(frozen=True, slots=True)
class LifeNeeds:
    """Slow activity drives, distinct from Akane's canonical emotional mood."""

    energy: float = 0.65
    social: float = 0.35
    curiosity: float = 0.45
    stimulation: float = 0.50

    @classmethod
    def from_dict(cls, payload: object) -> "LifeNeeds":
        values = payload if isinstance(payload, dict) else {}
        return cls(
            energy=_bounded(_timestamp(values.get("energy"), 0.65)),
            social=_bounded(_timestamp(values.get("social"), 0.35)),
            curiosity=_bounded(_timestamp(values.get("curiosity"), 0.45)),
            stimulation=_bounded(_timestamp(values.get("stimulation"), 0.50)),
        )


@dataclass(frozen=True, slots=True)
class ActivityRecord:
    """One canonical current, completed, or cancelled activity record."""

    activity_type: str
    source: str
    description: str
    started_at: float
    updated_at: float
    status: str = "active"
    confidence: float = 1.0
    expires_at: float = 0.0
    completed_at: float = 0.0
    scope: str = "profile"
    event_id: str = ""
    subject: str = ""
    reaction: str = ""
    authority: str = ""
    mood_effects: tuple[tuple[str, float], ...] = ()
    need_effects: tuple[tuple[str, float], ...] = ()
    preference_effects: tuple[tuple[str, float], ...] = ()
    short_emotion: str = ""

    @classmethod
    def from_dict(cls, payload: object) -> "ActivityRecord | None":
        if not isinstance(payload, dict):
            return None
        source = compact_text(payload.get("source"), 48).lower()
        description = compact_text(
            payload.get("description") or payload.get("summary"),
            240,
        )
        raw_type = compact_text(
            payload.get("activity_type") or payload.get("event_type"),
            48,
        ).lower()
        activity_type = _ACTIVITY_ALIASES.get(raw_type, raw_type)
        status = compact_text(payload.get("status"), 24).lower() or "active"
        if status == "observed":
            status = "completed"
        if (
            not source
            or source in _UNTRUSTED_ACTIVITY_SOURCES
            or not description
            or not activity_type
            or status not in _ACTIVITY_STATUSES
        ):
            return None
        started_at = _timestamp(payload.get("started_at"))
        updated_at = max(started_at, _timestamp(payload.get("updated_at"), started_at))
        expires_at = _timestamp(
            payload.get("expires_at") or payload.get("expected_completion_at")
        )
        completed_at = _timestamp(payload.get("completed_at"))
        if status != "active" and not completed_at:
            completed_at = updated_at
        confidence = min(1.0, _timestamp(payload.get("confidence"), 1.0))
        if confidence < 0.5:
            return None
        event_id = compact_text(
            payload.get("event_id") or payload.get("activity_id"),
            64,
        )
        if not event_id:
            seed = (
                f"migrated-activity:{source}:{activity_type}:{started_at:.6f}:"
                f"{description}"
            )
            event_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        return cls(
            activity_type=activity_type,
            source=(
                _OFFSCREEN_SOURCE if source in _LEGACY_OFFSCREEN_SOURCES else source
            ),
            description=description,
            started_at=started_at,
            updated_at=updated_at,
            status=status,
            confidence=confidence,
            expires_at=expires_at,
            completed_at=completed_at,
            scope=compact_text(payload.get("scope"), 120) or "profile",
            event_id=event_id,
            subject=compact_text(payload.get("subject") or payload.get("title"), 120),
            reaction=compact_text(payload.get("reaction"), 120),
            authority=_authority_for(
                source,
                activity_type,
                compact_text(payload.get("authority"), 24),
            ),
            mood_effects=_pairs(payload.get("mood_effects"), limit=5, delta_limit=0.10),
            need_effects=_pairs(payload.get("need_effects"), limit=4, delta_limit=0.25),
            preference_effects=_pairs(
                payload.get("preference_effects"),
                limit=3,
                delta_limit=0.03,
            ),
            short_emotion=compact_text(
                payload.get("short_emotion") or payload.get("short_emotion_effect"),
                24,
            ).lower(),
        )

    def is_active_for(self, now: float, scope: str) -> bool:
        current = _timestamp(now, self.updated_at)
        return (
            self.status == "active"
            and self.confidence >= 0.5
            and (not self.expires_at or current < self.expires_at)
            and self.scope in {"profile", compact_text(scope, 120)}
        )


@dataclass(frozen=True, slots=True)
class LifeState:
    activity: ActivityRecord | None = None
    recent_events: tuple[ActivityRecord, ...] = ()
    needs: LifeNeeds = LifeNeeds()
    activity_preferences: tuple[tuple[str, float], ...] = ()
    last_processed_at: float = 0.0
    next_opportunity_at: float = 0.0
    version: int = LIFE_STATE_VERSION
    legacy_activity_discarded: bool = False

    @classmethod
    def from_dict(cls, payload: object) -> "LifeState":
        if not isinstance(payload, dict):
            return cls()
        raw_events = payload.get("recent_events")
        events = tuple(
            event
            for item in (raw_events if isinstance(raw_events, list) else [])
            if (event := ActivityRecord.from_dict(item)) is not None
            and event.status in {"completed", "cancelled"}
        )[-_MAX_RECENT_ACTIVITIES:]
        activity = ActivityRecord.from_dict(payload.get("activity"))
        if activity is not None and activity.status != "active":
            events = _bounded_history([*events, activity])
            activity = None
        raw_version = int(_timestamp(payload.get("version")))
        if raw_version >= 3:
            return cls(
                activity=activity,
                recent_events=events,
                needs=LifeNeeds.from_dict(payload.get("needs")),
                activity_preferences=_pairs(
                    payload.get("activity_preferences"),
                    limit=12,
                    delta_limit=0.15,
                ),
                last_processed_at=_timestamp(payload.get("last_processed_at")),
                next_opportunity_at=(
                    _timestamp(payload.get("next_opportunity_at"))
                    if raw_version >= LIFE_STATE_VERSION
                    else 0.0
                ),
                legacy_activity_discarded=bool(
                    payload.get("legacy_activity_discarded")
                ),
            )
        legacy_present = bool(
            compact_text(payload.get("current_activity"), 120)
            or compact_text(payload.get("activity_detail"), 240)
        )
        return cls(
            activity=activity,
            legacy_activity_discarded=legacy_present,
        )


@dataclass(frozen=True, slots=True)
class LifeEvolution:
    state: LifeState
    new_events: tuple[ActivityRecord, ...] = ()
    generated_activities: tuple[ActivityRecord, ...] = ()
    previous_activity: ActivityRecord | None = None
    elapsed_seconds: float = 0.0
    needs_before: LifeNeeds = LifeNeeds()
    needs_after: LifeNeeds = LifeNeeds()
    need_changes: tuple[tuple[str, float], ...] = ()
    preference_changes: tuple[tuple[str, float], ...] = ()
    selection_bucket: int = 0
    selection_candidates: tuple[tuple[str, float], ...] = ()
    repetition_penalties: tuple[tuple[str, float], ...] = ()
    cooldowns: tuple[str, ...] = ()
    reason_no_activity: str = ""
    authority_source: str = "none"
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ActivitySpec:
    activity_type: str
    description: str
    base_weight: float
    duration_minutes: tuple[int, int]
    reactions: tuple[str, ...]


_ACTIVITY_CATALOG = (
    _ActivitySpec(
        "watching_anime",
        "watched some anime",
        0.10,
        (20, 60),
        ("enjoyed it", "found it interesting", "liked the atmosphere", "felt neutral about it"),
    ),
    _ActivitySpec(
        "reading_manga",
        "read manga for a while",
        0.08,
        (15, 60),
        (
            "became curious about what happens next",
            "liked the characters",
            "enjoyed the pace",
            "felt neutral about it",
        ),
    ),
    _ActivitySpec(
        "playing_game",
        "played a game",
        0.09,
        (30, 120),
        ("had fun", "felt competitive", "made some progress", "became mildly frustrated"),
    ),
    _ActivitySpec(
        "listening_music",
        "listened to music",
        0.11,
        (15, 90),
        ("felt relaxed", "enjoyed the quiet", "felt more settled", "felt neutral about it"),
    ),
    _ActivitySpec(
        "browsing_interests",
        "looked through something related to her interests",
        0.09,
        (10, 60),
        ("found it interesting", "became curious", "felt neutral about it"),
    ),
    _ActivitySpec(
        "relaxing",
        "relaxed for a while",
        0.12,
        (20, 120),
        ("felt relaxed", "felt more settled", "enjoyed the quiet"),
    ),
    _ActivitySpec(
        "resting",
        "rested",
        0.10,
        (30, 180),
        ("felt more settled", "felt relaxed", "felt neutral about it"),
    ),
    _ActivitySpec(
        "sleeping",
        "slept",
        0.08,
        (180, 480),
        ("felt more rested", "felt neutral about it"),
    ),
    _ActivitySpec(
        "quiet_downtime",
        "had some quiet downtime",
        0.30,
        (20, 180),
        ("enjoyed the quiet", "felt slightly bored", "felt more settled", "felt neutral about it"),
    ),
)
_SPEC_BY_TYPE = {spec.activity_type: spec for spec in _ACTIVITY_CATALOG}


def advance_life_state(state: LifeState | None, *, now: float) -> LifeState:
    """Compatibility wrapper for deterministic lazy life evolution."""

    return evolve_life_state(state, now=now).state


def evolve_life_state(
    state: LifeState | None,
    *,
    now: float,
    profile_seed: str = "local:owner",
    mood_energy: float = 0.65,
    mood_curiosity: float = 0.45,
    mood_patience: float = 0.72,
) -> LifeEvolution:
    """Advance one deterministic candidate timeline without persisting it."""

    prior = state or LifeState()
    current = max(prior.last_processed_at, _timestamp(now, prior.last_processed_at))
    if prior.last_processed_at <= 0.0:
        initialized = replace(
            prior,
            last_processed_at=current,
            next_opportunity_at=_next_opportunity(profile_seed, current, prior.activity),
            version=LIFE_STATE_VERSION,
        )
        return LifeEvolution(
            initialized,
            needs_before=prior.needs,
            needs_after=prior.needs,
            authority_source=_activity_authority(prior.activity),
            reason_no_activity="event_window_initialized",
            reason_codes=("event_window_initialized",),
        )

    elapsed = max(0.0, current - prior.last_processed_at)
    if elapsed < 2 * 60 * 60:
        completion_limit = 1
    elif elapsed <= 6 * 60 * 60:
        completion_limit = 2
    elif elapsed <= 12 * 60 * 60:
        completion_limit = 3
    else:
        completion_limit = _MAX_COMPLETED_PER_ADVANCE
    needs_before = prior.needs
    needs = _advance_needs(needs_before, elapsed)
    preferences = dict(prior.activity_preferences)
    history = list(prior.recent_events)
    known_ids = {event.event_id for event in history}
    activity = prior.activity
    completed: list[ActivityRecord] = []
    generated: list[ActivityRecord] = []
    preference_changes: dict[str, float] = {}
    reasons: list[str] = []
    finished_existing: ActivityRecord | None = None

    if activity is not None and activity.status == "active" and activity.expires_at:
        if current >= activity.expires_at:
            finished = _finish_activity(activity, activity.expires_at)
            finished_existing = finished
            activity = None
            if finished.event_id not in known_ids:
                completed.append(finished)
                generated.append(finished)
                history.append(finished)
                known_ids.add(finished.event_id)
                needs = _apply_need_effects(needs, finished.need_effects)
                _apply_preference_effects(
                    preferences,
                    preference_changes,
                    finished.preference_effects,
                )
                reasons.append("activity_completed")

    next_opportunity = prior.next_opportunity_at or _next_opportunity(
        profile_seed,
        prior.last_processed_at,
        prior.activity,
    )
    if finished_existing is not None:
        next_opportunity = _next_opportunity(
            profile_seed,
            finished_existing.completed_at,
            finished_existing,
        )
    if elapsed > 12 * 60 * 60:
        next_opportunity = max(next_opportunity, current - 12 * 60 * 60)
    last_candidates: tuple[tuple[str, float], ...] = ()
    last_penalties: tuple[tuple[str, float], ...] = ()
    last_cooldowns: tuple[str, ...] = ()
    last_bucket = int(next_opportunity // _MIN_OPPORTUNITY_DELAY)
    attempts = 0
    reason_no_activity = ""

    while (
        activity is None
        and next_opportunity <= current
        and len(completed) < completion_limit
        and attempts < _MAX_SELECTION_ATTEMPTS
    ):
        attempts += 1
        last_bucket = int(next_opportunity // _MIN_OPPORTUNITY_DELAY)
        selection = _select_activity(
            profile_seed=profile_seed,
            opportunity_at=next_opportunity,
            needs=needs,
            preferences=preferences,
            history=tuple(history),
            mood_energy=mood_energy,
            mood_curiosity=mood_curiosity,
            mood_patience=mood_patience,
        )
        spec, last_candidates, last_penalties, last_cooldowns = selection
        if spec is None:
            reason_no_activity = "quiet_opportunity_skipped"
            next_opportunity = _next_opportunity(profile_seed, next_opportunity, None)
            continue
        selected = _make_activity(profile_seed, spec, next_opportunity)
        generated.append(selected)
        if selected.expires_at <= current:
            finished = _finish_activity(selected, selected.expires_at)
            generated[-1] = finished
            if finished.event_id not in known_ids:
                completed.append(finished)
                history.append(finished)
                known_ids.add(finished.event_id)
                needs = _apply_need_effects(needs, finished.need_effects)
                _apply_preference_effects(
                    preferences,
                    preference_changes,
                    finished.preference_effects,
                )
            next_opportunity = _next_opportunity(
                profile_seed,
                finished.completed_at,
                finished,
            )
        else:
            activity = selected
            reasons.append("current_activity_selected")

    if len(completed) >= completion_limit and next_opportunity <= current:
        next_opportunity = _next_opportunity(profile_seed, current, completed[-1])
        reasons.append("bounded_activity_catch_up")
    elif attempts >= _MAX_SELECTION_ATTEMPTS and next_opportunity <= current:
        next_opportunity = _next_opportunity(profile_seed, current, None)
        reasons.append("bounded_selection_attempts")
    if not completed and activity is None:
        reasons.append("quiet_window")
        reason_no_activity = reason_no_activity or (
            "elapsed_time_too_short"
            if elapsed < _MIN_OPPORTUNITY_DELAY
            else "no_activity_selected"
        )
    if completed:
        reasons.append("offscreen_activity_completed")

    next_state = LifeState(
        activity=activity,
        recent_events=_bounded_history(history),
        needs=needs,
        activity_preferences=tuple(sorted(preferences.items())),
        last_processed_at=current,
        next_opportunity_at=max(current, next_opportunity),
        version=LIFE_STATE_VERSION,
        legacy_activity_discarded=prior.legacy_activity_discarded,
    )
    return LifeEvolution(
        state=next_state,
        new_events=tuple(completed),
        generated_activities=tuple(generated),
        previous_activity=prior.activity,
        elapsed_seconds=elapsed,
        needs_before=needs_before,
        needs_after=needs,
        need_changes=_value_changes(needs_before, needs),
        preference_changes=tuple(sorted(preference_changes.items())),
        selection_bucket=last_bucket,
        selection_candidates=last_candidates,
        repetition_penalties=last_penalties,
        cooldowns=last_cooldowns,
        reason_no_activity=reason_no_activity,
        authority_source=_activity_authority(activity),
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def begin_conversation_activity(
    evolution: LifeEvolution,
    *,
    now: float,
    profile_seed: str,
) -> LifeEvolution:
    """Make talking with Arcane current without overwriting higher authority."""

    current = _timestamp(now, evolution.state.last_processed_at)
    state = evolution.state
    activity = state.activity
    authority = _activity_authority(activity)
    if activity is not None and _AUTHORITY_RANKS[authority] > _AUTHORITY_RANKS["conversation"]:
        return replace(
            evolution,
            authority_source=authority,
            reason_codes=tuple(
                dict.fromkeys((*evolution.reason_codes, "higher_authority_activity_preserved"))
            ),
        )
    history = list(state.recent_events)
    generated = list(evolution.generated_activities)
    previous = activity or (
        generated[-1] if generated else evolution.previous_activity
    )
    if activity is not None and activity.authority == "conversation":
        conversation_reason = "conversation_activity_continued"
        conversation = replace(
            activity,
            updated_at=current,
            expires_at=current + _CONVERSATION_TTL_SECONDS,
        )
    else:
        conversation_reason = "conversation_activity_started"
        if activity is not None and activity.status == "active":
            interrupted = replace(
                activity,
                status="cancelled",
                updated_at=current,
                completed_at=current,
            )
            history.append(interrupted)
            generated = [
                interrupted if item.event_id == interrupted.event_id else item
                for item in generated
            ]
            previous = activity
        seed = f"offscreen-life:conversation:{profile_seed}:{current:.6f}"
        conversation = ActivityRecord(
            activity_type="conversation",
            source="conversation",
            description="talking with Arcane",
            started_at=current,
            updated_at=current,
            expires_at=current + _CONVERSATION_TTL_SECONDS,
            scope="profile",
            event_id=hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16],
            authority="conversation",
            need_effects=(("social", -0.12),),
        )
    next_state = replace(
        state,
        activity=conversation,
        recent_events=_bounded_history(history),
    )
    return replace(
        evolution,
        state=next_state,
        generated_activities=tuple(generated),
        previous_activity=previous,
        authority_source="conversation",
        reason_codes=tuple(
            dict.fromkeys((*evolution.reason_codes, conversation_reason))
        ),
    )


def _advance_needs(needs: LifeNeeds, elapsed_seconds: float) -> LifeNeeds:
    hours = min(_MAX_NEED_ELAPSED, max(0.0, elapsed_seconds)) / 3600.0
    return LifeNeeds(
        energy=_bounded(needs.energy - hours * 0.008),
        social=_bounded(needs.social + hours * 0.006),
        curiosity=_bounded(needs.curiosity + hours * 0.004),
        stimulation=_bounded(needs.stimulation - hours * 0.010),
    )


def _apply_need_effects(
    needs: LifeNeeds,
    effects: tuple[tuple[str, float], ...],
) -> LifeNeeds:
    values = {
        "energy": needs.energy,
        "social": needs.social,
        "curiosity": needs.curiosity,
        "stimulation": needs.stimulation,
    }
    for name, delta in effects:
        if name in values:
            values[name] = _bounded(values[name] + delta)
    return LifeNeeds(**values)


def _apply_preference_effects(
    preferences: dict[str, float],
    changes: dict[str, float],
    effects: tuple[tuple[str, float], ...],
) -> None:
    for name, delta in effects:
        previous = preferences.get(name, 0.0)
        current = _bounded(previous + delta, -0.15, 0.15)
        preferences[name] = current
        changes[name] = changes.get(name, 0.0) + current - previous


def _value_changes(before: LifeNeeds, after: LifeNeeds) -> tuple[tuple[str, float], ...]:
    return tuple(
        (name, getattr(after, name) - getattr(before, name))
        for name in ("energy", "social", "curiosity", "stimulation")
        if abs(getattr(after, name) - getattr(before, name)) >= 0.0005
    )


def _stable_unit(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _next_opportunity(
    profile_seed: str,
    after: float,
    activity: ActivityRecord | None,
) -> float:
    key = activity.event_id if activity is not None else "none"
    seed = f"offscreen-life:opportunity:{profile_seed}:{after:.6f}:{key}"
    delay = _MIN_OPPORTUNITY_DELAY + _stable_unit(seed) * (
        _MAX_OPPORTUNITY_DELAY - _MIN_OPPORTUNITY_DELAY
    )
    return after + delay


def _select_activity(
    *,
    profile_seed: str,
    opportunity_at: float,
    needs: LifeNeeds,
    preferences: dict[str, float],
    history: tuple[ActivityRecord, ...],
    mood_energy: float,
    mood_curiosity: float,
    mood_patience: float,
) -> tuple[
    _ActivitySpec | None,
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
    tuple[str, ...],
]:
    hour = time.localtime(opportunity_at).tm_hour
    recent_types = [event.activity_type for event in history[-6:] if event.status == "completed"]
    penalties: dict[str, float] = {}
    cooldowns: list[str] = []
    weighted: list[tuple[_ActivitySpec | None, float]] = [(None, 0.22)]
    for spec in _ACTIVITY_CATALOG:
        score = spec.base_weight + preferences.get(spec.activity_type, 0.0)
        score += _time_suitability(spec.activity_type, hour)
        score += _need_suitability(spec.activity_type, needs)
        score += _mood_suitability(
            spec.activity_type,
            mood_energy,
            mood_curiosity,
            mood_patience,
        )
        repetitions = recent_types.count(spec.activity_type)
        penalty = repetitions * 0.045
        if recent_types and recent_types[-1] == spec.activity_type:
            penalty += 0.12
            cooldowns.append(spec.activity_type)
        if penalty:
            penalties[spec.activity_type] = penalty
        weighted.append((spec, max(0.005, score - penalty)))
    total = sum(weight for _spec, weight in weighted)
    seed = f"offscreen-life:selection:{profile_seed}:{opportunity_at:.6f}"
    target = _stable_unit(seed) * total
    chosen: _ActivitySpec | None = None
    cursor = 0.0
    for spec, weight in weighted:
        cursor += weight
        if target <= cursor:
            chosen = spec
            break
    candidates = tuple(
        ((spec.activity_type if spec else "no_activity"), round(weight, 6))
        for spec, weight in weighted
    )
    return chosen, candidates, tuple(sorted(penalties.items())), tuple(cooldowns)


def _time_suitability(activity_type: str, hour: int) -> float:
    morning = 5 <= hour < 12
    afternoon = 12 <= hour < 17
    evening = 17 <= hour < 23
    late = not (morning or afternoon or evening)
    if morning and activity_type in {"quiet_downtime", "listening_music", "browsing_interests"}:
        return 0.08
    if afternoon and activity_type in {"playing_game", "reading_manga", "browsing_interests"}:
        return 0.07
    if evening and activity_type in {"watching_anime", "playing_game", "listening_music"}:
        return 0.09
    if late and activity_type in {"relaxing", "resting", "sleeping", "quiet_downtime"}:
        return 0.13
    if late and activity_type in {"playing_game", "watching_anime"}:
        return -0.04
    return 0.0


def _need_suitability(activity_type: str, needs: LifeNeeds) -> float:
    score = 0.0
    if activity_type in {"resting", "sleeping", "relaxing", "listening_music", "quiet_downtime"}:
        score += (1.0 - needs.energy) * 0.16
    if activity_type in {"watching_anime", "reading_manga", "browsing_interests"}:
        score += needs.curiosity * 0.11
    if activity_type in {"playing_game", "watching_anime"}:
        score += (1.0 - needs.stimulation) * 0.12
    return score


def _mood_suitability(
    activity_type: str,
    energy: float,
    curiosity: float,
    patience: float,
) -> float:
    if activity_type in {"resting", "sleeping", "relaxing"}:
        return (1.0 - energy) * 0.08
    if activity_type in {"watching_anime", "reading_manga", "browsing_interests"}:
        return curiosity * 0.05
    if activity_type == "playing_game":
        return energy * patience * 0.05
    return 0.0


def _make_activity(
    profile_seed: str,
    spec: _ActivitySpec,
    started_at: float,
) -> ActivityRecord:
    seed = f"offscreen-life:activity:{profile_seed}:{started_at:.6f}:{spec.activity_type}"
    duration_min, duration_max = spec.duration_minutes
    duration = (
        duration_min
        + _stable_unit(seed + ":duration") * (duration_max - duration_min)
    ) * 60
    reaction = spec.reactions[
        min(
            len(spec.reactions) - 1,
            int(_stable_unit(seed + ":reaction") * len(spec.reactions)),
        )
    ]
    mood_effects, need_effects, short_emotion = _activity_effects(
        spec.activity_type,
        reaction,
    )
    preference_delta = _preference_delta(spec.activity_type, reaction)
    authority = "quiet" if spec.activity_type == "quiet_downtime" else "simulated"
    return ActivityRecord(
        activity_type=spec.activity_type,
        source=_OFFSCREEN_SOURCE,
        description=spec.description,
        started_at=started_at,
        updated_at=started_at,
        expires_at=started_at + duration,
        scope="profile",
        event_id=hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16],
        reaction=reaction,
        authority=authority,
        mood_effects=mood_effects,
        need_effects=need_effects,
        preference_effects=(
            ((spec.activity_type, preference_delta),) if preference_delta else ()
        ),
        short_emotion=short_emotion,
    )


def _activity_effects(
    activity_type: str,
    reaction: str,
) -> tuple[
    tuple[tuple[str, float], ...],
    tuple[tuple[str, float], ...],
    str,
]:
    if activity_type == "watching_anime":
        return (
            (("curiosity", 0.02), ("warmth", 0.01)),
            (("curiosity", -0.06), ("stimulation", 0.10)),
            "amusement",
        )
    if activity_type == "reading_manga":
        return (("curiosity", 0.025),), (("curiosity", -0.07), ("stimulation", 0.07)), ""
    if activity_type == "playing_game":
        if reaction == "became mildly frustrated":
            return (("patience", -0.03),), (("stimulation", 0.08),), "annoyed"
        return (
            (("energy", 0.02), ("confidence", 0.02)),
            (("stimulation", 0.12), ("energy", -0.02)),
            "excitement",
        )
    if activity_type == "listening_music":
        return (("warmth", 0.01), ("patience", 0.01)), (("stimulation", 0.04),), ""
    if activity_type == "browsing_interests":
        return (("curiosity", 0.015),), (("curiosity", -0.04), ("stimulation", 0.04)), ""
    if activity_type == "relaxing":
        return (("patience", 0.015),), (("energy", 0.03),), ""
    if activity_type == "resting":
        return (("energy", 0.03), ("patience", 0.02)), (("energy", 0.10),), ""
    if activity_type == "sleeping":
        return (("energy", 0.04),), (("energy", 0.22),), ""
    if activity_type == "quiet_downtime" and reaction == "felt slightly bored":
        return (
            (("energy", -0.01), ("curiosity", 0.02)),
            (("curiosity", 0.04), ("stimulation", -0.03)),
            "",
        )
    return (), (), ""


def _preference_delta(activity_type: str, reaction: str) -> float:
    if reaction == "became mildly frustrated":
        return -0.008
    if reaction in {"felt neutral about it", "felt slightly bored"}:
        return 0.0
    return 0.005 if activity_type != "quiet_downtime" else 0.0


def _finish_activity(activity: ActivityRecord, completed_at: float) -> ActivityRecord:
    return replace(
        activity,
        status="completed",
        updated_at=completed_at,
        completed_at=completed_at,
    )


def _bounded_history(events: list[ActivityRecord]) -> tuple[ActivityRecord, ...]:
    by_id: dict[str, ActivityRecord] = {}
    anonymous: list[ActivityRecord] = []
    for event in events:
        if event.event_id:
            by_id[event.event_id] = event
        else:
            anonymous.append(event)
    combined = [*anonymous, *by_id.values()]
    combined.sort(key=lambda item: (item.completed_at or item.updated_at, item.event_id))
    return tuple(combined[-_MAX_RECENT_ACTIVITIES:])


def _activity_authority(activity: ActivityRecord | None) -> str:
    return activity.authority if activity is not None and activity.authority else "none"


def record_grounded_activity(
    state: LifeState | None,
    *,
    activity_type: str,
    source: str,
    description: str,
    now: float,
    scope: str = "profile",
    status: str = "active",
    confidence: float = 1.0,
    ttl_seconds: float = _DEFAULT_ACTIVITY_TTL_SECONDS,
    subject: str = "",
    reaction: str = "",
    authority: str = "",
) -> LifeState:
    """Record an explicit or verified activity without losing stronger authority."""

    source_name = compact_text(source, 48).lower()
    detail = compact_text(description, 240)
    kind = compact_text(activity_type, 48).lower()
    state_name = compact_text(status, 24).lower() or "active"
    if not source_name or source_name in _UNTRUSTED_ACTIVITY_SOURCES:
        raise ValueError("Activity source must identify a grounded subsystem event.")
    if not kind or not detail:
        raise ValueError("Grounded activity requires a type and description.")
    if state_name not in _ACTIVITY_STATUSES:
        raise ValueError("Grounded activity has an unsupported status.")
    confidence_value = _timestamp(confidence)
    if confidence_value < 0.5:
        raise ValueError("Grounded activity confidence must be at least 0.5.")
    current_state = state or LifeState()
    current = _timestamp(now)
    ttl = max(60.0, _timestamp(ttl_seconds, _DEFAULT_ACTIVITY_TTL_SECONDS))
    authority_name = _authority_for(source_name, kind, authority)
    existing_authority = _activity_authority(current_state.activity)
    if (
        current_state.activity is not None
        and current_state.activity.status == "active"
        and _AUTHORITY_RANKS[existing_authority] > _AUTHORITY_RANKS[authority_name]
    ):
        return current_state
    history = list(current_state.recent_events)
    existing = current_state.activity
    if (
        state_name == "active"
        and existing is not None
        and existing.status == "active"
        and existing.source == source_name
        and existing.activity_type == kind
        and existing.description == detail
    ):
        refreshed = replace(
            existing,
            updated_at=current,
            expires_at=current + ttl,
            subject=compact_text(subject, 120) or existing.subject,
            reaction=compact_text(reaction, 120) or existing.reaction,
            authority=authority_name,
        )
        return replace(
            current_state,
            activity=refreshed,
            last_processed_at=max(current_state.last_processed_at, current),
            next_opportunity_at=max(
                current_state.next_opportunity_at,
                current + ttl,
            ),
            version=LIFE_STATE_VERSION,
        )
    if existing is not None and existing.status == "active":
        history.append(
            replace(
                existing,
                status="cancelled",
                updated_at=current,
                completed_at=current,
            )
        )
    seed = f"grounded-activity:{source_name}:{kind}:{current:.6f}:{detail}"
    activity = ActivityRecord(
        activity_type=kind,
        source=source_name,
        description=detail,
        started_at=current,
        updated_at=current,
        status=state_name,
        confidence=min(1.0, confidence_value),
        expires_at=current + ttl,
        completed_at=current if state_name != "active" else 0.0,
        scope=compact_text(scope, 120) or "profile",
        event_id=hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16],
        subject=compact_text(subject, 120),
        reaction=compact_text(reaction, 120),
        authority=authority_name,
    )
    current_activity = activity
    if state_name != "active":
        history.append(activity)
        current_activity = None
    return replace(
        current_state,
        activity=current_activity,
        recent_events=_bounded_history(history),
        last_processed_at=max(current_state.last_processed_at, current),
        next_opportunity_at=max(
            current_state.next_opportunity_at,
            current + (ttl if state_name == "active" else _MIN_OPPORTUNITY_DELAY),
        ),
        version=LIFE_STATE_VERSION,
    )


def grounded_activity(
    state: LifeState,
    *,
    now: float,
    scope: str,
) -> ActivityRecord | None:
    activity = state.activity
    return activity if activity is not None and activity.is_active_for(now, scope) else None


def recent_activity(
    state: LifeState,
    *,
    now: float,
    scope: str,
) -> ActivityRecord | None:
    current = _timestamp(now)
    return max(
        (
            event
            for event in state.recent_events
            if event.status == "completed"
            and 0.0 <= current - (event.completed_at or event.updated_at) <= 72 * 60 * 60
            and event.scope in {"profile", compact_text(scope, 120)}
        ),
        key=lambda event: (event.completed_at or event.updated_at, event.event_id),
        default=None,
    )


_CURRENT_QUERY = re.compile(
    r"\b(?:what are you doing|what are you up to|doing right now|doing currently)\b",
    re.IGNORECASE,
)
_EARLIER_QUERY = re.compile(
    r"\b(?:what did you do|what have you been doing|been up to|how was your day|how's life)\b",
    re.IGNORECASE,
)
_CATEGORY_QUERIES = {
    "watching_anime": re.compile(r"\banime\b", re.IGNORECASE),
    "reading_manga": re.compile(r"\bmanga\b", re.IGNORECASE),
    "playing_game": re.compile(r"\b(?:game|gaming)\b", re.IGNORECASE),
    "listening_music": re.compile(r"\b(?:music|song)\b", re.IGNORECASE),
}


def format_life_state(
    state: LifeState,
    *,
    now: float = 0.0,
    scope: str = "profile",
    query: str = "",
) -> str:
    """Render only recorded life facts relevant to the current message."""

    text = compact_text(query, 500)
    current = grounded_activity(state, now=now, scope=scope)
    recent = [
        event
        for event in reversed(state.recent_events)
        if event.status == "completed"
        and event.activity_type != "conversation"
        and event.scope in {"profile", compact_text(scope, 120)}
        and 0.0 <= _timestamp(now) - (event.completed_at or event.updated_at) <= 72 * 60 * 60
    ]
    if not text:
        activity = current or (recent[0] if recent else None)
        return (
            f"Known recent activity: {_activity_fact(activity, include_reaction=True)}."
            if activity
            else ""
        )
    if _CURRENT_QUERY.search(text):
        parts = [f"Current activity: {_activity_fact(current)}."] if current else []
        if recent:
            parts.append(f"Before that: {_activity_fact(recent[0], include_reaction=True)}.")
        return " ".join(parts)
    if _EARLIER_QUERY.search(text):
        earlier = recent[0] if recent else (
            current if current is not None and current.activity_type != "conversation" else None
        )
        return (
            f"Recent activity: {_activity_fact(earlier, include_reaction=True)}."
            if earlier
            else f"Current activity: {_activity_fact(current)}."
            if current
            else ""
        )
    for activity_type, pattern in _CATEGORY_QUERIES.items():
        if pattern.search(text):
            match = next(
                (event for event in recent if event.activity_type == activity_type),
                None,
            )
            return (
                f"Relevant recent activity: {_activity_fact(match, include_reaction=True)}."
                if match
                else ""
            )
    return ""


def _activity_fact(
    activity: ActivityRecord | None,
    *,
    include_reaction: bool = False,
) -> str:
    if activity is None:
        return "none"
    description = activity.description
    if activity.subject and activity.subject.casefold() not in description.casefold():
        description = f"{description}: {activity.subject}"
    if include_reaction and activity.reaction:
        description += f" and {activity.reaction}"
    return description


def life_evolution_debug(evolution: LifeEvolution | None) -> dict[str, object]:
    if evolution is None:
        return {}
    state = evolution.state
    return {
        "elapsed_seconds": evolution.elapsed_seconds,
        "previous_activity": _debug_activity(evolution.previous_activity),
        "generated_activities": tuple(
            _debug_activity(activity) for activity in evolution.generated_activities
        ),
        "current_activity": _debug_activity(state.activity),
        "completed_count": len(evolution.new_events),
        "recent_reaction": (
            evolution.new_events[-1].reaction if evolution.new_events else ""
        ),
        "mood_effects": tuple(
            effect for event in evolution.new_events for effect in event.mood_effects
        ),
        "need_changes": evolution.need_changes,
        "needs_before": {
            name: getattr(evolution.needs_before, name)
            for name in ("energy", "social", "curiosity", "stimulation")
        },
        "needs_after": {
            name: getattr(evolution.needs_after, name)
            for name in ("energy", "social", "curiosity", "stimulation")
        },
        "preference_changes": evolution.preference_changes,
        "selection_bucket": evolution.selection_bucket,
        "selection_candidates": evolution.selection_candidates,
        "repetition_penalties": evolution.repetition_penalties,
        "cooldowns": evolution.cooldowns,
        "reason_no_activity": evolution.reason_no_activity,
        "authority_source": evolution.authority_source,
        "last_processed_at": state.last_processed_at,
        "next_opportunity_at": state.next_opportunity_at,
        "reason_codes": evolution.reason_codes,
    }


def _debug_activity(activity: ActivityRecord | None) -> dict[str, object]:
    if activity is None:
        return {}
    return {
        "event_id": activity.event_id,
        "activity_type": activity.activity_type,
        "status": activity.status,
        "description": activity.description,
        "source": activity.source,
        "authority": activity.authority,
        "started_at": activity.started_at,
        "expires_at": activity.expires_at,
        "completed_at": activity.completed_at,
        "reaction": activity.reaction,
    }
