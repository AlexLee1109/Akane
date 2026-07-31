"""The single event-driven autonomous-presence worker for this process."""

from __future__ import annotations

import inspect
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from app.core.config import MAX_TOKENS
from app.core.memory import (
    InitiativeOpportunity,
    StateStore,
    format_emotional_context,
    get_state_store,
)
from app.core.model_loader import (
    InferenceCancelled,
    InferenceTiming,
    ModelManager,
    compile_json_grammar,
)
from app.core.presence import (
    LIFE_JSON_SCHEMA,
    LifeParseError,
    format_presence_context,
    parse_life_decision,
)
from app.core.prompt import (
    PromptContext,
    build_initiative_prompt,
    build_life_prompt,
)
from app.core.time_context import build_time_context, format_time_context
from app.core.utils import OWNER_PROFILE_ID, canonical_profile_id, compact_text

_MIN_ERROR_BACKOFF_SECONDS = 5.0
_MAX_ERROR_BACKOFF_SECONDS = 60.0
_LOGGER = logging.getLogger(__name__)


class AutonomousLifeWorker:
    def __init__(
        self,
        store: StateStore,
        *,
        runner: Callable[..., bool] | None = None,
    ) -> None:
        self._store = store
        self._runner = runner
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._delivery_condition = threading.Condition(threading.RLock())
        self._adapters: dict[str, tuple[str, float]] = {}
        self._thread: threading.Thread | None = None
        self._pending_profiles: tuple[str, ...] = ()
        self._active_profile = ""
        self._consecutive_errors = 0
        self._last_error = ""
        self._next_retry_at = 0.0

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._store.set_autonomy_wake(self.wake)
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="AkaneAutonomousLife",
            )
            self._thread.start()
        self.wake()
        return True

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._delivery_condition:
            self._adapters.clear()
            self._delivery_condition.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._store.set_autonomy_wake(None)

    def wake(self, _profile_id: str = "") -> None:
        self._wake.set()
        with self._delivery_condition:
            self._delivery_condition.notify_all()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return {
                "Life Worker Started": running,
                "Pending Profiles": self._pending_profiles,
                "Active Profile": self._active_profile,
                "Last Error": self._last_error,
                "Next Retry At": self._next_retry_at,
                "Available Initiative Adapters": tuple(self._available_adapters()),
            }

    def _available_adapters(self, *, now: float | None = None) -> tuple[str, ...]:
        current = time.time() if now is None else now
        with self._delivery_condition:
            expired = tuple(
                name
                for name, (_conversation, expires) in self._adapters.items()
                if expires <= current
            )
            for name in expired:
                self._adapters.pop(name, None)
            return tuple(self._adapters)

    def claim_delivery(
        self,
        *,
        adapter: str,
        conversation_id: str,
        available: bool,
        wait_seconds: float,
    ) -> dict[str, object] | None:
        channel = compact_text(adapter, 16).casefold()
        if channel not in {"popup", "discord"}:
            return None
        conversation = compact_text(conversation_id, 160) or "popup:default"
        deadline = time.monotonic() + max(0.0, min(30.0, wait_seconds))
        with self._delivery_condition:
            if not available:
                self._adapters.pop(channel, None)
                self._store.release_initiative_delivery(
                    adapter=channel,
                    now=time.time(),
                )
                self._delivery_condition.notify_all()
                return None
            self._adapters[channel] = (conversation, time.time() + 35.0)
            self.wake()
            while not self._stop.is_set():
                current = time.time()
                claimed = self._store.claim_initiative_delivery(
                    adapter=channel,
                    available_adapters=self._available_adapters(now=current),
                    now=current,
                )
                if claimed is not None:
                    return {
                        "opportunity_id": claimed.opportunity_id,
                        "claim_token": claimed.claim_token or "",
                        "message": claimed.message or "",
                        "topic_key": claimed.topic_key,
                    }
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._delivery_condition.wait(remaining)
        return None

    def acknowledge_delivery(
        self,
        *,
        opportunity_id: str,
        claim_token: str,
        adapter: str,
        conversation_id: str,
        success: bool,
        message_id: str = "",
    ) -> bool:
        accepted = self._store.acknowledge_initiative_delivery(
            opportunity_id=opportunity_id,
            claim_token=claim_token,
            adapter=adapter,
            conversation_id=conversation_id,
            success=success,
            message_id=message_id,
            now=time.time(),
        )
        if not success:
            with self._delivery_condition:
                self._adapters.pop(compact_text(adapter, 16).casefold(), None)
                self._delivery_condition.notify_all()
        return accepted

    def _runner_kwargs(self, runner: Callable[..., bool]) -> dict[str, object]:
        try:
            parameters = inspect.signature(runner).parameters
        except (TypeError, ValueError):
            return {}
        if "cancellation" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return {"cancellation": self._stop}
        for name in ("cancel_event", "stop_event"):
            if name in parameters:
                return {name: self._stop}
        return {}

    def _run_profile(self, profile_id: str) -> bool:
        runner = self._runner
        if runner is None:
            runner = run_life_turn
        return runner(
            profile_id=profile_id,
            **self._runner_kwargs(runner),
        )

    def _record_error(self, exc: Exception) -> None:
        with self._lock:
            self._consecutive_errors += 1
            delay = min(
                _MAX_ERROR_BACKOFF_SECONDS,
                _MIN_ERROR_BACKOFF_SECONDS
                * (2 ** min(self._consecutive_errors - 1, 8)),
            )
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._next_retry_at = time.time() + delay

    def _clear_error(self) -> None:
        with self._lock:
            self._consecutive_errors = 0
            self._last_error = ""
            self._next_retry_at = 0.0

    def _wait_for_local_retry(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                timeout = self._next_retry_at - time.time()
            if timeout <= 0.0:
                return
            self._wake.clear()
            if self._stop.is_set():
                return
            self._wake.wait(timeout)

    def _schedule(self) -> tuple[tuple[str, ...], bool, float | None]:
        current = time.time()
        due, next_presence_at = self._store.presence_schedule(now=current)
        initiative_due, next_initiative_at = self._store.initiative_schedule(
            now=current,
        )
        with self._lock:
            self._pending_profiles = due
        next_due_at = min(
            (
                value
                for value in (next_presence_at, next_initiative_at)
                if value is not None
            ),
            default=None,
        )
        return due, initiative_due, next_due_at

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wait_for_local_retry()
            if self._stop.is_set():
                break
            try:
                due_profiles, initiative_due, next_due_at = self._schedule()
            except Exception as exc:
                self._record_error(exc)
                continue
            if due_profiles:
                for profile_id in due_profiles:
                    if self._stop.is_set():
                        break
                    with self._lock:
                        self._active_profile = profile_id
                    try:
                        self._run_profile(profile_id)
                    except Exception as exc:
                        self._record_error(exc)
                        break
                    else:
                        self._clear_error()
                    finally:
                        with self._lock:
                            self._active_profile = ""
                continue
            if initiative_due:
                with self._lock:
                    self._active_profile = "initiative"
                try:
                    run_initiative_evaluation(cancellation=self._stop)
                except Exception as exc:
                    self._record_error(exc)
                else:
                    self._clear_error()
                finally:
                    with self._lock:
                        self._active_profile = ""
                continue

            self._wake.clear()
            try:
                due_profiles, initiative_due, next_due_at = self._schedule()
            except Exception as exc:
                self._record_error(exc)
                continue
            if due_profiles or initiative_due:
                continue
            self._clear_error()
            if self._stop.is_set():
                break
            timeout = (
                None
                if next_due_at is None
                else max(0.0, next_due_at - time.time())
            )
            self._wake.wait(timeout)
            with self._delivery_condition:
                self._delivery_condition.notify_all()


def _profile_text(item: object) -> str:
    for name in ("content", "text", "summary"):
        value = compact_text(getattr(item, name, ""), 320)
        if value:
            return value
    return compact_text(item, 320)


def _life_context(
    profile,
    *,
    now: float,
    last_user_message_at: float,
    last_akane_message_at: float,
) -> PromptContext:
    memories = tuple(
        _profile_text(item)
        for item in sorted(
            profile.memories,
            key=lambda item: (item.confidence, item.updated_at),
            reverse=True,
        )[:4]
        if _profile_text(item)
    )
    relationship = tuple(
        _profile_text(item)
        for item in (
            *profile.relationship.unresolved_events[-2:],
            *profile.relationship.shared_context[-1:],
            *profile.relationship.patterns[-1:],
        )
        if _profile_text(item)
    )
    preferences = tuple(
        _profile_text(item) for item in profile.preferences[-3:] if _profile_text(item)
    )
    opinions = tuple(
        _profile_text(item) for item in profile.opinions[-3:] if _profile_text(item)
    )
    current_activity = profile.presence.current_activity
    return PromptContext(
        time_context=format_time_context(
            build_time_context(
                now=now,
                last_user_message_at=last_user_message_at,
                last_akane_message_at=last_akane_message_at,
                current_activity_started_at=(
                    current_activity.started_at if current_activity else None
                ),
            )
        ),
        memories=memories,
        relationship=relationship,
        preferences=preferences,
        interests=tuple(profile.interests[-8:]),
        opinions=opinions,
        emotion=format_emotional_context(
            profile,
            now=now,
            include_unappraised=True,
        ),
        presence=format_presence_context(
            profile.presence,
            now=now,
            continuity="ongoing" if profile.presence.current_activity else "none",
            include_previous=True,
        ),
    )


def run_life_turn(
    *,
    profile_id: str,
    now: float | None = None,
    cancellation: threading.Event | None = None,
) -> bool:
    """Run one claimed background decision without generating visible dialogue."""

    current = time.time() if now is None else max(0.0, float(now))
    profile_key = canonical_profile_id(profile_id)
    store = get_state_store()
    claimed = store.claim_presence_decision(profile_key, now=current)
    if claimed is None:
        return False
    claim_token = claimed.presence.claim_token or ""
    if not claim_token:
        return False
    prior_error = compact_text(claimed.presence.last_error, 120).casefold()
    correcting_structure = prior_error.startswith(
        ("invalid life block", "life parse:")
    )
    retry_note = (
        "Return one valid JSON object matching the required life-decision schema. "
        "Do not include explanation, Markdown, dialogue, or wrapper text."
        if correcting_structure
        else "The corrected proposal must explicitly appraise emotion with mode keep, "
        "shift, or settle."
        if "appraisal" in prior_error
        else "The previous proposal was too similar. Choose independently without "
        "reusing that activity."
        if "repeat" in prior_error
        else ""
    )
    try:
        manager = ModelManager.get_instance()
        timing = InferenceTiming(requested_at=time.perf_counter())
        with manager.reserve(
            priority="background",
            cancellation=cancellation,
        ) as reservation:
            prompt_now = current if now is not None else time.time()
            message_times = store.snapshot(
                profile_key,
                now=prompt_now,
                include_memory=False,
            )
            context = _life_context(
                claimed,
                now=prompt_now,
                last_user_message_at=message_times.last_profile_user_at,
                last_akane_message_at=message_times.last_profile_assistant_at,
            )
            plan = build_life_prompt(
                context,
                retry_note=retry_note,
                token_counter=lambda messages: manager.tokenize_prompt(
                    messages,
                    reservation=reservation,
                ),
            )
            raw = "".join(
                manager.stream(
                    prompt_tokens=plan.token_ids,
                    template_stop_sequences=plan.stop_sequences,
                    max_tokens=MAX_TOKENS,
                    cancellation=cancellation,
                    timing=timing,
                    grammar=compile_json_grammar(LIFE_JSON_SCHEMA),
                    reservation=reservation,
                )
            )
        grounded = "\n".join(
            message["content"]
            for message in plan.messages
        )
        try:
            decision = parse_life_decision(raw)
        except LifeParseError as exc:
            _LOGGER.warning(
                "life parse failed format=raw_json_schema chars=%d markers=%s/%s "
                "keys=%s preview=%r error=%s",
                len(raw),
                "<AKANE_LIFE>" in raw,
                "</AKANE_LIFE>" in raw,
                getattr(exc, "decoded_keys", ()),
                raw[:1500],
                exc,
            )
            store.fail_presence_decision(
                profile_key,
                claim_token=claim_token,
                now=current,
                error=f"life parse: {exc}",
            )
            return False
        accepted, _reason = store.commit_presence_decision(
            profile_key,
            decision,
            claim_token=claim_token,
            now=time.time() if now is None else current,
            grounded_context=grounded,
            expected_emotion_updated_at=claimed.emotion.updated_at,
        )
        if accepted:
            after = store.snapshot(
                profile_key,
                now=time.time() if now is None else current,
                include_memory=False,
            )
            offer_initiative_from_change(
                store,
                claimed,
                after.profile,
                now=after.now,
                conversation=False,
            )
        return accepted
    except InferenceCancelled:
        store.fail_presence_decision(
            profile_key,
            claim_token=claim_token,
            now=current,
            error="life inference cancelled",
        )
        return False
    except Exception:
        store.fail_presence_decision(
            profile_key,
            claim_token=claim_token,
            now=time.time() if now is None else current,
            error="life inference failed",
        )
        raise


def _topic_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()[:120]


def offer_initiative_from_change(
    store: StateStore,
    before,
    after,
    *,
    now: float,
    conversation: bool,
) -> bool:
    """Persist one grounded opportunity only when durable source state changed."""

    source: tuple[str, str, str, str] | None = None
    if conversation:
        previous_memory_ids = {item.id for item in before.memories}
        memory = next(
            (
                item
                for item in reversed(after.memories)
                if item.id not in previous_memory_ids
                and item.kind in {"commitment", "project", "concern"}
                and item.confidence >= 0.75
            ),
            None,
        )
        if memory is not None:
            source = (
                "unresolved grounded memory",
                "memory",
                memory.id,
                memory.text,
            )
        if source is None:
            previous_relationship = {
                (item.summary, item.updated_at)
                for item in before.relationship.unresolved_events
            }
            event = next(
                (
                    item
                    for item in reversed(after.relationship.unresolved_events)
                    if (item.summary, item.updated_at) not in previous_relationship
                ),
                None,
            )
            if event is not None:
                source = (
                    "meaningful unresolved relationship context",
                    "relationship",
                    f"relationship:{event.updated_at:.6f}:"
                    f"{_topic_key(event.summary)[:60]}",
                    event.summary,
                )
        if source is None:
            previous_opinions = {
                (item.topic, item.position, item.updated_at)
                for item in before.opinions
            }
            opinion = next(
                (
                    item
                    for item in reversed(after.opinions)
                    if (item.topic, item.position, item.updated_at)
                    not in previous_opinions
                ),
                None,
            )
            if opinion is not None:
                source = (
                    "a personally meaningful conclusion Akane reached",
                    "realization",
                    f"realization:{opinion.updated_at:.6f}:"
                    f"{_topic_key(opinion.topic)[:60]}",
                    opinion.content,
                )
    else:
        activity = after.presence.current_activity
        prior = before.presence.current_activity
        if (
            activity is not None
            and activity.detail
            and (prior is None or prior.started_at != activity.started_at)
        ):
            source = (
                "meaningful current offscreen activity",
                "offscreen_life",
                f"offscreen_life:{activity.started_at:.6f}",
                activity.fact(),
            )
    if source is None:
        return False
    reason, source_type, source_id, evidence = source
    delay = 15.0 * 60.0 if conversation else 0.0
    lifetime = 7.0 * 24.0 * 3600.0 if conversation else 12.0 * 3600.0
    return store.offer_initiative(
        InitiativeOpportunity(
            uuid4().hex,
            reason,
            source_type,
            source_id,
            evidence,
            _topic_key(evidence),
            now,
            now + delay,
            now + lifetime,
        ),
        now=now,
    )


@dataclass(frozen=True, slots=True)
class InitiativeDecision:
    decision: str
    topic: str | None
    message: str | None


def parse_initiative_decision(
    output: object,
    *,
    evidence: str,
) -> InitiativeDecision | None:
    matches = re.findall(
        r"<AKANE_INITIATIVE>\s*(.*?)\s*</AKANE_INITIATIVE>",
        str(output or ""),
        re.DOTALL | re.IGNORECASE,
    )
    if len(matches) != 1:
        return None
    try:
        payload = json.loads(matches[0])
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "decision",
        "topic",
        "message",
        "reason",
    }:
        return None
    decision = compact_text(payload.get("decision"), 16).casefold()
    reason = compact_text(payload.get("reason"), 180)
    if decision == "quiet":
        if payload.get("topic") is not None or payload.get("message") is not None:
            return None
        return InitiativeDecision("quiet", None, None) if reason else None
    raw_topic = payload.get("topic")
    raw_message = payload.get("message")
    if decision != "speak" or not isinstance(raw_topic, str) or not isinstance(
        raw_message,
        str,
    ):
        return None
    topic = compact_text(raw_topic, 120)
    message = compact_text(raw_message, 501)
    evidence_terms = set(_topic_key(evidence).split())
    grounded = set(_topic_key(f"{topic} {reason}").split()) & evidence_terms
    message_grounded = set(_topic_key(message).split()) & evidence_terms
    if (
        not topic
        or not message
        or len(message) > 500
        or not reason
        or not grounded
        or not message_grounded
        or any(value in message for value in ("@everyone", "@here", "<@"))
    ):
        return None
    return InitiativeDecision("speak", topic, message)


def _initiative_context(snapshot, opportunity, *, now: float) -> PromptContext:
    profile = snapshot.profile
    activity = profile.presence.current_activity
    recent_topics = ", ".join(
        item.topic_key for item in profile.initiative.recent[-4:]
    )
    opportunity_text = "\n".join(
        (
            f"Grounded reason: {opportunity.reason}.",
            f"Source type: {opportunity.source_type}.",
            f"Source evidence: {opportunity.context}.",
            (
                f"Recent sent initiative topics: {recent_topics}."
                if recent_topics
                else ""
            ),
        )
    )
    return PromptContext(
        time_context=format_time_context(
            build_time_context(
                now=now,
                last_user_message_at=snapshot.last_profile_user_at,
                last_akane_message_at=snapshot.last_profile_assistant_at,
                current_activity_started_at=(
                    activity.started_at if activity else None
                ),
            )
        ),
        emotion=format_emotional_context(profile, now=now),
        presence=(
            format_presence_context(
                profile.presence,
                now=now,
                continuity="ongoing",
            )
            if activity
            else "Current activity: none recorded."
        ),
        initiative_opportunity=opportunity_text,
    )


def run_initiative_evaluation(
    *,
    now: float | None = None,
    cancellation: threading.Event | None = None,
) -> bool:
    """Evaluate one persisted opportunity without choosing a delivery adapter."""

    current = time.time() if now is None else max(0.0, float(now))
    store = get_state_store()
    opportunity = None
    try:
        manager = ModelManager.get_instance()
        timing = InferenceTiming(requested_at=time.perf_counter())
        with manager.reserve(
            priority="background",
            cancellation=cancellation,
        ) as reservation:
            prompt_now = current if now is not None else time.time()
            opportunity = store.claim_initiative_evaluation(now=prompt_now)
            if opportunity is None or not opportunity.claim_token:
                return False
            snapshot = store.snapshot(
                OWNER_PROFILE_ID,
                now=prompt_now,
                include_memory=False,
            )
            plan = build_initiative_prompt(
                _initiative_context(snapshot, opportunity, now=prompt_now),
                token_counter=lambda messages: manager.tokenize_prompt(
                    messages,
                    reservation=reservation,
                ),
            )
            raw = "".join(
                manager.stream(
                    prompt_tokens=plan.token_ids,
                    template_stop_sequences=plan.stop_sequences,
                    max_tokens=MAX_TOKENS,
                    cancellation=cancellation,
                    timing=timing,
                    reservation=reservation,
                )
            ).strip()
        decision = parse_initiative_decision(
            raw,
            evidence=opportunity.context,
        )
        if decision is None:
            store.fail_initiative_evaluation(
                claim_token=opportunity.claim_token,
                now=current,
            )
            return False
        completed = store.complete_initiative_evaluation(
            claim_token=opportunity.claim_token,
            decision=decision.decision,
            topic=decision.topic,
            message=decision.message,
            now=time.time() if now is None else current,
        )
        return bool(completed and completed.status == "pending_delivery")
    except InferenceCancelled:
        if opportunity is not None and opportunity.claim_token:
            store.fail_initiative_evaluation(
                claim_token=opportunity.claim_token,
                now=current,
            )
        return False
    except Exception:
        if opportunity is not None and opportunity.claim_token:
            store.fail_initiative_evaluation(
                claim_token=opportunity.claim_token,
                now=time.time() if now is None else current,
            )
        raise


_WORKER_LOCK = threading.Lock()
_WORKER: AutonomousLifeWorker | None = None


def start_life_worker() -> AutonomousLifeWorker:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = AutonomousLifeWorker(get_state_store())
        _WORKER.start()
        return _WORKER


def stop_life_worker() -> None:
    with _WORKER_LOCK:
        worker = _WORKER
    if worker is not None:
        worker.stop()


def claim_initiative_delivery(
    *,
    adapter: str,
    conversation_id: str,
    available: bool,
    wait_seconds: float = 25.0,
) -> dict[str, object] | None:
    with _WORKER_LOCK:
        worker = _WORKER
    if worker is None:
        return None
    return worker.claim_delivery(
        adapter=adapter,
        conversation_id=conversation_id,
        available=available,
        wait_seconds=wait_seconds,
    )


def acknowledge_initiative_delivery(
    *,
    opportunity_id: str,
    claim_token: str,
    adapter: str,
    conversation_id: str,
    success: bool,
    message_id: str = "",
) -> bool:
    with _WORKER_LOCK:
        worker = _WORKER
    if worker is None:
        return False
    return worker.acknowledge_delivery(
        opportunity_id=opportunity_id,
        claim_token=claim_token,
        adapter=adapter,
        conversation_id=conversation_id,
        success=success,
        message_id=message_id,
    )


def life_worker_debug() -> dict[str, object]:
    with _WORKER_LOCK:
        worker = _WORKER
    if worker is None:
        return {
            "Life Worker Started": False,
            "Pending Profiles": (),
            "Active Profile": "",
            "Last Error": "",
            "Next Retry At": 0.0,
        }
    return worker.snapshot()
