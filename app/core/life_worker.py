"""The single event-driven offscreen-presence worker for this process."""

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
    effective_emotion,
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
    BOOTSTRAP_PRESENCE_JSON_SCHEMA,
    PRESENCE_JSON_SCHEMA,
    UNSUPPORTED_PHYSICAL_ACTIVITY_REASON,
    PresenceParseError,
    PresenceProposal,
    ProposedActivity,
    format_presence_context,
    needs_bootstrap,
    presence_activity_rejection,
    parse_presence_proposal,
)
from app.core.prompt import (
    PromptContext,
    build_initiative_prompt,
    build_presence_prompt,
)
from app.core.time_context import build_time_context, format_time_context
from app.core.utils import OWNER_PROFILE_ID, canonical_profile_id, compact_text

_MIN_ERROR_BACKOFF_SECONDS = 5.0
_MAX_ERROR_BACKOFF_SECONDS = 60.0
_LOGGER = logging.getLogger(__name__)
_BOOTSTRAP_FALLBACK = PresenceProposal(
    "new",
    ProposedActivity(
        "spending some quiet time with one of her interests",
        "thinking about what currently holds her attention",
    ),
    None,
    None,
)


class OffscreenPresenceWorker:
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
                name="AkaneOffscreenPresence",
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
                "Presence Worker Started": running,
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
            runner = run_presence_turn
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
        value = compact_text(getattr(item, name, ""), 240)
        if value:
            return value
    return compact_text(item, 240)


def _presence_context(profile, *, now: float) -> PromptContext:
    activity = profile.presence.current_activity
    emotion = effective_emotion(profile.emotion, now=now)
    emotion_context = (
        "Current immediate emotion: neutral."
        if emotion.primary == "neutral"
        else "\n".join(
            (
                f"Current immediate emotion: {emotion.primary} "
                f"at intensity {emotion.intensity:.2f}.",
                f"Cause: {emotion.cause}.",
            )
        )
    )
    return PromptContext(
        time_context=format_time_context(
            build_time_context(
                now=now,
                current_activity_started_at=(
                    activity.started_at if activity else None
                ),
            )
        ),
        preferences=tuple(
            text
            for item in profile.preferences[-3:]
            if (text := _profile_text(item))
        ),
        interests=tuple(profile.interests[-8:]),
        emotion=emotion_context,
        presence=format_presence_context(profile.presence),
        continuation_count=profile.presence.continuation_count,
    )


def _log_bootstrap_failure(raw: str, error: PresenceParseError | str) -> None:
    decoded = error.decoded if isinstance(error, PresenceParseError) else None
    if decoded is None:
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            decoded = None
    decoded_keys = tuple(sorted(decoded)[:32]) if isinstance(decoded, dict) else ()
    decision = decoded.get("decision") if isinstance(decoded, dict) else None
    activity = decoded.get("activity") if isinstance(decoded, dict) else None
    activity_keys = tuple(sorted(activity)[:32]) if isinstance(activity, dict) else ()
    activity_value: object = activity
    if len(repr(activity_value)) > 1_000:
        activity_value = repr(activity_value)[:1_000]
    _LOGGER.warning(
        "presence bootstrap failed entered=%s raw_length=%d raw=%r "
        "decoded_keys=%r decision=%r activity=%r activity_keys=%r reason=%s",
        True,
        len(raw),
        raw[:1_000],
        decoded_keys,
        decision,
        activity_value,
        activity_keys,
        error,
    )


def run_presence_turn(
    *,
    profile_id: str,
    now: float | None = None,
    cancellation: threading.Event | None = None,
) -> bool:
    """Run one claimed raw-JSON presence decision without visible dialogue."""

    current = time.time() if now is None else max(0.0, float(now))
    profile_key = canonical_profile_id(profile_id)
    store = get_state_store()
    claimed = store.claim_presence_decision(profile_key, now=current)
    if claimed is None:
        return False
    claim_token = claimed.presence.claim_token or ""
    if not claim_token:
        return False
    expected_activity_id = (
        claimed.presence.current_activity.activity_id
        if claimed.presence.current_activity is not None
        else None
    )
    bootstrap = needs_bootstrap(claimed.presence)
    try:
        manager = ModelManager.get_instance()
        timing = InferenceTiming(requested_at=time.perf_counter())
        with manager.reserve(
            priority="background",
            cancellation=cancellation,
        ) as reservation:
            prompt_now = current if now is not None else time.time()
            context = _presence_context(
                claimed,
                now=prompt_now,
            )
            correction_reason = ""
            proposal = None
            raw = ""
            for attempt in range(2):
                plan = build_presence_prompt(
                    context,
                    token_counter=lambda messages: manager.tokenize_prompt(
                        messages,
                        reservation=reservation,
                    ),
                    bootstrap=bootstrap,
                    correction_reason=correction_reason,
                )
                schema = (
                    BOOTSTRAP_PRESENCE_JSON_SCHEMA
                    if bootstrap
                    else PRESENCE_JSON_SCHEMA
                )
                raw = "".join(
                    manager.stream(
                        prompt_tokens=plan.token_ids,
                        template_stop_sequences=plan.stop_sequences,
                        max_tokens=MAX_TOKENS,
                        cancellation=cancellation,
                        timing=timing,
                        grammar=compile_json_grammar(schema),
                        reservation=reservation,
                    )
                )
                try:
                    proposal = parse_presence_proposal(raw, bootstrap=bootstrap)
                except PresenceParseError as exc:
                    if bootstrap:
                        _log_bootstrap_failure(raw, exc)
                        correction_reason = str(exc)
                        if attempt == 0:
                            continue
                        proposal = _BOOTSTRAP_FALLBACK
                        break
                    _LOGGER.warning(
                        "presence parse failed format=raw_json_schema chars=%d "
                        "preview=%r error=%s",
                        len(raw),
                        raw[:1_000],
                        exc,
                    )
                    store.fail_presence_decision(
                        profile_key,
                        claim_token=claim_token,
                        now=current,
                        error=str(exc),
                    )
                    return False
                activity_error = presence_activity_rejection(
                    proposal.activity if proposal.decision == "new" else None
                )
                if activity_error:
                    if bootstrap:
                        _log_bootstrap_failure(raw, activity_error)
                    else:
                        _LOGGER.warning(
                            "presence activity rejected chars=%d preview=%r reason=%s",
                            len(raw),
                            raw[:1_000],
                            activity_error,
                        )
                    correction_reason = activity_error
                    if attempt == 0:
                        continue
                    if bootstrap:
                        proposal = _BOOTSTRAP_FALLBACK
                        break
                    store.fail_presence_decision(
                        profile_key,
                        claim_token=claim_token,
                        now=time.time() if now is None else current,
                        error=UNSUPPORTED_PHYSICAL_ACTIVITY_REASON,
                    )
                    return False
                break
        if proposal is None:
            return False
        accepted, reason = store.commit_presence_decision(
            profile_key,
            proposal,
            claim_token=claim_token,
            now=time.time() if now is None else current,
            expected_activity_id=expected_activity_id,
            expected_bootstrap=bootstrap,
            expected_emotion_updated_at=claimed.emotion.updated_at,
        )
        if bootstrap and not accepted:
            _log_bootstrap_failure(raw, reason)
        return accepted
    except InferenceCancelled:
        store.fail_presence_decision(
            profile_key,
            claim_token=claim_token,
            now=current,
            error="presence inference cancelled",
        )
        return False
    except Exception:
        store.fail_presence_decision(
            profile_key,
            claim_token=claim_token,
            now=time.time() if now is None else current,
            error="presence inference failed",
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
            format_presence_context(profile.presence)
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
_WORKER: OffscreenPresenceWorker | None = None


def start_presence_worker() -> OffscreenPresenceWorker:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = OffscreenPresenceWorker(get_state_store())
        _WORKER.start()
        return _WORKER


def stop_presence_worker() -> None:
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


def presence_worker_debug() -> dict[str, object]:
    with _WORKER_LOCK:
        worker = _WORKER
    if worker is None:
        return {
            "Presence Worker Started": False,
            "Pending Profiles": (),
            "Active Profile": "",
            "Last Error": "",
            "Next Retry At": 0.0,
        }
    return worker.snapshot()
