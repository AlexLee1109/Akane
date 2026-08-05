"""Lifecycle-managed presence, initiative, delivery, and maintenance lanes."""

from __future__ import annotations

import inspect
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.config import MAX_TOKENS
from app.core.memory import (
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
    PresenceParseError,
    format_presence_context,
    needs_bootstrap,
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
_BACKGROUND_QUEUE_SECONDS = 60.0 * 60.0
_MAINTENANCE_INTERVAL_SECONDS = 60.0
_SERVICE_JOIN_SECONDS = 5.0
_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _LaneState:
    name: str
    wake: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    active: str = ""
    cancellation: threading.Event | None = None
    consecutive_errors: int = 0
    last_error: str = ""
    next_retry_at: float = 0.0
    next_due_at: float = 0.0
    last_success_at: float = 0.0
    job_started_at: float = 0.0


class BackgroundService:
    """Process-owned background lanes sharing one serialized model runtime."""

    def __init__(
        self,
        store: StateStore,
        *,
        runner: Callable[..., bool] | None = None,
        initiative_runner: Callable[..., bool] | None = None,
        maintenance_runner: Callable[..., object] | None = None,
        maintenance_interval_seconds: float = _MAINTENANCE_INTERVAL_SECONDS,
    ) -> None:
        self._store = store
        self._presence_runner = runner
        self._initiative_runner = initiative_runner
        self._maintenance_runner = maintenance_runner
        self._maintenance_interval_seconds = max(
            0.05,
            float(maintenance_interval_seconds),
        )
        self._shutdown = threading.Event()
        self._lock = threading.RLock()
        self._delivery_condition = threading.Condition(threading.RLock())
        self._adapters: dict[str, tuple[str, float]] = {}
        self._lanes = {
            name: _LaneState(name)
            for name in ("presence", "initiative", "delivery", "maintenance")
        }
        self._pending_profiles: tuple[str, ...] = ()

    def start(self) -> bool:
        with self._lock:
            if any(
                lane.thread is not None and lane.thread.is_alive()
                for lane in self._lanes.values()
            ):
                return False
            self._shutdown.clear()
            targets = {
                "presence": self._presence_loop,
                "initiative": self._initiative_loop,
                "delivery": self._delivery_loop,
                "maintenance": self._maintenance_loop,
            }
            for name, lane in self._lanes.items():
                lane.wake.clear()
                lane.active = ""
                lane.cancellation = None
                lane.consecutive_errors = 0
                lane.last_error = ""
                lane.next_retry_at = 0.0
                lane.next_due_at = 0.0
                lane.last_success_at = 0.0
                lane.job_started_at = 0.0
                lane.thread = threading.Thread(
                    target=targets[name],
                    daemon=True,
                    name=f"AkaneBackground{name.title()}",
                )
            self._store.set_autonomy_wake(self.wake)
            threads = tuple(lane.thread for lane in self._lanes.values())
            for thread in threads:
                thread.start()
        self.wake()
        return True

    def stop(self, *, join_seconds: float = _SERVICE_JOIN_SECONDS) -> bool:
        self._shutdown.set()
        with self._lock:
            for lane in self._lanes.values():
                lane.wake.set()
                if lane.cancellation is not None:
                    lane.cancellation.set()
        with self._delivery_condition:
            channels = tuple(self._adapters)
            self._adapters.clear()
            self._delivery_condition.notify_all()
        for channel in channels:
            try:
                self._store.release_initiative_delivery(
                    adapter=channel,
                    now=time.time(),
                )
            except Exception as exc:
                self._record_error(self._lanes["delivery"], exc)
        deadline = time.monotonic() + max(0.0, float(join_seconds))
        with self._lock:
            threads = tuple(
                lane.thread for lane in self._lanes.values() if lane.thread is not None
            )
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self._store.set_autonomy_wake(None)
        return not any(thread.is_alive() for thread in threads)

    def wake(self, _profile_id: str = "") -> None:
        for lane in self._lanes.values():
            lane.wake.set()
        with self._delivery_condition:
            self._delivery_condition.notify_all()

    def wake_lane(self, name: str) -> bool:
        lane = self._lanes.get(str(name).strip().casefold())
        if lane is None:
            return False
        lane.wake.set()
        if lane.name == "delivery":
            with self._delivery_condition:
                self._delivery_condition.notify_all()
        return True

    def snapshot(self) -> dict[str, object]:
        current = time.time()
        model_status = ModelManager.get_instance().status()
        recovery_reader = getattr(self._store, "expired_claim_recoveries", None)
        expired_recoveries = (
            int(recovery_reader()) if callable(recovery_reader) else 0
        )
        with self._lock:
            lanes = {
                name: {
                    "Running": lane.thread is not None and lane.thread.is_alive(),
                    "Active": lane.active,
                    "Job State": (
                        "stopped"
                        if lane.thread is None or not lane.thread.is_alive()
                        else "running"
                        if lane.active
                        else "retry_wait"
                        if lane.next_retry_at > current
                        else "failed"
                        if lane.last_error
                        else "waiting"
                    ),
                    "Next Due At": lane.next_due_at,
                    "Last Success At": lane.last_success_at,
                    "Job Started At": lane.job_started_at,
                    "Last Error": lane.last_error,
                    "Next Retry At": lane.next_retry_at,
                    "Wake Pending": lane.wake.is_set(),
                }
                for name, lane in self._lanes.items()
            }
            running = bool(lanes) and all(
                bool(values["Running"]) for values in lanes.values()
            )
            return {
                "Background Service Started": running,
                "Lanes": lanes,
                "Pending Presence Profiles": self._pending_profiles,
                "Next Presence Due": lanes["presence"]["Next Due At"],
                "Active Presence Profile": lanes["presence"]["Active"],
                "Available Initiative Adapters": tuple(self._available_adapters()),
                "Expired Claim Recoveries": expired_recoveries,
                "Model Queue Wait Seconds": model_status.get(
                    "last_queue_wait_seconds",
                    0.0,
                ),
                "Model Inference Started At": model_status.get(
                    "inference_started_at",
                    0.0,
                ),
            }

    def _available_adapters(self, *, now: float | None = None) -> tuple[str, ...]:
        current = time.time() if now is None else now
        with self._delivery_condition:
            return tuple(
                name
                for name, (_conversation, expires) in self._adapters.items()
                if expires > current
            )

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
                self._lanes["delivery"].wake.set()
                return None
            self._adapters[channel] = (conversation, time.time() + 35.0)
            self._lanes["delivery"].wake.set()
            while not self._shutdown.is_set():
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
        self._lanes["delivery"].wake.set()
        return accepted

    @staticmethod
    def _invoke(
        runner: Callable[..., object],
        *,
        cancellation: threading.Event,
        **kwargs: object,
    ) -> object:
        try:
            parameters = inspect.signature(runner).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        supplied = {
            name: value
            for name, value in kwargs.items()
            if accepts_kwargs or name in parameters
        }
        if "cancellation" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            supplied["cancellation"] = cancellation
        for name in ("cancel_event", "stop_event"):
            if name in parameters:
                supplied[name] = cancellation
                break
        return runner(**supplied)

    def _run_profile(
        self,
        profile_id: str,
        cancellation: threading.Event,
    ) -> bool:
        runner = self._presence_runner
        if runner is None:
            runner = run_presence_turn
        return bool(self._invoke(
            runner,
            cancellation=cancellation,
            profile_id=profile_id,
        ))

    def _record_error(self, lane: _LaneState, exc: Exception) -> None:
        with self._lock:
            lane.consecutive_errors += 1
            delay = min(
                _MAX_ERROR_BACKOFF_SECONDS,
                _MIN_ERROR_BACKOFF_SECONDS
                * (2 ** min(lane.consecutive_errors - 1, 8)),
            )
            lane.last_error = f"{type(exc).__name__}: {exc}"
            lane.next_retry_at = time.time() + delay

    def _clear_error(self, lane: _LaneState, *, mark_success: bool = False) -> None:
        with self._lock:
            lane.consecutive_errors = 0
            lane.last_error = ""
            lane.next_retry_at = 0.0
            if mark_success:
                lane.last_success_at = time.time()

    def _wait_for_local_retry(self, lane: _LaneState) -> bool:
        while not self._shutdown.is_set():
            with self._lock:
                timeout = lane.next_retry_at - time.time()
            if timeout <= 0.0:
                return True
            lane.wake.clear()
            if self._shutdown.is_set():
                return False
            lane.wake.wait(timeout)
        return False

    def _begin_job(self, lane: _LaneState, active: str) -> threading.Event:
        cancellation = threading.Event()
        with self._lock:
            lane.active = active
            lane.job_started_at = time.time()
            lane.cancellation = cancellation
            if self._shutdown.is_set():
                cancellation.set()
        return cancellation

    def _end_job(self, lane: _LaneState) -> None:
        with self._lock:
            lane.active = ""
            lane.job_started_at = 0.0
            lane.cancellation = None

    @staticmethod
    def _timeout(next_due_at: float | None) -> float | None:
        return (
            None
            if next_due_at is None
            else max(0.0, next_due_at - time.time())
        )

    def _presence_loop(self) -> None:
        lane = self._lanes["presence"]
        while not self._shutdown.is_set():
            if not self._wait_for_local_retry(lane):
                break
            try:
                due_profiles, next_due_at = self._store.presence_schedule(
                    now=time.time(),
                )
            except Exception as exc:
                self._record_error(lane, exc)
                continue
            with self._lock:
                self._pending_profiles = due_profiles
                lane.next_due_at = next_due_at or 0.0
            for profile_id in due_profiles:
                if self._shutdown.is_set():
                    break
                cancellation = self._begin_job(lane, profile_id)
                try:
                    succeeded = self._run_profile(profile_id, cancellation)
                except Exception as exc:
                    self._record_error(lane, exc)
                    break
                else:
                    if succeeded:
                        self._clear_error(lane, mark_success=True)
                    else:
                        self._record_error(
                            lane,
                            RuntimeError("presence job returned false"),
                        )
                        break
                finally:
                    self._end_job(lane)
            if due_profiles:
                continue
            lane.wake.clear()
            try:
                due_profiles, next_due_at = self._store.presence_schedule(
                    now=time.time(),
                )
            except Exception as exc:
                self._record_error(lane, exc)
                continue
            with self._lock:
                self._pending_profiles = due_profiles
                lane.next_due_at = next_due_at or 0.0
            if due_profiles:
                continue
            if self._shutdown.is_set():
                break
            lane.wake.wait(self._timeout(next_due_at))

    def _initiative_loop(self) -> None:
        lane = self._lanes["initiative"]
        runner = self._initiative_runner or run_initiative_evaluation
        while not self._shutdown.is_set():
            if not self._wait_for_local_retry(lane):
                break
            try:
                due, next_due_at = self._store.initiative_schedule(now=time.time())
            except Exception as exc:
                self._record_error(lane, exc)
                continue
            with self._lock:
                lane.next_due_at = next_due_at or 0.0
            if due:
                cancellation = self._begin_job(lane, "initiative")
                try:
                    succeeded = bool(self._invoke(runner, cancellation=cancellation))
                except Exception as exc:
                    self._record_error(lane, exc)
                else:
                    if succeeded:
                        self._clear_error(lane, mark_success=True)
                    else:
                        self._record_error(
                            lane,
                            RuntimeError("initiative job returned false"),
                        )
                finally:
                    self._end_job(lane)
                continue
            lane.wake.clear()
            try:
                due, next_due_at = self._store.initiative_schedule(now=time.time())
            except Exception as exc:
                self._record_error(lane, exc)
                continue
            with self._lock:
                lane.next_due_at = next_due_at or 0.0
            if due:
                continue
            if self._shutdown.is_set():
                break
            lane.wake.wait(self._timeout(next_due_at))

    def _delivery_loop(self) -> None:
        lane = self._lanes["delivery"]
        while not self._shutdown.is_set():
            if not self._wait_for_local_retry(lane):
                break
            lane.wake.clear()
            current = time.time()
            with self._delivery_condition:
                expired = tuple(
                    name
                    for name, (_conversation, expires) in self._adapters.items()
                    if expires <= current
                )
                for name in expired:
                    self._adapters.pop(name, None)
                next_expiry = min(
                    (expires for _conversation, expires in self._adapters.values()),
                    default=None,
                )
                self._delivery_condition.notify_all()
            with self._lock:
                lane.next_due_at = next_expiry or 0.0
            for channel in expired:
                try:
                    self._store.release_initiative_delivery(
                        adapter=channel,
                        now=current,
                    )
                except Exception as exc:
                    self._record_error(lane, exc)
                    break
            else:
                if expired:
                    self._clear_error(lane, mark_success=True)
            if self._shutdown.is_set():
                break
            lane.wake.wait(self._timeout(next_expiry))

    def _maintenance_pass(self, cancellation: threading.Event) -> None:
        runner = self._maintenance_runner
        if runner is not None:
            self._invoke(
                runner,
                cancellation=cancellation,
                store=self._store,
            )
            return
        current = time.time()
        presence_due, _presence_at = self._store.presence_schedule(now=current)
        initiative_due, _initiative_at = self._store.initiative_schedule(now=current)
        if presence_due:
            self._lanes["presence"].wake.set()
        if initiative_due:
            self._lanes["initiative"].wake.set()

    def _maintenance_loop(self) -> None:
        lane = self._lanes["maintenance"]
        while not self._shutdown.is_set():
            if not self._wait_for_local_retry(lane):
                break
            lane.wake.clear()
            cancellation = self._begin_job(lane, "maintenance")
            failed = False
            try:
                self._maintenance_pass(cancellation)
            except Exception as exc:
                failed = True
                self._record_error(lane, exc)
            else:
                self._clear_error(lane, mark_success=True)
            finally:
                self._end_job(lane)
            if self._shutdown.is_set():
                break
            if failed:
                continue
            with self._lock:
                lane.next_due_at = time.time() + self._maintenance_interval_seconds
            lane.wake.wait(self._maintenance_interval_seconds)


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
        presence=format_presence_context(profile.presence, now=now),
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
            queue_deadline=time.monotonic() + _BACKGROUND_QUEUE_SECONDS,
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
                    else:
                        _LOGGER.warning(
                            "presence parse failed format=raw_json_schema attempt=%d chars=%d "
                            "preview=%r error=%s",
                            attempt + 1,
                            len(raw),
                            raw[:1_000],
                            exc,
                        )
                    correction_reason = str(exc)
                    if attempt == 0:
                        continue
                    store.fail_presence_decision(
                        profile_key,
                        claim_token=claim_token,
                        now=current if now is not None else time.time(),
                        error=str(exc),
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
        store.defer_presence_decision(
            profile_key,
            claim_token=claim_token,
            now=current if now is not None else time.time(),
        )
        return True
    except Exception as exc:
        store.fail_presence_decision(
            profile_key,
            claim_token=claim_token,
            now=time.time() if now is None else current,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def _topic_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()[:120]


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
        presence=format_presence_context(profile.presence, now=now),
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
    opportunity = store.claim_initiative_evaluation(now=current)
    if opportunity is None or not opportunity.claim_token:
        return False
    try:
        manager = ModelManager.get_instance()
        timing = InferenceTiming(requested_at=time.perf_counter())
        with manager.reserve(
            priority="background",
            cancellation=cancellation,
            queue_deadline=time.monotonic() + _BACKGROUND_QUEUE_SECONDS,
        ) as reservation:
            prompt_now = current if now is not None else time.time()
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
                now=current if now is not None else time.time(),
            )
            return False
        completed = store.complete_initiative_evaluation(
            claim_token=opportunity.claim_token,
            decision=decision.decision,
            topic=decision.topic,
            message=decision.message,
            now=time.time() if now is None else current,
        )
        return completed is not None
    except InferenceCancelled:
        if opportunity is not None and opportunity.claim_token:
            store.fail_initiative_evaluation(
                claim_token=opportunity.claim_token,
                now=current if now is not None else time.time(),
            )
        return False
    except Exception:
        if opportunity is not None and opportunity.claim_token:
            store.fail_initiative_evaluation(
                claim_token=opportunity.claim_token,
                now=time.time() if now is None else current,
            )
        raise


_SERVICE_LOCK = threading.Lock()
_SERVICE: BackgroundService | None = None


def start_background_service() -> BackgroundService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = BackgroundService(get_state_store())
        _SERVICE.start()
        return _SERVICE


def stop_background_service() -> bool:
    with _SERVICE_LOCK:
        service = _SERVICE
    return True if service is None else service.stop()


def background_service_debug() -> dict[str, object]:
    with _SERVICE_LOCK:
        service = _SERVICE
    if service is None:
        model_status = ModelManager.get_instance().status()
        return {
            "Background Service Started": False,
            "Lanes": {},
            "Pending Presence Profiles": (),
            "Next Presence Due": 0.0,
            "Active Presence Profile": "",
            "Available Initiative Adapters": (),
            "Expired Claim Recoveries": get_state_store().expired_claim_recoveries(),
            "Model Queue Wait Seconds": model_status.get(
                "last_queue_wait_seconds",
                0.0,
            ),
            "Model Inference Started At": model_status.get(
                "inference_started_at",
                0.0,
            ),
        }
    return service.snapshot()


def claim_initiative_delivery(
    *,
    adapter: str,
    conversation_id: str,
    available: bool,
    wait_seconds: float = 25.0,
) -> dict[str, object] | None:
    with _SERVICE_LOCK:
        service = _SERVICE
    if service is None:
        return None
    return service.claim_delivery(
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
    with _SERVICE_LOCK:
        service = _SERVICE
    if service is None:
        return False
    return service.acknowledge_delivery(
        opportunity_id=opportunity_id,
        claim_token=claim_token,
        adapter=adapter,
        conversation_id=conversation_id,
        success=success,
        message_id=message_id,
    )