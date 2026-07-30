"""The single event-driven autonomous-presence worker for this process."""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Callable
from dataclasses import replace

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
)
from app.core.presence import format_presence_context, parse_life_decision
from app.core.prompt import PromptContext, build_life_prompt
from app.core.utils import canonical_profile_id, compact_text

_MIN_ERROR_BACKOFF_SECONDS = 5.0
_MAX_ERROR_BACKOFF_SECONDS = 60.0


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
            self._store.set_presence_wake(self.wake)
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
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._store.set_presence_wake(None)

    def wake(self, _profile_id: str = "") -> None:
        self._wake.set()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return {
                "Life Worker Started": running,
                "Pending Profiles": self._pending_profiles,
                "Active Profile": self._active_profile,
                "Last Error": self._last_error,
                "Next Retry At": self._next_retry_at,
            }

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

    def _schedule(self) -> tuple[tuple[str, ...], float | None]:
        due, next_due_at = self._store.presence_schedule(now=time.time())
        with self._lock:
            self._pending_profiles = due
        return due, next_due_at

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wait_for_local_retry()
            if self._stop.is_set():
                break
            try:
                due_profiles, next_due_at = self._schedule()
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

            self._wake.clear()
            try:
                due_profiles, next_due_at = self._schedule()
            except Exception as exc:
                self._record_error(exc)
                continue
            if due_profiles:
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


def _profile_text(item: object) -> str:
    for name in ("content", "text", "summary"):
        value = compact_text(getattr(item, name, ""), 320)
        if value:
            return value
    return compact_text(item, 320)


def _life_context(profile, *, now: float) -> PromptContext:
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
    return PromptContext(
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
    context = _life_context(claimed, now=current)
    retry_note = (
        "The previous proposal was too similar. Choose independently without "
        "reusing that activity."
        if claimed.presence.last_error
        and "repeat" in claimed.presence.last_error
        else ""
    )
    try:
        manager = ModelManager.get_instance()
        timing = InferenceTiming(requested_at=time.perf_counter())
        with manager.reserve(
            priority="background",
            cancellation=cancellation,
        ) as reservation:
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
                    reservation=reservation,
                )
            ).strip()
        grounded = "\n".join(
            message["content"]
            for message in plan.messages
        )
        decision = parse_life_decision(raw, grounded_context=grounded)
        if decision is None:
            store.fail_presence_decision(
                profile_key,
                claim_token=claim_token,
                now=current,
                error="invalid life block",
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


def _initiative_candidates(snapshot, *, now: float) -> tuple[InitiativeOpportunity, ...]:
    profile = snapshot.profile
    candidates: list[InitiativeOpportunity] = []
    activity = profile.presence.current_activity
    if activity is not None and activity.detail:
        candidates.append(
            InitiativeOpportunity(
                "meaningful current offscreen activity",
                activity.fact(),
                0.56,
                activity.started_at,
                activity.expected_end_at,
            )
        )
    previous = profile.presence.previous_activity
    if (
        previous is not None
        and previous.expected_end_at <= now
        < previous.expected_end_at + 12.0 * 3600.0
    ):
        candidates.append(
            InitiativeOpportunity(
                "recent completed offscreen activity",
                previous.fact(),
                0.62,
                previous.expected_end_at,
                previous.expected_end_at + 12.0 * 3600.0,
            )
        )
    relationship = (
        *profile.relationship.unresolved_events,
        *profile.relationship.shared_context,
    )
    if relationship:
        entry = max(relationship, key=lambda item: (item.confidence, item.updated_at))
        if entry.updated_at > 0.0:
            candidates.append(
                InitiativeOpportunity(
                    "meaningful relationship continuity",
                    entry.summary,
                    max(0.55, min(0.85, entry.confidence)),
                    entry.updated_at,
                    entry.updated_at + 7.0 * 24.0 * 3600.0,
                )
            )
    unresolved = tuple(
        item
        for item in profile.memories
        if item.kind in {"commitment", "project", "concern"}
        and item.updated_at > 0.0
    )
    if unresolved:
        memory = max(
            unresolved,
            key=lambda item: (item.confidence, item.updated_at),
        )
        candidates.append(
            InitiativeOpportunity(
                "unresolved grounded context",
                memory.text,
                max(0.55, min(0.85, memory.confidence)),
                memory.updated_at,
                memory.updated_at + 7.0 * 24.0 * 3600.0,
            )
        )
    return tuple(item for item in candidates if item.expires_at > now)


def run_initiative_turn(
    *,
    profile_id: str,
    conversation_id: str,
    source: str,
    display_name: str = "",
    now: float | None = None,
):
    """Offer one grounded outreach opportunity to the normal single-inference path."""

    from app.core.session import (
        CompanionDecision,
        CompanionTurnResult,
        GenerationBusyError,
        autonomous_dialogue_available,
        normalize_chat_input,
        run_companion_turn,
    )

    current = time.time() if now is None else max(0.0, float(now))
    profile = canonical_profile_id(profile_id)
    if not autonomous_dialogue_available(profile, conversation_id):
        return CompanionTurnResult(CompanionDecision("", should_respond=False))
    store = get_state_store()
    snapshot = store.snapshot(
        profile,
        conversation_id,
        query="",
        now=current,
    )
    opportunity = store.claim_initiative_opportunity(
        profile_id=profile,
        conversation_id=conversation_id,
        candidates=_initiative_candidates(snapshot, now=current),
        now=current,
        active_window_seconds=15.0 * 60.0,
    )
    if opportunity is None:
        return CompanionTurnResult(CompanionDecision("", should_respond=False))
    chat = replace(
        normalize_chat_input(
            text="A grounded initiative opportunity is available.",
            profile_id=profile,
            conversation_id=conversation_id,
            source=source,
            timestamp=current,
            display_name=display_name,
            autonomous=True,
        ),
        initiative_opportunity=opportunity,
    )
    try:
        return run_companion_turn(chat, skip_if_busy=True)
    except GenerationBusyError:
        return CompanionTurnResult(CompanionDecision("", should_respond=False))


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
