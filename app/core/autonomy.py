"""One low-duty-cycle coordinator for reflection, InnerLife, and delivery."""

from __future__ import annotations

import threading
import time

from app.core.config import SETTINGS
from app.core.context import ContextBuilder
from app.core.inference import InferencePreempted, InferenceRuntime, InferenceTiming
from app.core.mind import validate_inner_life
from app.core.prompt import build_inner_life_prompt
from app.core.reflection import ReflectionEngine
from app.core.state import StateChangeProposal
from app.core.store import Store, get_store
from app.core.utils import OWNER_PROFILE_ID, log_performance, log_timing


class AutonomyCoordinator:
    def __init__(self, store: Store | None = None):
        self.store = store or get_store()
        self.reflection = ReflectionEngine(self.store)
        self.runtime = InferenceRuntime.get_instance()
        self.context_builder = ContextBuilder(self.store)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Starting the server must not let background cognition jump ahead of a user turn.
        self._last_inner_life = time.time()
        self._last_error = ""
        if SETTINGS.prompt_debug:
            print(
                f"[Akane:debug:autonomy] coordinator_id={id(self):x} "
                f"runtime_id={id(self.runtime):x} reflection_id={id(self.reflection):x}",
                flush=True,
            )

    def start(self) -> "AutonomyCoordinator":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AkaneAutonomy")
        self._thread.start()
        return self

    def stop(self) -> bool:
        self._stop.set()
        self.runtime.notify_foreground()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._thread = None
        return thread is not None

    def _run(self) -> None:
        while not self._stop.is_set():
            cycle_started_at = time.perf_counter()
            try:
                self.run_due_once()
                self._last_error = ""
            except Exception as exc:  # background failure is isolated from conversation
                self._last_error = f"{type(exc).__name__}: {exc}"[:500]
            wait_started_at = time.perf_counter()
            self._stop.wait(SETTINGS.autonomy_interval_seconds)
            log_timing(
                "autonomy",
                total=wait_started_at - cycle_started_at,
                wait=time.perf_counter() - wait_started_at,
            )

    def run_due_once(self, *, now: float | None = None) -> str:
        current = time.time() if now is None else float(now)
        if not self.runtime.background_allowed(now=current):
            return "foreground_or_recent_activity"
        job = self.store.claim_reflection_job(now=current)
        if job is not None:
            try:
                self.reflection.run_job(job)
            except InferencePreempted:
                self.store.defer_reflection_job(str(job["id"]), now=now)
                return "reflection_preempted"
            except Exception as exc:
                self.store.finish_reflection_job(str(job["id"]), error=str(exc), now=now)
                raise
            else:
                return "reflection"
        if current - self._last_inner_life >= SETTINGS.inner_life_interval_seconds:
            try:
                self._run_inner_life(OWNER_PROFILE_ID)
            except InferencePreempted:
                return "inner_life_preempted"
            self._last_inner_life = current
            return "inner_life"
        return "quiet"

    def _run_inner_life(
        self,
        profile_id: str,
        conversation_id: str = "popup:default",
    ) -> StateChangeProposal:
        started_at = time.perf_counter()
        timing = InferenceTiming(started_at)
        context = self.context_builder.build(
            profile_id=profile_id,
            conversation_id=conversation_id,
            message="",
            allow_tool_context=False,
        )
        context_finished_at = time.perf_counter()
        wait_started_at = context_finished_at
        with self.runtime.reserve(priority="autonomy") as reservation:
            reservation_acquired_at = time.perf_counter()
            output = self.runtime.complete_messages(
                build_inner_life_prompt(context),
                max_tokens=SETTINGS.inner_life_tokens,
                reservation=reservation,
                timing=timing,
                temperature=0.35,
                call_kind="inner_life",
            )
        inference_finished_at = time.perf_counter()
        if reservation.preemption.is_set():
            raise InferencePreempted("InnerLife yielded before validation.")
        proposal = validate_inner_life(
            output,
            context=context,
            last_user_message_at=self.store.latest_turn_at(profile_id, role="user"),
        )
        parsed_at = time.perf_counter()
        if reservation.preemption.is_set():
            raise InferencePreempted("InnerLife yielded before persistence.")
        if proposal.thoughts or proposal.self_items or proposal.proactive_messages or proposal.rejected:
            self.store.commit(proposal)
        committed_at = time.perf_counter()
        log_timing(
            "autonomy.inner_life",
            context=context_finished_at - started_at,
            wait_for_model=reservation_acquired_at - wait_started_at,
            inference=inference_finished_at - reservation_acquired_at,
            parse=parsed_at - inference_finished_at,
            commit=committed_at - parsed_at,
            total=committed_at - started_at,
        )
        log_performance(
            "inner_life",
            input_tokens=timing.prompt_tokens,
            output_tokens=timing.generated_tokens,
            generation_ms=max(0.0, timing.model_finished_at - timing.model_started_at) * 1000,
            prefill_ms=timing.prefill_seconds * 1000,
            decode_tok_s=(
                timing.generated_tokens / timing.decode_seconds
                if timing.generated_tokens and timing.decode_seconds else 0.0
            ),
            reused_prefix_tokens=timing.reused_prefix_tokens,
            new_prompt_eval_tokens=timing.new_prompt_eval_tokens,
            model_wait_ms=max(0.0, reservation_acquired_at - wait_started_at) * 1000,
            total_ms=max(0.0, committed_at - started_at) * 1000,
            finish_reason=timing.finish_reason or "unavailable",
            accepted_thoughts=len(proposal.thoughts),
            accepted_self=len(proposal.self_items),
            rejection_codes=",".join(proposal.rejected) or "none",
        )
        return proposal

    def debug(self) -> dict[str, object]:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "last_inner_life_at": self._last_inner_life,
            "last_error": self._last_error,
        }


_COORDINATOR: AutonomyCoordinator | None = None
_LOCK = threading.Lock()


def get_autonomy() -> AutonomyCoordinator:
    global _COORDINATOR
    if _COORDINATOR is None:
        with _LOCK:
            if _COORDINATOR is None:
                _COORDINATOR = AutonomyCoordinator()
    return _COORDINATOR


def start_autonomy() -> AutonomyCoordinator:
    return get_autonomy().start()


def stop_autonomy() -> bool:
    return get_autonomy().stop()


def claim_proactive_delivery(
    *,
    adapter: str,
    conversation_id: str,
    available: bool,
    wait_seconds: float = 0.0,
) -> dict[str, object] | None:
    if not available:
        return None
    store = get_store()
    timeout = max(0.0, min(float(wait_seconds), 30.0))
    delivery = store.claim_proactive(
        OWNER_PROFILE_ID,
        adapter=adapter,
        conversation_id=conversation_id,
    )
    if delivery is not None or timeout <= 0:
        return delivery
    if not store.wait_for_proactive(OWNER_PROFILE_ID, timeout):
        return None
    return store.claim_proactive(
        OWNER_PROFILE_ID,
        adapter=adapter,
        conversation_id=conversation_id,
    )


def acknowledge_proactive_delivery(
    *,
    opportunity_id: str,
    claim_token: str,
    adapter: str,
    conversation_id: str,
    success: bool,
    message_id: str = "",
) -> bool:
    del adapter, conversation_id
    return get_store().acknowledge_proactive(
        opportunity_id,
        claim_token,
        success=success,
        delivery_message_id=message_id,
    )
