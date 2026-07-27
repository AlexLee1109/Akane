"""One process-wide autonomous-life worker with wake and recovery polling."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from app.core.memory import LongTermMemoryStore, get_internal_state_store, set_life_pending_notifier

_POLL_SECONDS = 5 * 60.0
_MAX_BACKOFF_SECONDS = 5 * 60.0


class AutonomousLifeWorker:
    def __init__(
        self,
        store: LongTermMemoryStore,
        *,
        runner: Callable[..., bool] | None = None,
        poll_seconds: float = _POLL_SECONDS,
    ) -> None:
        self._store = store
        self._runner = runner
        self._poll_seconds = max(0.05, poll_seconds)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._retry_at: dict[str, float] = {}
        self._attempts: dict[str, int] = {}
        self._debug: dict[str, object] = {
            "Life Worker Started": False,
            "Pending Profiles": (),
            "Life Job Claimed": "",
            "Claim Age": 0.0,
            "Life Inference Started": False,
            "Life Block Parsed": False,
            "Life Activity Persisted": False,
            "Life Job Completed": "",
            "Life Job Failed": "",
            "Next Retry At": 0.0,
            "Autonomous Proposal": "",
            "Proposal Rejected": False,
            "Rejection Reason": "",
            "Activity Pattern": {},
        }

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="AkaneAutonomousLife",
            )
            self._thread.start()
            self._debug["Life Worker Started"] = True
        self.wake()
        return True

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        with self._lock:
            self._debug["Life Worker Started"] = False

    def wake(self, _profile_id: str = "") -> None:
        self._wake.set()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._debug)

    def _run_profile(self, profile_id: str) -> bool:
        if self._runner is not None:
            return self._runner(profile_id=profile_id)
        from app.core.session import run_life_turn

        return run_life_turn(
            profile_id=profile_id,
            status_callback=self._record_status,
        )

    def _record_status(self, event: str, profile_id: str, value: object = True) -> None:
        fields = {
            "claimed": "Life Job Claimed",
            "claim_age": "Claim Age",
            "inference_started": "Life Inference Started",
            "block_parsed": "Life Block Parsed",
            "activity_persisted": "Life Activity Persisted",
            "completed": "Life Job Completed",
            "proposal": "Autonomous Proposal",
            "rejected": "Rejection Reason",
        }
        field = fields.get(event)
        if field is not None:
            with self._lock:
                self._debug[field] = profile_id if event in {"claimed", "completed"} else value
                if event == "rejected":
                    self._debug["Proposal Rejected"] = True

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            profiles = self._store.pending_life_profiles(now=now)
            with self._lock:
                self._debug["Pending Profiles"] = profiles
            for profile_id in profiles:
                if self._stop.is_set() or self._retry_at.get(profile_id, 0.0) > now:
                    continue
                with self._lock:
                    self._debug.update(
                        {
                            "Life Job Claimed": "",
                            "Claim Age": 0.0,
                            "Life Inference Started": False,
                            "Life Block Parsed": False,
                            "Life Activity Persisted": False,
                            "Life Job Completed": "",
                            "Life Job Failed": "",
                            "Autonomous Proposal": "",
                            "Proposal Rejected": False,
                            "Rejection Reason": "",
                            "Activity Pattern": {},
                            "Next Retry At": 0.0,
                        }
                    )
                try:
                    completed = self._run_profile(profile_id)
                    after = self._store.internal_state(profile_id).presence
                    if not completed:
                        if after.life_pending and after.life_claimed_at:
                            continue
                        attempt = self._attempts.get(profile_id, 0) + 1
                        self._attempts[profile_id] = attempt
                        retry_at = time.time() + min(
                            _MAX_BACKOFF_SECONDS,
                            5.0 * (2 ** min(5, attempt - 1)),
                        )
                        self._retry_at[profile_id] = retry_at
                        with self._lock:
                            self._debug.update(
                                {
                                    "Life Inference Started": False,
                                    "Next Retry At": retry_at,
                                }
                            )
                        continue
                    with self._lock:
                        self._debug.update(
                            {
                                "Life Block Parsed": True,
                                "Life Activity Persisted": bool(after.current_activity or after.next_activity),
                                "Life Job Completed": profile_id,
                                "Life Inference Started": False,
                                "Next Retry At": 0.0,
                                "Activity Pattern": after.activity_pattern.as_dict(),
                            }
                        )
                    self._retry_at.pop(profile_id, None)
                    self._attempts.pop(profile_id, None)
                except Exception as exc:
                    attempt = self._attempts.get(profile_id, 0) + 1
                    self._attempts[profile_id] = attempt
                    retry_at = time.time() + min(_MAX_BACKOFF_SECONDS, 5.0 * (2 ** min(5, attempt - 1)))
                    self._retry_at[profile_id] = retry_at
                    self._store.release_life_opportunity(
                        profile_id,
                        now=time.time(),
                        failure_reason=type(exc).__name__,
                    )
                    with self._lock:
                        self._debug.update(
                            {
                                "Life Job Failed": f"{profile_id}:{type(exc).__name__}",
                                "Life Inference Started": False,
                                "Next Retry At": retry_at,
                            }
                        )
            now = time.time()
            pending_after = self._store.pending_life_profiles(now=now)
            with self._lock:
                self._debug["Pending Profiles"] = pending_after
            deadlines = [
                value
                for value in (
                    self._store.next_life_wake_at(),
                    min(self._retry_at.values(), default=0.0),
                )
                if value > now
            ]
            timeout = self._poll_seconds
            if deadlines:
                timeout = min(timeout, max(0.05, min(deadlines) - now))
            self._wake.wait(timeout)
            self._wake.clear()


_WORKER_LOCK = threading.Lock()
_WORKER: AutonomousLifeWorker | None = None


def start_life_worker() -> AutonomousLifeWorker:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = AutonomousLifeWorker(get_internal_state_store())
            set_life_pending_notifier(_WORKER.wake)
        _WORKER.start()
        return _WORKER


def stop_life_worker() -> None:
    global _WORKER
    with _WORKER_LOCK:
        worker = _WORKER
    if worker is not None:
        worker.stop()


def life_worker_debug() -> dict[str, object]:
    with _WORKER_LOCK:
        worker = _WORKER
    return worker.snapshot() if worker is not None else {
        "Life Worker Started": False,
        "Pending Profiles": (),
    }
