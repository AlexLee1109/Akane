"""The single event-driven autonomous-presence worker for this process."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from app.core.memory import LongTermMemoryStore, get_internal_state_store


class AutonomousLifeWorker:
    def __init__(
        self,
        store: LongTermMemoryStore,
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
            }

    def _run_profile(self, profile_id: str) -> bool:
        if self._runner is not None:
            return self._runner(profile_id=profile_id)
        from app.core.session import run_life_turn

        return run_life_turn(profile_id=profile_id)

    def _schedule(self) -> tuple[tuple[str, ...], float | None]:
        due, next_due_at = self._store.presence_schedule(now=time.time())
        with self._lock:
            self._pending_profiles = due
        return due, next_due_at

    def _run(self) -> None:
        while not self._stop.is_set():
            due_profiles, next_due_at = self._schedule()
            if due_profiles:
                for profile_id in due_profiles:
                    if self._stop.is_set():
                        break
                    with self._lock:
                        self._active_profile = profile_id
                    try:
                        self._run_profile(profile_id)
                    except Exception:
                        pass
                    finally:
                        with self._lock:
                            self._active_profile = ""
                continue

            self._wake.clear()
            due_profiles, next_due_at = self._schedule()
            if due_profiles:
                continue
            if self._stop.is_set():
                break
            timeout = (
                None
                if next_due_at is None
                else max(0.0, next_due_at - time.time())
            )
            self._wake.wait(timeout)


_WORKER_LOCK = threading.Lock()
_WORKER: AutonomousLifeWorker | None = None


def start_life_worker() -> AutonomousLifeWorker:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = AutonomousLifeWorker(get_internal_state_store())
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
        }
    return worker.snapshot()
