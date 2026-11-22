"""Native L2 residency. The existing L2 worker thread is its sole mutator."""

from enum import Enum
from threading import Lock
from time import monotonic

from ..conversation.cancellation import GenerationCancelled
from .l2_memory import L2ResourceError


class Residency(str, Enum):
    UNLOADED = 'unloaded'
    LOADING = 'loading'
    ABORTING = 'aborting'
    READY = 'ready'
    UNLOADING = 'unloading'
    UNLOAD_FAILED = 'unload_failed'


class LazyL2:
    architecture = 'qwen35'

    def __init__(self, path, context_size, idle_seconds, load, *, clock=monotonic, retry_seconds=60):
        self.model_path = path
        self.context_size = context_size
        self.idle_seconds = idle_seconds
        self._load, self._clock, self._retry_seconds = load, clock, retry_seconds
        self._lock = Lock()
        self._model = None
        self._state = Residency.UNLOADED
        self._retry_after = None
        self._failure = None
        self._cold_starts = self._warm_starts = 0
        self._last_load_seconds = self._last_unload_reason = None

    @property
    def ready(self):
        with self._lock:
            return self._state == Residency.READY

    def acquire(self, cancellation, mark_aborting):
        cancellation.raise_if_cancelled()
        started = self._clock()
        with self._lock:
            if self._state == Residency.READY:
                self._warm_starts += 1
                return self._model
            if self._state != Residency.UNLOADED:
                raise RuntimeError('L2 is unavailable after an incomplete lifecycle operation.')
            if self._retry_after is not None and started < self._retry_after:
                raise L2ResourceError('L2 is cooling down after a failed cold start.')
            self._cold_starts += 1
            self._state = Residency.LOADING

        def on_abort():
            with self._lock:
                self._state = Residency.ABORTING
            mark_aborting()

        try:
            model = self._load(cancellation, on_abort)
        except GenerationCancelled:
            raise
        except Exception as error:
            failure = (str(error) if isinstance(error, L2ResourceError) else 'Qwen L2 initialization failed.')
            with self._lock:
                self._failure = failure
                self._retry_after = self._clock() + self._retry_seconds
            raise L2ResourceError(failure) from error
        else:
            # A successfully guarded model can serve another job if this task was
            # cancelled just after initialization. No cancelled task uses it.
            with self._lock:
                self._model = model
                self._state = Residency.READY
                self._failure = self._retry_after = None
            cancellation.raise_if_cancelled()
            return model
        finally:
            with self._lock:
                self._last_load_seconds = self._clock() - started
                if self._state in (Residency.LOADING, Residency.ABORTING):
                    self._state = Residency.UNLOADED

    def unload(self, reason):
        with self._lock:
            model = self._model
            if model is None:
                return
            self._state = Residency.UNLOADING
        try:
            model.close()
        except Exception:
            with self._lock:
                self._failure = 'Qwen L2 shutdown failed.'
                self._state = Residency.UNLOAD_FAILED
            raise
        else:
            with self._lock:
                self._model = None
                self._state = Residency.UNLOADED
                self._last_unload_reason = reason

    def snapshot(self):
        with self._lock:
            return {'lifecycle_state': self._state.value, 'loaded': self._model is not None,
                    'cold_start_count': self._cold_starts, 'warm_start_count': self._warm_starts,
                    'last_load_seconds': self._last_load_seconds, 'last_unload_reason': self._last_unload_reason,
                    'idle_unload_seconds': self.idle_seconds, 'last_load_failure': self._failure}
