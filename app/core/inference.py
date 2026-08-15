"""One priority-aware llama.cpp runtime with token-level streaming."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import inspect
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from app.core.character import load_character_profile
from app.core.config import SETTINGS


class InferenceCancelled(RuntimeError):
    pass


class InferencePreempted(InferenceCancelled):
    """Background inference stopped so foreground work can use the model."""


class InferenceQueueTimeout(RuntimeError):
    pass


@dataclass(slots=True)
class InferenceTiming:
    requested_at: float
    model_started_at: float = 0.0
    first_token_at: float = 0.0
    final_token_at: float = 0.0
    model_finished_at: float = 0.0
    prompt_tokens: int = 0
    prompt_token_method: str = "unavailable"
    prefill_tokens: int = 0
    new_prompt_eval_tokens: int | None = None
    reused_prefix_tokens: int | None = None
    generated_tokens: int = 0
    generated_token_method: str = "unavailable"
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    first_token_estimated: bool = False


@dataclass(frozen=True, slots=True)
class Reservation:
    llm: object
    priority: str
    preemption: threading.Event


DialogueCacheKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class _ForegroundSequenceState:
    owner: DialogueCacheKey
    input_ids: object
    n_tokens: int
    seed: object
    cache_epoch: int
    data: bytearray


def _content(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
            for item in value
        )
    return "" if value is None else str(value)


def _chat_formatter(llm):
    """Find llama-cpp-python's active embedded Jinja formatter when exposed."""

    try:
        from llama_cpp import llama_chat_format

        handler = (
            llm.chat_handler
            or llm._chat_handlers.get(llm.chat_format)
            or llama_chat_format.get_chat_completion_handler(llm.chat_format)
        )
        for cell in handler.__closure__ or ():
            if isinstance(cell.cell_contents, llama_chat_format.Jinja2ChatFormatter):
                return cell.cell_contents
    except Exception:  # formatter lookup differs across binding/model versions
        return None
    return None


def _render_chat_prompt(llm, messages) -> str | None:
    formatter = _chat_formatter(llm)
    if formatter is None:
        return None
    try:
        return str(formatter(messages=list(messages)).prompt)
    except (AttributeError, TypeError, ValueError):
        return None


class InferenceRuntime:
    _instance: "InferenceRuntime | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._path = Path(SETTINGS.model_path)
        self._llm = None
        self._load_error: Exception | None = None
        self._loading = False
        self._load_lock = threading.RLock()
        self._condition = threading.Condition()
        self._active = False
        self._active_priority = ""
        self._active_preemption: threading.Event | None = None
        self._waiters: list[tuple[int, int]] = []
        self._foreground_requests = 0
        self._last_foreground_activity_at = 0.0
        self._sequence = 0
        self._last_wait = 0.0
        self._load_count = 0
        self._gemma_thinking_disabled = False
        self._live_cache_owner: DialogueCacheKey | None = None
        self._cache_epoch = 0
        self._last_kv_snapshot_bytes = 0
        self._last_kv_restore_at = 0.0
        self._dialogue_prompt_dumped = False
        self._model_calls = {
            "dialogue": 0, "deliberation": 0, "reflection": 0,
            "inner_life": 0, "autonomy": 0, "other": 0,
        }
        if SETTINGS.prompt_debug:
            print(f"[Akane:debug:runtime] instance_id={id(self):x}", flush=True)

    @classmethod
    def get_instance(cls) -> "InferenceRuntime":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def status(self) -> dict[str, object]:
        with self._condition:
            return {
                "loading": self._loading,
                "loaded": self._llm is not None,
                "error": str(self._load_error) if self._load_error else None,
                "backend": "llama_cpp",
                "local_model_path": str(self._path),
                "inference_active": self._active,
                "inference_priority": self._active_priority,
                "foreground_active": self._foreground_requests > 0,
                "foreground_requests": self._foreground_requests,
                "last_foreground_activity_at": self._last_foreground_activity_at,
                "background_idle_grace_seconds": SETTINGS.background_idle_grace_seconds,
                "last_queue_wait_seconds": self._last_wait,
                "model_load_count": self._load_count,
                "gemma_thinking_disabled": self._gemma_thinking_disabled,
                "foreground_kv_preservation": "llama single-sequence state",
                "last_foreground_kv_snapshot_bytes": self._last_kv_snapshot_bytes,
                "last_foreground_kv_restore_at": self._last_kv_restore_at,
                "model_calls": dict(self._model_calls),
            }

    def inference_busy(self) -> bool:
        with self._condition:
            return self._active or bool(self._waiters)

    def model_call_counts(self) -> dict[str, int]:
        with self._condition:
            return dict(self._model_calls)

    def _record_model_call(self, kind: str) -> None:
        name = kind if kind in self._model_calls else "other"
        with self._condition:
            self._model_calls[name] += 1
            count = self._model_calls[name]
        if SETTINGS.timing_enabled:
            label = {
                "dialogue": "FOREGROUND_DIALOGUE",
                "deliberation": "FOREGROUND_DELIBERATION",
                "reflection": "REFLECTION",
                "inner_life": "INNER_LIFE",
                "autonomy": "AUTONOMY",
                "other": "OTHER",
            }[name]
            print(
                f"[Akane:model_call] kind={label} count={count} "
                f"runtime_id={id(self):x} at={time.time():.6f}",
                flush=True,
            )

    def notify_foreground(self) -> None:
        """Request cancellation of active optional work at its next token boundary."""

        with self._condition:
            if self._active_priority in {"reflection", "autonomy"}:
                if self._active_preemption is not None:
                    self._active_preemption.set()
            self._condition.notify_all()

    def foreground_started(self, *, now: float | None = None) -> None:
        """Mark the whole visible request active, including context and persistence."""

        with self._condition:
            self._foreground_requests += 1
            self._last_foreground_activity_at = time.time() if now is None else float(now)
            if self._active_priority in {"reflection", "autonomy"}:
                if self._active_preemption is not None:
                    self._active_preemption.set()
            self._condition.notify_all()

    def foreground_finished(self, *, now: float | None = None) -> None:
        with self._condition:
            self._foreground_requests = max(0, self._foreground_requests - 1)
            self._last_foreground_activity_at = time.time() if now is None else float(now)
            self._condition.notify_all()

    def background_allowed(self, *, now: float | None = None) -> bool:
        current = time.time() if now is None else float(now)
        with self._condition:
            foreground_waiting = any(rank <= 1 for rank, _ in self._waiters)
            foreground_model_active = self._active_priority in {"owner", "visible", "guest"}
            recent = bool(self._last_foreground_activity_at) and (
                current - self._last_foreground_activity_at
                < SETTINGS.background_idle_grace_seconds
            )
            return not (
                self._foreground_requests
                or foreground_waiting
                or foreground_model_active
                or recent
            )

    def runtime_report(self, *, include_model_hash: bool = False) -> dict[str, object]:
        path = self._path.expanduser().resolve()
        digest = "not requested"
        if include_model_hash and path.is_file():
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        try:
            binding = importlib.metadata.version("llama-cpp-python")
        except importlib.metadata.PackageNotFoundError:
            binding = "not installed"
        effective: dict[str, object] = {}
        try:
            from llama_cpp import Llama

            effective = self._load_options(Llama)
            signature = inspect.signature(Llama.__init__).parameters
            for name in ("numa", "type_k", "type_v", "seed"):
                if name in signature and name not in effective:
                    effective[name] = signature[name].default
        except (ImportError, TypeError, ValueError):
            pass
        effective.pop("model_path", None)
        llm = self._llm
        return {
            "model_path": str(path),
            "model_size": path.stat().st_size if path.is_file() else 0,
            "model_sha256": digest,
            "llama_cpp_python": binding,
            "context_window": SETTINGS.llama_context_window,
            "temperature": SETTINGS.temperature,
            "top_p": SETTINGS.top_p,
            "top_k": SETTINGS.top_k,
            "min_p": SETTINGS.min_p,
            "repeat_penalty": SETTINGS.repetition_penalty,
            "max_generation_tokens": SETTINGS.max_tokens,
            "stop_sequences": SETTINGS.generation_stop_sequences,
            "effective_init_options": effective,
            "chat_format": getattr(llm, "chat_format", None) if llm is not None else "embedded model template",
            "eos_token": llm.token_eos() if llm is not None else "model default",
            "seed_behavior": "llama.cpp automatic random seed sentinel on every request",
            "live_kv_prefix_reuse": "llama-cpp-python longest-prefix reuse",
            "background_kv_preservation": "single-sequence save/restore; one temporary snapshot",
            "response_cache_enabled": False,
            "gemma_thinking_disabled": self._gemma_thinking_disabled,
            "model_load_count": self._load_count,
        }

    def _load_options(self, llama_type) -> dict[str, object]:
        options: dict[str, object] = {
            "model_path": str(self._path),
            "n_ctx": SETTINGS.llama_context_window,
            "n_batch": SETTINGS.llama_batch_size,
            "n_ubatch": SETTINGS.llama_ubatch_size,
            "n_threads": SETTINGS.llama_threads,
            "n_threads_batch": SETTINGS.llama_threads_batch,
            "flash_attn": SETTINGS.llama_flash_attn,
            "n_gpu_layers": SETTINGS.llama_gpu_layers,
            "offload_kqv": SETTINGS.llama_offload_kqv,
            "op_offload": SETTINGS.llama_op_offload,
            "use_mmap": SETTINGS.llama_use_mmap,
            "use_mlock": SETTINGS.llama_use_mlock,
            "swa_full": SETTINGS.llama_swa_full,
            "embedding": False,
            "logits_all": False,
            "verbose": False,
        }
        try:
            supported = inspect.signature(llama_type.__init__).parameters
        except (TypeError, ValueError):
            return options
        return {key: value for key, value in options.items() if key in supported}

    def ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        with self._load_lock:
            if self._llm is not None:
                return
            self._loading = True
            self._load_error = None
            try:
                from llama_cpp import Llama

                self._llm = Llama(**self._load_options(Llama))
                self._load_count += 1
                metadata = getattr(self._llm, "metadata", {}) or {}
                model_text = " ".join((str(self._path), *(str(value) for value in metadata.values()))).casefold()
                template = metadata.get("tokenizer.chat_template") or metadata.get("tokenizer.ggml.chat_template")
                if "gemma" in model_text and not template:
                    raise RuntimeError("The configured Gemma GGUF has no embedded chat template.")
                if "gemma" in model_text:
                    formatter = _chat_formatter(self._llm)
                    if formatter is None and "enable_thinking" in str(template):
                        raise RuntimeError("Could not disable thinking in the configured Gemma chat template.")
                    if formatter is not None:
                        formatter._environment.globals["enable_thinking"] = False
                    self._gemma_thinking_disabled = True
                if SETTINGS.prompt_debug:
                    self._log_character_sources(self._llm)
            except Exception as exc:
                self._llm = None
                self._gemma_thinking_disabled = False
                self._load_error = exc
                raise
            finally:
                self._loading = False

    def close(self) -> None:
        with self._load_lock:
            llm, self._llm = self._llm, None
            self._gemma_thinking_disabled = False
            self._live_cache_owner = None
        close = getattr(llm, "close", None)
        if close is not None:
            close()

    @property
    def llm(self):
        if self._load_error is not None:
            raise RuntimeError(f"Model failed to load: {self._load_error}") from self._load_error
        self.ensure_loaded()
        return self._llm

    @staticmethod
    def _log_character_sources(llm) -> None:
        character = load_character_profile()
        contents = (
            (
                "identity",
                character.identity_path,
                character.identity_mtime_ns,
                character.identity_sha256,
                character.identity,
            ),
            (
                "soul",
                character.soul_path,
                character.soul_mtime_ns,
                character.soul_sha256,
                character.voice,
            ),
        )
        for label, path, mtime_ns, digest, content in contents:
            tokens = len(llm.tokenize(content.encode("utf-8"), add_bos=False, special=True))
            print(
                f"[Akane:debug:character] {label}_path={str(path)!r} "
                f"{label}_mtime_ns={mtime_ns} {label}_sha256={digest} "
                f"{label}_tokens={tokens}",
                flush=True,
            )
        combined = f"{character.identity}\n\n{character.voice}"
        combined_tokens = len(llm.tokenize(
            combined.encode("utf-8"), add_bos=False, special=True,
        ))
        print(
            f"[Akane:debug:character] content_sha256={character.content_sha256} "
            f"character_combined_tokens={combined_tokens} "
            "reload_semantics='cached until identity or soul source mtime changes'",
            flush=True,
        )

    @staticmethod
    def _clear_model_state(llm) -> None:
        llm._ctx.kv_cache_clear()
        llm.n_tokens = 0

    def _prepare_dialogue_cache(self, llm, owner: DialogueCacheKey) -> None:
        if self._live_cache_owner == owner:
            return
        if int(getattr(llm, "n_tokens", 0)) > 0:
            self._clear_model_state(llm)
        self._live_cache_owner = None

    def _capture_foreground_sequence(self, llm) -> _ForegroundSequenceState | None:
        owner = self._live_cache_owner
        n_tokens = int(getattr(llm, "n_tokens", 0))
        if owner is None or n_tokens <= 0:
            return None
        try:
            from llama_cpp import llama_cpp

            size = int(llama_cpp.llama_state_seq_get_size(llm._ctx.ctx, 0))
            if size <= 0:
                raise RuntimeError("llama returned an empty sequence state")
            data = bytearray(size)
            buffer = (ctypes.c_uint8 * size).from_buffer(data)
            written = int(llama_cpp.llama_state_seq_get_data(llm._ctx.ctx, buffer, size, 0))
            if written != size:
                raise RuntimeError(f"llama copied {written} of {size} sequence-state bytes")
            state = _ForegroundSequenceState(
                owner,
                llm.input_ids[:n_tokens].copy(),
                n_tokens,
                getattr(llm, "_seed", None),
                self._cache_epoch,
                data,
            )
        except (AttributeError, ImportError, TypeError, ValueError, RuntimeError) as exc:
            raise InferencePreempted(
                "Background inference was deferred because foreground KV state could not be preserved."
            ) from exc
        self._last_kv_snapshot_bytes = size
        if SETTINGS.prompt_debug:
            print(
                f"[Akane:debug:kv] action=save owner={self._cache_owner_label(owner)} "
                f"tokens={n_tokens} bytes={size}",
                flush=True,
            )
        return state

    def _restore_foreground_sequence(self, llm, state: _ForegroundSequenceState) -> None:
        try:
            from llama_cpp import llama_cpp

            self._clear_model_state(llm)
            if state.cache_epoch != self._cache_epoch:
                return
            size = len(state.data)
            buffer = (ctypes.c_uint8 * size).from_buffer(state.data)
            restored = int(llama_cpp.llama_state_seq_set_data(llm._ctx.ctx, buffer, size, 0))
            if restored != size:
                raise RuntimeError(f"llama restored {restored} of {size} sequence-state bytes")
            llm.input_ids[:state.n_tokens] = state.input_ids
            llm.n_tokens = state.n_tokens
            if state.seed is not None:
                llm._seed = state.seed
            self._live_cache_owner = state.owner
            self._last_kv_restore_at = time.time()
        except (AttributeError, ImportError, TypeError, ValueError, RuntimeError):
            self._live_cache_owner = None
            raise
        if SETTINGS.prompt_debug:
            print(
                f"[Akane:debug:kv] action=restore owner={self._cache_owner_label(state.owner)} "
                f"tokens={state.n_tokens} bytes={len(state.data)}",
                flush=True,
            )

    @staticmethod
    def _cache_owner_label(owner: DialogueCacheKey | None) -> str:
        if owner is None:
            return "none"
        return hashlib.sha256("\0".join(owner).encode("utf-8")).hexdigest()[:12]

    def discard_dialogue_cache(self, profile_id: str, conversation_id: str | None = None) -> None:
        with self._condition:
            owner = self._live_cache_owner
            if owner is not None and owner[0] == profile_id and (
                conversation_id is None or owner[1] == conversation_id
            ):
                self._live_cache_owner = None
                self._cache_epoch += 1

    @contextmanager
    def reserve(
        self,
        *,
        priority: str = "visible",
        cancellation: threading.Event | None = None,
        queue_deadline: float | None = None,
    ):
        ranks = {"owner": 0, "visible": 0, "guest": 1, "reflection": 2, "autonomy": 3}
        if priority not in ranks:
            raise ValueError("Unknown inference priority.")
        if ranks[priority] <= ranks["guest"]:
            self.notify_foreground()
        queued_at = time.monotonic()
        preemption = threading.Event()
        waiter: tuple[int, int] | None = None
        acquired = False
        preserved: _ForegroundSequenceState | None = None
        with self._condition:
            if ranks[priority] > ranks["guest"] and not self.background_allowed():
                raise InferencePreempted("Background inference requires an idle foreground window.")
            self._sequence += 1
            waiter = (ranks[priority], self._sequence)
            self._waiters.append(waiter)
            try:
                while self._active or waiter != min(self._waiters):
                    if cancellation is not None and cancellation.is_set():
                        raise InferenceCancelled("Inference was cancelled while queued.")
                    if queue_deadline is not None and time.monotonic() >= queue_deadline:
                        raise InferenceQueueTimeout("Inference timed out while queued.")
                    self._condition.wait(0.1)
                if ranks[priority] > ranks["guest"] and not self.background_allowed():
                    raise InferencePreempted("Background inference yielded to foreground work.")
                self._waiters.remove(waiter)
                waiter = None
                if cancellation is not None and cancellation.is_set():
                    raise InferenceCancelled("Inference was cancelled before it started.")
                self._active = True
                self._active_priority = priority
                self._active_preemption = preemption
                if ranks[priority] <= ranks["guest"]:
                    self._last_foreground_activity_at = time.time()
                self._last_wait = time.monotonic() - queued_at
                acquired = True
            finally:
                if waiter is not None and waiter in self._waiters:
                    self._waiters.remove(waiter)
                if not acquired:
                    self._condition.notify_all()
        try:
            llm = self.llm
            if ranks[priority] > ranks["guest"]:
                preserved = self._capture_foreground_sequence(llm)
            yield Reservation(llm, priority, preemption)
        finally:
            try:
                if preserved is not None:
                    self._restore_foreground_sequence(self._llm, preserved)
            finally:
                with self._condition:
                    self._active = False
                    self._active_priority = ""
                    self._active_preemption = None
                    if ranks[priority] <= ranks["guest"]:
                        self._last_foreground_activity_at = time.time()
                    self._condition.notify_all()

    def count_prompt_tokens(
        self,
        messages: tuple[dict[str, str], ...] | list[dict[str, str]],
        reservation: Reservation,
    ) -> tuple[int, str]:
        prompt = _render_chat_prompt(reservation.llm, messages)
        if prompt is None:
            characters = sum(len(message.get("content", "")) for message in messages)
            return max(1, characters // 4), "estimated_characters"
        try:
            tokens = reservation.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
            return len(tokens), "exact_active_chat_template"
        except (AttributeError, TypeError, ValueError):
            characters = sum(len(message.get("content", "")) for message in messages)
            return max(1, characters // 4), "estimated_characters"

    def debug_dialogue_prompt_once(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        llm,
        token_sections: dict[str, str],
        cache_key: DialogueCacheKey | None,
    ) -> None:
        if not SETTINGS.prompt_debug:
            return
        with self._condition:
            if self._dialogue_prompt_dumped:
                return
            self._dialogue_prompt_dumped = True
        prompt = _render_chat_prompt(llm, messages)
        if prompt is None:
            print(
                "[Akane:debug:dialogue_prompt_once] rendered_prompt=unavailable",
                flush=True,
            )
            return
        section_tokens = {
            name: len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))
            for name, text in token_sections.items()
            if text
        }
        prompt_tokens = len(llm.tokenize(
            prompt.encode("utf-8"), add_bos=False, special=True,
        ))
        payload = {
            "messages": list(messages),
            "section_tokens": section_tokens,
            "final_prompt_tokens": prompt_tokens,
            "final_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "cache_owner": self._cache_owner_label(cache_key),
            "rendered_prompt": prompt,
        }
        print(
            "[Akane:debug:dialogue_prompt_once] "
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )

    @staticmethod
    def count_text_tokens(text: str, reservation: Reservation) -> tuple[int, str]:
        try:
            return len(reservation.llm.tokenize(
                str(text).encode("utf-8"), add_bos=False, special=True,
            )), "exact_model_tokenizer"
        except (AttributeError, TypeError, ValueError):
            return max(0, len(str(text)) // 4), "estimated_characters"

    @staticmethod
    def _reset_backend_timings(llm) -> None:
        try:
            llm._ctx.reset_timings()
        except AttributeError:
            pass

    @staticmethod
    def _capture_backend_timings(llm, timing: InferenceTiming, stream_chunks: int) -> None:
        try:
            from llama_cpp import llama_cpp

            perf = llama_cpp.llama_perf_context(llm._ctx.ctx)
            timing.prefill_tokens = max(0, int(perf.n_p_eval))
            timing.new_prompt_eval_tokens = timing.prefill_tokens
            timing.reused_prefix_tokens = max(0, timing.prompt_tokens - timing.prefill_tokens)
            timing.generated_tokens = max(0, int(perf.n_eval))
            timing.prefill_seconds = max(0.0, float(perf.t_p_eval_ms) / 1000.0)
            timing.decode_seconds = max(0.0, float(perf.t_eval_ms) / 1000.0)
            timing.generated_token_method = "llama_perf_context"
        except (AttributeError, ImportError, TypeError, ValueError):
            if not timing.generated_tokens:
                timing.generated_tokens = stream_chunks
                timing.generated_token_method = "stream_chunks"
            if not timing.prefill_seconds and timing.first_token_at:
                timing.prefill_seconds = max(0.0, timing.first_token_at - timing.model_started_at)
            if not timing.decode_seconds and timing.final_token_at:
                timing.decode_seconds = max(0.0, timing.final_token_at - timing.first_token_at)

    @staticmethod
    def _generation_type(call_kind: str) -> str:
        return "reasoning" if call_kind == "deliberation" else call_kind

    def _log_inference_debug(
        self,
        call_kind: str,
        timing: InferenceTiming,
        cache_key: DialogueCacheKey | None,
    ) -> None:
        if not SETTINGS.prompt_debug:
            return
        decode_tok_s = (
            timing.generated_tokens / timing.decode_seconds
            if timing.generated_tokens > 0 and timing.decode_seconds > 0
            else 0.0
        )
        reused = (
            timing.reused_prefix_tokens
            if timing.reused_prefix_tokens is not None
            else "unavailable"
        )
        new_tokens = (
            timing.new_prompt_eval_tokens
            if timing.new_prompt_eval_tokens is not None
            else "unavailable"
        )
        print(
            f"[Akane:debug:inference] generation_type={self._generation_type(call_kind)} "
            f"prompt_tokens={timing.prompt_tokens} reused_prefix_tokens={reused} "
            f"new_prompt_eval_tokens={new_tokens} prompt_eval_ms={timing.prefill_seconds * 1000:.3f} "
            f"decode_tok_s={decode_tok_s:.3f} cache_owner={self._cache_owner_label(cache_key)}",
            flush=True,
        )

    @staticmethod
    def _options(max_tokens: int, *, temperature: float | None = None) -> dict[str, object]:
        from llama_cpp import LLAMA_DEFAULT_SEED

        options: dict[str, object] = {
            "max_tokens": max(1, min(int(max_tokens), SETTINGS.llama_context_window - 1)),
            "temperature": SETTINGS.temperature if temperature is None else temperature,
            "top_k": SETTINGS.top_k,
            "top_p": SETTINGS.top_p,
            "min_p": SETTINGS.min_p,
            "repeat_penalty": SETTINGS.repetition_penalty,
            # Omitting this lets the Python binding derive a repeatable first seed
            # in a fresh process. The backend sentinel requests automatic entropy.
            "seed": LLAMA_DEFAULT_SEED,
        }
        if SETTINGS.generation_stop_sequences:
            options["stop"] = list(SETTINGS.generation_stop_sequences)
        return options

    def stream_messages(
        self,
        messages: tuple[dict[str, str], ...] | list[dict[str, str]],
        *,
        max_tokens: int,
        reservation: Reservation,
        cancellation: threading.Event | None = None,
        timing: InferenceTiming | None = None,
        temperature: float | None = None,
        call_kind: str = "other",
        cache_key: DialogueCacheKey | None = None,
    ):
        if timing is not None and not timing.prompt_tokens:
            timing.prompt_tokens, timing.prompt_token_method = self.count_prompt_tokens(messages, reservation)

        def check_cancelled() -> None:
            if cancellation is not None and cancellation.is_set():
                raise InferenceCancelled("Inference was cancelled during generation.")
            if reservation.preemption.is_set():
                raise InferencePreempted("Background inference yielded to foreground work.")

        response = None
        stream_chunks = 0
        completed = False
        try:
            check_cancelled()
            if call_kind == "dialogue" and cache_key is not None:
                self._prepare_dialogue_cache(reservation.llm, cache_key)
            if timing is not None:
                self._reset_backend_timings(reservation.llm)
                timing.model_started_at = time.perf_counter()
            self._record_model_call(call_kind)
            response = reservation.llm.create_chat_completion(
                messages=list(messages),
                stream=True,
                **self._options(max_tokens, temperature=temperature),
            )
            for chunk in response:
                check_cancelled()
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                text = _content((choice.get("delta") or {}).get("content") or choice.get("text"))
                if not text:
                    continue
                stream_chunks += 1
                if timing is not None:
                    token_at = time.perf_counter()
                    if not timing.first_token_at:
                        timing.first_token_at = token_at
                    timing.final_token_at = token_at
                yield text
            completed = True
        finally:
            if timing is not None and timing.model_started_at:
                self._capture_backend_timings(reservation.llm, timing, stream_chunks)
                timing.model_finished_at = time.perf_counter()
                self._log_inference_debug(call_kind, timing, cache_key)
            close = getattr(response, "close", None)
            if close is not None:
                close()
            if completed and call_kind == "dialogue" and cache_key is not None:
                self._live_cache_owner = cache_key

    def complete_messages(
        self,
        messages: tuple[dict[str, str], ...] | list[dict[str, str]],
        *,
        max_tokens: int,
        reservation: Reservation,
        cancellation: threading.Event | None = None,
        timing: InferenceTiming | None = None,
        temperature: float = 0.2,
        call_kind: str = "other",
    ) -> str:
        return "".join(
            self.stream_messages(
                messages,
                max_tokens=max_tokens,
                reservation=reservation,
                cancellation=cancellation,
                timing=timing,
                temperature=temperature,
                call_kind=call_kind,
            )
        ).strip()
