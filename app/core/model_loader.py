"""Singleton llama.cpp model lifecycle and inference runtime."""

from __future__ import annotations

import inspect
import hashlib
import importlib.metadata
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from app.core.config import (
    GENERATION_STOP_SEQUENCES,
    LLAMA_BATCH_SIZE,
    LLAMA_CONTEXT_WINDOW,
    LLAMA_FLASH_ATTN,
    LLAMA_GPU_LAYERS,
    LLAMA_LAST_N_TOKENS_SIZE,
    LLAMA_OFFLOAD_KQV,
    LLAMA_OP_OFFLOAD,
    LLAMA_SWA_FULL,
    LLAMA_THREADS,
    LLAMA_THREADS_BATCH,
    LLAMA_UBATCH_SIZE,
    LLAMA_USE_MLOCK,
    LLAMA_USE_MMAP,
    MAX_TOKENS,
    MIN_P,
    MODEL_PATH,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)

_STOP_SEQUENCES = list(GENERATION_STOP_SEQUENCES)
_OPTIONAL_LOAD_ARGUMENTS = {
    "flash_attn",
    "last_n_tokens_size",
    "n_threads_batch",
    "n_ubatch",
    "no_perf",
    "offload_kqv",
    "op_offload",
    "swa_full",
}
_STREAM_QUEUE_SIZE = MAX_TOKENS + 2
_STREAM_END = object()


class InferenceCancelled(RuntimeError):
    pass


class InferenceQueueTimeout(RuntimeError):
    pass


@dataclass(slots=True)
class InferenceTiming:
    requested_at: float
    model_started_at: float = 0.0
    first_token_at: float = 0.0
    model_finished_at: float = 0.0
    prompt_tokens: int = 0


@dataclass(frozen=True, slots=True)
class _StreamFailure:
    error: Exception


@dataclass(frozen=True, slots=True)
class InferenceReservation:
    """Exclusive access to the shared llama.cpp runtime for one complete job."""

    llm: object
    priority: str


def content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def completion_kwargs(max_tokens: int, stream: bool) -> dict[str, object]:
    options: dict[str, object] = {
        "max_tokens": max(1, min(int(max_tokens), LLAMA_CONTEXT_WINDOW - 1)),
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "min_p": MIN_P,
        "repeat_penalty": REPETITION_PENALTY,
        "stream": stream,
    }
    if _STOP_SEQUENCES:
        options["stop"] = _STOP_SEQUENCES
    return options


@lru_cache(maxsize=8)
def compile_json_grammar(json_schema: str) -> object:
    from llama_cpp import LlamaGrammar

    return LlamaGrammar.from_json_schema(json_schema, verbose=False)


def _model_identifier_text(llm, model_path: Path) -> str:
    values = [str(model_path), str(getattr(llm, "model_path", ""))]
    metadata = getattr(llm, "metadata", {}) or {}
    if isinstance(metadata, dict):
        values.extend(str(value) for value in metadata.values())
    return " ".join(values).lower()


def _compact_text(*values: object) -> str:
    return "".join(char for value in values for char in str(value).lower() if char.isalnum())


def _is_gemma_model(llm, model_path: Path) -> bool:
    return "gemma" in _compact_text(_model_identifier_text(llm, model_path))


def _has_embedded_chat_template(llm) -> bool:
    metadata = getattr(llm, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return False
    for key in ("tokenizer.chat_template", "tokenizer.ggml.chat_template"):
        if str(metadata.get(key) or "").strip():
            return True
    return False


def _friendly_load_error(exc: Exception, model_path: Path) -> Exception:
    message = str(exc).lower()
    if "unknown model architecture" in message and "gemma" in _compact_text(message, model_path):
        return RuntimeError(
            "Installed llama-cpp-python does not support this Gemma model. "
            "Upgrade llama-cpp-python to a build with Gemma 4 support."
        )
    return exc


def _resolved_chat_formatter(llm):
    """Resolve the formatter used by llama.cpp's active chat-completion handler.

    The installed binding wraps its embedded Jinja template formatter in a handler
    closure. Resolving that exact object preserves its role markers, special-token
    behavior, and generation prefix without duplicating a model-specific template.
    Custom handlers are deliberately treated as unavailable rather than guessed.
    """

    try:
        from llama_cpp import llama_chat_format

        handler = (
            llm.chat_handler
            or llm._chat_handlers.get(llm.chat_format)
            or llama_chat_format.get_chat_completion_handler(llm.chat_format)
        )
        for cell in handler.__closure__ or ():
            value = cell.cell_contents
            if isinstance(value, llama_chat_format.Jinja2ChatFormatter):
                return value
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return None


class ModelManager:
    _instance: "ModelManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._local_model_path = Path(MODEL_PATH)
        self._llm = None
        self._loading = False
        self._load_error: Exception | None = None
        self._load_lock = threading.RLock()
        self._inference_condition = threading.Condition()
        self._inference_active = False
        self._inference_active_priority = ""
        self._inference_active_cancellation: threading.Event | None = None
        self._inference_waiters: list[tuple[int, int]] = []
        self._inference_waiter_sequence = 0
        self._visible_waiters = 0

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def status(self) -> dict[str, object]:
        return {
            "loading": self._loading,
            "loaded": self._llm is not None,
            "error": str(self._load_error) if self._load_error else None,
            "backend": "llama_cpp",
            "local_model_path": str(self._local_model_path),
        }

    def inference_busy(self) -> bool:
        with self._inference_condition:
            return self._inference_active or bool(self._inference_waiters)

    def runtime_report(self, *, include_model_hash: bool = False) -> dict[str, object]:
        path = self._local_model_path.expanduser().resolve()
        model_hash = "not requested"
        if include_model_hash and path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            model_hash = digest.hexdigest()
        llm = self._llm
        metadata = getattr(llm, "metadata", {}) if llm is not None else {}
        template = ""
        if isinstance(metadata, dict):
            template = str(
                metadata.get("tokenizer.chat_template")
                or metadata.get("tokenizer.ggml.chat_template")
                or ""
            )
        try:
            binding_version = importlib.metadata.version("llama-cpp-python")
        except importlib.metadata.PackageNotFoundError:
            binding_version = "not installed"
        return {
            "model_path": str(path),
            "model_size": path.stat().st_size if path.is_file() else 0,
            "model_sha256": model_hash,
            "llama_cpp_python": binding_version,
            "chat_template": hashlib.sha256(template.encode("utf-8")).hexdigest()[:12] if template else "unavailable until model load",
            "context_window": LLAMA_CONTEXT_WINDOW,
            "enable_thinking": False,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "min_p": MIN_P,
            "repeat_penalty": REPETITION_PENALTY,
            "seed": "backend random",
        }

    def _load_kwargs(self) -> dict[str, object]:
        n_batch = max(1, min(int(LLAMA_BATCH_SIZE), int(LLAMA_CONTEXT_WINDOW)))
        n_ubatch = max(1, min(int(LLAMA_UBATCH_SIZE), n_batch))
        kwargs: dict[str, object] = {
            "model_path": str(self._local_model_path),
            "n_ctx": LLAMA_CONTEXT_WINDOW,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "n_threads": LLAMA_THREADS or None,
            "n_threads_batch": LLAMA_THREADS_BATCH or None,
            "flash_attn": LLAMA_FLASH_ATTN,
            "offload_kqv": LLAMA_OFFLOAD_KQV,
            "op_offload": LLAMA_OP_OFFLOAD,
            "use_mmap": LLAMA_USE_MMAP,
            "use_mlock": LLAMA_USE_MLOCK,
            "last_n_tokens_size": max(0, int(LLAMA_LAST_N_TOKENS_SIZE)),
            "logits_all": False,
            "embedding": False,
            "no_perf": True,
            "verbose": False,
        }
        if LLAMA_GPU_LAYERS:
            kwargs["n_gpu_layers"] = LLAMA_GPU_LAYERS
        if LLAMA_SWA_FULL:
            kwargs["swa_full"] = True
        return kwargs

    def _resolved_load_kwargs(self, llama_type) -> dict[str, object]:
        """Filter version-dependent constructor options once before model creation."""

        kwargs = self._load_kwargs()
        try:
            supported = inspect.signature(llama_type.__init__).parameters
        except (TypeError, ValueError):
            return kwargs
        for name in _OPTIONAL_LOAD_ARGUMENTS - supported.keys():
            kwargs.pop(name, None)
        return kwargs

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

                kwargs = self._resolved_load_kwargs(Llama)
                print(
                    f"Loading model {self._local_model_path} "
                    f"n_ctx={kwargs['n_ctx']} n_batch={kwargs['n_batch']} "
                    f"n_ubatch={kwargs.get('n_ubatch', 'backend-default')} "
                    f"n_threads={kwargs['n_threads']} "
                    f"flash_attn={kwargs.get('flash_attn', False)} "
                    f"swa_full={kwargs.get('swa_full', False)}",
                    flush=True,
                )
                llm = Llama(**kwargs)
                self._validate_loaded_model(llm)
                self._llm = llm
            except Exception as exc:
                self._llm = None
                error = _friendly_load_error(exc, self._local_model_path)
                self._load_error = error
                if error is exc:
                    raise
                raise error from exc
            finally:
                self._loading = False

    def _validate_loaded_model(self, llm) -> None:
        if _is_gemma_model(llm, self._local_model_path) and not _has_embedded_chat_template(llm):
            raise RuntimeError(
                "Gemma GGUF is missing tokenizer.chat_template; use an instruction-tuned "
                "Gemma 4/E4B IT GGUF with the embedded chat template."
            )

    @property
    def llm(self):
        if self._load_error:
            raise RuntimeError(f"Model failed to load: {self._load_error}") from self._load_error
        self.ensure_loaded()
        return self._llm

    @contextmanager
    def reserve(
        self,
        *,
        priority: str = "visible",
        cancellation: threading.Event | None = None,
        queue_deadline: float | None = None,
    ):
        """Reserve tokenization and generation as one priority-aware operation."""

        ranks = {"owner": 0, "visible": 0, "guest": 1, "background": 2}
        normalized_priority = str(priority or "visible").strip().lower()
        if normalized_priority not in ranks:
            raise ValueError("Unknown inference priority.")
        visible = normalized_priority != "background"
        rank = ranks[normalized_priority]
        waiter: tuple[int, int] | None = None
        acquired = False
        with self._inference_condition:
            if cancellation is not None and cancellation.is_set():
                raise InferenceCancelled("Generation was cancelled before inference.")
            if queue_deadline is not None and time.monotonic() >= queue_deadline:
                raise InferenceQueueTimeout("Generation timed out while waiting for the model.")
            self._inference_waiter_sequence += 1
            waiter = (rank, self._inference_waiter_sequence)
            self._inference_waiters.append(waiter)
            if visible:
                self._visible_waiters += 1
            active_rank = ranks.get(self._inference_active_priority, 99)
            if (
                self._inference_active
                and rank < active_rank
                and self._inference_active_cancellation is not None
            ):
                self._inference_active_cancellation.set()
            try:
                while self._inference_active or waiter != min(self._inference_waiters):
                    if cancellation is not None and cancellation.is_set():
                        raise InferenceCancelled(
                            "Generation was cancelled before inference."
                        )
                    if (
                        queue_deadline is not None
                        and time.monotonic() >= queue_deadline
                    ):
                        raise InferenceQueueTimeout(
                            "Generation timed out while waiting for the model."
                        )
                    self._inference_condition.wait(timeout=0.1)
                self._inference_waiters.remove(waiter)
                waiter = None
                self._inference_active = True
                self._inference_active_priority = normalized_priority
                self._inference_active_cancellation = cancellation
                acquired = True
            finally:
                if waiter is not None and waiter in self._inference_waiters:
                    self._inference_waiters.remove(waiter)
                if visible:
                    self._visible_waiters -= 1
                if not acquired:
                    self._inference_condition.notify_all()
        try:
            if cancellation is not None and cancellation.is_set():
                raise InferenceCancelled("Generation was cancelled before inference.")
            reservation = InferenceReservation(
                self.llm,
                normalized_priority,
            )
            yield reservation
        finally:
            with self._inference_condition:
                self._inference_active = False
                self._inference_active_priority = ""
                self._inference_active_cancellation = None
                self._inference_condition.notify_all()

    def tokenize_prompt(
        self,
        messages: list[dict[str, str]],
        *,
        reservation: InferenceReservation,
    ):
        """Apply the active template with thinking disabled and return exact IDs."""

        from app.core.prompt import PromptTokenCount

        def tokenize(llm):
            formatter = _resolved_chat_formatter(llm)
            if formatter is None:
                raise RuntimeError(
                    "The active llama.cpp chat formatter is unavailable; exact Gemma "
                    "prompt tokenization cannot be guaranteed."
                )
            try:
                prompt = formatter._environment.render(
                    messages=messages,
                    eos_token=formatter.eos_token,
                    bos_token=formatter.bos_token,
                    raise_exception=lambda message: (_ for _ in ()).throw(ValueError(message)),
                    add_generation_prompt=formatter.add_generation_prompt,
                    functions=None,
                    function_call=None,
                    tools=None,
                    tool_choice=None,
                    strftime_now=formatter.strftime_now,
                    enable_thinking=False,
                )
            except AttributeError as exc:
                raise RuntimeError(
                    "The active chat formatter cannot expose its embedded template."
                ) from exc
            tokens = llm.tokenize(
                prompt.encode("utf-8"),
                add_bos=False,
                special=True,
            )
            return PromptTokenCount(
                tuple(tokens),
                "exact_active_chat_template_enable_thinking_false",
                (formatter.eos_token,) if formatter.eos_token else (),
            )
        return tokenize(reservation.llm)

    def _start_timing(
        self,
        timing: InferenceTiming | None,
        on_model_start: Callable[[], None] | None,
    ) -> None:
        if timing is not None:
            timing.model_started_at = time.perf_counter()
        if on_model_start is not None:
            on_model_start()

    def _finish_timing(self, timing: InferenceTiming | None) -> None:
        if timing is not None:
            timing.model_finished_at = time.perf_counter()

    def stream(
        self,
        *,
        prompt_tokens: tuple[int, ...],
        template_stop_sequences: tuple[str, ...] = (),
        max_tokens: int,
        cancellation: threading.Event | None = None,
        on_model_start: Callable[[], None] | None = None,
        timing: InferenceTiming | None = None,
        grammar: object | None = None,
        reservation: InferenceReservation,
    ):
        output: queue.Queue[object] = queue.Queue(maxsize=_STREAM_QUEUE_SIZE)
        stopped = threading.Event()

        def enqueue(item: object) -> bool:
            while not stopped.is_set():
                try:
                    output.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            response = None
            try:
                if not prompt_tokens:
                    raise RuntimeError("Exact prompt token IDs are required for inference.")
                if cancellation is not None and cancellation.is_set():
                    raise InferenceCancelled(
                        "Generation was cancelled before inference."
                    )
                options = completion_kwargs(max_tokens, True)
                if grammar is not None:
                    options["grammar"] = grammar
                configured_stops = list(options.pop("stop", []))
                stops = list(
                    dict.fromkeys(
                        (*template_stop_sequences, *configured_stops)
                    )
                )
                if timing is not None:
                    timing.prompt_tokens = len(prompt_tokens)
                self._start_timing(timing, on_model_start)
                response = reservation.llm.create_completion(
                    prompt=list(prompt_tokens),
                    stop=stops,
                    **options,
                )
                for chunk in response:
                    if cancellation is not None and cancellation.is_set():
                        raise InferenceCancelled("Generation was cancelled during inference.")
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    text = content_to_text(
                        choice.get("text") or (choice.get("delta") or {}).get("content")
                    )
                    if text and timing is not None and not timing.first_token_at:
                        timing.first_token_at = time.perf_counter()
                    if text and not enqueue(text):
                        break
            except Exception as exc:
                enqueue(_StreamFailure(exc))
            finally:
                if timing is not None and timing.model_started_at:
                    self._finish_timing(timing)
                close = getattr(response, "close", None)
                if close is not None:
                    close()
                enqueue(_STREAM_END)

        producer = threading.Thread(
            target=produce,
            daemon=True,
            name="AkaneInference",
        )
        producer.start()
        try:
            while True:
                item = output.get()
                if item is _STREAM_END:
                    break
                if isinstance(item, _StreamFailure):
                    raise item.error
                yield str(item)
        finally:
            stopped.set()
            producer.join()
