"""Singleton llama.cpp model lifecycle and inference runtime."""

from __future__ import annotations

import inspect
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
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
    LLAMA_WARMUP_STATIC_PROMPT,
    MAX_TOKENS,
    MIN_P,
    MODEL_PATH,
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)

_THINKING_OFF_TEMPLATE_KWARGS = {"enable_thinking": False}
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
    chat_template_seconds: float = 0.0
    prompt_tokenization_seconds: float = 0.0
    prompt_tokens: int = 0


@dataclass(frozen=True, slots=True)
class _StreamFailure:
    error: Exception


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
        "max_tokens": max(1, min(int(max_tokens), MAX_TOKENS)),
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


def _supports_chat_template_kwargs(llm) -> bool:
    try:
        parameters = inspect.signature(llm.create_chat_completion).parameters
    except (TypeError, ValueError):
        return False
    return "chat_template_kwargs" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


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
        self._inference_lock = threading.Lock()
        self._completion_capability_llm = None
        self._disable_thinking = False
        self._active_timing: InferenceTiming | None = None

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
                self._configure_completion_capabilities(llm)
                self._instrument_tokenizer(llm)
                self._warm_static_prompt(llm)
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

    def _warm_static_prompt(self, llm) -> None:
        if not LLAMA_WARMUP_STATIC_PROMPT:
            return
        from app.core.character import get_static_system_prompt

        started_at = time.perf_counter()
        print("Warming static prompt prefix", flush=True)
        kwargs = completion_kwargs(1, False)
        kwargs.update(
            {
                "messages": [
                    {"role": "system", "content": get_static_system_prompt()},
                    {"role": "user", "content": " "},
                ],
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "min_p": 0.0,
                "repeat_penalty": 1.0,
            }
        )
        llm.create_chat_completion(**self._completion_kwargs(llm, kwargs))
        print(
            f"Static prompt prefix ready in {time.perf_counter() - started_at:.2f}s",
            flush=True,
        )

    @property
    def llm(self):
        if self._load_error:
            raise RuntimeError(f"Model failed to load: {self._load_error}") from self._load_error
        self.ensure_loaded()
        return self._llm

    @contextmanager
    def inference(
        self,
        cancellation: threading.Event | None = None,
        queue_deadline: float | None = None,
    ):
        while not self._inference_lock.acquire(timeout=0.1):
            if cancellation is not None and cancellation.is_set():
                raise InferenceCancelled("Generation was cancelled before inference.")
            if queue_deadline is not None and time.monotonic() >= queue_deadline:
                raise InferenceQueueTimeout("Generation timed out while waiting for the model.")
        try:
            if cancellation is not None and cancellation.is_set():
                raise InferenceCancelled("Generation was cancelled before inference.")
            yield self.llm
        finally:
            self._inference_lock.release()

    def _configure_completion_capabilities(self, llm) -> None:
        if self._completion_capability_llm is not llm:
            self._disable_thinking = _is_gemma_model(
                llm,
                self._local_model_path,
            ) and _supports_chat_template_kwargs(llm)
            self._completion_capability_llm = llm

    def _instrument_tokenizer(self, llm) -> None:
        """Time the backend's existing tokenizer without another tokenization pass."""

        original = getattr(llm, "tokenize", None)
        if original is None:
            return

        def timed_tokenize(*args, **kwargs):
            started_at = time.perf_counter()
            timing = self._active_timing
            if timing is not None and timing.prompt_tokens == 0:
                timing.chat_template_seconds = max(
                    0.0,
                    started_at - timing.model_started_at,
                )
            tokens = original(*args, **kwargs)
            if timing is not None:
                timing.prompt_tokenization_seconds += time.perf_counter() - started_at
                if timing.prompt_tokens == 0:
                    try:
                        timing.prompt_tokens = len(tokens)
                    except TypeError:
                        pass
            return tokens

        llm.tokenize = timed_tokenize

    def count_prompt_tokens(self, messages: list[dict[str, str]]):
        """Count the exact prompt produced by the active llama.cpp chat formatter.

        A labeled unavailable result lets the canonical builder retain its
        conservative estimate when a binding uses an opaque/custom chat handler.
        """

        from app.core.prompt import PromptTokenCount

        with self.inference() as llm:
            self._configure_completion_capabilities(llm)
            if self._disable_thinking:
                return PromptTokenCount(
                    None,
                    False,
                    "estimated_chat_template_options_not_exposed",
                )
            formatter = _resolved_chat_formatter(llm)
            if formatter is None:
                return PromptTokenCount(
                    None,
                    False,
                    "estimated_chat_handler_formatter_not_exposed",
                )
            rendered = formatter(messages=messages)
            tokens = llm.tokenize(
                rendered.prompt.encode("utf-8"),
                add_bos=not rendered.added_special,
                special=True,
            )
            return PromptTokenCount(
                len(tokens),
                True,
                "exact_llama_cpp_active_chat_formatter",
            )

    def _start_timing(
        self,
        timing: InferenceTiming | None,
        on_model_start: Callable[[], None] | None,
    ) -> None:
        if timing is not None:
            timing.model_started_at = time.perf_counter()
            self._active_timing = timing
        if on_model_start is not None:
            on_model_start()

    def _finish_timing(self, timing: InferenceTiming | None) -> None:
        if timing is not None:
            timing.model_finished_at = time.perf_counter()
            self._active_timing = None

    def _completion_kwargs(self, llm, kwargs: dict[str, object]) -> dict[str, object]:
        self._configure_completion_capabilities(llm)
        if not self._disable_thinking:
            return kwargs
        updated = dict(kwargs)
        template_kwargs = dict(updated.get("chat_template_kwargs") or {})
        template_kwargs.update(_THINKING_OFF_TEMPLATE_KWARGS)
        updated["chat_template_kwargs"] = template_kwargs
        return updated

    def create_chat_completion(
        self,
        *,
        messages,
        max_tokens,
        temperature,
        top_k=None,
        top_p=None,
        min_p=None,
        repeat_penalty=None,
        stop=None,
        stream=False,
        cancellation: threading.Event | None = None,
        queue_deadline: float | None = None,
        on_model_start: Callable[[], None] | None = None,
        timing: InferenceTiming | None = None,
    ):
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
            "stream": stream,
        }
        if stop:
            kwargs["stop"] = stop
        if not stream:
            with self.inference(cancellation, queue_deadline) as llm:
                resolved_kwargs = self._completion_kwargs(llm, kwargs)
                try:
                    self._start_timing(timing, on_model_start)
                    return llm.create_chat_completion(**resolved_kwargs)
                finally:
                    self._finish_timing(timing)

        def wrapped():
            with self.inference(cancellation, queue_deadline) as llm:
                resolved_kwargs = self._completion_kwargs(llm, kwargs)
                try:
                    self._start_timing(timing, on_model_start)
                    yield from llm.create_chat_completion(**resolved_kwargs)
                finally:
                    self._finish_timing(timing)

        return wrapped()

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        cancellation: threading.Event | None = None,
        queue_deadline: float | None = None,
        on_model_start: Callable[[], None] | None = None,
        timing: InferenceTiming | None = None,
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
                response = self.create_chat_completion(
                    messages=messages,
                    cancellation=cancellation,
                    queue_deadline=queue_deadline,
                    on_model_start=on_model_start,
                    timing=timing,
                    **completion_kwargs(max_tokens, True),
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
