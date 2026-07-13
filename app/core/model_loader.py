"""Singleton llama.cpp model lifecycle and inference runtime."""

from __future__ import annotations

import inspect
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from app.core.config import (
    GENERATION_STOP_SEQUENCES,
    LLAMA_BATCH_SIZE,
    LLAMA_CONTEXT_WINDOW,
    LLAMA_FLASH_ATTN,
    LLAMA_GPU_LAYERS,
    LLAMA_LAST_N_TOKENS_SIZE,
    LLAMA_OFFLOAD_KQV,
    LLAMA_OP_OFFLOAD,
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

_THINKING_OFF_TEMPLATE_KWARGS = {"enable_thinking": False}


class InferenceCancelled(RuntimeError):
    pass


class InferenceQueueTimeout(RuntimeError):
    pass


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
    if GENERATION_STOP_SEQUENCES:
        options["stop"] = list(GENERATION_STOP_SEQUENCES)
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

                kwargs = self._load_kwargs()
                print(
                    f"Loading model {self._local_model_path} "
                    f"n_ctx={kwargs['n_ctx']} n_batch={kwargs['n_batch']} n_ubatch={kwargs['n_ubatch']}",
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

    def _completion_kwargs(self, llm, kwargs: dict[str, object]) -> dict[str, object]:
        if not _is_gemma_model(llm, self._local_model_path) or not _supports_chat_template_kwargs(llm):
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
                return llm.create_chat_completion(**self._completion_kwargs(llm, kwargs))

        def wrapped():
            with self.inference(cancellation, queue_deadline) as llm:
                yield from llm.create_chat_completion(**self._completion_kwargs(llm, kwargs))

        return wrapped()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        cancellation: threading.Event | None = None,
        queue_deadline: float | None = None,
    ) -> str:
        result = self.create_chat_completion(
            messages=messages,
            cancellation=cancellation,
            queue_deadline=queue_deadline,
            **completion_kwargs(max_tokens, False),
        )
        choices = result.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        return content_to_text(choice.get("text") or (choice.get("message") or {}).get("content"))

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        cancellation: threading.Event | None = None,
        queue_deadline: float | None = None,
    ):
        response = self.create_chat_completion(
            messages=messages,
            cancellation=cancellation,
            queue_deadline=queue_deadline,
            **completion_kwargs(max_tokens, True),
        )
        try:
            for chunk in response:
                if cancellation is not None and cancellation.is_set():
                    return
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                text = content_to_text(
                    choice.get("text") or (choice.get("delta") or {}).get("content")
                )
                if text:
                    yield text
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
