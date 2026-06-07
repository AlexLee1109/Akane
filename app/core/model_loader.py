"""Singleton llama.cpp model loader."""

from __future__ import annotations

import inspect
import threading
from pathlib import Path

from app.core.config import (
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
    MODEL_PATH,
)

_NO_THINK_DIRECTIVE = "/no_think"
_NO_THINK_ALIASES = ("/no_think", "/nothink")
_NO_THINK_TEMPLATE_KWARGS = {"enable_thinking": False}


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


def _is_qwen_model(llm, model_path: Path) -> bool:
    values = [str(model_path)]
    metadata = getattr(llm, "metadata", {}) or {}
    if isinstance(metadata, dict):
        values.extend(str(value) for value in metadata.values())
    return any("qwen" in value.lower() for value in values)


def _supports_chat_template_kwargs(llm) -> bool:
    try:
        parameters = inspect.signature(llm.create_chat_completion).parameters
    except (TypeError, ValueError):
        return False
    return "chat_template_kwargs" in parameters


def _has_no_think_directive(content) -> bool:
    text = content_to_text(content).lower()
    return any(alias in text for alias in _NO_THINK_ALIASES)


def _content_with_no_think(content):
    if _has_no_think_directive(content):
        return content
    if isinstance(content, str):
        value = content.rstrip()
        return f"{value}\n\n{_NO_THINK_DIRECTIVE}" if value else _NO_THINK_DIRECTIVE
    if isinstance(content, list):
        return [*content, {"type": "text", "text": f"\n{_NO_THINK_DIRECTIVE}"}]
    value = str(content or "").rstrip()
    return f"{value}\n\n{_NO_THINK_DIRECTIVE}" if value else _NO_THINK_DIRECTIVE


def _messages_with_no_think(messages):
    if not isinstance(messages, list):
        return messages
    copied = [dict(message) if isinstance(message, dict) else message for message in messages]
    for index in range(len(copied) - 1, -1, -1):
        message = copied[index]
        if isinstance(message, dict) and message.get("role") == "user":
            message["content"] = _content_with_no_think(message.get("content"))
            return copied
    for index, message in enumerate(copied):
        if isinstance(message, dict) and message.get("role") == "system":
            message["content"] = _content_with_no_think(message.get("content"))
            return copied
    return [{"role": "system", "content": _NO_THINK_DIRECTIVE}, *copied]


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
                self._llm = Llama(**kwargs)
            except Exception as exc:
                self._load_error = exc
                raise
            finally:
                self._loading = False

    @property
    def llm(self):
        if self._load_error:
            raise RuntimeError(f"Model failed to load: {self._load_error}") from self._load_error
        self.ensure_loaded()
        return self._llm

    def load_model(self):
        return self.llm

    def get_model(self):
        return self.llm

    def unload_local_model(self) -> None:
        with self._load_lock:
            self._llm = None

    def switch_backend(
        self,
        backend: str = "llama_cpp",
        *,
        local_model_path: str | None = None,
    ) -> dict[str, object]:
        if str(backend or "llama_cpp").strip().lower() != "llama_cpp":
            raise ValueError("Only the 'llama_cpp' backend is available.")
        if local_model_path is not None:
            value = str(local_model_path).strip()
            if not value:
                raise ValueError("Local model path cannot be empty.")
            path = Path(value)
            with self._load_lock:
                if path != self._local_model_path:
                    self._local_model_path = path
                    self._llm = None
                    self._load_error = None
        return self.status()

    def _completion_kwargs(self, llm, kwargs: dict[str, object]) -> dict[str, object]:
        if not _is_qwen_model(llm, self._local_model_path):
            return kwargs
        updated = dict(kwargs)
        if _supports_chat_template_kwargs(llm):
            updated["chat_template_kwargs"] = dict(_NO_THINK_TEMPLATE_KWARGS)
        else:
            updated["messages"] = _messages_with_no_think(updated.get("messages"))
        return updated

    def create_chat_completion(
        self,
        *,
        messages,
        max_tokens,
        temperature,
        top_k=None,
        top_p=None,
        repeat_penalty=None,
        stream=False,
        role: str = "main",
        response_format=None,
        timeout: float | None = None,
    ):
        del role, timeout
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stream": stream,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if not stream:
            with self._inference_lock:
                llm = self.llm
                return llm.create_chat_completion(**self._completion_kwargs(llm, kwargs))

        def wrapped():
            with self._inference_lock:
                llm = self.llm
                yield from llm.create_chat_completion(**self._completion_kwargs(llm, kwargs))

        return wrapped()
