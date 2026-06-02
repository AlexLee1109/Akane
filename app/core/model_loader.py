"""Minimal llama.cpp model manager."""

from __future__ import annotations

import threading
from pathlib import Path


def content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    out.append(value)
        return "".join(out)
    return str(content)


class ModelManager:
    _instance: "ModelManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
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

        self.LLAMA_CONTEXT_WINDOW = LLAMA_CONTEXT_WINDOW
        self.LLAMA_BATCH_SIZE = LLAMA_BATCH_SIZE
        self.LLAMA_UBATCH_SIZE = LLAMA_UBATCH_SIZE
        self.LLAMA_THREADS = LLAMA_THREADS
        self.LLAMA_THREADS_BATCH = LLAMA_THREADS_BATCH
        self.LLAMA_FLASH_ATTN = LLAMA_FLASH_ATTN
        self.LLAMA_GPU_LAYERS = LLAMA_GPU_LAYERS
        self.LLAMA_OFFLOAD_KQV = LLAMA_OFFLOAD_KQV
        self.LLAMA_OP_OFFLOAD = LLAMA_OP_OFFLOAD
        self.LLAMA_USE_MMAP = LLAMA_USE_MMAP
        self.LLAMA_USE_MLOCK = LLAMA_USE_MLOCK
        self.LLAMA_LAST_N_TOKENS_SIZE = max(0, int(LLAMA_LAST_N_TOKENS_SIZE))
        self._local_model_path = Path(MODEL_PATH)
        self._llm = None
        self._loading = False
        self._load_error: Exception | None = None
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "loading": self._loading,
                "loaded": self._llm is not None,
                "error": str(self._load_error) if self._load_error else None,
                "backend": "llama_cpp",
                "local_model_path": str(self._local_model_path),
            }

    def unload_local_model(self) -> None:
        with self._lock:
            self._llm = None

    def _batch_settings(self) -> tuple[int, int]:
        n_batch = max(1, min(int(self.LLAMA_BATCH_SIZE), int(self.LLAMA_CONTEXT_WINDOW)))
        n_ubatch = max(1, min(int(self.LLAMA_UBATCH_SIZE), n_batch))
        return n_batch, n_ubatch

    def switch_backend(
        self,
        backend: str = "llama_cpp",
        *,
        local_model_path: str | None = None,
    ) -> dict[str, object]:
        backend = str(backend or "llama_cpp").strip().lower() or "llama_cpp"
        if backend != "llama_cpp":
            raise ValueError("Only the 'llama_cpp' backend is available.")
        with self._lock:
            if local_model_path is not None:
                value = str(local_model_path).strip()
                if not value:
                    raise ValueError("Local model path cannot be empty.")
                path = Path(value)
                if path != self._local_model_path:
                    self._local_model_path = path
                    self._llm = None
                    self._load_error = None
        return self.status()

    def ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        with self._lock:
            if self._llm is not None:
                return
            self._loading = True
            self._load_error = None
            try:
                from llama_cpp import Llama

                n_batch, n_ubatch = self._batch_settings()
                kwargs = {
                    "model_path": str(self._local_model_path),
                    "n_ctx": self.LLAMA_CONTEXT_WINDOW,
                    "n_batch": n_batch,
                    "n_ubatch": n_ubatch,
                    "n_threads": self.LLAMA_THREADS or None,
                    "n_threads_batch": self.LLAMA_THREADS_BATCH or None,
                    "flash_attn": self.LLAMA_FLASH_ATTN,
                    "offload_kqv": self.LLAMA_OFFLOAD_KQV,
                    "op_offload": self.LLAMA_OP_OFFLOAD,
                    "use_mmap": self.LLAMA_USE_MMAP,
                    "use_mlock": self.LLAMA_USE_MLOCK,
                    "last_n_tokens_size": self.LLAMA_LAST_N_TOKENS_SIZE,
                    "logits_all": False,
                    "embedding": False,
                    "no_perf": True,
                    "verbose": False,
                }
                if self.LLAMA_GPU_LAYERS:
                    kwargs["n_gpu_layers"] = self.LLAMA_GPU_LAYERS
                print(
                    f"Loading model {self._local_model_path} "
                    f"n_ctx={self.LLAMA_CONTEXT_WINDOW} n_batch={n_batch} n_ubatch={n_ubatch}",
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
                return self.llm.create_chat_completion(**kwargs)

        def wrapped():
            with self._inference_lock:
                yield from self.llm.create_chat_completion(**kwargs)

        return wrapped()
