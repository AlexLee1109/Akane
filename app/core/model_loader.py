"""Minimal llama.cpp model manager."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return int(default)


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
            DEVICE,
            LLAMA_BATCH_SIZE,
            LLAMA_CONTEXT_WINDOW,
            LLAMA_FLASH_ATTN,
            LLAMA_GPU_LAYERS,
            LLAMA_IDLE_UNLOAD_SECONDS,
            LLAMA_OFFLOAD_KQV,
            LLAMA_OP_OFFLOAD,
            LLAMA_THREADS,
            LLAMA_THREADS_BATCH,
            LLAMA_UBATCH_SIZE,
            MODEL_PATH,
        )

        self.DEVICE = DEVICE
        self.LLAMA_CONTEXT_WINDOW = LLAMA_CONTEXT_WINDOW
        self.LLAMA_BATCH_SIZE = LLAMA_BATCH_SIZE
        self.LLAMA_UBATCH_SIZE = LLAMA_UBATCH_SIZE
        self.LLAMA_THREADS = LLAMA_THREADS
        self.LLAMA_THREADS_BATCH = LLAMA_THREADS_BATCH
        self.LLAMA_FLASH_ATTN = LLAMA_FLASH_ATTN
        self.LLAMA_GPU_LAYERS = LLAMA_GPU_LAYERS
        self.LLAMA_OFFLOAD_KQV = LLAMA_OFFLOAD_KQV
        self.LLAMA_OP_OFFLOAD = LLAMA_OP_OFFLOAD
        self.LLAMA_IDLE_UNLOAD_SECONDS = max(0.0, float(LLAMA_IDLE_UNLOAD_SECONDS))
        self.LLAMA_CACHE_MB = max(0, _env_int("AKANE_LLAMA_CACHE_MB", 256))
        self._local_model_path = Path(MODEL_PATH)
        self._llm = None
        self._loading = False
        self._load_error: Exception | None = None
        self._lock = threading.RLock()
        self._idle_timer: threading.Timer | None = None

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

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle_unload(self) -> None:
        if self.LLAMA_IDLE_UNLOAD_SECONDS <= 0:
            return
        with self._lock:
            self._cancel_idle_timer()
            self._idle_timer = threading.Timer(self.LLAMA_IDLE_UNLOAD_SECONDS, self.unload_local_model)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def unload_local_model(self) -> None:
        with self._lock:
            self._cancel_idle_timer()
            self._llm = None

    def _install_prompt_cache(self) -> None:
        if self.LLAMA_CACHE_MB <= 0 or self._llm is None or not hasattr(self._llm, "set_cache"):
            return
        try:
            from llama_cpp import LlamaRAMCache

            self._llm.set_cache(LlamaRAMCache(capacity_bytes=self.LLAMA_CACHE_MB * 1024 * 1024))
            print(f"Prompt cache enabled: {self.LLAMA_CACHE_MB} MB", flush=True)
        except Exception as exc:
            print(f"Prompt cache disabled: {exc}", flush=True)

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
                self._local_model_path = Path(value)
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

                kwargs = {
                    "model_path": str(self._local_model_path),
                    "n_ctx": self.LLAMA_CONTEXT_WINDOW,
                    "n_batch": self.LLAMA_BATCH_SIZE,
                    "n_ubatch": min(self.LLAMA_UBATCH_SIZE, self.LLAMA_BATCH_SIZE),
                    "n_threads": self.LLAMA_THREADS or None,
                    "n_threads_batch": self.LLAMA_THREADS_BATCH or None,
                    "flash_attn": self.LLAMA_FLASH_ATTN,
                    "offload_kqv": self.LLAMA_OFFLOAD_KQV,
                    "op_offload": self.LLAMA_OP_OFFLOAD,
                    "use_mmap": True,
                    "use_mlock": False,
                    "last_n_tokens_size": 64,
                    "logits_all": False,
                    "embedding": False,
                    "verbose": False,
                }
                if self.LLAMA_GPU_LAYERS:
                    kwargs["n_gpu_layers"] = self.LLAMA_GPU_LAYERS
                print(f"Loading model {self._local_model_path} n_ctx={self.LLAMA_CONTEXT_WINDOW}", flush=True)
                self._llm = Llama(**kwargs)
                self._install_prompt_cache()
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
        with self._lock:
            self._cancel_idle_timer()
        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stream=stream,
        )
        if not stream:
            self._schedule_idle_unload()
            return result

        def wrapped():
            try:
                yield from result
            finally:
                self._schedule_idle_unload()

        return wrapped()
