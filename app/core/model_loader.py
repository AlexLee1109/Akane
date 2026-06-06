"""Singleton llama.cpp model loader."""

from __future__ import annotations

import os
import threading
import time
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

USE_STATIC_PREFIX_CACHE = True
_PREFIX_MARKER = "\n__AKANE_STATIC_PREFIX_BOUNDARY__\n"
_PREFIX_TRIM_TOKENS = 8
_PREFIX_CACHE = {
    "revision": None,
    "include_memory": None,
    "state": None,
    "tokens": 0,
    "enabled": USE_STATIC_PREFIX_CACHE,
    "disabled_reason": "",
}


def _timing_enabled() -> bool:
    return str(os.environ.get("AKANE_TIMING", "")).strip().lower() in {"1", "true", "yes", "on"}


def _prefix_log(text: str) -> None:
    if _timing_enabled():
        print(f"[Akane:prefix_cache] {text}", flush=True)


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

    def prefix_cache_status(self) -> dict[str, object]:
        return {
            "enabled": bool(_PREFIX_CACHE["enabled"]),
            "tokens": int(_PREFIX_CACHE["tokens"] or 0),
            "include_memory": bool(_PREFIX_CACHE["include_memory"]),
            "disabled_reason": str(_PREFIX_CACHE["disabled_reason"] or ""),
        }

    def _clear_prefix_cache(self) -> None:
        _PREFIX_CACHE.update({
            "revision": None,
            "include_memory": None,
            "state": None,
            "tokens": 0,
            "enabled": USE_STATIC_PREFIX_CACHE,
            "disabled_reason": "",
        })

    def _disable_prefix_cache(self, reason: str) -> None:
        self._clear_prefix_cache()
        _PREFIX_CACHE["enabled"] = False
        _PREFIX_CACHE["disabled_reason"] = reason
        _prefix_log(f"disabled reason={reason}")

    def _chat_formatter(self, llm):
        try:
            from llama_cpp import llama_chat_format

            handler = (
                getattr(llm, "chat_handler", None)
                or getattr(llm, "_chat_handlers", {}).get(getattr(llm, "chat_format", ""))
                or llama_chat_format.get_chat_completion_handler(getattr(llm, "chat_format", ""))
            )
        except Exception:
            return None
        for cell in getattr(handler, "__closure__", None) or ():
            formatter = cell.cell_contents
            if not callable(formatter):
                continue
            try:
                if hasattr(formatter(messages=[{"role": "system", "content": "Akane"}]), "prompt"):
                    return formatter
            except Exception:
                continue
        return None

    def _prefix_tokens(self, llm, prompt: str) -> list[int] | None:
        formatter = self._chat_formatter(llm)
        if formatter is None:
            return None
        result = formatter(messages=[{"role": "system", "content": prompt + _PREFIX_MARKER}])
        formatted = str(result.prompt)
        marker_at = formatted.find(_PREFIX_MARKER)
        if marker_at < 0:
            return None
        tokens = list(llm.tokenize(
            formatted[:marker_at].encode("utf-8"),
            add_bos=not bool(getattr(result, "added_special", False)),
            special=True,
        ))
        if len(tokens) > _PREFIX_TRIM_TOKENS + 1:
            tokens = tokens[:-_PREFIX_TRIM_TOKENS]
        return tokens

    def _build_prefix_cache(self, llm, include_memory: bool) -> bool:
        if not USE_STATIC_PREFIX_CACHE:
            return False
        missing = [
            name
            for name in ("save_state", "load_state", "eval", "tokenize", "reset")
            if not hasattr(llm, name)
        ]
        if missing:
            self._disable_prefix_cache(f"{missing[0]}_unavailable")
            return False
        try:
            from app.core.character import get_static_system_prompt, prompt_revision

            prompt = get_static_system_prompt(include_memory=include_memory)
            tokens = self._prefix_tokens(llm, prompt)
            if not tokens:
                self._disable_prefix_cache("chat_formatter_unavailable")
                return False
            started = time.perf_counter()
            llm.reset()
            llm.eval(tokens)
            _PREFIX_CACHE.update({
                "revision": prompt_revision(),
                "include_memory": include_memory,
                "state": llm.save_state(),
                "tokens": len(tokens),
                "enabled": True,
                "disabled_reason": "",
            })
            _prefix_log(f"built tokens={len(tokens)} time={time.perf_counter() - started:.2f}s")
            return True
        except Exception as exc:
            self._disable_prefix_cache(type(exc).__name__)
            return False

    def _restore_prefix_cache(self, llm, messages) -> None:
        if not USE_STATIC_PREFIX_CACHE or not messages:
            return
        system = messages[0] if isinstance(messages[0], dict) else {}
        system_text = str(system.get("content") or "")
        if system.get("role") != "system" or not system_text:
            return
        include_memory = "[AKANE MEMORY RULES]" in system_text
        try:
            from app.core.character import get_static_system_prompt, prompt_revision

            static_prompt = get_static_system_prompt(include_memory=include_memory)
            if not system_text.startswith(static_prompt):
                return
            if (
                not _PREFIX_CACHE["enabled"]
                or _PREFIX_CACHE["revision"] != prompt_revision()
                or _PREFIX_CACHE["include_memory"] is not include_memory
                or _PREFIX_CACHE["state"] is None
            ):
                if not self._build_prefix_cache(llm, include_memory):
                    return
            llm.load_state(_PREFIX_CACHE["state"])
            _prefix_log(
                f"restored=true dynamic_chars={len(system_text) - len(static_prompt)} "
                f"tokens={_PREFIX_CACHE['tokens']}"
            )
        except Exception as exc:
            try:
                llm.reset()
            except Exception:
                pass
            _prefix_log(f"restored=false reason={type(exc).__name__}")

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
                self._build_prefix_cache(self._llm, include_memory=True)
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
            self._clear_prefix_cache()

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
                    self._clear_prefix_cache()
        return self.status()

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
                self._restore_prefix_cache(llm, messages)
                return llm.create_chat_completion(**kwargs)

        def wrapped():
            with self._inference_lock:
                llm = self.llm
                self._restore_prefix_cache(llm, messages)
                yield from llm.create_chat_completion(**kwargs)

        return wrapped()
