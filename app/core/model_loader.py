"""Minimal llama.cpp model manager."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

USE_STATIC_PREFIX_CACHE = True
_PREFIX_BOUNDARY_MARKER = "\n__AKANE_STATIC_PREFIX_BOUNDARY__\n"
_PREFIX_BOUNDARY_TRIM_TOKENS = 8
_STATIC_PREFIX_CACHE = {
    "revision": None,
    "include_memory": None,
    "state": None,
    "tokens": 0,
    "enabled": USE_STATIC_PREFIX_CACHE,
    "disabled_reason": "",
}


def _timing_enabled() -> bool:
    return str(os.environ.get("AKANE_TIMING", "")).strip().lower() in {"1", "true", "yes", "on"}


def _timing_log(text: str) -> None:
    print(f"[Akane:timing] {text}", flush=True)


def _prefix_log(text: str, *, timing_only: bool = False) -> None:
    if timing_only and not _timing_enabled():
        return
    print(f"[Akane:prefix_cache] {text}", flush=True)


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

    def _clear_prefix_cache(self) -> None:
        _STATIC_PREFIX_CACHE.update({
            "revision": None,
            "include_memory": None,
            "state": None,
            "tokens": 0,
            "enabled": USE_STATIC_PREFIX_CACHE,
            "disabled_reason": "",
        })

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

    def prefix_cache_status(self) -> dict[str, object]:
        return {
            "enabled": bool(_STATIC_PREFIX_CACHE.get("enabled")),
            "tokens": int(_STATIC_PREFIX_CACHE.get("tokens") or 0),
            "include_memory": bool(_STATIC_PREFIX_CACHE.get("include_memory")),
            "disabled_reason": str(_STATIC_PREFIX_CACHE.get("disabled_reason") or ""),
        }

    def load_model(self):
        self.ensure_loaded()
        return self._llm

    def get_model(self):
        return self.llm

    def unload_local_model(self) -> None:
        with self._lock:
            self._llm = None
            self._clear_prefix_cache()

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
                    self._clear_prefix_cache()
        return self.status()

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
            if callable(formatter):
                try:
                    result = formatter(messages=[{"role": "system", "content": "Akane"}])
                except Exception:
                    continue
                if hasattr(result, "prompt"):
                    return formatter
        return None

    def _static_prefix_tokens(self, llm, static_prompt: str) -> list[int] | None:
        formatter = self._chat_formatter(llm)
        if formatter is None:
            return None
        result = formatter(messages=[{"role": "system", "content": static_prompt + _PREFIX_BOUNDARY_MARKER}])
        prompt = str(result.prompt)
        marker_at = prompt.find(_PREFIX_BOUNDARY_MARKER)
        if marker_at < 0:
            return None
        tokens = llm.tokenize(
            prompt[:marker_at].encode("utf-8"),
            add_bos=not bool(getattr(result, "added_special", False)),
            special=True,
        )
        if len(tokens) > _PREFIX_BOUNDARY_TRIM_TOKENS + 1:
            tokens = tokens[:-_PREFIX_BOUNDARY_TRIM_TOKENS]
        return list(tokens)

    def _disable_prefix_cache(self, reason: str) -> None:
        _STATIC_PREFIX_CACHE.update({
            "revision": None,
            "include_memory": None,
            "state": None,
            "tokens": 0,
            "enabled": False,
            "disabled_reason": reason,
        })
        _prefix_log(f"disabled reason={reason}")

    def _build_static_prefix_cache(self, llm, *, include_memory: bool, reason: str = "startup") -> bool:
        if not USE_STATIC_PREFIX_CACHE:
            self._disable_prefix_cache("constant_disabled")
            return False
        required = ("save_state", "load_state", "eval", "tokenize", "reset")
        missing = [name for name in required if not hasattr(llm, name)]
        if missing:
            self._disable_prefix_cache(f"{missing[0]}_unavailable")
            return False
        try:
            from app.core.character import get_static_system_prompt, prompt_revision

            revision = prompt_revision()
            static_prompt = get_static_system_prompt(include_memory=include_memory)
            tokens = self._static_prefix_tokens(llm, static_prompt)
            if not tokens:
                self._disable_prefix_cache("chat_formatter_unavailable")
                return False
            _prefix_log(f"building static prefix reason={reason} include_memory={int(include_memory)}")
            start = time.perf_counter()
            llm.reset()
            llm.eval(tokens)
            state = llm.save_state()
            _STATIC_PREFIX_CACHE.update({
                "revision": revision,
                "include_memory": include_memory,
                "state": state,
                "tokens": len(tokens),
                "enabled": True,
                "disabled_reason": "",
            })
            _prefix_log(f"built tokens={len(tokens)} time={time.perf_counter() - start:.2f}s")
            return True
        except Exception as exc:
            self._disable_prefix_cache(type(exc).__name__)
            return False

    def _include_memory_from_messages(self, messages) -> bool | None:
        if not messages:
            return None
        first = messages[0] if isinstance(messages[0], dict) else {}
        if first.get("role") != "system":
            return None
        system_text = str(first.get("content") or "")
        if not system_text:
            return None
        return "[AKANE MEMORY RULES]" in system_text

    def _restore_static_prefix_cache(self, llm, messages) -> None:
        if not USE_STATIC_PREFIX_CACHE or not _STATIC_PREFIX_CACHE.get("enabled"):
            return
        include_memory = self._include_memory_from_messages(messages)
        if include_memory is None:
            return
        try:
            from app.core.character import get_static_system_prompt, prompt_revision

            revision = prompt_revision()
            static_prompt = get_static_system_prompt(include_memory=include_memory)
            system_text = str(messages[0].get("content") or "")
            if not system_text.startswith(static_prompt):
                _prefix_log("restored=false reason=system_prefix_mismatch", timing_only=True)
                return
            if (
                _STATIC_PREFIX_CACHE.get("revision") != revision
                or _STATIC_PREFIX_CACHE.get("include_memory") is not include_memory
                or _STATIC_PREFIX_CACHE.get("state") is None
            ):
                _prefix_log("rebuild reason=prompt_revision_changed")
                if not self._build_static_prefix_cache(llm, include_memory=include_memory, reason="prompt_revision_changed"):
                    return
            state = _STATIC_PREFIX_CACHE.get("state")
            if state is None:
                return
            llm.load_state(state)
            dynamic_chars = max(0, len(system_text) - len(static_prompt))
            _prefix_log(
                f"restored=true dynamic_chars={dynamic_chars} tokens={_STATIC_PREFIX_CACHE.get('tokens')}",
                timing_only=True,
            )
        except Exception as exc:
            try:
                llm.reset()
            except Exception:
                pass
            _prefix_log(f"restored=false reason={type(exc).__name__}")

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
                self._build_static_prefix_cache(self._llm, include_memory=True)
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
                if not _timing_enabled():
                    self._restore_static_prefix_cache(self.llm, messages)
                    return self.llm.create_chat_completion(**kwargs)
                start = time.perf_counter()
                self._restore_static_prefix_cache(self.llm, messages)
                result = self.llm.create_chat_completion(**kwargs)
                total = time.perf_counter() - start
                _timing_log(f"gen={total:.2f}s total={total:.2f}s stream=0")
                return result

        def wrapped():
            timing = _timing_enabled()
            start = time.perf_counter() if timing else 0.0
            first = 0.0
            chunks = 0
            with self._inference_lock:
                self._restore_static_prefix_cache(self.llm, messages)
                for chunk in self.llm.create_chat_completion(**kwargs):
                    if timing:
                        chunks += 1
                        if not first:
                            first = time.perf_counter()
                    yield chunk
            if timing:
                end = time.perf_counter()
                ttft = (first or end) - start
                gen = end - (first or end)
                tok_s = chunks / gen if gen > 0 else 0.0
                _timing_log(f"ttft={ttft:.2f}s gen={gen:.2f}s total={end - start:.2f}s tok_s={tok_s:.1f}")

        return wrapped()
