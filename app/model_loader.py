"""Lazy-loading model manager for Akane.

Provides a singleton ModelManager that loads the LLM on first use.
"""

import json
import threading
import gc
import ssl
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest
from typing import Optional

try:
    import certifi
except ImportError:  # pragma: no cover - optional
    certifi = None


class OpenRouterBackend:
    """Minimal OpenAI-compatible chat backend for OpenRouter."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        site_url: str = "",
        app_name: str = "Akane",
        ca_bundle: str = "",
        skip_ssl_verify: bool = False,
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required when using the OpenRouter coder model")
        if not model:
            raise RuntimeError("OPENROUTER_CODER_MODEL is required when using the OpenRouter coder model")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.site_url = site_url
        self.app_name = app_name or "Akane"
        self.ca_bundle = ca_bundle.strip()
        self.skip_ssl_verify = skip_ssl_verify

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        return headers

    def _ssl_context(self):
        if self.skip_ssl_verify:
            return ssl._create_unverified_context()
        if self.ca_bundle:
            return ssl.create_default_context(cafile=self.ca_bundle)
        if certifi is not None:
            return ssl.create_default_context(cafile=certifi.where())
        return ssl.create_default_context()

    def _request(self, payload: dict):
        req = urlrequest.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            return urlrequest.urlopen(req, timeout=120, context=self._ssl_context())
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed: {exc.code} {body}") from exc
        except urlerror.URLError as exc:
            reason = exc.reason
            if isinstance(reason, ssl.SSLCertVerificationError):
                hint = " Try OPENROUTER_CA_BUNDLE first, or as a last resort OPENROUTER_SKIP_SSL_VERIFY=1."
                raise RuntimeError(f"OpenRouter SSL verification failed: {reason}.{hint}") from exc
            raise RuntimeError(f"OpenRouter request failed: {reason}") from exc

    def create_chat_completion(
        self,
        *,
        messages,
        max_tokens,
        temperature,
        model=None,
        top_k=None,
        top_p=None,
        repeat_penalty=None,
        stream=False,
    ):
        payload = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if top_p is not None:
            payload["top_p"] = top_p

        response = self._request(payload)
        if not stream:
            with response as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(data)[:500]}")
            choice = choices[0]
            message = choice.get("message", {}) or {}
            content = message.get("content")
            if content in (None, ""):
                content = choice.get("text", "")
            return {
                "choices": [
                    {
                        "message": {
                            "content": content,
                        },
                        "finish_reason": choice.get("finish_reason", ""),
                    }
                ]
            }

        def event_stream():
            with response as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_line = line[5:].strip()
                    if data_line == "[DONE]":
                        break
                    event = json.loads(data_line)
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    yield {"choices": [{"delta": {"content": delta.get("content", "")}}]}

        return event_stream()


class ModelManager:
    """Singleton manager for the LLM model with lazy loading."""

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        from app.config import (
            DEVICE,
            LLAMA_BATCH_SIZE,
            LLAMA_CONTEXT_WINDOW,
            LLAMA_FLASH_ATTN,
            LLAMA_GPU_LAYERS,
            LLAMA_IDLE_UNLOAD_SECONDS,
            LLAMA_THREADS,
            LLAMA_UBATCH_SIZE,
            MODEL_PATH,
            OPENROUTER_API_KEY,
            OPENROUTER_APP_NAME,
            OPENROUTER_BASE_URL,
            OPENROUTER_CA_BUNDLE,
            OPENROUTER_CODER_MODEL,
            OPENROUTER_SKIP_SSL_VERIFY,
            OPENROUTER_SITE_URL,
        )

        self.DEVICE = DEVICE
        self.LLAMA_CONTEXT_WINDOW = LLAMA_CONTEXT_WINDOW
        self.LLAMA_BATCH_SIZE = LLAMA_BATCH_SIZE
        self.LLAMA_UBATCH_SIZE = LLAMA_UBATCH_SIZE
        self.LLAMA_THREADS = LLAMA_THREADS
        self.LLAMA_FLASH_ATTN = LLAMA_FLASH_ATTN
        self.LLAMA_GPU_LAYERS = LLAMA_GPU_LAYERS
        self.LLAMA_IDLE_UNLOAD_SECONDS = max(0.0, float(LLAMA_IDLE_UNLOAD_SECONDS))
        self.MODEL_PATH = Path(MODEL_PATH)
        self.OPENROUTER_API_KEY = OPENROUTER_API_KEY
        self.OPENROUTER_CODER_MODEL = OPENROUTER_CODER_MODEL
        self.OPENROUTER_BASE_URL = OPENROUTER_BASE_URL
        self.OPENROUTER_CA_BUNDLE = OPENROUTER_CA_BUNDLE
        self.OPENROUTER_SITE_URL = OPENROUTER_SITE_URL
        self.OPENROUTER_APP_NAME = OPENROUTER_APP_NAME
        self.OPENROUTER_SKIP_SSL_VERIFY = OPENROUTER_SKIP_SSL_VERIFY
        self._local_model_path = Path(MODEL_PATH)
        self._llm = None
        self._coder_llm = None
        self._loading = False
        self._coder_loading = False
        self._load_lock = threading.Lock()
        self._coder_load_lock = threading.Lock()
        self._load_error = None
        self._coder_load_error = None
        self._llama_idle_timer = None
        self._last_llama_use_at = 0.0

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _active_model_name_unlocked(self) -> str:
        return Path(self._local_model_path).name

    def model_name_for_role(self, role: str = "main") -> str:
        role = str(role or "main").strip().lower()
        if role == "coder":
            return self.OPENROUTER_CODER_MODEL or Path(self._local_model_path).name
        return Path(self._local_model_path).name

    def status(self) -> dict[str, object]:
        with self._load_lock:
            return {
                "loading": self._loading,
                "loaded": self._llm is not None,
                "error": str(self._load_error) if self._load_error else None,
                "backend": "llama_cpp",
                "coder_backend": "openrouter" if self.OPENROUTER_CODER_MODEL else None,
                "model_name": self._active_model_name_unlocked(),
                "local_model_path": str(self._local_model_path),
                "llama_idle_unload_seconds": self.LLAMA_IDLE_UNLOAD_SECONDS,
                "openrouter_coder_model": self.OPENROUTER_CODER_MODEL,
                "coder_loaded": self._coder_llm is not None,
                "coder_error": str(self._coder_load_error) if self._coder_load_error else None,
            }

    def _cancel_idle_timer_unlocked(self) -> None:
        if self._llama_idle_timer is not None:
            self._llama_idle_timer.cancel()
            self._llama_idle_timer = None

    def _finish_llama_request(self) -> None:
        if self.LLAMA_IDLE_UNLOAD_SECONDS <= 0:
            return
        with self._load_lock:
            self._last_llama_use_at = time.monotonic()
            self._cancel_idle_timer_unlocked()
            timer = threading.Timer(self.LLAMA_IDLE_UNLOAD_SECONDS, self._idle_unload_local_model)
            timer.daemon = True
            self._llama_idle_timer = timer
            timer.start()

    def _idle_unload_local_model(self) -> None:
        with self._load_lock:
            self._llama_idle_timer = None
            if self._llm is None or self._loading:
                return
            idle_for = time.monotonic() - self._last_llama_use_at
            if idle_for < self.LLAMA_IDLE_UNLOAD_SECONDS:
                remaining = self.LLAMA_IDLE_UNLOAD_SECONDS - idle_for
                timer = threading.Timer(remaining, self._idle_unload_local_model)
                timer.daemon = True
                self._llama_idle_timer = timer
                timer.start()
                return
            self._llm = None
        gc.collect()

    def switch_backend(
        self,
        backend: str,
        *,
        local_model_path: str | None = None,
        openrouter_model: str | None = None,
    ) -> dict[str, object]:
        backend = backend.strip().lower()
        if backend not in {"llama_cpp", "openrouter"}:
            raise ValueError("Backend must be 'llama_cpp' or 'openrouter'.")

        with self._load_lock:
            if local_model_path is not None:
                local_model_path = local_model_path.strip()
                if not local_model_path:
                    raise ValueError("Local model path cannot be empty.")
                self._local_model_path = Path(local_model_path)

            if openrouter_model is not None:
                openrouter_model = openrouter_model.strip()
                if not openrouter_model:
                    raise ValueError("OpenRouter model cannot be empty.")
                self.OPENROUTER_CODER_MODEL = openrouter_model

            if backend == "llama_cpp":
                self.MODEL_PATH = self._local_model_path
                self._cancel_idle_timer_unlocked()
                self._llm = None
                self._loading = False
                self._load_error = None
            else:
                self._coder_llm = None
                self._coder_loading = False
                self._coder_load_error = None

        gc.collect()
        return self.status()

    def ensure_loaded(self):
        """Ensure the model is loaded. Blocks until loaded or raises error."""
        if self._llm is not None:
            return

        with self._load_lock:
            if self._llm is not None:
                return

            self._loading = True
            try:
                from llama_cpp import Llama

                print(
                    f"Loading model from {self._local_model_path} with n_ctx={self.LLAMA_CONTEXT_WINDOW}...",
                    flush=True,
                )
                llm_kwargs = {
                    "model_path": str(self._local_model_path),
                    "n_ctx": self.LLAMA_CONTEXT_WINDOW,
                    "n_batch": self.LLAMA_BATCH_SIZE,
                    "n_ubatch": min(self.LLAMA_UBATCH_SIZE, self.LLAMA_BATCH_SIZE),
                    "flash_attn": self.LLAMA_FLASH_ATTN,
                    "verbose": False,
                }

                if self.DEVICE == "mps":
                    llm_kwargs["n_gpu_layers"] = self.LLAMA_GPU_LAYERS
                elif self.LLAMA_THREADS > 0:
                    llm_kwargs["n_threads"] = self.LLAMA_THREADS

                try:
                    self._llm = Llama(**llm_kwargs)
                except TypeError:
                    fallback_kwargs = dict(llm_kwargs)
                    fallback_kwargs.pop("flash_attn", None)
                    fallback_kwargs.pop("n_ubatch", None)
                    self._llm = Llama(**fallback_kwargs)
                self._last_llama_use_at = time.monotonic()
                print(f"Model loaded on {self.DEVICE}.", flush=True)
                self._loading = False
                self._load_error = None
            except Exception as e:
                self._loading = False
                self._load_error = e
                raise

    def ensure_coder_loaded(self):
        if self._coder_llm is not None:
            return
        with self._coder_load_lock:
            if self._coder_llm is not None:
                return
            self._coder_loading = True
            try:
                self._coder_llm = OpenRouterBackend(
                    api_key=self.OPENROUTER_API_KEY,
                    model=self.OPENROUTER_CODER_MODEL,
                    base_url=self.OPENROUTER_BASE_URL,
                    site_url=self.OPENROUTER_SITE_URL,
                    app_name=self.OPENROUTER_APP_NAME,
                    ca_bundle=self.OPENROUTER_CA_BUNDLE,
                    skip_ssl_verify=self.OPENROUTER_SKIP_SSL_VERIFY,
                )
                self._coder_loading = False
                self._coder_load_error = None
            except Exception as e:
                self._coder_loading = False
                self._coder_load_error = e
                raise

    @property
    def llm(self):
        """Get the LLM instance, loading if necessary."""
        if self._load_error:
            raise RuntimeError(f"Model failed to load: {self._load_error}") from self._load_error
        if self._llm is None:
            self.ensure_loaded()
        return self._llm

    @property
    def coder_llm(self):
        if self._coder_load_error:
            raise RuntimeError(f"Coder model failed to load: {self._coder_load_error}") from self._coder_load_error
        if self._coder_llm is None:
            self.ensure_coder_loaded()
        return self._coder_llm

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
    ):
        model_name = self.model_name_for_role(role)
        if role == "coder" and self.OPENROUTER_CODER_MODEL:
            llm = self.coder_llm
            return llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model_name,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=stream,
            )
        if self._load_error:
            raise RuntimeError(f"Model failed to load: {self._load_error}") from self._load_error
        with self._load_lock:
            self._cancel_idle_timer_unlocked()
        llm = self.llm
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stream=stream,
        )
        if not stream:
            self._finish_llama_request()
            return result

        manager = self

        def wrapped_stream():
            try:
                for chunk in result:
                    yield chunk
            finally:
                manager._finish_llama_request()

        return wrapped_stream()

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._llm is not None


# Convenience function for backward compatibility
def get_llm():
    """Get the LLM instance (lazy loads if needed)."""
    return ModelManager.get_instance().llm


class _LazyLLM:
    """Proxy that forwards all attribute access to the lazily-loaded LLM."""
    def __getattr__(self, name):
        return getattr(get_llm(), name)

    def create_chat_completion(self, *args, **kwargs):
        return ModelManager.get_instance().create_chat_completion(*args, **kwargs)

# Global LLM proxy for backward compatibility with code that does `from app.model_loader import LLM`
LLM = _LazyLLM()
