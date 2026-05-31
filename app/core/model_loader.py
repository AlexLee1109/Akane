"""Minimal llama.cpp model manager."""

from __future__ import annotations

import json
import ssl
import threading
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None


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


class OpenRouterBackend:
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
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        if not model:
            raise RuntimeError("OPENROUTER_CODER_MODEL is required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.site_url = site_url
        self.app_name = app_name or "Akane"
        self.ca_bundle = ca_bundle
        self.skip_ssl_verify = skip_ssl_verify
        self._ssl_context = None

    def _context(self):
        if self._ssl_context is not None:
            return self._ssl_context
        if self.skip_ssl_verify:
            self._ssl_context = ssl._create_unverified_context()
        elif self.ca_bundle:
            self._ssl_context = ssl.create_default_context(cafile=self.ca_bundle)
        elif certifi is not None:
            self._ssl_context = ssl.create_default_context(cafile=certifi.where())
        else:
            self._ssl_context = ssl.create_default_context()
        return self._ssl_context

    def _request(self, payload: dict, timeout: float | None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        req = urlrequest.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            return urlrequest.urlopen(req, timeout=timeout or 120, context=self._context())
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed: {exc.code} {body}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

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
        response_format=None,
        timeout: float | None = None,
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
        if response_format is not None:
            payload["response_format"] = response_format

        response = self._request(payload, timeout)
        if not stream:
            with response as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            return {"choices": [{"message": {"content": content_to_text(message.get("content"))}, "finish_reason": choice.get("finish_reason", "")}]}

        def events():
            with response as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choice = (event.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    yield {"choices": [{"delta": {"content": content_to_text(delta.get("content"))}, "finish_reason": choice.get("finish_reason", "")}]}

        return events()


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
        self.LLAMA_THREADS_BATCH = LLAMA_THREADS_BATCH
        self.LLAMA_FLASH_ATTN = LLAMA_FLASH_ATTN
        self.LLAMA_GPU_LAYERS = LLAMA_GPU_LAYERS
        self.LLAMA_OFFLOAD_KQV = LLAMA_OFFLOAD_KQV
        self.LLAMA_OP_OFFLOAD = LLAMA_OP_OFFLOAD
        self.LLAMA_IDLE_UNLOAD_SECONDS = max(0.0, float(LLAMA_IDLE_UNLOAD_SECONDS))
        self._local_model_path = Path(MODEL_PATH)
        self.OPENROUTER_API_KEY = OPENROUTER_API_KEY
        self.OPENROUTER_CODER_MODEL = OPENROUTER_CODER_MODEL
        self.OPENROUTER_BASE_URL = OPENROUTER_BASE_URL
        self.OPENROUTER_CA_BUNDLE = OPENROUTER_CA_BUNDLE
        self.OPENROUTER_SITE_URL = OPENROUTER_SITE_URL
        self.OPENROUTER_APP_NAME = OPENROUTER_APP_NAME
        self.OPENROUTER_SKIP_SSL_VERIFY = OPENROUTER_SKIP_SSL_VERIFY
        self._llm = None
        self._coder = None
        self._loading = False
        self._load_error: Exception | None = None
        self._coder_error: Exception | None = None
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
                "openrouter_model": self.OPENROUTER_CODER_MODEL,
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

    def switch_backend(
        self,
        backend: str,
        *,
        local_model_path: str | None = None,
        openrouter_model: str | None = None,
    ) -> dict[str, object]:
        backend = str(backend or "").strip().lower()
        if backend not in {"llama_cpp", "openrouter"}:
            raise ValueError("Backend must be 'llama_cpp' or 'openrouter'.")
        with self._lock:
            if local_model_path is not None:
                value = str(local_model_path).strip()
                if not value:
                    raise ValueError("Local model path cannot be empty.")
                self._local_model_path = Path(value)
                self._llm = None
                self._load_error = None
            if openrouter_model is not None:
                value = str(openrouter_model).strip()
                if not value:
                    raise ValueError("OpenRouter model cannot be empty.")
                self.OPENROUTER_CODER_MODEL = value
                self._coder = None
                self._coder_error = None
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
            except Exception as exc:
                self._load_error = exc
                raise
            finally:
                self._loading = False

    def ensure_coder_loaded(self) -> None:
        if self._coder is not None:
            return
        with self._lock:
            if self._coder is not None:
                return
            try:
                self._coder = OpenRouterBackend(
                    api_key=self.OPENROUTER_API_KEY,
                    model=self.OPENROUTER_CODER_MODEL,
                    base_url=self.OPENROUTER_BASE_URL,
                    site_url=self.OPENROUTER_SITE_URL,
                    app_name=self.OPENROUTER_APP_NAME,
                    ca_bundle=self.OPENROUTER_CA_BUNDLE,
                    skip_ssl_verify=self.OPENROUTER_SKIP_SSL_VERIFY,
                )
                self._coder_error = None
            except Exception as exc:
                self._coder_error = exc
                raise

    @property
    def llm(self):
        if self._load_error:
            raise RuntimeError(f"Model failed to load: {self._load_error}") from self._load_error
        self.ensure_loaded()
        return self._llm

    @property
    def coder_llm(self):
        if self._coder_error:
            raise RuntimeError(f"Coder model failed to load: {self._coder_error}") from self._coder_error
        self.ensure_coder_loaded()
        return self._coder

    def _local_messages_for_generation(self, messages):
        model_name = str(self._local_model_path).lower()
        if "qwen3" not in model_name and "qwen-3" not in model_name:
            return messages
        items = messages if isinstance(messages, list) else list(messages)
        if any("/think" in content_to_text((m if isinstance(m, dict) else {}).get("content")).lower() or "/no_think" in content_to_text((m if isinstance(m, dict) else {}).get("content")).lower() for m in items):
            return items
        for index in range(len(items) - 1, -1, -1):
            msg = items[index] if isinstance(items[index], dict) else {}
            if msg.get("role") == "user":
                patched = list(items)
                patched[index] = {**msg, "content": f"{content_to_text(msg.get('content')).rstrip()}\n/no_think"}
                return patched
        return [*items, {"role": "user", "content": "/no_think"}]

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
        if role == "coder" and self.OPENROUTER_CODER_MODEL:
            return self.coder_llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=stream,
                response_format=response_format,
                timeout=timeout,
            )

        with self._lock:
            self._cancel_idle_timer()
        result = self.llm.create_chat_completion(
            messages=self._local_messages_for_generation(messages),
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
