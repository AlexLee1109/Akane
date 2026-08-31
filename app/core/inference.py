"""One priority-aware llama.cpp runtime with token-level streaming."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from app.core.character import load_character_profile
from app.core.config import SETTINGS


class InferenceCancelled(RuntimeError):
    pass


class InferenceQueueTimeout(RuntimeError):
    pass


@dataclass(slots=True)
class InferenceTiming:
    model_started_at: float = 0.0
    first_token_at: float = 0.0
    final_token_at: float = 0.0
    stop_observed_at: float = 0.0
    model_finished_at: float = 0.0
    prompt_tokens: int = 0
    prompt_token_method: str = "unavailable"
    prefill_tokens: int = 0
    previous_prompt_tokens: int = 0
    current_prompt_tokens: int = 0
    actual_token_lcp: int = 0
    actual_lcp_percent: float = 0.0
    tokens_after_lcp: int = 0
    backend_prompt_eval_tokens: int | None = None
    backend_reused_tokens: int | None = None
    backend_compute_graph_reuses: int | None = None
    llama_n_tokens_before: int = 0
    llama_n_tokens_after: int = 0
    cache_owner: str = "none"
    cache_epoch: int = 0
    cache_invalidated: bool = False
    cache_invalidated_reason: str = "none"
    cache_invalidated_from_token: int | None = None
    prompt_architecture: str = "rebuilt"
    static_prefix_tokens: int = 0
    historical_prefix_tokens: int = 0
    dynamic_state_tokens: int = 0
    new_user_tokens: int = 0
    self_tokens: int = 0
    memory_tokens: int = 0
    experience_tokens: int = 0
    time_tokens: int = 0
    code_context_tokens: int = 0
    wrapper_tokens: int = 0
    chat_template_overhead_tokens: int = 0
    state_revision_before: int | None = None
    state_revision_after: int | None = None
    state_payload_tokens: int = 0
    state_payload_mode: str = "none"
    context_trimmed: bool = False
    tokens_removed: int = 0
    turns_removed: int = 0
    cache_rebuild_reason: str = "none"
    logical_prompt_tokens: int = 0
    resident_sequence_tokens: int = 0
    appended_tokens: int = 0
    state_delta_tokens: int = 0
    template_overhead_tokens: int = 0
    health_before: dict[str, object] = field(default_factory=dict)
    health_after_prefill: dict[str, object] = field(default_factory=dict)
    health_after_decode: dict[str, object] = field(default_factory=dict)
    sequence_state_bytes: int = 0
    foreground_queue_wait_seconds: float = 0.0
    model_lock_wait_seconds: float = 0.0
    generated_tokens: int = 0
    generated_token_method: str = "unavailable"
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    finish_reason: str = ""


@dataclass(frozen=True, slots=True)
class Reservation:
    llm: object
    preemption: threading.Event
    queue_wait_seconds: float = 0.0
    priority: str = "owner"


DialogueCacheKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class _ConversationInferenceState:
    owner: DialogueCacheKey
    token_ids: tuple[int, ...]
    canonical_history: tuple[tuple[str, str], ...]
    state_revision: int
    state_section_fingerprints: tuple[tuple[str, str], ...] = ()
    synchronized_state_items: tuple[tuple[str, str, str, str, str], ...] = ()
    cache_epoch: int = 0


@dataclass(frozen=True, slots=True)
class _ChatContinuationState:
    first_probe: tuple[dict[str, str], ...]
    second_probe: tuple[dict[str, str], ...]
    boundary_prefix: tuple[int, ...]
    stop_sequences: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _DialoguePromptSelection:
    messages: list[dict[str, str]]
    tokens: tuple[int, ...]
    token_method: str
    architecture: str
    rebuild_reason: str
    canonical_history: tuple[tuple[str, str], ...]
    payload_sections: tuple[tuple[str, str], ...]
    payload_mode: str
    current_user_content: str
    append_tokens: tuple[int, ...] = ()
    direct_append: bool = False
    synchronized_state_items: tuple[tuple[str, str, str, str, str], ...] = ()
    tokens_removed: int = 0
    turns_removed: int = 0


def _content(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
            for item in value
        )
    return "" if value is None else str(value)


def _chat_formatter(llm):
    """Find llama-cpp-python's active embedded Jinja formatter when exposed."""

    try:
        from llama_cpp import llama_chat_format

        chat_format = getattr(llm, "chat_format", None)
        handler = (
            getattr(llm, "chat_handler", None)
            or getattr(llm, "_chat_handlers", {}).get(chat_format)
            or llama_chat_format.get_chat_completion_handler(chat_format)
        )
        for cell in handler.__closure__ or ():
            if isinstance(cell.cell_contents, llama_chat_format.Jinja2ChatFormatter):
                return cell.cell_contents
    except Exception:  # formatter lookup differs across binding/model versions
        return None
    return None


def _render_chat_prompt(llm, messages) -> str | None:
    formatter = _chat_formatter(llm)
    if formatter is None:
        return None
    try:
        return str(formatter(messages=list(messages)).prompt)
    except (AttributeError, TypeError, ValueError):
        return None


def _render_chat_tokens(llm, messages) -> tuple[tuple[int, ...], str]:
    """Tokenize exactly as the active Jinja chat handler does when inspectable."""

    formatter = _chat_formatter(llm)
    if formatter is not None:
        try:
            rendered = formatter(messages=list(messages))
            tokens = llm.tokenize(
                rendered.prompt.encode("utf-8"),
                add_bos=not bool(rendered.added_special),
                special=True,
            )
            return tuple(int(token) for token in tokens), "exact_active_chat_handler"
        except (AttributeError, TypeError, ValueError):
            pass
    prompt = _render_chat_prompt(llm, messages)
    if prompt is None:
        return (), "unavailable"
    try:
        tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        return tuple(int(token) for token in tokens), "rendered_template_specials_unverified"
    except (AttributeError, TypeError, ValueError):
        return (), "unavailable"


class InferenceRuntime:
    _instance: "InferenceRuntime | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._path = Path(SETTINGS.model_path)
        self._llm = None
        self._load_error: Exception | None = None
        self._loading = False
        self._closing = False
        self._closed = False
        self._shutdown_epoch = 0
        self._load_lock = threading.RLock()
        self._condition = threading.Condition()
        self._active = False
        self._active_priority = ""
        self._active_preemption: threading.Event | None = None
        self._waiters: list[tuple[int, int]] = []
        self._foreground_requests = 0
        self._last_foreground_activity_at = time.time()
        self._sequence = 0
        self._last_wait = 0.0
        self._load_count = 0
        self._model_family = "unknown"
        self._thinking_disabled = False
        self._active_cache_types = {"k": "", "v": ""}
        self._live_cache_owner: DialogueCacheKey | None = None
        self._cache_epoch = 0
        self._last_cache_invalidated_reason = "runtime-created"
        self._previous_dialogue_prompt: tuple[DialogueCacheKey, tuple[int, ...]] | None = None
        self._previous_prompt_tokens_by_kind: dict[str, tuple[int, ...]] = {}
        self._conversation_state: _ConversationInferenceState | None = None
        self._chat_continuation_state: _ChatContinuationState | None = None
        self._chat_continuation_status = "not requested"
        self._pending_prompt_tokens: tuple[object, tuple[int, ...], str] | None = None
        self._dialogue_prompt_dumped = False
        self._system_role_supported: bool | None = None
        self._chat_template_sha256 = ""
        self._model_calls = {
            "dialogue": 0, "other": 0,
        }
        self._foreground_direct_append_count = 0
        self._foreground_canonical_rebuild_count = 0
        if SETTINGS.prompt_debug:
            print(f"[Akane:debug:runtime] instance_id={id(self):x}", flush=True)

    @classmethod
    def get_instance(cls) -> "InferenceRuntime":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def status(self) -> dict[str, object]:
        with self._condition:
            return {
                "loading": self._loading,
                "closing": self._closing,
                "closed": self._closed,
                "loaded": self._llm is not None,
                "error": str(self._load_error) if self._load_error else None,
                "backend": "llama_cpp",
                "local_model_path": str(self._path),
                "inference_active": self._active,
                "inference_priority": self._active_priority,
                "foreground_active": self._foreground_requests > 0,
                "foreground_requests": self._foreground_requests,
                "last_foreground_activity_at": self._last_foreground_activity_at,
                "last_queue_wait_seconds": self._last_wait,
                "model_load_count": self._load_count,
                "model_family": self._model_family,
                "thinking_disabled": self._thinking_disabled,
                "gemma_thinking_disabled": (
                    self._model_family == "gemma" and self._thinking_disabled
                ),
                "system_role_mode": self._system_role_mode(),
                "chat_template_sha256": self._chat_template_sha256 or "unavailable",
                "foreground_kv_ownership": "one live llama sequence",
                "cache_epoch": self._cache_epoch,
                "cache_invalidated_reason": self._last_cache_invalidated_reason,
                "direct_token_append": self._chat_continuation_status,
                "model_calls": dict(self._model_calls),
                "foreground_direct_append_count": self._foreground_direct_append_count,
                "foreground_canonical_rebuild_count": self._foreground_canonical_rebuild_count,
            }

    def inference_busy(self) -> bool:
        with self._condition:
            return self._active or bool(self._waiters)

    def model_call_counts(self) -> dict[str, int]:
        with self._condition:
            return dict(self._model_calls)

    def _record_model_call(self, kind: str) -> None:
        name = kind if kind in self._model_calls else "other"
        with self._condition:
            self._model_calls[name] += 1
            count = self._model_calls[name]
        if SETTINGS.timing_enabled:
            label = {
                "dialogue": "FOREGROUND_DIALOGUE",
                "other": "OTHER",
            }[name]
            print(
                f"[Akane:model_call] kind={label} count={count} "
                f"runtime_id={id(self):x} at={time.time():.6f}",
                flush=True,
            )

    def foreground_started(self, *, now: float | None = None) -> None:
        """Mark the whole visible request active, including context and persistence."""

        with self._condition:
            self._foreground_requests += 1
            self._last_foreground_activity_at = time.time() if now is None else float(now)
            self._condition.notify_all()

    def foreground_finished(self, *, now: float | None = None) -> None:
        with self._condition:
            self._foreground_requests = max(0, self._foreground_requests - 1)
            self._last_foreground_activity_at = time.time() if now is None else float(now)
            self._condition.notify_all()

    def _system_role_mode(self) -> str:
        if self._system_role_supported is True:
            return "native"
        if self._system_role_supported is False:
            return "first-user fallback"
        return "unverified"

    @staticmethod
    def _chat_template_source(llm) -> str:
        metadata = getattr(llm, "metadata", {}) or {}
        return str(
            metadata.get("tokenizer.chat_template")
            or metadata.get("tokenizer.ggml.chat_template")
            or getattr(llm, "chat_format", "")
            or ""
        )

    def _configure_chat_template(self, llm) -> None:
        template = self._chat_template_source(llm)
        self._chat_template_sha256 = (
            hashlib.sha256(template.encode("utf-8")).hexdigest() if template else ""
        )
        sentinel = "AKANE_SYSTEM_ROLE_PROBE_7B3D"
        rendered = _render_chat_prompt(llm, (
            {"role": "system", "content": sentinel},
            {"role": "user", "content": "hello"},
        ))
        # If the exact active formatter cannot be rendered, preserving a native
        # system role is unproven. The first-user form is safe for templates that
        # silently discard system messages (including some Gemma formatters).
        self._system_role_supported = rendered is not None and sentinel in rendered
        if SETTINGS.prompt_debug:
            print(
                f"[Akane:debug:chat_template] sha256={self._chat_template_sha256 or 'unavailable'} "
                f"system_role_mode={self._system_role_mode()!r}",
                flush=True,
            )
        self._prepare_chat_continuation(llm)

    @staticmethod
    def _token_suffix_length(left, right) -> int:
        matched = 0
        for first, second in zip(reversed(left), reversed(right)):
            if int(first) != int(second):
                break
            matched += 1
        return matched

    def _prepare_chat_continuation(self, llm) -> None:
        """Verify a history-independent token boundary for direct turn appends."""

        self._chat_continuation_state = None
        probe_user = "AKANE_CONTINUATION_USER_51C8"
        first_assistant = "AKANE_ASSISTANT_ALPHA_8D21"
        second_assistant = "AKANE_ASSISTANT_BETA_7F34"
        first = tuple(self._backend_messages((
            {"role": "user", "content": probe_user},
            {"role": "assistant", "content": first_assistant},
        )))
        second = tuple(self._backend_messages((
            {"role": "user", "content": probe_user},
            {"role": "assistant", "content": second_assistant},
        )))
        first_next = (*first, {"role": "user", "content": "X_AKANE_NEXT_3A19"})
        second_next = (*second, {"role": "user", "content": "X_AKANE_NEXT_3A19"})
        first_other = (*first, {"role": "user", "content": "Y_AKANE_NEXT_6C42"})
        second_other = (*second, {"role": "user", "content": "Y_AKANE_NEXT_6C42"})
        first_tokens, first_method = _render_chat_tokens(llm, first_next)
        second_tokens, second_method = _render_chat_tokens(llm, second_next)
        other_first_tokens, other_first_method = _render_chat_tokens(llm, first_other)
        other_second_tokens, other_second_method = _render_chat_tokens(llm, second_other)
        suffix_length = self._token_suffix_length(first_tokens, second_tokens)
        other_suffix_length = self._token_suffix_length(other_first_tokens, other_second_tokens)
        suffix = first_tokens[-suffix_length:] if suffix_length else ()
        other_suffix = other_first_tokens[-other_suffix_length:] if other_suffix_length else ()
        boundary_length = self._token_lcp(suffix, other_suffix)
        if (
            first_method != "exact_active_chat_handler"
            or second_method != first_method
            or other_first_method != first_method
            or other_second_method != first_method
            or not first_tokens
            or not suffix_length
            or not other_suffix_length
            or not boundary_length
        ):
            self._chat_continuation_status = "template boundary unavailable"
            return
        if suffix_length >= len(first_tokens) or other_suffix_length >= len(other_first_tokens):
            self._chat_continuation_status = "template boundary ambiguous"
            return
        self._chat_continuation_state = _ChatContinuationState(
            first, second, tuple(suffix[:boundary_length]),
            self._chat_template_stop_sequences(llm, first_next),
        )
        self._chat_continuation_status = "verified"

    def _chat_template_stop_sequences(self, llm, messages) -> tuple[str, ...]:
        formatter = _chat_formatter(llm)
        if formatter is None:
            return ()
        try:
            stop = formatter(messages=list(messages)).stop
        except (AttributeError, TypeError, ValueError):
            return ()
        values = stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else []
        result = []
        for value in values:
            text = str(value)
            if not text:
                continue
            try:
                tokens = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
            except (AttributeError, TypeError, ValueError):
                tokens = ()
            if tokens and all(self._is_eog_token(llm, int(token)) for token in tokens):
                continue
            result.append(text)
        return tuple(result)

    def _continuation_tokens(self, llm, user_content: str) -> tuple[int, ...]:
        state = self._chat_continuation_state
        if state is None:
            return ()
        first = (*state.first_probe, {"role": "user", "content": user_content})
        second = (*state.second_probe, {"role": "user", "content": user_content})
        first_tokens, first_method = _render_chat_tokens(llm, first)
        second_tokens, second_method = _render_chat_tokens(llm, second)
        suffix_length = self._token_suffix_length(first_tokens, second_tokens)
        suffix = first_tokens[-suffix_length:] if suffix_length else ()
        if (
            first_method != "exact_active_chat_handler"
            or second_method != first_method
            or not suffix_length
            or tuple(suffix[:len(state.boundary_prefix)]) != state.boundary_prefix
        ):
            self._chat_continuation_state = None
            self._chat_continuation_status = "runtime boundary mismatch"
            return ()
        return tuple(suffix)

    def _backend_messages(self, messages) -> list[dict[str, str]]:
        values = [
            {"role": str(message.get("role") or "user"), "content": str(message.get("content") or "")}
            for message in messages
        ]
        if self._system_role_supported is not False:
            return values
        system = "\n\n".join(
            message["content"] for message in values if message["role"] == "system"
        )
        values = [message for message in values if message["role"] != "system"]
        if not system:
            return values
        for message in values:
            if message["role"] == "user":
                message["content"] = (
                    f"# System instructions\n{system}\n\n# Conversation\n{message['content']}"
                )
                return values
        return [{"role": "user", "content": f"# System instructions\n{system}"}, *values]

    def dialogue_cache_fingerprint(
        self,
        static_prompt_hash: str,
        reservation: Reservation,
    ) -> str:
        material = "\0".join((
            static_prompt_hash,
            self._chat_template_source(reservation.llm),
            self._system_role_mode(),
        ))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def dialogue_fast_path_available(
        self,
        cache_key: DialogueCacheKey | None,
        history_messages,
        *,
        max_tokens: int,
    ) -> bool:
        """Cheap admission check; exact suffix fitting happens inside generation."""

        state = self._conversation_state
        if (
            cache_key is None
            or self._live_cache_owner != cache_key
            or state is None
            or state.owner != cache_key
            or state.cache_epoch != self._cache_epoch
            or self._chat_continuation_state is None
            or not self._history_continues(
                state.canonical_history, self._canonical_history(history_messages),
            )
        ):
            return False
        llm = self._llm
        n_tokens = max(0, int(getattr(llm, "n_tokens", 0)))
        resident = getattr(llm, "input_ids", ())[:n_tokens]
        if n_tokens > len(state.token_ids) or self._token_lcp(resident, state.token_ids) != n_tokens:
            return False
        # Near the boundary, use canonical fitting so a rebase cannot surprise
        # the low-level generator. Ordinary turns avoid full prompt tokenization.
        return len(state.token_ids) + max_tokens + 256 < SETTINGS.llama_context_window

    def runtime_report(self, *, include_model_hash: bool = False) -> dict[str, object]:
        path = self._path.expanduser().resolve()
        digest = "not requested"
        if include_model_hash and path.is_file():
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        try:
            binding = importlib.metadata.version("llama-cpp-python")
        except importlib.metadata.PackageNotFoundError:
            binding = "not installed"
        effective: dict[str, object] = {}
        supported_init_options: set[str] = set()
        try:
            from llama_cpp import Llama

            effective = self._load_options(Llama)
            signature = inspect.signature(Llama.__init__).parameters
            supported_init_options = set(signature)
            for name in ("numa", "type_k", "type_v", "seed"):
                if name in signature and name not in effective:
                    effective[name] = signature[name].default
        except (ImportError, TypeError, ValueError):
            pass
        effective.pop("model_path", None)
        requested_cache_types = {
            "k": SETTINGS.llama_cache_type_k or "model default",
            "v": SETTINGS.llama_cache_type_v or "model default",
        }
        llm = self._llm
        effective_cache_types = {}
        for axis in ("k", "v"):
            requested = requested_cache_types[axis]
            if requested == "model default":
                effective_cache_types[axis] = "model default"
            elif f"type_{axis}" not in supported_init_options:
                effective_cache_types[axis] = "unsupported by installed binding"
            elif llm is None:
                effective_cache_types[axis] = "pending model load"
            else:
                effective_cache_types[axis] = self._active_cache_types[axis] or "not applied"
        backend_system_info = "unavailable until model load"
        if llm is not None:
            try:
                from llama_cpp import llama_cpp

                raw_info = llama_cpp.llama_print_system_info()
                if isinstance(raw_info, bytes):
                    backend_system_info = raw_info.decode("utf-8", errors="replace") or "unavailable"
                elif raw_info:
                    backend_system_info = str(raw_info)
            except (AttributeError, ImportError, TypeError, ValueError):
                backend_system_info = "unavailable"
        return {
            "model_path": str(path),
            "model_size": path.stat().st_size if path.is_file() else 0,
            "model_sha256": digest,
            "llama_cpp_python": binding,
            "llama_cpp_backend_version": "not exposed by installed binding",
            "machine_architecture": platform.machine(),
            "platform": platform.platform(),
            "llama_backend_system_info": backend_system_info,
            "context_window": SETTINGS.llama_context_window,
            "temperature": SETTINGS.temperature,
            "top_p": SETTINGS.top_p,
            "top_k": SETTINGS.top_k,
            "min_p": SETTINGS.min_p,
            "repeat_penalty": SETTINGS.repetition_penalty,
            "max_generation_tokens": SETTINGS.max_tokens,
            "stop_sequences": SETTINGS.generation_stop_sequences,
            "effective_init_options": effective,
            "requested_cache_type_k": requested_cache_types["k"],
            "requested_cache_type_v": requested_cache_types["v"],
            "effective_cache_type_k": effective_cache_types["k"],
            "effective_cache_type_v": effective_cache_types["v"],
            "cache_type_k_supported": "type_k" in supported_init_options,
            "cache_type_v_supported": "type_v" in supported_init_options,
            "chat_format": getattr(llm, "chat_format", None) if llm is not None else "embedded model template",
            "eos_token": llm.token_eos() if llm is not None else "model default",
            "live_kv_prefix_reuse": "llama-cpp-python longest-prefix reuse",
            "response_cache_enabled": False,
            "model_family": self._model_family,
            "thinking_enabled_requested": SETTINGS.llama_enable_thinking,
            "thinking_disabled": self._thinking_disabled,
            "thinking_verification": (
                "active Jinja formatter enable_thinking global forced false"
                if self._thinking_disabled else "not disabled"
            ),
            "model_load_count": self._load_count,
            "system_role_mode": self._system_role_mode(),
            "chat_template_source": "GGUF embedded tokenizer.chat_template",
            "chat_template_sha256": self._chat_template_sha256 or "unavailable",
        }

    def _load_options(self, llama_type) -> dict[str, object]:
        options: dict[str, object] = {
            "model_path": str(self._path),
            "n_ctx": SETTINGS.llama_context_window,
            "n_batch": SETTINGS.llama_batch_size,
            "n_ubatch": SETTINGS.llama_ubatch_size,
            "n_threads": SETTINGS.llama_threads,
            "n_threads_batch": SETTINGS.llama_threads_batch,
            "flash_attn": SETTINGS.llama_flash_attn,
            "n_gpu_layers": SETTINGS.llama_gpu_layers,
            "offload_kqv": SETTINGS.llama_offload_kqv,
            "op_offload": SETTINGS.llama_op_offload,
            "use_mmap": SETTINGS.llama_use_mmap,
            "use_mlock": SETTINGS.llama_use_mlock,
            "swa_full": SETTINGS.llama_swa_full,
            "embedding": False,
            "logits_all": False,
            "verbose": False,
        }
        try:
            supported = inspect.signature(llama_type.__init__).parameters
        except (TypeError, ValueError):
            return options
        for name, configured in (
            ("type_k", SETTINGS.llama_cache_type_k),
            ("type_v", SETTINGS.llama_cache_type_v),
        ):
            if configured and name in supported:
                options[name] = self._ggml_cache_type(configured)
        return {key: value for key, value in options.items() if key in supported}

    @staticmethod
    def _ggml_cache_type(name: str) -> int:
        from llama_cpp import llama_cpp

        constant = f"GGML_TYPE_{name.upper()}"
        try:
            return int(getattr(llama_cpp, constant))
        except AttributeError as exc:
            raise RuntimeError(
                f"Installed llama-cpp-python does not expose requested KV cache type {name!r}."
            ) from exc

    def ensure_loaded(self) -> None:
        with self._condition:
            if self._closing or self._closed:
                raise InferenceCancelled("Inference runtime is shut down.")
        if self._llm is not None:
            return
        with self._load_lock:
            with self._condition:
                if self._closing or self._closed:
                    raise InferenceCancelled("Inference runtime is shut down.")
            if self._llm is not None:
                return
            self._loading = True
            self._load_error = None
            llm = None
            try:
                from llama_cpp import Llama

                load_options = self._load_options(Llama)
                llm = Llama(**load_options)
                active_cache_types = {
                    axis: (
                        getattr(SETTINGS, f"llama_cache_type_{axis}")
                        if f"type_{axis}" in load_options else ""
                    )
                    for axis in ("k", "v")
                }
                metadata = getattr(llm, "metadata", {}) or {}
                model_text = " ".join((str(self._path), *(str(value) for value in metadata.values()))).casefold()
                model_family = (
                    "qwen" if "qwen" in model_text else
                    "gemma" if "gemma" in model_text else
                    "unknown"
                )
                template = metadata.get("tokenizer.chat_template") or metadata.get("tokenizer.ggml.chat_template")
                embedded_handlers = getattr(llm, "_chat_handlers", {})
                if (
                    template
                    and getattr(llm, "chat_handler", None) is None
                    and "chat_template.default" in embedded_handlers
                ):
                    # Use the GGUF's actual template rather than a guessed
                    # built-in formatter so fitting, diagnostics, and generation
                    # all share one inspectable rendering path.
                    llm.chat_format = "chat_template.default"
                if model_family in {"gemma", "qwen"} and not template:
                    raise RuntimeError(
                        f"The configured {model_family.title()} GGUF has no embedded chat template."
                    )
                thinking_disabled = False
                if model_family in {"gemma", "qwen"} and not SETTINGS.llama_enable_thinking:
                    formatter = _chat_formatter(llm)
                    if formatter is None and "enable_thinking" in str(template):
                        raise RuntimeError(
                            f"Could not disable thinking in the configured {model_family.title()} "
                            "chat template."
                        )
                    if formatter is not None:
                        formatter._environment.globals["enable_thinking"] = False
                    thinking_disabled = True
                self._configure_chat_template(llm)
                if SETTINGS.prompt_debug:
                    self._log_character_sources(llm)
                self._active_cache_types = active_cache_types
                self._model_family = model_family
                self._thinking_disabled = thinking_disabled
                self._llm = llm
                self._load_count += 1
            except Exception as exc:
                self._llm = None
                self._model_family = "unknown"
                self._thinking_disabled = False
                self._active_cache_types = {"k": "", "v": ""}
                self._chat_continuation_state = None
                self._chat_continuation_status = "model load failed"
                self._system_role_supported = None
                self._chat_template_sha256 = ""
                self._load_error = exc
                close = getattr(llm, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception as close_error:
                        exc.add_note(
                            f"The partially initialized model also failed to close: {close_error}"
                        )
                raise
            finally:
                self._loading = False

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closing = True
            self._shutdown_epoch += 1
            if self._active_preemption is not None:
                self._active_preemption.set()
            deadline = time.monotonic() + 5.0
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._closing = False
                    self._condition.notify_all()
                    raise RuntimeError("Timed out waiting for active inference to stop during shutdown.")
                self._condition.wait(min(0.1, remaining))
        try:
            with self._load_lock:
                llm, self._llm = self._llm, None
                self._load_error = None
                self._model_family = "unknown"
                self._thinking_disabled = False
                self._active_cache_types = {"k": "", "v": ""}
                self._live_cache_owner = None
                self._previous_dialogue_prompt = None
                self._previous_prompt_tokens_by_kind.clear()
                self._conversation_state = None
                self._chat_continuation_state = None
                self._chat_continuation_status = "runtime closed"
                self._pending_prompt_tokens = None
                self._cache_epoch += 1
                self._last_cache_invalidated_reason = "runtime-closed"
                self._system_role_supported = None
                self._chat_template_sha256 = ""
            close = getattr(llm, "close", None)
            if close is not None:
                close()
        finally:
            with self._condition:
                self._closed = True
                self._closing = False
                self._condition.notify_all()

    @property
    def llm(self):
        if self._load_error is not None:
            raise RuntimeError(f"Model failed to load: {self._load_error}") from self._load_error
        self.ensure_loaded()
        return self._llm

    @staticmethod
    def _log_character_sources(llm) -> None:
        from app.core.prompt import stable_prompt_hash, stable_system_prompt

        character = load_character_profile()
        contents = (
            (
                "identity",
                character.identity_path,
                character.identity_mtime_ns,
                character.identity_sha256,
                character.identity,
            ),
            (
                "soul",
                character.soul_path,
                character.soul_mtime_ns,
                character.soul_sha256,
                character.voice,
            ),
        )
        for label, path, mtime_ns, digest, content in contents:
            tokens = len(llm.tokenize(content.encode("utf-8"), add_bos=False, special=True))
            print(
                f"[Akane:debug:character] {label}_path={str(path)!r} "
                f"{label}_mtime_ns={mtime_ns} {label}_sha256={digest} "
                f"{label}_tokens={tokens}",
                flush=True,
            )
        combined = f"{character.identity}\n\n{character.voice}"
        combined_tokens = len(llm.tokenize(
            combined.encode("utf-8"), add_bos=False, special=True,
        ))
        print(
            f"[Akane:debug:character] content_sha256={character.content_sha256} "
            f"character_combined_tokens={combined_tokens}",
            flush=True,
        )
        stable_prompt = stable_system_prompt(character)
        stable_tokens = len(llm.tokenize(
            stable_prompt.encode("utf-8"), add_bos=False, special=True,
        ))
        print(
            f"[Akane:debug:character] stable_character_prompt_sha256={stable_prompt_hash(character)} "
            f"stable_character_prompt_tokens={stable_tokens} "
            "reload_semantics='content hash checked on every load'",
            flush=True,
        )

    @staticmethod
    def _clear_model_state(llm) -> None:
        llm._ctx.kv_cache_clear()
        llm.n_tokens = 0

    def _prepare_dialogue_cache(self, llm, owner: DialogueCacheKey) -> str:
        current_owner = self._live_cache_owner
        n_tokens = int(getattr(llm, "n_tokens", 0))
        if current_owner == owner and n_tokens > 0:
            self._last_cache_invalidated_reason = "none"
            return "none"
        if current_owner == owner:
            reason = "llama-state-empty"
        elif current_owner is None:
            reason = "unowned-llama-state" if n_tokens > 0 else "no-live-cache"
        elif current_owner[0] != owner[0]:
            reason = "profile-changed"
        elif current_owner[1] != owner[1]:
            reason = "conversation-changed"
        else:
            reason = "static-prompt-or-template-changed"
        if int(getattr(llm, "n_tokens", 0)) > 0:
            self._clear_model_state(llm)
        if current_owner is not None or n_tokens > 0:
            self._cache_epoch += 1
        self._live_cache_owner = None
        if self._previous_dialogue_prompt is not None and self._previous_dialogue_prompt[0] != owner:
            self._previous_dialogue_prompt = None
        if self._conversation_state is not None and self._conversation_state.owner != owner:
            self._conversation_state = None
        self._last_cache_invalidated_reason = reason
        return reason

    @staticmethod
    def _cache_owner_label(owner: DialogueCacheKey | None) -> str:
        if owner is None:
            return "none"
        return hashlib.sha256("\0".join(owner).encode("utf-8")).hexdigest()[:12]

    def discard_dialogue_cache(
        self,
        profile_id: str,
        conversation_id: str | None = None,
        *,
        reason: str = "explicit-discard",
    ) -> None:
        with self._condition:
            owner = self._live_cache_owner
            previous_owner = self._previous_dialogue_prompt[0] if self._previous_dialogue_prompt else None
            state_owner = self._conversation_state.owner if self._conversation_state else None
            matches_live = owner is not None and owner[0] == profile_id and (
                conversation_id is None or owner[1] == conversation_id
            )
            matches_previous = previous_owner is not None and previous_owner[0] == profile_id and (
                conversation_id is None or previous_owner[1] == conversation_id
            )
            matches_state = state_owner is not None and state_owner[0] == profile_id and (
                conversation_id is None or state_owner[1] == conversation_id
            )
            if matches_live or matches_previous or matches_state:
                self._live_cache_owner = None
                self._previous_dialogue_prompt = None
                self._conversation_state = None
                self._pending_prompt_tokens = None
                self._cache_epoch += 1
                self._last_cache_invalidated_reason = reason

    @contextmanager
    def reserve(
        self,
        *,
        priority: str = "visible",
        cancellation: threading.Event | None = None,
        queue_deadline: float | None = None,
    ):
        ranks = {"owner": 0, "visible": 0, "guest": 1}
        if priority not in ranks:
            raise ValueError("Unknown inference priority.")
        queued_at = time.monotonic()
        preemption = threading.Event()
        waiter: tuple[int, int] | None = None
        acquired = False
        with self._condition:
            if self._closing or self._closed:
                raise InferenceCancelled("Inference runtime is shutting down.")
            shutdown_epoch = self._shutdown_epoch
            self._sequence += 1
            waiter = (ranks[priority], self._sequence)
            self._waiters.append(waiter)
            try:
                while self._active or waiter != min(self._waiters):
                    if self._closing or self._closed or self._shutdown_epoch != shutdown_epoch:
                        raise InferenceCancelled("Inference runtime is shutting down.")
                    if cancellation is not None and cancellation.is_set():
                        raise InferenceCancelled("Inference was cancelled while queued.")
                    if queue_deadline is not None and time.monotonic() >= queue_deadline:
                        raise InferenceQueueTimeout("Inference timed out while queued.")
                    self._condition.wait(0.1)
                if self._closing or self._closed or self._shutdown_epoch != shutdown_epoch:
                    raise InferenceCancelled("Inference runtime is shutting down.")
                self._waiters.remove(waiter)
                waiter = None
                if cancellation is not None and cancellation.is_set():
                    raise InferenceCancelled("Inference was cancelled before it started.")
                self._active = True
                self._active_priority = priority
                self._active_preemption = preemption
                self._last_foreground_activity_at = time.time()
                self._last_wait = time.monotonic() - queued_at
                acquired = True
            finally:
                if waiter is not None and waiter in self._waiters:
                    self._waiters.remove(waiter)
                if not acquired:
                    self._condition.notify_all()
        try:
            yield Reservation(self.llm, preemption, self._last_wait, priority)
        finally:
            with self._condition:
                self._active = False
                self._active_priority = ""
                self._active_preemption = None
                self._last_foreground_activity_at = time.time()
                self._condition.notify_all()

    def count_prompt_tokens(
        self,
        messages: tuple[dict[str, str], ...] | list[dict[str, str]],
        reservation: Reservation,
    ) -> tuple[int, str]:
        backend_messages = self._backend_messages(messages)
        prompt_tokens, method = _render_chat_tokens(reservation.llm, backend_messages)
        if not prompt_tokens:
            characters = sum(len(message.get("content", "")) for message in backend_messages)
            return max(1, characters // 4), "estimated_characters"
        key = self._prompt_token_key(reservation.llm, backend_messages)
        self._pending_prompt_tokens = (key, prompt_tokens, method)
        return len(prompt_tokens), method

    @staticmethod
    def _prompt_token_key(llm, messages) -> object:
        material = json.dumps(
            [
                (str(message.get("role") or ""), str(message.get("content") or ""))
                for message in messages
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return id(llm), hashlib.sha256(material).digest()

    @staticmethod
    def _token_lcp(left, right) -> int:
        matched = 0
        for first, second in zip(left, right):
            if int(first) != int(second):
                break
            matched += 1
        return matched

    def _dialogue_prompt_tokens(self, llm, backend_messages) -> tuple[tuple[int, ...], str]:
        key = self._prompt_token_key(llm, backend_messages)
        pending = self._pending_prompt_tokens
        if pending is not None and pending[0] == key:
            return pending[1], pending[2]
        return _render_chat_tokens(llm, backend_messages)

    @staticmethod
    def _canonical_history(messages) -> tuple[tuple[str, str], ...]:
        result: list[tuple[str, str]] = []
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
            elif isinstance(message, (tuple, list)) and len(message) == 2:
                role, content = message
            else:
                role = getattr(message, "role", "")
                content = getattr(message, "content", "")
            result.append((str(role or ""), str(content or "")))
        return tuple(result)

    @staticmethod
    def _history_continues(
        stored: tuple[tuple[str, str], ...],
        current: tuple[tuple[str, str], ...],
    ) -> bool:
        if current == stored:
            return True
        if not current or len(current) > len(stored) or stored[-len(current):] != current:
            return False
        # Amortize rolling-window rebases: retain append-only reuse for a few
        # exchanges, then rebuild from the configured recent-history window.
        slack = max(2, SETTINGS.recent_turn_limit // 2)
        slack -= slack % 2
        return len(stored) - len(current) <= slack

    @staticmethod
    def _state_section_fingerprints(
        sections: tuple[tuple[str, str], ...],
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, hashlib.sha256(text.encode("utf-8")).hexdigest())
            for name, text in sections
        )

    @staticmethod
    def _sync_record(item) -> tuple[str, str, str, str, str]:
        return (
            str(item.key), str(item.section), str(item.version),
            str(item.wire), str(item.clear_wire),
        )

    @staticmethod
    def _state_patch_wire(wire: str, operation: str) -> str:
        if wire.startswith(("S ", "M ", "E ")):
            return f"{wire[0]}{operation} {wire[2:]}"
        return wire

    def _state_delta(
        self,
        state: _ConversationInferenceState,
        state_items,
    ) -> tuple[
        tuple[tuple[str, str], ...],
        tuple[tuple[str, str, str, str, str], ...],
    ]:
        previous = {record[0]: record for record in state.synchronized_state_items}
        selected = {str(item.key): item for item in state_items}
        patches: list[tuple[str, str]] = []
        synchronized = dict(previous)

        for key, record in previous.items():
            if key in selected:
                continue
            patches.append((record[1], record[4]))
            synchronized.pop(key, None)

        for item in state_items:
            key = str(item.key)
            if not bool(item.persistent):
                synchronized.pop(key, None)
                patches.append((str(item.section), str(item.wire)))
                continue
            record = previous.get(key)
            if record is None:
                patches.append((str(item.section), self._state_patch_wire(str(item.wire), "+")))
            elif record[2] != str(item.version):
                patches.append((str(item.section), self._state_patch_wire(str(item.wire), "~")))
            synchronized[key] = self._sync_record(item)

        grouped: list[tuple[str, str]] = []
        for section, wire in patches:
            if grouped and grouped[-1][0] == section:
                grouped[-1] = (section, grouped[-1][1] + "\n" + wire)
            else:
                grouped.append((section, wire))
        return tuple(grouped), tuple(synchronized.values())

    def _select_dialogue_prompt(
        self,
        llm,
        messages,
        cache_key: DialogueCacheKey,
        *,
        max_tokens: int,
        history_messages,
        turn_user_content: str,
        state_sections: tuple[tuple[str, str], ...],
        state_items=(),
        canonical_rebuild=None,
    ) -> _DialoguePromptSelection:
        from app.core.prompt import TRANSIENT_STATE_SECTIONS, compose_dialogue_update

        current_history = self._canonical_history(history_messages)
        state = self._conversation_state
        if (
            self._live_cache_owner == cache_key
            and state is not None
            and state.owner == cache_key
            and state.cache_epoch == self._cache_epoch
            and self._history_continues(state.canonical_history, current_history)
        ):
            if state_items or state.synchronized_state_items:
                payload_sections, synchronized_items = self._state_delta(
                    state, state_items,
                )
                cleared_sections = ()
            else:
                previous_sections = dict(state.state_section_fingerprints)
                current_sections = dict(self._state_section_fingerprints(state_sections))
                payload_sections = tuple(
                    (name, text)
                    for name, text in state_sections
                    if name in TRANSIENT_STATE_SECTIONS
                    or previous_sections.get(name) != current_sections[name]
                )
                cleared_sections = tuple(
                    name for name in previous_sections
                    if name not in current_sections and name not in TRANSIENT_STATE_SECTIONS
                )
                synchronized_items = ()
            content = compose_dialogue_update(
                payload_sections, turn_user_content, cleared=cleared_sections,
            )
            continuation = self._continuation_tokens(llm, content)
            n_tokens = max(0, int(getattr(llm, "n_tokens", 0)))
            resident = tuple(int(token) for token in getattr(llm, "input_ids", ())[:n_tokens])
            resident_lcp = self._token_lcp(resident, state.token_ids)
            resident_is_valid = (
                n_tokens <= len(state.token_ids)
                and resident_lcp == n_tokens
            )
            append_tokens = (
                state.token_ids[n_tokens:] + continuation
                if continuation and resident_is_valid else ()
            )
            logical_tokens = state.token_ids + continuation if append_tokens else ()
            if (
                append_tokens
                and hasattr(llm, "generate")
                and hasattr(llm, "detokenize")
                and len(logical_tokens) + max_tokens <= SETTINGS.llama_context_window
            ):
                return _DialoguePromptSelection(
                    [{"role": "user", "content": content}],
                    logical_tokens,
                    "resident_tokens+verified_template_continuation",
                    "direct-token-append",
                    "none",
                    state.canonical_history,
                    payload_sections,
                    "delta" if payload_sections or cleared_sections else "none",
                    content,
                    append_tokens,
                    True,
                    synchronized_items,
                )
            if continuation and resident_is_valid:
                rebase_reason = "context-window-rebase"
                candidate_tokens = len(logical_tokens)
            elif not continuation:
                rebase_reason = "direct-append-template-unavailable"
                candidate_tokens = 0
            else:
                rebase_reason = "resident-sequence-mismatch"
                candidate_tokens = 0
        elif state is not None and state.owner == cache_key:
            rebase_reason = "canonical-history-mismatch"
            candidate_tokens = 0
        else:
            rebase_reason = "cold-or-untracked-conversation"
            candidate_tokens = 0
        if callable(canonical_rebuild):
            rebuilt = canonical_rebuild()
            messages = rebuilt.messages
            current_history = self._canonical_history(rebuilt.history_messages)
        fallback = self._backend_messages(messages)
        tokens, method = self._dialogue_prompt_tokens(llm, fallback)
        synchronized_items = tuple(
            self._sync_record(item) for item in state_items if bool(item.persistent)
        )
        return _DialoguePromptSelection(
            fallback,
            tokens,
            method,
            "canonical-rebuild",
            rebase_reason,
            current_history,
            state_sections,
            "full",
            str(messages[-1].get("content") or ""),
            append_tokens=(),
            direct_append=False,
            synchronized_state_items=synchronized_items,
            tokens_removed=max(0, candidate_tokens - len(tokens)),
            turns_removed=(
                max(0, len(state.canonical_history) - len(current_history))
                if rebase_reason == "context-window-rebase" and state is not None else 0
            ),
        )

    def _prepare_prompt_diagnostics(
        self,
        llm,
        timing: InferenceTiming,
        cache_key: DialogueCacheKey,
        prompt_tokens: tuple[int, ...],
        invalidated_reason: str,
    ) -> None:
        previous = self._previous_dialogue_prompt
        previous_tokens = previous[1] if previous is not None and previous[0] == cache_key else ()
        actual_lcp = self._token_lcp(previous_tokens, prompt_tokens)
        n_tokens = max(0, int(getattr(llm, "n_tokens", 0)))
        resident = getattr(llm, "input_ids", ())[:n_tokens]
        # llama-cpp-python deliberately leaves the final prompt token to evaluate
        # so it has fresh logits even when the entire preceding prefix matches.
        backend_reused = self._token_lcp(resident, prompt_tokens[:-1])
        timing.previous_prompt_tokens = len(previous_tokens)
        timing.current_prompt_tokens = len(prompt_tokens)
        timing.actual_token_lcp = actual_lcp
        timing.actual_lcp_percent = (
            actual_lcp * 100.0 / len(prompt_tokens) if prompt_tokens else 0.0
        )
        timing.tokens_after_lcp = max(0, len(prompt_tokens) - actual_lcp)
        timing.backend_reused_tokens = backend_reused
        timing.llama_n_tokens_before = n_tokens
        timing.cache_owner = self._cache_owner_label(cache_key)
        timing.cache_epoch = self._cache_epoch
        timing.cache_invalidated = invalidated_reason != "none"
        timing.cache_invalidated_reason = invalidated_reason
        timing.cache_invalidated_from_token = (
            backend_reused if timing.cache_invalidated else None
        )

    def _record_foreground_architecture(
        self,
        architecture: str,
        invalidated_reason: str,
    ) -> None:
        with self._condition:
            if architecture == "direct-token-append":
                self._foreground_direct_append_count += 1
            else:
                self._foreground_canonical_rebuild_count += 1

    def debug_dialogue_prompt_once(
        self,
        messages: tuple[dict[str, str], ...],
        *,
        llm,
        token_sections: dict[str, str],
        cache_key: DialogueCacheKey | None,
    ) -> None:
        if not SETTINGS.prompt_debug:
            return
        with self._condition:
            if self._dialogue_prompt_dumped:
                return
            self._dialogue_prompt_dumped = True
        backend_messages = self._backend_messages(messages)
        prompt = _render_chat_prompt(llm, backend_messages)
        if prompt is None:
            print(
                "[Akane:debug:dialogue_prompt_once] rendered_prompt=unavailable",
                flush=True,
            )
            return
        section_tokens = {
            name: len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))
            for name, text in token_sections.items()
            if text
        }
        exact_tokens, prompt_token_method = _render_chat_tokens(llm, backend_messages)
        prompt_tokens = len(exact_tokens)
        payload = {
            "messages": list(messages),
            "backend_messages": backend_messages,
            "section_tokens": section_tokens,
            "final_prompt_tokens": prompt_tokens,
            "final_prompt_token_method": prompt_token_method,
            "final_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "cache_owner": self._cache_owner_label(cache_key),
            "rendered_prompt": prompt,
        }
        print(
            "[Akane:debug:dialogue_prompt_once] "
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )

    @staticmethod
    def count_text_tokens(text: str, reservation: Reservation) -> tuple[int, str]:
        try:
            return len(reservation.llm.tokenize(
                str(text).encode("utf-8"), add_bos=False, special=True,
            )), "exact_model_tokenizer"
        except (AttributeError, TypeError, ValueError):
            return max(0, len(str(text)) // 4), "estimated_characters"

    @staticmethod
    def _reset_backend_timings(llm) -> None:
        try:
            llm._ctx.reset_timings()
        except AttributeError:
            pass

    @staticmethod
    def _capture_backend_timings(llm, timing: InferenceTiming, stream_chunks: int) -> None:
        try:
            from llama_cpp import llama_cpp

            perf = llama_cpp.llama_perf_context(llm._ctx.ctx)
            timing.prefill_tokens = max(0, int(perf.n_p_eval))
            timing.backend_prompt_eval_tokens = timing.prefill_tokens
            compute_graph_reuses = getattr(perf, "n_reused", None)
            timing.backend_compute_graph_reuses = (
                max(0, int(compute_graph_reuses))
                if compute_graph_reuses is not None else None
            )
            timing.generated_tokens = max(0, int(perf.n_eval))
            timing.prefill_seconds = max(0.0, float(perf.t_p_eval_ms) / 1000.0)
            timing.decode_seconds = max(0.0, float(perf.t_eval_ms) / 1000.0)
            timing.generated_token_method = "llama_perf_context"
        except (AttributeError, ImportError, TypeError, ValueError):
            if not timing.generated_tokens:
                timing.generated_tokens = stream_chunks
                timing.generated_token_method = "stream_chunks"
            if not timing.prefill_seconds and timing.first_token_at:
                timing.prefill_seconds = max(0.0, timing.first_token_at - timing.model_started_at)
            if not timing.decode_seconds and timing.final_token_at:
                timing.decode_seconds = max(0.0, timing.final_token_at - timing.first_token_at)

    def _log_inference_debug(
        self,
        call_kind: str,
        timing: InferenceTiming,
        cache_key: DialogueCacheKey | None,
    ) -> None:
        if not SETTINGS.prompt_debug:
            return
        decode_tok_s = (
            timing.generated_tokens / timing.decode_seconds
            if timing.generated_tokens > 0 and timing.decode_seconds > 0
            else 0.0
        )
        evaluated = (
            timing.backend_prompt_eval_tokens
            if timing.backend_prompt_eval_tokens is not None
            else "unavailable"
        )
        print(
            f"[Akane:debug:inference] generation_type={call_kind} "
            f"previous_prompt_tokens={timing.previous_prompt_tokens} "
            f"current_prompt_tokens={timing.current_prompt_tokens or timing.prompt_tokens} "
            f"actual_token_lcp={timing.actual_token_lcp} "
            f"actual_lcp_percent={timing.actual_lcp_percent:.3f} "
            f"tokens_after_lcp={timing.tokens_after_lcp} "
            f"backend_prompt_eval_tokens={evaluated} "
            f"backend_reused_prompt_tokens={timing.backend_reused_tokens if timing.backend_reused_tokens is not None else 'unavailable'} "
            f"backend_compute_graph_reuses={timing.backend_compute_graph_reuses if timing.backend_compute_graph_reuses is not None else 'unavailable'} "
            f"llama_n_tokens_before={timing.llama_n_tokens_before} "
            f"llama_n_tokens_after={timing.llama_n_tokens_after} "
            f"prompt_eval_ms={timing.prefill_seconds * 1000:.3f} "
            f"decode_tok_s={decode_tok_s:.3f} "
            f"generation_ms={max(0.0, timing.model_finished_at - timing.model_started_at) * 1000:.3f} "
            f"final_eos_or_stop_ms={max(0.0, timing.stop_observed_at - timing.final_token_at) * 1000 if timing.stop_observed_at and timing.final_token_at else 0.0:.3f} "
            f"stop_to_model_end_ms={max(0.0, timing.model_finished_at - timing.stop_observed_at) * 1000 if timing.stop_observed_at else 0.0:.3f} "
            f"finish_reason={timing.finish_reason or 'unavailable'} "
            f"cache_owner={timing.cache_owner} cache_epoch={timing.cache_epoch} "
            f"cache_invalidated={str(timing.cache_invalidated).lower()} "
            f"cache_invalidated_reason={timing.cache_invalidated_reason} "
            f"cache_invalidated_from_token={timing.cache_invalidated_from_token if timing.cache_invalidated_from_token is not None else 'unavailable'} "
            f"foreground_queue_wait_ms={timing.foreground_queue_wait_seconds * 1000:.3f} "
            f"model_lock_wait_ms={timing.model_lock_wait_seconds * 1000:.3f} "
            f"prompt_architecture={timing.prompt_architecture} "
            f"static_prefix_tokens={timing.static_prefix_tokens} "
            f"historical_prefix_tokens={timing.historical_prefix_tokens} "
            f"dynamic_state_tokens={timing.dynamic_state_tokens} "
            f"new_user_tokens={timing.new_user_tokens} "
            f"self_tokens={timing.self_tokens} memory_tokens={timing.memory_tokens} "
            f"experience_tokens={timing.experience_tokens} "
            f"time_tokens={timing.time_tokens} "
            f"code_context_tokens={timing.code_context_tokens} "
            f"wrapper_tokens={timing.wrapper_tokens} "
            f"chat_template_overhead_tokens={timing.chat_template_overhead_tokens} "
            f"state_revision_before={timing.state_revision_before if timing.state_revision_before is not None else 'unavailable'} "
            f"state_revision_after={timing.state_revision_after if timing.state_revision_after is not None else 'unavailable'} "
            f"state_payload_tokens={timing.state_payload_tokens} "
            f"state_payload_mode={timing.state_payload_mode} "
            f"context_trimmed={str(timing.context_trimmed).lower()} "
            f"tokens_removed={timing.tokens_removed} turns_removed={timing.turns_removed} "
            f"cache_rebuild_reason={timing.cache_rebuild_reason} "
            f"logical_prompt_tokens={timing.logical_prompt_tokens} "
            f"resident_sequence_tokens={timing.resident_sequence_tokens} "
            f"appended_tokens={timing.appended_tokens} "
            f"state_delta_tokens={timing.state_delta_tokens} "
            f"template_overhead_tokens={timing.template_overhead_tokens}",
            flush=True,
        )

    @staticmethod
    def _options(
        max_tokens: int,
        *,
        temperature: float | None = None,
    ) -> dict[str, object]:
        options: dict[str, object] = {
            "max_tokens": max(1, min(int(max_tokens), SETTINGS.llama_context_window - 1)),
            "temperature": SETTINGS.temperature if temperature is None else temperature,
            "top_k": SETTINGS.top_k,
            "top_p": SETTINGS.top_p,
            "min_p": SETTINGS.min_p,
            "repeat_penalty": SETTINGS.repetition_penalty,
        }
        if SETTINGS.generation_stop_sequences:
            options["stop"] = list(SETTINGS.generation_stop_sequences)
        return options

    @staticmethod
    def _performance_health() -> dict[str, object]:
        if not (SETTINGS.prompt_debug or SETTINGS.timing_enabled):
            return {}
        result: dict[str, object] = {}
        for name, path, divisor in (
            ("temperature_c", Path("/sys/class/thermal/thermal_zone0/temp"), 1000.0),
            ("cpu_frequency_mhz", Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"), 1000.0),
        ):
            try:
                result[name] = round(float(path.read_text().strip()) / divisor, 3)
            except (OSError, ValueError):
                result[name] = None
        try:
            result["load_average_1m"] = round(os.getloadavg()[0], 3)
        except (AttributeError, OSError):
            result["load_average_1m"] = None
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    result["process_rss_bytes"] = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            result["process_rss_bytes"] = None
        throttle_names = {
            0: "undervoltage_now",
            1: "frequency_capped_now",
            2: "throttled_now",
            3: "soft_temperature_limit_now",
            16: "undervoltage_occurred",
            17: "frequency_capped_occurred",
            18: "throttling_occurred",
            19: "soft_temperature_limit_occurred",
        }
        try:
            completed = subprocess.run(
                ("vcgencmd", "get_throttled"),
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            bits = int(completed.stdout.strip().split("=", 1)[-1], 16)
            result["throttled_bits"] = bits
            for bit, name in throttle_names.items():
                result[name] = bool(bits & (1 << bit))
        except (OSError, subprocess.SubprocessError, ValueError):
            result["throttled_bits"] = None
            for name in throttle_names.values():
                result[name] = None
        return result

    @staticmethod
    def _sequence_state_size(llm) -> int:
        try:
            from llama_cpp import llama_cpp

            return max(0, int(llama_cpp.llama_state_seq_get_size(llm._ctx.ctx, 0)))
        except (AttributeError, ImportError, TypeError, ValueError):
            return 0

    @staticmethod
    def _is_eog_token(llm, token: int) -> bool:
        try:
            from llama_cpp import llama_cpp

            return bool(llama_cpp.llama_token_is_eog(llm._model.vocab, int(token)))
        except (AttributeError, ImportError, TypeError, ValueError):
            try:
                return int(token) == int(llm.token_eos())
            except (AttributeError, TypeError, ValueError):
                return False

    @staticmethod
    def _decodable_prefix(data: bytes, limit: int) -> tuple[str, int]:
        end = max(0, min(len(data), int(limit)))
        while end:
            try:
                return data[:end].decode("utf-8"), end
            except UnicodeDecodeError as exc:
                if exc.reason == "unexpected end of data":
                    end = exc.start
                else:
                    return data[:end].decode("utf-8", errors="replace"), end
        return "", 0

    def _stream_token_completion(
        self,
        llm,
        input_tokens: tuple[int, ...],
        *,
        reset: bool,
        max_tokens: int,
        temperature: float | None,
        check_cancelled,
        completion_tokens: list[int],
        finish_reason: list[str],
        stop_observed_at: list[float],
        template_stop_sequences: tuple[str, ...] = (),
    ):
        """Stream a completion while retaining exact token IDs for the next append."""

        generator = llm.generate(
            input_tokens,
            top_k=SETTINGS.top_k,
            top_p=SETTINGS.top_p,
            min_p=SETTINGS.min_p,
            temp=SETTINGS.temperature if temperature is None else temperature,
            repeat_penalty=SETTINGS.repetition_penalty,
            reset=reset,
        )
        stop_sequences = tuple(dict.fromkeys(
            value.encode("utf-8")
            for value in (*template_stop_sequences, *SETTINGS.generation_stop_sequences)
            if value
        ))
        retained = max((len(value) for value in stop_sequences), default=1) - 1
        pending = b""
        stopped = False
        try:
            for token in generator:
                check_cancelled()
                if self._is_eog_token(llm, int(token)):
                    if pending:
                        text = pending.decode("utf-8", errors="replace")
                        pending = b""
                        if text:
                            yield text
                    finish_reason[:] = ["stop"]
                    stop_observed_at[:] = [time.perf_counter()]
                    stopped = True
                    break
                completion_tokens.append(int(token))
                piece = llm.detokenize([int(token)])
                pending += piece.encode("utf-8") if isinstance(piece, str) else bytes(piece)
                positions = [pending.find(value) for value in stop_sequences]
                positions = [position for position in positions if position >= 0]
                if positions:
                    text, consumed = self._decodable_prefix(pending, min(positions))
                    if text:
                        yield text
                    pending = pending[consumed:]
                    finish_reason[:] = ["stop"]
                    stop_observed_at[:] = [time.perf_counter()]
                    stopped = True
                    break
                safe_length = max(0, len(pending) - retained)
                text, consumed = self._decodable_prefix(pending, safe_length)
                if text:
                    pending = pending[consumed:]
                    yield text
                if len(completion_tokens) >= max_tokens:
                    finish_reason[:] = ["length"]
                    stop_observed_at[:] = [time.perf_counter()]
                    break
        finally:
            close = getattr(generator, "close", None)
            if close is not None:
                close()
        if pending and not stopped:
            text = pending.decode("utf-8", errors="replace")
            if text:
                yield text
        if not finish_reason:
            finish_reason.append("length")
            stop_observed_at[:] = [time.perf_counter()]

    def stream_messages(
        self,
        messages: tuple[dict[str, str], ...] | list[dict[str, str]],
        *,
        max_tokens: int,
        reservation: Reservation,
        cancellation: threading.Event | None = None,
        timing: InferenceTiming | None = None,
        temperature: float | None = None,
        call_kind: str = "other",
        cache_key: DialogueCacheKey | None = None,
        history_messages=(),
        canonical_user_content: str = "",
        turn_user_content: str = "",
        state_revision: int = 0,
        state_sections: tuple[tuple[str, str], ...] = (),
        state_items=(),
        static_prefix_tokens: int = 0,
        new_user_tokens: int = 0,
        context_trimmed: bool = False,
        tokens_removed: int = 0,
        turns_removed: int = 0,
        canonical_rebuild=None,
    ):
        if (
            timing is not None
            and not timing.prompt_tokens
            and not (call_kind == "dialogue" and cache_key is not None)
        ):
            timing.prompt_tokens, timing.prompt_token_method = self.count_prompt_tokens(messages, reservation)
        backend_messages = self._backend_messages(messages)
        prompt_tokens: tuple[int, ...] = ()
        prompt_method = "unavailable"
        architecture = "rebuilt"
        canonical_history = self._canonical_history(history_messages)

        def check_cancelled() -> None:
            if cancellation is not None and cancellation.is_set():
                raise InferenceCancelled("Inference was cancelled during generation.")
            if reservation.preemption.is_set():
                raise InferenceCancelled("Inference was cancelled.")

        response = None
        stream_chunks = 0
        completed = False
        output_parts: list[str] = []
        generated_token_ids: list[int] = []
        generated_finish_reason: list[str] = []
        generated_stop_observed_at: list[float] = []
        selection: _DialoguePromptSelection | None = None
        try:
            check_cancelled()
            invalidated_reason = "not-dialogue"
            if call_kind == "dialogue" and cache_key is not None:
                invalidated_reason = self._prepare_dialogue_cache(reservation.llm, cache_key)
                prior_state = self._conversation_state
                selection = self._select_dialogue_prompt(
                    reservation.llm,
                    messages,
                    cache_key,
                    max_tokens=max_tokens,
                    history_messages=history_messages,
                    turn_user_content=turn_user_content,
                    state_sections=state_sections,
                    state_items=state_items,
                    canonical_rebuild=canonical_rebuild,
                )
                backend_messages = selection.messages
                prompt_tokens = selection.tokens
                prompt_method = selection.token_method
                architecture = selection.architecture
                rebase_reason = selection.rebuild_reason
                canonical_history = selection.canonical_history
                if invalidated_reason == "none" and rebase_reason != "none":
                    invalidated_reason = rebase_reason
                if timing is not None:
                    if prompt_tokens:
                        timing.prompt_tokens = len(prompt_tokens)
                        timing.prompt_token_method = prompt_method
                    self._prepare_prompt_diagnostics(
                        reservation.llm,
                        timing,
                        cache_key,
                        prompt_tokens,
                        invalidated_reason,
                    )
                    if selection.direct_append and prior_state is not None:
                        timing.previous_prompt_tokens = len(prior_state.token_ids)
                        timing.actual_token_lcp = len(prior_state.token_ids)
                        timing.actual_lcp_percent = (
                            len(prior_state.token_ids) * 100.0 / len(prompt_tokens)
                            if prompt_tokens else 0.0
                        )
                        timing.tokens_after_lcp = max(
                            0, len(prompt_tokens) - len(prior_state.token_ids),
                        )
                    timing.prompt_architecture = architecture
                    timing.static_prefix_tokens = max(0, int(static_prefix_tokens))
                    timing.historical_prefix_tokens = max(
                        0, (timing.backend_reused_tokens or 0) - timing.static_prefix_tokens,
                    )
                    timing.new_user_tokens = max(0, int(new_user_tokens))
                    section_counts = {
                        name: self.count_text_tokens(text, reservation)[0]
                        for name, text in selection.payload_sections
                    }
                    for name in ("self", "memory", "experience", "time", "code_context"):
                        setattr(timing, f"{name}_tokens", max(0, int(section_counts.get(name, 0))))
                    timing.state_payload_tokens = sum(max(0, int(value)) for value in section_counts.values())
                    current_content_tokens = self.count_text_tokens(
                        selection.current_user_content, reservation,
                    )[0]
                    timing.wrapper_tokens = max(
                        0,
                        int(current_content_tokens)
                        - timing.state_payload_tokens
                        - timing.new_user_tokens,
                    )
                    timing.dynamic_state_tokens = (
                        timing.state_payload_tokens + timing.wrapper_tokens
                    )
                    timing.logical_prompt_tokens = len(prompt_tokens)
                    timing.resident_sequence_tokens = timing.llama_n_tokens_before
                    timing.appended_tokens = (
                        len(selection.append_tokens)
                        if selection.direct_append else
                        max(0, len(prompt_tokens) - (timing.backend_reused_tokens or 0))
                    )
                    timing.state_delta_tokens = timing.state_payload_tokens
                    timing.template_overhead_tokens = timing.wrapper_tokens
                    timing.state_payload_mode = selection.payload_mode
                    timing.state_revision_before = (
                        prior_state.state_revision if prior_state is not None else None
                    )
                    timing.state_revision_after = int(state_revision)
                    timing.context_trimmed = bool(
                        context_trimmed or selection.rebuild_reason == "context-window-rebase"
                    )
                    timing.tokens_removed = max(int(tokens_removed), selection.tokens_removed)
                    timing.turns_removed = max(int(turns_removed), selection.turns_removed)
                    timing.cache_rebuild_reason = selection.rebuild_reason
            elif call_kind != "dialogue":
                prompt_tokens, prompt_method = self._dialogue_prompt_tokens(
                    reservation.llm, backend_messages,
                )
                if timing is not None and prompt_tokens:
                    previous_tokens = self._previous_prompt_tokens_by_kind.get(call_kind, ())
                    actual_lcp = self._token_lcp(previous_tokens, prompt_tokens)
                    timing.prompt_tokens = len(prompt_tokens)
                    timing.prompt_token_method = prompt_method
                    timing.previous_prompt_tokens = len(previous_tokens)
                    timing.current_prompt_tokens = len(prompt_tokens)
                    timing.actual_token_lcp = actual_lcp
                    timing.actual_lcp_percent = actual_lcp * 100.0 / len(prompt_tokens)
                    timing.tokens_after_lcp = len(prompt_tokens) - actual_lcp
                    timing.logical_prompt_tokens = len(prompt_tokens)
            if timing is not None:
                queue_wait_seconds = max(
                    0.0, float(getattr(reservation, "queue_wait_seconds", 0.0)),
                )
                timing.model_lock_wait_seconds = queue_wait_seconds
                timing.foreground_queue_wait_seconds = queue_wait_seconds
                self._reset_backend_timings(reservation.llm)
                timing.health_before = self._performance_health()
                timing.model_started_at = time.perf_counter()
            check_cancelled()
            if call_kind == "dialogue" and selection is not None:
                self._record_foreground_architecture(architecture, invalidated_reason)
            self._record_model_call(call_kind)
            use_token_stream = bool(
                call_kind == "dialogue"
                and selection is not None
                and prompt_tokens
                and hasattr(reservation.llm, "generate")
                and hasattr(reservation.llm, "detokenize")
            )
            if use_token_stream:
                response = self._stream_token_completion(
                    reservation.llm,
                    selection.append_tokens if selection.direct_append else prompt_tokens,
                    reset=not selection.direct_append,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    check_cancelled=check_cancelled,
                    completion_tokens=generated_token_ids,
                    finish_reason=generated_finish_reason,
                    stop_observed_at=generated_stop_observed_at,
                    template_stop_sequences=(
                        self._chat_continuation_state.stop_sequences
                        if self._chat_continuation_state is not None else ()
                    ),
                )
            else:
                response = reservation.llm.create_chat_completion(
                    messages=backend_messages,
                    stream=True,
                    **self._options(max_tokens, temperature=temperature),
                )
            for chunk in response:
                check_cancelled()
                if use_token_stream:
                    text = str(chunk)
                else:
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if timing is not None and choice.get("finish_reason"):
                        timing.finish_reason = str(choice["finish_reason"])
                        timing.stop_observed_at = time.perf_counter()
                    text = _content((choice.get("delta") or {}).get("content") or choice.get("text"))
                if not text:
                    continue
                stream_chunks += 1
                first_visible_chunk = False
                if timing is not None:
                    token_at = time.perf_counter()
                    if not timing.first_token_at:
                        timing.first_token_at = token_at
                        first_visible_chunk = True
                    timing.final_token_at = token_at
                output_parts.append(text)
                yield text
                if timing is not None and first_visible_chunk:
                    timing.health_after_prefill = self._performance_health()
            completed = True
        finally:
            if timing is not None and generated_finish_reason:
                timing.finish_reason = generated_finish_reason[-1]
            if timing is not None and generated_stop_observed_at:
                timing.stop_observed_at = generated_stop_observed_at[-1]
            if timing is not None and timing.model_started_at:
                self._capture_backend_timings(reservation.llm, timing, stream_chunks)
                if timing.backend_prompt_eval_tokens is not None:
                    timing.chat_template_overhead_tokens = max(
                        0,
                        timing.backend_prompt_eval_tokens
                        - timing.state_payload_tokens
                        - timing.wrapper_tokens
                        - timing.new_user_tokens,
                    )
                timing.llama_n_tokens_after = max(
                    0, int(getattr(reservation.llm, "n_tokens", 0)),
                )
                timing.model_finished_at = time.perf_counter()
                timing.health_after_decode = self._performance_health()
                if SETTINGS.prompt_debug or SETTINGS.timing_enabled:
                    timing.sequence_state_bytes = self._sequence_state_size(reservation.llm)
                self._log_inference_debug(call_kind, timing, cache_key)
            close = getattr(response, "close", None)
            if close is not None:
                close()
            if completed and call_kind == "dialogue" and cache_key is not None:
                self._live_cache_owner = cache_key
                if prompt_tokens:
                    self._previous_dialogue_prompt = (cache_key, prompt_tokens)
                reply = "".join(output_parts)
                if reply.strip():
                    if selection is not None and selection.direct_append:
                        sequence_tokens = selection.tokens + tuple(generated_token_ids)
                    else:
                        sequence_tokens = prompt_tokens + tuple(generated_token_ids)
                    self._conversation_state = _ConversationInferenceState(
                        owner=cache_key,
                        token_ids=sequence_tokens,
                        canonical_history=canonical_history + (
                            ("user", canonical_user_content),
                            ("assistant", reply),
                        ),
                        state_revision=int(state_revision),
                        state_section_fingerprints=self._state_section_fingerprints(state_sections),
                        synchronized_state_items=(
                            selection.synchronized_state_items if selection is not None else ()
                        ),
                        cache_epoch=self._cache_epoch,
                    )
            elif completed and call_kind != "dialogue" and prompt_tokens:
                self._previous_prompt_tokens_by_kind[call_kind] = prompt_tokens
