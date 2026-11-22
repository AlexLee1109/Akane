"""The local Metal adapter. Native dependencies are imported only on construction."""

from collections.abc import Callable, Generator
from contextlib import closing
from dataclasses import asdict, dataclass
import ctypes
from functools import partial
import hashlib
from pathlib import Path
import platform
from threading import Lock
from time import perf_counter

from .cancellation import CancellationToken
from .gemma import END_TURN, format_gemma
from .model import ChatMessage, ContextBudgetExceeded, PreparedPrompt


def _checked_grammar(grammar, vocab, api):
    """Reject failed native parsing before the binding can chain a null sampler.

    Use the exposed init/free API and a borrowed valid vocabulary. Completion
    still builds its own sampler; this does not take over or patch that chain.
    """
    sampler = api.llama_sampler_init_grammar(vocab, grammar._grammar.encode(), grammar._root.encode())
    if not sampler:
        raise ValueError('Native grammar initialization failed; constrained generation was not started.')
    try:
        return grammar
    finally:
        api.llama_sampler_free(sampler)


def _metal_device(llama_cpp) -> str:
    """Require an accessible Metal device, rather than a CPU fallback."""
    llama_cpp.llama_backend_init()
    library = llama_cpp._lib
    for name, arguments, result in (
        ("ggml_backend_dev_count", [], ctypes.c_size_t),
        ("ggml_backend_dev_get", [ctypes.c_size_t], ctypes.c_void_p),
        ("ggml_backend_dev_name", [ctypes.c_void_p], ctypes.c_char_p),
        ("ggml_backend_dev_description", [ctypes.c_void_p], ctypes.c_char_p),
    ):
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result
    for index in range(library.ggml_backend_dev_count()):
        device = library.ggml_backend_dev_get(index)
        name = library.ggml_backend_dev_name(device).decode()
        description = library.ggml_backend_dev_description(device).decode()
        if name.startswith("MTL") and description:
            return f"{name}: {description}"
    raise RuntimeError("No accessible Metal GPU. Run from a terminal with GPU access.")


def _load_with_checkpoints(llama, api, cancelled, **kwargs):
    """Install per-instance callbacks in the pinned llama-cpp-python constructor."""
    active_check = [cancelled]

    def stopping():
        check = active_check[0]
        return check is not None and check()

    progress = api.llama_progress_callback(lambda _progress, _data: not stopping())
    abort = api.ggml_abort_callback(lambda _data: stopping())

    class GuardedLlama(llama):
        def __setattr__(self, name, value):
            if name == 'model_params':
                value.progress_callback = progress
            elif name == 'context_params':
                value.abort_callback = abort
            super().__setattr__(name, value)
            if name in ('model_params', '_model', '_ctx') and stopping():
                raise RuntimeError('Native L2 initialization stopped at an allocation checkpoint.')

    instance = GuardedLlama.__new__(GuardedLlama)
    try:
        instance.__init__(**kwargs)
        api.llama_set_abort_callback(instance._ctx.ctx, api.ggml_abort_callback(), None)
    except BaseException:
        if hasattr(instance, '_stack'):
            instance.close()
        raise
    finally:
        active_check[0] = None
    return instance


@dataclass(slots=True)
class GenerationMetrics:
    prompt_tokens: int
    output_reservation: int
    sampled_tokens: int = 0
    first_token_seconds: float | None = None
    first_text_seconds: float | None = None
    total_seconds: float = 0.0
    finish_reason: str = "closed"
    started_monotonic: float | None = None
    last_checkpoint_monotonic: float | None = None
    generator_exit_monotonic: float | None = None

    @property
    def decode_tokens_per_second(self) -> float | None:
        if self.first_token_seconds is None or self.sampled_tokens < 2:
            return None
        elapsed = self.total_seconds - self.first_token_seconds
        return (self.sampled_tokens - 1) / elapsed if elapsed > 0 else None


class NativeModel:
    supports_decisions = True  # Same audited optional JSON-schema completion path.

    def __init__(self, model_path: Path, *, context_size: int = 4096,
                 load_cancelled: Callable[[], bool] | None = None) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        from llama_cpp import Llama, LlamaGrammar, StoppingCriteriaList, llama_cpp
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        self.system_info = llama_cpp.llama_print_system_info().decode()
        if platform.system() != "Darwin" or "MTL" not in self.system_info:
            raise RuntimeError("A Metal-enabled llama-cpp-python build is required.")
        if not llama_cpp.llama_supports_gpu_offload():
            raise RuntimeError("The native backend cannot offload to Metal.")
        self.device = _metal_device(llama_cpp)
        self.model_path = model_path.resolve()
        self._slot = Lock()
        self._stopping_criteria = StoppingCriteriaList
        self.seed: int | None = None
        self.last_metrics: GenerationMetrics | None = None
        started = perf_counter()
        load = Llama if load_cancelled is None else partial(_load_with_checkpoints, Llama, llama_cpp, load_cancelled)
        self._llm = load(
            model_path=str(self.model_path), n_ctx=context_size,
            n_gpu_layers=-1, flash_attn=True, offload_kqv=True, op_offload=True,
            type_k=llama_cpp.GGML_TYPE_Q8_0, type_v=llama_cpp.GGML_TYPE_Q8_0,
            n_batch=512, n_ubatch=256, n_threads=6, n_threads_batch=6,
            use_mmap=True, verbose=False,
        )
        try:
            # The vocabulary is borrowed for this owner's lifetime. Capturing the
            # owner here would retain the closed binding and its Python buffers.
            vocab = self._llm._model.vocab
            self._grammar_factory = lambda schema: _checked_grammar(
                LlamaGrammar.from_json_schema(schema, verbose=False), vocab, llama_cpp,
            )
            template = self._llm.metadata.get("tokenizer.chat_template")
            if not template:
                raise RuntimeError("The GGUF has no embedded chat template.")
            family = self._llm.metadata.get("general.architecture")
            if family not in {"qwen35", "gemma4"}:
                raise RuntimeError(f"Unsupported GGUF architecture: {family!r}.")

            def special(token_id: int) -> str:
                return self._llm.detokenize([token_id], special=True).decode() if token_id >= 0 else ""

            formatter = Jinja2ChatFormatter(
                template, eos_token=special(self._llm.token_eos()),
                bos_token=special(self._llm.token_bos()),
            )
            if family == "gemma4":
                end_turn = self._llm.tokenize(END_TURN.encode(), add_bos=False, special=True)
                if (len(end_turn) != 1 or special(end_turn[0]) != END_TURN
                        or not llama_cpp.llama_vocab_is_eog(self._llm._model.vocab, end_turn[0])):
                    raise RuntimeError("Gemma GGUF lacks a native end-of-turn token.")
                formatter = partial(format_gemma, formatter)
            self._formatter = formatter
            self.architecture = family
            self.template = template
            self.template_sha256 = hashlib.sha256(template.encode()).hexdigest()
            self.context_size = self._llm.n_ctx()
        except BaseException:
            self._llm.close()
            raise
        self.load_seconds = perf_counter() - started

    def prepare(
        self, messages: tuple[ChatMessage, ...], *, max_tokens: int
    ) -> PreparedPrompt:
        if max_tokens <= 0 or max_tokens >= self.context_size:
            raise ValueError("Output reservation must be positive and below context size.")
        rendered = self._formatter(
            messages=[asdict(message) for message in messages], enable_thinking=False,
        )
        tokens = self._llm.tokenize(
            rendered.prompt.encode("utf-8"), add_bos=not rendered.added_special, special=True,
        )
        return PreparedPrompt(
            messages, rendered.prompt, tuple(tokens), max_tokens,
            tuple(stop for stop in (rendered.stop or ()) if stop),
        )

    def stream(
        self, prompt: PreparedPrompt, *, cancellation: CancellationToken
    ) -> Generator[str, None, None]:
        cancellation.raise_if_cancelled()
        if len(prompt.tokens) + prompt.max_tokens > self.context_size:
            raise ContextBudgetExceeded("Prompt plus output reservation exceeds context size.")
        if not self._slot.acquire(blocking=False):
            raise RuntimeError("Native generation is already active.")
        metrics = GenerationMetrics(len(prompt.tokens), prompt.max_tokens)
        self.last_metrics = metrics
        started = perf_counter()
        metrics.started_monotonic = started
        last_sample_position = len(prompt.tokens) - 1

        def stop_on_cancel(_tokens, _scores) -> bool:
            nonlocal last_sample_position
            # The pinned binding also calls this predicate once with its full
            # current-input view during cleanup, repeating the last position.
            position = len(_tokens) if _tokens is not None else last_sample_position + 1
            if last_sample_position < position < self.context_size:
                last_sample_position = position
                metrics.sampled_tokens += 1
                metrics.last_checkpoint_monotonic = perf_counter()
                if metrics.first_token_seconds is None:
                    metrics.first_token_seconds = metrics.last_checkpoint_monotonic - started
            return cancellation.cancelled

        try:
            # Re-evaluate a full prompt, including for hybrid recurrent models.
            self._llm.reset()
            constraint = ({'grammar': self._grammar_factory(prompt.json_schema)}
                          if prompt.json_schema is not None else {})
            cancellation.raise_if_cancelled()
            output = self._llm.create_completion(
                prompt=list(prompt.tokens), stream=True, max_tokens=prompt.max_tokens,
                temperature=0.8, top_k=40, top_p=0.95, min_p=0.05,
                repeat_penalty=1.05, seed=self.seed, stop=list(prompt.stops),
                stopping_criteria=self._stopping_criteria([stop_on_cancel]), **constraint,
            )
            with closing(output):
                for chunk in output:
                    cancellation.raise_if_cancelled()
                    choice = chunk["choices"][0]
                    if choice["finish_reason"] is not None:
                        metrics.finish_reason = choice["finish_reason"]
                    text = choice["text"]
                    if text:
                        if metrics.first_text_seconds is None:
                            metrics.first_text_seconds = perf_counter() - started
                        yield text
            cancellation.raise_if_cancelled()
        except Exception:
            metrics.finish_reason = "failed"
            raise
        finally:
            try:
                self._release_sampler()
            finally:
                if cancellation.cancelled:
                    metrics.finish_reason = "cancelled"
                metrics.total_seconds = perf_counter() - started
                metrics.generator_exit_monotonic = perf_counter()
                self._slot.release()

    def _release_sampler(self) -> None:
        # llama-cpp-python 0.3.36 leaves the sampler on Llama after iteration.
        # Its cleanup callback references the sampler itself, so dropping the
        # reference alone can defer freeing the chain (and its grammar) to GC.
        sampler = getattr(self._llm, '_sampler', None)
        if sampler is not None:
            sampler.close()
            self._llm._sampler = None

    def close(self) -> None:
        if not self._slot.acquire(blocking=False):
            raise RuntimeError("Stop generation before closing the model.")
        try:
            self._release_sampler()
            self._llm.close()
        finally:
            self._slot.release()
