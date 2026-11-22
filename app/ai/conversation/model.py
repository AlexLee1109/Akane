"""The local dialogue model boundary; no backend is imported or loaded here."""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .cancellation import CancellationToken

INITIAL_MODEL_FILENAME = "Qwen3.5-9B-Q4_K_M.gguf"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    # "reference" is host-supplied past context for the Gemma chat formatter.
    role: Literal["system", "user", "assistant", "tool", "reference"]
    content: str


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    messages: tuple[ChatMessage, ...]
    text: str
    tokens: tuple[int, ...]
    max_tokens: int
    stops: tuple[str, ...] = ()
    json_schema: str | None = None


class ContextBudgetExceeded(ValueError):
    """Identity, current input, and output reservation cannot fit together."""


class DialogueModel(Protocol):
    """One local model, borrowed from its runtime owner.

    The adapter receives a complete prompt for each call and does not own
    conversation history. Calls are serialized by the model's scheduler. Check the
    token before inference and between generated pieces, raising
    GenerationCancelled when cancellation is observed. It must release active
    generation resources when the iterator closes or raises.
    """

    @property
    def model_path(self) -> Path:
        """The explicitly supplied, existing local GGUF file."""
        ...

    @property
    def context_size(self) -> int:
        ...

    def prepare(
        self, messages: tuple[ChatMessage, ...], *, max_tokens: int
    ) -> PreparedPrompt:
        """Render the embedded template and tokenize exactly, without sampling."""
        ...

    def stream(
        self, prompt: PreparedPrompt, *, cancellation: CancellationToken
    ) -> Generator[str, None, None]:
        """Yield visible reply text without retaining the supplied context."""
        ...

    def close(self) -> None:
        """Release the model after active generation has stopped."""
        ...

