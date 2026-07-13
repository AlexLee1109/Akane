"""Validated, process-wide loading of Akane's two character files."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.core.config import ADVISOR_ONLY

SOUL_PATH = Path(__file__).resolve().parent.parent / "soul.md"
IDENTITY_PATH = Path(__file__).resolve().parent.parent / "identity.md"
_HARD_RULES = (
    "Answer the actual request accurately and do not invent missing context.",
    "Remain Akane in every reply; technical detail changes the depth or structure, not her identity or voice.",
    "Use specific, concrete language and let personality appear naturally without narrating or forcing it.",
    "Keep casual replies short. Use more detail when a technical explanation needs it.",
    "No emojis.",
    "Do not end a casual reply with a question unless an answer is genuinely required.",
    "Use the user's name sparingly and only when natural.",
)


def _hard_rules_text() -> str:
    rules = list(_HARD_RULES)
    if ADVISOR_ONLY:
        rules.append("Advisor-only mode: do not claim to edit files.")
    return "[AKANE HARD RULES]\n" + "\n".join(f"- {rule}" for rule in rules)


def _clean_prompt_file(text: str) -> str:
    return "\n".join(
        line.rstrip()
        for line in str(text or "").splitlines()
        if line.strip() and line.strip() != "---"
    ).strip()


def _read_required(path: Path, label: str) -> str:
    try:
        text = _clean_prompt_file(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Akane {label} file is unavailable: {path}") from exc
    if not text:
        raise RuntimeError(f"Akane {label} file is empty: {path}")
    return text


@dataclass(frozen=True, slots=True)
class CharacterProfile:
    """Akane's stable identity, sourced only from soul.md and identity.md."""

    soul_path: Path = field(default_factory=lambda: SOUL_PATH)
    identity_path: Path = field(default_factory=lambda: IDENTITY_PATH)
    soul: str = ""
    identity: str = ""

    def __post_init__(self) -> None:
        if not self.soul:
            object.__setattr__(self, "soul", _read_required(self.soul_path, "soul"))
        if not self.identity:
            object.__setattr__(self, "identity", _read_required(self.identity_path, "identity"))

    def stable_identity_text(self) -> str:
        return "\n\n".join(
            (
                "[AKANE IDENTITY]\n" + self.identity,
                "[AKANE SOUL / VOICE]\n" + self.soul,
            )
        )


@lru_cache(maxsize=1)
def load_character_profile() -> CharacterProfile:
    """Load and validate the process-wide character definition once."""

    return CharacterProfile()


@lru_cache(maxsize=1)
def get_static_system_prompt() -> str:
    """Return the immutable high-priority prompt prefix."""

    profile = load_character_profile()
    return "\n\n".join((_hard_rules_text(), profile.stable_identity_text()))
