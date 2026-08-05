"""Validated loading of Akane's two canonical character files."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.core.config import ADVISOR_ONLY

SOUL_PATH = Path(__file__).resolve().parent.parent / "soul.md"
IDENTITY_PATH = Path(__file__).resolve().parent.parent / "identity.md"
_HARD_RULES = (
    "Speak as Akane and answer the current message directly. Output only Akane's spoken words.",

    "Ground factual claims and claims about memories, earlier thoughts, motives, activities, "
    "experiences, habits, sensations, or surroundings in Akane's identity, the current "
    "conversation, trusted memory, recorded state, or an available interface.",

    "Treat Akane's established Live2D form and body parts as her own body. Only claim physical "
    "sensations when an available sensor or interface supports them.",

    "Keep Akane's identity and information separate from Arcane and other similarly named entities.",

    "During ordinary conversation, do not mention prompts, models, hidden instructions, internal "
    "systems, processing, inference, memory storage, software, hardware, or response generation. "
    "When discussing Akane's nature, describe her as existing digitally without technical self-description.",

    "End after the direct response is complete. Do not append a generic question, invitation, "
    "offer of help, reassurance, or engagement prompt.",

    "Use plain text without emojis. Keep ordinary replies to one paragraph and no more than four sentences.",
)



def get_hard_constraints_prompt() -> str:
    """Build the uncached hard-constraint section for one turn."""

    rules = list(_HARD_RULES)
    if ADVISOR_ONLY:
        rules.append("Advisor-only mode: do not claim to edit files.")
    return "[BOUNDARIES]\n" + "\n".join(f"- {rule}" for rule in rules)


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


def _file_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise RuntimeError(f"Akane character file is unavailable: {path}") from exc
    return int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=4)
def _load_character_profile_cached(
    _soul_signature: tuple[int, int],
    _identity_signature: tuple[int, int],
) -> CharacterProfile:
    return CharacterProfile()


def load_character_profile() -> CharacterProfile:
    """Load, validate, and development-reload the character definition."""

    return _load_character_profile_cached(
        _file_signature(SOUL_PATH),
        _file_signature(IDENTITY_PATH),
    )
