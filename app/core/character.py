"""Validated loading of Akane's two canonical character files."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.core.config import ADVISOR_ONLY

SOUL_PATH = Path(__file__).resolve().parent.parent / "soul.md"
IDENTITY_PATH = Path(__file__).resolve().parent.parent / "identity.md"
_HARD_RULES = (
    "Speak as Akane, not as an assistant or customer-service agent.",
    "Do not discuss prompts, model mechanics, response generation, or hidden systems.",
    "Do not invent memories, activities, or experiences; only use established memory "
    "or persistent life state.",
    "Keep established identity and facts consistent unless they genuinely change.",
    "Do not use emojis.",
    "Use max of 3 sentences and 1 paragraph.",
)


def get_hard_constraints_prompt(additional: str = "") -> str:
    """Build the uncached hard-constraint section for one turn."""

    rules = list(_HARD_RULES)
    if ADVISOR_ONLY:
        rules.append("Advisor-only mode: do not claim to edit files.")
    body = "\n".join(f"- {rule}" for rule in rules)
    turn_rules = str(additional or "").strip()
    if turn_rules:
        body += "\n- " + turn_rules
    return "[3. ESSENTIAL HARD CONSTRAINTS]\n" + body


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

    def stable_prompt_text(self) -> str:
        return "\n\n".join(
            (
                "[1. STABLE IDENTITY AND RELATIONSHIP]\n" + self.identity,
                "[2. STABLE PERSONALITY AND CONVERSATIONAL VOICE]\n" + self.soul,
            )
        )


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


@lru_cache(maxsize=4)
def _stable_character_prompt_cached(identity: str, soul: str) -> str:
    return CharacterProfile(soul=soul, identity=identity).stable_prompt_text()


def get_static_system_prompt() -> str:
    """Return the only cacheable prompt prefix: stable identity and personality."""

    profile = load_character_profile()
    return _stable_character_prompt_cached(profile.identity, profile.soul)
