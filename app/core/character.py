"""Validated loading of Akane's two canonical character files."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.core.config import ADVISOR_ONLY

SOUL_PATH = Path(__file__).resolve().parent.parent / "soul.md"
IDENTITY_PATH = Path(__file__).resolve().parent.parent / "identity.md"
_HARD_RULES = (
    "Use relevant recent context naturally, including repetition, corrections, "
    "contradictions, and unfinished threads; do not treat each message as isolated.",
    "Never describe yourself as processing input, generating replies, waiting, "
    "observing, or managing the conversation.",
    "Do not use emojis.",
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
                "[AKANE IDENTITY — CANONICAL FACTS]\n" + self.identity,
                "[AKANE SOUL / VOICE]\n" + self.soul,
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
def _static_system_prompt_cached(soul: str, identity: str) -> str:
    profile = CharacterProfile(soul=soul, identity=identity)
    return "\n\n".join((_hard_rules_text(), profile.stable_identity_text()))


def get_static_system_prompt() -> str:
    """Return the validated high-priority prompt prefix."""

    profile = load_character_profile()
    return _static_system_prompt_cached(profile.soul, profile.identity)
