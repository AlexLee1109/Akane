"""Akane's small, stable starting foundation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
IDENTITY_PATH = (_APP_DIR / "identity.md").resolve()
SOUL_PATH = (_APP_DIR / "soul.md").resolve()


@dataclass(frozen=True, slots=True)
class Character:
    name: str
    identity: str
    voice: str
    seed_interests: tuple[str, ...]
    appearance: str
    identity_path: Path
    soul_path: Path
    identity_mtime_ns: int
    soul_mtime_ns: int
    identity_sha256: str
    soul_sha256: str
    content_sha256: str


def _read(path: Path) -> tuple[str, int, str]:
    try:
        raw = path.read_bytes()
        value = raw.decode("utf-8").strip()
        mtime_ns = path.stat().st_mtime_ns
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Required character file is unavailable: {path.name}") from exc
    if not value:
        raise RuntimeError(f"Required character file is empty: {path.name}")
    return value, mtime_ns, hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=2)
def _load_character_profile(identity_mtime_ns: int, soul_mtime_ns: int) -> Character:
    identity, identity_mtime_ns, identity_sha256 = _read(IDENTITY_PATH)
    soul, soul_mtime_ns, soul_sha256 = _read(SOUL_PATH)
    return Character(
        name="Akane",
        identity=identity,
        voice=soul,
        seed_interests=("anime", "manga", "VTubers", "games"),
        appearance=(
            "Long blue hair fading toward gray, clear blue eyes, and a white-and-blue "
            "outfit with a dark skirt and necktie. This visible illustrated form is hers."
        ),
        identity_path=IDENTITY_PATH,
        soul_path=SOUL_PATH,
        identity_mtime_ns=identity_mtime_ns,
        soul_mtime_ns=soul_mtime_ns,
        identity_sha256=identity_sha256,
        soul_sha256=soul_sha256,
        content_sha256=hashlib.sha256(
            f"{identity}\0{soul}".encode("utf-8")
        ).hexdigest(),
    )


def load_character_profile() -> Character:
    try:
        identity_mtime_ns = IDENTITY_PATH.stat().st_mtime_ns
        soul_mtime_ns = SOUL_PATH.stat().st_mtime_ns
    except OSError as exc:
        raise RuntimeError("Required character files are unavailable.") from exc
    return _load_character_profile(identity_mtime_ns, soul_mtime_ns)
