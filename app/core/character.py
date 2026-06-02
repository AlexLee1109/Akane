"""Akane system prompt helpers."""

from __future__ import annotations

from pathlib import Path

from app.core.config import ADVISOR_ONLY

SOUL_PATH = Path(__file__).resolve().parent.parent / "soul.md"
IDENTITY_PATH = Path(__file__).resolve().parent.parent / "identity.md"

_PromptFileCache = tuple[int, str]
_SOUL_CACHE: _PromptFileCache | None = None
_IDENTITY_CACHE: _PromptFileCache | None = None
_BASE_PROMPT_CACHE: dict[tuple[tuple[int, int, int], bool], str] = {}
_FULL_PROMPT_CACHE: dict[tuple[tuple[int, int, int], bool, str], str] = {}


def _load_cached(path: Path, cache: _PromptFileCache | None) -> _PromptFileCache:
    try:
        mtime = path.stat().st_mtime_ns
        if cache and cache[0] == mtime:
            return cache
        return mtime, path.read_text(encoding="utf-8").strip()
    except OSError:
        return -1, ""


def load_soul() -> str:
    global _SOUL_CACHE
    _SOUL_CACHE = _load_cached(SOUL_PATH, _SOUL_CACHE)
    return _SOUL_CACHE[1]


def load_identity() -> str:
    global _IDENTITY_CACHE
    _IDENTITY_CACHE = _load_cached(IDENTITY_PATH, _IDENTITY_CACHE)
    return _IDENTITY_CACHE[1]


def prompt_revision() -> tuple[int, int, int]:
    load_soul()
    load_identity()
    return (
        _SOUL_CACHE[0] if _SOUL_CACHE else -1,
        _IDENTITY_CACHE[0] if _IDENTITY_CACHE else -1,
        int(ADVISOR_ONLY),
    )


def _base_system_prompt(*, include_memory: bool) -> str:
    key = (prompt_revision(), include_memory)
    cached = _BASE_PROMPT_CACHE.get(key)
    if cached is not None:
        return cached

    parts = [load_soul(), load_identity()]
    if include_memory:
        parts.append(
            "MEMORY:\n"
            "- Use stored memory only when relevant.\n"
            "- Never invent memories or claim something was stored if it was not.\n"
            "- When the user gives a durable fact or preference, append one hidden tag:\n"
            "[MEM]fact: <stable learned info>[/MEM]\n"
            "[MEM]preference: <stable user preference>[/MEM]\n"
            "[MEM]user: name: <name>[/MEM]\n"
            "[PROJECT]<project name>: <short durable detail>[/PROJECT]\n"
            "[FORGET]<thing to remove>[/FORGET]"
        )
    if ADVISOR_ONLY:
        parts.append("Advisor-only mode: do not claim to edit files.")
    result = "\n\n".join(part for part in parts if part)
    if len(_BASE_PROMPT_CACHE) > 8:
        _BASE_PROMPT_CACHE.clear()
    _BASE_PROMPT_CACHE[key] = result
    return result


def build_system_prompt(runtime_context: str = "", *, include_memory: bool = True) -> str:
    context = str(runtime_context or "").strip()
    key = (prompt_revision(), include_memory, context)
    cached = _FULL_PROMPT_CACHE.get(key)
    if cached is not None:
        return cached
    parts = [_base_system_prompt(include_memory=include_memory)]
    if context:
        parts.append(context)
    result = "\n\n".join(part for part in parts if part)
    if len(_FULL_PROMPT_CACHE) > 16:
        _FULL_PROMPT_CACHE.clear()
    _FULL_PROMPT_CACHE[key] = result
    return result
