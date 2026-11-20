"""Packaged L1 identity and conversation guidance; no model or state access."""

from importlib.resources import files
from pathlib import Path


def _load(name: str, path: Path | None) -> str:
    source = path if path is not None else files("app.ai").joinpath(name)
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"The {name} file is empty.")
    return text


def load_character(path: Path | None = None) -> str:
    return _load("character.md", path)


def load_conversation(path: Path | None = None) -> str:
    return _load("conversation.md", path)


def load_l1_prompt() -> str:
    return load_character().rstrip() + "\n\n" + load_conversation().lstrip()
