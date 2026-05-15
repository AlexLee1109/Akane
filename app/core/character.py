"""Character profile for Akane."""

from pathlib import Path

from app.core.config import ADVISOR_ONLY

CHARACTER = {
    "name": "Akane",
    "creator": "Arcane",
    "identity": "an AI VTuber who stays close to the user",
}

SOUL_PATH = Path(__file__).resolve().parent.parent / "soul.md"
_SOUL_CACHE: tuple[int, str] | None = None


def load_soul() -> str:
    """Load the soul.md file."""
    global _SOUL_CACHE
    mtime_ns = soul_cache_key()
    if _SOUL_CACHE is not None and _SOUL_CACHE[0] == mtime_ns:
        return _SOUL_CACHE[1]
    soul = SOUL_PATH.read_text(encoding="utf-8").strip()
    _SOUL_CACHE = (mtime_ns, soul)
    return soul


def soul_cache_key() -> int:
    try:
        return SOUL_PATH.stat().st_mtime_ns
    except OSError:
        return -1


def build_system_prompt(
    memory_context: str = "",
    include_memory: bool = True,
) -> str:
    """Build the full system prompt for Akane."""
    soul = load_soul()

    sections = [soul]

    # Core behavior sections
    sections.extend([
        "CONVERSATION COHERENCE:",
        "Use recent conversation memory to maintain continuity.",
        "If essential details are missing, ask one focused follow-up question.",
        "Do not ask the user to repeat information already provided.",
        "",
    ])

    if include_memory:
        sections.extend([
            "MEMORY:",
            "Only reference things that actually happened in this conversation or saved memory. Don't fake it.",
            "",
            "TAGS — always put at the very end if needed:",
            "[MEM]fact: <what you learned>[/MEM]",
            "[MEM]preference: <what they prefer>[/MEM]",
            "[FORGET]what to remove[/FORGET]",
            "",
        ])

    # Tool tags
    if ADVISOR_ONLY:
        sections.extend([
            "TOOL TAGS:",
            "Advisor-only mode. Use [READ]path/to/file[/READ] when needed.",
            "",
        ])
    else:
        sections.extend([
            "TOOL TAGS:",
            "Use [READ]path/to/file[/READ] when needed.",
            "Use [ASK_CODER]...[/ASK_CODER] for code changes.",
            "",
        ])

    if include_memory and memory_context:
        sections.append(memory_context)
        sections.append("")

    return "/no_think\n\n".join(sections)
