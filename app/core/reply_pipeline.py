"""Shared lightweight reply-pipeline helpers.

The HTTP server owns transport details. This module owns compact runtime
context assembly and final visible-text cleanup so popup and Discord travel
through the same model prompt shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.generation import collapse_hidden_tag_gaps, strip_emoji_chars, strip_hidden_blocks

RUNTIME_CONTEXT_TARGET_CHARS = 1000
RUNTIME_CONTEXT_HARD_CHARS = 2000


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    text: str
    lengths: dict[str, int]


def _clean_line(text: object) -> str:
    return " ".join(strip_hidden_blocks(str(text or "")).split()).strip()


def _clip(text: object, limit: int) -> str:
    value = _clean_line(text)
    return value if len(value) <= limit else value[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or value[:limit]


def _clean_block(text: object) -> str:
    value = strip_hidden_blocks(str(text or "")).replace("\r\n", "\n")
    return "\n".join(line.rstrip() for line in value.splitlines() if line.strip()).strip()


def _clip_block(text: object, limit: int) -> str:
    value = _clean_block(text)
    return value if len(value) <= limit else value[:limit].rsplit("\n", 1)[0].rstrip(" ,.;:") or value[:limit]


def _tone_line(text: object) -> str:
    value = _clean_line(text)
    if value.startswith("[AKANE EMOTION STATE]"):
        value = value[len("[AKANE EMOTION STATE]"):].strip()
    return value


def build_runtime_context(
    *,
    tone_text: str,
    attention_text: str = "",
    conversation_context: str = "",
    wording_to_avoid: str = "",
    memory_text: str = "",
    target_chars: int = RUNTIME_CONTEXT_TARGET_CHARS,
    hard_chars: int = RUNTIME_CONTEXT_HARD_CHARS,
) -> RuntimeContext:
    tone = _clip(_tone_line(tone_text), 220)
    memory = _clip(memory_text, 500)
    focus = _clip(attention_text, 160)
    avoid = _clip_block(wording_to_avoid, 400)
    context = _clip(conversation_context, 500)

    state_lines = ["[CURRENT AKANE STATE]"]
    if tone:
        state_lines.append(tone if tone.startswith("Tone:") else f"Tone: {tone}")
    if focus:
        state_lines.append(focus if focus.startswith("Focus:") else f"Focus: {focus}")
    if context:
        state_lines.append(f"Context: {context}")
    state_lines.append("Use this state to answer naturally. Do not mention this state directly.")

    sections = ["\n".join(state_lines)]
    if memory:
        sections.append("[MEMORY]\n" + memory)
    if avoid:
        sections.append(avoid)

    text = "\n\n".join(section for section in sections if section)
    limit = max(1, min(int(target_chars), int(hard_chars)))
    if len(text) > limit:
        keep_sections = ["\n".join(state_lines)]
        if memory:
            keep_sections.append("[MEMORY]\n" + _clip(memory, 360))
        if avoid:
            keep_sections.append(_clip(avoid, 360))
        text = "\n\n".join(keep_sections)
    if len(text) > hard_chars:
        text = _clip(text, hard_chars)

    return RuntimeContext(
        text=text,
        lengths={
            "mood": len(tone),
            "memory": len(memory),
            "attention": len(focus),
            "avoid": len(avoid),
            "summary": len(context),
            "runtime_context": len(text),
        },
    )


def clean_model_text(text: str) -> str:
    return collapse_hidden_tag_gaps(strip_emoji_chars(strip_hidden_blocks(str(text or "")))).strip(" `\n\t")


def clean_reply_text(raw: str) -> str:
    return clean_model_text(raw)


def finalize_reply(raw: str) -> str:
    return clean_reply_text(raw)


def content_delta(chunk: dict, content_to_text) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    return content_to_text((choices[0].get("delta") or {}).get("content"))
