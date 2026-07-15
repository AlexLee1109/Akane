"""Tiny shared helpers for Akane's hot path."""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[a-z0-9_+#./-]+", re.IGNORECASE)


def clamp(
    value: object,
    default: float = 0.0,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def compact_text(value: object, limit: int = 180) -> str:
    text = " ".join(
        str(value or "").replace("\r", " ").replace("\n", " ").split()
    ).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") or text[:limit]


def words(value: object) -> set[str]:
    return {item.lower().strip(".") for item in WORD_RE.findall(str(value or "").lower())}
