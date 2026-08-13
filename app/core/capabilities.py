"""Authoritative, non-persistent facts about Akane's current runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.core.config import DISCORD_BOT_TOKEN
from app.core.utils import OWNER_PROFILE_ID


@dataclass(frozen=True, slots=True)
class CapabilityRuntime:
    """Live inputs that can change without changing persistent state."""

    profile_id: str = OWNER_PROFILE_ID
    source: str = "popup"
    editor_connected: bool = False


@dataclass(frozen=True, slots=True)
class CapabilityFact:
    key: str
    available: bool
    description: str
    source_id: str
    reason: str = "direct_query"


@dataclass(frozen=True, slots=True)
class _CapabilityDefinition:
    key: str
    aliases: tuple[str, ...]
    supported: str
    unsupported: str
    source_id: str
    availability: str = "always"


_CAPABILITY_QUERY = re.compile(
    r"\b(?:can (?:you|your)|could you|are you able|do you have|"
    r"what can you (?:not )?do|"
    r"what are your capabilit|is .{0,40} (?:available|supported|connected)|"
    r"do you (?:remember|recall|see|hear|browse|speak))\b",
    re.IGNORECASE,
)
_TECHNICAL_QUERY = re.compile(
    r"\b(?:implementation|technical|internals?|runtime|configuration|configured|"
    r"architecture|source code)\b",
    re.IGNORECASE,
)
_WORDS = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_GENERIC_TERMS = {
    "able", "about", "access", "are", "available", "been", "can",
    "capability", "capabilities", "could", "current", "did", "do", "have",
    "is", "me", "my", "over", "supported", "the", "thing", "things", "time",
    "to", "we", "what", "you", "your",
}


def _terms(value: object) -> set[str]:
    return {
        term[:-1]
        if len(term) > 3 and term.endswith("s") and not term.endswith("ss")
        else term
        for term in _WORDS.findall(str(value or "").casefold())
        if len(term) > 1
    }


def _similar(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    lexical = len(left_terms & right_terms) / max(1, len(left_terms | right_terms))
    return max(
        lexical,
        SequenceMatcher(None, " ".join(sorted(left_terms)), " ".join(sorted(right_terms))).ratio(),
    )


_DEFINITIONS = (
    _CapabilityDefinition(
        "text_conversation",
        ("text conversation", "chat", "talk with you", "communicate"),
        "I can communicate through text and continue the recent conversation.",
        "Text conversation is unavailable.",
        "runtime:text-conversation:v1",
    ),
    _CapabilityDefinition(
        "persistent_memory",
        ("remember", "memory", "recall", "previous session", "yesterday"),
        "I can retain validated long-term memories across conversations and restarts.",
        "I do not have persistent memory in this conversation.",
        "runtime:persistent-memory:v1",
    ),
    _CapabilityDefinition(
        "persistent_opinions",
        ("opinion", "opinions change", "change your mind", "views over time"),
        "I can maintain persistent current opinions and revise them when my reasons change.",
        "I cannot retain changing opinions over time.",
        "runtime:persistent-opinions:v1",
    ),
    _CapabilityDefinition(
        "emotional_continuity",
        ("emotion", "emotions", "feelings", "mood", "feel emotions"),
        "I can maintain a grounded emotional state that changes with conversation and time.",
        "I do not maintain emotional continuity.",
        "runtime:emotional-continuity:v1",
    ),
    _CapabilityDefinition(
        "persistent_interests",
        ("interests", "curiosity", "become interested", "lose interest"),
        "I can maintain and revise my own interests over time.",
        "I cannot retain changing interests over time.",
        "runtime:persistent-interests:v1",
    ),
    _CapabilityDefinition(
        "relationship_continuity",
        ("relationship", "between us", "shared history", "know me over time"),
        "I can retain grounded shared relationship continuity.",
        "I do not retain relationship continuity.",
        "runtime:relationship-continuity:v1",
    ),
    _CapabilityDefinition(
        "self_model",
        ("self model", "understand yourself", "learn about yourself", "strength weakness"),
        "I can retain an evidence-based understanding of my behavior and limitations.",
        "A persistent self-model is not available in this conversation.",
        "runtime:self-model:v1",
        "owner",
    ),
    _CapabilityDefinition(
        "behavioral_strategies",
        ("improve yourself", "improvement strategy", "behavior strategy", "work on your weakness"),
        "I can retain and evaluate narrow behavioral improvement strategies.",
        "Persistent behavioral strategies are not available in this conversation.",
        "runtime:behavioral-strategies:v1",
        "owner",
    ),
    _CapabilityDefinition(
        "ambient_presence",
        ("presence", "what are you doing", "what have you been doing", "offscreen"),
        "I can maintain grounded, private background Presence between conversations.",
        "Persistent background Presence is not available in this conversation.",
        "runtime:ambient-presence:v1",
        "owner",
    ),
    _CapabilityDefinition(
        "editor_context",
        ("current file", "editor", "code context", "codebase", "repository", "vscode", "vs code"),
        "I can read the bounded code context supplied by the connected editor for this turn.",
        "I do not currently have code context from a connected editor.",
        "runtime:editor-context:v1",
        "editor",
    ),
    _CapabilityDefinition(
        "discord_text",
        ("discord", "discord bot", "discord message"),
        "A Discord text interface is available.",
        "A Discord text interface is not currently available.",
        "runtime:discord-text:v1",
        "discord",
    ),
    _CapabilityDefinition(
        "screen_vision",
        ("see me", "look at me", "see my screen", "view my screen", "screen vision", "screen capture", "camera", "visually inspect display"),
        "I can see the user's screen.",
        "I cannot see the user's screen.",
        "runtime:no-screen-interface:v1",
        "never",
    ),
    _CapabilityDefinition(
        "physical_sensation",
        ("feel touch", "touch", "smell", "taste", "physical sensation", "physical world"),
        "I can sense physical touch and surroundings.",
        "I cannot sense physical touch, smells, tastes, or surroundings.",
        "runtime:physical-sensation:v1",
        "never",
    ),
    _CapabilityDefinition(
        "live2d_control",
        ("live2d", "live 2d", "avatar animation", "move your avatar"),
        "I can control a Live2D avatar.",
        "I do not currently control a Live2D avatar or avatar animation.",
        "runtime:live2d-control:v1",
        "never",
    ),
    _CapabilityDefinition(
        "audio_input",
        ("hear me", "microphone", "audio input", "listen to me", "voice input"),
        "I can hear live audio input.",
        "I cannot hear live audio or use a microphone.",
        "runtime:audio-input:v1",
        "never",
    ),
    _CapabilityDefinition(
        "speech_output",
        ("speak", "talk out loud", "voice output", "text to speech", "tts"),
        "I can produce speech audio.",
        "I do not currently produce speech audio as part of my runtime.",
        "runtime:speech-output:v1",
        "never",
    ),
    _CapabilityDefinition(
        "web_access",
        ("browse the web", "internet", "web access", "search online", "look things up online"),
        "I can browse the web for current information.",
        "I cannot browse or search the web.",
        "runtime:web-access:v1",
        "never",
    ),
    _CapabilityDefinition(
        "general_file_access",
        ("file access", "filesystem", "arbitrary files", "open a file", "read files"),
        "I can access arbitrary files.",
        "I do not have general file-system access through conversation.",
        "runtime:general-file-access:v1",
        "never",
    ),
    _CapabilityDefinition(
        "runtime_control",
        ("control computer", "run commands", "external tools", "take control", "control apps"),
        "I can control the computer and external tools.",
        "I cannot control the computer or invoke external tools autonomously.",
        "runtime:runtime-control:v1",
        "never",
    ),
)
_ALIAS_TERMS = tuple(
    set().union(*(_terms(alias) for alias in definition.aliases))
    for definition in _DEFINITIONS
)


class CapabilityRegistry:
    """Derive capability truth from implementation and live interface state."""

    @property
    def count(self) -> int:
        return len(_DEFINITIONS)

    def _available(
        self,
        definition: _CapabilityDefinition,
        runtime: CapabilityRuntime,
    ) -> bool:
        if definition.availability == "never":
            return False
        if definition.availability == "owner":
            return runtime.profile_id == OWNER_PROFILE_ID
        if definition.availability == "editor":
            return runtime.editor_connected
        if definition.availability == "discord":
            return runtime.source == "discord" or bool(DISCORD_BOT_TOKEN)
        return True

    def _description(
        self,
        definition: _CapabilityDefinition,
        runtime: CapabilityRuntime,
        available: bool,
    ) -> str:
        if definition.key == "discord_text" and available:
            return (
                "I am currently communicating through Discord."
                if runtime.source == "discord"
                else "A Discord text interface is configured."
            )
        return definition.supported if available else definition.unsupported

    def snapshot(self, runtime: CapabilityRuntime) -> tuple[CapabilityFact, ...]:
        return tuple(
            CapabilityFact(
                definition.key,
                available,
                self._description(definition, runtime, available),
                definition.source_id,
                "current_authoritative_state",
            )
            for definition in _DEFINITIONS
            for available in (self._available(definition, runtime),)
        )

    def relevant(
        self,
        query: str,
        runtime: CapabilityRuntime,
        *,
        limit: int = 3,
    ) -> tuple[CapabilityFact, ...]:
        """Return only capability facts directly requested by this message."""

        text = str(query or "").strip()
        if not text or not _CAPABILITY_QUERY.search(text):
            return ()
        query_terms = _terms(text) - _GENERIC_TERMS
        scored: list[tuple[float, int, _CapabilityDefinition]] = []
        for index, (definition, alias_terms) in enumerate(
            zip(_DEFINITIONS, _ALIAS_TERMS)
        ):
            matched = query_terms & alias_terms
            score = len(matched) / max(1, min(len(query_terms), len(alias_terms)))
            if matched:
                scored.append((score, -index, definition))

        if not scored and re.search(
            r"\bwhat can you (?:not )?do\b|\bcapabilities\b",
            text,
            re.IGNORECASE,
        ):
            scored = [
                (1.0 - index * 0.01, -index, definition)
                for index, definition in enumerate(_DEFINITIONS[:4])
            ]

        technical = bool(_TECHNICAL_QUERY.search(text))
        facts: list[CapabilityFact] = []
        for _score, _index, definition in sorted(scored, reverse=True)[:limit]:
            available = self._available(definition, runtime)
            facts.append(
                CapabilityFact(
                    definition.key,
                    available,
                    self._description(definition, runtime, available),
                    definition.source_id,
                    "technical_query" if technical else "direct_query",
                )
            )
        return tuple(facts)

    def match_persistent_claim(
        self,
        category: str,
        area: str,
        description: str,
    ) -> CapabilityFact | None:
        """Ground self-model mirrors in static runtime truth, never model belief."""

        want_available = category == "capability"
        if category not in {"capability", "limitation"}:
            return None
        runtime = CapabilityRuntime()
        matches: list[tuple[float, _CapabilityDefinition]] = []
        for definition in _DEFINITIONS:
            if definition.availability in {"editor", "discord"}:
                continue
            available = self._available(definition, runtime)
            if available != want_available or _similar(area, definition.key) < 0.34:
                continue
            expected = definition.supported if available else definition.unsupported
            score = _similar(description, expected)
            if score >= 0.55:
                matches.append((score, definition))
        if not matches:
            return None
        definition = max(matches, key=lambda item: item[0])[1]
        available = self._available(definition, runtime)
        return CapabilityFact(
            definition.key,
            available,
            definition.supported if available else definition.unsupported,
            definition.source_id,
            "current_authoritative_state",
        )


CAPABILITY_REGISTRY = CapabilityRegistry()
