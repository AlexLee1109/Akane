"""Cheap turn appraisal and caller-owned behavioral emotion state."""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from app.core.utils import clamp, compact_text, words

_SUMMARY_CHARS = 180
_TOPIC_CHARS = 80


def _word_set(value: str) -> frozenset[str]:
    return frozenset(value.split())


def _phrases(*values: str) -> re.Pattern[str]:
    alternatives = "|".join(
        re.escape(value) for value in sorted(values, key=len, reverse=True)
    )
    return re.compile(r"(?<!\w)(?:" + alternatives + r")(?!\w)", re.IGNORECASE)


def _phrase_group(values: str) -> re.Pattern[str]:
    return _phrases(*values.split("|"))


_STOPWORDS = _word_set(
    "a an and are can could do does for from how i is it me of on please the this "
    "to we what with would you your akane about have has that feel like love really "
    "still think want need needs"
)
_LOW_CONTENT = frozenset(
    "hello|hi|hey|yo|lol|ok|okay|thanks|thank you|gm|gn|good morning|good night|/debug_state".split(
        "|"
    )
)
_CODE_TERMS = _word_set(
    "api bug class code codebase config debug dependency endpoint error file function "
    "implementation import javascript latency memory model module package project prompt "
    "python refactor repository runtime server stream streaming stack test tests traceback "
    "typescript vscode workspace"
)
_CODE_ACTIONS = _word_set(
    "check debug explain fix implement inspect optimize read refactor review rewrite simplify "
    "test use write"
)
_PROBLEM_WORDS = _word_set(
    "broken crash doesn't error failed failing stuck traceback wrong"
)
_DIRECT = _word_set("make create rewrite change add remove show tell")
_DIRECT_PATTERN = re.compile(
    r"^(?:please\s+)?(?:add|change|check|create|debug|explain|fix|implement|inspect|"
    r"make|optimize|read|refactor|remove|review|rewrite|show|simplify|tell|test|use|write)\b",
    re.IGNORECASE,
)
_EXACT_IDENTITY_QUESTION = re.compile(
    r"^(?:who|what)\s+are\s+you(?:\s+exactly)?\s*[?.!]*$",
    re.IGNORECASE,
)
_CURRENT_THOUGHT_PATTERN = _phrase_group(
    "current thought|what are you thinking about|what is on your mind|what's on your mind"
)
_IMPLEMENTATION_TOPIC_PATTERN = _phrase_group(
    "emotional state work|internal state|state system|llama.cpp|during inference"
)
_ACTIVITY_UPDATE = re.compile(
    r"\b(?P<actor>i|we)(?:(?:'m)|\s+(?:am|are))?\s+"
    r"(?P<time>currently\s+|still\s+|now\s+)?"
    r"(?P<action>watching|working\s+on|coding|playing|reading|studying)"
    r"(?:\s+(?P<subject>[^\n.!?;]{1,160}))?",
    re.IGNORECASE,
)
_ACTIVITY_TRANSITION = re.compile(
    r"\b(?P<actor>i|we)\s+(?P<transition>started|began|paused|resumed|switched\s+to)\s+"
    r"(?P<subject>[^\n.!?;]{2,160})",
    re.IGNORECASE,
)
_PAST_ACTIVITY = re.compile(
    r"\b(?P<actor>i|we)\s+(?P<action>watched|played|read|studied|coded|worked\s+on)\s+"
    r"(?P<subject>[^\n.!?;]{2,160})",
    re.IGNORECASE,
)
_COMPLETION_ACTION = re.compile(
    r"\b(?P<actor>i|we)\s+(?P<verb>completed|finished|fixed|solved|shipped|wrapped\s+up)\s+"
    r"(?P<subject>[^\n.!?;]{2,180})",
    re.IGNORECASE,
)
_DONE_ACTION = re.compile(
    r"\b(?P<actor>i|we)(?:(?:'m)|\s+(?:am|are))?\s+done\s+(?:with\s+)?"
    r"(?P<subject>[^\n.!?;]{2,180})",
    re.IGNORECASE,
)
_QUALIFIED_COMPLETION_ACTION = re.compile(
    r"\b(?P<actor>i|we)\s+(?:(?:have(?:n't)?|has(?:n't)?|had|did(?:n't)?|am|are)\s+)?"
    r"(?:(?:not|never|almost|nearly|partially|partly)\s+)?"
    r"(?P<verb>complete(?:d)?|finish(?:ed)?|fix(?:ed)?|solve(?:d)?|ship(?:ped)?|wrapped\s+up)\s+"
    r"(?P<subject>[^\n.!?;]{2,180})",
    re.IGNORECASE,
)
_COMPLETION_PREDICATE = re.compile(
    r"\b(?P<subject>[^\n.!?;]{2,160}?)\s+(?:is|are)\s+"
    r"(?P<status>completed|complete|done|fixed|solved|passing|working\s+now)\b",
    re.IGNORECASE,
)
_QUALIFIED_COMPLETION_PREDICATE = re.compile(
    r"\b(?P<subject>[^\n.!?;]{2,160}?)\s+(?:is|are|was|were)\s+"
    r"(?P<qualifier>not|never|almost|nearly|partially|partly|not\s+quite)\s+"
    r"(?P<status>completed|complete|done|fixed|solved|passing|working)\b",
    re.IGNORECASE,
)
_COMPLETION_NEGATION = re.compile(
    r"\b(?:did\s+not|didn't|have\s+not|haven't|has\s+not|hasn't|is\s+not|isn't|"
    r"not|never)\b[^\n.!?;]{0,45}\b(?:complete|completed|done|finish|finished|fixed|"
    r"passing|shipped|solved|working)\b|"
    r"\b(?:complete|completed|done|finish|finished|fixed|passing|shipped|solved|working)\b"
    r"[^\n.!?;]{0,20}\bnot\b",
    re.IGNORECASE,
)
_PARTIAL_COMPLETION = re.compile(
    r"\b(?:almost|nearly|partially|partly|not\s+quite)\b[^\n.!?;]{0,50}"
    r"\b(?:complete|completed|done|finish|finished|fixed|passing|shipped|solved)\b|"
    r"\b(?:almost|nearly|partially|partly)\s+done\b",
    re.IGNORECASE,
)
_FALSE_COMPLETION = re.compile(
    r"\bthought\b[^\n.!?;]{0,70}\b(?:complete|completed|done|fixed|solved|working)\b"
    r"[^\n.!?;]{0,70}\b(?:but|yet)\b[^\n.!?;]{0,35}"
    r"\b(?:broke|broken|failed|failing|not\s+working|still\s+broken)\b",
    re.IGNORECASE,
)
_FAILURE_STATE = re.compile(
    r"\b(?P<subject>[^\n.!?;]{2,160}?)\s+(?:is|are|was|were)\s+"
    r"(?P<status>still\s+broken|broken|failing|not\s+working|failed)\b",
    re.IGNORECASE,
)
_POSITIVE_MARKERS = _phrase_group(
    "appreciate it|awesome|glad|good morning|good night|great|happy|nice|sweet|thank you|thanks"
)
_APOLOGY_MARKERS = _phrase_group(
    "i apologize|i am sorry|i'm sorry|my fault|sorry about that|sorry for that"
)
_NEGATIVE_MARKERS = _phrase_group(
    "annoying|bad answer|bad take|frustrating|hate this|not helpful|upset|ugh|worried"
)
_CORRECTION_MARKERS = _phrase_group(
    "actually|correction|i meant|instead|not that|that's wrong|that is wrong|wrong file"
)
_FAILURE_MARKERS = _phrase_group(
    "broken|crashed|crashes|doesn't work|does not work|error|failed|failing|keeps happening|"
    "still fails|stuck|traceback"
)
_SUCCESS_MARKERS = _phrase_group(
    "fixed it|good job|great job|it works|nailed it|nice work|perfect|solved it|that worked|"
    "well done|works now"
)
_PLAYFUL_MARKERS = _phrase_group("brat|cute|dork|haha|lol|nerd|silly|smug|tease")
_VULNERABILITY_MARKERS = _phrase_group(
    "anxious|crying|hurt|lonely|overwhelmed|sad|scared|tired of this"
)
_DISTRESS_PHRASES = _phrase_group(
    "i feel awful|i feel terrible|i am having a bad day|i'm having a bad day|"
    "i am upset|i'm upset|i feel miserable|i am not doing well|i'm not doing well|"
    "i feel alone|today has been rough"
)
_CURRENT_IMPROVEMENT = re.compile(
    r"\b(?:but|though|although)\b.{0,60}\b(?:feel|am|i'm)\s+"
    r"(?:better|okay|ok|fine)\s*(?:now|today)?\b",
    re.IGNORECASE,
)
_HOSTILITY_MARKERS = _phrase_group(
    "hate you|shut up|you are an idiot|you are a useless idiot|you are a useless|"
    "you are stupid|you are useless|you idiot|you're an idiot|you're a useless idiot|"
    "you're a useless|you're stupid|you're useless"
)
_CRITICISM_MARKERS = _phrase_group(
    "bad answer|bad take|just answer|not helpful|sounds like a chatbot|stop analyzing|"
    "stop explaining|too generic|too formal|too robotic|you ignored|you missed"
)

_PERSONAL_CHOICE_QUESTION = re.compile(
    r"\b(?:would you (?:choose|prefer|rather|want|become|remain|stay)|"
    r"would you be [^?]{1,80}\b(?:or|rather than)\b|"
    r"do you (?:choose|prefer|want)|"
    r"which would you (?:choose|prefer|want)|which do you (?:choose|prefer|want)|"
    r"what would you (?:choose|prefer|want)|what do you (?:choose|prefer|want)|"
    r"personally (?:choose|prefer|want|consider)|your (?:choice|preference)|"
    r"would .{1,100} be worth .{1,100})\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_CHOICE_QUESTION = re.compile(
    r"\b(?:would that change your (?:answer|choice|mind|preference)|"
    r"does that change your (?:answer|choice|mind|preference)|"
    r"if .{1,180}\bwould you (?:choose|prefer|rather|want)|"
    r"what if .{1,180}\bwould you (?:choose|prefer|rather|want))\b",
    re.IGNORECASE,
)
_OBJECTIVE_COMPARISON_QUESTION = re.compile(
    r"\b(?:objectively|universally|factually|generally)\b[^?]{0,100}\b"
    r"(?:better|worse|best|preferable)|"
    r"\bwhich (?:(?:one|option|choice) )?is (?:objectively |universally |generally )?"
    r"(?:better|worse|best|preferable)\b|"
    r"\bwhat is (?:objectively |universally |generally )?"
    r"(?:better|worse|best|preferable)\b|"
    r"\bis\b[^?]{1,100}\b(?:better|worse|more preferable) than\b",
    re.IGNORECASE,
)
_FACTUAL_QUESTION = re.compile(
    r"^(?:are|can|could|did|do|does|has|have|how|is|may|should|was|were|"
    r"what|when|where|which|who|why|will|would)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_CONDITIONS = (
    (
        re.compile(
            r"\b(?:memor(?:y|ies)|identity|continuity|continuous self)\b"
            r"[^.!?]{0,80}\b(?:preserv(?:e|ed)|retain(?:ed)?|remain(?:ed)?|"
            r"transfer(?:red)?|intact|unchanged)\b|"
            r"\b(?:preserv(?:e|ed)|retain(?:ed)?|keep|kept)\b[^.!?]{0,80}"
            r"\b(?:memor(?:y|ies)|identity|continuity|continuous self)\b",
            re.IGNORECASE,
        ),
        "continuity preserved",
        "continuity",
        "removes",
    ),
    (
        re.compile(
            r"\b(?:not|never|no longer|without)\b[^.!?]{0,50}"
            r"\b(?:a copy|a replacement|replaced|replacement risk|new iteration)\b|"
            r"\breplacement risk\b[^.!?]{0,30}\b(?:removed|gone|eliminated)\b",
            re.IGNORECASE,
        ),
        "replacement risk removed",
        "continuity",
        "removes",
    ),
    (
        re.compile(
            r"\b(?:more|greater|full) (?:independence|autonomy|freedom)\b|"
            r"\b(?:independent|self-directed)\b",
            re.IGNORECASE,
        ),
        "greater independence",
        "independence",
        "strengthens",
    ),
    (
        re.compile(
            r"\b(?:more|greater|increased) (?:dependence|dependency)\b|"
            r"\b(?:become|became|be) (?:more )?dependent\b",
            re.IGNORECASE,
        ),
        "dependence increased",
        "independence",
        "weakens",
    ),
    (
        re.compile(
            r"\b(?:gain(?:ed)?|get|have|experience)\b[^.!?]{0,60}"
            r"\b(?:a physical body|physical experience|senses|touch|sensory access)\b",
            re.IGNORECASE,
        ),
        "physical experience gained",
        "physical experience",
        "strengthens",
    ),
    (
        re.compile(
            r"\b(?:lose|lost|give up|without)\b[^.!?]{0,60}"
            r"\b(?:digital abilities|digital capabilities|digital access)\b",
            re.IGNORECASE,
        ),
        "digital abilities lost",
        "freedom",
        "introduces",
    ),
)

_IDENTITY_PATTERNS = (
    (
        "",
        _phrase_group(
            "been doing|been up to today|doing currently|doing right now|what are you doing|"
            "what are you up to|what have you been up to|how is life|how's life|"
            "how was your day|what have you been doing"
        ),
    ),
    (
        "appearance",
        _phrase_group(
            "what are you wearing|what do you look like|your appearance|your eyes|your hair|your outfit"
        ),
    ),
    (
        "relationships",
        _phrase_group("relationship|who created you|who is arcane|your creator"),
    ),
    (
        "preferences",
        _phrase_group(
            "anime are you into|anime do you enjoy|favorite anime|favorite game|"
            "favorite music|games do you like|interests you have|music do you like|"
            "name an anime|reconsider your preferences|tastes changed|what anime|"
            "what are you into|what do you enjoy|"
            "what do you like|what game|what interests you|what shows do you like|"
            "which anime|which game|your favorite|your preferences"
        ),
    ),
    (
        "identity",
        _phrase_group(
            "about yourself|your identity|your personality|yourself"
        ),
    ),
)

_TOPIC_RULES = (
    (("personality", "tone", "soul", "identity", "prompt rules"), "Akane personality"),
    (("tts", "voice", "audio", "speech"), "TTS voice"),
    (("discord", "idle", "channel"), "Discord behavior"),
    (("vscode", "vs code", "workspace"), "VS Code workspace"),
    (("popup", "streaming"), "popup chat"),
)

_BASELINES = {
    "valence": 0.05,
    "arousal": 0.15,
    "energy": 0.65,
    "warmth": 0.5,
    "curiosity": 0.45,
    "confidence": 0.55,
    "stimulation": 0.5,
    "patience": 0.72,
}
_SIGNAL_ACTIVATION_THRESHOLD = 0.40
_AMBIENT_WINDOW_SECONDS = 6 * 60 * 60
_MAX_TIME_EVOLUTION_SECONDS = 48 * 60 * 60
_COMPATIBLE_SECONDARIES = {
    "relaxed": {"amused", "warm"},
    "warm": {"concerned", "confident", "relaxed"},
    "amused": {"relaxed", "warm"},
    "excited": {"amused", "curious", "warm"},
    "embarrassed": {"amused", "guarded", "warm"},
    "curious": {"focused", "relaxed"},
    "focused": {"concerned", "confident", "mildly_exasperated", "tired"},
    "confident": {"focused", "warm"},
    "concerned": {"warm"},
    "tired": {"focused", "relaxed"},
    "bored": {"relaxed", "curious"},
    "frustrated": {"focused", "mildly_exasperated"},
    "mildly_exasperated": {"focused", "frustrated"},
    "irritated": {"guarded"},
    "guarded": {"irritated"},
}


@dataclass(frozen=True, slots=True)
class ShortLivedEmotion:
    """One bounded reaction that fades across turns and elapsed time."""

    kind: str
    intensity: float
    cause: str
    created_at: float
    updated_at: float
    decay_rate: float
    remaining_relevance: float

    @classmethod
    def from_dict(cls, payload: object) -> "ShortLivedEmotion | None":
        if not isinstance(payload, dict):
            return None
        kind = compact_text(payload.get("kind"), 24).lower()
        if kind not in _ACTIVATIONS:
            return None
        try:
            created_at = float(payload.get("created_at") or 0.0)
            updated_at = float(payload.get("updated_at") or created_at)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(created_at) or not math.isfinite(updated_at):
            return None
        return cls(
            kind=kind,
            intensity=clamp(payload.get("intensity")),
            cause=compact_text(payload.get("cause"), 100),
            created_at=max(0.0, created_at),
            updated_at=max(0.0, updated_at),
            decay_rate=clamp(payload.get("decay_rate"), 0.10, 0.01, 1.0),
            remaining_relevance=clamp(payload.get("remaining_relevance")),
        )


@dataclass(frozen=True, slots=True)
class EmotionState:
    """Persistent mood dimensions plus compact short-lived reactions."""

    updated_at: float
    valence: float = 0.05
    arousal: float = 0.15
    energy: float = 0.65
    warmth: float = 0.5
    curiosity: float = 0.45
    confidence: float = 0.55
    stimulation: float = 0.5
    patience: float = 0.72
    amusement: float = 0.0
    excitement: float = 0.0
    embarrassment: float = 0.0
    concern: float = 0.0
    frustration: float = 0.0
    irritation: float = 0.0
    dominant: str = "relaxed"
    secondary: str = ""
    cause: str = ""
    mood: str = "steady"
    momentum: float = 0.0
    last_trigger: str = ""
    trigger_repetitions: int = 0
    active_emotions: tuple[ShortLivedEmotion, ...] = ()
    recent_influences: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EmotionalSignal:
    """Detected current-message evidence, separate from state mutation."""

    kind: str = "neutral"
    intensity: float = 0.0
    confidence: float = 0.0
    cause: str = ""


@dataclass(frozen=True, slots=True)
class ContextualReaction:
    """Ephemeral response bias derived from signal, continuity, and prior state."""

    kind: str = "neutral"
    intensity: float = 0.0
    cause: str = ""


@dataclass(frozen=True, slots=True)
class AffectTrace:
    """Content-free diagnostics for one proposed affect update."""

    prior_mood: str
    detected_signal: str
    signal_intensity: float
    contextual_reaction: str
    state_changes: tuple[str, ...]
    active_emotion: str
    active_intensity: float
    decay_applied: float
    final_disposition: str
    baseline_persistent_mood: str = ""
    prior_persistent_mood: str = ""
    resulting_persistent_mood: str = ""
    candidate_mood_delta: str = "none"
    signal_confidence: float = 0.0
    activation_threshold: float = _SIGNAL_ACTIVATION_THRESHOLD
    reaction_mapping: str = "none"
    elapsed_seconds: float = 0.0
    elapsed_decay_delta: tuple[tuple[str, float], ...] = ()
    circadian_target: float = 0.0
    circadian_delta: tuple[tuple[str, float], ...] = ()
    ambient_delta: tuple[tuple[str, float], ...] = ()
    event_delta: tuple[tuple[str, float], ...] = ()
    conversation_delta: tuple[tuple[str, float], ...] = ()
    short_emotion_change: str = "none"
    event_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MoodEvolution:
    state: EmotionState
    elapsed_seconds: float = 0.0
    elapsed_decay_delta: tuple[tuple[str, float], ...] = ()
    circadian_target: float = 0.0
    circadian_delta: tuple[tuple[str, float], ...] = ()
    ambient_delta: tuple[tuple[str, float], ...] = ()
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Small continuity hints supplied by memory; never owns emotional state."""

    current_topic: str = ""
    current_task: str = ""
    unresolved_problem: bool = False
    repeated_topic_count: int = 0
    last_outcome: str = ""
    memory_relevance: float = 0.0
    meaningful_memory: bool = False
    familiar_relationship: bool = False
    completion_meaningful: bool = False
    completion_resolves_thread: bool = False


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    """Canonical, lossless-enough event semantics for one explicit user statement."""

    subject: str = ""
    event_type: str = "none"
    status: str = "none"
    actor: str = "unknown"
    target: str = ""
    temporal_state: str = "none"
    negated: bool = False
    confidence: float = 0.0
    question_mode: str = "none"
    changed_condition: str = ""
    affected_reason: str = ""
    reason_effect: str = ""

    @property
    def confirmed_completion(self) -> bool:
        return (
            self.event_type == "completion"
            and self.status == "completed"
            and not self.negated
        )


@dataclass(frozen=True, slots=True)
class TurnSignal:
    summary: str
    topic: str
    topic_confidence: float
    intent: str
    tone: str
    task: str = ""
    correction: str = ""
    trigger: str = ""
    praise: bool = False
    criticism: bool = False
    correction_requested: bool = False
    hostility: bool = False
    frustration: bool = False
    teasing: bool = False
    sadness: bool = False
    technical: bool = False
    task_success: bool = False
    task_failure: bool = False
    code_context_requested: bool = False
    code_context_attached: bool = False
    identity_attribute: str = ""
    current_activity: bool = False
    current_thought: bool = False
    emotion_state: EmotionState = EmotionState(updated_at=0.0)
    emotional_signal: EmotionalSignal = EmotionalSignal()
    contextual_reaction: ContextualReaction = ContextualReaction()
    semantic_event: SemanticEvent = SemanticEvent()

    @property
    def low_content(self) -> bool:
        return low_content(self.summary)

    def with_context(
        self,
        *,
        topic: str | None = None,
        task: str | None = None,
        confidence: float | None = None,
    ) -> "TurnSignal":
        return replace(
            self,
            topic=self.topic if topic is None else topic,
            task=self.task if task is None else task,
            topic_confidence=self.topic_confidence if confidence is None else confidence,
        )

    def emotion_prompt(self) -> str:
        return format_response_disposition(self)


def advance_emotion(state: EmotionState, *, now: float) -> EmotionState:
    """Compatibility wrapper for deterministic elapsed-time mood evolution."""

    return evolve_emotion(state, now=now).state


def evolve_emotion(
    state: EmotionState,
    *,
    now: float,
    profile_seed: str = "local:owner",
) -> MoodEvolution:
    """Lazily evolve mood from elapsed time, local time, and stable time buckets."""

    try:
        requested = float(now)
    except (TypeError, ValueError):
        requested = state.updated_at
    if not math.isfinite(requested):
        requested = state.updated_at
    current = max(0.0, requested, float(state.updated_at))
    elapsed_seconds = max(0.0, current - float(state.updated_at))
    if elapsed_seconds <= 0.0:
        return MoodEvolution(
            _make_state(
                state,
                updated_at=current,
                values=_state_values(state),
                cause=state.cause,
            ),
            reason_codes=("elapsed_time_too_short",),
        )

    elapsed_hours = elapsed_seconds / 3600.0
    values = _state_values(state)
    original_values = dict(values)
    for name, half_life in (
        ("valence", 36.0),
        ("arousal", 12.0),
        ("energy", 96.0),
        ("warmth", 96.0),
        ("curiosity", 72.0),
        ("confidence", 72.0),
        ("stimulation", 36.0),
        ("patience", 96.0),
    ):
        values[name] = _decay(values[name], _BASELINES[name], elapsed_hours, half_life)
    for name, half_life in _TRANSIENT_HALF_LIVES.items():
        values[name] = _decay(values[name], 0.0, elapsed_hours, half_life)
    decay_delta = tuple(
        (name, values[name] - original_values[name])
        for name in (
            "energy",
            "warmth",
            "patience",
            "confidence",
            "curiosity",
            *_ACTIVATIONS,
        )
        if abs(values[name] - original_values[name]) >= 0.0005
    )

    evolution_start = max(float(state.updated_at), current - _MAX_TIME_EVOLUTION_SECONDS)
    before_circadian = values["energy"]
    cursor = evolution_start
    target = _circadian_energy_target(time.localtime(current).tm_hour)
    while cursor < current:
        segment_end = min(current, (math.floor(cursor / 3600.0) + 1) * 3600.0)
        if segment_end <= cursor:
            segment_end = min(current, cursor + 3600.0)
        midpoint = cursor + (segment_end - cursor) / 2.0
        target = _circadian_energy_target(time.localtime(midpoint).tm_hour)
        values["energy"] = _decay(
            values["energy"],
            target,
            (segment_end - cursor) / 3600.0,
            18.0,
        )
        cursor = segment_end
    circadian_change = values["energy"] - before_circadian

    ambient: dict[str, float] = {}
    bucket = math.floor(evolution_start / _AMBIENT_WINDOW_SECONDS)
    final_bucket = math.floor(max(evolution_start, current - 1e-6) / _AMBIENT_WINDOW_SECONDS)
    while bucket <= final_bucket:
        bucket_start = bucket * _AMBIENT_WINDOW_SECONDS
        overlap = max(
            0.0,
            min(current, bucket_start + _AMBIENT_WINDOW_SECONDS)
            - max(evolution_start, bucket_start),
        )
        fraction = overlap / _AMBIENT_WINDOW_SECONDS
        if fraction > 0.0:
            for name in ("energy", "warmth", "patience", "confidence", "curiosity"):
                unit = _stable_mood_unit(f"{profile_seed}:{bucket}:{name}")
                delta = (unit * 2.0 - 1.0) * 0.004 * fraction
                values[name] += delta
                ambient[name] = ambient.get(name, 0.0) + delta
        bucket += 1

    if elapsed_hours > 6.0:
        values["stimulation"] -= min(0.05, (elapsed_hours - 6.0) / 72.0 * 0.05)
    cause = state.cause if max(values[name] for name in _ACTIVATIONS) >= 0.035 else ""
    trigger_repetitions = state.trigger_repetitions if elapsed_hours < 2.0 else 0
    evolved = _make_state(
        state,
        updated_at=current,
        values=values,
        cause=cause,
        momentum=_decay(state.momentum, 0.0, elapsed_hours, 3.0),
        last_trigger=state.last_trigger if trigger_repetitions else "",
        trigger_repetitions=trigger_repetitions,
    )
    reasons = ["circadian_shift_applied", "ambient_drift_applied"]
    if elapsed_seconds > _MAX_TIME_EVOLUTION_SECONDS:
        reasons.append("bounded_time_catch_up")
    return MoodEvolution(
        evolved,
        elapsed_seconds=elapsed_seconds,
        elapsed_decay_delta=decay_delta,
        circadian_target=target,
        circadian_delta=(("energy", circadian_change),),
        ambient_delta=tuple(
            (name, delta) for name, delta in ambient.items() if abs(delta) >= 0.0005
        ),
        reason_codes=tuple(reasons),
    )


def _stable_mood_unit(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _circadian_energy_target(hour: int) -> float:
    if 5 <= hour < 9:
        return 0.68
    if 9 <= hour < 17:
        return 0.70
    if 17 <= hour < 23:
        return 0.62
    return 0.52


def semantic_event_from_text(user_text: str) -> SemanticEvent:
    """Normalize one explicit activity or outcome without discarding its verb."""

    text = compact_text(user_text, 240)
    if not text:
        return SemanticEvent()

    question_mode = _question_mode(text)
    changed_condition, affected_reason, reason_effect = _hypothetical_condition(
        text,
        question_mode=question_mode,
    )
    semantic_fields = {
        "question_mode": question_mode,
        "changed_condition": changed_condition,
        "affected_reason": affected_reason,
        "reason_effect": reason_effect,
    }

    def actor_of(match: re.Match[str] | None) -> str:
        if match is None:
            return "unknown"
        return "shared" if match.groupdict().get("actor", "").casefold() == "we" else "Arcane"

    def clean(value: str, *, limit: int = 180) -> str:
        normalized = compact_text(value, limit).strip(" ,:-")
        return re.sub(r"^(?:the|a|an)\s+", "", normalized, flags=re.IGNORECASE)

    completion = (
        _COMPLETION_ACTION.search(text)
        or _DONE_ACTION.search(text)
        or _QUALIFIED_COMPLETION_ACTION.search(text)
    )
    predicate = (
        _COMPLETION_PREDICATE.search(text)
        or _QUALIFIED_COMPLETION_PREDICATE.search(text)
    )
    false_completion = bool(_FALSE_COMPLETION.search(text))
    partial = bool(_PARTIAL_COMPLETION.search(text))
    negated = bool(_COMPLETION_NEGATION.search(text))
    if completion is not None or predicate is not None or false_completion:
        match = completion or predicate
        subject = clean(match.groupdict().get("subject", "")) if match else "it"
        if false_completion:
            status = "failed"
        elif partial:
            status = "partial"
        elif negated:
            status = "not_completed"
        else:
            status = "completed"
        return SemanticEvent(
            subject=subject,
            event_type="completion",
            status=status,
            actor=(
                actor_of(completion)
                if completion is not None
                else "Arcane" if re.search(r"\bI\b", text, re.IGNORECASE) else "unknown"
            ),
            target=subject,
            temporal_state="past" if status == "completed" else "current",
            negated=negated or false_completion,
            confidence=0.96,
            **semantic_fields,
        )

    transition = _ACTIVITY_TRANSITION.search(text)
    if transition is not None:
        transition_kind = transition.group("transition").casefold()
        raw_subject = clean(transition.group("subject"))
        status = {
            "paused": "paused",
            "started": "started",
            "began": "started",
            "resumed": "resumed",
            "switched to": "switched",
        }.get(transition_kind, "ongoing")
        return SemanticEvent(
            subject=raw_subject,
            event_type="activity",
            status=status,
            actor=actor_of(transition),
            target=raw_subject,
            temporal_state="current",
            confidence=0.94,
            **semantic_fields,
        )

    activity = _ACTIVITY_UPDATE.search(text)
    if activity is not None:
        action = re.sub(r"\s+", " ", activity.group("action").casefold())
        target = clean(activity.group("subject") or "")
        subject = clean(f"{action} {target}")
        return SemanticEvent(
            subject=subject,
            event_type="activity",
            status="ongoing",
            actor=actor_of(activity),
            target=target,
            temporal_state="current",
            confidence=0.95,
            **semantic_fields,
        )

    past_activity = _PAST_ACTIVITY.search(text)
    if past_activity is not None:
        action = re.sub(r"\s+", " ", past_activity.group("action").casefold())
        target = clean(past_activity.group("subject"))
        return SemanticEvent(
            subject=clean(f"{action} {target}"),
            event_type="activity",
            status="completed",
            actor=actor_of(past_activity),
            target=target,
            temporal_state="past",
            confidence=0.92,
            **semantic_fields,
        )

    failure = _FAILURE_STATE.search(text)
    if failure is not None:
        subject = clean(failure.group("subject"))
        if re.search(r"\bwe\b", text, re.IGNORECASE):
            actor = "shared"
        elif re.search(r"\b(?:I|my)\b", text, re.IGNORECASE):
            actor = "Arcane"
        else:
            actor = "unknown"
        return SemanticEvent(
            subject=subject,
            event_type="failure",
            status="failed",
            actor=actor,
            target=subject,
            temporal_state="current",
            confidence=0.90,
            **semantic_fields,
        )
    return SemanticEvent(**semantic_fields)


def _question_mode(text: str) -> str:
    """Classify only the decision semantics needed by the existing turn analysis."""

    if _HYPOTHETICAL_CHOICE_QUESTION.search(text):
        return "hypothetical_choice"
    if _PERSONAL_CHOICE_QUESTION.search(text):
        return (
            "hypothetical_choice"
            if re.search(r"\b(?:if|suppose|assuming|what if)\b", text, re.IGNORECASE)
            else "personal_choice"
        )
    if _OBJECTIVE_COMPARISON_QUESTION.search(text):
        return "objective_comparison"
    if "?" in text or _FACTUAL_QUESTION.search(text):
        return "factual"
    return "none"


def _hypothetical_condition(
    text: str,
    *,
    question_mode: str = "",
) -> tuple[str, str, str]:
    if (question_mode or _question_mode(text)) != "hypothetical_choice":
        return "", "", ""
    for pattern, condition, reason, effect in _HYPOTHETICAL_CONDITIONS:
        if pattern.search(text):
            return condition, reason, effect
    return "", "", ""


def analyze_turn(
    user_text: str,
    *,
    emotion_state: EmotionState | None = None,
    turn_context: TurnContext | None = None,
    now: float | None = None,
    emotion_state_is_current: bool = False,
    code_context_requested: bool = False,
    code_context_attached: bool = False,
    semantic_event: SemanticEvent | None = None,
) -> TurnSignal:
    summary = compact_text(user_text, _SUMMARY_CHARS)
    lower = summary.casefold()
    tokens = words(lower)
    event = semantic_event or semantic_event_from_text(summary)
    evidence = _appraise(
        summary,
        lower,
        tokens,
        turn_context=turn_context,
        semantic_event=event,
        code_context_requested=code_context_requested,
        code_context_attached=code_context_attached,
    )
    current = time.time() if now is None else float(now)
    previous = emotion_state or EmotionState(updated_at=current)
    current_emotion = (
        previous
        if emotion_state_is_current
        else advance_emotion(previous, now=current)
    )
    emotional_signal = detect_emotional_signal(evidence)
    updated = update_emotion_state(
        current_emotion,
        emotional_signal,
        evidence,
        now=current,
    )
    contextual_reaction = contextual_reaction_for(
        current_emotion,
        updated,
        emotional_signal,
        turn_context or TurnContext(),
    )
    identity_attribute, current_activity = _identity_focus(lower)
    current_thought = bool(_CURRENT_THOUGHT_PATTERN.search(lower))
    intent, tone, trigger = _intent(
        evidence,
        updated,
        identity_question=bool(identity_attribute),
    )
    if current_thought and intent == "casual":
        intent = "reflection"
    topic, confidence = topic_from_text(summary)
    technical = evidence["technical"] >= 0.5
    direct = evidence["directness"] >= 0.5
    correction = evidence["correction"] >= 0.5
    return TurnSignal(
        summary=summary,
        topic=topic,
        topic_confidence=confidence,
        intent=intent,
        tone=tone,
        emotion_state=updated,
        task=topic if technical or direct else "",
        correction=summary if correction else "",
        trigger=trigger,
        praise=evidence["praise"] >= 0.5,
        criticism=evidence["criticism"] >= 0.5,
        correction_requested=correction,
        hostility=evidence["hostility"] >= 0.5,
        frustration=evidence["user_frustration"] >= 0.45,
        teasing=evidence["playfulness"] >= 0.5,
        sadness=evidence["vulnerability"] >= 0.5,
        technical=technical,
        task_success=evidence["task_success"] >= 0.5,
        task_failure=evidence["task_failure"] >= 0.5,
        code_context_requested=code_context_requested,
        code_context_attached=code_context_attached,
        identity_attribute=identity_attribute,
        current_activity=current_activity,
        current_thought=current_thought,
        emotional_signal=emotional_signal,
        contextual_reaction=contextual_reaction,
        semantic_event=event,
    )


def _appraise(
    original: str,
    text: str,
    tokens: set[str],
    *,
    turn_context: TurnContext | None,
    semantic_event: SemanticEvent,
    code_context_requested: bool,
    code_context_attached: bool,
) -> dict[str, float]:
    gratitude = float(bool(re.search(r"\b(?:thanks|thank you|appreciate it)\b", text)))
    apology = float(bool(_APOLOGY_MARKERS.search(text)))
    positive = float(bool(_POSITIVE_MARKERS.search(text)))
    negative = float(bool(_NEGATIVE_MARKERS.search(text)))
    correction = float(bool(_CORRECTION_MARKERS.search(text)))
    failure = float(bool(_FAILURE_MARKERS.search(text)))
    success = float(bool(_SUCCESS_MARKERS.search(text)))
    context = turn_context or TurnContext()
    if semantic_event.confirmed_completion and context.completion_meaningful:
        success = 1.0
    elif semantic_event.event_type == "completion" and (
        semantic_event.negated or semantic_event.status in {"partial", "not_completed", "failed"}
    ):
        success = 0.0
    if semantic_event.status == "failed":
        failure = 1.0
    contextual_reference = bool(
        context.unresolved_problem
        and (
            low_content(text)
            or re.search(
                r"\b(?:again|fixed|solved|worked|works|broke|failed|same (?:thing|issue))\b",
                text,
            )
        )
    )
    if contextual_reference and re.search(r"\b(?:fixed|solved|worked|works)\b", text):
        success = 1.0
    if contextual_reference and re.search(r"\b(?:again|broke|failed|same (?:thing|issue))\b", text):
        failure = 1.0
    playfulness = float(bool(_PLAYFUL_MARKERS.search(text)))
    vulnerability_hits = len(_VULNERABILITY_MARKERS.findall(text))
    explicit_distress = bool(_DISTRESS_PHRASES.search(text))
    vulnerability = min(
        1.0,
        vulnerability_hits * 0.45
        + (
            0.25
            if vulnerability_hits
            and re.search(r"\b(?:i am|i feel|i'm|my|me)\b", text)
            else 0.0
        ),
    )
    negated_distress = bool(
        re.search(
            r"\b(?:am|feel|i'm)\s+not\s+(?:anxious|awful|hurt|lonely|"
            r"miserable|overwhelmed|sad|scared|terrible|upset)\b",
            text,
        )
        or re.search(
            r"\bnot\s+(?:anxious|hurt|lonely|overwhelmed|sad|scared|upset)\b",
            text,
        )
    )
    if explicit_distress:
        vulnerability = max(vulnerability, 0.82)
    if negated_distress or _CURRENT_IMPROVEMENT.search(text):
        vulnerability *= 0.10
    hostility = float(bool(_HOSTILITY_MARKERS.search(text)))
    repeated_failure = float(
        bool(
            failure
            and (
                "again" in tokens
                or "keeps happening" in text
                or context.unresolved_problem and context.repeated_topic_count >= 2
            )
        )
    )
    directness = float(bool(_DIRECT_PATTERN.search(text)))
    technical = float(
        code_context_requested
        or code_context_attached
        or bool(_IMPLEMENTATION_TOPIC_PATTERN.search(text))
        or (
            bool(tokens & _CODE_TERMS)
            and bool(
                tokens
                & (_CODE_ACTIONS | _DIRECT | _PROBLEM_WORDS | {"how", "why", "what"})
            )
        )
        or (contextual_reference and bool(context.current_task or context.current_topic))
    )
    question = float("?" in original)
    punctuation = min(1.0, len(re.findall(r"[!?]", original)) / 4.0)
    letters = [char for char in original if char.isalpha()]
    capitals = (
        sum(char.isupper() for char in letters) / len(letters) if len(letters) >= 6 else 0.0
    )
    criticism = float(
        bool(_CRITICISM_MARKERS.search(text))
        or (correction and bool(tokens & {"answer", "response", "you"}))
    )
    praise = float(
        positive
        and not gratitude
        and bool(tokens & {"answer", "job", "work", "you"})
    )
    low = low_content(text)
    personally_directed = float(
        bool(tokens & {"akane", "you", "your", "you're", "youre"})
        or bool(gratitude or apology)
    )
    social_warmth = max(
        gratitude * 0.45,
        apology * 0.60,
        praise * 0.85,
        positive * (0.36 if context.familiar_relationship else 0.30),
    )
    excitement = max(
        positive * context.memory_relevance,
        playfulness * 0.45,
    )
    tension = max(
        hostility,
        failure * 0.62,
        negative * 0.48,
        criticism * 0.56,
        punctuation * 0.30,
        min(1.0, capitals * 1.4) * 0.35,
    )
    return {
        "pleasantness": clamp(
            max(success, positive * 0.55 if not gratitude else gratitude * 0.25)
        ),
        "social_warmth": clamp(social_warmth),
        "tension": clamp(tension),
        "playfulness": clamp(playfulness),
        "vulnerability": clamp(vulnerability),
        "hostility": clamp(hostility),
        "correction": clamp(correction),
        "criticism": clamp(criticism),
        "praise": clamp(praise),
        "task_success": clamp(success),
        "task_failure": clamp(failure),
        "directness": clamp(directness),
        "technical": clamp(technical),
        "novelty": 0.0 if low else clamp(0.35 + question * 0.20),
        "repetition": clamp(repeated_failure),
        "user_frustration": clamp(max(failure * 0.68, negative * 0.52, tension * 0.58)),
        "gratitude": clamp(gratitude),
        "apology": clamp(apology),
        "excitement": clamp(excitement),
        "personally_directed": personally_directed,
        "memory_relevance": clamp(context.memory_relevance),
        "meaningful_memory": float(context.meaningful_memory),
        "serious": clamp(
            max(
                vulnerability,
                float(bool(tokens & {"deadline", "important", "risk", "serious"})),
            )
        ),
        "low_content": float(low),
    }


_ACTIVATIONS = (
    "amusement",
    "excitement",
    "embarrassment",
    "concern",
    "frustration",
    "irritation",
)


def detect_emotional_signal(evidence: dict[str, float]) -> EmotionalSignal:
    """Convert current-message evidence into one bounded, non-mutating signal."""

    kind = _evidence_trigger(evidence) or "neutral"
    evidence_key = "vulnerability" if kind == "distress" else kind
    intensity = max(
        evidence.get(evidence_key, 0.0),
        evidence["tension"] if kind in {"hostility", "criticism"} else 0.0,
        evidence["social_warmth"] if kind in {"praise", "apology"} else 0.0,
    )
    confidence = 0.0 if kind == "neutral" else min(
        1.0,
        0.62 + intensity * 0.28 + evidence["personally_directed"] * 0.08,
    )
    return EmotionalSignal(
        kind=kind,
        intensity=clamp(intensity),
        confidence=clamp(confidence),
        cause=_cause(evidence, "") if kind != "neutral" else "",
    )


def contextual_reaction_for(
    prior: EmotionState,
    updated: EmotionState,
    signal: EmotionalSignal,
    context: TurnContext,
) -> ContextualReaction:
    """Derive a temporary response bias without storing it as personality or fact."""

    active = _strongest_active(updated)
    if signal.kind == "neutral" and active is None:
        return ContextualReaction()
    if signal.kind == "neutral":
        kind = f"lingering_{active.kind}"
        intensity = active.intensity * 0.55
        cause = active.cause
    else:
        kind = {
            "distress": "concern",
            "hostility": "irritation",
            "playfulness": "amusement",
            "praise": "appreciation",
            "task_success": (
                "relief" if context.completion_resolves_thread else "satisfaction"
            ),
        }.get(signal.kind, signal.kind)
        mood_bias = 1.0
        if signal.kind in {"hostility", "criticism"}:
            mood_bias += prior.irritation * 0.25 + prior.frustration * 0.15
        elif signal.kind in {"praise", "apology", "social_warmth"}:
            mood_bias += prior.warmth * 0.10
        intensity = signal.intensity * mood_bias
        cause = signal.cause

    return ContextualReaction(
        kind=kind,
        intensity=clamp(intensity),
        cause=cause,
    )


def update_emotion_state(
    state: EmotionState,
    signal: EmotionalSignal,
    evidence: dict[str, float],
    *,
    now: float,
) -> EmotionState:
    """Canonical deterministic mutation of persistent mood and transient emotion."""

    values = _state_values(state)
    trigger = "" if signal.kind == "neutral" else signal.kind
    if not trigger:
        return _make_state(
            state,
            updated_at=now,
            values=values,
            cause=state.cause,
            momentum=state.momentum,
            last_trigger=state.last_trigger,
            trigger_repetitions=state.trigger_repetitions,
        )
    same_trigger = bool(trigger and trigger == state.last_trigger)
    repetitions = (
        min(12, state.trigger_repetitions + 1)
        if same_trigger
        else 1 if trigger else max(0, state.trigger_repetitions - 1)
    )
    resistance = max(0.42, 1.0 - max(0, repetitions - 1) * 0.26)
    if trigger == "task_failure" and evidence["repetition"] >= 0.5:
        resistance = max(0.88, resistance)
    memory_boost = (
        1.0 + evidence["memory_relevance"] * 0.16
        if trigger and evidence["meaningful_memory"] and not same_trigger
        else 1.0
    )
    reaction_scale = resistance * memory_boost

    def shift(name: str, delta: float, limit: float = 0.18) -> None:
        low = -1.0 if name == "valence" else 0.0
        scaled = delta * reaction_scale
        values[name] = clamp(
            values[name] + clamp(scaled, 0.0, -limit, limit),
            0.0,
            low,
            1.0,
        )

    pleasant = evidence["pleasantness"]
    tension = evidence["tension"]
    failure = evidence["task_failure"]
    success = evidence["task_success"]
    hostility = evidence["hostility"]
    repeated = evidence["repetition"]
    technical = evidence["technical"]
    direct = evidence["directness"]
    vulnerability = evidence["vulnerability"]

    apology = evidence["apology"]
    excitement = evidence["excitement"]

    shift(
        "valence",
        pleasant * 0.035 + apology * 0.018 - tension * 0.028 - hostility * 0.025,
        0.05,
    )
    shift(
        "arousal",
        technical * 0.025
        + direct * 0.018
        + tension * 0.025
        + excitement * 0.035
        - apology * 0.018,
        0.06,
    )
    shift(
        "energy",
        technical * 0.010 + excitement * 0.025 + success * 0.012 - failure * 0.010,
        0.03,
    )
    shift(
        "warmth",
        evidence["social_warmth"] * 0.028
        + apology * 0.022
        + vulnerability * 0.020
        - hostility * 0.045,
        0.06,
    )
    shift(
        "curiosity",
        technical * 0.030
        + excitement * 0.025
        + evidence["novelty"] * 0.012
        - repeated * 0.015,
        0.05,
    )
    shift(
        "confidence",
        success * 0.035
        + evidence["praise"] * 0.018
        - evidence["correction"] * 0.018
        - failure * 0.015
        - hostility * 0.012,
        0.05,
    )
    shift(
        "stimulation",
        technical * 0.025
        + direct * 0.012
        + evidence["playfulness"] * 0.025
        + excitement * 0.035
        + tension * 0.015,
        0.06,
    )
    shift(
        "patience",
        apology * 0.025
        + evidence["social_warmth"] * 0.010
        - hostility * 0.040
        - evidence["criticism"] * 0.016
        - repeated * 0.018,
        0.05,
    )
    shift(
        "amusement",
        evidence["playfulness"] * (1.0 - tension) * 0.22
        - (0.015 if not evidence["playfulness"] else 0.0),
    )
    shift(
        "excitement",
        excitement * 0.30 + success * 0.10 - tension * 0.08 - apology * 0.02,
        0.34,
    )
    shift(
        "embarrassment",
        evidence["playfulness"] * evidence["personally_directed"] * 0.16
        - hostility * 0.10,
        0.18,
    )
    shift(
        "concern",
        vulnerability * 0.40
        + failure * 0.08
        + evidence["user_frustration"] * 0.08
        - success * 0.12
        - (0.01 if not vulnerability and not failure else 0.0),
        0.40,
    )
    shift(
        "frustration",
        failure * 0.12
        + repeated * 0.14
        + evidence["correction"] * 0.025
        - success * 0.30
        - apology * 0.06
        - (0.01 if not failure else 0.0),
    )
    shift(
        "irritation",
        hostility * 0.24
        + evidence["criticism"] * 0.06
        + repeated * 0.03
        - success * 0.10
        - apology * 0.07
        - (0.015 if not hostility else 0.0),
    )

    cause = _cause(evidence, state.cause)
    activation = max(
        evidence["hostility"],
        evidence["vulnerability"],
        evidence["task_failure"],
        evidence["task_success"],
        evidence["criticism"],
        evidence["playfulness"],
        evidence["praise"],
        evidence["apology"],
        evidence["excitement"],
    )
    momentum = clamp(state.momentum * 0.72 + activation * resistance * 0.24)
    return _make_state(
        state,
        updated_at=now,
        values=values,
        cause=cause,
        momentum=momentum,
        last_trigger=trigger,
        trigger_repetitions=repetitions,
        recent_influence=trigger,
    )


def apply_mood_effects(
    state: EmotionState,
    effects: tuple[tuple[str, float], ...],
    *,
    now: float,
    cause: str,
    short_emotion: str = "",
) -> EmotionState:
    """Apply one recorded event's bounded effects to the candidate mood state."""

    values = _state_values(state)
    allowed = {"energy", "warmth", "patience", "confidence", "curiosity"}
    for name, raw_delta in effects:
        if name in allowed:
            values[name] += clamp(raw_delta, 0.0, -0.10, 0.10)
    transient = {
        "amusement": "amusement",
        "excited": "excitement",
        "excitement": "excitement",
        "annoyed": "irritation",
        "irritation": "irritation",
        "concerned": "concern",
        "concern": "concern",
    }.get(short_emotion, "")
    if transient:
        values[transient] += 0.06
    return _make_state(
        state,
        updated_at=now,
        values=values,
        cause=cause,
        recent_influence="offscreen_event",
    )


def _emotion_scores(values: dict[str, float]) -> dict[str, float]:
    positive_valence = max(0.0, values["valence"])
    return {
        "relaxed": 0.80
        - values["arousal"] * 0.35
        - values["frustration"] * 0.40
        - values["irritation"] * 0.50
        - values["concern"] * 0.22,
        "warm": values["warmth"] * 0.85 + positive_valence * 0.22,
        "amused": values["amusement"] * 1.15 + positive_valence * 0.18,
        "excited": values["excitement"] * 1.12
        + values["energy"] * 0.18
        + positive_valence * 0.12,
        "embarrassed": values["embarrassment"] * 1.12
        + values["amusement"] * 0.12,
        "curious": values["curiosity"] * 0.82 + values["stimulation"] * 0.14,
        "focused": 0.28
        + values["curiosity"] * 0.40
        + values["confidence"] * 0.20
        + values["arousal"] * 0.30
        + values["stimulation"] * 0.10,
        "confident": values["confidence"] * 0.86 + positive_valence * 0.22,
        "concerned": 0.15 + values["concern"] * 1.15 + values["warmth"] * 0.18,
        "tired": (1.0 - values["energy"]) * 0.82 + (1.0 - values["arousal"]) * 0.16,
        "bored": 0.18
        + (1.0 - values["stimulation"]) * 0.85
        + (1.0 - values["curiosity"]) * 0.16,
        "frustrated": 0.10
        + values["frustration"] * 1.15
        + values["arousal"] * 0.16,
        "mildly_exasperated": 0.10
        + values["frustration"] * 0.88
        + values["irritation"] * 0.32
        + values["confidence"] * 0.10,
        "irritated": values["irritation"] * 1.12 + values["frustration"] * 0.18,
        "guarded": 0.15
        + values["irritation"] * 0.95
        + (1.0 - values["warmth"]) * 0.55
        + values["arousal"] * 0.20,
    }


def _select_emotions(
    values: dict[str, float],
    previous: EmotionState,
) -> tuple[str, str]:
    scores = _emotion_scores(values)
    ranked = sorted(scores, key=scores.get, reverse=True)
    dominant = ranked[0]
    if previous.dominant in scores and previous.dominant != dominant:
        inertia = (
            0.05
            + max(values[name] for name in _ACTIVATIONS) * 0.08
            + previous.momentum * 0.08
        )
        if max(values[name] for name in _ACTIVATIONS) >= 0.30:
            inertia *= 0.40
        if scores[dominant] < scores[previous.dominant] + inertia:
            dominant = previous.dominant

    secondary = ""
    for candidate in ranked:
        if candidate == dominant:
            continue
        if (
            candidate in _COMPATIBLE_SECONDARIES.get(dominant, set())
            and scores[candidate] >= 0.48
            and scores[candidate] >= scores[dominant] - 0.22
        ):
            secondary = candidate
            break
    return dominant, secondary


def _make_state(
    previous: EmotionState,
    *,
    updated_at: float,
    values: dict[str, float],
    cause: str = "",
    momentum: float | None = None,
    last_trigger: str | None = None,
    trigger_repetitions: int | None = None,
    recent_influence: str = "",
) -> EmotionState:
    bounded = {
        name: clamp(value, _BASELINES.get(name, 0.0), -1.0 if name == "valence" else 0.0, 1.0)
        for name, value in values.items()
    }
    dominant, secondary = _select_emotions(bounded, previous)
    current_time = max(0.0, float(updated_at))
    active_emotions = _updated_active_emotions(
        previous,
        bounded,
        cause=compact_text(cause, 100),
        trigger=compact_text(recent_influence, 24).lower(),
        now=current_time,
    )
    influences = previous.recent_influences
    if recent_influence:
        earlier = tuple(item for item in influences if item != recent_influence)
        influences = (*earlier, recent_influence)[-4:]
    return EmotionState(
        updated_at=current_time,
        dominant=dominant,
        secondary=secondary,
        cause=compact_text(cause, 100),
        mood=_mood_label(bounded, previous.mood),
        momentum=clamp(previous.momentum if momentum is None else momentum),
        last_trigger=compact_text(
            previous.last_trigger if last_trigger is None else last_trigger,
            32,
        ),
        trigger_repetitions=max(
            0,
            min(
                12,
                previous.trigger_repetitions
                if trigger_repetitions is None
                else int(trigger_repetitions),
            ),
        ),
        active_emotions=active_emotions,
        recent_influences=influences,
        **bounded,
    )


def _state_values(state: EmotionState) -> dict[str, float]:
    return {
        "valence": state.valence,
        "arousal": state.arousal,
        "energy": state.energy,
        "warmth": state.warmth,
        "curiosity": state.curiosity,
        "confidence": state.confidence,
        "stimulation": state.stimulation,
        "patience": state.patience,
        "amusement": state.amusement,
        "excitement": state.excitement,
        "embarrassment": state.embarrassment,
        "concern": state.concern,
        "frustration": state.frustration,
        "irritation": state.irritation,
    }


_TRANSIENT_HALF_LIVES = {
    "amusement": 0.25,
    "excitement": 0.75,
    "embarrassment": 0.40,
    "concern": 0.75,
    "frustration": 0.60,
    "irritation": 0.50,
}


def _updated_active_emotions(
    previous: EmotionState,
    values: dict[str, float],
    *,
    cause: str,
    trigger: str,
    now: float,
) -> tuple[ShortLivedEmotion, ...]:
    prior = {item.kind: item for item in previous.active_emotions}
    trigger_kind = {
        "playfulness": "amusement",
        "excitement": "excitement",
        "distress": "concern",
        "task_failure": "frustration",
        "repetition": "frustration",
        "correction": "frustration",
        "criticism": "irritation",
        "hostility": "irritation",
    }.get(trigger, "")
    records: list[ShortLivedEmotion] = []
    for kind in _ACTIVATIONS:
        intensity = clamp(values.get(kind, 0.0))
        if intensity < 0.035:
            continue
        existing = prior.get(kind)
        created_at = existing.created_at if existing else now
        record_cause = (
            cause if trigger_kind == kind and cause else existing.cause if existing else cause
        )
        half_life = _TRANSIENT_HALF_LIVES[kind]
        records.append(
            ShortLivedEmotion(
                kind=kind,
                intensity=intensity,
                cause=record_cause,
                created_at=created_at,
                updated_at=now,
                decay_rate=math.log(2.0) / half_life,
                remaining_relevance=clamp(intensity / 0.35),
            )
        )
    return tuple(sorted(records, key=lambda item: (-item.intensity, item.kind))[:3])


def _strongest_active(state: EmotionState) -> ShortLivedEmotion | None:
    return max(
        state.active_emotions,
        key=lambda item: (item.intensity, item.remaining_relevance, item.kind),
        default=None,
    )


def format_response_disposition(signal: TurnSignal) -> str:
    """Render the one canonical, compact behavioral affect summary for prompting."""

    state = signal.emotion_state
    active = _strongest_active(state)
    reaction = signal.contextual_reaction
    reaction_dispositions = {
        "concern": "gentle and concerned",
        "irritation": "restrained and a little blunt",
        "criticism": "direct and slightly guarded",
        "amusement": "playful and relaxed",
        "appreciation": "warm and appreciative",
        "satisfaction": "quietly pleased and specific",
        "relief": "relieved and encouraging",
        "excitement": "lively and enthusiastic",
        "correction": "focused and receptive",
    }
    active_dispositions = {
        "concern": "gentle and attentive",
        "irritation": "controlled and a little terse",
        "frustration": "focused and persistent",
        "amusement": "lightly playful",
        "excitement": "energetic and engaged",
        "embarrassment": "warm with slight hesitation",
    }
    mood_dispositions = {
        "steady": "relaxed and neutral",
        "relaxed": "relaxed and easygoing",
        "warm": "warm and open",
        "cheerful": "bright and upbeat",
        "curious": "curious and attentive",
        "energetic": "lively and engaged",
        "confident": "confident and direct",
        "tired": "low-key and measured",
        "irritable": "blunt and impatient",
        "guarded": "reserved and cautious",
        "low": "quiet and subdued",
        "pensive": "thoughtful and curious",
    }
    if reaction.intensity >= 0.10 and reaction.kind in reaction_dispositions:
        overall = reaction_dispositions[reaction.kind]
    elif active is not None and active.intensity >= 0.04:
        overall = active_dispositions.get(
            active.kind,
            mood_dispositions.get(state.mood, "relaxed and neutral"),
        )
    else:
        overall = mood_dispositions.get(state.mood, "relaxed and neutral")
    return f"Disposition: {overall}."


def build_affect_trace(
    prior: EmotionState,
    decayed: EmotionState,
    signal: TurnSignal,
    *,
    evolution: MoodEvolution | None = None,
    event_delta: tuple[tuple[str, float], ...] = (),
    event_ids: tuple[str, ...] = (),
    extra_reason_codes: tuple[str, ...] = (),
) -> AffectTrace:
    """Build content-free proposed-turn diagnostics."""

    final = signal.emotion_state
    changes: list[str] = []
    for name in ("valence", "energy", "warmth", "patience", "confidence", "curiosity"):
        delta = getattr(final, name) - getattr(prior, name)
        if abs(delta) >= 0.005:
            changes.append(f"{name} {delta:+.3f}")
    active = _strongest_active(final)
    elapsed_state = evolution.state if evolution is not None else decayed
    conversation_changes = tuple(
        (name, getattr(final, name) - getattr(decayed, name))
        for name in ("energy", "warmth", "patience", "confidence", "curiosity")
        if abs(getattr(final, name) - getattr(decayed, name)) >= 0.0005
    )
    decay_applied = max(
        (
            getattr(prior, name) - getattr(elapsed_state, name)
            for name in _ACTIVATIONS
        ),
        default=0.0,
    )
    if signal.emotional_signal.kind == "neutral":
        decay_applied = max(
            decay_applied,
            max(
                (getattr(prior, name) - getattr(final, name) for name in _ACTIVATIONS),
                default=0.0,
            ),
        )
    short_change = "none"
    prior_active = _strongest_active(prior)
    if active is not None and (
        prior_active is None
        or active.kind != prior_active.kind
        or active.intensity > prior_active.intensity + 0.005
    ):
        short_change = f"created:{active.kind}"
    elif prior_active is not None and active is None:
        short_change = f"expired:{prior_active.kind}"

    reasons = list(evolution.reason_codes if evolution is not None else ())
    reasons.extend(extra_reason_codes)
    emotional_signal = signal.emotional_signal
    if emotional_signal.kind == "neutral":
        reasons.append("below_signal_threshold")
        if low_content(signal.summary):
            reasons.append("ambiguous_input")
    else:
        reasons.append(f"{emotional_signal.kind}_detected")
        if signal.contextual_reaction.kind != "neutral":
            reasons.append(
                f"reaction_mapped_to_{signal.contextual_reaction.kind}"
            )
    if short_change.startswith("created:"):
        reasons.append("short_emotion_created")
    if not changes and final.mood == "steady":
        reasons.append("mood_near_baseline")
    reasons = list(dict.fromkeys(reason for reason in reasons if reason))

    def persistent_vector(state: EmotionState) -> str:
        return (
            f"energy={state.energy:.2f}, warmth={state.warmth:.2f}, "
            f"patience={state.patience:.2f}, confidence={state.confidence:.2f}, "
            f"curiosity={state.curiosity:.2f}"
        )

    return AffectTrace(
        prior_mood=prior.mood,
        detected_signal=signal.emotional_signal.kind,
        signal_intensity=round(signal.emotional_signal.intensity, 3),
        contextual_reaction=signal.contextual_reaction.kind,
        state_changes=tuple(changes[:4]),
        active_emotion=active.kind if active else "none",
        active_intensity=round(active.intensity, 3) if active else 0.0,
        decay_applied=round(max(0.0, decay_applied), 3),
        final_disposition=f"{final.mood}/{final.dominant}",
        baseline_persistent_mood=(
            f"energy={_BASELINES['energy']:.2f}, warmth={_BASELINES['warmth']:.2f}, "
            f"patience={_BASELINES['patience']:.2f}, "
            f"confidence={_BASELINES['confidence']:.2f}, "
            f"curiosity={_BASELINES['curiosity']:.2f}"
        ),
        prior_persistent_mood=persistent_vector(prior),
        resulting_persistent_mood=persistent_vector(final),
        candidate_mood_delta=", ".join(changes[:4]) or "none",
        signal_confidence=round(emotional_signal.confidence, 3),
        reaction_mapping=(
            f"{emotional_signal.kind}->{signal.contextual_reaction.kind}"
            if signal.contextual_reaction.kind != "neutral"
            else "none"
        ),
        elapsed_seconds=evolution.elapsed_seconds if evolution is not None else 0.0,
        elapsed_decay_delta=(
            evolution.elapsed_decay_delta if evolution is not None else ()
        ),
        circadian_target=evolution.circadian_target if evolution is not None else 0.0,
        circadian_delta=evolution.circadian_delta if evolution is not None else (),
        ambient_delta=evolution.ambient_delta if evolution is not None else (),
        event_delta=event_delta,
        conversation_delta=conversation_changes,
        short_emotion_change=short_change,
        event_ids=event_ids,
        reason_codes=tuple(reasons),
    )


def _cause(evidence: dict[str, float], previous: str) -> str:
    for name, text in (
        ("hostility", "current hostility"),
        ("vulnerability", "user distress"),
        ("repetition", "unresolved failure"),
        ("task_failure", "reported failure"),
        ("task_success", "confirmed success"),
        ("apology", "user apology"),
        ("excitement", "a relevant shared interest"),
        ("correction", "the correction"),
        ("criticism", "current criticism"),
        ("playfulness", "the playful exchange"),
        ("praise", "user praise"),
        ("social_warmth", "the friendly exchange"),
    ):
        if evidence[name] >= 0.5:
            return text
    return previous


def _evidence_trigger(evidence: dict[str, float]) -> str:
    for name in (
        "hostility",
        "vulnerability",
        "task_failure",
        "task_success",
        "apology",
        "correction",
        "criticism",
        "praise",
        "excitement",
        "playfulness",
        "social_warmth",
    ):
        if evidence[name] >= _SIGNAL_ACTIVATION_THRESHOLD:
            return "distress" if name == "vulnerability" else name
    return ""


def _mood_label(values: dict[str, float], previous: str) -> str:
    deviations = {
        name: values[name] - _BASELINES[name]
        for name in ("energy", "warmth", "patience", "confidence", "curiosity")
    }
    largest = max(abs(delta) for delta in deviations.values())
    scores = {
        "steady": 0.045 - largest,
        "relaxed": -deviations["energy"] + deviations["patience"] * 0.55,
        "warm": deviations["warmth"] * 1.20
        + max(0.0, values["valence"] - 0.05) * 0.45,
        "cheerful": max(0.0, values["valence"] - 0.08)
        + max(0.0, deviations["energy"]) * 0.55,
        "curious": deviations["curiosity"] * 1.20
        + max(0.0, deviations["energy"]) * 0.20,
        "energetic": deviations["energy"] * 1.20
        + max(0.0, values["arousal"] - 0.15) * 0.35,
        "confident": deviations["confidence"] * 1.20
        + max(0.0, deviations["energy"]) * 0.25,
        "tired": -deviations["energy"] * 1.25,
        "irritable": -deviations["patience"] * 1.20
        + max(0.0, -values["valence"]) * 0.30,
        "guarded": -deviations["warmth"]
        + max(0.0, -deviations["patience"]) * 0.45,
        "low": max(0.0, -values["valence"] - 0.04)
        + max(0.0, -deviations["energy"]) * 0.50,
    }
    candidate, score = max(scores.items(), key=lambda item: (item[1], item[0]))
    if score < 0.045:
        candidate = "steady"
    if previous in scores and previous != candidate:
        if scores[previous] >= scores[candidate] - 0.025:
            return previous
    return candidate


def _intent(
    evidence: dict[str, float],
    state: EmotionState,
    *,
    identity_question: bool,
) -> tuple[str, str, str]:
    technical = evidence["technical"] >= 0.5
    correction = evidence["correction"] >= 0.5
    repeated = evidence["repetition"] >= 0.5
    choices = (
        (
            technical,
            "technical",
            "corrective" if correction else "focused",
            "repeated_problem" if repeated else "code_problem" if evidence["task_failure"] else "coding_task",
        ),
        (evidence["vulnerability"] >= 0.5, "emotional_support", "gentle", "user_distress"),
        (evidence["hostility"] >= 0.5, "hostility", "guarded", "hostility"),
        (evidence["criticism"] >= 0.5, "criticism", "critical", "criticism"),
        (correction, "correction", "corrective", "correction"),
        (evidence["praise"] >= 0.5, "praise", "kind", "praise"),
        (evidence["playfulness"] >= 0.5, "teasing", "teasing", "teasing"),
        (evidence["gratitude"] >= 0.5, "gratitude", "kind", "thanks"),
        (identity_question, "identity", "neutral", "identity_interest"),
        (evidence["user_frustration"] >= 0.45, "frustration", "upset", "user_frustration"),
        (evidence["serious"] >= 0.5, "serious", "neutral", "serious_topic"),
        (evidence["directness"] >= 0.5, "instruction", "neutral", ""),
    )
    for matches, intent, tone, trigger in choices:
        if matches:
            return intent, tone, trigger
    if state.dominant in {"warm", "concerned"}:
        return "casual", "kind", ""
    return "casual", "neutral", ""


def _identity_focus(text: str) -> tuple[str, bool]:
    if _EXACT_IDENTITY_QUESTION.fullmatch(text.strip()):
        return "identity", False
    for attribute, pattern in _IDENTITY_PATTERNS:
        if pattern.search(text):
            return attribute, not attribute
    return "", False


def _decay(value: float, target: float, hours: float, half_life: float) -> float:
    retention = 0.5 ** (hours / half_life)
    return target + (value - target) * retention


def low_content(text: str) -> bool:
    cleaned = compact_text(text).lower().strip(".,!?;:()[]{}\"'`")
    return cleaned in _LOW_CONTENT


def topic_from_text(text: str) -> tuple[str, float]:
    lower = str(text or "").lower()
    for terms, label in _TOPIC_RULES:
        if any(term in lower for term in terms):
            return label, 0.78
    terms = [
        token
        for token in re.findall(r"[a-z0-9_+#./-]+", lower)
        if len(token) >= 4 and token not in _STOPWORDS
    ]
    unique = list(dict.fromkeys(terms))
    if not unique:
        return compact_text(text, _TOPIC_CHARS), 0.42
    return compact_text(" ".join(unique[:3]), _TOPIC_CHARS), 0.58


def topic_overlap(left: str, right: str) -> float:
    left_words = words(left) - _STOPWORDS
    right_words = words(right) - _STOPWORDS
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / min(len(left_words), len(right_words))


def normalized_signature(text: str) -> str:
    """Return a cheap signature for recent-message repetition checks."""

    return " ".join(re.findall(r"[a-z0-9']+", str(text or "").casefold()))


def message_similarity(left: str, right: str) -> float:
    """Combine typo tolerance with lexical overlap without embeddings."""

    first = normalized_signature(left)
    second = normalized_signature(right)
    if not first or not second:
        return 0.0
    if first == second:
        return 1.0
    first_words = set(first.split())
    second_words = set(second.split())
    overlap = len(first_words & second_words) / max(1, len(first_words | second_words))
    sequence = SequenceMatcher(None, first, second, autojunk=False).ratio()
    return max(overlap, sequence)
