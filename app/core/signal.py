"""Cheap turn appraisal and caller-owned behavioral emotion state."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
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

_POSITIVE_MARKERS = _phrase_group(
    "appreciate it|awesome|glad|good morning|good night|great|happy|nice|sweet|thank you|thanks"
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
_HOSTILITY_MARKERS = _phrase_group(
    "hate you|shut up|you are an idiot|you are a useless idiot|you are a useless|"
    "you are stupid|you are useless|you idiot|you're an idiot|you're a useless idiot|"
    "you're a useless|you're stupid|you're useless"
)
_INDIRECT_REQUEST_MARKERS = _phrase_group(
    "any chance you could|could you maybe|do you think you could|i wonder if you could|"
    "it would be helpful if|it would be nice if|maybe you could|perhaps you could|"
    "would you mind"
)
_EXPLICIT_DISCLOSURE = re.compile(
    r"\b(?:i am|i feel|i have been feeling|i'm|i've been feeling)\b",
    re.IGNORECASE,
)
_CORRECTION_REFERENCE = _phrase_group(
    "but you said|i asked for|i meant|not what i asked|that's not what|that is not what|"
    "you said|your last answer"
)
_CRITICISM_MARKERS = _phrase_group(
    "bad answer|bad take|just answer|not helpful|sounds like a chatbot|stop analyzing|"
    "stop explaining|too generic|too formal|too robotic|you ignored|you missed"
)

_IDENTITY_PATTERNS = (
    (
        "",
        _phrase_group(
            "been doing|been up to today|doing currently|doing right now|what are you doing|"
            "what are you up to|what have you been up to"
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
            "about yourself|what are you|who are you|your identity|your personality|yourself"
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
}
_COMPATIBLE_SECONDARIES = {
    "relaxed": {"amused", "warm"},
    "warm": {"concerned", "confident", "relaxed"},
    "amused": {"relaxed", "warm"},
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
class EmotionState:
    """A compact simulated disposition; ownership remains with the caller."""

    updated_at: float
    valence: float = 0.05
    arousal: float = 0.15
    energy: float = 0.65
    warmth: float = 0.5
    curiosity: float = 0.45
    confidence: float = 0.55
    stimulation: float = 0.5
    amusement: float = 0.0
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
    friendliness: bool = False
    teasing: bool = False
    sadness: bool = False
    technical: bool = False
    debugging: bool = False
    task_success: bool = False
    task_failure: bool = False
    code_context_requested: bool = False
    code_context_attached: bool = False
    identity_attribute: str = ""
    current_activity: bool = False
    repetition: str = ""
    repetition_count: int = 0
    emotion_state: EmotionState = EmotionState(updated_at=0.0)

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
        state = self.emotion_state
        primary = _emotion_label(state.dominant)
        secondary = _emotion_label(state.secondary)
        if state.dominant == "amused" and state.secondary == "relaxed":
            disposition = "quietly amused and relaxed"
        elif secondary:
            disposition = f"{primary} with {secondary}"
        else:
            disposition = primary
        expression = {
            "warm": "be warm and attentive without becoming verbose",
            "amused": "allow light teasing while still answering directly",
            "curious": "show engaged curiosity and notice useful specifics",
            "focused": "be direct, precise, and persistent",
            "confident": "state judgments clearly without overclaiming",
            "concerned": "be patient, steady, and practically supportive",
            "tired": "stay concise while preserving care and accuracy",
            "bored": "look for the concrete detail that makes the exchange useful",
            "frustrated": "stay persistent and direct without snapping",
            "mildly_exasperated": "be firmer and more concise than usual",
            "irritated": "keep firm boundaries and a controlled tone",
            "guarded": "remain calm, bounded, and factual",
        }.get(state.dominant, "respond naturally and keep the answer proportionate")
        mood = {
            "bright": "bright and receptive",
            "pensive": "quietly thoughtful",
            "strained": "somewhat strained but controlled",
            "weary": "low-key and a little weary",
        }.get(state.mood, "steady")
        return (
            f"Current mood: {mood}.\n"
            f"Immediate emotional tendency: {disposition}.\n"
            f"Expression: {expression}.\n"
            "Let this affect expression, not factual judgment. Keep it silent; do not name or explain it."
        )


@dataclass(frozen=True, slots=True)
class SubtextAppraisal:
    """One ephemeral, uncertain interpretation used only for response shaping."""

    kind: str
    summary: str
    confidence: float
    behavioral_effect: str


def appraise_subtext(
    user_text: str,
    signal: TurnSignal,
    recent_turns: Iterable[object] | None,
    *,
    current_topic: str = "",
    current_task: str = "",
    unresolved_problem: bool = False,
    now: float | None = None,
) -> SubtextAppraisal | None:
    """Infer one bounded conversational hypothesis from combined evidence."""

    current = time.time() if now is None else max(0.0, float(now))
    recent = _active_recent_turns(recent_turns, now=current)[-12:]
    recent_users = [content for role, content in recent if role == "user"]
    recent_assistants = [content for role, content in recent if role == "assistant"]
    candidates: list[tuple[float, int, str, str, str]] = []

    def candidate(
        kind: str,
        score: float,
        summary: str,
        behavioral_effect: str,
        priority: int,
    ) -> None:
        if score >= 0.60:
            candidates.append(
                (
                    clamp(score),
                    priority,
                    kind,
                    compact_text(summary, 180),
                    compact_text(behavioral_effect, 180),
                )
            )

    similar_users = [
        content for content in recent_users if message_similarity(content, user_text) >= 0.74
    ]
    repeated_turns = len(similar_users) + 1
    repeated_assistant_pairs = sum(
        message_similarity(left, right) >= 0.82
        for left, right in zip(recent_assistants, recent_assistants[1:])
    )
    if repeated_turns >= 3:
        repetition_score = 0.60 + min(0.22, (repeated_turns - 3) * 0.07)
        repetition_score += min(0.08, repeated_assistant_pairs * 0.04)
        stalled = repeated_assistant_pairs > 0
        candidate(
            "repeated_pattern",
            repetition_score,
            (
                "The user may be checking consistency or signaling that earlier replies "
                "did not resolve the point."
                if stalled
                else "The user may be emphasizing the same point or checking whether the answer changes."
            ),
            (
                "Answer more directly and concisely; lightly recognize the established pattern "
                "only if natural, while remaining patient and non-dismissive."
                if repeated_turns >= 5
                else "Stay patient and answer directly. Change wording, structure, reason, or "
                "emphasis without changing established facts or preferences, and do not force "
                "acknowledgment of repetition."
            ),
            3,
        )

    prior_failures = sum(bool(_FAILURE_MARKERS.search(content)) for content in recent_users[-6:])
    failure_context = bool(current_task or current_topic or signal.technical)
    candidate(
        "unresolved_failure",
        0.38
        + (0.10 if signal.task_failure else 0.0)
        + (0.18 if prior_failures else 0.0)
        + (0.08 if unresolved_problem and prior_failures else 0.0)
        + (0.04 if failure_context and prior_failures else 0.0),
        "The user may be indicating that the earlier approach still has not solved the task.",
        "Acknowledge the prior failure briefly, simplify where useful, and give a concrete next step instead of repeating the same advice.",
        0,
    )

    correction_reference = bool(_CORRECTION_REFERENCE.search(user_text))
    candidate(
        "correction",
        0.43
        + (0.12 if signal.correction_requested else 0.0)
        + (0.09 if recent_assistants else 0.0)
        + (0.08 if correction_reference else 0.0),
        "The user may be redirecting the conversation toward a corrected understanding.",
        "Prioritize the correction, avoid defending the earlier answer, and confirm the revised target through the answer itself.",
        1,
    )

    indirect_hits = len(_INDIRECT_REQUEST_MARKERS.findall(user_text))
    candidate(
        "indirect_request",
        0.42
        + min(0.18, indirect_hits * 0.12)
        + (0.08 if "?" in user_text else 0.0)
        + (0.08 if current_task or current_topic else 0.0)
        + (0.05 if recent_assistants else 0.0),
        "The user may be making a request indirectly rather than merely commenting.",
        "Respond decisively to the likely request while leaving room for the interpretation to be wrong.",
        4,
    )

    disclosed = bool(_EXPLICIT_DISCLOSURE.search(user_text))
    disclosed_vulnerability = bool(_VULNERABILITY_MARKERS.search(user_text))
    candidate(
        "emotional_disclosure",
        0.48
        + (0.14 if disclosed and (signal.sadness or disclosed_vulnerability) else 0.0)
        + (0.07 if recent_assistants else 0.0),
        "The user may want their disclosed experience acknowledged before advice or problem-solving.",
        "Lead with measured warmth and acknowledgment; do not diagnose, overinterpret, or claim certainty about their feelings.",
        2,
    )

    prior_warmth = any(_POSITIVE_MARKERS.search(content) for content in recent_users[-4:])
    candidate(
        "tone_shift",
        0.46
        + (0.10 if signal.criticism or signal.frustration else 0.0)
        + (0.09 if prior_warmth else 0.0)
        + (0.06 if recent_assistants else 0.0),
        "The user's tone may have shifted because the conversation is not meeting their need.",
        "Increase caution and directness, acknowledge the concern without assigning a motive, and avoid repeating the same framing.",
        5,
    )

    if not candidates:
        return None
    score, _priority, kind, summary, effect = sorted(
        candidates,
        key=lambda item: (-item[0], item[1], item[2]),
    )[0]
    return SubtextAppraisal(kind, summary, score, effect)


def _active_recent_turns(
    recent_turns: Iterable[object] | None,
    *,
    now: float,
) -> list[tuple[str, str]]:
    active: list[tuple[str, str]] = []
    for turn in tuple(recent_turns or ())[-12:]:
        role = str(
            turn.get("role", "") if isinstance(turn, dict) else getattr(turn, "role", "")
        )
        content = str(
            turn.get("content", turn.get("text", ""))
            if isinstance(turn, dict)
            else getattr(turn, "content", "")
        ).strip()
        raw_timestamp = (
            turn.get("timestamp", 0.0)
            if isinstance(turn, dict)
            else getattr(turn, "timestamp", 0.0)
        )
        try:
            timestamp = max(0.0, float(raw_timestamp or 0.0))
        except (TypeError, ValueError):
            timestamp = 0.0
        if not content or role not in {"user", "assistant"}:
            continue
        if timestamp and now > timestamp and now - timestamp > 6 * 3600:
            continue
        active.append((role, content))
    return active


def advance_emotion(state: EmotionState, *, now: float) -> EmotionState:
    """Lazily evolve temporary state from elapsed time and weak daily rhythm."""

    current = max(float(now), float(state.updated_at))
    elapsed_hours = max(0.0, current - float(state.updated_at)) / 3600.0
    if elapsed_hours <= 0.0:
        return _make_state(
            state,
            updated_at=current,
            values=_state_values(state),
            cause=state.cause,
        )

    values = _state_values(state)
    for name, half_life in (
        ("valence", 24.0),
        ("arousal", 8.0),
        ("warmth", 72.0),
        ("curiosity", 48.0),
        ("confidence", 48.0),
        ("stimulation", 24.0),
    ):
        values[name] = _decay(
            values[name],
            _BASELINES[name],
            elapsed_hours,
            half_life,
        )
    for name, half_life in (
        ("amusement", 4.0),
        ("concern", 12.0),
        ("frustration", 8.0),
        ("irritation", 6.0),
    ):
        values[name] = _decay(values[name], 0.0, elapsed_hours, half_life)

    if elapsed_hours > 6.0:
        values["stimulation"] -= min(0.08, (elapsed_hours - 6.0) / 72.0 * 0.08)
    hour = time.localtime(current).tm_hour
    energy_target = 0.60 if hour < 6 or hour >= 23 else _BASELINES["energy"]
    values["energy"] = _decay(values["energy"], energy_target, elapsed_hours, 24.0)
    cause = state.cause if max(values[name] for name in _ACTIVATIONS) >= 0.12 else ""
    momentum = _decay(state.momentum, 0.0, elapsed_hours, 10.0)
    trigger_repetitions = state.trigger_repetitions if elapsed_hours < 6.0 else 0
    return _make_state(
        state,
        updated_at=current,
        values=values,
        cause=cause,
        momentum=momentum,
        last_trigger=state.last_trigger if trigger_repetitions else "",
        trigger_repetitions=trigger_repetitions,
    )


def apply_activity_effect(
    state: EmotionState,
    activity_mood: str,
    *,
    now: float,
) -> EmotionState:
    """Apply one small, non-repeating emotional effect from a completed activity."""

    current = state
    values = _state_values(current)
    effects = {
        "calm": {"arousal": -0.025, "stimulation": -0.015, "valence": 0.01},
        "curious": {"curiosity": 0.025, "stimulation": 0.015},
        "focused": {"confidence": 0.018, "stimulation": 0.012},
        "confident": {"confidence": 0.025, "valence": 0.012},
        "thoughtful": {"curiosity": 0.016, "arousal": -0.012},
    }
    for name, delta in effects.get(activity_mood, {}).items():
        low = -1.0 if name == "valence" else 0.0
        values[name] = clamp(values[name] + delta, 0.0, low, 1.0)
    return _make_state(
        current,
        updated_at=now,
        values=values,
        cause=current.cause,
        momentum=current.momentum,
    )


def analyze_turn(
    user_text: str,
    *,
    emotion_state: EmotionState | None = None,
    turn_context: TurnContext | None = None,
    now: float | None = None,
    emotion_state_is_current: bool = False,
    code_context_requested: bool = False,
    code_context_attached: bool = False,
) -> TurnSignal:
    summary = compact_text(user_text, _SUMMARY_CHARS)
    lower = summary.casefold()
    tokens = words(lower)
    evidence = _appraise(
        summary,
        lower,
        tokens,
        turn_context=turn_context,
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
    updated = _apply_evidence(current_emotion, evidence, now=current)
    identity_attribute, current_activity = _identity_focus(lower)
    intent, tone, trigger = _intent(
        evidence,
        updated,
        identity_question=bool(identity_attribute),
    )
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
        friendliness=evidence["social_warmth"] >= 0.25,
        teasing=evidence["playfulness"] >= 0.5,
        sadness=evidence["vulnerability"] >= 0.5,
        technical=technical,
        debugging=technical
        and max(evidence["task_failure"], evidence["correction"]) >= 0.5,
        task_success=evidence["task_success"] >= 0.5,
        task_failure=evidence["task_failure"] >= 0.5,
        code_context_requested=code_context_requested,
        code_context_attached=code_context_attached,
        identity_attribute=identity_attribute,
        current_activity=current_activity,
    )


def _appraise(
    original: str,
    text: str,
    tokens: set[str],
    *,
    turn_context: TurnContext | None,
    code_context_requested: bool,
    code_context_attached: bool,
) -> dict[str, float]:
    gratitude = float(bool(re.search(r"\b(?:thanks|thank you|appreciate it)\b", text)))
    positive = float(bool(_POSITIVE_MARKERS.search(text)))
    negative = float(bool(_NEGATIVE_MARKERS.search(text)))
    correction = float(bool(_CORRECTION_MARKERS.search(text)))
    failure = float(bool(_FAILURE_MARKERS.search(text)))
    success = float(bool(_SUCCESS_MARKERS.search(text)))
    context = turn_context or TurnContext()
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
    if re.search(
        r"\bnot\s+(?:anxious|hurt|lonely|overwhelmed|sad|scared|upset)\b",
        text,
    ):
        vulnerability *= 0.20
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
    praise = max(
        success,
        float(
            positive
            and not gratitude
            and bool(tokens & {"answer", "job", "work", "you"})
        ),
    )
    low = low_content(text)
    social_warmth = max(gratitude * 0.45, praise * 0.85, positive * 0.30)
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


_ACTIVATIONS = ("amusement", "concern", "frustration", "irritation")


def _apply_evidence(
    state: EmotionState,
    evidence: dict[str, float],
    *,
    now: float,
) -> EmotionState:
    values = _state_values(state)
    trigger = _evidence_trigger(evidence)
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

    shift("valence", pleasant * 0.11 - tension * 0.09 - hostility * 0.08, 0.16)
    shift(
        "arousal",
        technical * 0.10
        + direct * 0.08
        + tension * 0.07
        + evidence["playfulness"] * 0.05
        - evidence["low_content"] * 0.01,
    )
    shift("energy", technical * 0.04 + success * 0.05 - failure * 0.025)
    shift(
        "warmth",
        evidence["social_warmth"] * 0.09
        + vulnerability * 0.07
        - hostility * 0.14,
        0.14,
    )
    shift(
        "curiosity",
        technical * 0.12
        + evidence["novelty"] * 0.05
        + direct * 0.04
        - repeated * 0.04,
    )
    shift(
        "confidence",
        success * 0.14
        + evidence["praise"] * 0.06
        - evidence["correction"] * 0.035
        - failure * 0.04
        - hostility * 0.025,
    )
    shift(
        "stimulation",
        technical * 0.10
        + direct * 0.05
        + evidence["playfulness"] * 0.06
        + tension * 0.04,
    )
    shift(
        "amusement",
        evidence["playfulness"] * (1.0 - tension) * 0.22
        - (0.015 if not evidence["playfulness"] else 0.0),
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
        - (0.01 if not failure else 0.0),
    )
    shift(
        "irritation",
        hostility * 0.24
        + evidence["criticism"] * 0.06
        + repeated * 0.03
        - success * 0.10
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
) -> EmotionState:
    bounded = {
        name: clamp(value, _BASELINES.get(name, 0.0), -1.0 if name == "valence" else 0.0, 1.0)
        for name, value in values.items()
    }
    dominant, secondary = _select_emotions(bounded, previous)
    return EmotionState(
        updated_at=max(0.0, float(updated_at)),
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
        "amusement": state.amusement,
        "concern": state.concern,
        "frustration": state.frustration,
        "irritation": state.irritation,
    }


def _cause(evidence: dict[str, float], previous: str) -> str:
    for name, text in (
        ("hostility", "hostility in the current message"),
        ("vulnerability", "the user's distress"),
        ("repetition", "the unresolved failure"),
        ("task_failure", "the reported failure"),
        ("task_success", "the confirmed success"),
        ("correction", "the correction"),
        ("playfulness", "the playful exchange"),
        ("technical", "the current task"),
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
        "correction",
        "criticism",
        "praise",
        "playfulness",
        "technical",
        "social_warmth",
    ):
        if evidence[name] >= 0.5:
            return name
    return ""


def _mood_label(values: dict[str, float], previous: str) -> str:
    targets = {
        "strained": values["frustration"] * 0.65 + values["irritation"] * 0.75,
        "weary": (1.0 - values["energy"]) * 0.85 + values["concern"] * 0.18,
        "bright": max(0.0, values["valence"]) * 0.75 + values["warmth"] * 0.36,
        "pensive": values["curiosity"] * 0.48 + (1.0 - values["arousal"]) * 0.24,
    }
    target, score = max(targets.items(), key=lambda item: item[1])
    thresholds = {"strained": 0.48, "weary": 0.60, "bright": 0.46, "pensive": 0.52}
    if score < thresholds[target]:
        return "steady"
    if previous != "steady" and previous != target:
        previous_score = targets.get(previous, 0.0)
        if score < previous_score + 0.12:
            return previous
    return target


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
    for attribute, pattern in _IDENTITY_PATTERNS:
        if pattern.search(text):
            return attribute, not attribute
    return "", False


def _emotion_label(value: str) -> str:
    return {
        "mildly_exasperated": "mild exasperation",
    }.get(value, value.replace("_", " "))


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
