"""Cheap turn appraisal and caller-owned behavioral emotion state."""

from __future__ import annotations

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
        _phrase_group("what anime|what do you like|what game|your favorite|your preferences"),
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


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Small continuity hints supplied by memory; never owns emotional state."""

    current_topic: str = ""
    current_task: str = ""
    unresolved_problem: bool = False
    repeated_topic_count: int = 0
    last_outcome: str = ""


@dataclass(frozen=True, slots=True)
class TurnSignal:
    summary: str
    topic: str
    topic_confidence: float
    intent: str
    tone: str
    stance: str
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
    direct: bool = False
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
        return (
            f"Current disposition: {disposition}.\n"
            f"Expression: {expression}.\n"
            "Do not mention or explain the internal state."
        )


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
    return _make_state(state, updated_at=current, values=values, cause=cause)


def analyze_turn(
    user_text: str,
    *,
    emotion_state: EmotionState | None = None,
    turn_context: TurnContext | None = None,
    now: float | None = None,
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
    updated = _apply_evidence(advance_emotion(previous, now=current), evidence, now=current)
    identity_attribute, current_activity = _identity_focus(lower)
    intent, stance, tone, trigger = _intent(
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
        stance=stance,
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
        direct=direct,
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
        bool(re.search(r"\b(?:bad answer|bad take|not helpful|you missed|you ignored)\b", text))
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

    def shift(name: str, delta: float, limit: float = 0.18) -> None:
        low = -1.0 if name == "valence" else 0.0
        values[name] = clamp(values[name] + clamp(delta, 0.0, -limit, limit), 0.0, low, 1.0)

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
    return _make_state(
        state,
        updated_at=now,
        values=values,
        cause=cause,
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
        inertia = 0.05 + max(values[name] for name in _ACTIVATIONS) * 0.08
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


def _intent(
    evidence: dict[str, float],
    state: EmotionState,
    *,
    identity_question: bool,
) -> tuple[str, str, str, str]:
    technical = evidence["technical"] >= 0.5
    correction = evidence["correction"] >= 0.5
    repeated = evidence["repetition"] >= 0.5
    choices = (
        (
            technical,
            "technical",
            "careful" if evidence["task_failure"] else "attentive",
            "corrective" if correction else "focused",
            "repeated_problem" if repeated else "code_problem" if evidence["task_failure"] else "coding_task",
        ),
        (evidence["vulnerability"] >= 0.5, "emotional_support", "steady", "gentle", "user_distress"),
        (evidence["hostility"] >= 0.5, "hostility", "bounded", "guarded", "hostility"),
        (evidence["criticism"] >= 0.5, "criticism", "careful", "critical", "criticism"),
        (correction, "correction", "direct", "corrective", "correction"),
        (evidence["praise"] >= 0.5, "praise", "warm", "kind", "praise"),
        (evidence["playfulness"] >= 0.5, "teasing", "light", "teasing", "teasing"),
        (evidence["gratitude"] >= 0.5, "gratitude", "warm", "kind", "thanks"),
        (identity_question, "identity", "direct", "neutral", "identity_interest"),
        (evidence["user_frustration"] >= 0.45, "frustration", "reassuring", "upset", "user_frustration"),
        (evidence["serious"] >= 0.5, "serious", "careful", "neutral", "serious_topic"),
        (evidence["directness"] >= 0.5, "instruction", "direct", "neutral", ""),
    )
    for matches, intent, stance, tone, trigger in choices:
        if matches:
            return intent, stance, tone, trigger
    if state.dominant in {"warm", "concerned"}:
        return "casual", "warm", "kind", ""
    return "casual", "warm", "neutral", ""


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
