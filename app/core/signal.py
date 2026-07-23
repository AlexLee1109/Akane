"""Lightweight turn, repetition, embodiment, and emotion appraisal."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from app.core.utils import compact_text

_SUMMARY_CHARS = 180
_TOPIC_CHARS = 80
_DISCORD_PREFIX = re.compile(r"^\s*![a-z0-9_-]+\s*", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9']+")
_QUESTION_FILLER = {
    "a", "an", "are", "but", "do", "does", "did", "how", "i", "is", "it",
    "the", "to", "what", "would", "you", "your",
}
_TOPIC_STOPWORDS = _QUESTION_FILLER | {
    "about", "and", "be", "can", "could", "for", "from", "have", "more",
    "of", "on", "please", "that", "this", "think", "want", "with",
}
_LOW_CONTENT = {
    "hello", "hi", "hey", "yo", "lol", "ok", "okay", "thanks", "thank you",
    "gm", "gn", "good morning", "good night", "/debug_state",
}
_CODE_TERMS = {
    "api", "bug", "class", "code", "codebase", "config", "debug", "dependency",
    "endpoint", "error", "file", "function", "implementation", "import", "javascript",
    "latency", "model", "module", "package", "project", "prompt", "python", "refactor",
    "repository", "runtime", "server", "stream", "test", "tests", "traceback",
    "typescript", "vscode", "workspace",
}
_CODE_ACTIONS = {
    "check", "debug", "explain", "fix", "implement", "inspect", "optimize", "read",
    "refactor", "review", "rewrite", "simplify", "test", "write",
}
_DIRECT_PATTERN = re.compile(
    r"^(?:please\s+)?(?:add|change|check|create|debug|explain|fix|implement|inspect|"
    r"make|optimize|read|refactor|remove|review|rewrite|show|simplify|tell|test|use|write)\b",
    re.IGNORECASE,
)
_APOLOGY = re.compile(r"\b(?:sorry|i apologize|my fault)\b", re.IGNORECASE)
_PRAISE = re.compile(
    r"\b(?:awesome|good job|great job|nice work|perfect|well done)\b", re.IGNORECASE
)
_HOSTILITY = re.compile(
    r"\b(?:hate you|shut up|you(?:'re| are) (?:an? )?(?:idiot|stupid|useless))\b",
    re.IGNORECASE,
)
_CRITICISM = re.compile(
    r"\b(?:bad answer|bad take|just answer|not helpful|stop analyzing|stop explaining|"
    r"too generic|too formal|too robotic|you ignored|you missed|that's wrong|that is wrong)\b",
    re.IGNORECASE,
)
_CORRECTION = re.compile(
    r"\b(?:actually|correction|i meant|instead|not that|wrong file)\b", re.IGNORECASE
)
_FAILURE = re.compile(
    r"\b(?:broken|crashed|doesn't work|does not work|error|failed|failing|still fails|"
    r"stuck|traceback)\b",
    re.IGNORECASE,
)
_SUCCESS = re.compile(
    r"\b(?:fixed it|it works|solved it|that worked|works now)\b", re.IGNORECASE
)
_PLAYFUL = re.compile(r"\b(?:brat|cute|dork|haha|lol|nerd|silly|smug|tease)\b", re.IGNORECASE)
_DISTRESS = re.compile(
    r"\b(?:anxious|crying|hurt|lonely|overwhelmed|sad|scared|miserable|having a bad day)\b",
    re.IGNORECASE,
)
_SHORT_FOLLOW_UP = re.compile(
    r"^(?:are you sure|really|why|how so|what about that|do you mean that)\s*[?.!]*$",
    re.IGNORECASE,
)
_PERSONAL_CONTINUATION = re.compile(
    r"^(?:and\b|but\b|how about\b|what about\b|what if\b)", re.IGNORECASE
)
_OBJECTION = re.compile(
    r"\b(?:stop|don't do that|do not do that|please don't|could you not|quit it|"
    r"cut it out|enough|no more|leave me alone|i said no|i already asked|not okay|"
    r"knock it off)\b",
    re.IGNORECASE,
)
_ACTION = re.compile(
    r"\b(?P<verb>bonk(?:s|ed|ing)?|whack(?:s|ed|ing)?|hit(?:s|ting)?|"
    r"slap(?:s|ped|ping)?|kick(?:s|ed|ing)?|poke(?:s|d|ing)?|"
    r"pat(?:s|ted|ting)?|tap(?:s|ped|ping)?|grab(?:s|bed|bing)?|"
    r"hug(?:s|ged|ging)?|kiss(?:es|ed|ing)?|touch(?:es|ed|ing)?)\b"
    r"(?:\s+(?:at|on))?\s+(?P<target>(?:your|akane(?:'s)?)\s+"
    r"(?:head|arm|hand|shoulder|cheek|face|hair|ear|nose|forehead|waist|leg|back))\b",
    re.IGNORECASE,
)
_FORCEFUL_ACTIONS = {"bonk", "whack", "hit", "slap", "kick", "poke", "grab"}

_ACTIVITY_UPDATE = re.compile(
    r"\b(?P<actor>i|we)(?:(?:'m)|\s+(?:am|are))?\s+"
    r"(?P<action>watching|working\s+on|coding|playing|reading|studying)"
    r"(?:\s+(?P<subject>[^\n.!?;]{1,160}))?",
    re.IGNORECASE,
)
_PAST_ACTIVITY = re.compile(
    r"\b(?P<actor>i|we)\s+(?P<action>watched|played|read|studied|coded|worked\s+on)\s+"
    r"(?P<subject>[^\n.!?;]{2,160})",
    re.IGNORECASE,
)
_COMPLETION = re.compile(
    r"\b(?P<actor>i|we)\s+(?P<verb>completed|finished|fixed|solved|shipped|wrapped\s+up)\s+"
    r"(?P<subject>[^\n.!?;]{2,180})",
    re.IGNORECASE,
)
_FAILURE_STATE = re.compile(
    r"\b(?P<subject>[^\n.!?;]{2,160}?)\s+(?:is|are|was|were)\s+"
    r"(?P<status>still\s+broken|broken|failing|not\s+working|failed)\b",
    re.IGNORECASE,
)
_IDENTITY_PATTERNS = (
    ("appearance", re.compile(r"\b(?:what (?:are you wearing|do you look like)|your (?:appearance|eyes|hair|outfit))\b", re.I)),
    ("relationships", re.compile(r"\b(?:relationship|who created you|who is arcane|your creator)\b", re.I)),
    ("preferences", re.compile(r"\b(?:favorite|what do you (?:like|enjoy)|what are you into|your preferences)\b", re.I)),
    ("identity", re.compile(r"\b(?:who are you|what are you|about yourself|your identity|your personality|yourself)\b", re.I)),
)
_CURRENT_ACTIVITY = re.compile(
    r"\b(?:what are you doing|what are you up to|what have you been doing|how was your day)\b",
    re.IGNORECASE,
)
_CURRENT_THOUGHT = re.compile(
    r"\b(?:current thought|what are you thinking about|what(?:'s| is) on your mind)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ImmediateReaction:
    primary: str = "neutral"
    intensity: float = 0.0
    cause: str = ""


@dataclass(frozen=True, slots=True)
class EmotionState:
    primary: str = "neutral"
    intensity: float = 0.0
    cause: str = ""
    boundary_level: int = 0
    updated_at: float = 0.0


@dataclass(frozen=True, slots=True)
class EmotionalSignal:
    kind: str = "neutral"
    intensity: float = 0.0
    confidence: float = 0.0
    cause: str = ""


@dataclass(frozen=True, slots=True)
class ContextualReaction:
    kind: str = "neutral"
    intensity: float = 0.0
    cause: str = ""


@dataclass(frozen=True, slots=True)
class AffectTrace:
    previous: str
    immediate: str
    candidate: str
    boundary_level: int
    applied: bool
    decay_applied: float = 0.0
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MoodEvolution:
    state: EmotionState
    elapsed_seconds: float = 0.0
    decay_applied: float = 0.0
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnContext:
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
    recent_turns: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    subject: str = ""
    event_type: str = "none"
    status: str = "none"
    actor: str = "unknown"
    target: str = ""
    temporal_state: str = "none"
    negated: bool = False
    confidence: float = 0.0

    @property
    def confirmed_completion(self) -> bool:
        return self.event_type == "completion" and self.status == "completed" and not self.negated


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
    emotion_state: EmotionState = EmotionState()
    immediate_reaction: ImmediateReaction = ImmediateReaction()
    emotional_signal: EmotionalSignal = EmotionalSignal()
    contextual_reaction: ContextualReaction = ContextualReaction()
    semantic_event: SemanticEvent = SemanticEvent()
    repetition_count: int = 1
    embodied_action: str = ""
    embodied_target: str = ""
    continued_after_objection: bool = False
    emotion_applied: bool = False
    personal_continuation: bool = False

    @property
    def low_content(self) -> bool:
        return low_content(self.summary)

    def with_context(self, *, topic: str | None = None, task: str | None = None, confidence: float | None = None) -> "TurnSignal":
        return replace(
            self,
            topic=self.topic if topic is None else topic,
            task=self.task if task is None else task,
            topic_confidence=self.topic_confidence if confidence is None else confidence,
        )

    def emotion_prompt(self) -> str:
        parts: list[str] = []
        if self.embodied_action:
            again = " again" if self.repetition_count > 1 else ""
            parts.append(f"Arcane {_past_action(self.embodied_action)} {self.embodied_target}{again}.")
        if self.repetition_count > 1:
            parts.append(f"This is repetition {self.repetition_count} in the recent conversation.")
        if self.immediate_reaction.primary != "neutral":
            parts.append(f"Immediate reaction: {_reaction_phrase(self.immediate_reaction)}.")
        if self.emotion_state.primary != "neutral" and self.emotion_state.intensity >= 0.05:
            parts.append(
                f"Ongoing emotion: {_intensity_label(self.emotion_state.intensity)} "
                f"{_emotion_phrase(self.emotion_state.primary)} carried from earlier turns."
            )
        if self.emotion_state.boundary_level == 1 and self.embodied_action:
            parts.append("Akane may object plainly.")
        if self.continued_after_objection:
            parts.append("Akane has already objected.")
        if self.emotion_state.boundary_level >= 2:
            parts.append(
                "Set a firmer boundary. Akane may refuse to keep entertaining the behavior, "
                "but must not claim the application paused or shut down."
            )
        if parts:
            parts.append(
                "React directly and naturally. Do not mention software, avatars, simulations, "
                "internal state, or lasting injuries."
            )
        return "\n".join(parts)


def normalized_signature(text: str) -> str:
    value = _DISCORD_PREFIX.sub("", str(text or "").casefold())
    return " ".join(_WORD.findall(value))


def message_similarity(left: str, right: str) -> float:
    first = normalized_signature(left)
    second = normalized_signature(right)
    if not first or not second:
        return 0.0
    if first == second:
        return 1.0
    first_words = set(first.split()) - _QUESTION_FILLER
    second_words = set(second.split()) - _QUESTION_FILLER
    overlap = len(first_words & second_words) / max(1, len(first_words | second_words))
    containment = len(first_words & second_words) / max(1, min(len(first_words), len(second_words)))
    sequence = SequenceMatcher(None, first, second, autojunk=False).ratio()
    return max(overlap, containment * 0.94, sequence)


def topic_overlap(left: str, right: str) -> float:
    left_words = set(normalized_signature(left).split()) - _TOPIC_STOPWORDS
    right_words = set(normalized_signature(right).split()) - _TOPIC_STOPWORDS
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / min(len(left_words), len(right_words))


def topic_from_text(text: str) -> tuple[str, float]:
    terms = [
        token for token in normalized_signature(text).split()
        if len(token) >= 3 and token not in _TOPIC_STOPWORDS
    ]
    unique = list(dict.fromkeys(terms))
    if not unique:
        return compact_text(text, _TOPIC_CHARS), 0.42
    return compact_text(" ".join(unique[:4]), _TOPIC_CHARS), 0.62


def semantic_event_from_text(user_text: str) -> SemanticEvent:
    text = compact_text(user_text, 500)
    if not text:
        return SemanticEvent()
    completion = _COMPLETION.search(text)
    if completion:
        actor = "shared" if completion.group("actor").casefold() == "we" else "Arcane"
        subject = compact_text(completion.group("subject"), 180)
        return SemanticEvent(subject, "completion", "completed", actor, subject, "past", False, 0.96)
    activity = _ACTIVITY_UPDATE.search(text)
    if activity:
        actor = "shared" if activity.group("actor").casefold() == "we" else "Arcane"
        target = compact_text(activity.group("subject"), 160)
        action = compact_text(activity.group("action"), 40).casefold()
        subject = compact_text(f"{action} {target}", 180)
        return SemanticEvent(subject, "activity", "active", actor, target, "current", False, 0.95)
    past = _PAST_ACTIVITY.search(text)
    if past:
        actor = "shared" if past.group("actor").casefold() == "we" else "Arcane"
        target = compact_text(past.group("subject"), 160)
        subject = compact_text(f"{past.group('action')} {target}", 180)
        return SemanticEvent(subject, "activity", "completed", actor, target, "past", False, 0.92)
    failure = _FAILURE_STATE.search(text)
    if failure:
        subject = compact_text(failure.group("subject"), 160)
        actor = "shared" if re.search(r"\bwe\b", text, re.I) else "Arcane" if re.search(r"\b(?:i|my)\b", text, re.I) else "unknown"
        return SemanticEvent(subject, "failure", "failed", actor, subject, "current", False, 0.90)
    return SemanticEvent()


def advance_emotion(state: EmotionState, *, now: float) -> EmotionState:
    return evolve_emotion(state, now=now).state


def evolve_emotion(state: EmotionState, *, now: float, profile_seed: str = "local:owner") -> MoodEvolution:
    del profile_seed
    current = max(float(now), state.updated_at)
    elapsed = max(0.0, current - state.updated_at)
    if state.intensity <= 0.0:
        return MoodEvolution(EmotionState(updated_at=current), elapsed, 0.0, ("neutral",))
    retained = 0.5 ** (elapsed / (4.0 * 3600.0))
    intensity = state.intensity * retained
    decayed = max(0.0, state.intensity - intensity)
    if intensity < 0.04:
        return MoodEvolution(EmotionState(updated_at=current), elapsed, decayed, ("emotion_expired",))
    boundary = _boundary_level(intensity) if state.primary in {"irritated", "angry"} else 0
    return MoodEvolution(
        EmotionState(state.primary, intensity, state.cause, boundary, current),
        elapsed,
        decayed,
        (("elapsed_decay",) if decayed else ("unchanged",)),
    )


def apply_mood_effects(state: EmotionState, effects: tuple[tuple[str, float], ...], *, now: float, cause: str, short_emotion: str = "") -> EmotionState:
    del effects, cause, short_emotion
    return advance_emotion(state, now=now)


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
    current = time.time() if now is None else float(now)
    context = turn_context or TurnContext()
    prior = emotion_state or EmotionState(updated_at=current)
    decayed = prior if emotion_state_is_current else advance_emotion(prior, now=current)
    summary = compact_text(user_text, _SUMMARY_CHARS)
    normalized = normalized_signature(summary)
    token_set = set(normalized.split())
    action, target = _embodied_action(summary)
    repetition_count = _repetition_count(summary, action, target, context)
    continued = bool(action and _prior_objection(context.recent_turns))
    immediate = _immediate_reaction(summary, action, target, repetition_count, continued)
    persistent = _persistent_emotion(
        decayed,
        immediate,
        repetition_count=repetition_count,
        continued_after_objection=continued,
        now=current,
    )
    event = semantic_event or semantic_event_from_text(summary)
    personal_continuation = _personal_continuation(summary, context.recent_turns)
    technical = bool(
        code_context_requested
        or code_context_attached
        or token_set & _CODE_TERMS and token_set & (_CODE_ACTIONS | {"how", "why", "what"})
    ) and not personal_continuation
    hostility = bool(_HOSTILITY.search(summary))
    criticism = bool(_CRITICISM.search(summary))
    correction = bool(_CORRECTION.search(summary))
    failure = bool(_FAILURE.search(summary) or event.status == "failed")
    success = bool(_SUCCESS.search(summary) or event.confirmed_completion and context.completion_meaningful)
    praise = bool(_PRAISE.search(summary))
    teasing = bool(_PLAYFUL.search(summary))
    sadness = bool(_DISTRESS.search(summary))
    identity_attribute = _identity_focus(summary)
    current_activity = bool(_CURRENT_ACTIVITY.search(summary))
    current_thought = bool(_CURRENT_THOUGHT.search(summary))
    topic, topic_confidence = topic_from_text(summary)
    if personal_continuation and context.current_topic:
        topic, topic_confidence = context.current_topic, 0.82
    intent = (
        "hostility" if hostility else "emotional_support" if sadness else "technical" if technical
        else "correction" if correction else "identity" if identity_attribute else "teasing" if teasing
        else "instruction" if _DIRECT_PATTERN.search(summary) else "reflection" if personal_continuation
        else "casual"
    )
    tone = "guarded" if hostility or persistent.boundary_level else "gentle" if sadness else "teasing" if teasing else "neutral"
    reaction_kind = immediate.primary if immediate.primary != "neutral" else persistent.primary
    contextual = ContextualReaction(reaction_kind, max(immediate.intensity, persistent.intensity), immediate.cause or persistent.cause)
    emotional = EmotionalSignal(reaction_kind, immediate.intensity, 1.0 if reaction_kind != "neutral" else 0.0, immediate.cause)
    trigger = "embodied_action" if action else "repetition" if repetition_count > 1 else "hostility" if hostility else ""
    return TurnSignal(
        summary=summary,
        topic=topic,
        topic_confidence=topic_confidence,
        intent=intent,
        tone=tone,
        task=topic if technical else "",
        correction=summary if correction else "",
        trigger=trigger,
        praise=praise,
        criticism=criticism,
        correction_requested=correction,
        hostility=hostility,
        frustration=failure,
        teasing=teasing,
        sadness=sadness,
        technical=technical,
        task_success=success,
        task_failure=failure,
        code_context_requested=code_context_requested,
        code_context_attached=code_context_attached,
        identity_attribute=identity_attribute,
        current_activity=current_activity,
        current_thought=current_thought,
        emotion_state=persistent,
        immediate_reaction=immediate,
        emotional_signal=emotional,
        contextual_reaction=contextual,
        semantic_event=event,
        repetition_count=repetition_count,
        embodied_action=action,
        embodied_target=target,
        continued_after_objection=continued,
        emotion_applied=immediate.primary != "neutral" or persistent.intensity >= 0.05,
        personal_continuation=personal_continuation,
    )


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
    del decayed, event_delta, event_ids
    reasons = list(evolution.reason_codes if evolution else ())
    reasons.extend(extra_reason_codes)
    if signal.embodied_action:
        reasons.append("embodied_action")
    if signal.repetition_count > 1:
        reasons.append("repetition")
    if signal.continued_after_objection:
        reasons.append("continued_after_objection")
    return AffectTrace(
        previous=_emotion_label(prior),
        immediate=_reaction_label(signal.immediate_reaction),
        candidate=_emotion_label(signal.emotion_state),
        boundary_level=signal.emotion_state.boundary_level,
        applied=signal.emotion_applied,
        decay_applied=evolution.decay_applied if evolution else 0.0,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def format_response_disposition(signal: TurnSignal) -> str:
    return signal.emotion_prompt()


def _embodied_action(text: str) -> tuple[str, str]:
    match = _ACTION.search(text)
    if not match:
        return "", ""
    verb = match.group("verb").casefold()
    for root in ("bonk", "whack", "hit", "slap", "kick", "poke", "pat", "tap", "grab", "hug", "kiss", "touch"):
        if verb.startswith(root):
            verb = root
            break
    body_part = match.group("target").split()[-1].casefold()
    return verb, f"Akane's {body_part}"


def _repetition_count(text: str, action: str, target: str, context: TurnContext) -> int:
    prior_users = [content for role, content in context.recent_turns if role == "user"]
    if action:
        action_count = 1
        same_count = 1
        for prior in prior_users:
            prior_action, prior_target = _embodied_action(prior)
            if prior_action:
                action_count += 1
            if prior_action == action and prior_target == target:
                same_count += 1
        return max(action_count, same_count)
    matches = sum(message_similarity(text, prior) >= 0.78 for prior in prior_users)
    if matches:
        return max(matches + 1, context.repeated_topic_count + 1)
    if _SHORT_FOLLOW_UP.fullmatch(text.strip()) and prior_users:
        return max(2, context.repeated_topic_count + 1)
    return 1


def _prior_objection(recent_turns: tuple[tuple[str, str], ...]) -> bool:
    return any(role == "assistant" and _OBJECTION.search(content) for role, content in recent_turns)


def _personal_continuation(text: str, recent_turns: tuple[tuple[str, str], ...]) -> bool:
    if _SHORT_FOLLOW_UP.fullmatch(text.strip()):
        return bool(recent_turns)
    if not _PERSONAL_CONTINUATION.search(text.strip()):
        return False
    previous_user = next((content for role, content in reversed(recent_turns) if role == "user"), "")
    return bool(previous_user and re.search(r"\b(?:you|your|human|ai)\b", previous_user, re.I))


def _immediate_reaction(text: str, action: str, target: str, repetition: int, continued: bool) -> ImmediateReaction:
    if action:
        forceful = action in _FORCEFUL_ACTIONS
        primary = "startled" if forceful else "warm" if action in {"hug", "pat"} else "embarrassed"
        intensity = min(1.0, (0.48 if forceful else 0.30) + 0.08 * (repetition - 1))
        if continued:
            primary = "irritated"
            intensity = max(intensity, 0.68)
        return ImmediateReaction(primary, intensity, f"Arcane {_past_action(action)} {target}")
    if _HOSTILITY.search(text):
        return ImmediateReaction("irritated", 0.72, "Arcane was hostile")
    if _DISTRESS.search(text):
        return ImmediateReaction("concerned", 0.62, "Arcane sounded distressed")
    if _APOLOGY.search(text):
        return ImmediateReaction("softened", 0.40, "Arcane apologized")
    if _PRAISE.search(text):
        return ImmediateReaction("pleased", 0.42, "Arcane praised Akane")
    if repetition > 1:
        return ImmediateReaction("mildly exasperated", min(0.55, 0.18 + repetition * 0.08), "Arcane repeated the question")
    return ImmediateReaction()


def _persistent_emotion(
    prior: EmotionState,
    immediate: ImmediateReaction,
    *,
    repetition_count: int,
    continued_after_objection: bool,
    now: float,
) -> EmotionState:
    current = replace(prior, updated_at=now)
    if immediate.primary in {"startled", "irritated"}:
        base = current.intensity if current.primary in {"irritated", "angry"} else 0.12
        increase = 0.14 + 0.05 * max(0, repetition_count - 1)
        if continued_after_objection:
            increase += 0.08
        intensity = min(1.0, base + increase)
        primary = "angry" if intensity >= 0.72 else "irritated"
        return EmotionState(primary, intensity, immediate.cause, _boundary_level(intensity), now)
    if immediate.primary == "mildly exasperated":
        base = current.intensity if current.primary == "mildly exasperated" else 0.08
        intensity = min(0.62, base + 0.08 + 0.04 * max(0, repetition_count - 1))
        return EmotionState("mildly exasperated", intensity, immediate.cause, 0, now)
    if immediate.primary == "concerned":
        intensity = max(current.intensity * 0.75 if current.primary == "concerned" else 0.0, 0.30)
        return EmotionState("concerned", intensity, immediate.cause, 0, now)
    if immediate.primary in {"warm", "embarrassed", "pleased"}:
        primary = "warm" if immediate.primary in {"warm", "pleased"} else "embarrassed"
        return EmotionState(primary, max(0.16, current.intensity * 0.70), immediate.cause, 0, now)
    if immediate.primary == "softened":
        intensity = max(0.0, current.intensity - 0.24)
        if intensity < 0.05:
            return EmotionState(updated_at=now)
        return EmotionState(current.primary, intensity, current.cause, _boundary_level(intensity), now)
    intensity = max(0.0, current.intensity - 0.04)
    if intensity < 0.05:
        return EmotionState(updated_at=now)
    boundary = _boundary_level(intensity) if current.primary in {"irritated", "angry"} else 0
    return EmotionState(current.primary, intensity, current.cause, boundary, now)


def _boundary_level(intensity: float) -> int:
    return 3 if intensity >= 0.76 else 2 if intensity >= 0.52 else 1 if intensity >= 0.24 else 0


def _identity_focus(text: str) -> str:
    for name, pattern in _IDENTITY_PATTERNS:
        if pattern.search(text):
            return name
    return ""


def _reaction_phrase(reaction: ImmediateReaction) -> str:
    return {
        "startled": "startled recoil",
        "irritated": "an irritated flinch",
        "mildly exasperated": "mild exasperation",
        "concerned": "immediate concern",
        "softened": "a slight softening",
        "pleased": "quiet pleasure",
        "warm": "warmth",
        "embarrassed": "brief embarrassment",
    }.get(reaction.primary, reaction.primary)


def _past_action(action: str) -> str:
    return {
        "hit": "hit",
        "slap": "slapped",
        "poke": "poked",
        "pat": "patted",
        "tap": "tapped",
        "grab": "grabbed",
        "hug": "hugged",
        "kiss": "kissed",
        "touch": "touched",
    }.get(action, f"{action}ed")


def _intensity_label(intensity: float) -> str:
    return "strong" if intensity >= 0.72 else "clear" if intensity >= 0.48 else "mild"


def _emotion_phrase(primary: str) -> str:
    return {
        "irritated": "irritation",
        "angry": "anger",
        "mildly exasperated": "exasperation",
        "concerned": "concern",
        "warm": "warmth",
        "embarrassed": "embarrassment",
    }.get(primary, primary)


def _emotion_label(state: EmotionState) -> str:
    return f"{state.primary} {state.intensity:.2f}"


def _reaction_label(reaction: ImmediateReaction) -> str:
    return f"{reaction.primary} {reaction.intensity:.2f}"


def low_content(text: str) -> bool:
    return normalized_signature(text).strip(".,!?;:()[]{}\"'`") in _LOW_CONTENT
