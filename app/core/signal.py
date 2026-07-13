"""Lightweight turn signals without switching Akane's personality mode."""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.core.utils import compact_text, contains_any, words

_SUMMARY_CHARS = 180
_TOPIC_CHARS = 80
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "me",
    "of",
    "on",
    "please",
    "the",
    "this",
    "to",
    "we",
    "what",
    "with",
    "would",
    "you",
    "your",
    "akane",
    "about",
    "have",
    "has",
    "that",
    "feel",
    "like",
    "love",
    "really",
    "still",
    "think",
    "want",
    "need",
    "needs",
}
_LOW_CONTENT = {
    "hello",
    "hi",
    "hey",
    "yo",
    "lol",
    "ok",
    "okay",
    "thanks",
    "thank you",
    "gm",
    "gn",
    "good morning",
    "good night",
    "/debug_state",
}
_CODE_TERMS = {
    "api",
    "bug",
    "class",
    "code",
    "codebase",
    "config",
    "debug",
    "dependency",
    "endpoint",
    "error",
    "file",
    "function",
    "implementation",
    "import",
    "javascript",
    "latency",
    "memory",
    "model",
    "module",
    "package",
    "project",
    "prompt",
    "python",
    "refactor",
    "repository",
    "runtime",
    "server",
    "stream",
    "streaming",
    "stack",
    "test",
    "tests",
    "traceback",
    "typescript",
    "vscode",
    "workspace",
}
_CODE_ACTIONS = {
    "check",
    "debug",
    "explain",
    "fix",
    "implement",
    "inspect",
    "optimize",
    "read",
    "refactor",
    "review",
    "rewrite",
    "test",
    "use",
    "write",
}
_PROBLEM_WORDS = {
    "again",
    "broken",
    "crash",
    "doesn't",
    "error",
    "failed",
    "failing",
    "stuck",
    "traceback",
    "wrong",
}
_PRAISE = (
    "adorable",
    "amazing",
    "awesome",
    "good job",
    "great job",
    "nailed it",
    "nice work",
    "perfect",
    "proud",
    "well done",
)
_CRITICISM = (
    "bad answer",
    "bad take",
    "not helpful",
    "that was bad",
    "you missed",
    "you ignored",
)
_CORRECTION = (
    "actually",
    "correction",
    "i meant",
    "not that",
    "that's wrong",
    "that is wrong",
    "wrong file",
    "instead",
)
_HOSTILITY = (
    "shut up",
    "hate you",
    "you are stupid",
    "you're stupid",
    "you idiot",
    "you are useless",
    "you're useless",
)
_FRUSTRATION = (
    "again",
    "annoying",
    "broken",
    "doesn't work",
    "does not work",
    "failed",
    "frustrating",
    "hate this",
    "keeps happening",
    "stuck",
    "ugh",
)
_THANKS = ("appreciate it", "thank you", "thanks")
_FRIENDLY = ("glad", "happy", "nice", "sweet", "good morning", "good night")
_TEASING = ("brat", "cute", "dork", "nerd", "silly", "smug", "tease", "haha", "lol")
_SADNESS = (
    "sad",
    "lonely",
    "hurt",
    "scared",
    "anxious",
    "upset",
    "crying",
    "overwhelmed",
    "tired of this",
)
_SERIOUS = ("serious", "important", "honest", "worried", "risk", "deadline")
_IDENTITY = ("about yourself", "what are you", "who are you", "your identity", "your personality", "yourself")
_DIRECT = {"make", "create", "rewrite", "change", "add", "remove", "show", "tell"}


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
    direct: bool = False
    code_context_requested: bool = False
    code_context_attached: bool = False

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


def analyze_turn(
    user_text: str,
    *,
    code_context_requested: bool = False,
    code_context_attached: bool = False,
) -> TurnSignal:
    summary = compact_text(user_text, _SUMMARY_CHARS)
    lower = summary.lower()
    tokens = words(lower)

    praise = contains_any(lower, _PRAISE)
    criticism = contains_any(lower, _CRITICISM)
    correction = contains_any(lower, _CORRECTION)
    hostility = contains_any(lower, _HOSTILITY)
    frustration = contains_any(lower, _FRUSTRATION)
    gratitude = contains_any(lower, _THANKS)
    friendliness = gratitude or contains_any(lower, _FRIENDLY)
    teasing = contains_any(lower, _TEASING)
    sadness = contains_any(lower, _SADNESS)
    serious = contains_any(lower, _SERIOUS)
    identity = contains_any(lower, _IDENTITY)
    direct = bool(tokens & _DIRECT)
    technical = code_context_requested or (
        bool(tokens & _CODE_TERMS)
        and bool(tokens & (_CODE_ACTIONS | _PROBLEM_WORDS))
    )
    debugging = technical and (frustration or correction or bool(tokens & _PROBLEM_WORDS))

    if debugging:
        intent, stance, tone = "technical", "careful", "corrective" if correction else "frustrated"
        trigger = "repeated_problem" if "again" in tokens or "keeps happening" in lower else "code_problem"
    elif technical:
        intent, stance, tone, trigger = "technical", "attentive", "neutral", "coding_task"
    elif sadness:
        intent, stance, tone, trigger = "emotional_support", "steady", "upset", "user_distress"
    elif hostility:
        intent, stance, tone, trigger = "hostility", "bounded", "hostile", "hostility"
    elif criticism:
        intent, stance, tone, trigger = "criticism", "careful", "critical", "criticism"
    elif correction:
        intent, stance, tone, trigger = "correction", "direct", "corrective", "correction"
    elif praise:
        intent, stance, tone, trigger = "praise", "warm", "kind", "praise"
    elif teasing:
        intent, stance, tone, trigger = "teasing", "light", "teasing", "teasing"
    elif gratitude:
        intent, stance, tone, trigger = "gratitude", "warm", "kind", "thanks"
    elif identity:
        intent, stance, tone, trigger = "identity", "direct", "neutral", "identity_interest"
    elif frustration:
        intent, stance, tone, trigger = "frustration", "reassuring", "upset", "user_frustration"
    elif serious:
        intent, stance, tone, trigger = "serious", "careful", "neutral", "serious_topic"
    elif direct:
        intent, stance, tone, trigger = "instruction", "direct", "neutral", ""
    else:
        intent, stance, tone, trigger = "casual", "warm", "neutral", ""

    topic, confidence = topic_from_text(summary)
    task = topic if technical or direct else ""
    return TurnSignal(
        summary=summary,
        topic=topic,
        topic_confidence=confidence,
        intent=intent,
        tone=tone,
        stance=stance,
        task=task,
        correction=summary if correction else "",
        trigger=trigger,
        praise=praise,
        criticism=criticism,
        correction_requested=correction,
        hostility=hostility,
        frustration=frustration,
        friendliness=friendliness,
        teasing=teasing,
        sadness=sadness,
        technical=technical,
        debugging=debugging,
        direct=direct,
        code_context_requested=code_context_requested,
        code_context_attached=code_context_attached,
    )


def low_content(text: str) -> bool:
    cleaned = compact_text(text).lower().strip(".,!?;:()[]{}\"'`")
    return cleaned in _LOW_CONTENT


def topic_from_text(text: str) -> tuple[str, float]:
    lower = str(text or "").lower()
    known = (
        (("personality", "tone", "soul", "identity", "prompt rules"), "Akane personality"),
        (("tts", "voice", "audio", "speech"), "TTS voice"),
        (("discord", "idle", "channel"), "Discord behavior"),
        (("vscode", "vs code", "workspace"), "VS Code workspace"),
        (("popup", "streaming"), "popup chat"),
    )
    for terms, label in known:
        if any(term in lower for term in terms):
            return label, 0.78
    terms = [token for token in words(lower) if len(token) >= 4 and token not in _STOPWORDS]
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
